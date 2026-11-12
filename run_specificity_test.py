import os
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import wfdb
from src.hybrid_turbulence_detector import HybridTurbulenceGapDetector

def compute_mad_stats(arr):
    clean_arr = arr[~np.isnan(arr)]
    if len(clean_arr) == 0:
        return 0.0, 1.0
    med = float(np.median(clean_arr))
    mad = float(np.median(np.abs(clean_arr - med)))
    sigma_mad = 1.4826 * mad
    if sigma_mad < 1e-6:
        sigma_mad = float(np.std(clean_arr)) + 1e-6
    return med, sigma_mad

def compute_duty_cycle(arr_bool, window_samples):
    n = len(arr_bool)
    duty = np.zeros(n)
    cumsum = np.cumsum(np.insert(arr_bool.astype(float), 0, 0))
    for i in range(n):
        start = max(0, i - window_samples + 1)
        count = cumsum[i + 1] - cumsum[start]
        length = i - start + 1
        duty[i] = count / length
    return duty

def run_quiescent_stream(raw_ecg, target_fs=250.0):
    dt = 1.0 / target_fs
    d_dim = 5
    tau_step = 6
    window_m = 150
    
    detector = HybridTurbulenceGapDetector(
        embedding_dim=d_dim,
        delay_tau=tau_step,
        window_size=window_m,
        dt=dt,
        reg_alpha=1e-3
    )
    
    omega1, theta = detector.analyze_stream(raw_ecg)
    
    shift_idx = (d_dim - 1) * tau_step
    valid_mask = ~np.isnan(omega1)
    
    omega1_valid = omega1[valid_mask]
    theta_valid = theta[valid_mask]
    
    n_valid = len(omega1_valid)
    if n_valid < 1000:
        return 0, 0.0, 0.0
        
    init_baseline_len = min(int(60.0 * target_fs), int(n_valid * 0.2))
    baseline_omega = omega1_valid[:init_baseline_len]
    baseline_theta = theta_valid[:init_baseline_len]
    
    med_omega, mad_sigma_omega = compute_mad_stats(baseline_omega)
    med_theta, mad_sigma_theta = compute_mad_stats(baseline_theta)
    
    z_omega_robust = (omega1_valid - med_omega) / (mad_sigma_omega + 1e-8)
    z_theta_robust = (theta_valid - med_theta) / (mad_sigma_theta + 1e-8)
    
    prob_softening = 1.0 / (1.0 + np.exp(z_omega_robust))
    prob_expansion = 1.0 / (1.0 + np.exp(- z_theta_robust))
    ewi_raw = prob_softening * prob_expansion
    
    slow_span = int(1.5 * target_fs)
    kernel_slow = np.ones(slow_span) / slow_span
    ewi_slow = np.convolve(ewi_raw, kernel_slow, mode="same")
    
    duty_window = int(1.0 * target_fs)
    is_gap_negative = (omega1_valid < 0.0)
    duty_cycle = compute_duty_cycle(is_gap_negative, duty_window)
    
    baseline_ewi_slow = ewi_slow[:init_baseline_len]
    med_ewi_slow, mad_sigma_ewi_slow = compute_mad_stats(baseline_ewi_slow)
    threshold_3sigma = med_ewi_slow + 3.0 * mad_sigma_ewi_slow
    threshold_2sigma = med_ewi_slow + 2.0 * mad_sigma_ewi_slow
    
    eval_start = init_baseline_len
    eval_omega = omega1_valid[eval_start:]
    eval_theta = theta_valid[eval_start:]
    eval_ewi_slow = ewi_slow[eval_start:]
    eval_duty = duty_cycle[eval_start:]
    
    cond_a = (eval_ewi_slow > threshold_3sigma)
    cond_b = (eval_ewi_slow > threshold_2sigma) & (eval_duty >= 0.15)
    cond_c = (eval_omega < -3.0) & (eval_duty >= 0.20)
    
    alarm_active = cond_a | cond_b | cond_c
    
    diff_alarm = np.diff(np.insert(alarm_active.astype(int), 0, 0))
    n_false_alarms = int(np.sum(diff_alarm == 1))
    
    total_eval_samples = len(eval_omega)
    total_eval_seconds = total_eval_samples / target_fs
    total_eval_hours = total_eval_seconds / 3600.0
    
    alarm_samples = int(np.sum(alarm_active))
    alarm_seconds = alarm_samples / target_fs
    specificity_ratio = (1.0 - (alarm_seconds / total_eval_seconds)) * 100.0 if total_eval_seconds > 0 else 100.0
    fp_per_hour = (n_false_alarms / total_eval_hours) if total_eval_hours > 0 else 0.0
    
    return n_false_alarms, total_eval_hours, fp_per_hour, specificity_ratio, alarm_seconds

def test_nsrdb():
    print("Testing NSRDB (Normal Sinus Rhythm Database)...")
    records = ["16265", "16272"]
    results = []
    
    for r in records:
        try:
            dur_samples = 128 * 1800
            rec, info = wfdb.rdsamp(r, pn_dir="nsrdb", sampfrom=0, sampto=dur_samples)
            sig_128 = rec[:, 0]
            sig_250 = resample_poly(sig_128, 125, 64)
            
            n_fa, hours, fp_h, spec, sec = run_quiescent_stream(sig_250, target_fs=250.0)
            results.append({
                "database": "NSRDB",
                "record": r,
                "duration_hours": hours,
                "false_alarm_count": n_fa,
                "alarm_duration_sec": sec,
                "fp_per_hour": fp_h,
                "specificity_percent": spec
            })
            print(f"NSRDB Record {r}: {hours:.2f} h | False Alarms: {n_fa} | FP/h: {fp_h:.2f} | Specificity: {spec:.2f}%")
        except Exception as e:
            print(f"Error on NSRDB {r}: {e}")
            
    return results

def test_vfdb_quiescent():
    print("Testing VFDB Quiescent Baseline Intervals (Pre-Onset)...")
    cases = [
        ("420", 0, 250 * 900),
        ("422", 0, 250 * 900),
        ("424", 0, 250 * 900)
    ]
    results = []
    
    for r, s_start, s_end in cases:
        try:
            rec, info = wfdb.rdsamp(r, pn_dir="vfdb", sampfrom=s_start, sampto=s_end)
            sig_250 = rec[:, 0]
            
            n_fa, hours, fp_h, spec, sec = run_quiescent_stream(sig_250, target_fs=250.0)
            results.append({
                "database": "VFDB_Quiescent",
                "record": r,
                "duration_hours": hours,
                "false_alarm_count": n_fa,
                "alarm_duration_sec": sec,
                "fp_per_hour": fp_h,
                "specificity_percent": spec
            })
            print(f"VFDB {r} (Quiescent): {hours:.2f} h | False Alarms: {n_fa} | FP/h: {fp_h:.2f} | Specificity: {spec:.2f}%")
        except Exception as e:
            print(f"Error on VFDB {r}: {e}")
            
    return results

def main():
    os.makedirs("results", exist_ok=True)
    
    res_nsr = test_nsrdb()
    res_vf_q = test_vfdb_quiescent()
    
    all_res = res_nsr + res_vf_q
    df_all = pd.DataFrame(all_res)
    
    csv_path = os.path.join("results", "specificity_benchmark_results.csv")
    df_all.to_csv(csv_path, index=False)
    
    total_hours = np.sum(df_all["duration_hours"])
    total_false_alarms = np.sum(df_all["false_alarm_count"])
    grand_fp_h = total_false_alarms / total_hours if total_hours > 0 else 0.0
    mean_spec = np.mean(df_all["specificity_percent"])
    
    print("-" * 50)
    print("SPECIFICITY & FALSE POSITIVE BENCHMARK SUMMARY")
    print(f"Total Quiescent Monitoring Time: {total_hours:.2f} hours")
    print(f"Total False Alarms: {total_false_alarms}")
    print(f"Grand False Positives per Hour (FP/h): {grand_fp_h:.4f}")
    print(f"Average Specificity: {mean_spec:.2f}%")
    print("-" * 50)

if __name__ == "__main__":
    main()
