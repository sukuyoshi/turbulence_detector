import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def analyze_hybrid_record(rec_name, vf_label="VF"):
    pn_dir = "vfdb"
    try:
        ann = wfdb.rdann(rec_name, "atr", pn_dir=pn_dir)
    except Exception:
        return None
    
    vf_sample = None
    for s, note in zip(ann.sample, ann.aux_note):
        clean_note = note.replace("\x00", "").strip()
        if vf_label in clean_note:
            vf_sample = int(s)
            break
            
    if vf_sample is None:
        return None
        
    fs = 250.0
    dt = 1.0 / fs
    
    pre_samples = int(120.0 * fs)
    post_samples = int(30.0 * fs)
    
    start_sample = max(0, vf_sample - pre_samples)
    end_sample = vf_sample + post_samples
    
    try:
        record, info = wfdb.rdsamp(rec_name, pn_dir=pn_dir, sampfrom=start_sample, sampto=end_sample)
    except Exception:
        return None
        
    raw_ecg = record[:, 0]
    time_axis = (np.arange(start_sample, end_sample) - vf_sample) / fs
    
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
    time_embedded = time_axis[shift_idx:]
    raw_ecg_embedded = raw_ecg[shift_idx:]
    
    valid_mask = ~np.isnan(omega1)
    t_valid = time_embedded[valid_mask]
    ecg_valid = raw_ecg_embedded[valid_mask]
    omega1_valid = omega1[valid_mask]
    theta_valid = theta[valid_mask]
    
    baseline_mask = (t_valid >= -110.0) & (t_valid <= -50.0)
    
    med_omega, mad_sigma_omega = compute_mad_stats(omega1_valid[baseline_mask])
    med_theta, mad_sigma_theta = compute_mad_stats(theta_valid[baseline_mask])
    
    z_omega_robust = (omega1_valid - med_omega) / (mad_sigma_omega + 1e-8)
    z_theta_robust = (theta_valid - med_theta) / (mad_sigma_theta + 1e-8)
    
    prob_softening = 1.0 / (1.0 + np.exp(z_omega_robust))
    prob_expansion = 1.0 / (1.0 + np.exp(- z_theta_robust))
    ewi_raw = prob_softening * prob_expansion
    
    slow_span = int(1.5 * fs)
    fast_span = int(0.25 * fs)
    
    kernel_slow = np.ones(slow_span) / slow_span
    kernel_fast = np.ones(fast_span) / fast_span
    
    ewi_slow = np.convolve(ewi_raw, kernel_slow, mode="same")
    ewi_fast = np.convolve(ewi_raw, kernel_fast, mode="same")
    
    duty_window = int(1.0 * fs)
    is_gap_negative = (omega1_valid < 0.0)
    duty_cycle = compute_duty_cycle(is_gap_negative, duty_window)
    
    med_ewi_slow, mad_sigma_ewi_slow = compute_mad_stats(ewi_slow[baseline_mask])
    threshold_3sigma = med_ewi_slow + 3.0 * mad_sigma_ewi_slow
    threshold_2sigma = med_ewi_slow + 2.0 * mad_sigma_ewi_slow
    
    cond_a = (ewi_slow > threshold_3sigma)
    cond_b = (ewi_slow > threshold_2sigma) & (duty_cycle >= 0.15)
    cond_c = (omega1_valid < -3.0) & (duty_cycle >= 0.20)
    
    hybrid_alarm = cond_a | cond_b | cond_c
    
    pre_window = (t_valid < 0.0) & (t_valid > -40.0)
    alarm_pre_indices = np.where(pre_window & hybrid_alarm)[0]
    
    if len(alarm_pre_indices) > 0:
        lead_time = - float(t_valid[alarm_pre_indices[0]])
        trigger_type = "Cond_A" if cond_a[alarm_pre_indices[0]] else ("Cond_B" if cond_b[alarm_pre_indices[0]] else "Cond_C")
    else:
        lead_time = 0.0
        trigger_type = "None"
        
    return {
        "record": rec_name,
        "lead_time": lead_time,
        "trigger_type": trigger_type,
        "baseline_med_omega": med_omega,
        "baseline_mad_sigma_omega": mad_sigma_omega,
        "threshold_3sigma": threshold_3sigma,
        "threshold_2sigma": threshold_2sigma,
        "max_pre_ewi_slow": float(np.max(ewi_slow[pre_window])),
        "max_pre_duty": float(np.max(duty_cycle[pre_window])),
        "min_pre_omega": float(np.min(omega1_valid[pre_window])),
        "t_valid": t_valid,
        "ecg_valid": ecg_valid,
        "omega1_valid": omega1_valid,
        "theta_valid": theta_valid,
        "ewi_slow": ewi_slow,
        "ewi_fast": ewi_fast,
        "duty_cycle": duty_cycle,
        "hybrid_alarm": hybrid_alarm
    }

def main():
    target_records = [
        ("418", "VFL"),
        ("422", "VFIB"),
        ("424", "VFIB"),
        ("426", "VF")
    ]
    
    results = []
    os.makedirs("results", exist_ok=True)
    
    for r, label in target_records:
        res = analyze_hybrid_record(r, label)
        if res is not None:
            results.append({
                "record": res["record"],
                "lead_time_sec": res["lead_time"],
                "trigger_type": res["trigger_type"],
                "min_pre_omega1": res["min_pre_omega"],
                "max_pre_duty_cycle": res["max_pre_duty"],
                "max_pre_ewi_slow": res["max_pre_ewi_slow"],
                "threshold_3sigma": res["threshold_3sigma"]
            })
            
            plt.figure(figsize=(12, 11), dpi=300)
            
            plt.subplot(5, 1, 1)
            plt.plot(res["t_valid"], res["ecg_valid"], color="black", lw=0.8)
            plt.axvline(0, color="red", linestyle="--", lw=1.5, label="Clinical Onset (t=0)")
            plt.ylabel("ECG (mV)")
            plt.title(f"Hybrid Early Warning Architecture: Record {res['record']} (Lead Time = {res['lead_time']:.2f} s)")
            plt.legend(loc="upper left")
            plt.grid(True, alpha=0.3)
            
            plt.subplot(5, 1, 2)
            plt.plot(res["t_valid"], res["omega1_valid"], color="blue", lw=0.9, alpha=0.7, label="Turbulence Gap Omega_1")
            plt.axhline(0, color="gray", linestyle=":", lw=1.0)
            plt.axhline(-3.0, color="navy", linestyle="--", alpha=0.6, label="Collapse Depth (-3.0)")
            plt.axvline(0, color="red", linestyle="--", lw=1.5)
            plt.ylabel("Gap Omega_1")
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            
            plt.subplot(5, 1, 3)
            plt.plot(res["t_valid"], res["duty_cycle"], color="purple", lw=1.0, label="Duty Cycle (Omega_1 < 0)")
            plt.axhline(0.20, color="purple", linestyle="--", alpha=0.6, label="Duty Threshold (20%)")
            plt.axvline(0, color="red", linestyle="--", lw=1.5)
            plt.ylabel("Duty Cycle")
            plt.legend(loc="upper left")
            plt.grid(True, alpha=0.3)
            
            plt.subplot(5, 1, 4)
            plt.plot(res["t_valid"], res["ewi_slow"], color="crimson", lw=1.3, label="EWI Slow (1.5s)")
            plt.plot(res["t_valid"], res["ewi_fast"], color="salmon", lw=0.8, alpha=0.6, label="EWI Fast (0.25s)")
            plt.axhline(res["threshold_3sigma"], color="orange", linestyle="--", lw=1.1, label="3-Sigma MAD")
            plt.axhline(res["threshold_2sigma"], color="gold", linestyle=":", lw=1.1, label="2-Sigma MAD")
            plt.axvline(0, color="red", linestyle="--", lw=1.5)
            plt.ylabel("EWI")
            plt.legend(loc="upper left")
            plt.grid(True, alpha=0.3)
            
            plt.subplot(5, 1, 5)
            alarm_plot = np.where(res["hybrid_alarm"], 1.0, 0.0)
            plt.fill_between(res["t_valid"], 0, alarm_plot, color="red", alpha=0.4, label="Hybrid Alarm State")
            plt.axvline(0, color="red", linestyle="--", lw=1.5)
            if res["lead_time"] > 0:
                plt.axvline(-res["lead_time"], color="darkred", linestyle="-", lw=2.0, label=f"First Alarm ({res['lead_time']:.2f} s pre-VF)")
            plt.ylabel("Alarm")
            plt.xlabel("Time Relative to Clinical Onset (seconds)")
            plt.ylim(-0.1, 1.2)
            plt.legend(loc="upper left")
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            out_img = os.path.join("results", f"vfdb_{res['record']}_hybrid_plot.png")
            plt.savefig(out_img)
            plt.close()
            
    df_res = pd.DataFrame(results)
    csv_out = os.path.join("results", "hybrid_benchmark_summary.csv")
    df_res.to_csv(csv_out, index=False)
    print(df_res.to_string())

if __name__ == "__main__":
    main()
