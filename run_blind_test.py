import os
import numpy as np
import pandas as pd
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

def evaluate_record(rec_name):
    pn_dir = "vfdb"
    try:
        ann = wfdb.rdann(rec_name, "atr", pn_dir=pn_dir)
    except Exception:
        return None
        
    target_sample = None
    target_label = None
    
    for s, note in zip(ann.sample, ann.aux_note):
        clean_note = note.replace("\x00", "").strip()
        if any(k in clean_note for k in ["VF", "VFL", "VT"]):
            if int(s) > 30000:
                target_sample = int(s)
                target_label = clean_note
                break
                
    if target_sample is None:
        for s, note in zip(ann.sample, ann.aux_note):
            clean_note = note.replace("\x00", "").strip()
            if any(k in clean_note for k in ["VF", "VFL", "VT"]):
                target_sample = int(s)
                target_label = clean_note
                break

    if target_sample is None:
        return None

    fs = 250.0
    dt = 1.0 / fs
    
    pre_samples = int(120.0 * fs)
    post_samples = int(30.0 * fs)
    
    start_sample = max(0, target_sample - pre_samples)
    end_sample = target_sample + post_samples
    
    try:
        record, info = wfdb.rdsamp(rec_name, pn_dir=pn_dir, sampfrom=start_sample, sampto=end_sample)
    except Exception:
        return None
        
    raw_ecg = record[:, 0]
    time_axis = (np.arange(start_sample, end_sample) - target_sample) / fs
    
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
    
    valid_mask = ~np.isnan(omega1)
    t_valid = time_embedded[valid_mask]
    omega1_valid = omega1[valid_mask]
    theta_valid = theta[valid_mask]
    
    baseline_mask = (t_valid >= -110.0) & (t_valid <= -50.0)
    if np.sum(baseline_mask) < 100:
        return None
        
    med_omega, mad_sigma_omega = compute_mad_stats(omega1_valid[baseline_mask])
    med_theta, mad_sigma_theta = compute_mad_stats(theta_valid[baseline_mask])
    
    z_omega_robust = (omega1_valid - med_omega) / (mad_sigma_omega + 1e-8)
    z_theta_robust = (theta_valid - med_theta) / (mad_sigma_theta + 1e-8)
    
    prob_softening = 1.0 / (1.0 + np.exp(z_omega_robust))
    prob_expansion = 1.0 / (1.0 + np.exp(- z_theta_robust))
    ewi_raw = prob_softening * prob_expansion
    
    slow_span = int(1.5 * fs)
    kernel_slow = np.ones(slow_span) / slow_span
    ewi_slow = np.convolve(ewi_raw, kernel_slow, mode="same")
    
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
        first_idx = alarm_pre_indices[0]
        if cond_a[first_idx]:
            trigger_type = "Cond_A"
        elif cond_b[first_idx]:
            trigger_type = "Cond_B"
        else:
            trigger_type = "Cond_C"
        status = "Detected"
    else:
        lead_time = 0.0
        trigger_type = "None"
        status = "Undetected"
        
    return {
        "record": rec_name,
        "pathology": target_label,
        "status": status,
        "lead_time_sec": lead_time,
        "trigger_type": trigger_type,
        "min_pre_omega": float(np.min(omega1_valid[pre_window])) if np.sum(pre_window) > 0 else 0.0,
        "max_pre_duty": float(np.max(duty_cycle[pre_window])) if np.sum(pre_window) > 0 else 0.0,
        "max_pre_ewi": float(np.max(ewi_slow[pre_window])) if np.sum(pre_window) > 0 else 0.0
    }

def main():
    unseen_records = [
        "419", "420", "421", "423", "425", "427", "428", "429", "430",
        "602", "605", "607", "609", "610", "611", "612", "614", "615"
    ]
    
    results = []
    os.makedirs("results", exist_ok=True)
    
    for r in unseen_records:
        res = evaluate_record(r)
        if res is not None:
            results.append(res)
            print(f"Record {res['record']} [{res['pathology']}]: {res['status']} | Lead Time: {res['lead_time_sec']:.2f} s | Trigger: {res['trigger_type']}")

    df_out = pd.DataFrame(results)
    csv_path = os.path.join("results", "blind_test_unseen_records.csv")
    df_out.to_csv(csv_path, index=False)
    
    detected_count = np.sum(df_out["status"] == "Detected")
    total_count = len(df_out)
    sensitivity = (detected_count / total_count) * 100.0
    
    detected_leads = df_out[df_out["status"] == "Detected"]["lead_time_sec"]
    mean_lead = np.mean(detected_leads) if len(detected_leads) > 0 else 0.0
    med_lead = np.median(detected_leads) if len(detected_leads) > 0 else 0.0
    std_lead = np.std(detected_leads) if len(detected_leads) > 0 else 0.0
    
    print("-" * 50)
    print("BLIND TEST SUMMARY ON UNSEEN RECORDS")
    print(f"Total Evaluated Records: {total_count}")
    print(f"Detected: {detected_count} / {total_count}")
    print(f"Sensitivity: {sensitivity:.2f}%")
    print(f"Mean Lead Time: {mean_lead:.2f} s")
    print(f"Median Lead Time: {med_lead:.2f} s")
    print(f"Std Lead Time: {std_lead:.2f} s")
    print("-" * 50)

if __name__ == "__main__":
    main()
