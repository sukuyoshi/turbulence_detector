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

def consolidate_events(alarm_bool, target_fs, debounce_sec=5.0, min_persist_sec=0.5):
    min_persist_samples = int(min_persist_sec * target_fs)
    debounce_samples = int(debounce_sec * target_fs)
    
    n = len(alarm_bool)
    filtered = np.zeros(n, dtype=bool)
    
    in_event = False
    event_start = 0
    for i in range(n):
        if alarm_bool[i]:
            if not in_event:
                in_event = True
                event_start = i
        else:
            if in_event:
                in_event = False
                if (i - event_start) >= min_persist_samples:
                    filtered[event_start:i] = True
    if in_event:
        if (n - event_start) >= min_persist_samples:
            filtered[event_start:n] = True
            
    events = []
    last_event_end = - debounce_samples - 1
    in_block = False
    block_start = 0
    for i in range(n):
        if filtered[i]:
            if not in_block:
                in_block = True
                block_start = i
        else:
            if in_block:
                in_block = False
                if block_start - last_event_end > debounce_samples:
                    events.append((block_start, i))
                last_event_end = i
    if in_block:
        if block_start - last_event_end > debounce_samples:
            events.append((block_start, n))
            
    return len(events), filtered

def analyze_dataset_episodes(rec_name, pn_dir, fs_orig, target_fs=250.0):
    dur_samples = int(fs_orig * 1800) if pn_dir == "nsrdb" else int(fs_orig * 900)
    rec, info = wfdb.rdsamp(rec_name, pn_dir=pn_dir, sampfrom=0, sampto=dur_samples)
    sig = rec[:, 0]
    if pn_dir == "nsrdb":
        sig = resample_poly(sig, 125, 64)
        
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
    
    omega1, theta = detector.analyze_stream(sig)
    valid_mask = ~np.isnan(omega1)
    omega1_valid = omega1[valid_mask]
    theta_valid = theta[valid_mask]
    
    n_valid = len(omega1_valid)
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
    raw_pulse_count = int(np.sum(diff_alarm == 1))
    
    n_episodes_5s, filtered_5s = consolidate_events(alarm_active, target_fs, debounce_sec=5.0, min_persist_sec=0.5)
    n_episodes_10s, filtered_10s = consolidate_events(alarm_active, target_fs, debounce_sec=10.0, min_persist_sec=0.5)
    n_episodes_1s_persist, _ = consolidate_events(alarm_active, target_fs, debounce_sec=5.0, min_persist_sec=1.0)
    
    total_eval_samples = len(eval_omega)
    total_eval_hours = (total_eval_samples / target_fs) / 3600.0
    
    return {
        "record": rec_name,
        "database": pn_dir,
        "hours": total_eval_hours,
        "raw_pulse_count": raw_pulse_count,
        "episodes_persist05_deb5": n_episodes_5s,
        "fp_h_episodes_5s": n_episodes_5s / total_eval_hours,
        "episodes_persist10_deb5": n_episodes_1s_persist,
        "fp_h_episodes_1s_persist": n_episodes_1s_persist / total_eval_hours
    }

recs = [
    ("16265", "nsrdb", 128.0),
    ("16272", "nsrdb", 128.0),
    ("420", "vfdb", 250.0),
    ("422", "vfdb", 250.0),
    ("424", "vfdb", 250.0)
]

all_data = []
for r, d, fs in recs:
    res = analyze_dataset_episodes(r, d, fs)
    all_data.append(res)
    print(f"{r} ({d}): Raw Pulses = {res['raw_pulse_count']} | Episodes (min 0.5s, deb 5s) = {res['episodes_persist05_deb5']} (FP/h: {res['fp_h_episodes_5s']:.2f}) | Episodes (min 1.0s) = {res['episodes_persist10_deb5']} (FP/h: {res['fp_h_episodes_1s_persist']:.2f})")

df = pd.DataFrame(all_data)
df.to_csv("results/specificity_episode_consolidation.csv", index=False)
