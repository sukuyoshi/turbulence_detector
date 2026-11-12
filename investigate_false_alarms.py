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

def diagnose_record(rec_name, pn_dir, fs_orig, target_fs=250.0):
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
    
    n_a = int(np.sum(cond_a))
    n_b = int(np.sum(cond_b))
    n_c = int(np.sum(cond_c))
    
    total = len(eval_omega)
    
    print(f"Record {rec_name} ({pn_dir}):")
    print(f"  Baseline omega med: {med_omega:.4f}, mad_sigma: {mad_sigma_omega:.4f}")
    print(f"  Baseline ewi_slow med: {med_ewi_slow:.4f}, mad_sigma: {mad_sigma_ewi_slow:.4f}")
    print(f"  Threshold 3sigma: {threshold_3sigma:.4f}, 2sigma: {threshold_2sigma:.4f}")
    print(f"  Active Samples Cond A: {n_a} ({n_a/total*100:.2f}%)")
    print(f"  Active Samples Cond B: {n_b} ({n_b/total*100:.2f}%)")
    print(f"  Active Samples Cond C: {n_c} ({n_c/total*100:.2f}%)")
    print(f"  Min Omega in Eval: {np.min(eval_omega):.4f}")
    print(f"  Max Duty in Eval: {np.max(eval_duty):.4f}")

diagnose_record("16265", "nsrdb", 128.0)
diagnose_record("16272", "nsrdb", 128.0)
diagnose_record("420", "vfdb", 250.0)
diagnose_record("422", "vfdb", 250.0)
diagnose_record("424", "vfdb", 250.0)
