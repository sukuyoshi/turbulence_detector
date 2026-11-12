import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
from src.turbulence_detector import TurbulenceGapDetector

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

def analyze_robust_record(rec_name, vf_label="VF"):
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
    
    detector = TurbulenceGapDetector(
        embedding_dim=d_dim,
        delay_tau=tau_step,
        window_size=window_m,
        dt=dt,
        reg_alpha=1e-3,
        sigma_omega=1.0,
        theta_threshold=0.0
    )
    
    omega1, theta, _ = detector.analyze_stream(raw_ecg)
    
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
    
    ewi_robust = prob_softening * prob_expansion
    smooth_span = int(1.0 * fs)
    kernel = np.ones(smooth_span) / smooth_span
    ewi_smoothed = np.convolve(ewi_robust, kernel, mode="same")
    
    med_ewi, mad_sigma_ewi = compute_mad_stats(ewi_smoothed[baseline_mask])
    threshold_ewi_3sigma = med_ewi + 3.0 * mad_sigma_ewi
    threshold_ewi_2sigma = med_ewi + 2.0 * mad_sigma_ewi
    
    alarm_mask_3sig = (t_valid < 0.0) & (t_valid > -40.0) & (ewi_smoothed > threshold_ewi_3sigma)
    alarm_indices_3sig = np.where(alarm_mask_3sig)[0]
    lead_time_3sig = - float(t_valid[alarm_indices_3sig[0]]) if len(alarm_indices_3sig) > 0 else 0.0

    alarm_mask_2sig = (t_valid < 0.0) & (t_valid > -40.0) & (ewi_smoothed > threshold_ewi_2sigma)
    alarm_indices_2sig = np.where(alarm_mask_2sig)[0]
    lead_time_2sig = - float(t_valid[alarm_indices_2sig[0]]) if len(alarm_indices_2sig) > 0 else 0.0
    
    max_pre_ewi = float(np.max(ewi_smoothed[(t_valid < 0.0) & (t_valid > -40.0)]))

    return {
        "record": rec_name,
        "lead_time_3sigma": lead_time_3sig,
        "lead_time_2sigma": lead_time_2sig,
        "baseline_med_omega": med_omega,
        "baseline_mad_sigma_omega": mad_sigma_omega,
        "baseline_med_theta": med_theta,
        "baseline_mad_sigma_theta": mad_sigma_theta,
        "threshold_3sigma": threshold_ewi_3sigma,
        "threshold_2sigma": threshold_ewi_2sigma,
        "max_pre_ewi": max_pre_ewi,
        "t_valid": t_valid,
        "ecg_valid": ecg_valid,
        "omega1_valid": omega1_valid,
        "theta_valid": theta_valid,
        "ewi_smoothed": ewi_smoothed
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
        res = analyze_robust_record(r, label)
        if res is not None:
            results.append({
                "record": res["record"],
                "lead_time_3sigma_sec": res["lead_time_3sigma"],
                "lead_time_2sigma_sec": res["lead_time_2sigma"],
                "baseline_med_omega": res["baseline_med_omega"],
                "baseline_mad_sigma_omega": res["baseline_mad_sigma_omega"],
                "max_pre_ewi": res["max_pre_ewi"],
                "threshold_3sigma": res["threshold_3sigma"],
                "threshold_2sigma": res["threshold_2sigma"]
            })
            
            plt.figure(figsize=(12, 10), dpi=300)
            
            plt.subplot(4, 1, 1)
            plt.plot(res["t_valid"], res["ecg_valid"], color="black", lw=0.8)
            plt.axvline(0, color="red", linestyle="--", lw=1.5, label="Clinical Onset (t=0)")
            plt.ylabel("ECG (mV)")
            plt.title(f"Robust Early Warning Detection (MAD-Scaled): Record {res['record']}")
            plt.legend(loc="upper left")
            plt.grid(True, alpha=0.3)
            
            plt.subplot(4, 1, 2)
            plt.plot(res["t_valid"], res["omega1_valid"], color="blue", lw=0.9, alpha=0.7, label="Turbulence Gap Omega_1")
            plt.axhline(0, color="gray", linestyle=":", lw=1.0)
            plt.axhline(res["baseline_med_omega"], color="blue", linestyle="--", alpha=0.5, label=f"Median ({res['baseline_med_omega']:.2f})")
            plt.axvline(0, color="red", linestyle="--", lw=1.5)
            plt.ylabel("Gap Omega_1")
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            
            plt.subplot(4, 1, 3)
            plt.plot(res["t_valid"], res["theta_valid"], color="green", lw=0.9, alpha=0.7, label="Raychaudhuri Theta")
            plt.axhline(res["baseline_med_theta"], color="green", linestyle="--", alpha=0.5)
            plt.axvline(0, color="red", linestyle="--", lw=1.5)
            plt.ylabel("Theta")
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            
            plt.subplot(4, 1, 4)
            plt.plot(res["t_valid"], res["ewi_smoothed"], color="crimson", lw=1.5, label="Robust Adaptive EWI")
            plt.axhline(res["threshold_3sigma"], color="orange", linestyle="--", lw=1.2, label=f"3-Sigma MAD ({res['threshold_3sigma']:.3f})")
            plt.axhline(res["threshold_2sigma"], color="gold", linestyle=":", lw=1.2, label=f"2-Sigma MAD ({res['threshold_2sigma']:.3f})")
            plt.axvline(0, color="red", linestyle="--", lw=1.5)
            if res["lead_time_3sigma"] > 0:
                plt.axvline(-res["lead_time_3sigma"], color="purple", linestyle=":", lw=2.0, label=f"3-Sigma Alarm ({res['lead_time_3sigma']:.1f} s)")
            elif res["lead_time_2sigma"] > 0:
                plt.axvline(-res["lead_time_2sigma"], color="magenta", linestyle=":", lw=2.0, label=f"2-Sigma Alarm ({res['lead_time_2sigma']:.1f} s)")
            plt.ylabel("EWI")
            plt.xlabel("Time Relative to Clinical Onset (seconds)")
            plt.legend(loc="upper left")
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            out_img = os.path.join("results", f"vfdb_{res['record']}_robust_plot.png")
            plt.savefig(out_img)
            plt.close()
            
    df_res = pd.DataFrame(results)
    csv_out = os.path.join("results", "robust_benchmark_summary.csv")
    df_res.to_csv(csv_out, index=False)
    print(df_res.to_string())

if __name__ == "__main__":
    main()
