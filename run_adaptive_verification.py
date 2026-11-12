import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
from src.turbulence_detector import TurbulenceGapDetector

def run_adaptive_experiment():
    rec_name = "418"
    pn_dir = "vfdb"
    
    ann = wfdb.rdann(rec_name, "atr", pn_dir=pn_dir)
    vf_sample = None
    for s, note in zip(ann.sample, ann.aux_note):
        clean_note = note.replace("\x00", "").strip()
        if "VF" in clean_note:
            vf_sample = int(s)
            break
            
    if vf_sample is None:
        raise RuntimeError("No VF episode found in annotations")

    fs = 250.0
    dt = 1.0 / fs
    
    pre_samples = int(120.0 * fs)
    post_samples = int(30.0 * fs)
    
    start_sample = max(0, vf_sample - pre_samples)
    end_sample = vf_sample + post_samples
    
    record, info = wfdb.rdsamp(rec_name, pn_dir=pn_dir, sampfrom=start_sample, sampto=end_sample)
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
    mu_omega = np.mean(omega1_valid[baseline_mask])
    std_omega = np.std(omega1_valid[baseline_mask])
    
    mu_theta = np.mean(theta_valid[baseline_mask])
    std_theta = np.std(theta_valid[baseline_mask])
    
    z_omega = (omega1_valid - mu_omega) / (std_omega + 1e-8)
    z_theta = (theta_valid - mu_theta) / (std_theta + 1e-8)
    
    prob_softening = 1.0 / (1.0 + np.exp(z_omega))
    prob_expansion = 1.0 / (1.0 + np.exp(- z_theta))
    
    ewi_adaptive = prob_softening * prob_expansion
    
    smooth_span = int(1.0 * fs)
    kernel = np.ones(smooth_span) / smooth_span
    ewi_smoothed = np.convolve(ewi_adaptive, kernel, mode="same")
    
    df_out = pd.DataFrame({
        "time_relative_to_vf_sec": t_valid,
        "ecg_lead_0": ecg_valid,
        "turbulence_gap_omega1": omega1_valid,
        "raychaudhuri_theta": theta_valid,
        "z_score_omega1": z_omega,
        "z_score_theta": z_theta,
        "adaptive_ewi": ewi_smoothed
    })
    
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "vfdb_418_adaptive_verification.csv")
    df_out.to_csv(csv_path, index=False)
    
    baseline_ewi_max = np.max(ewi_smoothed[baseline_mask])
    baseline_ewi_mean = np.mean(ewi_smoothed[baseline_mask])
    baseline_ewi_std = np.std(ewi_smoothed[baseline_mask])
    
    threshold_ewi = baseline_ewi_mean + 3.0 * baseline_ewi_std
    
    alarm_mask = (t_valid < 0.0) & (t_valid > -40.0) & (ewi_smoothed > threshold_ewi)
    alarm_indices = np.where(alarm_mask)[0]
    
    if len(alarm_indices) > 0:
        first_alarm_idx = alarm_indices[0]
        lead_time = - t_valid[first_alarm_idx]
    else:
        lead_time = 0.0

    plt.figure(figsize=(12, 10), dpi=300)
    
    plt.subplot(4, 1, 1)
    plt.plot(t_valid, ecg_valid, color="black", lw=0.8)
    plt.axvline(0, color="red", linestyle="--", lw=1.5, label="Clinical VF Onset (t=0)")
    plt.ylabel("ECG (mV)")
    plt.title("Experimental Verification with Statistically Grounded EWI (PhysioNet VFDB 418)")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 1, 2)
    plt.plot(t_valid, omega1_valid, color="blue", lw=0.9, alpha=0.7, label="Turbulence Gap Omega_1")
    plt.axhline(0, color="gray", linestyle=":", lw=1.0)
    plt.axhline(mu_omega, color="blue", linestyle="--", alpha=0.5, label=f"Baseline Mean ({mu_omega:.2f})")
    plt.axvline(0, color="red", linestyle="--", lw=1.5)
    plt.ylabel("Gap Omega_1")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 1, 3)
    plt.plot(t_valid, theta_valid, color="green", lw=0.9, alpha=0.7, label="Raychaudhuri Theta")
    plt.axhline(mu_theta, color="green", linestyle="--", alpha=0.5, label=f"Baseline Mean ({mu_theta:.2f})")
    plt.axvline(0, color="red", linestyle="--", lw=1.5)
    plt.ylabel("Theta")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 1, 4)
    plt.plot(t_valid, ewi_smoothed, color="crimson", lw=1.5, label="Adaptive EWI (Smoothed)")
    plt.axhline(threshold_ewi, color="orange", linestyle="--", lw=1.2, label=f"3-Sigma Warning Threshold ({threshold_ewi:.3f})")
    plt.axvline(0, color="red", linestyle="--", lw=1.5)
    if lead_time > 0:
        plt.axvline(-lead_time, color="purple", linestyle=":", lw=2.0, label=f"Detection Trigger (Lead Time = {lead_time:.2f} s)")
    plt.ylabel("Adaptive EWI")
    plt.xlabel("Time Relative to Clinical VF Onset (seconds)")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    img_path = os.path.join("results", "vfdb_418_adaptive_early_warning_plot.png")
    plt.savefig(img_path)
    plt.close()
    
    print(f"EXPERIMENT_SUCCESS: true")
    print(f"BASELINE_MEAN_OMEGA: {mu_omega:.4f}")
    print(f"BASELINE_MEAN_THETA: {mu_theta:.4f}")
    print(f"THRESHOLD_EWI: {threshold_ewi:.4f}")
    print(f"LEAD_TIME_SECONDS: {lead_time:.2f}")
    print(f"CSV_PATH: {csv_path}")
    print(f"IMG_PATH: {img_path}")

if __name__ == "__main__":
    run_adaptive_experiment()
