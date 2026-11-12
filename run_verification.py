import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
from src.turbulence_detector import TurbulenceGapDetector

def run_experiment():
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
        sigma_omega=0.8,
        theta_threshold=0.0
    )
    
    omega1, theta, ewi = detector.analyze_stream(raw_ecg)
    
    shift_idx = (d_dim - 1) * tau_step
    time_embedded = time_axis[shift_idx:]
    raw_ecg_embedded = raw_ecg[shift_idx:]
    
    valid_mask = ~np.isnan(omega1)
    t_valid = time_embedded[valid_mask]
    ecg_valid = raw_ecg_embedded[valid_mask]
    omega1_valid = omega1[valid_mask]
    theta_valid = theta[valid_mask]
    ewi_valid = ewi[valid_mask]
    
    df_out = pd.DataFrame({
        "time_relative_to_vf_sec": t_valid,
        "ecg_lead_0": ecg_valid,
        "turbulence_gap_omega1": omega1_valid,
        "raychaudhuri_theta": theta_valid,
        "early_warning_index_ewi": ewi_valid
    })
    
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "vfdb_418_verification.csv")
    df_out.to_csv(csv_path, index=False)
    
    ewi_threshold = 0.5
    alarm_indices = np.where((t_valid < 0.0) & (ewi_valid > ewi_threshold))[0]
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
    plt.title("Experimental Verification: Real-Time Turbulence Gap and Raychaudhuri Expansion (PhysioNet VFDB 418)")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 1, 2)
    plt.plot(t_valid, omega1_valid, color="blue", lw=1.0)
    plt.axhline(0, color="gray", linestyle=":", lw=1.0)
    plt.axvline(0, color="red", linestyle="--", lw=1.5)
    plt.ylabel("Gap Omega_1")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 1, 3)
    plt.plot(t_valid, theta_valid, color="green", lw=1.0)
    plt.axhline(0, color="gray", linestyle=":", lw=1.0)
    plt.axvline(0, color="red", linestyle="--", lw=1.5)
    plt.ylabel("Raychaudhuri Theta")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 1, 4)
    plt.plot(t_valid, ewi_valid, color="crimson", lw=1.2, label="Early Warning Index (EWI)")
    plt.axhline(ewi_threshold, color="orange", linestyle="--", lw=1.0, label="Warning Threshold")
    plt.axvline(0, color="red", linestyle="--", lw=1.5)
    if lead_time > 0:
        plt.axvline(-lead_time, color="purple", linestyle=":", lw=1.5, label=f"First Warning (Lead Time = {lead_time:.1f} s)")
    plt.ylabel("EWI")
    plt.xlabel("Time Relative to Clinical VF Onset (seconds)")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    img_path = os.path.join("results", "vfdb_418_early_warning_plot.png")
    plt.savefig(img_path)
    plt.close()
    
    print(f"EXPERIMENT_SUCCESS: true")
    print(f"LEAD_TIME_SECONDS: {lead_time:.2f}")
    print(f"CSV_PATH: {csv_path}")
    print(f"IMG_PATH: {img_path}")

if __name__ == "__main__":
    run_experiment()
