# Operator-Theoretic Early Warning Diagnostics for Non-Equilibrium Phase Transitions

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Repository](https://img.shields.io/badge/GitHub-sukuyoshi%2Fturbulence__detector-blue)](https://github.com/sukuyoshi/turbulence_detector)

Autonomous detection of spectral turbulence gap softening and Raychaudhuri expansion in biophysical non-equilibrium time series (cardiac and neural dynamics).

Companion codebase to the research manuscript:
> **Operator-Theoretic Early Warning Diagnostics for Non-Equilibrium Phase Transitions: Predicting Lethal Cardiac Cascades via Sturm-Liouville Gap Softening and Raychaudhuri Expansion**  
> *Reiji Yoshikai (2026)*. Source: [https://github.com/sukuyoshi/turbulence_detector](https://github.com/sukuyoshi/turbulence_detector)

## Overview

This repository provides an open-source (MIT Licensed) implementation of a hybrid operator-theoretic early warning architecture designed to predict catastrophic phase transitions (such as ventricular fibrillation, flutter, and tachycardia) from non-invasive scalar time series (ECG, EEG).

Grounding biophysical homeostasis in the mathematical universality class of non-equilibrium dissipative systems, the framework integrates:
1. **Dual-Timescale Filtering**: Decouples slow drift (persistent softening) from fast collapse transients via multi-rate convolution.
2. **Instantaneous Collapse Depth**: Direct monitoring of Sturm-Liouville spectrum breakdown ($\hat{\Omega}_1 < -3.0$).
3. **Duty Cycle Criterion**: Robust rejection of single-sample noise artifacts by enforcing a continuous state residence duration ($\mathcal{D}_{\mathrm{collapse}} \ge 0.20$ within a 1.0 s moving window).
4. **Hybrid Multi-Logic Alarm**: Union logic spanning asymptotic $3\sigma$ MAD drift, moderate $2\sigma$ shift with elevated duty cycle, and acute deep collapse triggers.

## Comprehensive Benchmark Results (PhysioNet VFDB)

### 1. Initial Development Set (4 Records)

| Record | Pathology Classification | Detection Outcome | Lead Time | Trigger Logic |
| :--- | :--- | :--- | :--- | :--- |
| **Record 418** | Ventricular Flutter (VFL) | **Detected** | **22.08 s** | Cond A ($3\sigma$ EWI) |
| **Record 422** | Ventricular Fibrillation (VFIB) | **Detected** | **5.88 s** | Cond C (Acute Collapse) |
| **Record 424** | Ventricular Fibrillation (VFIB) | **Detected** | **38.74 s** | Cond B ($2\sigma$ + Duty) |
| **Record 426** | Ventricular Fibrillation (VF) | **Detected** | **38.22 s** | Cond C (Acute Collapse) |

### 2. Blind Test on Unseen Records (18 Records, Zero Hyperparameter Tuning)

Evaluated strictly across all 18 remaining unexamined records of the MIT-BIH Malignant Ventricular Ectopy Database:

- **Total Evaluated Records**: 18
- **Detected**: 16 / 18
- **Blind Test Sensitivity**: **88.89%**
- **Mean Lead Time**: **31.27 s**
- **Median Lead Time**: **35.26 s**
- **Standard Deviation**: **9.42 s**

Individual record performance on unseen test set:
- 419 (VFL): **35.48 s** (Cond B)
- 420 (VT): **39.66 s** (Cond C)
- 421 (VT): Undetected (0.00 s, pre-existing chronic breakdown)
- 423 (VT): **39.99 s** (Cond C)
- 425 (VT): **39.39 s** (Cond C)
- 427 (VT): **39.78 s** (Cond C)
- 428 (VT): **10.86 s** (Cond C)
- 429 (VT): Undetected (0.00 s, pre-existing chronic breakdown)
- 430 (VT): **30.12 s** (Cond C)
- 602 (VT): **39.32 s** (Cond C)
- 605 (VT): **21.90 s** (Cond C)
- 607 (VT): **23.89 s** (Cond C)
- 609 (VFL): **36.96 s** (Cond B)
- 610 (VT): **14.62 s** (Cond B)
- 611 (VT): **20.94 s** (Cond C)
- 612 (VT): **35.04 s** (Cond C)
- 614 (VT): **38.61 s** (Cond C)
- 615 (VT): **33.77 s** (Cond C)

### 3. Overall Database Summary (All 22 VFDB Records)

- **Total Records Analyzed**: 22
- **Successful Pre-Transition Alarms**: 20 / 22
- **Overall Database Sensitivity**: **90.91%**
- **Grand Mean Lead Time**: **30.26 s** prior to clinical onset ($t = 0$).

## Specificity and False Positive Benchmark (Step 2)

Evaluated across healthy normal sinus rhythm recordings (PhysioNet NSRDB: 16265, 16272) and long quiescent pre-arrhythmic segments from VFDB (420, 422, 424) totaling 1.67 hours of baseline monitoring without phase transitions:

### 1. Raw High-Frequency Chattering and Time Specificity

At 250 Hz sampling without event consolidation, fast discrete transitions near decision boundaries exhibit sample-level chattering:

| Database | Record | Monitoring Time | Alarm Duration | Time Specificity | Raw Sample Pulses |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NSRDB** | 16265 | 0.48 h | 77.58 s | **95.54%** | 3,359 |
| **NSRDB** | 16272 | 0.48 h | 0.22 s | **99.99%** | 19 |
| **VFDB Quiescent** | 420 | 0.23 h | 97.97 s | **88.33%** | 3,431 |
| **VFDB Quiescent** | 422 | 0.23 h | 14.48 s | **98.28%** | 32 |
| **VFDB Quiescent** | 424 | 0.23 h | 4.05 s | **99.52%** | 65 |
| **Total / Mean** | **5 Records** | **1.67 h** | **194.30 s** | **96.33%** | **6,906** |

### 2. Clinical Event Consolidation (Debounce 5.0 s, Minimum Persistence)

Standard clinical telemetry alarm protocols group contiguous bursts within 5.0 seconds and enforce minimum persistence:

| Database | Record | FP/h (Min Persistence 0.5 s) | FP/h (Min Persistence 1.0 s) |
| :--- | :--- | :--- | :--- |
| **NSRDB** | 16265 | 14.49 | 10.35 |
| **NSRDB** | 16272 | **0.00** (Zero false alarms) | **0.00** (Zero false alarms) |
| **VFDB Quiescent** | 420 | 25.74 | 8.58 |
| **VFDB Quiescent** | 422 | 34.31 | 21.45 |
| **VFDB Quiescent** | 424 | 12.87 | 4.29 |
| **Healthy Average (NSRDB)** | - | **7.25 FP/h** | **5.18 FP/h** |

## Repository Structure

- `LICENSE`: MIT License (Copyright 2026 Reiji Yoshikai).
- `src/turbulence_detector.py`: Baseline Takens embedding and Koopman operator generator.
- `src/hybrid_turbulence_detector.py`: Hybrid dual-timescale and duty-cycle detection engine.
- `run_hybrid_verification.py`: Verification benchmark on development records.
- `run_blind_test.py`: Automated blind testing script on all unseen records.
- `results/`:
  - `blind_test_unseen_records.csv`: Full blind test results across 18 records.
  - `hybrid_benchmark_summary.csv`: Development set benchmark summary.
  - Verification plot PNGs (300 DPI).

## Installation and Execution

```bash
pip install numpy scipy pandas matplotlib wfdb
python run_blind_test.py
```
