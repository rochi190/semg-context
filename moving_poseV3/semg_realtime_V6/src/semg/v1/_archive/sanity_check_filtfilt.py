"""
用 SoftPINCH 原始 preprocessing.py 的 filtfilt + uniform_filter1d RMS
重新跑 4 組實驗，排除前處理差異的影響。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import signal as sig
from scipy.ndimage import uniform_filter1d, median_filter

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")   # src/ 目錄，與檔案放在多深無關
sys.path.insert(0, str(_SRC))
from semg.core import pipeline as pp
from semg.core import quality as qm
from semg.core.softpinch_model import load_model

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "src").parent
RESULTS_ROOT = REPO_ROOT / "results"

RT_MODEL_PATH = (REPO_ROOT / "src/SoftPINCH/src/models/loggings/"
                 "Real_time_inference/SingleNet_CNN+LSTM_EMG/subject_0")
OFFLINE_MODEL_PATH = (REPO_ROOT / "src/SoftPINCH/src/models/loggings/"
                      "SingleNet_CNN+LSTM_EMG/all_subjects")
SOFTPINCH_DATA_ROOT = (REPO_ROOT / "src/SoftPINCH/src/experiment/data/"
                       "subject_independent/subject_0/EMG")

FS = 2000
NOTCH_FREQ = 50
BANDPASS = (20, 450)
HAMPEL_HALF_WINDOW = 100  # HAMPEL_WINDOWSIZE=100 → half_window for median_filter kernel=201
HAMPEL_SIGMA = 2
RMS_WINDOW = 500
RMS_STEP = 50
TRIM_PERIOD = 3  # seconds
TRIAL_PERIOD = 9  # seconds
PHASE_SHIFT_FRACTION = 0.23

PRED_MAP_7CLASS = {}
_idx = 0
for _t in ["Index", "Thumb", "Pinch"]:
    for _a in ["Contract", "Release"]:
        PRED_MAP_7CLASS[_idx] = f"{_t} {_a}"
        _idx += 1
PRED_MAP_7CLASS[_idx] = "Rest"

PRED_MAP_5CLASS = {
    0: "Index Contract", 1: "Index Release",
    2: "Thumb Contract", 3: "Thumb Release",
    4: "Rest",
}

CH_FOR_TARGET = {"Index": 1, "Thumb": 2}


def softpinch_preprocess(raw_2d):
    """
    Exact reproduction of SoftPINCH's EMG_preprocessing.preprocessing_routine:
    - filtfilt (zero-phase) notch and bandpass
    - trim both sides by TRIM_PERIOD
    - hampel filter (median_filter based)
    - RMS via uniform_filter1d (convolution approach)
    """
    # 1) Notch (filtfilt)
    b_n, a_n = sig.iirnotch(w0=NOTCH_FREQ, Q=30, fs=FS)
    emg_notch = sig.filtfilt(b_n, a_n, raw_2d, axis=0)

    # 2) Bandpass (sosfiltfilt)
    sos = sig.butter(4, [BANDPASS[0], BANDPASS[1]], btype='bandpass', fs=FS, output='sos')
    emg_bp = sig.sosfiltfilt(sos, emg_notch, axis=0)

    # 3) Trim — exact SoftPINCH method: trim_samples on both sides, then round to whole epochs
    trim_samples = FS * TRIM_PERIOD
    samples_per_epoch = FS * TRIAL_PERIOD
    valid_samples = emg_bp.shape[0] - 2 * trim_samples
    num_epochs = int(np.round(valid_samples / samples_per_epoch))
    trim_start = trim_samples
    trim_end = trim_start + num_epochs * samples_per_epoch
    emg_trim = emg_bp[trim_start:trim_end, :]

    # 4) Hampel (exact SoftPINCH: median_filter with half-window)
    kernel = 2 * HAMPEL_HALF_WINDOW + 1
    med = median_filter(emg_trim, size=(kernel, 1), mode="reflect")
    mad = 1.4826 * median_filter(np.abs(emg_trim - med), size=(kernel, 1), mode="reflect")
    threshold = HAMPEL_SIGMA * mad
    emg_hampel = np.where(np.abs(emg_trim - med) > threshold, med, emg_trim)

    # 5) RMS via uniform_filter1d (exact SoftPINCH rms_conv)
    power = emg_hampel ** 2
    mean_power = uniform_filter1d(power, size=RMS_WINDOW, axis=0, mode="nearest")
    rms = np.sqrt(mean_power)
    rms_downsampled = rms[::RMS_STEP]

    return rms_downsampled, num_epochs


def find_calibrated_anchor(env_ch, env_time):
    strength, period_s = qm.periodicity_strength(env_ch, env_time, min_period_s=6.0, max_period_s=15.0)
    if strength <= 0 or period_s <= 0:
        period_s = 9.0
    fs_env = 1.0 / np.median(np.diff(env_time))
    period_samples = int(round(period_s * fs_env))
    raw_anchor = qm.find_cycle_anchor_index(env_ch, env_time, period_s)
    shift = int(round(PHASE_SHIFT_FRACTION * period_samples))
    anchor = (raw_anchor + shift) % period_samples
    return anchor, period_samples, period_s


def evaluate_segments(env, model, pred_map, anchor_idx, period_samples, target,
                      mu=None, sigma=None):
    if mu is not None and sigma is not None:
        env_in = (env - mu) / (sigma + 1e-8)
    else:
        env_in = env

    n_epochs = (len(env_in) - anchor_idx) // period_samples
    seg_len = period_samples // 3
    rows = []
    device = next(model.parameters()).device
    for e in range(n_epochs):
        start = anchor_idx + e * period_samples
        epoch = env_in[start: start + period_samples]
        segments = {
            "Rest": epoch[0:seg_len],
            f"{target} Contract": epoch[seg_len:2 * seg_len],
            f"{target} Release": epoch[2 * seg_len:3 * seg_len],
        }
        for expected, seg in segments.items():
            inp = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _, _ = model(inp)
                probs = torch.softmax(logits, dim=1)
                pred_idx = int(torch.argmax(logits, dim=1).item())
            pred_label = pred_map.get(pred_idx, f"unknown_{pred_idx}")
            rows.append({
                "epoch_idx": e, "expected": expected,
                "predicted": pred_label,
                "confidence": float(probs[0, pred_idx].item()),
                "correct": pred_label == expected,
            })
    return pd.DataFrame(rows)


def run_experiment(label, model, pred_map, mu, sigma):
    files = []
    for posture in ("index", "thumb"):
        for p in sorted(SOFTPINCH_DATA_ROOT.glob(f"flex_{posture}_finger_*.csv")):
            files.append((p, posture))

    all_segs = []
    for csv_path, posture in files:
        df = pd.read_csv(csv_path)
        # SoftPINCH data: (samples, channels) 2D array
        raw_2d = df[["ch0", "ch1", "ch2"]].values
        env, _ = softpinch_preprocess(raw_2d)
        target = "Index" if posture == "index" else "Thumb"
        env_time = np.arange(len(env)) * (RMS_STEP / FS)
        anchor, period, _ = find_calibrated_anchor(env[:, CH_FOR_TARGET[target]], env_time)
        seg_df = evaluate_segments(env, model, pred_map, anchor, period, target, mu, sigma)
        seg_df["file"] = csv_path.stem
        all_segs.append(seg_df)

    result = pd.concat(all_segs, ignore_index=True)
    acc = result["correct"].mean()
    print(f"  [{label}] segment accuracy = {acc:.4f}  ({len(result)} segments)")
    confusion = result.groupby(["expected", "predicted"]).size().reset_index(name="count")
    for _, r in confusion.sort_values("count", ascending=False).head(8).iterrows():
        m = "✓" if r["expected"] == r["predicted"] else "✗"
        print(f"    {m} {r['expected']:>20s} → {r['predicted']:<20s}  n={r['count']}")
    print()
    return {"experiment": label, "accuracy": acc, "n_segments": len(result)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    rt_model, _ = load_model(RT_MODEL_PATH / "model.pth")
    rt_model.to(device)
    off_model, _ = load_model(OFFLINE_MODEL_PATH / "model.pth")
    off_model.to(device)
    mu = np.load(RT_MODEL_PATH / "mu.npy")
    sigma = np.load(RT_MODEL_PATH / "sigma.npy")
    print(f"mu={mu}, sigma={sigma}\n")

    results = []
    print("Using SoftPINCH's ORIGINAL preprocessing (filtfilt + uniform_filter1d)")
    print("=" * 60)
    results.append(run_experiment("A2: RT+norm (filtfilt)", rt_model, PRED_MAP_7CLASS, mu, sigma))
    print("=" * 60)
    results.append(run_experiment("B2: RT+raw (filtfilt)", rt_model, PRED_MAP_7CLASS, None, None))
    print("=" * 60)
    results.append(run_experiment("C2: OFF+norm (filtfilt)", off_model, PRED_MAP_5CLASS, mu, sigma))
    print("=" * 60)
    results.append(run_experiment("D2: OFF+raw (filtfilt)", off_model, PRED_MAP_5CLASS, None, None))

    print("=" * 60)
    print("\nSUMMARY (filtfilt preprocessing):")
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))
    summary.to_csv(RESULTS_ROOT / "experiment_matrix_filtfilt_results.csv", index=False)
    print(f"\nSaved to results/experiment_matrix_filtfilt_results.csv")


if __name__ == "__main__":
    main()
