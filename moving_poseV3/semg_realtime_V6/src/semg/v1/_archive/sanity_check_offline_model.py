"""
4 組對照實驗：診斷 SoftPINCH 66.5% vs 99.4% 準確率差距的根因。

實驗矩陣：
  A: Real-time 7-class + mu/sigma 正規化 (現狀 baseline)
  B: Real-time 7-class + 無正規化
  C: Offline 5-class  + mu/sigma 正規化
  D: Offline 5-class  + 無正規化 (預期最高)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
SOFTPINCH_NOTCH = 50
BANDPASS = (20, 450)
HAMPEL_HALF_WINDOW = 100
HAMPEL_SIGMA = 2
RMS_WINDOW = 500
RMS_STEP = 50
TRIM_SEC = 3.0
PHASE_SHIFT_FRACTION = 0.23

# Label mappings
PRED_MAP_7CLASS = {}
_idx = 0
for _t in ["Index", "Thumb", "Pinch"]:
    for _a in ["Contract", "Release"]:
        PRED_MAP_7CLASS[_idx] = f"{_t} {_a}"
        _idx += 1
PRED_MAP_7CLASS[_idx] = "Rest"

PRED_MAP_5CLASS = {
    0: "Index Contract",
    1: "Index Release",
    2: "Thumb Contract",
    3: "Thumb Release",
    4: "Rest",
}

CH_FOR_TARGET = {"Index": 1, "Thumb": 2}


def causal_notch_bandpass(x, fs, notch_freq, bandpass):
    from scipy.signal import butter, iirnotch, lfilter, lfilter_zi
    b_n, a_n = iirnotch(notch_freq, notch_freq / 2, fs=fs)
    zi = lfilter_zi(b_n, a_n) * x[0]
    y, _ = lfilter(b_n, a_n, x, zi=zi)
    b_b, a_b = butter(4, [bandpass[0] / (fs / 2), bandpass[1] / (fs / 2)], btype="band")
    zi = lfilter_zi(b_b, a_b) * y[0]
    y, _ = lfilter(b_b, a_b, y, zi=zi)
    return y


def causal_hampel(x, half_window, n_sigmas):
    from scipy.ndimage import median_filter
    kernel = 2 * half_window + 1
    med = median_filter(x, size=kernel, mode="reflect")
    mad = 1.4826 * median_filter(np.abs(x - med), size=kernel, mode="reflect")
    return np.where(np.abs(x - med) > n_sigmas * mad, med, x)


def rms_stream(x, window, step):
    n_win = (len(x) - window) // step + 1
    wins = np.lib.stride_tricks.sliding_window_view(x, window)[::step][:n_win]
    return np.sqrt(np.mean(wins ** 2, axis=1))


def preprocess_softpinch(raw_by_ch):
    envs = []
    for ch in (0, 1, 2):
        x = raw_by_ch[ch].astype(np.float64)
        x = pp.trim(x, FS, TRIM_SEC, TRIM_SEC)
        x = x - x.mean()
        filt = causal_notch_bandpass(x, FS, SOFTPINCH_NOTCH, BANDPASS)
        hampel = causal_hampel(filt, HAMPEL_HALF_WINDOW, HAMPEL_SIGMA)
        env = rms_stream(hampel, RMS_WINDOW, RMS_STEP)
        envs.append(env)
    min_len = min(len(e) for e in envs)
    return np.stack([e[:min_len] for e in envs], axis=1)


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
    for e in range(n_epochs):
        start = anchor_idx + e * period_samples
        epoch = env_in[start: start + period_samples]
        segments = {
            "Rest": epoch[0:seg_len],
            f"{target} Contract": epoch[seg_len:2 * seg_len],
            f"{target} Release": epoch[2 * seg_len:3 * seg_len],
        }
        for expected, seg in segments.items():
            device = next(model.parameters()).device
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
        raw = {ch: df[f"ch{ch}"].values for ch in (0, 1, 2)}
        env = preprocess_softpinch(raw)
        target = "Index" if posture == "index" else "Thumb"
        env_time = np.arange(len(env)) * (RMS_STEP / FS)
        anchor, period, _ = find_calibrated_anchor(env[:, CH_FOR_TARGET[target]], env_time)
        seg_df = evaluate_segments(env, model, pred_map, anchor, period, target, mu, sigma)
        seg_df["file"] = csv_path.stem
        all_segs.append(seg_df)

    result = pd.concat(all_segs, ignore_index=True)
    acc = result["correct"].mean()
    print(f"  [{label}] segment accuracy = {acc:.4f}  ({len(result)} segments)")

    # 混淆矩陣
    confusion = result.groupby(["expected", "predicted"]).size().reset_index(name="count")
    confusion = confusion.sort_values("count", ascending=False)
    print(f"  Top confusions:")
    for _, r in confusion.head(8).iterrows():
        marker = "✓" if r["expected"] == r["predicted"] else "✗"
        print(f"    {marker} {r['expected']:>20s} → {r['predicted']:<20s}  n={r['count']}")
    print()
    return {"experiment": label, "accuracy": acc, "n_segments": len(result)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load models
    rt_model, rt_ckpt = load_model(RT_MODEL_PATH / "model.pth")
    rt_model.to(device)
    print(f"Real-time model: output_dim={rt_ckpt['model_args']['output_dim']}")

    off_model, off_ckpt = load_model(OFFLINE_MODEL_PATH / "model.pth")
    off_model.to(device)
    print(f"Offline model: output_dim={off_ckpt['model_args']['output_dim']}")

    # Load mu/sigma from real-time checkpoint
    mu = np.load(RT_MODEL_PATH / "mu.npy")
    sigma = np.load(RT_MODEL_PATH / "sigma.npy")
    print(f"mu={mu}, sigma={sigma}\n")

    results = []

    # A: Real-time + norm (現狀)
    print("=" * 60)
    results.append(run_experiment("A: RT+norm", rt_model, PRED_MAP_7CLASS, mu, sigma))

    # B: Real-time + no norm
    print("=" * 60)
    results.append(run_experiment("B: RT+raw", rt_model, PRED_MAP_7CLASS, None, None))

    # C: Offline + norm
    print("=" * 60)
    results.append(run_experiment("C: OFF+norm", off_model, PRED_MAP_5CLASS, mu, sigma))

    # D: Offline + no norm (expected best)
    print("=" * 60)
    results.append(run_experiment("D: OFF+raw", off_model, PRED_MAP_5CLASS, None, None))

    # Summary
    print("=" * 60)
    print("\nSUMMARY:")
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))
    summary.to_csv(RESULTS_ROOT / "experiment_matrix_results.csv", index=False)
    print(f"\nSaved to results/experiment_matrix_results.csv")


if __name__ == "__main__":
    main()
