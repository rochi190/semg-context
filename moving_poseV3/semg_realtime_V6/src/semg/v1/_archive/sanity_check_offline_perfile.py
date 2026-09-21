"""
修正版 sanity check：offline 5-class model + per-file z-score 正規化。
基於診斷結果，這才是正確的組合。
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
from semg.core import quality as qm
from semg.core.softpinch_model import load_model

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "src").parent
RESULTS_ROOT = REPO_ROOT / "results"

OFFLINE_MODEL_PATH = (REPO_ROOT / "src/SoftPINCH/src/models/loggings/"
                      "SingleNet_CNN+LSTM_EMG/all_subjects")
SOFTPINCH_DATA_ROOT = (REPO_ROOT / "src/SoftPINCH/src/experiment/data/"
                       "subject_independent/subject_0/EMG")

FS = 2000; NOTCH = 50; BANDPASS = (20, 450)
HAMPEL_HW = 100; HAMPEL_S = 2
RMS_W = 500; RMS_S = 50; TRIM = 3; TRIAL = 9
PHASE_SHIFT = 0.23

PRED_MAP = {0:"Index Contract", 1:"Index Release",
            2:"Thumb Contract", 3:"Thumb Release", 4:"Rest"}
CH_FOR = {"Index": 1, "Thumb": 2}


def softpinch_preprocess(raw_2d):
    b, a = sig.iirnotch(NOTCH, 30, fs=FS)
    y = sig.filtfilt(b, a, raw_2d, axis=0)
    sos = sig.butter(4, list(BANDPASS), btype='bandpass', fs=FS, output='sos')
    y = sig.sosfiltfilt(sos, y, axis=0)
    ts = FS * TRIM; spe = FS * TRIAL
    ne = int(np.round((y.shape[0] - 2*ts) / spe))
    y = y[ts:ts+ne*spe, :]
    k = 2*HAMPEL_HW+1
    med = median_filter(y, size=(k,1), mode="reflect")
    mad = 1.4826 * median_filter(np.abs(y-med), size=(k,1), mode="reflect")
    y = np.where(np.abs(y-med) > HAMPEL_S*mad, med, y)
    rms = np.sqrt(uniform_filter1d(y**2, size=RMS_W, axis=0, mode="nearest"))
    return rms[::RMS_S]


def find_anchor(env_ch, env_time):
    s, p = qm.periodicity_strength(env_ch, env_time, 6.0, 15.0)
    if s <= 0 or p <= 0: p = 9.0
    fs_e = 1.0 / np.median(np.diff(env_time))
    ps = int(round(p * fs_e))
    ra = qm.find_cycle_anchor_index(env_ch, env_time, p)
    return (ra + int(round(PHASE_SHIFT * ps))) % ps, ps, p


def evaluate(env, model, anchor, period, target):
    # Per-file z-score
    mu, sigma = env.mean(axis=0), env.std(axis=0)
    env_n = (env - mu) / (sigma + 1e-8)

    ne = (len(env_n) - anchor) // period
    sl = period // 3
    rows = []
    dev = next(model.parameters()).device
    for e in range(ne):
        ep = env_n[anchor + e*period: anchor + (e+1)*period]
        segs = {"Rest": ep[:sl],
                f"{target} Contract": ep[sl:2*sl],
                f"{target} Release": ep[2*sl:3*sl]}
        for exp, s in segs.items():
            inp = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(dev)
            with torch.no_grad():
                l, _, _ = model(inp)
                p = torch.softmax(l, 1)
                pi = int(l.argmax(1).item())
            pl = PRED_MAP.get(pi, f"unk_{pi}")
            rows.append({"epoch_idx": e, "expected": exp, "predicted": pl,
                         "confidence": float(p[0,pi].item()), "correct": pl == exp})
    return pd.DataFrame(rows)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(OFFLINE_MODEL_PATH / "model.pth")
    model.to(dev)
    print(f"Model: output_dim={ckpt['model_args']['output_dim']}, device={dev}")

    files = []
    for pos in ("index", "thumb"):
        for p in sorted(SOFTPINCH_DATA_ROOT.glob(f"flex_{pos}_finger_*.csv")):
            files.append((p, pos))

    all_segs, file_rows = [], []
    for csv_path, pos in files:
        df = pd.read_csv(csv_path)
        env = softpinch_preprocess(df[["ch0","ch1","ch2"]].values)
        tgt = "Index" if pos == "index" else "Thumb"
        et = np.arange(len(env)) * (RMS_S / FS)
        anc, per, ps = find_anchor(env[:, CH_FOR[tgt]], et)
        seg_df = evaluate(env, model, anc, per, tgt)
        seg_df["file"] = csv_path.stem
        all_segs.append(seg_df)
        acc = seg_df["correct"].mean()
        file_rows.append({"file": csv_path.stem, "posture": pos,
                          "period_s": ps, "n_epochs": seg_df["epoch_idx"].nunique(),
                          "accuracy": acc})
        print(f"  {csv_path.stem}: {acc:.4f} ({seg_df['epoch_idx'].nunique()} epochs)")

    result = pd.concat(all_segs, ignore_index=True)
    overall = result["correct"].mean()
    print(f"\nOverall segment accuracy: {overall:.4f} ({len(result)} segments)")

    print("\nConfusion matrix:")
    for _, r in result.groupby(["expected","predicted"]).size().reset_index(name="n").sort_values("n",ascending=False).head(10).iterrows():
        m = "✓" if r["expected"]==r["predicted"] else "✗"
        print(f"  {m} {r['expected']:>20s} → {r['predicted']:<20s}  n={r['n']}")

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for cls in sorted(result["expected"].unique()):
        sub = result[result["expected"]==cls]
        print(f"  {cls:>20s}: {sub['correct'].mean():.4f} ({len(sub)} segments)")

    report = pd.DataFrame(file_rows)
    report.to_csv(RESULTS_ROOT / "sanity_check_offline_perfile_znorm.csv", index=False)
    result.to_csv(RESULTS_ROOT / "sanity_check_offline_perfile_znorm_segments.csv", index=False)
    print(f"\nSaved to results/sanity_check_offline_perfile_znorm*.csv")


if __name__ == "__main__":
    main()
