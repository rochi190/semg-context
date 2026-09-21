"""
完整的 SoftPINCH 重現 pipeline。
在 LOSO hold-out subject (subject_8) 和多個 subject 上驗證。

方法：
  1. SoftPINCH 原始前處理 (filtfilt notch+bandpass, hampel, uniform_filter1d RMS)
  2. 精確 epoch 邊界 (trim 3s 兩端, 每 9s 一個 epoch)
  3. Per-file z-score 正規化
  4. 三等分 segment 評估 (rest/contract/release 各 3s)
  5. Offline LOSO model (5-class, output_dim=5)
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
from semg.core.softpinch_model import load_model

# === 常數 ===
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "src").parent
MODEL_PATH = REPO_ROOT / "src/SoftPINCH/src/models/loggings/SingleNet_CNN+LSTM_EMG/all_subjects/model.pth"
DATA_ROOT = REPO_ROOT / "src/SoftPINCH/src/experiment/data/subject_independent"
RESULTS_ROOT = REPO_ROOT / "results"

FS = 2000           # 取樣率
NOTCH = 50          # SoftPINCH 用丹麥電網 50Hz（我們自己的資料改 60Hz）
BANDPASS = (20, 450)
HAMPEL_HW = 100     # half-window for hampel
HAMPEL_SIGMA = 2
RMS_WINDOW = 500    # 250ms @ 2kHz
RMS_STEP = 50       # 25ms → 90% overlap → 40Hz output
TRIM_SEC = 3        # 兩端各 trim 3 秒
TRIAL_SEC = 9       # 每個 trial 9 秒 (3s rest + 3s contract + 3s release)

# 5-class label mapping (2 motions × 2 actions + shared rest)
PRED_MAP = {
    0: "Index Contract", 1: "Index Release",
    2: "Thumb Contract", 3: "Thumb Release",
    4: "Rest",
}


def preprocess(raw_2d: np.ndarray) -> tuple[np.ndarray, int]:
    """
    完全複製 SoftPINCH 的 EMG_preprocessing.preprocessing_routine()。
    
    輸入: raw_2d — shape (n_samples, 3), 3 通道 EMG
    輸出: (rms_envelope, num_epochs)
           rms_envelope shape = (n_rms_points, 3)
    """
    # 1) Notch filter (filtfilt = 零相位)
    b, a = sig.iirnotch(w0=NOTCH, Q=30, fs=FS)
    y = sig.filtfilt(b, a, raw_2d, axis=0)

    # 2) Bandpass filter (sosfiltfilt = 零相位)
    sos = sig.butter(4, list(BANDPASS), btype='bandpass', fs=FS, output='sos')
    y = sig.sosfiltfilt(sos, y, axis=0)

    # 3) Trim + epoch 對齊
    trim_samples = FS * TRIM_SEC
    samples_per_epoch = FS * TRIAL_SEC
    valid_samples = y.shape[0] - 2 * trim_samples
    num_epochs = int(np.round(valid_samples / samples_per_epoch))
    trim_end = trim_samples + num_epochs * samples_per_epoch
    y = y[trim_samples:trim_end, :]

    # 4) Hampel filter
    kernel = 2 * HAMPEL_HW + 1
    med = median_filter(y, size=(kernel, 1), mode="reflect")
    mad = 1.4826 * median_filter(np.abs(y - med), size=(kernel, 1), mode="reflect")
    y = np.where(np.abs(y - med) > HAMPEL_SIGMA * mad, med, y)

    # 5) RMS via uniform_filter1d (SoftPINCH 的 rms_conv)
    rms = np.sqrt(uniform_filter1d(y ** 2, size=RMS_WINDOW, axis=0, mode="nearest"))
    rms = rms[::RMS_STEP]  # 降取樣

    return rms, num_epochs


def evaluate_subject(model, subject_dir: Path, device) -> pd.DataFrame:
    """對一個 subject 的所有 index/thumb 檔案做 segment-level 評估。"""
    emg_dir = subject_dir / "EMG"
    if not emg_dir.exists():
        return pd.DataFrame()

    pts_per_epoch = (FS * TRIAL_SEC) // RMS_STEP  # 360
    seg_len = pts_per_epoch // 3                    # 120

    all_rows = []
    for posture in ("index", "thumb"):
        target = "Index" if posture == "index" else "Thumb"
        for csv_path in sorted(emg_dir.glob(f"flex_{posture}_finger_*.csv")):
            df = pd.read_csv(csv_path)
            # 不同 subject 的 EMG channel 欄位命名不一致（ch0/1/2 vs ch4/5/6 等，
            # 硬體通道編號不同但都是 3 通道 EMG），一律取前 3 欄。
            raw_2d = df.iloc[:, :3].values
            rms, num_epochs = preprocess(raw_2d)

            # Per-file z-score 正規化
            mu, sigma = rms.mean(axis=0), rms.std(axis=0)
            rms_norm = (rms - mu) / (sigma + 1e-8)

            for e in range(num_epochs):
                epoch = rms_norm[e * pts_per_epoch: (e + 1) * pts_per_epoch]
                if len(epoch) < pts_per_epoch:
                    continue

                segments = {
                    "Rest": epoch[:seg_len],
                    f"{target} Contract": epoch[seg_len:2 * seg_len],
                    f"{target} Release": epoch[2 * seg_len:3 * seg_len],
                }
                for expected, seg in segments.items():
                    inp = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits, _, _ = model(inp)
                        pred_idx = int(logits.argmax(1).item())
                        conf = float(torch.softmax(logits, 1)[0, pred_idx].item())

                    pred_label = PRED_MAP.get(pred_idx, f"unknown_{pred_idx}")
                    all_rows.append({
                        "subject": subject_dir.name,
                        "file": csv_path.stem,
                        "posture": posture,
                        "epoch": e,
                        "expected": expected,
                        "predicted": pred_label,
                        "confidence": conf,
                        "correct": pred_label == expected,
                    })

    return pd.DataFrame(all_rows)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(MODEL_PATH)
    model.to(device)
    print(f"Model: {ckpt['model_name']}, output_dim={ckpt['model_args']['output_dim']}, device={device}\n")

    # 測試所有 subject
    subjects = sorted(DATA_ROOT.iterdir())
    all_results = []

    for subj_dir in subjects:
        if not subj_dir.is_dir():
            continue
        result = evaluate_subject(model, subj_dir, device)
        if result.empty:
            continue
        acc = result["correct"].mean()
        n = len(result)
        is_test = "← LOSO TEST" if subj_dir.name == "subject_8" else ""
        print(f"  {subj_dir.name}: {acc:.4f} ({n} segments) {is_test}")
        all_results.append(result)

    all_df = pd.concat(all_results, ignore_index=True)

    # 分 train vs test 統計
    test_df = all_df[all_df["subject"] == "subject_8"]
    train_df = all_df[all_df["subject"] != "subject_8"]

    print(f"\n{'='*50}")
    print(f"Train subjects (0-7, 9-16): {train_df['correct'].mean():.4f} ({len(train_df)} segments)")
    print(f"Test subject (subject_8):   {test_df['correct'].mean():.4f} ({len(test_df)} segments)")
    print(f"All subjects:               {all_df['correct'].mean():.4f} ({len(all_df)} segments)")

    if not test_df.empty:
        print(f"\n--- subject_8 per-class accuracy ---")
        for cls in sorted(test_df["expected"].unique()):
            sub = test_df[test_df["expected"] == cls]
            print(f"  {cls:>20s}: {sub['correct'].mean():.4f} ({len(sub)})")

        print(f"\n--- subject_8 confusion ---")
        conf = test_df.groupby(["expected", "predicted"]).size().reset_index(name="n")
        for _, r in conf.sort_values("n", ascending=False).head(10).iterrows():
            m = "✓" if r["expected"] == r["predicted"] else "✗"
            print(f"  {m} {r['expected']:>20s} → {r['predicted']:<20s}  n={r['n']}")

    all_df.to_csv(RESULTS_ROOT / "full_reproduction_all_subjects.csv", index=False)
    print(f"\nSaved to results/full_reproduction_all_subjects.csv")


if __name__ == "__main__":
    main()
