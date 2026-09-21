"""
重新校準相位偏移量，改用正確的 offline LOSO model（5-class，all_subjects），
不是舊版誤用的 real-time 7-class model。

背景：run_model_validation.py 裡的 PHASE_SHIFT_FRACTION=0.23 是拿「錯的
checkpoint（real-time 7-class）」校準出來的，那個模型本身的評估邏輯/訓練方式
都跟論文 99.4% 的來源（offline 5-class all_subjects model）不同，換了正確的
checkpoint 後這個校準值不一定還適用，必須重新掃描。

跟 full_reproduction.py 的差異：full_reproduction.py 用論文протокол已知的精確
3秒 trim 邊界（filtfilt離線前處理），這裡刻意改用「不知道精確邊界，靠
autocorrelation 找 plateau 起點」的方式（causal 前處理，模擬我們自己資料的
真實情境——我們的錄製沒有精確的 trigger 時間戳），目的是量化「在邊界只能用
訊號自己估計、不是精確已知」的前提下，換成正確模型後最高能拿到多少準確率。
這個數字才是拿去跟我們自己資料的結果做對照的公平基準。

用 subject_0（校準集）掃描最佳 PHASE_SHIFT_FRACTION，subject_1（保留集）驗證
不是過擬合單一檔案。
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
from semg.model_eval.run_model_validation import (
    FS, BANDPASS, HAMPEL_HALF_WINDOW, HAMPEL_SIGMA, RMS_WINDOW, RMS_STEP,
    causal_notch_bandpass, causal_hampel, rms_stream,
)

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "src").parent
RESULTS_ROOT = REPO_ROOT / "results"
OFFLINE_MODEL_PATH = (REPO_ROOT / "src/SoftPINCH/src/models/loggings/"
                       "SingleNet_CNN+LSTM_EMG/all_subjects/model.pth")
SOFTPINCH_DATA_ROOT = REPO_ROOT / "src/SoftPINCH/src/experiment/data/subject_independent"

SOFTPINCH_NOTCH_FREQ = 50
SOFTPINCH_TRIM_SEC = 3.0  # 前處理仍先粗略裁掉頭尾，只是不假設剛好對齊 trial 邊界

PRED_MAP_5CLASS = {
    0: "Index Contract", 1: "Index Release",
    2: "Thumb Contract", 3: "Thumb Release",
    4: "Rest",
}
CH_FOR_TARGET = {"Index": 1, "Thumb": 2}
POSTURE_TO_TARGET = {"index": "Index", "thumb": "Thumb"}


def preprocess_causal(raw_by_ch: dict) -> np.ndarray:
    envs = []
    for ch in (0, 1, 2):
        x = raw_by_ch[ch].astype(np.float64)
        x = pp.trim(x, FS, SOFTPINCH_TRIM_SEC, SOFTPINCH_TRIM_SEC)
        x = x - x.mean()
        filt = causal_notch_bandpass(x, FS, SOFTPINCH_NOTCH_FREQ, BANDPASS)
        hampel = causal_hampel(filt, HAMPEL_HALF_WINDOW, HAMPEL_SIGMA)
        env = rms_stream(hampel, RMS_WINDOW, RMS_STEP)
        envs.append(env)
    min_len = min(len(e) for e in envs)
    return np.stack([e[:min_len] for e in envs], axis=1)


def evaluate_segments_5class(env, mu, sigma, anchor_idx, period_samples, target, model):
    env_norm = (env - mu) / (sigma + 1e-8)
    n_epochs = (len(env_norm) - anchor_idx) // period_samples
    seg_len = period_samples // 3
    rows = []
    for e in range(n_epochs):
        epoch = env_norm[anchor_idx + e * period_samples: anchor_idx + (e + 1) * period_samples]
        segments = {
            "Rest": epoch[0:seg_len],
            f"{target} Contract": epoch[seg_len:2 * seg_len],
            f"{target} Release": epoch[2 * seg_len:3 * seg_len],
        }
        for expected_label, seg in segments.items():
            inp = torch.tensor(seg, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _, _ = model(inp)
                pred_idx = int(torch.argmax(logits, dim=1).item())
            pred_label = PRED_MAP_5CLASS[pred_idx]
            rows.append({"epoch_idx": e, "expected": expected_label,
                         "predicted": pred_label, "correct": pred_label == expected_label})
    return pd.DataFrame(rows)


def load_subject_envs(subject_dir: Path):
    """回傳 [(env, env_time, target), ...]，每個檔案各自算一次，不做正規化。"""
    out = []
    for posture in ("index", "thumb"):
        target = POSTURE_TO_TARGET[posture]
        for csv_path in sorted((subject_dir / "EMG").glob(f"flex_{posture}_finger_*.csv")):
            df = pd.read_csv(csv_path)
            raw = {ch: df.iloc[:, ch].values for ch in (0, 1, 2)}
            env = preprocess_causal(raw)
            env_time = np.arange(len(env)) * (RMS_STEP / FS)
            out.append((csv_path.stem, env, env_time, target))
    return out


def scan_shift(model, files, shift_fractions):
    results = {}
    for shift in shift_fractions:
        accs = []
        for stem, env, env_time, target in files:
            strength, period_s = qm.periodicity_strength(
                env[:, CH_FOR_TARGET[target]], env_time, min_period_s=6.0, max_period_s=15.0)
            if strength <= 0 or period_s <= 0:
                period_s = 9.0
            fs_env = 1.0 / np.median(np.diff(env_time))
            period_samples = int(round(period_s * fs_env))
            raw_anchor = qm.find_cycle_anchor_index(env[:, CH_FOR_TARGET[target]], env_time, period_s)
            anchor_idx = (raw_anchor + int(round(shift * period_samples))) % period_samples

            mu, sigma = env.mean(axis=0), env.std(axis=0)
            seg_df = evaluate_segments_5class(env, mu, sigma, anchor_idx, period_samples, target, model)
            if len(seg_df) > 0:
                accs.append(seg_df["correct"].mean())
        results[shift] = float(np.mean(accs)) if accs else 0.0
    return results


def main():
    model, ckpt = load_model(OFFLINE_MODEL_PATH)
    print(f"model: {ckpt['model_name']} output_dim={ckpt['model_args']['output_dim']}")

    calib_files = load_subject_envs(SOFTPINCH_DATA_ROOT / "subject_0")
    holdout_files = load_subject_envs(SOFTPINCH_DATA_ROOT / "subject_1")

    shifts = np.round(np.arange(0.0, 1.0, 0.02), 2)
    print(f"\n[掃描] subject_0 校準（{len(shifts)} 個候選偏移量）...")
    calib_results = scan_shift(model, calib_files, shifts)

    best_shift = max(calib_results, key=calib_results.get)
    print(f"\n最佳偏移量: {best_shift}  (subject_0 accuracy={calib_results[best_shift]:.4f})")

    print(f"\n[驗證] subject_1（保留集，未參與校準）在最佳偏移量下的準確率...")
    holdout_acc = scan_shift(model, holdout_files, [best_shift])[best_shift]
    print(f"subject_1 accuracy @ shift={best_shift}: {holdout_acc:.4f}")

    out = pd.DataFrame({"shift_fraction": list(calib_results.keys()),
                         "subject_0_accuracy": list(calib_results.values())})
    out.to_csv(RESULTS_ROOT / "offline_phase_calibration_scan.csv", index=False)
    print(f"\n完整掃描結果已存到 results/offline_phase_calibration_scan.csv")
    print(f"\n總結: best_shift={best_shift}, subject_0={calib_results[best_shift]:.4f}, "
          f"subject_1(holdout)={holdout_acc:.4f}")


if __name__ == "__main__":
    main()
