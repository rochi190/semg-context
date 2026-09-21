"""
拿 subject_0 自己的 EMG 資料，餵進我自己重建的前處理 pipeline
（run_model_validation.preprocess_for_model，只是 notch 改回 50Hz 因為這是
他們自己的丹麥電網資料）+ 正確的 offline LOSO checkpoint（all_subjects,
5-class），看能不能重現接近論文報告的準確率。

目的：控制變數測試——如果連「他們自己的資料 + 我重建的前處理」都測不出高準確率，
代表問題出在我的前處理/pipeline 實作本身，不是我們資料的 domain shift；
如果測出來準確率明顯比我們自己的資料高很多，代表 pipeline 本身沒問題，
之前的低準確率確實是硬體 domain shift，不是我做錯前處理。

修正紀錄（2026-07-20）：早期版本用錯 checkpoint（Real_time_inference/subject_0,
7-class），只測到 66.5%，被誤判為「repo 公開不完整」。換成正確的 all_subjects
offline checkpoint、重新校準相位偏移（見 calibrate_offline_phase.py）後，
subject_0（校準集）100%、subject_1（保留集）88.8%——證實 pipeline 重建本身
沒問題，先前的落差是評估碼本身的 bug，不是 domain shift 或 repo 缺碼。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")   # src/ 目錄，與檔案放在多深無關
sys.path.insert(0, str(_SRC))
from semg.core import pipeline as pp
from semg.core.softpinch_model import load_model
from semg.model_eval.run_model_validation import (
    MODEL_PATH, RESULTS_ROOT, RMS_STEP, FS,
    causal_notch_bandpass, causal_hampel, rms_stream,
    HAMPEL_HALF_WINDOW, HAMPEL_SIGMA, RMS_WINDOW, BANDPASS,
    evaluate_epoch_segments, find_calibrated_anchor, CH_FOR_TARGET,
)

SOFTPINCH_NOTCH_FREQ = 50  # 他們自己的丹麥電網資料，不是我們的 60Hz

# 論文協定的真實裁切秒數（TRIM_PERIOD=3），不是我們自己資料用的 2.0/1.0——
# 那組值是我為了我們自己的資料調的，套用在他們的資料上是錯的，會讓 epoch
# 邊界跟真實 trial 邊界對不齊。
SOFTPINCH_TRIM_SEC = 3.0

SOFTPINCH_DATA_ROOT = (next(p for p in Path(__file__).resolve().parents if p.name == "src").parent /
                       "src/SoftPINCH/src/experiment/data/subject_independent/subject_0/EMG")

POSTURE_TO_TARGET = {"index": "Index", "thumb": "Thumb"}


def preprocess_softpinch_own_data(raw_by_ch: dict) -> np.ndarray:
    """跟 run_model_validation.preprocess_for_model 幾乎一樣，只差 notch 頻率、
    裁切秒數，跟不用 adc_to_mv（他們的 CSV 已經是物理單位，不是 ADC counts）。"""
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


def main():
    model, ckpt = load_model(MODEL_PATH / "model.pth")
    print(f"model loaded: {ckpt['model_name']} / {ckpt['sensor_name']}")
    print("[正規化] offline checkpoint 沒有存 mu.npy/sigma.npy（那是 real-time "
          "checkpoint 才有的東西），改用 per-file 自我正規化——這也跟我們自己資料"
          "的測法一致，兩邊才能公平比較。")

    files = []
    for posture in ("index", "thumb"):
        for csv_path in sorted(SOFTPINCH_DATA_ROOT.glob(f"flex_{posture}_finger_*.csv")):
            files.append((csv_path, posture))

    all_segment_rows = []
    file_rows = []
    for csv_path, posture in files:
        print(f"[跑模型] subject_0/{csv_path.name}")
        df = pd.read_csv(csv_path)
        raw = {ch: df[f"ch{ch}"].values for ch in (0, 1, 2)}
        env = preprocess_softpinch_own_data(raw)
        mu, sigma = env.mean(axis=0), env.std(axis=0)
        target = POSTURE_TO_TARGET[posture]

        env_time = np.arange(len(env)) * (RMS_STEP / FS)
        anchor_idx, period_samples, period_s = find_calibrated_anchor(
            env[:, CH_FOR_TARGET[target]], env_time)

        seg_df = evaluate_epoch_segments(env, mu, sigma, anchor_idx, period_samples, target, model)
        seg_df["file_stem"] = csv_path.stem
        seg_df["posture"] = posture
        all_segment_rows.append(seg_df)

        file_rows.append({
            "file_stem": csv_path.stem,
            "posture": posture,
            "detected_period_s": period_s,
            "n_epochs": seg_df["epoch_idx"].nunique(),
            "segment_accuracy": seg_df["correct"].mean(),
            "mean_confidence": seg_df["confidence"].mean(),
        })

    seg_all = pd.concat(all_segment_rows, ignore_index=True)
    seg_all.to_csv(RESULTS_ROOT / "sanity_check_subject0_segments.csv", index=False)

    report = pd.DataFrame(file_rows).sort_values("segment_accuracy")
    out_csv = RESULTS_ROOT / "sanity_check_subject0_own_data.csv"
    report.to_csv(out_csv, index=False)
    print(f"\n[完成] {out_csv}")
    print(report.to_string(index=False))
    print(f"\n整體 segment accuracy（全部檔案 pooled）: {seg_all['correct'].mean():.3f}")
    print("\n混淆矩陣風格分布（expected -> predicted 前幾名）：")
    print(seg_all.groupby(['expected', 'predicted']).size().sort_values(ascending=False).head(15))


if __name__ == "__main__":
    main()
