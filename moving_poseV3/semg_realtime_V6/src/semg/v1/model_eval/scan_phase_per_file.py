"""
診斷用：對 predict_and_plot_epochs.py 這 4 個檔案，各自掃描最佳 PHASE_SHIFT_FRACTION，
不套用 run_model_validation.py 裡那個從 SoftPINCH 自己資料校出來的全域常數 0.06。

動機：neil1 index 那個檔案用全域 0.06 只有 6.1%，但把切法整格往後移一段
（segment_type 循環錯位）後準確率變 90.9%——證實不是模型不準，是這個檔案的
「Rest 起點錨點」本身就抓錯位置。我們自己的錄製沒有音效節拍器，每次自己抓
節奏，時間結構跟 SoftPINCH 有固定 cue 的協定不一樣，同一個全域相位校正值沒辦法
穩定套用在每個檔案上，必須每個檔案各自找。

**重要警告，report 裡務必附上**：這裡是拿同一份資料自己找最好的切法、自己拿來
評分——不是像 calibrate_offline_phase.py 那樣「校準集/保留集」分開驗證，所以
這裡算出來的準確率是「這個檔案在切法调到最好情況下能到多高」的診斷/樂觀上界，
不是可以直接拿來宣稱的公平準確率。用途是回答「這個檔案的低準確率主要是切法問題
還是真的模型辨識不出來」，不是拿來取代 our_data_segment_accuracy.csv 的正式數字。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")   # src/ 目錄，與檔案放在多深無關
sys.path.insert(0, str(_SRC))
from semg.core import pipeline as pp
from semg.core import quality as qm
from semg.core.softpinch_model import load_model
from semg.model_eval.run_model_validation import (
    MODEL_PATH, FS, RMS_STEP, CH_FOR_TARGET, POSTURE_TO_TARGET, preprocess_for_model,
)
from semg.model_eval.predict_and_plot_epochs import (
    TARGET_FILES, evaluate_epoch_segments_with_time, plot_file, plot_zoom, FIG_DIR, RESULTS_DIR,
)

SHIFT_GRID = np.arange(0.0, 1.0, 0.02)


def scan_best_shift(env_target_ch, env_time, env, mu, sigma, target, model):
    strength, period_s = qm.periodicity_strength(env_target_ch, env_time,
                                                   min_period_s=6.0, max_period_s=15.0)
    if strength <= 0 or period_s <= 0:
        period_s = 9.0
    fs_env = 1.0 / np.median(np.diff(env_time))
    period_samples = int(round(period_s * fs_env))
    raw_anchor_idx = qm.find_cycle_anchor_index(env_target_ch, env_time, period_s)

    best_shift, best_acc, best_seg_df = None, -1.0, None
    for shift in SHIFT_GRID:
        anchor_idx = (raw_anchor_idx + int(round(shift * period_samples))) % period_samples
        seg_df = evaluate_epoch_segments_with_time(env, mu, sigma, anchor_idx, period_samples,
                                                     target, model)
        acc = seg_df["correct"].mean()
        if acc > best_acc:
            best_acc, best_shift, best_seg_df = acc, shift, seg_df
    return best_shift, best_acc, best_seg_df, period_s


def main():
    model, ckpt = load_model(MODEL_PATH / "model.pth")
    print(f"model loaded: {ckpt['model_name']} / {ckpt['sensor_name']}")

    rows = []
    for subject, csv_path, posture in TARGET_FILES:
        print(f"[per-file 相位掃描] {subject}/{csv_path.name}")
        df = pp.load_channel(str(csv_path), channels=(1, 2, 3))
        raw = {ch: df[f"ch{ch}"].values for ch in (1, 2, 3)}
        env = preprocess_for_model(raw)
        mu, sigma = env.mean(axis=0), env.std(axis=0)
        target = POSTURE_TO_TARGET[posture]
        env_time = np.arange(len(env)) * (RMS_STEP / FS)

        best_shift, best_acc, best_seg_df, period_s = scan_best_shift(
            env[:, CH_FOR_TARGET[target]], env_time, env, mu, sigma, target, model)

        best_seg_df["subject"] = subject
        best_seg_df["file_stem"] = csv_path.stem
        best_seg_df["posture"] = posture

        out_png = FIG_DIR / f"{subject}_{csv_path.stem}_realigned.png"
        plot_file(subject, csv_path.stem + " [per-file re-aligned]", posture, env, env_time,
                   best_seg_df, out_png)
        out_zoom_png = FIG_DIR / f"{subject}_{csv_path.stem}_realigned_zoom.png"
        plot_zoom(subject, csv_path.stem + " [per-file re-aligned]", posture, env, env_time,
                   best_seg_df, out_zoom_png)

        rows.append({
            "subject": subject, "file_stem": csv_path.stem, "posture": posture,
            "period_s": period_s,
            "global_shift_0.06_accuracy": None,  # filled below from existing summary
            "best_per_file_shift": best_shift,
            "best_per_file_accuracy": best_acc,
        })
        print(f"    best_shift={best_shift:.2f}  best_acc={best_acc:.1%}  -> {out_png.name}")

    summary = pd.DataFrame(rows)
    orig = pd.read_csv(RESULTS_DIR / "two_good_days_accuracy_summary.csv")
    summary = summary.drop(columns=["global_shift_0.06_accuracy"]).merge(
        orig[["subject", "file_stem", "segment_accuracy"]].rename(
            columns={"segment_accuracy": "global_shift_0.06_accuracy"}),
        on=["subject", "file_stem"])
    summary = summary[["subject", "file_stem", "posture", "period_s",
                        "global_shift_0.06_accuracy", "best_per_file_shift", "best_per_file_accuracy"]]

    out_csv = RESULTS_DIR / "two_good_days_phase_scan_diagnostic.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\n[完成] {out_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
