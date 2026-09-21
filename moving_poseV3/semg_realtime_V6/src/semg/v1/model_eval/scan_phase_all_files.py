"""
把 scan_phase_per_file.py 驗證過的做法（每個檔案各自掃描最佳 PHASE_SHIFT_FRACTION，
不套用從 SoftPINCH 資料借來的全域常數 0.06）套用到全部 index/thumb 檔案，不只
predict_and_plot_epochs.py 那 4 個範例檔案。

起因：neil1 index 0720 用全域 0.06 只有 6.1%，整格移位後變 90.9%——證實
run_model_validation.py / progress_report_meeting_2026-07-20.md 第五節那個
「39.8% vs 88.8%，49 個百分點乾淨 domain shift」的結論，有一部分（多少不知道，
所以才要全部重跑）其實是邊界對齊的假象，不是真正的硬體 domain shift。這裡重新
量一次「如果每個檔案都用它自己最好的切法」，才能回答「拿掉對齊假象之後，domain
shift 真正剩多少」。

**警告，跟 scan_phase_per_file.py 一樣**：這裡是每個檔案自己找最好切法、自己評分，
是「切法调到最好情況下的樂觀上界」，不是公平的盲測——不能直接拿來取代 88.8% 那種
有校準集/保留集分開驗證的數字。用途是診斷「39.8%這個舊數字裡，多少是切法問題、
修正後剩下多少才是待解的真正落差」。
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
    MODEL_PATH, FS, RMS_STEP, CH_FOR_TARGET, POSTURE_TO_TARGET, preprocess_for_model,
)
from semg.model_eval.scan_phase_per_file import scan_best_shift

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "src").parent
PKG_ROOT = REPO_ROOT
RESULTS_DIR = REPO_ROOT / "results"


def discover_files():
    subject_dirs = {"gary": REPO_ROOT / "data/gary", "neil1": REPO_ROOT / "data/neil1",
                     "CBW1": REPO_ROOT / "data/CBW1"}
    files = []
    for subject, subj_dir in subject_dirs.items():
        for csv_path in sorted(subj_dir.glob("*.csv")):
            posture = None
            for p in ("index", "thumb"):
                if csv_path.stem.startswith(p):
                    posture = p
                    break
            if posture is not None:
                files.append((subject, csv_path, posture))
    return files


def main():
    model, ckpt = load_model(MODEL_PATH / "model.pth")
    print(f"model loaded: {ckpt['model_name']} / {ckpt['sensor_name']}")

    files = discover_files()
    print(f"共 {len(files)} 個 index/thumb 檔案")

    rows = []
    for subject, csv_path, posture in files:
        print(f"[per-file 相位掃描] {subject}/{csv_path.name}")
        df = pp.load_channel(str(csv_path), channels=(1, 2, 3))
        raw = {ch: df[f"ch{ch}"].values for ch in (1, 2, 3)}
        env = preprocess_for_model(raw)
        mu, sigma = env.mean(axis=0), env.std(axis=0)
        target = POSTURE_TO_TARGET[posture]
        env_time = np.arange(len(env)) * (RMS_STEP / FS)

        best_shift, best_acc, best_seg_df, period_s = scan_best_shift(
            env[:, CH_FOR_TARGET[target]], env_time, env, mu, sigma, target, model)

        # 也順便記錄舊的全域 0.06 準確率，方便對照
        from semg.model_eval.run_model_validation import find_calibrated_anchor
        anchor_06, period_samples_06, _ = find_calibrated_anchor(env[:, CH_FOR_TARGET[target]], env_time)
        from semg.model_eval.predict_and_plot_epochs import evaluate_epoch_segments_with_time
        seg_df_06 = evaluate_epoch_segments_with_time(env, mu, sigma, anchor_06, period_samples_06,
                                                        target, model)
        acc_06 = seg_df_06["correct"].mean()

        rows.append({
            "subject": subject, "file_stem": csv_path.stem, "posture": posture,
            "n_segments": len(best_seg_df),
            "global_shift_0.06_accuracy": acc_06,
            "best_per_file_shift": best_shift,
            "best_per_file_accuracy": best_acc,
        })
        print(f"    global(0.06)={acc_06:.1%}   best_shift={best_shift:.2f}  best_acc={best_acc:.1%}")

    summary = pd.DataFrame(rows).sort_values(["subject", "posture", "file_stem"])
    out_csv = RESULTS_DIR / "our_data_realigned_segment_accuracy.csv"
    summary.to_csv(out_csv, index=False)

    print(f"\n[完成] {out_csv}")
    print(summary.to_string(index=False))

    n = summary["n_segments"]
    pooled_06 = (summary["global_shift_0.06_accuracy"] * n).sum() / n.sum()
    pooled_best = (summary["best_per_file_accuracy"] * n).sum() / n.sum()
    print(f"\n[segment-weighted pooled，跟舊報告的 39.8% 算法一致]")
    print(f"舊全域 0.06 pooled 準確率: {pooled_06:.1%}")
    print(f"各檔案自己找最佳切法後 pooled 準確率: {pooled_best:.1%}")
    print(f"\n[簡單逐檔平均，不看 segment 數量]")
    print(f"舊全域 0.06 平均: {summary['global_shift_0.06_accuracy'].mean():.1%}")
    print(f"最佳切法後平均: {summary['best_per_file_accuracy'].mean():.1%}")


if __name__ == "__main__":
    main()
