"""
用已經跑好的 data/processed/*/*/chN_filtered.csv + chN_rms_envelope.csv +
preprocessing_log.json（含 hampel_outlier_frac）重新計算品質特徵，不用重跑整條
notch/bandpass/hampel/RMS pipeline，只在調整 quality.py 的特徵定義時用。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")   # src/ 目錄，與檔案放在多深無關
sys.path.insert(0, str(_SRC))
from semg.core import pipeline as pp
from semg.core import quality as qm
from semg.preprocess.run_pipeline import CHANNEL_LABELS, POSTURES, detect_posture

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "src").parent
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
RESULTS_ROOT = REPO_ROOT / "results"


def main():
    rows = []
    for subject_dir in sorted(PROCESSED_ROOT.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject = subject_dir.name
        for file_dir in sorted(subject_dir.iterdir()):
            if not file_dir.is_dir():
                continue
            file_stem = file_dir.name
            posture = detect_posture(file_stem)
            log = json.loads((file_dir / "preprocessing_log.json").read_text())

            for ch in (1, 2, 3):
                ch_tag = f"ch{ch}"
                filt = pd.read_csv(file_dir / f"{ch_tag}_filtered.csv")
                env = pd.read_csv(file_dir / f"{ch_tag}_rms_envelope.csv")
                hampel_frac = log["channels"][ch_tag]["hampel_outlier_frac"]

                res = {
                    "filtered_mv": filt[f"{ch_tag}_filtered_mv"].values,
                    "env_mv": env[f"{ch_tag}_rms_mv"].values,
                    "env_time": env["timestamp_s"].values,
                    "hampel_outlier_mask": np.zeros(1, dtype=bool),  # placeholder, overridden below
                }
                feats = qm.compute_channel_features(res, pp.FS)
                feats["hampel_outlier_frac"] = hampel_frac  # 用 log 裡準確值覆蓋 placeholder

                rows.append({
                    "subject": subject,
                    "file_stem": file_stem,
                    "posture": posture,
                    "channel": ch_tag,
                    "muscle": CHANNEL_LABELS[ch],
                    **feats,
                })

    df = pd.DataFrame(rows)
    df = qm.score_and_classify(df, channel_col="channel")
    df = df.sort_values("quality_score").reset_index(drop=True)

    report_csv = RESULTS_ROOT / "quality_report.csv"
    df.to_csv(report_csv, index=False)
    print(f"[完成] 重新計算並輸出：{report_csv}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
