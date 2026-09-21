"""
批次前處理 driver。

掃描 data/<subject>/*.csv（subject = gary / neil1 / CBW1，扁平存放，檔名本身
編碼 posture+subject+timestamp[+motion 編號]），對 ch1-3 各自跑
pipeline.process_channel()，輸出到 data/processed/<subject>/<file_stem>/，
同時把每個 channel 的品質特徵彙整進 results/quality_report.csv。

執行方式：
    conda activate semg
    python3 src/semg/preprocess/run_pipeline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")   # src/ 目錄，與檔案放在多深無關
sys.path.insert(0, str(_SRC))
from semg.core import pipeline as pp
from semg.core import quality as qm

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "src").parent
DATA_ROOT = REPO_ROOT / "data"
PROCESSED_ROOT = DATA_ROOT / "processed"
RESULTS_ROOT = REPO_ROOT / "results"

SUBJECT_DIRS = ["gary", "neil1", "CBW1"]
CHANNELS = (1, 2, 3)
CHANNEL_LABELS = {1: "wrist_flexor", 2: "index", 3: "thumb"}
POSTURES = ["grasp", "pinch", "thumb", "index"]

# 見 pipeline.py 檔頭說明：固定保守裁切，不是逐檔手動調校的版本。
TRIM_START_SEC = 2.0
TRIM_END_SEC = 1.0

PLOT_MAX_POINTS = 60_000  # 畫圖時把原始/濾波後訊號降採樣到這個點數以內，避免圖檔過重


def detect_posture(file_stem: str) -> str:
    for p in POSTURES:
        if file_stem.startswith(p):
            return p
    return "unknown"


def decimate_for_plot(t: np.ndarray, x: np.ndarray, max_points: int):
    if len(x) <= max_points:
        return t, x
    stride = int(np.ceil(len(x) / max_points))
    return t[::stride], x[::stride]


def make_plot(png_path: Path, file_stem: str, ch_tag: str, res: dict):
    t_plot, raw_plot = decimate_for_plot(res["t"], res["raw_mv"], PLOT_MAX_POINTS)
    _, filt_plot = decimate_for_plot(res["t"], res["filtered_mv"], PLOT_MAX_POINTS)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))

    ax1.plot(t_plot, raw_plot, color="#6FB2FF", lw=0.5, alpha=0.5, label="raw (DC removed)")
    ax1.plot(t_plot, filt_plot, color="#FF7A1A", lw=0.6, label="filtered")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("mV")
    ax1.set_title(f"{file_stem} [{ch_tag}] time domain")
    ax1.legend(loc="upper right", fontsize=8)

    ax2.plot(res["env_time"], res["env_mv"], color="#FFD933", lw=1.2)
    ax2.fill_between(res["env_time"], res["env_mv"], color="#FFD933", alpha=0.25)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("RMS envelope (mV)")
    ax2.set_title("RMS envelope (250ms win / 25ms step)")

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def process_file(csv_path: Path, subject: str) -> list[dict]:
    file_stem = csv_path.stem
    posture = detect_posture(file_stem)
    out_dir = PROCESSED_ROOT / subject / file_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pp.load_channel(str(csv_path), channels=CHANNELS)

    rows = []
    channel_log = {}
    for ch in CHANNELS:
        ch_tag = f"ch{ch}"
        res = pp.process_channel(df[ch_tag].values, trim_start_s=TRIM_START_SEC,
                                  trim_end_s=TRIM_END_SEC)
        feats = qm.compute_channel_features(res, pp.FS)

        filt_csv = out_dir / f"{ch_tag}_filtered.csv"
        pd.DataFrame({"timestamp_s": res["t"], f"{ch_tag}_filtered_mv": res["filtered_mv"]}
                     ).to_csv(filt_csv, index=False)

        rms_csv = out_dir / f"{ch_tag}_rms_envelope.csv"
        pd.DataFrame({"timestamp_s": res["env_time"], f"{ch_tag}_rms_mv": res["env_mv"]}
                     ).to_csv(rms_csv, index=False)

        png_path = out_dir / f"{ch_tag}_analysis.png"
        make_plot(png_path, file_stem, ch_tag, res)

        channel_log[ch_tag] = {"muscle": CHANNEL_LABELS[ch], **feats}

        rows.append({
            "subject": subject,
            "file_stem": file_stem,
            "posture": posture,
            "channel": ch_tag,
            "muscle": CHANNEL_LABELS[ch],
            **feats,
        })

    log = {
        "source_csv": str(csv_path.relative_to(REPO_ROOT)),
        "fs_hz": pp.FS,
        "trim_start_sec": TRIM_START_SEC,
        "trim_end_sec": TRIM_END_SEC,
        "notch_hz": pp.NOTCH_FREQ,
        "bandpass_hz": [pp.BP_LOW, pp.BP_HIGH],
        "hampel_half_window_samples": pp.HAMPEL_HALF_WINDOW,
        "rms_win_sec": pp.RMS_WIN_SEC,
        "rms_step_sec": pp.RMS_STEP_SEC,
        "channels": channel_log,
    }
    with open(out_dir / "preprocessing_log.json", "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    return rows


def main():
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for subject in SUBJECT_DIRS:
        subj_dir = DATA_ROOT / subject
        if not subj_dir.is_dir():
            continue
        csv_files = sorted(subj_dir.glob("*.csv"))
        for csv_path in csv_files:
            print(f"[處理] {subject}/{csv_path.name}")
            rows = process_file(csv_path, subject)
            all_rows.extend(rows)

    report_df = pd.DataFrame(all_rows)
    report_df = qm.score_and_classify(report_df, channel_col="channel")
    report_df = report_df.sort_values("quality_score").reset_index(drop=True)

    report_csv = RESULTS_ROOT / "quality_report.csv"
    report_df.to_csv(report_csv, index=False)
    print(f"\n[完成] 品質報告已輸出：{report_csv}")
    print(f"[統計] 共處理 {report_df[['subject','file_stem']].drop_duplicates().shape[0]} 個檔案，"
          f"{len(report_df)} 個 channel-recording")
    print(f"[統計] label 分布：\n{report_df['label'].value_counts()}")


if __name__ == "__main__":
    main()
