"""
拿 SoftPINCH 原始 repo 自己的 EMG 資料（src/SoftPINCH/src/experiment/data/subject_independent/）
跑同一條前處理+品質分類 pipeline，作為驗證：如果我們的分類方法在論文自己收的資料上跑出來
好樣本比例明顯偏低/怪異，代表分類方法本身有問題；如果大致都判 good、且跟我們自己資料的
好樣本長得像，代表方法合理，我們自己資料裡的 bad 判斷比較可信。

只跑一部分（3 個 subject，每個 subject 各 2 個檔案）做驗證，不是要復刻整個 experiment/。

跟我們自己資料的差異（已在這裡調整）：
    - CSV 沒有 timestamp 欄位，欄位是 ch0,ch1,ch2（0-indexed），不是 ch1,ch2,ch3
    - 數值已經是物理單位（不是 ADC counts），不需要再乘 LSB_MV
    - Notch 頻率是 50Hz（原始碼 preprocessing.py 第789行 EMG_notch(..., cutoff=50, Q=30)，
      丹麥電網），不是我們自己資料用的 60Hz
    - bandpass(20-450Hz order4)、hampel(window=200,sigma=3)、RMS(500 samples/50 samples
      =250ms/25ms) 三個參數跟我們的 pipeline 完全一樣（對照 classification_pipeline.py
      RMS_SAMPLING_WINDOW=500 / RMS_WINDOW_STEPSIZE=50 兩個常數，不是
      preprocessing_routine() 函式簽名裡容易誤讀的 default 200/50）
    - trim：原始碼用 trim_period=3（每個 trial 9 秒，前後各裁 3 秒），這裡簡化成
      對稱裁切開頭/結尾各 3 秒
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")   # src/ 目錄，與檔案放在多深無關
sys.path.insert(0, str(_SRC))
from semg.core import pipeline as pp
from semg.core import quality as qm
from semg.preprocess.run_pipeline import make_plot

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "src").parent
SOFTPINCH_DATA_ROOT = REPO_ROOT / "src/SoftPINCH/src/experiment/data/subject_independent"
OUT_ROOT = REPO_ROOT / "data/processed_softpinch_validation"
RESULTS_ROOT = REPO_ROOT / "results"

NOTCH_FREQ_SOFTPINCH = 50  # 丹麥電網，對齊 repo 原始碼
TRIM_SEC = 3.0
CHANNELS = (0, 1, 2)
CHANNEL_MUSCLE = {  # 對齊 preprocessing.py 第724-728行 emg_ch_names
    0: "palmaris_longus",
    1: "flexor_digitorum_superficialis",
    2: "flexor_pollicis_longus",
}

VALIDATION_SUBSET = {
    "subject_1": ["flex_index_finger_2026-02-06 10-32-15.csv", "flex_thumb_finger_2026-02-06 10-59-29.csv"],
    "subject_5": ["flex_index_finger_2026-02-12 14-29-22.csv", "flex_thumb_finger_2026-02-12 14-53-24.csv"],
    "subject_10": ["flex_index_finger_2026-02-27 09-30-03.csv", "flex_thumb_finger_2026-02-27 09-55-53.csv"],
}


def process_softpinch_file(csv_path: Path, subject: str) -> list[dict]:
    file_stem = csv_path.stem
    posture = "index" if "index" in file_stem else "thumb"
    out_dir = OUT_ROOT / subject / file_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    rows = []
    for ch in CHANNELS:
        ch_tag = f"ch{ch}"
        raw = df[ch_tag].values.astype(np.float64)  # 已經是物理單位，不需 ADC->mV
        raw_trim = pp.trim(raw, pp.FS, TRIM_SEC, TRIM_SEC)
        t = np.arange(len(raw_trim)) / pp.FS

        dc_offset = raw_trim.mean()
        work = raw_trim - dc_offset
        work = pp.notch_filter(work, pp.FS, freq=NOTCH_FREQ_SOFTPINCH)
        work = pp.bandpass_filter(work, pp.FS)
        work, hampel_mask = pp.hampel_filter(work)
        env_time, env = pp.rms_envelope(work, pp.FS)

        res = {
            "t": t,
            "raw_mv": raw_trim - dc_offset,
            "filtered_mv": work,
            "hampel_outlier_mask": hampel_mask,
            "env_time": env_time,
            "env_mv": env,
        }
        feats = qm.compute_channel_features(res, pp.FS)

        pd.DataFrame({"timestamp_s": t, f"{ch_tag}_filtered": work}).to_csv(
            out_dir / f"{ch_tag}_filtered.csv", index=False)
        pd.DataFrame({"timestamp_s": env_time, f"{ch_tag}_rms_mv": env}).to_csv(
            out_dir / f"{ch_tag}_rms_envelope.csv", index=False)
        make_plot(out_dir / f"{ch_tag}_analysis.png", file_stem, ch_tag, res)

        rows.append({
            "subject": subject,
            "file_stem": file_stem,
            "posture": posture,
            "channel": ch_tag,
            "muscle": CHANNEL_MUSCLE[ch],
            **feats,
        })

    return rows


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for subject, files in VALIDATION_SUBSET.items():
        for fname in files:
            csv_path = SOFTPINCH_DATA_ROOT / subject / "EMG" / fname
            if not csv_path.exists():
                print(f"[跳過] 找不到 {csv_path}")
                continue
            print(f"[處理] {subject}/{fname}")
            all_rows.extend(process_softpinch_file(csv_path, subject))

    df = pd.DataFrame(all_rows)
    df = qm.score_and_classify(df, channel_col="channel")
    df = df.sort_values("quality_score").reset_index(drop=True)

    out_csv = RESULTS_ROOT / "quality_report_softpinch_validation.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[完成] {out_csv}")
    print(f"[統計] {len(df)} 個 channel-recording，label 分布：\n{df['label'].value_counts()}")


if __name__ == "__main__":
    main()
