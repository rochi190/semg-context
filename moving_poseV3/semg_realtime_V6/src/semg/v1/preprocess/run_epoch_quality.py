"""
批次跑 epoch-level peak-to-peak/MAD 品質控管（見 epoch_quality.py 的方法說明），
輸出 results/epoch_quality_report.csv（每個 epoch 一列）跟
results/epoch_quality_summary.md（每個檔案的壞 epoch 比例彙整）。

執行方式：
    python3 src/semg/preprocess/run_epoch_quality.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")   # src/ 目錄，與檔案放在多深無關
sys.path.insert(0, str(_SRC))
from semg.core import pipeline as pp
from semg.preprocess import epoch_quality as eq
from semg.preprocess.run_pipeline import detect_posture, CHANNEL_LABELS

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "src").parent
DATA_ROOT = REPO_ROOT / "data"
RESULTS_ROOT = REPO_ROOT / "results"

SUBJECT_DIRS = ["gary", "neil1", "CBW1"]
CHANNELS = (1, 2, 3)
TRIM_START_SEC = 2.0
TRIM_END_SEC = 1.0

EPOCH_SEC_BY_POSTURE = {"index": 9.0, "thumb": 9.0, "pinch": 21.0, "grasp": 21.0}


def main():
    all_epoch_rows = []

    for subject in SUBJECT_DIRS:
        subj_dir = DATA_ROOT / subject
        if not subj_dir.is_dir():
            continue

        # 依 posture 分組（對齊 reject_routine：同一個 subject+finger 的所有 epoch 一起算門檻）
        files_by_posture: dict[str, list[Path]] = {}
        for csv_path in sorted(subj_dir.glob("*.csv")):
            posture = detect_posture(csv_path.stem)
            files_by_posture.setdefault(posture, []).append(csv_path)

        for posture, csv_paths in files_by_posture.items():
            epoch_sec = EPOCH_SEC_BY_POSTURE[posture]
            print(f"[處理] {subject}/{posture}（{len(csv_paths)} 個檔案，epoch={epoch_sec}s）")

            # 蒐集這個 (subject, posture) 群組裡所有檔案、所有 channel 的 epoch ptp
            group_ptp = []          # list of (n_epochs, 3) arrays, one per file
            group_file_info = []    # (csv_path, n_epochs)

            for csv_path in csv_paths:
                df = pp.load_channel(str(csv_path), channels=CHANNELS)
                ch_epochs = []
                for ch in CHANNELS:
                    env_time, env = eq.process_channel_for_epochs(
                        df[f"ch{ch}"].values, TRIM_START_SEC, TRIM_END_SEC)
                    epochs = eq.epoch_env(env_time, env, epoch_sec)  # (n_epochs, samples_per_epoch)
                    ch_epochs.append(epochs)

                n_epochs = min(e.shape[0] for e in ch_epochs)
                if n_epochs == 0:
                    continue
                ptp_per_ch = np.stack(
                    [ch_epochs[i][:n_epochs].max(axis=1) - ch_epochs[i][:n_epochs].min(axis=1)
                     for i in range(3)], axis=1)  # (n_epochs, 3)
                group_ptp.append(ptp_per_ch)
                group_file_info.append((csv_path, n_epochs))

            if not group_ptp:
                continue

            pooled_ptp = np.concatenate(group_ptp, axis=0)
            bad_mask, threshold, med, mad = eq.detect_bad_epochs_ptp(pooled_ptp)

            # 把 pooled 結果切回各檔案
            offset = 0
            for csv_path, n_epochs in group_file_info:
                file_bad_mask = bad_mask[offset: offset + n_epochs]
                file_ptp = pooled_ptp[offset: offset + n_epochs]
                offset += n_epochs
                for epoch_idx in range(n_epochs):
                    all_epoch_rows.append({
                        "subject": subject,
                        "file_stem": csv_path.stem,
                        "posture": posture,
                        "epoch_idx": epoch_idx,
                        "epoch_sec": epoch_sec,
                        "ch1_ptp": file_ptp[epoch_idx, 0],
                        "ch2_ptp": file_ptp[epoch_idx, 1],
                        "ch3_ptp": file_ptp[epoch_idx, 2],
                        "ch1_threshold": threshold[0],
                        "ch2_threshold": threshold[1],
                        "ch3_threshold": threshold[2],
                        "bad_epoch": bool(file_bad_mask[epoch_idx]),
                    })

    epoch_df = pd.DataFrame(all_epoch_rows)
    epoch_csv = RESULTS_ROOT / "epoch_quality_report.csv"
    epoch_df.to_csv(epoch_csv, index=False)
    print(f"\n[完成] {epoch_csv}  共 {len(epoch_df)} 個 epoch")

    within_vs_cross = build_within_vs_cross(epoch_df)
    within_vs_cross_csv = RESULTS_ROOT / "epoch_quality_within_vs_cross.csv"
    within_vs_cross.to_csv(within_vs_cross_csv, index=False)
    print(f"[完成] {within_vs_cross_csv}")
    print(within_vs_cross.to_string(index=False))

    build_summary_md(within_vs_cross)


def build_within_vs_cross(epoch_df: pd.DataFrame) -> pd.DataFrame:
    """
    對每個檔案，額外用「只跟自己比」（within_file_only）算一次 median+6*MAD 門檻，
    跟原本「跟同 posture 其他天 session 一起比」（cross_session_group）對照。

    兩者差很多的檔案，代表問題是「跟其他天比整體振幅偏移」（電極位置造成），不是
    這個 session 本身雜訊多；兩者都高的檔案，才是真的連自己內部都不穩定。
    """
    rows = []
    for (subject, file_stem), g in epoch_df.groupby(["subject", "file_stem"]):
        posture = g["posture"].iloc[0]
        bad_within = np.zeros(len(g), dtype=bool)
        for ch in ["ch1_ptp", "ch2_ptp", "ch3_ptp"]:
            ptp = g[ch].values
            _, threshold, _, _ = eq.detect_bad_epochs_ptp(ptp.reshape(-1, 1))
            bad_within |= (ptp > threshold[0])
        rows.append({
            "subject": subject, "file_stem": file_stem, "posture": posture,
            "n_epochs": len(g),
            "bad_epoch_frac_cross_session_group": g["bad_epoch"].mean(),
            "bad_epoch_frac_within_file_only": bad_within.mean(),
        })
    return pd.DataFrame(rows).sort_values("bad_epoch_frac_cross_session_group", ascending=False)


def build_summary_md(within_vs_cross: pd.DataFrame):
    lines = ["# Epoch-level 品質控管報告（peak-to-peak / MAD，復刻 SoftPINCH RejectBadEpochs）\n",
             "**方法**：epoch 邊界用資料自己抓的相位錨點對齊（依已知週期折疊平均出代表性單"
             "週期波形，取最安靜 plateau 的起點），不是固定裁切秒數——這樣才不受個別檔案"
             "開頭空閒時間長短不一影響。每個 epoch、每個 channel 算 RMS envelope 的"
             "peak-to-peak，跟前面 periodicity_strength 是獨立判準，建議兩者一起參考。\n",
             "\n**兩種門檻對照**：\n",
             "- `cross_session_group`：同 (subject, posture) 底下所有天的 session pooled 在"
             "一起算 median+6×MAD（對齊 REJECT_CONFIG_DICT，tolerance=6, bad_ch_acceptance=0）"
             "——這是 SoftPINCH RejectBadEpochs 的原始做法。\n",
             "- `within_file_only`：只用該檔案自己的 epoch 算門檻，不跟其他天比較，純粹看"
             "這個 session 內部一致不一致。\n",
             "\n兩者差很大的檔案，代表問題是跨日整體振幅偏移（電極位置造成），不是這個"
             "session 本身雜訊多；兩者都高，才是真的連自己內部都不穩定。\n"]

    lines.append("| subject | file | posture | epoch數 | cross_session壞比例 | within_file壞比例 | 判讀 |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in within_vs_cross.iterrows():
        cross, within = r["bad_epoch_frac_cross_session_group"], r["bad_epoch_frac_within_file_only"]
        if cross > 0.5 and within < 0.1:
            verdict = "⚠️ 疑似跨日振幅漂移（session本身穩定）"
        elif within > 0.15:
            verdict = "🔴 session內部真的不穩定"
        elif cross > 0.15:
            verdict = "🟡 輕微偏移，留意"
        else:
            verdict = "🟢 OK"
        lines.append(f"| {r['subject']} | {r['file_stem']} | {r['posture']} | {r['n_epochs']} | "
                     f"{cross*100:.1f}% | {within*100:.1f}% | {verdict} |")

    out_md = RESULTS_ROOT / "epoch_quality_summary.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[完成] {out_md}")


if __name__ == "__main__":
    main()
