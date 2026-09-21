"""
讀取 results/quality_report.csv + data/processed/ 底下的 RMS envelope CSV，
畫出每個 subject 一張總覽圖（RMS envelope 小圖 grid，good=綠框 / bad=紅框），
方便一次目視掃過所有 channel-recording，而不用開 72 張個別圖。

執行方式：
    python3 src/semg/_archive/build_report.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "src").parent
DATA_ROOT = REPO_ROOT / "data"
PROCESSED_ROOT = DATA_ROOT / "processed"
RESULTS_ROOT = REPO_ROOT / "results"

COLOR_GOOD = "#3CB371"
COLOR_BAD = "#E05252"


def build_subject_overview(df: pd.DataFrame, subject: str):
    sub = df[df["subject"] == subject].sort_values(["file_stem", "channel"]).reset_index(drop=True)
    n = len(sub)
    if n == 0:
        return
    ncols = 6
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.2))
    axes = axes.reshape(-1)

    for i, row in sub.iterrows():
        ax = axes[i]
        rms_csv = PROCESSED_ROOT / subject / row["file_stem"] / f"{row['channel']}_rms_envelope.csv"
        env = pd.read_csv(rms_csv)
        col = env.columns[1]
        color = COLOR_GOOD if row["label"] == "good" else COLOR_BAD
        ax.plot(env["timestamp_s"], env[col], color=color, lw=0.8)
        ax.fill_between(env["timestamp_s"], env[col], color=color, alpha=0.25)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
        title = f"{row['posture']}/{row['channel']}({row['muscle']})\nscore={row['quality_score']:.2f}"
        ax.set_title(title, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.02, 0.95, row["file_stem"][-15:], transform=ax.transAxes, fontsize=5.5,
                va="top", color="#444444")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"RMS envelope overview — subject: {subject}  "
                 f"(green=good, red=bad, first-pass classification)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = RESULTS_ROOT / f"quality_overview_{subject}.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"[輸出] {out_png}")


def build_summary_md(df: pd.DataFrame):
    lines = ["# sEMG 資料品質初篩報告（v2：絕對門檻分類）\n",
             "分類方法：量測 RMS envelope 去趨勢後的 autocorrelation，在合理任務週期"
             "範圍（2.5-45s，涵蓋 index/thumb 的 9s 週期跟 grasp/pinch 的 21s 週期）內"
             "找真正的局部極大值峰值，當作 `periodicity_strength`（絕對指標，不是跟"
             "同一批資料裡其他樣本比較出來的相對分數）。`periodicity_strength < 0.15` "
             "或 `hampel_outlier_frac > 3%`（脈衝雜訊比例過高）就判 bad，否則判 good。\n",
             "\n**v1（z-score + GMM 組內相對排名）已棄用**：拿 SoftPINCH 論文自己的 EMG "
             "資料驗證後發現 v1 是相對排名，不是絕對品質判斷，不管整批資料實際多好/多爛都會"
             "硬切出約一半 good 一半 bad（dynamic range 21dB、肉眼看是清楚週期性尖峰的通道"
             "被 v1 判 bad；dynamic range 2dB、幾乎是平的通道卻被 v1 判成分數最高的 good）。"
             "v2 改用絕對門檻後，同一個標準套用在所有資料上。\n",
             "\n**這仍是初篩，不是最終答案**，邊界案例請對照"
             " `data/processed/<subject>/<file>/chN_analysis.png` 目視確認。\n",
             "\n**已知限制**：分數是整個 channel 錄音算一個純量，抓不到「大部分時間"
             "訊號很好、只有中間一小段被動作偽影污染」這種局部問題（例如"
             "neil1/pinchlin_20260713_16-00-42/ch2、CBW1/indexcbw_20260716_20-10-57/ch1"
             "都是這種情況：整體被標 good，是因為異常區段以外的訊號真的很乾淨，"
             "但仍有局部異常）。這種情況建議之後切 trial 時手動排除異常時間段，"
             "而不是整個 session 丟掉，可對照對應的 `chN_analysis.png` 確認異常區間。\n"]

    lines.append(f"- 總計 {df[['subject','file_stem']].drop_duplicates().shape[0]} 個檔案，"
                 f"{len(df)} 個 channel-recording\n"
                 f"- good: {(df['label']=='good').sum()}　bad: {(df['label']=='bad').sum()}\n")

    lines.append("\n## 標記為 bad 的 channel-recording（依 quality_score 由差到好排序）\n")
    lines.append("| subject | file | posture | channel(muscle) | dyn.range(dB) | outlier% | noise ratio | score |")
    lines.append("|---|---|---|---|---|---|---|---|")
    bad = df[df["label"] == "bad"].sort_values("quality_score")
    for _, r in bad.iterrows():
        lines.append(f"| {r['subject']} | {r['file_stem']} | {r['posture']} | "
                     f"{r['channel']}({r['muscle']}) | {r['env_dynamic_range_db']:.1f} | "
                     f"{r['hampel_outlier_frac']*100:.2f}% | {r['broadband_noise_ratio']:.2f} | "
                     f"{r['quality_score']:.2f} |")

    lines.append("\n## 檔案層級摘要（該檔案 3 個 channel 中若有任一 bad 則整檔標記需複查）\n")
    file_level = df.groupby(["subject", "file_stem", "posture"]).agg(
        n_bad=("label", lambda s: (s == "bad").sum()),
        min_score=("quality_score", "min"),
    ).reset_index().sort_values("min_score")
    lines.append("| subject | file | posture | bad channel 數 | 最低 score |")
    lines.append("|---|---|---|---|---|")
    for _, r in file_level.iterrows():
        flag = "⚠️ 需複查" if r["n_bad"] > 0 else "OK"
        lines.append(f"| {r['subject']} | {r['file_stem']} | {r['posture']} | "
                     f"{r['n_bad']}/3 {flag} | {r['min_score']:.2f} |")

    out_md = RESULTS_ROOT / "quality_summary.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[輸出] {out_md}")


def main():
    df = pd.read_csv(RESULTS_ROOT / "quality_report.csv")
    for subject in df["subject"].unique():
        build_subject_overview(df, subject)
    build_summary_md(df)


if __name__ == "__main__":
    main()
