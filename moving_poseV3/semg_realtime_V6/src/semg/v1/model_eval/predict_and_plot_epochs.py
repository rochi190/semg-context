"""
針對「電極貼得比較好、訊號比較乾淨」的兩天資料——gary 2026-07-16、neil1(lin)
2026-07-20——的 index/thumb 錄音，逐 epoch 丟進正確的 offline checkpoint
（SingleNet_CNN+LSTM_EMG/all_subjects）做分類，並把結果直接畫在 3 通道 RMS
envelope 圖上：每個 epoch 的 rest/contract/release 三段，用背景色塊標出模型
預測的類別，邊框顏色標示對/錯（跟這個檔案的姿勢標籤比對，姿勢標籤來自檔名，
是我們自己知道的 ground truth，不是模型自己猜的）。

前處理/模型推論邏輯直接重用 run_model_validation.py 裡已經驗證過的函式
（causal 濾波、RMS envelope、PHASE_SHIFT_FRACTION=0.06 相位校準、segment-based
評估），這裡只是多存了每段的起訖時間，方便畫圖標記，核心邏輯没有另外發明一套。

2026-07-21 修正 + 新增：
    1. 修掉一個畫圖 bug——之前色塊顏色用的是 row["segment_type"]（我們切割時
       預先假設的位置標籤，每個 epoch 都固定循環 Rest→Contract→Release，
       跟模型輸出無關），改成 row["predicted"]（模型真正的輸出）。
    2. 新增 plot_sliding_window()：完全不猜週期/相位，固定 3 秒視窗每 0.3 秒
       滑動一次，每個位置都獨立丟進模型——用來驗證模型是不是真的跟著訊號
       變化在動，不依賴 evaluate_epoch_segments_with_time() 那套可能有錯的
       週期/相位假設。

註（2026-07-21）：連續預測的圖已改由 plot_realtime_prediction.py 產生（一個檔案
一張、5 類分開標色）。這支現在的定位是「segment-based 評估數字的來源」，圖只是
附帶產物，被 scan_phase_all_files.py import 使用。

輸出：
    results/segment_eval_figures/<subject>_<file_stem>*.png
    results/two_good_days_segment_predictions.csv
    results/two_good_days_accuracy_summary.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")   # src/ 目錄，與檔案放在多深無關
sys.path.insert(0, str(_SRC))
from semg.core import pipeline as pp
from semg.core.softpinch_model import load_model
from semg.model_eval.run_model_validation import (
    MODEL_PATH, FS, RMS_STEP, PRED_MAPPING, CH_FOR_TARGET, POSTURE_TO_TARGET,
    preprocess_for_model, find_calibrated_anchor,
)

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "src").parent
PKG_ROOT = REPO_ROOT
FIG_DIR = REPO_ROOT / "results" / "segment_eval_figures"
RESULTS_DIR = REPO_ROOT / "results"

CH_MUSCLE = {0: "ch1 (wrist flexor)", 1: "ch2 (index)", 2: "ch3 (thumb)"}

# 這兩天的資料電極貼得比較好、訊號品質比較乾淨（使用者目視判斷），只挑 index/thumb
# （跟 run_model_validation.py 一樣，pinch/grasp 的協定結構跟這個模型的三等分假設不符）
TARGET_FILES = [
    ("gary", REPO_ROOT / "data/gary/indexgary_20260716_17-06-56_motion1.csv", "index"),
    ("gary", REPO_ROOT / "data/gary/thumbgary_20260716_17-12-05_motion2.csv", "thumb"),
    ("neil1", REPO_ROOT / "data/neil1/indexlin_20260720_20-51-59.csv", "index"),
    ("neil1", REPO_ROOT / "data/neil1/thumblin_20260720_20-55-57.csv", "thumb"),
]

SEGMENT_COLOR = {
    "Rest": "#B0B0B0",
    "Contract": "#4C8BF5",
    "Release": "#F5A24C",
}
CORRECT_EDGE = "#2E7D32"
WRONG_EDGE = "#C62828"

# 滑動視窗連續預測用：完全不假設週期/相位，固定用協定裡已知的單階段時長（3秒）
# 當視窗寬度，逐步滑過整段訊號，每個位置都各自獨立丟進模型——用來驗證模型是不是
# 真的跟著訊號變化在動，不是靠我們猜的週期網格湊出來的表面規律。
SLIDING_WINDOW_SEC = 3.0
SLIDING_STEP_SEC = 0.3


def simplify_predicted(label: str) -> str:
    """把 'Index Contract'/'Thumb Release'/'Rest' 這種完整標籤，摺成畫圖用的三色
    (Rest/Contract/Release)——顏色要對應模型實際輸出，不是我們預先假設的位置。"""
    if label == "Rest":
        return "Rest"
    return "Contract" if "Contract" in label else "Release"


def evaluate_epoch_segments_with_time(env, mu, sigma, anchor_idx, period_samples, target, model):
    """跟 run_model_validation.evaluate_epoch_segments 邏輯相同，多回傳每段的
    起訖時間（秒），供畫圖標記用。"""
    env_norm = (env - mu) / (sigma + 1e-8)
    n_epochs = (len(env_norm) - anchor_idx) // period_samples
    seg_len = period_samples // 3
    sec_per_sample = RMS_STEP / FS

    rows = []
    for e in range(n_epochs):
        base = anchor_idx + e * period_samples
        epoch = env_norm[base: base + period_samples]
        seg_defs = [
            ("Rest", "Rest", 0, seg_len),
            ("Contract", f"{target} Contract", seg_len, 2 * seg_len),
            ("Release", f"{target} Release", 2 * seg_len, 3 * seg_len),
        ]
        for seg_type, expected_label, s0, s1 in seg_defs:
            seg = epoch[s0:s1]
            inp = torch.tensor(seg, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _, _ = model(inp)
                probs = torch.softmax(logits, dim=1)
                pred_idx = int(torch.argmax(logits, dim=1).item())
            pred_label = PRED_MAPPING[pred_idx]
            rows.append({
                "epoch_idx": e,
                "segment_type": seg_type,
                "expected": expected_label,
                "predicted": pred_label,
                "confidence": float(probs[0, pred_idx].item()),
                "correct": pred_label == expected_label,
                "start_time_s": (base + s0) * sec_per_sample,
                "end_time_s": (base + s1) * sec_per_sample,
            })
    return pd.DataFrame(rows)


ABBREV = {"Rest": "Rest", "Contract": "Con", "Release": "Rel"}


def _draw_legend(fig):
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=SEGMENT_COLOR["Rest"], alpha=0.35, edgecolor="none", label="pred: Rest"),
        Patch(facecolor=SEGMENT_COLOR["Contract"], alpha=0.35, edgecolor="none", label="pred: Contract"),
        Patch(facecolor=SEGMENT_COLOR["Release"], alpha=0.35, edgecolor="none", label="pred: Release"),
        Patch(facecolor="none", edgecolor=CORRECT_EDGE, linewidth=1.5, label="border: correct"),
        Patch(facecolor="none", edgecolor=WRONG_EDGE, linewidth=1.5, label="border: wrong"),
    ]
    fig.legend(handles=handles, loc="upper center", fontsize=8, ncol=5,
               bbox_to_anchor=(0.5, 0.90), frameon=False)


def plot_file(subject, file_stem, posture, env, env_time, seg_df, out_png):
    """整段錄音的總覽圖：色塊顏色 = 模型實際預測的類別（不是我們切割時假設的位置），
    邊框顏色 = 對照姿勢標籤後對/錯。只畫色塊不寫逐段文字——一個檔案動輒 90-100 個
    segment，全部塞文字會疊在一起看不清楚，細節文字留給 plot_zoom() 那張圖。

    2026-07-21 修正：先前版本這裡誤用 row["segment_type"]（我們切割時預先假設的
    位置標籤，每個 epoch 都固定 Rest→Contract→Release 循環，跟模型輸出無關）上色，
    導致色塊看起來像死板循環，跟模型是否真的有在判斷無關——這是畫圖的 bug，不是
    刻意設計。改用 row["predicted"] 才是模型真正的輸出。"""
    fig, axes = plt.subplots(3, 1, figsize=(20, 8), sharex=True)

    for ch in range(3):
        ax = axes[ch]
        ax.plot(env_time, env[:, ch], color="#222222", lw=0.8)
        ax.set_ylabel(f"{CH_MUSCLE[ch]}\nRMS (mV)", fontsize=9)
        ax.margins(x=0)
        for _, row in seg_df.iterrows():
            edge = CORRECT_EDGE if row["correct"] else WRONG_EDGE
            ax.axvspan(row["start_time_s"], row["end_time_s"],
                       color=SEGMENT_COLOR[simplify_predicted(row["predicted"])], alpha=0.28,
                       edgecolor=edge, linewidth=0.8)

    accuracy = seg_df["correct"].mean()
    fig.suptitle(
        f"{subject}/{file_stem}  (posture: {posture})  —  segment accuracy = {accuracy:.1%}  "
        f"({seg_df['correct'].sum()}/{len(seg_df)} segments correct)\n"
        f"overview — see the matching *_zoom.png for per-segment predicted labels",
        fontsize=11, y=0.985
    )
    _draw_legend(fig)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return accuracy


def plot_zoom(subject, file_stem, posture, env, env_time, seg_df, out_png, n_epochs=6):
    """放大前 n_epochs 個 epoch，段夠寬才塞得下逐段文字標籤（預測類別+信心值）。"""
    zoom_df = seg_df[seg_df["epoch_idx"] < n_epochs]
    if len(zoom_df) == 0:
        return
    t_end = zoom_df["end_time_s"].max()
    mask = env_time <= t_end + 1.0

    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
    for ch in range(3):
        ax = axes[ch]
        ax.plot(env_time[mask], env[mask, ch], color="#222222", lw=1.1)
        ax.set_ylabel(f"{CH_MUSCLE[ch]}\nRMS (mV)", fontsize=9)
        ax.set_xlim(0, t_end + 1.0)
        for _, row in zoom_df.iterrows():
            edge = CORRECT_EDGE if row["correct"] else WRONG_EDGE
            ax.axvspan(row["start_time_s"], row["end_time_s"],
                       color=SEGMENT_COLOR[simplify_predicted(row["predicted"])], alpha=0.28,
                       edgecolor=edge, linewidth=1.2)

    ax_top = axes[0]
    y_top = ax_top.get_ylim()[1]
    for _, row in zoom_df.iterrows():
        start = row["start_time_s"]
        color = CORRECT_EDGE if row["correct"] else WRONG_EDGE
        label = f"{row['predicted']} ({row['confidence']:.2f})"
        ax_top.text(start + 0.15, y_top * 1.05, label, ha="left", va="bottom",
                    fontsize=7.5, color=color, rotation=35, rotation_mode="anchor",
                    clip_on=False)

    accuracy = zoom_df["correct"].mean()
    fig.suptitle(
        f"{subject}/{file_stem}  (posture: {posture})  —  first {n_epochs} epochs, "
        f"zoom accuracy = {accuracy:.1%}\n"
        f"text = model predicted label + confidence, checked against filename posture ground truth",
        fontsize=11, y=0.985
    )
    _draw_legend(fig)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout(rect=[0, 0, 1, 0.76])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def sliding_window_predict(env, mu, sigma, model):
    """完全不猜週期、不猜相位——固定用 3 秒視窗，每 0.3 秒滑動一次，每個位置都是
    獨立丟進模型的一筆推論，不依賴任何我們自己假設的切割網格。用來直接檢驗「模型
    是不是真的跟著訊號變化在動」，不受 evaluate_epoch_segments_with_time() 那套
    週期/相位假設影響——那套假設本身可能有錯（見 scan_phase_per_file.py），這裡
    刻意繞開它，看模型的原始行為。"""
    env_norm = (env - mu) / (sigma + 1e-8)
    sec_per_sample = RMS_STEP / FS
    window_samples = int(round(SLIDING_WINDOW_SEC / sec_per_sample))
    step_samples = max(1, int(round(SLIDING_STEP_SEC / sec_per_sample)))

    rows = []
    for start in range(0, len(env_norm) - window_samples, step_samples):
        seg = env_norm[start:start + window_samples]
        inp = torch.tensor(seg, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _, _ = model(inp)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(logits, dim=1).item())
        rows.append({
            "center_time_s": (start + window_samples / 2) * sec_per_sample,
            "predicted": PRED_MAPPING[pred_idx],
            "confidence": float(probs[0, pred_idx].item()),
        })
    return pd.DataFrame(rows)


CLASS_CODE = {"Rest": 0, "Contract": 1, "Release": 2}


def plot_sliding_window(subject, file_stem, posture, env, env_time, slide_df, out_png):
    """3 通道 RMS envelope + 一條連續色帶，色帶完全來自 sliding_window_predict()
    的獨立推論結果，不套用任何週期/相位假設——拿來直接檢驗模型有沒有跟著訊號變化
    移動，不是靠我們切割網格湊出來的規律。"""
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(4, 1, figsize=(20, 9), sharex=True,
                              gridspec_kw={"height_ratios": [3, 3, 3, 1.2]})

    for ch in range(3):
        ax = axes[ch]
        ax.plot(env_time, env[:, ch], color="#222222", lw=0.8)
        ax.set_ylabel(f"{CH_MUSCLE[ch]}\nRMS (mV)", fontsize=9)
        ax.margins(x=0)

    ax_strip = axes[3]
    codes = slide_df["predicted"].map(simplify_predicted).map(CLASS_CODE).to_numpy()
    cmap = ListedColormap([SEGMENT_COLOR["Rest"], SEGMENT_COLOR["Contract"], SEGMENT_COLOR["Release"]])
    t0, t1 = slide_df["center_time_s"].iloc[0], slide_df["center_time_s"].iloc[-1]
    ax_strip.imshow(codes[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=2,
                     extent=[t0, t1, 0, 1], interpolation="nearest")
    ax_strip.set_yticks([])
    ax_strip.set_ylabel("model\npredicted\n(sliding)", fontsize=8)
    ax_strip.margins(x=0)

    _draw_legend(fig)
    fig.suptitle(
        f"{subject}/{file_stem}  (posture: {posture})  —  sliding-window continuous prediction "
        f"(window={SLIDING_WINDOW_SEC}s, step={SLIDING_STEP_SEC}s)\n"
        f"no period/phase guessing — every window is an independent model call; "
        f"compare the color strip directly against the RMS bursts above",
        fontsize=11, y=0.99
    )
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_model(MODEL_PATH / "model.pth")
    print(f"model loaded: {ckpt['model_name']} / {ckpt['sensor_name']}")

    all_seg_rows = []
    summary_rows = []
    for subject, csv_path, posture in TARGET_FILES:
        print(f"[跑模型+畫圖] {subject}/{csv_path.name}")
        df = pp.load_channel(str(csv_path), channels=(1, 2, 3))
        raw = {ch: df[f"ch{ch}"].values for ch in (1, 2, 3)}
        env = preprocess_for_model(raw)
        mu, sigma = env.mean(axis=0), env.std(axis=0)
        target = POSTURE_TO_TARGET[posture]

        env_time = np.arange(len(env)) * (RMS_STEP / FS)
        anchor_idx, period_samples, period_s = find_calibrated_anchor(
            env[:, CH_FOR_TARGET[target]], env_time)

        seg_df = evaluate_epoch_segments_with_time(env, mu, sigma, anchor_idx,
                                                     period_samples, target, model)
        seg_df["subject"] = subject
        seg_df["file_stem"] = csv_path.stem
        seg_df["posture"] = posture
        all_seg_rows.append(seg_df)

        out_png = FIG_DIR / f"{subject}_{csv_path.stem}.png"
        accuracy = plot_file(subject, csv_path.stem, posture, env, env_time, seg_df, out_png)

        out_zoom_png = FIG_DIR / f"{subject}_{csv_path.stem}_zoom.png"
        plot_zoom(subject, csv_path.stem, posture, env, env_time, seg_df, out_zoom_png)

        slide_df = sliding_window_predict(env, mu, sigma, model)
        out_slide_png = FIG_DIR / f"{subject}_{csv_path.stem}_sliding.png"
        plot_sliding_window(subject, csv_path.stem, posture, env, env_time, slide_df, out_slide_png)

        summary_rows.append({
            "subject": subject,
            "file_stem": csv_path.stem,
            "posture": posture,
            "detected_period_s": period_s,
            "n_epochs": seg_df["epoch_idx"].nunique(),
            "n_segments": len(seg_df),
            "segment_accuracy": accuracy,
            "mean_confidence": seg_df["confidence"].mean(),
            "figure_overview": str(out_png.relative_to(PKG_ROOT)),
            "figure_zoom": str(out_zoom_png.relative_to(PKG_ROOT)),
            "figure_sliding": str(out_slide_png.relative_to(PKG_ROOT)),
        })
        print(f"    -> {out_png.name} / {out_zoom_png.name} / {out_slide_png.name}  segment_accuracy={accuracy:.1%}")

    seg_all = pd.concat(all_seg_rows, ignore_index=True)
    seg_csv = RESULTS_DIR / "two_good_days_segment_predictions.csv"
    seg_all.to_csv(seg_csv, index=False)

    summary = pd.DataFrame(summary_rows).sort_values(["subject", "posture"])
    summary_csv = RESULTS_DIR / "two_good_days_accuracy_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print(f"\n[完成] {seg_csv}")
    print(f"[完成] {summary_csv}")
    print(summary.to_string(index=False))
    print(f"\n整體 pooled segment accuracy: {seg_all['correct'].mean():.1%}")


if __name__ == "__main__":
    main()
