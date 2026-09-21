"""
視窗長度比較圖：同一段訊號、同一個模型，只把滑動視窗從 3.0 秒改成 1.0 秒，
看模型輸出差多少。畫法跟 plot_realtime_prediction.py 一致（3 通道 RMS envelope
+ 模型輸出時間軸，5 類分開標色），只是下方疊兩條時間軸做對照。

動機：3 秒視窗代表模型需要「過去 3 秒的歷史」才能判斷，動作剛開始時視窗裡多半
還是舊資料 → 反應延遲，對外骨骼即時控制太慢。縮到 1 秒可以把延遲壓下來，但 1 秒
是模型沒訓練過的長度（訓練時每筆樣本 = 一個完整動作階段 = 3 秒），準確率會掉。
這張圖就是把這個 trade-off 視覺化，數字對照見 results/window_size_latency_test.csv。

輸出：results/window_comparison/<subject>_<file_stem>.png（另存一份 _zoom 版）
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
from semg.core.softpinch_model import load_model
from semg.model_eval.run_model_validation import MODEL_PATH, FS, RMS_STEP, PRED_MAPPING
from semg.realtime.plot_realtime_prediction import (
    load_and_preprocess, CLASS_ORDER, CLASS_COLOR, CLASS_Y,
    CH_MUSCLE, CH_MUSCLE_SOFTPINCH, GROUP_LABEL, SP_ROOT,
)

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "src").parent
FIG_DIR = REPO_ROOT / "results" / "window_comparison"
RESULTS_DIR = REPO_ROOT / "results"

# 要比較的兩個視窗長度：3.0 = 模型訓練長度；1.0 = 為了即時控制縮短後的候選值
WINDOWS = [3.0, 1.0]
STEP_SEC = 0.25
ZOOM_SEC = 60.0   # 放大版顯示的秒數，讓延遲/抖動看得出來

TARGET_FILES = [
    ("neil1", REPO_ROOT / "data/neil1/indexlin_20260716_21-58-56_motion1.csv", "index", "ours", "ours"),
    ("neil1", REPO_ROOT / "data/neil1/indexlin_20260720_20-51-59.csv", "index", "ours", "ours"),
    ("gary",  REPO_ROOT / "data/gary/thumbgary_20260716_17-12-05_motion2.csv", "thumb", "ours", "ours"),
    ("SP_subject_8", SP_ROOT / "subject_8/EMG/flex_index_finger_2026-02-19 10-50-44.csv",
     "index", "softpinch", "softpinch_held_out"),
]


def sliding_predict(env, mu, sigma, model, window_sec):
    env_norm = (env - mu) / (sigma + 1e-8)
    sec_per_sample = RMS_STEP / FS
    win = int(round(window_sec / sec_per_sample))
    step = max(1, int(round(STEP_SEC / sec_per_sample)))

    segs, centers = [], []
    for s in range(0, len(env_norm) - win + 1, step):
        segs.append(env_norm[s:s + win])
        centers.append((s + win / 2) * sec_per_sample)
    batch = torch.tensor(np.stack(segs), dtype=torch.float32)
    with torch.no_grad():
        logits, _, _ = model(batch)
        preds = torch.argmax(logits, dim=1).numpy()
    return pd.DataFrame({
        "center_time_s": centers,
        "predicted": [PRED_MAPPING[p] for p in preds],
    })


def _draw_strip(ax, pred_df, window_sec, expected_finger, t_max=None):
    """鋼琴捲軸式輸出時間軸，跟 plot_realtime_prediction.py 同一種畫法。"""
    df = pred_df if t_max is None else pred_df[pred_df["center_time_s"] <= t_max]
    t = df["center_time_s"].to_numpy()
    labels = df["predicted"].to_numpy()
    half = STEP_SEC / 2

    run_start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[run_start]:
            cls = labels[run_start]
            ax.broken_barh([(t[run_start] - half, t[i - 1] - t[run_start] + 2 * half)],
                            (CLASS_Y[cls] - 0.36, 0.72),
                            facecolors=CLASS_COLOR[cls], edgecolor="none")
            run_start = i

    ax.step(t, df["predicted"].map(CLASS_Y).to_numpy(), where="mid",
            color="#333333", lw=0.7, alpha=0.5)

    # 標出這個檔案應該落在哪兩列
    for cls in CLASS_ORDER:
        if cls.startswith(expected_finger):
            ax.axhspan(CLASS_Y[cls] - 0.5, CLASS_Y[cls] + 0.5,
                       color="#2E7D32", alpha=0.07, zorder=0)

    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_yticklabels(CLASS_ORDER, fontsize=8.5)
    ax.set_ylim(-0.6, len(CLASS_ORDER) - 0.4)
    ax.margins(x=0)
    ax.grid(axis="x", alpha=0.2)

    active = labels[labels != "Rest"]
    acc = np.mean([l.startswith(expected_finger) for l in active]) if len(active) else float("nan")
    n_switch = int((labels[1:] != labels[:-1]).sum())
    ax.set_ylabel(f"window = {window_sec}s\nmodel output", fontsize=9.5)
    return acc, n_switch


def plot_compare(subject, file_stem, posture, env, env_time, preds, out_png,
                  source, group, t_max=None):
    fig, axes = plt.subplots(3 + len(WINDOWS), 1, figsize=(19, 11), sharex=True,
                              gridspec_kw={"height_ratios": [1.6, 1.6, 1.6, 2.6, 2.6]})
    expected_finger = "Index" if posture == "index" else "Thumb"
    ch_labels = CH_MUSCLE if source == "ours" else CH_MUSCLE_SOFTPINCH

    mask = env_time <= t_max if t_max else np.ones(len(env_time), bool)
    for ch in range(3):
        ax = axes[ch]
        ax.plot(env_time[mask], env[mask, ch], color="#222222", lw=0.8)
        ax.set_ylabel(ch_labels[ch], fontsize=9.5)
        ax.margins(x=0)
        ax.grid(axis="x", alpha=0.2)

    stats = []
    for i, w in enumerate(WINDOWS):
        acc, n_switch = _draw_strip(axes[3 + i], preds[w], w, expected_finger, t_max)
        stats.append((w, acc, n_switch))

    axes[-1].set_xlabel("time (s)", fontsize=10.5)
    if t_max:
        axes[-1].set_xlim(0, t_max)

    line = "   vs   ".join(
        f"{w}s window: {acc:.0%} finger acc, {ns} output switches" for w, acc, ns in stats)
    fig.suptitle(
        f"[{GROUP_LABEL[group]}]   {subject} / {file_stem}   recorded motion = {expected_finger}\n"
        f"WINDOW LENGTH COMPARISON — same signal, same model, only the sliding window length differs\n"
        f"{line}\n"
        f"shorter window = lower latency for real-time control, but the model was trained on 3s segments",
        fontsize=11, y=0.99
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return stats


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    model, _ = load_model(MODEL_PATH / "model.pth")

    rows = []
    for subject, csv_path, posture, source, group in TARGET_FILES:
        print(f"[視窗比較] {subject}/{csv_path.name}")
        env = load_and_preprocess(csv_path, source)
        mu, sigma = env.mean(axis=0), env.std(axis=0)
        env_time = np.arange(len(env)) * (RMS_STEP / FS)

        preds = {w: sliding_predict(env, mu, sigma, model, w) for w in WINDOWS}
        safe = csv_path.stem.replace(" ", "_")

        stats = plot_compare(subject, csv_path.stem, posture, env, env_time, preds,
                              FIG_DIR / f"{subject}_{safe}.png", source, group)
        plot_compare(subject, csv_path.stem, posture, env, env_time, preds,
                     FIG_DIR / f"{subject}_{safe}_zoom.png", source, group, t_max=ZOOM_SEC)

        for w, acc, ns in stats:
            rows.append({"subject": subject, "file_stem": csv_path.stem, "posture": posture,
                          "group": group, "window_sec": w, "finger_accuracy": acc,
                          "output_switches": ns})
            print(f"    window={w}s  手指判別率={acc:.1%}  輸出切換次數={ns}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "window_comparison_summary.csv", index=False)
    print(f"\n[完成] 圖：{FIG_DIR}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
