"""
傳統特徵 v4 vs CNN+LSTM 的並排對照圖。

跟 traditional_vs_model_plot.py（舊版）的差別：舊版用的是 v1 世代的特徵
（算在包絡線上、跨通道正規化有 bug），結論不可信。這支改用 v4：
特徵算在原始 EMG 上、跨檔共用尺度正規化，是四個版本裡唯一行為正常的。

兩個分類器站在完全相同的位置：
    都用 SoftPINCH 訓練（排除 subject_8）、都 5 類、都 zero-shot、
    都 3 秒視窗 / 0.25 秒步進。唯一變數是分類器本身。

刻意挑四個檔案涵蓋 v4 的全部行為型態（見 results/traditional_v4_by_file.csv）：
    1. neil1 index 0720 — 傳統 99.7% 勝過 CNN 88.9%（傳統表現最好的情況）
    2. neil1 index 0713 — 傳統 0.0%，整檔只輸出 1 類、切換 0 次（完全退化）
    3. gary  thumb 0711 — 傳統 0.0% 但切換 60 次（有在判斷，但系統性判錯手指）
    4. SP subject_8     — 論文自己的硬體，傳統 100%、切換 119 次（對照組：特徵本身沒問題）

輸出：results/traditional_vs_model_v4/<subject>_<file>.png（另存 _zoom 版）
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
    CLASS_ORDER, CLASS_COLOR, CLASS_Y, CH_MUSCLE, CH_MUSCLE_SOFTPINCH,
)
from semg.traditional.traditional_baseline_v2 import preprocess_full, SEC_PER_PT, SP_ROOT
from semg.traditional.traditional_baseline_v3 import window_features_v3
from semg.traditional.traditional_baseline_v4 import normalize_shared, build_training_set_v4

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "src").parent
FIG_DIR = REPO_ROOT / "results" / "traditional_vs_model_v4"
RESULTS_ROOT = REPO_ROOT / "results"

WINDOW_SEC, STEP_SEC = 3.0, 0.25
ZOOM_SEC = 60.0

TARGETS = [
    ("neil1", REPO_ROOT / "data/neil1/indexlin_20260720_20-51-59.csv", "index", "ours",
     "traditional WINS (99.7% vs 88.9%)"),
    ("neil1", REPO_ROOT / "data/neil1/indexlin_20260713_15-24-43.csv", "index", "ours",
     "traditional COLLAPSES — 1 class, 0 switches (0.0%)"),
    ("gary", REPO_ROOT / "data/gary/thumbgary_20260711_00-05-15.csv", "thumb", "ours",
     "traditional switches 60x but picks the WRONG finger every time (0.0%)"),
    ("SP_subject_8", SP_ROOT / "subject_8/EMG/flex_index_finger_2026-02-19 10-50-44.csv",
     "index", "softpinch", "SoftPINCH's own hardware — both near-perfect (reference)"),
]


def _strip(ax, centers, labels, ylabel, expected, t_max=None):
    if t_max is not None:
        m = centers <= t_max
        centers, labels = centers[m], labels[m]
    half = STEP_SEC / 2
    run = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[run]:
            cls = labels[run]
            ax.broken_barh([(centers[run] - half, centers[i - 1] - centers[run] + 2 * half)],
                            (CLASS_Y[cls] - 0.36, 0.72),
                            facecolors=CLASS_COLOR[cls], edgecolor="none")
            run = i
    ax.step(centers, [CLASS_Y[l] for l in labels], where="mid",
            color="#333333", lw=0.7, alpha=0.5)
    for cls in CLASS_ORDER:
        if cls.startswith(expected):
            ax.axhspan(CLASS_Y[cls] - 0.5, CLASS_Y[cls] + 0.5,
                       color="#2E7D32", alpha=0.07, zorder=0)
    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_yticklabels(CLASS_ORDER, fontsize=8.5)
    ax.set_ylim(-0.6, len(CLASS_ORDER) - 0.4)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.margins(x=0)
    ax.grid(axis="x", alpha=0.2)
    act = labels[labels != "Rest"]
    acc = float(np.mean([l.startswith(expected) for l in act])) if len(act) else float("nan")
    sw = int((labels[1:] != labels[:-1]).sum())
    return acc, sw, len(set(labels))


def plot_one(subject, stem, posture, source, note, env, env_time,
              centers, m_lab, t_lab, out_png, t_max=None):
    fig, axes = plt.subplots(5, 1, figsize=(19, 11), sharex=True,
                              gridspec_kw={"height_ratios": [1.5, 1.5, 1.5, 2.6, 2.6]})
    expected = "Index" if posture == "index" else "Thumb"
    chl = CH_MUSCLE if source == "ours" else CH_MUSCLE_SOFTPINCH
    mask = env_time <= t_max if t_max else np.ones(len(env_time), bool)
    for c in range(3):
        axes[c].plot(env_time[mask], env[mask, c], color="#222222", lw=0.8)
        axes[c].set_ylabel(chl[c], fontsize=9.5)
        axes[c].margins(x=0)
        axes[c].grid(axis="x", alpha=0.2)

    am, sm, um = _strip(axes[3], centers, m_lab, "CNN+LSTM\noutput", expected, t_max)
    at, st, ut = _strip(axes[4], centers, t_lab, "traditional v4\n+ LDA output", expected, t_max)

    axes[-1].set_xlabel("time (s)", fontsize=10.5)
    if t_max:
        axes[-1].set_xlim(0, t_max)

    fig.suptitle(
        f"{subject} / {stem}   recorded motion = {expected}\n"
        f"{note}\n"
        f"same training data (SoftPINCH, subject_8 excluded), same 5 classes, same zero-shot setting, "
        f"{WINDOW_SEC}s window / {STEP_SEC}s step — only the classifier differs\n"
        f"CNN+LSTM: {am:.0%} finger acc, {sm} switches, {um} classes   |   "
        f"traditional v4: {at:.0%} finger acc, {st} switches, {ut} classes",
        fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return am, at, sm, st


def main():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=== 重建 v4 訓練集與 LDA ===", flush=True)
    Xtr, ytr = build_training_set_v4()
    lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    lda.fit(Xtr, ytr)
    print(f"  訓練完成 {len(Xtr)} 筆", flush=True)

    model, _ = load_model(MODEL_PATH / "model.pth")
    model.eval()

    win_pts = int(round(WINDOW_SEC / SEC_PER_PT))
    step_pts = int(round(STEP_SEC / SEC_PER_PT))
    rows = []

    for subject, path, posture, source, note in TARGETS:
        print(f"[畫圖] {subject}/{path.name}", flush=True)
        filt, env = preprocess_full(path, source)
        xn, eps = normalize_shared(filt)
        mu, sigma = env.mean(axis=0), env.std(axis=0)
        env_n = (env - mu) / (sigma + 1e-8)
        env_time = np.arange(len(env)) * SEC_PER_PT

        segs, feats, centers = [], [], []
        for s in range(0, len(env_n) - win_pts + 1, step_pts):
            a0, a1 = int(s * RMS_STEP), int((s + win_pts) * RMS_STEP)
            if a1 > len(xn):
                break
            segs.append(env_n[s:s + win_pts])
            feats.append(window_features_v3(xn[a0:a1], eps))
            centers.append((s + win_pts / 2) * SEC_PER_PT)
        centers = np.array(centers)

        with torch.no_grad():
            logits, _, _ = model(torch.tensor(np.stack(segs), dtype=torch.float32))
        m_lab = np.array([PRED_MAPPING[i] for i in logits.argmax(1).numpy()])
        t_lab = lda.predict(np.stack(feats))

        safe = path.stem.replace(" ", "_")
        am, at, sm, st = plot_one(subject, path.stem, posture, source, note, env, env_time,
                                   centers, m_lab, t_lab, FIG_DIR / f"{subject}_{safe}.png")
        plot_one(subject, path.stem, posture, source, note, env, env_time,
                 centers, m_lab, t_lab, FIG_DIR / f"{subject}_{safe}_zoom.png", t_max=ZOOM_SEC)
        rows.append({"subject": subject, "file_stem": path.stem, "note": note,
                      "cnn_acc": am, "trad_acc": at, "cnn_switches": sm, "trad_switches": st})
        print(f"    CNN {am:.1%}({sm}切換)  傳統v4 {at:.1%}({st}切換)", flush=True)

    pd.DataFrame(rows).to_csv(RESULTS_ROOT / "traditional_vs_model_v4_figures.csv", index=False)
    print(f"\n[完成] 圖：{FIG_DIR}")


if __name__ == "__main__":
    main()
