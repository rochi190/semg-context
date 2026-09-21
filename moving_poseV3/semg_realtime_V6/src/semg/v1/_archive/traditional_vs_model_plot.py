"""
傳統特徵分類器 vs CNN+LSTM：畫在同一張圖上直接對照。

跟 traditional_feature_baseline.py 的差別（很重要）：
    那支是「二元分手指」、而且在我們自己的資料上訓練，跟模型的 5 類 zero-shot
    不是同一個任務、也不是同一個難度，並排比會失焦。

    這支讓兩者站在**完全相同的位置**：
      - 都用 SoftPINCH 的資料訓練（他們協定固定 9 秒三段，標籤可靠，
        且排除 subject_8 —— 跟模型 checkpoint 的訓練集切分一致）
      - 都做同樣的 5 類（Index/Thumb × Contract/Release + Rest）
      - 都 zero-shot 套用到我們的資料，完全沒看過我們任何一筆
    這樣畫出來的兩條時間軸才是可以直接比對的。

前處理一律走 plot_realtime_prediction.load_and_preprocess()，訓練跟測試用同一套，
避免 filtfilt/causal 混用造成的差異。

輸出：
    results/traditional_vs_model/<subject>_<file_stem>.png（另存 _zoom 版）
    results/traditional_vs_model_summary.csv
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
from semg._archive.traditional_feature_baseline import window_features

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "src").parent
FIG_DIR = REPO_ROOT / "results" / "traditional_vs_model"
RESULTS_ROOT = REPO_ROOT / "results"

WINDOW_SEC = 3.0
STEP_SEC = 0.25
ZOOM_SEC = 60.0

TRIAL_SEC = 9          # 論文協定：每個 trial 9 秒
SEC_PER_PT = RMS_STEP / FS          # 0.025s
PTS_PER_EPOCH = int(TRIAL_SEC / SEC_PER_PT)   # 360
SEG_LEN = PTS_PER_EPOCH // 3                   # 120 = 3 秒

# 訓練用的 SoftPINCH subject（排除 subject_8，跟模型 checkpoint 的切分一致）
TRAIN_SUBJECTS = ["subject_0", "subject_1", "subject_2", "subject_3",
                   "subject_4", "subject_5"]

TEST_FILES = [
    ("neil1", REPO_ROOT / "data/neil1/indexlin_20260716_21-58-56_motion1.csv", "index", "ours", "ours"),
    ("neil1", REPO_ROOT / "data/neil1/thumblin_20260716_22-02-17_motion2.csv", "thumb", "ours", "ours"),
    ("gary",  REPO_ROOT / "data/gary/thumbgary_20260716_17-12-05_motion2.csv", "thumb", "ours", "ours"),
    ("SP_subject_8", SP_ROOT / "subject_8/EMG/flex_index_finger_2026-02-19 10-50-44.csv",
     "index", "softpinch", "softpinch_held_out"),
]


def build_training_set():
    """從 SoftPINCH 訓練集 subject 抽出 5 類的 3 秒 segment + 傳統特徵。"""
    X, y = [], []
    for subj in TRAIN_SUBJECTS:
        emg_dir = SP_ROOT / subj / "EMG"
        if not emg_dir.exists():
            continue
        for posture in ("index", "thumb"):
            target = "Index" if posture == "index" else "Thumb"
            for csv_path in sorted(emg_dir.glob(f"flex_{posture}_finger_*.csv")):
                env = load_and_preprocess(csv_path, "softpinch")
                mu, sigma = env.mean(axis=0), env.std(axis=0)
                env_n = (env - mu) / (sigma + 1e-8)
                n_epochs = len(env_n) // PTS_PER_EPOCH
                for e in range(n_epochs):
                    ep = env_n[e * PTS_PER_EPOCH:(e + 1) * PTS_PER_EPOCH]
                    for lab, seg in (("Rest", ep[:SEG_LEN]),
                                      (f"{target} Contract", ep[SEG_LEN:2 * SEG_LEN]),
                                      (f"{target} Release", ep[2 * SEG_LEN:])):
                        X.append(window_features(seg))
                        y.append(lab)
        print(f"    {subj} 完成，累積 {len(X)} 筆")
    return np.array(X), np.array(y)


def sliding_windows(env, mu, sigma):
    env_n = (env - mu) / (sigma + 1e-8)
    win = int(round(WINDOW_SEC / SEC_PER_PT))
    step = max(1, int(round(STEP_SEC / SEC_PER_PT)))
    segs, centers = [], []
    for s in range(0, len(env_n) - win + 1, step):
        segs.append(env_n[s:s + win])
        centers.append((s + win / 2) * SEC_PER_PT)
    return segs, np.array(centers)


def _draw_strip(ax, centers, labels, ylabel, expected_finger, t_max=None):
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
        if cls.startswith(expected_finger):
            ax.axhspan(CLASS_Y[cls] - 0.5, CLASS_Y[cls] + 0.5,
                       color="#2E7D32", alpha=0.07, zorder=0)
    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_yticklabels(CLASS_ORDER, fontsize=8.5)
    ax.set_ylim(-0.6, len(CLASS_ORDER) - 0.4)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.margins(x=0)
    ax.grid(axis="x", alpha=0.2)

    act = labels[labels != "Rest"]
    acc = np.mean([l.startswith(expected_finger) for l in act]) if len(act) else float("nan")
    return acc


def plot_compare(subject, file_stem, posture, env, env_time, centers,
                  model_labels, trad_labels, out_png, source, group, t_max=None):
    fig, axes = plt.subplots(5, 1, figsize=(19, 11), sharex=True,
                              gridspec_kw={"height_ratios": [1.5, 1.5, 1.5, 2.6, 2.6]})
    expected = "Index" if posture == "index" else "Thumb"
    ch_labels = CH_MUSCLE if source == "ours" else CH_MUSCLE_SOFTPINCH

    mask = env_time <= t_max if t_max else np.ones(len(env_time), bool)
    for ch in range(3):
        axes[ch].plot(env_time[mask], env[mask, ch], color="#222222", lw=0.8)
        axes[ch].set_ylabel(ch_labels[ch], fontsize=9.5)
        axes[ch].margins(x=0)
        axes[ch].grid(axis="x", alpha=0.2)

    acc_m = _draw_strip(axes[3], centers, model_labels, "CNN+LSTM\nmodel output", expected, t_max)
    acc_t = _draw_strip(axes[4], centers, trad_labels, "traditional features\n+ LDA output", expected, t_max)

    axes[-1].set_xlabel("time (s)", fontsize=10.5)
    if t_max:
        axes[-1].set_xlim(0, t_max)

    fig.suptitle(
        f"[{GROUP_LABEL[group]}]   {subject} / {file_stem}   recorded motion = {expected}\n"
        f"SAME TASK, SAME TRAINING DATA, SAME ZERO-SHOT SETTING — only the classifier differs\n"
        f"both trained on SoftPINCH subjects (subject_8 excluded), 5-class, {WINDOW_SEC}s window / {STEP_SEC}s step\n"
        f"finger discrimination (excl. Rest):   CNN+LSTM {acc_m:.0%}   vs   traditional+LDA {acc_t:.0%}",
        fontsize=11, y=0.99
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return acc_m, acc_t


def main():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 1. 從 SoftPINCH 訓練集建立傳統特徵訓練資料（排除 subject_8）===")
    Xtr, ytr = build_training_set()
    print(f"  訓練樣本 {len(Xtr)} 筆，特徵 {Xtr.shape[1]} 維，類別分布：")
    for c, n in zip(*np.unique(ytr, return_counts=True)):
        print(f"    {c:16s} {n}")

    lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    lda.fit(Xtr, ytr)
    print(f"  LDA 訓練集自身準確率（僅供參考，非驗證）: {lda.score(Xtr, ytr):.1%}")

    print("\n=== 2. 載入 CNN+LSTM ===")
    model, _ = load_model(MODEL_PATH / "model.pth")

    print("\n=== 3. 兩者 zero-shot 套用到測試檔案並畫圖 ===")
    rows = []
    for subject, path, posture, source, group in TEST_FILES:
        print(f"  {subject}/{path.name}")
        env = load_and_preprocess(path, source)
        mu, sigma = env.mean(axis=0), env.std(axis=0)
        env_time = np.arange(len(env)) * SEC_PER_PT
        segs, centers = sliding_windows(env, mu, sigma)

        batch = torch.tensor(np.stack(segs), dtype=torch.float32)
        with torch.no_grad():
            logits, _, _ = model(batch)
        model_labels = np.array([PRED_MAPPING[i] for i in logits.argmax(1).numpy()])

        feats = np.stack([window_features(s) for s in segs])
        trad_labels = lda.predict(feats)

        safe = path.stem.replace(" ", "_")
        acc_m, acc_t = plot_compare(subject, path.stem, posture, env, env_time, centers,
                                     model_labels, trad_labels,
                                     FIG_DIR / f"{subject}_{safe}.png", source, group)
        plot_compare(subject, path.stem, posture, env, env_time, centers,
                     model_labels, trad_labels,
                     FIG_DIR / f"{subject}_{safe}_zoom.png", source, group, t_max=ZOOM_SEC)

        rows.append({"subject": subject, "file_stem": path.stem, "posture": posture,
                      "group": group, "cnn_lstm_finger_acc": acc_m,
                      "traditional_lda_finger_acc": acc_t,
                      "cnn_lstm_rest_frac": (model_labels == "Rest").mean(),
                      "traditional_rest_frac": (trad_labels == "Rest").mean()})
        print(f"    CNN+LSTM {acc_m:.1%}   傳統+LDA {acc_t:.1%}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_ROOT / "traditional_vs_model_summary.csv", index=False)
    print(f"\n[完成] 圖：{FIG_DIR}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
