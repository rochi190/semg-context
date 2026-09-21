"""
回答「這個模型能不能拿來做 real-time 控制」的實測。

問題背景：
    offline checkpoint（all_subjects, 5-class）訓練時每筆樣本 = 一個完整動作階段
    = 3 秒。做連續預測時視窗是「滑動」的，所以輸出頻率由步進決定（可以 0.1 秒一次），
    不是 3 秒才輸出一次。但模型需要「過去 3 秒的歷史」才能判斷，代表動作剛開始時
    視窗裡大部分還是舊資料 → 反應延遲。視窗越短延遲越低，但短視窗是模型沒訓練過的
    長度，準確率會掉。

    這支程式就是量這個 trade-off：視窗從 3.0 秒一路縮到 0.25 秒，看手指判別率
    （排除 Rest，亂猜=50%）掉多少，藉此判斷「能不能靠縮短視窗把延遲壓到可接受範圍」。

輸出：results/window_size_latency_test.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")   # src/ 目錄，與檔案放在多深無關
sys.path.insert(0, str(_SRC))
from semg.core import pipeline as pp
from semg.core.softpinch_model import load_model
from semg.model_eval.run_model_validation import (
    MODEL_PATH, FS, RMS_STEP, PRED_MAPPING, preprocess_for_model,
    BANDPASS, HAMPEL_HALF_WINDOW, HAMPEL_SIGMA, RMS_WINDOW,
    causal_notch_bandpass, causal_hampel, rms_stream,
)

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "src").parent
RESULTS_ROOT = REPO_ROOT / "results"
SP_ROOT = REPO_ROOT / "src/SoftPINCH/src/experiment/data/subject_independent"

SOFTPINCH_NOTCH_FREQ = 50
SOFTPINCH_TRIM_SEC = 3.0

# 測試視窗長度（秒）。3.0 = 模型原本的訓練長度；0.5 = 論文 real-time 系統用的長度
WINDOW_GRID = [3.0, 2.5, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25]
STEP_SEC = 0.5  # 這裡只在意準確率，步進可以粗一點省時間

TEST_FILES = [
    # 我們最好的 + 中等的，加上論文 held-out 當標竿
    ("ours", "neil1", REPO_ROOT / "data/neil1/indexlin_20260716_21-58-56_motion1.csv", "index"),
    ("ours", "neil1", REPO_ROOT / "data/neil1/indexlin_20260720_20-51-59.csv", "index"),
    ("ours", "gary",  REPO_ROOT / "data/gary/thumbgary_20260716_17-12-05_motion2.csv", "thumb"),
    ("softpinch", "SP_subject_8",
     SP_ROOT / "subject_8/EMG/flex_index_finger_2026-02-19 10-50-44.csv", "index"),
    ("softpinch", "SP_subject_8",
     SP_ROOT / "subject_8/EMG/flex_thumb_finger_2026-02-19 11-13-00.csv", "thumb"),
]


def load_and_preprocess(csv_path: Path, source: str) -> np.ndarray:
    if source == "ours":
        df = pp.load_channel(str(csv_path), channels=(1, 2, 3))
        return preprocess_for_model({ch: df[f"ch{ch}"].values for ch in (1, 2, 3)})
    df = pd.read_csv(csv_path)
    raw_2d = df.iloc[:, :3].values
    envs = []
    for ch in range(3):
        x = raw_2d[:, ch].astype(np.float64)
        x = pp.trim(x, FS, SOFTPINCH_TRIM_SEC, SOFTPINCH_TRIM_SEC)
        x = x - x.mean()
        filt = causal_notch_bandpass(x, FS, SOFTPINCH_NOTCH_FREQ, BANDPASS)
        hampel = causal_hampel(filt, HAMPEL_HALF_WINDOW, HAMPEL_SIGMA)
        envs.append(rms_stream(hampel, RMS_WINDOW, RMS_STEP))
    min_len = min(len(e) for e in envs)
    return np.stack([e[:min_len] for e in envs], axis=1)


def eval_window(env, mu, sigma, model, window_sec, target):
    """回傳 (手指判別率, Rest 比例, 視窗數)。手指判別率 = 排除 Rest 後選對手指的比例。"""
    env_norm = (env - mu) / (sigma + 1e-8)
    sec_per_sample = RMS_STEP / FS
    win = int(round(window_sec / sec_per_sample))
    step = max(1, int(round(STEP_SEC / sec_per_sample)))
    if win < 4 or len(env_norm) < win:
        return float("nan"), float("nan"), 0

    segs = [env_norm[s:s + win] for s in range(0, len(env_norm) - win + 1, step)]
    batch = torch.tensor(np.stack(segs), dtype=torch.float32)
    with torch.no_grad():
        logits, _, _ = model(batch)
        preds = torch.argmax(logits, dim=1).numpy()
    labels = np.array([PRED_MAPPING[p] for p in preds])

    rest_frac = (labels == "Rest").mean()
    active = labels[labels != "Rest"]
    hit = np.mean([lab.startswith(target) for lab in active]) if len(active) else float("nan")
    return hit, rest_frac, len(labels)


def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    model, _ = load_model(MODEL_PATH / "model.pth")

    rows = []
    for source, subject, csv_path, posture in TEST_FILES:
        print(f"[視窗長度掃描] {subject}/{csv_path.name}")
        env = load_and_preprocess(csv_path, source)
        mu, sigma = env.mean(axis=0), env.std(axis=0)
        target = "Index" if posture == "index" else "Thumb"

        for w in WINDOW_GRID:
            hit, rest_frac, n = eval_window(env, mu, sigma, model, w, target)
            rows.append({
                "source": source, "subject": subject, "file_stem": csv_path.stem,
                "posture": posture, "window_sec": w,
                "finger_accuracy": hit, "frac_rest": rest_frac, "n_windows": n,
            })
            print(f"    window={w:4.2f}s   手指判別率={hit:.1%}   Rest比例={rest_frac:.0%}")

    df = pd.DataFrame(rows)
    out = RESULTS_ROOT / "window_size_latency_test.csv"
    df.to_csv(out, index=False)

    print(f"\n[完成] {out}\n")
    print("=== 手指判別率 vs 視窗長度（列=視窗秒數，欄=資料來源）===")
    pivot = df.pivot_table(index="window_sec", columns="source",
                            values="finger_accuracy", aggfunc="mean").sort_index(ascending=False)
    print((pivot * 100).round(1).to_string())
    print("\n（亂猜 = 50%。3.0s 是模型的訓練長度；0.5s 是論文 real-time 系統用的長度）")


if __name__ == "__main__":
    main()
