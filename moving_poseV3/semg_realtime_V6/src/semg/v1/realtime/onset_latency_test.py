"""
量測「反應延遲」：真實動作開始後，模型要多久才翻成 Contract？

為什麼需要這支：
    我們一直說「3 秒視窗很難做 real-time」，但那是推測，從來沒量過。
    視窗長度 ≠ 輸出頻率（輸出由步進決定，可以 0.1s 一次）；真正的問題是
    模型需要「過去 3 秒的歷史」，動作剛開始時視窗裡多半還是舊資料。
    這支就是把那個延遲量出來，決定要不要換模型。

方法（關鍵：用因果/trailing 視窗，不是置中視窗）：
    real-time 時只有過去的資料，所以時刻 t 的預測只能用 [t-W, t] 這段。
    先前的分析腳本用的是置中視窗（center = start + W/2），那對離線分析沒問題，
    但拿來談延遲會低估——這裡一律改成 trailing。

Ground truth：
    用論文 subject_8（LOSO held-out，訓練時完全沒看過）。他們的協定是每個 trial
    9 秒、三等分（rest 0-3s / contract 3-6s / release 6-9s），邊界精確已知，
    不需要我們自己猜。我們自己的資料沒有時間戳，量不了真延遲。

輸出：results/onset_latency.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
sys.path.insert(0, str(_SRC))
REPO_ROOT = _SRC.parent

from semg.core.softpinch_model import load_model
from semg.core import pipeline as pp
from semg.model_eval.run_model_validation import (
    MODEL_PATH, FS, RMS_STEP, RMS_WINDOW, BANDPASS, PRED_MAPPING,
    HAMPEL_HALF_WINDOW, HAMPEL_SIGMA,
    causal_notch_bandpass, causal_hampel, rms_stream,
)

RESULTS_ROOT = REPO_ROOT / "results"
SP_ROOT = REPO_ROOT / "src/SoftPINCH/src/experiment/data/subject_independent"

SOFTPINCH_NOTCH, SOFTPINCH_TRIM = 50, 3.0
TRIAL_SEC, PHASE_SEC = 9.0, 3.0
SEC_PER_PT = RMS_STEP / FS            # 0.025 s

STEP_SEC = 0.05                        # 量延遲要夠細
WINDOWS = [3.0, 2.0, 1.5, 1.0, 0.5]

FILES = [
    ("subject_8", "flex_index_finger_2026-02-19 10-50-44.csv", "Index"),
    ("subject_8", "flex_thumb_finger_2026-02-19 11-13-00.csv", "Thumb"),
]


def preprocess(csv_path: Path) -> np.ndarray:
    arr = pd.read_csv(csv_path).iloc[:, :3].values
    envs = []
    for c in range(3):
        x = arr[:, c].astype(np.float64)
        x = pp.trim(x, FS, SOFTPINCH_TRIM, SOFTPINCH_TRIM)
        x = x - x.mean()
        f = causal_notch_bandpass(x, FS, SOFTPINCH_NOTCH, BANDPASS)
        f = causal_hampel(f, HAMPEL_HALF_WINDOW, HAMPEL_SIGMA)
        envs.append(rms_stream(f, RMS_WINDOW, RMS_STEP))
    n = min(len(e) for e in envs)
    return np.stack([e[:n] for e in envs], axis=1)


def trailing_predict(env_n: np.ndarray, model, window_sec: float):
    """回傳 (時刻陣列, 預測標籤)。時刻 = 視窗的**結尾**（因果，real-time 能拿到的最新時間）。"""
    win = int(round(window_sec / SEC_PER_PT))
    step = max(1, int(round(STEP_SEC / SEC_PER_PT)))
    segs, t_end = [], []
    for s in range(0, len(env_n) - win + 1, step):
        segs.append(env_n[s:s + win])
        t_end.append((s + win) * SEC_PER_PT)      # trailing：視窗結尾才是「現在」
    with torch.no_grad():
        logits, _, _ = model(torch.tensor(np.stack(segs), dtype=torch.float32))
    return np.array(t_end), np.array([PRED_MAPPING[i] for i in logits.argmax(1).numpy()])


def main():
    RESULTS_ROOT.mkdir(exist_ok=True)
    model, _ = load_model(MODEL_PATH / "model.pth")
    model.eval()

    rows = []
    for subj, fname, target in FILES:
        env = preprocess(SP_ROOT / subj / "EMG" / fname)
        mu, sigma = env.mean(axis=0), env.std(axis=0)
        env_n = (env - mu) / (sigma + 1e-8)
        total_sec = len(env_n) * SEC_PER_PT
        n_trials = int(total_sec // TRIAL_SEC)
        print(f"\n[{subj}/{target}] 訊號 {total_sec:.0f}s，共 {n_trials} 個 trial", flush=True)

        for W in WINDOWS:
            t_end, labels = trailing_predict(env_n, model, W)
            want = f"{target} Contract"

            lats, hits = [], 0
            for k in range(n_trials):
                onset = k * TRIAL_SEC + PHASE_SEC          # contract 真實起點
                deadline = onset + PHASE_SEC               # contract 階段結束
                m = (t_end >= onset) & (t_end <= deadline)
                if not m.any():
                    continue
                seg_t, seg_l = t_end[m], labels[m]
                idx = np.where(seg_l == want)[0]
                if len(idx):
                    lats.append(seg_t[idx[0]] - onset)
                    hits += 1

            # 同時算這個視窗長度的手指判別率（排除 Rest），供 accuracy-latency 權衡
            act = labels[labels != "Rest"]
            facc = float(np.mean([l.startswith(target) for l in act])) if len(act) else np.nan

            med = float(np.median(lats)) if lats else np.nan
            p90 = float(np.percentile(lats, 90)) if lats else np.nan
            rows.append({"subject": subj, "target": target, "window_sec": W,
                          "n_trials": n_trials, "detect_rate": hits / max(n_trials, 1),
                          "latency_median_s": med, "latency_p90_s": p90,
                          "finger_accuracy": facc})
            print(f"    W={W:4.1f}s  偵測到 {hits}/{n_trials} 個 trial 的收縮  "
                  f"延遲中位數 {med if lats else float('nan'):.2f}s (p90 {p90 if lats else float('nan'):.2f}s)  "
                  f"手指判別率 {facc:.1%}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_ROOT / "onset_latency.csv", index=False)

    print("\n" + "=" * 72)
    print("延遲 vs 準確率權衡（兩個檔案平均；延遲從真實收縮起點算起）")
    print("=" * 72)
    g = df.groupby("window_sec").agg(偵測率=("detect_rate", "mean"),
                                       延遲中位數s=("latency_median_s", "mean"),
                                       延遲p90s=("latency_p90_s", "mean"),
                                       手指判別率=("finger_accuracy", "mean"))
    print(g.round(3).to_string())
    print("\n判讀：外骨骼控制一般可接受的延遲上限約 0.3 s（含致動）。")
    print(f"\n[完成] {RESULTS_ROOT/'onset_latency.csv'}")


if __name__ == "__main__":
    main()
