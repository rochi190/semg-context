"""
測試論文的 7 類 real-time checkpoint 在我們資料上的表現。

為什麼值得測：
    它多了 **Pinch** 類別（Index/Thumb/Pinch × Contract/Release + Rest），
    而我們自己也錄了 pinch/grasp——這是 offline 5 類模型做不到的。

已知的兩個不利條件（測之前就要說清楚，避免結果出來才找藉口）：
    1. 只用 subject_0 一個人訓練 → 對我們是跨受試者 + 跨硬體雙重挑戰
    2. 附的 mu/sigma 是 Delsys + subject_0 的**絕對 mV 值**，套 ADS1299 會偏移

所以正規化跑兩種，分開比較：
    A. checkpoint 附的 mu/sigma（論文原本的用法）
    B. per-file z-score（跟我們測 5 類模型時一致，公平比較）

輸入長度：這個 checkpoint 訓練時是 500 ms（11 個 RMS 點），
不是 offline 那個的 3 秒——見 docs/realtime-model-analysis.md §2.3。

輸出：results/realtime_7class_test.csv
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

from semg.core import pipeline as pp
from semg.core.softpinch_model import load_model
from semg.model_eval.run_model_validation import preprocess_for_model, RMS_STEP, FS

RESULTS = REPO_ROOT / "results"
RT_DIR = (REPO_ROOT / "src/SoftPINCH/src/models/loggings/"
          "Real_time_inference/SingleNet_CNN+LSTM_EMG/subject_0")

# 標籤順序：依 real_time_operation.py::Model.__init__（num_motions=3）
#   target = ['Index','Thumb','Pinch']，actions = ['Contract','Release']
#   編碼規則同 build_dataset_window_relabel：每個動作佔 2 格，rest 擺最後
PRED_7 = {0: "Index Contract", 1: "Index Release",
          2: "Thumb Contract", 3: "Thumb Release",
          4: "Pinch Contract", 5: "Pinch Release",
          6: "Rest"}

WINDOW_PTS = 11          # 500 ms —— 這個 checkpoint 的訓練長度
STEP_SEC = 0.1
SEC_PER_PT = RMS_STEP / FS


def sliding(env_n, model, step_sec=STEP_SEC):
    step = max(1, int(round(step_sec / SEC_PER_PT)))
    segs, ends = [], []
    for s in range(0, len(env_n) - WINDOW_PTS + 1, step):
        segs.append(env_n[s:s + WINDOW_PTS])
        ends.append((s + WINDOW_PTS) * SEC_PER_PT)
    with torch.no_grad():
        logits, _, _ = model(torch.tensor(np.stack(segs), dtype=torch.float32))
        prob = torch.softmax(logits, dim=1)
        idx = logits.argmax(1).numpy()
    return (np.array(ends),
            np.array([PRED_7[i] for i in idx]),
            prob.max(1).values.numpy())


def main():
    model, ckpt = load_model(RT_DIR / "model.pth")
    model.eval()
    mu_paper = np.load(RT_DIR / "mu.npy")
    sigma_paper = np.load(RT_DIR / "sigma.npy")
    print(f"7 類 checkpoint: output_dim={ckpt['model_args']['output_dim']}, "
          f"參數量={sum(p.numel() for p in model.parameters()):,}")
    print(f"論文 mu={np.round(mu_paper,5)}  sigma={np.round(sigma_paper,5)}\n")

    files = []
    for subj in ("gary", "neil1", "CBW1"):
        for p in sorted((REPO_ROOT / "data" / subj).glob("*.csv")):
            for post in ("index", "thumb", "pinch", "grasp"):
                if p.stem.startswith(post):
                    files.append((subj, p, post))

    rows = []
    for subj, path, posture in files:
        df = pp.load_channel(str(path), channels=(1, 2, 3))
        env = preprocess_for_model({c: df[f"ch{c}"].values for c in (1, 2, 3)})
        # 我們的資料量級跟 Delsys 不同，先看看差多少
        scale = float(np.median(env) / (np.median(mu_paper) + 1e-12))

        for norm, (m, s) in (("paper_mu_sigma", (mu_paper, sigma_paper)),
                              ("per_file_zscore", (env.mean(0), env.std(0)))):
            env_n = (env - m) / (s + 1e-8)
            ends, lab, conf = sliding(env_n, model)
            act = lab[lab != "Rest"]
            row = {"subject": subj, "file": path.stem, "posture": posture,
                   "norm": norm, "n_windows": len(lab),
                   "amp_ratio_vs_paper": scale,
                   "frac_rest": float((lab == "Rest").mean()),
                   "mean_conf": float(conf.mean())}
            # 各動作被判到的比例（排除 Rest）
            for mo in ("Index", "Thumb", "Pinch"):
                row[f"frac_{mo.lower()}"] = (float(np.mean([l.startswith(mo) for l in act]))
                                              if len(act) else np.nan)
            # 對 index/thumb/pinch 三種姿勢，正確動作被判到的比例
            tgt = {"index": "Index", "thumb": "Thumb", "pinch": "Pinch"}.get(posture)
            row["correct_motion_frac"] = (row[f"frac_{tgt.lower()}"] if tgt else np.nan)
            rows.append(row)

    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "realtime_7class_test.csv", index=False)

    print("=== 動作判別率（排除 Rest 後，判到正確動作的比例；亂猜=33%）===")
    for norm in ("paper_mu_sigma", "per_file_zscore"):
        sub = d[(d.norm == norm) & d.correct_motion_frac.notna()]
        print(f"\n[{norm}]  平均 Rest 佔比 {sub.frac_rest.mean():.1%}")
        for post in ("index", "thumb", "pinch"):
            g = sub[sub.posture == post]
            if len(g):
                print(f"  {post:6s} (n={len(g):2d})  正確動作 {g.correct_motion_frac.mean():6.1%}   "
                      f"Index {g.frac_index.mean():5.1%} / Thumb {g.frac_thumb.mean():5.1%} / "
                      f"Pinch {g.frac_pinch.mean():5.1%}")
        print(f"  {'整體':6s}          正確動作 {sub.correct_motion_frac.mean():6.1%}")

    print(f"\n我們的資料振幅 / 論文 mu 的比值（中位數）: "
          f"{d.amp_ratio_vs_paper.median():.2f}x")
    print(f"\n[完成] {RESULTS/'realtime_7class_test.csv'}")


if __name__ == "__main__":
    main()
