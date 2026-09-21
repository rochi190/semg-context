"""
校準的正規化方式對「休息/出力比例」有多敏感？有沒有更穩健的做法？

問題來源（使用者提出，很合理）：
    z-score 的 mu/sigma 取自校準期整段。
    如果休息時間拉長，mu 會下降、sigma 會變小 → 正規化後的數值整個位移，
    等於同一個動作在不同校準條件下餵給模型的數字不一樣。這確實不穩。

測試：
    把校準視窗的「休息佔比」從 30% 調到 90%，看準確率變動多少。
    同時比較四種正規化方式，看哪個對組成比例最不敏感：

      A. z-score            mu=mean, sigma=std              ← 目前用的
      B. 穩健 z-score        mu=median, sigma=1.4826*MAD     ← 中位數對組成較不敏感
      C. 基線-動態範圍       mu=p5,     sigma=p95-p5          ← 完全不看組成比例
      D. 只用休息段          mu,sigma 取自安靜區間            ← 定義最明確

判準：
    敏感度 = 不同休息佔比之間的準確率**標準差**，越小越穩健。
    但也要看**平均準確率**——穩健卻普遍很差沒有意義。

輸出：results/normalization_robustness.csv
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
from semg.model_eval.run_model_validation import (
    MODEL_PATH, RMS_STEP, FS, PRED_MAPPING, preprocess_for_model,
)

RESULTS = REPO_ROOT / "results"
SEC_PER_PT = RMS_STEP / FS
WIN_PTS, STEP_SEC = 120, 0.1
CALIB_PTS = int(45 / SEC_PER_PT)          # 校準期長度固定 45 秒

FILES = [
    ("gary",  "thumbgary_20260716_17-12-05_motion2", "Thumb"),
    ("neil1", "indexlin_20260716_21-58-56_motion1",  "Index"),
    ("neil1", "thumblin_20260716_22-02-17_motion2",  "Thumb"),
    ("neil1", "indexlin_20260720_20-51-59",          "Index"),
]

REST_FRACS = [0.3, 0.5, 0.7, 0.9]         # 校準期裡休息佔多少


def stats_for(method: str, cal: np.ndarray, env_full: np.ndarray):
    """回傳 (mu, sigma)。各方法都只用校準期資料，不偷看未來。"""
    if method == "A_zscore":
        return cal.mean(0), cal.std(0)
    if method == "B_robust_zscore":
        med = np.median(cal, axis=0)
        mad = np.median(np.abs(cal - med), axis=0) * 1.4826
        return med, mad
    if method == "C_baseline_range":
        p5 = np.percentile(cal, 5, axis=0)
        p95 = np.percentile(cal, 95, axis=0)
        return p5, (p95 - p5)
    if method == "D_rest_only":
        # 用校準期中最安靜的 40% 當基線
        act = cal.mean(axis=1)
        thr = np.percentile(act, 40)
        quiet = cal[act <= thr]
        return quiet.mean(0), quiet.std(0)
    raise ValueError(method)


def evaluate(env, mu, sigma, model, target):
    x = (env - mu) / (sigma + 1e-8)
    segs = np.stack([x[s:s + WIN_PTS]
                     for s in range(0, len(x) - WIN_PTS + 1, int(STEP_SEC / SEC_PER_PT))])
    with torch.no_grad():
        logits, _, _ = model(torch.tensor(segs, dtype=torch.float32))
    lab = np.array([PRED_MAPPING[i] for i in logits.argmax(1).numpy()])
    act = lab[lab != "Rest"]
    return (float(np.mean([l.startswith(target) for l in act])) if len(act) else np.nan,
            float((lab != "Rest").mean()))


def build_calib(env: np.ndarray, ch: int, rest_frac: float, rng) -> np.ndarray:
    """組出一段指定休息佔比的校準資料（從該檔案的安靜/活動區間各自取樣）。"""
    a = env[:, ch]
    base = np.median(a)
    mad = np.median(np.abs(a - base)) * 1.4826 + 1e-12
    quiet_idx = np.where(a < base + 1.0 * mad)[0]
    act_idx = np.where(a > base + 3.0 * mad)[0]
    if len(quiet_idx) < 100 or len(act_idx) < 100:
        return None
    n_rest = int(CALIB_PTS * rest_frac)
    n_act = CALIB_PTS - n_rest
    qi = rng.choice(quiet_idx, size=n_rest, replace=len(quiet_idx) < n_rest)
    ai = rng.choice(act_idx, size=n_act, replace=len(act_idx) < n_act)
    return env[np.sort(np.concatenate([qi, ai]))]


def main():
    model, _ = load_model(MODEL_PATH / "model.pth")
    model.eval()
    rng = np.random.default_rng(0)
    methods = ["A_zscore", "B_robust_zscore", "C_baseline_range", "D_rest_only"]

    rows = []
    for subj, stem, target in FILES:
        df = pp.load_channel(f"{REPO_ROOT}/data/{subj}/{stem}.csv", channels=(1, 2, 3))
        env = preprocess_for_model({c: df[f"ch{c}"].values for c in (1, 2, 3)})
        ch = 1 if target == "Index" else 2
        for rf in REST_FRACS:
            cal = build_calib(env, ch, rf, rng)
            if cal is None:
                continue
            for m in methods:
                mu, sg = stats_for(m, cal, env)
                acc, actrate = evaluate(env, mu, sg, model, target)
                rows.append({"subject": subj, "file": stem, "target": target,
                              "rest_frac": rf, "method": m,
                              "acc": acc, "active_rate": actrate})
        print(f"  {subj}/{stem[:28]} 完成", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "normalization_robustness.csv", index=False)

    print("\n" + "=" * 78)
    print("各正規化方式在不同「休息佔比」下的手指判別率")
    print("=" * 78)
    print(f"{'方法':<20s}" + "".join(f"{int(r*100):>7d}%" for r in REST_FRACS)
          + f"{'平均':>9s}{'標準差':>9s}")
    print("-" * 78)
    summary = []
    for m in methods:
        g = d[d.method == m]
        per = [g[g.rest_frac == r].acc.mean() for r in REST_FRACS]
        mean_, std_ = np.nanmean(per), np.nanstd(per)
        summary.append((m, mean_, std_))
        print(f"{m:<20s}" + "".join(f"{v:>7.1%}" for v in per)
              + f"{mean_:>9.1%}{std_:>9.1%}")

    print("\n判讀：標準差越小 = 對「休息多久」越不敏感；但平均要夠高才有意義")
    best_stable = min(summary, key=lambda z: z[2])
    best_acc = max(summary, key=lambda z: z[1])
    print(f"  最穩健：{best_stable[0]}（標準差 {best_stable[2]:.1%}）")
    print(f"  最準確：{best_acc[0]}（平均 {best_acc[1]:.1%}）")
    print(f"\n[完成] {RESULTS/'normalization_robustness.csv'}")


if __name__ == "__main__":
    main()
