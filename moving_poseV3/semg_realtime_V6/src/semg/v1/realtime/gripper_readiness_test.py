"""
夾爪控制就緒度評估：這個模型今天能不能拿去驅動 Lite 6 的夾爪？

為什麼需要這支：
    專案目標（CLAUDE.md §3）第一路是「前臂 sEMG → CNN+LSTM → **夾爪開合**」。
    夾爪只需要三態（出力/放鬆/休息），**不需要分食指還是拇指**——
    我們先前一路在量的 84% 手指判別率，對這個階段其實用不到。

    真正決定能不能驅動夾爪的是另外三件事，而我們從來沒量過：
      A. 反應延遲：使用者出力後多久夾爪才動
      B. 誤觸發率：休息時模型說 Contract 的比例（夾爪亂夾很危險）
      C. 輸出穩定度：持續出力期間輸出跳動的次數（會導致夾爪震盪）

怎麼在沒有時間戳的情況下量：
    用 RMS 包絡線自己的上升當客觀基準。包絡線超過基線一定幅度 = 肌肉真的開始出力，
    這是訊號本身的事實，不依賴我們猜的 trial 邊界（那正是先前踩坑的地方）。

同時跑一個對照組：
    論文 real-time 模型的標籤其實是一條規則（classification_pipeline.py:2057）——
        activity < REST_THRESH → rest；slope > 0 → contract；否則 release
    所以「不用 ML、直接用門檻+斜率」是一個合理的基準線。如果它跟 CNN+LSTM 差不多，
    那夾爪這條路根本不需要深度模型。

輸出：results/gripper_readiness.csv
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
    MODEL_PATH, FS, RMS_STEP, PRED_MAPPING, preprocess_for_model,
)

RESULTS_ROOT = REPO_ROOT / "results"
SEC_PER_PT = RMS_STEP / FS          # 0.025 s

WINDOW_SEC = 3.0                     # 模型的訓練長度，實測顯示不該縮短
STEP_SEC = 0.1                       # 夾爪控制的更新率 10 Hz

# 包絡線 onset 判定：超過「基線 + k × 穩健標準差」並持續一段時間
ONSET_K = 3.0
ONSET_MIN_SEC = 0.3                  # 持續這麼久才算真的出力（濾掉單點雜訊）
QUIET_K = 1.0                        # 低於這條線視為「安靜期」，用來算誤觸發


def collapse(label: str) -> str:
    """5 類 → 夾爪需要的三態。夾爪不在乎是食指還是拇指。"""
    if label == "Rest":
        return "Rest"
    return "Contract" if "Contract" in label else "Release"


def envelope_events(env_ch: np.ndarray):
    """用包絡線自己定義客觀的出力起點與安靜期（不依賴任何猜測的邊界）。"""
    base = np.median(env_ch)
    mad = np.median(np.abs(env_ch - base)) * 1.4826 + 1e-12
    hi = base + ONSET_K * mad
    lo = base + QUIET_K * mad

    active = env_ch > hi
    min_pts = int(round(ONSET_MIN_SEC / SEC_PER_PT))

    onsets, i = [], 0
    while i < len(active):
        if active[i]:
            j = i
            while j < len(active) and active[j]:
                j += 1
            if j - i >= min_pts:
                onsets.append(i)
            i = j
        else:
            i += 1
    quiet_mask = env_ch < lo
    return np.array(onsets), quiet_mask


def rule_predict(env: np.ndarray, idx: int, win_pts: int):
    """論文 real-time 的標籤規則，改成尺度不變版（用該檔自身穩健統計量取代絕對 mV 門檻）。"""
    seg = env[idx - win_pts:idx]
    m = seg.mean(axis=1)
    base = np.median(env.mean(axis=1))
    mad = np.median(np.abs(env.mean(axis=1) - base)) * 1.4826 + 1e-12
    if m.mean() < base + 1.0 * mad:
        return "Rest"
    return "Contract" if np.mean(np.diff(m)) > 0 else "Release"


def analyse(name, subject, posture, env, model):
    ch = 1 if posture == "index" else 2
    onsets, quiet = envelope_events(env[:, ch])
    if len(onsets) < 3:
        return None

    mu, sigma = env.mean(axis=0), env.std(axis=0)
    env_n = (env - mu) / (sigma + 1e-8)
    win = int(round(WINDOW_SEC / SEC_PER_PT))
    step = max(1, int(round(STEP_SEC / SEC_PER_PT)))

    ends, segs = [], []
    for s in range(0, len(env_n) - win + 1, step):
        segs.append(env_n[s:s + win])
        ends.append(s + win)          # trailing：視窗結尾才是「現在」
    ends = np.array(ends)
    with torch.no_grad():
        logits, _, _ = model(torch.tensor(np.stack(segs), dtype=torch.float32))
    model_lab = np.array([collapse(PRED_MAPPING[i]) for i in logits.argmax(1).numpy()])
    rule_lab = np.array([rule_predict(env, e, win) for e in ends])

    out = {}
    for tag, lab in (("model", model_lab), ("rule", rule_lab)):
        # A. 偵測率與延遲
        lats, hits = [], 0
        for o in onsets:
            m = (ends >= o) & (ends <= o + int(2.0 / SEC_PER_PT))   # 出力後 2 秒內要抓到
            if not m.any():
                continue
            k = np.where(lab[m] == "Contract")[0]
            if len(k):
                hits += 1
                lats.append((ends[m][k[0]] - o) * SEC_PER_PT)
        # B. 誤觸發：安靜期內模型說 Contract 的比例
        q = quiet[np.clip(ends - 1, 0, len(quiet) - 1)]
        false_act = float((lab[q] == "Contract").mean()) if q.any() else np.nan
        # C. 穩定度：輸出切換率
        sw = float((lab[1:] != lab[:-1]).mean())
        out[tag] = dict(detect_rate=hits / len(onsets),
                         latency_median=float(np.median(lats)) if lats else np.nan,
                         false_activation=false_act, switch_rate=sw)

    return {"subject": subject, "file": name, "posture": posture,
            "n_onsets": len(onsets),
            **{f"{t}_{k}": v for t, d in out.items() for k, v in d.items()}}


def main():
    RESULTS_ROOT.mkdir(exist_ok=True)
    model, _ = load_model(MODEL_PATH / "model.pth")
    model.eval()

    files = []
    for subj in ("gary", "neil1", "CBW1"):
        for p in sorted((REPO_ROOT / "data" / subj).glob("*.csv")):
            for post in ("index", "thumb"):
                if p.stem.startswith(post):
                    files.append((subj, p, post))

    rows = []
    for subj, path, posture in files:
        df = pp.load_channel(str(path), channels=(1, 2, 3))
        env = preprocess_for_model({c: df[f"ch{c}"].values for c in (1, 2, 3)})
        r = analyse(path.stem, subj, posture, env, model)
        if r is None:
            print(f"  跳過 {path.stem}（偵測到的出力事件太少）")
            continue
        rows.append(r)
        print(f"  {subj:6s} {posture:6s} {path.stem[:30]:32s} "
              f"模型: 偵測{r['model_detect_rate']:5.0%} 延遲{r['model_latency_median']:.2f}s "
              f"誤觸發{r['model_false_activation']:5.1%} 切換{r['model_switch_rate']:5.1%}  |  "
              f"規則: 偵測{r['rule_detect_rate']:5.0%} 誤觸發{r['rule_false_activation']:5.1%}",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_ROOT / "gripper_readiness.csv", index=False)

    print("\n" + "=" * 70)
    print("夾爪控制就緒度總結（14 個前臂 sEMG 檔案）")
    print("=" * 70)
    for tag, zh in (("model", "CNN+LSTM (5類摺成3態)"), ("rule", "門檻+斜率規則（無 ML）")):
        print(f"\n{zh}")
        print(f"  出力偵測率       {df[f'{tag}_detect_rate'].mean():6.1%}   （越高越好）")
        if f"{tag}_latency_median" in df:
            print(f"  偵測延遲中位數   {df[f'{tag}_latency_median'].median():6.2f}s  （相對包絡線上升）")
        print(f"  休息誤觸發率     {df[f'{tag}_false_activation'].mean():6.1%}   （越低越好，夾爪安全關鍵）")
        print(f"  輸出切換率       {df[f'{tag}_switch_rate'].mean():6.1%}   （越低越穩，避免夾爪震盪）")
    print(f"\n[完成] {RESULTS_ROOT/'gripper_readiness.csv'}")


if __name__ == "__main__":
    main()
