"""
驗證假設：傳統 v2 的 0%/100% 雙峰，是因為分類器讀到「檔案/受試者層級的常數」
而不是「視窗內容」。

不靠推論，用兩個可證偽的檢驗：

  檢驗1（輸出切換次數）：若分類器對整個檔案給同一個答案，則同檔案內的預測
      類別切換次數會趨近 0。拿 CNN+LSTM 當對照組（它的準確率是連續分布，
      應該有正常的切換次數）。
      → 若傳統切換次數 ≈ 0 而 CNN 正常，假設成立。

  檢驗2（變異數分解）：對跨通道 log 比值特徵，計算
        檔案內變異 (within-file variance) vs 檔案間變異 (between-file variance)
      若 between >> within，代表這個特徵幾乎只帶「檔案身分」資訊，
      不帶「視窗內容」資訊 → 分類器只能做出檔案層級的決定。
      ICC = between / (between + within)，越接近 1 越是「檔案指紋」。

輸出：results/traditional_failure_diagnosis.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")   # src/ 目錄，與檔案放在多深無關
sys.path.insert(0, str(_SRC))
from semg.core.softpinch_model import load_model
from semg.model_eval.run_model_validation import MODEL_PATH, FS, RMS_STEP, PRED_MAPPING
from semg.traditional.traditional_baseline_v2 import (
    preprocess_full, file_scales, window_features_v2, build_training_set,
    discover_our_files, WINDOW_SEC, STEP_SEC, SEC_PER_PT, SP_ROOT,
)

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "src").parent
RESULTS_ROOT = REPO_ROOT / "results"

# 跨通道特徵在 54 維向量中的位置：每通道 16 維 × 3 = 48，之後是
# [log α1-α2, log α1-α0, log α2-α0, p0, p1, p2]
IDX_LOGRATIO = [48, 49, 50]
LOGRATIO_NAME = ["log(idx/thumb)", "log(idx/flexor)", "log(thumb/flexor)"]


def main():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    print("=== 重建訓練集與 LDA（與 v2 相同）===", flush=True)
    Xtr, ytr = build_training_set()
    lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    lda.fit(Xtr, ytr)

    model, _ = load_model(MODEL_PATH / "model.pth")
    model.eval()

    files = discover_our_files()
    win_pts = int(round(WINDOW_SEC / SEC_PER_PT))
    step_pts = int(round(STEP_SEC / SEC_PER_PT))

    rows, feat_records = [], []
    print("\n=== 檢驗1：同檔案內的預測切換次數 ===", flush=True)
    for subject, path, posture, source in files:
        filt, env = preprocess_full(path, source)
        eps, base = file_scales(filt)
        mu, sigma = env.mean(axis=0), env.std(axis=0)
        env_n = (env - mu) / (sigma + 1e-8)

        segs, feats = [], []
        for s in range(0, len(env_n) - win_pts + 1, step_pts):
            a0, a1 = int(s * RMS_STEP), int((s + win_pts) * RMS_STEP)
            if a1 > len(filt):
                break
            segs.append(env_n[s:s + win_pts])
            feats.append(window_features_v2(filt[a0:a1], eps, base))
        if not segs:
            continue
        F = np.stack(feats)

        with torch.no_grad():
            logits, _, _ = model(torch.tensor(np.stack(segs), dtype=torch.float32))
        m_lab = np.array([PRED_MAPPING[i] for i in logits.argmax(1).numpy()])
        t_lab = lda.predict(F)

        m_sw = int((m_lab[1:] != m_lab[:-1]).sum())
        t_sw = int((t_lab[1:] != t_lab[:-1]).sum())
        n = len(t_lab)
        rows.append({"subject": subject, "file_stem": path.stem, "posture": posture,
                      "n_windows": n,
                      "cnn_switches": m_sw, "trad_switches": t_sw,
                      "cnn_switch_rate": m_sw / max(n - 1, 1),
                      "trad_switch_rate": t_sw / max(n - 1, 1),
                      "trad_n_unique": int(len(set(t_lab))),
                      "cnn_n_unique": int(len(set(m_lab)))})
        print(f"  {subject:6s} {posture:6s} {path.stem[:32]:34s} "
              f"CNN 切換 {m_sw:4d} ({m_sw/max(n-1,1):.2f}/視窗, {len(set(m_lab))} 類)  |  "
              f"傳統 切換 {t_sw:4d} ({t_sw/max(n-1,1):.2f}/視窗, {len(set(t_lab))} 類)", flush=True)

        for j, k in enumerate(IDX_LOGRATIO):
            feat_records.append(pd.DataFrame({
                "file": path.stem, "subject": subject, "posture": posture,
                "feature": LOGRATIO_NAME[j], "value": F[:, k]}))

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_ROOT / "traditional_failure_diagnosis.csv", index=False)

    print("\n  [檢驗1 結論]")
    print(f"    CNN+LSTM 平均切換率 : {df.cnn_switch_rate.mean():.3f} 次/視窗，"
          f"平均輸出 {df.cnn_n_unique.mean():.1f} 種類別")
    print(f"    傳統 v2  平均切換率 : {df.trad_switch_rate.mean():.3f} 次/視窗，"
          f"平均輸出 {df.trad_n_unique.mean():.1f} 種類別")
    n_const = int((df.trad_n_unique <= 1).sum())
    print(f"    傳統 v2 有 {n_const}/{len(df)} 個檔案『整檔只輸出單一類別』")

    print("\n=== 檢驗2：log 比值特徵的變異數分解（ICC）===")
    FR = pd.concat(feat_records, ignore_index=True)
    out2 = []
    for fname, g in FR.groupby("feature"):
        grand = g.value.mean()
        # between-file：各檔平均值的變異（以檔案樣本數加權）
        gm = g.groupby("file").value.agg(["mean", "count"])
        between = float((gm["count"] * (gm["mean"] - grand) ** 2).sum() / g.shape[0])
        # within-file：各檔內部的變異
        within = float(g.groupby("file").value.apply(lambda v: ((v - v.mean()) ** 2).sum()).sum()
                        / g.shape[0])
        icc = between / (between + within + 1e-20)
        out2.append({"feature": fname, "between_file_var": between,
                      "within_file_var": within, "ICC": icc})
        print(f"  {fname:20s}  between={between:8.4f}  within={within:8.4f}  ICC={icc:.3f}")
    pd.DataFrame(out2).to_csv(RESULTS_ROOT / "traditional_logratio_icc.csv", index=False)

    print("\n  [檢驗2 判讀] ICC 越接近 1，代表該特徵幾乎只帶『檔案身分』、不帶『視窗內容』。")
    print(f"\n[完成] {RESULTS_ROOT/'traditional_failure_diagnosis.csv'}")


if __name__ == "__main__":
    main()
