"""
傳統特徵基準線 v4 —— 修正絕對振幅特徵帶硬體增益的問題。

為什麼還要 v4（v3 的實證結果）：
    v3 修好了跨通道正規化（Cohen's d 從 0.12 回到 3.88 的那個問題），
    但實測仍然「整檔只輸出一類、切換 0 次」——代表跨通道特徵不是主導因素。
    剩下的嫌疑：MAV / RMS / WL / VAR 這四個**絕對振幅**特徵。
    實測我們的資料是論文的 1.6–2.3 倍（不同電極 + 不同增益），
    而 StandardScaler 是用論文資料配的 → 這些特徵一進來就整體偏移，
    且在同一檔案內接近常數 → LDA 讀到近似常數的輸入 → 整檔一個答案。

v4 的修正：**逐檔案、跨通道共用單一尺度**做正規化

    λ = median_c ( 1.4826 × MAD(x_c) )        ← 整個檔案、三通道共用的一個純量

    x̃_c[n] = x_c[n] / λ

    這個設計同時滿足三個互相拉扯的需求：
      1. 消除硬體/受試者的絕對增益差異（λ 每檔一個值）
      2. **保留檔案內的振幅動態**（rest vs contract 的差異還在）
         —— 這是為什麼不能逐視窗正規化：那會把 Rest/Contract 的差異也消掉
      3. **保留通道間相對關係**（三通道共用同一個 λ）
         —— 這是為什麼不能逐通道正規化：那就是 v2 犯的錯

    對照：CNN+LSTM 吃的是逐通道 z-score。它能容忍逐通道正規化，是因為它靠
    時序波形形態判別；傳統特徵靠振幅比值，所以必須用共用尺度。

其餘特徵定義與 v3 完全相同（單一變因）。ZC/SSC/MNF/MDF/AR 本來就尺度不變，
不受影響；受影響的是 MAV/RMS/logRMS/WL/VAR。

輸出：results/traditional_v4_by_file.csv, results/traditional_v4_summary.csv
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
    preprocess_full, discover_our_files,
    WINDOW_SEC, STEP_SEC, SEC_PER_PT, PTS_PER_EPOCH, SEG_LEN,
    TRAIN_SUBJECTS, SP_ROOT,
)
from semg.traditional.traditional_baseline_v3 import window_features_v3

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "src").parent
RESULTS_ROOT = REPO_ROOT / "results"


def shared_scale(filt: np.ndarray) -> float:
    """整檔、跨三通道共用的單一尺度 λ（用 MAD 以抵抗脈衝雜訊）。"""
    sig = [1.4826 * np.median(np.abs(filt[:, c] - np.median(filt[:, c]))) for c in range(3)]
    lam = float(np.median(sig))
    return lam if lam > 1e-12 else 1.0


def normalize_shared(filt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """回傳 (正規化後的訊號, 各通道死區門檻 eps)。
    正規化後訊號的尺度已統一，eps 可用固定值（相對於正規化後的單位）。"""
    lam = shared_scale(filt)
    x = filt / lam
    # 正規化後，1 個單位 ≈ 1 個穩健標準差，死區取 5%
    eps = np.full(3, 0.05)
    return x, eps


def build_training_set_v4():
    X, y = [], []
    for subj in TRAIN_SUBJECTS:
        emg_dir = SP_ROOT / subj / "EMG"
        if not emg_dir.exists():
            continue
        for posture in ("index", "thumb"):
            target = "Index" if posture == "index" else "Thumb"
            for csv_path in sorted(emg_dir.glob(f"flex_{posture}_finger_*.csv")):
                filt, env = preprocess_full(csv_path, "softpinch")
                xn, eps = normalize_shared(filt)
                for e in range(len(env) // PTS_PER_EPOCH):
                    for k, lab in enumerate(["Rest", f"{target} Contract", f"{target} Release"]):
                        p0 = e * PTS_PER_EPOCH + k * SEG_LEN
                        s0, s1 = int(p0 * RMS_STEP), int((p0 + SEG_LEN) * RMS_STEP)
                        if s1 > len(xn):
                            continue
                        X.append(window_features_v3(xn[s0:s1], eps))
                        y.append(lab)
        print(f"    {subj} 完成，累積 {len(X)} 筆", flush=True)
    return np.array(X), np.array(y)


def main():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    print("=== 1. 建立訓練集（v4：跨通道共用尺度正規化）===", flush=True)
    Xtr, ytr = build_training_set_v4()
    print(f"  {len(Xtr)} 筆 × {Xtr.shape[1]} 維")
    lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    lda.fit(Xtr, ytr)
    print(f"  LDA 訓練集自身準確率（非驗證）: {lda.score(Xtr, ytr):.1%}", flush=True)

    model, _ = load_model(MODEL_PATH / "model.pth")
    model.eval()

    files = discover_our_files() + [
        ("SP_subject_8", SP_ROOT / "subject_8/EMG/flex_index_finger_2026-02-19 10-50-44.csv",
         "index", "softpinch"),
        ("SP_subject_8", SP_ROOT / "subject_8/EMG/flex_thumb_finger_2026-02-19 11-13-00.csv",
         "thumb", "softpinch"),
    ]

    print(f"\n=== 2. zero-shot 跑 {len(files)} 個檔案 ===", flush=True)
    win_pts = int(round(WINDOW_SEC / SEC_PER_PT))
    step_pts = int(round(STEP_SEC / SEC_PER_PT))
    rows = []

    for subject, path, posture, source in files:
        filt, env = preprocess_full(path, source)
        xn, eps = normalize_shared(filt)
        target = "Index" if posture == "index" else "Thumb"
        mu, sigma = env.mean(axis=0), env.std(axis=0)
        env_n = (env - mu) / (sigma + 1e-8)

        segs, feats = [], []
        for s in range(0, len(env_n) - win_pts + 1, step_pts):
            a0, a1 = int(s * RMS_STEP), int((s + win_pts) * RMS_STEP)
            if a1 > len(xn):
                break
            segs.append(env_n[s:s + win_pts])
            feats.append(window_features_v3(xn[a0:a1], eps))
        if not segs:
            continue

        with torch.no_grad():
            logits, _, _ = model(torch.tensor(np.stack(segs), dtype=torch.float32))
        m_lab = np.array([PRED_MAPPING[i] for i in logits.argmax(1).numpy()])
        t_lab = lda.predict(np.stack(feats))

        def facc(lab):
            act = lab[lab != "Rest"]
            return float(np.mean([l.startswith(target) for l in act])) if len(act) else np.nan

        t_sw = int((t_lab[1:] != t_lab[:-1]).sum())
        rows.append({"subject": subject, "file_stem": path.stem, "posture": posture,
                      "source": source, "n_windows": len(m_lab),
                      "cnn_lstm": facc(m_lab), "traditional_v4": facc(t_lab),
                      "trad_n_unique": int(len(set(t_lab))), "trad_switches": t_sw,
                      "trad_switch_rate": t_sw / max(len(t_lab) - 1, 1)})
        print(f"  {subject:12s} {posture:6s} {path.stem[:30]:32s} "
              f"CNN {rows[-1]['cnn_lstm']:6.1%}  傳統v4 {rows[-1]['traditional_v4']:6.1%}  "
              f"(輸出{rows[-1]['trad_n_unique']}類/切換{t_sw})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_ROOT / "traditional_v4_by_file.csv", index=False)

    print("\n=== 3. 分組彙總（視窗數加權）===")
    summ = []
    for grp, g in df.groupby(df["source"].map({"ours": "我們的資料", "softpinch": "論文held-out"})):
        w = g["n_windows"]
        summ.append({"group": grp, "n_files": len(g), "n_windows": int(w.sum()),
                      "cnn_lstm": float((g.cnn_lstm * w).sum() / w.sum()),
                      "traditional_v4": float((g.traditional_v4 * w).sum() / w.sum())})
    sdf = pd.DataFrame(summ)
    sdf.to_csv(RESULTS_ROOT / "traditional_v4_summary.csv", index=False)
    print(sdf.to_string(index=False))

    ours = df[df.source == "ours"]
    print(f"\n=== 4. 「整檔一個答案」是否修好 ===")
    print(f"  v4 整檔只輸出單一類別: {int((ours.trad_n_unique <= 1).sum())}/{len(ours)} 檔"
          f"   （v2 是 6/14）")
    print(f"  v4 平均切換率: {ours.trad_switch_rate.mean():.3f} 次/視窗"
          f"   （v2 是 0.027，CNN+LSTM 是 0.291）")
    for tag, col in (("v2", "traditional_v2"), ("v3", "traditional_v3")):
        p = RESULTS_ROOT / f"traditional_{tag}_by_file.csv"
        if p.exists():
            prev = pd.read_csv(p)[["file_stem", col]]
            m = ours.merge(prev, on="file_stem")
            w = m.n_windows
            print(f"  我們的資料加權平均 {tag}: {float((m[col]*w).sum()/w.sum()):.1%}")
    w = ours.n_windows
    print(f"  我們的資料加權平均 v4: {float((ours.traditional_v4*w).sum()/w.sum()):.1%}")
    print(f"  我們的資料加權平均 CNN+LSTM: {float((ours.cnn_lstm*w).sum()/w.sum()):.1%}")

    print(f"\n[完成] {RESULTS_ROOT/'traditional_v4_by_file.csv'}")


if __name__ == "__main__":
    main()
