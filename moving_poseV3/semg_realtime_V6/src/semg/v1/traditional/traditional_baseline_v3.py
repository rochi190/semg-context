"""
傳統特徵基準線 v3 —— 修正 v2 的跨通道正規化錯誤。

三個版本的跨通道特徵演進（每次修正都有實證依據，不是憑感覺調）：

  v1: 對 **z-score 後**的包絡線取比值 a1/a2。
      問題：z-score 後整檔均值為 0，分母可趨近 0 → 比值爆炸 → LDA 被離群值帶偏。
      症狀：某檔案掉到 47%（低於亂猜）。

  v2: 先用「該通道整檔的視窗-RMS 中位數」b_c 做正規化，再取 log 比值。
      問題：**逐通道**正規化把每個通道都拉回自己的基線，
            「哪個通道比較活躍」的資訊被一起抹平。
      實證：同一人同一天的 index 檔 vs thumb 檔，
            p_index 的 Cohen's d 從 1.38 掉到 0.17，p_thumb 從 3.88 掉到 0.12。
      症狀：預測退化成「整檔一個答案」→ 準確率呈 0%/100% 雙峰。

  v3（本檔）: 直接用**視窗內**三通道 RMS 的總和歸一化比例：
            p_c = a_c / Σ_j a_j
      - 除以總和 → 全域增益自動消掉（尺度不變）
      - 保留通道間相對關係 → 判別資訊完整保留
      - 有界於 [0,1] → 不會有離群值
      - 三個正數相加不可能趨近 0 → 無除零風險
      另外保留 log 比值 log(a_i/a_j)（同樣尺度不變）。

除了跨通道那 6 維，**其餘 48 維（每通道時域/頻域/AR）與 v2 完全相同**——
刻意只改一個變因，才能歸因。

輸出：results/traditional_v3_by_file.csv, results/traditional_v3_summary.csv
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
    preprocess_full, file_scales, channel_features, discover_our_files,
    WINDOW_SEC, STEP_SEC, SEC_PER_PT, PTS_PER_EPOCH, SEG_LEN,
    TRAIN_SUBJECTS, SP_ROOT,
)

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "src").parent
RESULTS_ROOT = REPO_ROOT / "results"


def window_features_v3(win_emg: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """與 v2 唯一的差別：跨通道改用視窗內總和歸一化，不再除以逐通道基線。"""
    feats = []
    for c in range(3):
        feats += channel_features(win_emg[:, c], eps[c])

    a = np.array([np.sqrt(np.mean(win_emg[:, c] ** 2)) for c in range(3)]) + 1e-12
    la = np.log(a)
    feats += [la[1] - la[2], la[1] - la[0], la[2] - la[0]]   # log 比值（尺度不變）
    feats += list(a / a.sum())                                 # 比例，有界且尺度不變
    return np.array(feats, dtype=np.float64)


def build_training_set_v3():
    X, y = [], []
    for subj in TRAIN_SUBJECTS:
        emg_dir = SP_ROOT / subj / "EMG"
        if not emg_dir.exists():
            continue
        for posture in ("index", "thumb"):
            target = "Index" if posture == "index" else "Thumb"
            for csv_path in sorted(emg_dir.glob(f"flex_{posture}_finger_*.csv")):
                filt, env = preprocess_full(csv_path, "softpinch")
                eps, _ = file_scales(filt)
                for e in range(len(env) // PTS_PER_EPOCH):
                    for k, lab in enumerate(["Rest", f"{target} Contract", f"{target} Release"]):
                        p0 = e * PTS_PER_EPOCH + k * SEG_LEN
                        s0, s1 = int(p0 * RMS_STEP), int((p0 + SEG_LEN) * RMS_STEP)
                        if s1 > len(filt):
                            continue
                        X.append(window_features_v3(filt[s0:s1], eps))
                        y.append(lab)
        print(f"    {subj} 完成，累積 {len(X)} 筆", flush=True)
    return np.array(X), np.array(y)


def main():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    print("=== 1. 建立訓練集（v3 跨通道正規化）===", flush=True)
    Xtr, ytr = build_training_set_v3()
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
        eps, _ = file_scales(filt)
        target = "Index" if posture == "index" else "Thumb"
        mu, sigma = env.mean(axis=0), env.std(axis=0)
        env_n = (env - mu) / (sigma + 1e-8)

        segs, feats = [], []
        for s in range(0, len(env_n) - win_pts + 1, step_pts):
            a0, a1 = int(s * RMS_STEP), int((s + win_pts) * RMS_STEP)
            if a1 > len(filt):
                break
            segs.append(env_n[s:s + win_pts])
            feats.append(window_features_v3(filt[a0:a1], eps))
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
                      "cnn_lstm": facc(m_lab), "traditional_v3": facc(t_lab),
                      "trad_n_unique": int(len(set(t_lab))),
                      "trad_switches": t_sw})
        print(f"  {subject:12s} {posture:6s} {path.stem[:32]:34s} "
              f"CNN {rows[-1]['cnn_lstm']:6.1%}  傳統v3 {rows[-1]['traditional_v3']:6.1%}  "
              f"(輸出{rows[-1]['trad_n_unique']}類/切換{t_sw})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_ROOT / "traditional_v3_by_file.csv", index=False)

    print("\n=== 3. 分組彙總（視窗數加權）===")
    summ = []
    for grp, g in df.groupby(df["source"].map({"ours": "我們的資料", "softpinch": "論文held-out"})):
        w = g["n_windows"]
        summ.append({"group": grp, "n_files": len(g), "n_windows": int(w.sum()),
                      "cnn_lstm": float((g.cnn_lstm * w).sum() / w.sum()),
                      "traditional_v3": float((g.traditional_v3 * w).sum() / w.sum())})
    sdf = pd.DataFrame(summ)
    sdf.to_csv(RESULTS_ROOT / "traditional_v3_summary.csv", index=False)
    print(sdf.to_string(index=False))

    # 與 v2 對照，確認修正是否有效
    v2p = RESULTS_ROOT / "traditional_v2_by_file.csv"
    if v2p.exists():
        v2 = pd.read_csv(v2p)[["file_stem", "traditional_v2"]]
        cmp = df.merge(v2, on="file_stem")
        ours = cmp[cmp.source == "ours"]
        print("\n=== 4. v2 vs v3（我們的資料）===")
        print(f"  v2 呈 0%/100% 極端的檔案數: "
              f"{int(((ours.traditional_v2 < 0.05) | (ours.traditional_v2 > 0.95)).sum())}/{len(ours)}")
        print(f"  v3 呈 0%/100% 極端的檔案數: "
              f"{int(((ours.traditional_v3 < 0.05) | (ours.traditional_v3 > 0.95)).sum())}/{len(ours)}")
        print(f"  v3 整檔只輸出單一類別的檔案數: {int((ours.trad_n_unique <= 1).sum())}/{len(ours)}")
        w = ours.n_windows
        print(f"\n  加權平均  v2 {float((ours.traditional_v2*w).sum()/w.sum()):.1%}"
              f"  →  v3 {float((ours.traditional_v3*w).sum()/w.sum()):.1%}"
              f"   (CNN+LSTM {float((ours.cnn_lstm*w).sum()/w.sum()):.1%})")

    print(f"\n[完成] {RESULTS_ROOT/'traditional_v3_by_file.csv'}")


if __name__ == "__main__":
    main()
