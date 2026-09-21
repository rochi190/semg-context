"""
傳統特徵基準線 v2 —— 修正 v1 的兩個方法學錯誤，並跑完全部 14 個檔案。

v1（traditional_feature_baseline.py / traditional_vs_model_plot.py）的問題：
  問題1：特徵算在 40Hz 的 RMS 包絡線上，不是原始 EMG。這使得 ZC/SSC/MNF/MDF
         的物理意義完全改變（MNF 變成「包絡線調變節律 0-20Hz」，而不是文獻上
         的「EMG 功率譜頻率 20-450Hz」），等於閹割了傳統方法最有力的頻域資訊，
         很可能低估了傳統方法。
  問題2：跨通道特徵用 z-score 後訊號的比值。z-score 後整檔均值為 0，當某視窗
         落在該通道均值附近時分母趨近 0，比值爆炸；LDA 是線性模型，對這種離群
         值極脆弱——很可能就是 v1 某個檔案掉到 47%（低於亂猜）的原因。

v2 的修正：
  修正1：所有時域/頻域特徵改算在 **濾波後的原始 EMG（2000Hz）** 上。
  修正2：跨通道改用 **穩健正規化 + log 比值**：先用該通道自己在整個檔案的
         「視窗 RMS 中位數」當基線做正規化（中位數對離群值穩健），再取 log 比值
         （對稱、不會因分母小而爆炸），另外附上總和歸一化的比例（有界於 [0,1]）。

兩個分類器仍站在完全相同的位置：都用 SoftPINCH 訓練（排除 subject_8）、都做
5 類、都 zero-shot 套到我們的資料、都 3 秒視窗。

輸出：results/traditional_v2_summary.csv, results/traditional_v2_by_file.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import signal as sps

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")   # src/ 目錄，與檔案放在多深無關
sys.path.insert(0, str(_SRC))
from semg.core import pipeline as pp
from semg.core.softpinch_model import load_model
from semg.model_eval.run_model_validation import (
    MODEL_PATH, FS, NOTCH_FREQ, BANDPASS, HAMPEL_HALF_WINDOW, HAMPEL_SIGMA,
    RMS_WINDOW, RMS_STEP, TRIM_START_SEC, TRIM_END_SEC, PRED_MAPPING,
    causal_notch_bandpass, causal_hampel, rms_stream,
)

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "src").parent
RESULTS_ROOT = REPO_ROOT / "results"
SP_ROOT = REPO_ROOT / "src/SoftPINCH/src/experiment/data/subject_independent"

SOFTPINCH_NOTCH_FREQ, SOFTPINCH_TRIM_SEC = 50, 3.0
WINDOW_SEC, STEP_SEC = 3.0, 0.5
SEC_PER_PT = RMS_STEP / FS
TRIAL_SEC = 9
PTS_PER_EPOCH = int(TRIAL_SEC / SEC_PER_PT)
SEG_LEN = PTS_PER_EPOCH // 3

TRAIN_SUBJECTS = [f"subject_{i}" for i in [0, 1, 2, 3, 4, 5]]
AR_ORDER = 4


def preprocess_full(csv_path: Path, source: str):
    """回傳 (filtered_emg 2000Hz (N,3), rms_envelope 40Hz (M,3))。
    兩者都保留——傳統特徵吃原始 EMG，CNN+LSTM 吃包絡線。"""
    if source == "ours":
        df = pp.load_channel(str(csv_path), channels=(1, 2, 3))
        raws = [pp.adc_to_mv(df[f"ch{c}"].values) for c in (1, 2, 3)]
        trim_s, trim_e, notch = TRIM_START_SEC, TRIM_END_SEC, NOTCH_FREQ
    else:
        arr = pd.read_csv(csv_path).iloc[:, :3].values
        raws = [arr[:, c].astype(np.float64) for c in range(3)]
        trim_s = trim_e = SOFTPINCH_TRIM_SEC
        notch = SOFTPINCH_NOTCH_FREQ

    filts, envs = [], []
    for x in raws:
        x = pp.trim(x, FS, trim_s, trim_e)
        x = x - x.mean()
        f = causal_notch_bandpass(x, FS, notch, BANDPASS)
        f = causal_hampel(f, HAMPEL_HALF_WINDOW, HAMPEL_SIGMA)
        filts.append(f)
        envs.append(rms_stream(f, RMS_WINDOW, RMS_STEP))
    n = min(len(e) for e in envs)
    return np.stack(filts, axis=1), np.stack([e[:n] for e in envs], axis=1)


def ar_coeffs(x: np.ndarray, order: int = AR_ORDER) -> np.ndarray:
    """Yule-Walker 解 AR 係數（sEMG 文獻的經典特徵）。"""
    x = x - x.mean()
    r = np.correlate(x, x, mode="full")[len(x) - 1:len(x) + order]
    if r[0] <= 0:
        return np.zeros(order)
    R = np.array([[r[abs(i - j)] for j in range(order)] for i in range(order)])
    try:
        return np.linalg.solve(R + np.eye(order) * 1e-10 * r[0], r[1:order + 1])
    except np.linalg.LinAlgError:
        return np.zeros(order)


def channel_features(x: np.ndarray, eps: float) -> list[float]:
    """在**濾波後的原始 EMG**（2000Hz）上算標準 sEMG 特徵。
    eps = 該通道的雜訊死區門檻（ZC/SSC/WAMP 用），由整檔穩健尺度決定。"""
    N = len(x)
    dx = np.diff(x)

    mav = float(np.mean(np.abs(x)))
    rms = float(np.sqrt(np.mean(x ** 2)))
    wl = float(np.sum(np.abs(dx)))
    var = float(np.var(x))
    # ZC：變號且跨幅超過死區
    zc = int(np.sum((x[:-1] * x[1:] < 0) & (np.abs(dx) >= eps)))
    # SSC：斜率變號且變化量超過死區
    ssc = int(np.sum(((x[1:-1] - x[:-2]) * (x[1:-1] - x[2:]) >= eps)))
    # WAMP：相鄰差超過死區的次數
    wamp = int(np.sum(np.abs(dx) >= eps))

    # 頻域：真正的 EMG 功率譜（20-450Hz），這才是文獻定義的 MNF/MDF
    f, P = sps.welch(x, fs=FS, nperseg=min(N, 1024))
    band = (f >= BANDPASS[0]) & (f <= BANDPASS[1])
    f, P = f[band], P[band]
    Psum = P.sum() + 1e-20
    mnf = float((f * P).sum() / Psum)
    cum = np.cumsum(P)
    mdf = float(f[np.searchsorted(cum, cum[-1] / 2)])
    pkf = float(f[np.argmax(P)])
    # 頻譜二階矩（帶寬）
    bw = float(np.sqrt(((f - mnf) ** 2 * P).sum() / Psum))

    return [mav, rms, np.log(rms + 1e-12), wl, var, zc, ssc, wamp,
            mnf, mdf, pkf, bw] + list(ar_coeffs(x))


def window_features_v2(win_emg: np.ndarray, eps: np.ndarray,
                        baseline: np.ndarray) -> np.ndarray:
    """win_emg: (n_samples, 3) 濾波後原始 EMG 視窗。
    baseline: 每通道在整個檔案的「視窗RMS中位數」，用來做穩健正規化。"""
    feats = []
    for c in range(3):
        feats += channel_features(win_emg[:, c], eps[c])

    # 跨通道：穩健正規化後取 log 比值 + 總和比例
    a = np.array([np.sqrt(np.mean(win_emg[:, c] ** 2)) for c in range(3)])
    alpha = a / (baseline + 1e-12)              # 相對於各自基線的活化程度
    la = np.log(alpha + 1e-6)
    feats += [la[1] - la[2], la[1] - la[0], la[2] - la[0]]   # log 比值（對稱、不爆）
    s = alpha.sum() + 1e-12
    feats += list(alpha / s)                                   # 比例，有界 [0,1]
    return np.array(feats, dtype=np.float64)


def file_scales(filt: np.ndarray):
    """算每通道的死區門檻 eps 與 RMS 基線（都用穩健統計量）。"""
    eps, base = [], []
    w = int(WINDOW_SEC * FS)
    step = int(STEP_SEC * FS)
    for c in range(3):
        x = filt[:, c]
        mad = np.median(np.abs(x - np.median(x)))
        eps.append(0.05 * 1.4826 * mad)          # 死區 = 5% 穩健標準差
        rr = [np.sqrt(np.mean(x[s:s + w] ** 2))
              for s in range(0, max(len(x) - w, 1), step)]
        base.append(np.median(rr) if rr else 1.0)
    return np.array(eps), np.array(base)


def build_training_set():
    X, y = [], []
    for subj in TRAIN_SUBJECTS:
        emg_dir = SP_ROOT / subj / "EMG"
        if not emg_dir.exists():
            continue
        for posture in ("index", "thumb"):
            target = "Index" if posture == "index" else "Thumb"
            for csv_path in sorted(emg_dir.glob(f"flex_{posture}_finger_*.csv")):
                filt, env = preprocess_full(csv_path, "softpinch")
                eps, base = file_scales(filt)
                n_epochs = len(env) // PTS_PER_EPOCH
                for e in range(n_epochs):
                    for k, lab in enumerate(["Rest", f"{target} Contract", f"{target} Release"]):
                        p0 = (e * PTS_PER_EPOCH + k * SEG_LEN)
                        s0, s1 = int(p0 * RMS_STEP), int((p0 + SEG_LEN) * RMS_STEP)
                        if s1 > len(filt):
                            continue
                        X.append(window_features_v2(filt[s0:s1], eps, base))
                        y.append(lab)
        print(f"    {subj} 完成，累積 {len(X)} 筆", flush=True)
    return np.array(X), np.array(y)


def discover_our_files():
    out = []
    for subj in ("gary", "neil1", "CBW1"):
        d = REPO_ROOT / "data" / subj
        for p in sorted(d.glob("*.csv")):
            for post in ("index", "thumb"):
                if p.stem.startswith(post):
                    out.append((subj, p, post, "ours"))
    return out


def main():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    print("=== 1. 建立訓練集（SoftPINCH，排除 subject_8，特徵算在原始 EMG 上）===", flush=True)
    Xtr, ytr = build_training_set()
    print(f"  {len(Xtr)} 筆 × {Xtr.shape[1]} 維")
    for c, n in zip(*np.unique(ytr, return_counts=True)):
        print(f"    {c:16s} {n}")

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

    print(f"\n=== 2. 兩個分類器 zero-shot 跑 {len(files)} 個檔案 ===", flush=True)
    rows = []
    win_pts = int(round(WINDOW_SEC / SEC_PER_PT))
    step_pts = int(round(STEP_SEC / SEC_PER_PT))

    for subject, path, posture, source in files:
        filt, env = preprocess_full(path, source)
        eps, base = file_scales(filt)
        target = "Index" if posture == "index" else "Thumb"
        mu, sigma = env.mean(axis=0), env.std(axis=0)
        env_n = (env - mu) / (sigma + 1e-8)

        segs_env, feats = [], []
        for s in range(0, len(env_n) - win_pts + 1, step_pts):
            segs_env.append(env_n[s:s + win_pts])
            a0, a1 = int(s * RMS_STEP), int((s + win_pts) * RMS_STEP)
            if a1 > len(filt):
                segs_env.pop()
                break
            feats.append(window_features_v2(filt[a0:a1], eps, base))
        if not segs_env:
            continue

        with torch.no_grad():
            logits, _, _ = model(torch.tensor(np.stack(segs_env), dtype=torch.float32))
        m_lab = np.array([PRED_MAPPING[i] for i in logits.argmax(1).numpy()])
        t_lab = lda.predict(np.stack(feats))

        def finger_acc(lab):
            act = lab[lab != "Rest"]
            return float(np.mean([l.startswith(target) for l in act])) if len(act) else np.nan

        rows.append({"subject": subject, "file_stem": path.stem, "posture": posture,
                      "source": source, "n_windows": len(m_lab),
                      "cnn_lstm": finger_acc(m_lab), "traditional_v2": finger_acc(t_lab),
                      "cnn_rest_frac": float((m_lab == "Rest").mean()),
                      "trad_rest_frac": float((t_lab == "Rest").mean())})
        print(f"  {subject:12s} {posture:6s} {path.stem[:34]:36s} "
              f"CNN {rows[-1]['cnn_lstm']:.1%}  傳統v2 {rows[-1]['traditional_v2']:.1%}",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_ROOT / "traditional_v2_by_file.csv", index=False)

    print("\n=== 3. 分組彙總（以視窗數加權）===")
    summ = []
    for grp, g in df.groupby(df["source"].map({"ours": "我們的資料", "softpinch": "論文held-out"})):
        w = g["n_windows"]
        summ.append({"group": grp, "n_files": len(g), "n_windows": int(w.sum()),
                      "cnn_lstm": float((g["cnn_lstm"] * w).sum() / w.sum()),
                      "traditional_v2": float((g["traditional_v2"] * w).sum() / w.sum())})
    sdf = pd.DataFrame(summ)
    sdf.to_csv(RESULTS_ROOT / "traditional_v2_summary.csv", index=False)
    print(sdf.to_string(index=False))
    ours = df[df.source == "ours"]
    print(f"\n  我們的資料：CNN+LSTM 勝 {int((ours.cnn_lstm > ours.traditional_v2).sum())} 檔，"
          f"傳統v2 勝 {int((ours.traditional_v2 > ours.cnn_lstm).sum())} 檔（共 {len(ours)} 檔）")
    print(f"\n[完成] {RESULTS_ROOT/'traditional_v2_by_file.csv'}")


if __name__ == "__main__":
    main()
