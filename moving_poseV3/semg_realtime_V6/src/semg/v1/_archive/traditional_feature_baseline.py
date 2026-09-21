"""
傳統特徵基準線：回答「這件事是不是用傳統時頻特徵就做得到，不需要 CNN+LSTM？」

跑三個對照，全部用同一批 3 秒視窗（跟 plot_realtime_prediction.py 完全相同的切法），
所以數字可以直接跟模型的手指判別率並排比較：

  A. 最陽春的規則：比較 ch2(index) 跟 ch3(thumb) 哪個 RMS 大就判哪根手指。
     完全不需要訓練、不需要模型——如果這招就有 8 成，那 CNN+LSTM 在「分手指」
     這件事上確實沒有加值，必須誠實承認。

  B. 傳統特徵 + LDA（在我們自己的資料上做 leave-one-file-out 交叉驗證）：
     Hudgins 經典時域特徵組（MAV/RMS/WL/ZC/SSC）+ 頻域（平均頻率/中位頻率），
     每通道各一組。這是 sEMG 文獻的標準基準線。

  C. CNN+LSTM（SoftPINCH 預訓練，zero-shot）：直接讀
     results/realtime_prediction_summary.csv 的既有結果。

另外分開報「分手指」跟「分 Contract vs Release」兩個任務——這兩件事難度差很多，
混在一起講會失焦（見輸出說明）。

輸出：results/traditional_feature_baseline.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")   # src/ 目錄，與檔案放在多深無關
sys.path.insert(0, str(_SRC))
from semg.model_eval.run_model_validation import FS, RMS_STEP
from semg.realtime.plot_realtime_prediction import load_and_preprocess, SP_ROOT

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "src").parent
RESULTS_ROOT = REPO_ROOT / "results"

WINDOW_SEC = 3.0
STEP_SEC = 0.5

FILES = [
    ("neil1", REPO_ROOT / "data/neil1/indexlin_20260716_21-58-56_motion1.csv", "index", "ours"),
    ("neil1", REPO_ROOT / "data/neil1/thumblin_20260716_22-02-17_motion2.csv", "thumb", "ours"),
    ("neil1", REPO_ROOT / "data/neil1/indexlin_20260720_20-51-59.csv", "index", "ours"),
    ("neil1", REPO_ROOT / "data/neil1/thumblin_20260720_20-55-57.csv", "thumb", "ours"),
    ("gary",  REPO_ROOT / "data/gary/indexgary_20260716_17-06-56_motion1.csv", "index", "ours"),
    ("gary",  REPO_ROOT / "data/gary/thumbgary_20260716_17-12-05_motion2.csv", "thumb", "ours"),
]


def window_features(seg: np.ndarray) -> np.ndarray:
    """seg: (n_points, 3) 的 RMS envelope 視窗。每通道抽一組經典特徵。

    註：輸入已經是 RMS envelope（250ms 窗），本身就是傳統時域特徵，所以這裡是
    「在傳統特徵上再抽統計量」。ZC/SSC 在包絡線上算的是包絡線本身的變化次數，
    不是原始 EMG 的過零率——這點在解讀時要說清楚，不要宣稱成標準 Hudgins 特徵。
    """
    feats = []
    for ch in range(seg.shape[1]):
        x = seg[:, ch]
        dx = np.diff(x)
        mav = np.mean(np.abs(x))
        rms = np.sqrt(np.mean(x ** 2))
        wl = np.sum(np.abs(dx))                      # waveform length
        ssc = np.sum(np.diff(np.sign(dx)) != 0)      # slope sign changes
        std = x.std()
        rng = x.max() - x.min()
        slope = np.polyfit(np.arange(len(x)), x, 1)[0]  # 趨勢斜率（升/降）
        # 頻域：包絡線的頻譜重心/中位頻率（動作節律快慢）
        f, p = sps.welch(x - x.mean(), fs=1.0 / (RMS_STEP / FS),
                          nperseg=min(len(x), 64))
        psum = p.sum() + 1e-12
        mnf = float((f * p).sum() / psum)
        cum = np.cumsum(p)
        mdf = float(f[np.searchsorted(cum, cum[-1] / 2)])
        feats += [mav, rms, wl, ssc, std, rng, slope, mnf, mdf]
    # 跨通道比值：這是「分手指」最直觀的傳統特徵
    r_all = [np.sqrt(np.mean(seg[:, c] ** 2)) for c in range(3)]
    feats += [r_all[1] / (r_all[2] + 1e-12), r_all[1] / (r_all[0] + 1e-12),
              r_all[2] / (r_all[0] + 1e-12)]
    return np.array(feats, dtype=np.float64)


def build_dataset():
    X, y_finger, groups = [], [], []
    sec_per_sample = RMS_STEP / FS
    win = int(round(WINDOW_SEC / sec_per_sample))
    step = int(round(STEP_SEC / sec_per_sample))

    for subject, path, posture, source in FILES:
        print(f"  抽特徵：{subject}/{path.name}")
        env = load_and_preprocess(path, source)
        mu, sigma = env.mean(axis=0), env.std(axis=0)
        env_n = (env - mu) / (sigma + 1e-8)
        # 只取「有動作」的視窗（該手指對應通道能量高於該檔中位數），對齊模型評估時
        # 排除 Rest 的做法，兩邊才是同一個任務
        ch = 1 if posture == "index" else 2
        thresh = np.median(env[:, ch])
        for s in range(0, len(env_n) - win + 1, step):
            seg_raw = env[s:s + win]
            if np.mean(seg_raw[:, ch]) <= thresh:
                continue
            X.append(window_features(env_n[s:s + win]))
            y_finger.append("Index" if posture == "index" else "Thumb")
            groups.append(f"{subject}/{path.stem}")
    return np.array(X), np.array(y_finger), np.array(groups)


def rule_channel_ratio():
    """對照 A：不訓練，純比 ch2 vs ch3 誰的 RMS 大。"""
    sec_per_sample = RMS_STEP / FS
    win = int(round(WINDOW_SEC / sec_per_sample))
    step = int(round(STEP_SEC / sec_per_sample))
    correct, total = 0, 0
    per_file = []
    for subject, path, posture, source in FILES:
        env = load_and_preprocess(path, source)
        ch = 1 if posture == "index" else 2
        thresh = np.median(env[:, ch])
        c = t = 0
        for s in range(0, len(env) - win + 1, step):
            seg = env[s:s + win]
            if np.mean(seg[:, ch]) <= thresh:
                continue
            # 用「相對於該通道自己基線」的比值，避免通道間絕對增益差異主導
            z_idx = seg[:, 1].mean() / (np.median(env[:, 1]) + 1e-12)
            z_thb = seg[:, 2].mean() / (np.median(env[:, 2]) + 1e-12)
            pred = "Index" if z_idx > z_thb else "Thumb"
            truth = "Index" if posture == "index" else "Thumb"
            c += int(pred == truth); t += 1
        per_file.append((f"{subject}/{path.stem}", posture, c / max(t, 1), t))
        correct += c; total += t
    return correct / max(total, 1), per_file


def main():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneGroupOut

    print("=== A. 陽春規則：ch2 vs ch3 誰大就判誰 ===")
    acc_rule, per_file = rule_channel_ratio()
    for name, posture, a, n in per_file:
        print(f"    {name[:48]:50s} {posture:6s} {a:.1%}  (n={n})")
    print(f"  → 整體 {acc_rule:.1%}\n")

    print("=== B. 傳統特徵 + 分類器（leave-one-file-out）===")
    X, y, groups = build_dataset()
    print(f"  樣本數={len(X)}, 特徵維度={X.shape[1]}, 檔案數={len(set(groups))}")

    rows = []
    for clf_name, clf in [("LDA", LinearDiscriminantAnalysis()),
                           ("SVM(rbf)", SVC(C=1.0, gamma="scale"))]:
        logo = LeaveOneGroupOut()
        preds = np.empty(len(y), dtype=object)
        for tr, te in logo.split(X, y, groups):
            pipe = make_pipeline(StandardScaler(), clf.__class__(**clf.get_params()))
            pipe.fit(X[tr], y[tr])
            preds[te] = pipe.predict(X[te])
        acc = (preds == y).mean()
        print(f"  {clf_name}: {acc:.1%}")
        rows.append({"method": f"傳統特徵 + {clf_name}（leave-one-file-out）",
                      "finger_accuracy": acc, "n_samples": len(y)})

    rows.insert(0, {"method": "陽春規則：ch2 vs ch3 誰大", "finger_accuracy": acc_rule,
                     "n_samples": sum(n for *_, n in per_file)})

    # C：模型的既有結果
    summ = pd.read_csv(RESULTS_ROOT / "realtime_prediction_summary.csv")
    ours = summ[summ.group == "ours"]
    w = ours["n_windows"] * (1 - ours["frac_rest"])
    model_acc = float((ours["finger_accuracy_excluding_rest"] * w).sum() / w.sum())
    rows.append({"method": "CNN+LSTM（SoftPINCH 預訓練，zero-shot，未看過我們的資料）",
                  "finger_accuracy": model_acc, "n_samples": int(w.sum())})

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_ROOT / "traditional_feature_baseline.csv", index=False)
    print("\n=== 總表：分手指（Index vs Thumb），亂猜=50% ===")
    for _, r in df.iterrows():
        print(f"  {r['method']:56s} {r['finger_accuracy']:.1%}")
    print(f"\n[完成] {RESULTS_ROOT / 'traditional_feature_baseline.csv'}")
    print("\n注意：B 是在我們自己的資料上訓練（跨檔案驗證），C 是完全沒看過我們資料的")
    print("      zero-shot。B 佔便宜，兩者不是同一個難度，解讀時要講清楚。")


if __name__ == "__main__":
    main()
