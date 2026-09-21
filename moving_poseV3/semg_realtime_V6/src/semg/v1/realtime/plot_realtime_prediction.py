"""
一個檔案一張圖：把錄音當成 real-time 串流餵給模型，畫出「模型在每個時間點認為
現在是哪一類」，5 個類別分開標色（Index 和 Thumb 用不同色系，可以直接看出模型
判斷的是食指還是拇指）。

跟先前版本的差別（2026-07-21 重做）：
    - 先前畫圖時把 "Index Contract" 和 "Thumb Contract" 摺成同一個 "Contract"
      顏色，等於把「模型認為是食指還是拇指」這個最關鍵的輸出資訊丟掉了。
      這版 5 個類別各自一個顏色，Index=藍色系、Thumb=橘紅色系、Rest=灰色。
    - 先前一個檔案產出 5 張不同診斷版本的圖，太雜。這版一個檔案就一張。

為什麼用 3 秒視窗滑動，而不是論文 real-time 的 500ms：
    這個 offline checkpoint（all_subjects, 5-class）訓練時每筆樣本就是「一個
    完整動作階段 = 3 秒」，餵它 500ms 是它沒見過的長度，輸出沒有意義。論文的
    real-time 系統用的是另一個 checkpoint（real_time/subject_0, 7-class，含
    Pinch），那個才是為 500ms/11 個 RMS 點設計的，而且外面還包了五視窗多數決
    + 信心門檻 + 狀態機。這裡是「用 offline 模型做 real-time 風格的連續預測」，
    所以視窗長度必須跟它的訓練長度一致，只是把滑動步進調細（0.25 秒）。

這張圖完全不需要 autocorrelation 猜週期、也不套用任何相位校正常數——每個視窗
都是獨立的一次模型推論，所以看到的就是模型最原始的判斷。

輸出：
    results/model_predictions/<subject>_<file_stem>.png
    results/realtime_prediction_timeline.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
PKG_ROOT = REPO_ROOT                      # 相對路徑的基準（summary csv 裡記圖檔位置用）
FIG_DIR = REPO_ROOT / "results" / "model_predictions"
RESULTS_DIR = REPO_ROOT / "results"

CH_MUSCLE = {0: "ch1\nwrist flexor", 1: "ch2\nindex", 2: "ch3\nthumb"}
CH_MUSCLE_SOFTPINCH = {0: "ch0", 1: "ch1\n(index)", 2: "ch2\n(thumb)"}

SP_ROOT = REPO_ROOT / "src/SoftPINCH/src/experiment/data/subject_independent"

# source: "ours" = 我們的 ADS1299 錄音（ADC counts、台灣 60Hz 電網）
#         "softpinch" = 論文自己的 Delsys 錄音（已是 mV、丹麥 50Hz 電網）
# group : 報告分組用，讓「我們的資料 / 論文訓練集內 / 論文訓練集外」一眼分得開
TARGET_FILES = [
    # ---- 我們自己的資料 ----
    ("gary",  REPO_ROOT / "data/gary/indexgary_20260716_17-06-56_motion1.csv", "index", "ours", "ours"),
    ("gary",  REPO_ROOT / "data/gary/thumbgary_20260716_17-12-05_motion2.csv", "thumb", "ours", "ours"),
    ("neil1", REPO_ROOT / "data/neil1/indexlin_20260716_21-58-56_motion1.csv", "index", "ours", "ours"),
    ("neil1", REPO_ROOT / "data/neil1/thumblin_20260716_22-02-17_motion2.csv", "thumb", "ours", "ours"),
    ("neil1", REPO_ROOT / "data/neil1/indexlin_20260720_20-51-59.csv",         "index", "ours", "ours"),
    ("neil1", REPO_ROOT / "data/neil1/thumblin_20260720_20-55-57.csv",         "thumb", "ours", "ours"),

    # ---- 論文資料：subject_0 在訓練集「內」（classification_pipeline.py 只把 subject_8 排除）----
    ("SP_subject_0", SP_ROOT / "subject_0/EMG/flex_index_finger_2026-02-05 14-19-21.csv",
     "index", "softpinch", "softpinch_in_train"),
    ("SP_subject_0", SP_ROOT / "subject_0/EMG/flex_thumb_finger_2026-02-05 14-46-00.csv",
     "thumb", "softpinch", "softpinch_in_train"),

    # ---- 論文資料：subject_8 是 LOSO held-out，訓練時完全沒看過（TEST_SUBJECT=['subject_8']）----
    ("SP_subject_8", SP_ROOT / "subject_8/EMG/flex_index_finger_2026-02-19 10-50-44.csv",
     "index", "softpinch", "softpinch_held_out"),
    ("SP_subject_8", SP_ROOT / "subject_8/EMG/flex_thumb_finger_2026-02-19 11-13-00.csv",
     "thumb", "softpinch", "softpinch_held_out"),
]

# 論文資料的前處理差異（對齊 sanity_check_own_pipeline.py 已驗證過的做法）：
#   - 他們的 CSV 已經是物理單位（mV），不需要 adc_to_mv
#   - 丹麥電網 50Hz，不是我們的 60Hz
#   - 論文協定的裁切秒數是 TRIM_PERIOD=3（頭尾各 3 秒），不是我們的 2.0/1.0
SOFTPINCH_NOTCH_FREQ = 50
SOFTPINCH_TRIM_SEC = 3.0

# 視窗長度必須等於模型訓練時的單一階段長度（3秒），步進調細模擬 real-time 連續輸出
WINDOW_SEC = 3.0
STEP_SEC = 0.25

# 5 個類別各自一個顏色：Index=藍色系、Thumb=橘紅色系、Rest=灰色
# （深色=Contract 出力、淺色=Release 放鬆，同一根手指的兩個階段色系相同好對照）
CLASS_ORDER = ["Rest", "Index Contract", "Index Release", "Thumb Contract", "Thumb Release"]
CLASS_COLOR = {
    "Rest":           "#9E9E9E",
    "Index Contract": "#1546A0",
    "Index Release":  "#6FA8F5",
    "Thumb Contract": "#B23A0E",
    "Thumb Release":  "#F5A15C",
}
CLASS_Y = {name: i for i, name in enumerate(CLASS_ORDER)}


def load_and_preprocess(csv_path: Path, source: str) -> np.ndarray:
    """依資料來源選對應的前處理。回傳 (n_rms_points, 3) 的 RMS envelope。

    我們的資料直接用 run_model_validation.preprocess_for_model（已驗證過）；
    論文資料的單位/電網頻率/裁切秒數都不同，走下面這條，邏輯跟
    sanity_check_own_pipeline.py 一致，只是改用 causal 濾波保持跟我們這邊一致。"""
    if source == "ours":
        df = pp.load_channel(str(csv_path), channels=(1, 2, 3))
        raw = {ch: df[f"ch{ch}"].values for ch in (1, 2, 3)}
        return preprocess_for_model(raw)

    # source == "softpinch"
    df = pd.read_csv(csv_path)
    # 不同 subject 的欄位命名不一致（ch0/1/2 vs ch4/5/6），一律取前 3 欄
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


def sliding_predict(env, mu, sigma, model):
    """每個視窗都是獨立的一次模型推論，不猜週期、不套相位。"""
    env_norm = (env - mu) / (sigma + 1e-8)
    sec_per_sample = RMS_STEP / FS
    win = int(round(WINDOW_SEC / sec_per_sample))
    step = max(1, int(round(STEP_SEC / sec_per_sample)))

    rows = []
    for start in range(0, len(env_norm) - win + 1, step):
        seg = env_norm[start:start + win]
        inp = torch.tensor(seg, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _, _ = model(inp)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(logits, dim=1).item())
        rows.append({
            "center_time_s": (start + win / 2) * sec_per_sample,
            "predicted": PRED_MAPPING[pred_idx],
            "confidence": float(probs[0, pred_idx].item()),
        })
    return pd.DataFrame(rows)


GROUP_LABEL = {
    "ours": "our ADS1299 recording",
    "softpinch_in_train": "SoftPINCH data — subject IS in the training set",
    "softpinch_held_out": "SoftPINCH data — subject HELD OUT of training (LOSO test subject)",
}


def plot_one(subject, file_stem, posture, env, env_time, pred_df, out_png, source, group):
    fig, axes = plt.subplots(4, 1, figsize=(19, 10), sharex=True,
                              gridspec_kw={"height_ratios": [2, 2, 2, 3.2]})

    expected_finger = "Index" if posture == "index" else "Thumb"
    ch_labels = CH_MUSCLE if source == "ours" else CH_MUSCLE_SOFTPINCH

    for ch in range(3):
        ax = axes[ch]
        ax.plot(env_time, env[:, ch], color="#222222", lw=0.8)
        ax.set_ylabel(ch_labels[ch], fontsize=10)
        ax.margins(x=0)
        ax.grid(axis="x", alpha=0.2)

    # ---- 預測時間軸：階梯圖 + 色塊，y 軸就是 5 個類別 ----
    axp = axes[3]
    t = pred_df["center_time_s"].to_numpy()
    y = pred_df["predicted"].map(CLASS_Y).to_numpy()

    # 鋼琴捲軸式：色塊只畫在「該類別自己那一列」的高度上，不滿版——滿版色塊會把
    # 整張圖糊掉，也看不出模型在哪一類。把連續同類的視窗合併成一條長條，避免畫出
    # 上千個小方塊。
    labels = pred_df["predicted"].to_numpy()
    half = STEP_SEC / 2
    run_start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[run_start]:
            cls = labels[run_start]
            axp.broken_barh(
                [(t[run_start] - half, t[i - 1] - t[run_start] + 2 * half)],
                (CLASS_Y[cls] - 0.36, 0.72),
                facecolors=CLASS_COLOR[cls], edgecolor="none",
            )
            run_start = i

    # 疊一條細階梯線，看得出類別之間怎麼跳動
    axp.step(t, y, where="mid", color="#333333", lw=0.7, alpha=0.5)

    axp.set_yticks(range(len(CLASS_ORDER)))
    axp.set_yticklabels(CLASS_ORDER, fontsize=10)
    axp.set_ylim(-0.6, len(CLASS_ORDER) - 0.4)
    axp.set_ylabel("model output", fontsize=10)
    axp.set_xlabel("time (s)", fontsize=11)
    axp.margins(x=0)
    axp.grid(axis="x", alpha=0.2)

    # 把「這個檔案錄的是哪根手指」用水平底色標出來，方便一眼比對模型有沒有選對手指
    for cls in CLASS_ORDER:
        if cls.startswith(expected_finger):
            axp.axhspan(CLASS_Y[cls] - 0.5, CLASS_Y[cls] + 0.5,
                        color="#2E7D32", alpha=0.07, zorder=0)

    # 統計：模型有多少比例的時間認為是「正確的那根手指」
    finger_hit = pred_df["predicted"].str.startswith(expected_finger).mean()
    rest_frac = (pred_df["predicted"] == "Rest").mean()
    wrong_finger = 1.0 - finger_hit - rest_frac

    # 最不依賴切割對齊的指標：排除 Rest，只問「模型認為有動作時，手指選對了嗎」。
    # segment accuracy 那組數字會被 Rest/Contract/Release 的時序對齊問題汙染，
    # 這個不會——因為它不管動作發生在哪個時間點，只看手指判斷。亂猜 = 50%。
    active = pred_df[pred_df["predicted"] != "Rest"]
    finger_acc_active = (active["predicted"].str.startswith(expected_finger).mean()
                          if len(active) else float("nan"))

    fig.suptitle(
        f"[{GROUP_LABEL[group]}]\n"
        f"{subject} / {file_stem}      recorded motion = {expected_finger}"
        f"      [{WINDOW_SEC}s window, {STEP_SEC}s step — each window is one independent model call]\n"
        f"time spent in each output:   {expected_finger} (correct finger) {finger_hit:.0%}   |   "
        f"Rest {rest_frac:.0%}   |   other finger (wrong) {wrong_finger:.0%}\n"
        f"finger discrimination, excluding Rest windows: "
        f"{finger_acc_active:.0%} correct  (chance = 50%)       "
        f"green band = the two classes this file should produce",
        fontsize=11.5, y=0.99
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return finger_hit, rest_frac, wrong_finger, finger_acc_active


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_model(MODEL_PATH / "model.pth")
    print(f"model loaded: {ckpt['model_name']} / {ckpt['sensor_name']} "
          f"(output_dim={ckpt['model_args']['output_dim']}, 5-class offline checkpoint)")

    all_rows, summary = [], []
    for subject, csv_path, posture, source, group in TARGET_FILES:
        print(f"[real-time 風格預測] {subject}/{csv_path.name}  ({group})")
        env = load_and_preprocess(csv_path, source)
        mu, sigma = env.mean(axis=0), env.std(axis=0)
        env_time = np.arange(len(env)) * (RMS_STEP / FS)

        pred_df = sliding_predict(env, mu, sigma, model)
        pred_df["subject"] = subject
        pred_df["group"] = group
        pred_df["file_stem"] = csv_path.stem
        pred_df["posture"] = posture
        all_rows.append(pred_df)

        safe_stem = csv_path.stem.replace(" ", "_")
        out_png = FIG_DIR / f"{subject}_{safe_stem}.png"
        finger_hit, rest_frac, wrong_finger, finger_acc_active = plot_one(
            subject, csv_path.stem, posture, env, env_time, pred_df, out_png, source, group)

        summary.append({
            "group": group, "subject": subject, "file_stem": csv_path.stem, "posture": posture,
            "n_windows": len(pred_df),
            "frac_correct_finger": finger_hit,
            "frac_rest": rest_frac,
            "frac_wrong_finger": wrong_finger,
            "finger_accuracy_excluding_rest": finger_acc_active,
            "mean_confidence": pred_df["confidence"].mean(),
            "figure": str(out_png.relative_to(PKG_ROOT)),
        })
        print(f"    -> {out_png.name}   正確手指 {finger_hit:.0%} / Rest {rest_frac:.0%} "
              f"/ 判成另一根手指 {wrong_finger:.0%}   "
              f"｜排除Rest後手指判對率 {finger_acc_active:.0%}")

    timeline = pd.concat(all_rows, ignore_index=True)
    timeline.to_csv(RESULTS_DIR / "realtime_prediction_timeline.csv", index=False)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(RESULTS_DIR / "realtime_prediction_summary.csv", index=False)

    print(f"\n[完成] 圖：{FIG_DIR}")
    print(summary_df.drop(columns=["figure"]).to_string(index=False))

    # 分組彙總：這才是要放進報告的核心對照——同一個模型、同一套評估方式，
    # 只換資料來源，看手指判別率掉多少。
    print("\n=== 分組彙總（手指判別率，排除 Rest；亂猜=50%）===")
    timeline["target"] = timeline["posture"].map({"index": "Index", "thumb": "Thumb"})
    act = timeline[timeline["predicted"] != "Rest"].copy()
    act["hit"] = act.apply(lambda r: r["predicted"].startswith(r["target"]), axis=1)
    for grp, g in act.groupby("group"):
        print(f"  {GROUP_LABEL[grp]:62s}  {g['hit'].mean():.1%}   (n={len(g)})")


if __name__ == "__main__":
    main()
