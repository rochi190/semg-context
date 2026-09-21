"""
把我們自己的 index/thumb 錄音，餵進 SoftPINCH repo 真正產生論文 99.4% 準確率的
checkpoint（SingleNet_CNN+LSTM_EMG/all_subjects，offline LOSO 訓練，5-class：
Index/Thumb 各 Contract/Release + Rest），用論文真正的評估方法（見
evaluate_epoch_segments 說明）算 segment accuracy，跟 sanity_check_own_pipeline.py
（同一套流程跑在論文自己的 subject_0 資料上）的結果對照，量化 domain shift
的實際幅度。

修正紀錄（2026-07-20）：早期版本誤用了 Real_time_inference/subject_0 這個
7-class checkpoint（不同的訓練資料、不同的 label 空間），量出的 66.5%/31.3%
兩個數字都混雜了「checkpoint 用錯」跟「真正的 domain shift」，不能拿來下結論。
換成正確 checkpoint 後，用 calibrate_offline_phase.py 對 subject_0（校準）/
subject_1（保留驗證）重新掃描相位偏移，新的 PHASE_SHIFT_FRACTION=0.06（原本
0.23 是拿錯誤 checkpoint 校準出來的，不適用於這個模型）。

pinch/grasp 資料不測：pinch 的論文協定跟我們自己的協定結構不同（見
sanity_check_own_pipeline.py 開頭說明），grasp 這個模型沒有 Cylinder 類別。

跟論文原始 real-time pipeline（EMGStreamProcessor.update）的已知差異（為了
控制實作複雜度，有明確取捨，記錄在這裡供之後查證）：
    - Hampel/RMS 原始碼是「每次從 circular buffer 抓一個 500ms 局部視窗，
      重新算一次」，這裡改成對整段訊號連續算一次再切片——兩者數學上該給
      幾乎相同結果（因為 buffer 裡的歷史樣本本來就是真實樣本，不是補零），
      唯一差異在每個 500ms 視窗邊界的 reflect padding，影響範圍在
      window_size=100 samples（50ms）以內。
    - 跳過 LowpassFilter(cutoff=5Hz, order=2) 那一步：這個 filter 用
      self.fs=2000Hz 設計，卻套用在已經降到 ~40Hz 的 RMS 訊號上，正規化
      截止頻率只有 0.005，實際效果接近恆等變換，這裡直接省略。
    - PHASE_SHIFT_FRACTION：quality.find_cycle_anchor_index() 抓的是「安靜
      plateau 的起點」，但實測發現這個起點跟真正的 trial 起點（論文協定的
      t=0 rest cue）還有一個系統性偏移，用 calibrate_offline_phase.py 在
      subject_0 上掃描、subject_1 上驗證，找到最佳偏移量是週期的 0.06 倍
      （subject_1 holdout accuracy=88.8%，這是「邊界只能估計、不是精確已知」
      前提下的天花板，可以拿來跟我們自己資料的結果公平比較）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import median_filter
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")   # src/ 目錄，與檔案放在多深無關
sys.path.insert(0, str(_SRC))
from semg.core import pipeline as pp
from semg.core import quality as qm
from semg.core.softpinch_model import load_model

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "src").parent
MODEL_PATH = (REPO_ROOT / "src/SoftPINCH/src/models/loggings/"
              "SingleNet_CNN+LSTM_EMG/all_subjects")
RESULTS_ROOT = REPO_ROOT / "results"

FS = 2000
NOTCH_FREQ = 60  # 我們自己的資料是台灣電網，跟論文的 50Hz 不同，這裡照實際環境調整
BANDPASS = (20, 450)
HAMPEL_HALF_WINDOW = 100   # 對齊 classification_pipeline.py 的 HAMPEL_WINDOWSIZE=100
HAMPEL_SIGMA = 2           # 對齊 HAMPEL_SIGMA=2（不是 preprocessing.py 預設的 3）
RMS_WINDOW = 500           # 250ms，對齊 RMS_SAMPLING_WINDOW
RMS_STEP = 50              # 25ms，對齊 RMS_WINDOW_STEPSIZE

TRIM_START_SEC = 2.0
TRIM_END_SEC = 1.0

PHASE_SHIFT_FRACTION = 0.06  # 校準來源見上方 docstring 跟 calibrate_offline_phase.py

PRED_MAPPING = {
    0: "Index Contract", 1: "Index Release",
    2: "Thumb Contract", 3: "Thumb Release",
    4: "Rest",
}

POSTURE_TO_TARGET = {"index": "Index", "thumb": "Thumb"}
CH_FOR_TARGET = {"Index": 1, "Thumb": 2}  # env 陣列的 column index：0=ch1,1=ch2,2=ch3


def causal_notch_bandpass(x: np.ndarray, fs: int, notch_freq: float, bandpass: tuple):
    b_n, a_n = iirnotch(notch_freq, notch_freq / 2, fs=fs)  # Q=30 -> bw=2Hz -> Q=freq/bw
    zi = lfilter_zi(b_n, a_n) * x[0]
    y, _ = lfilter(b_n, a_n, x, zi=zi)

    b_b, a_b = butter(4, [bandpass[0] / (fs / 2), bandpass[1] / (fs / 2)], btype="band")
    zi = lfilter_zi(b_b, a_b) * y[0]
    y, _ = lfilter(b_b, a_b, y, zi=zi)
    return y


def causal_hampel(x: np.ndarray, half_window: int, n_sigmas: float):
    kernel = 2 * half_window + 1
    med = median_filter(x, size=kernel, mode="reflect")
    mad = 1.4826 * median_filter(np.abs(x - med), size=kernel, mode="reflect")
    threshold = n_sigmas * mad
    return np.where(np.abs(x - med) > threshold, med, x)


def rms_stream(x: np.ndarray, window: int, step: int):
    n_windows = (len(x) - window) // step + 1
    windows = np.lib.stride_tricks.sliding_window_view(x, window)[::step][:n_windows]
    return np.sqrt(np.mean(windows ** 2, axis=1))


def preprocess_for_model(raw_counts_by_ch: dict) -> np.ndarray:
    """raw_counts_by_ch: {1: arr, 2: arr, 3: arr} (ADC counts). Returns (n_rms_points, 3) RMS envelope."""
    envs = []
    for ch in (1, 2, 3):
        mv = pp.adc_to_mv(raw_counts_by_ch[ch])
        mv = pp.trim(mv, FS, TRIM_START_SEC, TRIM_END_SEC)
        mv = mv - mv.mean()
        filt = causal_notch_bandpass(mv, FS, NOTCH_FREQ, BANDPASS)
        hampel = causal_hampel(filt, HAMPEL_HALF_WINDOW, HAMPEL_SIGMA)
        env = rms_stream(hampel, RMS_WINDOW, RMS_STEP)
        envs.append(env)
    min_len = min(len(e) for e in envs)
    return np.stack([e[:min_len] for e in envs], axis=1)  # (n_rms_points, 3)


def evaluate_epoch_segments(env: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                             anchor_idx: int, period_samples: int, target: str, model):
    """
    對照 classification_pipeline.py 的 Manage3Split._segment_trials() +
    build_dataset_from_subjects()：真正產生論文報告準確率的評估方式，不是
    連續 sliding window 逐幀比對（那是 real-time demo 用的，是兩套不同的
    評估邏輯）。

    做法：每個 9s trial 從 anchor 切開後，三等分成 rest(前1/3) /
    contract(中1/3) / release(後1/3)，每一段整段（120 個 RMS 點，不是切成更
    小的滑動視窗）當「一筆樣本」一次餵進模型，得到一個預測，對照這一段唯一
    的正確答案。
    """
    env_norm = (env - mu) / (sigma + 1e-8)
    n_epochs = (len(env_norm) - anchor_idx) // period_samples
    seg_len = period_samples // 3

    rows = []
    for e in range(n_epochs):
        epoch = env_norm[anchor_idx + e * period_samples: anchor_idx + (e + 1) * period_samples]
        segments = {
            "Rest": epoch[0:seg_len],
            f"{target} Contract": epoch[seg_len:2 * seg_len],
            f"{target} Release": epoch[2 * seg_len:3 * seg_len],
        }
        for expected_label, seg in segments.items():
            inp = torch.tensor(seg, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _, _ = model(inp)
                probs = torch.softmax(logits, dim=1)
                pred_idx = int(torch.argmax(logits, dim=1).item())
            pred_label = PRED_MAPPING[pred_idx]
            rows.append({
                "epoch_idx": e,
                "expected": expected_label,
                "predicted": pred_label,
                "confidence": float(probs[0, pred_idx].item()),
                "correct": pred_label == expected_label,
            })
    return pd.DataFrame(rows)


def find_calibrated_anchor(env_target_ch: np.ndarray, env_time: np.ndarray):
    """算週期 + plateau 起點錨點，再套用 PHASE_SHIFT_FRACTION 校準值。"""
    strength, period_s = qm.periodicity_strength(env_target_ch, env_time,
                                                   min_period_s=6.0, max_period_s=15.0)
    if strength <= 0 or period_s <= 0:
        period_s = 9.0  # 抓不到週期就退回論文協定值
    fs_env = 1.0 / np.median(np.diff(env_time))
    period_samples = int(round(period_s * fs_env))
    raw_anchor_idx = qm.find_cycle_anchor_index(env_target_ch, env_time, period_s)
    shift = int(round(PHASE_SHIFT_FRACTION * period_samples))
    anchor_idx = (raw_anchor_idx + shift) % period_samples
    return anchor_idx, period_samples, period_s


def main():
    model, ckpt = load_model(MODEL_PATH / "model.pth")
    print(f"model loaded: {ckpt['model_name']} / {ckpt['sensor_name']}")

    # 只測 index/thumb：這個模型的協定結構（rest(0-3s)/onset(3-6s)/offset(6-9s)，
    # 9s 週期）是針對 index/thumb 驗證過的，pinch 我們自己的錄製協定跟論文原本
    # 的不一樣，套用同一套三等分規則語意不對。
    subject_dirs = {"gary": REPO_ROOT / "data/gary", "neil1": REPO_ROOT / "data/neil1",
                     "CBW1": REPO_ROOT / "data/CBW1"}
    files = []
    for subject, subj_dir in subject_dirs.items():
        for csv_path in sorted(subj_dir.glob("*.csv")):
            posture = None
            for p in ("index", "thumb"):
                if csv_path.stem.startswith(p):
                    posture = p
                    break
            if posture is not None:
                files.append((subject, csv_path, posture))

    # 我們的硬體（ADS1299 + 自己的增益設定）跟論文用的 Delsys Trigno 系統，RMS envelope
    # 絕對數值量級不一樣——這不是訊號品質問題，是兩套硬體的增益/單位不同。用「每個檔案
    # 自己的 mu/sigma」正規化（比照 EMG 常見的單一 session 自我校準做法），跨 session
    # 的振幅差異不會汙染判斷。（對照：測論文自己的資料時，該用他們 checkpoint 存的
    # mu.npy/sigma.npy，不是這裡的 per-file 版本——兩套資料來源，正規化基準不一樣。）
    print("[正規化] 每個檔案各自算自己的 mu/sigma（跟我們硬體 scale 相符，不用論文的常數）")

    all_segment_rows = []
    file_rows = []
    for subject, csv_path, posture in files:
        print(f"[跑模型] {subject}/{csv_path.name}")
        df = pp.load_channel(str(csv_path), channels=(1, 2, 3))
        raw = {ch: df[f"ch{ch}"].values for ch in (1, 2, 3)}
        env = preprocess_for_model(raw)
        mu, sigma = env.mean(axis=0), env.std(axis=0)
        target = POSTURE_TO_TARGET[posture]

        env_time = np.arange(len(env)) * (RMS_STEP / FS)
        anchor_idx, period_samples, period_s = find_calibrated_anchor(
            env[:, CH_FOR_TARGET[target]], env_time)

        seg_df = evaluate_epoch_segments(env, mu, sigma, anchor_idx, period_samples, target, model)
        seg_df["subject"], seg_df["file_stem"], seg_df["posture"] = subject, csv_path.stem, posture
        all_segment_rows.append(seg_df)

        file_rows.append({
            "subject": subject,
            "file_stem": csv_path.stem,
            "posture": posture,
            "detected_period_s": period_s,
            "n_epochs": seg_df["epoch_idx"].nunique(),
            "segment_accuracy": seg_df["correct"].mean(),
            "mean_confidence": seg_df["confidence"].mean(),
        })

    seg_all = pd.concat(all_segment_rows, ignore_index=True)
    seg_all.to_csv(RESULTS_ROOT / "our_data_segments.csv", index=False)

    report = pd.DataFrame(file_rows).sort_values("segment_accuracy")
    report.to_csv(RESULTS_ROOT / "our_data_segment_accuracy.csv", index=False)
    print(f"\n[完成] results/our_data_segment_accuracy.csv, our_data_segments.csv")
    print(report.to_string(index=False))
    print(f"\n整體 segment accuracy（全部檔案 pooled）: {seg_all['correct'].mean():.3f}")


if __name__ == "__main__":
    main()
