"""
sEMG 前處理 pipeline（Python 版）

直接對照 src/semg_analysisV4.m 移植，處理順序與參數保持一致：
    1. ADC counts -> mV
    2. 開頭/結尾裁切
    3. 去 DC
    4. 60Hz notch（iirnotch, Q=30，zero-phase）
    5. 20-450Hz bandpass（Butterworth 4 階，zero-phase）
    6. Hampel filter（半寬 200 samples / 100ms，n_sigma=3）
    7. RMS envelope（250ms window / 25ms step，90% overlap）

與 MATLAB 腳本的差異只有一處：MATLAB 版本的 TRIM_SEC_START 是每個檔案手動
看訊號調的（V4 注解說明「依錄音實際情況調整」），這裡批次處理沒辦法逐檔手動看，
改用固定的保守裁切秒數（見 batch_process.py 的 DEFAULT_TRIM_START_SEC /
DEFAULT_TRIM_END_SEC），純粹是為了避開 filtfilt 邊際效應，不是為了裁掉不穩定的
啟動段——抽樣看過幾個檔案開頭沒有明顯的啟動不穩定段。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import median_filter

# ---- ADS1299 硬體參數（對齊 semg_analysisV4.m）----
ADS_GAIN = 8
ADS_VREF = 4.5
ADS_BITS = 24
LSB_UV = (ADS_VREF / (ADS_GAIN * 2 ** (ADS_BITS - 1))) * 1e6
LSB_MV = LSB_UV / 1000

FS = 2000

NOTCH_FREQ = 60
NOTCH_BW = 2  # -> Q = 30

BP_LOW = 20
BP_HIGH = 450
BP_ORDER = 4

HAMPEL_HALF_WINDOW = 200  # samples (100ms @ 2000Hz)
HAMPEL_NSIGMA = 3

RMS_WIN_SEC = 0.250
RMS_STEP_SEC = 0.025


def load_channel(csv_path: str, channels=(1, 2, 3)) -> pd.DataFrame:
    cols = ["timestamp_s"] + [f"ch{c}" for c in channels]
    df = pd.read_csv(csv_path, usecols=cols)
    return df


def adc_to_mv(counts: np.ndarray) -> np.ndarray:
    return counts.astype(np.float64) * LSB_MV


def trim(x: np.ndarray, fs: int, start_s: float, end_s: float) -> np.ndarray:
    start_n = round(start_s * fs)
    end_n = round(end_s * fs)
    if len(x) <= start_n + end_n:
        raise ValueError(
            f"訊號長度（{len(x)}）不足以裁切開頭 {start_s}s + 結尾 {end_s}s"
        )
    return x[start_n: len(x) - end_n] if end_n > 0 else x[start_n:]


def notch_filter(x: np.ndarray, fs: int = FS, freq: float = NOTCH_FREQ,
                  bw: float = NOTCH_BW) -> np.ndarray:
    q = freq / bw
    b, a = signal.iirnotch(freq, q, fs=fs)
    return signal.filtfilt(b, a, x)


def bandpass_filter(x: np.ndarray, fs: int = FS, low: float = BP_LOW,
                     high: float = BP_HIGH, order: int = BP_ORDER) -> np.ndarray:
    b, a = signal.butter(order, [low, high], btype="bandpass", fs=fs)
    return signal.filtfilt(b, a, x)


def hampel_filter(x: np.ndarray, half_window: int = HAMPEL_HALF_WINDOW,
                   n_sigma: float = HAMPEL_NSIGMA):
    """向量化 Hampel filter，對齊 SoftPINCH repo 用 scipy.ndimage.median_filter 的作法。"""
    k = 1.4826  # MAD -> std（高斯分佈下的比例常數）
    size = 2 * half_window + 1
    med = median_filter(x, size=size, mode="nearest")
    mad = median_filter(np.abs(x - med), size=size, mode="nearest")
    threshold = n_sigma * k * mad
    outlier_mask = np.abs(x - med) > threshold
    y = np.where(outlier_mask, med, x)
    return y, outlier_mask


def rms_envelope(x: np.ndarray, fs: int = FS, win_sec: float = RMS_WIN_SEC,
                  step_sec: float = RMS_STEP_SEC):
    win_n = round(win_sec * fs)
    step_n = round(step_sec * fs)
    if win_n > len(x):
        raise ValueError("RMS 視窗長度超過訊號總長度")
    n_windows = (len(x) - win_n) // step_n + 1
    windows = np.lib.stride_tricks.sliding_window_view(x, win_n)[::step_n][:n_windows]
    env = np.sqrt(np.mean(windows ** 2, axis=1))
    env_time = (np.arange(n_windows) * step_n + win_n / 2) / fs
    return env_time, env


def process_channel(raw_counts: np.ndarray, fs: int = FS,
                     trim_start_s: float = 2.0, trim_end_s: float = 1.0):
    """完整處理單一通道，回傳 dict，內容對齊 semg_analysisV4.m 的輸出。"""
    raw_mv = adc_to_mv(raw_counts)
    raw_mv = trim(raw_mv, fs, trim_start_s, trim_end_s)
    t = np.arange(len(raw_mv)) / fs

    dc_offset = raw_mv.mean()
    work = raw_mv - dc_offset

    work = notch_filter(work, fs)
    work = bandpass_filter(work, fs)
    work, hampel_mask = hampel_filter(work)

    env_time, env = rms_envelope(work, fs)

    return {
        "t": t,
        "raw_mv": raw_mv - dc_offset,
        "filtered_mv": work,
        "dc_offset": dc_offset,
        "hampel_outlier_mask": hampel_mask,
        "env_time": env_time,
        "env_mv": env,
    }
