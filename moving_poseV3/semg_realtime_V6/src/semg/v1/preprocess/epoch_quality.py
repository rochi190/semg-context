"""
復刻 SoftPINCH repo 真正的品質控管方法：
    src/SoftPINCH/src/utilities/preprocessing.py 的 RejectBadEpochs.detect_bad_epochs_ptp()

方法（原始碼邏輯，不是我自己發明的）：
    1. 訊號切成一個一個 trial epoch（我們的 index/thumb 是 9s 一個循環，
       grasp/pinch 實測是 21s 一個循環——見 quality.py 的 periodicity_strength
       校準過程，兩個都跟論文的 9s trial_period 概念一致，只是我們的 grasp/pinch
       協定本身循環比較長）。
    2. 每個 epoch、每個 channel，算 RMS envelope 的 peak-to-peak（max-min）。
    3. 同一個 (subject, posture) 群組內，對每個 channel 算 ptp 的 median 跟
       MAD（Median Absolute Deviation），threshold = median + tolerance * MAD
       （tolerance=6，對齊 REJECT_CONFIG_DICT 的 EMG_epoch_rejection_tolerance）。
    4. 任一 channel 的 ptp 超過該 channel 自己的 threshold，這個 epoch 就判 bad
       （EMG_ch_acceptance=0，沒有容忍度，一個 channel 超標就整個 epoch 壞）。

這跟前面 periodicity_strength 那套方法是兩個獨立的判準，一個看「有沒有週期性
節律」，一個看「每個 trial 內部振幅是不是離群」——建議兩個一起看，不是只信一個。

Hampel 參數說明：SoftPINCH repo 裡這個值其實有兩個候選——
`EMGStreamProcessor.__init__` 的 class default 是 window=200, sigma=3（跟
`EMG_preprocessing.preprocessing_routine()` 的函式簽名預設值一樣，也是我們
主要 quality.py pipeline 目前用的值）；但 `classification_pipeline.py` 頂端
`HAMPEL_WINDOWSIZE=100, HAMPEL_SIGMA=2` 這組常數，才是實際透過 EMG_CONFIG_DICT
傳入、真正產生論文報告準確率的訓練資料所用的值。這裡改用 100/2（實際被呼叫
的版本），不是 200/3（只是預設值，沒被覆寫的地方才會用到）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from semg.core import pipeline as pp
from semg.core import quality as qm

TOLERANCE = 6.0          # 對齊 REJECT_CONFIG_DICT['EMG_epoch_rejection_tolerance']
BAD_CH_ACCEPTANCE = 0    # 對齊 REJECT_CONFIG_DICT['EMG_ch_acceptance']：0 = 無容忍度
EPOCH_HAMPEL_HALF_WINDOW = 100
EPOCH_HAMPEL_SIGMA = 2


def process_channel_for_epochs(raw_counts: np.ndarray, trim_start_s: float, trim_end_s: float):
    """跟 pipeline.process_channel 幾乎一樣，只是 Hampel 參數換成 100/2。"""
    mv = pp.adc_to_mv(raw_counts)
    mv = pp.trim(mv, pp.FS, trim_start_s, trim_end_s)
    dc = mv.mean()
    work = mv - dc
    work = pp.notch_filter(work, pp.FS)
    work = pp.bandpass_filter(work, pp.FS)
    work, _ = pp.hampel_filter(work, half_window=EPOCH_HAMPEL_HALF_WINDOW, n_sigma=EPOCH_HAMPEL_SIGMA)
    env_time, env = pp.rms_envelope(work, pp.FS)
    return env_time, env


def epoch_env(env_time: np.ndarray, env: np.ndarray, epoch_sec: float, anchor: bool = True):
    """
    把連續 RMS envelope 切成固定長度、不重疊的 epoch。回傳 (n_epochs, samples_per_epoch)。

    anchor=True（預設）：起點不是陣列第 0 個樣本，而是用
    quality.find_cycle_anchor_index() 找到的「每個週期裡最安靜的相位點」，
    這樣才不會受個別檔案開頭空閒時間長短不一影響，每個 epoch 都是從一次
    循環最安靜的時刻開始、涵蓋完整一次收縮循環。anchor=False 保留舊行為
    （純粹從陣列開頭切），只在需要跟舊結果對照時使用。
    """
    fs_env = 1.0 / np.median(np.diff(env_time))
    samples_per_epoch = int(round(epoch_sec * fs_env))

    start_idx = 0
    if anchor:
        start_idx = qm.find_cycle_anchor_index(env, env_time, epoch_sec)

    remaining = env[start_idx:]
    n_epochs = len(remaining) // samples_per_epoch
    if n_epochs == 0:
        return np.empty((0, samples_per_epoch))
    trimmed = remaining[: n_epochs * samples_per_epoch]
    return trimmed.reshape(n_epochs, samples_per_epoch)


def detect_bad_epochs_ptp(ptp: np.ndarray, tolerance: float = TOLERANCE,
                           bad_ch_acceptance: int = BAD_CH_ACCEPTANCE):
    """直接對照 SoftPINCH repo 的 RejectBadEpochs.detect_bad_epochs_ptp()。
    ptp: shape (n_epochs, n_channels)"""
    med = np.median(ptp, axis=0)
    mad = np.median(np.abs(ptp - med), axis=0)
    mad[mad == 0] = 1e-12
    threshold = med + tolerance * mad

    bad_mask = np.any(ptp > threshold, axis=1)
    bad_indices = np.where(bad_mask)[0]
    for idx in bad_indices:
        n_bad_ch = np.sum(ptp[idx] > threshold)
        if n_bad_ch <= bad_ch_acceptance:
            bad_mask[idx] = False

    return bad_mask, threshold, med, mad
