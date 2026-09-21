"""前處理：因果濾波（可直接用於即時串流）+ 包絡線。

沿用 v1 已驗證的實作（`semg/realtime/control.py::StreamingPreprocessor` 的邏輯），
因為那部分已經在 14 個錄音上跑過、且封包/LSB 換算與 ads1299_readerV6.py 對齊過。
差別只在這裡同時回傳「整流後的訊號」與「包絡線」——
整流是 Lee et al. 的做法，包絡線用於自動標籤與部分特徵。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi

from semg.v2 import config as C


def adc_to_mv(counts: np.ndarray) -> np.ndarray:
    return counts.astype(np.float64) * C.LSB_MV


def load_recording(csv_path, trim: bool = True) -> np.ndarray:
    """讀 ADS1299 CSV，回傳 (n_samples, N_CH) mV、已去 DC。

    通道由 `C.CH_CSV_COLS` 指定（目前 ch1–ch7 = 背闊/三角/三頭/二頭/曲腕/食指/拇指）。
    只取存在的欄位 —— 舊的 3 通道錄音也讀得進來（缺的通道補 0 並在回傳時警告）。

    trim=False 時**不裁切頭尾**：用 cue 檔標籤時樣本索引必須與 CSV 對齊，
    任何裁切都會讓 cue 的 sample_index 對不上。
    """
    head = pd.read_csv(csv_path, nrows=0)
    want = [f"ch{c}" for c in C.CH_CSV_COLS]
    have = [c for c in want if c in head.columns]
    if not have:
        raise ValueError(f"{csv_path} 找不到任何 {want} 欄位")
    df = pd.read_csv(csv_path, usecols=have)
    cols = []
    for c in want:
        cols.append(adc_to_mv(df[c].values) if c in have
                    else np.zeros(len(df), dtype=np.float64))
    raw = np.stack(cols, axis=1)
    if len(have) < len(want):
        missing = [c for c in want if c not in have]
        print(f"  ⚠️ {Path(csv_path).name} 缺通道 {missing}（補 0）——"
              f"這是 3 通道舊資料，特徵會失真")
    if trim:
        s = int(C.TRIM_START_SEC * C.FS)
        e = len(raw) - int(C.TRIM_END_SEC * C.FS)
        raw = raw[s:e]
    return raw - raw.mean(axis=0)


def _design():
    b_n, a_n = iirnotch(C.NOTCH_FREQ, C.NOTCH_Q, fs=C.FS)
    b_b, a_b = butter(C.BANDPASS_ORDER,
                      [C.BANDPASS[0] / (C.FS / 2), C.BANDPASS[1] / (C.FS / 2)],
                      btype="band")
    return (b_n, a_n), (b_b, a_b)


HAMPEL_K = 2 * C.HAMPEL_HALF_WINDOW + 1


class StreamFilter:
    """★ 唯一的前處理實作：notch → bandpass → 因果 Hampel，全部保留狀態。

    ## 為什麼離線也要用這個（實測踩過的坑）

    原本離線走 `filter_causal`（含**置中視窗**的 Hampel），
    即時走另一份只有 notch+bandpass 的實作。兩者產出的訊號
    **相關係數只有 0.85**，MFCC3 之類的特徵差到 600%，
    結果訓練 99.6% 全對、即時重播只剩 58%，97% 的預測都變成 rest。

    修法不是「把 Hampel 補進串流版」，而是**只留一份實作**：
    離線的 `filter_causal()` 直接把整段訊號餵給這個類別。
    兩條路徑不可能再分岔 —— 這是結構性的保證，不是靠記得同步。

    ## 為什麼 Hampel 改成因果

    原本用 `median_filter(mode="nearest")`，視窗置中 → 每個樣本會用到
    **未來 100 個樣本**，即時根本拿不到（函式卻叫 `filter_causal`）。
    改用 `origin=K//2` 把視窗整個移到過去側，涵蓋 [i-200, i]。
    代價是對突波的抑制稍差，換來離線與即時**完全一致**。
    """

    def __init__(self, n_ch: int = C.N_CH, use_hampel: bool | None = None,
                 denoise: "dict | None" = None):
        (b_n, a_n), (b_b, a_b) = _design()
        # ★ 兩段 IIR 拆開持有，因為降噪的 comb/lms 要插在**中間**
        #   （市電諧波要在帶通把它們埋進通帶前處理掉）。
        self.notch = (b_n, a_n)
        self.band = (b_b, a_b)
        self.stages = [(b_n, a_n), (b_b, a_b)]        # 保留給既有程式讀取
        self.zi = [[lfilter_zi(b, a) for _ in range(n_ch)] for b, a in self.stages]
        self._primed = False
        # ⚠️ 與 use_hampel 同理：開了哪些降噪會改變輸入分布。
        #    但設計目標不同 —— 降噪的目標是把「吵環境」映射回「乾淨環境」，
        #    所以**用乾淨資料訓練的模型搭配降噪**才是預期用法。
        #    前提是每個演算法都通過「乾淨訊號中性測試」，見 denoise.py 檔頭。
        from semg.v2.denoise import DenoiseChain
        d = dict(denoise or {})
        if "hampel" in d:                              # hampel 由既有機制處理
            if d.pop("hampel"):
                use_hampel = True
        self.denoise = DenoiseChain(n_ch, **d) if d else None
        # ⚠️ 關掉 Hampel 會改變輸入分布 → 模型必須用同樣的設定重訓。
        #    所以這個值會被寫進 bundle，推論時由 bundle 決定而不是由 config 決定。
        self.use_hampel = C.USE_HAMPEL if use_hampel is None else bool(use_hampel)
        # Hampel 是**兩段式**（先算 median，再對 |z-med| 算 median）：
        # 第二段會吃到第一段的輸出，所以要保留 2×(K-1) 個樣本，
        # 只留 K-1 會讓 MAD 視窗吃到「用 padding 算出來的 med」→ 與離線不一致。
        self.hist = np.zeros((0, n_ch))

    def __call__(self, block: np.ndarray) -> np.ndarray:
        x = np.asarray(block, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]

        def _iir(x, si):
            b, a = self.stages[si]
            out = np.empty_like(x)
            for c in range(x.shape[1]):
                if not self._primed:
                    # 用第一個樣本初始化，避免起始暫態（v1 已驗證過的做法）
                    self.zi[si][c] = self.zi[si][c] * x[0, c]
                out[:, c], self.zi[si][c] = lfilter(b, a, x[:, c],
                                                    zi=self.zi[si][c])
            return out

        x = _iir(x, 0)                                  # 60Hz 陷波
        if self.denoise is not None:
            x = self.denoise.pre_bandpass(x)            # comb / lms
        x = _iir(x, 1)                                  # 20–450Hz 帶通
        if self.denoise is not None:
            x = self.denoise.post_bandpass(x)           # car / wavelet
        self._primed = True

        if not self.use_hampel:
            return x                     # notch + 帶通就結束（Lee / Zhao 的做法）

        # 因果 Hampel：把先前保留的樣本接在前面，算完再切掉
        z = np.concatenate([self.hist, x]) if len(self.hist) else x
        med = median_filter(z, size=(HAMPEL_K, 1), mode="nearest",
                            origin=(HAMPEL_K // 2, 0))
        mad = 1.4826 * median_filter(np.abs(z - med), size=(HAMPEL_K, 1),
                                     mode="nearest", origin=(HAMPEL_K // 2, 0))
        filt = np.where(np.abs(z - med) > C.HAMPEL_SIGMA * mad, med, z)
        res = filt[len(self.hist):]
        self.hist = z[-2 * (HAMPEL_K - 1):]
        return res


def filter_causal(x: np.ndarray, use_hampel: bool | None = None,
                  denoise: "dict | None" = None) -> np.ndarray:
    """(n, ch) → (n, ch)。離線版 = 把整段餵給 StreamFilter。

    刻意**不另外實作**：只要有第二份實作，就會有第二種行為。
    """
    return StreamFilter(n_ch=x.shape[1], use_hampel=use_hampel,
                        denoise=denoise)(x)


def rms_envelope(filt: np.ndarray) -> np.ndarray:
    """(n, ch) → (m, ch)，40 Hz 包絡線。用於自動標籤。"""
    w, s = C.ENV_WINDOW, C.ENV_STEP
    n_out = (len(filt) - w) // s + 1
    if n_out <= 0:
        return np.zeros((0, filt.shape[1]))
    idx = np.arange(n_out) * s
    return np.stack([
        np.sqrt(np.mean(np.stack([filt[i:i + w, c] for i in idx]) ** 2, axis=1))
        for c in range(filt.shape[1])
    ], axis=1)


def robust_scale(x: np.ndarray) -> np.ndarray:
    """每通道的穩健尺度 1.4826·MAD。用於讓振幅特徵變成尺度不變。

    比 std 穩健（不被少數尖峰帶偏），也比 z-score 對「休息/出力比例」不敏感——
    v1 實測 z-score 會因校準期組成不同漂移 ±8.8pp。
    """
    med = np.median(x, axis=0)
    return 1.4826 * np.median(np.abs(x - med), axis=0) + 1e-12
