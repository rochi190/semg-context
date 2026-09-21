"""
訊號品質特徵萃取 + good/bad 分類。

**2026-07-20 更新：`score_and_classify()` 的 good/bad 結論已從報告/簡報移除**，
不要在對外報告裡引用。問題是 periodicity_strength 量的是「動作節律夠不夠規律」，
不是「訊號乾不乾淨」，同一份錄音的不同 channel 會被判成不一致的結果（門檻
0.15 附近極不穩定），沒有論文依據，難以說明。`compute_channel_features()` 底下的
特徵數字本身還能用（診斷參考），但不要再用 `score_and_classify()` 的 label 下結論。

v2（目前版本）：絕對門檻分類，用 periodicity_strength 為主。
    v1 用「同一個 channel 編號的樣本群組內做 z-score + GMM」分 good/bad，
    問題是這是相對排名，不是絕對品質判斷——不管整批資料實際多好或多爛，
    都會硬切出約一半 good、一半 bad。拿 SoftPINCH 論文自己的 EMG 資料驗證後
    直接證實這個問題（dynamic range 21dB、肉眼看是清楚週期性尖峰的通道被判
    bad；dynamic range 2dB、幾乎是平的通道卻被判成分數最高的 good）。

    v2 改成量測「RMS envelope 有沒有出現跟任務動作節律鎖定的週期性」
    （periodicity_strength，見下方函式），這是絕對指標，同一個門檻套用在
    所有資料上，不會因為同一批裡剛好大家都差/都好就湊出一半一半。

特徵（每個 channel 各自算一組）：
    periodicity_strength   : 去趨勢後的 RMS envelope autocorrelation，在合理任務
                              週期範圍內找真正的局部極大值。主要分類依據。
    hampel_outlier_frac    : Hampel filter 標記的脈衝型離群值比例，絕對門檻，
                              過高直接判 bad（不自主抽搐/crosstalk/接觸不良）。
    env_dynamic_range_db, env_cv, broadband_noise_ratio, mean_freq_hz：
                              輔助診斷用，目前不參與分類，留著給人工複查時參考。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal


def welch_band_features(filtered_mv: np.ndarray, fs: int, low_band=(20, 150),
                         high_band=(150, 450)):
    win = min(round(1.0 * fs), max(len(filtered_mv) // 4, 64))
    freqs, psd = signal.welch(filtered_mv, fs=fs, nperseg=win, noverlap=win // 2)

    def band_power(band):
        mask = (freqs >= band[0]) & (freqs <= band[1])
        return np.trapz(psd[mask], freqs[mask])

    low_power = band_power(low_band)
    high_power = band_power(high_band)
    eps = 1e-12
    broadband_noise_ratio = high_power / max(low_power, eps)

    full_mask = (freqs >= low_band[0]) & (freqs <= high_band[1])
    mean_freq = np.sum(freqs[full_mask] * psd[full_mask]) / max(np.sum(psd[full_mask]), eps)

    return broadband_noise_ratio, mean_freq


def periodicity_strength(env: np.ndarray, env_time: np.ndarray,
                          min_period_s: float = 2.5, max_period_s: float = 45.0):
    """
    量測 RMS envelope 有沒有出現「跟任務動作節律鎖定」的週期性峰值，用正規化
    autocorrelation：在 [min_period_s, max_period_s] 這個合理的任務週期範圍內找
    autocorrelation 的最大值（已經除以 lag=0 的 variance，所以範圍大致在
    [-1, 1]，類似相關係數）。

    這是絕對指標，不是跟同一批資料裡其他樣本比較出來的相對分數：乾淨、動作鎖定
    的收縮/放鬆循環會在這個 lag 範圍內產生明顯的 autocorrelation 峰值；雜訊蓋掉
    看不出收縮節律的訊號則不會，不管訊號振幅或跟別的樣本比起來如何。

    max_period_s=45 是實測校準過的：index/thumb（3s 彎曲+6s 放鬆）週期約 9s，
    grasp/pinch（抓握-舉起-放下-放鬆四階段）週期約 21s，兩者都要涵蓋在搜尋範圍
    內，原本設 20s 太窄，把 grasp/pinch 整批系統性判成 bad（不是資料真的差，
    是週期比搜尋窗口長，偵測不到）。
    """
    if len(env) < 4:
        return 0.0, 0.0
    fs_env = 1.0 / np.median(np.diff(env_time))

    # 先去除慢速趨勢（電極接觸阻抗漂移、姿勢改變造成的基線緩慢上升/下降）。
    # 不去趨勢的話，任何平滑、緩慢上升的訊號（不管有沒有真的週期性收縮）都會在
    # 長 lag 出現看起來很高的 autocorrelation，跟真正的動作鎖定週期峰值混在一起
    # 分不出來——這是原本版本會把很多其實沒有清楚收縮節律的訊號誤判成 good 的主因。
    detrend_win = min(len(env) - 1, max(3, int(round(max_period_s * 2 * fs_env))))
    if detrend_win >= 3:
        from scipy.ndimage import uniform_filter1d
        trend = uniform_filter1d(env, size=detrend_win, mode="nearest")
        env_c = env - trend
    else:
        env_c = env - env.mean()

    var0 = np.dot(env_c, env_c)
    if var0 <= 0:
        return 0.0, 0.0

    autocorr_full = np.correlate(env_c, env_c, mode="full")
    autocorr = autocorr_full[len(autocorr_full) // 2:] / var0

    min_lag = max(1, int(round(min_period_s * fs_env)))
    max_lag = min(len(autocorr) - 1, int(round(max_period_s * fs_env)))
    if min_lag >= max_lag:
        return 0.0, 0.0

    search = autocorr[min_lag: max_lag + 1]

    # 只接受「真正的局部極大值」，排除單調爬升撞到搜尋範圍邊界的假峰值
    # （那種邊界最大值通常代表殘餘趨勢或非週期性慢變化，不是真的動作節律）。
    peaks, _ = signal.find_peaks(search)
    if len(peaks) == 0:
        return 0.0, 0.0

    best = peaks[np.argmax(search[peaks])]
    peak_val = float(search[best])
    peak_period_s = (min_lag + best) / fs_env
    return peak_val, peak_period_s


def find_cycle_anchor_index(env: np.ndarray, env_time: np.ndarray, period_s: float) -> int:
    """
    給定已知週期，用「epoch-synchronous averaging」找出每個週期裡最安靜的那一段
    plateau 的「起點」，回傳它在 env 陣列裡的樣本 index（[0, period_samples) 範圍內）。

    這是為了解決「每個檔案開頭的空閒時間長短不一」的問題：與其用同一個固定秒數
    裁切當 epoch 起點，改成把整段訊號依照已知週期折疊、逐點平均出一個代表性單
    週期波形，找出其中最長的一段低於中位數的安靜區間（對應 rest 那一段），用它的
    起點當錨點——而不是整個週期裡「單一一個最低點」。

    這裡的關鍵是「起點」不是「最低點」：一段幾秒鐘的安靜 plateau，最低點通常落在
    plateau 中間某處，不是 plateau 剛開始的那一刻。如果拿最低點當每個 epoch 的
    起點，會把 epoch 邊界往後偏移到 plateau 中間，導致三等分（rest/onset/offset）
    切出來的每一段內容都跟真實的動作階段對不齊，係統性地把 offset 段的尾巴跟下一
    個 trial 的 rest 開頭混在一起判斷（複測時觀察到 Release 大量被誤判成 Rest，
    正是這種邊界偏移的典型徵狀）。
    """
    fs_env = 1.0 / np.median(np.diff(env_time))
    period_samples = int(round(period_s * fs_env))
    n_folds = len(env) // period_samples
    if n_folds < 2 or period_samples < 2:
        return 0
    folded = env[: n_folds * period_samples].reshape(n_folds, period_samples)
    template = folded.mean(axis=0)

    threshold = np.median(template)
    below = template < threshold
    below_doubled = np.concatenate([below, below])  # 處理 plateau 跨越週期邊界（頭尾相接）的情況

    best_start, best_len = 0, 0
    cur_start, cur_len = None, 0
    for i, v in enumerate(below_doubled):
        if v:
            if cur_start is None:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_start, cur_len = None, 0
        if i >= 2 * period_samples - 1:
            break

    return best_start % period_samples


def compute_channel_features(res: dict, fs: int) -> dict:
    env = res["env_mv"]
    eps = 1e-12
    p5, p80, p95 = np.percentile(env, [5, 80, 95])
    # 用「80-95th percentile 的平均值」當作 active plateau，而不是直接用 p95：
    # 單一兩個尖峰（動作偽影/接觸瞬間雜訊）就能把 p95 撐高，讓一個大部分時間死平
    # 的通道被誤判成 dynamic range 很大的「好」訊號。取一段 percentile 區間的平均，
    # 逼近「真的有沒有持續一段時間的收縮 plateau」，對單點離群值更穩健。
    active_plateau = env[(env >= p80) & (env <= p95)]
    active_level = active_plateau.mean() if len(active_plateau) > 0 else p95
    env_dynamic_range_db = 20 * np.log10(max(active_level, eps) / max(p5, eps))
    env_cv = env.std() / max(env.mean(), eps)
    hampel_outlier_frac = res["hampel_outlier_mask"].mean()
    broadband_noise_ratio, mean_freq = welch_band_features(res["filtered_mv"], fs)
    periodicity, periodicity_period_s = periodicity_strength(env, res["env_time"])

    return {
        "periodicity_strength": periodicity,
        "periodicity_period_s": periodicity_period_s,
        "env_dynamic_range_db": env_dynamic_range_db,
        "env_cv": env_cv,
        "hampel_outlier_frac": hampel_outlier_frac,
        "broadband_noise_ratio": broadband_noise_ratio,
        "mean_freq_hz": mean_freq,
        "env_mean_mv": env.mean(),
        "env_p5_mv": p5,
        "env_p95_mv": p95,
        "env_active_level_mv": active_level,
    }


# 絕對門檻（不是跟同一批資料裡其他樣本比較出來的），校準方式見
# docs/quality_threshold_calibration 討論：用 periodicity_strength 的分布
# 對照已經目視確認過的 good/bad 面板反推。
PERIODICITY_GOOD_THRESHOLD = 0.15
HAMPEL_OUTLIER_ABS_THRESHOLD = 0.03  # 3% 以上樣本被 Hampel 判定離群值，直接判 bad


def score_and_classify(df: pd.DataFrame, channel_col: str = "channel") -> pd.DataFrame:
    """
    絕對門檻分類，不做組內 z-score/GMM 相對排名。

    label = bad，如果：
        periodicity_strength < PERIODICITY_GOOD_THRESHOLD（訊號看不出跟任務動作
        鎖定的收縮/放鬆節律）
        或 hampel_outlier_frac > HAMPEL_OUTLIER_ABS_THRESHOLD（脈衝雜訊比例過高）
    否則 label = good。

    這樣同一個門檻套用在所有資料上（包含不同資料集之間），不會因為同一批裡剛好
    大家都差、或剛好大家都好，就硬凑出一半一半的 good/bad。
    """
    df = df.copy()
    df["quality_score"] = df["periodicity_strength"]
    is_bad = (
        (df["periodicity_strength"] < PERIODICITY_GOOD_THRESHOLD)
        | (df["hampel_outlier_frac"] > HAMPEL_OUTLIER_ABS_THRESHOLD)
    )
    df["label"] = np.where(is_bad, "bad", "good")
    return df
