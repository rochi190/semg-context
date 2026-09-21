"""多域特徵：依 Zhao et al. 2025 Table 2，加上尺度不變的擴充。

為什麼用特徵而非原始波形（與 SoftPINCH 的差別）：
    SoftPINCH 直接把 RMS 包絡線餵給 CNN+LSTM，所以必須 z-score，
    而 z-score 對校準期組成敏感（實測 ±8.8pp 漂移）。
    Zhao et al. 改成先抽多域特徵再進網路，而**多數特徵本身就是尺度不變的**
    （MNF/MDF/PKF/BW/ZC/SSC/SF），振幅特徵也可用相對量表示。
    這樣就繞開了 z-score 的脆弱性。

Zhao et al. Table 2 原本的特徵集：
    RMS, Peak-to-Peak（時域）／Shape Factor（形態）／MNF, MDF（頻域）
    Root Sum of Square（統計）／MFCC1, MFCC3（倒頻譜）

本檔的擴充（都標明理由）：
    + ZC, SSC, WL   —— 經典 Hudgins 特徵，用穩健死區門檻（尺度不變）
    + PKF, BW       —— 頻譜峰值與帶寬，補充 MNF/MDF
    + 跨通道 log 比值與比例 —— v1 實測證實這是唯一無爆炸風險的跨通道表示
                            （z-score 後取比值會因分母趨零而爆掉）
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.fftpack import dct
from scipy.signal import welch

from semg.v2 import config as C

FEAT_VERSION = C.FEAT_VERSION

# 每通道的特徵名（順序即輸出順序，方便除錯與特徵重要度分析）
# ⚠️ 三個修正過的問題（實測抓到，見 docs/pipeline-methodology.md 第 6.4 節）：
#   1. 原本有 `rss_rel`，但 rss = √(Σx²)/√n = √(Σx²/n) = rms —— **與 rms 數學上完全相同**，
#      7 個維度純浪費。改成 WAMP（Willison amplitude），是真正不同的資訊。
#   2. SSC 的門檻比較 `(Δ₁·Δ₂) ≥ eps` 量綱不一致（左 mV²、右 mV）→ 不是尺度不變
#      （實測訊號 ×5 時變化 197%）。改成 `≥ eps²`。
#   3. MFCC 原本餵未正規化的 x，c₀ 是總能量項 → 隨 log(α²) 平移（實測 26%）。
#      改成餵已除以尺度的 xs。
PER_CH_NAMES = [
    "rms_rel", "mav_rel", "p2p_rel", "wamp", "log_rms",         # 振幅（相對化）
    "wl_rel", "zc", "ssc", "shape_factor",                      # 形態
    "mnf", "mdf", "pkf", "bandwidth",                           # 頻域
    "mfcc1", "mfcc3",                                           # 倒頻譜
]
# 跨通道特徵名由 config 推導，改通道數不用動這裡
CROSS_NAMES = (
    [f"logr_{a}_{b}" for a, b in C.CROSS_PAIRS]          # 指定通道對的 log 比值
    + [f"prop_{n}" for n in C.CH_NAMES]                   # 各通道佔總能量比例
    + [f"grp_{g}" for g in C.CH_GROUPS]                    # 各分組佔總能量比例
    + ["logr_distal_proximal"]                             # 遠端/近端 log 能量比
)
# 通道名放進特徵名（不用再回頭查 ch3 是哪塊肌肉）
FEATURE_NAMES = ([f"{name}_{n}" for name in C.CH_NAMES for n in PER_CH_NAMES]
                 + CROSS_NAMES)
N_FEATURES = len(FEATURE_NAMES)


def _mfcc(f: np.ndarray, p: np.ndarray, n_mel: int = 20, n_keep: int = 4) -> np.ndarray:
    """簡化版 MFCC（Zhao et al. 用 MFCC1 與 MFCC3）。

    用 Mel 濾波器組對功率譜取對數後做 DCT。EMG 不是語音，
    但 Zhao et al. 實測這兩個係數對動作狀態有判別力，所以照用。

    ⚠️ 接收**已算好的** PSD。原本這裡自己再呼叫一次 welch，
    等於每通道算兩次功率譜 —— CPU 實測那是特徵抽取的最大單一開銷。
    """
    if p.sum() <= 0:
        return np.zeros(n_keep)
    W = _mel_bank(len(f), float(f[-1]), n_mel)      # 只跟頻率軸有關 → 快取
    bank = W @ p
    return dct(np.log(bank + 1e-12), type=2, norm="ortho")[:n_keep]


@lru_cache(maxsize=8)
def _mel_bank(n_freq: int, f_max: float, n_mel: int) -> np.ndarray:
    """Mel 濾波器組矩陣 (n_mel, n_freq)。頻率軸不變時完全可重用。

    原本每個視窗、每個通道都重建一次濾波器組（含 20 次布林遮罩），
    但頻率軸只取決於 nperseg 與 fs，錄製過程中固定不變。
    快取後 MFCC 變成一次矩陣乘法。
    """
    f = np.linspace(0.0, f_max, n_freq)
    mel = 2595 * np.log10(1 + f / 700.0)
    edges = np.linspace(mel.min(), mel.max(), n_mel + 2)
    W = np.zeros((n_mel, n_freq))
    for i in range(n_mel):
        lo, ctr, hi = edges[i], edges[i + 1], edges[i + 2]
        m = (mel >= lo) & (mel <= hi)
        if m.any():
            w = 1 - np.abs(mel[m] - ctr) / max(hi - lo, 1e-12) * 2
            W[i, m] = np.clip(w, 0, 1)
    return W


def channel_features(x: np.ndarray, scale: float, eps: float) -> list[float]:
    """單通道特徵。

    x     : 濾波後的原始 EMG 視窗（不是包絡線）
    scale : 該通道的穩健尺度（1.4826·MAD，取自整檔或校準期）→ 讓振幅特徵尺度不變
    eps   : ZC/SSC 的死區門檻，抑制雜訊誤觸發
    """
    n = len(x)
    d = np.diff(x)
    xs = x / scale                      # 相對化後再算振幅特徵 → 尺度不變

    rms = float(np.sqrt(np.mean(xs ** 2)))
    mav = float(np.mean(np.abs(xs)))
    p2p = float(xs.max() - xs.min())
    log_rms = float(np.log(rms + 1e-12))
    wl = float(np.sum(np.abs(np.diff(xs))) / n)           # 除以 n → 與視窗長度無關

    # WAMP：相鄰樣本差超過門檻的次數／樣本數。與 ZC/SSC 互補
    #（ZC 看過零、SSC 看斜率反轉、WAMP 看變化幅度），且天然尺度不變
    #（|Δx| 與 eps 同樣隨 α 縮放）。取代了與 rms 完全重複的 rss。
    wamp = float(np.sum(np.abs(d) >= eps) / n)

    zc = int(np.sum((x[:-1] * x[1:] < 0) & (np.abs(d) >= eps)))
    # ★ 門檻用 eps²：左邊是兩個差值的乘積（mV²），與 eps（mV）量綱不符，
    #   直接比較會讓 SSC 隨振幅漂移（實測 ×5 時變化 197%）。
    ssc = int(np.sum((x[1:-1] - x[:-2]) * (x[1:-1] - x[2:]) >= eps ** 2))
    shape = float(rms / (mav + 1e-12))                    # Shape Factor（Zhao）

    # PSD 只算一次，頻域特徵與 MFCC 共用（省掉一半的 welch 呼叫）。
    # ★ 餵 **xs**（已除以尺度）而非 x：MNF/MDF/PKF/BW 因為都被 psum 正規化，
    #   餵哪個都一樣；但 MFCC 的 c₀ 是總能量項，餵 x 會隨 log(α²) 平移
    #   （實測訊號 ×5 時 mfcc1 變化 26%）。餵 xs 才真正尺度不變。
    f_all, p_all = welch(xs, fs=C.FS, nperseg=min(n, 512))
    band = (f_all >= C.BANDPASS[0]) & (f_all <= C.BANDPASS[1])
    f, p = f_all[band], p_all[band]
    psum = p.sum() + 1e-20
    mnf = float((f * p).sum() / psum)
    cum = np.cumsum(p)
    mdf = float(f[np.searchsorted(cum, cum[-1] / 2)])
    pkf = float(f[np.argmax(p)])
    bw = float(np.sqrt(((f - mnf) ** 2 * p).sum() / psum))

    mf = _mfcc(f_all, p_all)
    return [rms, mav, p2p, wamp, log_rms, wl, zc, ssc, shape,
            mnf, mdf, pkf, bw, float(mf[0]), float(mf[2])]


_PAIR_IDX = [(C.ch_index(a), C.ch_index(b)) for a, b in C.CROSS_PAIRS]
_GROUP_IDX = {g: [C.ch_index(n) for n in members] for g, members in C.CH_GROUPS.items()}


def cross_channel_features(win: np.ndarray) -> list[float]:
    """跨通道特徵：log 比值 + 總和歸一化比例 + 遠端/近端能量比。

    v1 的教訓：對 z-score 後訊號取比值會因分母趨零而爆炸（曾造成某檔案 47%，低於亂猜）；
    逐通道各自正規化又會把判別訊號一起消掉（Cohen's d 從 3.88 掉到 0.12）。
    這裡用「視窗內總和歸一化」——尺度不變、有界、無除零風險。

    7 通道版新增**分組能量比**：近端四塊（背闊/三角/三頭/二頭）反映手臂姿態與負載，
    遠端三塊（曲腕/食指/拇指）反映手指動作。遠端/近端比值可以區分
    「同樣的手指動作但手臂姿態不同」，這是 3 通道版做不到的。
    """
    a = np.sqrt(np.mean(win ** 2, axis=0)) + 1e-12       # 每通道 RMS
    la = np.log(a)
    tot = a.sum() + 1e-12
    out = [float(la[i] - la[j]) for i, j in _PAIR_IDX]
    grp = {g: float(a[idx].sum()) for g, idx in _GROUP_IDX.items()}

    # ── prop_<ch>：該通道的能量佔比 ────────────────────────────────
    # ★ v2 起改成**組內**佔比（分母是自己那一組），不再是全 7 通道總和。
    #
    #   為什麼改（實測到的設計缺陷）：用全通道總和當分母時，
    #   **手臂一出力就會把手部通道的佔比壓下去**，即使手指完全沒動。
    #   量化：手部完全靜止、只有手臂動時，各類特徵相對 rest 的位移
    #   （單位 = 該類特徵在 rest 時的標準差）
    #
    #       手臂動作        遠端逐通道(45維)   prop_(7維)
    #       elbow_flex           0.86            2.70
    #       shoulder_flex        0.66            4.19
    #
    #   逐通道特徵幾乎不動（正確，手沒動），prop_ 卻被推了 2.7~4.2 個標準差。
    #   後果：訓練資料裡 hand=index 只跟「手臂不動」共同出現過，
    #   模型學到「index ⟹ prop_index 高」；手臂一動 prop_index 掉下來，
    #   手部頭就認不得了 → 使用者實測「舉 elbow + index 時 hand 不觸發」。
    #
    #   改成組內佔比之後，遠端三通道的比例只跟彼此有關，
    #   近端出力多少都不影響 —— 各自由度在特徵層才真正解耦。
    #
    # ⚠️ 跨組的資訊沒有丟：底下的 `grp_*` 與 `logr_distal_proximal`
    #    就是專門表達跨組關係的，那是**刻意**要有的耦合（3 維），
    #    與 prop_ 那種**不小心**造成的耦合（7 維）不同。
    if FEAT_VERSION >= 2:
        gtot = {g: grp[g] + 1e-12 for g in C.CH_GROUPS}
        ch_grp = {c: g for g, idx in _GROUP_IDX.items() for c in idx}
        out += [float(a[c] / gtot[ch_grp[c]]) for c in range(C.N_CH)]
    else:
        out += [float(v / tot) for v in a]      # v1：全通道總和（舊模型用）

    out += [grp[g] / tot for g in C.CH_GROUPS]
    out.append(float(np.log(grp.get("distal", 1e-12) + 1e-12)
                     - np.log(grp.get("proximal", 1e-12) + 1e-12)))
    return out


def extract(win: np.ndarray, scales: np.ndarray, epss: np.ndarray) -> np.ndarray:
    """win: (n_samples, 3) 濾波後 EMG → (N_FEATURES,)"""
    out: list[float] = []
    for c in range(C.N_CH):
        out += channel_features(win[:, c], float(scales[c]), float(epss[c]))
    out += cross_channel_features(win)
    return np.asarray(out, dtype=np.float32)


def file_scales(filt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """整檔的穩健尺度與死區門檻。

    即時使用時改成從校準期算（同一個函式，餵校準段即可）。
    用 MAD 而非 std：對脈衝雜訊穩健，且對「休息/出力比例」不敏感
    （v1 實測 z-score 用 std 會因組成不同漂移 ±8.8pp）。
    """
    med = np.median(filt, axis=0)
    mad = 1.4826 * np.median(np.abs(filt - med), axis=0) + 1e-12
    return mad, 0.05 * mad          # 死區 = 5% 穩健標準差
