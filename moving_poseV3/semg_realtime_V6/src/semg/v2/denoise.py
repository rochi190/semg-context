"""環境雜訊抑制：可獨立開關、可組合的因果濾波級。

## 為什麼需要這個

錄訓練資料的地方很安靜，但機械手臂放在**實驗室**（運轉中的電器很多）。
兩者的雜訊環境不同 → 離線量到的數字不保證能轉移到實機。

★ 這裡所有演算法的設計目標是**同一件事**：
  把「吵的環境下錄到的訊號」映射回「安靜環境下的樣子」，
  好讓**用乾淨資料訓練的模型**在吵的環境仍然適用。

  所以每個演算法都必須通過兩個測試：
    ① 乾淨訊號中性  —— 開了不能讓乾淨錄音的準確率變差
    ② 吵訊號可回復  —— 加噪後開了要把準確率救回來
  只過②不過①的演算法**不能用** —— 那會讓模型在乾淨環境下反而變糟。

## ⚠️ 為什麼掛在 StreamFilter 裡，不另外寫一份

本專案踩過：離線與即時各有一份前處理實作，產出訊號相關係數只有 0.85，
訓練 99.6% 全對、即時重播剩 58%。修法是**只留一份實作**。
新增的每一級都必須進 `StreamFilter`，離線與即時才不可能分岔。

## 各演算法

| 旗標 | 名稱 | 對付什麼 | 位置 |
|---|---|---|---|
| `comb` | 諧波梳狀陷波 | 120/180/…Hz 市電諧波（60Hz 陷波器**濾不掉**） | 帶通前 |
| `lms` | 自適應市電消除 | 會漂移的市電基頻+諧波，且**不挖頻譜洞** | 帶通前 |
| `car` | 分組共同平均參考 | 共模環境場（所有電極收到同一個干擾） | 帶通後 |
| `hampel` | 突波抑制 | 繼電器/馬達切換的脈衝干擾 | 最後 |
| `wavelet` | 小波軟閾值 | 寬頻雜訊 | 帶通後 |

`nf`（雜訊底噪補償）不是濾波器，作用在**校準**上，見 `infer.Recognizer`。
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi

from semg.v2 import config as C

# ── 各演算法的預設參數（改這裡，不要散在別處）────────────────────
HARMONICS = (120.0, 180.0, 240.0, 300.0, 360.0, 420.0)   # 60Hz 的 2~7 次諧波
COMB_Q = 30.0                    # 與基頻陷波器相同 → 每個凹陷 f0/Q 寬

LMS_FREQS = (60.0, 120.0, 180.0, 240.0)   # 只追前四個，更高次能量已很小
LMS_MU = 2e-3                    # ⚠️ 太大會把肌電也學走，見 __call__ 的說明

# ⚠️ **必須是 mean，不能是 median。**
#    第一版用 median（想說比較穩健），但遠端只有 3 個通道 ——
#    3 個數的中位數**就是其中一個**，所以 `sub - median(sub)` 會把
#    「當下是中位的那條通道」**逐樣本歸零**。那不是穩健，那是把訊號毀掉，
#    而且是隨樣本跳動的非線性破壞（還會讓後面的小波軟閾值除以零變成 NaN）。
#    mean 的話每條通道只減掉 1/len(group) 的共同成分，沒有通道被歸零。
CAR_MODE = "mean"
WAVELET = "db4"
WAVELET_LEVEL = 4


class CombNotch:
    """60Hz 諧波的梳狀陷波（120/180/…/420Hz）。

    ★ 為什麼需要：`config.NOTCH_FREQ` 的陷波器只殺 60Hz 基頻，
      諧波全部落在 20–450Hz 通帶內，原封不動進到特徵。

    ★ 證據：實測諧波（120–420Hz）功率佔比與該份錄音的 LORO 準確率
      相關 **r = −0.849 (p = 0.016, n = 7)**，而陷波**後**的 60Hz 殘餘
      與準確率不相關（r = −0.166）。也就是說基頻已經被處理掉了，
      **還在傷害準確率的是諧波。**

    ⚠️ 代價：6 個凹陷 × (f/Q) 共約 12 Hz，佔 430 Hz 通帶的 2.8%。
      肌電在這些頻率有能量，所以這是**有代價的**，不是免費的。
    """

    def __init__(self, n_ch: int = C.N_CH, freqs=HARMONICS, q: float = COMB_Q):
        self.stages = [iirnotch(f, q, fs=C.FS) for f in freqs
                       if f < C.FS / 2 - 10]
        self.zi = [[lfilter_zi(b, a) for _ in range(n_ch)] for b, a in self.stages]
        self._primed = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for si, (b, a) in enumerate(self.stages):
            out = np.empty_like(x)
            for c in range(x.shape[1]):
                if not self._primed:
                    self.zi[si][c] = self.zi[si][c] * x[0, c]
                out[:, c], self.zi[si][c] = lfilter(b, a, x[:, c], zi=self.zi[si][c])
            x = out
        self._primed = True
        return x


class LMSMains:
    """自適應市電消除：合成參考弦波 + LMS 權重追蹤。

    ## 比陷波器好在哪

    陷波器在 60Hz 挖一個永久的洞 —— 不管當下有沒有干擾，
    **肌電在那個頻率的成分也一起被挖掉**。

    LMS 只減去「與參考弦波同調的成分」：

        ŷ[n] = w_c·cos(2πf n/fs) + w_s·sin(2πf n/fs)
        e[n] = x[n] − ŷ[n]
        w ← w + μ·e[n]·r[n]

    干擾大時權重自動長大，干擾消失時權重衰減回 0 → **不留下頻譜洞**。
    而且它會追蹤市電的振幅與相位漂移（市電不是剛好 60.000 Hz）。

    ⚠️ **μ 的取捨**：μ 太大 → 權重會去擬合肌電裡的隨機成分，
      等於把訊號也減掉。μ = 2e-3 是保守值（時間常數約
      1/(μ·fs) ≈ 0.25 s，遠慢於肌電的變化）。
      調大之前一定要跑「乾淨訊號中性測試」。

    ⚠️ 這一級與 `comb` **功能重疊**。兩個都開不會更好，
      只會多付一次計算與一次失真。文件裡有實測對照。
    """

    def __init__(self, n_ch: int = C.N_CH, freqs=LMS_FREQS, mu: float = LMS_MU):
        self.f = np.asarray(freqs, dtype=np.float64)      # (K,)
        self.mu = float(mu)
        self.wc = np.zeros((len(self.f), n_ch))           # (K, ch)
        self.ws = np.zeros((len(self.f), n_ch))
        self.n = 0                                        # 全域樣本計數（相位連續）

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # ★ **逐樣本**更新，不做區塊近似。
        #
        #   第一版寫成「用區塊平均更新一次」，結果分塊餵與整段餵的輸出
        #   差到訊號振幅的 2% —— 也就是離線評估與即時部署會分岔，
        #   正是本專案踩過最貴的那個坑（相關係數 0.85、99.6% → 58%）。
        #   逐樣本更新讓輸出**與區塊切法無關**，這是正確性要求，不是效能取捨。
        out = np.asarray(x, dtype=np.float64).copy()
        w = 2.0 * np.pi * self.f / C.FS                   # (K,)
        for i in range(len(out)):
            ph = w * (self.n + i)                         # (K,)
            rc = np.cos(ph)[:, None]                      # (K,1)
            rs = np.sin(ph)[:, None]
            est = (self.wc * rc + self.ws * rs).sum(axis=0)   # (ch,)
            err = out[i] - est
            self.wc += self.mu * rc * err                 # (K,ch)
            self.ws += self.mu * rs * err
            out[i] = err
        self.n += len(out)
        return out


class GroupCAR:
    """分組共同平均參考：每個通道減去**自己那一組**的共模成分。

    ## 原理

    環境電磁場對所有電極的耦合量幾乎相同（共模），而肌電是**局部**的。
    減掉跨通道的共同成分，就減掉了環境干擾而保留肌電。

    ## ★ 為什麼是「分組」而不是全部 7 通道一起

    這是**架構一致性**的要求，不是調參：

    `SplitMultiHead` 的整個設計目的就是讓近端與遠端**完全不互通**
    （留一組合測試 5.0% → 85.4%）。若做全陣列 CAR，
    每個遠端通道都會被減去含有近端活動的平均值 ——
    **等於把我們特地切斷的耦合又接回去**。

    所以近端 4 通道彼此做 CAR、遠端 3 通道彼此做 CAR，兩組不互相污染。

    ## ⚠️ 這一級風險最高

    遠端只有 3 個通道。若三者的肌電本來就相關（前臂肌群串擾本來就有），
    減掉共同成分會**連訊號一起減掉**。用 median 而非 mean 可以緩解
    （單一通道大幅出力時不會拉動估計值），但緩解不等於沒有。

    → 必須通過「乾淨訊號中性測試」才能用。文件裡有實測數字。
    """

    def __init__(self, mode: str = CAR_MODE):
        self.mode = mode
        self.groups = [[C.CH_NAMES.index(n) for n in ch]
                       for ch in C.CH_GROUPS.values()]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = x.copy()
        for idx in self.groups:
            if len(idx) < 2:
                continue
            sub = x[:, idx]
            if self.mode == "median" and len(idx) <= 3:
                raise ValueError(
                    f"通道數 {len(idx)} ≤ 3 時不可用 median CAR —— "
                    "中位數就是其中一條通道，會把它逐樣本歸零（見 CAR_MODE 說明）")
            com = (np.median(sub, axis=1) if self.mode == "median"
                   else np.mean(sub, axis=1))
            out[:, idx] = sub - com[:, None]
        return out


class WaveletDenoise:
    """小波軟閾值去噪（固定內部分塊 + 歷史緩衝，因果且**區塊不變**）。

    對 [歷史 512 + 本塊] 做 stationary wavelet transform，
    用持續更新的雜訊估計套 universal threshold σ√(2 ln N) 軟閾值，
    逆轉換後只取本塊對應的部分。

    ## ⚠️ 為什麼要固定內部分塊

    第一版直接對「傳進來的區塊」做 SWT，結果：

        分塊餵（即時，每塊 100 樣本） vs 整段餵（離線，4000 樣本）
        → 相對誤差 **91%**

    也就是離線評估與即時部署會得到**完全不同的訊號** ——
    正是本專案最貴的那個坑（前處理兩份實作，99.6% → 58%）。

    修法：不管外面餵多大的區塊，內部一律切成 `C.STEP_SAMPLES` 處理，
    並保留固定長度的歷史。這樣「第 i 個樣本的輸出」只由它前面固定長度的
    樣本決定，與外部怎麼切塊無關。

    ## ⚠️ 仍然列為實驗性，理由不變

      ① universal threshold 假設雜訊是**高斯白雜訊**，
         但市電干擾是窄頻確定性訊號 —— 小波對它的效率不如陷波/LMS
      ② 計算成本明顯高於其他級（見文件的實測表）
      ③ 軟閾值會壓低**所有**小幅訊號，包含 chou 那種本來就很小的肌電

      文獻上 EMG 去噪常引用小波，所以實作出來給你比較用。
    """

    HIST = 512                      # 歷史長度，需 > db4/level4 的有效支撐(~112)

    def __init__(self, n_ch: int = C.N_CH, wavelet: str = WAVELET,
                 level: int = WAVELET_LEVEL):
        import pywt
        self.pywt = pywt
        self.wavelet = wavelet
        self.level = level
        self.chunk = C.STEP_SAMPLES
        self.hist = np.zeros((self.HIST, n_ch))
        self.sigma = np.zeros(n_ch)            # 持續的雜訊估計（EMA）
        self._seen = False

    def _one(self, blk: np.ndarray) -> np.ndarray:
        pywt = self.pywt
        z = np.concatenate([self.hist, blk])
        n = len(z)
        pad = (-n) % (2 ** self.level)
        zp = np.pad(z, ((0, pad), (0, 0)), mode="edge") if pad else z
        out = np.empty_like(blk)
        for c in range(blk.shape[1]):
            co = pywt.swt(zp[:, c], self.wavelet, level=self.level,
                          trim_approx=True, norm=True)
            det = co[1:]
            s_now = np.median(np.abs(det[-1])) / 0.6745
            # EMA：讓閾值不會因單一塊的內容劇烈跳動（也讓它與塊內容的耦合變弱）
            self.sigma[c] = s_now if not self._seen else 0.9 * self.sigma[c] + 0.1 * s_now
            thr = self.sigma[c] * np.sqrt(2 * np.log(max(n, 2)))
            # ⚠️ 訊號恰好為 0 時 pywt 的軟閾值會 0/0 → NaN。自己算，不呼叫它。
            new = [co[0]] + [np.sign(d) * np.maximum(np.abs(d) - thr, 0.0)
                             for d in det]
            rec = pywt.iswt(new, self.wavelet, norm=True)[:n]
            out[:, c] = rec[self.HIST:]
        self._seen = True
        self.hist = z[-self.HIST:]
        return out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # ★ 一律切成固定大小處理 → 輸出與外部的切塊方式無關
        parts = [self._one(x[i:i + self.chunk])
                 for i in range(0, len(x), self.chunk)]
        return np.concatenate(parts) if parts else x


# ── 設定物件 ────────────────────────────────────────────────────
ALGOS = ("comb", "lms", "car", "hampel", "wavelet")


class DenoiseChain:
    """把選用的演算法組裝成一條鏈。由 `StreamFilter` 持有。

    順序是固定的，不開放調整 —— 每一級都有它該在的位置：

        [原始] → notch(60) → **comb** → **lms** → bandpass → **car**
               → **wavelet** → **hampel** → [特徵]

    comb / lms 在帶通**前**：市電與諧波要在還看得到的時候處理掉。
    car 在帶通**後**：帶通已經去掉各通道不同的直流偏壓，
                      否則 CAR 會把偏壓差當成共模訊號減錯。
    hampel 在最後：它抓的是殘餘的脈衝，要等其他級都處理完再看。
    """

    def __init__(self, n_ch: int = C.N_CH, **flags):
        unknown = set(flags) - set(ALGOS)
        if unknown:
            raise ValueError(f"未知的降噪演算法 {sorted(unknown)}，可選 {ALGOS}")
        self.flags = {a: bool(flags.get(a, False)) for a in ALGOS}
        self.comb = CombNotch(n_ch) if self.flags["comb"] else None
        self.lms = LMSMains(n_ch) if self.flags["lms"] else None
        self.car = GroupCAR() if self.flags["car"] else None
        self.wav = WaveletDenoise(n_ch) if self.flags["wavelet"] else None

    @property
    def active(self) -> list[str]:
        return [a for a in ALGOS if self.flags[a]]

    def pre_bandpass(self, x: np.ndarray) -> np.ndarray:
        if self.comb is not None:
            x = self.comb(x)
        if self.lms is not None:
            x = self.lms(x)
        return x

    def post_bandpass(self, x: np.ndarray) -> np.ndarray:
        if self.car is not None:
            x = self.car(x)
        if self.wav is not None:
            x = self.wav(x)
        return x

    def __repr__(self):
        return f"DenoiseChain({', '.join(self.active) or 'off'})"


def parse_flags(spec: str | None) -> dict:
    """把 `--denoise comb,car` 這種字串轉成旗標 dict。"""
    if not spec:
        return {}
    out = {}
    for tok in str(spec).replace(" ", "").split(","):
        if not tok:
            continue
        if tok not in ALGOS:
            raise SystemExit(f"未知的降噪演算法 {tok!r}，可選：{', '.join(ALGOS)}")
        out[tok] = True
    return out


# ── 雜訊底噪估計（給校準用，不是濾波器）──────────────────────────
def noise_floor_rms(filt_block: np.ndarray,
                    band: tuple[float, float] = (350.0, 450.0)) -> np.ndarray:
    """從**高頻帶**估每通道的寬頻雜訊底噪 RMS（mV）。

    ★ 原理：350–450 Hz 幾乎沒有肌電能量（sEMG 99% 在 450Hz 以下，
      主峰在 50–150Hz），但寬頻儀器/環境雜訊在這裡仍然滿的。
      所以這一段的功率密度可以當成「白雜訊底」，外推到整個 20–450Hz。

    ⚠️ 這個假設在**窄頻**干擾（市電諧波）下不成立 ——
      那種干擾是線狀譜不是白的。所以 `nf` 要跟 `comb`/`lms` 一起用：
      先把線狀譜殺掉，剩下的才適合用這個模型估。
    """
    from scipy.signal import welch
    f, p = welch(filt_block, fs=C.FS, nperseg=min(1024, len(filt_block)), axis=0)
    m = (f >= band[0]) & (f <= band[1])
    if not m.any():
        return np.zeros(filt_block.shape[1])
    psd = np.median(p[m], axis=0)                    # 中位 → 不被殘餘線狀譜拉高
    bw = C.BANDPASS[1] - C.BANDPASS[0]
    return np.sqrt(np.maximum(psd * bw, 0.0))
