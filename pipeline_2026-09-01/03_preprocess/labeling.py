"""★ 自動標籤：從包絡線推導 gesture + phase，**不需要 trial 時間戳**。

為什麼要重做標籤（v1 的教訓）：
    SoftPINCH 假設每個 trial 固定 9 秒、切三等分成 rest/contract/release。
    我們的錄音沒有時間戳，只能用 autocorrelation 猜週期 + 猜相位錨點，
    結果產生大量假象——曾把某個檔案的 6.1% 當成「模型失效」，
    重新對齊後其實是 92.4%。而且我們的 pinch/grasp 是 21 秒四階段，
    跟 index/thumb 的 9 秒三階段語意根本對不上。

這裡的做法：
    gesture 標籤 ← 檔名（每個錄音只做一個手勢，這個資訊可靠）
    phase   標籤 ← 包絡線本身的形狀（不假設任何固定時長）

        rest    : 包絡線低於 baseline + QUIET_K·MAD
        onset   : 上升緣（活化中，斜率為正且夠大）
        hold    : 高原（活化中，斜率接近 0）      ← SoftPINCH 沒有這一類
        release : 下降緣（斜率為負且夠大）

    好處：
      1. 不需重錄，現有 24 個檔案全部可用
      2. 自動吸收「21 秒四階段 vs 9 秒三階段」的差異
      3. hold 對外骨骼很重要（持續施力輔助），SoftPINCH 完全沒有這個概念

⚠️ 這是訓練目標本身。標籤錯 → 模型學錯 → 一切白做。
   所以 `plot_labels()` 會把結果畫出來，**必須人工抽樣目視確認**再進訓練。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semg.v2 import config as C


@dataclass
class Burst:
    """一次連續的活化事件（在包絡線索引空間）。"""
    start: int
    peak: int
    end: int

    @property
    def dur_sec(self) -> float:
        return (self.end - self.start) * C.SEC_PER_ENV_PT


def gesture_from_filename(stem: str) -> str:
    """檔名 → 手勢。`indexgary_20260716_...` → "index"。"""
    for g in ("index", "thumb", "pinch", "grasp"):
        if stem.startswith(g):
            return g
    return "unknown"


def dominant_channel(gesture: str) -> int | list[int]:
    """該手勢主要活化哪個通道（用來抓 burst）。

    ch0=手腕屈肌, ch1=食指, ch2=拇指（見 CLAUDE.md 的電極配置）
    pinch/grasp 是多指同時，用「食指與拇指的最大值」當偵測依據。
    """
    return {"index": 1, "thumb": 2}.get(gesture, [1, 2])


def detect_bursts(env_ch: np.ndarray) -> tuple[list[Burst], dict]:
    """從單一通道包絡線抓出活化事件。

    回傳 (bursts, 診斷資訊)。門檻用該檔自身的穩健統計量，所以不依賴絕對 mV。
    """
    base = float(np.percentile(env_ch, C.LBL_BASE_PCT))
    peak = float(np.percentile(env_ch, C.LBL_PEAK_PCT))
    rng = max(peak - base, 1e-12)
    hi = base + C.LBL_ACTIVE_FRAC * rng
    lo = base + C.LBL_QUIET_FRAC * rng

    above = env_ch > hi
    min_pts = int(round(C.LBL_MIN_BURST_SEC / C.SEC_PER_ENV_PT))

    bursts: list[Burst] = []
    i = 0
    while i < len(above):
        if not above[i]:
            i += 1
            continue
        j = i
        while j < len(above) and above[j]:
            j += 1
        if j - i >= min_pts:
            # 往兩側擴張到低門檻，才涵蓋完整的上升與下降緣
            s = i
            while s > 0 and env_ch[s - 1] > lo:
                s -= 1
            e = j
            while e < len(env_ch) - 1 and env_ch[e] > lo:
                e += 1
            pk = s + int(np.argmax(env_ch[s:e])) if e > s else i
            if not bursts or s > bursts[-1].end:      # 避免重疊
                bursts.append(Burst(s, pk, e))
        i = j
    return bursts, {"baseline": base, "peak_level": peak, "range": rng,
                    "hi": hi, "lo": lo}


def label_phases(env: np.ndarray, gesture: str) -> tuple[np.ndarray, list[Burst], dict]:
    """回傳 (phase_labels 長度同 env, bursts, 診斷)。

    phase_labels 是 int，對應 C.PHASES 的索引。
    """
    ch = dominant_channel(gesture)
    sig = env[:, ch] if isinstance(ch, int) else env[:, ch].max(axis=1)

    bursts, diag = detect_bursts(sig)
    labels = np.zeros(len(sig), dtype=np.int64)      # 預設 rest(0)

    hold_min = int(round(C.LBL_HOLD_MIN_SEC / C.SEC_PER_ENV_PT))
    for b in bursts:
        seg = sig[b.start:b.end]
        if len(seg) < 3:
            continue
        peak_rel = b.peak - b.start
        amp = float(seg.max() - seg.min()) + 1e-12
        # 斜率門檻：相對於此 burst 的振幅，所以不同強度的動作都適用
        slope_thr = C.LBL_SLOPE_FRAC * amp / (1.0 / C.SEC_PER_ENV_PT)
        d = np.gradient(seg)

        for k in range(len(seg)):
            if k < peak_rel and d[k] > slope_thr:
                labels[b.start + k] = C.PHASES.index("onset")
            elif k > peak_rel and d[k] < -slope_thr:
                labels[b.start + k] = C.PHASES.index("release")
            else:
                labels[b.start + k] = C.PHASES.index("hold")

        # hold 太短就併入鄰近相位（避免產生零碎的 hold）
        hseg = labels[b.start:b.end] == C.PHASES.index("hold")
        i = 0
        while i < len(hseg):
            if hseg[i]:
                j = i
                while j < len(hseg) and hseg[j]:
                    j += 1
                if j - i < hold_min:
                    tgt = (C.PHASES.index("onset") if i < peak_rel
                           else C.PHASES.index("release"))
                    labels[b.start + i: b.start + j] = tgt
                i = j
            else:
                i += 1

    diag.update({"n_bursts": len(bursts),
                 "mean_burst_sec": float(np.mean([b.dur_sec for b in bursts])) if bursts else 0.0})
    return labels, bursts, diag


def channel_activation(env: np.ndarray) -> np.ndarray:
    """每通道各自的 active/rest（0/1），用於 Head A 的多標籤目標。

    這是 Lee et al. 的 per-muscle 概念：同時彎曲時多個通道都會是 1，
    不需要為每種組合開一個類別。門檻用各通道自身統計量。
    """
    out = np.zeros(env.shape, dtype=np.float32)
    for c in range(env.shape[1]):
        s = env[:, c]
        base = np.percentile(s, C.LBL_BASE_PCT)
        rng = max(np.percentile(s, C.LBL_PEAK_PCT) - base, 1e-12)
        out[:, c] = (s > base + C.LBL_ACTIVE_FRAC * rng).astype(np.float32)
    return out


def plot_labels(env: np.ndarray, labels: np.ndarray, bursts: list[Burst],
                gesture: str, title: str, out_png):
    """把標籤畫在包絡線上，供人工目視驗證。**進訓練前一定要看這張圖。**"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"rest": "#DDDDDD", "onset": "#1546A0",
              "hold": "#B23A0E", "release": "#F5A15C"}
    t = np.arange(len(env)) * C.SEC_PER_ENV_PT
    fig, axes = plt.subplots(4, 1, figsize=(19, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 2, 1.2]})
    for c in range(3):
        axes[c].plot(t, env[:, c], color="#222", lw=0.8)
        axes[c].set_ylabel(f"ch{c+1}\n{C.CH_NAMES[c]}", fontsize=9)
        axes[c].margins(x=0)

    ax = axes[3]
    for ph_i, ph in enumerate(C.PHASES):
        m = labels == ph_i
        if not m.any():
            continue
        # 合併連續區段再畫，避免上千個小方塊
        idx = np.where(m)[0]
        splits = np.where(np.diff(idx) > 1)[0]
        for grp in np.split(idx, splits + 1):
            if len(grp):
                ax.axvspan(t[grp[0]], t[grp[-1]], color=colors[ph], lw=0)
    for b in bursts:
        ax.axvline(t[b.start], color="k", lw=0.4, alpha=0.4)
    ax.set_yticks([])
    ax.set_ylabel("phase", fontsize=9)
    ax.set_xlabel("time (s)")
    ax.margins(x=0)

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=colors[p], label=p) for p in C.PHASES],
               loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.955),
               frameon=False)
    fig.suptitle(f"{title}   gesture={gesture}   {len(bursts)} bursts", fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
