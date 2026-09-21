"""錄完立刻跑：把一份錄音的 7 通道畫成圖 + 逐通道品質判定。

## 用途

回答一個問題：**這份錄音要不要採用？** 以及如果不採用，是哪個通道、哪一段壞了。

## ★ 輸出兩張圖，看的是同一份判定、畫的是不同的域

| 檔名 | 畫什麼 | 用途 |
|---|---|---|
| `_qc.png` | RMS 包絡線（40 Hz，視窗均方根） | 快速判讀：起伏一眼看出動作段落 |
| `_qc_wave.png` | ★ 濾波後**原始波形**（2000 Hz，未取包絡） | **給別人看** —— 這才是特徵計算真正吃的域 |

⚠️ **兩張圖都不是模型的逐位元輸入。** 模型吃的是每個 0.5s 視窗抽出的
120 維特徵（`features.PER_CH_NAMES`），RMS 只是其中 1 種統計量（14/120 維）。
但兩者不一樣：包絡線是**視窗聚合後**的量，波形是**特徵計算前**的量 ——
過零率、頻譜形狀、突波的真實樣子，只有在波形圖上看得出來，
包絡線會把這些全部平滑掉。

**跟教授 / 別人展示訊號能不能用，請用 `_qc_wave.png`。**
包絡線圖留給你自己快速篩檔（起伏比較好用肉眼掃）。
兩張圖的通道判定（OK/⚠/✗）完全共用同一套指標，不會有兩套結論打架。

與其他工具的分工：

| 工具 | 需要模型 | 回答 |
|---|---|---|
| **本檔** | ✗ | 訊號本身有沒有問題（電極、導線、電源干擾） |
| `audit_trials` | ✗（自帶 RF） | 受試者**動作**有沒有做錯 |
| `predict_and_plot` | ✓ | 模型在這份錄音上表現如何 |

所以順序是：錄完 → **本檔**（訊號能不能用）→ `audit_trials`（動作對不對）
→ 進 pipeline。

## ⚠️ 這裡刻意**關掉 Hampel**

Hampel 濾波器的工作就是把突波換成鄰域中位數。診斷電極時那正好是
**要看的東西** —— 開著 Hampel 去找突波，等於戴著手套摸東西。
（順帶一提，訓練/部署開不開 Hampel 對準確率沒有可測差異，
　p=0.926，見 `results/no_hampel/`。）

## 為什麼 60 Hz 佔比是這裡最有用的指標

在既有 7 份錄音上，**notch 前**的 60 Hz 功率佔比與該份錄音的 LORO 準確率
相關 **r = −0.942 (p = 0.001)**（2026-08-23 量測）：

    17-23-12   4.4%  → 98.1%        16-26-55  27.6%  → 87.8%
    17-11-49   4.7%  → 97.4%        14-56-09  42.5%  → 67.9%

⚠️ 但**因果不是「60 Hz 本身害的」** —— pipeline 的 notch 把它壓到 1.3~1.8%，
   殘餘量與準確率就**不相關**了（r = −0.166, p = 0.72）。
   它之所以有預測力，是因為它是**電極接觸阻抗**的代理指標：
   接觸差 → 市電耦合進來更多、同時肌電訊號更弱。
   真正跟著一起進到訓練資料裡的是諧波（120/180/…Hz，notch **不會**濾掉，
   r = −0.849）與變差的訊噪比。

   → 所以看到 f60 高，要做的是**重貼電極**（清潔、去角質、等導電膠穩定），
     不是加更多濾波。

⚠️ n=7，這是相關不是因果，而且 60 Hz 高低本身也跟場地有關。
   當成「這份要不要重錄」的紅旗來用剛好，不要拿去當論文結論。

## 順帶一提：ch1 的心電汙染是常態，不是故障

背闊肌貼在軀幹上，ch1 包絡線幾乎一定看得到 ~1.4 Hz（80~100 bpm）的規律起伏。
實測自相關峰值：22-03-40 0.89、22-17-38 0.82、17-23-12 0.63、neil 21-54-56 0.78
—— 連**準確率最高**的 17-23-12 都有。所以看到它不用緊張，
它與準確率沒有可辨識的關係。

## ⚠️ 「飽和」要在濾波後的域量，不是原始域

本專案踩過這個坑：拿 `|raw| > 50 mV` 當飽和指標，量到的其實是**直流偏壓**
（電極半電池電位，正常可以到幾十 mV），結果把一條好通道判成「整份全壞」。
真正的飽和是 **ADC 撞到滿刻度**（±2^23 counts），所以 `clip%` 直接在
counts 上量；肌電振幅則一律在 20–450 Hz 濾波後的域量。

用法：
    PYTHONPATH=src python -m semg.v2.plot_signal 某份錄音.csv
    PYTHONPATH=src python -m semg.v2.plot_signal 資料夾/            # 整批
    PYTHONPATH=src python -m semg.v2.plot_signal *.csv --out results/qc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

from semg.v2 import config as C
from semg.v2 import cues as Q
from semg.v2 import preprocess as P

FULL_SCALE = 2 ** 23            # ADS1299 24-bit 有號滿刻度
YMAX_DEFAULT = 0.2              # mV，與 predict_and_plot 一致

# ★ 濾波後波形（未取包絡）的縱軸上限。
#   波形的瞬時峰值比 RMS 包絡高很多（RMS 是視窗內的均方根，會把峰壓掉）——
#   實測乾淨通道 p99 約 0.02~0.16 mV，瞬時峰值可以到那的好幾倍。
#   0.5 mV 抓的是「典型出力的峰值」，不是「絕對不會被裁」；
#   真正的動作假影（例如拉到導線）本來就該裁出來讓你看見超標比例，
#   而不是自動放大到看不出異常。
YMAX_WAVE_DEFAULT = 0.5         # mV

# ── 中文字型 ────────────────────────────────────────────────────────
# 圖上的中文要顯示得出來，matplotlib 才畫得對。Windows 一定有微軟正黑體，
# Linux 常常一套 CJK 字型都沒有（會畫成一格格的豆腐方塊，而且**不會報錯**）。
# 所以偵測不到就整張圖退回英文標籤 —— 寧可英文，也不要看起來像壞掉。
_CJK_CANDIDATES = ("Microsoft JhengHei", "Microsoft YaHei", "PingFang TC",
                   "Heiti TC", "Noto Sans CJK TC", "Noto Sans CJK SC",
                   "Source Han Sans TW", "WenQuanYi Zen Hei", "SimHei")


def _setup_font() -> bool:
    """設定中文字型，回傳「找不找得到」。"""
    from matplotlib import font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in have:
            plt.rcParams["font.sans-serif"] = [name] + list(
                plt.rcParams["font.sans-serif"])
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


HAS_CJK = _setup_font()

CH_LABEL_ZH = {
    "latissimus_dorsi": "ch1 背闊肌", "biceps": "ch2 二頭肌",
    "triceps": "ch3 三頭肌", "deltoid_anterior": "ch4 三角肌前束",
    "wrist_flexor": "ch5 曲腕肌", "index": "ch6 食指", "thumb": "ch7 拇指",
}
CH_LABEL_EN = {
    "latissimus_dorsi": "ch1 lat dorsi", "biceps": "ch2 biceps",
    "triceps": "ch3 triceps", "deltoid_anterior": "ch4 deltoid ant",
    "wrist_flexor": "ch5 wrist flex", "index": "ch6 index", "thumb": "ch7 thumb",
}
CH_LABEL = CH_LABEL_ZH if HAS_CJK else CH_LABEL_EN
GESTURE_COLOR = {
    "index_flex": "#4C9F70", "thumb_flex": "#E4A13B", "pinch": "#7B68A6",
    "elbow_flex": "#4472C4", "shoulder_flex": "#5BA3C7", "arm_lift": "#2E5E8A",
    "pick_and_lift": "#8C6BB1", "pick_and_raise": "#6A3D9A",
}
OK, WARN, BAD = "OK", "⚠", "✗"
BAD_COLOR = "#C62828"
WARN_COLOR = "#E08A00"

# ── 判定門檻 ────────────────────────────────────────────────────────
# 這些值來自本專案實際看過的壞通道，不是文獻值。每一條都註明出處，
# 因為「門檻是怎麼來的」比門檻本身重要 —— 換硬體就要重新校。
TH = {
    "clip_bad":   0.010,   # ADC 撞滿刻度 1%。16-55-23 的 ch4 是 99.84%
    "clip_warn":  0.001,
    "flat_bad":   0.010,   # 連續不變 = 線斷或 ADC 卡住
    "flat_warn":  0.001,
    "f60_bad":    0.200,   # 16-26-55 的 ch7 有 42% 功率在 60Hz → 判定不可用
    "f60_warn":   0.100,
    "dyn_bad":    1.30,    # p95/baseline。整份沒有調變 = 沒接到肌肉
    "dyn_warn":   1.80,
    "noise_warn": 0.030,   # mV，基線 RMS。好的通道在 0.005~0.02
    "noise_bad":  0.080,
    "art_warn":   0.05,    # 突波段落佔全長比例
    "art_bad":    0.20,
}
ART_ABS_MV = 0.30           # 生理 RMS 包絡幾乎不會超過 0.2 mV
ART_REL = 8.0               # 或超過自身基線的 8 倍
ART_MIN_SEC = 0.10
ART_GAP_SEC = 0.50          # 間隔小於此就併成同一段


def _runs_of_equal(a: np.ndarray, min_len: int) -> float:
    """回傳「連續數值完全不變且長度 ≥ min_len」的樣本佔比。

    線脫落或 ADC 卡住時輸出會是完全相同的 counts。用 diff==0 找，
    但**必須要求連續長度** —— 單點相同在安靜時很常見（LSB 解析度限制），
    直接算 diff==0 的比例會把好通道也算進去。
    """
    d = np.diff(a) == 0
    if not d.any():
        return 0.0
    idx = np.flatnonzero(np.diff(np.concatenate(([0], d.view(np.int8), [0]))))
    starts, ends = idx[0::2], idx[1::2]
    lens = ends - starts
    keep = lens >= min_len
    return float(lens[keep].sum()) / len(a)


def _f60_fraction(x_mv: np.ndarray) -> float:
    """60 Hz（±2 Hz）功率佔 20–450 Hz 總功率的比例。

    ⚠️ 必須在**沒有 notch** 的訊號上量 —— notch 就是為了移除它的，
    量濾完的訊號只會量到 notch 的殘餘，看不出電極接觸有多差。
    所以這裡只過 20–450 帶通。
    """
    sos = butter(4, [C.BANDPASS[0] / (C.FS / 2), C.BANDPASS[1] / (C.FS / 2)],
                 btype="band", output="sos")
    y = sosfiltfilt(sos, x_mv)
    f, pxx = welch(y, fs=C.FS, nperseg=4096)
    band = (f >= C.BANDPASS[0]) & (f <= C.BANDPASS[1])
    line = (f >= 58.0) & (f <= 62.0)
    tot = float(pxx[band].sum())
    return float(pxx[line].sum()) / tot if tot > 0 else 0.0


def _artifact_spans(env: np.ndarray, base: float) -> list[tuple[float, float]]:
    """在包絡線上找突波段落，回傳 [(t0, t1)] 秒。"""
    thr = max(ART_ABS_MV, base * ART_REL)
    hot = env > thr
    if not hot.any():
        return []
    idx = np.flatnonzero(np.diff(np.concatenate(([0], hot.view(np.int8), [0]))))
    spans = [(s / C.ENV_FS, e / C.ENV_FS) for s, e in zip(idx[0::2], idx[1::2])]
    merged: list[list[float]] = []
    for s, e in spans:
        if merged and s - merged[-1][1] <= ART_GAP_SEC:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if e - s >= ART_MIN_SEC]


def _minmax_decimate(t: np.ndarray, y: np.ndarray, n_target: int = 4000):
    """把波形壓到約 n_target 點，但**保留每個區間的最大最小值**。

    直接 `y[::k]` 抽樣會把波峰抽掉，畫出來的振幅比實際小很多 ——
    對 EMG 這種零均值高頻訊號尤其嚴重（看起來會像沒訊號）。
    min/max 抽樣讓波形外觀與全解析度一致（做法與 `predict_and_plot.py` 相同）。
    """
    n = len(y)
    if n <= n_target * 2:
        return t, y
    k = n // n_target
    m = (n // k) * k
    tr = t[:m].reshape(-1, k)
    yr = y[:m].reshape(-1, k)
    lo, hi = yr.min(axis=1), yr.max(axis=1)
    tt = np.repeat(tr[:, 0], 2)
    yy = np.empty(len(lo) * 2)
    yy[0::2], yy[1::2] = lo, hi
    return tt, yy


def channel_metrics(counts: np.ndarray, filt: np.ndarray, raw_mv: np.ndarray,
                    dur_s: float) -> dict:
    """單一通道的品質指標。counts = 原始 ADC 值（未去 DC）。"""
    env = P.rms_envelope(filt[:, None])[:, 0]
    base = float(np.percentile(env, 10)) if len(env) else 0.0
    p95 = float(np.percentile(env, 95)) if len(env) else 0.0
    spans = _artifact_spans(env, base)
    art = sum(e - s for s, e in spans) / dur_s if dur_s > 0 else 0.0
    return {
        "clip": float((np.abs(counts) >= FULL_SCALE * 0.98).mean()),
        "flat": _runs_of_equal(counts, min_len=int(0.025 * C.FS)),
        "dc_mv": float(np.median(counts)) * C.LSB_MV,
        "noise_mv": base,
        "p95_mv": p95,
        "dyn": p95 / base if base > 1e-9 else 0.0,
        "f60": _f60_fraction(raw_mv),
        "art": art,
        "spans": spans,
        "env": env,
    }


def verdict(m: dict) -> tuple[str, list[str]]:
    """把指標轉成 OK / ⚠ / ✗ 與理由。"""
    bad, warn = [], []
    def chk(key, val, fmt):
        if val >= TH[f"{key}_bad"]:
            bad.append(f"{key} {fmt.format(val)}")
        elif val >= TH[f"{key}_warn"]:
            warn.append(f"{key} {fmt.format(val)}")
    chk("clip", m["clip"], "{:.1%}")
    chk("flat", m["flat"], "{:.1%}")
    chk("f60", m["f60"], "{:.0%}")
    chk("noise", m["noise_mv"], "{:.3f}mV")
    chk("art", m["art"], "{:.1%}")
    # dyn 方向相反：越小越糟
    if m["dyn"] <= TH["dyn_bad"]:
        bad.append(f"dyn {m['dyn']:.2f}")
    elif m["dyn"] <= TH["dyn_warn"]:
        warn.append(f"dyn {m['dyn']:.2f}")
    if bad:
        return BAD, bad + warn
    if warn:
        return WARN, warn
    return OK, []


def _cue_spans(cue_csv: Path) -> list[tuple[float, float, str]]:
    """回傳 [(t0, t1, gesture)] —— 只取 go 段（維持姿勢那 4 秒）。"""
    if not cue_csv.exists():
        return []
    df = Q.load_cues(cue_csv)
    out = []
    ev = df.reset_index(drop=True)
    for i, r in ev.iterrows():
        if r.event != "go":
            continue
        nxt = ev.iloc[i + 1:]
        nxt = nxt[nxt.event == "return"]
        if not len(nxt):
            continue
        out.append((r.sample_index / C.FS, nxt.iloc[0].sample_index / C.FS,
                    str(r.gesture)))
    return out


def plot(stem: str, mets: list[dict], vers: list[tuple[str, list[str]]],
         cue_spans, out_png: Path, ymax: float, dur_s: float) -> None:
    n = len(mets)
    fig, axes = plt.subplots(n + 1, 1, figsize=(19, 12), sharex=True,
                             gridspec_kw={"height_ratios": [2] * n + [1.0]})
    for i, (m, (v, reasons)) in enumerate(zip(mets, vers)):
        ax = axes[i]
        env = m["env"]
        t = np.arange(len(env)) / C.ENV_FS
        # 突波段落先鋪紅底，訊號畫在上面
        for s, e in m["spans"]:
            ax.axvspan(s, e, color="#FFCDD2", lw=0, zorder=0)
        ax.plot(t, env, color="#222222", lw=0.4, zorder=2)

        peak = float(np.abs(env).max()) if len(env) else 0.0
        lim = min(ymax, peak * 1.05) if peak > 0 else ymax
        ax.set_ylim(0, lim)
        over = float((env > lim).mean()) if len(env) else 0.0

        col = {OK: "black", WARN: WARN_COLOR, BAD: BAD_COLOR}[v]
        label = f"{CH_LABEL.get(C.CH_NAMES[i], C.CH_NAMES[i])}  {v}"
        if over > 0.001:
            label += f"\n⚠{over:.1%}>{lim:.2g}mV"
        if reasons:
            label += "\n" + " ".join(reasons[:3])
        ax.set_ylabel(label, fontsize=7.5, rotation=0, ha="right", va="center",
                      color=col)
        ax.margins(x=0)
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if v != OK:
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(col)

    axc = axes[-1]
    for s, e, g in cue_spans:
        axc.axvspan(s, e, color=GESTURE_COLOR.get(g, "#999"), lw=0)
    axc.set_ylim(0, 1)
    axc.set_yticks([])
    axc.set_ylabel("cue (go)", fontsize=8, rotation=0, ha="right", va="center")
    axc.margins(x=0)
    for sp in ("top", "right", "left"):
        axc.spines[sp].set_visible(False)
    axc.set_xlabel(
        (f"time (s)   —   縱軸為 RMS 包絡線 (mV)，上限 {ymax:g} mV、"
         "以下自適應；紅底 = 突波段落") if HAS_CJK else
        (f"time (s)   —   y = RMS envelope (mV), capped at {ymax:g} mV, "
         "adaptive below; red band = artifact span"), fontsize=9)

    worst = BAD if any(v == BAD for v, _ in vers) else (
        WARN if any(v == WARN for v, _ in vers) else OK)
    tone = {OK: "#2E7D32", WARN: WARN_COLOR, BAD: BAD_COLOR}[worst]
    fig.suptitle(f"{stem}   ({dur_s:.0f}s)   "
                 + (f"整體：{worst}" if HAS_CJK else f"overall: {worst}"),
                 fontsize=12, y=0.985, color=tone)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_waveform(stem: str, filt: np.ndarray, vers: list[tuple[str, list[str]]],
                  cue_spans, out_png: Path, ymax: float, dur_s: float) -> None:
    """★ 濾波後的**原始波形**（不取包絡）—— 模型實際抽特徵的那個訊號。

    與 `plot()`（RMS 包絡線）的差別**只在於畫什麼**，通道判定（OK/⚠/✗）
    完全共用同一份 `channel_metrics`／`verdict`，不會有兩套結論。

    ⚠️ **這張圖仍然不是模型的原始輸入** —— 模型吃的是每個 0.5s 視窗抽出的
    120 維特徵（`features.PER_CH_NAMES`），不是波形本身。但它是**特徵計算
    的起點**：120 維裡的每一維都是這條波形的某種統計量。
    RMS 包絡線只反映其中 1 種統計量（`rms_rel`／`log_rms`），
    看不出過零率、頻譜、突波形狀 —— 這些都要看原始波形才判斷得出來。
    """
    fs = C.FS
    n = filt.shape[0]
    n_ch = filt.shape[1]
    fig, axes = plt.subplots(n_ch + 1, 1, figsize=(19, 12), sharex=True,
                             gridspec_kw={"height_ratios": [2] * n_ch + [1.0]})
    t_full = np.arange(n) / fs
    for i in range(n_ch):
        ax = axes[i]
        v, reasons = vers[i]
        td, yd = _minmax_decimate(t_full, filt[:, i])
        ax.plot(td, yd, color="#222222", lw=0.3, zorder=2)

        peak = float(np.abs(filt[:, i]).max()) if n else 0.0
        lim = min(ymax, peak * 1.05) if peak > 0 else ymax
        ax.set_ylim(-lim, lim)
        over = float((np.abs(filt[:, i]) > lim).mean()) if n else 0.0

        col = {OK: "black", WARN: WARN_COLOR, BAD: BAD_COLOR}[v]
        label = f"{CH_LABEL.get(C.CH_NAMES[i], C.CH_NAMES[i])}  {v}"
        if over > 0.001:
            label += f"\n⚠{over:.1%}>{lim:.2g}mV"
        if reasons:
            label += "\n" + " ".join(reasons[:3])
        ax.set_ylabel(label, fontsize=7.5, rotation=0, ha="right", va="center",
                      color=col)
        ax.margins(x=0)
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if v != OK:
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(col)

    axc = axes[-1]
    for s, e, g in cue_spans:
        axc.axvspan(s, e, color=GESTURE_COLOR.get(g, "#999"), lw=0)
    axc.set_ylim(0, 1)
    axc.set_yticks([])
    axc.set_ylabel("cue (go)", fontsize=8, rotation=0, ha="right", va="center")
    axc.margins(x=0)
    for sp in ("top", "right", "left"):
        axc.spines[sp].set_visible(False)
    axc.set_xlabel(
        (f"time (s)   —   縱軸為濾波後波形 (mV)（20-450Hz 帶通 + 60Hz 陷波，"
         f"**未取包絡、未套 Hampel**），上限 {ymax:g} mV、以下自適應") if HAS_CJK else
        (f"time (s)   —   y = filtered waveform (mV) after 20-450Hz bandpass + "
         f"60Hz notch (no envelope, no Hampel), capped at {ymax:g} mV, "
         "adaptive below"), fontsize=9)

    worst = BAD if any(v == BAD for v, _ in vers) else (
        WARN if any(v == WARN for v, _ in vers) else OK)
    tone = {OK: "#2E7D32", WARN: WARN_COLOR, BAD: BAD_COLOR}[worst]
    fig.suptitle(f"{stem}   ({dur_s:.0f}s)   waveform   "
                 + (f"整體：{worst}" if HAS_CJK else f"overall: {worst}"),
                 fontsize=12, y=0.985, color=tone)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def check_one(csv: Path, out_dir: Path, ymax: float,
              write_badspans: bool, ymax_wave: float = YMAX_WAVE_DEFAULT,
              waveform: bool = True) -> dict:
    counts_df = pd.read_csv(csv, usecols=[f"ch{c}" for c in C.CH_CSV_COLS])
    counts = counts_df.values.astype(np.float64)
    raw_mv = counts * C.LSB_MV
    raw_mv = raw_mv - raw_mv.mean(axis=0)
    # ⚠️ Hampel 關掉 —— 見檔頭。要看的就是它會抹掉的東西。
    filt = P.filter_causal(raw_mv, use_hampel=False)
    dur_s = len(counts) / C.FS

    mets, vers = [], []
    for i in range(counts.shape[1]):
        m = channel_metrics(counts[:, i], filt[:, i], raw_mv[:, i], dur_s)
        mets.append(m)
        vers.append(verdict(m))

    cue_spans = _cue_spans(csv.with_name(csv.stem + "_cues.csv"))
    out_png = out_dir / f"{csv.stem}_qc.png"          # 沿用原檔名，不動既有慣例
    plot(csv.stem, mets, vers, cue_spans, out_png, ymax, dur_s)

    out_png_wave = None
    if waveform:
        # ★ 這一張才是「跟丟進模型的一樣」的版本 —— 見檔頭。
        out_png_wave = out_dir / f"{csv.stem}_qc_wave.png"
        plot_waveform(csv.stem, filt, vers, cue_spans, out_png_wave, ymax_wave, dur_s)

    print(f"\n{'='*88}\n{csv.stem}   {dur_s:.1f}s")
    print(f"  包絡線圖（快速判讀，適合初篩）→ {out_png}")
    if out_png_wave is not None:
        print(f"  波形圖（給教授看／跟模型輸入同域）→ {out_png_wave}")
    print(f"{'ch':<16}{'判定':<5}{'clip':>7}{'flat':>7}{'DC mV':>9}"
          f"{'基線mV':>9}{'p95mV':>8}{'dyn':>7}{'60Hz':>7}{'突波':>7}  理由")
    for i, (m, (v, reasons)) in enumerate(zip(mets, vers)):
        print(f"{CH_LABEL.get(C.CH_NAMES[i], C.CH_NAMES[i]):<16}{v:<5}"
              f"{m['clip']:>7.2%}{m['flat']:>7.2%}{m['dc_mv']:>9.1f}"
              f"{m['noise_mv']:>9.4f}{m['p95_mv']:>8.4f}{m['dyn']:>7.2f}"
              f"{m['f60']:>7.1%}{m['art']:>7.1%}  {' '.join(reasons)}")

    worst = BAD if any(v == BAD for v, _ in vers) else (
        WARN if any(v == WARN for v, _ in vers) else OK)
    n_bad = sum(v == BAD for v, _ in vers)
    if worst == OK:
        print("  → 採用。七通道皆無異常。")
    elif worst == WARN:
        print("  → 可採用，但上面標 ⚠ 的通道值得看圖確認；"
              "若突波集中在少數幾段，寫進 _badspans.txt 遮蔽即可。")
    else:
        print(f"  → ⚠️ 有 {n_bad} 條通道判定為 ✗。先看圖：整條壞 → 這份不要用；"
              "只壞幾段 → 寫 _badspans.txt。")

    if write_badspans:
        lines = []
        for i, m in enumerate(mets):
            for s, e in m["spans"]:
                lines.append(f"{s:.2f} {e:.2f} ch{i+1}")
        if lines:
            bs = csv.with_name(csv.stem + "_badspans_suggested.txt")
            bs.write_text(
                "# 由 plot_signal.py 自動偵測的突波段落 —— **建議，不是結論**。\n"
                "# 確認過再改名成 _badspans.txt。整條通道都被列出來代表\n"
                "# 該通道應該整份排除，遮蔽沒有意義。\n" + "\n".join(lines) + "\n",
                encoding="utf-8")
            print(f"  → 建議遮蔽段落寫入 {bs.name}（{len(lines)} 段，需人工確認）")

    return {"stem": csv.stem, "verdict": worst, "n_bad": n_bad,
            "png": out_png, "png_wave": out_png_wave}


def main():
    ap = argparse.ArgumentParser(
        description="錄完立刻跑：7 通道品質圖 + 逐通道判定。"
                    "預設輸出兩張圖：_qc.png（RMS 包絡，快速判讀用）"
                    "與 _qc_wave.png（濾波後波形，跟模型輸入同域，給教授看用）。")
    ap.add_argument("paths", nargs="+", type=Path,
                    help="CSV 檔或含 CSV 的資料夾")
    ap.add_argument("--out", type=Path, default=Path("results/signal_qc"))
    ap.add_argument("--ymax", type=float, default=YMAX_DEFAULT,
                    help=f"包絡線圖縱軸上限 mV（預設 {YMAX_DEFAULT}）")
    ap.add_argument("--ymax-wave", type=float, default=YMAX_WAVE_DEFAULT,
                    help=f"波形圖縱軸上限 mV（預設 {YMAX_WAVE_DEFAULT}）")
    ap.add_argument("--no-waveform", action="store_true",
                    help="只畫包絡線圖，不畫波形圖（比較快）")
    ap.add_argument("--write-badspans", action="store_true",
                    help="把偵測到的突波段落寫成 _badspans_suggested.txt")
    a = ap.parse_args()

    csvs: list[Path] = []
    for p in a.paths:
        if p.is_dir():
            csvs += sorted(q for q in p.rglob("*.csv")
                           if not q.stem.endswith("_cues"))
        elif p.suffix == ".csv" and not p.stem.endswith("_cues"):
            csvs.append(p)
    if not csvs:
        sys.exit("找不到任何錄音 CSV（_cues.csv 會自動略過）")

    rows = [check_one(c, a.out, a.ymax, a.write_badspans,
                      ymax_wave=a.ymax_wave, waveform=not a.no_waveform)
            for c in csvs]
    if len(rows) > 1:
        print(f"\n{'='*88}\n總表")
        for r in rows:
            note = f"{r['n_bad']} 條通道 ✗" if r["n_bad"] else ""
            print(f"  {r['verdict']:<3} {r['stem']:<40}{note}")


if __name__ == "__main__":
    main()
