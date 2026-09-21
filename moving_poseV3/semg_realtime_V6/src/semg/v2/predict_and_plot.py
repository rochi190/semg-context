"""把一份錄音逐視窗丟進已訓練的多輸出模型，並把預測直接畫在 RMS 包絡線上。

風格沿用 v1 的 `model_eval/predict_and_plot_epochs.py`：
    通道包絡線用黑線畫，模型**預測**的類別用背景色塊標出，
    對/錯用邊框顏色標示（ground truth 來自 cue 檔，不是模型自己猜的）。

與 v1 版本的差別，全部來自「多輸出」這件事：

    v1 是單一互斥類別 → 一條預測帶就夠。
    這裡有 3 個自由度（hand 4 態 / elbow 2 態 / shoulder 2 態），
    **各自**有預測與正確與否，所以畫成 3 條獨立的預測帶。
    只用「全對率」一個數字會看不出是哪個關節在錯。

★ 推論走的是 `infer.Recognizer`（串流因果濾波 + 逐 DOF 投票），
  不是離線 `filtfilt` —— 也就是說這張圖呈現的是**實際部署時**會看到的行為，
  包含投票造成的延遲。用離線濾波會畫出一張比真實情況樂觀的圖。

⚠️ 跨協定版本使用時（例如 V2 訓練的模型跑 V3 錄音）：
   模型吃的是 0.5s 視窗 × 10 的序列，**不依賴 trial 長度**，所以協定改了仍能跑。
   但 cue 的段落長度變了，`rest` 段的乾淨程度會不同，比較數字時要記得。

用法：
    PYTHONPATH=src python -m semg.v2.predict_and_plot \\
        --bundle results/dof_model.pt --csv data/raw/multi/gary/xxx.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from semg.v2 import config as C
from semg.v2 import cues as Q
from semg.v2 import preprocess as P
from semg.v2.infer import Recognizer

# 每個 DOF 的狀態配色。刻意讓 rest 是淺灰 —— rest 是最大的類別，
# 用飽和色會讓整張圖被它淹沒，看不出出力段。
# ⚠️ 不要把任何狀態設成紅色系 —— 紅色保留給「預測錯誤」的標記，
#    第一版把 pinch 設成 #C0504D，結果與錯誤標記撞色，圖上根本分不出來。
STATE_COLORS = {
    "rest": "#e8e8e8", "index": "#4C9F70", "thumb": "#E4A13B", "pinch": "#7B68A6",
    "flex": "#4472C4",
}
ERR_COLOR = "#C62828"
CH_LABEL = {
    "latissimus_dorsi": "ch1 lat dorsi", "biceps": "ch2 biceps",
    "triceps": "ch3 triceps", "deltoid_anterior": "ch4 deltoid ant",
    "wrist_flexor": "ch5 wrist flex", "index": "ch6 index", "thumb": "ch7 thumb",
}


def predict_recording(bundle: Path, csv: Path, votes: int, conf: float,
                      use_badspans: bool = True) -> pd.DataFrame:
    """逐視窗預測，回傳含時間、預測、真值的表。

    校準區段取自 cue 檔的開頭空白段 —— **必須**與 `dataset.py` 訓練時同一段。
    踩過的坑：訓練用整檔 MAD、推論用別段，系統性差 1.27×，全對率 99.6% → 58%。
    """
    r = Recognizer(bundle, votes=votes, conf=conf)
    cue_csv = Q.cue_path_for(csv)
    if cue_csv is None:
        raise SystemExit(f"{csv} 沒有配對的 _cues.csv")
    cues = Q.load_cues(cue_csv)
    raw = P.load_recording(csv, trim=False)

    # ★ badspans 必須與 dataset.build 傳同一份 —— 尺度來源要一致（見 cues 的說明）
    from semg.v2 import dataset as D
    spans = D.load_badspans(csv) if use_badspans else []
    cs, ce = Q.calibration_span(cues, len(raw), badspans=spans)
    # 從 cs 之前多餵 2 秒當濾波器暖機，但只用 [cs, ce) 估尺度 ——
    # 這樣與 dataset.build（對整份離線濾波後才取該段）算在同樣的訊號上。
    warm = max(0, cs - 2 * C.FS)
    r.calibrate(raw[warm:ce], n_tail=ce - cs)

    # ★ 評估的起點與校準區段**無關**。
    #   校準可能被重定位到錄音中段甚至結尾（開頭空白段壞掉時），
    #   若直接從 ce 開始跑，整份錄音只會評估到最後幾秒。
    #   實際踩到：17-11-49 的校準被移到 541.0~544.5s，
    #   而錄音長 550s → 只剩 5.5 秒被評分。
    start = ce if cs == 0 else Q.calibration_span(cues, len(raw))[1]
    if cs != 0:
        r.reset_stream()          # 尺度留著，濾波器/視窗/投票重來

    # ★ 真值也用**這份錄音實際負擔得起**的 rest margin，跟建資料集時一致。
    m_rest, _ = Q.auto_rest_margin(cues, len(raw))
    truth = Q.sample_labels(cues, len(raw), margin_rest_start_sec=m_rest)

    # ★ 壞段落遮蔽必須與 dataset.build 用同一份清單，否則圖上會出現
    #   「訓練時被丟掉、評分時卻拿來算」的段落，數字與模型看到的世界不一致。
    from semg.v2 import dataset as D
    spans = D.load_badspans(csv) if use_badspans else []
    for a_, b_ in spans:
        truth[max(0, int(a_ * C.FS)):min(len(truth), int(b_ * C.FS))] = -1

    # 每個 trial 的 cue 姿勢名，供圖上標註
    pose_at = np.full(len(raw), "", dtype=object)
    for iv in Q.intervals(cues, len(raw), margin_rest_start_sec=m_rest):
        if iv["phase"] == "action":
            pose_at[iv["start"]:iv["end"]] = iv["gesture"]

    rows = []
    block = C.STEP_SAMPLES
    for s0 in range(start, len(raw) - block + 1, block):
        for res in r.push(raw[s0:s0 + block]):
            end = s0 + block
            start = end - C.WIN_SAMPLES
            t = truth[start:end]
            # ★ 一列只要**有任何一個** DOF 有定義就算有效。
            #   V5 的 `return` 段標籤是 (-1,-1,-1,1)：三個姿勢頭未定義、
            #   moving 頭定義為 1。舊的 `.all(axis=1)` 會把整段判成不可評分，
            #   結果 **moving 頭的正類一次都沒被評分過** ——
            #   當時報出的「moving 93.4%」其實只是 steady 那一類的召回率。
            valid = (t >= 0).any(axis=1)
            rec = {"start_s": start / C.FS, "end_s": end / C.FS,
                   "pose_cue": pose_at[end - 1],
                   # 整個視窗要落在同一個有標籤區間內（與 window_label 的 95% 規則一致）
                   "scorable": bool(valid.mean() >= 0.95)}
            for j, dn in enumerate(r.dof_names):
                st = C.DOF_STATES[dn]
                p = res["raw"][j]
                s = res["stable"][j]
                rec[f"pred_{dn}"] = st[p]
                rec[f"stable_{dn}"] = st[s] if s is not None else ""
                rec[f"conf_{dn}"] = res["conf"][j]
                # ⚠️ 逐 DOF 判斷有沒有定義 —— 不能用整列的 `valid`，
                #    否則 moving-only 的視窗會被填上錯誤的姿勢真值。
                vj = t[:, j] >= 0
                w = int(np.bincount(t[vj, j]).argmax()) if vj.mean() >= 0.95 else -1
                rec[f"true_{dn}"] = st[w] if w >= 0 else ""
            rows.append(rec)
    return pd.DataFrame(rows)


def score(df: pd.DataFrame, dofs: list[str], use_stable: bool = True) -> dict:
    """逐 DOF 與全對率。只算 scorable 的視窗（丟棄段沒有真值可比）。"""
    key = "stable_" if use_stable else "pred_"
    # ★ `exact` 只在**所有** DOF 都有真值的視窗上算（語意才清楚）；
    #   逐 DOF 的指標則各自用自己的真值欄位過濾 ——
    #   拿 dofs[0]（hand）當總開關會讓 moving-only 的視窗整批消失。
    full = df.scorable & np.logical_and.reduce([(df[f"true_{d_}"] != "").values
                                                for d_ in dofs])
    d = df[full]
    if not len(d) and not df.scorable.any():
        return {}
    out, allok = {}, np.ones(len(d), bool)
    for dn in dofs:
        # 投票還沒穩定（空字串）時當作預測錯誤 —— 部署時那就是「沒有輸出」
        dj = df[df.scorable & (df["true_" + dn] != "")]
        okj = (dj[key + dn] == dj["true_" + dn]).values
        out[dn] = float(okj.mean()) if len(dj) else float("nan")
        ok = (d[key + dn] == d["true_" + dn]).values if len(d) else np.zeros(0, bool)
        # ★ 平衡準確率（各狀態召回率的平均）—— 沒有它會被 rest 灌水。
        #   shoulder 有 74% 的視窗是 rest，只看原始準確率會看到 99.6%，
        #   但那幾乎只是在說「模型會判 rest」，看不出 flex 抓不抓得到。
        recalls = [float(okj[(dj["true_" + dn] == s).values].mean())
                   for s in C.DOF_STATES[dn] if (dj["true_" + dn] == s).any()]
        out[dn + "_bal"] = float(np.mean(recalls))
        allok &= ok
    out["exact"] = float(allok.mean())
    out["n"] = int(len(d))
    return out


def confusion(df: pd.DataFrame, dn: str) -> pd.DataFrame:
    """逐 DOF 混淆矩陣（列 = cue 真值，欄 = 模型輸出）。"""
    d = df[df.scorable & (df[f"true_{dn}"] != "")]
    st = C.DOF_STATES[dn]
    m = pd.crosstab(pd.Categorical(d[f"true_{dn}"], st),
                    pd.Categorical(d[f"stable_{dn}"].replace("", "(none)"),
                                   list(st) + ["(none)"]),
                    dropna=False)
    return m


def signal_for_plot(filt: np.ndarray, mode: str) -> tuple[np.ndarray, float, str]:
    """回傳 (要畫的訊號, 取樣率, 說明文字)。

    ⚠️ **兩種都不是模型的輸入。** 模型吃的是每個 0.5s 視窗抽出的 120 維特徵
    （見 `features.PER_CH_NAMES`），不是波形也不是包絡線。
    RMS 只佔其中 14 維（每通道 `rms_rel` 與 `log_rms`）。

    mode="env"  : RMS 包絡線 —— 看整份錄音的活動起伏最清楚
    mode="filt" : **濾波後的原始波形**（模型抽特徵的那個訊號）——
                  看得到真正的肌電形態，但 2000Hz 下整份 550 秒有 110 萬點，
                  概觀圖會用 min/max 抽樣壓縮（見 `_minmax_decimate`）。
    """
    if mode == "filt":
        return filt, float(C.FS), "filtered waveform (what features are computed from)"
    return P.rms_envelope(filt), float(C.ENV_FS), "RMS envelope (display only)"


def _minmax_decimate(t: np.ndarray, y: np.ndarray, n_target: int = 4000):
    """把波形壓到約 n_target 點，但**保留每個區間的最大最小值**。

    直接 `y[::k]` 抽樣會把波峰抽掉，畫出來的振幅比實際小很多 ——
    對 EMG 這種零均值高頻訊號尤其嚴重（看起來會像沒訊號）。
    min/max 抽樣讓包絡外觀與全解析度一致。
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


YMAX_DEFAULT = 0.2          # mV。生理 EMG 的 RMS 包絡線幾乎都在這以下


def _set_ch_ylim(ax, y: np.ndarray, name: str, cap: float, bipolar: bool) -> None:
    """設定單一通道的縱軸：**上限固定 `cap`、以下自適應**。

    ★ 為什麼要有上限：一次飽和突波（可以到 534 mV，滿刻度的 95%）會讓
      自適應縮放把整條肌電壓成一條平線 —— 實際踩到的後果是
      15-50-34 的 ch4 明明**整份 550 秒都 100% 飽和**，
      圖上卻看起來只有 380s 之後才壞，因為前面被壓扁成看不見的一條線。

    ★ 為什麼被裁掉時一定要標出來：這張圖的用途是**診斷電極**。
      靜靜地把超出範圍的部分裁掉，等於用相反的方式犯同一個錯 ——
      壞掉的通道會看起來跟正常的一模一樣。所以超出比例直接寫進 y 標籤。
    """
    a = np.abs(y)
    peak = float(a.max()) if len(a) else 0.0
    lim = min(cap, peak * 1.05) if peak > 0 else cap
    ax.set_ylim(-lim, lim) if bipolar else ax.set_ylim(0, lim)
    over = float((a > lim).mean()) if len(a) else 0.0
    label = CH_LABEL.get(name, name)
    if over > 0.001:
        label += f"\n⚠{over:.1%}>{lim:.2g}"
        ax.tick_params(axis="y", colors=ERR_COLOR)
    ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center",
                  color=ERR_COLOR if over > 0.001 else "black")


def _draw_dof_bands(ax, df: pd.DataFrame, dn: str, t0=None, t1=None) -> None:
    """一個 DOF 畫成上下兩條帶：上 = cue 真值，下 = 模型預測。

    ★ 為什麼不用「邊框顏色表示對錯」（第一版的做法）：
      邊框只有幾個 pixel 寬，在整份 550 秒的概觀圖上完全看不見；
      而且會與狀態填色撞色。直接把真值畫成第二條帶，
      「哪裡對不上」用眼睛掃就看得出來，還能看出**錯成什麼**。
    """
    for _, r in df.iterrows():
        if t0 is not None and (r.end_s < t0 or r.start_s > t1):
            continue
        # 上半：真值（沒有真值的丟棄段留白）
        tv = r[f"true_{dn}"]
        if r.scorable and tv != "":
            ax.axvspan(r.start_s, r.end_s, ymin=0.55, ymax=1.0,
                       color=STATE_COLORS.get(tv, "#999"), lw=0)
        # 下半：預測
        st = r[f"stable_{dn}"]
        if st != "":
            ax.axvspan(r.start_s, r.end_s, ymin=0.0, ymax=0.45,
                       color=STATE_COLORS.get(st, "#999"), lw=0)
            # 錯的地方在最底下補一條紅條 —— 縮圖時仍看得見
            if r.scorable and tv != "" and st != tv:
                ax.axvspan(r.start_s, r.end_s, ymin=0.0, ymax=0.12,
                           color=ERR_COLOR, lw=0)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.22, 0.78])
    ax.set_yticklabels(["pred", "cue"], fontsize=6)
    ax.set_ylabel(dn, fontsize=8, rotation=0, ha="right", va="center")
    ax.margins(x=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


def _legend(fig, dofs: list[str]) -> None:
    seen = {s for dn in dofs for s in C.DOF_STATES[dn]}
    handles = [Patch(fc=STATE_COLORS.get(s, "#999"), ec="#888", label=s)
               for s in sorted(seen)]
    handles.append(Patch(fc=ERR_COLOR, label="prediction != cue"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.995))


def plot_overview(csv: Path, df: pd.DataFrame, env: np.ndarray, dofs: list[str],
                  out_png: Path, title: str, fs: float = C.ENV_FS,
                  sig_desc: str = "RMS envelope",
                  ymax: float = YMAX_DEFAULT, bipolar: bool = False) -> None:
    """整份錄音：上半 7 通道訊號，下半每個 DOF 一組 cue/pred 對照帶。"""
    t_env = np.arange(len(env)) / fs
    n_ch = env.shape[1]
    fig, axes = plt.subplots(n_ch + len(dofs), 1, figsize=(19, 13), sharex=True,
                             gridspec_kw={"height_ratios": [2] * n_ch + [1.3] * len(dofs)})
    for i in range(n_ch):
        ax = axes[i]
        td, yd = _minmax_decimate(t_env, env[:, i])
        ax.plot(td, yd, color="#222222", lw=0.4)
        _set_ch_ylim(ax, env[:, i], C.CH_NAMES[i], ymax, bipolar)
        ax.margins(x=0)
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    for k, dn in enumerate(dofs):
        _draw_dof_bands(axes[n_ch + k], df, dn)
    axes[-1].set_xlabel(f"time (s)   —   channels show {sig_desc}"
                        f"   (y capped at {ymax:g} mV)", fontsize=9)
    _legend(fig, dofs)
    fig.suptitle(title, fontsize=11, y=0.955)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_zoom(csv: Path, df: pd.DataFrame, env: np.ndarray, dofs: list[str],
              cues: pd.DataFrame, out_png: Path, n_trials: int = 6,
              fs: float = C.ENV_FS, sig_desc: str = "RMS envelope",
              ymax: float = YMAX_DEFAULT, bipolar: bool = False) -> None:
    """放大前幾個 trial，看得到 onset/offset 的延遲。"""
    go = cues[cues.event == "go"].reset_index(drop=True)
    if not len(go):
        return
    t0 = 0.0
    t1 = (float(go.sample_index[min(n_trials, len(go) - 1)]) / C.FS
          + C.CUE_ACTION_SEC + C.CUE_RETURN_SEC)
    t_env = np.arange(len(env)) / fs
    m = (t_env >= t0) & (t_env <= t1)
    sub = df[(df.end_s >= t0) & (df.start_s <= t1)]

    n_ch = env.shape[1]
    fig, axes = plt.subplots(n_ch + len(dofs), 1, figsize=(17, 12), sharex=True,
                             gridspec_kw={"height_ratios": [2] * n_ch + [1] * len(dofs)})
    for i in range(n_ch):
        ax = axes[i]
        ax.plot(t_env[m], env[m, i], color="#222222",
                lw=0.9 if fs <= C.ENV_FS else 0.25)
        _set_ch_ylim(ax, env[m, i], C.CH_NAMES[i], ymax, bipolar)
        ax.margins(x=0)
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    # cue 邊界畫成垂直線，方便看模型落後多少
    for _, r in cues.iterrows():
        t = float(r.sample_index) / C.FS
        if not (t0 <= t <= t1) or r.event not in ("go", "return"):
            continue
        for ax in axes:
            ax.axvline(t, color="#1565C0" if r.event == "go" else "#8E24AA",
                       lw=0.7, ls="--", alpha=0.7)
    for k, dn in enumerate(dofs):
        ax = axes[n_ch + k]
        _draw_dof_bands(ax, sub, dn, t0, t1)
        ax.set_xlim(t0, t1)
    _legend(fig, dofs)
    for _, r in go.iterrows():
        t = float(r.sample_index) / C.FS
        if t0 <= t <= t1:
            axes[0].text(t, axes[0].get_ylim()[1] * 0.95, str(r.gesture),
                         fontsize=7, rotation=90, va="top", color="#1565C0")
    axes[-1].set_xlabel(f"time (s)   (blue dashed = go, purple dashed = return)"
                        f"   —   channels show {sig_desc}", fontsize=9)
    fig.suptitle(f"{csv.stem} — first {n_trials} trials (zoom)", fontsize=11, y=0.955)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="用已訓練模型預測一份錄音並畫圖")
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/predict_plots"))
    ap.add_argument("--votes", type=int, default=5)
    ap.add_argument("--conf", type=float, default=0.6)
    ap.add_argument("--zoom-trials", type=int, default=6)
    ap.add_argument("--signal", choices=("env", "filt"), default="env",
                    help="通道要畫 RMS 包絡線(env) 還是濾波後波形(filt)。兩者都**不是**模型輸入，模型吃的是 120 維特徵。")
    ap.add_argument("--ymax", type=float, default=YMAX_DEFAULT,
                    help="每個通道的縱軸上限 (mV)；低於上限時自適應。"
                         "設 0 = 完全自適應（不建議，飽和突波會壓扁肌電）")
    ap.add_argument("--no-badspans", action="store_true",
                    help="忽略 _badspans.txt（預設會遮蔽，與建資料集一致）")
    a = ap.parse_args()
    if a.ymax <= 0:
        a.ymax = float("inf")

    print(f"模型 : {a.bundle}")
    print(f"錄音 : {a.csv}  (協定 v{Q.protocol_version(a.csv)})")

    df = predict_recording(a.bundle, a.csv, a.votes, a.conf,
                           use_badspans=not a.no_badspans)
    import torch
    ck = torch.load(a.bundle, map_location="cpu", weights_only=False)
    dofs = list(ck["dof_names"])

    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / f"{a.csv.stem}_predictions.csv", index=False)

    for nm, use_stable in (("逐窗（未投票）", False), ("投票後（實際輸出）", True)):
        s = score(df, dofs, use_stable)
        if not s:
            continue
        per = "  ".join(f"{d}={s[d]:.1%}/{s[d + '_bal']:.1%}" for d in dofs)
        print(f"  {nm:<20s} 全對={s['exact']:.1%}   {per}   (n={s['n']})")
    print("  （逐 DOF 格式 = 原始準確率／平衡準確率；"
          "rest 佔多數，只看原始值會高估）")

    print("\n混淆矩陣（列 = cue 真值，欄 = 模型投票後輸出）")
    for dn in dofs:
        print(f"\n  ── {dn}")
        print("     " + confusion(df, dn).to_string().replace("\n", "\n     "))

    raw = P.load_recording(a.csv, trim=False)
    sig, fs, desc = signal_for_plot(P.filter_causal(raw), a.signal)
    cues = Q.load_cues(Q.cue_path_for(a.csv))
    sfx = "" if a.signal == "env" else "_filt"
    ttl = (f"{a.csv.stem}   model={a.bundle.parent.name}/{a.bundle.name}   "
           f"protocol v{Q.protocol_version(a.csv)}")
    bip = a.signal == "filt"          # 濾波波形是雙極的，包絡線是單邊的
    plot_overview(a.csv, df, sig, dofs, a.out / f"{a.csv.stem}_overview{sfx}.png",
                  ttl, fs, desc, ymax=a.ymax, bipolar=bip)
    plot_zoom(a.csv, df, sig, dofs, cues,
              a.out / f"{a.csv.stem}_zoom{sfx}.png", a.zoom_trials, fs, desc,
              ymax=a.ymax, bipolar=bip)
    print(f"  → {a.out}/{a.csv.stem}_{{overview,zoom}}{sfx}.png"
          f"   （通道畫的是 {desc}）")


if __name__ == "__main__":
    main()
