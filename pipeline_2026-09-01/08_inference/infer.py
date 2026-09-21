"""即時推論：串流前處理 → 多域特徵 → GestureNet。

與 v1（SoftPINCH）的關鍵差異：
    v1 需要一段校準錄音算 mu/sigma，而 z-score 對「校準期休息佔多少比例」敏感
    （實測 ±8.8pp 漂移）。這裡改用 **MAD 穩健尺度**，且多數特徵本身尺度不變，
    所以校準只需要一段**任意**訊號（連純休息都行）來估雜訊尺度。

沿用 v1 已在合成封包與真實錄音上驗證過的因果串流結構：
    lfilter + zi 狀態接續 → 滑動視窗 → 特徵 → 序列緩衝 → 模型 → 多數投票
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from semg.v2 import config as C
from semg.v2 import cues as Q
from semg.v2 import features as F
from semg.v2.preprocess import StreamFilter
from semg.v2.model import build_model, describe


class MultiVoter:
    """★ 每個自由度**各自**投票，不是對整個組合投票。

    對組合投票會有一個很糟的副作用：只要任一關節抖動，整組結果就被否決，
    於是「食指穩定彎曲但肩膀在抖」會連食指的輸出一起丟掉。
    逐 DOF 投票讓每個關節獨立穩定 —— 這正是多輸出架構該有的行為。

    每個 DOF 維持自己的緩衝：多數決 + 該類別的平均信心 ≥ conf 才輸出，
    否則沿用上一次的穩定值（None 表示還沒有穩定值）。
    """

    def __init__(self, dof_n_states, votes: int = 5, conf=0.6, dwell=0):
        """`conf` / `dwell` 可以是單一值，也可以是**逐 DOF** 的 list。

        ★ 為什麼要逐 DOF：實測回報「elbow 不敏感、shoulder 過於敏感」——
          那是**每個自由度的操作點**問題，不是模型問題。
          同一個 0.6 門檻套在三個生理性質完全不同的自由度上本來就沒道理：
          三角肌維持前舉是持續中等出力，二頭肌維持彎肘則容易隨疲勞衰減。
          調門檻不需要重訓，而且可以現場調。
        """
        self.n_states = list(dof_n_states)
        n = len(self.n_states)
        self.votes = votes
        self.conf = list(conf) if isinstance(conf, (list, tuple)) else [float(conf)] * n
        self.dwell = ([int(v) for v in dwell] if isinstance(dwell, (list, tuple))
                      else [int(dwell)] * n)
        self.bufs = [deque(maxlen=votes) for _ in self.n_states]
        self.stable: list[int | None] = [None] * len(self.n_states)
        # ★ dwell（遲滯）：候選新狀態必須**連續出現 dwell 次**才允許換過去。
        #   與投票的差別很重要：
        #     投票  對抗的是**抖動** —— 逐窗預測在兩個狀態間亂跳
        #     遲滯  對抗的是**轉換期** —— 移動過程中會穩定地經過一個錯的狀態
        #   使用者實測：動 index 的移動過程會**穩定地**標出 pinch，
        #   回 rest 的過程也是。那不是抖動，投票擋不住（它本來就很「穩定」）。
        #   接機械手臂時這一層是必要的 —— 誤判會讓手臂在轉換期亂動。
        self._cand: list[int | None] = [None] * len(self.n_states)
        self._cnt = [0] * len(self.n_states)

    def push(self, probs_list) -> list[int | None]:
        for j, p in enumerate(probs_list):
            self.bufs[j].append(np.asarray(p))
            if len(self.bufs[j]) < self.votes:
                continue
            arr = np.stack(self.bufs[j])
            cls = int(np.bincount(arr.argmax(1), minlength=self.n_states[j]).argmax())
            if arr[:, cls].mean() < self.conf[j]:
                continue
            if self.dwell[j] <= 0 or cls == self.stable[j]:
                self.stable[j] = cls
                self._cand[j], self._cnt[j] = None, 0
                continue
            # 想換狀態 —— 先累積連續次數
            if cls == self._cand[j]:
                self._cnt[j] += 1
            else:
                self._cand[j], self._cnt[j] = cls, 1
            if self._cnt[j] >= self.dwell[j]:
                self.stable[j] = cls
                self._cand[j], self._cnt[j] = None, 0
        return list(self.stable)


class Recognizer:
    """完整即時流程。餵 raw 資料塊，取回 (每窗預測, 投票後穩定預測)。"""

    def __init__(self, bundle: Path, votes: int = 5, conf=0.6,
                 device: str | None = None, threads: int = C.CPU_THREADS,
                 dwell=0, bias=0.0):
        # 筆電同時要收樣本與控制，限制 thread 數避免搶 CPU 造成掉樣本
        torch.set_num_threads(threads)
        ck = torch.load(bundle, map_location="cpu", weights_only=False)
        self.dof_names: list[str] = list(ck["dof_names"])
        self.dof_n_states: list[int] = [int(v) for v in ck["dof_n_states"]]
        self.seq_len: int = int(ck["seq_len"])
        self.mean = np.asarray(ck["scaler_mean"], dtype=np.float32)
        self.std = np.asarray(ck["scaler_scale"], dtype=np.float32)
        self.dev = device or "cpu"        # 即時推論固定 CPU（部署在筆電）

        # 優先載 TorchScript 版（CPU 實測 0.52 → 0.28 ms）
        ts = Path(bundle).with_name(Path(bundle).stem + "_scripted.pt")
        self.scripted = ts.exists()
        if self.scripted:
            self.model = torch.jit.load(str(ts), map_location=self.dev).eval()
        else:
            # checkpoint 記錄了 arch，載入時建同一種（舊 checkpoint 沒有就當 tiny）
            self.model = build_model(ck.get("arch", "tiny"), len(self.mean),
                                     tuple(self.dof_n_states)).to(self.dev)
            self.model.load_state_dict(ck["state_dict"])
            self.model.eval()

        # ★ 前處理設定由 **bundle** 決定，不是由 config 決定。
        #   理由：關掉 Hampel 會改變輸入分布，而模型是配著那個分布訓練的。
        #   若這裡讀 C.USE_HAMPEL，別人改了 config 就會拿舊模型配新前處理 ——
        #   不會報錯，只會靜靜地全錯。舊 bundle 沒有這個欄位，
        #   預設 True（它們就是用 Hampel 訓練的）。
        self.use_hampel = bool(ck.get("use_hampel", True))
        self.arch = str(ck.get("arch", "tiny"))
        # ★ 特徵版本必須相符，否則模型會拿到定義不同的 120 維輸入。
        #   這種錯**不會有任何症狀** —— 維度一樣、不報錯，只是準確率莫名其妙差。
        #   舊 bundle 沒有這個欄位（那時只有 v1），所以預設 1。
        self.feat_version = int(ck.get("feat_version", 1))
        if self.feat_version != C.FEAT_VERSION:
            raise SystemExit(
                f"模型的特徵版本是 v{self.feat_version}，"
                f"但目前的程式產生的是 v{C.FEAT_VERSION}。\n"
                f"   兩者的 120 維定義不同（v1→v2 改了 prop_* 的分母），"
                f"混用不會報錯但結果無意義。\n"
                f"   請改用相符版本的模型，或把 config.FEAT_VERSION 設回 "
                f"{self.feat_version} 重跑。")
        self.filt = StreamFilter(use_hampel=self.use_hampel)
        self.raw: deque = deque(maxlen=C.WIN_SAMPLES)     # 濾波後樣本的滑動視窗
        self.seq: deque = deque(maxlen=self.seq_len)      # 特徵序列緩衝
        self.voter = MultiVoter(self.dof_n_states, votes, conf, dwell)
        self.bias = (list(bias) if isinstance(bias, (list, tuple))
                     else [float(bias)] * len(self.dof_n_states))
        self.scales = np.asarray(ck.get("scales", np.ones(C.N_CH)), dtype=np.float64)
        self.epss = 0.05 * self.scales
        self._since_step = 0
        self.calibrated = "scales" in ck

    def reset_stream(self) -> None:
        """清掉串流狀態（濾波器、視窗、序列、投票），**保留已估好的尺度**。

        用在「校準區段不是錄音開頭」的情況：開頭空白段壞掉時
        `cues.calibration_span` 會改用錄音中段或結尾的休息段估尺度，
        但**評估仍然要從錄音前段開始跑**。不重設的話，濾波器狀態會停在
        校準段的結尾，接著從錄音前段餵資料 = 時間軸接不起來。

        ⚠️ 不重設 `self.scales` —— 那正是校準的產物。
        """
        self.filt = StreamFilter(use_hampel=self.use_hampel)   # 保留 bundle 的設定
        self.raw.clear()
        self.seq.clear()
        self.voter = MultiVoter(self.dof_n_states, self.voter.votes,
                                self.voter.conf, self.voter.dwell)
        self._since_step = 0

    def calibrate(self, raw_block: np.ndarray, n_tail: int | None = None) -> np.ndarray:
        """從**開頭空白段**估穩健尺度。

        ⚠️ 必須餵開頭空白段（純休息），因為 `dataset.py` 訓練時也是用同一段。
        我一度以為「MAD 夠穩健，餵任意訊號都行」，實測推翻了：
        整檔 MAD 與休息段 MAD 差 1.27×，混用會讓全對率從 99.6% 掉到 58%。

        重點不是統計量穩不穩健，而是**訓練與推論必須算在同性質的區段上**。
        錄製協定保證前 CUE_LEAD_SEC 秒是純休息，所以離線線上都拿得到這段。

        `n_tail`：只用**最後** n_tail 個樣本估尺度，前面的當濾波器暖機。
        開頭空白段壞掉、校準被迫改用錄音中段的休息段時要用它
        （見 `cues.calibration_span` 的 badspans 分支）——
        `dataset.py` 是對**整份**離線濾波後才取那一段，濾波器已有完整歷史；
        串流這邊若從中途冷啟動，前約 100 ms（Hampel 視窗長度）會有邊界效應。
        餵一段前綴當暖機就能讓兩邊算在同樣的訊號上。
        """
        f = self.filt(raw_block)
        if n_tail is not None:
            f = f[-int(n_tail):]
        self.scales, self.epss = F.file_scales(f)
        self.calibrated = True
        return self.scales

    def push(self, raw_block: np.ndarray):
        """餵入一塊 raw 資料。

        回傳 [dict] —— 每個新視窗一筆：
            raw    : (n_dof,) 逐窗預測（會抖）
            conf   : (n_dof,) 各頭的信心
            stable : (n_dof,) 投票後的穩定輸出（None = 該 DOF 還沒穩定）
            action : 穩定輸出對應的動作名（可讀）
        """
        return [r for feat in self.windows(raw_block)
                if (r := self.classify(feat)) is not None]

    def windows(self, raw_block: np.ndarray) -> list[np.ndarray]:
        """串流濾波 → 滑動視窗 → 120 維特徵。回傳這一塊產生的所有特徵向量。

        ★ 與 `classify` 拆開是為了讓**多個模型吃同一份特徵**（`live.py --bundle-b`）。
          兩個架構若各自跑一次 `push`，濾波與特徵會算兩遍（各佔 77% / 22% 的
          計算量），而且沒有必要 —— 特徵只由 `scales` 決定，與模型無關。
          共用之後 A/B 比較還多一個好處：**兩個模型看到的輸入逐位元相同**，
          差異一定來自模型本身。
        """
        f = self.filt(raw_block)
        out = []
        for i in range(len(f)):
            self.raw.append(f[i])
            self._since_step += 1
            if len(self.raw) < C.WIN_SAMPLES or self._since_step < C.STEP_SAMPLES:
                continue
            self._since_step = 0
            out.append(F.extract(np.stack(self.raw), self.scales, self.epss))
        return out

    def classify(self, feat: np.ndarray):
        """一個特徵向量 → 標準化 → 序列緩衝 → 模型 → 投票。序列還沒滿就回傳 None。

        ⚠️ 標準化用的 `mean`/`std` 是**每個模型自己的**（來自它訓練集的
           StandardScaler），所以這一步不能共用；能共用的只有 `feat` 本身。
        """
        self.seq.append((feat - self.mean) / (self.std + 1e-8))
        if len(self.seq) < self.seq_len:
            return None
        x = torch.from_numpy(np.stack(self.seq)[None].astype(np.float32)).to(self.dev)
        with torch.inference_mode():
            logits = self.model(x)
        # ★ 逐 DOF 的靈敏度偏移：直接加在**非 rest 狀態的 logit** 上。
        #
        #   為什麼不是調信心門檻：實測模型的輸出信心**飽和在 1.00** ——
        #   正確時中位 1.00、錯誤時也是 1.00。門檻設在 0.96 以下完全沒有作用，
        #   設在以上則全部砍掉。過度自信是 cross-entropy 訓練的常態。
        #   偏移 logit 才是有效的旋鈕：+0.5 讓該 DOF 更容易離開 rest，
        #   −0.5 讓它更難。單位是 logit，約略對應機率比的 e^bias 倍。
        if any(self.bias):
            logits = [l.clone() for l in logits]
            for j, b in enumerate(self.bias):
                if b:
                    logits[j][:, 1:] += float(b)      # 索引 0 一律是該 DOF 的中性狀態
        probs = [torch.softmax(l, dim=1).cpu().numpy()[0] for l in logits]
        stable = self.voter.push(probs)
        filled = [s if s is not None else 0 for s in stable]
        return {
            "raw": [int(p.argmax()) for p in probs],
            "conf": [float(p.max()) for p in probs],
            "stable": stable,
            "action": Q.dof_to_action(np.asarray(filled)),
            "text": describe(filled, tuple(self.dof_names)),
        }

    @property
    def latency_budget(self) -> dict:
        """理論延遲拆解（不含模型計算，那個要實測）。"""
        return {
            "window_s": C.WIN_SEC,
            "seq_context_s": (self.seq_len - 1) * C.STEP_SEC,
            "voting_s": (self.voter.votes - 1) * C.STEP_SEC,
            "dwell_s": max(self.voter.dwell) * C.STEP_SEC,
            "total_s": C.WIN_SEC + (self.seq_len - 1) * C.STEP_SEC
                       + (self.voter.votes - 1) * C.STEP_SEC
                       + max(self.voter.dwell) * C.STEP_SEC,
        }


def replay(bundle: Path, csv: Path, votes: int, conf: float, calib_sec: float):
    """用已錄好的 CSV 重播，驗證即時流程與離線訓練一致。

    有 cue 檔時會直接比對逐 DOF 正確率 —— 這是**最接近實際使用**的評估：
    走的是串流因果濾波（不是離線 filtfilt）、有投票、有校準，全部跟真實部署一致。
    """
    from semg.v2 import preprocess as P

    r = Recognizer(bundle, votes=votes, conf=conf)
    cue_csv = Q.cue_path_for(csv)
    raw = P.load_recording(csv, trim=cue_csv is None)
    truth = Q.sample_labels(Q.load_cues(cue_csv), len(raw)) if cue_csv else None

    # ★ 有 cue 檔時**用與訓練完全相同的區段**算校準尺度。
    #   先前踩過的坑：訓練用整檔 MAD、推論用另一段 → 差 1.27× → 全對率 99.6%→58%。
    #   這裡直接呼叫 dataset.py 用的同一個函式，結構上保證一致。
    if cue_csv is not None:
        # badspans 也要傳 —— 開頭空白段若壞掉，尺度會毀掉整份錄音，
        # 而 dataset.py 已經會改用乾淨的休息段；兩邊必須一致。
        from semg.v2 import dataset as D
        cs, ce = Q.calibration_span(Q.load_cues(cue_csv), len(raw),
                                    badspans=D.load_badspans(csv))
        print(f"校準區段取自 cue 檔（與訓練同來源）："
              f"{cs/C.FS:.2f}–{ce/C.FS:.2f}s")
    else:
        cs, ce = 0, int(calib_sec * C.FS)
    # ⚠️ `cs` 不一定是 0 —— 開頭空白段壞掉時會改用錄音中段的休息段。
    #    早期版本寫死 `raw[:n_cal]`，那會在**錯誤的區段**上估尺度而且不報錯。
    if ce > cs:
        warm = max(0, cs - 2 * C.FS)             # 前綴只當濾波器暖機
        r.calibrate(raw[warm:ce], n_tail=ce - cs)
    # ★ 推論起點與校準區段無關 —— 校準被重定位時若從 ce 開始，
    #   整份錄音只會剩最後幾秒被處理（見 predict_and_plot 的同一段說明）。
    if cs != 0:
        r.reset_stream()
        n_cal = Q.calibration_span(Q.load_cues(cue_csv), len(raw))[1]
    else:
        n_cal = ce
        print(f"→ 穩健尺度 (mV):")
        for n, v in zip(C.CH_NAMES, r.scales):
            print(f"    {n:18s} {v:.5f}")

    lat = r.latency_budget
    print(f"\n理論延遲：視窗 {lat['window_s']:.2f}s + 序列 {lat['seq_context_s']:.2f}s "
          f"+ 投票 {lat['voting_s']:.2f}s = {lat['total_s']:.2f}s")
    print(f"模型：{'TorchScript' if r.scripted else 'eager'}，"
          f"{len(r.dof_names)} 個自由度 {r.dof_names}")

    rows, t0 = [], time.perf_counter()
    block = C.STEP_SAMPLES
    for s0 in range(n_cal, len(raw) - block + 1, block):
        for res in r.push(raw[s0:s0 + block]):
            rec = {"end_sample": s0 + block, "action": res["action"]}
            for j, dn in enumerate(r.dof_names):
                rec[f"raw_{dn}"] = res["raw"][j]
                rec[f"stable_{dn}"] = res["stable"][j]
            rows.append(rec)
    dt = time.perf_counter() - t0
    audio_s = (len(raw) - n_cal) / C.FS
    print(f"\n處理 {audio_s:.0f}s 訊號耗時 {dt:.1f}s "
          f"（{audio_s/max(dt,1e-9):.1f}× 即時，>1 表示 CPU 跟得上）")
    print(f"視窗數 {len(rows)}")
    if not rows:
        return rows

    df = pd.DataFrame(rows)
    if truth is not None:
        print("\n逐自由度正確率（與 cue 標籤比對）：")
        for j, dn in enumerate(r.dof_names):
            # 視窗結尾對應的真實標籤；-1（丟棄區）不計分
            t = truth[np.clip(df.end_sample.values - 1, 0, len(truth) - 1), j]
            m = t >= 0
            if not m.any():
                continue
            raw_acc = (df[f"raw_{dn}"].values[m] == t[m]).mean()
            sv = df[f"stable_{dn}"].values[m]
            ok = np.array([v is not None for v in sv])
            st_acc = (np.array([v if v is not None else -1 for v in sv])[ok]
                      == t[m][ok]).mean() if ok.any() else float("nan")
            print(f"    {C.DOF_LABELS_ZH[dn]:4s} 逐窗={raw_acc:6.1%}  "
                  f"投票後={st_acc:6.1%}  (n={m.sum()})")
        # 全對率
        T = np.stack([truth[np.clip(df.end_sample.values - 1, 0, len(truth) - 1), j]
                      for j in range(len(r.dof_names))], axis=1)
        m = (T >= 0).all(axis=1)
        P_ = np.stack([df[f"raw_{dn}"].values for dn in r.dof_names], axis=1)
        print(f"    {'全對':4s} 逐窗={(P_[m] == T[m]).all(axis=1).mean():6.1%}")
    print("\n預測出的動作分布：")
    for a, n in df.action.value_counts().head(12).items():
        print(f"    {a:28s} {n:6d}  ({n/len(df):5.1%})")
    return rows


def main():
    ap = argparse.ArgumentParser(description="v2 多輸出即時推論 / CSV 重播")
    ap.add_argument("--bundle", type=Path, default=C.RESULTS_ROOT / "dof_model.pt")
    ap.add_argument("--csv", type=Path, help="重播用的錄音；不給就只印模型資訊")
    ap.add_argument("--votes", type=int, default=5)
    ap.add_argument("--conf", type=float, default=0.6)
    ap.add_argument("--calib-sec", type=float, default=5.0,
                    help="用開頭幾秒估穩健尺度（開頭空白段剛好，不需含出力）")
    a = ap.parse_args()

    if not a.bundle.exists():
        raise SystemExit(f"找不到模型 {a.bundle}，請先跑 train.py")
    if a.csv:
        replay(a.bundle, a.csv, a.votes, a.conf, a.calib_sec)
    else:
        r = Recognizer(a.bundle, a.votes, a.conf)
        print(f"自由度：")
        for dn, ns in zip(r.dof_names, r.dof_n_states):
            print(f"  {C.DOF_LABELS_ZH[dn]:4s} {ns} 態 {C.DOF_STATES[dn]}")
        print(f"SEQ_LEN={r.seq_len}  特徵維度={len(r.mean)}  "
              f"{'TorchScript' if r.scripted else 'eager'}")
        print(f"理論延遲：{json.dumps(r.latency_budget, indent=2)}")


if __name__ == "__main__":
    main()
