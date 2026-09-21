"""逐 trial 稽核：錄完立刻跑，找出「做錯 / 沒放鬆 / 反應太慢」的 trial。

## 為什麼需要

錄製時受試者無法全程 100% 專注，而且**來不及記下哪一次做錯了**
（下一個 cue 馬上就來）。第一份真實錄音 40 個 trial 裡就有 4 個做錯。

**dropout 解決不了這件事** —— dropout 是防特徵共適應的正則化，不是防標籤雜訊。
每個姿勢只有 5 次重複，錯 1 次 = 該姿勢 20% 的訓練資料是錯的，
那是**系統性**錯誤，模型會確實地學進去。

## 三個獨立準則

1. **姿勢做錯**（leave-one-trial-out 交叉預測）
   對每個 trial，用**其他所有 trial** 訓練 RandomForest（該模型從沒看過它），
   再拿去預測它的 `go` 段視窗。若穩定被判成別的狀態 → 訊號與 cue 標籤不符。

   逐 DOF 做，不是對整體姿勢做 —— 實測逐 DOF 比較可靠：
   整體姿勢比對在第一份錄音上給了 2 個誤報（trial 3、21），
   而逐 DOF 診斷顯示那兩個其實三個部位全部相符。
   逐 DOF 還能直接告訴你**是哪個部位做錯**。

2. **rest 沒回到基線**
   以開頭空白段為絕對基線，檢查該 trial 的 rest 段是否還在高活動。
   實測從 return 起點衰減 90% 需要 p90 2.86s、max 3.65s；
   協定的衰減預算是 `CUE_RETURN_SEC + CUE_MARGIN_REST_START_SEC`（目前 3.5s，覆蓋 98%），
   剩下的尾巴就是靠這一項抓出來。

3. **反應太慢**
   量該 trial 從 prepare 起點到訊號到達高原的時間（t_reach）。
   ⚠️ **不能假設「cue 時間戳當下受試者已經在做該姿勢」** —— 每個人反應時間不同，
   實測 gary 是中位 0.40s / max 1.00s，但換人、換疲勞、換姿勢都會變
   （複合姿勢 arm_lift 0.52s vs index_flex 0.22s）。
   若 t_reach 超過 `CUE_PREPARE_SEC` 的餘裕，`go` 段的開頭就被移動過程汙染了。

## ★ 稽核範圍必須限定在「同一次錄音內」

參考基準是「這個姿勢長什麼樣」，而那**取決於電極位置**：

| 範圍 | 用不用 | 理由 |
|---|---|---|
| 同一次錄音（5 次重複，留一比四） | ✅ 預設 | 電極位置完全相同 |
| 同一人跨 session | ⚠️ 僅供第二意見 | 第二次會重貼電極，訊號會變 |
| 跨受試者混合 | ❌ 禁止 | 會把「這個人訊號不一樣」誤判成「他一直做錯」 |

**副產品**：若某份錄音被標記的比例明顯偏高（例如 40 標 15），
多半不是他一直做錯，而是**電極貼歪了** —— 這個訊號比逐 trial 的標記更有價值。

## 這個工具報的是「共識」不是「真理」

它說的是「其他 trial 認為這個訊號像什麼」。若某個姿勢 5 次裡**錯了 2 次而且錯法一樣**，
共識就會翻轉，工具會反過來標記那 3 個做對的。
所以**預設只產生清單、不自動刪除**，由人確認。

## 用法

    PYTHONPATH=src python -m semg.v2.audit_trials <錄音.csv> [...]

輸出 `<stem>_audit.csv`（完整報告）與 `<stem>_exclude.txt`（建議排除的 rep 編號）。
`dataset.py` 會自動讀取 `_exclude.txt`。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from semg.v2 import config as C
from semg.v2 import cues as Q
from semg.v2 import features as F
from semg.v2 import preprocess as P

# 各姿勢的主導通道 —— 只用於「訊號有沒有起來 / 有沒有降下去」這類**振幅**判斷，
# 不用於分類（分類用完整的 120 維特徵）。
POSE_CHANNELS: dict[str, tuple[str, ...]] = {
    "index_flex":     ("index",),
    "thumb_flex":     ("thumb",),
    "pinch":          ("wrist_flexor", "index", "thumb"),
    "elbow_flex":     ("biceps",),
    "shoulder_flex":  ("deltoid_anterior",),
    "arm_lift":       ("deltoid_anterior", "biceps"),
    "pick_and_lift":  ("biceps", "wrist_flexor", "index", "thumb"),
    "pick_and_raise": ("deltoid_anterior", "biceps", "wrist_flexor", "index", "thumb"),
}
DECAY_FRAC = 0.10       # 「已放鬆」= 衰減掉 90%（相對於該 trial 的維持水準）
REACH_FRAC = 0.90       # 「已到位」= 到達維持水準的 90%
AGREE_BAD = 0.30        # 逐 DOF 一致率低於此 → 高度可疑
AGREE_WARN = 0.60       # 低於此 → 可疑


def _pose_channels(pose: str) -> list[int]:
    names = POSE_CHANNELS.get(pose)
    if not names:                       # 未知姿勢就用全部遠端通道當保底
        names = C.CH_GROUPS["distal"]
    return [C.ch_index(n) for n in names if n in C.CH_NAMES]


def envelope_metrics(env: np.ndarray, cues: pd.DataFrame,
                     m_rest: float | None = None) -> pd.DataFrame:
    """逐 trial 的振幅時序指標：反應時間、rest 殘餘。

    全部以**開頭空白段**為絕對基線 —— 那是整份錄音裡唯一
    「定義明確、離線與線上都拿得到」的靜止參考。
    """
    e2i = lambda s: int(s / C.ENV_STEP)          # noqa: E731
    ev = {k: cues[cues.event == k].reset_index(drop=True)
          for k in ("preview", "prepare", "go", "return", "release")}
    if not len(ev["go"]):
        return pd.DataFrame()

    # 絕對基線：第一個 preview 之前，扣掉開機暖機那 1 秒
    lead_end = e2i(int(ev["preview"].sample_index[0]))
    warm = e2i(int(C.CUE_WARMUP_DISCARD_SEC * C.FS))
    base_all = np.median(env[warm:lead_end], axis=0) if lead_end > warm else \
        np.median(env, axis=0)

    rows = []
    for i in range(len(ev["go"])):
        pose = str(ev["go"].gesture[i])
        ch = _pose_channels(pose)
        base = float(base_all[ch].mean())
        sig = env[:, ch].mean(axis=1)

        a_prep = e2i(int(ev["prepare"].sample_index[i]))
        a_go = e2i(int(ev["go"].sample_index[i]))
        a_ret = e2i(int(ev["return"].sample_index[i]))
        a_rel = e2i(int(ev["release"].sample_index[i]))
        end = (e2i(int(ev["preview"].sample_index[i + 1]))
               if i + 1 < len(ev["preview"]) else len(env))

        hold = float(np.median(sig[a_go:a_ret])) if a_ret > a_go else np.nan
        rise = hold - base

        # ── 反應時間：從 prepare 起點到達維持水準的 90%
        t_reach = np.nan
        if rise > 0:
            thr = base + REACH_FRAC * rise
            hit = np.where(sig[a_prep:a_ret] >= thr)[0]
            if len(hit):
                t_reach = hit[0] / C.ENV_FS

        # ── rest 殘餘：從 return 起點衰減 90% 所需時間
        t_decay = np.nan
        if rise > 0 and end > a_ret:
            thr = base + DECAY_FRAC * rise
            hit = np.where(sig[a_ret:end] <= thr)[0]
            if len(hit):
                t_decay = hit[0] / C.ENV_FS

        # ── rest 標籤區間內的實際殘餘倍率（真正會進訓練的那一段）
        # 用**這份錄音實際採用的** rest margin，才跟 dataset.py 取到的視窗一致
        ms = int((m_rest if m_rest is not None else C.CUE_MARGIN_REST_START_SEC) * C.FS)
        me = int(C.CUE_MARGIN_END_SEC * C.FS)
        r0, r1 = e2i(int(ev["release"].sample_index[i]) + ms), \
            (e2i(int(ev["preview"].sample_index[i + 1]) - me)
             if i + 1 < len(ev["preview"]) else len(env))
        rest_ratio = (float(np.median(sig[r0:r1]) / base)
                      if r1 > r0 and base > 0 else np.nan)

        rows.append({"rep": int(ev["go"].rep[i]), "pose": pose,
                     "t_reach": t_reach, "t_decay": t_decay,
                     "rest_ratio": rest_ratio,
                     "snr": hold / base if base > 0 else np.nan})
    return pd.DataFrame(rows)


def loo_agreement(X: np.ndarray, Y: np.ndarray, rep: np.ndarray,
                  seed: int = C.SEED) -> pd.DataFrame:
    """Leave-one-trial-out：每個 trial 逐 DOF 的一致率。

    ⚠️ 判斷 trial t 的模型**只用其他 trial 訓練**，所以 t 自己的（可能錯的）
       標籤完全沒有參與判斷它自己 —— 這是這個方法能成立的關鍵。
    """
    out = []
    for t in sorted(np.unique(rep)):
        te = rep == t
        tr = ~te
        if te.sum() == 0 or tr.sum() < 10:
            continue
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        row = {"rep": int(t)}
        for j, dn in enumerate(C.DOFS):
            want = int(Y[te, j][0])
            if len(np.unique(Y[tr, j])) < 2:      # 訓練集只有一種狀態 → 判不了
                row[f"agree_{dn}"] = np.nan
                row[f"pred_{dn}"] = ""
                continue
            m = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                       random_state=seed,
                                       class_weight="balanced").fit(Xtr, Y[tr, j])
            p = m.predict(Xte)
            top = int(np.bincount(p, minlength=len(C.DOF_STATES[dn])).argmax())
            row[f"agree_{dn}"] = float((p == want).mean())
            row[f"pred_{dn}"] = C.DOF_STATES[dn][top]
            row[f"want_{dn}"] = C.DOF_STATES[dn][want]
        out.append(row)
    return pd.DataFrame(out)


def audit(csv_path: Path, seed: int = C.SEED) -> tuple[pd.DataFrame, dict]:
    cue_csv = Q.cue_path_for(csv_path)
    if cue_csv is None:
        raise SystemExit(f"{csv_path} 沒有對應的 _cues.csv，無法稽核")
    cues = Q.load_cues(cue_csv)

    raw = P.load_recording(csv_path, trim=False)
    filt = P.filter_causal(raw)
    env = P.rms_envelope(filt)

    # ── 特徵（只取 go 段；rest 段不用來判「姿勢做錯」）
    from semg.v2 import dataset as D
    cs, ce = Q.calibration_span(cues, len(filt), badspans=D.load_badspans(csv_path))
    scales, epss = F.file_scales(filt[cs:ce])
    m_rest, m_want = Q.auto_rest_margin(cues, len(filt))
    if m_rest < m_want:
        print(f"  ⚠️ release 段偏短 → rest 起始 margin {m_want:.1f}s → {m_rest:.1f}s"
              f"（不縮的話 rest 完全沒有訓練序列）")
    slab = Q.sample_labels(cues, len(filt), margin_rest_start_sec=m_rest)
    rep_of = np.full(len(filt), -1, np.int32)
    phase_of = np.full(len(filt), "", dtype=object)
    for iv in Q.intervals(cues, len(filt), margin_rest_start_sec=m_rest):
        rep_of[iv["start"]:iv["end"]] = iv["rep"]
        phase_of[iv["start"]:iv["end"]] = iv["phase"]

    X, Y, R = [], [], []
    for s in range(0, len(filt) - C.WIN_SAMPLES + 1, C.STEP_SAMPLES):
        e = s + C.WIN_SAMPLES
        dof = Q.window_label(slab, s, e)
        if dof is None:
            continue
        mid = phase_of[(s + e) // 2]
        if mid != "action":                       # 只稽核「維持姿勢」段
            continue
        X.append(F.extract(filt[s:e], scales, epss))
        Y.append(dof)
        R.append(int(np.bincount(np.maximum(rep_of[s:e], 0)).argmax()))
    if not X:
        raise SystemExit("沒有任何 action 視窗，檢查 cue 檔")
    X, Y, R = np.stack(X), np.stack(Y), np.asarray(R)

    agree = loo_agreement(X, Y, R, seed)
    metrics = envelope_metrics(env, cues, m_rest)
    df = metrics.merge(agree, on="rep", how="outer").sort_values("rep")

    # ── 判定
    # 衰減預算 = return 段長度 + 實際採用的 rest 起始 margin。
    # ⚠️ 用**這份錄音實際錄的** return 長度，不是 config 的值（舊協定是 1.5s）。
    rows_s = cues.sort_values("sample_index").reset_index(drop=True)
    ret_len = [
        (int(rows_s.loc[i + 1, "sample_index"]) - int(r["sample_index"])) / C.FS
        for i, r in rows_s.iterrows()
        if str(r["event"]) == "return" and i + 1 < len(rows_s)]
    budget = (float(np.median(ret_len)) if ret_len else C.CUE_RETURN_SEC) + m_rest
    reach_budget = C.CUE_PREPARE_SEC + C.CUE_MARGIN_START_SEC
    acols = [f"agree_{d}" for d in C.DOFS if f"agree_{d}" in df.columns]
    df["min_agree"] = df[acols].min(axis=1) if acols else np.nan

    # ★ 三種標記的**成因不同**，要分開統計，否則會誤導：
    #     pose  = 姿勢做錯 → 受試者執行問題，或電極位置有問題
    #     rest  = 沒放鬆   → **協定問題**（return 段太短），跟受試者無關
    #     slow  = 反應太慢 → prepare 段太短
    #   把三者混在一起算「標記率」，會讓協定太趕的錄音看起來像電極壞掉。
    reasons, kinds = [], []
    for _, r in df.iterrows():
        why, kind = [], set()
        for d in C.DOFS:
            a = r.get(f"agree_{d}", np.nan)
            if pd.notna(a) and a < AGREE_BAD:
                why.append(f"{C.DOF_LABELS_ZH[d]}做錯"
                           f"(要{r.get(f'want_{d}','?')}→判{r.get(f'pred_{d}','?')},{a:.0%})")
                kind.add("pose")
        if pd.notna(r.get("t_decay")) and r["t_decay"] > budget:
            why.append(f"rest沒放鬆(需{r['t_decay']:.1f}s>預算{budget:.1f}s)")
            kind.add("rest")
        if pd.notna(r.get("t_reach")) and r["t_reach"] > reach_budget:
            why.append(f"反應太慢({r['t_reach']:.1f}s>{reach_budget:.1f}s)")
            kind.add("slow")
        reasons.append("; ".join(why))
        kinds.append(",".join(sorted(kind)))
    df["reason"] = reasons
    df["kind"] = kinds
    df["exclude"] = df.reason != ""

    n_pose = int(df.kind.str.contains("pose").sum())
    summary = {
        "n_trials": len(df),
        "n_excluded": int(df.exclude.sum()),
        "n_pose": n_pose,
        "n_rest": int(df.kind.str.contains("rest").sum()),
        "n_slow": int(df.kind.str.contains("slow").sum()),
        "pose_rate": float(n_pose / len(df)) if len(df) else 0.0,
        "flag_rate": float(df.exclude.mean()) if len(df) else 0.0,
        "decay_budget": budget,
        "t_reach_median": float(df.t_reach.median(skipna=True)),
        "t_decay_median": float(df.t_decay.median(skipna=True)),
    }
    return df, summary


def main():
    ap = argparse.ArgumentParser(description="逐 trial 稽核（做錯 / 沒放鬆 / 反應太慢）")
    ap.add_argument("csv", nargs="+", type=Path)
    ap.add_argument("--write", action="store_true",
                    help="寫出 _audit.csv 與 _exclude.txt（預設只印報告）")
    a = ap.parse_args()

    for p in a.csv:
        print("=" * 78)
        print(f"{p.name}")
        print("=" * 78)
        df, s = audit(p)

        print(f"{'rep':>4s} {'姿勢':<16s} {'反應':>6s} {'衰減':>6s} {'rest倍率':>8s} "
              f"{'最低一致':>8s}  判定")
        print("-" * 78)
        for _, r in df.iterrows():
            mark = "❌" if r.exclude else "✅"
            print(f"{int(r.rep):>4d} {str(r['pose']):<16s} "
                  f"{r.t_reach if pd.notna(r.t_reach) else float('nan'):>5.2f}s "
                  f"{r.t_decay if pd.notna(r.t_decay) else float('nan'):>5.2f}s "
                  f"{r.rest_ratio if pd.notna(r.rest_ratio) else float('nan'):>7.1f}x "
                  f"{r.min_agree if pd.notna(r.min_agree) else float('nan'):>7.0%}  "
                  f"{mark} {r.reason}")

        print(f"\n建議排除 {s['n_excluded']}/{s['n_trials']} 個 trial "
              f"（{s['flag_rate']:.0%}）")
        print(f"  成因拆解：姿勢做錯 {s['n_pose']} ／ rest 沒放鬆 {s['n_rest']} "
              f"／ 反應太慢 {s['n_slow']}（可重複計數）")
        if s["n_rest"] > s["n_pose"]:
            print("  → 多數是 **rest 沒放鬆**，那是協定的 return 段太短造成的，"
                  "\n     不是受試者做錯。重錄時 return 加長即可（現行 config 已是 2.5s）。")
        print(f"  反應時間中位 {s['t_reach_median']:.2f}s"
              f"（協定給 {C.CUE_PREPARE_SEC + C.CUE_MARGIN_START_SEC:.1f}s）")
        print(f"  衰減時間中位 {s['t_decay_median']:.2f}s"
              f"（協定預算 {s['decay_budget']:.1f}s）")
        if s["pose_rate"] > 0.25:
            print("\n⚠️ **姿勢做錯**的比例超過 25% —— 這通常不是受試者一直做錯，"
                  "\n   而是**電極位置有問題**（貼歪、接觸不良、或某個通道沒訊號）。"
                  "\n   先跑 check_states.py 看每個 DOF 的狀態分不分得開，"
                  "\n   不要急著把這些 trial 全部刪掉。")
        print("\n⚠️ 這是「其他 trial 的共識」不是真理：某個姿勢 5 次裡錯 2 次且錯法一樣，"
              "\n   共識會翻轉、反而標記做對的那幾個。請人工確認再排除。")

        if a.write:
            df.to_csv(p.with_name(p.stem + "_audit.csv"), index=False,
                      encoding="utf-8")
            ex = p.with_name(p.stem + "_exclude.txt")
            ex.write_text("\n".join(str(int(r)) for r in df[df.exclude].rep),
                          encoding="utf-8")
            print(f"\n已寫出：{p.stem}_audit.csv / {p.stem}_exclude.txt")


if __name__ == "__main__":
    main()
