"""從 record_session.py 的 cue 檔產生標籤。

這個模組取代 labeling.py 的啟發法。差別很大，值得寫清楚：

    labeling.py（舊資料用）—— 沒有 cue，只能：
        gesture ← 檔名（整檔一個手勢 → 手勢與錄音檔共線，模型可作弊）
        相位 ← 從包絡線猜（試三次都失敗：門檻失效 / 相位定義錯 / 每通道反相關）

    cues.py（新資料用）—— 有 cue，直接讀：
        gesture ← 該視窗落在哪一個 cue 區間
        相位   ← cue 事件本身（prepare→rest, go→action, release→rest）
        rest   ← 開頭/結尾空白區 + 每個 trial 的休息段，是**真正的 rest**，
                 不是「包絡線比較低的地方」

而且因為一段錄音內有多個手勢，手勢不再與錄音檔共線 ——
這才讓「模型學到的是肌電形態」變成可驗證的主張。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from semg.v2 import config as C


def cue_path_for(csv_path) -> Path | None:
    p = Path(csv_path)
    c = p.with_name(p.stem + "_cues.csv")
    return c if c.exists() else None


def meta_path_for(csv_path) -> Path | None:
    p = Path(csv_path)
    m = p.with_name(p.stem + "_meta.json")
    return m if m.exists() else None


def load_meta(csv_path) -> dict:
    """讀 `<stem>_meta.json`，**容忍舊檔的編碼**。

    ⚠️ 2026-08-07 之前錄的 meta.json 是 **cp950** —— `record_session.py` 當時
    用 `Path.write_text()` 沒指定編碼，Windows(zh-TW) 就用了系統 locale。
    那個 bug 已修（現在一律 utf-8），但**既有檔案不會自己變好**，
    所以這裡保留 fallback。直接 `json.load()` 會 UnicodeDecodeError。
    """
    m = meta_path_for(csv_path)
    if m is None:
        return {}
    raw = m.read_bytes()
    for enc in ("utf-8", "cp950", "big5", "gbk"):
        try:
            return json.loads(raw.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return {}


def protocol_version(csv_path) -> int:
    """這份錄音是用哪一版協定錄的。

    v1（沒有 `protocol_version` 欄位）= go 3.0 / return 1.5 / preview 2.5
    v2 = go 4.0 / return 2.5 / preview 1.5（依第一份真實錄音量測後調整）

    ⚠️ 兩版**都可以**用同一套程式建資料集 —— 段落邊界來自 cue 檔的實際
    sample_index，不是 config 的秒數。差別在於 v1 的 rest 段衰減預算只有
    `1.5 + 1.0 = 2.5s`（覆蓋約 80%），v2 是 `2.5 + 1.0 = 3.5s`（覆蓋 98%），
    所以**v1 的 rest 類別比較容易混到殘餘活動**。混用時要知道這件事。
    """
    return int(load_meta(csv_path).get("protocol", {}).get("protocol_version", 1))


def load_cues(cue_csv) -> pd.DataFrame:
    df = pd.read_csv(cue_csv)
    need = {"sample_index", "event", "gesture"}
    if not need.issubset(df.columns):
        raise ValueError(f"{cue_csv} 缺欄位 {need - set(df.columns)}")
    return df


def auto_rest_margin(cues: pd.DataFrame, n_samples: int,
                     want: float = C.CUE_MARGIN_REST_START_SEC,
                     floor: float = C.CUE_MARGIN_START_SEC) -> tuple[float, float]:
    """這份錄音的 rest 段**負擔得起**多大的起始 margin。回傳 (實際採用, 理想值)。

    ★ 為什麼需要這個（實際踩到的靜默失敗）：
      margin 是依「協定給 rest 3.5s」設計的。但**舊協定的錄音只有 2.0s**，
      扣掉 1.0s margin + 0.2s 尾巴只剩 0.8s < 一條序列所需的 0.95s
      → `make_sequences` 對「視窗數 < SEQ_LEN」的群組是 `continue`（不報錯）
      → **rest 這個最大的類別完全消失，而且沒有任何徵兆**。

      `config.validate()` 擋不住這個 —— 它檢查的是 config 的秒數，
      不是「手上這份錄音實際錄了多久」。

    取捨（兩邊都不能靜默）：
      margin 太小 → rest 標籤混到肌肉還在放鬆的殘餘活動
      margin 太大 → rest 完全沒有訓練資料
    所以取「還能產出至少一條序列的最大 margin」，並由呼叫端**印出來**。
    """
    rows = cues.sort_values("sample_index").reset_index(drop=True)
    lens = []
    for i, r in rows.iterrows():
        if str(r["event"]) != "release":
            continue
        e = int(rows.loc[i + 1, "sample_index"]) if i + 1 < len(rows) else n_samples
        lens.append((e - int(r["sample_index"])) / C.FS)
    if not lens:
        return want, want
    raw = float(np.median(lens))
    need = C.WIN_SEC + (C.SEQ_LEN - 1) * C.STEP_SEC     # 一條序列涵蓋的時間
    afford = raw - need - C.CUE_MARGIN_END_SEC
    return float(min(want, max(floor, afford))), want


def intervals(cues: pd.DataFrame, n_samples: int,
              margin_start_sec: float = C.CUE_MARGIN_START_SEC,
              margin_end_sec: float = C.CUE_MARGIN_END_SEC,
              margin_rest_start_sec: float = C.CUE_MARGIN_REST_START_SEC) -> list[dict]:
    """把 cue 事件轉成 [start, end) 樣本區間。

    有標籤的段落兩端各往內縮一段 margin，**三個值都不同**，因為吸收的東西不同：

      go 起始（0.5s）  ：受試者可能還沒完全擺好、還在微調 → 變異大
      rest 起始（1.0s）：★ 肌肉還在**放鬆** —— 這是生理衰減，比擺姿勢慢得多
      結束（0.2s）     ：提前放鬆，通常很短

    ⚠️ 起始 margin **不是**在補「看到提示的反應延遲」——
       反應時間由 `preview` + `prepare` 提供，
       `go` 開始時受試者應該已經維持住目標姿勢了。
       （舊的等長協定裡 `go` 才是「現在開始出力」，那時 margin 確實是補反應延遲。）

    ★ 為什麼 rest 要另外給更大的起始 margin（第一份真實錄音量出來的）：
       量 release 段多久才降回基線 1.5× 內 → 中位 0.28s，但 **p90 1.74s**。
       用 0.5s 時 **13/40 個 trial** 的 rest 標籤仍高於基線 1.5×，
       舉手臂的姿勢最嚴重（shoulder_flex 的前 0.5s 是基線的 8.07×）。
       rest 是最大的類別，被高活動訊號汙染會讓模型把「還在出力」學成 rest
       → 即時使用時機械手臂過早放下。
       （`CUE_RETURN_SEC` 也同步加長到 2.5s 先吸收一部分；兩者都需要，
         因為衰減有長尾，只靠 margin 會把 rest 的有標籤長度吃光。）

    這整套邊界處理是 v1 完全沒有、只能靠猜的東西。
    """
    ms = int(margin_start_sec * C.FS)
    ms_rest = int(margin_rest_start_sec * C.FS)
    me = int(margin_end_sec * C.FS)
    out = []
    rows = cues.sort_values("sample_index").reset_index(drop=True)
    for i, r in rows.iterrows():
        ev = str(r["event"])
        if ev not in ("go", "return", "release", "prepare", "preview", "record_start"):
            continue
        s = int(r["sample_index"])
        e = int(rows.loc[i + 1, "sample_index"]) if i + 1 < len(rows) else n_samples
        if ev == "go":
            # 移動段 → 該動作對應的 DOF 狀態組合
            act = str(r["gesture"])           # cue 檔沿用 gesture 欄位名，內容是動作名
            out.append({"start": s + ms, "end": e - me,
                        "gesture": act, "dof": C.action_to_dof(act),
                        "phase": "action", "rep": int(r.get("rep", -1))})
        elif ev in ("release", "record_start"):
            # 靜止休息段與開頭空白 → 所有 DOF 都 rest。
            # ★ 用**較大的** rest 起始 margin 吸收肌肉放鬆的生理衰減（見 docstring）。
            #   record_start（開頭空白段）本來就從靜止開始，但用同一個值不會有壞處，
            #   而且讓「rest 這一類的取樣方式」在整份錄音裡完全一致。
            out.append({"start": s + ms_rest, "end": e - me,
                        "gesture": "rest", "dof": C.REST_DOF,
                        "phase": "rest", "rep": int(r.get("rep", -1))})
        elif ev in ("prepare", "return", "preview"):
            # ★ 三段都**整段丟棄**，理由不同：
            #   preview：已預告下一個姿勢，受試者可能提前緊張 → 汙染「完全放鬆」
            #   prepare：受試者已知道要做什麼但還沒動，且正在擺起始位置 → 語意模糊
            #   return ：回程的關節運動方向與去程**相反**（elbow_flex 的回程是伸展），
            #            標成 rest 會教模型「伸展的動作 = 靜止」，標成 ext 又不對
            #            （回程多半是重力輔助的離心控制，與主動伸展的訊號不同）。
            #            丟掉是唯一不會教錯的選擇。
            out.append({"start": s, "end": e, "gesture": "", "dof": None,
                        "phase": "discard", "rep": int(r.get("rep", -1))})
    return [iv for iv in out if iv["end"] > iv["start"]]


def sample_labels(cues: pd.DataFrame, n_samples: int,
                  margin_start_sec: float = C.CUE_MARGIN_START_SEC,
                  margin_end_sec: float = C.CUE_MARGIN_END_SEC,
                  margin_rest_start_sec: float = C.CUE_MARGIN_REST_START_SEC
                  ) -> np.ndarray:
    """★ 逐樣本的**多輸出**標籤 (n_samples, N_DOF)。

    每一欄是一個自由度的狀態索引；-1 = 丟棄（準備段與 margin 區）。
    這取代了原本「一個手勢類別」的單欄標籤 —— 因為同時動作
    （食指+拇指、肩+肘）在單欄表示法下必須額外開類別，多欄則自然成立。
    """
    lab = np.full((n_samples, C.N_DOF), -1, dtype=np.int8)
    for iv in intervals(cues, n_samples, margin_start_sec, margin_end_sec,
                        margin_rest_start_sec):
        if iv["phase"] == "discard" or iv["dof"] is None:
            continue
        lab[iv["start"]:iv["end"]] = np.asarray(iv["dof"], dtype=np.int8)
    return lab


def window_label(lab: np.ndarray, start: int, end: int,
                 purity: float = 0.95) -> np.ndarray | None:
    """一個視窗的多輸出標籤 (N_DOF,)；不夠純就回傳 None。

    要求**整段視窗的 DOF 組合完全一致**且佔比 ≥ purity。
    不用多數決是刻意的：跨越 cue 邊界的視窗同時含 rest 與出力波形，
    拿它訓練會把混合波形教成單一狀態，直接汙染 onset 判別。
    """
    seg = lab[start:end]
    if len(seg) == 0:
        return None
    valid = seg[(seg >= 0).all(axis=1)]
    if len(valid) == 0:
        return None
    # 把每列的 DOF 組合壓成一個可比較的鍵
    combos, counts = np.unique(valid, axis=0, return_counts=True)
    j = int(counts.argmax())
    if counts[j] / len(seg) < purity:
        return None
    return combos[j].astype(np.int8)


def calibration_span(cues: pd.DataFrame, n_samples: int,
                     badspans: "list[tuple[float, float]] | None" = None,
                     quiet: bool = False) -> tuple[int, int]:
    """開頭空白段的樣本範圍 [start, end) —— 尺度校準**唯一**該用的區段。

    ★ 為什麼這件事重要（實測發現的 bug）：
    原本訓練時用「整檔 MAD」算尺度，推論時用「開頭休息段 MAD」，
    兩者系統性差 1.27×（休息段沒有出力，MAD 較小）。
    結果推論時所有振幅特徵被放大 1.27×，分布偏移 →
    訓練 99.6% 全對，即時重播只剩 58%，93.6% 的預測都變成 rest。

    修法：**訓練與推論用同一種來源**。開頭空白段是唯一
    「離線和線上都一定拿得到、且內容定義相同」的區段
    （協定保證前 CUE_LEAD_SEC 秒純休息不下 cue）。

    這不是「MAD 比 z-score 穩健」的問題 —— 再穩健的統計量，
    只要訓練和推論算在不同區段上，就一定會偏移。
    """
    rows = cues.sort_values("sample_index")
    # ★ 停在**第一個 cue 事件**（含 preview）—— 只要畫面上出現任何提示，
    #   受試者就可能有預期性的張力，那段就不是「純空白」了。
    first_cue = rows[rows.event.isin(["preview", "prepare", "go"])]
    end = int(first_cue.iloc[0]["sample_index"]) if len(first_cue) else n_samples
    s, e = 0, min(max(end, C.FS), n_samples)      # 至少 1 秒
    if not badspans:
        return s, e

    # ★ 開頭空白段本身壞掉時，必須改用別的休息段（2026-08-13 加）。
    #
    #   為什麼非改不可：`_badspans.txt` 只遮**標籤**，遮不到尺度估計。
    #   如果壞段落剛好蓋住開頭空白段，`file_scales` 仍會從汙染的訊號估尺度，
    #   而尺度是**整份錄音共用**的 —— 一段 3.5 秒的假影會毀掉全部 550 秒。
    #
    #   實測：17-11-49 的 ch4 校準尺度 43.73 µV（乾淨錄音是 1.44 µV，30 倍），
    #   15-50-34 的 ch1 是 267.29 µV（74 倍）。後果是該通道所有振幅相對特徵
    #   被除以一個大 30~74 倍的數 → 看起來永遠接近零，
    #   而且 `prop_` / `grp_` 這些跨通道比例會把災情擴散到其他通道。
    #   17-11-49 的手肘因此在 61.4% 的 rest 視窗上誤觸發。
    #
    #   ⚠️ 這會讓尺度來源從「開頭空白段」變成「某個 release 休息段」。
    #      兩者都是休息，但 release 緊接在出力之後，殘餘張力較高。
    #      能這樣做的前提是**訓練與推論呼叫同一個函式、傳同一份 badspans**
    #      —— 訓練/推論用不同區段估尺度正是本專案踩過最貴的坑
    #      （全對率 99.6% → 58%）。
    def hits_bad(a: int, b: int) -> bool:
        return any(not (b <= int(x * C.FS) or a >= int(y * C.FS))
                   for x, y in badspans)

    if not hits_bad(s, e):
        return s, e

    # 收集所有**完全乾淨**的休息區間，取最長的那個。
    # ⚠️ 不能要求「長度 ≥ 開頭空白段」—— V3 的 release 扣掉 margin 只剩約 2.3s，
    #    而開頭空白段有 3.5s，這個條件會篩掉每一個 release 段，
    #    只剩結尾那段空白能通過（第一版就是這樣誤打誤撞選到 541s 的）。
    #    改成「取最長的乾淨休息段」，長度下限 1.5s 就夠估 MAD。
    MIN_SEC = 1.5
    cand = [(iv["end"] - iv["start"], iv["start"], iv["end"])
            for iv in intervals(cues, n_samples)
            if iv["phase"] == "rest" and not hits_bad(iv["start"], iv["end"])]
    cand = [c for c in cand if c[0] >= MIN_SEC * C.FS]
    if cand:
        n, a, b = max(cand)                       # 最長的；同長取最早
        b = min(b, a + (e - s))                   # 不需要比原本更長
        if not quiet:
            print(f"     ⚠️ 開頭空白段落在壞段落內，校準改用 "
                  f"{a / C.FS:.1f}~{b / C.FS:.1f}s 的休息段"
                  f"（{(b - a) / C.FS:.1f}s，共 {len(cand)} 段可選）")
        return a, b
    if not quiet:
        print("     ❌ 找不到任何乾淨的休息段可以校準 —— 這份錄音的尺度不可信")
    return s, e


def dof_to_action(dof: np.ndarray) -> str:
    """DOF 狀態組合 → 已知的動作名；不在 ACTIONS 裡就回傳組合描述。

    模型**可以**輸出訓練時沒 cue 過的組合，這個函式讓那種情況也有可讀名稱。
    """
    t = tuple(int(v) for v in dof)
    for name in C.ACTIONS:
        if C.action_to_dof(name) == t:
            return name
    if t == C.REST_DOF:
        return "rest"
    parts = [f"{d}_{C.DOF_STATES[d][v]}" for d, v in zip(C.DOFS, t)
             if C.DOF_STATES[d][v] != "rest"]
    return "+".join(parts) if parts else "rest"


def trial_split(cues: pd.DataFrame, n_folds: int = 5, seed: int = C.SEED):
    """按 **trial** 切分，不是按樣本或視窗切。

    這是避免重演「99.9% 其實是洩漏」的關鍵：同一個 trial 內的視窗高度重疊，
    只要 trial 不跨組，測試集就不會有訓練樣本的近鄰。
    """
    reps = sorted({int(r) for r in cues.get("rep", pd.Series(dtype=int)) if int(r) >= 0})
    if not reps:
        return []
    rng = np.random.default_rng(seed)
    shuffled = np.array(reps)
    rng.shuffle(shuffled)
    return [list(part) for part in np.array_split(shuffled, n_folds)]


def summarise(csv_path) -> dict:
    """給人看的 cue 檔摘要，錄完可以立刻確認協定跑對了。"""
    cp = cue_path_for(csv_path)
    if cp is None:
        return {"has_cues": False}
    cues = load_cues(cp)
    n = int(pd.read_csv(csv_path, usecols=["sample_index"])["sample_index"].max()) + 1
    lab = sample_labels(cues, n)
    ok = (lab >= 0).all(axis=1)
    out = {"has_cues": True, "n_samples": n, "duration_s": n / C.FS,
           "n_trials": int((cues.event == "go").sum()),
           "per_dof": {}, "per_action": {}}
    # 每個 DOF 各狀態的時間佔比 —— 檢查有沒有哪個狀態幾乎沒錄到
    for j, d in enumerate(C.DOFS):
        out["per_dof"][d] = {}
        for si, st in enumerate(C.DOF_STATES[d]):
            k = int(((lab[:, j] == si) & ok).sum())
            if k:
                out["per_dof"][d][st] = {"sec": round(k / C.FS, 1),
                                         "frac": round(k / max(ok.sum(), 1), 3)}
    # 實際出現的 DOF 組合
    if ok.any():
        combos, counts = np.unique(lab[ok], axis=0, return_counts=True)
        for cb, ct in zip(combos, counts):
            out["per_action"][dof_to_action(cb)] = round(float(ct) / C.FS, 1)
    out["discarded_frac"] = round(float((~ok).mean()), 3)
    return out


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        print(f"\n{p}")
        s = summarise(p)
        if not s["has_cues"]:
            print("  ⚠️ 沒有對應的 _cues.csv → 只能用舊的檔名啟發法")
            continue
        print(f"  時長 {s['duration_s']:.1f}s，{s['n_trials']} 個 trial，"
              f"丟棄 {s['discarded_frac']:.1%}")
        print("  每個自由度的狀態時間：")
        for d, states in s["per_dof"].items():
            body = "  ".join(f"{st}={v['sec']:.0f}s({v['frac']:.0%})"
                             for st, v in states.items())
            print(f"    {C.DOF_LABELS_ZH[d]:4s} {body}")
        print("  實際出現的動作組合：")
        for a, sec in sorted(s["per_action"].items(), key=lambda kv: -kv[1]):
            print(f"    {a:24s} {sec:6.1f}s")
