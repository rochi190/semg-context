"""建資料集：切窗 → 抽特徵 → LOSO 分割。

標籤策略（見 docs/pipeline-v2-design.md 決策 2，已經歷三次修正）：
    gesture ← 檔名（唯一 100% 可靠的資訊）
    只保留「有在動」的視窗（三通道合併能量的百分位門檻）——
    相位與每通道標籤都試過且失敗，這裡只用能可靠取得的部分。

輸出快取成 .npz，避免每次訓練都重算特徵（28 個檔案 × 78k 視窗，抽特徵很慢）。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from semg.v2 import config as C
from semg.v2 import cues as Q
from semg.v2 import features as F
from semg.v2 import labeling as L
from semg.v2 import preprocess as P

CACHE = C.RESULTS_ROOT / "dataset.npz"


def discover() -> list[tuple[str, Path, str]]:
    """找出所有可用錄音，回傳 (subject, csv_path, gesture_or_multi)。

    兩種來源：
      1. `data/raw/multi/<subject>/*.csv` —— 新協定，一檔多手勢，配 `_cues.csv`
      2. `data/<subject>/*.csv`          —— 舊協定，一檔一手勢，靠檔名
    有 cue 檔的優先，且不會被檔名解析卡住（新檔名裡沒有手勢字串）。
    """
    found: list[tuple[str, Path, str]] = []
    multi_root = C.DATA_ROOT / "raw" / "multi"
    if multi_root.exists():
        for sub in sorted(p for p in multi_root.iterdir() if p.is_dir()):
            # ★ rglob 而非 glob —— 錄音可能再分子資料夾（例如依場地分
            #   `106測得` / `實驗室測得`）。用 glob 會**靜默地找不到**那些錄音，
            #   資料集少一半卻不報錯。
            for p in sorted(sub.rglob("*.csv")):
                # ★ 判準是「**有沒有配對的 _cues.csv**」，不是排除已知的後綴。
                #   錄製與分析會在同一個資料夾產生一堆附屬 CSV
                #   （_cues / _audit / 之後可能還有別的），列黑名單遲早會漏。
                #   實際踩過：加了 audit_trials 之後，_audit.csv 被當成錄音檔，
                #   建資料集直接 ValueError（找不到 ch1..ch7 欄位）。
                if Q.cue_path_for(p) is None:
                    continue
                found.append((sub.name, p, "multi"))
    if C.USE_LEGACY_DATA:
        for s in C.SUBJECTS:
            d = C.DATA_ROOT / s
            if not d.exists():
                continue
            for p in sorted(d.glob("*.csv")):
                g = L.gesture_from_filename(p.stem)
                if g != "unknown":
                    found.append((s, p, g))
    return found


def load_manifest(path) -> set[str] | None:
    """讀錄音清單檔（一行一個 stem，`#` 可註解），回傳 stem 集合。

    ★ 為什麼要有這個：先前用 symlink 農場做資料子集，實測很脆 ——
      使用者把錄音搬進場地子資料夾時所有連結一次全斷，而且是**跑到一半**才斷。
      清單檔不動檔案系統，而且進得了版控，
      是「這批結果用了哪些錄音」的唯一事實來源。
    """
    if path is None:
        return None
    out = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            out.add(line)
    if not out:
        raise SystemExit(f"{path} 沒有列出任何錄音")
    return out


def load_badspans(csv_path) -> list[tuple[float, float]]:
    """讀 `<stem>_badspans.txt`，回傳 [(起始秒, 結束秒), …]。

    每行 `起始秒 結束秒 [備註]`，`#` 可註解。用來遮蔽肉眼判讀出的壞段落
    （電極飽和、突波、接觸不良），與 `_exclude.txt`（整個 trial 做錯）互補：
    前者是**訊號**問題、後者是**執行**問題，成因不同所以分開記。

    ⚠️ 遮蔽**不能用切片實作** —— `load_recording(trim=False)` 的樣本索引必須與
       cue 檔的 `sample_index` 對齊，任何裁切都會讓整份錄音的標籤錯位。
       呼叫端要做的是把該範圍的標籤設成 -1，讓視窗自然產不出來。
    """
    p = Path(csv_path)
    f = p.with_name(p.stem + "_badspans.txt")
    if not f.exists():
        return []
    out = []
    for ln, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            raise SystemExit(f"{f}:{ln} 格式錯誤，需要「起始秒 結束秒」：{line!r}")
        a, b = float(parts[0]), float(parts[1])
        if b <= a:
            raise SystemExit(f"{f}:{ln} 結束秒必須大於起始秒：{line!r}")
        out.append((a, b))
    return out


def apply_badspans(labels: np.ndarray, seg_of: np.ndarray,
                   spans: list[tuple[float, float]]) -> int:
    """把壞段落範圍內的標籤設成 -1，回傳被遮蔽的樣本數。

    `window_label` 要求視窗內 95% 的樣本標籤一致，所以只要視窗與壞段落
    重疊超過 5%，該視窗就會被丟掉 —— 不需要額外的邊界處理。
    """
    n = len(labels)
    total = 0
    for a, b in spans:
        s, e = max(0, int(a * C.FS)), min(n, int(b * C.FS))
        if e <= s:
            continue
        labels[s:e] = -1
        seg_of[s:e] = -1
        total += e - s
    return total


def load_deadch(csv_path) -> list[int]:
    """讀 `<stem>_deadch.txt`（一行一個 1-based 通道號），回傳 0-based 索引。

    給「整條通道壞掉」用 —— 那種問題沒有時間範圍，`_badspans.txt` 處理不了。
    ⚠️ 置零本身是**分布偏移**：其他錄音從沒出現過「整條通道完全靜止」，
       模型會看到一個假的「這塊肌肉完全沒出力」訊號。
       所以預設**不啟用**，只在 A/B 比較的變體 D 裡開來量它值不值得。
    """
    p = Path(csv_path)
    f = p.with_name(p.stem + "_deadch.txt")
    if not f.exists():
        return []
    out = []
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            out.append(int(line) - 1)
    return out


def load_exclude(csv_path) -> set[int]:
    """讀 `<stem>_exclude.txt`（一行一個 rep 編號），沒有就回傳空集合。

    由 `audit_trials.py --write` 產生。刻意做成**獨立的檔案**而不是寫回 cue 檔：
    cue 檔是錄製當下的原始紀錄，不該被事後的分析修改；
    排除清單是可以反覆調整、也可以人工編輯的判斷。
    """
    p = Path(csv_path)
    f = p.with_name(p.stem + "_exclude.txt")
    if not f.exists():
        return set()
    out = set()
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()         # 允許 `12  # 手肘做錯` 這種註解
        if line:
            out.add(int(line))
    return out


def activity_mask(env: np.ndarray) -> np.ndarray:
    """哪些包絡線點算「有在動」。三通道合併能量 + 百分位門檻。

    只需二分（有動/沒動），不需分相位或分通道 —— 這是唯一夠穩健的判定。
    """
    e = env.mean(axis=1)
    base = np.percentile(e, C.LBL_BASE_PCT)
    rng = max(np.percentile(e, C.LBL_PEAK_PCT) - base, 1e-12)
    return e > base + C.LBL_ACTIVE_FRAC * rng


def build(force: bool = False, verbose: bool = True,
          include: str | Path | None = None, use_badspans: bool = True,
          zero_dead_channels: bool = False, use_hampel: bool | None = None) -> dict:
    """建資料集。

    `include`            ：錄音清單檔路徑（見 `load_manifest`）。None = 全收。
    `use_badspans`       ：是否套用 `<stem>_badspans.txt`。做 A/B 比較時可關掉。
    `zero_dead_channels` ：是否套用 `<stem>_deadch.txt` 把整條壞通道置零。
                           **預設關閉**，理由見 `load_deadch`。
    `use_hampel`         ：None = 照 config；False = 拿掉 Hampel（省 77% 計算，
                           但**必須配著重訓的模型用**）。會記進資料集與 bundle。
    """
    hampel = C.USE_HAMPEL if use_hampel is None else bool(use_hampel)
    if CACHE.exists() and not force:
        z = np.load(CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}

    C.RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    X, y_g, y_dof, subj, src_file, trial, segment, lab_src = ([] for _ in range(8))
    t0 = time.perf_counter()

    files = discover()
    if not files:
        raise SystemExit(f"在 {C.DATA_ROOT} 底下找不到任何錄音")

    want = load_manifest(include)
    if want is not None:
        have = {p.stem for _, p, _ in files}
        # ★ 清單裡列了卻找不到的檔案要**大聲報錯**，不能靜默少收 ——
        #   先前 discover() 用 glob 漏掉子資料夾，資料集少一半卻不報錯。
        missing = want - have
        if missing:
            raise SystemExit(
                f"清單 {include} 列了 {len(missing)} 份在 {C.DATA_ROOT} 找不到的錄音：\n"
                + "\n".join(f"     {m}" for m in sorted(missing)))
        files = [f for f in files if f[1].stem in want]
        if verbose:
            print(f"依清單 {include} 選用 {len(files)} 份錄音"
                  f"（{len(have)} 份可用中）\n")

    # ★ 先掃一遍協定版本 —— 混用新舊協定的錄音不會報錯，但 rest 類別的
    #   汙染程度不同（v1 的衰減預算 2.5s 覆蓋 ~80%，v2 是 3.5s 覆蓋 98%），
    #   等於同一個類別裡混了兩種分布。可以混，但必須**知道自己在混**。
    rest_warn: list[tuple[str, float, float]] = []
    bad_note: list[tuple[str, int, float]] = []
    dead_note: list[tuple[str, list[int]]] = []
    vers = {p: Q.protocol_version(p) for _, p, _ in files if Q.cue_path_for(p)}
    if len(set(vers.values())) > 1:
        print("⚠️ 這個資料集混用了不同版本的錄製協定：")
        for v in sorted(set(vers.values())):
            names = [p.stem for p, vv in vers.items() if vv == v]
            print(f"     v{v}: {len(names)} 份 —— {', '.join(n[:28] for n in names[:3])}"
                  + (" …" if len(names) > 3 else ""))
        print("   v1 的 rest 段衰減預算較短（2.5s vs 3.5s），rest 類別較易混到殘餘活動。")
        print("   建議：先分開訓練比較，確認影響再決定要不要合併。\n")

    for i, (s, path, g) in enumerate(files, 1):
        cue_csv = Q.cue_path_for(path)
        has_cues = cue_csv is not None and C.LABEL_SOURCE_PREFER_CUES

        # 有 cue 時**不裁切頭尾**，否則 cue 的 sample_index 會對不上
        raw = P.load_recording(path, trim=not has_cues)
        dead = load_deadch(path) if zero_dead_channels else []
        if dead:
            raw[:, dead] = 0.0                    # 濾波**之前**置零，避免飽和值溢到鄰近時間
            dead_note.append((path.stem, [d + 1 for d in dead]))
        filt = P.filter_causal(raw, use_hampel=hampel)

        if has_cues:
            cues = Q.load_cues(cue_csv)
            # ★ 尺度只從開頭空白段算 —— 必須與即時推論的校準來源完全一致。
            #   用整檔算會讓訓練/推論差 1.27×，實測會把 99.6% 打到 58%。
            # ★ badspans 要傳進去 —— 開頭空白段若壞掉，尺度會毀掉整份錄音。
            #   `predict_and_plot` / `infer.replay` 必須傳同一份，否則
            #   訓練與推論的尺度來源不同 = 本專案踩過最貴的坑。
            spans = load_badspans(path) if use_badspans else []
            cs, ce = Q.calibration_span(cues, len(filt), badspans=spans)
            scales, epss = F.file_scales(filt[cs:ce])
            # ★ rest 起始 margin 依**這份錄音實際的 release 長度**決定。
            #   舊協定只錄 2.0s，扣滿額 margin 後不足一條序列 →
            #   rest 類別會**靜默消失**（make_sequences 對太短的群組是 continue）。
            #   config.validate() 擋不住，因為它看的是 config 不是手上的檔案。
            m_rest, m_want = Q.auto_rest_margin(cues, len(filt))
            if m_rest < m_want:
                rest_warn.append((path.stem, m_rest, m_want))
            slab = Q.sample_labels(cues, len(filt), margin_rest_start_sec=m_rest)
            rep_of = np.full(len(filt), -1, dtype=np.int32)
            # ★ seg_of：每個**有標籤區間**一個唯一編號。
            #   一個 trial 含兩段不連續的有標籤區間（go 與 release），
            #   中間隔著被丟棄的 return 段。若用 trial 當序列分組鍵，
            #   `make_sequences` 會把 go 的尾巴與 release 的開頭黏成一條序列 ——
            #   那兩個視窗實際相隔 1.5s 以上，卻被當成相隔 50ms（偽造的時間連續性），
            #   而且標籤取序列最後一個視窗 → 標成 rest 卻含最多 9 個姿勢視窗。
            #   實測未修正前有 18.0% 的序列是這樣。
            seg_of = np.full(len(filt), -1, dtype=np.int32)
            for k, iv in enumerate(Q.intervals(cues, len(filt),
                                               margin_rest_start_sec=m_rest)):
                rep_of[iv["start"]:iv["end"]] = iv["rep"]
                if iv["phase"] != "discard":
                    seg_of[iv["start"]:iv["end"]] = k

            # ★ 排除稽核標記的 trial（做錯姿勢 / rest 沒放鬆 / 反應太慢）。
            #   清單由 `python -m semg.v2.audit_trials <csv> --write` 產生，
            #   **人工確認過**才會生效 —— 稽核報的是「其他 trial 的共識」不是真理。
            #   dropout 之類的正則化擋不住標籤錯誤：每個姿勢只有 5 次重複，
            #   錯 1 次就是該姿勢 20% 的資料錯，那是系統性偏差不是隨機雜訊。
            excl = load_exclude(path)
            if excl:
                drop = np.isin(rep_of, list(excl))
                seg_of[drop] = -1                 # 讓這些樣本產不出視窗
                slab[drop] = -1

            # ★ 訊號層面的壞段落（電極飽和/突波），與上面的「做錯 trial」互補。
            #   一樣是把標籤設 -1 而**不是**切掉樣本 —— cue 的 sample_index
            #   必須與陣列索引對齊，切一刀整份錄音的標籤就全錯位了。
            if spans:
                n_mask = apply_badspans(slab, seg_of, spans)
                bad_note.append((path.stem, len(spans), n_mask / C.FS))
        else:
            scales, epss = F.file_scales(filt)     # 舊資料沒有空白段，只能用整檔
            env = P.rms_envelope(filt)
            act = activity_mask(env)

        n_kept, kept_g = 0, {}
        for start in range(0, len(filt) - C.WIN_SAMPLES + 1, C.STEP_SAMPLES):
            end = start + C.WIN_SAMPLES
            if has_cues:
                dof = Q.window_label(slab, start, end)
                if dof is None:                  # 跨 cue 邊界或準備段 → 丟棄
                    continue
                gname = Q.dof_to_action(dof)
                gi = C.GESTURES.index(gname) if gname in C.GESTURES else -1
                rep = int(np.bincount(np.maximum(rep_of[start:end], 0)).argmax())
                seg = int(np.bincount(np.maximum(seg_of[start:end], 0)).argmax())
            else:
                # 舊資料退路：整檔一個手勢 + 活動門檻（已知會讓手勢與檔案共線）
                e0 = start // C.ENV_STEP
                e1 = max(e0 + 1, end // C.ENV_STEP)
                if act[e0:e1].mean() < 0.6:
                    continue
                gi, gname, rep, seg = C.GESTURES.index(g), g, -1, -1
                dof = np.asarray(C.REST_DOF, dtype=np.int8)   # 舊資料沒有 DOF 標籤
            X.append(F.extract(filt[start:end], scales, epss))
            y_g.append(gi)
            y_dof.append(np.asarray(dof, dtype=np.int8))
            subj.append(s)
            src_file.append(path.stem)
            trial.append(f"{path.stem}#{rep}")        # 交叉驗證的分組鍵
            segment.append(f"{path.stem}#{rep}#{seg}")  # 序列堆疊的分組鍵（見上方 seg_of）
            lab_src.append("cue" if has_cues else "filename")
            n_kept += 1
            kept_g[gname] = kept_g.get(gname, 0) + 1
        if verbose:
            n_ex = len(load_exclude(path)) if has_cues else 0
            ex_tag = f"  ⛔排除 {n_ex} trial" if n_ex else ""
            tag = "cue " if has_cues else "檔名"
            dist = " ".join(f"{k}={v}" for k, v in sorted(kept_g.items()))
            print(f"  [{i:2d}/{len(files)}] {s:8s} {tag} {path.stem[:26]:28s} "
                  f"{n_kept:6d} 視窗  ({time.perf_counter()-t0:5.0f}s){ex_tag}  {dist}",
                  flush=True)

    if dead_note:
        print("\n⚠️ 套用了整條通道置零（_deadch.txt）：")
        for stem, chs in dead_note:
            print(f"     {stem[:34]:36s} ch{', ch'.join(map(str, chs))}")
        print("   這會產生其他錄音沒有的輸入分布（整條通道恆為 0），"
              "只應在 A/B 比較裡使用。\n")

    if bad_note:
        print("\n🩹 套用了壞段落遮蔽（_badspans.txt）：")
        for stem, n, sec in bad_note:
            print(f"     {stem[:34]:36s} {n} 段、共 {sec:.0f}s")
        print("   （遮蔽的是**標籤**不是樣本；與該段重疊 >5% 的視窗會被丟掉。）\n")

    if rest_warn:
        print("\n⚠️ 下列錄音的 release 段太短，rest 起始 margin 被迫縮小：")
        for stem, got, want in rest_warn:
            print(f"     {stem[:34]:36s} {want:.1f}s → {got:.1f}s")
        print("   （不縮的話 rest 類別會**完全沒有訓練序列**，而且不會報錯。）")
        print("   代價：這些錄音的 rest 標籤較容易混到肌肉還在放鬆的殘餘活動。")
        print("   根治：重錄時把 release 段加長（現行 config 已是 5.0s）。\n")

    out = {"X": np.stack(X).astype(np.float32),
           "y_dof": np.stack(y_dof).astype(np.int64),      # ★ (N, N_DOF) 多輸出標籤
           "dof_names": np.asarray(C.DOFS),
           "dof_n_states": np.asarray(C.DOF_N_STATES),
           "y_gesture": np.asarray(y_g, dtype=np.int64),   # 舊單欄標籤（-1 = 沒對應動作名）
           "subject": np.asarray(subj),
           "src_file": np.asarray(src_file),
           "trial": np.asarray(trial),          # CV 分組：go 與 rest 同屬一個 trial
           "segment": np.asarray(segment),      # 序列堆疊：go 與 rest 是**不同**區間
           "label_source": np.asarray(lab_src),
           "use_hampel": np.asarray(hampel),
           "feat_version": np.asarray(C.FEAT_VERSION),
           "channel_names": np.asarray(C.CH_NAMES),
           "feature_names": np.asarray(F.FEATURE_NAMES)}
    np.savez_compressed(str(CACHE), **out)
    if verbose:
        print(f"\n  已快取 → {CACHE}  ({time.perf_counter()-t0:.0f}s)")
    return out


def loso_splits(d: dict):
    """LOSO：每次留一位受試者當測試集。回傳 (test_subject, train_idx, test_idx)。"""
    subj = d["subject"]
    for s in C.SUBJECTS:
        te = np.where(subj == s)[0]
        tr = np.where(subj != s)[0]
        if len(te) and len(tr):
            yield s, tr, te


def trial_group_splits(d: dict, n_folds: int = 5, seed: int = C.SEED):
    """按 **trial** 分組的 K 折。同一個 trial 的視窗高度重疊，絕不可跨越訓練/測試。

    這是修掉「隨機切窗 99.9%」那個洩漏的正確做法：
    分組鍵是 `<檔名>#<rep>`，所以同一次出力的所有視窗一定在同一折。
    """
    tr_key = d["trial"]
    groups = np.unique(tr_key)
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    for part in np.array_split(groups, n_folds):
        te = np.isin(tr_key, part)
        if te.any() and (~te).any():
            yield list(part), np.where(~te)[0], np.where(te)[0]


def summary(d: dict) -> pd.DataFrame:
    """每個受試者 × 每個 DOF 狀態的視窗數。多輸出沒有單一「類別」可以樞紐。"""
    rows = []
    for j, dof in enumerate(d["dof_names"]):
        states = C.DOF_STATES[str(dof)]
        for si, st in enumerate(states):
            m = d["y_dof"][:, j] == si
            for s in np.unique(d["subject"]):
                rows.append({"subject": s, "dof": str(dof), "state": st,
                             "n": int((m & (d["subject"] == s)).sum())})
    return (pd.DataFrame(rows)
            .pivot_table(index=["dof", "state"], columns="subject",
                         values="n", fill_value=0))


def action_summary(d: dict) -> pd.DataFrame:
    """實際出現的 DOF 組合（含訓練時沒 cue 過的組合，如果有的話）。"""
    from semg.v2 import cues as Q
    combos, counts = np.unique(d["y_dof"], axis=0, return_counts=True)
    return (pd.DataFrame({"action": [Q.dof_to_action(c) for c in combos],
                          "n_windows": counts})
            .sort_values("n_windows", ascending=False)
            .reset_index(drop=True))


if __name__ == "__main__":
    print("建立資料集（切窗 + 抽特徵）…")
    d = build(force=True)
    print(f"\nX shape: {d['X'].shape}   特徵維度: {d['X'].shape[1]}")
    print(f"\n每受試者 × DOF 狀態的視窗數：\n{summary(d).to_string()}")
    print(f"\n實際出現的動作組合：\n{action_summary(d).to_string()}")
    ns = int((d["label_source"] == "cue").sum())
    print(f"\n標籤來源：cue {ns:,} 視窗 / 檔名 {len(d['y_gesture'])-ns:,} 視窗")
    print(f"通道（{len(d['channel_names'])}）：{list(d['channel_names'])}")
    print(f"trial 數：{len(np.unique(d['trial']))}")
    print(f"\n每個自由度的狀態分布（這是模型真正要學的目標）：")
    N = len(d["y_dof"])
    for j, dof in enumerate(d["dof_names"]):
        parts = []
        for si, st in enumerate(C.DOF_STATES[str(dof)]):
            n = int((d["y_dof"][:, j] == si).sum())
            parts.append(f"{st}={n}({n/N:.0%})")
        print(f"  {C.DOF_LABELS_ZH[str(dof)]:4s} " + "  ".join(parts))
    # 檢查特徵有沒有 NaN/Inf（會讓訓練直接爆掉）
    bad = ~np.isfinite(d["X"])
    if bad.any():
        print(f"\n⚠️ 有 {bad.sum()} 個非有限值，分布在特徵：")
        for j in np.where(bad.any(axis=0))[0]:
            print(f"    {d['feature_names'][j]}: {bad[:, j].sum()}")
    else:
        print("\n✅ 特徵全部有限（無 NaN/Inf）")
