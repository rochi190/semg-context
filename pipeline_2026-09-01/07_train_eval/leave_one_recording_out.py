"""留一份錄音（leave-one-recording-out）：每份錄音都當一次測試集。

★ **本專案的主要評估指標。** 使用者定義：「LOSO 以各錄音當作一個 subject」——
  因為目前只有兩位受試者，而且**不打算做到跨人泛化**：
  即時 demo 的操作者會是訓練集裡的人。所以真正要回答的問題是

      「換一次貼片 / 換一天 / 換場地，還能不能用？」

  而不是「換一個人還能不能用」。留一份錄音正好對應前者。

⚠️ `--scheme subject`（真正的跨受試者）仍然保留，但只當**參考數字**。
   它回答的是不同的問題，而且目前只有 2 位受試者 = 2 折，統計上很弱。

流程：
    整批建一次資料集（特徵很貴，只算一次）
      → 對每份錄音：用其餘錄音訓練一個模型
      → 用 `predict_and_plot` 對留出的那份做**串流**推論並畫圖
      → 匯總

★ 用 `predict_and_plot.predict_recording` 而不是直接評估序列陣列，
  是因為前者走的是 `infer.Recognizer`（因果串流濾波 + 逐 DOF 投票），
  跟實際部署完全一致。直接評估序列陣列會少掉投票延遲，數字偏樂觀。

用法：
    PYTHONPATH=src python -m semg.v2.leave_one_recording_out \\
        --data-root <合併後的 data 目錄> --out results/loro
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from semg.v2 import config as C


def _bundle_from_fold(rf, d, arch: str, path: Path) -> None:
    """把 run_fold 的產物存成 infer.Recognizer 吃得下的 bundle。"""
    m = rf["model"].cpu().eval()
    torch.save({"state_dict": m.state_dict(), "arch": arch,
                "dof_names": list(d["dof_names"]),
                "dof_n_states": [int(v) for v in d["dof_n_states"]],
                "seq_len": C.SEQ_LEN, "n_features": int(d["X"].shape[1]),
                # ★ 前處理設定必須跟著模型走，推論時由它決定濾波器
                "use_hampel": bool(d["use_hampel"]) if "use_hampel" in d else C.USE_HAMPEL,
                "feat_version": int(d["feat_version"]) if "feat_version" in d else 1,
                "feature_names": [str(v) for v in d["feature_names"]],
                "channel_names": [str(v) for v in d["channel_names"]],
                "scaler_mean": rf["scaler"].mean_.astype(np.float32),
                "scaler_scale": rf["scaler"].scale_.astype(np.float32),
                "config": {"win_sec": C.WIN_SEC, "step_sec": C.STEP_SEC, "fs": C.FS}},
               path)


_G: dict = {}          # fork 前由 main() 填好，子程序繼承


def _do_fold(job):
    """跑一折：訓練 → 存 bundle → 對留出的錄音做串流評估 → 畫圖。

    ⚠️ **必須定義在模組層級**，不能寫成 `main()` 裡的閉包 ——
       `ProcessPoolExecutor` 即使用 fork context 也會 pickle 這個 callable，
       而閉包不可 pickle（`Can't pickle local object`）。
       實際踩到：第一版寫成閉包，序列模式正常、`--jobs 4` 一啟動就整批失敗。

    狀態透過模組層級的 `_G` 傳遞：fork 出來的子程序會繼承父程序記憶體，
    所以 X/y/seg 這些大陣列不需要序列化，只有 job 元組會被 pickle。
    """
    i, r, tr_w, te_w = job
    X, y, seg, src = _G["X"], _G["y"], _G["seg"], _G["src"]
    d, dofs, dof_ns = _G["d"], _G["dofs"], _G["dof_ns"]
    folds, csv_of = _G["folds"], _G["csv_of"]
    from semg.v2 import train as T
    from semg.v2 import predict_and_plot as PP
    from semg.v2 import preprocess as P
    from semg.v2 import cues as Q
    out_lines = [f"[{i}/{len(folds)}] 留出 {r}   訓練 {tr_w.sum():,} 視窗 / "
                 f"測試 {te_w.sum():,}"]
    Xtr, ytr = T.make_sequences(X[tr_w], y[tr_w], seg[tr_w])
    Xte, yte = T.make_sequences(X[te_w], y[te_w], seg[te_w])
    if not len(Xtr) or not len(Xte):
        return out_lines + ["    跳過（序列不足）"], []
    rf = T.run_fold(Xtr, ytr, Xte, yte, dof_ns, f"loro {r}",
                    verbose=False, arch=_G["arch"])
    bundle = _G["out"] / f"{r}_model.pt"
    _bundle_from_fold(rf, d, _G["arch"], bundle)

    # 留出受試者時要對他的**每一份**錄音各跑一次串流評估
    targets = (sorted(np.unique(src[te_w]))
               if _G["scheme"] in ("subject", "session") else [r])
    got = []
    for tgt in targets:
        csv = csv_of.get(tgt)
        if csv is None:
            out_lines.append(f"    ⚠️ 找不到 {tgt} 的原始 CSV，跳過")
            continue
        # ★ 評分時的遮蔽必須與建資料集時**完全一致**，否則會出現
        #   「訓練時丟掉、測試時卻拿來算」的段落，數字沒有意義。
        df = PP.predict_recording(bundle, csv, votes=5, conf=0.6,
                                  use_badspans=not _G["no_badspans"])
        sc_ = PP.score(df, dofs, use_stable=True)
        df.to_csv(_G["out"] / f"{tgt}_predictions.csv", index=False)
        env = P.rms_envelope(P.filter_causal(P.load_recording(csv, trim=False)))
        # ⚠️ 標題一律用英文 —— 環境沒有 CJK 字型，中文會變成豆腐方塊
        tag = {"subject": "LOSO held-out subject",
               "session": "held-out session"}.get(_G["scheme"], "held-out recording")
        PP.plot_overview(csv, df, env, dofs, _G["out"] / f"{tgt}_overview.png",
                         f"{tgt}   {tag}={r}   protocol v{Q.protocol_version(csv)}"
                         f"   exact={sc_['exact']:.1%}")
        rec = {"held_out": r, "recording": tgt,
               "protocol": Q.protocol_version(csv),
               "n_windows": int(sc_["n"]), "exact": sc_["exact"]}
        for dn in dofs:
            rec[dn] = sc_[dn]
            rec[dn + "_bal"] = sc_[dn + "_bal"]
        got.append(rec)
        per = "  ".join(f"{dn}={sc_[dn]:.1%}" for dn in dofs)
        out_lines.append(f"    {tgt[-14:]} → 全對={sc_['exact']:.1%}   {per}")
    return out_lines, got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/loro"))
    ap.add_argument("--arch", default="tiny")
    ap.add_argument("--force", action="store_true", help="重建資料集快取")
    ap.add_argument("--include", type=Path, default=None,
                    help="錄音清單檔（一行一個 stem）。省略 = 收 data-root 底下全部")
    ap.add_argument("--no-badspans", action="store_true",
                    help="忽略 _badspans.txt（做 A/B 比較的基準組用）")
    ap.add_argument("--zero-dead-channels", action="store_true",
                    help="套用 _deadch.txt 把整條壞通道置零（比較用，預設關）")
    ap.add_argument("--no-hampel", action="store_true",
                    help="拿掉因果 Hampel（佔即時計算 77%%）。⚠️ 這會改變輸入分布，"
                         "所以是**重新訓練**一組模型，不能配舊模型用。"
                         "設定會寫進 bundle，推論時自動照著建濾波器。")
    ap.add_argument("--jobs", type=int, default=1,
                    help="同時跑幾折。★ 瓶頸是**串流重播**（逐視窗因果濾波，"
                         "佔 77% 的時間、CPU-bound、單執行緒），不是訓練 "
                         "——訓練早就在 GPU 上。所以加速要靠開多個 process，"
                         "不是靠顯卡。建議設成核心數。")
    ap.add_argument("--scheme", choices=("recording", "subject", "session"),
                    default="recording",
                    help="recording=留一份錄音（跨 session）；"
                         "subject=留一位受試者（★ 真正的 LOSO，最難也最重要）；"
                         "session=留一次**貼片場次**（同一人同一天的錄音視為一組）")
    a = ap.parse_args()

    C.DATA_ROOT = a.data_root
    C.RESULTS_ROOT = a.out
    a.out.mkdir(parents=True, exist_ok=True)

    from semg.v2 import dataset as D
    from semg.v2 import train as T
    from semg.v2 import predict_and_plot as PP
    from semg.v2 import cues as Q
    from semg.v2 import preprocess as P
    D.CACHE = a.out / "dataset.npz"

    d = D.build(force=a.force, verbose=True, include=a.include,
                use_badspans=not a.no_badspans,
                zero_dead_channels=a.zero_dead_channels,
                use_hampel=not a.no_hampel)
    X, y, seg, src = d["X"], d["y_dof"], d["segment"], d["src_file"]
    dof_ns = [int(v) for v in d["dof_n_states"]]
    dofs = [str(v) for v in d["dof_names"]]
    subj = d["subject"]
    if a.scheme == "session":
        # ★ 留一「貼片場次」：同一位受試者、同一天的錄音全部一起留出。
        #
        #   為什麼需要這個切法：`recording` 會**高估**。同一次貼片錄的兩份錄音
        #   共用電極位置、阻抗、皮膚狀態 —— 留一份出來時，它的「同場次雙胞胎」
        #   還在訓練集裡，模型只要認出那組電極的長相就能答對。
        #
        #   實測（2026-08-23，V5 的 7 份）：gary 的四份剛好是兩對相隔 11~14 分鐘
        #   的同場次錄音，recording 切法給 92.4~96.2%；而舊資料裡
        #   14-56-09 是**唯一沒有同場次雙胞胎**的一份，就掉到 67.9%。
        #
        #   部署時面對的永遠是「今天剛貼上去的電極」，所以這個數字才是
        #   誠實的期望值。`recording` 保留下來只為了跟舊結果對比。
        ses = np.array([f"{sb}_{sf.split('_')[2]}"
                        for sb, sf in zip(subj, src)])
        folds = [(s_, ses != s_, ses == s_) for s_ in sorted(np.unique(ses))]
        print(f"\n留一貼片場次：{len(folds)} 個場次、{len(X):,} 個視窗")
        for s_, _, te in folds:
            print(f"   {s_}: {te.sum():,} 視窗 / "
                  f"{len(np.unique(src[te]))} 份錄音")
        print()
    elif a.scheme == "subject":
        # ★ LOSO：一次留一位受試者的**全部**錄音。這是唯一能回答
        #   「換人還能不能用」的切法 —— 留一份錄音時，同一人的其他錄音
        #   仍在訓練集裡，模型可以認出「這個人的肌電長相」。
        folds = [(s_, subj != s_, subj == s_) for s_ in sorted(np.unique(subj))]
        print(f"\nLOSO：{len(folds)} 位受試者、{len(X):,} 個視窗")
        for s_, _, te in folds:
            print(f"   {s_}: {te.sum():,} 視窗 / "
                  f"{len(np.unique(src[te]))} 份錄音")
        print()
    else:
        folds = [(r, src != r, src == r) for r in sorted(np.unique(src))]
        print(f"\n{len(folds)} 份錄音、{len(X):,} 個視窗\n")

    # 錄音檔的實際路徑（畫圖與串流推論要用原始 CSV）
    csv_of = {p.stem: p for p in a.data_root.rglob("*.csv")
              if Q.cue_path_for(p) is not None}

    _G.update(X=X, y=y, seg=seg, src=src, d=d, dofs=dofs, dof_ns=dof_ns,
              folds=folds, csv_of=csv_of, out=a.out, arch=a.arch,
              scheme=a.scheme, no_badspans=a.no_badspans)


    jobs = [(i, r, tr_w, te_w) for i, (r, tr_w, te_w) in enumerate(folds, 1)]
    rows = []
    if a.jobs <= 1:
        for job in jobs:
            lines, got = _do_fold(job)
            print("\n".join(lines), flush=True)
            rows += got
    else:
        # ⚠️ 每個 worker 限制成 1 個 torch thread —— 否則 N 個 process 各開
        #    C.CPU_THREADS 條會超訂 CPU，反而比序列還慢。
        import torch as _t
        _t.set_num_threads(1)
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        n = min(a.jobs, len(jobs))
        print(f"平行跑 {n} 折（瓶頸是 CPU 串流重播，不是 GPU 訓練）\n", flush=True)
        # ⚠️ **fork 之前絕對不能在父程序初始化 CUDA** —— 子程序會繼承到一個
        #    壞掉的 context，症狀通常是 "CUDA error: initialization error"
        #    或更糟：靜靜地退回 CPU。這裡明確擋住，因為未來有人在 main()
        #    早一點加一行 torch.cuda.* 就會踩到，而且不一定會報錯。
        if _t.cuda.is_initialized():
            raise SystemExit(
                "父程序已經初始化 CUDA，不能用 fork 平行化。\n"
                "   把 CUDA 相關呼叫移到 do_fold 裡面（子程序），或改用 --jobs 1。")
        ctx = mp.get_context("fork")     # fork 才能共用已載入的 X/y/seg，不必重新序列化
        with ProcessPoolExecutor(max_workers=n, mp_context=ctx) as ex:
            for lines, got in ex.map(_do_fold, jobs):
                print("\n".join(lines), flush=True)
                rows += got

    if not rows:
        return
    res = pd.DataFrame(rows)
    res.to_csv(a.out / "summary.csv", index=False)
    print("\n" + "=" * 78)
    print("留一份錄音 —— 每份都用其餘錄音訓練的模型預測（串流 + 投票）")
    print("=" * 78)
    print(res.to_string(index=False,
                        formatters={c: "{:.1%}".format for c in res.columns
                                    if res[c].dtype == float}))
    print(f"\n平均全對率 {res.exact.mean():.1%} ± {res.exact.std():.1%}")
    for dn in dofs:
        print(f"  {dn:9s} 原始 {res[dn].mean():.1%}   平衡 {res[dn + '_bal'].mean():.1%}")


if __name__ == "__main__":
    main()
