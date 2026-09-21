"""前處理與架構的消融實驗：每個設計決策**單獨**驗證一次。

為什麼要有這支：`docs/pipeline-methodology.md` 裡的每個決策都有理由，
但「有理由」不等於「有幫助」。這裡把每一項拆開量，讓文件裡的每個
「我們用 X 因為 Y」都能附上一個數字，而不是只有論證。

★ 評估用**批次**路徑（直接餵序列陣列），不是串流重播。
  原因是速度：串流只比即時快 2.5 倍，跑一次完整消融要好幾小時。
  代價是數字會比串流樂觀（少了投票延遲與濾波狀態接續的差異），
  所以**消融結果只能看相對差異，不能當部署預期**。
  絕對數字一律以 `leave_one_recording_out.py`（走串流）為準。

分組一律用 trial 或 subject，絕不用隨機切窗 ——
V1 實測隨機切窗會給出 99.9% 的假象（相鄰視窗有 90% 重疊）。
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


def eval_split(X, y, groups, tr_mask, te_mask, arch="tiny", seed=C.SEED,
               seq_len: int | None = None) -> dict:
    """訓練一次、評估一次。回傳逐 DOF 與全對率（含平衡準確率）。

    ⚠️ `seq_len` **必須明確傳入**，不能靠改 `train.SEQ_LEN` 模組變數 ——
       `make_sequences(..., seq_len=SEQ_LEN)` 的預設值在**模組定義時**就綁定了，
       事後改模組變數對它完全沒有影響。
       實際踩到：SEQ_LEN 消融跑出來每一組數字都跟基線一模一樣。
    """
    from semg.v2 import train as T
    sl = seq_len if seq_len is not None else T.SEQ_LEN
    Xtr, ytr = T.make_sequences(X[tr_mask], y[tr_mask], groups[tr_mask], seq_len=sl)
    Xte, yte = T.make_sequences(X[te_mask], y[te_mask], groups[te_mask], seq_len=sl)
    if not len(Xtr) or not len(Xte):
        return {}
    rf = T.run_fold(Xtr, ytr, Xte, yte, list(C.DOF_N_STATES), "ablation",
                    verbose=False, arch=arch, seed=seed)
    p, t = rf["pred"], rf["true"]
    out = {"n_train": int(len(Xtr)), "n_test": int(len(Xte))}
    ok_all = np.ones(len(t), bool)
    for j, dn in enumerate(C.DOFS):
        ok = p[:, j] == t[:, j]
        out[dn] = float(ok.mean())
        rec = [float(ok[t[:, j] == s].mean())
               for s in range(C.DOF_N_STATES[j]) if (t[:, j] == s).any()]
        out[dn + "_bal"] = float(np.mean(rec))
        ok_all &= ok
    out["exact"] = float(ok_all.mean())
    return out


def subject_folds(d) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """LOSO：留一位受試者。**這才是真正的泛化測試。**"""
    subj = d["subject"]
    return [(s, subj != s, subj == s) for s in sorted(np.unique(subj))]


def recording_folds(d) -> list[tuple[str, np.ndarray, np.ndarray]]:
    src = d["src_file"]
    return [(r, src != r, src == r) for r in sorted(np.unique(src))]


def trial_folds(d, k: int = 5) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """trial 分組 k 折 —— 同 session，最樂觀的評估。"""
    tr = d["trial"]
    g = np.unique(tr)
    perm = np.random.default_rng(C.SEED).permutation(len(g))
    out = []
    for i, idx in enumerate(np.array_split(perm, k)):
        te_g = set(g[idx])
        m = np.isin(tr, list(te_g))
        out.append((f"fold{i}", ~m, m))
    return out


# ── 特徵子集：用特徵名前綴判斷屬於哪一類 ───────────────────────────
def feature_mask(kind: str) -> np.ndarray:
    """挑出某一類特徵的欄位遮罩。用來回答「這一類特徵有沒有用」。"""
    from semg.v2 import features as F
    names = list(F.FEATURE_NAMES)
    amp = ("rms_rel", "mav_rel", "p2p_rel", "wamp", "log_rms")
    morph = ("wl_rel", "zc", "ssc", "shape_factor")
    freq = ("mnf", "mdf", "pkf", "bandwidth")
    cep = ("mfcc1", "mfcc3")
    def suf(n):                       # 通道名在前，特徵名在後
        return n.rsplit("_", 1)[-1] if not n.startswith(("logr", "prop", "grp")) else "cross"
    groups = {"amplitude": amp, "morphology": morph, "frequency": freq, "cepstral": cep}
    if kind == "cross":
        return np.array([n.startswith(("logr_", "prop_", "grp_")) for n in names])
    if kind == "per_channel":
        return np.array([not n.startswith(("logr_", "prop_", "grp_")) for n in names])
    keys = groups[kind]
    return np.array([any(n.endswith("_" + k) for k in keys) for n in names])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True, help="dataset.npz")
    ap.add_argument("--out", type=Path, default=Path("results/full_eval"))
    ap.add_argument("--only", default="", help="只跑某一組實驗（逗號分隔）")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    z = np.load(a.dataset, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    X, y, seg = d["X"], d["y_dof"], d["segment"]
    print(f"資料集 {X.shape}，受試者 {sorted(set(d['subject']))}\n")

    schemes = {"loso": subject_folds(d), "loro": recording_folds(d),
               "trial5": trial_folds(d)}
    rows = []

    def run(exp: str, variant: str, scheme: str, **kw):
        Xu = kw.pop("X", X)
        for name, tr, te in schemes[scheme]:
            t0 = time.perf_counter()
            r = eval_split(Xu, y, seg, tr, te, **kw)
            if not r:
                continue
            r.update(exp=exp, variant=variant, scheme=scheme, fold=name,
                     sec=round(time.perf_counter() - t0, 1))
            rows.append(r)
            print(f"  {exp:<16s} {variant:<22s} {scheme:<7s} {name:<30s} "
                  f"全對={r['exact']:.1%}")
        pd.DataFrame(rows).to_csv(a.out / "ablation_raw.csv", index=False)

    only = set(a.only.split(",")) if a.only else None
    def want(e):
        return only is None or e in only

    # ── 1. 架構 ──────────────────────────────────────────────
    if want("arch"):
        print("【1】模型架構")
        for arch in ("tiny", "zhao"):
            for sch in ("loso", "loro"):
                run("arch", arch, sch, arch=arch)

    # ── 2. 序列長度 ──────────────────────────────────────────
    if want("seq"):
        print("\n【2】序列長度 SEQ_LEN")
        for sl in (1, 3, 5, 10, 20):
            run("seq_len", f"SEQ_LEN={sl}", "loso", seq_len=sl)
            run("seq_len", f"SEQ_LEN={sl}", "loro", seq_len=sl)

    # ── 3. 特徵子集 ──────────────────────────────────────────
    if want("feat"):
        print("\n【3】特徵子集")
        subsets = {
            "全部 120 維": np.ones(X.shape[1], bool),
            "只有振幅 (35)": feature_mask("amplitude"),
            "只有形態 (28)": feature_mask("morphology"),
            "只有頻域 (28)": feature_mask("frequency"),
            "只有倒頻譜 (14)": feature_mask("cepstral"),
            "只有跨通道 (15)": feature_mask("cross"),
            "去掉跨通道 (105)": feature_mask("per_channel"),
            "振幅+頻域 (63)": feature_mask("amplitude") | feature_mask("frequency"),
        }
        for nm, m in subsets.items():
            if m.sum() == 0:
                continue
            run("features", f"{nm}", "loso", X=X[:, m])
            run("features", f"{nm}", "loro", X=X[:, m])

    # ── 4. 通道子集 ──────────────────────────────────────────
    if want("chan"):
        print("\n【4】通道子集（回答「近端電極有沒有幫上手部判別」）")
        from semg.v2 import features as F
        names = list(F.FEATURE_NAMES)
        def ch_mask(keep: tuple[str, ...]) -> np.ndarray:
            return np.array([n.startswith(tuple(k + "_" for k in keep))
                             or n.startswith(("logr_", "prop_", "grp_")) for n in names])
        sets = {"全 7 通道": tuple(C.CH_NAMES),
                "只有遠端 3ch": ("wrist_flexor", "index", "thumb"),
                "只有近端 4ch": ("latissimus_dorsi", "deltoid_anterior", "triceps", "biceps")}
        for nm, keep in sets.items():
            m = ch_mask(keep)
            run("channels", nm, "loso", X=X[:, m])
            run("channels", nm, "loro", X=X[:, m])

    if not rows:
        print("沒有跑任何實驗")
        return
    res = pd.DataFrame(rows)
    res.to_csv(a.out / "ablation_raw.csv", index=False)
    piv = res.pivot_table(index=["exp", "variant"], columns="scheme",
                          values="exact", aggfunc="mean")
    piv.to_csv(a.out / "ablation_summary.csv")
    print("\n" + "=" * 78)
    print(piv.to_string(float_format=lambda v: f"{v:.1%}"))


if __name__ == "__main__":
    main()
