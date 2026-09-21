"""誠實評估：逐自由度混淆矩陣、訓練曲線、特徵重要度。

讀 train.py 產出的 train_results.csv / confusion_matrices.json，
再加上兩件 train.py 沒做的事：
  1. 混淆矩陣視覺化（看錯在哪，不只看總分）
  2. permutation 特徵重要度（哪些特徵真的有用 —— 對應 Ari 2023 的 MRMR 選特徵想法）

多輸出版本：每個自由度各一張混淆矩陣。
圖用英文標籤（系統無 CJK 字型，v1 踩過這個坑）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report

from semg.v2 import config as C
from semg.v2 import dataset as D
from semg.v2.model import TinyMultiHead


def plot_confusions(dof_names: list[str], dof_states: dict, loso: dict, out: Path):
    """每個 held-out 受試者一列、每個自由度一欄。"""
    subs = list(loso)
    if not subs:
        return
    n_row, n_col = len(subs), len(dof_names)
    fig, axes = plt.subplots(n_row, n_col, figsize=(3.4 * n_col, 3.2 * n_row),
                             squeeze=False)
    im = None
    for r, sub in enumerate(subs):
        for c, dn in enumerate(dof_names):
            ax = axes[r][c]
            cm = np.asarray(loso[sub][dn], dtype=float)
            states = dof_states[dn]
            norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(states)), states, rotation=45, ha="right",
                          fontsize=8)
            ax.set_yticks(range(len(states)), states, fontsize=8)
            acc = np.trace(cm) / max(cm.sum(), 1)
            ax.set_title(f"{sub} / {dn}\nacc = {acc:.1%}", fontsize=9)
            if c == 0:
                ax.set_ylabel("true", fontsize=8)
            if r == n_row - 1:
                ax.set_xlabel("predicted", fontsize=8)
            for i in range(len(states)):
                for j in range(len(states)):
                    ax.text(j, i, f"{norm[i, j]:.0%}", ha="center", va="center",
                            fontsize=7,
                            color="white" if norm[i, j] > 0.5 else "black")
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015)
    fig.suptitle("LOSO confusion matrices per degree of freedom (row-normalised)")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def plot_curves(history: dict, out: Path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for subj, h in history.items():
        ep = [r["epoch"] for r in h]
        ax[0].plot(ep, [r["val_loss"] for r in h], label=f"held-out {subj}")
        ax[1].plot(ep, [r.get("val_exact", np.nan) for r in h],
                   label=f"held-out {subj}")
    ax[0].set(xlabel="epoch", ylabel="val loss (sum over heads)",
              title="validation loss")
    ax[1].set(xlabel="epoch", ylabel="val exact-match",
              title="all DOFs correct simultaneously")
    for a in ax:
        a.grid(alpha=.3)
        a.legend(fontsize=8)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def permutation_importance(bundle: Path, n_rep: int = 3, max_n: int = 4000,
                           seed: int = C.SEED):
    """打亂單一特徵欄，看**全對率**掉多少。掉得多 = 該特徵重要。

    ⚠️ 這是在**部署模型**（全資料訓練）上做的，只能當「模型依賴什麼」的診斷，
    不能當泛化能力證據。
    """
    ck = torch.load(bundle, map_location="cpu", weights_only=False)
    dof_names = list(ck["dof_names"])
    dof_n_states = [int(v) for v in ck["dof_n_states"]]
    fnames = list(ck["feature_names"])
    seq_len = int(ck["seq_len"])
    mean = np.asarray(ck["scaler_mean"], np.float32)
    std = np.asarray(ck["scaler_scale"], np.float32)

    model = TinyMultiHead(len(mean), tuple(dof_n_states))
    model.load_state_dict(ck["state_dict"])
    model.eval()

    from semg.v2.train import make_sequences
    d = D.build(verbose=False)
    Xs, ys = make_sequences(d["X"], d["y_dof"], d["trial"], seq_len)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Xs))[:min(max_n, len(Xs))]
    Xs, ys = Xs[idx], ys[idx]
    Z = ((Xs - mean) / (std + 1e-8)).astype(np.float32)

    def exact_of(A: np.ndarray) -> float:
        outs = []
        with torch.inference_mode():
            for s in range(0, len(A), 512):
                lg = model(torch.from_numpy(A[s:s + 512]))
                outs.append(torch.stack([l.argmax(1) for l in lg], 1).numpy())
        return float((np.concatenate(outs) == ys).all(axis=1).mean())

    base = exact_of(Z)
    rows = []
    for j, fn in enumerate(fnames):
        drops = []
        for _ in range(n_rep):
            Zp = Z.copy()
            flat = Zp[:, :, j].ravel().copy()
            rng.shuffle(flat)
            Zp[:, :, j] = flat.reshape(Zp.shape[0], Zp.shape[1])
            drops.append(base - exact_of(Zp))
        rows.append({"feature": fn, "drop": float(np.mean(drops)),
                     "std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("drop", ascending=False), base


def plot_importance(imp: pd.DataFrame, base: float, out: Path, top: int = 25):
    t = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(t) + 1.2))
    ax.barh(t.feature, t.drop, xerr=t["std"], color="#4472c4")
    ax.set_xlabel("exact-match drop when this feature is shuffled")
    ax.set_title(f"Top {top} features (baseline exact = {base:.1%})")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x", alpha=.3)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def main():
    ap = argparse.ArgumentParser(description="v2 多輸出評估報告")
    ap.add_argument("--importance", action="store_true",
                    help="跑 permutation 特徵重要度（120 特徵 × 3 次，較慢）")
    a = ap.parse_args()

    R = C.RESULTS_ROOT
    cmj = json.loads((R / "confusion_matrices.json").read_text())
    dof_names, dof_states, loso = cmj["dof_names"], cmj["dof_states"], cmj["loso"]
    res = pd.read_csv(R / "train_results.csv")

    print("=" * 78)
    print("v2 多輸出評估報告（全對率 = 所有自由度同時正確）")
    print("=" * 78)
    for scheme in ("trial_kfold", "within_session", "within_subject", "LOSO"):
        g = res[res.scheme == scheme]
        if not len(g):
            continue
        print(f"\n{scheme}：全對 {g.exact.mean():.1%} ± {g.exact.std():.1%}  "
              f"macroF1 {g.macro_f1.mean():.3f}")
        for dn in dof_names:
            col = f"acc_{dn}"
            if col in g:
                print(f"    {C.DOF_LABELS_ZH.get(dn, dn):4s} {g[col].mean():.1%}")

    print("\nLOSO 逐自由度、逐狀態表現（三折合併）：")
    for dn in dof_names:
        yt, yp = [], []
        for sub in loso:
            cm = np.asarray(loso[sub][dn])
            for i, row in enumerate(cm):
                for j, v in enumerate(row):
                    yt += [i] * int(v)
                    yp += [j] * int(v)
        if not yt:
            continue
        print(f"\n  {C.DOF_LABELS_ZH.get(dn, dn)}")
        print(classification_report(yt, yp, target_names=dof_states[dn],
                                    zero_division=0, digits=3))

    print("圖表：")
    plot_confusions(dof_names, dof_states, loso, R / "confusion_loso.png")
    hp = R / "history.json"
    if hp.exists():
        plot_curves(json.loads(hp.read_text()), R / "training_curves.png")

    if a.importance:
        bundle = R / "dof_model.pt"
        if not bundle.exists():
            print("  找不到 dof_model.pt，略過特徵重要度")
        else:
            print("\n跑 permutation 特徵重要度…")
            imp, base = permutation_importance(bundle)
            imp.to_csv(R / "feature_importance.csv", index=False)
            plot_importance(imp, base, R / "feature_importance.png")
            print(f"\n最有用的 12 個特徵（baseline 全對 {base:.1%}）：")
            for _, r in imp.head(12).iterrows():
                print(f"    {r.feature:34s} 掉 {r.drop:+.2%}")


if __name__ == "__main__":
    main()
