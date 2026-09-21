"""訓練 + LOSO 交叉驗證，誠實評估。

依 Zhao et al. 的做法**同時報兩個數字**：
    within-subject（同一人資料切分）—— 樂觀，用來確認模型學得動
    LOSO（留一人測試）             —— 這才是真正的泛化能力

SoftPINCH 只報一個數字，我們花了很多力氣才搞清楚它對應哪個設定。這裡不重複那個錯誤。

⚠️ 我們只有 3 位受試者 → LOSO 只有 3 折，信賴區間很寬。報告時必須寫清楚。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from semg.v2 import config as C
from semg.v2 import dataset as D
from semg.v2.model import build_model, count_params, describe

SEQ_LEN = C.SEQ_LEN     # 定義在 config.py（validate() 需要它，config 不能 import train）


def make_sequences(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seq_len: int = SEQ_LEN):
    """把逐視窗特徵堆成序列。**只在同一個 group 內堆疊**。

    ★ group 必須是 **segment** 鍵（`<檔名>#<rep>#<區間>`），不是 trial 鍵。

    一個 trial 含兩段不連續的有標籤區間（`go` 維持段與 `release` 放鬆段），
    中間隔著被丟棄的 `return` 段。用 trial 當 group 的話，這個函式只看
    陣列相鄰、不看時間，會把 go 的尾巴與 release 的開頭黏成一條序列：

      - 那兩個視窗實際相隔 1.5s 以上，卻被當成相隔 50ms → **偽造的時間連續性**
      - 標籤取序列**最後一個視窗** → 標成 rest 卻含最多 seq_len-1 個姿勢視窗

    實測（合成資料）未修正前有 **18.0%** 的序列是這樣。用 segment 當 group
    就結構性排除，跨檔案與跨區間兩種情況一次解決。

    ⚠️ 交叉驗證的分組仍用 **trial** 鍵（`dataset.trial_group_splits`）——
       go 與 rest 時間相鄰、高度相關，不該被拆到不同折。
    """
    Xs, ys = [], []
    for f in np.unique(groups):
        idx = np.where(groups == f)[0]
        if len(idx) < seq_len:
            continue
        Xf, yf = X[idx], y[idx]
        for i in range(len(idx) - seq_len + 1):
            Xs.append(Xf[i:i + seq_len])
            ys.append(yf[i + seq_len - 1])      # 以序列最後一個視窗的標籤為準
    if not Xs:
        return (np.zeros((0, seq_len, X.shape[1]), np.float32),
                np.zeros((0, y.shape[1]), np.int64))
    return np.stack(Xs).astype(np.float32), np.stack(ys).astype(np.int64)


def run_fold(Xtr, ytr, Xte, yte, dof_n_states, tag: str, verbose=True,
             arch: str = "tiny", seed: int | None = None):
    """訓練並評估一折。

    ⚠️ `seed` 必須從**這裡**設定，不能由呼叫端先 `torch.manual_seed()` ——
       這個函式開頭就會重設種子，呼叫端設的會被蓋掉。
       實際踩到：架構多種子實驗跑出 tiny 五個種子結果**完全相同**
       （標準差 0.0），因為每次都被重設成 C.SEED。
    """
    sd = C.SEED if seed is None else int(seed)
    torch.manual_seed(sd)
    np.random.seed(sd)

    # 特徵標準化：統計量**只從訓練集算**（避免洩漏）
    sc = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[-1]))
    tr = sc.transform(Xtr.reshape(-1, Xtr.shape[-1])).reshape(Xtr.shape).astype(np.float32)
    te = sc.transform(Xte.reshape(-1, Xte.shape[-1])).reshape(Xte.shape).astype(np.float32)

    # 訓練集裡再切 15% 當驗證集做 early stopping
    n = len(tr)
    perm = np.random.permutation(n)          # 受上面 np.random.seed(sd) 控制
    n_val = max(int(0.15 * n), 1)
    vi, ti = perm[:n_val], perm[n_val:]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(arch, Xtr.shape[-1], tuple(dof_n_states)).to(dev)

    # 每個頭各自算類別權重。rest 一定是多數（休息段比出力段長），
    # 不加權的話所有頭都會學成「永遠輸出 rest」。
    losses = []
    for j, ns in enumerate(dof_n_states):
        # ★ -1 = 該視窗對這個頭**未定義**（V5 的 `return` 段：三個姿勢頭沒有意義，
        #   只有 moving 頭有）。類別權重要先把 -1 濾掉，否則 bincount 直接爆。
        col = ytr[ti][:, j]
        cnt = np.bincount(col[col >= 0], minlength=ns).astype(np.float64)
        w = torch.tensor((cnt.sum() / (cnt + 1e-9)) / ns,
                         dtype=torch.float32, device=dev)
        # ignore_index=-1 讓那些視窗**不產生梯度** —— 比「隨便給個標籤」正確得多，
        # 也比「把整個視窗丟掉」有效（丟掉的話 moving 頭就沒有訓練資料了）。
        losses.append(nn.CrossEntropyLoss(weight=w, ignore_index=-1))

    def total_loss(logits, yb):
        """多頭損失 = 各頭 cross-entropy 之和（等權）。

        等權是刻意的：各自由度在控制上同等重要，
        沒有理由讓某個關節的錯誤比較不重要。
        """
        return sum(lf(lg, yb[:, j]) for j, (lf, lg) in enumerate(zip(losses, logits)))

    opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)

    def loader(Xa, ya, shuffle):
        return DataLoader(TensorDataset(torch.from_numpy(Xa), torch.from_numpy(ya)),
                          batch_size=C.BATCH_SIZE, shuffle=shuffle)

    dl_tr = loader(tr[ti], ytr[ti], True)
    dl_va = loader(tr[vi], ytr[vi], False)

    best_vl, best_state, patience, hist = np.inf, None, 0, []
    for ep in range(C.MAX_EPOCHS):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            total_loss(model(xb), yb).backward()
            opt.step()

        model.eval()
        vl, vc, vn = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(dev), yb.to(dev)
                lg = model(xb)
                vl += total_loss(lg, yb).item() * len(yb)
                pred = torch.stack([l.argmax(1) for l in lg], dim=1)
                vc += (pred == yb).all(dim=1).sum().item()      # 全部 DOF 都對才算對
                vn += len(yb)
        vl /= max(vn, 1)
        hist.append({"epoch": ep, "val_loss": vl, "val_exact": vc / max(vn, 1)})
        if vl < best_vl - 1e-5:
            best_vl, patience = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= C.EARLY_STOP_PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader(te, yte, False):
            lg = model(xb.to(dev))
            preds.append(torch.stack([l.argmax(1) for l in lg], dim=1).cpu().numpy())
    pred = np.concatenate(preds) if preds else np.zeros((0, len(dof_n_states)), int)

    # 三個層次的指標，缺一不可：
    #   exact  —— 所有自由度**全部**正確（最嚴格，也是控制上真正要的）
    #   per-dof—— 每個自由度各自的正確率（看是哪個關節拖累）
    #   macroF1—— 每個自由度的 macro F1（rest 佔多數，只看 acc 會被灌水）
    # ★ -1 = 該視窗對這個頭未定義（V5 的 return 段）。評估時必須**逐頭**跳過，
    #   不能把它算成錯 —— 那會憑空製造一個永遠達不到的上限。
    #   `exact` 只在「全部頭都有定義」的視窗上算，否則語意不清。
    exact = np.nan
    if len(pred):
        full = (yte >= 0).all(axis=1)
        exact = float((pred[full] == yte[full]).all(axis=1).mean()) if full.any() else np.nan
    per_dof, per_f1 = {}, {}
    for j, d in enumerate(C.DOFS[:len(dof_n_states)]):
        m = yte[:, j] >= 0 if len(pred) else np.zeros(0, bool)
        per_dof[d] = float((pred[m, j] == yte[m, j]).mean()) if m.any() else np.nan
        per_f1[d] = (float(f1_score(yte[m, j], pred[m, j], average="macro",
                                    zero_division=0)) if m.any() else np.nan)
    mean_f1 = float(np.mean(list(per_f1.values()))) if per_f1 else np.nan
    if verbose:
        body = "  ".join(f"{C.DOF_LABELS_ZH[d]}={per_dof[d]:.0%}" for d in per_dof)
        print(f"    {tag}: 全對={exact:.1%}  macroF1={mean_f1:.3f}  [{body}]  "
              f"({ep+1} epochs, val_loss={best_vl:.4f})", flush=True)
    return {"acc": exact, "exact": exact, "macro_f1": mean_f1,
            "per_dof": per_dof, "per_dof_f1": per_f1,
            "pred": pred, "true": yte,
            "epochs": ep + 1, "history": hist, "model": model, "scaler": sc}


def main(arch: str = "tiny"):
    C.RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    d = D.build(verbose=False)
    X = d["X"]
    y = d["y_dof"]                              # ★ (N, N_DOF) 多輸出標籤
    subj, files = d["subject"], d["src_file"]
    trial = d["trial"]        # 交叉驗證分組（go 與 rest 同屬一個 trial）
    segment = d["segment"]    # 序列堆疊分組（go 與 rest 是不同區間，絕不可黏起來）
    dof_names = [str(v) for v in d["dof_names"]]
    dof_n_states = [int(v) for v in d["dof_n_states"]]
    has_cue = bool((d["label_source"] == "cue").any())

    print(f"架構：{arch}")
    print(f"資料集：{X.shape[0]:,} 視窗 × {X.shape[1]} 特徵")
    print(f"多輸出：{len(dof_names)} 個自由度 "
          + "  ".join(f"{C.DOF_LABELS_ZH[dn]}({ns}態)"
                      for dn, ns in zip(dof_names, dof_n_states)))
    print(f"序列長度 SEQ_LEN={SEQ_LEN}（涵蓋 {(SEQ_LEN-1)*C.STEP_SEC + C.WIN_SEC:.2f}s 上下文）")
    # 「全部猜 rest」的基準線 —— 多輸出一定要報這個，否則 90% 看起來很厲害
    rest_base = float((y == 0).all(axis=1).mean())
    print(f"基準線：全部猜 rest = {rest_base:.1%} 全對率\n")

    def _date(f):
        m = re.search(r"_(\d{8})_", str(f))
        return m.group(1) if m else "unknown"
    sessions = np.array([f"{a}_{_date(b)}" for a, b in zip(subj, files)])

    results, all_hist, cms = [], {}, {}

    def do(tr_i, te_i, scheme, name):
        Xtr, ytr = make_sequences(X[tr_i], y[tr_i], segment[tr_i])
        Xte, yte = make_sequences(X[te_i], y[te_i], segment[te_i])
        if not len(Xtr) or not len(Xte):
            return None
        r = run_fold(Xtr, ytr, Xte, yte, dof_n_states, name, arch=arch)
        row = {"scheme": scheme, "split": name, "exact": r["exact"],
               "macro_f1": r["macro_f1"], "epochs": r["epochs"],
               "n_train": len(Xtr), "n_test": len(Xte)}
        row.update({f"acc_{k}": v for k, v in r["per_dof"].items()})
        row.update({f"f1_{k}": v for k, v in r["per_dof_f1"].items()})
        results.append(row)
        return r

    # ---------- (1) trial 分組 5 折：同一次出力不跨折 ----------
    if has_cue:
        print("─── trial 分組 5 折（同 session；無洩漏的主要指標）───")
        for part, tr_i, te_i in D.trial_group_splits(d, n_folds=5):
            do(tr_i, te_i, "trial_kfold", f"{len(part)} trials")

    # ---------- (2) within-session：同一天，時間切分 ----------
    print("\n─── within-session（同一天，前 70% 訓練／後 30% 測試）───")
    GAP = 2 * SEQ_LEN
    for sn in sorted(set(sessions)):
        tr_i, te_i = [], []
        for f in sorted(np.unique(files[sessions == sn])):
            idx = np.where(files == f)[0]
            k = int(len(idx) * 0.7)
            if k < 50 or len(idx) - k - GAP < 50:
                continue
            tr_i += list(idx[:k]); te_i += list(idx[k + GAP:])
        if tr_i and te_i:
            do(np.asarray(tr_i), np.asarray(te_i), "within_session", sn)

    # ---------- (3) within-subject：同一人，留最後一次錄音 ----------
    # 新協定下每個檔案都含全部動作，所以「留一檔」不會再抽走整個類別
    # （這正是舊協定造成三人都 0.0% 的那個 bug，見 docs/v1-problems-and-v2-fixes.md 4.3）
    print("\n─── within-subject（同一人，留最後一次錄音；跨 session）───")
    for s in C.SUBJECTS if len(set(subj) & set(C.SUBJECTS)) else sorted(set(subj)):
        m = subj == s
        fs = np.unique(files[m])
        if len(fs) < 2:
            print(f"    {s}: 只有 {len(fs)} 次錄音，無法跨錄音評估 → 略過")
            continue
        te_f = sorted(fs)[-1]
        do(np.where(m & (files != te_f))[0], np.where(m & (files == te_f))[0],
           "within_subject", s)

    # ---------- (4) LOSO ----------
    print("\n─── LOSO（留一位受試者；真正的泛化能力）───")
    for s, tr_i, te_i in D.loso_splits(d):
        r = do(tr_i, te_i, "LOSO", s)
        if r is not None:
            all_hist[s] = r["history"]
            cms[s] = {dn: confusion_matrix(r["true"][:, j], r["pred"][:, j],
                                           labels=range(dof_n_states[j])).tolist()
                      for j, dn in enumerate(dof_names)}

    # ---------- (5) 最終部署模型：全資料訓練 + TorchScript ----------
    print("\n─── 最終部署模型（全部資料訓練，準確率不可用於報告）───")
    Xa, ya = make_sequences(X, y, segment)
    hold = np.random.default_rng(C.SEED).permutation(len(Xa))[:max(1, len(Xa) // 10)]
    rf = run_fold(Xa, ya, Xa[hold], ya[hold], dof_n_states, "final (in-sample)",
                  arch=arch)
    model = rf["model"].cpu().eval()

    suffix = "" if arch == "tiny" else f"_{arch}"
    bundle = C.RESULTS_ROOT / f"dof_model{suffix}.pt"
    torch.save({"state_dict": model.state_dict(),
                "arch": arch,
                "dof_names": dof_names, "dof_n_states": dof_n_states,
                "seq_len": SEQ_LEN, "n_features": int(X.shape[1]),
                "feature_names": [str(v) for v in d["feature_names"]],
                "channel_names": [str(v) for v in d["channel_names"]],
                "scaler_mean": rf["scaler"].mean_.astype(np.float32),
                "scaler_scale": rf["scaler"].scale_.astype(np.float32),
                "params": count_params(model),
                "config": {"win_sec": C.WIN_SEC, "step_sec": C.STEP_SEC, "fs": C.FS,
                           "width": C.MODEL_WIDTH, "blocks": C.MODEL_BLOCKS}},
               bundle)
    # TorchScript：CPU 上實測 0.52 → 0.28 ms（1.8×）。即時推論載這個。
    ts_path = C.RESULTS_ROOT / "dof_model_scripted.pt"
    try:
        torch.jit.optimize_for_inference(torch.jit.script(model)).save(str(ts_path))
        print(f"    TorchScript → {ts_path.name}")
    except Exception as e:                        # noqa: BLE001
        print(f"    ⚠️ TorchScript 失敗（不影響訓練，即時推論會退回 eager）：{e}")
    print(f"    已存 → {bundle.name}  ({count_params(model):,} 參數)")

    # ---------- 報告 ----------
    res = pd.DataFrame(results)
    res.to_csv(C.RESULTS_ROOT / "train_results.csv", index=False)
    # ★ 一律指定 encoding="utf-8"：`write_text()` 不給編碼時用系統 locale，
    #   Windows(zh-TW) 會寫成 cp950，含中文的 JSON 在 Linux 端讀不開。
    #   （record_session 的 meta.json 就是這樣壞掉過。）
    (C.RESULTS_ROOT / "confusion_matrices.json").write_text(
        json.dumps({"dof_names": dof_names,
                    "dof_states": {dn: list(C.DOF_STATES[dn]) for dn in dof_names},
                    "loso": cms}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    (C.RESULTS_ROOT / "history.json").write_text(
        json.dumps(all_hist, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"結果彙總（全對率 = {len(dof_names)} 個自由度同時正確）")
    print("=" * 78)
    for scheme in ("trial_kfold", "within_session", "within_subject", "LOSO"):
        g = res[res.scheme == scheme] if len(res) else res
        if not len(g):
            continue
        print(f"\n{scheme}")
        for _, r in g.iterrows():
            body = "  ".join(f"{C.DOF_LABELS_ZH[dn]}={r[f'acc_{dn}']:.0%}"
                             for dn in dof_names)
            print(f"  {str(r.split)[:16]:18s} 全對={r.exact:6.1%}  "
                  f"macroF1={r.macro_f1:.3f}  [{body}]")
        print(f"  {'平均':18s} 全對={g.exact.mean():6.1%} ± {g.exact.std():.1%}  "
              f"macroF1={g.macro_f1.mean():.3f}")

    print(f"\n基準線（全部猜 rest）= {rest_base:.1%}")
    print("\nLOSO 逐自由度混淆矩陣（列=真實, 行=預測）：")
    for s, per_dof in cms.items():
        print(f"\n  held-out = {s}")
        for dn, cm in per_dof.items():
            states = C.DOF_STATES[dn]
            print(f"    {C.DOF_LABELS_ZH[dn]}  " + "".join(f"{st:>9s}" for st in states))
            for i, row in enumerate(cm):
                tot = sum(row) or 1
                print(f"      {states[i]:>6s}" + "".join(f"{v/tot:>9.1%}" for v in row))
    print(f"\n[完成] {C.RESULTS_ROOT}")


if __name__ == "__main__":
    import argparse
    from semg.v2.model import ARCHITECTURES
    ap = argparse.ArgumentParser(description="訓練 + 多層次評估")
    ap.add_argument("--arch", default="tiny", choices=list(ARCHITECTURES),
                    help="tiny=輕量卷積多頭（預設）；zhao=論文 baseline "
                         "CNN+BiLSTM+Attention（改成多頭）")
    main(**vars(ap.parse_args()))
