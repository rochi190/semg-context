"""訓練**部署用**的最終模型：所有納入的錄音一起訓練，不留測試集。

★ 這支跟 `leave_one_recording_out.py` 的關係要講清楚，否則很容易誤用：

    leave_one_recording_out.py  → 產生**數字**（10 個模型，每個用 9 份訓練、1 份測試）
    train_final.py              → 產生**要部署的那一個模型**（全部資料一起訓練）

  最終模型**沒有自己的準確率**，因為沒有留出資料可以評。
  它誠實的期望值是 LORO 的平均 —— 而且那個平均還是**保守**的，
  因為 LORO 的每個模型都少看了一份錄音。
  ⚠️ 絕對不要拿「訓練集準確率」當成效能數字報出去。

用法：
    PYTHONPATH=src python -m semg.v2.train_final \\
        --data-root V3moving/data --include configs/recordings_v3.txt \\
        --out results/full_eval/models/final_v3.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from semg.v2 import config as C


def save_bundle(model, scaler, d, arch: str, path: Path,
                seq_len: int, scales: np.ndarray | None = None) -> None:
    """存成 `infer.Recognizer` 吃得下的 bundle。

    格式與 `leave_one_recording_out._bundle_from_fold` 相同 ——
    兩邊必須一致，否則離線評估用的模型與部署用的模型會走不同的載入路徑。
    """
    m = model.cpu().eval()
    ck = {"state_dict": m.state_dict(), "arch": arch,
          "dof_names": list(d["dof_names"]),
          "dof_n_states": [int(v) for v in d["dof_n_states"]],
          "seq_len": int(seq_len), "n_features": int(d["X"].shape[1]),
          "feature_names": [str(v) for v in d["feature_names"]],
          "channel_names": [str(v) for v in d["channel_names"]],
          "scaler_mean": scaler.mean_.astype(np.float32),
          "scaler_scale": scaler.scale_.astype(np.float32),
          "use_hampel": bool(d["use_hampel"]) if "use_hampel" in d else C.USE_HAMPEL,
          "feat_version": int(d["feat_version"]) if "feat_version" in d else 1,
          "config": {"win_sec": C.WIN_SEC, "step_sec": C.STEP_SEC, "fs": C.FS}}
    if scales is not None:
        ck["scales"] = np.asarray(scales, dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, path)


def script_model(bundle: Path, n_feat: int, seq_len: int) -> Path | None:
    """另存 TorchScript 版。`Recognizer` 會優先載它（CPU 0.52 → 0.28 ms）。"""
    from semg.v2.model import build_model
    ck = torch.load(bundle, map_location="cpu", weights_only=False)
    m = build_model(ck["arch"], n_feat, tuple(ck["dof_n_states"]))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    out = bundle.with_name(bundle.stem + "_scripted.pt")
    try:
        ts = torch.jit.trace(m, torch.zeros(1, seq_len, n_feat))
        torch.jit.save(ts, str(out))
        return out
    except Exception as e:                       # noqa: BLE001
        # 不讓 TorchScript 失敗擋住部署 —— Recognizer 會自動退回一般載入
        print(f"  ⚠️ TorchScript 匯出失敗（不影響部署，會用一般載入）：{e}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--include", type=Path, default=None)
    ap.add_argument("--out", type=Path,
                    default=Path("results/full_eval/models/final_v3.pt"))
    ap.add_argument("--arch", default="tiny")
    ap.add_argument("--no-badspans", action="store_true")
    ap.add_argument("--zero-dead-channels", action="store_true")
    ap.add_argument("--no-hampel", action="store_true",
                    help="拿掉因果 Hampel（見 leave_one_recording_out 的說明）")
    ap.add_argument("--cache", type=Path, default=None,
                    help="沿用既有的 dataset.npz（省下重算特徵的時間）")
    a = ap.parse_args()

    C.DATA_ROOT = a.data_root
    C.RESULTS_ROOT = a.out.parent
    from semg.v2 import dataset as D
    from semg.v2 import train as T

    if a.cache is not None:
        D.CACHE = a.cache
        d = D.build(force=False, verbose=True)
        print(f"沿用既有資料集 {a.cache}")
    else:
        D.CACHE = a.out.parent / f"{a.out.stem}_dataset.npz"
        d = D.build(force=True, verbose=True, include=a.include,
                    use_badspans=not a.no_badspans,
                    zero_dead_channels=a.zero_dead_channels,
                    use_hampel=not a.no_hampel)

    X, y, seg = d["X"], d["y_dof"], d["segment"]
    recs = sorted(np.unique(d["src_file"]))
    print(f"\n最終模型：{len(recs)} 份錄音、{len(X):,} 個視窗")
    for r in recs:
        print(f"   {r}")

    Xs, ys = T.make_sequences(X, y, seg)
    print(f"\n序列 {Xs.shape}（SEQ_LEN={T.SEQ_LEN}）")

    # ★ 沒有測試集，所以把全部資料都當訓練集餵進去。
    #   run_fold 仍需要一個「測試集」參數 —— 傳同一份進去，
    #   它回報的準確率是**訓練集上的**，只能拿來確認有學到東西，不是效能數字。
    rf = T.run_fold(Xs, ys, Xs, ys, [int(v) for v in d["dof_n_states"]],
                    "final", verbose=True, arch=a.arch)

    save_bundle(rf["model"], rf["scaler"], d, a.arch, a.out, T.SEQ_LEN)
    ts = script_model(a.out, X.shape[1], T.SEQ_LEN)
    meta = {"recordings": recs, "n_windows": int(len(X)),
            "n_sequences": int(len(Xs)), "arch": a.arch,
            "seq_len": int(T.SEQ_LEN), "n_features": int(X.shape[1]),
            "channel_names": [str(v) for v in d["channel_names"]],
            "include": str(a.include), "badspans": not a.no_badspans,
            "zero_dead_channels": a.zero_dead_channels,
            "use_hampel": not a.no_hampel,
            "feat_version": int(C.FEAT_VERSION),
            "train_set_accuracy_NOT_a_performance_number": {
                dn: float((rf["pred"][:, j] == rf["true"][:, j]).mean())
                for j, dn in enumerate(d["dof_names"])}}
    a.out.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n→ {a.out}")
    if ts:
        print(f"→ {ts}  (TorchScript)")
    print(f"→ {a.out.with_suffix('.json')}")
    print("\n⚠️ 這個模型沒有自己的準確率數字（沒有留出資料）。")
    print("   誠實的期望值請引用同一批資料的 LORO 平均，")
    print("   **不要**引用上面那個訓練集準確率。")


if __name__ == "__main__":
    main()
