"""
校準協定測試：校準期需要做「單一動作」還是「食指+拇指都要做」？

問題來源（使用者提出，很合理）：
    正規化是**逐通道**的 —— mu/sigma 各 3 個值。
    若校準時只做拇指，ch2（食指）只看到雜訊 → sigma_2 很小 →
    之後真的做食指時，ch2 除以小 sigma 會變成極大的 z-score，
    落到模型訓練時沒見過的範圍。

    而模型本來就要分 Index / Thumb，所以理論上兩根都要校準。

測試設計（同一人、同一天的兩個檔案，控制其他變因）：
    A. 用「同動作」的前 30 秒校準            ← 目前的建議
    B. 用「另一個動作」的前 30 秒校準        ← 錯配，測試會壞多少
    C. 用「兩個動作各 30 秒」合併校準        ← 使用者提出的做法
    D. 用整個檔案校準（離線基準，實務上做不到）

兩個指標分開看，因為它們對應不同用途：
    手指判別率  —— 模型分 Index/Thumb 的能力（若之後要做多指控制才重要）
    夾爪三態    —— 摺成 Contract/Release/Rest 後的表現（Lite 6 這階段真正用到的）

輸出：results/calibration_protocol_test.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
sys.path.insert(0, str(_SRC))
REPO_ROOT = _SRC.parent

from semg.core import pipeline as pp
from semg.core.softpinch_model import load_model
from semg.model_eval.run_model_validation import (
    MODEL_PATH, RMS_STEP, FS, PRED_MAPPING, preprocess_for_model,
)

RESULTS = REPO_ROOT / "results"
SEC_PER_PT = RMS_STEP / FS
WIN_PTS, STEP_SEC = 120, 0.1
CALIB_SEC = 30.0

# 同一人、同一天的 index / thumb 配對（控制受試者與電極貼放）
PAIRS = [
    ("gary",  "indexgary_20260716_17-06-56_motion1", "thumbgary_20260716_17-12-05_motion2"),
    ("neil1", "indexlin_20260716_21-58-56_motion1",  "thumblin_20260716_22-02-17_motion2"),
    ("neil1", "indexlin_20260720_20-51-59",          "thumblin_20260720_20-55-57"),
]


def to_gripper(l):
    return "Rest" if l == "Rest" else ("Contract" if "Contract" in l else "Release")


def envelope(subj, stem):
    df = pp.load_channel(f"{REPO_ROOT}/data/{subj}/{stem}.csv", channels=(1, 2, 3))
    return preprocess_for_model({c: df[f"ch{c}"].values for c in (1, 2, 3)})


def evaluate(env, mu, sigma, model, target):
    x = (env - mu) / (sigma + 1e-8)
    segs = np.stack([x[s:s + WIN_PTS]
                     for s in range(0, len(x) - WIN_PTS + 1, int(STEP_SEC / SEC_PER_PT))])
    with torch.no_grad():
        logits, _, _ = model(torch.tensor(segs, dtype=torch.float32))
    lab = np.array([PRED_MAPPING[i] for i in logits.argmax(1).numpy()])
    act = lab[lab != "Rest"]
    finger = float(np.mean([l.startswith(target) for l in act])) if len(act) else np.nan
    g = np.array([to_gripper(l) for l in lab])
    return {"finger_acc": finger,
            "frac_rest": float((lab == "Rest").mean()),
            "frac_contract": float((g == "Contract").mean()),
            "frac_release": float((g == "Release").mean()),
            # 三態是否合理：出力與放鬆應該都有相當比例，不該只剩一種
            "three_state_balance": float(min((g == "Contract").mean(),
                                              (g == "Release").mean()))}


def main():
    model, _ = load_model(MODEL_PATH / "model.pth")
    model.eval()
    n_cal = int(CALIB_SEC / SEC_PER_PT)

    rows = []
    for subj, idx_stem, thb_stem in PAIRS:
        env_i, env_t = envelope(subj, idx_stem), envelope(subj, thb_stem)
        cal_i, cal_t = env_i[:n_cal], env_t[:n_cal]
        both = np.concatenate([cal_i, cal_t])          # 模擬「兩個動作都做」的校準期

        for tag, test_env, target, stem in (("index", env_i, "Index", idx_stem),
                                             ("thumb", env_t, "Thumb", thb_stem)):
            same = cal_i if tag == "index" else cal_t
            other = cal_t if tag == "index" else cal_i
            for scheme, cal in (("A_同動作校準", same),
                                 ("B_另一動作校準", other),
                                 ("C_兩動作都校準", both),
                                 ("D_整檔校準(離線基準)", test_env)):
                r = evaluate(test_env, cal.mean(0), cal.std(0), model, target)
                rows.append({"subject": subj, "posture": tag, "file": stem,
                             "scheme": scheme, **r})
                print(f"  {subj:6s} {tag:6s} {scheme:22s} "
                      f"手指判別 {r['finger_acc']:6.1%}  Rest {r['frac_rest']:5.1%}  "
                      f"出力 {r['frac_contract']:5.1%} 放鬆 {r['frac_release']:5.1%}",
                      flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "calibration_protocol_test.csv", index=False)

    print("\n" + "=" * 74)
    print("彙總（3 對檔案 × index/thumb = 6 個測試情境）")
    print("=" * 74)
    print(f"{'校準方式':<24s}{'手指判別率':>12s}{'Rest佔比':>10s}{'三態平衡':>10s}")
    print("-" * 74)
    for scheme in ("A_同動作校準", "B_另一動作校準", "C_兩動作都校準", "D_整檔校準(離線基準)"):
        g = d[d.scheme == scheme]
        print(f"{scheme:<24s}{g.finger_acc.mean():>11.1%}{g.frac_rest.mean():>10.1%}"
              f"{g.three_state_balance.mean():>10.1%}")

    print("\n判讀：")
    print("  手指判別率 —— 若要做多指控制才重要（亂猜 50%）")
    print("  三態平衡   —— 出力/放鬆兩者中較小者的佔比；太低代表某一態幾乎不出現")
    print(f"\n[完成] {RESULTS/'calibration_protocol_test.csv'}")


if __name__ == "__main__":
    main()
