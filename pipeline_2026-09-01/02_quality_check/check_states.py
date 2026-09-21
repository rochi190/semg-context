"""錄完短測後，確認每個自由度的狀態**真的分得出來**。

## 為什麼要有這一步

多輸出模型的每個頭都在做一個分類問題。如果某個自由度的兩個狀態
在訊號上根本沒有差異，那個頭再怎麼訓練都只會學到「猜多數類」，
而且**整體全對率會被它一個人拖垮**（全對 = 所有頭同時正確）。

正式錄 8.4 分鐘之前先花 2 分鐘確認，比事後發現整批資料某個 DOF 沒用好。

各自由度的先驗預期：

| 自由度 | 狀態 | 依據 |
|---|---|---|
| hand | rest / index / thumb / pinch | 3 顆遠端電極的**協同模式**。四者兩兩要能分開 —— 其中 `index vs pinch`、`thumb vs pinch` 最難，因為捏會同時用到食指與拇指的屈肌 |
| elbow | rest / flex | flex = 前臂抬起**懸空維持**，二頭肌持續撐住重量 → 振幅明顯高於 rest |
| shoulder | rest / flex | flex = 手臂前舉**懸空維持**，三角肌前束持續撐住 → 同上 |

⚠️ 標的是**維持中的姿勢**，不是移動過程。若 `flex vs rest` 偏低，
最常見的原因**不是電極位置，而是受試者把手臂靠著東西** ——
手肘撐在桌面或上臂貼死身側，重力被外物支撐掉，肌肉就不需要出力。
先確認錄製時手臂全程懸空，再考慮移電極。

若 hand 的 `index vs pinch` 偏低，代表捏與單指彎的協同在你的電極上分不開 ——
這時要考慮把 hand 簡化成 {rest, pinch} 兩態，不要硬留分不出來的狀態。

## 用法（約 2 分鐘）

    # 1. 錄一段只含單一自由度動作的短錄音
    python src/hardware/record_session.py --port COM9 --subject gary \
        --gestures index_flex,thumb_flex,elbow_flex,elbow_ext,shoulder_flex,shoulder_ext \
        --reps 3 --outdir data/raw/statecheck/gary

    # 2. 檢查
    PYTHONPATH=src python src/semg/v2/check_states.py data/raw/statecheck/gary/multi_*.csv

## 怎麼看結果

    ≥ 85%  → 該狀態可用
    60–85% → 勉強，建議調整電極位置再測一次
    < 60%  → 分不出來，該電極位置無效
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from semg.v2 import config as C
from semg.v2 import cues as Q
from semg.v2 import features as F
from semg.v2 import preprocess as P


def build(csv_path: Path):
    """切窗 + 抽特徵，回傳 (X, 每個 DOF 的狀態, trial 分組鍵)。"""
    cue_csv = Q.cue_path_for(csv_path)
    if cue_csv is None:
        raise SystemExit(f"{csv_path} 沒有對應的 _cues.csv")
    cues = Q.load_cues(cue_csv)
    raw = P.load_recording(csv_path, trim=False)
    filt = P.filter_causal(raw)
    cs, ce = Q.calibration_span(cues, len(filt))
    scales, epss = F.file_scales(filt[cs:ce])

    slab = Q.sample_labels(cues, len(filt))
    rep_of = np.full(len(filt), -1, dtype=np.int32)
    for iv in Q.intervals(cues, len(filt)):
        rep_of[iv["start"]:iv["end"]] = iv["rep"]

    X, Y, G = [], [], []
    for s in range(0, len(filt) - C.WIN_SAMPLES + 1, C.STEP_SAMPLES):
        e = s + C.WIN_SAMPLES
        dof = Q.window_label(slab, s, e)
        if dof is None:
            continue
        X.append(F.extract(filt[s:e], scales, epss))
        Y.append(dof)
        G.append(int(np.bincount(np.maximum(rep_of[s:e], 0)).argmax()))
    if not X:
        raise SystemExit("沒有任何可用視窗 —— 檢查 cue 檔是否正確")
    return np.stack(X), np.stack(Y), np.asarray(G)


def check(X, Y, G, dof_idx: int, a_state: int, b_state: int, seed: int = C.SEED):
    """某個 DOF 的兩個狀態能不能分開。

    兩個必要的隔離，少一個結果就沒有意義：

    1. **其他自由度必須都在 rest**。
       否則「index=rest」這一類會混進 thumb_flex trial 的視窗
       （那時 index 確實是 rest），分類器就能靠「拇指安不安靜」作答，
       跟食指有沒有伸展完全無關。
       實測：不做這個隔離時，即使 ext 在訊號上**毫無資訊**（串音=0），
       也會得到 89.1% —— 差最大的特徵全是 `thumb_*`。

    2. **按 trial 分組切分**。同一次出力的視窗重疊 90%，
       隨機切窗會讓測試集有訓練樣本的近鄰。

    指標用 **balanced accuracy**（= 兩類召回率平均），不是原始正確率。
    rest 的視窗數是 ext 的 6–7 倍，原始正確率下「全部猜 rest」就有 ~87%，
    看起來像分得很好，其實完全沒有判別力。balanced accuracy 對這種
    退化解會給 50%。
    """
    y = Y[:, dof_idx]
    others = [k for k in range(Y.shape[1]) if k != dof_idx]
    isolated = (Y[:, others] == 0).all(axis=1) if others else np.ones(len(Y), bool)
    m = np.isin(y, [a_state, b_state]) & isolated
    if m.sum() < 40 or len(np.unique(y[m])) < 2:
        return None
    Xa, ya, Ga = X[m], (y[m] == b_state).astype(int), G[m]
    groups = np.unique(Ga)
    if len(groups) < 2:
        return None
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(3):
        gs = groups.copy()
        rng.shuffle(gs)
        te_g = gs[:max(1, len(gs) // 3)]
        te = np.isin(Ga, te_g)
        tr = ~te
        if tr.sum() < 20 or te.sum() < 10 or len(np.unique(ya[tr])) < 2:
            continue
        sc = StandardScaler().fit(Xa[tr])
        clf = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=0,
                                     class_weight="balanced")
        clf.fit(sc.transform(Xa[tr]), ya[tr])
        accs.append(float(balanced_accuracy_score(
            ya[te], clf.predict(sc.transform(Xa[te])))))
    if not accs:
        return None
    return float(np.mean(accs)), float(np.std(accs)), int(m.sum())


def main():
    ap = argparse.ArgumentParser(description="檢查 ext 狀態在目前電極配置下分不分得出來")
    ap.add_argument("csv", nargs="+", type=Path)
    a = ap.parse_args()

    Xs, Ys, Gs = [], [], []
    for i, p in enumerate(a.csv):
        X, Y, G = build(p)
        Xs.append(X); Ys.append(Y); Gs.append(G + i * 10000)
        print(f"  {p.name}: {len(X)} 視窗")
    X = np.concatenate(Xs); Y = np.concatenate(Ys); G = np.concatenate(Gs)

    print(f"\n通道配置：{C.N_CH} 通道"
          f"（USE_EXTENSOR_CHANNEL={C.USE_EXTENSOR_CHANNEL}）")
    print(f"總計 {len(X)} 視窗，{len(np.unique(G))} 個 trial\n")
    print("兩兩狀態的可分性")
    print("（只用其他自由度都在 rest 的視窗；按 trial 分組切分；"
          "指標 = balanced accuracy，50% = 完全分不出來）")
    print(f"{'自由度':<8s} {'比較':<18s} {'正確率':>14s} {'n':>7s}   判定")
    print("-" * 72)

    verdicts = {}
    for j, dn in enumerate(C.DOFS):
        states = C.DOF_STATES[dn]
        for ai in range(len(states)):
            for bi in range(ai + 1, len(states)):
                r = check(X, Y, G, j, ai, bi)
                if r is None:
                    continue
                acc, sd, n = r
                tag = ("✅ 可用" if acc >= 0.85 else
                       "⚠️ 勉強" if acc >= 0.60 else "❌ 分不出來")
                print(f"{C.DOF_LABELS_ZH[dn]:<8s} {states[ai]+' vs '+states[bi]:<18s} "
                      f"{acc:8.1%}±{sd:4.1%} {n:7d}   {tag}")
                # 每個 DOF 取**最差**的一組狀態對當判定依據
                verdicts[dn] = min(verdicts.get(dn, 1.0), acc)

    print("\n" + "=" * 72)
    weak = [(k, v) for k, v in verdicts.items() if v < 0.85]
    if not verdicts:
        print("沒有足夠資料做判定 —— 每個自由度都要錄到才行")
        return
    print(f"每個自由度最差的一組狀態對：")
    for k, v in sorted(verdicts.items(), key=lambda kv: kv[1]):
        tag = "✅" if v >= 0.85 else ("⚠️" if v >= 0.60 else "❌")
        print(f"    {tag} {C.DOF_LABELS_ZH.get(k, k):4s} {v:.1%}")
    if not weak:
        print("\n✅ 所有自由度的狀態都分得出來，可以錄正式資料。")
        return
    print(f"\n⚠️ 有 {len(weak)} 個自由度偏弱：")
    for k, v in weak:
        hint = {
            "elbow": "先確認維持時前臂是**懸空**的（沒撐在桌上）；"
                     "再檢查二頭肌電極是否在肌腹正中",
            "shoulder": "先確認維持時手臂是**懸空**的（沒靠著東西）；"
                        "再檢查三角肌前束電極 —— 應在肩峰前下方的肌腹上",
            "hand": "檢查曲腕肌/食指/拇指電極 —— 都在前臂掌側近端 1/3 處，"
                    "食指偏尺側、拇指偏橈側；三顆要拉開距離以降低串音",
        }.get(k, "檢查該部位電極位置")
        print(f"    {C.DOF_LABELS_ZH.get(k, k)} {v:.1%} → {hint}")
    print("\n移動電極位置重貼後再測一次，比事後調模型有效得多。")


if __name__ == "__main__":
    main()
