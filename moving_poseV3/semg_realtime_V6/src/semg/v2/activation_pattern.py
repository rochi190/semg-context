"""各姿勢的通道活化模式：go 段 RMS 相對 rest 段的倍率。

★ 這是診斷「電極貼對地方了嗎 / 受試者用對肌肉了嗎」最直接的量。
   模型看的不是絕對振幅（那被校準除掉了），是**哪些通道相對地亮起來**。
"""
import sys, numpy as np, pandas as pd, json
sys.path.insert(0, "src")
from semg.v2 import config as C, preprocess as P, cues as Q

def analyse(stem_path, subject):
    cue = Q.load_cues(stem_path + "_cues.csv")
    mv = pd.read_csv(stem_path + ".csv", usecols=[f"ch{c}" for c in C.CH_CSV_COLS]).values.astype(float)*C.LSB_MV
    mv -= mv.mean(0)
    filt = P.filter_causal(mv, use_hampel=False)
    env = P.rms_envelope(filt)                      # (m, 7) @40Hz
    iv = Q.intervals(cue, len(mv))
    # 排除的 trial
    exc = set()
    import os
    ex = stem_path + "_exclude.txt"
    if os.path.exists(ex):
        for line in open(ex, encoding="utf-8"):
            line = line.split("#")[0].strip()
            if line: exc.add(int(line))
    rest, per = [], {}
    for it in iv:
        if it.get("rep") in exc: continue
        a, b = int(it["start"]/C.ENV_STEP), int(it["end"]/C.ENV_STEP)
        if b <= a: continue
        seg = env[a:b]
        if it.get("phase") == "rest": rest.append(seg)
        elif it.get("phase") == "action": per.setdefault(it["gesture"], []).append(seg)
    base = np.median(np.concatenate(rest), axis=0)          # (7,) rest 中位
    out = {}
    for g, segs in per.items():
        out[g] = (np.median(np.concatenate(segs), axis=0) / base).tolist()
    return {"subject": subject, "rest_uV": (base*1000).tolist(), "ratio": out}

res = [analyse(p, s) for p, s in [
    ("V5moving/data/raw/multi/neil/multi_neil_20260810_21-54-56", "neil"),
    ("V5moving/data/raw/multi/neil/multi_neil_20260810_21-17-16", "neil2"),
    ("V5moving/data/raw/multi/gary/multi_gary_20260811_17-23-12", "gary"),
    ("V4moving/data/raw/multi/chou/multi_chou_20260910_18-27-41", "chou"),
    ("V4moving/data/raw/multi/leon/multi_leon_20260911_16-00-14", "leon"),
]]
json.dump(res, open(sys.argv[1], "w"), indent=1)
print("saved", sys.argv[1])
