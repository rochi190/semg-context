"""觸發率：每個「要離開 rest」的 trial×DOF，模型在維持段有沒有輸出過正確狀態。

★ 這是最貼近「不好觸發」體感的指標 —— 比準確率有用：
   準確率會被 rest（多數類）稀釋，觸發率只看「該動的時候有沒有動」。
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from semg.v2.infer import Recognizer
from semg.v2 import config as C, cues as Q

MODEL = os.environ.get("MODEL", "V5moving/models/final_v5_split.pt")
CSV, TAG = sys.argv[1], sys.argv[2]
cue = Q.load_cues(CSV.replace(".csv", "_cues.csv"))
mv = pd.read_csv(CSV, usecols=[f"ch{c}" for c in C.CH_CSV_COLS]).values.astype(float)*C.LSB_MV
mv -= mv.mean(0)
r = Recognizer(MODEL, votes=5, conf=0.6)
r.calibrate(mv[:10*C.FS]); r.reset_stream()
T, STB = [], []
STEP = C.STEP_SAMPLES
for s0 in range(0, len(mv)-STEP+1, STEP):
    for f in r.windows(mv[s0:s0+STEP]):
        d = r.classify(f)
        if d is None: continue
        T.append((s0+STEP)/C.FS); STB.append(tuple(x if x is not None else -9 for x in d["stable"]))
T = np.array(T); STB = np.array(STB)

exc = set()
ex = CSV.replace(".csv", "_exclude.txt")
if os.path.exists(ex):
    for ln in open(ex, encoding="utf-8"):
        ln = ln.split("#")[0].strip()
        if ln: exc.add(int(ln))

res = {d: {"n":0, "trig":0, "cov":[]} for d in C.DOFS}
for it in Q.intervals(cue, len(mv)):
    if it.get("phase") != "action" or it.get("rep") in exc: continue
    t0, t1 = it["start"]/C.FS, it["end"]/C.FS
    m = (T >= t0) & (T < t1)
    if not m.any(): continue
    for j, dof in enumerate(C.DOFS):
        tgt = it["dof"][j]
        if tgt == 0: continue                     # 只看「該離開 rest」的
        hit = (STB[m, j] == tgt)
        res[dof]["n"] += 1
        res[dof]["trig"] += int(hit.any())
        res[dof]["cov"].append(float(hit.mean()))
out = [TAG]
for dof in C.DOFS:
    v = res[dof]
    out.append(f"{v['trig']}/{v['n']}={v['trig']/v['n']:.0%}" if v["n"] else "—")
    out.append(f"{np.mean(v['cov']):.0%}" if v["n"] else "—")
print(f"{TAG:<8}" + "".join(f"{x:>12}" for x in out[1:]))
