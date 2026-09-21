"""量測真實 onset 延遲：從肌電實際開始變化 → 到各層輸出翻轉。

參考點刻意**不用 cue 時間戳** —— cue 之後還有受試者的反應時間，
量到的會是「反應+系統」。這裡用**肌電包絡實際起跳**當 t=0。
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
sys.path.insert(0, "moving_poseV2/src")
from semg.v2.infer import Recognizer
from semg.v2 import config as C, preprocess as P, cues as Q

MODEL = os.environ.get("ONSET_MODEL", "V5moving/models/final_v5_split.pt")
CSV   = sys.argv[1]
VOTES, CONF, DWELL = 1, 0.6, 6          # moving_poseV2 的實際設定
VOTE_M, VOTE_N = 6, 4                   # 控制層
MIRROR_HOLD = 0.5

cues = Q.load_cues(CSV.replace(".csv", "_cues.csv"))
mv = pd.read_csv(CSV, usecols=[f"ch{c}" for c in C.CH_CSV_COLS]).values.astype(float)*C.LSB_MV
mv -= mv.mean(0)

rec = Recognizer(MODEL, votes=VOTES, conf=CONF, dwell=DWELL)
rec.calibrate(mv[:10*C.FS]); rec.reset_stream()

# 逐塊餵，記下每個視窗的時間、raw、stable
STEP = C.STEP_SAMPLES
recs = []
for s0 in range(0, len(mv)-STEP+1, STEP):
    for f in rec.windows(mv[s0:s0+STEP]):
        d = rec.classify(f)
        if d is None: continue
        # 視窗結束於 s0+STEP，視窗涵蓋 [end-WIN, end]
        t_end = (s0+STEP)/C.FS
        recs.append((t_end, tuple(d["raw"]), tuple(x if x is not None else 0 for x in d["stable"])))
T   = np.array([r[0] for r in recs])
RAW = np.array([r[1] for r in recs])
STB = np.array([r[2] for r in recs])

# 濾波後包絡，用來抓真實 onset
env = P.rms_envelope(P.filter_causal(mv, use_hampel=False))
GRP = {g: [C.CH_NAMES.index(n) for n in ch] for g, ch in C.CH_GROUPS.items()}

DOFI = {"hand":0, "elbow":1, "shoulder":2}
rows=[]
ev = cues.reset_index(drop=True)
for i, r in ev.iterrows():
    if r.event != "go": continue
    nxt = ev.iloc[i+1:]; nxt = nxt[nxt.event=="return"]
    if not len(nxt): continue
    t_go, t_ret = r.sample_index/C.FS, nxt.iloc[0].sample_index/C.FS
    prep = ev.iloc[:i]; prep = prep[prep.event=="prepare"]
    if not len(prep): continue
    t_prep = prep.iloc[-1].sample_index/C.FS
    spec = C.ACTION_TO_DOF.get(str(r.gesture)) if hasattr(C,"ACTION_TO_DOF") else None
    tgt = Q.action_to_dof(str(r.gesture)) if hasattr(Q,"action_to_dof") else None
    if tgt is None:
        from semg.v2.config import action_to_dof as a2d
        tgt = a2d(str(r.gesture))
    for dof, j in DOFI.items():
        if tgt[j] == 0: continue                     # 只看「離開 rest」的那些
        grp = GRP[C.DOF_CHANNEL_GROUP[dof]]
        # 真實 onset：該組通道包絡總和，從 prepare 起算，超過 prepare 前 2s 基線的 3 倍
        b0, b1 = int((t_prep-2.0)*C.ENV_FS), int(t_prep*C.ENV_FS)
        e = env[:, grp].sum(1)
        base = np.median(e[max(0,b0):b1]) if b1>b0 else np.median(e)
        seg = slice(int(t_prep*C.ENV_FS), int(t_ret*C.ENV_FS))
        idx = np.flatnonzero(e[seg] > base*3.0)
        if not len(idx): continue
        t_on = t_prep + idx[0]/C.ENV_FS
        # 各層首次正確
        def first(arr):
            m = (T > t_on) & (T < t_ret) & (arr[:, j] == tgt[j])
            return T[m][0]-t_on if m.any() else np.nan
        t_raw, t_stb = first(RAW), first(STB)
        # 控制層 4-of-6：在 stable 序列上模擬
        buf=[]; t_ctl=np.nan
        for k in range(len(T)):
            if T[k] <= t_on or T[k] >= t_ret: continue
            buf.append(STB[k, j]); buf = buf[-VOTE_M:]
            if len(buf)==VOTE_M and buf.count(tgt[j])>=VOTE_N:
                t_ctl = T[k]-t_on; break
        rows.append(dict(gesture=str(r.gesture), dof=dof, t_on=t_on,
                         raw=t_raw, stable=t_stb, ctl=t_ctl))
d = pd.DataFrame(rows)
print(f"\n{CSV.split('/')[-1]}   {len(d)} 個 (trial × DOF) 樣本\n")
print(f"{'層級':<38}{'中位':>8}{'p25':>8}{'p75':>8}{'p90':>8}")
for col, name in [("raw","① 逐窗 raw 首次正確"),
                  ("stable","② + Recognizer dwell=6 → stable"),
                  ("ctl","③ + 控制層 4-of-6")]:
    v = d[col].dropna()
    print(f"{name:<38}{v.median():>7.2f}s{v.quantile(.25):>7.2f}s{v.quantile(.75):>7.2f}s{v.quantile(.90):>7.2f}s")
print()
for dof in ("hand","elbow","shoulder"):
    s = d[d.dof==dof]
    if not len(s): continue
    print(f"  {dof:<10} raw {s.raw.median():.2f}s  stable {s.stable.median():.2f}s  ctl {s.ctl.median():.2f}s   (n={len(s)})")
d.insert(0, "recording", CSV.split("/")[-1].replace(".csv",""))
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/onset_out.csv"
d.to_csv(out, index=False)
