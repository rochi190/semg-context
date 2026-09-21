"""
離線串流驗證：把錄好的檔案「當成即時串流」逐塊餵進 RealtimeController，
量後處理層（多數決 + 信心門檻 + 狀態機）到底把誤觸發壓下來多少。

為什麼要這支：
    gripper_readiness_test.py 量的是**模型原始輸出**，誤觸發 12.8%。
    但論文從來不直接用原始輸出驅動馬達——中間夾了一整層我們沒測過。
    這支就是把那層補上，用同一批檔案重測，判斷能不能接 Lite 6。

這也同時驗證 control.py 的串流實作正確（濾波器狀態、buffer 管理、時序），
是接硬體前的最後一道離線關卡。

判準（接 Lite 6 的門檻）：
    誤觸發 < 2%  且  偵測延遲 < 0.8s   → 可以接
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
sys.path.insert(0, str(_SRC))
REPO_ROOT = _SRC.parent

from semg.core import pipeline as pp
from semg.model_eval.run_model_validation import MODEL_PATH
from semg.realtime.control import (
    RealtimeController, StreamingPreprocessor, SEC_PER_PT, MODEL_WINDOW_PTS,
)

RESULTS_ROOT = REPO_ROOT / "results"

CHUNK_SEC = 0.1                 # 模擬 STM32 每 100ms 送一批（跟論文主迴圈同步率）
CALIB_SEC = 30.0                # 開機校準期長度
ONSET_K, QUIET_K, ONSET_MIN_SEC = 3.0, 1.0, 0.3


def envelope_events(env_ch: np.ndarray):
    """用包絡線自己定義客觀的出力起點與安靜期（不依賴猜測的 trial 邊界）。"""
    base = np.median(env_ch)
    mad = np.median(np.abs(env_ch - base)) * 1.4826 + 1e-12
    active = env_ch > base + ONSET_K * mad
    min_pts = int(round(ONSET_MIN_SEC / SEC_PER_PT))
    onsets, i = [], 0
    while i < len(active):
        if active[i]:
            j = i
            while j < len(active) and active[j]:
                j += 1
            if j - i >= min_pts:
                onsets.append(i)
            i = j
        else:
            i += 1
    return np.array(onsets), env_ch < base + QUIET_K * mad


def run_file(path: Path, posture: str, required_votes, actuator_sec):
    df = pp.load_channel(str(path), channels=(1, 2, 3))
    raw = np.stack([pp.adc_to_mv(df[f"ch{c}"].values) for c in (1, 2, 3)], axis=1)
    raw = raw - raw.mean(axis=0)

    ctl = RealtimeController(MODEL_PATH / "model.pth", actuator_sec=actuator_sec,
                              required_votes=required_votes)
    n_calib = int(CALIB_SEC * 2000)
    if len(raw) < n_calib + 2000 * 20:
        return None
    ctl.calibrate(raw[:n_calib])

    # 供評分用的參考包絡線（用同一套前處理，離線一次算完）
    ref_env = StreamingPreprocessor().add_chunk(raw[n_calib:])
    ch = 1 if posture == "index" else 2
    onsets, quiet = envelope_events(ref_env[:, ch])
    if len(onsets) < 3:
        return None

    step = int(CHUNK_SEC * 2000)
    raw_hist, voted_hist, t_hist, cmds = [], [], [], []
    t_proc = []
    for s in range(n_calib, len(raw) - step, step):
        now = (s - n_calib) / 2000.0
        t0 = time.perf_counter()
        cmd = ctl.push(raw[s:s + step], now)
        t_proc.append(time.perf_counter() - t0)
        if len(ctl.env_buf) >= MODEL_WINDOW_PTS and ctl.calibrated:
            raw_hist.append(ctl.last_raw_pred)
            voted_hist.append(ctl.last_voted)
            t_hist.append(now)
        if cmd:
            cmds.append((now, cmd))

    if not t_hist:
        return None
    t_hist = np.array(t_hist)
    # 預測時刻對應到 ref_env 的索引（trailing：現在 = 視窗結尾）
    idx = np.clip((t_hist / SEC_PER_PT).astype(int), 0, len(quiet) - 1)

    res = {"file": path.stem, "posture": posture, "n_onsets": len(onsets),
           "n_cmd_close": sum(1 for _, c in cmds if c == "CLOSE"),
           "n_cmd_open": sum(1 for _, c in cmds if c == "OPEN"),
           "proc_ms_median": float(np.median(t_proc) * 1000),
           "proc_ms_p95": float(np.percentile(t_proc, 95) * 1000)}

    for tag, seq in (("raw", np.array(raw_hist, dtype=object)),
                      ("voted", np.array([v if v else "None" for v in voted_hist], dtype=object))):
        lats, hits = [], 0
        for o in onsets:
            t_on = o * SEC_PER_PT
            m = (t_hist >= t_on) & (t_hist <= t_on + 2.0)
            if not m.any():
                continue
            k = np.where(seq[m] == "Contract")[0]
            if len(k):
                hits += 1
                lats.append(t_hist[m][k[0]] - t_on)
        q = quiet[idx]
        res[f"{tag}_detect_rate"] = hits / len(onsets)
        res[f"{tag}_latency"] = float(np.median(lats)) if lats else np.nan
        res[f"{tag}_false_act"] = float((seq[q] == "Contract").mean()) if q.any() else np.nan
    return res


def main():
    files = []
    for subj in ("gary", "neil1", "CBW1"):
        for p in sorted((REPO_ROOT / "data" / subj).glob("*.csv")):
            for post in ("index", "thumb"):
                if p.stem.startswith(post):
                    files.append((subj, p, post))

    all_rows = []
    for rv, tag in [(None, "原版（不強制票數）"), (3, "強制 ≥3 票")]:
        print(f"\n{'='*72}\n設定：多數決 5 視窗、信心 >0.9、{tag}、致動 0.8s\n{'='*72}", flush=True)
        rows = []
        for subj, path, posture in files:
            r = run_file(path, posture, rv, 0.8)
            if r is None:
                continue
            r["subject"], r["setting"] = subj, tag
            rows.append(r)
            print(f"  {subj:6s} {posture:6s} {path.stem[:28]:30s} "
                  f"原始 誤觸發{r['raw_false_act']:5.1%} → 後處理 誤觸發{r['voted_false_act']:5.1%}  "
                  f"偵測{r['voted_detect_rate']:5.0%} 延遲{r['voted_latency']:.2f}s  "
                  f"指令 {r['n_cmd_close']}關/{r['n_cmd_open']}開", flush=True)
        d = pd.DataFrame(rows)
        all_rows.append(d)
        print(f"\n  ── 平均 ──")
        print(f"    誤觸發   原始 {d.raw_false_act.mean():5.1%}  →  後處理 {d.voted_false_act.mean():5.1%}")
        print(f"    偵測率   原始 {d.raw_detect_rate.mean():5.1%}  →  後處理 {d.voted_detect_rate.mean():5.1%}")
        print(f"    延遲     原始 {d.raw_latency.median():5.2f}s  →  後處理 {d.voted_latency.median():5.2f}s")
        print(f"    每步運算 中位數 {d.proc_ms_median.median():.2f} ms / p95 {d.proc_ms_p95.max():.2f} ms"
              f"  （預算 {CHUNK_SEC*1000:.0f} ms）")

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(RESULTS_ROOT / "control_offline_test.csv", index=False)
    print(f"\n[完成] {RESULTS_ROOT/'control_offline_test.csv'}")


if __name__ == "__main__":
    main()
