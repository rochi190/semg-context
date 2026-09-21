"""
計算成本量測：回答「傳統方法 + CNN+LSTM 一起跑，算力吃得消嗎？」以及
「我自己跑前處理很久，是不是繪圖拖慢的？」

量三種情境，因為它們的成本天差地遠，混在一起講會得到錯誤結論：

  情境 A「離線批次」：一次處理整個檔案（我們所有分析腳本的做法）。
      這是你覺得「跑很久」的那個。慢是正常的，因為要處理 300 秒 × 2000Hz 的全部資料，
      而且 Hampel 的中位數濾波是 O(N·k)。

  情境 B「即時串流」：每 0.25 秒只處理「新進來的 500 個取樣點」，濾波器帶狀態
      （lfilter zi）延續，不重算歷史。這才是實際做 real-time 控制時的成本，
      跟 A 差好幾個數量級。這是判斷「來不來得及」的唯一依據。

  情境 C「繪圖」：單獨計時，確認你的直覺對不對。

輸出：results/compute_benchmark.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import median_filter
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")   # src/ 目錄，與檔案放在多深無關
sys.path.insert(0, str(_SRC))
from semg.core import pipeline as pp
from semg.core.softpinch_model import load_model
from semg.model_eval.run_model_validation import (
    MODEL_PATH, FS, NOTCH_FREQ, BANDPASS, HAMPEL_HALF_WINDOW, HAMPEL_SIGMA,
    RMS_WINDOW, RMS_STEP, TRIM_START_SEC, TRIM_END_SEC,
    causal_notch_bandpass, causal_hampel, rms_stream,
)

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "src").parent
RESULTS_ROOT = REPO_ROOT / "results"
TEST_CSV = REPO_ROOT / "data/neil1/indexlin_20260716_21-58-56_motion1.csv"

WINDOW_SEC, STEP_SEC = 3.0, 0.25
N_REPEAT = 20          # 即時情境重複次數，取中位數避免單次抖動


def timeit(fn, n=1):
    """回傳 (中位數秒, 結果)。n>1 時重複執行取中位數。"""
    ts = []
    out = None
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out


def main():
    rows = []
    print("=" * 74)
    print("情境 A：離線批次（處理整個檔案一次）")
    print("=" * 74)

    t_load, df = timeit(lambda: pp.load_channel(str(TEST_CSV), channels=(1, 2, 3)))
    n_samples = len(df)
    dur = n_samples / FS
    print(f"  檔案長度 {dur:.0f} 秒（{n_samples:,} 取樣點 × 3 通道）")
    print(f"  1. 讀取 CSV                 {t_load*1000:8.1f} ms")
    rows.append({"scenario": "offline_batch", "stage": "read_csv", "ms": t_load * 1000})

    raw = {ch: df[f"ch{ch}"].values for ch in (1, 2, 3)}

    def stage_prep():
        out = []
        for ch in (1, 2, 3):
            mv = pp.adc_to_mv(raw[ch])
            mv = pp.trim(mv, FS, TRIM_START_SEC, TRIM_END_SEC)
            out.append(mv - mv.mean())
        return out
    t_prep, mvs = timeit(stage_prep)
    print(f"  2. ADC→mV + 裁切 + 去DC     {t_prep*1000:8.1f} ms")
    rows.append({"scenario": "offline_batch", "stage": "adc_trim_dc", "ms": t_prep * 1000})

    t_filt, filts = timeit(
        lambda: [causal_notch_bandpass(m, FS, NOTCH_FREQ, BANDPASS) for m in mvs])
    print(f"  3. Notch + Bandpass         {t_filt*1000:8.1f} ms")
    rows.append({"scenario": "offline_batch", "stage": "notch_bandpass", "ms": t_filt * 1000})

    t_hamp, hamps = timeit(
        lambda: [causal_hampel(f, HAMPEL_HALF_WINDOW, HAMPEL_SIGMA) for f in filts])
    print(f"  4. Hampel 濾波              {t_hamp*1000:8.1f} ms   <-- 通常是瓶頸")
    rows.append({"scenario": "offline_batch", "stage": "hampel", "ms": t_hamp * 1000})

    t_rms, envs = timeit(lambda: [rms_stream(h, RMS_WINDOW, RMS_STEP) for h in hamps])
    print(f"  5. RMS envelope             {t_rms*1000:8.1f} ms")
    rows.append({"scenario": "offline_batch", "stage": "rms_envelope", "ms": t_rms * 1000})

    total_prep = t_load + t_prep + t_filt + t_hamp + t_rms
    print(f"  ── 前處理小計                {total_prep*1000:8.1f} ms  "
          f"（= {dur/total_prep:.0f}× 即時速度）")

    env = np.stack([e[:min(len(x) for x in envs)] for e in envs], axis=1)
    mu, sigma = env.mean(axis=0), env.std(axis=0)
    env_n = (env - mu) / (sigma + 1e-8)

    model, _ = load_model(MODEL_PATH / "model.pth")
    model.eval()
    win = int(round(WINDOW_SEC / (RMS_STEP / FS)))
    step = int(round(STEP_SEC / (RMS_STEP / FS)))
    segs = np.stack([env_n[s:s + win] for s in range(0, len(env_n) - win + 1, step)])
    print(f"\n  整段共 {len(segs)} 個視窗")

    def infer_batch():
        with torch.no_grad():
            return model(torch.tensor(segs, dtype=torch.float32))
    t_batch, _ = timeit(infer_batch, n=3)
    print(f"  6. 模型推論（{len(segs)} 視窗一次批次）  {t_batch*1000:8.1f} ms "
          f"（每視窗 {t_batch/len(segs)*1000:.3f} ms）")
    rows.append({"scenario": "offline_batch", "stage": f"model_infer_{len(segs)}_windows",
                  "ms": t_batch * 1000})

    print("\n" + "=" * 74)
    print("情境 B：即時串流（每 0.25 秒只處理新進來的資料）")
    print("=" * 74)
    new_n = int(STEP_SEC * FS)          # 500 個新取樣點
    print(f"  每步新資料：{new_n} 取樣點 × 3 通道  |  預算：{STEP_SEC*1000:.0f} ms")

    # 濾波器帶狀態，只處理新片段
    b_n, a_n = iirnotch(NOTCH_FREQ, NOTCH_FREQ / 2, fs=FS)
    b_b, a_b = butter(4, [BANDPASS[0] / (FS / 2), BANDPASS[1] / (FS / 2)], btype="band")
    zi_n = [lfilter_zi(b_n, a_n) * 0 for _ in range(3)]
    zi_b = [lfilter_zi(b_b, a_b) * 0 for _ in range(3)]
    chunk = [m[:new_n].copy() for m in mvs]
    # Hampel 需要前後文，串流時保留 kernel 半寬的歷史
    hist = [f[-HAMPEL_HALF_WINDOW:].copy() for f in filts]

    def stream_filter():
        outs = []
        for c in range(3):
            y, _ = lfilter(b_n, a_n, chunk[c], zi=zi_n[c])
            y, _ = lfilter(b_b, a_b, y, zi=zi_b[c])
            outs.append(y)
        return outs
    t_sfilt, souts = timeit(stream_filter, n=N_REPEAT)
    print(f"  1. Notch + Bandpass（增量）      {t_sfilt*1000:7.3f} ms")
    rows.append({"scenario": "realtime_stream", "stage": "notch_bandpass", "ms": t_sfilt * 1000})

    def stream_hampel():
        outs = []
        for c in range(3):
            buf = np.concatenate([hist[c], souts[c]])
            k = 2 * HAMPEL_HALF_WINDOW + 1
            med = median_filter(buf, size=k, mode="reflect")
            mad = 1.4826 * median_filter(np.abs(buf - med), size=k, mode="reflect")
            outs.append(np.where(np.abs(buf - med) > HAMPEL_SIGMA * mad, med, buf)[-new_n:])
        return outs
    t_shamp, shamps = timeit(stream_hampel, n=N_REPEAT)
    print(f"  2. Hampel（增量，含歷史緩衝）    {t_shamp*1000:7.3f} ms")
    rows.append({"scenario": "realtime_stream", "stage": "hampel", "ms": t_shamp * 1000})

    def stream_rms():
        return [rms_stream(np.concatenate([hist[c][-RMS_WINDOW:], shamps[c]]),
                            RMS_WINDOW, RMS_STEP) for c in range(3)]
    t_srms, _ = timeit(stream_rms, n=N_REPEAT)
    print(f"  3. RMS envelope（增量）          {t_srms*1000:7.3f} ms")
    rows.append({"scenario": "realtime_stream", "stage": "rms_envelope", "ms": t_srms * 1000})

    one = torch.tensor(segs[:1], dtype=torch.float32)

    def infer_one():
        with torch.no_grad():
            return model(one)
    t_one, _ = timeit(infer_one, n=N_REPEAT * 5)
    print(f"  4. CNN+LSTM 推論（單一視窗）     {t_one*1000:7.3f} ms")
    rows.append({"scenario": "realtime_stream", "stage": "cnn_lstm_infer_1window",
                  "ms": t_one * 1000})

    # 傳統特徵（在原始 EMG 上算，這才是修正後的版本會做的事）
    def trad_feats():
        w = int(WINDOW_SEC * FS)
        out = []
        for c in range(3):
            x = filts[c][-w:]
            dx = np.diff(x)
            out += [np.mean(np.abs(x)), np.sqrt(np.mean(x ** 2)), np.sum(np.abs(dx)),
                    int(np.sum(np.diff(np.sign(dx)) != 0)),
                    int(np.sum((x[:-1] * x[1:]) < 0))]
        return np.array(out)
    t_trad, _ = timeit(trad_feats, n=N_REPEAT)
    print(f"  5. 傳統特徵（原始EMG，時域5項）  {t_trad*1000:7.3f} ms")
    rows.append({"scenario": "realtime_stream", "stage": "traditional_features_timedomain",
                  "ms": t_trad * 1000})

    shared = t_sfilt + t_shamp + t_srms
    print(f"\n  ── 共用前處理           {shared*1000:7.3f} ms")
    print(f"  ── 只用 CNN+LSTM        {(shared+t_one)*1000:7.3f} ms  "
          f"（佔預算 {(shared+t_one)/STEP_SEC*100:5.1f}%）")
    print(f"  ── 只用傳統特徵         {(shared+t_trad)*1000:7.3f} ms  "
          f"（佔預算 {(shared+t_trad)/STEP_SEC*100:5.1f}%）")
    print(f"  ── 兩者都跑             {(shared+t_one+t_trad)*1000:7.3f} ms  "
          f"（佔預算 {(shared+t_one+t_trad)/STEP_SEC*100:5.1f}%）")

    print("\n" + "=" * 74)
    print("情境 C：繪圖成本")
    print("=" * 74)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def make_plot():
        fig, axes = plt.subplots(4, 1, figsize=(19, 10))
        t = np.arange(len(env)) * (RMS_STEP / FS)
        for c in range(3):
            axes[c].plot(t, env[:, c], lw=0.8)
        for i in range(0, len(segs), 4):
            axes[3].broken_barh([(i * STEP_SEC, STEP_SEC)], (0, 1))
        fig.savefig("/tmp/_bench_plot.png", dpi=150)
        plt.close(fig)
    t_plot, _ = timeit(make_plot, n=3)
    print(f"  單張圖（4 面板 + 存檔 dpi=150）  {t_plot*1000:8.1f} ms")
    rows.append({"scenario": "plotting", "stage": "one_figure", "ms": t_plot * 1000})
    print(f"  → 相較整個檔案的前處理（{total_prep*1000:.0f} ms），"
          f"繪圖佔 {t_plot/(total_prep+t_plot)*100:.0f}%")

    pd.DataFrame(rows).to_csv(RESULTS_ROOT / "compute_benchmark.csv", index=False)
    print(f"\n[完成] {RESULTS_ROOT / 'compute_benchmark.csv'}")


if __name__ == "__main__":
    main()
