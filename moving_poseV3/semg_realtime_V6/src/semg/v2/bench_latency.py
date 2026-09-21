"""即時推論延遲量測：確認在 CPU 上跑得贏取樣速度。

量的是**每個新視窗**要做的完整工作，拆成三段：
    串流濾波（notch → bandpass → 因果 Hampel）
    120 維特徵抽取
    模型前向（TorchScript vs eager）

判準不是「快不快」而是**跟不跟得上**：
每 `STEP_SEC`(0.05s) 產生一個新視窗，所以三段加起來必須 < 50ms，
否則佇列會越積越多，延遲無限增長。實務上要留餘裕 ——
同一台筆電還要同時收 2000Hz × 7ch 的序列埠資料並控制機械手臂。

⚠️ 一定要固定 thread 數再量。PyTorch 預設會吃滿所有核心，
   量出來很快，但部署時取樣執行緒被搶走 CPU 就會掉樣本。
   `C.CPU_THREADS` 就是為此存在。

用法：
    PYTHONPATH=src python -m semg.v2.bench_latency --bundle results/models/dof_model.pt
"""
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import torch

from semg.v2 import config as C
from semg.v2 import features as F
from semg.v2.infer import Recognizer
from semg.v2.preprocess import StreamFilter


def _timeit(fn, n: int, warmup: int = 10) -> tuple[float, float, float]:
    """回傳 (中位數, p95, 最大值) 毫秒。用中位數不用平均 —— 偶發的
    GC / 排程抖動會把平均拉走，但實際上不影響是否跟得上。"""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.asarray(ts)
    return float(np.median(ts)), float(np.percentile(ts, 95)), float(ts.max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--threads", type=int, default=C.CPU_THREADS)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    print(f"平台     : {platform.processor() or platform.machine()}  "
          f"{platform.system()}")
    print(f"torch    : {torch.__version__}   threads={torch.get_num_threads()}")
    print(f"CUDA 可用 : {torch.cuda.is_available()}"
          f"（即時推論一律走 CPU，Recognizer 內部寫死 device='cpu'）")

    r = Recognizer(a.bundle, votes=5, conf=0.6, threads=a.threads)
    dev = next(r.model.parameters()).device if not r.scripted else "cpu(scripted)"
    print(f"模型     : {'TorchScript' if r.scripted else 'eager'}   device={dev}   "
          f"seq_len={r.seq_len}   {len(r.mean)} 維特徵")

    rng = np.random.default_rng(0)
    block = rng.normal(0, 0.02, (C.STEP_SAMPLES, C.N_CH))     # 一步的原始資料
    win = rng.normal(0, 0.02, (C.WIN_SAMPLES, C.N_CH))
    scales, epss = F.file_scales(win)
    seq = torch.from_numpy(rng.standard_normal((1, r.seq_len, len(r.mean))
                                               ).astype(np.float32))

    sf = StreamFilter()
    sf(rng.normal(0, 0.02, (C.WIN_SAMPLES, C.N_CH)))          # 暖機填滿歷史
    parts = {}
    parts["串流濾波（每步 %d 樣本）" % C.STEP_SAMPLES] = _timeit(
        lambda: sf(block), a.n)
    parts["特徵抽取（120 維）"] = _timeit(lambda: F.extract(win, scales, epss), a.n)

    def fwd():
        with torch.inference_mode():
            r.model(seq)
    parts["模型前向"] = _timeit(fwd, a.n)

    print(f"\n{'步驟':<28s} {'中位':>8s} {'p95':>8s} {'max':>8s}")
    print("-" * 56)
    tot = 0.0
    for k, (m, p95, mx) in parts.items():
        print(f"{k:<28s} {m:>7.3f}ms {p95:>7.3f}ms {mx:>7.3f}ms")
        tot += m
    print("-" * 56)
    budget = C.STEP_SEC * 1e3
    print(f"{'合計（每個新視窗）':<28s} {tot:>7.3f}ms")
    print(f"{'即時預算（STEP_SEC）':<28s} {budget:>7.3f}ms   "
          f"→ 用掉 {tot / budget:.1%}，餘裕 {budget / max(tot, 1e-9):.0f}×")

    # 端到端：直接餵 Recognizer.push，涵蓋緩衝、投票等所有開銷
    r2 = Recognizer(a.bundle, votes=5, conf=0.6, threads=a.threads)
    r2.calibrate(rng.normal(0, 0.02, (C.FS * 3, C.N_CH)))
    for _ in range(r2.seq_len + 5):               # 填滿序列緩衝
        r2.push(block)
    m, p95, mx = _timeit(lambda: r2.push(block), a.n)
    print(f"\n端到端 Recognizer.push()   中位 {m:.3f}ms  p95 {p95:.3f}ms  max {mx:.3f}ms")
    print(f"  → 每秒可處理 {1e3 / max(m, 1e-9):.0f} 步，需求 {1 / C.STEP_SEC:.0f} 步/秒")

    lat = r.latency_budget
    print(f"\n演算法延遲（與計算速度無關，改參數才會變）：")
    print(f"  視窗 {lat['window_s']:.2f}s + 序列脈絡 {lat['seq_context_s']:.2f}s "
          f"+ 投票 {lat['voting_s']:.2f}s = {lat['total_s']:.2f}s")
    print(f"  ⚠️ 這 {lat['total_s']:.2f}s 才是使用者感受到的反應延遲，"
          f"計算只佔 {tot:.2f}ms（{tot / (lat['total_s'] * 1e3):.3%}）。")


if __name__ == "__main__":
    main()
