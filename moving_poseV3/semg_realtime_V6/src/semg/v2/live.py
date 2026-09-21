"""即時分類（純螢幕，不接機械手臂）。

    序列埠 → find_sync/read_one_frame → Recognizer.push() → 終端機顯示

★ 為什麼先做純螢幕版，不直接接 Lite 6：
  目前**所有**準確率數字都來自照腳本做的 cue 錄音 —— 受試者跟著提示做、
  rest 是「已經沉降的 rest」、轉換過程被 margin 切掉了。
  即時操作完全不是這樣：人隨時想動就動，坐著不動時還會抓癢、調整姿勢。
  **靜止時的誤觸發率**是接上真手臂之前唯一還沒量過的數字，
  而它量不到於任何離線資料上 —— 只能開即時迴路量。
  手部 DOF 誤判只是夾爪亂開合，但 elbow / shoulder 誤判會讓真的手臂動起來。

★ 前處理與投票**完全重用** `semg.v2.infer.Recognizer` ——
  不要在這裡另外實作一套。v1 就是因為前處理有兩份實作
  （source.py 與 run_live.py）而出過「只改了一邊」的錯。

用法：
    # 接板子
    PYTHONPATH=src python -m semg.v2.live --bundle <model.pt> --port /dev/ttyUSB0

    # 沒有板子時用錄好的 CSV 模擬序列埠（先驗證顯示與統計再接硬體）
    PYTHONPATH=src python -m semg.v2.live --bundle <model.pt> --replay <錄音.csv>

    # 靜止誤觸發測試：完全不動 60 秒，統計非 rest 的輸出比例
    PYTHONPATH=src python -m semg.v2.live --bundle <model.pt> --port /dev/ttyUSB0 \\
        --idle-test 60
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

from semg.v2 import config as C
from semg.v2 import preprocess as P
from semg.v2.infer import Recognizer

# ⚠️ Windows 主控台預設 cp950（繁中），印不出程式裡的 ✅ ⚠️ ★ 這些字元，
#    會直接丟 UnicodeEncodeError 把整個程式打斷 —— 而且是在跑到一半才炸。
#    .bat 有設 chcp 65001，但使用者直接 `python -m semg.v2.live` 就沒有，
#    所以在程式裡自己防禦。errors="replace" 讓真的無法顯示時印 ? 而不是崩潰。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# `hardware.record_session` 在 src/ 底下，與 semg/ 平行。
# 平常靠 PYTHONPATH=src 就找得到，這行是沒設環境變數時的保險。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── 電極自檢門檻 ──────────────────────────────────────────────────────────
# ⚠️ 2026-08-13 修正：先前用「|x| > 50 mV 的比例」當飽和率，**那是錯的**。
#    ADS1299 的訊號帶著各通道各自的直流偏移，50 mV 這種絕對門檻量到的是
#    **偏移**而不是飽和。實際踩到：15-50-34 的 ch4 被判定「整份 100% 飽和」，
#    但它其實只是穩定坐在 +71.6 mV（滿刻度的 12.7%，p1~p99 只跨 2.6 mV），
#    帶通一濾就乾淨了 —— 那份錄音的 ch4 在 380s 之前 shoulder_flex 調變 31.9×，
#    完全健康。真正的判準有兩個，而且要分開看：
#      1. **真的軌到底**：|raw| 接近滿刻度 ±562 mV → ADC 削波，肌電被削掉
#      2. **濾波後基線太高**：帶通之後仍然吵 → 接觸阻抗高或環境干擾
FULL_SCALE_MV = 2.0 * C.ADS_VREF / C.ADS_GAIN * 1e3 / 2      # ±562 mV
CLIP_MV = 0.89 * FULL_SCALE_MV        # 接近滿刻度才算削波
CLIP_WARN, CLIP_BAD = 0.001, 0.01     # 削波比例 0.1% / 1%
BASE_WARN_UV = 15.0                   # 濾波後基線 RMS > 15 µV → 偏高
BASE_BAD_UV = 40.0                    # > 40 µV → 這個通道大概沒法用
OFFSET_NOTE_MV = 50.0                 # 直流偏移超過這個值只是**提示**，不是問題


# ─────────────────────────────────────────────────────────────────────────────
# 取樣來源：真的序列埠，或用 CSV 重播模擬
# ─────────────────────────────────────────────────────────────────────────────

def serial_source(port: str, baud: int, q: queue.Queue, stop: threading.Event):
    """讀序列埠，把 (n, 7) mV 的資料塊丟進 queue。

    ⚠️ 解幀邏輯直接從 `hardware.record_session` import，**不複製一份** ——
       幀格式改了只該改一個地方。
    """
    import serial
    from hardware.record_session import find_sync, read_one_frame

    # ★ 開埠失敗要**立刻**讓主執行緒知道。舊版是讓例外死在這個執行緒裡，
    #   主執行緒只會傻等到逾時 —— 使用者看到的是「卡住」而不是「權限不足」。
    try:
        ser = serial.Serial(port, baud, timeout=1)
    except Exception as e:                      # noqa: BLE001
        q.put(("__error__", f"開不了 {port}：{e}"))
        return
    ser.reset_input_buffer()                  # 丟掉開機前積在 buffer 的舊資料
    buf: list[list[int]] = []
    n_bad = 0
    while not stop.is_set():
        if not find_sync(ser):
            continue
        fr = read_one_frame(ser)
        if fr is None:
            n_bad += 1
            continue
        _, ch = fr
        buf.append([ch[c - 1] for c in C.CH_CSV_COLS])
        if len(buf) >= C.STEP_SAMPLES:
            q.put(P.adc_to_mv(np.asarray(buf, dtype=np.float64)))
            buf = []
    ser.close()
    q.put(None)


def replay_source(csv: Path, q: queue.Queue, stop: threading.Event,
                  realtime: bool = True):
    """用錄好的 CSV 模擬序列埠。給沒有硬體時驗證顯示與統計用。"""
    raw = P.load_recording(csv, trim=False)
    step = C.STEP_SAMPLES
    dt = step / C.FS
    t_next = time.perf_counter()
    for s in range(0, len(raw) - step + 1, step):
        if stop.is_set():
            break
        q.put(raw[s:s + step])
        if realtime:                          # 照真實速度餵，才量得到真的延遲
            t_next += dt
            slack = t_next - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
    q.put(None)


# ─────────────────────────────────────────────────────────────────────────────
# 電極自檢
# ─────────────────────────────────────────────────────────────────────────────

def electrode_check(raw: np.ndarray) -> list[dict]:
    """對一段靜止資料做逐通道健檢。`raw` 是**未去直流**的 mV。

    量三個獨立的東西，因為它們是不同的故障：
      offset_mv : 直流偏移。**本身不是問題**（帶通會濾掉），只當提示
      clip      : |x| 接近滿刻度的比例 → ADC 削波，肌電被削掉，無解
      base_uv   : **濾波後**的靜止基線 RMS → 接觸阻抗與環境干擾的實際指標
    """
    filt = P.filter_causal(raw - raw.mean(axis=0))
    out = []
    for i, name in enumerate(C.CH_NAMES):
        x, fx = raw[:, i], filt[:, i]
        off = float(np.median(x))
        clip = float((np.abs(x) > CLIP_MV).mean())
        base = float(np.sqrt((fx ** 2).mean()) * 1000.0)      # µV
        if clip > CLIP_BAD:
            status, note = "❌", f"ADC 削波 {clip:.1%}，肌電被削掉"
        elif base > BASE_BAD_UV:
            status, note = "❌", "濾波後基線過高，重貼"
        elif clip > CLIP_WARN:
            status, note = "⚠️", "偶有削波"
        elif base > BASE_WARN_UV:
            status, note = "⚠️", "基線偏高（接觸阻抗或環境干擾）"
        else:
            status, note = "✅", ""
        if not note and abs(off) > OFFSET_NOTE_MV:
            note = f"（直流偏移 {off:+.0f} mV，帶通會濾掉，不影響）"
        out.append({"ch": i + 1, "name": name, "clip": clip, "offset_mv": off,
                    "base_uv": base, "status": status, "note": note})
    return out


def print_electrode_check(rows: list[dict]) -> bool:
    print("\n電極自檢（靜止段）")
    print(f"  {'ch':<4}{'肌肉':<18}{'濾波後基線 µV':>15}{'削波':>8}{'直流 mV':>10}   狀態")
    ok = True
    for r, zh in zip(rows, C.CH_LABELS_ZH):
        print(f"  {r['ch']:<4}{zh:<18}{r['base_uv']:>15.2f}{r['clip']:>7.2%}"
              f"{r['offset_mv']:>10.1f}   {r['status']} {r['note']}")
        if r["status"] == "❌":
            ok = False
    if not ok:
        print("\n  ❌ 有通道不可用 —— 建議**重貼電極再錄**。")
        print("     訊號品質與表現高度相關：同一間教室、相隔 12 分鐘的兩份錄音，")
        print("     全對率差了 62.7% vs 97.0%。")
        print("     ⚠️ 先前把差異歸因於「飽和率」，那個指標後來證實是量到直流偏移，")
        print("        真正的成因待用濾波域指標重新確認。")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# 顯示
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(res: dict, dof_names: list[str]) -> str:
    parts = []
    for j, dn in enumerate(dof_names):
        st = C.DOF_STATES[dn]
        s = res["stable"][j]
        txt = st[s] if s is not None else "…"
        parts.append(f"{dn}={txt:<6s}({res['conf'][j]:.2f})")
    return "  ".join(parts)


def render(res: dict, dof_names: list[str], wps: float, elapsed: float) -> str:
    return f"\r[{elapsed:6.1f}s] " + _fmt(res, dof_names) + f"   {wps:4.1f} win/s "


def render_dual(ra: dict, rb: dict, dof_names: list[str], tags: tuple[str, str],
                elapsed: float, agree: float) -> str:
    """兩個模型並排。不一致的自由度用 ▲ 標出來。"""
    diff = "".join("▲" if ra["stable"][j] != rb["stable"][j] else "·"
                   for j in range(len(dof_names)))
    return (f"\r[{elapsed:6.1f}s] {tags[0]}| {_fmt(ra, dof_names)}\n"
            f"          {tags[1]}| {_fmt(rb, dof_names)}   {diff} 一致率 {agree:.0%}\033[1A")


def main() -> None:
    ap = argparse.ArgumentParser(description="即時 sEMG 分類（純螢幕）")
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--bundle-b", type=Path, default=None,
                    help="★ 第二個模型，與第一個**吃同一條資料流**並排比較。"
                         "分兩次跑來比模型沒有意義 —— 實測同一間教室相隔 12 分鐘的"
                         "兩份錄音差 62.7%% vs 97.0%%，session 與貼片的變異"
                         "遠大於架構差異（1~2pp）。同流比才看得出模型本身。")
    ap.add_argument("--port", default=None, help="序列埠，例如 /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=1_600_000)
    ap.add_argument("--replay", type=Path, default=None,
                    help="用錄好的 CSV 模擬序列埠（沒有硬體時驗證用）")
    ap.add_argument("--replay-fast", action="store_true",
                    help="重播時不照真實速度（只驗證邏輯，不量延遲）")
    ap.add_argument("--calib-sec", type=float, default=C.CUE_LEAD_SEC,
                    help="開機靜止校準秒數")
    ap.add_argument("--votes", type=int, default=5)
    # ⚠️ 幾乎確定不該用，加這個旗標是為了讓你**自己量**而不是聽我說。
    #    實測拆解（每視窗 7.52 ms，lms+nf）：
    #        串流濾波 1.58 / 特徵抽取 ~5.2 / 標準化+序列 0.26  ← 全是 numpy/scipy，上不了 GPU
    #        純模型前向 0.75 ms                                  ← 只有這段能上
    #    也就是模型只佔 10%。就算 GPU 把它變成 0，端到端 1500 ms 只省 0.05%。
    #    而 batch=1、17,530 參數的模型在 GPU 上，kernel launch 與主機↔裝置傳輸
    #    的固定開銷通常比 0.75 ms 的運算還大 → 大概率更慢，而且**抖動變大**。
    #    即時控制怕抖動不怕平均（選 tiny 而非 zhao 的理由之一就是抖動小一半）。
    # ⚠️ 同樣是「給你自己量」的旗標。實測（4 核機器，lms+nf）：
    #      threads=1  中位 7.68 ms   抖動 0.37
    #      threads=2  中位 7.76 ms   抖動 0.41   ← 預設
    #      threads=3  中位 7.90 ms   抖動 4.76   最大 148 ms(!)
    #      threads=4  中位 **137 ms**（兩次重測 137.96 / 136.89）← 慢 17 倍
    #    原因：每次呼叫的工作量很小（100×7 的陣列），執行緒同步開銷遠大於運算，
    #    核心數用滿時還會跟收樣本的執行緒搶 CPU → 掉樣本。
    #    ★ 加執行緒**不會**變快，而且可能災難性地變慢。
    ap.add_argument("--threads", type=int, default=C.CPU_THREADS,
                    help=f"CPU 執行緒數（預設 {C.CPU_THREADS}）。"
                         "⚠️ 實測加到核心數會慢 17 倍（執行緒同步開銷 + 跟收樣本搶 CPU）。"
                         "1~2 最好，別調高")
    ap.add_argument("--device", default="cpu",
                    help="推論裝置（cpu / xpu / cuda）。⚠️ 預設 cpu，"
                         "模型前向只佔每視窗計算的 10%%，換 GPU 省不到端到端的 0.1%%，"
                         "而且 batch=1 小模型在 GPU 上通常更慢、抖動更大。"
                         "加這個旗標是給你自己量用的")

    # ── 環境雜訊抑制（2026-09-12 加入）─────────────────────────────
    # ★ 錄訓練資料的地方安靜，機械手臂放在吵的實驗室。這些演算法的目標是
    #   把「吵環境的訊號」映射回「乾淨環境的樣子」，讓乾淨資料訓練的模型
    #   仍然適用。每個都通過了「乾淨訊號中性測試」才留下來，
    #   詳見 semg/v2/denoise.py 檔頭與 docs/denoise.md。
    ap.add_argument("--denoise", default=None, metavar="a,b,c",
                    help="降噪演算法，逗號分隔：comb(諧波梳狀陷波) / "
                         "lms(自適應市電消除) / car(分組共同平均參考) / "
                         "hampel(突波抑制) / wavelet(小波軟閾值，實驗性)。"
                         "例：--denoise comb,car")
    for _a, _h in (("comb", "60Hz 諧波梳狀陷波（120/180/…/420Hz）"),
                   ("lms", "自適應市電消除（不挖頻譜洞、會追漂移）"),
                   ("car", "分組共同平均參考（去共模環境場）"),
                   ("wavelet", "小波軟閾值去噪（實驗性，最貴）")):
        ap.add_argument(f"--{_a}", action="store_true", help=_h)
    ap.add_argument("--hampel", action="store_true",
                    help="突波抑制（繼電器/馬達切換的脈衝）")
    ap.add_argument("--nf", "--noise-floor", dest="nf", action="store_true",
                    help="★ 雜訊底噪補償：校準時把環境雜訊從尺度 α 扣掉。"
                         "吵環境下 α 被墊高會讓所有相對特徵被壓扁 → "
                         "該自由度不觸發。與 comb/lms 搭配使用")
    ap.add_argument("--bias", default="0",
                    help="★ 逐 DOF 的**靈敏度偏移**（logit）。正 = 更容易觸發、"
                         "負 = 更難。`--bias elbow=0.7,shoulder=-0.5`。"
                         "⚠️ 這才是有效的靈敏度旋鈕 —— `--conf` 對這個模型沒用，"
                         "因為輸出信心飽和在 1.00（正確與錯誤時都是）。")
    ap.add_argument("--conf", default="0.6",
                    help="信心門檻。可給單一值，或**逐 DOF**："
                         "`--conf hand=0.6,elbow=0.45,shoulder=0.75`。"
                         "★ 調低 = 該 DOF 更容易觸發（治「不敏感」）；"
                         "調高 = 更難觸發（治「誤觸發」）。不用重訓。")
    ap.add_argument("--dwell", default="0",
                    help="遲滯：候選新狀態要連續出現 N 次才換過去。"
                         "對付**移動過程**的誤判（投票對付的是抖動，兩者不同）。"
                         "預設 0 = 現行行為。每 +1 延遲增加 50 ms。"
                         "同樣支援逐 DOF：`--dwell hand=5,elbow=0,shoulder=0`。"
                         "★ 治「捏的時候先判成 index」——那個中間狀態只有 100~200ms，"
                         "dwell 5 就擋得掉，而且只影響 hand。")
    ap.add_argument("--calib-log", type=Path, default=Path("calib_log.csv"),
                    help="每次校準的逐通道尺度存到這裡（只診斷，不改行為）")
    ap.add_argument("--idle-test", type=float, default=0.0,
                    help="靜止誤觸發測試：受試者完全不動 N 秒，統計非 rest 輸出比例")
    a = ap.parse_args()


    if (a.port is None) == (a.replay is None):
        raise SystemExit("要指定 --port（接板子）或 --replay（模擬），二選一")

    # ★ 逐 DOF 的 conf/dwell 必須以**模型的** dof_names 為準，不是 config。
    #   V5 的模型有 4 個頭（多一個 `moving`），而 config 預設只有 3 個 ——
    #   用 config 會產生長度不符的 list，MultiVoter 直接 IndexError。
    #   模型檔記了 dof_names，那才是這個模型實際的樣子。
    def _per_dof(val, cast, default, names):
        """`0.6` → 每個 DOF 都用它；`hand=0.6,elbow=0.45` → 逐 DOF（沒提到的用 default）。"""
        if "=" not in str(val):
            return [cast(val)] * len(names)
        out = dict.fromkeys(names, default)
        for part in str(val).split(","):
            k, _, v = part.partition("=")
            k = k.strip()
            if k not in out:
                raise SystemExit(f"未知的自由度 {k!r}，這個模型有：{names}")
            out[k] = cast(v)
        return [out[n] for n in names]

    import torch as _t
    _names = [str(v) for v in _t.load(a.bundle, map_location="cpu",
                                      weights_only=False)["dof_names"]]
    a.conf = _per_dof(a.conf, float, 0.6, _names)
    a.dwell = _per_dof(a.dwell, int, 0, _names)
    a.bias = _per_dof(a.bias, float, 0.0, _names)

    from semg.v2.denoise import parse_flags, ALGOS
    dn = parse_flags(a.denoise)
    for _a in ALGOS:                       # 個別旗標與 --denoise 是「或」的關係
        if getattr(a, _a, False):
            dn[_a] = True
    if dn or a.nf:
        print(f"降噪   : {', '.join(sorted(dn)) or '（無）'}"
              f"{'  + 雜訊底噪補償(nf)' if a.nf else ''}")
    if a.device != "cpu":
        print(f"⚠️ 推論裝置 = {a.device}（非預設）。請用 --idle-test 或重播比較"
              f"**中位與 p95** 再決定要不要留著 —— 抖動比平均重要。")
    r = Recognizer(a.bundle, votes=a.votes, conf=a.conf, dwell=a.dwell, bias=a.bias,
                   denoise=dn or None, noise_floor=a.nf, device=a.device,
                   threads=a.threads)
    rb = (Recognizer(a.bundle_b, votes=a.votes, conf=a.conf, dwell=a.dwell, bias=a.bias)
          if a.bundle_b else None)
    if rb is not None and list(rb.dof_names) != list(r.dof_names):
        raise SystemExit(f"兩個模型的自由度不同（{list(r.dof_names)} vs "
                         f"{list(rb.dof_names)}），不能並排比較")
    if rb is not None:
        if rb.seq_len != r.seq_len:
            raise SystemExit(f"兩個模型的 SEQ_LEN 不同（{r.seq_len} vs {rb.seq_len}），"
                             "不能共用特徵序列")
        if len(rb.mean) != len(r.mean):
            raise SystemExit("兩個模型的特徵維度不同，不能比較")
        # ★ 共用特徵的前提是「兩邊的前處理相同」。這兩個旗標一旦不同，
        #   B 就會吃到用 A 的設定算出來的特徵 —— 而且**不會報錯**，
        #   只會安靜地表現變差，然後被誤讀成「B 這個模型比較爛」。
        if rb.use_hampel != r.use_hampel:
            raise SystemExit(
                f"兩個模型的 Hampel 設定不同（A={r.use_hampel} / "
                f"B={rb.use_hampel}）。共用特徵時只會套用 A 的設定，B 會吃到"
                "錯的前處理，比出來的差異是假的。\n"
                "  → 請改用兩個 Hampel 設定相同的模型。")
        if getattr(rb, "feat_version", 1) != getattr(r, "feat_version", 1):
            raise SystemExit(
                f"兩個模型的 feat_version 不同（A={r.feat_version} / "
                f"B={rb.feat_version}），特徵定義不一樣，不能共用。")
        print(f"模型 A : {a.bundle}")
        print(f"模型 B : {a.bundle_b}")
        print("   ★ 兩個模型吃**同一份特徵**（濾波與特徵只算一次），"
              "所以差異一定來自模型本身。")
    else:
        print(f"模型   : {a.bundle}")
    print(f"通道   : {list(C.CH_NAMES)}")
    print(f"推論   : device={r.dev}  scripted={r.scripted}  "
          f"SEQ_LEN={r.seq_len}  threads={a.threads}  device={a.device}")
    lb = r.latency_budget
    print(f"演算法延遲 : 視窗 {lb['window_s']:.2f} + 序列 {lb['seq_context_s']:.2f}"
          f" + 投票 {lb['voting_s']:.2f}"
          + (f" + 遲滯 {lb['dwell_s']:.2f}" if lb.get("dwell_s") else "")
          + f" = **{lb['total_s']:.2f} s**")
    dw = a.dwell if isinstance(a.dwell, list) else [a.dwell] * len(r.dof_names)
    cf = a.conf if isinstance(a.conf, list) else [a.conf] * len(r.dof_names)
    if any(dw) or len(set(cf)) > 1 or any(a.bias):
        print("   逐自由度設定：")
        for dn, c_, d_, b_ in zip(r.dof_names, cf, dw, a.bias):
            print(f"     {dn:<9s} bias={b_:+.2f}  conf={c_:.2f}  dwell={d_}"
                  + (f"（+{d_*50} ms 延遲）" if d_ else ""))

    q: queue.Queue = queue.Queue(maxsize=200)
    stop = threading.Event()
    if a.port:
        th = threading.Thread(target=serial_source,
                              args=(a.port, a.baud, q, stop), daemon=True)
    else:
        th = threading.Thread(target=replay_source,
                              args=(a.replay, q, stop, not a.replay_fast),
                              daemon=True)
    th.start()

    # ── 1. 靜止校準 + 電極自檢 ────────────────────────────────────────
    # ⚠️ 校準必須用**開頭的純靜止段**，與 dataset.py 訓練時的來源一致。
    #    用別的區段估尺度會系統性差 1.27×，實測全對率 99.6% → 58%。
    need = int(a.calib_sec * C.FS)
    print(f"\n請保持靜止 {a.calib_sec:.0f} 秒（校準 + 電極自檢）…")
    # ★ 一定要有進度與逾時。舊版是 `q.get()` 無限等 ——
    #   序列埠收不到資料時 `find_sync` 會一直 continue，畫面就卡在上一行，
    #   **完全沒有任何線索**。使用者實際踩到了。
    blocks, got = [], 0
    t_wait = time.perf_counter()
    warned = False
    while got < need:
        try:
            b = q.get(timeout=1.0)
        except queue.Empty:
            el = time.perf_counter() - t_wait
            if not warned and el > 3:
                print(f"\r   ⚠️ 等了 {el:.0f} 秒還沒收到資料 …", end="", flush=True)
                warned = True
            if el > 15:
                raise SystemExit(
                    f"\n\n15 秒內收不到任何有效資料（已收 {got}/{need} 樣本）。\n"
                    f"   先跑探測模式看是哪一層的問題：\n"
                    f"     python src/hardware/record_session.py "
                    f"--port {a.port} --probe\n"
                    f"   它會分辨「埠選錯」「baud 不對」「板子沒在送」「接線/供電」。\n"
                    f"   也可以先列出所有埠：python list_ports.py")
            continue
        if isinstance(b, tuple) and b and b[0] == "__error__":
            raise SystemExit(f"\n\n{b[1]}\n"
                             f"   先列出可用的埠：python list_ports.py")
        if b is None:
            raise SystemExit("資料來源在校準完成前就結束了")
        blocks.append(b)
        got += len(b)
        # 進度節流成 10 次，否則每 100 樣本印一行會洗版
        if got % max(need // 10, 1) < len(b):
            print(f"\r   收到 {got}/{need} 樣本（{got/need:.0%}）", end="", flush=True)
    print(f"\r   收到 {need}/{need} 樣本（100%）")
    calib = np.concatenate(blocks)[:need]
    rows_chk = electrode_check(calib)
    print_electrode_check(rows_chk)
    r.calibrate(calib)
    print(f"\n校準完成。逐通道尺度：")
    for n, v in zip(C.CH_NAMES, r.scales):
        print(f"     {n:<20s} {v * 1000:7.2f} µV")

    # ★ 把每次校準的尺度存起來（只診斷，**不改變任何行為**）。
    #
    #   為什麼：使用者回報「每次 run_live 準確率差很多，有時 shoulder 完全不觸發」。
    #   校準算的是每個通道的穩健振幅尺度（MAD），而 120 維裡所有 `*_rel`
    #   都要除以它 —— 校準段若不夠安靜，該通道尺度被高估 → 相對特徵被壓扁
    #   → 那個 DOF 不觸發。shoulder 對應 ch4，症狀吻合。
    #
    #   但目前**還不知道實際變異有多大**，所以先累積資料再決定門檻，
    #   不要憑猜測加一個會擋掉正常錄音的檢查。
    log = a.calib_log
    log.parent.mkdir(parents=True, exist_ok=True)
    new = not log.exists()
    with log.open("a", encoding="utf-8") as fh:
        if new:
            fh.write("time,bundle," + ",".join(f"scale_uv_{n}" for n in C.CH_NAMES)
                     + "," + ",".join(f"base_uv_{n}" for n in C.CH_NAMES) + "\n")
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S") + f",{a.bundle.stem},"
                 + ",".join(f"{v * 1000:.3f}" for v in r.scales) + ","
                 + ",".join(f"{x['base_uv']:.3f}" for x in rows_chk) + "\n")
    n_prev = sum(1 for _ in log.open(encoding="utf-8")) - 1
    print(f"     → 已記錄到 {log}（累積 {n_prev} 次校準）")
    if n_prev >= 3:
        import csv as _csv
        with log.open(encoding="utf-8") as fh:
            hist = [row for row in _csv.DictReader(fh)]
        print("\n   逐通道尺度的歷史變異（跨 "
              f"{len(hist)} 次校準，最大/最小）：")
        for n in C.CH_NAMES:
            v = [float(h[f"scale_uv_{n}"]) for h in hist]
            print(f"     {n:<20s} {min(v):6.2f} – {max(v):6.2f} µV"
                  f"   （{max(v) / max(min(v), 1e-9):.1f}×）")
        print("   ⚠️ 這只是紀錄，還沒有任何門檻。倍率若很大，"
              "就是「每次準確率差很多」的來源。")

    # ── 2. 即時分類 ──────────────────────────────────────────────────
    mode = "靜止誤觸發測試" if a.idle_test else "即時分類"
    print(f"\n── {mode} ──   Ctrl-C 結束")
    if a.idle_test:
        print(f"   請**完全不動** {a.idle_test:.0f} 秒。")
    # ⚠️ 這個組合會給出**看起來很嚇人但沒有意義**的數字：重播的是照 cue
    #    做動作的錄音，受試者一直在動，模型正確偵測到動作卻被算成「誤觸發」。
    #    實測過一次 49.88%，差點被當成真的誤觸發率。
    if a.idle_test and a.replay:
        print("\n   ⚠️⚠️ 你正在對**錄好的檔案**做靜止測試。")
        print("        如果那份錄音裡受試者有在動（cue 錄音都會），")
        print("        算出來的「誤觸發率」其實是「模型正確偵測到動作的比例」，")
        print("        **沒有意義**。真正的誤觸發率必須接板子、人真的不動才量得到。")
    t0 = time.perf_counter()
    n_win = 0
    counts = [Counter() for _ in r.dof_names]
    counts_b = [Counter() for _ in r.dof_names] if rb else None
    agree = [0] * len(r.dof_names)      # 兩模型輸出相同的次數（逐 DOF）
    n_cmp = 0
    lat: list[float] = []
    tags = (a.bundle.stem[:12].ljust(12), a.bundle_b.stem[:12].ljust(12)) if rb else ("", "")
    try:
        while True:
            blk = q.get()
            if isinstance(blk, tuple) and blk and blk[0] == "__error__":
                print(f"\n{blk[1]}")
                break
            if blk is None:
                break
            t_in = time.perf_counter()
            # ★ 特徵只算一次，兩個模型共用（濾波 77% + 特徵 22% 的計算不重複）
            for feat in r.windows(blk):
                res = r.classify(feat)
                res_b = rb.classify(feat) if rb else None
                if res is None:
                    continue
                n_win += 1
                lat.append((time.perf_counter() - t_in) * 1000.0)
                el = time.perf_counter() - t0
                for j, s in enumerate(res["stable"]):
                    counts[j][C.DOF_STATES[r.dof_names[j]][s] if s is not None
                              else "(none)"] += 1
                if res_b is not None:
                    n_cmp += 1
                    for j, s in enumerate(res_b["stable"]):
                        counts_b[j][C.DOF_STATES[r.dof_names[j]][s] if s is not None
                                    else "(none)"] += 1
                        if res["stable"][j] == s:
                            agree[j] += 1
                if not a.idle_test:
                    if res_b is not None:
                        sys.stdout.write(render_dual(
                            res, res_b, r.dof_names, tags, el,
                            sum(agree) / max(n_cmp * len(agree), 1)))
                    else:
                        sys.stdout.write(render(res, r.dof_names,
                                                n_win / max(el, 1e-9), el))
                    sys.stdout.flush()
            if a.idle_test and time.perf_counter() - t0 >= a.idle_test:
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()

    el = time.perf_counter() - t0
    print(f"\n\n── 統計 ──   {el:.1f}s、{n_win} 個視窗（{n_win / max(el, 1e-9):.1f} win/s）")
    if lat:
        print(f"每視窗計算耗時：中位 {np.median(lat):.2f} ms、"
              f"p95 {np.percentile(lat, 95):.2f} ms、最大 {max(lat):.2f} ms"
              f"   （預算 {C.STEP_SEC * 1000:.0f} ms）")
    for j, dn in enumerate(r.dof_names):
        tot = sum(counts[j].values()) or 1
        dist = "  ".join(f"{k}={v / tot:.1%}" for k, v in counts[j].most_common())
        print(f"  {dn:<9s} {dist}")

    if rb is not None:
        print(f"\n── 模型 B（{a.bundle_b.name}）──")
        for j, dn in enumerate(r.dof_names):
            tot = sum(counts_b[j].values()) or 1
            dist = "  ".join(f"{k}={v / tot:.1%}" for k, v in counts_b[j].most_common())
            print(f"  {dn:<9s} {dist}")
        print(f"\n★ 兩模型輸出一致率（{n_cmp} 個共同視窗）")
        for j, dn in enumerate(r.dof_names):
            print(f"     {dn:<9s} {agree[j] / max(n_cmp, 1):.1%}")
        print(f"     {'整體':<9s} {sum(agree) / max(n_cmp * len(agree), 1):.1%}")
        print("\n   ⚠️ 一致率高**不代表兩者都對**，只代表它們犯一樣的錯或都對。")
        print("      沒有 cue 的即時流沒有真值，所以這裡量的是**分歧程度**，不是準確率。")
        print("      要比準確率請看離線的 LORO（results/arch_zhao/）。")

    if a.idle_test:
        print("\n★ 靜止誤觸發率（非 rest 且非 (none) 的輸出比例）")
        for tag, cc in ([("A", counts)] + ([("B", counts_b)] if rb else [])):
            worst = 0.0
            for j, dn in enumerate(r.dof_names):
                tot = sum(cc[j].values()) or 1
                bad = sum(v for k, v in cc[j].items() if k not in ("rest", "(none)"))
                worst = max(worst, bad / tot)
                print(f"     {tag} {dn:<9s} {bad / tot:.2%}")
            print(f"     {tag} 最差的一個 DOF：{worst:.2%}\n")
        print("   ★ 這個數字決定能不能接機械手臂 —— 誤觸發會讓手臂在使用者沒有意圖時動作。")
        print("     **也是這個 A/B 比較最有意義的指標**：靜止時人沒有意圖，")
        print("     所以「輸出非 rest」一定是錯的 —— 這是即時流上唯一有真值的情況。")


if __name__ == "__main__":
    main()
