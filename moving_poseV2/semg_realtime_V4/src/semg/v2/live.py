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
import unicodedata
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

    ser = serial.Serial(port, baud, timeout=1)
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

# cue 事件的中文（只有 --replay 會用到；接板子時沒有 cue 檔）。
# ★ 只有 `go` 是「應該正在維持該動作」，其餘都是過渡或休息——這決定了
#   哪些視窗該打勾/打叉，見 _cue_at 與統計區。
CUE_EVENT_ZH = {
    "record_start": "開始錄",
    "preview": "預告",        # 已顯示下一個姿勢，但還在休息
    "prepare": "準備",         # 正在擺姿勢，還沒開始維持
    "go": "維持中",             # ★ 唯一有明確預期狀態的階段
    "return": "放下",           # 正在移回中性
    "release": "休息",          # 已回到中性
}


def _dw(s: str) -> int:
    """字串的**顯示寬度**：全形字算 2 欄。

    直接用 len() 會讓含中文的欄位對不齊——ljust 是按字元數補的，
    但終端機是按顯示寬度排版的（2026-08-20 的統計文字被殘留字元蓋掉
    也是同一個原因）。
    """
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s: str, width: int) -> str:
    """按顯示寬度靠左補滿到 width 欄。"""
    return s + " " * max(0, width - _dw(s))


def _fmt(res: dict, dof_names: list[str]) -> str:
    parts = []
    for j, dn in enumerate(dof_names):
        st = C.DOF_STATES[dn]
        s = res["stable"][j]
        txt = st[s] if s is not None else "…"
        parts.append(f"{dn}={txt:<6s}({res['conf'][j]:.2f})")
    return "  ".join(parts)


def render(res: dict, dof_names: list[str], wps: float, elapsed: float,
           cue: str | None = None, mark: str = "") -> str:
    """一行即時狀態。

    ⚠️ `win/s` 與關節角度刻意**不顯示**（2026-08-19 使用者要求）：判讀
    「模型預測的動作」對不對時，那兩項只是雜訊，會把真正要看的東西擠掉。
    win/s 仍然有算，結束時的統計區照樣印；關節角度改看控制事件 log。

    `cue`/`mark` 只有 `--replay` 且有 cue 檔時才會有值（接板子時沒有真值可比）。
    """
    line = f"[{elapsed:6.1f}s] "
    if cue is not None:
        line += _pad(cue, 26) + "│ "
    line += _fmt(res, dof_names) + f"  ➜  {res['text']}{mark}"
    return "\r" + _pad(line, 150)


def render_dual(ra: dict, rb: dict, dof_names: list[str], tags: tuple[str, str],
                elapsed: float, agree: float) -> str:
    """兩個模型並排。不一致的自由度用 ▲ 標出來。"""
    diff = "".join("▲" if ra["stable"][j] != rb["stable"][j] else "·"
                   for j in range(len(dof_names)))
    line1 = f"[{elapsed:6.1f}s] {tags[0]}| " + _fmt(ra, dof_names)
    line2 = f"          {tags[1]}| " + _fmt(rb, dof_names) + f"   {diff} 一致率 {agree:.0%}"
    return f"\r{line1:<95s}\n{line2:<95s}\033[1A"


def main() -> None:
    ap = argparse.ArgumentParser(description="即時 sEMG 分類（純螢幕）")
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--bundle-b", type=Path, default=None,
                    help="★ 第二個模型，與第一個**吃同一條資料流**並排比較。"
                         "分兩次跑來比模型沒有意義 —— 實測同一間教室相隔 12 分鐘的"
                         "兩份錄音差 62.7% vs 97.0%，session 與貼片的變異"
                         "遠大於架構差異（1~2pp）。同流比才看得出模型本身。")
    ap.add_argument("--port", default=None, help="序列埠，例如 /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=1_600_000)
    ap.add_argument("--replay", type=Path, default=None,
                    help="用錄好的 CSV 模擬序列埠（沒有硬體時驗證用）")
    ap.add_argument("--replay-fast", action="store_true",
                    help="重播時不照真實速度（只驗證邏輯，不量延遲）")
    ap.add_argument("--calib-sec", type=float, default=C.CUE_LEAD_SEC,
                    help="開機靜止校準秒數")
    # ⚠️ 預設值刻意留 None，等知道是不是接手臂之後才決定（見下方 resolve）：
    #    純螢幕沿用原本的 votes=5 / dwell=0；接手臂時改用 arm_config 那組
    #    （votes=1 交給控制層的 VOTE_M/VOTE_N，避免疊兩層投票疊加延遲；
    #    dwell=6 擋移動過程的穩定誤判，那是投票擋不住的）。
    ap.add_argument("--votes", type=int, default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--dwell", type=int, default=None,
                    help="遲滯：候選新狀態要連續出現 N 次才換過去。"
                         "對付**移動過程**的誤判（投票對付的是抖動，兩者不同）。"
                         "純螢幕預設 0；接手臂時預設 6。每 +1 延遲增加 50 ms。")
    ap.add_argument("--calib-log", type=Path, default=Path("calib_log.csv"),
                    help="每次校準的逐通道尺度存到這裡（只診斷，不改行為）")
    ap.add_argument("--idle-test", type=float, default=0.0,
                    help="靜止誤觸發測試：受試者完全不動 N 秒，統計非 rest 輸出比例")
    # ── 手臂控制（2026-08-19 併入；不給 --arm-ip/--arm-dry-run 就是純螢幕）──
    ap.add_argument("--arm-ip", default=None,
                    help="接上 UFACTORY Lite 6 並用分類結果驅動它。"
                         "不給就是原本的純螢幕模式。")
    ap.add_argument("--arm-dry-run", action="store_true",
                    help="用 MockArm 取代真手臂（指令只印到終端機），不連線")
    ap.add_argument("--arm-yes", action="store_true",
                    help="校準完成後直接 ARMED，不等按 Enter")
    ap.add_argument("--arm-log", type=Path, default=None,
                    help="控制事件 CSV log；預設 results/control_logs/live_<時間戳>.csv")
    ap.add_argument("--force-bad-electrodes", action="store_true",
                    help="電極自檢有 ❌ 時仍然接手臂。⚠️ 校準尺度壞掉會讓該 DOF "
                         "靜默失效或誤觸發，手臂會照著錯的判讀動——只在你確定"
                         "自己在做什麼時才用")
    a = ap.parse_args()

    if (a.port is None) == (a.replay is None):
        raise SystemExit("要指定 --port（接板子）或 --replay（模擬），二選一")

    arm_on = bool(a.arm_ip) or a.arm_dry_run
    if arm_on and a.idle_test:
        raise SystemExit(
            "--idle-test 不能跟手臂一起用：靜止測試的目的是量「人沒有意圖時的"
            "誤觸發率」，而那些誤觸發正是會讓手臂亂動的東西。先純螢幕跑完"
            "--idle-test 確認可接受，再接手臂。")
    if arm_on and a.bundle_b:
        raise SystemExit("手臂只有一支，無法同時照兩個模型動。--bundle-b 請單獨用（純螢幕比較）。")

    # ── 控制層與 votes/dwell 的解析 ──────────────────────────────────
    bridge_mod = None
    if arm_on:
        # control 套件在 moving_poseV2/src/，semg_realtime_V4 是它底下的自足打包，
        # 所以往上兩層再進 src。放在這裡而不是模組層級：純螢幕模式完全不該
        # 依賴控制層（這一包本來就要能獨立下載執行）。
        _v4_root = Path(__file__).resolve().parents[3]        # semg_realtime_V4/
        _ctrl_src = _v4_root.parent / "src"                    # moving_poseV2/src/
        if not (_ctrl_src / "control").is_dir():
            raise SystemExit(
                f"找不到手臂控制層 {_ctrl_src / 'control'}。"
                "--arm-ip/--arm-dry-run 需要完整的 moving_poseV2/（semg_realtime_V4 "
                "單獨下載時沒有控制層），請改用純螢幕模式。")
        sys.path.insert(0, str(_ctrl_src))
        from control import arm_config as AC      # noqa: PLC0415
        AC.validate()
        from control import live_bridge as bridge_mod          # noqa: PLC0415
        # ★ 單層投票：沿用 Recognizer 這層的投票（run_live 原本就有的機制），
        #   控制層那層改成 pass-through（見 arm_config.LIVE_VOTE_M/N）。
        votes = AC.LIVE_RECOGNIZER_VOTES if a.votes is None else a.votes
        conf = AC.RECOGNIZER_CONF if a.conf is None else a.conf
        dwell = AC.DEFAULT_DWELL if a.dwell is None else a.dwell
    else:
        votes = 5 if a.votes is None else a.votes
        conf = 0.6 if a.conf is None else a.conf
        dwell = 0 if a.dwell is None else a.dwell
    a.votes, a.conf, a.dwell = votes, conf, dwell

    r = Recognizer(a.bundle, votes=a.votes, conf=a.conf, dwell=a.dwell)
    rb = (Recognizer(a.bundle_b, votes=a.votes, conf=a.conf, dwell=a.dwell)
          if a.bundle_b else None)
    if rb is not None:
        if rb.seq_len != r.seq_len:
            raise SystemExit(f"兩個模型的 SEQ_LEN 不同（{r.seq_len} vs {rb.seq_len}），"
                             "不能共用特徵序列")
        if len(rb.mean) != len(r.mean):
            raise SystemExit("兩個模型的特徵維度不同，不能比較")
        print(f"模型 A : {a.bundle}")
        print(f"模型 B : {a.bundle_b}")
        print("   ★ 兩個模型吃**同一份特徵**（濾波與特徵只算一次），"
              "所以差異一定來自模型本身。")
    else:
        print(f"模型   : {a.bundle}")
    print(f"通道   : {list(C.CH_NAMES)}")
    print(f"推論   : device={r.dev}  scripted={r.scripted}  "
          f"SEQ_LEN={r.seq_len}  threads={C.CPU_THREADS}")
    lb = r.latency_budget
    print(f"演算法延遲 : 視窗 {lb['window_s']:.2f} + 序列 {lb['seq_context_s']:.2f}"
          f" + 投票 {lb['voting_s']:.2f}"
          + (f" + 遲滯 {lb['dwell_s']:.2f}" if lb.get("dwell_s") else "")
          + f" = **{lb['total_s']:.2f} s**")
    if a.dwell:
        print(f"   遲滯 --dwell {a.dwell}：換狀態要連續 {a.dwell} 次確認"
              f"（對付移動過程的誤判，不是抖動）")
    if arm_on:
        print("=" * 78)
        print("⚠️  維持姿勢時手臂必須懸空 —— 手肘/前臂/上臂不能靠在桌面、扶手或身側，")
        print("    否則重力被外物撐住、EMG 消失，系統會判成 rest 讓機械手臂回 home。")
        print("=" * 78)
        print(f"手臂   : {'MockArm（--arm-dry-run）' if a.arm_dry_run else a.arm_ip}")
        print(f"   單層投票：Recognizer votes={a.votes}、dwell={a.dwell} 是唯一的投票層；"
              "控制層的 VOTE_M/VOTE_N 走 pass-through，")
        print("   但 P_MIN（逐 DOF 信心門檻）與 REFRACTORY（不反應期）照常生效。")

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
    blocks, got = [], 0
    while got < need:
        b = q.get()
        if b is None:
            raise SystemExit("資料來源在校準完成前就結束了")
        blocks.append(b)
        got += len(b)
    calib = np.concatenate(blocks)[:need]
    rows_chk = electrode_check(calib)
    electrodes_ok = print_electrode_check(rows_chk)
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

    # ── 1.5 手臂：校準完成之後才連線 ────────────────────────────────
    # ★ 順序刻意放在校準之後：校準要 5 秒，期間若手臂已經 enable 並回過 home，
    #   使用者的手還在鍵盤旁而手臂已經在動；而且校準失敗（資料來源提早結束）
    #   時會 SystemExit，手臂就會留在 enable 狀態沒人收。
    bridge = None
    if arm_on:
        # ★ 電極自檢有 ❌ 就不准接手臂（2026-08-20 加入）。
        #
        #   為什麼要擋：校準尺度是**整份訊號共用**的，某個通道在校準視窗內
        #   軌到底（削波）時，估出來的尺度會大幾十倍，該通道所有 `*_rel`
        #   振幅特徵被除以那個大數 → 看起來永遠接近零，而 `prop_`/`grp_`
        #   這些跨通道比例還會把災情擴散到其他通道。
        #
        #   實測（multi_gary_20260811_17-11-49，ch4 削波 100%、尺度 54.87 µV
        #   vs 乾淨錄音的 1.65 µV）：go 段一致率 45.5%；換乾淨錄音
        #   （17-23-12）同一套程式是 **100.0%**。也就是說模型沒問題，
        #   壞的是校準——**而且畫面上完全看不出來**，只會覺得「模型很爛」。
        #
        #   純螢幕時只警告就好（研究者可能就是要看壞資料的行為），但接了
        #   手臂之後，錯的判讀會變成手臂真的亂動，必須擋下來。
        if not electrodes_ok and not a.force_bad_electrodes:
            raise SystemExit(
                "\n✖ 電極自檢有 ❌，拒絕接手臂。\n"
                "  校準尺度會因此偏掉幾十倍，該自由度會靜默失效或誤觸發，\n"
                "  而手臂會照著錯的判讀動——畫面上看不出是校準壞掉。\n"
                "  處理方式：\n"
                "    · 接板子時 → 重貼那個通道的電極，確認校準期間完全靜止\n"
                "    · --replay 時 → 換一份乾淨的錄音（這份的開頭空白段被汙染了）\n"
                "  確定要繼續請加 --force-bad-electrodes（不建議）。")
        bridge = bridge_mod.ArmBridge(list(r.dof_names), arm_ip=a.arm_ip,
                                      dry_run=a.arm_dry_run, log_path=a.arm_log)
        if a.arm_yes:
            bridge.engage()
        else:
            print("\n★ 手臂目前 DISARMED，不會動。確認下面的分類與信心值合理之後，")
            print("  按 Enter 進入 ARMED（按下的瞬間手臂會先回一次 home，是正常的）；")
            print("  按 q 結束、按 z 將 J6 歸零。")

    # ── 1.6 cue 對照（只有 --replay 且有 _cues.csv 時）─────────────────
    # ★ 為什麼要這個：光看預測輸出無法判斷「模型現在標的動作對不對」——
    #   尤其手臂在抖時，你需要知道那一刻受試者**被要求做的**是什麼動作。
    #   cue 是錄製當下的指示，是這份錄音唯一的真值來源。
    #   ⚠️ cue 標的是「該做什麼」，不是「實際做了什麼」；轉換期（prepare/
    #   return/preview）在 cue 檔裡是丟棄段，這裡顯示為 "—"，那段預測跟
    #   cue 對不上是正常的，不要當成誤判。
    cue_t = cue_ev = cue_ge = None
    cue_offset = got            # 校準吃掉的樣本數＝分類串流的起點
    if a.replay is not None:
        from semg.v2 import cues as Q                          # noqa: PLC0415
        cue_csv = Q.cue_path_for(a.replay)
        if cue_csv is None:
            print(f"\n[cue] 找不到 {Path(a.replay).stem}_cues.csv，不顯示 cue 對照")
        else:
            # ★ 直接讀 cue 檔的 `gesture` 欄，用 `t_s` 對時（2026-08-20 使用者指正）。
            #   原本走 `Q.sample_labels()`，那是**訓練用**的標籤：它會套用
            #   margin、並把 prepare/return/preview 整段標成丟棄，所以畫面上
            #   有 59% 的時間顯示「—」，看不出當下究竟在 cue 什麼動作。
            #   這裡要的是「受試者此刻被指示做什麼」，那就是 cue 檔的原始內容。
            _cu = Q.load_cues(cue_csv).sort_values("t_s").reset_index(drop=True)
            cue_t = _cu["t_s"].to_numpy(dtype=float)
            cue_ev = _cu["event"].astype(str).tolist()
            cue_ge = _cu["gesture"].astype(str).tolist()
            print(f"\n[cue] 已載入 {cue_csv.name}（{len(cue_t)} 個 cue 事件）"
                  "——畫面並排顯示「cue（此刻要求做什麼）vs 模型預測」。")
            print("      格式：動作（階段）。只有階段是「維持中」時才有明確的")
            print("      正確答案，那時才會標 ✓（相符）或 ✗（不符）並列入準確率；")
            print("      預告/準備 是還沒開始動、放下/休息 是正在放鬆，不標記。")
            print("      對照表：" + "、".join(
                f"{C.ACTION_LABELS_ZH[k]}" for k in
                ("arm_lift", "pick_and_lift", "pick_and_raise")))

    def _cue_at(win_end_sample: int) -> tuple[str, str, bool, float]:
        """該視窗**結束**時刻的 cue。

        回傳 (中文顯示字串, 動作英文名, 是否為 go 段, 進入該 cue 幾秒)。
        動作英文名保留是為了拿去查 `C.action_to_dof()` 做比對——顯示用中文，
        比對用英文，兩者不要混。

        錄音時間 = (校準吃掉的樣本 + 串流位置) / FS，再用 t_s 找出最後一個
        已經發生的 cue 事件。最後一個回傳值用來扣掉 go 段開頭的演算法延遲——
        那段必然對不上，不扣的話統計出來的是「延遲」而不是「準不準」。
        """
        if cue_t is None:
            return "—", "", False, 0.0
        t_rec = (win_end_sample + cue_offset) / C.FS
        i = int(np.searchsorted(cue_t, t_rec, side="right")) - 1
        if i < 0:
            return "—", "", False, 0.0           # 第一個 cue 之前
        ev, ge = cue_ev[i], cue_ge[i]
        since = t_rec - float(cue_t[i])
        ev_zh = CUE_EVENT_ZH.get(ev, ev)
        if ge in ("", "nan", "None"):             # record_start 沒有 gesture
            return f"（{ev_zh}）", "", False, since
        ge_zh = C.ACTION_LABELS_ZH.get(ge, ge)    # 例：arm_lift → 肩前舉 + 肘彎曲
        return f"{ge_zh}（{ev_zh}）", ge, ev == "go", since

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
    last_stable: tuple[int | None, ...] | None = None
    last_stable_b: tuple[int | None, ...] | None = None
    stream_pos = 0                  # 已送進 recognizer 的樣本數（查 cue 用）
    n_cue_cmp = n_cue_hit = 0       # go 段的整體比對統計（三個 DOF 都要對）
    n_cue_dof_cmp = [0] * len(r.dof_names)      # 逐 DOF —— 看是誰在拖後腿
    n_cue_dof_hit = [0] * len(r.dof_names)
    # 「_s」= settled，扣掉 go 段開頭的演算法延遲之後才開始算
    settle_sec = float(r.latency_budget["total_s"])
    n_cue_cmp_s = n_cue_hit_s = 0
    n_cue_dof_cmp_s = [0] * len(r.dof_names)
    n_cue_dof_hit_s = [0] * len(r.dof_names)
    try:
        while True:
            blk = q.get()
            if blk is None:
                break
            t_in = time.perf_counter()
            # 這一塊送進 recognizer 之後的串流位置——用來查 cue（見 _cue_at）。
            # 一塊 = STEP_SAMPLES，通常剛好產出一個視窗，該視窗的結束位置就是
            # 這裡的 stream_pos；一塊產出多個視窗時，後面的比較新。
            stream_pos += len(blk)
            # ★ 特徵只算一次，兩個模型共用（濾波 77% + 特徵 22% 的計算不重複）
            feats = r.windows(blk)
            for w_i, feat in enumerate(feats):
                res = r.classify(feat)
                res_b = rb.classify(feat) if rb else None
                if res is None:
                    continue
                if bridge is not None:
                    # ⚠️ 用 time.monotonic() 而不是上面的 perf_counter——控制層
                    #    內部（看門狗、jog 推進、節流）全部以 monotonic 為基準，
                    #    混用兩個時鐘會讓 watchdog 立刻誤觸發。
                    now_m = time.monotonic()
                    bridge.feed(res, now_m)
                    bridge.tick(now_m)
                n_win += 1
                lat.append((time.perf_counter() - t_in) * 1000.0)
                el = time.perf_counter() - t0
                # 這個視窗結束在哪個樣本：一塊裡越後面的視窗越新
                cue_now, cue_act, cue_is_go, cue_since = _cue_at(
                    stream_pos - (len(feats) - 1 - w_i) * C.STEP_SAMPLES)
                # ✓/✗ 只在 go 段有意義——其餘段落（preview/prepare/return/
                # release）沒有「應該是什麼」的明確答案，打勾打叉都不對。
                cue_mark = ""
                # 只在 go 段統計一致率。
                if cue_is_go:
                    ok_all = cue_act == res["action"]
                    cue_mark = " ✓" if ok_all else " ✗"
                    n_cue_cmp += 1
                    n_cue_hit += ok_all
                    # ★ 再記一份「扣掉 go 開頭 settle 秒」的版本。go 段只有
                    #   CUE_ACTION_SEC(4.0s)，而演算法延遲約 1.2s → 有 30% 的
                    #   視窗是**必然**對不上的。不扣的話量到的是延遲，不是準確度。
                    settled = cue_since >= settle_sec
                    if settled:
                        n_cue_cmp_s += 1
                        n_cue_hit_s += ok_all
                    # 逐 DOF 也記一份——「三個都要對」的整體數字看不出**是誰**
                    # 在拖後腿，而那才是要改的東西。
                    want = C.action_to_dof(cue_act)
                    for j, dn in enumerate(r.dof_names):
                        k = C.DOFS.index(dn)
                        got_j = res["stable"][j]
                        ok_j = (got_j if got_j is not None else 0) == want[k]
                        n_cue_dof_cmp[j] += 1
                        n_cue_dof_hit[j] += ok_j
                        if settled:
                            n_cue_dof_cmp_s[j] += 1
                            n_cue_dof_hit_s[j] += ok_j
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
                    st = tuple(res["stable"])
                    st_b = tuple(res_b["stable"]) if res_b is not None else None
                    if res_b is not None:
                        if (None not in st and None not in st_b) and (st != last_stable or st_b != last_stable_b):
                            last_stable, last_stable_b = st, st_b
                            diff = "".join("▲" if res["stable"][j] != res_b["stable"][j] else "·"
                                           for j in range(len(r.dof_names)))
                            line_a = f"[{el:6.1f}s] ★ {tags[0]}| " + _fmt(res, r.dof_names) + f"  ➜  {res['text']}"
                            line_b = f"         ★ {tags[1]}| " + _fmt(res_b, r.dof_names) + f"  ➜  {res_b['text']}   {diff}"
                            sys.stdout.write(f"\r{line_a:<95s}\n{line_b:<95s}\n")
                        sys.stdout.write(render_dual(
                            res, res_b, r.dof_names, tags, el,
                            sum(agree) / max(n_cmp * len(agree), 1)))
                    else:
                        # ⚠️ 比較的是「未穩定當 rest」之後的組合，不是原始的
                        #    `stable`（2026-08-19 使用者回報）。原本直接比 `st`，
                        #    而某個 DOF 從 0 變成 None 時 `st` 改變了、但畫面上
                        #    兩者都顯示「全部放鬆」——結果是同一行中文標示被
                        #    重印一次，看起來像發生了什麼事，其實沒有。
                        st_f = tuple(0 if s is None else s for s in st)
                        if st_f != last_stable:
                            last_stable = st_f
                            line_change = (f"[{el:6.1f}s] ★ "
                                           + (f"{_pad(cue_now, 26)}│ " if cue_t is not None else "")
                                           + _fmt(res, r.dof_names)
                                           + f"  ➜  {res['text']}{cue_mark}")
                            sys.stdout.write("\r" + _pad(line_change, 150) + "\n")
                        sys.stdout.write(render(
                            res, r.dof_names, n_win / max(el, 1e-9), el,
                            cue_now if cue_t is not None else None, cue_mark))
                    sys.stdout.flush()
            if bridge is not None and not bridge.handle_keys():
                break
            if a.idle_test and time.perf_counter() - t0 >= a.idle_test:
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        # ★ 手臂安全優先於統計輸出：不管迴圈怎麼結束（q、Ctrl-C、資料來源
        #   結束、例外），都要停止 → 回 home → 斷線，不留下 enable 的手臂。
        if bridge is not None:
            print()
            bridge.close()

    el = time.perf_counter() - t0
    # ⚠️ 即時狀態列是用 \r 原地更新、沒有換行，而它含全形字：ljust 是按
    #    **字元數**補的，實際顯示寬度更長。直接 print 統計會蓋不掉尾巴，
    #    殘留的舊字元會跟統計文字混在一起（2026-08-20 使用者回報）。
    #    先用空白把整行清乾淨再輸出。
    sys.stdout.write("\r" + " " * 200 + "\r")
    sys.stdout.flush()
    print(f"\n── 統計 ──   {el:.1f}s、{n_win} 個視窗（{n_win / max(el, 1e-9):.1f} win/s）")
    if lat:
        print(f"每視窗計算耗時：中位 {np.median(lat):.2f} ms、"
              f"p95 {np.percentile(lat, 95):.2f} ms、最大 {max(lat):.2f} ms"
              f"   （預算 {C.STEP_SEC * 1000:.0f} ms）")
    for j, dn in enumerate(r.dof_names):
        tot = sum(counts[j].values()) or 1
        dist = "  ".join(f"{k}={v / tot:.1%}" for k, v in counts[j].most_common())
        print(f"  {dn:<9s} {dist}")

    if n_cue_cmp:
        print(f"\n★ 準確率（與 cue 相符的比例）：{n_cue_hit / n_cue_cmp:.1%}"
              f"　✓ {n_cue_hit} ／ ✗ {n_cue_cmp - n_cue_hit}"
              f"（共 {n_cue_cmp} 個「維持中」視窗，三個自由度都要對）")
        print(f"   ★ 扣掉每個「維持中」開頭 {settle_sec:.2f}s 的演算法延遲之後："
              f"**{n_cue_hit_s / max(n_cue_cmp_s, 1):.1%}**"
              f"（{n_cue_hit_s}/{n_cue_cmp_s}）")
        print(f"     演算法延遲 {settle_sec:.2f}s（視窗＋序列＋投票/dwell），而"
              f"「維持中」只有 {C.CUE_ACTION_SEC:.1f}s。扣掉開頭那段是為了")
        print("     分辨「跟不上」與「判錯」——兩者在上面那個數字裡混在一起。")
        print("     ⚠️ 兩個數字通常很接近：受試者在「準備」階段就開始擺姿勢了，")
        print("        所以「維持中」一開始模型往往已經跟上。差距大才代表延遲是瓶頸。")
        print("   逐 DOF —— 整體數字被最差的那個 DOF 拉低（左=全 go 段，右=扣延遲後）：")
        for j, dn in enumerate(r.dof_names):
            c, h = n_cue_dof_cmp[j], n_cue_dof_hit[j]
            cs, hs = n_cue_dof_cmp_s[j], n_cue_dof_hit_s[j]
            print(f"     {dn:<9s} {h / max(c, 1):5.1%}   →  {hs / max(cs, 1):5.1%}")
        print("   只統計「維持中」的區段——預告/準備 還沒開始動、放下/休息 正在")
        print("   放鬆，那些段落沒有明確的正確答案。")
        print("   ⚠️ cue 是「要求做什麼」不是「實際做了什麼」：受試者提早動、")
        print("      晚放鬆、或根本沒照做，都會算成 ✗。")

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
