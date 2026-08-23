"""ADS1299 錄製 + 同步 cue：先標註，再跟著標註做。

沿用 ads1299_readerV6.py 完全相同的讀取與存檔方式（40-byte frame、SYNC 0xA5 0x5A、
"<I8i" body、XOR checksum、END 0x0D、baud 1,600,000、CSV 寫 8 通道 + status），
但解決舊流程的四個問題：

問題 1：不知道什麼時候真的開始存資料
    舊版 `t_start = perf_counter()` 在 plt.show() 之前就設好，而第一次 update()
    callback 何時觸發取決於 matplotlib 開窗速度（每次不一樣）。而且 UART buffer
    裡可能積了開機前的舊資料。
  → 本版：(a) 明確 reset_input_buffer() 丟掉舊資料
          (b) 丟棄最初 CUE_WARMUP_DISCARD_SEC 的樣本（硬體/UART 未穩定）
          (c) **t0 定義為第一個被接受的樣本**，之後每個樣本都有確定的 sample_index
          (d) 讀取搬到獨立 thread，不再綁在繪圖 callback 上（不會因為畫圖卡住掉樣本）

問題 2：不知道自己什麼時候該做動作
  → terminal 顯示倒數與當前指示，並用 terminal bell 提示。
    先印出指示 → 受試者再跟著做（「先標註再跟著標註做」）。

問題 3：cue 與資料對不齊
  → cue 記錄的是**當下的 sample_index**，不是牆上時鐘。取樣率固定 2000 Hz，
    sample_index 是唯一無漂移的時間軸。切窗時直接用索引，不需要對齊猜測。

問題 4：頭尾起始誤差
  → 開頭 CUE_LEAD_SEC、結尾 CUE_TAIL_SEC 保留純休息，不下任何 cue。
    這段同時當作濾波器暖機區與基線估計區。

★ 手勢順序是**隨機打散**的。舊資料每個檔案只有一個手勢、且固定循環，
  模型可以靠「認出這是哪一個錄音」或「猜循環到第幾秒」就得高分
  （實測：同 session 93%，換到沒見過的錄音掉到 28.5%，亂猜 25%）。
  一段錄音內隨機交替多個手勢，這兩條捷徑同時消失。

輸出三個檔案（同一個 stem，方便回溯）：
    <stem>.csv        timestamp_s, sample_index, ch1..ch8, status
    <stem>_cues.csv   sample_index, t_s, event, gesture, rep
    <stem>_meta.json  受試者、電極配置、協定參數、品質統計（滿足 CLAUDE.md metadata 規則）

用法：
    python src/hardware/record_session.py --port /dev/ttyACM0 --subject gary
    python src/hardware/record_session.py --port COM5 --subject gary --reps 8 --plot
    python src/hardware/record_session.py --dry-run          # 不接硬體，只看 cue 流程
"""
from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
import shutil
import threading
import time
import unicodedata
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semg.v2 import config as C  # noqa: E402

FRAME_LEN = 40
SYNC0, SYNC1, END_BYTE = 0xA5, 0x5A, 0x0D
NUM_CHANNELS = 8
FRAME_BODY_FMT = "<I8i"
FRAME_BODY_LEN = struct.calcsize(FRAME_BODY_FMT)


# ─────────────────────────────────────────────────────────────────────────────
# 解幀（與 ads1299_readerV6.py 完全相同的邏輯）
# ─────────────────────────────────────────────────────────────────────────────

def find_sync(ser) -> bool:
    prev = 0
    while True:
        b = ser.read(1)
        if not b:
            return False
        if prev == SYNC0 and b[0] == SYNC1:
            return True
        prev = b[0]


def read_one_frame(ser):
    rest = ser.read(FRAME_LEN - 2)
    if len(rest) != FRAME_LEN - 2:
        return None
    body = rest[:FRAME_BODY_LEN]
    checksum_byte = rest[FRAME_BODY_LEN]
    if rest[FRAME_BODY_LEN + 1] != END_BYTE:
        return None
    calc = 0
    for bv in bytes([SYNC0, SYNC1]) + body:
        calc ^= bv
    if calc != checksum_byte:
        return None
    status, *ch = struct.unpack(FRAME_BODY_FMT, body)
    return status, ch


# ─────────────────────────────────────────────────────────────────────────────
# cue 排程
# ─────────────────────────────────────────────────────────────────────────────

def build_schedule(gestures, reps, seed, prepare, action, ret, rest, lead,
                   preview=0.0, shuffle=True) -> list[dict]:
    """產生 cue 時程（相對於 t0 的秒數）。

    每個 trial 的階段，對應 `cues.py::intervals()` 的處理：

        preview  預告下一個姿勢（仍在休息、身體不動）→ 丟棄
        prepare  移動到目標姿勢                      → 丟棄
        go       **維持**目標姿勢                    → 標該姿勢  ← 訓練資料
        return   移動回中性                          → 丟棄
        release  中性姿勢完全放鬆                    → 標 rest   ← 訓練資料

    時間軸上 `preview` 位於**前一個 trial 的 rest 尾端**，內容是**下一個**姿勢 ——
    這樣「讀字判斷」與「身體移動」分開，反應時間 = preview + prepare，
    但只有 prepare 需要動。

    ★ preview / prepare / return 三段都丟棄，理由不同：
      preview：受試者已知道下一個是什麼，可能提前緊張 → 汙染「完全放鬆」
      prepare：正在移動，語意介於 rest 與目標姿勢之間
      return ：移動方向與去程相反，標成 rest 或該姿勢都會教錯

    shuffle=False 時**固定順序**（受試者可預期、動作品質較好，
    但前一個姿勢的殘留會系統性混入 → 必須在報告裡列為限制）。
    """
    order = list(gestures) * reps
    if shuffle:
        rng = np.random.default_rng(seed)
        order = []
        for _ in range(reps):
            blk = list(gestures)
            rng.shuffle(blk)
            # 避免與前一個 block 的結尾重複造成連三次同姿勢
            if order and blk[0] == order[-1] and len(blk) > 1:
                blk[0], blk[1] = blk[1], blk[0]
            order += blk

    sched, t = [], float(lead)
    for i, g in enumerate(order):
        # 預告放在**本 trial 的 prepare 之前**，也就是上一段休息的尾端
        if preview > 0:
            sched.append({"t": max(t - preview, 0.0), "event": "preview",
                          "gesture": g, "rep": i})
        sched.append({"t": t, "event": "prepare", "gesture": g, "rep": i})
        t += prepare
        sched.append({"t": t, "event": "go", "gesture": g, "rep": i})
        t += action
        sched.append({"t": t, "event": "return", "gesture": g, "rep": i})
        t += ret
        sched.append({"t": t, "event": "release", "gesture": g, "rep": i})
        t += rest
    sched.append({"t": t, "event": "end", "gesture": "", "rep": -1})
    return sorted(sched, key=lambda c: c["t"])


BAR = "#"          # 用 ASCII，Windows 主控台的預設字型不一定有 █


def _w(text: str) -> int:
    """顯示寬度。中文是**雙寬字元**，len() 會低估一半。

    先前 bug：把每行 ljust(190) 再輸出，中文行的實際顯示寬度超過 250 欄，
    Windows 主控台預設 80/120 欄 → 自動折行 → `\r` 只回到最後一行開頭，
    看起來就是每 0.05 秒印一行新的。
    """
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _fit(text: str, width: int) -> str:
    """截斷到指定顯示寬度，並補空白蓋掉上一次殘留的內容。"""
    out, w = [], 0
    for c in text:
        cw = 2 if unicodedata.east_asian_width(c) in "WF" else 1
        if w + cw > width:
            break
        out.append(c)
        w += cw
    return "".join(out) + " " * (width - w)


def term_width(default: int = 100) -> int:
    try:
        return max(40, shutil.get_terminal_size((default, 24)).columns - 1)
    except Exception:                       # noqa: BLE001
        return default


def render(event: str, gesture: str, remain: float, idx: int, total: int,
           elapsed: float, total_dur: float, samples: int, bad: int, fs_est: float,
           pred: str | None = None, width: int | None = None):
    """單行更新的 terminal 介面。

    ★ 一定要塞進終端機寬度內，否則會折行、`\r` 就失效（見 `_w`）。

    做法：欄位按**重要性排序**，逐項嘗試加入，放不下就**整項略過**
    （不做半截截斷 —— 「預測:」後面空白比不顯示更糟）。
    受試者最需要看到的是「現在該做什麼、還剩幾秒」，所以那兩項優先。
    """
    W = width or term_width()
    zh = C.ACTION_LABELS_ZH.get(gesture, gesture) if gesture else ""

    # ★ 排版原則：**受試者現在需要知道的東西放最左邊**，眼睛第一時間就看到。
    #   進度條與統計是給操作者看的，放最右邊，空間不夠就砍掉。
    #   preview 與 go 是兩個需要立刻反應的階段，用符號框起來讓它們跳出來。
    if event == "preview":
        head = f"▶▶ 下一個：{zh} ◀◀"          # 讀字判斷，身體還不用動
    elif event == "go":
        head = f"██ 維持：{zh} ██"             # 這 3 秒是訓練資料
    elif event == "prepare":
        head = f"→ 移動到：{zh}"
    else:
        head = {"return": "← 回中性", "release": "放鬆（完全不動）",
                "idle": "休息", "lead": "開頭空白（放鬆不動）",
                "tail": "結尾空白（放鬆不動）"}.get(event, event)

    # 倒數秒數緊跟在動作後面 —— 這是第二重要的資訊
    line = f"{head}  {remain:4.1f}s"
    optional = [
        f"({idx}/{total})",                         # 第幾個 trial
        f"| 預測:{pred}" if pred else None,          # 即時預測
        f"{elapsed:.0f}/{total_dur:.0f}s",          # 總進度
        f"{samples//1000}k樣本",                     # 統計
        f"{bad}壞幀" if bad else None,
    ]
    for opt in optional:
        if opt and _w(line) + _w(opt) + 1 <= W - 2:
            line += " " + opt
    # 進度條放**最右邊**（純裝飾，空間夠才給）
    room = W - _w(line) - 3
    if room >= 12:
        n = min(room, 20)
        prog = int(n * min(elapsed / max(total_dur, 1e-9), 1.0))
        line += f" [{BAR * prog}{'.' * (n - prog)}]"
    sys.stdout.write("\r" + _fit(line, W))
    sys.stdout.flush()


def clear_line(width: int | None = None):
    """把當前行清乾淨（要印固定訊息前呼叫，避免殘留字元）。"""
    sys.stdout.write("\r" + " " * (width or term_width()) + "\r")
    sys.stdout.flush()



# ─────────────────────────────────────────────────────────────────────────────
# 串列埠診斷
# ─────────────────────────────────────────────────────────────────────────────

def list_ports():
    try:
        from serial.tools import list_ports as lp
    except Exception:                       # noqa: BLE001
        return []
    return [(p.device, p.description) for p in lp.comports()]


def probe(port: str, baud: int, seconds: float = 3.0):
    """診斷「收不到資料」卡在哪一層。逐層檢查，每層失敗都給明確下一步。

    層次：能不能開埠 → 有沒有位元組進來 → 找不找得到 SYNC → checksum 對不對
    這樣就能區分「埠選錯」「baud 不對」「板子沒在送」「接線/供電」。
    """
    import serial

    print(f"探測 {port} @ {baud} baud，{seconds:.0f} 秒…\n")
    avail = list_ports()
    if avail:
        print("系統上看得到的串列埠：")
        for dev, desc in avail:
            mark = " ← 你指定的" if dev.upper() == port.upper() else ""
            print(f"    {dev:12s} {desc}{mark}")
        if not any(d.upper() == port.upper() for d, _ in avail):
            print(f"\n❌ {port} 不在清單裡 —— 埠名可能打錯，或板子沒插好。")
            return False
        print()

    try:
        ser = serial.Serial(port, baud, timeout=0.5)
    except Exception as e:                  # noqa: BLE001
        print(f"❌ 開不了 {port}：{e}")
        print("   常見原因：埠被其他程式佔用（Arduino IDE / 另一個 python 還開著）、"
              "\n   驅動未安裝、權限不足（Linux 要加入 dialout 群組）。")
        return False

    ser.reset_input_buffer()
    t0 = time.perf_counter()
    buf = bytearray()
    while time.perf_counter() - t0 < seconds:
        chunk = ser.read(4096)
        if chunk:
            buf += chunk
    ser.close()

    n = len(buf)
    print(f"收到 {n:,} bytes（{n/seconds:,.0f} B/s）")
    if n == 0:
        print("\n❌ 完全沒有資料。依序檢查：")
        print("   1. 板子是否已開始串流（有些韌體要先按 reset 或送啟動指令）")
        print("   2. 供電是否足夠（ADS1299 類比電源獨立）")
        print("   3. TX/RX 有沒有接反")
        print(f"   4. baud 是否為 {baud}（韌體端要一致）")
        return False

    # 期望位元率：40 bytes/frame × 2000 Hz = 80,000 B/s
    expect = FRAME_LEN * C.FS
    print(f"預期速率 {expect:,} B/s（{FRAME_LEN} bytes/frame × {C.FS} Hz），"
          f"實測比值 {n/seconds/expect:.2f}×")

    syncs = sum(1 for i in range(len(buf) - 1)
                if buf[i] == SYNC0 and buf[i + 1] == SYNC1)
    print(f"SYNC (0x{SYNC0:02X} 0x{SYNC1:02X}) 出現 {syncs:,} 次"
          f"（預期約 {int(C.FS*seconds):,} 次）")
    if syncs == 0:
        print("\n❌ 有位元組但找不到 SYNC —— 幾乎確定是 **baud 不對**。")
        print(f"   目前用 {baud}。前 32 bytes：{buf[:32].hex(' ')}")
        print("   若韌體是別的鮑率，用 --baud 指定。")
        return False

    # 實際解幀成功率
    good = bad = 0
    i = 0
    while i < len(buf) - FRAME_LEN:
        if buf[i] == SYNC0 and buf[i + 1] == SYNC1:
            fr = buf[i:i + FRAME_LEN]
            body = fr[2:2 + FRAME_BODY_LEN]
            calc = 0
            for bv in fr[:2 + FRAME_BODY_LEN]:
                calc ^= bv
            if (fr[2 + FRAME_BODY_LEN] == calc
                    and fr[3 + FRAME_BODY_LEN] == END_BYTE):
                good += 1
                i += FRAME_LEN
                continue
            bad += 1
        i += 1
    tot = good + bad
    print(f"完整幀 {good:,}／壞幀 {bad:,}"
          + (f"（壞幀率 {bad/tot:.2%}）" if tot else ""))
    if good == 0:
        print("\n❌ 找得到 SYNC 但沒有一幀通過 checksum —— 幀格式可能與本程式不同。")
        print(f"   本程式假設：{FRAME_LEN} bytes = SYNC(2) + "
              f'"{FRAME_BODY_FMT}"({FRAME_BODY_LEN}) + XOR checksum(1) + '
              f"END 0x{END_BYTE:02X}(1)")
        return False

    fs = good / seconds
    print(f"實測取樣率 ≈ {fs:,.0f} Hz（設定 {C.FS} Hz）")
    if bad and bad / tot > 0.01:
        print(f"\n⚠️ 壞幀率 {bad/tot:.1%} 偏高 —— 接線品質或 USB 頻寬問題，"
              "錄出來的資料會有缺樣本。")
    if abs(fs - C.FS) / C.FS > 0.05:
        print(f"\n⚠️ 實測取樣率與設定差 {abs(fs-C.FS)/C.FS:.0%}，"
              "cue 的 sample_index 對齊會受影響。")
        return False
    print("\n✅ 串列埠正常，可以開始錄製。")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="ADS1299 錄製 + 同步 cue")
    ap.add_argument("--port", help="例如 /dev/ttyACM0 或 COM5；--dry-run 時可省略")
    ap.add_argument("--baud", type=int, default=1_600_000)
    ap.add_argument("--subject", help="受試者代號，例如 gary（--probe/--list-ports 時可省略）")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="預設 data/raw/multi/<subject>/")
    ap.add_argument("--sps", type=int, default=C.FS)
    ap.add_argument("--vref", type=float, default=C.ADS_VREF)
    ap.add_argument("--gain", type=int, default=C.ADS_GAIN)

    ap.add_argument("--gestures", default=",".join(C.CUE_GESTURES))
    ap.add_argument("--reps", type=int, default=C.CUE_REPS, help="每個手勢重複幾次")
    ap.add_argument("--prepare-sec", type=float, default=C.CUE_PREPARE_SEC)
    ap.add_argument("--action-sec", type=float, default=C.CUE_ACTION_SEC)
    ap.add_argument("--preview-sec", type=float, default=C.CUE_PREVIEW_SEC,
                    help="預告下一個姿勢的秒數（仍在休息，這段丟棄）")
    ap.add_argument("--shuffle", action="store_true", default=C.CUE_SHUFFLE,
                    help="打亂姿勢順序（預設固定順序）")
    ap.add_argument("--return-sec", type=float, default=C.CUE_RETURN_SEC,
                    help="移動回中性的秒數（這段丟棄，不列入訓練）")
    ap.add_argument("--rest-sec", type=float, default=C.CUE_REST_SEC)
    ap.add_argument("--lead-sec", type=float, default=C.CUE_LEAD_SEC)
    ap.add_argument("--tail-sec", type=float, default=C.CUE_TAIL_SEC)
    ap.add_argument("--warmup-sec", type=float, default=C.CUE_WARMUP_DISCARD_SEC,
                    help="丟棄開始後這幾秒的樣本（硬體未穩定）")
    ap.add_argument("--seed", type=int, default=None, help="不給則用時間，每次順序不同")
    ap.add_argument("--no-beep", action="store_true")
    ap.add_argument("--notes", default="", help="寫進 meta.json 的備註")
    ap.add_argument("--dry-run", action="store_true", help="不開串列埠，只演練 cue 流程")
    ap.add_argument("--yes", "-y", action="store_true", help="跳過開始前的 Enter 確認")
    ap.add_argument("--probe", action="store_true",
                    help="只診斷串列埠（收不到資料時先跑這個），不錄製")
    ap.add_argument("--list-ports", action="store_true", help="列出系統上的串列埠後結束")
    ap.add_argument("--live", type=Path, default=None,
                    help="邊錄邊即時分類，指向訓練好的模型（results/v2/dof_model.pt）。"
                         "會即時顯示「提示 vs 預測」，錄完給一致率")
    ap.add_argument("--live-conf", type=float, default=0.6)
    ap.add_argument("--live-votes", type=int, default=5)
    a = ap.parse_args()

    if not (a.probe or a.list_ports) and not a.subject:
        ap.error("需要 --subject")
    if a.list_ports:
        ports = list_ports()
        if not ports:
            print("找不到任何串列埠（pyserial 沒裝？板子沒插？）")
        for dev, desc in ports:
            print(f"  {dev:12s} {desc}")
        return
    if a.probe:
        if not a.port:
            ap.error("--probe 需要 --port")
        raise SystemExit(0 if probe(a.port, a.baud) else 1)
    if not a.dry_run and not a.port:
        ap.error("需要 --port（或加 --dry-run 演練）")

    # ★ 錄之前先驗證時間參數自洽。`make_sequences` 對太短的段落是靜默 continue，
    #   參數設錯不會有任何徵兆 —— 直到你發現某個類別完全沒有訓練資料。
    #   這裡用**實際會用到的**秒數（CLI 覆寫後）而非 config 預設值。
    _saved = (C.CUE_PREVIEW_SEC, C.CUE_ACTION_SEC, C.CUE_REST_SEC)
    C.CUE_PREVIEW_SEC, C.CUE_ACTION_SEC, C.CUE_REST_SEC = (
        a.preview_sec, a.action_sec, a.rest_sec)
    problems = C.validate()
    seg_rep = C.segment_report()
    C.CUE_PREVIEW_SEC, C.CUE_ACTION_SEC, C.CUE_REST_SEC = _saved
    if problems:
        print("=" * 72)
        print("❌ 時間參數有問題，錄了也用不了：\n")
        for x in problems:
            print(f"  • {x}\n")
        print("  調整 --preview-sec / --action-sec / --rest-sec 後再試。")
        print("=" * 72)
        raise SystemExit(1)

    gestures = tuple(g.strip() for g in a.gestures.split(",") if g.strip())
    seed = a.seed if a.seed is not None else int(time.time()) % 100000
    sched = build_schedule(gestures, a.reps, seed, a.prepare_sec,
                           a.action_sec, a.return_sec, a.rest_sec, a.lead_sec,
                           preview=a.preview_sec, shuffle=a.shuffle)
    total_dur = sched[-1]["t"] + a.tail_sec

    stem = f"multi_{a.subject}_{datetime.now().strftime('%Y%m%d_%H-%M-%S')}"
    outdir = a.outdir or (C.DATA_ROOT / "raw" / "multi" / a.subject)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{stem}.csv"
    cue_path = outdir / f"{stem}_cues.csv"
    meta_path = outdir / f"{stem}_meta.json"

    print("=" * 72)
    print(f"受試者   : {a.subject}")
    print(f"姿勢     : {len(gestures)} 種 × {a.reps} 次 = {len(gestures)*a.reps} 個 trial")
    for g in gestures:
        dof = C.action_to_dof(g)
        detail = " + ".join(
            f"{C.DOF_LABELS_ZH[d]}:{C.STATE_LABELS_ZH[C.DOF_STATES[d][v]]}"
            for d, v in zip(C.DOFS, dof) if C.DOF_STATES[d][v] != "rest") or "全放鬆"
        print(f"           {C.ACTION_LABELS_ZH.get(g, g):<16s} → {detail}")
    if a.shuffle:
        print(f"順序     : 隨機打散（seed={seed}）")
    else:
        print(f"順序     : **固定**（每輪都一樣，方便預期）")
        print(f"           {' → '.join(C.ACTION_LABELS_ZH.get(g, g) for g in gestures)}")
        print(f"           ⚠️ 前一個姿勢的殘留會系統性混入，報告時需列為限制")
    print(f"時程     : 開頭空白 {a.lead_sec}s → 每 trial（移動到位 {a.prepare_sec}s "
          f"→ **維持 {a.action_sec}s** → 回中性 {a.return_sec}s → 放鬆 {a.rest_sec}s）"
          f"→ 結尾空白 {a.tail_sec}s")
    print(f"           只有「維持」與「放鬆」兩段列入訓練，移動過程全部丟棄")
    print(f"訓練資料 : 每個 trial 產生")
    for name, raw, lab, nw, ns, slack in seg_rep:
        warn = "  ⚠️ 餘裕不多" if slack < 0.5 else ""
        print(f"           {name:16s} {raw:.1f}s → 扣邊界 {lab:.1f}s → "
              f"{nw:3d} 視窗 → {ns:3d} 條序列{warn}")
    print(f"總長度   : {total_dur:.0f}s ≈ {total_dur/60:.1f} 分鐘")
    print(f"輸出     : {csv_path.name} / {cue_path.name} / {meta_path.name}")
    print("電極配置 :")
    for i, (n, zh) in enumerate(zip(C.CH_NAMES, C.CH_LABELS_ZH), 1):
        print(f"           ch{i} {zh}  ({n})")
    print("=" * 72)

    # ★ 姿勢規範 —— 這是「不用 IMU」的前提，每次錄製前都要確認
    ps = C.POSE_SPEC
    print("\n【錄製協定】標的是「維持中的姿勢」，不是移動的過程")
    print(f"  中性姿勢：{ps['neutral']}")
    print(f"\n  ⚠️ {ps['airborne']}")
    print(f"\n  節奏：{ps['timing']}")
    print(f"  範圍：{ps['range']}")
    print(f"  疲勞：{ps['fatigue']}")
    print("\n  各動作要維持的姿勢：")
    for g in gestures:
        pose, tip = ps["actions"].get(g, ("（未定義，請補進 config.POSE_SPEC）", ""))
        print(f"    {C.ACTION_LABELS_ZH.get(g, g):<16s} {pose}")
        if tip:
            print(f"    {'':16s}   → {tip}")
    print("=" * 72)
    print("\n⚠️ 貼好電極後，請保持放鬆不動，直到看到「準備」提示。")
    print("   開頭與結尾各有一段空白，那段**不要做任何動作**。")
    if a.yes:
        print("\n（--yes：直接開始）")
    else:
        try:
            input("\n按 Enter 開始（開始後 terminal 會告訴你每一刻該做什麼）… ")
        except EOFError:          # 非互動環境（測試、管線）→ 直接開始
            print("\n（無 stdin，直接開始）")

    # ── 共享狀態（讀取 thread ↔ 主 thread）─────────────────────────────────
    state = {"samples": 0, "bad": 0, "t0": None, "stop": False, "err": None}
    lock = threading.Lock()

    # ── 即時分類（可選）─────────────────────────────────────────────────
    # 讀取 thread 只負責把樣本丟進佇列，推論放在主 thread ——
    # 絕不能讓推論拖慢讀取，否則會掉樣本（這是 readerV6 把讀取放在繪圖
    # callback 裡的同一個錯誤）。
    rec, live_q = None, deque()
    if a.live:
        if not a.live.exists():
            raise SystemExit(f"找不到模型 {a.live}，請先跑 train.py")
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from semg.v2.infer import Recognizer
        rec = Recognizer(a.live, votes=a.live_votes, conf=a.live_conf)
        print(f"即時分類：載入 {a.live.name}"
              f"（{'TorchScript' if rec.scripted else 'eager'}，"
              f"{len(rec.dof_names)} 個自由度）")

    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    w = csv.writer(csv_file)
    # ★ 多寫一欄 sample_index：cue 就是靠這個對齊，比 timestamp 可靠（無漂移）
    w.writerow(["timestamp_s", "sample_index"]
               + [f"ch{i+1}" for i in range(NUM_CHANNELS)] + ["status"])

    def reader_thread(ser):
        """獨立 thread 只做「讀 frame → 寫 CSV」，不做任何繪圖或計算。"""
        discard_until = None
        try:
            while not state["stop"]:
                if not find_sync(ser):
                    continue
                r = read_one_frame(ser)
                if r is None:
                    with lock:
                        state["bad"] += 1
                    continue
                status, ch = r
                now = time.perf_counter()

                # (b) 丟棄暖機期樣本
                if discard_until is None:
                    discard_until = now + a.warmup_sec
                    continue
                if now < discard_until:
                    continue

                # (c) t0 = 第一個「被接受」的樣本
                with lock:
                    if state["t0"] is None:
                        state["t0"] = now
                    idx = state["samples"]
                    state["samples"] += 1
                    t_rel = now - state["t0"]
                w.writerow([f"{t_rel:.6f}", idx] + list(ch) + [status])
                if rec is not None:
                    live_q.append([ch[c - 1] for c in C.CH_CSV_COLS])
        except Exception as e:                       # noqa: BLE001
            state["err"] = repr(e)

    def dry_thread():
        """--dry-run：假裝以 sps 產生樣本，讓時程與統計都能演練。"""
        t_begin = time.perf_counter()
        while not state["stop"]:
            time.sleep(0.02)
            now = time.perf_counter()
            if now - t_begin < a.warmup_sec:
                continue
            with lock:
                if state["t0"] is None:
                    state["t0"] = now
                target = int((now - state["t0"]) * a.sps)
                n_new = target - state["samples"]
                state["samples"] = target
            if rec is not None and n_new > 0:
                # dry-run 用雜訊填充，只為了驗證即時路徑不會卡住
                for _ in range(n_new):
                    live_q.append(list(np.random.randn(C.N_CH) * 200))

    ser = None
    if a.dry_run:
        th = threading.Thread(target=dry_thread, daemon=True)
    else:
        import serial
        ser = serial.Serial(a.port, a.baud, timeout=1)
        # (a) 明確丟掉 UART 裡開機前累積的舊資料 —— 這是「開始時間不確定」的主因之一
        ser.reset_input_buffer()
        th = threading.Thread(target=reader_thread, args=(ser,), daemon=True)
    th.start()

    # ── 等到真的收到第一個樣本，才開始跑 cue 時程 ─────────────────────────
    # 這一步就是「知道正確開始測量的時間點」：cue 的 t=0 與資料的 sample 0 同一刻。
    print("\n等待第一個有效樣本…", end="", flush=True)
    t_wait = time.perf_counter()
    while state["t0"] is None and state["err"] is None:
        time.sleep(0.002)
        if time.perf_counter() - t_wait > 10:
            state["stop"] = True
            csv_file.close()
            if ser:
                ser.close()
            raise SystemExit("\n❌ 10 秒內收不到有效 frame。檢查 --port / --baud / 板子供電。")
    if state["err"]:
        state["stop"] = True
        csv_file.close()
        raise SystemExit(f"\n❌ 讀取失敗：{state['err']}")
    t0 = state["t0"]
    print(f" 收到！t0 已鎖定（等待 {t0 - t_wait:.2f}s）\n")

    # ── cue 迴圈 ──────────────────────────────────────────────────────────
    cue_f = open(cue_path, "w", newline="", encoding="utf-8")
    cw = csv.writer(cue_f)
    cw.writerow(["sample_index", "t_s", "event", "gesture", "rep"])

    def log_cue(event: str, gesture: str, rep: int):
        """(3) cue 記錄當下的 sample_index —— 唯一無漂移的時間軸。"""
        with lock:
            idx, t_rel = state["samples"], time.perf_counter() - t0
        cw.writerow([idx, f"{t_rel:.6f}", event, gesture, rep])
        cue_f.flush()
        return idx

    log_cue("record_start", "", -1)
    n_trial = len([s for s in sched if s["event"] == "go"])
    last_fs_t, last_fs_n, fs_est = t0, 0, 0.0

    # 即時分類狀態
    live_pred, live_hits, live_calibrated = None, [], False

    def drain_live(cue_gesture: str | None):
        """把佇列裡的樣本餵給模型。回傳目前的預測文字。

        校準用開頭空白段：MAD 只需要一段任意訊號就能估雜訊底線，
        不需要受試者先出力（這是 v1 z-score 做不到的）。
        """
        nonlocal live_pred, live_calibrated
        if rec is None or not live_q:
            return live_pred
        n = len(live_q)
        blk = np.array([live_q.popleft() for _ in range(n)], dtype=np.float64) * C.LSB_MV
        if not live_calibrated:
            if (time.perf_counter() - t0) < a.lead_sec * 0.8:
                return live_pred                      # 開頭空白段還在累積
            rec.calibrate(blk)
            live_calibrated = True
            return live_pred
        for res in rec.push(blk):
            live_pred = res["text"]
            if cue_gesture:                           # 出力段才記一致率
                live_hits.append((cue_gesture, res["action"]))
        return live_pred
    try:
        for k, cue in enumerate(sched):
            # 等到這個 cue 的時間點
            while True:
                el = time.perf_counter() - t0
                remain = cue["t"] - el
                if remain <= 0:
                    break
                with lock:
                    ns, nb = state["samples"], state["bad"]
                now = time.perf_counter()
                if now - last_fs_t >= 1.0:
                    fs_est = (ns - last_fs_n) / (now - last_fs_t)
                    last_fs_t, last_fs_n = now, ns
                # `cue` 是**下一個**要發生的事件，所以現在所處的階段是前一個事件
                prev = sched[k - 1] if k else {"event": "lead", "gesture": ""}
                cur = prev["event"] if k else "lead"
                # 只在「移動到位」與「維持」時顯示目標姿勢名稱；
                # 回中性與放鬆階段不顯示（階段標籤本身已經夠清楚）
                show_gesture = cur in ("preview", "prepare", "go")
                # 只有「維持」那段才是有標籤的訓練資料 → 只有它拿來算即時一致率
                in_action = cur == "go"
                pred = drain_live(prev.get("gesture") if in_action else None)
                render(cur, prev.get("gesture", "") if show_gesture else "",
                       remain, cue["rep"] + 1 if cue["rep"] >= 0 else n_trial,
                       n_trial, el, total_dur, ns, nb, fs_est, pred)
                time.sleep(0.05)

            idx = log_cue(cue["event"], cue["gesture"], cue["rep"])
            if cue["event"] == "prepare":
                clear_line()
                zh = C.ACTION_LABELS_ZH.get(cue["gesture"], cue["gesture"])
                dof = C.action_to_dof(cue["gesture"])
                detail = " ".join(
                    f"{C.DOF_LABELS_ZH[d]}={C.STATE_LABELS_ZH[C.DOF_STATES[d][v]]}"
                    for d, v in zip(C.DOFS, dof) if C.DOF_STATES[d][v] != "rest")
                print(f"\r  trial {cue['rep']+1:2d}/{n_trial}  下一個要維持的姿勢 → 【{zh}】"
                      f"  [{detail}]  (sample {idx})")
                if not a.no_beep:
                    sys.stdout.write("\a")
            elif cue["event"] == "go" and not a.no_beep:
                sys.stdout.write("\a")
            elif cue["event"] == "end":
                break
            sys.stdout.flush()

        # ── 結尾空白 ──────────────────────────────────────────────────────
        t_end = time.perf_counter() - t0
        while time.perf_counter() - t0 < t_end + a.tail_sec:
            with lock:
                ns, nb = state["samples"], state["bad"]
            el = time.perf_counter() - t0
            pred = drain_live(None)
            render("tail", "", t_end + a.tail_sec - el, n_trial, n_trial,
                   el, total_dur, ns, nb, fs_est, pred)
            time.sleep(0.05)
        log_cue("record_end", "", -1)

    except KeyboardInterrupt:
        log_cue("aborted", "", -1)
        print("\n\n⚠️ 手動中止 —— 已寫入的資料仍然有效（cue 檔記錄了 aborted）")
    finally:
        state["stop"] = True
        time.sleep(0.2)
        with lock:
            n_samp, n_bad = state["samples"], state["bad"]
        dur = n_samp / a.sps if a.sps else 0.0
        csv_file.flush(); csv_file.close()
        cue_f.close()
        if ser:
            ser.close()

        if rec is not None and live_hits:
            import collections
            agree = sum(1 for cue_g, pred_a in live_hits if cue_g == pred_a)
            print(f"\n即時分類一致率（出力段，預測動作 == cue 動作）："
                  f"{agree}/{len(live_hits)} = {agree/len(live_hits):.1%}")
            per = collections.defaultdict(lambda: [0, 0])
            for cue_g, pred_a in live_hits:
                per[cue_g][1] += 1
                per[cue_g][0] += int(cue_g == pred_a)
            print("  逐動作：")
            for g, (ok, tot) in sorted(per.items()):
                print(f"    {C.ACTION_LABELS_ZH.get(g, g):<18s} {ok:4d}/{tot:<4d} "
                      f"{ok/max(tot,1):6.1%}")
            meta_live = {"n_windows": len(live_hits), "agreement": agree / len(live_hits),
                         "per_action": {g: {"ok": v[0], "n": v[1]} for g, v in per.items()}}
        else:
            meta_live = None

        meta = {
            "stem": stem,
            "subject": a.subject,
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "protocol": {
                # ★ protocol_version 讓 dataset 能擋掉「新舊協定混進同一個資料集」。
                #   v1 = go 3.0 / return 1.5 / preview 2.5（第一份 gary 錄音）
                #   v2 = go 4.0 / return 2.5 / preview 1.5（依真實資料量測後調整）
                "protocol_version": C.PROTOCOL_VERSION,
                "gestures": list(gestures), "reps": a.reps, "order_seed": seed,
                # ★ 從實際排程推導，不要寫死 —— 先前這裡固定寫 True，
                #   但協定早已改成固定順序，meta 與事實不符。
                "order_randomised": bool(a.shuffle),
                "lead_sec": a.lead_sec, "tail_sec": a.tail_sec,
                "preview_sec": a.preview_sec,
                "prepare_sec": a.prepare_sec, "action_sec": a.action_sec,
                "return_sec": a.return_sec,
                "rest_sec": a.rest_sec, "warmup_discard_sec": a.warmup_sec,
                "margin_start_sec": C.CUE_MARGIN_START_SEC,
                "margin_rest_start_sec": C.CUE_MARGIN_REST_START_SEC,
                "margin_end_sec": C.CUE_MARGIN_END_SEC,
                "cue_before_action": True,
            },
            "hardware": {"sps": a.sps, "gain": a.gain, "vref": a.vref,
                         "lsb_mv": (2.0 * a.vref) / (a.gain * 2 ** 24) * 1e3,
                         "baud": a.baud, "port": a.port, "dry_run": a.dry_run},
            "dof_definition": {d: {"states": list(C.DOF_STATES[d]),
                                   "label": C.DOF_LABELS_ZH[d]} for d in C.DOFS},
            "action_to_dof": {g: dict(zip(C.DOFS,
                              [C.DOF_STATES[d][v] for d, v in
                               zip(C.DOFS, C.action_to_dof(g))])) for g in gestures},
            "electrode_placement": {f"ch{i+1}": {"muscle": zh, "name": n}
                                    for i, (n, zh) in
                                    enumerate(zip(C.CH_NAMES, C.CH_LABELS_ZH))},
            "quality": {"n_samples": n_samp, "n_bad_frames": n_bad,
                        "duration_s": round(dur, 2),
                        "expected_duration_s": round(total_dur, 2),
                        "mean_fs_hz": round(n_samp / dur, 1) if dur else 0.0,
                        "bad_frame_rate": round(n_bad / max(n_samp + n_bad, 1), 5)},
            "notes": a.notes,
            # 姿勢規範寫進 metadata —— 之後回頭看資料時才知道是在什麼條件下錄的
            # （CLAUDE.md 的規則：沒有 metadata 的資料不進 pipeline）
            "pose_spec": {
                "neutral": C.POSE_SPEC["neutral"],
                "airborne": C.POSE_SPEC["airborne"],
                "timing": C.POSE_SPEC["timing"],
                "labelled_phases": "go(維持)=該姿勢, release(放鬆)=rest; "
                                   "prepare/return(移動過程)=丟棄",
                "actions": {g: C.POSE_SPEC["actions"].get(g, ("", ""))[0]
                            for g in gestures},
            },
            "live_inference": meta_live,
            "files": {"data": csv_path.name, "cues": cue_path.name},
        }
        # ★ 一定要指定 encoding="utf-8"。
        #   `Path.write_text()` 不給編碼時用**系統 locale** —— 在 Windows(zh-TW)
        #   是 cp950，寫出來的 meta.json 在 Linux 端 json.load 會直接 UnicodeDecodeError。
        #   第一份真實錄音就是這樣（實際踩到，只能用 cp950 硬解才讀得出來）。
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                             encoding="utf-8")

        print("\n" + "=" * 72)
        print(f"樣本數     : {n_samp:,}（預期 {int(total_dur*a.sps):,}）")
        print(f"實際時長   : {dur:.1f}s（預期 {total_dur:.0f}s）")
        print(f"平均取樣率 : {n_samp/dur:.1f} Hz" if dur else "")
        print(f"壞幀       : {n_bad}（{n_bad/max(n_samp+n_bad,1):.3%}）")
        loss = 1 - n_samp / max(total_dur * a.sps, 1)
        if loss > 0.02:
            print(f"⚠️ 樣本數比預期少 {loss:.1%} —— 可能掉樣本，這批資料要謹慎使用")
        else:
            print(f"✅ 樣本完整度 {1-loss:.1%}")
        print(f"\n已存：\n  {csv_path}\n  {cue_path}\n  {meta_path}")
        print("=" * 72)


if __name__ == "__main__":
    main()
