"""CLI 進入點：手動選擇 (4,2,2) 自由度組合，直接驅動 Lite 6 —— 不經過分類模型。

適合在模型還沒訓練好、或想單獨驗證控制層（映射/限位/去彈跳/安全機制）時使用。
背後跑的是與 `run_arm_control.py` 完全相同的 `ArmController`——debounce、
位置鏡像、pinch toggle、J6 jog、雙層限位、看門狗、SAFE 全部照跑，差別只在
`Prediction` 的來源：這裡是你自己選的（`ManualSource`），不是模型跑出來的。

用法：
    python src/control/run_manual_control.py --dry-run          # 假手臂，先驗證選單/映射
    python src/control/run_manual_control.py --arm-ip 192.168.1.166   # 真手臂

互動：
    先按 Enter 進入 ARMED，再用 `<hand> <elbow> <shoulder>` 三個數字選狀態，
    例如 `3 1 0` = pinch + 手肘彎曲 + 肩膀放鬆。其他指令：a=ARMED d=DISARMED
    z=J6歸零 s=查看目前狀態 q=結束（回 home + 斷線）。
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # src/ 加進 sys.path（給 control package）

from control import arm_config as AC                                # noqa: E402

# semg_realtime_V6/src 插在最前面，蓋過 src/semg 的舊快照——DOF_STATES 定義雖然
# 兩邊一致，但統一來源才不會日後跟真的分類器（run_arm_control.py）分岔。
sys.path.insert(0, str(AC.SEMG_REALTIME_SRC))
AC.validate()          # 與 run_arm_control.py 共用同一份 arm_config，檢查一併做

from control.arm_controller import ArmController, MockArm           # noqa: E402
from control.sources import ManualSource                             # noqa: E402
from semg.v2 import config as C                                      # noqa: E402

HAND_STATES = C.DOF_STATES["hand"]           # ("rest", "index", "thumb", "pinch")
ELBOW_STATES = C.DOF_STATES["elbow"]          # ("rest", "flex")
SHOULDER_STATES = C.DOF_STATES["shoulder"]     # ("rest", "flex")

# 選定後等待去彈跳（多數決 + refractory）穩定再印狀態，用最大的 refractory 當保守值。
# +0.2s 是排程抖動的安全餘裕（背景 thread 的 tick 週期、OS 排程延遲都不是零延遲）——
# 理論最小值算出來剛好卡在邊界上時，實機測試發現偶爾會因為這些餘裕不夠而漏轉態。
_SETTLE_SEC = max(AC.REFRACTORY.values()) + AC.VOTE_M * AC.STEP_SEC + 0.2


def _connect_real_arm(ip: str, timeout: float = 5.0):
    """連線實體手臂，並確認連線真的成功。跟 `run_arm_control.py` 的
    `_connect_real_arm` 是同一套邏輯（那裡有完整說明）。

    ⚠️ SDK 1.18.5 的 `XArmAPI.__init__` 內部就會 `connect()`（`base.py:282`），
    IP 不可達時**建構子本身就會拋例外**，不是建好之後才發現連不上——所以
    必須把建構子包在 try 裡，否則下面的檢查清單一行都印不出來。
    """
    try:
        from xarm.wrapper import XArmAPI
    except ImportError:
        sys.path.insert(0, str(AC.REPO_ROOT / "xArm-Python-SDK-master"))
        from xarm.wrapper import XArmAPI

    def _fail(reason: str) -> SystemExit:
        return SystemExit(
            f"\n✖ 無法連線手臂 {ip}（{reason}）。請檢查控制箱電源/網路線、"
            f"IP 是否正確（目前 {AC.ARM_IP}）、是否被 UFACTORY Studio 佔用連線。"
            f"沒有手臂時請用 --dry-run。")

    print(f"連線手臂 {ip} …")
    t0 = time.monotonic()
    try:
        arm = XArmAPI(ip, is_radian=False)
    except Exception as e:                 # noqa: BLE001  SDK 拋的是裸 Exception
        raise _fail(f"socket 連線失敗，{time.monotonic() - t0:.1f}s：{e}") from None

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if arm.connected:
            break
        time.sleep(0.2)
    else:
        try:
            arm.disconnect()
        except Exception:                  # noqa: BLE001
            pass
        raise _fail(f"connected 逾時 {timeout:.0f}s")
    time.sleep(0.5)          # 等 report 通道推送第一輪狀態，見 startup() 的說明
    print(f"已連線。firmware={arm.version}")
    return arm


def _print_menu() -> None:
    print()
    print("=" * 64)
    print("手動選擇模式 —— 輸入 <hand> <elbow> <shoulder> 直接驅動手臂")
    print("=" * 64)
    print("hand     : " + "  ".join(f"{i}={s}" for i, s in enumerate(HAND_STATES)))
    print("elbow    : " + "  ".join(f"{i}={s}" for i, s in enumerate(ELBOW_STATES)))
    print("shoulder : " + "  ".join(f"{i}={s}" for i, s in enumerate(SHOULDER_STATES)))
    print("範例：3 1 0  →  pinch + 手肘彎曲 + 肩膀放鬆")
    print("指令：a=ARMED  d=DISARMED  z=J6歸零  s=查看狀態  q=結束（回home+斷線）")
    print("=" * 64)


def _parse_selection(line: str):
    parts = line.strip().split()
    if len(parts) != 3:
        return None
    try:
        h, e, s = (int(p) for p in parts)
    except ValueError:
        return None
    if not (0 <= h < len(HAND_STATES) and 0 <= e < len(ELBOW_STATES)
            and 0 <= s < len(SHOULDER_STATES)):
        return None
    return h, e, s


def _format_status(status: dict) -> str:
    return (f"[{status['state']}] hand={status['hand_name']} "
            f"elbow={status['elbow_name']} shoulder={status['shoulder_name']} | "
            f"J2={status['J2']:+.1f} J3={status['J3']:+.1f} J6={status['J6']:+.1f} | "
            f"grip={status['grip']} | cmdnum={status['cmdnum']}")


def _control_loop(controller: ArmController, source: ManualSource,
                  stop_evt: threading.Event) -> None:
    """背景 thread：持續把目前選擇的狀態餵給 ArmController（J6 jog 需要這個持續跑，
    不然使用者停在 input() 等下一個指令時，jog 或看門狗都會停擺）。不印狀態，
    避免跟前景的 input() 提示互相干擾——狀態顯示交給前景在選擇後同步印出。
    """
    while not stop_evt.is_set():
        now = time.monotonic()
        for pred in source.poll(now):
            controller.on_prediction(pred, now)
        controller.tick(now)
        time.sleep(AC.STEP_SEC)


def main() -> None:
    ap = argparse.ArgumentParser(description="手動選擇 (4,2,2) 自由度，直接驅動 Lite 6")
    ap.add_argument("--dry-run", action="store_true", help="用 MockArm 取代真手臂")
    ap.add_argument("--arm-ip", default=AC.ARM_IP)
    ap.add_argument("--log", type=Path, default=None)
    a = ap.parse_args()

    print("=" * 78)
    print("⚠️  維持姿勢時手臂必須懸空 —— 手肘/前臂/上臂不能靠在桌面、扶手或身側，")
    print("    否則重力被外物撐住、EMG 消失（此工具不涉及 EMG，但姿勢限位邏輯相同）。")
    print("=" * 78)

    arm = MockArm() if a.dry_run else _connect_real_arm(a.arm_ip)
    log_path = a.log or (AC.LOG_ROOT / f"manual_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    controller = ArmController(arm, log_path=log_path)
    controller.startup()

    source = ManualSource()
    stop_evt = threading.Event()
    th = threading.Thread(target=_control_loop, args=(controller, source, stop_evt),
                          daemon=True)
    th.start()

    def _engage() -> None:
        controller.engage()
        # engage() 內含 wait=True 的回 home，會阻塞（真手臂上可達數秒）；期間
        # 背景 thread 仍在跑，ManualSource 的節奏基準會落後。resync 對齊一次，
        # 避免恢復後一次補發一串同狀態的 Prediction（跟 run_arm_control.py 同理）。
        source.resync(time.monotonic())

    try:
        input("按 Enter 進入 ARMED（之後手臂才會真的動）... ")
    except EOFError:
        pass
    _engage()
    print(_format_status(controller.status()))

    _print_menu()
    try:
        while True:
            try:
                line = input("\n> ").strip()
            except EOFError:
                break
            if not line:
                continue
            cmd = line.lower()
            if cmd == "q":
                break
            if cmd == "z":
                controller.zero_j6()
                print(_format_status(controller.status()))
                continue
            if cmd == "s":
                print(_format_status(controller.status()))
                continue
            if cmd == "a":
                _engage()
                print(_format_status(controller.status()))
                continue
            if cmd == "d":
                controller.disarm()
                print(_format_status(controller.status()))
                continue

            sel = _parse_selection(line)
            if sel is None:
                print("格式錯誤，請輸入三個數字，例如：3 1 0（或 a/d/z/s/q）")
                _print_menu()
                continue
            h, e, s = sel
            source.set_state(h, e, s)
            print(f"→ 已選擇：hand={HAND_STATES[h]} elbow={ELBOW_STATES[e]} "
                  f"shoulder={SHOULDER_STATES[s]}（等待去彈跳穩定 {_SETTLE_SEC:.1f}s…）")
            time.sleep(_SETTLE_SEC)
            print(_format_status(controller.status()))
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        th.join(timeout=1.0)
        # shutdown() 內部會自行回報是否成功回 home（見 arm_controller.py），
        # 這裡不再無條件印一句「已回 home」——那可能是謊話（例如手臂帶著
        # error_code 結束時根本回不了 home）。包 try 避免它失敗時整個中斷。
        try:
            controller.shutdown()
        except Exception as e:              # noqa: BLE001
            print(f"[error] shutdown 失敗：{e!r}")
            print("        ⚠️ 手臂可能仍 enable，請手動用 UFACTORY Studio 停機。")


if __name__ == "__main__":
    main()
