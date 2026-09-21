"""把 `semg.v2.live` 的即時分類輸出接到手臂控制層。

存在的理由：`run_live.py` 是使用者實際在用的肌電進入點（校準、電極自檢、
calib_log、靜止誤觸發測試都在那條路徑上），而手臂控制邏輯全部在
`control.arm_controller.ArmController`。這個模組是兩者之間唯一的黏合層——
它**不含任何控制邏輯**，只做三件事：

  1. 連手臂（或 MockArm）、建 ArmController、跑 startup()
  2. 把 `Recognizer.push()` 的一筆結果（`stable` + `conf`）轉成 `Prediction`
     餵給 controller
  3. 非阻塞按鍵（Enter 進 ARMED / q 結束 / z 把 J6 歸零）

★ 分層原則（不要在這裡加控制邏輯）：
    調參數（速度/限位/逾時）          → control/arm_config.py
    改手臂行為（jog 演算法/夾爪邏輯）  → control/arm_controller.py
    改模型/特徵/辨識過濾               → semg_realtime_V4 的 semg.v2.*
    只改終端顯示或資料來源             → semg/v2/live.py 或本檔
  這樣手動版（run_manual_control.py）才能永遠當作隔離排查的對照工具。
"""
from __future__ import annotations

import time
from pathlib import Path

from control import arm_config as AC
from control.arm_controller import ArmController, MockArm, Prediction

_KEYS_AVAILABLE = False
try:
    import msvcrt as _msvcrt          # noqa: N812
    _KEYS_AVAILABLE = True
except ImportError:
    _msvcrt = None


def poll_key() -> str | None:
    """非阻塞讀一個按鍵；沒有按鍵回傳 None。僅 Windows（msvcrt）。"""
    if _msvcrt is None:
        return None
    if _msvcrt.kbhit():
        return _msvcrt.getwch()
    return None


def connect_real_arm(ip: str, timeout: float = 5.0):
    """連線實體手臂並確認連線真的成功。

    ⚠️ SDK 1.18.5 的 `XArmAPI.__init__` 內部就會 `connect()`（`xarm/x3/base.py:282`），
    IP 不可達時**建構子本身**就 raise，不是建好之後才發現連不上——所以建構子
    必須包在 try 裡，否則下面的檢查清單一行都印不出來，而「IP 打錯」的症狀
    （畫面全綠、target 在變、手臂不動）跟「手臂在 STOP」「夾爪沒裝」幾乎無法
    區分，這則訊息是唯一的分辨依據。
    """
    try:
        from xarm.wrapper import XArmAPI
    except ImportError:
        import sys
        sys.path.insert(0, str(AC.REPO_ROOT / "xArm-Python-SDK-master"))
        from xarm.wrapper import XArmAPI

    def _fail(reason: str) -> SystemExit:
        return SystemExit(
            f"\n✖ 無法連線手臂 {ip}（{reason}）。請檢查：\n"
            f"  1. 控制箱電源與網路線\n"
            f"  2. IP 是否正確（arm_config.ARM_IP = {AC.ARM_IP}）\n"
            f"  3. 電腦與控制箱是否在同一網段（試 ping {ip}）\n"
            f"  4. UFACTORY Studio 是否正佔用連線\n"
            f"※ 沒有手臂時請用 --arm-dry-run。")

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

    # report 通道要幾百 ms 才推第一輪狀態；startup() 讀 warn_code/error_code
    # 之前必須等，否則讀到的是還沒更新的 0（誤判為「沒有錯誤」）。
    time.sleep(0.5)
    print(f"已連線。firmware={arm.version}")
    return arm


class ArmBridge:
    """`semg.v2.live` ←→ `ArmController` 的黏合層。"""

    def __init__(self, dof_names: list[str], arm_ip: str | None = None,
                 dry_run: bool = False, log_path: Path | None = None):
        self._idx = {name: i for i, name in enumerate(dof_names)}
        for need in ("hand", "elbow", "shoulder"):
            if need not in self._idx:
                raise SystemExit(
                    f"模型的自由度是 {dof_names}，缺少 '{need}'——控制層的映射"
                    "（hand→夾爪/J6、elbow→J3、shoulder→J2）需要這三個都在。")

        self.arm = MockArm() if dry_run else connect_real_arm(arm_ip or AC.ARM_IP)
        log_path = log_path or (AC.LOG_ROOT
                                / f"live_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        # ★ 單層投票：Recognizer 那層（run_live 原本就有的 votes）是唯一的投票，
        #   控制層這層改成 1/1 的 pass-through，避免疊加約 0.2s 延遲。
        #   P_MIN 與 REFRACTORY 照常生效——它們不是投票，見 arm_config.py。
        self.controller = ArmController(self.arm, log_path=log_path,
                                        vote_m=AC.LIVE_VOTE_M, vote_n=AC.LIVE_VOTE_N)
        self.controller.startup()

        if not _KEYS_AVAILABLE:
            print("!" * 78)
            print("⚠️  本平台不支援即時按鍵（僅 Windows 的 msvcrt）：")
            print("    Enter / q / z 全部無效。請用 --arm-yes 自動 ARM、Ctrl-C 結束。")
            print("!" * 78)

    # -- Prediction 餵入 ------------------------------------------------

    def feed(self, res: dict, now: float) -> None:
        """把 `Recognizer.push()`／`classify()` 的一筆結果餵給 controller。

        `stable[j]` 還沒穩定過時是 None——跟 `ArmController` 自己的慣例一致，
        未穩定就當 rest（index 0，`DOF_STATES` 每個 DOF 都以 rest 開頭）。
        """
        stable, conf = res["stable"], res["conf"]
        h, e, s = (self._idx["hand"], self._idx["elbow"], self._idx["shoulder"])
        self.controller.on_prediction(Prediction(
            t=now,
            hand=stable[h] if stable[h] is not None else 0,
            elbow=stable[e] if stable[e] is not None else 0,
            shoulder=stable[s] if stable[s] is not None else 0,
            p_hand=conf[h], p_elbow=conf[e], p_shoulder=conf[s],
        ), now)

    def tick(self, now: float) -> dict:
        return self.controller.tick(now)

    # -- 操作 ------------------------------------------------------------

    def engage(self) -> None:
        self.controller.engage()
        print("\n已 ARMED。按 q 結束；按 z 將 J6 歸零。")

    def handle_keys(self) -> bool:
        """處理一次按鍵。回傳 False 代表使用者要求結束。"""
        key = poll_key()
        if key in ("\r", "\n"):
            self.engage()
        elif key in ("q", "Q", "\x03"):
            return False
        elif key in ("z", "Z"):
            self.controller.zero_j6()
        return True

    def status_line(self) -> str:
        st = self.controller.status()
        return (f"[{st['state']:8s}] J2={st['J2']:+6.1f} J3={st['J3']:+6.1f} "
                f"J6={st['J6']:+6.1f} | grip={st['grip']:6s} | cmdnum={st['cmdnum']}")

    def close(self) -> None:
        try:
            self.controller.shutdown()     # 內部會自行回報是否成功回 home
        except Exception as e:             # noqa: BLE001
            print(f"[error] shutdown 失敗：{e!r}")
            print("        ⚠️ 手臂可能仍 enable，請手動用 UFACTORY Studio 停機。")
