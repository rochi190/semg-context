"""讀出 Lite 6 的規格參數（唯讀，不會讓手臂動）。

放在 moving_pose/xArm-Python-SDK-master/read_arm_specs.py
用法：python xArm-Python-SDK-master/read_arm_specs.py

⚠️ SDK 1.18.5 沒有 arm.joint_limits 這個屬性（早期版本這樣寫過，一跑就
   AttributeError）。reduced_joint_limits 是「精簡模式」限位，不是出廠硬
   限位，但 2026-08-17 讀出值與 Lite 6 硬體手冊 V2.6.0 逐項相符。
⚠️ XArmAPI.__init__ 內部就會呼叫 connect()（base.py:282），IP 不可達時
   直接 raise Exception('connect socket failed')——建構子不是安全的，
   所以不能寫成「先建構、再 assert connected」。
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                      # 本檔就在 SDK 根目錄內

from xarm.wrapper import XArmAPI                   # noqa: E402

ARM_IP = "192.168.1.166"

print(f"連線 {ARM_IP} …")
try:
    arm = XArmAPI(ARM_IP, is_radian=False)
except Exception as e:                             # noqa: BLE001
    raise SystemExit(
        f"✖ 連線失敗：{e}\n"
        f"  1. 控制箱電源與網路線\n"
        f"  2. ping {ARM_IP} 通不通\n"
        f"  3. UFACTORY Studio 是否正佔用連線") from None

time.sleep(1.0)                                    # 等 report 通道建立，不可省略
if not arm.connected:
    arm.disconnect()
    raise SystemExit("✖ 建構成功但 connected 仍為 False")

print("-" * 60)
print("firmware              :", arm.version)
print("sn                    :", arm.sn)
print("axis                  :", arm.axis)
print("reduced_joint_limits  :", arm.reduced_joint_limits)
print("joint_speed_limit     :", arm.joint_speed_limit)
print("joint_acc_limit       :", arm.joint_acc_limit)
print("collision_sensitivity :", arm.collision_sensitivity)
print("angles (current)      :", arm.angles)
print("state / mode          :", arm.state, "/", arm.mode)
print("error_code/warn_code  :", arm.error_code, "/", arm.warn_code)
print("-" * 60)
arm.disconnect()
print("已斷線。")
