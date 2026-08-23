#!/usr/bin/env python3
"""
test_gripper.py — 第一階段：Lite6 夾爪比例控制獨立測試腳本
================================================================
訊號源：CH2（Flexor Digitorum Superficialis，FDS）+
         CH3（Flexor Carpi Radialis，FCR）的 RMS 合成強度

Lite6 夾爪限制說明：
    Lite6 只有 open_lite6_gripper() / close_lite6_gripper() 兩個離散指令，
    無位置連續控制 API。此處採用「PWM 時間切割法」：
        - 在每個控制週期 (CTRL_PERIOD_S) 內，
          依 gripper_level (0.0~1.0) 分配 close 與 open 的持續時間
        - gripper_level=1.0 → 全程 close（完全夾緊）
        - gripper_level=0.0 → 全程 open（完全張開）
        - 中間值 → 週期內切換，模擬中間位置
    此法在低頻感知下足夠實用；若需真正的連續位置控制，
    需改用支援 set_gripper_position(pos) 的 xArm Gripper 型號。

Mock Data 模式：
    MOCK_MODE = True  → 使用正弦波模擬 sEMG 強度（不連接手臂）
    MOCK_MODE = False → 接受外部傳入的 channel_rms dict

作者：自動生成（sEMG → Lite6 控制專案）
"""

import time
import math
import threading
from dataclasses import dataclass, field
from collections import deque

# ── xArm SDK ──────────────────────────────────────────────────────────────────
# [xarm_api 呼叫點] 連線時替換 IP，MOCK_MODE=False 才實際連線
MOCK_MODE = True          # True=模擬，False=連接真實手臂
ARM_IP    = '192.168.1.181'

if not MOCK_MODE:
    from xarm.wrapper import XArmAPI


# ══════════════════════════════════════════════════════════════════════════════
# 可調參數
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class GripperConfig:
    # 訊號通道（0-indexed，對應 ads1299_reader 輸出的 ch 索引）
    ch_flex: int   = 1   # CH2 → FDS（屈指）
    ch_pinch: int  = 2   # CH3 → FCR（捏合輔助）

    # 訊號正規化範圍（單位：µV RMS）
    # 建議先量測靜止與最大自願收縮（MVC）後填入
    rms_min_uv: float = 10.0    # 死區上界（靜止噪音水位）
    rms_max_uv: float = 300.0   # MVC 對應的 RMS（需量測後校正）

    # 死區：低於此正規化值視為靜止，不驅動夾爪
    deadband: float = 0.05      # 正規化後的死區門檻 (0.0~1.0)

    # EMA 平滑係數（0→不平滑，接近 1→極慢響應）
    ema_alpha: float = 0.3

    # PWM 控制週期（秒）—— 每個週期內切換一次 open/close
    ctrl_period_s: float = 0.08   # 建議 60~120ms，太短馬達嗡嗡聲明顯

    # 最小夾爪動作時間（s）—— 避免過短脈衝傷馬達
    min_pulse_s: float = 0.02


CFG = GripperConfig()


# ══════════════════════════════════════════════════════════════════════════════
# 訊號處理工具
# ══════════════════════════════════════════════════════════════════════════════
class SignalSmoother:
    """單通道 EMA（指數移動平均）平滑器"""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._value: float | None = None

    def update(self, raw: float) -> float:
        if self._value is None:
            self._value = raw
        else:
            self._value = self.alpha * raw + (1.0 - self.alpha) * self._value
        return self._value

    @property
    def value(self) -> float:
        return self._value if self._value is not None else 0.0


def normalize_rms(rms_uv: float, rms_min: float, rms_max: float) -> float:
    """
    將原始 RMS（µV）正規化至 [0.0, 1.0]。
    低於 rms_min 夾至 0，高於 rms_max 夾至 1。
    """
    if rms_max <= rms_min:
        return 0.0
    val = (rms_uv - rms_min) / (rms_max - rms_min)
    return max(0.0, min(1.0, val))


def apply_deadband(normalized: float, deadband: float) -> float:
    """
    套用死區並重新線性映射：
        [0, deadband] → 0
        [deadband, 1] → [0, 1]
    """
    if normalized < deadband:
        return 0.0
    return (normalized - deadband) / (1.0 - deadband)


# ══════════════════════════════════════════════════════════════════════════════
# 夾爪控制器
# ══════════════════════════════════════════════════════════════════════════════
class GripperController:
    """
    Lite6 夾爪比例控制器。

    核心邏輯：
        1. 輸入 channel_rms dict（單位 µV）
        2. 合成 CH2+CH3 → 正規化 → 死區 → EMA → gripper_level [0,1]
        3. PWM 時間切割輸出 close/open 指令
    """

    def __init__(self, arm=None, cfg: GripperConfig = CFG):
        self.arm  = arm
        self.cfg  = cfg
        self._smoother = SignalSmoother(alpha=cfg.ema_alpha)
        self.gripper_level: float = 0.0   # 最終輸出，公開供 main_control 讀取
        self._last_state: str = 'open'    # 'open' | 'close'
        self._lock = threading.Lock()

    # ── 核心訊號處理 ─────────────────────────────────────────────────────────
    def process_channels(self, channel_rms: dict[int, float]) -> float:
        """
        輸入：channel_rms = {ch_index: rms_uv, ...}
        輸出：gripper_level [0.0, 1.0]

        合成方式：取 CH2、CH3 的最大值（誰先動誰主導），
        未來可改為平均或加權和。
        """
        rms_flex  = channel_rms.get(self.cfg.ch_flex,  0.0)
        rms_pinch = channel_rms.get(self.cfg.ch_pinch, 0.0)
        combined  = max(rms_flex, rms_pinch)   # 或改 (rms_flex + rms_pinch) / 2

        norm   = normalize_rms(combined, self.cfg.rms_min_uv, self.cfg.rms_max_uv)
        gated  = apply_deadband(norm, self.cfg.deadband)
        smooth = self._smoother.update(gated)

        with self._lock:
            self.gripper_level = smooth
        return smooth

    # ── PWM 輸出 ─────────────────────────────────────────────────────────────
    def execute_pwm(self, level: float):
        """
        在一個 ctrl_period 內用 PWM 時間切割驅動夾爪。
        level=0 → 全 open；level=1 → 全 close。
        """
        T      = self.cfg.ctrl_period_s
        t_close = T * level
        t_open  = T * (1.0 - level)

        if t_close > self.cfg.min_pulse_s:
            self._send_close()
            time.sleep(t_close)

        if t_open > self.cfg.min_pulse_s:
            self._send_open()
            time.sleep(t_open)
        elif t_close <= self.cfg.min_pulse_s:
            # 兩段都太短 → 保持靜止，填補剩餘時間
            time.sleep(T)

    # ── xArm API 呼叫點 ──────────────────────────────────────────────────────
    def _send_close(self):
        """
        [xarm_api] close_lite6_gripper(sync=False)
        sync=False：立即執行，不進入佇列，適合實時控制迴圈。
        """
        if self.arm is not None:
            self.arm.close_lite6_gripper(sync=False)
        else:
            print(f"  [MOCK] close_lite6_gripper()")
        self._last_state = 'close'

    def _send_open(self):
        """
        [xarm_api] open_lite6_gripper(sync=False)
        """
        if self.arm is not None:
            self.arm.open_lite6_gripper(sync=False)
        else:
            print(f"  [MOCK] open_lite6_gripper()")
        self._last_state = 'open'

    def release(self):
        """安全釋放：確保夾爪最終停在開啟狀態"""
        if self.arm is not None:
            self.arm.open_lite6_gripper(sync=True)
            self.arm.stop_lite6_gripper(sync=True)
        else:
            print("  [MOCK] gripper released (open)")


# ══════════════════════════════════════════════════════════════════════════════
# Mock Data 產生器
# ══════════════════════════════════════════════════════════════════════════════
class MockEMGSource:
    """
    模擬 sEMG RMS 輸出（替代神經網路/ads1299_reader）。
    波形：緩慢正弦（模擬漸進用力→放鬆），加上少量噪音。
    """

    def __init__(self, cfg: GripperConfig = CFG, freq_hz: float = 0.3):
        self.cfg = cfg
        self.freq = freq_hz
        self._t0  = time.time()

    def get_rms(self) -> dict[int, float]:
        t = time.time() - self._t0
        # 正弦波：振幅在 [rms_min, rms_max] 之間週期變化
        mid    = (self.cfg.rms_max_uv + self.cfg.rms_min_uv) / 2.0
        amp    = (self.cfg.rms_max_uv - self.cfg.rms_min_uv) / 2.0
        signal = mid + amp * math.sin(2 * math.pi * self.freq * t)
        noise  = (math.sin(t * 37.1) + math.cos(t * 19.3)) * 5.0   # ±5µV 噪音
        rms    = max(0.0, signal + noise)
        return {
            self.cfg.ch_flex:  rms,
            self.cfg.ch_pinch: rms * 0.8,   # CH3 稍弱
        }


# ══════════════════════════════════════════════════════════════════════════════
# 手臂初始化工具
# ══════════════════════════════════════════════════════════════════════════════
def init_arm(ip: str):
    """
    [xarm_api] 標準初始化序列。
    set_mode(0)：位置控制模式，夾爪指令在此模式下有效。
    """
    arm = XArmAPI(ip)
    time.sleep(0.5)
    if arm.warn_code != 0:
        arm.clean_warn()
    if arm.error_code != 0:
        arm.clean_error()
    arm.motion_enable(enable=True)
    arm.set_mode(0)     # [xarm_api] 位置模式：Lite6 夾爪指令所需
    arm.set_state(0)
    return arm


# ══════════════════════════════════════════════════════════════════════════════
# 主測試迴圈
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("test_gripper.py — Lite6 夾爪比例控制測試")
    print(f"模式：{'Mock（模擬）' if MOCK_MODE else f'真實手臂 {ARM_IP}'}")
    print("=" * 60)

    # 初始化手臂
    arm = None if MOCK_MODE else init_arm(ARM_IP)

    # 建立控制器與資料源
    gripper_ctrl = GripperController(arm=arm, cfg=CFG)
    mock_src     = MockEMGSource(cfg=CFG)

    print(f"\n參數設定：")
    print(f"  通道：CH{CFG.ch_flex+1}（FDS） + CH{CFG.ch_pinch+1}（FCR）")
    print(f"  RMS 範圍：{CFG.rms_min_uv:.0f} ~ {CFG.rms_max_uv:.0f} µV")
    print(f"  死區：{CFG.deadband:.0%}，EMA α={CFG.ema_alpha}")
    print(f"  控制週期：{CFG.ctrl_period_s*1000:.0f} ms\n")
    print("按 Ctrl+C 結束\n")

    loop_count  = 0
    try:
        while True:
            # ── 取得 RMS 資料（真實系統替換此行為 reader.get_latest_rms()）──
            channel_rms = mock_src.get_rms()

            # ── 訊號處理 ─────────────────────────────────────────────────────
            level = gripper_ctrl.process_channels(channel_rms)

            # ── 狀態列印（每 5 圈印一次，避免刷屏）─────────────────────────
            if loop_count % 5 == 0:
                ch2_val = channel_rms.get(CFG.ch_flex, 0)
                ch3_val = channel_rms.get(CFG.ch_pinch, 0)
                bar_len = int(level * 30)
                bar     = '█' * bar_len + '░' * (30 - bar_len)
                print(f"  CH2={ch2_val:6.1f}µV  CH3={ch3_val:6.1f}µV  "
                      f"level={level:.3f}  [{bar}]")

            # ── PWM 執行（此函式內部含 sleep，已佔滿一個 ctrl_period）───────
            gripper_ctrl.execute_pwm(level)
            loop_count += 1

    except KeyboardInterrupt:
        print("\n\n中斷，正在安全釋放夾爪...")
        gripper_ctrl.release()
        if arm is not None:
            arm.disconnect()
        print("完成。")


if __name__ == '__main__':
    main()
