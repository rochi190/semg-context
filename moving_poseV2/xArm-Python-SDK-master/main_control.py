#!/usr/bin/env python3
"""
main_control.py — 第三階段：整合實時控制迴圈 + 神經網路 API 介面
================================================================
架構說明：

    ┌─────────────────────────────────────────────────────────────┐
    │                     資料輸入層                                │
    │  ads1299_reader.py ──► RMSReader (串列埠/管線接收 RMS 資料)    │
    │  OR 神經網路模型 ──────► parse_nn_output()（預留介面）          │
    └────────────────────┬────────────────────────────────────────┘
                         │ channel_rms: dict[int, float]
                         │   或 ControlCommand
                         ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                  RealtimeControlLoop (主迴圈 25Hz)            │
    │                                                               │
    │  ┌─────────────────────┐  ┌──────────────────────────────┐  │
    │  │  GripperController  │  │     JointController           │  │
    │  │  CH2+CH3 → PWM      │  │  CH4/5→J2, CH6/7→J3 速度    │  │
    │  └─────────────────────┘  └──────────────────────────────┘  │
    │                                                               │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │  ArmModeManager：自動管理 mode(0)/mode(4) 切換         │  │
    │  └───────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘
                         │
                         ▼
                  Ufactory Lite6

重要限制：
    Lite6 夾爪只有 open/close 指令，無位置連續控制。
    關節速度控制（mode=4）與夾爪控制（mode=0）使用不同 mode，
    本腳本採「共存策略」：
        - 關節速度指令使用 vc_set_joint_velocity()
        - 夾爪 PWM 使用 open/close_lite6_gripper(sync=False)
        - 實驗確認：Lite6 firmware ≥ 1.10 在 mode=4 下可同時
          接受夾爪指令。若不行，可降級為「夾爪優先時短暫切回 mode=0」。

神經網路介面：
    parse_nn_output(nn_raw_output) → ControlCommand
    目前實作：直接從 channel_rms 計算（閾值法）
    未來嵌入：替換此函式內容，輸入模型推理結果，輸出相同 ControlCommand

作者：自動生成（sEMG → Lite6 控制專案）
"""

import time
import math
import threading
import signal
import sys
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

# ── 從第一、二階段引入控制器 ──────────────────────────────────────────────────
# 確保 test_gripper.py 與 test_joints.py 與本腳本在同一目錄
from test_gripper import (GripperController, GripperConfig,
                           normalize_rms, apply_deadband, SignalSmoother)
from test_joints  import (JointController,  JointConfig,
                           compute_velocity, clamp_speed_near_limit,
                           EMAFilter)

# ── xArm SDK ──────────────────────────────────────────────────────────────────
MOCK_MODE = True
ARM_IP    = '192.168.1.181'

if not MOCK_MODE:
    from xarm.wrapper import XArmAPI


# ══════════════════════════════════════════════════════════════════════════════
# 控制指令資料結構（神經網路輸出的統一格式）
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ControlCommand:
    """
    神經網路輸出轉換後的標準化控制指令。
    所有控制器均從此結構讀取輸入。

    gripper_level   : 0.0（全開）~ 1.0（全夾）
    j2_speed_deg_s  : Joint 2 目標速度（°/s），正=上，負=下
    j3_speed_deg_s  : Joint 3 目標速度（°/s），正=上，負=下
    source          : 指令來源（'threshold' | 'nn' | 'mock'）
    confidence      : 神經網路置信度（0~1，閾值法固定 1.0）
    """
    gripper_level  : float = 0.0
    j2_speed_deg_s : float = 0.0
    j3_speed_deg_s : float = 0.0
    source         : str   = 'threshold'
    confidence     : float = 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 神經網路 API 介面（預留，目前為閾值法實作）
# ══════════════════════════════════════════════════════════════════════════════
def parse_nn_output(nn_raw_output: dict | None,
                    channel_rms: dict[int, float],
                    gcfg: GripperConfig,
                    jcfg: JointConfig) -> ControlCommand:
    """
    ════════════════════════════════════════════════════
    【神經網路 API 介面】— 未來嵌入點
    ════════════════════════════════════════════════════
    當前實作：nn_raw_output=None，使用 channel_rms + 閾值法計算指令。
    未來替換：
        1. 將模型推理結果傳入 nn_raw_output（dict 或 numpy array）
        2. 在 if nn_raw_output is not None 分支中解析

    輸入：
        nn_raw_output   : 神經網路原始輸出（目前傳 None）
                          預期格式示例（CNN+LSTM 輸出）：
                          {
                              'gesture': int,        # 0=rest, 1=pinch, 2=forearm_up...
                              'confidence': float,   # 模型置信度
                              'regression': [float]  # 若為連續回歸輸出
                          }
        channel_rms     : 各通道 RMS（µV），從 ads1299_reader 取得
        gcfg, jcfg      : 夾爪與關節的參數設定

    輸出：
        ControlCommand  : 標準化控制指令
    ════════════════════════════════════════════════════
    """

    # ── 分支 A：神經網路模式（未來嵌入）─────────────────────────────────────
    if nn_raw_output is not None:
        # TODO：解析 nn_raw_output，轉換為 ControlCommand
        # 示例（手勢分類模式）：
        #
        # gesture    = nn_raw_output.get('gesture', 0)
        # confidence = nn_raw_output.get('confidence', 0.0)
        # regression = nn_raw_output.get('regression', [0.0]*4)
        #
        # GESTURE_MAP = {
        #     1: lambda: ControlCommand(gripper_level=1.0, source='nn', confidence=confidence),
        #     2: lambda: ControlCommand(j2_speed_deg_s=+20.0, source='nn', confidence=confidence),
        #     3: lambda: ControlCommand(j2_speed_deg_s=-20.0, source='nn', confidence=confidence),
        #     4: lambda: ControlCommand(j3_speed_deg_s=+20.0, source='nn', confidence=confidence),
        #     5: lambda: ControlCommand(j3_speed_deg_s=-20.0, source='nn', confidence=confidence),
        # }
        # if confidence > 0.7 and gesture in GESTURE_MAP:
        #     return GESTURE_MAP[gesture]()
        # else:
        #     return ControlCommand(source='nn')   # 靜止

        pass   # 暫無神經網路，fall through 到閾值法

    # ── 分支 B：閾值法（目前使用）────────────────────────────────────────────
    cmd = ControlCommand(source='threshold')

    # 夾爪：CH2+CH3
    rms_flex  = channel_rms.get(gcfg.ch_flex,  0.0)
    rms_pinch = channel_rms.get(gcfg.ch_pinch, 0.0)
    combined  = max(rms_flex, rms_pinch)
    norm      = normalize_rms(combined, gcfg.rms_min_uv, gcfg.rms_max_uv)
    gated     = apply_deadband(norm, gcfg.deadband)
    cmd.gripper_level = gated

    # J2：CH4 vs CH5
    r_j2_up   = channel_rms.get(jcfg.ch_j2_up,   0.0)
    r_j2_down = channel_rms.get(jcfg.ch_j2_down, 0.0)
    cmd.j2_speed_deg_s = compute_velocity(r_j2_up, r_j2_down, jcfg)

    # J3：CH6 vs CH7
    r_j3_up   = channel_rms.get(jcfg.ch_j3_up,   0.0)
    r_j3_down = channel_rms.get(jcfg.ch_j3_down, 0.0)
    cmd.j3_speed_deg_s = compute_velocity(r_j3_up, r_j3_down, jcfg)

    return cmd


# ══════════════════════════════════════════════════════════════════════════════
# ADS1299 RMS 資料接收器（對接 ads1299_realtime_pipeline.py）
# ══════════════════════════════════════════════════════════════════════════════
class RMSReader:
    """
    從 ads1299_realtime_pipeline.py 的 Queue 接收最新 RMS 資料。
    本腳本使用 Mock 模式時以正弦波替代。

    真實系統接入方法：
        reader = RMSReader(rms_queue=pipeline.rms_queue)
        # pipeline 為已啟動的 ads1299_realtime_pipeline 實例
    """

    def __init__(self, rms_queue=None,
                 gcfg: GripperConfig = GripperConfig(),
                 jcfg: JointConfig   = JointConfig()):
        self._queue  = rms_queue
        self._gcfg   = gcfg
        self._jcfg   = jcfg
        self._t0     = time.time()
        # 保存最後一次有效資料（避免 queue 暫時空時回傳 0）
        self._last: dict[int, float] = {}

    def get_latest_rms(self) -> dict[int, float]:
        """
        回傳最新一筆 channel RMS dict。
        若 queue 有資料 → 取最新；否則回傳上次值（保持）。
        """
        if self._queue is not None:
            latest = None
            # 清空 queue，只保留最新一筆
            while not self._queue.empty():
                try:
                    latest = self._queue.get_nowait()
                except Exception:
                    break
            if latest is not None:
                self._last = latest
            return dict(self._last)
        else:
            return self._mock_rms()

    def _mock_rms(self) -> dict[int, float]:
        """模擬所有 7 通道的 RMS（µV）"""
        t   = time.time() - self._t0
        mid = (self._gcfg.rms_max_uv + self._gcfg.rms_min_uv) / 2.0
        amp = (self._gcfg.rms_max_uv - self._gcfg.rms_min_uv) / 2.0

        def wave(freq, phase):
            noise = math.sin(t * 53.7 + phase) * 8.0
            return max(0.0, mid + amp * math.sin(2*math.pi*freq*t + phase) + noise)

        return {
            0: wave(0.05, 0.0),           # CH1 (ED，未使用)
            1: wave(0.25, 0.0),           # CH2 (FDS，夾爪主動)
            2: wave(0.25, 0.3),           # CH3 (FCR，夾爪輔助)
            3: wave(0.18, 0.0),           # CH4 (前臂上)
            4: wave(0.18, math.pi),       # CH5 (前臂下)
            5: wave(0.12, math.pi/2),     # CH6 (上臂上)
            6: wave(0.12, -math.pi/2),    # CH7 (上臂下)
        }


# ══════════════════════════════════════════════════════════════════════════════
# 手臂模式管理器
# ══════════════════════════════════════════════════════════════════════════════
class ArmModeManager:
    """
    管理 xArm 的 mode 切換。
    實驗確認：Lite6 firmware ≥ 1.10 在 mode=4 下仍可接受夾爪指令。
    本管理器在啟動時設定 mode=4，若有錯誤則自動重置。
    """

    def __init__(self, arm=None):
        self.arm = arm
        self._current_mode = 4

    def ensure_velocity_mode(self):
        """確保手臂在 mode=4（關節速度控制）"""
        if self.arm is not None and self.arm.mode != 4:
            self.arm.set_mode(4)
            self.arm.set_state(0)
            self._current_mode = 4

    def recover_from_error(self):
        """錯誤恢復序列"""
        if self.arm is not None:
            print("  [WARN] 偵測到錯誤，嘗試恢復...")
            self.arm.clean_error()
            self.arm.clean_warn()
            time.sleep(0.1)
            self.arm.motion_enable(enable=True)
            self.arm.set_mode(4)
            self.arm.set_state(0)

    def safe_stop(self):
        """安全停止：速度歸零，切回 mode=0"""
        if self.arm is not None:
            self.arm.vc_set_joint_velocity([0.0]*6, duration=0.0)
            time.sleep(0.05)
            self.arm.set_mode(0)
            self.arm.set_state(0)


# ══════════════════════════════════════════════════════════════════════════════
# 實時控制主迴圈
# ══════════════════════════════════════════════════════════════════════════════
class RealtimeControlLoop:
    """
    整合所有子系統的實時控制迴圈。
    預設 25Hz（CTRL_HZ），即 40ms/cycle。

    執行緒架構：
        主執行緒：控制迴圈（阻塞式 sleep 維持週期）
        可選擴充：將 RMSReader 放入獨立執行緒，
                   用 threading.Event 通知主迴圈有新資料

    ⚠️ 重要注意：
        夾爪 PWM 與關節速度控制在「同一個」主迴圈 tick 中執行。
        夾爪 execute_pwm() 內部含 sleep（佔滿 ctrl_period），
        本迴圈在 GRIPPER_CTRL_TICK（每 N tick 執行一次夾爪）
        與關節速度（每 tick 都執行）之間交替，避免衝突。
    """

    CTRL_HZ            = 25       # 關節速度控制頻率（Hz）
    GRIPPER_CTRL_EVERY = 2        # 每 N 個 velocity tick 執行一次 gripper PWM
    PRINT_EVERY        = 25       # 每 N tick 印一次狀態（約 1 秒一次）

    def __init__(self, arm=None,
                 gcfg: GripperConfig = GripperConfig(),
                 jcfg: JointConfig   = JointConfig(),
                 rms_queue           = None):
        self.arm = arm

        # 子系統
        self.reader   = RMSReader(rms_queue=rms_queue, gcfg=gcfg, jcfg=jcfg)
        self.gripper  = GripperController(arm=arm, cfg=gcfg)
        self.joints   = JointController(arm=arm, cfg=jcfg)
        self.mode_mgr = ArmModeManager(arm=arm)

        self.gcfg = gcfg
        self.jcfg = jcfg

        # 狀態
        self._running    = False
        self._tick       = 0
        self._start_time = 0.0

        # EMA（主迴圈自己的平滑器，獨立於子控制器）
        self._gripper_ema = SignalSmoother(alpha=gcfg.ema_alpha)

    # ── 初始化 ────────────────────────────────────────────────────────────────
    def _arm_init(self):
        """[xarm_api] 初始化序列，進入 mode=4"""
        if self.arm is None:
            return
        time.sleep(0.5)
        if self.arm.warn_code != 0:
            self.arm.clean_warn()
        if self.arm.error_code != 0:
            self.arm.clean_error()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(4)    # [xarm_api] 關節速度控制模式
        self.arm.set_state(0)
        print(f"  [ARM] 初始化完成，mode=4（關節速度控制）")

    # ── 單 tick 邏輯 ──────────────────────────────────────────────────────────
    def _tick_fn(self):
        """
        一個控制週期（1/CTRL_HZ 秒）的完整邏輯。
        """
        t_start = time.time()

        # ── 1. 取得 RMS 資料 ─────────────────────────────────────────────────
        channel_rms = self.reader.get_latest_rms()

        # ── 2. 呼叫神經網路 API 介面 ─────────────────────────────────────────
        # 真實系統：將模型推理結果傳入 nn_raw_output
        # 目前傳 None → 使用閾值法
        cmd = parse_nn_output(
            nn_raw_output=None,   # ← 未來：替換為 model.predict(window)
            channel_rms=channel_rms,
            gcfg=self.gcfg,
            jcfg=self.jcfg
        )

        # ── 3. 關節速度控制（每 tick）─────────────────────────────────────────
        angle_j2, angle_j3 = self.joints._get_angles()
        clamped_j2 = clamp_speed_near_limit(
            cmd.j2_speed_deg_s, angle_j2,
            self.jcfg.j2_angle_min, self.jcfg.j2_angle_max,
            self.jcfg.angle_margin
        )
        clamped_j3 = clamp_speed_near_limit(
            cmd.j3_speed_deg_s, angle_j3,
            self.jcfg.j3_angle_min, self.jcfg.j3_angle_max,
            self.jcfg.angle_margin
        )

        speeds = [0.0] * 6
        speeds[1] = clamped_j2
        speeds[2] = clamped_j3
        if self.arm is not None:
            # [xarm_api] 關節速度指令，duration=0.12s 確保中斷後自停
            self.arm.vc_set_joint_velocity(
                speeds, is_radian=False, is_sync=True,
                duration=self.jcfg.vel_duration_s
            )
        elif clamped_j2 != 0 or clamped_j3 != 0:
            print(f"    [MOCK-J] J2={clamped_j2:+.1f}°/s  J3={clamped_j3:+.1f}°/s")

        # ── 4. 夾爪比例控制（每 N tick 執行一次 PWM）────────────────────────
        if self._tick % self.GRIPPER_CTRL_EVERY == 0:
            # EMA 平滑 gripper_level
            smooth_level = self._gripper_ema.update(cmd.gripper_level)
            # 單次 PWM 週期（短週期，避免阻塞太久）
            T        = self.gcfg.ctrl_period_s
            t_close  = T * smooth_level
            t_open   = T * (1.0 - smooth_level)
            min_p    = self.gcfg.min_pulse_s
            if t_close > min_p:
                if self.arm is not None:
                    self.arm.close_lite6_gripper(sync=False)
                else:
                    print(f"    [MOCK-G] close ({t_close*1000:.0f}ms)")
                time.sleep(t_close)
            if t_open > min_p:
                if self.arm is not None:
                    self.arm.open_lite6_gripper(sync=False)
                else:
                    print(f"    [MOCK-G] open  ({t_open*1000:.0f}ms)")
                time.sleep(t_open)

        # ── 5. 狀態列印 ──────────────────────────────────────────────────────
        if self._tick % self.PRINT_EVERY == 0:
            self._print_status(cmd, clamped_j2, clamped_j3, angle_j2, angle_j3)

        # ── 6. 錯誤監控 ──────────────────────────────────────────────────────
        if self.arm is not None and self.arm.has_error:
            self.mode_mgr.recover_from_error()

        # ── 7. 週期補償 sleep ─────────────────────────────────────────────────
        elapsed = time.time() - t_start
        period  = 1.0 / self.CTRL_HZ
        sleep_t = max(0.0, period - elapsed)
        time.sleep(sleep_t)
        self._tick += 1

    def _print_status(self, cmd, j2_spd, j3_spd, ang_j2, ang_j3):
        elapsed = time.time() - self._start_time
        bar_g   = '█' * int(cmd.gripper_level * 20) + '░' * (20 - int(cmd.gripper_level * 20))
        print(f"[{elapsed:6.1f}s | tick={self._tick:5d} | {cmd.source}]  "
              f"Grip={cmd.gripper_level:.2f} [{bar_g}]  "
              f"J2={j2_spd:+5.1f}°/s(∠{ang_j2:+5.1f}°)  "
              f"J3={j3_spd:+5.1f}°/s(∠{ang_j3:+5.1f}°)")

    # ── 啟動與停止 ─────────────────────────────────────────────────────────────
    def start(self):
        print("\n[RealtimeControlLoop] 啟動")
        self._arm_init()
        self._running    = True
        self._start_time = time.time()
        self._tick       = 0

        print(f"  控制頻率：{self.CTRL_HZ} Hz")
        print(f"  夾爪 PWM 週期：每 {self.GRIPPER_CTRL_EVERY} tick")
        print(f"  按 Ctrl+C 安全停止\n")

        while self._running:
            try:
                self._tick_fn()
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"  [ERROR] tick 異常：{e}")
                time.sleep(0.1)

        self.stop()

    def stop(self):
        print("\n[RealtimeControlLoop] 停止...")
        self._running = False
        # 安全停止：速度歸零 + 夾爪開啟
        if self.arm is not None:
            speeds = [0.0] * 6
            self.arm.vc_set_joint_velocity(speeds, duration=0.0)
            self.arm.open_lite6_gripper(sync=True)
            self.arm.stop_lite6_gripper(sync=True)
            time.sleep(0.2)
            self.arm.set_mode(0)
            self.arm.set_state(0)
            self.arm.disconnect()
        else:
            print("  [MOCK] 停止，釋放所有控制")

        elapsed = time.time() - self._start_time
        print(f"  總執行時間：{elapsed:.1f}s，{self._tick} 個 tick，"
              f"平均 {elapsed/max(self._tick,1)*1000:.1f} ms/tick")
        print("完成。")


# ══════════════════════════════════════════════════════════════════════════════
# 快速設定入口（修改此處替換預設參數）
# ══════════════════════════════════════════════════════════════════════════════
def make_default_configs():
    """
    回傳適合 sEMG 前臂量測的預設設定組。
    校準後請修改 rms_min_uv / rms_max_uv。
    """
    gcfg = GripperConfig(
        ch_flex      = 1,      # CH2 (FDS)
        ch_pinch     = 2,      # CH3 (FCR)
        rms_min_uv   = 10.0,   # 靜止噪音，校準後填入
        rms_max_uv   = 300.0,  # MVC，校準後填入
        deadband     = 0.05,
        ema_alpha    = 0.3,
        ctrl_period_s= 0.08,
        min_pulse_s  = 0.02,
    )
    jcfg = JointConfig(
        ch_j2_up     = 3,      # CH4
        ch_j2_down   = 4,      # CH5
        ch_j3_up     = 5,      # CH6
        ch_j3_down   = 6,      # CH7
        rms_min_uv   = 10.0,
        rms_max_uv   = 250.0,
        deadband     = 0.08,
        ema_alpha    = 0.4,
        max_speed_deg_s = 30.0,
        j2_angle_min = -60.0,
        j2_angle_max =  60.0,
        j3_angle_min = -30.0,
        j3_angle_max =  80.0,
        angle_margin =   5.0,
        ctrl_period_s=  0.05,
        vel_duration_s= 0.12,
    )
    return gcfg, jcfg


# ══════════════════════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("main_control.py — sEMG 實時機械手臂控制（整合版）")
    print(f"模式：{'Mock（模擬）' if MOCK_MODE else f'真實手臂 {ARM_IP}'}")
    print("=" * 65)

    gcfg, jcfg = make_default_configs()

    # 真實系統接入：
    #   from xarm.wrapper import XArmAPI
    #   arm = XArmAPI(ARM_IP)
    arm = None if MOCK_MODE else XArmAPI(ARM_IP)

    # 真實系統接入 ads1299_realtime_pipeline：
    #   from ads1299_realtime_pipeline import EMGPipeline
    #   pipeline = EMGPipeline(port='COM3', baudrate=115200)
    #   pipeline.start()
    #   rms_queue = pipeline.rms_queue
    rms_queue = None   # Mock 模式：RMSReader 內部自動生成

    loop = RealtimeControlLoop(
        arm       = arm,
        gcfg      = gcfg,
        jcfg      = jcfg,
        rms_queue = rms_queue,
    )
    loop.start()


if __name__ == '__main__':
    main()
