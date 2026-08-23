#!/usr/bin/env python3
"""
test_joints.py — 第二階段：前臂／上臂關節速度控制獨立測試腳本
================================================================
訊號源與關節對映：
    Joint 2（前臂關節）：CH4（前臂抬起）vs CH5（前臂放下）
    Joint 3（上臂關節）：CH6（上臂抬起）vs CH7（上臂放下）

控制方法：
    使用 set_mode(4) + vc_set_joint_velocity()（關節速度控制模式）。
    根據拮抗通道 RMS 差值（agonist−antagonist）決定旋轉方向與速度。

        差值 > +deadband → 正方向旋轉（速度 ∝ 差值）
        差值 < -deadband → 負方向旋轉
        |差值| < deadband → 速度 = 0（靜止）

安全機制：
    1. 角度軟極限（Angle Soft Limit）：每個控制週期讀取當前關節角，
       若接近極限則強制速度歸零。
    2. 最大速度限制（MAX_JOINT_SPEED_DEG_S）。
    3. vc_set_joint_velocity(duration=0.12)：每個指令僅持續 120ms，
       若控制迴圈中斷，手臂自動停止（安全保護）。

Mock Data 模式：
    MOCK_MODE = True  → 不連接手臂，印出速度指令
    MOCK_MODE = False → 使用真實 xArm

作者：自動生成（sEMG → Lite6 控制專案）
"""

import time
import math
import threading
from dataclasses import dataclass

# ── xArm SDK ──────────────────────────────────────────────────────────────────
MOCK_MODE = True
ARM_IP    = '192.168.1.181'

if not MOCK_MODE:
    from xarm.wrapper import XArmAPI


# ══════════════════════════════════════════════════════════════════════════════
# 可調參數
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class JointConfig:
    # ── 通道對映（0-indexed）─────────────────────────────────────────────────
    # Joint 2 對應 Lite6 的 servo_id=2（肩部俯仰）
    ch_j2_up:   int = 3    # CH4，前臂抬起 → J2 正轉
    ch_j2_down: int = 4    # CH5，前臂放下 → J2 負轉
    # Joint 3 對應 Lite6 的 servo_id=3（肘部）
    ch_j3_up:   int = 5    # CH6，上臂抬起 → J3 正轉
    ch_j3_down: int = 6    # CH7，上臂放下 → J3 負轉

    # ── 訊號正規化（µV RMS）──────────────────────────────────────────────────
    rms_min_uv: float = 10.0
    rms_max_uv: float = 250.0

    # ── 死區（正規化後）──────────────────────────────────────────────────────
    deadband: float = 0.08

    # ── EMA 平滑 ─────────────────────────────────────────────────────────────
    ema_alpha: float = 0.4

    # ── 速度限制（°/s）───────────────────────────────────────────────────────
    max_speed_deg_s: float = 30.0    # 最大關節速度，建議先從 20~30 開始
    min_speed_deg_s: float = 2.0     # 低於此值視為靜止（避免微小抖動）

    # ── 角度軟極限（°）───────────────────────────────────────────────────────
    # 請根據實際裝置與場景調整，以下為保守估計
    j2_angle_min: float = -60.0
    j2_angle_max: float =  60.0
    j3_angle_min: float = -30.0
    j3_angle_max: float =  80.0
    angle_margin: float =  5.0     # 接近極限前 5° 開始減速

    # ── 控制週期（s）─────────────────────────────────────────────────────────
    ctrl_period_s: float = 0.05    # 20Hz 控制迴圈

    # ── velocity duration（s）────────────────────────────────────────────────
    # vc_set_joint_velocity 的 duration 參數：指令有效時間
    # 設為 ctrl_period * 2.5 保留餘裕，避免停頓，同時確保中斷後快速停止
    vel_duration_s: float = 0.12


JCFG = JointConfig()


# ══════════════════════════════════════════════════════════════════════════════
# 訊號處理工具
# ══════════════════════════════════════════════════════════════════════════════
class EMAFilter:
    """雙通道 EMA（供拮抗通道對使用）"""

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self._vals: dict[int, float] = {}

    def update(self, ch: int, raw: float) -> float:
        if ch not in self._vals:
            self._vals[ch] = raw
        else:
            self._vals[ch] = self.alpha * raw + (1.0 - self.alpha) * self._vals[ch]
        return self._vals[ch]


def normalize_rms(rms_uv: float, rms_min: float, rms_max: float) -> float:
    if rms_max <= rms_min:
        return 0.0
    return max(0.0, min(1.0, (rms_uv - rms_min) / (rms_max - rms_min)))


def compute_velocity(rms_agonist: float,
                     rms_antagonist: float,
                     cfg: JointConfig) -> float:
    """
    計算關節速度（°/s），正值=正轉，負值=反轉，0=靜止。

    步驟：
        1. 分別正規化主動與拮抗通道
        2. 差值 = 主動 − 拮抗
        3. 套用死區
        4. 線性映射至 [−max_speed, +max_speed]
    """
    norm_ag   = normalize_rms(rms_agonist,    cfg.rms_min_uv, cfg.rms_max_uv)
    norm_ant  = normalize_rms(rms_antagonist, cfg.rms_min_uv, cfg.rms_max_uv)
    diff      = norm_ag - norm_ant

    # 死區
    if abs(diff) < cfg.deadband:
        return 0.0

    # 死區補償：重新映射至 [-1, 1]
    sign = 1.0 if diff > 0 else -1.0
    scaled = sign * (abs(diff) - cfg.deadband) / (1.0 - cfg.deadband)
    speed  = scaled * cfg.max_speed_deg_s

    # 最小速度門檻
    if abs(speed) < cfg.min_speed_deg_s:
        return 0.0
    return speed


def clamp_speed_near_limit(speed: float,
                            current_angle: float,
                            angle_min: float,
                            angle_max: float,
                            margin: float) -> float:
    """
    接近角度軟極限時線性衰減速度，到達極限時強制歸零。
        - current_angle < angle_min + margin 且 speed < 0 → 減速/歸零
        - current_angle > angle_max - margin 且 speed > 0 → 減速/歸零
    """
    if speed < 0 and current_angle < angle_min + margin:
        ratio = max(0.0, (current_angle - angle_min) / margin)
        return speed * ratio
    if speed > 0 and current_angle > angle_max - margin:
        ratio = max(0.0, (angle_max - current_angle) / margin)
        return speed * ratio
    return speed


# ══════════════════════════════════════════════════════════════════════════════
# 關節控制器
# ══════════════════════════════════════════════════════════════════════════════
class JointController:
    """
    雙關節（J2、J3）速度比例控制器。
    使用 xArm mode=4（關節速度控制模式）。

    注意：
        set_mode(4) 之後 set_servo_angle() 無效，
        只能用 vc_set_joint_velocity() 控制速度。
        若需回到位置控制，必須先 set_mode(0)。
    """

    # Lite6 關節數（6 軸），速度陣列索引 0~5 對應 J1~J6
    JOINT_COUNT = 6

    def __init__(self, arm=None, cfg: JointConfig = JCFG):
        self.arm  = arm
        self.cfg  = cfg
        self._ema = EMAFilter(alpha=cfg.ema_alpha)

        # 模擬當前角度（Mock 模式用）
        self._mock_angle_j2: float = 0.0
        self._mock_angle_j3: float = 0.0

        # 上一次速度指令（用於抑制重複送出 0 速）
        self._last_speed_j2: float = None
        self._last_speed_j3: float = None

    # ── 讀取當前關節角 ────────────────────────────────────────────────────────
    def _get_angles(self) -> tuple[float, float]:
        """
        [xarm_api] get_servo_angle(is_real=True)
        回傳 (code, [J1..J6])，取索引 1(J2) 與 2(J3)。
        Mock 模式下用積分模擬。
        """
        if self.arm is not None:
            code, angles = self.arm.get_servo_angle(is_real=True)
            if code == 0 and angles is not None:
                return float(angles[1]), float(angles[2])
        return self._mock_angle_j2, self._mock_angle_j3

    # ── 速度指令發送 ──────────────────────────────────────────────────────────
    def _send_velocity(self, speed_j2: float, speed_j3: float):
        """
        [xarm_api] vc_set_joint_velocity(speeds, duration=vel_duration_s)
            speeds: 7 元素 list（Lite6 實為 6 軸，填 0 補位）
            duration: 指令有效時間（s），超時自動停止 → 安全保護
        """
        speeds = [0.0] * self.JOINT_COUNT
        speeds[1] = speed_j2   # J2，索引 1
        speeds[2] = speed_j3   # J3，索引 2

        if self.arm is not None:
            # [xarm_api] 呼叫點：關節速度控制
            code = self.arm.vc_set_joint_velocity(
                speeds,
                is_radian=False,
                is_sync=True,
                duration=self.cfg.vel_duration_s
            )
            if code != 0:
                print(f"  [WARN] vc_set_joint_velocity 回傳 code={code}")
        else:
            if speed_j2 != 0 or speed_j3 != 0:
                print(f"  [MOCK] vc_set_joint_velocity: J2={speed_j2:+.1f}°/s, "
                      f"J3={speed_j3:+.1f}°/s  (duration={self.cfg.vel_duration_s}s)")

        # Mock 模式：積分更新模擬角度
        dt = self.cfg.ctrl_period_s
        self._mock_angle_j2 = max(self.cfg.j2_angle_min,
                                   min(self.cfg.j2_angle_max,
                                       self._mock_angle_j2 + speed_j2 * dt))
        self._mock_angle_j3 = max(self.cfg.j3_angle_min,
                                   min(self.cfg.j3_angle_max,
                                       self._mock_angle_j3 + speed_j3 * dt))

    # ── 主更新函式 ────────────────────────────────────────────────────────────
    def update(self, channel_rms: dict[int, float]) -> tuple[float, float]:
        """
        輸入：channel_rms = {ch_index: rms_uv}
        輸出：(speed_j2, speed_j3) 實際下達的速度（°/s）
        """
        # EMA 平滑各通道
        r_j2_up   = self._ema.update(self.cfg.ch_j2_up,   channel_rms.get(self.cfg.ch_j2_up,   0.0))
        r_j2_down = self._ema.update(self.cfg.ch_j2_down, channel_rms.get(self.cfg.ch_j2_down, 0.0))
        r_j3_up   = self._ema.update(self.cfg.ch_j3_up,   channel_rms.get(self.cfg.ch_j3_up,   0.0))
        r_j3_down = self._ema.update(self.cfg.ch_j3_down, channel_rms.get(self.cfg.ch_j3_down, 0.0))

        # 計算原始速度（拮抗差值）
        raw_j2 = compute_velocity(r_j2_up, r_j2_down, self.cfg)
        raw_j3 = compute_velocity(r_j3_up, r_j3_down, self.cfg)

        # 讀取當前角度（用於軟極限保護）
        angle_j2, angle_j3 = self._get_angles()

        # 軟極限夾速
        clamped_j2 = clamp_speed_near_limit(
            raw_j2, angle_j2,
            self.cfg.j2_angle_min, self.cfg.j2_angle_max,
            self.cfg.angle_margin
        )
        clamped_j3 = clamp_speed_near_limit(
            raw_j3, angle_j3,
            self.cfg.j3_angle_min, self.cfg.j3_angle_max,
            self.cfg.angle_margin
        )

        # 下達速度指令
        self._send_velocity(clamped_j2, clamped_j3)
        return clamped_j2, clamped_j3

    def stop(self):
        """強制停止：發送全零速度"""
        self._send_velocity(0.0, 0.0)
        if self.arm is not None:
            # 停止後切回位置模式較安全
            self.arm.set_mode(0)
            self.arm.set_state(0)


# ══════════════════════════════════════════════════════════════════════════════
# Mock Data 產生器
# ══════════════════════════════════════════════════════════════════════════════
class MockEMGSource:
    """
    模擬 4 通道 sEMG：
        CH4/CH5（J2）：低頻正弦，交替主導
        CH6/CH7（J3）：相位差 90° 的正弦，交替主導
    """

    def __init__(self, cfg: JointConfig = JCFG):
        self.cfg = cfg
        self._t0 = time.time()

    def get_rms(self) -> dict[int, float]:
        t   = time.time() - self._t0
        mid = (self.cfg.rms_max_uv + self.cfg.rms_min_uv) / 2.0
        amp = (self.cfg.rms_max_uv - self.cfg.rms_min_uv) / 2.0

        def wave(freq, phase):
            return max(0.0, mid + amp * math.sin(2 * math.pi * freq * t + phase))

        return {
            self.cfg.ch_j2_up:   wave(0.2, 0.0),      # 前臂上抬
            self.cfg.ch_j2_down: wave(0.2, math.pi),   # 前臂下放（反相）
            self.cfg.ch_j3_up:   wave(0.15, math.pi/2),  # 上臂上抬
            self.cfg.ch_j3_down: wave(0.15, -math.pi/2), # 上臂下放（反相）
        }


# ══════════════════════════════════════════════════════════════════════════════
# 手臂初始化（速度控制模式）
# ══════════════════════════════════════════════════════════════════════════════
def init_arm_velocity_mode(ip: str):
    """
    [xarm_api] 初始化並切換至關節速度控制模式（mode=4）。
    ⚠️ 進入 mode=4 後 set_servo_angle() 無效。
    """
    arm = XArmAPI(ip)
    time.sleep(0.5)
    if arm.warn_code != 0:
        arm.clean_warn()
    if arm.error_code != 0:
        arm.clean_error()
    arm.motion_enable(enable=True)
    arm.set_mode(4)     # [xarm_api] 關節速度控制模式
    arm.set_state(0)
    return arm


# ══════════════════════════════════════════════════════════════════════════════
# 主測試迴圈
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("test_joints.py — J2/J3 關節速度控制測試")
    print(f"模式：{'Mock（模擬）' if MOCK_MODE else f'真實手臂 {ARM_IP}'}")
    print("=" * 60)

    arm  = None if MOCK_MODE else init_arm_velocity_mode(ARM_IP)
    ctrl = JointController(arm=arm, cfg=JCFG)
    src  = MockEMGSource(cfg=JCFG)

    print(f"\n參數設定：")
    print(f"  J2 通道：CH{JCFG.ch_j2_up+1}(上) / CH{JCFG.ch_j2_down+1}(下)")
    print(f"  J3 通道：CH{JCFG.ch_j3_up+1}(上) / CH{JCFG.ch_j3_down+1}(下)")
    print(f"  最大速度：±{JCFG.max_speed_deg_s}°/s")
    print(f"  J2 軟極限：{JCFG.j2_angle_min}° ~ {JCFG.j2_angle_max}°")
    print(f"  J3 軟極限：{JCFG.j3_angle_min}° ~ {JCFG.j3_angle_max}°")
    print(f"  控制頻率：{1/JCFG.ctrl_period_s:.0f} Hz\n")
    print("按 Ctrl+C 結束\n")

    loop_count = 0
    t_loop_start = time.time()

    try:
        while True:
            t_start = time.time()

            # ── 取得 RMS 資料 ──────────────────────────────────────────────
            channel_rms = src.get_rms()

            # ── 更新關節速度 ───────────────────────────────────────────────
            spd_j2, spd_j3 = ctrl.update(channel_rms)

            # ── 狀態列印（每 10 圈一次）───────────────────────────────────
            if loop_count % 10 == 0:
                angle_j2, angle_j3 = ctrl._get_angles()

                def speed_bar(s, max_s):
                    """雙向速度條"""
                    half = 15
                    filled = int(abs(s) / max_s * half)
                    if s >= 0:
                        return ' ' * half + '│' + '▶' * filled + ' ' * (half - filled)
                    else:
                        return ' ' * (half - filled) + '◀' * filled + '│' + ' ' * half

                print(f"  J2: {spd_j2:+6.1f}°/s {speed_bar(spd_j2, JCFG.max_speed_deg_s)}  "
                      f"角度={angle_j2:+6.1f}°")
                print(f"  J3: {spd_j3:+6.1f}°/s {speed_bar(spd_j3, JCFG.max_speed_deg_s)}  "
                      f"角度={angle_j3:+6.1f}°")
                print()

            # ── 精確週期控制 ───────────────────────────────────────────────
            elapsed = time.time() - t_start
            sleep_t = max(0.0, JCFG.ctrl_period_s - elapsed)
            time.sleep(sleep_t)
            loop_count += 1

    except KeyboardInterrupt:
        print("\n\n中斷，發送停止指令...")
        ctrl.stop()
        if arm is not None:
            arm.disconnect()
        elapsed_total = time.time() - t_loop_start
        print(f"共執行 {loop_count} 個控制週期，"
              f"平均 {1000*elapsed_total/max(loop_count,1):.1f} ms/cycle。")
        print("完成。")


if __name__ == '__main__':
    main()
