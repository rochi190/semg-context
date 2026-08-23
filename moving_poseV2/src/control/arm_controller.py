"""ArmController：把分類器輸出（Prediction）轉成 xArm 關節指令。

三種控制範式共存於同一支程式，刻意不統一（規格書 §4.2）：
    elbow / shoulder → 位置鏡像（無狀態，每幀直接重算目標角）
    hand=pinch       → 邊緣觸發 toggle（只看 rest→pinch 的上升緣，切換夾爪）
    hand=index/thumb → 持續 jog（狀態維持期間，J6 每個節流 tick 累積推進）

J6 是唯一有累積狀態的軸（§4.2.1）：其餘軸的目標角都是純函數，J6 則是
「使用者過去 jog 了多久」的積分量，會漂移，需要獨立的歸零手段（`zero_j6`）。
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

from semg.v2 import config as C

from control import arm_config as AC
from control.debounce import DofDebouncer


@dataclass(frozen=True)
class Prediction:
    """上游分類器輸出契約（規格書 §2）。索引順序對應 config.DOF_STATES。"""
    t: float            # 產生時間戳。⚠️ ArmController 不使用此欄位——
                        #   on_prediction() 一律用外部傳入的 now（單一時間基準），
                        #   避免上游與控制層各有一套時鐘。sources.py 內部節奏用。
    hand: int            # 0=rest 1=index 2=thumb 3=pinch
    elbow: int            # 0=rest 1=flex
    shoulder: int          # 0=rest 1=flex
    p_hand: float         # 該 head 的 softmax 最大值 (0~1)。
                          #   ⚠️ 這是【原始】信心值，不是去彈跳後的——
                          #   log 記錄它是為了校準 P_MIN（見上機注意事項）。
    p_elbow: float
    p_shoulder: float


class State(Enum):
    DISARMED = auto()      # 預設狀態：不接受動作指令
    ARMED = auto()          # 接受動作指令
    SAFE = auto()            # 發生錯誤/看門狗逾時，需人工重新 arm 才能恢復


def clamp_angles(angles) -> list[float]:
    """送出前必須先過 SOFT_LIMITS、再過出廠 JOINT_LIMITS，兩層都要（規格書 §4.1）。"""
    out = [float(a) for a in angles]
    for j, (lo, hi) in AC.SOFT_LIMITS.items():
        out[j - 1] = min(max(out[j - 1], lo), hi)
    for j, (lo, hi) in AC.JOINT_LIMITS.items():
        out[j - 1] = min(max(out[j - 1], lo), hi)
    return out


class MockArm:
    """`--dry-run` 用的假手臂：實作與 XArmAPI 相同的介面子集，每次指令印到終端機。

    規格書 §7：「這是最重要的開發手段——手臂控制的 bug 在實機上除錯代價太高。」
    """

    def __init__(self):
        self.warn_code = 0
        self.error_code = 0
        self.mode = 0
        self._cmd_num = 0
        self._err_cb = None

    # ---- 連線 / 狀態 ----
    def clean_warn(self):
        print("[MockArm] clean_warn()")

    def clean_error(self):
        print("[MockArm] clean_error()")

    def motion_enable(self, enable=True, servo_id=None):
        print(f"[MockArm] motion_enable(enable={enable})")

    def set_mode(self, mode):
        self.mode = mode                 # 讓測試能驗證 mode 6/0 的切換順序
        print(f"[MockArm] set_mode({mode})")

    def set_state(self, state):
        print(f"[MockArm] set_state({state})")

    def set_collision_sensitivity(self, value):
        print(f"[MockArm] set_collision_sensitivity({value})")

    def register_error_warn_changed_callback(self, callback):
        self._err_cb = callback
        print("[MockArm] register_error_warn_changed_callback(...)")

    # ---- 運動 ----
    @property
    def cmd_num(self):
        return self._cmd_num

    def set_servo_angle(self, angle=None, speed=None, mvacc=None, wait=False,
                        is_radian=False, radius=None):
        ang = ", ".join(f"{a:+.1f}" for a in angle)
        print(f"[MockArm] set_servo_angle([{ang}], speed={speed}, "
              f"mvacc={mvacc}, wait={wait}, radius={radius})")
        if not wait:
            self._cmd_num = min(self._cmd_num + 1, 1)   # 模擬單一插槽立刻執行完

    # ---- 夾爪 ----
    def open_lite6_gripper(self, sync=False):
        print(f"[MockArm] open_lite6_gripper(sync={sync})")

    def close_lite6_gripper(self, sync=False):
        print(f"[MockArm] close_lite6_gripper(sync={sync})")

    def stop_lite6_gripper(self, sync=False):
        print(f"[MockArm] stop_lite6_gripper(sync={sync})")

    def disconnect(self):
        print("[MockArm] disconnect()")


class ArmController:
    """Prediction → xArm 指令。三道去彈跳閘 + 三種控制範式 + 安全層。"""

    def __init__(self, arm, log_path: Path | None = None,
                 vote_m: int | None = None, vote_n: int | None = None):
        """`vote_m`/`vote_n` 不給就用 arm_config 的 VOTE_M/VOTE_N。

        會給的只有 `live_bridge`：走 run_live.py 的路徑沿用 Recognizer 自己的
        投票層，控制層這層改成 1/1 的 pass-through，避免兩層投票疊加延遲。
        P_MIN 與 REFRACTORY 不受影響（它們不是投票，沒有重疊）。
        見 arm_config.py 的 LIVE_VOTE_M/LIVE_VOTE_N。
        """
        vote_m = AC.VOTE_M if vote_m is None else vote_m
        vote_n = AC.VOTE_N if vote_n is None else vote_n
        self.arm = arm
        self.state = State.DISARMED

        self.target: list[float] = list(AC.HOME_ANGLES)
        self.last_sent_target: list[float] = list(AC.HOME_ANGLES)
        self.last_cmd_t = 0.0
        self._last_jog_t = 0.0
        self.jog_dir = 0            # +1 index / -1 thumb / 0 停

        self.grip_state = "OPEN"
        self._grip_closed_since = 0.0
        self._grip_stopped = True
        self._prev_hand = 0          # 上一幀的穩定 hand 狀態，用來偵測 pinch 上升緣

        # 位置鏡像軸的最小維持閘（見 _mirror_hold 與 arm_config.MIRROR_MIN_HOLD_SEC）
        self._mirror_state = {"elbow": 0, "shoulder": 0}       # 目前真正生效的狀態
        self._mirror_pending: dict[str, tuple[int, float] | None] = {
            "elbow": None, "shoulder": None}                    # (候選狀態, 第一次看到的時間)
        # 位置鏡像的「要去的姿勢」[J2, J3]。target 由 tick() 逐步推進過去，
        # 不是一次跳過去——見 on_prediction 與 arm_config.MIRROR_RATE_DEG_S。
        self.mirror_goal = [AC.HOME_ANGLES[1], AC.HOME_ANGLES[2]]
        self._last_mirror_t = 0.0

        self._last_cmd_err_t = 0.0   # set_servo_angle 回傳錯誤碼的節流計時器
        self.last_prediction_t = time.monotonic()
        self.last_hand = self.last_elbow = self.last_shoulder = 0
        self.last_hand_conf = self.last_elbow_conf = self.last_shoulder_conf = 0.0

        # 索引動態從 config.DOF_STATES 取得，不硬編碼（規格書 §2）。
        self._pinch_idx = C.DOF_STATES["hand"].index("pinch")
        self._index_idx = C.DOF_STATES["hand"].index("index")
        self._thumb_idx = C.DOF_STATES["hand"].index("thumb")
        self._elbow_flex_idx = C.DOF_STATES["elbow"].index("flex")
        self._shoulder_flex_idx = C.DOF_STATES["shoulder"].index("flex")

        self._deb = {
            dof: DofDebouncer(len(C.DOF_STATES[dof]), AC.P_MIN[dof],
                              vote_m, vote_n, AC.REFRACTORY[dof],
                              AC.VOTE_WINDOW_SEC)
            for dof in ("hand", "elbow", "shoulder")
        }

        self._csv_file = None
        self._csv_writer = None
        self._last_flush_t = 0.0
        if log_path is not None:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(log_path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(AC.LOG_FIELDS)
            self._log_event(f"控制事件 log → {log_path}")

    # ------------------------------------------------------------------
    # 連線 / 初始化 / 結束（§6.3）
    # ------------------------------------------------------------------

    def _switch_mode(self, mode: int, why: str) -> None:
        """切換手臂控制模式。序列照 SDK 範例：set_mode → set_state(0) → 沉澱。

        ⚠️ 沉澱那一秒不能省——`2006-joint_online_trajectory_planning.py` 與
        `2000-joint_velocity_control.py` 兩個官方範例都在 `set_state(0)` 之後
        `time.sleep(1)`。省掉的話接下來的第一個指令可能落在模式還沒生效的
        空窗期，症狀是「第一個動作沒反應」而後續正常，極難診斷。
        """
        self.arm.set_mode(mode)
        self.arm.set_state(0)
        # MockArm 沒有實體要沉澱，跳過——否則每次 startup()/engage() 都真的睡
        # 1 秒，整個測試套件會從 7 秒變成 50 秒以上（慢測試就是不會有人跑的測試）。
        if not isinstance(self.arm, MockArm):
            time.sleep(AC.ARM_MODE_SETTLE_SEC)
        self._log_event(f"控制模式 → mode {mode}（{why}）")

    def startup(self) -> None:
        self._log_event("連線手臂中...")
        time.sleep(0.5)             # coffee.py 慣例：等 report 通道建立再讀狀態
        if self.arm.warn_code != 0:
            self.arm.clean_warn()
        if self.arm.error_code != 0:
            self.arm.clean_error()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(AC.ARM_MODE_HOME)          # 回 home 要用位置模式
        self.arm.set_state(0)
        self.arm.set_collision_sensitivity(AC.COLLISION_SENSITIVITY)
        self.arm.register_error_warn_changed_callback(self._on_error_warn)
        self.arm.set_servo_angle(angle=AC.HOME_ANGLES, speed=AC.HOME_SPEED, wait=True)
        self.arm.open_lite6_gripper(sync=False)
        self.grip_state = "OPEN"
        self.target = list(AC.HOME_ANGLES)
        self.last_sent_target = list(AC.HOME_ANGLES)
        self.mirror_goal = [AC.HOME_ANGLES[1], AC.HOME_ANGLES[2]]
        # 回完 home 才切到控制迴圈要用的模式（mode 6：新指令中斷舊指令，
        # 連續小步不會每步減速到零）。見 arm_config.ARM_MODE_RUN。
        self._switch_mode(AC.ARM_MODE_RUN, "控制迴圈")
        self._log_event(
            "已回 home，DISARMED（按 Enter 進入 ARMED）。"
            "⚠️ 維持姿勢時手臂必須懸空——手肘/前臂/上臂都不能靠在桌面、扶手或身側，"
            "否則重力被外物撐住、EMG 會消失而被判成 rest。")

    def shutdown(self) -> None:
        """安全結束：停止 → （盡量）回 home → 斷線。可被安全地重複呼叫。

        ⚠️ 若手臂帶著 error_code（碰撞、超限）結束，`set_state(0)` 會被拒絕，
        後續 `set_servo_angle` 全部無效——這種情況下手臂會停在原地而不回 home，
        而不是無條件宣稱「已回 home」（2026-08-19 code review FIX-E）。
        """
        self._log_event("結束中：停止並回 home...")
        self.arm.set_state(4)
        self.arm.stop_lite6_gripper(sync=False)

        # 手臂若帶著 error_code（碰撞/超限），set_state(0) 會被拒絕，後續
        # set_servo_angle 全部無效——手臂會停在碰撞位置而不回 home。這裡不自動
        # clean_error()（會掩蓋碰撞事件），而是明確告知使用者。
        homed = False
        if self.arm.error_code != 0:
            self._log_event(
                f"WARN 手臂 error_code={self.arm.error_code}，無法自動回 home。"
                "手臂將停在當前位置。請人工確認原因後於 UFACTORY Studio 復歸。")
        else:
            self.arm.motion_enable(enable=True)
            # 切回 mode 0 再回 home（控制迴圈跑在 mode 6，見 _switch_mode 的說明）。
            self.arm.set_mode(AC.ARM_MODE_HOME)
            code = self.arm.set_state(0)
            if code not in (0, None):     # MockArm 沒有回傳值（None），真 SDK 回 0 表成功
                self._log_event(f"WARN set_state(0) 回傳 {code}，跳過回 home。")
            else:
                if not isinstance(self.arm, MockArm):
                    time.sleep(AC.ARM_MODE_SETTLE_SEC)     # 等模式切換生效
                self.target = list(AC.HOME_ANGLES)         # J6 一併歸零
                self.arm.set_servo_angle(angle=AC.HOME_ANGLES,
                                         speed=AC.HOME_SPEED, wait=True)
                homed = True

        self.arm.disconnect()
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None          # 允許 shutdown() 被安全地重複呼叫
            self._csv_writer = None
        self._log_event("已回 home、斷線。" if homed else "已斷線（未回 home）。")

    def _on_error_warn(self, info: dict) -> None:
        """`register_error_warn_changed_callback` 的回呼。error_code != 0 立刻進 SAFE，
        不自動 clean_error 重試——自動恢復會掩蓋碰撞事件，demo 反而更危險（§6.3）。
        """
        err = info.get("error_code", 0)
        if err:
            self._enter_safe(time.monotonic(), f"手臂回報 error_code={err}")

    # ------------------------------------------------------------------
    # arm / disarm / 歸零
    # ------------------------------------------------------------------

    def engage(self, now: float | None = None) -> None:
        """DISARMED / SAFE → ARMED。按 Enter 時由 run_arm_control 呼叫。

        ⚠️ 本方法含 `wait=True` 的回 home，會阻塞數秒。呼叫端在它返回後應該
           呼叫 `source.resync(now)` 丟棄阻塞期間累積的資料（見 `sources.py`）。
        """
        # 記住呼叫端是否指定了明確時間戳——正式路徑（run_arm_control.py）一律
        # 不指定（now=None），下面 wait=True 阻塞後會重新對時；但測試常會指定
        # 合成的 now 以精確控制時間軸，這種情況必須全程尊重呼叫端給的值，
        # 不要在中途被 wait=True 之後的重新對時悄悄蓋掉，否則測試的時間基準
        # 會被真實牆上時鐘取代而失去可預測性。
        now_given = now is not None
        now = time.monotonic() if now is None else now

        # ---- 從 SAFE 恢復：必須把手臂拉回運動狀態（2026-08-19 code review FIX-A）----
        # _enter_safe() 對手臂下了 set_state(4)（STOP）。在 STOP 狀態下
        # set_servo_angle 會被【靜默丟棄】：手臂不動、不回報錯誤，而終端機
        # 顯示 state=ARMED、target 在變、cmd_sent=1，完全看不出問題在哪。
        # 正確序列與 shutdown() 一致：motion_enable → set_mode → set_state(0)。
        if self.state == State.SAFE:
            self._log_event("人工重新 ARM：離開 SAFE")
            # error_code 未清除就重新 ARM 等於忽略碰撞事件——拒絕，要求人工確認。
            # 不在這裡自動 clean_error()：自動恢復會掩蓋碰撞，demo 時反而更危險。
            if self.arm.error_code != 0:
                self._log_event(
                    f"WARN 手臂 error_code={self.arm.error_code} 尚未清除，拒絕重新 ARM。"
                    "請先確認碰撞或超限原因，手動 clean_error() 後再按 Enter。")
                return
            self.arm.motion_enable(enable=True)
            self.arm.set_mode(AC.ARM_MODE_HOME)
            self.arm.set_state(0)

        # ---- 進 ARMED 前把姿勢基準重設回 home（FIX-B）----
        # on_prediction() 的位置鏡像（target[1] / target[2]）沒有檢查 state，
        # DISARMED / SAFE 期間仍持續更新。若不重設就 ARM，按 Enter 的同一個
        # tick 就會送出一個從 home 到當前維持姿勢的大幅運動（J2/J3 各可達 90°）
        # ——而此刻使用者的手正放在鍵盤上、身體靠近手臂工作空間。
        # 用 wait=True 實體回到 home，確保軟體 target 與手臂真實位置一致。
        # ⚠️ 回 home 必須在 mode 0 做——mode 6 的語意是「新指令會中斷舊指令」，
        #    那正是我們在控制迴圈要的，但對「確實走到定點再往下」這種一次性
        #    動作不適合。做完再切回控制迴圈的模式。
        self._switch_mode(AC.ARM_MODE_HOME, "回 home")
        self.arm.set_servo_angle(angle=AC.HOME_ANGLES, speed=AC.HOME_SPEED, wait=True)
        self.target = list(AC.HOME_ANGLES)
        self.last_sent_target = list(AC.HOME_ANGLES)
        self._switch_mode(AC.ARM_MODE_RUN, "控制迴圈")

        # ---- 清空去彈跳與 jog 狀態 ----
        for deb in self._deb.values():
            deb.reset()
        self._prev_hand = 0
        self.jog_dir = 0
        # 位置鏡像閘也要歸零——上面剛實體回到 home（elbow/shoulder 都是 rest），
        # 若沿用停機前的 _mirror_state，第一幀就會以為「已經在 flex」而立刻
        # 送出一個 90° 的大動作。
        self._mirror_state = {"elbow": 0, "shoulder": 0}
        self._mirror_pending = {"elbow": None, "shoulder": None}
        # goal 也要跟著回 home——上面剛實體回到 home，若沿用停機前的 goal，
        # 第一個 tick 就會開始往舊姿勢推進（雖然是斜坡，但方向是錯的）。
        self.mirror_goal = [AC.HOME_ANGLES[1], AC.HOME_ANGLES[2]]
        self._last_mirror_t = now

        # ★ 呼叫端沒指定明確時間戳時，用「現在」重新對時：上面的 wait=True
        #   可能阻塞數秒，若沿用進入函式時的 now，下面幾個計時器一開始就落後。
        #   呼叫端若指定了 now（測試會這樣做），則尊重它、不做這次重新對時。
        if not now_given:
            now = time.monotonic()
        self._last_jog_t = now

        self.state = State.ARMED
        self.last_prediction_t = now      # 避免用舊時間戳立刻觸發看門狗
        # 原本是 last_cmd_t = 0.0（讓下一個 tick 立刻送出目前姿勢）。既然上面
        # 已經實體回到 home，就不需要立刻再送一次指令；設成 now 可避免
        # 「重設 target」與「立刻送指令」兩件事疊在一起造成突發運動。
        self.last_cmd_t = now

    def disarm(self) -> None:
        """ARMED → DISARMED。SAFE 狀態下不允許直接 disarm——只能靠 engage() 重新 arm。"""
        if self.state == State.ARMED:
            self.state = State.DISARMED
            self._log_event("已 DISARMED：不再接受新指令，手臂停在目前姿勢")

    def zero_j6(self) -> None:
        self.target[5] = 0.0
        self._log_event("J6 手動歸零 → 0°")

    # ------------------------------------------------------------------
    # 主流程：Prediction → 去彈跳 → 目標角 / 夾爪
    # ------------------------------------------------------------------

    def on_prediction(self, pred: Prediction, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self.last_prediction_t = now

        hand = self._deb["hand"].push(pred.hand, pred.p_hand, now)
        elbow = self._deb["elbow"].push(pred.elbow, pred.p_elbow, now)
        shoulder = self._deb["shoulder"].push(pred.shoulder, pred.p_shoulder, now)

        hand_i = hand if hand is not None else 0
        # ★ 位置鏡像軸要多過一道「最小維持時間」閘，擋 0.25~0.45s 的短脈衝造成
        #   的 90° 來回擺動（抽動）。hand 不過這道閘——它要的是即時反應。
        elbow_i = self._mirror_hold("elbow", elbow if elbow is not None else 0, now)
        shoulder_i = self._mirror_hold(
            "shoulder", shoulder if shoulder is not None else 0, now)

        # pinch：邊緣觸發 toggle。rest 不會讓夾爪自動打開——夾爪只由 pinch
        # 的上升緣改變（規格書 §4.2.1，寫錯的話手一放鬆東西就掉了）。
        if hand_i == self._pinch_idx and self._prev_hand != self._pinch_idx:
            if self.state == State.ARMED:
                self._toggle_gripper(now)
        self._prev_hand = hand_i

        # index/thumb：持續 jog；rest/pinch：J6 停在當前角度，不回歸。
        if hand_i == self._index_idx:
            self.jog_dir = 1
        elif hand_i == self._thumb_idx:
            self.jog_dir = -1
        else:
            self.jog_dir = 0

        # elbow/shoulder：位置鏡像，無狀態，每幀直接重算（§4.2.1）。
        # ★ 2026-08-19 起這裡只更新 **goal**（要去的姿勢），實際的 target 由
        #   tick() 每個 CMD_PERIOD 逐步推進過去——mode 6 要的是連續的小步設定點，
        #   一次丟一個 90° 外的遠端目標是 mode 0 的用法（見 arm_config 的說明）。
        self.mirror_goal[1] = (AC.ELBOW_FLEX_ANGLE if elbow_i == self._elbow_flex_idx
                               else AC.HOME_ANGLES[2])                    # J3
        self.mirror_goal[0] = (AC.SHOULDER_FLEX_ANGLE
                               if shoulder_i == self._shoulder_flex_idx
                               else AC.HOME_ANGLES[1])                     # J2

        self.last_hand, self.last_elbow, self.last_shoulder = hand_i, elbow_i, shoulder_i
        self.last_hand_conf = pred.p_hand
        self.last_elbow_conf = pred.p_elbow
        self.last_shoulder_conf = pred.p_shoulder

    def _mirror_hold(self, dof: str, cand: int, now: float) -> int:
        """位置鏡像軸（elbow/shoulder）的最小維持時間閘。

        候選狀態必須連續維持 `MIRROR_MIN_HOLD_SEC` 才允許生效；期間只要跳回
        原狀態就重新計時。回傳「目前真正該用的狀態」。

        為什麼只有這兩軸需要、既有三道閘為何都擋不住，見
        `arm_config.MIRROR_MIN_HOLD_SEC` 的完整說明（含實測數據）。
        設 0 等於關掉這道閘，行為回到 2026-08-19 之前。
        """
        cur = self._mirror_state[dof]
        if AC.MIRROR_MIN_HOLD_SEC <= 0.0:
            # 閘停用 → 直接放行。⚠️ 這裡一定要同步 `_mirror_state` 並回傳 `cand`，
            # 不能回傳 `cur`——早期版本在這條路徑上回傳 `cur` 卻從不更新它，
            # 等於把 elbow/shoulder 永久鎖在 rest，J2/J3 完全不會動。
            self._mirror_state[dof] = cand
            self._mirror_pending[dof] = None
            return cand
        if cand == cur:
            self._mirror_pending[dof] = None      # 回到現狀 → 取消候選、重新計時
            return cur
        pend = self._mirror_pending[dof]
        if pend is None or pend[0] != cand:
            self._mirror_pending[dof] = (cand, now)
            return cur
        if now - pend[1] >= AC.MIRROR_MIN_HOLD_SEC:
            self._mirror_state[dof] = cand
            self._mirror_pending[dof] = None
            return cand
        return cur

    def _toggle_gripper(self, now: float) -> None:
        if self.grip_state == "OPEN":
            self.arm.close_lite6_gripper(sync=False)
            self.grip_state = "CLOSED"
            self._grip_closed_since = now
            self._grip_stopped = False
            self._log_event(
                "pinch 上升緣 → 夾爪 CLOSED（夾爪狀態獨立於 hand 的當下讀值，"
                "接下來換成 index/thumb 手勢仍可在夾住的同時 jog J6，"
                "elbow/shoulder 本來就不受 hand 影響，三者可同時動）")
            return
        self.arm.open_lite6_gripper(sync=False)
        self.grip_state = "OPEN"
        self._log_event(f"pinch 上升緣 → 夾爪 {self.grip_state}")

    def _enter_safe(self, now: float, reason: str) -> None:
        if self.state == State.SAFE:
            return
        self.state = State.SAFE
        self.arm.set_state(4)
        self._log_event(f"ERROR 進入 SAFE：{reason}")

    # ------------------------------------------------------------------
    # 週期性 tick：J6 jog 推進 + 節流送指令 + 看門狗 + 夾爪過熱保護
    # ------------------------------------------------------------------

    def tick(self, now: float | None = None) -> dict:
        now = time.monotonic() if now is None else now

        if self.state == State.ARMED and now - self.last_prediction_t > AC.WATCHDOG_SEC:
            self._enter_safe(now, "watchdog：超過 WATCHDOG_SEC 未收到 Prediction")

        # 夾爪過熱保護：任何狀態都要顧，避免夾著不放又忘記降溫。
        if (self.grip_state == "CLOSED" and not self._grip_stopped
                and now - self._grip_closed_since >= AC.GRIP_HOLD_MAX_SEC):
            self.arm.stop_lite6_gripper(sync=False)
            self._grip_stopped = True
            self._log_event(f"WARN 夾爪持續閉合超過 {AC.GRIP_HOLD_MAX_SEC:.0f}s，已停止供電降溫")

        sent = False
        if self.state == State.ARMED:
            # J6 jog：只要方向非零，每 JOG_PERIOD 就往該方向推進 JOG_STEP_DEG。
            # ⚠️ 這裡用 JOG_PERIOD 不是 CMD_PERIOD——2026-08-18 把兩者分離，
            #    用較大的步距換取更高的等速佔比（見 arm_config.py 的完整說明）。
            if self.jog_dir != 0 and now - self._last_jog_t >= AC.JOG_PERIOD:
                self.target[5] += self.jog_dir * AC.JOG_STEP_DEG
                self._last_jog_t = now

            # J2/J3 位置鏡像：往 goal 逐步推進，不是一次跳過去。中途 goal 反轉時
            # 只是斜率反轉（平滑折返），不會變成「猛然打斷一個遠端目標」——
            # 那正是舊做法在 mode 6 下造成抽動的原因。
            #
            # ⚠️ 推進量用**實際經過時間 × 速率**，不是「每 CMD_PERIOD 走固定一步」。
            #    後者寫成 `if now - last >= CMD_PERIOD` 時對浮點誤差極敏感：
            #    tick 間隔略小於門檻就整步跳過，實測合成時間軸下變成每 0.15s
            #    才走一步（設定 0.10s），實際速率只有設定值的 2/3 而且看不出來。
            #    按時間比例推進則與 tick 時序完全無關，MIRROR_RATE_DEG_S 是準的。
            dt = now - self._last_mirror_t
            self._last_mirror_t = now
            if dt > 0.0:
                # 夾住 dt：engage() 的 wait=True 或主迴圈被阻塞時可能隔了數秒，
                # 不夾的話會一次跳過大半個行程，等於退回舊的「大跳躍」行為。
                step = AC.MIRROR_RATE_DEG_S * min(dt, AC.CMD_PERIOD * 3)
                for j, g in ((1, 0), (2, 1)):        # (target 索引, mirror_goal 索引)
                    diff = self.mirror_goal[g] - self.target[j]
                    if abs(diff) <= step:
                        self.target[j] = self.mirror_goal[g]
                    else:
                        self.target[j] += step if diff > 0 else -step

            self.target = clamp_angles(self.target)

            # 送指令節流：三個條件都要成立才送（§6.2）。
            deadzone_ok = max(abs(a - b) for a, b in
                              zip(self.target, self.last_sent_target)) >= AC.CMD_DEADZONE_DEG
            if (now - self.last_cmd_t >= AC.CMD_PERIOD and deadzone_ok
                    and self.arm.cmd_num < AC.CMDNUM_MAX):
                # ★ 2026-08-19：J6 jog 用專門調過的 speed/accel/radius，不跟
                # elbow/shoulder 位置鏡像共用 JOINT_SPEED/JOINT_ACC——原本共用
                # 低加速度時，J6 每個 4.5° 小步都還沒加速到 JOG_RATE_DEG_S
                # 就要減速（三角形速度曲線，峰值遠低於目標速率），而且沒有
                # 帶 radius 的話每個 waypoint 之間手臂會完全停下再重新啟動，
                # 兩者疊加就是使用者回報的「卡卡的、不等速」。判斷依據：
                # J2/J3（elbow/shoulder）相對上次送出的值有沒有變——沒變
                # 就代表這次純粹是 J6 在推進，用 jog 專用參數＋轉角融合
                # （見 arm_config.py 的完整根因分析）；J2/J3 真的在動（少見的
                # 一次性大幅度姿勢轉態）時維持原本保守的參數，且不做 blending
                # （一次性的大動作沒有下一步要銜接，不需要）。
                pure_jog = (abs(self.target[1] - self.last_sent_target[1]) < 1e-9
                           and abs(self.target[2] - self.last_sent_target[2]) < 1e-9)
                if pure_jog:
                    speed, acc = AC.JOG_SPEED_CAP, AC.JOG_ACC
                    # ⚠️ radius（轉角融合）只在 mode 0 有意義：它是「佇列中相鄰
                    #    waypoint 之間」的融合，而 mode 6 的新指令是【中斷】舊
                    #    指令、不排隊，沒有東西可以融合。SDK 收到 radius>=0 會
                    #    改走 move_jointb 而非 move_joint，在 mode 6 下會被拒絕
                    #    → **J6 完全不動**（2026-08-18 實機踩到）。官方的 mode 6
                    #    範例也沒有帶 radius。
                    radius = AC.JOG_BLEND_RADIUS_DEG if AC.ARM_MODE_RUN == 0 else None
                else:
                    speed, acc, radius = AC.JOINT_SPEED, AC.JOINT_ACC, None
                code = self.arm.set_servo_angle(angle=self.target, speed=speed, mvacc=acc,
                                                wait=False, is_radian=False, radius=radius)
                # ⚠️ 一定要看回傳碼。先前完全忽略它，導致指令被手臂拒絕時
                #    **完全靜默**——畫面照樣顯示 target 在變、cmd_sent=1，但手臂
                #    不動，跟「IP 錯」「還在 STOP」的症狀一模一樣，極難分辨。
                #    只在第一次與每 5 秒回報一次，避免洗版蓋掉狀態列。
                if code not in (0, None) and now - self._last_cmd_err_t >= 5.0:
                    self._last_cmd_err_t = now
                    self._log_event(
                        f"WARN set_servo_angle 回傳 {code}（mode={AC.ARM_MODE_RUN}, "
                        f"radius={radius}）——指令被手臂拒絕，手臂不會動。")
                self.last_sent_target = list(self.target)
                self.last_cmd_t = now
                sent = True

        self._log_row(now, sent)
        return self._status(sent)

    # ------------------------------------------------------------------
    # log / 顯示
    # ------------------------------------------------------------------

    def _log_event(self, message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}")

    def _log_row(self, now: float, sent: bool) -> None:
        if self._csv_writer is None:
            return
        self._csv_writer.writerow([
            f"{now:.3f}", self.last_hand, self.last_elbow, self.last_shoulder,
            f"{self.last_hand_conf:.3f}", f"{self.last_elbow_conf:.3f}",
            f"{self.last_shoulder_conf:.3f}",
            f"{self.target[1]:.2f}", f"{self.target[2]:.2f}", f"{self.target[5]:.2f}",
            self.grip_state, int(sent),
        ])
        # 即時控制迴圈裡不做每 tick 的同步磁碟 I/O（主迴圈約 100 Hz，等於每秒
        # 100 次 flush）。有送出指令時立刻落地（那是要保留的事件），否則每秒
        # flush 一次即可——意外中斷最多損失 1 秒的 log。
        if sent or now - self._last_flush_t >= 1.0:
            self._csv_file.flush()
            self._last_flush_t = now

    def status(self) -> dict:
        """給外部程式（例如手動選擇 CLI）查詢目前狀態用的公開介面。"""
        return self._status()

    def _status(self, sent: bool = False) -> dict:
        return {
            "state": self.state.name,
            "hand_name": C.DOF_STATES["hand"][self.last_hand],
            "elbow_name": C.DOF_STATES["elbow"][self.last_elbow],
            "shoulder_name": C.DOF_STATES["shoulder"][self.last_shoulder],
            "p_hand": self.last_hand_conf,
            "p_elbow": self.last_elbow_conf,
            "p_shoulder": self.last_shoulder_conf,
            "J2": self.target[1], "J3": self.target[2], "J6": self.target[5],
            "grip": self.grip_state,
            "cmdnum": self.arm.cmd_num,
            "sent": sent,
        }
