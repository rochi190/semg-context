"""狀態 → 目標角映射測試，含限位 clamp（規格書 §8 驗收條件 2、3）。不需要硬體，用 MockArm。"""
from __future__ import annotations

import random
import time

import pytest

from control import arm_config as AC
from control.arm_config import (ELBOW_FLEX_ANGLE, HOME_ANGLES, JOINT_LIMITS,
                                SHOULDER_FLEX_ANGLE, SOFT_LIMITS)
from control.arm_controller import ArmController, MockArm, Prediction, clamp_angles
from semg.v2 import config as C


def make_controller() -> ArmController:
    ctrl = ArmController(MockArm())
    ctrl.startup()
    # 明確傳 now=0.0：engage() 內部的計時器（last_cmd_t 等）會對齊到這個值，
    # 跟測試接下來用的合成時間軸（_drive 系列從 t=0.0 開始）一致。
    ctrl.engage(now=0.0)
    return ctrl


def pred(hand="rest", elbow="rest", shoulder="rest", conf=0.99, t=0.0) -> Prediction:
    return Prediction(
        t=t,
        hand=C.DOF_STATES["hand"].index(hand),
        elbow=C.DOF_STATES["elbow"].index(elbow),
        shoulder=C.DOF_STATES["shoulder"].index(shoulder),
        p_hand=conf, p_elbow=conf, p_shoulder=conf,
    )


def _drive(ctrl, hand, elbow, shoulder, t, n=56, dt=0.05) -> float:
    """連續餵同一姿勢 n 幀，讓所有閘都跑完**且位置鏡像走到定位**；回傳時間游標。

    ⚠️ n 的下限是三段的總和，2026-08-19 起：
        VOTE_M(6) × STEP_SEC(0.05)        = 0.30s   投票
      + MIRROR_MIN_HOLD_SEC               = 0.50s   elbow/shoulder 的最小維持閘
      + 90° ÷ MIRROR_RATE_DEG_S(60)       = 1.50s   目標角逐步推進到定位
      = 2.30s → 至少 46 幀。取 56 幀（2.8s）留餘裕。

    這個值改過兩次，每次都是因為**多了一段延遲**而不是程式壞了：
      15 幀 → 24 幀：加入 MIRROR_MIN_HOLD_SEC
      24 幀 → 56 幀：J2/J3 改成逐步推進（不再一次跳到定位）
    之後若再加新的閘，先算總延遲再調這個數字。
    """
    for _ in range(n):
        ctrl.on_prediction(pred(hand, elbow, shoulder, t=t), now=t)
        ctrl.tick(now=t)
        t += dt
    return t


# ---------------- 位置鏡像映射 ----------------

def test_elbow_flex_maps_to_flex_angle():
    ctrl = make_controller()
    _drive(ctrl, "rest", "flex", "rest", t=0.0)
    assert ctrl.target[2] == pytest.approx(ELBOW_FLEX_ANGLE)


def test_elbow_rest_maps_to_home_angle():
    ctrl = make_controller()
    t = _drive(ctrl, "rest", "flex", "rest", t=0.0)
    _drive(ctrl, "rest", "rest", "rest", t=t + 1.0)     # 跨過 refractory 再切回 rest
    assert ctrl.target[2] == pytest.approx(HOME_ANGLES[2])


def test_shoulder_flex_maps_to_flex_angle():
    ctrl = make_controller()
    _drive(ctrl, "rest", "rest", "flex", t=0.0)
    assert ctrl.target[1] == pytest.approx(SHOULDER_FLEX_ANGLE)


@pytest.mark.skipif(AC.MIRROR_MIN_HOLD_SEC <= 0.0,
                    reason="MIRROR_MIN_HOLD_SEC=0，最小維持閘已停用（2026-08-19 還原）")
def test_short_flex_pulse_does_not_move_mirror_axes():
    """★ 短脈衝不可以驅動 J2/J3（2026-08-19 的抽動修正）。

    ⚠️ 這道閘目前是停用的（`MIRROR_MIN_HOLD_SEC = 0`），所以本測試會被 skip。
    把該值改回 0.5 重新啟用時，這個測試會自動恢復把關。

    實機症狀：模型短暫誤判成 flex 又跳回 rest，每個 0.25~0.45s 的脈衝都讓手臂
    往 90° 外衝一段再被拉回來 —— 整隻手臂抽動。P_MIN 擋不住（那些脈衝的信心
    值 0.78~1.00）、投票擋不住（實測兩層投票的反向比例與單層幾乎相同）、
    REFRACTORY 也擋不住（它對「轉回 rest」是刻意豁免的）。
    這裡驗證 MIRROR_MIN_HOLD_SEC 這道閘確實補上了缺口。
    """
    ctrl = make_controller()
    # 先讓投票穩定在 flex（0.3s），但**總時長刻意短於** MIRROR_MIN_HOLD_SEC + 投票，
    # 模擬一個撐不滿維持時間就消失的脈衝。
    n_pulse = int((AC.VOTE_M * AC.STEP_SEC + AC.MIRROR_MIN_HOLD_SEC * 0.6) / 0.05)
    t = _drive(ctrl, "rest", "flex", "flex", t=0.0, n=n_pulse)
    assert ctrl.target[2] == pytest.approx(HOME_ANGLES[2]), "J3 不該被短脈衝帶動"
    assert ctrl.target[1] == pytest.approx(HOME_ANGLES[1]), "J2 不該被短脈衝帶動"

    # 但撐得夠久的話還是要生效——這道閘是延遲，不是永久封鎖。
    _drive(ctrl, "rest", "flex", "flex", t=t)
    assert ctrl.target[2] == pytest.approx(ELBOW_FLEX_ANGLE)
    assert ctrl.target[1] == pytest.approx(SHOULDER_FLEX_ANGLE)


# ---------------- pinch 邊緣觸發 toggle（驗收條件 2）----------------

def test_pinch_edge_toggle_sequence():
    """連續 5 次 pinch 上升緣 → 夾爪狀態序列為 CLOSE, OPEN, CLOSE, OPEN, CLOSE（不多不少）。"""
    ctrl = make_controller()
    seq = []
    t = 0.0
    for _ in range(5):
        t = _drive(ctrl, "rest", "rest", "rest", t)
        t += 1.0                       # 跨過 hand 的 REFRACTORY（0.6s）
        t = _drive(ctrl, "pinch", "rest", "rest", t)
        seq.append(ctrl.grip_state)
        t += 1.0
    assert seq == ["CLOSED", "OPEN", "CLOSED", "OPEN", "CLOSED"]


def test_hand_rest_does_not_move_gripper():
    """rest 只停 J6，不動夾爪——不能讓 rest 誤觸發 toggle。"""
    ctrl = make_controller()
    _drive(ctrl, "rest", "rest", "rest", t=0.0)
    assert ctrl.grip_state == "OPEN"


def test_gripper_stays_closed_while_jogging_and_moving_other_joints():
    """夾住之後換成 index/thumb，夾爪應維持 CLOSED，且 J6 jog、elbow、shoulder 三者
    可以跟「夾住」同時成立——grip_state 是獨立於 hand 當下讀值的持續狀態，不會因為
    hand 換成別的手勢就被重置或重新觸發 toggle（使用者要求：夾住的同時還要能操控
    其他兩個關節與夾爪本身的 J6 轉動）。
    """
    ctrl = make_controller()
    t = _drive(ctrl, "rest", "rest", "rest", t=0.0)
    t = _drive(ctrl, "pinch", "rest", "rest", t=t)
    assert ctrl.grip_state == "CLOSED"

    t += 1.0                              # 跨過 hand 的 REFRACTORY，換手勢
    j6_before = ctrl.target[5]
    # 用預設的 56 幀（2.8s）而不是 30——J2/J3 現在要逐步推進 90° 才到定位
    # （投票 0.30 + 維持閘 0.50 + 斜坡 1.50 = 2.30s），30 幀只走得到一半。
    t = _drive(ctrl, "index", "flex", "flex", t=t)

    assert ctrl.grip_state == "CLOSED"                       # 夾爪沒被重置或誤 toggle
    assert ctrl.target[5] > j6_before                        # J6 持續往 + jog
    assert ctrl.target[2] == pytest.approx(ELBOW_FLEX_ANGLE)  # 手肘同時彎曲
    assert ctrl.target[1] == pytest.approx(SHOULDER_FLEX_ANGLE)  # 肩膀同時前舉


# ---------------- 限位 clamp（驗收條件 3）----------------

def test_clamp_always_within_soft_and_joint_limits():
    """任何情況下送出的角度都在 SOFT_LIMITS / JOINT_LIMITS 內（隨機序列測試）。"""
    rng = random.Random(42)
    for _ in range(500):
        angles = [rng.uniform(-720, 720) for _ in range(6)]
        clamped = clamp_angles(angles)
        for j, (lo, hi) in SOFT_LIMITS.items():
            assert lo - 1e-9 <= clamped[j - 1] <= hi + 1e-9
        for j, (lo, hi) in JOINT_LIMITS.items():
            assert lo - 1e-9 <= clamped[j - 1] <= hi + 1e-9


def test_clamp_respects_joint_limits_on_unconstrained_axes():
    clamped = clamp_angles([500.0, 0.0, 0.0, 500.0, 500.0, 0.0])
    for j in (1, 4, 5):
        lo, hi = JOINT_LIMITS[j]
        assert lo <= clamped[j - 1] <= hi


def test_controller_target_never_leaves_soft_limits_during_jog():
    """J6 持續 jog 很久也不該超出 SOFT_LIMITS[6]（累積量必須被 clamp 擋住）。"""
    ctrl = make_controller()
    t = _drive(ctrl, "index", "rest", "rest", t=0.0, n=400, dt=0.05)
    lo, hi = SOFT_LIMITS[6]
    assert lo - 1e-9 <= ctrl.target[5] <= hi + 1e-9
