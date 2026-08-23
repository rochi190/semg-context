"""ManualSource + ArmController.disarm()/status()：手動選擇 CLI 背後的邏輯。不需要硬體。"""
from __future__ import annotations

import time

import pytest

from control.arm_config import ELBOW_FLEX_ANGLE, HOME_ANGLES
from control.arm_controller import ArmController, MockArm, Prediction, State
from control.sources import ManualSource
from semg.v2 import config as C


def make_controller() -> ArmController:
    ctrl = ArmController(MockArm())
    ctrl.startup()
    # 明確傳 now=0.0：engage() 內部的計時器會對齊到這個值，跟測試接下來用的
    # 合成時間軸（_run 系列從 t=0.0 開始）一致。
    ctrl.engage(now=0.0)
    return ctrl


def _run(ctrl, source, seconds, t0=0.0, dt=0.05):
    t = t0
    end = t0 + seconds
    while t <= end:
        for pred in source.poll(t):
            ctrl.on_prediction(pred, now=t)
        ctrl.tick(now=t)
        t += dt
    return t


def test_manual_source_emits_selected_state_at_step_sec():
    src = ManualSource()
    src._next_t = 0.0                 # 讓輸出節奏可預期，方便測試
    src.set_state(hand=C.DOF_STATES["hand"].index("pinch"), elbow=0, shoulder=0)
    preds = src.poll(now=0.2)         # 0.2s / STEP_SEC(0.05) → 應該吐出 4~5 筆
    assert len(preds) >= 4
    assert all(p.hand == C.DOF_STATES["hand"].index("pinch") for p in preds)
    assert all(p.p_hand == pytest.approx(1.0) for p in preds)


def test_manual_selection_drives_controller_elbow_flex():
    ctrl = make_controller()
    src = ManualSource()
    src._next_t = 0.0
    src.set_state(hand=0, elbow=C.DOF_STATES["elbow"].index("flex"), shoulder=0)
    # 3.0s 而非 1.0s：2026-08-19 起 J2/J3 是逐步推進到定位的，總延遲 =
    # 投票 0.30 + MIRROR_MIN_HOLD_SEC 0.50 + 90°÷MIRROR_RATE_DEG_S(60) 1.50 = 2.30s
    _run(ctrl, src, seconds=3.0)
    assert ctrl.target[2] == pytest.approx(ELBOW_FLEX_ANGLE)


def test_disarm_stops_sending_commands():
    ctrl = make_controller()
    src = ManualSource()
    src._next_t = 0.0
    ctrl.disarm()
    src.set_state(hand=0, elbow=C.DOF_STATES["elbow"].index("flex"), shoulder=0)
    _run(ctrl, src, seconds=1.0)
    # DISARMED 期間 debounce/target 計算仍在跑（純內部狀態），但 tick() 不該把
    # 指令實際送給手臂——MockArm.cmd_num 應該維持 0，last_sent_target 不變。
    assert ctrl.arm.cmd_num == 0
    assert ctrl.last_sent_target[2] == pytest.approx(HOME_ANGLES[2])


def test_disarm_from_safe_is_noop():
    ctrl = make_controller()
    ctrl.state = State.SAFE
    ctrl.disarm()
    assert ctrl.state == State.SAFE      # 只能靠 engage() 離開 SAFE，disarm() 不行


# ---------------- engage()/shutdown() 的安全行為（2026-08-19 code review）----------------

def test_engage_from_safe_recovers_arm_state(capsys):
    """離開 SAFE 必須真的把手臂拉回運動狀態，不能只是改軟體狀態——SAFE 是靠
    `arm.set_state(4)` 下的，不重新 motion_enable/set_mode/set_state(0) 的話，
    後續指令會被手臂靜默丟棄（畫面顯示 ARMED、手臂完全不動、零錯誤訊息）。
    """
    ctrl = make_controller()
    ctrl.state = State.SAFE
    ctrl.engage(now=1.0)
    out = capsys.readouterr().out
    assert "[MockArm] motion_enable(enable=True)" in out
    assert "[MockArm] set_mode(0)" in out
    assert ctrl.state == State.ARMED


def test_engage_refuses_from_safe_with_uncleared_error():
    """error_code 未清除時拒絕重新 ARM——自動忽略碰撞事件比停在 SAFE 更危險。"""
    ctrl = make_controller()
    ctrl.state = State.SAFE
    ctrl.arm.error_code = 1
    ctrl.engage(now=1.0)
    assert ctrl.state == State.SAFE      # 被拒絕，還是 SAFE


def test_disarmed_predictions_do_not_move_target_and_engage_resets_goal():
    """DISARMED 期間收到的預測不可以動到 target；engage() 也要把 goal 拉回 home。

    ⚠️ 這個測試的前提在 2026-08-19 變了，**原本的名字是
    `test_engage_resets_drifted_target_to_home_before_arming`**：
      舊行為：`on_prediction()` 直接寫 target[1]/target[2]，而它不檢查 state，
              所以 DISARMED 期間 target 會漂移到維持姿勢去；engage() 必須
              補救式地把它拉回 home，否則按 Enter 的瞬間會送出一個 90° 的
              大動作——而此刻使用者的手正放在鍵盤上。
      新行為：`on_prediction()` 只更新 **goal**，target 由 `tick()` 推進，
              而推進只在 ARMED 時發生 → **target 根本不會漂移**，那個危險
              從結構上消失了，不再依賴 engage() 的補救。
    所以這裡改成驗證更強的性質：target 全程不動 + goal 被 engage() 重設。
    """
    ctrl = make_controller()
    ctrl.disarm()

    t = 0.0
    for _ in range(20):               # DISARMED 期間餵一個「要求彎肘」的姿勢
        ctrl.on_prediction(Prediction(
            t=t, hand=0, elbow=C.DOF_STATES["elbow"].index("flex"), shoulder=0,
            p_hand=0.99, p_elbow=0.99, p_shoulder=0.99), now=t)
        ctrl.tick(now=t)              # DISARMED 下 tick 不該推進 target
        t += 0.05
    assert ctrl.target == pytest.approx(HOME_ANGLES), "DISARMED 期間 target 不該動"
    assert ctrl.mirror_goal[1] == pytest.approx(ELBOW_FLEX_ANGLE), "但 goal 有跟著更新"

    ctrl.engage(now=t)
    assert ctrl.target == pytest.approx(HOME_ANGLES)
    # goal 也要回 home，否則 ARM 後第一個 tick 就往舊姿勢推進
    assert ctrl.mirror_goal == pytest.approx([HOME_ANGLES[1], HOME_ANGLES[2]])


def test_shutdown_does_not_claim_homed_with_uncleared_error(capsys):
    """帶著 error_code 結束時，set_state(0) 會被手臂拒絕，不能無條件宣稱回了
    home——必須誠實回報「未回 home」，讓使用者知道要人工用 Studio 復歸。
    """
    ctrl = make_controller()
    ctrl.arm.error_code = 1
    ctrl.shutdown()
    out = capsys.readouterr().out
    assert "未回 home" in out
