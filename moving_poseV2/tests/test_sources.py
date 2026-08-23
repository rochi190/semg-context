"""PredictionSource 介面契約測試（不需要硬體、不需要模型）。

只測 MockSource / ManualSource 與基底介面——ReplaySource / LiveSource 需要
模型或串列，留給實機/replay 驗證（見 docs/on-machine-checklist.md）。
"""
from __future__ import annotations

from control.arm_controller import Prediction
from control.sources import ManualSource, MockSource, PredictionSource
from semg.v2 import config as C


def test_base_interface_defaults():
    """基底類別的預設值必須讓子類免於強制實作 is_ready / resync。"""
    s = PredictionSource()
    assert s.is_ready() is True
    assert s.status_text() == ""
    s.resync(0.0)          # 不得拋例外
    s.close()


def test_mock_emits_at_step_sec_rhythm():
    """MockSource 必須依 STEP_SEC 節奏輸出，否則去彈跳的幀數語意失效。"""
    s = MockSource()
    t0 = s._t0
    assert s.poll(t0) != []                       # 第一幀立刻給
    assert s.poll(t0 + C.STEP_SEC * 0.4) == []    # 還沒到下一個節拍
    assert len(s.poll(t0 + C.STEP_SEC * 1.1)) == 1


def test_mock_predictions_are_valid_indices():
    """輸出的狀態索引必須落在 DOF_STATES 範圍內。"""
    s = MockSource()
    t = s._t0
    for _ in range(400):
        for p in s.poll(t):
            assert isinstance(p, Prediction)
            assert 0 <= p.hand < len(C.DOF_STATES["hand"])
            assert 0 <= p.elbow < len(C.DOF_STATES["elbow"])
            assert 0 <= p.shoulder < len(C.DOF_STATES["shoulder"])
            assert 0.0 <= p.p_hand <= 1.0
        t += C.STEP_SEC


def test_mock_resync_restarts_script():
    """resync() 之後腳本必須從頭開始，讓重新 ARM 有已知的起點。"""
    s = MockSource()
    t = s._t0
    for _ in range(100):                          # 推進到腳本中段
        s.poll(t)
        t += C.STEP_SEC
    s.resync(t)
    first = s.poll(t)
    assert len(first) == 1
    # 腳本第一段是 (2.0, "rest", "rest", "rest")
    assert first[0].hand == C.DOF_STATES["hand"].index("rest")
    assert first[0].elbow == C.DOF_STATES["elbow"].index("rest")


def test_mock_no_loop_becomes_not_ready():
    """loop=False 跑完之後 is_ready() 必須轉為 False（供 engage 前的把關用）。"""
    s = MockSource(loop=False)
    t = s._t0
    for _ in range(int(s._total / C.STEP_SEC) + 20):
        s.poll(t)
        t += C.STEP_SEC
    assert s.is_ready() is False


def test_manual_source_is_always_ready():
    """ManualSource 沒有校準/播放進度的概念，永遠可以立刻 ARM（沿用基底預設）。"""
    s = ManualSource()
    assert s.is_ready() is True
    assert s.status_text() == ""


def test_manual_source_resync_prevents_backlog_burst():
    """resync() 之後不該補發積壓期間的 Prediction——只從 resync 當下重新起算，
    避免 engage() 阻塞（wait=True 回 home）期間累積的落後量一次倒給去彈跳。
    """
    s = ManualSource()
    s._next_t = 0.0
    s.set_state(hand=0, elbow=0, shoulder=0)
    # 模擬長時間沒有呼叫 poll()（例如 engage() 正在阻塞）
    s.resync(now=100.0)
    preds = s.poll(now=100.0)
    assert len(preds) == 1                         # 沒有一次補發過去 100 秒份的幀
    assert preds[0].t == 100.0
