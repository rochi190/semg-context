"""DofDebouncer 純邏輯測試，不需要硬體（規格書 §8 驗收條件 1）。"""
from __future__ import annotations

from control.debounce import DofDebouncer


def make(p_min=0.8, vote_m=6, vote_n=4, refractory=0.5):
    return DofDebouncer(n_states=4, p_min=p_min, vote_m=vote_m, vote_n=vote_n,
                        refractory_sec=refractory)


def test_starts_unstable():
    d = make()
    assert d.stable is None


def test_single_frame_noise_does_not_trigger():
    """單幀雜訊不觸發：5 幀 rest 中混 1 幀高信心的 pinch 雜訊，多數決仍應是 rest。"""
    d = make(vote_m=6, vote_n=4)
    t = 0.0
    result = None
    for cls in (0, 0, 3, 0, 0, 0):
        result = d.push(cls, conf=0.95, now=t)
        t += 0.05
    assert result == 0


def test_low_confidence_frame_does_not_advance_vote():
    """低信心幀不推進投票：緩衝區只填了 5/6，第 6 幀信心不足時不該被計入。"""
    d = make(p_min=0.8, vote_m=6, vote_n=4)
    t = 0.0
    for _ in range(5):
        d.push(1, conf=0.95, now=t)
        t += 0.05
    result = d.push(1, conf=0.5, now=t)      # 信心不足，跳過
    assert result is None
    result = d.push(1, conf=0.95, now=t + 0.05)   # 補一幀高信心才滿足 vote_m
    assert result == 1


def test_refractory_rejects_transition():
    """refractory 期間拒絕轉態：穩定後短時間內滿足多數決的相反手勢也不該立刻生效。"""
    d = make(vote_m=4, vote_n=3, refractory=0.6)
    t = 0.0
    for _ in range(4):
        d.push(0, conf=0.95, now=t)
        t += 0.05
    assert d.stable == 0

    # 緊接著切到 pinch，四幀都高信心，但仍在 refractory 內
    for _ in range(4):
        result = d.push(3, conf=0.95, now=t)
        t += 0.05
    assert result == 0            # 拒絕轉態

    # 超過 refractory 後再給滿足多數決的幀，才允許轉態
    t = 1.0
    result = None
    for _ in range(4):
        result = d.push(3, conf=0.95, now=t)
        t += 0.05
    assert result == 3


def test_rest_entry_bypasses_refractory():
    """轉入 rest（狀態 0）不受 refractory 限制——這是「index/thumb 放鬆後 J6 該
    立刻停止 jog」的直接依據：refractory 只該保護「動作」狀態不被雜訊拆成兩次
    觸發，套用在「回到 rest」上只會讓放鬆的反應變慢（甚至讓 jog 撞到限位）。
    """
    d = make(vote_m=4, vote_n=3, refractory=0.6)
    t = 0.0
    for _ in range(4):
        d.push(1, conf=0.95, now=t)
        t += 0.05
    assert d.stable == 1

    # 緊接著切回 rest，四幀都高信心，仍在 refractory 視窗內——但應該立刻生效
    result = None
    for _ in range(4):
        result = d.push(0, conf=0.95, now=t)
        t += 0.05
    assert result == 0


def test_refractory_still_protects_reentry_after_rest_bounce():
    """回到 rest 後若立刻又跳回同一個動作，refractory 仍然要擋——保護沒有變弱，
    只是量測基準點變成最近一次轉態（這次是 rest），不是原本進入動作的那一刻。
    """
    d = make(vote_m=4, vote_n=3, refractory=0.6)
    t = 0.0
    for _ in range(4):
        d.push(1, conf=0.95, now=t)
        t += 0.05
    for _ in range(4):
        d.push(0, conf=0.95, now=t)
        t += 0.05
    assert d.stable == 0

    result = None
    for _ in range(4):
        result = d.push(1, conf=0.95, now=t)
        t += 0.05
    assert result == 0            # 拒絕轉態，還是 rest


def test_vote_n_cannot_exceed_vote_m():
    import pytest
    with pytest.raises(ValueError):
        DofDebouncer(n_states=2, p_min=0.5, vote_m=3, vote_n=4, refractory_sec=0.1)


# ---------------- window_sec：投票視窗的時間上限（2026-08-19 code review）----------------

def test_stale_frames_do_not_trigger_transition():
    """跨度過久的幀不得觸發轉態。

    4 幀 pinch 之後隔 3 秒（模擬疲勞期的低信心幀全被閘一擋掉）才來 2 幀，
    雖然湊滿 vote_m 且票數足夠，但這批幀橫跨 3 秒，不代表「現在」。
    """
    d = DofDebouncer(n_states=4, p_min=0.8, vote_m=6, vote_n=4,
                     refractory_sec=0.6, window_sec=0.5)
    t = 0.0
    for _ in range(4):
        d.push(3, conf=0.95, now=t)
        t += 0.05
    t += 3.0                              # 疲勞期：完全沒有幀進入緩衝區
    for _ in range(2):
        d.push(3, conf=0.95, now=t)
        t += 0.05
    assert d.stable is None               # 緩衝區已滿，但跨度超過 window_sec

    for _ in range(6):                    # 正常間距補滿一輪才允許轉態
        d.push(3, conf=0.95, now=t)
        t += 0.05
    assert d.stable == 3


def test_window_recovers_after_stale_period():
    """時效檢查不得造成永久凍結：舊幀被新幀擠掉後，投票能力必須恢復。"""
    d = DofDebouncer(n_states=2, p_min=0.8, vote_m=6, vote_n=4,
                     refractory_sec=0.4, window_sec=0.5)
    t = 0.0
    for _ in range(3):
        d.push(1, conf=0.95, now=t)
        t += 0.05
    t += 5.0                              # 長時間空窗
    for _ in range(6):                    # 連續 6 幀正常間距，舊幀全被擠出
        d.push(1, conf=0.95, now=t)
        t += 0.05
    assert d.stable == 1


def test_window_sec_does_not_defeat_rest_bypass():
    """window_sec 與「進入 rest 不受 refractory 限制」必須同時成立，互不干擾。

    這是 2026-08-19 合併 window_sec 機制時的回歸測試：window_sec 只影響閘二
    之前的時效檢查，rest 的 refractory 豁免是閘三的邏輯，兩者作用在不同的閘、
    不應互相削弱彼此。
    """
    d = DofDebouncer(n_states=4, p_min=0.8, vote_m=4, vote_n=3,
                     refractory_sec=0.6, window_sec=0.5)
    t = 0.0
    for _ in range(4):                    # 正常間距轉入 index，不受 window_sec 影響
        d.push(1, conf=0.95, now=t)
        t += 0.05
    assert d.stable == 1

    result = None                          # 緊接著轉回 rest，仍在 refractory 視窗內
    for _ in range(4):                    # 但幀的時間跨度在 window_sec 之內，應立刻生效
        result = d.push(0, conf=0.95, now=t)
        t += 0.05
    assert result == 0


def test_reset_clears_all_state():
    """reset() 之後行為必須與剛建立的實例相同。"""
    d = DofDebouncer(n_states=4, p_min=0.8, vote_m=6, vote_n=4,
                     refractory_sec=0.6, window_sec=0.5)
    t = 0.0
    for _ in range(6):
        d.push(3, conf=0.95, now=t)
        t += 0.05
    assert d.stable == 3

    d.reset()
    assert d.stable is None
    for _ in range(5):                    # 緩衝區也要清空：5 幀不應該有結論
        d.push(0, conf=0.95, now=t)
        t += 0.05
    assert d.stable is None
    d.push(0, conf=0.95, now=t)           # 第 6 幀才成立
    assert d.stable == 0
