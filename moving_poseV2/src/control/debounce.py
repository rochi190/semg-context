"""去彈跳與安全層：信心門檻 → N-of-M 多數決 → 不反應期（refractory）。

模型 20 Hz 輸出，任一單幀誤判都會讓機械手臂抖動或誤觸發（規格書 §5）。
每個自由度各自維護一個 `DofDebouncer`，各自獨立跑自己的狀態機，互不阻塞
——這是刻意的設計：對整個組合投票會讓「食指穩定彎曲但肩膀在抖」連食指
的輸出一起被否決。
"""
from __future__ import annotations

import time
from collections import deque


class DofDebouncer:
    """單一自由度的三道閘。

    `push()` 每幀呼叫一次，回傳「目前的穩定狀態」（int），還沒穩定過回傳 None。
    回傳值不是只有轉態那一刻才給，而是每次呼叫都給目前的穩定值——呼叫端只要
    讀「現在的狀態」即可，這也讓 elbow/shoulder 的位置鏡像可以每幀直接重算
    （規格書 §4.2.1：J2/J3 是無狀態的，隨時可以重算）。

    ⚠️ 沿用全專案的慣例：狀態索引 0 一律是該 DOF 的 `rest`（見
    `config.DOF_STATES`，每個 DOF 的狀態 tuple 都以 "rest" 開頭）。
    `push()` 對「轉入 0」的 refractory 處理跟其他狀態不同，見下方註解。
    """

    def __init__(self, n_states: int, p_min: float, vote_m: int, vote_n: int,
                 refractory_sec: float, window_sec: float | None = None):
        if vote_n > vote_m:
            raise ValueError(f"vote_n({vote_n}) 不能大於 vote_m({vote_m})")
        self.n_states = n_states
        self.p_min = p_min
        self.vote_m = vote_m
        self.vote_n = vote_n
        self.refractory_sec = refractory_sec
        # 投票視窗的時間上限（2026-08-19 code review 加入）。None = 不限制
        # （只給既有測試沿用舊行為，正式路徑一律由 ArmController 傳入
        # arm_config.VOTE_WINDOW_SEC）。
        self.window_sec = window_sec
        # 緩衝區存 (時間戳, 狀態)，不是只存狀態——時間戳的用途是驗證這 vote_m
        # 幀真的發生在近期：閘一會跳過低信心幀，所以「幀數」不等於「時間跨度」。
        # 疲勞或姿勢過渡期會連續產生低信心幀，若不驗證時間跨度，緩衝區裡的
        # vote_m 幀可能橫跨數秒——幾秒前的舊姿勢會在此刻湊滿多數決並觸發轉態
        # （例：使用者已放鬆，夾爪卻因為 3 秒前的舊 pinch 幀而閉合）。
        self._buf: deque[tuple[float, int]] = deque(maxlen=vote_m)
        self.stable: int | None = None
        self._last_transition_t: float = -float("inf")

    def reset(self) -> None:
        """清空所有狀態，回到「剛建立」的樣子。

        重新 ARM（DISARMED/SAFE → ARMED）時必須呼叫。否則 `_buf` 裡還留著停機
        前的幀、`stable` 還是停機前的狀態、`_last_transition_t` 也還是舊時間戳
        ——剛 ARM 的第一幀就可能因為舊票湊滿多數決而立刻觸發手臂動作或夾爪切換。

        `_last_transition_t` 設回 `-inf`（而非 0.0）代表「上次轉態在無限久以
        前」，讓重新 ARM 後的第一次轉態不被 refractory 擋住，且不依賴時鐘的
        起始值（測試會傳 `now=0.0`）。
        """
        self._buf.clear()
        self.stable = None
        self._last_transition_t = -float("inf")

    def push(self, cls: int, conf: float, now: float | None = None) -> int | None:
        now = time.monotonic() if now is None else now

        # 閘一：信心門檻。低信心幀視為「無意見」——不推進投票也不重置，
        # 只是跳過，避免中間過渡期的低信心幀被當成雜訊污染多數決。
        if conf < self.p_min:
            return self.stable

        self._buf.append((now, cls))
        if len(self._buf) < self.vote_m:
            return self.stable

        # 閘二之前的時效檢查（2026-08-19 code review 加入）。
        # 閘一跳過低信心幀，所以「vote_m 幀」不等於「vote_m × STEP_SEC 秒」。
        # 若這批幀橫跨太久，它們代表的是過去而非現在，不足以作為轉態依據。
        # 不清空緩衝區——新幀會把舊幀自然擠掉，投票能力會自行恢復；清空的話
        # 在 confidence 斷續時會永遠塞不滿 vote_m 而讓該 DOF 永久凍結。
        if self.window_sec is not None and now - self._buf[0][0] > self.window_sec:
            return self.stable

        # 閘二：N-of-M 多數決。
        # ⚠️ 平手時 max() 回傳索引最小者（通常是 rest）——這是隱含的偏好。
        #    目前 VOTE_M=6 / VOTE_N=4 不可能兩態同時 ≥4，所以平手一定被下面的
        #    票數檢查擋掉。若之後把 VOTE_N 調到 3，這個偏好就會生效，要記得。
        counts = [0] * self.n_states
        for _t, c in self._buf:
            counts[c] += 1
        winner = max(range(self.n_states), key=lambda i: counts[i])
        if counts[winner] < self.vote_n:
            return self.stable

        if winner == self.stable:
            return self.stable          # 沒有變化，不算轉態

        # 閘三：不反應期。轉態成功後，該 DOF 在 refractory_sec 內不接受新轉態。
        #
        # ⚠️ 進入 rest（狀態 0）不受這道閘限制。refractory 存在的目的是防止
        # 「單一個動作」被雜訊拆成兩次觸發（例如 pinch 上升緣被誤判成捏兩下），
        # 這個保護只在進入「動作」狀態時有意義。若同樣套用在「回到 rest」上，
        # 會製造兩個反效果：
        #   1. index/thumb 放鬆後，J6 會多轉 refractory_sec 那段時間才停
        #      （使用者回報「一次轉到底」的根因——放鬆訊號被 refractory 卡住，
        #      jog 在这段延遲裡持續累積，嚴重時直接撞到 SOFT_LIMITS）。
        #   2. 「放開再夾一次」的第二次 pinch 上升緣，如果剛好卡在前一次
        #      rest→pinch 的 refractory 視窗內，會被靜默吃掉、完全不觸發。
        # 仍然更新 `_last_transition_t`——「動作→動作」之間的防抖沒有變弱：
        # 進 rest 之後如果雜訊又跳回同一個動作，還是要等滿 refractory 才算數。
        if winner != 0 and now - self._last_transition_t < self.refractory_sec:
            return self.stable

        self.stable = winner
        self._last_transition_t = now
        return self.stable
