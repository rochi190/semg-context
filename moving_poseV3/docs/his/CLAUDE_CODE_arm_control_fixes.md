# Claude Code 修改單 —— `src/control/` 手臂控制層

> **狀態：Rev. 2（完整版，取代 Rev. 1）**
> 五個檔案的 code review 已全部完成：`arm_config.py`、`debounce.py`、
> `arm_controller.py`、`sources.py`、`run_arm_control.py`。
>
> 需要接上實機才能決定的參數與驗證項，見另一份 **「上機注意事項」**，本單不處理。

---

## 0. 給執行者的前置說明

### 0.1 適用範圍

```
src/control/arm_config.py
src/control/debounce.py
src/control/arm_controller.py
src/control/sources.py
src/control/run_arm_control.py
tests/test_debounce.py          （只新增，不改既有測試）
tests/test_sources.py           （新檔）
```

**不要動 `src/semg/v2/` 與 `src/hardware/` 底下任何既有檔案。**

### 0.2 不要做的事

- 不要「順手」重構或改變命名風格；不要移除既有註解
  （本專案註解密度是刻意的，說明「為什麼」而非「做什麼」）
- 不要改任何控制範式（位置鏡像 / 邊緣觸發 toggle / 持續 jog 三者不變）
- 不要改 `SOFT_LIMITS` / `JOINT_LIMITS` / `JOG_RATE_DEG_S` /
  `GRIP_HOLD_MAX_SEC` / `P_MIN` / `COLLISION_SENSITIVITY` 的**數值**
  （那些要接上實機才能決定）
- 不要為了讓測試通過而放寬斷言

### 0.3 修改總表與優先序

| 編號 | 檔案 | 嚴重度 | 一句話 |
|---|---|---|---|
| **FIX-A** | `arm_controller.py` | **P0** | 從 SAFE 重新 ARM 後手臂靜默失效 |
| **FIX-B** | `arm_controller.py` | **P0** | 按 Enter 瞬間可能大幅運動 |
| **FIX-K** | `sources.py` + `run_arm_control.py` | **P0** | `--source live` 必定在 5 秒校準期內誤觸發 SAFE |
| **FIX-P** | `run_arm_control.py` | **P0** | `source.close()` 拋例外會讓手臂留在 enable 狀態 |
| **FIX-Q** | `run_arm_control.py` | **P0** | source 建構失敗時手臂留在 enable 狀態 |
| **FIX-R** | `run_arm_control.py` | **P0** | IP 錯誤時症狀與 Bug A 完全相同（畫面全綠、手臂不動） |
| **FIX-C** | `debounce.py` + `arm_config.py` | P1 | 投票視窗以幀計數，數秒前的舊幀可觸發誤動作 |
| FIX-D | `debounce.py` | P1 | 缺 `reset()`（FIX-B 依賴） |
| FIX-E | `arm_controller.py` | P1 | `shutdown()` 在錯誤狀態下不會回 home |
| FIX-L | `sources.py` + `run_arm_control.py` | P1 | 突發 Prediction 破壞去彈跳的時間語意 |
| FIX-M | `sources.py` | P1 | reader thread 無例外處理；同步失敗時 busy loop 燒滿一核 |
| FIX-F | `arm_config.py` | P2 | 缺 `validate()` 自我檢查 |
| FIX-G | `arm_config.py` | P2 | `STEP_SEC` 與上游可能不同步 |
| FIX-N | `sources.py` | P2 | 掉包偵測未實作（規格書 §10） |
| FIX-S | `run_arm_control.py` | P2 | 非 Windows 平台按鍵靜默失效 |
| FIX-H | `arm_controller.py` | P3 | `Prediction.t` 未使用（僅加註解） |
| FIX-I | `arm_controller.py` | P3 | log 每 tick `flush()` |
| FIX-O | `sources.py` | P3 | replay 播完無訊息 |
| FIX-T | `run_arm_control.py` | P3 | 主迴圈是固定 sleep 而非固定週期（僅改註解） |

### 0.4 相依關係（會影響改動順序）

```
FIX-D（reset）          ────► FIX-B（engage 呼叫 reset）
FIX-C（window）         ────► arm_config 需先有 VOTE_WINDOW_SEC
FIX-B（wait=True 阻塞） ────► FIX-L（阻塞期間累積的資料必須丟棄）
FIX-K（is_ready）       ────► run_arm_control 的 Enter 處理
FIX-Q（重排順序）       ────► FIX-P（finally 的保護範圍）
```

建議順序：**F、G → C、D → A、B、E → K、L、M、N、O → Q、P、R、S → H、I、T**

### 0.5 驗收

1. `cd moving_pose && pytest tests/` 全綠
2. `python src/control/run_arm_control.py --source mock --dry-run` 能啟動、
   按 Enter 進 ARMED、按 q 正常結束
3. 完成本文件最後的「自我檢查清單」全部項目

---

# PART 1 — `arm_config.py`

## FIX-C（config 部分）＋ FIX-L（config 部分）：新增兩個常數

把去彈跳那一節改成：

```python
# ---------------- 去彈跳三道閘（§5）----------------
P_MIN = {"hand": 0.80, "elbow": 0.75, "shoulder": 0.75}
VOTE_M = 6                          # (VOTE_M - 1) × STEP_SEC = 0.25 s 名目視窗
VOTE_N = 4
REFRACTORY = {"hand": 0.60, "elbow": 0.40, "shoulder": 0.40}

# 投票視窗的【時間】上限。
#
# 為什麼需要這個：DofDebouncer 的閘一會跳過低信心幀（不進緩衝區），所以
# 「最近 VOTE_M 幀」不等於「最近 VOTE_M × STEP_SEC 秒」。疲勞或姿勢過渡期
# 會產生連續的低信心幀，若不加時間上限，緩衝區裡的 VOTE_M 幀可能橫跨數秒
# —— 幾秒前的舊姿勢會在此刻湊滿多數決並觸發轉態（例：使用者已放鬆，夾爪
# 卻因為 3 秒前的舊 pinch 幀而閉合）。
#
# 取名目視窗的 2 倍：容許最多約一半的幀被信心門檻濾掉，仍能正常轉態。
# ⚠️ 代價：若 confidence 長期低迷（超過一半的幀 < P_MIN），該 DOF 會凍結在
#    最後的穩定狀態不再轉態。這是刻意的取捨 —— 寧可不動，也不要拿數秒前的
#    資料去動手臂。實機若常凍結，正確處置是【先降 P_MIN】而非放寬本值。
VOTE_WINDOW_SEC = (VOTE_M - 1) * STEP_SEC * 2.0      # 0.50 s

# ---------------- 主迴圈突發保護（見修改單 FIX-L）----------------
# 主迴圈約 100 Hz、模型 20 Hz → 正常情況每個 tick 只會拿到 0 或 1 筆 Prediction。
# 但主迴圈一旦被阻塞（engage() 的 wait=True 回 home、GC、磁碟 I/O），來源端會
# 累積大量待處理資料，下一次 poll 一口氣吐出數十筆 —— 而它們全部會帶著同一個
# now 灌進 DofDebouncer，讓「VOTE_M 幀 = VOTE_M × STEP_SEC 秒」的假設完全失效，
# 可能瞬間湊滿多數決並觸發轉態。
# 超過此上限時只保留【最新的】幾筆（即時控制要的是現在，不是補完歷史）。
MAX_PREDS_PER_TICK = 4              # ≈ 落後 200 ms 就開始丟棄舊幀
```

---

## FIX-G：`STEP_SEC` 的一致性（保留參數分離）

### 使用者的明確決定

> 「手臂控制的參數跟我錄資料或者模型的參數定義區分開我會比較好處理，
> 或者你可以在文件中告訴 Claude Code 說要在程式中的該參數定義區註解，
> 讓使用者在調整時也能同時注意原始 config 中的定義。」

**→ 不要改成 `STEP_SEC = C.STEP_SEC`。** 控制層與訊號層的參數保持分離是使用者
要的架構。改法是：**保留本地定義 + 定義區加註解 + 在 `validate()` 裡加一致性
斷言**（斷言部分含在 FIX-F 的 `(d0)` 區塊）。

### 問題

`arm_config.STEP_SEC = 0.05` 目前是硬編碼複本，而全專案實際讀的都是
`semg.v2.config.STEP_SEC`（`sources.py`、`arm_controller.py` 皆用 `C.STEP_SEC`）
—— **`arm_config.STEP_SEC` 目前沒有任何人讀，是個孤兒常數**。

兩邊若不同步：`VOTE_M = 6  # 300 ms 視窗` 的註解會失效、`VOTE_WINDOW_SEC` 會
偏離真實時序，而程式不會給任何警告。

### 改法：`STEP_SEC` 定義區加註解

```python
# ---------------- 分類器輸出節奏 ----------------
# ★ 這是控制層自己的參數副本，刻意與 semg/v2/config.py 分離
#   （控制層參數與訊號/模型參數分開管理，見專案慣例）。
#
# ⚠️ 但它的【值】必須與 semg/v2/config.py 的 STEP_SEC 相同，因為控制層所有
#    以「幀」為單位的參數都靠它換算成秒：
#        VOTE_M = 6           → 6 幀 = 6 × STEP_SEC 秒
#        VOTE_WINDOW_SEC      → 由 STEP_SEC 推導
#        REFRACTORY / WATCHDOG_SEC 是秒，要跟幀數對得上也靠它
#        MAX_PREDS_PER_TICK   → 由「主迴圈頻率 / 模型頻率」推導
#
#    ★ 若你調整了 semg/v2/config.py 的 STEP_SEC（模型輸出頻率），
#      這裡必須一起改。檔尾 validate() 會在兩者不一致時 assert 失敗，
#      不會讓你靜默跑在錯誤的時序假設上。
STEP_SEC = 0.05                   # 20 Hz。必須等於 semg.v2.config.STEP_SEC
```

---

## FIX-F：新增 `validate()` 載入時自我檢查

### 問題

`arm_config.py` 裡的常數彼此有隱含相依，但沒有任何檢查機制。這些矛盾的共同
特徵是**症狀與原因看起來毫無關聯**，在實機上除錯代價極高。三個具體風險：

**(a) `HOME_ANGLES` 目前剛好踩在 `SOFT_LIMITS` 邊界上**

```python
HOME_ANGLES = [0.0, -90.0, 180.0, 0.0, 0.0, 0.0]
SOFT_LIMITS = {2: (-90.0, 0.0), 3: (90.0, 180.0)}
#                  ↑ J2 home 等於下限     ↑ J3 home 等於上限
```

只要有人動一下 `HOME_ANGLES[1]`（例如改成 −95），`clamp_angles()` 會**靜默**
把它夾回 −90，使用者只會覺得「改了沒生效」而找不到原因。

**(b) `JOG_STEP_DEG` 必須大於 `CMD_DEADZONE_DEG`（這條最值得防）**

```python
JOG_STEP_DEG = JOG_RATE_DEG_S * CMD_PERIOD    # 45.0 × 0.10 = 4.5°
CMD_DEADZONE_DEG = 0.5
```

實機調 jog 手感時若把 `JOG_RATE_DEG_S` 降到 4.0（很合理的直覺調整）：

```
JOG_STEP_DEG = 4.0 × 0.10 = 0.4° < CMD_DEADZONE_DEG = 0.5°
→ 每個 tick 的變化量都被死區擋掉 → 指令一次都送不出去 → J6 完全不動
```

而終端機上會看到 `J6=+0.4 → +0.8 → +1.2` 一直在跳（`target[5]` 確實在累加），
只有 `cmd_sent` 恆為 0。**「target 在動、手臂不動」幾乎不可能聯想到是死區。**

**(c) `SOFT_LIMITS` 是否確實落在 `JOINT_LIMITS` 內**，目前沒有驗證。

### 改法：`arm_config.py` 檔尾追加

```python
# ---------------- 載入時自我檢查 ----------------
# 比照 semg/v2/config.py 的 validate() 慣例：把參數之間的矛盾擋在實機之前。
# 這些矛盾的共同特徵是「症狀與原因看起來毫無關聯」，實機除錯代價極高。

def validate() -> None:
    # (a) 軟限位必須落在出廠限位內，否則 clamp_angles() 的第二道防線形同虛設
    for j, (lo, hi) in SOFT_LIMITS.items():
        jlo, jhi = JOINT_LIMITS[j]
        assert lo < hi, f"SOFT_LIMITS[J{j}] 下限 {lo} 不小於上限 {hi}"
        assert jlo <= lo and hi <= jhi, (
            f"SOFT_LIMITS[J{j}]=({lo},{hi}) 超出 JOINT_LIMITS=({jlo},{jhi})")

    # (b) home 姿態必須在軟限位內，否則 clamp 會靜默改掉你設的 home
    for j, (lo, hi) in SOFT_LIMITS.items():
        h = HOME_ANGLES[j - 1]
        assert lo <= h <= hi, (
            f"HOME_ANGLES[J{j}]={h} 不在 SOFT_LIMITS=({lo},{hi}) 內，"
            "clamp_angles() 會把它夾走，你會以為「改了沒生效」")

    # (c) 位置鏡像的目標角必須在軟限位內
    assert SOFT_LIMITS[3][0] <= ELBOW_FLEX_ANGLE <= SOFT_LIMITS[3][1], (
        f"ELBOW_FLEX_ANGLE={ELBOW_FLEX_ANGLE} 不在 J3 軟限位 {SOFT_LIMITS[3]} 內")
    assert SOFT_LIMITS[2][0] <= SHOULDER_FLEX_ANGLE <= SOFT_LIMITS[2][1], (
        f"SHOULDER_FLEX_ANGLE={SHOULDER_FLEX_ANGLE} 不在 J2 軟限位 {SOFT_LIMITS[2]} 內")

    # (d0) 與上游訊號層的時序一致性（見修改單 FIX-G）。
    # 控制層刻意保留自己的 STEP_SEC 副本（參數分離），但值必須相同 ——
    # 所有以「幀」為單位的去彈跳參數都靠它換算成秒。
    # 在函式內 import 以避免模組層級的循環引用風險。
    from semg.v2 import config as _C
    assert abs(STEP_SEC - _C.STEP_SEC) < 1e-9, (
        f"arm_config.STEP_SEC={STEP_SEC} 與 semg.v2.config.STEP_SEC={_C.STEP_SEC} "
        f"不一致。控制層的 VOTE_M / VOTE_WINDOW_SEC / REFRACTORY / WATCHDOG_SEC "
        f"全部建立在這個值上，不一致會讓去彈跳的時間語意錯誤。請改為相同值。")

    # (d) 去彈跳參數自洽
    assert 0 < VOTE_N <= VOTE_M, f"VOTE_N={VOTE_N} 必須落在 1..VOTE_M={VOTE_M}"
    assert VOTE_WINDOW_SEC >= (VOTE_M - 1) * STEP_SEC, (
        f"VOTE_WINDOW_SEC={VOTE_WINDOW_SEC} 小於名目視窗 "
        f"{(VOTE_M - 1) * STEP_SEC}，連正常速率的幀都會被時效檢查擋掉")
    for dof, p in P_MIN.items():
        assert 0.0 < p < 1.0, f"P_MIN[{dof}]={p} 必須在 (0,1) 之間"

    # (e) ★ jog 步階必須大於死區，否則 J6 永遠不會動且症狀極具誤導性
    assert JOG_STEP_DEG > CMD_DEADZONE_DEG, (
        f"JOG_STEP_DEG={JOG_STEP_DEG:.2f}° ≤ CMD_DEADZONE_DEG={CMD_DEADZONE_DEG}°："
        f"每個 tick 的 J6 變化量都會被死區擋掉，J6 完全不會動，"
        f"而終端機上 target 仍在累加（cmd_sent 恆為 0）。"
        f"請提高 JOG_RATE_DEG_S（目前 {JOG_RATE_DEG_S}，"
        f"下限約 {CMD_DEADZONE_DEG / CMD_PERIOD:.1f} °/s）或降低 CMD_DEADZONE_DEG。")

    # (f) 節流與看門狗合理性
    assert CMD_PERIOD > 0 and CMDNUM_MAX >= 1
    assert WATCHDOG_SEC > STEP_SEC * 2, (
        f"WATCHDOG_SEC={WATCHDOG_SEC} 太接近單幀週期，正常抖動就會誤觸發 SAFE")
    assert MAX_PREDS_PER_TICK >= 1


validate()
```

> 用 `assert` 而非 `raise` 是刻意的：這些是開發期的設定錯誤，不是執行期例外。
> 訊息寫長一點沒關係 —— 它是使用者在 traceback 裡唯一會看到的線索。

---

# PART 2 — `debounce.py`

## FIX-C（主體）：投票視窗必須以時間為準

### 問題與因果鏈

`VOTE_M = 6` 的註解寫「300 ms 視窗」，但實作是 `deque(maxlen=6)`，只保證
「最近 6 **筆**」，不保證「最近 300 **毫秒**」。這兩件事只有在「每一幀都通過
信心門檻」時才等價，而閘一（`conf < p_min` 就 `return`，不進緩衝區）的存在
意義就是「會有幀被跳過」。

```
使用者維持 pinch，但肌肉疲勞導致 confidence 落在 0.7 左右（低於 P_MIN=0.80）

t=0.00~0.15s   4 幀 pinch，conf 0.9+   → 進 buffer，_buf = [3,3,3,3]
t=0.20~3.15s   60 幀，conf 全部 < 0.80 → 全被閘一擋掉，_buf 不變
               （這 3 秒內使用者早已放鬆，手臂應該停止）
t=3.20s        1 幀 pinch，conf 0.90   → _buf = [3,3,3,3,3]
t=3.25s        1 幀 pinch，conf 0.91   → _buf = [3,3,3,3,3,3]
               → 計票 pinch=6 ≥ VOTE_N=4 → 轉態成 pinch → 夾爪動作
```

`_buf` 裡有 4 幀是 **3.2 秒前**的資料。系統拿過去的票 + 現在的 2 票判定
「現在是 pinch」，於是在使用者已放鬆的情況下切換夾爪。

`P_MIN` 設得越高、被跳過的幀越多，問題越嚴重。目前 `hand` 是 0.80，偏高。

### 為什麼不用「超時就清空 buffer」

清空的話，在 confidence 斷斷續續（一半通過一半不通過）的情境下，`_buf` 會
反覆被清空、永遠塞不滿 `vote_m`，該 DOF 就**永久凍結**在最後的穩定狀態。

正確做法是「**檢查跨度、但不清空**」：舊幀會隨新幀進來被 `deque` 自然擠掉，
投票能力會自行恢復。

## FIX-D：新增 `reset()`

重新 ARM 時 debouncer 的 `_buf` / `stable` / `_last_transition_t` 都還是停機前
的值，可能剛 ARM 就因舊幀湊滿多數決而立刻觸發動作。

## 改法：`debounce.py` 完整替換

```python
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
        # 投票視窗的時間上限。None = 不限制（僅供既有測試沿用舊行為，
        # 正式路徑一律由 arm_controller 傳入 AC.VOTE_WINDOW_SEC）。
        self.window_sec = window_sec
        # 緩衝區存 (時間戳, 狀態)。時間戳的用途是驗證這 vote_m 幀真的發生在
        # 近期 —— 閘一會跳過低信心幀，所以「幀數」不等於「時間跨度」。
        self._buf: deque[tuple[float, int]] = deque(maxlen=vote_m)
        self.stable: int | None = None
        self._last_transition_t: float = -float("inf")

    def reset(self) -> None:
        """清空所有狀態，回到「剛建立」的樣子。

        重新 ARM（DISARMED/SAFE → ARMED）時必須呼叫。否則 _buf 裡還留著停機前
        的幀、stable 還是停機前的狀態、_last_transition_t 也還是舊時間戳 ——
        剛 ARM 的第一幀就可能因為舊票湊滿多數決而立刻觸發手臂動作或夾爪切換。

        _last_transition_t 設回 -inf（而非 0.0）代表「上次轉態在無限久以前」，
        讓重新 ARM 後的第一次轉態不被 refractory 擋住，且不依賴時鐘的起始值
        （測試會傳 now=0.0）。
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

        # 閘二之前的時效檢查（見修改單 FIX-C）。
        # 閘一跳過低信心幀，所以「vote_m 幀」不等於「vote_m × STEP_SEC 秒」。
        # 若這批幀橫跨太久，它們代表的是過去而非現在，不足以作為轉態依據。
        # 不清空緩衝區 —— 新幀會把舊幀自然擠掉，投票能力會自行恢復；清空的話
        # 在 confidence 斷續時會永遠塞不滿 vote_m 而讓該 DOF 永久凍結。
        if self.window_sec is not None and now - self._buf[0][0] > self.window_sec:
            return self.stable

        # 閘二：N-of-M 多數決。
        # ⚠️ 平手時 max() 回傳索引最小者（通常是 rest）—— 這是隱含的偏好。
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
        if now - self._last_transition_t < self.refractory_sec:
            return self.stable

        self.stable = winner
        self._last_transition_t = now
        return self.stable
```

## 改法：`tests/test_debounce.py` 新增三個測試

**不要修改既有的 5 個測試**（它們用 `window_sec=None` 的舊行為，仍應通過）。
在檔尾追加：

```python
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
    d.push(0, conf=0.95, now=t)           # 第 6 幀才成立，且不被 refractory 擋
    assert d.stable == 0
```

---

# PART 3 — `arm_controller.py`

## FIX-A：從 SAFE 重新 ARM 之後，手臂靜默失效（P0）

### 因果鏈

```
watchdog 逾時或手臂回報 error
  → _enter_safe() 執行 self.arm.set_state(4)    ← 手臂進入 STOP
  → 使用者排除問題後按 Enter
  → engage() 只做 self.state = State.ARMED      ← 完全沒有碰手臂
  → tick() 判定 ARMED，節流三條件通過，送出 set_servo_angle(...)
  → 手臂仍在 STOP，指令被【靜默丟棄】
```

**症狀**：終端機顯示 `[ARMED]`、分類正常、`J2/J3` 目標角在變、`cmd_sent=1`，
但手臂完全不動，且無任何錯誤訊息。除錯時會誤往網路 / 手臂硬體 / 分類器找。

**證據**：同檔案的 `shutdown()` 已經知道正確序列
（`set_state(4)` → `motion_enable(True)` → `set_state(0)`），`engage()` 漏掉。

## FIX-B：按 Enter 的瞬間可能觸發大幅運動（P0）

### 因果鏈

```
startup() 回 home，target = [0, -90, 180, 0, 0, 0]
  → 【DISARMED】使用者在貼電極 / 測試姿勢，source 持續吐 Prediction
  → on_prediction() 的位置鏡像兩行沒有檢查 state：
        self.target[2] = 90     (elbow flex)
        self.target[1] = 0      (shoulder flex)
    → target 悄悄變成 [0, 0, 90, 0, 0, 0]
  → 使用者按 Enter
  → engage() 設 last_cmd_t = 0.0  → 節流第一條件 (now - 0.0) 立刻成立
  → 下一個 tick（10 ms 後）三條件全過
  → set_servo_angle([0, 0, 90, ...], speed=40)
  → J2 走 90°、J3 走 90°，約 2.3 秒的大範圍運動
```

**危險點**：使用者按 Enter 的那一刻手在鍵盤上、身體靠近手臂工作空間。

**證據**：夾爪的 toggle 有 `if self.state == State.ARMED` 保護，J2/J3 沒有。
這是實作不一致，不是設計意圖。

## 改法：完整替換 `engage()`

```python
    def engage(self, now: float | None = None) -> None:
        """DISARMED / SAFE → ARMED。按 Enter 時由 run_arm_control 呼叫。

        ⚠️ 本方法含 wait=True 的回 home，會阻塞數秒。呼叫端必須在返回後
           呼叫 source.resync(now) 丟棄阻塞期間累積的資料（見修改單 FIX-L）。
        """
        now = time.monotonic() if now is None else now

        # ---- 從 SAFE 恢復：必須把手臂拉回運動狀態 ----
        # _enter_safe() 對手臂下了 set_state(4)（STOP）。在 STOP 狀態下
        # set_servo_angle 會被【靜默丟棄】：手臂不動、不回報錯誤，而終端機
        # 顯示 state=ARMED、target 在變、cmd_sent=1，完全看不出問題在哪。
        # 正確序列與 shutdown() 一致：motion_enable → set_mode → set_state(0)。
        if self.state == State.SAFE:
            self._log_event("人工重新 ARM：離開 SAFE")
            # error_code 未清除就重新 ARM 等於忽略碰撞事件 —— 拒絕，要求人工確認。
            # 不在這裡自動 clean_error()：自動恢復會掩蓋碰撞，demo 時反而更危險。
            if self.arm.error_code != 0:
                self._log_event(
                    f"WARN 手臂 error_code={self.arm.error_code} 尚未清除，拒絕重新 ARM。"
                    "請先確認碰撞或超限原因，手動 clean_error() 後再按 Enter。")
                return
            self.arm.motion_enable(enable=True)
            self.arm.set_mode(0)
            self.arm.set_state(0)

        # ---- 進 ARMED 前把姿勢基準重設回 home ----
        # on_prediction() 的位置鏡像（target[1] / target[2]）沒有檢查 state，
        # DISARMED / SAFE 期間仍持續更新。若不重設就 ARM，按 Enter 的同一個
        # tick 就會送出一個從 home 到當前維持姿勢的大幅運動（J2/J3 各可達 90°）
        # —— 而此刻使用者的手正放在鍵盤上、身體靠近手臂工作空間。
        # 用 wait=True 實體回到 home，確保軟體 target 與手臂真實位置一致。
        self.arm.set_servo_angle(angle=AC.HOME_ANGLES, speed=AC.HOME_SPEED, wait=True)
        self.target = list(AC.HOME_ANGLES)
        self.last_sent_target = list(AC.HOME_ANGLES)

        # ---- 清空去彈跳與 jog 狀態 ----
        for deb in self._deb.values():
            deb.reset()
        self._prev_hand = 0
        self.jog_dir = 0

        # ★ 用「現在」重新對時：上面的 wait=True 可能阻塞數秒，
        #   若沿用進入函式時的 now，下面幾個計時器一開始就落後。
        now = time.monotonic()
        self._last_jog_t = now

        self.state = State.ARMED
        self.last_prediction_t = now      # 避免用舊時間戳立刻觸發看門狗
        # ★ 原本是 last_cmd_t = 0.0（讓下一個 tick 立刻送出目前姿勢）。
        #   既然上面已經實體回到 home，就不需要立刻送指令；設成 now 可避免
        #   「重設 target」與「立刻送指令」兩件事疊在一起造成突發運動。
        self.last_cmd_t = now
```

### ⚠️ `list(...)` 一定要保留

```python
self.target = list(AC.HOME_ANGLES)          # ✅ 產生新的 list
self.target = AC.HOME_ANGLES                # ❌ 兩者指向同一個物件
```

寫錯的話 `self.target[5] += 4.5` 會直接改到 `arm_config.HOME_ANGLES` 這個全域
常數，之後所有「回 home」都會回到錯的位置，而且誤差持續累積。

## 改法：`__init__` 建構 debouncer 時傳入 `VOTE_WINDOW_SEC`

```python
        self._deb = {
            "hand": DofDebouncer(len(C.DOF_STATES["hand"]), AC.P_MIN["hand"],
                                 AC.VOTE_M, AC.VOTE_N, AC.REFRACTORY["hand"],
                                 AC.VOTE_WINDOW_SEC),
            "elbow": DofDebouncer(len(C.DOF_STATES["elbow"]), AC.P_MIN["elbow"],
                                  AC.VOTE_M, AC.VOTE_N, AC.REFRACTORY["elbow"],
                                  AC.VOTE_WINDOW_SEC),
            "shoulder": DofDebouncer(len(C.DOF_STATES["shoulder"]), AC.P_MIN["shoulder"],
                                     AC.VOTE_M, AC.VOTE_N, AC.REFRACTORY["shoulder"],
                                     AC.VOTE_WINDOW_SEC),
        }
```

---

## FIX-E：`shutdown()` 在錯誤狀態下不會回 home（P1）

### 問題

若手臂帶著 `error_code`（碰撞、超限）而使用者按 `q` 結束，`set_state(0)` 會被
拒絕，後續 `set_servo_angle` 也無效 —— **手臂停在碰撞位置，不會回 home**。
規格書 §8 驗收條件 5「Ctrl-C 一定回 home 並 disconnect」此情境下不成立。
回傳碼被完全忽略，log 裡也看不出發生了什麼。

### 改法：完整替換 `shutdown()`

```python
    def shutdown(self) -> None:
        self._log_event("結束中：停止並回 home...")
        self.arm.set_state(4)
        self.arm.stop_lite6_gripper(sync=False)

        # 手臂若帶著 error_code（碰撞 / 超限），set_state(0) 會被拒絕，
        # 後續 set_servo_angle 全部無效 —— 手臂會停在碰撞位置而不回 home。
        # 這裡不自動 clean_error()（會掩蓋碰撞事件），而是明確告知使用者。
        homed = False
        if self.arm.error_code != 0:
            self._log_event(
                f"WARN 手臂 error_code={self.arm.error_code}，無法自動回 home。"
                "手臂將停在當前位置。請人工確認原因後於 UFACTORY Studio 復歸。")
        else:
            self.arm.motion_enable(enable=True)
            code = self.arm.set_state(0)
            if code not in (0, None):     # MockArm 回 None，真 SDK 回 0 表成功
                self._log_event(f"WARN set_state(0) 回傳 {code}，跳過回 home。")
            else:
                self.target = list(AC.HOME_ANGLES)         # J6 一併歸零
                self.arm.set_servo_angle(angle=AC.HOME_ANGLES,
                                         speed=AC.HOME_SPEED, wait=True)
                homed = True

        self.arm.disconnect()
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None          # 允許 shutdown 被安全地重複呼叫
            self._csv_writer = None
        self._log_event("已回 home、斷線。" if homed else "已斷線（未回 home）。")
```

> `code not in (0, None)`：`MockArm.set_state()` 目前沒有回傳值（等於 `None`），
> 真 SDK 回 `0` 表示成功。兩者都放行，避免 dry-run 誤判成失敗。
> **不要**為此修改 `MockArm.set_state` 的簽名。
>
> `self._csv_file = None` 是為了讓 `shutdown()` 可以被安全地重複呼叫
> （FIX-P 的 `finally` 區塊會確保它一定被呼叫，可能與其他路徑重疊）。

---

## FIX-H：`Prediction.t` 未被使用（只加註解，不刪欄位）

`ArmController.on_prediction(pred, now)` 用的是外部傳入的 `now`，不是 `pred.t`。
`sources.py` 的 `ReplaySource._to_pred()` 甚至把它填成 `0.0`。

**不要刪除這個欄位**（`sources.py` 用它做內部節奏控制）。只改註解：

```python
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
                          #   ⚠️ 這是【原始】信心值，不是去彈跳後的 ——
                          #   log 記錄它是為了校準 P_MIN（見上機注意事項）。
    p_elbow: float
    p_shoulder: float
```

---

## FIX-I：log 每 tick `flush()`（可選，低優先）

主迴圈約 100 Hz，代表每秒 100 次 `flush()` —— 在即時控制迴圈裡做同步磁碟 I/O。
SSD 上影響不大，但原則上不好。

`__init__` 加一個計時器：

```python
        self._last_flush_t = 0.0
```

`_log_row` 尾端的 `self._csv_file.flush()` 改成：

```python
        # 即時控制迴圈裡不做每 tick 的同步磁碟 I/O。有送出指令時立刻落地
        # （那是要保留的事件），否則每秒 flush 一次即可 —— 意外中斷最多
        # 損失 1 秒的 log。
        if sent or now - self._last_flush_t >= 1.0:
            self._csv_file.flush()
            self._last_flush_t = now
```

> 若覺得風險大於收益，**可以跳過 FIX-I**，不影響正確性。

---

# PART 4 — `sources.py`

## FIX-K：`--source live` 必定在校準期內誤觸發 SAFE（P0）

### 因果鏈

`LiveSource.__init__` 只啟動 reader thread 就返回，**校準是在前 5 秒的 `poll()`
裡做的**：

```python
def poll(self, now=None):
    ...
    if not self._calibrated:
        self._calib_chunks.append(blk)
        if now < self._calib_deadline:
            return []                     # ★ 校準期間回傳空 list
        ...
        self._calibrated = True
        return []
```

所以 `--source live` 啟動後**前 5 秒完全沒有 Prediction**。而 `WATCHDOG_SEC = 0.5`：

```
使用者用 --yes（或啟動後立刻按 Enter）
  → engage() 被呼叫，state = ARMED，last_prediction_t = now
  → 0.5 秒後 tick() 判定「超過 WATCHDOG_SEC 未收到 Prediction」
  → _enter_safe() → arm.set_state(4) → 進 SAFE
  → 再按 Enter 也只是重新開始，還在校準期 → 再度進 SAFE
  → 使用者被困住，且以為是串列或模型壞了
```

**這是必然發生，不是機率問題。** `--yes --source live` 100% 觸發。

### 改法：`PredictionSource` 基底類別新增三個方法

（`resync` 屬 FIX-L，`status_text` 屬 FIX-O，一併加。）

```python
class PredictionSource:
    """介面：`poll()` 回傳這次呼叫以來新產生的 Prediction 列表（可能是空 list）。"""

    def poll(self, now: float | None = None) -> list[Prediction]:
        raise NotImplementedError

    def is_ready(self) -> bool:
        """來源是否已經可以穩定產出 Prediction。

        為什麼需要這個（見修改單 FIX-K）：LiveSource 的前 calib_sec 秒在做校準，
        poll() 一律回傳空 list。若在這段期間 engage()，WATCHDOG_SEC=0.5 s 會
        【必然】觸發並把手臂推進 SAFE，而使用者會誤以為是串列或模型壞了。
        呼叫端必須在 engage() 之前確認本方法回傳 True。

        預設 True —— mock / replay 在建構完成後就能立刻產出。
        """
        return True

    def status_text(self) -> str:
        """給終端機顯示的一行來源狀態（例如校準進度）。預設空字串。"""
        return ""

    def resync(self, now: float | None = None) -> None:
        """把來源的時間基準對齊到 now，並丟棄已累積的待處理資料。

        見修改單 FIX-L：engage() 內含 wait=True 的回 home，會阻塞數秒；
        期間 LiveSource 的 reader thread 持續累積樣本、ReplaySource 的播放
        時鐘持續落後。若不對時，engage 之後的第一次 poll 會一次吐出數十筆
        Prediction，全部帶著同一個 now 灌進 DofDebouncer，讓
        「VOTE_M 幀 = VOTE_M × STEP_SEC 秒」的假設完全失效，可能瞬間湊滿
        多數決並觸發轉態。
        """
        pass

    def close(self) -> None:
        pass
```

`MockSource` 新增：

```python
    def is_ready(self) -> bool:
        return not self._done

    def resync(self, now: float | None = None) -> None:
        # 腳本從頭重播 —— 重新 ARM 時從已知的起點開始，比從中間接續好驗證。
        now = time.monotonic() if now is None else now
        self._t0 = now
        self._next_t = now
        self._done = False
```

`ReplaySource` 新增（`__init__` 需加 `self._announced_end = False`）：

```python
    def is_ready(self) -> bool:
        return bool(self._preds)          # 播完就不再 ready

    def status_text(self) -> str:
        return "" if self._preds else "重播已結束"

    def resync(self, now: float | None = None) -> None:
        # 不丟棄內容（重播的資料是有限且珍貴的），只把播放時鐘對齊到現在，
        # 避免 engage() 阻塞期間累積的落後量在下一次 poll 被一次倒出。
        self._t_next = time.monotonic() if now is None else now
```

`LiveSource` 新增：

```python
    def is_ready(self) -> bool:
        return self._calibrated

    def status_text(self) -> str:
        if self._reader_error is not None:
            return f"串列已中斷：{self._reader_error}"
        if not self._calibrated:
            remain = max(0.0, self._calib_deadline - time.monotonic())
            return f"校準中 還需 {remain:.1f}s（請保持中性放鬆、手臂懸空）"
        return ""

    def resync(self, now: float | None = None) -> None:
        # 丟棄 engage() 阻塞期間累積的原始樣本。即時控制要的是「現在」，
        # 補完數秒前的歷史只會讓去彈跳拿到錯誤的時間語意。
        n = len(self._q)
        self._q.clear()
        if n:
            print(f"\n[live] resync：丟棄 {n} 個累積樣本（{n / C.FS:.2f}s）")
```

---

## FIX-L：突發 Prediction 破壞去彈跳的時間語意（P1）

### 因果鏈

主迴圈約 100 Hz、模型 20 Hz → 正常每 tick 拿到 0 或 1 筆。但主迴圈一旦被阻塞：

| 阻塞來源 | 時長 |
|---|---|
| `engage()` 的 `wait=True` 回 home（FIX-B 新增） | 1–3 s |
| `startup()` 的 `wait=True` 回 home | 1–3 s |
| GC、磁碟 flush、Windows 排程 | 數十 ms |

阻塞期間 `LiveSource` 的 reader thread 持續填 `self._q`（2000 SPS）。恢復後
第一次 `poll()`：

```python
n = len(self._q)                          # 例如 6000（3 秒）
blk = np.array([...])                     # 一次餵 6000 樣本進 Recognizer
for res in self.rec.push(blk):            # 產出約 60 筆 Prediction
```

而主迴圈是這樣消化的：

```python
now = time.monotonic()
for pred in source.poll(now):
    controller.on_prediction(pred, now)   # ★ 60 筆全部用【同一個 now】
```

於是 60 幀以相同時間戳灌進 `DofDebouncer`：

- FIX-C 的時效檢查 `now - _buf[0][0] = 0` → 一定通過（**防不到這一種**）
- 6 幀同類即湊滿多數決 → **立刻轉態**
- `refractory` 剛被 `reset()` 設成 `-inf` → 不擋
- **結果：engage 之後可能立刻觸發夾爪切換或關節運動**

`ReplaySource` 同理：`while self._preds and now >= self._t_next` 會把落後量一次倒出。

### 改法 1：`resync()`（已含在 FIX-K 的程式碼中）

### 改法 2：主迴圈加突發上限（見 PART 5 的完整 `main()`）

---

## FIX-M：reader thread 無例外處理 + busy loop 燒滿一核（P1）

### 問題

```python
def _reader():
    while not self._stop:
        if not find_sync(self.ser):
            continue                       # ← ⚠️ 無 sleep 的 busy loop
        r = read_one_frame(self.ser)
        if r is None:
            continue
        _status, ch = r
        self._q.append([ch[c - 1] for c in C.CH_CSV_COLS])
```

**問題 1：完全沒有 try/except。** USB 被拔掉、STM32 重啟、驅動異常 →
`serial.SerialException` → thread 靜默死亡。`self._q` 停止成長 → watchdog 觸發
→ SAFE。行為上是 fail-safe，但**沒有任何訊息說明串列埠掛了**，使用者會往
分類器方向找。

**問題 2：`find_sync` 失敗時是無 sleep 的緊迴圈。** baud 設錯、STM32 沒在送資料、
接錯 COM port 時，這個 thread 會**燒滿一個 CPU 核心**，跟推論與控制搶 CPU。
考慮到 `CPU_THREADS=2` 的設定，這會直接拖慢整條管線。

### 改法：`__init__` 加狀態欄位（在啟動 thread 之前）

```python
        self._reader_error: str | None = None
        self._frames_ok = 0
        self._frames_bad = 0
        self._t_first_frame: float | None = None
        self._last_drop_report_t = 0.0
```

### 改法：替換 `_reader`

```python
        def _reader():
            import serial as _serial
            miss = 0
            while not self._stop:
                try:
                    if not find_sync(self.ser):
                        # 找不到 SYNC：可能 baud 設錯、STM32 沒在送、接錯 COM port。
                        # 原本是無 sleep 的緊迴圈，會燒滿一個 CPU 核心並跟推論
                        # 搶資源（本專案 CPU_THREADS=2，影響直接）。
                        miss += 1
                        if miss % 200 == 0:
                            time.sleep(0.01)
                        continue
                    r = read_one_frame(self.ser)
                    if r is None:
                        self._frames_bad += 1
                        continue
                    miss = 0
                    _status, ch = r
                    if self._t_first_frame is None:
                        self._t_first_frame = time.monotonic()
                    self._frames_ok += 1
                    self._q.append([ch[c - 1] for c in C.CH_CSV_COLS])
                except (_serial.SerialException, OSError) as e:
                    # USB 被拔、STM32 重啟、驅動異常。原本會讓 thread 靜默死亡
                    # —— watchdog 雖然會把手臂推進 SAFE（fail-safe），但沒有任何
                    # 訊息說明是串列埠掛了，使用者會往分類器方向找錯。
                    self._reader_error = f"{type(e).__name__}: {e}"
                    print(f"\n[live] ✖ 串列讀取失敗，reader thread 結束：{e}")
                    return
                except Exception as e:      # noqa: BLE001 不讓未預期例外靜默吞掉
                    self._reader_error = f"{type(e).__name__}: {e}"
                    print(f"\n[live] ✖ reader thread 未預期例外：{e!r}")
                    return
```

### 改法：`poll()` 在 reader 已死時明確拋出

在 `LiveSource.poll()` 開頭插入：

```python
        if self._reader_error is not None and not self._q:
            # 讓上層停下來並顯示原因，而不是只靠 watchdog 靜默進 SAFE。
            raise RuntimeError(f"串列來源已中斷：{self._reader_error}")
```

> `run_arm_control` 的主迴圈會捕捉這個 `RuntimeError` 並走 `finally`
> （見 FIX-P），手臂會正常回 home 並斷線。

---

## FIX-N：掉包偵測未實作（P2）

### 問題

規格書 §10 明確要求：

> 串列緩衝溢位：80 kB/s 下 OS 驅動緩衝只有 ~51 ms 餘裕，掉包會靜默推進
> `sample_index`。對策：上游 `set_buffer_size(rx_size=262144)`；**控制層用
> wall-clock 對照 `sample_index/FS` 偵測掉包**。

`LiveSource` 只做了 `set_buffer_size`，**對照偵測完全沒實作**。而且
`_status, ch = r` 把 status byte 直接丟掉 —— `record_session.py` 是有統計
bad frame 的，這裡沒有。

**後果**：掉包會靜默發生。掉包造成訊號不連續 → 濾波與特徵算錯 → confidence
下降 → 觸發 FIX-C 的凍結或誤判。而**現象與「模型訓練得不好」完全無法區分**。

> ⚠️ `read_one_frame` 目前只回傳 `(status, channels)`，**沒有回傳 frame 裡的
> sample_index**。要做完整的索引連續性檢查得改 `src/hardware/record_session.py`，
> 那超出本修改單範圍。因此這裡實作**次佳版本**：用「實際收到的 frame 數」對照
> 「wall-clock × FS 應該收到的 frame 數」。

### 改法：新增 `drop_stats()`

```python
    def drop_stats(self, now: float | None = None) -> tuple[int, int, float]:
        """回傳 (收到的 frame 數, 壞 frame 數, 估計掉包率)。

        規格書 §10 的次佳實作：read_one_frame 不回傳 frame 內的 sample_index，
        無法做索引連續性檢查，所以改用 wall-clock 對照 ——
        期望 frame 數 = 經過秒數 × FS，實收少於期望即代表掉包。
        """
        now = time.monotonic() if now is None else now
        if self._t_first_frame is None:
            return 0, self._frames_bad, 0.0
        elapsed = now - self._t_first_frame
        if elapsed < 1.0:
            return self._frames_ok, self._frames_bad, 0.0
        expected = elapsed * C.FS
        rate = max(0.0, 1.0 - self._frames_ok / expected)
        return self._frames_ok, self._frames_bad, rate
```

### 改法：`poll()` 尾端定期告警

在 `LiveSource.poll()` 的 `return out` 之前插入：

```python
        # 每 5 秒回報一次掉包狀況（規格書 §10）。掉包會靜默破壞訊號連續性 ——
        # 濾波與特徵算錯 → confidence 下降 → 誤判或凍結，而現象與「模型不好」
        # 完全無法區分。必須主動量測。
        if now - self._last_drop_report_t >= 5.0:
            self._last_drop_report_t = now
            ok, bad, rate = self.drop_stats(now)
            if rate > 0.01 or bad > 0:
                print(f"\n[live] ⚠ 掉包率約 {rate * 100:.2f}%"
                      f"（收到 {ok} frame，壞 {bad}）—— 檢查 CPU 負載與 USB")
```

---

## FIX-O：`ReplaySource` 播完之後沒有任何訊息（P3）

`poll()` 回傳空 list，0.5 秒後 watchdog 觸發 → SAFE，使用者看到的是
`ERROR 進入 SAFE：watchdog` —— 誤以為出錯，其實是重播正常結束。

FIX-K 的 `status_text()` 已回傳「重播已結束」；再在 `poll()` 尾端加一次性訊息：

```python
        if not self._preds and not self._announced_end:
            self._announced_end = True
            print(f"\n[replay] 重播結束。接下來 watchdog 會在 {AC.WATCHDOG_SEC}s 後"
                  "把手臂推進 SAFE —— 這是正常結束，不是錯誤。")
```

`sources.py` 檔頭需加：

```python
from control import arm_config as AC
```

---

## 新檔：`tests/test_sources.py`

```python
"""PredictionSource 介面契約測試（不需要硬體、不需要模型）。

只測 MockSource 與基底介面 —— ReplaySource / LiveSource 需要模型或串列，
留給實機驗證（見「上機注意事項」）。
"""
from __future__ import annotations

from control.arm_controller import Prediction
from control.sources import MockSource, PredictionSource
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
```

---

# PART 5 — `run_arm_control.py`

## FIX-Q：source 建構失敗時手臂留在 enable 狀態（P0）

### 問題

現在的順序是：

```python
    arm = MockArm() if a.dry_run else _connect_real_arm(a.arm_ip)
    controller = ArmController(arm, log_path=log_path)
    controller.startup()                   # ★ 手臂已 enable、已回 home

    if a.source == "replay":
        source = ReplaySource(a.csv, a.model, ...)      # ← 可能 raise
    else:
        source = LiveSource(a.port, a.baud, a.model, ...)  # ← 可能 raise

    try:
        ...
    finally:
        ...
```

`ReplaySource.__init__` 有 `raise SystemExit("找不到模型 ...")`，`LiveSource`
同樣有，而且 `serial.Serial(port, baud)` 在 COM port 被佔用或不存在時會拋
`SerialException`。

這些例外發生在 `try` **之前** → `finally` 不會執行 →
**手臂已 enable、已回 home，然後程式退出，手臂就這樣留著。**

這很容易發生：打錯 COM port、忘記關掉另一個佔用序列埠的程式、模型路徑拼錯。

### 改法：先建 source，後連手臂；`startup()` 移進 `try`

（完整程式碼見下方 FIX-K/L 的 `main()`。）

---

## FIX-P：`finally` 區塊會讓手臂留在 enable 狀態（P0）

### 問題

```python
    finally:
        print()
        source.close()          # ← 若這裡拋例外...
        controller.shutdown()   # ← ...這行就不會執行
        print("已停止、回 home、斷線。")
```

`LiveSource.close()` 做 `self.ser.close()`，USB 已被拔掉時可以拋
`SerialException`。此時 `controller.shutdown()` **完全不會執行** →
**手臂沒有停止、沒有回 home、沒有 disconnect，仍處於 motion_enable 狀態**。

這直接違反規格書 §8 驗收條件 5，而且是最不該失敗的路徑。

另外最後那行 `print("已停止、回 home、斷線。")` 是**無條件印出的謊話** ——
FIX-E 之後 `shutdown()` 已經會自己回報是否真的回了 home，這行必須拿掉。

### 改法

見下方 `main()` 的 `finally` 區塊。要點：
1. **`controller.shutdown()` 排在 `source.close()` 前面**（手臂安全優先於資源清理）
2. **兩者各自包 try** —— 任一失敗都不能阻止另一個執行
3. 移除無條件的成功訊息

---

## FIX-R：IP 錯誤時症狀與 Bug A 完全相同（P0）

### 問題

```python
def _connect_real_arm(ip: str):
    ...
    return XArmAPI(ip, is_radian=False)
```

`XArmAPI(ip)` **不會**在 IP 不可達時拋例外 —— 它在背景執行緒重試連線。所以：

```
IP 打錯 / 手臂沒開機 / 網路線沒插
  → XArmAPI 物件建立成功
  → startup() 讀 arm.warn_code / arm.error_code 都是初始值 0
  → motion_enable / set_mode / set_state / set_servo_angle 全部靜默失敗
  → 主迴圈跑起來，畫面顯示 [ARMED]、target 在變、cmd_sent=1
  → 手臂完全不動
```

**這個症狀跟 Bug A 一模一樣**，而且更常發生（打錯一個數字就中）。修好 Bug A
之後若不修這條，實機上還是會遇到同樣的「畫面全綠、手臂不動」，而且無法從
畫面判斷是哪一種。`XArmAPI` 有 `connected` 屬性（bool），可以直接檢查。

### 改法：完整替換 `_connect_real_arm`

```python
def _connect_real_arm(ip: str, timeout: float = 5.0):
    """連線實體手臂，並【確認連線真的成功】。

    XArmAPI(ip) 在 IP 不可達時不會拋例外（背景重試），所以物件一定建得起來。
    若不主動檢查 connected，後續所有指令都會靜默失敗，而終端機顯示 ARMED、
    target 在變、cmd_sent=1 —— 症狀與「手臂處於 STOP 狀態」（見 FIX-A）
    完全相同，幾乎不可能從畫面判斷是哪一種。這裡把它擋在最前面。
    """
    try:
        from xarm.wrapper import XArmAPI
    except ImportError:
        sdk_root = AC.REPO_ROOT / "xArm-Python-SDK-master"
        sys.path.insert(0, str(sdk_root))
        from xarm.wrapper import XArmAPI

    print(f"連線手臂 {ip} …")
    arm = XArmAPI(ip, is_radian=False)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if arm.connected:
            break
        time.sleep(0.2)
    else:
        try:
            arm.disconnect()
        except Exception:                  # noqa: BLE001
            pass
        raise SystemExit(
            f"✖ 無法連線手臂 {ip}（{timeout:.0f}s 逾時）。請檢查：\n"
            f"  1. 控制箱電源與網路線\n"
            f"  2. IP 是否正確（arm_config.ARM_IP = {AC.ARM_IP}）\n"
            f"  3. 電腦與控制箱是否在同一網段（試 ping {ip}）\n"
            f"  4. UFACTORY Studio 是否正佔用連線\n"
            f"※ 沒有手臂時請用 --dry-run。")

    # report 通道要幾百 ms 才會推送第一輪狀態；startup() 讀 warn_code /
    # error_code 之前必須等，否則讀到的是還沒更新的 0（誤判為「沒有錯誤」）。
    time.sleep(0.5)
    print(f"已連線。firmware={arm.version}")
    return arm
```

> `startup()` 裡原本的 `time.sleep(0.5)` **保留**（多等無害，且 `startup()`
> 可能被測試單獨呼叫）。

---

## FIX-S：非 Windows 平台按鍵靜默失效（P2）

### 問題

```python
def _poll_key() -> str | None:
    try:
        import msvcrt
    except ImportError:
        return None                        # ← 靜默降級
```

在 Linux / macOS 上永遠回傳 `None` → **Enter / q / z 全部失效**：使用者會一直
按 Enter 但程式停在 DISARMED，以為程式壞了；無法按 `z` 歸零 J6。而且**完全沒有
任何提示**。使用者目前用 Windows（COM10），優先度不高，但至少要說一句。

### 改法

```python
_KEYS_AVAILABLE = False
try:
    import msvcrt as _msvcrt          # noqa: N812
    _KEYS_AVAILABLE = True
except ImportError:
    _msvcrt = None


def _poll_key() -> str | None:
    """非阻塞讀取一個按鍵；沒有按鍵回傳 None。

    ⚠️ 目前只支援 Windows（msvcrt）。非 Windows 平台永遠回傳 None，代表
    Enter / q / z 全部失效 —— 只能用 --yes 啟動、Ctrl-C 結束。
    main() 會在啟動時明確警告，不要讓它靜默降級（使用者會一直按 Enter
    然後以為程式壞了）。
    """
    if _msvcrt is None:
        return None
    if _msvcrt.kbhit():
        return _msvcrt.getwch()
    return None
```

---

## FIX-T：主迴圈是固定 sleep 而非固定週期（P3，只改註解）

### 問題

實際週期 = 10 ms + 工作時間（`print`、CSV 寫入 + `flush`、模型推論）。而
**Windows 上 `time.sleep(0.01)` 的實際粒度約 15.6 ms**（預設 timer resolution），
所以真實 tick 率可能是 **40–60 Hz，不是文件宣稱的 100 Hz**。

| 項目 | 受影響嗎 | 原因 |
|---|---|---|
| `CMD_PERIOD` 節流 | ❌ | 以 `time.monotonic()` 比較，時間基準正確 |
| J6 jog 速率 | ❌ | 同上 |
| watchdog | ⚠️ 解析度變差 | 0.5 s 判定最差延遲 ~16 ms，可接受 |
| `MAX_PREDS_PER_TICK` | ⚠️ 略受影響 | 60 Hz 下每 tick 可能拿到 1 筆而非 0.2 筆 |

**結論：功能上安全，但註解與文件宣稱的「100 Hz 輪詢主迴圈」不準確。**

### 改法

**不要**引入 `time.perf_counter()` 補償或 `timeBeginPeriod`。只把註解寫對，
並在結束時印出實測頻率供上機時記錄（見上機注意事項 M-16）。

---

## `main()` 完整替換（含 FIX-K / L / P / Q / S / T）

`_render_status` 先加一個參數：

```python
def _render_status(status: dict, source_note: str = "") -> None:
    line = (f"[{status['state']:8s}] "
            f"hand={status['hand_name']:6s}({status['p_hand']:.2f}) "
            f"elbow={status['elbow_name']:5s}({status['p_elbow']:.2f}) "
            f"shoulder={status['shoulder_name']:5s}({status['p_shoulder']:.2f}) | "
            f"J2={status['J2']:+6.1f} J3={status['J3']:+6.1f} J6={status['J6']:+6.1f} | "
            f"grip={status['grip']:6s} | cmdnum={status['cmdnum']}")
    if source_note:
        line += f" | {source_note}"
    print("\r" + line.ljust(150), end="", flush=True)
```

`main()` 從參數檢查之後全部替換：

```python
    if a.source == "replay" and not a.csv:
        ap.error("--source replay 需要 --csv")
    if a.source == "live" and not a.port:
        ap.error("--source live 需要 --port")

    print("=" * 78)
    print("⚠️  維持姿勢時手臂必須懸空 —— 手肘/前臂/上臂不能靠在桌面、扶手或身側，")
    print("    否則重力被外物撐住、EMG 消失，系統會判成 rest 讓機械手臂回 home。")
    print("=" * 78)

    if not _KEYS_AVAILABLE:
        print("!" * 78)
        print("⚠️  本平台不支援即時按鍵（僅 Windows 的 msvcrt）：")
        print("    Enter / q / z 全部無效。請用 --yes 自動 ARM，用 Ctrl-C 結束。")
        print("    J6 無法用 z 歸零 —— 結束時 shutdown() 回 home 會一併歸零。")
        print("!" * 78)

    # ★ 先建 source，再連手臂（見修改單 FIX-Q）。
    #   source 的建構是最容易失敗的一步（模型路徑錯、COM port 被佔用、CSV 不存在），
    #   而它一旦在 controller.startup() 之後失敗，例外會發生在 try/finally 之外
    #   —— 手臂已 enable、已回 home 然後程式退出，就這樣留著。
    #   把它移到前面，失敗時手臂根本還沒被碰過。
    print("建立 Prediction 來源…")
    if a.source == "mock":
        source = MockSource()
    elif a.source == "replay":
        source = ReplaySource(a.csv, a.model, calib_sec=a.calib_sec)
    else:
        source = LiveSource(a.port, a.baud, a.model, calib_sec=a.calib_sec)

    arm = MockArm() if a.dry_run else _connect_real_arm(a.arm_ip)
    log_path = a.log or (AC.LOG_ROOT / f"control_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    controller = ArmController(arm, log_path=log_path)

    def _try_engage() -> None:
        """把關：來源沒 ready 就不准 ARM（見修改單 FIX-K）。

        LiveSource 的前 calib_sec 秒 poll() 一律回傳空 list。若在這段期間
        engage()，WATCHDOG_SEC=0.5s 會【必然】觸發並把手臂推進 SAFE，而使用者
        會誤以為是串列或模型壞了。
        """
        if not source.is_ready():
            print(f"\n[拒絕 ARM] 來源尚未就緒：{source.status_text() or '請稍候'}")
            return
        controller.engage()
        # engage() 內含 wait=True 的回 home，會阻塞數秒；期間來源端持續累積。
        # 若不 resync，下一次 poll 會一次吐出數十筆 Prediction，全部帶同一個
        # now 灌進 DofDebouncer → 時間語意崩壞 → 可能立刻觸發轉態（FIX-L）。
        source.resync(time.monotonic())
        print("\n已 ARMED。按 q 結束；按 z 將 J6 歸零。")

    n_tick = 0
    t_loop0 = time.monotonic()
    try:
        controller.startup()               # ★ 移進 try —— 見 FIX-Q

        if a.yes:
            # --yes 不能無條件立刻 engage（見 FIX-K）：live 來源要等校準完成。
            print("--yes：等待來源就緒後自動 ARM…")
            deadline = time.monotonic() + a.calib_sec + 10.0
            while not source.is_ready() and time.monotonic() < deadline:
                source.poll(time.monotonic())      # 推進校準
                print(f"\r  {source.status_text()}".ljust(78), end="", flush=True)
                time.sleep(0.05)
            print()
            _try_engage()
        else:
            print("按 Enter 進入 ARMED；按 q 結束；按 z 將 J6 歸零。")

        while True:
            now = time.monotonic()
            n_tick += 1

            preds = source.poll(now)
            if len(preds) > AC.MAX_PREDS_PER_TICK:
                # 主迴圈被阻塞導致來源端積壓（見修改單 FIX-L）。這些 Prediction
                # 全部會帶同一個 now 進入 DofDebouncer，讓「N 幀 = N × STEP_SEC 秒」
                # 的假設失效。即時控制要的是「現在」，所以只保留最新的幾筆。
                total = len(preds)
                preds = preds[-AC.MAX_PREDS_PER_TICK:]
                print(f"\n[warn] 單 tick 收到 {total} 筆 Prediction，"
                      f"丟棄較舊的 {total - AC.MAX_PREDS_PER_TICK} 筆"
                      f"（主迴圈可能被阻塞）")
            for pred in preds:
                controller.on_prediction(pred, now)

            status = controller.tick(now)
            _render_status(status, source.status_text())

            key = _poll_key()
            if key in ("\r", "\n"):
                _try_engage()
            elif key in ("q", "Q", "\x03"):
                break
            elif key in ("z", "Z"):
                controller.zero_j6()

            # ⚠️ 這是固定 sleep，不是固定週期：實際 tick 間隔 = 10 ms + 工作時間，
            #    而 Windows 的 sleep 粒度約 15.6 ms → 真實約 40–60 Hz，不是 100 Hz
            #    （見修改單 FIX-T）。所有時間相關判定（CMD_PERIOD、jog、watchdog）
            #    都以 time.monotonic() 為基準、不依賴 tick 數，所以功能上安全；
            #    但別在文件裡宣稱 100 Hz。
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[Ctrl-C] 中斷，開始收尾…")
    except RuntimeError as e:              # 例如 FIX-M 的「串列來源已中斷」
        print(f"\n[error] 來源異常，開始收尾：{e}")
    finally:
        print()
        hz = n_tick / max(1e-9, time.monotonic() - t_loop0)
        print(f"[info] 主迴圈實測平均 {hz:.0f} Hz（{n_tick} ticks）")
        # ★ 手臂安全優先於資源清理，且兩者各自包 try —— 任一失敗都不能阻止
        #   另一個執行（見修改單 FIX-P）。原本 source.close() 拋例外會讓
        #   controller.shutdown() 整段跳過，手臂留在 motion_enable 狀態。
        try:
            controller.shutdown()          # 內部會自行回報是否成功回 home
        except Exception as e:             # noqa: BLE001
            print(f"[error] shutdown 失敗：{e!r}")
            print("        ⚠️ 手臂可能仍 enable，請手動用 UFACTORY Studio 停機。")
        try:
            source.close()
        except Exception as e:             # noqa: BLE001
            print(f"[warn] source.close() 失敗（不影響手臂安全）：{e!r}")
```

> 注意：`log_path` 的計算從原本位置移到 `controller` 建立之前，`main()` 裡
> 原本那一行要刪掉，不要留兩份。

---

# 自我檢查清單

## 測試

- [ ] `cd moving_pose && pytest tests/ -v` 全綠
- [ ] `test_debounce.py` 有 **8** 個測試（既有 5 + 新增 3）
- [ ] `test_sources.py` 有 **5** 個測試
- [ ] `test_mapping.py` 既有測試**未修改**且全部通過

## `validate()` 負向測試（測完記得改回來）

- [ ] `JOG_RATE_DEG_S = 4.0` → import 時 assert 失敗，訊息說明死區問題 → **改回 45.0**
- [ ] `HOME_ANGLES[1] = -95.0` → assert 失敗 → **改回 -90.0**
- [ ] `VOTE_N = 8` → assert 失敗 → **改回 4**
- [ ] 暫時把 `semg/v2/config.py` 的 `STEP_SEC` 改成 `0.1` → `arm_config` import
      時 assert 失敗並指出兩邊不一致 → **改回 0.05**

## dry-run 行為

- [ ] `python src/control/run_arm_control.py --source mock --dry-run` 能啟動
- [ ] 按 Enter 後印出 `已 ARMED`，且終端機第一個 `set_servo_angle` 是 **home**
      `[+0.0, -90.0, +180.0, +0.0, +0.0, +0.0]`（**FIX-B 迴歸**）
- [ ] 把 `MockSource()` 暫時改成 `MockSource(loop=False)`：
  - [ ] 腳本跑完 0.5 s 後 log 出現 `ERROR 進入 SAFE：watchdog`
  - [ ] 此時按 Enter → 應顯示 `[拒絕 ARM] 來源尚未就緒`（**FIX-K 迴歸**：
        `is_ready()` 在 `_done` 之後回 False）
  - [ ] 改回 `MockSource()`
- [ ] 用 `MockSource()`（循環）測 SAFE 恢復（**FIX-A 迴歸**）：暫時把
      `AC.WATCHDOG_SEC` 改成 `0.05` 逼出 SAFE，再按 Enter，確認印出
      `[MockArm] motion_enable(enable=True)` / `set_mode(0)` / `set_state(0)`
      三行 → **改回 0.5**
- [ ] 按 q，最後印出 `已回 home、斷線。`（不是舊的無條件字串）
- [ ] 結束時印出 `[info] 主迴圈實測平均 xx Hz`（FIX-T）

## 錯誤路徑

- [ ] `--source replay --csv /nonexistent.csv --dry-run` → 報錯退出，
      **且完全沒有印出任何 `[MockArm]` 指令**（證明 FIX-Q 生效：手臂還沒被碰）
- [ ] `--source mock --arm-ip 10.99.99.99` → 5 秒後 `✖ 無法連線手臂` 並列出
      四項檢查清單（FIX-R 生效）

## 靜態檢查

- [ ] `grep -rn "last_cmd_t = 0.0" src/control/` → 找不到（已改成 `now`）
- [ ] `grep -rn "已停止、回 home、斷線" src/control/` → 找不到（已移除謊話）
- [ ] `grep -rn "STEP_SEC" src/control/arm_config.py` → 定義區有完整註解
- [ ] `grep -n "if not find_sync" src/control/sources.py` → 附近有 `time.sleep`
- [ ] 沒有動到 `src/semg/v2/`、`src/hardware/` 任何檔案

---

# 本單不涵蓋的事項

| 項目 | 為什麼不在這裡 |
|---|---|
| `SOFT_LIMITS` / `JOINT_LIMITS` 數值 | 需接實機讀 `arm.joint_limits` |
| `JOG_RATE_DEG_S` / `GRIP_HOLD_MAX_SEC` / `COLLISION_SENSITIVITY` | 需實機手感與實測 |
| `P_MIN` 三個門檻 | 需模型 confidence 分佈 |
| 合併 `infer.Recognizer` 與 `DofDebouncer` 的兩層投票（延遲最佳化） | 需先 review `infer.py`，且會動到 `src/semg/v2/` |
| `read_one_frame` 回傳 sample_index 以做完整掉包偵測 | 會動到 `src/hardware/record_session.py` |
| `config.ACTUATOR_SEMANTICS` 與控制層不一致（index/thumb 已改配 J6） | 會動到 `src/semg/v2/config.py`；僅註解不一致，不影響執行 |
| `MockArm.cmd_num` 永遠是 1，dry-run 測不出佇列堆積 | 是 `MockArm` 的固有限制，只能實機驗證 |
| 非 Windows 的按鍵支援（termios 分支） | 使用者用 Windows，不值得為此增加平台相依 |

以上見另一份 **「上機注意事項」**。
