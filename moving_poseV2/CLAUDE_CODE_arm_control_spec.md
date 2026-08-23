# sEMG → UFACTORY Lite 6 即時控制層實作規格

> **給 Claude Code 的任務書。**
> 目標：新增 `src/control/` 模組，把 v2 分類模型的輸出轉成 Lite 6 機械手臂動作。
> 語言：程式碼註解與 log 訊息用繁體中文，識別字（變數/函式/類別名）用英文。
> 本文件同時作為專案脈絡說明，讓你了解上游硬體與訊號管線的現況。

---

## 0. 一頁摘要（先讀這段）

| 項目 | 值 |
|---|---|
| 要做的事 | 寫 `ArmController`，吃分類器輸出，驅動 Lite 6 |
| 模型輸出 | **三頭多輸出**，不是 8 類 argmax（見 §3，最容易寫錯的地方） |
| 輸出速率 | 20 Hz（`STEP_SEC = 0.05`） |
| 控制範式 | 肘/肩 = 位置鏡像；pinch = **邊緣觸發 toggle**；index/thumb = J6 雙向 jog（皆已確認） |
| 手臂 IP | `192.168.1.181`（沿用 `coffee.py`） |
| 既有腳本 | `coffee.py` 可沿用連線慣例，**不可**沿用佇列式寫法（見 §6.5） |
| 手臂控制模式 | `set_mode(0)` 位置控制模式，**全程單一模式**，不切換 |
| 最高優先原則 | 安全 > 可驗證 > 平順 > 功能完整 |
| 硬性禁止 | 不引入 RTOS 概念、不改韌體、不動 `src/semg/v2/` 既有檔案 |

---

## 1. 專案脈絡（研究進度）

### 1.1 主題與時程
生醫工程專題：即時 sEMG 擷取 + 機械手臂控制。**硬性 deadline 2026/9/4。**
原本規劃 EEG Motor Imagery，2026/6 因時程改為 sEMG；硬體與韌體架構未變，
只調整了 ADS1299 暫存器參數與 Python 端 scaling factor。

### 1.2 硬體鏈路

```
前臂/上臂表面電極 (7ch)
      │  差動輸入
      ▼
TI ADS1299EEG-FE Rev A   ── SPI Mode 1 (CPOL=0,CPHA=1), SCLK≈250kHz ──┐
  24-bit ΔΣ ADC × 8ch                                                  │
      │ DRDY (PB0/EXTI0)                                               │
      ▼                                                                │
STM32L475VGT6 (B-L475E-IOT01A)  ◄──────────────────────────────────────┘
  EXTI → SPI DMA(27B) → ParseFrame → UART DMA(40B frame)
      │ USART1 PB6(TX)/PB7(RX), 1,600,000 baud
      ▼
PC (Python)  ── 分類 ──► 本文件要寫的控制層 ── Ethernet TCP ──► Lite 6
```

### 1.3 韌體階段完成度

| Stage | 內容 | 狀態 |
|---|---|---|
| 1 | Device ID 讀取 | ✅ |
| 2 | DRDY EXTI 驗證 | ✅ |
| 3 | 27-byte SPI DMA 讀取（EXTI→DMA→`HAL_SPI_TxRxCpltCallback`→ParseFrame） | ✅ |
| 4 | UART DMA 40-byte frame 串流到 PC（`uart_stream.c/h`） | ✅ |
| 5 | sEMG 暫存器切換 + 硬體驗證 | 🔲 進行中 |
| 6 | Python 分類管線（程式碼已存在，模型尚未訓練驗證） | 🔲 |
| 7 | **機械手臂介面 ← 本文件的任務** | 🔲 |

> ⚠️ Stage 6 的模型還沒訓練完。**本任務要能在沒有真模型的情況下開發與測試**，
> 所以必須提供 `--source mock` 的假輸入來源（見 §7）。

### 1.4 目前有效參數（以 `config.py` 為準，README 有舊值請忽略）

```
FS              = 2000            # SPS，CONFIG1 = 0x93 (DR=011)
ADS_GAIN        = 8               # CHxSET = 0x40 → 滿刻度 ±562.5 mV
ADS_VREF        = 4.5, ADS_BITS = 24
LSB_MV          = 2*4.5/(8*2^24)*1e3 ≈ 6.706e-5 mV  (≈ 67.06 nV)
CONFIG3         = 0xEC            # 內部參考 + BIAS amp
BIAS_SENSP/N    = 0x7F            # CH1–CH7 全部參與 BIAS 迴路
UART baud       = 1_600_000       # UART_OVERSAMPLING_8 @ HSI 16MHz
Frame           = 40 B: 0xA5 0x5A + "<I8i" + XOR + 0x0D  → 80 kB/s
WIN_SEC         = 0.5, STEP_SEC = 0.05, SEQ_LEN = 10   → 序列涵蓋 0.95 s
```

> ❌ **任何寫著 `0x40 = Gain 4`、`LSB = 134 nV`、`4kSPS/8kSPS`、`115200 baud`
> 的舊註解都是錯的，不要沿用。** `config.py` 是唯一權威。

### 1.5 電極配置（7 通道）

🔴 **BLOCKER：ch2 / ch4 的對應目前有未解決的衝突，實作控制層時不需要用到通道名稱，
但若你要動任何跟 `CROSS_PAIRS` / `CH_NAMES` 有關的程式碼，先停下來問使用者。**

| ch | `config.py` 現行（程式碼） | 使用者口述的實際貼法 | 角色 |
|---|---|---|---|
| 1 | `latissimus_dorsi` 背闊肌 | 背闊肌 ✅ 一致 | 肩伸展（拮抗） |
| 2 | `deltoid_anterior` 三角肌前束 | **二頭肌** ❌ | — |
| 3 | `triceps` 三頭肌 | 三頭肌 ✅ 一致 | 肘伸展（拮抗） |
| 4 | `biceps` 二頭肌 | **三角肌前束** ❌ | — |
| 5 | `wrist_flexor` 曲腕肌 | 曲腕肌 ✅ | 手部協同 |
| 6 | `index` 食指屈肌 | 食指屈肌 ✅ | 手部協同 |
| 7 | `thumb` 拇指屈肌 | 拇指屈肌 ✅ | 手部協同 |
| 8 | 未使用（`USE_EXTENSOR_CHANNEL = False`） | — | — |

衝突的後果：對分類準確率**無**影響（模型吃全部通道），但會讓 `config.CROSS_PAIRS`
裡的兩個拮抗對算錯東西 ——

```python
("biceps", "triceps")                     # 宣稱是肘的屈/伸拮抗
("deltoid_anterior", "latissimus_dorsi")  # 宣稱是肩的屈/伸拮抗
```

若實際貼法是右欄，這兩個 log ratio 實際算的是「三角肌/三頭肌」與「二頭肌/背闊肌」，
橫跨兩個關節、無拮抗意義。修正方式二擇一（由使用者依實體貼片決定）：
把 `CH_NAMES` 的 ch2/ch4 對調，或把電極線對調。**不要自行猜測。**

### 1.6 ⚠️ 操作硬性前提：維持姿勢時手臂必須懸空

手肘靠桌面、前臂撐扶手、上臂貼死身側 → 重力被外物支撐 → 肌肉不出力 →
EMG 消失 → 系統判成 rest → **機械手臂會自己回 home**。
訓練與操作都必須懸空。這一點請寫進程式的啟動提示訊息裡。

---

## 2. 上游介面：分類器輸出契約

控制層**不負責**推論。你要定義一個資料類別，由 `src/semg/v2/infer.py`
（或 mock 來源）餵進來：

```python
@dataclass(frozen=True)
class Prediction:
    t: float            # time.monotonic() 時間戳
    hand: int           # 0=rest 1=index 2=thumb 3=pinch
    elbow: int          # 0=rest 1=flex
    shoulder: int       # 0=rest 1=flex
    p_hand: float       # 該 head 的 softmax 最大值 (0~1)
    p_elbow: float
    p_shoulder: float
```

索引順序必須從 `config.DOFS` 與 `config.DOF_STATES` 動態取得，**不要硬編碼**：

```python
DOFS = ("hand", "elbow", "shoulder")
DOF_STATES = {
    "hand":     ("rest", "index", "thumb", "pinch"),
    "elbow":    ("rest", "flex"),
    "shoulder": ("rest", "flex"),
}
```

---

## 3. ⚠️ 最容易寫錯的地方：這是三頭多輸出，不是 8 類分類

使用者口語會說「我有 8 種分類」。那指的是**8 個狀態分散在 3 個 head**
（4 + 2 + 2），以及錄製時用的 8 個 cue 動作。**模型每個 timestep 同時輸出
三個獨立的類別**，所以：

```python
# ❌ 絕對不要這樣寫
gesture = np.argmax(logits)          # 8 選 1
dispatch(GESTURE_TABLE[gesture])

# ✅ 正確
hand     = np.argmax(logits_hand)     # 4 選 1
elbow    = np.argmax(logits_elbow)    # 2 選 1
shoulder = np.argmax(logits_shoulder) # 2 選 1
```

後果差異：正確版本支援「一邊捏著一邊彎肘」（`pick_and_lift`）與
「捏+彎肘+前舉」（`pick_and_raise`）這種**同時**成立的組合，而 8 選 1 做不到。
錄製協定裡本來就有這些跨體段組合的 cue，模型是為此設計的。

三個 head 各自獨立跑自己的去彈跳狀態機，互不阻塞。

---

## 4. 控制映射規格（使用者指定）

### 4.1 手臂 home 姿態

```python
HOME_ANGLES = [0.0, -90.0, 180.0, 0.0, 0.0, 0.0]   # J1..J6，單位：度
```

角度語意（使用者定義）：
- **J2 = 0°** → 上臂平舉、與地面平行。home 的 -90° 是自然下垂。
  → **肩屈曲方向 = J2 遞增**（-90 → 0）
- **J3 = 90°** → 屈肘至前臂垂直於上臂。home 的 180° 是伸直。
  → **肘屈曲方向 = J3 遞減**（180 → 90）

Lite 6 出廠限位（已查證，寫成常數並在送指令前 clamp）：

```python
JOINT_LIMITS = {
    1: (-360.0, 360.0),
    2: (-150.0, 150.0),
    3: ( -3.5,  300.0),
    4: (-360.0, 360.0),
    5: (-124.0, 124.0),
    6: (-360.0, 360.0),
}
# 關節速度上限 180 °/s、關節加速度上限 1145 °/s²（本專案請用遠低於此的值）
```

**再加一層本專案自訂的軟限位**（比出廠限位更保守，可用參數調）：

```python
SOFT_LIMITS = {
    2: (-90.0,   0.0),    # 肩：下垂 ↔ 平舉
    3: ( 90.0, 180.0),    # 肘：屈曲 ↔ 伸直
    6: (-180.0, 180.0),   # 腕旋轉
}
```

送出前必須 `clamp(SOFT_LIMITS) → clamp(JOINT_LIMITS)`，兩層都要。

### 4.2 映射表

**以下範式全部由使用者確認過，不是預設值，不要自行更動。**

| DOF | 狀態 | 控制範式 | 動作 |
|---|---|---|---|
| hand | `pinch` | **邊緣觸發 toggle** | 每次 rest→pinch 的**上升緣**切換夾爪：閉 ↔ 開 |
| hand | `index` | **持續 jog** ✅確認 | 維持期間 J6 往 **+** 方向持續轉 |
| hand | `thumb` | **持續 jog** ✅確認 | 維持期間 J6 往 **−** 方向持續轉 |
| hand | `rest` | — | J6 **停在當前角度**，不回歸 |
| elbow | `flex` | **位置鏡像** ✅確認 | J3 → `ELBOW_FLEX_ANGLE` = 90° |
| elbow | `rest` | 位置鏡像 | J3 → `HOME_ANGLES[2]` = 180° |
| shoulder | `flex` | **位置鏡像** ✅確認 | J2 → `SHOULDER_FLEX_ANGLE` = 0° |
| shoulder | `rest` | 位置鏡像 | J2 → `HOME_ANGLES[1]` = −90° |

**注意三種範式共存於同一支程式**：
- pinch 是**事件式**（只看 rest→pinch 的轉態，不看持續狀態）
- index/thumb 是**持續式**（看狀態有沒有維持）
- elbow/shoulder 是**位置式**（狀態直接對應絕對目標角）

這是刻意的設計，不要為了「一致性」把三者統一。

### 4.2.1 ⚠️ J6 是唯一有累積狀態的軸

J2/J3 是無狀態的：手勢 → 目標角是純函數，隨時可以重算。
**J6 不是** —— 它的角度取決於使用者過去 jog 了多久，是積分量。三個後果：

1. **會漂移**。誤判的 index/thumb 幀不會像位置鏡像那樣被下一幀糾正回來，
   誤差會累積。所以 J6 的信心門檻要比 J2/J3 高（`P_MIN["hand"] = 0.80`）。
2. **需要獨立的歸零手段**：終端機按 `z` → `J6 → 0°`（不影響其他軸）。
   `--dry-run` 也要支援。
3. **回 home 時 J6 一併歸零**，否則下一次啟動的起點不確定。

另外：`hand` head 的 rest 同時代表「夾爪要開嗎」與「J6 要停嗎」兩件事，
但 pinch 是邊緣觸發、rest 不會讓夾爪自動打開 —— **rest 只停 J6，不動夾爪**。
夾爪只由 pinch 的上升緣改變。這點務必寫對，否則手一放鬆東西就掉了。

### 4.3 夾爪

Lite 6 專屬夾爪只有開/關兩態（無位置回授、無力控）：

```python
arm.open_lite6_gripper(sync=False)    # sync=False → 立即執行，不進運動佇列
arm.close_lite6_gripper(sync=False)
arm.stop_lite6_gripper(sync=False)    # 停止供電，長時間夾持後呼叫可降溫
```

> `sync=True` 會把夾爪動作排進運動佇列，等前面的關節運動跑完才動 →
> 在即時控制下會有明顯延遲。**本專案一律用 `sync=False`。**

因為只有開/關，`config.ACTUATOR_SEMANTICS` 裡「半閉合」的語意在 Lite 6 硬體上
做不到，故 index/thumb 改配給 J6 —— 這是使用者的決定，與模型無關
（模型仍然分得出 index/thumb/pinch 三種手部協同）。

---

## 5. 去彈跳與安全層（本任務的技術核心）

模型 20 Hz 輸出，任何單幀誤判都會直接變成手臂抖動。**每個 head 都要過三道閘。**

### 5.1 閘一：信心門檻

```python
P_MIN = {"hand": 0.80, "elbow": 0.75, "shoulder": 0.75}   # 可調
```
`p < P_MIN` 的幀視為「無意見」，**不推進投票也不重置**（只是跳過），
避免中間過渡期的低信心幀被當成 rest。

### 5.2 閘二：N-of-M 多數決

環形緩衝區保留最近 M 幀，只有當某狀態在其中出現 ≥ N 次才允許轉態。

```python
VOTE_M = 6      # 300 ms 視窗
VOTE_N = 4
```

理由：模型的序列跨度已經是 0.95 s，20 Hz 相鄰輸出高度相關，
單純的「連續 K 幀相同」太容易被一個 dropout 幀打斷；多數決比較穩。

### 5.3 閘三：不反應期（refractory）

轉態成功後，該 DOF 在 `REFRACTORY_SEC` 內不接受新的轉態。

```python
REFRACTORY = {"hand": 0.60, "elbow": 0.40, "shoulder": 0.40}
```

pinch toggle 特別重要：0.6 s 不反應期可以避免「捏一下」被判成「捏兩下」
而讓夾爪開了又關。

### 5.4 看門狗（watchdog）

超過 `WATCHDOG_SEC = 0.5` 沒收到任何 `Prediction`（串列斷線、Reader thread 卡死），
立刻：
1. `arm.set_state(4)`（停止）
2. log ERROR
3. 進入 `SAFE` 狀態，需人工重新 arm 才能恢復

### 5.5 啟動 arm / disarm

程式啟動後**預設不接受動作指令**（`DISARMED`）。
需在終端機按 Enter 或指定的鍵才進入 `ARMED`。
隨時可按 `q` / Ctrl-C → 停止 + 回 home + disconnect。

---

## 6. 指令發送策略

### 6.1 為什麼用 mode 0 而不是 mode 1/mode 4

- mode 1（servo）+ `set_servo_angle_j` 要求 ~100 Hz 穩定餵點，
  一旦分類 thread 卡頓就會產生大加速度 → 危險。
- mode 4（關節速度）雖然最適合 jog，但 **mode 是全域的**，
  J2/J3 的位置鏡像同時需要 mode 0，無法共存。
- 因此：**全程 mode 0**，用「目標角追蹤」模擬 jog。

### 6.2 目標角追蹤迴圈

維護一組 `target_angles[6]`：

```python
# 位置鏡像（J2/J3）：狀態轉態時直接改 target
# jog（J6）：只要 jog 狀態維持，每個 tick 就把 target 往該方向推進 JOG_STEP_DEG

JOG_RATE_DEG_S = 45.0                       # J6 jog 角速度
JOG_STEP_DEG   = JOG_RATE_DEG_S * CMD_PERIOD
```

**送指令的節流條件（三個都要成立才送）：**

```python
now - last_cmd_t >= CMD_PERIOD                    # CMD_PERIOD = 0.10 s
and max(|target - last_sent_target|) >= 0.5       # 死區，避免無意義指令
and arm.cmd_num < CMDNUM_MAX                      # CMDNUM_MAX = 2，防佇列堆積
```

`cmd_num` 這道閘最重要 —— `set_servo_angle(wait=False)` 是排進佇列的，
20 Hz 無節制發送會讓佇列爆掉，手臂會在你放開手勢後繼續跑好幾秒。

```python
arm.set_servo_angle(
    angle=target_angles, speed=JOINT_SPEED, mvacc=JOINT_ACC,
    wait=False, is_radian=False,
)
JOINT_SPEED = 40.0     # °/s（上限 180，保守取值）
JOINT_ACC   = 200.0    # °/s²（上限 1145）
```

### 6.3 連線與初始化序列

沿用專案既有腳本 `coffee.py` 的慣例（**條件式**清錯誤，而不是無條件呼叫）：

```python
ARM_IP = "192.168.1.181"          # 沿用 coffee.py 的實機位址

arm = XArmAPI(ARM_IP, is_radian=False)
time.sleep(0.5)                   # coffee.py 慣例：等 report 通道建立再讀狀態
if arm.warn_code != 0:
    arm.clean_warn()
if arm.error_code != 0:
    arm.clean_error()
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(0)

arm.set_collision_sensitivity(3)                              # 中等靈敏度
arm.set_servo_angle(angle=HOME_ANGLES, speed=30, wait=True)   # 回 home，等完成
arm.open_lite6_gripper(sync=False)
```

> `time.sleep(0.5)` 不是迷信 —— `warn_code` / `error_code` 是從 report socket
> 更新的屬性，剛 `XArmAPI()` 完就讀會拿到初始值 0，等於跳過清錯誤。

錯誤處理：註冊 `register_error_warn_changed_callback`，
`error_code != 0` 時立刻進 `SAFE`，不自動 `clean_error()` 重試
（自動恢復會掩蓋碰撞事件，專題 demo 反而更危險）。

結束序列：

```python
arm.set_state(4)
arm.stop_lite6_gripper(sync=False)
arm.motion_enable(True); arm.set_state(0)
arm.set_servo_angle(angle=HOME_ANGLES, speed=30, wait=True)
arm.disconnect()
```

---

## 6.5 從 `coffee.py` 沿用什麼、不沿用什麼

專案裡已有一支可運作的實機腳本 `coffee.py`（自動泡咖啡序列）。
它是**佇列式的離線腳本**，本任務是**即時回饋控制**，兩者的架構要求相反。

### ✅ 沿用

| 項目 | 說明 |
|---|---|
| IP `192.168.1.181` | 實機位址 |
| `time.sleep(0.5)` + 條件式 `clean_warn/clean_error` | 見 §6.3 |
| `motion_enable → set_mode(0) → set_state(0)` 順序 | 標準初始化 |
| `open_lite6_gripper()` / `close_lite6_gripper()` | 夾爪 API 已驗證可用 |
| 角度全部用度（`is_radian` 預設 False） | 與既有腳本一致 |

### ❌ 不要沿用（會直接毀掉即時性）

| coffee.py 的做法 | 為什麼不能用 |
|---|---|
| `arm.set_pause_time(0.8)` 大量穿插 | 這是把延遲**排進運動佇列**。即時控制下等於故意製造 0.8 s 的指令延遲，且無法中斷。**本任務完全不使用 `set_pause_time`。** |
| 夾爪用預設 `sync=True` | 排進佇列 → 要等前面關節動完才夾。即時控制一律 `sync=False`（§4.3） |
| 連續發射 `set_position` / `set_servo_angle` 不看 `cmd_num` | 腳本模式下佇列堆積是預期行為；即時模式下會讓手臂在你放開手勢後繼續跑好幾秒。必須加 `cmd_num < 2` 閘門（§6.2） |
| `set_position` / `set_tool_position`（笛卡爾） | 本任務**只用關節空間** `set_servo_angle`。笛卡爾指令會經過 IK，在奇異點附近可能解出大幅跳動的關節角 |
| 7 元素角度陣列 `[..., 0.0]` | Lite 6 是 6 軸，第 7 個值多餘。本任務一律傳 6 元素 |
| `oriAcc = arm.last_used_joint_acc` | coffee.py 裡取了但沒用到，不要複製這行 |

### 一個 coffee.py 透露的有用事實

`can_angle` 用到 `J6 = -280`、`close_angle` 用到 `J6 = 265` —— 實機的 J6
確實可以轉超過 ±180°。所以 §4.1 的 `SOFT_LIMITS[6] = (-180, 180)` 是保守值，
若使用者覺得 jog 範圍不夠，這個數字可以放寬到 ±270 而不碰到出廠限位（±360）。

---

## 7. 檔案結構與 CLI

新增（**不要動 `src/semg/v2/` 底下任何既有檔案**）：

```
src/control/
  __init__.py
  arm_config.py       # 本文件所有常數，集中一處
  debounce.py         # DofDebouncer（信心門檻 + N-of-M + refractory）
  arm_controller.py   # ArmController：Prediction → xArm 指令
  sources.py          # PredictionSource：mock / replay / live 三種
  run_arm_control.py  # CLI 進入點
tests/
  test_debounce.py    # 純邏輯測試，不需要硬體
  test_mapping.py     # 狀態 → 目標角，含限位 clamp
```

### CLI

四種模式（`--source mock`/`replay`/`live` + 獨立的手動版）的實際啟動指令與
參數見 `docs/control-commands.md`（唯一的指令去處，這裡不重複維護）。

`--dry-run` 用一個 `MockArm` 假物件實作相同介面，把每次指令印到終端機。
**這是最重要的開發手段** —— 手臂控制的 bug 在實機上除錯代價太高。

### 即時終端機顯示（每 tick 更新一行）

```
[ARMED] hand=pinch (0.94) elbow=flex (0.88) shoulder=rest (0.91) | J2=-90.0 J3=134.2 J6=+12.0 | grip=CLOSED | cmdnum=1
```

---

## 8. 驗收條件

必須全部通過才算完成：

1. `pytest tests/` 全綠，且 `test_debounce.py` 涵蓋：
   單幀雜訊不觸發、低信心幀不推進投票、refractory 期間拒絕轉態。
2. `--dry-run` 下，連續 5 次 pinch 上升緣 → 夾爪狀態序列為
   `CLOSE, OPEN, CLOSE, OPEN, CLOSE`（不多不少）。
3. 任何情況下送出的角度都在 `SOFT_LIMITS` 內（用 property-based 或
   隨機序列測試驗證）。
4. 拔掉 `Prediction` 來源 0.5 s 後，log 出現 watchdog ERROR 且進入 SAFE。
5. Ctrl-C 一定回 home 並 disconnect，不留下 enable 狀態的手臂。
6. 所有可調參數都在 `arm_config.py`，程式碼本體沒有魔術數字。

---

## 9. 待確認事項

### 🔴 BLOCKER — 開工前必須先問使用者

1. **ch2 / ch4 的實際貼法**（見 §1.5）。
   `config.py` 說 ch2=三角肌前束、ch4=二頭肌；使用者口述是相反的。
   對本任務的控制邏輯**沒有**影響（控制層不碰通道名），但若你被要求
   一併修 `config.CROSS_PAIRS` 或分析腳本，**先確認再改**，不要猜。

### ✅ 已確認（不要再問，也不要改）

| 項目 | 值 |
|---|---|
| 手臂 IP | `192.168.1.181`（來自 `coffee.py`） |
| index / thumb → J6 | 持續 jog，兩指對應正反方向 |
| elbow / shoulder | 位置鏡像 |
| `ELBOW_FLEX_ANGLE` | 90°（J3，屈肘至垂直上臂） |
| `SHOULDER_FLEX_ANGLE` | 0°（J2，平舉水平） |
| home | `[0, -90, 180, 0, 0, 0]` |

### 🟡 次要 — 用預設值實作並標 `# TODO 待確認`，不要停下來等

1. J1 / J4 / J5 是否全程固定在 0（預設：是）
2. J6 jog 角速度 `JOG_RATE_DEG_S`（預設 45 °/s，實機試了再調）
3. `SOFT_LIMITS[6]` 取 ±180 還是 ±270（預設 ±180，見 §6.5）
4. 夾爪連續夾持自動斷電的門檻 `GRIP_HOLD_MAX_SEC`（預設 30 s）
5. 是否需要把控制事件寫成 log 檔供論文分析（預設：是，CSV 格式，
   欄位 `t, hand, elbow, shoulder, p_*, J2, J3, J6, grip_state, cmd_sent`）

## 10. 已知風險（實作時請主動避開）

| 風險 | 說明 | 對策 |
|---|---|---|
| 串列緩衝溢位 | 80 kB/s 下 OS 驅動緩衝只有 ~51 ms 餘裕，掉包會靜默推進 `sample_index` | 上游 `set_buffer_size(rx_size=262144)`；控制層用 wall-clock 對照 `sample_index/FS` 偵測掉包 |
| 佇列堆積 | `wait=False` 的指令會排隊，放手後手臂繼續跑 | `cmd_num < 2` 閘門（§6.2） |
| 手臂沒懸空 | EMG 消失 → 判成 rest → 手臂突然回 home | 啟動提示 + 位置鏡像回 home 用較慢速度 |
| 夾爪過熱 | Lite 6 夾爪長時間夾持會發熱 | 夾持超過 `GRIP_HOLD_MAX_SEC`（預設 30 s）自動 `stop_lite6_gripper` |
| 疲勞漂移 | sEMG 頻譜隨疲勞下移，信心下降 | 信心門檻不要設太高（0.8 已接近上限），並在 log 記錄平均信心 |
| 模型尚未訓練 | Stage 6 未完成 | 全部開發用 `--source mock` 完成，模型到位後只換 source |
