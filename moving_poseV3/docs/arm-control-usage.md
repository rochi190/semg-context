# sEMG → UFACTORY Lite 6 即時控制層 —— 使用說明

> 本文件說明 `src/control/` 模組怎麼裝、怎麼跑、每個參數是什麼意思。
> 設計依據見 `CLAUDE_CODE_arm_control_spec.md`（任務書，含所有決策理由）。
> 這是**控制層**，不含推論——分類模型（已訓練）與程式碼實際來自
> **`semg_realtime_V6/`**（repo 根目錄下的自足打包，跟 `moving_pose/`
> 同層，**不在**它底下），不是 `src/semg/`（那是舊快照，ch2/ch4 標籤未修正、
> 沒有模型）。`run_arm_control.py` 啟動時會把 `semg_realtime_V6/src` 插到
> `sys.path` 最前面，蓋過 `src/semg`。
>
> ⚠️ 2026-08-17 由 `moving_pose/semg_realtime/`（V3）換成 `semg_realtime_V6/`。
> V4 用 **split 架構**（hand 頭只看遠端 3 通道、elbow/shoulder 頭只看近端
> 4 通道，物理上互不干擾），直接解決了 V3 「抬肘/抬肩時彎食指被誤判成 pinch」
> 的問題（沒 cue 過的組合，hand 準確率從 5.0% → 85.4%）。同時新增 **dwell**
> 機制擋「移動過程被穩定地標成錯的狀態」。詳見 `semg_realtime_V6/README.md`。
>
> **上機前的準備、待決參數、實機檢查清單**都整理在
> `docs/on-machine-checklist.md`——這是本專案「上機類」資訊的固定去處，
> 之後有新的上機注意事項一律加進那份文件，不要開新檔案分散。

---

## 這包裡有什麼

```
src/control/
  __init__.py
  arm_config.py       所有可調參數集中一處（限位、速度、去彈跳門檻…）
  debounce.py          DofDebouncer：信心門檻 → N-of-M 多數決 → refractory
  arm_controller.py    Prediction 契約、MockArm、ArmController（核心狀態機）
  sources.py            PredictionSource：mock / replay / live 三種來源
  run_arm_control.py    CLI 進入點
tests/
  conftest.py           把 src/ 加進 sys.path，讓 `pytest tests/` 不需要額外設定
  test_debounce.py       DofDebouncer 純邏輯測試
  test_mapping.py         狀態 → 目標角映射 + 限位 clamp 測試
docs/
  arm-control-usage.md   ★ 本文件
```

**沒有包含推論邏輯或訓練資料**——控制層只負責「拿到 `Prediction` 之後怎麼讓手臂動」。

---

## ★ 快速調整指南——想改行為，去哪裡改

**所有可調參數都在同一個檔案：`src/control/arm_config.py`。** 沒有第二個地方
藏著同名的參數，改這裡就好，不用到處找。下表按「你想調整什麼」查：

| 想調整… | 改這個參數 | 說明 |
|---|---|---|
| **啟動/回 home 時的姿態**（初始狀態） | `HOME_ANGLES` | `[J1, J2, J3, J4, J5, J6]`，單位度。啟動、`shutdown()`、看門狗觸發 SAFE 前都會回到這個姿態 |
| 回 home 的速度 | `HOME_SPEED` | 較慢，避免回 home 時嚇到人 |
| 手臂控制 IP | `ARM_IP` | 見檔案內註解，別跟 `.181`（xArm Studio 用）／`.166`（回傳資料用）搞混 |
| 動作範圍太小/太大（肩、肘、腕能動的角度） | `SOFT_LIMITS` | 本專案自訂的保守限位；`JOINT_LIMITS` 是出廠硬限位，兩層都會 clamp，一般只需要改 `SOFT_LIMITS` |
| J6（腕）jog 太快/太慢 | `JOG_RATE_DEG_S` | 目前 45°/s，標 TODO 待確認，本來就預期要依實機手感調 |
| J6 jog **卡卡的/不等速** | `JOG_SPEED_CAP` / `JOG_ACC` / `JOG_BLEND_RADIUS_DEG` | 2026-08-19 加入，見下方專節說明；跟 `JOG_RATE_DEG_S` 是不同問題，不要搞混 |
| 送指令的節流頻率 | `CMD_PERIOD` | 也同時是 jog 的推進週期；調小會讓動作更即時但送指令更頻繁 |
| 手臂移動速度/加速度 | `JOINT_SPEED` / `JOINT_ACC` | 送 `set_servo_angle` 時用的值，出廠上限分別是 180°/s、1145°/s² |
| 動作辨識**多快生效**（靈敏度 vs 穩定度） | `P_MIN` / `VOTE_M` / `VOTE_N` | 信心門檻、多數決視窗大小、門檻票數；調小 `VOTE_M`/`VOTE_N` 或降 `P_MIN` 會更快反應但更容易誤判 |
| 轉態後**多久才接受下一次轉態**（防手勢誤判連環觸發） | `REFRACTORY` | 依 DOF 分開設定；⚠️ 只影響「轉入非 rest 狀態」，轉回 rest 一律立即生效（2026-08-17 修正，見 `debounce.py`），改這個不會讓放鬆變慢 |
| 多久沒收到訊號就進入 SAFE | `WATCHDOG_SEC` | 安全機制，不建議為了流暢度調大 |
| 夾爪連續閉合多久自動降溫斷電 | `GRIP_HOLD_MAX_SEC` | 目前 30s，標 TODO 待確認；斷電後夾爪仍可正常再次開/合，不需要額外操作 |
| 碰撞偵測靈敏度 | `COLLISION_SENSITIVITY` | 0~5，數字越小越不容易誤觸發，但真的碰撞時也比較晚反應 |
| 換模型 | `DEFAULT_MODEL` | 指到 `semg_realtime_V6/models/` 底下的 `.pt`（`final_v4_split`／`_tiny`／`_zhao`），要跟同一版 `semg_realtime_V6/src` 配對 |
| 分類器逐窗抖動的過濾強度 | `RECOGNIZER_VOTES` / `RECOGNIZER_CONF` | 目前刻意設 `votes=1`（關掉，交給 `VOTE_M`/`VOTE_N` 那層），一般不需要動 |
| 「移動過程被穩定標成錯的狀態」要擋多久 | `DEFAULT_DWELL`（或 CLI `--dwell`） | 候選新狀態要連續出現這麼多次才真的換過去；預設 6，這是 `VOTE_M`/`VOTE_N`/`REFRACTORY` 都沒有涵蓋的保護（見下方「三種 Prediction 來源」） |

改完直接重跑程式就會生效，不需要重新編譯/安裝。改完**去彈跳相關參數**
（`P_MIN`/`VOTE_M`/`VOTE_N`/`REFRACTORY`）建議先跑 `pytest tests/` 確認沒改壞
既有行為，再上機測試。

---

## 安裝

沿用專案既有的 `requirements.txt`：

```bash
pip install -r requirements.txt
```

| 套件 | 版本 | 控制層用途 |
|---|---|---|
| `pyserial` | ≥3.5 | `--source live` 讀 STM32 UART frame |
| `torch` | ≥2.0 | `--source replay` / `live` 底層跑分類模型（`semg_realtime_V6` 的 `semg.v2.infer.Recognizer`） |
| `numpy` | ≥1.24 | 訊號陣列運算 |

⚠️ **xArm SDK 是 vendored 進來的，不在 `requirements.txt` 裡**：
`run_arm_control.py` 會先嘗試 `import xarm`，找不到的話自動把
`xArm-Python-SDK-master/` 加進 `sys.path` 再 import 一次，不需要額外 `pip install`。

測試額外需要 `pytest`（僅開發環境需要，不影響正式執行）：

```bash
pip install pytest
```

---

## 三種 Prediction 來源

控制層本身**不做推論**，`Prediction`（`hand/elbow/shoulder` 三個狀態 + 三個信心值）
由 `src/control/sources.py` 的三種來源之一產生：

| 來源 | 檔案 | 需要模型？ | 需要硬體？ | 用途 |
|---|---|:---:|:---:|---|
| `mock` | `MockSource` | ❌ | ❌ | 腳本化假手勢，驗證狀態機邏輯 |
| `replay` | `ReplaySource` | ✅ | ❌ | 重播已錄 CSV 過真模型，驗證即時路徑 |
| `live` | `LiveSource` | ✅ | ✅ | 真串列 + 真模型，正式運作 |

`replay`/`live` 預設用的模型是 `semg_realtime_V6/models/final_v4_split.pt`
（見 `semg_realtime_V6/README.md`「三個模型」）：

| 模型 | 架構 | 一般情況 | 沒見過的組合 |
|---|---|---|---|
| **`final_v4_split`（預設）** | 分離編碼器 | 87.9% | **85.4%** |
| `final_v4_tiny` | 共享 backbone | 88.7% | 5.0% |
| `final_v4_zhao` | CNN+BiLSTM+Attn | — | 0.2% |

「沒見過的組合」欄就是先前實機踩到的問題（抬肘/抬肩時彎食指被誤判成
pinch）的直接量化——這是預設用 `split` 而不是準確率略高一點的 `tiny` 的原因。
`--model` 可以指到 `final_v4_tiny.pt`／`final_v4_zhao.pt`，但要跟
`semg_realtime_V6/src/semg/v2/config.py` 的 `CH_NAMES`／`DOF_STATES`／
`FEAT_VERSION` 是同一版才會對——**不要**混用 `src/semg/` 或舊版
`semg_realtime/`（V3）訓練出的模型，`Recognizer` 載入時會用 `feat_version`
擋掉明顯不相容的組合，但版本管理還是要靠自己別搞混。

### `--dwell`：擋「移動過程被穩定地標成錯的狀態」

`sources.py` 的 `ReplaySource`/`LiveSource` 讀的是 `Recognizer.push()` 回傳的
`stable`（votes+dwell 處理過），不是逐窗抖動的 `raw`。這跟控制層自己的三道
去彈跳閘（`arm_config.py` 的 `P_MIN`/`VOTE_M`/`VOTE_N`/`REFRACTORY`）分工不同：

| 保護層 | 對付什麼 | 為什麼另一層擋不住 |
|---|---|---|
| `RECOGNIZER_VOTES`（固定 1，等於關閉） | — | 跟控制層的 `VOTE_M`/`VOTE_N` 功能重疊，疊兩層只會疊加延遲 |
| **`--dwell`**（預設 6） | 移動過程中**穩定地**標成錯的狀態 | 這種誤判本身就很穩定，投票（不管在哪一層）都擋不住——`VOTE_M`/`VOTE_N` 看到的是「連續 4/6 幀都是同一個答案」，而穩定的誤判正好長這樣 |
| 控制層 `P_MIN`/`VOTE_M`/`VOTE_N` | 逐窗抖動、單幀誤判 | — |
| 控制層 `REFRACTORY` | 轉態頻率（防手勢誤判連環觸發） | — |

四層各司其職，不要為了「簡化」把 dwell 拿掉或跟 votes 混在一起。

---

## 使用流程

**啟動指令、CLI 參數總表、操作按鍵已整併到 `docs/control-commands.md`**
（四種模式——手動版／流程版／replay 版／肌電控制版——唯一的指令去處，
不在這裡重複維護一份）。這裡只留設計/參數調整相關的說明。

---

## 控制範式（規格書 §4，已由使用者確認，不要自行更動）

| DOF | 狀態 | 範式 | 動作 |
|---|---|---|---|
| hand | `pinch` | **邊緣觸發 toggle** | 每次 rest→pinch 的上升緣切換夾爪：閉 ↔ 開 |
| hand | `index` | **持續 jog** | 維持期間 J6 往 **+** 方向持續轉 |
| hand | `thumb` | **持續 jog** | 維持期間 J6 往 **−** 方向持續轉 |
| hand | `rest` | — | J6 停在當前角度，不回歸；**夾爪不動**（只由 pinch 上升緣改變） |
| elbow | `flex` / `rest` | **位置鏡像** | J3 → `ELBOW_FLEX_ANGLE`(90°) / `HOME_ANGLES[2]`(180°) |
| shoulder | `flex` / `rest` | **位置鏡像** | J2 → `SHOULDER_FLEX_ANGLE`(0°) / `HOME_ANGLES[1]`(−90°) |

⚠️ J6 是唯一有累積狀態的軸（會漂移），其餘軸都是無狀態的純函數映射，隨時可重算。

### 夾住的同時可以動哪些軸

`grip_state`（夾爪開/閉）是**獨立於 hand 當下讀值**的持續狀態，只在 pinch 的上升緣被
改變，換成別的手勢**不會**把它重置或重新觸發 toggle。所以：

1. 先做一次 `pinch` → 夾爪 CLOSED（夾住東西）
2. 接下來換成 `index` / `thumb` → J6 照常 jog，夾爪維持 CLOSED，**不會**被鬆開
3. `elbow` / `shoulder` 本來就是獨立的頭，任何時候都不受 `hand` 影響

也就是說：**夾住之後，J6 轉動 + elbow + shoulder 三者可以跟「夾著東西」同時成立**——
不需要一直維持 `pinch` 這個手勢，物理上的抓握是機械夾爪在做，你的手隨時可以換成
`index`/`thumb` 手勢來微調腕部角度。要放開時再做一次 `pinch`（rest→pinch 或
index/thumb→pinch 都算上升緣）即可切回 OPEN。這個行為在 `test_mapping.py` 有回歸測試
（`test_gripper_stays_closed_while_jogging_and_moving_other_joints`）。

---

## 參數設置（`src/control/arm_config.py`）

所有可調參數都集中在這個檔案，改參數只需要動這裡，程式碼本體沒有魔術數字。

### 手臂連線 / Home

| 參數 | 值 | 說明 |
|---|---|---|
| `ARM_IP` | `192.168.1.166` | 手臂控制箱實際控制 IP（2026-08-19 以實際連線成功確定，見下方說明） |
| `HOME_ANGLES` | `[90, -90, 180, 0, 0, 0]` | J1..J6 home 姿態（度）。J1 2026-08-17 由 0° 改成 90° |
| `HOME_SPEED` | `15.0 °/s` | 回 home 用的較慢速度。2026-08-17 由 30 調低 |
| `COLLISION_SENSITIVITY` | `3`（0~5） | 中等靈敏度 |

⚠️ **這個網段有三個長得很像的 IP，這裡的判斷唯一以「實際連線成功過」為準**：

| IP | 用途 |
|---|---|
| **`192.168.1.166`** | **控制指令要送的位址**——2026-08-19 用 `XArmAPI` 實際連線成功，讀到 `firmware=v2.7.1`、`sn=LI1006`、`axis=6`，是唯一有真實連線證據支持的值 |
| `192.168.1.181` | 連線會 TCP 502 埠 `connect socket failed`。是 xArm Studio 自己用的 IP，不是給 Python SDK 連的（`coffee.py` 裡寫的 `.181` 是誤用，不要照抄） |
| `192.168.1.180` | 2026-08-17 一度被判斷是對的（來自使用者對網路配置的說明），但**從未實際連線驗證過**，之後被 `.166` 的真實連線結果取代 |

`ARM_IP` 在 2026-08-17～19 之間改了三次（`.181`→`.180`→`.166`）。
**之後若又有人建議改成別的 IP，先問「有沒有實際連線成功過」再改**，
光憑推論或文件記載都不夠——這正是前兩次改錯的原因。

### 限位（送出前兩層都要 clamp：先 `SOFT_LIMITS` 再 `JOINT_LIMITS`）

| 軸 | `SOFT_LIMITS`（本專案） | `JOINT_LIMITS`（出廠） |
|---|---|---|
| J2（肩） | −90° ~ 0° | −150° ~ 150° |
| J3（肘） | 90° ~ 180° | −3.5° ~ 300° |
| J6（腕） | −180° ~ 180°　*可放寬到 ±270* | −360° ~ 360° |

### J6 jog / 指令節流

| 參數 | 值 | 說明 |
|---|---|---|
| `JOG_RATE_DEG_S` | `45.0 °/s` | J6 jog **目標角**推進速率（TODO 待確認，實機試了再調） |
| `CMD_PERIOD` | `0.10 s` | 送指令節流週期，也是 J6 jog 目標角的推進週期 |
| `CMD_DEADZONE_DEG` | `0.5°` | 目標角變化小於此值不送指令 |
| `CMDNUM_MAX` | `2` | `arm.cmd_num` 到此值就跳過這個 tick，防佇列堆積 |
| `JOINT_SPEED` / `JOINT_ACC` | `40 °/s` / `200 °/s²` | elbow/shoulder 位置鏡像用的關節速度/加速度（出廠上限 180 / 1145）——**J6 jog 不用這組**，見下方 |
| `JOG_SPEED_CAP` / `JOG_ACC` / `JOG_BLEND_RADIUS_DEG` | `60 °/s` / `300 °/s²` / `2.0°` | J6 jog **實際送指令**用的速度/加速度/轉角融合半徑，見下方說明。⚠️ `JOG_ACC` 2026-08-19 由 1100 降到 300 解決「整隻手臂抖動」，**不要調回去** |

#### ★ 為什麼 J6 jog 原本會「卡卡的、不等速」

`JOG_RATE_DEG_S` 只決定**目標角**每個 `CMD_PERIOD` 推進多少（`JOG_STEP_DEG =
JOG_RATE_DEG_S × CMD_PERIOD = 4.5°`），不是手臂實際的運動速度——那是靠
`set_servo_angle` 的 `speed`/`mvacc`/`radius` 決定的。原本這三個小步移動跟
elbow/shoulder 共用同一組保守的 `JOINT_SPEED`/`JOINT_ACC`，會有兩個問題疊加：

1. **加速度不夠，追不上目標速率**：用 `JOINT_ACC=200°/s²` 時，單一 4.5° 小步
   能達到的三角形速度曲線峰值只有 `sqrt(200×4.5)≈30°/s`，連
   `JOG_RATE_DEG_S=45°/s` 都追不到——每一步都還沒加速到位就要開始減速，
   不是穩定的等速巡航。
2. **每個 waypoint 之間完全停下來**：`set_servo_angle` 不帶 `radius`（或
   `radius<0`，未設定時的預設）時走 `MOVE_JOINT`，手臂會在每個 waypoint
   之間**完全減速到 0** 才開始下一段。要讓連續的小步真正銜接成平滑運動，
   必須帶正值 `radius`（走 `MOVE_JOINTB`，「關節融合運動」），讓控制器做
   轉角融合、不完全停下來——這需要韌體版本 > 1.5.20（本機 v2.7.1，符合）
   且 `wait=False`（`wait=True` 每次都強迫跑完整段，會讓融合完全失效）。

兩者要一起修才有效：只加大加速度沒有 `radius`，還是每 100ms 停一次；
只加 `radius` 沒有夠高的加速度，還是追不上目標速率一樣卡。

**修法**：`ArmController.tick()` 判斷這次要送的指令是不是「純 jog」
（elbow/shoulder 相對上次送出的值沒有變）——是的話用 `JOG_SPEED_CAP`/
`JOG_ACC`/`JOG_BLEND_RADIUS_DEG`；elbow/shoulder 真的在動（少見的一次性
大幅度姿勢轉態）時維持原本保守的 `JOINT_SPEED`/`JOINT_ACC`，且不帶
`radius`（一次性的大動作沒有下一步要銜接，不需要融合）。

#### 🔴 但上面整段的前提是 mode 0 —— 現在是 mode 6，`radius` 與高加速度都不再適用

上面「每個 waypoint 之間完全停下來」的分析只在 **mode 0** 成立。2026-08-18
改成 **mode 6**（線上軌跡規劃，新指令**中斷**舊指令）之後：

- `radius` 轉角融合**失效且有害**——mode 6 不排隊，沒有相鄰 waypoint 可融合，
  帶 `radius` 會讓 SDK 走 `move_jointb` 而被拒絕（**J6 完全不動**，實機踩過）。
  程式已改成只在 `ARM_MODE_RUN == 0` 時才帶 `radius`。
- 高 `JOG_ACC` 從「取回巡航段」變成**純粹的加速度衝擊**——每 0.1s 一次接近
  出廠上限的力矩脈衝，激振整條手臂。這就是 2026-08-19 使用者回報的
  「J6 在轉但整隻手臂在抖」，把 `JOG_ACC` 從 1100 降到 **300** 後解決。

⚠️ **教訓：改控制模式時，所有依賴「每步必停」假設推導出來的參數都要重新
檢視。** 完整經過見 `docs/on-machine-checklist.md` 的 M-18 專節。

### 去彈跳三道閘（規格書 §5）

| DOF | `P_MIN`（信心門檻） | `REFRACTORY`（不反應期） |
|---|---|---|
| hand | 0.80 | 0.60 s |
| elbow | 0.75 | 0.40 s |
| shoulder | 0.75 | 0.40 s |

`VOTE_M = 6`、`VOTE_N = 4`：最近 6 幀（300 ms）裡要出現 ≥4 次同一狀態才允許轉態。

### 其他

| 參數 | 值 | 說明 |
|---|---|---|
| `WATCHDOG_SEC` | `0.5 s` | 超過此時間沒收到 `Prediction` → 停止 + 進 `SAFE` |
| `GRIP_HOLD_MAX_SEC` | `30 s` | 夾爪連續閉合超過此秒數自動 `stop_lite6_gripper` 降溫（TODO 待確認） |

---

## 安全機制

1. **三道去彈跳閘**（信心門檻 → 多數決 → refractory）：任一單幀誤判不會直接變成手臂動作。
2. **看門狗**：`Prediction` 來源斷線 0.5 秒 → 立刻 `set_state(4)` 停止 + log ERROR + 進 `SAFE`，需人工按 `Enter` 重新 arm 才能恢復。
3. **DISARMED 預設**：程式啟動後不接受任何動作指令，必須手動按 `Enter`。
4. **雙層限位 clamp**：任何送出的角度都保證落在 `SOFT_LIMITS` 內（進而在出廠 `JOINT_LIMITS` 內）。
5. **`error_code` 不自動清除**：手臂回報錯誤（碰撞等）直接進 `SAFE`，不自動 `clean_error()` 重試——自動恢復會掩蓋碰撞事件。
6. **`Ctrl-C` / `q` 一定回 home 並斷線**，不留下 enable 狀態的手臂。

---

## Log 欄位（`--log` 指定的 CSV，預設存在 `results/control_logs/`）

```
t, hand, elbow, shoulder, p_hand, p_elbow, p_shoulder, J2, J3, J6, grip_state, cmd_sent
```

每個控制 tick 寫一列，供論文分析用（分類穩定度、指令發送頻率、夾爪切換次數…）。

---

## 測試

```bash
cd moving_pose
pytest tests/          # 不需要另外設定 PYTHONPATH，conftest.py 已處理
```

涵蓋範圍：

- `test_debounce.py`：單幀雜訊不觸發、低信心幀不推進投票、refractory 期間拒絕轉態
- `test_mapping.py`：elbow/shoulder 位置鏡像映射、連續 5 次 pinch 上升緣的夾爪序列
  （`CLOSE, OPEN, CLOSE, OPEN, CLOSE`）、隨機序列下限位 clamp 恆成立、長時間 jog 不超出軟限位

---

## 已知限制 / 待確認事項

以下項目按規格書指示用建議預設值實作，**不需要停下來等使用者確認**，
但實機測試後可能需要調整（見 `arm_config.py` 裡的 `TODO` 註解）：

| 項目 | 目前預設 | 之後怎麼調 |
|---|---|---|
| `JOG_RATE_DEG_S` | `45°/s` | 實機試 J6 jog 手感後調整 |
| `SOFT_LIMITS[6]` | `±180°` | 若 jog 範圍不夠，可放寬到 `±270°`（仍在出廠限位內） |
| `GRIP_HOLD_MAX_SEC` | `30 s` | 依實際夾持需求調整 |

🔴 **與本模組無關的 BLOCKER**：`config.py` 裡 ch2/ch4 電極貼法的衝突（規格書 §1.5）
不影響控制層邏輯（控制層不使用通道名稱），未在本次實作中處理。
