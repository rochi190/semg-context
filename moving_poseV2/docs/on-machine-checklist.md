# 上機注意事項 —— UFACTORY Lite 6 整合清單

> **這是「上機準備/操作」類資訊的固定去處。** 之後有類似的注意事項、
> 待決參數、上機流程異動，都整合進這一份文件，不要開新檔案分散。
>
> 本文取代並整合了兩份外部文件：`CLAUDE_CODE_arm_control_fixes.md`
> （程式碼修改單）與 `ON_MACHINE_CHECKLIST.md`（原始上機清單）。
> 修改單裡的 P0/P1 修正已於 **2026-08-19** 套用完成，`pytest tests/`
> **35 個測試全綠**——這道門檻已經跨過，可以往下進行。
>
> ⚠️ 那兩份原始文件裡有些「已確認的事實」後來被實機測試推翻了
> （最明顯的是手臂 IP），本文件以**這裡**寫的為準，不要回頭參考原始文件。
>
> ⚠️ **2026-08-19 架構變更**：肌電路徑（讀檔版與肌電操控版）已整併進
> `semg_realtime_V4/run_live.py`，加 `--arm-ip`／`--arm-dry-run` 即驅動手臂，
> 並改成**單層投票**。本文件下面 STEP 3／STEP 4 的
> `run_arm_control.py --source replay/live` 指令仍可運作但**不再是主路徑**——
> 指令與最新的 replay／live 驗證流程一律以 **`docs/control-commands.md`** 為準。
> 本文件保留的價值是上機前準備、待決事項總表與量測記錄。

---

## 已確認的事實（不需要再查）

| 項目 | 值 | 來源 |
|---|---|---|
| 手臂控制 IP | **`192.168.1.166`** | 2026-08-19 **實際連線成功**確認（讀到 firmware/sn/axis）。`arm_config.ARM_IP` 已是這個值 |
| 手臂 IP（容易搞混，不要用） | `192.168.1.181` = xArm Studio 自己連線用、連線會失敗；`192.168.1.180` = 一度被誤判是對的但從未實際連線驗證過 | 只有 `.166` 有真實連線成功的證據；改 `ARM_IP` 前一定要先實際連過，不要只憑推論 |
| STM32 序列埠 | **`COM10`** | 2026-08-13 確認。範例指令若寫 `COM9`，那是舊的 |
| Baud | `1,600,000` | `--baud` 預設值正確 |
| 控制模式 | mode 0（位置控制）全程 | mode 是全域的，位置鏡像與 J6 jog 無法用不同 mode |
| 取樣率 | 2000 SPS，`CONFIG1=0x93` | 以 `semg_realtime_V4/src/semg/v2/config.py` 為準 |
| Gain | 8（`CHnSET=0x40`），1 LSB ≈ 67.06 nV | 舊文件寫 Gain=4 / 134 nV 是錯的 |
| 分類器套件位置 | **`semg_realtime_V4/`**，跟 `moving_pose/` 同層（repo 根目錄），不在它底下 | 2026-08-19 從 `moving_pose/semg_realtime/`（V3）遷移，見 `arm_config.py` 開頭說明 |
| 預設模型 | `semg_realtime_V4/models/final_v4_split.pt` | split 架構，「沒 cue 過的組合」準確率 5.0%→85.4%（直接解決抬肘同時彎食指被誤判成 pinch 的問題） |
| 電極貼法 | ch1=latissimus_dorsi, **ch2=biceps**, ch3=triceps, **ch4=deltoid_anterior**, ch5=wrist_flexor, ch6=index, ch7=thumb | ch2/ch4 衝突已解決：`final_v4_split.pt` 是用**這個**（正確）貼法訓練的，不需要再重訓 |
| Home 姿態 | `[+90, -90, +180, 0, 0, 0]`（J1..J6，度） | J1 於 2026-08-17 由 0° 調成 90°（使用者要求） |
| 回 home 速度 | `15.0 °/s` | 2026-08-17 由 30 調低（使用者要求） |
| ⚠️ 手臂靜止姿態 | **全零** `[0,0,0,0,0,0]`，**不是** home | 2026-08-17 讀出。所以 `startup()` 首次回 home 是 J1 0→90、J2 0→−90、**J3 0→180**，@15°/s 下 J3 那段約 **12 秒**——這是本專案至今最大範圍的一次自動運動，第一次跑務必手放急停 |

---

## 🔴 STEP 0：接上手臂之前必須先做

### 0-1 夾爪必須先實體安裝

`ArmController.startup()` 與 `engage()`（重新 ARM 前也會回 home）都會呼叫
`set_servo_angle`；`startup()` 之後**無條件**呼叫 `open_lite6_gripper(sync=False)`。
沒有裝夾爪就跑這行，行為未定義（Lite 6 夾爪走 tool GPIO，沒有回授可判斷是否存在）。

- [ ] 夾爪已實體裝上並接好線
- [ ] 用 UFACTORY Studio 手動開合一次，確認夾爪本身正常

### 0-2 確認急停與工作空間

第一次接手臂會執行大範圍運動（回 home：J2 到 −90°、J3 到 180°；J1 現在也會
轉到 90°，跟舊版行為不同，工作空間要重新確認一次）。

- [ ] 急停按鈕在手邊、確認按下有效
- [ ] 手臂工作空間內清空（J1 轉到 90°、J2 從 −90° 掃到 0° 的路徑都要淨空）
- [ ] 手臂底座已鎖緊、桌面固定

### 0-3 模型狀態 —— ✅ 已完成，不需要再做

舊版文件這裡列的是「等 ch2/ch4 修正後的模型重訓」，**現在已經做完**：
`semg_realtime_V4/models/final_v4_split.pt` 就是用修正後的電極標籤、
split 架構訓練出來的，`--source replay`/`live` 的預設值已經指向它。
不需要額外動作。

> 控制層本來就不使用通道名稱，所以就算模型還沒備妥，STEP 1、STEP 2
> 也不受影響，可以先做。

### 0-4 確認 CPU 資源

`CPU_THREADS = 2`，而 `--source live` 會同時跑：reader thread（串列）+
主 thread（推論 + 控制）+ xArm SDK 的 report thread。

- [ ] 錄影前關掉瀏覽器、Teams、其他吃 CPU 的程式
- [ ] reader thread 的 busy-loop 問題已修好（找不到 SYNC 時會間歇 sleep），
      不會再無條件燒滿一個核心

---

## STEP 1：`--dry-run` 離線驗證（不接手臂）

```bash
cd moving_pose
python src/control/run_arm_control.py --source mock --dry-run
```

**這一步在接手臂之前就要跑通。** 大部分的迴歸已經有自動化測試覆蓋
（`pytest tests/`，35 個），這裡只列**跑實際 CLI** 才能驗的項目：

- [ ] 程式正常啟動，顯示「已回 home，DISARMED」
- [ ] 按 Enter → 印出「已 ARMED」；因為 `engage()` 現在**一律**會先實體回一次
      home 才進 ARMED，觀察會看到 **兩次** `set_servo_angle` 到 home 角度
      `[+90.0, -90.0, +180.0, +0.0, +0.0, +0.0]`（一次在 `startup()`、一次在
      `engage()`）——這是設計行為，不是 bug
- [ ] 連續觀察 5 次 pinch 上升緣，夾爪序列為
      `CLOSED → OPEN → CLOSED → OPEN → CLOSED`
- [ ] 夾住東西的同時換成 index/thumb，夾爪維持 CLOSED、J6 同時 jog、
      elbow/shoulder 同時可動（見 `docs/arm-control-usage.md` 的說明與回歸測試）
- [ ] 按 `z`，J6 立刻歸零
- [ ] 按 q，印出「已回 home、斷線。」（若曾經模擬 error_code，應改印
      「已斷線（未回 home）。」——見下方 1-1）
- [ ] 結束時印出 `[info] 主迴圈實測平均 xx Hz` → **記下這個數字**（見 M-16）

### 1-1 已被自動化測試取代的迴歸項目

以下項目過去需要手動暫改參數、觀察終端機輸出才能驗證，現在已經寫成
`pytest` 測試、每次跑 `pytest tests/` 就會自動檢查，**不需要再手動做**：

| 項目 | 對應測試 |
|---|---|
| SAFE 恢復時手臂真的被拉回運動狀態 | `test_engage_from_safe_recovers_arm_state` |
| error_code 未清除時拒絕重新 ARM | `test_engage_refuses_from_safe_with_uncleared_error` |
| DISARMED 期間漂移的 target 在 ARM 前被拉回 home | `test_engage_resets_drifted_target_to_home_before_arming` |
| 帶著 error_code 結束時誠實回報「未回 home」 | `test_shutdown_does_not_claim_homed_with_uncleared_error` |
| 連續 5 次 pinch → 正確的開合序列 | `test_pinch_edge_toggle_sequence` |
| 夾住同時操控其他軸 | `test_gripper_stays_closed_while_jogging_and_moving_other_joints` |
| 轉入 rest 不被 refractory 卡住 | `test_rest_entry_bypasses_refractory` |
| 投票視窗有時間上限，不會拿數秒前的舊幀轉態 | `test_stale_frames_do_not_trigger_transition` |
| `MockSource.is_ready()` 在跑完後轉 False（來源把關） | `test_mock_no_loop_becomes_not_ready` |

### 1-2 錯誤路徑（CLI 層級，測試沒覆蓋到）

- [ ] `--source replay --csv /nonexistent.csv --dry-run`
      → 報錯退出，**且完全沒有印出任何 `[MockArm]` 指令**
      → 證明 source 建構在手臂初始化之前
- [ ] `--source mock --arm-ip 10.99.99.99`（故意打錯，不加 `--dry-run`）
      → 5 秒後 `✖ 無法連線手臂 10.99.99.99` 並列出檢查清單
      → **這一項一定要測**：打錯 IP 的症狀（畫面全綠、手臂不動）
        跟其他連線問題長得一模一樣，沒有這道檢查你分不出來

---

## STEP 2：實體手臂

★ **順序很重要：先跑 2-0 的唯讀腳本，確認過限位之後才跑 2-1 以後會讓手臂
動的控制程式。** 兩個理由：
1. `read_arm_specs.py` 完全不會讓手臂動（只讀屬性），拿它當第一次連線比
   用「會執行大範圍回 home」的控制程式安全得多，同時也驗證了 IP 真的通。
2. 2-0 讀到的 `reduced_joint_limits` 可能會讓你需要調 `SOFT_LIMITS`——那
   應該在手臂動起來**之前**決定，不是動完才發現限位設錯。

### 2-0 🔴 讀出手臂規格（唯讀，不會讓手臂動）

```bash
cd moving_pose
python xArm-Python-SDK-master/read_arm_specs.py
```

腳本本體見 `xArm-Python-SDK-master/read_arm_specs.py`（已修正兩個坑：
沒有 `arm.joint_limits` 這個屬性、以及建構子本身就會 raise 所以不能寫成
「先建構再 assert connected」）。

- [ ] 整段輸出貼進文末「量測記錄」區保存
- [ ] `error_code` / `warn_code` **應為 0**——非 0 先用 UFACTORY Studio 復歸，
      **不要往下走**（帶著 error_code 的話 `startup()` 會拒絕動作，而你會
      以為是控制層壞了）
- [ ] `state` / `mode` 記錄下來當基準
- [ ] `joint_speed_limit` / `joint_acc_limit` → 對照 M-10（見 2-2）
- [ ] `reduced_joint_limits` → 對照 M-2（見 2-1）

### 2-1 🔴 M-2：對照並修正 `JOINT_LIMITS`（必做）

`arm_config.JOINT_LIMITS` 目前的值來源不明確，是 `clamp_angles()` 的第二道
防線——`SOFT_LIMITS` 現在完全落在它裡面，所以第二道防線**永遠不會啟動**，
寫錯也看不出來。

用 2-0 的 `read_arm_specs.py` 讀出來（腳本本體在磁碟上，這裡不再複製一份
——複製會走樣，之前就發生過文件裡的版本還帶著已修掉的 bug）。

- [ ] 把讀到的 `reduced_joint_limits` 對照 `arm_config.JOINT_LIMITS`，數字明顯
      不合理（例如某軸範圍為 0，代表精簡模式沒設定過）就**不要**拿來覆蓋
- [ ] 若要覆蓋，把註解改成
      `# 由 arm.reduced_joint_limits 讀出，YYYY-MM-DD，firmware x.x.x`
- [ ] 確認 `arm_config.validate()` 仍然通過（`run_arm_control.py` 啟動時會自動
      呼叫）。**若 `SOFT_LIMITS` 超出新讀到的限位，要調 `SOFT_LIMITS`，
      不是刪掉檢查**

### 2-2 M-10：確認速度/加速度上限

- [ ] 用 2-0 讀到的 `joint_speed_limit` / `joint_acc_limit` 對照
- [ ] 更新 `arm_config.py` 的註解，標上讀出日期
- [ ] 確認 `JOINT_SPEED = 40.0` / `JOINT_ACC = 200.0` 確實在上限內
- [ ] ⚠️ 也要確認 **jog 專用**的 `JOG_SPEED_CAP = 60.0` / `JOG_ACC = 300.0`
      在上限內。這組是 J6 jog 專用（不與 elbow/shoulder 共用）；`JOG_ACC`
      已於 2026-08-19 由 1100 降到 300 解決「整隻手臂抖動」，**不要調回去**
      （見 M-18）

---

### 🔴 2-3 以後開始會讓手臂動——第一次請用這個指令

```bash
python src/control/run_arm_control.py --source mock --arm-ip 192.168.1.166
```

★ **這是第一次讓實機動起來時唯一該用的指令。** 不要跳到 `--source live`：
`mock` 的手勢是腳本化的（輸入已知且完美），手臂不照腳本動就一定是控制層或
手臂的問題，與訊號無關；直接上真訊號的話，手臂不動你分不清是分類錯、串列
掉包、還是手臂連線問題。

⚠️ **手放在急停上，等 `startup()` 的回 home 走完再放開。**
J1→90°、J2→−90°、J3→180°，@`HOME_SPEED=15 °/s`，其中 J3 那段約 12 秒，
是本專案至今最大範圍的一次自動運動。

以下 2-3 ~ 2-9 可以在同一輪裡一起觀察，不必每項重跑一次。

### 2-3 M-13：位置鏡像方向驗證

`MockSource` 的腳本會依序跑過 index / thumb / pinch / elbow flex / shoulder flex。

- [ ] `elbow=flex` 時 J3 走到 90°，**視覺上是「屈肘」**（不是伸直）
- [ ] `elbow=rest` 時 J3 回到 180°，視覺上是「伸直」
- [ ] `shoulder=flex` 時 J2 走到 0°，**視覺上是「平舉水平」**
- [ ] `shoulder=rest` 時 J2 回到 −90°，視覺上是「下垂」
- [ ] 回 home 時 J1 轉到 90°——確認這個角度在物理上合理（沒有跟桌子/其他
      設備打架），不合理的話這是要跟 2-1 一起調整的項目

⚠️ 若方向反了，**改 `arm_config.ELBOW_FLEX_ANGLE` / `SHOULDER_FLEX_ANGLE`
與對應的 `SOFT_LIMITS`**，不要改 `arm_controller.py` 的映射邏輯。
改完 `validate()` 會幫你檢查新值是否還在限位內。

### 2-4 M-3：J6 jog 手感 ＋ 🔴 死區陷阱 ＋ 平滑度

`JOG_RATE_DEG_S = 45.0` 標著 TODO 待確認，是猜的值。

- [ ] `hand=index` 時 J6 往 **+** 方向轉；`hand=thumb` 往 **−** 方向轉
      （若相反，交換 `arm_controller.on_prediction()` 裡 `jog_dir` 的 ±1）
- [ ] 感受速度：太快 → 手勢一抖就轉過頭；太慢 → 影片裡像蠕動
- [ ] 撞到 ±180° 時停住、不報錯、`cmdnum` 欄位對應的送出動作變 0
- [ ] 放鬆手勢後 J6 應該**立刻**停（2026-08-17 已修正 refractory 不再卡住
      rest 轉態），不該再看到「鬆手後還繼續轉一小段」
      ⚠️ **這項要用 `--source mock` 驗，不要用手動版**：`MockSource` 的腳本
      會自己從 index/thumb 切回 rest，走的是 `DofDebouncer` 的完整轉態路徑
      （投票 + refractory 豁免），跟真實使用情境一致；手動版是你自己打
      `0 0 0`，繞過了「模型判定放鬆」這一段，驗不到真正要驗的東西
- [x] ~~轉動時應該是平滑的等速旋轉~~ → 見下方 M-18 專節，**已結案為「接受現狀」**

#### ✅ M-18：J6 抖動 —— 2026-08-19 已解決（`JOG_ACC` 1100 → **300**）

> 🔴 **這一節在 2026-08-19 整個翻案了。** 下面先寫結論，舊的推導保留在後面
> 當作教訓——**不要照舊結論把 `JOG_ACC` 調回 1100**。

**症狀**：`--source mock` 跑到 index / thumb 時，J6 在轉，但**抖的是整隻手臂**
（不是 J6 自己頓挫，兩者是不同現象，別混為一談）。

**根因**：`JOG_ACC = 1100` 是 **mode 0 時代的遺留值，前提在 mode 6 下已經不成立**。

| | mode 0（2026-08-17，定 1100 時） | mode 6（2026-08-18 起，現行） |
|---|---|---|
| 指令語意 | 排隊，手臂必須能**停在每個 waypoint** | **新指令中斷舊指令**並重新規劃 |
| 每小步的運動 | 獨立的加速—巡航—減速曲線 | 連續追一個移動中的目標，不會停 |
| 高 `JOG_ACC` 的作用 | 取回巡航段佔比，**有意義** | 只剩每 0.1s 一次 96% 上限的加速度衝擊 → **力矩脈衝激振整條手臂** |

**修法**：`JOG_ACC` 1100 → **300 °/s²**（出廠上限 1145 的 26%）。
mode 6 下 J6 只需平順跟上 45 °/s 的目標推進速率，0→45 °/s 只要 0.15 s。
**2026-08-19 實機確認：不再抖動。**

代價：放開手勢時多滑約 3.4°（45²/(2×300)），遠小於下面那個被否決的
lookahead 方案的 9~18°。若實機發現 J6 跟不上目標角（持續落後、放手後還在追）
才需要往上調，先試 500。

<details>
<summary>舊的（已被推翻的）2026-08-17 推導 —— 保留以免有人重走</summary>

| 嘗試 | 當時結果 |
|---|---|
| `JOG_ACC` 200 → 800 | ✅ 修掉「忽快忽慢」（加速度追不上目標速率） |
| `JOG_ACC` 800 → **1100** | 🟡 取回約 27% 巡航段（800 時單步峰值 √(800×4.5)=60.0 恰好等於 `JOG_SPEED_CAP`，巡航佔比 0%）。單步 0.150s → 0.130s |
| `radius` 轉角融合 | ❌ 幫不上忙——連續的 J6 小步在關節空間**共線**，沒有轉角可融合 |

當時的結論是「殘留的規律卡頓（約 6.7 Hz）是結構性的：`CMDNUM_MAX = 2` 使
控制器隨時只有一個待執行點，必須保證能停在該點 → 每步都得減速到底」，
並判定**接受現狀、不再投入**。

⚠️ **那整套推導的前提是 mode 0**（每步必停）。隔天 2026-08-18 改成 mode 6
之後前提就消失了，但這一節沒有跟著更新——結果 1100 這個為 mode 0 調出來的
極端值留在 mode 6 底下，變成抖動的來源。**教訓：改控制模式時，所有依賴
「每步必停」假設推導出來的參數都要重新檢視。**

</details>

⚠️ **平滑度與「放手即停」的衝突仍然成立**：加深 lookahead 能消除卡頓，但
放鬆手勢後 J6 會多轉 9~18°——M-11 的「放手即停」已驗證通過，不能拿去換。

`vc_set_joint_velocity`（mode 4，關節速度控制）仍然不可行：mode 是全域的，
而 elbow/shoulder 的位置鏡像需要位置語意。**但 mode 6 已經拿到大部分好處**
（位置語意 + 不必停在 waypoint），不需要再往 mode 4 走。

#### 🔴 調整 `JOG_RATE_DEG_S` 的有效區間：**20 ~ 60 °/s**

⚠️ **本節在 2026-08-17 整個重算過。** 舊版寫「下限 5 °/s、建議不要低於
8 °/s」，那是在 `JOG_BLEND_RADIUS_DEG` 加入**之前**算的，已經失效——
照舊版調到 8 °/s 會直接 import 失敗。

兩條約束共同決定下限，取較緊者：

```
JOG_STEP_DEG = JOG_RATE_DEG_S × CMD_PERIOD

(e) JOG_STEP_DEG > CMD_DEADZONE_DEG(0.5°)     → JOG_RATE >  5 °/s
(g) JOG_STEP_DEG > JOG_BLEND_RADIUS_DEG(2.0°) → JOG_RATE > 20 °/s  ← 較緊，實際下限
上限：JOG_RATE_DEG_S ≤ JOG_SPEED_CAP(60)      → 否則手臂追不上目標角推進
```

低於下限時，J6 的目標角雖然還在累加，但每個 tick 的變化量會被死區擋掉／
轉角融合半徑被 SDK 拒絕，**J6 完全不動或指令被拒**。
`validate()` 的 `(e)`/`(g)`/`(i)` 三條斷言已經把這個區間鎖住，
改壞了 import 就會直接失敗，錯誤訊息會直接印出當下的有效區間。

- [ ] 若要調慢，**不要低於 20 °/s**
- [ ] 真的需要更慢的話，必須**同時**降低 `JOG_BLEND_RADIUS_DEG`
      （但那會削弱它要解決的「卡卡的」問題）與 `CMD_DEADZONE_DEG`，
      而不是繞過檢查

### 2-5 M-4：`SOFT_LIMITS[6]` 是否需要放寬

- [ ] 實際 jog 時 ±180° 的範圍夠不夠？
- [ ] 不夠的話可放寬到 ±270°（仍在出廠 ±360° 內），但要先完成 2-1 確認限位

### 2-6 M-11：佇列堆積測試（`MockArm` 測不到的盲區）

`MockArm.cmd_num` 的模擬永遠是 1，節流條件 3（`cmd_num < CMDNUM_MAX = 2`）
在 dry-run 下永遠成立，**佇列堆積只能在實機上驗證**。

- [ ] 觀察終端機的 `cmdnum` 欄位，正常運作時應在 0~1 之間跳動
- [ ] **關鍵測試**：讓 `MockSource` 跑到 `elbow=flex`（J3 走 90°，約 2.3 秒）
      期間，觀察腳本切回 `rest` 之後手臂多久停止。
      **若手臂在腳本已切回 rest 之後仍繼續動 > 1 秒，就是佇列堆積。**
- [ ] 若 `cmdnum` 持續達到 2 並卡住：提高 `JOINT_SPEED` 或提高 `CMD_PERIOD`，
      **不要**提高 `CMDNUM_MAX`（那是把問題藏起來）

### 2-7 M-5：夾爪過熱斷電行為（⚠️ 會掉東西）

`GRIP_HOLD_MAX_SEC = 30.0`（TODO 待確認）：夾爪連續閉合超過 30 秒 →
`stop_lite6_gripper` → 斷電、夾持力消失 → **夾著的物體會掉**。

⚠️ **這項要用手動版驗，不能用 `--source mock`**：mock 腳本的 pinch 段只有
1.5~2.5 秒，維持不到 30 秒，永遠觸發不了斷電。改用：

```bash
python src/control/run_manual_control.py --arm-ip 192.168.1.166
# 進 ARMED 後選 `3 0 0`（pinch → 夾爪閉合），然後就放著不動，等 30 秒
```

- [ ] 實測夾爪連續閉合 30 秒後是否真的明顯發熱（決定這個值合不合理）
- [ ] 評估延長到 60 s，錄影腳本若有「夾住物體」的橋段先估算時間
- [ ] 斷電後 `grip_state` 仍是 `"CLOSED"`，下一次 pinch 上升緣會呼叫
      `open_lite6_gripper()`，語意上正確不會出錯

### 2-8 M-6：碰撞靈敏度

`COLLISION_SENSITIVITY = 3`（0~5，中等）。

- [ ] 用手輕推手臂，確認會觸發 `error_code` → 進 SAFE
- [ ] 確認進 SAFE 之後**按 Enter 會被拒絕**（因為 `error_code != 0`），
      log 印出「WARN 手臂 error_code=... 尚未清除，拒絕重新 ARM」
- [ ] 太靈敏（正常運動就誤觸發）→ 降到 2；太鈍 → 升到 4

### 2-9 結束流程

- [ ] 按 `q`，手臂停止 → 回 home → 斷線，log 印「已回 home、斷線。」
- [ ] Ctrl-C 也一樣
- [ ] **在有 `error_code` 的狀態下按 `q`**（先用手推觸發碰撞）：應印出
      「WARN 手臂 error_code=...，無法自動回 home」+「已斷線（未回 home）」，
      手臂停在原地，需人工用 UFACTORY Studio 復歸
- [ ] 任何情況下結束後，手臂都**不應該**還處於 motion_enable 狀態
      （用 UFACTORY Studio 確認）

---

## STEP 3：`--source replay` 校準去彈跳門檻

```bash
python src/control/run_arm_control.py --source replay \
    --csv data/raw/multi/gary/multi_gary_xxx.csv --arm-ip 192.168.1.166
```

### 3-1 🔴 M-7：`P_MIN` 必須用實際 confidence 分佈校準

```python
P_MIN = {"hand": 0.80, "elbow": 0.75, "shoulder": 0.75}
```

這些值沒有任何模型 confidence 分佈支撐。**風險**：4 類的 `hand` head 若
softmax 最大值長期落在 0.5~0.7，所有 hand 幀會被閘一擋掉，夾爪一次都不會
動——而終端機只會顯示 `hand=rest`，看不出是被門檻擋的。**這是整條鏈上最
容易出事、又最容易被誤診成「分類器完全失效」的一項。**

Log（`results/control_logs/control_*.csv`）記錄的 `p_hand`/`p_elbow`/`p_shoulder`
是**原始**信心值（`on_prediction` 存的是 `pred.p_*`，不是去彈跳後的值）：

```python
# tools/plot_conf_hist.py
import sys
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(sys.argv[1])
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, cur in zip(axes, ["p_hand", "p_elbow", "p_shoulder"], [0.80, 0.75, 0.75]):
    ax.hist(df[col], bins=50, range=(0, 1))
    ax.axvline(cur, color="r", ls="--", label=f"P_MIN={cur}")
    passed = (df[col] >= cur).mean() * 100
    ax.set_title(f"{col}  通過率 {passed:.1f}%  中位數 {df[col].median():.2f}")
    ax.set_xlabel("softmax max")
    ax.legend()
plt.tight_layout()
plt.show()
```

- [ ] 跑一次 replay，畫出三個 head 的 confidence 直方圖，**記錄通過率與中位數**
- [ ] 判斷準則：通過率 <50% → 門檻太高會頻繁凍結，**降 `P_MIN`**；
      50~85% → 合理，保持；>95% → 門檻太低形同沒有閘一，升 `P_MIN`
- [ ] 定案後把 `arm_config.P_MIN` 的註解改成
      `# 由 <受試者>_<日期> replay log 的 confidence 分佈定出，通過率 xx%`

### 3-2 M-8：檢查 `VOTE_WINDOW_SEC` 是否造成凍結

2026-08-19 新增的 `VOTE_WINDOW_SEC`（投票視窗時間上限）代價是：若超過一半
的幀被 `P_MIN` 擋掉，該 DOF 會凍結在最後的穩定狀態不再轉態。

- [ ] 觀察 log 裡 `hand`/`elbow`/`shoulder` 欄位是否有長時間不變、
      但 `p_*` 明顯在波動的區段 → 那就是凍結
- [ ] **若常凍結，正確處置順序是：先降 `P_MIN`（3-1），不要先放寬
      `VOTE_WINDOW_SEC`**
- [ ] 若 `P_MIN` 已降到 0.5 仍常凍結 → 問題在模型不在控制層，回去看訓練

### 3-3 M-9：量測端到端延遲（📹 錄影與論文用）

目前的延遲構成（2026-08-19 架構更新後，理論估計，**必須實測**）：

| 環節 | 估計 | 備註 |
|---|---|---|
| 模型證據累積 | ~0.3–0.5 s | 視窗 0.5 s + 序列跨度 0.95 s |
| `Recognizer` dwell | ~0.3 s | `RECOGNIZER_VOTES=1`（不重複計）+ `dwell=6` |
| `DofDebouncer` 投票 | ~0.2 s | `VOTE_N=4` 幀最少 0.20 s；⚠️ 只在**進入非 rest 狀態**時計入，回到 rest 已不受 refractory 拖慢，但仍要過這道投票 |
| 指令節流 | ≤0.10 s | `CMD_PERIOD` |
| **指令發出前小計** | **≈0.8–1.1 s** | 比 2026-08-13 估的 1.0–1.5 s 略低，但**沒有實測過**，別直接引用 |
| 手臂運動 | J3 走 90° 需約 2.25 s | `JOINT_SPEED=40 °/s` |

- [ ] 用 log 的 `t` 與 `cmd_sent` 欄位量出實際值，**記錄下來（論文用）**
- [ ] 若影片上延遲明顯：先確認 `RECOGNIZER_VOTES=1` 確實生效（不是不小心
      改回預設 5，造成雙重投票疊加）
- [ ] 次選：`VOTE_N` 從 4 降到 3（省約 0.05s）。⚠️ 平手時 `max()` 偏好索引小的
      rest，這個隱含偏好寫在 `debounce.py` 的註解裡
- [ ] **不要**用提高 `JOINT_SPEED` 來「看起來變快」，那只縮短運動時間，
      不縮短反應延遲，而且提高碰撞風險

### 3-4 M-12：replay 結束後的行為確認

- [ ] 重播結束時應印出「[replay] 重播結束。接下來 watchdog 會在 0.5s 後...」
- [ ] 之後進 SAFE 是**正常的**，不是錯誤

---

## STEP 4：`--source live` 正式運作

```bash
python src/control/run_arm_control.py --source live --port COM10 \
    --baud 1600000 --arm-ip 192.168.1.166
```

⚠️ **`--port` 是 `COM10`**。`--model`/`--dwell` 不給就用預設值
（`final_v4_split.pt`、`dwell=6`）。

### 4-1 校準期間不要按 Enter

前 5 秒（`--calib-sec`）在估計穩健尺度，`poll()` 回傳空 list；按 Enter
會顯示「[拒絕 ARM] 來源尚未就緒：校準中 還需 x.xs」。

- [ ] 確認終端機右側顯示校準倒數
- [ ] 確認校準完成後印出「[live] 校準完成（xxxx 樣本）」
- [ ] **校準期間請保持中性放鬆、手臂懸空**——這段是基線估計，動作會污染尺度
- [ ] 若用 `--yes`，程式會自動等到就緒才 ARM，確認這個行為正確

### 4-2 錄製條件（沿用既有協定）

- [ ] 筆電**使用電池**，不接充電器（充電器引入共模雜訊）
- [ ] STM32 供電 3.3 V + 5 V 給 EVM
- [ ] BIAS 電極貼在肘關節（olecranon）
- [ ] EVM jumper：J6 全拔、JP1=1-2、JP6 **不裝**、JP7/JP8=1-2、JP22=2-3
- [ ] 電極貼法：ch1=latissimus_dorsi, ch2=biceps, ch3=triceps,
      ch4=deltoid_anterior, ch5=wrist_flexor, ch6=index, ch7=thumb
      （已跟訓練 `final_v4_split.pt` 用的貼法一致，不需要再確認衝突）

### 4-3 🔴 維持姿勢時手臂必須懸空

程式啟動時會印這段警告，但錄影時很容易忘：手肘/前臂/上臂**都不能靠在
桌面、扶手或身側**。否則重力被外物撐住、EMG 消失 → 被判成 `rest` →
機械手臂突然回 home。

- [ ] 錄影前確認受試者坐姿，手臂全程懸空
- [ ] 注意疲勞：懸空維持姿勢很累，sEMG 頻譜隨疲勞左移 → confidence 下降
      → 可能觸發 3-2 的凍結。**單次錄影不要太長**

### 4-4 M-15：掉包率量測

每 5 秒的掉包告警（wall-clock 對照法）已內建。

> ⚠️ 這是次佳實作。`read_one_frame` 不回傳 frame 內的 `sample_index`，
> 無法做完整的索引連續性檢查（要改 `record_session.py`，超出控制層範圍）。

- [ ] 觀察是否出現「[live] ⚠ 掉包率約 x.xx%」
- [ ] **記錄穩定狀態下的掉包率**（論文用）
- [ ] 掉包率 > 1% 的處置方向：關掉其他吃 CPU 的程式、換 USB port/線、
      確認 `set_buffer_size(rx_size=262144)` 有生效（非 Windows 無此方法）
- [ ] 掉包會靜默破壞訊號連續性 → 濾波與特徵算錯 → confidence 下降。
      **現象與「模型訓練得不好」完全無法區分**，所以必須量

### 4-5 M-14：CPU 佔用觀察

- [ ] 用工作管理員觀察 python 行程的 CPU 佔用
- [ ] 若某一核持續 100%，檢查是不是 reader thread 在 busy loop
      （baud 設錯、COM port 錯 → `find_sync` 一直失敗）；已加 sleep 緩解，
      但根因仍要排除

### 4-6 M-16：主迴圈實測頻率

程式結束時會印出實測 Hz。**預期**：文件宣稱 100 Hz，但 Windows 的
`time.sleep(0.01)` 粒度約 15.6 ms，實際可能只有 **40–60 Hz**。

- [ ] 記錄 dry-run 的實測值
- [ ] 記錄 live 的實測值（會更低，因為多了推論）
- [ ] 功能上安全（所有時間判定都用 `time.monotonic()`，不依賴 tick 數），
      但**不要在論文或文件裡寫「100 Hz 控制迴圈」**
- [ ] 若實測 < 30 Hz，`MAX_PREDS_PER_TICK = 4` 可能太小（會頻繁丟幀），
      觀察是否出現「[warn] 單 tick 收到 x 筆 Prediction」

### 4-7 串列中斷的行為確認

- [ ] 運作中**故意拔掉 USB**：應印出「[live] ✖ 串列讀取失敗，reader thread
      結束：...」，然後「[error] 來源異常，開始收尾：串列來源已中斷：...」，
      最後手臂正常回 home 並斷線
- [ ] **不應該**只是靜默進 SAFE 然後卡住

---

## 錄影日速查（📹）

按這個順序做，不要跳步：

1. 夾爪已裝 → 急停在手邊 → 工作空間清空（含 J1 現在會轉到 90°）→ 關掉吃 CPU 的程式
2. `--source mock --dry-run` 跑一次，確認程式沒壞
3. `--source mock --arm-ip 192.168.1.166` 空跑一輪，確認手臂正常
4. 貼電極 → 筆電拔充電器 → 確認手臂懸空坐姿
5. `--source live --port COM10 --baud 1600000 --arm-ip 192.168.1.166`
6. **保持 DISARMED**，等校準完成（右側倒數消失）
7. 校準完先看終端機的分類與 confidence 是否合理——這是最後一道人工檢查
8. 確認無誤才按 Enter。**按下的瞬間手臂會先回 home，這是正常的**
9. 錄影。注意：夾爪閉合不要超過 `GRIP_HOLD_MAX_SEC`（會斷電掉東西）
10. `q` 結束。確認 log 存在 `results/control_logs/`，並記下實測 Hz 與掉包率

---

## 待決事項總表

| 編號 | 項目 | 目前值 | 決定依據 | 對應步驟 | 狀態 |
|---|---|---|---|---|---|
| — | 手臂控制 IP | `192.168.1.166` | 2026-08-19 實際連線成功確認（讀到 firmware/sn/axis） | — | ✅ |
| — | `dof_model` 重訓（ch2/ch4 + split 架構） | `final_v4_split.pt` | 2026-08-19 遷移到 semg_realtime_V4 | 0-3 | ✅ |
| — | Home 姿態 J1、回 home 速度 | `[90,...]`／`15°/s` | 使用者要求，2026-08-17 | — | ✅（可能還會再調） |
| M-1 | 夾爪安裝 | — | 首次連線前必須完成 | 0-1 | 🔲 |
| M-2 | `JOINT_LIMITS` | 六組值不變 | 2026-08-17 `reduced_joint_limits` 讀出，與硬體手冊 V2.6.0 逐項相符；`SOFT_LIMITS` 全落在內、餘裕充足 | 2-1 | ✅ |
| M-3 | `JOG_RATE_DEG_S` | 45 °/s | 實機手感（**有效區間 20~60 °/s**，`validate()` (e)/(g)/(i) 已鎖住） | 2-4 | 🔲 |
| M-18 | `JOG_SPEED_CAP`/`JOG_ACC`/`JOG_BLEND_RADIUS_DEG`（J6 抖動） | 60°/s / **300**°/s² / 2.0° | **2026-08-19 實機確認不再抖動。** 1100 是 mode 0 時代的遺留值，其推導前提（每步必停）在 mode 6 下已不成立，只剩每 0.1s 一次 96% 上限的加速度衝擊激振整條手臂。⚠️ **不要調回 1100**，見 M-18 專節 | 2-4 | ✅ |
| M-4 | `SOFT_LIMITS[6]` | ±180° | jog 範圍不夠可放寬到 ±270° | 2-5 | 🔲 |
| M-5 | `GRIP_HOLD_MAX_SEC` | 30 s | 實測發熱 + 錄影腳本時長。⚠️ 改由**手動版**驗（mock 腳本維持不了 30 秒） | 2-7 | 🔲 |
| M-6 | `COLLISION_SENSITIVITY` | 3 | 2026-08-17 實機：偏鈍但確實會觸發。維持 3——錄影時手不進入工作範圍，優先保錄影不被誤觸發中斷 | 2-8 | ✅ |
| M-7 | `P_MIN` | 0.80/0.75/0.75 | replay log 的 confidence 直方圖 | 3-1 | 🔲 |
| M-8 | `VOTE_WINDOW_SEC` 凍結 | 0.50 s | 觀察 log 是否有凍結區段 | 3-2 | 🔲 |
| M-9 | 端到端延遲 | 估 0.8–1.1 s | log 實測，論文用 | 3-3 | 🔲 |
| M-10 | 速度/加速度上限 | 180 °/s、1145.9 °/s² | 2026-08-17 讀出，已寫成 `FACTORY_JOINT_SPEED_MAX`/`FACTORY_JOINT_ACC_MAX`，`validate()` (h) 據此檢查全部速度/加速度參數 | 2-2 | ✅ |
| M-11 | 佇列堆積 | 無堆積 | 2026-08-17 實機：`cmdnum` 穩定在 0~1，放手後手臂即停 | 2-6 | ✅ |
| M-13 | 位置鏡像方向 | 方向正確 | 2026-08-17 實機視覺確認，`ELBOW_FLEX_ANGLE`/`SHOULDER_FLEX_ANGLE` 不需改 | 2-3 | ✅ |
| M-14 | CPU 佔用 | 未量測 | 工作管理員觀察 | 4-5 | 🔲 |
| M-15 | 掉包率 | 未量測 | 掉包告警輸出 | 4-4 | 🔲 |
| M-16 | 主迴圈實測 Hz | dry-run **93**、mock+真臂 **91**（另一輪 83，輸出量大） | 2026-08-17 實測，優於原估 40–60 Hz；`MAX_PREDS_PER_TICK=4` 不需調整。⚠️ **論文/文件不可寫「100 Hz 控制迴圈」**。live 值待 STEP 4 | 4-6 | 🟡 部分 |
| M-17 | `RECOGNIZER_VOTES`/`RECOGNIZER_CONF`/`dwell` | 1／0.6／6 | 新架構，尚未在真手臂上驗證過 | 3-3 | 🔲 |

---

## 量測記錄（貼實機輸出的地方）

> 每次上機量到新數字，貼在這裡，附日期。舊數字不要刪，畫刪除線保留歷史。

### 2026-08-17　`read_arm_specs.py`（STEP 2-0，唯讀）

```
firmware              : 6,9,LI1006,DL1000,v2.7.1
sn                    : LI1006
axis                  : 6
reduced_joint_limits  : [[-360.0, 360.0], [-150.0, 150.0], [-3.5, 300.0],
                         [-360.0, 360.0], [-124.0, 124.0], [-360.0, 360.0],
                         [0.0, 0.0]]
joint_speed_limit     : [0.05729578223448582, 180.00000500895632]
joint_acc_limit       : [0.5729577823242186, 1145.9155902616465]
collision_sensitivity : 3
angles (current)      : [~0, ~0, ~0, ~0, 0.0, ~0, 0.0]
state / mode          : 4 / 0
error_code/warn_code  : 0 / 0
```

判讀：

- `reduced_joint_limits` 前 6 筆與 `arm_config.JOINT_LIMITS` **逐項相同**，
  且與 Lite 6 硬體手冊 V2.6.0 的 Joint Range 相符（J3 的 `-3.5` 這個非整數
  值是最強證據——精簡模式若從未設定過不會湊出這種數字）→ M-2 結案，值不用改。
  第 7 筆 `[0.0, 0.0]` 是不存在的 J7（`axis=6`），忽略。
- `joint_speed_limit`/`joint_acc_limit` 是 **[min, max] 全域上下限**，不是
  per-joint 陣列 → 寫成 `FACTORY_JOINT_SPEED_MAX=180.0` /
  `FACTORY_JOINT_ACC_MAX=1145.9`，`validate()` (h) 據此檢查 → M-10 結案。
- `state=4` 是 STOP，開機未 `motion_enable` 的正常值；`startup()` 會做
  `motion_enable(True)` → `set_mode(0)` → `set_state(0)` 拉起來。
- `error_code=0` / `warn_code=0` → 可以往下走。
- `angles` 幾乎全零 → 手臂靜止在全零姿態，不是 home（見上方「已確認的事實」）。

### 2026-08-17　dry-run 主迴圈頻率（M-16 部分）

```
[info] 主迴圈實測平均 93 Hz（3404 ticks）
```

優於原估的 40–60 Hz，`MAX_PREDS_PER_TICK = 4` 不需調整。live 值待 STEP 4。

---

## 留給之後的事項（需要動控制層以外的檔案）

| 項目 | 影響範圍 | 優先度 |
|---|---|---|
| `read_one_frame` 回傳 `sample_index` 以做完整掉包偵測 | `src/hardware/record_session.py` | 中 |
| `config.ACTUATOR_SEMANTICS` 與控制層不一致（index/thumb 已改配 J6） | `semg_realtime_V4/src/semg/v2/config.py` | 低（僅註解，不影響執行） |
| 夾爪「即將斷電」預警 | `src/control/arm_controller.py` | 依 2-7 的結論決定 |
| 非 Windows 的按鍵支援（termios 分支） | `src/control/run_arm_control.py` | 極低（使用者用 Windows） |
