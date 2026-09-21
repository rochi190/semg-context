# 上機注意事項 —— UFACTORY Lite 6 首次整合清單

> **狀態：Rev. 2（完整版，取代 Rev. 1）**
> 收錄 `src/control/` 五個檔案 code review 中「必須接上實機才能驗證或決定」的項目。
>
> 純程式碼修正另見 **`CLAUDE_CODE_arm_control_fixes.md`**。
> ⚠️ **那份修改單必須先套用完成、`pytest tests/` 全綠，才可以進行本清單。**
> 修改單裡有 **6 個 P0**，其中 4 個的症狀都是「畫面全綠、手臂不動、零錯誤訊息」
> —— 沒修就上機，你會花整天在錯的方向找問題。

---

## 已確認的事實（不需要再查）

| 項目 | 值 | 來源 |
|---|---|---|
| 手臂 IP | **`192.168.1.181`** | 使用者已確認。`arm_config.ARM_IP` 正確 |
| 手臂 IP（錯誤來源） | ~~192.168.1.166~~ | 某份 handover md 寫的，**該文件應刪除** |
| STM32 序列埠 | **`COM10`** | 2026-08-13 確認。⚠️ 文件範例寫 `COM9`，是舊的 |
| Baud | `1,600,000` | `--baud` 預設值正確 |
| 控制模式 | mode 0（位置控制）全程 | mode 是全域的，位置鏡像與 J6 jog 無法用不同 mode |
| 取樣率 | 2000 SPS，`CONFIG1=0x93` | `config.py` 為準 |
| Gain | 8（`CHnSET=0x40`），1 LSB = 67.06 nV | ⚠️ 舊文件寫 Gain=4 / 134 nV 是**錯的** |

---

## 🔴 STEP 0：接上手臂之前必須先做

### 0-1 夾爪必須先實體安裝

`ArmController.startup()` **無條件**呼叫 `open_lite6_gripper(sync=False)`：

```python
self.arm.set_servo_angle(angle=AC.HOME_ANGLES, speed=AC.HOME_SPEED, wait=True)
self.arm.open_lite6_gripper(sync=False)     # ← 沒有任何條件判斷
self.grip_state = "OPEN"
```

沒有裝夾爪就跑這行，行為未定義（Lite 6 夾爪走 tool GPIO，沒有回授可判斷是否存在）。

- [ ] 夾爪已實體裝上並接好線
- [ ] 用 UFACTORY Studio 手動開合一次，確認夾爪本身正常

### 0-2 確認急停與工作空間

第一次接手臂會執行大範圍運動（回 home：J2 到 −90°、J3 到 180°）。

- [ ] 急停按鈕在手邊、確認按下有效
- [ ] 手臂工作空間內清空（特別是 J2 從 −90° 掃到 0° 的 90° 掃掠路徑）
- [ ] 手臂底座已鎖緊、桌面固定

### 0-3 確認 `dof_model.pt` 狀態

**電極貼法衝突（2026-08-13 已確認）**：實際貼法是 `ch2=biceps`、
`ch4=deltoid_anterior`，與原 `config.py` 的 `CH_NAMES` **相反**。

修正 `config.py` 會改變 `CROSS_PAIRS` 的通道索引 → 特徵向量改變 →
**`dof_model.pt` 必須重新訓練**。

- [ ] `config.py` 的 `CH_NAMES` 已修正為正確貼法
- [ ] `dof_model.pt` 已用修正後的 `config.py` 重新訓練

> 控制層不使用通道名稱，所以 **STEP 1、STEP 2 不受此項阻擋**，可以先做。
> 只有 `--source replay` / `--source live`（STEP 3 之後）需要正確的模型。

### 0-4 確認 CPU 資源

`CPU_THREADS = 2`，而 `--source live` 會同時跑：reader thread（串列）+
主 thread（推論 + 控制）+ xArm SDK 的 report thread。

- [ ] 錄影前關掉瀏覽器、Teams、其他吃 CPU 的程式
- [ ] 修改單的 FIX-M 修好了 reader thread 的 busy loop（原本 baud 設錯時會
      燒滿一核）—— 確認已套用

---

## STEP 1：`--dry-run` 離線驗證（不接手臂）

```bash
cd moving_pose
python src/control/run_arm_control.py --source mock --dry-run
```

**這一步在接手臂之前就要跑通。** 目的是驗證修改單的 P0 修正都生效了。

- [ ] 程式正常啟動，顯示「已回 home，DISARMED」
- [ ] 按 Enter → 印出 `已 ARMED`，且**第一個 `set_servo_angle` 是 home 角度**
      `[+0.0, -90.0, +180.0, +0.0, +0.0, +0.0]`
      → **FIX-B 迴歸**。若印出的是偏移姿勢（例如 J3=+90），修改沒生效
- [ ] 連續觀察 5 次 pinch 上升緣，夾爪序列為
      `CLOSED → OPEN → CLOSED → OPEN → CLOSED`
- [ ] 按 `z`，J6 立刻歸零
- [ ] 按 q，印出 `已回 home、斷線。`（不是舊的無條件字串）
- [ ] 結束時印出 `[info] 主迴圈實測平均 xx Hz` → **記下這個數字**（見 M-16）

### 1-1 SAFE 恢復測試（FIX-A 迴歸，必做）

Bug A 的症狀在實機上完全看不出來（畫面全綠、手臂不動、零錯誤），
**只能在 dry-run 用 `[MockArm]` 的輸出當證據**。

暫時把 `arm_config.WATCHDOG_SEC` 改成 `0.05`（逼 watchdog 誤觸發）：

- [ ] 啟動後按 Enter，很快出現 `ERROR 進入 SAFE：watchdog...`
- [ ] 此時按 Enter，應出現 `人工重新 ARM：離開 SAFE`，
      **並且印出 `[MockArm] motion_enable(enable=True)`、`set_mode(0)`、
      `set_state(0)` 三行**
      → 這三行就是 FIX-A。沒印出來就是沒修好
- [ ] 測完把 `WATCHDOG_SEC` **改回 0.5**

### 1-2 來源把關測試（FIX-K 迴歸）

暫時把 `run_arm_control.py` 裡的 `MockSource()` 改成 `MockSource(loop=False)`：

- [ ] 腳本跑完後按 Enter → 應顯示 `[拒絕 ARM] 來源尚未就緒`
      → 證明 `is_ready()` 把關生效
- [ ] 測完改回 `MockSource()`

### 1-3 錯誤路徑測試（FIX-Q / FIX-R 迴歸）

- [ ] `--source replay --csv /nonexistent.csv --dry-run`
      → 報錯退出，**且完全沒有印出任何 `[MockArm]` 指令**
      → 證明 source 建構在手臂初始化之前（FIX-Q）
- [ ] `--source mock --arm-ip 10.99.99.99`（故意打錯，不加 `--dry-run`）
      → 5 秒後 `✖ 無法連線手臂 10.99.99.99` 並列出四項檢查清單
      → 證明 FIX-R 生效。**這一項一定要測**，因為打錯 IP 的症狀跟 Bug A
        一模一樣，沒有這道檢查你分不出來

---

## STEP 2：`--source mock` + 實體手臂（第一次接手臂）

```bash
python src/control/run_arm_control.py --source mock --arm-ip 192.168.1.181
```

★ **這是第一次接實機時唯一該用的指令。** 不要跳到 `--source live`。

### 2-1 🔴 M-2：讀出並修正 `JOINT_LIMITS`（必做）

目前的值：

```python
JOINT_LIMITS = {                  # Lite 6 出廠限位（已查證）← 來源不明
    1: (-360.0, 360.0),
    2: (-150.0, 150.0),
    3: (-3.5, 300.0),             # ← ⚠️ 只有這軸是小數，形式可疑
    4: (-360.0, 360.0),
    5: (-124.0, 124.0),
    6: (-360.0, 360.0),
}
```

**為什麼要查**：`JOINT_LIMITS` 是 `clamp_angles()` 的**第二道防線**。目前
`SOFT_LIMITS`（J2: −90~0、J3: 90~180、J6: ±180）完全落在它裡面，所以第二道
**永遠不會啟動** —— 寫錯也看不出來。等到哪天放寬 `SOFT_LIMITS` 時才會出事，
而屆時已經沒人記得這個值沒驗證過。

**做法**（獨立小腳本，不要在控制程式裡跑）：

```python
# tools/read_arm_specs.py
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "xArm-Python-SDK-master"))
from xarm.wrapper import XArmAPI

arm = XArmAPI("192.168.1.181", is_radian=False)
time.sleep(1.0)                      # 等 report 通道建立，不可省略
assert arm.connected, "連線失敗"
print("firmware              :", arm.version)
print("sn                    :", arm.sn)
print("axis                  :", arm.axis)
print("joint_limits          :", arm.joint_limits)
print("joint_speed_limit     :", arm.joint_speed_limit)
print("joint_acc_limit       :", arm.joint_acc_limit)
print("collision_sensitivity :", arm.collision_sensitivity)
print("angles (current)      :", arm.angles)
arm.disconnect()
```

- [ ] 執行並**把輸出貼進本文件保存**
- [ ] 把讀到的 `joint_limits` 填回 `arm_config.JOINT_LIMITS`
- [ ] 把註解 `# Lite 6 出廠限位（已查證）` 改成
      `# 由 arm.joint_limits 讀出，YYYY-MM-DD，firmware x.x.x`
- [ ] 確認 `validate()` 仍然通過。**若 `SOFT_LIMITS` 超出新讀到的限位，
      要調 `SOFT_LIMITS`，不是註解掉檢查**

### 2-2 M-10：確認速度/加速度上限

`arm_config` 的註解寫「出廠上限 180 / 1145」也是未經驗證的數字。

- [ ] 用上面的腳本讀 `joint_speed_limit` / `joint_acc_limit`
- [ ] 更新 `arm_config.py` 的註解，標上讀出日期
- [ ] 確認 `JOINT_SPEED = 40.0` / `JOINT_ACC = 200.0` 確實在上限內

### 2-3 位置鏡像方向驗證

`MockSource` 的腳本會依序跑過 index / thumb / pinch / elbow flex / shoulder flex。

- [ ] `elbow=flex` 時 J3 走到 90°，**視覺上是「屈肘」**（不是伸直）
- [ ] `elbow=rest` 時 J3 回到 180°，視覺上是「伸直」
- [ ] `shoulder=flex` 時 J2 走到 0°，**視覺上是「平舉水平」**
- [ ] `shoulder=rest` 時 J2 回到 −90°，視覺上是「下垂」

⚠️ 若方向反了，**改 `arm_config.ELBOW_FLEX_ANGLE` / `SHOULDER_FLEX_ANGLE`
與對應的 `SOFT_LIMITS`**，不要改 `arm_controller.py` 的映射邏輯。
改完 `validate()` 會幫你檢查新值是否還在限位內。

### 2-4 M-3：J6 jog 手感 ＋ 🔴 死區陷阱

`JOG_RATE_DEG_S = 45.0` 標著 `TODO 待確認`，是猜的值。
45 °/s 代表：**從 0° 轉到 ±180° 軟限位需要 4 秒**。

- [ ] `hand=index` 時 J6 往 **+** 方向轉；`hand=thumb` 往 **−** 方向轉
      （若相反，交換 `arm_controller.on_prediction()` 裡 `jog_dir` 的 ±1）
- [ ] 感受速度：太快 → 手勢一抖就轉過頭；太慢 → 影片裡像蠕動
- [ ] 撞到 ±180° 時停住、不報錯、`cmd_sent` 變 0

#### 🔴 調整 `JOG_RATE_DEG_S` 的硬下限

```
JOG_STEP_DEG = JOG_RATE_DEG_S × CMD_PERIOD
CMD_DEADZONE_DEG = 0.5° ，CMD_PERIOD = 0.10 s

→ JOG_RATE_DEG_S 必須 > 0.5 / 0.10 = 5.0 °/s
```

**低於 5 °/s 會發生什麼**：每 tick 的變化量 < 死區 → 節流條件 2 永遠不成立
→ **指令一次都送不出去 → J6 完全不動**。

而終端機上你會看到 `J6=+0.4 → +0.8 → +1.2` 一直在跳（`target[5]` 確實在
累加），只有 `cmd_sent` 恆為 0。**「target 在動、手臂不動」這個現象幾乎不可能
聯想到是死區造成的**，會浪費大量時間去懷疑手臂、網路、分類器。

- [ ] 若要調慢，**不要低於 8 °/s**（留安全邊際）
- [ ] 修改單 FIX-F 的 `validate()` 已加入這條斷言，違規時 import 就會失敗
- [ ] 真的需要更慢的話，改法是**同時**降低 `CMD_DEADZONE_DEG`
      （例如 0.5 → 0.2），而不是註解掉檢查

### 2-5 M-4：`SOFT_LIMITS[6]` 是否需要放寬

- [ ] 實際 jog 時 ±180° 的範圍夠不夠？
- [ ] 不夠的話可放寬到 ±270°（仍在出廠 ±360° 內），但要先完成 2-1 確認限位

### 2-6 M-11：佇列堆積測試（`MockArm` 測不到的盲區）

`MockArm.cmd_num` 的模擬是 `min(self._cmd_num + 1, 1)` —— **永遠是 1**。
所以節流條件 3（`cmd_num < CMDNUM_MAX = 2`）在 dry-run 下永遠成立，
**佇列堆積只能在實機上驗證**。

- [ ] 觀察終端機的 `cmdnum` 欄位。正常運作時應在 0~1 之間跳動
- [ ] **關鍵測試**：讓 `MockSource` 跑到 `elbow=flex`（J3 走 90°，約 2.3 秒）
      的期間，觀察腳本切回 `rest` 之後手臂多久停止。
      **若手臂在腳本已切回 rest 之後仍繼續動 > 1 秒，就是佇列堆積。**
- [ ] 若 `cmdnum` 持續達到 2 並卡住，調整方向：
      - 提高 `JOINT_SPEED`（讓指令執行得比下發更快）
      - 或提高 `CMD_PERIOD`（降低下發頻率）
      - **不要**提高 `CMDNUM_MAX`，那是把問題藏起來

### 2-7 M-5：夾爪過熱斷電行為（⚠️ 會掉東西）

```python
GRIP_HOLD_MAX_SEC = 30.0        # TODO 待確認
```

夾爪連續閉合超過 30 秒 → `stop_lite6_gripper(sync=False)` → **斷電、夾持力
消失 → 夾著的物體會掉**。

**demo 時很容易踩到**：夾起物體、對鏡頭講解一下，30 秒就到了。

- [ ] 實測夾爪連續閉合 30 秒後是否真的明顯發熱（決定這個值合不合理）
- [ ] 評估延長到 60 s。錄影腳本若有「夾住物體」的橋段，先估算時間
- [ ] 目前**沒有**「即將斷電」的預警。錄影前決定要不要加（屬程式修改，
      要加就開 Rev. 3 修改單）
- [ ] 注意：斷電後 `grip_state` 仍是 `"CLOSED"`，所以下一次 pinch 上升緣會
      呼叫 `open_lite6_gripper()`。語意上有點怪但不會出錯

### 2-8 M-6：碰撞靈敏度

```python
COLLISION_SENSITIVITY = 3       # 0~5，中等
```

- [ ] 用手輕推手臂，確認會觸發 `error_code` → 進 SAFE
- [ ] 確認進 SAFE 之後**按 Enter 會被拒絕**（因為 `error_code != 0`），
      log 印出 `WARN 手臂 error_code=... 尚未清除，拒絕重新 ARM`
      → 這是 FIX-A 的一部分
- [ ] 太靈敏（正常運動就誤觸發）→ 降到 2；太鈍 → 升到 4

### 2-9 結束流程（FIX-E / FIX-P 迴歸）

- [ ] 按 `q`，手臂停止 → 回 home → 斷線，log 印 `已回 home、斷線。`
- [ ] Ctrl-C 也一樣
- [ ] **在有 `error_code` 的狀態下按 `q`**（先用手推觸發碰撞）：應印出
      `WARN 手臂 error_code=...，無法自動回 home` + `已斷線（未回 home）`
      → **FIX-E 迴歸**。手臂會停在原地，需人工用 UFACTORY Studio 復歸
- [ ] 任何情況下結束後，手臂都**不應該**還處於 motion_enable 狀態
      （用 UFACTORY Studio 確認）→ **FIX-P 迴歸**

---

## STEP 3：`--source replay` 校準去彈跳門檻

**前置**：`dof_model.pt` 已用修正後的 `config.py` 重新訓練（見 0-3）。

```bash
python src/control/run_arm_control.py --source replay \
    --csv data/raw/multi/gary/multi_gary_xxx.csv --arm-ip 192.168.1.181
```

### 3-1 🔴 M-7：`P_MIN` 必須用實際 confidence 分佈校準

```python
P_MIN = {"hand": 0.80, "elbow": 0.75, "shoulder": 0.75}
```

**這些值沒有任何模型 confidence 分佈支撐**，是規格書寫的建議值。

**風險**：4 類的 `hand` head 若沒做 temperature calibration，softmax 最大值
長期落在 0.5~0.7 是很常見的。那樣**所有 hand 幀都被閘一擋掉，夾爪一次都不會
動** —— 而終端機只會顯示 `hand=rest`，看不出是被門檻擋的。

**這是整條鏈上最容易出事、又最容易被誤診成「分類器完全失效」的一項。**

**做法**：控制 log（`results/control_logs/control_*.csv`）記錄的是
**原始**的 `p_hand` / `p_elbow` / `p_shoulder`（`on_prediction` 存 `pred.p_*`，
不是去彈跳後的值）。跑一次 replay 之後畫直方圖：

```python
# tools/plot_conf_hist.py
import sys
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(sys.argv[1])
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, cur in zip(axes,
                        ["p_hand", "p_elbow", "p_shoulder"],
                        [0.80, 0.75, 0.75]):
    ax.hist(df[col], bins=50, range=(0, 1))
    ax.axvline(cur, color="r", ls="--", label=f"P_MIN={cur}")
    passed = (df[col] >= cur).mean() * 100
    ax.set_title(f"{col}  通過率 {passed:.1f}%  中位數 {df[col].median():.2f}")
    ax.set_xlabel("softmax max")
    ax.legend()
plt.tight_layout()
plt.show()
```

- [ ] 跑一次 replay，畫出三個 head 的 confidence 直方圖
- [ ] **記錄每個 head 的通過率與中位數**
- [ ] 判斷準則：

      | 通過率 | 判斷 | 處置 |
      |---|---|---|
      | < 50% | 門檻太高，會頻繁凍結 | **降 `P_MIN`** |
      | 50~85% | 合理 | 保持 |
      | > 95% | 門檻太低，形同沒有閘一 | 升 `P_MIN` |

- [ ] 定案後把 `arm_config.P_MIN` 的註解從「可調」改成
      `# 由 <受試者>_<日期> replay log 的 confidence 分佈定出，通過率 xx%`

### 3-2 M-8：檢查 `VOTE_WINDOW_SEC` 是否造成凍結

修改單 FIX-C 新增了 `VOTE_WINDOW_SEC = 0.50 s`，代價是：**若超過一半的幀被
`P_MIN` 擋掉，該 DOF 會凍結在最後的穩定狀態不再轉態**。

- [ ] 觀察 log 裡 `hand` / `elbow` / `shoulder` 欄位是否有長時間不變、
      但 `p_*` 明顯在波動的區段 → 那就是凍結
- [ ] **若常凍結，正確處置順序是：先降 `P_MIN`（3-1），不要先放寬
      `VOTE_WINDOW_SEC`。** 放寬 window 等於把 Bug C 放回來
- [ ] 若 `P_MIN` 已降到 0.5 仍常凍結 → 問題在模型不在控制層，回去看訓練

### 3-3 M-9：量測端到端延遲（📹 錄影與論文用）

目前估計的延遲構成：

| 環節 | 估計 | 備註 |
|---|---|---|
| 模型證據累積 | ~0.3–0.5 s | 視窗 0.5 s + 序列跨度 0.95 s |
| `infer.Recognizer` 內部投票 | ~0.2 s | 與下面功能重疊 |
| `DofDebouncer` 投票 | ~0.2–0.3 s | `VOTE_N=4` 幀最少 0.20 s |
| 指令節流 | ≤0.10 s | `CMD_PERIOD` |
| **指令發出前小計** | **≈1.0–1.5 s** | |
| 手臂運動 | J3 走 90° 需 2.25 s | `JOINT_SPEED=40 °/s` |

- [ ] 用 log 的 `t` 與 `cmd_sent` 欄位量出實際值，**記錄下來（論文用）**
- [ ] 影片上若延遲明顯：**最高 ROI 的改善是合併兩層投票**
      —— `infer.Recognizer` 的投票與 `DofDebouncer` 的 N-of-M 功能重疊。
      這需要先 review `infer.py`，且會動到 `src/semg/v2/`，屬 Rev. 3
- [ ] 次選：`VOTE_N` 從 4 降到 3（省 0.05 s）。⚠️ 但要注意 3-3 平手時
      `max()` 會偏好索引小的 rest —— 這個隱含偏好目前只寫在 `debounce.py`
      的註解裡
- [ ] **不要**用提高 `JOINT_SPEED` 來「看起來變快」，那只縮短運動時間，
      不縮短反應延遲，而且提高碰撞風險

### 3-4 M-12：replay 結束後的行為確認

修改單 FIX-O 讓 replay 播完時印出說明。

- [ ] 重播結束時應印出 `[replay] 重播結束。接下來 watchdog 會在 0.5s 後...`
- [ ] 之後進 SAFE 是**正常的**，不是錯誤

---

## STEP 4：`--source live` 正式運作

```bash
python src/control/run_arm_control.py --source live --port COM10 \
    --baud 1600000 --model results/v2/dof_model.pt --arm-ip 192.168.1.181
```

⚠️ **`--port` 是 `COM10`**，不是文件範例裡的 `COM9`。

### 4-1 校準期間不要按 Enter

前 5 秒（`--calib-sec`）在估計穩健尺度，`poll()` 回傳空 list。

修改單 FIX-K 已加把關：按 Enter 會顯示 `[拒絕 ARM] 來源尚未就緒：校準中 還需 x.xs`。

- [ ] 確認終端機右側顯示校準倒數
- [ ] 確認校準完成後 `[live] 校準完成（xxxx 樣本）`
- [ ] **校準期間請保持中性放鬆、手臂懸空** —— 這段是基線估計，動作會污染尺度
- [ ] 若用 `--yes`，程式會自動等到就緒才 ARM（FIX-K），確認這個行為正確

### 4-2 錄製條件（沿用既有協定）

- [ ] 筆電**使用電池**，不接充電器（充電器引入共模雜訊）
- [ ] STM32 供電 3.3 V + 5 V 給 EVM
- [ ] BIAS 電極貼在肘關節（olecranon）
- [ ] EVM jumper：J6 全拔、JP1=1-2、JP6 **不裝**、JP7/JP8=1-2、JP22=2-3
- [ ] 電極貼法：ch1=latissimus_dorsi, **ch2=biceps**, ch3=triceps,
      **ch4=deltoid_anterior**, ch5=wrist_flexor, ch6=index, ch7=thumb

### 4-3 🔴 維持姿勢時手臂必須懸空

程式啟動時會印這段警告，但錄影時很容易忘：

> 手肘 / 前臂 / 上臂**都不能靠在桌面、扶手或身側**。否則重力被外物撐住、
> EMG 消失 → 被判成 `rest` → 機械手臂突然回 home。

- [ ] 錄影前確認受試者坐姿，手臂全程懸空
- [ ] 注意疲勞：懸空維持姿勢很累，sEMG 頻譜隨疲勞左移 → confidence 下降
      → 可能觸發 3-2 的凍結。**單次錄影不要太長**

### 4-4 M-15：掉包率量測

修改單 FIX-N 加了每 5 秒的掉包告警（wall-clock 對照法）。

> ⚠️ 這是**次佳實作**。`read_one_frame` 不回傳 frame 內的 `sample_index`，
> 無法做完整的索引連續性檢查（要改 `record_session.py`，屬 Rev. 3）。

- [ ] 觀察是否出現 `[live] ⚠ 掉包率約 x.xx%`
- [ ] **記錄穩定狀態下的掉包率**（論文用）
- [ ] 掉包率 > 1% 的處置方向：
      - 關掉其他吃 CPU 的程式（`CPU_THREADS=2`，資源很緊）
      - 換 USB port / USB 線
      - 確認 `set_buffer_size(rx_size=262144)` 有生效（非 Windows 平台無此方法）
- [ ] 掉包會靜默破壞訊號連續性 → 濾波與特徵算錯 → confidence 下降。
      **現象與「模型訓練得不好」完全無法區分**，所以必須量

### 4-5 M-14：CPU 佔用觀察

- [ ] 用工作管理員觀察 python 行程的 CPU 佔用
- [ ] 若某一核持續 100%，檢查是不是 reader thread 在 busy loop
      （baud 設錯、COM port 錯 → `find_sync` 一直失敗）
      → 修改單 FIX-M 已加 sleep 緩解，但根因仍要排除

### 4-6 M-16：主迴圈實測頻率

修改單 FIX-T 讓程式結束時印出實測 Hz。

**預期**：文件宣稱 100 Hz，但 Windows 的 `time.sleep(0.01)` 粒度約 15.6 ms，
實際可能只有 **40–60 Hz**。

- [ ] 記錄 dry-run 的實測值
- [ ] 記錄 live 的實測值（會更低，因為多了推論）
- [ ] 功能上安全（所有時間判定都用 `time.monotonic()`，不依賴 tick 數），
      但**不要在論文或文件裡寫「100 Hz 控制迴圈」**
- [ ] 若實測 < 30 Hz，`MAX_PREDS_PER_TICK = 4` 可能太小（會頻繁丟幀），
      觀察是否出現 `[warn] 單 tick 收到 x 筆 Prediction`

### 4-7 串列中斷的行為確認（FIX-M 迴歸）

- [ ] 運作中**故意拔掉 USB**：應印出
      `[live] ✖ 串列讀取失敗，reader thread 結束：...`
      然後 `[error] 來源異常，開始收尾：串列來源已中斷：...`
      最後手臂正常回 home 並斷線
- [ ] **不應該**只是靜默進 SAFE 然後卡住

---

## 錄影日速查（📹）

按這個順序做，不要跳步：

1. 夾爪已裝 → 急停在手邊 → 工作空間清空 → 關掉吃 CPU 的程式
2. `--source mock --dry-run` 跑一次，確認程式沒壞
3. `--source mock --arm-ip 192.168.1.181` 空跑一輪，確認手臂正常
4. 貼電極 → 筆電拔充電器 → 確認手臂懸空坐姿
5. `--source live --port COM10 --baud 1600000 --model ... --arm-ip 192.168.1.181`
6. **保持 DISARMED**，等校準完成（右側倒數消失）
7. 校準完先看終端機的分類與 confidence 是否合理 —— 這是最後一道人工檢查
8. 確認無誤才按 Enter。**按下的瞬間手臂會先回 home（FIX-B），這是正常的**
9. 錄影。注意：夾爪閉合不要超過 `GRIP_HOLD_MAX_SEC`（會斷電掉東西）
10. `q` 結束。確認 log 存在 `results/control_logs/`，並記下實測 Hz 與掉包率

---

## 待決事項總表

| 編號 | 項目 | 目前值 | 決定依據 | 對應步驟 | 狀態 |
|---|---|---|---|---|---|
| M-1 | 夾爪安裝 | — | 首次連線前必須完成 | 0-1 | 🔲 |
| M-2 | `JOINT_LIMITS` | 未驗證 | `arm.joint_limits` 讀出 | 2-1 | 🔲 |
| M-3 | `JOG_RATE_DEG_S` | 45 °/s | 實機手感（硬下限 8 °/s） | 2-4 | 🔲 |
| M-4 | `SOFT_LIMITS[6]` | ±180° | jog 範圍不夠可放寬到 ±270° | 2-5 | 🔲 |
| M-5 | `GRIP_HOLD_MAX_SEC` | 30 s | 實測發熱 + 錄影腳本時長 | 2-7 | 🔲 |
| M-6 | `COLLISION_SENSITIVITY` | 3 | 實機誤觸發率 | 2-8 | 🔲 |
| M-7 | `P_MIN` | 0.80/0.75/0.75 | replay log 的 confidence 直方圖 | 3-1 | 🔲 |
| M-8 | `VOTE_WINDOW_SEC` 凍結 | 0.50 s | 觀察 log 是否有凍結區段 | 3-2 | 🔲 |
| M-9 | 端到端延遲 | 估 1.0–1.5 s | log 實測，論文用 | 3-3 | 🔲 |
| M-10 | 速度/加速度上限 | 註解寫 180/1145 | `arm.joint_speed_limit` 讀出 | 2-2 | 🔲 |
| M-11 | 佇列堆積 | 未驗證 | 實機觀察 `cmdnum` | 2-6 | 🔲 |
| M-12 | `dof_model.pt` 重訓 | 待做 | `config.py` ch2/ch4 修正後 | 0-3 | 🔲 |
| M-13 | 位置鏡像方向 | 未驗證 | 視覺確認屈肘/平舉方向 | 2-3 | 🔲 |
| M-14 | CPU 佔用 | 未量測 | 工作管理員觀察 | 4-5 | 🔲 |
| M-15 | 掉包率 | 未量測 | FIX-N 的告警輸出 | 4-4 | 🔲 |
| M-16 | 主迴圈實測 Hz | 宣稱 100，實際未知 | FIX-T 的結束輸出 | 4-6 | 🔲 |

---

## 留給 Rev. 3 的事項（需要動控制層以外的檔案）

| 項目 | 影響範圍 | 優先度 |
|---|---|---|
| 合併 `infer.Recognizer` 與 `DofDebouncer` 的兩層投票（延遲最佳化） | `src/semg/v2/infer.py` | ★ 最高 ROI（若延遲在影片上明顯） |
| `read_one_frame` 回傳 `sample_index` 以做完整掉包偵測 | `src/hardware/record_session.py` | 中 |
| `config.ACTUATOR_SEMANTICS` 與控制層不一致（index/thumb 已改配 J6） | `src/semg/v2/config.py` | 低（僅註解，不影響執行） |
| 夾爪「即將斷電」預警 | `src/control/arm_controller.py` | 依 2-7 的結論決定 |
| 非 Windows 的按鍵支援（termios 分支） | `src/control/run_arm_control.py` | 極低（使用者用 Windows） |
