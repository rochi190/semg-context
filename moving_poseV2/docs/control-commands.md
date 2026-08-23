# 指令總表與驗證流程

> **這是本專案所有執行指令的唯一去處。** 其他文件（`arm-control-usage.md`、
> `manual-control-usage.md`、`on-machine-checklist.md`、`README.md`、
> `CLAUDE_CODE_arm_control_spec.md`、`semg_realtime_V4/README.md`）需要指令時
> 一律連來這裡，不要各自複製一份。設計理由、參數調整依據等**非指令**內容
> 仍留在原本的文件。
>
> 相對路徑的基準：`run_live.py` 從 `moving_poseV2/semg_realtime_V4/` 執行，
> `run_manual_control.py` 從 `moving_poseV2/` 執行。

---

## 0. 兩個進入點，分工不同

2026-08-19 把肌電路徑整併進 `run_live.py`——**讀檔版與肌電操控版都走它**，
加不加 `--arm-ip` 決定接不接手臂。手動版刻意**維持獨立**，因為它的用途是
「驗證不同指令下手臂的動作」，跟訊號無關，要能當作隔離排查的對照工具。

| 進入點 | 位置 | 訊號來源 | 用途 |
|---|---|---|---|
| **`run_live.py`** | `semg_realtime_V4/` | `--replay <csv>`（讀檔）或 `COM10`（真串列） | 分類 → 螢幕，可選擇同時驅動手臂 |
| **`run_manual_control.py`** | `moving_poseV2/` | 你自己打三個數字 | 不經模型，直接驗證手臂動作 |

> `run_compare.py`（兩個模型並排比較）維持純螢幕、不接手臂——手臂只有一支，
> 沒辦法同時照兩個模型的輸出動。它是研究用工具，暫時擱置。

---

## 1. `run_live.py` —— 分類（＋可選的手臂）

### 純螢幕（不接手臂）

```bash
cd semg_realtime_V4

# 讀檔：用錄好的 CSV 模擬序列埠
python run_live.py --replay ../data/raw/multi/gary/multi_gary_20260811_17-23-12.csv

# 讀檔但不照真實速度跑（只驗證邏輯，不量延遲）
python run_live.py --replay <錄音.csv> --replay-fast

# 真串列
python run_live.py COM10

# ★ 靜止誤觸發測試 —— 接手臂前必做，且不能接手臂做
python run_live.py COM10 --idle-test 60
```

### 接手臂（加 `--arm-dry-run` 或 `--arm-ip`，其餘用法完全相同）

```bash
# 讀檔版 + 假手臂：完全離線，驗證整條鏈
python run_live.py --replay <錄音.csv> --arm-dry-run

# 讀檔版 + 真手臂：用錄好的訊號驅動實機
python run_live.py --replay <錄音.csv> --arm-ip 192.168.1.166

# 肌電操控版（正式運作／錄影）
python run_live.py COM10 --arm-ip 192.168.1.166
```

### CLI 參數

| 參數 | 預設值 | 說明 |
|---|---|---|
| *第一個位置參數* | — | 序列埠，本專案是 `COM10`。用 `--replay` 時不給 |
| `--replay <csv>` | — | 用錄好的 CSV 模擬序列埠。與序列埠二選一 |
| `--replay-fast` | 關閉 | 重播時不照真實速度（只驗證邏輯，量不到真延遲） |
| `--model <名稱>` | `final_v4_split` | 換模型即換架構（記在 `.pt` 的 `arch` 欄位）。可選 `final_v4_tiny`／`final_v4_zhao` |
| `--idle-test <秒>` | 關閉 | 靜止誤觸發測試。**不能與手臂並用** |
| `--calib-sec <秒>` | `5` | 開機靜止校準秒數 |
| `--votes` / `--conf` / `--dwell` | 見下方 | 不給時依「接不接手臂」自動選預設 |
| **`--arm-ip <IP>`** | 不接手臂 | 接上 Lite 6 並用分類結果驅動它。本專案是 `192.168.1.166` |
| **`--arm-dry-run`** | 關閉 | 用 `MockArm` 取代真手臂，指令只印到終端機 |
| **`--arm-yes`** | 關閉 | 校準完成後直接 ARMED，不等按 Enter |
| **`--arm-log <路徑>`** | 自動產生 | 控制事件 CSV；預設 `results/control_logs/live_<時間戳>.csv` |

### 操作按鍵（接手臂時）

| 按鍵 | 動作 |
|---|---|
| `Enter` | `DISARMED → ARMED`。校準完成後手臂預設 DISARMED、不會動；按下的瞬間手臂會先回一次 home，是正常的 |
| `q` / `Ctrl-C` | 停止 → 回 home → 斷線 |
| `z` | 只把 J6（腕旋轉）歸零 |

### ★ 單層投票（2026-08-19 決定）

`run_live.py` 本來就有 Recognizer 的投票機制，接手臂時**沿用它**，不在控制層
再疊第二層——疊兩層只會疊加約 0.2 s 延遲，而兩層對付的是同一件事（逐窗抖動）。

| 層 | 純螢幕 | 接手臂 | 對付什麼 |
|---|---|---|---|
| Recognizer `--votes` | 5 | **5**（唯一的投票層） | 逐窗抖動 |
| Recognizer `--dwell` | 0 | **6** | 移動過程中**穩定地**標錯的狀態（投票擋不住） |
| 控制層 `VOTE_M`/`VOTE_N` | — | **pass-through（1/1）** | — |
| 控制層 `P_MIN` | — | 照常 | 逐 DOF 的信心門檻（比 `--conf` 細） |
| 控制層 `REFRACTORY` | — | 照常 | 轉態頻率（防一個動作被拆成兩次觸發） |

`--votes`/`--dwell` 明確指定時一律以你給的為準。詳見 `src/control/arm_config.py`
的 `LIVE_RECOGNIZER_VOTES` / `LIVE_VOTE_M`。

---

## 2. `run_manual_control.py` —— 手動驗證手臂動作

不經過模型與訊號，你自己扮演分類器。**這支刻意不跟 `run_live.py` 合併**：
它的價值就在於排除訊號變因，之後手臂出問題時能拿它做隔離量測。

```bash
cd moving_poseV2

# 假手臂，先驗證選單與映射
python src/control/run_manual_control.py --dry-run

# 真手臂
python src/control/run_manual_control.py --arm-ip 192.168.1.166
```

| 參數 | 預設值 | 說明 |
|---|---|---|
| `--dry-run` | 關閉 | 用 `MockArm` 取代真手臂 |
| `--arm-ip` | `192.168.1.166` | 手臂控制箱控制用 IP |
| `--log` | 自動產生 | 預設 `results/control_logs/manual_<時間戳>.csv` |

**操作**：先按 Enter 進 ARMED，再輸入 `<hand> <elbow> <shoulder>` 三個數字。
`hand`＝0 rest／1 index／2 thumb／3 pinch，`elbow`／`shoulder`＝0 rest／1 flex。
例：`3 1 0` = pinch + 彎肘 + 肩放鬆。選完會等約 1.1 秒（去彈跳穩定）才印結果。

| 選單指令 | 動作 |
|---|---|
| `a` / `d` | 進入 ARMED（或從 SAFE 重新 arm）／進入 DISARMED |
| `z` | J6 歸零 |
| `s` | 印出目前狀態 |
| `q` / `Ctrl-C` | 回 home → 斷線 |

---

## 3. `run_arm_control.py` —— mock 流程版

`--source mock` 用腳本化的假手勢驗證手臂會動、限位正確。**第一次讓實機動起來
只能用這個**：輸入已知且完美，手臂不照腳本動就一定是控制層或手臂的問題。

```bash
cd moving_poseV2

# 完全離線
python src/control/run_arm_control.py --source mock --dry-run

# 假手勢 + 真手臂 ★ 第一次接手臂用這個
python src/control/run_arm_control.py --source mock --arm-ip 192.168.1.166
```

> `--source replay` / `--source live` 仍可運作（行為不變、兩層投票），但
> **肌電路徑請改用 `run_live.py`**——校準、電極自檢、calib_log、靜止誤觸發
> 測試都只在那條路徑上。

---

## 4. 其他工具

```bash
cd semg_realtime_V4
python check_install.py       # 套件 + 三個模型的架構 + 電極對照
python list_ports.py          # 找出板子接在哪個埠
python run_compare.py COM10 --idle-test 60      # 兩模型並排（純螢幕，不接手臂）

cd moving_poseV2
python xArm-Python-SDK-master/read_arm_specs.py  # 讀手臂規格（唯讀，不會讓手臂動）
pytest tests/ -q                                  # 35 passed
```

---

# 驗證流程

以下按實際執行順序排列。每一步的「怎麼看」就是判準——看不到那個現象就不要往下走。

## STEP A — Replay 版驗證（不接真手臂／接真手臂皆可）

```bash
cd semg_realtime_V4
python run_live.py --replay ../data/raw/multi/gary/multi_gary_20260811_17-23-12.csv --arm-dry-run
```

先用 `--arm-dry-run` 跑通，再換 `--arm-ip 192.168.1.166` 上實機。
⚠️ **不要加 `--replay-fast`**——量延遲需要真實速度。

### A-1　M-7：`P_MIN` 閾值校準 ★ 最關鍵

**風險**：4 類的 `hand` head 若 softmax 最大值長期落在 0.5~0.7，所有 hand 幀
會被 `P_MIN=0.80` 擋掉，夾爪一次都不會動——而終端機只顯示 `hand=rest`，
**看不出是被門檻擋的**。這是整條鏈上最容易出事、又最容易被誤診成
「分類器完全失效」的一項。

跑完後用控制事件 log（`results/control_logs/live_*.csv`）的 `p_hand`/`p_elbow`/
`p_shoulder` 欄位畫直方圖——那是**原始**信心值，不是去彈跳後的：

```python
import sys, pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv(sys.argv[1])
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, cur in zip(axes, ["p_hand", "p_elbow", "p_shoulder"], [0.80, 0.75, 0.75]):
    ax.hist(df[col], bins=50, range=(0, 1))
    ax.axvline(cur, color="r", ls="--", label=f"P_MIN={cur}")
    ax.set_title(f"{col}  通過率 {(df[col] >= cur).mean():.1%}  中位數 {df[col].median():.2f}")
    ax.legend()
plt.tight_layout(); plt.show()
```

- [ ] 記錄三個 head 的通過率與中位數
- [ ] 判準：通過率 **<50%** → 門檻太高會頻繁凍結，**降 `P_MIN`**；
      **50~85%** → 合理，保持；**>95%** → 形同沒有閘一，升 `P_MIN`
- [ ] 定案後把 `arm_config.P_MIN` 的註解改成
      `# 由 <受試者>_<日期> 的 confidence 分佈定出，通過率 xx%`

### A-2　M-8：決策凍結檢查

`VOTE_WINDOW_SEC=0.50s` 的代價：若超過一半的幀被 `P_MIN` 擋掉，該 DOF 會
**凍結**在最後的穩定狀態不再轉態。

- [ ] 看 log 裡 `hand`/`elbow`/`shoulder` 欄位是否有長時間不變、
      但 `p_*` 明顯在波動的區段 → 那就是凍結
- [ ] **常凍結的正確處置順序：先降 `P_MIN`（A-1），不要先放寬 `VOTE_WINDOW_SEC`**
- [ ] `P_MIN` 已降到 0.5 仍常凍結 → 問題在模型不在控制層，回去看訓練

### A-3　M-9：端到端延遲（📹 論文用）

改成單層投票後的理論構成：

| 環節 | 估計 |
|---|---|
| 模型證據累積（視窗 0.5s + 序列 0.45s） | ~0.95 s |
| Recognizer 投票（votes=5） | ~0.20 s |
| Recognizer dwell（=6） | ~0.30 s |
| 控制層投票 | **0（pass-through）** |
| 指令節流 `CMD_PERIOD` | ≤0.10 s |
| **指令發出前小計** | **≈1.3–1.5 s** |

- [ ] 用 log 的 `t` 與 `cmd_sent` 欄位量出**實際值**，記錄下來
- [ ] 程式啟動時印的「演算法延遲」那一行也記下來（它只算模型端）
- [ ] **不要**用提高 `JOINT_SPEED` 來「看起來變快」——那只縮短運動時間，
      不縮短反應延遲，而且提高碰撞風險

### A-4　主迴圈穩定度

- [ ] 回 home 結束後**沒有**假警報（不該出現「單 tick 收到 x 筆 Prediction」）
- [ ] 結束時印出的 win/s 穩定；先前實測主迴圈 83~93 Hz
- [ ] ⚠️ **論文/文件不可寫「100 Hz 控制迴圈」**——Windows 的 sleep 粒度約 15.6 ms

### A-5　M-12：replay 結束後的行為

- [ ] 重播結束、資料來源關閉後手臂會正常回 home 並斷線
- [ ] 若在結束前看到 watchdog 進 SAFE，那是**正常的**（來源沒資料了），不是錯誤

---

## STEP B — Live 肌電操控驗證

```bash
cd semg_realtime_V4
python run_live.py COM10 --arm-ip 192.168.1.166
```

### B-0　接手臂前的前置閘門（不可跳過）

- [ ] 夾爪已實體裝上並接好線（`startup()` 會無條件呼叫 `open_lite6_gripper`）
- [ ] 急停按鈕在手邊、確認按下有效
- [ ] 工作空間清空（回 home 時 J1→90°、J2→−90°、**J3→180° 約 12 秒**，
      是本專案最大範圍的一次自動運動，第一次跑務必手放急停）
- [ ] 關掉瀏覽器/Teams 等吃 CPU 的程式（`CPU_THREADS=2`）
- [ ] **先純螢幕跑過 `--idle-test 60`**，靜止誤觸發率可接受才接手臂

### B-1　修正 B：通訊防護

- [ ] `python run_live.py COM10 --arm-ip 10.99.99.99`（故意打錯 IP）
      → 5 秒後印出「✖ 無法連線手臂 10.99.99.99」並列出檢查清單
- [ ] **確認 COM10 有被正常釋放**：緊接著再跑一次正常指令應該能開得起來，
      不會出現「序列埠被佔用」或程式卡住不退出
- [ ] 運作中**故意拔掉 USB**：應該印出串列讀取失敗，然後手臂正常回 home 並斷線，
      **不應該**只是靜默進 SAFE 然後卡住

### B-2　7 通道訊號品質

啟動時的電極自檢會逐通道印出濾波後基線。

- [ ] 2000 SPS、Gain 8、Baud 1,600,000（`--baud` 預設值正確）
- [ ] ch1~ch7 全部 ✅。基線 **1–5 µV 正常／>15 µV 偏高／>40 µV 重貼**
- [ ] ⚠️ **直流偏移那一欄不是問題**，帶通會濾掉——要看的是濾波後那一欄
- [ ] 校準期間保持中性放鬆、**手臂懸空**（這段是基線估計，動作會污染尺度）
- [ ] 記錄穩定狀態下有沒有出現掉包告警（論文用）

### B-3　動作控制完整性（三大功能）

- [ ] **Pinch 夾爪**：每次 rest→pinch 的上升緣切換一次開/閉。連續 5 次應為
      `CLOSED → OPEN → CLOSED → OPEN → CLOSED`
- [ ] **夾爪保持 60 秒**（`GRIP_HOLD_MAX_SEC=60`）：超過會自動斷電降溫、
      **夾著的東西會掉**。實測是否明顯發熱，決定這個值合不合理
- [ ] **夾住的同時**換成 index/thumb → 夾爪維持 CLOSED、J6 照常 jog、
      elbow/shoulder 同時可動（三者不衝突）
- [ ] **J6 Jog**：`index` 往 + 、`thumb` 往 −；撞到 ±180° 停住不報錯
- [ ] **隨放即停**：放鬆手勢後 J6 應該立刻停，不該「鬆手後還繼續轉一小段」。
      ⚠️ 這項要用真訊號驗，手動版打 `0 0 0` 繞過了「模型判定放鬆」那一段
- [ ] **J2/J3 位置鏡像**：`elbow=flex`→J3 走 90°（視覺上是**屈肘**）、
      `rest`→180°（伸直）；`shoulder=flex`→J2 走 0°（**平舉水平**）、
      `rest`→−90°（下垂）。方向反了要改 `arm_config` 的 `ELBOW_FLEX_ANGLE`/
      `SHOULDER_FLEX_ANGLE`，**不要改 `arm_controller.py` 的映射邏輯**
- [ ] `cmdnum` 欄位正常在 0~1 之間跳動（持續卡在 2 代表佇列堆積）

### B-4　投票層優化評估

單層投票已套用（見 §1）。上機後依即時手感決定要不要再收：

- [ ] 手感仍鈍 → 先試 `--votes 3`（省約 0.1 s），再考慮 `--dwell 4`
- [ ] 誤觸發變多 → 往回加 `--votes`／`--dwell`，不要動 `P_MIN`
- [ ] ⚠️ `--dwell` 只擋得掉**比它短**的誤判，而且正確狀態也會延遲同樣時間

### B-5　錄影條件與最終交付

- [ ] 筆電**使用電池**，不接充電器（充電器引入共模雜訊）
- [ ] 電極：ch1 背闊肌／ch2 二頭肌／ch3 三頭肌／ch4 三角肌**前**束／
      ch5 曲腕肌／ch6 食指／ch7 拇指。BIAS 貼肘關節（olecranon）
- [ ] EVM jumper：J6 全拔、JP1=1-2、JP6 **不裝**、JP7/JP8=1-2、JP22=2-3
- [ ] **全程手臂懸空**——靠在桌面/扶手會讓 EMG 消失 → 判成 rest → 手臂突然回 home
- [ ] 注意疲勞：sEMG 頻譜隨疲勞左移 → confidence 下降 → 可能觸發 A-2 的凍結。
      **單次錄影不要太長**
- [ ] 📹 錄製完整「sEMG 驅動夾持 → 移動 → 放置」連續測試影片
- [ ] 夾爪閉合不要超過 60 秒（會斷電掉東西）

### 錄影日速查

1. 夾爪已裝 → 急停在手邊 → 工作空間清空 → 關掉吃 CPU 的程式
2. `python run_live.py COM10 --idle-test 60`（純螢幕）確認誤觸發率可接受
3. 貼電極 → 筆電拔充電器 → 確認懸空坐姿
4. `python run_live.py COM10 --arm-ip 192.168.1.166`
5. **保持 DISARMED**，等 5 秒校準完成
6. 看終端機的分類與 confidence 是否合理——最後一道人工檢查
7. 確認無誤才按 Enter。**按下的瞬間手臂會先回 home，是正常的**
8. 錄影 → `q` 結束 → 確認 log 在 `results/control_logs/`，記下實測數字

---

# 之後要改東西時，改哪個檔案

系統是分層的，**不要只改 `run_live.py`**。依修改性質選目標：

| 你想改什麼 | 改這裡 | 為什麼 |
|---|---|---|
| **參數**：速度／限位／逾時／門檻 | `src/control/arm_config.py` | 改一處，流程版與手動版自動同步生效 |
| **手臂行為**：新增動作／改 jog 演算法／夾爪邏輯 | `src/control/arm_controller.py` | 底層邏輯（`tick()` 等）由所有版本共用；改在這裡手動版才能繼續當隔離排查工具 |
| **模型／特徵／辨識過濾** | `semg_realtime_V4/src/semg/v2/` | 訊號與模型層 |
| **只改終端顯示或資料來源** | `semg_realtime_V4/src/semg/v2/live.py` | 純顯示層 |
| **分類↔手臂的黏合**（餵 Prediction、按鍵） | `src/control/live_bridge.py` | 只有黏合，**不放控制邏輯** |

保持「控制邏輯留在 `arm_controller.py`、參數留在 `arm_config.py`」的分層，
未來出新 bug 時才還能用手動版做隔離量測。

---

## 相關文件

- 參數**調整依據**（限位、去彈跳門檻、jog 速度）：`docs/arm-control-usage.md`
- 手動版設計理由與範例互動：`docs/manual-control-usage.md`
- 上機前準備、待決事項總表、量測記錄：`docs/on-machine-checklist.md`
- 模型架構、電極位置、切窗方式：`semg_realtime_V4/README.md`
- 所有決策的原始理由：`CLAUDE_CODE_arm_control_spec.md`
