# 手動選擇控制 —— 使用說明（`run_manual_control.py`）

> 這支程式讓你**自己扮演分類模型**：直接輸入 `(hand, elbow, shoulder)` 三個數字，
> 選你想要的 (4,2,2) 自由度組合，程式就會像收到真的 `Prediction` 一樣驅動 Lite 6。
> 目的：在 `src/semg/v2/` 的分類模型還沒訓練完成前，**先把後端手臂控制驗證完**。
> 背後邏輯跟 `run_arm_control.py` 完全共用 `ArmController`——去彈跳、位置鏡像、
> pinch toggle、J6 jog、雙層限位、看門狗、SAFE 全部照跑，差別只在 `Prediction`
> 的來源：這裡是你手動選的，不是模型跑出來的。詳細設計依據見
> `CLAUDE_CODE_arm_control_spec.md` 與 `docs/arm-control-usage.md`。

---

## 這支程式新增了什麼

```
src/control/
  sources.py             ★ 新增 ManualSource：外部呼叫 set_state() 指定目前狀態，
                          poll() 依 STEP_SEC(20Hz) 節奏持續吐出該狀態的 Prediction
  arm_controller.py       ★ 新增 ArmController.disarm() / .status()（公開介面）
  run_manual_control.py   ★ 新的 CLI 進入點：互動式選單
tests/
  test_manual_source.py   ★ ManualSource + disarm()/status() 的測試
docs/
  manual-control-usage.md ★ 本文件
```

沒有改動 `run_arm_control.py`、`debounce.py`、`arm_config.py` 既有邏輯——
只是在既有架構上多接一種 `Prediction` 來源。

---

## 安裝

跟 `run_arm_control.py` 共用同一套環境，見 `docs/arm-control-usage.md`「安裝」一節。
這支程式**不需要**分類模型、也不需要序列埠，只需要能連到手臂（或 `--dry-run`）。

---

## 使用流程

**啟動指令、CLI 參數、選單內指令（`a`/`d`/`z`/`s`/`q`）已整併到
`docs/control-commands.md`**（四種模式唯一的指令去處，不在這裡重複維護一份）。
這裡只留互動範例與設計理由。

### 互動流程

1. 程式會先跑跟 `run_arm_control.py` 一樣的連線/回 home 初始化序列（見
   `docs/arm-control-usage.md`「安全機制」）。
2. **按 Enter 進入 ARMED**——跟即時控制版一樣，預設 `DISARMED`，不會亂動。
3. 看到選單後，輸入三個數字選狀態：

```
hand     : 0=rest  1=index  2=thumb  3=pinch
elbow    : 0=rest  1=flex
shoulder : 0=rest  1=flex
```

```
> 3 1 0
→ 已選擇：hand=pinch elbow=flex shoulder=rest（等待去彈跳穩定 1.1s…）
[ARMED] hand=pinch elbow=flex shoulder=rest | J2=-90.0 J3=+90.0 J6=+0.0 | grip=CLOSED | cmdnum=1
```

`3 1 0` = pinch（夾爪閉合）+ 手肘彎曲（J3→90°）+ 肩膀放鬆（J2 維持 −90°）。

4. 選完之後程式會**先等一小段時間**（去彈跳的多數決視窗 + 不反應期 + 排程
   抖動的安全餘裕，約 1.1 秒）再印出結果，因為背後真的是走跟模型一樣的
   去彈跳流程，不是選了就立刻生效。

其他指令（`a`/`d`/`z`/`s`/`q`）對照表見 `docs/control-commands.md`。

---

## (4,2,2) 對照表

跟即時控制版用同一套映射（規格書 §4，已由使用者確認）：

| 選擇 | 對應動作 |
|---|---|
| `hand=1`（index） | J6 持續往 **+** 方向 jog（背景 thread 持續推進，直到你選別的狀態） |
| `hand=2`（thumb） | J6 持續往 **−** 方向 jog |
| `hand=3`（pinch） | 夾爪切換：從非 pinch 選到 pinch 的那一刻，開 ↔ 閉 toggle 一次 |
| `hand=0`（rest） | J6 停在當前角度；夾爪不受影響 |
| `elbow=1`（flex） | J3 → 90° | `elbow=0`（rest） | J3 → 180° |
| `shoulder=1`（flex） | J2 → 0° | `shoulder=0`（rest） | J2 → −90° |

### ★ 夾住之後仍可同時操控其他兩軸 + J6

夾爪狀態是獨立持續的，不會因為你下一次選了別的 `hand` 而被重置。例如：

```
> 3 0 0
→ 已選擇：hand=pinch elbow=rest shoulder=rest（等待去彈跳穩定 1.1s…）
[ARMED] ... grip=CLOSED ...

> 1 1 1
→ 已選擇：hand=index elbow=flex shoulder=flex（等待去彈跳穩定 1.1s…）
[ARMED] hand=index elbow=flex shoulder=flex | J2=+0.0 J3=+90.0 J6=+36.0 | grip=CLOSED | ...
```

第二次選 `1 1 1`（index + 彎肘 + 前舉）之後，夾爪**維持 CLOSED**，J6 同時持續 jog，
J2/J3 也同時移動到位——夾住東西的狀態跟操控其他三個軸完全不衝突。要放開就再選一次
`3 <elbow> <shoulder>`（觸發 pinch 上升緣，CLOSED→OPEN）。

⚠️ 選 `hand=1` 或 `2` 之後，J6 會**持續**轉動，直到你輸入新的選擇把 `hand` 改掉
（或 `d` 進入 DISARMED）——背景有一個獨立 thread 在跑，即使你停在 `>` 提示字元
等輸入，jog 也不會停。想馬上停住就選 `0 <elbow> <shoulder>`（hand=rest）或按 `z`。

---

## 為什麼選完之後要「等待去彈跳穩定」

這支工具刻意**不繞過**去彈跳三道閘（信心門檻 → N-of-M 多數決 → refractory），
因為它存在的目的就是幫你在真模型上線前，用同一套控制路徑驗證安全機制對不對。
`ManualSource` 給的信心固定是 `1.0`（永遠 ≥ `P_MIN`），所以唯一的延遲來源是：

- 多數決視窗：`VOTE_M`(6) 幀 × `STEP_SEC`(0.05s) = 0.3s
- 不反應期：`REFRACTORY` 依 DOF 是 0.4~0.6s

程式用 `max(REFRACTORY) + VOTE_M × STEP_SEC ≈ 0.9s` 當保守等待時間，
等完才印狀態，這樣你看到的畫面保證是「已經穩定」的結果，不會看到過渡期的雜訊。

---

## 測試

```bash
cd moving_pose
pytest tests/test_manual_source.py -q
```

涵蓋：`ManualSource` 依 STEP_SEC 節奏正確吐出所選狀態、手動選擇能驅動
`ArmController` 產生正確的目標角、`disarm()` 期間不會真的送出指令給手臂、
`disarm()` 對 `SAFE` 狀態無效（只能靠 `engage()`/`a` 離開 SAFE）。

---

## 跟 `run_arm_control.py` 的差異一覽

| | `run_arm_control.py` | `run_manual_control.py` |
|---|---|---|
| Prediction 來源 | mock（腳本）/ replay（CSV+模型）/ live（真串列+模型） | 你手動輸入 |
| 需要分類模型？ | replay/live 需要 | 不需要 |
| 操作方式 | 非阻塞按鍵（Enter/q/z），畫面即時刷新 | 阻塞式選單（`input()`），選完等去彈跳穩定才印結果 |
| 用途 | 驗證整條 pipeline（含模型）／正式運作 | 只驗證後端控制邏輯，或手動 demo 特定動作 |
