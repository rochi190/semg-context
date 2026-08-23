# moving_poseV2 —— sEMG → UFACTORY Lite 6 即時控制

> 2026-08-18 從 `moving_pose/` 整理而來，**只保留現在與未來會用到的東西**。
> 整包**自足**：`semg_realtime_V4/`（分類器 + 已訓練模型）已經放進來，
> 不依賴這個資料夾以外的任何路徑，可以整個搬走或複製。

---

## 結構

```
src/control/              ★ 手臂控制層（唯一在維護的程式）
  arm_config.py             所有可調參數集中一處 + validate() 自我檢查
  debounce.py               DofDebouncer：信心門檻 → N-of-M 多數決 → refractory
  arm_controller.py         Prediction 契約、MockArm、ArmController 狀態機
  sources.py                mock / manual / replay / live 四種 Prediction 來源
  live_bridge.py            ★ run_live.py ←→ ArmController 的黏合層（無控制邏輯）
  run_arm_control.py        CLI：模型預測驅動手臂（mock 流程版；肌電請用 run_live.py）
  run_manual_control.py     CLI：手動選 (4,2,2) 自由度驅動手臂

semg_realtime_V4/         ★ 分類器自足打包（含 3 個已訓練模型 + 2 份錄音）
  models/                   final_v4_split(預設) / _tiny / _zhao
  src/semg/v2/              config / preprocess / features / cues / model / infer / live
  src/hardware/             record_session.py（序列埠解幀的正本）
  data/raw/multi/           gary / neil 各一份錄音（STEP 3 replay 校準要用）

tests/                    35 個測試，`pytest tests/` 不需額外設定
docs/
  control-commands.md       ★ 所有指令＋replay/live 驗證流程的唯一去處
  arm-control-usage.md      ★ 控制層完整說明 + 快速調參指南
  manual-control-usage.md   手動選擇版說明
  on-machine-checklist.md   ★ 上機注意事項（這類資訊的固定去處）
  semg-pipeline-and-usage.md  訊號處理管線的方法說明
  his/                      歷史文件（交接紀錄、設計決策、已解決問題）

xArm-Python-SDK-master/   廠商 SDK（vendored）+ read_arm_specs.py（唯讀規格讀取）
CLAUDE_CODE_arm_control_spec.md   控制層任務書（所有設計決策的理由）
```

---

## 快速開始

```bash
cd moving_poseV2
pip install -r requirements.txt

# 離線驗證（不接硬體）
python src/control/run_arm_control.py --source mock --dry-run

# 讀手臂規格（唯讀，不會讓手臂動）—— 第一次接手臂時先跑這個
python xArm-Python-SDK-master/read_arm_specs.py

# 測試
pytest tests/ -q          # 35 passed
```

**所有指令與驗證流程都在 `docs/control-commands.md`。** 重點：肌電路徑
（讀檔版與肌電操控版）已於 2026-08-19 整併進 `semg_realtime_V4/run_live.py`
——加 `--arm-ip` 或 `--arm-dry-run` 就會驅動手臂：

```bash
cd semg_realtime_V4
python run_live.py --replay <錄音.csv> --arm-dry-run          # 讀檔 + 假手臂
python run_live.py COM10 --arm-ip 192.168.1.166               # 肌電操控（正式）
```

手動版（`run_manual_control.py`）維持獨立，用途是排除訊號變因、驗證手臂動作。

**上機前務必先看 `docs/on-machine-checklist.md`**——尤其 STEP 0 的四道閘門
（夾爪實裝、急停、工作空間、CPU）與「手臂靜止在全零姿態，首次回 home
J3 要走 180°、約 12 秒」這件事。

---

## 相對 `moving_pose/` 的差異

| 項目 | 說明 |
|---|---|
| `semg_realtime_V4/` 位置 | 從 repo 根目錄搬進本資料夾內 → `arm_config.SEMG_REALTIME_ROOT` 由 `REPO_ROOT.parent / ...` 改成 `REPO_ROOT / ...` |
| `tests/conftest.py` | 路徑順序改成與正式執行一致（`semg_realtime_V4/src` 優先）。舊版只加 `src`，測試其實是對著舊快照驗的 |
| **未搬入**：`src/semg/`、`src/hardware/` | v3 時期的舊快照，已被 `semg_realtime_V4/` 取代（ch2/ch4 標籤未修正、無模型）。留著只會製造「import 到錯的版本」的風險 |
| **未搬入**：`data/`（3.2 GB） | 舊錄音。要用的話直接指路徑，或之後再挑需要的搬 |
| **未搬入**：`results/` | 執行時自動產生（控制事件 log 會寫到 `results/control_logs/`） |

⚠️ **舊的訓練/分析腳本沒有搬過來**：`train.py`、`evaluate.py`、`audit_trials.py`、
`check_states.py`、`labeling.py` 只存在於 `moving_pose/src/semg/v2/`，
而 `semg_realtime_V4/` 這個推論打包裡沒有它們。
**如果之後要重新訓練模型，需要把那幾支搬過來**（它們是舊版、對應舊 config，
搬之前要先確認與 `semg_realtime_V4` 的 `FEAT_VERSION` / `CH_NAMES` 相容）。

---

## 目前狀態（2026-08-18）

- STEP 0（上機前四道閘門）✅、STEP 1（dry-run）✅、STEP 2（實機 mock）大致完成
- 已結案：M-2 限位、M-6 碰撞靈敏度、M-10 速度上限、M-11 佇列堆積、M-13 鏡像方向
- **下一步：STEP 3 replay** —— M-7 `P_MIN` 校準（最關鍵）、M-8 凍結檢查、M-9 端到端延遲
- 細節與待決清單見 `docs/on-machine-checklist.md`
