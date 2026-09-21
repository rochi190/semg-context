# sEMG 多自由度姿勢辨識 — 完整 Pipeline 程式包

> 2026-09-01 整理。從**錄資料**到**訓練**到**機械手臂控制**，依實際執行順序排列。
> 每個資料夾一個階段，裡面有該階段的程式 + 一份 `README.md` 說明它做什麼、
> 為什麼這樣做、輸入輸出是什麼。

---

## ⚠️ 先讀這段：這包是「閱讀用」，不是「執行用」

程式是**依 pipeline 順序重新排列**的複本，方便對照報告閱讀。
但 Python 的 import 路徑（`from semg.v2 import config as C`）是照原本的
套件結構寫的，**直接在這個資料夾裡跑會 import 失敗**。

要實際執行請用：

| 目的 | 用哪個 |
|---|---|
| 即時分類 / 訊號品質檢查 | `V5moving/`（自足，含程式 + 模型 + 資料） |
| 機械手臂控制 | `moving_poseV2/` |
| 重新訓練 / 跑 LORO | repo 根目錄，`PYTHONPATH=src python -m semg.v2.<模組>` |

各階段 README 裡的指令都是寫「在 repo 根目錄怎麼跑」。

---

## Pipeline 全圖

```
 ┌─────────────────────────────────────────────────────────────────┐
 │  00_config          所有超參數集中一處（其他每個階段都 import 它）│
 └─────────────────────────────────────────────────────────────────┘
            │
 ① 01_record          ADS1299 錄製 + 螢幕 cue          → .csv + _cues.csv + _meta.json
            │
 ② 02_quality_check   訊號品質 / 動作執行稽核           → 決定這份要不要用
            │
 ③ 03_preprocess      60Hz 陷波 → 20-450Hz 帶通 → 標籤區間
            │
 ④ 04_features        每 0.5s 視窗 → 120 維特徵
            │
 ⑤ 05_dataset         切窗 + 抽特徵 + 快取             → dataset.npz
            │
 ⑥ 06_model           4 種架構（tiny / split / zhao / split_zhao）
            │
 ⑦ 07_train_eval      ★ 訓練 + LORO / 留一場次交叉驗證  → 數字 + 最終模型
            │
 ⑧ 08_inference       串流推論 + 投票 + 即時分類
            │
 ⑨ 09_arm_control     去彈跳閘 → uFactory Lite 6
```

★ **⑦ 是報告的重點**，`07_train_eval/README.md` 裡有 LORO 的完整說明，
　 包括為什麼後來改用「留一貼片場次」。

---

## 資料夾說明

| 資料夾 | 階段 | 主要檔案 |
|---|---|---|
| `00_config/` | 共用基礎 | `config.py` — 所有超參數與其依據 |
| `01_record/` | 收資料 | `record_session.py` — 錄製 + cue 同步 |
| `02_quality_check/` | 品質把關 | `plot_signal.py`（訊號）、`audit_trials.py`（動作）、`check_states.py` |
| `03_preprocess/` | 前處理 | `preprocess.py`（濾波）、`cues.py`（標籤區間）、`labeling.py`（自動標籤） |
| `04_features/` | 特徵 | `features.py` — 120 維 |
| `05_dataset/` | 建資料集 | `dataset.py` |
| `06_model/` | 模型 | `model.py` — 4 種架構 |
| `07_train_eval/` | ★ 訓練與評估 | `train.py`、**`leave_one_recording_out.py`**、`train_final.py`、`evaluate.py`、`ablation.py` |
| `08_inference/` | 推論 | `infer.py`、`live.py`、`predict_and_plot.py`、`bench_latency.py` |
| `09_arm_control/` | 手臂控制 | `debounce.py`、`arm_controller.py`、`arm_config.py` 等 7 個 |
| `configs/` | 錄音清單 | 哪一版用了哪些錄音的**單一事實來源** |
| `data_manifest/` | 資料 metadata | 每份錄音的 `_meta.json` / `_cues.csv` / `_badspans.txt` |
| `models/` | 訓練好的模型 | `final_v5_split` 等 4 個 bundle |
| `results_summary/` | 結果 | LORO / 留一場次的 `summary.csv`、逐折模型、預測圖 |

---

## 資料在哪

`data_manifest/` 只放**小檔**（metadata、cue、壞段落標記），
原始錄音 CSV（每份約 60 MB、9 份共 534 MB）**沒有放進這包**，避免檔案過大。

原始 CSV 在：

```
V5moving/data/raw/multi/gary/      gary 6 份
V5moving/data/raw/multi/neil/      neil 3 份
```

需要含原始資料的版本再跟我說，我另外打一包。

---

## 相關文件

| 文件 | 內容 |
|---|---|
| `完整執行順序.md` | 從零開始跑一遍的**逐條指令** |
| `資料清單.md` | 用到哪 9 份錄音、各自的品質與用途 |
| **`延遲分析.md`** | ★ 機械手臂即時控制的延遲拆解（切窗 vs 運算） |
| `meeting_2026-08-24/`（repo 根目錄） | 進度報告：數學推導、選型論證、結果總表 |
| `docs/pipeline-methodology.md`（repo） | 踩過的 22 個方法論坑 |
