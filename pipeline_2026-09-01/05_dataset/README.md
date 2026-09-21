# 05 — dataset：切窗 + 抽特徵 + 快取

**檔案**：`dataset.py`（460 行）
**輸出**：`dataset.npz`

---

## 怎麼跑

通常不直接跑 —— 由 `leave_one_recording_out.py` 或 `train_final.py` 呼叫。
單獨跑：

```bash
PYTHONPATH=src python -m semg.v2.dataset --data-root V5moving/data --include configs/recordings_v5.txt
```

---

## 做什麼

```
對每份錄音：
  載入 CSV → counts 轉 mV → 去 DC
      ↓
  因果濾波（StreamFilter，與部署同一個實作）
      ↓
  校準：從乾淨的休息段算每通道穩健尺度 α（避開 _badspans）
      ↓
  切窗：0.5s / 0.05s 步進
      ↓
  每視窗抽 120 維特徵
      ↓
  用 cues.py 的區間貼標籤（prepare/return 標 -1）
      ↓
  套用 _badspans（遮蔽標籤）與 _exclude（排除 trial）
      ↓
  存進 dataset.npz
```

### npz 裡有什麼

| key | 內容 |
|---|---|
| `X` | (n_windows, 120) 特徵 |
| `y_dof` | (n_windows, 3) 每個 DOF 的狀態索引，−1 = 無標籤 |
| `segment` | 每個視窗屬於哪個連續片段（做序列時不能跨片段） |
| `src_file` | 每個視窗來自哪份錄音（★ LORO 分折用這個） |
| `subject` | 每個視窗來自誰（LOSO 分折用） |
| `feature_names`, `channel_names`, `dof_names`, `dof_n_states` | 中繼資料 |
| `use_hampel`, `feat_version` | ★ 寫進模型 bundle，防止誤配 |

---

## ★ 為什麼遮蔽的是「標籤」不是「樣本」

`_badspans` 標記的壞段落，處理方式是把該段的**標籤**設成無效，
與該段重疊 >5% 的視窗丟棄 —— 但**樣本仍然留在串流裡**。

理由：串流濾波器**有狀態**。把樣本抽掉會讓濾波器狀態與實際部署不一致，
等於在訓練時餵了一條部署時不會出現的訊號。

V5 實際套用：

```
multi_gary_20260811_17-11-49    1 段、共 20 s
multi_gary_20260811_17-23-12    1 段、共  9 s
multi_neil_20260810_21-17-16    1 段、共 89 s
multi_neil_20260810_21-35-03    排除 2 個 trial（錄音中斷）
```

---

## 快取

建一次資料集要 **124 s**（7 份、no-Hampel）或 **341 s**（開 Hampel）。
所以會存成 `dataset.npz`，之後跑不同架構直接沿用：

```bash
--cache results/loro_v5_nohampel/dataset.npz
```

⚠️ 快取**不記錄**它是用什麼設定建的以外的東西 —— 換 `--include` 清單或
換 Hampel 設定時要記得 `--force` 重建，否則會靜靜地用到舊的。

V5 的規模：**7 份錄音 → 24,925 個視窗**。
