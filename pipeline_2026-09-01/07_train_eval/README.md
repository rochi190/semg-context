# 07 — train_eval：訓練與交叉驗證 ★ 報告重點

| 檔案 | 做什麼 |
|---|---|
| `train.py` | 單折訓練（給定訓練/測試切分，訓一個模型並評估） |
| **`leave_one_recording_out.py`** | ★ **交叉驗證主程式** —— 產生所有「數字」 |
| `train_final.py` | 訓練**部署用**的最終模型（全部資料，不留測試集） |
| `evaluate.py` | 逐自由度混淆矩陣、訓練曲線 |
| `ablation.py` | 消融實驗（每個設計決策值多少） |

---

## ★ 兩支程式的分工，一定要講清楚

```
leave_one_recording_out.py  → 產生【數字】（k 個模型，每個留一份出來測）
train_final.py              → 產生【要部署的那一個模型】（全部資料一起訓練）
```

⚠️ **最終模型沒有自己的準確率**，因為沒有留出資料可以評。
它誠實的期望值是**交叉驗證的平均** —— 而且那個平均還是**保守**的，
因為交叉驗證的每個模型都少看了一份資料。

**絕對不要拿「訓練集準確率」當效能數字報出去。**

---

# ★ LORO：交叉驗證的三種切法

`leave_one_recording_out.py` 支援三種 `--scheme`：

| `--scheme` | 一折留出什麼 | 回答什麼問題 | V5 的折數 |
|---|---|---|---|
| `recording`（預設） | 一份錄音 | 「換一份錄音還能不能用」 | 7 |
| **`session`** ★ | 同一人同一天的**全部**錄音 | ★「今天重新貼電極，還能不能用」 | 3 |
| `subject` | 一位受試者的全部錄音 | 「換人還能不能用」 | 2 |

## 怎麼跑

```bash
# 留一份錄音（舊指標）
PYTHONPATH=src python -m semg.v2.leave_one_recording_out \
    --data-root V5moving/data --include configs/recordings_v5.txt \
    --arch split --no-hampel --jobs 4 --out results/loro_v5_nohampel

# ★ 留一貼片場次（正確指標）
PYTHONPATH=src python -m semg.v2.leave_one_recording_out \
    --data-root V5moving/data --include configs/recordings_v5.txt \
    --arch split --no-hampel --scheme session --jobs 3 \
    --out results/loso_session_v5_nohampel
```

主要參數：

| 參數 | 用途 |
|---|---|
| `--include` | 錄音清單檔（`configs/` 底下），**單一事實來源** |
| `--arch` | `tiny` / `split` / `zhao` / `split_zhao` |
| `--scheme` | `recording` / `session` / `subject` |
| `--no-hampel` | 關掉 Hampel（交付模型用這個） |
| `--jobs N` | 平行跑 N 折 |
| `--force` | 重建資料集快取 |

## 每一折做什麼

```
1. 從 dataset.npz 取出訓練/測試視窗（依 scheme 決定怎麼切）
2. 組序列（SEQ_LEN=10，★ 不跨 segment 邊界）
3. StandardScaler 用【訓練集】統計量
4. 訓練模型（CrossEntropyLoss，ignore_index=-1，逐頭類別權重）
5. ★ 對留出的錄音做【串流重播】—— 用 infer.Recognizer 逐塊餵，
   含因果濾波、狀態保留、逐 DOF 投票
6. 算全對率與逐 DOF 準確率（原始 + 平衡）
7. 存這一折的模型、預測 CSV、對照圖
```

### ★ 第 5 步為什麼必須是串流重播

**不是**用 `filtfilt` + 逐視窗獨立預測。

後者會畫出一張比真實情況樂觀的圖 —— 零相位濾波用到未來樣本，
而且沒有投票延遲。**評估流程必須與部署路徑逐位元一致**，
否則量到的是一個不存在的系統。

---

# ★★ 最重要的發現：`recording` 切法系統性高估 22 個百分點

## 洩漏的機制

同一次貼片錄的兩份錄音共用：電極的**精確位置**（毫米級差異就會改變 crosstalk）、
皮膚**接觸阻抗**、導電膠**含水狀態**、受試者當下的**疲勞程度**。

留一「份」出來時，它的**同場次雙胞胎還在訓練集裡** ——
模型只要學會「這一組電極長什麼樣」就能答對，不需要真的學會姿勢。

形式化：設錄音 r 的特徵分布為 $P(\mathbf{f}\mid y, s(r))$，s(r) 是貼片場次。
`recording` 切法的訓練集**包含** s(r) 的樣本 → 測試分布與訓練分布同源；
`session` 切法時 s(r) 完全未見。

## V5 的實測

7 份錄音構成 **3 個場次**：

```
gary_20260811   17-11-49, 17-23-12          相隔 11 分鐘
gary_20260821   22-03-40, 22-17-38          相隔 14 分鐘
neil_20260810   21-17-16, 21-35-03, 21-54-56
```

| 切法 | 平均全對率 | hand | elbow | shoulder |
|---|---|---|---|---|
| 留一份錄音 | **95.3% ± 2.9%** | 99.5% | 95.9% | 99.1% |
| **留一貼片場次** | **73.0% ± 12.5%** | 94.1% | **79.9%** | 94.4% |

$$\Delta = 22.3\ \text{百分點}$$

## 獨立佐證（不是事後解釋）

舊清單裡的 `multi_gary_20260810_14-56-09` 是**唯一沒有同場次雙胞胎**的錄音。
它的 `recording` 分數是 **67.9%**，而有雙胞胎的都在 87–98%。

$$\text{沒有雙胞胎的錄音，}\texttt{recording}\text{ 分數自動退化成 }\texttt{session}\text{ 的水準。}$$

★ 這個異常值在提出洩漏假說**之前**就存在，當時被歸因為「這份品質差」。

## 逐折結果（`session`，no-Hampel）

| 留出的場次 | 錄音 | 全對率 | 60Hz 汙染 |
|---|---|---|---|
| gary_20260811 | 17-11-49 | 87.7% | 4.7% |
| gary_20260811 | 17-23-12 | 88.5% | 4.4% |
| gary_20260821 | 22-03-40 | 75.7% | 48.1% |
| gary_20260821 | 22-17-38 | **62.4%** | 21.2% |
| neil_20260810 | 21-17-16 | **55.0%** | 18.8% |
| neil_20260810 | 21-35-03 | 74.8% | 10.6% |
| neil_20260810 | 21-54-56 | 67.1% | 15.3% |

⚠️ **neil 那一折不能當「換場次」讀** —— neil 全部錄音都在同一場，
留出它等於**留出整個受試者**，混進了跨人的難度。
純粹「同一人、新的一次貼片」只有 gary 的四筆：**平均 78.6%**。

## 寫論文時的用詞

⚠️ 本專案內部曾把 `recording` 切法稱作「LOSO（以各錄音當一個 subject）」。
**論文請用 leave-one-session-out / leave-one-recording-out**，
不要用 LOSO 指稱錄音層級的切法 —— 那個縮寫在文獻上專指跨受試者。

---

## train.py — 單折訓練

| 項目 | 值 |
|---|---|
| 損失 | 三個頭的交叉熵**相加**，`ignore_index = -1` |
| 類別權重 | 逐頭計算（rest 佔比高，不加權會偏向 rest） |
| 優化器 | Adam |
| width / blocks / kernel / dropout | 48 / 2 / 3 / 0.1 |

$$\mathcal{L}=\sum_{d\in\{\text{hand},\text{elbow},\text{shoulder}\}}\mathrm{CE}(z^{(d)}, y^{(d)})$$

⚠️ **`ignore_index=-1` 的用途**：`prepare`/`return` 段沒有標籤（受試者正在移動），
標成 −1 讓損失跳過。沒有這個機制，移動過程會被強制標成某個狀態，
模型學到的邊界就是錯的。

### ⚠️ 一個讓整批實驗作廢的 bug

`run_fold` 曾經寫死 `torch.manual_seed(C.SEED)`，導致多種子實驗跑出
標準差 **0.0** —— 看起來「非常穩定」，其實根本沒換種子。

★ 教訓：**任何架構/超參數的比較，種子必須是被明確控制、而且被掃過的變因。**

---

## train_final.py — 部署模型

```bash
PYTHONPATH=src python -m semg.v2.train_final \
    --data-root V5moving/data --include configs/recordings_v5.txt \
    --arch split --no-hampel \
    --cache results/loro_v5_nohampel/dataset.npz \
    --out results/final_v5/models/final_v5_split.pt
```

輸出三個檔：`.pt`（bundle）、`_scripted.pt`（TorchScript）、`.json`（可讀的中繼資料）。

bundle 裡存了：權重、架構名、DOF 定義、scaler 統計量、特徵名、通道名、
**`use_hampel`**、**`feat_version`**、校準尺度。
後兩者是防呆用的 —— 載入時比對，不一致就拒絕。

---

## ablation.py — 消融

| 消融項 | 結果 |
|---|---|
| Hampel 開 vs 關（3 種子 × 7 折） | −0.13% ± 6.33%，**p = 0.926**（分不出差異，但快 3 倍） |
| 去掉跨通道特徵 | 78.3% → 71.9% |
| 只用跨通道特徵 | 72.9% |

---

## ⚠️ 統計檢定的教訓（建議寫進報告）

### 單一種子的顯著性檢定不可信

架構比較第一輪：p = 0.0132、Wilcoxon p = 0.0156、**7/7 全勝** —— 看起來很乾淨。
加兩個種子後：p = 0.0514、**12/21**。

$$\text{只跑第一輪，就會寫出一個}\textbf{有統計檢定背書的錯結論}。$$

### 檢定力很低，要誠實說

n = 7（`session` 時 n = 3）的配對檢定檢定力極低。
先前 A/B/E 三組資料集比較全部不顯著（p 從 0.12 到 0.45），差 1.9–5.3 pp 都分不出來。
**所以合理的結論通常是「分不出差異」，而不是「A 比 B 好」。**

### 被推翻過的結論清單

| 原本宣稱 | 重驗後 |
|---|---|
| B 資料集比 A 好 +9.2 pp | 共同 7 份上只有 +3.42%，p = 0.44 |
| zhao 架構顯著較好（7/7, p=0.013） | 3 種子後 +1.77% ± 3.92%, p = 0.051 |
| 關 Hampel 會變差 −3.48% | 3 種子後 −0.13% ± 6.33%, p = 0.926 |
| 某實驗的 moving 頭 93.4% | 正類從未被計分；真值 74.1% ± 10.8% |
| tiny 多種子標準差 0.0（很穩定） | `run_fold` 寫死種子，根本沒換 |
| **`recording` 切法 95.3%** | ★ **同場次洩漏；`session` 只有 73.0%** |

★ 這張表本身可以放進報告 —— 它說明評估流程有在自我校正。
