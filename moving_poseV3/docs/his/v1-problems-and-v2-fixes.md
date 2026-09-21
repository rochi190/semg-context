# V1 所有問題 × V2 如何修正

> V1 = 沿用 SoftPINCH 模型與其前處理／訓練方式的那一版（`src/semg/`、`semg_deploy/`）
> V2 = 自建的整套 pipeline（`src/semg/v2/`、`src/hardware/record_session.py`）
>
> 這份文件的目的是**不要再犯同一個錯**。每一條都寫實測證據，不寫猜測。
> 標記說明：
> - 🔴 **方法層問題**（沿用該論文的做法必然會遇到）
> - 🟠 **資料層問題**（錄製協定造成，換模型無法解決）
> - 🟣 **我自己寫錯的 bug**（跟論文無關，記下來避免重犯）
>
> 狀態：✅ 已修並驗證 ／ 🟡 已緩解但需真實資料確認 ／ ⬜ 未解（已知限制）

---

## 目錄

1. [資料與標籤](#1-資料與標籤)
2. [正規化](#2-正規化)
3. [模型與架構](#3-模型與架構)
4. [評估方法](#4-評估方法)
5. [工程實作](#5-工程實作)
6. [問題總表](#6-問題總表)
7. [V1 → V2 對照速查](#7-v1--v2-對照速查)
8. [仍未解決的事](#8-仍未解決的事)

---

## 1. 資料與標籤

### 1.1 🟠 沒有 trial 時間戳，只能猜邊界 ✅

**V1 狀況**
SoftPINCH 的 `_segment_trials` 把每個 trial 固定切成三等分（rest / contract / release），
這**要求精確的 trial 邊界**。我們的錄音沒有任何時間戳，只能用 autocorrelation
猜週期再推邊界。

**造成的實際傷害**
一度算出模型只有 **6.1%** 準確率，判斷成「模型在我們資料上失效」。
重新對齊邊界後同一份資料同一個模型變成 **92.4%**。
也就是說：**那個結論完全是標籤對齊誤差製造的假象**，差了 86 個百分點。

**V2 修正**
`src/hardware/record_session.py` 錄製時同步寫 `<stem>_cues.csv`，
記錄每個 cue 事件當下的 **`sample_index`**（不是牆上時鐘）：

```csv
sample_index,t_s,event,gesture,rep
1986,1.011182,prepare,index_flex,0      # 移動到目標姿勢 → 丟棄
2794,1.414845,go,index_flex,0           # 維持姿勢     → 標該姿勢
4010,2.020861,return,index_flex,0       # 移動回中性   → 丟棄
4617,2.322861,release,index_flex,0      # 中性放鬆     → 標 rest
```

取樣率固定 2000 Hz，`sample_index` 是唯一無漂移的時間軸。
切窗時直接用索引比對，**不需要任何對齊猜測**。

---

### 1.2 🟠 一個檔案只有一個手勢 → 手勢與錄音檔完全共線 ✅

**V1 狀況**
每個錄音只做一個手勢（`indexgary_20260711_00-17-09.csv` 整檔都是 index）。
標籤來自檔名，所以「手勢」與「錄音檔身分」在統計上**完全共線**。

**為什麼這是致命的**
> 沒有任何切分方式能證明模型學到的是肌電形態，而不是「認出這是哪一個錄音檔」。

實測（4 類，亂猜 25%）：

| 評估方式 | 電極位置 | RandomForest | CNN+BiLSTM+Attn |
|---|---|---|---|
| 隨機切窗 80/20 | 固定 | **99.9%** | val_loss≈0.0000 |
| 同 session（前70%訓/後30%測） | 固定 | **93.0% ± 4.1%** | — |
| within-subject（每手勢留 1 檔） | 重貼過 | 24–39% | — |
| LOSO（留一人） | 重貼過+換人 | 23.5–45.0% | **28.5% ± 3.7%** |

逐檔 leave-one-file-out：**18/28 檔低於 10%**。
但這個數字**不能**直接解讀成「這 18 檔壞掉」——
每個手勢每個 session 只錄一次，所以「同手勢不同檔」永遠等於「不同電極位置」，
兩個解釋在這份資料上分不開。

**關鍵反證：訊號本身是有資訊的**
SoftPINCH 預訓練模型（16 位受試者訓練，**從未見過我們任何檔案**）
在同一批資料上手指判別率 **84%**。
→ 問題不在訊號、不在模型架構，**在於 28 個「一檔一手勢」錄音無法支撐從零訓練**。

**V2 修正**
一段連續錄音內**隨機交替多個姿勢**（`build_schedule`），例如：

```
姿勢順序: pinch elbow_flex index_flex arm_lift thumb_flex pick_and_lift shoulder_flex ...
最長連續同姿勢 = 1
```

- 順序打散且避免同姿勢連續出現 → 模型無法用「上一個是什麼」推測
- 一檔多姿勢 → 姿勢不再與檔案共線，「模型學到肌電形態」變成**可驗證的主張**

---

### 1.3 🟠 固定循環錄製 → 模型可能只學到「循環到第幾秒」 ✅

**V1 狀況**
跟著 SoftPINCH 的方式錄：單一姿勢一直循環（出力 3s → 放鬆 3s → 休息 3s → 重複）。
SoftPINCH 自己的 train set 也是這種資料。

**為什麼是問題**
固定週期下，「現在是第幾秒」與「現在是什麼相位」是同一件事。
模型可以完全不看肌電，只數時間就得高分。
這與 1.2 疊加，是 93% → 28.5% 落差的另一半來源。

**V2 修正**
隨機順序（同 1.2）+ trial 級分組交叉驗證（見 4.1）。
兩條捷徑同時消失。

---

### 1.4 🟠🟣 沒有 metadata 就進 pipeline ✅

**V1 狀況**
`CLAUDE.md` 明文規定：
> ⚠️ 任何資料檔案在處理前，必須先確認對應的 session metadata。沒有 metadata 的資料不要進 pipeline。

我把 28 個檔案全部丟進 pipeline，沒有先確認哪些可用。
使用者後來說「很多都不是能用的」——那批數字裡有一部分本來就被壞檔案拖著。

**V2 修正**
`record_session.py` 錄製時**強制產生** `<stem>_meta.json`：

```json
{
  "protocol": {"gestures": [...], "reps": 5, "order_seed": 41234,
               "order_randomised": true, "lead_sec": 5.0, "tail_sec": 5.0,
               "prepare_sec": 2.0, "action_sec": 3.0, "return_sec": 1.5,
               "rest_sec": 3.0, "warmup_discard_sec": 1.0, "cue_before_action": true},
  "pose_spec": {"neutral": "...", "airborne": "維持姿勢時手臂必須懸空 ...",
                "labelled_phases": "go(維持)=該姿勢, release(放鬆)=rest; "
                                   "prepare/return(移動過程)=丟棄"},
  "hardware": {"sps": 2000, "gain": 8, "vref": 4.5, "lsb_mv": 6.7e-05, "baud": 1600000},
  "electrode_placement": {"ch1": {"muscle": "背闊肌", "name": "latissimus_dorsi"}, ...},
  "quality": {"n_samples": 596000, "n_bad_frames": 0, "duration_s": 298.0,
              "expected_duration_s": 298.0, "mean_fs_hz": 2000.0, "bad_frame_rate": 0.0},
  "notes": "..."
}
```

metadata 不再是「事後補的筆記」，而是**錄製流程的產出**，不可能忘記。

---

### 1.5 🔴 相位標籤：三次嘗試全部失敗 ⬜（改變策略而非硬解）

這一節完整記錄，因為它說明「不能硬幹」。

#### 嘗試 1（失敗）：`median + 3·MAD` 門檻抓 burst
**8/28 個檔案偵測到 0 個 burst。**
原因：這個公式假設「大部分時間在休息」。活動佔比高或訊號分布寬時，
median 落進活動區、MAD 跟著膨脹，算出的門檻**比 p95 還高**。

#### 嘗試 2（部分失敗）：改用百分位數門檻
0-burst 問題解決（0/28），但 burst 數量 **9 到 149 個**、長度 **0.7–18.4 秒**，極不一致。
畫圖看出根因**不在門檻而在相位定義**：
EMG burst 形狀是「幾乎垂直上升 → 高原緩慢衰減 → 落下」，所以
- `onset` 只有 1–2 個樣本（上升太快）
- **高原的緩慢衰減被判成 `release`**（斜率為負）
- 結果 release 29% vs hold 3%

#### 嘗試 3（失敗）：每通道 active/rest 多標籤（Lee et al. 風格）

| 手勢 | ch2−ch3 差值（正 = 食指主導） |
|---|---|
| index | **−0.082** ← 方向相反 |
| thumb | **+0.146** ← 方向相反 |

**方向完全反了**，標準差 0.15–0.20 重疊嚴重。
但對照 SoftPINCH 在同資料上 84% 手指判別率
→ 判別資訊在**波形形態**裡，簡單門檻抓不到。

#### V1 的結論：放棄相位，只用「手勢（檔名）+ 有沒有在動」

#### V2 修正：相位不再需要推導，直接從 cue 讀

```
prepare 段 → 整段丟棄（正在移動到目標姿勢，語意模糊）
go 段      → 該**姿勢**，兩端各內縮 0.3s（避開反應延遲與提前放鬆）
return 段  → 整段丟棄（正在移動回中性，方向與去程相反）
release 段 → 真正的 rest（不是「包絡線比較低的地方」）
```

`CUE_MARGIN_SEC = 0.3` 這個內縮是 V1 完全沒有、只能靠猜的東西。

★ 現行 V2 標的是「**維持中的姿勢**」而非「動作方向」——
移動過程（prepare / return）全部丟棄，約佔錄音的 50%。
詳見 `pipeline-v2-design.md` 第 11 節。

---

### 1.6 🟣 視窗標籤用多數決會汙染邊界 ✅

**V1 狀況**
V1 沒有明確處理跨邊界視窗（因為邊界本身就是猜的）。

**V2 修正**
`cues.py::window_label` 要求視窗內標籤 **95% 純度**，否則整個視窗丟掉：

```python
if cnt[j] / len(seg) < purity:
    return -1
```

不用多數決是刻意的：跨越 cue 邊界的視窗同時含 rest 與出力波形，
拿它訓練會把混合波形教成單一類別，**直接汙染 onset 判別**。

---

## 2. 正規化

### 2.1 🔴 z-score 對校準期組成高度敏感 ✅

**V1 狀況**
SoftPINCH realtime checkpoint 附 `mu.npy` / `sigma.npy`（絕對 mV），
模型輸入前做 `(env_buf - mu) / (sigma + 1e-8)`。
校準協定是「出力 3s / 放鬆 3s / 休息 3s」。

**實測脆弱性**
把校準期的休息佔比從 30% 掃到 90%：

| 休息佔比 | 手指判別率 |
|---|---|
| 30% | ~88% |
| 90% | ~65% |

**漂移 ±8.8pp。** 也就是使用者只要休息久一點，準確率就變了 —— 這不是能接受的部署特性。

**其他實測**
- 校準**必須包含出力**：只錄休息 67.5%，含出力 84.5%
- 校準長度：15s = 79.6%、45s = 86.0%、60s = 87.1%、90s 之後持平
- 校準手勢要對：同手勢 79.2% > 兩手勢 70.7% > 錯手勢 54.7%
  （⚠️ 兩手勢那組有混淆：兩段獨立錄音接起來會讓 sigma 膨脹 1.4–2.4×，
  沒有「交替手勢的單一錄音」就無法排除，這是 V1 的已知限制）

**V2 修正：改用 MAD 穩健尺度 + 尺度不變特徵**

`features.py::file_scales`：
```python
mad = 1.4826 * np.median(np.abs(filt - med), axis=0) + 1e-12
return mad, 0.05 * mad      # 死區 = 5% 穩健標準差
```

三層防護：
1. **頻域特徵本身尺度不變**（MNF / MDF / PKF / bandwidth）—— 乘一個常數不會變
2. **形態特徵尺度不變**（shape factor；ZC / SSC 用相對死區門檻）
3. **振幅特徵除以 MAD** —— MAD 是中位數統計量，對「休息佔多少比例」不敏感（median 不動）

而且校準只要一段**任意**訊號就能估雜訊底線，連純休息都行
（`infer.py::calibrate` 的 docstring 有寫）。V1 的 mu/sigma 必須涵蓋出力。

---

### 2.2 🔴 跨通道比值在 z-score 後會爆炸 ✅

**V1 狀況**
想用「食指/拇指能量比」當判別特徵。z-score 後取比值，分母趨零 → 數值爆炸。
某個檔案掉到 **47%（低於亂猜）**。
改成逐通道各自正規化，又把判別訊號一起消掉：**Cohen's d 從 3.88 掉到 0.12**。

**V2 修正**
`features.py::cross_channel_features` 三種都是有界、無除零風險的形式：

```python
a = np.sqrt(np.mean(win ** 2, axis=0)) + 1e-12   # 每通道 RMS
out  = [la[i] - la[j] for i, j in _PAIR_IDX]      # log 比值（等價於比值取 log，不爆）
out += [v / tot for v in a]                       # 視窗內總和歸一化（有界 [0,1]）
out += [grp[g] / tot for g in C.CH_GROUPS]        # 分組佔比
out.append(log(distal) - log(proximal))           # 遠端/近端 log 能量比
```

log 比值把除法變減法，數值範圍受控；總和歸一化天然有界。

---

### 2.3 🔴 無法從原始碼確認 SoftPINCH 離線用什麼正規化 ⬜

**V1 狀況**
`load_and_visualize_data.py` 沒有公開，所以離線訓練的正規化方式**讀不到**。
只能反推：per-file z-score 能重現他們宣稱的 99.4%（我們算到 98.65% / 99.07%），
不做正規化則全部預測 Rest。

**V2 修正**
自己訓練 → **訓練時用什麼正規化，模型就學什麼**，不存在「猜論文怎麼做」的問題。
`train.py` 的 `StandardScaler` 統計量只從訓練集算（`run_fold` 第 59 行），
所有正規化決策都在 `config.py` 與 `features.py` 裡明文可讀。

---

## 3. 模型與架構

### 3.1 🔴 7 類 realtime checkpoint 在我們資料上等於亂猜 ✅

**V1 實測**
`Real_time_inference/SingleNet_CNN+LSTM_EMG/subject_0`（7 類 = Index/Thumb/Pinch × Contract/Release + Rest）：

| 指標 | 數值 |
|---|---|
| 準確率 | **32.3%**（亂猜 33%） |
| 預測分布 | **77.6% 的預測都是 Index**，不管真實手勢是什麼 |

**乾淨對照**（同一份資料、同一套前處理）：

| checkpoint | 訓練資料 | 我們資料上的準確率 |
|---|---|---|
| 5 類離線 | 16 位受試者 LOSO | **84%** |
| 7 類 realtime | **只有 subject_0** | **32%** |

→ 泛化能力來自**訓練受試者的多樣性**，不是架構。

**還發現兩件事**
- `classification_pipeline.py:2398` 的 `MAX_NUM_TRIALS = 1`
  → realtime 的超參數是**一次隨機 SHERPA draw**（離線版是 70 trials）
- `classification_pipeline.py:2057-2073` realtime 標籤是**斜率式**
  （`REST_THRESH=0.003`，`slope>0`→contract），位置式版本被註解掉
  → 兩個 checkpoint 學的是**不同的任務**（相位分類 vs onset/offset 偵測），不是同一個模型的兩種設定

**V2 修正**
不再依賴任何預訓練 checkpoint。自己訓練，訓練目標明確寫在 `cues.py`，
評估同時報 trial 分組 / within-session / within-subject / LOSO 四個數字。

---

### 3.2 🔴 5 類互斥 → 無法表示「同時彎曲」 ✅

**V1 狀況**
5 類是互斥 softmax，沒有 pinch。7 類有 pinch 但如 3.1 不可用。
使用者的核心需求「同時彎食指和拇指」在 V1 表達不出來。

**V2 修正**
`GESTURES = ("rest", "index", "thumb", "pinch", "grasp")`
pinch 與 grasp **各自成為一個類別**，且訓練目標完全來自 cue，不依賴任何啟發法。

> 使用者提過「理想的 model 應該能分出不同物件，like YOLO」——
> 那需要每個物件的標註資料。目前協定先解決手勢層級；
> 要做物件層級必須在 cue 裡加入物件維度（協定可擴充，`CUE_GESTURES` 改一行）。

---

### 3.3 🔴 3 秒視窗，延遲 0.70s 且無法靠縮短視窗改善 ✅

**V1 實測延遲拆解**

| 來源 | 時間 | 佔比 |
|---|---|---|
| 模型（3s 視窗） | 0.70s | **70%** |
| 後處理（3 窗 × 100ms） | 0.30s | 30% |

（我一度說「延遲主要來自 5 窗多數投票」，**這是錯的**，實測後已更正。）

縮短視窗會掉準確率，因為模型沒見過那個長度 —— 不是延遲問題而是分布外問題。

**V2 修正**

| 項目 | V1 | V2 |
|---|---|---|
| 視窗 | 3.0s | **0.5s**（Zhao et al.） |
| 序列上下文 | — | 0.45s（10 窗 × 50ms） |
| 投票 | 3×100ms = 0.30s | 5×50ms = **0.20s** |
| **總延遲** | ~3.3s | **1.15s** |

因為自己訓練，短視窗不會有「模型沒見過的長度」問題。
投票變便宜是因為步進從 100ms 縮到 50ms。

---

### 3.4 🔴 RMS 包絡線是有損摘要 ✅

**V1 狀況**
SoftPINCH 把 40 Hz RMS 包絡線（3s → 120 點）餵進模型，
論文說法是「提前幫模型濾特徵、減輕模型負擔」。

**問題**
RMS 把 500 個樣本壓成 1 個數字，**頻譜資訊全部消失**：
MNF / MDF / PKF / bandwidth / MFCC 一個都算不出來。
肌纖維傳導速度與募集模式就藏在那裡。他們的說法沒錯，但沒說那是**有損**的。

| 論文 | 進模型的東西 |
|---|---|
| SoftPINCH | 40 Hz RMS 包絡線 |
| Lee 2024 | bandpass + 整流，保留 2000 Hz |
| Zhao 2025 | 從原始波形算多域特徵（時域+頻域+倒頻譜） |

**V2 修正**
走 Zhao 路線：從**濾波後原始波形**算 120 維多域特徵，頻域資訊保留。
旁證：permutation 特徵重要度裡 MNF / MDF / PKF 都排得進前段。

---

### 3.5 🔴 LSTM 只取最後隱狀態 ✅

**V1 狀況**
SoftPINCH 是單向 LSTM 取最後一步的隱狀態。動作若不發生在序列結尾就吃虧。

**V2 修正**
`model.py::Attention` —— 加性 attention 加權聚合整個序列，
且 LSTM 改**雙向**：

```
輸入 (batch, 10, 120)
  → Conv1d(120→64, k=3) → BatchNorm → ReLU → MaxPool(2)
  → BiLSTM(64→128, 雙向輸出 256) → Dropout(0.2)
  → Additive Attention（學權重聚合，不是取最後一步）
  → FC(256) → ReLU → Dropout(0.5) → Linear(5)
輸出 (batch, 5)   rest / index / thumb / pinch / grasp
```

355,013 參數（SoftPINCH 離線版 42,245 / realtime 版 344,583）。

---

### 3.6 🟣 我把目標設計錯了 ✅

**發生什麼**
我寫了 `simplify_predicted()` 把 5 類輸出收斂成 3 個夾爪狀態（open/close/hold），
**丟掉了 Index/Thumb 的區分** —— 那正是使用者要的東西。
使用者直接指出：「我要做的是手勢分類，所以你要我只分類一隻?」

**修法**
加 `GestureStateMachine`、`--mode gesture` 設為預設、歷史紀錄用 I/i/T/t 符號。

**教訓（寫進 V2）**
V2 的模型輸出就是手勢類別本身（`GESTURES`），
致動器映射放在**下游**（`infer.py` 之後），不在模型輸出層做壓縮。

---

## 4. 評估方法

### 4.1 🔴 隨機切窗 99.9% 是洩漏 ✅

**V1/V2 共同的陷阱**（V2 建好後第一次跑就撞到）

視窗步進 50ms、序列涵蓋 0.95s → 隨機切分時
**幾乎每個測試序列在訓練集裡都有 95% 重疊的鄰居**。
加上一檔一標籤（1.2），模型只要認出檔案就滿分。

鐵證：**所有折的 `val_loss ≈ 0.0000`**。

> 任何論文只報隨機切分的數字，都要先問它有沒有處理這件事。

**V2 修正**
`dataset.py::trial_group_splits` —— 分組鍵是 **trial**（`<檔名>#<rep>`）：

```python
groups = np.unique(d["trial"])
rng.shuffle(groups)
for part in np.array_split(groups, n_folds):
    te = np.isin(tr_key, part)
```

同一次出力的所有視窗一定在同一折。
`train.py::make_sequences` 也改成**只在同一 trial 內堆疊序列**（原本是同檔案內）。

---

### 4.2 🔴 只報一個數字 ✅

**V1 狀況**
SoftPINCH 論文只報一個準確率，我們花了很多力氣才搞清楚那對應哪個設定
（per-file z-score + 位置式標籤 + 16 人 LOSO）。

**V2 修正**
`train.py` 一次跑完並全部報出來，四個都寫進 `results/v2/train_results.csv`：

| scheme | 意義 | 樂觀程度 |
|---|---|---|
| `trial_kfold` | trial 分組 5 折（同 session） | 樂觀但無洩漏 |
| `within_session` | 同一天，前 70% 訓 / 後 30% 測 | 電極位置固定的上界 |
| `within_subject` | 同一人，每手勢留 1 檔（跨 session） | 跨電極重貼 |
| `LOSO` | 留一位受試者 | 真正的泛化 |

外加：混淆矩陣（`confusion_matrices.json` + `evaluate.py` 畫圖）、
逐類別 precision/recall、訓練曲線、permutation 特徵重要度。

---

### 4.3 🟣 within-subject 切分會抽走整個類別 ✅

**Bug**
```python
te_files = set(rng.choice(fs, size=max(1, len(fs) // 4), replace=False))
```
CBW1 只有 4 檔（每手勢 1 檔），抽走 1 檔 → **該手勢在訓練集完全消失** → 必然 0%。
gary/neil1 同 seed 抽到同手勢群，也是 0%。

三個受試者**都恰好 0.0%** 就是這個 bug 的指紋（三人獨立完全 0% 機率上不可能）。

**修法**
```python
for gi in range(n_cls):
    fg = np.unique(files[m & (y == gi)])
    if len(fg) < 2:
        ok = False; break        # 每手勢不足 2 檔 → 略過該受試者
    te_files.append(sorted(fg)[-1])   # 每手勢各留 1 檔
```

**教訓**：異常整齊的數字（0.0%、100.0%）優先懷疑 bug，不要先解釋成模型行為。

---

### 4.4 🟣 品質門檻用錯代理指標 ✅

**V1 狀況**
用「動態範圍」當訊號品質門檻，想自動篩掉壞錄音。

**實測失效**
- 與誤觸發率相關只有 **r = −0.40**
- 誤觸發 <5% 的檔案：範圍 2.1–25.0×
- 誤觸發 ≥5% 的檔案：範圍 2.2–6.5×
→ **完全重疊**，這個門檻沒有判別力。

**修法**
`check_signal.py` 改成**直接量測** pipeline 的誤觸發率／偵測率／延遲，不用代理指標。

**V2 做法**
`record_session.py` 錄完立刻印**直接可測的**品質數字：
```
樣本數     : 596,000（預期 596,000）
平均取樣率 : 2000.0 Hz
壞幀       : 0（0.000%）
✅ 樣本完整度 100.0%
```
低於 98% 會警告。這些是**定義明確**的量，不是相關性可疑的代理。

---

## 5. 工程實作

### 5.1 🟣 封包格式猜錯 ✅

**Bug**
我猜 `SerialSource._parse` 是 12-byte frame（3×int32）。
實際（`ads1299_readerV6.py`）：**40-byte frame**、SYNC `0xA5 0x5A`、
`"<I8i"`（uint32 status + 8×int32）、XOR checksum、END `0x0D`、baud **1,600,000**。

**後果**：接上硬體會**完全無法讀取**。

**修法**
重寫 `_extract_frames()`，並用合成封包驗證四種情況：
完整幀、前面有垃圾、checksum 錯、不完整幀。

**V2 做法**
`record_session.py` 的 `find_sync` / `read_one_frame` 是**直接沿用**
`ads1299_readerV6.py` 的邏輯，不重新發明。

---

### 5.2 🟣 改了檔案但沒重新打包 ✅

**發生什麼**
我改了 `deploy_realtime/{source,run_live,check_signal}.py`，
只把 `source.py` 複製進 `semg_deploy/`，然後刪掉來源目錄。
`run_live.py` 與 `check_signal.py` 裡的 baud 還是舊的 **921600**。
使用者發現：「你還是沒有重新打包阿… 舊的你都沒修」。

**修法與制度**
建立規則：打包後**解壓到乾淨目錄**，跑 12 項驗證清單才算完成。

---

### 5.3 🟣 繪圖用錯欄位 ✅

**Bug**（使用者發現）
顏色用 `row["segment_type"]`（事先假設的循環位置標籤），
圖例卻寫「predicted class」。

**修法**：改用 `row["predicted"]`。

**教訓**：圖表的圖例與資料來源必須是同一個變數，這種錯不會報錯只會誤導。

---

### 5.4 🟣 延遲歸因錯誤 ✅

我說延遲主要來自 5 窗多數投票。實測拆解後：模型 0.70s（70%）、
後處理 0.30s（30%，正好是理論最小值 3×100ms）。已更正（見 3.3）。

**教訓**：延遲要拆解量測，不要從架構推論。

---

### 5.5 🟣 `np.savez_compressed` key 撞名 ✅

```python
out = {..., "file": np.asarray(src_file), ...}
np.savez_compressed(CACHE, **out)
# TypeError: savez_compressed() got multiple values for argument 'file'
```

`out` 有個 key 叫 `"file"`，與函式第一個參數 `file` 撞名。
**28 個檔案全部處理完（363 秒）才在最後存檔那行爆掉。**

**修法**：改名 `src_file`，並用小樣本先驗證存/讀再跑全量。

**教訓**：長時間任務的**最後一步**要先用小樣本測過。

---

### 5.6 🟣 參數散落各檔 ✅

**V1 狀況**
`RMS_WINDOW` 在 `run_model_validation`、`PHASE_SHIFT` 在別處、notch 頻率又在另一處。
改一個忘了同步另一個（5.2 就是這樣發生的）。

**V2 修正**
`src/semg/v2/config.py` 集中所有超參數。
特徵名、跨通道對、分組都從 config 推導 —— 改通道數不用動 `features.py`。

---

### 5.7 🟠 不知道什麼時候真的開始存資料 ✅

**V1 狀況**（使用者原話：「開啟 code 會延遲每次不一樣時間才出現圖，也不知道什麼時候會開始儲存值」）

`ads1299_readerV6.py`：
```python
t_start = time.perf_counter()          # ← 在 plt.show() 之前
...
ani = animation.FuncAnimation(fig, update, interval=50)
plt.show()                             # ← 第一次 update() 何時觸發不確定
```

三個獨立問題：
1. `t_start` 設在開窗之前，但第一筆資料在第一次 `update()` callback 才讀
   → 兩者相差多少取決於 matplotlib 開窗速度，**每次不一樣**
2. UART buffer 裡可能積了**開機前的舊資料**
3. 讀取寫在繪圖 callback 裡 → 畫圖卡住就可能 buffer 溢出掉樣本

**V2 修正**（`record_session.py`）

| 問題 | 解法 | 程式位置 |
|---|---|---|
| UART 有舊資料 | `ser.reset_input_buffer()` | 開埠後立刻 |
| 硬體未穩定 | 丟棄最初 `--warmup-sec`（預設 1s）樣本 | `reader_thread` |
| t0 不確定 | **t0 = 第一個被接受的樣本**的時刻 | `state["t0"]` |
| 掉樣本 | 讀取搬到**獨立 thread**，完全不碰繪圖 | `threading.Thread` |
| cue 對齊 | 等到 `state["t0"]` 有值才開始跑 cue 時程 | 「等待第一個有效樣本…」 |

而且 CSV 多寫一欄 `sample_index`，cue 檔記錄的就是這個索引。

**實測（`--dry-run`）**
```
等待第一個有效樣本… 收到！t0 已鎖定（等待 0.30s）
樣本數     : 28,107（預期 27,999）
平均取樣率 : 2000.0 Hz
壞幀       : 0（0.000%）
✅ 樣本完整度 100.4%
```

---

### 5.8 🟠 受試者不知道何時該做動作 ✅

**V1 狀況**
外部循環提示音，與資料無任何關聯記錄。

**V2 修正：先標註，再跟著標註做**

```
  trial  3/32  下一個動作 → 【食指+拇指 捏】  (sample 10031)
[████████····················]   14.0/154s  ★ 出力 ★  食指+拇指 捏  剩 1.2s  (3/32)  樣本 27945 壞幀 0 2000.0Hz
```

- terminal 單行更新（`\r` 覆寫），顯示進度、當前指示、倒數、即時取樣率、壞幀數
- `prepare` 事件先印指示 + terminal bell → 2 秒倒數 → `go` 再 bell
- **提示先寫進 cue 檔，受試者才動** → 標籤在動作之前就固定了

---

### 5.9 🟠 頭尾起始誤差 ✅

**V1 狀況**
只能事後用 `TRIM_START_SEC = 2.0` / `TRIM_END_SEC = 1.0` 硬切，
切多少全憑猜測，而且切了之後索引就對不上原始 CSV。

**V2 修正**
- 錄製時開頭 `CUE_LEAD_SEC = 5.0`、結尾 `CUE_TAIL_SEC = 5.0` **純休息不下 cue**
  → 同時當濾波器暖機區與基線估計區，而且這段是**真正的 rest 訓練資料**
- `load_recording(path, trim=False)` 在有 cue 時**不裁切**
  → `sample_index` 與 CSV 嚴格一一對應，不會對不上

---

### 5.10 🟣 誤報的問題：`PER_CH_NAMES` 長度不符

我曾懷疑 `PER_CH_NAMES`（以為 16 個名稱）與 `channel_features`（回傳 15 個值）不一致。
實際查核：**15 = 15，沒有問題**。寫在這裡是因為「懷疑但查核後不成立」也該留紀錄，
避免下次又花時間重查。

現在有自動檢查：
```
PER_CH_NAMES=15  channel_features 回傳=15  ✓
CROSS_NAMES=15   cross 回傳=15  ✓
FEATURE_NAMES=120  extract 回傳=120  ✓
```

---

## 6. 問題總表

| # | 類別 | 問題 | 類型 | 狀態 |
|---|---|---|---|---|
| 1.1 | 資料 | 沒有 trial 時間戳，靠猜邊界（6.1% vs 92.4% 假象） | 🟠 | ✅ cue 的 sample_index |
| 1.2 | 資料 | 一檔一手勢 → 手勢與檔案共線 | 🟠 | ✅ 一檔多手勢隨機交替 |
| 1.3 | 資料 | 固定循環 → 可猜「第幾秒」 | 🟠 | ✅ 隨機順序 |
| 1.4 | 資料 | 沒 metadata 就進 pipeline | 🟠🟣 | ✅ meta.json 強制產出 |
| 1.5 | 標籤 | 相位標籤三次嘗試全失敗 | 🔴 | ✅ 改從 cue 讀 |
| 1.6 | 標籤 | 跨邊界視窗汙染 | 🟣 | ✅ 95% 純度門檻 |
| 2.1 | 正規化 | z-score 對校準組成敏感（±8.8pp） | 🔴 | ✅ MAD + 尺度不變特徵 |
| 2.2 | 正規化 | 跨通道比值爆炸（47%）／逐通道正規化消訊號（d 3.88→0.12） | 🔴 | ✅ log 比值 + 總和歸一化 |
| 2.3 | 正規化 | 讀不到論文的離線正規化 | 🔴 | ✅ 自己訓練，決策明文 |
| 3.1 | 模型 | 7 類 checkpoint = 亂猜（32.3%）；MAX_NUM_TRIALS=1；斜率式標籤是不同任務 | 🔴 | ✅ 不用預訓練 |
| 3.2 | 模型 | 5 類互斥無法表示同時彎曲 | 🔴 | ✅ pinch/grasp 各自成類 |
| 3.3 | 模型 | 3s 視窗、延遲 3.3s | 🔴 | ✅ 0.5s 視窗、1.15s |
| 3.4 | 模型 | RMS 包絡線丟掉頻譜 | 🔴 | ✅ 從原始波形算 120 維 |
| 3.5 | 模型 | LSTM 只取最後隱狀態 | 🔴 | ✅ BiLSTM + Attention |
| 3.6 | 模型 | 我把目標壓成 3 狀態夾爪 | 🟣 | ✅ 輸出手勢，映射放下游 |
| 4.1 | 評估 | 隨機切窗 99.9% 洩漏 | 🔴 | ✅ trial 分組切分 |
| 4.2 | 評估 | 只報一個數字 | 🔴 | ✅ 四種 scheme 全報 |
| 4.3 | 評估 | 切分抽走整個類別（三人都 0.0%） | 🟣 | ✅ 每手勢各留 1 檔 |
| 4.4 | 評估 | 動態範圍當品質門檻失效（r=−0.40） | 🟣 | ✅ 直接量測 |
| 5.1 | 工程 | 封包格式猜錯（12 vs 40 byte） | 🟣 | ✅ 沿用 readerV6 |
| 5.2 | 工程 | 改了檔案沒重新打包 | 🟣 | ✅ 解壓驗證流程 |
| 5.3 | 工程 | 繪圖用 segment_type 而非 predicted | 🟣 | ✅ |
| 5.4 | 工程 | 延遲歸因錯誤 | 🟣 | ✅ 拆解量測 |
| 5.5 | 工程 | savez key 撞名，跑 363s 才爆 | 🟣 | ✅ 小樣本先測 |
| 5.6 | 工程 | 參數散落各檔 | 🟣 | ✅ config.py 集中 |
| 5.7 | 錄製 | 不知道何時真的開始存 | 🟠 | ✅ t0 = 第一個接受的樣本 |
| 5.8 | 錄製 | 受試者不知道何時該動 | 🟠 | ✅ terminal cue + bell |
| 5.9 | 錄製 | 頭尾起始誤差靠猜著切 | 🟠 | ✅ lead/tail 空白 + 不裁切 |
| 5.10 | — | （誤報）特徵名長度不符 | 🟣 | ✅ 查核後不成立 |

---

## 7. V1 → V2 對照速查

| 面向 | V1（SoftPINCH） | V2 |
|---|---|---|
| **通道** | 3（曲腕肌 / 食指 / 拇指） | **7**（+ 背闊肌 / 三角肌 / 三頭肌 / 二頭肌） |
| **進模型的東西** | 40 Hz RMS 包絡線 | **120 維多域特徵**（從濾波後原始波形） |
| **視窗** | 3.0s | **0.5s / 步進 50ms** |
| **正規化** | z-score（mu/sigma 絕對 mV） | **MAD 穩健尺度 + 尺度不變特徵** |
| **標籤來源** | 猜 trial 邊界 → 固定切三等分 | **cue 檔的 sample_index** |
| **相位** | 位置式（離線）／斜率式（realtime），兩者不同任務 | prepare 丟棄 / go = 手勢 / release = rest |
| **類別** | 5 類互斥（無 pinch） | **rest / index / thumb / pinch / grasp** |
| **模型** | Conv1D + 單向 LSTM 取最後步 | **Conv1D + BiLSTM(128) + Attention + FC(256)** |
| **參數** | 42,245（離線）／344,583（realtime） | 355,013 |
| **延遲** | ~3.3s | **1.15s** |
| **交叉驗證分組** | 隨機切窗（洩漏） | **trial 分組** |
| **報告的數字** | 一個 | **四個**（trial_kfold / within_session / within_subject / LOSO） |
| **錄製協定** | 單一手勢固定循環，無時間戳 | **一檔多手勢隨機交替 + cue + metadata** |
| **錄製起點** | 不確定（開窗延遲 + UART 舊資料） | **t0 = 第一個被接受的樣本** |
| **頭尾** | 事後硬切 2s/1s | 錄製時保留 5s/5s 空白 |
| **參數管理** | 散落各檔 | `config.py` 集中 |

---

## 8. 仍未解決的事

誠實列出來，避免以後把這些當成已解決。

| # | 未解問題 | 影響 | 需要什麼才能解 |
|---|---|---|---|
| 1 | **只有 3 位受試者** | LOSO 只有 3 折，任何泛化結論信賴區間都很寬 | 更多受試者。新協定下每人 **5 分鐘**（32 trial / 298s）就能錄一輪 |
| 2 | **跨 session 電極位移吃掉全部效能** | V1 實測 93% → 28.5%。V2 是否改善**尚未驗證** | 同一人多天錄製 + 電極位置照片；必要時做 domain adaptation |
| 3 | **V2 在真實資料上的準確率未知** | 合成資料 100% 只驗證管線正確，**不預測真實表現** | 新協定的第一批真實錄音 |
| 4 | **真實硬體上的 threaded `find_sync` 未驗證** | `--dry-run` 走模擬路徑，實際串列埠 timeout 與 buffer 行為不同 | 先 `--reps 1` 短跑，確認 meta.json 的樣本完整度接近 100% |
| 5 | **物件層級分類（使用者提的「like YOLO」）** | 目前只到手勢層級 | cue 協定要加入物件維度並重新錄製 |
| 6 | **校準協定「兩手勢」那組測試有混淆** | 兩段獨立錄音接起來讓 sigma 膨脹 1.4–2.4× | 需要「單一錄音內交替手勢」—— **新協定正好提供這個** |
| 7 | **相位輸出（onset/hold/release）** | V2 只輸出手勢，沒有相位 | cue 檔已有 prepare/go/release，要做相位頭隨時可加 |
| 8 | **`quality.py::score_and_classify` 去留未定** | 死碼風險 | 使用者說不重要，暫緩 |
| 9 | **Lite 6 `send_command()` 未接上** | 還不能實際驅動夾爪 | 硬體整合階段 |

---

## 附錄：相關檔案

```
docs/
├── v1-problems-and-v2-fixes.md     ← 本文件
├── pipeline-v2-design.md            設計決策與論文依據（第 6、7 節是實測結果）
├── project-context.md               論文方法論筆記、原始碼比對
└── papers/                          論文 PDF

src/semg/v2/
├── config.py       所有超參數（7 通道、cue 協定、模型、訓練）
├── preprocess.py   因果濾波 + 包絡線（trim 可關，供 cue 對齊）
├── features.py     120 維多域特徵（15×7 + 跨通道 15）
├── cues.py         ★ 從 cue 檔產生標籤 + trial 分組
├── labeling.py     舊資料的啟發法（保留供對照，預設不用）
├── dataset.py      切窗 + 抽特徵 + trial/LOSO 分割
├── model.py        Conv1D → BiLSTM → Attention → FC
├── train.py        四種評估 scheme + 最終部署模型
├── evaluate.py     混淆矩陣、訓練曲線、permutation 特徵重要度
└── infer.py        即時推論（串流前處理 + 投票）

src/hardware/
└── record_session.py   ★ 錄製 + 同步 cue（解 5.7 / 5.8 / 5.9）
```
