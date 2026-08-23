# sEMG Pipeline：現行版本程式說明與使用方法

> **這份文件只描述「現在的程式長什麼樣、每支程式做什麼、怎麼用」。**
> 為什麼變成這樣、歷史上試錯過什麼，請看 `semg-history-and-issues.md`。
>
> 對應版本：v2.5（姿勢框架 (4,2,2)）+ v2.6（固定順序 + preview + 架構對照）
> ⚠️ 所有參數以 `src/semg/v2/config.py` 為最終依據；本文件的數值為撰寫當下的值。

---

## 目錄

1. [總覽與資料流](#1-總覽與資料流)
2. [檔案結構與各程式的職責](#2-檔案結構與各程式的職責)
3. [硬體與資料格式](#3-硬體與資料格式)
4. [錄製協定與 `record_session.py`](#4-錄製協定與-record_sessionpy)
5. [標籤規則（`cues.py`）](#5-標籤規則cuespy)
6. [前處理（`preprocess.py`）](#6-前處理preprocesspy)
7. [特徵（`features.py`）](#7-特徵featurespy)
8. [正規化的兩個層次](#8-正規化的兩個層次)
9. [資料集建構（`dataset.py`）](#9-資料集建構datasetpy)
10. [模型（`model.py`）](#10-模型modelpy)
11. [訓練與評估（`train.py` / `evaluate.py`）](#11-訓練與評估trainpy--evaluatepy)
12. [即時推論與控制映射（`infer.py`）](#12-即時推論與控制映射inferpy)
13. [檢查與稽核工具](#13-檢查與稽核工具)
14. [端到端操作流程](#14-端到端操作流程)
15. [參數與數值速查](#15-參數與數值速查)

---

## 1. 總覽與資料流

```
ADC counts  z[n,c] ∈ ℤ         (int32, 8 通道中取 7)
   │ ×LSB
電壓  x[n,c] (mV)
   │ notch → bandpass → Hampel   (全部因果、狀態保留)
濾波  y[n,c] (mV)
   │ 滑動視窗 (N_w=1000, S=100)
視窗  W_t ∈ ℝ^{1000×7}
   │ φ: 120 維特徵
特徵  u_t ∈ ℝ^120
   │ 標準化（訓練集統計量）  v_t = (u_t − μ) ⊘ σ
   │ 堆疊 T=10 個
序列  V_t ∈ ℝ^{10×120}
   │ 神經網路 g_θ（11,625 參數）
輸出  3 個 DOF 的 logits，形狀 (4, 2, 2)
   │ softmax + 逐 DOF 多數決投票
狀態  ŝ = (hand, elbow, shoulder)
   │ 控制層映射
Lite 6 機械手臂指令
```

**關鍵性質**：整條鏈上**沒有任何非因果運算**。離線訓練與線上推論走完全相同的
`preprocess.StreamFilter`，已驗證逐位元相同（區塊 100/250/1000 樣本，最大差 0.000e+00）。

| 符號 | 意義 | 值 |
|---|---|---|
| $f_s$ | 取樣率 | 2000 Hz |
| $C$ | 通道數 | 7 |
| $N_w$ | 視窗長度 | 1000 樣本（500 ms） |
| $S$ | 步進 | 100 樣本（50 ms） |
| $T$ | 序列長度 | 10 視窗 |
| $F$ | 特徵維度 | 120 |
| $D$ | 自由度數 | 3 |
| $K_d$ | 各 DOF 狀態數 | (4, 2, 2) — hand / elbow / shoulder |

---

## 2. 檔案結構與各程式的職責

```
src/semg/v2/
├── config.py        所有超參數集中一處；含 validate() 與 segment_report()
├── preprocess.py    因果濾波鏈（StreamFilter）；離線與即時共用同一份實作
├── features.py      120 維多域特徵（每通道 15×7 + 跨通道 15）+ 檔案尺度估計
├── cues.py          ★ 從 cue 檔產生逐樣本標籤、視窗標籤、trial/segment 分組、校準區段
├── dataset.py       切窗 → 抽特徵 → 建 trial/segment 鍵 → 各種切分
├── model.py         TinyMultiHead（採用）+ ZhaoMultiHead（論文對照 baseline）
├── train.py         四種評估 scheme + 最終部署模型 + TorchScript 匯出
├── evaluate.py      混淆矩陣、訓練曲線、permutation 特徵重要度
├── infer.py         即時推論（串流前處理 + 逐 DOF 投票）
├── check_states.py  正式錄製前的電極／狀態可分性檢查
├── check_ext.py     串音／額外通道可分性檢查
├── audit_trials.py  ★ 逐 trial leave-one-trial-out 稽核，產生排除清單
└── labeling.py      舊資料的啟發法標籤（保留供對照，預設不用）

src/hardware/
├── record_session.py   ★ 錄製 + 同步 cue + meta.json
└── ads1299_readerV6.py 串列讀取邏輯的來源（find_sync / read_one_frame）
```

### 各程式一句話說明

| 程式 | 做什麼 |
|---|---|
| **`config.py`** | 唯一的參數來源。通道對應、cue 秒數、濾波參數、視窗、模型、訓練超參數、致動器語意全在這裡。改通道數不用動 `features.py`（特徵名、跨通道對、分組都從 config 推導）。錄製前會被呼叫做 `validate()` |
| **`preprocess.py`** | 提供 `StreamFilter`（保留狀態的因果濾波器）與 `filter_causal()`（把整段餵給同一個 StreamFilter）。**只有一份實作**，確保離線＝線上 |
| **`features.py`** | `channel_features()` 每通道 15 個、`cross_channel_features()` 跨通道 15 個、`extract()` 合成 120 維；`file_scales()` 從校準區段算 MAD 穩健尺度與死區門檻 |
| **`cues.py`** | 讀 `_cues.csv` → 逐樣本標籤陣列；`window_label()` 做 95% 純度判定；`calibration_span()` 取開頭空白段；產生 `trial` 與 `segment` 兩個分組鍵 |
| **`dataset.py`** | `discover()` 找出成對的錄音（判準是「有沒有配對的 `_cues.csv`」）；`build()` 切窗抽特徵；`trial_group_splits()` 等切分函式 |
| **`model.py`** | `TinyMultiHead`：深度可分離時間 CNN + attention pooling + 3 個頭（11,625 參數）。`ZhaoMultiHead`：論文架構改多頭，供 `--arch zhao` 對照 |
| **`train.py`** | 跑四種評估 scheme、算 balanced accuracy 與 exact match、輸出 `train_results.csv`、訓練最終部署模型並匯出 TorchScript |
| **`evaluate.py`** | 畫混淆矩陣、訓練曲線、permutation 特徵重要度 |
| **`infer.py`** | 即時推論：`calibrate()` 取尺度 → StreamFilter → 特徵 → 序列緩衝 → 模型 → 逐 DOF 投票 → 狀態輸出 |
| **`record_session.py`** | 錄製主程式。開埠、丟暖機、鎖 t0、跑 cue 時程、terminal 提示、寫三個輸出檔、印品質統計 |
| **`check_states.py`** | 2 分鐘短錄音，確認每個 DOF 的各狀態真的分得出來；偏低時給具體排查方向 |
| **`check_ext.py`** | 用真實錄音回答「兩個相鄰電極的串音夠不夠小」 |
| **`audit_trials.py`** | 逐 trial 交叉預測，找出「做錯的那幾次」，輸出 `_exclude.txt` 供人確認 |

---

## 3. 硬體與資料格式

### 3.1 通道配置（7 通道）

| ch | 肌肉 | 英文名 | 分組 | 在協定中的角色 |
|---|---|---|---|---|
| 1 | 背闊肌 | `latissimus_dorsi` | proximal | 肩拮抗（非主動肌，提供比值特徵） |
| 2 | 三角肌**前束** | `deltoid_anterior` | proximal | **肩屈曲主動肌** |
| 3 | 三頭肌 | `triceps` | proximal | 肘拮抗（非主動肌，提供比值特徵） |
| 4 | 二頭肌 | `biceps` | proximal | **肘屈曲主動肌** |
| 5 | 曲腕肌 | `wrist_flexor` | distal | 手部協同 |
| 6 | 食指屈肌 | `index` | distal | 手部協同 |
| 7 | 拇指屈肌 | `thumb` | distal | 手部協同 |
| 8 | — | — | — | **不使用**（`USE_EXTENSOR_CHANNEL=False`） |

> ⚠️ **ch2 必須貼三角肌前束**。前束 → 屈曲（往前舉）；中束 → 外展（往外舉）。
> 一顆電極分不出這兩者，選前束是因為目標場景「往前伸手取物」用的是屈曲。

### 3.2 量化

$$
\text{LSB} = \frac{2 V_{\text{ref}}}{G \cdot 2^{24}} = \frac{2 \times 4.5}{8 \times 16777216}\ \text{V}
= 6.7055\times10^{-5}\ \text{mV}
$$

$$x[n,c] = z[n,c] \cdot \text{LSB}$$

分子的 2 來自雙極性（滿刻度 $\pm V_{\text{ref}}/G$）。
載入後做**去 DC**（整段平均）：電極半電池電位會造成數百 mV 直流偏移，
遠大於 sEMG 本身，不去掉會讓濾波器暫態極長。

### 3.3 串列格式

| 項目 | 值 |
|---|---|
| Baud | **1,600,000** |
| Frame | **40 bytes** |
| SYNC | `0xA5 0x5A` |
| Payload | `struct "<I8i"`（uint32 status + 8×int32） |
| Checksum | XOR |
| END | `0x0D` |

`record_session.py` 的 `find_sync` / `read_one_frame` **直接沿用 `ads1299_readerV6.py` 的邏輯**。

### 3.4 輸出檔案（每次錄製三個）

| 檔案 | 內容 |
|---|---|
| `<stem>.csv` | 原始資料，含 `sample_index` 欄 |
| `<stem>_cues.csv` | cue 事件，記錄**當下的 sample_index** |
| `<stem>_meta.json` | 協定參數、姿勢規範、硬體設定、電極配置、品質統計 |

`_cues.csv` 格式：

```csv
sample_index,t_s,event,gesture,rep
1986,1.011182,prepare,index_flex,0      # 移動到目標姿勢 → 丟棄
2794,1.414845,go,index_flex,0           # 維持姿勢     → 標該姿勢
4010,2.020861,return,index_flex,0       # 移動回中性   → 丟棄
4617,2.322861,release,index_flex,0      # 中性放鬆     → 標 rest
```

取樣率固定 2000 Hz，`sample_index` 是**唯一無漂移的時間軸**（牆上時鐘會受排程抖動影響）。

`_meta.json` 結構：

```json
{
  "protocol": {"gestures": [...], "reps": 5, "order_seed": 41234,
               "order_randomised": false, "lead_sec": 5.0, "tail_sec": 5.0,
               "preview_sec": ..., "prepare_sec": 2.0, "action_sec": 4.0,
               "return_sec": 2.5, "rest_sec": 5.0, "warmup_discard_sec": 1.0},
  "pose_spec": {"neutral": "...", "airborne": "維持姿勢時手臂必須懸空 ...",
                "labelled_phases": "go(維持)=該姿勢, release(放鬆)=rest; "
                                   "preview/prepare/return(移動過程)=丟棄"},
  "hardware": {"sps": 2000, "gain": 8, "vref": 4.5, "lsb_mv": 6.7e-05, "baud": 1600000},
  "electrode_placement": {"ch1": {"muscle": "背闊肌", "name": "latissimus_dorsi"}, ...},
  "quality": {"n_samples": 596000, "n_bad_frames": 0, "duration_s": 298.0,
              "expected_duration_s": 298.0, "mean_fs_hz": 2000.0, "bad_frame_rate": 0.0},
  "notes": "..."
}
```

metadata 是**錄製流程的產出**，不是事後補的筆記。

---

## 4. 錄製協定與 `record_session.py`

### 4.1 自由度與動作

| DOF | 狀態 | 維持時的主動肌 | 通道 |
|---|---|---|---|
| **hand** | rest / index / thumb / pinch | 前臂屈肌群的不同協同 | ch5–7 |
| **elbow** | rest（伸直下垂）/ flex（彎曲 90° 懸空維持） | 肱二頭肌 | ch4 |
| **shoulder** | rest（下垂）/ flex（前舉 90° 懸空維持） | 三角肌前束 | ch2 |

→ `DOF_N_STATES = (4, 2, 2)`，組合數 16，cue 過 **8 種動作**。

動作由單一 DOF 與組合姿勢構成（例如 `index_flex`、`thumb_flex`、`elbow_flex`、`shoulder_flex`、
`pinch`、`arm_lift`、`pick_and_lift`、`pick_and_raise`）。實際清單以 `C.ACTIONS` 為準 —— **改一處全流程跟著變**。

### 4.2 單一 trial 的時序

```
rest(標 rest) → preview(丟棄) → prepare(丟棄) → go(標該姿勢) → return(丟棄) → rest…
5.0s            2.5s            2.0s            4.0s            2.5s
↑ 完全放鬆       ↑ 讀字判斷      ↑ 開始移動      ↑ **維持**      ↑ 回中性
                 身體還不動
```

| 階段 | 秒數 | 標籤 | 說明 |
|---|---|---|---|
| `preview` | 2.5 | **丟棄** | 讀字判斷，身體不動。丟棄是因為受試者已知道下一個是什麼、可能提前緊張，留在 rest 會汙染「完全放鬆」 |
| `prepare` | 2.0 | **丟棄** | 移動到目標姿勢 |
| `go` | 4.0 | **該姿勢** | **維持**目標姿勢 ← 訓練資料 |
| `return` | 2.5 | **丟棄** | 移動回中性 |
| `release` / rest | 5.0 | **rest** | 中性姿勢放鬆 ← 訓練資料 |

頭尾各 `CUE_LEAD_SEC = 5.0` / `CUE_TAIL_SEC = 5.0` **純休息不下 cue** ——
同時當濾波器暖機區、基線估計區、校準區段，而且這段是**真正的 rest 訓練資料**。

錄製順序**固定**（非隨機）。丟棄比例約 **65%**，一輪約 7.5 分鐘。

### 4.3 錄製時的可靠性機制

| 問題 | 解法 | 程式位置 |
|---|---|---|
| UART 有舊資料 | `ser.reset_input_buffer()` | 開埠後立刻 |
| 硬體未穩定 | 丟棄最初 `--warmup-sec`（預設 1s）樣本 | `reader_thread` |
| t0 不確定 | **t0 = 第一個被接受的樣本**的時刻 | `state["t0"]` |
| 掉樣本 | 讀取搬到**獨立 thread**，完全不碰繪圖 | `threading.Thread` |
| cue 對齊 | 等到 `state["t0"]` 有值才開始跑 cue 時程 | 「等待第一個有效樣本…」 |
| 參數設錯 | 錄製前呼叫 `config.validate()`，有問題就拒絕開始 | 啟動時 |

### 4.4 Terminal 顯示

```
▶▶ 下一個：捏住 + 彎肘 + 前舉 ◀◀   2.5s (3/40) 120/450s [#####......]
██ 維持：捏住 + 彎肘 + 前舉 ██     3.0s ...
→ 移動到：捏住 + 彎肘 + 前舉       2.0s ...
← 回中性                          1.5s ...
放鬆（完全不動）                   2.0s ...
```

- **動作名稱 + 倒數固定在最左邊**，進度條與統計放最右、空間不夠就整項略過
- `▶▶` 與 `██` 讓兩個需要立刻反應的階段跳出來
- 用 `unicodedata.east_asian_width` 正確計算中文雙寬字元，`shutil.get_terminal_size()` 取實際欄數
- 進度條字元用 ASCII `#`（Windows 預設字型不一定有 `█`）
- 實測 40/60/80/100/120/200 六種終端機寬度都不超寬

### 4.5 常用指令

```bash
# 列出可用串列埠
python src/hardware/record_session.py --list-ports

# 逐層診斷（收不到資料時用）
python src/hardware/record_session.py --port COM10 --probe

# 不接硬體，驗證時序與 cue 排程
python src/hardware/record_session.py --dry-run

# 正式錄製
python src/hardware/record_session.py --port COM9 --subject gary \
    --reps 5 --outdir data/raw/gary

# 短跑驗證（第一次接真實硬體建議先跑這個）
python src/hardware/record_session.py --port COM9 --subject gary --reps 1
```

### 4.6 `--probe` 的逐層判讀

| 檢查層 | 失敗時的判讀 |
|---|---|
| 埠在不在清單裡 | 埠名打錯 / 板子沒插 |
| 開得開嗎 | 被其他程式佔用（Arduino IDE）、驅動、權限 |
| 有沒有位元組進來 | 板子沒開始串流 / 供電 / TX-RX 接反 |
| 找不找得到 SYNC | **幾乎確定是 baud 不對**（會印出前 32 bytes） |
| checksum 過不過 | 幀格式與本程式假設不同 |
| 實測取樣率 | 偏差 >5% 會影響 cue 的 sample_index 對齊 |

> 註：pyserial 對 Windows `COM10` 以上會自動加 `\\.\` 前綴，所以埠號 ≥10 本身不是問題。

### 4.7 錄完立刻看的品質數字

```
樣本數     : 596,000（預期 596,000）
平均取樣率 : 2000.0 Hz
壞幀       : 0（0.000%）
✅ 樣本完整度 100.0%
```

低於 98% 會警告。這些是**定義明確**的量，不是相關性可疑的代理指標。

### 4.8 ★ 操作硬性前提：手臂必須懸空

> 維持姿勢時手臂靠在桌面、扶手或貼死身側 → 重力被外物支撐 →
> 肌肉不需出力 → **EMG 消失** → 系統判成 `rest` → 機械手臂放下。

**訓練與操作都必須懸空。** 這是整個姿勢框架唯一的硬性前提，
也是 `check_states.py` 在 `flex vs rest` 偏低時**優先提示「先確認手臂懸空」而不是「檢查電極」**的原因。

生理依據：維持一個對抗重力的姿勢，本身就需要持續的等長收縮。

| 姿勢 | 力學 | EMG |
|---|---|---|
| 手臂下垂（中性） | 重力就位，關節力矩 ≈ 0 | ≈ 0 → `rest` |
| 手臂前舉 90° 並維持 | 三角肌前束需產生 $\tau = m g L \cos\theta$ | 中等、**持平** → `flex` |
| 手肘彎曲 90° 並維持 | 二頭肌撐住前臂重量 | 中等、**持平** → `flex` |

等長收縮 → 500 ms 視窗內訊號**弱平穩** → 所有頻域特徵的前提成立。

---

## 5. 標籤規則（`cues.py`）

### 5.1 區間 → 逐樣本標籤

$$
L[n] = \begin{cases}
\Psi(a_k) & \text{go 段（維持目標姿勢）}\\
\mathbf{0} & \text{release 段（中性姿勢放鬆）}\\
\varnothing & \text{preview / prepare / return 段（移動過程，丟棄）}
\end{cases}
$$

其中 $\Psi$ 是動作名 → DOF 狀態組合的映射。

### 5.2 非對稱 margin

| 邊界 | 吸收什麼 | 值 |
|---|---|---|
| **go 起始** | 還沒擺好、還在微調（變異大） | **0.5 s** |
| **rest 起始** | 訊號從高活動衰減到基線 | **1.0 s** |
| **兩段結束** | 提前放鬆（通常很短） | **0.2 s** |

> **關鍵觀念：`CUE_RETURN_SEC + CUE_MARGIN_REST_START_SEC` 才是真正的衰減預算**（現為 3.5s，覆蓋 98%）。
> 剩下的 2% 尾巴交給 `audit_trials.py` 抓。

### 5.3 視窗標籤：95% 純度準則

視窗 $[s,e)$ 的標籤只在**整段一致**時才給：

$$
\ell_t = \begin{cases}
\mathbf{c}^* & \text{若 } \dfrac{|\{n \in [s,e) : L[n] = \mathbf{c}^*\}|}{e-s} \ge 0.95 \\
\varnothing & \text{否則（丟棄）}
\end{cases}
$$

```python
if cnt[j] / len(seg) < purity:
    return -1
```

**不用多數決是刻意的**：跨越 cue 邊界的視窗同時含 rest 與出力的混合波形，
多數決會把混合教成單一狀態，直接汙染 onset 的判別邊界。寧可丟掉也不要教錯。

### 5.4 兩個分組鍵，各司其職

| 鍵 | 用途 | 值 | 為什麼 |
|---|---|---|---|
| `segment` | `make_sequences` 序列堆疊 | `<檔名>#<rep>#<區間序號>` | 一個 trial 含 `go` 與 `release` 兩段**不連續**區間，中間隔著被丟棄的 `return`。用 trial 當鍵會把兩段黏成一條序列 |
| `trial` | `trial_group_splits` 交叉驗證 | `<檔名>#<rep>` | go 與 rest 時間相鄰、高度相關，不該拆到不同折 |

### 5.5 config 秒數的硬約束

一條序列涵蓋 `WIN_SEC + (SEQ_LEN-1)×STEP_SEC = 0.95 s`，
**任何有標籤的段落扣掉兩端 margin 後必須 ≥ 0.95 s。**

`make_sequences` 對「視窗數 < SEQ_LEN」的群組是靜默 `continue`，
所以 `config.validate()` 會在錄製前擋下：

```
❌ 時間參數有問題，錄了也用不了：
  • rest（中性放鬆） 只有 0.50s，扣掉 margin (0.5+0.2) 剩 -0.20s，
    不足一條序列所需的 0.95s（差 1.15s） → 該類別會靜默產生 0 條訓練序列
```

`config.segment_report()` 會印出每段能產生幾條序列（用**樣本數**算，與 `dataset.build` 的切窗迴圈一致）：

```
訓練資料 : 每個 trial 產生
           go（維持姿勢）   4.0s → 扣邊界 3.3s →  ... 視窗 →  ... 條序列
           rest（中性放鬆） 5.0s → 扣邊界 3.8s →  ... 視窗 →  ... 條序列
```

---

## 6. 前處理（`preprocess.py`）

三級串接，全部保留狀態以支援串流：

$$x \xrightarrow{\ H_{\text{notch}}\ } \xrightarrow{\ H_{\text{bp}}\ } \xrightarrow{\ \text{Hampel}\ } y$$

### 6.1 陷波器（60 Hz）

二階 IIR，$f_0 = 60$、$Q = 30$、$f_s = 2000$：

```
b = [ 0.996868, -1.958422,  0.996868]
a = [ 1.000000, -1.958422,  0.993736]
```

−3 dB 頻寬 = $f_0/Q$ = **2 Hz**。實測衰減：

| 頻率 | 50 Hz | 59 Hz | **60 Hz** | 61 Hz | 70 Hz |
|---|---|---|---|---|---|
| 衰減 | −0.04 dB | −2.97 dB | **−260 dB** | −3.05 dB | −0.05 dB |

> ⚠️ 量測時要在**精確頻率點**求值。用 `freqz` 預設 512 點網格（間距 3.91 Hz）會量到 −6.4 dB，那是量測假象。

$Q=30$ 的選擇：對電網頻率漂移（±0.5 Hz）在 59.5–60.5 Hz 仍有 ≥7 dB 衰減，是覆蓋率與選擇性的折衷。

> ⚠️ 台灣市電是 **60 Hz**，不是 50 Hz（MATLAB 分析腳本曾用錯）。

### 6.2 帶通（20–450 Hz）

Butterworth `butter(4, ...)` 帶通 → **實際 8 階**。

| 頻率 | 5 | 10 | **20** | 100 | 300 | **450** | 700 |
|---|---|---|---|---|---|---|---|
| dB | −49.4 | −25.1 | **−3.0** | 0.0 | −0.04 | **−3.0** | −30.0 |

- 下限 20 Hz：濾掉運動假影與電極位移造成的基線漂移（能量集中在 <20 Hz）
- 上限 450 Hz：sEMG 功率譜在 >400 Hz 已接近雜訊底線，且 $f_s/2 = 1000$ Hz 留足餘裕
- Butterworth：通帶最平坦，不會引入漣波扭曲振幅特徵

**群延遲**：通帶內 1.14–31.68 ms（中位數 1.46 ms）+ 陷波器 0.007 ms，總計 ≈ 1.47 ms。
最大值出現在 20 Hz 截止頻率附近。相對 500 ms 視窗可忽略。

### 6.3 因果 Hampel 濾波（脈衝雜訊）

視窗 $K = 201$（100 ms），**因果**視窗 $\mathcal{W}_i = \{i-K+1, \dots, i\}$（`origin=K//2`）：

$$
m_i = \operatorname{median}(y_{\mathcal{W}_i}), \quad
\hat{\sigma}_i = 1.4826 \cdot \operatorname{median}(|y_{\mathcal{W}_i} - m_i|)
$$

$$
y_i \leftarrow m_i \quad \text{若 } |y_i - m_i| > 2\hat{\sigma}_i
$$

- **1.4826**：高斯下 $\text{MAD} \approx 0.6745\sigma$，所以 $1.4826\cdot\text{MAD}$ 是 $\sigma$ 的穩健一致估計量
- **為什麼用 MAD 而不用標準差**：崩潰點 50% vs 0%。偵測離群值的統計量本身若被離群值影響，會發生 masking

### 6.4 ★ 串流實作的關鍵約束

Hampel 是**兩段式**的（先算 $m_i$，再對 $|y-m|$ 算中位數），
第二段會吃到第一段的輸出，所以串流時必須保留 $2(K-1) = 400$ 個歷史樣本。
只留 $K-1$ 會讓 MAD 的視窗吃到「用邊界填充算出來的 $m$」，與離線結果不同。

**只有一份實作**：`filter_causal()` 就是把整段餵給 `StreamFilter`。

初始化：$\xi_0 = \text{lfilter\_zi}(b,a) \cdot x[0]$ —— 用第一個樣本的穩態值，
避免從零開始的暫態（會產生一個假的 onset）。

---

## 7. 特徵（`features.py`）

$\varphi: \mathbb{R}^{1000 \times 7} \to \mathbb{R}^{120}$，每通道 15 個 × 7 + 跨通道 15 個。

### 7.1 核心設計原則：尺度不變

**目標**：對任意 $\alpha > 0$，$\varphi(\alpha W) = \varphi(W)$。

因為 sEMG 的絕對振幅受電極阻抗、皮膚狀態、脂肪厚度、電極位置影響，同一個人不同天可以差好幾倍。
若特徵依賴絕對振幅，模型學到的一部分就是「今天的電極貼得怎樣」。

做法：先用穩健尺度相對化 $x_s = x / \hat\sigma_{\text{cal}}$。
由於 MAD 一階齊次，$\hat\sigma(\alpha y) = \alpha\hat\sigma(y)$，所以任何**只依賴 $x_s$** 的特徵自動尺度不變。

**現況：0/120 違反尺度不變**（最大偏差 $8.4\times10^{-8}$，純浮點誤差）。

### 7.2 每通道 15 個特徵

令 $n = N_w$、$\Delta_i = x_{i+1} - x_i$、死區門檻 $\epsilon = 0.05\hat\sigma$。

**振幅類（5）**
$$\text{RMS},\quad \text{MAV},\quad \text{P2P},\quad
\text{WAMP} = \tfrac{1}{n}\sum \mathbb{1}[|\Delta_i| \ge \epsilon],\quad
\text{logRMS}$$

**形態類（4）**
$$\text{WL} = \tfrac{1}{n}\sum|\Delta_i|,\quad \text{SF} = \tfrac{\text{RMS}}{\text{MAV}},\quad
\text{ZC},\quad \text{SSC}$$

- **Shape Factor**：零均值高斯時 $\text{RMS}/\text{MAV} = \sqrt{\pi/2} \approx 1.2533$。偏離代表峰度改變（募集模式改變）。**無量綱**
- **ZC/SSC 的死區**：沒有門檻時靜息期的微小雜訊會產生大量假過零，讓 ZC/SSC 在 rest 反而比出力時還高。門檻設 5% 穩健標準差，隨訊號縮放
- ⚠️ SSC 的門檻必須是 $\epsilon^2$（量綱：$(\Delta_1\Delta_2)$ 是 mV²）

**頻域類（4）**（Welch PSD，`nperseg=512` → 解析度 3.91 Hz，通帶 110 bin）
$$\text{MNF},\quad \text{MDF},\quad \text{PKF},\quad \text{BW}$$

MNF 是功率譜一階矩、BW 是二階中心矩。四個都被 $\sum P_j$ 正規化 → **天然尺度不變**。

> 頻域特徵的意義：肌纖維傳導速度隨疲勞下降會讓功率譜往低頻移動（MNF/MDF 下降），
> 這是肌電學的標準疲勞指標。不同肌纖維類型的募集也會改變頻譜形狀。

**倒頻譜類（2）**：MFCC1（$c_0$）、MFCC3（$c_2$）

Mel 濾波器組 20 個三角窗，$\text{mel}(f) = 2595\log_{10}(1 + f/700)$，DCT-II 正交化。

> ⚠️ **必須餵 $x_s$ 而非 $x$**：$c_0$ 是總能量項，會隨振幅平移。
> MNF/MDF/PKF/BW 有正規化所以餵哪個都一樣，但 MFCC 不行。

### 7.3 跨通道 15 個特徵

令 $a_c$ 為第 $c$ 通道的原始 RMS：

```python
a = np.sqrt(np.mean(win ** 2, axis=0)) + 1e-12   # 每通道 RMS
out  = [la[i] - la[j] for i, j in _PAIR_IDX]      # 5 個 log 比值（拮抗對）
out += [v / tot for v in a]                       # 7 個通道佔比（總和歸一化）
out += [grp[g] / tot for g in C.CH_GROUPS]        # 2 個分組佔比
out.append(log(distal) - log(proximal))           # 1 個遠端/近端 log 能量比
```

**5 組拮抗對**：(index, thumb)、(index, wrist_flexor)、(thumb, wrist_flexor)、
**(biceps, triceps)**、**(deltoid_anterior, latissimus_dorsi)**。

- **用 log 比值而非直接比值**：$a_j \to 0$ 時比值爆炸，log 把除法變減法，值域受控
- **用總和歸一化而非逐通道 z-score**：逐通道各自正規化會把跨通道相對關係一起消掉
- **肘/肩拮抗對的作用**：「彎曲」與「伸直」的差別主要體現在兩者的**比值**而非各自絕對強度。
  ch1（背闊肌）與 ch3（三頭肌）在目前協定下不是主動肌，但維持姿勢時拮抗肌有共同收縮，比值能反映出力大小與姿勢穩定程度

### 7.4 自動一致性檢查

```
PER_CH_NAMES=15  channel_features 回傳=15  ✓
CROSS_NAMES=15   cross 回傳=15  ✓
FEATURE_NAMES=120  extract 回傳=120  ✓
```

---

## 8. 正規化的兩個層次

容易混淆，明確分開：

| 層次 | 統計量來源 | 目的 | 何時算 |
|---|---|---|---|
| **① 訊號尺度** $\hat\sigma_{\text{cal}}$ | 該次錄音的**開頭空白段** | 消除電極/皮膚造成的振幅差異 | 每次錄音一次 |
| **② 特徵標準化** $(\mu, \sigma)$ | **訓練集**的所有視窗 | 讓各特徵量級相近，利於梯度下降 | 訓練時算一次，凍結 |

$$u_t = \varphi(W_t;\ \hat\sigma_{\text{cal}}), \qquad v_t = \frac{u_t - \mu_{\text{train}}}{\sigma_{\text{train}} + 10^{-8}}$$

### ★ ① 一定要用開頭空白段

```python
mad = 1.4826 * np.median(np.abs(filt - med), axis=0) + 1e-12
return mad, 0.05 * mad      # 死區 = 5% 穩健標準差
```

訓練與推論**都**用 `cues.calibration_span()` 取的開頭空白段。
錄製協定保證前 5 秒純休息不下 cue，所以離線線上都拿得到、且**內容定義相同**。

> **原則：穩健性不能替代一致性。** 訓練/推論的每一個統計量都必須問：它在線上拿得到嗎？定義一樣嗎？

### ② 統計量只能從訓練集算

```python
sc = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[-1]))   # 只 fit 訓練集
```

否則測試集的資訊會透過標準化參數洩漏到訓練過程。

---

## 9. 資料集建構（`dataset.py`）

### 9.1 流程

```
discover()          找出成對的錄音（判準：有沒有配對的 _cues.csv）
   ↓
load_recording()    載入 CSV（有 cue 時 trim=False，sample_index 與 CSV 嚴格一一對應）
   ↓
filter_causal()     整段餵 StreamFilter
   ↓
file_scales()       從 calibration_span 算 MAD 尺度與死區
   ↓
切窗迴圈             N_w=1000, S=100
   ↓
window_label()      95% 純度判定，不純就丟
   ↓
extract()           120 維特徵
   ↓
建立 trial / segment 兩個鍵
```

### 9.2 分窗

第 $t$ 個視窗涵蓋樣本 $[tS,\ tS+N_w)$，重疊率 $1 - S/N_w = $ **90%**。

**為什麼 500 ms**：對頻寬 $B$ 的隨機訊號，用時間 $\tau$ 估 RMS 的相對標準誤約為 $1/\sqrt{2B\tau}$。
代入 $B = 430$ Hz、$\tau = 0.5$ s → **4.8%**。縮到 125 ms 變 9.7%（誤差加倍），
且 Welch 頻率解析度從 3.91 Hz 惡化到 15.6 Hz。

### 9.3 切分函式

```python
groups = np.unique(d["trial"])
rng.shuffle(groups)
for part in np.array_split(groups, n_folds):
    te = np.isin(tr_key, part)
```

- `trial_group_splits`：同一次出力的所有視窗一定在同一折
- `make_sequences`：**只在同一 `segment` 內堆疊**
- within-subject 切分：**每手勢各留 1 檔**，避免抽走整個類別

```python
for gi in range(n_cls):
    fg = np.unique(files[m & (y == gi)])
    if len(fg) < 2:
        ok = False; break        # 每手勢不足 2 檔 → 略過該受試者
    te_files.append(sorted(fg)[-1])
```

### 9.4 排除清單

`audit_trials.py` 產生的 `_exclude.txt` 由**人確認後** `dataset.py` 才套用。
排除是**整個 trial**，不是只遮蔽出錯的 DOF（通道之間耦合，且 15 個跨通道特徵本來就把所有通道混在一起）。

### 9.5 舊資料

`C.USE_LEGACY_DATA = False` —— 28 個舊錄音預設不進 pipeline。要做對照時再開啟。

---

## 10. 模型（`model.py`）

$$g_\theta:\ \mathbb{R}^{10 \times 120} \to \prod_{d=1}^{3} \mathbb{R}^{K_d}, \qquad (K_d) = (4,2,2)$$

### 10.1 `TinyMultiHead`（採用）

$$
\begin{aligned}
H^{(0)} &= \text{ReLU}(\text{BN}(\mathbf{W}_{\text{stem}} * V^\top)) && \in \mathbb{R}^{48 \times T}\\
H^{(b)} &= H^{(b-1)} + \text{Drop}(\text{ReLU}(\text{BN}(\mathbf{W}_{\text{pw}}^{(b)} * (\mathbf{W}_{\text{dw}}^{(b)} \circledast H^{(b-1)})))) && b = 1,2\\
\alpha &= \text{softmax}_t(\mathbf{w}_a^\top H^{(2)}) && \in \mathbb{R}^{T}\\
h &= \textstyle\sum_t \alpha_t H^{(2)}_{:,t} && \in \mathbb{R}^{48}\\
\ell^{(d)} &= \mathbf{W}_d \text{Drop}(h) + \mathbf{b}_d && \in \mathbb{R}^{K_d}
\end{aligned}
$$

逐層參數：

| 層 | 形狀 | 參數 |
|---|---|---|
| `stem.conv` (1×1, 120→48) | (48,120,1) | 5,760 |
| `stem.bias` + `BN` | | 144 |
| `block0.dw` (k=3, dilation=1) | (48,1,3) | 144 + 48 |
| `block0.pw` (1×1) | (48,48,1) | 2,304 + 48 |
| `block0.BN` | | 96 |
| `block1.dw` (k=3, dilation=2) | (48,1,3) | 144 + 48 |
| `block1.pw` (1×1) | (48,48,1) | 2,304 + 48 |
| `block1.BN` | | 96 |
| `attn.score` (1×1, 48→1) | (1,48,1) | 48 + 1 |
| 3 個頭 (48→4,2,2) | | 384 + 8 |
| **總計** | | **11,625** |

### 10.2 各設計元素

| 元素 | 理由 |
|---|---|
| **深度可分離卷積** | 參數比 $\frac{kC + C^2}{kC_{in}C_{out}} = 0.354$，**省 65%**；且把「時間上的模式」與「特徵間的混合」解耦 |
| **dilation 1, 2** | $\text{RF} = 1 + 2(1+2) = 7$ 個視窗 = 0.80s 原始訊號。用 dilation 只要 2 層，不用要 4 層 —— **層數直接決定 CPU 延遲**，所以省的是延遲不是參數 |
| **Attention pooling** | 動作可能發生在序列任何位置（onset 剛發生時關鍵資訊在開頭）。成本只有一個 48→1 線性層（49 參數）。優於「取最後一步」 |
| **多輸出（multi-head）** | 互斥多類只能輸出訓練時見過的組合；多輸出把聯合分布分解成 $\prod_d p(s_d \mid V)$，參數從 $\mathcal{O}(\prod K_d)$ 降到 $\mathcal{O}(\sum K_d)$（48×16=768 → 48×8=384） |
| **小模型（11.6k）** | 是為了**防過擬合，不是為了速度**（見下） |

**多頭分解的代價**：假設各 DOF 在給定訊號下條件獨立。真實情況並非如此，
但**相關性由共享 backbone 承載** —— $h$ 是所有頭共用的表示，只有最後線性層分開。

跨體段（手/肘/肩）的多輸出成立，因為它們由解剖上分離的肌群驅動。
跨體段的非線性交互靠**在 cue 裡明確錄到這些組合**（`arm_lift` / `pick_and_lift` / `pick_and_raise`）+ 共享 backbone 處理。

### 10.3 ★ 模型大小與延遲的實測（違反直覺）

CPU 實測（筆電 2 threads，batch=1，$T=10$）：

| 設定 | 參數 | 中位數 | p95 |
|---|---|---|---|
| Conv + BiLSTM(128) + Attn | 355,013 | 0.506 ms | 0.641 ms |
| blocks=1 width=32 | 5,547 | 0.348 ms | 0.362 ms |
| **blocks=2 width=48（採用）** | **11,625** | 0.522 ms | 0.585 ms |
| 同上 + TorchScript | 11,625 | **0.284 ms** | 0.292 ms |
| blocks=3 width=64 | 22,219 | 0.701 ms | 0.789 ms |

1. **參數量與延遲幾乎無關**。延遲只跟**層數**線性相關（1/2/3 層 = 0.35/0.52/0.70 ms）。
   原因：batch=1、$T=10$ 的規模下，每層 kernel dispatch 開銷（約 0.17 ms/層）遠大於實際浮點運算
2. **14k 參數的模型比 355k 的 BiLSTM 還慢**，因為層數更多

→ **實務準則：資料變多時調寬度，不要加層數。**

### 10.4 `ZhaoMultiHead`（對照 baseline，`--arch zhao`）

論文原架構是單一互斥多類，我們改成多頭；其餘（CNN 64@3 → BiLSTM 128 → Attention → FC 256）完全照論文。
另一處必要調整：**不做 MaxPool**（論文輸入時間軸 100 點，我們只有 SEQ_LEN=10，再 pool 一半剩 5）。

參數 355,784。

---

## 11. 訓練與評估（`train.py` / `evaluate.py`）

### 11.1 損失函數

$$
\mathcal{L}(\theta) = \sum_{d=1}^{3} \sum_{k=0}^{K_d-1} w_k^{(d)} \mathbb{1}[y_d = k] (-\log p_k^{(d)})
$$

**各頭等權相加**是刻意的：三個自由度在控制上同等重要。

類別權重（**逆頻率加權**，除以 $K_d$ 讓各頭損失量級相當）：

$$w_k^{(d)} = \frac{1}{K_d} \cdot \frac{\sum_{k'} n_{k'}^{(d)}}{n_k^{(d)} + \varepsilon}$$

必須加權：rest 在每個頭都是壓倒性多數（合成資料實測佔 61.2%），
不加權時「永遠輸出 rest」就是一個很好的局部最優。

### 11.2 最佳化設定

| 項目 | 設定 | 依據 |
|---|---|---|
| 最佳化器 | Adam | Lee et al. 2024 |
| 學習率 | $10^{-4}$ | 同上 |
| 權重衰減 | $10^{-5}$ | 小資料集需要正則化 |
| Batch | 64 | |
| Early stopping | 驗證損失 20 epoch 不降 | 防過擬合 |
| Dropout | 0.1（block）/ 0.1（head） | |

### 11.3 四個層次的評估（全部都報）

| Scheme | 訓練/測試的差異 | 回答什麼問題 |
|---|---|---|
| `trial_kfold` | 不同次出力，同 session | 模型學得動嗎（無洩漏的上界）**← 主要指標** |
| `within_session` | 同一天，時間前後切（70/30） | 電極位置固定時的表現 |
| `within_subject` | 同一人，不同天（電極重貼） | 跨 session 位移的代價 |
| `LOSO` | 完全沒見過的人 | 真正的泛化能力 |

**基準線必報**：「全部猜 rest」的全對率。沒有這個數字，90% 看起來很厲害，實際上可能還不如常數預測器。

### 11.4 指標

**Balanced accuracy**（每類召回率的平均）：
$$\text{BA} = \frac{1}{K}\sum_k \frac{\text{TP}_k}{n_k}$$

原始正確率在不平衡下沒有意義（實測 rest:ext = 1023:156 時，全猜 rest 的原始正確率 86.8%、BA 只有 50%）。

**Exact match**（全對率）：
$$\text{EM} = \frac{1}{N}\sum_i \mathbb{1}\Big[\bigwedge_{d=1}^{3} \hat{s}_d^{(i)} = s_d^{(i)}\Big]$$

這是控制上真正要的 —— 三個自由度同時正確才是一個正確的姿態指令。
若各頭獨立且各有準確率 $a_d$，則 $\text{EM} \approx \prod_d a_d$（三個 95% 只給 85.7%）。
**所以逐 DOF 準確率與全對率必須一起看。**

### 11.5 輸出

- `results/v2/train_results.csv`：四個 scheme 的數字
- `confusion_matrices.json` + `evaluate.py` 畫圖
- 逐類別 precision / recall、訓練曲線
- permutation 特徵重要度
- 最終部署模型（TorchScript）

⚠️ 所有 JSON 讀寫都必須 `encoding="utf-8"`（Windows 端 `write_text()` 預設用 cp950）。

### 11.6 指令

```bash
# 預設架構（tiny）
PYTHONPATH=src python src/semg/v2/train.py

# 論文架構對照
PYTHONPATH=src python src/semg/v2/train.py --arch zhao

# 評估與繪圖
PYTHONPATH=src python src/semg/v2/evaluate.py
```

---

## 12. 即時推論與控制映射（`infer.py`）

### 12.1 串流狀態

$$(y_{[m]},\ \xi_m) = \mathcal{F}(x_{[m]},\ \xi_{m-1})$$

- IIR 部分：`lfilter` 的 $\xi$ 是直接二型延遲線（每 stage 每通道各一份）
- Hampel 部分：$\xi$ 是最近 $2(K-1) = 400$ 個濾波後樣本

`calibrate()` 從開頭空白段取 MAD 尺度（與離線訓練同一個定義）。

### 12.2 逐 DOF 投票

每個自由度維持自己的 $V=5$ 窗緩衝：

$$k^* = \arg\max_k \sum_{j=1}^{5} \mathbb{1}[\arg\max p_j^{(d)} = k]$$

$$
\hat{s}_d \leftarrow \begin{cases}
k^* & \text{若 } \frac{1}{5}\sum_j p_{j,k^*}^{(d)} \ge 0.6\\
\hat{s}_d^{\text{prev}} & \text{否則（沿用上次的穩定值）}
\end{cases}
$$

**為什麼逐 DOF 而不是對整個組合投票**：對組合投票時，只要任一關節抖動整組就被否決
——「食指穩定彎曲但肩膀在抖」會連食指的輸出一起丟掉。逐 DOF 投票讓每個關節獨立穩定。

### 12.3 延遲拆解

$$
\Delta_{\text{total}} = \underbrace{N_w/f_s}_{0.50\ \text{s}} +
\underbrace{(T-1)S/f_s}_{0.45\ \text{s}} +
\underbrace{(V-1)S/f_s}_{0.20\ \text{s}} = 1.15\ \text{s}
$$

| 項 | 意義 | 能不能減 |
|---|---|---|
| 視窗 | 最新樣本要等視窗填滿 | 減短會讓 RMS/頻譜估計變不穩 |
| 序列 | 需要 $T$ 個視窗的上下文 | 減 $T$ 會失去時間動態資訊 |
| 投票 | 抑制單窗抖動 | 減 $V$ 會讓致動器抖動 |

計算延遲（19.63 ms）相對 1.15 s **可忽略** —— 要減延遲只能動上面三項的設計。

### 12.4 CPU 預算（每 50 ms 步進）

| 階段 | 時間 | 佔比 |
|---|---|---|
| 串流濾波（含因果 Hampel） | 14.02 ms | 71% |
| 120 維特徵抽取 | 3.81 ms | 19% |
| 模型（TorchScript） | 0.28 ms | **1.4%** |
| **合計** | **19.63 ms**（p95 20.28） | **/ 50 ms = 39%** |

特徵抽取原本 10.36 ms，優化後 3.77 ms（welch 每通道只算一次 + Mel 濾波器組快取）。
因果 Hampel 讓濾波從 ~0 變成 14 ms —— 這是離線/即時一致性的代價。

### 12.5 控制層映射（位置鏡像）

模型只輸出**狀態**，怎麼動由控制層決定（`C.ACTUATOR_SEMANTICS`）：

| DOF | 狀態 | 致動器指令 |
|---|---|---|
| hand | `rest` | 夾爪張開（回到預設位置） |
| | `index` / `thumb` / `pinch` | 對應夾爪閉合模式 |
| elbow / shoulder | `rest` | 維持現在角度 |
| | `flex` | 往該方向動到對應姿勢並停住 |

**控制範式是位置鏡像**：人維持在什麼姿勢 → 機械手臂移動到對應姿勢並停住。
不是速率控制（那需要「正在往哪個方向動」的標籤，而我們刻意不標移動過程）。

---

## 13. 檢查與稽核工具

### 13.1 `check_states.py`：正式錄製前的 2 分鐘檢查

多輸出的全對率 = 所有頭同時正確，**任何一個自由度失效都會拖垮整體**。

```bash
python src/hardware/record_session.py --port COM9 --subject gary \
    --gestures index_flex,thumb_flex,elbow_flex,shoulder_flex \
    --reps 3 --outdir data/raw/statecheck/gary

PYTHONPATH=src python src/semg/v2/check_states.py data/raw/statecheck/gary/multi_*.csv
```

判定標準：

| 分數 | 判定 |
|---|---|
| ≥85% | 可用 |
| 60–85% | 調整電極位置 |
| <60% | 該電極無效 |

工具內建的正確性保證（都是踩過坑後加的）：
- 按 **trial 分組**切分，不隨機切窗
- **只用其他 DOF 都 rest 的視窗**（否則分類器能靠「拇指安不安靜」作答）
- 用 **balanced accuracy**，不用原始正確率
- `flex vs rest` 偏低時**優先提示「先確認手臂懸空」**，而不是「檢查電極」

### 13.2 `check_ext.py`：串音檢查

用**真實錄音**回答「兩個相鄰電極的訊號差異夠不夠大」。
⚠️ 這個問題**合成資料回答不了**（合成訊號是純高斯雜訊，RMS 估計極穩，當然分得出來）。

### 13.3 `audit_trials.py`：逐 trial 稽核

**方法**：leave-one-trial-out 交叉預測，逐 DOF 做。
判斷 trial *t* 的模型只用**其他** trial 訓練，所以 *t* 自己的錯誤標籤不參與判斷它自己。

輸出範例：

| rep | cue | 稽核判定 |
|---|---|---|
| 0 | index_flex | 手部做錯（要 index → 判 pinch，3%） |
| 19 | elbow_flex | rest 沒放鬆（需 3.6s > 預算 3.5s） |
| 35 | elbow_flex | 肩膀做錯（要 rest → 判 flex，22%） |

**逐 DOF 比整體姿勢比對可靠**：整體法會誤報，逐 DOF 能直接說出**是哪個部位做錯**。

#### ★ 稽核範圍必須限定在同一次錄音內

| 範圍 | 用不用 | 理由 |
|---|---|---|
| 同一次錄音（5 次重複，留一比四） | ✅ 預設 | 電極位置完全相同 |
| 同一人跨 session | ⚠️ 僅第二意見 | 第二次會重貼電極 |
| 跨受試者混合 | ❌ 禁止 | 會把「這個人訊號不一樣」誤判成「他一直做錯」 |

**副產品**：若某份錄音標記率 > 25%，多半**不是**受試者一直做錯，而是**電極貼歪了**
—— 工具會直接這樣提示，並要求先跑 `check_states.py`。

**它報的是共識不是真理**：某姿勢 5 次錯 2 次且錯法一樣，共識會翻轉。
所以只產生清單（`_exclude.txt`），**由人確認後** `dataset.py` 才套用。

也會固定輸出反應時間統計（`t_reach`）—— 換人若系統性變慢就加長 `CUE_PREPARE_SEC`。

---

## 14. 端到端操作流程

```bash
# ── 0. 環境 ──────────────────────────────────────
pip install -r requirements.txt

# ── 1. 硬體連線檢查 ───────────────────────────────
python src/hardware/record_session.py --list-ports
python src/hardware/record_session.py --port COM9 --probe

# ── 2. 電極/狀態可分性檢查（約 2 分鐘）────────────
python src/hardware/record_session.py --port COM9 --subject gary \
    --gestures index_flex,thumb_flex,elbow_flex,shoulder_flex \
    --reps 3 --outdir data/raw/statecheck/gary
PYTHONPATH=src python src/semg/v2/check_states.py data/raw/statecheck/gary/multi_*.csv
#   → 任一 DOF <60% 就先調電極，不要往下走

# ── 3. 短跑驗證（第一次接真實硬體）────────────────
python src/hardware/record_session.py --port COM9 --subject gary --reps 1
#   → 確認 meta.json 的樣本完整度接近 100%

# ── 4. 正式錄製（約 7.5 分鐘）─────────────────────
python src/hardware/record_session.py --port COM9 --subject gary \
    --reps 5 --outdir data/raw/gary
#   → 錄製時務必保持手臂懸空

# ── 5. 逐 trial 稽核 ─────────────────────────────
PYTHONPATH=src python src/semg/v2/audit_trials.py data/raw/gary/multi_*.csv
#   → 人工確認 _exclude.txt 後才套用

# ── 6. 訓練 + 四種評估 ───────────────────────────
PYTHONPATH=src python src/semg/v2/train.py
PYTHONPATH=src python src/semg/v2/train.py --arch zhao   # 對照

# ── 7. 評估圖表 ─────────────────────────────────
PYTHONPATH=src python src/semg/v2/evaluate.py

# ── 8. 即時推論 ─────────────────────────────────
PYTHONPATH=src python src/semg/v2/infer.py --port COM9
```

### 真實資料到手後要跑的三件事

1. **架構對照**：`train.py`（tiny）與 `--arch zhao` 各跑一次
2. **SEQ_LEN 消融**：1 / 3 / 10 各跑一次，用準確率與延遲的取捨決定
3. **★ 固定順序的影響**：正式資料用固定順序錄，**另外錄一小段（約 2 分鐘）打亂順序的資料當獨立測試集**。
   用固定順序訓練、打亂順序測試 —— 這是固定順序下**唯一**能量化該影響的方法

---

## 15. 參數與數值速查

```
取樣          f_s = 2000 Hz，7 通道，LSB = 6.7055e-5 mV（Gain=8, VREF=4.5V）
串列          baud 1,600,000，40-byte frame，SYNC A5 5A，"<I8i"，XOR，END 0D

濾波          notch 60Hz Q=30（-260 dB @60Hz，-3dB 頻寬 2 Hz）
              bandpass 20–450 Hz，butter(4) → 實際 8 階
              Hampel K=201（因果，origin=K//2），κ=2σ，串流保留 2(K-1)=400 樣本
              總群延遲 ≈ 1.47 ms

分窗          N_w=1000 (500ms)，S=100 (50ms)，重疊 90%
標籤          視窗純度門檻 0.95
              margin：go 起始 0.5s / rest 起始 1.0s / 兩段結束 0.2s
              分組鍵：segment=<檔名>#<rep>#<區間序號>；trial=<檔名>#<rep>

特徵          120 維 = 15×7 + 15，秩 97（23 個線性相依）
              welch nperseg=512 → 3.91 Hz 解析度，通帶 110 bin
              死區 ε = 0.05×MAD（SSC 用 ε²）

序列          T=10 → 0.95 s 上下文
模型          11,625 參數，width=48，blocks=2，dilation 1/2，RF=7 視窗 (0.80 s)
              TorchScript CPU 0.284 ms (p95 0.292)，CPU_THREADS=2
訓練          Adam 1e-4，wd 1e-5，batch 64，early stop 20 epoch，dropout 0.1/0.1
              逆頻率類別加權，各頭等權相加

輸出          3 頭 (4,2,2) → 16 種組合，cue 過 8 種
投票          V=5 窗，逐 DOF 多數決，信心門檻 τ=0.6
延遲          0.50 (視窗) + 0.45 (序列) + 0.20 (投票) = 1.15 s
CPU 預算      濾波 14.02 + 特徵 3.81 + 模型 0.28 = 19.63 ms / 50 ms (39%)

協定          preview 2.5 / prepare 2.0 / go 4.0 / return 2.5 / rest 5.0
              lead 5.0 / tail 5.0（純休息不下 cue，兼校準區段）
              固定順序，8 姿勢 × 5 次 = 40 trial ≈ 7.5 分鐘
              衰減預算 = return 2.5 + rest 起始 margin 1.0 = 3.5s（覆蓋 98%）
              丟棄比例 ≈ 65%
```

### 硬約束檢查表

| 約束 | 值 | 違反的後果 |
|---|---|---|
| 有標籤段落扣 margin 後 ≥ 一條序列 | 0.95 s | `make_sequences` **靜默丟掉**該群組 → 該類別 0 條訓練序列 |
| `CUE_RETURN_SEC + CUE_MARGIN_REST_START_SEC` | ≥ 3.5 s | rest 標籤被高活動訊號汙染 |
| 維持姿勢時手臂懸空 | 必要 | EMG 消失 → 判成 rest → 機械手臂放下 |
| 校準區段的定義 | 訓練與推論必須相同 | 尺度偏移（實測 1.27× → 全對率 99.6% 掉到 58%） |
| 前處理實作份數 | **只能有一份** | 離線與線上不一致（實測相關係數只有 0.85） |
| JSON 讀寫編碼 | `encoding="utf-8"` | Windows 錄、Linux 分析時 UnicodeDecodeError |
