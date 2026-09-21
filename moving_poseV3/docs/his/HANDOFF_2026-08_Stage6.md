# sEMG → 機械手臂即時控制系統 —— 交接文件

**版本**：2026-08 / Stage 6 尾聲
**專案 deadline**：2026/9/4
**文件目的**：讓接手者在不重讀全部原始碼的前提下，理解目前的參數、資料流與待辦

---

## 0. 讀這份文件之前 —— 權威順序

當不同來源的參數互相矛盾時，依此順序採信：

```
1. 實際燒錄進 STM32 的韌體        （最高）
2. src/semg/v2/config.py
3. src/control/arm_config.py
4. 本文件
5. README.md / 程式碼註解 / 舊交接文件   （最低，已知有 v1 殘留）
```

> ❌ **任何寫著 `0x40 = Gain 4`、`LSB = 134 nV`、`4kSPS / 8kSPS`、`115200 baud`、
> 「RMS envelope + threshold 分類」的舊敘述都是 v1 遺跡，全部作廢。**

---

## 1. 系統總覽

### 1.1 資料鏈路

```
上肢表面電極 (7ch, 差動)
      │
      ▼
TI ADS1299EEG-FE Rev A          8ch × 24-bit ΔΣ，Gain=8，2000 SPS
      │  SPI Mode 1 (CPOL=0/CPHA=1), SCLK = 1 MHz, 27 bytes/sample
      │  DRDY → PB0 / EXTI0
      ▼
STM32L475VGT6 (B-L475E-IOT01A)
      │  EXTI → SPI1 DMA(27B) → ParseFrame → USART1 DMA(40B frame)
      │  USART1 PB6(TX)/PB7(RX), 1,600,000 baud, 80 kB/s
      ▼
PC / Python
      │  StreamFilter → 特徵 → 多輸出 CNN → Recognizer 投票
      ▼
控制層 src/control/
      │  DofDebouncer → 目標角追蹤 → 節流
      │  Ethernet TCP (xArm SDK)
      ▼
UFACTORY Lite 6 (6-DOF) + Lite 6 專屬夾爪
```

### 1.2 控制範式

**多輸出姿勢分類 + 位置鏡像**，不是比例控制、不是速度控制。

三個自由度各自獨立輸出（three-head），**不是 8 類 argmax**：

| DOF | 狀態 | 手臂範式 | 動作 |
|---|---|---|---|
| `hand` | `rest` / `index` / `thumb` / `pinch` | 混合 | 見下 |
| `elbow` | `rest` / `flex` | 位置鏡像 | J3 → 180° / 90° |
| `shoulder` | `rest` / `flex` | 位置鏡像 | J2 → −90° / 0° |

`hand` 三種非 rest 狀態的語意不同：

- `pinch` → **邊緣觸發 toggle**，每次 rest→pinch 上升緣切換夾爪（閉 ↔ 開）
- `index` → J6 往 **+** 方向持續 jog
- `thumb` → J6 往 **−** 方向持續 jog
- `rest` → J6 停在當前角度不回歸；**夾爪不動**（放鬆時東西不能掉）

⚠️ J6 是唯一有累積狀態的軸，會漂移；其餘軸是無狀態純函數映射，每幀可重算。

### 1.3 8 個 cue ACTION

由 (hand, elbow, shoulder) 的組合中取 8 種錄製。完整定義在
`config.ACTIONS` 與 `config.POSE_SPEC["actions"]` —— **要改姿勢清單必須同時改這兩處**，
錄製 / 訓練 / 推論會自動跟著變。

---

## 2. Stage 狀態（2026-08 更新）

| Stage | 內容 | 狀態 | 驗收依據 |
|---|---|---|---|
| 1 | ADS1299 Device ID 讀取 | ✅ | `g_ads1299_id == 0x3E` |
| 2 | DRDY EXTI 驗證 | ✅ | 中斷頻率 = CONFIG1 設定的 DR |
| 3 | 27-byte SPI DMA 讀取 | ✅ | `g_ads_dma_sample_count` 遞增、overrun = 0 |
| 4 | UART DMA 40-byte frame 串流 | ✅ | PC 端連續收 frame、checksum 全過 |
| 5 | sEMG 暫存器切換 + 硬體波形確效 | ✅ **已通關** | 貼片實測波形乾淨，收縮/靜息對比明顯 |
| 6 | Python 分類管線 | 🟡 **實質完成，品質待均化** | 模型可訓練、可即時預測；**但各錄音檔準確率不齊**，非全部達 99% |
| 7 | 機械手臂介面 | 🟡 **程式碼已完成，待實機整合** | `pytest tests/` 全綠；`--dry-run` 通過；**尚未接真手臂** |

### 剩餘工作（依優先序）

1. **實機整合**：`--source mock --arm-ip 192.168.1.181`，先驗證手臂會動且限位正確
2. **端到端延遲下修**（見 §7，目前理論值 ≈ 1.5 s，偏高）
3. **準確率均化**：找出低準確率錄音檔的共同因素（見 §8.3）
4. 修正 `BIAS_SENSP/N`（見 §9 ⚠️-1）
5. 釐清 ch2 / ch4 電極對應衝突（見 §9 ⚠️-2）

---

## 3. 硬體層

### 3.1 接線表（已確認）

| 訊號 | STM32 腳位 | 接頭位置 |
|---|---|---|
| SPI1_SCK | PA5 | CN1 |
| SPI1_MOSI (DIN) | PA7 | CN1 |
| SPI1_MISO (DOUT) | PA6 | CN1 |
| ADS_CS | PA4 | CN3 pin 8 |
| ADS_DRDY | PB0 / EXTI0 | CN3 pin 4 |
| ADS_RESET | PB1 | CN3 pin 7 |
| ADS_START | PB3 | CN3 pin 6 |
| ADS_PWDN | PB2 | CN1 pin 1 |
| USART1_TX | PB6 | → PC (USB-TTL) |
| USART1_RX | PB7 | ← PC |

> PA4/PA5 被 CS / SCK 佔用，**MCU 的 DAC 不可用**。訊號注入測試必須用外部 AWG。

### 3.2 電極配置（7 通道，proximal → distal）

`config.py CH_NAMES` 的宣告值：

| ch | 名稱 | 中文 | 角色 |
|---|---|---|---|
| 1 | `latissimus_dorsi` | 背闊肌 | 肩伸展（拮抗） |
| 2 | `deltoid_anterior` | 三角肌前束 | 肩屈曲主動肌 ⚠️ 必須是**前束**，非中束 |
| 3 | `triceps` | 三頭肌 | 肘伸展（拮抗） |
| 4 | `biceps` | 二頭肌 | 肘屈曲主動肌 |
| 5 | `wrist_flexor` | 曲腕肌 | 手部協同 |
| 6 | `index` | 食指屈肌 | 手部協同 |
| 7 | `thumb` | 拇指屈肌 | 手部協同 |
| 8 | — | 未使用 | `USE_EXTENSOR_CHANNEL = False`，CHxSET = `0x81`（PD + 短路） |

🔴 **ch2 / ch4 有未解決的衝突**：使用者口述的實體貼法是 ch2=二頭肌、ch4=三角肌前束，
與程式碼相反。對分類準確率**無**影響（模型吃全部通道），但會讓
`config.CROSS_PAIRS` 的兩組拮抗對算錯東西：

```python
("biceps", "triceps")                       # 宣稱是肘的屈/伸拮抗
("deltoid_anterior", "latissimus_dorsi")    # 宣稱是肩的屈/伸拮抗
```

若實體貼法是右欄，這兩個 log ratio 實際橫跨兩個關節、無拮抗意義。
**修正方式二擇一**：對調 `CH_NAMES` 的 ch2/ch4，或對調電極線。不要用猜的。

### 3.3 供電與參考

- 單極供電：AVDD = 5 V, AVSS = 0 V
- 共模參考必須是 **AVDD/2 ≈ 2.5 V**，否則雙極訊號的負半週被截斷
- CONFIG3 = `0xEC` → 內部 4.5 V 參考 + BIAS amp 啟用
- MISC1 = `0x00` → **SRB1 關閉**（本專案用真差動輸入，不是共同負端）

### 3.4 ⬜ 待補：EVM 跳線實體狀態

**這一節只有看得到板子的人能填。** 交接前請補完：

| 跳線 | 目前狀態 | 用途 |
|---|---|---|
| J6 (pins 5–36) | ⬜ | 拔除 = 允許外部差動輸入（EVM UG §4.6.1）。已知必須拔除 |
| JP7 / JP8 (SRB1 / SRB2) | ⬜ | 本專案 MISC1=0x00 不用 SRB1，實體應對應 |
| JP25 (Dedicated Ref & Bias) | ⬜ | |
| JP1 / JP6 | ⬜ | 已知：JP1 裝上時 JP6 必須拔除，否則 U11 與 BIASOUT 對打 |
| JP22 | ⬜ | 影響 START 訊號來源，錯了會導致暫存器寫不進去 |

| 電極 | 接到哪根 pin | |
|---|---|---|
| ch1–ch7 IN+ / IN− | ⬜ | |
| BIAS / RLD 電極 | ⬜ | |
| AWG / 外部儀器共地 | ⬜ | |

---

## 4. ADS1299 暫存器設定（完整）

### 4.1 實際寫入值（`ADS1299_Init()`）

| 暫存器 | 位址 | 寫入值 | 意義 |
|---|---|---|---|
| CONFIG1 | 0x01 | **`0x93`** | DR=011 → **2000 SPS**，內部時鐘輸出關閉 |
| CONFIG2 | 0x02 | `0xC0` | 正常模式，內部測試訊號關閉 |
| CONFIG3 | 0x03 | `0xEC` | PD_REFBUF=1（內部 4.5 V 參考）+ BIASREF_INT=1 + PD_BIAS=1 |
| CH1–CH7SET | 0x05–0x0B | **`0x40`** | **Gain = 8**，MUX=000（normal input） |
| CH8SET | 0x0C | `0x81` | PD=1（通道關電）+ MUX=001（內部短路） |
| BIAS_SENSP | 0x0D | ⚠️ `0x10` | **快照值 = 僅 CH5。應為 `0x7F`（CH1–CH7）** |
| BIAS_SENSN | 0x0E | ⚠️ `0x10` | 同上。SENSP 必須等於 SENSN |
| MISC1 | 0x15 | `0x00` | SRB1 關閉 |

### 4.2 GAIN 對照（datasheet Table 17，CHnSET bits[6:4]）

```
000 = 1    001 = 2    010 = 4    011 = 6
100 = 8    101 = 12   110 = 24
```

所以 `0x40` → bits[6:4] = 100 → **Gain 8**。
（任何說 `0x40 = Gain 4` 的舊註解都是錯的。）

### 4.3 尺度換算

```
LSB = 2 × VREF / (GAIN × 2^24)
    = 2 × 4.5 / (8 × 16777216)
    = 6.706e-5 mV  ≈  67.06 nV

滿刻度 = ± VREF / GAIN = ± 562.5 mV
```

sEMG 典型振幅 0.1–10 mV，Gain=8 留有充分餘裕，實測未見飽和截斷。

### 4.4 CONFIG1 可選值（`ads1299.h`）

| 巨集 | 值 | SPS |
|---|---|---|
| `ADS_CONFIG1_4KSPS` | 0x92 | 4000 |
| **`ADS_CONFIG1_2KSPS`** | **0x93** | **2000 ← 目前使用** |
| `ADS_CONFIG1_1KSPS` | 0x94 | 1000 |
| `ADS_CONFIG1_500SPS` | 0x95 | 500 |
| `ADS_CONFIG1_250SPS` | 0x96 | 250 |

⚠️ `ads1299.h` 第 129 行的註解寫「0x92（sEMG 模式）」是 v1 殘留，
實際 `ADS_CONFIG1_INIT_VALUE = ADS_CONFIG1_2KSPS`。

### 4.5 CHxSET 巨集（單行切換設計）

```c
#define ADS_CHSET_BRINGUP   0x61   /* Gain=24, MUX=001 內部短路（測雜訊底） */
#define ADS_CHSET_NORMAL    0x60   /* Gain=24, MUX=000 */
#define ADS_CHSET_TESTSIG   0x45   /* Gain=8,  MUX=101 內部測試方波 */
#define ADS_CHSET_CLOSE     0x81   /* PD=1,    MUX=001 通道關閉 */
#define ADS_CHSET_ECG       0x30   /* Gain=6,  MUX=000 */
#define ADS_CHSET_SEMG      0x40   /* Gain=8,  MUX=000  ← 目前使用 */
#define ADS_CHSET_INIT_VALUE ADS_CHSET_SEMG
```

除錯時改這一行即可切換全通道模式，不需要動 `.c`。

### 4.6 SPI 時序（這是硬性約束，不是偏好）

```
PCLK2 = 16 MHz（HSI 直驅，無 PLL）
SPI_BAUDRATEPRESCALER_16  →  SCLK = 1 MHz
27 bytes × 8 bits = 216 SCLK = 216 µs
DRDY 週期 @2000 SPS = 500 µs
餘裕 = 284 µs（57%）
```

⚠️ 若把 prescaler 改回 64（SCLK 250 kHz），單次讀取需 864 µs > 500 µs，
**每一個 sample 都會 overrun**。這個值不能動。

### 4.7 初始化序列（`ADS1299_Init()`）

```
START pin 拉低（停止轉換，解鎖暫存器寫入權限）
  ↓
PWDN 拉高 → RESET 拉高 → HAL_Delay(100)   ← ⚠️ 見 §9 ⚠️-3
  ↓
RESET 拉低 2 ms → 拉高 → HAL_Delay(20)
  ↓
送 SDATAC（0x11）+ 強制排空 STM32 SPI RX FIFO
  ↓
讀 REG_ID，必須 == 0x3E，否則 return ADS_STATUS_ERROR
  ↓
寫 CONFIG1 / CONFIG2 / CONFIG3 → HAL_Delay(150)（等參考電壓穩定）
  ↓
寫 CH1–CH8SET / BIAS_SENSP / BIAS_SENSN / MISC1
  ↓
readback 驗證 CONFIG1 與 CONFIG3，不符則 return ERROR
  ↓
送 RDATAC（0x10）→ 進入連續讀取模式
```

**SDATAC / RDATAC 的規則**：RDATAC 模式下暫存器**不可寫入**。
任何要改暫存器的操作都必須先 `SDATAC` → 改 → `RDATAC`。

---

## 5. 韌體層

### 5.1 檔案分工

| 檔案 | 來源 | CubeMX 重生成會覆蓋？ |
|---|---|---|
| `ads1299.c/.h` | 手寫 | ❌ 安全 |
| `uart_stream.c/.h` | 手寫 | ❌ 安全 |
| `main.c` | CubeMX + USER CODE | ⚠️ **會還原非 USER CODE 區塊** |
| `gpio.c` | CubeMX | ⚠️ **完全覆蓋** |
| `spi.c` / `usart.c` / `dma.c` | CubeMX | ⚠️ 覆蓋 |
| `stm32l4xx_it.c` | CubeMX + USER CODE | ⚠️ 曾產生重複 IRQ handler |

> **重生成前務必備份所有 `USER CODE BEGIN/END` 區塊內容。**

### 5.2 中斷資料流

```
DRDY 下降緣
  → EXTI0_IRQHandler
  → ADS1299_StartReadDataDMA()      [若 g_ads_streaming_enabled]
  → HAL_SPI_TransmitReceive_DMA(27 bytes)
  → HAL_SPI_TxRxCpltCallback()
  → ADS1299_ParseFrame()            [status 3B + 8ch × 3B → int32]
  → g_ads_frame_ready = 1

main while(1)
  → 偵測 g_ads_frame_ready
  → __disable_irq() 臨界區內複製快照 → __enable_irq()
  → UART_Stream_SendFrame_DMA()
```

**臨界區的必要性**：DMA callback 可能在主迴圈複製到一半時覆寫
`g_ads_dma_ch_data[]`，造成同一 sample 內混到兩個時刻的資料。

**背壓策略**：若上一筆 UART DMA 未完成（`g_uart_tx_busy == 1`），
本筆**靜默丟棄**並累計 `g_uart_tx_overrun_count`，不阻塞 SPI 採集。
採集永遠優先於傳輸。

### 5.3 UART frame 格式（40 bytes）

```
byte  0–1   : SYNC       0xA5 0x5A
byte  2–37  : payload    struct "<I8i"  = status(uint32) + 8 × int32
byte  38    : checksum   payload 的 XOR
byte  39    : END        0x0D
```

```
資料率 = 40 B × 2000 Hz = 80,000 B/s
鏈路容量 = 1,600,000 / 10 = 160,000 B/s
佔用率 = 50%
```

### 5.4 USART1 設定

```c
huart1.Init.BaudRate     = 1600000;
huart1.Init.OverSampling = UART_OVERSAMPLING_8;   // ← 必須
```

⚠️ 16 MHz HSI 下，`UART_OVERSAMPLING_16` **無法**產生 1,600,000 baud。
必須用 OVERSAMPLING_8（USARTDIV = 10）。

### 5.5 SPI1 設定

```c
hspi1.Init.CLKPolarity       = SPI_POLARITY_LOW;    // CPOL = 0
hspi1.Init.CLKPhase          = SPI_PHASE_2EDGE;     // CPHA = 1  → Mode 1
hspi1.Init.DataSize          = SPI_DATASIZE_8BIT;
hspi1.Init.FirstBit          = SPI_FIRSTBIT_MSB;
hspi1.Init.NSS               = SPI_NSS_SOFT;        // CS 由 PA4 軟體控制
hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_16;
```

### 5.6 除錯全域變數（Live Expressions）

| 變數 | 用途 |
|---|---|
| `debug_boot_stage` | 開機進度標記（5 = 進 AFE 初始化，8 = 全通關） |
| `debug_error_code` | 201 = ID 讀取失敗 / 線路斷開 |
| `g_ads1299_id` | 應為 `0x3E` |
| `g_debug_config1` / `g_debug_config3` | 暫存器 readback 值 |
| `g_ads_dma_sample_count` | 成功解析的 sample 數 |
| `g_ads_dma_overrun_count` | SPI DMA 來不及的次數，應為 0 |
| `g_spi_error_count` | SPI 錯誤 |
| `g_uart_tx_success_count` | UART 成功送出數 |
| `g_uart_tx_overrun_count` | UART 忙碌丟棄數，應為 0 |
| `g_ads_streaming_enabled` | 串流防護旗標 |

**Stage 驗證原則（staged verification）**：每一階段用上表確認通過才進下一階段，
避免多層 bug 疊在一起無法定位。

---

## 6. Python 訊號處理管線（v2）

### 6.1 前處理（`preprocess.StreamFilter`）

```
60 Hz notch (Q = 30)          ← 台灣電網。論文原文是丹麥 50 Hz，不要照抄
  ↓
bandpass 20–450 Hz, order 4
  ↓
causal Hampel filter (half-window 100 samples = 50 ms, σ = 2.0)
```

> ⚠️ **前處理只有一份實作。** 離線的 `filter_causal()` 就是把整段餵給
> `StreamFilter`，已驗證逐位元相同。**不要另外寫第二份**——訓練與推論的
> 前處理只要有一點差異，準確率就會崩。

`filtfilt` 只用於**離線分析**（MATLAB `semg_analysisV5.m`）；
即時路徑一律 `lfilter` + persistent state。

### 6.2 視窗與序列

| 參數 | 值 | 說明 |
|---|---|---|
| `FS` | 2000 | 與韌體 CONFIG1 一致 |
| `WIN_SEC` | 0.5 | 500 ms 特徵視窗 |
| `STEP_SEC` | 0.05 | 50 ms 步進 → **輸出 20 Hz**，overlap 450 ms |
| `WIN_SAMPLES` | 1000 | |
| `STEP_SAMPLES` | 100 | |
| `SEQ_LEN` | 10 | 堆疊 10 個視窗 → 涵蓋 **0.95 s** 上下文 |
| `TRIM_START_SEC` | 2.0 | 避開濾波器邊際效應 |
| `TRIM_END_SEC` | 1.0 | |

**硬約束**：任何有標籤的段落，扣掉兩端 margin 後必須 ≥ 0.95 s
（原始長度 ≥ 1.65 s）。`make_sequences` 對太短的群組是**靜默 `continue`** ——
不報錯、不警告，直到你發現某個類別完全沒有訓練資料。
因此 `record_session.py` 開錄前會強制跑 `config.validate()`，不過就拒絕開始。

### 6.3 模型

| 參數 | 值 |
|---|---|
| 架構 | depthwise-separable temporal CNN |
| `MODEL_WIDTH` | 48 |
| `MODEL_BLOCKS` | 2（dilation 1, 2 → 感受野 7） |
| `MODEL_KERNEL` | 3 |
| `MODEL_DROPOUT` | 0.1 |
| 參數量 | ~11,625 |
| 匯出格式 | TorchScript |
| `CPU_THREADS` | 2（筆電還要同時收樣本 + 控制手臂） |

對照組（同資料、同評估流程）：CNN + BiLSTM + Attention，
存成 `dof_model_zhao.pt`，用於論文比較，不是主線。

### 6.4 CPU 即時預算（實測，筆電 2 threads）

| 階段 | 每 50 ms 步進 |
|---|---|
| 串流濾波（含因果 Hampel） | 14.02 ms |
| 120 維特徵抽取 | 3.81 ms |
| 模型推論 | 0.28 ms |
| **合計** | **19.63 / 50 ms = 39%** |

計算不是瓶頸，**演算法結構才是**（見 §7）。

### 6.5 錄製協定 v2

```
preview 1.5s  「▶▶ 下一個：手肘彎曲 ◀◀」   → 丟棄（看到、讀懂、想好）
prepare 2.0s  「準備…」                      → 丟棄（把姿勢擺到定位）
go      4.0s  「██ 維持：手肘彎曲 ██」        → ★ 標該姿勢（訓練資料）
return  2.5s                                  → 丟棄
rest    5.0s                                  → 前 3.5s 標 rest，後 1.5s 是下個 preview
```

- `CUE_LEAD_SEC = 5.0` / `CUE_TAIL_SEC = 5.0`（開頭空白同時是**校準區段**）
- `CUE_REPS = 5`，8 姿勢 × 5 = 40 trial ≈ 9.2 分鐘
- `CUE_SHUFFLE = False` —— 順序固定。副作用：前一個姿勢的殘留會**系統性**混入，
  論文的 limitation 章節必須列出這一點
- `PROTOCOL_VERSION = 2`

**三個不同的 margin**（因為吸收的東西不同）：

| margin | 值 | 吸收什麼 |
|---|---|---|
| go 起始 | 0.5 s | 姿勢還沒完全擺好、還在微調 |
| **rest 起始** | **1.0 s** | ★ 肌肉還在放鬆 —— **生理衰減**，慢得多 |
| 段落結束 | 0.2 s | 提前放鬆 |

rest 特別大的依據（第一份真實錄音量出來）：從 return 起點算，
訊號衰減 90% 需要 **中位 1.80 s / p90 2.86 s / max 3.65 s**，
舉手臂的姿勢最慢（`shoulder_flex` 2.38 s、`pick_and_raise` 2.70 s）。
`CUE_RETURN_SEC + rest 起始 margin = 2.5 + 1.0 = 3.5 s` → 覆蓋 98%，
剩下的長尾由 `audit_trials.py` 逐 trial 標記。

### 6.6 兩件必須知道的事

1. **校準一定要用開頭空白段。** 訓練與推論的尺度來源必須相同，
   混用會讓全對率從 **99.6% 掉到 58%**（實測數字）。
2. **維持姿勢時手臂必須懸空。** 手肘靠桌面、前臂撐扶手、上臂貼身側
   → 重力被外物支撐 → 肌肉不出力 → EMG 消失 → 判成 rest →
   **機械手臂自己回 home**。訓練與操作都適用，已寫進啟動提示。

### 6.7 主要腳本

| 檔案 | 用途 | 狀態 |
|---|---|---|
| `record_session.py` / `record_sessionV2.py` | cue 導引錄製，8 姿勢 × 5 次 | 使用中 |
| `ads1299_reader.py` | 單執行緒即時波形 + CSV | 除錯用 |
| `ads1299_realtime_pipeline.py` | 多執行緒 Reader/Classifier/ArmControl | 已被 `src/control/` 取代 |
| `semg/v2/preprocess.py` | `StreamFilter`（唯一前處理實作） | 使用中 |
| `semg/v2/train.py` | 訓練 | 使用中 |
| `semg/v2/infer.py` | `Recognizer` 即時推論 | 使用中 |
| `semg/v2/cues.py` | cue 檔對齊、校準區段擷取 | 使用中 |
| `audit_trials.py` | 逐 trial 稽核（共識投票標記） | 使用中 |
| `semg_analysisV5.m` | MATLAB 離線分析 | 使用中 |

MATLAB 慣例：`TIME_YAXIS_MODE`（auto/manual/sym）、`RMS_YAXIS_MAX/STEP`、
輸出存到同名子目錄、**60 Hz notch**（不是 50 Hz）。

---

## 7. 控制層 + 延遲預算 ← **下一步的重點**

### 7.1 檔案結構

```
src/control/
  arm_config.py       所有可調參數集中一處（程式碼本體無魔術數字）
  debounce.py         DofDebouncer：信心門檻 → N-of-M → refractory
  arm_controller.py   Prediction 契約、MockArm、ArmController 狀態機
  sources.py          PredictionSource：mock / replay / live
  run_arm_control.py  CLI 進入點
tests/
  test_debounce.py    單幀雜訊、低信心幀、refractory
  test_mapping.py     位置鏡像映射、夾爪 toggle 序列、限位 clamp
```

### 7.2 四種執行模式

```bash
# A. 完全離線：假手勢 + 假手臂
python src/control/run_arm_control.py --source mock --dry-run

# B. 假手勢 + 真手臂  ★ 第一次接手臂時用這個
python src/control/run_arm_control.py --source mock --arm-ip 192.168.1.181

# C. 重播 CSV 過真模型 → 真手臂
python src/control/run_arm_control.py --source replay \
    --csv data/raw/multi/gary/multi_gary_xxx.csv --arm-ip 192.168.1.181

# D. 真串列 + 真模型 + 真手臂（最終部署）
python src/control/run_arm_control.py --source live --port COM9 \
    --model results/v2/dof_model.pt --arm-ip 192.168.1.181
```

按鍵：`Enter` = DISARMED→ARMED；`q` = 停止+回 home+斷線；`z` = J6 單獨歸零。

### 7.3 控制參數（`arm_config.py`）

```python
ARM_IP        = "192.168.1.181"
HOME_ANGLES   = [0.0, -90.0, 180.0, 0.0, 0.0, 0.0]      # J1..J6
HOME_SPEED    = 30.0        # °/s
COLLISION_SENSITIVITY = 3   # 0~5

ELBOW_FLEX_ANGLE    = 90.0  # J3
SHOULDER_FLEX_ANGLE = 0.0   # J2

JOG_RATE_DEG_S   = 45.0     # J6 jog 角速度（TODO 實機調）
CMD_PERIOD       = 0.10     # 指令節流週期，也是 jog 推進週期
JOG_STEP_DEG     = 4.5
CMD_DEADZONE_DEG = 0.5
CMDNUM_MAX       = 2        # arm.cmd_num 到此值跳過該 tick，防佇列堆積
JOINT_SPEED      = 40.0     # °/s（出廠上限 180）
JOINT_ACC        = 200.0    # °/s²（出廠上限 1145）

P_MIN      = {"hand": 0.80, "elbow": 0.75, "shoulder": 0.75}
VOTE_M     = 6              # 300 ms 視窗
VOTE_N     = 4
REFRACTORY = {"hand": 0.60, "elbow": 0.40, "shoulder": 0.40}

WATCHDOG_SEC      = 0.5
GRIP_HOLD_MAX_SEC = 30.0
```

**雙層限位**（送出前先 clamp SOFT 再 clamp JOINT）：

| 軸 | SOFT_LIMITS（本專案） | JOINT_LIMITS（出廠） |
|---|---|---|
| J2 肩 | −90° ~ 0° | −150° ~ 150° |
| J3 肘 | 90° ~ 180° | −3.5° ~ 300° |
| J6 腕 | −180° ~ 180° | −360° ~ 360° |

### 7.4 為什麼用 mode 0

- mode 1（servo）需 ~100 Hz 穩定餵點，分類 thread 一卡頓就產生大加速度 → 危險
- mode 4（關節速度）最適合 jog，但 **mode 是全域的**，J2/J3 位置鏡像需要 mode 0，無法共存
- 結論：**全程 mode 0**，用「目標角追蹤」模擬 jog

指令送出的三個條件（全成立才送）：

```python
now - last_cmd_t >= CMD_PERIOD                  # 0.10 s
and max(|target - last_sent|) >= CMD_DEADZONE   # 0.5°
and arm.cmd_num < CMDNUM_MAX                    # 2
```

`cmd_num` 這道閘最重要 —— `set_servo_angle(wait=False)` 是排進佇列的，
20 Hz 無節制發送會讓佇列爆掉，手臂在你放開手勢後繼續跑好幾秒。

夾爪一律 `sync=False`（`sync=True` 會排進運動佇列，等關節運動跑完才動）。

### 7.5 ★ 端到端延遲預算

| 環節 | 來源 | 貢獻 |
|---|---|---|
| 特徵視窗填滿 | `WIN_SEC = 0.5` | 0.50 s |
| 序列上下文 | `(SEQ_LEN−1) × STEP_SEC = 9 × 0.05` | 0.45 s |
| Recognizer 投票 | `votes = 4` × 0.05 | 0.20 s |
| **DofDebouncer N-of-M** | `VOTE_M = 6` × 0.05 | **0.30 s** |
| 指令節流 | `CMD_PERIOD` | 0.10 s |
| 計算 | 實測 | 0.02 s |
| **合計** | | **≈ 1.57 s** |

> `REFRACTORY` **不**計入穩態延遲——它只限制轉態頻率，不延後第一次轉態。

### 7.6 降延遲方案（依 ROI 排序）

**① 折疊兩層投票 —— 最高 ROI，零成本，不需重訓**

`Recognizer` 的 `votes` 與控制層的 `DofDebouncer` N-of-M 是**功能重疊的兩層投票**，
串聯後延遲相加、穩定性只有邊際增益。控制層那層更好（分 DOF 獨立、有信心門檻、
有 refractory），所以砍掉模型端那層：

```python
# LiveSource / ReplaySource 建構 Recognizer 時
rec = Recognizer(model_path, votes=1)     # 原本 4
```

節省 **0.20 s**，不需重訓、不改模型、不影響離線準確率。

**② 縮小 DofDebouncer 視窗**

```python
VOTE_M = 4      # 原 6
VOTE_N = 3      # 原 4
```

節省 **0.10 s**。3-of-4 仍能擋掉單幀雜訊。
可分 DOF 給不同值：`elbow`/`shoulder` 是位置鏡像、誤判下一幀就糾正回來，
可以更激進；`hand` 的 J6 jog 有累積誤差、pinch 是邊緣觸發，要保守。

**③ 縮短指令節流**

```python
CMD_PERIOD = 0.05     # 原 0.10
```

節省 **0.05 s**，但發送頻率翻倍。`CMDNUM_MAX = 2` 那道閘仍在，理論上安全，
**但必須實機驗證佇列不堆積**（放開手勢後手臂是否立刻停）。風險中等。

> ①+②+③ 合計 **1.57 → 1.22 s**，全部不需要重訓模型。先做這三個再談重訓。

**④ 縮短序列長度（需重訓）**

```python
SEQ_LEN = 6      # 原 10 → (6−1)×0.05 = 0.25 s
```

節省 **0.20 s**。但 `MODEL_BLOCKS = 2` 的感受野是 7，`SEQ_LEN = 6` 會讓
感受野超過序列長度（浪費容量）；建議同時把 `MODEL_BLOCKS` 降到 1（感受野 3）
或維持 2 但接受 padding 主導。**必須重跑訓練與評估。**

**⑤ 縮短特徵視窗（需重訓，風險最高）**

```python
WIN_SEC = 0.3    # 原 0.5
```

節省 **0.20 s**。但 RMS 類特徵的估計變異隨視窗縮短而上升，
低振幅通道（thumb、index）SNR 會明顯變差。**最後才動這個。**

> ①–⑤ 全做 → **≈ 0.82 s**。

**⑥ 不要動的**

- `STEP_SEC`（0.05）：改小會等比放大 CPU 負載，39% 已不算低
- `WATCHDOG_SEC`（0.5）：安全機制，不是延遲來源
- SPI prescaler：見 §4.6，改了會 overrun

### 7.7 量測而不是推算

上面全是**理論值**。實機整合時請用 `--log` 產生的 CSV 做端到端量測：

1. 在 `Prediction` 加一個「該預測所依據的最後一個 sample 的 `sample_index`」欄位
2. log 記錄 `cmd_sent` 的 wall clock
3. `t_cmd − sample_index / FS` 就是真實端到端延遲的分布

理論值與實測差距通常來自 OS 排程與序列埠緩衝，那是 §9 的問題，不是演算法的。

### 7.8 安全機制（不要為了降延遲拆掉）

1. 三道去彈跳閘：信心門檻 → 多數決 → refractory
2. 看門狗：0.5 s 沒收到 `Prediction` → `set_state(4)` + log ERROR + 進 SAFE，需人工重 arm
3. DISARMED 預設：啟動後不接受任何動作指令
4. 雙層限位 clamp
5. `error_code` **不自動清除**：碰撞後直接進 SAFE，自動 `clean_error()` 會掩蓋碰撞事件
6. `Ctrl-C` / `q` 一定回 home 並 disconnect，不留下 enable 狀態的手臂

---

## 8. 已知問題與待辦

### 8.1 🔴 需要立刻處理

| # | 問題 | 說明 | 行動 |
|---|---|---|---|
| ⚠️-1 | `BIAS_SENSP/N = 0x10` | 快照的 `ads1299.c` L179–180 只讓 CH5 參與 BIAS 迴路 → 單點失效。CH5 電極一脫落，全通道共模抑制崩潰 | 改為 `0x7F`（CH1–CH7）。SENSP 必須等於 SENSN，否則差動 EMG 會進入回授路徑。CH8 排除 |
| ⚠️-2 | ch2 / ch4 對應衝突 | 見 §3.2 | 對調 `CH_NAMES` 或對調電極線，二擇一 |
| ⚠️-3 | 快照韌體疑似過舊 | 快照沒有「500 ms post-PWDN 延遲」與 `ADS1299_WriteRegVerified`（記錄中已修正冷啟動問題），且 `HAL_Delay(100)` 與紀錄不符 | **以 CubeIDE 實際燒錄的版本為準**，並把該版本回存到專案 |
| ⚠️-4 | 序列埠緩衝 | 80 kB/s 下 OS 驅動 RX 緩衝只有 ~51 ms 餘裕。掉包會**靜默推進** `sample_index`，cue 標籤錯位而 bad-frame counter 不會報警 | `ser.set_buffer_size(rx_size=262144)`（`LiveSource` 已有，錄製端請確認）。另外用 wall clock 對照 `sample_index / FS`：線性偏移 = 時鐘漂移，階梯跳躍 = 掉包 |

### 8.2 已解決（歷史紀錄，勿重踩）

**韌體**

- `main.c` while 迴圈遺漏結尾大括號
- `stm32l4xx_it.c` 重複 IRQ handler（CubeMX 重生成造成）
- 缺少 `DMA1_Channel4_IRQHandler` → `g_uart_tx_busy` 永久鎖死
- USART1 TX/RX 腳位對調（正確為 PB6=TX, PB7=RX）
- `SystemClock_Config` 被改成 PLL + AHB/4 → 已還原 HSI 直驅 16 MHz
- `UART_OVERSAMPLING_16` 在 16 MHz HSI 下無法產生 1,600,000 baud
- SPI RX FIFO 殘留造成 ID 讀取錯位 → SDATAC 後強制排空

**硬體**

- J6 jumper cap（pins 5–36）未拔除 → 無法輸入外部差動訊號
- 單極供電未給 midsupply 參考 → 負半週被截斷
- `BIAS_SENS = 0xFF` + 浮接 BIAS 電極 → offset 漂移（bring-up 期用 `0x00`）
- BIAS 回授迴路頻寬僅 ~41 Hz（R8=390 kΩ, C20=10 nF）→ 120/180 Hz 干擾**在迴路抑制範圍外**
- JP1 裝上時 JP6 必須拔除，否則 U11 與 BIASOUT 對打
- 冷啟動要按 Reset 才有訊號 → 加長 PWDN 後延遲 + 暫存器 readback 重試
- 單芯線扭轉產生微裂痕（把線扭在一起反而讓訊號變差）→ 不扭轉重焊解決

**軟體**

- MATLAB 用 50 Hz notch（台灣是 60 Hz）
- 校準區段來源與訓練不一致 → 全對率 99.6% → 58%

### 8.3 Stage 6 準確率不均 —— 建議的排查順序

各錄音檔準確率不齊，非全部達 99%。依「便宜先做」排序：

1. **掉包檢查**：對每個檔案跑 wall clock vs `sample_index / FS`。
   階梯跳躍 = 掉包 = cue 標籤錯位 = 該檔的標籤本身就是壞的（見 ⚠️-4）
2. **校準區段一致性**：確認每個檔案的校準都取自開頭空白段、且用同一套邏輯
   （`cues.calibration_span`）。這是已知能造成 99.6% → 58% 的因素
3. **`audit_trials.py` 逐 trial 稽核**：找出「維持期間手臂沒懸空」「提前放鬆」
   的 trial，這些的標籤與實際生理狀態不符
4. **受試者間差異**：`SUBJECTS = ("gary", "neil1", "CBW1")`。
   若低分檔案集中在某一位，那是電極位置或體型差異，不是模型問題
5. **固定順序的殘留效應**：`CUE_SHUFFLE = False` 使前一姿勢的殘留系統性混入。
   若低分集中在某些姿勢轉換對，這就是原因（且必須寫進論文 limitation）

**先做 1 和 2** —— 這兩項是資料品質問題，在資料是壞的情況下調模型只會浪費時間。

### 8.4 未實作

- **LOFF（Lead-Off Detection）**：暫存器已在 `ads1299.h` 定義，`ads1299.c` 未實作。
  可即時偵測電極脫落，優先度中等
- **IMU**：不改 frame 格式、不改模型。定位是**協定驗證儀器** ——
  客觀證明維持期間手臂確實懸空、量化姿勢一致性，把 `audit_trials.py`
  從共識投票升級為儀器驗證標記。
  範圍界定：3 顆 IMU（軀幹 + 上臂 + 前臂）可估關節角；
  **力/力矩估不了**（IMU 是運動學不是動力學）；
  **手部 DOF 完全觀測不到** —— 那正是 EMG 不可取代的地方

### 8.5 為什麼是分類不是回歸（已定論，勿重開）

主要障礙**不是資料量**，是**缺乏 ground-truth 儀器**
（測角儀、力規、動作捕捉）來產生連續標籤。

次要障礙：

- sEMG → 力的映射在疲勞下不穩定（時間常數 ~28 s，完全恢復 ~240 s），
  遠超過協定的 5 s 休息間隔
- 纜線 spike artifact 被分類的投票層吸收掉，但會**直接污染**回歸輸出

---

## 9. 工具與參考資料

**硬體**：STM32L475 B-L475E-IOT01A、TI ADS1299EEG-FE Rev A、
含 AWG 的示波器、UFACTORY Lite 6 + 專屬夾爪

**工具鏈**：STM32CubeIDE、STM32CubeMX、STM32 HAL（SPI DMA / USART DMA / EXTI）、
Python（pyserial ≥3.5, torch ≥2.0, numpy ≥1.24, pytest）、MATLAB、
xArm Python SDK（vendored，不在 `requirements.txt`；
`run_arm_control.py` 會自動把 `xArm-Python-SDK-master/` 加進 `sys.path`）

**參考文獻**：

- SoftPINCH（Grønvall et al.）—— EMG 驅動軟性外骨骼
- Intelligent Upper-Limb Exoskeleton（npj Flexible Electronics, 2024）
- Cram & Kasman, *Introduction to Surface Electromyography*, Ch. 3, 4, 16, 17
- SENIAM guidelines
- ADS1299x datasheet (SBAS499C)、ADS1299 EVM user guide (SLAU443B)

> 📎 **文件格式陷阱**：兩份 ADS1299 PDF 實際上是**逐頁 JPEG 的 ZIP 壓縮檔**。
> 解壓方式：
> ```bash
> mkdir -p /tmp/ads_extract && unzip -o ADS1299x_datasheet.pdf -d /tmp/ads_extract/
> ```
> 只擷取到前 ~29 頁，programming 章節（Table 9、Table 11、p.38–49）**不在範圍內**。

---

## 10. 交接檢查清單

接手者請逐項確認：

- [ ] 取得**實際燒錄**的韌體版本，並與專案快照 diff（重點看 §8.1 ⚠️-1、⚠️-3）
- [ ] 補完 §3.4 的跳線與電極接線表
- [ ] 決定 ch2/ch4 的修正方式並執行（§3.2）
- [ ] 確認 `BIAS_SENSP/N` 已改為 `0x7F` 並重新驗證波形
- [ ] 跑 `PYTHONPATH=src python -m semg.v2.config` 確認協定摘要與 `validate()` 通過
- [ ] 確認 Python 匯入的是哪一份 `config.py`（`print(C.__file__)`）
- [ ] `pytest tests/` 全綠
- [ ] `--source mock --dry-run` 通過（含 5 次 pinch 上升緣 → `CLOSE, OPEN, CLOSE, OPEN, CLOSE`）
- [ ] `--source mock --arm-ip ...` 實機驗證限位與回 home
- [ ] 套用 §7.6 ①②③ 並實機量測端到端延遲
- [ ] 跑 §8.3 的 1 與 2，確認資料品質後再調模型

---

*文件產出：2026-08。所有參數以當時的 `config.py` / `arm_config.py` / 韌體快照為據。
標 ⚠️ 的項目是已知的不一致，交接前應解決。*
