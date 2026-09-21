# 00 — config：所有超參數集中一處

**檔案**：`config.py`（631 行）
**被誰用**：後面**每一個**階段都 `from semg.v2 import config as C`

---

## 為什麼所有參數集中在一個檔

同一個常數散在多個檔案裡，改一個忘一個時**不會報錯**，只會靜靜地算錯。
本專案踩過這類坑很多次，所以規則是：任何超參數只能有**一個**定義位置。

---

## 主要區塊

### 硬體

```python
FS = 2000                                # ADS1299 取樣率
ADS_GAIN, ADS_VREF, ADS_BITS = 8, 4.5, 24
LSB_MV = (2*ADS_VREF)/(ADS_GAIN*2**ADS_BITS)*1e3   # = 6.7055e-5 mV
```

### 電極配置（7 通道）

```python
CH_NAMES = ("latissimus_dorsi", "biceps", "triceps", "deltoid_anterior",
            "wrist_flexor", "index", "thumb")
_PROXIMAL = ("latissimus_dorsi", "deltoid_anterior", "triceps", "biceps")
CH_GROUPS = {"proximal": _PROXIMAL, "distal": (其餘 3 條)}
```

⚠️ **ch2/ch4 曾經貼反**（V2/V3 時期）。2026-08-13 由 4 份乾淨錄音、
2 位受試者一致佐證後修正 —— **只改標籤名稱，不動原始 CSV**。
CSV 是量測事實，標籤是註解，改錯的那一個。

### 前處理

```python
NOTCH_FREQ = 60.0    # 台灣電網（論文是丹麥 50Hz）
NOTCH_Q = 30.0       # → 帶寬 60/30 = 2 Hz
BANDPASS = (20.0, 450.0)
BANDPASS_ORDER = 4
HAMPEL_HALF_WINDOW = 100   # ±50 ms
HAMPEL_SIGMA = 2.0
USE_HAMPEL = True          # ⚠️ 交付模型是關的，見下
```

### 切窗

```python
WIN_SEC = 0.5      STEP_SEC = 0.05     # 90% 重疊，依 Zhao et al. 2025
SEQ_LEN = 10                           # 序列涵蓋 0.5 + 9×0.05 = 0.95 s
```

### 自由度定義

```python
DOFS = ("hand", "elbow", "shoulder")
DOF_N_STATES = (4, 2, 2)
DOF_CHANNEL_GROUP = {"hand": "distal", "elbow": "proximal", "shoulder": "proximal"}
```

★ `DOF_CHANNEL_GROUP` 不只是註解 —— `split` 架構直接用它推導特徵遮罩
（見 `06_model/README.md`）。

---

## ★ 兩個會寫進模型檔的設定

這兩個一旦不一致，**舊模型配新特徵不會報錯，只會靜靜地算錯**：

| 設定 | 意義 | 保護機制 |
|---|---|---|
| `FEAT_VERSION` | 特徵**定義**的版本（目前 2） | `infer.Recognizer` 載入時比對，不一致就**拒絕載入** |
| `USE_HAMPEL` | 有沒有套 Hampel 濾波器 | 寫進 bundle，載入時照 bundle 的值建濾波器 |

`FEAT_VERSION` v1 → v2 的改動：`prop_<ch>` 的分母從「全 7 通道總能量」
改成「自己那一組的總能量」。理由見 `04_features/README.md`。

⚠️ 目前交付的模型全部是 `use_hampel=False`。Hampel 佔每視窗計算的 77%
（14.12 ms / 19.4 ms），而 3 種子 × 7 折的消融顯示它對準確率
**分不出差異（p = 0.926）**，所以關掉是免費的。

---

## 自我檢查

```bash
PYTHONPATH=src python -m semg.v2.config
```

會印出協定時序的一致性檢查（例如「一條序列涵蓋 0.95 s，而 go 段有 4.0 s，
夠不夠塞得下」），參數改壞時會在這裡先被抓到。
