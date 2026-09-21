# 08 — inference：串流推論與即時分類

| 檔案 | 做什麼 |
|---|---|
| **`infer.py`** | ★ `Recognizer` —— 串流推論的核心，訓練評估與部署**共用同一個** |
| `live.py` | 即時分類（純螢幕，不接手臂） |
| `predict_and_plot.py` | 離線重播一份錄音，把預測畫在訊號上 |
| `bench_latency.py` | 延遲量測 |

---

## infer.py — Recognizer

```
raw 資料塊 → StreamFilter（狀態保留的因果濾波）
           → 滑動視窗（0.5s / 0.05s）
           → 120 維特徵（用校準尺度 α）
           → StandardScaler（模型自己的統計量）
           → 序列緩衝（10 個視窗）
           → 模型 → 三組 logits
           → 逐 DOF 偏移 bias
           → MultiVoter（多數決 + 信心門檻 + 遲滯 dwell）
           → stable 狀態
```

### ★ 為什麼訓練評估與部署共用同一個 Recognizer

只要有第二份實作，就會有第二種行為。
離線評估用 `filtfilt` + 逐視窗獨立預測會畫出一張**比真實情況樂觀**的圖。

`predict_and_plot.py` 與 `leave_one_recording_out.py` 的評估都是**重播串流**，
所以圖上呈現的就是實際部署會看到的行為，包含投票造成的延遲。

### MultiVoter：三個旋鈕，效果差很多

| 旋鈕 | 對抗什麼 | 實測有沒有用 |
|---|---|---|
| `votes` 多數決 | **抖動**（逐窗預測在兩狀態間亂跳） | 有用 |
| **`dwell` 遲滯** | **轉換期**（移動過程中穩定地經過錯的狀態） | ★ 有用，做 pinch 中途會穩定判成 index，投票擋不住 |
| `conf` 信心門檻 | 低信心幀 | ⚠️ **沒用** —— 見下 |
| `bias` logit 偏移 | 逐 DOF 靈敏度 | ★ 有用 |

### ⚠️ `--conf` 是無效旋鈕

實測模型輸出的信心值**飽和在 1.00** —— 正確時中位 1.00、**錯誤時也是 1.00**。
門檻設在 0.96 以下完全沒有作用，設在以上則全部砍掉。
過度自信是 cross-entropy 訓練的常態。

★ 這個現象在控制層也被獨立驗證：`arm_config.py` 記錄了
「P_MIN 擋不住那些誤判脈衝 —— 它們的信心值是 0.78~1.00，本來就高」。

### 有效的旋鈕是 `bias`（偏移 logit）

$$\text{logits}[j][:, 1:] \mathrel{+}= b_j$$

（索引 0 一律是該 DOF 的中性狀態，所以只加在非 rest 上。）

實測掃出來的曲線（LORO 折模型、全訊號、最差三份錄音）：

| DOF | 建議 | 依據 |
|---|---|---|
| **elbow** | **`--bias elbow=3`** | 0→+3 時每 1 pp 誤觸發換到 **4.0 pp** 靈敏度 —— 整條曲線最划算的一點 |
| shoulder | **不要開** | 每省 1 pp 誤觸發要付 **4.7 pp** 漏判 |
| hand | 用 `--dwell hand=5` 代替 | 擋的是轉換期不是靈敏度 |

⚠️ 早期建議的 `--bias 0.5~0.7` 是**無效**的 —— 實測 logit 差距中位 16、p10 = 10，
要 ±3 到 ±10 才動得了。

### 防呆機制

| 檢查 | 為什麼 |
|---|---|
| `feat_version` 不符 → **拒絕載入** | 舊模型配新特徵不會報錯，只會靜靜算錯 |
| `use_hampel` 照 bundle 建濾波器 | 換濾波器等於換輸入分布 |
| 雙模型比較時兩者 `use_hampel` 不同 → **報錯** | ★ 共用特徵時只會套用 A 的設定，B 吃到錯的前處理，比出來的差異是假的 |

---

## live.py — 即時分類

```bash
# repo：
PYTHONPATH=src python -m semg.v2.live --bundle 模型.pt --port COM9 --baud 1600000
# V5moving 包：
python run_live.py COM9 --bias elbow=3 --dwell hand=5
```

| 旗標 | 用途 |
|---|---|
| `--bias elbow=3` | ★ 建議開 |
| `--dwell hand=5` | ★ 建議開（逐 DOF 語法） |
| `--idle-test 60` | 靜止誤觸發率測試 |
| `--replay 某份.csv` | 沒有板子時用檔案模擬 |
| `--bundle-b` | 兩個模型並排比較（吃同一條資料流） |
| `--calib-log` | 把每次校準的尺度存成 CSV |

⚠️ **不要分兩次跑再比模型。** 場次與貼片的變異遠大於模型差異 ——
實測同一間教室、相隔 12 分鐘的兩份錄音差 62.7% vs 97.0%。
分兩次比出來的是「哪次貼得比較好」。

### 校準在做什麼（常見誤解）

⚠️ **校準不是在濾直流** —— 直流是 20 Hz 高通做掉的。

校準算的是**每通道的穩健振幅尺度**：

$$\alpha_c = 1.4826\cdot \mathrm{median}(|x_c - \mathrm{median}(x_c)|)$$

所有「相對振幅」特徵都要除以它。**這是尺度不變性的來源。**

若校準段不夠安靜 → α 被高估 → 該通道相對特徵被壓扁 → **該自由度不觸發**。
使用者回報「有時 shoulder 完全不觸發」與這個機制吻合。

---

## predict_and_plot.py — 離線重播畫圖

```bash
PYTHONPATH=src python -m semg.v2.predict_and_plot --bundle 模型.pt --csv 某份錄音.csv
```

輸出 `_overview.png`（整份）與 `_zoom.png`（前幾個 trial，看 onset 延遲）。
每個 DOF 畫成上下兩條帶：上 = cue 真值，下 = 模型預測，錯的地方標紅。

★ 為什麼不用「邊框顏色表示對錯」（第一版做法）：邊框只有幾個 pixel，
在 550 秒的概觀圖上完全看不見。直接把真值畫成第二條帶，
哪裡對不上用眼睛掃就看得出來，還能看出**錯成什麼**。

---

## bench_latency.py — 延遲量測

```bash
PYTHONPATH=src python -m semg.v2.bench_latency --bundle 模型.pt --n 500
```

⚠️ **分項表要小心讀**：逐步分項用**全域** `C.USE_HAMPEL`（預設 True）建濾波器，
但端到端 `Recognizer.push()` 照**模型 bundle**（部署模型是 False）。
所以會出現「分項合計 18.39 ms、端到端 5.61 ms」的矛盾。
**真實運算時間是端到端那個 5.61 ms。**

完整的延遲分析見這包的 `延遲分析.md`。
