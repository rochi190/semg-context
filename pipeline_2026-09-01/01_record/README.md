# 01 — record：錄資料

**檔案**：`record_session.py`（829 行）
**輸出**：`<stem>.csv` + `<stem>_cues.csv` + `<stem>_meta.json`

---

## 怎麼跑

```bash
PYTHONPATH=src python -m hardware.record_session --port COM9 --subject gary
```

（`V5moving/` 包裡也可以錄，電極位置照 `V5moving/README.md` §1 的表。）

---

## ★ 核心設計：先標註，再跟著標註做

程式**先**把 cue 時間表寫下來，**再**照著它在螢幕上跑。
不是「做完再回頭猜每段是什麼」。

這解決了 V1 最大的問題：V1 靠猜 trial 邊界，產生大量假象。

---

## 一個 trial 的時序（V3 協定，`protocol_version = 2`）

```
▶▶ preview   1.5 s   螢幕預告下一個姿勢，受試者仍在休息、只用眼睛讀
   prepare   2.0 s   移動到目標姿勢          ← 不列入訓練
   go        4.0 s   ★ 維持不動              ← 這段才是訓練資料
   return    2.5 s   回到中性                ← 不列入訓練
   rest      5.0 s   在中性完全放鬆          ← 標成 rest
                     ─────────
                     15.0 s
```

8 個動作 × 5 次 = 40 trials × 15 s + 前後各 5 s = **550 s（9 分 10 秒）**

### 為什麼要有 preview

cue 出現當下就要求做動作的話，`go` 段開頭會被**反應時間**汙染。
實測 gary 反應時間中位 0.40 s、最大 1.00 s，而且因人、因疲勞、
因姿勢複雜度而異（`arm_lift` 0.52 s vs `index_flex` 0.22 s）。

⚠️ **不能假設「cue 時間戳當下受試者已經在做該姿勢」** —— 這是 V1 的錯誤之一。

---

## 8 個動作 = 9 種自由度組合（含 rest）

| 動作 | hand | elbow | shoulder |
|---|---|---|---|
| （trial 間的 rest） | rest | rest | rest |
| `index_flex` | index | rest | rest |
| `thumb_flex` | thumb | rest | rest |
| `pinch` | pinch | rest | rest |
| `elbow_flex` | rest | flex | rest |
| `shoulder_flex` | rest | rest | flex |
| `arm_lift` | rest | flex | flex |
| `pick_and_lift` | pinch | flex | rest |
| `pick_and_raise` | pinch | flex | flex |

理論組合是 4×2×2 = **16 種**，只錄 9 種。

**為什麼不錄滿**：16 個動作要 17 分鐘，受試者專注度會掉（第一份錄音
40 個 trial 就有 4 個做錯）、肌肉疲勞讓前後半段的振幅不可比。

★ **這正是選多輸出架構的理由** —— 三個頭各自預測，應該能組合出沒錄過的姿態。
但實測發現這件事**不自動成立**，見 `06_model/README.md`。

---

## 姿勢必須懸空（寫進 metadata 的規範）

> 維持姿勢時手臂必須**懸空** —— 手肘靠桌面、前臂撐在扶手、
> 或上臂貼死在身側，都會讓重力被外物支撐掉，肌肉不需出力、EMG 消失。

等長收縮下 EMG 振幅正比於肌肉張力。重力一旦被外物承擔，訊號直接消失，
訓練資料就會出現「標成 flex 但看起來像 rest」的樣本。

---

## 輸出的三個檔

| 檔案 | 內容 |
|---|---|
| `<stem>.csv` | `timestamp_s, sample_index, ch1..ch8, status`，2000 Hz 原始 counts |
| `<stem>_cues.csv` | `sample_index, t_s, event, gesture, rep` —— event ∈ {record_start, preview, prepare, go, return, release, end, record_end} |
| `<stem>_meta.json` | 協定參數、硬體設定、電極位置、DOF 定義、姿勢說明、品質統計 |

⚠️ **CLAUDE.md 的規則**：沒有 metadata 的資料不進 pipeline。
metadata 記錄的是「這份資料是在什麼條件下錄的」，缺了就無法回溯。
