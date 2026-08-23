"""模型：共享 backbone → 每個自由度一個分類頭（multi-input / multi-output）。

## 模型大小與延遲：實測結果推翻了直覺

CPU 實測（筆電 2 threads = 實際部署環境，batch=1，T=10）：

| 設定 | 參數 | 中位數 | p95 |
|---|---|---|---|
| 舊 Conv+BiLSTM(128)+Attn | 355,013 | 0.506 ms | 0.641 ms |
| blocks=1 width=32 | 5,547 | 0.348 ms | 0.362 ms |
| **blocks=2 width=48（採用）** | **11,723** | **0.522 ms** | **0.585 ms** |
| blocks=3 width=48 | 14,363 | 0.696 ms | 0.759 ms |
| blocks=3 width=64 | 22,219 | 0.701 ms | 0.789 ms |

**兩個違反直覺但實測成立的結論：**

1. **參數量與延遲幾乎無關。** blocks=1 時 width 32→64（參數 5.5k→13k）
   延遲完全不變（0.348→0.353 ms）。延遲只跟**層數**線性相關
   （1/2/3 層 = 0.35/0.52/0.70 ms），因為 batch=1、T=10 的規模下
   每層的 dispatch overhead 遠大於實際計算量。
2. **14k 參數的模型比 355k 參數的 BiLSTM 還慢**（0.696 vs 0.506 ms），
   因為它有更多層。「縮小模型 = 降低延遲」在這個尺度下不成立。

⚠️ 而且**模型從來不是瓶頸**：模型 0.5 ms vs 特徵抽取 3.77 ms，步進預算 50 ms。
模型佔總預算 1%。

所以換成小模型的真正理由**不是延遲**，而是：
  1. **資料量小**：一個 session 幾千個視窗，355k 參數太大，實測 V1 就是靠記憶
     session 得到 93%（換錄音掉到 28.5%）
  2. **拆成 4 個頭後**每個頭的有效訓練訊號更少，更需要小模型
  3. **延遲抖動**：blocks=2 的 p95−中位數 = 0.064 ms，比 BiLSTM 的 0.134 ms 小一半
     —— 即時控制怕的是抖動不是平均值
  4. **可 TorchScript**：本版 `forward` 回傳型別固定，可以 `torch.jit.script`
     再 `optimize_for_inference`，BiLSTM 版的條件式回傳做不到

## 為什麼是多頭而不是一個互斥多類

互斥多類必須為**每一種組合**開一個類別，組合數隨自由度指數成長，
而且**訓練時沒見過的組合永遠預測不出來**。多頭讓每個自由度獨立輸出：

    hand     ∈ {rest, index, thumb, pinch}
    elbow    ∈ {rest, flex}
    shoulder ∈ {rest, flex}

輸出層參數從 O(∏ K_d) = 16 降到 O(Σ K_d) = 8，
且就算訓練時沒 cue 過某個組合，模型也能輸出它。

★ 標籤是「**現在維持在什麼姿勢**」，不是「正在往哪個方向動」——
  移動過程在錄製時就被丟棄（見 config.DOF_STATES 註解）。

## 為什麼手部是**一個** DOF 而不是 index/thumb 兩個

多頭的分解假設是「給定共享表示 h 之後各自由度條件獨立」——
它**沒有**假設 EMG 線性相加，非線性協同可以由 backbone 表示。
但手部有一個分解解決不了的問題：

    捏（對掌）= 拇指對掌肌群 + 食指屈肌 + 額外的穩定性共同收縮
              ≠ 「食指彎曲」與「拇指彎曲」的疊加，是**不同的協同模式**

若把 pinch 標成 (index=flex, thumb=flex)，等於要模型把「pinch 的訊號」與
「單獨彎食指 + 單獨彎拇指的疊加」視為相同輸出 → 失去區分兩者的能力。
加上 3 顆遠端電極本來就解析不出 2 個獨立手指自由度（前臂屈肌群高度重疊），
所以手部改成單一 categorical DOF —— 這也是義肢領域對 grip type 的標準做法。

跨體段（手/肘/肩）的多輸出仍然成立，因為它們由**解剖上分離**的肌群驅動。

這是 Lee et al. 2024「每條肌肉一個獨立模型」的想法，但改成
**共享 backbone + 多頭**：完全獨立的模型無法利用跨通道資訊
（二頭/三頭的拮抗關係、遠端/近端能量比），共享 backbone 可以。

## 輸入輸出

    輸入  (B, T, F)              T = SEQ_LEN 個連續視窗，F = 120 維多域特徵
    輸出  list[(B, K_d)]        每個 DOF 一組 logits，K = (4, 2, 2)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from semg.v2 import config as C


class SepConvBlock(nn.Module):
    """深度可分離時間卷積：depthwise（每通道各自捲時間）+ pointwise（1×1 混通道）。

    參數量從 `k·Cin·Cout` 降到 `k·C + C·C`：k=3、C=48 時是 144+2304 vs 6912。
    dilation 讓少數幾層就涵蓋整個序列，不需要遞迴結構。
    """

    def __init__(self, ch: int, kernel: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.dw = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation, groups=ch)
        self.pw = nn.Conv1d(ch, ch, 1)
        self.bn = nn.BatchNorm1d(ch)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                      # (B, C, T)
        h = self.drop(self.act(self.bn(self.pw(self.dw(x)))))
        return x + h                           # 殘差：層數少也能穩定訓練


class AttnPool(nn.Module):
    """注意力池化：(B, C, T) → (B, C)。

    比「取最後一個時間步」好（動作可能發生在序列任何位置），
    又比 BiLSTM 便宜非常多（只有一個 C→1 的 1×1 卷積）。
    """

    def __init__(self, ch: int):
        super().__init__()
        self.score = nn.Conv1d(ch, 1, 1)

    def forward(self, x):                      # (B, C, T)
        a = torch.softmax(self.score(x), dim=-1)
        return (x * a).sum(-1), a.squeeze(1)


class TinyMultiHead(nn.Module):
    """共享 backbone + 每個 DOF 一個分類頭。

    預設 width=48 / blocks=2 → 11,625 參數，CPU eager 0.52 ms、TorchScript 0.28 ms。
    資料變多時**調大 width**（幾乎不影響延遲），不要加 blocks（延遲線性上升）。
    """

    def __init__(self, n_features: int,
                 dof_n_states: tuple[int, ...] = C.DOF_N_STATES,
                 width: int = C.MODEL_WIDTH,
                 n_blocks: int = C.MODEL_BLOCKS,
                 dropout: float = C.MODEL_DROPOUT,
                 dof_names: tuple[str, ...] = C.DOFS):
        super().__init__()
        self.dof_names = tuple(dof_names)
        self.dof_n_states = tuple(dof_n_states)

        # 1×1 投影：120 維特徵 → width。唯一與輸入維度有關的層
        self.stem = nn.Sequential(
            nn.Conv1d(n_features, width, 1),
            nn.BatchNorm1d(width),
            nn.ReLU(),
        )
        # dilation 1,2,4… → 幾層就涵蓋整個序列
        self.blocks = nn.Sequential(*[
            SepConvBlock(width, C.MODEL_KERNEL, dilation=2 ** i, dropout=dropout)
            for i in range(n_blocks)
        ])
        self.pool = AttnPool(width)
        self.drop = nn.Dropout(dropout)
        # 每個 DOF 一個頭。頭本身很小（width → n_states），判別力來自共享 backbone
        self.heads = nn.ModuleList([nn.Linear(width, n) for n in self.dof_n_states])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """回傳型別固定成 list[Tensor]，這樣才能 torch.jit.script（條件式回傳不行）。"""
        h = self.stem(x.transpose(1, 2))       # (B, T, F) → (B, width, T)
        h = self.blocks(h)
        pooled, _ = self.pool(h)
        pooled = self.drop(pooled)
        return [head(pooled) for head in self.heads]

    def forward_with_attn(self, x: torch.Tensor):
        """要看 attention 權重時用這個（分析用，不在即時路徑上）。"""
        h = self.blocks(self.stem(x.transpose(1, 2)))
        pooled, attn = self.pool(h)
        return [head(self.drop(pooled)) for head in self.heads], attn

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """(states, probs)：states (B, n_dof) 狀態索引；probs 每個頭的機率。"""
        logits = self.forward(x)
        probs = [torch.softmax(l, dim=1) for l in logits]
        states = torch.stack([p.argmax(1) for p in probs], dim=1)
        return states, probs


class ZhaoMultiHead(nn.Module):
    """★ 論文 baseline：Zhao et al. 2025（Sci Rep）的 CNN+BiLSTM+Attention，改成多頭。

    論文原文的架構描述：
        "1D CNN layer (64 filters of size 3) → maxpooling (size 2) →
         BiLSTM (128 units, dropout 0.2) → attention mechanism →
         flatten → FC (256 neurons, ReLU) → dropout (0.5) → softmax"

    **為什麼要留這個**：不是為了用它，是為了**在自己的資料上做對照**。
    「我們的架構比較好」這種話不能用推理支撐，只能用同一份資料、
    同一套評估流程跑出來的數字支撐。`train.py --arch zhao` 就會用這個。

    ⚠️ 一處必要的改造：論文是**單一互斥多類** softmax，我們是多輸出
    （每個自由度一個頭）。所以最後一層改成多頭，其餘（CNN→BiLSTM→Attention→FC）
    完全照論文。這個改造本身是必要的 —— 原版無法表達「同時維持多個姿勢」。

    參數量約 355k，是 TinyMultiHead 的 30 倍。
    """

    def __init__(self, n_features: int,
                 dof_n_states: tuple[int, ...] = C.DOF_N_STATES,
                 cnn_filters: int = C.CNN_FILTERS,
                 kernel: int = C.CNN_KERNEL,
                 lstm_units: int = C.BILSTM_UNITS,
                 lstm_dropout: float = C.BILSTM_DROPOUT,
                 fc_units: int = C.FC_UNITS,
                 fc_dropout: float = C.FC_DROPOUT,
                 dof_names: tuple[str, ...] = C.DOFS):
        super().__init__()
        self.dof_names = tuple(dof_names)
        self.dof_n_states = tuple(dof_n_states)
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, cnn_filters, kernel, padding=kernel // 2),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
        )
        # ⚠️ 不做 MaxPool：論文的輸入是「單一視窗切成 100×3 的矩陣」（時間軸 100 點），
        #    我們的時間軸只有 SEQ_LEN=10 個視窗，再 pool 一半就剩 5，資訊損失太大。
        #    這是輸入表示不同造成的必要調整，不是架構偏好。
        self.lstm = nn.LSTM(cnn_filters, lstm_units, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.lstm_drop = nn.Dropout(lstm_dropout)
        self.attn_proj = nn.Linear(lstm_units * 2, lstm_units * 2)
        self.attn_score = nn.Linear(lstm_units * 2, 1, bias=False)
        self.fc = nn.Sequential(
            nn.Linear(lstm_units * 2, fc_units),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
        )
        self.heads = nn.ModuleList([nn.Linear(fc_units, n) for n in self.dof_n_states])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        h = self.cnn(x.transpose(1, 2)).transpose(1, 2)      # (B, T, filters)
        h, _ = self.lstm(h)                                   # (B, T, 2*units)
        h = self.lstm_drop(h)
        a = torch.softmax(self.attn_score(torch.tanh(self.attn_proj(h))), dim=1)
        pooled = (h * a).sum(dim=1)                           # (B, 2*units)
        z = self.fc(pooled)
        return [head(z) for head in self.heads]

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        logits = self.forward(x)
        probs = [torch.softmax(l, dim=1) for l in logits]
        return torch.stack([p.argmax(1) for p in probs], dim=1), probs


ARCHITECTURES = {"tiny": TinyMultiHead, "zhao": ZhaoMultiHead}


def build_model(arch: str, n_features: int, dof_n_states, **kw) -> nn.Module:
    """依名稱建模型。`train.py --arch {tiny,zhao}` 用這個。"""
    if arch not in ARCHITECTURES:
        raise ValueError(f"未知架構 {arch!r}，可選 {list(ARCHITECTURES)}")
    return ARCHITECTURES[arch](n_features, tuple(dof_n_states), **kw)


def describe(states, dof_names: tuple[str, ...] = C.DOFS) -> str:
    """狀態索引 → 人看得懂的字串，例如「食指:屈曲 + 拇指:屈曲」。"""
    parts = []
    for d, s in zip(dof_names, states):
        name = C.DOF_STATES[d][int(s)]
        if name != "rest":
            parts.append(f"{C.DOF_LABELS_ZH[d]}:{C.STATE_LABELS_ZH[name]}")
    return " + ".join(parts) if parts else "全部放鬆"


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class GestureNet(nn.Module):
    """單一多類版本（舊 5 類）。保留供舊資料對照，新流程請用 TinyMultiHead。"""

    def __init__(self, n_features: int, n_classes: int, width: int = C.MODEL_WIDTH,
                 n_blocks: int = C.MODEL_BLOCKS, dropout: float = C.MODEL_DROPOUT):
        super().__init__()
        self.net = TinyMultiHead(n_features, (n_classes,), width, n_blocks, dropout,
                                 dof_names=("gesture",))

    def forward(self, x):
        return self.net(x)[0]


class SplitMultiHead(nn.Module):
    """★ 每個自由度**只看自己那組通道**，各自有獨立的 backbone。

    ## 為什麼需要這個（不是為了準確率，是為了「能不能組合」）

    共享 backbone 的問題不在容量，在於它是用只有 9 種姿勢組合的資料訓練的，
    於是把「哪些自由度會同時出現」一起學進了表示層。
    使用者實測的症狀：舉 elbow + index 時 hand 不觸發 ——
    因為訓練資料裡 index 從來只跟「手臂不動」共同出現。

    留一組合測試（訓練時排除 pinch+彎肘，測模型能不能自己組合）：

        特徵子集            hand    elbow   shoulder   全對
        全部 120 維         66.7%   29.9%    93.7%     8.9%
        只有近端 60 維       0.0%   91.8%    97.5%      —
        只有遠端 45 維      89.1%    0.5%    52.7%      —

    手肘只看近端時 29.9% → **91.8%**。另一組的特徵不是沒幫助，是**在干擾**。

    ## 代價（必須量，不能只看好處）

    共享 backbone 存在的理由是跨通道資訊（例如二頭/三頭的拮抗比）。
    完全分離會失去跨組資訊，**在所有組合都見過的情況下（LORO）可能變差**。
    消融曾顯示去掉跨通道特徵會讓 LORO 從 78.3% 掉到 71.9%。

    所以這個架構要同時報兩個數字：留一組合（應大幅改善）與 LORO（不能垮）。

    ⚠️ 特徵遮罩由 `config.DOF_CHANNEL_GROUP` + `features.FEATURE_NAMES` **推導**，
       不存進 bundle —— 存了就會有兩個事實來源。改通道分組時兩邊自動一致。
    """

    def __init__(self, n_features: int,
                 dof_n_states: tuple[int, ...] = C.DOF_N_STATES,
                 width: int = C.MODEL_WIDTH,
                 n_blocks: int = C.MODEL_BLOCKS,
                 dropout: float = C.MODEL_DROPOUT,
                 dof_names: tuple[str, ...] = C.DOFS):
        super().__init__()
        self.dof_names = tuple(dof_names)
        self.dof_n_states = tuple(dof_n_states)

        masks = feature_masks_by_group(n_features)
        # 每個**通道分組**一個 backbone（不是每個 DOF 一個 —— elbow 與 shoulder
        # 共用近端，硬拆成兩個只是浪費參數，它們看的輸入完全相同）
        self.groups = sorted({C.DOF_CHANNEL_GROUP[d] for d in self.dof_names})
        self.register_buffer("_dummy", torch.zeros(1), persistent=False)
        self.masks = {}
        enc = {}
        for g in self.groups:
            m = masks[g]
            self.register_buffer(f"mask_{g}", torch.as_tensor(m), persistent=True)
            enc[g] = nn.Sequential(
                nn.Conv1d(int(m.sum()), width, 1), nn.BatchNorm1d(width), nn.ReLU(),
                *[SepConvBlock(width, C.MODEL_KERNEL, dilation=2 ** i, dropout=dropout)
                  for i in range(n_blocks)],
            )
        self.enc = nn.ModuleDict(enc)
        self.pool = nn.ModuleDict({g: AttnPool(width) for g in self.groups})
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(width, n) for n in self.dof_n_states])
        self._dof_group = [C.DOF_CHANNEL_GROUP[d] for d in self.dof_names]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        pooled = {}
        for g in self.groups:
            m = getattr(self, f"mask_{g}")
            h = self.enc[g](x[:, :, m].transpose(1, 2))
            p, _ = self.pool[g](h)
            pooled[g] = self.drop(p)
        return [head(pooled[g]) for head, g in zip(self.heads, self._dof_group)]

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        logits = self.forward(x)
        probs = [torch.softmax(l, dim=1) for l in logits]
        return torch.stack([p.argmax(1) for p in probs], dim=1), probs


def feature_masks_by_group(n_features: int) -> dict:
    """每個通道分組對應哪些特徵欄位。

    規則：
      - 逐通道特徵 → 該通道所屬的組
      - 跨通道特徵 → 只有**兩端都在同一組**的 logr 對才給那一組；
        `prop_`（v2 已是組內）給該通道的組；
        `grp_*` 與 `logr_distal_proximal` 是**跨組**的，兩組都不給
        （它們正是造成耦合的東西）
    """
    from semg.v2 import features as F
    names = list(F.FEATURE_NAMES)
    if len(names) != n_features:            # 特徵子集實驗時可能不同，退回全給
        return {g: np.ones(n_features, bool) for g in C.CH_GROUPS}
    ch_of = {}
    for g, members in C.CH_GROUPS.items():
        for c in members:
            ch_of[c] = g
    out = {}
    for g in C.CH_GROUPS:
        mine = set(C.CH_GROUPS[g])
        m = []
        for nm in names:
            if nm.startswith("logr_") and not nm.startswith("logr_distal"):
                # 名稱是 logr_<a>_<b>，而通道名本身含底線（latissimus_dorsi），
                # 所以不能用 split 還原，要拿 CROSS_PAIRS 回頭比對
                hit = [p for p in C.CROSS_PAIRS if nm == f"logr_{p[0]}_{p[1]}"]
                m.append(bool(hit) and hit[0][0] in mine and hit[0][1] in mine)
            elif nm.startswith("prop_"):
                m.append(nm[len("prop_"):] in mine)
            elif nm.startswith(("grp_", "logr_distal")):
                m.append(False)             # 跨組 —— 兩邊都不給
            else:
                m.append(any(nm.startswith(c + "_") for c in mine))
        out[g] = np.asarray(m, dtype=bool)
    return out


ARCHITECTURES["split"] = SplitMultiHead
