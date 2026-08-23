"""所有超參數集中一處。

v1 的教訓：參數散落在各個腳本裡（RMS_WINDOW 在 run_model_validation、
PHASE_SHIFT 在別處、notch 頻率又在另一處），改一個地方常忘了同步另一個，
先前就因此出過「只改了 source.py 忘了 run_live.py」的錯。這裡集中管理。
"""
from __future__ import annotations

from pathlib import Path

# ---------------- 路徑 ----------------
SRC_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "src")
REPO_ROOT = SRC_ROOT.parent
DATA_ROOT = REPO_ROOT / "data"
RESULTS_ROOT = REPO_ROOT / "results" / "v2"
SUBJECTS = ("gary", "neil1", "CBW1")

# ---------------- 硬體 ----------------
FS = 2000                       # ADS1299 取樣率
ADS_GAIN, ADS_VREF, ADS_BITS = 8, 4.5, 24
LSB_MV = (2.0 * ADS_VREF) / (ADS_GAIN * (2 ** ADS_BITS)) * 1e3

# 7 通道電極配置：ch1–4 近端（Lee et al. 2024 的四塊肌肉）、ch5–7 前臂/手部（SoftPINCH）
# CSV 欄位 ch1..ch8 由 ads1299 reader 寫出，這裡指定實際使用哪幾欄。
# ch8 是空的。手指的 DOF 只有 (rest, flex)，所以**不需要**指伸肌電極；
# 但如果之後想貼，它仍能提供有用的**特徵**（屈/伸拮抗比值有助於分辨
# 「用力握」與「放鬆」，也能反映握力大小）。改成 True 即可，
# 通道名、分組、跨通道拮抗對、特徵維度全部自動跟著變。
USE_EXTENSOR_CHANNEL = False

# ⚠️ ch4 必須貼**三角肌前束**（不是中束）。
#    前束 → 肩屈曲（手臂往**前**舉）← 目標場景「往前伸手取物」用的是這個
#    中束 → 肩外展（手臂往**外**舉）
#    一顆三角肌電極分不出這兩者（同一塊肌肉的不同束），所以只能擇一。
#    選前束是因為它對應「幫長者往前拿東西」的實際動作。
#
# ★ 2026-08-13：ch2 與 ch4 對調過一次。原本標成 ch2=三角肌前束、ch4=二頭肌，
#   但實際貼片位置相反。用「每個 go 段相對開頭空白段基線的倍率」驗證，
#   四份乾淨錄音（gary 17-23-12 / 14-56-09 / 16-26-55、neil 21-35-03）
#   **跨兩位受試者一致**指向 elbow_flex→ch2、shoulder_flex→ch4：
#
#       17-23-12   elbow_flex ch2 6.7×    shoulder_flex ch4 30.1×
#       14-56-09   elbow_flex ch2 2.6×    shoulder_flex ch4  6.8×
#       16-26-55   elbow_flex ch2 4.9×    shoulder_flex ch4 20.9×
#       21-35-03   elbow_flex ch2 10.9×   shoulder_flex ch4 21.1×
#
#   這也解掉了先前「elbow_flex 時三角肌反而最強」的怪現象 ——
#   那其實是「elbow_flex 時二頭肌最強」，完全正常。
#   ⚠️ 只改名字，**不動 CSV**。CSV 的 ch2 欄位就是二頭肌、ch4 欄位就是三角肌前束。
#   之後錄音與 realtime 都以現在這個貼片順序為準。
if USE_EXTENSOR_CHANNEL:
    CH_CSV_COLS = (1, 2, 3, 4, 5, 6, 7, 8)
    CH_NAMES = ("latissimus_dorsi", "biceps", "triceps", "deltoid_anterior",
                "wrist_flexor", "index", "thumb", "extensor_digitorum")
    CH_LABELS_ZH = ("背闊肌", "二頭肌", "三頭肌", "三角肌前束",
                    "曲腕肌", "食指", "拇指", "指伸肌")
else:
    CH_CSV_COLS = (1, 2, 3, 4, 5, 6, 7)
    CH_NAMES = ("latissimus_dorsi",  # ch1 背闊肌       → 肩伸展（拮抗）
                "biceps",            # ch2 二頭肌       → 肘屈曲
                "triceps",           # ch3 三頭肌       → 肘伸展
                "deltoid_anterior",  # ch4 三角肌前束   → 肩屈曲
                "wrist_flexor",      # ch5 曲腕肌       → 手部協同
                "index",             # ch6 食指（屈肌側）→ 手部協同
                "thumb")             # ch7 拇指（屈肌側）→ 手部協同
    CH_LABELS_ZH = ("背闊肌", "二頭肌", "三頭肌", "三角肌前束", "曲腕肌", "食指", "拇指")
N_CH = len(CH_NAMES)

# 分組：近端（手臂姿態/負載）vs 遠端（手指動作）。跨組能量比是重要的判別線索，
# 也是把 Lee et al. 的「多肌肉聚合」概念放進特徵的方式。
_PROXIMAL = ("latissimus_dorsi", "deltoid_anterior", "triceps", "biceps")
CH_GROUPS = {
    "proximal": _PROXIMAL,
    "distal": tuple(n for n in CH_NAMES if n not in _PROXIMAL),
}
# 要計算 log 比值的通道對（尺度不變、無除零風險；v1 實測唯一安全的跨通道表示）
CROSS_PAIRS = (("index", "thumb"),
               ("index", "wrist_flexor"),
               ("thumb", "wrist_flexor"),
               ("biceps", "triceps"),                       # 肘的拮抗對：屈 ↔ 伸
               ("deltoid_anterior", "latissimus_dorsi"))    # 肩的拮抗對：屈 ↔ 伸
if USE_EXTENSOR_CHANNEL:
    # 手指的拮抗對 —— 這是 ext 狀態唯一可靠的判別依據
    CROSS_PAIRS += (("index", "extensor_digitorum"),
                    ("thumb", "extensor_digitorum"))


def ch_index(name: str) -> int:
    return CH_NAMES.index(name)

# ---------------- 前處理 ----------------
NOTCH_FREQ = 60.0               # 台灣電網（論文是丹麥 50Hz）
NOTCH_Q = 30.0
BANDPASS = (20.0, 450.0)
BANDPASS_ORDER = 4
HAMPEL_HALF_WINDOW = 100        # 100 ms @2000Hz
HAMPEL_SIGMA = 2.0

# ★ Hampel 要不要開。
#   實測它佔**整個即時計算的 77%**（14.86 ms / 19.4 ms），
#   而 IIR 那兩段只花 0.02 ms —— 也就是說「濾波很貴」講的其實就是 Hampel。
#   文獻上只有 SoftPINCH 用它，Lee / Zhao 都沒有。
#
#   ⚠️ 關掉它**必須重新訓練**：特徵是從濾波後的訊號算的，
#      換掉濾波器等於換掉輸入分布，舊 scaler 與權重全部失效。
#      為了不讓人誤用，這個設定會被寫進模型 bundle，
#      `infer.Recognizer` 載入時照 bundle 裡的值建濾波器（見那裡的說明）。
USE_HAMPEL = True

# ★ 特徵版本。改變任何特徵**定義**時 +1，並且會寫進模型 bundle。
#   v1 → v2：`prop_<ch>` 從「佔全 7 通道總能量」改成「佔自己那一組」。
#            理由見 features.cross_channel_features 的說明
#            （全通道分母會讓手臂出力壓掉手部通道的佔比 → 各 DOF 在特徵層耦合）。
#   ⚠️ 舊模型配新特徵不會報錯，只會靜靜地算錯，所以 `infer.Recognizer`
#      載入時會比對 bundle 的 feat_version 與目前的定義，不一致就拒絕。
FEAT_VERSION = 2

# ★ 每個自由度該看哪一組通道（`split` 架構用）。
#
#   為什麼要有這個（留一組合測試量出來的）：共享 backbone 在只有 9 種
#   姿勢組合的資料上訓練，會把「哪些自由度同時出現」背進表示層。
#   後果是**組合不出沒 cue 過的姿態** —— 使用者實測「舉 elbow + index
#   時 hand 不觸發」。
#
#   決定性的證據：訓練時排除 pinch+彎肘（唯二兩者同時出現的動作），
#   再測模型能不能自己組合出來 ——
#
#       特徵子集            hand    elbow   shoulder
#       全部 120 維         66.7%   29.9%    93.7%
#       只有近端 60 維       0.0%   91.8%    97.5%     ← 手肘 29.9 → 91.8
#       只有遠端 45 維      89.1%    0.5%    52.7%     ← 手部 66.7 → 89.1
#
#   每個自由度**只看自己那組通道**時，在沒見過的組合上幾乎完全正確。
#   也就是說另一組的特徵是在**主動干擾**，不是提供資訊。
DOF_CHANNEL_GROUP = {"hand": "distal", "elbow": "proximal", "shoulder": "proximal"}

# 包絡線（用於自動標籤與部分特徵）
ENV_WINDOW = 500                # 250 ms
ENV_STEP = 50                   # 25 ms → 包絡線 40 Hz
ENV_FS = FS / ENV_STEP          # 40 Hz
SEC_PER_ENV_PT = ENV_STEP / FS  # 0.025 s

TRIM_START_SEC = 2.0            # 避開濾波器邊際效應
TRIM_END_SEC = 1.0

# ---------------- 切窗（依 Zhao et al. 2025）----------------
WIN_SEC = 0.5                   # 500 ms
STEP_SEC = 0.05                 # 50 ms（overlap 450 ms）
WIN_SAMPLES = int(WIN_SEC * FS)         # 1000
STEP_SAMPLES = int(STEP_SEC * FS)       # 100

# ---------------- 自動標籤 ----------------
# 相位由包絡線推導，不需要 trial 時間戳（v1 靠猜邊界產生大量假象）
#
# ⚠️ 門檻用「百分位數」而非 median+K·MAD。
# 實測 median+3·MAD 在 8/28 個檔案上完全失效（門檻算出來比 p95 還高），
# 原因是那個公式假設「大部分時間在休息」；活動佔比高或訊號分布寬時
# median 會落進活動區、MAD 也跟著膨脹。百分位數對活動佔比不敏感。
LBL_BASE_PCT = 10               # 基線取 p10（確定落在安靜區）
LBL_PEAK_PCT = 90               # 峰值水準取 p90
LBL_ACTIVE_FRAC = 0.35          # 活化門檻 = base + 0.35 × (p90 - p10)
LBL_QUIET_FRAC = 0.15           # 低於 base + 0.15 × range 視為 rest
LBL_MIN_BURST_SEC = 0.4         # 短於此的活化視為雜訊，不算一次動作
LBL_SLOPE_FRAC = 0.15           # 斜率門檻：相對於該 burst 峰高的比例/秒
LBL_HOLD_MIN_SEC = 0.3          # 高原至少這麼長才標 hold

PHASES = ("rest", "onset", "hold", "release")
GESTURES = ("rest", "index", "thumb", "pinch", "grasp")   # 舊的單一多類（保留供舊資料）

# ---------------- ★ 多輸出：每個自由度一個頭 ----------------
# 為什麼不用「一個互斥多類」：
#   互斥多類必須為每一種組合開一個類別（index / thumb / index+thumb / ...），
#   組合數隨自由度指數成長，而且**訓練時沒見過的組合永遠預測不出來**。
#   多輸出（每個 DOF 一個 softmax 頭）讓「同時彎曲」自然成立：
#   index=flex 且 thumb=flex 就是捏，不需要 pinch 這個類別存在。
#   這是 Lee et al. 2024「每條肌肉一個獨立模型」的想法，改成共享 backbone + 多頭
#   （共享 backbone 才能利用跨通道資訊，例如二頭/三頭的拮抗關係）。
# 自由度的**切分方式**與**狀態數**都依「訊號實際量得到什麼」而定：
#
# ★ 手部 = 一個自由度，4 個互斥狀態（不是兩個獨立手指自由度）
#     理由 1：生理上手部動作是**協同模式**而非獨立自由度。
#             捏（對掌）用的是拇指對掌肌群 + 食指屈肌 + 額外的穩定性共同收縮，
#             **不是**「食指彎曲」與「拇指彎曲」的疊加，是完全不同的協同。
#             把 pinch 標成 (index=flex, thumb=flex) 等於要模型把
#             「pinch 的訊號」與「單獨彎食指+單獨彎拇指的疊加」視為相同輸出，
#             會失去區分兩者的能力。
#     理由 2：3 顆遠端電極（曲腕肌/食指/拇指）解析不出 2 個獨立手指自由度 ——
#             前臂屈肌群高度重疊，且 ECG 型電極接觸面積大，crosstalk 嚴重。
#     理由 3：這是義肢領域對 grip type 的標準做法（pattern recognition over
#             grip patterns，不是 per-finger DOF）。
#
# 跨體段（手/肘/肩）的多輸出成立 —— 它們由**解剖上分離**的肌群驅動，
# 條件獨立的假設在這裡合理得多。跨體段的非線性交互（例如捏著東西同時彎肘
# 會有額外的穩定性共同收縮）靠「在 cue 裡明確錄到這些組合」+ 共享 backbone 處理。
#
# ★★ 標籤語意：標「**現在處於什麼姿勢**」，不是「正在往哪個方向動」。
#
# 為什麼（這一版推翻了「標動作方向」的設計）：
#   舉起 → 保持 → 放下 三段在時間上連在一起，而**移動過程很短**
#   （手指彎曲 ~0.3s、手臂舉起 ~0.5s），相對於維持的時間可忽略。
#   特意去標「正在移動」沒有意義，而且會製造三個無解的問題：
#     - 「保持」與「放鬆」都要塞進 rest，但兩者訊號差極遠（中等持平 vs 接近零）
#     - 「放下」活躍的是三角肌**離心**收縮，不是背闊肌 → 標成 ext 會教錯
#     - 需要額外的 hold 狀態或彈力帶才能自圓其說
#   改成標姿勢後，這三個問題同時消失：移動過程直接**丟棄**。
#
# 這在生理上成立：**維持一個對抗重力的姿勢，本身就需要持續的等長收縮**。
#     手臂下垂（中性）  → 重力就位，不需出力      → 訊號 ≈ 0
#     手臂前舉並維持    → 三角肌前束持續等長撐住  → 中等、持平
#     手肘彎曲並維持    → 二頭肌撐住前臂重量      → 中等、持平
#
# ⚠️ 推論：**沒有 ext 狀態**。伸展姿勢（手臂後伸、肘過度伸直）在重力下
#    幾乎沒有負載，維持它不需要出力 → 量不到。所以每個關節就是二態。
#    （要把「主動往後壓」做成一個指令的話得加阻力，那是另一種設計。）
#
# 手部維持 4 態的理由不同：它們是**互斥的協同模式**（見下方註解），
# 而且捏合中與維持捏住都是高振幅屈肌活動、彼此相似，不像 rest vs 維持姿勢
# 是振幅的兩個極端。
DOFS = ("hand", "elbow", "shoulder")
DOF_STATES = {
    "hand":     ("rest", "index", "thumb", "pinch"),   # 互斥的手部協同，維持中
    "elbow":    ("rest", "flex"),                      # rest=伸直下垂, flex=彎曲維持
    "shoulder": ("rest", "flex"),                      # rest=下垂,     flex=前舉維持
}
DOF_LABELS_ZH = {"hand": "手部", "elbow": "手肘", "shoulder": "肩膀"}
STATE_LABELS_ZH = {"rest": "中性放鬆", "flex": "屈曲維持",
                   "index": "食指彎", "thumb": "拇指彎", "pinch": "捏住"}
N_DOF = len(DOFS)
DOF_N_STATES = tuple(len(DOF_STATES[d]) for d in DOFS)

# cue 動作 = 一組要**維持**的 DOF 姿勢組合。**這是給受試者看的指示**，
# 不是模型的輸出類別（模型輸出的是上面 3 個頭）。改這裡一處，
# record / train / infer 全部自動跟著變。
# 沒有 _ext 類動作（伸展姿勢在重力下量不到，見上方 DOF_STATES 註解）。
# 16 種可能組合裡選 8 個錄：涵蓋每個 DOF 的每個非 rest 狀態，
# 再加跨體段組合讓模型見過非線性交互（同時維持多個姿勢時會有額外的穩定性共同收縮）。
ACTIONS: dict[str, dict[str, str]] = {
    # --- 單一自由度 ---
    "index_flex":       {"hand": "index"},
    "thumb_flex":       {"hand": "thumb"},
    "pinch":            {"hand": "pinch"},
    "elbow_flex":       {"elbow": "flex"},
    "shoulder_flex":    {"shoulder": "flex"},
    # --- 跨體段組合 ---
    "arm_lift":         {"elbow": "flex", "shoulder": "flex"},
    # --- 複合功能姿勢（目標場景：協助長者取物）---
    "pick_and_lift":    {"hand": "pinch", "elbow": "flex"},
    "pick_and_raise":   {"hand": "pinch", "elbow": "flex", "shoulder": "flex"},
}
ACTION_LABELS_ZH = {
    "index_flex": "食指彎曲", "thumb_flex": "拇指彎曲", "pinch": "食指+拇指 捏住",
    "elbow_flex": "手肘彎曲", "shoulder_flex": "肩膀前舉",
    "arm_lift": "肩前舉 + 肘彎曲",
    "pick_and_lift": "捏住 + 彎肘",
    "pick_and_raise": "捏住 + 彎肘 + 前舉",
}

# 致動器指令對照（控制層用；模型只輸出狀態，不決定怎麼動）
#
# ★ 控制範式是**位置鏡像**：人維持在什麼姿勢 → 機械手臂就移動到對應姿勢並停住。
#   不是速率控制（那需要「正在往哪個方向動」的標籤，而我們刻意不標移動過程）。
#   模型輸出的每個狀態直接對應一個**目標位置**，路徑規劃由控制層負責。
#
# 二爪夾爪只有開/合，所以 index / thumb 兩個單指狀態在硬體上無法各自呈現。
# 先映射成不同的夾持程度（也可改成模式切換等輔助指令）—— 這是控制層的選擇，
# 不影響模型：模型仍然分得出這三種不同的手部協同。
ACTUATOR_SEMANTICS = {
    ("hand", "rest"):  "夾爪張開",
    ("hand", "index"): "夾爪半閉合（輕夾）",
    ("hand", "thumb"): "夾爪半閉合（輔助指令，可改）",
    ("hand", "pinch"): "夾爪閉合（精準夾取）",
    ("elbow", "rest"): "移動到伸直位置",
    ("elbow", "flex"): "移動到彎曲位置",
    ("shoulder", "rest"): "移動到下垂位置",
    ("shoulder", "flex"): "移動到前舉位置",
}


def action_to_dof(action: str) -> tuple[int, ...]:
    """動作名 → 每個 DOF 的狀態索引。未提及的 DOF 一律 rest。"""
    spec = ACTIONS.get(action, {})
    return tuple(DOF_STATES[d].index(spec.get(d, "rest")) for d in DOFS)


REST_DOF = tuple(0 for _ in DOFS)          # 全部 rest

# ---------------- 錄製協定（record_session.py）----------------
# ★ 為什麼要改協定：舊資料「一個檔案只有一個手勢」，手勢與錄音檔完全共線，
#   沒有任何切分方式能證明模型學到肌電形態而不是「認出這是哪一檔」
#   （實測：同 session 93%，換到沒見過的錄音 28.5%，亂猜 25%）。
#   解法 = 一段連續錄音內**隨機順序**交替多個手勢，並記錄 cue 時間。
# ★★ 姿勢規範 —— 受試者移動到目標姿勢後**維持住**，標的是姿勢不是移動過程。
#
# 為什麼是動態而不是等長（這一版推翻了上一版的決定）：
#   致動器是**機械手臂**（外部裝置，不穿戴在身上），人的手臂全程自由。
#   這個專案的本質是**遙操作/鏡像**：人動自己的手臂 → 機械手臂跟著動。
#   訓練資料就該是「人實際移動手臂時的訊號」。
#
#   等長協定是為「穿戴式外骨骼把手臂固定住」設計的，那是另一個專案。
#   ⚠️ 而且動態協定**並沒有**解決外骨骼的問題 —— 外骨骼真正的難處是
#      「它一旦幫你出力，你的 EMG 就下降，系統以為你不想動了」，
#      那需要力/阻抗控制加 IMU，兩種協定都解不了。
#      所以選動態的理由是「它符合現在這個專案」，不是「它比較接近外骨骼」。
#
# ⚠️ 動態的代價（必須靠協定控制，否則資料會亂）：
#   1. 姿勢會變 → 肌肉長度與速度都在變 → 教授說的 IMU 問題重新成立。
#      但**對分類**而言沒有致命：只要動作的**範圍與節奏可重現**，
#      模型學到的就是整段軌跡的訊號模式。所以要求從「固定姿勢」
#      改成「**固定起始位置 + 固定範圍 + 固定節奏**」。
#   2. 每個 trial 做完要回到起始位置，而**回程的動作方向與去程相反** ——
#      回程若標成 rest 會教錯。所以新增 `return` 階段並**整段丟棄**。
#   3. 「伸」的方向常常是重力幫忙的（離心控制），主動肌活化較弱。
#      先用徒手錄，若 check_states.py 顯示 ext 分不出來，再加彈力帶。
POSE_SPEC = {
    "neutral": "坐姿，上臂自然下垂於體側，手肘伸直，前臂與手自然下垂，手掌張開",
    # ★★ 最重要的操作限制
    "airborne": "維持姿勢時手臂必須**懸空** —— 手肘靠桌面、前臂撐在扶手、"
                "或上臂貼死在身側，都會讓重力被外物支撐掉，肌肉不需出力、EMG 消失，"
                "系統會判成 rest 讓機械手臂放下。**訓練與操作都必須懸空。**",
    # timing 由實際參數生成 —— 寫死的秒數會在調參後變成錯的說明
    "timing": None,        # 由下方 _fill_timing() 填入
    "range": "角度不用精準，但**每次要盡量一致** —— 範圍變動會讓維持時的出力大小改變。",
    # ⚠️ 秒數不要寫死 —— 調參後說明會變成錯的（這裡原本寫「維持 3 秒」，
    #    CUE_ACTION_SEC 改成 4.0 之後就不對了）。由 _fill_timing() 一起生成。
    "fatigue": None,       # 由下方 _fill_timing() 填入
    # 每個動作的 (要維持的目標姿勢, 維持時的要點)
    "actions": {
        "index_flex":    ("食指彎向掌心並維持，其餘手指盡量放鬆張開",
                          "手臂保持中性下垂懸空，只有食指出力"),
        "thumb_flex":    ("拇指彎向掌心並維持，其餘手指放鬆",
                          "同上，只有拇指出力"),
        "pinch":         ("食指與拇指指腹相碰、輕輕捏住並維持",
                          "捏的力道固定（像捏住一張紙不讓它掉），不要越捏越用力"),
        "elbow_flex":    ("手肘彎曲約 90–120°、前臂抬起並**懸空**維持",
                          "上臂仍自然下垂貼體側，只有前臂抬起；不要把手肘撐在任何東西上"),
        "shoulder_flex": ("整條手臂向**前**平舉約 90°、肘保持伸直並懸空維持",
                          "手臂打直往前平舉，不要聳肩，也不要靠在桌上"),
        "arm_lift":      ("肩前舉約 90° + 同時肘彎曲約 90°（像把東西端在胸前）",
                          "兩個關節同時維持，全程懸空"),
        "pick_and_lift": ("捏住 + 肘彎曲抬起（像撿起小東西拿到面前）",
                          "捏的力道與肘的角度都要維持住"),
        "pick_and_raise": ("捏住 + 肘彎曲 + 肩前舉（三個一起）",
                          "最費力的一個，撐完該撐的秒數即可，之後務必完全放鬆"),
    },
}

def _fill_fatigue() -> str:
    """疲勞提醒也用實際參數生成（原本寫死「維持 3 秒」，改 4.0 後就錯了）。"""
    return (f"每個 trial 維持 {CUE_ACTION_SEC:.0f} 秒、"
            f"動作間有 {CUE_REST_SEC - CUE_PREVIEW_SEC:.1f} 秒完全放鬆，"
            f"避免疲勞讓訊號漂移。"
            f"若中途覺得痠，寧可中止重錄，不要硬撐（疲勞會讓頻譜整體下移）。")


def _fill_timing() -> str:
    """用**實際的**時間參數描述流程。寫死的秒數在調參後會變成錯的說明。"""
    return (f"▶▶預告下一個姿勢（{CUE_PREVIEW_SEC:.1f}s，仍在休息、只用眼睛讀，不列入訓練）→ "
            f"移動到目標姿勢（{CUE_PREPARE_SEC:.1f}s，不列入）→ "
            f"**維持不動 {CUE_ACTION_SEC:.1f}s**（這段才是訓練資料）→ "
            f"回到中性（{CUE_RETURN_SEC:.1f}s，不列入）→ "
            f"在中性完全放鬆 {CUE_REST_SEC - CUE_PREVIEW_SEC:.1f}s（標成 rest）")


CUE_ACTIONS = tuple(ACTIONS)    # 預設錄全部動作
CUE_GESTURES = CUE_ACTIONS      # 舊名保留（record_session 的 --gestures 參數）
CUE_LEAD_SEC = 5.0              # 開頭空白（濾波器暖機 + 避開起始暫態）
CUE_TAIL_SEC = 5.0              # 結尾空白
# trial 結構（時間軸由左到右）：
#
#   rest(標 rest) → preview(丟棄) → prepare(丟棄) → go(標該姿勢) → return(丟棄) → rest…
#   ↑ 完全放鬆      ↑ 預告下一個    ↑ 移動到位     ↑ **維持**      ↑ 移動回中性
#                   還在休息、不動   開始移動
#
# ★ preview 的用意：把「讀字 + 判斷」與「移動」分開。
#   受試者在 preview 段仍然靜止，只用眼睛讀；prepare 段才開始動。
#   反應時間 = preview + prepare，但只有 prepare 需要身體動作。
#   preview 必須**丟棄**：受試者已知道下一個是什麼，可能提前緊張，
#   留在 rest 裡會汙染「完全放鬆」這一類。
CUE_PREVIEW_SEC = 1.5           # 預告下一個姿勢（仍在休息）→ **丟棄**
CUE_PREPARE_SEC = 2.0           # 移動到目標姿勢 → **丟棄**（移動過程語意模糊）
CUE_ACTION_SEC = 4.0            # **維持**目標姿勢不動 → 標該姿勢（這才是訓練資料）
CUE_RETURN_SEC = 2.5            # 移動回中性 → **丟棄**（同上）
CUE_REST_SEC = 5.0              # 中性姿勢放鬆；前 (REST-PREVIEW)=3.5s 標 rest，後 1.5s 是 preview
                                # ★ 別低於 PREVIEW + 1.7s，否則 rest 段產不出序列（validate() 會擋）
CUE_REPS = 5                    # 每個姿勢重複次數（8 姿勢 × 5 = 40 trial ≈ 9.2 分鐘）

# ─────────────────────────────────────────────────────────────────────────
# ★ 上面四個秒數是用**第一份真實錄音**（gary, 40 trial）量出來後調的，
#   不是憑感覺。三個依據：
#
# 1) CUE_ACTION_SEC 3.0 → 4.0：**不靠回收 prepare 後段**來增加資料
#    量到「從 prepare 起算到訊號到達高原」= 中位 0.40s / p90 0.61s / max 1.00s，
#    所以 prepare 的後 1.6s 其實已經在維持姿勢。但**不能回收** ——
#    那等於假設「每個人看到 cue 後 1.0s 內一定到位」，而那是 gary 這一次的數字，
#    換人、換疲勞、換姿勢都不保證（複合姿勢 arm_lift 0.52s vs index_flex 0.22s）。
#    改用加長 go 取得**完全相同**的資料量（48 條序列），且零假設：
#    go 起點仍在 prepare+2.0s，最慢到位 1.00s → 還有 1.0s 餘裕 + 0.5s margin。
#
# 2) CUE_RETURN_SEC 1.5 → 2.5：**修 rest 類別被汙染**
#    從 **return 起點**（開始放下手臂）量到訊號衰減 90% 所需時間：
#      中位 1.80s / p90 2.86s / max 3.65s
#    舉手臂的姿勢最慢（shoulder_flex 中位 2.38s、pick_and_raise 2.70s），
#    手指最快（index_flex 0.65s、thumb_flex 0.72s）。
#    舊設定 return 1.5 + margin 0.5 = 2.0s → 只覆蓋約一半，
#    rest 標籤被高活動訊號汙染（shoulder_flex 的 release 前 0.5s 是基線 8.07×）。
#
#    **return + CUE_MARGIN_REST_START_SEC 才是真正的衰減預算**：
#      3.0s → 覆蓋 90%    3.5s → **98%（採用）**    4.0s → 100%
#    取 3.5s 而不是 4.0s：最後 2% 的尾巴交給 audit_trials.py 逐 trial 標記，
#    比每個 trial 都多花 0.5s 划算。
#
#    這也解釋了先前量到的「offset 偵測慢 1.75s」—— 與衰減中位數 1.80s 吻合，
#    那不只是生理現象，有一部分是**標籤汙染教出來的**。
#
# 3) CUE_PREVIEW_SEC 2.5 → 1.5：把時間還給 rest（讀一個姿勢名 1.5s 夠），
#    讓 rest 有標籤長度從 2.5s 變 3.5s，吸收得起加大的起始 margin。
# ─────────────────────────────────────────────────────────────────────────

PROTOCOL_VERSION = 2            # ★ 協定版本。v1 = go 3.0/return 1.5/preview 2.5。
                                # 寫進 meta.json，避免新舊協定的錄音被混進同一個資料集。

# ★★ 姿勢順序：固定 or 打亂
#
# 選 **固定**（CUE_SHUFFLE = False

POSE_SPEC["timing"] = _fill_timing()
POSE_SPEC["fatigue"] = _fill_fatigue()   # 時間參數都定義完了才能生成）的理由與代價：
#   ✓ 受試者能預期 → 動作做得更準、範圍更一致 → 訊號品質較好
#   ✓ 搭配 preview 之後幾乎不會做錯動作
#   ✗ 姿勢 B 永遠緊跟在 A 之後，前一個動作的殘留（收縮後增強、代謝物累積、
#     疲勞造成的頻譜下移）會系統性地混進 B 的「維持」段 →
#     模型學到的是「B + A 的殘留」，實際操作時 B 前面接任何東西就對不上。
#
# 打亂能把這個**系統性偏差**轉成**雜訊**（殘留與標籤不再相關），
# 但每個姿勢只重複 5 次、前面可能接 7 種姿勢 —— 平均不掉，
# 所以小樣本下打亂只是讓偏差變隨機，換來的動作品質下降可能不划算。
#
# ⚠️ 用固定順序就**必須在報告裡列為限制**，並用
#    「另錄一小段不同順序的資料當測試集」來量化影響（見 docs 第 12 節）。
CUE_SHUFFLE = False
CUE_WARMUP_DISCARD_SEC = 1.0    # 丟棄開機後最初這段（硬體/UART 尚未穩定）
# ★ 非對稱 margin。兩端吸收的東西不同，變異也不同：
#   起始：受試者可能還沒完全擺好、還在微調 → 變異大 → 給多一點
#   結束：提前放鬆，通常很短 → 給少一點
#   ⚠️ 這裡**不是**在補「反應延遲」—— 反應時間由 preview + prepare 提供，
#      go 開始時受試者應該已經維持住了。
CUE_MARGIN_START_SEC = 0.5      # 段落起始往後縮
CUE_MARGIN_END_SEC = 0.2        # 段落結束往前縮
CUE_MARGIN_SEC = CUE_MARGIN_START_SEC   # 舊名保留（其他地方仍以單一 margin 呼叫）

# ★ rest 段的起始 margin **另外給，而且更大**。
#   go 段的起始 margin 吸收的是「還沒完全擺好」（小），
#   rest 段吸收的是「肌肉還在放鬆」—— 那是**生理衰減**，慢得多：
#   實測從 return 起點算，衰減 90% 需要 p90 2.86s（舉手臂的姿勢更久）。
#   **CUE_RETURN_SEC + 這個值 = 總衰減預算**，目前 2.5+1.0=3.5s → 覆蓋 98%。
#   兩者都需要 —— 只加長 return 不夠（衰減有長尾），
#   只加大 margin 又會把 rest 的有標籤長度吃光（rest 只有 3.5s 可標）。
CUE_MARGIN_REST_START_SEC = 1.0

# 序列長度放這裡而不是 train.py —— validate() 需要它來算「段落夠不夠長」，
# 而 config 不能 import train（會循環）。
SEQ_LEN = 10                    # 堆疊幾個視窗成一條序列 → 涵蓋 0.95s 上下文


def _labelled_segments() -> dict[str, tuple[float, float]]:
    """有標籤的段落 → (原始長度, 該段的起始 margin)。

    起始 margin 兩段不同：rest 要吸收肌肉放鬆的生理衰減，比 go 慢得多。
    """
    return {"go（維持姿勢）": (CUE_ACTION_SEC, CUE_MARGIN_START_SEC),
            "rest（中性放鬆）": (CUE_REST_SEC - CUE_PREVIEW_SEC,
                              CUE_MARGIN_REST_START_SEC)}


def segment_report() -> list[tuple[str, float, float, int, int, float]]:
    """每個有標籤段落能產生幾個視窗、幾條序列、餘裕多少。

    回傳 [(名稱, 原始秒數, 扣 margin 後, 視窗數, 序列數, 餘裕秒數)]
    """
    need = WIN_SEC + (SEQ_LEN - 1) * STEP_SEC        # 一條序列涵蓋的時間
    out = []
    for name, (raw, m_start) in _labelled_segments().items():
        lab = raw - m_start - CUE_MARGIN_END_SEC
        # ⚠️ 用**樣本數**算，不要用秒數 —— (2.3-0.5)/0.05 在浮點下是 35.999…，
        #    int() 之後少一個視窗，validate() 會比實際保守。
        #    dataset.build 的切窗迴圈本來就是用樣本數，這裡跟它一致。
        n_samp = int(round(lab * FS))
        nw = (n_samp - WIN_SAMPLES) // STEP_SAMPLES + 1 if n_samp >= WIN_SAMPLES else 0
        ns = max(nw - SEQ_LEN + 1, 0)
        out.append((name, raw, lab, nw, ns, lab - need))
    return out


def validate() -> list[str]:
    """檢查時間參數是否自洽。回傳問題清單（空 list = 沒問題）。

    ★ 為什麼需要這個：`train.make_sequences` 對「視窗數 < SEQ_LEN」的群組是
      `continue` —— **不報錯、靜默丟掉**。所以參數設錯時不會有任何徵兆，
      直到你發現某個類別完全沒有訓練資料。錄之前就要擋下來。
    """
    problems = []
    need = WIN_SEC + (SEQ_LEN - 1) * STEP_SEC
    if CUE_PREVIEW_SEC >= CUE_REST_SEC:
        problems.append(
            f"CUE_PREVIEW_SEC({CUE_PREVIEW_SEC}) ≥ CUE_REST_SEC({CUE_REST_SEC})："
            f"預告會吃掉整個休息段，rest 類別將完全沒有資料")
    for name, raw, lab, nw, ns, slack in segment_report():
        if ns <= 0:
            problems.append(
                f"{name} 只有 {raw:.2f}s，扣掉 margin 剩 {lab:.2f}s，"
                f"不足一條序列所需的 {need:.2f}s（差 {need - lab:.2f}s）"
                f" → **該類別會靜默產生 0 條訓練序列**")
        elif slack < 0.3:
            problems.append(
                f"{name} 餘裕只有 {slack:+.2f}s（僅 {ns} 條序列），"
                f"建議加長至少 {0.3 - slack:.2f}s 以免資料量過少")
    return problems

# 標籤來源優先序：有 cue 檔就用 cue（可靠），否則退回檔名 + 活動門檻（舊資料）
LABEL_SOURCE_PREFER_CUES = True

# 舊的「一檔一手勢、無 cue」資料預設**不使用** ——
# 手勢與錄音檔完全共線，實測跨錄音掉到亂猜（見 docs/pipeline-v2-design.md 第 6 節）。
# 要拿來對照時再設成 True。
USE_LEGACY_DATA = False

# ---------------- 模型 ----------------
# ★ 從 Conv+BiLSTM(128)+Attention（355k 參數）改成深度可分離時間卷積 + 多頭。
#   CPU 實測：模型 0.64ms vs 特徵抽取 3.77ms —— 瓶頸不是模型。
#   縮小的真正理由是 (1) LSTM 在 CPU 上延遲抖動大 (2) 資料量小易過擬合
#   (3) 拆多頭後每個頭的訓練訊號更少。詳見 model.py docstring。
MODEL_WIDTH = 48                # backbone 通道數。實測：寬度對延遲幾乎無影響，只影響容量
MODEL_BLOCKS = 2                # SepConv 區塊數，dilation 1,2 → 感受野 7（SEQ_LEN=10，
                                # 其餘由 attention pooling 覆蓋）。★ 延遲只跟這個線性相關
MODEL_KERNEL = 3
MODEL_DROPOUT = 0.1
CPU_THREADS = 2                 # 筆電還要同時收樣本+控制，不能吃滿核心

# 舊架構參數（保留供對照，新流程不用）
CNN_FILTERS = 64
CNN_KERNEL = 3
POOL_SIZE = 2
BILSTM_UNITS = 128
BILSTM_DROPOUT = 0.2
FC_UNITS = 256
FC_DROPOUT = 0.5

# ---------------- 訓練 ----------------
LR = 1e-4                       # Lee et al.: Adam lr=1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 20
WEIGHT_DECAY = 1e-5
SEED = 42


if __name__ == "__main__":
    # 不用開始錄製就能檢查時間參數。改秒數之後先跑這個。
    print("=" * 60)
    print("cue 時程")
    print(f"  準備 prepare  {CUE_PREPARE_SEC:5.1f}s  （丟棄）")
    print(f"  維持 go       {CUE_ACTION_SEC:5.1f}s  ← 標姿勢")
    print(f"  回中性 return {CUE_RETURN_SEC:5.1f}s  （丟棄）")
    print(f"  放鬆 release  {CUE_REST_SEC:5.1f}s  ← 標 rest，但最後 "
          f"{CUE_PREVIEW_SEC:.1f}s 被 preview 佔用")
    print(f"    └ 預告 preview {CUE_PREVIEW_SEC:5.1f}s （丟棄，**在 rest 段內**，"
          f"不是額外時間）")
    # preview 不另外加 —— 它發生在 rest 段的尾巴
    per = CUE_PREPARE_SEC + CUE_ACTION_SEC + CUE_RETURN_SEC + CUE_REST_SEC
    n_trial = len(ACTIONS) * CUE_REPS
    total = CUE_LEAD_SEC * 2 + per * n_trial
    print(f"\n  {len(ACTIONS)} 姿勢 × {CUE_REPS} 次 = {n_trial} trial"
          f"，每 trial {per:.1f}s，總長 {total:.0f}s ≈ {total/60:.1f} 分鐘")

    span = WIN_SEC + (SEQ_LEN - 1) * STEP_SEC
    print(f"\n一條序列涵蓋 {span:.2f}s"
          f"（WIN_SEC {WIN_SEC} + {SEQ_LEN-1} × STEP_SEC {STEP_SEC}）")
    print(f"margin 起始 go={CUE_MARGIN_START_SEC}s / "
          f"rest={CUE_MARGIN_REST_START_SEC}s（吸收肌肉放鬆的生理衰減）"
          f" / 結束 {CUE_MARGIN_END_SEC}s")
    print("\n每個 trial 產生的訓練資料：")
    for name, raw, lab, nw, ns, slack in segment_report():
        print(f"  {name:16s} {raw:4.1f}s → 扣邊界 {lab:4.1f}s → "
              f"{nw:4d} 視窗 → {ns:4d} 條序列   （餘裕 {slack:+.2f}s）")

    problems = validate()
    print()
    if problems:
        print("❌ 時間參數有問題，錄了也用不了：")
        for p in problems:
            print(f"  • {p}")
        raise SystemExit(1)
    print("✅ 時間參數自洽")
    print("=" * 60)
