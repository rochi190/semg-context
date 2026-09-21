"""
即時控制模組：可直接部署到本機（STM32 + ADS1299 + Lite 6）。

這支把 SoftPINCH `real_time_operation.py` 裡「模型輸出 → 馬達指令」之間那層搬過來，
並拿掉 pytrigno（Delsys 專用、僅限 Windows）依賴，改成你自己餵資料進來。

包含四個元件：
    StreamingPreprocessor  串流前處理（濾波器帶狀態，只處理新樣本，不重算歷史）
    PredictionVoter        多數決 + 信心門檻（忠實移植 PredictionVoting）
    GripperStateMachine    夾爪狀態機（由 StateLogic 改寫成夾爪二態）
    RealtimeController     把上面串起來，你只要餵 raw chunk 進去

與論文原版的差異（都有理由，不是隨意改的）：
    1. notch 頻率 50Hz → 60Hz（台灣電網）
    2. 正規化不用 checkpoint 附的 mu/sigma——那是 Delsys + subject_0 的絕對 mV 值，
       套在 ADS1299 上會系統性偏移（本專案早期踩過的 bug）。改成開機校準期自行估計。
    3. 狀態機由「每指 Index/Thumb」簡化成「夾爪開/關」——Lite 6 這階段只需要二態。
    4. required_votes 在原版其實被註解掉了（real_time_operation.py:350-351），
       這裡保留成可開關的參數，預設沿用原版行為（不強制票數）。

CPU 即可，不需要顯卡：模型 42K 參數 / 165 KB，單次推論約 0.5 ms。
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import median_filter
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi

# ---- 訊號參數（對齊論文，除了 notch 頻率）----
FS = 2000
NOTCH_FREQ = 60.0          # 台灣電網；論文丹麥是 50
BANDPASS = (20.0, 450.0)
HAMPEL_HALF_WINDOW = 100   # 對齊 HAMPEL_WINDOWSIZE=100
HAMPEL_SIGMA = 2.0         # 對齊 HAMPEL_SIGMA=2
RMS_WINDOW = 500           # 250 ms
RMS_STEP = 50              # 25 ms → 包絡線 40 Hz

MODEL_WINDOW_PTS = 120     # 3 秒 = 模型訓練長度，實測不該縮短
SEC_PER_PT = RMS_STEP / FS

PRED_MAPPING = {0: "Index Contract", 1: "Index Release",
                2: "Thumb Contract", 3: "Thumb Release", 4: "Rest"}


def to_gripper(label: str) -> str:
    """5 類 → 夾爪三態。夾爪不在乎是食指還是拇指。"""
    if label == "Rest":
        return "Rest"
    return "Contract" if "Contract" in label else "Release"


# ======================================================================
class StreamingPreprocessor:
    """濾波器帶狀態，每次只處理新進來的樣本，不重算歷史。

    離線批次每次重算整段（Hampel 是 O(N·k)，最貴）；串流只處理新樣本，
    實測從 108 ms 降到 0.42 ms。這是能即時的關鍵。
    """

    def __init__(self, fs=FS, notch_freq=NOTCH_FREQ, bandpass=BANDPASS,
                 hampel_half=HAMPEL_HALF_WINDOW, hampel_sigma=HAMPEL_SIGMA,
                 rms_window=RMS_WINDOW, rms_step=RMS_STEP, n_ch=3):
        self.n_ch = n_ch
        self.hampel_half = hampel_half
        self.hampel_sigma = hampel_sigma
        self.rms_window = rms_window
        self.rms_step = rms_step

        self.b_n, self.a_n = iirnotch(notch_freq, notch_freq / 2, fs=fs)
        self.b_b, self.a_b = butter(4, [bandpass[0] / (fs / 2), bandpass[1] / (fs / 2)],
                                     btype="band")
        self._zi_n = [lfilter_zi(self.b_n, self.a_n) * 0.0 for _ in range(n_ch)]
        self._zi_b = [lfilter_zi(self.b_b, self.a_b) * 0.0 for _ in range(n_ch)]

        # Hampel 需要前後文；RMS 需要湊滿一個視窗
        self._hist = [np.zeros(0) for _ in range(n_ch)]     # 濾波後、待 hampel 的歷史
        self._rms_carry = [np.zeros(0) for _ in range(n_ch)]  # hampel 後、湊 RMS 用

    def reset(self):
        self.__init__(n_ch=self.n_ch)

    def add_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """chunk: (n_samples, n_ch) 原始訊號（已轉 mV、已去 DC）。
        回傳這次新產生的 RMS 點 (n_new_pts, n_ch)，可能是空的。"""
        outs = []
        for c in range(self.n_ch):
            x = chunk[:, c]
            y, self._zi_n[c] = lfilter(self.b_n, self.a_n, x, zi=self._zi_n[c])
            y, self._zi_b[c] = lfilter(self.b_b, self.a_b, y, zi=self._zi_b[c])

            # Hampel：把歷史尾巴接上新資料一起濾，只取新的部分
            buf = np.concatenate([self._hist[c], y])
            k = 2 * self.hampel_half + 1
            if len(buf) >= k:
                med = median_filter(buf, size=k, mode="nearest")
                mad = 1.4826 * median_filter(np.abs(buf - med), size=k, mode="nearest")
                filt = np.where(np.abs(buf - med) > self.hampel_sigma * mad, med, buf)
            else:
                filt = buf
            new_hampel = filt[len(self._hist[c]):]
            self._hist[c] = buf[-self.hampel_half:]          # 留下必要的歷史

            # RMS：湊滿一個視窗才輸出一個點
            work = np.concatenate([self._rms_carry[c], new_hampel])
            pts, i = [], 0
            while i + self.rms_window <= len(work):
                pts.append(np.sqrt(np.mean(work[i:i + self.rms_window] ** 2)))
                i += self.rms_step
            self._rms_carry[c] = work[i:] if i else work
            outs.append(np.array(pts))

        n = min(len(o) for o in outs) if outs else 0
        if n == 0:
            return np.zeros((0, self.n_ch))
        return np.stack([o[:n] for o in outs], axis=1)


# ======================================================================
class PredictionVoter:
    """忠實移植 real_time_operation.py::PredictionVoting（第 318-366 行）。

    原版邏輯：取眾數 → 只算「該眾數類別」的平均信心 → 超過門檻才輸出，否則 None。
    注意原版的 required_votes 檢查是被註解掉的（第 350-351 行），
    這裡保留成參數，預設 None = 沿用原版行為。
    """

    def __init__(self, window_size: int = 5, conf_threshold: float = 0.9,
                 required_votes: int | None = None):
        self.window_size = window_size
        self.conf_threshold = conf_threshold
        self.required_votes = required_votes
        self.pred_buffer: deque = deque(maxlen=window_size)
        self.conf_buffer: deque = deque(maxlen=window_size)

    def reset(self):
        self.pred_buffer.clear()
        self.conf_buffer.clear()

    def update(self, prediction: str, confidence: float) -> str | None:
        self.pred_buffer.append(prediction)
        self.conf_buffer.append(confidence)
        if len(self.pred_buffer) < self.window_size:
            return None

        voted, count = Counter(self.pred_buffer).most_common(1)[0]
        if self.required_votes is not None and count < self.required_votes:
            return None

        confs = [c for p, c in zip(self.pred_buffer, self.conf_buffer) if p == voted]
        return voted if float(np.mean(confs)) > self.conf_threshold else None


# ======================================================================
@dataclass
class GripperState:
    mode: str = "OPEN"          # OPEN / CLOSED
    busy: bool = False          # 致動中，忽略所有預測
    busy_until: float = 0.0     # 模擬/保護用的致動完成時間
    n_close: int = 0
    n_open: int = 0


class GripperStateMachine:
    """由 real_time_operation.py::StateLogic（第 434-499 行）改寫成夾爪二態。

    原版是每根手指一個狀態；Lite 6 這階段只需要夾爪開/關，所以簡化。
    保留最重要的 `busy` 機制：動作啟動後在致動完成前忽略所有預測——
    這是原版抑制誤觸發最有效的一環。

    actuator_sec: 致動器動作耗時。真機上應改由致動器回報 finished，
                  這裡用時間近似，方便離線評估與沒有回饋時的保護。
    """

    def __init__(self, actuator_sec: float = 0.8):
        self.actuator_sec = actuator_sec
        self.state = GripperState()

    def reset(self):
        self.state = GripperState()

    def update(self, voted: str | None, now: float) -> tuple[GripperState, str | None]:
        """回傳 (狀態, 這一步發出的指令或 None)。"""
        if self.state.busy:
            if now >= self.state.busy_until:      # 真機請改成致動器回報 finished
                self.state.busy = False
            return self.state, None

        if voted is None:                          # 信心不足 → 不動作
            return self.state, None

        cmd = None
        if self.state.mode == "OPEN" and voted == "Contract":
            self.state.mode, self.state.busy = "CLOSED", True
            self.state.busy_until = now + self.actuator_sec
            self.state.n_close += 1
            cmd = "CLOSE"
        elif self.state.mode == "CLOSED" and voted == "Release":
            self.state.mode, self.state.busy = "OPEN", True
            self.state.busy_until = now + self.actuator_sec
            self.state.n_open += 1
            cmd = "OPEN"
        return self.state, cmd


# ======================================================================
class RealtimeController:
    """把上面全部串起來。你只要持續呼叫 push()，它回傳夾爪指令。

    用法：
        ctl = RealtimeController(model_path)
        ctl.calibrate(baseline_chunk)      # 開機時錄一段（含休息與出力）估 mu/sigma
        while True:
            chunk = read_from_stm32()      # (n_samples, 3) mV
            cmd = ctl.push(chunk, time.time())
            if cmd: send_to_lite6(cmd)
    """

    def __init__(self, model_path: Path, actuator_sec: float = 0.8,
                 voter_window: int = 5, conf_threshold: float = 0.9,
                 required_votes: int | None = None, torch_threads: int = 2):
        torch.set_num_threads(torch_threads)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        from semg.core.softpinch_model import SingleNet_CNN_LSTM
        self.model = SingleNet_CNN_LSTM(**ckpt["model_args"])
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.pre = StreamingPreprocessor()
        self.voter = PredictionVoter(voter_window, conf_threshold, required_votes)
        self.fsm = GripperStateMachine(actuator_sec)

        self.env_buf: deque = deque(maxlen=MODEL_WINDOW_PTS)
        self.mu = np.zeros(3)
        self.sigma = np.ones(3)
        self.calibrated = False
        self.last_raw_pred: str | None = None
        self.last_voted: str | None = None

    # ---- 校準：取代論文的 mu.npy/sigma.npy ----
    def calibrate_from_env(self, env: np.ndarray):
        """env: (n_pts, 3) 校準期的 RMS 包絡線。建議錄 20-30 秒，含休息與數次出力。"""
        self.mu, self.sigma = env.mean(axis=0), env.std(axis=0)
        self.calibrated = True

    def calibrate(self, raw_chunk: np.ndarray):
        pre = StreamingPreprocessor()
        env = pre.add_chunk(raw_chunk)
        if len(env) < 10:
            raise ValueError("校準資料太短，至少需要數秒")
        self.calibrate_from_env(env)

    # ---- 主迴圈 ----
    def push(self, raw_chunk: np.ndarray, now: float) -> str | None:
        """餵入新的原始樣本，回傳夾爪指令（"CLOSE"/"OPEN"/None）。"""
        new_pts = self.pre.add_chunk(raw_chunk)
        for p in new_pts:
            self.env_buf.append(p)

        if len(self.env_buf) < MODEL_WINDOW_PTS or not self.calibrated:
            return None

        x = (np.array(self.env_buf) - self.mu) / (self.sigma + 1e-8)
        with torch.no_grad():
            logits, _, _ = self.model(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
            prob = torch.softmax(logits, dim=1)
            i = int(logits.argmax(1).item())
        self.last_raw_pred = to_gripper(PRED_MAPPING[i])
        conf = float(prob[0, i].item())

        self.last_voted = self.voter.update(self.last_raw_pred, conf)
        _, cmd = self.fsm.update(self.last_voted, now)
        return cmd
