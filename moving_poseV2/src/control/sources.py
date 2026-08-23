"""PredictionSource：mock / replay / live 三種 Prediction 來源（規格書 §7）。

控制層不負責推論——推論全部委派給既有的 `semg.v2.infer.Recognizer`。
這裡只負責「把某種輸入轉成 Prediction 串流」，餵給 `ArmController`。

★ `Recognizer` 與 `config` 實際載入的是 `semg_realtime_V4/` 這個自足打包
（已訓練模型 + 修正過 ch2/ch4 標籤的程式碼），不是 `src/semg/` 舊快照——
由呼叫端（`run_arm_control.py`）在 import 這個模組之前把
`semg_realtime_V4/src` 插到 `sys.path` 最前面決定，這裡不用管路徑。

mock 來源仍然保留：不需要模型也不需要硬體，是驗證控制層邏輯本身
（映射/限位/去彈跳/安全機制）最快的手段（規格書 §1.3）。
"""
from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from semg.v2 import config as C

from control import arm_config as AC
from control.arm_controller import Prediction


class PredictionSource:
    """介面：`poll()` 回傳這次呼叫以來新產生的 Prediction 列表（可能是空 list）。"""

    def poll(self, now: float | None = None) -> list[Prediction]:
        raise NotImplementedError

    def is_ready(self) -> bool:
        """來源是否已經可以穩定產出 Prediction。

        為什麼需要這個：`LiveSource` 的前 `calib_sec` 秒在做校準，`poll()`
        一律回傳空 list。若在這段期間 `engage()`，`WATCHDOG_SEC=0.5s` 會
        【必然】觸發並把手臂推進 SAFE，而使用者會誤以為是串列或模型壞了。
        呼叫端必須在 `engage()` 之前確認本方法回傳 True。

        預設 True——mock / manual 在建構完成後就能立刻產出。
        """
        return True

    def status_text(self) -> str:
        """給終端機顯示的一行來源狀態（例如校準進度）。預設空字串。"""
        return ""

    def resync(self, now: float | None = None) -> None:
        """把來源的時間基準對齊到 `now`，並丟棄已累積的待處理資料。

        為什麼需要這個：`ArmController.engage()` 內含 `wait=True` 的回 home，
        會阻塞數秒；期間 `LiveSource` 的 reader thread 持續累積樣本、
        `ReplaySource` 的播放時鐘持續落後。若不對時，`engage()` 之後的第一次
        `poll()` 會一次吐出數十筆 Prediction，全部帶著同一個 `now` 灌進
        `DofDebouncer`，讓「VOTE_M 幀 = VOTE_M × STEP_SEC 秒」的假設完全失效，
        可能瞬間湊滿多數決並觸發轉態。
        """
        pass

    def notify_armed(self, now: float | None = None) -> None:
        """`ArmController.engage()` 成功、真的進入 ARMED 之後呼叫一次。

        預設不做事——大多數來源（mock/live）不需要區分「還沒 ARM」跟
        「已經 ARM」。`ReplaySource` 覆寫它來把播放時鐘的啟動點延後到這裡，
        見那裡的說明。
        """
        pass

    def close(self) -> None:
        pass


class MockSource(PredictionSource):
    """完全離線的假手勢來源：依固定腳本切換 (hand, elbow, shoulder) 姿勢，
    驗證控制層狀態機邏輯，不需要真模型也不需要硬體（規格書 §7 用法 A/B）。

    依 STEP_SEC（20 Hz）節奏輸出，行為與真分類器的輸出頻率一致，這樣
    debounce 的 VOTE_M/VOTE_N/REFRACTORY（以「幀」為單位）才有意義。
    """

    _SCRIPT: tuple[tuple[float, str, str, str], ...] = (
        # (維持秒數, hand, elbow, shoulder)
        (2.0, "rest", "rest", "rest"),
        (2.0, "index", "rest", "rest"),
        (2.0, "thumb", "rest", "rest"),
        (1.5, "pinch", "rest", "rest"),      # 上升緣 → 夾爪 CLOSE
        (1.5, "rest", "rest", "rest"),
        (1.5, "pinch", "rest", "rest"),      # 上升緣 → 夾爪 OPEN
        (2.0, "rest", "flex", "rest"),        # 手肘彎曲
        (2.0, "rest", "rest", "flex"),         # 肩前舉
        (2.5, "pinch", "flex", "rest"),         # pick_and_lift
        (2.0, "rest", "rest", "rest"),
    )

    def __init__(self, loop: bool = True, conf: float = 0.95):
        self.loop = loop
        self.conf = conf
        self._total = sum(seg[0] for seg in self._SCRIPT)
        self._t0 = time.monotonic()
        self._next_t = self._t0
        self._done = False

    def _state_at(self, elapsed: float):
        if elapsed >= self._total:
            if not self.loop:
                return None
            elapsed %= self._total
        acc = 0.0
        for dur, hand, elbow, shoulder in self._SCRIPT:
            acc += dur
            if elapsed < acc:
                return hand, elbow, shoulder
        return self._SCRIPT[-1][1:]

    def poll(self, now: float | None = None) -> list[Prediction]:
        now = time.monotonic() if now is None else now
        out: list[Prediction] = []
        while not self._done and self._next_t <= now:
            state = self._state_at(self._next_t - self._t0)
            if state is None:
                self._done = True
                break
            hand, elbow, shoulder = state
            out.append(Prediction(
                t=self._next_t,
                hand=C.DOF_STATES["hand"].index(hand),
                elbow=C.DOF_STATES["elbow"].index(elbow),
                shoulder=C.DOF_STATES["shoulder"].index(shoulder),
                p_hand=self.conf, p_elbow=self.conf, p_shoulder=self.conf,
            ))
            self._next_t += C.STEP_SEC
        return out

    def is_ready(self) -> bool:
        return not self._done

    def resync(self, now: float | None = None) -> None:
        # 腳本從頭重播——重新 ARM 時從已知的起點開始，比從中間接續好驗證。
        now = time.monotonic() if now is None else now
        self._t0 = now
        self._next_t = now
        self._done = False


class ManualSource(PredictionSource):
    """手動選擇來源：外部（例如一個輸入 thread）呼叫 `set_state()` 指定目前要
    模擬的分類器輸出，`poll()` 依 STEP_SEC 節奏持續吐出該狀態的 Prediction。

    用於模型還沒訓練好、或想直接驗證控制層（映射/限位/去彈跳/安全機制）時，
    讓使用者自己扮演分類器——信心固定給 1.0，其餘完全走跟真模型一樣的路徑
    （debounce、位置鏡像、pinch toggle、J6 jog 全部照跑）。
    """

    def __init__(self, conf: float = 1.0):
        self.conf = conf
        self._lock = threading.Lock()
        self._hand = 0
        self._elbow = 0
        self._shoulder = 0
        self._next_t = time.monotonic()

    def set_state(self, hand: int, elbow: int, shoulder: int) -> None:
        with self._lock:
            self._hand, self._elbow, self._shoulder = hand, elbow, shoulder

    def poll(self, now: float | None = None) -> list[Prediction]:
        now = time.monotonic() if now is None else now
        with self._lock:
            hand, elbow, shoulder = self._hand, self._elbow, self._shoulder
        out: list[Prediction] = []
        while self._next_t <= now:
            out.append(Prediction(
                t=self._next_t, hand=hand, elbow=elbow, shoulder=shoulder,
                p_hand=self.conf, p_elbow=self.conf, p_shoulder=self.conf,
            ))
            self._next_t += C.STEP_SEC
        return out

    def resync(self, now: float | None = None) -> None:
        # 沒有「補完歷史」的問題（永遠吐目前的選擇），但若 poll() 隔了很久沒被
        # 呼叫（例如 engage() 的 wait=True 阻塞期間），_next_t 會落後 now 很多，
        # 下一次 poll() 會一次吐出一大串同狀態的 Prediction——時間戳對不上
        # STEP_SEC 節奏，一樣會讓去彈跳的時間語意跑掉，所以還是要對時。
        self._next_t = time.monotonic() if now is None else now


class ReplaySource(PredictionSource):
    """重播已錄的 CSV，過真模型（`semg.v2.infer.Recognizer`），產生真 Prediction。

    離線先把整段算完（跟 `infer.replay()` 同樣的流程），再依 STEP_SEC 節奏
    播放，讓 debounce/看門狗能用真實的節奏驗證，而不是一次性灌爆。

    ★ 播放時鐘要等 `notify_armed()`（真的 ARMED 之後）才開始走（2026-08-19
    使用者回報）。`run_arm_control.py` 的主迴圈在「按 Enter 進入 ARMED」的
    等待期間也會持續呼叫 `poll()`；若播放時鐘從建構那刻就開始跑，這段等待
    時間會把 queue 空耗掉——DISARMED 期間 `on_prediction()` 仍會被呼叫，
    但手臂不動，被跳過的那段 Prediction 就永遠沒了。結果是使用者按下 Enter
    之後看到的，其實已經是錄音中段，整段重播看起來「比預期快、提早結束」。
    """

    def __init__(self, csv_path: Path, model_path: Path, calib_sec: float = 5.0,
                dwell: int = AC.DEFAULT_DWELL):
        from semg.v2 import cues as Q
        from semg.v2 import preprocess as P
        from semg.v2.infer import Recognizer

        if not Path(model_path).exists():
            raise SystemExit(f"找不到模型 {model_path}，請先跑 train.py 或改用 --source mock")

        # ★ 讀 Recognizer 的 `stable`（votes+dwell 處理過），不是逐窗抖動的
        #   `raw`。votes 設 1 是關掉 Recognizer 自己的多數決——那一層跟
        #   ArmController.debounce 的 VOTE_M/VOTE_N 功能重疊，疊兩層只會疊加
        #   延遲；但 dwell 沒有重複，它擋的是「移動過程中穩定地標成錯的狀態」，
        #   單純投票（不管在哪一層）都擋不住，因為那種誤判本身就很穩定
        #   （見 semg_realtime_V4/README.md「移動過程的誤判」）。詳見 arm_config.py。
        self.rec = Recognizer(model_path, votes=AC.RECOGNIZER_VOTES,
                              conf=AC.RECOGNIZER_CONF, dwell=dwell)
        self._idx = {name: i for i, name in enumerate(self.rec.dof_names)}

        cue_csv = Q.cue_path_for(csv_path)
        raw = P.load_recording(csv_path, trim=cue_csv is None)
        if cue_csv is not None:
            cs, ce = Q.calibration_span(Q.load_cues(cue_csv), len(raw))
            n_cal = ce - cs
            print(f"[replay] 校準區段取自 cue 檔（與訓練同來源）：0–{n_cal / C.FS:.2f}s")
        else:
            n_cal = int(calib_sec * C.FS)
        if n_cal > 0:
            self.rec.calibrate(raw[:n_cal])

        preds: list[Prediction] = []
        block = C.STEP_SAMPLES
        for s0 in range(n_cal, len(raw) - block + 1, block):
            for res in self.rec.push(raw[s0:s0 + block]):
                preds.append(self._to_pred(res))
        self._preds: deque[Prediction] = deque(preds)
        self._t_next = time.monotonic()
        self._started = False          # 播放時鐘要等 notify_armed() 才起跑，見 class docstring
        self._announced_end = False
        print(f"[replay] 離線算出 {len(preds)} 筆預測，依 STEP_SEC={C.STEP_SEC}s 節奏播放"
              "（按 Enter 進入 ARMED 後才開始播放）")

    def _to_pred(self, res: dict) -> Prediction:
        # stable[j] 在還沒穩定過時是 None——跟 ArmController 自己的慣例一致，
        # 未穩定就當 rest（index 0，config.DOF_STATES 每個 DOF 都以 rest 開頭）。
        stable, conf = res["stable"], res["conf"]
        hand = stable[self._idx["hand"]]
        elbow = stable[self._idx["elbow"]]
        shoulder = stable[self._idx["shoulder"]]
        return Prediction(
            t=0.0,
            hand=hand if hand is not None else 0,
            elbow=elbow if elbow is not None else 0,
            shoulder=shoulder if shoulder is not None else 0,
            p_hand=conf[self._idx["hand"]], p_elbow=conf[self._idx["elbow"]],
            p_shoulder=conf[self._idx["shoulder"]],
        )

    def poll(self, now: float | None = None) -> list[Prediction]:
        now = time.monotonic() if now is None else now
        if not self._started:
            return []
        out: list[Prediction] = []
        while self._preds and now >= self._t_next:
            p = self._preds.popleft()
            out.append(Prediction(t=now, hand=p.hand, elbow=p.elbow, shoulder=p.shoulder,
                                  p_hand=p.p_hand, p_elbow=p.p_elbow,
                                  p_shoulder=p.p_shoulder))
            self._t_next += C.STEP_SEC
        if not self._preds and not self._announced_end:
            # 播完之後 poll() 會一直回傳空 list，0.5 秒後 watchdog 會把手臂推進
            # SAFE——不講清楚的話，使用者看到的是「ERROR 進入 SAFE：watchdog」，
            # 會誤以為出錯，其實是重播正常結束。
            self._announced_end = True
            print(f"\n[replay] 重播結束。接下來 watchdog 會在 {AC.WATCHDOG_SEC}s 後"
                  "把手臂推進 SAFE——這是正常結束，不是錯誤。")
        return out

    def is_ready(self) -> bool:
        return bool(self._preds)          # 播完就不再 ready

    def status_text(self) -> str:
        return "" if self._preds else "重播已結束"

    def resync(self, now: float | None = None) -> None:
        # 不丟棄內容（重播的資料是有限且珍貴的），只把播放時鐘對齊到現在，
        # 避免 engage() 阻塞期間累積的落後量在下一次 poll 被一次倒出。
        # 播放還沒開始（notify_armed() 沒被呼叫過）的話這裡對不對時都無所謂
        # ——poll() 會直接因為 _started=False 提早回傳，不會去讀 _t_next。
        self._t_next = time.monotonic() if now is None else now

    def notify_armed(self, now: float | None = None) -> None:
        # 真的 ARMED 了，播放時鐘才開始走（見 class docstring）。之後每次
        # 重新 ARM（例如 SAFE 之後）只會再對一次時，不會讓 queue 重頭來過
        # ——已經播放掉的部分不會重播，跟 resync() 的「不丟棄內容」是同一個
        # 設計：重播是接續，不是重來。
        self._started = True
        self._t_next = time.monotonic() if now is None else now


class LiveSource(PredictionSource):
    """真串列 + 真模型：讀 STM32 UART frame → 濾波/特徵/模型 → Prediction。

    解幀邏輯沿用 `hardware.record_session` 已驗證過的 `find_sync`/`read_one_frame`
    （不重寫第二份實作）。讀取放在獨立 thread，只做「讀 frame → 進佇列」，
    推論留在主 thread 的 `poll()`——絕不能讓推論拖慢讀取導致掉樣本
    （record_session.py 已踩過這個坑，見其 reader_thread 註解）。
    """

    def __init__(self, port: str, baud: int, model_path: Path, calib_sec: float = 5.0,
                dwell: int = AC.DEFAULT_DWELL):
        import serial

        from hardware.record_session import find_sync, read_one_frame
        from semg.v2.infer import Recognizer

        if not Path(model_path).exists():
            raise SystemExit(f"找不到模型 {model_path}，請先跑 train.py 或改用 --source mock")

        # 同 ReplaySource：讀 votes+dwell 處理過的 `stable`，不是逐窗 `raw`。見那裡的說明。
        self.rec = Recognizer(model_path, votes=AC.RECOGNIZER_VOTES,
                              conf=AC.RECOGNIZER_CONF, dwell=dwell)
        self._idx = {name: i for i, name in enumerate(self.rec.dof_names)}

        self.ser = serial.Serial(port, baud, timeout=1)
        self.ser.reset_input_buffer()
        try:
            # 見規格書 §10：80 kB/s 下 OS 驅動緩衝只有 ~51 ms 餘裕，掉包會靜默推進索引。
            self.ser.set_buffer_size(rx_size=262144)
        except AttributeError:
            pass          # 非 Windows 平台的 pyserial 沒有這個方法

        self._q: deque = deque()
        self._stop = False
        self._calibrated = False
        self._calib_deadline = time.monotonic() + calib_sec
        self._calib_chunks: list[np.ndarray] = []
        self._calib_sec = calib_sec

        # reader thread 的健康狀態（2026-08-19 code review FIX-M/FIX-N）。
        self._reader_error: str | None = None
        self._frames_ok = 0
        self._frames_bad = 0
        self._t_first_frame: float | None = None
        self._last_drop_report_t = 0.0

        def _reader():
            miss = 0
            while not self._stop:
                try:
                    if not find_sync(self.ser):
                        # 找不到 SYNC：可能 baud 設錯、STM32 沒在送、接錯 COM port。
                        # 原本是無 sleep 的緊迴圈，會燒滿一個 CPU 核心並跟推論
                        # 搶資源（本專案 CPU_THREADS=2，影響直接）。
                        miss += 1
                        if miss % 200 == 0:
                            time.sleep(0.01)
                        continue
                    r = read_one_frame(self.ser)
                    if r is None:
                        self._frames_bad += 1
                        continue
                    miss = 0
                    _status, ch = r
                    if self._t_first_frame is None:
                        self._t_first_frame = time.monotonic()
                    self._frames_ok += 1
                    self._q.append([ch[c - 1] for c in C.CH_CSV_COLS])
                except (serial.SerialException, OSError) as e:
                    # USB 被拔、STM32 重啟、驅動異常。原本會讓 thread 靜默死亡
                    # ——watchdog 雖然會把手臂推進 SAFE（fail-safe），但沒有任何
                    # 訊息說明是串列埠掛了，使用者會往分類器方向找錯。
                    self._reader_error = f"{type(e).__name__}: {e}"
                    print(f"\n[live] ✖ 串列讀取失敗，reader thread 結束：{e}")
                    return
                except Exception as e:      # noqa: BLE001 不讓未預期例外靜默吞掉
                    self._reader_error = f"{type(e).__name__}: {e}"
                    print(f"\n[live] ✖ reader thread 未預期例外：{e!r}")
                    return

        self._thread = threading.Thread(target=_reader, daemon=True)
        self._thread.start()
        print(f"[live] 已連線 {port} @ {baud} baud，前 {calib_sec:.0f}s 用於估計校準尺度"
              f"（請保持中性放鬆、手臂懸空）")

    def poll(self, now: float | None = None) -> list[Prediction]:
        now = time.monotonic() if now is None else now

        if self._reader_error is not None and not self._q:
            # 讓上層停下來並顯示原因，而不是只靠 watchdog 靜默進 SAFE。
            raise RuntimeError(f"串列來源已中斷：{self._reader_error}")

        if not self._q:
            return []
        n = len(self._q)
        blk = np.array([self._q.popleft() for _ in range(n)], dtype=np.float64) * C.LSB_MV

        if not self._calibrated:
            self._calib_chunks.append(blk)
            if now < self._calib_deadline:
                return []
            cal = np.concatenate(self._calib_chunks, axis=0)
            self.rec.calibrate(cal)
            self._calibrated = True
            self._calib_chunks = []
            print(f"[live] 校準完成（{len(cal)} 樣本），開始輸出 Prediction")
            return []

        out: list[Prediction] = []
        for res in self.rec.push(blk):
            stable, conf = res["stable"], res["conf"]
            hand = stable[self._idx["hand"]]
            elbow = stable[self._idx["elbow"]]
            shoulder = stable[self._idx["shoulder"]]
            out.append(Prediction(
                t=now,
                hand=hand if hand is not None else 0,
                elbow=elbow if elbow is not None else 0,
                shoulder=shoulder if shoulder is not None else 0,
                p_hand=conf[self._idx["hand"]], p_elbow=conf[self._idx["elbow"]],
                p_shoulder=conf[self._idx["shoulder"]],
            ))

        # 每 5 秒回報一次掉包狀況（規格書 §10）。掉包會靜默破壞訊號連續性——
        # 濾波與特徵算錯 → confidence 下降 → 誤判或凍結，而現象與「模型不好」
        # 完全無法區分。必須主動量測。
        if now - self._last_drop_report_t >= 5.0:
            self._last_drop_report_t = now
            ok, bad, rate = self.drop_stats(now)
            if rate > 0.01 or bad > 0:
                print(f"\n[live] ⚠ 掉包率約 {rate * 100:.2f}%"
                      f"（收到 {ok} frame，壞 {bad}）—— 檢查 CPU 負載與 USB")
        return out

    def drop_stats(self, now: float | None = None) -> tuple[int, int, float]:
        """回傳 (收到的 frame 數, 壞 frame 數, 估計掉包率)。

        規格書 §10 的次佳實作：`read_one_frame` 不回傳 frame 內的
        `sample_index`，無法做索引連續性檢查，所以改用 wall-clock 對照——
        期望 frame 數 = 經過秒數 × FS，實收少於期望即代表掉包。
        """
        now = time.monotonic() if now is None else now
        if self._t_first_frame is None:
            return 0, self._frames_bad, 0.0
        elapsed = now - self._t_first_frame
        if elapsed < 1.0:
            return self._frames_ok, self._frames_bad, 0.0
        expected = elapsed * C.FS
        rate = max(0.0, 1.0 - self._frames_ok / expected)
        return self._frames_ok, self._frames_bad, rate

    def is_ready(self) -> bool:
        return self._calibrated

    def status_text(self) -> str:
        if self._reader_error is not None:
            return f"串列已中斷：{self._reader_error}"
        if not self._calibrated:
            remain = max(0.0, self._calib_deadline - time.monotonic())
            return f"校準中 還需 {remain:.1f}s（請保持中性放鬆、手臂懸空）"
        return ""

    def resync(self, now: float | None = None) -> None:
        # 丟棄 engage() 阻塞期間累積的原始樣本。即時控制要的是「現在」，
        # 補完數秒前的歷史只會讓去彈跳拿到錯誤的時間語意。
        n = len(self._q)
        self._q.clear()
        if n:
            print(f"\n[live] resync：丟棄 {n} 個累積樣本（{n / C.FS:.2f}s）")

    def close(self) -> None:
        self._stop = True
        self._thread.join(timeout=1.0)
        self.ser.close()
