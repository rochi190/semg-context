"""CLI 進入點：sEMG → UFACTORY Lite 6 即時控制（規格書 §7）。

★ 分類器程式碼與已訓練模型來自 `semg_realtime_V4/`（repo 根目錄下的自足打包，
跟 `moving_pose/` 同層，不在它底下），不是 `src/semg/`（那是舊快照）。
`--model` 預設指到 `semg_realtime_V4/models/final_v4_split.pt`——split 架構
把「沒 cue 過的動作組合」（例如舉肘同時彎食指）的準確率從 5% 拉到 85.4%，
是先前實機測試踩到的問題（該組合被誤判成 pinch）的直接解方。
`--source replay`/`live` 不指定 `--model` 的話就是用這個。

用法：
    # A. 完全離線：假手勢 + 假手臂，驗證狀態機邏輯
    python src/control/run_arm_control.py --source mock --dry-run

    # B. 假手勢 + 真手臂：驗證手臂會動且限位正確（★ 第一次接手臂時用這個）
    python src/control/run_arm_control.py --source mock --arm-ip 192.168.1.166

    # C. 重播已錄的 CSV 過模型 → 真手臂（--model 不給就用預設的 final_v4_split.pt）
    python src/control/run_arm_control.py --source replay --csv data/raw/xxx.csv --arm-ip 192.168.1.166

    # D. 真串列 + 真模型 + 真手臂（最終）
    python src/control/run_arm_control.py --source live --port COM10 --arm-ip 192.168.1.166

操作：按 Enter 進入 ARMED；按 q 或 Ctrl-C 結束（回 home + disconnect）；
按 z 將 J6 手動歸零。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # src/ 加進 sys.path（給 control package）

from control import arm_config as AC                              # noqa: E402

# semg_realtime_V4/src 插在最前面，蓋過 src/semg 的舊快照——真正在用的分類器
# 程式碼與已訓練模型都在 semg_realtime_V4/（見 arm_config.py 的說明）。
sys.path.insert(0, str(AC.SEMG_REALTIME_SRC))
# 必須在這裡呼叫、不能在 arm_config.py 檔尾自動呼叫——validate() 的時序一致性
# 檢查要 import semg.v2.config，得等 semg_realtime_V4 路徑插好之後才能撿到
# 正確的版本（見 arm_config.py 該函式的說明）。
AC.validate()

from control.arm_controller import ArmController, MockArm         # noqa: E402
from control.sources import LiveSource, MockSource, ReplaySource   # noqa: E402

# 只有 Windows（msvcrt）支援非阻塞按鍵。非 Windows 平台 Enter/q/z 全部失效——
# 不要讓它靜默降級，main() 會在啟動時明確警告（2026-08-19 code review FIX-S）。
_KEYS_AVAILABLE = False
try:
    import msvcrt as _msvcrt          # noqa: N812
    _KEYS_AVAILABLE = True
except ImportError:
    _msvcrt = None


def _poll_key() -> str | None:
    """非阻塞讀取一個按鍵；沒有按鍵回傳 None。

    ⚠️ 目前只支援 Windows（msvcrt）。非 Windows 平台永遠回傳 None，代表
    Enter / q / z 全部失效——只能用 --yes 啟動、Ctrl-C 結束。
    """
    if _msvcrt is None:
        return None
    if _msvcrt.kbhit():
        return _msvcrt.getwch()
    return None


def _connect_real_arm(ip: str, timeout: float = 5.0):
    """連線實體手臂，並【確認連線真的成功】。

    ⚠️ SDK 1.18.5 的 `XArmAPI.__init__` 內部就會呼叫 `Base.connect()`
    （`xarm/x3/base.py:282`），IP 不可達時直接 `raise Exception(
    'connect socket failed')`——**建構子不是安全的**。早期版本這裡假設
    「物件一定建得起來、之後再檢查 connected」，結果例外從建構子穿出、
    下面的檢查清單一行都印不到（2026-08-19 實測 traceback 確認）。

    IP 打錯的症狀（畫面全綠、target 在變、手臂不動）與「手臂處於 STOP
    狀態」（SAFE 恢復沒做好）、「夾爪沒裝」幾乎無法從畫面區分，所以這則
    訊息是唯一的分辨依據，必須確實印得出來。

    兩道防線都保留：
      (1) 建構子拋例外 → 攔下來轉成同一則訊息（1.18.5 的實際路徑）
      (2) 建構成功但 `connected` 遲遲不 True → 逾時（防未來 SDK 改成
          背景重試、或 TCP 通了但 report 通道沒建起來）
    """
    try:
        from xarm.wrapper import XArmAPI
    except ImportError:
        sdk_root = AC.REPO_ROOT / "xArm-Python-SDK-master"
        sys.path.insert(0, str(sdk_root))
        from xarm.wrapper import XArmAPI

    def _fail(reason: str) -> SystemExit:
        return SystemExit(
            f"\n✖ 無法連線手臂 {ip}（{reason}）。請檢查：\n"
            f"  1. 控制箱電源與網路線\n"
            f"  2. IP 是否正確（arm_config.ARM_IP = {AC.ARM_IP}）\n"
            f"  3. 電腦與控制箱是否在同一網段（試 ping {ip}）\n"
            f"  4. UFACTORY Studio 是否正佔用連線\n"
            f"※ 沒有手臂時請用 --dry-run。")

    print(f"連線手臂 {ip} …")
    t0 = time.monotonic()
    try:
        arm = XArmAPI(ip, is_radian=False)
    except Exception as e:                 # noqa: BLE001  SDK 拋的是裸 Exception
        raise _fail(f"socket 連線失敗，{time.monotonic() - t0:.1f}s：{e}") from None

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if arm.connected:
            break
        time.sleep(0.2)
    else:
        try:
            arm.disconnect()
        except Exception:                  # noqa: BLE001
            pass
        raise _fail(f"connected 逾時 {timeout:.0f}s")

    # report 通道要幾百 ms 才會推送第一輪狀態；startup() 讀 warn_code /
    # error_code 之前必須等，否則讀到的是還沒更新的 0（誤判為「沒有錯誤」）。
    time.sleep(0.5)
    print(f"已連線。firmware={arm.version}")
    return arm


def _render_status(status: dict, source_note: str = "") -> None:
    line = (f"[{status['state']:8s}] "
            f"hand={status['hand_name']:6s}({status['p_hand']:.2f}) "
            f"elbow={status['elbow_name']:5s}({status['p_elbow']:.2f}) "
            f"shoulder={status['shoulder_name']:5s}({status['p_shoulder']:.2f}) | "
            f"J2={status['J2']:+6.1f} J3={status['J3']:+6.1f} J6={status['J6']:+6.1f} | "
            f"grip={status['grip']:6s} | cmdnum={status['cmdnum']}")
    if source_note:
        line += f" | {source_note}"
    print("\r" + line.ljust(150), end="", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="sEMG → UFACTORY Lite 6 即時控制層")
    ap.add_argument("--source", choices=("mock", "replay", "live"), required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="用 MockArm 取代真手臂，只把指令印到終端機")
    ap.add_argument("--arm-ip", default=AC.ARM_IP)
    ap.add_argument("--csv", type=Path, help="--source replay 用：重播的錄音 CSV")
    ap.add_argument("--model", type=Path, default=AC.DEFAULT_MODEL,
                    help="--source replay/live 用：模型 bundle（預設 semg_realtime_V4 的 "
                         "final_v4_split.pt，同目錄的 _scripted.pt 會被自動優先載入）")
    ap.add_argument("--port", help="--source live 用：序列埠，本專案實際是 COM10")
    ap.add_argument("--baud", type=int, default=1_600_000)
    ap.add_argument("--calib-sec", type=float, default=5.0,
                    help="--source replay/live 用：開頭幾秒估穩健尺度")
    ap.add_argument("--dwell", type=int, default=AC.DEFAULT_DWELL,
                    help="--source replay/live 用：候選新狀態要連續出現幾次才真的換過去，"
                         "擋「移動過程被穩定地標成錯的狀態」（見 semg_realtime_V4/README.md）")
    ap.add_argument("--log", type=Path, default=None, help="控制事件 CSV log 路徑；預設自動產生")
    ap.add_argument("--yes", action="store_true", help="跳過『按 Enter 開始』，直接進 ARMED")
    a = ap.parse_args()

    if a.source == "replay" and not a.csv:
        ap.error("--source replay 需要 --csv")
    if a.source == "live" and not a.port:
        ap.error("--source live 需要 --port")

    print("=" * 78)
    print("⚠️  維持姿勢時手臂必須懸空 —— 手肘/前臂/上臂不能靠在桌面、扶手或身側，")
    print("    否則重力被外物撐住、EMG 消失，系統會判成 rest 讓機械手臂回 home。")
    print("=" * 78)

    if not _KEYS_AVAILABLE:
        print("!" * 78)
        print("⚠️  本平台不支援即時按鍵（僅 Windows 的 msvcrt）：")
        print("    Enter / q / z 全部無效。請用 --yes 自動 ARM，用 Ctrl-C 結束。")
        print("    J6 無法用 z 歸零 —— 結束時 shutdown() 回 home 會一併歸零。")
        print("!" * 78)

    # ★ 先建 source，再連手臂（2026-08-19 code review FIX-Q）。
    #   source 的建構是最容易失敗的一步（模型路徑錯、COM port 被佔用、CSV 不存在），
    #   若它在手臂連線/啟動之後才失敗，例外會讓手臂已 enable、已回 home，
    #   然後程式退出，就這樣留著。把它移到最前面，失敗時手臂根本還沒被碰過。
    print("建立 Prediction 來源…")
    if a.source == "mock":
        source = MockSource()
    elif a.source == "replay":
        source = ReplaySource(a.csv, a.model, calib_sec=a.calib_sec, dwell=a.dwell)
    else:
        source = LiveSource(a.port, a.baud, a.model, calib_sec=a.calib_sec, dwell=a.dwell)

    # 手臂連線失敗時 source 已經建好了（FIX-Q 的順序），而 SystemExit 會
    # 繞過下面的 try/finally → source.close() 不會被呼叫。--source live 時
    # 這代表 COM port 保持開啟、reader thread 留著，行為是「程式卡住不退出」，
    # 又是一個難診斷的形狀。這裡確保任何連線失敗都先收掉 source。
    if a.dry_run:
        arm = MockArm()
    else:
        try:
            arm = _connect_real_arm(a.arm_ip)
        except BaseException:              # 含 SystemExit / KeyboardInterrupt
            try:
                source.close()
            except Exception as e:         # noqa: BLE001
                print(f"[warn] source.close() 失敗：{e!r}")
            raise

    log_path = a.log or (AC.LOG_ROOT / f"control_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    controller = ArmController(arm, log_path=log_path)

    def _try_engage() -> None:
        """把關：來源沒 ready 就不准 ARM（FIX-K）。

        `LiveSource` 的前 `calib_sec` 秒 `poll()` 一律回傳空 list。若在這段
        期間 `engage()`，`WATCHDOG_SEC=0.5s` 會【必然】觸發並把手臂推進
        SAFE，而使用者會誤以為是串列或模型壞了。
        """
        if not source.is_ready():
            print(f"\n[拒絕 ARM] 來源尚未就緒：{source.status_text() or '請稍候'}")
            return
        controller.engage()
        # engage() 內含 wait=True 的回 home，會阻塞數秒；期間來源端持續累積。
        # 若不 resync，下一次 poll 會一次吐出數十筆 Prediction，全部帶同一個
        # now 灌進 DofDebouncer，時間語意崩壞，可能立刻觸發轉態（FIX-L）。
        source.resync(time.monotonic())
        # ReplaySource 的播放時鐘要等這裡（真的 ARMED 了）才開始走，見
        # sources.py 的說明；其他來源這個呼叫預設不做事。
        source.notify_armed(time.monotonic())
        print("\n已 ARMED。按 q 結束；按 z 將 J6 歸零。")

    n_tick = 0
    t_loop0 = time.monotonic()
    try:
        controller.startup()               # ★ 移進 try——見 FIX-Q

        # ★ startup() 的回 home 是 wait=True，真手臂上會阻塞數秒（實測手臂
        #   靜止於全零姿態時 J3 要走 180° @HOME_SPEED=15°/s ≈ 12s）。這段時間
        #   來源端持續累積 Prediction，第一個 tick 會一次吐出數百筆，觸發
        #   「主迴圈可能被阻塞」的假警報——而主迴圈其實沒問題，是 startup()
        #   在阻塞。假警報會讓 checklist 4-6 的「是否出現單 tick 警告」判準
        #   失效（2026-08-17 實測 dry-run 就有 11 筆）。跟 _try_engage() 一樣
        #   做一次 resync。
        source.resync(time.monotonic())

        if a.yes:
            # --yes 不能無條件立刻 engage（FIX-K）：live 來源要等校準完成。
            print("--yes：等待來源就緒後自動 ARM…")
            deadline = time.monotonic() + a.calib_sec + 10.0
            while not source.is_ready() and time.monotonic() < deadline:
                source.poll(time.monotonic())      # 推進校準
                print(f"\r  {source.status_text()}".ljust(78), end="", flush=True)
                time.sleep(0.05)
            print()
            _try_engage()
        else:
            print("按 Enter 進入 ARMED；按 q 結束；按 z 將 J6 歸零。")

        while True:
            now = time.monotonic()
            n_tick += 1

            preds = source.poll(now)
            if len(preds) > AC.MAX_PREDS_PER_TICK:
                # 主迴圈被阻塞導致來源端積壓（FIX-L）。這些 Prediction 全部會
                # 帶同一個 now 進入 DofDebouncer，讓「N 幀 = N × STEP_SEC 秒」
                # 的假設失效。即時控制要的是「現在」，所以只保留最新的幾筆。
                total = len(preds)
                preds = preds[-AC.MAX_PREDS_PER_TICK:]
                print(f"\n[warn] 單 tick 收到 {total} 筆 Prediction，"
                      f"丟棄較舊的 {total - AC.MAX_PREDS_PER_TICK} 筆"
                      f"（主迴圈可能被阻塞）")
            for pred in preds:
                controller.on_prediction(pred, now)

            status = controller.tick(now)
            _render_status(status, source.status_text())

            key = _poll_key()
            if key in ("\r", "\n"):
                _try_engage()
            elif key in ("q", "Q", "\x03"):
                break
            elif key in ("z", "Z"):
                controller.zero_j6()

            # ⚠️ 這是固定 sleep，不是固定週期：實際 tick 間隔 = 10 ms + 工作
            #    時間，而 Windows 的 sleep 粒度約 15.6 ms → 真實約 40–60 Hz，
            #    不是「100 Hz 輪詢主迴圈」（FIX-T）。所有時間相關判定
            #    （CMD_PERIOD、jog、watchdog）都以 time.monotonic() 為基準、
            #    不依賴 tick 數，所以功能上安全；但別在文件裡宣稱 100 Hz，
            #    結束時會印出實測值，以那個為準。
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[Ctrl-C] 中斷，開始收尾…")
    except RuntimeError as e:              # 例如 LiveSource 的「串列來源已中斷」
        print(f"\n[error] 來源異常，開始收尾：{e}")
    finally:
        print()
        hz = n_tick / max(1e-9, time.monotonic() - t_loop0)
        print(f"[info] 主迴圈實測平均 {hz:.0f} Hz（{n_tick} ticks）")
        # ★ 手臂安全優先於資源清理，且兩者各自包 try——任一失敗都不能阻止
        #   另一個執行（2026-08-19 code review FIX-P）。原本 source.close()
        #   拋例外會讓 controller.shutdown() 整段跳過，手臂留在 enable 狀態。
        try:
            controller.shutdown()          # 內部會自行回報是否成功回 home
        except Exception as e:             # noqa: BLE001
            print(f"[error] shutdown 失敗：{e!r}")
            print("        ⚠️ 手臂可能仍 enable，請手動用 UFACTORY Studio 停機。")
        try:
            source.close()
        except Exception as e:             # noqa: BLE001
            print(f"[warn] source.close() 失敗（不影響手臂安全）：{e!r}")


if __name__ == "__main__":
    main()
