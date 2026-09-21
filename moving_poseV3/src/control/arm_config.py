"""控制層所有可調參數集中一處（規格書 §0：所有可調參數都在這裡，程式碼本體沒有魔術數字）。

來源：CLAUDE_CODE_arm_control_spec.md。標 `TODO 待確認` 的是規格書列為
「次要」的項目 —— 先用建議預設值實作，不要停下來等，待實機測試後再調整。
"""
from __future__ import annotations

from pathlib import Path

CONTROL_ROOT = Path(__file__).resolve().parent      # src/control/
SRC_ROOT = CONTROL_ROOT.parent                        # src/
REPO_ROOT = SRC_ROOT.parent                           # moving_pose/
LOG_ROOT = REPO_ROOT / "results" / "control_logs"

# ---------------- 分類器來源套件（semg_realtime_V6）----------------
# 分類器程式碼與已訓練模型都在 semg_realtime_V6/ 這個自足打包裡，control 層
# 一律從這裡載入 `semg.v2.*` / `hardware.record_session`——兩個 CLI 進入點都會
# 把 SEMG_REALTIME_SRC 插在 sys.path 最前面。見 semg_realtime_V6/README.md。
#
# ★ 2026-08-18 起它就在 moving_poseV2/ 底下（本目錄內），整包自足、可以整個
#   搬走或複製而不會斷路徑。舊的 moving_pose/ 佈局不同——那時它是 repo 根目錄
#   下與 moving_pose/ 同層的目錄，寫的是 `REPO_ROOT.parent / ...`。從舊目錄
#   複製檔案過來時要注意這一行，路徑寫錯的症狀是 import 不到 semg.v2。
SEMG_REALTIME_ROOT = REPO_ROOT / "semg_realtime_V6"
SEMG_REALTIME_SRC = SEMG_REALTIME_ROOT / "src"
# 預設模型：split 架構（編碼器依 DOF 拆開，hand 頭看不到近端通道，反之亦然）。
# v3 的共享 backbone 架構在「舉肘同時彎食指」這種沒 cue 過的組合上幾乎失能
# （沒見過的組合 hand 準確率只有 5.0%），這正是先前實機測試踩到的問題
# （抬肘抬肩時彎食指被誤判成 pinch，見 docs/arm-control-usage.md 的歷史記錄）。
# split 架構把這個數字拉到 85.4%，代價是一般情況（組合都見過）小幅變差
# 0.8–3.8pp——即時操作本來就會做沒 cue 過的動作，這個代價換得的安全邊際划算，
# 所以預設用它。見 semg_realtime_V6/README.md「v4 相對 v3 改了什麼」。
DEFAULT_MODEL = SEMG_REALTIME_ROOT / "models" / "final_v6_split.pt"

# ---------------- Recognizer（semg_realtime_V6 的 MultiVoter）----------------
# ⚠️ 這三個參數只影響 Recognizer.push() 回傳的 `stable` 欄位；control 層讀的
#   就是這個 `stable`，不是逐窗抖動的 `raw`——去彈跳的責任因此拆成兩層，
#   各自對付不同問題，缺一不可：
#     Recognizer（這裡）  votes 對付逐窗抖動、dwell 對付「移動過程中穩定地
#                         標成錯的狀態」（純投票擋不住，因為它本來就很穩定）
#     ArmController.debounce（arm_controller.py）  P_MIN/VOTE_M/VOTE_N 對付
#                         殘留抖動、REFRACTORY 對付轉態頻率（防手勢誤觸發連環）
# RECOGNIZER_VOTES 刻意設 1（等於關掉 Recognizer 自己的逐窗多數決）——
# 那一層跟 ArmController 的 VOTE_M/VOTE_N 功能重疊，疊兩層只會疊加延遲，
# 抖動保護留給控制層那份就夠（見 docs/HANDOFF 的延遲分析）。
RECOGNIZER_VOTES = 1
RECOGNIZER_CONF = 0.6              # Recognizer 自己的信心門檻，比較粗；細節交給各 DOF 的 P_MIN
# dwell（遲滯）：候選新狀態要連續出現這麼多次才真的換過去，用來擋「移動過程被
# 穩定地標成錯的狀態」（例如彎 index 途中經過的姿勢被模型判成 pinch）——
# 這跟 VOTE_M/VOTE_N 對付的是不同問題：投票對付逐窗抖動，dwell 對付「穩定地
# 標錯」的轉換期，投票擋不住後者。README 建議先試 6，此處採用同一預設值。
DEFAULT_DWELL = 6

# ---------------- run_live 進入點（live_bridge）的投票分層 ----------------
# ★ 2026-08-19 使用者決定：run_live.py 本來就有 Recognizer 的投票機制，手臂
#   這邊【沿用】它，不要在控制層再疊第二層投票——疊兩層只會疊加延遲（約
#   0.2 s），而它們對付的是同一種東西（逐窗抖動）。
#
#   所以走 live_bridge 的路徑改成【單層投票】：
#     Recognizer     votes=LIVE_RECOGNIZER_VOTES（5，run_live 原本的值）→ 唯一的投票層
#     ArmController  VOTE_M/VOTE_N 改成 1/1 → pass-through，不再投票
#
# ⚠️ 控制層**仍然保留** P_MIN（信心門檻）與 REFRACTORY（不反應期）這兩道閘。
#    它們不是投票，跟 Recognizer 那層沒有重疊：P_MIN 是逐 DOF 的信心門檻
#    （Recognizer 的 conf 是單一粗門檻），REFRACTORY 擋的是「轉態頻率」
#    （防一個動作被拆成兩次觸發，例如 pinch 被誤判成捏兩下）——投票擋不住。
#    dwell 也照常（擋移動過程的穩定誤判，投票同樣擋不住）。
#
# ⚠️ 這組只影響 live_bridge（run_live.py --arm-ip/--arm-dry-run）。
#    run_arm_control.py 的 mock/replay/live 與 run_manual_control.py 仍用
#    上面的 RECOGNIZER_VOTES=1 + VOTE_M/VOTE_N=6/4，行為不變——手動版要能
#    當作隔離排查的對照工具，就不能跟著新路徑一起變。
LIVE_RECOGNIZER_VOTES = 5
LIVE_VOTE_M = 1
LIVE_VOTE_N = 1

# ---------------- 分類器輸出節奏 ----------------
# ★ 這是控制層自己的參數副本，刻意與 semg_realtime_V6/src/semg/v2/config.py
#   分離（控制層參數與訊號/模型參數分開管理，是本專案的既定架構決定）。
#
# ⚠️ 但它的【值】必須與該檔案的 STEP_SEC 相同，因為控制層所有以「幀」為單位
#    的參數都靠它換算成秒：VOTE_M、VOTE_WINDOW_SEC、MAX_PREDS_PER_TICK 都是
#    由它推導；REFRACTORY / WATCHDOG_SEC 是秒，要跟幀數對得上也靠它。
#    若你調整了 semg_realtime_V6 那邊的 STEP_SEC（模型輸出頻率），這裡必須
#    一起改——`validate()` 會檢查兩者是否一致，見檔尾說明。
STEP_SEC = 0.05                   # 20 Hz。必須等於 semg.v2.config.STEP_SEC

# ---------------- 手臂連線 ----------------
ARM_IP = "192.168.1.166"          # 手臂控制箱實際控制 IP（2026-08-19 以實際連線成功更正）
# ⚠️ 這個網段的 IP 在 2026-08-17～19 之間反覆更正過三次，這裡記錄最終依據：
#   .181 → 連線失敗（xArm Studio 自己用的 IP，不是給 Python SDK 連的）
#   .180 → 當時「推測」是對的（使用者的網路配置說明），但**從未實際連線驗證過**
#   .166 → 2026-08-19 用 XArmAPI 實際連線成功，讀到 firmware=v2.7.1、sn=LI1006、
#          axis=6，是唯一有【真實連線證據】支持的值，以此為準。
#   coffee.py 裡寫的 .181 是誤用，不要照抄；若之後又有人說要改回 .180，
#   先問「有沒有實際連線成功過」，光靠推論不算數。
HOME_SPEED = 15.0                 # 回 home 用的較慢速度（°/s）。2026-08-17 從 30 調低
COLLISION_SENSITIVITY = 3         # 0~5，中等靈敏度

# ---------------- 控制模式（2026-08-18 由 mode 0 改成 mode 6）----------------
# ★ 為什麼改：mode 0（位置控制）的指令是【排隊】的——手臂必須能停在每一個
#   waypoint，所以連續送小步位置點時每步都會減速到零，造成規律頓挫。
#   實機比對 xArm Studio 的 jog 明顯平滑很多，證明硬體做得到，是下指令的
#   方式不對。加大步距（JOG_PERIOD 0.10→0.25）只把「多次小頓」換成
#   「少次大頓」，實測「速度變快、頓的次數變少，但平滑度沒明顯改善」。
#
#   mode 6 = joint online trajectory planning：**新指令會中斷正在執行的指令**
#   並重新規劃（見 SDK 範例 2006-joint_online_trajectory_planning.py），
#   所以「必須能停在每個 waypoint」這個限制消失了 → 連續送目標角會被平滑地
#   銜接成連續運動。
#
#   ★ 相對 mode 4（關節速度控制）的優勢：mode 6 仍然是**位置語意**，所以
#     J2/J3 的位置鏡像不用改寫成 P 控制器、`clamp_angles()` 的雙層軟限位
#     照常有效、送指令介面仍是 `set_servo_angle`。mode 4 要把這些全部重做，
#     而且限位保護得自己寫在速度空間，風險高得多。
#
#   ⚠️ mode 是全域的，所以**回 home 這種需要「確實走到定點」的動作要切回
#      mode 0**（startup / engage / shutdown 都會這樣做）。切換序列是
#      `set_mode(n)` → `set_state(0)` → 沉澱約 1 秒（照 SDK 範例）。
#   ⚠️ 出問題想快速退回舊行為：把這個值改成 0 即可，其餘程式不用動。
ARM_MODE_RUN = 6                  # 控制迴圈用的模式（6=線上軌跡規劃，0=傳統位置控制）
ARM_MODE_HOME = 0                 # 回 home 用的模式，必須是 0
ARM_MODE_SETTLE_SEC = 1.0         # 切換模式後的沉澱時間（SDK 範例用 1.0s）

# ---------------- Home 姿態 ----------------
HOME_ANGLES = [90.0, -90.0, 180.0, 0.0, 0.0, 0.0]     # J1..J6，單位：度。J1 2026-08-17 從 0 調成 90

# ---------------- 限位：送出前必須 clamp(SOFT_LIMITS) → clamp(JOINT_LIMITS)，兩層都要 ----------------
# 由 arm.reduced_joint_limits 讀出，2026-08-17，firmware v2.7.1 / sn LI1006。
# ⚠️ SDK 1.18.5 沒有 arm.joint_limits（出廠硬限位沒有直接的 API），只能讀
#    reduced_joint_limits（精簡模式限位）。讀出值與 Lite 6 硬體手冊 V2.6.0
#    的 Joint Range 逐項完全相符（J3 的 -3.5 這個非整數值是最強證據——
#    精簡模式若從未設定過不會湊出這種數字），因此判定為出廠硬限位。
#    讀取腳本：xArm-Python-SDK-master/read_arm_specs.py
# ⚠️ 讀出的陣列有 7 筆，第 7 筆是 [0.0, 0.0]（不存在的 J7，arm.axis=6），忽略。
JOINT_LIMITS = {
    1: (-360.0, 360.0),
    2: (-150.0, 150.0),
    3: (-3.5, 300.0),
    4: (-360.0, 360.0),
    5: (-124.0, 124.0),
    6: (-360.0, 360.0),
}
SOFT_LIMITS = {                   # 本專案自訂的軟限位，比出廠限位更保守
    2: (-90.0, 0.0),               # 肩：下垂 ↔ 平舉
    3: (90.0, 180.0),              # 肘：屈曲 ↔ 伸直
    6: (-180.0, 180.0),            # 腕旋轉。TODO 待確認：實機試了若 jog 範圍不夠可放寬到 ±270
}

# ---------------- 位置鏡像目標角（elbow / shoulder，已確認）----------------
ELBOW_FLEX_ANGLE = 90.0           # J3，屈肘至垂直上臂
SHOULDER_FLEX_ANGLE = 0.0         # J2，平舉水平

# ---------------- J6 jog（index / thumb 持續觸發，已確認方向相反）----------------
JOG_RATE_DEG_S = 45.0              # TODO 待確認：實機試了再調。J6 的平均角速度
CMD_PERIOD = 0.10                  # 送指令節流週期（所有軸共用）

# ★ 2026-08-18：J6 jog 的推進週期【與 CMD_PERIOD 分離】。
#
#   為什麼要分開：使用者比對 xArm Studio 的 jog（明顯平滑很多）後發現，
#   我們每 CMD_PERIOD=0.1s 送一個小位置點的做法，每個點都必須「能停在該點」
#   → 每步都要減速到零 → 規律頓挫。步距越小，加減速佔比越高、頓挫越明顯：
#
#     步距 4.5° (舊, JOG_PERIOD=CMD_PERIOD=0.10)：加速1.64°+減速1.64°+巡航1.23°
#         → 等速只佔 27%，單步 0.130s，實際平均只有 4.5/0.13 ≈ 35°/s（比設定值還慢）
#     步距 11.25° (現行, JOG_PERIOD=0.25)：加速1.64°+減速1.64°+巡航7.98°
#         → 等速佔 71%，單步 0.242s，貼近設定的 45°/s
#
#   ⚠️ 代價：放開手勢時要跑完「當前這一步」才會停，步距越大殘留越多
#      （4.5° → 11.25°，多約 7°）。這個殘留是**疊加在**去彈跳的 ~0.5s
#      延遲之上的（見 docs/arm-control-usage.md）。
#
#   ⚠️ 這只是緩解，不是根治。真正等速要用 mode 4 速度控制
#      （`vc_set_joint_velocity`，Studio 用的機制），但那需要把 J2/J3 的位置
#      鏡像改寫成 P 控制器、軟限位保護重做，屬架構變更——見
#      docs/on-machine-checklist.md 的 M-18。
#
#   ★★ 2026-08-18 改用 mode 6 之後，最佳值【反過來】了：
#      mode 0（排隊、每步必停）→ 步距**越大**越平滑（減少完全停止的次數）
#      mode 6（新指令中斷舊指令）→ 步距**越小**越平滑（軌跡更新更細緻，
#          而且放開手勢時殘留的「當前步」更短）
#      所以改回 0.10（步距 4.5°）。mode 6 下不需要靠大步距來換等速佔比——
#      指令根本不會跑到「停在 waypoint」那一步就被下一個中斷了。
#      ⚠️ 若把 ARM_MODE_RUN 改回 0，這個值應該同時改回 0.25，否則會退回
#         最early那個 27% 等速、實際只有 35°/s 的版本。
JOG_PERIOD = 0.10
JOG_STEP_DEG = JOG_RATE_DEG_S * JOG_PERIOD

# ★ 2026-08-19：使用者回報 J6 jog「卡卡的、不等速」。根因查證（見
#   xArm SDK `xarm/x3/xarm.py` 的 `set_servo_angle`/`move_jointb`）：
#
#   1. 用 JOINT_ACC=200°/s² 時，單一 4.5° 的小步（JOG_STEP_DEG）能達到的
#      三角形速度曲線峰值只有 sqrt(a·d)=sqrt(200×4.5)≈30°/s，**連
#      JOG_RATE_DEG_S=45°/s 都追不到**——每一步都還沒加速到位就要減速，
#      不是穩定的等速巡航，這正是「卡卡的」的來源，加速度不夠是真的問題。
#   2. 更根本的原因：`set_servo_angle` 預設不帶 `radius`（或 <0）時走
#      `MOVE_JOINT`，每個 waypoint 之間手臂會**完全減速到 0** 才開始下一段；
#      要讓連續多個小步真正銜接成平滑運動，必須帶正值 `radius`（走
#      `MOVE_JOINTB`，「關節融合運動」），讓控制器在 waypoint 之間做轉角
#      融合、不完全停下來。這個機制原本完全沒用到。
#
#   兩者要一起做才有效：只加大 accel 沒有 radius，還是每 100ms 停一次；
#   只加 radius 沒有夠高的 accel，追不上目標速率一樣卡。
#
#   J6 jog 專用一組更激進的參數（跟 elbow/shoulder 位置鏡像共用的
#   JOINT_SPEED/JOINT_ACC 分開——那兩軸是少見的大幅度移動，保守值才安全；
#   J6 jog 是連續小步，需要能跟上 JOG_RATE_DEG_S 的巡航速度）：
JOG_SPEED_CAP = 60.0               # °/s，需 > JOG_RATE_DEG_S 才追得上目標速率，遠低於出廠 180

# ★ 2026-08-19：JOG_ACC 由 1100 → 300（使用者回報「thumb/index 轉 J6 時整隻手臂抖動」）。
#
#   1100 是 **mode 0 時代的遺留值**，它的推導前提在 mode 6 下已經不成立：
#     舊推導（2026-08-17，當時 ARM_MODE_RUN=0）：mode 0 的指令會排隊，手臂必須
#     能停在每個 waypoint，所以每一小步都是獨立的「加速—巡航—減速」三角形/
#     梯形曲線。800 時單步峰值 √(800×4.5)=60.0 恰好等於 JOG_SPEED_CAP → 巡航段
#     佔比 0%、全程在加減速 → 規律卡頓；調到 1100 讓巡航門檻降到 60²/1100=3.27°
#     < JOG_STEP_DEG，取回約 27% 巡航段。
#
#   ⚠️ 但 2026-08-18 改成 mode 6（ARM_MODE_RUN=6，線上軌跡規劃）之後，
#      **新指令會中斷舊指令並重新規劃**，根本不會跑到「停在 waypoint」那一步
#      —— 上面整套「單步三角形曲線 / 巡航段佔比」的算法失去意義。
#      留下來的實際效果只剩：每 JOG_PERIOD(0.1s) 對 J6 施加一次接近出廠上限
#      （1100/1145 = 96%）的加速度衝擊 → 力矩脈衝激振整條手臂。
#      這就是「J6 在轉，但抖的是整隻手臂」的來源（與 M-18 那種 J6 自身的
#      規律頓挫是不同現象，不要混為一談）。
#
#   300 °/s² 的依據：mode 6 下 J6 只需要平順地跟上 45 °/s 的目標推進速率，
#   0→45 °/s 只要 0.15 s。代價是放開手勢時多滑約 3.4°（45²/(2×300)），
#   遠小於 M-18 記錄的「加深 lookahead 會多轉 9~18°」那個被否決的方案。
#   ⚠️ 若實機發現 J6 跟不上目標角（持續落後、放手後還在追），才需要往上調；
#      調整前先確認不是 JOG_RATE_DEG_S 設太高。
JOG_ACC = 300.0                    # °/s²，出廠上限 1145（26%）
JOG_BLEND_RADIUS_DEG = 2.0         # 轉角融合半徑，必須 < JOG_STEP_DEG（4.5°），否則 SDK 會拒絕

# ---------------- 出廠運動上限（唯讀參考值，validate() 會據此檢查）----------------
# 由 arm.joint_speed_limit / arm.joint_acc_limit 讀出，2026-08-17，firmware v2.7.1。
# ⚠️ 這兩個屬性回傳的是 [min, max] 全域上下限，**不是 per-joint 陣列**：
#      joint_speed_limit = [0.0573, 180.0]     °/s
#      joint_acc_limit   = [0.5730, 1145.9155] °/s²
#    與 Lite 6 硬體手冊 V2.6.0 的「Joint Motion: Speed 0～180°/s,
#    Acceleration 0～1145°/s²」相符。
FACTORY_JOINT_SPEED_MAX = 180.0
FACTORY_JOINT_ACC_MAX = 1145.9

# ---------------- 指令節流（三個條件都要成立才送，§6.2）----------------
CMD_DEADZONE_DEG = 0.5             # 目標角變化小於此值不送指令
CMDNUM_MAX = 2                     # arm.cmd_num 到此值就跳過這個 tick，防佇列堆積

# ★ 2026-08-19：J2/J3 位置鏡像改成【逐步推進目標角】，跟 J6 jog 同一套做法。
#
#   這是與 JOG_ACC=1100 同一類的 **mode 0 遺留用法**：
#     舊做法 = `on_prediction()` 直接把 target[1]/target[2] 一次設成最終角度，
#              等於一個指令丟出最大 90° 的大跳躍。那在 **mode 0** 是對的
#              （指令排隊、手臂照著走完），但 mode 6 的設計本意是**連續餵
#              小步設定點**（見 SDK 範例 2006-joint_online_trajectory_planning），
#              一次丟一個遠端點再放著不管，是拿 mode 0 的用法在跑 mode 6。
#
#   實測後果（17-11-49 真實速度重播，543s）：J3 走完 90° 在 40 °/s 下要 2.25 s，
#   但 elbow/shoulder 的 flex 脈衝長度中位只有 1.35–1.50 s ——**連正確的姿勢
#   都走不完**，手臂總在半路收到「回 rest」的新目標。mode 6 會中斷舊指令並
#   重新規劃，於是變成一次猛然反向：J3 有 47%、J2 有 56% 的目標變動屬於
#   「未走完就反向」。
#
#   新做法：`on_prediction()` 只更新 **goal**（要去的姿勢），`tick()` 每個
#   CMD_PERIOD 把 target 往 goal 推進 MIRROR_STEP_DEG。中途反向時只是斜率
#   反轉（目標角平滑地折返），不再是「猛然打斷一個遠端目標」。
#
#   ★ 附帶好處：手臂的實際速度由 MIRROR_RATE_DEG_S 決定，不再是 JOINT_SPEED。
#     JOINT_SPEED 退化成「跟上斜坡用的上限」，就像 JOG_SPEED_CAP 之於
#     JOG_RATE_DEG_S。所以碰撞動能由 60 °/s 決定，不是 80。
MIRROR_RATE_DEG_S = 60.0           # J2/J3 目標角推進速率。90° 行程 → 1.5s，
                                   # 落在實測脈衝長度中位（1.35–1.50s）附近
# 每個送指令週期能走多少——target 本身是按「實際經過時間 × 速率」連續推進的
# （見 ArmController.tick），這個值只用來檢查它不會被死區吃掉。
MIRROR_STEP_DEG = MIRROR_RATE_DEG_S * CMD_PERIOD      # 6.0°，須 > CMD_DEADZONE_DEG

JOINT_SPEED = 80.0                 # °/s，出廠上限 180（44%）。須 > MIRROR_RATE_DEG_S
                                   # 才追得上斜坡；實際速度由 MIRROR_RATE_DEG_S 決定
JOINT_ACC = 200.0                  # °/s²，出廠上限 1145（17%）。
                                   # ⚠️ **刻意保持低值**——JOG_ACC 1100 的教訓：
                                   # mode 6 下高加速度不會讓動作更順，只會變成
                                   # 每個指令週期一次的力矩衝擊，激振整條手臂。
                                   # 想讓 J2/J3 更快請調 MIRROR_RATE_DEG_S，不是這個。

# ★ 位置鏡像軸（elbow/shoulder）的最小維持時間（2026-08-19 加入，同日設為 0
#    停用——見下方值的說明）。
#
#   為什麼只有這兩軸需要：它們是**位置鏡像**，狀態一變就是 90° 的大擺動。
#   實測有 0.25–0.45 s 的短脈衝（模型短暫誤判成 flex 又跳回 rest），每一個都會
#   讓手臂往 90° 外衝一段再被拉回來 —— 這是抽動的成因之一。
#
#   ⚠️ 為什麼既有的三道閘都擋不住：
#     P_MIN      擋不住——這些脈衝的信心值是 0.78~1.00，本來就高
#     VOTE_M/N   擋不住——實測換成 6/4 的兩層投票，反向比例 77%/79%，
#                跟單層的 74%/79% 幾乎相同（脈衝夠穩定，投票會通過）
#     REFRACTORY 擋不住——它對「轉回 rest」是**刻意豁免**的（debounce.py 的
#                設計，當初為了修 J6「放手不停」）。那是 hand 的需求，
#                但套在會做 90° 擺動的 elbow/shoulder 上反而製造了這個問題。
#
#   所以另外加這一道，且**只作用在 elbow/shoulder**：新狀態要連續維持這麼久，
#   才允許更新 J2/J3 的目標角。hand（夾爪 toggle 與 J6 jog）完全不受影響——
#   那裡要的就是即時反應，「放手即停」已驗證通過，不能拿去換。
#
#   代價：elbow/shoulder 的反應延遲 +這個秒數。
#
#   實測（連同當時的 JOINT_SPEED=90 一起）：J3 未走完就反向 47%→12%、
#   J2 56%→15%，最短脈衝 0.25–0.30s→0.70–0.80s。當時 8/44 個脈衝短於 0.5s，
#   所以這道閘擋掉的是少數；現在它與上面的逐步推進是互補的——
#     這道閘  擋「根本不該動」的短脈衝（0.25~0.45s 的誤判）
#     逐步推進 讓「該動但中途反悔」的情況平滑折返，而不是猛然打斷
#   調整方向：仍會抽 → 往 0.7 調（別超過實測脈衝中位 ~1.35s 的一半）；
#             覺得手臂太遲鈍 → 0.3。設 0 = 停用（`_mirror_hold` 直接短路）。
MIRROR_MIN_HOLD_SEC = 0.5

# ---------------- 去彈跳三道閘（§5）----------------
P_MIN = {"hand": 0.80, "elbow": 0.75, "shoulder": 0.75}
VOTE_M = 6                          # 300 ms 視窗（6 幀 × STEP_SEC）
VOTE_N = 4
REFRACTORY = {"hand": 0.60, "elbow": 0.40, "shoulder": 0.40}

# 投票視窗的【時間】上限（2026-08-19 code review 加入）。
#
# 為什麼需要這個：DofDebouncer 的閘一會跳過低信心幀（不進緩衝區），所以
# 「最近 VOTE_M 幀」不等於「最近 VOTE_M × STEP_SEC 秒」。疲勞或姿勢過渡期
# 會產生連續的低信心幀，若不加時間上限，緩衝區裡的 VOTE_M 幀可能橫跨數秒
# —— 幾秒前的舊姿勢會在此刻湊滿多數決並觸發轉態（例：使用者已放鬆，夾爪
# 卻因為 3 秒前的舊 pinch 幀而閉合）。
#
# 取名目視窗的 2 倍：容許最多約一半的幀被信心門檻濾掉，仍能正常轉態。
# ⚠️ 代價：若 confidence 長期低迷（超過一半的幀 < P_MIN），該 DOF 會凍結在
#    最後的穩定狀態不再轉態。這是刻意的取捨 —— 寧可不動，也不要拿數秒前的
#    資料去動手臂。實機若常凍結，正確處置是【先降 P_MIN】而非放寬本值。
VOTE_WINDOW_SEC = (VOTE_M - 1) * STEP_SEC * 2.0      # 0.50 s

# ---------------- 看門狗（§5.4）----------------
WATCHDOG_SEC = 0.5

# ---------------- 主迴圈突發保護（2026-08-19 code review 加入）----------------
# 主迴圈約 100 Hz、模型 20 Hz → 正常情況每個 tick 只會拿到 0 或 1 筆 Prediction。
# 但主迴圈一旦被阻塞（engage() 的 wait=True 回 home、GC、磁碟 I/O），來源端會
# 累積大量待處理資料，下一次 poll 一口氣吐出數十筆 —— 而它們全部會帶著同一個
# now 灌進 DofDebouncer，讓「VOTE_M 幀 = VOTE_M × STEP_SEC 秒」的假設完全失效，
# 可能瞬間湊滿多數決並觸發轉態。超過此上限時只保留【最新的】幾筆
# （即時控制要的是現在，不是補完歷史）。
MAX_PREDS_PER_TICK = 4              # ≈ 落後 200 ms 就開始丟棄舊幀

# ---------------- 夾爪 ----------------
# 2026-08-18 實機：連續閉合 30 秒後用手摸**沒有明顯升溫**，30s 過度保守。
# 放寬到 60s——錄影時「夾住東西對鏡頭講解」很容易超過 30 秒，而斷電會直接
# 掉東西，是影片上最難看的失誤。⚠️ 60s 這個值同樣需要實測確認不會過熱。
GRIP_HOLD_MAX_SEC = 60.0

# ---------------- log（§7，CSV 欄位）----------------
LOG_FIELDS = ("t", "hand", "elbow", "shoulder",
              "p_hand", "p_elbow", "p_shoulder",
              "J2", "J3", "J6", "grip_state", "cmd_sent")


# ---------------- 載入時自我檢查（2026-08-19 code review 加入）----------------
# 比照 semg/v2/config.py 的 validate() 慣例：把參數之間的矛盾擋在實機之前。
# 這些矛盾的共同特徵是「症狀與原因看起來毫無關聯」，實機除錯代價極高。
#
# ⚠️ 刻意【不】在檔尾自動呼叫（跟 semg.v2.config.validate() 的自動呼叫慣例
#    不同）。原因：本檔案的 (d0) 檢查需要 import semg.v2.config 來比對
#    STEP_SEC，但 semg_realtime_V6/src 是由 run_arm_control.py／
#    run_manual_control.py 在 import 完 arm_config 之後才插進 sys.path
#    最前面（見檔頭「分類器來源套件」那段說明）。若在這裡自動呼叫，
#    import 順序會讓它撿到 src/semg/ 的舊快照而不是實際在用的
#    semg_realtime_V6，檢查結果沒有意義。所以呼叫端必須在把
#    SEMG_REALTIME_SRC 插進 sys.path 之後，自己呼叫一次 `arm_config.validate()`
#    （兩個 CLI 進入點都已經這樣做，見 run_arm_control.py／run_manual_control.py）。
def validate() -> None:
    # (a) 軟限位必須落在出廠限位內，否則 clamp_angles() 的第二道防線形同虛設
    for j, (lo, hi) in SOFT_LIMITS.items():
        jlo, jhi = JOINT_LIMITS[j]
        assert lo < hi, f"SOFT_LIMITS[J{j}] 下限 {lo} 不小於上限 {hi}"
        assert jlo <= lo and hi <= jhi, (
            f"SOFT_LIMITS[J{j}]=({lo},{hi}) 超出 JOINT_LIMITS=({jlo},{jhi})")

    # (b) home 姿態必須在軟限位內，否則 clamp 會靜默改掉你設的 home
    for j, (lo, hi) in SOFT_LIMITS.items():
        h = HOME_ANGLES[j - 1]
        assert lo <= h <= hi, (
            f"HOME_ANGLES[J{j}]={h} 不在 SOFT_LIMITS=({lo},{hi}) 內，"
            "clamp_angles() 會把它夾走，你會以為「改了沒生效」")

    # (c) 位置鏡像的目標角必須在軟限位內
    assert SOFT_LIMITS[3][0] <= ELBOW_FLEX_ANGLE <= SOFT_LIMITS[3][1], (
        f"ELBOW_FLEX_ANGLE={ELBOW_FLEX_ANGLE} 不在 J3 軟限位 {SOFT_LIMITS[3]} 內")
    assert SOFT_LIMITS[2][0] <= SHOULDER_FLEX_ANGLE <= SOFT_LIMITS[2][1], (
        f"SHOULDER_FLEX_ANGLE={SHOULDER_FLEX_ANGLE} 不在 J2 軟限位 {SOFT_LIMITS[2]} 內")

    # (d0) 與上游訊號層的時序一致性。控制層刻意保留自己的 STEP_SEC 副本
    # （參數分離），但值必須相同——所有以「幀」為單位的去彈跳參數都靠它
    # 換算成秒。在函式內 import 以避免模組層級的循環引用風險，也因為呼叫端
    # 必須保證這裡 import 到的是 semg_realtime_V6（見上方檔案層級的說明）。
    from semg.v2 import config as _C
    assert abs(STEP_SEC - _C.STEP_SEC) < 1e-9, (
        f"arm_config.STEP_SEC={STEP_SEC} 與 semg.v2.config.STEP_SEC={_C.STEP_SEC} "
        f"不一致。控制層的 VOTE_M / VOTE_WINDOW_SEC / REFRACTORY / WATCHDOG_SEC "
        f"全部建立在這個值上，不一致會讓去彈跳的時間語意錯誤。請改為相同值。")

    # (d) 去彈跳參數自洽
    assert 0 < VOTE_N <= VOTE_M, f"VOTE_N={VOTE_N} 必須落在 1..VOTE_M={VOTE_M}"
    assert VOTE_WINDOW_SEC >= (VOTE_M - 1) * STEP_SEC, (
        f"VOTE_WINDOW_SEC={VOTE_WINDOW_SEC} 小於名目視窗 "
        f"{(VOTE_M - 1) * STEP_SEC}，連正常速率的幀都會被時效檢查擋掉")
    for dof, p in P_MIN.items():
        assert 0.0 < p < 1.0, f"P_MIN[{dof}]={p} 必須在 (0,1) 之間"

    # (e) ★ jog 步階必須大於死區，否則 J6 永遠不會動且症狀極具誤導性
    assert JOG_STEP_DEG > CMD_DEADZONE_DEG, (
        f"JOG_STEP_DEG={JOG_STEP_DEG:.2f}° ≤ CMD_DEADZONE_DEG={CMD_DEADZONE_DEG}°："
        f"每個 tick 的 J6 變化量都會被死區擋掉，J6 完全不會動，"
        f"而終端機上 target 仍在累加（cmd_sent 恆為 0）。"
        f"請提高 JOG_RATE_DEG_S（目前 {JOG_RATE_DEG_S}，"
        f"下限約 {CMD_DEADZONE_DEG / CMD_PERIOD:.1f} °/s）或降低 CMD_DEADZONE_DEG。")

    # (e2) 位置鏡像的推進步距也必須大於死區，理由與 (e) 相同——步距被死區
    #      吃掉時 J2/J3 的 target 會一直累加但永遠送不出去，症狀是「手臂完全
    #      不動但畫面上目標角在變」，與 IP 打錯／手臂在 STOP 難以區分。
    assert MIRROR_STEP_DEG > CMD_DEADZONE_DEG, (
        f"MIRROR_STEP_DEG={MIRROR_STEP_DEG:.2f}° ≤ CMD_DEADZONE_DEG="
        f"{CMD_DEADZONE_DEG}°：J2/J3 的每步變化量會被死區擋掉，位置鏡像完全"
        f"失效。請提高 MIRROR_RATE_DEG_S（目前 {MIRROR_RATE_DEG_S}，下限約 "
        f"{CMD_DEADZONE_DEG / CMD_PERIOD:.1f} °/s）或降低 CMD_DEADZONE_DEG。")

    # (e3) 手臂的速度上限要追得上目標角的推進速率，否則 target 會持續領先
    #      實際位置，放開姿勢後手臂還在追之前的斜坡（與 (g) 對 jog 的檢查同理）。
    assert JOINT_SPEED > MIRROR_RATE_DEG_S, (
        f"JOINT_SPEED={JOINT_SPEED} 必須大於 MIRROR_RATE_DEG_S="
        f"{MIRROR_RATE_DEG_S}，否則手臂追不上 J2/J3 目標角的推進速率")

    # (f) 節流與看門狗合理性
    assert CMD_PERIOD > 0 and CMDNUM_MAX >= 1
    assert WATCHDOG_SEC > STEP_SEC * 2, (
        f"WATCHDOG_SEC={WATCHDOG_SEC} 太接近單幀週期，正常抖動就會誤觸發 SAFE")
    assert MAX_PREDS_PER_TICK >= 1

    # (g) jog 專用參數自洽（2026-08-19 加入，見上方「卡卡的」根因分析）
    assert JOG_BLEND_RADIUS_DEG < JOG_STEP_DEG, (
        f"JOG_BLEND_RADIUS_DEG={JOG_BLEND_RADIUS_DEG}° 必須小於 JOG_STEP_DEG="
        f"{JOG_STEP_DEG:.2f}°——xArm SDK 規定轉角融合半徑不能大於路徑長度，"
        f"設太大 set_servo_angle 的 radius 參數會被拒絕")
    assert JOG_SPEED_CAP > JOG_RATE_DEG_S, (
        f"JOG_SPEED_CAP={JOG_SPEED_CAP} 必須大於 JOG_RATE_DEG_S={JOG_RATE_DEG_S}，"
        f"否則手臂的實際速度上限追不上目標角的推進速率，J6 會持續落後")

    # (h) 所有速度/加速度參數必須在出廠上限內（2026-08-17，讀出實機上限後加入）。
    #     原本只寫在註解裡（「出廠上限 180」），沒有斷言——JOG_ACC=800 這種
    #     刻意激進的值一旦有人再往上調，SDK 只會靜默夾掉或拒絕單筆指令，
    #     症狀是「手臂動得比設定慢」，極難聯想到參數超限。
    for name, v, cap in (
        ("JOINT_SPEED", JOINT_SPEED, FACTORY_JOINT_SPEED_MAX),
        ("JOG_SPEED_CAP", JOG_SPEED_CAP, FACTORY_JOINT_SPEED_MAX),
        ("HOME_SPEED", HOME_SPEED, FACTORY_JOINT_SPEED_MAX),
        ("JOINT_ACC", JOINT_ACC, FACTORY_JOINT_ACC_MAX),
        ("JOG_ACC", JOG_ACC, FACTORY_JOINT_ACC_MAX),
    ):
        assert 0.0 < v <= cap, f"{name}={v} 超出出廠上限 {cap}"

    # (i) jog 速率的【真正】下限，由 (e) 與 (g) 兩條約束共同決定：
    #       (e) JOG_STEP_DEG > CMD_DEADZONE_DEG      → JOG_RATE >  5 °/s
    #       (g) JOG_STEP_DEG > JOG_BLEND_RADIUS_DEG  → JOG_RATE > 20 °/s  ← 較緊
    #     舊版 ON_MACHINE_CHECKLIST 寫「不要低於 8 °/s」是在 JOG_BLEND_RADIUS_DEG
    #     加入之前算的，已經失效。這裡把有效區間直接算出來寫進錯誤訊息，
    #     免得實機調參時拿到一個指向「轉角融合半徑」的訊息卻在想「jog 太快」。
    #     ⚠️ 分母是 JOG_PERIOD 不是 CMD_PERIOD——2026-08-18 把 jog 推進週期
    #        獨立出來之後，JOG_STEP_DEG 是由 JOG_PERIOD 算的，用錯分母會讓
    #        這條檢查的區間跟實際步距對不上。
    _jog_floor = max(CMD_DEADZONE_DEG, JOG_BLEND_RADIUS_DEG) / JOG_PERIOD
    assert _jog_floor < JOG_RATE_DEG_S <= JOG_SPEED_CAP, (
        f"JOG_RATE_DEG_S={JOG_RATE_DEG_S} 必須落在 "
        f"({_jog_floor:.1f}, {JOG_SPEED_CAP:.1f}] °/s。"
        f"下限由 max(CMD_DEADZONE_DEG={CMD_DEADZONE_DEG}, "
        f"JOG_BLEND_RADIUS_DEG={JOG_BLEND_RADIUS_DEG}) / JOG_PERIOD={JOG_PERIOD} 決定；"
        f"要更慢就得同時降 JOG_BLEND_RADIUS_DEG，但那會削弱它要解決的「卡卡的」問題。")

    # (j) jog 推進週期與送指令週期的關係（2026-08-18 分離後新增）。
    #     JOG_PERIOD 必須 ≥ CMD_PERIOD：目標角每 JOG_PERIOD 才變一次，而送指令
    #     的節流是 CMD_PERIOD——若 JOG_PERIOD 比 CMD_PERIOD 還小，等於要求送得
    #     比節流允許的還快，多出來的推進會被死區與節流吃掉，實際速率會低於
    #     JOG_RATE_DEG_S 而且無法從參數看出來。
    assert JOG_PERIOD >= CMD_PERIOD, (
        f"JOG_PERIOD={JOG_PERIOD} 不可小於 CMD_PERIOD={CMD_PERIOD}——"
        f"目標角推進得比指令送得出去還快，實際 jog 速率會低於 JOG_RATE_DEG_S。")
