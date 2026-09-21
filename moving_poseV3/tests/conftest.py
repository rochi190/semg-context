"""讓 `pytest tests/` 不需要任何額外設定就能跑。

⚠️ 路徑順序要跟正式執行（run_arm_control.py / run_manual_control.py）一致：
   `semg_realtime_V6/src` 必須排在 `src` **前面**。

   為什麼重要：測試裡的 `from semg.v2 import config as C` 會拿到先找到的那一份。
   舊的 moving_pose/ 佈局底下還留著 `src/semg/`（v3 時期的舊快照），conftest
   當時只加了 `src`，所以測試其實是對著**舊快照**驗證的——只是因為 DOF_STATES
   兩邊剛好相同才沒暴露出來。moving_poseV2 已經不放那份舊快照，這裡也把順序
   寫成與正式路徑相同，讓測試與實際執行看到同一份 config。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))                        # control package
sys.path.insert(0, str(ROOT / "semg_realtime_V6" / "src"))   # 蓋過去，優先
