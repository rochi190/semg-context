"""即時分類（單一模型），可選擇同時驅動 UFACTORY Lite 6。

純螢幕（不接手臂）：
    python run_live.py COM10                       一般即時分類
    python run_live.py COM10 --idle-test 60        ★ 靜止誤觸發測試（接手臂前必做）
    python run_live.py COM10 --model final_v4_zhao     換 zhao 架構
    python run_live.py --replay 某份錄音.csv       沒有板子時用檔案模擬

接手臂（加 --arm-ip 或 --arm-dry-run 即可，其餘用法完全相同）：
    python run_live.py --replay 某份錄音.csv --arm-dry-run     讀檔版 + 假手臂
    python run_live.py --replay 某份錄音.csv --arm-ip 192.168.1.166   讀檔版 + 真手臂
    python run_live.py COM10 --arm-ip 192.168.1.166            肌電操控版（正式）

    校準完成後手臂是 DISARMED（不會動），按 Enter 才進 ARMED；q 結束、z 把 J6 歸零。
    接手臂時 --votes/--dwell 自動改用控制層的預設（1／6），見 src/control/arm_config.py。

★ 架構**不用指定** —— 記在模型檔的 `arch` 欄位裡，載入時自動建對應的網路。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

argv = sys.argv[1:]
model = "final_v4_split"
if "--model" in argv:
    i = argv.index("--model")
    model = argv[i + 1]
    del argv[i:i + 2]

# 第一個不以 - 開頭的參數當序列埠（--replay 模式不需要）
port = None
if argv and not argv[0].startswith("-"):
    port = argv.pop(0)
if port is None and "--replay" not in argv:
    raise SystemExit(__doc__)

bundle = ROOT / "models" / f"{model}.pt"
if not bundle.exists():
    have = sorted(p.stem for p in (ROOT / "models").glob("*.pt")
                  if not p.stem.endswith("_scripted"))
    raise SystemExit(f"找不到模型 {bundle}\n可用的：{have}")

sys.argv = ["live", "--bundle", str(bundle)] + (["--port", port, "--baud", "1600000"]
                                                if port else []) + argv
from semg.v2.live import main      # noqa: E402
main()
