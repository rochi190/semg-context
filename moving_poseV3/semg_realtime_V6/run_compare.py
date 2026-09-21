"""★ 兩個模型並排比較 —— 吃**同一條資料流**。

    python run_compare.py COM3                      預設 split vs tiny
    python run_compare.py COM3 --b final_v4_zhao    改比 zhao
    python run_compare.py COM3 --idle-test 60       靜止誤觸發（唯一有真值的指標）
    python run_compare.py --replay 某份錄音.csv      沒有板子時

★ 預設是 split vs tiny，因為那是「修好的 vs 原本的」——
  split 把各自由度的編碼器分開，解決「舉 elbow + index 時 hand 不觸發」。
  留一組合測試（3 種子）：tiny 5.0% ± 2.0%、split 85.4% ± 1.9%。

⚠️ **不要分兩次跑再比。** session 與貼片的變異遠大於模型差異：
   實測同一間教室、相隔 12 分鐘的兩份錄音差 62.7% vs 97.0%。
   分兩次比出來的是「哪次貼得比較好」。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

argv = sys.argv[1:]
a_name, b_name = "final_v6_split", "final_v5_split"
for flag, attr in (("--a", "a"), ("--b", "b")):
    if flag in argv:
        i = argv.index(flag)
        if attr == "a":
            a_name = argv[i + 1]
        else:
            b_name = argv[i + 1]
        del argv[i:i + 2]

port = None
if argv and not argv[0].startswith("-"):
    port = argv.pop(0)
if port is None and "--replay" not in argv:
    raise SystemExit(__doc__)

for n in (a_name, b_name):
    if not (ROOT / "models" / f"{n}.pt").exists():
        have = sorted(p.stem for p in (ROOT / "models").glob("*.pt")
                      if not p.stem.endswith("_scripted"))
        raise SystemExit(f"找不到模型 {n}\n可用的：{have}")

sys.argv = (["live",
             "--bundle", str(ROOT / "models" / f"{a_name}.pt"),
             "--bundle-b", str(ROOT / "models" / f"{b_name}.pt")]
            + (["--port", port, "--baud", "1600000"] if port else []) + argv)
from semg.v2.live import main      # noqa: E402
main()
