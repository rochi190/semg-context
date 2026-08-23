"""環境自檢：不接板子也能跑。確認套件、模型、電極對照。

    python check_install.py

★ 自己把 src/ 加進 sys.path，所以**不需要設 PYTHONPATH** ——
  Windows 的 cmd 沒有 `PYTHONPATH=src python ...` 這種寫法。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# Windows 主控台預設 cp950，印不出 ✓ ⚠ 這類字元會直接 UnicodeEncodeError。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ok = True
print("套件：")
for m in ("numpy", "scipy", "pandas", "torch"):
    try:
        __import__(m)
        print(f"  [OK ] {m}")
    except ImportError:
        print(f"  [缺 ] {m} 沒裝")
        ok = False
try:
    import serial          # noqa: F401
    print("  [OK ] pyserial")
except ImportError:
    print("  [!  ] pyserial 沒裝 —— 只有接板子時需要")

import torch                                 # noqa: E402
from semg.v2 import config as C              # noqa: E402
from semg.v2.infer import Recognizer         # noqa: E402

print("\n可用的模型（換檔名就換架構，不用改設定）：")
for p in sorted((ROOT / "models").glob("*.pt")):
    if p.stem.endswith("_scripted"):
        continue
    r = Recognizer(p)
    n = sum(v.numel() for v in torch.load(p, map_location="cpu",
                                          weights_only=False)["state_dict"].values())
    print(f"  {p.stem:<24} 架構={r.arch:<5} 參數={n:>8,}  "
          f"Hampel={'開' if r.use_hampel else '關'}")

lb = Recognizer(ROOT / "models/final_v4_split.pt").latency_budget
print(f"\n演算法延遲 {lb['total_s']:.2f}s"
      f"（視窗 {lb['window_s']:.2f} + 序列 {lb['seq_context_s']:.2f}"
      f" + 投票 {lb['voting_s']:.2f}）")

print("\n★ 電極對照 —— 貼片必須與此一致，否則模型全錯")
for i, (n_, zh) in enumerate(zip(C.CH_NAMES, C.CH_LABELS_ZH), 1):
    print(f"  ch{i}  {zh:<8s} {n_}")
print("\n  ⚠️ ch2 是二頭肌、ch4 是三角肌前束。")
print("     V3moving/README.md 寫的是相反的（舊版），**以這裡為準**。")

print("\n" + ("[OK ] 可以接板子了" if ok else "[缺 ] 先補齊套件"))
