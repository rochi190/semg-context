"""列出序列埠，找出板子接在哪裡。

    python list_ports.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
try:
    from serial.tools import list_ports
except ImportError:
    raise SystemExit("pyserial 沒裝：pip install pyserial")

ports = list(list_ports.comports())
if not ports:
    print("找不到任何序列埠。檢查：USB 線、板子供電、驅動程式。")
else:
    print(f"找到 {len(ports)} 個序列埠：")
    for p in ports:
        print(f"  {p.device:<12} {p.description}")
    print("\nWindows 用 COM3 這種名稱，Linux 用 /dev/ttyUSB0。")
