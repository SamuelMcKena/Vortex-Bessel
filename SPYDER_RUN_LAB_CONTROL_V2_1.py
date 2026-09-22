"""Run Vortex-Bessel Lab Control v2.1 from Spyder (F5)."""
from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "lab_gui"))
sys.path.insert(0, str(ROOT))

from labcontrol.ui.virtual_advanced_v3 import main

if __name__ == "__main__":
    raise SystemExit(main())
