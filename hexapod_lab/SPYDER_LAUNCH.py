"""Spyder-friendly launcher for Hexapod + Laser Lab.

Open THIS file in Spyder and press Run.  The GUI itself is launched with the
private .venv next to the project, so Spyder's own Python environment does not
need PySide6/PyVista/CadQuery installed.

Run INSTALL_AND_START.bat once if .venv does not exist yet.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
APP = ROOT / "run_hexapod_lab.py"

if not VENV_PYTHON.is_file():
    raise SystemExit(
        "\nHexapod Lab private environment is not installed.\n"
        "Double-click INSTALL_AND_START.bat once, then run SPYDER_LAUNCH.py again.\n"
        f"Expected: {VENV_PYTHON}\n"
    )

print(f"Launching Hexapod Lab with: {VENV_PYTHON}")
subprocess.Popen(
    [str(VENV_PYTHON), str(APP)],
    cwd=str(ROOT),
)
print("Hexapod Lab launched in its private environment.")
