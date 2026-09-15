"""Build the one-folder Anaconda/Spyder lab-PC distribution."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


LAUNCHER = '''"""Run Vortex-Bessel Lab Control v2.1 from Spyder (F5)."""
from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "lab_gui"))
sys.path.insert(0, str(ROOT))

from labcontrol.ui.advanced import main

if __name__ == "__main__":
    raise SystemExit(main())
'''

DIAGNOSTICS = '''from __future__ import annotations
import importlib
import platform
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "lab_gui"))
sys.path.insert(0, str(root))

print("Python:", sys.executable)
print("Version:", sys.version.replace("\\n", " "))
print("OS:", platform.platform())
for module in ("numpy", "PIL", "PySide6", "win32file", "HEDS"):
    try:
        loaded = importlib.import_module(module)
        print(f"{module}: OK", getattr(loaded, "__version__", ""))
    except Exception as exc:
        required = module in {"numpy", "PIL", "PySide6"}
        print(f"{module}:", "MISSING (required)" if required else "not visible / optional", "-", exc)
try:
    from labcontrol.ui.advanced import AdvancedLabWindow
    print("Lab GUI import: OK")
except Exception as exc:
    print("Lab GUI import: FAILED -", exc)
input("\\nPress Enter to close...")
'''

README = """VORTEX-BESSEL LAB GUI v2.1 — LAB PC

Keep this folder separate from your old working GUI.

Recommended (Spyder / Anaconda)
1. Open SPYDER_RUN_LAB_CONTROL_V2_1.py in Spyder.
2. Press F5.

Or double-click START_LAB_CONTROL_V2_1.bat.

First-time checks:
- RUN_DIAGNOSTICS.bat
- INSTALL_OR_REPAIR_DEPENDENCIES.bat if NumPy, Pillow, PySide6 or pywin32 is missing.
- Start PC-Beamage, connect the Beamage 4M, and enable Pipeline.
- TEST_BEAMAGE_PIPE.bat checks the official named-pipe connection.

Camera viewing:
- Beamage frames stay at full acquired pixel resolution.
- Fit beam / Auto-fit beam changes only the view; it never crops stored data.
- Full frame restores the whole sensor view.
- Mouse wheel or +/- zooms; drag pans; double-click fits the beam.
- Inferno, Gentec-like, Turbo, Viridis and grayscale are display-only maps.
- Exposure is read back in this GUI. The supplied official Pipeline commands do
  not include exposure writing, so use the PC-Beamage controls button to bring
  the manufacturer window forward and change it there.

The PC-Beamage BMP remains LIVE PREVIEW ONLY until validated against a numeric
camera export. Formal quantitative capture remains disabled for this route.
"""


def write_text(path: Path, text: str) -> None:
    path.write_text(text.replace("\n", "\r\n") if path.suffix.lower() in {".bat", ".txt"} else text, encoding="utf-8")


def build(repo: Path, destination: Path) -> Path:
    root = destination / "Vortex_Bessel_Lab_GUI_v2_1"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "outputs", "lab_sessions")
    shutil.copytree(repo / "lab_gui", root / "lab_gui", ignore=ignore)
    shutil.copytree(repo / "vbb_study", root / "vbb_study", ignore=ignore)
    (root / "tools").mkdir()
    shutil.copy2(repo / "tools" / "test_beamage_pipe.py", root / "tools" / "test_beamage_pipe.py")
    shutil.copy2(repo / "requirements-lab.txt", root / "requirements-lab.txt")

    write_text(root / "SPYDER_RUN_LAB_CONTROL_V2_1.py", LAUNCHER)
    write_text(root / "LAB_PC_DIAGNOSTICS.py", DIAGNOSTICS)
    write_text(root / "README_FIRST.txt", README)
    write_text(root / "START_LAB_CONTROL_V2_1.bat", """@echo off
setlocal
cd /d "%~dp0"
python SPYDER_RUN_LAB_CONTROL_V2_1.py
if errorlevel 1 (
  echo.
  echo Lab Control exited with an error. Preserve lab_gui_crash.log.
  pause
)
""")
    write_text(root / "RUN_DIAGNOSTICS.bat", """@echo off
setlocal
cd /d "%~dp0"
python LAB_PC_DIAGNOSTICS.py
""")
    write_text(root / "INSTALL_OR_REPAIR_DEPENDENCIES.bat", """@echo off
setlocal
cd /d "%~dp0"
python -m pip install -r requirements-lab.txt
pause
""")
    write_text(root / "TEST_BEAMAGE_PIPE.bat", """@echo off
setlocal
cd /d "%~dp0"
python tools\test_beamage_pipe.py
pause
""")
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = build(repo, output)
    archive = shutil.make_archive(str(output / root.name), "zip", root.parent, root.name)
    print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
