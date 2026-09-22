from __future__ import annotations
import importlib
import platform
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "lab_gui"))
sys.path.insert(0, str(root))

print("Python:", sys.executable)
print("Version:", sys.version.replace("\n", " "))
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
input("\nPress Enter to close...")
