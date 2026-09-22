"""Compatibility launcher; prefer python run_lab_gui.py from repository root."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'run_lab_gui.py'), run_name='__main__')
