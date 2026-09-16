"""Launch the unified experiment-control interface with Virtual Lab support."""

from pathlib import Path
import sys

# Running this file directly places only lab_gui/ on sys.path.  Virtual Lab also
# reuses the repository's vbb_study digital-twin modules, so add the repository
# root explicitly rather than relying on the caller's working directory.
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for path in (REPO_ROOT, HERE):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

# The experiment-builder window is a strict extension of the existing unified
# cockpit: Home/live/recorded behaviour stays intact, while Virtual Lab gains
# explicit multi-fault perturbations, measurement budgets and auto-convergence.
from labcontrol.ui.virtual_advanced_v2 import main


if __name__ == "__main__":
    raise SystemExit(main())
