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

# This remains the same unified cockpit.  The v3 window adds explicit multi-fault
# perturbations, measurement budgets, auto-converging SLM-only recovery and
# Beamage/max-resolution virtual camera choices.
from labcontrol.ui.virtual_advanced_v3 import main


if __name__ == "__main__":
    raise SystemExit(main())
