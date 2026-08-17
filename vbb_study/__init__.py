"""Shared helpers for the Vortex-Bessel structured-beam study.

The package remains lazily imported so lightweight commands such as
``python run_study.py --list`` do not load the full numerical stack.  The
workspace path is the directory containing ``vbb_study``; this is correct both
for the standalone Vortex-Bessel repository and for older checkouts where the
workspace happened to be named ``Publication_Study``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
# ROOT is retained as a compatibility name used by a few historical helpers.
ROOT = WORKSPACE
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

_SUBMODULES = [
    "config",
    "design",
    "equations",
    "facade",
    "setup_study",
    "study_taxonomy",
    "vbb_axicon",
    "vbb_capsule",
    "vbb_discrete",
    "vbb_hex_outline",
    "vbb_hexagon_metrics",
    "vbb_hexagon_study",
    "vbb_materials",
    "vbb_materials_study",
    "vbb_metrics",
    "vbb_planes",
    "vbb_polygonal",
    "vbb_polarized_train",
    "publication",
    "vbb_regime",
    "vbb_sample_study",
    "vbb_studies",
    "vbb_style",
    "vbb_train_viz",
    "vbb_validation",
    "vbb_vector",
    "vbb_viz",
]

__all__ = list(_SUBMODULES)


def __getattr__(name: str):
    """Import helper submodules on first use."""

    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
