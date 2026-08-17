"""Shared helpers for the vortex-Bessel-beam publication study.

I keep this package separate from the physics engine so that the core model can
stay importable by older notebooks while the publication workflow gains a clean
place for style, validation, metrics, and vector-analysis utilities.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    """Import helper submodules on first use.

    This keeps lightweight runner commands such as ``--list`` from importing
    the full scientific stack before notebooks actually need it.
    """

    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
