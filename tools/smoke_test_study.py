"""Smoke test for the structured-beam study workspace.

Checks imports, path contract, canonical docs, notebooks, and source files
without executing any notebook.  Exit code 0 = all clear; 1 = failures found.

Run from the repo root or from inside Publication_Study:

    python Publication_Study/tools/smoke_test_study.py

or:

    python tools/smoke_test_study.py   (from inside Publication_Study/)
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap — locate the study before any study imports
# ---------------------------------------------------------------------------

_here = Path(__file__).resolve().parent          # tools/
_pub = _here.parent                              # Publication_Study/
_root = _pub.parent                              # repo root

for _p in (_root, _pub):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_passed: list[str] = []
_failed: list[str] = []


def _ok(label: str) -> None:
    _passed.append(label)
    print(f"  PASS  {label}")


def _fail(label: str, detail: str = "") -> None:
    _failed.append(label)
    msg = f"  FAIL  {label}"
    if detail:
        msg += f"\n        {detail}"
    print(msg)


# ---------------------------------------------------------------------------
# 1. Core imports
# ---------------------------------------------------------------------------

print("\n[smoke] 1. Core imports")

for _mod in [
    "bessel_twin_core",
    "vbb_study",
    "vbb_study.setup_study",
    "vbb_study.study_taxonomy",
    "vbb_study.equations",
    "vbb_study.equations.scalar_bessel",
    "vbb_study.equations.propagation",
    "vbb_study.equations.holography",
    "vbb_study.equations.vector_jones",
    "vbb_study.equations.materials",
    "vbb_study.equations.objective_pupil",
    "vbb_study.equations.interface",
    "vbb_study.equations.polygonal",
]:
    try:
        importlib.import_module(_mod)
        _ok(_mod)
    except Exception as exc:
        _fail(_mod, str(exc))

# ---------------------------------------------------------------------------
# 2. Path contract and workspace validation
# ---------------------------------------------------------------------------

print("\n[smoke] 2. Path contract")

try:
    from vbb_study import setup_study

    paths = setup_study.bootstrap(_pub, apply_plot_style=False)
    _ok("setup_study.bootstrap()")
except Exception as exc:
    _fail("setup_study.bootstrap()", str(exc))
    paths = {}

if paths:
    try:
        report = setup_study.validate_workspace(paths, strict=False)
        for category, missing in report.items():
            if missing:
                for item in missing:
                    _fail(f"workspace/{category}", item)
            else:
                _ok(f"workspace/{category} — all present")
    except Exception as exc:
        _fail("setup_study.validate_workspace()", str(exc))

# ---------------------------------------------------------------------------
# 3. Required docs
# ---------------------------------------------------------------------------

print("\n[smoke] 3. Required docs")

if paths:
    docs_dir = Path(paths["docs"])
    for name in setup_study.REQUIRED_DOCS:
        p = docs_dir / name
        if p.exists():
            _ok(f"docs/{name}")
        else:
            _fail(f"docs/{name}", f"not found: {p}")

# ---------------------------------------------------------------------------
# 4. Required notebooks
# ---------------------------------------------------------------------------

print("\n[smoke] 4. Required notebooks")

if paths:
    pub = Path(paths["publication"])
    for name in setup_study.REQUIRED_NOTEBOOKS:
        p = pub / name
        if p.exists():
            _ok(name)
        else:
            _fail(name, f"not found: {p}")

# ---------------------------------------------------------------------------
# 5. Source files
# ---------------------------------------------------------------------------

print("\n[smoke] 5. Required source files")

if paths:
    root = Path(paths["root"])
    for name in setup_study.REQUIRED_SOURCE_FILES:
        p = root / name
        if p.exists():
            _ok(name)
        else:
            _fail(name, f"not found: {p}")

# ---------------------------------------------------------------------------
# 6. Runner sanity
# ---------------------------------------------------------------------------

print("\n[smoke] 6. Runner sanity")

try:
    from run_study import ORDERED_NOTEBOOKS, STAGE_NOTEBOOKS, selected_notebooks

    all_nbs = selected_notebooks()
    if len(all_nbs) == len(ORDERED_NOTEBOOKS):
        _ok(f"run_study.selected_notebooks() — {len(all_nbs)} notebooks")
    else:
        _fail("run_study.selected_notebooks()", f"expected {len(ORDERED_NOTEBOOKS)}, got {len(all_nbs)}")

    for stage in list(STAGE_NOTEBOOKS.keys()) + ["all"]:
        try:
            nbs = selected_notebooks(stage=stage)
            _ok(f"run_study.selected_notebooks(stage={stage!r}) — {len(nbs)} notebooks")
        except Exception as exc:
            _fail(f"run_study.selected_notebooks(stage={stage!r})", str(exc))
except Exception as exc:
    _fail("run_study import", str(exc))

# ---------------------------------------------------------------------------
# 7. Quick physics sanity (no propagation)
# ---------------------------------------------------------------------------

print("\n[smoke] 7. Quick physics sanity")

try:
    import bessel_twin_core as bt

    cfg = bt.default_config("fast")
    d = bt.compute_design_from_targets(cfg.laser, cfg.target, cfg.material)
    _ok(f"bessel_twin_core: design computed — target_scale={d.target_scale_definition}")
    if hasattr(d, "vortex_main_ring_diameter_m"):
        _ok(f"bessel_twin_core: vortex_main_ring_diameter_m = {d.vortex_main_ring_diameter_m / bt.um:.2f} um")
    else:
        _fail("bessel_twin_core: vortex_main_ring_diameter_m attribute missing")
except Exception as exc:
    _fail("bessel_twin_core quick design", str(exc))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n[smoke] -- Summary --------------------------------------------------")
print(f"[smoke]   passed: {len(_passed)}")
print(f"[smoke]   failed: {len(_failed)}")

if _failed:
    print("[smoke]   FAILED items:")
    for f in _failed:
        print(f"    • {f}")
    print("[smoke] -- SMOKE TEST FAILED --")
    sys.exit(1)
else:
    print("[smoke] -- ALL CHECKS PASSED --")
    sys.exit(0)
