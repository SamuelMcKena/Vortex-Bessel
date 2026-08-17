"""
Stage 8C notebook wiring tests.

Checks the SurfaceField energy-scaled cockpit notebook: existence, visible
controls, required caveat text, save-guard logic, synthetic-demo gating, and the
absence of any unguarded placeholder/donut demo beam in the production path.
"""

import json
from pathlib import Path

import pytest

NB_DIR = Path(__file__).parent.parent / "notebooks" / "digital_twin"
NB_PATH = NB_DIR / "02_surfacefield_energy_scaled_optical_cockpit.ipynb"

REQUIRED_CONTROLS = [
    "field_source_mode",
    "field_source_path",
    "require_real_field",
    "allow_synthetic_demo_field",
    "show_caveats",
    "save_outputs",
    "planning_mode",
    "pulse_duration_fs",
    "pulse_energy_before_optics_uJ",
]

REQUIRED_CAVEAT_PHRASES = [
    "scales real optical-field intensity arrays to pulse fluence",
    "does not model absorption",
    "not material",
]


def _load_notebook() -> dict:
    assert NB_PATH.is_file(), f"Notebook not found: {NB_PATH}"
    with open(NB_PATH, encoding="utf-8") as f:
        return json.load(f)


def _load_source() -> str:
    nb = _load_notebook()
    parts = []
    for cell in nb.get("cells", []):
        src = cell.get("source", [])
        parts.append("".join(src) if isinstance(src, list) else str(src))
    return "\n".join(parts)


# --- existence / validity ---


def test_notebook_exists():
    assert NB_PATH.is_file(), f"Notebook missing: {NB_PATH}"


def test_notebook_valid_json():
    nb = _load_notebook()
    assert "cells" in nb and "nbformat" in nb


def test_notebook_code_cells_compile():
    import ast
    nb = _load_notebook()
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            ast.parse(src)  # raises SyntaxError on failure


# --- visible controls ---


@pytest.mark.parametrize("control", REQUIRED_CONTROLS)
def test_notebook_contains_control(control):
    assert control in _load_source(), f"Missing top-level control: {control}"


def test_synthetic_demo_disabled_by_default():
    src = _load_source()
    assert "allow_synthetic_demo_field = False" in src, \
        "Synthetic demo field must be disabled by default."


def test_require_real_field_default_true():
    src = _load_source()
    assert "require_real_field = True" in src


def test_save_outputs_default_false():
    src = _load_source()
    assert "save_outputs = False" in src


# --- caveats ---


@pytest.mark.parametrize("phrase", REQUIRED_CAVEAT_PHRASES)
def test_notebook_contains_caveat_phrase(phrase):
    assert phrase.lower() in _load_source().lower(), f"Caveat phrase missing: {phrase!r}"


# --- save guard ---


def test_notebook_save_guard_blocks_without_caveats():
    src = _load_source()
    # Must check save_outputs together with show_caveats and raise.
    assert "save_outputs" in src and "show_caveats" in src
    assert "raise" in src
    # The exact guard condition should be present.
    assert "save_outputs and not show_caveats" in src


# --- no fabricated beam in production path ---


def test_notebook_has_no_donut_or_placeholder_demo():
    src = _load_source().lower()
    assert "donut" not in src, "Notebook must not draw a placeholder donut beam."
    assert "synthetic_placeholder" not in src, \
        "Notebook must not use the reserved synthetic_placeholder source status."


def test_notebook_demo_field_is_guarded():
    """Any demo-field construction must sit behind allow_synthetic_demo_field."""
    src = _load_source()
    assert "if allow_synthetic_demo_field" in src or "elif allow_synthetic_demo_field" in src, \
        "Demo field construction must be guarded by allow_synthetic_demo_field."
    # The demo, if present, must be labelled demo-only.
    if "demo" in src.lower():
        assert "unit_test_or_demo_only" in src


def test_notebook_missing_field_raises():
    src = _load_source()
    assert "MissingOpticalFieldError" in src


# --- real bridge references ---


def test_notebook_uses_real_field_extractors():
    src = _load_source()
    assert "extract_plane_from_surfacefield" in src
    assert "extract_stack_from_surfacefield" in src


def test_notebook_uses_energy_ledger_bridge():
    src = _load_source()
    assert "compute_energy_ledger" in src
    assert "energy_at_sample_uJ" in src


def test_notebook_scales_to_fluence():
    src = _load_source()
    assert "scale_plane_to_fluence" in src or "scale_stack_to_fluence" in src
    assert "fluence" in src.lower()
