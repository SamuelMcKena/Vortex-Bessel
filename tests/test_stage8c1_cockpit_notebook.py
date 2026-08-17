"""Stage 8C.1 integrated cockpit notebook wiring tests."""

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
NB_PATH = ROOT / "notebooks" / "digital_twin" / "00_full_beam_to_write_cockpit_MVP.ipynb"

CONTROL_GROUP_TERMS = [
    "wavelength_nm",
    "pre_slm_transmission",
    "telescope_enabled",
    "slm1_enabled",
    "generation_method",
    "first_order_filter_enabled",
    "relay_transmission",
    "objective_NA",
    "material_name",
    "selected_z_um",
    "scan_speed_mm_s",
    "enable_material_response",
]


def _load_notebook() -> dict:
    assert NB_PATH.is_file(), f"Notebook missing: {NB_PATH}"
    return json.loads(NB_PATH.read_text(encoding="utf-8"))


def _source() -> str:
    nb = _load_notebook()
    parts = []
    for cell in nb["cells"]:
        src = cell.get("source", [])
        parts.append("".join(src) if isinstance(src, list) else str(src))
    return "\n".join(parts)


def test_notebook_exists():
    assert NB_PATH.is_file()


def test_notebook_valid_json():
    nb = _load_notebook()
    assert nb["nbformat"] == 4
    assert "cells" in nb


def test_notebook_code_cells_compile():
    nb = _load_notebook()
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])))


def test_one_top_control_cell_contains_required_control_groups():
    nb = _load_notebook()
    first_code = next(cell for cell in nb["cells"] if cell.get("cell_type") == "code")
    src = "".join(first_code.get("source", []))
    assert "FULL BEAM-TO-WRITE COCKPIT MVP - USER CONTROLS" in src
    for term in CONTROL_GROUP_TERMS:
        assert term in src, f"missing control term {term}"


def test_synthetic_demo_disabled_by_default():
    assert "allow_synthetic_demo_field = False" in _source()


def test_save_outputs_and_show_caveats_exist():
    src = _source()
    assert "save_outputs = False" in src
    assert "show_caveats = True" in src
    assert "save_outputs and not show_caveats" in src


def test_caveat_text_exists():
    src = _source().lower()
    assert "optical/energy/exposure cockpit" in src
    assert "does not predict material modification" in src


def test_disabled_future_physics_panel_exists():
    src = _source().lower()
    assert "disabled future physics panels" in src
    assert "material response: disabled" in src
    assert "dose accumulation: stage 8e" in src


def test_notebook_includes_central_roi_and_target_depth_diagnostics():
    src = _source().lower()
    assert "central_roi_half_width_um" in src
    assert "central roi" in src
    assert "target-depth" in src or "target_depth" in src


def test_notebook_includes_lab_realism_hardware_feasibility_section():
    src = _source().lower()
    assert "lab realism / hardware feasibility" in src
    assert "build_lab_realism_report" in src

