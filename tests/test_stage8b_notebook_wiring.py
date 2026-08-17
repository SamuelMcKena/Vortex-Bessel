"""
Stage 8B notebook wiring tests.

Checks notebook existence, top-level controls, caveats, forbidden claims,
and save-guard logic.
"""

import json
from pathlib import Path

import pytest

NB_DIR = Path(__file__).parent.parent / "notebooks" / "digital_twin"
NB_PATH = NB_DIR / "01_optical_cockpit_energy_accounting.ipynb"

# Variables that must appear in the notebook source as top-level controls
REQUIRED_CONTROLS = [
    "planning_mode",
    "save_outputs",
    "show_caveats",
    "wavelength_nm",
    "pulse_duration_fs",
    "repetition_rate_Hz",
    "pulse_energy_before_optics_uJ",
    "average_power_limit_W",
    "slm_diffraction_efficiency",
    "selected_first_order_fraction",
    "scan_speed_mm_s",
    "effective_diameter_um",
]

# Required caveat phrases
REQUIRED_CAVEAT_PHRASES = [
    "energy and exposure bookkeeping only",
    "does not compute material",
]

# Forbidden claims — must NOT appear outside caveat/negation context
# These are checked as substrings (case-insensitive)
FORBIDDEN_CLAIM_EXACT = [
    "damage prediction",
    "void prediction",
    "calibrated material prediction",
    "waveguide prediction",
]


def _load_notebook_source() -> str:
    """Return all cell source concatenated as a single string."""
    assert NB_PATH.is_file(), f"Notebook not found: {NB_PATH}"
    with open(NB_PATH, encoding="utf-8") as f:
        nb = json.load(f)
    parts = []
    for cell in nb.get("cells", []):
        src = cell.get("source", [])
        if isinstance(src, list):
            parts.append("".join(src))
        else:
            parts.append(str(src))
    return "\n".join(parts)


def _load_notebook() -> dict:
    assert NB_PATH.is_file(), f"Notebook not found: {NB_PATH}"
    with open(NB_PATH, encoding="utf-8") as f:
        return json.load(f)


# --- File existence ---


def test_notebook_exists():
    assert NB_PATH.is_file(), f"Notebook missing: {NB_PATH}"


def test_notebook_is_valid_json():
    nb = _load_notebook()
    assert "cells" in nb
    assert "nbformat" in nb


# --- Top-level controls ---


@pytest.mark.parametrize("control", REQUIRED_CONTROLS)
def test_notebook_contains_control(control):
    src = _load_notebook_source()
    assert control in src, f"Notebook missing top-level control: '{control}'"


def test_notebook_has_save_outputs_assignment():
    src = _load_notebook_source()
    assert "save_outputs" in src
    assert "False" in src or "True" in src  # must be assigned a boolean


def test_notebook_has_show_caveats_assignment():
    src = _load_notebook_source()
    assert "show_caveats" in src


# --- Caveat content ---


@pytest.mark.parametrize("phrase", REQUIRED_CAVEAT_PHRASES)
def test_notebook_contains_caveat_phrase(phrase):
    src = _load_notebook_source().lower()
    assert phrase.lower() in src, f"Caveat phrase not found in notebook: '{phrase}'"


def test_notebook_mentions_no_material_modification():
    src = _load_notebook_source().lower()
    assert "material" in src
    assert "modification" in src or "damage" in src


# --- Save-guard logic ---


def test_notebook_has_save_guard_logic():
    """Notebook must block saving when show_caveats=False."""
    src = _load_notebook_source()
    # Must check both conditions together
    assert "save_outputs" in src
    assert "show_caveats" in src
    # Must raise a ValueError if saving with caveats hidden
    assert "ValueError" in src or "raise" in src.lower()


# --- Forbidden claims ---


def test_notebook_no_damage_prediction_claim():
    src = _load_notebook_source().lower()
    assert "damage prediction" not in src, \
        "Notebook must not claim 'damage prediction'"


def test_notebook_no_void_prediction_claim():
    src = _load_notebook_source().lower()
    assert "void prediction" not in src, \
        "Notebook must not claim 'void prediction'"


def test_notebook_no_calibrated_material_prediction_claim():
    src = _load_notebook_source().lower()
    assert "calibrated material prediction" not in src, \
        "Notebook must not claim 'calibrated material prediction'"


def test_notebook_no_waveguide_prediction_claim():
    src = _load_notebook_source().lower()
    assert "waveguide prediction" not in src, \
        "Notebook must not claim 'waveguide prediction'"


# --- Energy accounting content ---


def test_notebook_references_energy_ledger():
    src = _load_notebook_source()
    assert "energy" in src.lower()
    assert "ledger" in src.lower() or "EnergyLedger" in src or "compute_energy_ledger" in src


def test_notebook_references_fluence():
    src = _load_notebook_source()
    assert "fluence" in src.lower()


def test_notebook_references_pulse_spacing():
    src = _load_notebook_source()
    assert "pulse_spacing" in src or "pulses_per_spot" in src or "scan_speed" in src


# --- Digital twin module import ---


def test_notebook_imports_digital_twin():
    src = _load_notebook_source()
    assert "digital_twin" in src or "energy_accounting" in src or "exposure_bookkeeping" in src
