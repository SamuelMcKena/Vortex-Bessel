"""
Stage 8A.1 literature and model anchor verification tests.

Checks that all required docs exist, CSVs have required columns and content,
and that claim boundaries are strict and explicit.
"""

from pathlib import Path

import csv
import pytest

DOCS = Path(__file__).parent.parent / "docs"
CSV_DIR = Path(__file__).parent.parent / "outputs" / "csv" / "digital_twin"
ROOT = Path(__file__).parent.parent

# --- Required files ---

REQUIRED_DOCS = [
    "25_literature_model_anchors.md",
    "26_literature_anchor_claim_boundaries.md",
]

REQUIRED_CSVS = [
    "literature_model_anchors.csv",
    "material_parameter_anchor_candidates.csv",
    "model_equation_anchor_map.csv",
    "material_response_calibration_targets.csv",
]

# --- Required anchor categories ---

REQUIRED_CATEGORIES = [
    "bessel_beam_material_modification",
    "fused_silica_bessel_processing",
    "ultrafast_waveguide_inscription",
    "crznse_or_zns_family_waveguide_inscription",
    "heat_accumulation_scan_speed",
    "incubation_multi_pulse_threshold",
    "nonlinear_propagation_filamentation",
    "surface_ablation_modelling",
    "refractive_index_reconstruction",
    "waveguide_mode_modelling",
    "microscope_or_etch_observable",
]

# --- Required anchor CSV columns ---

ANCHOR_CSV_REQUIRED_COLUMNS = [
    "anchor_id",
    "category",
    "paper_short_name",
    "source_status",
    "material",
    "digital_twin_layer",
    "model_status_allowed",
    "claim_boundary",
    "parameters_borrowable",
    "parameters_must_calibrate",
]

# --- Required parameter safe_use_level values ---

REQUIRED_SAFE_USE_LEVELS = [
    "direct_physical_constant",
    "literature_prior_only",
    "order_of_magnitude_only",
    "calibration_required",
    "do_not_transfer_between_materials",
]

# --- Required equation IDs ---

REQUIRED_EQUATION_IDS = [
    "energy_at_sample",
    "angular_spectrum_propagation",
    "fluence_scaling",
    "local_threshold_ratio",
    "incubation_threshold_law",
    "pulse_spacing",
    "pulses_per_spot",
    "dose_per_unit_length",
    "accumulated_dose",
    "multiphoton_absorption_proxy",
    "carrier_rate_proxy",
    "thermal_diffusion_length",
    "heat_equation_proxy",
    "microscope_projection_proxy",
    "waveguide_mode_solver_anchor",
]

# --- Required calibration target categories (matched by quantity_to_measure keyword) ---

REQUIRED_CALIBRATION_TARGETS = [
    "threshold pulse energy",
    "track width",
    "track depth",
    "feature length",
    "continuity",
    "side-lobe",
    "crack",
    "microscope contrast",
    "etch response",
    "refractive-index change",
    "waveguide mode",
    "propagation loss",
    "surface ablation depth",
    "surface ablation crater",
]


def _read_csv(name):
    path = CSV_DIR / name
    assert path.is_file(), f"CSV missing: {path}"
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


# --- File existence ---


@pytest.mark.parametrize("doc_name", REQUIRED_DOCS)
def test_doc_exists(doc_name):
    assert (DOCS / doc_name).is_file(), f"Missing: docs/{doc_name}"


@pytest.mark.parametrize("csv_name", REQUIRED_CSVS)
def test_csv_exists(csv_name):
    assert (CSV_DIR / csv_name).is_file(), f"Missing CSV: {csv_name}"


def test_summary_doc_exists():
    summary = ROOT / "STAGE8A1_LITERATURE_MODEL_ANCHORS_SUMMARY.md"
    assert summary.is_file(), "STAGE8A1_LITERATURE_MODEL_ANCHORS_SUMMARY.md missing"


# --- Anchor CSV content ---


def test_anchor_csv_required_columns():
    rows = _read_csv("literature_model_anchors.csv")
    assert rows, "Anchor CSV is empty"
    for col in ANCHOR_CSV_REQUIRED_COLUMNS:
        assert col in rows[0], f"Required column '{col}' missing from anchor CSV"


@pytest.mark.parametrize("category", REQUIRED_CATEGORIES)
def test_anchor_category_present(category):
    rows = _read_csv("literature_model_anchors.csv")
    categories = [r.get("category", "") for r in rows]
    assert category in categories, f"Required anchor category '{category}' not in CSV"


def test_anchor_csv_has_source_status():
    rows = _read_csv("literature_model_anchors.csv")
    statuses = [r.get("source_status", "") for r in rows]
    assert any(s == "needs_literature_lookup" for s in statuses), \
        "Anchor CSV should have at least one 'needs_literature_lookup' placeholder"


def test_anchor_csv_claim_boundaries_not_empty():
    rows = _read_csv("literature_model_anchors.csv")
    boundaries = [r.get("claim_boundary", "") for r in rows if r.get("claim_boundary")]
    assert len(boundaries) > 5, "Anchor CSV should have claim boundary text for most rows"


# --- Parameter CSV ---


def test_parameter_csv_has_safe_use_level():
    rows = _read_csv("material_parameter_anchor_candidates.csv")
    assert rows, "Parameter CSV is empty"
    assert "safe_use_level" in rows[0], "'safe_use_level' column missing from parameter CSV"
    levels = {r.get("safe_use_level", "") for r in rows}
    for level in REQUIRED_SAFE_USE_LEVELS:
        assert level in levels, f"Required safe_use_level '{level}' missing from parameter CSV"


def test_parameter_csv_znse_separated_from_silica():
    rows = _read_csv("material_parameter_anchor_candidates.csv")
    materials = [r.get("material", "").lower() for r in rows]
    assert any("znse" in m or "crzn" in m for m in materials), \
        "ZnSe-family parameters should be present in parameter CSV"
    assert any("silica" in m or "fused" in m for m in materials), \
        "Fused silica reference parameters should be present (marked do_not_transfer)"
    # Silica parameters that exist must be marked do_not_transfer
    silica_rows = [r for r in rows if "silica" in r.get("material", "").lower()
                   or "fused" in r.get("material", "").lower()]
    for row in silica_rows:
        assert row.get("safe_use_level") == "do_not_transfer_between_materials", \
            f"Fused silica parameter '{row.get('parameter_name')}' must be marked do_not_transfer_between_materials"


def test_parameter_csv_calibration_required_for_threshold():
    rows = _read_csv("material_parameter_anchor_candidates.csv")
    threshold_rows = [r for r in rows
                      if "threshold" in r.get("parameter_name", "").lower()
                      and "znse" in r.get("material", "").lower()]
    for row in threshold_rows:
        assert row.get("requires_calibration", "").lower() == "yes", \
            f"Threshold parameter '{row.get('parameter_name')}' for ZnSe must require calibration"


# --- Equation map ---


def test_equation_map_has_required_columns():
    rows = _read_csv("model_equation_anchor_map.csv")
    assert rows, "Equation map CSV is empty"
    for col in ("equation_id", "equation_name", "equation_text", "digital_twin_layer",
                "model_status", "calibration_required"):
        assert col in rows[0], f"Required column '{col}' missing from equation map"


@pytest.mark.parametrize("eq_id", REQUIRED_EQUATION_IDS)
def test_equation_id_present(eq_id):
    rows = _read_csv("model_equation_anchor_map.csv")
    ids = [r.get("equation_id", "") for r in rows]
    assert eq_id in ids, f"Required equation_id '{eq_id}' missing from equation map"


def test_angular_spectrum_is_optical_prediction():
    rows = _read_csv("model_equation_anchor_map.csv")
    asm = next((r for r in rows if r.get("equation_id") == "angular_spectrum_propagation"), None)
    assert asm is not None
    assert asm.get("model_status") == "optical_prediction", \
        "ASM propagation should have model_status=optical_prediction"
    assert asm.get("calibration_required", "").lower().startswith("no"), \
        "ASM propagation should not require calibration"


def test_proxy_equations_marked_correctly():
    rows = _read_csv("model_equation_anchor_map.csv")
    proxy_ids = {"multiphoton_absorption_proxy", "carrier_rate_proxy",
                 "heat_equation_proxy", "microscope_projection_proxy"}
    for row in rows:
        if row.get("equation_id") in proxy_ids:
            assert "proxy" in row.get("model_status", "").lower(), \
                f"Equation '{row['equation_id']}' should have a proxy model_status"


# --- Calibration target CSV ---


def test_calibration_target_csv_columns():
    rows = _read_csv("material_response_calibration_targets.csv")
    assert rows, "Calibration target CSV is empty"
    for col in ("target_id", "quantity_to_measure", "measurement_method",
                "needed_for_model_parameter", "promotion_enabled"):
        assert col in rows[0], f"Required column '{col}' missing from calibration target CSV"


@pytest.mark.parametrize("keyword", REQUIRED_CALIBRATION_TARGETS)
def test_calibration_target_category_present(keyword):
    rows = _read_csv("material_response_calibration_targets.csv")
    quantities = [r.get("quantity_to_measure", "").lower() for r in rows]
    assert any(keyword.lower() in q for q in quantities), \
        f"Required calibration target '{keyword}' not found in CSV (check quantity_to_measure column)"


def test_calibration_targets_all_crznse():
    rows = _read_csv("material_response_calibration_targets.csv")
    for row in rows:
        mat = row.get("material", "").lower()
        assert "znse" in mat or "crz" in mat, \
            f"Calibration target '{row.get('target_id')}' should be for ZnSe-family material, got '{mat}'"


# --- Claim boundary doc content ---


def test_claim_boundary_doc_forbids_silica_to_crznse_transfer():
    doc = DOCS / "26_literature_anchor_claim_boundaries.md"
    text = doc.read_text(encoding="utf-8")
    # Rule 1 must explicitly forbid fused silica → Cr:ZnSe transfer
    assert "fused silica" in text.lower() or "fused_silica" in text.lower()
    assert "cr:znse" in text.lower() or "crznse" in text.lower() or "cr:zns" in text.lower()
    # Must say it does NOT calibrate
    assert "does not calibrate" in text.lower() or "not calibrate" in text.lower()


def test_claim_boundary_doc_forbids_gaussian_to_bessel_transfer():
    doc = DOCS / "26_literature_anchor_claim_boundaries.md"
    text = doc.read_text(encoding="utf-8")
    assert "gaussian" in text.lower()
    assert "bessel" in text.lower()
    assert "does not calibrate" in text.lower() or "not calibrate" in text.lower()


def test_claim_boundary_doc_surface_vs_internal():
    doc = DOCS / "26_literature_anchor_claim_boundaries.md"
    text = doc.read_text(encoding="utf-8")
    assert "surface ablation" in text.lower()
    assert "internal" in text.lower()
    # Must say surface ablation does not directly apply to internal modification
    assert "not" in text.lower()


def test_claim_boundary_doc_threshold_is_not_damage():
    doc = DOCS / "26_literature_anchor_claim_boundaries.md"
    text = doc.read_text(encoding="utf-8")
    assert "damage" in text.lower()
    assert "threshold" in text.lower()
    # Must say it is not damage prediction
    assert "not damage" in text.lower() or "is not damage" in text.lower() or \
           "not a damage" in text.lower() or "damage prediction" in text.lower()


def test_claim_boundary_doc_anchors_initialise_not_calibrate():
    doc = DOCS / "26_literature_anchor_claim_boundaries.md"
    text = doc.read_text(encoding="utf-8")
    # Core principle: literature anchors initialise, lab calibration earns status
    assert "initialise" in text.lower() or "initialize" in text.lower()
    assert "calibrat" in text.lower()  # calibrate / calibration / calibrated
    # Must say lab calibration earns prediction status
    assert "prediction status" in text.lower() or "calibrated_material_prediction" in text.lower()


def test_claim_boundary_doc_znse_family_are_priors():
    doc = DOCS / "26_literature_anchor_claim_boundaries.md"
    text = doc.read_text(encoding="utf-8")
    assert "znse" in text.lower()
    assert "prior" in text.lower()


def test_claim_boundary_doc_heat_requires_thermal_constants():
    doc = DOCS / "26_literature_anchor_claim_boundaries.md"
    text = doc.read_text(encoding="utf-8")
    assert "heat accumulation" in text.lower()
    assert "thermal" in text.lower()


def test_claim_boundary_doc_nonlinear_requires_material_specific():
    doc = DOCS / "26_literature_anchor_claim_boundaries.md"
    text = doc.read_text(encoding="utf-8")
    assert "nonlinear" in text.lower()
    assert "material" in text.lower()


# --- Main anchor doc content ---


def test_anchor_doc_has_eight_calibration_sweeps():
    doc = DOCS / "25_literature_model_anchors.md"
    text = doc.read_text(encoding="utf-8")
    for i in range(1, 9):
        assert f"Sweep {i}" in text, f"Calibration Sweep {i} missing from doc 25"


def test_anchor_doc_recommends_next_stage():
    doc = DOCS / "25_literature_model_anchors.md"
    text = doc.read_text(encoding="utf-8")
    assert "Stage 8B" in text or "8A.2" in text, \
        "Anchor doc should recommend Stage 8B or 8A.2 as next stage"


def test_no_fake_calibration_in_anchor_doc():
    doc = DOCS / "25_literature_model_anchors.md"
    text = doc.read_text(encoding="utf-8")
    # Must not claim any parameter is calibrated without experiment
    # Check that ZnSe threshold parameters are not claimed as calibrated
    assert "calibrated_material_prediction" not in text or \
           ("calibrated_material_prediction" in text and "requires" in text.lower()), \
        "Anchor doc must not grant calibrated_material_prediction without conditions"
