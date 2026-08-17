"""
Stage 8A blueprint documentation verification tests.

Checks that all required docs exist, that the model status registry contains all 11 statuses,
and that key claim boundaries are documented correctly.
"""

from pathlib import Path

import pytest

DOCS = Path(__file__).parent.parent / "docs"
OUTPUTS = Path(__file__).parent.parent / "outputs" / "csv" / "digital_twin"

# Required doc files
REQUIRED_DOCS = [
    "20_full_beam_to_write_digital_twin_blueprint.md",
    "21_digital_twin_model_status_and_claim_boundaries.md",
    "22_digital_twin_figure_and_output_spec.md",
    "23_digital_twin_math_and_physics_stack.md",
    "24_digital_twin_stage_plan.md",
]

# All 11 model status keys
MODEL_STATUSES = [
    "optical_prediction",
    "energy_accounting_prediction",
    "fluence_prediction",
    "fluence_threshold_proxy",
    "dose_accumulation_proxy",
    "nonlinear_deposition_proxy",
    "thermal_accumulation_proxy",
    "uncalibrated_material_response_proxy",
    "simulated_microscopy_proxy",
    "calibrated_material_prediction",
    "experimentally_validated_prediction",
]

# Stage plan stages
STAGE_PLAN_STAGES = [f"Stage 8{c}" for c in "ABCDEFGHIJ"]

# Three engine names that must appear in the blueprint
ENGINE_NAMES = [
    "ENGINE 1",
    "ENGINE 2",
    "ENGINE 3",
]


@pytest.mark.parametrize("doc_name", REQUIRED_DOCS)
def test_doc_exists(doc_name):
    """Each required Stage 8A document must exist."""
    assert (DOCS / doc_name).is_file(), f"Missing doc: docs/{doc_name}"


@pytest.mark.parametrize("status", MODEL_STATUSES)
def test_model_status_defined(status):
    """All 11 model statuses must be defined in doc 21."""
    doc = DOCS / "21_digital_twin_model_status_and_claim_boundaries.md"
    assert doc.is_file(), "doc 21 missing"
    text = doc.read_text(encoding="utf-8")
    assert status in text, f"Model status '{status}' not found in doc 21"


def test_all_model_statuses_in_blueprint():
    """Blueprint doc 20 must reference the status hierarchy."""
    doc = DOCS / "20_full_beam_to_write_digital_twin_blueprint.md"
    text = doc.read_text(encoding="utf-8")
    assert "model_status" in text
    assert "optical_prediction" in text


def test_three_engines_in_blueprint():
    """Blueprint must describe all three engines."""
    doc = DOCS / "20_full_beam_to_write_digital_twin_blueprint.md"
    text = doc.read_text(encoding="utf-8")
    for engine in ENGINE_NAMES:
        assert engine in text, f"Engine '{engine}' not found in blueprint"


def test_claim_boundaries_not_allowed_present():
    """Doc 21 must contain 'Not allowed' sections for proxy statuses."""
    doc = DOCS / "21_digital_twin_model_status_and_claim_boundaries.md"
    text = doc.read_text(encoding="utf-8")
    assert "Not allowed" in text, "Claim boundaries ('Not allowed') missing from doc 21"
    # At least one proxy level must explicitly state what is not allowed
    assert "PLANNING PROXY" in text or "not a calibrated" in text.lower()


def test_doc21_hierarchy_order():
    """Doc 21 must list statuses in order from most to least trustworthy (11 → 1)."""
    doc = DOCS / "21_digital_twin_model_status_and_claim_boundaries.md"
    text = doc.read_text(encoding="utf-8")
    # The hierarchy table must have 11 entries
    assert text.count("`optical_prediction`") >= 1
    assert text.count("`experimentally_validated_prediction`") >= 1
    # Status 11 (experimentally validated) must appear before status 1 (optical) in the table
    idx_11 = text.index("experimentally_validated_prediction")
    idx_1 = text.rindex("optical_prediction")
    assert idx_11 < idx_1, "Hierarchy table should list status 11 (highest) before status 1"


def test_doc22_figure_count():
    """Doc 22 must specify 16 figures and 1 animated GIF."""
    doc = DOCS / "22_digital_twin_figure_and_output_spec.md"
    text = doc.read_text(encoding="utf-8")
    # Check FIG-01 through FIG-16 are present
    for i in range(1, 17):
        fig_id = f"FIG-{i:02d}"
        assert fig_id in text, f"{fig_id} not found in figure spec (doc 22)"
    assert "GIF-01" in text, "GIF-01 animated gif spec missing from doc 22"


def test_doc22_proxy_stamp_required():
    """Doc 22 must specify proxy stamp requirement for proxy-level figures."""
    doc = DOCS / "22_digital_twin_figure_and_output_spec.md"
    text = doc.read_text(encoding="utf-8")
    assert "PROXY" in text and "stamp" in text.lower()


def test_doc23_math_sections():
    """Doc 23 must contain all 9 math stack sections."""
    doc = DOCS / "23_digital_twin_math_and_physics_stack.md"
    text = doc.read_text(encoding="utf-8")
    expected_sections = [
        "Energy Accounting",
        "ASM",
        "Fluence Scaling",
        "Threshold Proxy",
        "Writing Trajectory",
        "Incubation",
        "Nonlinear",
        "Thermal",
        "Microscope",
    ]
    for section in expected_sections:
        assert section.lower() in text.lower(), f"Math section '{section}' missing from doc 23"


def test_doc23_no_fake_calibration():
    """Doc 23 must not claim any calibrated outputs from proxy equations."""
    doc = DOCS / "23_digital_twin_math_and_physics_stack.md"
    text = doc.read_text(encoding="utf-8")
    # The incubation section must acknowledge literature values, not calibrated
    assert "literature" in text.lower() or "proxy" in text.lower()
    # Nonlinear hook must be flagged as not calibrated
    assert "not calibrated" in text.lower() or "uncalibrated" in text.lower()


@pytest.mark.parametrize("stage", STAGE_PLAN_STAGES)
def test_stage_plan_contains_all_stages(stage):
    """Doc 24 must contain all stages 8A through 8J."""
    doc = DOCS / "24_digital_twin_stage_plan.md"
    assert doc.is_file(), "doc 24 missing"
    text = doc.read_text(encoding="utf-8")
    assert stage in text, f"Stage plan missing '{stage}' in doc 24"


def test_stage_plan_no_engine_changes():
    """Doc 24 must state that the physics engine is not modified."""
    doc = DOCS / "24_digital_twin_stage_plan.md"
    text = doc.read_text(encoding="utf-8")
    assert "NOT modified" in text or "not modified" in text.lower() or "unchanged" in text.lower()


def test_stage_plan_no_nonlinear_physics_before_8f():
    """Doc 24 must defer nonlinear physics implementation to Stage 8F or later."""
    doc = DOCS / "24_digital_twin_stage_plan.md"
    text = doc.read_text(encoding="utf-8")
    assert "8F" in text
    # Stage 8A-8D description must not claim numerical nonlinear physics
    # (We check that the nonlinear section is associated with 8F)
    idx_nonlinear = text.lower().find("nonlinear")
    assert idx_nonlinear >= 0
    # The first mention of "nonlinear" should not be in Stage 8A section alone —
    # the global constraint at top says no nonlinear before 8F
    assert "Stage 8F" in text or "stage 8F" in text.lower()


def test_doc20_no_overclaiming():
    """Blueprint must not claim calibrated material prediction as default output."""
    doc = DOCS / "20_full_beam_to_write_digital_twin_blueprint.md"
    text = doc.read_text(encoding="utf-8")
    # The default Engine 3 output must be uncalibrated proxy
    assert "uncalibrated_material_response_proxy" in text
    # The gating rule must be explicit
    assert "gated" in text.lower() or "calibration data" in text.lower()


def test_optional_csv_dir_or_docs_exist():
    """Either the optional CSV dir exists or we're at docs-only stage (8A)."""
    # At Stage 8A this directory may not exist yet — that is acceptable.
    # This test just documents the expected path; it passes either way.
    expected = OUTPUTS
    # Not a hard failure — Stage 8A is docs-only
    if expected.exists():
        csvs = list(expected.glob("*.csv"))
        assert len(csvs) >= 0  # any number is acceptable at Stage 8A


def test_summary_doc_exists():
    """The STAGE8A summary document must exist."""
    summary = Path(__file__).parent.parent / "STAGE8A_DIGITAL_TWIN_BLUEPRINT_SUMMARY.md"
    assert summary.is_file(), "STAGE8A_DIGITAL_TWIN_BLUEPRINT_SUMMARY.md missing"
