"""Stage 8C.3 governance tests."""

import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.text as mtext  # noqa: E402
import numpy as np
import pytest

from vbb_study.digital_twin.cockpit_dashboard import (
    build_warning_flags,
    compute_peak_location_diagnostics,
    plot_integrated_cockpit_dashboard,
)
from vbb_study.digital_twin.field_coupling import stack_from_arrays
from vbb_study.digital_twin.field_fluence import scale_stack_to_fluence
from vbb_study.digital_twin.lab_perturbations import apply_lab_perturbations_to_stack


ROOT = Path(__file__).parent.parent
DOC = ROOT / "docs" / "32_active_lab_realism_coupling.md"
SUMMARY = ROOT / "STAGE8C3_ACTIVE_LAB_REALISM_SUMMARY.md"
PREVIEW = ROOT / "outputs" / "figures" / "digital_twin" / "stage8c3_baseline_vs_perturbed_preview.png"
SWEEP_PREVIEW = ROOT / "outputs" / "figures" / "digital_twin" / "stage8c3_misalignment_sensitivity_sweep_preview.png"
C3C_PREVIEW = ROOT / "outputs" / "figures" / "digital_twin" / "stage8c3c_genuine_degradation_sweep_preview.png"
C3D_PREVIEW = ROOT / "outputs" / "figures" / "digital_twin" / "stage8c3d_conservation_axis_diagnostics_preview.png"

STAGE8C3_FILES = [
    ROOT / "vbb_study" / "digital_twin" / "lab_perturbations.py",
    ROOT / "vbb_study" / "digital_twin" / "active_realism_metrics.py",
    ROOT / "vbb_study" / "digital_twin" / "lab_realism_controls.py",
    ROOT / "vbb_study" / "digital_twin" / "cockpit_dashboard.py",
    ROOT / "notebooks" / "digital_twin" / "00_full_beam_to_write_cockpit_MVP.ipynb",
]

FORBIDDEN_STATUSES = [
    "fluence_threshold_proxy",
    "dose_accumulation_proxy",
    "uncalibrated_material_response_proxy",
    "calibrated_material_prediction",
    "experimentally_validated_prediction",
]

BOUNDARY_TERMS = [
    "damage prediction",
    "void prediction",
    "waveguide prediction",
    "ablation prediction",
    "material response",
]

BOUNDARY_MARKERS = (
    "not", "no ", "disabled", "future", "later", "does not",
    "do not", "without", "claim boundary", "optical fluence only",
    "diagnostic only", "remain",
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8").lower()


def test_stage8c3_docs_exist():
    assert DOC.is_file(), DOC
    assert SUMMARY.is_file(), SUMMARY


def test_docs_explain_active_report_future_and_poynting_boundary():
    text = _text(DOC)
    for term in ["physics_active", "energy_active", "warning_only", "future_not_implemented"]:
        assert term in text
    assert "direct poynting-vector editing" in text
    assert "angular-spectrum phase ramp" in text
    assert "baseline-vs-perturbed" in text
    assert "translation" in text
    assert "registered similarity" in text
    assert "physical placement" in text
    assert "passive throughput" in text
    assert "transmitted fraction" in text
    assert "commanded-axis" in text
    assert "field-of-view" in text
    assert "no material response" in text or "not material response" in text


def test_no_forbidden_output_statuses_in_stage8c3_files():
    blob = "\n".join(_text(path) for path in STAGE8C3_FILES if path.is_file())
    for status in FORBIDDEN_STATUSES:
        assert status not in blob


def test_boundary_terms_only_in_boundary_context():
    blob = "\n".join(_text(path) for path in STAGE8C3_FILES + [DOC, SUMMARY] if path.is_file())
    norm = re.sub(r"\s+", " ", blob)
    violations = []
    for term in BOUNDARY_TERMS:
        start = 0
        while True:
            idx = norm.find(term, start)
            if idx == -1:
                break
            context = norm[max(0, idx - 100): idx + len(term) + 100]
            if not any(marker in context for marker in BOUNDARY_MARKERS):
                violations.append(context)
            start = idx + len(term)
    assert not violations, violations


def test_dashboard_warns_if_enabled_control_is_uncoupled():
    x = np.linspace(-5.0, 5.0, 21)
    y = x.copy()
    z = np.array([0.0, 50.0])
    X, Y = np.meshgrid(x, y, indexing="xy")
    stack = stack_from_arrays(
        np.stack([np.exp(-(X**2 + Y**2) / 8.0), np.exp(-(X**2 + Y**2) / 9.0)]),
        x,
        y,
        z,
        source_status="unit_test_fixture",
    )
    result = apply_lab_perturbations_to_stack(
        stack,
        {"enable_first_order_filter_decentre": True, "first_order_filter_decentre_x_px": 5.0},
    )
    flags = build_warning_flags(perturbation_result=result)
    assert flags["active_lab_realism"]["status"] == "caution"
    assert "report-only/future" in flags["active_lab_realism"]["message"]


def test_dashboard_renders_active_lab_realism_summary_text():
    x = np.linspace(-5.0, 5.0, 21)
    y = x.copy()
    z = np.array([0.0, 50.0])
    X, Y = np.meshgrid(x, y, indexing="xy")
    stack = stack_from_arrays(
        np.stack([np.exp(-(X**2 + Y**2) / 8.0), np.exp(-(X**2 + Y**2) / 9.0)]),
        x,
        y,
        z,
        source_status="unit_test_fixture",
    )
    fluence = scale_stack_to_fluence(stack, 10.0)
    diagnostics = compute_peak_location_diagnostics(
        stack,
        fluence,
        target_depth_um=0.0,
        central_roi_half_width_um=3.0,
        pulse_duration_fs=260.0,
    )
    result = apply_lab_perturbations_to_stack(
        stack,
        {"enable_first_order_filter_decentre": True, "first_order_filter_decentre_x_px": 5.0},
    )
    fig = plot_integrated_cockpit_dashboard(
        stack,
        fluence,
        controls={"focus_depth_um": 0.0, "central_roi_half_width_um": 3.0},
        diagnostics=diagnostics,
        perturbation_result=result,
        degradation_metrics={
            "centroid_x_um": 0.0,
            "centroid_y_um": 0.0,
            "symmetry_score": 0.9,
            "azimuthal_uniformity_score": 0.8,
            "core_fill_fraction": 0.2,
            "peak_fluence_change_fraction": 0.1,
            "pupil_clipped_power_fraction": 0.0,
            "captured_power_drift_fraction": 0.0,
        },
    )
    text = "\n".join(t.get_text() for t in fig.findobj(mtext.Text))
    assert "Active lab realism perturbations" in text
    assert "Active perturbation sensitivity available" in text
    assert "stage8c3_misalignment_sensitivity_sweep_preview.png" in text
    assert "REPORT_ONLY/FUTURE" in text
    assert "Compact degradation metrics" in text
    plt.close(fig)


def test_preview_png_metadata_if_present():
    if not PREVIEW.is_file():
        pytest.skip("Stage 8C.3 preview has not been generated in this checkout.")
    from PIL import Image
    with Image.open(PREVIEW) as img:
        meta = dict(getattr(img, "text", {}) or {})
    assert meta.get("final_export_allowed") == "False"
    assert meta.get("figure_status") == "diagnostic_allowed"
    assert meta.get("model_status") == "diagnostic_preview"


def test_sensitivity_sweep_preview_metadata_if_present():
    if not SWEEP_PREVIEW.is_file():
        pytest.skip("Stage 8C.3B sensitivity preview has not been generated in this checkout.")
    from PIL import Image
    with Image.open(SWEEP_PREVIEW) as img:
        meta = dict(getattr(img, "text", {}) or {})
    assert meta.get("final_export_allowed") == "False"
    assert meta.get("figure_status") == "diagnostic_allowed"
    assert meta.get("model_status") == "diagnostic_preview"
    assert meta.get("stage") == "stage8c3b_misalignment_sensitivity_sweep"


def test_stage8c3c_genuine_degradation_preview_metadata_if_present():
    if not C3C_PREVIEW.is_file():
        pytest.skip("Stage 8C.3C genuine-degradation preview has not been generated in this checkout.")
    from PIL import Image
    with Image.open(C3C_PREVIEW) as img:
        meta = dict(getattr(img, "text", {}) or {})
    assert meta.get("final_export_allowed") == "False"
    assert meta.get("figure_status") == "diagnostic_allowed"
    assert meta.get("model_status") == "diagnostic_preview"
    assert meta.get("stage") == "stage8c3c_genuine_degradation_sweep"


def test_stage8c3d_conservation_axis_preview_metadata_if_present():
    if not C3D_PREVIEW.is_file():
        pytest.skip("Stage 8C.3D conservation/axis preview has not been generated in this checkout.")
    from PIL import Image
    with Image.open(C3D_PREVIEW) as img:
        meta = dict(getattr(img, "text", {}) or {})
    assert meta.get("final_export_allowed") == "False"
    assert meta.get("figure_status") == "diagnostic_allowed"
    assert meta.get("model_status") == "diagnostic_preview"
    assert meta.get("stage") == "stage8c3d_conservation_axis_diagnostics"


def test_core_optical_physics_not_modified():
    lock_sensitive = [
        "Publication_Study/bessel_twin_core.py",
        "Publication_Study/vbb_study/equations/propagation.py",
        "Publication_Study/vbb_study/equations/scalar_bessel.py",
        "Publication_Study/tests/test_characterisation_lock.py",
    ]
    out = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(ROOT.parent),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if out.returncode != 0:
        pytest.skip("git status failed")
    modified = {line[3:].strip().strip('"').replace("\\", "/") for line in out.stdout.splitlines()}
    for path in lock_sensitive:
        assert path not in modified, f"lock-sensitive file modified: {path}"
