"""Stage 8C.2 polished cockpit dashboard tests.

Covers the visual rescue: GridSpec layout, header status badge, beam-path strip,
warning cards, annotated panels, save guard, visible warning status, and the
diagnostic preview save with stamped PNG metadata.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.text as mtext  # noqa: E402

import numpy as np
import pytest

from vbb_study.digital_twin.cockpit_dashboard import (
    build_beam_path_strip,
    compute_overall_status,
    plot_integrated_cockpit_dashboard,
)
from vbb_study.digital_twin.field_coupling import stack_from_arrays
from vbb_study.digital_twin.field_figures import CaveatsRequiredError
from vbb_study.digital_twin.field_fluence import scale_stack_to_fluence
from vbb_study.digital_twin.lab_realism_controls import (
    build_energy_ledger_from_controls,
    build_exposure_summary_from_controls,
    build_lab_realism_report,
    default_lab_controls,
)
from vbb_study.digital_twin.cockpit_dashboard import compute_peak_location_diagnostics


def _stack_with_edge_peak():
    x = np.linspace(-2.0, 2.0, 5)
    y = np.linspace(-2.0, 2.0, 5)
    z = np.array([0.0, 50.0, 100.0, 150.0])
    intensity = np.ones((4, 5, 5))
    intensity[2, 2, 2] = 20.0
    intensity[3, 0, 0] = 1000.0
    return stack_from_arrays(intensity, x, y, z)


def _fixtures():
    controls = default_lab_controls()
    controls["focus_depth_um"] = 100.0
    stack = _stack_with_edge_peak()
    fluence = scale_stack_to_fluence(stack, 10.0)
    diagnostics = compute_peak_location_diagnostics(
        stack, fluence, target_depth_um=controls["focus_depth_um"],
        central_roi_half_width_um=1.5, selected_z_mode="optical_peak",
        pulse_duration_fs=controls["pulse_duration_fs"],
    )
    ledger = build_energy_ledger_from_controls(controls)
    exposure = build_exposure_summary_from_controls(controls, ledger.energy_at_sample_uJ)
    report = build_lab_realism_report(
        controls, energy_ledger=ledger, exposure_summary=exposure,
        field_summary={"source_status": "unit_test_fixture", "dx_um": stack.dx_um, "dy_um": stack.dy_um},
        diagnostics=diagnostics,
    )
    return controls, stack, fluence, diagnostics, ledger, exposure, report


def _all_text(fig) -> str:
    return "\n".join(t.get_text() for t in fig.findobj(mtext.Text))


# ---------------------------------------------------------------------------
# compute_overall_status
# ---------------------------------------------------------------------------


def test_overall_status_pass_when_all_clear():
    assert compute_overall_status({"a": {"status": "pass"}}, diagnostics={"warning_level": "pass"}) == "pass"


def test_overall_status_caution_from_flag():
    assert compute_overall_status({"a": {"status": "caution"}}) == "caution"


def test_overall_status_fail_from_flag():
    assert compute_overall_status({"a": {"status": "fail"}}) == "fail"


def test_overall_status_caution_from_diagnostics():
    assert compute_overall_status({}, diagnostics={"warning_level": "caution"}) == "caution"


def test_overall_status_missing_is_not_fail_or_caution():
    # 'missing' is informational for the headline badge.
    assert compute_overall_status({"a": {"status": "missing"}}) == "pass"


# ---------------------------------------------------------------------------
# build_beam_path_strip
# ---------------------------------------------------------------------------


def test_beam_path_strip_has_all_stages():
    _, _, _, _, _, _, report = _fixtures()
    strip = build_beam_path_strip(report)
    assert len(strip) == 13
    for chip in strip:
        assert chip["short_label"]
        assert chip["status_level"]
        assert "enabled" in chip


def test_beam_path_strip_none_is_empty():
    assert build_beam_path_strip(None) == []


# ---------------------------------------------------------------------------
# Dashboard layout / figure creation
# ---------------------------------------------------------------------------


def test_dashboard_returns_large_multipanel_figure():
    controls, stack, fluence, diagnostics, ledger, exposure, report = _fixtures()
    fig = plot_integrated_cockpit_dashboard(
        stack, fluence, controls=controls, energy_ledger=ledger,
        exposure_summary=exposure, lab_report=report, diagnostics=diagnostics,
    )
    # Polished layout uses many panels and a wide canvas.
    assert len(fig.axes) >= 14
    assert fig.get_size_inches()[0] >= 18
    assert fig.stage8c2_metadata["final_export_allowed"] is False
    assert fig.stage8c2_metadata["overall_status"] in {"pass", "caution", "fail"}
    # Back-compat metadata retained.
    assert fig.stage8c1_metadata["final_export_allowed"] is False


def test_dashboard_has_header_badge_and_regions():
    controls, stack, fluence, diagnostics, ledger, exposure, report = _fixtures()
    fig = plot_integrated_cockpit_dashboard(
        stack, fluence, controls=controls, energy_ledger=ledger,
        exposure_summary=exposure, lab_report=report, diagnostics=diagnostics,
    )
    text = _all_text(fig)
    for region in [
        "DISPLAY TRUST",
        "Beam path - stage status",
        "Warnings & feasibility flags",
        "Energy ledger",
        "Experiment request",
        "Exposure bookkeeping",
        "XZ optical fluence along propagation",
        "Central ROI fluence zoom",
        "Peak fluence vs z",
        "Raw captured-power fraction vs z",
        "Interpretation & claim boundary",
        "Future physics - disabled",
    ]:
        assert region in text, f"missing dashboard region: {region}"


def test_dashboard_visible_warning_status_is_legible():
    """The edge-peak fixture must surface a non-pass status badge word."""
    controls, stack, fluence, diagnostics, ledger, exposure, report = _fixtures()
    overall = compute_overall_status(
        {}, diagnostics=diagnostics, lab_report=report
    )
    assert overall in {"caution", "fail"}
    fig = plot_integrated_cockpit_dashboard(
        stack, fluence, controls=controls, energy_ledger=ledger,
        exposure_summary=exposure, lab_report=report, diagnostics=diagnostics,
    )
    text = _all_text(fig)
    assert ("CAUTION" in text) or ("FAIL" in text)
    # Crop-edge peak warning must be visible on the XZ panel.
    assert "global peak near crop boundary" in text.lower()


def test_dashboard_annotations_present():
    controls, stack, fluence, diagnostics, ledger, exposure, report = _fixtures()
    fig = plot_integrated_cockpit_dashboard(
        stack, fluence, controls=controls, energy_ledger=ledger,
        exposure_summary=exposure, lab_report=report, diagnostics=diagnostics,
    )
    text = _all_text(fig).lower()
    assert "selected z" in text
    assert "roi peak" in text
    assert "e@sample" in text
    assert "drift" in text
    assert "target depth" in text


def test_dashboard_claim_boundary_text_present():
    controls, stack, fluence, diagnostics, ledger, exposure, report = _fixtures()
    fig = plot_integrated_cockpit_dashboard(
        stack, fluence, controls=controls, energy_ledger=ledger,
        exposure_summary=exposure, lab_report=report, diagnostics=diagnostics,
    )
    text = _all_text(fig).lower()
    assert "optical fluence only" in text
    assert "not material modification" in text


# ---------------------------------------------------------------------------
# Save guard + preview save
# ---------------------------------------------------------------------------


def test_saving_with_caveats_hidden_raises():
    controls, stack, fluence, diagnostics, ledger, exposure, report = _fixtures()
    with pytest.raises(CaveatsRequiredError):
        plot_integrated_cockpit_dashboard(
            stack, fluence, controls=controls, energy_ledger=ledger,
            exposure_summary=exposure, lab_report=report, diagnostics=diagnostics,
            output_path="outputs/figures/digital_twin/_should_not_write.png",
            show_caveats=False,
        )


def test_preview_save_writes_stamped_png(tmp_path):
    controls, stack, fluence, diagnostics, ledger, exposure, report = _fixtures()
    out = tmp_path / "stage8c2_preview.png"
    plot_integrated_cockpit_dashboard(
        stack, fluence, controls=controls, energy_ledger=ledger,
        exposure_summary=exposure, lab_report=report, diagnostics=diagnostics,
        output_path=out, show_caveats=True, dpi=80,
    )
    assert out.is_file()
    from PIL import Image
    with Image.open(out) as img:
        meta = dict(getattr(img, "text", {}) or {})
    assert meta.get("final_export_allowed") == "False"
    assert meta.get("figure_status") == "diagnostic_allowed"
    assert meta.get("model_status") == "diagnostic_preview"
    assert meta.get("overall_status") in {"pass", "caution", "fail"}
