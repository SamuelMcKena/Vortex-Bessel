"""Stage 8C.1 cockpit dashboard tests."""

import numpy as np
import pytest

from vbb_study.digital_twin.cockpit_dashboard import (
    build_warning_flags,
    choose_display_plane,
    compute_peak_location_diagnostics,
    make_interpretation_text,
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
        stack,
        fluence,
        target_depth_um=controls["focus_depth_um"],
        central_roi_half_width_um=1.5,
        selected_z_mode="optical_peak",
        pulse_duration_fs=controls["pulse_duration_fs"],
    )
    ledger = build_energy_ledger_from_controls(controls)
    exposure = build_exposure_summary_from_controls(controls, ledger.energy_at_sample_uJ)
    report = build_lab_realism_report(
        controls,
        energy_ledger=ledger,
        exposure_summary=exposure,
        field_summary={"source_status": "unit_test_fixture", "dx_um": stack.dx_um, "dy_um": stack.dy_um},
        diagnostics=diagnostics,
    )
    return controls, stack, fluence, diagnostics, ledger, exposure, report


def test_peak_location_diagnostics_detect_crop_edge_peak():
    _, _, _, diagnostics, _, _, _ = _fixtures()
    assert diagnostics["global_peak_near_boundary"] is True
    assert diagnostics["global_peak_distance_to_boundary_px"] == 0


def test_central_roi_peak_is_computed():
    _, _, _, diagnostics, _, _, _ = _fixtures()
    assert diagnostics["central_roi_peak_value"] > 0
    assert diagnostics["central_roi_peak_z_um"] in {0.0, 50.0, 100.0, 150.0}


def test_target_depth_peak_is_computed():
    _, _, _, diagnostics, _, _, _ = _fixtures()
    assert diagnostics["target_depth_z_um"] == 100.0
    assert diagnostics["target_depth_peak_value"] > 0


def test_sample_surface_peak_is_computed():
    _, _, _, diagnostics, _, _, _ = _fixtures()
    assert diagnostics["sample_surface_z_um"] == 0.0
    assert diagnostics["sample_surface_peak_value"] > 0


@pytest.mark.parametrize("mode", ["target_depth", "optical_peak", "sample_surface", "custom"])
def test_selected_plane_modes(mode):
    z = np.array([0.0, 50.0, 100.0])
    selected = choose_display_plane(
        z,
        selected_z_mode=mode,
        target_depth_um=100.0,
        custom_z_um=50.0,
        optical_peak_z_um=0.0,
        global_peak_near_boundary=False,
    )
    assert selected["selected_plane_z_um"] in set(z.tolist())
    assert selected["selected_plane_reason"] in {mode, "target_depth"}


def test_safe_selected_plane_avoids_crop_edge_headline():
    _, _, _, diagnostics, _, _, _ = _fixtures()
    assert diagnostics["selected_plane_reason"].startswith("target_depth_safe")
    assert diagnostics["selected_plane_z_um"] == 100.0


def test_warning_flags_include_required_categories():
    flags = build_warning_flags(
        first_order_geometry_valid=False,
        pupil_clipping_fraction=0.1,
        power_limit_exceeded=True,
        pulse_overlap_fraction=-0.2,
        captured_power_drift_fraction=0.5,
        diagnostics={"global_peak_near_boundary": True},
    )
    for key in [
        "captured_power_drift",
        "crop_edge_peak",
        "first_order_geometry",
        "pupil_clipping",
        "power_limit",
        "pulse_overlap",
    ]:
        assert key in flags
        assert flags[key]["status"] in {"pass", "caution", "fail"}


def test_dashboard_plotting_returns_matplotlib_figure():
    controls, stack, fluence, diagnostics, ledger, exposure, report = _fixtures()
    fig = plot_integrated_cockpit_dashboard(
        stack,
        fluence,
        controls=controls,
        energy_ledger=ledger,
        exposure_summary=exposure,
        lab_report=report,
        diagnostics=diagnostics,
    )
    assert hasattr(fig, "savefig")
    assert fig.stage8c1_metadata["final_export_allowed"] is False


def test_saving_with_caveats_hidden_raises():
    controls, stack, fluence, diagnostics, ledger, exposure, report = _fixtures()
    with pytest.raises(CaveatsRequiredError):
        plot_integrated_cockpit_dashboard(
            stack,
            fluence,
            controls=controls,
            energy_ledger=ledger,
            exposure_summary=exposure,
            lab_report=report,
            diagnostics=diagnostics,
            output_path="outputs/figures/digital_twin/should_not_write_without_caveats.png",
            show_caveats=False,
        )


def test_interpretation_text_includes_required_content():
    controls, _, _, diagnostics, ledger, exposure, report = _fixtures()
    text = make_interpretation_text(
        controls=controls,
        energy_ledger=ledger,
        exposure_summary=exposure,
        diagnostics=diagnostics,
        lab_report=report,
    ).lower()
    for phrase in [
        "energy at sample",
        "pulse spacing",
        "selected plane",
        "central roi",
        "target-depth",
        "crop boundary",
        "captured-power drift",
        "lab realism status",
        "no material modification",
    ]:
        assert phrase in text
