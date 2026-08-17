"""Stage 8C.3R.2 annular-axis tracking and sensitivity-study lock tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from vbb_study.digital_twin.annular_axis_tracking import (
    RAW_PEAK_LABEL,
    estimate_annular_axis,
    track_axis_trajectory,
)
from vbb_study.digital_twin.component_plane_pipeline import (
    ComponentPlaneConfig,
    run_component_plane_pipeline,
)
from vbb_study.digital_twin.component_plane_metrics import (
    DIAGNOSTIC_SWEEP_LABEL,
    build_response_curve_families,
    compute_axis_tracking,
    run_response_curve,
)
from vbb_study.digital_twin.component_plane_validation import (
    fov_convergence_check,
    validate_beam_tilt,
)

CFG = ComponentPlaneConfig.fast()


def _peak_plane(stack):
    return int(np.argmax(stack.intensity_zyx.max(axis=(1, 2))))


def _synthetic_ring(azimuth_phase: float = 0.0, broken: bool = False):
    x = np.linspace(-24.0, 24.0, 121)
    X, Y = np.meshgrid(x, x)
    r = np.hypot(X, Y)
    th = np.arctan2(Y, X)
    ring = np.exp(-0.5 * ((r - 7.0) / 0.9) ** 2)
    halo = 0.08 * np.exp(-0.5 * ((r - 13.0) / 3.2) ** 2)
    az = 1.0 + 0.22 * np.cos(th - azimuth_phase)
    if broken:
        az = np.where(np.cos(th) > 0.45, 1.0, 0.03)
    return (ring * az + halo).astype(float), x


def test_clean_annular_baseline_has_near_zero_fitted_axis():
    st = run_component_plane_pipeline({}, config=CFG).propagated_stack
    est = estimate_annular_axis(st.intensity_zyx[_peak_plane(st)], st.x_um, st.y_um)
    assert est["beam_axis_method"] in ("ring_fit", "core_fit")
    assert est["beam_axis_error_um"] <= 0.35 * CFG.dx_um
    assert est["ring_fit_reliable"] is True
    assert est["brightest_pixel_status"] == RAW_PEAK_LABEL


def test_raw_brightest_pixel_can_move_without_moving_fitted_ring_centre():
    p0, x = _synthetic_ring(0.0)
    p1, _ = _synthetic_ring(np.pi)
    e0 = estimate_annular_axis(p0, x, x)
    e1 = estimate_annular_axis(p1, x, x)
    fitted_delta = float(np.hypot(e0["beam_axis_x_um"] - e1["beam_axis_x_um"],
                                  e0["beam_axis_y_um"] - e1["beam_axis_y_um"]))
    raw_delta = float(np.hypot(e0["brightest_pixel_x_um"] - e1["brightest_pixel_x_um"],
                               e0["brightest_pixel_y_um"] - e1["brightest_pixel_y_um"]))
    assert fitted_delta < 0.5
    assert raw_delta > 10.0
    assert e0["brightest_pixel_status"] == RAW_PEAK_LABEL
    assert e1["brightest_pixel_status"] == RAW_PEAK_LABEL


def test_ring_core_hierarchy_is_used_instead_of_raw_peak_for_axis_tracking():
    st = run_component_plane_pipeline({}, config=CFG).propagated_stack
    ax = compute_axis_tracking(st)
    assert ax["beam_axis_method"] in ("ring_fit", "core_fit", "roi_centroid")
    assert ax["beam_axis_method"] != "raw_brightest_pixel"
    assert ax["peak_status"] == RAW_PEAK_LABEL
    assert ax["radial_axis_error_um"] < 1.0
    assert abs(ax["peak_x_um"]) > 1.0 or abs(ax["peak_y_um"]) > 1.0


def test_beam_tilt_fitted_axis_slope_agrees_with_theory():
    tv = validate_beam_tilt(24.0, config=CFG)
    assert tv["relative_error"] <= 0.06
    assert tv["fit_quality"] >= 0.90
    assert tv["grid_resolved_displacement"] >= 3.0
    assert np.sign(tv["measured_slope_x"]) == np.sign(tv["expected_slope_x"])


def test_deformed_annulus_reduces_ring_fit_confidence_or_falls_back():
    plane, x = _synthetic_ring(broken=True)
    est = estimate_annular_axis(plane, x, x)
    assert est["ring_fit_quality"] < 0.70 or est["beam_axis_method"] != "ring_fit"
    assert est["beam_axis_method"] != "raw_brightest_pixel"
    assert est["brightest_pixel_status"] == RAW_PEAK_LABEL


def test_fov_convergence_is_not_driven_by_raw_peak_wandering():
    fov = fov_convergence_check({}, config=CFG)
    assert fov["metric_convergence_status"] == "numerically_reliable"
    assert fov["ring_centre_difference_um"] < 0.5
    assert fov["axis_trajectory_difference_um"] < 0.5
    assert fov["raw_peak_position_difference_um"] > 1.0
    assert fov["raw_peak_status"] == RAW_PEAK_LABEL


def test_undersized_fov_is_rejected():
    bad_cfg = replace(CFG, grid_N=44, dx_um=0.5)
    bad = fov_convergence_check({}, config=bad_cfg)
    assert bad["metric_convergence_status"] == "invalid_out_of_frame"


def test_response_curves_are_labelled_diagnostic_not_tolerances():
    families = build_response_curve_families()
    assert set(families) == {
        "vortex_offset", "axicon_offset", "beam_decentre", "beam_tilt",
        "pupil_decentre", "defocus", "astigmatism", "coma", "spherical",
        "zero_order",
    }
    curve = run_response_curve("zero_order", config=ComponentPlaneConfig.fast(grid_N=96, n_z=12))
    assert curve.label == DIAGNOSTIC_SWEEP_LABEL
    assert "Diagnostic sensitivity sweep" in curve.label
    assert "Not an experimentally measured laboratory tolerance" in curve.label
    assert all(row["diagnostic_label"] == DIAGNOSTIC_SWEEP_LABEL for row in curve.rows)


def test_no_material_response_claims_in_c3r2_public_files():
    paths = [
        Path("vbb_study/digital_twin/annular_axis_tracking.py"),
        Path("vbb_study/digital_twin/component_plane_figures.py"),
        Path("vbb_study/digital_twin/component_plane_metrics.py"),
    ]
    blob = "\n".join(p.read_text(encoding="utf-8").lower() for p in paths)
    assert "final_export_allowed=false" in blob
    assert "no material response" in blob
    forbidden_claims = (
        "calibrated_material_prediction",
        "experimentally_validated_prediction",
        "waveguide_prediction",
        "material response is predicted",
    )
    assert not any(term in blob for term in forbidden_claims)


def test_axis_trajectory_rejects_unreliable_crop_limited_planes():
    st = run_component_plane_pipeline({}, config=replace(CFG, grid_N=44, dx_um=0.5)).propagated_stack
    traj = track_axis_trajectory(st.intensity_zyx, st.x_um, st.y_um, st.z_um)
    assert traj["valid_plane_fraction"] < 1.0
