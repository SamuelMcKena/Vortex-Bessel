"""Stage 8C.3R.5.3 bench inventory + physical-4F readiness gate tests."""

from dataclasses import replace

import numpy as np

from vbb_study.digital_twin.cslm_route import CSLMRouteConfig, run_cslm_baseline_route
from vbb_study.digital_twin.control_contract import (
    build_default_demo_profile, build_measured_bench_template, validate_hardware_profile,
)
from vbb_study.digital_twin.coordinate_contract import (
    build_coordinate_frames, build_coordinate_transforms, validate_coordinate_contract,
    VALID_UNITS, VALID_CALIBRATION_STATUS, VALID_PROVENANCE,
)
from vbb_study.digital_twin.bench_inventory import (
    PHYSICAL_4F_HARD_BLOCKERS, MEASURED_BENCH_EXTRA,
    build_bench_inventory, build_bench_inventory_profile,
    evaluate_physical_4f_readiness,
)


def _full_4f_overlay(provenance="estimated"):
    vals = {
        "wavelength_nm": 1030.0, "slm2_pixel_pitch_um": 8.0, "slm2_carrier_frequency_cpm": 30000.0,
        "slm2_to_lens1_distance_mm": 100.0, "fourier_lens1_focal_length_mm": 100.0,
        "fourier_lens1_clear_aperture_mm": 25.4, "lens1_to_fourier_plane_distance_mm": 100.0,
        "fourier_filter_centre_x_um": 0.0, "fourier_filter_centre_y_um": 0.0,
        "fourier_filter_radius_um": 250.0, "fourier_filter_shape": "circular",
        "fourier_plane_to_lens2_distance_mm": 100.0, "fourier_lens2_focal_length_mm": 100.0,
        "fourier_lens2_clear_aperture_mm": 25.4, "lens2_to_output_plane_distance_mm": 100.0,
    }
    return {cid: {"value": v, "provenance": provenance} for cid, v in vals.items()}


def _frames_fourier_known():
    frames = list(build_coordinate_frames())
    for i, f in enumerate(frames):
        if f.frame_id == "Fourier_plane_physical_position_frame":
            frames[i] = replace(f, calibration_status="declared_model_convention", provenance="estimated")
    return tuple(frames)


# 1. Demo profile remains diagnostic and does not claim measured-bench readiness.
def test_demo_profile_not_measured_bench():
    demo = build_default_demo_profile()
    assert demo["profile_status"] == "diagnostic_demo_not_measured_bench"
    r = evaluate_physical_4f_readiness()
    assert r["D_measured_bench_camera"]["ready"] is False


# 2. Measured-bench template contains no invented values.
def test_measured_bench_template_blank():
    tmpl = build_measured_bench_template()
    assert validate_hardware_profile(tmpl) == []
    for cid, entry in tmpl["controls"].items():
        if entry["status"] == "derived_read_only":
            continue
        assert entry["value"] is None and entry["provenance"] == "unknown", cid


# 3. Initial scalar 4F readiness is false when any hard blocker is null.
def test_initial_4f_false_when_blocker_null():
    r = evaluate_physical_4f_readiness()  # default demo: several blockers null
    assert r["C_initial_scalar_4f_model"]["ready"] is False
    assert len(r["C_initial_scalar_4f_model"]["blocked_by"]) > 0


# 4. 4F readiness stays false if values exist but the coordinate convention is unknown.
def test_4f_false_when_coordinate_unknown_even_if_values_present():
    overlay = _full_4f_overlay("estimated")  # all 4F values present...
    r = evaluate_physical_4f_readiness(inventory_overlay=overlay)  # ...but default frames unknown
    assert r["C_initial_scalar_4f_model"]["ready"] is False
    assert any("coordinate convention unknown" in b for b in r["C_initial_scalar_4f_model"]["blocked_by"])


# 5. Fully populated fixture may pass C, but D requires measured provenance + transforms.
def test_full_fixture_passes_C_but_not_measured_bench():
    overlay = _full_4f_overlay("estimated")
    frames = _frames_fourier_known()
    r = evaluate_physical_4f_readiness(inventory_overlay=overlay, frames=frames)
    assert r["C_initial_scalar_4f_model"]["ready"] is True       # values + coord convention known
    assert r["D_measured_bench_camera"]["ready"] is False        # estimated, not measured


# 6. Inventory-only 4F data does not change the active field or enable 4F physics.
def test_inventory_only_does_not_change_field_or_enable_4f():
    cfg = CSLMRouteConfig.fast()
    base = run_cslm_baseline_route(cfg).post_slm2_unfiltered_state.field
    # changing the inventory overlay does not touch the config or the field
    overlay = _full_4f_overlay("measured")
    again = run_cslm_baseline_route(cfg).post_slm2_unfiltered_state.field
    assert np.allclose(base, again)
    r = evaluate_physical_4f_readiness(inventory_overlay=overlay)
    assert r["fourier_filter_physics_available"] is False


# 7. Coordinate frames/transforms use valid unit/status/provenance values.
def test_coordinate_contract_valid():
    assert validate_coordinate_contract() == []
    for f in build_coordinate_frames():
        assert f.transverse_units in VALID_UNITS
        assert f.axial_units in VALID_UNITS
        assert f.calibration_status in VALID_CALIBRATION_STATUS
        assert f.provenance in VALID_PROVENANCE
    ids = {f.frame_id for f in build_coordinate_frames()}
    for t in build_coordinate_transforms():
        assert t.source_frame_id in ids and t.destination_frame_id in ids


# 9-ish. Bench inventory profile is diagnostic and unknowns remain null.
def test_inventory_profile_diagnostic_and_unknowns_null():
    prof = build_bench_inventory_profile()
    assert prof["profile_status"] == "diagnostic_demo_inventory_not_measured_bench"
    for cid, entry in prof["items"].items():
        if entry["provenance"] == "unknown":
            assert entry["value"] is None, cid
    # nothing is presented as measured in the demo inventory
    assert all(entry["provenance"] != "measured" for entry in prof["items"].values())


# 10. No material / camera-physics / 4F-physics / GUI / Stage 8D claims introduced.
def test_no_unsafe_claims_in_readiness_and_inventory():
    import json
    r = evaluate_physical_4f_readiness()
    prof = build_bench_inventory_profile()
    blob = (json.dumps(r) + json.dumps(prof)).lower()
    for token in ("dose", "thermal", "plasma", "nonlinear", "ablation", "stage8d",
                  "camera_physics", "thin_lens_applied", "4f_field_generated"):
        assert token not in blob
    assert r["final_export_allowed"] is False
    # every D blocker references a real measurement/transform gap (no silent readiness)
    assert r["D_measured_bench_camera"]["ready"] is False
    assert len(MEASURED_BENCH_EXTRA) > 0 and len(PHYSICAL_4F_HARD_BLOCKERS) > 0
