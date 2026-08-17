from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from vbb_study.digital_twin import (
    CANONICAL_OPERATING_POINT_ID,
    MODE2U2F_DEFAULT_OUTPUT_ROOT,
    MODE2V_DEFAULT_OUTPUT_ROOT,
    OLD_BEST_COMPROMISE_ID,
    STRICT_COMPROMISE_ID,
    assert_not_forbidden,
    architecture_decision,
    build_native_masks,
    closed_loop_cases,
    component_table,
    fourf_final_design,
    load_operating_points,
    measurement_responsibility_rows,
    mode2v_outcome,
    power_budget_rows,
    qwp_lab_axis_statement,
    waveplate_table,
    write_mode2u2_fix_strict_hexagon_optimisation,
    write_mode2v_lab_ready_build,
)
from vbb_study.digital_twin.nathan_mode2v_lab_ready_build import (
    FOURF_FOCAL_M,
    LAB_WAVELENGTH_M,
    SLM_HEIGHT_PX,
    SLM_PITCH_M,
    SLM_WIDTH_PX,
)

ROOT = Path(MODE2V_DEFAULT_OUTPUT_ROOT)
FIX_ROOT = Path(MODE2U2F_DEFAULT_OUTPUT_ROOT)


def _ensure_outputs() -> Path:
    required = [
        ROOT / "09_mask_package" / "mode2v_slm_masks_metadata.json",
        ROOT / "08_closed_loop" / "mode2v_closed_loop_results.json",
        ROOT / "10_final_status" / "nathan_mode2v_manifest.json",
    ]
    if not all(path.exists() for path in required):
        write_mode2v_lab_ready_build(
            output_dir=ROOT,
            grid_n=384,
            z_planes=9,
            loop_nm_maxiter=40,
        )
    return ROOT


def _ensure_fix_outputs() -> Path:
    required = [
        FIX_ROOT / "hexagon_classifier_calibration.csv",
        FIX_ROOT / "old_optima_strict_audit.csv",
        FIX_ROOT / "strict_hexagon_candidates.json",
    ]
    if not all(path.exists() for path in required):
        write_mode2u2_fix_strict_hexagon_optimisation(
            output_dir=FIX_ROOT,
            grid_n=384,
            z_planes=9,
            search_max_cases=36,
            highres_grid_n=384,
            run_highres=True,
        )
    return FIX_ROOT


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _json_load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_canonical_operating_point_is_realistic_4f_reference() -> None:
    canonical, secondary = load_operating_points()

    assert canonical["candidate_id"] == CANONICAL_OPERATING_POINT_ID
    assert canonical["candidate_id"] == "REALISTIC_4F_HEXAGON_REFERENCE"
    assert canonical["strict_hexagon_eligible"] is True
    assert secondary["candidate_id"] == STRICT_COMPROMISE_ID
    assert secondary["strict_hexagon_eligible"] is True


def test_old_forbidden_optimizer_candidates_cannot_be_used() -> None:
    with pytest.raises(ValueError, match="strict hexagon gate"):
        assert_not_forbidden(OLD_BEST_COMPROMISE_ID)

    assert_not_forbidden(CANONICAL_OPERATING_POINT_ID)
    assert_not_forbidden(STRICT_COMPROMISE_ID)


def test_native_slm_masks_are_exactly_1920_by_1080() -> None:
    canonical, _ = load_operating_points()
    masks = build_native_masks(canonical)

    assert masks["phi_H"].shape == (SLM_HEIGHT_PX, SLM_WIDTH_PX)
    assert masks["phi_V"].shape == (SLM_HEIGHT_PX, SLM_WIDTH_PX)
    assert masks["metadata"]["panel"]["width_px"] == 1920
    assert masks["metadata"]["panel"]["height_px"] == 1080


def test_wrapped_phase_remains_in_0_to_2pi() -> None:
    canonical, _ = load_operating_points()
    masks = build_native_masks(canonical)

    for key in ("phi_H", "phi_V"):
        phi = masks[key]
        assert float(np.nanmin(phi)) >= 0.0
        assert float(np.nanmax(phi)) < 2.0 * np.pi


def test_mask_metadata_records_candidate_and_phase_convention() -> None:
    root = _ensure_outputs()
    metadata = _json_load(root / "09_mask_package" / "mode2v_slm_masks_metadata.json")

    assert metadata["candidate_id"] == CANONICAL_OPERATING_POINT_ID
    assert metadata["carrier_lpmm"] == pytest.approx(6.25)
    assert "phi_H = +alpha" in metadata["phase_convention"]
    assert "phi_V = -alpha + pi/2" in metadata["phase_convention"]
    assert metadata["qwp_convention"].startswith("code -45 deg")


def test_uncalibrated_png_previews_are_not_hardware_ready_masks() -> None:
    root = _ensure_outputs()
    metadata = _json_load(root / "09_mask_package" / "mode2v_slm_masks_metadata.json")
    readme = (root / "09_mask_package" / "README_MASK_PACKAGE.md").read_text(encoding="utf-8")

    assert metadata["uint8_png_is_preview_only"] is True
    assert metadata["lut_applied"] is False
    assert metadata["hardware_ready"] is False
    assert "PREVIEW ONLY" in readme
    assert "NOT hardware-ready" in readme or "NOT calibrated hardware masks" in readme


def test_hv_route_implements_opposite_alpha_masks_with_common_carrier() -> None:
    canonical, _ = load_operating_points()
    masks = build_native_masks(canonical)
    grid = masks["grid"]
    carrier = 2.0 * np.pi * float(canonical["carrier_lpmm"]) * 1.0e3 * grid["X"]

    sample = (slice(None, None, 80), slice(None, None, 80))
    alpha_from_h = np.mod(masks["phi_H"][sample] - carrier[sample], 2.0 * np.pi)
    alpha_from_v = np.mod(-masks["phi_V"][sample] + 0.5 * np.pi + carrier[sample], 2.0 * np.pi)
    phase_error = np.angle(np.exp(1j * (alpha_from_h - alpha_from_v)))

    assert float(np.max(np.abs(phase_error))) < 1e-10


def test_qwp_convention_is_explicit_in_lab_axis_language() -> None:
    statement = qwp_lab_axis_statement()
    waveplates = waveplate_table(statement)
    qwp = next(row for row in waveplates if row["component_id"] == "QWP1_final")

    text = str(statement["physical_statement"]).lower()
    assert "fast axis" in text
    assert "clockwise" in text
    assert "lab-horizontal" in text or "horizontal" in text
    assert "looking" in str(statement["viewing_convention"]).lower()
    assert qwp["nominal_angle_code_deg"] == -45.0
    assert qwp["essential"] is True


def test_v_arm_hwp_requirement_remains_conditional_until_panel_orientation_check() -> None:
    rows = {row["component_id"]: row for row in waveplate_table()}

    assert rows["HWP2_v_arm_pre_slm"]["conditional"] == "conditional_on_panel_orientation_test"
    assert rows["HWP3_v_arm_post_slm"]["conditional"] == "conditional_on_panel_orientation_test"
    assert rows["HWP2_v_arm_pre_slm"]["essential"] is False
    assert "panel orientation test" in rows["HWP2_v_arm_pre_slm"]["calibration_method"]


def test_4f_displacement_satisfies_lambda_f_carrier() -> None:
    fourf = fourf_final_design(
        simulated_first_order_efficiency=0.9495,
        simulated_zero_order_leakage=0.0,
    )
    carrier_cpm = float(fourf["carrier_lpmm"]) * 1.0e3
    expected_mm = LAB_WAVELENGTH_M * FOURF_FOCAL_M * carrier_cpm / 1.0e-3

    assert fourf["first_order_displacement_mm"] == pytest.approx(expected_mm, rel=1e-12)
    assert fourf["carrier_period_slm_pixels"] == pytest.approx(1.0 / (carrier_cpm * SLM_PITCH_M))


def test_iris_diameter_is_physically_reported_in_mm() -> None:
    fourf = fourf_final_design(
        simulated_first_order_efficiency=0.9495,
        simulated_zero_order_leakage=0.0,
    )

    assert fourf["iris_radius_mm"] > 0.0
    assert fourf["iris_diameter_mm"] == pytest.approx(2.0 * fourf["iris_radius_mm"])
    assert fourf["iris_diameter_mm"] == pytest.approx(1.54, abs=0.02)
    assert fourf["zero_order_clearance_mm"] > 0.0


def test_power_ledger_closes_within_tolerance() -> None:
    canonical, _ = load_operating_points()
    rows = {row["stage"]: row for row in power_budget_rows(canonical, grid_n=384)}
    f = lambda stage: float(rows[stage]["model_fraction_of_input"])

    assert f("09_selected_plus1_order_h") + f("11_rejected_power_h") == pytest.approx(f("07_after_slm_h"))
    assert f("10_selected_plus1_order_v") + f("12_rejected_power_v") == pytest.approx(f("08_after_slm_v"))
    assert f("13_zero_order_total") + f("07_after_slm_h") + f("08_after_slm_v") == pytest.approx(1.0)
    assert f("20_useful_central_hexagon_power") + f("21_power_outside_useful_region") == pytest.approx(f("19_total_power_at_z60"))


def test_camera_shack_hartmann_and_stokes_roles_remain_separate() -> None:
    rows = measurement_responsibility_rows()
    instruments = {row["instrument"] for row in rows}

    assert {"camera", "shack_hartmann", "stokes_polarimetry"} <= instruments
    assert all(row["instrument"] for row in rows)
    assert all(not str(row["measurement"]).startswith("{") for row in rows)
    camera_roles = " ".join(row["role"] for row in rows if row["instrument"] == "camera").lower()
    shack_roles = " ".join(row["role"] for row in rows if row["instrument"] == "shack_hartmann").lower()
    stokes_roles = " ".join(row["role"] for row in rows if row["instrument"] == "stokes_polarimetry").lower()
    assert "centr" in camera_roles
    assert "wavefront" in shack_roles or "zernike" in shack_roles
    assert "qwp" in stokes_roles or "stokes" in stokes_roles


def test_closed_loop_outputs_do_not_expose_injected_truth_to_the_search() -> None:
    root = _ensure_outputs()
    results = _json_load(root / "08_closed_loop" / "mode2v_closed_loop_results.json")

    assert {row["case_id"] for row in results} == set(closed_loop_cases())
    for row in results:
        assert row["search_received_injected_truth"] is False
        assert "injected_truth_revealed_after" in row
        for iteration in row["iterations"]:
            assert all("truth" not in key and "injected" not in key for key in iteration)


def test_repaired_strict_hexagon_gate_rejects_x_shaped_and_triangular_fields() -> None:
    fix_root = _ensure_fix_outputs()
    rows = _csv_rows(fix_root / "hexagon_classifier_calibration.csv")
    by_id = {row["case_id"]: row for row in rows}

    assert by_id["synthetic_h4_fourfold_failure"]["strict_hexagon_eligible"] == "False"
    assert by_id["synthetic_h4_fourfold_failure"]["fourfold_x_veto"] == "True"
    assert by_id["old_triangular_mode1_failure_proxy"]["strict_hexagon_eligible"] == "False"


def test_old_invalid_compromise_cannot_be_revived_in_mode2v_outputs() -> None:
    root = _ensure_outputs()
    outcome = _json_load(root / "10_final_status" / "m2v_outcome_report.json")
    manifest = _json_load(root / "10_final_status" / "nathan_mode2v_manifest.json")

    assert outcome["canonical_operating_point"] == CANONICAL_OPERATING_POINT_ID
    assert outcome["secondary_operating_point"] == STRICT_COMPROMISE_ID
    assert OLD_BEST_COMPROMISE_ID in outcome["forbidden_note"]
    assert manifest["canonical_operating_point"] != OLD_BEST_COMPROMISE_ID
    assert manifest["secondary_operating_point"] != OLD_BEST_COMPROMISE_ID


def test_mask_package_distinguishes_preview_panel_simulation_and_lut_states() -> None:
    root = _ensure_outputs()
    readme = (root / "09_mask_package" / "README_MASK_PACKAGE.md").read_text(encoding="utf-8")
    metadata = _json_load(root / "09_mask_package" / "mode2v_slm_masks_metadata.json")

    assert "PANEL-space wrapped phase" in readme
    assert "Simulation-space vs panel-space" in readme
    assert "PREVIEW ONLY" in readme
    assert "LUT" in readme
    assert metadata["lut_applied"] is False
    assert metadata["hardware_ready_condition"].startswith("apply the per-panel measured LUT")


def test_no_microfabrication_or_sample_plane_success_is_claimed() -> None:
    root = _ensure_outputs()
    outcome = _json_load(root / "10_final_status" / "m2v_outcome_report.json")
    manifest = _json_load(root / "10_final_status" / "nathan_mode2v_manifest.json")
    master = Path("docs/81_nathan_mode2v_lab_ready_master_report.md").read_text(encoding="utf-8")

    assert outcome["microfabrication_sample_plane_claim"] is False
    assert manifest["microfabrication_sample_plane_claim"] is False
    assert "No microfabrication/sample-plane success is claimed" in master


def test_m2v_a_requires_complete_architecture_masks_4f_polarisation_and_loop_demo() -> None:
    root = _ensure_outputs()
    metadata = _json_load(root / "09_mask_package" / "mode2v_slm_masks_metadata.json")
    loop_results = _json_load(root / "08_closed_loop" / "mode2v_closed_loop_results.json")
    fourf = fourf_final_design(
        simulated_first_order_efficiency=0.9495,
        simulated_zero_order_leakage=0.0,
    )

    ok = mode2v_outcome(
        decision=architecture_decision(),
        masks_metadata=metadata,
        fourf_design=fourf,
        waveplates=waveplate_table(),
        loop_results=loop_results,
    )
    no_loop = mode2v_outcome(
        decision=architecture_decision(),
        masks_metadata=metadata,
        fourf_design=fourf,
        waveplates=waveplate_table(),
        loop_results=[],
    )

    assert ok["selected_outcome"] == "M2V-A"
    assert no_loop["selected_outcome"] == "M2V-B"
    assert ok["six_piece_segmented_optic_required"] is False


def test_exported_npy_masks_match_native_panel_shape_and_preview_status() -> None:
    root = _ensure_outputs()
    slm_h = np.load(root / "09_mask_package" / "mode2v_slmH_phase_rad.npy")
    slm_v = np.load(root / "09_mask_package" / "mode2v_slmV_phase_rad.npy")
    metadata = _json_load(root / "09_mask_package" / "mode2v_slm_masks_metadata.json")

    assert slm_h.shape == (1080, 1920)
    assert slm_v.shape == (1080, 1920)
    assert float(slm_h.min()) >= 0.0
    assert float(slm_v.max()) < 2.0 * np.pi
    assert metadata["hardware_ready"] is False


def test_component_table_contains_exact_lab_chain_and_no_segmented_optic() -> None:
    components = component_table()
    ids = [row["component_id"] for row in components]
    route_text = " ".join(row["physical_role"] for row in components).lower()

    for component_id in ["LASER", "HWP1", "PBS1", "SLM_H", "SLM_V", "PBS2", "IRIS", "QWP1", "AXICON", "CAM"]:
        assert component_id in ids
    assert "six-piece" not in route_text
    assert "segmented optic" not in route_text
