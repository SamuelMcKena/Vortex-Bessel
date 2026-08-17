from __future__ import annotations

import numpy as np
import pytest

from vbb_study.digital_twin import (
    MODE2N_PRE_AXICON_OVERLAP_PASS,
    mode2n_outcome_report,
    mode2n_scope_manifest,
    mode2n_source_target,
    run_mode2n_source_replication,
    source_parity_grid,
)

# MODE 2N is a source-scale study: the V0 classifier itself needs grid_n >= 384
# (docs/53); tests use that minimum with a short z-stack for speed.
TEST_GRID_N = 384
TEST_Z_PLANES = 9


@pytest.fixture(scope="module")
def study():
    return run_mode2n_source_replication(grid_n=TEST_GRID_N, z_planes=TEST_Z_PLANES)


def test_source_scale_grid_matches_nathan_axis_sampled_convention() -> None:
    data = mode2n_source_target(grid_n=256, z_planes=3)
    grid = data["grid"]
    reference = source_parity_grid(data["config"])

    assert data["axis_sampled"] is True
    assert np.min(np.asarray(grid["R"], dtype=float)) == pytest.approx(0.0, abs=1e-15)
    assert grid["N"] == reference["N"]
    assert grid["dx"] == pytest.approx(reference["dx"])
    assert np.array_equal(np.asarray(grid["x"]), np.asarray(reference["x"]))
    assert data["config"].window_m == pytest.approx(10.0e-3)
    assert data["config"].beam_radius_m == pytest.approx(2.0e-3)


def test_patterned_hwp_pre_axicon_overlap_matches_v0_target(study) -> None:
    route = study["patterned_hwp"]

    assert float(route.pre_axicon_metrics["complex_vector_overlap"]) >= MODE2N_PRE_AXICON_OVERLAP_PASS


def test_dual_slm_qwp_pre_axicon_overlap_matches_v0_target(study) -> None:
    route = study["dual_slm_qwp"]

    assert float(route.pre_axicon_metrics["complex_vector_overlap"]) >= MODE2N_PRE_AXICON_OVERLAP_PASS


def test_patterned_hwp_propagated_result_matches_v0_z60_intensity(study) -> None:
    route = study["patterned_hwp"]

    assert float(route.v0_comparison["z60_full_field_correlation"]) >= 0.999
    assert route.reference_z_m == pytest.approx(60.0e-3)
    assert route.passes_v0_match is True


def test_dual_slm_qwp_propagated_result_matches_v0_z60_intensity(study) -> None:
    route = study["dual_slm_qwp"]

    assert float(route.v0_comparison["z60_full_field_correlation"]) >= 0.999
    assert route.passes_v0_match is True


def test_carrier_4f_route_power_ledger_is_consistent(study) -> None:
    report = study["dual_slm_4f"].slm_4f_report

    assert report["power_ledger_relative_error"] < 1e-9
    assert 0.0 < report["first_order_efficiency"] <= 1.0
    assert report["signal_power"] + report["blocked_power"] == pytest.approx(report["incident_power"], rel=1e-9)
    assert report["blocked_power_fraction"] == pytest.approx(1.0 - report["first_order_efficiency"], abs=1e-9)


def test_carrier_4f_zero_order_leakage_is_reported(study) -> None:
    report = study["dual_slm_4f"].slm_4f_report

    assert "zero_order_content_before_iris" in report
    assert "zero_order_leakage_after_iris" in report
    assert np.isfinite(report["zero_order_content_before_iris"])
    assert np.isfinite(report["zero_order_leakage_after_iris"])
    # The iris disk is disjoint from the zero-order disk at the default geometry.
    assert report["zero_order_leakage_after_iris"] <= report["zero_order_content_before_iris"] + 1e-15


def test_source_scale_outcome_makes_no_microfabrication_claim(study) -> None:
    outcome = study["outcome"]
    manifest = mode2n_scope_manifest(outcome)

    assert outcome["suggested_outcome"] in outcome["allowed_outcomes"]
    assert outcome["inherited_objective_sample_geometry_used"] is False
    assert outcome["microfabrication_sample_plane_claim"] is False
    assert manifest["inherited_objective_sample_geometry"] is False
    assert manifest["micro_scale_sample_plane_simulated"] is False
    assert manifest["microfabrication_sample_plane_claim"] is False
    assert "sample" not in str(outcome["outcome_statement"]).lower() or "sample-plane" not in str(outcome["outcome_statement"]).lower()


def test_route_success_requires_propagated_match_not_pre_axicon_overlap(study) -> None:
    outcome = mode2n_outcome_report(
        v0=study["v0"],
        patterned_hwp=study["patterned_hwp"],
        dual_slm_qwp=study["dual_slm_qwp"],
        dual_slm_4f=study["dual_slm_4f"],
        data=study["data"],
    )

    assert outcome["suggested_outcome"] == study["outcome"]["suggested_outcome"]
    for key in ("route_patterned_hwp", "route_dual_slm_qwp", "route_dual_slm_4f"):
        summary = outcome[key]
        if summary["passes_v0_match"]:
            assert summary["z60_full_field_correlation"] >= 0.90
            assert summary["symmetry_class"] == "visual_hexagonal_field"
