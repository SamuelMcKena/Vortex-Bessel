from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from vbb_study.calibration.io import dump_calibration_bundle, load_calibration_bundle
from vbb_study.calibration.schema import CalibrationBundle, canonical_calibration_template
from vbb_study.calibration.uncertainty import UncertaintyConfig, propagate_calibration_uncertainty
from vbb_study.calibration.validation import calibration_readiness_for_claim, validate_calibration_bundle
from vbb_study.digital_twin.phase2d_canonical_pipeline import (
    CanonicalOpticalRequest,
    run_canonical_optical_prediction,
)
from vbb_study.solver_policy import select_solver_for_claim


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "validation" / "phase2d"
FULL_SYNTHETIC = OUT / "synthetic_bundles" / "full_synthetic_validation.json"


def _request(case: str, claims: tuple[str, ...], bundle: Path | None = None, samples: int = 0) -> CanonicalOpticalRequest:
    return CanonicalOpticalRequest(
        beam_case=case,
        claim_types=claims,
        fidelity_mode="automatic_by_claim",
        calibration_bundle_path=None if bundle is None else str(bundle),
        include_interface=True,
        material_propagation_distance_m=10e-6,
        uncertainty_samples=samples,
    )


def _reference() -> dict[str, float]:
    return {
        "reference_wavelength_m": 1.029e-6,
        "reference_objective_NA": 0.45,
        "carrier_frequency_cpm": 6250.0,
        "feature_radius_m": 0.66885e-6,
        "reference_output_pixel_m": 0.2058e-6,
        "first_order_efficiency": 0.9493536222723651,
        "input_aperture_fraction": 0.9999543102655302,
        "objective_pupil_fraction": 0.7946576086635864,
        "peak_shape_factor_m_inv2": 1.5717e7,
    }


def test_01_g0_global_morphology_selects_scalar() -> None:
    decision = select_solver_for_claim("G0", "global_transverse_morphology", "automatic_by_claim")
    assert decision.selected_objective_solver == "scalar_fft"
    assert decision.scalar_allowed is True


def test_02_g0_longitudinal_selects_vector() -> None:
    decision = select_solver_for_claim("G0", "longitudinal_field", "automatic_by_claim")
    assert decision.selected_objective_solver == "vector_debye"
    assert decision.vector_required is True


@pytest.mark.parametrize("case", ["V1", "V3"])
def test_03_vortex_peak_location_selects_vector(case: str) -> None:
    assert select_solver_for_claim(case, "peak_location", "automatic_by_claim").selected_objective_solver == "vector_debye"


def test_04_h1_feature_radius_selects_vector() -> None:
    assert select_solver_for_claim("H1", "feature_radius", "automatic_by_claim").selected_objective_solver == "vector_debye"


def test_05_h1_fast_shape_screening_remains_available() -> None:
    decision = select_solver_for_claim("H1", "global_transverse_morphology", "fast_scalar_screening")
    assert decision.selected_objective_solver == "scalar_fft"


def test_06_forbidden_scalar_quantitative_h1_raises() -> None:
    with pytest.raises(ValueError, match="requires the quantitative vector reference"):
        select_solver_for_claim("H1", "edge_sharpness", "fast_scalar_screening")


def test_07_interface_component_claim_selects_vector_fresnel() -> None:
    decision = select_solver_for_claim("B0", "polarisation_component", "automatic_by_claim")
    assert decision.selected_interface_solver == "vector_spectral_fresnel"


def test_08_schema_version_validation() -> None:
    good = CalibrationBundle(canonical_calibration_template())
    assert validate_calibration_bundle(good).valid_schema is True
    bad_data = good.copy_data()
    bad_data["schema_version"] = "9.9"
    assert validate_calibration_bundle(CalibrationBundle(bad_data)).valid_schema is False


def test_09_null_values_remain_null() -> None:
    bundle = CalibrationBundle(canonical_calibration_template())
    assert bundle.data["laser"]["pulse_energy_J"]["value"] is None
    assert bundle.data["camera"]["object_plane_scale_m_per_pixel"]["value"] is None


def test_10_missing_fields_block_correct_claims() -> None:
    bundle = CalibrationBundle(canonical_calibration_template())
    dimensions = calibration_readiness_for_claim(bundle, "absolute_dimensions")
    fluence = calibration_readiness_for_claim(bundle, "absolute_fluence")
    assert "camera.object_plane_scale_m_per_pixel" in dimensions.missing_measurements
    assert "laser.pulse_energy_J" in fluence.missing_measurements


def test_11_physically_impossible_values_are_rejected() -> None:
    data = canonical_calibration_template()
    data["transmissions"]["slm1"]["value"] = 1.2
    report = validate_calibration_bundle(CalibrationBundle(data))
    assert not report.valid_schema
    assert any("transmissions.slm1" in item for item in report.physically_impossible_values)


def test_12_unit_consistency_is_enforced() -> None:
    data = canonical_calibration_template()
    data["laser"]["wavelength_m"]["unit"] = "nm"
    report = validate_calibration_bundle(CalibrationBundle(data))
    assert not report.valid_schema
    assert any("expected 'm'" in item for item in report.inconsistent_units)


def test_13_bundle_round_trip_serialisation() -> None:
    original = CalibrationBundle(canonical_calibration_template())
    path = ROOT / ".phase2d_round_trip_test.json"
    try:
        dump_calibration_bundle(original, path)
        assert load_calibration_bundle(path).data == original.data
    finally:
        path.unlink(missing_ok=True)


def test_14_uncertainty_seed_is_deterministic() -> None:
    bundle = load_calibration_bundle(FULL_SYNTHETIC)
    request = _request("H1", ("feature_radius",), FULL_SYNTHETIC, 128)
    config = UncertaintyConfig(samples=128, random_seed=21)
    first = propagate_calibration_uncertainty(request, bundle, config, reference_metrics=_reference())
    second = propagate_calibration_uncertainty(request, bundle, config, reference_metrics=_reference())
    assert first == second


def test_15_zero_input_uncertainty_propagates_zero() -> None:
    data = load_calibration_bundle(FULL_SYNTHETIC).copy_data()
    for section in data.values():
        if isinstance(section, dict):
            for value in section.values():
                if isinstance(value, dict) and value.get("uncertainty") is not None:
                    value["uncertainty"] = 0.0
    bundle = CalibrationBundle(data)
    request = _request("H1", ("feature_radius",), samples=64)
    result = propagate_calibration_uncertainty(
        request, bundle, UncertaintyConfig(samples=64), reference_metrics=_reference()
    )
    assert result["feature_or_ring_radius_m"]["standard_uncertainty"] == pytest.approx(0.0)


def test_16_uncertainty_intervals_contain_nominal() -> None:
    bundle = load_calibration_bundle(FULL_SYNTHETIC)
    result = propagate_calibration_uncertainty(
        _request("H1", ("feature_radius",), FULL_SYNTHETIC, 128),
        bundle,
        UncertaintyConfig(samples=128),
        reference_metrics=_reference(),
    )
    for row in result.values():
        if row["uncertainty_status"] == "available_from_supplied_uncertainty":
            assert row["lower_95"] <= row["nominal"] <= row["upper_95"]


def test_17_missing_calibration_returns_unavailable() -> None:
    bundle = CalibrationBundle(canonical_calibration_template())
    result = propagate_calibration_uncertainty(
        _request("H1", ("absolute_fluence",)),
        bundle,
        UncertaintyConfig(samples=32),
        reference_metrics=_reference(),
    )
    assert result["peak_fluence_J_cm2"]["uncertainty_status"] == "unavailable_missing_calibration"


def test_18_energy_ledger_uses_exactly_one_first_order_factor() -> None:
    result = run_canonical_optical_prediction(_request("H1", ("global_transverse_morphology",)))
    assert result.provenance["selected_first_order_factor_count"] == 1
    assert result.provenance["configured_first_order_factor_reapplied"] is False


def test_19_no_phase2a_energy_output_changed() -> None:
    manifest = json.loads((OUT / "phase2d_manifest.json").read_text(encoding="utf-8"))
    before = manifest["upstream_hashes_before"]
    after = manifest["upstream_hashes_after"]
    for path in (
        "outputs/validation/phase2a/canonical_case_summary.csv",
        "outputs/validation/phase2a/canonical_power_ledgers.csv",
    ):
        assert before[path] == after[path]


def test_20_no_phase2c_solver_output_changed() -> None:
    manifest = json.loads((OUT / "phase2d_manifest.json").read_text(encoding="utf-8"))
    before = manifest["upstream_hashes_before"]
    after = manifest["upstream_hashes_after"]
    for path in (
        "outputs/validation/phase2c/phase2c_objective_benchmark.csv",
        "outputs/validation/phase2c/phase2c_interface_benchmark.csv",
        "outputs/validation/phase2c/phase2c_outcome_report.json",
    ):
        assert before[path] == after[path]
    assert manifest["phase2c_solver_outputs_recomputed"] is False


def test_21_synthetic_bundle_is_visibly_marked() -> None:
    bundle = load_calibration_bundle(FULL_SYNTHETIC)
    assert bundle.is_synthetic
    assert bundle.data_classification == "synthetic_not_experimental"


def test_22_synthetic_bundle_cannot_claim_real_calibration_maturity() -> None:
    result = run_canonical_optical_prediction(
        _request("H1", ("absolute_dimensions", "absolute_fluence"), FULL_SYNTHETIC, 16)
    )
    assert set(result.calibration_status["claim_maturity"].values()) == {"calibration_ready_prediction"}
    assert result.calibration_status["absolute_dimensions_unlocked"] is False
    assert result.calibration_status["absolute_fluence_unlocked"] is False
    assert result.calibration_status["experimentally_validated"] is False


def test_23_h1_automatic_route_uses_debye() -> None:
    result = run_canonical_optical_prediction(_request("H1", ("feature_radius",)))
    assert result.focal_field.solver == "vector_debye"


@pytest.mark.parametrize("case", ["G0", "B0"])
def test_24_g0_b0_scalar_screening_remains_valid(case: str) -> None:
    result = run_canonical_optical_prediction(_request(case, ("global_transverse_morphology",)))
    assert result.focal_field.solver == "scalar_fft"
    assert result.metrics["morphology_classification"] == "morphology_equivalent"


@pytest.mark.parametrize(("case", "charge"), [("V1", 1.0), ("V3", 3.0)])
def test_25_vortex_winding_remains_correct(case: str, charge: float) -> None:
    result = run_canonical_optical_prediction(_request(case, ("peak_location",)))
    assert result.metrics["measured_winding"] == pytest.approx(charge, abs=1e-12)


def test_26_h1_strict_hexagon_remains_true() -> None:
    result = run_canonical_optical_prediction(_request("H1", ("edge_sharpness",)))
    assert result.metrics["strict_hexagon"] is True


def test_27_phase2b_transverse_surface_convention_unchanged() -> None:
    source = (ROOT / "vbb_study/digital_twin/phase2b_figures.py").read_text(encoding="utf-8")
    assert "Plot a pure transverse 3D intensity surface" in source
    assert 'zlabel="normalised intensity"' in source
    assert 'xlabel="x (mm)"' in source and 'ylabel="y (mm)"' in source


def test_28_upstream_hashes_remain_unchanged() -> None:
    manifest = json.loads((OUT / "phase2d_manifest.json").read_text(encoding="utf-8"))
    assert manifest["upstream_hashes_unchanged"] is True
    assert manifest["upstream_hashes_before"] == manifest["upstream_hashes_after"]


def test_29_phase1_contracts_remain_unchanged() -> None:
    manifest = json.loads((OUT / "phase2d_manifest.json").read_text(encoding="utf-8"))
    for path in (
        "outputs/validation/phase1_critical_repairs/phase1_repair_summary.json",
        "outputs/validation/phase1_reconciliation/phase1r_regeneration_manifest.json",
    ):
        assert manifest["upstream_hashes_before"][path] == manifest["upstream_hashes_after"][path]


def test_30_nathan_mode2_regression_contract_remains_unchanged() -> None:
    phase2a = json.loads(
        (ROOT / "outputs/validation/phase2a/phase2a_outcome_report.json").read_text(encoding="utf-8")
    )
    phase2d = json.loads((OUT / "phase2d_manifest.json").read_text(encoding="utf-8"))
    assert phase2a["nathan_outputs_changed"] is False
    assert phase2d["accepted_upstream_outputs_overwritten"] is False


def test_31_required_outputs_templates_and_docs_exist() -> None:
    required = [
        OUT / "solver_claim_policy.csv",
        OUT / "calibration_dependency_graph.csv",
        OUT / "calibration_readiness.json",
        OUT / "canonical_case_summary.csv",
        OUT / "uncertainty_validation.csv",
        OUT / "claim_maturity_registry.csv",
        OUT / "phase2d_claim_registry.csv",
        OUT / "phase2d_outcome_report.json",
        OUT / "phase2d_manifest.json",
        ROOT / "calibration/templates/canonical_lab_calibration_template.json",
        ROOT / "docs/93_phase2d_experimental_calibration_bridge.md",
    ]
    assert all(path.is_file() for path in required)


def test_32_outcome_is_honest_calibration_limited_b() -> None:
    report = json.loads((OUT / "phase2d_outcome_report.json").read_text(encoding="utf-8"))
    assert report["outcome"] == "PHASE2D-B"
    assert report["absolute_dimensions_unlocked"] is False
    assert report["absolute_fluence_unlocked"] is False
    assert report["experimentally_validated_prediction_present"] is False


def test_33_vortex_ring_radius_names_its_reference_solver() -> None:
    result = run_canonical_optical_prediction(_request("V1", ("ring_radius",)))
    assert result.metrics["metric_reference_solver"] == "scalar_fft"
    decision = result.solver_decisions[0]
    assert "scalar or vector reference" in decision.reason


def test_34_h1_edge_uncertainty_is_not_invented() -> None:
    result = run_canonical_optical_prediction(
        _request("H1", ("edge_sharpness",), FULL_SYNTHETIC, 32)
    )
    edge = result.uncertainty_summary["H1_edge_sharpness_mm_inv"]
    assert edge["uncertainty_status"] == "unavailable_numerical_stability_not_demonstrated"
    assert edge["nominal"] is None
