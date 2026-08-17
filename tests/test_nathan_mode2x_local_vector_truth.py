from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vbb_study.digital_twin.nathan_local_vector_truth import (
    MODE2X_SECTOR_CONVENTION,
    build_mode2x_route_results,
    build_radial_azimuthal_sector_masks,
    cartesian_to_local_cylindrical,
    evaluate_local_vector_truth,
    write_mode2x_local_vector_truth,
)
from vbb_study.digital_twin.nathan_mode2u2_fix_strict_hexagon_optimisation import (
    OLD_BEST_COMPROMISE_ID,
)
from vbb_study.digital_twin.nathan_mode2u3_hardware_closure import (
    CANONICAL_OPERATING_POINT_ID,
    STRICT_COMPROMISE_ID,
)
from vbb_study.digital_twin.nathan_vector_hexagon import nathan_alpha_map


def _analytic_target(n: int = 65) -> dict[str, np.ndarray]:
    x = np.linspace(-2.0e-3, 2.0e-3, n)
    X, Y = np.meshgrid(x, x, indexing="xy")
    theta = np.arctan2(Y, X)
    alpha, radial = nathan_alpha_map(theta, sector_rotation=0.0)
    amplitude = np.exp(-(X**2 + Y**2) / (1.0e-3) ** 2)
    return {
        "x": x,
        "X": X,
        "Y": Y,
        "theta": theta,
        "alpha": alpha,
        "radial": radial,
        "A": amplitude,
        "Ex": amplitude * np.cos(alpha),
        "Ey": amplitude * np.sin(alpha),
    }


@pytest.fixture(scope="module")
def target() -> dict[str, np.ndarray]:
    return _analytic_target()


@pytest.fixture(scope="module")
def route_results():
    return build_mode2x_route_results(grid_n=256, z_planes=3)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("mode2x")
    return write_mode2x_local_vector_truth(
        root / "outputs",
        document_path=root / "docs/84_nathan_mode2x_local_vector_truth.md",
        grid_n=192,
        z_planes=3,
    )


def test_cartesian_to_cylindrical_preserves_transverse_power(target):
    basis = cartesian_to_local_cylindrical(target["Ex"], target["Ey"], target["x"], target["x"])
    cartesian = np.abs(target["Ex"]) ** 2 + np.abs(target["Ey"]) ** 2
    assert np.allclose(cartesian, basis.transverse_power, rtol=5e-13, atol=1e-14)


def test_pure_radial_field_maps_to_er(target):
    ex = target["A"] * np.cos(target["theta"])
    ey = target["A"] * np.sin(target["theta"])
    basis = cartesian_to_local_cylindrical(ex, ey, target["x"], target["x"])
    assert np.max(np.abs(basis.etheta[basis.valid_mask])) < 1e-14
    assert np.allclose(basis.er[basis.valid_mask], target["A"][basis.valid_mask], atol=1e-14)


def test_pure_azimuthal_field_maps_to_etheta(target):
    ex = -target["A"] * np.sin(target["theta"])
    ey = target["A"] * np.cos(target["theta"])
    basis = cartesian_to_local_cylindrical(ex, ey, target["x"], target["x"])
    assert np.max(np.abs(basis.er[basis.valid_mask])) < 1e-14
    assert np.allclose(basis.etheta[basis.valid_mask], target["A"][basis.valid_mask], atol=1e-14)


def _target_truth(target):
    return evaluate_local_vector_truth(
        "analytic",
        target["Ex"],
        target["Ey"],
        target["x"],
        target["x"],
        target["alpha"],
        gate_class="ideal",
    )


def test_authoritative_target_radial_purity_is_unity(target):
    assert _target_truth(target).metrics.radial_purity == pytest.approx(1.0, abs=1e-13)


def test_authoritative_target_azimuthal_purity_is_unity(target):
    assert _target_truth(target).metrics.azimuthal_purity == pytest.approx(1.0, abs=1e-13)


def test_ideal_target_s3_is_zero(target):
    result = _target_truth(target)
    assert result.metrics.s3_rms_fraction < 1e-14


def test_ideal_target_angle_error_is_zero_modulo_pi(target):
    result = _target_truth(target)
    assert result.metrics.local_angle_rms_rad < 1e-14


def test_sector_averaged_constant_arrows_fail_local_truth(target):
    sector = np.mod(np.floor(np.mod(target["theta"], 2.0 * np.pi) / (np.pi / 3.0)).astype(int), 6)
    centres = (sector + 0.5) * np.pi / 3.0
    _, centre_radial = build_radial_azimuthal_sector_masks(centres, sector_rotation_rad=0.0)
    constant_alpha = centres + np.where(centre_radial, 0.0, 0.5 * np.pi)
    ex = target["A"] * np.cos(constant_alpha)
    ey = target["A"] * np.sin(constant_alpha)
    result = evaluate_local_vector_truth(
        "constant_arrow_schematic",
        ex,
        ey,
        target["x"],
        target["x"],
        target["alpha"],
        gate_class="ideal",
    )
    assert not result.metrics.passed_full_vector_truth_gate
    assert result.metrics.local_angle_rms_rad > 0.1


def test_sector_masks_match_authoritative_v0_convention(target):
    radial, azimuthal = build_radial_azimuthal_sector_masks(
        target["theta"], sector_rotation_rad=0.0, existing_convention=MODE2X_SECTOR_CONVENTION
    )
    _, expected = nathan_alpha_map(target["theta"], sector_rotation=0.0)
    assert np.array_equal(radial, expected)
    assert np.array_equal(azimuthal, ~expected)


def test_sector_boundary_guard_is_excluded_consistently():
    theta = np.asarray([0.0, np.pi / 3.0, 2.0 * np.pi / 3.0, np.pi / 6.0])
    radial, azimuthal = build_radial_azimuthal_sector_masks(
        theta, sector_rotation_rad=0.0, angular_guard_band_rad=1e-5
    )
    assert not np.any((radial | azimuthal)[:3])
    assert bool((radial | azimuthal)[3])


def test_centre_singularity_policy_is_explicit(target):
    basis = cartesian_to_local_cylindrical(target["Ex"], target["Ey"], target["x"], target["x"])
    centre = target["x"].size // 2
    assert not bool(basis.valid_mask[centre, centre])
    result = _target_truth(target)
    assert result.metrics.centre_policy == "exclude_singular_neighbourhood"
    assert result.metrics.excluded_low_intensity_fraction > 0.0


def test_ideal_patterned_hwp_route_passes_ideal_truth_gate(route_results):
    result = next(r for r in route_results if r.route_id == "ideal_patterned_hwp")
    assert result.metadata["gate_class"] == "ideal"
    assert result.metrics.passed_full_vector_truth_gate


def test_ideal_abstract_dual_slm_route_passes_ideal_truth_gate(route_results):
    result = next(r for r in route_results if r.route_id == "ideal_abstract_dual_slm_qwp")
    assert result.metrics.passed_full_vector_truth_gate


def test_ideal_sequential_dual_slm_route_passes_ideal_truth_gate(route_results):
    result = next(r for r in route_results if r.route_id == "ideal_sequential_dual_slm")
    assert result.metrics.passed_full_vector_truth_gate


def test_realistic_sequential_route_uses_realistic_gate(route_results):
    result = next(r for r in route_results if r.route_id == "realistic_sequential_carrier_common_4f")
    assert result.metadata["gate_class"] == "realistic"
    assert "gate" in result.metadata


def test_bad_realism_is_not_silently_perfect(route_results):
    result = next(r for r in route_results if r.route_id == "m2s_combined_bad_lab")
    assert result.metrics.radial_purity < 0.999
    assert result.metrics.azimuthal_purity < 0.999
    assert not result.metrics.passed_full_vector_truth_gate


def test_final_hexagon_and_pre_axicon_truth_are_separate(route_results):
    canonical = next(r for r in route_results if r.route_id == CANONICAL_OPERATING_POINT_ID)
    assert "final_strict_hexagon_pass" in canonical.metadata
    assert isinstance(canonical.metrics.passed_full_vector_truth_gate, bool)
    assert isinstance(canonical.metadata["final_strict_hexagon_pass"], bool)


def test_forbidden_old_candidate_cannot_become_canonical(route_results):
    ids = {result.route_id for result in route_results}
    assert OLD_BEST_COMPROMISE_ID not in ids
    assert CANONICAL_OPERATING_POINT_ID in ids
    assert STRICT_COMPROMISE_ID in ids


def test_no_parallel_split_architecture_is_introduced(generated):
    scope = json.loads(Path(generated["scope_manifest"]).read_text(encoding="utf-8"))
    assert scope["superseded_parallel_arm_architecture_reintroduced"] is False
    assert "sequential collinear beam" in scope["accepted_architecture"]


def test_both_required_meeting_figures_are_generated(generated):
    root = Path(generated["output_root"])
    for stem in ("sector_averaged_polarisation_schematic", "true_local_polarisation_field"):
        assert (root / f"01_figures/{stem}.png").is_file()
        assert (root / f"01_figures/{stem}.pdf").is_file()


def test_summary_and_outcome_outputs_are_generated(generated):
    root = Path(generated["output_root"])
    assert (root / "local_vector_truth_summary.csv").is_file()
    assert (root / "local_vector_truth_summary.json").is_file()
    assert Path(generated["outcome_report"]).is_file()


def test_no_microfabrication_sample_plane_success_is_claimed(generated):
    scope = json.loads(Path(generated["scope_manifest"]).read_text(encoding="utf-8"))
    report = json.loads(Path(generated["outcome_report"]).read_text(encoding="utf-8"))
    document = Path(generated["document_path"]).read_text(encoding="utf-8").lower()
    assert scope["microfabrication_sample_plane_success_claim"] is False
    assert report["no_microfabrication_sample_plane_success_claim"] is True
    assert "no microfabrication/sample-plane success claim" in document
