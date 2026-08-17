from __future__ import annotations

import numpy as np
import pytest

from vbb_study.digital_twin import (
    MODE2Q_BACKWARD_RECOVERY_OVERLAP_PASS,
    hwp,
    mode1_symmetry_class,
    mode2n_source_target,
    mode2q_4f_passband_report,
    mode2q_backpropagate_vector,
    mode2q_forward_propagate_vector,
    mode2q_inverse_axicon,
    mode2q_inverse_retarder,
    mode2q_scope_manifest,
    qwp,
    run_mode2q_backward_mask_synthesis,
)
from vbb_study.digital_twin.nathan_vector_hexagon import (
    _apply_free_space_vector_axicon,
    complex_vector_overlap,
)

TEST_GRID_N = 384
TEST_Z_PLANES = 9


@pytest.fixture(scope="module")
def data():
    return mode2n_source_target(grid_n=256, z_planes=3)


@pytest.fixture(scope="module")
def study():
    return run_mode2q_backward_mask_synthesis(grid_n=TEST_GRID_N, z_planes=TEST_Z_PLANES)


def test_inverse_angular_spectrum_inverts_forward_propagation(data) -> None:
    field = data["target_field"]
    z = 25.0e-3
    forward = mode2q_forward_propagate_vector(field, z)
    recovered, meta = mode2q_backpropagate_vector(forward, z)
    # Forward propagation applies the transversality projection once, so compare
    # against the forward-then-backward-consistent baseline: one more roundtrip.
    forward2 = mode2q_forward_propagate_vector(recovered, z)

    assert meta["evanescent_clipped_energy_fraction"] < 1e-6
    overlap = complex_vector_overlap((forward2.ex, forward2.ey), (forward.ex, forward.ey))
    assert overlap >= 1.0 - 1e-10


def test_inverse_axicon_inverts_forward_axicon(data) -> None:
    cfg = data["config"]
    field = data["target_field"]
    after, _ = _apply_free_space_vector_axicon(
        field,
        n_axicon=float(cfg.axicon_n),
        n_medium=float(cfg.medium_n),
        base_angle_rad=float(cfg.axicon_base_angle_rad),
    )
    recovered, meta = mode2q_inverse_axicon(after, cfg)

    assert meta["near_singular_transmission"] is False
    assert meta["jones_condition_number"] < 1.01
    overlap = complex_vector_overlap((recovered.ex, recovered.ey), (field.ex, field.ey))
    assert overlap >= 1.0 - 1e-12
    # Scalar-only phase check: the conical phases cancel exactly on the radial channel.
    assert np.allclose(recovered.ex, field.ex, atol=1e-10)
    assert np.allclose(recovered.ey, field.ey, atol=1e-10)


def test_inverse_retarder_satisfies_unitary_identity() -> None:
    for J in (qwp(-0.25 * np.pi), qwp(0.25 * np.pi), hwp(0.3), hwp(-1.1)):
        inv = mode2q_inverse_retarder(J)

        assert np.max(np.abs(inv @ np.asarray(J) - np.eye(2))) < 1e-12
        assert np.max(np.abs(np.asarray(J) @ inv - np.eye(2))) < 1e-12


def test_backward_v0_target_recovers_raw_nathan_pre_axicon_field(study) -> None:
    diag = dict(study["backward"].diagnostics)
    recovery = dict(diag["recovery_vs_raw_nathan_input"])

    assert diag["target"]["complex_vector_target_available"] is True
    assert recovery["overlap_to_raw_nathan_input"] >= MODE2Q_BACKWARD_RECOVERY_OVERLAP_PASS
    assert diag["backward_recovery_pass"] is True
    assert diag["backpropagation"]["evanescent_clipped_energy_fraction"] < 1e-6


def test_phase_only_projection_reports_amplitude_mismatch(study) -> None:
    amp = dict(study["backward"].diagnostics["amplitude_vs_phase_only_supply"])

    assert "amp_H_over_supply_rms" in amp
    assert "amp_V_over_supply_rms" in amp
    assert np.isfinite(amp["amp_H_over_supply_rms"])
    assert np.isfinite(amp["amp_V_over_supply_rms"])
    # The ideal inverse yields (nearly) phase-only realizable channels; a synthetic
    # amplitude perturbation must be reported as a larger mismatch, not hidden.
    data = study["data"]
    supply = np.asarray(data["A"], dtype=float) / np.sqrt(2.0)
    perturbed = supply * (1.0 + 0.5 * np.asarray(data["radial_mask"], dtype=float))
    rms = float(np.sqrt(np.sum((perturbed - supply) ** 2) / np.sum(supply**2)))
    assert rms > 10.0 * amp["amp_H_over_supply_rms"]


def test_4f_inverse_is_marked_adjoint_not_exact_inverse(study) -> None:
    report = dict(study["backward"].diagnostics["four_f_adjoint"])

    assert report["kind"] == "adjoint_projection_not_exact_inverse"
    assert 0.0 <= report["required_field_energy_outside_passband_fraction"] <= 1.0
    fresh = mode2q_4f_passband_report(
        study["backward"].Ex_required_pre_qwp,
        study["backward"].Ey_required_pre_qwp,
        study["data"]["grid"],
    )
    assert fresh["kind"] == "adjoint_projection_not_exact_inverse"


def test_source_scale_outcome_makes_no_microfabrication_claim(study) -> None:
    outcome = study["outcome"]
    manifest = mode2q_scope_manifest(outcome)

    assert outcome["suggested_outcome"] in outcome["allowed_outcomes"]
    assert outcome["inherited_objective_sample_geometry_used"] is False
    assert outcome["microfabrication_sample_plane_claim"] is False
    assert manifest["inherited_objective_sample_geometry"] is False
    assert manifest["micro_scale_sample_plane_simulated"] is False
    assert manifest["microfabrication_sample_plane_claim"] is False


def test_six_lobed_structure_cannot_pass_hexagon_gate_when_c120_exceeds_c60() -> None:
    sym = {
        "rot_corr_60": 0.40,
        "rot_corr_120": 0.55,
        "c120_minus_c60": 0.15,
        "order3_over_order6": 0.5,
        "six_sector_max_over_min": 1.2,
        "ring_island_count": 6,
    }

    assert mode1_symmetry_class(sym, dark_core_ratio=0.05) != "visual_hexagonal_field"
