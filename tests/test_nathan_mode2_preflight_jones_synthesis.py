from __future__ import annotations

import numpy as np

from vbb_study.digital_twin import (
    MODE2P_ACCEPTANCE_OVERLAP,
    NathanSourceParityConfig,
    complex_vector_overlap,
    jones_stokes_rms,
    mode2p_centre_treatment_report,
    mode2p_outcome_report,
    mode2p_target_arrays,
    nathan_alpha,
    nathan_alpha_map,
    route_dual_slm_circular_ideal,
    route_dual_slm_linear_then_qwp_ideal,
    route_patterned_hwp_ideal,
    run_mode2p_jones_synthesis,
    source_parity_grid,
    synthesize_from_circular_components,
)


def test_nathan_alpha_map_matches_v0_sector_convention() -> None:
    centres = (np.arange(6, dtype=float) + 0.5) * np.pi / 3.0
    alpha, radial_mask = nathan_alpha_map(centres)

    assert radial_mask.tolist() == [False, True, False, True, False, True]
    assert np.allclose(alpha[radial_mask], centres[radial_mask])
    assert np.allclose(alpha[~radial_mask], centres[~radial_mask] + 0.5 * np.pi)

    grid = {"PHI": centres.reshape(1, -1), "R": np.ones((1, 6)), "X": np.cos(centres).reshape(1, -1), "Y": np.sin(centres).reshape(1, -1)}
    assert np.allclose(nathan_alpha(grid).reshape(-1), alpha)


def test_continuous_patterned_hwp_overlap_to_target_is_unity() -> None:
    data = mode2p_target_arrays()
    result = route_patterned_hwp_ideal(data["A"], data["alpha"], data["target"], mask=data["metric_mask"])

    assert result.overlap_to_target >= MODE2P_ACCEPTANCE_OVERLAP
    assert result.rms_error < 1.0e-12


def test_dual_circular_component_identity_overlap_to_target_is_unity() -> None:
    data = mode2p_target_arrays()
    result = route_dual_slm_circular_ideal(data["A"], data["alpha"], data["target"], mask=data["metric_mask"])

    assert result.overlap_to_target >= MODE2P_ACCEPTANCE_OVERLAP
    assert result.metadata["selected_sign"] == 1
    assert result.metadata["selected_piston_rad"] == 0.0


def test_dual_slm_linear_plus_qwp_sweep_finds_unity_overlap() -> None:
    data = mode2p_target_arrays()
    result = route_dual_slm_linear_then_qwp_ideal(data["A"], data["alpha"], data["target"], mask=data["metric_mask"])

    assert result.overlap_to_target >= MODE2P_ACCEPTANCE_OVERLAP
    assert result.metadata["selected_h_phase_sign"] == 1
    assert result.metadata["selected_v_phase_sign"] == -1
    assert np.isclose(result.metadata["selected_v_piston_rad"], 0.5 * np.pi)
    assert np.isclose(result.metadata["selected_qwp_angle_rad"], -0.25 * np.pi)


def test_wrong_sign_and_piston_controls_score_lower() -> None:
    data = mode2p_target_arrays()
    best = route_dual_slm_circular_ideal(data["A"], data["alpha"], data["target"], mask=data["metric_mask"])
    wrong_sign = synthesize_from_circular_components(data["A"], data["alpha"], sign=-1, piston=0.0)
    wrong_piston = synthesize_from_circular_components(data["A"], data["alpha"], sign=1, piston=np.pi)

    wrong_sign_overlap = complex_vector_overlap((wrong_sign[0], wrong_sign[1]), data["target"], data["metric_mask"])
    wrong_piston_overlap = complex_vector_overlap((wrong_piston[0], wrong_piston[1]), data["target"], data["metric_mask"])
    assert wrong_sign_overlap < best.overlap_to_target
    assert wrong_piston_overlap < 0.25


def test_stokes_maps_match_target_for_accepted_routes() -> None:
    data = mode2p_target_arrays()
    hwp = route_patterned_hwp_ideal(data["A"], data["alpha"], data["target"], mask=data["metric_mask"])
    qwp = route_dual_slm_linear_then_qwp_ideal(data["A"], data["alpha"], data["target"], mask=data["metric_mask"])

    assert jones_stokes_rms((hwp.Ex, hwp.Ey), data["target"], data["metric_mask"]) < 1.0e-12
    assert jones_stokes_rms((qwp.Ex, qwp.Ey), data["target"], data["metric_mask"]) < 1.0e-12


def test_centre_treatment_does_not_create_fake_pass_or_fail() -> None:
    grid = source_parity_grid(NathanSourceParityConfig(grid_n=64))
    data = mode2p_target_arrays(grid=grid)
    result = route_patterned_hwp_ideal(data["A"], data["alpha"], data["target"], mask=data["metric_mask"])
    centre = mode2p_centre_treatment_report(result, data["target"], data["grid"], metric_mask=data["metric_mask"])

    assert centre["axis_sampled"] is True
    assert centre["overlap_with_centre_policy"] >= MODE2P_ACCEPTANCE_OVERLAP
    assert centre["overlap_excluding_centre_pixels"] >= MODE2P_ACCEPTANCE_OVERLAP


def test_m2p_a_does_not_open_mode2a_2b_while_m1c_c_is_active() -> None:
    report = run_mode2p_jones_synthesis(mode1c_outcome="M1C-C")
    outcome = mode2p_outcome_report(
        report["patterned_hwp"],
        report["dual_slm_qwp"],
        circular_identity=report["circular_identity"],
        mode1c_outcome="M1C-C",
    )

    assert outcome["suggested_outcome"] == "M2P-A"
    assert outcome["mode2a_2b_realisation_allowed"] is False
    assert outcome["mode2a_2b_gate"] == "blocked_by_M1C-C"
