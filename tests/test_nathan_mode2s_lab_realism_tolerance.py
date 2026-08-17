from __future__ import annotations

import numpy as np
import pytest

from vbb_study.digital_twin import (
    Mode2SPerturbation,
    mode1_symmetry_class,
    mode2n_source_target,
    mode2s_apply_4f,
    mode2s_quantise_phase,
    mode2s_scope_manifest,
    mode2s_slm_aperture_fit_report,
    mode2s_tolerance_from_sweep,
    run_mode2n_dual_slm_4f_route,
    run_mode2n_v0_reference,
    run_mode2q_backward_initialisation,
    run_mode2s_degraded_forward,
    run_mode2s_precompensation,
)

TEST_GRID_N = 384
TEST_Z_PLANES = 9


@pytest.fixture(scope="module")
def bench():
    data = mode2n_source_target(grid_n=TEST_GRID_N, z_planes=TEST_Z_PLANES)
    v0 = run_mode2n_v0_reference(data)
    backward = run_mode2q_backward_initialisation(data)
    return data, v0, backward


def test_clean_baseline_reproduces_m2n_carrier_4f_metrics(bench) -> None:
    data, v0, backward = bench
    reference = run_mode2n_dual_slm_4f_route(data, v0)
    clean = run_mode2s_degraded_forward(data, v0, backward, Mode2SPerturbation(label="clean"))

    ref_corr = float(reference.v0_comparison["z60_full_field_correlation"])
    assert float(clean["comparison"]["z60_full_field_correlation"]) == pytest.approx(ref_corr, abs=1e-3)
    assert float(clean["iris"]["first_order_efficiency"]) == pytest.approx(
        float(reference.slm_4f_report["first_order_efficiency"]), abs=1e-3
    )
    assert clean["passes"] is True
    assert clean["failure_mode"] == "none"


def test_slm_pixel_grid_and_aperture_fit_is_reported_not_assumed(bench) -> None:
    data, _, _ = bench
    fit = mode2s_slm_aperture_fit_report(data)

    # The 10 mm source window exceeds the 8.64 mm SLM short axis: reported, not hidden.
    assert fit["window_fits_vertically"] is False
    assert fit["window_fits_horizontally"] is True
    assert fit["largest_valid_square_window_m"] == pytest.approx(8.64e-3)
    assert 0.0 <= fit["beam_power_clipped_by_active_area_fraction"] < 1e-3
    assert fit["pixelation_resolvable_on_this_grid"] is False
    assert fit["carrier_period_in_slm_pixels"] == pytest.approx(20.0)


def test_phase_quantisation_preserves_wrapped_phase_bounds() -> None:
    phase = np.linspace(-25.0, 25.0, 4001)
    for levels in (256, 1024):
        quantised, meta = mode2s_quantise_phase(phase, phase_levels=levels)

        assert float(np.min(quantised)) >= 0.0
        assert float(np.max(quantised)) < 2.0 * np.pi
        assert meta["phase_rms_error_rad"] <= (np.pi / levels) * 1.2
    clipped, meta = mode2s_quantise_phase(phase, phase_levels=None, phase_stroke_rad=0.8 * 2.0 * np.pi)
    assert meta["stroke_clipped_fraction"] > 0.0


def test_hv_piston_sweep_changes_metrics_smoothly_and_reports_optimum(bench) -> None:
    data, v0, backward = bench
    values = [0.0, 0.6, 1.2, 1.8, 2.4]
    cases = []
    for value in values:
        case = run_mode2s_degraded_forward(
            data, v0, backward,
            Mode2SPerturbation(label=f"piston_{value}", hv_piston_rad=value),
            sweep_parameter="hv_piston_rad", sweep_value=value,
        )
        cases.append(case)
    stokes = [float(c["pre_axicon"]["stokes_rms"]) for c in cases]

    # The piston is a uniform polarisation rotation: the pre-axicon Stokes error grows
    # smoothly away from the optimum at zero, while the intensity observable stays robust.
    assert stokes[0] == pytest.approx(min(stokes), abs=1e-9)
    assert all(b >= a - 1e-6 for a, b in zip(stokes[:3], stokes[1:4]))
    assert all(np.isfinite(float(c["comparison"]["z60_full_field_correlation"])) for c in cases)
    tol = mode2s_tolerance_from_sweep("hv_piston_rad", cases, nominal_value=0.0, wrap_period=2.0 * np.pi)
    assert tol["n_cases"] == len(values)
    assert "max_passing_deviation" in tol


def test_qwp_angle_error_changes_stokes_output(bench) -> None:
    data, v0, backward = bench
    clean = run_mode2s_degraded_forward(data, v0, backward, Mode2SPerturbation(label="clean"))
    erred = run_mode2s_degraded_forward(
        data, v0, backward,
        Mode2SPerturbation(label="qwp_2deg", qwp_angle_error_rad=float(np.deg2rad(2.0))),
    )

    assert float(erred["pre_axicon"]["stokes_rms"]) > float(clean["pre_axicon"]["stokes_rms"]) + 1e-3


def test_iris_decentre_reports_changed_efficiency_and_leakage(bench) -> None:
    data, _, _ = bench
    grid = data["grid"]
    A = np.asarray(data["A"], dtype=float) / np.sqrt(2.0)
    alpha = np.asarray(data["alpha"], dtype=float)
    carrier = np.exp(1j * 2.0 * np.pi * 6.25e3 * np.asarray(grid["X"], dtype=float))
    eh = A * np.exp(1j * alpha) * carrier
    ev = A * np.exp(1j * (-alpha + 0.5 * np.pi)) * carrier
    _, _, centred = mode2s_apply_4f(eh, ev, grid, carrier_lpmm=6.25, iris_radius_frac=0.4)
    _, _, shifted = mode2s_apply_4f(
        eh, ev, grid, carrier_lpmm=6.25, iris_radius_frac=0.4, iris_decentre_fx_lpmm=1.5,
    )

    assert shifted["first_order_efficiency"] < centred["first_order_efficiency"]
    assert "zero_order_leakage_after_iris" in shifted
    assert shifted["rejected_power_fraction"] > centred["rejected_power_fraction"]


def test_compensation_variables_are_bounded_and_physically_interpretable(bench) -> None:
    data, v0, backward = bench
    perturbation = Mode2SPerturbation(label="moderate_piston", hv_piston_rad=0.4, slm_aperture_clip=True)
    correction, meta = run_mode2s_precompensation(data, v0, backward, perturbation, maxiter=3)

    assert meta["n_parameters"] == 16
    assert set(meta["bounds"]) == {
        "sector_piston_rad", "global_v_piston_rad", "sector_rotation_rad", "sector_duty_scale",
        "qwp_angle_correction_rad", "zernike_rad", "iris_recentre_lpmm", "mask_recentre_m",
    }
    row = correction.as_row()
    assert all(abs(v) <= 0.5 * np.pi + 1e-12 for v in row["sector_pistons_rad"])
    assert abs(row["global_v_piston_rad"]) <= np.pi + 1e-12
    assert abs(row["qwp_angle_correction_deg"]) <= 3.0 + 1e-9
    assert 0.7 <= row["sector_duty_scale"] <= 1.3
    assert abs(row["mask_recentre_x_um"]) <= 1000.0 + 1e-9


def test_six_lobed_structure_cannot_pass_strict_hexagon_gate() -> None:
    sym = {
        "rot_corr_60": 0.35,
        "rot_corr_120": 0.60,
        "c120_minus_c60": 0.25,
        "order3_over_order6": 0.6,
        "six_sector_max_over_min": 1.3,
        "ring_island_count": 6,
    }

    assert mode1_symmetry_class(sym, dark_core_ratio=0.05) != "visual_hexagonal_field"


def test_source_scale_outcome_makes_no_microfabrication_claim(bench) -> None:
    data, v0, backward = bench
    clean = run_mode2s_degraded_forward(data, v0, backward, Mode2SPerturbation(label="clean"))
    from vbb_study.digital_twin import mode2s_outcome_report

    outcome = mode2s_outcome_report(
        clean_case=clean,
        clean_reference_correlation=float(clean["comparison"]["z60_full_field_correlation"]),
        fit_report=mode2s_slm_aperture_fit_report(data),
        tolerances=[],
        combined_cases=[clean],
        compensated_cases=[],
    )
    manifest = mode2s_scope_manifest(outcome)

    assert outcome["suggested_outcome"] in outcome["allowed_outcomes"]
    assert outcome["inherited_objective_sample_geometry_used"] is False
    assert outcome["microfabrication_sample_plane_claim"] is False
    assert manifest["inherited_objective_sample_geometry"] is False
    assert manifest["micro_scale_sample_plane_simulated"] is False
    assert manifest["microfabrication_sample_plane_claim"] is False
