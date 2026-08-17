from __future__ import annotations

import math

import numpy as np

from vbb_study.digital_twin.phase2e_production_repair import build_nominal_source
from vbb_study.digital_twin.vortex_physical_errors import (
    PhysicalPerturbation,
    axicon_sag_m,
    build_physical_route_checkpoints,
    build_physical_source,
    incident_plane_wave_phase,
    physical_axicon_transmission,
)
from vbb_study.equations.fields import make_xy_grid


def _normalised_overlap(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.complex128).ravel()
    bb = np.asarray(b, dtype=np.complex128).ravel()
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(abs(np.vdot(aa, bb)) / max(float(denom), np.finfo(float).tiny))


def test_nominal_physical_route_matches_repaired_nominal_route() -> None:
    physical, _, physical_meta = build_physical_source("V1", grid_n=256)
    canonical, _, canonical_meta = build_nominal_source("V1", grid_n=256, aperture_model="none")
    assert physical_meta["additional_objective_pupil_application_count"] == 0
    assert canonical_meta["historical_objective_pupil_application_count"] == 0
    assert _normalised_overlap(physical, canonical) > 0.999999


def test_input_angle_is_applied_before_slm_and_changes_upstream_field() -> None:
    nominal = build_physical_route_checkpoints("B0", grid_n=256)
    tilted = build_physical_route_checkpoints(
        "B0",
        grid_n=256,
        perturbation=PhysicalPerturbation(input_beam_angle_rad=(1.0e-3, 0.0)),
    )
    assert tilted["metadata"]["input_angle_applied_plane"] == "before_SLM1"
    assert not np.allclose(nominal["raw_input"], tilted["raw_input"])
    assert not np.allclose(nominal["post_filter"], tilted["post_filter"])


def test_input_plane_wave_uses_direction_cosines() -> None:
    grid = make_xy_grid(128, 10e-3 / 128)
    wavelength = 1029e-9
    angle = 0.01
    phase = incident_plane_wave_phase(grid, wavelength, angle, 0.0)
    x = np.asarray(grid["x"], dtype=float)
    row = phase[64]
    measured_step = np.angle(row[65] * np.conj(row[64]))
    expected_step = 2.0 * math.pi / wavelength * math.sin(angle) * (x[65] - x[64])
    expected_step = math.atan2(math.sin(expected_step), math.cos(expected_step))
    assert np.isclose(measured_step, expected_step, atol=1e-12)


def test_beam_radius_change_is_rebuilt_before_slm() -> None:
    small = build_physical_route_checkpoints(
        "B0", grid_n=256, perturbation=PhysicalPerturbation(beam_radius_scale=0.7)
    )
    large = build_physical_route_checkpoints(
        "B0", grid_n=256, perturbation=PhysicalPerturbation(beam_radius_scale=1.3)
    )
    assert small["metadata"]["beam_radius_m"] < large["metadata"]["beam_radius_m"]
    assert not np.allclose(np.abs(small["raw_input"]), np.abs(large["raw_input"]))
    assert not np.allclose(small["post_filter"], large["post_filter"])


def test_hyperboloidal_tip_has_zero_central_slope_and_conical_asymptote() -> None:
    gamma = math.radians(2.0)
    r = np.linspace(0.0, 3e-3, 20001)
    rounded = axicon_sag_m(
        r,
        gamma,
        tip_model="hyperboloidal_round",
        rounding_parameter_m=10e-6,
    )
    sharp = axicon_sag_m(r, gamma, tip_model="sharp")
    central_slope = (rounded[1] - rounded[0]) / (r[1] - r[0])
    outer_slope = (rounded[-1] - rounded[-101]) / (r[-1] - r[-101])
    assert abs(central_slope) < 0.05 * math.tan(gamma)
    assert np.isclose(outer_slope, math.tan(gamma), rtol=0.0, atol=3e-4)
    assert np.max(np.abs((rounded[-100:] - rounded[-1]) - (sharp[-100:] - sharp[-1]))) < 1e-7


def test_flat_blunt_tip_has_zero_sag_in_declared_flat_radius() -> None:
    r = np.linspace(0.0, 500e-6, 1001)
    sag = axicon_sag_m(
        r,
        math.radians(2.0),
        tip_model="flat_blunt",
        flat_tip_radius_m=100e-6,
    )
    assert np.allclose(sag[r <= 100e-6], 0.0)
    assert np.any(sag[r > 100e-6] > 0.0)


def test_axicon_decentre_moves_physical_sag_not_field_afterwards() -> None:
    grid = make_xy_grid(256, 10e-3 / 256)
    nominal, _ = physical_axicon_transmission(
        grid,
        wavelength_m=1029e-9,
        refractive_index=1.458,
        external_index=1.0,
        base_angle_rad=math.radians(2.0),
    )
    shifted, _ = physical_axicon_transmission(
        grid,
        wavelength_m=1029e-9,
        refractive_index=1.458,
        external_index=1.0,
        base_angle_rad=math.radians(2.0),
        decentre_m=(200e-6, 0.0),
    )
    assert not np.allclose(nominal, shifted)
    assert np.allclose(np.abs(nominal), 1.0)
    assert np.allclose(np.abs(shifted), 1.0)


def test_axicon_tilt_is_not_labelled_as_full_snell_solution() -> None:
    grid = make_xy_grid(256, 10e-3 / 256)
    untilted, _ = physical_axicon_transmission(
        grid,
        wavelength_m=1029e-9,
        refractive_index=1.458,
        external_index=1.0,
        base_angle_rad=math.radians(2.0),
    )
    tilted, meta = physical_axicon_transmission(
        grid,
        wavelength_m=1029e-9,
        refractive_index=1.458,
        external_index=1.0,
        base_angle_rad=math.radians(2.0),
        tilt_rad=(0.0, math.radians(1.0)),
    )
    assert meta["axicon_tilt_model"] == "rotated_thin_element_opd_small_angle"
    assert meta["full_vector_snell_fresnel"] is False
    assert not np.allclose(untilted, tilted)


def test_large_axicon_tilt_is_refused_by_fidelity_gate() -> None:
    perturbation = PhysicalPerturbation(axicon_tilt_rad=(math.radians(6.0), 0.0))
    try:
        perturbation.validate()
    except ValueError as exc:
        assert "small-angle" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("large tilt should be rejected")
