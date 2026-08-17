from __future__ import annotations

import math

import numpy as np

from vbb_study.digital_twin.vortex_axicon_tip_reference import (
    flat_blunt_tip_sag_m,
    hyperbolic_round_tip_sag_m,
    normalised_intensity,
    radial_fresnel_field,
    shallow_exact_phase_gradient_relative_error,
    tip_resolution,
)


WAVELENGTH = 1.029e-6
GAMMA = math.radians(2.0)
N_AXICON = 1.45
BEAM_RADIUS = 2.0e-3


def test_hyperbolic_tip_has_sharp_cone_limit_and_zero_vertex_slope() -> None:
    r = np.linspace(0.0, 1.0e-3, 2001)
    cone = math.tan(GAMMA) * r
    sharp = hyperbolic_round_tip_sag_m(
        r,
        base_angle_rad=GAMMA,
        curvature_radius_m=0.0,
    )
    rounded = hyperbolic_round_tip_sag_m(
        r,
        base_angle_rad=GAMMA,
        curvature_radius_m=200e-6,
    )
    np.testing.assert_allclose(sharp, cone, rtol=0.0, atol=1e-15)
    assert rounded[0] == 0.0
    # Rounded vertex is locally flat; far from the tip its slope approaches cone slope.
    dr = r[1] - r[0]
    vertex_slope = (rounded[1] - rounded[0]) / dr
    far_slope = (rounded[-1] - rounded[-2]) / dr
    assert abs(vertex_slope) < 0.01 * math.tan(GAMMA)
    assert far_slope > 0.98 * math.tan(GAMMA)


def test_flat_blunt_tip_is_continuous_and_has_resolved_plateau() -> None:
    rf = 200e-6
    r = np.asarray([0.0, 0.5 * rf, rf, 1.5 * rf, 2.0 * rf])
    sag = flat_blunt_tip_sag_m(r, base_angle_rad=GAMMA, flat_radius_m=rf)
    np.testing.assert_allclose(sag[:3], 0.0, atol=1e-15)
    assert sag[3] > 0.0
    assert sag[4] > sag[3]


def test_tip_resolution_rejects_old_subpixel_scale() -> None:
    dx = 10e-3 / 1536
    old = tip_resolution(10e-6, dx, minimum_pixels=12.0)
    new = tip_resolution(100e-6, dx, minimum_pixels=12.0)
    assert old.radius_pixels < 2.0
    assert not old.resolved
    assert new.radius_pixels > 15.0
    assert new.resolved


def test_two_degree_shallow_phase_gradient_is_close_to_exact_snell() -> None:
    error = shallow_exact_phase_gradient_relative_error(
        base_angle_rad=GAMMA,
        refractive_index=N_AXICON,
    )
    assert error < 1e-3


def test_round_tip_generates_axial_modulation_in_high_resolution_fresnel_reference() -> None:
    z = np.linspace(20e-3, 120e-3, 161)
    sharp = radial_fresnel_field(
        radial_observation_m=[0.0],
        z_values_m=z,
        wavelength_m=WAVELENGTH,
        beam_radius_m=BEAM_RADIUS,
        base_angle_rad=GAMMA,
        refractive_index=N_AXICON,
        vortex_charge=0,
        tip_model="sharp",
        radial_step_m=1.0e-6,
    )[:, 0]
    rounded = radial_fresnel_field(
        radial_observation_m=[0.0],
        z_values_m=z,
        wavelength_m=WAVELENGTH,
        beam_radius_m=BEAM_RADIUS,
        base_angle_rad=GAMMA,
        refractive_index=N_AXICON,
        vortex_charge=0,
        tip_model="hyperbolic_round",
        tip_radius_m=200e-6,
        radial_step_m=1.0e-6,
    )[:, 0]
    ish = normalised_intensity(sharp)
    iro = normalised_intensity(rounded)
    # A real rounded tip must not collapse to the sharp-tip axial trace.
    rms = float(np.sqrt(np.mean((iro - ish) ** 2)))
    assert rms > 0.02
    assert np.all(np.isfinite(iro))
