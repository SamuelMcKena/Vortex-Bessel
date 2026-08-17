from __future__ import annotations

import math

import numpy as np

from vbb_study.digital_twin.vortex_round_tip_reference import (
    axisymmetric_fresnel_field,
    hyperboloidal_round_tip_sag_m,
)


def test_round_tip_sag_recovers_sharp_cone_at_zero_parameter() -> None:
    r = np.linspace(0.0, 2e-3, 1024)
    gamma = math.radians(2.0)
    sharp = hyperboloidal_round_tip_sag_m(
        r,
        base_angle_rad=gamma,
        rounding_parameter_m=0.0,
    )
    assert np.allclose(sharp, r * math.tan(gamma), rtol=0.0, atol=1e-15)


def test_round_tip_sag_has_zero_apex_slope_and_conical_outer_slope() -> None:
    r = np.linspace(0.0, 4e-3, 20001)
    gamma = math.radians(2.0)
    sag = hyperboloidal_round_tip_sag_m(
        r,
        base_angle_rad=gamma,
        rounding_parameter_m=10e-6,
    )
    central_slope = (sag[1] - sag[0]) / (r[1] - r[0])
    outer_slope = (sag[-1] - sag[-101]) / (r[-1] - r[-101])
    assert abs(central_slope) < 0.05 * math.tan(gamma)
    assert np.isclose(outer_slope, math.tan(gamma), rtol=0.0, atol=3e-4)


def test_axisymmetric_reference_is_finite_and_independent_of_2d_solver() -> None:
    result = axisymmetric_fresnel_field(
        wavelength_m=1029e-9,
        refractive_index=1.458,
        external_index=1.0,
        base_angle_rad=math.radians(2.0),
        beam_radius_m=2e-3,
        rounding_parameter_m=5e-6,
        rho_values_m=(0.0, 10e-6, 20e-6),
        z_values_m=(20e-3, 40e-3, 60e-3),
        radial_extent_m=4e-3,
        radial_samples=4096,
    )
    assert result.intensity.shape == (3, 3)
    assert np.all(np.isfinite(result.intensity))
    assert np.all(result.intensity >= 0.0)
    assert result.metadata["not_called"] == "2D angular-spectrum production solver"
