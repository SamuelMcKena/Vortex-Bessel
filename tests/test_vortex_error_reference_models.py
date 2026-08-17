from __future__ import annotations

import math

import numpy as np

from vbb_study.digital_twin.vortex_error_reference_models import (
    error_model_fidelity_registry,
    first_order_iris_geometry,
    grating_order_direction_cosines,
    rounded_tip_modulation_period_m,
    snell_axicon_geometry,
)


LAMBDA_M = 1029e-9
CARRIER_CPM = 6250.0
FOURF_F_M = 0.300
IRIS_RADIUS_CPM = 2500.0


def test_nominal_plus_one_order_reproduces_fourier_geometry() -> None:
    result = first_order_iris_geometry(
        wavelength_m=LAMBDA_M,
        carrier_cpm=CARRIER_CPM,
        iris_radius_cpm=IRIS_RADIUS_CPM,
        focal_length_m=FOURF_F_M,
    )
    # Paraxial lambda*f*G is 1.929375 mm; the exact f*tan(theta) correction is tiny.
    assert np.isclose(result["expected_fourier_x_m"], 1.9294149e-3, rtol=0.0, atol=1e-9)
    assert np.isclose(result["fourier_shift_x_m"], 0.0, atol=1e-15)
    assert np.isclose(result["order_offset_over_iris_radius"], 0.0, atol=1e-15)


def test_one_mrad_input_angle_moves_order_by_about_point_three_mm() -> None:
    result = first_order_iris_geometry(
        wavelength_m=LAMBDA_M,
        carrier_cpm=CARRIER_CPM,
        iris_radius_cpm=IRIS_RADIUS_CPM,
        focal_length_m=FOURF_F_M,
        input_angle_x_rad=1.0e-3,
    )
    assert np.isclose(result["fourier_shift_x_m"], 0.3000216e-3, rtol=0.0, atol=2e-8)
    assert np.isclose(result["order_offset_over_iris_radius"], 0.38872685, rtol=0.0, atol=1e-6)


def test_grating_direction_cosines_obey_tangential_momentum() -> None:
    angle = 0.7e-3
    result = grating_order_direction_cosines(
        wavelength_m=LAMBDA_M,
        carrier_cpm=CARRIER_CPM,
        input_angle_x_rad=angle,
        order=1,
    )
    assert np.isclose(result.sx_order, math.sin(angle) + LAMBDA_M * CARRIER_CPM)
    assert result.propagating is True
    assert np.isclose(result.sx_order**2 + result.sy_order**2 + result.sz_order**2, 1.0)


def test_shallow_axicon_formula_is_excellent_at_two_degrees() -> None:
    geometry = snell_axicon_geometry(
        base_angle_rad=math.radians(2.0),
        refractive_index=1.458,
        external_index=1.0,
    )
    assert np.isclose(math.degrees(geometry.deflection_rad), 0.9166674, atol=1e-6)
    assert abs(geometry.shallow_relative_error) < 5e-4


def test_shallow_axicon_formula_is_not_quantitative_at_twenty_degrees() -> None:
    geometry = snell_axicon_geometry(
        base_angle_rad=math.radians(20.0),
        refractive_index=1.458,
        external_index=1.0,
    )
    assert np.isclose(math.degrees(geometry.deflection_rad), 9.9117407, atol=1e-6)
    assert abs(geometry.shallow_relative_error) > 0.03


def test_round_tip_interference_period_matches_reference_scale() -> None:
    geometry = snell_axicon_geometry(
        base_angle_rad=math.radians(2.0),
        refractive_index=1.458,
        external_index=1.0,
    )
    period = rounded_tip_modulation_period_m(
        wavelength_in_medium_m=LAMBDA_M,
        cone_angle_rad=geometry.deflection_rad,
    )
    assert np.isclose(period, 8.0403756e-3, rtol=0.0, atol=1e-9)


def test_axicon_tilt_is_blocked_for_report_until_tilted_plane_backend_exists() -> None:
    policy = error_model_fidelity_registry()
    assert policy["axicon_tilt"]["status"] == "blocked_for_report"
    assert "rotated-angular-spectrum" in policy["axicon_tilt"]["required_backend"]
