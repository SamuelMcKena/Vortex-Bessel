from __future__ import annotations

import numpy as np

from vbb_study.digital_twin.vortex_explicit_4f import (
    FourFError,
    LensError,
    explicit_4f_relay,
    nominal_order_position_m,
)
from vbb_study.equations.fields import make_xy_grid


def test_canonical_plus_one_iris_centre_is_about_1p929_mm() -> None:
    x, y = nominal_order_position_m(
        wavelength_m=1029e-9,
        focal_length_m=0.300,
        carrier_cpm=6250.0,
    )
    assert np.isclose(x, 1.9294149e-3, rtol=0.0, atol=1e-9)
    assert y == 0.0


def test_iris_error_offset_is_relative_to_nominal_plus_one_order() -> None:
    grid = make_xy_grid(128, 10e-3 / 128)
    X = np.asarray(grid["X"])
    Y = np.asarray(grid["Y"])
    field = np.exp(-(X**2 + Y**2) / (1.5e-3) ** 2) * np.exp(1j * 2*np.pi*6250.0*X)
    route = explicit_4f_relay(
        field,
        grid,
        wavelength_m=1029e-9,
        nominal_focal_length_m=0.05,
        nominal_iris_radius_m=1.5e-3,
        nominal_carrier_cpm=6250.0,
        error=FourFError(iris_offset_m=(100e-6, 0.0)),
    )
    nominal = route["metadata"]["nominal_selected_order_centre_m"]
    centre = route["metadata"]["physical_iris_centre_m"]
    assert np.isclose(centre[0] - nominal[0], 100e-6, atol=1e-12)


def test_rigid_lens_tilt_uses_rotated_plane_backend() -> None:
    grid = make_xy_grid(128, 10e-3 / 128)
    field = np.exp(-(np.asarray(grid["R"]) / 1.5e-3) ** 2).astype(complex)
    route = explicit_4f_relay(
        field,
        grid,
        wavelength_m=1029e-9,
        nominal_focal_length_m=0.05,
        nominal_iris_radius_m=2e-3,
        nominal_carrier_cpm=0.0,
        error=FourFError(lens1=LensError(tilt_rad=(0.0, 1e-3))),
    )
    assert route["metadata"]["lens1"]["tilt_status"] == "scalar_rotated_angular_spectrum_thin_lens"
    assert route["metadata"]["lens1"]["tilt_fidelity"] == "scalar_paraxial_rotated_plane"


def test_parallel_lens_keeps_identity_tilt_status() -> None:
    grid = make_xy_grid(128, 10e-3 / 128)
    field = np.exp(-(np.asarray(grid["R"]) / 1.5e-3) ** 2).astype(complex)
    route = explicit_4f_relay(
        field,
        grid,
        wavelength_m=1029e-9,
        nominal_focal_length_m=0.05,
        nominal_iris_radius_m=2e-3,
    )
    assert route["metadata"]["lens1"]["tilt_status"] == "none"
    assert route["metadata"]["lens2"]["tilt_status"] == "none"
