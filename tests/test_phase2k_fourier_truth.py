"""Phase 2K 4f/Fourier-coordinate truth checks."""

from __future__ import annotations

import math

from vbb_study.equations.objective_pupil import (
    fourier_plane_carrier_separation_exact_geometric_m,
    fourier_plane_carrier_separation_m,
    fourier_plane_ring_radius_exact_geometric_m,
    fourier_plane_ring_radius_m,
    paraxial_fourier_relative_error,
)


def test_paraxial_fourier_coordinate_is_lambda_f_nu_by_definition() -> None:
    wavelength = 1029.0e-9
    focal_length = 0.300
    carrier = 6250.0
    expected = wavelength * focal_length * carrier
    got = fourier_plane_carrier_separation_m(carrier, focal_length, wavelength)
    assert math.isclose(got, expected, rel_tol=0.0, abs_tol=1.0e-18)


def test_nominal_carrier_paraxial_error_is_negligible_but_explicit() -> None:
    wavelength = 1029.0e-9
    focal_length = 0.300
    carrier = 6250.0
    paraxial = fourier_plane_carrier_separation_m(carrier, focal_length, wavelength)
    exact = fourier_plane_carrier_separation_exact_geometric_m(
        carrier, focal_length, wavelength, n_medium=1.0
    )
    rel = paraxial_fourier_relative_error(paraxial, exact)
    # Current nominal carrier is strongly paraxial.  This gate ensures that a
    # future carrier change cannot silently invalidate the physical-distance
    # interpretation of the FFT/Fresnel coordinate.
    assert abs(rel) < 3.0e-5


def test_fourier_ring_exact_reference_matches_wavevector_angle_geometry() -> None:
    wavelength = 1029.0e-9
    focal_length = 0.300
    n_medium = 1.0
    kr = 1.0e5
    k = 2.0 * math.pi * n_medium / wavelength
    theta = math.asin(kr / k)
    expected = focal_length * math.tan(theta)
    exact = fourier_plane_ring_radius_exact_geometric_m(
        kr, focal_length, wavelength, n_medium=n_medium
    )
    assert math.isclose(exact, expected, rel_tol=2.0e-15, abs_tol=0.0)


def test_high_angle_fourier_coordinate_is_not_silently_called_exact() -> None:
    wavelength = 1029.0e-9
    focal_length = 0.010
    n_medium = 1.0
    k = 2.0 * math.pi / wavelength
    theta = math.radians(20.0)
    kr = k * math.sin(theta)
    paraxial = fourier_plane_ring_radius_m(kr, focal_length, wavelength)
    exact = fourier_plane_ring_radius_exact_geometric_m(
        kr, focal_length, wavelength, n_medium=n_medium
    )
    rel = paraxial_fourier_relative_error(paraxial, exact)
    # x=lambda f nu = f sin(theta), whereas the geometric reference is
    # f tan(theta).  At 20 degrees the distinction is several percent and must
    # be exposed instead of hidden by a generic 'physical radius' label.
    assert abs(rel) > 0.05
