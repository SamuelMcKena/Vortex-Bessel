"""Phase 2K mathematical truth tests.

These tests are intentionally reference-first rather than regression-first. A
historical numerical baseline is not accepted merely because the old code
reproduces it. Each check compares the implementation with an analytic law or
an independently written formulation.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.special as sp

from vbb_study.design import J0_FIRST_ZERO, compute_design_from_config, default_config
from vbb_study.digital_twin.vortex_error_reference_models import (
    exact_refractive_axicon_kr_m_inv,
)
from vbb_study.equations.fields import make_xy_grid
from vbb_study.equations.propagation import (
    angular_spectrum_propagate_bl,
    bandlimit_mask_matsushima,
    discrete_power,
    matsushima_bandlimit_mask,
)
from vbb_study.equations.scalar_bessel import (
    axicon_cone_angle_exact_rad,
    bessel_gauss_field,
    bessel_gauss_ring_radius_m,
    ring_radius_from_jprime_zero_m,
    transverse_wavevector_from_axicon,
)
from vbb_study.equations.vector_debye import DebyeConfig, debye_focus_plane
from vbb_study.equations.vector_fresnel_interface import fresnel_coefficients


def _fresnel_fft(field: np.ndarray, dx_m: float, z_m: float, k_m_inv: float) -> np.ndarray:
    """Independent paraxial transfer-function propagation used only in tests."""

    n = int(field.shape[0])
    fx = np.fft.fftfreq(n, d=float(dx_m))
    kx = 2.0 * np.pi * fx
    kx_grid, ky_grid = np.meshgrid(kx, kx)
    transfer = np.exp(-1j * (kx_grid**2 + ky_grid**2) * float(z_m) / (2.0 * float(k_m_inv)))
    return np.fft.ifft2(np.fft.fft2(field) * transfer)


def _grid(n: int = 384, dx_m: float = 0.35e-6):
    x = (np.arange(n, dtype=float) - n // 2) * float(dx_m)
    xg, yg = np.meshgrid(x, x)
    return xg, yg, np.hypot(xg, yg), np.arctan2(yg, xg)


def _debye_symmetry_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, DebyeConfig]:
    pupil_radius = 1.5e-3
    pupil_axis = np.linspace(-pupil_radius, pupil_radius, 129)
    X, Y = np.meshgrid(pupil_axis, pupil_axis, indexing="xy")
    config = DebyeConfig(
        wavelength_m=1029.0e-9,
        refractive_index=1.0,
        numerical_aperture=0.35,
        focal_length_m=4.0e-3,
        pupil_radius_m=pupil_radius,
        quadrature_order_r=48,
        quadrature_order_phi=192,
        max_output_points=16,
    )
    return pupil_axis, X, Y, config


def test_inverse_design_uses_exact_j0_first_zero_and_round_trips() -> None:
    cfg = default_config("fast")
    design = compute_design_from_config(cfg)
    expected_kr = 2.0 * J0_FIRST_ZERO / float(cfg.target.target_core_diameter_m)
    recovered_diameter = 2.0 * J0_FIRST_ZERO / float(design.kr_sample_m_inv)
    assert math.isclose(design.kr_sample_m_inv, expected_kr, rel_tol=2.0e-15, abs_tol=0.0)
    assert math.isclose(
        recovered_diameter,
        float(cfg.target.target_core_diameter_m),
        rel_tol=2.0e-15,
        abs_tol=0.0,
    )


def test_bessel_gauss_z0_is_exact_requested_waist_field() -> None:
    _, _, radius, phi = _grid(192, 0.45e-6)
    ell = 3
    kr = 0.85e6
    waist = 18.0e-6
    got = bessel_gauss_field(radius, phi, ell=ell, kr_m_inv=kr, waist_m=waist)
    expected = sp.jv(ell, kr * radius) * np.exp(-(radius**2) / waist**2) * np.exp(1j * ell * phi)
    assert np.allclose(got, expected, rtol=2.0e-13, atol=2.0e-13)


def test_bessel_gauss_analytic_propagation_matches_independent_fresnel_fft() -> None:
    xg, _, radius, phi = _grid()
    wavelength = 1029.0e-9
    n_medium = 1.0
    k = 2.0 * np.pi * n_medium / wavelength
    ell = 3
    kr = 0.75e6
    waist = 16.0e-6
    z = 22.0e-6

    initial = bessel_gauss_field(
        radius,
        phi,
        ell=ell,
        kr_m_inv=kr,
        waist_m=waist,
        z_m=0.0,
    )
    numeric = _fresnel_fft(initial, float(xg[0, 1] - xg[0, 0]), z, k)
    analytic = bessel_gauss_field(
        radius,
        phi,
        ell=ell,
        kr_m_inv=kr,
        waist_m=waist,
        z_m=z,
        wavelength0_m=wavelength,
        n_medium=n_medium,
    )

    mask = np.abs(analytic) >= 1.0e-5 * float(np.max(np.abs(analytic)))
    rel_l2 = float(np.linalg.norm((numeric - analytic)[mask]) / np.linalg.norm(analytic[mask]))
    assert rel_l2 < 2.0e-4


def test_nonzero_z_bessel_gauss_requires_wavelength() -> None:
    _, _, radius, phi = _grid(64, 0.8e-6)
    try:
        bessel_gauss_field(
            radius,
            phi,
            ell=1,
            kr_m_inv=0.5e6,
            waist_m=15.0e-6,
            z_m=1.0e-6,
        )
    except ValueError as exc:
        assert "wavelength0_m" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("non-zero-z BG propagation silently omitted wavelength")


def test_finite_bg_ring_is_not_assumed_to_be_jprime_peak() -> None:
    ell = 3
    kr = 1.0e6
    finite_waist = 4.0e-6
    finite = bessel_gauss_ring_radius_m(ell, kr, finite_waist)
    pure = ring_radius_from_jprime_zero_m(ell, kr)
    assert finite < pure
    assert (pure - finite) / pure > 1.0e-3

    almost_flat = bessel_gauss_ring_radius_m(ell, kr, 2.0e-3)
    assert abs(almost_flat - pure) / pure < 2.0e-5


def test_exact_asm_propagates_fft_bin_plane_wave_with_correct_kz_phase() -> None:
    n = 192
    dx = 2.0e-6
    wavelength = 1029.0e-9
    z = 8.0e-3
    grid = make_xy_grid(n, dx)
    fx = 9.0 / (n * dx)
    fy = -6.0 / (n * dx)
    initial = np.exp(1j * 2.0 * np.pi * (fx * grid["X"] + fy * grid["Y"]))
    propagated = angular_spectrum_propagate_bl(
        initial,
        grid,
        wavelength,
        z,
        n_medium=1.0,
        bandlimit=False,
        include_evanescent=False,
    )
    k = 2.0 * np.pi / wavelength
    kx = 2.0 * np.pi * fx
    ky = 2.0 * np.pi * fy
    kz = math.sqrt(k * k - kx * kx - ky * ky)
    expected = initial * np.exp(1j * kz * z)
    rel_l2 = float(np.linalg.norm(propagated - expected) / np.linalg.norm(expected))
    assert rel_l2 < 2.0e-12


def test_unfiltered_asm_conserves_discrete_power_for_propagating_spectrum() -> None:
    n = 192
    dx = 2.0e-6
    wavelength = 1029.0e-9
    grid = make_xy_grid(n, dx)
    sigma = 35.0e-6
    initial = np.exp(-(grid["R"] ** 2) / sigma**2).astype(complex)
    propagated = angular_spectrum_propagate_bl(
        initial,
        grid,
        wavelength,
        5.0e-3,
        n_medium=1.0,
        bandlimit=False,
        include_evanescent=False,
    )
    p0 = discrete_power(initial, dx)
    p1 = discrete_power(propagated, dx)
    assert abs(p1 - p0) / p0 < 2.0e-12


def test_two_matsushima_mask_helpers_are_identical_in_air() -> None:
    n = 128
    dx = 2.0e-6
    wavelength = 1029.0e-9
    z = 4.0e-3
    grid = make_xy_grid(n, dx)
    standalone = matsushima_bandlimit_mask(
        grid["FX"],
        grid["FY"],
        wavelength_m=wavelength,
        z_m=z,
        N=n,
        dx_m=dx,
    )
    engine = bandlimit_mask_matsushima(grid, wavelength, z, n_medium=1.0)
    assert np.array_equal(standalone, engine)


def test_exact_snell_axicon_kr_matches_independent_digital_twin_reference() -> None:
    wavelength = 1029.0e-9
    k0 = 2.0 * np.pi / wavelength
    n_ax = 1.458
    n_ext = 1.0
    gamma = math.radians(2.0)
    kr_scalar_reference = transverse_wavevector_from_axicon(
        k0,
        n_ax,
        n_ext,
        gamma,
        mode="snell_exact",
    )
    kr_digital_twin = exact_refractive_axicon_kr_m_inv(
        wavelength_m=wavelength,
        base_angle_rad=gamma,
        refractive_index=n_ax,
        external_index=n_ext,
    )
    assert math.isclose(kr_scalar_reference, kr_digital_twin, rel_tol=2.0e-14, abs_tol=0.0)


def test_axicon_thin_phase_uses_vacuum_k0_not_medium_k() -> None:
    wavelength = 1029.0e-9
    k0 = 2.0 * np.pi / wavelength
    n_ax = 1.70
    n_medium = 1.45
    gamma = math.radians(1.5)
    expected = k0 * (n_ax - n_medium) * math.tan(gamma)
    got = transverse_wavevector_from_axicon(k0, n_ax, n_medium, gamma, mode="tan")
    assert math.isclose(got, expected, rel_tol=2.0e-15, abs_tol=0.0)


def test_axicon_angle_convention_obeys_snell_law() -> None:
    n_ax = 1.458
    n_ext = 1.0
    gamma = math.radians(2.0)
    theta = axicon_cone_angle_exact_rad(n_ax, n_ext, gamma)
    lhs = n_ax * math.sin(gamma)
    rhs = n_ext * math.sin(gamma + theta)
    assert math.isclose(lhs, rhs, rel_tol=2.0e-14, abs_tol=2.0e-14)


def test_lossless_fresnel_coefficients_obey_energy_conservation() -> None:
    n1 = 1.0
    n2 = 1.45
    for theta_deg in (0.0, 20.0, 50.0):
        coeff = fresnel_coefficients(n1, n2, math.radians(theta_deg))
        assert abs(float(coeff["R_s"]) + float(coeff["T_s"]) - 1.0) < 2.0e-13
        assert abs(float(coeff["R_p"]) + float(coeff["T_p"]) - 1.0) < 2.0e-13
        assert float(coeff["snell_residual"]) < 2.0e-15


def test_fresnel_p_reflection_vanishes_at_brewster_angle_for_lossless_dielectric() -> None:
    n1 = 1.0
    n2 = 1.45
    theta_b = math.atan(n2 / n1)
    coeff = fresnel_coefficients(n1, n2, theta_b)
    assert abs(complex(coeff["r_p"])) < 2.0e-14
    assert abs(float(coeff["R_p"])) < 5.0e-28


def test_vector_debye_uniform_x_polarisation_has_expected_on_axis_symmetry() -> None:
    pupil_axis, X, _, config = _debye_symmetry_fixture()
    output_axis = np.asarray([-0.15e-6, 0.0, 0.15e-6])
    result = debye_focus_plane(
        np.ones_like(X, dtype=np.complex128),
        np.zeros_like(X, dtype=np.complex128),
        pupil_axis,
        pupil_axis,
        output_axis,
        output_axis,
        0.0,
        config,
    )
    center = 1
    ex = complex(result.Ex[center, center])
    ey = complex(result.Ey[center, center])
    ez = complex(result.Ez[center, center])
    assert abs(ex) > 0.0
    assert abs(ey) / abs(ex) < 1.0e-12
    assert abs(ez) / abs(ex) < 1.0e-12
    assert float(result.metadata["vector_transversality_residual"]) < 1.0e-12


def test_vector_debye_radial_polarisation_produces_longitudinal_on_axis_field() -> None:
    pupil_axis, X, Y, config = _debye_symmetry_fixture()
    output_axis = np.asarray([-0.15e-6, 0.0, 0.15e-6])
    radius = np.hypot(X, Y)
    ex_pupil = np.divide(X, radius, out=np.zeros_like(X), where=radius > 0.0).astype(np.complex128)
    ey_pupil = np.divide(Y, radius, out=np.zeros_like(Y), where=radius > 0.0).astype(np.complex128)
    result = debye_focus_plane(
        ex_pupil,
        ey_pupil,
        pupil_axis,
        pupil_axis,
        output_axis,
        output_axis,
        0.0,
        config,
    )
    center = 1
    ex = complex(result.Ex[center, center])
    ey = complex(result.Ey[center, center])
    ez = complex(result.Ez[center, center])
    assert abs(ez) > 0.0
    assert (abs(ex) + abs(ey)) / abs(ez) < 1.0e-11
    assert float(result.metadata["vector_transversality_residual"]) < 1.0e-12
