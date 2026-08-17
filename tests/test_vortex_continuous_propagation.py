from __future__ import annotations

import numpy as np

from vbb_study.digital_twin.vortex_continuous_propagation import (
    adjacent_row_continuity_metrics,
    build_fixed_plane_longitudinal_map,
    build_fixed_support_spectrum,
    native_field_at_z,
)
from vbb_study.equations.fields import fft2c, ifft2c, make_xy_grid


WAVELENGTH = 1.029e-6


def _gaussian(n: int = 256, window_m: float = 10e-3):
    grid = make_xy_grid(n, window_m / n)
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    field = np.exp(-(X * X + Y * Y) / (1.3e-3**2)).astype(np.complex128)
    return field, grid


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, complex).ravel()
    bb = np.asarray(b, complex).ravel()
    return float(abs(np.vdot(aa, bb)) / (np.linalg.norm(aa) * np.linalg.norm(bb)))


def test_fixed_support_retains_gaussian_spectrum() -> None:
    field, grid = _gaussian()
    prop = build_fixed_support_spectrum(
        field,
        grid,
        wavelength_m=WAVELENGTH,
        z_max_m=0.14,
        minimum_retained_spectral_power=0.999999,
    )
    assert prop.retained_spectral_power_fraction > 0.999999
    assert prop.metadata["z_dependent_binary_mask"] is False


def test_fixed_support_propagation_has_semigroup_consistency() -> None:
    field, grid = _gaussian()
    prop = build_fixed_support_spectrum(
        field,
        grid,
        wavelength_m=WAVELENGTH,
        z_max_m=0.14,
        minimum_retained_spectral_power=0.999999,
    )
    z1 = 0.037
    z2 = 0.041
    u1 = native_field_at_z(prop, z1)
    direct = native_field_at_z(prop, z1 + z2)
    sequential = ifft2c(fft2c(u1) * np.exp(1j * prop.kz_m_inv * z2))
    assert _overlap(direct, sequential) > 0.999999999


def test_fixed_physical_plane_map_is_smooth_for_gaussian() -> None:
    field, grid = _gaussian(n=384)
    prop = build_fixed_support_spectrum(
        field,
        grid,
        wavelength_m=WAVELENGTH,
        z_max_m=0.14,
        minimum_retained_spectral_power=0.999999,
    )
    z = np.linspace(5e-3, 140e-3, 136)
    coordinate = np.linspace(-0.6e-3, 0.6e-3, 301)
    mapped = build_fixed_plane_longitudinal_map(
        prop,
        z_values_m=z,
        x_coordinates_m=coordinate,
        y_coordinates_m=coordinate,
        fixed_x_m=0.0,
        fixed_y_m=0.0,
        source_label="gaussian-continuity-test",
    )
    mx = adjacent_row_continuity_metrics(mapped.xz_intensity)
    my = adjacent_row_continuity_metrics(mapped.yz_intensity)
    assert mx["adjacent_row_rms_change_max_over_median"] < 3.0
    assert my["adjacent_row_rms_change_max_over_median"] < 3.0
    np.testing.assert_allclose(mapped.xz_intensity, mapped.yz_intensity, rtol=2e-10, atol=1e-14)


def test_fixed_plane_coordinates_do_not_move_with_z() -> None:
    field, grid = _gaussian()
    prop = build_fixed_support_spectrum(
        field,
        grid,
        wavelength_m=WAVELENGTH,
        z_max_m=0.10,
        minimum_retained_spectral_power=0.999999,
    )
    z = np.linspace(0.01, 0.10, 32)
    x = np.linspace(-0.4e-3, 0.4e-3, 129)
    y = np.linspace(-0.5e-3, 0.5e-3, 131)
    mapped = build_fixed_plane_longitudinal_map(
        prop,
        z_values_m=z,
        x_coordinates_m=x,
        y_coordinates_m=y,
        fixed_x_m=0.17e-3,
        fixed_y_m=-0.11e-3,
        source_label="fixed-plane-coordinate-test",
    )
    assert mapped.xz_fixed_y_m == -0.11e-3
    assert mapped.yz_fixed_x_m == 0.17e-3
    assert mapped.metadata["per_z_recentering"] is False
    np.testing.assert_array_equal(mapped.x_coordinates_m, x)
    np.testing.assert_array_equal(mapped.y_coordinates_m, y)
