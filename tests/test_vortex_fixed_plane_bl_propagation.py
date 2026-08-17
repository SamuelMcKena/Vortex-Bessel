from __future__ import annotations

import numpy as np

from vbb_study.digital_twin.vortex_fixed_plane_bl_propagation import (
    build_bandlimited_fixed_plane_longitudinal_map,
)
from vbb_study.equations.fields import make_xy_grid
from vbb_study.equations.propagation import angular_spectrum_propagate_bl


def test_fixed_plane_bl_map_matches_native_asm_on_native_physical_lines() -> None:
    grid = make_xy_grid(128, 8e-6)
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    field = np.exp(-(X * X + Y * Y) / (0.20e-3**2)) * np.exp(1j * 2.0e3 * X)
    wavelength = 1029e-9
    z_values = np.linspace(0.5e-3, 2.0e-3, 16)
    x = np.asarray(grid["x"], dtype=float)
    mid = int(grid["N"]) // 2
    native_centre = float(x[mid])
    result = build_bandlimited_fixed_plane_longitudinal_map(
        field,
        grid,
        wavelength_m=wavelength,
        z_values_m=z_values,
        x_coordinates_m=x,
        y_coordinates_m=x,
        fixed_x_m=native_centre,
        fixed_y_m=native_centre,
        minimum_retained_spectral_power=0.99,
        source_label="unit-native-parity",
    )
    for iz, z in enumerate(z_values):
        propagated = angular_spectrum_propagate_bl(
            field,
            grid,
            wavelength,
            float(z),
            bandlimit=True,
            include_evanescent=True,
        )
        intensity = np.abs(propagated) ** 2
        assert np.max(np.abs(result.xz_intensity[iz] - intensity[mid, :])) < 1e-10
        assert np.max(np.abs(result.yz_intensity[iz] - intensity[:, mid])) < 1e-10


def test_fixed_plane_coordinates_do_not_change_with_z() -> None:
    grid = make_xy_grid(128, 8e-6)
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    field = np.exp(-((X - 0.12e-3) ** 2 + (Y + 0.05e-3) ** 2) / (0.18e-3**2))
    z_values = np.linspace(0.5e-3, 2.0e-3, 16)
    relative = np.linspace(-0.15e-3, 0.15e-3, 121)
    result = build_bandlimited_fixed_plane_longitudinal_map(
        field,
        grid,
        wavelength_m=1029e-9,
        z_values_m=z_values,
        x_coordinates_m=0.12e-3 + relative,
        y_coordinates_m=-0.05e-3 + relative,
        fixed_x_m=0.12e-3,
        fixed_y_m=-0.05e-3,
        minimum_retained_spectral_power=0.99,
        source_label="unit-fixed-coordinates",
    )
    assert result.xz_fixed_y_m == -0.05e-3
    assert result.yz_fixed_x_m == 0.12e-3
    assert result.metadata["per_z_recentering"] is False
    assert result.metadata["coordinate_warping"] is False
    assert float(np.min(result.support_retained_spectral_power_fraction)) >= 0.99
