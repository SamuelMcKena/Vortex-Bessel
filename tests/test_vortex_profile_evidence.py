from __future__ import annotations

import numpy as np

from vbb_study.digital_twin.vortex_profile_evidence import (
    build_transverse_profile_evidence,
    profile_metrics,
    spectral_line_fields,
)
from vbb_study.equations.fields import make_xy_grid


def test_spectral_line_fields_match_native_centre_lines() -> None:
    grid = make_xy_grid(128, 8e-6)
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    field = np.exp(-(X * X + Y * Y) / (0.23e-3**2)) * np.exp(1j * 3.0e3 * X)
    x = np.asarray(grid["x"], dtype=float)
    mid = int(grid["N"]) // 2
    # make_xy_grid is deliberately half-pixel centred for even N, so the native
    # centre row/column lies at +dx/2 rather than physical zero.  Compare the
    # Fourier-series sampler with that exact native physical coordinate.
    native_centre = float(x[mid])
    fx, fy = spectral_line_fields(
        field,
        grid,
        x_coordinates_m=x,
        y_coordinates_m=x,
        fixed_x_m=native_centre,
        fixed_y_m=native_centre,
    )
    assert np.max(np.abs(fx - field[mid, :])) < 1e-11
    assert np.max(np.abs(fy - field[:, mid])) < 1e-11


def test_physical_zero_profile_is_subpixel_not_native_centre_alias() -> None:
    grid = make_xy_grid(128, 8e-6)
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    field = np.exp(-(X * X + Y * Y) / (0.20e-3**2))
    x = np.asarray(grid["x"], dtype=float)
    zero_x, zero_y = spectral_line_fields(
        field,
        grid,
        x_coordinates_m=np.asarray([0.0]),
        y_coordinates_m=np.asarray([0.0]),
        fixed_x_m=0.0,
        fixed_y_m=0.0,
    )
    # For an even half-pixel grid the physical origin is not a native pixel, so
    # the direct spectral evaluation should not be silently replaced by either
    # adjacent sample.
    mid = int(grid["N"]) // 2
    assert not np.isclose(abs(zero_x[0]) ** 2, abs(field[mid, mid]) ** 2, rtol=0.0, atol=1e-12)
    assert np.isclose(abs(zero_x[0]) ** 2, abs(zero_y[0]) ** 2, rtol=1e-12, atol=1e-12)


def test_vortex_profile_axis_is_topological_not_energy_centroid() -> None:
    grid = make_xy_grid(256, 8e-6)
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    x0 = 210e-6
    y0 = -80e-6
    r2 = (X - x0) ** 2 + (Y - y0) ** 2
    field = ((X - x0) + 1j * (Y - y0)) * np.exp(-r2 / (0.34e-3**2))
    field *= 1.0 + 0.65 * X / 1.0e-3
    evidence = build_transverse_profile_evidence(
        field,
        grid,
        vortex_charge=1,
        lab_coordinate_m=np.linspace(-0.6e-3, 0.6e-3, 241),
        relative_coordinate_m=np.linspace(-0.20e-3, 0.20e-3, 201),
    )
    dx = float(grid["dx"])
    assert abs(evidence.morphology_axis.x_m - x0) < 1.5 * dx
    assert abs(evidence.morphology_axis.y_m - y0) < 1.5 * dx
    assert evidence.morphology_axis.detected_topological_charge == 1


def test_common_profile_metrics_preserve_absolute_intensity_change() -> None:
    grid = make_xy_grid(192, 7e-6)
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    base = np.exp(-(X * X + Y * Y) / (0.22e-3**2))
    coordinates = np.linspace(-0.5e-3, 0.5e-3, 301)
    relative = np.linspace(-0.18e-3, 0.18e-3, 221)
    nominal = build_transverse_profile_evidence(
        base,
        grid,
        vortex_charge=0,
        lab_coordinate_m=coordinates,
        relative_coordinate_m=relative,
    )
    weaker = build_transverse_profile_evidence(
        0.7 * base,
        grid,
        vortex_charge=0,
        lab_coordinate_m=coordinates,
        relative_coordinate_m=relative,
    )
    metrics = profile_metrics(weaker, nominal_peak_2d_au=nominal.peak_2d_au)
    assert np.isclose(metrics["peak_2d_over_nominal"], 0.49, rtol=2e-3)
    assert np.isclose(
        metrics["axis_x_line_peak_over_nominal_2d_peak"],
        0.49,
        rtol=2e-3,
    )
    assert np.isclose(
        metrics["axis_y_line_peak_over_nominal_2d_peak"],
        0.49,
        rtol=2e-3,
    )
