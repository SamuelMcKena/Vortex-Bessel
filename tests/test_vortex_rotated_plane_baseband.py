from __future__ import annotations

import math

import numpy as np

from vbb_study.digital_twin.vortex_rotated_plane import rotate_angular_spectrum
from vbb_study.digital_twin.vortex_rotated_plane_baseband import rotate_baseband_angular_spectrum
from vbb_study.equations.fields import make_xy_grid


WAVELENGTH = 1.029e-6


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.complex128).ravel()
    bb = np.asarray(b, dtype=np.complex128).ravel()
    return float(abs(np.vdot(aa, bb)) / max(np.linalg.norm(aa) * np.linalg.norm(bb), 1e-30))


def _gaussian(n: int = 512, window_m: float = 10e-3) -> tuple[np.ndarray, dict]:
    grid = make_xy_grid(n, window_m / n)
    X = np.asarray(grid["X"])
    Y = np.asarray(grid["Y"])
    field = np.exp(-(X * X + Y * Y) / (1.5e-3**2)).astype(np.complex128)
    return field, grid


def test_baseband_matches_explicit_carrier_where_carrier_is_sampleable() -> None:
    field, grid = _gaussian()
    angle = math.radians(1.0)
    explicit, meta = rotate_angular_spectrum(
        field,
        grid,
        wavelength_m=WAVELENGTH,
        tilt_x_rad=0.0,
        tilt_y_rad=angle,
        spectral_center_cpm=(0.0, 0.0),
    )
    baseband, meta_bb = rotate_baseband_angular_spectrum(
        field,
        grid,
        wavelength_m=WAVELENGTH,
        tilt_x_rad=0.0,
        tilt_y_rad=angle,
        source_spectral_center_cpm=(0.0, 0.0),
    )
    fdx, fdy = map(float, meta["destination_spectral_center_cpm"])
    X = np.asarray(grid["X"])
    Y = np.asarray(grid["Y"])
    demodulated = explicit * np.exp(-2j * np.pi * (fdx * X + fdy * Y))
    assert _overlap(demodulated, baseband) > 0.99999
    np.testing.assert_allclose(
        meta_bb["destination_spectral_center_cpm"],
        meta["destination_spectral_center_cpm"],
        rtol=0.0,
        atol=1e-9,
    )
    assert float(meta_bb["normal_flux_power_ratio"]) > 0.999


def test_ten_degree_baseband_roundtrip_does_not_require_sampled_carrier() -> None:
    field, grid = _gaussian(n=768)
    angle = math.radians(10.0)
    forward, meta_f = rotate_baseband_angular_spectrum(
        field,
        grid,
        wavelength_m=WAVELENGTH,
        tilt_x_rad=angle,
        tilt_y_rad=0.0,
        source_spectral_center_cpm=(0.0, 0.0),
    )
    centre = tuple(map(float, meta_f["destination_spectral_center_cpm"]))
    # The 10-degree carrier is intentionally far beyond this grid's ordinary
    # sampled Nyquist frequency; it exists only as analytic carrier metadata.
    nyquist_cpm = 0.5 / float(grid["dx"])
    assert abs(centre[1]) > nyquist_cpm
    backward, meta_b = rotate_baseband_angular_spectrum(
        forward,
        grid,
        wavelength_m=WAVELENGTH,
        tilt_x_rad=angle,
        tilt_y_rad=0.0,
        source_spectral_center_cpm=centre,
        inverse=True,
    )

    # Raw spectral L2 is a projected-plane quantity at finite tilt.  For a
    # narrow carrier it changes approximately by sec(theta) forward and
    # cos(theta) backward; the product returns to unity.  The invariant used for
    # numerical validation is normal optical flux, not either raw L2 ratio.
    raw_product = float(meta_f["spectral_power_ratio"]) * float(
        meta_b["spectral_power_ratio"]
    )
    assert abs(raw_product - 1.0) < 2e-3
    assert float(meta_f["normal_flux_power_ratio"]) > 0.995
    assert float(meta_b["normal_flux_power_ratio"]) > 0.995
    assert _overlap(field, backward) > 0.999
