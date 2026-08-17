from __future__ import annotations

import numpy as np

from vbb_study.digital_twin.propagation_audit import (
    central_roi_mask,
    compare_intensity_arrays,
    compare_intensity_fields,
    scalar_padded_reference,
    vector_padded_reference_from_projected_spectra,
)
from vbb_study.equations.fields import fft2c, make_xy_grid
from vbb_study.equations.propagation import angular_spectrum_propagate_bl


def test_scalar_padded_reference_matches_native_gaussian_in_central_roi() -> None:
    grid = make_xy_grid(128, 8e-6)
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    field = np.exp(-(X * X + Y * Y) / (0.20e-3**2))
    wavelength = 1029e-9
    z = 1.5e-3
    candidate = angular_spectrum_propagate_bl(
        field,
        grid,
        wavelength,
        z,
        bandlimit=True,
        include_evanescent=True,
    )
    reference = scalar_padded_reference(
        field,
        grid,
        wavelength_m=wavelength,
        z_m=z,
        pad_factor=2,
    )
    comparison = compare_intensity_fields(
        candidate,
        reference,
        roi_mask=central_roi_mask(grid, 0.30e-3),
        dx_m=float(grid["dx"]),
    )
    assert comparison.intensity_correlation > 0.999
    assert comparison.normalised_relative_l2 < 0.02


def test_vector_padded_reference_preserves_uniform_transverse_polarisation_shape() -> None:
    grid = make_xy_grid(96, 10e-6)
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    ex0 = np.exp(-(X * X + Y * Y) / (0.19e-3**2)).astype(np.complex128)
    ey0 = np.zeros_like(ex0)
    ez0 = np.zeros_like(ex0)
    wavelength = 1029e-9
    k = 2.0 * np.pi / wavelength
    kx = 2.0 * np.pi * np.asarray(grid["FX"], dtype=float)
    ky = 2.0 * np.pi * np.asarray(grid["FY"], dtype=float)
    arg = k * k - kx * kx - ky * ky
    kz = np.where(arg >= 0.0, np.sqrt(np.maximum(arg, 0.0)), 1j * np.sqrt(np.maximum(-arg, 0.0)))
    ax = fft2c(ex0)
    ay = fft2c(ey0)
    az = fft2c(ez0)
    sx = kx / k
    sy = ky / k
    sz = kz / k
    dot = sx * ax + sy * ay + sz * az
    prepared = (ax - sx * dot, ay - sy * dot, az - sz * dot, kz)
    z = 1.0e-3
    transfer = np.exp(1j * prepared[3] * z)
    from vbb_study.equations.fields import ifft2c
    ex = ifft2c(prepared[0] * transfer)
    ey = ifft2c(prepared[1] * transfer)
    ez = ifft2c(prepared[2] * transfer)
    candidate = np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2
    reference = vector_padded_reference_from_projected_spectra(
        prepared,
        grid,
        wavelength_m=wavelength,
        z_m=z,
        pad_factor=2,
    )
    comparison = compare_intensity_arrays(
        candidate,
        reference,
        roi_mask=central_roi_mask(grid, 0.30e-3),
        dx_m=float(grid["dx"]),
    )
    assert comparison.intensity_correlation > 0.995
    assert comparison.normalised_relative_l2 < 0.08
