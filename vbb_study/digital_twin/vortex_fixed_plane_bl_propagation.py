"""Fixed-physical-plane, distance-aware band-limited ASM diagnostics.

This module separates two ideas that were previously conflated:

1. the propagation transfer function may legitimately use the distance-specific
   Matsushima band limit required by the sampled FFT representation; and
2. the *plot coordinate system* must not move from one z plane to the next.

The recent axicon diagnostic failure was caused by (2): a tracked/recentred
coordinate frame could jump between neighbouring Bessel features.  A later
three-way Phase-2B audit also showed that, on its N=512 grid, replacing the
normal distance-specific BL-ASM support with one max-z mask everywhere was
actually less accurate against a 2x padded unbandlimited reference.

Therefore this module keeps real fixed x-z/y-z laboratory planes while allowing
only the physically/numerically motivated distance-dependent BL-ASM support to
change with z.  No centroid/core tracking, image interpolation, or coordinate
warping is performed during the sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.equations.fields import fft2c
from vbb_study.equations.propagation import (
    asm_longitudinal_wavenumber_m_inv,
    bandlimit_mask_matsushima,
)


@dataclass(frozen=True)
class BandlimitedFixedPlaneLongitudinalMap:
    x_coordinates_m: np.ndarray = field(repr=False, compare=False)
    y_coordinates_m: np.ndarray = field(repr=False, compare=False)
    z_m: np.ndarray = field(repr=False, compare=False)
    xz_intensity: np.ndarray = field(repr=False, compare=False)
    yz_intensity: np.ndarray = field(repr=False, compare=False)
    support_retained_spectral_power_fraction: np.ndarray = field(repr=False, compare=False)
    xz_fixed_y_m: float
    yz_fixed_x_m: float
    metadata: Mapping[str, Any]


def _phase_matrix(
    coordinates_m: np.ndarray,
    *,
    n: int,
    dx_m: float,
    centre_coordinate_m: float,
) -> np.ndarray:
    frequency_index = np.arange(int(n), dtype=float) - int(n) / 2.0
    sample_coordinate = (
        np.asarray(coordinates_m, dtype=float) - float(centre_coordinate_m)
    ) / float(dx_m)
    return np.exp(2j * np.pi * np.outer(frequency_index, sample_coordinate) / float(n))


def build_bandlimited_fixed_plane_longitudinal_map(
    scalar_field: np.ndarray,
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    z_values_m: Sequence[float],
    x_coordinates_m: Sequence[float],
    y_coordinates_m: Sequence[float],
    fixed_x_m: float,
    fixed_y_m: float,
    n_medium: float = 1.0,
    minimum_retained_spectral_power: float = 0.985,
    source_label: str,
) -> BandlimitedFixedPlaneLongitudinalMap:
    """Build fixed x-z/y-z planes using distance-specific BL-ASM support.

    The complex source spectrum is computed exactly once.  For each propagation
    distance only the BL-ASM transfer function changes; the physical x/y lines
    and their fixed orthogonal coordinates remain unchanged.

    ``support_retained_spectral_power_fraction`` is reported for every z.  The
    minimum is hard-gated so a plot cannot silently rely on severe spectral
    truncation.
    """

    n = int(grid["N"])
    dx = float(grid["dx"])
    field_value = np.asarray(scalar_field, dtype=np.complex128)
    if field_value.shape != (n, n):
        raise ValueError("scalar field must match the declared square grid")
    z = np.asarray(z_values_m, dtype=float)
    x = np.asarray(x_coordinates_m, dtype=float)
    y = np.asarray(y_coordinates_m, dtype=float)
    if z.ndim != 1 or x.ndim != 1 or y.ndim != 1:
        raise ValueError("z/x/y coordinates must be one-dimensional")
    if z.size < 16 or x.size < 64 or y.size < 64:
        raise ValueError("fixed-plane diagnostics require >=16 z and >=64 transverse samples")

    spectrum = fft2c(field_value)
    spectrum_power = float(np.sum(np.abs(spectrum) ** 2))
    kz = asm_longitudinal_wavenumber_m_inv(
        np.asarray(grid["FX"], dtype=float),
        np.asarray(grid["FY"], dtype=float),
        wavelength_m=float(wavelength_m) / float(n_medium),
        include_evanescent=True,
    )
    native = np.asarray(grid["x"], dtype=float)
    centre = float(native[n // 2])
    px = _phase_matrix(x, n=n, dx_m=dx, centre_coordinate_m=centre)
    py = _phase_matrix(y, n=n, dx_m=dx, centre_coordinate_m=centre)
    fixed_y_phase = _phase_matrix(
        np.asarray([float(fixed_y_m)]),
        n=n,
        dx_m=dx,
        centre_coordinate_m=centre,
    )[:, 0]
    fixed_x_phase = _phase_matrix(
        np.asarray([float(fixed_x_m)]),
        n=n,
        dx_m=dx,
        centre_coordinate_m=centre,
    )[:, 0]

    xz = np.empty((z.size, x.size), dtype=float)
    yz = np.empty((z.size, y.size), dtype=float)
    retained = np.empty(z.size, dtype=float)
    for iz, zz in enumerate(z):
        support = bandlimit_mask_matsushima(
            dict(grid),
            float(wavelength_m),
            float(zz),
            n_medium=float(n_medium),
        )
        retained[iz] = float(
            np.sum(np.abs(spectrum[support]) ** 2) / max(spectrum_power, np.finfo(float).tiny)
        )
        transfer = np.exp(1j * kz * float(zz)) * support
        propagated_spectrum = spectrum * transfer
        x_line_spectrum = fixed_y_phase @ propagated_spectrum
        y_line_spectrum = propagated_spectrum @ fixed_x_phase
        x_field = x_line_spectrum @ px / float(n * n)
        y_field = y_line_spectrum @ py / float(n * n)
        xz[iz] = np.abs(x_field) ** 2
        yz[iz] = np.abs(y_field) ** 2

    minimum = float(np.min(retained))
    required = float(minimum_retained_spectral_power)
    if minimum < required:
        raise RuntimeError(
            "distance-aware BL-ASM support removes too much source spectrum for a validated longitudinal figure: "
            f"minimum retained={minimum:.8f} < required={required:.8f}"
        )

    return BandlimitedFixedPlaneLongitudinalMap(
        x_coordinates_m=x,
        y_coordinates_m=y,
        z_m=z,
        xz_intensity=xz,
        yz_intensity=yz,
        support_retained_spectral_power_fraction=retained,
        xz_fixed_y_m=float(fixed_y_m),
        yz_fixed_x_m=float(fixed_x_m),
        metadata={
            "outcome": "DISTANCE-AWARE-BL-ASM-FIXED-PHYSICAL-PLANES",
            "source_label": str(source_label),
            "propagator": "Matsushima band-limited angular spectrum",
            "support_model": "distance-specific Matsushima mask",
            "longitudinal_coordinates": "fixed physical planes",
            "xz_plane": f"y={float(fixed_y_m):.12g} m",
            "yz_plane": f"x={float(fixed_x_m):.12g} m",
            "per_z_recentering": False,
            "coordinate_warping": False,
            "minimum_retained_spectral_power_fraction": minimum,
            "required_minimum_retained_spectral_power_fraction": required,
        },
    )


__all__ = [
    "BandlimitedFixedPlaneLongitudinalMap",
    "build_bandlimited_fixed_plane_longitudinal_map",
]
