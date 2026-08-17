"""Continuous fixed-support angular-spectrum diagnostics.

A physical free-space field evolves continuously with propagation distance.  A
longitudinal diagnostic should therefore never acquire discontinuities because
its numerical spectral support or plotting coordinate system changes from one
z row to the next.

This module freezes one angular-spectrum support mask for an entire z sweep.
The mask is the most restrictive Matsushima support required at ``z_max`` and
is applied once to the source spectrum.  Every subsequent plane is then the
same finite spectral sum multiplied only by ``exp(i kz z)``.  Consequently the
propagation operator has a fixed domain and the represented field is a
continuous function of z.

The retained source spectral power is reported and can be hard-gated.  If the
fixed support removes appreciable physical spectrum, the correct response is to
increase the computational window / use a different propagation representation,
not to allow a z-dependent binary mask to alter the model during the sweep.

Longitudinal x-z and y-z maps are sampled on *fixed physical planes*.  No
per-z centroid/core recentering is performed in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.equations.fields import fft2c, ifft2c
from vbb_study.equations.propagation import (
    asm_longitudinal_wavenumber_m_inv,
    bandlimit_mask_matsushima,
)


EPS = np.finfo(float).tiny


@dataclass(frozen=True)
class FixedSupportAngularSpectrum:
    spectrum: np.ndarray = field(repr=False, compare=False)
    kz_m_inv: np.ndarray = field(repr=False, compare=False)
    support_mask: np.ndarray = field(repr=False, compare=False)
    grid: Mapping[str, Any]
    wavelength_m: float
    n_medium: float
    z_max_m: float
    retained_spectral_power_fraction: float
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class FixedPlaneLongitudinalMap:
    x_coordinates_m: np.ndarray = field(repr=False, compare=False)
    y_coordinates_m: np.ndarray = field(repr=False, compare=False)
    z_m: np.ndarray = field(repr=False, compare=False)
    xz_intensity: np.ndarray = field(repr=False, compare=False)
    yz_intensity: np.ndarray = field(repr=False, compare=False)
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


def build_fixed_support_spectrum(
    scalar_field: np.ndarray,
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    z_max_m: float,
    n_medium: float = 1.0,
    minimum_retained_spectral_power: float = 0.995,
) -> FixedSupportAngularSpectrum:
    """Freeze the most restrictive required bandlimit for a complete z sweep."""

    n = int(grid["N"])
    field_value = np.asarray(scalar_field, dtype=np.complex128)
    if field_value.shape != (n, n):
        raise ValueError("scalar field must match the declared square grid")
    z_max = abs(float(z_max_m))
    if z_max <= 0.0:
        raise ValueError("z_max_m must be positive")

    raw_spectrum = fft2c(field_value)
    support = bandlimit_mask_matsushima(
        dict(grid),
        float(wavelength_m),
        z_max,
        n_medium=float(n_medium),
    )
    raw_power = float(np.sum(np.abs(raw_spectrum) ** 2))
    retained_power = float(np.sum(np.abs(raw_spectrum[support]) ** 2))
    retained_fraction = retained_power / max(raw_power, EPS)
    minimum = float(minimum_retained_spectral_power)
    if retained_fraction < minimum:
        raise RuntimeError(
            "fixed z-sweep angular-spectrum support removes appreciable source spectrum: "
            f"retained={retained_fraction:.8f} < required={minimum:.8f}; increase window/sampling"
        )

    fixed = np.where(support, raw_spectrum, 0.0j)
    kz = asm_longitudinal_wavenumber_m_inv(
        np.asarray(grid["FX"]),
        np.asarray(grid["FY"]),
        wavelength_m=float(wavelength_m) / float(n_medium),
        include_evanescent=True,
    )
    return FixedSupportAngularSpectrum(
        spectrum=np.asarray(fixed, dtype=np.complex128),
        kz_m_inv=np.asarray(kz, dtype=np.complex128),
        support_mask=np.asarray(support, dtype=bool),
        grid=dict(grid),
        wavelength_m=float(wavelength_m),
        n_medium=float(n_medium),
        z_max_m=z_max,
        retained_spectral_power_fraction=float(retained_fraction),
        metadata={
            "outcome": "FIXED-SUPPORT-CONTINUOUS-ANGULAR-SPECTRUM",
            "support_model": "single_Matsushima_mask_at_max_abs_z_applied_once",
            "z_support_m": z_max,
            "retained_spectral_power_fraction": float(retained_fraction),
            "minimum_retained_spectral_power": minimum,
            "z_dependent_binary_mask": False,
            "coordinate_warping": False,
        },
    )


def native_field_at_z(
    propagator: FixedSupportAngularSpectrum,
    z_m: float,
) -> np.ndarray:
    """Return the native-grid field at z from the frozen source spectrum."""

    transfer = np.exp(1j * propagator.kz_m_inv * float(z_m))
    return ifft2c(propagator.spectrum * transfer)


def _line_x(
    propagator: FixedSupportAngularSpectrum,
    z_m: float,
    x_coordinates_m: np.ndarray,
    *,
    fixed_y_m: float,
) -> np.ndarray:
    grid = propagator.grid
    n = int(grid["N"])
    dx = float(grid["dx"])
    native = np.asarray(grid["x"], dtype=float)
    centre = float(native[n // 2])
    px = _phase_matrix(
        np.asarray(x_coordinates_m, dtype=float),
        n=n,
        dx_m=dx,
        centre_coordinate_m=centre,
    )
    py = _phase_matrix(
        np.asarray([float(fixed_y_m)]),
        n=n,
        dx_m=dx,
        centre_coordinate_m=centre,
    )[:, 0]
    propagated = propagator.spectrum * np.exp(1j * propagator.kz_m_inv * float(z_m))
    line_spectrum = py @ propagated
    return np.asarray(line_spectrum @ px / float(n * n), dtype=np.complex128)


def _line_y(
    propagator: FixedSupportAngularSpectrum,
    z_m: float,
    y_coordinates_m: np.ndarray,
    *,
    fixed_x_m: float,
) -> np.ndarray:
    grid = propagator.grid
    n = int(grid["N"])
    dx = float(grid["dx"])
    native = np.asarray(grid["x"], dtype=float)
    centre = float(native[n // 2])
    py = _phase_matrix(
        np.asarray(y_coordinates_m, dtype=float),
        n=n,
        dx_m=dx,
        centre_coordinate_m=centre,
    )
    px = _phase_matrix(
        np.asarray([float(fixed_x_m)]),
        n=n,
        dx_m=dx,
        centre_coordinate_m=centre,
    )[:, 0]
    propagated = propagator.spectrum * np.exp(1j * propagator.kz_m_inv * float(z_m))
    line_spectrum = propagated @ px
    return np.asarray(line_spectrum @ py / float(n * n), dtype=np.complex128)


def line_fields_at_z(
    propagator: FixedSupportAngularSpectrum,
    *,
    z_m: float,
    x_coordinates_m: Sequence[float],
    y_coordinates_m: Sequence[float],
    fixed_x_m: float,
    fixed_y_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return complex x/y line fields on two fixed orthogonal physical planes."""

    x = np.asarray(x_coordinates_m, dtype=float)
    y = np.asarray(y_coordinates_m, dtype=float)
    return (
        _line_x(propagator, float(z_m), x, fixed_y_m=float(fixed_y_m)),
        _line_y(propagator, float(z_m), y, fixed_x_m=float(fixed_x_m)),
    )


def build_fixed_plane_longitudinal_map(
    propagator: FixedSupportAngularSpectrum,
    *,
    z_values_m: Sequence[float],
    x_coordinates_m: Sequence[float],
    y_coordinates_m: Sequence[float],
    fixed_x_m: float,
    fixed_y_m: float,
    source_label: str,
) -> FixedPlaneLongitudinalMap:
    """Build real x-z/y-z physical-plane maps without per-z coordinate shifts."""

    z = np.asarray(z_values_m, dtype=float)
    x = np.asarray(x_coordinates_m, dtype=float)
    y = np.asarray(y_coordinates_m, dtype=float)
    if z.ndim != 1 or x.ndim != 1 or y.ndim != 1:
        raise ValueError("z/x/y coordinates must be one-dimensional")
    if z.size < 16 or x.size < 64 or y.size < 64:
        raise ValueError("fixed-plane diagnostics require >=16 z and >=64 transverse samples")

    xz = np.empty((z.size, x.size), dtype=float)
    yz = np.empty((z.size, y.size), dtype=float)
    for iz, zz in enumerate(z):
        x_field, y_field = line_fields_at_z(
            propagator,
            z_m=float(zz),
            x_coordinates_m=x,
            y_coordinates_m=y,
            fixed_x_m=float(fixed_x_m),
            fixed_y_m=float(fixed_y_m),
        )
        xz[iz] = np.abs(x_field) ** 2
        yz[iz] = np.abs(y_field) ** 2

    return FixedPlaneLongitudinalMap(
        x_coordinates_m=x,
        y_coordinates_m=y,
        z_m=z,
        xz_intensity=xz,
        yz_intensity=yz,
        xz_fixed_y_m=float(fixed_y_m),
        yz_fixed_x_m=float(fixed_x_m),
        metadata={
            **dict(propagator.metadata),
            "source_label": str(source_label),
            "longitudinal_map": "fixed_physical_planes",
            "xz_plane": f"y={float(fixed_y_m):.12g} m",
            "yz_plane": f"x={float(fixed_x_m):.12g} m",
            "per_z_recentering": False,
        },
    )


def adjacent_row_continuity_metrics(intensity: np.ndarray) -> dict[str, float]:
    """Report discontinuity-sensitive differences between neighbouring z rows.

    These are diagnostics, not universal physical pass thresholds.  The important
    structural property is that no numerical support or coordinate transform
    changes between rows.  Large outliers can then be investigated as genuine
    rapid field evolution rather than an implementation switch.
    """

    values = np.asarray(intensity, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError("intensity must be a 2-D z-by-coordinate map")
    normalised = values / np.maximum(np.max(values, axis=1, keepdims=True), EPS)
    delta = np.linalg.norm(np.diff(normalised, axis=0), axis=1) / np.sqrt(values.shape[1])
    median = float(np.median(delta))
    return {
        "adjacent_row_rms_change_median": median,
        "adjacent_row_rms_change_max": float(np.max(delta)),
        "adjacent_row_rms_change_max_over_median": float(np.max(delta) / max(median, EPS)),
    }


__all__ = [
    "FixedPlaneLongitudinalMap",
    "FixedSupportAngularSpectrum",
    "adjacent_row_continuity_metrics",
    "build_fixed_plane_longitudinal_map",
    "build_fixed_support_spectrum",
    "line_fields_at_z",
    "native_field_at_z",
]
