"""Dense fixed-coordinate propagation diagnostics for Phase 2E.

The accepted fixed-grid stacks are intentionally retained for accepted metrics,
but their source-scale transverse sampling is too coarse for a tight report ROI.
This module evaluates the same discrete angular spectrum directly on requested
physical x/y coordinates.  It is a Fourier-series field synthesis, not image
interpolation, and uses the same Matsushima band limit as the native BL-ASM.
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


EPS = np.finfo(float).eps


@dataclass(frozen=True)
class DensePropagationMap:
    """Physical transverse-line intensity maps on a common x/y/z grid."""

    x_m: np.ndarray = field(repr=False, compare=False)
    y_m: np.ndarray = field(repr=False, compare=False)
    z_m: np.ndarray = field(repr=False, compare=False)
    xz_intensity: np.ndarray = field(repr=False, compare=False)
    yz_intensity: np.ndarray = field(repr=False, compare=False)
    metadata: Mapping[str, Any]


def _phase_matrix(
    coordinates_m: np.ndarray,
    *,
    n: int,
    dx_m: float,
    centre_coordinate_m: float,
) -> np.ndarray:
    """Return the centred inverse-DFT phase matrix at arbitrary coordinates."""

    frequency_index = np.arange(int(n), dtype=float) - int(n) / 2.0
    sample_coordinate = (
        np.asarray(coordinates_m, dtype=float) - float(centre_coordinate_m)
    ) / float(dx_m)
    return np.exp(
        2j * np.pi * np.outer(frequency_index, sample_coordinate) / float(n)
    )


def _synthesise_axis_fields(
    spectra: Sequence[np.ndarray],
    transfer: np.ndarray,
    *,
    phase_requested: np.ndarray,
    phase_orthogonal: np.ndarray,
    axis: str,
) -> tuple[np.ndarray, ...]:
    n = int(transfer.shape[0])
    fields: list[np.ndarray] = []
    for spectrum in spectra:
        propagated = np.asarray(spectrum, dtype=np.complex128) * transfer
        if axis == "x":
            line_spectrum = phase_orthogonal @ propagated
        elif axis == "y":
            line_spectrum = propagated @ phase_orthogonal
        else:
            raise ValueError(f"unsupported spectral-line axis {axis!r}")
        fields.append(np.asarray(line_spectrum @ phase_requested / float(n * n)))
    return tuple(fields)


def _normalised_correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.array(a, dtype=float, copy=True).ravel()
    bb = np.array(b, dtype=float, copy=True).ravel()
    aa -= float(np.mean(aa))
    bb -= float(np.mean(bb))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / max(denom, EPS))


def native_line_parity(
    spectra: Sequence[np.ndarray],
    kz_m_inv: np.ndarray,
    grid: Mapping[str, Any],
    wavelength_m: float,
    *,
    z_m: float,
    n_medium: float = 1.0,
) -> dict[str, float]:
    """Validate direct synthesis against native centred inverse FFT samples."""

    n = int(grid["N"])
    dx_m = float(grid["dx"])
    native = np.asarray(grid["x"], dtype=float)
    centre_coordinate = float(native[n // 2])
    phase_native = _phase_matrix(
        native,
        n=n,
        dx_m=dx_m,
        centre_coordinate_m=centre_coordinate,
    )
    phase_centre = _phase_matrix(
        np.asarray([centre_coordinate]),
        n=n,
        dx_m=dx_m,
        centre_coordinate_m=centre_coordinate,
    )[:, 0]
    transfer = np.exp(1j * np.asarray(kz_m_inv) * float(z_m))
    transfer *= bandlimit_mask_matsushima(
        dict(grid), float(wavelength_m), float(z_m), n_medium=float(n_medium)
    )
    synthesised = _synthesise_axis_fields(
        spectra,
        transfer,
        phase_requested=phase_native,
        phase_orthogonal=phase_centre,
        axis="x",
    )
    direct = tuple(
        ifft2c(np.asarray(spectrum) * transfer)[n // 2]
        for spectrum in spectra
    )
    synth_i = np.sum([np.abs(value) ** 2 for value in synthesised], axis=0)
    direct_i = np.sum([np.abs(value) ** 2 for value in direct], axis=0)
    scale = max(float(np.max(np.abs(direct_i))), EPS)
    return {
        "native_line_max_abs_intensity_error": float(
            np.max(np.abs(synth_i - direct_i)) / scale
        ),
        "native_line_intensity_correlation": _normalised_correlation(
            synth_i, direct_i
        ),
    }


def on_axis_spectral_intensity(
    *,
    grid: Mapping[str, Any],
    wavelength_m: float,
    z_values_m: Sequence[float],
    scalar_field: np.ndarray,
    n_medium: float = 1.0,
    bandlimit: bool = True,
) -> np.ndarray:
    """Evaluate scalar on-axis intensity without constructing transverse maps."""

    n = int(grid["N"])
    dx_m = float(grid["dx"])
    field = np.asarray(scalar_field, dtype=np.complex128)
    if field.shape != (n, n):
        raise ValueError("scalar field must match the declared square grid")
    spectrum = fft2c(field)
    kz = asm_longitudinal_wavenumber_m_inv(
        np.asarray(grid["FX"]),
        np.asarray(grid["FY"]),
        wavelength_m=float(wavelength_m) / float(n_medium),
        include_evanescent=True,
    )
    native_coordinates = np.asarray(grid["x"], dtype=float)
    centre_coordinate = float(native_coordinates[n // 2])
    phase_zero = _phase_matrix(
        np.asarray([0.0]),
        n=n,
        dx_m=dx_m,
        centre_coordinate_m=centre_coordinate,
    )[:, 0]
    values = np.empty(len(z_values_m), dtype=np.float64)
    for index, z_m in enumerate(z_values_m):
        transfer = np.exp(1j * kz * float(z_m))
        if bandlimit:
            transfer *= bandlimit_mask_matsushima(
                dict(grid),
                float(wavelength_m),
                float(z_m),
                n_medium=float(n_medium),
            )
        on_axis = phase_zero @ (spectrum * transfer) @ phase_zero / float(n * n)
        values[index] = float(np.abs(on_axis) ** 2)
    return values


def build_dense_spectral_propagation(
    *,
    grid: Mapping[str, Any],
    wavelength_m: float,
    z_values_m: Sequence[float],
    transverse_coordinates_m: Sequence[float],
    scalar_field: np.ndarray | None = None,
    projected_spectra: Sequence[np.ndarray] | None = None,
    kz_m_inv: np.ndarray | None = None,
    n_medium: float = 1.0,
    source_label: str,
) -> DensePropagationMap:
    """Evaluate scalar or projected-vector propagation on fixed physical lines."""

    if (scalar_field is None) == (projected_spectra is None):
        raise ValueError("provide exactly one of scalar_field or projected_spectra")
    n = int(grid["N"])
    dx_m = float(grid["dx"])
    if projected_spectra is None:
        spectra = (fft2c(np.asarray(scalar_field, dtype=np.complex128)),)
    else:
        spectra = tuple(np.asarray(value, dtype=np.complex128) for value in projected_spectra)
    if any(value.shape != (n, n) for value in spectra):
        raise ValueError("all propagation spectra must match the declared square grid")
    if kz_m_inv is None:
        kz = asm_longitudinal_wavenumber_m_inv(
            np.asarray(grid["FX"]),
            np.asarray(grid["FY"]),
            wavelength_m=float(wavelength_m) / float(n_medium),
            include_evanescent=True,
        )
    else:
        kz = np.asarray(kz_m_inv, dtype=np.complex128)
    coordinates = np.asarray(transverse_coordinates_m, dtype=float)
    z_values = np.asarray(z_values_m, dtype=float)
    if coordinates.ndim != 1 or z_values.ndim != 1:
        raise ValueError("transverse and z coordinates must be one-dimensional")
    if coordinates.size < 64 or z_values.size < 32:
        raise ValueError("dense propagation diagnostics require at least 64x32 samples")

    native_coordinates = np.asarray(grid["x"], dtype=float)
    centre_coordinate = float(native_coordinates[n // 2])
    phase_requested = _phase_matrix(
        coordinates,
        n=n,
        dx_m=dx_m,
        centre_coordinate_m=centre_coordinate,
    )
    phase_zero = _phase_matrix(
        np.asarray([0.0]),
        n=n,
        dx_m=dx_m,
        centre_coordinate_m=centre_coordinate,
    )[:, 0]
    xz = np.empty((z_values.size, coordinates.size), dtype=np.float64)
    yz = np.empty_like(xz)
    for index, z_m in enumerate(z_values):
        transfer = np.exp(1j * kz * float(z_m))
        transfer *= bandlimit_mask_matsushima(
            dict(grid), float(wavelength_m), float(z_m), n_medium=float(n_medium)
        )
        x_fields = _synthesise_axis_fields(
            spectra,
            transfer,
            phase_requested=phase_requested,
            phase_orthogonal=phase_zero,
            axis="x",
        )
        y_fields = _synthesise_axis_fields(
            spectra,
            transfer,
            phase_requested=phase_requested,
            phase_orthogonal=phase_zero,
            axis="y",
        )
        xz[index] = np.sum([np.abs(value) ** 2 for value in x_fields], axis=0)
        yz[index] = np.sum([np.abs(value) ** 2 for value in y_fields], axis=0)

    parity = native_line_parity(
        spectra,
        kz,
        grid,
        wavelength_m,
        z_m=float(z_values[np.argmin(np.abs(z_values - 60.0e-3))]),
        n_medium=n_medium,
    )
    return DensePropagationMap(
        x_m=coordinates,
        y_m=coordinates.copy(),
        z_m=z_values,
        xz_intensity=xz,
        yz_intensity=yz,
        metadata={
            "method": "direct centred inverse-DFT spectral-line synthesis",
            "propagator": "Matsushima band-limited angular spectrum",
            "source_label": source_label,
            "source_grid_n": n,
            "source_dx_m": dx_m,
            "transverse_samples": int(coordinates.size),
            "z_samples": int(z_values.size),
            "transverse_dx_m": float(np.mean(np.diff(coordinates))),
            "z_step_m": float(np.mean(np.diff(z_values))),
            "display_interpolation": "none",
            "metrics_computed_on_spectral_synthesis": True,
            **parity,
        },
    )


def map_correlation(a: DensePropagationMap, b: DensePropagationMap) -> dict[str, float]:
    """Compare two maps that share physical x/y/z coordinates."""

    if not (
        np.allclose(a.x_m, b.x_m)
        and np.allclose(a.y_m, b.y_m)
        and np.allclose(a.z_m, b.z_m)
    ):
        raise ValueError("dense propagation maps require matched coordinates")
    return {
        "xz_correlation": _normalised_correlation(a.xz_intensity, b.xz_intensity),
        "yz_correlation": _normalised_correlation(a.yz_intensity, b.yz_intensity),
        "xz_relative_l2": float(
            np.linalg.norm(a.xz_intensity / max(float(np.max(a.xz_intensity)), EPS) - b.xz_intensity / max(float(np.max(b.xz_intensity)), EPS))
            / max(float(np.linalg.norm(a.xz_intensity / max(float(np.max(a.xz_intensity)), EPS))), EPS)
        ),
        "yz_relative_l2": float(
            np.linalg.norm(a.yz_intensity / max(float(np.max(a.yz_intensity)), EPS) - b.yz_intensity / max(float(np.max(b.yz_intensity)), EPS))
            / max(float(np.linalg.norm(a.yz_intensity / max(float(np.max(a.yz_intensity)), EPS))), EPS)
        ),
    }
