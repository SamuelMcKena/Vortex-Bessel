"""Independent numerical references for propagation-figure auditing.

The production code contains several propagation representations because the
project spans scalar Bessel/vortex beams, scalable focus views and vector
hexagonal fields.  This module supplies deliberately simple, independent
references used to decide whether a longitudinal figure is trustworthy.

The main reference is centre zero-padding in the *spatial* domain followed by
unbandlimited angular-spectrum propagation on the physically larger window.
This reduces the artificial periodic-boundary interaction of an FFT propagation
without sharing the candidate's z-dependent support logic.  Comparisons are
made only in a declared central physical ROI.

For projected vector fields the padded reference reconstructs the z=0 complex
components, pads them, and re-projects the padded angular spectrum onto the
transverse subspace before propagation.  This prevents the reference from
silently treating the three field components as unrelated scalar waves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.equations.fields import fft2c, ifft2c, make_xy_grid
from vbb_study.equations.propagation import angular_spectrum_propagate_bl


EPS = np.finfo(float).tiny
TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class PropagationComparison:
    intensity_correlation: float
    normalised_relative_l2: float
    peak_ratio_candidate_to_reference: float
    roi_power_ratio_candidate_to_reference: float


def pad_center(field: np.ndarray, factor: int = 2) -> np.ndarray:
    arr = np.asarray(field, dtype=np.complex128)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("padding reference expects one square field")
    pf = int(factor)
    if pf < 1:
        raise ValueError("pad factor must be >=1")
    if pf == 1:
        return arr.copy()
    n = arr.shape[0]
    out = np.zeros((pf * n, pf * n), dtype=np.complex128)
    start = (out.shape[0] - n) // 2
    out[start : start + n, start : start + n] = arr
    return out


def crop_center(array: np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("centre crop expects one square array")
    nn = int(n)
    if nn > arr.shape[0]:
        raise ValueError("requested crop is larger than source")
    start = (arr.shape[0] - nn) // 2
    return arr[start : start + nn, start : start + nn]


def central_roi_mask(grid: Mapping[str, Any], halfwidth_m: float) -> np.ndarray:
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    return (np.abs(X) <= float(halfwidth_m)) & (np.abs(Y) <= float(halfwidth_m))


def compare_intensity_fields(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    roi_mask: np.ndarray,
    dx_m: float,
) -> PropagationComparison:
    a = np.abs(np.asarray(candidate, dtype=np.complex128)) ** 2
    b = np.abs(np.asarray(reference, dtype=np.complex128)) ** 2
    roi = np.asarray(roi_mask, dtype=bool)
    if a.shape != b.shape or a.shape != roi.shape:
        raise ValueError("candidate/reference/ROI shapes must match")
    aa = np.asarray(a[roi], dtype=float)
    bb = np.asarray(b[roi], dtype=float)
    ac = aa - float(np.mean(aa))
    bc = bb - float(np.mean(bb))
    correlation = float(np.dot(ac, bc) / max(np.linalg.norm(ac) * np.linalg.norm(bc), EPS))
    an = aa / max(float(np.max(aa)), EPS)
    bn = bb / max(float(np.max(bb)), EPS)
    rel_l2 = float(np.linalg.norm(an - bn) / max(np.linalg.norm(bn), EPS))
    pixel_area = float(dx_m) ** 2
    return PropagationComparison(
        intensity_correlation=correlation,
        normalised_relative_l2=rel_l2,
        peak_ratio_candidate_to_reference=float(np.max(aa) / max(float(np.max(bb)), EPS)),
        roi_power_ratio_candidate_to_reference=float(
            np.sum(aa) * pixel_area / max(float(np.sum(bb) * pixel_area), EPS)
        ),
    )


def scalar_padded_reference(
    field0: np.ndarray,
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    z_m: float,
    n_medium: float = 1.0,
    pad_factor: int = 2,
) -> np.ndarray:
    """Return central native-size field from a spatially padded ASM reference."""

    source = np.asarray(field0, dtype=np.complex128)
    n = int(grid["N"])
    padded = pad_center(source, int(pad_factor))
    padded_grid = make_xy_grid(int(pad_factor) * n, float(grid["dx"]))
    result = angular_spectrum_propagate_bl(
        padded,
        padded_grid,
        float(wavelength_m),
        float(z_m),
        n_medium=float(n_medium),
        bandlimit=False,
        include_evanescent=True,
    )
    return np.asarray(crop_center(result, n), dtype=np.complex128)


def _project_vector_spectrum(
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    n_medium: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k = TWOPI * float(n_medium) / float(wavelength_m)
    kx = TWOPI * np.asarray(grid["FX"], dtype=float)
    ky = TWOPI * np.asarray(grid["FY"], dtype=float)
    arg = k * k - kx * kx - ky * ky
    kz = np.where(
        arg >= 0.0,
        np.sqrt(np.maximum(arg, 0.0)),
        1j * np.sqrt(np.maximum(-arg, 0.0)),
    )
    ax = fft2c(ex)
    ay = fft2c(ey)
    az = fft2c(ez)
    sx = kx / max(k, EPS)
    sy = ky / max(k, EPS)
    sz = kz / max(k, EPS)
    dot = sx * ax + sy * ay + sz * az
    return ax - sx * dot, ay - sy * dot, az - sz * dot, kz


def vector_padded_reference_from_projected_spectra(
    prepared: Sequence[np.ndarray],
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    z_m: float,
    n_medium: float = 1.0,
    pad_factor: int = 2,
) -> np.ndarray:
    """Return vector intensity from an independently padded transverse reference.

    ``prepared`` is the production tuple ``(A_x,A_y,A_z,k_z)``.  Only the first
    three arrays are used to reconstruct the z=0 projected complex field.  The
    reconstructed components are spatially padded and projected again on the
    padded k-grid before propagation.
    """

    if len(prepared) != 4:
        raise ValueError("prepared vector propagation must contain Ax, Ay, Az, kz")
    ax0, ay0, az0, _ = (np.asarray(value, dtype=np.complex128) for value in prepared)
    n = int(grid["N"])
    if any(value.shape != (n, n) for value in (ax0, ay0, az0)):
        raise ValueError("prepared spectra do not match grid")
    ex0 = ifft2c(ax0)
    ey0 = ifft2c(ay0)
    ez0 = ifft2c(az0)
    pf = int(pad_factor)
    ex_p = pad_center(ex0, pf)
    ey_p = pad_center(ey0, pf)
    ez_p = pad_center(ez0, pf)
    padded_grid = make_xy_grid(pf * n, float(grid["dx"]))
    ax, ay, az, kz = _project_vector_spectrum(
        ex_p,
        ey_p,
        ez_p,
        padded_grid,
        wavelength_m=float(wavelength_m),
        n_medium=float(n_medium),
    )
    transfer = np.exp(1j * kz * float(z_m))
    ex = ifft2c(ax * transfer)
    ey = ifft2c(ay * transfer)
    ez = ifft2c(az * transfer)
    intensity = np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2
    return np.asarray(crop_center(intensity, n), dtype=float)


def compare_intensity_arrays(
    candidate_intensity: np.ndarray,
    reference_intensity: np.ndarray,
    *,
    roi_mask: np.ndarray,
    dx_m: float,
) -> PropagationComparison:
    """Intensity-only equivalent of :func:`compare_intensity_fields`."""

    a = np.sqrt(np.maximum(np.asarray(candidate_intensity, dtype=float), 0.0)).astype(np.complex128)
    b = np.sqrt(np.maximum(np.asarray(reference_intensity, dtype=float), 0.0)).astype(np.complex128)
    return compare_intensity_fields(a, b, roi_mask=roi_mask, dx_m=float(dx_m))


__all__ = [
    "PropagationComparison",
    "central_roi_mask",
    "compare_intensity_arrays",
    "compare_intensity_fields",
    "crop_center",
    "pad_center",
    "scalar_padded_reference",
    "vector_padded_reference_from_projected_spectra",
]
