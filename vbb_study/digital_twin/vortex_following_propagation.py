"""Beam-following dense propagation diagnostics for system-error figures.

A fixed laboratory x-z/y-z slice is useful for steering, but it is a poor beam
morphology diagnostic once the beam is translated: the orthogonal slice can
simply miss the beam and appear blank.  This module evaluates the same discrete
angular spectrum on transverse coordinates that follow a supplied x(z), y(z)
beam axis.  No image interpolation is used.

Two different centres are deliberately distinguished:

* intensity centroid: useful for power/steering bookkeeping;
* Bessel/vortex morphology axis: central peak for ell=0, phase singularity for
  a vortex transverse field, or the central dark channel between the nearest
  longitudinal lobes for ell!=0.

That distinction matters whenever decentre/truncation makes the rings
asymmetric: the energy centroid can move far away from the topological core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from vbb_study.equations.fields import fft2c
from vbb_study.equations.propagation import (
    asm_longitudinal_wavenumber_m_inv,
    bandlimit_mask_matsushima,
)


EPS = np.finfo(float).tiny
TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class BeamFollowingPropagationMap:
    transverse_offset_m: np.ndarray = field(repr=False, compare=False)
    z_m: np.ndarray = field(repr=False, compare=False)
    x_axis_m: np.ndarray = field(repr=False, compare=False)
    y_axis_m: np.ndarray = field(repr=False, compare=False)
    xz_intensity: np.ndarray = field(repr=False, compare=False)
    yz_intensity: np.ndarray = field(repr=False, compare=False)
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class TransverseMorphologyAxis:
    x_m: float
    y_m: float
    method: str
    detected_topological_charge: int
    selected_singularity_count: int
    distance_from_seed_m: float


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


def _field_line_x(
    spectrum: np.ndarray,
    transfer: np.ndarray,
    *,
    x_coordinates_m: np.ndarray,
    y_coordinate_m: float,
    n: int,
    dx_m: float,
    centre_coordinate_m: float,
) -> np.ndarray:
    phase_x = _phase_matrix(
        np.asarray(x_coordinates_m, dtype=float),
        n=n,
        dx_m=dx_m,
        centre_coordinate_m=centre_coordinate_m,
    )
    phase_y = _phase_matrix(
        np.asarray([float(y_coordinate_m)]),
        n=n,
        dx_m=dx_m,
        centre_coordinate_m=centre_coordinate_m,
    )[:, 0]
    propagated = np.asarray(spectrum, dtype=np.complex128) * transfer
    line_spectrum = phase_y @ propagated
    return np.asarray(line_spectrum @ phase_x / float(n * n))


def _field_line_y(
    spectrum: np.ndarray,
    transfer: np.ndarray,
    *,
    y_coordinates_m: np.ndarray,
    x_coordinate_m: float,
    n: int,
    dx_m: float,
    centre_coordinate_m: float,
) -> np.ndarray:
    phase_y = _phase_matrix(
        np.asarray(y_coordinates_m, dtype=float),
        n=n,
        dx_m=dx_m,
        centre_coordinate_m=centre_coordinate_m,
    )
    phase_x = _phase_matrix(
        np.asarray([float(x_coordinate_m)]),
        n=n,
        dx_m=dx_m,
        centre_coordinate_m=centre_coordinate_m,
    )[:, 0]
    propagated = np.asarray(spectrum, dtype=np.complex128) * transfer
    line_spectrum = propagated @ phase_x
    return np.asarray(line_spectrum @ phase_y / float(n * n))


def line_centroid_m(intensity: np.ndarray, coordinates_m: Sequence[float]) -> np.ndarray:
    """Intensity centroid of every z row in a longitudinal line map."""

    values = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    coordinate = np.asarray(coordinates_m, dtype=float)
    if values.ndim != 2 or values.shape[1] != coordinate.size:
        raise ValueError("line intensity shape does not match coordinates")
    power = np.sum(values, axis=1)
    centroid = np.sum(values * coordinate[None, :], axis=1) / np.maximum(power, EPS)
    return np.asarray(centroid, dtype=float)


def robust_axis_path_m(
    intensity: np.ndarray,
    coordinates_m: Sequence[float],
    *,
    peak_floor_fraction: float = 0.05,
) -> np.ndarray:
    """Track an energy centroid while avoiding noise-only z rows."""

    values = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    centroid = line_centroid_m(values, coordinates_m)
    peak = np.max(values, axis=1)
    valid = peak >= float(peak_floor_fraction) * max(float(np.max(peak)), EPS)
    index = np.arange(values.shape[0], dtype=float)
    if int(np.count_nonzero(valid)) < 2:
        return np.zeros(values.shape[0], dtype=float)
    filled = np.interp(index, index[valid], centroid[valid])
    return np.asarray(filled, dtype=float)


def _wrap_phase(delta: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * np.asarray(delta, dtype=float)))


def phase_winding_charge_map(field: np.ndarray) -> np.ndarray:
    """Integer phase winding on every native grid plaquette.

    The loop follows +x, +y, -x, -y, i.e. positive mathematical orientation
    for a grid whose y coordinate increases with row index.
    """

    phase = np.angle(np.asarray(field, dtype=np.complex128))
    p00 = phase[:-1, :-1]
    p01 = phase[:-1, 1:]
    p11 = phase[1:, 1:]
    p10 = phase[1:, :-1]
    winding = (
        _wrap_phase(p01 - p00)
        + _wrap_phase(p11 - p01)
        + _wrap_phase(p10 - p11)
        + _wrap_phase(p00 - p10)
    ) / TWOPI
    return np.rint(winding).astype(int)


def transverse_morphology_axis(
    field: np.ndarray,
    grid: Mapping[str, Any],
    *,
    vortex_charge: int,
    seed_x_m: float = 0.0,
    seed_y_m: float = 0.0,
    search_radius_m: float = 0.8e-3,
    b0_peak_search_radius_m: float = 0.35e-3,
) -> TransverseMorphologyAxis:
    """Locate the physical Bessel/vortex axis in one transverse complex field.

    For ``ell=0`` the central Bessel peak nearest the supplied seed is used.
    For ``ell!=0`` native-grid phase winding identifies topological
    singularities.  Same-sign singularities closest to the seed are accumulated
    until the requested total charge is reached; this naturally handles a
    higher-order core that has split into several unit-charge vortices.
    """

    u = np.asarray(field, dtype=np.complex128)
    x = np.asarray(grid["x"], dtype=float)
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    if u.shape != X.shape or X.shape != Y.shape:
        raise ValueError("field/grid shape mismatch")
    ell = int(vortex_charge)
    sx = float(seed_x_m)
    sy = float(seed_y_m)

    if ell == 0:
        intensity = np.abs(u) ** 2
        radius = np.hypot(X - sx, Y - sy)
        mask = radius <= float(b0_peak_search_radius_m)
        if not np.any(mask):
            raise ValueError("B0 peak-search mask is empty")
        masked = np.where(mask, intensity, -np.inf)
        iy, ix = np.unravel_index(int(np.argmax(masked)), masked.shape)
        px = float(x[ix])
        py = float(x[iy])
        return TransverseMorphologyAxis(
            x_m=px,
            y_m=py,
            method="central_intensity_peak_near_seed",
            detected_topological_charge=0,
            selected_singularity_count=0,
            distance_from_seed_m=float(math_hypot(px - sx, py - sy)),
        )

    charge = phase_winding_charge_map(u)
    cell_x = 0.5 * (x[:-1] + x[1:])
    cell_y = 0.5 * (x[:-1] + x[1:])
    CX, CY = np.meshgrid(cell_x, cell_y, indexing="xy")
    sign = 1 if ell > 0 else -1
    candidate = (charge * sign) > 0
    candidate &= np.hypot(CX - sx, CY - sy) <= float(search_radius_m)
    iy, ix = np.nonzero(candidate)
    if iy.size == 0:
        raise RuntimeError(
            f"no same-sign phase singularity found within {search_radius_m:g} m of seed"
        )
    q = charge[iy, ix]
    cx = CX[iy, ix]
    cy = CY[iy, ix]
    distance = np.hypot(cx - sx, cy - sy)
    order = np.argsort(distance)

    selected: list[int] = []
    total = 0
    target = abs(ell)
    for idx in order:
        selected.append(int(idx))
        total += abs(int(q[idx]))
        if total >= target:
            break
    qsel = np.abs(q[selected]).astype(float)
    px = float(np.sum(qsel * cx[selected]) / np.sum(qsel))
    py = float(np.sum(qsel * cy[selected]) / np.sum(qsel))
    detected = int(np.sum(q[selected]))
    if abs(detected) < target:
        raise RuntimeError(
            f"detected topological charge {detected} does not recover requested ell={ell}"
        )
    return TransverseMorphologyAxis(
        x_m=px,
        y_m=py,
        method="phase_winding_topological_core",
        detected_topological_charge=detected,
        selected_singularity_count=len(selected),
        distance_from_seed_m=float(math_hypot(px - sx, py - sy)),
    )


def math_hypot(x: float, y: float) -> float:
    return float(np.hypot(float(x), float(y)))


def bessel_feature_axis_path_m(
    intensity: np.ndarray,
    coordinates_m: Sequence[float],
    *,
    vortex_charge: int,
    peak_floor_fraction: float = 0.05,
    search_halfwidth_m: float = 0.16e-3,
) -> np.ndarray:
    """Track the central Bessel feature instead of the energy centroid.

    ``ell=0`` follows the strongest local central peak near the energy-centroid
    seed.  ``ell!=0`` finds the nearest bright lobe on either side of that seed
    and tracks the dark minimum between them, which is the 1-D intersection of
    the vortex/Bessel core with the longitudinal slice.
    """

    values = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    coordinate = np.asarray(coordinates_m, dtype=float)
    if values.ndim != 2 or values.shape[1] != coordinate.size:
        raise ValueError("line intensity shape does not match coordinates")
    seed = line_centroid_m(values, coordinate)
    peak_trace = np.max(values, axis=1)
    active = peak_trace >= float(peak_floor_fraction) * max(float(np.max(peak_trace)), EPS)
    result = np.full(values.shape[0], np.nan, dtype=float)
    ell = int(vortex_charge)

    step = float(np.median(np.diff(coordinate)))
    smooth_sigma = max(0.8, 2.0e-6 / max(abs(step), EPS))
    for iz, row in enumerate(values):
        if not active[iz]:
            continue
        smooth = gaussian_filter1d(row, sigma=smooth_sigma, mode="nearest")
        local = np.abs(coordinate - seed[iz]) <= float(search_halfwidth_m)
        indices = np.flatnonzero(local)
        if indices.size < 9:
            continue
        segment = smooth[indices]
        if ell == 0:
            # Prefer a peak nearest the centroid seed rather than a distant side
            # lobe that happens to be marginally brighter.
            peaks, _ = find_peaks(segment)
            if peaks.size == 0:
                chosen = indices[int(np.argmax(segment))]
            else:
                candidates = indices[peaks]
                score = np.abs(coordinate[candidates] - seed[iz])
                chosen = int(candidates[int(np.argmin(score))])
            result[iz] = coordinate[chosen]
            continue

        peaks, props = find_peaks(segment, prominence=0.01 * max(float(np.max(segment)), EPS))
        if peaks.size < 2:
            continue
        candidates = indices[peaks]
        left = candidates[coordinate[candidates] < seed[iz]]
        right = candidates[coordinate[candidates] > seed[iz]]
        if left.size == 0 or right.size == 0:
            continue
        il = int(left[np.argmin(np.abs(coordinate[left] - seed[iz]))])
        ir = int(right[np.argmin(np.abs(coordinate[right] - seed[iz]))])
        if ir <= il + 1:
            continue
        core_index = il + int(np.argmin(smooth[il : ir + 1]))
        result[iz] = coordinate[core_index]

    valid = np.isfinite(result) & active
    index = np.arange(values.shape[0], dtype=float)
    if int(np.count_nonzero(valid)) < 2:
        # A hard fallback to the energy path is preferable to inventing a core,
        # but callers can record that the feature tracker could not resolve it.
        return robust_axis_path_m(
            values,
            coordinate,
            peak_floor_fraction=peak_floor_fraction,
        )
    return np.interp(index, index[valid], result[valid])


def build_beam_following_propagation(
    *,
    grid: Mapping[str, Any],
    wavelength_m: float,
    z_values_m: Sequence[float],
    transverse_offsets_m: Sequence[float],
    scalar_field: np.ndarray,
    x_axis_m: Sequence[float] | float = 0.0,
    y_axis_m: Sequence[float] | float = 0.0,
    n_medium: float = 1.0,
    source_label: str,
) -> BeamFollowingPropagationMap:
    """Evaluate x-z and y-z maps in coordinates relative to a tracked axis.

    For every z plane the x-z line is sampled at ``y=y_axis(z)`` over
    ``x=x_axis(z)+offset``.  The y-z line is sampled analogously at
    ``x=x_axis(z)``.  This is a direct Fourier-series field synthesis from the
    angular spectrum and therefore does not shift/interpolate rendered images.
    """

    n = int(grid["N"])
    dx_m = float(grid["dx"])
    field = np.asarray(scalar_field, dtype=np.complex128)
    if field.shape != (n, n):
        raise ValueError("scalar field must match the declared square grid")
    z = np.asarray(z_values_m, dtype=float)
    offsets = np.asarray(transverse_offsets_m, dtype=float)
    if z.ndim != 1 or offsets.ndim != 1:
        raise ValueError("z and transverse offsets must be one-dimensional")
    if z.size < 16 or offsets.size < 64:
        raise ValueError("beam-following diagnostics require at least 64x16 samples")

    def _path(value: Sequence[float] | float) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(z.size, float(arr), dtype=float)
        if arr.shape != z.shape:
            raise ValueError("axis path must be scalar or match z_values_m")
        return arr

    x_axis = _path(x_axis_m)
    y_axis = _path(y_axis_m)
    spectrum = fft2c(field)
    kz = asm_longitudinal_wavenumber_m_inv(
        np.asarray(grid["FX"]),
        np.asarray(grid["FY"]),
        wavelength_m=float(wavelength_m) / float(n_medium),
        include_evanescent=True,
    )
    native = np.asarray(grid["x"], dtype=float)
    centre_coordinate = float(native[n // 2])

    xz = np.empty((z.size, offsets.size), dtype=float)
    yz = np.empty_like(xz)
    for iz, zz in enumerate(z):
        transfer = np.exp(1j * kz * float(zz))
        transfer *= bandlimit_mask_matsushima(
            dict(grid), float(wavelength_m), float(zz), n_medium=float(n_medium)
        )
        x_line = _field_line_x(
            spectrum,
            transfer,
            x_coordinates_m=x_axis[iz] + offsets,
            y_coordinate_m=float(y_axis[iz]),
            n=n,
            dx_m=dx_m,
            centre_coordinate_m=centre_coordinate,
        )
        y_line = _field_line_y(
            spectrum,
            transfer,
            y_coordinates_m=y_axis[iz] + offsets,
            x_coordinate_m=float(x_axis[iz]),
            n=n,
            dx_m=dx_m,
            centre_coordinate_m=centre_coordinate,
        )
        xz[iz] = np.abs(x_line) ** 2
        yz[iz] = np.abs(y_line) ** 2

    return BeamFollowingPropagationMap(
        transverse_offset_m=offsets,
        z_m=z,
        x_axis_m=x_axis,
        y_axis_m=y_axis,
        xz_intensity=xz,
        yz_intensity=yz,
        metadata={
            "outcome": "BEAM-FOLLOWING-DENSE-SPECTRAL-PROPAGATION",
            "source_label": str(source_label),
            "sampling_model": "direct_Fourier_series_no_image_interpolation",
            "axis_model": "supplied_xz_yz_path",
        },
    )


__all__ = [
    "BeamFollowingPropagationMap",
    "TransverseMorphologyAxis",
    "bessel_feature_axis_path_m",
    "build_beam_following_propagation",
    "line_centroid_m",
    "phase_winding_charge_map",
    "robust_axis_path_m",
    "transverse_morphology_axis",
]
