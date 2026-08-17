"""Physically seeded longitudinal Bessel/vortex morphology tracking.

Energy centroids are useful for power/steering bookkeeping but are not a robust
beam-axis definition for asymmetric Bessel or vortex-Bessel fields.  In
particular, an asymmetric outer lobe can move the centroid far from the central
vortex channel.

This module tracks the central feature by continuity along z:

* ell=0: nearest resolved local maximum to the previous central-peak position;
* ell!=0: dark minimum between an adjacent pair of resolved bright lobes,
  selecting the candidate whose core remains closest to the previous core.

The first active z row is anchored to a supplied physical seed.  For an axicon
lateral-decentre sweep that seed is the imposed decentre coordinate; for
nominal/tip/material sweeps it is zero.  This prevents the algorithm from
silently redefining the physical axis using the intensity centroid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


EPS = np.finfo(float).tiny


@dataclass(frozen=True)
class LongitudinalAxisTrack:
    coordinate_m: np.ndarray
    detected_mask: np.ndarray
    method: str
    seed_coordinate_m: float
    detected_fraction: float
    maximum_detected_step_m: float


def _candidate_peaks(
    row: np.ndarray,
    coordinate_m: np.ndarray,
    *,
    expected_m: float,
    search_halfwidth_m: float,
    smooth_sigma_pixels: float,
    prominence_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    smooth = gaussian_filter1d(
        np.maximum(np.asarray(row, dtype=float), 0.0),
        sigma=float(smooth_sigma_pixels),
        mode="nearest",
    )
    local = np.abs(coordinate_m - float(expected_m)) <= float(search_halfwidth_m)
    indices = np.flatnonzero(local)
    if indices.size < 9:
        return np.asarray([], dtype=int), smooth
    segment = smooth[indices]
    prominence = float(prominence_fraction) * max(float(np.max(segment)), EPS)
    peaks, _ = find_peaks(segment, prominence=prominence)
    return indices[peaks], smooth


def _vortex_core_candidates(
    smooth: np.ndarray,
    coordinate_m: np.ndarray,
    peak_indices: np.ndarray,
) -> list[tuple[float, float, int, int]]:
    """Return (core, pair_midpoint, left_index, right_index) candidates."""

    ordered = np.asarray(sorted(map(int, peak_indices)), dtype=int)
    candidates: list[tuple[float, float, int, int]] = []
    for left, right in zip(ordered[:-1], ordered[1:]):
        if right <= left + 1:
            continue
        core_index = left + int(np.argmin(smooth[left : right + 1]))
        core = float(coordinate_m[core_index])
        midpoint = 0.5 * float(coordinate_m[left] + coordinate_m[right])
        candidates.append((core, midpoint, int(left), int(right)))
    return candidates


def track_bessel_feature_axis(
    intensity: np.ndarray,
    coordinates_m: Sequence[float],
    *,
    vortex_charge: int,
    seed_coordinate_m: float = 0.0,
    peak_floor_fraction: float = 0.05,
    search_halfwidth_m: float = 0.18e-3,
    prominence_fraction: float = 0.01,
    smoothing_scale_m: float = 2.0e-6,
    maximum_step_m: float = 45e-6,
) -> LongitudinalAxisTrack:
    """Track the central Bessel/vortex feature continuously along z.

    The detector walks through active z rows in order.  The previous accepted
    core is the prediction for the next row.  Candidate peaks are restricted to
    a local physical search window around that prediction.

    For a vortex, *every adjacent bright-peak pair* in that window is examined;
    the dark minimum between each pair is a core candidate.  The chosen pair
    minimises a continuity score based predominantly on the core position, with
    a smaller penalty on the pair midpoint.  This avoids the common failure
    where an energy-centroid seed selects the valley between a vortex side lobe
    and an unrelated broad outer lobe.
    """

    values = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    coordinate = np.asarray(coordinates_m, dtype=float)
    if values.ndim != 2 or values.shape[1] != coordinate.size:
        raise ValueError("line intensity shape does not match coordinates")
    if coordinate.ndim != 1 or coordinate.size < 32:
        raise ValueError("coordinates must be a sufficiently sampled 1-D array")
    spacing = np.diff(coordinate)
    if not np.all(spacing > 0.0):
        raise ValueError("coordinates must increase monotonically")

    peak_trace = np.max(values, axis=1)
    active = peak_trace >= float(peak_floor_fraction) * max(float(np.max(peak_trace)), EPS)
    result = np.full(values.shape[0], np.nan, dtype=float)
    detected = np.zeros(values.shape[0], dtype=bool)
    ell = int(vortex_charge)
    previous = float(seed_coordinate_m)
    step_m = float(np.median(spacing))
    smooth_sigma_pixels = max(0.8, float(smoothing_scale_m) / step_m)

    for iz, row in enumerate(values):
        if not active[iz]:
            continue
        peak_indices, smooth = _candidate_peaks(
            row,
            coordinate,
            expected_m=previous,
            search_halfwidth_m=float(search_halfwidth_m),
            smooth_sigma_pixels=smooth_sigma_pixels,
            prominence_fraction=float(prominence_fraction),
        )
        if peak_indices.size == 0:
            continue

        if ell == 0:
            positions = coordinate[peak_indices]
            candidate = float(positions[int(np.argmin(np.abs(positions - previous)))])
        else:
            candidates = _vortex_core_candidates(smooth, coordinate, peak_indices)
            if not candidates:
                continue
            # Core continuity dominates.  Pair-midpoint continuity is a
            # secondary discriminator when two minima are similarly close.
            scores = [
                abs(core - previous) + 0.20 * abs(midpoint - previous)
                for core, midpoint, _, _ in candidates
            ]
            candidate = float(candidates[int(np.argmin(scores))][0])

        if detected.any() and abs(candidate - previous) > float(maximum_step_m):
            # Reject an implausible one-plane jump rather than silently jumping
            # to a different ring/core.  Missing rows are interpolated only
            # after the physically continuous detections are established.
            continue
        result[iz] = candidate
        detected[iz] = True
        previous = candidate

    valid = detected & active & np.isfinite(result)
    if int(np.count_nonzero(valid)) < 2:
        raise RuntimeError(
            "Bessel/vortex central feature could not be resolved continuously; "
            "do not replace it with an energy-centroid axis"
        )

    index = np.arange(values.shape[0], dtype=float)
    filled = np.interp(index, index[valid], result[valid])
    accepted = result[valid]
    max_detected_step = (
        float(np.max(np.abs(np.diff(accepted)))) if accepted.size >= 2 else 0.0
    )
    return LongitudinalAxisTrack(
        coordinate_m=np.asarray(filled, dtype=float),
        detected_mask=np.asarray(valid, dtype=bool),
        method=(
            "continuous_central_peak_from_physical_seed"
            if ell == 0
            else "continuous_dark_core_between_adjacent_bessel_lobes_from_physical_seed"
        ),
        seed_coordinate_m=float(seed_coordinate_m),
        detected_fraction=float(np.mean(valid[active])) if np.any(active) else 0.0,
        maximum_detected_step_m=max_detected_step,
    )


__all__ = ["LongitudinalAxisTrack", "track_bessel_feature_axis"]
