"""Stage 7 intensity metrics and controls for the vector-arm hexagon study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import map_coordinates

from vbb_study.viz_fields import _azimuthal_power_spectrum

EPS = 1.0e-30
TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class H6Result:
    """Order-6 dominance result for one transverse plane."""

    h6: float
    order6_power: float
    non_dc_power: float
    profile: np.ndarray
    power: np.ndarray


def sample_ring_profile(
    intensity: np.ndarray,
    grid: Mapping[str, Any],
    radius_m: float,
    *,
    angular_samples: int = 4096,
    radial_samples: int = 5,
    radial_half_width_m: float | None = None,
) -> np.ndarray:
    """Sample mean intensity around an annulus by bilinear interpolation."""

    arr = np.asarray(intensity, dtype=float)
    x = np.asarray(grid["x"], dtype=float)
    dx = float(grid["dx"])
    x0 = float(x[0])
    half_width = float(radial_half_width_m) if radial_half_width_m is not None else 2.0 * dx
    radii = np.linspace(float(radius_m) - half_width, float(radius_m) + half_width, int(radial_samples))
    theta = np.linspace(0.0, TWOPI, int(angular_samples), endpoint=False)
    samples = []
    for radius in radii:
        cols = (radius * np.cos(theta) - x0) / dx
        rows = (radius * np.sin(theta) - x0) / dx
        samples.append(map_coordinates(arr, [rows, cols], order=1, mode="nearest"))
    return np.mean(np.asarray(samples, dtype=float), axis=0)


def h6_from_profile(profile: np.ndarray) -> H6Result:
    """Return order-6 dominance using the repo's validated azimuthal FFT."""

    prof = np.asarray(profile, dtype=float)
    power = _azimuthal_power_spectrum(prof)
    order6 = float(power[6]) if power.size > 6 else 0.0
    non_dc = float(np.sum(power[1:]))
    h6 = order6 / max(non_dc, EPS)
    return H6Result(h6=float(h6), order6_power=order6, non_dc_power=non_dc, profile=prof, power=power)


def h6_from_intensity(
    intensity: np.ndarray,
    grid: Mapping[str, Any],
    radius_m: float,
    *,
    angular_samples: int = 4096,
) -> H6Result:
    """Return H6 from a 2D intensity plane sampled on a ring."""

    return h6_from_profile(sample_ring_profile(intensity, grid, radius_m, angular_samples=angular_samples))


def h6_z_curve(
    intensity_stack: np.ndarray,
    z_values_m: Sequence[float],
    grid: Mapping[str, Any],
    radius_m: float,
    *,
    angular_samples: int = 2048,
) -> dict[str, Any]:
    """Return full H6(z), not just the best plane."""

    stack = np.asarray(intensity_stack, dtype=float)
    h6_values = []
    order6 = []
    non_dc = []
    for plane in stack:
        result = h6_from_intensity(plane, grid, radius_m, angular_samples=angular_samples)
        h6_values.append(result.h6)
        order6.append(result.order6_power)
        non_dc.append(result.non_dc_power)
    h6_arr = np.asarray(h6_values, dtype=float)
    best = int(np.nanargmax(h6_arr)) if h6_arr.size else 0
    return {
        "z_m": np.asarray(z_values_m, dtype=float),
        "z_um": np.asarray(z_values_m, dtype=float) / 1e-6,
        "h6": h6_arr,
        "order6_power": np.asarray(order6, dtype=float),
        "non_dc_power": np.asarray(non_dc, dtype=float),
        "best_index": best,
        "best_h6": float(h6_arr[best]) if h6_arr.size else 0.0,
        "best_z_m": float(np.asarray(z_values_m, dtype=float)[best]) if h6_arr.size else 0.0,
    }


def proposed_h6_threshold(ideal_max_h6: float, *, fraction: float = 0.6) -> float:
    """Return a recorded proposal threshold derived from ideal-limit H6."""

    return float(fraction) * float(ideal_max_h6)


def assert_three_sided_acceptance(
    segmented_h6_curve: Mapping[str, Any],
    radial_control_h6_curve: Mapping[str, Any],
    scalar_control_h6_curve: Mapping[str, Any],
    threshold: float,
) -> None:
    """Assert segmented signal passes and both controls fail."""

    segmented = np.asarray(segmented_h6_curve["h6"], dtype=float)
    radial = np.asarray(radial_control_h6_curve["h6"], dtype=float)
    scalar = np.asarray(scalar_control_h6_curve["h6"], dtype=float)
    assert float(np.nanmax(segmented)) >= float(threshold)
    assert np.all(radial < 0.25 * float(threshold))
    assert np.all(scalar < 0.25 * float(threshold))


def synthetic_ring_profile(n: int = 4096) -> np.ndarray:
    """Return a perfect circular-ring angular profile."""

    return np.ones(int(n), dtype=float)


def synthetic_lattice_profile(n_spots: int = 6, n: int = 4096) -> np.ndarray:
    """Return a discrete equally spaced n-spot angular lattice profile."""

    profile = np.zeros(int(n), dtype=float)
    for idx in np.linspace(0, int(n), int(n_spots), endpoint=False, dtype=int):
        profile[idx % int(n)] = 1.0
    return profile


def lattice_artifact_ratio(profile: np.ndarray, order: int = 6) -> float:
    """Return existing A4-style ``power[order] / power[0]`` lattice flag."""

    power = _azimuthal_power_spectrum(np.asarray(profile, dtype=float))
    if int(order) >= power.size:
        return 0.0
    return float(power[int(order)] / max(float(power[0]), EPS))


def assert_degenerate_rejection(threshold: float, *, n: int = 4096) -> dict[str, float]:
    """Assert a perfect ring fails H6 and a discrete lattice is flagged."""

    ring = synthetic_ring_profile(n)
    lattice = synthetic_lattice_profile(6, n)
    ring_h6 = h6_from_profile(ring).h6
    lattice_h6 = h6_from_profile(lattice).h6
    ring_lattice_ratio = lattice_artifact_ratio(ring, 6)
    lattice_ratio = lattice_artifact_ratio(lattice, 6)
    assert ring_h6 < float(threshold)
    assert ring_lattice_ratio < 0.01
    assert lattice_ratio >= 0.5
    return {
        "ring_h6": float(ring_h6),
        "lattice_h6": float(lattice_h6),
        "ring_lattice_ratio": float(ring_lattice_ratio),
        "lattice_artifact_ratio": float(lattice_ratio),
    }


__all__ = [
    "H6Result",
    "assert_degenerate_rejection",
    "assert_three_sided_acceptance",
    "h6_from_intensity",
    "h6_from_profile",
    "h6_z_curve",
    "lattice_artifact_ratio",
    "proposed_h6_threshold",
    "sample_ring_profile",
    "synthetic_lattice_profile",
    "synthetic_ring_profile",
]
