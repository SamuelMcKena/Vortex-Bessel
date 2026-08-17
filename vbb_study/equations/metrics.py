"""Engine-compatible radial and axial region metric adapters."""

from __future__ import annotations

from typing import Any

import numpy as np

from vbb_study import vbb_metrics

EPS = 1e-30
um = 1e-6


def extract_radial_metrics(
    I: np.ndarray,
    grid: dict[str, Any],
    ell: int,
    kr: float,
) -> dict[str, Any]:
    """Extract core/ring radius and width from a transverse intensity plane.

    I delegate to `vbb_study.vbb_metrics` so the legacy peak-plane API and the
    new per-z metric engine share one implementation.
    """

    metrics = vbb_metrics.peak_plane_radial_metrics(I, grid, int(ell), float(kr), center_mode="centroid")
    keep = [
        "r_profile_m",
        "radial_profile_norm",
        "radial_profile_smooth",
        "feature_radius_m",
        "feature_diameter_m",
        "ring_radius_m",
        "ring_diameter_m",
        "core_radius_m",
        "core_radius_definition",
        "core_hwhm_radius_m",
        "core_hwhm_diameter_m",
        "core_first_zero_radius_m",
        "core_first_zero_diameter_m",
        "diameter_m",
        "ring_width_m",
        "r_half_inner_m",
        "r_half_outer_m",
    ]
    return {key: metrics[key] for key in keep}


def _contiguous_mask_zone(
    z: np.ndarray,
    mask: np.ndarray,
    *,
    reference_index: int,
    prefix: str,
) -> dict[str, float | bool]:
    """Return the contiguous mask segment around a reference z plane.

    I keep this deliberately simple and sample-based because the combined
    Bessel-region criteria mix different observables; interpolating a hard
    boolean intersection would imply precision the simulation has not earned.
    """

    z = np.asarray(z, dtype=float)
    ok = np.asarray(mask, dtype=bool)
    if z.size == 0 or ok.size != z.size or not np.any(ok):
        return {
            f"{prefix}_um": 0.0,
            f"{prefix}_start_um": np.nan,
            f"{prefix}_end_um": np.nan,
            f"{prefix}_capped": False,
        }
    ref = int(np.clip(reference_index, 0, z.size - 1))
    if not ok[ref]:
        true_idx = np.flatnonzero(ok)
        ref = int(true_idx[np.argmin(np.abs(true_idx - ref))])
    i0 = ref
    while i0 > 0 and ok[i0 - 1]:
        i0 -= 1
    i1 = ref
    while i1 < z.size - 1 and ok[i1 + 1]:
        i1 += 1
    return {
        f"{prefix}_um": float(max(0.0, z[i1] - z[i0]) / um),
        f"{prefix}_start_um": float(z[i0] / um),
        f"{prefix}_end_um": float(z[i1] / um),
        f"{prefix}_capped": bool(i0 == 0 or i1 == z.size - 1),
    }


def bessel_region_metrics(
    volume: dict[str, Any],
    design: Any,
    *,
    peak_level: float = 0.5,
    feature_power_level: float = 0.5,
    radius_tolerance: float = 0.15,
) -> dict[str, float | bool]:
    """Measure the strict Bessel region from peak, power, and radius stability.

    The canonical `bessel_zone_um` is a FWHM of the peak-in-plane trace. That is
    a useful scalar observable, but it can overstate a useful writing region if the
    ring power drops or the ring/core radius drifts. Here I form three per-z
    masks and report their intersection:

    - peak-in-plane >= `peak_level` of its maximum;
    - fixed-bucket ring/core power >= `feature_power_level` of its maximum;
    - ring/core radius remains within `radius_tolerance` of the reference plane.

    The result is exposed as `bessel_region_um`; `strict_bessel_region_um` is
    added as an explicit alias by `extract_vortex_safe_metrics`.
    """

    z = np.asarray(volume.get("z", []), dtype=float)
    peak = np.asarray(volume.get("peak", []), dtype=float)
    if z.size == 0 or peak.size != z.size or "intensity_stack" not in volume:
        return {
            "bessel_region_um": 0.0,
            "bessel_region_start_um": np.nan,
            "bessel_region_end_um": np.nan,
            "bessel_region_capped": False,
            "radius_stability_zone_um": 0.0,
            "feature_power_zone_um": 0.0,
            "bessel_region_peak_level": float(peak_level),
            "bessel_region_power_level": float(feature_power_level),
            "bessel_region_radius_tolerance": float(radius_tolerance),
        }

    ref = int(volume.get("peak_index", int(np.nanargmax(peak))))
    per_z = vbb_metrics.per_z_metrics_from_volume(
        volume,
        ell=design.ell,
        kr_m_inv=design.kr_sample_m_inv,
        center_mode="centroid",
        reference_index=ref,
    )
    peak_norm = peak / (float(np.nanmax(peak)) + EPS)
    ell_abs = abs(int(design.ell))
    feature_power = np.asarray(per_z["core_power" if ell_abs == 0 else "ring_power"], dtype=float)
    feature_power_norm = feature_power / (float(np.nanmax(feature_power)) + EPS)
    feature_radius = np.asarray(per_z["core_radius_m" if ell_abs == 0 else "ring_radius_m"], dtype=float)
    ref_radius = float(feature_radius[int(np.clip(ref, 0, len(feature_radius) - 1))])
    if ref_radius <= EPS:
        positive = feature_radius[np.isfinite(feature_radius) & (feature_radius > EPS)]
        ref_radius = float(np.nanmedian(positive)) if positive.size else 1.0
    radius_drift = np.abs(feature_radius - ref_radius) / (abs(ref_radius) + EPS)

    peak_mask = np.isfinite(peak_norm) & (peak_norm >= float(peak_level))
    power_mask = np.isfinite(feature_power_norm) & (feature_power_norm >= float(feature_power_level))
    radius_mask = np.isfinite(radius_drift) & (radius_drift <= float(radius_tolerance))
    combined = peak_mask & power_mask & radius_mask

    region = _contiguous_mask_zone(z, combined, reference_index=ref, prefix="bessel_region")
    power_zone = _contiguous_mask_zone(z, power_mask, reference_index=ref, prefix="feature_power_zone")
    radius_zone = _contiguous_mask_zone(z, radius_mask, reference_index=ref, prefix="radius_stability_zone")
    in_region = combined
    return {
        **region,
        **power_zone,
        **radius_zone,
        "bessel_region_peak_level": float(peak_level),
        "bessel_region_power_level": float(feature_power_level),
        "bessel_region_radius_tolerance": float(radius_tolerance),
        "ring_or_core_power_fraction_min_in_region": float(np.nanmin(feature_power_norm[in_region])) if np.any(in_region) else np.nan,
        "ring_or_core_radius_drift_fraction_max_in_region": float(np.nanmax(radius_drift[in_region])) if np.any(in_region) else np.nan,
        "ring_or_core_radius_drift_fraction_at_peak": float(radius_drift[int(np.clip(ref, 0, len(radius_drift) - 1))]),
    }
