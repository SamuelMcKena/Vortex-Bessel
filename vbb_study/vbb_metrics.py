"""Per-z scalar beam metrics for the vortex-Bessel-beam study.

I keep the radial and axial metric definitions here so the notebooks, the
physics engine, and the tests all measure the same thing. The central idea is
simple: reduce each transverse plane to an azimuthally averaged radial profile,
extract the feature radii from that profile, then use fixed reference buckets
for power bookkeeping along z.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import special as sp

try:  # SciPy is available in the study venv, but I keep a graceful fallback.
    from scipy.ndimage import gaussian_filter1d
except Exception:  # pragma: no cover - exercised only in stripped-down envs.
    gaussian_filter1d = None

EPS = 1.0e-15


@dataclass(frozen=True)
class RadialBinning:
    """Description of the radial bins used for azimuthal averages.

    The bins are linear in radius and stop at the largest complete circle that
    fits inside the sampled image for the chosen beam centre. That avoids
    partial-annulus bias near the grid edges, which otherwise looks like a
    physical side-lobe change.
    """

    centers_m: np.ndarray
    edges_m: np.ndarray
    counts: np.ndarray
    r_max_m: float


def grid_xy(grid: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    """Return x/y mesh coordinates and a square-grid spacing in metres."""

    if "X" in grid and "Y" in grid:
        X = np.asarray(grid["X"], dtype=float)
        Y = np.asarray(grid["Y"], dtype=float)
    else:
        x = np.asarray(grid["x"], dtype=float)
        y = np.asarray(grid.get("y", x), dtype=float)
        X, Y = np.meshgrid(x, y, indexing="xy")
    if "dx" in grid:
        dx = float(grid["dx"])
    else:
        xs = np.asarray(grid["x"], dtype=float)
        dx = float(np.median(np.diff(xs)))
    return X, Y, dx


def estimate_beam_center(
    intensity: np.ndarray,
    grid: Mapping[str, Any],
    *,
    mode: str = "centroid",
) -> tuple[float, float]:
    """Estimate the beam centre in metres.

    `centroid` is the default because it follows modestly off-centre beams while
    still behaving well for symmetric vortex rings. `origin` is useful when I
    want to force comparison with an analytically centred grid.
    """

    mode_l = str(mode).lower().strip()
    X, Y, _ = grid_xy(grid)
    if mode_l == "origin":
        return (0.0, 0.0)
    if mode_l == "peak":
        iy, ix = np.unravel_index(int(np.nanargmax(intensity)), intensity.shape)
        return (float(X[iy, ix]), float(Y[iy, ix]))
    if mode_l != "centroid":
        raise ValueError(f"Unknown centre-estimation mode: {mode}")
    I = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    total = float(np.sum(I))
    if total <= EPS:
        return (0.0, 0.0)
    return (float(np.sum(X * I) / total), float(np.sum(Y * I) / total))


def inscribed_r_max(grid: Mapping[str, Any], center_m: tuple[float, float]) -> float:
    """Largest complete annulus radius supported by the finite grid."""

    X, Y, dx = grid_xy(grid)
    x0, y0 = center_m
    x_min = float(np.min(X))
    x_max = float(np.max(X))
    y_min = float(np.min(Y))
    y_max = float(np.max(Y))
    r_max = min(x0 - x_min, x_max - x0, y0 - y_min, y_max - y0)
    return max(float(r_max), 2.0 * dx)


def radial_binning(
    grid: Mapping[str, Any],
    *,
    center_m: tuple[float, float] = (0.0, 0.0),
    bins: int | None = None,
    r_max_m: float | None = None,
) -> RadialBinning:
    """Build linear radial bins and count pixels in each bin."""

    X, Y, dx = grid_xy(grid)
    r_max = inscribed_r_max(grid, center_m) if r_max_m is None else float(r_max_m)
    if bins is None:
        bins = max(32, int(np.floor(r_max / max(dx, EPS))))
    bins = max(8, int(bins))
    edges = np.linspace(0.0, r_max, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    R = np.hypot(X - center_m[0], Y - center_m[1])
    idx = np.digitize(R.ravel(), edges) - 1
    valid = (idx >= 0) & (idx < bins)
    counts = np.bincount(idx[valid], minlength=bins).astype(int)
    return RadialBinning(centers_m=centers, edges_m=edges, counts=counts, r_max_m=r_max)


def azimuthal_average(
    intensity: np.ndarray,
    grid: Mapping[str, Any],
    *,
    center_m: tuple[float, float] | None = None,
    center_mode: str = "centroid",
    bins: int | None = None,
    r_max_m: float | None = None,
    min_count: int = 4,
) -> dict[str, np.ndarray | float | tuple[float, float]]:
    """Compute an azimuthal mean radial profile and radial power bins.

    Pixels are assigned to linear radial bins around the estimated centre. The
    profile is an unweighted mean per bin; the power bins are area-weighted by
    the actual pixel count and `dx**2`.
    """

    I = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    if I.ndim != 2:
        raise ValueError("azimuthal_average expects one 2D intensity plane.")
    center = estimate_beam_center(I, grid, mode=center_mode) if center_m is None else tuple(map(float, center_m))
    X, Y, dx = grid_xy(grid)
    binning = radial_binning(grid, center_m=center, bins=bins, r_max_m=r_max_m)
    R = np.hypot(X - center[0], Y - center[1])
    idx = np.digitize(R.ravel(), binning.edges_m) - 1
    valid = (idx >= 0) & (idx < len(binning.centers_m))
    sums = np.bincount(idx[valid], weights=I.ravel()[valid], minlength=len(binning.centers_m))
    counts = np.bincount(idx[valid], minlength=len(binning.centers_m))
    profile = np.full(len(binning.centers_m), np.nan, dtype=float)
    good = counts >= int(min_count)
    profile[good] = sums[good] / counts[good]
    bin_power = sums * dx * dx
    total_power = float(np.sum(I) * dx * dx)
    encircled = np.cumsum(bin_power) / (total_power + EPS)
    return {
        "r_m": binning.centers_m,
        "r_edges_m": binning.edges_m,
        "profile": profile,
        "bin_power": bin_power,
        "encircled_energy_fraction": encircled,
        "counts": counts.astype(int),
        "center_m": center,
        "r_max_m": float(binning.r_max_m),
        "total_power": total_power,
    }


def _smooth_profile(profile: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    arr = np.asarray(profile, dtype=float)
    if gaussian_filter1d is None or sigma <= 0:
        return arr
    filled = np.array(arr, copy=True)
    finite = np.isfinite(filled)
    if not np.any(finite):
        return filled
    filled[~finite] = np.interp(np.flatnonzero(~finite), np.flatnonzero(finite), filled[finite])
    return gaussian_filter1d(filled, sigma=float(sigma), mode="nearest")


def _interpolate_crossing(r0: float, y0: float, r1: float, y1: float, level: float) -> float:
    if not np.isfinite(y0) or not np.isfinite(y1) or abs(y1 - y0) <= EPS:
        return float(0.5 * (r0 + r1))
    t = (level - y0) / (y1 - y0)
    return float(r0 + np.clip(t, 0.0, 1.0) * (r1 - r0))


def _half_max_bounds(r: np.ndarray, profile: np.ndarray, peak_idx: int, half: float) -> tuple[float, float]:
    left = int(peak_idx)
    while left > 0 and np.isfinite(profile[left]) and profile[left] >= half:
        left -= 1
    if left == peak_idx:
        r_inner = float(r[peak_idx])
    elif left <= 0 and profile[left] >= half:
        r_inner = 0.0
    else:
        r_inner = _interpolate_crossing(float(r[left]), float(profile[left]), float(r[left + 1]), float(profile[left + 1]), half)

    right = int(peak_idx)
    while right < len(profile) - 1 and np.isfinite(profile[right]) and profile[right] >= half:
        right += 1
    if right == peak_idx:
        r_outer = float(r[peak_idx])
    elif right >= len(profile) - 1 and profile[right] >= half:
        r_outer = float(r[right])
    else:
        r_outer = _interpolate_crossing(float(r[right - 1]), float(profile[right - 1]), float(r[right]), float(profile[right]), half)
    return (float(max(0.0, r_inner)), float(max(r_outer, r_inner)))


def radial_feature_metrics(
    intensity: np.ndarray,
    grid: Mapping[str, Any],
    *,
    ell: int,
    kr_m_inv: float,
    center_m: tuple[float, float] | None = None,
    center_mode: str = "centroid",
    bins: int | None = None,
    r_max_m: float | None = None,
    smoothing_sigma: float = 1.0,
) -> dict[str, Any]:
    """Measure the core/ring feature from one transverse intensity plane.

    For `abs(ell)>0`, I search near the first `J_l'` zero and measure the
    annular HWHM radii. For `ell=0`, I measure the bright-core HWHM radius.
    The J0 first-zero radius is reported separately because it is a design
    scale, not the same observable as the HWHM bright-core metric.
    """

    avg = azimuthal_average(
        intensity,
        grid,
        center_m=center_m,
        center_mode=center_mode,
        bins=bins,
        r_max_m=r_max_m,
    )
    r = np.asarray(avg["r_m"], dtype=float)
    profile = np.asarray(avg["profile"], dtype=float)
    peak_norm = float(np.nanmax(profile)) if np.any(np.isfinite(profile)) else 0.0
    profile_norm = profile / (peak_norm + EPS)
    profile_smooth = _smooth_profile(profile_norm, sigma=smoothing_sigma)
    ell_abs = abs(int(ell))
    core_first_zero_radius = float(sp.jn_zeros(0, 1)[0] / max(float(kr_m_inv), EPS))
    finite = np.isfinite(profile_smooth)
    if not np.any(finite):
        raise ValueError("Cannot measure radial metrics from an empty radial profile.")

    if ell_abs == 0:
        search = finite & (r <= max(3.0 / max(float(kr_m_inv), EPS), r[min(len(r) - 1, max(3, len(r) // 5))]))
        if not np.any(search):
            search = finite
        idxs = np.flatnonzero(search)
        peak_idx = int(idxs[int(np.nanargmax(profile_smooth[idxs]))])
        half = 0.5 * float(profile_smooth[peak_idx])
        _, r_outer = _half_max_bounds(r, profile_smooth, peak_idx, half)
        core_radius = float(r_outer)
        core_hwhm_radius = core_radius
        r_inner = 0.0
        ring_radius = 0.0
        ring_width = 2.0 * core_radius
        feature_radius = core_radius
        core_radius_definition = "ell0_bright_core_hwhm_radius"
    else:
        # Vortex beams have a dark center. Their feature radius is the bright
        # annular peak near the first zero of J'_ell, while r_inner/r_outer
        # describe the HWHM annulus around that peak.
        predicted = float(sp.jnp_zeros(ell_abs, 1)[0] / max(float(kr_m_inv), EPS))
        search = finite & (r >= 0.45 * predicted) & (r <= 1.8 * predicted)
        if not np.any(search):
            search = finite
        idxs = np.flatnonzero(search)
        peak_idx = int(idxs[int(np.nanargmax(profile_smooth[idxs]))])
        half = 0.5 * float(profile_smooth[peak_idx])
        r_inner, r_outer = _half_max_bounds(r, profile_smooth, peak_idx, half)
        ring_radius = float(r[peak_idx])
        core_radius = float(r_inner)
        core_hwhm_radius = np.nan
        ring_width = float(max(0.0, r_outer - r_inner))
        feature_radius = ring_radius
        core_radius_definition = "legacy_vortex_inner_hwhm_radius"

    return {
        "r_profile_m": r,
        "radial_profile_norm": profile_norm,
        "radial_profile_smooth": profile_smooth,
        "feature_radius_m": float(feature_radius),
        "feature_diameter_m": float(2.0 * feature_radius),
        "ring_radius_m": float(ring_radius),
        "ring_diameter_m": float(2.0 * ring_radius),
        "core_radius_m": float(core_radius),
        "core_radius_definition": core_radius_definition,
        "core_hwhm_radius_m": float(core_hwhm_radius),
        "core_hwhm_diameter_m": float(2.0 * core_hwhm_radius) if np.isfinite(core_hwhm_radius) else np.nan,
        "core_first_zero_radius_m": core_first_zero_radius,
        "core_first_zero_diameter_m": float(2.0 * core_first_zero_radius),
        "diameter_m": float(2.0 * feature_radius),
        "ring_width_m": float(ring_width),
        "r_half_inner_m": float(r_inner),
        "r_half_outer_m": float(r_outer),
        "center_x_m": float(avg["center_m"][0]),
        "center_y_m": float(avg["center_m"][1]),
        "r_max_m": float(avg["r_max_m"]),
        "encircled_energy_fraction": np.asarray(avg["encircled_energy_fraction"], dtype=float),
        "radial_bin_power": np.asarray(avg["bin_power"], dtype=float),
        "total_power": float(avg["total_power"]),
    }


def _as_stack(planes: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
    arr = np.asarray(planes, dtype=float)
    if arr.ndim == 2:
        return arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError("Expected a 2D plane or a 3D stack with shape (z, y, x).")
    return arr


def per_z_metrics_from_stack(
    planes: np.ndarray | Sequence[np.ndarray],
    grid: Mapping[str, Any],
    *,
    ell: int,
    kr_m_inv: float,
    z_m: Sequence[float] | None = None,
    bins: int | None = None,
    center_mode: str = "centroid",
    reference_index: int | None = None,
    side_lobe_exclusion_radius_factor: float = 1.6,
    smoothing_sigma: float = 1.0,
) -> dict[str, Any]:
    """Return per-z metric arrays for a transverse intensity stack.

    The returned `ring_power` is the canonical fixed-reference bucket power:
    the annulus is defined from the peak plane and then reused at every z. The
    `local_ring_power` array uses each plane's local HWHM annulus and is kept as
    a morphology diagnostic, not the canonical axial power metric.
    """

    stack = _as_stack(planes)
    nz = int(stack.shape[0])
    z = np.arange(nz, dtype=float) if z_m is None else np.asarray(z_m, dtype=float)
    if len(z) != nz:
        raise ValueError("z_m length must match the number of planes.")

    centers = [estimate_beam_center(stack[i], grid, mode=center_mode) for i in range(nz)]
    common_r_max = min(inscribed_r_max(grid, c) for c in centers)
    per_plane = [
        radial_feature_metrics(
            stack[i],
            grid,
            ell=ell,
            kr_m_inv=kr_m_inv,
            center_m=centers[i],
            bins=bins,
            r_max_m=common_r_max,
            smoothing_sigma=smoothing_sigma,
        )
        for i in range(nz)
    ]

    peak_in_plane = np.asarray([float(np.nanmax(stack[i])) for i in range(nz)], dtype=float)
    ref = int(np.nanargmax(peak_in_plane)) if reference_index is None else int(reference_index)
    ref = max(0, min(ref, nz - 1))
    ell_abs = abs(int(ell))
    ref_inner = float(per_plane[ref]["r_half_inner_m"])
    ref_outer = float(per_plane[ref]["r_half_outer_m"])
    if ell_abs == 0:
        ref_inner = 0.0
        ref_outer = float(per_plane[ref]["core_radius_m"])
    X, Y, dx = grid_xy(grid)

    total_power = np.zeros(nz, dtype=float)
    core_power = np.zeros(nz, dtype=float)
    ring_power = np.zeros(nz, dtype=float)
    side_lobe_power = np.zeros(nz, dtype=float)
    local_ring_power = np.zeros(nz, dtype=float)
    contrast = np.full(nz, np.nan, dtype=float)
    visibility = np.full(nz, np.nan, dtype=float)
    side_lobe_ratio = np.full(nz, np.nan, dtype=float)

    for i in range(nz):
        I = np.maximum(stack[i], 0.0)
        R = np.hypot(X - centers[i][0], Y - centers[i][1])
        total = float(np.sum(I) * dx * dx)
        total_power[i] = total
        if ell_abs == 0:
            core_mask = R <= ref_outer
            ring_mask = np.zeros_like(core_mask, dtype=bool)
        else:
            core_mask = R < ref_inner
            ring_mask = (R >= ref_inner) & (R <= ref_outer)
        core_power[i] = float(np.sum(I[core_mask]) * dx * dx)
        ring_power[i] = float(np.sum(I[ring_mask]) * dx * dx) if np.any(ring_mask) else 0.0
        side_lobe_power[i] = max(0.0, total_power[i] - core_power[i] - ring_power[i])

        local_inner = float(per_plane[i]["r_half_inner_m"])
        local_outer = float(per_plane[i]["r_half_outer_m"])
        if ell_abs > 0:
            local_mask = (R >= local_inner) & (R <= local_outer)
            local_ring_power[i] = float(np.sum(I[local_mask]) * dx * dx)
            feature_peak = float(np.max(I[ring_mask])) if np.any(ring_mask) else float(np.nanmax(I))
            central_radius = max(0.5 * ref_inner, dx)
            central_mask = R <= central_radius
            core_mean = float(np.mean(I[central_mask])) if np.any(central_mask) else 0.0
            contrast[i] = 1.0 - core_mean / (feature_peak + EPS)
            visibility[i] = (feature_peak - core_mean) / (feature_peak + core_mean + EPS)
            side_mask = R >= side_lobe_exclusion_radius_factor * max(float(per_plane[ref]["feature_radius_m"]), dx)
            side_peak = float(np.max(I[side_mask])) if np.any(side_mask) else np.nan
            side_lobe_ratio[i] = side_peak / (feature_peak + EPS) if np.isfinite(side_peak) else np.nan
        else:
            local_ring_power[i] = 0.0
            feature_peak = float(np.max(I[core_mask])) if np.any(core_mask) else float(np.nanmax(I))
            side_mask = R >= side_lobe_exclusion_radius_factor * max(float(per_plane[ref]["feature_radius_m"]), dx)
            side_peak = float(np.max(I[side_mask])) if np.any(side_mask) else np.nan
            side_lobe_ratio[i] = side_peak / (feature_peak + EPS) if np.isfinite(side_peak) else np.nan

    keys = [
        "ring_radius_m",
        "ring_diameter_m",
        "core_radius_m",
        "core_hwhm_radius_m",
        "core_hwhm_diameter_m",
        "core_first_zero_radius_m",
        "core_first_zero_diameter_m",
        "feature_radius_m",
        "feature_diameter_m",
        "diameter_m",
        "r_half_inner_m",
        "r_half_outer_m",
        "ring_width_m",
        "center_x_m",
        "center_y_m",
        "r_max_m",
    ]
    out = {key: np.asarray([float(p[key]) for p in per_plane], dtype=float) for key in keys}
    out.update(
        {
            "z_m": z,
            "reference_index": ref,
            "reference_r_inner_m": ref_inner,
            "reference_r_outer_m": ref_outer,
            "r_profile_m": np.asarray(per_plane[0]["r_profile_m"], dtype=float),
            "radial_profile_norm": np.vstack([np.asarray(p["radial_profile_norm"], dtype=float) for p in per_plane]),
            "radial_profile_smooth": np.vstack([np.asarray(p["radial_profile_smooth"], dtype=float) for p in per_plane]),
            "encircled_energy_fraction": np.vstack([np.asarray(p["encircled_energy_fraction"], dtype=float) for p in per_plane]),
            "radial_bin_power": np.vstack([np.asarray(p["radial_bin_power"], dtype=float) for p in per_plane]),
            "peak_in_plane": peak_in_plane,
            "total_power": total_power,
            "core_power": core_power,
            "ring_power": ring_power,
            "side_lobe_power": side_lobe_power,
            "local_ring_power": local_ring_power,
            "contrast": contrast,
            "visibility": visibility,
            "side_lobe_ratio": side_lobe_ratio,
            "center_mode": center_mode,
            "ell": int(ell),
            "kr_m_inv": float(kr_m_inv),
        }
    )
    return out


def per_z_metrics_from_volume(
    volume: Mapping[str, Any],
    *,
    ell: int,
    kr_m_inv: float,
    bins: int | None = None,
    center_mode: str = "centroid",
    reference_index: int | None = None,
    side_lobe_exclusion_radius_factor: float = 1.6,
    smoothing_sigma: float = 1.0,
) -> dict[str, Any]:
    """Return per-z metrics from a `bt.propagate_volume` result."""

    if "intensity_stack" not in volume:
        raise ValueError(
            "volume does not contain intensity_stack. Re-run propagation with the Stage 2 bessel_twin_core.propagate_volume."
        )
    return per_z_metrics_from_stack(
        np.asarray(volume["intensity_stack"], dtype=float),
        volume["crop_grid"],
        ell=ell,
        kr_m_inv=kr_m_inv,
        z_m=volume.get("z"),
        bins=bins,
        center_mode=center_mode,
        reference_index=reference_index,
        side_lobe_exclusion_radius_factor=side_lobe_exclusion_radius_factor,
        smoothing_sigma=smoothing_sigma,
    )


def peak_plane_radial_metrics(
    intensity: np.ndarray,
    grid: Mapping[str, Any],
    ell: int,
    kr_m_inv: float,
    *,
    bins: int | None = None,
    center_mode: str = "centroid",
    smoothing_sigma: float = 1.0,
) -> dict[str, Any]:
    """Compatibility wrapper for the old peak-plane radial metric function."""

    return radial_feature_metrics(
        intensity,
        grid,
        ell=ell,
        kr_m_inv=kr_m_inv,
        bins=bins,
        center_mode=center_mode,
        smoothing_sigma=smoothing_sigma,
    )


__all__ = [
    "EPS",
    "RadialBinning",
    "azimuthal_average",
    "estimate_beam_center",
    "grid_xy",
    "inscribed_r_max",
    "peak_plane_radial_metrics",
    "per_z_metrics_from_stack",
    "per_z_metrics_from_volume",
    "radial_binning",
    "radial_feature_metrics",
]
