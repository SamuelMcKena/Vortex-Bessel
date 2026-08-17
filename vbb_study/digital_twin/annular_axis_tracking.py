"""Stage 8C.3R.2 robust annular beam-axis tracking.

For a vortex / Bessel annulus the global intensity maximum is azimuthally
degenerate around the bright ring: the brightest-pixel position can jump to a
different point on the same ring between runs.  Raw brightest-pixel position is
therefore **not a valid primary beam-axis metric for annular fields** and must
not drive steering validation, FOV-convergence classification, or alignment
interpretation.

This module provides multiple candidate centre estimators with explicit
confidence/reliability, and a documented decision hierarchy:

  1. primary   : fitted annular ring centre (good ring-fit quality)
  2. secondary : fitted dark-core centre (core sufficiently dark/defined)
  3. secondary : central-ROI intensity centroid (ring/core fitting unreliable)
  4. optional  : phase-singularity estimate (complex field available)
  5. never primary : raw brightest-pixel position on an annular ring

Free-space reference-plane study, n = 1.0; optical/fluence diagnostic only;
``final_export_allowed=False``.  No material model is involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates

# Reliability thresholds (documented in docs/35).
RING_FIT_QUALITY_RELIABLE = 0.70
CORE_DARKNESS_RELIABLE = 0.55
PHASE_NULL_DEPTH_RELIABLE = 0.25

RAW_PEAK_LABEL = "not_a_primary_axis_metric_for_annular_fields"
UNRELIABLE_AXIS_LABEL = "axis_estimate_unreliable"


@dataclass(frozen=True)
class AnnularAxisEstimate:
    """All candidate centre estimates for one transverse plane + the selected axis."""

    data: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------


def _pixel_coords(vals_um: np.ndarray, axis_um: np.ndarray) -> np.ndarray:
    x0 = float(axis_um[0]); dx = float(axis_um[1] - axis_um[0])
    return (np.asarray(vals_um, float) - x0) / dx


def _intensity_centroid(plane: np.ndarray, x: np.ndarray, y: np.ndarray,
                        mask: np.ndarray | None = None) -> tuple[float, float]:
    w = plane if mask is None else plane * mask
    s = float(np.sum(w))
    if s <= 0:
        return 0.0, 0.0
    X, Y = np.meshgrid(x, y)
    return float(np.sum(w * X) / s), float(np.sum(w * Y) / s)


def _border_power_fraction(plane: np.ndarray, border_px: int = 2) -> float:
    total = float(np.sum(plane))
    if total <= 0:
        return 1.0
    b = max(1, int(border_px))
    edge = np.zeros_like(plane, dtype=bool)
    edge[:b, :] = True
    edge[-b:, :] = True
    edge[:, :b] = True
    edge[:, -b:] = True
    return float(np.sum(plane[edge]) / total)


def _radial_profile_ring_radius(plane: np.ndarray, x: np.ndarray, y: np.ndarray,
                                cx: float, cy: float) -> float:
    """Dominant ring radius from the azimuthally-averaged radial profile."""
    X, Y = np.meshgrid(x, y)
    r = np.hypot(X - cx, Y - cy)
    dx = float(abs(x[1] - x[0]))
    rmax = 0.5 * min(float(x.max() - x.min()), float(y.max() - y.min()))
    bins = np.arange(0.0, rmax, dx)
    if bins.size < 3:
        return dx
    idx = np.clip((r / dx).astype(int), 0, bins.size - 1)
    prof = np.bincount(idx.ravel(), weights=plane.ravel(), minlength=bins.size)
    cnt = np.bincount(idx.ravel(), minlength=bins.size)
    prof = prof / np.maximum(cnt, 1)
    # ignore the innermost bin (core) when locating the ring
    k = int(np.argmax(prof[1:])) + 1
    return float(bins[k]) if k < bins.size else dx


def _sample_rays(plane: np.ndarray, x: np.ndarray, y: np.ndarray,
                 centres: np.ndarray, thetas: np.ndarray, radii_um: np.ndarray) -> np.ndarray:
    """Bilinear-sample intensity for a stack of candidate centres.

    Returns array [n_centres, n_theta, n_radii].
    """
    nC = centres.shape[0]; nT = thetas.size; nR = radii_um.size
    ct = np.cos(thetas)[None, :, None]; st = np.sin(thetas)[None, :, None]
    rr = radii_um[None, None, :]
    xs = centres[:, 0][:, None, None] + rr * ct          # microns
    ys = centres[:, 1][:, None, None] + rr * st
    cols = _pixel_coords(xs.ravel(), x)
    rows = _pixel_coords(ys.ravel(), y)
    vals = map_coordinates(plane, [rows, cols], order=1, mode="constant", cval=0.0)
    return vals.reshape(nC, nT, nR)


def _ring_fit(plane: np.ndarray, x: np.ndarray, y: np.ndarray,
              c0: tuple[float, float]) -> dict[str, Any]:
    """Robust ring centre: choose the centre giving the most circular, most
    azimuthally-uniform bright ring."""
    dx = float(abs(x[1] - x[0]))
    r_ring = _radial_profile_ring_radius(plane, x, y, c0[0], c0[1])
    r_ring = max(r_ring, 2.0 * dx)
    radii = np.linspace(0.4 * r_ring, 1.8 * r_ring, 24)
    thetas = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    best_centre = np.asarray(c0, dtype=float)
    best_payload: tuple[np.ndarray, np.ndarray, np.ndarray, int] | None = None
    search = max(0.8 * r_ring, 3.0 * dx)
    for step in (max(dx, search / 6.0), max(0.25 * dx, search / 18.0)):
        offs = np.arange(-search, search + 1e-9, step)
        gx, gy = np.meshgrid(best_centre[0] + offs, best_centre[1] + offs)
        centres = np.column_stack([gx.ravel(), gy.ravel()])
        samp = _sample_rays(plane, x, y, centres, thetas, radii)  # [nC,nT,nR]
        peak_per_angle = samp.max(axis=2)                         # [nC,nT]
        r_at_peak = radii[samp.argmax(axis=2)]                    # [nC,nT]
        mean_amp = peak_per_angle.mean(axis=1)
        amp_unif = 1.0 - peak_per_angle.std(axis=1) / np.maximum(mean_amp, 1e-30)
        rad_spread = r_at_peak.std(axis=1) / np.maximum(r_at_peak.mean(axis=1), 1e-30)
        score = (
            np.clip(amp_unif, 0.0, 1.0)
            * np.clip(1.0 - rad_spread, 0.0, 1.0)
            * (mean_amp / max(mean_amp.max(), 1e-30))
        )
        best = int(np.argmax(score))
        best_centre = centres[best]
        best_payload = (peak_per_angle, r_at_peak, amp_unif, best)
        search = max(1.25 * dx, 2.5 * step)
    if best_payload is None:
        best_centre = np.asarray(c0, dtype=float)
        peak_per_angle = np.zeros((1, thetas.size))
        r_at_peak = np.full((1, thetas.size), r_ring)
        amp_unif = np.zeros(1)
        best = 0
    else:
        peak_per_angle, r_at_peak, amp_unif, best = best_payload
    rad_spread = r_at_peak.std(axis=1) / np.maximum(r_at_peak.mean(axis=1), 1e-30)
    cx, cy = float(best_centre[0]), float(best_centre[1])
    quality = float(np.clip(np.clip(amp_unif[best], 0, 1) * np.clip(1.0 - rad_spread[best], 0, 1), 0.0, 1.0))
    if quality >= 0.75:
        # Sub-pixel ring-centre correction.  If the trial centre is displaced,
        # the fitted radius has a first-harmonic wobble around the azimuth.
        # Remove that wobble without using the arbitrary brightest ring pixel.
        rr = r_at_peak[best]
        corr_x = float(2.0 * np.mean(rr * np.cos(thetas)))
        corr_y = float(2.0 * np.mean(rr * np.sin(thetas)))
        max_corr = 1.5 * dx
        corr_norm = float(np.hypot(corr_x, corr_y))
        if corr_norm <= max_corr:
            cx += corr_x
            cy += corr_y
    return {
        "ring_centre_x_um": cx, "ring_centre_y_um": cy,
        "ring_radius_um": float(r_at_peak[best].mean()),
        "ring_fit_quality": quality,
        "ring_fit_method": "azimuthal_uniformity_radius_constancy_search",
        "ring_fit_reliable": bool(quality >= RING_FIT_QUALITY_RELIABLE),
        "azimuthal_uniformity": float(np.clip(amp_unif[best], 0.0, 1.0)),
        "ring_circularity": float(np.clip(1.0 - rad_spread[best], 0.0, 1.0)),
    }


def _core_fit(plane: np.ndarray, x: np.ndarray, y: np.ndarray,
              ring_centre: tuple[float, float], ring_radius: float) -> dict[str, Any]:
    """Dark-core centre from the genuinely-dark pixels inside the ring."""
    X, Y = np.meshgrid(x, y)
    r = np.hypot(X - ring_centre[0], Y - ring_centre[1])
    pk = float(np.max(plane))
    inside = r <= 0.85 * max(ring_radius, abs(x[1] - x[0]))
    core_region = r <= 0.35 * max(ring_radius, abs(x[1] - x[0]))
    core_fill = float(np.mean(plane[core_region]) / pk) if (pk > 0 and np.any(core_region)) else 1.0
    darkness_contrast = float(np.clip(1.0 - core_fill, 0.0, 1.0))
    # weight by how dark each inside-pixel is (only genuinely dark pixels)
    dark_w = np.clip(0.3 * pk - plane, 0.0, None) * inside
    if float(np.sum(dark_w)) > 0:
        cx = float(np.sum(dark_w * X) / np.sum(dark_w))
        cy = float(np.sum(dark_w * Y) / np.sum(dark_w))
    else:
        cx, cy = ring_centre
    return {
        "core_centre_x_um": cx, "core_centre_y_um": cy,
        "core_fit_quality": darkness_contrast,
        "core_fill_fraction": core_fill,
        "central_darkness_contrast": darkness_contrast,
        "core_fit_reliable": bool(darkness_contrast >= CORE_DARKNESS_RELIABLE),
    }


def _phase_singularity(field: np.ndarray, x: np.ndarray, y: np.ndarray,
                       roi_radius_um: float) -> dict[str, Any]:
    """Amplitude-null / phase-singularity estimate (optional diagnostic)."""
    amp = np.abs(field)
    X, Y = np.meshgrid(x, y)
    roi = np.hypot(X, Y) <= roi_radius_um
    if not np.any(roi):
        return {"phase_singularity_x_um": float("nan"), "phase_singularity_y_um": float("nan"),
                "phase_singularity_reliable": False, "phase_null_depth": float("nan")}
    amp_roi = np.where(roi, amp, np.inf)
    iy, ix = np.unravel_index(int(np.argmin(amp_roi)), amp.shape)
    null_depth = float(amp[iy, ix] / max(np.max(amp), 1e-30))
    return {
        "phase_singularity_x_um": float(x[ix]),
        "phase_singularity_y_um": float(y[iy]),
        "phase_null_depth": null_depth,
        "phase_singularity_reliable": bool(null_depth <= PHASE_NULL_DEPTH_RELIABLE),
    }


# ---------------------------------------------------------------------------
# public estimator
# ---------------------------------------------------------------------------


def estimate_annular_axis(
    plane: np.ndarray,
    x_um: np.ndarray,
    y_um: np.ndarray,
    *,
    field: np.ndarray | None = None,
    roi_radius_um: float | None = None,
) -> AnnularAxisEstimate:
    """Return all candidate centre estimates and the selected primary axis."""
    plane = np.asarray(plane, float)
    x = np.asarray(x_um, float); y = np.asarray(y_um, float)
    c0 = _intensity_centroid(plane, x, y)

    ring = _ring_fit(plane, x, y, c0)
    core = _core_fit(plane, x, y, (ring["ring_centre_x_um"], ring["ring_centre_y_um"]), ring["ring_radius_um"])
    roi_r = float(roi_radius_um) if roi_radius_um is not None else 1.5 * max(ring["ring_radius_um"], abs(x[1] - x[0]))
    X, Y = np.meshgrid(x, y)
    roi_mask = np.hypot(X, Y) <= roi_r
    roi_cx, roi_cy = _intensity_centroid(plane, x, y, roi_mask)
    roi_power = float(np.sum(plane[roi_mask])) if np.any(roi_mask) else 0.0
    roi_reliable = bool(np.isfinite(roi_cx) and np.isfinite(roi_cy) and roi_power > 0.0)

    iy, ix = np.unravel_index(int(np.argmax(plane)), plane.shape)
    brightest = (float(x[ix]), float(y[iy]))

    phase = (_phase_singularity(field, x, y, roi_r) if field is not None else
             {"phase_singularity_x_um": float("nan"), "phase_singularity_y_um": float("nan"),
              "phase_singularity_reliable": False, "phase_null_depth": float("nan")})

    ring_margin_x = min(
        ring["ring_centre_x_um"] - float(x.min()),
        float(x.max()) - ring["ring_centre_x_um"],
    ) - float(ring["ring_radius_um"])
    ring_margin_y = min(
        ring["ring_centre_y_um"] - float(y.min()),
        float(y.max()) - ring["ring_centre_y_um"],
    ) - float(ring["ring_radius_um"])
    fov_margin = float(min(ring_margin_x, ring_margin_y))
    out_of_frame = _border_power_fraction(plane)
    crop_limited = bool(out_of_frame > 0.02 or fov_margin < 0.0)

    # decision hierarchy.  The raw brightest annular pixel is never selected as
    # the primary axis, even when every fit is unreliable.
    if ring["ring_fit_reliable"]:
        axis = (ring["ring_centre_x_um"], ring["ring_centre_y_um"]); method = "ring_fit"
        quality = ring["ring_fit_quality"]; reliability = "reliable"
    elif core["core_fit_reliable"]:
        axis = (core["core_centre_x_um"], core["core_centre_y_um"]); method = "core_fit"
        quality = core["core_fit_quality"]; reliability = "reliable"
    elif roi_reliable:
        axis = (roi_cx, roi_cy); method = "roi_centroid"
        quality = ring["ring_fit_quality"]
        reliability = ("caution" if quality > 0.35 and not crop_limited else UNRELIABLE_AXIS_LABEL)
    elif phase["phase_singularity_reliable"]:
        axis = (phase["phase_singularity_x_um"], phase["phase_singularity_y_um"]); method = "phase_singularity"
        quality = float(1.0 - min(float(phase["phase_null_depth"]), 1.0))
        reliability = "caution"
    else:
        axis = (float("nan"), float("nan")); method = "no_reliable_axis"
        quality = ring["ring_fit_quality"]
        reliability = UNRELIABLE_AXIS_LABEL
    if crop_limited and reliability == "reliable":
        reliability = "caution_crop_limited"

    data = {
        "commanded_axis_x_um": 0.0, "commanded_axis_y_um": 0.0,
        **ring, **core,
        "roi_intensity_centroid_x_um": roi_cx, "roi_intensity_centroid_y_um": roi_cy,
        "roi_centroid_reliable": roi_reliable,
        "roi_radius_um": roi_r,
        **phase,
        "brightest_pixel_x_um": brightest[0], "brightest_pixel_y_um": brightest[1],
        "brightest_pixel_status": RAW_PEAK_LABEL,
        "brightest_annular_pixel_location": brightest,
        "field_of_view_margin_um": fov_margin,
        "out_of_frame_fraction": out_of_frame,
        "beam_axis_x_um": float(axis[0]), "beam_axis_y_um": float(axis[1]),
        "beam_axis_method": method,
        "beam_axis_fit_quality": float(quality),
        "beam_axis_reliability": reliability,
        "beam_axis_error_um": float(np.hypot(axis[0], axis[1])),
    }
    return AnnularAxisEstimate(data)


# ---------------------------------------------------------------------------
# trajectory tracking (robust)
# ---------------------------------------------------------------------------


def track_axis_trajectory(
    intensity_zyx: np.ndarray,
    x_um: np.ndarray,
    y_um: np.ndarray,
    z_um: np.ndarray,
    *,
    min_relative_power: float = 0.3,
    estimator_mode: str = "auto",
) -> dict[str, Any]:
    """Per-plane annular axis estimate + robust straight-axis fit.

    Planes that are too weak, out of frame, or too deformed for a reliable centre
    are rejected from the fit (``valid_plane_fraction`` reports how many remain).
    """
    I = np.asarray(intensity_zyx, float)
    x = np.asarray(x_um, float); y = np.asarray(y_um, float); z = np.asarray(z_um, float)
    nz = I.shape[0]
    plane_power = I.sum(axis=(1, 2))
    strong = plane_power > min_relative_power * float(plane_power.max())

    bx = np.full(nz, np.nan); by = np.full(nz, np.nan)
    methods: list[str] = []; reliab: list[str] = []
    for i in range(nz):
        est = estimate_annular_axis(I[i], x, y)
        if estimator_mode == "ring_fit":
            cx, cy, ok = est["ring_centre_x_um"], est["ring_centre_y_um"], est["ring_fit_reliable"]
        elif estimator_mode == "core_fit":
            cx, cy, ok = est["core_centre_x_um"], est["core_centre_y_um"], est["core_fit_reliable"]
        elif estimator_mode == "roi_centroid":
            cx, cy, ok = est["roi_intensity_centroid_x_um"], est["roi_intensity_centroid_y_um"], True
        else:  # auto
            cx, cy = est["beam_axis_x_um"], est["beam_axis_y_um"]
            ok = est["beam_axis_reliability"] in ("reliable", "caution")
        methods.append(est["beam_axis_method"]); reliab.append(est["beam_axis_reliability"])
        in_frame = (
            float(est["out_of_frame_fraction"]) <= 0.02
            and float(est["field_of_view_margin_um"]) >= -2.0 * abs(x[1] - x[0])
        )
        if strong[i] and ok and in_frame:
            bx[i] = cx; by[i] = cy

    valid = np.isfinite(bx) & np.isfinite(by)
    valid_fraction = float(np.sum(valid) / max(nz, 1))
    if np.sum(valid) >= 3:
        zc = z[valid]
        sx, ix0 = np.polyfit(zc, bx[valid], 1)
        sy, iy0 = np.polyfit(zc, by[valid], 1)
        fx = ix0 + sx * zc; fy = iy0 + sy * zc
        ss_res = float(np.sum((bx[valid] - fx) ** 2) + np.sum((by[valid] - fy) ** 2))
        ss_tot = float(np.sum((bx[valid] - bx[valid].mean()) ** 2)
                       + np.sum((by[valid] - by[valid].mean()) ** 2)) + 1e-30
        fit_q = float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))
        z_range = (float(zc.min()), float(zc.max()))
        intercept = (float(ix0), float(iy0))
        first_valid = (float(ix0 + sx * float(zc.min())), float(iy0 + sy * float(zc.min())))
    else:
        sx = sy = ix0 = iy0 = float("nan"); fit_q = 0.0
        z_range = (float("nan"), float("nan"))
        intercept = (float("nan"), float("nan"))
        first_valid = (float("nan"), float("nan"))

    return {
        "axis_x_by_z_um": bx.tolist(), "axis_y_by_z_um": by.tolist(),
        "valid_plane_mask": valid.tolist(), "valid_plane_fraction": valid_fraction,
        "per_plane_method": methods, "per_plane_reliability": reliab,
        "axis_intercept_at_z0_x_um": intercept[0], "axis_intercept_at_z0_y_um": intercept[1],
        "axis_intercept_at_first_valid_z_x_um": first_valid[0],
        "axis_intercept_at_first_valid_z_y_um": first_valid[1],
        "reference_plane_axis_error_um": float(np.hypot(intercept[0], intercept[1])) if np.isfinite(intercept[0]) else float("nan"),
        "beam_steering_angle_x_mrad": float(np.arctan(sx) * 1000.0) if np.isfinite(sx) else float("nan"),
        "beam_steering_angle_y_mrad": float(np.arctan(sy) * 1000.0) if np.isfinite(sy) else float("nan"),
        "measured_slope_x": float(sx), "measured_slope_y": float(sy),
        "trajectory_fit_quality": fit_q,
        "valid_z_fit_range_um": z_range,
        "estimator_mode": estimator_mode,
    }
