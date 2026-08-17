"""Stage 9A measured-image quality + optical metric extraction (pixel space).

Computes only metrics supported by the image and its coordinate status.  Pixel
metrics are allowed before camera calibration; physical-unit metrics require an
explicitly declared, calibrated camera mapping.  This module does NOT infer
optical phase from intensity, does NOT infer aberration coefficients, and makes
NO material claim.  No camera-imaging physics is implemented.

Boundary unchanged: n = 1.0 free-space diagnostics; ``camera_model_enabled =
False``; ``final_export_allowed = False``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from vbb_study.digital_twin.annular_axis_tracking import estimate_annular_axis

RING_ANNULAR_QUALITY_MIN = 0.55
DARK_CORE_MIN_FOR_ANNULAR = 0.35


def load_image(path: str | Path) -> dict[str, Any]:
    """Load PNG / TIFF / NumPy .npy into a 2D float array (raw values preserved).

    Returns the array plus detected bit depth and source dtype.  No crop / rotate
    / normalise is applied.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        arr = np.load(p)
        bit_depth = int(np.asarray(arr).dtype.itemsize) * 8
        source_dtype = str(np.asarray(arr).dtype)
    elif suffix in (".png", ".tif", ".tiff"):
        from PIL import Image
        im = Image.open(p)
        source_dtype = str(im.mode)
        arr = np.asarray(im)
        bit_depth = {"L": 8, "P": 8, "I;16": 16, "I": 32, "F": 32}.get(im.mode, 8 * arr.dtype.itemsize)
    else:
        raise ValueError(f"unsupported image format {suffix!r}; supported: .png .tif .tiff .npy")
    arr = np.asarray(arr)
    if arr.ndim == 3:  # collapse colour channels to a single intensity plane (no crop)
        arr = arr.mean(axis=2)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {arr.shape}")
    return {"image": arr.astype(float), "bit_depth": int(bit_depth),
            "source_dtype": source_dtype, "shape": tuple(int(s) for s in arr.shape)}


def _saturation_level(image: np.ndarray, bit_depth: int | None, explicit: float | None) -> float | None:
    if explicit is not None:
        return float(explicit)
    if bit_depth in (8, 16, 32) and np.issubdtype(np.asarray(image).dtype, np.integer):
        return float((1 << bit_depth) - 1)
    # heuristic for integer-valued floats from an 8/16-bit source
    mx = float(np.max(image))
    if mx <= 255.0:
        return 255.0
    if mx <= 65535.0:
        return 65535.0
    return None


def _background_estimate(image: np.ndarray, border_frac: float = 0.06) -> float:
    h, w = image.shape
    b = max(1, int(round(min(h, w) * border_frac)))
    border = np.concatenate([
        image[:b, :].ravel(), image[-b:, :].ravel(),
        image[:, :b].ravel(), image[:, -b:].ravel(),
    ])
    return float(np.median(border))


def compute_measured_image_metrics(
    image: np.ndarray,
    *,
    bit_depth: int | None = None,
    saturation_level: float | None = None,
    coordinate_calibrated: bool = False,
    camera_scale_um_per_px: float | None = None,
    subtract_background: bool = True,
) -> dict[str, Any]:
    """Pixel-space optical metrics for a measured image.

    ``coordinate_calibrated`` + ``camera_scale_um_per_px`` are required before any
    physical-unit (um) metric is reported; otherwise only pixel metrics appear and
    physical metrics are explicitly blocked.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError("image must be 2D")
    h, w = image.shape
    sat = _saturation_level(image, bit_depth, saturation_level)
    background = _background_estimate(image)
    work = np.clip(image - background, 0.0, None) if subtract_background else image.copy()

    finite = np.isfinite(image)
    if sat is not None:
        saturated = image >= sat - 1e-9
        saturation_fraction = float(np.mean(saturated))
    else:
        saturated = np.zeros_like(image, dtype=bool)
        saturation_fraction = 0.0
    valid_pixel_fraction = float(np.mean(finite & ~saturated))

    xs = np.arange(w, dtype=float)
    ys = np.arange(h, dtype=float)
    total = float(np.sum(work))
    if total > 0:
        X, Y = np.meshgrid(xs, ys)
        cx = float(np.sum(work * X) / total)
        cy = float(np.sum(work * Y) / total)
    else:
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    est = estimate_annular_axis(work, xs, ys)
    ring_q = float(est["ring_fit_quality"])
    dark_core = float(np.clip(1.0 - est["core_fill_fraction"], 0.0, 1.0))
    is_annular = bool(ring_q >= RING_ANNULAR_QUALITY_MIN and dark_core >= DARK_CORE_MIN_FOR_ANNULAR)

    # radial profile about the intensity centroid (pixels)
    R = np.hypot(np.meshgrid(xs, ys)[0] - cx, np.meshgrid(xs, ys)[1] - cy)
    rmax = float(min(cx, cy, w - 1 - cx, h - 1 - cy))
    nb = max(8, int(rmax))
    edges = np.linspace(0, max(rmax, 1.0), nb + 1)
    idx = np.clip(np.digitize(R.ravel(), edges) - 1, 0, nb - 1)
    prof = np.bincount(idx, weights=work.ravel(), minlength=nb) / np.maximum(np.bincount(idx, minlength=nb), 1)

    ring_x = float(est["ring_centre_x_um"]); ring_y = float(est["ring_centre_y_um"])
    fov_margin_px = float(rmax - (np.hypot(ring_x - cx, ring_y - cy) + est["ring_radius_um"]))

    flags: list[str] = []
    if saturation_fraction > 0.005:
        flags.append("saturated")
    if (float(np.max(work)) <= 3.0 * (np.std(image[finite]) + 1e-9)):
        flags.append("low_signal")
    if fov_margin_px < 2.0:
        flags.append("near_edge_or_clipped")
    if not is_annular:
        flags.append("not_annular")

    metrics: dict[str, Any] = {
        "image_width_px": int(w),
        "image_height_px": int(h),
        "bit_depth": bit_depth,
        "background_estimate": background,
        "saturation_level": sat,
        "saturation_fraction": saturation_fraction,
        "valid_pixel_fraction": valid_pixel_fraction,
        "intensity_centroid_x_px": cx,
        "intensity_centroid_y_px": cy,
        "ring_centre_x_px": ring_x,
        "ring_centre_y_px": ring_y,
        "ring_radius_px": float(est["ring_radius_um"]),
        "ring_fit_quality": ring_q,
        "dark_core_fraction": dark_core,
        "azimuthal_uniformity": float(est["azimuthal_uniformity"]),
        "radial_profile_px": prof.tolist(),
        "field_of_view_margin_px": fov_margin_px,
        "is_annular": is_annular,
        "image_quality_flags": flags,
        "units": "pixel",
        "coordinate_status": "calibrated" if coordinate_calibrated else "pixel_only_uncalibrated",
    }

    if coordinate_calibrated and camera_scale_um_per_px is not None:
        s = float(camera_scale_um_per_px)
        metrics["physical_metrics_status"] = "calibrated_camera_scale_declared"
        metrics["ring_radius_um"] = metrics["ring_radius_px"] * s
        metrics["ring_centre_x_um"] = ring_x * s
        metrics["ring_centre_y_um"] = ring_y * s
        metrics["camera_scale_um_per_px"] = s
    else:
        metrics["physical_metrics_status"] = "blocked_coordinate_uncalibrated"
        metrics["physical_metrics_note"] = (
            "pixel metrics only; declare a calibrated camera scale + reference-plane relation "
            "before reporting um/mm metrics")
    return metrics


def image_quality_report(
    image: np.ndarray,
    *,
    bit_depth: int | None = None,
    saturation_level: float | None = None,
    coordinate_calibrated: bool = False,
    expect_annular: bool | None = None,
) -> dict[str, Any]:
    """Per-capture serialisable quality report."""
    m = compute_measured_image_metrics(
        image, bit_depth=bit_depth, saturation_level=saturation_level,
        coordinate_calibrated=coordinate_calibrated)
    flags = list(m["image_quality_flags"])
    if "saturated" in flags or m["valid_pixel_fraction"] < 0.5:
        quality_status = "rejected"
        recommended = "reduce exposure/gain and re-acquire"
    elif "low_signal" in flags or "near_edge_or_clipped" in flags:
        quality_status = "caution"
        recommended = "increase signal or re-centre/zoom; review before analysis"
    else:
        quality_status = "ok"
        recommended = "proceed; pixel metrics valid"
    annular_note = ""
    if expect_annular is True and not m["is_annular"]:
        flags.append("expected_annular_but_not_detected")
        annular_note = "expected an annular field but ring fit is weak; do not force an annular fit"
    metric_validity = "pixel_metrics_valid"
    if coordinate_calibrated:
        metric_validity = "pixel_and_physical_metrics_valid"
    return {
        "quality_status": quality_status,
        "flags": flags,
        "background_handling": "median border background estimate; subtraction is non-destructive",
        "saturation_status": ("saturated" if "saturated" in flags else "ok"),
        "saturation_fraction": m["saturation_fraction"],
        "valid_pixel_fraction": m["valid_pixel_fraction"],
        "coordinate_status": m["coordinate_status"],
        "is_annular": m["is_annular"],
        "metric_validity": metric_validity,
        "recommended_action": recommended,
        "annular_note": annular_note,
        "physical_metrics_status": m["physical_metrics_status"],
    }


# ---------------------------------------------------------------------------
# Work package F — measured vs model comparison boundary
# ---------------------------------------------------------------------------


def compare_measured_to_model(
    measured_metrics: Mapping[str, Any],
    model_metrics: Mapping[str, Any],
    *,
    camera_scale_um_per_px: float | None = None,
    reference_plane_relation: str | None = None,
) -> dict[str, Any]:
    """Compare like-for-like only.

    An absolute physical comparison (camera ring radius in physical units vs model
    ring radius) is allowed only when a declared camera scale AND a named
    reference-plane relation exist.  Otherwise the comparison is explicitly
    labelled not physically calibrated, and only normalised shape descriptors are
    offered for exploratory use.
    """
    physically_calibrated = bool(
        camera_scale_um_per_px is not None and reference_plane_relation
        and measured_metrics.get("coordinate_status") == "calibrated"
    )
    out: dict[str, Any] = {
        "final_export_allowed": False,
        "no_fitting_or_inverse_correction": True,
    }
    if physically_calibrated:
        meas_um = float(measured_metrics["ring_radius_px"]) * float(camera_scale_um_per_px)
        model_um = float(model_metrics.get("ring_radius_um", float("nan")))
        out["comparison_status"] = "physically_calibrated_like_for_like"
        out["reference_plane_relation"] = reference_plane_relation
        out["measured_ring_radius_um"] = meas_um
        out["model_ring_radius_um"] = model_um
        out["ring_radius_abs_diff_um"] = abs(meas_um - model_um)
    else:
        out["comparison_status"] = "comparison_not_physically_calibrated"
        out["reason"] = ("no declared camera scale + named reference-plane relation; "
                         "absolute physical comparison is blocked")

    # Normalised, shape-only descriptors (exploratory; never absolute validation).
    def _shape(m):
        rr = float(m.get("ring_radius_px", 0.0)) or 1.0
        return {
            "azimuthal_uniformity": float(m.get("azimuthal_uniformity", 0.0)),
            "dark_core_fraction": float(m.get("dark_core_fraction", 0.0)),
            "ring_fit_quality": float(m.get("ring_fit_quality", 0.0)),
        }
    out["shape_only_diagnostic_comparison"] = {
        "label": "not_absolute_physical_validation",
        "measured": _shape(measured_metrics),
        "model": _shape(model_metrics),
    }
    return out
