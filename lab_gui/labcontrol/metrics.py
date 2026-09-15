"""State-aware metrics computed only from quantitative camera matrices."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .devices.camera import CameraFrame
from .state import ExperimentState


@dataclass(frozen=True)
class BeamMetrics:
    family: str
    values: dict[str, float | bool | str | None]
    centre_yx_px: tuple[float, float]
    radial_radius_px: np.ndarray = field(repr=False)
    radial_signal: np.ndarray = field(repr=False)
    warnings: tuple[str, ...] = ()

    def to_dict(self, *, include_profile: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "family": self.family,
            "values": dict(self.values),
            "centre_yx_px": list(self.centre_yx_px),
            "warnings": list(self.warnings),
        }
        if include_profile:
            payload["radial_profile"] = {
                "radius_px": self.radial_radius_px.tolist(),
                "mean_signal": self.radial_signal.tolist(),
            }
        return payload


def preprocess(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Subtract a robust background and clip negative residuals.

    This produces an analysis array only.  The raw frame is never modified.
    """

    raw = np.asarray(image, dtype=np.float64)
    if raw.ndim != 2 or min(raw.shape) < 4 or not np.isfinite(raw).all():
        raise ValueError("Metrics require a finite quantitative 2-D matrix.")
    border = np.concatenate((raw[0], raw[-1], raw[:, 0], raw[:, -1]))
    background = float(np.median(border))
    noise_sigma = 1.4826 * float(np.median(np.abs(border - background)))
    # Suppress the positive half of zero-mean camera noise before moments are
    # taken; otherwise a large sensor can dominate widths with faint background.
    signal = np.maximum(raw - background - 3.0 * noise_sigma, 0.0)
    if not np.any(signal > 0):
        raise ValueError("No positive signal remains after background subtraction.")
    return signal, background


def intensity_centroid(signal: np.ndarray) -> tuple[float, float]:
    total = float(np.sum(signal))
    if total <= 0:
        raise ValueError("Cannot find a centroid without positive signal.")
    y, x = np.indices(signal.shape, dtype=np.float64)
    return float(np.sum(y * signal) / total), float(np.sum(x * signal) / total)


def gradient_symmetry_center(image: np.ndarray, half_width: int | None = None) -> tuple[float, float]:
    """Estimate a central-spot or annulus centre from radial gradient lines."""

    signal, _ = preprocess(image)
    if half_width is None:
        half_width = max(24, min(160, min(signal.shape) // 3))
    seed_y, seed_x = np.unravel_index(int(np.argmax(signal)), signal.shape)
    y0 = max(1, int(seed_y) - half_width)
    y1 = min(signal.shape[0] - 1, int(seed_y) + half_width + 1)
    x0 = max(1, int(seed_x) - half_width)
    x1 = min(signal.shape[1] - 1, int(seed_x) + half_width + 1)
    crop = signal[y0:y1, x0:x1]
    if min(crop.shape) < 25:
        raise ValueError("Signal is too close to the sensor edge for symmetry-centre estimation.")

    padded = np.pad(crop, 1, mode="edge")
    smooth = sum(
        padded[dy : dy + crop.shape[0], dx : dx + crop.shape[1]]
        for dy in range(3)
        for dx in range(3)
    ) / 9.0
    gy, gx = np.gradient(smooth)
    magnitude = np.hypot(gx, gy)
    yy, xx = np.indices(smooth.shape, dtype=np.float64)
    use = (smooth >= np.percentile(smooth, 65.0)) & (
        magnitude >= np.percentile(magnitude, 70.0)
    )
    if np.count_nonzero(use) < 50:
        use = magnitude >= np.percentile(magnitude, 60.0)
    if np.count_nonzero(use) < 20:
        raise ValueError("Insufficient structured signal for a stable symmetry-centre fit.")

    gxv, gyv = gx[use], gy[use]
    xv, yv = xx[use], yy[use]
    weights = np.maximum(magnitude[use], 1e-12)
    matrix = np.column_stack((gyv, -gxv))
    rhs = gyv * xv - gxv * yv
    sqrt_weight = np.sqrt(weights / np.max(weights))
    solution, *_ = np.linalg.lstsq(
        matrix * sqrt_weight[:, None], rhs * sqrt_weight, rcond=None
    )
    cx_local, cy_local = map(float, solution)
    if not (
        -half_width <= cx_local <= crop.shape[1] + half_width
        and -half_width <= cy_local <= crop.shape[0] + half_width
    ):
        raise ValueError("Symmetry-centre fit escaped the local signal region.")
    return y0 + cy_local, x0 + cx_local


def radial_profile(
    signal: np.ndarray,
    centre_yx: tuple[float, float],
    *,
    bin_width_px: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    if bin_width_px <= 0:
        raise ValueError("Radial bin width must be positive.")
    y, x = np.indices(signal.shape, dtype=np.float64)
    radius = np.hypot(y - centre_yx[0], x - centre_yx[1])
    indices = np.floor(radius / bin_width_px).astype(np.int64)
    count = np.bincount(indices.ravel())
    total = np.bincount(indices.ravel(), weights=signal.ravel())
    valid = count > 0
    centres = (np.arange(len(count), dtype=np.float64) + 0.5) * bin_width_px
    return centres[valid], total[valid] / count[valid]


def _covariance_metrics(
    signal: np.ndarray, centre_yx: tuple[float, float]
) -> tuple[float, float, float, float]:
    y, x = np.indices(signal.shape, dtype=np.float64)
    dy, dx = y - centre_yx[0], x - centre_yx[1]
    total = float(np.sum(signal))
    cov = np.array(
        [
            [np.sum(signal * dx * dx), np.sum(signal * dx * dy)],
            [np.sum(signal * dx * dy), np.sum(signal * dy * dy)],
        ],
        dtype=np.float64,
    ) / total
    eigenvalues = np.maximum(np.linalg.eigvalsh(cov), 0.0)
    sigma_minor, sigma_major = np.sqrt(eigenvalues)
    ellipticity = float(sigma_major / sigma_minor) if sigma_minor > 0 else math.inf
    angle = float(np.degrees(0.5 * np.arctan2(2 * cov[0, 1], cov[0, 0] - cov[1, 1])))
    return float(sigma_major), float(sigma_minor), ellipticity, angle


def _generic_values(
    raw: np.ndarray,
    signal: np.ndarray,
    full_scale: float,
    background: float,
    reference: np.ndarray | None,
) -> dict[str, float | bool | str | None]:
    total = float(np.sum(signal))
    edge = max(1, min(signal.shape) // 50)
    edge_mask = np.zeros(signal.shape, dtype=bool)
    edge_mask[:edge] = edge_mask[-edge:] = True
    edge_mask[:, :edge] = edge_mask[:, -edge:] = True
    values: dict[str, float | bool | str | None] = {
        "background": background,
        "total_signal": total,
        "power_proxy": total,
        "peak": float(np.max(signal)),
        "saturation_fraction": float(np.mean(raw >= full_scale)),
        "edge_power_fraction": float(np.sum(signal[edge_mask]) / total),
        "clipped": bool(np.sum(signal[edge_mask]) / total > 0.01),
    }
    if reference is not None:
        ref, _ = preprocess(reference)
        if ref.shape != signal.shape:
            raise ValueError("Reference image shape differs from the camera frame.")
        a, b = signal.ravel(), ref.ravel()
        denom = float(np.linalg.norm(a - np.mean(a)) * np.linalg.norm(b - np.mean(b)))
        values["image_correlation"] = (
            float(np.dot(a - np.mean(a), b - np.mean(b)) / denom) if denom else None
        )
    return values


def analyse_central_beam(
    frame: CameraFrame, *, reference: np.ndarray | None = None
) -> BeamMetrics:
    signal, background = preprocess(frame.data)
    centre = intensity_centroid(signal)
    sigma_major, sigma_minor, ellipticity, angle = _covariance_metrics(signal, centre)
    radii, profile = radial_profile(signal, centre)
    y, x = np.indices(signal.shape, dtype=np.float64)
    total = float(np.sum(signal))
    sigma_x = float(np.sqrt(np.sum(signal * (x - centre[1]) ** 2) / total))
    sigma_y = float(np.sqrt(np.sum(signal * (y - centre[0]) ** 2) / total))
    values = _generic_values(frame.data, signal, frame.full_scale, background, reference)
    values.update(
        {
            "centre_y_px": centre[0],
            "centre_x_px": centre[1],
            "sigma_major_px": sigma_major,
            "sigma_minor_px": sigma_minor,
            "width_x_sigma_px": sigma_x,
            "width_y_sigma_px": sigma_y,
            "fwhm_x_px": 2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma_x,
            "fwhm_y_px": 2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma_y,
            "fwhm_major_px": 2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma_major,
            "fwhm_minor_px": 2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma_minor,
            "ellipticity": ellipticity,
            "principal_axis_deg": angle,
            "symmetry": 0.0 if not np.isfinite(ellipticity) else 1.0 / ellipticity,
        }
    )
    return BeamMetrics("central_beam", values, centre, radii, profile)


def analyse_vortex_beam(
    frame: CameraFrame, *, reference: np.ndarray | None = None
) -> BeamMetrics:
    signal, background = preprocess(frame.data)
    warnings: list[str] = []
    try:
        centre = gradient_symmetry_center(signal)
    except ValueError as exc:
        centre = intensity_centroid(signal)
        warnings.append(f"Symmetry centre unavailable; used intensity centroid: {exc}")
    radii, profile = radial_profile(signal, centre)
    if len(profile) < 5:
        raise ValueError("Insufficient radial samples for annulus analysis.")
    # Ignore only the first two bins so a small ring remains discoverable.
    peak_index = 2 + int(np.argmax(profile[2:]))
    ring_radius = float(radii[peak_index])
    half_width = max(1.5, 0.12 * ring_radius)
    y, x = np.indices(signal.shape, dtype=np.float64)
    dy, dx = y - centre[0], x - centre[1]
    radius = np.hypot(dy, dx)
    annulus = np.abs(radius - ring_radius) <= half_width
    if np.count_nonzero(annulus) < 16 or np.sum(signal[annulus]) <= 0:
        raise ValueError("Principal ring is too small or weak for annular metrics.")

    ring_weights = signal[annulus]
    dx_ring, dy_ring = dx[annulus], dy[annulus]
    covariance = np.cov(np.vstack((dx_ring, dy_ring)), aweights=ring_weights, bias=True)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    ring_eccentricity = (
        float(np.sqrt(1.0 - eigenvalues[0] / eigenvalues[1])) if eigenvalues[1] > 0 else 0.0
    )

    theta = np.mod(np.arctan2(dy[annulus], dx[annulus]), 2.0 * np.pi)
    sectors = np.floor(theta / (2.0 * np.pi) * 72).astype(int)
    sector_signal = np.bincount(sectors, weights=ring_weights, minlength=72)
    sector_count = np.bincount(sectors, minlength=72)
    valid = sector_count > 0
    sector_mean = sector_signal[valid] / sector_count[valid]
    azimuthal_cv = float(np.std(sector_mean) / np.mean(sector_mean)) if np.mean(sector_mean) else math.inf
    core = radius <= max(1.0, ring_radius * 0.35)
    ring_mean = float(np.mean(signal[annulus]))
    core_mean = float(np.mean(signal[core])) if np.any(core) else 0.0

    values = _generic_values(frame.data, signal, frame.full_scale, background, reference)
    values.update(
        {
            "centre_y_px": centre[0],
            "centre_x_px": centre[1],
            "annulus_centre_y_px": centre[0],
            "annulus_centre_x_px": centre[1],
            "principal_ring_radius_px": ring_radius,
            "ring_eccentricity": ring_eccentricity,
            "azimuthal_cv": azimuthal_cv,
            "ring_uniformity": 0.0 if not np.isfinite(azimuthal_cv) else 1.0 / (1.0 + azimuthal_cv),
            "dark_core_fraction": core_mean / ring_mean if ring_mean > 0 else math.inf,
            "ring_power_fraction": float(np.sum(signal[annulus]) / np.sum(signal)),
        }
    )
    if reference is not None:
        reference_signal, _ = preprocess(reference)
        _, reference_profile = radial_profile(reference_signal, centre)
        upper = min(len(profile), len(reference_profile))
        current_norm = profile[:upper] / max(float(np.sum(profile[:upper])), 1e-12)
        reference_norm = reference_profile[:upper] / max(
            float(np.sum(reference_profile[:upper])), 1e-12
        )
        values["radial_profile_l1"] = float(np.sum(np.abs(current_norm - reference_norm)))
    return BeamMetrics("vortex_bessel", values, centre, radii, profile, tuple(warnings))


class MetricEngine:
    """Select the analysis family from the authoritative optical state."""

    analysis_version = "labcontrol-metrics-1"

    def analyse(
        self,
        frame: CameraFrame,
        state: ExperimentState,
        *,
        reference: np.ndarray | None = None,
    ) -> BeamMetrics:
        if state.analysis_family() == "vortex_bessel":
            return analyse_vortex_beam(frame, reference=reference)
        return analyse_central_beam(frame, reference=reference)

    @staticmethod
    def repeat_stability(metrics: Iterable[BeamMetrics]) -> dict[str, float]:
        items = list(metrics)
        if len(items) < 2:
            return {"repeat_count": float(len(items)), "centre_rms_px": 0.0, "power_cv": 0.0}
        centres = np.asarray([m.centre_yx_px for m in items], dtype=np.float64)
        powers = np.asarray([m.values["total_signal"] for m in items], dtype=np.float64)
        centre_rms = float(np.sqrt(np.mean(np.sum((centres - centres.mean(axis=0)) ** 2, axis=1))))
        power_cv = float(np.std(powers, ddof=1) / np.mean(powers)) if np.mean(powers) else math.inf
        return {
            "repeat_count": float(len(items)),
            "centre_rms_px": centre_rms,
            "power_cv": power_cv,
        }
