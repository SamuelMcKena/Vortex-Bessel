"""Calibrated camera ingestion and simulation-to-experiment comparison.

The routines here keep camera calibration separate from optical-model fitting.
A physical metres-per-pixel scale, camera rotation and reference centre are
inputs; the comparison is not allowed to tune scale until a picture looks good.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares


EPS = np.finfo(float).tiny


@dataclass(frozen=True)
class CameraCalibration:
    object_plane_scale_m_per_pixel: float
    rotation_rad: float = 0.0
    centre_pixel_x: float | None = None
    centre_pixel_y: float | None = None
    saturation_level: float | None = None

    def validate(self) -> None:
        if not np.isfinite(self.object_plane_scale_m_per_pixel) or self.object_plane_scale_m_per_pixel <= 0.0:
            raise ValueError("object_plane_scale_m_per_pixel must be positive")
        if not np.isfinite(self.rotation_rad):
            raise ValueError("rotation_rad must be finite")
        if self.saturation_level is not None and self.saturation_level <= 0.0:
            raise ValueError("saturation_level must be positive when supplied")


@dataclass(frozen=True)
class GaussianFit2D:
    amplitude: float
    x0_m: float
    y0_m: float
    wx_m: float
    wy_m: float
    rotation_rad: float
    background: float
    residual_rms_fraction: float
    success: bool


@dataclass(frozen=True)
class CameraComparison:
    measured_intensity: np.ndarray
    simulated_on_camera: np.ndarray
    valid_mask: np.ndarray
    metrics: Mapping[str, float]
    x_m: np.ndarray
    y_m: np.ndarray
    measured_x_profile: np.ndarray
    measured_y_profile: np.ndarray
    simulated_x_profile: np.ndarray
    simulated_y_profile: np.ndarray


def camera_axes(
    image_shape: tuple[int, int],
    calibration: CameraCalibration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return camera pixel axes and rotated physical X/Y coordinates."""

    calibration.validate()
    ny, nx = map(int, image_shape)
    cx = (nx - 1.0) / 2.0 if calibration.centre_pixel_x is None else float(calibration.centre_pixel_x)
    cy = (ny - 1.0) / 2.0 if calibration.centre_pixel_y is None else float(calibration.centre_pixel_y)
    q = float(calibration.object_plane_scale_m_per_pixel)
    px = (np.arange(nx, dtype=float) - cx) * q
    py = (np.arange(ny, dtype=float) - cy) * q
    PX, PY = np.meshgrid(px, py, indexing="xy")
    c = math.cos(float(calibration.rotation_rad))
    s = math.sin(float(calibration.rotation_rad))
    X = c * PX - s * PY
    Y = s * PX + c * PY
    return px, py, X, Y


def preprocess_camera_image(
    image: np.ndarray,
    *,
    background: np.ndarray | float | None = None,
    saturation_level: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Subtract background, clip negatives and return a non-saturated mask."""

    raw = np.asarray(image, dtype=float)
    if raw.ndim != 2 or raw.size == 0 or np.any(~np.isfinite(raw)):
        raise ValueError("camera image must be a finite non-empty 2-D array")
    bg = 0.0 if background is None else np.asarray(background, dtype=float)
    corrected = raw - bg
    corrected = np.maximum(corrected, 0.0)
    sat = float(saturation_level) if saturation_level is not None else math.inf
    valid = raw < sat
    fraction = float(np.mean(~valid))
    return corrected, valid, {
        "saturated_fraction": fraction,
        "background_subtracted": bool(background is not None),
        "raw_peak": float(np.max(raw)),
        "corrected_peak": float(np.max(corrected)),
    }


def _weighted_moments(intensity: np.ndarray, X: np.ndarray, Y: np.ndarray) -> tuple[float, float, float, float, float]:
    I = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    total = max(float(np.sum(I)), EPS)
    x0 = float(np.sum(I * X) / total)
    y0 = float(np.sum(I * Y) / total)
    dx = X - x0
    dy = Y - y0
    cxx = float(np.sum(I * dx * dx) / total)
    cyy = float(np.sum(I * dy * dy) / total)
    cxy = float(np.sum(I * dx * dy) / total)
    angle = 0.5 * math.atan2(2.0 * cxy, cxx - cyy)
    c = math.cos(angle)
    s = math.sin(angle)
    major = c * c * cxx + 2.0 * c * s * cxy + s * s * cyy
    minor = s * s * cxx - 2.0 * c * s * cxy + c * c * cyy
    # For I=I0 exp[-2 x^2/w^2], variance=w^2/4.
    wx = 2.0 * math.sqrt(max(major, EPS))
    wy = 2.0 * math.sqrt(max(minor, EPS))
    return x0, y0, wx, wy, angle


def fit_gaussian_2d(
    intensity: np.ndarray,
    calibration: CameraCalibration,
    *,
    valid_mask: np.ndarray | None = None,
) -> GaussianFit2D:
    """Fit a rotated 1/e^2-intensity Gaussian source model."""

    I = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    _, _, X, Y = camera_axes(I.shape, calibration)
    valid = np.ones_like(I, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != I.shape or np.count_nonzero(valid) < 20:
        raise ValueError("valid_mask is incompatible with the camera image")
    x0, y0, wx, wy, theta = _weighted_moments(np.where(valid, I, 0.0), X, Y)
    amp = max(float(np.max(I[valid]) - np.median(I[valid])), EPS)
    bg0 = max(float(np.median(I[valid])), 0.0)

    xv = X[valid]
    yv = Y[valid]
    iv = I[valid]
    scale = max(float(np.max(iv)), EPS)

    def residual(p: np.ndarray) -> np.ndarray:
        A, xc, yc, log_wx, log_wy, ang, B = p
        c = np.cos(ang)
        s = np.sin(ang)
        xr = c * (xv - xc) + s * (yv - yc)
        yr = -s * (xv - xc) + c * (yv - yc)
        model = np.maximum(A, 0.0) * np.exp(
            -2.0 * (xr * xr / np.exp(2.0 * log_wx) + yr * yr / np.exp(2.0 * log_wy))
        ) + np.maximum(B, 0.0)
        return (model - iv) / scale

    p0 = np.asarray([amp, x0, y0, math.log(max(wx, 1e-12)), math.log(max(wy, 1e-12)), theta, bg0])
    fit = least_squares(residual, p0, method="trf", max_nfev=500)
    A, xc, yc, log_wx, log_wy, ang, B = fit.x
    rms = float(np.sqrt(np.mean(residual(fit.x) ** 2)))
    return GaussianFit2D(
        amplitude=float(max(A, 0.0)),
        x0_m=float(xc),
        y0_m=float(yc),
        wx_m=float(np.exp(log_wx)),
        wy_m=float(np.exp(log_wy)),
        rotation_rad=float(ang),
        background=float(max(B, 0.0)),
        residual_rms_fraction=rms,
        success=bool(fit.success),
    )


def _centroid_and_covariance(I: np.ndarray, X: np.ndarray, Y: np.ndarray, valid: np.ndarray) -> tuple[float, float, np.ndarray]:
    values = np.where(valid, np.maximum(I, 0.0), 0.0)
    total = max(float(np.sum(values)), EPS)
    cx = float(np.sum(values * X) / total)
    cy = float(np.sum(values * Y) / total)
    dx = X - cx
    dy = Y - cy
    cov = np.asarray(
        [
            [np.sum(values * dx * dx) / total, np.sum(values * dx * dy) / total],
            [np.sum(values * dx * dy) / total, np.sum(values * dy * dy) / total],
        ],
        dtype=float,
    )
    return cx, cy, cov


def compare_simulation_to_camera(
    measured_intensity: np.ndarray,
    calibration: CameraCalibration,
    simulated_intensity: np.ndarray,
    simulated_x_m: np.ndarray,
    simulated_y_m: np.ndarray,
    *,
    background: np.ndarray | float | None = None,
) -> CameraComparison:
    """Resample a simulated physical plane onto the calibrated camera pixels."""

    measured, valid, _ = preprocess_camera_image(
        measured_intensity,
        background=background,
        saturation_level=calibration.saturation_level,
    )
    sx = np.asarray(simulated_x_m, dtype=float)
    sy = np.asarray(simulated_y_m, dtype=float)
    sim = np.maximum(np.asarray(simulated_intensity, dtype=float), 0.0)
    if sim.shape != (sy.size, sx.size):
        raise ValueError("simulated_intensity does not match simulated axes")
    if np.any(np.diff(sx) <= 0.0) or np.any(np.diff(sy) <= 0.0):
        raise ValueError("simulated axes must increase")
    _, _, X, Y = camera_axes(measured.shape, calibration)
    interp = RegularGridInterpolator((sy, sx), sim, bounds_error=False, fill_value=np.nan)
    sampled = interp(np.column_stack([Y.ravel(), X.ravel()])).reshape(measured.shape)
    valid = valid & np.isfinite(sampled)
    sampled = np.where(np.isfinite(sampled), sampled, 0.0)
    if np.count_nonzero(valid) < 20:
        raise ValueError("camera and simulated fields have insufficient calibrated overlap")

    m = np.where(valid, measured, 0.0)
    s = np.where(valid, sampled, 0.0)
    m_energy = m / max(float(np.sum(m)), EPS)
    s_energy = s / max(float(np.sum(s)), EPS)
    corr = float(np.sum(m_energy * s_energy) / max(np.linalg.norm(m_energy) * np.linalg.norm(s_energy), EPS))
    l2 = float(np.linalg.norm(m_energy - s_energy) / max(np.linalg.norm(m_energy), EPS))
    mcx, mcy, mcov = _centroid_and_covariance(m, X, Y, valid)
    scx, scy, scov = _centroid_and_covariance(s, X, Y, valid)

    # True physical lines through the calibrated origin are obtained by nearest
    # camera row/column; sub-pixel line extraction belongs in the simulation
    # profile layer, not by inventing extra camera resolution.
    origin_index = np.unravel_index(int(np.argmin(X * X + Y * Y)), X.shape)
    iy0, ix0 = origin_index
    return CameraComparison(
        measured_intensity=m,
        simulated_on_camera=s,
        valid_mask=valid,
        metrics={
            "energy_normalised_correlation": corr,
            "energy_normalised_l2": l2,
            "measured_centroid_x_m": mcx,
            "measured_centroid_y_m": mcy,
            "simulated_centroid_x_m": scx,
            "simulated_centroid_y_m": scy,
            "centroid_error_m": float(math.hypot(mcx - scx, mcy - scy)),
            "covariance_relative_frobenius_error": float(
                np.linalg.norm(mcov - scov) / max(np.linalg.norm(mcov), EPS)
            ),
            "valid_overlap_fraction": float(np.mean(valid)),
        },
        x_m=np.asarray(X[iy0, :], dtype=float),
        y_m=np.asarray(Y[:, ix0], dtype=float),
        measured_x_profile=np.asarray(m[iy0, :], dtype=float),
        measured_y_profile=np.asarray(m[:, ix0], dtype=float),
        simulated_x_profile=np.asarray(s[iy0, :], dtype=float),
        simulated_y_profile=np.asarray(s[:, ix0], dtype=float),
    )


__all__ = [
    "CameraCalibration",
    "CameraComparison",
    "GaussianFit2D",
    "camera_axes",
    "compare_simulation_to_camera",
    "fit_gaussian_2d",
    "preprocess_camera_image",
]
