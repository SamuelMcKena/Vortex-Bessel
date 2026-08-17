"""Shack-Hartmann -> SLM wavefront-correction bridge.

The sensor is treated as a slope instrument. Local x/y wavefront slopes are
integrated in a Southwell-style least-squares system on the lenslet grid. A
single piston constraint removes the null mode. The reconstructed optical path
difference (OPD) is then interpolated onto the SLM plane and converted to the
negative optical phase required for correction.

The correction is intentionally an additive phase term. It must never be
implemented by conjugating the complete structured field, because doing so can
remove a desired vortex phase.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.sparse import coo_matrix, vstack
from scipy.sparse.linalg import lsqr


TWOPI = 2.0 * np.pi
EPS = np.finfo(float).tiny


@dataclass(frozen=True)
class ShackHartmannReconstruction:
    opd_m: np.ndarray
    valid_mask: np.ndarray
    residual_rms_m: float
    slope_residual_rms: float
    piston_m: float
    metadata: Mapping[str, object]

    @property
    def opd_rms_m(self) -> float:
        active = np.asarray(self.valid_mask, dtype=bool)
        values = np.asarray(self.opd_m, dtype=float)[active]
        centred = values - float(np.mean(values))
        return float(np.sqrt(np.mean(centred * centred)))


def _uniform_axis(axis: np.ndarray, name: str) -> tuple[np.ndarray, float]:
    values = np.asarray(axis, dtype=float).ravel()
    if values.ndim != 1 or values.size < 2 or np.any(~np.isfinite(values)):
        raise ValueError(f"{name} must be a finite 1-D axis")
    delta = np.diff(values)
    if np.any(delta <= 0.0) or not np.allclose(delta, delta[0], rtol=1e-8, atol=1e-15):
        raise ValueError(f"{name} must be uniformly increasing")
    return values, float(delta[0])


def reconstruct_opd_from_slopes(
    slope_x: np.ndarray,
    slope_y: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    piston_weight: float = 1.0,
    atol: float = 1e-12,
    btol: float = 1e-12,
) -> ShackHartmannReconstruction:
    """Reconstruct OPD from local slopes by sparse least squares.

    ``slope_x`` and ``slope_y`` are dimensionless OPD gradients dW/dx and
    dW/dy. A Shack-Hartmann centroid displacement divided by lenslet focal
    length is the small-angle estimate of these slopes.

    For adjacent nodes, the Southwell/trapezoidal equations are

        W[i,j+1]-W[i,j] = dx*(sx[i,j+1]+sx[i,j])/2
        W[i+1,j]-W[i,j] = dy*(sy[i+1,j]+sy[i,j])/2.

    ``valid_mask`` may describe a circular/irregular illuminated lenslet pupil.
    The graph of valid nearest-neighbour lenslets must remain connected.
    """

    sx = np.asarray(slope_x, dtype=float)
    sy = np.asarray(slope_y, dtype=float)
    if sx.shape != sy.shape or sx.ndim != 2:
        raise ValueError("slope_x and slope_y must be same-shape 2-D arrays")
    x, dx = _uniform_axis(x_m, "x_m")
    y, dy = _uniform_axis(y_m, "y_m")
    if sx.shape != (y.size, x.size):
        raise ValueError("slope arrays do not match x/y axes")
    valid = (
        np.isfinite(sx) & np.isfinite(sy)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool) & np.isfinite(sx) & np.isfinite(sy)
    )
    if valid.shape != sx.shape:
        raise ValueError("valid_mask shape does not match slopes")
    if np.count_nonzero(valid) < 4:
        raise ValueError("too few valid Shack-Hartmann samples")

    ny, nx = sx.shape
    index = -np.ones((ny, nx), dtype=int)
    index[valid] = np.arange(np.count_nonzero(valid), dtype=int)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs: list[float] = []
    eq = 0

    for iy in range(ny):
        for ix in range(nx - 1):
            if valid[iy, ix] and valid[iy, ix + 1]:
                rows.extend([eq, eq])
                cols.extend([int(index[iy, ix]), int(index[iy, ix + 1])])
                data.extend([-1.0, +1.0])
                rhs.append(0.5 * dx * float(sx[iy, ix] + sx[iy, ix + 1]))
                eq += 1
    for iy in range(ny - 1):
        for ix in range(nx):
            if valid[iy, ix] and valid[iy + 1, ix]:
                rows.extend([eq, eq])
                cols.extend([int(index[iy, ix]), int(index[iy + 1, ix])])
                data.extend([-1.0, +1.0])
                rhs.append(0.5 * dy * float(sy[iy, ix] + sy[iy + 1, ix]))
                eq += 1

    n_unknown = int(np.count_nonzero(valid))
    if eq < n_unknown - 1:
        raise ValueError("valid Shack-Hartmann pupil is disconnected or underconstrained")
    A = coo_matrix((data, (rows, cols)), shape=(eq, n_unknown)).tocsr()
    b = np.asarray(rhs, dtype=float)

    # Mean-piston removal avoids privileging one reference lenslet.
    piston_row = coo_matrix(
        (
            np.full(n_unknown, float(piston_weight) / math.sqrt(n_unknown)),
            (np.zeros(n_unknown, dtype=int), np.arange(n_unknown, dtype=int)),
        ),
        shape=(1, n_unknown),
    ).tocsr()
    A_aug = vstack([A, piston_row], format="csr")
    b_aug = np.concatenate([b, np.asarray([0.0])])
    solution = lsqr(
        A_aug,
        b_aug,
        atol=float(atol),
        btol=float(btol),
        iter_lim=max(1000, 20 * n_unknown),
    )
    w = np.asarray(solution[0], dtype=float)

    predicted = A @ w
    residual = predicted - b
    opd = np.full_like(sx, np.nan, dtype=float)
    opd[valid] = w
    piston = float(np.mean(w))
    return ShackHartmannReconstruction(
        opd_m=opd,
        valid_mask=np.asarray(valid, dtype=bool),
        residual_rms_m=float(np.sqrt(np.mean(residual * residual))) if residual.size else 0.0,
        slope_residual_rms=float(np.linalg.norm(residual) / max(math.sqrt(residual.size), 1.0)),
        piston_m=piston,
        metadata={
            "method": "Southwell_trapezoidal_sparse_least_squares",
            "valid_lenslet_count": n_unknown,
            "equation_count": int(eq),
            "lsqr_istop": int(solution[1]),
            "lsqr_iterations": int(solution[2]),
            "lsqr_condition_estimate": float(solution[6]),
            "piston_constraint": "mean_OPD_zero",
            "masked_pupil_supported": True,
        },
    )


def slopes_from_spot_displacements(
    delta_x_m: np.ndarray,
    delta_y_m: np.ndarray,
    *,
    lenslet_focal_length_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert focal-spot displacements to local wavefront slopes."""

    focal = float(lenslet_focal_length_m)
    if not np.isfinite(focal) or focal <= 0.0:
        raise ValueError("lenslet_focal_length_m must be positive")
    return np.asarray(delta_x_m, dtype=float) / focal, np.asarray(delta_y_m, dtype=float) / focal


def _registered_query_coordinates(
    target_X_m: np.ndarray,
    target_Y_m: np.ndarray,
    *,
    rotation_rad: float,
    scale_x: float,
    scale_y: float,
    offset_x_m: float,
    offset_y_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    if scale_x <= 0.0 or scale_y <= 0.0:
        raise ValueError("registration scales must be positive")
    X = np.asarray(target_X_m, dtype=float) - float(offset_x_m)
    Y = np.asarray(target_Y_m, dtype=float) - float(offset_y_m)
    c = math.cos(float(rotation_rad))
    s = math.sin(float(rotation_rad))
    xr = (c * X + s * Y) / float(scale_x)
    yr = (-s * X + c * Y) / float(scale_y)
    return xr, yr


def _interpolate_reconstructed_opd(
    reconstruction: ShackHartmannReconstruction,
    x: np.ndarray,
    y: np.ndarray,
    xq: np.ndarray,
    yq: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Interpolate reconstructed OPD without extrapolating beyond measured support."""

    opd = np.asarray(reconstruction.opd_m, dtype=float)
    valid = np.asarray(reconstruction.valid_mask, dtype=bool) & np.isfinite(opd)
    if opd.shape != (y.size, x.size) or valid.shape != opd.shape:
        raise ValueError("reconstruction shape does not match sensor axes")

    if np.all(valid):
        interp = RegularGridInterpolator(
            (y, x),
            opd,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        points = np.column_stack([yq.ravel(), xq.ravel()])
        return interp(points).reshape(np.shape(xq)), "regular_grid_linear"

    Y, X = np.meshgrid(y, x, indexing="ij")
    points = np.column_stack([X[valid], Y[valid]])
    values = opd[valid]
    if points.shape[0] < 4:
        raise ValueError("too few valid reconstructed lenslets for 2-D interpolation")
    interp = LinearNDInterpolator(points, values, fill_value=np.nan)
    sampled = interp(np.column_stack([xq.ravel(), yq.ravel()])).reshape(np.shape(xq))
    return np.asarray(sampled, dtype=float), "masked_linear_triangulation_no_extrapolation"


def correction_phase_on_slm(
    reconstruction: ShackHartmannReconstruction,
    sensor_x_m: np.ndarray,
    sensor_y_m: np.ndarray,
    target_X_m: np.ndarray,
    target_Y_m: np.ndarray,
    *,
    wavelength_m: float,
    rotation_rad: float = 0.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    offset_x_m: float = 0.0,
    offset_y_m: float = 0.0,
    gain: float = 1.0,
    outside_value_rad: float = 0.0,
) -> np.ndarray:
    """Interpolate measured OPD to SLM coordinates and return ``-k*OPD``.

    A complete rectangular Shack-Hartmann pupil uses regular-grid linear
    interpolation. A circular/irregular valid-lenslet pupil uses Delaunay-based
    linear interpolation over the measured points. No OPD is extrapolated beyond
    their convex hull; those target pixels receive ``outside_value_rad``.
    """

    if not (0.0 < float(gain) <= 1.0):
        raise ValueError("gain must lie in (0,1]")
    wavelength = float(wavelength_m)
    if wavelength <= 0.0:
        raise ValueError("wavelength_m must be positive")
    x, _ = _uniform_axis(sensor_x_m, "sensor_x_m")
    y, _ = _uniform_axis(sensor_y_m, "sensor_y_m")
    xq, yq = _registered_query_coordinates(
        target_X_m,
        target_Y_m,
        rotation_rad=float(rotation_rad),
        scale_x=float(scale_x),
        scale_y=float(scale_y),
        offset_x_m=float(offset_x_m),
        offset_y_m=float(offset_y_m),
    )
    sampled, _ = _interpolate_reconstructed_opd(reconstruction, x, y, xq, yq)
    phase = -float(gain) * TWOPI * sampled / wavelength
    return np.where(np.isfinite(phase), phase, float(outside_value_rad))


def update_iterative_correction(
    previous_correction_phase_rad: np.ndarray,
    residual_correction_phase_rad: np.ndarray,
    *,
    gain: float = 0.7,
) -> np.ndarray:
    """Add a gain-weighted residual correction for one closed-loop iteration."""

    if not (0.0 < float(gain) <= 1.0):
        raise ValueError("gain must lie in (0,1]")
    previous = np.asarray(previous_correction_phase_rad, dtype=float)
    residual = np.asarray(residual_correction_phase_rad, dtype=float)
    if previous.shape != residual.shape:
        raise ValueError("previous and residual correction maps must match")
    return previous + float(gain) * residual


__all__ = [
    "ShackHartmannReconstruction",
    "correction_phase_on_slm",
    "reconstruct_opd_from_slopes",
    "slopes_from_spot_displacements",
    "update_iterative_correction",
]
