"""Residual phase estimation through a supplied full optical forward model.

This module provides the model-based residual stage used after physical system
parameters have been estimated or calibrated.  It intentionally does not know
about a particular poster benchmark: callers provide a function that propagates
a candidate transverse phase through the actual digital twin and returns the
predicted intensity stack.

The unknown residual is represented by a compact angular Fourier basis,

    psi(theta) = sum_m [a_m cos(m theta) + b_m sin(m theta)],

with the deliberately programmed vortex phase excluded.  Fitting is performed
only on explicitly supplied training z planes.  A separate scoring helper is
provided for held-out planes so the same data cannot silently validate the
optimizer that fitted them.

For the q=20 workflow the Miao et al. retrieval is a useful analytical
initialisation/baseline, while this stage removes the stationary-phase forward
approximation by evaluating candidate phases through the complete numerical
optical route.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import optimize

EPS = np.finfo(float).tiny
PhaseStackSimulator = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ResidualPhaseFit:
    modes: tuple[int, ...]
    coefficients_rad: tuple[float, ...]
    initial_cost: float
    final_cost: float
    success: bool
    message: str
    iterations: int
    function_evaluations: int
    train_indices: tuple[int, ...]

    def as_dict(self) -> dict:
        return {
            "modes": list(self.modes),
            "coefficients_rad": list(self.coefficients_rad),
            "initial_cost": self.initial_cost,
            "final_cost": self.final_cost,
            "success": self.success,
            "message": self.message,
            "iterations": self.iterations,
            "function_evaluations": self.function_evaluations,
            "train_indices": list(self.train_indices),
        }


def angular_phase_from_coefficients(
    theta_rad: np.ndarray,
    coefficients_rad: Sequence[float],
    *,
    modes: Sequence[int] = tuple(range(1, 7)),
) -> np.ndarray:
    """Evaluate an angular residual phase; q*theta is intentionally absent."""
    modes_t = tuple(int(m) for m in modes)
    c = np.asarray(coefficients_rad, dtype=float)
    if c.shape != (2 * len(modes_t),):
        raise ValueError("coefficients must contain cosine/sine pairs for every mode")
    theta = np.asarray(theta_rad, dtype=float)
    phase = np.zeros_like(theta, dtype=float)
    for j, m in enumerate(modes_t):
        phase += c[2*j] * np.cos(m*theta) + c[2*j+1] * np.sin(m*theta)
    return phase


def _crop_indices(x_m: np.ndarray, half_width_m: float) -> np.ndarray:
    x = np.asarray(x_m, dtype=float)
    ids = np.flatnonzero(np.abs(x) <= float(half_width_m))
    if ids.size < 8:
        raise ValueError("phase-fit crop is too small for the supplied grid")
    return ids


def plane_normalise(stack: np.ndarray) -> np.ndarray:
    a = np.maximum(np.asarray(stack, dtype=float), 0.0)
    if a.ndim != 3:
        raise ValueError("stack must have shape (z,y,x)")
    return a / np.maximum(np.max(a, axis=(1, 2), keepdims=True), EPS)


def cropped_stack_rmse(
    model: np.ndarray,
    data: np.ndarray,
    *,
    x_m: np.ndarray,
    half_width_m: float,
) -> float:
    """Plane-normalized intensity RMSE on a fixed laboratory-coordinate crop."""
    m = np.asarray(model, dtype=float)
    d = np.asarray(data, dtype=float)
    if m.shape != d.shape or m.ndim != 3:
        raise ValueError("model and data stacks must have the same (z,y,x) shape")
    ids = _crop_indices(x_m, half_width_m)
    mn = plane_normalise(m[:, ids[:, None], ids])
    dn = plane_normalise(d[:, ids[:, None], ids])
    return float(np.sqrt(np.mean((mn-dn)**2)))


def fit_angular_residual_phase(
    measured_stack: np.ndarray,
    z_m: np.ndarray,
    *,
    theta_grid_rad: np.ndarray,
    x_m: np.ndarray,
    simulate_phase_stack: PhaseStackSimulator,
    train_indices: Sequence[int],
    modes: Sequence[int] = tuple(range(1, 7)),
    initial_coefficients_rad: Sequence[float] | None = None,
    phase_bound_rad: float = 1.0,
    crop_half_width_m: float = 1.15e-3,
    regularization: float = 2e-4,
    maxiter: int = 32,
) -> ResidualPhaseFit:
    """Fit a compact angular residual using the caller's complete forward model.

    ``simulate_phase_stack(phase, z_subset)`` must apply ``phase`` at the chosen
    residual/input plane of the digital twin and return an intensity stack at the
    requested z values.  No re-centering is performed.
    """
    data = np.asarray(measured_stack, dtype=float)
    z = np.asarray(z_m, dtype=float)
    if data.ndim != 3 or data.shape[0] != z.size:
        raise ValueError("measured_stack and z_m dimensions are inconsistent")
    train = np.asarray(tuple(int(i) for i in train_indices), dtype=int)
    if train.size < 2 or np.any(train < 0) or np.any(train >= z.size):
        raise ValueError("train_indices must contain at least two valid z indices")
    modes_t = tuple(int(m) for m in modes)
    ncoef = 2*len(modes_t)
    x0 = np.zeros(ncoef, dtype=float) if initial_coefficients_rad is None else np.asarray(initial_coefficients_rad, dtype=float)
    if x0.shape != (ncoef,):
        raise ValueError("initial coefficient vector has the wrong length")
    ids = _crop_indices(x_m, crop_half_width_m)
    target = plane_normalise(data[train][:, ids[:, None], ids])
    z_train = z[train]

    def objective(c: np.ndarray) -> float:
        phase = angular_phase_from_coefficients(theta_grid_rad, c, modes=modes_t)
        pred = np.asarray(simulate_phase_stack(phase, z_train), dtype=float)
        if pred.shape[0] != train.size:
            raise ValueError("forward phase simulator returned the wrong number of z planes")
        pred = plane_normalise(pred[:, ids[:, None], ids])
        data_term = float(np.sqrt(np.mean((pred-target)**2)))
        return data_term + float(regularization)*float(np.mean(np.asarray(c, float)**2))

    initial_cost = float(objective(x0))
    bounds = [(-float(phase_bound_rad), float(phase_bound_rad))] * ncoef
    result = optimize.minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(maxiter), "ftol": 2e-8, "gtol": 1e-5, "maxls": 20},
    )
    return ResidualPhaseFit(
        modes=modes_t,
        coefficients_rad=tuple(float(v) for v in result.x),
        initial_cost=initial_cost,
        final_cost=float(result.fun),
        success=bool(result.success),
        message=str(result.message),
        iterations=int(result.nit),
        function_evaluations=int(result.nfev),
        train_indices=tuple(int(i) for i in train),
    )


def score_residual_phase_on_indices(
    measured_stack: np.ndarray,
    z_m: np.ndarray,
    *,
    phase_rad: np.ndarray,
    simulate_phase_stack: PhaseStackSimulator,
    indices: Sequence[int],
    x_m: np.ndarray,
    crop_half_width_m: float = 1.15e-3,
) -> dict:
    """Evaluate a fixed phase on z planes that were not used to fit it."""
    data = np.asarray(measured_stack, dtype=float)
    z = np.asarray(z_m, dtype=float)
    ids_z = np.asarray(tuple(int(i) for i in indices), dtype=int)
    if ids_z.size == 0:
        raise ValueError("at least one scoring index is required")
    crop = _crop_indices(x_m, crop_half_width_m)
    pred = np.asarray(simulate_phase_stack(np.asarray(phase_rad, float), z[ids_z]), float)
    pn = plane_normalise(pred[:, crop[:, None], crop])
    dn = plane_normalise(data[ids_z][:, crop[:, None], crop])
    rows = []
    for j, iz in enumerate(ids_z):
        a = pn[j].ravel(); b = dn[j].ravel()
        rows.append({
            "index": int(iz),
            "z_m": float(z[iz]),
            "pearson_r": float(np.corrcoef(a, b)[0, 1]),
            "rmse": float(np.sqrt(np.mean((a-b)**2))),
        })
    return {
        "mean_pearson_r": float(np.mean([r["pearson_r"] for r in rows])),
        "mean_rmse": float(np.mean([r["rmse"] for r in rows])),
        "planes": rows,
    }
