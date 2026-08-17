"""Empirical material-response layer that is impossible to use uncalibrated.

The optical digital twin predicts incident field, fluence, peak intensity and
pulse exposure.  Permanent material modification is a separate physical system.
This module therefore offers only an explicitly empirical binary response model
that must be fitted to measured processing outcomes.

The logistic model uses log fluence, log effective pulse count and their
interaction.  It is a compact response-surface tool, not a mechanistic plasma,
thermal, stress or refractive-index-change solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
from scipy.optimize import minimize


EPS = np.finfo(float).tiny


@dataclass(frozen=True)
class MaterialResponseFit:
    material_name: str
    coefficients: np.ndarray
    covariance: np.ndarray
    sample_count: int
    negative_log_likelihood: float
    converged: bool
    metadata: Mapping[str, object]


def _design(fluence: np.ndarray, pulses: np.ndarray) -> np.ndarray:
    F = np.maximum(np.asarray(fluence, dtype=float), EPS)
    N = np.maximum(np.asarray(pulses, dtype=float), 1.0)
    if F.shape != N.shape:
        raise ValueError("fluence and pulse-count arrays must match")
    lf = np.log(F)
    ln = np.log(N)
    return np.column_stack([np.ones(F.size), lf.ravel(), ln.ravel(), (lf * ln).ravel()])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(np.asarray(z, dtype=float))
    positive = z >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    ez = np.exp(z[~positive])
    out[~positive] = ez / (1.0 + ez)
    return out


def fit_binary_material_response(
    fluence_J_cm2: np.ndarray,
    effective_pulses: np.ndarray,
    modified: np.ndarray,
    *,
    material_name: str,
    l2_regularisation: float = 1e-6,
) -> MaterialResponseFit:
    """Fit an empirical modification-probability response surface."""

    X = _design(fluence_J_cm2, effective_pulses)
    y = np.asarray(modified, dtype=float).ravel()
    if y.size != X.shape[0] or y.size < 12:
        raise ValueError("material response fit requires >=12 matched observations")
    if np.any((y != 0.0) & (y != 1.0)):
        raise ValueError("modified must be binary 0/1 observations")
    if np.all(y == y[0]):
        raise ValueError("material response dataset must contain both modified and unmodified outcomes")
    lam = float(l2_regularisation)
    if lam < 0.0:
        raise ValueError("l2_regularisation cannot be negative")

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        z = X @ beta
        p = np.clip(_sigmoid(z), 1e-12, 1.0 - 1e-12)
        nll = -float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
        penalty = 0.5 * lam * float(np.dot(beta[1:], beta[1:]))
        grad = X.T @ (p - y)
        grad[1:] += lam * beta[1:]
        return nll + penalty, np.asarray(grad, dtype=float)

    result = minimize(
        lambda b: objective(b)[0],
        np.zeros(X.shape[1], dtype=float),
        jac=lambda b: objective(b)[1],
        method="BFGS",
    )
    beta = np.asarray(result.x, dtype=float)
    p = _sigmoid(X @ beta)
    W = np.maximum(p * (1.0 - p), 1e-9)
    hessian = X.T @ (W[:, None] * X)
    hessian[1:, 1:] += lam * np.eye(X.shape[1] - 1)
    covariance = np.linalg.pinv(hessian)
    return MaterialResponseFit(
        material_name=str(material_name),
        coefficients=beta,
        covariance=np.asarray(covariance, dtype=float),
        sample_count=int(y.size),
        negative_log_likelihood=float(objective(beta)[0]),
        converged=bool(result.success),
        metadata={
            "model": "empirical_logistic_log_fluence_log_pulses_interaction",
            "mechanistic_material_model": False,
            "requires_independent_validation": True,
            "l2_regularisation": lam,
        },
    )


def modification_probability(
    fit: MaterialResponseFit,
    fluence_J_cm2: np.ndarray | float,
    effective_pulses: np.ndarray | float,
) -> np.ndarray:
    """Evaluate a previously fitted empirical material response."""

    if not fit.converged:
        raise ValueError("material response fit did not converge; prediction is blocked")
    F, N = np.broadcast_arrays(np.asarray(fluence_J_cm2, dtype=float), np.asarray(effective_pulses, dtype=float))
    X = _design(F, N)
    return _sigmoid(X @ np.asarray(fit.coefficients, dtype=float)).reshape(F.shape)


__all__ = ["MaterialResponseFit", "fit_binary_material_response", "modification_probability"]
