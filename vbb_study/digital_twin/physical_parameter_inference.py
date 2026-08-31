"""Low-dimensional physical-parameter inference from multi-plane intensity stacks.

This module sits between the forward digital-twin model and the experimental
residual-phase retrieval.  It does not claim that a fitted parameter is uniquely
identified by intensity data; instead it provides explicit candidate-grid
metrics so a measured or synthetic z-stack can be compared against physically
meaningful model parameters such as axicon decentre, input pointing, SLM
registration, or Fourier-iris offset.

The intended workflow is:

    measured/synthetic z-stack
        -> generate candidate stacks with the same forward model
        -> rank candidates by a documented multi-plane morphology cost
        -> report the best physical parameter and fit-quality metrics
        -> leave any unexplained residual to the phase-retrieval/correction layer.

A grid-search estimate is not a statistical uncertainty.  ``grid_step`` and the
second-best cost margin are reported so the numerical resolution/identifiability
of the discrete search remains visible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

EPS = np.finfo(float).tiny


def _normalise_plane(plane: np.ndarray) -> np.ndarray:
    arr = np.maximum(np.asarray(plane, dtype=float), 0.0)
    return arr / max(float(np.max(arr)), EPS)


def normalised_stack_rmse(
    candidate: np.ndarray,
    target: np.ndarray,
    *,
    plane_weights: Sequence[float] | None = None,
) -> float:
    """Return per-plane peak-normalised RMSE across a complete z-stack.

    Peak-normalising each plane makes this a morphology metric rather than an
    absolute-power metric.  Absolute throughput should be fitted separately if
    it is experimentally calibrated and scientifically required.
    """

    c = np.asarray(candidate, dtype=float)
    t = np.asarray(target, dtype=float)
    if c.shape != t.shape or c.ndim < 3:
        raise ValueError("candidate and target must be matching [z, y, x] or [z, ...] stacks")
    nz = int(c.shape[0])
    if plane_weights is None:
        weights = np.ones(nz, dtype=float)
    else:
        weights = np.asarray(plane_weights, dtype=float)
        if weights.shape != (nz,):
            raise ValueError("plane_weights must contain one value per z plane")
        if np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
            raise ValueError("plane_weights must be non-negative with non-zero total")
    weights = weights / float(np.sum(weights))

    mse = np.empty(nz, dtype=float)
    for i in range(nz):
        cc = _normalise_plane(c[i])
        tt = _normalise_plane(t[i])
        mse[i] = float(np.mean((cc - tt) ** 2))
    return float(np.sqrt(np.sum(weights * mse)))


@dataclass(frozen=True)
class ScalarParameterFit:
    parameter_name: str
    parameter_unit: str
    candidate_values: tuple[float, ...]
    costs: tuple[float, ...]
    best_value: float
    best_cost: float
    nominal_value: float
    nominal_cost: float
    improvement_fraction_vs_nominal: float
    second_best_value: float
    second_best_cost: float
    second_best_margin: float
    grid_step: float
    cost_metric: str = "per-plane peak-normalised multi-plane RMSE"
    claim_boundary: str = "candidate-grid estimate; not a statistical uncertainty or proof of unique identifiability"

    def as_dict(self) -> dict:
        return asdict(self)


def fit_scalar_parameter(
    target_stack: np.ndarray,
    candidate_stacks: Sequence[np.ndarray],
    candidate_values: Sequence[float],
    *,
    parameter_name: str,
    parameter_unit: str,
    nominal_value: float = 0.0,
    plane_weights: Sequence[float] | None = None,
) -> ScalarParameterFit:
    """Rank a one-dimensional physical-parameter sweep against a target stack."""

    values = np.asarray(candidate_values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("candidate_values must contain at least two scalar candidates")
    if len(candidate_stacks) != values.size:
        raise ValueError("candidate_stacks and candidate_values must have the same length")

    costs = np.asarray(
        [normalised_stack_rmse(stack, target_stack, plane_weights=plane_weights) for stack in candidate_stacks],
        dtype=float,
    )
    order = np.argsort(costs)
    best_idx = int(order[0])
    second_idx = int(order[1])
    nominal_idx = int(np.argmin(np.abs(values - float(nominal_value))))
    best_cost = float(costs[best_idx])
    nominal_cost = float(costs[nominal_idx])
    improvement = float((nominal_cost - best_cost) / max(nominal_cost, EPS))

    unique = np.unique(np.sort(values))
    grid_step = float(np.min(np.diff(unique))) if unique.size > 1 else float("nan")

    return ScalarParameterFit(
        parameter_name=str(parameter_name),
        parameter_unit=str(parameter_unit),
        candidate_values=tuple(float(v) for v in values),
        costs=tuple(float(v) for v in costs),
        best_value=float(values[best_idx]),
        best_cost=best_cost,
        nominal_value=float(values[nominal_idx]),
        nominal_cost=nominal_cost,
        improvement_fraction_vs_nominal=improvement,
        second_best_value=float(values[second_idx]),
        second_best_cost=float(costs[second_idx]),
        second_best_margin=float(costs[second_idx] - best_cost),
        grid_step=grid_step,
    )
