"""Low-dimensional physical-error inference against multi-plane intensity data.

This module is intentionally small and model-facing.  It does not claim that an
experimental z-stack uniquely identifies every bench error.  Instead it provides
the reusable objective machinery needed to test whether a chosen physical
parameter is identifiable from a measured or synthetic intensity stack.

The intended workflow is:

    target z-stack -> candidate SystemErrorConfig values -> forward model ->
    plane-normalised morphology loss -> best physical parameter + diagnostics

Synthetic benchmarks can therefore report quantities such as an injected versus
recovered axicon decentre, beam pointing angle, SLM registration offset, or 4F
iris offset.  Experimental use must additionally report calibration provenance,
parameter bounds and identifiability; a low loss alone is not evidence that a
single physical cause is unique.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

EPS = np.finfo(float).tiny


@dataclass(frozen=True)
class ParameterFitResult:
    """Result of one bounded one-parameter forward-model grid search."""

    parameter: str
    units: str
    candidate_values: np.ndarray
    costs: np.ndarray
    best_index: int
    best_value: float
    best_cost: float
    second_best_cost: float
    relative_cost_margin: float

    def as_dict(self) -> dict[str, object]:
        return {
            "parameter": self.parameter,
            "units": self.units,
            "candidate_values": self.candidate_values.tolist(),
            "costs": self.costs.tolist(),
            "best_index": int(self.best_index),
            "best_value": float(self.best_value),
            "best_cost": float(self.best_cost),
            "second_best_cost": float(self.second_best_cost),
            "relative_cost_margin": float(self.relative_cost_margin),
        }


def plane_normalise_stack(stack: np.ndarray) -> np.ndarray:
    """Peak-normalise each z plane independently while preserving coordinates."""

    arr = np.maximum(np.asarray(stack, dtype=float), 0.0)
    if arr.ndim != 3:
        raise ValueError("intensity stack must have shape (z, y, x) or equivalent 3-D form")
    flat = arr.reshape(arr.shape[0], -1)
    peaks = np.max(flat, axis=1)
    peaks = np.maximum(peaks, EPS)
    return arr / peaks[:, None, None]


def morphology_rmse(candidate_stack: np.ndarray, target_stack: np.ndarray) -> float:
    """Return equal-plane-weighted RMSE after independent plane normalisation."""

    candidate = plane_normalise_stack(candidate_stack)
    target = plane_normalise_stack(target_stack)
    if candidate.shape != target.shape:
        raise ValueError(f"stack shape mismatch: {candidate.shape} != {target.shape}")
    per_plane = np.sqrt(np.mean((candidate - target) ** 2, axis=(1, 2)))
    return float(np.mean(per_plane))


def grid_search_parameter(
    *,
    parameter: str,
    units: str,
    candidate_values: Sequence[float],
    target_stack: np.ndarray,
    simulate: Callable[[float], np.ndarray],
) -> ParameterFitResult:
    """Fit one physical parameter by replaying candidates through the forward model.

    ``simulate(value)`` must return an intensity stack on exactly the same fixed
    laboratory coordinates and z planes as ``target_stack``.  No recentering is
    performed here because centroid walk is part of the diagnostic signal.
    """

    values = np.asarray(list(candidate_values), dtype=float)
    if values.ndim != 1 or values.size < 3:
        raise ValueError("candidate_values must contain at least three 1-D values")
    target = np.asarray(target_stack, dtype=float)
    costs = np.asarray([morphology_rmse(simulate(float(v)), target) for v in values], dtype=float)
    order = np.argsort(costs)
    best_index = int(order[0])
    best_cost = float(costs[best_index])
    second = float(costs[int(order[1])])
    margin = float((second - best_cost) / max(second, EPS))
    return ParameterFitResult(
        parameter=str(parameter),
        units=str(units),
        candidate_values=values,
        costs=costs,
        best_index=best_index,
        best_value=float(values[best_index]),
        best_cost=best_cost,
        second_best_cost=second,
        relative_cost_margin=margin,
    )
