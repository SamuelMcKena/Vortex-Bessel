"""Low-dimensional physical-error inference against multi-plane intensity data.

This module is intentionally model-facing.  It does not claim that an
experimental z-stack uniquely identifies every bench error.  Instead it provides
reusable objective machinery for asking whether selected physical parameters are
identifiable from measured or synthetic multi-plane intensity data.

The intended workflow is:

    target z-stack -> candidate SystemErrorConfig values -> forward model ->
    plane-normalised morphology loss -> best physical parameters + diagnostics

Synthetic benchmarks can therefore report injected versus recovered axicon
decentre, beam pointing angle, SLM registration offset, 4F iris offset, or small
joint parameter sets.  Experimental use must additionally report calibration
provenance, parameter bounds and identifiability; a low loss alone is not
evidence that a fitted physical cause is unique.
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


@dataclass(frozen=True)
class TwoParameterFitResult:
    """Result of a bounded two-parameter forward-model grid search.

    The result deliberately reports a cost landscape and a best-versus-second
    separation metric rather than pretending that a discrete deterministic grid
    supplies a statistical uncertainty.  Experimental confidence intervals must
    come from a separate noise/provenance-aware analysis.
    """

    parameter_x: str
    units_x: str
    values_x: np.ndarray
    parameter_y: str
    units_y: str
    values_y: np.ndarray
    costs: np.ndarray
    best_index_yx: tuple[int, int]
    best_x: float
    best_y: float
    best_cost: float
    second_best_cost: float
    relative_cost_margin: float

    def as_dict(self) -> dict[str, object]:
        return {
            "parameter_x": self.parameter_x,
            "units_x": self.units_x,
            "values_x": self.values_x.tolist(),
            "parameter_y": self.parameter_y,
            "units_y": self.units_y,
            "values_y": self.values_y.tolist(),
            "costs": self.costs.tolist(),
            "best_index_yx": [int(v) for v in self.best_index_yx],
            "best_x": float(self.best_x),
            "best_y": float(self.best_y),
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
    performed because centroid walk is part of the diagnostic signal.
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


def grid_search_two_parameters(
    *,
    parameter_x: str,
    units_x: str,
    values_x: Sequence[float],
    parameter_y: str,
    units_y: str,
    values_y: Sequence[float],
    target_stack: np.ndarray,
    simulate: Callable[[float, float], np.ndarray],
) -> TwoParameterFitResult:
    """Fit two selected physical parameters by a full bounded grid search.

    The first parameter indexes columns of the returned cost array and the second
    indexes rows.  Keeping the complete landscape is useful for diagnosing broad
    valleys and parameter degeneracy instead of reporting only an optimizer's
    point estimate.
    """

    vx = np.asarray(list(values_x), dtype=float)
    vy = np.asarray(list(values_y), dtype=float)
    if vx.ndim != 1 or vy.ndim != 1 or vx.size < 3 or vy.size < 3:
        raise ValueError("values_x and values_y must each contain at least three values")
    target = np.asarray(target_stack, dtype=float)
    costs = np.empty((vy.size, vx.size), dtype=float)
    for iy, y in enumerate(vy):
        for ix, x in enumerate(vx):
            costs[iy, ix] = morphology_rmse(simulate(float(x), float(y)), target)

    flat_order = np.argsort(costs, axis=None)
    best_flat = int(flat_order[0])
    second_flat = int(flat_order[1])
    best_iy, best_ix = np.unravel_index(best_flat, costs.shape)
    best_cost = float(costs[best_iy, best_ix])
    second = float(costs.flat[second_flat])
    margin = float((second - best_cost) / max(second, EPS))
    return TwoParameterFitResult(
        parameter_x=str(parameter_x),
        units_x=str(units_x),
        values_x=vx,
        parameter_y=str(parameter_y),
        units_y=str(units_y),
        values_y=vy,
        costs=costs,
        best_index_yx=(int(best_iy), int(best_ix)),
        best_x=float(vx[best_ix]),
        best_y=float(vy[best_iy]),
        best_cost=best_cost,
        second_best_cost=second,
        relative_cost_margin=margin,
    )
