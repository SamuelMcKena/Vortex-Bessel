"""Fit multi-plane data against the physical system-error dictionary.

This module connects the repository's broad forward-error sweep registry to the
inverse workflow.  The simulated error families are therefore not a separate
visual atlas: they are candidate physical models that can be replayed against a
measured or synthetic z-stack.

The intended hierarchy is

    intensity z-stack
        -> rank/fine-fit physically parameterised forward models
        -> update the digital-twin state with supported physical parameters
        -> acquire/reanalyse the residual discrepancy
        -> run the existing q=20 residual-phase retrieval
        -> map the additive residual correction to native SLM2 coordinates

The final residual-phase stage remains the calibrated Miao-style experimental
pipeline in ``notebooks/experimental/axicon_aberration_correction``.  This module
does not turn an intensity difference into a phase map and does not conjugate
the programmed vortex phase.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from vbb_study.digital_twin.physical_error_inference import morphology_rmse
from vbb_study.digital_twin.vortex_system_error_sweeps import system_sweep_registry
from vbb_study.digital_twin.vortex_system_route import SystemErrorConfig


EPS = np.finfo(float).tiny
StackLoss = Callable[[np.ndarray, np.ndarray], float]
ConfigSimulator = Callable[[SystemErrorConfig], np.ndarray]


def _same_value(a: Any, b: Any) -> bool:
    """Scalar/tuple equality helper for the current frozen error dataclasses."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    try:
        return bool(a == b)
    except Exception:
        return False


def _merge_nondefault_dataclass(base: Any, overlay: Any, default: Any) -> Any:
    """Overlay only fields that differ from that dataclass's nominal default.

    Sweep builders intentionally return a complete ``SystemErrorConfig`` with
    only one physical family changed.  Recursive merging lets several fitted
    families coexist without one sweep resetting the previously fitted state.
    """
    if not (is_dataclass(base) and is_dataclass(overlay) and is_dataclass(default)):
        return overlay if not _same_value(overlay, default) else base

    updates: dict[str, Any] = {}
    for f in fields(default):
        b = getattr(base, f.name)
        o = getattr(overlay, f.name)
        d = getattr(default, f.name)
        if is_dataclass(d):
            merged = _merge_nondefault_dataclass(b, o, d)
            if not _same_value(merged, b):
                updates[f.name] = merged
        elif not _same_value(o, d):
            updates[f.name] = o
    return replace(base, **updates) if updates else base


def combine_error_configs(
    base: SystemErrorConfig,
    overlay: SystemErrorConfig,
) -> SystemErrorConfig:
    """Combine independently parameterised physical error families."""
    return _merge_nondefault_dataclass(base, overlay, SystemErrorConfig())


@dataclass(frozen=True)
class FamilyFitResult:
    family: str
    units: str
    fidelity: str
    values: np.ndarray
    costs: np.ndarray
    best_index: int
    best_value: float
    best_cost: float
    baseline_cost: float
    improvement_fraction: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "units": self.units,
            "fidelity": self.fidelity,
            "values": self.values.tolist(),
            "costs": self.costs.tolist(),
            "best_index": int(self.best_index),
            "best_value": float(self.best_value),
            "best_cost": float(self.best_cost),
            "baseline_cost": float(self.baseline_cost),
            "improvement_fraction": float(self.improvement_fraction),
        }


@dataclass(frozen=True)
class GreedyFitStage:
    stage: int
    accepted_family: str | None
    accepted_value: float | None
    accepted_units: str | None
    cost_before: float
    cost_after: float
    improvement_fraction: float
    rankings: tuple[FamilyFitResult, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": int(self.stage),
            "accepted_family": self.accepted_family,
            "accepted_value": self.accepted_value,
            "accepted_units": self.accepted_units,
            "cost_before": float(self.cost_before),
            "cost_after": float(self.cost_after),
            "improvement_fraction": float(self.improvement_fraction),
            "rankings": [r.as_dict() for r in self.rankings],
        }


@dataclass(frozen=True)
class GreedyPhysicalFitResult:
    initial_cost: float
    final_cost: float
    fitted_config: SystemErrorConfig
    stages: tuple[GreedyFitStage, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "initial_cost": float(self.initial_cost),
            "final_cost": float(self.final_cost),
            "fitted_config": asdict(self.fitted_config),
            "stages": [s.as_dict() for s in self.stages],
        }


def fit_one_family(
    *,
    family: str,
    target_stack: np.ndarray,
    simulate_config: ConfigSimulator,
    base_config: SystemErrorConfig = SystemErrorConfig(),
    registry: Mapping[str, Mapping[str, Any]] | None = None,
    loss_fn: StackLoss | None = None,
) -> FamilyFitResult:
    """Fit one registered physical error family on top of ``base_config``."""
    reg = system_sweep_registry() if registry is None else registry
    if family not in reg:
        raise KeyError(f"unknown physical error family: {family}")
    spec = reg[family]
    values = np.asarray(tuple(spec["values"]), dtype=float)
    if values.ndim != 1 or values.size < 3:
        raise ValueError(f"family {family!r} does not contain a usable 1-D sweep")
    loss = morphology_rmse if loss_fn is None else loss_fn
    target = np.asarray(target_stack, dtype=float)

    baseline_stack = np.asarray(simulate_config(base_config), dtype=float)
    baseline_cost = float(loss(baseline_stack, target))
    costs = np.empty(values.size, dtype=float)
    for i, value in enumerate(values):
        candidate = combine_error_configs(base_config, spec["builder"](float(value)))
        costs[i] = float(loss(np.asarray(simulate_config(candidate), dtype=float), target))

    best_index = int(np.argmin(costs))
    best_cost = float(costs[best_index])
    improvement = float((baseline_cost - best_cost) / max(baseline_cost, EPS))
    return FamilyFitResult(
        family=str(family),
        units=str(spec.get("units", "")),
        fidelity=str(spec.get("fidelity", "unspecified")),
        values=values,
        costs=costs,
        best_index=best_index,
        best_value=float(values[best_index]),
        best_cost=best_cost,
        baseline_cost=baseline_cost,
        improvement_fraction=improvement,
    )


def rank_error_families(
    *,
    families: Sequence[str],
    target_stack: np.ndarray,
    simulate_config: ConfigSimulator,
    base_config: SystemErrorConfig = SystemErrorConfig(),
    registry: Mapping[str, Mapping[str, Any]] | None = None,
    loss_fn: StackLoss | None = None,
) -> list[FamilyFitResult]:
    """Rank candidate physical causes by their best forward-model fit."""
    results = [
        fit_one_family(
            family=family,
            target_stack=target_stack,
            simulate_config=simulate_config,
            base_config=base_config,
            registry=registry,
            loss_fn=loss_fn,
        )
        for family in families
    ]
    return sorted(results, key=lambda r: (r.best_cost, -r.improvement_fraction, r.family))


def greedy_fit_error_dictionary(
    *,
    families: Sequence[str],
    target_stack: np.ndarray,
    simulate_config: ConfigSimulator,
    max_stages: int = 2,
    minimum_improvement_fraction: float = 0.02,
    base_config: SystemErrorConfig = SystemErrorConfig(),
    registry: Mapping[str, Mapping[str, Any]] | None = None,
    loss_fn: StackLoss | None = None,
) -> GreedyPhysicalFitResult:
    """Greedily add distinct physical families when they improve the z-stack fit.

    This is deliberately a transparent bounded search, not a claim that the
    global multi-parameter inverse problem is solved.  It is useful for turning
    the existing sweep atlas into a first diagnostic layer and for deciding
    which low-dimensional physical state should be carried into a later refined
    fit.
    """
    if max_stages < 1:
        raise ValueError("max_stages must be >= 1")
    if minimum_improvement_fraction < 0:
        raise ValueError("minimum_improvement_fraction must be non-negative")

    loss = morphology_rmse if loss_fn is None else loss_fn
    target = np.asarray(target_stack, dtype=float)
    current = base_config
    current_cost = float(loss(np.asarray(simulate_config(current), dtype=float), target))
    initial_cost = current_cost
    remaining = list(dict.fromkeys(str(f) for f in families))
    stages: list[GreedyFitStage] = []
    reg = system_sweep_registry() if registry is None else registry

    for stage_index in range(1, int(max_stages) + 1):
        if not remaining:
            break
        rankings = rank_error_families(
            families=remaining,
            target_stack=target,
            simulate_config=simulate_config,
            base_config=current,
            registry=reg,
            loss_fn=loss,
        )
        best = rankings[0]
        improvement = float((current_cost - best.best_cost) / max(current_cost, EPS))
        if improvement < float(minimum_improvement_fraction):
            stages.append(GreedyFitStage(
                stage=stage_index,
                accepted_family=None,
                accepted_value=None,
                accepted_units=None,
                cost_before=current_cost,
                cost_after=current_cost,
                improvement_fraction=improvement,
                rankings=tuple(rankings),
            ))
            break

        current = combine_error_configs(current, reg[best.family]["builder"](best.best_value))
        stages.append(GreedyFitStage(
            stage=stage_index,
            accepted_family=best.family,
            accepted_value=best.best_value,
            accepted_units=best.units,
            cost_before=current_cost,
            cost_after=best.best_cost,
            improvement_fraction=improvement,
            rankings=tuple(rankings),
        ))
        current_cost = best.best_cost
        remaining.remove(best.family)

    return GreedyPhysicalFitResult(
        initial_cost=initial_cost,
        final_cost=current_cost,
        fitted_config=current,
        stages=tuple(stages),
    )


def correction_handoff_manifest(
    fit: GreedyPhysicalFitResult,
    *,
    require_new_measurement_after_physical_adjustment: bool = True,
) -> dict[str, Any]:
    """Describe the safe hand-off from physical fitting to q=20 phase retrieval.

    The q=20 code remains authoritative for residual phase reconstruction and
    SLM2 mapping.  A physical fit can guide alignment/model state, but an
    intensity residual is not itself a phase correction.
    """
    accepted = [
        {
            "family": stage.accepted_family,
            "value": stage.accepted_value,
            "units": stage.accepted_units,
        }
        for stage in fit.stages
        if stage.accepted_family is not None
    ]
    return {
        "workflow": "physical-model fit -> residual-phase retrieval -> additive SLM2 correction",
        "physical_fit": {
            "accepted_parameters": accepted,
            "initial_cost": float(fit.initial_cost),
            "final_cost": float(fit.final_cost),
            "fitted_system_error_config": asdict(fit.fitted_config),
        },
        "next_measurement": (
            "acquire an independent z-stack after any physical alignment/hardware adjustment"
            if require_new_measurement_after_physical_adjustment
            else "use a provenance-controlled stack appropriate to the residual retrieval"
        ),
        "residual_phase_stage": {
            "authoritative_runner": "notebooks/experimental/axicon_aberration_correction/run_q20_miao_retrieval.py",
            "retrieved_phase": "retrieved_full_residual_phase_input_plane_rad.npy",
            "input_plane_correction": "conjugate_correction_input_plane_rad.npy",
            "native_slm2_correction": "slm2_correction_phase_rad.npy",
            "programmed_vortex_removed_from_aberration": True,
        },
        "application": "add the calibrated low-gain residual correction to the existing SLM2 programmed phase, wrap once, then encode with the measured LUT",
        "not_permitted": [
            "treating target-minus-model intensity as a phase map",
            "conjugating the complete structured field including q*theta",
            "claiming an SLM2 correction before branch/coordinate/LUT calibration",
        ],
    }
