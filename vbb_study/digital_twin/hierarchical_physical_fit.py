"""Hierarchical fitting of the physical Vortex-Bessel digital twin.

This module is the bridge between the forward error-study library and the
intensity-only correction workflow.  It deliberately reuses
``system_sweep_registry()``: the same beam, SLM, 4F and axicon perturbations used
for sensitivity studies become candidate physical explanations of a measured
multi-plane intensity data set.

The fit is staged rather than a single opaque 40+ dimensional optimisation:

    measured z-stack
      -> replay each candidate error family through the complete optical route
      -> rank the candidate families on fixed laboratory coordinates
      -> accumulate the best supported physical perturbation in SystemErrorConfig
      -> repeat until no useful physical improvement remains
      -> hand the remaining optical mismatch to residual-phase retrieval

The fitter is intentionally agnostic about how a stack is generated.  Production
use normally supplies a simulator built from ``build_system_route`` plus free-space
propagation, while unit tests can use a lightweight surrogate.  The default loss
is the fixed-coordinate, plane-normalised morphology RMSE from
``physical_error_inference``.  No recentering is performed.

A low forward-model loss is not, by itself, proof of a unique physical cause.
The returned trace therefore retains best/second parameter and family separation,
loss improvement and the full per-family profiles.  Poster/rendering code can use
those diagnostics without turning them into statistical confidence intervals.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from vbb_study.digital_twin.physical_error_inference import (
    ParameterFitResult,
    StackLoss,
    grid_search_parameter,
    morphology_rmse,
)
from vbb_study.digital_twin.vortex_system_error_sweeps import system_sweep_registry
from vbb_study.digital_twin.vortex_system_route import SystemErrorConfig

EPS = np.finfo(float).tiny
ConfigSimulator = Callable[[SystemErrorConfig], np.ndarray]


@dataclass(frozen=True)
class FamilyRanking:
    family: str
    units: str
    fidelity: str
    fit: ParameterFitResult
    improvement_fraction: float

    def as_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "units": self.units,
            "fidelity": self.fidelity,
            "improvement_fraction": float(self.improvement_fraction),
            "fit": self.fit.as_dict(),
        }


@dataclass(frozen=True)
class PhysicalFitStep:
    stage: int
    cost_before: float
    accepted: bool
    selected_family: str | None
    selected_value: float | None
    selected_units: str | None
    cost_after: float
    improvement_fraction: float
    family_separation: float
    parameter_separation: float
    rankings: tuple[FamilyRanking, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": int(self.stage),
            "cost_before": float(self.cost_before),
            "accepted": bool(self.accepted),
            "selected_family": self.selected_family,
            "selected_value": None if self.selected_value is None else float(self.selected_value),
            "selected_units": self.selected_units,
            "cost_after": float(self.cost_after),
            "improvement_fraction": float(self.improvement_fraction),
            "family_separation": float(self.family_separation),
            "parameter_separation": float(self.parameter_separation),
            "rankings": [r.as_dict() for r in self.rankings],
        }


@dataclass(frozen=True)
class HierarchicalPhysicalFitResult:
    initial_cost: float
    final_cost: float
    final_config: SystemErrorConfig
    steps: tuple[PhysicalFitStep, ...]
    fitted_families: tuple[str, ...]

    @property
    def total_improvement_fraction(self) -> float:
        return float((self.initial_cost - self.final_cost) / max(self.initial_cost, EPS))

    def as_dict(self) -> dict[str, object]:
        return {
            "initial_cost": float(self.initial_cost),
            "final_cost": float(self.final_cost),
            "total_improvement_fraction": self.total_improvement_fraction,
            "fitted_families": list(self.fitted_families),
            "steps": [s.as_dict() for s in self.steps],
        }


def _same_value(a: object, b: object) -> bool:
    if is_dataclass(a) and is_dataclass(b):
        return all(_same_value(getattr(a, f.name), getattr(b, f.name)) for f in fields(a))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    try:
        return bool(a == b)
    except Exception:
        return False


def _overlay_dataclass(base: object, patch: object, default: object) -> object:
    """Overlay only non-default fields from ``patch`` onto ``base`` recursively."""
    if not (is_dataclass(base) and is_dataclass(patch) and is_dataclass(default)):
        return patch if not _same_value(patch, default) else base
    updates: dict[str, object] = {}
    for f in fields(base):
        bv = getattr(base, f.name)
        pv = getattr(patch, f.name)
        dv = getattr(default, f.name)
        if is_dataclass(bv) and is_dataclass(pv) and is_dataclass(dv):
            merged = _overlay_dataclass(bv, pv, dv)
            if not _same_value(merged, bv):
                updates[f.name] = merged
        elif not _same_value(pv, dv):
            updates[f.name] = pv
    return replace(base, **updates) if updates else base


def overlay_error_config(base: SystemErrorConfig, patch: SystemErrorConfig) -> SystemErrorConfig:
    """Accumulate one registry-generated perturbation without resetting prior fits."""
    return _overlay_dataclass(base, patch, SystemErrorConfig())  # type: ignore[return-value]


def apply_registry_family(
    config: SystemErrorConfig,
    family: str,
    value: float,
    *,
    registry: Mapping[str, Mapping[str, object]] | None = None,
) -> SystemErrorConfig:
    reg = system_sweep_registry() if registry is None else registry
    if family not in reg:
        raise KeyError(f"unknown physical-error family {family!r}")
    builder = reg[family].get("builder")
    if not callable(builder):
        raise TypeError(f"registry family {family!r} has no callable builder")
    patch = builder(float(value))
    if not isinstance(patch, SystemErrorConfig):
        raise TypeError(f"registry builder {family!r} did not return SystemErrorConfig")
    return overlay_error_config(config, patch)


def registry_family_groups(
    families: Iterable[str] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Return the model library grouped by physical plane for display/auditing."""
    reg = system_sweep_registry()
    selected = tuple(reg) if families is None else tuple(families)
    groups: dict[str, list[str]] = {"beam": [], "slm": [], "4f": [], "axicon": [], "other": []}
    for family in selected:
        if family.startswith("beam_"):
            groups["beam"].append(family)
        elif family.startswith("slm"):
            groups["slm"].append(family)
        elif family.startswith("fourf_"):
            groups["4f"].append(family)
        elif family.startswith("axicon_"):
            groups["axicon"].append(family)
        else:
            groups["other"].append(family)
    return {k: tuple(v) for k, v in groups.items() if v}


def rank_error_families(
    *,
    target_stack: np.ndarray,
    current_config: SystemErrorConfig,
    simulate_config: ConfigSimulator,
    families: Sequence[str] | None = None,
    registry: Mapping[str, Mapping[str, object]] | None = None,
    loss_fn: StackLoss | None = None,
    current_cost: float | None = None,
) -> tuple[FamilyRanking, ...]:
    """Replay every requested registry family through the current digital twin."""
    reg = system_sweep_registry() if registry is None else registry
    use = tuple(reg.keys()) if families is None else tuple(families)
    target = np.asarray(target_stack, dtype=float)
    loss = morphology_rmse if loss_fn is None else loss_fn
    if current_cost is None:
        current_cost = float(loss(simulate_config(current_config), target))

    rankings: list[FamilyRanking] = []
    for family in use:
        if family not in reg:
            raise KeyError(f"unknown physical-error family {family!r}")
        entry = reg[family]
        values = tuple(float(v) for v in entry["values"])  # type: ignore[index]

        def simulate_value(v: float, fam: str = family) -> np.ndarray:
            cfg = apply_registry_family(current_config, fam, v, registry=reg)
            return np.asarray(simulate_config(cfg), dtype=float)

        fit = grid_search_parameter(
            parameter=family,
            units=str(entry.get("units", "")),
            candidate_values=values,
            target_stack=target,
            simulate=simulate_value,
            loss_fn=loss,
        )
        improvement = float((current_cost - fit.best_cost) / max(current_cost, EPS))
        rankings.append(FamilyRanking(
            family=family,
            units=str(entry.get("units", "")),
            fidelity=str(entry.get("fidelity", "")),
            fit=fit,
            improvement_fraction=improvement,
        ))
    rankings.sort(key=lambda r: r.fit.best_cost)
    return tuple(rankings)


def hierarchical_physical_fit(
    *,
    target_stack: np.ndarray,
    simulate_config: ConfigSimulator,
    initial_config: SystemErrorConfig = SystemErrorConfig(),
    families: Sequence[str] | None = None,
    registry: Mapping[str, Mapping[str, object]] | None = None,
    max_stages: int = 4,
    min_improvement_fraction: float = 0.01,
    min_family_separation: float = 0.0,
    min_parameter_separation: float = 0.0,
    loss_fn: StackLoss | None = None,
) -> HierarchicalPhysicalFitResult:
    """Sequentially accumulate physical perturbations supported by the z-stack.

    ``families=None`` means every family in ``system_sweep_registry`` participates.
    A family is fitted at most once per run.  The separation thresholds are
    diagnostics/safeguards, not confidence levels.
    """
    if max_stages < 1:
        raise ValueError("max_stages must be >= 1")
    reg = system_sweep_registry() if registry is None else registry
    available = list(reg.keys()) if families is None else list(families)
    target = np.asarray(target_stack, dtype=float)
    loss = morphology_rmse if loss_fn is None else loss_fn

    config = initial_config
    initial_cost = float(loss(simulate_config(config), target))
    current_cost = initial_cost
    fitted: list[str] = []
    steps: list[PhysicalFitStep] = []

    for stage in range(1, int(max_stages) + 1):
        remaining = [f for f in available if f not in fitted]
        if not remaining:
            break
        rankings = rank_error_families(
            target_stack=target,
            current_config=config,
            simulate_config=simulate_config,
            families=remaining,
            registry=reg,
            loss_fn=loss,
            current_cost=current_cost,
        )
        best = rankings[0]
        second_family_cost = rankings[1].fit.best_cost if len(rankings) > 1 else np.inf
        family_sep = float((second_family_cost - best.fit.best_cost) / max(second_family_cost, EPS)) if np.isfinite(second_family_cost) else 1.0
        improvement = float((current_cost - best.fit.best_cost) / max(current_cost, EPS))
        parameter_sep = float(best.fit.relative_cost_margin)
        accepted = bool(
            improvement >= float(min_improvement_fraction)
            and family_sep >= float(min_family_separation)
            and parameter_sep >= float(min_parameter_separation)
        )

        if accepted:
            config = apply_registry_family(config, best.family, best.fit.best_value, registry=reg)
            fitted.append(best.family)
            after = float(best.fit.best_cost)
        else:
            after = current_cost

        steps.append(PhysicalFitStep(
            stage=stage,
            cost_before=current_cost,
            accepted=accepted,
            selected_family=best.family if accepted else None,
            selected_value=best.fit.best_value if accepted else None,
            selected_units=best.units if accepted else None,
            cost_after=after,
            improvement_fraction=improvement,
            family_separation=family_sep,
            parameter_separation=parameter_sep,
            rankings=rankings,
        ))
        if not accepted:
            break
        current_cost = after

    return HierarchicalPhysicalFitResult(
        initial_cost=initial_cost,
        final_cost=current_cost,
        final_config=config,
        steps=tuple(steps),
        fitted_families=tuple(fitted),
    )
