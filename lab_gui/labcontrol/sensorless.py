"""Stateful sensorless command sweeps with fresh verification and rollback."""

from __future__ import annotations

import json
import math
import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from .state import ExperimentState, ExperimentStore, utc_now


class OptimisationStatus(str, Enum):
    PLANNED = "PLANNED"
    SWEEP_COMPLETE = "SWEEP_COMPLETE"
    AWAITING_VERIFICATION = "AWAITING_VERIFICATION"
    ACCEPTED = "ACCEPTED"
    ROLLED_BACK = "ROLLED_BACK"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    attribute: str
    units: str
    switch_group: str


PARAMETERS: dict[str, ParameterSpec] = {
    "tip_x": ParameterSpec("tip_x", "steering_x_mrad", "mrad", "steering"),
    "tip_y": ParameterSpec("tip_y", "steering_y_mrad", "mrad", "steering"),
    "astig_x": ParameterSpec("astig_x", "z22_cos_amp_waves", "waves command", "zernike_z40"),
    "astig_xy": ParameterSpec("astig_xy", "z22_sin_amp_waves", "waves command", "zernike_z40"),
    "coma_x": ParameterSpec("coma_x", "z31_cos_amp_waves", "waves command", "zernike_z40"),
    "coma_y": ParameterSpec("coma_y", "z31_sin_amp_waves", "waves command", "zernike_z40"),
    "defocus": ParameterSpec("defocus", "z20_amp_waves", "waves command", "zernike_z40"),
    "spherical": ParameterSpec("spherical", "z40_amp_waves", "waves command", "zernike_z40"),
    "retrieved_gain": ParameterSpec(
        "retrieved_gain", "retrieved_correction_gain", "dimensionless gain", "retrieved_correction"
    ),
}


@dataclass
class SensorlessTrial:
    trial_id: str
    role: str
    delta: float
    command: float
    score: float | None = None
    capture_id: str | None = None
    recorded_utc: str | None = None


@dataclass
class OptimisationRun:
    run_id: str
    parameter: str
    slm_name: str
    units: str
    baseline_state: dict[str, Any]
    baseline_value: float
    higher_is_better: bool
    minimum_fractional_improvement: float
    max_control_drift_fraction: float
    trials: list[SensorlessTrial]
    status: OptimisationStatus = OptimisationStatus.PLANNED
    recommended_trial_id: str | None = None
    verification_trial_id: str | None = None
    accepted_command: float | None = None
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "parameter": self.parameter,
            "slm_name": self.slm_name,
            "units": self.units,
            "baseline_state": self.baseline_state,
            "baseline_value": self.baseline_value,
            "higher_is_better": self.higher_is_better,
            "minimum_fractional_improvement": self.minimum_fractional_improvement,
            "max_control_drift_fraction": self.max_control_drift_fraction,
            "trials": [trial.__dict__ for trial in self.trials],
            "status": self.status.value,
            "recommended_trial_id": self.recommended_trial_id,
            "verification_trial_id": self.verification_trial_id,
            "accepted_command": self.accepted_command,
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OptimisationRun":
        if int(value.get("schema_version", 1)) != 1:
            raise ValueError("Unsupported sensorless run schema.")
        return cls(
            run_id=value["run_id"],
            parameter=value["parameter"],
            slm_name=value["slm_name"],
            units=value["units"],
            baseline_state=value["baseline_state"],
            baseline_value=float(value["baseline_value"]),
            higher_is_better=bool(value["higher_is_better"]),
            minimum_fractional_improvement=float(value["minimum_fractional_improvement"]),
            max_control_drift_fraction=float(value["max_control_drift_fraction"]),
            trials=[SensorlessTrial(**raw) for raw in value["trials"]],
            status=OptimisationStatus(value["status"]),
            recommended_trial_id=value.get("recommended_trial_id"),
            verification_trial_id=value.get("verification_trial_id"),
            accepted_command=value.get("accepted_command"),
            failure_reason=value.get("failure_reason"),
        )


class SensorlessOptimiser:
    """Optimises commanded coefficients; it does not infer physical aberrations."""

    def __init__(self, store: ExperimentStore):
        self.store = store

    def plan(
        self,
        parameter: str,
        deltas: Iterable[float],
        *,
        slm_name: str = "SLM2",
        seed: int = 42,
        higher_is_better: bool = False,
        minimum_fractional_improvement: float = 0.02,
        max_control_drift_fraction: float = 0.05,
    ) -> OptimisationRun:
        if parameter not in PARAMETERS:
            raise ValueError(f"Unknown parameter {parameter!r}; choose from {sorted(PARAMETERS)}")
        slm_name = slm_name.upper()
        if slm_name not in {"SLM1", "SLM2"}:
            raise ValueError("slm_name must be SLM1 or SLM2.")
        values = [float(value) for value in deltas]
        if len(values) < 2 or not all(math.isfinite(value) for value in values):
            raise ValueError("Supply at least two finite candidate deltas.")
        if len(set(values)) != len(values) or 0.0 in values:
            raise ValueError("Candidate deltas must be distinct and nonzero; controls are added automatically.")
        if min(values) >= 0 or max(values) <= 0:
            raise ValueError("A sensorless sweep must contain signed perturbations around the baseline.")
        if minimum_fractional_improvement < 0 or max_control_drift_fraction < 0:
            raise ValueError("Improvement and drift thresholds must be non-negative.")
        baseline = self.store.snapshot()
        spec = PARAMETERS[parameter]
        phase = getattr(baseline, slm_name.lower()).phase
        baseline_value = float(getattr(phase, spec.attribute))
        order = values[:]
        random.Random(seed).shuffle(order)
        run_id = f"optimise-{parameter}-{uuid.uuid4().hex[:10]}"
        trials = [
            SensorlessTrial(f"{run_id}-control-start", "CONTROL_START", 0.0, baseline_value)
        ]
        trials.extend(
            SensorlessTrial(
                f"{run_id}-candidate-{index:02d}",
                "CANDIDATE",
                delta,
                baseline_value + delta,
            )
            for index, delta in enumerate(order, 1)
        )
        trials.append(
            SensorlessTrial(f"{run_id}-control-end", "CONTROL_END", 0.0, baseline_value)
        )
        return OptimisationRun(
            run_id=run_id,
            parameter=parameter,
            slm_name=slm_name,
            units=spec.units,
            baseline_state=baseline.to_dict(),
            baseline_value=baseline_value,
            higher_is_better=higher_is_better,
            minimum_fractional_improvement=float(minimum_fractional_improvement),
            max_control_drift_fraction=float(max_control_drift_fraction),
            trials=trials,
        )

    def _set_command(self, run: OptimisationRun, command: float, *, reason: str) -> None:
        spec = PARAMETERS[run.parameter]

        def apply(state: ExperimentState) -> None:
            phase = getattr(state, run.slm_name.lower()).phase
            setattr(phase, spec.attribute, float(command))
            setattr(phase.switches, spec.switch_group, True)

        self.store.update(apply, source="sensorless", reason=reason)

    def apply_trial(self, run: OptimisationRun, trial_id: str) -> SensorlessTrial:
        if run.status in {OptimisationStatus.ACCEPTED, OptimisationStatus.ROLLED_BACK}:
            raise ValueError(f"Run is already terminal: {run.status.value}")
        trial = next((item for item in run.trials if item.trial_id == trial_id), None)
        if trial is None:
            raise KeyError(trial_id)
        self._set_command(
            run,
            trial.command,
            reason=f"Applied unaccepted sensorless trial {trial.trial_id}",
        )
        return trial

    def record_score(
        self, run: OptimisationRun, trial_id: str, score: float, *, capture_id: str
    ) -> None:
        if not math.isfinite(score):
            raise ValueError("Sensorless score must be finite.")
        trial = next((item for item in run.trials if item.trial_id == trial_id), None)
        if trial is None:
            raise KeyError(trial_id)
        if trial.score is not None:
            raise ValueError("Trial score is immutable; create a fresh trial instead of overwriting it.")
        if not capture_id:
            raise ValueError("A score must be linked to a formal capture id.")
        trial.score = float(score)
        trial.capture_id = capture_id
        trial.recorded_utc = utc_now()

    def recommend(self, run: OptimisationRun) -> SensorlessTrial:
        sweep = [trial for trial in run.trials if trial.role != "VERIFICATION"]
        if any(trial.score is None for trial in sweep):
            raise ValueError("Every candidate and both controls need a recorded score.")
        start = next(trial for trial in sweep if trial.role == "CONTROL_START")
        end = next(trial for trial in sweep if trial.role == "CONTROL_END")
        denominator = max(abs(float(start.score)), 1e-12)
        drift = abs(float(end.score) - float(start.score)) / denominator
        if drift > run.max_control_drift_fraction:
            run.status = OptimisationStatus.FAILED
            run.failure_reason = (
                f"Control drift {drift:.4g} exceeded {run.max_control_drift_fraction:.4g}."
            )
            self.rollback(run, reason=run.failure_reason)
            raise ValueError(run.failure_reason)
        candidates = [trial for trial in sweep if trial.role == "CANDIDATE"]
        recommended = (
            max(candidates, key=lambda item: float(item.score))
            if run.higher_is_better
            else min(candidates, key=lambda item: float(item.score))
        )
        run.status = OptimisationStatus.AWAITING_VERIFICATION
        run.recommended_trial_id = recommended.trial_id
        verification = SensorlessTrial(
            trial_id=f"{run.run_id}-verification-{uuid.uuid4().hex[:6]}",
            role="VERIFICATION",
            delta=recommended.delta,
            command=recommended.command,
        )
        run.trials.append(verification)
        run.verification_trial_id = verification.trial_id
        self._set_command(
            run,
            recommended.command,
            reason=f"Applied recommended command pending fresh verification for {run.run_id}",
        )
        return verification

    def verify_and_decide(
        self, run: OptimisationRun, score: float, *, capture_id: str
    ) -> bool:
        if run.status != OptimisationStatus.AWAITING_VERIFICATION or not run.verification_trial_id:
            raise ValueError("Recommend a candidate before verification.")
        recommended = next(
            trial for trial in run.trials if trial.trial_id == run.recommended_trial_id
        )
        if capture_id == recommended.capture_id:
            raise ValueError("Verification must use a fresh capture, not the candidate measurement.")
        self.record_score(run, run.verification_trial_id, score, capture_id=capture_id)
        controls = [
            float(trial.score)
            for trial in run.trials
            if trial.role in {"CONTROL_START", "CONTROL_END"}
        ]
        baseline = sum(controls) / len(controls)
        improvement = (
            (float(score) - baseline) / max(abs(baseline), 1e-12)
            if run.higher_is_better
            else (baseline - float(score)) / max(abs(baseline), 1e-12)
        )
        if improvement < run.minimum_fractional_improvement:
            self.rollback(
                run,
                reason=(
                    f"Fresh verification improvement {improvement:.4g} did not meet "
                    f"{run.minimum_fractional_improvement:.4g}."
                ),
            )
            return False
        self._set_command(run, recommended.command, reason=f"Accepted verified command {run.run_id}")

        def accept(state: ExperimentState) -> None:
            target = getattr(state, run.slm_name.lower())
            target.accepted_correction_id = run.run_id
            state.session.accepted_trial = run.verification_trial_id

        self.store.update(
            accept,
            source="sensorless",
            reason=f"Accepted correction command after fresh verification: {run.run_id}",
        )
        run.accepted_command = recommended.command
        run.status = OptimisationStatus.ACCEPTED
        return True

    def rollback(self, run: OptimisationRun, *, reason: str = "Operator rollback") -> None:
        baseline = ExperimentState.from_dict(run.baseline_state)
        spec = PARAMETERS[run.parameter]
        baseline_phase = getattr(baseline, run.slm_name.lower()).phase
        baseline_switch = bool(getattr(baseline_phase.switches, spec.switch_group))

        def restore(state: ExperimentState) -> None:
            phase = getattr(state, run.slm_name.lower()).phase
            setattr(phase, spec.attribute, run.baseline_value)
            setattr(phase.switches, spec.switch_group, baseline_switch)
            getattr(state, run.slm_name.lower()).accepted_correction_id = getattr(
                baseline, run.slm_name.lower()
            ).accepted_correction_id
            state.session.accepted_trial = baseline.session.accepted_trial

        self.store.update(
            restore,
            source="sensorless",
            reason=f"Rolled back unaccepted correction: {reason}",
        )
        run.status = OptimisationStatus.ROLLED_BACK
        run.failure_reason = reason

    @staticmethod
    def save(run: OptimisationRun, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
        temporary.replace(destination)
        return destination

    @staticmethod
    def load(path: str | Path) -> OptimisationRun:
        return OptimisationRun.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
