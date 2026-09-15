from __future__ import annotations

from pathlib import Path

import pytest

from labcontrol.calibration import CalibrationKey, CalibrationRegistry, CalibrationStatus
from labcontrol.recipes import (
    RecipeEngine,
    RecipePrerequisiteError,
    RecipeRunStatus,
    builtin_recipes,
)
from labcontrol.sensorless import OptimisationStatus, SensorlessOptimiser
from labcontrol.state import ExperimentStore


def valid_for(*keys: CalibrationKey) -> CalibrationRegistry:
    registry = CalibrationRegistry()
    for key in keys:
        registry.mark(key, CalibrationStatus.VALID)
    return registry


def test_recipe_prerequisites_progression_and_resume(tmp_path: Path) -> None:
    store = ExperimentStore()
    registry = CalibrationRegistry()
    engine = RecipeEngine(store, registry)
    with pytest.raises(RecipePrerequisiteError, match="slm_phase_path"):
        engine.start("recover_q20")
    for key in (
        CalibrationKey.SLM_PHASE_PATH,
        CalibrationKey.CAMERA_PIXEL_SCALE,
        CalibrationKey.CAMERA_SENSOR_GEOMETRY,
    ):
        registry.mark(key, CalibrationStatus.VALID)
    run = engine.start("recover_q20", {"z_mm": [30.0, 38.0, 46.0]})
    assert engine.current_step(run).step_id == "configure"
    assert engine.complete_step(run, {"confirmed": True}).step_id == "q0_walk"
    assert run.status == RecipeRunStatus.WAITING_FOR_OPERATOR
    path = engine.save(run, tmp_path / "run.json")
    resumed = engine.load(path)
    assert resumed.run_id == run.run_id
    assert engine.current_step(resumed).step_id == "q0_walk"
    while engine.current_step(resumed):
        engine.complete_step(resumed, {"test": True})
    assert resumed.status == RecipeRunStatus.COMPLETE


def test_q20_is_one_recipe_in_a_general_catalogue() -> None:
    names = set(builtin_recipes())
    assert {"recover_q20", "z_stack", "measure_beam_walk", "vortex_charge_scan"} <= names
    assert len(names) >= 7


def _score_sweep(optimiser: SensorlessOptimiser, run, candidate_score: float = 0.6) -> None:
    for trial in list(run.trials):
        optimiser.apply_trial(run, trial.trial_id)
        score = 1.0 if trial.role.startswith("CONTROL") else candidate_score + abs(trial.delta)
        optimiser.record_score(run, trial.trial_id, score, capture_id=f"capture-{trial.trial_id}")


def test_sensorless_requires_fresh_verification_before_acceptance() -> None:
    store = ExperimentStore()
    optimiser = SensorlessOptimiser(store)
    run = optimiser.plan("astig_x", [-0.2, 0.2], seed=1)
    _score_sweep(optimiser, run)
    verification = optimiser.recommend(run)
    recommended = next(t for t in run.trials if t.trial_id == run.recommended_trial_id)
    with pytest.raises(ValueError, match="fresh capture"):
        optimiser.verify_and_decide(run, 0.7, capture_id=recommended.capture_id)
    accepted = optimiser.verify_and_decide(run, 0.7, capture_id="fresh-verification")
    assert accepted
    assert run.status == OptimisationStatus.ACCEPTED
    state = store.snapshot()
    assert state.slm2.accepted_correction_id == run.run_id
    assert state.session.accepted_trial == verification.trial_id


def test_sensorless_failed_verification_rolls_back_baseline() -> None:
    store = ExperimentStore()
    optimiser = SensorlessOptimiser(store)
    baseline = store.snapshot().slm2.phase.z22_cos_amp_waves
    run = optimiser.plan("astig_x", [-0.2, 0.2], seed=1)
    _score_sweep(optimiser, run)
    optimiser.recommend(run)
    assert not optimiser.verify_and_decide(run, 1.1, capture_id="fresh-but-worse")
    assert run.status == OptimisationStatus.ROLLED_BACK
    assert store.snapshot().slm2.phase.z22_cos_amp_waves == baseline
    assert store.snapshot().slm2.accepted_correction_id is None


def test_sensorless_control_drift_forces_rollback() -> None:
    store = ExperimentStore()
    optimiser = SensorlessOptimiser(store)
    run = optimiser.plan("coma_y", [-0.1, 0.1], max_control_drift_fraction=0.01)
    for trial in run.trials:
        score = 1.2 if trial.role == "CONTROL_END" else 1.0
        optimiser.record_score(run, trial.trial_id, score, capture_id=trial.trial_id)
    with pytest.raises(ValueError, match="Control drift"):
        optimiser.recommend(run)
    assert run.status == OptimisationStatus.ROLLED_BACK
