from __future__ import annotations

from labcontrol.calibration import (
    CalibrationKey,
    CalibrationRegistry,
    CalibrationStatus,
    PhysicalChange,
    PROCESSING_REQUIREMENTS,
)
from labcontrol.state import ExperimentStore, PhysicalAxiconState


def make_valid_registry() -> CalibrationRegistry:
    registry = CalibrationRegistry()
    for key in CalibrationKey:
        registry.mark(key, CalibrationStatus.VALID, calibration_id=f"cal-{key.value}")
    return registry


def test_axicon_reinsertion_invalidates_only_dependency_closure() -> None:
    registry = make_valid_registry()
    affected = set(
        registry.report_physical_change(PhysicalChange.AXICON_REMOVED_REINSERTED)
    )
    assert CalibrationKey.AXICON_PLACEMENT in affected
    assert CalibrationKey.Q0_PROPAGATION_REFERENCE in affected
    assert CalibrationKey.VORTEX_PROPAGATION_REFERENCE in affected
    assert CalibrationKey.LOW_ORDER_CORRECTION in affected
    assert CalibrationKey.RETRIEVED_RESIDUAL_MAP in affected
    assert CalibrationKey.VORTEX_VERIFICATION in affected
    assert registry.status(CalibrationKey.SLM_SERIAL_IDENTITY) == CalibrationStatus.VALID
    assert registry.status(CalibrationKey.CAMERA_PIXEL_SCALE) == CalibrationStatus.VALID
    assert registry.status(CalibrationKey.SLM_PHASE_PATH) == CalibrationStatus.VALID
    assert registry.status(CalibrationKey.CARRIER_CONVENTION) == CalibrationStatus.VALID


def test_readiness_is_derived_from_required_calibrations() -> None:
    registry = make_valid_registry()
    assert registry.readiness().ready
    registry.report_physical_change(PhysicalChange.AXICON_XY_CHANGED)
    report = registry.readiness()
    assert not report.ready
    assert CalibrationKey.AXICON_PLACEMENT.value in report.blockers
    assert "quick vortex recovery" in report.recommended_action


def test_physical_event_publishes_status_and_state_change() -> None:
    registry = make_valid_registry()
    store = ExperimentStore()
    affected = registry.apply_physical_change(
        store, PhysicalChange.AXICON_REMOVED_REINSERTED
    )
    state = store.snapshot()
    assert state.system.physical_axicon == PhysicalAxiconState.IN
    assert state.calibration_statuses[CalibrationKey.AXICON_PLACEMENT.value] == "STALE"
    assert len(affected) > 1
    assert store.recent_events()[-1].source == "physical_change"


def test_unknown_records_do_not_become_fake_stale_evidence() -> None:
    registry = CalibrationRegistry()
    registry.report_physical_change(PhysicalChange.CAMERA_REMOUNTED)
    assert registry.status(CalibrationKey.CAMERA_SENSOR_GEOMETRY) == CalibrationStatus.UNKNOWN
    assert not registry.readiness(PROCESSING_REQUIREMENTS).ready
