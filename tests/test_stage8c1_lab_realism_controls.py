"""Stage 8C.1 lab-realism control tests."""

import pytest

from vbb_study.digital_twin.lab_realism_controls import (
    ALLOWED_STATUS_LEVELS,
    REQUIRED_STAGE_NAMES,
    LabRealismReport,
    build_energy_ledger_from_controls,
    build_lab_realism_report,
    default_lab_controls,
    validate_future_physics_disabled,
)


def _report() -> LabRealismReport:
    controls = default_lab_controls()
    ledger = build_energy_ledger_from_controls(controls)
    return build_lab_realism_report(controls, energy_ledger=ledger)


def test_all_required_beam_path_stages_exist():
    report = _report()
    names = [stage.stage_name for stage in report.stages]
    assert names == REQUIRED_STAGE_NAMES


def test_each_stage_has_editable_inputs():
    report = _report()
    for stage in report.stages:
        assert stage.editable_inputs, f"{stage.stage_name} has no editable inputs"


def test_each_stage_has_outputs_or_missing_future_marker():
    report = _report()
    for stage in report.stages:
        assert stage.computed_outputs or stage.missing_metrics, stage.stage_name


def test_each_stage_has_handoff_metadata():
    report = _report()
    for stage in report.stages:
        assert stage.handoff_to_next_stage


def test_future_material_response_toggles_raise_when_enabled():
    controls = default_lab_controls()
    controls["enable_material_response"] = True
    with pytest.raises(NotImplementedError):
        validate_future_physics_disabled(controls)
    with pytest.raises(NotImplementedError):
        build_lab_realism_report(controls)


def test_lab_realism_report_includes_every_required_stage():
    report = _report()
    rows = report.to_rows()
    assert len(rows) == len(REQUIRED_STAGE_NAMES)
    assert {row["stage_name"] for row in rows} == set(REQUIRED_STAGE_NAMES)


def test_status_levels_are_limited_to_allowed_values():
    report = _report()
    for stage in report.stages:
        assert stage.status_level in ALLOWED_STATUS_LEVELS


def test_report_dataframe_or_rows_contains_required_columns():
    report = _report()
    rows = report.to_rows()
    required = {
        "stage_name",
        "enabled",
        "key_inputs",
        "key_outputs",
        "model_status",
        "status_level",
        "warnings",
        "missing_metrics",
        "handoff_to_next_stage",
    }
    assert required <= set(rows[0])

