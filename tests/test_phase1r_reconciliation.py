from __future__ import annotations

from dataclasses import replace
import csv
import json
from collections import Counter
from pathlib import Path

import bessel_twin_core as bt
import pytest
from vbb_study import vbb_regime
from vbb_study.vbb_train_viz import method_comparison_table
from vbb_study.viz_fields import phase_winding


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "validation" / "phase1_reconciliation"


def _csv_rows(name: str) -> list[dict[str, str]]:
    with (OUT / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_public_phase_winding_measures_requested_physical_charge() -> None:
    base = replace(bt.default_config("fast"), generation_method="physical")
    cfg = vbb_regime.config_for_regime(base, "general")
    result = bt.run_case(cfg, preset="fast", path="ideal", case_id="phase1r_winding")
    surface = result["surface_field"]
    design = result["design"]
    measured = phase_winding(surface.Ex, surface.grid, design.vortex_main_ring_radius_m)
    assert abs(measured - design.ell) < 0.1


def test_method_comparison_exports_topology_and_power_validity() -> None:
    table = method_comparison_table(
        bt.default_config("fast"),
        regimes=("general",),
        methods=("physical",),
    )
    row = table.iloc[0]
    assert row["slm2_conjugate_mode"] == "preserve_vortex"
    assert bool(row["winding_pass"])
    assert abs(float(row["measured_winding"]) - float(row["requested_vortex_charge"])) < 0.1
    assert "quantitative_metrics_valid" in table.columns


def test_all_phase1_artifacts_have_exactly_one_disposition() -> None:
    rows = _csv_rows("phase1r_artifact_disposition.csv")
    assert len(rows) == 85
    assert len({row["artifact_record_id"] for row in rows}) == 85
    assert Counter(row["reconciliation_action"] for row in rows) == {
        "regenerate_unchanged_sampling": 21,
        "rerun_with_convergence_repair": 57,
        "reexport_metadata_only": 3,
        "retain_blocked_historical_diagnostic": 4,
    }


def test_convergence_manifest_accounts_for_every_high_drift_source_row() -> None:
    manifest = json.loads((OUT / "phase1r_convergence_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_file_count"] == 57
    assert manifest["source_row_count_above_limit"] == 415
    assert manifest["unique_physical_case_count"] == 119
    assert manifest["recovered_unique_case_count"] == 3
    assert manifest["blocked_unique_case_count"] == 116
    assert sum(case["source_row_count"] for case in manifest["unique_cases"]) == 415
    assert manifest["crop_only_excluded_as_power_drift_cause"] is True


def test_recovered_cases_pass_predeclared_power_and_metric_gates() -> None:
    selected = json.loads(
        (OUT / "phase1r_selected_convergence_results.json").read_text(encoding="utf-8")
    )
    recovered = [case for case in selected["cases"] if case["convergence_pass"]]
    assert {case["case_id"] for case in recovered} == {
        "general_holographic_ideal",
        "general_physical_ideal",
        "near_threshold_D8_L150_ideal",
    }
    tolerances = selected["predeclared_metric_tolerances"]
    for case in recovered:
        assert len(case["final_adequate_pair"]) == 2
        for metric, delta in case["metric_relative_deltas"].items():
            assert delta <= tolerances[metric]


def test_regenerated_physical_rows_measure_requested_winding() -> None:
    paths = (
        ROOT / "outputs/csv/stage_c/physical_axicon_design_summary.csv",
        ROOT / "outputs/csv/stage_c/holographic_physical_method_comparison.csv",
        ROOT / "outputs/csv/stage_d/through_sample_summary.csv",
        ROOT / "outputs/csv/stage_e/full_source_to_sample_journey_summary.csv",
    )
    count = 0
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        physical = [
            row
            for row in rows
            if row.get("method") == "physical"
            or row.get("generation_method") == "physical_axicon"
        ]
        for row in physical:
            count += 1
            assert row["slm2_conjugate_mode"] == "preserve_vortex"
            assert row["vortex_removal_acknowledged"].lower() == "false"
            assert row["winding_pass"].lower() == "true"
            assert abs(float(row["measured_winding"]) - float(row["requested_vortex_charge"])) < 0.1
    assert count >= 18


def test_fourier_outputs_are_regenerated_with_wavelength() -> None:
    path = ROOT / "outputs/csv/stage_c/objective_pupil_geometry_summary.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = {row["regime"]: row for row in csv.DictReader(handle)}
    assert float(rows["general"]["fourier_ring_radius_mm"]) == pytest.approx(0.008477154227779409)
    assert float(rows["limits"]["fourier_ring_radius_mm"]) == pytest.approx(0.038147194025007346)


def test_mapping_reexport_is_inverse_design_scoped() -> None:
    manifest = json.loads((OUT / "phase1r_regeneration_manifest.json").read_text(encoding="utf-8"))
    assert manifest["mapping_metadata_reexport_file_count"] == 34
    for relative in manifest["mapping_metadata_reexport_files"]:
        with (ROOT / relative).open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            assert row["mapping_mode"] == "target_matched_inverse_design"
            assert row["objective_map_source"] == "compute_design_from_targets:w0_sample/beam_radius_on_slm"
            assert row["mapping_claim_scope"] == "inverse_design_feasibility"
            assert row["hardware_target_achieved"] == "false"
            if row.get("magnification_to_sample"):
                assert row["objective_map_demag"] == row["magnification_to_sample"]


def test_final_claim_registry_uses_only_closed_status_vocabulary() -> None:
    rows = _csv_rows("phase1r_final_claim_registry.csv")
    assert len(rows) == 14
    allowed = {
        "validated",
        "validated_with_scope",
        "diagnostic_only",
        "blocked_unconverged",
        "superseded",
        "historical_only",
    }
    assert {row["post_phase1r_status"] for row in rows} <= allowed
    blocked = [row for row in rows if row["post_phase1r_status"] == "blocked_unconverged"]
    assert [row["claim_id"] for row in blocked] == ["P1R-C1"]


def test_affected_lab_notebooks_and_controlled_copies_are_cleanly_executed() -> None:
    names = (
        "02_physical_axicon_route",
        "03_holographic_vs_physical_axicon",
        "04_objective_pupil_and_first_order_filtering",
        "05_through_sample_interface",
        "06_full_source_to_sample_journey",
    )
    for name in names:
        source = ROOT / "notebooks" / "lab_realism" / f"{name}.ipynb"
        executed = ROOT / "outputs" / "notebook_triage" / f"{name}_executed.ipynb"
        assert executed.read_bytes() == source.read_bytes()
        notebook = json.loads(source.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert code_cells
        assert all(cell["execution_count"] is not None for cell in code_cells)
        errors = [
            output
            for cell in code_cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        assert errors == []
