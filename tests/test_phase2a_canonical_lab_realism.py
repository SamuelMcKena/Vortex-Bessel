from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from vbb_study.digital_twin.phase2a_canonical import (
    PHASE2A_CASE_IDS,
    PHASE2A_POWER_DRIFT_LIMIT,
    PHASE2A_VARIANTS,
)
from vbb_study.digital_twin.phase2a_contracts import (
    ALLOWED_HARDWARE_PROVENANCE,
    PHASE2A_CANONICAL_SLM_MODEL,
    canonical_hardware_manifest,
    compute_unified_power_ledger,
    EnergyFactor,
    error_injection_registry_rows,
    slm_model_comparison_rows,
    validate_error_registry,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "validation" / "phase2a"


def _rows(name: str) -> list[dict[str, str]]:
    with (OUT / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_phase2a_hardware_manifest_is_fixed_and_fully_provenanced() -> None:
    manifest = canonical_hardware_manifest()
    assert manifest["mapping_mode"] == "fixed_physical_optics"
    assert manifest["slm_fill_factor_model"] == PHASE2A_CANONICAL_SLM_MODEL == "throughput_only"
    parameters = {row["parameter"]: row for row in manifest["parameters"]}
    required = {
        "wavelength_m", "beam_radius_on_slm_m", "slm_model", "slm_resolution_x_px",
        "slm_resolution_y_px", "slm_pixel_pitch_m", "slm_active_width_m", "slm_active_height_m",
        "slm_phase_bits", "carrier_frequency_cpm", "fourf_focal_length_m", "fourier_iris_radius_m",
        "objective_NA", "objective_focal_length_m", "relay_magnification_to_sample",
        "axicon_base_angle_deg", "axicon_refractive_index",
    }
    assert required <= set(parameters)
    assert all(row["provenance"] in ALLOWED_HARDWARE_PROVENANCE for row in manifest["parameters"])
    assert parameters["wavelength_m"]["value"] == pytest.approx(1029e-9)
    assert parameters["fourier_iris_radius_m"]["value"] == pytest.approx(0.77175e-3)
    assert parameters["slm_phase_stroke_rad"]["value"] is None
    assert manifest["absolute_sample_plane_claim_ready"] is False


def test_phase2a_slm_models_are_explicit_and_power_consistent() -> None:
    by_model = {row["slm_model"]: row for row in slm_model_comparison_rows()}
    assert set(by_model) == {
        "throughput_only", "resolved_pixel_aperture", "coherent_unmodulated_deadspace"
    }
    assert by_model["throughput_only"]["output_power_fraction"] == pytest.approx(0.93, abs=1e-12)
    assert by_model["resolved_pixel_aperture"]["output_power_fraction"] == pytest.approx(0.93, abs=0.015)
    coherent = by_model["coherent_unmodulated_deadspace"]
    assert coherent["output_power_fraction"] == pytest.approx(1.0, abs=1e-12)
    assert coherent["unmodulated_power_fraction"] > 0.0
    assert coherent["zero_order_fraction_within_unmodulated_component"] > 0.0
    assert all(row["ledger_relative_error"] < 1e-12 for row in by_model.values())
    assert not any(row["fill_factor_loss_counted_in_external_energy_ledger"] for row in by_model.values())


def test_phase2a_bookkeeping_closure_synthetic_contract() -> None:
    factors = [
        EnergyFactor("SLM1", "slm1", 0.93, "manufacturer", "physical"),
        EnergyFactor("first-order filter", "selected", 0.71, "simulated", "physical"),
        EnergyFactor("sample", "surface", 0.96, "assumed", "physical"),
    ]
    rows, closure = compute_unified_power_ledger("synthetic", "synthetic", 10e-6, factors)
    assert len(rows) == 3
    assert closure["closure_relative_residual"] <= 1e-10
    assert closure["closure_pass"] is True


def test_phase2a_error_registry_and_plane_operators() -> None:
    rows = error_injection_registry_rows()
    validate_error_registry(rows)
    by_id = {row["error_id"]: row for row in rows}
    required = {
        "input_beam_decentre", "input_tilt", "hologram_offset", "slm_phase_error",
        "slm_quantisation", "iris_offset_radius", "axicon_decentre", "axicon_tilt",
        "objective_pupil_clipping", "low_order_aberration", "sample_interface_tilt",
        "camera_noise", "post_processing_display_shift",
    }
    assert required == set(by_id)
    for error_id in ("camera_noise", "post_processing_display_shift"):
        assert by_id[error_id]["physical_or_diagnostic"] == "diagnostic"
        assert by_id[error_id]["acts_before_or_after_propagation"] == "after"
        assert by_id[error_id]["acts_on_complex_field"] is False
        assert by_id[error_id]["diagnostic_only"] is True
        assert by_id[error_id]["affected_field"] == "measured_or_display_intensity"
    assert by_id["input_beam_decentre"]["injection_plane"] == "before SLM1"
    assert by_id["input_beam_decentre"]["physical_operator"] == by_id["input_beam_decentre"]["mathematical_operator"]
    assert by_id["iris_offset_radius"]["injection_plane"] == "4F Fourier plane"


def test_phase2a_outputs_contain_exact_controlled_family() -> None:
    rows = _rows("canonical_case_summary.csv")
    assert len(rows) == len(PHASE2A_CASE_IDS) * len(PHASE2A_VARIANTS) == 25
    assert Counter(row["case_id"] for row in rows) == Counter({case: 5 for case in PHASE2A_CASE_IDS})
    assert Counter(row["route_variant"] for row in rows) == Counter({variant: 5 for variant in PHASE2A_VARIANTS})
    assert all(row["mapping_mode"] == "fixed_physical_optics" for row in rows)
    assert all(row["slm_fill_factor_model"] == "throughput_only" for row in rows)


def test_phase2a_all_quantitative_cases_pass_power_gate() -> None:
    rows = _rows("canonical_case_summary.csv")
    drifts = [float(row["propagation_power_drift_fraction"]) for row in rows]
    assert max(drifts) <= PHASE2A_POWER_DRIFT_LIMIT
    assert all(row["quantitative_metrics_valid"].lower() == "true" for row in rows)
    assert all(row["energy_ledger_closure_pass"].lower() == "true" for row in rows)
    assert max(float(row["energy_ledger_closure_relative_residual"]) for row in rows) <= 1e-10
    assert max(float(row["fluence_energy_residual_fraction"]) for row in rows) <= 1e-10


def test_phase2a_first_order_is_simulated_and_not_double_counted() -> None:
    rows = _rows("canonical_power_ledgers.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["case_id"], row["route_variant"]), []).append(row)
    assert len(grouped) == 25
    for case_rows in grouped.values():
        selected = [row for row in case_rows if row["factor_id"] == "simulated_selected_first_order"]
        assert len(selected) == 1
        assert selected[0]["source_of_efficiency"] == "simulated"
        assert not any("configured_first_order" in row["factor_id"] for row in case_rows)
        assert sum("panel_transfer" in row["factor_id"] for row in case_rows) <= 2


def test_phase2a_h1_uses_vector_route_and_expected_degradation_ladder() -> None:
    rows = {row["route_variant"]: row for row in _rows("canonical_case_summary.csv") if row["case_id"] == "H1"}
    assert all(row["route_kind"] == "parallel_vector" for row in rows.values())
    assert rows["realistic_fixed_bench_route"]["morphology_class"] == "visual_hexagonal_field"
    assert float(rows["realistic_fixed_bench_route"]["local_vector_purity"]) > 0.95
    assert rows["deliberately_degraded_route"]["morphology_class"] != "visual_hexagonal_field"
    assert float(rows["deliberately_degraded_route"]["local_vector_purity"]) < 0.5


def test_phase2a_outcome_is_calibration_limited_not_forced_a() -> None:
    report = json.loads((OUT / "phase2a_outcome_report.json").read_text(encoding="utf-8"))
    assert report["outcome"] == "PHASE2A-B"
    assert report["canonical_case_count"] == 25
    assert report["quantitative_valid_case_count"] == 25
    assert report["energy_ledger_closure_pass_count"] == 25
    assert report["configured_first_order_efficiency_reapplied"] is False
    assert report["nathan_outputs_changed"] is False
    assert report["phase1_contracts_reopened"] is False
    assert set(report["blocked_or_calibration_limited_claims"]) == {"P2A-C9", "P2A-C10", "P2A-C11"}


def test_phase2a_claim_classes_and_docs_are_complete() -> None:
    claims = _rows("phase2a_claim_registry.csv")
    assert {row["claim_class"] for row in claims} >= {
        "optical_prediction", "fixed_bench_prediction", "energy_accounting_prediction",
        "fluence_prediction", "diagnostic_only", "calibration_required",
    }
    assert (ROOT / "docs" / "90_phase2a_canonical_lab_realism.md").is_file()
    assert (ROOT / "docs" / "90_phase2a_error_injection_registry.md").is_file()
