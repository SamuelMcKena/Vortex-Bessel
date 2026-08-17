"""Generate governed PHASE 2D calibration-bridge artifacts without changing upstream outputs."""

from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from vbb_study.calibration.io import dump_calibration_bundle
from vbb_study.calibration.schema import CalibrationBundle, canonical_calibration_template
from vbb_study.calibration.validation import (
    calibration_dependency_rows,
    calibration_readiness_for_claim,
    validate_calibration_bundle,
)
from vbb_study.digital_twin.phase2d_canonical_pipeline import (
    ROOT,
    CanonicalOpticalRequest,
    result_to_dict,
    run_canonical_optical_prediction,
    sha256_file,
    upstream_hashes,
)
from vbb_study.digital_twin.phase2d_figures import (
    plot_calibration_readiness,
    plot_calibration_state_comparison,
    plot_solver_policy_matrix,
    plot_uncertainty_summary,
)
from vbb_study.solver_policy import BEAM_CASES, CLAIM_TYPES, solver_policy_rows


VALIDATION_ROOT = ROOT / "outputs" / "validation" / "phase2d"
FIGURE_ROOT = ROOT / "outputs" / "figures" / "phase2d"
TEMPLATE_ROOT = ROOT / "calibration" / "templates"


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str] | None = None) -> Path:
    materialised = list(rows)
    if fieldnames is None:
        fieldnames = list(materialised[0]) if materialised else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialised)
    return path


def _measurement(value: float, uncertainty: float, unit: str) -> dict[str, Any]:
    return {
        "value": value,
        "uncertainty": uncertainty,
        "source": "synthetic_measurement",
        "unit": unit,
    }


def _synthetic_bundles() -> tuple[CalibrationBundle, CalibrationBundle]:
    partial = canonical_calibration_template()
    partial.update({
        "calibration_id": "phase2d_partial_synthetic",
        "created_utc": "2026-08-04T00:00:00Z",
        "operator": "software_validation",
        "data_classification": "synthetic_not_experimental",
    })
    partial["laser"]["wavelength_m"] = _measurement(1.029e-6, 0.2e-9, "m")
    partial["laser"]["beam_radius_on_slm_m"] = _measurement(2.0e-3, 20e-6, "m")
    partial["fourier_filter"]["focal_length_m"] = _measurement(0.300, 0.0005, "m")
    partial["objective"]["numerical_aperture"] = _measurement(0.45, 0.005, "1")
    partial["objective"]["focal_length_m"] = _measurement(0.004, 20e-6, "m")
    partial["objective"]["effective_pupil_radius_m"] = _measurement(1.8e-3, 10e-6, "m")
    partial["objective"]["pupil_fill_fraction"] = _measurement(0.95, 0.01, "1")
    partial["relay"]["magnification"] = _measurement(0.0080710635, 0.0001, "1")
    partial["axicon"]["base_angle_deg"] = _measurement(2.0, 0.01, "deg")
    partial["axicon"]["refractive_index"] = _measurement(1.458, 0.001, "1")

    full = deepcopy(partial)
    full["calibration_id"] = "phase2d_full_synthetic"
    full["laser"]["pulse_energy_J"] = _measurement(10.0e-6, 0.10e-6, "J")
    full["slm"].update({
        "phase_lut_path": "synthetic://software_validation/slm_lut.csv",
        "phase_stroke_rad": 6.283185307179586,
        "phase_stroke_uncertainty_rad": 0.02,
        "panel_orientation_verified": True,
        "calibration_date": "2026-08-04",
    })
    full["fourier_filter"]["iris_radius_m"] = _measurement(0.77175e-3, 5e-6, "m")
    full["fourier_filter"]["plus_one_position_m"] = _measurement(1.929375e-3, 5e-6, "m")
    full["axicon"]["clear_aperture_m"] = _measurement(12.5e-3, 0.1e-3, "m")
    full["camera"]["pixel_pitch_m"] = _measurement(6.5e-6, 0.01e-6, "m")
    full["camera"]["magnification"] = _measurement(31.584, 0.3, "1")
    full["camera"]["object_plane_scale_m_per_pixel"] = _measurement(0.2058e-6, 0.002e-6, "m/pixel")
    full["camera"]["rotation_deg"] = _measurement(0.0, 0.05, "deg")
    full["camera"]["centre_pixel"] = [960.0, 540.0]
    for name, value, uncertainty in (
        ("slm1", 0.75, 0.01),
        ("slm2", 0.75, 0.01),
        ("four_f", 0.90, 0.01),
        ("objective", 0.90, 0.01),
        ("interface", 0.96, 0.005),
    ):
        full["transmissions"][name] = _measurement(value, uncertainty, "1")
    full["material"]["refractive_index"] = _measurement(2.44, 0.01, "1")
    full["material"]["coating_state"] = "uncoated"
    full["material"]["surface_orientation_verified"] = True
    return CalibrationBundle(partial), CalibrationBundle(full)


def _write_templates() -> list[Path]:
    paths: list[Path] = []
    paths.append(dump_calibration_bundle(
        CalibrationBundle(canonical_calibration_template()),
        TEMPLATE_ROOT / "canonical_lab_calibration_template.json",
    ))
    templates: dict[str, tuple[list[str], list[dict[str, Any]]]] = {
        "beam_radius_measurement_template.csv": (
            ["measurement_id", "timestamp_utc", "wx_m", "wy_m", "u_wx_m", "u_wy_m", "covariance_xy_m2", "residual_rms", "saturation_fraction", "fit_crop_px", "source_file_sha256", "operator", "data_classification"],
            [{"measurement_id": "", "data_classification": "laboratory_measurement"}],
        ),
        "energy_transmission_template.csv": (
            ["measurement_id", "timestamp_utc", "stage", "input_energy_J", "u_input_energy_J", "output_energy_J", "u_output_energy_J", "transmission", "u_transmission", "repeats", "source_file_sha256", "operator", "data_classification"],
            [{"measurement_id": "", "data_classification": "laboratory_measurement"}],
        ),
        "slm_phase_lut_template.csv": (
            ["panel_id", "calibration_date", "wavelength_m", "drive_code", "phase_rad", "u_phase_rad", "interferogram_sha256", "operator", "data_classification"],
            [{"panel_id": "SLM-H", "data_classification": "laboratory_measurement"}, {"panel_id": "SLM-V", "data_classification": "laboratory_measurement"}],
        ),
        "camera_scale_template.csv": (
            ["measurement_id", "timestamp_utc", "known_spacing_m", "u_known_spacing_m", "delta_pixels", "u_delta_pixels", "object_plane_scale_m_per_pixel", "u_scale_m_per_pixel", "rotation_deg", "u_rotation_deg", "source_file_sha256", "operator", "data_classification"],
            [{"measurement_id": "", "data_classification": "laboratory_measurement"}],
        ),
        "objective_relay_template.csv": (
            ["measurement_id", "timestamp_utc", "parameter", "value", "uncertainty", "unit", "method", "source_file_sha256", "operator", "data_classification"],
            [{"parameter": name, "unit": unit, "data_classification": "laboratory_measurement"} for name, unit in (("objective_numerical_aperture", "1"), ("objective_focal_length", "m"), ("effective_pupil_radius", "m"), ("pupil_fill_fraction", "1"), ("relay_magnification", "1"))],
        ),
        "material_interface_template.csv": (
            ["measurement_id", "timestamp_utc", "parameter", "value", "uncertainty", "unit", "coating_state", "surface_orientation_verified", "wavelength_m", "source_reference", "operator", "data_classification"],
            [{"parameter": "refractive_index", "unit": "1", "data_classification": "laboratory_measurement"}],
        ),
    }
    for name, (fieldnames, rows) in templates.items():
        paths.append(_write_csv(TEMPLATE_ROOT / name, rows, fieldnames))
    return paths


def _claims_for_case(case: str) -> tuple[str, ...]:
    common = ("global_transverse_morphology", "absolute_dimensions", "absolute_fluence")
    if case in {"G0", "B0"}:
        return common + ("longitudinal_field",)
    if case in {"V1", "V3"}:
        return common + ("ring_radius", "peak_location", "longitudinal_field")
    return common + (
        "feature_radius", "edge_sharpness", "ridge_width", "transition_width",
        "longitudinal_field", "polarisation_component",
    )


def _summary_row(scenario: str, result: Any) -> dict[str, Any]:
    status = result.calibration_status
    available_uncertainty = sum(
        row["uncertainty_status"] == "available_from_supplied_uncertainty"
        for row in result.uncertainty_summary.values()
    )
    readiness = status["claim_readiness"]
    return {
        "calibration_scenario": scenario,
        "data_classification": status["data_classification"],
        "beam_case": result.request.beam_case,
        "requested_mode": result.request.fidelity_mode,
        "objective_model": result.focal_field.solver,
        "interface_model": "not_requested" if result.transmitted_field is None else result.transmitted_field.solver,
        "calibration_readiness": "complete_for_requested_claims" if not status["blocked_claims"] else "blocked",
        "dimensional_readiness": "ready" if readiness["absolute_dimensions"]["ready"] else "blocked",
        "energy_status": "synthetic_input_complete" if readiness["absolute_fluence"]["ready"] else "calibration_required",
        "fluence_readiness": "ready" if readiness["absolute_fluence"]["ready"] else "blocked",
        "absolute_dimensions_unlocked": status["absolute_dimensions_unlocked"],
        "absolute_fluence_unlocked": status["absolute_fluence_unlocked"],
        "experimentally_validated": status["experimentally_validated"],
        "uncertainty_metrics_available": available_uncertainty,
        "blocked_claims": ";".join(status["blocked_claims"]),
        "enabled_claims": ";".join(claim for claim, row in readiness.items() if row["ready"]),
        "strict_hexagon": result.metrics.get("strict_hexagon"),
        "measured_winding": result.metrics.get("measured_winding"),
        "selected_first_order_factor_count": result.provenance["selected_first_order_factor_count"],
        "first_order_factor_reapplied": result.provenance["configured_first_order_factor_reapplied"],
    }


def _hash_outputs(paths: Iterable[Path]) -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
        for path in paths
        if path.is_file()
    }


def generate_phase2d_outputs() -> dict[str, Any]:
    upstream_before = upstream_hashes()
    template_paths = _write_templates()
    partial, full = _synthetic_bundles()
    synthetic_root = VALIDATION_ROOT / "synthetic_bundles"
    partial_path = dump_calibration_bundle(partial, synthetic_root / "partial_synthetic_validation.json")
    full_path = dump_calibration_bundle(full, synthetic_root / "full_synthetic_validation.json")

    scenarios = (
        ("no_calibration_bundle", None, 0),
        ("partial_synthetic", str(partial_path), 256),
        ("full_synthetic", str(full_path), 512),
    )
    results: list[tuple[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for scenario, bundle_path, samples in scenarios:
        for case in BEAM_CASES:
            result = run_canonical_optical_prediction(CanonicalOpticalRequest(
                beam_case=case,
                claim_types=_claims_for_case(case),
                fidelity_mode="automatic_by_claim",
                calibration_bundle_path=bundle_path,
                include_interface=True,
                material_propagation_distance_m=10.0e-6,
                uncertainty_samples=samples,
            ))
            results.append((scenario, result))
            summary_rows.append(_summary_row(scenario, result))

    policy = solver_policy_rows()
    blank_bundle = CalibrationBundle(canonical_calibration_template())
    dependencies = calibration_dependency_rows(blank_bundle)
    validation = validate_calibration_bundle(blank_bundle)
    readiness_payload = {
        "schema_version": blank_bundle.schema_version,
        "current_bundle": "calibration/templates/canonical_lab_calibration_template.json",
        "schema_valid": validation.valid_schema,
        "missing_values": list(validation.missing_values),
        "warnings": list(validation.warnings),
        "claim_readiness": {
            claim: asdict(calibration_readiness_for_claim(blank_bundle, claim))
            for claim in CLAIM_TYPES
        },
        "absolute_dimensions_unlocked": False,
        "absolute_fluence_unlocked": False,
        "experimentally_validated_prediction_present": False,
    }

    uncertainty_rows: list[dict[str, Any]] = []
    for scenario, result in results:
        if scenario == "no_calibration_bundle":
            continue
        for metric, row in result.uncertainty_summary.items():
            uncertainty_rows.append({
                "calibration_scenario": scenario,
                "data_classification": result.calibration_status["data_classification"],
                "beam_case": result.request.beam_case,
                "metric": metric,
                **{key: row.get(key) for key in (
                    "uncertainty_status", "nominal", "standard_uncertainty", "lower_95", "upper_95",
                    "samples", "failed_samples",
                )},
                "missing_calibration": ";".join(row.get("missing_calibration", [])),
                "parameter_contributions": json.dumps(row.get("parameter_contributions", {}), sort_keys=True),
            })

    no_bundle_results = {result.request.beam_case: result for scenario, result in results if scenario == "no_calibration_bundle"}
    maturity_rows: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    for case, result in no_bundle_results.items():
        decision_by_claim = {decision.claim_type: decision for decision in result.solver_decisions}
        for claim in result.request.claim_types:
            ready = result.calibration_status["claim_readiness"][claim]
            blockers = list(ready["missing_measurements"]) + list(ready["non_calibrated_measurements"])
            maturity_rows.append({
                "claim_id": f"P2D-{case}-{claim.upper()}",
                "beam_case": case,
                "claim": claim,
                "solver": decision_by_claim[claim].selected_objective_solver,
                "current_maturity": result.calibration_status["claim_maturity"][claim],
                "required_next_evidence": "laboratory calibration bundle" if blockers else "measured output comparison",
                "calibration_blockers": ";".join(blockers),
                "experimental_validation_required": True,
                "evidence_path": "outputs/validation/phase2c/phase2c_objective_benchmark.csv",
                "notes": "Synthetic calibration can validate software flow but cannot raise experimental maturity.",
            })
            decision = decision_by_claim[claim]
            claim_rows.append({
                "claim_id": f"P2D-{case}-{claim.upper()}",
                "beam_case": case,
                "claim_type": claim,
                "objective_solver": decision.selected_objective_solver,
                "interface_solver": decision.selected_interface_solver,
                "vector_required": decision.vector_required,
                "calibration_required": decision.calibration_required,
                "current_status": "enabled_nominal" if not blockers else "calibration_blocked",
                "current_maturity": result.calibration_status["claim_maturity"][claim],
                "evidence_path": "outputs/validation/phase2d/canonical_case_summary.csv",
                "notes": decision.reason,
            })

    output_paths: list[Path] = list(template_paths) + [partial_path, full_path]
    output_paths += [
        _write_csv(VALIDATION_ROOT / "solver_claim_policy.csv", policy),
        _write_csv(VALIDATION_ROOT / "calibration_dependency_graph.csv", dependencies),
        _write_json(VALIDATION_ROOT / "calibration_readiness.json", readiness_payload),
        _write_csv(VALIDATION_ROOT / "canonical_case_summary.csv", summary_rows),
        _write_csv(VALIDATION_ROOT / "uncertainty_validation.csv", uncertainty_rows),
        _write_csv(VALIDATION_ROOT / "claim_maturity_registry.csv", maturity_rows),
        _write_csv(VALIDATION_ROOT / "phase2d_claim_registry.csv", claim_rows),
    ]

    full_h1 = next(result for scenario, result in results if scenario == "full_synthetic" and result.request.beam_case == "H1")
    figures = [
        plot_calibration_readiness(dependencies, FIGURE_ROOT),
        plot_solver_policy_matrix(policy, FIGURE_ROOT),
        plot_uncertainty_summary(full_h1.uncertainty_summary, FIGURE_ROOT),
        plot_calibration_state_comparison(summary_rows, FIGURE_ROOT),
    ]
    figure_paths = [Path(row[key]) for row in figures for key in ("png_path", "pdf_path")]
    output_paths.extend(figure_paths)

    upstream_after = upstream_hashes()
    unchanged = upstream_before == upstream_after
    outcome = "PHASE2D-B" if unchanged else "PHASE2D-C"
    outcome_report = {
        "phase": "PHASE 2D",
        "outcome": outcome,
        "outcome_statement": (
            "Solver governance is complete, but important predictions remain blocked by missing laboratory calibration values."
            if outcome == "PHASE2D-B"
            else "Calibration integration changed accepted upstream evidence and requires repair."
        ),
        "solver_governance_complete": True,
        "calibration_schema_valid": validation.valid_schema,
        "canonical_case_demonstrations": len(summary_rows),
        "synthetic_bundle_classification": "synthetic_not_experimental",
        "absolute_dimensions_unlocked": False,
        "absolute_fluence_unlocked": False,
        "experimentally_validated_prediction_present": False,
        "H1_automatic_objective_solver": full_h1.focal_field.solver,
        "H1_strict_hexagon": full_h1.metrics["strict_hexagon"],
        "upstream_hashes_unchanged": unchanged,
        "configured_first_order_efficiency_reapplied": False,
        "important_missing_measurements": sorted({
            row["required_measurement"] for row in dependencies if row["blocks_claim"]
        }),
        "scientific_scope": "linear optical prediction and calibration governance only",
    }
    outcome_path = _write_json(VALIDATION_ROOT / "phase2d_outcome_report.json", outcome_report)
    output_paths.append(outcome_path)

    manifest = {
        "phase": "PHASE 2D",
        "outcome": outcome,
        "generated_artifacts": sorted(str(path.relative_to(ROOT)).replace("\\", "/") for path in output_paths),
        "figure_manifest": figures,
        "upstream_hashes_before": upstream_before,
        "upstream_hashes_after": upstream_after,
        "upstream_hashes_unchanged": unchanged,
        "output_hashes": _hash_outputs(output_paths),
        "accepted_upstream_outputs_overwritten": False,
        "phase2c_solver_outputs_recomputed": False,
        "synthetic_results_experimentally_validated": False,
        "source_files": [
            "vbb_study/solver_policy.py",
            "vbb_study/calibration/schema.py",
            "vbb_study/calibration/io.py",
            "vbb_study/calibration/validation.py",
            "vbb_study/calibration/uncertainty.py",
            "vbb_study/digital_twin/phase2d_canonical_pipeline.py",
            "vbb_study/digital_twin/phase2d_figures.py",
            "tools/validate_calibration_bundle.py",
            "tools/run_phase2d_calibration_bridge.py",
        ],
    }
    manifest_path = _write_json(VALIDATION_ROOT / "phase2d_manifest.json", manifest)
    return {
        "outcome": outcome,
        "outcome_report": outcome_report,
        "manifest_path": str(manifest_path),
        "canonical_results": [result_to_dict(result) for _, result in results],
    }


def main() -> int:
    result = generate_phase2d_outputs()
    print(json.dumps({
        "outcome": result["outcome"],
        "manifest_path": result["manifest_path"],
        "upstream_hashes_unchanged": result["outcome_report"]["upstream_hashes_unchanged"],
    }, indent=2))
    return 0 if result["outcome"] in {"PHASE2D-A", "PHASE2D-B"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
