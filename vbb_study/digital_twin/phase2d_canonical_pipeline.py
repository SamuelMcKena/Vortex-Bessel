"""Governed Phase 2D integration over accepted Phase 2A and Phase 2C evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from vbb_study.calibration.io import load_calibration_bundle
from vbb_study.calibration.schema import CalibrationBundle, canonical_calibration_template
from vbb_study.calibration.uncertainty import UncertaintyConfig, propagate_calibration_uncertainty
from vbb_study.calibration.validation import calibration_readiness_for_claim, validate_calibration_bundle
from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest
from vbb_study.solver_policy import (
    BEAM_CASES,
    CLAIM_TYPES,
    ClaimType,
    OpticalFidelityMode,
    SolverDecision,
    select_solver_for_claim,
)


ROOT = Path(__file__).resolve().parents[2]
PHASE2A_ROOT = ROOT / "outputs" / "validation" / "phase2a"
PHASE2C_ROOT = ROOT / "outputs" / "validation" / "phase2c"

UPSTREAM_GOVERNANCE_FILES = (
    ROOT / "outputs" / "validation" / "phase1_critical_repairs" / "phase1_repair_summary.json",
    ROOT / "outputs" / "validation" / "phase1_reconciliation" / "phase1r_regeneration_manifest.json",
    PHASE2A_ROOT / "canonical_hardware_manifest.json",
    PHASE2A_ROOT / "canonical_case_summary.csv",
    PHASE2A_ROOT / "canonical_power_ledgers.csv",
    ROOT / "outputs" / "figures" / "phase2b_visual_diagnostics" / "00_manifests" / "phase2b_final_manifest.json",
    PHASE2C_ROOT / "phase2c_objective_benchmark.csv",
    PHASE2C_ROOT / "phase2c_interface_benchmark.csv",
    PHASE2C_ROOT / "phase2c_outcome_report.json",
)


@dataclass(frozen=True)
class CanonicalOpticalRequest:
    beam_case: str
    claim_types: tuple[str, ...]
    fidelity_mode: str = "automatic_by_claim"
    calibration_bundle_path: str | None = None
    include_interface: bool = True
    material_propagation_distance_m: float = 10.0e-6
    uncertainty_samples: int = 0


@dataclass(frozen=True)
class CanonicalFieldReference:
    field_stage: str
    solver: str
    evidence_path: str
    plane: str
    data_status: str
    notes: str


@dataclass
class CanonicalOpticalResult:
    request: CanonicalOpticalRequest
    solver_decisions: list[SolverDecision]
    pupil_field: CanonicalFieldReference
    focal_field: CanonicalFieldReference
    transmitted_field: CanonicalFieldReference | None
    material_field: CanonicalFieldReference | None
    metrics: dict[str, Any]
    calibration_status: dict[str, Any]
    uncertainty_summary: dict[str, Any]
    provenance: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upstream_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
        for path in UPSTREAM_GOVERNANCE_FILES
        if path.is_file()
    }


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _row(path: Path, **matches: str) -> dict[str, str]:
    rows = [
        row for row in _csv_rows(path)
        if all(row.get(key) == value for key, value in matches.items())
    ]
    if len(rows) != 1:
        raise RuntimeError(f"expected one accepted row in {path.name} for {matches}, found {len(rows)}")
    return rows[0]


def _number(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _boolean(value: str | None) -> bool | None:
    if value in (None, ""):
        return None
    return str(value).lower() == "true"


def _claim_maturity(
    claim: str,
    readiness: dict[str, Any],
    bundle: CalibrationBundle,
    bundle_supplied: bool,
) -> str:
    if not bundle_supplied:
        return "fixed_bench_nominal_prediction"
    if bundle.is_synthetic:
        return "calibration_ready_prediction"
    if not readiness["ready"]:
        return "calibration_ready_prediction"
    if claim == "absolute_fluence":
        maturity = "calibrated_fluence_prediction"
    else:
        maturity = "calibrated_optical_prediction"
    comparison = bundle.data.get("experimental_comparison", {})
    if comparison.get("performed") and comparison.get("acceptance_passed") and comparison.get("evidence_path"):
        return "experimentally_validated_prediction"
    return maturity


def _selected_metrics(
    beam: str,
    solver: str,
    objective: dict[str, str],
    interface: dict[str, str] | None,
    phase2a: dict[str, str],
) -> dict[str, Any]:
    vector = solver == "vector_debye"
    prefix = "vector" if vector else "scalar"
    metrics: dict[str, Any] = {
        "beam_case": beam,
        "objective_solver": solver,
        "metric_reference_solver": solver,
        "feature_definition": objective.get("feature_definition"),
        "feature_or_ring_radius_um": _number(objective.get(f"{prefix}_feature_radius_um")),
        "peak_location_shift_um_scalar_vs_vector": _number(objective.get("peak_location_shift_um")),
        "scalar_vector_intensity_correlation": _number(objective.get("scalar_vector_intensity_correlation")),
        "longitudinal_power_fraction": _number(objective.get("longitudinal_power_fraction")) if vector else None,
        "longitudinal_field_status": "modelled_vector_reference" if vector else "not_modelled_by_scalar_route",
        "morphology_classification": objective.get("morphology_classification"),
        "scalar_output_pixel_um": _number(objective.get("scalar_output_pixel_um")),
        "requested_topological_charge": _number(objective.get("requested_topological_charge")),
        "measured_winding": _number(objective.get("measured_vector_transverse_winding" if vector else "measured_scalar_winding")),
        "phase2a_route_variant": phase2a.get("route_variant"),
        "phase2a_first_order_efficiency": _number(phase2a.get("first_order_efficiency")),
        "phase2a_sample_plane_energy_J_nominal": _number(phase2a.get("sample_plane_energy_J")),
        "phase2a_peak_fluence_J_cm2_nominal_calibration_blocked": _number(phase2a.get("model_plane_peak_fluence_J_cm2")),
        "strict_hexagon": None,
    }
    if beam == "H1":
        metrics.update({
            "strict_hexagon": _boolean(objective.get(f"H1_strict_hexagon_{prefix}")),
            "C6": _number(objective.get(f"H1_C6_{prefix}")),
            "C3": _number(objective.get(f"H1_C3_{prefix}")),
            "edge_sharpness_mm_inv": _number(objective.get(f"H1_edge_sharpness_{prefix}_mm_inv")),
            "ridge_width_um": _number(objective.get(f"H1_ridge_width_{prefix}_um")),
            "transition_width_um": _number(objective.get(f"H1_transition_width_{prefix}_um")),
            "relative_feature_radius_difference_scalar_vector": _number(
                objective.get("relative_feature_radius_difference")
            ),
        })
    if interface is not None:
        interface_solver = metrics["interface_solver"] = interface.get("selected_interface_solver")
        if interface_solver == "vector_spectral_fresnel":
            metrics["interface_transmitted_power_fraction"] = _number(
                interface.get("transmitted_power_fraction_vector")
            )
        else:
            metrics["interface_transmitted_power_fraction"] = _number(
                interface.get("transmitted_power_fraction_scalar")
            )
        metrics["post_interface_scalar_vector_correlation"] = _number(
            interface.get("post_interface_intensity_correlation")
        )
    return metrics


def run_canonical_optical_prediction(request: CanonicalOpticalRequest) -> CanonicalOpticalResult:
    """Resolve a governed prediction from immutable accepted upstream evidence."""

    beam = str(request.beam_case).upper()
    if beam not in BEAM_CASES:
        raise ValueError(f"unknown canonical beam case: {request.beam_case!r}")
    if not request.claim_types:
        raise ValueError("at least one claim type is required")
    if request.material_propagation_distance_m < 0.0:
        raise ValueError("material propagation distance must be non-negative")
    invalid_claims = sorted(set(request.claim_types) - set(CLAIM_TYPES))
    if invalid_claims:
        raise ValueError(f"unknown claim types: {invalid_claims}")

    decisions = [
        select_solver_for_claim(
            beam,
            cast(ClaimType, claim),
            cast(OpticalFidelityMode, request.fidelity_mode),
        )
        for claim in request.claim_types
    ]
    focal_solver = "vector_debye" if any(
        decision.selected_objective_solver == "vector_debye" for decision in decisions
    ) else "scalar_fft"
    interface_solver = "vector_spectral_fresnel" if any(
        decision.selected_interface_solver == "vector_spectral_fresnel" for decision in decisions
    ) else "scalar_normal_incidence_fresnel"

    objective_path = PHASE2C_ROOT / "phase2c_objective_benchmark.csv"
    interface_path = PHASE2C_ROOT / "phase2c_interface_benchmark.csv"
    phase2a_summary_path = PHASE2A_ROOT / "canonical_case_summary.csv"
    ledger_path = PHASE2A_ROOT / "canonical_power_ledgers.csv"
    objective_row = _row(objective_path, case_id=beam)
    phase2a_row = _row(
        phase2a_summary_path,
        case_id=beam,
        route_variant="realistic_fixed_bench_route",
    )

    interface_row: dict[str, str] | None = None
    if request.include_interface:
        interface_row = _row(interface_path, case_id=beam)
        interface_row = dict(interface_row)
        interface_row["selected_interface_solver"] = interface_solver

    ledger_rows = [
        row for row in _csv_rows(ledger_path)
        if row["case_id"] == beam and row["route_variant"] == "realistic_fixed_bench_route"
    ]
    selected_order_rows = [row for row in ledger_rows if row["factor_id"] == "simulated_selected_first_order"]
    if len(selected_order_rows) != 1:
        raise RuntimeError("accepted Phase 2A ledger must contain exactly one selected first-order factor")

    metrics = _selected_metrics(beam, focal_solver, objective_row, interface_row, phase2a_row)
    bundle_supplied = request.calibration_bundle_path is not None
    bundle = (
        load_calibration_bundle(Path(request.calibration_bundle_path))
        if bundle_supplied
        else CalibrationBundle(canonical_calibration_template())
    )
    validation = validate_calibration_bundle(bundle)
    if bundle_supplied and not validation.valid_schema:
        raise ValueError("invalid calibration bundle: " + "; ".join(validation.errors))

    readiness: dict[str, Any] = {}
    maturity: dict[str, str] = {}
    for claim in request.claim_types:
        status = calibration_readiness_for_claim(bundle, claim)
        readiness[claim] = asdict(status)
        maturity[claim] = _claim_maturity(claim, readiness[claim], bundle, bundle_supplied)

    sample_energy = float(phase2a_row["sample_plane_energy_J"])
    nominal_peak_fluence = float(phase2a_row["model_plane_peak_fluence_J_cm2"])
    reference_metrics = {
        "reference_wavelength_m": 1.029e-6,
        "reference_objective_NA": 0.45,
        "carrier_frequency_cpm": 6250.0,
        "feature_radius_m": float(metrics["feature_or_ring_radius_um"]) * 1.0e-6,
        "reference_output_pixel_m": float(objective_row["scalar_output_pixel_um"]) * 1.0e-6,
        "first_order_efficiency": float(phase2a_row["first_order_efficiency"]),
        "input_aperture_fraction": float(phase2a_row["input_aperture_fraction"]),
        "objective_pupil_fraction": float(phase2a_row["objective_pupil_fraction"]),
        "peak_shape_factor_m_inv2": nominal_peak_fluence * 1.0e4 / sample_energy,
    }
    uncertainty = propagate_calibration_uncertainty(
        request,
        bundle,
        UncertaintyConfig(samples=request.uncertainty_samples),
        reference_metrics=reference_metrics,
    )

    abs_dimensions = readiness.get("absolute_dimensions", {"ready": False})["ready"]
    abs_fluence = readiness.get("absolute_fluence", {"ready": False})["ready"]
    real_calibration = bundle_supplied and not bundle.is_synthetic
    calibration_status = {
        "bundle_supplied": bundle_supplied,
        "calibration_id": bundle.calibration_id,
        "data_classification": bundle.data_classification,
        "synthetic_not_experimental": bundle.is_synthetic,
        "schema_valid": validation.valid_schema,
        "schema_errors": list(validation.errors),
        "schema_warnings": list(validation.warnings),
        "missing_values": list(validation.missing_values),
        "claim_readiness": readiness,
        "claim_maturity": maturity,
        "enabled_without_calibration": [
            claim for claim, status in readiness.items() if status["status"] == "ready_without_calibration"
        ],
        "blocked_claims": [claim for claim, status in readiness.items() if not status["ready"]],
        "absolute_dimensions_unlocked": bool(real_calibration and abs_dimensions),
        "absolute_fluence_unlocked": bool(real_calibration and abs_fluence),
        "experimentally_validated": any(
            level == "experimentally_validated_prediction" for level in maturity.values()
        ),
    }

    pupil_field = CanonicalFieldReference(
        field_stage="validated SLM / 4F pupil handoff",
        solver="phase2a_realistic_fixed_bench_route",
        evidence_path=str(phase2a_summary_path.relative_to(ROOT)).replace("\\", "/"),
        plane="objective pupil input",
        data_status="accepted_upstream_reference_not_recomputed",
        notes="References the canonical Phase 2A hardware route and existing beam generators.",
    )
    focal_field = CanonicalFieldReference(
        field_stage="objective focal field",
        solver=focal_solver,
        evidence_path=str(objective_path.relative_to(ROOT)).replace("\\", "/"),
        plane="matched objective focal plane z=0",
        data_status="accepted_phase2c_benchmark_reference",
        notes="Phase 2D does not duplicate or silently rerun the Phase 2C field generator.",
    )
    transmitted_field = None
    material_field = None
    if request.include_interface:
        transmitted_field = CanonicalFieldReference(
            field_stage="post-interface field",
            solver=interface_solver,
            evidence_path=str(interface_path.relative_to(ROOT)).replace("\\", "/"),
            plane="air-to-material interface",
            data_status="accepted_phase2c_benchmark_reference",
            notes="No Phase 2A sample-surface transmission factor is reapplied.",
        )
        if request.material_propagation_distance_m > 0.0:
            material_field = CanonicalFieldReference(
                field_stage="material optical field",
                solver="vector_angular_spectrum" if interface_solver.startswith("vector") else "scalar_angular_spectrum",
                evidence_path=str(interface_path.relative_to(ROOT)).replace("\\", "/"),
                plane=f"material z={request.material_propagation_distance_m:.9g} m",
                data_status="accepted_phase2c_benchmark_reference",
                notes="Linear optical propagation only; no nonlinear or material-response claim.",
            )

    provenance = {
        "hardware_manifest_source": "vbb_study.digital_twin.phase2a_contracts.canonical_hardware_manifest",
        "hardware_id": canonical_hardware_manifest()["hardware_id"],
        "phase2a_energy_ledger_path": str(ledger_path.relative_to(ROOT)).replace("\\", "/"),
        "phase2a_energy_ledger_row_count": len(ledger_rows),
        "selected_first_order_factor_count": len(selected_order_rows),
        "selected_first_order_factor_id": selected_order_rows[0]["factor_id"],
        "configured_first_order_factor_reapplied": False,
        "phase2c_objective_evidence": str(objective_path.relative_to(ROOT)).replace("\\", "/"),
        "phase2c_interface_evidence": str(interface_path.relative_to(ROOT)).replace("\\", "/"),
        "phase2c_solver_outputs_recomputed": False,
        "upstream_hashes": upstream_hashes(),
    }
    return CanonicalOpticalResult(
        request=request,
        solver_decisions=decisions,
        pupil_field=pupil_field,
        focal_field=focal_field,
        transmitted_field=transmitted_field,
        material_field=material_field,
        metrics=metrics,
        calibration_status=calibration_status,
        uncertainty_summary=uncertainty,
        provenance=provenance,
    )


def result_to_dict(result: CanonicalOpticalResult) -> dict[str, Any]:
    return {
        "request": asdict(result.request),
        "solver_decisions": [asdict(decision) for decision in result.solver_decisions],
        "pupil_field": asdict(result.pupil_field),
        "focal_field": asdict(result.focal_field),
        "transmitted_field": None if result.transmitted_field is None else asdict(result.transmitted_field),
        "material_field": None if result.material_field is None else asdict(result.material_field),
        "metrics": result.metrics,
        "calibration_status": result.calibration_status,
        "uncertainty_summary": result.uncertainty_summary,
        "provenance": result.provenance,
    }
