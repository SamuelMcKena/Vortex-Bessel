"""Canonical advanced-stage CSV metadata and honesty labels."""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd

from . import materials as material_schema
from .tables import propagation_power_label

ADVANCED_BEAM_OUTPUT_SCHEMA_VERSION = "1.0.0"

BEAM_FAMILIES = {
    "hexagonal_polygonal",
    "discrete_nfold",
    "nfold_vortex_ring",
    "hollow_polygon",
    "scalar_reference",
    "diagnostic",
}

MODEL_LEVELS = {
    "ideal_target",
    "focal_plane_target",
    "numerical_propagation",
    "lab_realistic",
    "hardware_route",
    "geometry_proxy",
    "diagnostic_only",
}

GENERATION_METHODS = {
    "phase_only_slm",
    "complex_amplitude_proxy",
    "amplitude_phase_target",
    "discrete_superposition",
    "holographic_phase_mask",
    "future_hardware_required",
    "simulation_only",
}

HARDWARE_STATUS = {
    "current_lab_realizable",
    "future_hardware_required",
    "simulation_only",
    "diagnostic_only",
}

PROPAGATION_STABILITY_STATUS = {
    "not_tested",
    "focal_plane_only",
    "propagation_tested_pass",
    "propagation_tested_marginal",
    "propagation_tested_fail",
    "diagnostic_only",
}

OPTICAL_MODEL_STATUS = {
    "optical_only",
    "geometry_proxy",
    "numerical_propagation",
    "diagnostic_visualisation",
}

ADVANCED_BEAM_COLUMNS = [
    "run_id",
    "generated_at_utc",
    "source_schema_version",
    "case_id",
    "preset",
    "path",
    "beam_family",
    "model_level",
    "generation_method",
    "hardware_status",
    "qa_status",
    "optical_model_status",
    "propagation_stability_status",
    "material_model_status",
    "calibration_status",
    "target_symmetry_order",
    "measured_symmetry_order",
    "symmetry_score",
    "target_polygon_sides",
    "outline_fidelity_score",
    "edge_uniformity_score",
    "core_suppression_score",
    "side_lobe_contamination_score",
    "accepted_depth_um",
    "accepted_depth_definition",
    "accepted_depth_fraction",
    "focal_plane_only",
    "propagation_tested",
    "complex_amplitude_required",
    "phase_only_compatible",
    "current_lab_realizable",
    "future_hardware_required",
    "simulation_only",
    "canonical_zone_um",
    "strict_bessel_region_um",
    "propagation_power_drift_fraction",
    "propagation_power_label",
]


def _run_id(run_id: str | None = None) -> str:
    return (
        run_id
        or os.environ.get("STRUCTURED_BEAM_RUN_ID")
        or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )


def _generated_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_known(value: str | None, allowed: set[str], field: str) -> str:
    text = str(value or "").strip()
    if text not in allowed:
        raise ValueError(f"{field} must be one of {sorted(allowed)}, got {value!r}")
    return text


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def propagation_stability_label(
    *,
    propagation_tested: bool,
    focal_plane_only: bool = False,
    accepted_depth_um: Any = None,
    accepted_depth_fraction: Any = None,
    diagnostic_only: bool = False,
    pass_fraction: float = 0.65,
    marginal_fraction: float = 0.25,
) -> str:
    """Return an honest z-stability label from measured propagation metrics."""

    if diagnostic_only:
        return "diagnostic_only"
    if focal_plane_only:
        return "focal_plane_only"
    if not propagation_tested:
        return "not_tested"
    fraction = _finite(accepted_depth_fraction)
    depth = _finite(accepted_depth_um)
    if fraction is not None:
        if fraction >= float(pass_fraction):
            return "propagation_tested_pass"
        if fraction >= float(marginal_fraction):
            return "propagation_tested_marginal"
        return "propagation_tested_fail"
    if depth is not None:
        if depth > 0.0:
            return "propagation_tested_marginal"
        return "propagation_tested_fail"
    return "propagation_tested_fail"


def phase_only_feasibility_label(
    *,
    generation_method: str,
    phase_only_compatible: bool,
    complex_amplitude_required: bool = False,
    simulation_only: bool = False,
    diagnostic_only: bool = False,
    phase_only_encoding_tested: bool = False,
) -> str:
    """Return the conservative hardware status for one advanced-beam row."""

    method = _coerce_known(generation_method, GENERATION_METHODS, "generation_method")
    if diagnostic_only or method == "diagnostic_only":
        return "diagnostic_only"
    if simulation_only or method == "simulation_only":
        return "simulation_only"
    if complex_amplitude_required and not phase_only_encoding_tested:
        return "future_hardware_required"
    if method in {"amplitude_phase_target", "complex_amplitude_proxy", "future_hardware_required"}:
        return "future_hardware_required"
    if phase_only_compatible and method in {"phase_only_slm", "holographic_phase_mask"}:
        return "current_lab_realizable"
    return "simulation_only"


def polygonal_acceptance_label(row: dict[str, Any]) -> str:
    """Return an optical/geometry acceptance label, never a material claim."""

    if str(row.get("optical_model_status", "")) == "diagnostic_visualisation":
        return "diagnostic_only"
    if _bool(row.get("focal_plane_only")):
        return "focal_plane_only"
    stability = str(row.get("propagation_stability_status", "not_tested"))
    hardware = str(row.get("hardware_status", "simulation_only"))
    if stability == "propagation_tested_pass" and hardware == "current_lab_realizable":
        return "propagation_tested_current_lab_candidate"
    if stability == "propagation_tested_pass":
        return "propagation_tested_simulation_candidate"
    if stability == "propagation_tested_marginal":
        return "propagation_tested_marginal"
    if stability == "propagation_tested_fail":
        return "propagation_tested_fail"
    return "exploratory_optical_case"


def annotate_advanced_beam_row(
    row: dict[str, Any],
    *,
    run_id: str | None = None,
    qa_status: str = "exploratory",
) -> dict[str, Any]:
    """Stamp one advanced-stage row with native schema metadata and guardrails."""

    row.setdefault("run_id", _run_id(run_id))
    row.setdefault("generated_at_utc", _generated_at())
    row.setdefault("source_schema_version", ADVANCED_BEAM_OUTPUT_SCHEMA_VERSION)
    row.setdefault("case_id", row.get("case_id", "advanced_case"))
    row.setdefault("preset", row.get("preset", "fast"))
    row.setdefault("path", row.get("path", "not_applicable"))
    row.setdefault("beam_family", row.get("beam_family", "diagnostic"))
    row["beam_family"] = _coerce_known(row.get("beam_family"), BEAM_FAMILIES, "beam_family")
    row.setdefault("model_level", row.get("model_level", "diagnostic_only"))
    row["model_level"] = _coerce_known(row.get("model_level"), MODEL_LEVELS, "model_level")
    row.setdefault("generation_method", row.get("generation_method", "simulation_only"))
    row["generation_method"] = _coerce_known(row.get("generation_method"), GENERATION_METHODS, "generation_method")
    row.setdefault("qa_status", qa_status)
    row.setdefault("optical_model_status", row.get("optical_model_status", "optical_only"))
    row["optical_model_status"] = _coerce_known(
        row.get("optical_model_status"),
        OPTICAL_MODEL_STATUS,
        "optical_model_status",
    )
    row.setdefault("material_model_status", "optical_only")
    row["material_model_status"] = _coerce_known(
        row.get("material_model_status"),
        material_schema.MATERIAL_MODEL_STATUS,
        "material_model_status",
    )
    row.setdefault("calibration_status", "uncalibrated")
    row["calibration_status"] = _coerce_known(
        row.get("calibration_status"),
        material_schema.CALIBRATION_STATUS,
        "calibration_status",
    )
    if row["material_model_status"] not in {"optical_only", "diagnostic_only"}:
        raise ValueError("Stage 8 advanced rows must remain optical/geometry only.")

    focal_plane_only = _bool(row.get("focal_plane_only", False))
    propagation_tested = _bool(row.get("propagation_tested", False))
    if focal_plane_only and propagation_tested:
        raise ValueError("focal-plane-only rows cannot be marked propagation-tested")
    row["focal_plane_only"] = focal_plane_only
    row["propagation_tested"] = propagation_tested

    row.setdefault("complex_amplitude_required", row["generation_method"] in {"complex_amplitude_proxy", "amplitude_phase_target"})
    row.setdefault("phase_only_compatible", row["generation_method"] in {"phase_only_slm", "holographic_phase_mask"})
    row["complex_amplitude_required"] = _bool(row.get("complex_amplitude_required"))
    row["phase_only_compatible"] = _bool(row.get("phase_only_compatible"))
    phase_only_encoding_tested = _bool(row.get("phase_only_encoding_tested", False))
    if row["complex_amplitude_required"] and row["phase_only_compatible"] and not phase_only_encoding_tested:
        raise ValueError("complex-amplitude rows need tested encoding before phase_only_compatible=True")

    inferred_hardware = phase_only_feasibility_label(
        generation_method=str(row["generation_method"]),
        phase_only_compatible=bool(row["phase_only_compatible"]),
        complex_amplitude_required=bool(row["complex_amplitude_required"]),
        simulation_only=_bool(row.get("simulation_only", False)),
        diagnostic_only=row["model_level"] == "diagnostic_only",
        phase_only_encoding_tested=phase_only_encoding_tested,
    )
    row.setdefault("hardware_status", inferred_hardware)
    row["hardware_status"] = _coerce_known(row.get("hardware_status"), HARDWARE_STATUS, "hardware_status")
    if row["hardware_status"] == "current_lab_realizable" and inferred_hardware != "current_lab_realizable":
        raise ValueError("current_lab_realizable requires a tested phase-only-compatible route")
    row["current_lab_realizable"] = row["hardware_status"] == "current_lab_realizable"
    row["future_hardware_required"] = row["hardware_status"] == "future_hardware_required"
    row["simulation_only"] = row["hardware_status"] == "simulation_only"

    row.setdefault(
        "propagation_stability_status",
        propagation_stability_label(
            propagation_tested=bool(row["propagation_tested"]),
            focal_plane_only=bool(row["focal_plane_only"]),
            accepted_depth_um=row.get("accepted_depth_um"),
            accepted_depth_fraction=row.get("accepted_depth_fraction"),
            diagnostic_only=row["model_level"] == "diagnostic_only",
        ),
    )
    row["propagation_stability_status"] = _coerce_known(
        row.get("propagation_stability_status"),
        PROPAGATION_STABILITY_STATUS,
        "propagation_stability_status",
    )
    if row["focal_plane_only"] and row["propagation_stability_status"] != "focal_plane_only":
        raise ValueError("focal-plane-only rows require propagation_stability_status='focal_plane_only'")
    if not row["propagation_tested"] and row["propagation_stability_status"].startswith("propagation_tested"):
        raise ValueError("propagation-tested labels require propagation_tested=True")

    row.setdefault("target_symmetry_order", pd.NA)
    row.setdefault("measured_symmetry_order", pd.NA)
    row.setdefault("symmetry_score", pd.NA)
    row.setdefault("target_polygon_sides", row.get("target_symmetry_order", pd.NA))
    row.setdefault("outline_fidelity_score", pd.NA)
    row.setdefault("edge_uniformity_score", pd.NA)
    row.setdefault("core_suppression_score", pd.NA)
    row.setdefault("side_lobe_contamination_score", pd.NA)
    row.setdefault("accepted_depth_um", 0.0 if row["focal_plane_only"] else pd.NA)
    row.setdefault("accepted_depth_definition", "not_applicable" if row["focal_plane_only"] else "not_reported")
    row.setdefault("accepted_depth_fraction", 0.0 if row["focal_plane_only"] else pd.NA)
    row.setdefault("canonical_zone_um", pd.NA)
    row.setdefault("strict_bessel_region_um", pd.NA)
    if "propagation_power_label" not in row and "propagation_power_drift_fraction" in row:
        try:
            row["propagation_power_label"] = propagation_power_label(float(row["propagation_power_drift_fraction"]))
        except (TypeError, ValueError):
            row["propagation_power_label"] = "unknown"
    row.setdefault("propagation_power_label", "unknown")
    row["advanced_acceptance_label"] = polygonal_acceptance_label(row)
    row["material_writing_success_claimed"] = False
    row["stable_written_channel_claimed"] = False
    row["overclaim_guardrail"] = (
        "advanced optical/geometry result; not a material-writing or stable-channel claim"
    )
    return row


def ordered_advanced_beam_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return one row ordered with canonical advanced-stage columns first."""

    ordered = {key: row.get(key, pd.NA) for key in ADVANCED_BEAM_COLUMNS}
    extra = {key: value for key, value in row.items() if key not in ADVANCED_BEAM_COLUMNS}
    return {**ordered, **extra}


def ordered_advanced_beam_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Return a DataFrame with canonical advanced-stage columns first."""

    df = pd.DataFrame([ordered_advanced_beam_row(dict(row)) for row in rows])
    if df.empty:
        return df
    extra = [col for col in df.columns if col not in set(ADVANCED_BEAM_COLUMNS)]
    return df[ADVANCED_BEAM_COLUMNS + extra].copy()


__all__ = [
    "ADVANCED_BEAM_COLUMNS",
    "ADVANCED_BEAM_OUTPUT_SCHEMA_VERSION",
    "BEAM_FAMILIES",
    "GENERATION_METHODS",
    "HARDWARE_STATUS",
    "MODEL_LEVELS",
    "OPTICAL_MODEL_STATUS",
    "PROPAGATION_STABILITY_STATUS",
    "annotate_advanced_beam_row",
    "ordered_advanced_beam_frame",
    "ordered_advanced_beam_row",
    "phase_only_feasibility_label",
    "polygonal_acceptance_label",
    "propagation_stability_label",
]
