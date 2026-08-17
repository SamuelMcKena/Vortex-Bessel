"""Canonical capsule-stage CSV metadata and geometry-proxy labels."""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd

from . import materials as material_schema
from .tables import propagation_power_label

CAPSULE_OUTPUT_SCHEMA_VERSION = "1.0.0"

GEOMETRY_MODEL_STATUS = {
    "optical_geometry_proxy",
    "thresholded_fluence_proxy",
    "normalised_visualisation",
    "experimentally_measured",
    "diagnostic_only",
}

FEATURE_TARGET_TYPES = {
    "capsule",
    "elongated_channel",
    "weld_line_proxy",
    "through_glass_via_proxy",
    "waveguide_proxy",
    "generic_feature",
}

FEATURE_PROXY_TYPES = {
    "thresholded_xy_area",
    "thresholded_xz_length",
    "capsule_overlap_proxy",
    "accepted_depth_proxy",
    "line_fluence_visualisation",
    "diagnostic_only",
}

CAPSULE_COLUMNS = [
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
    "material_model_status",
    "material_response_model",
    "calibration_status",
    "geometry_model_status",
    "feature_target_type",
    "feature_proxy_type",
    "target_width_um",
    "target_length_um",
    "target_depth_um",
    "target_aspect_ratio",
    "predicted_width_um",
    "predicted_length_um",
    "predicted_depth_um",
    "predicted_aspect_ratio",
    "width_error_pct",
    "length_error_pct",
    "depth_error_pct",
    "overlap_score",
    "capsule_fit_score",
    "edge_uniformity_score",
    "core_suppression_score",
    "side_lobe_contamination_score",
    "threshold_fluence_J_cm2",
    "peak_fluence_J_cm2",
    "fluence_to_threshold_ratio",
    "xz_proxy_definition",
    "xz_energy_conservation_status",
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


def _is_present(value: Any) -> bool:
    if value is None or value is pd.NA:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    return str(value).strip().lower() not in {"", "none", "nan", "not_applicable"}


def _pct_error(predicted: Any, target: Any) -> Any:
    try:
        pred = float(predicted)
        tgt = float(target)
    except (TypeError, ValueError):
        return pd.NA
    if not math.isfinite(pred) or not math.isfinite(tgt) or abs(tgt) < 1e-30:
        return pd.NA
    return float(100.0 * (pred - tgt) / tgt)


def capsule_fit_label(score: Any) -> str:
    """Return a compact fit-quality label for a capsule score."""

    try:
        value = float(score)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(value):
        return "unknown"
    if value >= 0.80:
        return "strong_proxy_match"
    if value >= 0.55:
        return "moderate_proxy_match"
    return "weak_proxy_match"


def geometry_status_label(
    *,
    feature_proxy_type: str,
    xz_energy_conservation_status: str = "normalised_visualisation",
    measured_geometry_evidence: bool = False,
) -> str:
    """Return an honest geometry model status for one capsule row."""

    proxy = _coerce_known(feature_proxy_type, FEATURE_PROXY_TYPES, "feature_proxy_type")
    xz_status = _coerce_known(
        xz_energy_conservation_status,
        material_schema.XZ_ENERGY_CONSERVATION_STATUS,
        "xz_energy_conservation_status",
    )
    if measured_geometry_evidence:
        return "experimentally_measured"
    if proxy == "diagnostic_only":
        return "diagnostic_only"
    if proxy == "line_fluence_visualisation" or xz_status == "non_energy_conserving_line_proxy":
        return "normalised_visualisation"
    if proxy in {"thresholded_xy_area", "thresholded_xz_length", "capsule_overlap_proxy", "accepted_depth_proxy"}:
        return "thresholded_fluence_proxy"
    return "optical_geometry_proxy"


def capsule_acceptance_label(row: dict[str, Any]) -> str:
    """Return a planning acceptance label; never a physical weld-success claim."""

    if str(row.get("geometry_model_status", "")) in {"diagnostic_only", "normalised_visualisation"}:
        return "diagnostic_only"
    if str(row.get("propagation_power_label", "unknown")) == "fail":
        return "not_design_ready"
    if str(row.get("calibration_status", "uncalibrated")) == "uncalibrated":
        if str(row.get("propagation_power_label", "unknown")) == "pass" and float(row.get("capsule_fit_score", 0.0) or 0.0) >= 0.55:
            return "planning_proxy_candidate"
        return "exploratory_proxy"
    return "review_calibration_before_use"


def annotate_capsule_row(
    row: dict[str, Any],
    *,
    run_id: str | None = None,
    qa_status: str = "exploratory",
) -> dict[str, Any]:
    """Stamp one capsule-stage row with native schema metadata."""

    row.setdefault("run_id", _run_id(run_id))
    row.setdefault("generated_at_utc", _generated_at())
    row.setdefault("source_schema_version", CAPSULE_OUTPUT_SCHEMA_VERSION)
    row.setdefault("case_id", row.get("case_id", "capsule_case"))
    row.setdefault("preset", row.get("preset", "fast"))
    row.setdefault("path", row.get("path", "not_applicable"))
    row.setdefault("beam_family", row.get("beam_family", "capsule_geometry_proxy"))
    row.setdefault("model_level", row.get("model_level", "application_geometry_proxy"))
    row.setdefault("generation_method", row.get("generation_method", "scalar_propagation"))
    row.setdefault("hardware_status", row.get("hardware_status", "not_applicable"))
    row.setdefault("qa_status", qa_status)

    row.setdefault("material_model_status", "planning_proxy")
    row.setdefault("material_response_model", "incubation_threshold_proxy")
    row.setdefault("calibration_status", "uncalibrated")
    row["material_model_status"] = _coerce_known(
        row.get("material_model_status"),
        material_schema.MATERIAL_MODEL_STATUS,
        "material_model_status",
    )
    row["material_response_model"] = _coerce_known(
        row.get("material_response_model"),
        material_schema.MATERIAL_RESPONSE_MODELS,
        "material_response_model",
    )
    row["calibration_status"] = _coerce_known(
        row.get("calibration_status"),
        material_schema.CALIBRATION_STATUS,
        "calibration_status",
    )
    if row["calibration_status"] == "uncalibrated" and row["material_model_status"] == "experimentally_calibrated":
        raise ValueError("uncalibrated capsule rows cannot be experimentally_calibrated")

    row.setdefault("feature_target_type", "capsule")
    row.setdefault("feature_proxy_type", "capsule_overlap_proxy")
    row["feature_target_type"] = _coerce_known(
        row.get("feature_target_type"),
        FEATURE_TARGET_TYPES,
        "feature_target_type",
    )
    row["feature_proxy_type"] = _coerce_known(
        row.get("feature_proxy_type"),
        FEATURE_PROXY_TYPES,
        "feature_proxy_type",
    )
    row.setdefault("xz_energy_conservation_status", "normalised_visualisation")
    row["xz_energy_conservation_status"] = _coerce_known(
        row.get("xz_energy_conservation_status"),
        material_schema.XZ_ENERGY_CONSERVATION_STATUS,
        "xz_energy_conservation_status",
    )
    row.setdefault(
        "geometry_model_status",
        geometry_status_label(
            feature_proxy_type=str(row["feature_proxy_type"]),
            xz_energy_conservation_status=str(row["xz_energy_conservation_status"]),
            measured_geometry_evidence=_is_present(row.get("measured_geometry_dataset_id")),
        ),
    )
    row["geometry_model_status"] = _coerce_known(
        row.get("geometry_model_status"),
        GEOMETRY_MODEL_STATUS,
        "geometry_model_status",
    )
    if row["geometry_model_status"] == "experimentally_measured" and not _is_present(row.get("measured_geometry_dataset_id")):
        raise ValueError("experimentally_measured geometry requires measured_geometry_dataset_id")

    if "target_aspect_ratio" not in row:
        try:
            row["target_aspect_ratio"] = float(row["target_length_um"]) / max(float(row["target_width_um"]), 1e-30)
        except (TypeError, ValueError, KeyError):
            row["target_aspect_ratio"] = pd.NA
    if "predicted_aspect_ratio" not in row:
        try:
            row["predicted_aspect_ratio"] = float(row["predicted_length_um"]) / max(float(row["predicted_width_um"]), 1e-30)
        except (TypeError, ValueError, KeyError):
            row["predicted_aspect_ratio"] = pd.NA
    row.setdefault("width_error_pct", _pct_error(row.get("predicted_width_um"), row.get("target_width_um")))
    row.setdefault("length_error_pct", _pct_error(row.get("predicted_length_um"), row.get("target_length_um")))
    row.setdefault("depth_error_pct", _pct_error(row.get("predicted_depth_um"), row.get("target_depth_um")))
    row.setdefault("overlap_score", pd.NA)
    row.setdefault("capsule_fit_score", row.get("overlap_score", pd.NA))
    row.setdefault("edge_uniformity_score", pd.NA)
    row.setdefault("core_suppression_score", pd.NA)
    row.setdefault("side_lobe_contamination_score", pd.NA)
    if "fluence_to_threshold_ratio" not in row:
        try:
            row["fluence_to_threshold_ratio"] = float(row["peak_fluence_J_cm2"]) / float(row["threshold_fluence_J_cm2"])
        except (TypeError, ValueError, ZeroDivisionError, KeyError):
            row["fluence_to_threshold_ratio"] = pd.NA
    if "propagation_power_label" not in row and "propagation_power_drift_fraction" in row:
        try:
            row["propagation_power_label"] = propagation_power_label(float(row["propagation_power_drift_fraction"]))
        except (TypeError, ValueError):
            row["propagation_power_label"] = "unknown"
    row.setdefault("propagation_power_label", "unknown")
    row["capsule_fit_label"] = capsule_fit_label(row.get("capsule_fit_score"))
    row["capsule_acceptance_label"] = capsule_acceptance_label(row)
    row["actual_weld_success_claimed"] = False
    row["overclaim_guardrail"] = (
        "application-planning geometry proxy; not actual weld, bonding, void, ablation, or index-change prediction"
    )
    return row


def ordered_capsule_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return one row ordered with canonical capsule columns first."""

    ordered = {key: row.get(key, pd.NA) for key in CAPSULE_COLUMNS}
    extra = {key: value for key, value in row.items() if key not in CAPSULE_COLUMNS}
    return {**ordered, **extra}


def ordered_capsule_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Return a DataFrame with canonical capsule columns first."""

    df = pd.DataFrame([ordered_capsule_row(dict(row)) for row in rows])
    if df.empty:
        return df
    extra = [col for col in df.columns if col not in set(CAPSULE_COLUMNS)]
    return df[CAPSULE_COLUMNS + extra].copy()


__all__ = [
    "CAPSULE_COLUMNS",
    "CAPSULE_OUTPUT_SCHEMA_VERSION",
    "FEATURE_PROXY_TYPES",
    "FEATURE_TARGET_TYPES",
    "GEOMETRY_MODEL_STATUS",
    "annotate_capsule_row",
    "capsule_acceptance_label",
    "capsule_fit_label",
    "geometry_status_label",
    "ordered_capsule_frame",
    "ordered_capsule_row",
]
