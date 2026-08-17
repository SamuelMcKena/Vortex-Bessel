"""Canonical materials-stage CSV metadata and honesty labels.

The materials stage compares optical fluence fields with material-facing
planning thresholds. These helpers keep every row explicit about whether it is
optical-only, a planning proxy, experimentally calibrated, or diagnostic-only.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd

from .tables import propagation_power_label

MATERIAL_OUTPUT_SCHEMA_VERSION = "1.0.0"

MATERIAL_MODEL_STATUS = {
    "optical_only",
    "planning_proxy",
    "experimentally_calibrated",
    "diagnostic_only",
}

MATERIAL_RESPONSE_MODELS = {
    "none",
    "fluence_threshold_proxy",
    "incubation_threshold_proxy",
    "line_fluence_proxy",
    "calibrated_ablation_model",
    "calibrated_refractive_index_change_model",
    "diagnostic_visualisation",
}

CALIBRATION_STATUS = {
    "uncalibrated",
    "literature_placeholder",
    "measured_threshold_only",
    "partially_calibrated",
    "fully_calibrated",
}

XZ_ENERGY_CONSERVATION_STATUS = {
    "energy_conserving_plane_fluence",
    "non_energy_conserving_line_proxy",
    "normalised_visualisation",
    "not_applicable",
}

MATERIAL_COLUMNS = [
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
    "material_name",
    "material_refractive_index",
    "material_model_status",
    "material_response_model",
    "calibration_status",
    "threshold_source",
    "threshold_fluence_J_cm2",
    "incubation_model",
    "incubation_coefficient",
    "pulse_count",
    "pulse_energy_uJ",
    "pulse_energy_at_sample_uJ",
    "transmission_fraction",
    "peak_fluence_J_cm2",
    "fluence_to_threshold_ratio",
    "thresholded_area_um2",
    "thresholded_equivalent_diameter_um",
    "xz_proxy_length_um",
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
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    if value is pd.NA:
        return False
    return str(value).strip().lower() not in {"", "none", "nan", "not_applicable"}


def _has_calibration_evidence(row: dict[str, Any]) -> bool:
    evidence_fields = (
        "calibration_dataset_id",
        "calibration_curve_id",
        "measured_threshold_fluence_J_cm2",
        "calibration_evidence",
    )
    return any(_is_present(row.get(field)) for field in evidence_fields)


def threshold_source_label(
    *,
    calibration_status: str = "uncalibrated",
    threshold_source: str | None = None,
) -> str:
    """Return a conservative threshold-source label for one material row."""

    if _is_present(threshold_source):
        return str(threshold_source)
    status = _coerce_known(calibration_status, CALIBRATION_STATUS, "calibration_status")
    if status == "fully_calibrated":
        return "calibrated_fit"
    if status == "partially_calibrated":
        return "partial_calibration"
    if status == "measured_threshold_only":
        return "measured_threshold"
    if status == "literature_placeholder":
        return "literature_placeholder"
    return "configured_placeholder"


def xz_proxy_label(
    *,
    xz_proxy_definition: str | None = None,
    xz_energy_conservation_status: str | None = None,
) -> str:
    """Return the XZ energy-conservation label implied by a proxy definition."""

    if _is_present(xz_energy_conservation_status):
        return _coerce_known(
            xz_energy_conservation_status,
            XZ_ENERGY_CONSERVATION_STATUS,
            "xz_energy_conservation_status",
        )
    text = str(xz_proxy_definition or "").strip().lower()
    if not text or text in {"none", "not_applicable", "na"}:
        return "not_applicable"
    if "line" in text:
        return "non_energy_conserving_line_proxy"
    if "visual" in text or "normal" in text:
        return "normalised_visualisation"
    if "plane" in text or "energy_conserving" in text:
        return "energy_conserving_plane_fluence"
    return "normalised_visualisation"


def material_model_label(
    *,
    material_response_model: str,
    calibration_status: str = "uncalibrated",
    calibration_evidence: bool = False,
    xz_energy_conservation_status: str = "not_applicable",
) -> str:
    """Return the honest material-model status for one row.

    Calibrated model labels are deliberately hard to obtain: they require a
    calibrated response model, ``fully_calibrated`` status, and evidence fields.
    Threshold-only and incubation-only rows remain planning proxies.
    """

    model = _coerce_known(material_response_model, MATERIAL_RESPONSE_MODELS, "material_response_model")
    status = _coerce_known(calibration_status, CALIBRATION_STATUS, "calibration_status")
    xz_status = _coerce_known(
        xz_energy_conservation_status,
        XZ_ENERGY_CONSERVATION_STATUS,
        "xz_energy_conservation_status",
    )
    if model == "none":
        return "optical_only"
    if model in {"line_fluence_proxy", "diagnostic_visualisation"}:
        return "diagnostic_only"
    if xz_status in {"non_energy_conserving_line_proxy", "normalised_visualisation"} and model == "line_fluence_proxy":
        return "diagnostic_only"
    if model in {"calibrated_ablation_model", "calibrated_refractive_index_change_model"}:
        if status != "fully_calibrated" or not bool(calibration_evidence):
            raise ValueError(
                "experimentally_calibrated material rows require fully_calibrated "
                "status and calibration evidence fields"
            )
        return "experimentally_calibrated"
    return "planning_proxy"


def annotate_material_row(
    row: dict[str, Any],
    *,
    run_id: str | None = None,
    qa_status: str = "exploratory",
) -> dict[str, Any]:
    """Stamp one materials-stage row with native schema metadata.

    Rows are born with metadata through this helper. It does not turn
    uncalibrated threshold comparisons into calibrated material predictions.
    """

    row.setdefault("run_id", _run_id(run_id))
    row.setdefault("generated_at_utc", _generated_at())
    row.setdefault("source_schema_version", MATERIAL_OUTPUT_SCHEMA_VERSION)
    row.setdefault("case_id", row.get("case_id", "material_case"))
    row.setdefault("preset", row.get("preset", "fast"))
    row.setdefault("path", row.get("path", "not_applicable"))
    row.setdefault("beam_family", row.get("beam_family", "material_proxy"))
    row.setdefault("model_level", row.get("model_level", "material_proxy"))
    row.setdefault("generation_method", row.get("generation_method", "not_applicable"))
    row.setdefault("hardware_status", row.get("hardware_status", "not_applicable"))
    row.setdefault("qa_status", qa_status)
    row.setdefault("material_name", row.get("material", "not_specified"))
    row.setdefault("material_refractive_index", row.get("refractive_index", pd.NA))
    row.setdefault("calibration_status", "uncalibrated")
    row["calibration_status"] = _coerce_known(row.get("calibration_status"), CALIBRATION_STATUS, "calibration_status")
    row.setdefault("material_response_model", "fluence_threshold_proxy")
    row["material_response_model"] = _coerce_known(
        row.get("material_response_model"),
        MATERIAL_RESPONSE_MODELS,
        "material_response_model",
    )
    row.setdefault(
        "threshold_source",
        threshold_source_label(
            calibration_status=str(row["calibration_status"]),
            threshold_source=row.get("threshold_source"),
        ),
    )
    row.setdefault("incubation_model", "power_law_proxy")
    row.setdefault(
        "xz_energy_conservation_status",
        xz_proxy_label(
            xz_proxy_definition=row.get("xz_proxy_definition"),
            xz_energy_conservation_status=row.get("xz_energy_conservation_status"),
        ),
    )
    row["xz_energy_conservation_status"] = _coerce_known(
        row.get("xz_energy_conservation_status"),
        XZ_ENERGY_CONSERVATION_STATUS,
        "xz_energy_conservation_status",
    )

    inferred = material_model_label(
        material_response_model=str(row["material_response_model"]),
        calibration_status=str(row["calibration_status"]),
        calibration_evidence=_has_calibration_evidence(row),
        xz_energy_conservation_status=str(row["xz_energy_conservation_status"]),
    )
    existing_status = row.get("material_model_status")
    if _is_present(existing_status):
        row["material_model_status"] = _coerce_known(
            existing_status,
            MATERIAL_MODEL_STATUS,
            "material_model_status",
        )
        if row["material_model_status"] == "experimentally_calibrated" and inferred != "experimentally_calibrated":
            raise ValueError("experimentally_calibrated status requires calibration evidence")
        if row["material_model_status"] == "planning_proxy" and inferred == "diagnostic_only":
            raise ValueError("line-fluence diagnostic rows cannot be labelled planning_proxy")
    else:
        row["material_model_status"] = inferred

    if "fluence_to_threshold_ratio" not in row:
        peak = row.get("peak_fluence_J_cm2")
        threshold = row.get("threshold_fluence_J_cm2")
        try:
            row["fluence_to_threshold_ratio"] = float(peak) / float(threshold)
        except (TypeError, ValueError, ZeroDivisionError):
            row["fluence_to_threshold_ratio"] = pd.NA

    if "propagation_power_label" not in row and "propagation_power_drift_fraction" in row:
        try:
            row["propagation_power_label"] = propagation_power_label(float(row["propagation_power_drift_fraction"]))
        except (TypeError, ValueError):
            row["propagation_power_label"] = "unknown"
    return row


def ordered_material_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return one row ordered with canonical material columns first."""

    ordered: dict[str, Any] = {key: row.get(key, pd.NA) for key in MATERIAL_COLUMNS}
    extra = {key: value for key, value in row.items() if key not in MATERIAL_COLUMNS}
    return {**ordered, **extra}


def ordered_material_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Return a DataFrame with canonical material columns first."""

    df = pd.DataFrame([ordered_material_row(dict(row)) for row in rows])
    if df.empty:
        return df
    extra = [col for col in df.columns if col not in set(MATERIAL_COLUMNS)]
    return df[MATERIAL_COLUMNS + extra].copy()


def with_material_metadata(
    df: pd.DataFrame,
    *,
    run_id: str | None = None,
    qa_status: str = "exploratory",
) -> pd.DataFrame:
    """Stamp all rows in ``df`` with native Stage 6 materials metadata."""

    generated_at = _generated_at()
    effective_run_id = _run_id(run_id)
    rows = []
    for row in df.to_dict("records"):
        row["generated_at_utc"] = generated_at
        annotate_material_row(row, run_id=effective_run_id, qa_status=qa_status)
        rows.append(row)
    return ordered_material_frame(rows)


__all__ = [
    "CALIBRATION_STATUS",
    "MATERIAL_COLUMNS",
    "MATERIAL_MODEL_STATUS",
    "MATERIAL_OUTPUT_SCHEMA_VERSION",
    "MATERIAL_RESPONSE_MODELS",
    "XZ_ENERGY_CONSERVATION_STATUS",
    "annotate_material_row",
    "material_model_label",
    "ordered_material_frame",
    "ordered_material_row",
    "threshold_source_label",
    "with_material_metadata",
    "xz_proxy_label",
]
