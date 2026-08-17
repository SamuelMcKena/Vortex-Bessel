"""Canonical vector-stage CSV metadata and hardware-feasibility labels."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd

from .tables import propagation_power_label

VECTOR_OUTPUT_SCHEMA_VERSION = "1.0.0"

VECTOR_MODES = {
    "scalar_reference",
    "radial",
    "azimuthal",
    "hybrid",
    "paper_replica",
    "sop_encoded_case1",
    "diagnostic",
}

VECTOR_MODELS = {
    "ideal_jones_target",
    "scalar_sas_with_jones_overlay",
    "paper_replica_baliyan_nishchal",
    "current_lab_case1_sop_encoded",
    "future_true_vector_route",
    "diagnostic_only",
}

HARDWARE_STATUS = {
    "current_lab_realizable",
    "future_hardware_required",
    "simulation_only",
    "diagnostic_only",
}

GENERATION_METHODS = {
    "scalar_reference",
    "two_slm_same_axis_sop",
    "qplate_or_vector_converter",
    "interferometric_vector_combiner",
    "paper_replica_simulation",
    "simulation_only",
}

MODEL_LEVELS = {
    "ideal_target",
    "paper_replica",
    "current_lab_approximation",
    "future_hardware_route",
    "diagnostic",
    "scalar_reference",
}

VECTOR_COLUMNS = [
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
    "vector_mode",
    "vector_model",
    "vector_program",
    "vector_method",
    "vector_encoder_hardware",
    "lab_realizable",
    "simulation_only",
    "requires_element",
    "uses_waveplates",
    "uses_two_slm",
    "uses_shared_director_axis",
    "encoded_power_fraction",
    "scalar_reference_case_id",
    "ell",
    "target_core_diameter_um",
    "vortex_main_ring_diameter_um",
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


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return bool(default)
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"true", "1", "yes", "y"}:
            return True
        if key in {"false", "0", "no", "n", ""}:
            return False
    return bool(value)


def vector_hardware_status(
    *,
    vector_model: str,
    generation_method: str,
    lab_realizable: bool,
    simulation_only: bool,
    requires_element: str | None = None,
) -> str:
    """Return the honest hardware status for one vector route."""

    if simulation_only:
        return "simulation_only"
    if lab_realizable:
        return "current_lab_realizable"
    if str(vector_model) == "diagnostic_only" or str(generation_method) == "simulation_only":
        return "diagnostic_only"
    if requires_element and str(requires_element).strip().lower() not in {"", "none", "not_applicable"}:
        return "future_hardware_required"
    return "future_hardware_required"


def vector_lab_realizability_label(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize lab-realizability booleans and derive ``hardware_status``."""

    lab_realizable = _coerce_bool(row.get("lab_realizable"), default=False)
    simulation_only = _coerce_bool(row.get("simulation_only"), default=False)
    if lab_realizable and simulation_only:
        raise ValueError("A vector row cannot be both lab_realizable and simulation_only.")
    row["lab_realizable"] = lab_realizable
    row["simulation_only"] = simulation_only
    row["requires_element"] = row.get("requires_element") or "none"
    row["hardware_status"] = row.get("hardware_status") or vector_hardware_status(
        vector_model=str(row.get("vector_model", "")),
        generation_method=str(row.get("generation_method", "")),
        lab_realizable=lab_realizable,
        simulation_only=simulation_only,
        requires_element=str(row.get("requires_element", "none")),
    )
    return row


def annotate_vector_row(
    row: dict[str, Any],
    *,
    run_id: str | None = None,
    qa_status: str = "exploratory",
) -> dict[str, Any]:
    """Stamp one vector-stage row with native Phase 9 metadata."""

    row.setdefault("run_id", _run_id(run_id))
    row.setdefault("generated_at_utc", _generated_at())
    row.setdefault("source_schema_version", VECTOR_OUTPUT_SCHEMA_VERSION)
    row.setdefault("preset", row.get("preset", "fast"))
    row.setdefault("path", row.get("path", "not_applicable"))
    row.setdefault("beam_family", "vector")
    row.setdefault("qa_status", qa_status)
    row.setdefault("vector_program", row.get("vector_mode", "diagnostic"))
    row.setdefault("vector_method", "not_applicable")
    row.setdefault("vector_encoder_hardware", "not_applicable")
    row.setdefault("requires_element", "none")
    row.setdefault("uses_waveplates", False)
    row.setdefault("uses_two_slm", False)
    row.setdefault("uses_shared_director_axis", False)
    row.setdefault("encoded_power_fraction", pd.NA)
    row.setdefault("scalar_reference_case_id", pd.NA)

    row["vector_mode"] = _coerce_known(row.get("vector_mode"), VECTOR_MODES, "vector_mode")
    row["vector_model"] = _coerce_known(row.get("vector_model"), VECTOR_MODELS, "vector_model")
    row["model_level"] = _coerce_known(row.get("model_level"), MODEL_LEVELS, "model_level")
    row["generation_method"] = _coerce_known(row.get("generation_method"), GENERATION_METHODS, "generation_method")
    vector_lab_realizability_label(row)
    row["hardware_status"] = _coerce_known(row.get("hardware_status"), HARDWARE_STATUS, "hardware_status")

    for key in ("lab_realizable", "simulation_only", "uses_waveplates", "uses_two_slm", "uses_shared_director_axis"):
        row[key] = _coerce_bool(row.get(key), default=False)

    if "propagation_power_label" not in row and "propagation_power_drift_fraction" in row:
        try:
            row["propagation_power_label"] = propagation_power_label(float(row["propagation_power_drift_fraction"]))
        except (TypeError, ValueError):
            row["propagation_power_label"] = "unknown"
    return row


def ordered_vector_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return one row ordered with canonical vector columns first."""

    canonical = {key: row[key] for key in VECTOR_COLUMNS if key in row}
    extra = {key: value for key, value in row.items() if key not in VECTOR_COLUMNS}
    return {**canonical, **extra}


def ordered_vector_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Return a DataFrame with canonical vector columns first."""

    ordered = [ordered_vector_row(dict(row)) for row in rows]
    df = pd.DataFrame(ordered)
    if df.empty:
        return df
    canonical = [col for col in VECTOR_COLUMNS if col in df.columns]
    extra = [col for col in df.columns if col not in set(VECTOR_COLUMNS)]
    return df[canonical + extra].copy()


def with_vector_metadata(
    df: pd.DataFrame,
    *,
    run_id: str | None = None,
    qa_status: str = "exploratory",
) -> pd.DataFrame:
    """Stamp all rows in ``df`` with native Phase 9 vector metadata."""

    generated_at = _generated_at()
    effective_run_id = _run_id(run_id)
    rows = []
    for row in df.to_dict("records"):
        row["generated_at_utc"] = generated_at
        annotate_vector_row(row, run_id=effective_run_id, qa_status=qa_status)
        rows.append(row)
    return ordered_vector_frame(rows)


__all__ = [
    "GENERATION_METHODS",
    "HARDWARE_STATUS",
    "MODEL_LEVELS",
    "VECTOR_COLUMNS",
    "VECTOR_MODELS",
    "VECTOR_MODES",
    "VECTOR_OUTPUT_SCHEMA_VERSION",
    "annotate_vector_row",
    "ordered_vector_frame",
    "ordered_vector_row",
    "vector_hardware_status",
    "vector_lab_realizability_label",
    "with_vector_metadata",
]
