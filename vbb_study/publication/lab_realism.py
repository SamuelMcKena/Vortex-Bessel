"""Canonical lab-realism CSV metadata and route terminology.

This module keeps Phase 8 lab-realism output rows honest about where a
quantity lives in the optical train and whether a hardware route is currently
lab-realizable, future hardware, simulation-only, or diagnostic-only.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd

from .tables import propagation_power_label

LAB_REALISM_SCHEMA_VERSION = "1.0.0"

HARDWARE_STATUS = {
    "current_lab_realizable",
    "future_hardware_required",
    "simulation_only",
    "diagnostic_only",
}

GENERATION_METHODS = {
    "holographic_axicon",
    "physical_axicon",
    "slm_phase_only",
    "objective_pupil_limited",
    "interface_corrected_numerical",
    "interface_uncorrected",
}

MODEL_LEVELS = {
    "ideal_target",
    "numerical_propagation",
    "lab_realistic",
    "hardware_route",
    "interface_model",
}

PLANE_LABELS = {
    "slm_plane",
    "fourier_filter_plane",
    "objective_pupil_plane",
    "surface_plane",
    "sample_plane",
    "in_medium_plane",
    "propagation_axis_z",
}

LAB_REALISM_COLUMNS = [
    "run_id",
    "generated_at_utc",
    "source_schema_version",
    "case_id",
    "preset",
    "path",
    "generation_method",
    "model_level",
    "hardware_status",
    "plane_label",
    "coordinate_frame",
    "objective_NA",
    "objective_f_eff_mm",
    "pupil_radius_mm",
    "first_order_selected_fraction",
    "pupil_clipped_fraction",
    "requested_vortex_charge",
    "measured_winding",
    "winding_error",
    "winding_pass",
    "slm2_conjugate_mode",
    "vortex_removal_acknowledged",
    "propagation_power_drift_fraction",
    "propagation_power_label",
    "quantitative_metrics_valid",
    "quantitative_metrics_invalid_reason",
    "canonical_zone_um",
    "strict_bessel_region_um",
    "qa_status",
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


def annotate_lab_realism_row(
    row: dict[str, Any],
    *,
    generation_method: str,
    model_level: str,
    hardware_status: str,
    plane_label: str,
    coordinate_frame: str,
    run_id: str | None = None,
    preset: str | None = None,
    path: str | None = None,
    qa_status: str = "exploratory",
) -> dict[str, Any]:
    """Stamp one lab-realism CSV row with native Phase 8 metadata."""

    row.setdefault("run_id", _run_id(run_id))
    row.setdefault("generated_at_utc", _generated_at())
    row.setdefault("source_schema_version", LAB_REALISM_SCHEMA_VERSION)
    row.setdefault("preset", preset or row.get("preset", "fast"))
    row.setdefault("path", path or row.get("path", "not_applicable"))
    row["generation_method"] = _coerce_known(generation_method, GENERATION_METHODS, "generation_method")
    row["model_level"] = _coerce_known(model_level, MODEL_LEVELS, "model_level")
    row["hardware_status"] = _coerce_known(hardware_status, HARDWARE_STATUS, "hardware_status")
    row["plane_label"] = _coerce_known(plane_label, PLANE_LABELS, "plane_label")
    row.setdefault("coordinate_frame", coordinate_frame)
    row.setdefault("qa_status", qa_status)
    if "propagation_power_label" not in row and "propagation_power_drift_fraction" in row:
        try:
            row["propagation_power_label"] = propagation_power_label(float(row["propagation_power_drift_fraction"]))
        except (TypeError, ValueError):
            row["propagation_power_label"] = "unknown"
    return row


def ordered_lab_realism_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Return a DataFrame with canonical lab-realism columns first."""

    df = pd.DataFrame(list(rows))
    if df.empty:
        return df
    canonical = [col for col in LAB_REALISM_COLUMNS if col in df.columns]
    extra = [col for col in df.columns if col not in set(LAB_REALISM_COLUMNS)]
    return df[canonical + extra].copy()


def with_lab_realism_metadata(
    df: pd.DataFrame,
    *,
    generation_method: str,
    model_level: str,
    hardware_status: str,
    plane_label: str,
    coordinate_frame: str,
    run_id: str | None = None,
    preset: str | None = None,
    path: str | None = None,
    qa_status: str = "exploratory",
) -> pd.DataFrame:
    """Stamp all rows in ``df`` with native Phase 8 lab-realism metadata."""

    rows = []
    generated_at = _generated_at()
    effective_run_id = _run_id(run_id)
    for row in df.to_dict("records"):
        row["generated_at_utc"] = generated_at
        annotate_lab_realism_row(
            row,
            generation_method=generation_method,
            model_level=model_level,
            hardware_status=hardware_status,
            plane_label=plane_label,
            coordinate_frame=coordinate_frame,
            run_id=effective_run_id,
            preset=preset,
            path=path,
            qa_status=qa_status,
        )
        rows.append(row)
    return ordered_lab_realism_frame(rows)
