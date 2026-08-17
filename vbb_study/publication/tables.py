"""Canonical scalar CSV schema for the structured-beam simulation atlas.

This module is the SINGLE SOURCE OF TRUTH for:

* The current scalar output schema version (``SCALAR_OUTPUT_SCHEMA_VERSION``).
* The canonical column list for all scalar summary CSVs
  (``SCALAR_SUMMARY_COLUMNS``).
* The propagation power QA label function (``propagation_power_label``).
* A helper that annotates a metrics row-dict with required run/schema metadata
  (``annotate_scalar_row``).

Why one place?
--------------
When the column list or naming convention changes, updating this file updates
all notebooks and CSVs that import from it — without hunting for duplicated
column lists in notebook cells.

Naming conventions (cross-reference ``docs/01_conventions.md``):
  canonical_zone_um        — axial peak FWHM zone (``bessel_zone_um``).
  strict_bessel_region_um  — triple-intersection fabrication region.
                             MUST NOT be confused with canonical zone.
  vortex_main_ring_*       — ell>0 bright ring; NOT target core diameter.
  target_equivalent_l0_*   — always the J_0 first-zero interpretation.
  propagation_power_label  — pass / marginal / fail at 5 % / 20 % thresholds.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

# Bump this when the column list or naming convention changes.
SCALAR_OUTPUT_SCHEMA_VERSION = "2.0.0"

# ---------------------------------------------------------------------------
# Canonical column list
# ---------------------------------------------------------------------------

# Every scalar summary CSV should contain these columns where applicable.
# Columns that cannot be computed for a given case should be NaN / "".
SCALAR_SUMMARY_COLUMNS: list[str] = [
    # ---- Provenance ----
    "run_id",
    "generated_at_utc",
    "source_schema_version",
    # ---- Case identity ----
    "preset",
    "path",
    "case_id",
    # ---- Target scale definitions ----
    "target_scale_definition",
    "target_core_diameter_um",
    "target_equivalent_l0_core_diameter_um",
    # ---- J_0 first-zero equivalents ----
    "equivalent_l0_first_zero_radius_um",
    "equivalent_l0_first_zero_diameter_um",
    # ---- Vortex ring (ell > 0) ----
    "vortex_main_ring_radius_um",
    "vortex_main_ring_diameter_um",
    # ---- Core radii ----
    "core_radius_definition",
    "core_hwhm_radius_um",
    "core_hwhm_diameter_um",
    "core_first_zero_radius_um",
    "core_first_zero_diameter_um",
    # ---- Feature size (reported metric for fabrication) ----
    "feature_radius_um",
    "feature_diameter_um",
    # ---- Canonical axial zone (FWHM) ----
    "canonical_zone_um",
    "canonical_zone_start_um",
    "canonical_zone_end_um",
    # ---- Strict Bessel region (triple intersection) ----
    "strict_bessel_region_um",
    "strict_bessel_region_start_um",
    "strict_bessel_region_end_um",
    "bessel_region_definition",
    # ---- Propagation power QA ----
    "propagation_power_drift_fraction",
    "propagation_power_label",
    # ---- Study QA ----
    "qa_status",
]

# Description for each column — useful for generating data dictionaries.
SCALAR_COLUMN_DESCRIPTIONS: dict[str, str] = {
    "run_id": "Run identifier from the study runner (YYYYMMDDTHHMMSSz format).",
    "generated_at_utc": "ISO-8601 UTC timestamp when this row was written.",
    "source_schema_version": "Version of SCALAR_OUTPUT_SCHEMA_VERSION when this row was written.",
    "preset": "Grid/speed preset name (fast, standard, publication, validation).",
    "path": "Propagation path type (ideal, realistic).",
    "case_id": "Unique identifier for this simulation case.",
    "target_scale_definition": "How target_core_diameter was specified (equivalent_l0_first_zero_diameter).",
    "target_core_diameter_um": "Input target core diameter in microns (as specified by the caller).",
    "target_equivalent_l0_core_diameter_um": "Target diameter re-expressed as an equivalent ell=0 J_0 first-zero diameter.",
    "equivalent_l0_first_zero_radius_um": "J_0 first-zero radius at the design k_r, in microns.",
    "equivalent_l0_first_zero_diameter_um": "J_0 first-zero diameter at the design k_r, in microns.",
    "vortex_main_ring_radius_um": "Radius of the first bright ring of the vortex Bessel beam (J'_ell = 0). "
                                   "For ell=0, equals the J_0 first-zero radius.",
    "vortex_main_ring_diameter_um": "Diameter of the vortex ring (2 x ring radius).",
    "core_radius_definition": "String describing which definition was used for core_radius_um.",
    "core_hwhm_radius_um": "Radial HWHM of the on-axis peak in the propagated field.",
    "core_hwhm_diameter_um": "Full-width at half-maximum diameter.",
    "core_first_zero_radius_um": "Radius to the first zero of the propagated radial profile.",
    "core_first_zero_diameter_um": "Diameter to the first zero.",
    "feature_radius_um": "Fabrication-relevant feature radius (definition in core_radius_definition).",
    "feature_diameter_um": "Fabrication-relevant feature diameter.",
    "canonical_zone_um": "Canonical axial zone: FWHM of the on-axis peak (bessel_zone_um). "
                          "DO NOT confuse with strict_bessel_region_um.",
    "canonical_zone_start_um": "Axial start of the canonical zone.",
    "canonical_zone_end_um": "Axial end of the canonical zone.",
    "strict_bessel_region_um": "Strict fabrication-planning region: intersection of peak >= 50%, "
                                "ring power >= 50%, and ring radius within 15% tolerance. "
                                "Always <= canonical_zone_um.",
    "strict_bessel_region_start_um": "Axial start of the strict Bessel region.",
    "strict_bessel_region_end_um": "Axial end of the strict Bessel region.",
    "bessel_region_definition": "String identifying the strict region definition (strict_intersection_peak_power_radius).",
    "propagation_power_drift_fraction": "Fraction of input power lost during propagation (BL-ASM bandlimit + SAS retention). "
                                         "0 = perfect conservation; 1 = all power lost.",
    "propagation_power_label": "pass (drift <= 5%), marginal (5-20%), or fail (>20%).",
    "qa_status": "Study QA status from vbb_study.study_taxonomy.QA_STATUS.",
}

# ---------------------------------------------------------------------------
# Power QA
# ---------------------------------------------------------------------------


def propagation_power_label(drift_fraction: float) -> str:
    """Return the power QA label for a given drift fraction.

    Parameters
    ----------
    drift_fraction:
        Fraction of input power lost to BL-ASM bandlimit clipping or SAS
        retention losses during propagation.  Computed as
        ``1 - min(retained_power_fraction)``.

    Returns
    -------
    str
        One of ``"pass"``, ``"marginal"``, ``"fail"``, or ``"unknown"``
        if drift is NaN/infinite.

    Thresholds
    ----------
    pass:     drift <= 0.05 (< 5 % power loss)
    marginal: 0.05 < drift <= 0.20 (5–20 %)
    fail:     drift > 0.20 (> 20 %)
    """

    if not math.isfinite(float(drift_fraction)):
        return "unknown"
    d = float(drift_fraction)
    if d <= 0.05:
        return "pass"
    if d <= 0.20:
        return "marginal"
    return "fail"


# ---------------------------------------------------------------------------
# Row annotation helper
# ---------------------------------------------------------------------------


def annotate_scalar_row(
    row: dict[str, Any],
    *,
    run_id: str | None = None,
    qa_status: str = "exploratory",
) -> dict[str, Any]:
    """Add required provenance and schema metadata to a scalar metrics row.

    This is the canonical way to stamp a row dict before writing it to a
    scalar summary CSV.  Call it after computing all physics metrics.

    run_id resolution order:
      1. Explicit ``run_id`` argument.
      2. ``STRUCTURED_BEAM_RUN_ID`` environment variable (set by run_study.py
         before launching notebook kernels).
      3. A fresh UTC timestamp (always non-empty — never writes NaN to CSV).

    Parameters
    ----------
    row:
        Existing row dict from ``extract_vortex_safe_metrics`` or similar.
        Modified in-place and returned.
    run_id:
        Runner run ID, or None to fall back to the env var / timestamp.
    qa_status:
        QA status label from ``study_taxonomy.QA_STATUS``.

    Returns
    -------
    dict
        The same ``row`` with added provenance fields.
    """

    effective_run_id = (
        run_id
        or os.environ.get("STRUCTURED_BEAM_RUN_ID")
        or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    row.setdefault("run_id", effective_run_id)
    row.setdefault("generated_at_utc", datetime.now(timezone.utc).isoformat())
    row.setdefault("source_schema_version", SCALAR_OUTPUT_SCHEMA_VERSION)
    row.setdefault("qa_status", qa_status)
    # If power label not already set by extract_vortex_safe_metrics, derive it.
    if "propagation_power_label" not in row and "propagation_power_drift_fraction" in row:
        row["propagation_power_label"] = propagation_power_label(
            float(row["propagation_power_drift_fraction"])
        )
    return row


def ordered_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict with canonical columns first, extra columns appended.

    Ensures that when a DataFrame or CSV is written the columns appear in the
    canonical order defined by ``SCALAR_SUMMARY_COLUMNS``.

    Parameters
    ----------
    row:
        Row dict, typically the output of ``annotate_scalar_row``.
    """

    ordered: dict[str, Any] = {}
    for col in SCALAR_SUMMARY_COLUMNS:
        ordered[col] = row.get(col, float("nan"))
    # Append any extra columns not in the canonical list.
    for key, value in row.items():
        if key not in ordered:
            ordered[key] = value
    return ordered
