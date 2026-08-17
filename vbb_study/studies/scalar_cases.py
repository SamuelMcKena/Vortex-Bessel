"""Scalar Bessel-Gauss study cases for the structured-beam atlas.

This module owns the shortlist presets and helpers for regenerating scalar
summary CSVs.  It is the recommended entry point for:

* Defining which cases belong to the canonical shortlist sweep.
* Running those cases through the scalar physics engine.
* Annotating results with the canonical schema metadata from
  ``vbb_study.publication.tables``.
* Writing output CSVs in the correct column order.

Usage from a notebook or script::

    from vbb_study.studies import scalar_cases
    df = scalar_cases.run_shortlist(preset="fast", path="realistic")
    scalar_cases.save_shortlist_csv(df, paths["csv"])

What this module does NOT own:

* The physics propagation logic (``bessel_twin_core``).
* The SLM hologram model (``bessel_twin_core``).
* Visualisation / plotting (``vbb_study.vbb_viz``, ``vbb_study.vbb_style``).
* Material proxy maps (``vbb_study.vbb_materials``).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vbb_study import study_taxonomy
from vbb_study.config import PathKind, um
from vbb_study.design import default_config
from vbb_study.facade import core as _bt
from vbb_study.publication.tables import (
    SCALAR_OUTPUT_SCHEMA_VERSION,
    SCALAR_SUMMARY_COLUMNS,
    annotate_scalar_row,
    ordered_row,
    propagation_power_label,
)

# ---------------------------------------------------------------------------
# Canonical shortlist cases
# ---------------------------------------------------------------------------
# These match the shortlist historically used in `publication_diagnostics.py`.
# The shortlist is the minimum reproducible set for scalar paper figures.

DEFAULT_SHORTLIST: list[dict[str, Any]] = [
    {"ell": 0, "core_um": 3.0, "length_um": 150},
    {"ell": 1, "core_um": 3.0, "length_um": 150},
    {"ell": 2, "core_um": 3.0, "length_um": 150},
    {"ell": 3, "core_um": 3.0, "length_um": 150},
    {"ell": 0, "core_um": 5.0, "length_um": 300},
    {"ell": 1, "core_um": 5.0, "length_um": 300},
    {"ell": 0, "core_um": 2.0, "length_um": 100},
]


def _case_id(ell: int, core_um: float, length_um: float) -> str:
    return f"ell{ell}_core{core_um:.0f}_L{length_um:.0f}"


# ---------------------------------------------------------------------------
# Single case runner
# ---------------------------------------------------------------------------


def run_scalar_case(
    *,
    ell: int,
    core_um: float,
    length_um: float,
    preset: str = "fast",
    path: PathKind = "realistic",
    run_id: str | None = None,
    qa_status: str = "exploratory",
) -> dict[str, Any]:
    """Run one scalar case and return an annotated metrics row.

    The returned dict has all columns from ``SCALAR_SUMMARY_COLUMNS`` plus
    any additional columns produced by the physics engine.  It is ready to
    be appended to a DataFrame or written directly to a CSV row.

    Parameters
    ----------
    ell:
        Topological charge.
    core_um:
        Target equivalent-J_0-first-zero core diameter in microns.
    length_um:
        Target non-diffracting Bessel zone length in microns.
    preset:
        Grid preset name (``"fast"``, ``"standard"``, ``"publication"``).
    path:
        Propagation path (``"realistic"`` or ``"ideal"``).
    run_id:
        Run identifier for provenance.  If None, omitted.
    qa_status:
        QA status label from ``study_taxonomy.QA_STATUS``.
    """

    case_id = _case_id(ell, core_um, length_um)
    cfg = default_config(preset)
    cfg = replace(
        cfg,
        target=replace(
            cfg.target,
            ell=ell,
            target_core_diameter_m=core_um * um,
            target_bessel_length_m=length_um * um,
        ),
    )
    result = _bt().run_case(cfg, preset=preset, path=path, case_id=case_id)
    metrics = dict(result["metrics"])
    # Stamp with schema metadata and required provenance fields.
    metrics["preset"] = preset
    metrics["path"] = path
    metrics["case_id"] = case_id
    annotate_scalar_row(metrics, run_id=run_id, qa_status=qa_status)
    return ordered_row(metrics)


# ---------------------------------------------------------------------------
# Shortlist runner
# ---------------------------------------------------------------------------


def run_shortlist(
    cases: list[dict[str, Any]] | None = None,
    *,
    preset: str = "fast",
    path: PathKind = "realistic",
    run_id: str | None = None,
    qa_status: str = "exploratory",
    continue_on_error: bool = False,
) -> pd.DataFrame:
    """Run a list of scalar cases and return a tidy summary DataFrame.

    Parameters
    ----------
    cases:
        List of case dicts with ``ell``, ``core_um``, ``length_um`` keys.
        Defaults to ``DEFAULT_SHORTLIST``.
    continue_on_error:
        If True, failed cases are recorded with NaN metrics instead of
        raising.  If False (default), the first failure re-raises.

    Returns
    -------
    pd.DataFrame
        One row per case, columns in canonical order (``SCALAR_SUMMARY_COLUMNS``
        first, then engine-specific extras).
    """

    cases = cases if cases is not None else DEFAULT_SHORTLIST
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows: list[dict[str, Any]] = []
    for spec in cases:
        try:
            row = run_scalar_case(
                ell=int(spec["ell"]),
                core_um=float(spec["core_um"]),
                length_um=float(spec["length_um"]),
                preset=preset,
                path=path,
                run_id=run_id,
                qa_status=qa_status,
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            case_id = _case_id(int(spec["ell"]), float(spec["core_um"]), float(spec["length_um"]))
            row = ordered_row(
                annotate_scalar_row(
                    {"case_id": case_id, "preset": preset, "path": path, "error": str(exc)},
                    run_id=run_id,
                    qa_status="out_of_validity",
                )
            )
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------


def save_shortlist_csv(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    filename: str = "scalar_shortlist_summary.csv",
) -> Path:
    """Write the shortlist DataFrame to a CSV in the canonical column order.

    Canonical columns (``SCALAR_SUMMARY_COLUMNS``) appear first; additional
    engine columns follow.
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename

    # Build ordered column list: canonical first, rest in original order.
    canonical = [c for c in SCALAR_SUMMARY_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in set(SCALAR_SUMMARY_COLUMNS)]
    df[canonical + extra].to_csv(path, index=False)
    return path
