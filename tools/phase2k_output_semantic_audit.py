"""Semantic Phase 2K disposition for every checked-in output artifact.

This is the second audit layer after ``phase2k_output_truth_audit.py``.  It does
not decide that a result is correct from its filename.  Instead it records
(1) provenance/family, (2) explicit validity failures already stored in the
artifact, (3) propagation-drift evidence, (4) calibration/claim blockers and
(5) the conservative scientific disposition that must be satisfied before a
figure or table can be selected for a thesis/presentation.

All pre-Phase-2K numerical outputs remain non-authoritative until regenerated
from a producer that has passed the mathematical and numerical truth gates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


FALSE_FLAG_COLUMNS = (
    "phase1r_quantitative_valid",
    "quantitative_metrics_valid",
    "acceptance_pass",
    "first_order_geometry_valid",
    "validity_valid",
    "quantitative_valid",
    "winding_pass",
)

DRIFT_NAME_TOKENS = (
    "propagation_power_drift_fraction",
    "power_drift_fraction",
    "plane_power_drift_fraction",
)

GENERATED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".pdf", ".svg", ".gif", ".mp4",
    ".csv", ".json", ".jsonl", ".txt", ".npy", ".npz", ".html",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "pass", "passed", "valid"}:
        return True
    if text in {"false", "0", "no", "fail", "failed", "invalid"}:
        return False
    return None


def _csv_diagnostics(path: Path) -> dict[str, Any]:
    result = {
        "tabular_rows": None,
        "explicit_false_rows": 0,
        "false_flag_columns": "",
        "max_propagation_power_drift_fraction": None,
        "rows_over_5pct_power_drift": 0,
    }
    try:
        frame = pd.read_csv(path)
    except Exception:
        return result
    result["tabular_rows"] = int(len(frame))
    false_rows: set[int] = set()
    false_columns: list[str] = []
    for name in FALSE_FLAG_COLUMNS:
        if name not in frame.columns:
            continue
        false_columns.append(name)
        for index, value in frame[name].items():
            parsed = _as_bool(value)
            if parsed is False:
                false_rows.add(int(index))
    result["explicit_false_rows"] = int(len(false_rows))
    result["false_flag_columns"] = ";".join(false_columns)

    drift_columns = [
        name for name in frame.columns
        if any(token == str(name).lower() or token in str(name).lower() for token in DRIFT_NAME_TOKENS)
    ]
    drift_values: list[float] = []
    drift_bad_rows: set[int] = set()
    for name in drift_columns:
        series = pd.to_numeric(frame[name], errors="coerce")
        for index, value in series.items():
            if pd.isna(value):
                continue
            v = float(value)
            if math.isfinite(v):
                drift_values.append(v)
                if v > 0.05:
                    drift_bad_rows.add(int(index))
    if drift_values:
        result["max_propagation_power_drift_fraction"] = float(max(drift_values))
        result["rows_over_5pct_power_drift"] = int(len(drift_bad_rows))
    return result


def _walk_json(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, child
            yield from _walk_json(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_json(child, f"{prefix}[{index}]")


def _json_diagnostics(path: Path) -> dict[str, Any]:
    out = {
        "json_blockers": "",
        "json_report_authorised_false": False,
        "json_experimentally_validated_false": False,
        "json_absolute_claims_locked": False,
    }
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return out
    blockers: list[str] = []
    for key, child in _walk_json(value):
        leaf = key.split(".")[-1].split("[")[0].lower()
        parsed = _as_bool(child)
        if leaf in {"report_figures_authorised", "report_figures_authorized"} and parsed is False:
            out["json_report_authorised_false"] = True
            blockers.append(f"{key}=false")
        if leaf == "experimentally_validated_prediction_present" and parsed is False:
            out["json_experimentally_validated_false"] = True
            blockers.append(f"{key}=false")
        if leaf in {
            "absolute_dimensions_unlocked",
            "absolute_fluence_unlocked",
            "absolute_sample_dimensions_ready",
            "absolute_focal_fluence_ready",
        } and parsed is False:
            out["json_absolute_claims_locked"] = True
            blockers.append(f"{key}=false")
        if leaf in {
            "convergence_all_predeclared_gates_pass",
            "grid_resolution_gates_pass",
            "z_sampling_convergence_pass",
            "z_step_convergence_pass",
        } and parsed is False:
            blockers.append(f"{key}=false")
    out["json_blockers"] = ";".join(sorted(set(blockers)))
    return out


def _base_disposition(rel: Path) -> tuple[str, str, str]:
    """Return (status, claim_scope, reason) before data-level overrides."""

    p = rel.as_posix().lower()
    suffix = rel.suffix.lower()

    if "/presentation_phase2" in p:
        return (
            "PRESENTATION_DERIVATIVE",
            "none",
            "rendered presentation derivative; source arrays/producer must be validated first",
        )
    if "/publication_study/" in p or "legacy" in p:
        return (
            "LEGACY_OR_SUPERSEDED",
            "historical_diagnostic_only",
            "legacy study predates current Phase 2K mathematical/reference corrections",
        )
    if p.startswith("outputs/holograms/"):
        return (
            "UNCALIBRATED_HARDWARE_COMMAND",
            "planning_only",
            "SLM command/mask cannot be hardware truth without confirmed panel/LUT/stroke/orientation",
        )
    if "nominal_4f_candidate_runs" in p:
        return (
            "NOMINAL_4F_DIAGNOSTIC",
            "calibration_blocked_numerical_model",
            "useful 4F/order diagnostic but focal length, iris and SLM calibration are unresolved",
        )
    if "phase2e_report_visualisation" in p:
        return (
            "STALE_NUMERICAL_VISUAL",
            "diagnostic_only",
            "Phase 2E finite-propagation report visuals require regeneration after convergence/math audit",
        )
    if "/phase2c/" in p:
        return (
            "NUMERICAL_SOLVER_BENCHMARK_REGENERATE",
            "calibration_blocked_numerical_model",
            "independent scalar/vector solver benchmark is valuable but pre-Phase-2K outputs must be regenerated",
        )
    if "phase2b_visual_diagnostics" in p:
        return (
            "NUMERICAL_DIAGNOSTIC_REGENERATE",
            "calibration_blocked_numerical_model",
            "publication-resolution numerical diagnostic predates Phase 2K core-reference correction",
        )
    if "/phase2d/" in p or "phase2d" in p:
        return (
            "CALIBRATION_GOVERNANCE",
            "governance_provenance",
            "records calibration blockers/claim boundaries rather than physical validation",
        )
    if "phase1" in p and "validation" in p:
        return (
            "REPAIR_PROVENANCE",
            "historical_validation_provenance",
            "use for audit history; numerical rows still subject to explicit validity/drift flags",
        )
    if "stage7_materials" in p or "material" in p and p.startswith("outputs/stage"):
        return (
            "MATERIAL_OR_VECTOR_PROXY",
            "planning_only",
            "linear/proxy material study is not calibrated material-modification prediction",
        )
    if "stage7_vector_arm" in p or "stage_g_validation" in p:
        return (
            "NUMERICAL_DIAGNOSTIC_NOT_REPORT_AUTHORISED",
            "diagnostic_only",
            "vector/validation diagnostic requires producer-specific truth and convergence audit",
        )
    if "/manifests/" in p or "manifest" in rel.name.lower():
        return (
            "PROVENANCE_OR_CONFIG",
            "provenance_only",
            "manifest/configuration metadata; may document evidence but does not itself validate physics",
        )
    if "calibration" in p:
        return (
            "CALIBRATION_TEMPLATE_OR_STATUS",
            "governance_provenance",
            "calibration template/status is not measurement unless populated from traceable experiment",
        )
    if suffix in GENERATED_EXTENSIONS:
        return (
            "QUARANTINE_UNREVIEWED",
            "none",
            "generated output not yet bound to a green Phase 2K producer/reference gate",
        )
    return (
        "NONSCIENTIFIC_DERIVATIVE",
        "none",
        "nonstandard output artifact; preserve for provenance until traced",
    )


def audit_file(root: Path, path: Path) -> dict[str, Any]:
    rel = path.relative_to(root)
    status, claim_scope, reason = _base_disposition(rel)
    csv_diag = _csv_diagnostics(path) if path.suffix.lower() == ".csv" else {
        "tabular_rows": None,
        "explicit_false_rows": 0,
        "false_flag_columns": "",
        "max_propagation_power_drift_fraction": None,
        "rows_over_5pct_power_drift": 0,
    }
    json_diag = _json_diagnostics(path) if path.suffix.lower() == ".json" else {
        "json_blockers": "",
        "json_report_authorised_false": False,
        "json_experimentally_validated_false": False,
        "json_absolute_claims_locked": False,
    }

    blockers: list[str] = []
    if csv_diag["explicit_false_rows"]:
        blockers.append(f"{csv_diag['explicit_false_rows']} rows carry explicit false validity/acceptance flags")
    if csv_diag["rows_over_5pct_power_drift"]:
        blockers.append(f"{csv_diag['rows_over_5pct_power_drift']} rows exceed 5% propagation-power drift")
    if json_diag["json_report_authorised_false"]:
        blockers.append("JSON explicitly sets report_figures_authorised=false")
    if json_diag["json_experimentally_validated_false"]:
        blockers.append("JSON explicitly states experimentally_validated_prediction_present=false")
    if json_diag["json_absolute_claims_locked"]:
        blockers.append("JSON explicitly leaves absolute dimensions/fluence locked")
    if json_diag["json_blockers"]:
        blockers.append(json_diag["json_blockers"])

    if csv_diag["explicit_false_rows"] or csv_diag["rows_over_5pct_power_drift"]:
        status = "CONTAINS_INVALID_QUANTITATIVE_ROWS"
        claim_scope = "invalid_rows_diagnostic_only"
        reason = "stored numerical table contains rows that fail its own quantitative validity/conservation gate"
    if json_diag["json_report_authorised_false"]:
        status = "REPORT_USE_EXPLICITLY_BLOCKED"
        claim_scope = "diagnostic_only"
        reason = "the artifact's own governance explicitly blocks report figures"

    return {
        "path": rel.as_posix(),
        "suffix": path.suffix.lower(),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
        "phase2k_disposition": status,
        "claim_scope": claim_scope,
        "scientific_use_allowed_now": False,
        "reason": reason,
        "blockers": " | ".join(blockers),
        **csv_diag,
        **json_diag,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["path"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/validation/phase2k_truth_audit"),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        audit_file(root, path)
        for path in sorted((root / "outputs").rglob("*"))
        if path.is_file() and "phase2k_truth_audit" not in path.as_posix()
    ]
    write_csv(output_dir / "complete_output_semantic_disposition.csv", rows)

    dispositions: dict[str, int] = {}
    total_invalid_rows = 0
    total_drift_rows = 0
    for row in rows:
        dispositions[row["phase2k_disposition"]] = dispositions.get(row["phase2k_disposition"], 0) + 1
        total_invalid_rows += int(row["explicit_false_rows"] or 0)
        total_drift_rows += int(row["rows_over_5pct_power_drift"] or 0)

    summary = {
        "phase": "PHASE2K_SEMANTIC_OUTPUT_AUDIT",
        "audited_file_count": len(rows),
        "scientific_use_allowed_now_count": sum(bool(row["scientific_use_allowed_now"]) for row in rows),
        "disposition_counts": dict(sorted(dispositions.items())),
        "sum_explicit_false_rows_across_tables": total_invalid_rows,
        "sum_rows_over_5pct_power_drift_across_tables": total_drift_rows,
        "policy": (
            "Every pre-Phase-2K output remains blocked for scientific reuse until its producer passes "
            "analytic/reference, independent-numerical, convergence and hardware-provenance gates and the "
            "output is regenerated. Presentation derivatives never outrank source data."
        ),
    }
    (output_dir / "complete_output_semantic_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
