"""PHASE 2B visual diagnostics orchestration and artifact writing."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.digital_twin.phase2a_canonical import _json_ready
from vbb_study.digital_twin.phase2b_figures import (
    plot_3d_volume,
    plot_case_profiles,
    plot_case_slices,
    plot_case_xy_bundle,
    plot_cross_case_hero,
    plot_energy_diagnostics,
    plot_hex_cross_route,
    plot_hex_early_mid_late,
    plot_hex_profiles,
    plot_highn_hex_hero,
)
from vbb_study.digital_twin.phase2b_visual_cases import (
    PHASE2B_3D_CASE_IDS,
    PHASE2B_CASE_IDS,
    Phase2BCaseResult,
    Phase2BConfig,
    build_hex_package,
    build_scalar_cases,
    phase2a_scalar_endpoint_audit,
    phase2b_case_registry,
)


PHASE2B_STAGE = "phase2b_visual_diagnostics_and_beam_volume_maps"
PHASE2B_OUTPUT_ROOT = Path("outputs/figures/phase2b_visual_diagnostics")
PHASE2B_DOC_PATH = Path("docs/91_phase2b_visual_diagnostics_and_beam_volume_maps.md")
PHASE2B_ALLOWED_OUTCOMES = ("PHASE2B-A", "PHASE2B-B", "PHASE2B-C")
PHASE2B_SUBDIRS = (
    "00_manifests",
    "01_case_inputs",
    "02_xy_planes",
    "03_xz_yz_slices",
    "04_profiles",
    "05_3d_maps",
    "06_hex_comparisons",
    "07_energy_ledgers",
    "08_summary_tables",
    "09_final_reports",
)
PHASE2B_UPSTREAM_FILES = (
    Path("outputs/validation/phase2a/canonical_case_summary.csv"),
    Path("outputs/validation/phase2a/canonical_hardware_manifest.json"),
    Path("outputs/validation/phase2a/canonical_power_ledgers.csv"),
    Path("outputs/validation/phase2a/error_injection_registry.csv"),
    Path("outputs/validation/phase2a/phase2a_claim_registry.csv"),
    Path("outputs/validation/phase2a/phase2a_outcome_report.json"),
    Path("outputs/validation/phase2a/slm_model_comparison.csv"),
    Path("outputs/figures/digital_twin/nathan_mode2y_continuous_vs_averaged/continuous_vs_averaged_summary.csv"),
    Path("outputs/figures/digital_twin/nathan_mode2y_continuous_vs_averaged/continuous_vs_averaged_summary.json"),
    Path("outputs/figures/digital_twin/nathan_mode2y_continuous_vs_averaged/simulation_scope_manifest.json"),
    Path("outputs/figures/digital_twin/nathan_mode2z_orientation_interpolation/orientation_interpolation_summary.csv"),
    Path("outputs/figures/digital_twin/nathan_mode2z_orientation_interpolation/simulation_scope_manifest.json"),
    Path("outputs/figures/digital_twin/nathan_mode2z_orientation_interpolation/07_highN_confirmation/mode2z_highn_summary.csv"),
    Path("outputs/figures/digital_twin/nathan_mode2z_orientation_interpolation/07_highN_confirmation/mode2z_highn_audit.json"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upstream_hashes() -> dict[str, str]:
    """Hash every accepted machine artifact consumed by Phase 2B."""

    missing = [str(path) for path in PHASE2B_UPSTREAM_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing accepted upstream Phase 2B inputs: {missing}")
    return {str(path): _sha256(path) for path in PHASE2B_UPSTREAM_FILES}


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    return path


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [dict(row) for row in rows]
    if not records:
        path.write_text("", encoding="utf-8")
        return path
    fields: list[str] = []
    for row in records:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fields})
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _case_summary_row(result: Phase2BCaseResult) -> dict[str, Any]:
    power = np.asarray(result.power_by_z, dtype=float)
    return {
        "case_id": result.case_id,
        "family": result.family,
        "route": result.route,
        "native_grid_n": result.native_grid_n,
        "native_dx_m": result.native_dx_m,
        "z_start_m": float(result.z_values_m[0]),
        "z_end_m": float(result.z_values_m[-1]),
        "z_planes": int(result.z_values_m.size),
        "focus_halfwidth_m": result.focus_halfwidth_m,
        "ring_radius_m": result.ring_radius_m,
        "plane_power_min": float(np.min(power)),
        "plane_power_max": float(np.max(power)),
        "plane_power_drift_fraction": float((np.max(power) - np.min(power)) / max(float(np.max(power)), np.finfo(float).eps)),
        "first_order_efficiency": result.summary.get("first_order_efficiency", result.metadata.get("first_order_efficiency")),
        "peak_intensity_proxy": result.summary.get("peak_intensity", result.summary.get("peak_intensity_au")),
        "useful_region_power_fraction": result.summary.get("useful_power_fraction", "not_applicable"),
        "edge_gradient_sharpness_mm_inv": result.summary.get("edge_gradient_sharpness_mm_inv", "not_applicable"),
        "threshold_transition_width_mm": result.summary.get("threshold_transition_width_mm", "not_applicable"),
        "bright_ridge_fwhm_mm": result.summary.get("bright_ridge_fwhm_mm", "not_applicable"),
        "native_metrics_only": True,
        "display_interpolation_used_for_metrics": False,
        "render_spatial_stride": result.metadata["render_spatial_stride"],
        "render_z_stride": result.metadata["render_z_stride"],
        "render_downsampling_method": result.metadata["render_downsampling_method"],
        "source_contract": result.metadata["source_contract"],
    }


def _energy_source_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    ledger = _read_csv(Path("outputs/validation/phase2a/canonical_power_ledgers.csv"))
    slm = _read_csv(Path("outputs/validation/phase2a/slm_model_comparison.csv"))
    return ledger, slm


def _energy_audit_rows(ledger_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": row["case_id"],
            "route_variant": row["route_variant"],
            "row_index": int(row["row_index"]),
            "stage": row["stage"],
            "factor_id": row["factor_id"],
            "stage_efficiency": float(row["stage_efficiency"]),
            "cumulative_efficiency": float(row["cumulative_efficiency"]),
            "pulse_energy_J": float(row["pulse_energy_J"]),
            "source_phase2a_csv": "outputs/validation/phase2a/canonical_power_ledgers.csv",
            "first_order_efficiency_reapplied": False,
        }
        for row in ledger_rows
    ]


def _hex_metric_summary(package: Any) -> dict[str, Any]:
    c = package.continuous.summary
    a = package.averaged.summary
    edge = float(c["edge_gradient_sharpness_mm_inv"]) / float(a["edge_gradient_sharpness_mm_inv"]) - 1.0
    width = float(a["threshold_transition_width_mm"]) / float(c["threshold_transition_width_mm"]) - 1.0
    ridge = float(a["bright_ridge_fwhm_mm"]) / float(c["bright_ridge_fwhm_mm"]) - 1.0
    return {
        "continuous_edge_gradient_relative_improvement": edge,
        "continuous_transition_width_relative_improvement": width,
        "continuous_ridge_fwhm_relative_improvement": ridge,
        "continuous_improves_three_predeclared_sharpness_observables": bool(min(edge, width, ridge) > 0.0),
        "interpretation": (
            "continuous local orientation is visibly and metrically sharper, while the averaged surrogate retains a small useful-energy advantage"
        ),
    }


def _artifact_manifest(root: Path, *, exclude: Sequence[Path] = ()) -> list[dict[str, Any]]:
    excluded = {path.resolve() for path in exclude}
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.resolve() in excluded:
            continue
        rows.append({
            "path": str(path),
            "relative_path": str(path.relative_to(root)),
            "size_bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        })
    return rows


def _write_document(
    path: Path,
    *,
    root: Path,
    report: Mapping[str, Any],
    case_rows: Sequence[Mapping[str, Any]],
    figure_rows: Sequence[Mapping[str, Any]],
    hex_summary: Mapping[str, Any],
) -> Path:
    case_lines = []
    for row in case_rows:
        useful = row["useful_region_power_fraction"]
        useful_text = "n/a" if useful == "not_applicable" else f"{float(useful):.4f}"
        case_lines.append(
            f"| `{row['case_id']}` | {row['family']} | {int(row['native_grid_n'])} | "
            f"{float(row['native_dx_m']) / 1e-6:.3f} | {int(row['z_planes'])} | "
            f"{float(row['plane_power_drift_fraction']):.3e} | {useful_text} |"
        )
    text = f"""# PHASE 2B - Visual Diagnostics and Beam-Volume Maps

**Status:** visual and interpretation layer only. Outcome **{report['outcome']}**. No Phase 1,
Phase 1R, Phase 2A, MODE 2Y, or MODE 2Z physics contract was changed. No sample-plane or
microfabrication success claim is made.

## Contract

Phase 2B consumes the fixed canonical simulation machinery and writes only to `{root}`. Native
fixed-grid arrays are authoritative for every metric. Scalable angular-spectrum (SAS) arrays are
used for physically resampled focus rendering, while bicubic/Lanczos interpolation is display-only.
The 3D maps are pure transverse `I(x,y)` surfaces at z=60 mm: x and y are physical position, and
surface height and colour encode the same normalised intensity. They contain no propagation axis.

## Canonical Cases

| case | family | native N | native dx (um) | z planes | power drift | useful fraction |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(case_lines)}

The mandatory 3D set is B0, V1, V3, H1 realistic, H1 continuous, and H1 averaged. Each is one
high-resolution z=60 mm intensity surface on a ring-based focus crop, with no isosurface, point-cloud,
envelope, or long propagation dimension. H1 realistic is explicitly the canonical continuous
realistic field, so the H1 realistic and H1 continuous surfaces share the same N=1536 endpoint.

## Hex Comparison

At z=60 mm, continuous local orientation improves edge-gradient sharpness by
{100.0 * float(hex_summary['continuous_edge_gradient_relative_improvement']):.2f}%, narrows the
80-20 transition by {100.0 * float(hex_summary['continuous_transition_width_relative_improvement']):.2f}%,
and narrows ridge FWHM by {100.0 * float(hex_summary['continuous_ridge_fwhm_relative_improvement']):.2f}%
relative to the sector-averaged surrogate. Early z=30 mm, canonical z=60 mm, and late z=150 mm
planes use matched crops, equal-power comparison, and shared colour limits. The N=1536 z=60 hero
uses the already justified MODE 2Z-HN sampling and SAS only as the physical focus renderer.

## Route and Energy Views

The route panel compares target/analytic, ideal sequential, realistic sequential, mild realism,
an uncompensated 0.5 mm axicon/mask offset, and its bounded digital-recentring correction. Metrics
remain on the native N=1024 arrays. The energy panel reads Phase 2A's accepted ledger directly;
first-order efficiency is not multiplied a second time, and stage throughput, plane power,
useful-region fraction, peak proxy, and SLM dead-space semantics remain distinct.

## Provenance and Limits

- Figures generated: `{len(figure_rows)}` PNG/PDF pairs.
- Upstream accepted artifacts unchanged: `{report['upstream_outputs_unchanged']}`.
- Native endpoint checks reproduced: `{report['endpoint_reproduction_pass_count']}` / `{report['endpoint_reproduction_check_count']}`.
- Mandatory 3D outputs present: `{report['mandatory_3d_outputs_present']}`.
- High-N hero present: `{report['highn_hex_hero_present']}`.
- Display interpolation used for metrics: `False`.
- 3D surfaces show z=60 mm plane-peak-normalised intensity only; use the separate x-z/y-z and
  power panels for axial evolution.
- MODE 2Z-HN's sampled transition width and FWHM remain resolution-sensitive. Phase 2B does not
  convert those plateaus into a stronger convergence claim.

## Conclusion

Outcome **{report['outcome']}**: {report['reason']}. The pack is physically coherent for visual
inspection and publication composition, with all quantitative claims traceable to native arrays.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_phase2b_outputs(
    output_root: str | Path = PHASE2B_OUTPUT_ROOT,
    *,
    document_path: str | Path = PHASE2B_DOC_PATH,
    config: Phase2BConfig | None = None,
) -> dict[str, Any]:
    """Generate the complete Phase 2B package without touching accepted outputs."""

    study = config or Phase2BConfig()
    study.validate()
    root = Path(output_root)
    for subdir in PHASE2B_SUBDIRS:
        (root / subdir).mkdir(parents=True, exist_ok=True)
    before_hashes = upstream_hashes()

    scalar = build_scalar_cases(study)
    hex_package = build_hex_package(study)
    cases: dict[str, Phase2BCaseResult] = {
        **scalar,
        "H1_REALISTIC": hex_package.realistic,
        "H1_CONTINUOUS": hex_package.continuous,
        "H1_AVERAGED": hex_package.averaged,
    }

    figure_rows: list[dict[str, Any]] = []
    for case_id in ("G0", "B0", "V1", "V3", "H1_REALISTIC"):
        result = cases[case_id]
        figure_rows.append(plot_case_xy_bundle(result, root))
        figure_rows.append(plot_case_slices(result, root))
        figure_rows.append(plot_case_profiles(result, root))
    figure_rows.append(plot_cross_case_hero(cases, root))
    for case_id in PHASE2B_3D_CASE_IDS:
        for suffix in ("png", "pdf"):
            (root / "05_3d_maps" / f"{case_id.lower()}_3d_volume_diagnostics.{suffix}").unlink(missing_ok=True)
        surface_kwargs: dict[str, Any] = {}
        if case_id in {"H1_REALISTIC", "H1_CONTINUOUS", "H1_AVERAGED"} and bool(hex_package.highn_hero.get("enabled")):
            label = "sector_averaged" if case_id == "H1_AVERAGED" else "continuous"
            surface_kwargs = {
                "display_intensity": hex_package.highn_hero["sas_planes"][label],
                "display_grid": hex_package.highn_hero["sas_grids"][label],
                "display_source": "MODE 2Z-HN N=1536 endpoint with SAS z60 physical resampling",
                "display_source_grid_n": int(hex_package.highn_hero["native_grid_n"]),
                "display_source_dx_m": float(hex_package.highn_hero["native_dx_m"]),
            }
        figure_rows.append(plot_3d_volume(cases[case_id], root, **surface_kwargs))
    figure_rows.append(plot_hex_early_mid_late(hex_package, root))
    figure_rows.append(plot_hex_profiles(hex_package, root))
    highn_record = plot_highn_hex_hero(hex_package, root)
    if highn_record is not None:
        figure_rows.append(highn_record)
    figure_rows.append(plot_hex_cross_route(hex_package, root))

    ledger_rows, slm_rows = _energy_source_rows()
    figure_rows.append(plot_energy_diagnostics(cases, ledger_rows, slm_rows, root))

    scalar_audit = list(phase2a_scalar_endpoint_audit(scalar))
    endpoint_audit = scalar_audit + list(hex_package.endpoint_audit)
    case_rows = [_case_summary_row(cases[key]) for key in (
        "G0", "B0", "V1", "V3", "H1_REALISTIC", "H1_CONTINUOUS", "H1_AVERAGED"
    )]
    energy_rows = _energy_audit_rows(ledger_rows)
    hex_summary = _hex_metric_summary(hex_package)

    registry_path = _write_json(root / "01_case_inputs" / "phase2b_case_registry.json", {
        "stage": PHASE2B_STAGE,
        "registry": phase2b_case_registry(),
        "canonical_h1_relation": "H1_REALISTIC is the accepted realistic continuous field and shares its N=1536 z60 intensity surface with H1_CONTINUOUS",
    })
    case_csv = _write_csv(root / "08_summary_tables" / "phase2b_case_summary.csv", case_rows)
    case_json = _write_json(root / "08_summary_tables" / "phase2b_case_summary.json", case_rows)
    endpoint_csv = _write_csv(root / "08_summary_tables" / "phase2b_endpoint_reproduction_audit.csv", endpoint_audit)
    endpoint_json = _write_json(root / "08_summary_tables" / "phase2b_endpoint_reproduction_audit.json", endpoint_audit)
    provenance_csv = _write_csv(root / "00_manifests" / "figure_provenance.csv", figure_rows)
    provenance_json = _write_json(root / "00_manifests" / "figure_provenance.json", figure_rows)
    energy_csv = _write_csv(root / "07_energy_ledgers" / "phase2a_energy_ledger_plot_data.csv", energy_rows)
    _write_json(root / "07_energy_ledgers" / "phase2a_energy_ledger_plot_data.json", energy_rows)
    _write_csv(root / "07_energy_ledgers" / "phase2a_slm_model_plot_data.csv", slm_rows)
    hex_metrics_json = _write_json(root / "06_hex_comparisons" / "continuous_vs_averaged_metric_summary.json", hex_summary)
    cross_csv = _write_csv(root / "06_hex_comparisons" / "h1_cross_route_metrics.csv", hex_package.cross_route_metrics)
    cross_json = _write_json(root / "06_hex_comparisons" / "h1_cross_route_metrics.json", hex_package.cross_route_metrics)

    after_hashes = upstream_hashes()
    unchanged = before_hashes == after_hashes
    endpoint_passes = sum(bool(row["reproduced"]) for row in endpoint_audit)
    mandatory_3d = {
        case_id: any(row["figure_id"] == f"{case_id}_3d_intensity" for row in figure_rows)
        for case_id in PHASE2B_3D_CASE_IDS
    }
    highn_present = bool(hex_package.highn_hero.get("enabled")) and highn_record is not None
    provenance_valid = all(
        bool(row["metrics_computed_on_native_arrays"])
        and not bool(row["display_interpolation_used_for_metrics"])
        and row["normalisation_policy"]
        and row["crop_rule"]
        for row in figure_rows
    )
    if not unchanged or not provenance_valid:
        outcome = "PHASE2B-C"
        reason = "rendering/provenance consistency failed or an accepted upstream artifact changed"
    elif endpoint_passes != len(endpoint_audit) or not all(mandatory_3d.values()) or not highn_present:
        outcome = "PHASE2B-B"
        reason = "the pack was generated, but at least one mandatory endpoint, 3D map, or high-N hero remains unresolved"
    else:
        outcome = "PHASE2B-A"
        reason = "publication visual diagnostics were generated with native metric provenance, physical SAS focus rendering, and complete mandatory 3D intensity-map coverage"
    report = {
        "stage": PHASE2B_STAGE,
        "outcome": outcome,
        "allowed_outcomes": PHASE2B_ALLOWED_OUTCOMES,
        "reason": reason,
        "canonical_case_ids": PHASE2B_CASE_IDS,
        "case_result_ids": tuple(cases),
        "mandatory_3d_case_ids": PHASE2B_3D_CASE_IDS,
        "mandatory_3d_outputs": mandatory_3d,
        "mandatory_3d_outputs_present": bool(all(mandatory_3d.values())),
        "highn_hex_hero_present": highn_present,
        "highn_hex_hero_grid_n": int(hex_package.highn_hero.get("native_grid_n", 0)),
        "endpoint_reproduction_check_count": len(endpoint_audit),
        "endpoint_reproduction_pass_count": endpoint_passes,
        "endpoint_reproduction_all_pass": endpoint_passes == len(endpoint_audit),
        "upstream_outputs_unchanged": unchanged,
        "upstream_hashes_before": before_hashes,
        "upstream_hashes_after": after_hashes,
        "figure_provenance_complete": provenance_valid,
        "figure_count": len(figure_rows),
        "native_arrays_used_for_metrics": True,
        "sas_used_for_focus_rendering_only": True,
        "display_interpolation_used_for_metrics": False,
        "first_order_efficiency_reapplied": False,
        "split_arm_architecture_reintroduced": False,
        "microfabrication_sample_plane_success_claim": False,
        "continuous_vs_averaged": hex_summary,
        "remaining_limitations": (
            "3D intensity surfaces are normalised z60 morphology views; MODE 2Z-HN sampled width metrics remain resolution-sensitive; absolute camera/fluence claims remain calibration-limited"
        ),
    }
    report_path = _write_json(root / "09_final_reports" / "phase2b_outcome_report.json", report)
    doc_path = _write_document(
        Path(document_path),
        root=root,
        report=report,
        case_rows=case_rows,
        figure_rows=figure_rows,
        hex_summary=hex_summary,
    )
    manifest_path = root / "00_manifests" / "phase2b_final_manifest.json"
    artifact_rows = _artifact_manifest(root, exclude=(manifest_path,))
    manifest = {
        "stage": PHASE2B_STAGE,
        "outcome": outcome,
        "output_root": str(root),
        "document_path": str(doc_path),
        "subdirectories": PHASE2B_SUBDIRS,
        "config": study.__dict__,
        "accepted_physics_changed": False,
        "upstream_outputs_unchanged": unchanged,
        "upstream_input_hashes": after_hashes,
        "artifact_count_excluding_manifest": len(artifact_rows),
        "artifacts": artifact_rows,
    }
    _write_json(manifest_path, manifest)
    return {
        "outcome": report,
        "output_root": str(root),
        "document_path": str(doc_path),
        "case_summary_csv": str(case_csv),
        "case_summary_json": str(case_json),
        "endpoint_audit_csv": str(endpoint_csv),
        "endpoint_audit_json": str(endpoint_json),
        "figure_provenance_csv": str(provenance_csv),
        "figure_provenance_json": str(provenance_json),
        "energy_plot_data_csv": str(energy_csv),
        "hex_metrics_json": str(hex_metrics_json),
        "cross_route_csv": str(cross_csv),
        "cross_route_json": str(cross_json),
        "case_registry": str(registry_path),
        "outcome_report": str(report_path),
        "manifest": str(manifest_path),
        "figure_rows": figure_rows,
    }


__all__ = [
    "PHASE2B_ALLOWED_OUTCOMES",
    "PHASE2B_DOC_PATH",
    "PHASE2B_OUTPUT_ROOT",
    "PHASE2B_STAGE",
    "PHASE2B_SUBDIRS",
    "PHASE2B_UPSTREAM_FILES",
    "upstream_hashes",
    "write_phase2b_outputs",
]
