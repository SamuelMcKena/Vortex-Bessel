"""Artifact writer for the Phase 2E report visual bible."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.digital_twin.phase2e_report_figures import generate_phase2e_figures
from vbb_study.digital_twin.phase2e_report_visualisation import (
    PHASE2E_3D_CASE_IDS,
    PHASE2E_CASE_IDS,
    PHASE2E_DOC_PATH,
    PHASE2E_OUTPUT_ROOT,
    PHASE2E_STAGE,
    Phase2EConfig,
    Phase2EData,
    build_phase2e_data,
    json_ready,
    phase2e_upstream_hashes,
)


EXPECTED_HERO_IDS = (
    "hero_vortex_beam_family",
    "hero_vortex_parameter_dependence",
    "hero_vortex_ideal_realistic_degraded",
    "hero_h1_continuous_vs_averaged",
    "hero_scalar_vector_objective_benchmark",
    "hero_energy_loss_efficiency",
)
EXPECTED_SWEEP_IDS = (
    "vortex_charge",
    "radial_wavevector",
    "input_beam_radius",
    "propagation_distance",
    "aperture_radius",
    "effective_objective_na",
    "defocus_aberration",
    "error_input_beam_decentre",
    "error_input_beam_tilt",
    "error_slm_phase",
    "error_fourier_iris_offset",
    "error_pupil_decentre",
    "error_axicon_decentre",
    "error_zernike_defocus",
    "error_zernike_astigmatism",
    "error_zernike_coma",
    "error_zernike_spherical",
)
EXPECTED_FIGURE_COUNT = 47
SAS_LINE_CORRELATION_FLOOR = 0.95
SCALAR_GRID_RELATIVE_L2_CEILING = 0.20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2) + "\n", encoding="utf-8")
    return path


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"))
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    materialised = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialised:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fields:
            return path
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in materialised:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fields})
    return path


def _case_summary(data: Phase2EData) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_id in ("G0", "B0", "V1", "V3"):
        visual = data.scalar_cases[case_id]
        result = visual.accepted_result
        propagation = visual.propagation
        rows.append({
            "case_id": case_id,
            "family": result.family,
            "route": result.route,
            "native_grid_n": result.native_grid_n,
            "native_dx_m": result.native_dx_m,
            "metric_plane_z_m": result.metadata["metric_plane_z_m"],
            "ring_radius_m": result.ring_radius_m,
            "power_drift_fraction": result.summary["propagation_power_drift_fraction"],
            "metrics_computed_on_native_arrays": True,
            "display_interpolation_used_for_metrics": False,
            "maturity": "accepted_fixed_bench_visual_reconstruction",
            "propagation_method": propagation.metadata["method"],
            "propagation_source_grid_n": propagation.metadata["source_grid_n"],
            "propagation_convergence_grid_n": propagation.metadata["convergence_control_grid_n"],
            "propagation_transverse_samples": propagation.metadata["transverse_samples"],
            "propagation_z_samples": propagation.metadata["z_samples"],
            "propagation_native_parity_error": propagation.metadata["native_line_max_abs_intensity_error"],
            "propagation_xz_grid_convergence_correlation": propagation.metadata["xz_correlation"],
            "propagation_yz_grid_convergence_correlation": propagation.metadata["yz_correlation"],
            "propagation_xz_grid_convergence_relative_l2": propagation.metadata["xz_relative_l2"],
            "propagation_yz_grid_convergence_relative_l2": propagation.metadata["yz_relative_l2"],
            "propagation_sas_x_line_correlation": propagation.metadata["sas_x_line_correlation"],
            "propagation_sas_y_line_correlation": propagation.metadata["sas_y_line_correlation"],
            "propagation_sas_line_validation_applicability": "scalar_same_source_sas",
            "propagation_accepted_grid_sas_x_line_correlation": propagation.metadata[
                "accepted_grid_sas_x_line_correlation"
            ],
            "propagation_accepted_grid_sas_y_line_correlation": propagation.metadata[
                "accepted_grid_sas_y_line_correlation"
            ],
        })
    for case_id, result in (
        ("H1_CONTINUOUS", data.hex_package.continuous),
        ("H1_AVERAGED", data.hex_package.averaged),
    ):
        propagation = data.h1_propagation[case_id]
        rows.append({
            "case_id": case_id,
            "family": result.family,
            "route": result.route,
            "native_grid_n": data.hex_package.highn_hero["native_grid_n"],
            "native_dx_m": data.hex_package.highn_hero["native_dx_m"],
            "metric_plane_z_m": result.metadata["metric_plane_z_m"],
            "ring_radius_m": result.ring_radius_m,
            "power_drift_fraction": result.summary["propagation_power_drift_fraction"],
            "edge_gradient_sharpness_mm_inv": data.hex_package.highn_hero["local_metrics"]["continuous" if case_id.endswith("CONTINUOUS") else "sector_averaged"]["edge_gradient_sharpness_mm_inv"],
            "threshold_transition_width_mm": data.hex_package.highn_hero["local_metrics"]["continuous" if case_id.endswith("CONTINUOUS") else "sector_averaged"]["threshold_transition_width_mm"],
            "bright_ridge_fwhm_mm": data.hex_package.highn_hero["local_metrics"]["continuous" if case_id.endswith("CONTINUOUS") else "sector_averaged"]["bright_ridge_fwhm_mm"],
            "metrics_computed_on_native_arrays": True,
            "display_interpolation_used_for_metrics": False,
            "maturity": "accepted_h1_endpoint_reconstruction",
            "propagation_method": propagation.metadata["method"],
            "propagation_source_grid_n": propagation.metadata["source_grid_n"],
            "propagation_transverse_samples": propagation.metadata["transverse_samples"],
            "propagation_z_samples": propagation.metadata["z_samples"],
            "propagation_native_parity_error": propagation.metadata["native_line_max_abs_intensity_error"],
            "propagation_sas_x_line_correlation": None,
            "propagation_sas_y_line_correlation": None,
            "propagation_sas_line_validation_applicability": propagation.metadata[
                "sas_line_validation_applicability"
            ],
            "propagation_highn_cross_grid_sas_x_line_correlation": propagation.metadata[
                "highn_cross_grid_sas_x_line_correlation"
            ],
            "propagation_highn_cross_grid_sas_y_line_correlation": propagation.metadata[
                "highn_cross_grid_sas_y_line_correlation"
            ],
        })
    return rows


def _sweep_summary(data: Phase2EData) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sweep_id in EXPECTED_SWEEP_IDS:
        for plane in data.sweep_planes[sweep_id]:
            rows.append({
                **dict(plane.metrics),
                "display_label": plane.display_label,
                "source_model": plane.provenance["source_model"],
                "claim_scope": plane.provenance["claim_scope"],
                "accepted_result_replaced": plane.provenance["accepted_result_replaced"],
            })
    return rows


def _headline_propagation_summary(data: Phase2EData) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_id in ("G0", "B0", "V1", "V3"):
        case = data.scalar_cases[case_id]
        propagation = case.propagation
        rows.append({
            "case_id": case_id,
            "source_model": "Phase 2A realistic_fixed_bench_route",
            "hard_pupil_active": True,
            "slm_or_4f_errors_active": True,
            "propagation_method": propagation.metadata["method"],
            "source_grid_n": propagation.metadata["source_grid_n"],
            "source_dx_m": propagation.metadata["source_dx_m"],
            "transverse_samples": propagation.metadata["transverse_samples"],
            "z_samples": propagation.metadata["z_samples"],
            "x_min_m": float(propagation.x_m[0]),
            "x_max_m": float(propagation.x_m[-1]),
            "z_min_m": float(propagation.z_m[0]),
            "z_max_m": float(propagation.z_m[-1]),
            "native_line_max_abs_intensity_error": propagation.metadata[
                "native_line_max_abs_intensity_error"
            ],
            "display_role": "headline accepted finite-aperture route",
            "accepted_fixed_bench_route": True,
            "display_intensity_mapping": "paired global linear I/Imax at 0--1 and 0--0.01 colour ranges",
            "axial_scale_display": "shared z-peak / global Imax, linear",
            "per_z_renormalisation": False,
            "display_spatial_interpolation": "none",
        })
    return rows


def _artifact_rows(root: Path, *, exclude: Sequence[Path] = ()) -> list[dict[str, Any]]:
    excluded = {path.resolve() for path in exclude}
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.resolve() in excluded:
            continue
        rows.append({
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        })
    return rows


def _write_document(
    path: Path,
    *,
    root: Path,
    figures: Sequence[Mapping[str, Any]],
    case_rows: Sequence[Mapping[str, Any]],
    sweep_rows: Sequence[Mapping[str, Any]],
    endpoint_rows: Sequence[Mapping[str, Any]],
    boundary_metrics: Mapping[str, Any],
    style: Any,
) -> Path:
    hero_rows = [row for row in figures if row["report_role"] == "hero_figure"]
    text = f"""# Phase 2E - Report Visualisation and Parameter Sweeps

**Status:** outcome **PHASE2E-A**. This is a visual/presentation layer and a separately scoped
diagnostic-screening layer. It does not rewrite Phase 1/1R/2A/2B/2C physics, overwrite accepted
arrays, or promote uncalibrated dimensions, fluence, or material response.

## Purpose

The pack gives G0, B0, V1, V3, H1 continuous and H1 sector-averaged one report-grade visual
language. It contains core intensity/phase/propagation/profile atlases; pure transverse 3D intensity
surfaces; dedicated full-computational-window x-z/y-z intensity maps; H1 Stokes/orientation comparisons;
ideal/realistic/degraded routes; energy accounting;
pedagogical schematics; seven broad beam-parameter families; and ten source-scale physical-error
families.

Output root: `{root.as_posix()}`.

## Consistency Contract

- Fonts, panel labels, line widths and colour maps come from one serialised style object.
- Source-scale transverse axes are in millimetres. The accepted Phase 2C inset retains micrometres.
- Like-for-like scalar core/3D crops are +/-{style.scalar_focus_halfwidth_m / 1e-3:.2f} mm; H1
  crops are +/-{style.h1_focus_halfwidth_m / 1e-3:.2f} mm. Route-realism crops remain
  +/-{style.realism_focus_halfwidth_m / 1e-3:.2f} mm so the displaced degraded endpoint is visible.
- Transverse ROI occupancy is presentation-governed: the H1 x-y crop is tight enough to expose the sixfold core;
  effective-NA and displaced-realism plates retain their matched full fields but add labelled common
  +/-{style.sweep_focus_halfwidth_m / 1e-3:.2f} mm detail insets so compact beam structure remains legible.
- Propagation comparisons use z=0--200 mm, the complete 10 mm source-grid window, matched transverse limits and {case_rows[0]['propagation_transverse_samples']}x{case_rows[0]['propagation_z_samples']}
  fixed-coordinate spectral maps. They are physical Fourier-series evaluations of the BL-ASM,
  not resized inherited stack cuts. No x-z/y-z ROI is applied. Every atlas pairs an uncapped global
  linear `I/Imax` map with the same global linear ratio shown on a fixed 0--0.01 colour range; values
  above 0.01 are visibly saturated only in the second view. A separate globally linear shared-peak
  curve preserves axial intensity evolution. No per-z renormalisation, logarithm, gamma,
  percentile-based limit or spatial interpolation is applied, and no metric or physical coordinate
  is supplied by the renderer.
- Scalar headline core and propagation plates use the accepted Phase 2A finite-aperture
  `realistic_fixed_bench_route`, including its 1.8 mm hard pupil and fixed-bench terms. The ideal
  untruncated Gaussian-axicon field appears only as a labelled control in the pupil-boundary audit.
  H1 retains its validated common-4F/vector-ASM route, which does not apply the scalar objective-pupil stop.
- Every 3D surface uses x/y in mm, height in normalised intensity, z limits 0--1 and one view angle.
- The top-down parity panel uses the same cropped array and colour limits as its oblique surface.
- Linear/log state and global/per-panel normalisation are explicit in each record.

## Native Versus Display Data

Accepted Phase 2B fixed-grid arrays retain authority for accepted endpoint metrics. Dense x-z/y-z
maps are recomputed by direct centred inverse-DFT synthesis over the complete source-grid window using
the same Matsushima band limit; native inverse-FFT parity and adjacent N=768-to-N=1024 convergence
are recorded in the case summary. The convergence gate requires correlation >=0.98 and relative
L2 <=0.20. Scalar same-source scalable-ASM line parity has a declared correlation floor of
{SAS_LINE_CORRELATION_FLOOR:.2f}; inherited N=512 SAS comparisons are retained separately as
cross-grid diagnostics. Projected-vector H1 is gated by exact native vector-ASM parity and accepted
endpoint reproduction; its N=1024-to-N=1536 line comparison is diagnostic because scalar SAS is
not an interchangeable validator for a divergence-projected vector spectrum. Scalable angular spectrum arrays are physical resamplings used for
focus presentation. Only transverse x-y fields are cropped. For report rendering, scalar SAS crops are
cubic-resampled x{style.scalar_display_resample_factor} and H1 SAS crops x{style.h1_display_resample_factor};
the same display array feeds each 3D/top-down pair. Dense propagation plates use the native
{case_rows[0]['propagation_transverse_samples']}x{case_rows[0]['propagation_z_samples']} spectral array without spatial interpolation. The route-realism plate uses bicubic `imshow`
for its wide view and cubic display-only beam-centred detail insets. All metrics are computed before display
interpolation. No metric is read from a raster image. The Phase 2C hero is a presentation-only
vertical composite of accepted Phase 2C figures so their objective ROIs use the report width; its
numerical evidence remains the Phase 2C CSV files.

## Propagation Boundary Audit

The apparent B0 axial beading and upper-field flare were tested rather than cosmetically removed.
The canonical route clips a {boundary_metrics['beam_radius_m'] / 1e-3:.1f} mm 1/e Gaussian radius
with a {boundary_metrics['pupil_radius_m'] / 1e-3:.1f} mm hard pupil, retaining
{boundary_metrics['hard_pupil_power_fraction']:.3f} of the ideal source power. The corresponding
geometric hard-pupil Bessel-zone end is {boundary_metrics['geometric_pupil_bessel_zone_m'] / 1e-3:.1f} mm;
the 1/e beam-radius estimate is {boundary_metrics['gaussian_radius_bessel_zone_m'] / 1e-3:.1f} mm.
Removing the hard pupil reduces 20--100 mm axial ripple RMS from
{boundary_metrics['hard_pupil']['ripple_rms_normalised']:.4f} to
{boundary_metrics['ideal_untruncated']['ripple_rms_normalised']:.4f}. BL-ASM and unbandlimited ASM
agree at correlation {boundary_metrics['bandlimited_to_unbandlimited_on_axis_correlation']:.8f}, with
maximum normalised difference
{boundary_metrics['bandlimited_to_unbandlimited_max_abs_normalised_difference']:.2e}. The modulation
and post-zone flare are therefore finite-aperture effects in the accepted hard-pupil model, not
Nyquist failure or a Matsushima-mask artifact.

## Diagnostic Sweep Boundary

The {len(EXPECTED_SWEEP_IDS)} sweep families contain {len(sweep_rows)} native SAS points. Seven broad
beam-parameter families use an analytic finite-energy vortex-Bessel screening field. Ten physical
error families retain canonical dual-SLM quantisation, common-4F filtering, objective pupil and
axicon propagation while varying one pre-propagation control at its declared plane: input decentre,
input tilt, SLM phase error, Fourier-iris offset, pupil decentre, axicon decentre, defocus,
astigmatism, coma or spherical aberration. All remain diagnostic, not calibrated tolerances or
replacement fixed-bench claims. Every point records Nyquist and SAS validity plus native metrics.

## Report Hero Figures

{chr(10).join(f"- `{row['figure_id']}` -> `{row['png_path']}`" for row in hero_rows)}

## Governance

- Cases summarised: {len(case_rows)}.
- Figures: {len(figures)} PNG/PDF pairs.
- Endpoint checks: {len(endpoint_rows)}, all reproduced.
- Unity-loss energy rows are explicitly labelled as assumptions/placeholders, and simulated losses
  below 0.001 are marked numerically rather than rendered as unexplained blank bars.
- Upstream files are SHA-256 checked before and after in-memory reconstruction.
- A normal run refuses to overwrite an existing Phase 2E root; replacement requires explicit
  `--overwrite` and still cannot write into accepted upstream roots.

## Limitations

The parameter sweeps are trend-screening simulations, not calibrated experimental predictions. The
effective-NA sweep is a source-plane spectral cutoff diagnostic and is not a replacement for the
vector Debye objective in Phase 2C. Absolute sample dimensions, pulse fluence, damage thresholds,
nonlinear material modification and experimental validation remain calibration-blocked.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def generate_phase2e_outputs(
    config: Phase2EConfig | None = None,
    *,
    output_root: Path = PHASE2E_OUTPUT_ROOT,
    document_path: Path = PHASE2E_DOC_PATH,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate Phase 2E in a new root with explicit overwrite semantics."""

    root = Path(output_root)
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"Phase 2E output root already exists: {root}; pass overwrite=True explicitly")
        resolved = root.resolve()
        accepted_roots = {
            Path("outputs/validation/phase2a").resolve(),
            Path("outputs/figures/phase2b_visual_diagnostics").resolve(),
            Path("outputs/validation/phase2c").resolve(),
            Path("outputs/figures/phase2c").resolve(),
        }
        if resolved in accepted_roots:
            raise ValueError("Phase 2E cannot overwrite an accepted upstream root")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=False)
    data = build_phase2e_data(config)
    figures = generate_phase2e_figures(data, root)
    if len(figures) != EXPECTED_FIGURE_COUNT:
        raise AssertionError(f"expected {EXPECTED_FIGURE_COUNT} Phase 2E figures, got {len(figures)}")
    ids = {str(row["figure_id"]) for row in figures}
    missing_heroes = sorted(set(EXPECTED_HERO_IDS) - ids)
    if missing_heroes:
        raise AssertionError(f"missing Phase 2E hero figures: {missing_heroes}")
    case_rows = _case_summary(data)
    headline_rows = _headline_propagation_summary(data)
    sweep_rows = _sweep_summary(data)
    endpoint_rows = [dict(row) for row in data.endpoint_audit]
    manifest_root = root / "00_manifest"
    summary_root = root / "09_summary"
    final_root = root / "10_final"
    _write_json(manifest_root / "phase2e_style_contract.json", data.config.style.as_dict())
    _write_json(manifest_root / "phase2e_upstream_hashes.json", data.upstream_hashes)
    _write_csv(manifest_root / "phase2e_figure_manifest.csv", figures)
    _write_json(manifest_root / "phase2e_figure_manifest.json", figures)
    _write_csv(summary_root / "phase2e_case_summary.csv", case_rows)
    _write_json(summary_root / "phase2e_case_summary.json", case_rows)
    _write_csv(
        summary_root / "phase2e_headline_propagation_summary.csv",
        headline_rows,
    )
    _write_json(
        summary_root / "phase2e_headline_propagation_summary.json",
        headline_rows,
    )
    _write_csv(summary_root / "phase2e_sweep_summary.csv", sweep_rows)
    _write_json(summary_root / "phase2e_sweep_summary.json", sweep_rows)
    _write_csv(summary_root / "phase2e_endpoint_reproduction_audit.csv", endpoint_rows)
    _write_json(summary_root / "phase2e_endpoint_reproduction_audit.json", endpoint_rows)
    boundary_metrics = dict(data.propagation_boundary_audit.metrics)
    _write_csv(
        summary_root / "phase2e_propagation_boundary_audit.csv",
        [boundary_metrics],
    )
    _write_json(
        summary_root / "phase2e_propagation_boundary_audit.json",
        boundary_metrics,
    )
    _write_document(
        document_path,
        root=root,
        figures=figures,
        case_rows=case_rows,
        sweep_rows=sweep_rows,
        endpoint_rows=endpoint_rows,
        boundary_metrics=boundary_metrics,
        style=data.config.style,
    )
    hashes_after = phase2e_upstream_hashes()
    upstream_unchanged = hashes_after == dict(data.upstream_hashes)
    if not upstream_unchanged:
        raise RuntimeError("accepted upstream artifact changed during Phase 2E output generation")
    dense_native_valid = all(
        float(row["propagation_native_parity_error"]) <= 1.0e-10
        for row in case_rows
    )
    headline_propagation_valid = all(
        bool(row["hard_pupil_active"]) is True
        and bool(row["slm_or_4f_errors_active"]) is True
        and float(row["native_line_max_abs_intensity_error"]) <= 1.0e-10
        and int(row["transverse_samples"]) == 1025
        and int(row["z_samples"]) == 601
        and float(row["x_max_m"]) - float(row["x_min_m"]) >= 9.9e-3
        and float(row["z_max_m"]) - float(row["z_min_m"]) >= 0.2
        and bool(row["accepted_fixed_bench_route"]) is True
        and row["display_intensity_mapping"]
        == "paired global linear I/Imax at 0--1 and 0--0.01 colour ranges"
        and row["axial_scale_display"] == "shared z-peak / global Imax, linear"
        and bool(row["per_z_renormalisation"]) is False
        and row["display_spatial_interpolation"] == "none"
        for row in headline_rows
    )
    if not headline_propagation_valid:
        raise AssertionError("headline scalar propagation contract failed")
    scalar_grid_converged = all(
        float(row["propagation_xz_grid_convergence_correlation"]) >= 0.98
        and float(row["propagation_yz_grid_convergence_correlation"]) >= 0.98
        and float(row["propagation_xz_grid_convergence_relative_l2"])
        <= SCALAR_GRID_RELATIVE_L2_CEILING
        and float(row["propagation_yz_grid_convergence_relative_l2"])
        <= SCALAR_GRID_RELATIVE_L2_CEILING
        for row in case_rows
        if row["case_id"] in {"G0", "B0", "V1", "V3"}
    )
    minimum_sas_line_correlation = min(
        min(
            float(row["propagation_sas_x_line_correlation"]),
            float(row["propagation_sas_y_line_correlation"]),
        )
        for row in case_rows
        if row["case_id"] in {"G0", "B0", "V1", "V3"}
    )
    if not dense_native_valid:
        raise AssertionError("dense propagation failed native inverse-FFT parity")
    if not scalar_grid_converged:
        raise AssertionError("dense scalar propagation failed adjacent-grid convergence")
    if minimum_sas_line_correlation < SAS_LINE_CORRELATION_FLOOR:
        raise AssertionError("dense propagation failed same-source SAS line parity")
    boundary_diagnosis_valid = bool(
        float(boundary_metrics["bandlimited_to_unbandlimited_on_axis_correlation"])
        >= 0.9999
        and float(
            boundary_metrics[
                "bandlimited_to_unbandlimited_max_abs_normalised_difference"
            ]
        )
        <= 1.0e-3
        and float(boundary_metrics["hard_pupil"]["ripple_rms_normalised"])
        >= 5.0
        * float(boundary_metrics["ideal_untruncated"]["ripple_rms_normalised"])
        and float(
            boundary_metrics["hard_pupil_to_realistic_on_axis_correlation"]
        )
        >= 0.99
    )
    if not boundary_diagnosis_valid:
        raise AssertionError("B0 propagation-boundary diagnosis did not reproduce")

    report = {
        "schema_version": "1.0.0",
        "stage": PHASE2E_STAGE,
        "outcome": "PHASE2E-A",
        "output_root": root.as_posix(),
        "document_path": document_path.as_posix(),
        "accepted_physics_changed": False,
        "accepted_arrays_overwritten": False,
        "upstream_outputs_unchanged": True,
        "figure_count": len(figures),
        "hero_figure_ids": list(EXPECTED_HERO_IDS),
        "case_ids": list(PHASE2E_CASE_IDS),
        "pure_3d_case_ids": list(PHASE2E_3D_CASE_IDS),
        "sweep_ids": list(EXPECTED_SWEEP_IDS),
        "sweep_point_count": len(sweep_rows),
        "sweep_maturity": "mixed_diagnostic_screening_and_physical_error_sweeps",
        "endpoint_check_count": len(endpoint_rows),
        "all_endpoints_reproduced": all(bool(row.get("reproduced", False)) for row in endpoint_rows),
        "all_sweep_points_nyquist_valid": all(bool(row["nyquist_pass"]) for row in sweep_rows),
        "all_sweep_points_sas_valid": all(bool(row["sas_valid"]) for row in sweep_rows),
        "all_dense_propagation_native_parity_valid": dense_native_valid,
        "all_headline_scalar_propagation_valid": headline_propagation_valid,
        "all_scalar_dense_propagation_grid_converged": scalar_grid_converged,
        "minimum_dense_propagation_sas_line_correlation": minimum_sas_line_correlation,
        "dense_propagation_sas_line_correlation_floor": SAS_LINE_CORRELATION_FLOOR,
        "dense_propagation_sas_line_gate_scope": "scalar_cases_only",
        "scalar_grid_relative_l2_ceiling": SCALAR_GRID_RELATIVE_L2_CEILING,
        "metrics_from_display_interpolation": False,
        "propagation_boundary_diagnosis_valid": boundary_diagnosis_valid,
        "propagation_boundary_diagnosis": boundary_metrics["diagnosis"],
        "geometric_hard_pupil_bessel_zone_m": boundary_metrics[
            "geometric_pupil_bessel_zone_m"
        ],
        "gaussian_radius_bessel_zone_m": boundary_metrics[
            "gaussian_radius_bessel_zone_m"
        ],
        "unresolved_issues": [
            "absolute dimensions and fluence remain calibration-required",
            "effective-NA sweep is analytic spectral screening, not the Phase 2C vector Debye model",
            "nonlinear material modification and experimental validation are not predicted",
        ],
    }
    report_path = _write_json(final_root / "phase2e_outcome_report.json", report)
    artifact_manifest_path = manifest_root / "phase2e_artifact_manifest.json"
    artifact_rows = _artifact_rows(root, exclude=(artifact_manifest_path,))
    _write_json(artifact_manifest_path, {
        "schema_version": "1.0.0",
        "stage": PHASE2E_STAGE,
        "artifact_count_excluding_manifest": len(artifact_rows),
        "artifacts": artifact_rows,
    })
    return {
        **report,
        "outcome_report": report_path.as_posix(),
        "artifact_manifest": artifact_manifest_path.as_posix(),
        "artifact_count_excluding_manifest": len(artifact_rows),
    }


__all__ = [
    "EXPECTED_FIGURE_COUNT",
    "EXPECTED_HERO_IDS",
    "EXPECTED_SWEEP_IDS",
    "generate_phase2e_outputs",
]
