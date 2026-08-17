"""Capsule-shaped application-geometry planning helpers.

This module is where I collect the capsule study logic: a small joint
spot/zone design solver, an axial-apodization proxy, and feature-shape metrics
from thresholded r-z fluence maps. The model remains a planning simulation. It
does not predict actual weld success, bonding, void formation, ablation, or
refractive-index change without experimental calibration.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vbb_study.config import EPS as BT_EPS, PathKind, TwinConfig, nm as BT_NM, uJ as BT_UJ, um as BT_UM
from vbb_study.design import compute_design_from_config
from vbb_study.facade import core as _bt
from . import vbb_materials, vbb_metrics, vbb_style
from .equations import capsule_geometry
from .publication import capsule as capsule_schema
from .publication.tables import propagation_power_label


def design_solver(
    base_config: TwinConfig,
    *,
    spot_range_um: tuple[float, float] = (2.0, 5.0),
    target_zone_um: float = 200.0,
    wavelength_m: float | None = None,
    ell_values: Sequence[int] = (0, 1, 2, 3),
    spot_samples_um: Sequence[float] = (2.0, 3.0, 4.0, 5.0),
    energy_samples_uJ: Sequence[float] = (5.0, 10.0, 20.0, 40.0),
) -> pd.DataFrame:
    """Return candidate `(ell, k_r, w0, energy)` designs for the capsule study.

    I use the engine inverse design for each spot/length target, then report
    whether the resulting SLM/sample design is hardware-reachable. The pulse
    energy is swept because threshold reach is material- and throughput-
    dependent; the notebooks filter the table rather than hiding rejected rows.
    These are optical geometry candidates, not weld-success predictions.
    """

    rows: list[dict[str, Any]] = []
    low, high = map(float, spot_range_um)
    for ell in ell_values:
        for spot_um in spot_samples_um:
            if not (low <= float(spot_um) <= high):
                continue
            for energy_uJ in energy_samples_uJ:
                cfg = replace(
                    base_config,
                    laser=replace(
                        base_config.laser,
                        wavelength_m=float(wavelength_m or base_config.laser.wavelength_m),
                    ),
                    target=replace(
                        base_config.target,
                        ell=int(ell),
                        target_core_diameter_m=float(spot_um) * BT_UM,
                        target_bessel_length_m=float(target_zone_um) * BT_UM,
                    ),
                    energy=replace(base_config.energy, pulse_energy_in_J=float(energy_uJ) * BT_UJ),
                )
                design = compute_design_from_config(cfg)
                refs = _bt().analytic_references(cfg, design)
                feature_radius_um = (
                    float(refs["core_radius_2405_um"])
                    if int(ell) == 0
                    else float(refs["vortex_first_ring_radius_um"])
                )
                rows.append(
                    {
                        "ell": int(ell),
                        "target_spot_um": float(spot_um),
                        "target_zone_um": float(target_zone_um),
                        "input_energy_uJ": float(energy_uJ),
                        "wavelength_nm": float(cfg.laser.wavelength_m / BT_NM),
                        "kr_m_inv": float(design.kr_sample_m_inv),
                        "w0_sample_um": float(design.w0_sample_m / BT_UM),
                        "gamma_slm_deg": float(design.gamma_slm_deg),
                        "mapping_mode": str(design.mapping_mode),
                        "objective_map_source": str(design.objective_map_source),
                        "objective_map_demag": float(design.objective_map_demag),
                        "predicted_feature_radius_um": feature_radius_um,
                        "predicted_feature_diameter_um": 2.0 * feature_radius_um,
                        "predicted_zmax_um": float(refs["zmax_baliyan_um"]),
                        "predicted_bessel_length_um": float(design.predicted_bessel_length_m / BT_UM),
                        "hardware_reachable": bool(_bt()._hardware_reachable(cfg, design)),
                        "config": cfg,
                        "design": design,
                    }
                )
    frame = pd.DataFrame(rows)
    frame["spot_pass"] = frame["predicted_feature_diameter_um"].between(low, high)
    frame["zone_pass"] = np.abs(frame["predicted_zmax_um"] - float(target_zone_um)) <= 0.25 * float(target_zone_um)
    frame["candidate_pass"] = frame["spot_pass"] & frame["zone_pass"] & frame["hardware_reachable"]
    return frame


def _axial_gain_from_peak(peak: np.ndarray, strength: float) -> np.ndarray:
    peak_arr = np.maximum(np.asarray(peak, dtype=float), 0.0)
    peak_norm = peak_arr / (float(np.max(peak_arr)) + BT_EPS)
    gain = np.where(peak_norm > 0.0, peak_norm ** (-float(strength)), 0.0)
    return np.clip(gain, 0.0, 8.0)


def apply_axial_flatness_proxy(volume: Mapping[str, Any], strength: float) -> dict[str, Any]:
    """Apply an axial flat-top gain proxy to a propagated volume.

    I use the simulated axial peak trace as the control signal: `strength=0`
    leaves the teardrop-like profile alone, and increasing strength boosts weak
    z planes toward a flat-top. This stands in for radial amplitude
    apodization, where larger input radii feed the farther Bessel zone. It is
    not a calibrated CGH-transfer or material-response model.
    """

    out = dict(volume)
    stack = np.asarray(volume["intensity_stack"], dtype=float)
    xz = np.asarray(volume["xz"], dtype=float)
    total = np.asarray(volume["total_power"], dtype=float)
    gain = _axial_gain_from_peak(volume["peak"], strength)
    out["intensity_stack"] = (stack * gain[:, None, None]).astype(np.float32)
    out["xz"] = (xz * gain[None, :]).astype(np.float32)
    out["peak"] = np.max(out["intensity_stack"], axis=(1, 2)).astype(float)
    c = int(out["intensity_stack"].shape[1] // 2)
    out["onaxis"] = out["intensity_stack"][:, c, c].astype(float)
    out["total_power"] = (total * gain).astype(float)
    out["peak_index"] = int(np.argmax(out["peak"]))
    stack2 = out["intensity_stack"]
    out["planes"] = {
        "start": stack2[0].astype(np.float32),
        "middle": stack2[len(stack2) // 2].astype(np.float32),
        "peak": stack2[out["peak_index"]].astype(np.float32),
        "end": stack2[-1].astype(np.float32),
    }
    out["axial_apodization_strength"] = float(strength)
    out["axial_gain"] = gain
    return out


def _radius_by_z_from_threshold(F_xz: np.ndarray, x_m: np.ndarray, threshold: float) -> np.ndarray:
    mask = np.asarray(F_xz, dtype=float) >= float(threshold)
    radii = np.full(mask.shape[1], np.nan, dtype=float)
    for col in range(mask.shape[1]):
        if np.any(mask[:, col]):
            radii[col] = float(np.max(np.abs(x_m[mask[:, col]])))
    return radii


def _absolute_centerline_fluence_xz(
    volume: Mapping[str, Any],
    *,
    pulse_energy_J: float,
) -> np.ndarray:
    """Return an axial fluence proxy that preserves the z-envelope.

    Stage 7 intentionally normalizes each transverse plane independently when I
    only want a threshold contour in one plane. For the capsule question that
    would erase the teardrop: the whole point is whether the axial intensity
    ramp is flattened. Here I normalize every z slice by the power in the peak
    plane, so the centerline map keeps the simulated axial envelope. It remains
    a planning proxy, not a calibrated deposited-energy model.
    """

    stack = np.maximum(np.asarray(volume["intensity_stack"], dtype=float), 0.0)
    grid = volume["crop_grid"]
    peak_idx = int(volume.get("peak_index", int(np.argmax(volume.get("peak", [0.0])))))
    peak_idx = int(np.clip(peak_idx, 0, stack.shape[0] - 1))
    denom = float(np.sum(stack[peak_idx]) * float(grid["dx"]) ** 2 + BT_EPS)
    center_y = stack.shape[1] // 2
    return (float(pulse_energy_J) * stack[:, center_y, :] / denom / 1.0e4).T


def capsule_shape_metrics(
    F_xz_J_cm2: np.ndarray,
    x_m: np.ndarray,
    z_m: np.ndarray,
    threshold_J_cm2: float,
) -> dict[str, Any]:
    """Measure capsule-vs-teardrop geometry from an above-threshold XZ proxy.

    The canonical radius trace is the half-width of the threshold contour at
    each z. A capsule has a long central interval with nearly constant radius
    and roughly symmetric rounded end-caps; a teardrop has a strong radius
    gradient and asymmetric end widths. The output is a planning geometry
    proxy, not measured material modification.
    """

    radius_m = _radius_by_z_from_threshold(F_xz_J_cm2, np.asarray(x_m, dtype=float), threshold_J_cm2)
    valid = np.isfinite(radius_m) & (radius_m > 0.0)
    if np.count_nonzero(valid) < 3:
        return {
            "feature_length_um": 0.0,
            "mean_diameter_um": 0.0,
            "diameter_cv": np.nan,
            "sidewall_straightness": 0.0,
            "taper_angle_deg": np.nan,
            "endcap_roundness": 0.0,
            "top_bottom_asymmetry": np.nan,
            "capsule_index": 0.0,
            "teardrop_index": 1.0,
            "radius_by_z_um": radius_m / BT_UM,
        }
    z = np.asarray(z_m, dtype=float)
    rv = radius_m[valid]
    zv = z[valid]
    length_um = float((zv[-1] - zv[0]) / BT_UM)
    mean_radius = float(np.mean(rv))
    mean_diameter_um = float(2.0 * mean_radius / BT_UM)
    diameter_cv = float(np.std(2.0 * rv) / (np.mean(2.0 * rv) + BT_EPS))
    slope = float(np.polyfit(zv / BT_UM, rv / BT_UM, 1)[0])
    taper_angle_deg = float(np.degrees(np.arctan(abs(slope))))
    sidewall_straightness = float(1.0 / (1.0 + 6.0 * diameter_cv + abs(slope)))
    edge_count = max(1, int(0.15 * len(rv)))
    start_radius = float(np.mean(rv[:edge_count]))
    end_radius = float(np.mean(rv[-edge_count:]))
    asym = float(abs(start_radius - end_radius) / (mean_radius + BT_EPS))
    end_round = float(np.clip(1.0 - 0.5 * (start_radius + end_radius) / (mean_radius + BT_EPS), 0.0, 1.0))
    flat_score = float(1.0 / (1.0 + 8.0 * diameter_cv))
    symmetry_score = float(1.0 / (1.0 + 4.0 * asym))
    capsule = float(np.clip(0.55 * flat_score + 0.30 * sidewall_straightness + 0.15 * symmetry_score, 0.0, 1.0))
    return {
        "feature_length_um": length_um,
        "mean_diameter_um": mean_diameter_um,
        "diameter_cv": diameter_cv,
        "sidewall_straightness": sidewall_straightness,
        "taper_angle_deg": taper_angle_deg,
        "endcap_roundness": end_round,
        "top_bottom_asymmetry": asym,
        "capsule_index": capsule,
        "teardrop_index": float(1.0 - capsule),
        "radius_by_z_um": radius_m / BT_UM,
    }


def simulate_capsule_case(
    config: TwinConfig,
    *,
    path: PathKind,
    case_id: str,
    apodization_strength: float,
    z_values_m: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Run one ideal or lab-realistic capsule case and measure shape metrics."""

    sample_config = replace(config, study_kind="full_source_to_sample")
    result = _bt().run_case(sample_config, preset=str(config.grid.label or "fast"), path=path, case_id=case_id, z_values_m=z_values_m)
    shaped_volume = apply_axial_flatness_proxy(result["volume"], apodization_strength)
    shaped_result = dict(result)
    shaped_result["volume"] = shaped_volume
    metrics = dict(result.get("metrics", {}))
    metrics.update(_bt().extract_vortex_safe_metrics(shaped_result, sample_config))
    metrics["case_id"] = case_id
    metrics["path"] = str(path)
    shaped_result["metrics"] = metrics
    geom = vbb_materials.feature_geometry_from_result(shaped_result, sample_config)
    F_xz = _absolute_centerline_fluence_xz(
        shaped_volume,
        pulse_energy_J=float(sample_config.energy.pulse_energy_at_sample_J),
    )
    geom["centerline_fluence_xz_J_cm2"] = F_xz
    geom["capsule_axial_extent_method"] = "peak_plane_normalized_centerline_fluence"
    geom.update(
        {
            k: v
            for k, v in vbb_materials.xz_feature_geometry(
                F_xz,
                shaped_volume,
                geom["threshold_J_cm2"],
                bessel_zone_um=float(metrics.get("bessel_zone_um", np.nan)),
            ).items()
            if k != "radius_by_z_um"
        }
    )
    shape = capsule_shape_metrics(
        F_xz,
        shaped_volume["crop_grid"]["x"],
        shaped_volume["z"],
        geom["threshold_J_cm2"],
    )
    peak = np.asarray(shaped_volume["peak"], dtype=float)
    axial_cv = float(np.std(peak) / (np.mean(peak) + BT_EPS)) if peak.size else np.nan
    axial_flatness = float(1.0 / (1.0 + axial_cv)) if np.isfinite(axial_cv) else np.nan
    shape["contour_capsule_index"] = float(shape["capsule_index"])
    shape["axial_flatness_score"] = axial_flatness
    shape["axial_peak_cv"] = axial_cv
    if np.isfinite(axial_flatness):
        shape["capsule_index"] = float(np.clip(0.25 * shape["contour_capsule_index"] + 0.75 * axial_flatness, 0.0, 1.0))
        shape["teardrop_index"] = float(1.0 - shape["capsule_index"])
    geom.update(shape)
    geom["apodization_strength"] = float(apodization_strength)
    geom["spot_pass_2_5_um"] = bool(2.0 <= float(shaped_result["metrics"]["feature_diameter_um"]) <= 5.0)
    geom["zone_pass_200_um"] = bool(abs(float(shaped_result["metrics"]["bessel_zone_um"]) - 200.0) <= 50.0)
    geom["lab_or_ideal"] = str(path)
    return {"config": config, "result": shaped_result, "geometry": geom}


def sweep_capsule_apodization(
    config: TwinConfig,
    *,
    strengths: Sequence[float] = (0.0, 0.35, 0.70, 1.0),
    z_values_m: Sequence[float] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run ideal and lab-realistic capsule sweeps side by side."""

    cases: list[dict[str, Any]] = []
    for strength in strengths:
        for path in ("ideal", "realistic"):
            case = simulate_capsule_case(
                config,
                path=path,
                case_id=f"capsule_{path}_a{float(strength):.2f}",
                apodization_strength=float(strength),
                z_values_m=z_values_m,
            )
            cases.append(case)
    rows = []
    for case in cases:
        geom = case["geometry"]
        metrics = case["result"]["metrics"]
        rows.append(
            {
                "case_id": geom["case_id"],
                "path": geom["lab_or_ideal"],
                "apodization_strength": geom["apodization_strength"],
                "feature_diameter_um": float(metrics["feature_diameter_um"]),
                "bessel_zone_um": float(metrics["bessel_zone_um"]),
                "bessel_region_um": float(metrics.get("bessel_region_um", np.nan)),
                "feature_length_um": geom["feature_length_um"],
                "mean_diameter_um": geom["mean_diameter_um"],
                "sidewall_straightness": geom["sidewall_straightness"],
                "taper_angle_deg": geom["taper_angle_deg"],
                "endcap_roundness": geom["endcap_roundness"],
                "top_bottom_asymmetry": geom["top_bottom_asymmetry"],
                "contour_capsule_index": geom["contour_capsule_index"],
                "axial_flatness_score": geom["axial_flatness_score"],
                "axial_peak_cv": geom["axial_peak_cv"],
                "capsule_index": geom["capsule_index"],
                "teardrop_index": geom["teardrop_index"],
                "spot_pass_2_5_um": geom["spot_pass_2_5_um"],
                "zone_pass_200_um": geom["zone_pass_200_um"],
                "hardware_reachable": bool(metrics.get("hardware_reachable", False)),
                "first_order_selected_fraction": float(metrics.get("first_order_selected_fraction", np.nan)),
            }
        )
    return pd.DataFrame(rows), cases


def _score_from_percent_error(error_pct: Any) -> float:
    try:
        value = abs(float(error_pct))
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return float(1.0 / (1.0 + value / 100.0))


def _strict_region_from_metrics(metrics: Mapping[str, Any]) -> float:
    for key in ("strict_bessel_region_um", "bessel_region_um"):
        if key in metrics:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                return np.nan
    return np.nan


def candidate_ranking_from_design_solver(
    design_df: pd.DataFrame,
    *,
    target_width_um: float,
    target_length_um: float,
    target_depth_um: float | None = None,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Return a schema-native optical-geometry candidate ranking table."""

    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(design_df.to_dict("records"), start=1):
        width = float(record.get("predicted_feature_diameter_um", np.nan))
        length = float(record.get("predicted_zmax_um", np.nan))
        row = {
            "case_id": f"capsule_candidate_ell{int(record.get('ell', 0))}_w{float(record.get('target_spot_um', 0.0)):.1f}_e{float(record.get('input_energy_uJ', 0.0)):.0f}",
            "preset": "fast",
            "path": "design_solver",
            "beam_family": "capsule_geometry_proxy",
            "model_level": "optical_geometry_proxy",
            "generation_method": "inverse_design_sweep",
            "hardware_status": "lab_proxy" if bool(record.get("hardware_reachable", False)) else "unsupported",
            "qa_status": "exploratory",
            "material_model_status": "optical_only",
            "material_response_model": "none",
            "calibration_status": "uncalibrated",
            "geometry_model_status": "optical_geometry_proxy",
            "feature_target_type": "capsule",
            "feature_proxy_type": "accepted_depth_proxy",
            "target_width_um": float(target_width_um),
            "target_length_um": float(target_length_um),
            "target_depth_um": float(target_depth_um if target_depth_um is not None else target_length_um),
            "predicted_width_um": width,
            "predicted_length_um": length,
            "predicted_depth_um": length,
            "xz_proxy_definition": "not_applicable",
            "xz_energy_conservation_status": "not_applicable",
            "canonical_zone_um": length,
            "strict_bessel_region_um": np.nan,
            "propagation_power_label": "unknown",
            "ell": int(record.get("ell", 0)),
            "input_energy_uJ": float(record.get("input_energy_uJ", np.nan)),
            "gamma_slm_deg": float(record.get("gamma_slm_deg", np.nan)),
            "hardware_reachable": bool(record.get("hardware_reachable", False)),
            "candidate_pass": bool(record.get("candidate_pass", False)),
            "planning_rank": idx,
            "output_category": "optical_geometry_proxy",
        }
        capsule_schema.annotate_capsule_row(row, run_id=run_id, qa_status="exploratory")
        width_score = _score_from_percent_error(row.get("width_error_pct"))
        length_score = _score_from_percent_error(row.get("length_error_pct"))
        row["capsule_fit_score"] = float(0.5 * width_score + 0.5 * length_score)
        row["capsule_fit_label"] = capsule_schema.capsule_fit_label(row["capsule_fit_score"])
        row["capsule_acceptance_label"] = capsule_schema.capsule_acceptance_label(row)
        rows.append(capsule_schema.ordered_capsule_row(row))
    frame = capsule_schema.ordered_capsule_frame(rows)
    if "capsule_fit_score" in frame:
        frame = frame.sort_values(["candidate_pass", "capsule_fit_score"], ascending=[False, False]).reset_index(drop=True)
        frame["planning_rank"] = np.arange(1, len(frame) + 1)
    return frame


def capsule_summary_from_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    target_width_um: float,
    target_length_um: float,
    target_depth_um: float | None = None,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Return schema-native capsule geometry proxy rows from simulated cases."""

    rows: list[dict[str, Any]] = []
    for case in cases:
        geom = dict(case["geometry"])
        result = case["result"]
        metrics = dict(result.get("metrics", {}))
        volume = result["volume"]
        x_um = np.asarray(volume["crop_grid"]["x"], dtype=float) / BT_UM
        z_um = np.asarray(volume["z"], dtype=float) / BT_UM
        F_xz = np.asarray(geom["centerline_fluence_xz_J_cm2"], dtype=float)
        threshold = float(geom.get("threshold_J_cm2", geom.get("threshold_fluence_J_cm2", np.nan)))
        proxy_mask = F_xz >= threshold if np.isfinite(threshold) else np.zeros_like(F_xz, dtype=bool)
        target_mask = capsule_geometry.capsule_mask_2d(
            x_um,
            z_um,
            width_um=float(target_width_um),
            length_um=float(target_length_um),
            center_z_um=0.5 * float(target_length_um),
        )
        dims = capsule_geometry.width_length_from_mask(proxy_mask, x_um, z_um)
        overlap = capsule_geometry.overlap_score(proxy_mask, target_mask)
        edge = capsule_geometry.edge_uniformity_score(proxy_mask, x_um, z_um)
        side_lobe = capsule_geometry.side_lobe_contamination_score(proxy_mask, target_mask)
        accepted_depth = capsule_geometry.accepted_depth_from_xz_proxy(proxy_mask, z_um)
        power_drift = metrics.get("propagation_power_drift_fraction", np.nan)
        try:
            power_label = propagation_power_label(float(power_drift))
        except (TypeError, ValueError):
            power_label = "unknown"
        width = float(geom.get("mean_diameter_um", dims["width_um"]))
        length = float(geom.get("feature_length_um", dims["length_um"]))
        fit = float(np.clip(0.45 * overlap + 0.35 * float(geom.get("capsule_index", 0.0)) + 0.20 * edge, 0.0, 1.0))
        row = {
            "case_id": str(geom.get("case_id", metrics.get("case_id", "capsule_case"))),
            "preset": str(metrics.get("preset", "fast")),
            "path": str(geom.get("lab_or_ideal", metrics.get("path", "not_applicable"))),
            "beam_family": "capsule_geometry_proxy",
            "model_level": "application_geometry_proxy",
            "generation_method": "thresholded_scalar_xz_proxy",
            "hardware_status": "lab_proxy" if bool(metrics.get("hardware_reachable", False)) else "unsupported",
            "qa_status": str(metrics.get("qa_status", "exploratory") or "exploratory"),
            "material_model_status": "planning_proxy",
            "material_response_model": "incubation_threshold_proxy",
            "calibration_status": "uncalibrated",
            "geometry_model_status": "thresholded_fluence_proxy",
            "feature_target_type": "capsule",
            "feature_proxy_type": "capsule_overlap_proxy",
            "target_width_um": float(target_width_um),
            "target_length_um": float(target_length_um),
            "target_depth_um": float(target_depth_um if target_depth_um is not None else target_length_um),
            "predicted_width_um": width,
            "predicted_length_um": length,
            "predicted_depth_um": accepted_depth,
            "overlap_score": overlap,
            "capsule_fit_score": fit,
            "edge_uniformity_score": edge,
            "core_suppression_score": np.nan,
            "side_lobe_contamination_score": side_lobe,
            "threshold_fluence_J_cm2": threshold,
            "peak_fluence_J_cm2": float(geom.get("peak_fluence_J_cm2", metrics.get("peak_fluence_J_cm2", np.nan))),
            "fluence_to_threshold_ratio": float(geom.get("fluence_to_threshold_ratio", np.nan)),
            "xz_proxy_definition": str(geom.get("capsule_axial_extent_method", "peak_plane_normalized_centerline_fluence")),
            "xz_energy_conservation_status": "normalised_visualisation",
            "canonical_zone_um": float(metrics.get("canonical_zone_um", metrics.get("bessel_zone_um", np.nan))),
            "strict_bessel_region_um": _strict_region_from_metrics(metrics),
            "propagation_power_drift_fraction": power_drift,
            "propagation_power_label": power_label,
            "apodization_strength": float(geom.get("apodization_strength", np.nan)),
            "contour_capsule_index": float(geom.get("contour_capsule_index", geom.get("capsule_index", np.nan))),
            "axial_flatness_score": float(geom.get("axial_flatness_score", np.nan)),
            "spot_pass_2_5_um": bool(geom.get("spot_pass_2_5_um", False)),
            "zone_pass_200_um": bool(geom.get("zone_pass_200_um", False)),
            "hardware_reachable": bool(metrics.get("hardware_reachable", False)),
            "output_category": "thresholded_fluence_proxy",
        }
        capsule_schema.annotate_capsule_row(row, run_id=run_id, qa_status=row["qa_status"])
        rows.append(capsule_schema.ordered_capsule_row(row))
    return capsule_schema.ordered_capsule_frame(rows)


def capsule_acceptance_summary(summary_df: pd.DataFrame, *, run_id: str | None = None) -> pd.DataFrame:
    """Return a schema-native capsule acceptance table from summary rows."""

    rows: list[dict[str, Any]] = []
    for record in summary_df.to_dict("records"):
        row = dict(record)
        row["acceptance_pass"] = bool(
            row.get("capsule_acceptance_label") == "planning_proxy_candidate"
            and row.get("propagation_power_label") != "fail"
        )
        row["acceptance_note"] = (
            "proxy geometry only; experimental calibration required before physical feature claims"
        )
        capsule_schema.annotate_capsule_row(row, run_id=run_id, qa_status=str(row.get("qa_status", "exploratory")))
        rows.append(capsule_schema.ordered_capsule_row(row))
    return capsule_schema.ordered_capsule_frame(rows)


def plot_capsule_rz_pair(cases: Sequence[Mapping[str, Any]], output_path: str | Path) -> Path:
    """Save ideal-vs-lab r-z fluence maps with threshold contours."""

    vbb_style.apply_style()
    selected = list(cases)[:2]
    if len(selected) != 2:
        raise ValueError("plot_capsule_rz_pair expects two cases.")
    vmax = max(float(np.nanmax(c["geometry"]["centerline_fluence_xz_J_cm2"])) for c in selected)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    artist = None
    for ax, case in zip(axes, selected):
        geom = case["geometry"]
        volume = case["result"]["volume"]
        grid = volume["crop_grid"]
        F = np.asarray(geom["centerline_fluence_xz_J_cm2"], dtype=float)
        x_um = np.asarray(grid["x"], dtype=float) / BT_UM
        z_um = np.asarray(volume["z"], dtype=float) / BT_UM
        artist = ax.imshow(
            vbb_style.display_scale(F / (vmax + BT_EPS), gamma=0.55, normalise=False),
            origin="lower",
            aspect="auto",
            extent=[float(z_um[0]), float(z_um[-1]), float(x_um[0]), float(x_um[-1])],
            cmap=vbb_style.INTENSITY_CMAP,
            vmin=0.0,
            vmax=1.0,
        )
        if np.nanmin(F) <= geom["threshold_J_cm2"] <= np.nanmax(F):
            ax.contour(z_um, x_um, F, levels=[geom["threshold_J_cm2"]], colors="white", linewidths=1.1)
        ax.set_title(f"{geom['lab_or_ideal']} | a={geom['apodization_strength']:.2f}")
        ax.set_xlabel("z [um, sample plane]")
        ax.set_ylabel("x [um, sample plane]")
    if artist is not None:
        cbar = fig.colorbar(artist, ax=axes, shrink=0.92)
        cbar.set_label("matched display fluence, gamma=0.55 [a.u.]")
    caption = (
        "Ideal and lab-realistic capsule-planning XZ fluence proxies with the incubated threshold contour overlaid. "
        "The display uses matched gamma scaling; this is not an actual weld or material-response prediction."
    )
    path = vbb_style.save_figure(fig, output_path, caption, metadata={"figure": "capsule_rz_pair"})
    plt.close(fig)
    return path


def plot_capsule_sweep(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save capsule-index and geometry trends versus apodization strength."""

    vbb_style.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), constrained_layout=True)
    for path, group in df.groupby("path"):
        group = group.sort_values("apodization_strength")
        axes[0].plot(group["apodization_strength"], group["capsule_index"], marker="o", label=path)
        axes[1].plot(group["apodization_strength"], group["mean_diameter_um"], marker="o", label=path)
        axes[2].plot(group["apodization_strength"], group["feature_length_um"], marker="o", label=path)
    axes[0].set_ylabel("capsule index [0-1]")
    axes[1].set_ylabel("mean threshold diameter [um, sample plane]")
    axes[2].set_ylabel("threshold length [um, sample plane]")
    for ax in axes:
        ax.set_xlabel("apodization strength [a.u.]")
    axes[0].legend(frameon=False)
    caption = (
        "Capsule application-geometry proxy metrics versus axial-apodization strength for ideal and lab-realistic paths. "
        "The capsule index combines radius flatness, sidewall straightness, and end symmetry; it is not weld success."
    )
    path = vbb_style.save_figure(fig, output_path, caption, metadata={"figure": "capsule_sweep_metrics"})
    plt.close(fig)
    return path


__all__ = [
    "apply_axial_flatness_proxy",
    "candidate_ranking_from_design_solver",
    "capsule_acceptance_summary",
    "capsule_shape_metrics",
    "capsule_summary_from_cases",
    "design_solver",
    "plot_capsule_rz_pair",
    "plot_capsule_sweep",
    "simulate_capsule_case",
    "sweep_capsule_apodization",
]
