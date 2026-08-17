"""Materials-facing feature proxies for the vortex-Bessel-beam study.

This module is where I connect optical simulations to fabrication questions:
what fluence crosses a threshold, what modified feature geometry that predicts,
and whether a target is reachable with the current SLM/laser assumptions. The
model is intentionally labelled as a simulation-side proxy. It is not a
calibrated ablation, melt, void, or refractive-index-change law.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import special as sp

from vbb_study.config import EPS as BT_EPS, PathKind, TwinConfig, mm as BT_MM, uJ as BT_UJ, um as BT_UM
from vbb_study.design import default_config
from vbb_study.facade import core as _bt
from . import vbb_metrics, vbb_style
from .equations import materials as material_eq
from .publication import materials as material_schema


def positive_intensity(values: np.ndarray) -> np.ndarray:
    """Return the non-negative intensity array used by all fluence proxies."""

    return material_eq.positive_intensity(values)


def fluence_from_intensity(intensity: np.ndarray, dx_m: float, pulse_energy_J: float) -> np.ndarray:
    """Convert one transverse intensity plane to fluence in J/cm^2.

    I use the same energy-conserving normalization as the legacy
    `bt.fluence_from_intensity`: `F = E_pulse I / integral(I dA)`, followed by
    `J/m^2 -> J/cm^2`.
    """

    return material_eq.fluence_from_intensity_J_cm2(intensity, dx_m=dx_m, pulse_energy_J=pulse_energy_J)


def line_fluence_proxy_xz(xz_intensity: np.ndarray, dx_m: float, pulse_energy_J: float) -> np.ndarray:
    """Return the labelled XZ line-write fluence proxy in J/cm^2.

    Each z column is independently normalized by `sum_x I(x,z) dx`. This makes a
    useful writing-trend proxy, but it is not a conserved energy density.
    """

    return material_eq.line_fluence_proxy_J_cm2(xz_intensity, dx_m=dx_m, pulse_energy_J=pulse_energy_J)


def plane_fluence_stack_from_intensity(
    intensity_stack: np.ndarray,
    dx_m: float,
    pulse_energy_J: float,
) -> np.ndarray:
    """Normalize every propagated XY plane to a comparable fluence in J/cm^2.

    I use this for thresholded axial geometry because each z slice then uses the
    same two-dimensional energy budget as `fluence_from_intensity`. This is a
    threshold proxy, not a statement that the pulse deposits the same energy at
    every depth.
    """

    stack = positive_intensity(intensity_stack)
    denom = np.sum(stack, axis=(1, 2), keepdims=True) * float(dx_m) * float(dx_m) + BT_EPS
    return (float(pulse_energy_J) * stack / denom) / 1.0e4


def centerline_fluence_xz_from_stack(fluence_stack_J_cm2: np.ndarray) -> np.ndarray:
    """Return the y=0 centerline XZ fluence view from a `(z, y, x)` stack."""

    stack = np.asarray(fluence_stack_J_cm2, dtype=float)
    if stack.ndim != 3:
        raise ValueError("fluence_stack_J_cm2 must have shape (z, y, x)")
    center_y = stack.shape[1] // 2
    return np.asarray(stack[:, center_y, :].T, dtype=float)


def effective_pulses(material: Any, rep_rate_Hz: float) -> float:
    """Return the effective pulse count used by the incubation proxy."""

    if hasattr(material, "effective_pulses"):
        return float(material.effective_pulses(float(rep_rate_Hz)))
    mode = str(getattr(material, "static_or_scan", "scan")).lower().strip()
    if mode == "scan":
        width = float(getattr(material, "feature_width_m", 3.0 * BT_UM))
        speed = max(float(getattr(material, "scan_speed_m_s", 1.0 * BT_MM)), BT_EPS)
        return max(1.0, float(rep_rate_Hz) * width / speed)
    return max(1.0, float(getattr(material, "n_static_pulses", 1)))


def incubated_threshold_J_cm2(material: Any, rep_rate_Hz: float) -> float:
    """Return the incubated threshold proxy in J/cm^2.

    The model is `F_th,N = F_th,1 * N_eff**(S - 1)`, where `S` is the incubation
    exponent. This is a planning proxy until measured calibration data exist.
    """

    if hasattr(material, "incubated_threshold_J_cm2"):
        return float(material.incubated_threshold_J_cm2(float(rep_rate_Hz)))
    n_eff = effective_pulses(material, rep_rate_Hz)
    single = float(getattr(material, "single_pulse_threshold_J_cm2", 2.0))
    exponent = float(getattr(material, "incubation_exponent", 0.84))
    return material_eq.incubated_threshold_J_cm2(
        single_pulse_threshold_J_cm2=single,
        effective_pulses=n_eff,
        incubation_exponent=exponent,
    )


def threshold_mask(values_J_cm2: np.ndarray, threshold_J_cm2: float) -> np.ndarray:
    """Return the above-threshold mask for a fluence field."""

    return material_eq.threshold_mask(values_J_cm2, threshold_J_cm2)


def _threshold_crossing(
    radius_m: np.ndarray,
    profile: np.ndarray,
    threshold: float,
    *,
    ell: int,
    kr_m_inv: float,
) -> tuple[float, float, int, float]:
    """Find local threshold radii around the core/ring feature."""

    r = np.asarray(radius_m, dtype=float)
    y = np.asarray(profile, dtype=float)
    finite = np.isfinite(y)
    if not np.any(finite) or np.nanmax(y) < threshold:
        return (np.nan, np.nan, -1, float(np.nanmax(y)) if np.any(finite) else np.nan)
    ell_abs = abs(int(ell))
    if ell_abs == 0:
        search = finite & (r <= max(3.0 / max(float(kr_m_inv), BT_EPS), r[min(len(r) - 1, max(3, len(r) // 5))]))
    else:
        predicted = float(sp.jnp_zeros(ell_abs, 1)[0] / max(float(kr_m_inv), BT_EPS))
        search = finite & (r >= 0.45 * predicted) & (r <= 1.8 * predicted)
    if not np.any(search):
        search = finite
    idxs = np.flatnonzero(search)
    peak_idx = int(idxs[int(np.nanargmax(y[idxs]))])
    if y[peak_idx] < threshold:
        return (np.nan, np.nan, peak_idx, float(y[peak_idx]))

    left = peak_idx
    while left > 0 and np.isfinite(y[left]) and y[left] >= threshold:
        left -= 1
    if ell_abs == 0:
        r_inner = 0.0
    elif left == peak_idx:
        r_inner = float(r[peak_idx])
    elif left <= 0 and y[left] >= threshold:
        r_inner = 0.0
    else:
        r_inner = _interp_level(float(r[left]), float(y[left]), float(r[left + 1]), float(y[left + 1]), threshold)

    right = peak_idx
    while right < len(y) - 1 and np.isfinite(y[right]) and y[right] >= threshold:
        right += 1
    if right == peak_idx:
        r_outer = float(r[peak_idx])
    elif right >= len(y) - 1 and y[right] >= threshold:
        r_outer = float(r[right])
    else:
        r_outer = _interp_level(float(r[right - 1]), float(y[right - 1]), float(r[right]), float(y[right]), threshold)
    return (float(r_inner), float(max(r_outer, r_inner)), peak_idx, float(y[peak_idx]))


def _interp_level(x0: float, y0: float, x1: float, y1: float, level: float) -> float:
    if abs(y1 - y0) <= BT_EPS:
        return float(0.5 * (x0 + x1))
    t = (float(level) - y0) / (y1 - y0)
    return float(x0 + np.clip(t, 0.0, 1.0) * (x1 - x0))


def _angular_uniformity(
    values: np.ndarray,
    grid: Mapping[str, Any],
    radial_mask: np.ndarray,
    *,
    angle_bins: int = 72,
) -> dict[str, float]:
    """Return azimuthal CV and a bounded uniformity score for a feature mask."""

    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    theta = np.mod(np.arctan2(Y, X), 2.0 * np.pi)
    edges = np.linspace(0.0, 2.0 * np.pi, int(angle_bins) + 1)
    vals = np.asarray(values, dtype=float)
    means = []
    mask = np.asarray(radial_mask, dtype=bool)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sector = mask & (theta >= lo) & (theta < hi)
        if np.any(sector):
            means.append(float(np.mean(vals[sector])))
    if len(means) < 4:
        return {"azimuthal_cv": np.nan, "azimuthal_uniformity": np.nan, "azimuthal_bins_used": float(len(means))}
    arr = np.asarray(means, dtype=float)
    mean = float(np.mean(arr))
    cv = float(np.std(arr) / (mean + BT_EPS))
    return {
        "azimuthal_cv": cv,
        "azimuthal_uniformity": float(1.0 / (1.0 + cv)),
        "azimuthal_bins_used": float(len(means)),
    }


def xy_feature_geometry(
    fluence_xy_J_cm2: np.ndarray,
    grid: Mapping[str, Any],
    threshold_J_cm2: float,
    *,
    ell: int,
    kr_m_inv: float,
) -> dict[str, Any]:
    """Measure above-threshold transverse planning-proxy geometry.

    The thresholded mask is an optical comparison to a planning threshold. It is
    not a calibrated material modification, damage, ablation, void, weld, or
    index-change prediction.
    """

    F = np.asarray(fluence_xy_J_cm2, dtype=float)
    mask = threshold_mask(F, threshold_J_cm2)
    dx = float(grid["dx"])
    area_m2 = float(np.count_nonzero(mask) * dx * dx)
    equivalent_diameter_um = float(2.0 * np.sqrt(area_m2 / np.pi) / BT_UM) if area_m2 > 0 else 0.0

    avg = vbb_metrics.azimuthal_average(F, grid, center_mode="origin")
    r = np.asarray(avg["r_m"], dtype=float)
    profile = np.asarray(avg["profile"], dtype=float)
    r_inner, r_outer, peak_idx, feature_peak = _threshold_crossing(
        r,
        profile,
        float(threshold_J_cm2),
        ell=ell,
        kr_m_inv=kr_m_inv,
    )
    if np.any(mask):
        R = np.asarray(grid["R"], dtype=float)
        r_mask = R[mask]
        max_radius_um = float(np.max(r_mask) / BT_UM)
        min_radius_um = float(np.min(r_mask) / BT_UM)
    else:
        max_radius_um = min_radius_um = 0.0

    ell_abs = abs(int(ell))
    R = np.asarray(grid["R"], dtype=float)
    if np.isfinite(r_outer) and r_outer > r_inner:
        radial_mask = (R >= r_inner) & (R <= r_outer)
    elif np.any(mask):
        radial_mask = mask
    else:
        radial_mask = np.zeros_like(mask, dtype=bool)
    uniformity = _angular_uniformity(F, grid, radial_mask)
    threshold_ring_radius_um = float(0.5 * (r_inner + r_outer) / BT_UM) if ell_abs > 0 and np.isfinite(r_outer) else np.nan
    threshold_wall_um = float((r_outer - r_inner) / BT_UM) if np.isfinite(r_outer) else np.nan

    return {
        "threshold_J_cm2": float(threshold_J_cm2),
        "xy_threshold_area_um2": float(area_m2 / (BT_UM**2)),
        "xy_threshold_equivalent_diameter_um": equivalent_diameter_um,
        "thresholded_proxy_area_um2": float(area_m2 / (BT_UM**2)),
        "thresholded_equivalent_diameter_um": equivalent_diameter_um,
        "xy_threshold_min_radius_um": min_radius_um,
        "xy_threshold_max_radius_um": max_radius_um,
        "xy_threshold_inner_radius_um": float(r_inner / BT_UM) if np.isfinite(r_inner) else np.nan,
        "xy_threshold_outer_radius_um": float(r_outer / BT_UM) if np.isfinite(r_outer) else np.nan,
        "xy_threshold_wall_thickness_um": threshold_wall_um,
        "annular_threshold_proxy_ring_radius_um": threshold_ring_radius_um,
        "annular_threshold_proxy_wall_thickness_um": threshold_wall_um if ell_abs > 0 else np.nan,
        "annular_modified_ring_radius_um": threshold_ring_radius_um,
        "annular_wall_thickness_um": threshold_wall_um if ell_abs > 0 else np.nan,
        "annular_peak_mean_fluence_J_cm2": float(feature_peak),
        "xy_threshold_peak_bin_index": int(peak_idx),
        "xy_thresholded_proxy_fraction": float(np.count_nonzero(mask) / mask.size),
        "xy_modified_fraction": float(np.count_nonzero(mask) / mask.size),
        **uniformity,
    }


def xz_feature_geometry(
    line_fluence_J_cm2: np.ndarray,
    volume: Mapping[str, Any],
    threshold_J_cm2: float,
    *,
    bessel_zone_um: float | None = None,
) -> dict[str, Any]:
    """Measure above-threshold axial planning extent from an XZ view.

    This summarizes a displayed XZ planning proxy. It is not a true
    energy-conserving 3D deposition model unless the caller supplies and labels
    one explicitly.
    """

    F = np.asarray(line_fluence_J_cm2, dtype=float)
    mask = threshold_mask(F, threshold_J_cm2)
    x = np.asarray(volume["crop_grid"]["x"], dtype=float)
    z = np.asarray(volume["z"], dtype=float)
    dx = float(volume["crop_grid"]["dx"])
    dz = float(abs(z[1] - z[0])) if len(z) > 1 else 0.0
    if np.any(mask):
        inds = np.argwhere(mask)
        ix0, iz0 = inds.min(axis=0)
        ix1, iz1 = inds.max(axis=0)
        length_um = float(max(0.0, z[iz1] - z[iz0]) / BT_UM)
        width_um = float(max(0.0, x[ix1] - x[ix0]) / BT_UM)
        area_um2 = float(np.count_nonzero(mask) * dx * dz / (BT_UM**2))
        z_start_um = float(z[iz0] / BT_UM)
        z_end_um = float(z[iz1] / BT_UM)
    else:
        length_um = width_um = area_um2 = z_start_um = z_end_um = 0.0

    radius_by_z_um = np.full(len(z), np.nan, dtype=float)
    for col in range(len(z)):
        xs = np.abs(x[mask[:, col]]) if np.any(mask[:, col]) else np.asarray([], dtype=float)
        if xs.size:
            radius_by_z_um[col] = float(np.max(xs) / BT_UM)
    valid = np.isfinite(radius_by_z_um)
    if np.count_nonzero(valid) >= 2:
        taper_um = float(np.nanmax(radius_by_z_um[valid]) - np.nanmin(radius_by_z_um[valid]))
        mean_radius = float(np.nanmean(radius_by_z_um[valid]))
        coeff = np.polyfit(z[valid] / BT_UM, radius_by_z_um[valid], 1)
        slope = float(coeff[0])
    else:
        taper_um = mean_radius = slope = np.nan
    zone = float(bessel_zone_um) if bessel_zone_um is not None and np.isfinite(bessel_zone_um) else np.nan
    ratio = float(length_um / zone) if zone and zone > 0 else np.nan
    return {
        "xz_threshold_length_um": length_um,
        "xz_threshold_width_um": width_um,
        "xz_threshold_area_um2": area_um2,
        "xz_proxy_length_um": length_um,
        "xz_proxy_width_um": width_um,
        "xz_proxy_area_um2": area_um2,
        "xz_threshold_start_um": z_start_um,
        "xz_threshold_end_um": z_end_um,
        "thresholded_proxy_to_bessel_zone_ratio": ratio,
        "written_feature_to_bessel_zone_ratio": ratio,
        "threshold_radius_mean_um": mean_radius,
        "threshold_taper_um": taper_um,
        "threshold_taper_fraction": float(taper_um / (mean_radius + BT_EPS)) if np.isfinite(taper_um) and np.isfinite(mean_radius) else np.nan,
        "tgv_radius_mean_um": mean_radius,
        "tgv_taper_um": taper_um,
        "tgv_taper_fraction": float(taper_um / (mean_radius + BT_EPS)) if np.isfinite(taper_um) and np.isfinite(mean_radius) else np.nan,
        "tgv_radius_slope_um_per_um": slope,
        "radius_by_z_um": radius_by_z_um,
    }


def _strict_region_um(metrics: Mapping[str, Any]) -> float:
    for key in ("strict_bessel_region_um", "bessel_region_um"):
        if key in metrics:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                return np.nan
    return np.nan


def _transmission_fraction(config: TwinConfig) -> float:
    energy = config.energy
    factors = [
        float(getattr(energy, "pre_slm_transmission", 1.0)),
        float(getattr(energy, "first_order_efficiency", 1.0)),
        float(getattr(energy, "relay_transmission", 1.0)),
        float(getattr(energy, "focusing_transmission", 1.0)),
        float(getattr(energy, "sample_surface_transmission", 1.0)),
        float(getattr(energy, "user_extra_transmission", 1.0)),
    ]
    return material_eq.transmission_fraction(*factors)


def feature_geometry_from_result(result: Mapping[str, Any], config: TwinConfig) -> dict[str, Any]:
    """Compute material-facing planning proxies for one simulated case.

    The returned thresholded quantities compare optical fluence with a planning
    threshold. They are not calibrated material-response predictions.
    """

    volume = result["volume"]
    design = result["design"]
    grid = volume["crop_grid"]
    peak_idx = int(volume.get("peak_index", 0))
    if "intensity_stack" in volume:
        plane = np.asarray(volume["intensity_stack"][peak_idx], dtype=float)
    else:
        plane = np.asarray(volume["planes"]["peak"], dtype=float)
    threshold = incubated_threshold_J_cm2(config.material, config.laser.rep_rate_Hz)
    pulse_energy = float(config.energy.pulse_energy_at_sample_J)
    F_xy = fluence_from_intensity(plane, grid["dx"], pulse_energy)
    line_proxy_xz = line_fluence_proxy_xz(volume["xz"], grid["dx"], pulse_energy)
    if "intensity_stack" in volume:
        fluence_stack = plane_fluence_stack_from_intensity(volume["intensity_stack"], grid["dx"], pulse_energy)
        F_xz = centerline_fluence_xz_from_stack(fluence_stack)
        axial_method = "plane_normalized_centerline_fluence"
        xz_energy_status = "normalised_visualisation"
    else:
        fluence_stack = None
        F_xz = line_proxy_xz
        axial_method = "legacy_line_fluence_proxy"
        xz_energy_status = "non_energy_conserving_line_proxy"
    xy = xy_feature_geometry(F_xy, grid, threshold, ell=design.ell, kr_m_inv=design.kr_sample_m_inv)
    bessel_zone_um = float(result.get("metrics", {}).get("bessel_zone_um", np.nan))
    xz = xz_feature_geometry(F_xz, volume, threshold, bessel_zone_um=bessel_zone_um)
    metrics = result.get("metrics", {})
    return {
        "case_id": str(metrics.get("case_id", "")),
        "path": str(metrics.get("path", result.get("path", ""))),
        "ell": int(design.ell),
        "target_core_diameter_um": float(design.target_core_diameter_m / BT_UM),
        "target_bessel_length_um": float(design.target_bessel_length_m / BT_UM),
        "gamma_slm_deg": float(design.gamma_slm_deg),
        "input_energy_uJ": float(config.energy.pulse_energy_in_J / BT_UJ),
        "pulse_energy_at_sample_uJ": float(config.energy.pulse_energy_at_sample_J / BT_UJ),
        "threshold_J_cm2": threshold,
        "threshold_fluence_J_cm2": threshold,
        "effective_pulses": effective_pulses(config.material, config.laser.rep_rate_Hz),
        "pulse_count": effective_pulses(config.material, config.laser.rep_rate_Hz),
        "bessel_zone_um": bessel_zone_um,
        "canonical_zone_um": bessel_zone_um,
        "strict_bessel_region_um": _strict_region_um(metrics),
        "ring_radius_um": float(metrics.get("ring_radius_um", np.nan)),
        "ring_width_um": float(metrics.get("ring_width_um", np.nan)),
        "peak_fluence_J_cm2": float(metrics.get("peak_fluence_J_cm2", np.nan)),
        "fluence_to_threshold_ratio": material_eq.fluence_to_threshold_ratio(
            fluence_J_cm2=float(metrics.get("peak_fluence_J_cm2", np.nan)),
            threshold_J_cm2=threshold,
        ),
        "side_to_core_peak_ratio": float(metrics.get("side_to_core_peak_ratio", np.nan)),
        "hardware_reachable": bool(metrics.get("hardware_reachable", False)),
        "qa_status": str(metrics.get("qa_status", "")),
        "material_name": str(getattr(config.material, "name", "not_specified")),
        "material_refractive_index": float(getattr(config.material, "refractive_index", np.nan)),
        "material_model_status": "planning_proxy",
        "material_response_model": "incubation_threshold_proxy",
        "calibration_status": "uncalibrated",
        "threshold_source": "configured_placeholder",
        "incubation_model": "power_law_proxy",
        "incubation_coefficient": float(getattr(config.material, "incubation_exponent", np.nan)),
        "transmission_fraction": _transmission_fraction(config),
        "pulse_energy_uJ": float(config.energy.pulse_energy_in_J / BT_UJ),
        "xz_proxy_definition": axial_method,
        "xz_energy_conservation_status": xz_energy_status,
        "beam_family": "vortex_bessel" if int(design.ell) else "scalar_bessel",
        "model_level": "material_proxy",
        "generation_method": str(metrics.get("generation_method", "scalar_propagation")),
        "hardware_status": "lab_proxy" if bool(metrics.get("hardware_reachable", False)) else "unsupported",
        **xy,
        **{k: v for k, v in xz.items() if k != "radius_by_z_um"},
        "radius_by_z_um": xz["radius_by_z_um"],
        "fluence_xy_J_cm2": F_xy,
        "fluence_stack_J_cm2": fluence_stack,
        "centerline_fluence_xz_J_cm2": F_xz,
        "line_fluence_proxy_xz_J_cm2": line_proxy_xz,
        "axial_extent_method": axial_method,
    }


def material_proxy_row_from_geometry(geom: Mapping[str, Any], config: TwinConfig) -> dict[str, Any]:
    """Return one canonical Stage 6 material proxy row."""

    row = {
        "case_id": str(geom.get("case_id", "")),
        "preset": str(geom.get("preset", "fast")),
        "path": str(geom.get("path", "realistic")),
        "beam_family": str(geom.get("beam_family", "material_proxy")),
        "model_level": "material_proxy",
        "generation_method": str(geom.get("generation_method", "scalar_propagation")),
        "hardware_status": str(geom.get("hardware_status", "unsupported")),
        "qa_status": str(geom.get("qa_status", "exploratory") or "exploratory"),
        "material_name": str(geom.get("material_name", getattr(config.material, "name", "not_specified"))),
        "material_refractive_index": float(geom.get("material_refractive_index", getattr(config.material, "refractive_index", np.nan))),
        "material_model_status": "planning_proxy",
        "material_response_model": "incubation_threshold_proxy",
        "calibration_status": "uncalibrated",
        "threshold_source": "configured_placeholder",
        "threshold_fluence_J_cm2": float(geom.get("threshold_fluence_J_cm2", geom.get("threshold_J_cm2", np.nan))),
        "incubation_model": "power_law_proxy",
        "incubation_coefficient": float(getattr(config.material, "incubation_exponent", np.nan)),
        "pulse_count": float(geom.get("pulse_count", geom.get("effective_pulses", np.nan))),
        "pulse_energy_uJ": float(config.energy.pulse_energy_in_J / BT_UJ),
        "pulse_energy_at_sample_uJ": float(config.energy.pulse_energy_at_sample_J / BT_UJ),
        "transmission_fraction": float(geom.get("transmission_fraction", _transmission_fraction(config))),
        "peak_fluence_J_cm2": float(geom.get("peak_fluence_J_cm2", np.nan)),
        "fluence_to_threshold_ratio": float(geom.get("fluence_to_threshold_ratio", np.nan)),
        "thresholded_area_um2": float(geom.get("thresholded_proxy_area_um2", geom.get("xy_threshold_area_um2", np.nan))),
        "thresholded_equivalent_diameter_um": float(
            geom.get("thresholded_equivalent_diameter_um", geom.get("xy_threshold_equivalent_diameter_um", np.nan))
        ),
        "xz_proxy_length_um": float(geom.get("xz_proxy_length_um", geom.get("xz_threshold_length_um", np.nan))),
        "xz_proxy_definition": str(geom.get("xz_proxy_definition", "plane_normalized_centerline_fluence")),
        "xz_energy_conservation_status": str(geom.get("xz_energy_conservation_status", "normalised_visualisation")),
        "canonical_zone_um": float(geom.get("canonical_zone_um", geom.get("bessel_zone_um", np.nan))),
        "strict_bessel_region_um": float(geom.get("strict_bessel_region_um", np.nan)),
        "propagation_power_drift_fraction": geom.get("propagation_power_drift_fraction", np.nan),
        # Extra optical/planning comparison columns.
        "ell": int(geom.get("ell", 0)),
        "target_core_diameter_um": float(geom.get("target_core_diameter_um", np.nan)),
        "target_bessel_length_um": float(geom.get("target_bessel_length_um", np.nan)),
        "gamma_slm_deg": float(geom.get("gamma_slm_deg", np.nan)),
        "ring_radius_um": float(geom.get("ring_radius_um", np.nan)),
        "ring_width_um": float(geom.get("ring_width_um", np.nan)),
        "annular_threshold_proxy_ring_radius_um": geom.get("annular_threshold_proxy_ring_radius_um", np.nan),
        "annular_threshold_proxy_wall_thickness_um": geom.get("annular_threshold_proxy_wall_thickness_um", np.nan),
        "azimuthal_uniformity": geom.get("azimuthal_uniformity", np.nan),
        "threshold_taper_fraction": geom.get("threshold_taper_fraction", np.nan),
        "side_to_core_peak_ratio": geom.get("side_to_core_peak_ratio", np.nan),
        "hardware_reachable": bool(geom.get("hardware_reachable", False)),
        "output_category": "planning_proxy",
        "safe_for_design_comparison": True,
        "overclaim_guardrail": (
            "thresholded fluence proxy; not calibrated ablation, void, index-change, or weld prediction"
        ),
    }
    material_schema.annotate_material_row(row, qa_status=row["qa_status"])
    return material_schema.ordered_material_row(row)


def design_row_from_case(result: Mapping[str, Any], config: TwinConfig) -> dict[str, Any]:
    """Return one canonical material planning row with blank calibration fields."""

    geom = feature_geometry_from_result(result, config)
    row = material_proxy_row_from_geometry(geom, config)
    row.update({
        "measured_feature_diameter_um": pd.NA,
        "measured_feature_length_um": pd.NA,
        "measured_ring_wall_thickness_um": pd.NA,
        "measured_uniformity_note": "",
        "microscope_image_path": "",
    })
    return row


def build_shortlist_design_table(
    shortlist: Sequence[Mapping[str, Any]],
    base_config: TwinConfig | None = None,
    *,
    preset: str = "fast",
    path: PathKind = "realistic",
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Simulate the shortlist and return the Stage 6 material proxy table."""

    base = default_config(preset) if base_config is None else base_config
    rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for item in shortlist:
        cfg = replace(
            base,
            target=replace(
                base.target,
                ell=int(item["ell"]),
                target_core_diameter_m=float(item["D_um"]) * BT_UM,
                target_bessel_length_m=float(item["L_um"]) * BT_UM,
            ),
            energy=replace(base.energy, pulse_energy_in_J=float(item["Ein_uJ"]) * BT_UJ),
        )
        result = _bt().run_case(cfg, preset=preset, path=path, case_id=str(item["label"]))
        geom = feature_geometry_from_result(result, cfg)
        rows.append(design_row_from_case(result, cfg))
        cases.append({"config": cfg, "result": result, "geometry": geom, "spec": dict(item)})
    return material_schema.ordered_material_frame(rows), cases


def save_design_table(
    df: pd.DataFrame,
    output_dir: str | Path,
    filename: str = "material_application_design_table.csv",
) -> Path:
    """Save a Stage 6 material CSV and return the path."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    df.to_csv(path, index=False)
    return path


def calibration_template_from_proxy_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a calibration-template table with required measurement fields."""

    measurement_defaults = {
        "wavelength_nm": pd.NA,
        "pulse_duration_fs": pd.NA,
        "repetition_rate_Hz": pd.NA,
        "NA": pd.NA,
        "cone_angle_deg": pd.NA,
        "measured_threshold_fluence_J_cm2": pd.NA,
        "measured_line_width_um": pd.NA,
        "measured_line_depth_um": pd.NA,
        "microscope_or_etch_method": "",
        "measurement_uncertainty_um": pd.NA,
        "calibration_dataset_id": "",
        "calibration_notes": "",
    }
    rows: list[dict[str, Any]] = []
    for record in df.to_dict("records"):
        row = dict(record)
        row.update(measurement_defaults)
        row["case_id"] = f"{row.get('case_id', 'case')}_calibration_template"
        row["calibration_status"] = "uncalibrated"
        row["material_model_status"] = "planning_proxy"
        row["material_response_model"] = "incubation_threshold_proxy"
        row["output_category"] = "planning_proxy"
        material_schema.annotate_material_row(row, qa_status=str(row.get("qa_status", "exploratory")))
        rows.append(row)
    return material_schema.ordered_material_frame(rows)


def application_design_table_from_proxy_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return an application-facing planning table ranked for comparison only."""

    rows: list[dict[str, Any]] = []
    work = df.copy()
    if "fluence_to_threshold_ratio" in work:
        work = work.sort_values(["safe_for_design_comparison", "fluence_to_threshold_ratio"], ascending=[False, False])
    for rank, record in enumerate(work.to_dict("records"), start=1):
        row = dict(record)
        row["planning_rank"] = rank
        row["output_category"] = "planning_proxy"
        row["application_readiness"] = "compare_optical_fluence_proxy_only"
        row["design_table_note"] = (
            "safe for relative planning comparison; requires calibration before material-response claims"
        )
        material_schema.annotate_material_row(row, qa_status=str(row.get("qa_status", "exploratory")))
        rows.append(row)
    return material_schema.ordered_material_frame(rows)


def plot_feature_geometry(
    case: Mapping[str, Any],
    output_path: str | Path,
    *,
    caption: str | None = None,
) -> Path:
    """Save a threshold-contour plus axial-extent figure for one simulated case."""

    vbb_style.apply_style()
    result = case["result"]
    geom = case["geometry"]
    volume = result["volume"]
    grid = volume["crop_grid"]
    F_xy = np.asarray(geom["fluence_xy_J_cm2"], dtype=float)
    F_xz = np.asarray(geom["centerline_fluence_xz_J_cm2"], dtype=float)
    threshold = float(geom["threshold_J_cm2"])
    x_um = np.asarray(grid["x"], dtype=float) / BT_UM
    z_um = np.asarray(volume["z"], dtype=float) / BT_UM
    extent_xy = [float(x_um[0]), float(x_um[-1]), float(x_um[0]), float(x_um[-1])]
    extent_xz = [float(z_um[0]), float(z_um[-1]), float(x_um[0]), float(x_um[-1])]

    fig, ax = plt.subplots(1, 2, figsize=(10.8, 4.2))
    im0 = ax[0].imshow(
        vbb_style.display_scale(F_xy, gamma=0.55),
        origin="lower",
        extent=extent_xy,
        cmap=vbb_style.INTENSITY_CMAP,
    )
    if np.nanmin(F_xy) <= threshold <= np.nanmax(F_xy):
        ax[0].contour(x_um, x_um, F_xy, levels=[threshold], colors="white", linewidths=1.2)
    ax[0].set_title("XY threshold contour")
    ax[0].set_xlabel("x [um, sample plane]")
    ax[0].set_ylabel("y [um, sample plane]")
    cb0 = fig.colorbar(im0, ax=ax[0])
    cb0.set_label("display fluence, gamma=0.55 [a.u.]")

    xz_display = vbb_style.display_scale(F_xz, gamma=0.55)
    im1 = ax[1].imshow(xz_display, origin="lower", aspect="auto", extent=extent_xz, cmap=vbb_style.INTENSITY_CMAP)
    if np.nanmin(F_xz) <= threshold <= np.nanmax(F_xz):
        ax[1].contour(z_um, x_um, F_xz, levels=[threshold], colors="white", linewidths=1.2)
    ax[1].axvspan(float(geom["xz_threshold_start_um"]), float(geom["xz_threshold_end_um"]), color="white", alpha=0.10, lw=0)
    ax[1].set_title("XZ threshold extent")
    ax[1].set_xlabel("z [um, sample plane]")
    ax[1].set_ylabel("x [um, sample plane]")
    cb1 = fig.colorbar(im1, ax=ax[1])
    cb1.set_label("display line-fluence proxy, gamma=0.55 [a.u.]")

    fig.suptitle(str(geom.get("case_id", "feature geometry")))
    text = caption or (
        f"Threshold-contour feature proxy for {geom.get('case_id', 'case')}. "
        f"The white contour is the incubated threshold {threshold:.3g} J/cm^2; "
        f"the XZ panel uses plane-normalized centerline fluence, not a calibrated deposited-energy law."
    )
    path = vbb_style.save_figure(fig, output_path, text, metadata={"case_id": geom.get("case_id", ""), "threshold_J_cm2": threshold})
    plt.close(fig)
    return path


__all__ = [
    "build_shortlist_design_table",
    "application_design_table_from_proxy_summary",
    "calibration_template_from_proxy_summary",
    "design_row_from_case",
    "effective_pulses",
    "feature_geometry_from_result",
    "fluence_from_intensity",
    "incubated_threshold_J_cm2",
    "line_fluence_proxy_xz",
    "material_proxy_row_from_geometry",
    "centerline_fluence_xz_from_stack",
    "plane_fluence_stack_from_intensity",
    "plot_feature_geometry",
    "positive_intensity",
    "save_design_table",
    "threshold_mask",
    "xy_feature_geometry",
    "xz_feature_geometry",
]
