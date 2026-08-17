"""Localized polygonal vortex-ring beams.

This module is deliberately separate from `vbb_discrete`: the discrete module
builds kaleidoscope/lattice beams from a finite set of plane-wave deltas, while
this module builds one localized closed polygonal ring from a continuous
angular spectrum on the Bessel k-ring. I use an analytic caustic phase as the
seed and a small Fourier-projection refinement when I need the real-space ring
to be connected and uniform enough for fabrication-facing figures.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import vbb_materials, vbb_style
from vbb_study.config import BeamDesign, EPS as BT_EPS, TWOPI as BT_TWOPI, TwinConfig, um as BT_UM
from vbb_study.design import J0_FIRST_ZERO, compute_design_from_config
from vbb_study.equations.fields import (
    fft2c,
    gaussian_amplitude,
    ifft2c,
    make_rect_grid,
    make_xy_grid,
    phase_to_gray,
    phase_wrap,
    quantize_phase,
)
from vbb_study.equations.holography import fill_factor_amplitude as _fill_factor_amplitude
from vbb_study.equations.interface import interface_aberration_pupil, interface_correction_phase
from vbb_study.equations.propagation import focus_to_focal_plane
from vbb_study.facade import core as _bt

try:
    from scipy.ndimage import gaussian_filter1d, label as nd_label
except Exception:  # pragma: no cover
    gaussian_filter1d = None
    nd_label = None


def _fill_factor_amplitude_grid(grid: Mapping[str, Any], pixel_pitch_m: float, fill_factor: float) -> np.ndarray:
    return _fill_factor_amplitude(
        grid["X"],
        grid["Y"],
        pixel_pitch_m=pixel_pitch_m,
        fill_factor=fill_factor,
        sample_spacing_m=float(grid["dx"]),
    )


@dataclass(frozen=True)
class PolygonalVortexConfig:
    """Parameters for one localized polygonal vortex-ring design.

    All distances are SI. `flat_radius_m` is the side-to-centre apothem in the
    user's formula `r = R/cos(delta)`; the corner radius is larger by
    `1/cos(pi/N)`.
    """

    N: int = 6
    flat_radius_m: float = 7.0e-6
    ring_width_m: float = 1.0e-6
    kr_m_inv: float = 1.2e6
    charge: int = 2
    orientation_rad: float = 0.0
    corner_sharpness: float = 0.75
    k_ring_width_fraction: float = 0.08
    refinement_iterations: int = 80
    angular_samples: int = 4096
    phase_bits: int = 8


def wrap_to_period(angle_rad: np.ndarray | float, period_rad: float) -> np.ndarray:
    """Wrap an angle to `[-period/2, period/2)`."""

    return (np.asarray(angle_rad, dtype=float) + 0.5 * float(period_rad)) % float(period_rad) - 0.5 * float(period_rad)


def polygon_radius(phi_rad: np.ndarray, *, flat_radius_m: float, N: int = 6, orientation_rad: float = 0.0) -> np.ndarray:
    """Return the regular polygon radius `R/cos(delta)`.

    I use this as the target contour. For `N=6`, `flat_radius_m` is the radius
    to the middle of a flat side and the corners lie at
    `flat_radius_m/cos(pi/6)`.
    """

    sector = 2.0 * np.pi / int(N)
    delta = wrap_to_period(np.asarray(phi_rad, dtype=float) - float(orientation_rad), sector)
    return float(flat_radius_m) / np.maximum(np.cos(delta), 1.0e-12)


def target_polygon_ring_amplitude(grid: Mapping[str, Any], config: PolygonalVortexConfig) -> np.ndarray:
    """Return a one-ring target amplitude around the polygon contour."""

    r_target = polygon_radius(
        np.asarray(grid["PHI"], dtype=float),
        flat_radius_m=config.flat_radius_m,
        N=config.N,
        orientation_rad=config.orientation_rad,
    )
    distance = np.asarray(grid["R"], dtype=float) - r_target
    return np.exp(-0.5 * (distance / max(float(config.ring_width_m), BT_EPS)) ** 2)


def _support_from_polygon(config: PolygonalVortexConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return support function h(theta) for the polygon contour.

    The analytic stationary-phase seed uses the support-function form of the
    caustic. I derive it from the requested polar contour so the formula stays
    valid for N=3, 4, 6, and future rounded variants.
    """

    samples = int(config.angular_samples)
    phi = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    r = polygon_radius(phi, flat_radius_m=config.flat_radius_m, N=config.N, orientation_rad=config.orientation_rad)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    dphi = 2.0 * np.pi / samples
    dx = np.gradient(x, dphi, edge_order=2)
    dy = np.gradient(y, dphi, edge_order=2)
    nx = dy
    ny = -dx
    norm = np.maximum(np.hypot(nx, ny), BT_EPS)
    nx = nx / norm
    ny = ny / norm
    theta = np.mod(np.arctan2(ny, nx), 2.0 * np.pi)
    support = x * nx + y * ny
    order = np.argsort(theta)
    theta_sorted = theta[order]
    support_sorted = support[order]
    theta_uniform = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    support_uniform = np.interp(
        theta_uniform,
        np.r_[theta_sorted - 2.0 * np.pi, theta_sorted, theta_sorted + 2.0 * np.pi],
        np.r_[support_sorted, support_sorted, support_sorted],
    )
    sigma = max(0.0, 1.0 - float(config.corner_sharpness)) * 12.0
    if gaussian_filter1d is not None and sigma > 0.2:
        support_uniform = gaussian_filter1d(support_uniform, sigma=sigma, mode="wrap")
    return theta_uniform, support_uniform


def analytic_caustic_phase_table(config: PolygonalVortexConfig) -> dict[str, np.ndarray]:
    """Return the analytic angular phase seed for the polygonal caustic.

    For an angular-spectrum point at angle alpha, the stationary-phase line has
    normal angle theta = alpha - pi/2. If h(theta) is the support function of
    the target contour, then I use `dTheta/dalpha = k_r h(alpha-pi/2)`. The
    wrapped phase is allowed to have a branch jump, just as a vortex ramp does.
    """

    theta, support = _support_from_polygon(config)
    alpha = np.mod(theta + 0.5 * np.pi, 2.0 * np.pi)
    order = np.argsort(alpha)
    alpha = alpha[order]
    support = support[order]
    phase = np.zeros_like(alpha)
    # I remove the mean support before integrating so `Theta_hex` is periodic
    # and does not sneak in an extra topological charge. The vortex term
    # `l*phi` remains the only net winding.
    support_for_phase = support - float(np.mean(support))
    phase[1:] = np.cumsum(0.5 * (support_for_phase[:-1] + support_for_phase[1:]) * np.diff(alpha) * float(config.kr_m_inv))
    phase = phase - phase[0]
    return {"alpha_rad": alpha, "theta_phase_rad": phase, "support_m": support}


def _interp_periodic(angle: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    period = 2.0 * np.pi
    span = float(fp[-1] - fp[0] + (fp[1] - fp[0] if len(fp) > 1 else 0.0))
    return np.interp(
        np.mod(angle, period).ravel(),
        np.r_[xp - period, xp, xp + period],
        np.r_[fp - span, fp, fp + span],
    ).reshape(np.shape(angle))


def angular_ring_support(grid: Mapping[str, Any], config: PolygonalVortexConfig) -> np.ndarray:
    """Return a Gaussian radial support on the continuous Bessel k-ring."""

    K = 2.0 * np.pi * np.hypot(np.asarray(grid["FX"], dtype=float), np.asarray(grid["FY"], dtype=float))
    sigma = max(float(config.k_ring_width_fraction) * float(config.kr_m_inv), BT_EPS)
    return np.exp(-0.5 * ((K - float(config.kr_m_inv)) / sigma) ** 2)


def analytic_angular_spectrum(grid: Mapping[str, Any], config: PolygonalVortexConfig) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return the analytic continuous-ring angular spectrum seed."""

    table = analytic_caustic_phase_table(config)
    kphi = np.mod(np.arctan2(np.asarray(grid["FY"], dtype=float), np.asarray(grid["FX"], dtype=float)), 2.0 * np.pi)
    theta_phase = _interp_periodic(kphi, table["alpha_rad"], table["theta_phase_rad"])
    support = angular_ring_support(grid, config)
    spectrum = support * np.exp(1j * (int(config.charge) * kphi + theta_phase))
    return spectrum, table


def refined_polygonal_field(
    grid: Mapping[str, Any],
    config: PolygonalVortexConfig,
) -> dict[str, Any]:
    """Build a localized polygonal vortex field from a continuous k-ring.

    I start from the analytic caustic phase, then alternate between the desired
    real-space polygonal ring amplitude and the continuous k-ring support. This
    is the small generalized-axicon refinement step that removes the worst
    corner hot spots without turning the beam into a discrete lattice.
    """

    target = target_polygon_ring_amplitude(grid, config)
    support = angular_ring_support(grid, config)
    spectrum, table = analytic_angular_spectrum(grid, config)
    convergence: list[dict[str, float]] = []
    target_norm = target / (float(np.max(target)) + BT_EPS)
    for iteration in range(int(max(0, config.refinement_iterations))):
        field = ifft2c(spectrum)
        constrained = target * np.exp(1j * np.angle(field))
        next_spectrum = fft2c(constrained)
        spectrum = support * np.exp(1j * np.angle(next_spectrum))
        if iteration == 0 or (iteration + 1) % 10 == 0 or iteration + 1 == int(config.refinement_iterations):
            image = np.abs(ifft2c(spectrum))
            image = image / (float(np.max(image)) + BT_EPS)
            error = float(np.sqrt(np.mean((image - target_norm) ** 2)))
            convergence.append({"iteration": float(iteration + 1), "amplitude_rmse": error})
    field = ifft2c(spectrum)
    return {
        "field": field,
        "spectrum": spectrum,
        "target_amplitude": target,
        "ring_support": support,
        "analytic_phase_table": table,
        "convergence": convergence,
    }


def _bilinear_sample(image: np.ndarray, grid: Mapping[str, Any], xs_m: np.ndarray, ys_m: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    x = np.asarray(grid["x"], dtype=float)
    dx = float(grid["dx"])
    fx = (np.asarray(xs_m, dtype=float) - float(x[0])) / dx
    fy = (np.asarray(ys_m, dtype=float) - float(x[0])) / dx
    x0 = np.floor(fx).astype(int)
    y0 = np.floor(fy).astype(int)
    x0 = np.clip(x0, 0, arr.shape[1] - 2)
    y0 = np.clip(y0, 0, arr.shape[0] - 2)
    tx = np.clip(fx - x0, 0.0, 1.0)
    ty = np.clip(fy - y0, 0.0, 1.0)
    return (
        (1.0 - tx) * (1.0 - ty) * arr[y0, x0]
        + tx * (1.0 - ty) * arr[y0, x0 + 1]
        + (1.0 - tx) * ty * arr[y0 + 1, x0]
        + tx * ty * arr[y0 + 1, x0 + 1]
    )


def polygon_contour_samples(config: PolygonalVortexConfig, *, samples: int = 720) -> dict[str, np.ndarray]:
    phi = np.linspace(0.0, 2.0 * np.pi, int(samples), endpoint=False)
    radius = polygon_radius(phi, flat_radius_m=config.flat_radius_m, N=config.N, orientation_rad=config.orientation_rad)
    return {"phi_rad": phi, "radius_m": radius, "x_m": radius * np.cos(phi), "y_m": radius * np.sin(phi)}


def phase_winding(field: np.ndarray, grid: Mapping[str, Any], radius_m: float, *, samples: int = 720) -> float:
    contour = {
        "phi_rad": np.linspace(0.0, 2.0 * np.pi, int(samples), endpoint=False),
    }
    phi = contour["phi_rad"]
    vals = _bilinear_sample(field, grid, float(radius_m) * np.cos(phi), float(radius_m) * np.sin(phi))
    phase = np.unwrap(np.angle(vals))
    return float((phase[-1] - phase[0]) / (2.0 * np.pi))


def _autocorrelation_side_peak(image: np.ndarray, grid: Mapping[str, Any], exclusion_radius_m: float) -> float:
    arr = np.asarray(image, dtype=float)
    arr = arr - float(np.mean(arr))
    corr = np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(arr)) ** 2).real)
    corr = corr / (float(np.max(corr)) + BT_EPS)
    N = corr.shape[0]
    x = (np.arange(N) - N // 2) * float(grid["dx"])
    X, Y = np.meshgrid(x, x, indexing="xy")
    mask = np.hypot(X, Y) >= float(exclusion_radius_m)
    return float(np.max(corr[mask])) if np.any(mask) else np.nan


def polygonal_ring_metrics(
    field_or_intensity: np.ndarray,
    grid: Mapping[str, Any],
    config: PolygonalVortexConfig,
    *,
    complex_field: np.ndarray | None = None,
    threshold_fraction: float = 0.35,
) -> dict[str, Any]:
    """Measure whether a field is one localized polygonal vortex ring."""

    if np.iscomplexobj(field_or_intensity):
        field = np.asarray(field_or_intensity, dtype=complex)
        intensity = np.abs(field) ** 2
    else:
        field = complex_field
        intensity = np.asarray(field_or_intensity, dtype=float)
    peak = float(np.max(intensity)) + BT_EPS
    norm = intensity / peak
    mask = norm >= float(threshold_fraction)
    if nd_label is not None:
        labels, count = nd_label(mask)
        areas = np.bincount(labels.ravel())[1:]
    else:  # pragma: no cover
        count = int(np.any(mask))
        areas = np.asarray([np.count_nonzero(mask)], dtype=int)
    largest = int(np.max(areas)) if areas.size else 0
    largest_fraction = float(largest / (np.sum(areas) + BT_EPS)) if areas.size else 0.0

    corner_radius = float(config.flat_radius_m / np.cos(np.pi / int(config.N)))
    surround_radius = 1.75 * corner_radius + 3.0 * config.ring_width_m
    R = np.asarray(grid["R"], dtype=float)
    localization = float(np.sum(intensity[R <= surround_radius]) / (np.sum(intensity) + BT_EPS))
    lattice_side_peak = _autocorrelation_side_peak(norm, grid, exclusion_radius_m=2.2 * corner_radius)

    contour = polygon_contour_samples(config, samples=720)
    contour_values = _bilinear_sample(norm, grid, contour["x_m"], contour["y_m"])
    centered = contour_values - float(np.mean(contour_values))
    coeff = np.fft.rfft(centered)
    amps = np.abs(coeff)
    if amps.size:
        amps[0] = 0.0
    expected = int(config.N)
    order_amp = float(amps[expected]) if expected < amps.size else 0.0
    other = np.array(amps, copy=True)
    if expected < other.size:
        other[expected] = 0.0
    contour_order_fidelity = float(order_amp / (float(np.max(other)) + order_amp + BT_EPS))

    radii = contour["radius_m"]
    side_phase = np.mod(contour["phi_rad"] - config.orientation_rad, 2.0 * np.pi / int(config.N))
    side_distance = np.abs(wrap_to_period(side_phase, 2.0 * np.pi / int(config.N)))
    corner_mask = side_distance >= 0.42 * (np.pi / int(config.N))
    edge_mask = side_distance <= 0.18 * (np.pi / int(config.N))
    corner_mean = float(np.mean(contour_values[corner_mask])) if np.any(corner_mask) else np.nan
    edge_mean = float(np.mean(contour_values[edge_mask])) if np.any(edge_mask) else np.nan
    edge_uniformity = float(1.0 / (1.0 + np.std(contour_values) / (np.mean(contour_values) + BT_EPS)))

    center = norm[norm.shape[0] // 2, norm.shape[1] // 2]
    core_null_depth = float(center / (np.max(contour_values) + BT_EPS))
    winding = (
        phase_winding(field, grid, radius_m=0.45 * float(config.flat_radius_m))
        if field is not None
        else np.nan
    )
    return {
        "component_count": int(count),
        "largest_component_fraction": largest_fraction,
        "single_closed_contour_pass": bool(count == 1 and largest_fraction > 0.98),
        "localization_fraction": localization,
        "localization_pass": bool(localization >= 0.85),
        "lattice_autocorr_side_peak": lattice_side_peak,
        "no_lattice_periodicity_pass": bool(lattice_side_peak <= 0.55),
        "core_null_depth": core_null_depth,
        "dark_core_pass": bool(core_null_depth <= 1.0e-2),
        "phase_winding": winding,
        "oam_charge_error": float(abs(winding - int(config.charge))) if np.isfinite(winding) else np.nan,
        "oam_pass": bool(np.isfinite(winding) and abs(winding - int(config.charge)) <= 0.25),
        "contour_order": int(expected),
        "contour_order_fidelity": contour_order_fidelity,
        "sixfold_ring_pass": bool(contour_order_fidelity >= 0.45),
        "edge_uniformity": edge_uniformity,
        "corner_to_edge_fluence_ratio": float(corner_mean / (edge_mean + BT_EPS)) if np.isfinite(corner_mean) and np.isfinite(edge_mean) else np.nan,
        "flat_radius_um": float(config.flat_radius_m / BT_UM),
        "corner_radius_um": float(corner_radius / BT_UM),
        "surround_radius_um": float(surround_radius / BT_UM),
        "threshold_fraction": float(threshold_fraction),
    }


def hollow_hex_side_lobe_metrics(
    field_or_intensity: np.ndarray,
    grid: Mapping[str, Any],
    config: PolygonalVortexConfig,
    *,
    guard_widths: float = 1.75,
    ring_widths_for_mean: float = 0.75,
    outer_margin_widths: float = 8.0,
    core_radius_fraction: float = 0.5,
    core_clearance_widths: float = 2.5,
) -> dict[str, float]:
    """Measure halo and core leakage around the requested hollow polygon ring.

    The side-lobe mask is tied to the requested polygon contour. If a lab
    encoding makes a bright ring at the wrong radius, this metric should fail:
    that is a geometry error, not merely a harmless shift.
    """

    if np.iscomplexobj(field_or_intensity):
        intensity = np.abs(np.asarray(field_or_intensity, dtype=complex)) ** 2
    else:
        intensity = np.asarray(field_or_intensity, dtype=float)
    peak = float(np.max(intensity)) + BT_EPS
    norm = intensity / peak

    R = np.asarray(grid["R"], dtype=float)
    PHI = np.asarray(grid["PHI"], dtype=float)
    r_target = polygon_radius(PHI, flat_radius_m=config.flat_radius_m, N=config.N, orientation_rad=config.orientation_rad)
    ring_distance = np.abs(R - r_target)
    ring_guard = ring_distance <= float(guard_widths) * float(config.ring_width_m)
    ring_mask = ring_distance <= float(ring_widths_for_mean) * float(config.ring_width_m)

    core_radius = max(
        float(core_radius_fraction) * float(config.flat_radius_m),
        float(config.flat_radius_m) - float(core_clearance_widths) * float(config.ring_width_m),
    )
    core_mask = R <= core_radius
    corner_radius = float(config.flat_radius_m / np.cos(np.pi / int(config.N)))
    eval_mask = R <= (corner_radius + float(outer_margin_widths) * float(config.ring_width_m))
    side_mask = eval_mask & (~ring_guard) & (~core_mask)

    peak_idx = np.unravel_index(int(np.argmax(intensity)), intensity.shape)
    return {
        "side_lobe_peak_ratio": float(np.max(norm[side_mask])) if np.any(side_mask) else np.nan,
        "core_peak_ratio": float(np.max(norm[core_mask])) if np.any(core_mask) else np.nan,
        "ring_energy_fraction": float(np.sum(intensity[ring_guard]) / (np.sum(intensity[eval_mask]) + BT_EPS))
        if np.any(eval_mask)
        else np.nan,
        "ring_mean_fraction": float(np.mean(norm[ring_mask])) if np.any(ring_mask) else np.nan,
        "side_lobe_energy_fraction": float(np.sum(intensity[side_mask]) / (np.sum(intensity[eval_mask]) + BT_EPS))
        if np.any(eval_mask)
        else np.nan,
        "peak_radius_um": float(R[peak_idx] / BT_UM),
        "target_flat_radius_um": float(config.flat_radius_m / BT_UM),
        "target_corner_radius_um": float(corner_radius / BT_UM),
        "core_mask_radius_um": float(core_radius / BT_UM),
    }


def _extract_refined_phase_table(grid: Mapping[str, Any], spectrum: np.ndarray, config: PolygonalVortexConfig) -> dict[str, np.ndarray]:
    samples = int(config.angular_samples)
    alpha = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    K = float(config.kr_m_inv)
    fx = (K / (2.0 * np.pi)) * np.cos(alpha)
    fy = (K / (2.0 * np.pi)) * np.sin(alpha)
    phase = np.angle(_bilinear_sample(spectrum, grid, fx, fy))
    theta = np.unwrap(phase - int(config.charge) * alpha)
    theta = theta - theta[0]
    return {"alpha_rad": alpha, "theta_phase_rad": theta}


def build_ideal_case(
    grid: Mapping[str, Any],
    config: PolygonalVortexConfig,
    *,
    wavelength_m: float,
    n_medium: float,
    z_values_m: Sequence[float],
    crop_pixels: int | None = None,
) -> dict[str, Any]:
    """Build and propagate the ideal continuous-ring polygonal vortex beam."""

    refined = refined_polygonal_field(grid, config)
    volume = _bt().propagate_volume(
        refined["field"],
        dict(grid),
        wavelength_m,
        z_values_m,
        n_medium=n_medium,
        crop_pixels=crop_pixels,
        bandlimit=True,
        method="bl_asm",
    )
    metrics = polygonal_ring_metrics(refined["field"], grid, config)
    refined_phase = _extract_refined_phase_table(grid, refined["spectrum"], config)
    return {
        "path": "ideal",
        "grid": dict(grid),
        "config": config,
        "field": refined["field"],
        "spectrum": refined["spectrum"],
        "target_amplitude": refined["target_amplitude"],
        "analytic_phase_table": refined["analytic_phase_table"],
        "refined_phase_table": refined_phase,
        "convergence": refined["convergence"],
        "volume": volume,
        "metrics": metrics,
        "encoded_power_fraction": 1.0,
    }


def _pad_rect_to_square(U_rect: np.ndarray, N: int) -> np.ndarray:
    out = np.zeros((int(N), int(N)), dtype=U_rect.dtype)
    ny, nx = U_rect.shape
    y0 = (int(N) - ny) // 2
    x0 = (int(N) - nx) // 2
    if y0 < 0 or x0 < 0:
        raise ValueError("square grid is too small for the rectangular SLM field")
    out[y0 : y0 + ny, x0 : x0 + nx] = U_rect
    return out


def _slm_polygonal_phase(grid: Mapping[str, Any], design: BeamDesign, config: PolygonalVortexConfig, phase_table: Mapping[str, np.ndarray]) -> np.ndarray:
    theta = _interp_periodic(np.asarray(grid["PHI"], dtype=float), np.asarray(phase_table["alpha_rad"]), np.asarray(phase_table["theta_phase_rad"]))
    return -float(design.kr_slm_m_inv) * np.asarray(grid["R"], dtype=float) + int(config.charge) * np.asarray(grid["PHI"], dtype=float) + theta


def build_lab_realistic_case(
    twin_config: TwinConfig,
    config: PolygonalVortexConfig,
    phase_table: Mapping[str, np.ndarray],
    *,
    z_values_m: Sequence[float],
    crop_pixels: int | None = None,
) -> dict[str, Any]:
    """Build a lab-realistic SLM/focal-plane version of the polygonal ring."""

    target_core = 2.0 * J0_FIRST_ZERO / max(float(config.kr_m_inv), BT_EPS)
    cfg = replace(
        twin_config,
        target=replace(
            twin_config.target,
            ell=int(config.charge),
            target_core_diameter_m=target_core,
            target_bessel_length_m=twin_config.target.target_bessel_length_m,
        ),
    )
    design = compute_design_from_config(cfg)
    ds = max(1, int(cfg.grid.device_downsample))
    nx = int(np.ceil(cfg.slm.resolution_x / ds))
    ny = int(np.ceil(cfg.slm.resolution_y / ds))
    rect_grid = make_rect_grid(nx, ny, cfg.slm.pixel_pitch_m * ds)
    phase = _slm_polygonal_phase(rect_grid, design, config, phase_table)
    if cfg.include_blaze:
        phase = phase + BT_TWOPI * cfg.slm.carrier_cpm * np.asarray(rect_grid["X"], dtype=float)
    phase_wrapped = phase_wrap(phase)
    phase_quantized = quantize_phase(phase, int(config.phase_bits))
    amp = gaussian_amplitude(rect_grid["R"], cfg.laser.beam_radius_on_slm_m)
    fill = _fill_factor_amplitude_grid(rect_grid, cfg.slm.pixel_pitch_m, cfg.slm.fill_factor) if cfg.include_fill_factor else np.ones_like(amp)
    aperture = (
        (np.abs(rect_grid["X"]) <= 0.5 * cfg.slm.active_width_m)
        & (np.abs(rect_grid["Y"]) <= 0.5 * cfg.slm.active_height_m)
    ).astype(float)
    U_rect = amp * fill * aperture * np.exp(1j * (phase_quantized if cfg.include_quantization else phase_wrapped))
    pupil_grid = make_xy_grid(cfg.grid.N, rect_grid["dx"])
    U = _pad_rect_to_square(U_rect, int(cfg.grid.N))
    if cfg.include_first_order_isolation and cfg.include_blaze:
        filter_geometry = _bt().first_order_filter_geometry(pupil_grid, cfg.slm, design)
        order = _bt().isolate_first_order(
            U,
            pupil_grid,
            cfg.slm,
            filter_radius_lpmm=filter_geometry["effective_filter_radius_lpmm"],
        )
        order.update(filter_geometry)
        U = order["U_selected"]
    else:
        order = {"selected_fraction": 1.0, "order_mask": np.ones_like(U, dtype=bool)}
    pupil = (pupil_grid["R"] <= cfg.objective.pupil_radius_m).astype(float)
    U = U * pupil
    W = np.zeros_like(U, dtype=float)
    if cfg.apply_interface:
        W = interface_aberration_pupil(pupil_grid, cfg.laser, cfg.objective, cfg.material)
        if cfg.correct_interface:
            W = W + interface_correction_phase(pupil_grid, cfg.laser, cfg.objective, cfg.material)
        U = U * np.exp(1j * W)
    U_focus, focal_grid = focus_to_focal_plane(U, pupil_grid, cfg.laser, cfg.objective)
    volume = _bt().propagate_volume(
        U_focus,
        focal_grid,
        cfg.laser.wavelength_m,
        z_values_m,
        n_medium=cfg.material.refractive_index,
        crop_pixels=crop_pixels or cfg.grid.crop_pixels,
        bandlimit=True,
        method=cfg.propagation.method,
        propagation_config=cfg.propagation,
    )
    crop_grid = volume["crop_grid"]
    c = int(focal_grid["N"] // 2)
    h = int(crop_grid["N"] // 2)
    sl = slice(c - h, c + h)
    focus_crop = U_focus[sl, sl]
    metrics = polygonal_ring_metrics(focus_crop, crop_grid, config, complex_field=focus_crop)
    gray = phase_to_gray(phase_quantized, int(config.phase_bits), invert=cfg.slm.invert_gray)
    return {
        "path": "lab_realistic",
        "config": config,
        "twin_config": cfg,
        "design": design,
        "field": focus_crop,
        "full_focus_field": U_focus,
        "focal_grid": focal_grid,
        "grid": crop_grid,
        "pupil_grid": pupil_grid,
        "phase_wrapped": phase_wrapped,
        "phase_quantized": phase_quantized,
        "gray": gray,
        "volume": volume,
        "metrics": metrics,
        "encoded_power_fraction": float(order.get("selected_fraction", np.nan)),
        "order": order,
    }


def propagation_stability(case: Mapping[str, Any], config: PolygonalVortexConfig) -> pd.DataFrame:
    """Return polygon-ring metrics for each propagated z plane."""

    rows: list[dict[str, Any]] = []
    volume = case["volume"]
    for idx, z_m in enumerate(np.asarray(volume["z"], dtype=float)):
        metrics = polygonal_ring_metrics(np.asarray(volume["intensity_stack"][idx], dtype=float), volume["crop_grid"], config)
        rows.append(
            {
                "z_um": float(z_m / BT_UM),
                "localization_fraction": metrics["localization_fraction"],
                "component_count": metrics["component_count"],
                "contour_order_fidelity": metrics["contour_order_fidelity"],
                "core_null_depth": metrics["core_null_depth"],
            }
        )
    return pd.DataFrame(rows)


def material_hex_ring_metrics(case: Mapping[str, Any], pulse_energy_J: float, threshold_J_cm2: float) -> dict[str, Any]:
    """Map the polygonal ring to a thresholded hexagonal feature proxy."""

    if "volume" in case:
        plane = np.asarray(case["volume"]["planes"]["peak"], dtype=float)
        grid = case["volume"]["crop_grid"]
    else:
        plane = np.abs(case["field"]) ** 2
        grid = case["grid"]
    F = vbb_materials.fluence_from_intensity(plane, float(grid["dx"]), float(pulse_energy_J))
    mask = F >= float(threshold_J_cm2)
    contour = polygon_contour_samples(case["config"], samples=720)
    contour_f = _bilinear_sample(F, grid, contour["x_m"], contour["y_m"])
    phase = np.mod(contour["phi_rad"] - case["config"].orientation_rad, 2.0 * np.pi / int(case["config"].N))
    edge = np.abs(wrap_to_period(phase, 2.0 * np.pi / int(case["config"].N))) <= 0.18 * (np.pi / int(case["config"].N))
    corner = np.abs(wrap_to_period(phase, 2.0 * np.pi / int(case["config"].N))) >= 0.42 * (np.pi / int(case["config"].N))
    return {
        "threshold_area_um2": float(np.count_nonzero(mask) * float(grid["dx"]) ** 2 / (BT_UM**2)),
        "edge_fluence_uniformity": float(1.0 / (1.0 + np.std(contour_f) / (np.mean(contour_f) + BT_EPS))),
        "corner_to_edge_fluence_ratio": float(np.mean(contour_f[corner]) / (np.mean(contour_f[edge]) + BT_EPS)),
        "dark_core_radius_um": float(0.45 * case["config"].flat_radius_m / BT_UM),
        "fluence_xy_J_cm2": F,
        "threshold_mask": mask,
    }


def export_lab_cgh(case: Mapping[str, Any], output_dir: str | Path, *, label: str = "hexagonal_vortex_ring") -> dict[str, Path]:
    """Export the lab phase map and params JSON for SLM display."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    png_path = out / f"11_polygonal_{label}.png"
    json_path = out / f"11_polygonal_{label}.json"
    try:
        from PIL import Image

        Image.fromarray(np.asarray(case["gray"], dtype=np.uint8), mode="L").save(png_path)
    except Exception:  # pragma: no cover
        plt.imsave(png_path, np.asarray(case["gray"], dtype=np.uint8), cmap="gray", vmin=0, vmax=255)
    payload = {
        "kind": "localized polygonal vortex ring",
        "not_the_default_discrete_lattice": True,
        "polygon_config": asdict(case["config"]),
        "encoded_power_fraction": float(case.get("encoded_power_fraction", np.nan)),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"phase_png": png_path, "params_json": json_path}


def plot_hex_ring_checkpoint(ideal: Mapping[str, Any], lab: Mapping[str, Any], output_path: str | Path) -> Path:
    """Save ideal/lab XY and XZ checkpoint panels."""

    vbb_style.apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.0), constrained_layout=True)
    cases = [ideal, lab]
    labels = ["ideal continuous ring", "lab-realistic SLM"]
    xy_artist = None
    xz_artist = None
    for col, (case, label) in enumerate(zip(cases, labels)):
        I = np.abs(case["field"]) ** 2
        grid = case["grid"]
        x_um = np.asarray(grid["x"], dtype=float) / BT_UM
        xy_artist = axes[0, col].imshow(
            vbb_style.display_scale(I, gamma=0.45),
            origin="lower",
            extent=[float(x_um[0]), float(x_um[-1]), float(x_um[0]), float(x_um[-1])],
            cmap=vbb_style.INTENSITY_CMAP,
            vmin=0.0,
            vmax=1.0,
        )
        contour = polygon_contour_samples(case["config"])
        axes[0, col].plot(contour["x_m"] / BT_UM, contour["y_m"] / BT_UM, color="white", lw=0.9, alpha=0.9)
        axes[0, col].set_title(label)
        axes[0, col].set_xlabel("x [um, sample plane]")
        axes[0, col].set_ylabel("y [um, sample plane]")
        vol = case["volume"]
        gx = np.asarray(vol["crop_grid"]["x"], dtype=float) / BT_UM
        z_um = np.asarray(vol["z"], dtype=float) / BT_UM
        xz = np.asarray(vol["xz"], dtype=float)
        xz_artist = axes[1, col].imshow(
            vbb_style.display_scale(xz, gamma=0.45),
            origin="lower",
            aspect="auto",
            extent=[float(z_um[0]), float(z_um[-1]), float(gx[0]), float(gx[-1])],
            cmap=vbb_style.INTENSITY_CMAP,
            vmin=0.0,
            vmax=1.0,
            interpolation="bilinear",
        )
        axes[1, col].set_xlabel("z [um, sample plane]")
        axes[1, col].set_ylabel("x [um, sample plane]")
    cb0 = fig.colorbar(xy_artist, ax=axes[0, :], shrink=0.9)
    cb0.set_label("display intensity, gamma=0.45 [a.u.]")
    cb1 = fig.colorbar(xz_artist, ax=axes[1, :], shrink=0.9)
    cb1.set_label("XZ display intensity, gamma=0.45 [a.u.]")
    caption = (
        "Localized hexagonal vortex-ring checkpoint. The white contour is the requested regular hexagon; "
        "the panels show one continuous ring with a dark vortex core rather than the old tiled N-wave lattice."
    )
    out = vbb_style.save_figure(fig, output_path, caption, metadata={"figure": "polygonal_hex_ring_checkpoint"})
    plt.close(fig)
    return out


__all__ = [
    "PolygonalVortexConfig",
    "analytic_caustic_phase_table",
    "analytic_angular_spectrum",
    "build_ideal_case",
    "build_lab_realistic_case",
    "export_lab_cgh",
    "hollow_hex_side_lobe_metrics",
    "material_hex_ring_metrics",
    "phase_winding",
    "plot_hex_ring_checkpoint",
    "polygon_radius",
    "polygon_contour_samples",
    "polygonal_ring_metrics",
    "propagation_stability",
    "refined_polygonal_field",
    "target_polygon_ring_amplitude",
]
