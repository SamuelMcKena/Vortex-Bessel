"""Tolerance sweep helpers for the Stage 7 vector-arm study."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from vbb_study.vector_arm_chain import (
    default_vector_arm_grid,
    headless_angle_delta,
    local_polarization_angle,
    run_vector_arm,
)
from vbb_study.vector_arm_config import SLMPanelConfig, VectorArmConfig
from vbb_study.vector_arm_metrics import h6_from_intensity, sample_ring_profile

TWOPI = 2.0 * np.pi
EPS = 1.0e-30


def _ring_radius(cfg: VectorArmConfig) -> float:
    return 0.5 * float(cfg.waist_m)


def _stokes_complex_on_ring(field: Any, grid: Mapping[str, Any], radius_m: float, samples: int = 2048) -> np.ndarray:
    stokes = field.stokes()
    profile_s1 = sample_ring_profile(stokes["S1"], grid, radius_m, angular_samples=samples)
    profile_s2 = sample_ring_profile(stokes["S2"], grid, radius_m, angular_samples=samples)
    return profile_s1 + 1j * profile_s2


def measure_pattern_rotation(
    reference_field: Any,
    shifted_field: Any,
    grid: Mapping[str, Any],
    radius_m: float,
    *,
    samples: int = 2048,
) -> float:
    """Measure headless Stokes-pattern rotation by circular cross-correlation."""

    ref = _stokes_complex_on_ring(reference_field, grid, radius_m, samples=samples)
    cur = _stokes_complex_on_ring(shifted_field, grid, radius_m, samples=samples)
    corr = np.fft.ifft(np.fft.fft(cur) * np.conj(np.fft.fft(ref)))
    idx = int(np.argmax(np.real(corr)))
    if idx > samples // 2:
        idx -= samples
    return float(idx * TWOPI / samples)


def delta_rotation_sweep(
    cfg: VectorArmConfig | None = None,
    *,
    deltas: Sequence[float] = tuple(np.linspace(0.0, np.pi, 13)),
    grid: Mapping[str, Any] | None = None,
) -> list[dict[str, float]]:
    """Sweep SLM2 piston and measure the delta/2 rotation law."""

    cfg = cfg or VectorArmConfig(ideal_components=True)
    grid_dict = dict(default_vector_arm_grid(cfg) if grid is None else grid)
    base_cfg = replace(cfg, piston_delta_rad=0.0)
    base = run_vector_arm(base_cfg, grid=grid_dict)
    base_angle = local_polarization_angle(base)
    radius = _ring_radius(cfg)
    ring_mask = (np.asarray(grid_dict["R"], dtype=float) > 0.9 * radius) & (
        np.asarray(grid_dict["R"], dtype=float) < 1.1 * radius
    )
    rows = []
    for delta in deltas:
        swept = run_vector_arm(replace(cfg, piston_delta_rad=float(delta)), grid=grid_dict)
        delta_angle = headless_angle_delta(local_polarization_angle(swept)[ring_mask], base_angle[ring_mask])
        measured = float(np.angle(np.mean(np.exp(2j * delta_angle))) / 2.0)
        expected = 0.5 * float(delta)
        rows.append(
            {
                "delta_rad": float(delta),
                "measured_rotation_rad": measured,
                "expected_rotation_rad": expected,
                "error_rad": float(measured - expected),
            }
        )
    return rows


def sector_rotation_sweep(
    cfg: VectorArmConfig | None = None,
    *,
    rotations_rad: Iterable[float] = np.deg2rad(np.linspace(-10.0, 10.0, 9)),
    grid: Mapping[str, Any] | None = None,
) -> list[dict[str, float]]:
    """Sweep sector-registration error and report QWP-output H6."""

    cfg = cfg or VectorArmConfig(ideal_components=True)
    grid_dict = dict(default_vector_arm_grid(cfg) if grid is None else grid)
    rows = []
    for rot in rotations_rad:
        local_cfg = replace(cfg, sector_rotation_rad=float(rot))
        field = run_vector_arm(local_cfg, grid=grid_dict)
        h6 = h6_from_intensity(field.intensity, grid_dict, _ring_radius(local_cfg), angular_samples=2048).h6
        rows.append({"sector_rotation_rad": float(rot), "sector_rotation_deg": float(np.rad2deg(rot)), "h6": float(h6)})
    return rows


def fill_factor_sweep(
    cfg: VectorArmConfig | None = None,
    *,
    fill_factors: Sequence[float] = (1.0, 0.93, 0.85),
    grid: Mapping[str, Any] | None = None,
) -> list[dict[str, float]]:
    """Sweep SLM fill factor at the QWP output."""

    cfg = cfg or VectorArmConfig()
    grid_dict = dict(default_vector_arm_grid(cfg) if grid is None else grid)
    rows = []
    for ff in fill_factors:
        slm1 = replace(cfg.slm1, fill_factor=float(ff))
        slm2 = replace(cfg.slm2, fill_factor=float(ff))
        field = run_vector_arm(replace(cfg, slm1=slm1, slm2=slm2), grid=grid_dict)
        h6 = h6_from_intensity(field.intensity, grid_dict, _ring_radius(cfg), angular_samples=2048).h6
        rows.append({"fill_factor": float(ff), "h6": float(h6), "iris": 0.0})
    return rows


def quantisation_sweep(
    cfg: VectorArmConfig | None = None,
    *,
    levels: Sequence[int | None] = (None, 256, 64),
    grid: Mapping[str, Any] | None = None,
) -> list[dict[str, float | int | str]]:
    """Sweep quantisation off/on at requested phase levels."""

    cfg = cfg or VectorArmConfig()
    grid_dict = dict(default_vector_arm_grid(cfg) if grid is None else grid)
    rows = []
    for level in levels:
        quantise = level is not None
        slm1 = cfg.slm1 if level is None else replace(cfg.slm1, phase_levels=int(level))
        slm2 = cfg.slm2 if level is None else replace(cfg.slm2, phase_levels=int(level))
        field = run_vector_arm(replace(cfg, quantise=quantise, slm1=slm1, slm2=slm2), grid=grid_dict)
        h6 = h6_from_intensity(field.intensity, grid_dict, _ring_radius(cfg), angular_samples=2048).h6
        rows.append({"phase_levels": "off" if level is None else int(level), "quantise": int(quantise), "h6": float(h6)})
    return rows


def retardance_sweep(
    cfg: VectorArmConfig | None = None,
    *,
    errors_rad: Sequence[float] = (0.0, 2.0 * np.pi / 100.0, 2.0 * np.pi / 50.0),
    grid: Mapping[str, Any] | None = None,
) -> list[dict[str, float]]:
    """Sweep HWP/QWP retardance errors."""

    cfg = cfg or VectorArmConfig()
    grid_dict = dict(default_vector_arm_grid(cfg) if grid is None else grid)
    rows = []
    for err in errors_rad:
        field = run_vector_arm(replace(cfg, hwp_retardance_error_rad=float(err), qwp_retardance_error_rad=float(err)), grid=grid_dict)
        psi = local_polarization_angle(field)
        h6 = h6_from_intensity(field.intensity, grid_dict, _ring_radius(cfg), angular_samples=2048).h6
        rows.append(
            {
                "retardance_error_rad": float(err),
                "retardance_error_waves": float(err / TWOPI),
                "h6": float(h6),
                "mean_angle_rad": float(np.angle(np.mean(np.exp(2j * psi))) / 2.0),
            }
        )
    return rows


def bandwidth_report(
    cfg: VectorArmConfig | None = None,
    *,
    wavelengths_m: Sequence[float] = (1026e-9, 1029e-9, 1032e-9),
    grid: Mapping[str, Any] | None = None,
) -> list[dict[str, float]]:
    """Optional three-wavelength report; no assertions are implied."""

    cfg = cfg or VectorArmConfig()
    grid_dict = dict(default_vector_arm_grid(cfg) if grid is None else grid)
    rows = []
    for wavelength in wavelengths_m:
        local_cfg = replace(cfg, wavelength_m=float(wavelength))
        field = run_vector_arm(local_cfg, grid=grid_dict)
        h6 = h6_from_intensity(field.intensity, grid_dict, _ring_radius(local_cfg), angular_samples=2048).h6
        rows.append({"wavelength_m": float(wavelength), "wavelength_nm": float(wavelength / 1e-9), "h6": float(h6)})
    return rows


def plot_sweep_rows(
    rows: Sequence[Mapping[str, Any]],
    x_key: str,
    y_key: str = "h6",
    *,
    ax: plt.Axes | None = None,
    title: str = "",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot raw sweep rows without smoothing."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    else:
        fig = ax.figure
    x = [row[x_key] for row in rows]
    y = [row[y_key] for row in rows]
    ax.plot(x, y, marker="o", lw=1.2)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    if title:
        ax.set_title(title)
    return fig, ax


__all__ = [
    "bandwidth_report",
    "delta_rotation_sweep",
    "fill_factor_sweep",
    "measure_pattern_rotation",
    "plot_sweep_rows",
    "quantisation_sweep",
    "retardance_sweep",
    "sector_rotation_sweep",
]
