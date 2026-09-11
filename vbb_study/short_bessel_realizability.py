"""Numerical realizability checks for the short-Bessel design.

This module builds on :mod:`vbb_study.short_bessel` and the repository BL-ASM
propagator.  It answers two practical questions without pretending the complete
bench is calibrated yet:

1. What illuminated radius is required for a *propagated* ell=0 Bessel field to
   have a requested contiguous half-maximum axial zone (rather than merely the
   geometrical w*k/k_r reference length)?
2. If the same conical spectrum is given helical phase, do ell=1 and ell=3
   vortex-Bessel fields preserve the intended winding and form the expected
   bright annular core under propagation?

The input is still an equivalent sample-plane conical field.  SLM diffraction,
4F order selection, physical-axicon geometry, objective aberrations and the
sample interface remain outside this module's claim boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable
import math

import numpy as np
from scipy import special

from vbb_study.equations.fields import gaussian_amplitude, make_xy_grid
from vbb_study.equations.propagation import discrete_power, make_bl_asm_propagator
from vbb_study.short_bessel import ShortBesselDesignResult
from vbb_study.short_bessel_propagation import (
    EPS,
    MM,
    UM,
    ShortBesselPropagationConfig,
    _crossing,
    _fwhm_width,
    _halfmax_zone,
)
from vbb_study.viz_fields import phase_winding


TWOPI = 2.0 * math.pi


@dataclass(frozen=True)
class VortexPropagationMetrics:
    case_id: str
    ell: int
    sample_beam_radius_um: float
    pre_beam_radius_mm: float
    peak_z_um: float
    axial_metric_name: str
    halfmax_zone_start_um: float
    halfmax_zone_end_um: float
    halfmax_zone_length_um: float
    central_fwhm_um: float | None
    predicted_main_ring_diameter_um: float | None
    measured_main_ring_diameter_um: float | None
    measured_ring_thickness_fwhm_um: float | None
    measured_phase_winding: float
    winding_error_turns: float
    input_power_arb: float
    min_power_ratio: float
    max_power_ratio: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VortexPropagationCase:
    case_id: str
    ell: int
    x_um: np.ndarray
    z_um: np.ndarray
    input_field: np.ndarray
    xz_intensity: np.ndarray
    axial_metric: np.ndarray
    power_ratio: np.ndarray
    peak_field: np.ndarray
    peak_xy_intensity: np.ndarray
    snapshots_xy_intensity: dict[float, np.ndarray]
    metrics: VortexPropagationMetrics


@dataclass(frozen=True)
class RadiusSearchPoint:
    sample_radius_um: float
    pre_radius_mm: float
    halfmax_zone_um: float
    peak_z_um: float
    core_fwhm_um: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class RadiusOptimizationResult:
    requested_halfmax_zone_um: float
    optimized_sample_radius_um: float
    optimized_pre_radius_mm: float
    achieved_halfmax_zone_um: float
    achieved_core_fwhm_um: float
    safe_pre_radius_mm: float
    full_half_height_pre_radius_mm: float
    safe_limit_halfmax_zone_um: float
    full_limit_halfmax_zone_um: float
    within_90pct_safe_radius: bool
    within_full_slm_half_height: bool
    recommended_status: str
    search_points: tuple[RadiusSearchPoint, ...]
    claim_boundary: str

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["search_points"] = [p.as_dict() for p in self.search_points]
        return data


def vortex_conical_field(
    design: ShortBesselDesignResult,
    *,
    ell: int,
    sample_beam_radius_m: float,
    grid: dict[str, Any],
) -> np.ndarray:
    """Return ``exp(-(r/w)^2) exp(-i k_r r) exp(i ell phi)``."""

    radius = float(sample_beam_radius_m)
    if radius <= 0.0:
        raise ValueError("sample_beam_radius_m must be positive")
    R = np.asarray(grid["R"], dtype=float)
    PHI = np.asarray(grid["PHI"], dtype=float)
    amp = gaussian_amplitude(R, radius)
    conical = np.exp(-1j * float(design.target_kr_sample_m_inv) * R)
    vortex = np.exp(1j * int(ell) * PHI)
    return np.asarray(amp * conical * vortex, dtype=complex)


def _positive_ring_metrics(
    x_um: np.ndarray,
    intensity_line: np.ndarray,
) -> tuple[float, float]:
    """Return positive-side ring radius and radial FWHM thickness in microns."""

    x = np.asarray(x_um, dtype=float)
    y = np.maximum(np.asarray(intensity_line, dtype=float), 0.0)
    pos = x > 0.0
    if not np.any(pos):
        return float("nan"), float("nan")
    xp = x[pos]
    yp = y[pos]
    if float(np.max(yp)) <= 0.0:
        return float("nan"), float("nan")
    i_peak = int(np.argmax(yp))
    peak = float(yp[i_peak])
    radius = float(xp[i_peak])
    level = 0.5 * peak

    left = i_peak
    while left > 0 and yp[left] >= level:
        left -= 1
    right = i_peak
    while right < yp.size - 1 and yp[right] >= level:
        right += 1

    if left == i_peak or right == i_peak:
        return radius, float("nan")
    xl = _crossing(xp[left], yp[left], xp[left + 1], yp[left + 1], level)
    xr = _crossing(xp[right - 1], yp[right - 1], xp[right], yp[right], level)
    return radius, float(xr - xl)


def _predicted_ring_diameter_um(ell: int, kr_m_inv: float) -> float | None:
    ell_abs = abs(int(ell))
    if ell_abs == 0:
        return None
    root = float(special.jnp_zeros(ell_abs, 1)[0])
    return float(2.0 * root / float(kr_m_inv) / UM)


def propagate_vortex_case(
    design: ShortBesselDesignResult,
    *,
    ell: int,
    sample_beam_radius_m: float,
    config: ShortBesselPropagationConfig | None = None,
    snapshot_z_um: Iterable[float] = (100.0, 250.0, 400.0, 550.0),
) -> VortexPropagationCase:
    """Propagate one equivalent Bessel or vortex-Bessel field.

    For ell=0 the axial metric is on-axis intensity.  For |ell|>0 the axis is a
    phase singularity, so the axial metric is the peak transverse intensity in
    each plane (the bright vortex ring).
    """

    cfg = (config or ShortBesselPropagationConfig()).validated()
    dx_m = float(cfg.dx_um) * UM
    grid = make_xy_grid(int(cfg.grid_n), dx_m)
    x_um = np.asarray(grid["x"], dtype=float) / UM
    z_um = np.linspace(float(cfg.z_min_um), float(cfg.z_max_um), int(cfg.z_points))
    z_m = z_um * UM

    U0 = vortex_conical_field(
        design,
        ell=int(ell),
        sample_beam_radius_m=float(sample_beam_radius_m),
        grid=grid,
    )
    wavelength_m = float(design.inputs.wavelength_nm) * 1.0e-9
    prop = make_bl_asm_propagator(
        U0,
        grid,
        wavelength_m,
        n_medium=float(design.inputs.sample_index),
        bandlimit=bool(cfg.bandlimit),
        include_evanescent=bool(cfg.include_evanescent),
    )

    centre = int(np.argmin(np.abs(x_um)))
    xz = np.empty((z_um.size, x_um.size), dtype=np.float32)
    axial = np.empty(z_um.size, dtype=float)
    power = np.empty(z_um.size, dtype=float)
    input_power = discrete_power(U0, dx_m)

    requested_indices = {
        int(np.argmin(np.abs(z_um - float(value)))): float(z_um[int(np.argmin(np.abs(z_um - float(value))))])
        for value in snapshot_z_um
    }
    snapshots: dict[float, np.ndarray] = {}

    peak_value = -np.inf
    peak_index = 0
    peak_field: np.ndarray | None = None

    for i, zz in enumerate(z_m):
        U = np.asarray(prop(float(zz)), dtype=complex)
        I = np.abs(U) ** 2
        xz[i] = np.asarray(I[centre, :], dtype=np.float32)
        if int(ell) == 0:
            metric_value = float(I[centre, centre])
        else:
            metric_value = float(np.max(I))
        axial[i] = metric_value
        power[i] = discrete_power(U, dx_m) / max(input_power, EPS)
        if i in requested_indices:
            snapshots[requested_indices[i]] = np.asarray(I, dtype=np.float32)
        if metric_value > peak_value:
            peak_value = metric_value
            peak_index = i
            peak_field = U.copy()

    assert peak_field is not None
    peak_xy = np.asarray(np.abs(peak_field) ** 2, dtype=np.float32)
    peak_line = peak_xy[centre, :]
    z0, z1, zone = _halfmax_zone(z_um, axial)

    if int(ell) == 0:
        central_fwhm = _fwhm_width(x_um, peak_line)
        predicted_ring_d = None
        measured_ring_d = None
        ring_thickness = None
        winding_radius_m = max(0.5 * float(design.target_fwhm_um) * UM, 2.0 * dx_m)
    else:
        central_fwhm = None
        predicted_ring_d = _predicted_ring_diameter_um(int(ell), design.target_kr_sample_m_inv)
        ring_radius_um, ring_thickness_value = _positive_ring_metrics(x_um, peak_line)
        measured_ring_d = float(2.0 * ring_radius_um)
        ring_thickness = float(ring_thickness_value)
        winding_radius_m = max(float(ring_radius_um) * UM, 2.0 * dx_m)

    winding = phase_winding(
        peak_field,
        grid,
        winding_radius_m,
        n_phi=720,
    )

    pre_radius_mm = float(sample_beam_radius_m) / float(design.relay_demag) / MM
    metrics = VortexPropagationMetrics(
        case_id=str(case_id := f"ell{int(ell)}"),
        ell=int(ell),
        sample_beam_radius_um=float(sample_beam_radius_m / UM),
        pre_beam_radius_mm=float(pre_radius_mm),
        peak_z_um=float(z_um[peak_index]),
        axial_metric_name="on_axis_intensity" if int(ell) == 0 else "peak_ring_intensity",
        halfmax_zone_start_um=float(z0),
        halfmax_zone_end_um=float(z1),
        halfmax_zone_length_um=float(zone),
        central_fwhm_um=None if central_fwhm is None else float(central_fwhm),
        predicted_main_ring_diameter_um=predicted_ring_d,
        measured_main_ring_diameter_um=measured_ring_d,
        measured_ring_thickness_fwhm_um=ring_thickness,
        measured_phase_winding=float(winding),
        winding_error_turns=float(winding - int(ell)),
        input_power_arb=float(input_power),
        min_power_ratio=float(np.min(power)),
        max_power_ratio=float(np.max(power)),
    )

    return VortexPropagationCase(
        case_id=case_id,
        ell=int(ell),
        x_um=x_um,
        z_um=z_um,
        input_field=U0,
        xz_intensity=xz,
        axial_metric=axial,
        power_ratio=power,
        peak_field=peak_field,
        peak_xy_intensity=peak_xy,
        snapshots_xy_intensity=snapshots,
        metrics=metrics,
    )


def optimize_radius_for_halfmax_zone(
    design: ShortBesselDesignResult,
    *,
    requested_halfmax_zone_um: float = 500.0,
    config: ShortBesselPropagationConfig | None = None,
    lower_sample_radius_um: float = 20.0,
    upper_sample_radius_um: float = 45.0,
    tolerance_um: float = 3.0,
    max_iterations: int = 10,
) -> RadiusOptimizationResult:
    """Bisection-search the ell=0 illuminated radius for a propagated L50 target."""

    cfg = (config or ShortBesselPropagationConfig(grid_n=257, dx_um=0.35, z_max_um=850.0, z_points=171)).validated()
    target = float(requested_halfmax_zone_um)
    if target <= 0.0:
        raise ValueError("requested_halfmax_zone_um must be positive")

    cache: dict[float, VortexPropagationCase] = {}

    def evaluate(radius_um: float) -> VortexPropagationCase:
        key = round(float(radius_um), 9)
        if key not in cache:
            cache[key] = propagate_vortex_case(
                design,
                ell=0,
                sample_beam_radius_m=float(radius_um) * UM,
                config=cfg,
                snapshot_z_um=(),
            )
        return cache[key]

    lo = float(lower_sample_radius_um)
    hi = float(upper_sample_radius_um)
    c_lo = evaluate(lo)
    c_hi = evaluate(hi)
    if c_lo.metrics.halfmax_zone_length_um > target:
        raise ValueError("lower radius already exceeds requested half-maximum zone; lower the search bound")
    if c_hi.metrics.halfmax_zone_length_um < target:
        raise ValueError("upper radius does not reach requested half-maximum zone; increase the search bound")

    best = min((c_lo, c_hi), key=lambda c: abs(c.metrics.halfmax_zone_length_um - target))
    for _ in range(int(max_iterations)):
        mid = 0.5 * (lo + hi)
        c_mid = evaluate(mid)
        if abs(c_mid.metrics.halfmax_zone_length_um - target) < abs(best.metrics.halfmax_zone_length_um - target):
            best = c_mid
        if abs(c_mid.metrics.halfmax_zone_length_um - target) <= float(tolerance_um):
            best = c_mid
            break
        if c_mid.metrics.halfmax_zone_length_um < target:
            lo = mid
        else:
            hi = mid

    safe_pre_mm = float(design.slm_safe_radius_mm)
    full_pre_mm = 0.5 * float(design.inputs.slm_resolution_y) * float(design.inputs.slm_pixel_pitch_um) * 1.0e-3
    safe_sample_um = safe_pre_mm * MM * float(design.relay_demag) / UM
    full_sample_um = full_pre_mm * MM * float(design.relay_demag) / UM
    c_safe = evaluate(safe_sample_um)
    c_full = evaluate(full_sample_um)

    optimized_pre_mm = best.metrics.pre_beam_radius_mm
    within_safe = optimized_pre_mm <= safe_pre_mm + 1.0e-12
    within_full = optimized_pre_mm <= full_pre_mm + 1.0e-12
    if within_safe:
        status = "target_reached_inside_90pct_slm_safe_radius"
    elif within_full:
        status = "target_reached_only_near_slm_vertical_edge"
    else:
        status = "target_requires_radius_beyond_slm_vertical_half_height"

    points = tuple(
        RadiusSearchPoint(
            sample_radius_um=float(case.metrics.sample_beam_radius_um),
            pre_radius_mm=float(case.metrics.pre_beam_radius_mm),
            halfmax_zone_um=float(case.metrics.halfmax_zone_length_um),
            peak_z_um=float(case.metrics.peak_z_um),
            core_fwhm_um=float(case.metrics.central_fwhm_um or float("nan")),
        )
        for _, case in sorted(cache.items())
    )

    return RadiusOptimizationResult(
        requested_halfmax_zone_um=target,
        optimized_sample_radius_um=float(best.metrics.sample_beam_radius_um),
        optimized_pre_radius_mm=float(optimized_pre_mm),
        achieved_halfmax_zone_um=float(best.metrics.halfmax_zone_length_um),
        achieved_core_fwhm_um=float(best.metrics.central_fwhm_um or float("nan")),
        safe_pre_radius_mm=float(safe_pre_mm),
        full_half_height_pre_radius_mm=float(full_pre_mm),
        safe_limit_halfmax_zone_um=float(c_safe.metrics.halfmax_zone_length_um),
        full_limit_halfmax_zone_um=float(c_full.metrics.halfmax_zone_length_um),
        within_90pct_safe_radius=bool(within_safe),
        within_full_slm_half_height=bool(within_full),
        recommended_status=status,
        search_points=points,
        claim_boundary=(
            "Equivalent sample-plane scalar propagation. Radius feasibility uses the repository relay mapping "
            "and SLM vertical active height; real 4F clipping, Gaussian truncation, physical axicon and interface "
            "must still be measured before loading a bench command."
        ),
    )


def combined_slm_phase_preview(
    design: ShortBesselDesignResult,
    *,
    ell: int,
    nx: int | None = None,
    ny: int | None = None,
) -> dict[str, Any]:
    """Return wrapped radial + helical phase at the pre/SLM coordinate scale.

    This is a topology/sampling preview only.  It intentionally omits blaze,
    measured wavefront correction and physical-axicon residual calibration.
    """

    nx_i = int(design.inputs.slm_resolution_x if nx is None else nx)
    ny_i = int(design.inputs.slm_resolution_y if ny is None else ny)
    pitch_m = float(design.inputs.slm_pixel_pitch_um) * UM
    x = (np.arange(nx_i) - nx_i / 2 + 0.5) * pitch_m
    y = (np.arange(ny_i) - ny_i / 2 + 0.5) * pitch_m
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.hypot(X, Y)
    PHI = np.arctan2(Y, X)
    phase = -float(design.target_kr_pre_m_inv) * R + int(ell) * PHI
    wrapped = np.mod(phase, TWOPI)
    gray = (np.floor(wrapped / TWOPI * 256.0 + 0.5).astype(np.int64) % 256).astype(np.uint8)
    return {
        "ell": int(ell),
        "phase_wrapped_rad": wrapped,
        "gray_uint8": gray,
        "x_m": x,
        "y_m": y,
        "claim_boundary": "radial + vortex topology preview only; no blaze/correction/residual physical-axicon calibration",
    }


__all__ = [
    "RadiusOptimizationResult",
    "RadiusSearchPoint",
    "VortexPropagationCase",
    "VortexPropagationMetrics",
    "combined_slm_phase_preview",
    "optimize_radius_for_halfmax_zone",
    "propagate_vortex_case",
    "vortex_conical_field",
]
