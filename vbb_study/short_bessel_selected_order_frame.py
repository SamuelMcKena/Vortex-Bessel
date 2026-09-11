"""Re-reference the 4F-selected diffraction order to its own optical axis.

The physical bench is aligned to the transmitted +1 order after the Fourier
stop.  A global-axis simulation that leaves the carrier tilt on the field makes
that beam walk laterally through the downstream axicon/sample and therefore
measures the wrong axis.  This module takes the upstream hardware-aware run,
estimates the residual mean transverse phase ramp directly from the complex 4F
relay field, removes that ramp, then applies the physical-axicon equivalent
phase and propagates in the sample.

This is still a *candidate* model: it does not know the measured 4F geometry,
objective aberration, SLM correction map or physical-axicon calibration.  The
axis re-referencing is not an optimisation knob; it is a coordinate change to
the selected-order beam frame, equivalent to aligning downstream optics/camera
to that transmitted beam.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any
import math

import numpy as np

from vbb_study.equations.fields import make_xy_grid
from vbb_study.short_bessel_bench_candidate import (
    BenchCandidateConfig,
    BenchCandidateRun,
    _predicted_ring_diameter_um,
    _sample_metrics,
    run_bench_candidate,
)
from vbb_study.short_bessel import ShortBesselDesignResult
from vbb_study.short_bessel_propagation import UM


TWOPI = 2.0 * math.pi


@dataclass(frozen=True)
class SelectedOrderFrameMetrics:
    ell: int
    pinhole_radius_mm: float
    physical_only_fwhm_um_assumption: float
    input_pre_radius_mm: float
    raw_mean_tilt_x_cpm: float
    raw_mean_tilt_y_cpm: float
    raw_mean_tilt_angle_mrad: float
    fourier_order_peak_x_mm: float
    fourier_order_peak_y_mm: float
    pinhole_transmitted_fraction: float
    lens1_transmitted_fraction: float
    lens2_transmitted_fraction: float
    slm1_capture_fraction: float
    slm2_capture_fraction: float
    sample_peak_z_um: float
    sample_halfmax_zone_start_um: float
    sample_halfmax_zone_end_um: float
    sample_halfmax_zone_length_um: float
    central_fwhm_um: float | None
    predicted_ring_diameter_um: float | None
    measured_ring_diameter_um: float | None
    ring_thickness_fwhm_um: float | None
    measured_phase_winding: float
    winding_magnitude_error_turns: float
    sample_power_ratio_min: float
    sample_power_ratio_max: float
    claim_boundary: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SelectedOrderFrameRun:
    upstream: BenchCandidateRun
    aligned_relay_field: np.ndarray
    pre_objective_after_physical_axicon_field: np.ndarray
    sample_input_field: np.ndarray
    sample_x_um: np.ndarray
    sample_z_um: np.ndarray
    sample_xz_intensity: np.ndarray
    sample_axial_metric: np.ndarray
    sample_peak_xy_intensity: np.ndarray
    sample_peak_field: np.ndarray
    metrics: SelectedOrderFrameMetrics


def _mean_phase_ramp_cpm(field: np.ndarray, dx_m: float) -> tuple[float, float]:
    """Estimate the intensity-weighted mean phase ramp in cycles/m.

    Neighbor correlations are robust to 2*pi wraps.  A symmetric radial phase
    has approximately zero global mean slope, while the selected carrier order
    contributes a strong uniform ramp.
    """

    U = np.asarray(field, dtype=complex)
    amp2 = np.abs(U) ** 2
    threshold = 0.05 * float(np.max(amp2)) if amp2.size else 0.0

    wx = np.sqrt(amp2[:, :-1] * amp2[:, 1:])
    mx = (amp2[:, :-1] >= threshold) & (amp2[:, 1:] >= threshold)
    cx = np.sum(wx[mx] * np.conj(U[:, :-1][mx]) * U[:, 1:][mx])

    wy = np.sqrt(amp2[:-1, :] * amp2[1:, :])
    my = (amp2[:-1, :] >= threshold) & (amp2[1:, :] >= threshold)
    cy = np.sum(wy[my] * np.conj(U[:-1, :][my]) * U[1:, :][my])

    fx = float(np.angle(cx) / (TWOPI * float(dx_m))) if abs(cx) > 0 else 0.0
    fy = float(np.angle(cy) / (TWOPI * float(dx_m))) if abs(cy) > 0 else 0.0
    return fx, fy


def re_reference_selected_order(field: np.ndarray, grid: dict[str, Any]) -> tuple[np.ndarray, float, float]:
    """Remove the measured uniform phase ramp from a selected-order field."""

    dx = float(grid["dx"])
    fx, fy = _mean_phase_ramp_cpm(field, dx)
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    aligned = np.asarray(field, dtype=complex) * np.exp(-1j * TWOPI * (fx * X + fy * Y))
    return aligned, fx, fy


def run_selected_order_frame(
    design: ShortBesselDesignResult,
    *,
    ell: int,
    config: BenchCandidateConfig | None = None,
    sample_z_max_um: float = 800.0,
    sample_z_points: int = 161,
) -> SelectedOrderFrameRun:
    """Run hardware-aware upstream optics, then propagate on the +1-order axis."""

    cfg = (config or BenchCandidateConfig()).validated()

    # The upstream function also performs its legacy global-axis sample check.
    # Keep that cheap here because this routine recomputes the physically useful
    # selected-order-frame sample propagation below.
    cheap_cfg = replace(cfg, z_min_um=0.0, z_max_um=1.0, z_points=21)
    upstream = run_bench_candidate(design, ell=int(ell), config=cheap_cfg)

    pre_grid = dict(upstream.pre_grid)
    aligned_relay, fx, fy = re_reference_selected_order(upstream.relay_output_field, pre_grid)

    R = np.asarray(pre_grid["R"], dtype=float)
    physical_kr_pre = (
        float(upstream.metrics.physical_kr_sample_m_inv) * float(design.relay_demag)
    )
    physical_phase = np.exp(-1j * physical_kr_pre * R)
    pre_after_axicon = aligned_relay * physical_phase

    sample_dx_m = float(pre_grid["dx"]) * float(design.relay_demag)
    sample_grid = make_xy_grid(int(cfg.grid_n), sample_dx_m)
    sample_input = pre_after_axicon.copy()
    predicted_ring = _predicted_ring_diameter_um(int(ell), float(design.target_kr_sample_m_inv))

    x_um, z_um, xz, axial, peak_xy, sm = _sample_metrics(
        sample_input,
        ell=int(ell),
        sample_grid=sample_grid,
        wavelength_m=float(cfg.wavelength_nm) * 1e-9,
        sample_index=float(cfg.sample_index),
        z_min_um=0.0,
        z_max_um=float(sample_z_max_um),
        z_points=int(sample_z_points),
        bandlimit=bool(cfg.bandlimit),
        predicted_ring_diameter_um=predicted_ring,
    )

    # Convert mean spatial frequency to small-angle air tilt for diagnostics.
    lam = float(cfg.wavelength_nm) * 1e-9
    tilt_mag = math.hypot(fx, fy)
    arg = min(max(lam * tilt_mag, 0.0), 1.0)
    tilt_mrad = 1e3 * math.asin(arg)

    winding = float(sm["winding"])
    metrics = SelectedOrderFrameMetrics(
        ell=int(ell),
        pinhole_radius_mm=float(cfg.pinhole_radius_mm),
        physical_only_fwhm_um_assumption=float(cfg.measured_physical_only_fwhm_um),
        input_pre_radius_mm=float(cfg.input_pre_radius_mm),
        raw_mean_tilt_x_cpm=float(fx),
        raw_mean_tilt_y_cpm=float(fy),
        raw_mean_tilt_angle_mrad=float(tilt_mrad),
        fourier_order_peak_x_mm=float(upstream.metrics.measured_order_peak_x_mm),
        fourier_order_peak_y_mm=float(upstream.metrics.measured_order_peak_y_mm),
        pinhole_transmitted_fraction=float(upstream.metrics.pinhole_transmitted_fraction),
        lens1_transmitted_fraction=float(upstream.metrics.lens1_transmitted_fraction),
        lens2_transmitted_fraction=float(upstream.metrics.lens2_transmitted_fraction),
        slm1_capture_fraction=float(upstream.metrics.slm1_capture_fraction),
        slm2_capture_fraction=float(upstream.metrics.slm2_capture_fraction),
        sample_peak_z_um=float(sm["peak_z_um"]),
        sample_halfmax_zone_start_um=float(sm["zone_start_um"]),
        sample_halfmax_zone_end_um=float(sm["zone_end_um"]),
        sample_halfmax_zone_length_um=float(sm["zone_length_um"]),
        central_fwhm_um=None if sm["central_fwhm_um"] is None else float(sm["central_fwhm_um"]),
        predicted_ring_diameter_um=None if predicted_ring is None else float(predicted_ring),
        measured_ring_diameter_um=None if sm["measured_ring_diameter_um"] is None else float(sm["measured_ring_diameter_um"]),
        ring_thickness_fwhm_um=None if sm["ring_thickness_fwhm_um"] is None else float(sm["ring_thickness_fwhm_um"]),
        measured_phase_winding=winding,
        winding_magnitude_error_turns=float(abs(winding) - abs(int(ell))),
        sample_power_ratio_min=float(sm["power_ratio_min"]),
        sample_power_ratio_max=float(sm["power_ratio_max"]),
        claim_boundary=(
            "selected-order-frame candidate feasibility only: the mean +1-order carrier tilt is measured from the simulated 4F output and removed as a coordinate re-reference before the downstream axicon/sample; rectangular SLM apertures, 8-bit phase and the nominal f=300 mm 4F are represented, but measured 4F geometry, real pinhole/lens apertures, physical-axicon-only FWHM, SLM correction map and objective aberrations are not yet calibrated."
        ),
    )
    return SelectedOrderFrameRun(
        upstream=upstream,
        aligned_relay_field=aligned_relay,
        pre_objective_after_physical_axicon_field=pre_after_axicon,
        sample_input_field=sample_input,
        sample_x_um=x_um,
        sample_z_um=z_um,
        sample_xz_intensity=xz,
        sample_axial_metric=axial,
        sample_peak_xy_intensity=peak_xy,
        sample_peak_field=np.asarray(sm["peak_field"], dtype=complex),
        metrics=metrics,
    )


__all__ = [
    "SelectedOrderFrameMetrics",
    "SelectedOrderFrameRun",
    "re_reference_selected_order",
    "run_selected_order_frame",
]
