"""Hardware-aware *candidate* screen for the short-Bessel experiment.

This module deliberately moves one step closer to the laboratory than the
sample-equivalent short-Bessel checks while retaining a strict claim boundary.
It represents, in order,

    Gaussian -> rectangular SLM1 -> SLM1-to-SLM2 propagation
    -> rectangular SLM2 (residual radial phase + carrier + optional correction)
    -> nominal f=300 mm 4F relay + Fourier stop
    -> physical-axicon equivalent radial phase
    -> fixed pre-objective-to-sample demagnification
    -> homogeneous fused-silica BL-ASM propagation.

Known repository hardware bindings are used where available: 1920x1080,
8 um-pitch SLMs, 8-bit phase, 0.93 fill factor and the canonical 20-pixel
(6.25 lp/mm) carrier.  The actual SLM pixel lattice is NOT spatially resolved on
the default numerical grid; fill-factor deadspace therefore uses the repository
``unresolved_effective_duty`` coherent-zero-order approximation.  The 4F lens
separations are the repository nominal 300 mm scenario, not measured bench
geometry.  The physical axicon is parameterised by a *measured physical-only
sample FWHM* rather than by an unverified nominal cone angle.  Until that
measurement and the real pinhole/lens apertures/distances are supplied, outputs
are candidate feasibility predictions, not calibrated bench predictions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import numpy as np

from vbb_study.config import SLMConfig
from vbb_study.equations.fields import make_xy_grid
from vbb_study.equations.propagation import discrete_power, make_bl_asm_propagator
from vbb_study.slm_model import apply_slm, quantise
from vbb_study.vector_arm_config import SLMPanelConfig
from vbb_study.digital_twin.nominal_f300_4f import NominalF300Config, run_nominal_f300_4f
from vbb_study.short_bessel import (
    ShortBesselDesignResult,
    kr_from_j0_fwhm_diameter_m,
)
from vbb_study.short_bessel_propagation import EPS, MM, UM, _fwhm_width, _halfmax_zone
from vbb_study.short_bessel_realizability import _positive_ring_metrics, _predicted_ring_diameter_um
from vbb_study.viz_fields import phase_winding


TWOPI = 2.0 * math.pi


@dataclass(frozen=True)
class BenchCandidateConfig:
    """Numerical and nominal-hardware settings for one candidate screen."""

    grid_n: int = 512
    plane_width_mm: float = 16.384
    wavelength_nm: float = 1029.0
    sample_index: float = 1.45
    input_pre_radius_mm: float = 4.220
    slm1_to_slm2_distance_mm: float = 0.040  # repository diagnostic placeholder

    four_f_focal_length_mm: float = 300.0
    lens_clear_radius_mm: float = 7.0
    pinhole_radius_mm: float = 0.18
    carrier_lp_per_mm: float = 6.25
    carrier_sign: int = +1

    measured_physical_only_fwhm_um: float = 3.0
    correction_phase_rad: float = 0.0
    vortex_on_slm1: bool = True

    z_min_um: float = 0.0
    z_max_um: float = 800.0
    z_points: int = 161
    bandlimit: bool = True

    def validated(self) -> "BenchCandidateConfig":
        if int(self.grid_n) < 257:
            raise ValueError("grid_n must be >=257 for the candidate bench screen")
        if float(self.plane_width_mm) <= 0.0:
            raise ValueError("plane_width_mm must be positive")
        if float(self.input_pre_radius_mm) <= 0.0:
            raise ValueError("input_pre_radius_mm must be positive")
        if float(self.four_f_focal_length_mm) <= 0.0:
            raise ValueError("four_f_focal_length_mm must be positive")
        if float(self.lens_clear_radius_mm) <= 0.0:
            raise ValueError("lens_clear_radius_mm must be positive")
        if float(self.pinhole_radius_mm) <= 0.0:
            raise ValueError("pinhole_radius_mm must be positive")
        if float(self.measured_physical_only_fwhm_um) <= 0.0:
            raise ValueError("measured_physical_only_fwhm_um must be positive")
        if int(self.carrier_sign) not in {-1, +1}:
            raise ValueError("carrier_sign must be +/-1")
        if int(self.z_points) < 21 or float(self.z_max_um) <= float(self.z_min_um):
            raise ValueError("invalid sample z scan")
        return self

    @property
    def plane_width_m(self) -> float:
        return float(self.plane_width_mm) * MM

    @property
    def dx_pre_m(self) -> float:
        return self.plane_width_m / int(self.grid_n)


@dataclass(frozen=True)
class BenchCandidateMetrics:
    ell: int
    physical_only_fwhm_um_assumption: float
    input_pre_radius_mm: float
    slm_active_width_mm: float
    slm_active_height_mm: float
    slm1_capture_fraction: float
    slm2_capture_fraction: float
    physical_kr_sample_m_inv: float
    residual_slm_kr_sample_m_inv: float
    residual_slm_kr_pre_m_inv: float
    residual_phase_wrap_period_mm: float
    carrier_lp_per_mm: float
    predicted_order_shift_mm: float
    measured_order_peak_x_mm: float
    measured_order_peak_y_mm: float
    pinhole_radius_mm: float
    pinhole_transmitted_fraction: float
    lens1_transmitted_fraction: float
    lens2_transmitted_fraction: float
    relay_output_power_fraction_vs_slm2_input: float
    sample_peak_z_um: float
    sample_halfmax_zone_start_um: float
    sample_halfmax_zone_end_um: float
    sample_halfmax_zone_length_um: float
    central_fwhm_um: float | None
    predicted_ring_diameter_um: float | None
    measured_ring_diameter_um: float | None
    ring_thickness_fwhm_um: float | None
    measured_phase_winding: float
    winding_error_turns: float
    sample_power_ratio_min: float
    sample_power_ratio_max: float
    claim_boundary: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchCandidateRun:
    config: BenchCandidateConfig
    design: ShortBesselDesignResult
    ell: int
    pre_grid: Mapping[str, Any]
    sample_grid: Mapping[str, Any]
    slm1_phase_rad: np.ndarray
    slm2_phase_rad: np.ndarray
    slm1_output_field: np.ndarray
    slm2_input_field: np.ndarray
    slm2_output_field: np.ndarray
    fourier_pre_stop_field: np.ndarray
    fourier_stop: np.ndarray
    relay_output_field: np.ndarray
    pre_objective_after_physical_axicon_field: np.ndarray
    sample_input_field: np.ndarray
    sample_x_um: np.ndarray
    sample_z_um: np.ndarray
    sample_xz_intensity: np.ndarray
    sample_axial_metric: np.ndarray
    sample_peak_xy_intensity: np.ndarray
    sample_peak_field: np.ndarray
    metrics: BenchCandidateMetrics


@dataclass(frozen=True)
class FilterSweepPoint:
    ell: int
    pinhole_radius_mm: float
    transmitted_fraction: float
    halfmax_zone_um: float
    central_fwhm_um: float | None
    measured_ring_diameter_um: float | None
    winding: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _panel(config: BenchCandidateConfig) -> SLMPanelConfig:
    base = SLMConfig()
    return SLMPanelConfig(
        n_x=int(base.resolution_x),
        n_y=int(base.resolution_y),
        pitch_m=float(base.pixel_pitch_m),
        phase_levels=1 << int(base.phase_bits),
        fill_factor=float(base.fill_factor),
        carrier_lp_per_mm=float(config.carrier_lp_per_mm),
        carrier_sign=int(config.carrier_sign),
    )


def _gaussian(grid: Mapping[str, Any], radius_m: float) -> np.ndarray:
    R = np.asarray(grid["R"], dtype=float)
    return np.exp(-(R / max(float(radius_m), EPS)) ** 2).astype(complex)


def _prepared_phase(phase: np.ndarray, levels: int) -> np.ndarray:
    return quantise(np.mod(np.asarray(phase, dtype=float), TWOPI), int(levels))


def _capture_fraction(before: np.ndarray, after: np.ndarray, grid: Mapping[str, Any]) -> float:
    return discrete_power(after, float(grid["dx"])) / max(discrete_power(before, float(grid["dx"])), EPS)


def _prop_air(field: np.ndarray, grid: Mapping[str, Any], wavelength_m: float, z_m: float, bandlimit: bool) -> np.ndarray:
    prop = make_bl_asm_propagator(
        np.asarray(field, dtype=complex),
        dict(grid),
        float(wavelength_m),
        n_medium=1.0,
        bandlimit=bool(bandlimit),
        include_evanescent=True,
    )
    return np.asarray(prop(float(z_m)), dtype=complex)


def _carrier_order_peak(field: np.ndarray, grid: Mapping[str, Any], predicted_x_m: float) -> tuple[float, float]:
    I = np.abs(np.asarray(field, dtype=complex)) ** 2
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    # Search a generous window around the paraxial f*lambda*f_carrier prediction.
    radius = max(0.75e-3, 0.45 * abs(float(predicted_x_m)))
    mask = (X - float(predicted_x_m)) ** 2 + Y**2 <= radius**2
    if not np.any(mask) or float(np.max(I[mask])) <= 0.0:
        idx = np.unravel_index(int(np.argmax(I)), I.shape)
    else:
        candidate = np.where(mask, I, -1.0)
        idx = np.unravel_index(int(np.argmax(candidate)), I.shape)
    return float(X[idx]), float(Y[idx])


def _sample_metrics(
    field0: np.ndarray,
    *,
    ell: int,
    sample_grid: Mapping[str, Any],
    wavelength_m: float,
    sample_index: float,
    z_min_um: float,
    z_max_um: float,
    z_points: int,
    bandlimit: bool,
    predicted_ring_diameter_um: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    z_um = np.linspace(float(z_min_um), float(z_max_um), int(z_points))
    x_um = np.asarray(sample_grid["x"], dtype=float) / UM
    centre = int(np.argmin(np.abs(x_um)))
    prop = make_bl_asm_propagator(
        np.asarray(field0, dtype=complex),
        dict(sample_grid),
        float(wavelength_m),
        n_medium=float(sample_index),
        bandlimit=bool(bandlimit),
        include_evanescent=True,
    )
    xz = np.empty((z_um.size, x_um.size), dtype=np.float32)
    axial = np.empty(z_um.size, dtype=float)
    power = np.empty(z_um.size, dtype=float)
    p0 = discrete_power(field0, float(sample_grid["dx"]))
    peak_value = -np.inf
    peak_i = 0
    peak_field: np.ndarray | None = None
    for i, zz in enumerate(z_um):
        U = np.asarray(prop(float(zz) * UM), dtype=complex)
        I = np.abs(U) ** 2
        xz[i] = np.asarray(I[centre, :], dtype=np.float32)
        axial[i] = float(I[centre, centre]) if int(ell) == 0 else float(np.max(I))
        power[i] = discrete_power(U, float(sample_grid["dx"])) / max(p0, EPS)
        if axial[i] > peak_value:
            peak_value = axial[i]
            peak_i = i
            peak_field = U.copy()
    assert peak_field is not None
    peak_xy = np.asarray(np.abs(peak_field) ** 2, dtype=np.float32)
    line = peak_xy[centre]
    z0, z1, l50 = _halfmax_zone(z_um, axial)
    if int(ell) == 0:
        fwhm = float(_fwhm_width(x_um, line))
        ring_d = None
        ring_t = None
        winding_r_m = max(0.5 * fwhm * UM, 4.0 * float(sample_grid["dx"]))
    else:
        fwhm = None
        rr_um, rt_um = _positive_ring_metrics(x_um, line)
        ring_d = float(2.0 * rr_um)
        ring_t = float(rt_um)
        winding_r_m = max(float(rr_um) * UM, 4.0 * float(sample_grid["dx"]))
    winding = phase_winding(peak_field, dict(sample_grid), winding_r_m, n_phi=720)
    metrics = {
        "peak_z_um": float(z_um[peak_i]),
        "zone_start_um": float(z0),
        "zone_end_um": float(z1),
        "zone_length_um": float(l50),
        "central_fwhm_um": fwhm,
        "predicted_ring_diameter_um": predicted_ring_diameter_um,
        "measured_ring_diameter_um": ring_d,
        "ring_thickness_fwhm_um": ring_t,
        "winding": float(winding),
        "power_ratio_min": float(np.min(power)),
        "power_ratio_max": float(np.max(power)),
    }
    return x_um, z_um, xz, axial, peak_xy, {"peak_field": peak_field, **metrics}


def run_bench_candidate(
    design: ShortBesselDesignResult,
    *,
    ell: int,
    config: BenchCandidateConfig | None = None,
) -> BenchCandidateRun:
    """Run one B0/Vell candidate through the hardware-aware nominal route."""

    cfg = (config or BenchCandidateConfig()).validated()
    wavelength_m = float(cfg.wavelength_nm) * 1e-9
    pre_grid = make_xy_grid(int(cfg.grid_n), float(cfg.dx_pre_m))
    panel = _panel(cfg)
    phi = np.asarray(pre_grid["PHI"], dtype=float)
    R = np.asarray(pre_grid["R"], dtype=float)
    X = np.asarray(pre_grid["X"], dtype=float)

    physical_kr_sample = kr_from_j0_fwhm_diameter_m(float(cfg.measured_physical_only_fwhm_um) * UM)
    residual_kr_sample = float(design.target_kr_sample_m_inv) - float(physical_kr_sample)
    residual_kr_pre = residual_kr_sample * float(design.relay_demag)
    physical_kr_pre = physical_kr_sample * float(design.relay_demag)

    source = _gaussian(pre_grid, float(cfg.input_pre_radius_mm) * MM)
    slm1_target = int(ell) * phi if cfg.vortex_on_slm1 else np.zeros_like(phi)
    slm1_phase = _prepared_phase(slm1_target, panel.phase_levels)
    slm1_app = apply_slm(
        source,
        slm1_phase,
        pre_grid,
        panel,
        phase_is_prepared=True,
        quantise_phase=False,
        apply_fill_factor=True,
        apply_carrier=False,
        fill_factor_model="coherent_unmodulated_deadspace",
    )
    slm1_capture = _capture_fraction(source, np.where(slm1_app.aperture, source, 0.0), pre_grid)
    slm2_in = _prop_air(
        slm1_app.total,
        pre_grid,
        wavelength_m,
        float(cfg.slm1_to_slm2_distance_mm) * MM,
        cfg.bandlimit,
    )

    carrier = TWOPI * float(panel.carrier_lp_per_m) * X
    radial = -float(residual_kr_pre) * R
    vortex2 = np.zeros_like(phi) if cfg.vortex_on_slm1 else int(ell) * phi
    correction = np.full_like(phi, float(cfg.correction_phase_rad))
    slm2_phase = _prepared_phase(radial + vortex2 + carrier + correction, panel.phase_levels)
    slm2_app = apply_slm(
        slm2_in,
        slm2_phase,
        pre_grid,
        panel,
        phase_is_prepared=True,
        quantise_phase=False,
        apply_fill_factor=True,
        apply_carrier=False,
        fill_factor_model="coherent_unmodulated_deadspace",
    )
    slm2_capture = _capture_fraction(slm2_in, np.where(slm2_app.aperture, slm2_in, 0.0), pre_grid)

    f_m = float(cfg.four_f_focal_length_mm) * MM
    predicted_shift_m = f_m * wavelength_m * float(panel.carrier_lp_per_m)
    four_cfg_probe = NominalF300Config(
        simulation_plane_width_m=float(cfg.plane_width_m),
        simulation_grid_size=int(cfg.grid_n),
        wavelength_m=wavelength_m,
        n_medium=1.0,
        input_beam_radius_m=float(cfg.input_pre_radius_mm) * MM,
        lens_clear_radius_m=float(cfg.lens_clear_radius_mm) * MM,
        pinhole_radius_m=float(cfg.pinhole_radius_mm) * MM,
        pinhole_offset_x_m=float(predicted_shift_m),
        pinhole_offset_y_m=0.0,
        command_domain_carrier_cycles_x=0.0,
        command_domain_carrier_cycles_y=0.0,
        numerical_model_carrier_cycles_x=0.0,
        numerical_model_carrier_cycles_y=0.0,
        slm2_to_lens1_m=f_m,
        lens1_focal_length_m=f_m,
        lens1_to_fourier_plane_m=f_m,
        fourier_plane_to_lens2_m=f_m,
        lens2_focal_length_m=f_m,
        lens2_to_nominal_relay_output_m=f_m,
        bandlimit=bool(cfg.bandlimit),
        slm_phase_quantisation_levels=panel.phase_levels,
    )
    probe = run_nominal_f300_4f(four_cfg_probe, field_post_slm2=slm2_app.total)
    peak_x_m, peak_y_m = _carrier_order_peak(probe.fourier_plane_field_pre_stop, pre_grid, predicted_shift_m)
    four_cfg = NominalF300Config(**{**asdict(four_cfg_probe), "pinhole_offset_x_m": peak_x_m, "pinhole_offset_y_m": peak_y_m})
    four = run_nominal_f300_4f(four_cfg, field_post_slm2=slm2_app.total)

    physical_phase = np.exp(-1j * float(physical_kr_pre) * R)
    pre_after_axicon = np.asarray(four.nominal_relay_output_field, dtype=complex) * physical_phase

    # Objective surrogate: use the repository's fixed pre-to-sample demagnification.
    # The complex array is preserved while its transverse coordinate pitch is scaled.
    sample_dx_m = float(pre_grid["dx"]) * float(design.relay_demag)
    sample_grid = make_xy_grid(int(cfg.grid_n), sample_dx_m)
    sample_input = pre_after_axicon.copy()
    pred_ring = _predicted_ring_diameter_um(int(ell), float(design.target_kr_sample_m_inv))
    x_um, z_um, xz, axial, peak_xy, sm = _sample_metrics(
        sample_input,
        ell=int(ell),
        sample_grid=sample_grid,
        wavelength_m=wavelength_m,
        sample_index=float(cfg.sample_index),
        z_min_um=float(cfg.z_min_um),
        z_max_um=float(cfg.z_max_um),
        z_points=int(cfg.z_points),
        bandlimit=bool(cfg.bandlimit),
        predicted_ring_diameter_um=pred_ring,
    )

    p_slm2_in = discrete_power(slm2_in, float(pre_grid["dx"]))
    p_relay = discrete_power(four.nominal_relay_output_field, float(pre_grid["dx"]))
    residual_period_mm = float("inf") if abs(residual_kr_pre) <= EPS else TWOPI / abs(residual_kr_pre) / MM
    slm_cfg = SLMConfig()
    metrics = BenchCandidateMetrics(
        ell=int(ell),
        physical_only_fwhm_um_assumption=float(cfg.measured_physical_only_fwhm_um),
        input_pre_radius_mm=float(cfg.input_pre_radius_mm),
        slm_active_width_mm=float(slm_cfg.active_width_m / MM),
        slm_active_height_mm=float(slm_cfg.active_height_m / MM),
        slm1_capture_fraction=float(slm1_capture),
        slm2_capture_fraction=float(slm2_capture),
        physical_kr_sample_m_inv=float(physical_kr_sample),
        residual_slm_kr_sample_m_inv=float(residual_kr_sample),
        residual_slm_kr_pre_m_inv=float(residual_kr_pre),
        residual_phase_wrap_period_mm=float(residual_period_mm),
        carrier_lp_per_mm=float(cfg.carrier_lp_per_mm),
        predicted_order_shift_mm=float(predicted_shift_m / MM),
        measured_order_peak_x_mm=float(peak_x_m / MM),
        measured_order_peak_y_mm=float(peak_y_m / MM),
        pinhole_radius_mm=float(cfg.pinhole_radius_mm),
        pinhole_transmitted_fraction=float(four.diagnostics["pinhole_transmitted_fraction"]),
        lens1_transmitted_fraction=float(four.diagnostics["lens1_pupil_transmitted_fraction"]),
        lens2_transmitted_fraction=float(four.diagnostics["lens2_pupil_transmitted_fraction"]),
        relay_output_power_fraction_vs_slm2_input=float(p_relay / max(p_slm2_in, EPS)),
        sample_peak_z_um=float(sm["peak_z_um"]),
        sample_halfmax_zone_start_um=float(sm["zone_start_um"]),
        sample_halfmax_zone_end_um=float(sm["zone_end_um"]),
        sample_halfmax_zone_length_um=float(sm["zone_length_um"]),
        central_fwhm_um=None if sm["central_fwhm_um"] is None else float(sm["central_fwhm_um"]),
        predicted_ring_diameter_um=None if pred_ring is None else float(pred_ring),
        measured_ring_diameter_um=None if sm["measured_ring_diameter_um"] is None else float(sm["measured_ring_diameter_um"]),
        ring_thickness_fwhm_um=None if sm["ring_thickness_fwhm_um"] is None else float(sm["ring_thickness_fwhm_um"]),
        measured_phase_winding=float(sm["winding"]),
        winding_error_turns=float(sm["winding"] - int(ell)),
        sample_power_ratio_min=float(sm["power_ratio_min"]),
        sample_power_ratio_max=float(sm["power_ratio_max"]),
        claim_boundary=(
            "candidate feasibility only: rectangular SLM apertures, 8-bit phase, unresolved fill-factor/zero-order surrogate, "
            "canonical 20-pixel carrier and nominal 300-mm 4F are represented; real SLM pixel diffraction, measured 4F geometry, "
            "measured pinhole/lens apertures, measured physical-axicon-only FWHM, spatial correction map and objective aberrations remain uncalibrated."
        ),
    )
    return BenchCandidateRun(
        config=cfg,
        design=design,
        ell=int(ell),
        pre_grid=pre_grid,
        sample_grid=sample_grid,
        slm1_phase_rad=slm1_phase,
        slm2_phase_rad=slm2_phase,
        slm1_output_field=np.asarray(slm1_app.total, dtype=complex),
        slm2_input_field=np.asarray(slm2_in, dtype=complex),
        slm2_output_field=np.asarray(slm2_app.total, dtype=complex),
        fourier_pre_stop_field=np.asarray(four.fourier_plane_field_pre_stop, dtype=complex),
        fourier_stop=np.asarray(four.fourier_stop_transmission, dtype=float),
        relay_output_field=np.asarray(four.nominal_relay_output_field, dtype=complex),
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


def sweep_filter_radii(
    design: ShortBesselDesignResult,
    *,
    ell: int,
    radii_mm: tuple[float, ...] = (0.12, 0.18, 0.25, 0.35, 0.50),
    base_config: BenchCandidateConfig | None = None,
) -> tuple[FilterSweepPoint, ...]:
    base = (base_config or BenchCandidateConfig()).validated()
    points: list[FilterSweepPoint] = []
    for radius in radii_mm:
        cfg = BenchCandidateConfig(**{**asdict(base), "pinhole_radius_mm": float(radius)})
        run = run_bench_candidate(design, ell=int(ell), config=cfg)
        m = run.metrics
        points.append(
            FilterSweepPoint(
                ell=int(ell),
                pinhole_radius_mm=float(radius),
                transmitted_fraction=float(m.pinhole_transmitted_fraction),
                halfmax_zone_um=float(m.sample_halfmax_zone_length_um),
                central_fwhm_um=m.central_fwhm_um,
                measured_ring_diameter_um=m.measured_ring_diameter_um,
                winding=float(m.measured_phase_winding),
            )
        )
    return tuple(points)


__all__ = [
    "BenchCandidateConfig",
    "BenchCandidateMetrics",
    "BenchCandidateRun",
    "FilterSweepPoint",
    "run_bench_candidate",
    "sweep_filter_radii",
]
