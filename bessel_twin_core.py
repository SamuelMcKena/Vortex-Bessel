"""Authoritative scalar physics engine for the structured-beam simulation atlas.

Public facade for the PHAROS + SLM vortex/Bessel digital-twin.  This module
consolidates the lab-facing PHAROS toolkit, the Elite forward model, and the
best-tested legacy metric machinery into one scalar, planning-grade source of
truth.

**Facade note (Phase 4):**
This file is currently the single combined physics engine and public API.  A
future internal split will move the engine internals into ``vbb_study/core/``
subpackage (``config.py``, ``grids.py``, ``design.py``, ``propagation_engine.py``,
``hologram_engine.py``, ``metrics.py``, ``energy.py``, ``validation.py``).
Until that migration is complete, notebooks and scripts should import all
public functions through this module — the public API will not break when the
internals migrate.

**Scalar-only scope:**
Vectorial high-NA focusing and calibrated nonlinear material response are
explicit future hooks.  All propagation is scalar paraxial BL-ASM or SAS.

**Key naming conventions:**
- ``canonical_zone_um`` — axial peak FWHM (``bessel_zone_metrics``).
  This is the single-observable optical zone length.
- ``strict_bessel_region_um`` — triple-intersection fabrication-planning region
  (``bessel_region_metrics``).  Always ≤ canonical zone.
- ``vortex_main_ring_diameter_m`` — bright-ring diameter for ell > 0, from J'_ell.
  NOT the same as ``target_core_diameter_m`` which is the equivalent ell=0 first-zero.
- ``propagation_power_drift_fraction`` - numerical transverse-power drift over
  the propagated z stack. Quantitative metrics are valid only at or below 5 %;
  intentional optical filtering is reported separately.
- ``build_sample_field_ideal`` — deprecated; use ``build_conical_axicon_field_ideal``
  (axicon source plane) or ``build_bessel_gauss_field_ideal`` (true J_ell target).
"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import math
import warnings

import numpy as np
import pandas as pd
import scipy.special as sp

try:
    from scipy.ndimage import gaussian_filter1d
except Exception:  # pragma: no cover
    gaussian_filter1d = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


from vbb_study.config import (
    BeamDesign,
    BeamTarget,
    EnergyBudget,
    EPS,
    GenerationMethod,
    GridConfig,
    LaserConfig,
    MaterialConfig,
    ObjectiveConfig,
    OpticalMappingMode,
    PathKind,
    PhysicalAxiconConfig,
    PropagationConfig,
    PropagationMethod,
    RegimeName,
    RelayConfig,
    SLMConfig,
    SimulationPreset,
    Slm2ConjugateMode,
    StudyKind,
    SurfacePlacement,
    TWOPI,
    TwinConfig,
    ValidityViolationAction,
    cm,
    fs,
    kHz,
    m,
    mm,
    nm,
    uJ,
    um,
)
from vbb_study.design import (
    J0_FIRST_ZERO,
    axial_scan_values,
    compute_design_from_config,
    compute_design_from_targets,
    default_config,
    fixed_objective_map_from_config,
    get_preset,
    objective_map_from_config,
)
from vbb_study import vbb_axicon, vbb_regime, vbb_studies


# ---------------------------------------------------------------------------
# Grids, phase, and FFT helpers
# ---------------------------------------------------------------------------


from vbb_study.equations.fields import (
    compute_kr,
    fft2c,
    gaussian_amplitude,
    gray_to_phase,
    ifft2c,
    make_rect_grid,
    make_xy_grid,
    next_power_of_two,
    phase_to_gray,
    phase_wrap,
    quantize_phase,
)


def apply_orientation(gray: np.ndarray, slm: SLMConfig) -> np.ndarray:
    """Apply lab orientation flips to a grayscale hologram."""

    out = np.asarray(gray).copy()
    if slm.flip_x:
        out = np.fliplr(out)
    if slm.flip_y:
        out = np.flipud(out)
    if slm.rotate_180:
        out = np.rot90(out, 2)
    return out


# ---------------------------------------------------------------------------
# Design and analytic references
# ---------------------------------------------------------------------------


from vbb_study.equations.objective_pupil import headline_length_tags, objective_map_from_design_inputs


def analytic_references(config: TwinConfig, design: Optional[BeamDesign] = None) -> Dict[str, float]:
    """Return Baliyan/Bessel reference quantities for QA and tables."""

    design = design or compute_design_from_config(config)
    k_medium = config.laser.k0 * config.material.refractive_index
    zmax = design.w0_sample_m * k_medium / max(design.kr_sample_m_inv, EPS)
    ell_abs = abs(int(design.ell))
    ring = 0.0 if ell_abs == 0 else float(sp.jnp_zeros(ell_abs, 1)[0] / design.kr_sample_m_inv)
    second = np.nan if ell_abs == 0 else float(sp.jnp_zeros(ell_abs, 2)[1] / design.kr_sample_m_inv)
    return {
        "kr_sample_m_inv": float(design.kr_sample_m_inv),
        "kr_slm_m_inv": float(design.kr_slm_m_inv),
        "gamma_slm_deg": float(design.gamma_slm_deg),
        "zmax_baliyan_um": float(zmax / um),
        "target_scale_definition": str(design.target_scale_definition),
        "target_equivalent_l0_core_diameter_um": float(design.target_equivalent_l0_core_diameter_m / um),
        "core_radius_2405_um": float(J0_FIRST_ZERO / design.kr_sample_m_inv / um),  # legacy key; exact j_0,1 value
        "core_diameter_2405_um": float(2.0 * J0_FIRST_ZERO / design.kr_sample_m_inv / um),  # legacy key; exact j_0,1 value
        "core_first_zero_radius_um": float(design.equivalent_l0_first_zero_radius_m / um),
        "core_first_zero_diameter_um": float(design.equivalent_l0_first_zero_diameter_m / um),
        "equivalent_l0_first_zero_radius_um": float(design.equivalent_l0_first_zero_radius_m / um),
        "equivalent_l0_first_zero_diameter_um": float(design.equivalent_l0_first_zero_diameter_m / um),
        "vortex_first_ring_radius_um": float(ring / um),
        "vortex_first_ring_diameter_um": float(2.0 * ring / um),
        "vortex_second_ring_radius_um": float(second / um) if np.isfinite(second) else np.nan,
        "magnification_to_sample": float(design.magnification_to_sample),
        "mapping_mode": str(design.mapping_mode),
        "objective_map_source": str(design.objective_map_source),
        "objective_map_demag": float(design.objective_map_demag),
        "w0_sample_um": float(design.w0_sample_m / um),
        "predicted_bessel_length_um": float(design.predicted_bessel_length_m / um),
    }


def inverse_design_round_trip(config: TwinConfig, rtol: float = 0.03) -> Dict[str, Any]:
    """Check target recovery under the explicit optical mapping contract."""

    design = compute_design_from_config(config)
    refs = analytic_references(config, design)
    core_err = abs(refs["core_first_zero_diameter_um"] * um - config.target.target_core_diameter_m) / max(config.target.target_core_diameter_m, EPS)
    length_err = abs(refs["zmax_baliyan_um"] * um - config.target.target_bessel_length_m) / max(config.target.target_bessel_length_m, EPS)
    mapping_mode = str(design.mapping_mode)
    target_within_tolerance = bool(core_err <= rtol and length_err <= rtol)
    inverse_mode = mapping_mode == "target_matched_inverse_design"
    return {
        "target_core_um": config.target.target_core_diameter_m / um,
        "recovered_core_um": refs["core_first_zero_diameter_um"],
        "target_length_um": config.target.target_bessel_length_m / um,
        "recovered_length_um": refs["zmax_baliyan_um"],
        "core_relative_error": float(core_err),
        "length_relative_error": float(length_err),
        "mapping_mode": mapping_mode,
        "objective_map_source": str(design.objective_map_source),
        "objective_map_demag": float(design.objective_map_demag),
        "claim_scope": "inverse_design_feasibility" if inverse_mode else "fixed_bench_prediction",
        "hardware_target_achieved": bool((not inverse_mode) and target_within_tolerance),
        "pass": target_within_tolerance,
    }


# ---------------------------------------------------------------------------
# Interface and low-order fitting
# ---------------------------------------------------------------------------


from vbb_study.equations.interface import (
    fit_interface_zernike_terms,
    interface_aberration_pupil,
    interface_correction_phase,
)


# ---------------------------------------------------------------------------
# Hologram and device model
# ---------------------------------------------------------------------------


def _continuous_phase(
    grid: Dict[str, Any],
    config: TwinConfig,
    design: BeamDesign,
    include_blaze: Optional[bool] = None,
    include_correction: Optional[bool] = None,
) -> Dict[str, np.ndarray]:
    include_blaze = config.include_blaze if include_blaze is None else bool(include_blaze)
    include_correction = config.correct_interface if include_correction is None else bool(include_correction)

    R = grid["R"]
    PHI = grid["PHI"]
    X = grid["X"]
    phi_axicon = -design.kr_slm_m_inv * R
    phi_vortex = design.ell * PHI
    phi_blaze = TWOPI * config.slm.carrier_cpm * X if include_blaze else np.zeros_like(R)
    phi_signum = np.zeros_like(R)
    if design.signum_pi_flip:
        J = sp.jv(abs(int(design.ell)), design.kr_slm_m_inv * R)
        phi_signum = np.where(J < 0.0, np.pi, 0.0)
    phi_corr = np.zeros_like(R)
    if include_correction and config.apply_interface:
        phi_corr = interface_correction_phase(grid, config.laser, config.objective, config.material)
    phase = phi_axicon + phi_vortex + phi_signum + phi_blaze + phi_corr
    return {
        "phase_continuous": phase,
        "phase_wrapped": phase_wrap(phase),
        "phase_axicon": phase_wrap(phi_axicon),
        "phase_vortex": phase_wrap(phi_vortex),
        "phase_signum": phase_wrap(phi_signum),
        "phase_blaze": phase_wrap(phi_blaze),
        "phase_interface_correction": phi_corr,
    }


def render_device_hologram(
    config: TwinConfig,
    design: Optional[BeamDesign] = None,
    quantize: Optional[bool] = None,
) -> Dict[str, Any]:
    """Render the exact rectangular device hologram used for SLM upload."""

    design = design or compute_design_from_config(config)
    quantize = config.include_quantization if quantize is None else bool(quantize)
    grid = make_rect_grid(config.slm.resolution_x, config.slm.resolution_y, config.slm.pixel_pitch_m)
    parts = _continuous_phase(grid, config, design)
    phase = quantize_phase(parts["phase_continuous"], config.slm.phase_bits) if quantize else parts["phase_wrapped"]
    gray = apply_orientation(phase_to_gray(phase, config.slm.phase_bits, invert=config.slm.invert_gray), config.slm)
    info = {
        "resolution_x": config.slm.resolution_x,
        "resolution_y": config.slm.resolution_y,
        "pixel_pitch_um": config.slm.pixel_pitch_m / um,
        "phase_bits": config.slm.phase_bits,
        "blaze_period_px": config.slm.blaze_period_px,
        "carrier_lpmm": config.slm.carrier_lpmm,
        "gamma_slm_deg": design.gamma_slm_deg,
        "ell": design.ell,
        "correct_interface": bool(config.correct_interface),
        "write_depth_um": config.material.write_depth_m / um,
    }
    return {"grid": grid, "phase": phase, "gray": gray, "info": info, **parts}


def _reduced_device_grid(config: TwinConfig) -> Dict[str, Any]:
    ds = max(1, int(config.grid.device_downsample))
    nx = int(math.ceil(config.slm.resolution_x / ds))
    ny = int(math.ceil(config.slm.resolution_y / ds))
    dx = config.slm.pixel_pitch_m * ds
    return make_rect_grid(nx, ny, dx)


def _pad_rect_to_square(U_rect: np.ndarray, N: int) -> np.ndarray:
    ny, nx = U_rect.shape
    if N < max(nx, ny):
        raise ValueError(f"Grid N={N} is too small for rectangular device field {nx}x{ny}.")
    out = np.zeros((N, N), dtype=complex)
    y0 = (N - ny) // 2
    x0 = (N - nx) // 2
    out[y0 : y0 + ny, x0 : x0 + nx] = U_rect
    return out


def _pad_mask_to_square(M_rect: np.ndarray, N: int) -> np.ndarray:
    ny, nx = M_rect.shape
    out = np.zeros((N, N), dtype=float)
    y0 = (N - ny) // 2
    x0 = (N - nx) // 2
    out[y0 : y0 + ny, x0 : x0 + nx] = M_rect
    return out


def fill_factor_amplitude(grid: Dict[str, Any], pixel_pitch_m: float, fill_factor: float) -> np.ndarray:
    """Effective fill-factor amplitude mask.

    With one sample per displayed pixel the subpixel inactive border is not
    spatially resolved, so a uniform sqrt(fill_factor) amplitude preserves the
    intended energy throughput.
    """

    ff = float(np.clip(fill_factor, 0.0, 1.0))
    if ff >= 1.0:
        return np.ones_like(grid["R"], dtype=float)
    dx = float(grid["dx"])
    if dx >= 0.5 * float(pixel_pitch_m):
        return np.sqrt(ff) * np.ones_like(grid["R"], dtype=float)

    duty = math.sqrt(ff)
    xmod = np.mod(grid["X"] / pixel_pitch_m + 0.5, 1.0) - 0.5
    ymod = np.mod(grid["Y"] / pixel_pitch_m + 0.5, 1.0) - 0.5
    return ((np.abs(xmod) <= 0.5 * duty) & (np.abs(ymod) <= 0.5 * duty)).astype(float)


def build_realistic_slm_field(
    config: TwinConfig,
    design: Optional[BeamDesign] = None,
    include_quantization: Optional[bool] = None,
    include_fill_factor: Optional[bool] = None,
    include_active_aperture: Optional[bool] = None,
    include_blaze: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build the realistic SLM/pupil field that is propagated to the sample."""

    design = design or compute_design_from_config(config)
    include_quantization = config.include_quantization if include_quantization is None else bool(include_quantization)
    include_fill_factor = config.include_fill_factor if include_fill_factor is None else bool(include_fill_factor)
    include_active_aperture = config.include_active_aperture if include_active_aperture is None else bool(include_active_aperture)

    rect_grid = _reduced_device_grid(config)
    defer_interface_correction = bool(config.correct_interface and config.apply_interface)
    parts = _continuous_phase(
        rect_grid,
        config,
        design,
        include_blaze=include_blaze,
        include_correction=bool(config.correct_interface and not defer_interface_correction),
    )
    # I defer the interface precompensation until after first-order isolation.
    # The correction is still a unit-modulus pupil phase, but sending its
    # many-wave low-order curvature through the finite carrier filter clips the
    # corrected order and masquerades as a bad focus. This keeps the uncorrected
    # path untouched while modelling the intended pupil-plane conjugation.
    deferred_phi_corr = (
        interface_correction_phase(rect_grid, config.laser, config.objective, config.material)
        if defer_interface_correction
        else np.zeros_like(rect_grid["R"])
    )
    phase = quantize_phase(parts["phase_continuous"], config.slm.phase_bits) if include_quantization else parts["phase_wrapped"]
    amp = gaussian_amplitude(rect_grid["R"], config.laser.beam_radius_on_slm_m)

    fill = fill_factor_amplitude(rect_grid, config.slm.pixel_pitch_m, config.slm.fill_factor) if include_fill_factor else np.ones_like(amp)
    if include_active_aperture:
        aperture = (
            (np.abs(rect_grid["X"]) <= 0.5 * config.slm.active_width_m)
            & (np.abs(rect_grid["Y"]) <= 0.5 * config.slm.active_height_m)
        ).astype(float)
    else:
        aperture = np.ones_like(amp)
    U_rect = amp * fill * aperture * np.exp(1j * phase)

    N = int(config.grid.N)
    square = make_xy_grid(N, rect_grid["dx"])
    U = _pad_rect_to_square(U_rect, N)
    fill_sq = _pad_mask_to_square(fill, N)
    aperture_sq = _pad_mask_to_square(aperture, N)
    phase_sq = _pad_mask_to_square(phase, N)
    correction_sq = _pad_mask_to_square(deferred_phi_corr, N)
    return {
        "U": U,
        "grid": square,
        "rect_grid": rect_grid,
        "phase": phase_sq,
        "pupil_interface_correction_phase": correction_sq,
        "interface_correction_deferred_to_pupil": defer_interface_correction,
        "fill": fill_sq,
        "aperture": aperture_sq,
        "device_downsample": int(config.grid.device_downsample),
        "effective_device_dx_m": rect_grid["dx"],
        **parts,
    }


def isolate_first_order(
    U: np.ndarray,
    grid: Dict[str, Any],
    slm: SLMConfig,
    filter_radius_lpmm: Optional[float] = None,
    recenter: bool = True,
) -> Dict[str, Any]:
    """Fourier-plane first-order isolation around the blaze carrier."""

    A = fft2c(U)
    FX, FY = grid["FX"], grid["FY"]
    fc = slm.carrier_cpm
    radius = (slm.first_order_filter_radius_lpmm if filter_radius_lpmm is None else float(filter_radius_lpmm)) * 1e3
    mask = ((FX - fc) ** 2 + FY**2) <= radius**2
    A_selected = A * mask
    total = float(np.sum(np.abs(A) ** 2)) + EPS
    selected = float(np.sum(np.abs(A_selected) ** 2))
    power = np.abs(A) ** 2
    if np.any(mask):
        masked_power = np.where(mask, power, 0.0)
        peak_iy, peak_ix = np.unravel_index(int(np.argmax(masked_power)), masked_power.shape)
        peak_fx = float(FX[peak_iy, peak_ix])
        peak_fy = float(FY[peak_iy, peak_ix])
    else:
        peak_fx = peak_fy = float("nan")
    if recenter:
        fx = np.fft.fftshift(np.fft.fftfreq(int(grid["N"]), d=float(grid["dx"])))
        order_ix = int(np.argmin(np.abs(fx - fc)))
        center_ix = int(np.argmin(np.abs(fx)))
        A_ifft = np.roll(A_selected, center_ix - order_ix, axis=1)
    else:
        A_ifft = A_selected
    return {
        "U_selected": ifft2c(A_ifft),
        "A": A,
        "A_selected": A_selected,
        "order_mask": mask,
        "selected_fraction": float(selected / total),
        "carrier_lpmm": float(fc / 1e3),
        "filter_radius_lpmm": float(radius / 1e3),
        "order_peak_fx_lpmm": float(peak_fx / 1e3),
        "order_peak_fy_lpmm": float(peak_fy / 1e3),
        "order_peak_distance_to_carrier_lpmm": float(math.hypot(peak_fx - fc, peak_fy) / 1e3)
        if np.isfinite(peak_fx) and np.isfinite(peak_fy)
        else float("nan"),
    }


def first_order_filter_geometry(
    grid: Dict[str, Any],
    slm: SLMConfig,
    design: Optional[BeamDesign] = None,
    filter_radius_lpmm: Optional[float] = None,
    *,
    margin_bins: float = 2.0,
) -> Dict[str, Any]:
    """Return a first-order filter radius that contains the encoded cone.

    The SLM blaze translates the axicon/vortex spectrum by the carrier.  The
    Fourier stop therefore has to pass the conical spectrum radius, not just a
    small spot at the carrier.  If the requested stop would overlap the zero
    order, the usable stop is clipped and the geometry is flagged as invalid.
    """

    configured = float(slm.first_order_filter_radius_lpmm if filter_radius_lpmm is None else filter_radius_lpmm)
    N = int(grid.get("N", 0))
    dx = float(grid.get("dx", 0.0))
    bin_lpmm = float(1.0 / (max(N, 1) * max(dx, EPS)) / 1e3)
    cone_lpmm = 0.0
    if design is not None:
        cone_lpmm = float(abs(design.kr_slm_m_inv) / TWOPI / 1e3)
    requested = max(configured, cone_lpmm + float(margin_bins) * bin_lpmm if design is not None else configured)
    carrier_lpmm = float(abs(slm.carrier_lpmm))
    geometry_valid = bool(carrier_lpmm > requested + 0.5 * bin_lpmm)
    effective = requested
    if not geometry_valid and carrier_lpmm > EPS:
        effective = max(bin_lpmm, 0.95 * carrier_lpmm)
    return {
        "configured_filter_radius_lpmm": configured,
        "recommended_filter_radius_lpmm": float(requested),
        "effective_filter_radius_lpmm": float(effective),
        "carrier_lpmm": float(carrier_lpmm),
        "axicon_cone_radius_lpmm": float(cone_lpmm),
        "frequency_bin_lpmm": float(bin_lpmm),
        "first_order_geometry_valid": geometry_valid,
        "first_order_geometry_margin_lpmm": float(carrier_lpmm - requested),
    }


def export_hologram_png(
    config: TwinConfig,
    out_dir: str | Path = "outputs/holograms",
    basename: Optional[str] = None,
    design: Optional[BeamDesign] = None,
) -> Dict[str, Path]:
    """Export exact SLM PNG plus JSON metadata."""

    if Image is None:
        raise RuntimeError("Pillow is required for PNG export.")
    design = design or compute_design_from_config(config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if basename is None:
        basename = (
            f"PHAROS_1029nm_Bessel_l{design.ell}_"
            f"D{design.target_core_diameter_m/um:.2f}um_"
            f"L{design.target_bessel_length_m/um:.0f}um_"
            f"z{config.material.write_depth_m/um:.0f}um"
        ).replace(".", "p")
    rendered = render_device_hologram(config, design)
    png_path = out / f"{basename}.png"
    json_path = out / f"{basename}.json"
    Image.fromarray(rendered["gray"], mode="L").save(png_path)
    meta = {
        "model": "PHAROS + SLM Bessel digital twin",
        "laser": asdict(config.laser),
        "slm": asdict(config.slm),
        "objective": asdict(config.objective),
        "material": asdict(config.material),
        "energy": asdict(config.energy),
        "target": asdict(config.target),
        "propagation": asdict(config.propagation),
        "design": asdict(design),
        "hologram_info": rendered["info"],
        "notes": "Exact rectangular upload hologram: wrapped phase quantised to device bit depth.",
    }
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"png": png_path, "json": json_path}


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------


from vbb_study.equations.propagation import (
    _kz_medium,
    _sas_z_limit_m,
    _transfer_function_medium,
    _zero_pad_center,
    _zero_unpad_center,
    angular_spectrum_propagate_bl,
    bandlimit_mask_matsushima,
    focus_to_focal_plane,
    make_bl_asm_propagator,
    sas_validity_report,
    scalable_angular_spectrum_propagate,
)


def _crop_grid(grid: Dict[str, Any], sl: slice) -> Dict[str, Any]:
    x = grid["x"][sl]
    X, Y = np.meshgrid(x, x, indexing="xy")
    R = np.hypot(X, Y)
    PHI = np.arctan2(Y, X)
    return {"N": len(x), "dx": grid["dx"], "x": x, "X": X, "Y": Y, "R": R, "PHI": PHI}


def _grid_summary(grid: Dict[str, Any]) -> Dict[str, float | int]:
    x = np.asarray(grid["x"], dtype=float)
    return {
        "N": int(grid["N"]),
        "dx_m": float(grid["dx"]),
        "x_min_m": float(x[0]),
        "x_max_m": float(x[-1]),
        "side_length_m": float(int(grid["N"]) * float(grid["dx"])),
    }


def _resample_intensity_to_common_grid(I: np.ndarray, source_grid: Dict[str, Any], target_x: np.ndarray) -> np.ndarray:
    """Bilinearly resample a square real intensity image onto target_x/target_x."""

    src_x = np.asarray(source_grid["x"], dtype=float)
    tgt = np.asarray(target_x, dtype=float)
    arr = np.asarray(I, dtype=float)
    rows = np.empty((arr.shape[0], len(tgt)), dtype=float)
    for row in range(arr.shape[0]):
        rows[row, :] = np.interp(tgt, src_x, arr[row, :], left=0.0, right=0.0)
    out = np.empty((len(tgt), len(tgt)), dtype=float)
    for col in range(len(tgt)):
        out[:, col] = np.interp(tgt, src_x, rows[:, col], left=0.0, right=0.0)
    return out


PROPAGATION_POWER_DRIFT_QUANTITATIVE_MAX = 0.05


def _propagation_power_drift_from_values(total_power: Any) -> float:
    power = np.asarray(total_power, dtype=float)
    power = power[np.isfinite(power)]
    if power.size < 2:
        return float("nan")
    return float((np.max(power) - np.min(power)) / (np.mean(power) + EPS))


def _propagation_power_label(drift: float) -> str:
    if not np.isfinite(drift):
        return "not_evaluated"
    if drift <= PROPAGATION_POWER_DRIFT_QUANTITATIVE_MAX:
        return "pass"
    if drift <= 0.20:
        return "marginal"
    return "fail"


def propagation_power_validity_report(volume: Dict[str, Any]) -> Dict[str, Any]:
    """Return numerical-propagation validity from one authoritative drift.

    Optical filtering before propagation is intentionally outside this report.
    The drift compares propagated transverse powers across the evaluated axial
    stack and therefore governs only numerical propagation convergence.
    """

    raw = volume.get("propagation_power_drift_fraction", np.nan)
    drift = float(raw) if raw is not None else float("nan")
    if not np.isfinite(drift):
        drift = _propagation_power_drift_from_values(volume.get("total_power", []))
    evaluated = bool(np.isfinite(drift))
    valid = bool(evaluated and drift <= PROPAGATION_POWER_DRIFT_QUANTITATIVE_MAX)
    if not evaluated:
        reason = "numerical propagation power drift was not evaluated over at least two finite planes"
        violations = ["propagation_power_drift_not_evaluated"]
    elif not valid:
        reason = (
            f"numerical propagation power drift {drift:.6g} exceeds the "
            f"quantitative limit {PROPAGATION_POWER_DRIFT_QUANTITATIVE_MAX:.2f}"
        )
        violations = ["propagation_power_drift_exceeds_0p05"]
    else:
        reason = ""
        violations = []
    return {
        "validity_name": "Propagation quantitative validity",
        "valid": valid,
        "violations": violations,
        "quantitative_metrics_valid": valid,
        "quantitative_metrics_invalid_reason": reason,
        "propagation_power_drift_evaluated": evaluated,
        "propagation_power_drift_fraction": drift,
        "propagation_power_label": _propagation_power_label(drift),
        "quantitative_power_drift_limit_fraction": PROPAGATION_POWER_DRIFT_QUANTITATIVE_MAX,
    }


def _propagation_power_metric_fields(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "quantitative_metrics_valid": bool(report["quantitative_metrics_valid"]),
        "quantitative_metrics_invalid_reason": str(report["quantitative_metrics_invalid_reason"]),
        "propagation_power_drift_evaluated": bool(report["propagation_power_drift_evaluated"]),
        "propagation_power_drift_fraction": float(report["propagation_power_drift_fraction"]),
        "propagation_power_label": str(report["propagation_power_label"]),
        "quantitative_power_drift_limit_fraction": float(report["quantitative_power_drift_limit_fraction"]),
    }


def enforce_propagation_power_validity(
    volume: Dict[str, Any],
    on_violation: ValidityViolationAction = "flag",
) -> Dict[str, Any]:
    """Apply the shared flag/warn/raise policy to numerical power drift."""

    report = propagation_power_validity_report(volume)
    report["on_violation"] = str(on_violation)
    return vbb_regime.enforce_validity(report, on_violation)


def propagate_volume(
    U0: np.ndarray,
    grid: Dict[str, Any],
    wavelength_m: float,
    z_values_m: Sequence[float],
    n_medium: float = 1.0,
    crop_pixels: Optional[int] = None,
    bandlimit: bool = True,
    method: PropagationMethod = "bl_asm",
    propagation_config: Optional[PropagationConfig] = None,
) -> Dict[str, Any]:
    """Propagate a stack and keep compact metrics/planes."""

    z = np.asarray(z_values_m, dtype=float)
    N = int(grid["N"])
    c = N // 2
    if crop_pixels is None:
        h = c
    else:
        h = max(2, min(int(crop_pixels), N) // 2)
    sl = slice(c - h, c + h)
    crop_grid = _crop_grid(grid, sl)
    xz = np.zeros((2 * h, len(z)), dtype=np.float32)
    intensity_stack = np.zeros((len(z), 2 * h, 2 * h), dtype=np.float32)
    peak = np.zeros(len(z), dtype=float)
    onaxis = np.zeros(len(z), dtype=float)
    total_power = np.zeros(len(z), dtype=float)
    planes: Dict[str, np.ndarray] = {}
    prop_cfg = propagation_config or PropagationConfig(method=method, sas_bandlimit=bandlimit)
    method_key = str(prop_cfg.method).lower().strip()
    prop = make_bl_asm_propagator(U0, grid, wavelength_m, n_medium=n_medium, bandlimit=bandlimit) if method_key == "bl_asm" else None
    best_peak = -np.inf
    best_idx = 0
    sas_meta: List[Dict[str, Any]] = []
    native_grids: List[Dict[str, float | int]] = []
    retained = np.ones(len(z), dtype=float)
    output_dx = np.full(len(z), float(grid["dx"]), dtype=float)
    for i, zi in enumerate(z):
        if method_key == "sas":
            U, native_grid, meta = scalable_angular_spectrum_propagate(
                U0,
                grid,
                wavelength_m,
                float(zi),
                n_medium=n_medium,
                pad_factor=prop_cfg.sas_pad_factor,
                bandlimit=prop_cfg.sas_bandlimit,
                skip_final_phase=prop_cfg.sas_skip_final_phase,
                allow_invalid=prop_cfg.sas_allow_invalid,
            )
            sas_meta.append(meta)
            native_grids.append(_grid_summary(native_grid))
            retained[i] = float(meta.get("retained_power_fraction", np.nan))
            output_dx[i] = float(native_grid["dx"])
            I = np.abs(U) ** 2
            Ic = _resample_intensity_to_common_grid(I, native_grid, crop_grid["x"])
            native_line = I[int(native_grid["N"]) // 2, :]
            xz[:, i] = np.interp(crop_grid["x"], native_grid["x"], native_line, left=0.0, right=0.0).astype(np.float32)
            total_power[i] = float(np.sum(I) * native_grid["dx"] ** 2)
        elif method_key == "bl_asm":
            assert prop is not None
            U = prop(float(zi))
            I = np.abs(U) ** 2
            Ic = I[sl, sl]
            xz[:, i] = I[c, sl].astype(np.float32)
            total_power[i] = float(np.sum(I) * grid["dx"] ** 2)
        else:
            raise ValueError(f"Unsupported propagation method: {prop_cfg.method}")
        peak[i] = float(np.max(Ic))
        onaxis[i] = float(Ic[h, h])
        intensity_stack[i] = Ic.astype(np.float32)
        if peak[i] > best_peak:
            best_peak = peak[i]
            best_idx = i
            planes["peak"] = Ic.astype(np.float32)
        if i == 0:
            planes["start"] = Ic.astype(np.float32)
        if i == len(z) // 2:
            planes["middle"] = Ic.astype(np.float32)
        if i == len(z) - 1:
            planes["end"] = Ic.astype(np.float32)
    propagation_power_drift = _propagation_power_drift_from_values(total_power)
    power_report = propagation_power_validity_report({
        "propagation_power_drift_fraction": propagation_power_drift,
        "total_power": total_power,
    })
    return {
        "z": z,
        "xz": xz,
        "intensity_stack": intensity_stack,
        "peak": peak,
        "onaxis": onaxis,
        "total_power": total_power,
        "planes": planes,
        "peak_index": int(best_idx),
        "crop_grid": crop_grid,
        "propagation_method": method_key,
        "output_dx_m": output_dx,
        "retained_power_fraction": retained,
        "native_grids": native_grids,
        "sas_metadata": sas_meta,
        **_propagation_power_metric_fields(power_report),
    }


from vbb_study.equations.scalar_bessel import (
    build_bessel_gauss_field_ideal,
    build_conical_axicon_field_ideal,
    build_sample_field_ideal,
)


def realistic_slm_to_sample(
    config: TwinConfig,
    design: Optional[BeamDesign] = None,
    z_values_m: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Primary high-fidelity scalar path from SLM device realism to sample.

    The propagated axial coordinate is reported as a forward-only sample-space
    distance from the nominal focal plane, so downstream figures never mix
    signed focal-plane offsets with forward propagation distance.
    """

    design = design or compute_design_from_config(config)
    slm_field = build_realistic_slm_field(config, design)
    grid = slm_field["grid"]
    U = slm_field["U"]

    order: Dict[str, Any]
    if config.include_first_order_isolation and config.include_blaze:
        filter_geometry = first_order_filter_geometry(grid, config.slm, design)
        order = isolate_first_order(
            U,
            grid,
            config.slm,
            filter_radius_lpmm=filter_geometry["effective_filter_radius_lpmm"],
        )
        order.update(filter_geometry)
        U = order["U_selected"]
    else:
        order = {
            "U_selected": U,
            "selected_fraction": 1.0,
            "carrier_lpmm": 0.0,
            "filter_radius_lpmm": 0.0,
            "order_mask": np.ones_like(U, dtype=bool),
        }

    if bool(slm_field.get("interface_correction_deferred_to_pupil", False)):
        U = U * np.exp(1j * np.asarray(slm_field["pupil_interface_correction_phase"], dtype=float))

    pupil = (grid["R"] <= config.objective.pupil_radius_m).astype(float)
    U = U * pupil

    W = np.zeros_like(grid["R"])
    if config.apply_interface:
        W = interface_aberration_pupil(grid, config.laser, config.objective, config.material)
        U = U * np.exp(1j * W)
    zern = fit_interface_zernike_terms(grid, W, config.objective.pupil_radius_m) if config.apply_interface else {}

    U_focus, focal_grid = focus_to_focal_plane(U, grid, config.laser, config.objective)
    prop = make_bl_asm_propagator(
        U_focus,
        focal_grid,
        config.laser.wavelength_m,
        n_medium=config.material.refractive_index,
        bandlimit=True,
    )

    span = float(config.grid.axial_range_m)
    coarse_max = max(
        2.0 * float(config.grid.coarse_scan_factor) * span,
        1.30 * float(design.target_bessel_length_m),
    )
    coarse = np.linspace(0.0, coarse_max, int(config.grid.coarse_scan_points))
    coarse_peak = []
    c = int(focal_grid["N"] // 2)
    h = max(8, min(config.grid.crop_pixels, focal_grid["N"]) // 4)
    sl = slice(c - h, c + h)
    for zi in coarse:
        I = np.abs(prop(float(zi))) ** 2
        coarse_peak.append(float(np.max(I[sl, sl])))
    z0 = float(coarse[int(np.argmax(coarse_peak))])
    if z_values_m is None:
        z_values = axial_scan_values(config, design, z_anchor_m=z0)
    else:
        z_values = np.asarray(z_values_m, dtype=float)
        if z_values.ndim != 1 or z_values.size == 0:
            raise ValueError("z_values_m must be a non-empty 1D sequence.")
    volume = propagate_volume(
        U_focus,
        focal_grid,
        config.laser.wavelength_m,
        z_values,
        n_medium=config.material.refractive_index,
        crop_pixels=config.grid.crop_pixels,
        bandlimit=True,
        method=config.propagation.method,
        propagation_config=config.propagation,
    )
    return {
        "path": "realistic",
        "design": design,
        "slm_field": slm_field,
        "order": order,
        "pupil_grid": grid,
        "pupil_phase_interface": W,
        "interface_zernike_fit": zern,
        "U_focus": U_focus,
        "focal_grid": focal_grid,
        "z_zone_centre_m": z0,
        "volume": volume,
        "first_order_selected_fraction": float(order["selected_fraction"]),
    }


# ---------------------------------------------------------------------------
# Metrics and QA
# ---------------------------------------------------------------------------


def radial_profile(I: np.ndarray, grid: Dict[str, Any], bins: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Azimuthal mean radial profile."""

    r = grid["R"].ravel()
    v = np.asarray(I, float).ravel()
    bins = int(grid["N"] // 2 if bins is None else bins)
    edges = np.linspace(0.0, float(np.max(r)), max(8, bins) + 1)
    idx = np.digitize(r, edges) - 1
    valid = (idx >= 0) & (idx < len(edges) - 1)
    sums = np.bincount(idx[valid], weights=v[valid], minlength=len(edges) - 1)
    cnt = np.bincount(idx[valid], minlength=len(edges) - 1)
    prof = np.divide(sums, cnt, out=np.zeros_like(sums), where=cnt > 0)
    rc = 0.5 * (edges[:-1] + edges[1:])
    return rc, prof


def _smooth_profile(y: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    if gaussian_filter1d is None:
        return np.asarray(y, float)
    return gaussian_filter1d(np.asarray(y, float), sigma=sigma, mode="nearest")


from vbb_study.equations.metrics import _contiguous_mask_zone, bessel_region_metrics, extract_radial_metrics


def bessel_zone_metrics(z: np.ndarray, peak: np.ndarray, level: float = 0.5) -> Dict[str, float | bool]:
    """Canonical axial-peak FWHM zone around the peak-in-plane maximum.

    ``bessel_zone_um`` is deliberately only this single-observable FWHM length.
    The stricter fabrication-planning intersection is reported separately as
    ``bessel_region_um``/``strict_bessel_region_um``.
    """

    z = np.asarray(z, float)
    y = np.asarray(peak, float)
    if z.size == 0 or not np.any(np.isfinite(y)) or np.nanmax(y) <= 0:
        return {"bessel_zone_um": 0.0, "zone_start_um": np.nan, "zone_end_um": np.nan, "zone_capped": False}
    yn = y / (float(np.nanmax(y)) + EPS)
    ref = int(np.nanargmax(yn))
    if yn[ref] < level:
        return {"bessel_zone_um": 0.0, "zone_start_um": float(z[ref] / um), "zone_end_um": float(z[ref] / um), "zone_capped": False}
    i0 = ref
    while i0 > 0 and yn[i0 - 1] >= level:
        i0 -= 1
    i1 = ref
    while i1 < len(yn) - 1 and yn[i1 + 1] >= level:
        i1 += 1

    def interp_edge(ia: int, ib: int) -> float:
        if ia == ib:
            return float(z[ia])
        ya, yb = yn[ia], yn[ib]
        if abs(yb - ya) < 1e-12:
            return float(0.5 * (z[ia] + z[ib]))
        return float(z[ia] + (level - ya) * (z[ib] - z[ia]) / (yb - ya))

    z_start = float(z[0] if i0 == 0 else interp_edge(i0 - 1, i0))
    z_end = float(z[-1] if i1 == len(z) - 1 else interp_edge(i1, i1 + 1))
    return {
        "bessel_zone_um": float(max(0.0, z_end - z_start) / um),
        "zone_start_um": float(z_start / um),
        "zone_end_um": float(z_end / um),
        "zone_capped": bool(i0 == 0 or i1 == len(z) - 1),
    }


def absolute_zone_length_um(z: np.ndarray, peak: np.ndarray, threshold: float) -> float:
    """Axial length where peak-in-plane exceeds a fixed absolute threshold."""

    z = np.asarray(z, float)
    y = np.asarray(peak, float)
    mask = y >= float(threshold)
    if not np.any(mask):
        return 0.0
    return float((np.max(z[mask]) - np.min(z[mask])) / um)


def fluence_from_intensity(I: np.ndarray, dx_m: float, pulse_energy_J: float) -> np.ndarray:
    """Energy-conserving 2D fluence in J/cm^2."""

    positive = np.maximum(np.asarray(I, float), 0.0)
    denom = float(np.sum(positive) * dx_m * dx_m) + EPS
    return (float(pulse_energy_J) * positive / denom) / 1e4


def line_fluence_proxy_xz(xz: np.ndarray, dx_m: float, pulse_energy_J: float) -> np.ndarray:
    """Non-energy-conserving line fluence proxy for axial trend plots."""

    positive = np.maximum(np.asarray(xz, float), 0.0)
    denom = np.sum(positive, axis=0, keepdims=True) * float(dx_m) + EPS
    return (float(pulse_energy_J) * positive / denom) / 1e4


def fluence_metrics(
    I_plane: np.ndarray,
    grid: Dict[str, Any],
    radial: Dict[str, Any],
    config: TwinConfig,
    pulse_energy_J: Optional[float] = None,
) -> Dict[str, float]:
    """Peak/core/side-lobe fluence metrics from one XY plane.

    ``side_to_core_peak_ratio`` compares the brightest excluded side lobe with
    the bright feature peak. For vortices the feature is the annular HWHM
    bucket; for ell=0 it is the bright-core HWHM disk.
    """

    energy_J = config.energy.pulse_energy_at_sample_J if pulse_energy_J is None else float(pulse_energy_J)
    F = fluence_from_intensity(I_plane, grid["dx"], energy_J)
    R = grid["R"]
    ell_abs = abs(int(config.target.ell))
    if ell_abs == 0:
        core_mask = R <= max(radial["core_radius_m"], grid["dx"])
    else:
        core_mask = (R >= radial["r_half_inner_m"]) & (R <= radial["r_half_outer_m"])
        if not np.any(core_mask):
            core_mask = np.abs(R - radial["ring_radius_m"]) <= max(radial["ring_width_m"], 2 * grid["dx"])
    feature_radius = max(float(radial["feature_radius_m"]), float(radial["core_radius_m"]), grid["dx"])
    side_mask = R >= config.material.side_lobe_exclusion_radius_factor * feature_radius
    core_peak = float(np.max(F[core_mask])) if np.any(core_mask) else float("nan")
    side_peak = float(np.max(F[side_mask])) if np.any(side_mask) else float("nan")
    return {
        "peak_fluence_J_cm2": float(np.max(F)),
        "core_or_ring_peak_fluence_J_cm2": core_peak,
        "side_lobe_peak_fluence_J_cm2": side_peak,
        "side_to_core_peak_ratio": float(side_peak / (core_peak + EPS)) if np.isfinite(side_peak) and np.isfinite(core_peak) else np.nan,
        "plane_fluence_integral_J": float(np.sum(F) * 1e4 * grid["dx"] * grid["dx"]),
    }


def modification_proxy_metrics(
    volume: Dict[str, Any],
    config: TwinConfig,
    pulse_energy_J: Optional[float] = None,
) -> Dict[str, float]:
    """Thresholded XZ planning proxy based on the labelled line fluence."""

    energy_J = config.energy.pulse_energy_at_sample_J if pulse_energy_J is None else float(pulse_energy_J)
    xzF = line_fluence_proxy_xz(volume["xz"], volume["crop_grid"]["dx"], energy_J)
    threshold = config.material.incubated_threshold_J_cm2(config.laser.rep_rate_Hz)
    mask = xzF >= threshold
    x = volume["crop_grid"]["x"]
    z = volume["z"]
    if np.any(mask):
        inds = np.argwhere(mask)
        ix0, iz0 = inds.min(axis=0)
        ix1, iz1 = inds.max(axis=0)
        length = float(max(0.0, z[iz1] - z[iz0]) / um)
        width = float(max(0.0, x[ix1] - x[ix0]) / um)
        dz = float(abs(z[1] - z[0])) if len(z) > 1 else 0.0
        area = float(np.count_nonzero(mask) * volume["crop_grid"]["dx"] * dz / (um**2))
    else:
        length = width = area = 0.0
    return {
        "effective_pulses": float(config.material.effective_pulses(config.laser.rep_rate_Hz)),
        "incubated_threshold_J_cm2": float(threshold),
        "line_proxy_threshold_length_um": length,
        "line_proxy_threshold_width_um": width,
        "line_proxy_threshold_area_um2": area,
    }


def extract_vortex_safe_metrics(result: Dict[str, Any], config: TwinConfig) -> Dict[str, Any]:
    """Collect optical, fluence, and material-proxy metrics for one result."""

    volume = result["volume"]
    plane = volume["planes"]["peak"]
    cgrid = volume["crop_grid"]
    design = result["design"]
    radial = extract_radial_metrics(plane, cgrid, design.ell, design.kr_sample_m_inv)
    zone = bessel_zone_metrics(volume["z"], volume["peak"], level=0.5)
    strict_region = bessel_region_metrics(volume, design)
    region = dict(strict_region)
    region["bessel_region_definition"] = "strict_intersection_peak_power_radius"
    region.update(
        {
            "strict_bessel_region_um": float(strict_region.get("bessel_region_um", np.nan)),
            "strict_bessel_region_start_um": float(strict_region.get("bessel_region_start_um", np.nan)),
            "strict_bessel_region_end_um": float(strict_region.get("bessel_region_end_um", np.nan)),
            "strict_bessel_region_capped": bool(strict_region.get("bessel_region_capped", False)),
            "canonical_zone_um": float(zone.get("bessel_zone_um", np.nan)),
            "canonical_zone_start_um": float(zone.get("zone_start_um", np.nan)),
            "canonical_zone_end_um": float(zone.get("zone_end_um", np.nan)),
            "canonical_zone_capped": bool(zone.get("zone_capped", False)),
            "canonical_zone_definition": "axial_peak_fwhm",
        }
    )
    study_kind = str(result.get("study_kind", getattr(config, "study_kind", "full_source_to_sample")))
    is_beam_study = study_kind == "beam_to_surface"
    objective_map = objective_map_from_config(config, design)
    try:
        legacy_objective_map = fixed_objective_map_from_config(config)
    except ValueError:
        legacy_objective_map = objective_map
    pulse_energy_for_fluence_J = (
        config.energy.pulse_energy_at_surface_air_J if is_beam_study else config.energy.pulse_energy_at_sample_J
    )
    flu = fluence_metrics(plane, cgrid, radial, config, pulse_energy_J=pulse_energy_for_fluence_J)
    mod = (
        {
            "effective_pulses": np.nan,
            "incubated_threshold_J_cm2": np.nan,
            "line_proxy_threshold_length_um": 0.0,
            "line_proxy_threshold_width_um": 0.0,
            "line_proxy_threshold_area_um2": 0.0,
        }
        if is_beam_study
        else modification_proxy_metrics(volume, config, pulse_energy_J=pulse_energy_for_fluence_J)
    )
    peak_onaxis = float(np.max(volume["onaxis"]))
    peak_in_plane = float(np.max(volume["peak"]))
    peak_idx = int(volume.get("peak_index", int(np.argmax(volume["peak"]))))
    output_dx = np.asarray(volume.get("output_dx_m", [cgrid["dx"]]), dtype=float)
    retained = np.asarray(volume.get("retained_power_fraction", [1.0]), dtype=float)
    sas_meta = volume.get("sas_metadata", [])
    if sas_meta:
        z_limits = np.asarray([float(m.get("z_limit_m", np.nan)) for m in sas_meta], dtype=float)
        z_over = np.asarray([float(m.get("z_over_limit", np.nan)) for m in sas_meta], dtype=float)
        mags = np.asarray([float(m.get("output_magnification", np.nan)) for m in sas_meta], dtype=float)
        band_ret = np.asarray([float(m.get("bandlimit_retained_fraction", np.nan)) for m in sas_meta], dtype=float)
    else:
        z_limits = z_over = mags = band_ret = np.asarray([np.nan], dtype=float)
    order_meta = result.get("order", result.get("axicon_metadata", {}))
    metrics = {
        "ell": int(design.ell),
        "target_core_diameter_um": design.target_core_diameter_m / um,
        "target_scale_definition": str(design.target_scale_definition),
        "target_equivalent_l0_core_diameter_um": design.target_equivalent_l0_core_diameter_m / um,
        "equivalent_l0_first_zero_radius_um": design.equivalent_l0_first_zero_radius_m / um,
        "equivalent_l0_first_zero_diameter_um": design.equivalent_l0_first_zero_diameter_m / um,
        "vortex_main_ring_radius_um": design.vortex_main_ring_radius_m / um,
        "vortex_main_ring_diameter_um": design.vortex_main_ring_diameter_m / um,
        "target_bessel_length_um": design.target_bessel_length_m / um,
        "predicted_bessel_length_um": design.predicted_bessel_length_m / um,
        "gamma_slm_deg": design.gamma_slm_deg,
        "mapping_mode": str(design.mapping_mode),
        # Preserve the historical field while making the authoritative map
        # contract explicit in objective_map_* below.
        "magnification_to_sample": float(legacy_objective_map.demag),
        "objective_map_demag": float(objective_map.demag),
        "objective_map_source": str(getattr(objective_map, "source", "unknown")),
        "legacy_objective_map_demag": float(legacy_objective_map.demag),
        "legacy_objective_map_source": str(getattr(legacy_objective_map, "source", "unknown")),
        "waist_matched_design_magnification_to_sample": float(design.magnification_to_sample),
        "kr_sample_m_inv": design.kr_sample_m_inv,
        "hardware_target_achieved": bool(
            str(design.mapping_mode) == "fixed_physical_optics"
            and abs(float(design.predicted_bessel_length_m) - float(design.target_bessel_length_m))
            / max(float(design.target_bessel_length_m), EPS) <= 0.03
        ),
        "mapping_claim_scope": (
            "inverse_design_feasibility"
            if str(design.mapping_mode) == "target_matched_inverse_design"
            else "fixed_bench_prediction"
        ),
        "study_kind": study_kind,
        "focused_plane": "surface_in_air" if is_beam_study else "sample_in_medium",
        "beam_medium_n": float(result.get("beam_medium_n", config.material.refractive_index)),
        "ring_radius_um": radial["ring_radius_m"] / um,
        "ring_diameter_um": radial["ring_diameter_m"] / um,
        "core_radius_um": radial["core_radius_m"] / um,
        "core_radius_definition": radial["core_radius_definition"],
        "core_hwhm_radius_um": radial["core_hwhm_radius_m"] / um if np.isfinite(radial["core_hwhm_radius_m"]) else np.nan,
        "core_hwhm_diameter_um": radial["core_hwhm_diameter_m"] / um if np.isfinite(radial["core_hwhm_diameter_m"]) else np.nan,
        "core_first_zero_radius_um": radial["core_first_zero_radius_m"] / um,
        "core_first_zero_diameter_um": radial["core_first_zero_diameter_m"] / um,
        "feature_radius_um": radial["feature_radius_m"] / um,
        "feature_diameter_um": radial["feature_diameter_m"] / um,
        "ring_width_um": radial["ring_width_m"] / um,
        "peak_in_plane": peak_in_plane,
        "peak_onaxis": peak_onaxis,
        "onaxis_to_peak_ratio": peak_onaxis / (peak_in_plane + EPS),
        "pulse_energy_at_sample_uJ": config.energy.pulse_energy_at_sample_J / uJ,
        "pulse_energy_at_surface_air_uJ": config.energy.pulse_energy_at_surface_air_J / uJ,
        "fluence_pulse_energy_uJ": pulse_energy_for_fluence_J / uJ,
        "first_order_selected_fraction": float(result.get("first_order_selected_fraction", np.nan)),
        "first_order_carrier_lpmm": float(order_meta.get("carrier_lpmm", np.nan)),
        "first_order_configured_filter_radius_lpmm": float(order_meta.get("configured_filter_radius_lpmm", np.nan)),
        "first_order_effective_filter_radius_lpmm": float(order_meta.get("effective_filter_radius_lpmm", order_meta.get("filter_radius_lpmm", np.nan))),
        "first_order_recommended_filter_radius_lpmm": float(order_meta.get("recommended_filter_radius_lpmm", np.nan)),
        "first_order_axicon_cone_radius_lpmm": float(order_meta.get("axicon_cone_radius_lpmm", np.nan)),
        "first_order_peak_fx_lpmm": float(order_meta.get("order_peak_fx_lpmm", np.nan)),
        "first_order_peak_fy_lpmm": float(order_meta.get("order_peak_fy_lpmm", np.nan)),
        "first_order_peak_distance_to_carrier_lpmm": float(order_meta.get("order_peak_distance_to_carrier_lpmm", np.nan)),
        "first_order_geometry_valid": bool(order_meta.get("first_order_geometry_valid", True)),
        "propagation_method": str(volume.get("propagation_method", config.propagation.method)),
        "output_dx_at_peak_um": float(output_dx[min(peak_idx, len(output_dx) - 1)] / um),
        "output_dx_min_um": float(np.nanmin(output_dx) / um),
        "output_dx_max_um": float(np.nanmax(output_dx) / um),
        "sas_retained_power_fraction_min": float(np.nanmin(retained)),
        "sas_retained_power_fraction_at_peak": float(retained[min(peak_idx, len(retained) - 1)]),
        "sas_z_limit_min_um": float(np.nanmin(z_limits) / um) if np.any(np.isfinite(z_limits)) else np.nan,
        "sas_z_over_limit_max": float(np.nanmax(z_over)) if np.any(np.isfinite(z_over)) else np.nan,
        "sas_output_magnification_at_peak": float(mags[min(peak_idx, len(mags) - 1)]) if np.any(np.isfinite(mags)) else np.nan,
        "sas_bandlimit_retained_fraction_min": float(np.nanmin(band_ret)) if np.any(np.isfinite(band_ret)) else np.nan,
        **zone,
        **region,
        **flu,
        **mod,
    }
    metrics.update(_propagation_power_metric_fields(propagation_power_validity_report(volume)))
    return metrics


def sampling_report(
    config: TwinConfig,
    design: Optional[BeamDesign] = None,
    result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return SLM, phase, focal, and ASM sampling QA flags."""

    design = design or compute_design_from_config(config)
    dx_p = config.slm.pixel_pitch_m * max(1, int(config.grid.device_downsample))
    nyq_lpmm = 1.0 / (2.0 * dx_p) / 1e3
    phase_lpmm = (abs(design.kr_slm_m_inv) / TWOPI + abs(config.slm.carrier_cpm if config.include_blaze else 0.0)) / 1e3
    if phase_lpmm > nyq_lpmm:
        phase_label = "fail"
    elif phase_lpmm > 0.8 * nyq_lpmm:
        phase_label = "marginal"
    else:
        phase_label = "pass"
    dx_f = config.laser.wavelength_m * config.objective.f_eff_m / (config.grid.N * dx_p)
    focal_nyq_kt = np.pi / dx_f
    k_med = config.laser.k0 * config.material.refractive_index
    axial_values = axial_scan_values(config, design, z_anchor_m=0.0)
    axial_dz = float(np.median(np.diff(axial_values))) if axial_values.size > 1 else np.nan
    kr_margin = float(focal_nyq_kt / max(design.kr_sample_m_inv, EPS))
    focal_samples_per_radial_period = float((TWOPI / max(design.kr_sample_m_inv, EPS)) / max(dx_f, EPS))
    axial_label = "pass" if np.isfinite(axial_dz) and axial_dz <= 0.05 * max(design.target_bessel_length_m, EPS) else "marginal"
    focal_label = "pass" if focal_nyq_kt > 1.5 * design.kr_sample_m_inv else "marginal"
    device_label = "exact" if config.grid.device_downsample == 1 else "downsampled"
    labels = [phase_label, focal_label, axial_label]
    qa = "fail" if "fail" in labels else ("marginal" if "marginal" in labels or device_label == "downsampled" else "pass")
    out = {
        "qa_status": qa,
        "propagation_method": str(config.propagation.method),
        "device_grid_label": device_label,
        "device_downsample": int(config.grid.device_downsample),
        "pupil_N": int(config.grid.N),
        "pupil_dx_um": dx_p / um,
        "pupil_window_mm": config.grid.N * dx_p / mm,
        "phase_frequency_lpmm": float(phase_lpmm),
        "pupil_nyquist_lpmm": float(nyq_lpmm),
        "phase_sampling_label": phase_label,
        "focal_dx_um": float(dx_f / um),
        "focal_kt_nyquist_over_kr": kr_margin,
        "focal_samples_per_radial_period": focal_samples_per_radial_period,
        "focal_sampling_label": focal_label,
        "axial_dz_um": float(axial_dz / um) if np.isfinite(axial_dz) else np.nan,
        "axial_scan_end_um": float(axial_values[-1] / um) if axial_values.size else np.nan,
        "axial_scan_target_factor": float(getattr(config.grid, "axial_target_factor", np.nan)),
        "axial_sampling_label": axial_label,
        "k_medium_m_inv": float(k_med),
        "NA": float(config.objective.NA),
    }
    if config.propagation.method == "sas":
        focus_grid = make_xy_grid(int(config.grid.N), dx_f)
        sas_ref_z = max(float(config.grid.axial_range_m), float(design.target_bessel_length_m))
        sas_rep = sas_validity_report(
            focus_grid,
            config.laser.wavelength_m,
            sas_ref_z,
            n_medium=config.material.refractive_index,
            pad_factor=config.propagation.sas_pad_factor,
        )
        out.update({
            "sas_reference_z_um": float(sas_ref_z / um),
            "sas_z_limit_um": float(sas_rep["z_limit_m"] / um),
            "sas_z_limit_margin_um": float(sas_rep["z_limit_margin_m"] / um),
            "sas_z_over_limit": float(sas_rep["z_over_limit"]),
            "sas_output_dx_um": float(sas_rep["output_dx_m"] / um),
            "sas_output_magnification": float(sas_rep["output_magnification"]),
            "sas_scaled_grid": bool(abs(float(sas_rep["output_magnification"]) - 1.0) > 0.05),
        })
    if result is not None:
        power_report = propagation_power_validity_report(result["volume"])
        out.update(_propagation_power_metric_fields(power_report))
        out["propagation_power_clipping_note"] = (
            "Numerical total-power drift across the propagated stack. Quantitative "
            "metrics require drift<=0.05; intentional upstream filtering is excluded."
        )
        out["output_dx_min_um"] = float(np.nanmin(np.asarray(result["volume"].get("output_dx_m", [dx_f]), float)) / um)
        out["output_dx_max_um"] = float(np.nanmax(np.asarray(result["volume"].get("output_dx_m", [dx_f]), float)) / um)
        retained = np.asarray(result["volume"].get("retained_power_fraction", [1.0]), float)
        out["sas_retained_power_fraction_min"] = float(np.nanmin(retained))
        out["sas_retained_power_fraction_mean"] = float(np.nanmean(retained))
    else:
        out.update(_propagation_power_metric_fields(propagation_power_validity_report({})))
        out["propagation_power_clipping_note"] = (
            "No propagated volume was supplied; total-power drift and SAS clipping were not evaluated."
        )
    return out


def energy_conservation_report(
    wavelength_m: float = 1029.0 * nm,
    n_medium: float = 1.0,
    N: int = 256,
    dx_m: float = 0.5 * um,
    z_values_m: Optional[Sequence[float]] = None,
) -> Dict[str, float | bool]:
    """Lossless BL-ASM power-conservation smoke check."""

    if z_values_m is None:
        z_values_m = np.linspace(0.0, 30.0 * um, 7)
    grid = make_xy_grid(N, dx_m)
    U0 = gaussian_amplitude(grid["R"], 12.0 * um) * np.exp(-1j * 0.1 / um * grid["R"])
    powers = []
    for z in z_values_m:
        U = angular_spectrum_propagate_bl(U0, grid, wavelength_m, float(z), n_medium=n_medium, bandlimit=True)
        powers.append(float(np.sum(np.abs(U) ** 2) * dx_m * dx_m))
    powers = np.asarray(powers)
    drift = float((np.max(powers) - np.min(powers)) / (np.mean(powers) + EPS))
    return {"power_drift_fraction": drift, "pass": bool(drift <= 0.01)}


def run_sas_self_checks(
    output_dir: str | Path = "outputs",
    save: bool = True,
) -> pd.DataFrame:
    """Run lightweight SAS sanity checks and optionally save a CSV report."""

    wavelength = 1029.0 * nm
    n_medium = 1.0
    N = 128
    dx = 0.5 * um
    grid = make_xy_grid(N, dx)
    U0 = gaussian_amplitude(grid["R"], 10.0 * um) * np.exp(1j * 0.04 / um * grid["X"])
    lam = wavelength / n_medium
    z_match = 2.0 * (N * dx) ** 2 / (lam * N)

    rows: List[Tuple[str, bool, float, str]] = []
    rep = sas_validity_report(grid, wavelength, z_match, n_medium=n_medium, pad_factor=2)
    rows.append(("sas_z_limit_valid", bool(rep["valid"]), float(rep["z_over_limit"]), "z_over_limit"))

    U_sas, g_sas, meta = scalable_angular_spectrum_propagate(
        U0,
        grid,
        wavelength,
        z_match,
        n_medium=n_medium,
        pad_factor=2,
        bandlimit=True,
        skip_final_phase=True,
    )
    rows.append(("sas_output_grid_metadata", bool(abs(g_sas["dx"] - rep["output_dx_m"]) <= 1e-15), float(g_sas["dx"] / um), "output_dx_um"))
    rows.append(("sas_full_power_conservation", bool(abs(meta["full_power_ratio"] - 1.0) <= 0.03), float(meta["full_power_ratio"]), "full_power_ratio"))
    rows.append(("sas_retained_window_fraction", bool(meta["retained_power_fraction"] >= 0.90), float(meta["retained_power_fraction"]), "retained_fraction"))

    U_bl = angular_spectrum_propagate_bl(U0, grid, wavelength, z_match, n_medium=n_medium, bandlimit=True)
    I_bl = _resample_intensity_to_common_grid(np.abs(U_bl) ** 2, grid, g_sas["x"])
    I_sas = np.abs(U_sas) ** 2
    I_bl_n = I_bl / (float(np.max(I_bl)) + EPS)
    I_sas_n = I_sas / (float(np.max(I_sas)) + EPS)
    rel_l2 = float(np.linalg.norm(I_bl_n - I_sas_n) / (np.linalg.norm(I_bl_n) + EPS))
    rows.append(("sas_bl_asm_intensity_overlap", bool(rel_l2 <= 0.35), rel_l2, "relative_l2"))

    too_far = 1.05 * float(rep["z_limit_m"])
    raised = False
    try:
        scalable_angular_spectrum_propagate(U0, grid, wavelength, too_far, n_medium=n_medium, allow_invalid=False)
    except ValueError:
        raised = True
    rows.append(("sas_invalid_distance_guard", raised, float(too_far / (float(rep["z_limit_m"]) + EPS)), "z_over_limit"))

    df = pd.DataFrame(rows, columns=["check", "pass", "value", "metric"])
    if save:
        paths = ensure_output_tree(output_dir)
        df.to_csv(paths["csv"] / "sas_self_checks.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Case runner and studies
# ---------------------------------------------------------------------------


def _case_validity_report(config: TwinConfig, design: BeamDesign, result: Dict[str, Any]) -> Dict[str, Any]:
    report = vbb_regime.sampling_validity(config, design, result)
    return vbb_regime.enforce_validity(report, getattr(config, "validity_on_violation", "flag"))


def _case_propagation_power_validity_report(config: TwinConfig, result: Dict[str, Any]) -> Dict[str, Any]:
    return enforce_propagation_power_validity(
        result.get("volume", {}),
        getattr(config, "validity_on_violation", "flag"),
    )


def _validity_metric_fields(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "validity_valid": bool(report.get("valid", False)),
        "validity_violations": ",".join(report.get("violations", [])),
        "validity_samples_per_feature": float(report.get("samples_per_feature", np.nan)),
        "validity_samples_per_radial_period": float(report.get("samples_per_radial_period", np.nan)),
        "validity_samples_per_axial_zone": float(report.get("samples_per_axial_zone", np.nan)),
        "validity_output_dx_um": float(report.get("output_dx_um", np.nan)),
        "validity_action": str(report.get("validity_action", "pass")),
        "regime": str(report.get("regime", "general")),
    }


def _hardware_reachable(config: TwinConfig, design: BeamDesign) -> bool:
    E_sample = config.energy.pulse_energy_at_sample_J / uJ
    return bool(
        int(abs(design.ell)) in {0, 1, 2, 3, 5, 8, 10}
        and 1.0 <= design.target_core_diameter_m / um <= 6.0
        and 50.0 <= design.target_bessel_length_m / um <= 500.0
        and 0.3 <= config.objective.NA <= 0.65
        and 50.0 <= config.material.write_depth_m / um <= 1000.0
        and 1.0 <= E_sample <= 50.0
        and config.laser.input_pulse_energy_J <= config.laser.max_pulse_energy_J
        and config.laser.average_power_W <= config.laser.max_average_power_W
    )


def _run_full_source_to_sample_case(
    config: Optional[TwinConfig] = None,
    preset: str = "fast",
    path: PathKind = "realistic",
    case_id: str = "case",
    z_values_m: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Run one ideal or realistic case and return result plus tidy metrics."""

    config = default_config(preset) if config is None else config
    design = compute_design_from_config(config)
    generation_method = str(getattr(config, "generation_method", "holographic")).lower().strip()
    axicon_result = None
    if generation_method == "physical":
        axicon_result = vbb_axicon.PhysicalAxicon().generate({"design": design}, config)
        grid = axicon_result.grid
        U0 = axicon_result.Ex
        if z_values_m is None:
            z_values = axial_scan_values(config, design, z_anchor_m=0.0)
        else:
            z_values = np.asarray(z_values_m, dtype=float)
            if z_values.ndim != 1 or z_values.size == 0:
                raise ValueError("z_values_m must be a non-empty 1D sequence.")
        volume = propagate_volume(
            U0,
            grid,
            config.laser.wavelength_m,
            z_values,
            n_medium=config.material.refractive_index,
            crop_pixels=min(config.grid.crop_pixels, config.grid.ideal_N),
            bandlimit=True,
            method=config.propagation.method,
            propagation_config=config.propagation,
        )
        result = {
            "path": path,
            "design": design,
            "focal_grid": grid,
            "volume": volume,
            "first_order_selected_fraction": float(axicon_result.metadata.get("efficiency", 1.0)),
            "z_zone_centre_m": 0.0,
            "interface_zernike_fit": {},
            "generation_method": generation_method,
            "axicon_result": axicon_result,
            "axicon_metadata": dict(axicon_result.metadata),
        }
    elif path == "ideal":
        axicon_result = vbb_axicon.HolographicAxicon().generate({"design": design}, config)
        grid = axicon_result.grid
        U0 = axicon_result.Ex
        if z_values_m is None:
            z_values = axial_scan_values(config, design, z_anchor_m=0.0)
        else:
            z_values = np.asarray(z_values_m, dtype=float)
            if z_values.ndim != 1 or z_values.size == 0:
                raise ValueError("z_values_m must be a non-empty 1D sequence.")
        volume = propagate_volume(
            U0,
            grid,
            config.laser.wavelength_m,
            z_values,
            n_medium=config.material.refractive_index,
            crop_pixels=min(config.grid.crop_pixels, config.grid.ideal_N),
            bandlimit=True,
            method=config.propagation.method,
            propagation_config=config.propagation,
        )
        result = {
            "path": "ideal",
            "design": design,
            "focal_grid": grid,
            "volume": volume,
            "first_order_selected_fraction": 1.0,
            "z_zone_centre_m": 0.0,
            "interface_zernike_fit": {},
            "generation_method": generation_method,
            "axicon_result": axicon_result,
            "axicon_metadata": dict(axicon_result.metadata),
        }
    else:
        if generation_method != "holographic":
            raise ValueError(f"Unsupported generation method: {generation_method!r}")
        result = realistic_slm_to_sample(config, design, z_values_m=z_values_m)
        result["generation_method"] = generation_method
    metrics = extract_vortex_safe_metrics(result, config)
    qa = sampling_report(config, design, result)
    validity = _case_validity_report(config, design, result)
    power_validity = _case_propagation_power_validity_report(config, result)
    metrics.update({
        "study": "single_case",
        "case_id": case_id,
        "path": path,
        "material": config.material.name,
        "write_depth_um": config.material.write_depth_m / um,
        "NA": config.objective.NA,
        "phase_bits": config.slm.phase_bits,
        "fill_factor": config.slm.fill_factor,
        "blaze_period_px": config.slm.blaze_period_px,
        "beam_radius_on_slm_mm": config.laser.beam_radius_on_slm_m / mm,
        "hardware_reachable": _hardware_reachable(config, design),
        **qa,
        **_propagation_power_metric_fields(power_validity),
        **_validity_metric_fields(validity),
    })
    result["metrics"] = metrics
    result["sampling_report"] = qa
    result["validity_report"] = validity
    result["propagation_power_validity_report"] = power_validity
    return result


def run_case(
    config: Optional[TwinConfig] = None,
    preset: str = "fast",
    path: PathKind = "realistic",
    case_id: str = "case",
    z_values_m: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Run one case through the selected study boundary.

    The default Stage-B headline path is the air-only beam-to-surface study.
    The pre-Stage-B in-medium path remains available as
    ``study_kind="full_source_to_sample"`` for audits and sample-side work.
    """

    config = default_config(preset) if config is None else config
    study_kind = str(getattr(config, "study_kind", "beam_to_surface")).lower().strip()
    if study_kind == "full_source_to_sample":
        return _run_full_source_to_sample_case(config, preset=preset, path=path, case_id=case_id, z_values_m=z_values_m)
    if study_kind == "through_sample":
        raise NotImplementedError("run_case(study_kind='through_sample') requires an explicit SurfaceField hand-off.")
    if study_kind != "beam_to_surface":
        raise ValueError(f"Unsupported study kind: {study_kind!r}")

    result = vbb_studies.build_beam_to_surface_result(config, path=path, z_values_m=z_values_m)
    metrics_config = result.get("metrics_config", config)
    design = result["design"]
    metrics = extract_vortex_safe_metrics(result, metrics_config)
    qa = sampling_report(metrics_config, design, result)
    validity = _case_validity_report(metrics_config, design, result)
    power_validity = _case_propagation_power_validity_report(metrics_config, result)
    metrics.update({
        "study": "single_case",
        "case_id": case_id,
        "path": path,
        "material": "air",
        "write_depth_um": np.nan,
        "surface_z_um": float(result["surface_field"].z_surface_m / um),
        "surface_placement": str(getattr(metrics_config, "surface_placement", "zone_center")),
        "NA": metrics_config.objective.NA,
        "phase_bits": metrics_config.slm.phase_bits,
        "fill_factor": metrics_config.slm.fill_factor,
        "blaze_period_px": metrics_config.slm.blaze_period_px,
        "beam_radius_on_slm_mm": metrics_config.laser.beam_radius_on_slm_m / mm,
        "hardware_reachable": _hardware_reachable(config, design),
        **qa,
        **_propagation_power_metric_fields(power_validity),
        **_validity_metric_fields(validity),
    })
    result["metrics"] = metrics
    result["sampling_report"] = qa
    result["validity_report"] = validity
    result["propagation_power_validity_report"] = power_validity
    return result


def _row_from_result(result: Dict[str, Any], study: str, case_id: str, knob: str, value: Any) -> Dict[str, Any]:
    row = dict(result["metrics"])
    row.update({"study": study, "case_id": case_id, "knob": knob, "knob_value": value})
    return row


def ensure_output_tree(base: str | Path = "outputs") -> Dict[str, Path]:
    base = Path(base)
    paths = {
        "base": base,
        "csv": base / "csv",
        "figures": base / "figures",
        "holograms": base / "holograms",
        "json": base / "json",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _save_df(df: pd.DataFrame, output_dir: str | Path, filename: str) -> pd.DataFrame:
    paths = ensure_output_tree(output_dir)
    df.to_csv(paths["csv"] / filename, index=False)
    return df


def _normalised_sensitivities(df: pd.DataFrame, metric: str = "bessel_zone_um") -> pd.DataFrame:
    rows = []
    for knob, g in df.groupby("knob", dropna=False):
        vals = pd.to_numeric(g["knob_value"], errors="coerce")
        mets = pd.to_numeric(g[metric], errors="coerce")
        ok = vals.notna() & mets.notna()
        if ok.sum() >= 2 and np.nanmean(np.abs(mets[ok])) > 0 and np.nanmean(np.abs(vals[ok])) > 0:
            slope = np.polyfit(vals[ok], mets[ok], 1)[0]
            sens = slope * float(np.nanmean(vals[ok])) / (float(np.nanmean(mets[ok])) + EPS)
            rows.append({"knob": knob, "metric": metric, "normalised_sensitivity": float(sens), "abs_rank_value": float(abs(sens))})
    return pd.DataFrame(rows).sort_values("abs_rank_value", ascending=False) if rows else pd.DataFrame()


def run_oat_sensitivity(
    config: Optional[TwinConfig] = None,
    preset: str = "fast",
    output_dir: str | Path = "outputs",
    save: bool = True,
) -> pd.DataFrame:
    """One-at-a-time sweep around the baseline over lab-controllable knobs."""

    base = default_config(preset) if config is None else config
    rows: List[Dict[str, Any]] = []

    def add(cfg: TwinConfig, knob: str, value: Any, path: PathKind = "ideal") -> None:
        cid = f"{knob}_{str(value).replace('.', 'p')}"
        rows.append(_row_from_result(run_case(cfg, preset=preset, path=path, case_id=cid), "oat_sensitivity", cid, knob, value))

    for ell in [0, 1, 3, 5, 8]:
        add(replace(base, target=replace(base.target, ell=ell)), "ell", ell, path="ideal")
    for D in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]:
        add(replace(base, target=replace(base.target, target_core_diameter_m=D * um)), "target_core_diameter_um", D, path="ideal")
    for L in [50.0, 150.0, 500.0, 800.0]:
        add(replace(base, target=replace(base.target, target_bessel_length_m=L * um)), "target_bessel_length_um", L, path="ideal")
    for NA in [0.30, 0.45, 0.65]:
        add(replace(base, objective=replace(base.objective, NA=NA)), "objective_NA", NA, path="ideal")
    for depth in [50.0, 300.0, 1000.0]:
        add(replace(base, material=replace(base.material, write_depth_m=depth * um)), "write_depth_um", depth, path="realistic")
    for Ein in [4.0, 20.0, 80.0]:
        add(replace(base, energy=replace(base.energy, pulse_energy_in_J=Ein * uJ)), "pulse_energy_in_uJ", Ein, path="ideal")
    for rep in [100.0, 200.0, 1000.0]:
        add(replace(base, laser=replace(base.laser, rep_rate_Hz=rep * kHz)), "rep_rate_kHz", rep, path="ideal")
    for speed in [0.1, 1.0, 10.0]:
        add(replace(base, material=replace(base.material, scan_speed_m_s=speed * mm)), "scan_speed_mm_s", speed, path="ideal")
    for bits in [8, 6, 4]:
        add(replace(base, slm=replace(base.slm, phase_bits=bits)), "slm_phase_bits", bits, path="realistic")
    for ff in [1.0, 0.93, 0.85]:
        add(replace(base, slm=replace(base.slm, fill_factor=ff)), "fill_factor", ff, path="realistic")
    for blaze in [12, 20, 32]:
        add(replace(base, slm=replace(base.slm, blaze_period_px=blaze)), "blaze_period_px", blaze, path="realistic")
    for w in [1.0, 2.0, 3.0]:
        add(replace(base, laser=replace(base.laser, beam_radius_on_slm_m=w * mm)), "beam_radius_on_slm_mm", w, path="ideal")

    df = pd.DataFrame(rows)
    sens = _normalised_sensitivities(df, metric="canonical_zone_um")
    if not sens.empty:
        rank_map = sens.set_index("knob")["normalised_sensitivity"].to_dict()
        df["normalised_canonical_zone_sensitivity"] = df["knob"].map(rank_map)
        df["normalised_zone_sensitivity"] = df["normalised_canonical_zone_sensitivity"]
    if save:
        _save_df(df, output_dir, "oat_sensitivity.csv")
        if not sens.empty:
            _save_df(sens, output_dir, "oat_sensitivity_ranked.csv")
    return df


def run_tradeoff_map(
    config: Optional[TwinConfig] = None,
    preset: str = "fast",
    output_dir: str | Path = "outputs",
    save: bool = True,
) -> pd.DataFrame:
    """2D core-diameter x Bessel-length map with feasibility metrics."""

    base = default_config(preset) if config is None else config
    rows = []
    for D in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]:
        for L in [50.0, 100.0, 150.0, 300.0, 500.0, 800.0]:
            cfg = replace(base, target=replace(base.target, target_core_diameter_m=D * um, target_bessel_length_m=L * um))
            result = run_case(cfg, preset=preset, path="ideal", case_id=f"D{D}_L{L}")
            row = _row_from_result(result, "tradeoff_core_length", f"D{D}_L{L}", "core_um_x_length_um", f"{D},{L}")
            row["clears_incubated_threshold"] = bool(row["core_or_ring_peak_fluence_J_cm2"] >= row["incubated_threshold_J_cm2"])
            rows.append(row)
    df = pd.DataFrame(rows)
    if save:
        _save_df(df, output_dir, "tradeoff_core_length.csv")
    return df


def run_ell_family_comparison(
    config: Optional[TwinConfig] = None,
    preset: str = "fast",
    output_dir: str | Path = "outputs",
    save: bool = True,
) -> pd.DataFrame:
    """Compare topological-charge family at fixed energy and fixed peak fluence."""

    base = default_config(preset) if config is None else config
    rows = []
    for ell in [0, 1, 2, 3, 5, 8, 10]:
        cfg = replace(base, target=replace(base.target, ell=ell))
        result = run_case(cfg, preset=preset, path="ideal", case_id=f"ell{ell}_fixed_energy")
        row = _row_from_result(result, "ell_family", f"ell{ell}_fixed_energy", "ell", ell)
        per_uJ = row["core_or_ring_peak_fluence_J_cm2"] / max(row["pulse_energy_at_sample_uJ"], EPS)
        row["energy_at_sample_for_threshold_uJ"] = row["incubated_threshold_J_cm2"] / max(per_uJ, EPS)
        row["fixed_peak_fluence_framing"] = "energy_needed_for_threshold"
        rows.append(row)
    df = pd.DataFrame(rows)
    if save:
        _save_df(df, output_dir, "ell_family_comparison.csv")
    return df


def run_device_realism_ablation(
    config: Optional[TwinConfig] = None,
    preset: str = "fast",
    output_dir: str | Path = "outputs",
    save: bool = True,
) -> pd.DataFrame:
    """Quantify idealised sample beam versus staged device-realism effects."""

    base = default_config(preset) if config is None else config
    variants: List[Tuple[str, TwinConfig, PathKind]] = [
        ("analytic_ideal", base, "ideal"),
        ("continuous_phase_no_fill", replace(base, include_quantization=False, include_fill_factor=False, include_active_aperture=False), "realistic"),
        ("quantized_only", replace(base, include_quantization=True, include_fill_factor=False, include_active_aperture=False), "realistic"),
        ("quantized_fill", replace(base, include_quantization=True, include_fill_factor=True, include_active_aperture=False), "realistic"),
        ("full_device", replace(base, include_quantization=True, include_fill_factor=True, include_active_aperture=True), "realistic"),
    ]
    rows = []
    reference: Optional[Dict[str, Any]] = None
    for name, cfg, path in variants:
        result = run_case(cfg, preset=preset, path=path, case_id=name)
        row = _row_from_result(result, "device_realism_ablation", name, "realism_variant", name)
        if reference is None:
            reference = row
        for metric in ["ring_radius_um", "feature_radius_um", "canonical_zone_um", "strict_bessel_region_um", "peak_fluence_J_cm2", "side_to_core_peak_ratio"]:
            row[f"{metric}_shift_vs_ideal_pct"] = 100.0 * (row[metric] - reference[metric]) / (abs(reference[metric]) + EPS)
        rows.append(row)
    df = pd.DataFrame(rows)
    if save:
        _save_df(df, output_dir, "device_realism_ablation.csv")
    return df


def run_interface_depth_sweep(
    config: Optional[TwinConfig] = None,
    preset: str = "fast",
    output_dir: str | Path = "outputs",
    save: bool = True,
) -> pd.DataFrame:
    """Interface aberration and SLM correction versus write depth."""

    base = default_config(preset) if config is None else config
    base = replace(base, include_first_order_isolation=False, include_blaze=False, study_kind="full_source_to_sample")
    rows = []
    ideal_cfg = replace(base, apply_interface=False, correct_interface=False)
    ideal = run_case(ideal_cfg, preset=preset, path="realistic", case_id="no_interface")
    ideal_peak = ideal["metrics"]["peak_in_plane"]
    fixed_threshold = 0.5 * ideal_peak
    for depth in [50.0, 150.0, 300.0, 600.0, 1000.0]:
        for corrected in [False, True]:
            cfg = replace(
                base,
                material=replace(base.material, write_depth_m=depth * um),
                apply_interface=True,
                correct_interface=corrected,
            )
            label = f"z{depth:.0f}_{'corrected' if corrected else 'uncorrected'}"
            result = run_case(cfg, preset=preset, path="realistic", case_id=label)
            row = _row_from_result(result, "interface_depth_sweep", label, "write_depth_um", depth)
            row["interface_corrected"] = bool(corrected)
            row["strehl_like_peak_ratio"] = float(row["peak_in_plane"] / (ideal_peak + EPS))
            row["absolute_half_ideal_zone_um"] = absolute_zone_length_um(result["volume"]["z"], result["volume"]["peak"], fixed_threshold)
            rows.append(row)
    df = pd.DataFrame(rows)
    if save:
        _save_df(df, output_dir, "interface_depth_sweep.csv")
    return df


def run_sampling_feasibility_envelope(
    config: Optional[TwinConfig] = None,
    preset: str = "fast",
    output_dir: str | Path = "outputs",
    save: bool = True,
) -> pd.DataFrame:
    """Flag numerical/hardware feasibility over the requested sweep envelope."""

    base = default_config(preset) if config is None else config
    rows = []
    for ell in [0, 1, 2, 3, 5, 8, 10]:
        for D in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]:
            for L in [50.0, 100.0, 150.0, 300.0, 500.0, 800.0]:
                cfg = replace(base, target=replace(base.target, ell=ell, target_core_diameter_m=D * um, target_bessel_length_m=L * um))
                eval_cfg = vbb_studies.beam_air_config(cfg) if getattr(cfg, "study_kind", "beam_to_surface") == "beam_to_surface" else cfg
                design = compute_design_from_config(eval_cfg)
                rep = sampling_report(eval_cfg, design)
                validity = vbb_regime.enforce_validity(vbb_regime.sampling_validity(eval_cfg, design), eval_cfg.validity_on_violation)
                rows.append({
                    "study": "sampling_feasibility_envelope",
                    "case_id": f"ell{ell}_D{D}_L{L}",
                    "ell": ell,
                    "target_core_diameter_um": D,
                    "target_scale_definition": str(design.target_scale_definition),
                    "target_equivalent_l0_core_diameter_um": design.target_equivalent_l0_core_diameter_m / um,
                    "equivalent_l0_first_zero_radius_um": design.equivalent_l0_first_zero_radius_m / um,
                    "equivalent_l0_first_zero_diameter_um": design.equivalent_l0_first_zero_diameter_m / um,
                    "vortex_main_ring_radius_um": design.vortex_main_ring_radius_m / um,
                    "vortex_main_ring_diameter_um": design.vortex_main_ring_diameter_m / um,
                    "target_bessel_length_um": L,
                    "predicted_bessel_length_um": design.predicted_bessel_length_m / um,
                    "gamma_slm_deg": design.gamma_slm_deg,
                    "magnification_to_sample": design.magnification_to_sample,
                    "mapping_mode": str(design.mapping_mode),
                    "objective_map_source": str(design.objective_map_source),
                    "objective_map_demag": float(design.objective_map_demag),
                    "mapping_claim_scope": (
                        "inverse_design_feasibility"
                        if str(design.mapping_mode) == "target_matched_inverse_design"
                        else "fixed_bench_prediction"
                    ),
                    "hardware_target_achieved": bool(
                        str(design.mapping_mode) == "fixed_physical_optics"
                        and abs(float(design.predicted_bessel_length_m) - float(design.target_bessel_length_m))
                        / max(float(design.target_bessel_length_m), EPS) <= 0.03
                    ),
                    "core_radius_definition": "not_evaluated_sampling_envelope",
                    "core_hwhm_radius_um": np.nan,
                    "core_hwhm_diameter_um": np.nan,
                    "core_first_zero_radius_um": design.equivalent_l0_first_zero_radius_m / um,
                    "core_first_zero_diameter_um": design.equivalent_l0_first_zero_diameter_m / um,
                    "canonical_zone_um": np.nan,
                    "canonical_zone_start_um": np.nan,
                    "canonical_zone_end_um": np.nan,
                    "strict_bessel_region_um": np.nan,
                    "strict_bessel_region_start_um": np.nan,
                    "strict_bessel_region_end_um": np.nan,
                    "bessel_region_definition": "not_evaluated_sampling_envelope",
                    "hardware_reachable": _hardware_reachable(cfg, design),
                    **rep,
                    **_validity_metric_fields(validity),
                })
    df = pd.DataFrame(rows)
    if save:
        _save_df(df, output_dir, "sampling_feasibility_envelope.csv")
    return df


def run_self_checks(preset: str = "fast", output_dir: str | Path = "outputs") -> pd.DataFrame:
    """Run acceptance self-checks and save a compact CSV report."""

    cfg = default_config(preset)
    design = compute_design_from_config(cfg)
    refs = analytic_references(cfg, design)
    roundtrip = inverse_design_round_trip(cfg)
    energy = energy_conservation_report(wavelength_m=cfg.laser.wavelength_m, n_medium=cfg.material.refractive_index)
    case = run_case(cfg, preset=preset, path="realistic", case_id="baseline_realistic")
    samp = sampling_report(cfg, design, case)

    interface_cfg = replace(cfg, include_first_order_isolation=False, include_blaze=False)
    ideal_cfg = replace(interface_cfg, apply_interface=False, correct_interface=False)
    aberr_cfg = replace(interface_cfg, apply_interface=True, correct_interface=False, material=replace(cfg.material, write_depth_m=300.0 * um))
    corr_cfg = replace(interface_cfg, apply_interface=True, correct_interface=True, material=replace(cfg.material, write_depth_m=300.0 * um))
    ideal = run_case(ideal_cfg, preset=preset, path="realistic", case_id="interface_ideal")
    aberr = run_case(aberr_cfg, preset=preset, path="realistic", case_id="interface_aberrated")
    corr = run_case(corr_cfg, preset=preset, path="realistic", case_id="interface_corrected")
    ideal_zone = ideal["metrics"]["canonical_zone_um"]
    aberr_zone = aberr["metrics"]["canonical_zone_um"]
    corr_zone = corr["metrics"]["canonical_zone_um"]
    fixed_threshold = 0.5 * ideal["metrics"]["peak_in_plane"]
    ideal_abs_zone = absolute_zone_length_um(ideal["volume"]["z"], ideal["volume"]["peak"], fixed_threshold)
    aberr_abs_zone = absolute_zone_length_um(aberr["volume"]["z"], aberr["volume"]["peak"], fixed_threshold)
    corr_abs_zone = absolute_zone_length_um(corr["volume"]["z"], corr["volume"]["peak"], fixed_threshold)
    interface_pass = bool(corr_abs_zone >= 0.85 * ideal_abs_zone and aberr_abs_zone <= 0.85 * ideal_abs_zone)

    checks = [
        ("inverse_design_round_trip", roundtrip["pass"], max(roundtrip["core_relative_error"], roundtrip["length_relative_error"])),
        ("energy_conservation_lossless_ASM", energy["pass"], energy["power_drift_fraction"]),
        ("sampling_QA_not_fail", samp["qa_status"] != "fail", samp.get("propagation_power_drift_fraction", np.nan)),
        ("interface_correction_recovers_zone", interface_pass, corr_abs_zone / max(ideal_abs_zone, EPS)),
        ("analytic_core_diameter", abs(refs["core_first_zero_diameter_um"] - cfg.target.target_core_diameter_m / um) <= 0.1, refs["core_first_zero_diameter_um"]),
        ("analytic_zmax", abs(refs["zmax_baliyan_um"] - cfg.target.target_bessel_length_m / um) <= 2.0, refs["zmax_baliyan_um"]),
    ]
    df = pd.DataFrame(checks, columns=["check", "pass", "value"])
    details = {
        "analytic_references": refs,
        "roundtrip": roundtrip,
        "energy_conservation": energy,
        "sampling_report": samp,
        "interface_metrics": {
            "ideal_canonical_zone_um": ideal_zone,
            "aberrated_canonical_zone_um": aberr_zone,
            "corrected_canonical_zone_um": corr_zone,
            "fixed_threshold_peak": fixed_threshold,
            "ideal_absolute_zone_um": ideal_abs_zone,
            "aberrated_absolute_zone_um": aberr_abs_zone,
            "corrected_absolute_zone_um": corr_abs_zone,
            "ideal_peak": ideal["metrics"]["peak_in_plane"],
            "aberrated_peak": aberr["metrics"]["peak_in_plane"],
            "corrected_peak": corr["metrics"]["peak_in_plane"],
        },
    }
    paths = ensure_output_tree(output_dir)
    df.to_csv(paths["csv"] / "self_checks.csv", index=False)
    (paths["json"] / "self_checks_details.json").write_text(json.dumps(_json_safe(details), indent=2), encoding="utf-8")
    return df


def run_all_studies(
    config: Optional[TwinConfig] = None,
    preset: str = "fast",
    output_dir: str | Path = "outputs",
) -> Dict[str, pd.DataFrame]:
    """Run the complete fast analysis set used by the bible notebook."""

    cfg = default_config(preset) if config is None else config
    return {
        "oat": run_oat_sensitivity(cfg, preset=preset, output_dir=output_dir, save=True),
        "tradeoff": run_tradeoff_map(cfg, preset=preset, output_dir=output_dir, save=True),
        "ell_family": run_ell_family_comparison(cfg, preset=preset, output_dir=output_dir, save=True),
        "device_realism": run_device_realism_ablation(cfg, preset=preset, output_dir=output_dir, save=True),
        "interface_depth": run_interface_depth_sweep(cfg, preset=preset, output_dir=output_dir, save=True),
        "sampling": run_sampling_feasibility_envelope(cfg, preset=preset, output_dir=output_dir, save=True),
    }


def generate_standard_figures(output_dir: str | Path = "outputs") -> List[Path]:
    """Generate standard comparison figures from the study CSV outputs."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    paths = ensure_output_tree(output_dir)
    csv = paths["csv"]
    figdir = paths["figures"]
    made: List[Path] = []

    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(csv / name, keep_default_na=False)

    sens = read("oat_sensitivity_ranked.csv")
    if not sens.empty:
        sens["normalised_sensitivity"] = pd.to_numeric(sens["normalised_sensitivity"])
        sens["abs_rank_value"] = pd.to_numeric(sens["abs_rank_value"])
        s = sens.sort_values("abs_rank_value", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
        ax.barh(s["knob"].astype(str), s["normalised_sensitivity"], color="#3a7ca5")
        ax.axvline(0, color="0.2", lw=0.8)
        ax.set_xlabel("normalised d(zone)/d(knob)")
        ax.set_title("One-at-a-time sensitivity ranking")
        p = figdir / "oat_sensitivity_ranked.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        made.append(p)

    trade = read("tradeoff_core_length.csv")
    for col in ["target_bessel_length_um", "target_core_diameter_um", "gamma_slm_deg"]:
        trade[col] = pd.to_numeric(trade[col])
    piv = trade.pivot(index="target_bessel_length_um", columns="target_core_diameter_um", values="gamma_slm_deg")
    fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
    im = ax.imshow(piv.values, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [f"{i:g}" for i in piv.index])
    ax.set_xlabel("target core diameter [um, sample plane]")
    ax.set_ylabel("target Bessel length [um, sample plane]")
    ax.set_title("Required SLM axicon angle")
    fig.colorbar(im, ax=ax, label="gamma [deg]")
    p = figdir / "tradeoff_core_length_gamma.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    made.append(p)

    ell = read("ell_family_comparison.csv")
    for col in ["ell", "ring_radius_um", "energy_at_sample_for_threshold_uJ"]:
        ell[col] = pd.to_numeric(ell[col])
    fig, ax1 = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax1.plot(ell["ell"], ell["ring_radius_um"], "o-", color="#2f6690", label="ring radius")
    ax1.set_xlabel("topological charge ell")
    ax1.set_ylabel("ring radius [um, sample plane]", color="#2f6690")
    ax2 = ax1.twinx()
    ax2.plot(ell["ell"], ell["energy_at_sample_for_threshold_uJ"], "s--", color="#b23a48", label="energy for threshold")
    ax2.set_ylabel("sample energy for threshold [uJ]", color="#b23a48")
    ax1.set_title("ell-family comparison")
    p = figdir / "ell_family_comparison.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    made.append(p)

    dev = read("device_realism_ablation.csv")
    for col in ["canonical_zone_um_shift_vs_ideal_pct", "feature_radius_um_shift_vs_ideal_pct"]:
        dev[col] = pd.to_numeric(dev[col])
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    x = np.arange(len(dev))
    ax.bar(x - 0.18, dev["canonical_zone_um_shift_vs_ideal_pct"], width=0.36, label="canonical FWHM zone shift")
    ax.bar(x + 0.18, dev["feature_radius_um_shift_vs_ideal_pct"], width=0.36, label="feature-radius shift")
    ax.axhline(0, color="0.2", lw=0.8)
    ax.set_xticks(x, dev["knob_value"].astype(str), rotation=25, ha="right")
    ax.set_ylabel("shift vs analytic ideal [%]")
    ax.set_title("Device-realism impact")
    ax.legend(frameon=False)
    p = figdir / "device_realism_impact.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    made.append(p)

    inter = read("interface_depth_sweep.csv")
    for col in ["write_depth_um", "absolute_half_ideal_zone_um", "strehl_like_peak_ratio"]:
        inter[col] = pd.to_numeric(inter[col])
    inter["interface_corrected_bool"] = inter["interface_corrected"].astype(str).str.lower().isin(["true", "1"])
    fig, ax1 = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    for corrected, g in inter.groupby("interface_corrected_bool"):
        label = "corrected" if corrected else "uncorrected"
        ax1.plot(g["write_depth_um"], g["absolute_half_ideal_zone_um"], "o-", label=f"zone {label}")
    ax1.set_xlabel("write depth [um, sample plane]")
    ax1.set_ylabel("fixed-threshold zone [um, sample plane]")
    ax2 = ax1.twinx()
    for corrected, g in inter.groupby("interface_corrected_bool"):
        label = "corrected" if corrected else "uncorrected"
        ax2.plot(g["write_depth_um"], g["strehl_like_peak_ratio"], "s--", alpha=0.65, label=f"peak {label}")
    ax2.set_ylabel("Strehl-like peak ratio")
    ax1.set_title("Interface aberration and SLM pre-correction")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], frameon=False, fontsize=8)
    p = figdir / "interface_depth_sweep.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    made.append(p)

    samp = read("sampling_feasibility_envelope.csv")
    for col in ["target_core_diameter_um", "target_bessel_length_um"]:
        samp[col] = pd.to_numeric(samp[col])
    counts = samp.groupby(["target_core_diameter_um", "target_bessel_length_um", "qa_status"]).size().unstack(fill_value=0)
    fail = counts["fail"] if "fail" in counts else pd.Series(0, index=counts.index)
    mat = fail.unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    im = ax.imshow(mat.values, origin="lower", aspect="auto", cmap="magma")
    ax.set_xticks(range(len(mat.columns)), [f"{c:g}" for c in mat.columns])
    ax.set_yticks(range(len(mat.index)), [f"{i:g}" for i in mat.index])
    ax.set_xlabel("target Bessel length [um, sample plane]")
    ax.set_ylabel("target core diameter [um, sample plane]")
    ax.set_title("Fast-preset sampling failures across ell family")
    fig.colorbar(im, ax=ax, label="number of ell cases flagged fail")
    p = figdir / "sampling_feasibility_envelope.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    made.append(p)
    return made


# ---------------------------------------------------------------------------
# Deferred material/mode hooks
# ---------------------------------------------------------------------------


def dose_to_delta_n(*args: Any, **kwargs: Any) -> None:
    """Deferred calibrated material-response hook."""

    raise NotImplementedError("dose_to_delta_n is deferred: requires calibrated Cr:ZnSe nonlinear/material model.")


def depressed_cladding_layout(*args: Any, **kwargs: Any) -> None:
    """Deferred geometry hook for negative-Delta-n depressed cladding tracks."""

    raise NotImplementedError("depressed_cladding_layout is deferred: geometry generation only in the next stage.")


def solve_guided_modes(*args: Any, **kwargs: Any) -> None:
    """Deferred guided-mode solver hook."""

    raise NotImplementedError("solve_guided_modes is deferred: mode solving is outside the scalar planning twin.")


# ---------------------------------------------------------------------------
# Small utilities for notebooks/docs
# ---------------------------------------------------------------------------


def config_summary(config: TwinConfig, design: Optional[BeamDesign] = None) -> pd.DataFrame:
    design = design or compute_design_from_config(config)
    obj_map = objective_map_from_config(config, design)
    rows = [
        ("laser", "wavelength_nm", config.laser.wavelength_m / nm),
        ("laser", "pulse_duration_fs", config.laser.pulse_duration_s / fs),
        ("laser", "input_pulse_energy_uJ", config.energy.pulse_energy_in_J / uJ),
        ("laser", "rep_rate_kHz", config.laser.rep_rate_Hz / kHz),
        ("study", "study_kind", config.study_kind),
        ("study", "regime", config.regime),
        ("study", "mapping_mode", config.mapping_mode),
        ("study", "surface_placement", config.surface_placement),
        ("study", "surface_z_um", None if config.surface_z_m is None else config.surface_z_m / um),
        ("study", "air_scan_half_span_factor", config.air_scan_half_span_factor),
        ("slm", "resolution", f"{config.slm.resolution_x}x{config.slm.resolution_y}"),
        ("slm", "pixel_pitch_um", config.slm.pixel_pitch_m / um),
        ("slm", "phase_bits", config.slm.phase_bits),
        ("slm", "fill_factor", config.slm.fill_factor),
        ("slm", "blaze_period_px", config.slm.blaze_period_px),
        ("objective", "NA", config.objective.NA),
        ("objective", "f_eff_mm", config.objective.f_eff_m / mm),
        ("material", "name", config.material.name),
        ("material", "n", config.material.refractive_index),
        ("material", "write_depth_um", config.material.write_depth_m / um),
        ("target", "ell", design.ell),
        ("target", "core_diameter_um", design.target_core_diameter_m / um),
        ("target", "scale_definition", design.target_scale_definition),
        ("target", "equivalent_l0_first_zero_diameter_um", design.equivalent_l0_first_zero_diameter_m / um),
        ("target", "bessel_length_um", design.target_bessel_length_m / um),
        ("design", "vortex_main_ring_diameter_um", design.vortex_main_ring_diameter_m / um),
        ("design", "gamma_slm_deg", design.gamma_slm_deg),
        ("design", "magnification_to_sample", design.magnification_to_sample),
        ("design", "objective_map_demag", obj_map.demag),
        ("design", "objective_map_source", obj_map.source),
        ("design", "predicted_bessel_length_um", design.predicted_bessel_length_m / um),
        ("design", "kr_sample_m_inv", design.kr_sample_m_inv),
        ("energy", "total_transmission", config.energy.total_transmission),
        ("energy", "total_transmission_to_surface_air", config.energy.total_transmission_to_surface_air),
        ("energy", "pulse_energy_at_sample_uJ", config.energy.pulse_energy_at_sample_J / uJ),
        ("energy", "pulse_energy_at_surface_air_uJ", config.energy.pulse_energy_at_surface_air_J / uJ),
        ("propagation", "method", config.propagation.method),
        ("propagation", "sas_pad_factor", config.propagation.sas_pad_factor),
        ("propagation", "sas_bandlimit", config.propagation.sas_bandlimit),
        ("propagation", "sas_skip_final_phase", config.propagation.sas_skip_final_phase),
        ("propagation", "sas_allow_invalid", config.propagation.sas_allow_invalid),
    ]
    return pd.DataFrame(rows, columns=["section", "quantity", "value"])


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.size > 64:
            return {"shape": list(obj.shape), "min": float(np.nanmin(obj)), "max": float(np.nanmax(obj))}
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


__all__ = [
    "LaserConfig",
    "SLMConfig",
    "ObjectiveConfig",
    "OpticalMappingMode",
    "RelayConfig",
    "MaterialConfig",
    "EnergyBudget",
    "BeamTarget",
    "BeamDesign",
    "GridConfig",
    "PropagationConfig",
    "PhysicalAxiconConfig",
    "SimulationPreset",
    "TwinConfig",
    "GenerationMethod",
    "Slm2ConjugateMode",
    "StudyKind",
    "RegimeName",
    "ValidityViolationAction",
    "SurfacePlacement",
    "default_config",
    "get_preset",
    "axial_scan_values",
    "compute_design_from_targets",
    "compute_design_from_config",
    "fixed_objective_map_from_config",
    "objective_map_from_design_inputs",
    "objective_map_from_config",
    "headline_length_tags",
    "analytic_references",
    "inverse_design_round_trip",
    "render_device_hologram",
    "phase_to_gray",
    "gray_to_phase",
    "apply_orientation",
    "build_realistic_slm_field",
    "isolate_first_order",
    "export_hologram_png",
    "angular_spectrum_propagate_bl",
    "make_bl_asm_propagator",
    "sas_validity_report",
    "scalable_angular_spectrum_propagate",
    "focus_to_focal_plane",
    "realistic_slm_to_sample",
    "build_conical_axicon_field_ideal",
    "build_bessel_gauss_field_ideal",
    "build_sample_field_ideal",
    "propagate_volume",
    "PROPAGATION_POWER_DRIFT_QUANTITATIVE_MAX",
    "propagation_power_validity_report",
    "enforce_propagation_power_validity",
    "interface_aberration_pupil",
    "interface_correction_phase",
    "fit_interface_zernike_terms",
    "extract_radial_metrics",
    "extract_vortex_safe_metrics",
    "bessel_zone_metrics",
    "bessel_region_metrics",
    "absolute_zone_length_um",
    "sampling_report",
    "energy_conservation_report",
    "run_sas_self_checks",
    "fluence_metrics",
    "modification_proxy_metrics",
    "run_case",
    "run_self_checks",
    "run_oat_sensitivity",
    "run_tradeoff_map",
    "run_ell_family_comparison",
    "run_device_realism_ablation",
    "run_interface_depth_sweep",
    "run_sampling_feasibility_envelope",
    "run_all_studies",
    "generate_standard_figures",
    "dose_to_delta_n",
    "depressed_cladding_layout",
    "solve_guided_modes",
    "config_summary",
    "ensure_output_tree",
    "m",
    "cm",
    "mm",
    "um",
    "nm",
    "fs",
    "kHz",
    "uJ",
]
