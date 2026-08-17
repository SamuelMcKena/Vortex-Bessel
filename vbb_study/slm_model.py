"""SLM pixelation, quantisation, carrier, and fill-factor bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import numpy as np

from vbb_study.config import EPS
from vbb_study.equations.fields import phase_wrap
from vbb_study.vector_arm_config import SLMPanelConfig

TWOPI = 2.0 * np.pi

SLMFillFactorModel = Literal[
    "throughput_only",
    "resolved_pixel_aperture",
    "coherent_unmodulated_deadspace",
]
SLM_FILL_FACTOR_MODELS: tuple[str, ...] = (
    "throughput_only",
    "resolved_pixel_aperture",
    "coherent_unmodulated_deadspace",
)


def _grid_xy(grid: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, float, float]:
    if "X" in grid and "Y" in grid:
        X = np.asarray(grid["X"], dtype=float)
        Y = np.asarray(grid["Y"], dtype=float)
    else:
        if "x" not in grid:
            raise ValueError("grid must provide X/Y arrays or an x coordinate.")
        x = np.asarray(grid["x"], dtype=float)
        y = np.asarray(grid.get("y", x), dtype=float)
        X, Y = np.meshgrid(x, y, indexing="xy")
    dx = float(grid.get("dx", np.median(np.diff(np.asarray(grid["x"], dtype=float)))))
    dy = float(grid.get("dy", dx))
    return X, Y, dx, dy


def field_power(field: np.ndarray, grid: Mapping[str, Any]) -> float:
    """Return ``sum(|field|^2) dx dy``."""

    _, _, dx, dy = _grid_xy(grid)
    return float(np.sum(np.abs(np.asarray(field, dtype=complex)) ** 2) * dx * dy)


def slm_active_aperture(grid: Mapping[str, Any], panel_cfg: SLMPanelConfig) -> np.ndarray:
    """Return the rectangular active-area aperture for one SLM panel."""

    X, Y, _, _ = _grid_xy(grid)
    half_w = 0.5 * panel_cfg.active_width_m
    half_h = 0.5 * panel_cfg.active_height_m
    return (np.abs(X) <= half_w) & (np.abs(Y) <= half_h)


def resolved_pixel_aperture(
    grid: Mapping[str, Any],
    panel_cfg: SLMPanelConfig,
) -> np.ndarray:
    """Return a resolved square-pixel active-area mask with mean area ``FF``.

    A binary pixel aperture is meaningful only when the computational grid has
    at least two samples per physical pixel in both axes. Coarser simulations
    must use ``throughput_only`` or the explicitly labelled unresolved
    coherent fallback instead of presenting aliased borders as hardware.
    """

    X, Y, dx, dy = _grid_xy(grid)
    pitch = float(panel_cfg.pitch_m)
    if max(abs(dx), abs(dy)) > 0.5 * pitch:
        raise ValueError(
            "resolved_pixel_aperture requires at least two computational "
            "samples per SLM pixel in both axes"
        )
    ff = float(np.clip(panel_cfg.fill_factor, 0.0, 1.0))
    if ff >= 1.0:
        return np.ones_like(X, dtype=float)
    duty = np.sqrt(ff)
    xmod = np.mod(X / pitch + 0.5, 1.0) - 0.5
    ymod = np.mod(Y / pitch + 0.5, 1.0) - 0.5
    return ((np.abs(xmod) <= 0.5 * duty) & (np.abs(ymod) <= 0.5 * duty)).astype(float)


def _coherent_deadspace_mask(
    grid: Mapping[str, Any],
    panel_cfg: SLMPanelConfig,
) -> tuple[np.ndarray, str]:
    """Return the coherent active-region mask and its sampling classification."""

    X, _, dx, dy = _grid_xy(grid)
    if max(abs(dx), abs(dy)) <= 0.5 * float(panel_cfg.pitch_m):
        return resolved_pixel_aperture(grid, panel_cfg), "resolved_binary_pixel_aperture"
    return (
        np.full_like(X, float(np.clip(panel_cfg.fill_factor, 0.0, 1.0)), dtype=float),
        "unresolved_effective_duty",
    )


def pixelate(
    phase: np.ndarray,
    grid: Mapping[str, Any],
    panel_cfg: SLMPanelConfig,
) -> np.ndarray:
    """Area-average a continuous phase target onto the SLM pixel grid.

    The average is a circular mean of ``exp(i phase)`` over computational
    samples whose cell centres fall inside each SLM top-hat pixel.  The result
    is embedded back onto the computational window; samples outside the
    rectangular active area receive zero phase and are masked by
    :func:`slm_active_aperture` during application.
    """

    phi = np.asarray(phase, dtype=float)
    X, Y, _, _ = _grid_xy(grid)
    if phi.shape != X.shape:
        raise ValueError("phase and grid arrays must have the same shape.")

    aperture = slm_active_aperture(grid, panel_cfg)
    x0 = X + 0.5 * panel_cfg.active_width_m
    y0 = Y + 0.5 * panel_cfg.active_height_m
    ix = np.floor(x0 / float(panel_cfg.pitch_m)).astype(np.int64)
    iy = np.floor(y0 / float(panel_cfg.pitch_m)).astype(np.int64)
    valid = aperture & (ix >= 0) & (ix < panel_cfg.n_x) & (iy >= 0) & (iy < panel_cfg.n_y)
    linear = iy[valid] * int(panel_cfg.n_x) + ix[valid]
    n_pix = int(panel_cfg.n_x) * int(panel_cfg.n_y)

    phasor = np.exp(1j * phi[valid])
    real_sum = np.bincount(linear, weights=np.real(phasor), minlength=n_pix)
    imag_sum = np.bincount(linear, weights=np.imag(phasor), minlength=n_pix)
    counts = np.bincount(linear, minlength=n_pix)
    pixel_phase = np.zeros(n_pix, dtype=float)
    occupied = counts > 0
    pixel_phase[occupied] = np.angle(real_sum[occupied] + 1j * imag_sum[occupied])

    out = np.zeros_like(phi, dtype=float)
    out[valid] = pixel_phase[linear]
    return phase_wrap(out)


def quantise(phase: np.ndarray, levels: int) -> np.ndarray:
    """Round phase to ``levels`` equally spaced samples over ``2*pi``."""

    levels_i = int(levels)
    if levels_i < 2:
        raise ValueError("levels must be at least 2.")
    step = TWOPI / float(levels_i)
    return phase_wrap(np.round(np.asarray(phase, dtype=float) / step) * step)


def quantize(phase: np.ndarray, levels: int) -> np.ndarray:
    """American-spelling alias for :func:`quantise`."""

    return quantise(phase, levels)


def carrier_phase(grid: Mapping[str, Any], panel_cfg: SLMPanelConfig) -> np.ndarray:
    """Return the signed blazed carrier phase for one panel."""

    X, _, _, _ = _grid_xy(grid)
    return TWOPI * panel_cfg.carrier_lp_per_m * X


def prepare_slm_phase(
    phase: np.ndarray,
    grid: Mapping[str, Any],
    panel_cfg: SLMPanelConfig,
    *,
    quantise_phase: bool = True,
    apply_carrier: bool = True,
) -> np.ndarray:
    """Pixelate, optionally add carrier, and optionally quantise a phase map."""

    prepared = pixelate(phase, grid, panel_cfg)
    if apply_carrier:
        prepared = prepared + carrier_phase(grid, panel_cfg)
    if quantise_phase:
        prepared = quantise(prepared, panel_cfg.phase_levels)
    return phase_wrap(prepared)


@dataclass(frozen=True)
class SLMLedger:
    """Power accounting immediately after one SLM reflection."""

    input_power: float
    modulated_power: float
    unmodulated_power: float
    interference_power: float
    total_power: float
    relative_error: float

    def as_dict(self) -> dict[str, float]:
        return {
            "input_power": self.input_power,
            "modulated_power": self.modulated_power,
            "unmodulated_power": self.unmodulated_power,
            "interference_power": self.interference_power,
            "total_power": self.total_power,
            "relative_error": self.relative_error,
        }


@dataclass(frozen=True)
class SLMApplication:
    """Output fields and ledger from one SLM application."""

    total: np.ndarray
    modulated: np.ndarray
    unmodulated: np.ndarray
    phase_rad: np.ndarray
    aperture: np.ndarray
    ledger: SLMLedger
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _ledger(
    modulated: np.ndarray,
    unmodulated: np.ndarray,
    total: np.ndarray,
    incident: np.ndarray,
    grid: Mapping[str, Any],
) -> SLMLedger:
    _, _, dx, dy = _grid_xy(grid)
    scale = dx * dy
    p_in = float(np.sum(np.abs(incident) ** 2) * scale)
    p_mod = float(np.sum(np.abs(modulated) ** 2) * scale)
    p_unmod = float(np.sum(np.abs(unmodulated) ** 2) * scale)
    interference = float(2.0 * np.real(np.sum(modulated * np.conj(unmodulated)) * scale))
    p_total = float(np.sum(np.abs(total) ** 2) * scale)
    rel = abs(p_total - (p_mod + p_unmod + interference)) / max(p_in, EPS)
    return SLMLedger(
        input_power=p_in,
        modulated_power=p_mod,
        unmodulated_power=p_unmod,
        interference_power=interference,
        total_power=p_total,
        relative_error=float(rel),
    )


def apply_slm(
    field: np.ndarray,
    phase: np.ndarray,
    grid: Mapping[str, Any],
    panel_cfg: SLMPanelConfig,
    *,
    phase_is_prepared: bool = False,
    quantise_phase: bool = True,
    apply_fill_factor: bool = True,
    apply_carrier: bool = True,
    fill_factor_model: SLMFillFactorModel = "coherent_unmodulated_deadspace",
) -> SLMApplication:
    """Apply the Stage 7 phase-only SLM model to one scalar component.

    ``throughput_only`` applies ``sqrt(FF) exp(i psi)`` and owns the associated
    power loss. ``resolved_pixel_aperture`` applies a binary active-pixel mask.
    ``coherent_unmodulated_deadspace`` applies
    ``M exp(i psi) + (1-M)`` and therefore retains the coherent zero order.

    The coherent model is the compatibility default for the established vector
    route. On an unresolved grid it uses the old effective-duty expression
    ``M=FF`` and labels that approximation in metadata.
    """

    incident_full = np.asarray(field, dtype=complex)
    if phase_is_prepared:
        psi = phase_wrap(np.asarray(phase, dtype=float))
    else:
        psi = prepare_slm_phase(
            phase,
            grid,
            panel_cfg,
            quantise_phase=quantise_phase,
            apply_carrier=apply_carrier,
        )
    if incident_full.shape != psi.shape:
        raise ValueError("field and phase arrays must have the same shape.")
    aperture = slm_active_aperture(grid, panel_cfg)
    incident = np.where(aperture, incident_full, 0.0)
    model = str(fill_factor_model).strip().lower()
    if model not in SLM_FILL_FACTOR_MODELS:
        raise ValueError(
            f"fill_factor_model must be one of {SLM_FILL_FACTOR_MODELS}; got {fill_factor_model!r}"
        )
    ff = float(panel_cfg.fill_factor) if apply_fill_factor else 1.0
    sampling = "not_applicable"
    if model == "throughput_only":
        modulated = np.sqrt(ff) * np.exp(1j * psi) * incident
        unmodulated = np.zeros_like(modulated)
    elif model == "resolved_pixel_aperture":
        mask = resolved_pixel_aperture(grid, panel_cfg) if apply_fill_factor else np.ones_like(psi)
        modulated = mask * np.exp(1j * psi) * incident
        unmodulated = np.zeros_like(modulated)
        sampling = "resolved_binary_pixel_aperture"
    else:
        mask, sampling = (
            _coherent_deadspace_mask(grid, panel_cfg)
            if apply_fill_factor
            else (np.ones_like(psi), "fill_factor_disabled")
        )
        modulated = mask * np.exp(1j * psi) * incident
        unmodulated = (1.0 - mask) * incident
    total = modulated + unmodulated
    ledger = _ledger(modulated, unmodulated, total, incident, grid)
    assert ledger.relative_error < 1e-12
    return SLMApplication(
        total=np.asarray(total, dtype=np.complex128),
        modulated=np.asarray(modulated, dtype=np.complex128),
        unmodulated=np.asarray(unmodulated, dtype=np.complex128),
        phase_rad=psi,
        aperture=aperture,
        ledger=ledger,
        metadata={
            "fill_factor": ff,
            "fill_factor_model": model,
            "fill_factor_sampling": sampling,
            "fill_factor_loss_already_in_field": bool(model != "coherent_unmodulated_deadspace"),
            "energy_ledger_must_not_reapply_fill_factor": True,
            "quantise_phase": bool(quantise_phase),
            "apply_carrier": bool(apply_carrier),
            "phase_is_prepared": bool(phase_is_prepared),
        },
    )


__all__ = [
    "SLMApplication",
    "SLMFillFactorModel",
    "SLM_FILL_FACTOR_MODELS",
    "SLMLedger",
    "apply_slm",
    "carrier_phase",
    "field_power",
    "pixelate",
    "prepare_slm_phase",
    "quantise",
    "quantize",
    "resolved_pixel_aperture",
    "slm_active_aperture",
]
