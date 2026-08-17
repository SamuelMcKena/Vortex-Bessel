"""Dual-SLM segmented radial/azimuthal vector-arm chain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from vbb_study.config import EPS
from vbb_study.equations.fields import make_xy_grid, phase_wrap
from vbb_study.slm_model import SLMApplication, apply_slm, slm_active_aperture
from vbb_study.vector_arm_config import VectorArmConfig
from vbb_study.vector_field import VectorField

TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class VectorArmRun:
    """Debuggable result for one vector-arm chain run."""

    field: VectorField
    target: VectorField
    grid: Mapping[str, Any]
    psi1_rad: np.ndarray
    psi2_rad: np.ndarray
    slm1: SLMApplication | None
    slm2: SLMApplication | None
    ledgers: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


def default_vector_arm_grid(cfg: VectorArmConfig, n: int = 512) -> dict[str, Any]:
    """Return a symmetric SLM-plane grid for vector-arm tests and notebooks."""

    active = min(cfg.slm1.active_width_m, cfg.slm1.active_height_m, cfg.slm2.active_width_m, cfg.slm2.active_height_m)
    side = min(3.6 * float(cfg.waist_m), 0.90 * active)
    return make_xy_grid(int(n), side / int(n))


def assert_grid_symmetric(grid: Mapping[str, Any]) -> None:
    """Assert the centred half-cell grid supports exact relay flips."""

    x = np.asarray(grid["x"], dtype=float)
    y = np.asarray(grid.get("y", x), dtype=float)
    assert np.allclose(x, -x[::-1], rtol=0.0, atol=max(abs(float(x[1] - x[0])), EPS) * 1e-12)
    assert np.allclose(y, -y[::-1], rtol=0.0, atol=max(abs(float(y[1] - y[0])), EPS) * 1e-12)


def _theta(grid: Mapping[str, Any]) -> np.ndarray:
    if "PHI" in grid:
        return np.asarray(grid["PHI"], dtype=float)
    return np.arctan2(np.asarray(grid["Y"], dtype=float), np.asarray(grid["X"], dtype=float))


def _radius(grid: Mapping[str, Any]) -> np.ndarray:
    if "R" in grid:
        return np.asarray(grid["R"], dtype=float)
    return np.hypot(np.asarray(grid["X"], dtype=float), np.asarray(grid["Y"], dtype=float))


def gaussian_envelope(grid: Mapping[str, Any], cfg: VectorArmConfig) -> np.ndarray:
    """Return the real PHAROS Gaussian envelope at the SLM plane."""

    return np.exp(-(_radius(grid) ** 2) / max(float(cfg.waist_m) ** 2, EPS))


def sector_offset(theta_rad: np.ndarray, cfg: VectorArmConfig, *, all_radial: bool = False) -> np.ndarray:
    """Return the segmented radial/azimuthal sector phase offset."""

    theta = np.asarray(theta_rad, dtype=float)
    if all_radial:
        return np.zeros_like(theta, dtype=float)
    n_pairs = int(cfg.n_pairs)
    cell = TWOPI / float(n_pairs)
    local = np.mod(theta - float(cfg.sector_rotation_rad), cell)
    azimuthal = local < float(cfg.sector_duty) * cell
    return np.where(azimuthal, 0.5 * np.pi, 0.0)


def segmented_alpha(
    grid: Mapping[str, Any],
    cfg: VectorArmConfig,
    *,
    all_radial: bool = False,
) -> np.ndarray:
    """Return ``theta + phi0(theta)`` for the segmented RA target."""

    theta = _theta(grid)
    return theta + sector_offset(theta, cfg, all_radial=all_radial)


def target_vector_field(
    cfg: VectorArmConfig,
    grid: Mapping[str, Any],
    *,
    all_radial: bool = False,
) -> VectorField:
    """Return the analytic segmented radial/azimuthal target field."""

    alpha = segmented_alpha(grid, cfg, all_radial=all_radial)
    amp = gaussian_envelope(grid, cfg)
    return VectorField(
        ex=amp * np.cos(alpha),
        ey=amp * np.sin(alpha),
        ez=np.zeros_like(amp, dtype=complex),
        grid=grid,
        wavelength_m=cfg.wavelength_m,
        medium_index=1.0,
        metadata={"field": "segmented_ra_target", "all_radial": bool(all_radial)},
    )


def synthesise_psi1(
    cfg: VectorArmConfig,
    grid: Mapping[str, Any],
    *,
    all_radial: bool = False,
) -> np.ndarray:
    """Return SLM1 phase in the SLM1 frame.

    The SLM1 phase is written so that after the 4f relay inversion and HWP
    swap, the V component arriving at the QWP has phase
    ``theta + phi0(theta) - pi/2`` in the SLM2/output frame.
    """

    alpha = segmented_alpha(grid, cfg, all_radial=all_radial)
    return phase_wrap(np.flip(alpha, axis=(0, 1)) - 0.5 * np.pi)


def synthesise_psi2(
    cfg: VectorArmConfig,
    grid: Mapping[str, Any],
    *,
    naive_unflipped: bool = False,
    all_radial: bool = False,
) -> np.ndarray:
    """Return SLM2 phase in SLM2's own frame.

    ``naive_unflipped=True`` intentionally computes the command in the SLM1
    frame and applies it in the flipped SLM2 frame.  This is the wrong-variant
    hook used by the two-sided relay-flip assertion.
    """

    alpha = segmented_alpha(grid, cfg, all_radial=all_radial)
    if naive_unflipped:
        alpha = np.flip(alpha, axis=(0, 1))
    return phase_wrap(-alpha - float(cfg.piston_delta_rad))


def hwp45_matrix(retardance_error_rad: float = 0.0) -> np.ndarray:
    """Return the zero-order HWP-at-45 matrix in the lab H/V basis."""

    err = float(retardance_error_rad)
    if abs(err) <= EPS:
        return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    return np.array([[0.0, np.exp(0.5j * err)], [np.exp(-0.5j * err), 0.0]], dtype=complex)


def qwp45_matrix(retardance_error_rad: float = 0.0) -> np.ndarray:
    """Return the QWP-at-45 matrix pinned to the radial-beam orientation test."""

    err = float(retardance_error_rad)
    if abs(err) <= EPS:
        return (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0j], [1.0j, 1.0]], dtype=complex)
    phase = 1.0j * np.exp(1.0j * err)
    return (1.0 / np.sqrt(2.0)) * np.array([[1.0, phase], [phase, 1.0]], dtype=complex)


def _apply_matrix(ex: np.ndarray, ey: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return matrix[0, 0] * ex + matrix[0, 1] * ey, matrix[1, 0] * ex + matrix[1, 1] * ey


def _apply_h_slm(
    field_h: np.ndarray,
    phase: np.ndarray,
    grid: Mapping[str, Any],
    cfg: VectorArmConfig,
    *,
    panel: str,
) -> tuple[np.ndarray, SLMApplication | None]:
    panel_cfg = cfg.effective_slm1 if panel == "slm1" else cfg.effective_slm2
    if cfg.ideal_components:
        return field_h * np.exp(1j * phase), None
    out = apply_slm(
        field_h,
        phase,
        grid,
        panel_cfg,
        phase_is_prepared=False,
        quantise_phase=bool(cfg.quantise),
        apply_fill_factor=bool(cfg.apply_fill_factor),
        apply_carrier=bool(cfg.apply_carrier),
        fill_factor_model=cfg.fill_factor_model,
    )
    return out.total, out


def _passive_component(component: np.ndarray, grid: Mapping[str, Any], cfg: VectorArmConfig, *, panel: str) -> np.ndarray:
    if cfg.ideal_components:
        return component
    panel_cfg = cfg.effective_slm1 if panel == "slm1" else cfg.effective_slm2
    return np.where(slm_active_aperture(grid, panel_cfg), component, 0.0)


def run_vector_arm(
    cfg: VectorArmConfig | None = None,
    *,
    grid: Mapping[str, Any] | None = None,
    naive_psi2: bool = False,
    all_radial: bool = False,
    return_debug: bool = False,
) -> VectorField | VectorArmRun:
    """Run the dual-SLM vector arm through the QWP output plane."""

    cfg = cfg or VectorArmConfig()
    grid_dict = dict(default_vector_arm_grid(cfg) if grid is None else grid)
    assert_grid_symmetric(grid_dict)

    amp = gaussian_envelope(grid_dict, cfg).astype(np.complex128)
    h = amp / np.sqrt(2.0)
    v = amp / np.sqrt(2.0)
    psi1 = synthesise_psi1(cfg, grid_dict, all_radial=all_radial)
    psi2 = synthesise_psi2(cfg, grid_dict, naive_unflipped=naive_psi2, all_radial=all_radial)

    h, slm1 = _apply_h_slm(h, psi1, grid_dict, cfg, panel="slm1")
    v = _passive_component(v, grid_dict, cfg, panel="slm1")

    h = np.flip(h, axis=(0, 1))
    v = np.flip(v, axis=(0, 1))

    h, v = _apply_matrix(h, v, hwp45_matrix(0.0 if cfg.ideal_components else cfg.hwp_retardance_error_rad))

    h, slm2 = _apply_h_slm(h, psi2, grid_dict, cfg, panel="slm2")
    v = _passive_component(v, grid_dict, cfg, panel="slm2")

    h, v = _apply_matrix(h, v, qwp45_matrix(0.0 if cfg.ideal_components else cfg.qwp_retardance_error_rad))
    field_out = VectorField(
        ex=h,
        ey=v,
        ez=np.zeros_like(h, dtype=complex),
        grid=grid_dict,
        wavelength_m=cfg.wavelength_m,
        medium_index=1.0,
        metadata={
            "stage": "qwp_output",
            "naive_psi2": bool(naive_psi2),
            "all_radial": bool(all_radial),
            "ideal_components": bool(cfg.ideal_components),
        },
    )
    target = target_vector_field(cfg, grid_dict, all_radial=all_radial)
    if not return_debug:
        return field_out
    ledgers: dict[str, Any] = {}
    if slm1 is not None:
        ledgers["slm1"] = slm1.ledger.as_dict()
    if slm2 is not None:
        ledgers["slm2"] = slm2.ledger.as_dict()
    return VectorArmRun(
        field=field_out,
        target=target,
        grid=grid_dict,
        psi1_rad=psi1,
        psi2_rad=psi2,
        slm1=slm1,
        slm2=slm2,
        ledgers=ledgers,
        metadata=dict(field_out.metadata),
    )


def fit_global_phase_error(field: VectorField, target: VectorField) -> tuple[float, float]:
    """Return ``(gamma, normalized_max_error)`` after fitting one global phase."""

    inner = np.sum(np.conj(target.ex) * field.ex + np.conj(target.ey) * field.ey)
    gamma = float(np.angle(inner))
    ex = field.ex * np.exp(-1j * gamma)
    ey = field.ey * np.exp(-1j * gamma)
    numerator = np.maximum(np.abs(ex - target.ex), np.abs(ey - target.ey))
    denom = max(float(np.max(np.sqrt(np.abs(target.ex) ** 2 + np.abs(target.ey) ** 2))), EPS)
    return gamma, float(np.max(numerator) / denom)


def local_polarization_angle(field: VectorField) -> np.ndarray:
    """Return headless local linear-polarization angle from transverse Stokes."""

    stokes = field.stokes()
    return 0.5 * np.arctan2(stokes["S2"], stokes["S1"])


def headless_angle_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return angle difference modulo pi."""

    return 0.5 * np.angle(np.exp(2j * (np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


__all__ = [
    "VectorArmRun",
    "assert_grid_symmetric",
    "default_vector_arm_grid",
    "fit_global_phase_error",
    "gaussian_envelope",
    "headless_angle_delta",
    "hwp45_matrix",
    "local_polarization_angle",
    "qwp45_matrix",
    "run_vector_arm",
    "sector_offset",
    "segmented_alpha",
    "synthesise_psi1",
    "synthesise_psi2",
    "target_vector_field",
]
