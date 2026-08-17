"""SLM2 conjugation modes: full flattening vs vortex-preserving correction."""

from __future__ import annotations

import numpy as np

from vbb_study.equations.fields import make_xy_grid
from vbb_study.vbb_axicon import slm2_conjugate


def _phase_winding(field: np.ndarray, grid: dict, radius_m: float, samples: int = 720) -> float:
    phi = np.linspace(0.0, 2.0 * np.pi, samples + 1)
    x = radius_m * np.cos(phi)
    y = radius_m * np.sin(phi)
    axis = np.asarray(grid["x"], dtype=float)
    dx = float(grid["dx"])
    ix = np.clip(np.round((x - axis[0]) / dx).astype(int), 0, len(axis) - 1)
    iy = np.clip(np.round((y - axis[0]) / dx).astype(int), 0, len(axis) - 1)
    phase = np.angle(field[iy, ix])
    dphi = np.angle(np.exp(1j * np.diff(phase)))
    return float(np.sum(dphi) / (2.0 * np.pi))


def test_full_slm2_conjugation_strips_vortex_charge() -> None:
    grid = make_xy_grid(256, 0.25e-6)
    charge = 3
    amp = np.exp(-(grid["R"] ** 2) / (18e-6) ** 2)
    residual = 0.4 * grid["X"] / np.max(np.abs(grid["X"])) + 0.2 * grid["R"] / np.max(grid["R"])
    field = amp * np.exp(1j * (charge * grid["PHI"] + residual))

    corrected = slm2_conjugate(field, mode="full", stroke_levels=None)
    winding = _phase_winding(corrected, grid, radius_m=8e-6)

    assert abs(winding) < 0.1


def test_preserve_vortex_slm2_conjugation_keeps_charge() -> None:
    grid = make_xy_grid(256, 0.25e-6)
    charge = 3
    amp = np.exp(-(grid["R"] ** 2) / (18e-6) ** 2)
    residual = 0.4 * grid["X"] / np.max(np.abs(grid["X"])) + 0.2 * grid["R"] / np.max(grid["R"])
    reference_phase = charge * grid["PHI"]
    field = amp * np.exp(1j * (reference_phase + residual))

    corrected, diag = slm2_conjugate(
        field,
        mode="preserve_vortex",
        stroke_levels=None,
        reference_phase=reference_phase,
        return_diagnostics=True,
    )
    winding = _phase_winding(corrected, grid, radius_m=8e-6)

    assert abs(winding - charge) < 0.1
    assert float(diag["residual_phase_rms_after_rad"]) < 1e-10
    assert float(diag["residual_phase_rms_before_rad"]) > 0.01
