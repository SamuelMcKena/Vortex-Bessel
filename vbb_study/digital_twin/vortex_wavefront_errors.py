"""Declared-plane wavefront-error maps for controlled aberration studies.

Zernikes here are *OPD specifications*, not physical surrogates for a named
misaligned optic.  A map may be applied to the incoming beam, a 4F lens, or a
measured/declared pupil plane.  Physical misalignment families remain separate.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Mapping


EPS = np.finfo(float).tiny

ZERNIKE_NAMES = (
    "defocus",
    "astigmatism_x",
    "astigmatism_y",
    "coma_x",
    "coma_y",
    "spherical",
)


def unit_rms_zernike(
    name: str,
    grid: Mapping[str, Any],
    *,
    pupil_radius_m: float,
    centre_m: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    if name not in ZERNIKE_NAMES:
        raise ValueError(f"unsupported Zernike {name!r}")
    X = np.asarray(grid["X"], dtype=float) - float(centre_m[0])
    Y = np.asarray(grid["Y"], dtype=float) - float(centre_m[1])
    rho = np.hypot(X, Y) / float(pupil_radius_m)
    theta = np.arctan2(Y, X)
    if name == "defocus":
        raw = 2.0 * rho**2 - 1.0
    elif name == "astigmatism_x":
        raw = rho**2 * np.cos(2.0 * theta)
    elif name == "astigmatism_y":
        raw = rho**2 * np.sin(2.0 * theta)
    elif name == "coma_x":
        raw = (3.0 * rho**3 - 2.0 * rho) * np.cos(theta)
    elif name == "coma_y":
        raw = (3.0 * rho**3 - 2.0 * rho) * np.sin(theta)
    else:
        raw = 6.0 * rho**4 - 6.0 * rho**2 + 1.0
    mask = rho <= 1.0
    rms = float(np.sqrt(np.mean(raw[mask] ** 2)))
    return np.where(mask, raw / max(rms, EPS), 0.0)


def zernike_opd_map_m(
    name: str,
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    waves_rms: float,
    pupil_radius_m: float,
    centre_m: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Return OPD in metres for a declared RMS aberration in waves."""

    return (
        float(waves_rms)
        * float(wavelength_m)
        * unit_rms_zernike(
            name,
            grid,
            pupil_radius_m=float(pupil_radius_m),
            centre_m=centre_m,
        )
    )


def opd_to_phase_rad(opd_map_m: np.ndarray, wavelength_m: float) -> np.ndarray:
    return 2.0 * np.pi * np.asarray(opd_map_m, dtype=float) / float(wavelength_m)
