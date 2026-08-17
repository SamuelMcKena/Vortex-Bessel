"""Holographic axicon, vortex, and SLM phase equations.

These helpers collect the phase formulas used to build phase-only structured
beam holograms.  A key Phase 2K convention is explicit here: the thin-element
axicon optical-path phase uses the **vacuum** wavenumber ``k0=2*pi/lambda0``.
Using a propagation-medium wavenumber and then multiplying by
``(n_axicon-n_medium)`` would count the external refractive index twice.

The physical refractive-axicon route is validated separately against Snell-law
geometry; this module describes the digital/thin phase-screen command.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import special as sp

TWOPI = 2.0 * math.pi
EPS = 1.0e-30


def wrap_phase_rad(phase_rad: Any) -> np.ndarray:
    """Wrap a phase array into ``[0, 2*pi)`` radians."""

    return np.mod(np.asarray(phase_rad, dtype=float), TWOPI)


def axicon_phase_rad(
    R_m: Any,
    *,
    k_m_inv: float,
    n_axicon: float,
    n_medium: float = 1.0,
    gamma_rad: float,
) -> np.ndarray:
    """Return a thin-element holographic axicon phase in radians.

    The retained keyword name ``k_m_inv`` is historical API compatibility. Its
    value **must be the vacuum wavenumber** ``k0 = 2*pi/lambda0``.  The optical
    path difference of a conical thickness profile gives

    ``phi(r) = -k0 (n_axicon - n_medium) r tan(gamma)``.

    This is the phase-screen approximation, not the exact ray-deflection law of
    a real refractive axicon.  For quantitative physical-axicon geometry use the
    Snell-law reference in :mod:`vbb_study.equations.scalar_bessel` /
    ``vortex_error_reference_models``.
    """

    R = np.asarray(R_m, dtype=float)
    return -float(k_m_inv) * (float(n_axicon) - float(n_medium)) * R * math.tan(float(gamma_rad))


def spp_phase_rad(Phi_rad: Any, ell: int) -> np.ndarray:
    """Return the spiral phase plate term ``ell * phi`` in radians."""

    return int(ell) * np.asarray(Phi_rad, dtype=float)


def combined_axicon_spp_phase_rad(
    R_m: Any,
    Phi_rad: Any,
    *,
    k_m_inv: float,
    n_axicon: float,
    n_medium: float = 1.0,
    gamma_rad: float,
    ell: int,
) -> np.ndarray:
    """Return thin axicon phase plus spiral phase ``ell*phi``.

    ``k_m_inv`` retains the historical name but, exactly as in
    :func:`axicon_phase_rad`, must be the vacuum wavenumber ``k0``.
    """

    return axicon_phase_rad(
        R_m,
        k_m_inv=k_m_inv,
        n_axicon=n_axicon,
        n_medium=n_medium,
        gamma_rad=gamma_rad,
    ) + spp_phase_rad(Phi_rad, ell)


def signum_bessel_pi_flip_phase_rad(R_m: Any, *, ell: int, kr_m_inv: float) -> np.ndarray:
    """Return the optional pi flip used to keep Bessel radial signs explicit.

    This is separate from the canonical conical+spiral phase because it is a
    signed-amplitude encoding choice, not part of the physical axicon phase.
    """

    R = np.asarray(R_m, dtype=float)
    J = sp.jv(abs(int(ell)), float(kr_m_inv) * R)
    return np.where(J < 0.0, math.pi, 0.0)


def blazed_carrier_phase_rad(X_m: Any, carrier_cpm: float) -> np.ndarray:
    """Return a linear carrier phase ``2*pi*f_x*x`` in radians."""

    return TWOPI * float(carrier_cpm) * np.asarray(X_m, dtype=float)


def quantize_phase_rad(phase_rad: Any, bits: int) -> np.ndarray:
    """Round a wrapped phase to the nearest SLM phase level."""

    bits_i = int(bits)
    if bits_i <= 0:
        raise ValueError("bits must be positive.")
    levels = 1 << bits_i
    idx = np.floor(wrap_phase_rad(phase_rad) / TWOPI * levels + 0.5).astype(np.int64) % levels
    return idx.astype(float) * TWOPI / float(levels)


def phase_to_gray(phase_rad: Any, bits: int = 8, *, invert: bool = False) -> np.ndarray:
    """Convert a wrapped phase command to an 8-bit upload image."""

    bits_i = int(bits)
    levels = 1 << bits_i
    idx = np.floor(wrap_phase_rad(phase_rad) / TWOPI * levels + 0.5).astype(np.int64) % levels
    gray = idx.astype(np.uint8) if levels == 256 else np.round(idx * 255.0 / max(levels - 1, 1)).astype(np.uint8)
    return np.uint8(255) - gray if invert else gray


def gray_to_phase(gray: Any, bits: int = 8, *, invert: bool = False) -> np.ndarray:
    """Approximate the wrapped phase represented by an SLM grayscale image."""

    g = np.asarray(gray, dtype=np.uint8)
    if invert:
        g = np.uint8(255) - g
    levels = 1 << int(bits)
    idx = g.astype(np.int64) if levels == 256 else np.round(g.astype(float) * (levels - 1) / 255.0).astype(np.int64)
    return idx.astype(float) * TWOPI / float(levels)


def fill_factor_amplitude(
    X_m: Any,
    Y_m: Any,
    *,
    pixel_pitch_m: float,
    fill_factor: float,
    sample_spacing_m: float,
) -> np.ndarray:
    """Return a square-pixel fill-factor amplitude mask.

    When the simulation has roughly one sample per SLM pixel a uniform
    ``sqrt(fill_factor)`` amplitude preserves the intended total-power factor.
    At finer sampling the inactive pixel border is resolved explicitly.
    """

    X = np.asarray(X_m, dtype=float)
    Y = np.asarray(Y_m, dtype=float)
    ff = float(np.clip(fill_factor, 0.0, 1.0))
    if ff >= 1.0:
        return np.ones_like(X, dtype=float)
    if float(sample_spacing_m) >= 0.5 * float(pixel_pitch_m):
        return math.sqrt(ff) * np.ones_like(X, dtype=float)
    duty = math.sqrt(ff)
    xmod = np.mod(X / float(pixel_pitch_m) + 0.5, 1.0) - 0.5
    ymod = np.mod(Y / float(pixel_pitch_m) + 0.5, 1.0) - 0.5
    return ((np.abs(xmod) <= 0.5 * duty) & (np.abs(ymod) <= 0.5 * duty)).astype(float)
