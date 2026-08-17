"""Air-to-sample interface and in-medium propagation equations.

This module owns the compact scalar formulas for:

* Snell's law for a cone angle at a planar air-sample interface,
* transverse and longitudinal wavevectors in the sample medium,
* how the non-diffracting length changes in the sample medium,
* the pupil-plane aberration phase from focusing through a flat interface
  (piston-free scalar model used for Zernike fitting),
* low-order Zernike coefficient extraction (defocus, primary spherical).

What this module does NOT own:

* the full ASM propagation through the sample
  (``bessel_twin_core`` owns that),
* Zernike mode-set management or full wavefront reconstruction,
* non-planar, tilted, or multi-layer interface models.

Coordinate convention:
  ``n_air`` is the refractive index on the laser side of the interface
  (usually 1.0 for air or the objective immersion medium).
  ``n_sample`` is the refractive index on the propagation side.
  Propagation is assumed along +z perpendicular to the flat interface.
  The transverse wavevector ``k_r`` is conserved across the interface
  (consequence of translational symmetry at a flat surface).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

TWOPI = 2.0 * math.pi
EPS = 1.0e-30


# ---------------------------------------------------------------------------
# Snell's law and wavevector relations
# ---------------------------------------------------------------------------


def snell_cone_angle_rad(theta_air_rad: float, n_sample: float, n_air: float = 1.0) -> float:
    """Return the cone half-angle in the sample from Snell's law.

    Snell: ``n_air * sin(theta_air) = n_sample * sin(theta_sample)``.

    Parameters
    ----------
    theta_air_rad:
        Cone half-angle in air (or the first medium), radians.
    n_sample:
        Refractive index of the sample medium.
    n_air:
        Refractive index of the air/first medium (default 1.0).
    """

    sin_sample = float(n_air) * math.sin(float(theta_air_rad)) / max(float(n_sample), EPS)
    sin_sample = float(np.clip(sin_sample, -1.0, 1.0))
    return float(math.asin(sin_sample))


def in_sample_transverse_wavevector_m_inv(
    kr_air_m_inv: float,
    n_sample: float,
    n_air: float = 1.0,
) -> float:
    """Return the transverse wavevector in the sample medium in rad/m.

    For a planar flat interface, translational symmetry requires the
    transverse component of the wave vector to be conserved:
    ``k_r,sample = k_r,air``.  This function makes that explicit.

    Parameters
    ----------
    kr_air_m_inv:
        Transverse wavevector in air, rad/m.
    n_sample:
        Refractive index of the sample.
    n_air:
        Refractive index of the incident medium (default 1.0).
    """

    # k_r is conserved at a flat interface regardless of n.
    # Explicitly: k_r = k * sin(theta); Snell gives n1*sin(theta1) = n2*sin(theta2),
    # but k1*sin(theta1) = (2*pi*n1/lambda)*sin(theta1) = k_r too.
    # Hence k_r,in = k_r,out for flat interface.
    return float(kr_air_m_inv)


def in_sample_wavenumber_m_inv(wavelength0_m: float, n_sample: float) -> float:
    """Return the total wavevector magnitude in the sample, ``k = 2*pi*n/lambda0``."""

    return TWOPI * float(n_sample) / max(float(wavelength0_m), EPS)


def in_sample_longitudinal_wavevector_m_inv(
    k_sample_m_inv: float,
    kr_m_inv: float,
) -> float:
    """Return the longitudinal wavevector ``k_z = sqrt(k^2 - k_r^2)`` in the sample.

    Parameters
    ----------
    k_sample_m_inv:
        Total wavenumber in the sample, rad/m.
    kr_m_inv:
        Transverse wavevector (conserved through flat interface), rad/m.
    """

    return float(math.sqrt(max(float(k_sample_m_inv) ** 2 - float(kr_m_inv) ** 2, 0.0)))


# ---------------------------------------------------------------------------
# Non-diffracting length in the sample
# ---------------------------------------------------------------------------


def non_diffracting_length_in_sample_m(
    z_max_air_m: float,
    n_sample: float,
    n_air: float = 1.0,
) -> float:
    """Return the Bessel non-diffracting length in the sample medium.

    The non-diffracting length ``z_max = w0 * k / k_r``.  Since ``k ∝ n``
    and ``k_r`` is conserved at a flat interface, the length scales linearly
    with refractive index::

        z_max,sample = z_max,air * (n_sample / n_air)

    Parameters
    ----------
    z_max_air_m:
        Non-diffracting length in air (or first medium), m.
    n_sample:
        Refractive index of the sample medium.
    n_air:
        Refractive index of the incident medium (default 1.0).
    """

    return float(z_max_air_m) * float(n_sample) / max(float(n_air), EPS)


def in_sample_core_radius_m(core_radius_air_m: float) -> float:
    """Return the Bessel core radius in the sample medium.

    The core radius ``r_core = j_{0,1} / k_r`` and ``k_r`` is conserved
    at a flat interface, so the core radius is unchanged in the sample.
    This function is a documentary reminder; it returns the argument unchanged.
    """

    return float(core_radius_air_m)


# ---------------------------------------------------------------------------
# Interface aberration phase
# ---------------------------------------------------------------------------


def interface_aberration_pupil_rad(
    rho: Any,
    *,
    depth_m: float,
    NA: float,
    n_sample: float,
    n_air: float = 1.0,
    k0_m_inv: float,
) -> np.ndarray:
    """Return the pupil-plane aberration phase from focusing through a flat interface.

    For a planar air-to-sample interface at depth ``d`` below the surface,
    the scalar path-length aberration at normalised pupil radius ``rho`` is::

        W(rho) = k0 * d * (n_sample * cos(theta_sample) - n_air * cos(theta_air))

    where ``sin(theta_air) = NA * rho / n_air`` and Snell gives
    ``sin(theta_sample) = n_air * sin(theta_air) / n_sample``.
    Piston is **not** removed here; subtract the mean over the pupil
    yourself if you need the piston-free version.

    Parameters
    ----------
    rho:
        Normalised pupil radius array (0 = axis, 1 = full aperture).
    depth_m:
        Focus depth inside the sample, m.
    NA:
        Numerical aperture of the objective (in n_air).
    n_sample:
        Refractive index of the sample.
    n_air:
        Refractive index of the immersion medium (default 1.0).
    k0_m_inv:
        Vacuum wavenumber ``2*pi/lambda0``, rad/m.
    """

    rho_arr = np.asarray(rho, dtype=float)
    sin1 = np.clip(float(NA) * rho_arr / max(float(n_air), EPS), 0.0, 1.0 - 1e-9)
    cos1 = np.sqrt(1.0 - sin1**2)
    sin2 = np.clip(float(n_air) * sin1 / max(float(n_sample), EPS), 0.0, 1.0 - 1e-9)
    cos2 = np.sqrt(1.0 - sin2**2)
    return float(k0_m_inv) * float(depth_m) * (float(n_sample) * cos2 - float(n_air) * cos1)


def interface_aberration_pupil(
    grid: dict[str, Any],
    laser: Any,
    objective: Any,
    material: Any,
    depth_m: float | None = None,
    n1: float | None = None,
) -> np.ndarray:
    """Planar air-to-crystal pupil phase in radians, with piston removed."""

    R = grid["R"]
    R_pupil = objective.pupil_radius_m
    n_in = objective.immersion_n if n1 is None else float(n1)
    n2 = float(material.refractive_index)
    depth = float(material.write_depth_m if depth_m is None else depth_m)

    rho = np.clip(R / max(R_pupil, EPS), 0.0, 1.0)
    sin1 = np.clip(objective.NA * rho / max(n_in, EPS), 0.0, 0.999999)
    cos1 = np.sqrt(1.0 - sin1**2)
    sin2 = np.clip(n_in * sin1 / max(n2, EPS), 0.0, 0.999999)
    cos2 = np.sqrt(1.0 - sin2**2)
    W = laser.k0 * depth * (n2 * cos2 - n_in * cos1)
    mask = R <= R_pupil
    if np.any(mask):
        W = W - float(np.mean(W[mask]))
    return np.where(mask, W, 0.0)


def interface_correction_phase(
    grid: dict[str, Any],
    laser: Any,
    objective: Any,
    material: Any,
    depth_m: float | None = None,
) -> np.ndarray:
    """SLM conjugate phase for the planar interface aberration."""

    return -interface_aberration_pupil(grid, laser, objective, material, depth_m=depth_m)


# ---------------------------------------------------------------------------
# Zernike coefficient extraction (piston, defocus, primary spherical)
# ---------------------------------------------------------------------------


def fit_piston_defocus_spherical_rad(
    rho: Any,
    W_rad: Any,
) -> dict[str, float]:
    """Least-squares fit of piston, defocus, and primary spherical to a pupil phase.

    Uses the Noll/Zernike convention for radially symmetric terms:

    * Piston: Z = 1
    * Defocus: Z = 2*rho^2 - 1
    * Primary spherical: Z = 6*rho^4 - 6*rho^2 + 1

    Parameters
    ----------
    rho:
        1-D normalised pupil radius array (values in [0, 1]).
    W_rad:
        1-D pupil phase array in radians, same length as rho.

    Returns
    -------
    dict with keys: ``piston_rad``, ``defocus_rad``, ``spherical_rad``,
    ``residual_rms_rad``, ``defocus_waves``, ``spherical_waves``.
    """

    r = np.asarray(rho, dtype=float).ravel()
    y = np.asarray(W_rad, dtype=float).ravel()
    A = np.vstack([np.ones_like(r), 2.0 * r**2 - 1.0, 6.0 * r**4 - 6.0 * r**2 + 1.0]).T
    coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
    fit = A @ coeff
    rms = float(np.sqrt(np.mean((y - fit) ** 2))) if y.size else float("nan")
    return {
        "piston_rad": float(coeff[0]),
        "defocus_rad": float(coeff[1]),
        "spherical_rad": float(coeff[2]),
        "residual_rms_rad": rms,
        "defocus_waves": float(coeff[1]) / (2.0 * math.pi),
        "spherical_waves": float(coeff[2]) / (2.0 * math.pi),
    }


def fit_interface_zernike_terms(
    grid: dict[str, Any],
    phase: np.ndarray,
    pupil_radius_m: float,
) -> dict[str, float]:
    """Least-squares fit to piston, defocus, and primary spherical terms."""

    rho = grid["R"] / max(float(pupil_radius_m), EPS)
    mask = rho <= 1.0
    r = rho[mask].ravel()
    y = np.asarray(phase, float)[mask].ravel()
    A = np.vstack([np.ones_like(r), 2.0 * r**2 - 1.0, 6.0 * r**4 - 6.0 * r**2 + 1.0]).T
    coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
    fit = A @ coeff
    rms = float(np.sqrt(np.mean((y - fit) ** 2))) if y.size else np.nan
    return {
        "piston_rad": float(coeff[0]),
        "defocus_rad": float(coeff[1]),
        "spherical_rad": float(coeff[2]),
        "residual_rms_rad": rms,
        "defocus_waves": float(coeff[1] / TWOPI),
        "spherical_waves": float(coeff[2] / TWOPI),
    }
