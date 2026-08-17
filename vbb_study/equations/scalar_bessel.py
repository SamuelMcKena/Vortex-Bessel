"""Scalar Bessel and Bessel--Gauss reference equations.

This module contains the analytic equations used to validate the numerical
structured-beam model.  The reference functions deliberately distinguish three
objects that were conflated in older repository outputs:

1. an ideal infinite Bessel field, ``J_l(k_r r) exp(i l phi)``;
2. a finite-energy Bessel--Gauss (BG) beam;
3. a Gaussian field carrying the conical phase of an axicon.

The finite BG propagation formula follows the standard paraxial solution
introduced by Gori, Guattari & Padovani (Optics Communications 64, 491--495,
1987) and summarized, for example, in the Optica review by Zhan (Advances in
Optics and Photonics 1, 1--57, 2009).  The implementation is independently
checked in ``tests/test_phase2k_mathematical_truth.py`` against FFT Fresnel
propagation of the z=0 field.

Axicon geometry is likewise split into a thin phase-screen approximation and
an exact geometrical Snell-law reference.  The latter is the quantitative
reference whenever a real refractive axicon angle is known.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from scipy import optimize, special as sp

from .fields import gaussian_amplitude

TWOPI = 2.0 * math.pi
EPS = 1.0e-30


def medium_wavenumber_m_inv(wavelength0_m: float, n_medium: float = 1.0) -> float:
    """Return ``k = 2*pi*n/lambda0`` in rad/m."""

    return TWOPI * float(n_medium) / max(float(wavelength0_m), EPS)


def axicon_cone_angle_exact_rad(
    n_axicon: float,
    n_medium: float,
    base_angle_rad: float,
) -> float:
    """Return the exact ray deflection of a plano-conical refractive axicon.

    The incident beam is assumed normal to the flat face and parallel to the
    optical axis inside the axicon.  If ``gamma`` is the angle between the
    conical-surface normal and the optical axis, Snell's law gives

    ``n_ax sin(gamma) = n_medium sin(gamma + theta)``

    and therefore

    ``theta = asin((n_ax/n_medium) sin(gamma)) - gamma``.

    ``base_angle_rad`` is therefore a *base/surface-normal tilt* convention,
    not the full apex angle.  A value reported by a manufacturer as an apex
    angle must be converted before use.
    """

    n_ax = float(n_axicon)
    n_med = float(n_medium)
    gamma = float(base_angle_rad)
    if n_ax <= 0.0 or n_med <= 0.0:
        raise ValueError("refractive indices must be positive")
    argument = (n_ax / n_med) * math.sin(gamma)
    if abs(argument) > 1.0:
        raise ValueError("axicon exit surface is beyond the propagating Snell branch")
    return float(math.asin(argument) - gamma)


def transverse_wavevector_from_axicon(
    k_m_inv: float,
    n_axicon: float,
    n_medium: float,
    gamma_rad: float,
    *,
    mode: str = "tan",
) -> float:
    """Return the axicon radial wavevector ``k_r`` in rad/m.

    ``k_m_inv`` is retained as the historical argument name for API
    compatibility, but for the ``tan``/``phase_screen`` and ``small_angle``
    modes it must be the **vacuum** wavenumber ``k0 = 2*pi/lambda0``.  Older
    prose that treated it as the propagation-medium wavenumber introduced an
    erroneous extra factor of ``n_medium`` outside air.

    Modes
    -----
    ``tan`` / ``phase_screen``
        Thin-element optical-path phase gradient,
        ``k_r = k0 (n_ax - n_medium) tan(gamma)``.
    ``small_angle``
        First-order approximation,
        ``k_r = k0 (n_ax - n_medium) gamma``.
    ``snell_exact``
        Exact geometrical reference,
        ``k_r = k0*n_medium*sin(theta)`` with ``theta`` from Snell's law.
    """

    k0 = float(k_m_inv)
    n_ax = float(n_axicon)
    n_med = float(n_medium)
    gamma = float(gamma_rad)
    key = str(mode).lower().strip()
    if key == "small_angle":
        return k0 * (n_ax - n_med) * gamma
    if key in {"snell_exact", "exact", "snell"}:
        theta = axicon_cone_angle_exact_rad(n_ax, n_med, gamma)
        return k0 * n_med * math.sin(theta)
    if key in {"tan", "phase_screen", "thin_phase"}:
        return k0 * (n_ax - n_med) * math.tan(gamma)
    raise ValueError(f"unsupported axicon wavevector mode {mode!r}")


def cone_angle_from_kr(kr_m_inv: float, k_m_inv: float) -> float:
    """Return cone angle from ``sin(theta)=k_r/k`` for a propagating cone."""

    ratio = float(kr_m_inv) / max(float(k_m_inv), EPS)
    if abs(ratio) > 1.0 + 1.0e-12:
        raise ValueError("|k_r| exceeds the propagation-medium wavenumber")
    return float(math.asin(float(np.clip(ratio, -1.0, 1.0))))


def longitudinal_wavevector_m_inv(k_m_inv: float, kr_m_inv: float) -> float:
    """Return ``k_z = sqrt(k^2-k_r^2)`` for a propagating cone."""

    k = float(k_m_inv)
    kr = float(kr_m_inv)
    if abs(kr) > abs(k) + 1.0e-12 * max(abs(k), 1.0):
        raise ValueError("|k_r| exceeds the propagation-medium wavenumber")
    return float(math.sqrt(max(k * k - kr * kr, 0.0)))


def bessel_gauss_field(
    R_m: Any,
    Phi_rad: Any,
    *,
    ell: int,
    kr_m_inv: float,
    waist_m: float,
    z_m: float = 0.0,
    wavelength0_m: float | None = None,
    n_medium: float = 1.0,
    amplitude: complex = 1.0,
    include_carrier: bool = False,
) -> np.ndarray:
    """Return the paraxial finite-energy Bessel--Gauss field.

    At the waist plane the field is exactly

    ``A J_|l|(k_r r) exp(-r^2/w0^2) exp(i l phi)``.

    For non-zero ``z_m`` the Gaussian width, wavefront curvature, complex
    Bessel argument and Bessel-cone axial phase must all evolve.  Writing

    ``q = 1 + i z/z_R`` and ``z_R = k w0^2/2``, the slowly varying envelope is

    ``U = A/q * J_|l|(k_r r/q) * exp[-r^2/(w0^2 q)]``
    ``    * exp[-i k_r^2 z/(2 k q)] * exp(i l phi)``.

    This expression is the paraxial BG solution and reduces identically to the
    requested waist field at ``z=0``.  The previous repository implementation
    incorrectly froze the transverse envelope and multiplied it only by an
    axial carrier; that is not a propagated finite Bessel--Gauss beam.

    ``wavelength0_m`` is required when ``z_m != 0``.  By default the function
    returns the slowly varying envelope.  Set ``include_carrier=True`` to also
    multiply by ``exp(i k z)``.
    """

    R = np.asarray(R_m, dtype=float)
    Phi = np.asarray(Phi_rad, dtype=float)
    ell_i = int(ell)
    kr = float(kr_m_inv)
    w0 = max(float(waist_m), EPS)
    z = float(z_m)

    if abs(z) <= EPS:
        envelope = sp.jv(abs(ell_i), kr * R) * np.exp(-(R**2) / (w0**2))
        return np.asarray(amplitude, dtype=complex) * envelope * np.exp(1j * ell_i * Phi)

    if wavelength0_m is None:
        raise ValueError("wavelength0_m is required for Bessel--Gauss propagation at z != 0")
    k = medium_wavenumber_m_inv(float(wavelength0_m), float(n_medium))
    z_rayleigh = 0.5 * k * w0 * w0
    q = 1.0 + 1j * z / max(z_rayleigh, EPS)
    radial = sp.jv(abs(ell_i), (kr * R) / q)
    gaussian = np.exp(-(R**2) / (w0 * w0 * q))
    cone_phase = np.exp(-1j * kr * kr * z / (2.0 * k * q))
    field = (
        np.asarray(amplitude, dtype=complex)
        * radial
        * gaussian
        * cone_phase
        * np.exp(1j * ell_i * Phi)
        / q
    )
    if include_carrier:
        field = field * np.exp(1j * k * z)
    return np.asarray(field, dtype=complex)


def first_jprime_zero(ell: int, order: int = 1) -> float:
    """Return a zero of ``J'_ell`` for the *infinite-Bessel* reference."""

    ell_abs = abs(int(ell))
    order_i = int(order)
    if ell_abs == 0:
        raise ValueError("The l=0 mode has a bright core, not a vortex ring.")
    if order_i < 1:
        raise ValueError("order must be at least 1")
    return float(sp.jnp_zeros(ell_abs, order_i)[order_i - 1])


def ring_radius_from_jprime_zero_m(ell: int, kr_m_inv: float, order: int = 1) -> float:
    """Return the pure/infinite-Bessel ring radius ``J'_ell=0``.

    This is an asymptotic reference.  It is **not** the exact peak of a
    Gaussian-apodized finite BG beam unless the Gaussian envelope is locally
    flat over the ring.
    """

    return first_jprime_zero(ell, order=order) / max(abs(float(kr_m_inv)), EPS)


def bessel_gauss_ring_radius_m(
    ell: int,
    kr_m_inv: float,
    waist_m: float,
    order: int = 1,
) -> float:
    """Return the actual z=0 bright-ring radius of a finite BG intensity.

    The maximum is found from

    ``I(r) = |J_l(k_r r)|^2 exp(-2 r^2/w0^2)``.

    This removes a systematic error in older metrics that used a zero of
    ``J'_ell`` even when the Gaussian apodization was non-negligible.
    """

    ell_abs = abs(int(ell))
    if ell_abs == 0:
        raise ValueError("The l=0 mode has a bright core, not a vortex ring.")
    order_i = int(order)
    if order_i < 1:
        raise ValueError("order must be at least 1")
    kr = abs(float(kr_m_inv))
    w0 = max(float(waist_m), EPS)
    roots = sp.jn_zeros(ell_abs, order_i)
    lo_x = 0.0 if order_i == 1 else float(roots[order_i - 2])
    hi_x = float(roots[order_i - 1])

    def neg_intensity(x: float) -> float:
        r = x / max(kr, EPS)
        return -float(sp.jv(ell_abs, x) ** 2 * math.exp(-2.0 * r * r / (w0 * w0)))

    result = optimize.minimize_scalar(
        neg_intensity,
        bounds=(lo_x, hi_x),
        method="bounded",
        options={"xatol": 1.0e-13},
    )
    if not result.success:
        raise RuntimeError("failed to locate finite Bessel--Gauss ring maximum")
    return float(result.x / max(kr, EPS))


def j0_first_null_radius_m(kr_m_inv: float) -> float:
    """Return the exact first zero of ``J_0(k_r r)`` in metres."""

    return float(sp.jn_zeros(0, 1)[0] / max(abs(float(kr_m_inv)), EPS))


def l0_hwhm_core_radius_m(kr_m_inv: float) -> float:
    """Return the l=0 Bessel intensity half-width-at-half-maximum radius."""

    root = optimize.brentq(
        lambda x: float(sp.jv(0, x) ** 2 - 0.5),
        0.0,
        float(sp.jn_zeros(0, 1)[0]),
    )
    return float(root / max(abs(float(kr_m_inv)), EPS))


def non_diffracting_length_m(
    waist_m: float,
    kr_m_inv: float,
    *,
    wavelength0_m: float,
    n_medium: float = 1.0,
) -> float:
    """Return the standard geometric BG overlap-length estimate ``w0*k/k_r``.

    This is a design/reference length, not an exact FWHM of a propagated
    numerical intensity trace.  Those are separate observables and must not be
    labelled interchangeably.
    """

    k = medium_wavenumber_m_inv(float(wavelength0_m), float(n_medium))
    return float(float(waist_m) * k / max(abs(float(kr_m_inv)), EPS))


def normalized_radial_intensity(ell: int, radius_m: Any, kr_m_inv: float) -> np.ndarray:
    """Return the *infinite-Bessel* reference ``J_l(k_r r)^2``, normalized."""

    radius = np.asarray(radius_m, dtype=float)
    I = np.abs(sp.jv(abs(int(ell)), float(kr_m_inv) * radius)) ** 2
    scale = float(np.nanmax(I)) if I.size else 0.0
    return I / max(scale, EPS)


# ---------------------------------------------------------------------------
# Conical axicon input field (distinct from the Bessel--Gauss target)
# ---------------------------------------------------------------------------


def conical_axicon_input_field(
    R_m: Any,
    Phi_rad: Any,
    *,
    ell: int,
    kr_m_inv: float,
    waist_m: float,
    amplitude: complex = 1.0,
) -> np.ndarray:
    """Return a Gaussian-apodized conical axicon-phase source field."""

    R = np.asarray(R_m, dtype=float)
    Phi = np.asarray(Phi_rad, dtype=float)
    envelope = np.exp(-(R**2) / max(float(waist_m), EPS) ** 2)
    conical_phase = np.exp(1j * (-float(kr_m_inv) * R + int(ell) * Phi))
    return complex(amplitude) * envelope * conical_phase


def build_conical_axicon_field_ideal(
    grid: dict[str, Any],
    design: Any,
    laser: Any,
    include_vortex: bool = True,
) -> np.ndarray:
    """Return the idealized Gaussian conical-phase field at an axicon plane."""

    amp = gaussian_amplitude(grid["R"], max(design.w0_sample_m, grid["dx"]))
    phase = -design.kr_sample_m_inv * grid["R"]
    if include_vortex:
        phase = phase + design.ell * grid["PHI"]
    return amp * np.exp(1j * phase)


def build_bessel_gauss_field_ideal(
    grid: dict[str, Any],
    design: Any,
    laser: Any | None = None,
    include_vortex: bool = True,
) -> np.ndarray:
    """Return the z=0 finite Bessel--Gauss target field."""

    ell = int(design.ell) if include_vortex else 0
    envelope = sp.jv(abs(ell), design.kr_sample_m_inv * grid["R"])
    envelope = envelope * np.exp(
        -(grid["R"] ** 2) / max(design.w0_sample_m, grid["dx"], EPS) ** 2
    )
    return envelope * np.exp(1j * ell * grid["PHI"])


def build_sample_field_ideal(
    grid: dict[str, Any],
    design: Any,
    laser: Any,
    include_vortex: bool = True,
) -> np.ndarray:
    """Compatibility wrapper for the legacy conical axicon-source field."""

    warnings.warn(
        "build_sample_field_ideal() returns the legacy conical axicon-phase field; "
        "use build_conical_axicon_field_ideal() or build_bessel_gauss_field_ideal() explicitly.",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_conical_axicon_field_ideal(grid, design, laser, include_vortex=include_vortex)


# ---------------------------------------------------------------------------
# Readable aliases
# ---------------------------------------------------------------------------


def vortex_ring_radius_m(ell: int, kr_m_inv: float, order: int = 1) -> float:
    """Return the *infinite-Bessel* ``J'_ell`` ring reference.

    For a finite Bessel--Gauss beam use :func:`bessel_gauss_ring_radius_m`.
    """

    return ring_radius_from_jprime_zero_m(ell, kr_m_inv, order=order)
