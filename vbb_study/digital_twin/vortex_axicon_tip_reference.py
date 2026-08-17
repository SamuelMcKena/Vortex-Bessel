"""Independent high-resolution reference models for non-ideal axicon tips.

The 2-D production grid is deliberately *not* used here.  Rounded/blunt axicon
features can be much smaller than the 10 mm system window sampling, so this
module evaluates the axisymmetric scalar Fresnel integral directly on a fine
radial mesh.  It provides a validation reference and a hard spatial-resolution
contract for the 2-D system-error sweeps.

References
----------
Brzobohaty, Cizmar & Zemanek, Optics Express 16, 12688-12700 (2008): a
rounded tip produces a second refracted component whose interference with the
conical quasi-Bessel component causes axial intensity modulation.

Mylnikov & Sokolovskii, Optik 268, 169797 (2022), and Mylnikov et al.,
Technical Physics Letters 48 (2022): hyperbolic / parabolic-hyperbolic rounded
axicon surfaces can be parameterised by a *radial* curvature scale, and the
axial droplet period depends on the detailed surface profile.

This module is a scalar, axisymmetric reference.  It does not authorise a
large-angle vector/refractive claim for the physical laboratory axicon.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from vbb_study.digital_twin.vortex_error_reference_models import (
    exact_refractive_axicon_kr_m_inv,
    snell_axicon_geometry,
)


EPS = np.finfo(float).tiny
TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class TipResolution:
    characteristic_radius_m: float
    grid_dx_m: float
    radius_pixels: float
    minimum_pixels: float
    resolved: bool


def hyperbolic_round_tip_sag_m(
    radius_m: np.ndarray,
    *,
    base_angle_rad: float,
    curvature_radius_m: float,
) -> np.ndarray:
    """Hyperbolic rounded-tip surface with a physical radial curvature scale.

    With ``v = tan(gamma)`` and radial hyperbola parameter ``r_h`` the surface
    relative to its vertex is

        f(r) = v [sqrt(r^2 + r_h^2) - r_h].

    It is smooth at the vertex and tends to the ideal cone ``v*r`` away from
    the rounded region, up to an irrelevant constant thickness offset.
    ``curvature_radius_m`` is therefore the physically interpretable radial
    scale; it is *not* the old vertical parameter ``a = v*r_h``.
    """

    r = np.maximum(np.asarray(radius_m, dtype=float), 0.0)
    rh = float(curvature_radius_m)
    if rh < 0.0:
        raise ValueError("curvature_radius_m cannot be negative")
    slope = math.tan(float(base_angle_rad))
    if rh == 0.0:
        return slope * r
    return slope * (np.sqrt(r * r + rh * rh) - rh)


def flat_blunt_tip_sag_m(
    radius_m: np.ndarray,
    *,
    base_angle_rad: float,
    flat_radius_m: float,
) -> np.ndarray:
    """Continuous flat-centre / conical-outer-surface blunt-tip model."""

    r = np.maximum(np.asarray(radius_m, dtype=float), 0.0)
    rf = float(flat_radius_m)
    if rf < 0.0:
        raise ValueError("flat_radius_m cannot be negative")
    return math.tan(float(base_angle_rad)) * np.maximum(r - rf, 0.0)


def tip_resolution(
    characteristic_radius_m: float,
    grid_dx_m: float,
    *,
    minimum_pixels: float = 12.0,
) -> TipResolution:
    """Resolution contract for a localised radial tip feature on a 2-D grid."""

    radius = float(characteristic_radius_m)
    dx = float(grid_dx_m)
    minimum = float(minimum_pixels)
    if radius < 0.0 or dx <= 0.0 or minimum <= 0.0:
        raise ValueError("invalid tip-resolution inputs")
    pixels = radius / dx
    return TipResolution(
        characteristic_radius_m=radius,
        grid_dx_m=dx,
        radius_pixels=float(pixels),
        minimum_pixels=minimum,
        resolved=bool(radius == 0.0 or pixels >= minimum),
    )


def production_style_tip_phase_rad(
    radius_m: np.ndarray,
    *,
    wavelength_m: float,
    base_angle_rad: float,
    refractive_index: float,
    external_index: float = 1.0,
    tip_model: str = "sharp",
    tip_radius_m: float = 0.0,
) -> np.ndarray:
    """Phase used for shallow-angle 2-D production/reference parity.

    The ideal cone uses the exact normal-incidence Snell transverse wavevector.
    The local tip departure is added as an optical-path defect
    ``-k0 (n_ax-n_ext) [f_tip(r)-f_cone(r)]``.  This hybrid is quantitatively
    permitted only in the shallow-angle regime; high-angle work must use the
    full refractive surface branch.
    """

    r = np.maximum(np.asarray(radius_m, dtype=float), 0.0)
    gamma = float(base_angle_rad)
    n_ax = float(refractive_index)
    n_ext = float(external_index)
    kr = exact_refractive_axicon_kr_m_inv(
        wavelength_m=float(wavelength_m),
        base_angle_rad=gamma,
        refractive_index=n_ax,
        external_index=n_ext,
    )
    cone = math.tan(gamma) * r
    if tip_model == "sharp":
        sag = cone
    elif tip_model == "hyperbolic_round":
        sag = hyperbolic_round_tip_sag_m(
            r,
            base_angle_rad=gamma,
            curvature_radius_m=float(tip_radius_m),
        )
    elif tip_model == "flat_blunt":
        sag = flat_blunt_tip_sag_m(
            r,
            base_angle_rad=gamma,
            flat_radius_m=float(tip_radius_m),
        )
    else:
        raise ValueError(f"unsupported tip model {tip_model!r}")
    k0 = TWOPI / float(wavelength_m)
    return -kr * r - k0 * (n_ax - n_ext) * (sag - cone)


def thin_sag_phase_rad(
    radius_m: np.ndarray,
    *,
    wavelength_m: float,
    base_angle_rad: float,
    refractive_index: float,
    external_index: float = 1.0,
    tip_model: str = "sharp",
    tip_radius_m: float = 0.0,
) -> np.ndarray:
    """Classical thin-element phase directly from the physical sag profile."""

    r = np.maximum(np.asarray(radius_m, dtype=float), 0.0)
    gamma = float(base_angle_rad)
    if tip_model == "sharp":
        sag = math.tan(gamma) * r
    elif tip_model == "hyperbolic_round":
        sag = hyperbolic_round_tip_sag_m(
            r,
            base_angle_rad=gamma,
            curvature_radius_m=float(tip_radius_m),
        )
    elif tip_model == "flat_blunt":
        sag = flat_blunt_tip_sag_m(
            r,
            base_angle_rad=gamma,
            flat_radius_m=float(tip_radius_m),
        )
    else:
        raise ValueError(f"unsupported tip model {tip_model!r}")
    k0 = TWOPI / float(wavelength_m)
    return -k0 * (float(refractive_index) - float(external_index)) * sag


def shallow_exact_phase_gradient_relative_error(
    *,
    base_angle_rad: float,
    refractive_index: float,
    external_index: float = 1.0,
) -> float:
    """Relative cone-gradient error of thin OPD versus exact Snell geometry."""

    geom = snell_axicon_geometry(
        base_angle_rad=float(base_angle_rad),
        refractive_index=float(refractive_index),
        external_index=float(external_index),
    )
    return float(abs(geom.shallow_relative_error))


def radial_fresnel_field(
    *,
    radial_observation_m: Sequence[float],
    z_values_m: Sequence[float],
    wavelength_m: float,
    beam_radius_m: float,
    base_angle_rad: float,
    refractive_index: float,
    external_index: float = 1.0,
    vortex_charge: int = 0,
    tip_model: str = "sharp",
    tip_radius_m: float = 0.0,
    integration_radius_m: float | None = None,
    radial_step_m: float = 0.5e-6,
    phase_model: str = "production_style",
) -> np.ndarray:
    """Axisymmetric paraxial Fresnel reference for Bessel/vortex-Bessel fields.

    The input amplitude is the same 1/e field-radius convention as the system
    route, ``exp[-r^2/w^2]``.  Angular integration is analytic, leaving the
    order-``ell`` Bessel integral in radius.  The returned array has shape
    ``(len(z_values_m), len(radial_observation_m))`` and contains complex field
    amplitudes apart from a common azimuthal factor ``exp(i ell phi)``.
    """

    from scipy.special import jv

    rho = np.maximum(np.asarray(radial_observation_m, dtype=float), 0.0)
    z = np.asarray(z_values_m, dtype=float)
    if rho.ndim != 1 or z.ndim != 1 or np.any(z <= 0.0):
        raise ValueError("rho and positive z must be one-dimensional")
    w = float(beam_radius_m)
    dr = float(radial_step_m)
    if w <= 0.0 or dr <= 0.0:
        raise ValueError("beam radius and radial step must be positive")
    rmax = float(integration_radius_m) if integration_radius_m is not None else 3.5 * w
    if rmax <= 0.0:
        raise ValueError("integration radius must be positive")
    r = np.arange(0.0, rmax + 0.5 * dr, dr, dtype=float)
    if r.size < 1024:
        raise ValueError("radial reference requires at least 1024 integration samples")

    if phase_model == "production_style":
        phase = production_style_tip_phase_rad(
            r,
            wavelength_m=float(wavelength_m),
            base_angle_rad=float(base_angle_rad),
            refractive_index=float(refractive_index),
            external_index=float(external_index),
            tip_model=tip_model,
            tip_radius_m=float(tip_radius_m),
        )
    elif phase_model == "thin_sag":
        phase = thin_sag_phase_rad(
            r,
            wavelength_m=float(wavelength_m),
            base_angle_rad=float(base_angle_rad),
            refractive_index=float(refractive_index),
            external_index=float(external_index),
            tip_model=tip_model,
            tip_radius_m=float(tip_radius_m),
        )
    else:
        raise ValueError("phase_model must be 'production_style' or 'thin_sag'")

    envelope = np.exp(-(r * r) / (w * w)) * np.exp(1j * phase)
    k = TWOPI * float(external_index) / float(wavelength_m)
    ell = int(vortex_charge)
    result = np.empty((z.size, rho.size), dtype=np.complex128)
    for iz, zz in enumerate(z):
        quadratic = np.exp(0.5j * k * r * r / zz)
        if rho.size == 1 and rho[0] == 0.0:
            if ell == 0:
                radial_integral = np.trapezoid(envelope * quadratic * r, r)
                result[iz, 0] = radial_integral / (1j * zz)
            else:
                result[iz, 0] = 0.0j
            continue
        arg = (k / zz) * np.outer(rho, r)
        kernel = jv(abs(ell), arg)
        integrand = kernel * (envelope * quadratic * r)[None, :]
        radial_integral = np.trapezoid(integrand, r, axis=1)
        observation_phase = np.exp(0.5j * k * rho * rho / zz)
        result[iz] = ((-1j) ** abs(ell)) * observation_phase * radial_integral / (1j * zz)
    return result


def normalised_intensity(field: np.ndarray) -> np.ndarray:
    intensity = np.abs(np.asarray(field, dtype=np.complex128)) ** 2
    return intensity / max(float(np.max(intensity)), EPS)


__all__ = [
    "TipResolution",
    "flat_blunt_tip_sag_m",
    "hyperbolic_round_tip_sag_m",
    "normalised_intensity",
    "production_style_tip_phase_rad",
    "radial_fresnel_field",
    "shallow_exact_phase_gradient_relative_error",
    "thin_sag_phase_rad",
    "tip_resolution",
]
