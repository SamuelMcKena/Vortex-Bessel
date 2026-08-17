"""Objective-pupil and first-order-filter equations.

This module owns the compact scalar formulas for:

* objective pupil radius from NA and focal length,
* focal-plane pixel size from SLM pixel pitch and demagnification,
* SLM-plane to pupil-plane magnification in a 4f relay,
* Fourier-plane ring radius for a Bessel field (where the filter sits),
* Gaussian beam fill fraction through a hard pupil aperture,
* first-order filter inner and outer radius estimates.

What this module does NOT own:

* the full propagation pipeline from SLM to focal plane
  (``bessel_twin_core`` owns that),
* the ``ObjectiveMap`` coordinate-plane bookkeeping
  (``vbb_study.vbb_planes`` owns that),
* aberration phase maps
  (``equations.interface`` owns the low-order formulae).

Coordinate convention:
  All radii are at the plane implied by the argument name.
  ``f_eff_m`` is the effective focal length of the objective at the
  design tube-lens spacing, not the nominal manufacturer value.

Phase-2K note on Fourier-plane distances:
  ``x = lambda*f*nu`` is the paraxial Fourier-optics coordinate used by the
  FFT/Fresnel 4f model.  It is not silently called an exact high-angle ray
  position.  Exact geometric comparison helpers based on ``sin(theta)`` and
  ``x=f*tan(theta)`` are provided below so the paraxial approximation can be
  bounded for every generated output family.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import scipy.special as sp

from vbb_study import vbb_planes

EPS = 1.0e-30


# ---------------------------------------------------------------------------
# Pupil size and pixel scale
# ---------------------------------------------------------------------------


def pupil_radius_m(f_eff_m: float, NA: float, n_medium: float = 1.0) -> float:
    """Return the objective back-pupil radius in metres.

    Under the aplanatic sine-condition convention,
    ``R_pupil = f_eff * sin(theta_max) = f_eff * NA / n``.
    """

    return float(f_eff_m) * float(NA) / max(float(n_medium), EPS)


def focal_plane_pixel_size_m(slm_pixel_pitch_m: float, demag: float) -> float:
    """Return the equivalent sample-plane pixel pitch after transverse demagnification."""

    return float(slm_pixel_pitch_m) * abs(float(demag))


def slm_to_pupil_magnification(f_relay_m: float, f_tube_m: float) -> float:
    """Return the transverse 4f relay magnification ``f_tube/f_relay``."""

    return float(f_tube_m) / max(float(f_relay_m), EPS)


# ---------------------------------------------------------------------------
# Fourier-plane geometry (where the first-order filter lives)
# ---------------------------------------------------------------------------


def fourier_plane_ring_radius_m(
    kr_m_inv: float,
    f_lens_m: float,
    wavelength_m: float,
) -> float:
    """Return the **paraxial Fourier-coordinate** Bessel-ring radius.

    A Fresnel/Fourier-transform lens maps transverse spatial frequency
    ``nu_r=k_r/(2*pi)`` to

    ``r_paraxial = lambda*f*nu_r = lambda*f*k_r/(2*pi)``.

    This is the correct coordinate for the repository's paraxial FFT 4f model.
    It is not an exact high-angle ray-intersection law.  Use
    :func:`fourier_plane_ring_radius_exact_geometric_m` to quantify the
    approximation whenever the support is not strongly paraxial.
    """

    return float(wavelength_m) * float(kr_m_inv) * float(f_lens_m) / (2.0 * math.pi)


def fourier_plane_ring_radius_exact_geometric_m(
    kr_m_inv: float,
    f_lens_m: float,
    wavelength_m: float,
    *,
    n_medium: float = 1.0,
) -> float:
    """Return the exact geometric ray intersection ``f*tan(theta)``.

    The transverse wavevector obeys
    ``k_r = k0*n_medium*sin(theta)``.  This helper is a reference/bound on the
    paraxial Fourier coordinate, not a replacement for a full nonparaxial lens
    diffraction model.
    """

    k = 2.0 * math.pi * float(n_medium) / max(float(wavelength_m), EPS)
    ratio = float(kr_m_inv) / max(k, EPS)
    if abs(ratio) > 1.0 + 1.0e-12:
        raise ValueError("|k_r| exceeds the propagation-medium wavenumber")
    theta = math.asin(float(np.clip(ratio, -1.0, 1.0)))
    return float(f_lens_m) * math.tan(theta)


def fourier_plane_carrier_separation_m(
    carrier_cpm: float,
    f_lens_m: float,
    wavelength_m: float,
) -> float:
    """Return the **paraxial Fourier-coordinate** carrier separation.

    ``x_paraxial = lambda*f*carrier_cpm`` for a grating spatial frequency in
    cycles/m.  Use :func:`fourier_plane_carrier_separation_exact_geometric_m`
    to bound the small-angle approximation for a physical lens plane.
    """

    return float(wavelength_m) * float(carrier_cpm) * float(f_lens_m)


def fourier_plane_carrier_separation_exact_geometric_m(
    carrier_cpm: float,
    f_lens_m: float,
    wavelength_m: float,
    *,
    n_medium: float = 1.0,
) -> float:
    """Return exact geometric separation from ``sin(theta)=lambda0*nu/n``."""

    ratio = float(wavelength_m) * float(carrier_cpm) / max(float(n_medium), EPS)
    if abs(ratio) > 1.0 + 1.0e-12:
        raise ValueError("carrier spatial frequency is not a propagating diffraction order")
    theta = math.asin(float(np.clip(ratio, -1.0, 1.0)))
    return float(f_lens_m) * math.tan(theta)


def paraxial_fourier_relative_error(
    paraxial_distance_m: float,
    exact_geometric_distance_m: float,
) -> float:
    """Return ``(paraxial-exact)/exact`` for an auditable approximation bound."""

    exact = float(exact_geometric_distance_m)
    return (float(paraxial_distance_m) - exact) / max(abs(exact), EPS)


# ---------------------------------------------------------------------------
# First-order filter radius estimates
# ---------------------------------------------------------------------------


def first_order_filter_inner_radius_m(
    carrier_cpm: float,
    f_lens_m: float,
    kr_m_inv: float,
    wavelength_m: float,
    *,
    safety_factor: float = 0.5,
) -> float:
    """Estimate the minimum safe inner radius in the paraxial Fourier plane."""

    ring_r = fourier_plane_ring_radius_m(kr_m_inv, f_lens_m, wavelength_m)
    carrier_shift = fourier_plane_carrier_separation_m(carrier_cpm, f_lens_m, wavelength_m)
    inner = max(carrier_shift - ring_r * (1.0 + float(safety_factor)), 0.0)
    return float(inner)


def first_order_filter_outer_radius_m(
    carrier_cpm: float,
    f_lens_m: float,
    kr_m_inv: float,
    wavelength_m: float,
    *,
    safety_factor: float = 0.5,
) -> float:
    """Estimate the maximum safe outer radius in the paraxial Fourier plane."""

    ring_r = fourier_plane_ring_radius_m(kr_m_inv, f_lens_m, wavelength_m)
    carrier_shift = fourier_plane_carrier_separation_m(carrier_cpm, f_lens_m, wavelength_m)
    outer = carrier_shift + ring_r * (1.0 + float(safety_factor))
    outer = min(outer, 2.0 * carrier_shift - ring_r * float(safety_factor))
    return float(max(outer, ring_r))


# ---------------------------------------------------------------------------
# Gaussian beam fill and clipping
# ---------------------------------------------------------------------------


def gaussian_pupil_fill_fraction(beam_1e_radius_m: float, pupil_radius_m_val: float) -> float:
    """Return Gaussian power inside a circular pupil.

    For a ``1/e`` field-amplitude radius ``w``, intensity is
    ``I(r) proportional exp(-2 r^2/w^2)`` and the enclosed fraction is
    ``1-exp(-2 R^2/w^2)``.
    """

    w = max(float(beam_1e_radius_m), EPS)
    R = float(pupil_radius_m_val)
    return float(1.0 - math.exp(-2.0 * R**2 / w**2))


def gaussian_clipping_power_fraction(beam_1e_radius_m: float, pupil_radius_m_val: float) -> float:
    """Return the Gaussian power fraction clipped at a circular pupil."""

    return 1.0 - gaussian_pupil_fill_fraction(beam_1e_radius_m, pupil_radius_m_val)


def pupil_fill_ratio(beam_1e_radius_m: float, pupil_radius_m_val: float) -> float:
    """Return the dimensionless fill ratio ``w/R_pupil``."""

    return float(beam_1e_radius_m) / max(float(pupil_radius_m_val), EPS)


def objective_map_from_design_inputs(
    laser: Any,
    target: Any,
    material: Any,
    beam_radius_on_slm_m: float | None = None,
):
    """Return the explicit target-matched inverse-design transverse map.

    The requested sample BG reference length determines the required sample
    Gaussian radius; the ratio to the declared SLM beam radius then defines the
    **required** demagnification.  This is an inverse-design feasibility map,
    not a measured relay calibration.
    """

    D = max(float(target.target_core_diameter_m), EPS)
    L = max(float(target.target_bessel_length_m), EPS)
    w_slm = float(laser.beam_radius_on_slm_m if beam_radius_on_slm_m is None else beam_radius_on_slm_m)
    k_medium = laser.k0 * float(material.refractive_index)
    kr_sample = 2.0 * float(sp.jn_zeros(0, 1)[0]) / D
    w0_sample = L * kr_sample / max(k_medium, EPS)
    return vbb_planes.objective_map_from_waists(
        pre_objective_radius_m=w_slm,
        sample_radius_m=w0_sample,
        n_sample=float(material.refractive_index),
        source="compute_design_from_targets:w0_sample/beam_radius_on_slm",
    )


def headline_length_tags(config: Any) -> dict[str, Any]:
    """Return JSON-friendly plane tags for headline length quantities."""

    return dict(vbb_planes.headline_lengths_jsonable(config))
