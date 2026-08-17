"""Independent axisymmetric reference for rounded-tip B0 axicon propagation.

The production source-scale route is a two-dimensional angular-spectrum model.
This module intentionally does not call that solver.  It evaluates the scalar
axisymmetric Fresnel/Hankel integral for a Gaussian illuminating a thin axicon,
providing an independent B0 reference for the rounded-tip study.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import j0


EPS = np.finfo(float).tiny


@dataclass(frozen=True)
class RadialReferenceResult:
    rho_m: np.ndarray
    z_m: np.ndarray
    intensity: np.ndarray
    metadata: dict[str, float | int | str]


def hyperboloidal_round_tip_sag_m(
    radius_m: np.ndarray,
    *,
    base_angle_rad: float,
    rounding_parameter_m: float,
) -> np.ndarray:
    """Hyperboloidal shallow-cone sag with a smooth, zero-slope apex.

    h(r) = sqrt(a^2 + [r tan(gamma)]^2) - a.

    It approaches r*tan(gamma) up to an irrelevant additive constant at large r
    and tends continuously to the ideal cone as a -> 0.
    """

    r = np.maximum(np.asarray(radius_m, dtype=float), 0.0)
    slope = math.tan(float(base_angle_rad))
    a = float(rounding_parameter_m)
    if a <= 0.0:
        return r * slope
    return np.sqrt(a * a + (r * slope) ** 2) - a


def axisymmetric_fresnel_field(
    *,
    wavelength_m: float,
    refractive_index: float,
    external_index: float,
    base_angle_rad: float,
    beam_radius_m: float,
    rounding_parameter_m: float,
    rho_values_m: Sequence[float],
    z_values_m: Sequence[float],
    radial_extent_m: float = 5.0e-3,
    radial_samples: int = 32768,
) -> RadialReferenceResult:
    """Axisymmetric thin-axicon Fresnel/Hankel propagation.

    The source immediately after the axicon is

        U0(r) = exp(-r^2/w0^2) exp[-i k0 (n_ax-n_ext) h(r)].

    Cylindrical Fresnel propagation gives, apart from an overall phase,

        U(rho,z) = C/z exp[i k rho^2/(2z)]
                   integral U0(r) exp[i k r^2/(2z)]
                            J0(k r rho / z) r dr.

    Absolute prefactors are retained up to a common source-amplitude constant;
    morphology and relative axial modulation are the intended observables.
    """

    wavelength_m = float(wavelength_m)
    n_ext = float(external_index)
    k_medium = 2.0 * math.pi * n_ext / wavelength_m
    k0 = 2.0 * math.pi / wavelength_m
    r = np.linspace(0.0, float(radial_extent_m), int(radial_samples), dtype=float)
    sag = hyperboloidal_round_tip_sag_m(
        r,
        base_angle_rad=base_angle_rad,
        rounding_parameter_m=rounding_parameter_m,
    )
    source = np.exp(-(r**2) / float(beam_radius_m) ** 2) * np.exp(
        -1j * k0 * (float(refractive_index) - n_ext) * sag
    )

    rho = np.asarray(tuple(rho_values_m), dtype=float)
    z = np.asarray(tuple(z_values_m), dtype=float)
    if np.any(z <= 0.0):
        raise ValueError("axisymmetric Fresnel reference requires z>0")
    field = np.empty((z.size, rho.size), dtype=np.complex128)
    for iz, z_m in enumerate(z):
        common = source * np.exp(1j * k_medium * r**2 / (2.0 * z_m)) * r
        for irho, rho_m in enumerate(rho):
            kernel = j0(k_medium * r * float(rho_m) / z_m)
            integral = np.trapezoid(common * kernel, x=r)
            prefactor = (
                2.0
                * math.pi
                * np.exp(1j * k_medium * z_m)
                * np.exp(1j * k_medium * float(rho_m) ** 2 / (2.0 * z_m))
                / (1j * (wavelength_m / n_ext) * z_m)
            )
            field[iz, irho] = prefactor * integral
    intensity = np.abs(field) ** 2
    return RadialReferenceResult(
        rho_m=rho,
        z_m=z,
        intensity=np.asarray(intensity, dtype=float),
        metadata={
            "method": "independent_axisymmetric_Fresnel_Hankel",
            "radial_samples": int(radial_samples),
            "radial_extent_m": float(radial_extent_m),
            "rounding_parameter_m": float(rounding_parameter_m),
            "not_called": "2D angular-spectrum production solver",
        },
    )


def axisymmetric_on_axis_trace(**kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper returning z and I(rho=0,z)."""

    result = axisymmetric_fresnel_field(rho_values_m=(0.0,), **kwargs)
    return result.z_m, result.intensity[:, 0]


def dominant_axial_ripple_period_m(
    z_m: Sequence[float],
    intensity: Sequence[float],
    *,
    detrend_window_points: int = 101,
) -> float:
    """Estimate the strongest rapid axial period after removing a slow envelope.

    This is a diagnostic, not the analytic Brzobohaty formula.  Uniform z
    sampling is required so the FFT peak has a well-defined spatial frequency.
    """

    z = np.asarray(tuple(z_m), dtype=float)
    values = np.asarray(tuple(intensity), dtype=float)
    if z.size < 32 or values.shape != z.shape:
        raise ValueError("need at least 32 uniformly sampled axial values")
    dz = np.diff(z)
    if not np.allclose(dz, dz[0], rtol=1e-6, atol=1e-15):
        raise ValueError("z sampling must be uniform")
    window = int(detrend_window_points)
    window = max(5, min(window, z.size - (1 - z.size % 2)))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    envelope = np.convolve(values, kernel, mode="same")
    residual = values - envelope
    residual -= float(np.mean(residual))
    spectrum = np.abs(np.fft.rfft(residual)) ** 2
    frequencies = np.fft.rfftfreq(z.size, d=float(dz[0]))
    spectrum[0] = 0.0
    index = int(np.argmax(spectrum))
    if index <= 0 or frequencies[index] <= 0.0:
        return float("inf")
    return float(1.0 / frequencies[index])
