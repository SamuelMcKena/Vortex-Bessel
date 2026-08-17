"""Carrier-tracked rotated angular spectrum for large oblique plane angles.

The standard sampled field cannot represent a many-degree optical carrier on a
10 mm / O(10^3) grid.  This module keeps the absolute carrier wavevector as
metadata and samples only the baseband envelope.  Spectral rotation and the
Matsushima solid-angle Jacobian are evaluated with the *absolute* wavevectors,
so large plane rotations do not require aliasing the carrier onto the FFT grid.

A key large-angle distinction is explicit here: the unweighted spectral L2 norm
``sum(|A|^2)`` is not invariant between differently oriented planes.  With the
rotated-spectrum Jacobian, the conserved scalar power-flux quantity is the
normal spectral flux ``integral |A|^2 f_n df_x df_y``.  At a single plane-wave
carrier, the raw L2 ratio therefore approaches the familiar projection factor
(e.g. cos(theta) in one direction and sec(theta) in the reverse direction),
while the normal-flux ratio remains unity.  Small-angle code could treat the raw
ratio as approximately one only because cos(theta) was numerically close to one.

This is still a scalar tilted-plane coordinate transform.  It is a numerical
building block for an oblique thin-axicon wave model, not a substitute for a
full vector two-surface Fresnel/Snell solver.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from vbb_study.digital_twin.vortex_rotated_plane import (
    DEFAULT_INTERPOLATION_ORDER,
    _spline_uniform_complex,
    rotation_matrix,
)
from vbb_study.equations.fields import fft2c, ifft2c


EPS = np.finfo(float).tiny


def _normal_flux(
    spectrum: np.ndarray,
    normal_frequency_cpm: np.ndarray,
    propagating: np.ndarray,
) -> float:
    """Discrete scalar normal flux, apart from common physical constants."""

    weight = np.where(
        np.asarray(propagating, dtype=bool),
        np.maximum(np.asarray(normal_frequency_cpm, dtype=float), 0.0),
        0.0,
    )
    return float(np.sum(np.abs(np.asarray(spectrum, dtype=np.complex128)) ** 2 * weight))


def rotate_baseband_angular_spectrum(
    baseband_field: np.ndarray,
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    tilt_x_rad: float,
    tilt_y_rad: float,
    source_spectral_center_cpm: tuple[float, float],
    inverse: bool = False,
    interpolation_order: int = DEFAULT_INTERPOLATION_ORDER,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rotate a sampled envelope while tracking its absolute carrier exactly.

    ``spectral_power_ratio`` is retained as the legacy/unweighted spectral L2
    diagnostic.  For appreciable tilt it is *not* expected to equal one.
    ``normal_flux_power_ratio`` is the physically meaningful invariant used for
    large-angle numerical gates.
    """

    envelope = np.asarray(baseband_field, dtype=np.complex128)
    tx = float(tilt_x_rad)
    ty = float(tilt_y_rad)
    fsx, fsy = map(float, source_spectral_center_cpm)
    inv_lam = 1.0 / float(wavelength_m)
    source_center_sq = fsx * fsx + fsy * fsy
    if source_center_sq >= inv_lam * inv_lam:
        raise ValueError("source carrier is non-propagating")
    fsz = math.sqrt(max(inv_lam * inv_lam - source_center_sq, 0.0))
    source_center = np.asarray([fsx, fsy, fsz], dtype=float)

    if tx == 0.0 and ty == 0.0:
        return envelope.copy(), {
            "tilt_x_rad": 0.0,
            "tilt_y_rad": 0.0,
            "inverse": bool(inverse),
            "source_spectral_center_cpm": [fsx, fsy],
            "destination_spectral_center_cpm": [fsx, fsy],
            "spectral_power_ratio": 1.0,
            "normal_flux_power_ratio": 1.0,
            "spectral_clipped_fraction": 0.0,
            "interpolation_model": "identity",
            "carrier_representation": "analytic_unsampled",
            "power_invariant_model": "normal_flux_integral_|A|^2_fnormal",
            "fidelity": "exact_identity_baseband",
        }

    R = rotation_matrix(tx, ty)
    if inverse:
        R = R.T
    destination_center = R.T @ source_center
    fdx = float(destination_center[0])
    fdy = float(destination_center[1])

    foff = np.asarray(grid["FX"], dtype=float)
    goff = np.asarray(grid["FY"], dtype=float)
    axis = np.asarray(grid.get("fx", foff[0]), dtype=float)
    if axis.ndim != 1 or axis.size != foff.shape[1]:
        axis = np.asarray(foff[0], dtype=float)

    spectrum = fft2c(envelope)

    # Normal spectral flux on the regular source-frame grid.
    fx_source_regular = fsx + foff
    fy_source_regular = fsy + goff
    source_regular_sq = fx_source_regular * fx_source_regular + fy_source_regular * fy_source_regular
    source_regular_propagating = source_regular_sq < inv_lam * inv_lam
    fz_source_regular = np.sqrt(
        np.maximum(inv_lam * inv_lam - source_regular_sq, 0.0)
    )

    # Destination-frame regular grid about the analytically tracked carrier.
    fx_destination = fdx + foff
    fy_destination = fdy + goff
    destination_transverse_sq = fx_destination * fx_destination + fy_destination * fy_destination
    propagating_destination = destination_transverse_sq < inv_lam * inv_lam
    fz_destination = np.sqrt(
        np.maximum(inv_lam * inv_lam - destination_transverse_sq, 0.0)
    )

    # Map every destination wavevector back into the source frame and sample the
    # source baseband spectrum at the corresponding carrier-relative offset.
    fx_source = (
        R[0, 0] * fx_destination
        + R[0, 1] * fy_destination
        + R[0, 2] * fz_destination
    )
    fy_source = (
        R[1, 0] * fx_destination
        + R[1, 1] * fy_destination
        + R[1, 2] * fz_destination
    )
    fz_source = (
        R[2, 0] * fx_destination
        + R[2, 1] * fy_destination
        + R[2, 2] * fz_destination
    )
    valid = propagating_destination & (fz_source > 0.0)

    sampled = _spline_uniform_complex(
        spectrum,
        axis,
        fx_source - fsx,
        fy_source - fsy,
        order=int(interpolation_order),
    )

    # The projected spectral area element obeys
    # dfx_source dfy_source / fz_source =
    # dfx_destination dfy_destination / fz_destination.
    # Therefore the field-spectrum transform carries the Jacobian below, while
    # the conserved power quantity is the normal-flux weighted norm.
    jacobian = np.zeros_like(fz_destination)
    jacobian[valid] = np.abs(fz_source[valid]) / np.maximum(fz_destination[valid], EPS)
    rotated = np.where(valid, sampled * jacobian, 0.0)
    output_envelope = ifft2c(rotated)

    source_l2 = float(np.sum(np.abs(spectrum) ** 2))
    destination_l2 = float(np.sum(np.abs(rotated) ** 2))
    l2_ratio = destination_l2 / max(source_l2, EPS)

    source_flux = _normal_flux(
        spectrum,
        fz_source_regular,
        source_regular_propagating,
    )
    destination_flux = _normal_flux(
        rotated,
        fz_destination,
        propagating_destination,
    )
    flux_ratio = destination_flux / max(source_flux, EPS)

    return np.asarray(output_envelope, dtype=np.complex128), {
        "tilt_x_rad": tx,
        "tilt_y_rad": ty,
        "inverse": bool(inverse),
        "source_spectral_center_cpm": [fsx, fsy],
        "destination_spectral_center_cpm": [fdx, fdy],
        "source_spectral_center_z_cpm": float(fsz),
        "destination_spectral_center_z_cpm": float(destination_center[2]),
        "spectral_power_ratio": float(l2_ratio),
        "normal_flux_power_ratio": float(flux_ratio),
        # At finite plane tilt, raw spectral L2 projection is not clipping.
        # Only a deficit in the conserved normal-flux quantity is interpreted as
        # a numerical loss diagnostic.
        "spectral_clipped_fraction": float(max(0.0, 1.0 - flux_ratio)),
        "interpolation_model": f"spline_order_{int(interpolation_order)}",
        "jacobian_model": "abs(fz_source/fz_destination)",
        "power_invariant_model": "normal_flux_integral_|A|^2_fnormal",
        "carrier_representation": "analytic_unsampled",
        "rotation_matrix": R.tolist(),
        "fidelity": "scalar_rotated_angular_spectrum_carrier_tracked_baseband",
    }


__all__ = ["rotate_baseband_angular_spectrum"]
