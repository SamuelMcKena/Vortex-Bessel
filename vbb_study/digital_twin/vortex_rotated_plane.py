"""Scalar rotated-angular-spectrum utilities for rigidly tilted optical planes.

The implementation follows the coordinate-rotation idea of Matsushima,
Schimmel & Wyrowski, JOSA A 20, 1755-1762 (2003): the plane-wave spectrum is
rotated in Fourier space and resampled on the regular spectral grid of the new
plane. The Jacobian ``|fz/fz'|`` is included from invariance of solid angle.

The physical 4F route can carry a non-zero diffraction-order carrier. Resampling
that off-axis spectrum directly on an origin-centred FFT grid creates a strong,
axis-dependent interpolation artefact. The production path therefore recentres
each input spectrum about its measured spectral centroid, rotates the absolute
wavevector coordinates, resamples the baseband envelope, and restores the
rotated carrier in the destination plane. This is a numerical coordinate change;
it does not remove or fabricate physical beam steering.

A previous research implementation used bilinear spectral interpolation. That
was adequate to expose morphology changes but introduced unphysical numerical
power loss in repeated tilted-plane transforms. Cubic-spline interpolation is
the default, with explicit spectral-power bookkeeping and validation gates.

This remains a scalar propagating-wave model. It does not replace vector
Fresnel/Snell treatment at a strongly tilted refractive interface.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import map_coordinates

from vbb_study.equations.fields import fft2c, ifft2c


EPS = np.finfo(float).tiny
TWOPI = 2.0 * np.pi
DEFAULT_INTERPOLATION_ORDER = 3


def rotation_matrix(tilt_x_rad: float, tilt_y_rad: float) -> np.ndarray:
    """Return R mapping coordinates in the tilted plane frame into lab frame."""

    tx = float(tilt_x_rad)
    ty = float(tilt_y_rad)
    cx, sx = math.cos(tx), math.sin(tx)
    cy, sy = math.cos(ty), math.sin(ty)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    ry = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    return ry @ rx


def _spline_uniform_complex(
    data: np.ndarray,
    coordinate: np.ndarray,
    xq: np.ndarray,
    yq: np.ndarray,
    *,
    order: int = DEFAULT_INTERPOLATION_ORDER,
) -> np.ndarray:
    """Spline-resample complex data on a square, uniform spectral grid.

    Real and imaginary components are interpolated independently. Cubic
    interpolation is the default because the old bilinear map was found to
    attenuate smooth spectra significantly during forward/inverse plane
    rotations. Samples outside the represented spectrum are set to zero rather
    than extrapolated.
    """

    arr = np.asarray(data, dtype=np.complex128)
    c = np.asarray(coordinate, dtype=float)
    if c.ndim != 1 or arr.shape != (c.size, c.size):
        raise ValueError("expected square data on one shared coordinate axis")
    if c.size < 4:
        raise ValueError("spectral interpolation requires at least four samples")
    step = float(c[1] - c[0])
    if not np.allclose(
        np.diff(c),
        step,
        rtol=1e-9,
        atol=1e-12 * max(1.0, abs(step)),
    ):
        raise ValueError("spectral coordinate must be uniform")
    order_i = int(order)
    if order_i < 1 or order_i > 5:
        raise ValueError("spline interpolation order must lie in [1, 5]")

    ix = (np.asarray(xq, dtype=float) - c[0]) / step
    iy = (np.asarray(yq, dtype=float) - c[0]) / step
    query = np.vstack([iy.ravel(), ix.ravel()])

    real = map_coordinates(
        np.real(arr),
        query,
        order=order_i,
        mode="constant",
        cval=0.0,
        prefilter=order_i > 1,
    ).reshape(ix.shape)
    imag = map_coordinates(
        np.imag(arr),
        query,
        order=order_i,
        mode="constant",
        cval=0.0,
        prefilter=order_i > 1,
    ).reshape(ix.shape)
    return np.asarray(real + 1j * imag, dtype=np.complex128)


def spectral_centroid_cpm(
    field: np.ndarray,
    grid: Mapping[str, Any],
) -> tuple[float, float]:
    """Return the intensity-weighted angular-spectrum centroid in cycles/metre."""

    spectrum = fft2c(np.asarray(field, dtype=np.complex128))
    weight = np.abs(spectrum) ** 2
    total = float(np.sum(weight))
    if total <= EPS:
        return 0.0, 0.0
    fx = np.asarray(grid["FX"], dtype=float)
    fy = np.asarray(grid["FY"], dtype=float)
    if fx.shape != weight.shape or fy.shape != weight.shape:
        raise ValueError("spectral grid shape does not match field")
    return (
        float(np.sum(weight * fx) / total),
        float(np.sum(weight * fy) / total),
    )


def rotate_angular_spectrum(
    field: np.ndarray,
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    tilt_x_rad: float,
    tilt_y_rad: float,
    inverse: bool = False,
    interpolation_order: int = DEFAULT_INTERPOLATION_ORDER,
    spectral_center_cpm: tuple[float, float] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resample a scalar angular spectrum between lab and tilted planes.

    The two planes intersect at their origins. ``inverse=False`` maps a field on
    the lab plane onto the tilted plane; ``inverse=True`` performs the inverse
    coordinate rotation.

    For a non-zero tilt the spectrum is first shifted to baseband around either
    ``spectral_center_cpm`` or, by default, its measured power centroid. The
    absolute wavevector of that centre is rotated exactly, interpolation is
    performed only on spectral offsets around the centre, and the destination
    carrier is restored after the inverse FFT. This avoids interpolation error
    caused solely by an off-axis carrier while preserving the physical angular
    spectrum.

    ``spectral_power_ratio`` remains a numerical diagnostic, not an optical
    transmission coefficient. With a sufficiently represented spectrum it
    should remain close to unity for a rigid coordinate transformation.
    """

    arr = np.asarray(field, dtype=np.complex128)
    tx = float(tilt_x_rad)
    ty = float(tilt_y_rad)
    if tx == 0.0 and ty == 0.0:
        return arr.copy(), {
            "tilt_x_rad": 0.0,
            "tilt_y_rad": 0.0,
            "inverse": bool(inverse),
            "spectral_clipped_fraction": 0.0,
            "spectral_power_ratio": 1.0,
            "interpolation_model": "identity",
            "jacobian_model": "identity",
            "spectral_center_model": "identity",
            "source_spectral_center_cpm": None,
            "destination_spectral_center_cpm": None,
            "fidelity": "exact_identity_parallel_plane",
        }

    R = rotation_matrix(tx, ty)
    if inverse:
        R = R.T

    foff = np.asarray(grid["FX"], dtype=float)
    goff = np.asarray(grid["FY"], dtype=float)
    axis = np.asarray(grid.get("fx", foff[0]), dtype=float)
    if axis.ndim != 1 or axis.size != foff.shape[1]:
        axis = np.asarray(foff[0], dtype=float)

    if spectral_center_cpm is None:
        fsx, fsy = spectral_centroid_cpm(arr, grid)
        center_model = "intensity_weighted_spectral_centroid"
    else:
        fsx, fsy = map(float, spectral_center_cpm)
        center_model = "user_supplied_absolute_spectral_center"

    inv_lam = 1.0 / float(wavelength_m)
    source_center_sq = fsx * fsx + fsy * fsy
    if source_center_sq >= inv_lam * inv_lam:
        raise ValueError("spectral centre is non-propagating at the supplied wavelength")
    fsz = math.sqrt(max(inv_lam * inv_lam - source_center_sq, 0.0))
    source_center = np.asarray([fsx, fsy, fsz], dtype=float)

    # R maps destination-frame wavevectors into the source frame. Therefore the
    # corresponding destination carrier is R.T @ source_center.
    destination_center = R.T @ source_center
    fdx = float(destination_center[0])
    fdy = float(destination_center[1])

    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    baseband_field = arr * np.exp(-1j * TWOPI * (fsx * X + fsy * Y))
    spectrum = fft2c(baseband_field)

    # The regular FFT coordinates now describe offsets around the destination
    # carrier. Convert them to absolute wavevectors, rotate those into the source
    # frame, then subtract the source carrier before interpolation of the
    # baseband source spectrum.
    fx_destination = fdx + foff
    fy_destination = fdy + goff
    transverse_sq_destination = fx_destination * fx_destination + fy_destination * fy_destination
    propagating_destination = transverse_sq_destination < inv_lam * inv_lam
    fz_destination = np.sqrt(
        np.maximum(inv_lam * inv_lam - transverse_sq_destination, 0.0)
    )

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
    source_propagating = fz_source > 0.0
    valid = propagating_destination & source_propagating

    sampled = _spline_uniform_complex(
        spectrum,
        axis,
        fx_source - fsx,
        fy_source - fsy,
        order=int(interpolation_order),
    )

    # dfx dfy / fz is invariant under rotation of the wavevector sphere.
    jacobian = np.zeros_like(fz_destination)
    jacobian[valid] = np.abs(fz_source[valid]) / np.maximum(
        fz_destination[valid], EPS
    )
    rotated = np.where(valid, sampled * jacobian, 0.0)
    envelope = ifft2c(rotated)
    output = envelope * np.exp(1j * TWOPI * (fdx * X + fdy * Y))

    source_power = float(np.sum(np.abs(spectrum) ** 2))
    rotated_power = float(np.sum(np.abs(rotated) ** 2))
    power_ratio = rotated_power / max(source_power, EPS)
    interpolation_model = f"spline_order_{int(interpolation_order)}"
    return np.asarray(output, dtype=np.complex128), {
        "tilt_x_rad": tx,
        "tilt_y_rad": ty,
        "inverse": bool(inverse),
        # Backwards-compatible field: this is a numerical spectral-power
        # deficit diagnostic, not necessarily true physical clipping.
        "spectral_clipped_fraction": float(max(0.0, 1.0 - power_ratio)),
        "spectral_power_ratio": float(power_ratio),
        "interpolation_model": interpolation_model,
        "jacobian_model": "abs(fz_source/fz_destination)",
        "spectral_center_model": center_model,
        "source_spectral_center_cpm": [float(fsx), float(fsy)],
        "destination_spectral_center_cpm": [float(fdx), float(fdy)],
        "rotation_matrix": R.tolist(),
        "fidelity": (
            "scalar_rotated_angular_spectrum_propagating_components_"
            f"{interpolation_model}"
        ),
    }


def lab_to_tilted_plane(
    field: np.ndarray,
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    tilt_x_rad: float,
    tilt_y_rad: float,
    interpolation_order: int = DEFAULT_INTERPOLATION_ORDER,
    spectral_center_cpm: tuple[float, float] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    return rotate_angular_spectrum(
        field,
        grid,
        wavelength_m=wavelength_m,
        tilt_x_rad=tilt_x_rad,
        tilt_y_rad=tilt_y_rad,
        inverse=False,
        interpolation_order=interpolation_order,
        spectral_center_cpm=spectral_center_cpm,
    )


def tilted_to_lab_plane(
    field: np.ndarray,
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    tilt_x_rad: float,
    tilt_y_rad: float,
    interpolation_order: int = DEFAULT_INTERPOLATION_ORDER,
    spectral_center_cpm: tuple[float, float] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    return rotate_angular_spectrum(
        field,
        grid,
        wavelength_m=wavelength_m,
        tilt_x_rad=tilt_x_rad,
        tilt_y_rad=tilt_y_rad,
        inverse=True,
        interpolation_order=interpolation_order,
        spectral_center_cpm=spectral_center_cpm,
    )


__all__ = [
    "DEFAULT_INTERPOLATION_ORDER",
    "lab_to_tilted_plane",
    "rotate_angular_spectrum",
    "rotation_matrix",
    "spectral_centroid_cpm",
    "tilted_to_lab_plane",
]
