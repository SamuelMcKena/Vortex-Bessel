"""The bench 4F relay between the SLMs and the axicon.

Operator-reported layout (2026-09-17)::

    SLM --300mm--> L1 (f=300) --300mm--> aperture --300mm--> L2 (f=300) --300mm--> axicon

Every spacing equals a focal length, so this is a *symmetric* 4F: the axicon
plane is an image of the SLM plane at unit magnification, rotated by 180
degrees, and the aperture sits in the shared focal (Fourier) plane, where it
selects the +1 diffraction order of the blazed mask.

Modelled here
    * unit magnification and the 180-degree rotation (exact for an ideal 4F);
    * the aperture as a circular low-pass filter of the SLM field, centred on
      the selected order.  A stop of diameter D at focal length f passes SLM
      spatial frequencies up to D / (2 lambda f): a small aperture cleans the
      order but also smooths the mask's finest structure.

Not modelled
    * lens aberrations, the finite lens diameters and any wedge/tilt;
    * the aperture's exact axial position and centring;
    * the residual light of the unselected orders leaking past the stop.

The physical aperture diameter has not been measured, so the default geometry
applies no filtering and says so in its metadata.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def fourier_plane_cutoff_cycles_per_m(
    *, diameter_m: float, wavelength_m: float, focal_length_m: float
) -> float:
    """Highest SLM spatial frequency that passes a stop of this diameter.

    A point at radius rho in the shared focal plane carries the SLM spatial
    frequency nu = rho / (lambda f), so the stop's edge sets nu_max.
    """

    for name, value in (("diameter", diameter_m), ("wavelength", wavelength_m), ("focal length", focal_length_m)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"4F {name} must be finite and positive.")
    return 0.5 * diameter_m / (wavelength_m * focal_length_m)


def fourier_aperture_report(
    *, diameter_mm: float, wavelength_nm: float, focal_length_mm: float, window_mm: float, grid_n: int
) -> dict[str, Any]:
    """What a given stop does, in numbers a person can check against the bench."""

    cutoff = fourier_plane_cutoff_cycles_per_m(
        diameter_m=float(diameter_mm) * 1e-3,
        wavelength_m=float(wavelength_nm) * 1e-9,
        focal_length_m=float(focal_length_mm) * 1e-3,
    )
    # Half a period of the highest transmitted frequency: the smallest mask
    # detail that survives the relay.
    feature_um = 0.5 / cutoff * 1e6
    frequency_step = 1.0 / (float(window_mm) * 1e-3)
    return {
        "cutoff_cycles_per_mm": cutoff * 1e-3,
        "smallest_relayed_feature_um": feature_um,
        "aperture_samples_across": 2.0 * cutoff / frequency_step,
        "grid_max_frequency_cycles_per_mm": 0.5 * int(grid_n) * frequency_step * 1e-3,
    }


def apply_fourier_plane_aperture(
    field: np.ndarray,
    *,
    dx_m: float,
    wavelength_m: float,
    focal_length_m: float,
    diameter_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Low-pass the relayed field with the stop in the shared focal plane."""

    cutoff = fourier_plane_cutoff_cycles_per_m(
        diameter_m=diameter_m, wavelength_m=wavelength_m, focal_length_m=focal_length_m
    )
    array = np.asarray(field)
    n = int(array.shape[0])
    frequency = np.fft.fftfreq(n, d=float(dx_m))
    radius_squared = frequency[:, None] ** 2 + frequency[None, :] ** 2
    passband = radius_squared <= cutoff * cutoff
    spectrum = np.fft.fft2(array)
    incident = float((np.abs(spectrum).astype(np.float64) ** 2).sum())
    spectrum *= passband
    transmitted = float((np.abs(spectrum).astype(np.float64) ** 2).sum())
    filtered = np.fft.ifft2(spectrum)
    samples_across = 2.0 * cutoff * n * float(dx_m)
    meta: dict[str, Any] = {
        "fourier_aperture_diameter_mm": diameter_m * 1e3,
        "fourier_aperture_focal_length_mm": focal_length_m * 1e3,
        "fourier_aperture_cutoff_cycles_per_mm": cutoff * 1e-3,
        "fourier_aperture_smallest_relayed_feature_um": 0.5 / cutoff * 1e6,
        "fourier_aperture_power_fraction": transmitted / incident if incident > 0.0 else 1.0,
        "fourier_aperture_samples_across": samples_across,
    }
    if samples_across < 8.0:
        meta["fourier_aperture_warning"] = (
            "The stop spans fewer than 8 samples of this grid's spectrum: its effect is "
            "under-resolved. Use a finer model grid before trusting this number."
        )
    return filtered, meta


def rotate_180(field: np.ndarray) -> np.ndarray:
    """Map (x, y) to (-x, -y): the image rotation of a symmetric 4F.

    The centred grids used here satisfy x[i] = -x[N-1-i] for even N, so
    reversing both axes is exact, not an interpolation.
    """

    return np.ascontiguousarray(np.asarray(field)[::-1, ::-1])
