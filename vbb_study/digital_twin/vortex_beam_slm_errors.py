"""Physical input-beam and SLM error models for the vortex/Bessel system study.

This module is deliberately separate from accepted Phase 2A/2B/2C contracts.
It inserts perturbations at the physical plane where they occur and exposes
calibration-only hooks instead of silently inventing measured SLM behaviour.

Model scope
-----------
* Gaussian beam decentre, pointing, ellipticity and finite wavefront curvature
  are applied before SLM1.
* SLM coordinate misregistration is applied to the commanded hologram itself.
* Phase quantisation, finite phase stroke, phase bias and a measured grey->phase
  LUT can be represented explicitly.
* Static measured phase maps can be added per SLM.
* Fringing-field/crosstalk is represented by a direction-dependent convolution
  surrogate.  Its kernel widths are calibration parameters; non-zero default
  values are never assumed.

The fringing surrogate follows the modelling *strategy* of Lingel, Haist &
Osten, Applied Optics 52, 6877-6883 (2013), who model the direction-dependent
blur of sharp SLM phase edges.  The exact kernel here is not claimed to reproduce
a specific HOLOEYE panel until fitted to measured diffraction data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import gaussian_filter

from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.equations.fields import phase_wrap


EPS = np.finfo(float).tiny
TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class GaussianBeamError:
    """Input-beam state at the SLM1 plane.

    ``radius_x_scale`` and ``radius_y_scale`` multiply the canonical 1/e field-
    amplitude radius.  ``curvature_radius_*_m`` are wavefront radii of curvature;
    ``math.inf`` means collimated.  The sign convention follows
    exp[-i k (u^2/(2 R_x) + v^2/(2 R_y))].
    """

    radius_x_scale: float = 1.0
    radius_y_scale: float = 1.0
    decentre_m: tuple[float, float] = (0.0, 0.0)
    pointing_rad: tuple[float, float] = (0.0, 0.0)
    curvature_radius_x_m: float = math.inf
    curvature_radius_y_m: float = math.inf

    def validate(self) -> None:
        if self.radius_x_scale <= 0.0 or self.radius_y_scale <= 0.0:
            raise ValueError("Gaussian beam radii must be positive")
        sx = math.sin(float(self.pointing_rad[0]))
        sy = math.sin(float(self.pointing_rad[1]))
        if sx * sx + sy * sy >= 1.0:
            raise ValueError("input pointing gives a non-propagating direction")
        for value in (self.curvature_radius_x_m, self.curvature_radius_y_m):
            if np.isfinite(value) and abs(float(value)) < 1e-9:
                raise ValueError("wavefront curvature radius cannot be zero")


@dataclass(frozen=True)
class SLMError:
    """One phase-only SLM error model.

    Registration parameters act on the commanded pattern coordinates.
    ``phase_stroke_scale`` rescales the actual phase swing relative to ideal.
    ``fringing_sigma_*_px`` are phenomenological convolution widths and remain
    zero unless fitted to measured data.
    """

    pattern_offset_m: tuple[float, float] = (0.0, 0.0)
    pattern_rotation_rad: float = 0.0
    pattern_scale_x: float = 1.0
    pattern_scale_y: float = 1.0
    phase_stroke_scale: float = 1.0
    phase_bias_rad: float = 0.0
    phase_levels: int = 256
    fringing_sigma_x_px: float = 0.0
    fringing_sigma_y_px: float = 0.0

    def validate(self) -> None:
        if self.pattern_scale_x <= 0.0 or self.pattern_scale_y <= 0.0:
            raise ValueError("SLM pattern scales must be positive")
        if self.phase_stroke_scale <= 0.0:
            raise ValueError("phase_stroke_scale must be positive")
        if int(self.phase_levels) < 2:
            raise ValueError("phase_levels must be >= 2")
        if self.fringing_sigma_x_px < 0.0 or self.fringing_sigma_y_px < 0.0:
            raise ValueError("fringing kernel widths cannot be negative")


def _beam_frame(pointing_rad: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return orthonormal (e1,e2,s) for the tilted Gaussian beam."""

    tx, ty = map(float, pointing_rad)
    sx = math.sin(tx)
    sy = math.sin(ty)
    sz = math.sqrt(max(0.0, 1.0 - sx * sx - sy * sy))
    s = np.asarray([sx, sy, sz], dtype=float)
    trial = np.asarray([1.0, 0.0, 0.0], dtype=float)
    e1 = trial - float(np.dot(trial, s)) * s
    if np.linalg.norm(e1) < 1e-12:
        trial = np.asarray([0.0, 1.0, 0.0], dtype=float)
        e1 = trial - float(np.dot(trial, s)) * s
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(s, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2, s


def gaussian_input_field(
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    canonical_radius_m: float,
    error: GaussianBeamError = GaussianBeamError(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return an obliquely incident astigmatic/elliptical Gaussian at z=0.

    The Gaussian envelope is evaluated in a beam-fixed orthonormal transverse
    basis, so finite pointing angles project the beam onto the laboratory SLM
    plane instead of imposing an artificial angle-independent circular footprint.
    """

    error.validate()
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    dx0, dy0 = map(float, error.decentre_m)
    rx = X - dx0
    ry = Y - dy0
    rz = np.zeros_like(rx)
    e1, e2, s = _beam_frame(error.pointing_rad)

    u = e1[0] * rx + e1[1] * ry + e1[2] * rz
    v = e2[0] * rx + e2[1] * ry + e2[2] * rz
    wx = float(canonical_radius_m) * float(error.radius_x_scale)
    wy = float(canonical_radius_m) * float(error.radius_y_scale)
    amplitude = np.exp(-(u * u / (wx * wx) + v * v / (wy * wy)))

    k0 = TWOPI / float(wavelength_m)
    plane_phase = np.exp(1j * k0 * (s[0] * X + s[1] * Y))
    curvature_phase = np.ones_like(amplitude, dtype=np.complex128)
    if np.isfinite(error.curvature_radius_x_m):
        curvature_phase *= np.exp(-1j * k0 * u * u / (2.0 * float(error.curvature_radius_x_m)))
    if np.isfinite(error.curvature_radius_y_m):
        curvature_phase *= np.exp(-1j * k0 * v * v / (2.0 * float(error.curvature_radius_y_m)))

    field = np.asarray(amplitude * plane_phase * curvature_phase, dtype=np.complex128)
    return field, {
        "beam_radius_x_m": wx,
        "beam_radius_y_m": wy,
        "beam_decentre_m": (dx0, dy0),
        "beam_pointing_rad": tuple(map(float, error.pointing_rad)),
        "beam_direction_cosines": tuple(float(vv) for vv in s),
        "beam_curvature_radius_x_m": float(error.curvature_radius_x_m),
        "beam_curvature_radius_y_m": float(error.curvature_radius_y_m),
        "beam_footprint_model": "beam-fixed elliptical Gaussian projected onto SLM plane",
    }


def transformed_pattern_coordinates(
    grid: Mapping[str, Any], error: SLMError
) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates in which the commanded hologram is evaluated."""

    error.validate()
    X = np.asarray(grid["X"], dtype=float) - float(error.pattern_offset_m[0])
    Y = np.asarray(grid["Y"], dtype=float) - float(error.pattern_offset_m[1])
    c = math.cos(float(error.pattern_rotation_rad))
    s = math.sin(float(error.pattern_rotation_rad))
    xr = c * X + s * Y
    yr = -s * X + c * Y
    return xr / float(error.pattern_scale_x), yr / float(error.pattern_scale_y)


def quantise_commanded_phase(phase_rad: np.ndarray, levels: int) -> np.ndarray:
    levels = int(levels)
    wrapped = phase_wrap(np.asarray(phase_rad, dtype=float))
    idx = np.floor(wrapped / TWOPI * levels + 0.5).astype(np.int64) % levels
    return idx.astype(float) * TWOPI / float(levels)


def phase_from_lut(commanded_phase_rad: np.ndarray, lut_phase_rad: np.ndarray) -> np.ndarray:
    """Map a wrapped command to a supplied measured grey->phase LUT."""

    lut = np.asarray(lut_phase_rad, dtype=float).ravel()
    if lut.size < 2:
        raise ValueError("phase LUT must contain at least two entries")
    wrapped = phase_wrap(np.asarray(commanded_phase_rad, dtype=float))
    idx = np.floor(wrapped / TWOPI * lut.size + 0.5).astype(np.int64) % lut.size
    return lut[idx]


def apply_fringing_surrogate(
    phase_rad: np.ndarray,
    grid: Mapping[str, Any],
    *,
    pixel_pitch_m: float,
    sigma_x_px: float,
    sigma_y_px: float,
) -> np.ndarray:
    """Direction-dependent convolution surrogate for LC fringing-field blur.

    Lingel et al. model fringing as a direction-dependent convolution of the SLM
    phase response.  Here the wrapped command is first locally unwrapped along
    both computational axes, convolved with an anisotropic Gaussian kernel, then
    wrapped again.  This correctly turns a commanded sharp phase edge into a
    finite-width phase transition, unlike complex-phasor averaging which can
    leave a 0/pi edge discontinuous.

    The Gaussian kernel is deliberately a *fit surrogate*: its x/y widths must
    be calibrated from the actual panel (for example from measured grating
    diffraction efficiencies).  It is not a claimed manufacturer model.
    """

    if sigma_x_px == 0.0 and sigma_y_px == 0.0:
        return np.asarray(phase_rad, dtype=float)
    dx = float(grid["dx"])
    sigma_x_samples = float(sigma_x_px) * float(pixel_pitch_m) / dx
    sigma_y_samples = float(sigma_y_px) * float(pixel_pitch_m) / dx
    phase = np.asarray(phase_rad, dtype=float)
    unwrapped = np.unwrap(np.unwrap(phase, axis=1), axis=0)
    blurred = gaussian_filter(
        unwrapped,
        sigma=(sigma_y_samples, sigma_x_samples),
        mode="nearest",
    )
    return phase_wrap(np.asarray(blurred, dtype=float))


def actual_slm_phase(
    commanded_phase_rad: np.ndarray,
    grid: Mapping[str, Any],
    *,
    error: SLMError,
    pixel_pitch_m: float,
    lut_phase_rad: np.ndarray | None = None,
    static_phase_map_rad: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert an ideal phase command into the phase physically applied."""

    error.validate()
    quantised = quantise_commanded_phase(commanded_phase_rad, error.phase_levels)
    if lut_phase_rad is None:
        actual = quantised * float(error.phase_stroke_scale) + float(error.phase_bias_rad)
        lut_status = "identity_unmeasured"
    else:
        actual = phase_from_lut(quantised, np.asarray(lut_phase_rad, dtype=float))
        actual = actual * float(error.phase_stroke_scale) + float(error.phase_bias_rad)
        lut_status = "measured_or_user_supplied"

    actual = apply_fringing_surrogate(
        actual,
        grid,
        pixel_pitch_m=float(pixel_pitch_m),
        sigma_x_px=float(error.fringing_sigma_x_px),
        sigma_y_px=float(error.fringing_sigma_y_px),
    )
    if static_phase_map_rad is not None:
        correction = np.asarray(static_phase_map_rad, dtype=float)
        if correction.shape != actual.shape:
            raise ValueError("static SLM phase map shape does not match simulation grid")
        actual = actual + correction
        static_status = "measured_or_user_supplied"
    else:
        static_status = "none"

    return np.asarray(actual, dtype=float), {
        "phase_levels": int(error.phase_levels),
        "phase_stroke_scale": float(error.phase_stroke_scale),
        "phase_bias_rad": float(error.phase_bias_rad),
        "fringing_sigma_x_px": float(error.fringing_sigma_x_px),
        "fringing_sigma_y_px": float(error.fringing_sigma_y_px),
        "phase_lut_status": lut_status,
        "static_phase_map_status": static_status,
        "fringing_fidelity": (
            "calibration_required_direction_dependent_phase_convolution_surrogate"
            if (error.fringing_sigma_x_px or error.fringing_sigma_y_px)
            else "disabled"
        ),
    }


def canonical_beam_parameters() -> dict[str, float]:
    manifest = canonical_hardware_manifest()
    return {
        "wavelength_m": float(hardware_value(manifest, "wavelength_m")),
        "beam_radius_m": float(hardware_value(manifest, "beam_radius_on_slm_m")),
        "slm_pixel_pitch_m": float(hardware_value(manifest, "slm_pixel_pitch_m")),
    }
