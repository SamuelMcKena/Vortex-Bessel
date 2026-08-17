"""Measured-LUT SLM phase preparation for the physical dual-PLUTO bench.

This module is deliberately stricter than the generic simulation SLM model.
A numerical wrapped phase is not called a hardware mask until a wavelength-
and polarisation-specific grey->phase calibration is supplied for that panel.

The physical bench carrier is a 20-pixel blaze period on an 8 um panel,
corresponding to 6.25 line-pairs/mm.  Carrier, beam-shaping phase and measured
wavefront correction are kept as separate terms and only then wrapped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from vbb_study.vector_arm_config import SLMPanelConfig


TWOPI = 2.0 * np.pi
EPS = np.finfo(float).tiny


@dataclass(frozen=True)
class SLMPhaseCalibration:
    """One measured panel grey-level -> optical phase calibration."""

    panel_id: str
    wavelength_m: float
    grey_levels: np.ndarray
    phase_rad: np.ndarray
    polarisation_label: str = "phase-modulating-axis"
    calibration_date: str | None = None

    def validate(self) -> None:
        grey = np.asarray(self.grey_levels, dtype=float).ravel()
        phase = np.asarray(self.phase_rad, dtype=float).ravel()
        if not self.panel_id:
            raise ValueError("panel_id is required")
        if not np.isfinite(self.wavelength_m) or float(self.wavelength_m) <= 0.0:
            raise ValueError("wavelength_m must be positive")
        if grey.size != phase.size or grey.size < 16:
            raise ValueError("grey_levels and phase_rad must contain the same >=16 samples")
        if not np.all(np.isfinite(grey)) or not np.all(np.isfinite(phase)):
            raise ValueError("SLM calibration arrays must be finite")
        if np.any(np.diff(grey) <= 0.0):
            raise ValueError("grey_levels must increase strictly")
        if grey[0] < 0.0 or grey[-1] > 255.0:
            raise ValueError("grey_levels must lie in [0,255]")
        if np.any(np.diff(phase) <= 0.0):
            raise ValueError(
                "phase_rad must be an unwrapped monotonically increasing response; "
                "preprocess a non-monotonic raw interferometric scan before mask export"
            )

    @property
    def phase_stroke_rad(self) -> float:
        self.validate()
        phase = np.asarray(self.phase_rad, dtype=float).ravel()
        return float(phase[-1] - phase[0])


@dataclass(frozen=True)
class CalibratedSLMMask:
    desired_wrapped_phase_rad: np.ndarray
    grey_u8: np.ndarray
    realised_phase_rad: np.ndarray
    phase_error_rad: np.ndarray
    metadata: Mapping[str, object]


def slm_device_grid(panel: SLMPanelConfig) -> dict[str, np.ndarray | float | int]:
    """Return the native rectangular half-pixel-centred SLM device grid."""

    pitch = float(panel.pitch_m)
    x = (np.arange(int(panel.n_x), dtype=float) - int(panel.n_x) / 2.0 + 0.5) * pitch
    y = (np.arange(int(panel.n_y), dtype=float) - int(panel.n_y) / 2.0 + 0.5) * pitch
    X, Y = np.meshgrid(x, y, indexing="xy")
    return {
        "nx": int(panel.n_x),
        "ny": int(panel.n_y),
        "pitch_m": pitch,
        "x": x,
        "y": y,
        "X": X,
        "Y": Y,
    }


def physical_carrier_phase(
    X_m: np.ndarray,
    *,
    panel: SLMPanelConfig,
) -> np.ndarray:
    """Return the physical blaze phase using the panel carrier convention."""

    return TWOPI * float(panel.carrier_lp_per_m) * np.asarray(X_m, dtype=float)


def compose_display_phase(
    beam_phase_rad: np.ndarray,
    *,
    carrier_phase_rad: np.ndarray | float = 0.0,
    correction_phase_rad: np.ndarray | float = 0.0,
    auxiliary_phase_rad: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Compose independent phase terms without conjugating desired topology."""

    beam = np.asarray(beam_phase_rad, dtype=float)
    total = (
        beam
        + np.asarray(carrier_phase_rad, dtype=float)
        + np.asarray(correction_phase_rad, dtype=float)
        + np.asarray(auxiliary_phase_rad, dtype=float)
    )
    return np.mod(total, TWOPI)


def _circular_phase_error(realised: np.ndarray, desired: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (np.asarray(realised) - np.asarray(desired))))


def calibrated_phase_to_grey(
    desired_phase_rad: np.ndarray,
    calibration: SLMPhaseCalibration,
    *,
    require_full_2pi_stroke: bool = True,
) -> CalibratedSLMMask:
    """Invert a measured LUT and return an actual uint8 hardware mask.

    ``phase_rad`` in the calibration is the *unwrapped* measured optical phase.
    Desired phase is wrapped to one 2*pi interval and embedded into the first
    calibrated 2*pi interval.  A full-stroke hardware export is rejected when
    the measured response does not cover 2*pi.
    """

    calibration.validate()
    grey_axis = np.asarray(calibration.grey_levels, dtype=float).ravel()
    phase_axis = np.asarray(calibration.phase_rad, dtype=float).ravel()
    stroke = float(phase_axis[-1] - phase_axis[0])
    if require_full_2pi_stroke and stroke < TWOPI * (1.0 - 1.0e-3):
        raise ValueError(
            f"measured phase stroke {stroke:.6f} rad is below 2*pi; "
            "do not export a nominal full-range hardware mask"
        )

    desired = np.mod(np.asarray(desired_phase_rad, dtype=float), TWOPI)
    target_unwrapped = float(phase_axis[0]) + desired
    if float(np.max(target_unwrapped)) > float(phase_axis[-1]) + 1.0e-12:
        raise ValueError("desired wrapped phase exceeds the measured panel phase stroke")

    grey_float = np.interp(target_unwrapped, phase_axis, grey_axis)
    grey_u8 = np.asarray(np.clip(np.rint(grey_float), 0.0, 255.0), dtype=np.uint8)
    realised_unwrapped = np.interp(grey_u8.astype(float), grey_axis, phase_axis)
    realised_wrapped = np.mod(realised_unwrapped - float(phase_axis[0]), TWOPI)
    error = _circular_phase_error(realised_wrapped, desired)

    return CalibratedSLMMask(
        desired_wrapped_phase_rad=np.asarray(desired, dtype=float),
        grey_u8=grey_u8,
        realised_phase_rad=np.asarray(realised_wrapped, dtype=float),
        phase_error_rad=np.asarray(error, dtype=float),
        metadata={
            "panel_id": calibration.panel_id,
            "wavelength_m": float(calibration.wavelength_m),
            "polarisation_label": calibration.polarisation_label,
            "calibration_date": calibration.calibration_date,
            "phase_stroke_rad": stroke,
            "full_2pi_stroke": bool(stroke >= TWOPI * (1.0 - 1.0e-3)),
            "phase_error_rms_rad": float(np.sqrt(np.mean(error * error))),
            "phase_error_peak_abs_rad": float(np.max(np.abs(error))),
            "hardware_mask_calibrated": True,
        },
    )


def build_calibrated_device_mask(
    beam_phase_rad: np.ndarray,
    *,
    panel: SLMPanelConfig,
    calibration: SLMPhaseCalibration,
    correction_phase_rad: np.ndarray | float = 0.0,
    auxiliary_phase_rad: np.ndarray | float = 0.0,
    include_carrier: bool = True,
) -> CalibratedSLMMask:
    """Compose beam+20-px-carrier+correction and invert the measured LUT."""

    grid = slm_device_grid(panel)
    shape = (int(panel.n_y), int(panel.n_x))
    beam = np.asarray(beam_phase_rad, dtype=float)
    if beam.shape != shape:
        raise ValueError(f"beam_phase_rad must have native SLM shape {shape}")
    carrier = (
        physical_carrier_phase(np.asarray(grid["X"]), panel=panel)
        if include_carrier
        else 0.0
    )
    desired = compose_display_phase(
        beam,
        carrier_phase_rad=carrier,
        correction_phase_rad=correction_phase_rad,
        auxiliary_phase_rad=auxiliary_phase_rad,
    )
    result = calibrated_phase_to_grey(desired, calibration)
    metadata = dict(result.metadata)
    metadata.update(
        {
            "panel_resolution_px": [int(panel.n_x), int(panel.n_y)],
            "pixel_pitch_m": float(panel.pitch_m),
            "carrier_lp_per_mm": float(panel.carrier_lp_per_mm) if include_carrier else 0.0,
            "carrier_period_px": float(panel.carrier_period_px) if include_carrier else None,
            "carrier_contract": "20_px_physical_bench" if include_carrier else "disabled",
        }
    )
    return CalibratedSLMMask(
        desired_wrapped_phase_rad=result.desired_wrapped_phase_rad,
        grey_u8=result.grey_u8,
        realised_phase_rad=result.realised_phase_rad,
        phase_error_rad=result.phase_error_rad,
        metadata=metadata,
    )


__all__ = [
    "CalibratedSLMMask",
    "SLMPhaseCalibration",
    "build_calibrated_device_mask",
    "calibrated_phase_to_grey",
    "compose_display_phase",
    "physical_carrier_phase",
    "slm_device_grid",
]
