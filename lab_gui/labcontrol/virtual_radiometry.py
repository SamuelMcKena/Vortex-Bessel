"""Fixed relative radiometry for synthetic cameras, not a Beamage calibration."""
from __future__ import annotations

import math
import numpy as np

# Reference Gaussian: 2 mm 1/e field radius, unit on-axis amplitude.
# Preserve its incident power before finite-window/aperture losses.
REFERENCE_RADIUS_M = 2e-3
COUNTS_PER_INTENSITY_AT_1MS = 0.72 * 4095.0 / 100.0


def fixed_power_amplitude(radius_x_m: float, radius_y_m: float) -> float:
    if not all(math.isfinite(v) and v > 0 for v in (radius_x_m, radius_y_m)):
        raise ValueError("Gaussian radii must be finite and positive.")
    return REFERENCE_RADIUS_M / math.sqrt(radius_x_m * radius_y_m)


def camera_response_scale(exposure_us: float, gain: float) -> float:
    if not math.isfinite(exposure_us) or exposure_us < 0 or not math.isfinite(gain) or gain < 0:
        raise ValueError("Virtual exposure and gain must be finite and non-negative.")
    return COUNTS_PER_INTENSITY_AT_1MS * exposure_us / 1000.0 * (1.0 + gain / 100.0)


def linear_camera_signal(intensity, *, exposure_us: float, gain: float):
    scale = camera_response_scale(float(exposure_us), float(gain))
    signal = np.asarray(intensity, dtype=np.float64) * scale
    return signal, {
        "virtual_radiometry": "fixed_relative_sensitivity_v1",
        "counts_per_model_intensity": scale,
        "per_frame_peak_normalisation": False,
        "input_power_convention": "fixed incident power before window/aperture losses; reference 2 mm Gaussian",
    }
