"""Material-facing planning equations used by the publication study.

These are formula helpers only. The richer feature extraction and design-table
logic remains in ``vbb_materials`` because it needs grids, masks, and study
outputs.

Important boundary
------------------
The fluence and threshold helpers here produce optical planning quantities.
They do not model absorption, heating, plasma dynamics, melt flow, void
formation, ablation, stress, or refractive-index change. A threshold comparison
is a proxy unless a separate experimentally calibrated material-response model
is supplied and labelled as such.
"""

from __future__ import annotations

from typing import Any

import numpy as np

EPS = 1.0e-30


def positive_intensity(values: Any) -> np.ndarray:
    """Return a non-negative intensity array for fluence normalisation."""

    return np.maximum(np.asarray(values, dtype=float), 0.0)


def pulse_fluence_from_energy_area_J_cm2(*, pulse_energy_J: float, area_m2: float) -> float:
    """Return uniform pulse fluence ``E/A`` in J/cm^2.

    Parameters are SI: pulse energy in joules and illuminated area in m^2. The
    returned value is divided by ``1e4`` to convert J/m^2 to J/cm^2.
    """

    area = max(float(area_m2), EPS)
    return float(float(pulse_energy_J) / area / 1.0e4)


def gaussian_peak_fluence_J_cm2(*, pulse_energy_J: float, waist_radius_m: float) -> float:
    """Return the peak fluence of a circular Gaussian pulse in J/cm^2.

    The convention is ``F(r) = F0 exp(-2 r^2 / w^2)``, so the transverse
    integral is ``E = F0*pi*w^2/2``. This is an optical fluence formula, not a
    deposited-energy model.
    """

    w = max(float(waist_radius_m), EPS)
    return float(2.0 * float(pulse_energy_J) / (np.pi * w * w) / 1.0e4)


def fluence_from_intensity_J_cm2(intensity: Any, *, dx_m: float, pulse_energy_J: float) -> np.ndarray:
    """Convert one XY intensity plane into fluence in J/cm^2.

    I normalise by ``integral I dA`` so the plane carries the requested pulse
    energy before converting ``J/m^2`` to ``J/cm^2``. This is energy-conserving
    for that single transverse plane.
    """

    I = positive_intensity(intensity)
    denom = float(np.sum(I) * float(dx_m) * float(dx_m)) + EPS
    return (float(pulse_energy_J) * I / denom) / 1.0e4


def line_fluence_proxy_J_cm2(xz_intensity: Any, *, dx_m: float, pulse_energy_J: float) -> np.ndarray:
    """Return the XZ line-write fluence proxy in J/cm^2.

    Each z column is normalised independently; I use this as a comparison
    proxy, not as a conserved deposited-energy model. The correct schema label
    for this helper is ``non_energy_conserving_line_proxy``.
    """

    I = positive_intensity(xz_intensity)
    denom = np.sum(I, axis=0, keepdims=True) * float(dx_m) + EPS
    return (float(pulse_energy_J) * I / denom) / 1.0e4


def line_fluence_proxy_energy_conservation_status() -> str:
    """Return the schema label for ``line_fluence_proxy_J_cm2`` outputs."""

    return "non_energy_conserving_line_proxy"


def effective_pulses_from_scan(*, rep_rate_Hz: float, feature_width_m: float, scan_speed_m_s: float) -> float:
    """Return ``N_eff = f_rep * width / speed`` with a minimum of one pulse."""

    return max(1.0, float(rep_rate_Hz) * float(feature_width_m) / max(float(scan_speed_m_s), EPS))


def incubated_threshold_J_cm2(
    *,
    single_pulse_threshold_J_cm2: float,
    effective_pulses: float,
    incubation_exponent: float,
) -> float:
    """Return ``F_th,N = F_th,1 * N_eff**(S - 1)`` in J/cm^2.

    With the common incubation convention ``S < 1``, the effective threshold
    decreases as pulse count increases. If ``S > 1``, the formula increases
    with pulse count. In this repo the formula is a planning proxy until
    calibrated by experiment.
    """

    return float(single_pulse_threshold_J_cm2) * max(float(effective_pulses), 1.0) ** (float(incubation_exponent) - 1.0)


def fluence_to_threshold_ratio(*, fluence_J_cm2: float, threshold_J_cm2: float) -> float:
    """Return the dimensionless optical fluence-to-threshold proxy ratio."""

    return float(fluence_J_cm2) / max(float(threshold_J_cm2), EPS)


def threshold_mask(values_J_cm2: Any, threshold_J_cm2: float) -> np.ndarray:
    """Return the above-threshold Boolean mask for one fluence field.

    This mask means an optical planning threshold was crossed. It is not a
    calibrated damage, ablation, void, weld, or index-change prediction.
    """

    return np.asarray(values_J_cm2, dtype=float) >= float(threshold_J_cm2)


def thresholded_proxy_area_m2(mask: Any, *, dx_m: float, dy_m: float | None = None) -> float:
    """Return the geometric area of a thresholded planning mask in m^2."""

    dy = float(dx_m if dy_m is None else dy_m)
    return float(np.count_nonzero(np.asarray(mask, dtype=bool)) * float(dx_m) * dy)


def equivalent_diameter_from_area_m(area_m2: float) -> float:
    """Return the diameter of a circle with the supplied area in metres."""

    area = max(float(area_m2), 0.0)
    return float(2.0 * np.sqrt(area / np.pi)) if area > 0.0 else 0.0


def transmission_fraction(*factors: float) -> float:
    """Return the product of optical transmission factors.

    Each factor must lie in ``[0, 1]``. The returned fraction is therefore also
    bounded in ``[0, 1]``.
    """

    product = 1.0
    for factor in factors:
        f = float(factor)
        if f < 0.0 or f > 1.0:
            raise ValueError(f"transmission factors must be in [0, 1], got {factor!r}")
        product *= f
    return float(product)


__all__ = [
    "effective_pulses_from_scan",
    "equivalent_diameter_from_area_m",
    "fluence_from_intensity_J_cm2",
    "fluence_to_threshold_ratio",
    "gaussian_peak_fluence_J_cm2",
    "incubated_threshold_J_cm2",
    "line_fluence_proxy_J_cm2",
    "line_fluence_proxy_energy_conservation_status",
    "positive_intensity",
    "pulse_fluence_from_energy_area_J_cm2",
    "threshold_mask",
    "thresholded_proxy_area_m2",
    "transmission_fraction",
]
