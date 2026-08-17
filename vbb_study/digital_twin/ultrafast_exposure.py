"""Linear ultrafast pulse and optical-exposure utilities.

This module extends the spatial digital twin without introducing an unvalidated
nonlinear material model.  It supports transform-limited/measured spectral
weights, spectral phase (GDD/TOD), coherent reconstruction of wavelength-
resolved spatial fields, peak intensity from fluence, and discrete pulse
accumulation during a line scan.

It does *not* model Kerr self-action, plasma, nonlinear absorption, thermal
transport, stress, ablation or refractive-index modification.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import shift as nd_shift


C0 = 299_792_458.0
TWOPI = 2.0 * np.pi
EPS = np.finfo(float).tiny


@dataclass(frozen=True)
class GaussianPulse:
    central_wavelength_m: float
    intensity_fwhm_s: float
    pulse_energy_J: float
    repetition_rate_Hz: float
    gdd_s2: float = 0.0
    tod_s3: float = 0.0

    def validate(self) -> None:
        for value, name in (
            (self.central_wavelength_m, "central_wavelength_m"),
            (self.intensity_fwhm_s, "intensity_fwhm_s"),
            (self.pulse_energy_J, "pulse_energy_J"),
            (self.repetition_rate_Hz, "repetition_rate_Hz"),
        ):
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.gdd_s2) or not np.isfinite(self.tod_s3):
            raise ValueError("dispersion coefficients must be finite")

    @property
    def central_omega_rad_s(self) -> float:
        return TWOPI * C0 / float(self.central_wavelength_m)

    @property
    def transform_limited_frequency_fwhm_Hz(self) -> float:
        """Gaussian intensity time-bandwidth product Delta_nu*Delta_t=0.44127."""

        return 2.0 * math.log(2.0) / (math.pi * float(self.intensity_fwhm_s))


@dataclass(frozen=True)
class SpectralPulseGrid:
    omega_rad_s: np.ndarray
    wavelength_m: np.ndarray
    field_amplitude_weight: np.ndarray
    spectral_phase_rad: np.ndarray
    metadata: dict[str, float]


def gaussian_spectral_grid(
    pulse: GaussianPulse,
    *,
    n_omega: int = 41,
    half_span_intensity_sigma: float = 4.0,
) -> SpectralPulseGrid:
    """Return a Gaussian field-spectrum quadrature around the carrier.

    For an intensity FWHM ``tau``, the transform-limited field amplitude has

        E(t) ~ exp[-2 ln(2) t^2/tau^2]
        E(Omega) ~ exp[-Omega^2 tau^2/(8 ln(2))].
    """

    pulse.validate()
    n = int(n_omega)
    if n < 5 or n % 2 == 0:
        raise ValueError("n_omega must be an odd integer >=5")
    tau = float(pulse.intensity_fwhm_s)
    # Spectral intensity exp[-Omega^2 tau^2/(4 ln2)] has variance
    # sigma_Omega^2 = 2 ln2 / tau^2.
    sigma_omega_intensity = math.sqrt(2.0 * math.log(2.0)) / tau
    offsets = np.linspace(
        -float(half_span_intensity_sigma) * sigma_omega_intensity,
        +float(half_span_intensity_sigma) * sigma_omega_intensity,
        n,
    )
    omega0 = pulse.central_omega_rad_s
    omega = omega0 + offsets
    if np.any(omega <= 0.0):
        raise ValueError("spectral grid reached non-positive optical frequency")
    amplitude = np.exp(-(offsets * offsets) * tau * tau / (8.0 * math.log(2.0)))
    phase = 0.5 * float(pulse.gdd_s2) * offsets**2 + (1.0 / 6.0) * float(pulse.tod_s3) * offsets**3
    # Discrete energy weights are |amplitude|^2 dOmega. Normalize so the
    # sampled spectral intensity integrates to unity in the trapezoidal rule.
    intensity = amplitude * amplitude
    norm = float(np.trapezoid(intensity, omega))
    amplitude = amplitude / math.sqrt(max(norm, EPS))
    wavelength = TWOPI * C0 / omega
    return SpectralPulseGrid(
        omega_rad_s=np.asarray(omega, dtype=float),
        wavelength_m=np.asarray(wavelength, dtype=float),
        field_amplitude_weight=np.asarray(amplitude, dtype=float),
        spectral_phase_rad=np.asarray(phase, dtype=float),
        metadata={
            "central_wavelength_m": float(pulse.central_wavelength_m),
            "intensity_fwhm_s": tau,
            "transform_limited_frequency_fwhm_Hz": float(pulse.transform_limited_frequency_fwhm_Hz),
            "gdd_s2": float(pulse.gdd_s2),
            "tod_s3": float(pulse.tod_s3),
        },
    )


def coherent_spatiotemporal_field(
    spectral_fields: np.ndarray,
    spectral_grid: SpectralPulseGrid,
    time_s: np.ndarray,
) -> np.ndarray:
    """Reconstruct the complex pulse envelope from wavelength-resolved fields.

    ``spectral_fields`` has shape ``(N_omega, ...)`` and should contain the
    complex spatial transfer result for each sampled wavelength.  The fast
    optical carrier is removed by using ``omega-omega0`` in the reconstruction.
    """

    fields = np.asarray(spectral_fields, dtype=np.complex128)
    omega = np.asarray(spectral_grid.omega_rad_s, dtype=float)
    if fields.shape[0] != omega.size:
        raise ValueError("spectral_fields first axis must match spectral grid")
    t = np.asarray(time_s, dtype=float).ravel()
    if t.size < 2 or np.any(~np.isfinite(t)):
        raise ValueError("time_s must be a finite 1-D array")
    domega = np.gradient(omega)
    omega0 = float(omega[omega.size // 2])
    weights = (
        np.asarray(spectral_grid.field_amplitude_weight, dtype=float)
        * np.exp(1j * np.asarray(spectral_grid.spectral_phase_rad, dtype=float))
        * domega
    )
    phase_t = np.exp(-1j * np.outer(omega - omega0, t))
    flat = fields.reshape(fields.shape[0], -1)
    envelope = np.einsum("w,wt,wp->tp", weights, phase_t, flat, optimize=True)
    return envelope.reshape((t.size,) + fields.shape[1:])


def gaussian_peak_intensity_from_fluence_W_cm2(
    fluence_J_cm2: np.ndarray | float,
    intensity_fwhm_s: float,
) -> np.ndarray:
    """Convert fluence to peak intensity for a Gaussian temporal intensity."""

    tau = float(intensity_fwhm_s)
    if tau <= 0.0:
        raise ValueError("intensity_fwhm_s must be positive")
    factor = math.sqrt(4.0 * math.log(2.0) / math.pi) / tau
    return np.asarray(fluence_J_cm2, dtype=float) * factor


def pulse_spacing_m(*, scan_speed_m_s: float, repetition_rate_Hz: float) -> float:
    if scan_speed_m_s <= 0.0 or repetition_rate_Hz <= 0.0:
        raise ValueError("scan speed and repetition rate must be positive")
    return float(scan_speed_m_s) / float(repetition_rate_Hz)


def accumulated_line_scan_fluence(
    single_pulse_fluence_J_cm2: np.ndarray,
    *,
    dx_m: float,
    scan_speed_m_s: float,
    repetition_rate_Hz: float,
    passes: int = 1,
    shift_axis: int = 1,
    interpolation_order: int = 3,
) -> tuple[np.ndarray, dict[str, float]]:
    """Sum discrete translated pulse fluence maps over a line scan.

    The pulse spacing is exactly ``v/f_rep``.  The scan is long enough that
    pulse centres cover the field of view along the selected axis; contributions
    from pulses whose centres lie outside the array are included while their
    translated footprint still overlaps it.
    """

    fluence = np.maximum(np.asarray(single_pulse_fluence_J_cm2, dtype=float), 0.0)
    if fluence.ndim != 2 or fluence.size == 0:
        raise ValueError("single_pulse_fluence_J_cm2 must be a non-empty 2-D array")
    if dx_m <= 0.0 or passes < 1:
        raise ValueError("dx_m must be positive and passes >=1")
    spacing = pulse_spacing_m(scan_speed_m_s=scan_speed_m_s, repetition_rate_Hz=repetition_rate_Hz)
    spacing_px = spacing / float(dx_m)
    if spacing_px <= 0.0:
        raise ValueError("invalid pulse spacing")
    length_px = fluence.shape[int(shift_axis)]
    # Include a one-FOV halo on both sides so tails from exterior pulse centres
    # are not artificially discarded.
    n_half = int(math.ceil(length_px / max(spacing_px, EPS)))
    shifts = np.arange(-n_half, n_half + 1, dtype=float) * spacing_px
    accumulated = np.zeros_like(fluence, dtype=float)
    shift_vector = [0.0, 0.0]
    for shift_px in shifts:
        shift_vector[int(shift_axis)] = float(shift_px)
        accumulated += nd_shift(
            fluence,
            shift=tuple(shift_vector),
            order=int(interpolation_order),
            mode="constant",
            cval=0.0,
            prefilter=True,
        )
    accumulated *= int(passes)
    return accumulated, {
        "pulse_spacing_m": spacing,
        "pulse_spacing_px": spacing_px,
        "summed_pulse_centres": float(shifts.size * int(passes)),
        "passes": float(passes),
        "model": "linear_incident_fluence_superposition_no_material_memory",
    }


__all__ = [
    "GaussianPulse",
    "SpectralPulseGrid",
    "accumulated_line_scan_fluence",
    "coherent_spatiotemporal_field",
    "gaussian_peak_intensity_from_fluence_W_cm2",
    "gaussian_spectral_grid",
    "pulse_spacing_m",
]
