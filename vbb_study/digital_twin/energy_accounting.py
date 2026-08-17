"""
Energy accounting and fluence estimation for the beam-to-write digital twin.

Model status: energy_accounting_prediction / fluence_prediction
Final export allowed: False at Stage 8B (no material response implemented)

All functions in this module operate on well-established analytical formulas.
They do NOT implement:
  - Material damage, ablation, modification, or waveguide formation.
  - Nonlinear propagation or plasma defocusing.
  - Heat accumulation.
  - Any calibrated material-response model.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence, List

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_J_PER_uJ = 1e-6          # µJ → J
_CM2_PER_UM2 = 1e-8       # µm² → cm²
_FS_PER_S = 1e15           # fs → s (inverse: s × 1e15 = fs)
_S_PER_FS = 1e-15          # fs → s

STAGE = "stage8b_optical_cockpit_energy_accounting"
MODEL_STATUS = "energy_accounting_prediction"
FINAL_EXPORT_ALLOWED = False

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LaserSource:
    """Laser source specification (user-facing physical parameters)."""

    wavelength_nm: float
    pulse_duration_fs: float
    repetition_rate_Hz: float
    pulse_energy_before_optics_uJ: float
    average_power_limit_W: float | None = None
    beam_radius_mm: float = 2.0
    polarisation_state: str = "linear"

    def __post_init__(self):
        if self.wavelength_nm <= 0:
            raise ValueError(f"wavelength_nm must be positive, got {self.wavelength_nm}")
        if self.pulse_duration_fs <= 0:
            raise ValueError(f"pulse_duration_fs must be positive, got {self.pulse_duration_fs}")
        if self.repetition_rate_Hz <= 0:
            raise ValueError(f"repetition_rate_Hz must be positive, got {self.repetition_rate_Hz}")
        if self.pulse_energy_before_optics_uJ <= 0:
            raise ValueError(
                f"pulse_energy_before_optics_uJ must be positive, got {self.pulse_energy_before_optics_uJ}"
            )
        if self.beam_radius_mm <= 0:
            raise ValueError(f"beam_radius_mm must be positive, got {self.beam_radius_mm}")


@dataclass
class OpticalComponent:
    """A single element in the optical chain with a transmission/efficiency factor."""

    component_name: str
    component_type: str
    transmission_fraction: float
    efficiency_fraction: float = 1.0
    notes: str = ""
    enabled: bool = True

    def combined_fraction(self) -> float:
        """Product of transmission and efficiency fractions."""
        return self.transmission_fraction * self.efficiency_fraction

    def __post_init__(self):
        if self.enabled:
            validate_fraction(self.transmission_fraction, f"{self.component_name}.transmission_fraction")
            validate_fraction(self.efficiency_fraction, f"{self.component_name}.efficiency_fraction")


@dataclass
class EnergyLedgerRow:
    """One row in the sequential energy ledger."""

    component_name: str
    component_type: str
    enabled: bool
    fraction_this_component: float
    cumulative_fraction: float
    energy_in_uJ: float
    energy_out_uJ: float
    average_power_out_W: float
    notes: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class EnergyLedger:
    """Complete sequential energy accounting through all optical components."""

    source: LaserSource | None
    rows: List[EnergyLedgerRow]
    energy_at_sample_uJ: float
    total_throughput_fraction: float
    average_power_at_sample_W: float
    ledger_warnings: List[str]
    model_status: str = MODEL_STATUS
    final_export_allowed: bool = FINAL_EXPORT_ALLOWED


@dataclass
class FluenceEstimate:
    """Fluence estimate from effective area or normalised intensity array."""

    peak_fluence_j_cm2: float
    method: str
    pulse_energy_at_sample_uJ: float
    model_status: str = "fluence_prediction"
    notes: str = ""
    final_export_allowed: bool = FINAL_EXPORT_ALLOWED


@dataclass
class PeakIntensityEstimate:
    """Approximate peak intensity from fluence and pulse duration."""

    peak_intensity_w_cm2: float
    peak_fluence_j_cm2: float
    pulse_duration_fs: float
    temporal_shape: str
    approximation_note: str = (
        "Peak intensity estimate assumes no nonlinear reshaping and no plasma/thermal feedback."
    )
    model_status: str = "energy_accounting_prediction"
    final_export_allowed: bool = FINAL_EXPORT_ALLOWED


# ---------------------------------------------------------------------------
# Core utility functions
# ---------------------------------------------------------------------------


def validate_fraction(value: float, name: str) -> float:
    """Raise ValueError if value is not in [0, 1]."""
    if not (0.0 <= float(value) <= 1.0):
        raise ValueError(
            f"{name} must be in [0, 1]; got {value:.6g}. "
            "Transmission and efficiency fractions must not exceed 1."
        )
    return float(value)


def average_power_w(pulse_energy_uJ: float, repetition_rate_Hz: float) -> float:
    """
    Average power in watts.

    P_avg = E_p [J] * f_rep [Hz]
    """
    if pulse_energy_uJ < 0:
        raise ValueError(f"pulse_energy_uJ must be non-negative, got {pulse_energy_uJ}")
    if repetition_rate_Hz <= 0:
        raise ValueError(f"repetition_rate_Hz must be positive, got {repetition_rate_Hz}")
    return float(pulse_energy_uJ) * _J_PER_uJ * float(repetition_rate_Hz)


def fresnel_normal_incidence_transmission(n1: float, n2: float) -> float:
    """
    Power transmission at normal incidence from medium n1 to n2 (no coating).

    T = 1 - R,  R = ((n1-n2)/(n1+n2))^2
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError(f"Refractive indices must be positive; got n1={n1}, n2={n2}")
    r = (n1 - n2) / (n1 + n2)
    return float(1.0 - r * r)


def energy_fraction_after_components(components: Sequence[OpticalComponent]) -> float:
    """
    Cumulative throughput fraction after all enabled components.

    Sequential multiplication: f_total = prod(T_i * eta_i for enabled i)
    """
    frac = 1.0
    for c in components:
        if c.enabled:
            frac *= c.combined_fraction()
    return frac


# ---------------------------------------------------------------------------
# Energy ledger
# ---------------------------------------------------------------------------


def compute_energy_ledger(
    pulse_energy_before_optics_uJ: float,
    repetition_rate_Hz: float,
    components: Sequence[OpticalComponent],
    average_power_limit_W: float | None = None,
    *,
    source: LaserSource | None = None,
) -> EnergyLedger:
    """
    Compute sequential energy flow through all optical components.

    E_out,i = E_in,i * T_i * eta_i

    Returns an EnergyLedger with a row for each component plus summary.
    Appends warnings for low throughput, power-limit violations, zero energy.
    """
    if source is not None:
        pulse_energy_before_optics_uJ = source.pulse_energy_before_optics_uJ
        repetition_rate_Hz = source.repetition_rate_Hz
        if average_power_limit_W is None:
            average_power_limit_W = source.average_power_limit_W

    if pulse_energy_before_optics_uJ <= 0:
        raise ValueError(
            f"pulse_energy_before_optics_uJ must be positive, got {pulse_energy_before_optics_uJ}"
        )
    if repetition_rate_Hz <= 0:
        raise ValueError(f"repetition_rate_Hz must be positive, got {repetition_rate_Hz}")

    rows: List[EnergyLedgerRow] = []
    ledger_warnings: List[str] = []
    energy_in_uJ = float(pulse_energy_before_optics_uJ)
    cumulative_fraction = 1.0

    for comp in components:
        row_warnings: List[str] = []

        if not comp.enabled:
            row = EnergyLedgerRow(
                component_name=comp.component_name,
                component_type=comp.component_type,
                enabled=False,
                fraction_this_component=1.0,
                cumulative_fraction=cumulative_fraction,
                energy_in_uJ=energy_in_uJ,
                energy_out_uJ=energy_in_uJ,
                average_power_out_W=average_power_w(energy_in_uJ, repetition_rate_Hz),
                notes=comp.notes + " [DISABLED — component skipped]",
                warnings=["Component disabled; energy passes through unchanged."],
            )
            rows.append(row)
            continue

        frac = comp.combined_fraction()
        energy_out_uJ = energy_in_uJ * frac
        cumulative_fraction *= frac
        avg_power_out = average_power_w(energy_out_uJ, repetition_rate_Hz)

        if frac < 0.05:
            row_warnings.append(
                f"Unusually low throughput at '{comp.component_name}': {frac:.1%}"
            )
        if comp.transmission_fraction == 0.0 or frac == 0.0:
            row_warnings.append(
                f"Zero throughput at '{comp.component_name}': no energy passes."
            )
        if average_power_limit_W is not None and avg_power_out > average_power_limit_W:
            msg = (
                f"Average power after '{comp.component_name}' = {avg_power_out:.3f} W "
                f"exceeds limit of {average_power_limit_W:.3f} W."
            )
            row_warnings.append(msg)
            ledger_warnings.append(msg)

        row = EnergyLedgerRow(
            component_name=comp.component_name,
            component_type=comp.component_type,
            enabled=True,
            fraction_this_component=frac,
            cumulative_fraction=cumulative_fraction,
            energy_in_uJ=energy_in_uJ,
            energy_out_uJ=energy_out_uJ,
            average_power_out_W=avg_power_out,
            notes=comp.notes,
            warnings=row_warnings,
        )
        rows.append(row)
        ledger_warnings.extend(row_warnings)
        energy_in_uJ = energy_out_uJ

    energy_at_sample = energy_in_uJ

    if cumulative_fraction < 0.01:
        msg = (
            f"Total throughput {cumulative_fraction:.2%} is very low; "
            "check component efficiencies."
        )
        ledger_warnings.append(msg)
    if energy_at_sample <= 0.0:
        ledger_warnings.append("No energy reaches the sample — check chain configuration.")

    avg_at_sample = average_power_w(energy_at_sample, repetition_rate_Hz)
    if average_power_limit_W is not None and avg_at_sample > average_power_limit_W:
        msg = (
            f"Average power at sample ({avg_at_sample:.3f} W) exceeds "
            f"limit ({average_power_limit_W:.3f} W)."
        )
        if msg not in ledger_warnings:
            ledger_warnings.append(msg)

    return EnergyLedger(
        source=source,
        rows=rows,
        energy_at_sample_uJ=energy_at_sample,
        total_throughput_fraction=cumulative_fraction,
        average_power_at_sample_W=avg_at_sample,
        ledger_warnings=ledger_warnings,
    )


# ---------------------------------------------------------------------------
# Fluence estimates
# ---------------------------------------------------------------------------


def fluence_from_effective_area_j_cm2(
    pulse_energy_uJ: float,
    effective_area_um2: float,
) -> float:
    """
    Quick cockpit fluence estimate from effective beam area.

    F = E [J] / A [cm²]

    Model status: fluence_prediction (approximate; uses assumed effective area)
    """
    if pulse_energy_uJ < 0:
        raise ValueError(f"pulse_energy_uJ must be non-negative, got {pulse_energy_uJ}")
    if effective_area_um2 <= 0:
        raise ValueError(f"effective_area_um2 must be positive, got {effective_area_um2}")
    E_j = pulse_energy_uJ * _J_PER_uJ
    A_cm2 = effective_area_um2 * _CM2_PER_UM2
    return float(E_j / A_cm2)


def scale_intensity_to_fluence_j_cm2(
    intensity: np.ndarray,
    dx_um: float,
    dy_um: float,
    pulse_energy_uJ: float,
) -> np.ndarray:
    """
    Scale a normalised intensity array to a physical fluence map [J/cm²].

    F(x,y) = E_sample * I(x,y) / (sum(I) * dA)

    where dA = dx_um * dy_um [µm²] converted to cm².

    Raises ValueError if intensity is empty, contains negatives, or
    integrates to zero.

    Model status: fluence_prediction
    """
    intensity = np.asarray(intensity, dtype=float)
    if intensity.size == 0:
        raise ValueError("intensity array is empty.")
    if not np.all(np.isfinite(intensity)):
        raise ValueError("intensity array contains non-finite values (NaN/inf).")
    if np.any(intensity < 0):
        raise ValueError("intensity array contains negative values.")
    if not np.isfinite(dx_um) or dx_um <= 0:
        raise ValueError(f"dx_um must be positive, got {dx_um}")
    if not np.isfinite(dy_um) or dy_um <= 0:
        raise ValueError(f"dy_um must be positive, got {dy_um}")
    if not np.isfinite(pulse_energy_uJ):
        raise ValueError(f"pulse_energy_uJ must be finite, got {pulse_energy_uJ}")
    if pulse_energy_uJ < 0:
        raise ValueError(f"pulse_energy_uJ must be non-negative, got {pulse_energy_uJ}")

    dA_cm2 = dx_um * dy_um * _CM2_PER_UM2
    total = float(np.sum(intensity)) * dA_cm2
    if total == 0.0:
        raise ValueError(
            "intensity integral is zero — cannot normalise to fluence. "
            "Check that the intensity array is non-zero."
        )

    E_j = pulse_energy_uJ * _J_PER_uJ
    return intensity * (E_j / total)


def peak_fluence_j_cm2(fluence: np.ndarray) -> float:
    """Peak value of a fluence array [J/cm²]."""
    fluence = np.asarray(fluence, dtype=float)
    if fluence.size == 0:
        raise ValueError("fluence array is empty.")
    return float(np.max(fluence))


def peak_intensity_w_cm2(
    peak_fluence_j_cm2_val: float,
    pulse_duration_fs: float,
    temporal_shape: str = "flat_top_equivalent",
) -> PeakIntensityEstimate:
    """
    Approximate peak intensity from peak fluence and pulse duration.

    I_peak ≈ F_peak / τ

    Temporal shape 'gaussian' applies a factor of sqrt(4 ln 2 / π) ≈ 0.94
    relative to the flat-top equivalent, but the default is flat-top for
    conservatism.  The result is explicitly labelled approximate.

    Model status: energy_accounting_prediction
    Approximation note: no nonlinear reshaping, plasma, or thermal feedback.
    """
    if peak_fluence_j_cm2_val < 0:
        raise ValueError(f"peak_fluence_j_cm2 must be non-negative, got {peak_fluence_j_cm2_val}")
    if pulse_duration_fs <= 0:
        raise ValueError(f"pulse_duration_fs must be positive, got {pulse_duration_fs}")

    tau_s = pulse_duration_fs * _S_PER_FS

    if temporal_shape == "gaussian":
        # FWHM pulse → peak intensity factor: I_peak = F / (τ_FWHM * sqrt(π/(4 ln 2)))
        # Approximately 0.94 × (F / τ_FWHM) — small correction
        gaussian_factor = np.sqrt(4.0 * np.log(2.0) / np.pi)
        I_peak = float(peak_fluence_j_cm2_val / tau_s * gaussian_factor)
        note = (
            f"Gaussian temporal profile factor {gaussian_factor:.4f} applied. "
            "Peak intensity estimate assumes no nonlinear reshaping and no plasma/thermal feedback."
        )
    else:
        I_peak = float(peak_fluence_j_cm2_val / tau_s)
        note = (
            "Flat-top equivalent: I_peak = F_peak / tau. Conservative approximation. "
            "Peak intensity estimate assumes no nonlinear reshaping and no plasma/thermal feedback."
        )

    return PeakIntensityEstimate(
        peak_intensity_w_cm2=I_peak,
        peak_fluence_j_cm2=peak_fluence_j_cm2_val,
        pulse_duration_fs=pulse_duration_fs,
        temporal_shape=temporal_shape,
        approximation_note=note,
    )


# ---------------------------------------------------------------------------
# Default optical chain helper
# ---------------------------------------------------------------------------


def default_holographic_chain(
    slm_diffraction_efficiency: float = 0.75,
    selected_first_order_fraction: float = 0.73,
    relay_transmission: float = 0.90,
    objective_transmission: float = 0.85,
    sample_interface_transmission: float = 0.95,
    additional_user_loss: float = 1.0,
) -> List[OpticalComponent]:
    """
    Return a default holographic SLM optical chain for cockpit use.

    All fractions must be in [0, 1].  Raises ValueError if not.
    """
    return [
        OpticalComponent(
            component_name="pre_slm_optics",
            component_type="passive_optics",
            transmission_fraction=1.0,
            notes="Pre-SLM beam-shaping optics (default lossless unless user adjusts).",
        ),
        OpticalComponent(
            component_name="slm_diffraction_efficiency",
            component_type="slm",
            transmission_fraction=slm_diffraction_efficiency,
            notes="SLM diffraction efficiency into all orders.",
        ),
        OpticalComponent(
            component_name="selected_first_order_fraction",
            component_type="filter",
            transmission_fraction=selected_first_order_fraction,
            notes="Fraction of total SLM output that enters the +1 diffraction order.",
        ),
        OpticalComponent(
            component_name="relay_optics",
            component_type="passive_optics",
            transmission_fraction=relay_transmission,
            notes="4f relay optics transmission.",
        ),
        OpticalComponent(
            component_name="objective_transmission",
            component_type="objective",
            transmission_fraction=objective_transmission,
            notes="Microscope objective transmission at laser wavelength.",
        ),
        OpticalComponent(
            component_name="sample_interface_transmission",
            component_type="interface",
            transmission_fraction=sample_interface_transmission,
            notes="Air-to-sample Fresnel + anti-reflection coating transmission.",
        ),
        OpticalComponent(
            component_name="additional_user_loss",
            component_type="user_defined",
            transmission_fraction=additional_user_loss,
            notes="Additional user-defined loss factor (default 1.0 = no extra loss).",
        ),
    ]
