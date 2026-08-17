"""
Exposure bookkeeping for the beam-to-write digital twin.

Model status: exposure_bookkeeping
Final export allowed: False at Stage 8B

This module computes geometric exposure quantities:
  - pulse spacing
  - effective pulses per spot
  - dose-per-unit-length proxy
  - static multi-pulse total energy
  - line-scan exposure summary

IMPORTANT: These quantities describe laser exposure geometry.
They do NOT predict material modification, damage, ablation,
waveguide formation, void formation, or crack propagation.
Exposure bookkeeping is not a dose-calibrated material-response model.
"""

from __future__ import annotations

from typing import Mapping

MODEL_STATUS = "exposure_bookkeeping"
FINAL_EXPORT_ALLOWED = False

# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------

_MM_PER_S_TO_UM_PER_S = 1_000.0   # mm/s → µm/s
_MM_PER_S_TO_M_PER_S = 1e-3       # mm/s → m/s
_J_PER_uJ = 1e-6                   # µJ → J


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require_positive(value: float, name: str) -> float:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive; got {value!r}.")
    return float(value)


def _require_non_negative_int(value: int, name: str) -> int:
    if int(value) < 0:
        raise ValueError(f"{name} must be non-negative; got {value!r}.")
    return int(value)


# ---------------------------------------------------------------------------
# Core exposure functions
# ---------------------------------------------------------------------------


def pulse_spacing_um(scan_speed_mm_s: float, repetition_rate_Hz: float) -> float:
    """
    Centre-to-centre spacing between consecutive pulses along the scan axis.

    Δs = v [µm/s] / f_rep [Hz]

    Parameters
    ----------
    scan_speed_mm_s : float
        Linear scan speed in mm/s.
    repetition_rate_Hz : float
        Laser repetition rate in Hz.

    Returns
    -------
    float
        Pulse spacing in µm.
    """
    v = _require_positive(scan_speed_mm_s, "scan_speed_mm_s")
    f = _require_positive(repetition_rate_Hz, "repetition_rate_Hz")
    v_um_s = v * _MM_PER_S_TO_UM_PER_S
    return v_um_s / f


def pulses_per_spot(
    effective_diameter_um: float,
    scan_speed_mm_s: float,
    repetition_rate_Hz: float,
) -> float:
    """
    Effective number of pulses within one beam-spot diameter during a scan.

    N_eff ≈ d_eff [µm] * f_rep [Hz] / v [µm/s]

    This is a geometric estimate — it does not account for the fluence
    profile of individual pulses.  For continuous exposure at the centre
    of the beam, N_eff pulses contribute to the cumulative dose.

    Parameters
    ----------
    effective_diameter_um : float
        Effective beam diameter (1/e² diameter or threshold-crossing diameter) in µm.
    scan_speed_mm_s : float
        Linear scan speed in mm/s.
    repetition_rate_Hz : float
        Laser repetition rate in Hz.

    Returns
    -------
    float
        Effective pulses per spot (may be fractional; < 1 means dotted regime).
    """
    d = _require_positive(effective_diameter_um, "effective_diameter_um")
    v = _require_positive(scan_speed_mm_s, "scan_speed_mm_s")
    f = _require_positive(repetition_rate_Hz, "repetition_rate_Hz")
    v_um_s = v * _MM_PER_S_TO_UM_PER_S
    return d * f / v_um_s


def dose_per_unit_length_proxy(
    pulse_energy_at_sample_uJ: float,
    repetition_rate_Hz: float,
    scan_speed_mm_s: float,
) -> float:
    """
    Dose-per-unit-length proxy: E_p * f_rep / v.

    D_L ∝ E_p [J] * f_rep [Hz] / v [m/s]   [J/m]

    This is a bookkeeping proxy for total energy delivered per unit scan
    length.  It is NOT a material dose or absorbed energy density.

    Parameters
    ----------
    pulse_energy_at_sample_uJ : float
        Pulse energy at the sample in µJ.
    repetition_rate_Hz : float
        Laser repetition rate in Hz.
    scan_speed_mm_s : float
        Linear scan speed in mm/s.

    Returns
    -------
    float
        Dose-per-unit-length in J/m (proxy — not calibrated material dose).
    """
    E = _require_positive(pulse_energy_at_sample_uJ, "pulse_energy_at_sample_uJ")
    f = _require_positive(repetition_rate_Hz, "repetition_rate_Hz")
    v = _require_positive(scan_speed_mm_s, "scan_speed_mm_s")
    E_j = E * _J_PER_uJ
    v_m_s = v * _MM_PER_S_TO_M_PER_S
    return E_j * f / v_m_s


def static_exposure_total_energy_uJ(
    pulse_energy_at_sample_uJ: float,
    num_pulses: int,
) -> float:
    """
    Total energy deposited for a static (stopped-beam) multi-pulse exposure.

    E_total = E_p * N

    Parameters
    ----------
    pulse_energy_at_sample_uJ : float
        Per-pulse energy at sample in µJ.
    num_pulses : int
        Number of pulses in the exposure.

    Returns
    -------
    float
        Total cumulative energy in µJ.
    """
    E = _require_positive(pulse_energy_at_sample_uJ, "pulse_energy_at_sample_uJ")
    N = _require_non_negative_int(num_pulses, "num_pulses")
    return E * N


def line_exposure_summary(
    pulse_energy_at_sample_uJ: float,
    repetition_rate_Hz: float,
    scan_speed_mm_s: float,
    line_length_um: float,
    effective_diameter_um: float,
) -> Mapping[str, float | str | list]:
    """
    Compute a full line-scan exposure summary.

    Parameters
    ----------
    pulse_energy_at_sample_uJ : float
        Per-pulse energy at sample in µJ.
    repetition_rate_Hz : float
        Laser repetition rate in Hz.
    scan_speed_mm_s : float
        Linear scan speed in mm/s.
    line_length_um : float
        Total line length in µm.
    effective_diameter_um : float
        Effective beam diameter in µm (determines N_eff and overlap).

    Returns
    -------
    dict with keys:
        pulse_spacing_um        : Δs between pulses [µm]
        pulses_per_spot         : N_eff [dimensionless]
        dose_per_unit_length_J_m: E_p * f / v [J/m]
        line_duration_s         : line_length / v [s]
        total_pulses_on_line    : N_total [integer]
        total_energy_on_line_uJ : E_p * N_total [µJ]
        overlap_fraction        : 1 - Δs/d_eff (negative means gaps)
        model_status            : exposure_bookkeeping
        final_export_allowed    : False
        warnings                : list of warning strings
    """
    E = _require_positive(pulse_energy_at_sample_uJ, "pulse_energy_at_sample_uJ")
    f = _require_positive(repetition_rate_Hz, "repetition_rate_Hz")
    v = _require_positive(scan_speed_mm_s, "scan_speed_mm_s")
    L = _require_positive(line_length_um, "line_length_um")
    d = _require_positive(effective_diameter_um, "effective_diameter_um")

    v_um_s = v * _MM_PER_S_TO_UM_PER_S
    v_m_s = v * _MM_PER_S_TO_M_PER_S
    L_m = L * 1e-6  # µm → m

    ds_um = v_um_s / f
    N_eff = d * f / v_um_s
    D_L = E * _J_PER_uJ * f / v_m_s
    t_line = L_m / v_m_s
    N_total = int(t_line * f)
    E_total_uJ = E * N_total
    overlap = 1.0 - ds_um / d

    wrns: list = []
    if N_eff < 1.0:
        wrns.append(
            f"Pulses per spot = {N_eff:.2f} < 1: likely dotted/discontinuous write regime. "
            f"Increase f_rep or decrease scan speed."
        )
    if N_eff > 1000:
        wrns.append(
            f"Pulses per spot = {N_eff:.0f} is very high. "
            f"Consider heat accumulation effects at {repetition_rate_Hz:.0f} Hz."
        )
    if ds_um > d:
        wrns.append(
            f"Pulse spacing ({ds_um:.2f} µm) > effective diameter ({d:.2f} µm): "
            f"gaps between spots — track will not be continuous."
        )
    if N_total == 0:
        wrns.append(
            f"No pulses on line of length {L:.1f} µm at this scan speed and rep rate."
        )

    return {
        "pulse_spacing_um": ds_um,
        "pulses_per_spot": N_eff,
        "dose_per_unit_length_J_m": D_L,
        "line_duration_s": t_line,
        "total_pulses_on_line": N_total,
        "total_energy_on_line_uJ": E_total_uJ,
        "overlap_fraction": overlap,
        "model_status": MODEL_STATUS,
        "final_export_allowed": FINAL_EXPORT_ALLOWED,
        "warnings": wrns,
    }
