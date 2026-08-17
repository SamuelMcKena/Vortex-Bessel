"""
Energy-conserving field-to-fluence scaling for the beam-to-write digital twin
(Stage 8C).

This module takes canonical optical-field containers (from
``vbb_study.digital_twin.field_coupling``) and the pulse energy at the sample
(from the Stage 8B energy ledger) and produces transverse fluence maps that
*conserve the pulse energy* over each transverse plane:

    F(x, y) = E_sample * I(x, y) / ( sum(I) * dA )

with the area element dA converted to cm^2 so that F is in J/cm^2.

This is "Mode B" fluence: it scales the *real simulated optical field* rather
than assuming an effective beam area (Stage 8B "Mode A").

Model status: ``fluence_prediction``.  Final export allowed: ``False``.

Hard claim boundary (enforced by accompanying docs/tests):
    Energy-scaled fluence maps are OPTICAL fluence predictions.  They are NOT
    absorbed-energy maps, NOT dose maps, NOT material-modification maps, and NOT
    damage predictions.  For a 3D stack, the per-plane transverse scaling does
    NOT make the volume a deposited-energy volume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from vbb_study.digital_twin.energy_accounting import (
    peak_fluence_j_cm2,
    peak_intensity_w_cm2,
    scale_intensity_to_fluence_j_cm2,
)
from vbb_study.digital_twin.field_coupling import (
    InvalidOpticalFieldError,
    OpticalFieldPlane,
    OpticalFieldStack,
    validate_intensity_plane,
)

STAGE = "stage8c_surfacefield_energy_scaled_cockpit"
MODEL_STATUS = "fluence_prediction"
FINAL_EXPORT_ALLOWED = False

_CM2_PER_UM2 = 1e-8   # um^2 -> cm^2
_J_PER_uJ = 1e-6      # uJ -> J
_uJ_PER_J = 1e6       # J -> uJ

CAVEAT_PLANE = (
    "Energy-scaled optical fluence map (Mode B). This is an OPTICAL fluence "
    "prediction only: not absorbed energy, not dose, not material modification, "
    "not damage."
)
CAVEAT_STACK = (
    "Fluence stack is transverse-plane fluence scaling of the optical field. "
    "It is not a deposited-energy volume, not absorbed energy, and not material "
    "modification."
)

_ALLOWED_NORMALISATIONS = frozenset({"per_plane_transverse_energy"})


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FluencePlaneResult:
    """Energy-scaled fluence for a single transverse plane."""

    fluence_j_cm2: np.ndarray
    pulse_energy_uJ: float
    dx_um: float
    dy_um: float
    integrated_energy_uJ: float
    peak_fluence_j_cm2: float
    mean_fluence_j_cm2: float
    model_status: str = MODEL_STATUS
    caveat: str = CAVEAT_PLANE
    metadata: Mapping[str, Any] = field(default_factory=dict)
    final_export_allowed: bool = FINAL_EXPORT_ALLOWED


@dataclass(frozen=True)
class FluenceStackResult:
    """Per-plane energy-scaled fluence for a 3D optical stack."""

    fluence_zyx_j_cm2: np.ndarray
    pulse_energy_uJ: float
    transverse_energy_by_z_uJ: np.ndarray
    raw_transverse_integral_by_z: np.ndarray
    raw_captured_power_fraction_by_z: np.ndarray
    peak_fluence_by_z_j_cm2: np.ndarray
    peak_z_um: float
    propagation_energy_drift_fraction: float
    model_status: str = MODEL_STATUS
    caveat: str = CAVEAT_STACK
    metadata: Mapping[str, Any] = field(default_factory=dict)
    final_export_allowed: bool = FINAL_EXPORT_ALLOWED


# ---------------------------------------------------------------------------
# Low-level integrals
# ---------------------------------------------------------------------------


def transverse_integral_um2(intensity: np.ndarray, dx_um: float, dy_um: float) -> float:
    """Return the transverse integral sum(I) * dx_um * dy_um (units: intensity * um^2)."""
    arr = validate_intensity_plane(intensity, name="intensity")
    if dx_um <= 0 or dy_um <= 0:
        raise InvalidOpticalFieldError(f"dx_um, dy_um must be positive; got {dx_um}, {dy_um}.")
    return float(np.sum(arr) * dx_um * dy_um)


def integrated_energy_uJ_from_fluence(
    fluence_j_cm2: np.ndarray,
    dx_um: float,
    dy_um: float,
) -> float:
    """Integrate a fluence map [J/cm^2] over area to recover total energy [uJ].

    E = sum(F) * dA[cm^2];  dA[cm^2] = dx_um * dy_um * 1e-8;  J -> uJ via 1e6.
    """
    arr = np.asarray(fluence_j_cm2, dtype=float)
    if arr.size == 0:
        raise InvalidOpticalFieldError("fluence array is empty.")
    if not np.all(np.isfinite(arr)):
        raise InvalidOpticalFieldError("fluence array contains non-finite values.")
    if dx_um <= 0 or dy_um <= 0:
        raise InvalidOpticalFieldError(f"dx_um, dy_um must be positive; got {dx_um}, {dy_um}.")
    dA_cm2 = dx_um * dy_um * _CM2_PER_UM2
    energy_J = float(np.sum(arr)) * dA_cm2
    return energy_J * _uJ_PER_J


def _validate_pulse_energy(pulse_energy_uJ: float) -> float:
    if not np.isfinite(pulse_energy_uJ):
        raise ValueError(f"pulse_energy_uJ must be finite; got {pulse_energy_uJ}.")
    if pulse_energy_uJ < 0:
        raise ValueError(f"pulse_energy_uJ must be non-negative; got {pulse_energy_uJ}.")
    return float(pulse_energy_uJ)


# ---------------------------------------------------------------------------
# Plane scaling
# ---------------------------------------------------------------------------


def scale_plane_to_fluence(
    plane: OpticalFieldPlane,
    pulse_energy_uJ: float,
) -> FluencePlaneResult:
    """Scale a canonical :class:`OpticalFieldPlane` to an energy-conserving fluence map."""
    if not isinstance(plane, OpticalFieldPlane):
        raise TypeError(f"plane must be an OpticalFieldPlane; got {type(plane).__name__}.")
    energy = _validate_pulse_energy(pulse_energy_uJ)

    fluence = scale_intensity_to_fluence_j_cm2(
        plane.intensity, plane.dx_um, plane.dy_um, energy
    )
    integrated = integrated_energy_uJ_from_fluence(fluence, plane.dx_um, plane.dy_um)
    peak = peak_fluence_j_cm2(fluence)
    mean = float(np.mean(fluence))

    metadata = {
        "stage": STAGE,
        "field_label": plane.field_label,
        "source_status": plane.source_status,
        "z_um": plane.z_um,
        "intensity_shape": tuple(int(s) for s in plane.intensity.shape),
        **dict(plane.metadata),
    }
    return FluencePlaneResult(
        fluence_j_cm2=fluence,
        pulse_energy_uJ=energy,
        dx_um=plane.dx_um,
        dy_um=plane.dy_um,
        integrated_energy_uJ=integrated,
        peak_fluence_j_cm2=peak,
        mean_fluence_j_cm2=mean,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Stack scaling
# ---------------------------------------------------------------------------


def scale_stack_to_fluence(
    stack: OpticalFieldStack,
    pulse_energy_uJ: float,
    *,
    normalisation: str = "per_plane_transverse_energy",
) -> FluenceStackResult:
    """Scale a canonical :class:`OpticalFieldStack` to per-plane fluence maps.

    With ``normalisation='per_plane_transverse_energy'`` (the only supported and
    default mode), every transverse z-plane is independently scaled so that it
    integrates to ``pulse_energy_uJ``.  This treats each plane as the transverse
    fluence of the same pulse at that propagation distance.

    The 3D volume integral is deliberately NOT interpreted as the total pulse
    energy (that would double-count the pulse across planes and is physically
    meaningless for a single pulse).

    Propagation energy drift is computed from the *raw* transverse intensity
    integrals (before renormalisation), reporting how much transverse power the
    propagator retains across z.
    """
    if not isinstance(stack, OpticalFieldStack):
        raise TypeError(f"stack must be an OpticalFieldStack; got {type(stack).__name__}.")
    if normalisation not in _ALLOWED_NORMALISATIONS:
        raise ValueError(
            f"Unsupported normalisation {normalisation!r}; "
            f"allowed: {sorted(_ALLOWED_NORMALISATIONS)}."
        )
    energy = _validate_pulse_energy(pulse_energy_uJ)

    dx_um = stack.dx_um
    dy_um = stack.dy_um
    intensity = stack.intensity_zyx
    nz = intensity.shape[0]

    # Raw transverse integrals (before renormalisation) for drift reporting.
    raw_integrals = np.array(
        [float(np.sum(intensity[i])) * dx_um * dy_um for i in range(nz)], dtype=float
    )
    if float(np.max(raw_integrals)) > 0.0:
        raw_captured_power_fraction = raw_integrals / float(np.max(raw_integrals))
    else:
        raw_captured_power_fraction = np.full(nz, np.nan, dtype=float)

    fluence_zyx = np.empty_like(intensity, dtype=float)
    transverse_energy_by_z = np.empty(nz, dtype=float)
    peak_fluence_by_z = np.empty(nz, dtype=float)

    for i in range(nz):
        plane_I = intensity[i]
        if float(np.sum(plane_I)) <= 0.0:
            raise InvalidOpticalFieldError(
                f"z-plane index {i} has zero transverse intensity integral; "
                "cannot normalise to fluence."
            )
        f_plane = scale_intensity_to_fluence_j_cm2(plane_I, dx_um, dy_um, energy)
        fluence_zyx[i] = f_plane
        transverse_energy_by_z[i] = integrated_energy_uJ_from_fluence(f_plane, dx_um, dy_um)
        peak_fluence_by_z[i] = float(np.max(f_plane))

    # Propagation drift from raw integrals (max-min relative to max).
    finite = raw_integrals[np.isfinite(raw_integrals)]
    if finite.size and float(np.max(finite)) > 0.0:
        drift = float((np.max(finite) - np.min(finite)) / np.max(finite))
    else:
        drift = float("nan")

    peak_plane_idx = int(np.argmax(peak_fluence_by_z))
    peak_z_um = float(stack.z_um[peak_plane_idx])

    metadata = {
        "stage": STAGE,
        "field_label": stack.field_label,
        "source_status": stack.source_status,
        "normalisation": normalisation,
        "n_planes": int(nz),
        "dx_um": dx_um,
        "dy_um": dy_um,
        "peak_plane_index": peak_plane_idx,
        "raw_transverse_integral_by_z": raw_integrals.tolist(),
        "raw_captured_power_fraction_by_z": raw_captured_power_fraction.tolist(),
        **dict(stack.metadata),
    }
    return FluenceStackResult(
        fluence_zyx_j_cm2=fluence_zyx,
        pulse_energy_uJ=energy,
        transverse_energy_by_z_uJ=transverse_energy_by_z,
        raw_transverse_integral_by_z=raw_integrals,
        raw_captured_power_fraction_by_z=raw_captured_power_fraction,
        peak_fluence_by_z_j_cm2=peak_fluence_by_z,
        peak_z_um=peak_z_um,
        propagation_energy_drift_fraction=drift,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Peak intensity from a fluence result (reuses Stage 8B conversion)
# ---------------------------------------------------------------------------


def peak_intensity_from_fluence_result(
    result: FluencePlaneResult | FluenceStackResult,
    pulse_duration_fs: float,
    temporal_shape: str = "flat_top_equivalent",
) -> float:
    """Approximate peak intensity [W/cm^2] from a fluence result.

    Uses the Stage 8B :func:`peak_intensity_w_cm2` conversion (I = F_peak / tau).
    For a stack, uses the global peak over all planes.
    """
    if isinstance(result, FluencePlaneResult):
        peak_F = result.peak_fluence_j_cm2
    elif isinstance(result, FluenceStackResult):
        peak_F = float(np.max(result.peak_fluence_by_z_j_cm2))
    else:
        raise TypeError(
            f"result must be a FluencePlaneResult or FluenceStackResult; "
            f"got {type(result).__name__}."
        )
    estimate = peak_intensity_w_cm2(peak_F, pulse_duration_fs, temporal_shape)
    return float(estimate.peak_intensity_w_cm2)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def field_fluence_summary(
    plane_or_stack: OpticalFieldPlane | OpticalFieldStack,
    pulse_energy_uJ: float,
    pulse_duration_fs: float,
) -> Mapping[str, Any]:
    """Compute a flat summary dict for a plane or stack.

    Always carries ``model_status='fluence_prediction'``, a ``caveat`` string, and
    ``final_export_allowed=False``.
    """
    if isinstance(plane_or_stack, OpticalFieldPlane):
        res = scale_plane_to_fluence(plane_or_stack, pulse_energy_uJ)
        peak_I = peak_intensity_from_fluence_result(res, pulse_duration_fs)
        return {
            "kind": "plane",
            "field_label": plane_or_stack.field_label,
            "source_status": plane_or_stack.source_status,
            "pulse_energy_uJ": res.pulse_energy_uJ,
            "pulse_duration_fs": float(pulse_duration_fs),
            "dx_um": res.dx_um,
            "dy_um": res.dy_um,
            "z_um": plane_or_stack.z_um,
            "integrated_energy_uJ": res.integrated_energy_uJ,
            "energy_conservation_residual_uJ": abs(res.integrated_energy_uJ - res.pulse_energy_uJ),
            "peak_fluence_j_cm2": res.peak_fluence_j_cm2,
            "mean_fluence_j_cm2": res.mean_fluence_j_cm2,
            "peak_intensity_w_cm2": peak_I,
            "model_status": MODEL_STATUS,
            "final_export_allowed": FINAL_EXPORT_ALLOWED,
            "caveat": CAVEAT_PLANE,
        }
    if isinstance(plane_or_stack, OpticalFieldStack):
        res = scale_stack_to_fluence(plane_or_stack, pulse_energy_uJ)
        peak_I = peak_intensity_from_fluence_result(res, pulse_duration_fs)
        return {
            "kind": "stack",
            "field_label": plane_or_stack.field_label,
            "source_status": plane_or_stack.source_status,
            "pulse_energy_uJ": res.pulse_energy_uJ,
            "pulse_duration_fs": float(pulse_duration_fs),
            "n_planes": int(plane_or_stack.intensity_zyx.shape[0]),
            "dx_um": plane_or_stack.dx_um,
            "dy_um": plane_or_stack.dy_um,
            "peak_z_um": res.peak_z_um,
            "peak_fluence_j_cm2": float(np.max(res.peak_fluence_by_z_j_cm2)),
            "propagation_energy_drift_fraction": res.propagation_energy_drift_fraction,
            "raw_transverse_integral_by_z": res.raw_transverse_integral_by_z.tolist(),
            "raw_captured_power_fraction_by_z": res.raw_captured_power_fraction_by_z.tolist(),
            "max_transverse_energy_residual_uJ": float(
                np.max(np.abs(res.transverse_energy_by_z_uJ - res.pulse_energy_uJ))
            ),
            "peak_intensity_w_cm2": peak_I,
            "model_status": MODEL_STATUS,
            "final_export_allowed": FINAL_EXPORT_ALLOWED,
            "caveat": CAVEAT_STACK,
        }
    raise TypeError(
        f"plane_or_stack must be OpticalFieldPlane or OpticalFieldStack; "
        f"got {type(plane_or_stack).__name__}."
    )
