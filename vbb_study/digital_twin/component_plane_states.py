"""Component-plane field states for the Stage 8C.3R physical lab-realism path.

Each state carries a *complex* optical field (or an explicit "not represented"
marker) at one modelled optical plane, together with the honest energy
bookkeeping for that plane.  Unlike the Stage 8C.3 diagnostic stack transforms,
these states are produced *before* propagation, so perturbations applied to them
re-propagate naturally through the locked angular-spectrum engine.

Model status: optical / fluence diagnostic only.  ``final_export_allowed=False``.
No material response, absorbed energy, dose, plasma, or refractive-index change
is represented or claimed anywhere in this module.

Canonical conventions (shared with ``field_coupling``):

    field    : complex amplitude E[y, x]  (None when a plane is not modelled)
    x_um     : x coordinate array (microns), strictly monotonic ascending
    y_um     : y coordinate array (microns), strictly monotonic ascending
    intensity: |E|^2  (non-negative)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

FINAL_EXPORT_ALLOWED = False
MODEL_STATUS = "optical_prediction"


def field_power(field_yx: np.ndarray, dx_um: float, dy_um: float) -> float:
    """Return the discrete transverse power ``sum(|E|^2) * dx * dy`` (arb. * um^2)."""
    arr = np.asarray(field_yx)
    return float(np.sum(np.abs(arr) ** 2) * float(dx_um) * float(dy_um))


@dataclass(frozen=True)
class ComponentPlaneState:
    """Complex field and energy ledger at one modelled optical plane.

    ``field`` is the complex amplitude ``E[y, x]`` at this plane, or ``None`` for
    planes that the present engine cannot physically represent (these carry a
    warning and do not silently fabricate a field).
    """

    plane_name: str
    field: np.ndarray | None
    x_um: np.ndarray
    y_um: np.ndarray
    dx_um: float
    dy_um: float
    pulse_energy_before_uJ: float
    pulse_energy_after_uJ: float
    transmitted_fraction: float
    applied_components: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    representation: str = "complex_field"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.field is not None:
            arr = np.asarray(self.field)
            if arr.ndim != 2:
                raise ValueError(
                    f"{self.plane_name}: field must be 2D [y, x]; got ndim={arr.ndim}."
                )
            if not np.all(np.isfinite(arr.real)) or not np.all(np.isfinite(arr.imag)):
                raise ValueError(f"{self.plane_name}: field contains non-finite values.")
            object.__setattr__(self, "field", arr.astype(complex))
        object.__setattr__(self, "x_um", np.asarray(self.x_um, dtype=float))
        object.__setattr__(self, "y_um", np.asarray(self.y_um, dtype=float))
        object.__setattr__(self, "applied_components", tuple(self.applied_components))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def intensity(self) -> np.ndarray | None:
        if self.field is None:
            return None
        return np.abs(self.field) ** 2

    @property
    def power_before_arb(self) -> float | None:
        return None if self.field is None else field_power(self.field, self.dx_um, self.dy_um)

    def thumbnail(self) -> dict[str, Any]:
        """Compact summary used by the preview figure / report rows."""
        return {
            "plane_name": self.plane_name,
            "represented": self.field is not None,
            "representation": self.representation,
            "pulse_energy_before_uJ": float(self.pulse_energy_before_uJ),
            "pulse_energy_after_uJ": float(self.pulse_energy_after_uJ),
            "transmitted_fraction": float(self.transmitted_fraction),
            "applied_components": list(self.applied_components),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class PropagatedFieldStack:
    """Propagated optical intensity stack [z, y, x] with honest energy ledger.

    The intensity stack comes from genuine angular-spectrum propagation of the
    sample-entrance complex field.  ``sample_pulse_energy_uJ`` is the energy that
    actually survived the upstream passive losses (apertures / pupil / device
    area); it is NOT the pre-clip pulse energy.
    """

    intensity_zyx: np.ndarray
    x_um: np.ndarray
    y_um: np.ndarray
    z_um: np.ndarray
    input_pulse_energy_uJ: float
    sample_pulse_energy_uJ: float
    transmitted_fraction: float
    plane_states: tuple[ComponentPlaneState, ...]
    warnings: tuple[str, ...] = ()
    representation: str = "angular_spectrum_propagated"
    final_export_allowed: bool = FINAL_EXPORT_ALLOWED
    model_status: str = MODEL_STATUS
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arr = np.asarray(self.intensity_zyx, dtype=float)
        if arr.ndim != 3:
            raise ValueError(f"intensity_zyx must be 3D [z, y, x]; got ndim={arr.ndim}.")
        object.__setattr__(self, "intensity_zyx", arr)
        object.__setattr__(self, "x_um", np.asarray(self.x_um, dtype=float))
        object.__setattr__(self, "y_um", np.asarray(self.y_um, dtype=float))
        object.__setattr__(self, "z_um", np.asarray(self.z_um, dtype=float))
        object.__setattr__(self, "plane_states", tuple(self.plane_states))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def dx_um(self) -> float:
        return float(np.mean(np.abs(np.diff(self.x_um))))

    @property
    def dy_um(self) -> float:
        return float(np.mean(np.abs(np.diff(self.y_um))))

    @property
    def reference_plane_pulse_energy_uJ(self) -> float:
        """Stage 8C.3R.1 free-space reference-plane energy (n=1.0).

        Alias of ``sample_pulse_energy_uJ``: the genuinely transmitted energy at the
        intended sample-entrance reference plane in air.  No material model is active.
        """
        return float(self.sample_pulse_energy_uJ)

    def plane_state(self, name: str) -> ComponentPlaneState | None:
        for s in self.plane_states:
            if s.plane_name == name:
                return s
        return None
