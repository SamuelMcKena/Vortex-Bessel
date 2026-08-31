"""Measurement policy for inverse use of the Vortex-Bessel digital twin.

The forward model contains many physical perturbations, but they are not all
observable from the same camera data.  A defensible inverse model therefore
assigns each parameter to the measurement that contains the relevant physical
information instead of putting every forward-model parameter into one generic
optimizer.

The categories below are operational rather than poster terminology:

``calibration``
    Bind from a direct bench measurement/manufacturer calibration before the
    propagation inverse problem (for example beam radius and SLM phase LUT).

``trajectory``
    Candidate parameter may be screened against first-moment beam trajectory
    through a measured z stack.

``radial_morphology``
    Candidate parameter may be screened against centered azimuthally averaged
    transverse intensity.

``longitudinal_structure``
    Candidate parameter changes the formation/evolution of the Bessel field and
    requires a z-dependent forward-model comparison; it must not be inferred
    from one arbitrary transverse plane.

``dedicated_calibration``
    A non-zero value represents device physics that requires its own calibration
    dataset rather than an intensity-only propagation fit.

This policy does not assert that a parameter is identifiable merely because a
measurement class is assigned.  Synthetic injection/recovery and uncertainty
checks remain required before a parameter is promoted into an experimental fit.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterMeasurementPolicy:
    parameter: str
    measurement: str
    physical_reason: str
    experimental_action: str


# Explicit policies for the parameters currently used by the correction study.
# Additional forward-model parameters can be added after injection/recovery tests.
POLICIES: dict[str, ParameterMeasurementPolicy] = {
    "beam_radius_scale": ParameterMeasurementPolicy(
        "beam_radius_scale",
        "calibration",
        "Gaussian radius is a direct input-plane quantity; downstream Bessel brightness is confounded by phase, filtering and aperture transmission.",
        "measure beam radius at/relayed to SLM1 and bind it before inverse propagation fitting",
    ),
    "beam_lateral_decentre_x": ParameterMeasurementPolicy(
        "beam_lateral_decentre_x",
        "trajectory",
        "input-beam translation produces a persistent first-moment displacement through the optical route",
        "screen against measured centroid trajectory and verify separation from downstream decentre terms",
    ),
    "beam_lateral_decentre_y": ParameterMeasurementPolicy(
        "beam_lateral_decentre_y",
        "trajectory",
        "input-beam translation produces a persistent first-moment displacement through the optical route",
        "screen against measured centroid trajectory and verify separation from downstream decentre terms",
    ),
    "slm1_hologram_offset_x": ParameterMeasurementPolicy(
        "slm1_hologram_offset_x",
        "trajectory",
        "registration error moves the programmed vortex relative to the illuminated beam and can shift downstream intensity",
        "screen only when camera/SLM coordinate registration is independently known",
    ),
    "slm1_hologram_offset_y": ParameterMeasurementPolicy(
        "slm1_hologram_offset_y",
        "trajectory",
        "registration error moves the programmed vortex relative to the illuminated beam and can shift downstream intensity",
        "screen only when camera/SLM coordinate registration is independently known",
    ),
    "fourf_iris_offset_x": ParameterMeasurementPolicy(
        "fourf_iris_offset_x",
        "trajectory",
        "displacing the spatial filter changes selected-order clipping asymmetrically and shifts the transmitted field",
        "screen against trajectory together with transmitted-power/order-selection evidence",
    ),
    "fourf_iris_offset_y": ParameterMeasurementPolicy(
        "fourf_iris_offset_y",
        "trajectory",
        "displacing the spatial filter changes selected-order clipping asymmetrically and shifts the transmitted field",
        "screen against trajectory together with transmitted-power/order-selection evidence",
    ),
    "fourf_iris_radius_scale": ParameterMeasurementPolicy(
        "fourf_iris_radius_scale",
        "radial_morphology",
        "iris opening changes spatial-frequency transmission and therefore centered radial beam morphology without requiring a lateral shift",
        "fit centered azimuthal transverse structure over multiple z planes",
    ),
    "fourf_lens1_despace": ParameterMeasurementPolicy(
        "fourf_lens1_despace",
        "longitudinal_structure",
        "relay despace changes phase curvature and the subsequent formation/evolution of the beam",
        "fit a z-dependent forward model; do not diagnose from one image",
    ),
    "fourf_lens2_despace": ParameterMeasurementPolicy(
        "fourf_lens2_despace",
        "longitudinal_structure",
        "relay despace changes phase curvature and the subsequent formation/evolution of the beam",
        "fit a z-dependent forward model; do not diagnose from one image",
    ),
    "axicon_lateral_decentre_x": ParameterMeasurementPolicy(
        "axicon_lateral_decentre_x",
        "trajectory",
        "translating the axicon sag relative to the selected-order beam shifts the downstream conical field",
        "fit first-moment trajectory through the Bessel region",
    ),
    "axicon_lateral_decentre_y": ParameterMeasurementPolicy(
        "axicon_lateral_decentre_y",
        "trajectory",
        "translating the axicon sag relative to the selected-order beam shifts the downstream conical field",
        "fit first-moment trajectory through the Bessel region",
    ),
    "axicon_round_tip": ParameterMeasurementPolicy(
        "axicon_round_tip",
        "longitudinal_structure",
        "tip rounding changes the axial interference structure and cannot be reduced to a lateral or radial single-plane metric",
        "compare measured and simulated longitudinal field structure over the formation region",
    ),
    "axicon_flat_tip": ParameterMeasurementPolicy(
        "axicon_flat_tip",
        "longitudinal_structure",
        "a blunt central region changes axial interference and the early Bessel formation region",
        "compare measured and simulated longitudinal field structure over the formation region",
    ),
    "slm_phase_stroke": ParameterMeasurementPolicy(
        "slm_phase_stroke",
        "dedicated_calibration",
        "phase stroke is a panel response quantity and is degenerate with unknown wavefront phase in an intensity-only propagation fit",
        "measure 1030-nm grey-to-phase response/diffraction efficiency before using a non-unity value",
    ),
    "slm_fringing_sigma_x": ParameterMeasurementPolicy(
        "slm_fringing_sigma_x",
        "dedicated_calibration",
        "the fringing model is a phenomenological convolution surrogate whose width is device-specific",
        "fit to dedicated SLM grating/diffraction data, not the Bessel correction stack",
    ),
    "slm_fringing_sigma_y": ParameterMeasurementPolicy(
        "slm_fringing_sigma_y",
        "dedicated_calibration",
        "the fringing model is a phenomenological convolution surrogate whose width is device-specific",
        "fit to dedicated SLM grating/diffraction data, not the Bessel correction stack",
    ),
}


def policy_for(parameter: str) -> ParameterMeasurementPolicy:
    try:
        return POLICIES[str(parameter)]
    except KeyError as exc:
        raise KeyError(f"no inverse-measurement policy has been validated for {parameter!r}") from exc


def parameters_for_measurement(measurement: str) -> tuple[str, ...]:
    name = str(measurement)
    return tuple(k for k, v in POLICIES.items() if v.measurement == name)
