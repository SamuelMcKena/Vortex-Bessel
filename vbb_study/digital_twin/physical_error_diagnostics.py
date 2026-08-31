"""Interpretable physical-error parameters for digital-twin diagnosis.

The residual-phase retrieval answers "what phase error best explains the measured
Bessel field?".  This module adds a complementary question first:

    "Can a low-dimensional physical bench error explain a substantial part of
    the measured multi-plane intensity signature?"

It maps named scalar diagnostic parameters onto the existing physical
``SystemErrorConfig`` at the plane where the error occurs.  These parameters can
therefore be swept through the same forward model and fitted with
``physical_parameter_inference.fit_scalar_parameter``.

Important scope
---------------
* A fitted value is a model-based diagnostic estimate, not automatically a
  calibrated metrology result.
* Parameter degeneracy must be checked before claiming unique identification.
* The diagnostic layer must not bypass the q=20 hardware gates.  Any remaining
  residual can be handed to the calibrated phase-retrieval/correction path.
* Values are deliberately expressed in human-facing engineering units so the
  resulting report can say, for example, ``axicon x-decentre = +300 um``.
"""

from __future__ import annotations

from dataclasses import dataclass

from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.vortex_beam_slm_errors import GaussianBeamError, SLMError
from vbb_study.digital_twin.vortex_explicit_4f import FourFError
from vbb_study.digital_twin.vortex_system_route import AxiconError, SystemErrorConfig


@dataclass(frozen=True)
class PhysicalDiagnosticParameter:
    name: str
    unit: str
    physical_plane: str
    description: str
    recommended_screen: tuple[float, ...]


CATALOG: dict[str, PhysicalDiagnosticParameter] = {
    "axicon_decentre_x_um": PhysicalDiagnosticParameter(
        "axicon_decentre_x_um", "um", "physical axicon",
        "lateral x displacement of the axicon relative to the laboratory beam axis",
        (-500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500),
    ),
    "axicon_decentre_y_um": PhysicalDiagnosticParameter(
        "axicon_decentre_y_um", "um", "physical axicon",
        "lateral y displacement of the axicon relative to the laboratory beam axis",
        (-500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500),
    ),
    "input_pointing_x_mrad": PhysicalDiagnosticParameter(
        "input_pointing_x_mrad", "mrad", "input beam before SLM1",
        "input-beam pointing angle about laboratory x",
        (-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0),
    ),
    "input_pointing_y_mrad": PhysicalDiagnosticParameter(
        "input_pointing_y_mrad", "mrad", "input beam before SLM1",
        "input-beam pointing angle about laboratory y",
        (-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0),
    ),
    "slm1_registration_x_um": PhysicalDiagnosticParameter(
        "slm1_registration_x_um", "um", "SLM1 commanded hologram",
        "x offset of the SLM1 programmed phase coordinates",
        (-400, -300, -200, -100, 0, 100, 200, 300, 400),
    ),
    "slm1_registration_y_um": PhysicalDiagnosticParameter(
        "slm1_registration_y_um", "um", "SLM1 commanded hologram",
        "y offset of the SLM1 programmed phase coordinates",
        (-400, -300, -200, -100, 0, 100, 200, 300, 400),
    ),
    "fourf_iris_offset_x_um": PhysicalDiagnosticParameter(
        "fourf_iris_offset_x_um", "um", "4F Fourier-plane iris",
        "x offset of the physical +1-order iris from its nominal selected-order centre",
        (-300, -225, -150, -75, 0, 75, 150, 225, 300),
    ),
    "fourf_iris_offset_y_um": PhysicalDiagnosticParameter(
        "fourf_iris_offset_y_um", "um", "4F Fourier-plane iris",
        "y offset of the physical +1-order iris from its nominal selected-order centre",
        (-300, -225, -150, -75, 0, 75, 150, 225, 300),
    ),
}


def parameter_catalog() -> dict[str, PhysicalDiagnosticParameter]:
    """Return a copy of the supported interpretable diagnostic parameter set."""
    return dict(CATALOG)


def parameter_definition(name: str) -> PhysicalDiagnosticParameter:
    try:
        return CATALOG[str(name)]
    except KeyError as exc:
        raise ValueError(f"unsupported physical diagnostic parameter: {name!r}") from exc


def system_error_config_for_parameter(name: str, value: float) -> SystemErrorConfig:
    """Map one engineering-unit parameter value onto the existing physical route."""
    key = str(name)
    v = float(value)

    if key == "axicon_decentre_x_um":
        return SystemErrorConfig(axicon=AxiconError(decentre_m=(v * 1e-6, 0.0)))
    if key == "axicon_decentre_y_um":
        return SystemErrorConfig(axicon=AxiconError(decentre_m=(0.0, v * 1e-6)))
    if key == "input_pointing_x_mrad":
        return SystemErrorConfig(beam=GaussianBeamError(pointing_rad=(v * 1e-3, 0.0)))
    if key == "input_pointing_y_mrad":
        return SystemErrorConfig(beam=GaussianBeamError(pointing_rad=(0.0, v * 1e-3)))
    if key == "slm1_registration_x_um":
        return SystemErrorConfig(slm1=SLMError(pattern_offset_m=(v * 1e-6, 0.0)))
    if key == "slm1_registration_y_um":
        return SystemErrorConfig(slm1=SLMError(pattern_offset_m=(0.0, v * 1e-6)))
    if key == "fourf_iris_offset_x_um":
        return SystemErrorConfig(fourf=FourFError(iris_offset_m=(v * 1e-6, 0.0)))
    if key == "fourf_iris_offset_y_um":
        return SystemErrorConfig(fourf=FourFError(iris_offset_m=(0.0, v * 1e-6)))

    parameter_definition(key)  # raises a consistent error
    raise AssertionError("unreachable")


def diagnostic_context() -> dict[str, float | str]:
    """Return nominal scale information useful when reporting fitted parameters."""
    manifest = canonical_hardware_manifest()
    return {
        "fourier_iris_radius_um": float(hardware_value(manifest, "fourier_iris_radius_m")) * 1e6,
        "slm_pixel_pitch_um": float(hardware_value(manifest, "slm_pixel_pitch_m")) * 1e6,
        "claim_boundary": (
            "physical-parameter fit is a model-based diagnostic estimate; retain residual-phase "
            "retrieval and independent experimental acceptance as separate layers"
        ),
    }
