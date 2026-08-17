"""Standard labels for publication-study cases.

This module is metadata only. It gives notebooks and reports a shared language
for describing what kind of structured-beam case is being run, how close it is
to lab hardware, and how much QA has been applied.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


BEAM_FAMILY = {
    "scalar_bessel": "Scalar Bessel or Bessel-Gauss beam",
    "vortex_bessel": "Vortex Bessel beam with topological charge",
    "vector_bessel": "Polarization-structured vector Bessel beam",
    "polygonal_hexagonal": "Polygonal or hexagonal structured beam",
    "discrete_nfold": "Discrete N-fold plane-wave beam",
    "material_capsule": "Material-response or capsule-feature application study",
}

MODEL_LEVEL = {
    "analytic": "Closed-form or direct analytic reference",
    "scalar_paraxial": "Scalar paraxial propagation model",
    "vector_jones": "Jones/vector field model built on scalar propagation",
    "lab_encoded": "Lab-encoded route with finite aperture/filter/SLM constraints",
    "material_proxy": "Material-response planning proxy layered on optical fields",
}

GENERATION_METHOD = {
    "ideal_target": "Ideal target field with no hardware encoding losses",
    "holographic_slm": "SLM holographic axicon/vortex route with first-order filtering",
    "physical_axicon": "Physical refractive axicon route",
    "segmented_vector_element": "Segmented or spatially varying polarization element route",
    "discrete_plane_wave": "Finite plane-wave superposition route",
    "zernike_hex_seed": "Hexagonal or polygonal seed/modulation route",
}

HARDWARE_STATUS = {
    "ideal_math": "Mathematical target only",
    "lab_proxy": "Lab-realistic proxy with modeled limitations",
    "case1_existing_hardware": "Case-1 route compatible with the current modeled lab stack",
    "proposed_hardware": "Requires a future or unmodeled element",
    "unsupported": "No credible hardware route claimed in this study",
}

MATERIAL_MODEL_STATUS = {
    "not_applicable": "No material-response layer used",
    "planning_proxy": "Planning proxy; not calibrated to measured modification outcomes",
    "calibrated_pending": "Calibration workflow exists but experimental fit is pending",
    "experiment_calibrated": "Calibrated against measured data",
}

QA_STATUS = {
    "exploratory": "Exploratory or diagnostic result",
    "smoke_checked": "Imports, paths, docs, and stage existence checked",
    "validation_pipeline_checked": "Reported by the validation pipeline when run",
    "publication_candidate": "Candidate result after validation and figure review",
    "out_of_validity": "Known sampling, hardware, or model-validity issue",
}

STANDARD_LABELS = {
    "beam_family": BEAM_FAMILY,
    "model_level": MODEL_LEVEL,
    "generation_method": GENERATION_METHOD,
    "hardware_status": HARDWARE_STATUS,
    "material_model_status": MATERIAL_MODEL_STATUS,
    "qa_status": QA_STATUS,
}


@dataclass(frozen=True)
class StudyCaseMetadata:
    """Minimal metadata block for a notebook stage or exported result."""

    beam_family: str
    model_level: str
    generation_method: str
    hardware_status: str
    material_model_status: str
    qa_status: str
    stage: str = ""
    notes: str = ""

    def asdict(self) -> dict[str, str]:
        return asdict(self)


def validate_metadata(metadata: StudyCaseMetadata | Mapping[str, str]) -> dict[str, str]:
    """Return label errors for a metadata block without mutating it."""

    values = metadata.asdict() if isinstance(metadata, StudyCaseMetadata) else dict(metadata)
    errors: dict[str, str] = {}
    for field, allowed in STANDARD_LABELS.items():
        value = values.get(field, "")
        if value not in allowed:
            errors[field] = f"{value!r} is not one of: {', '.join(sorted(allowed))}"
    return errors


__all__ = [
    "BEAM_FAMILY",
    "GENERATION_METHOD",
    "HARDWARE_STATUS",
    "MATERIAL_MODEL_STATUS",
    "MODEL_LEVEL",
    "QA_STATUS",
    "STANDARD_LABELS",
    "StudyCaseMetadata",
    "validate_metadata",
]
