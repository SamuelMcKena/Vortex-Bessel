"""Claim-driven solver governance for the canonical optical workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, cast


OpticalFidelityMode = Literal[
    "fast_scalar_screening",
    "quantitative_vector_reference",
    "automatic_by_claim",
]

ClaimType = Literal[
    "global_transverse_morphology",
    "feature_radius",
    "ring_radius",
    "peak_location",
    "edge_sharpness",
    "ridge_width",
    "transition_width",
    "longitudinal_field",
    "polarisation_component",
    "interface_power",
    "absolute_dimensions",
    "absolute_fluence",
]

BEAM_CASES = ("G0", "B0", "V1", "V3", "H1")
FIDELITY_MODES = (
    "fast_scalar_screening",
    "quantitative_vector_reference",
    "automatic_by_claim",
)
CLAIM_TYPES = (
    "global_transverse_morphology",
    "feature_radius",
    "ring_radius",
    "peak_location",
    "edge_sharpness",
    "ridge_width",
    "transition_width",
    "longitudinal_field",
    "polarisation_component",
    "interface_power",
    "absolute_dimensions",
    "absolute_fluence",
)


@dataclass(frozen=True)
class SolverDecision:
    requested_mode: OpticalFidelityMode
    selected_objective_solver: str
    selected_interface_solver: str
    claim_type: ClaimType
    reason: str
    scalar_allowed: bool
    vector_required: bool
    calibration_required: bool
    metric_reference_solver: str


_ALWAYS_VECTOR = {"longitudinal_field", "polarisation_component"}
_H1_VECTOR = {
    "feature_radius",
    "ring_radius",
    "peak_location",
    "edge_sharpness",
    "ridge_width",
    "transition_width",
    "absolute_dimensions",
    "absolute_fluence",
}
_VORTEX_VECTOR = {"peak_location"}
_DIMENSIONAL = {
    "feature_radius",
    "ring_radius",
    "peak_location",
    "edge_sharpness",
    "ridge_width",
    "transition_width",
    "absolute_dimensions",
    "absolute_fluence",
}


def select_solver_for_claim(
    beam_case: str,
    claim_type: ClaimType,
    fidelity_mode: OpticalFidelityMode,
) -> SolverDecision:
    """Select the minimum solver permitted by the Phase 2C evidence contract."""

    beam = str(beam_case).upper()
    if beam not in BEAM_CASES:
        raise ValueError(f"unknown canonical beam case: {beam_case!r}")
    if claim_type not in CLAIM_TYPES:
        raise ValueError(f"unknown claim type: {claim_type!r}")
    if fidelity_mode not in FIDELITY_MODES:
        raise ValueError(f"unknown optical fidelity mode: {fidelity_mode!r}")

    vector_claims = set(_ALWAYS_VECTOR)
    if beam == "H1":
        vector_claims.update(_H1_VECTOR)
    if beam in {"V1", "V3"}:
        vector_claims.update(_VORTEX_VECTOR)
    vector_required = claim_type in vector_claims

    if fidelity_mode == "fast_scalar_screening" and vector_required:
        raise ValueError(
            f"{beam}/{claim_type} requires the quantitative vector reference; "
            "fast scalar screening is not eligible for this claim"
        )

    if fidelity_mode == "quantitative_vector_reference" or (
        fidelity_mode == "automatic_by_claim" and vector_required
    ):
        objective = "vector_debye"
    else:
        objective = "scalar_fft"

    interface_vector = claim_type in _ALWAYS_VECTOR or fidelity_mode == "quantitative_vector_reference"
    interface = "vector_spectral_fresnel" if interface_vector else "scalar_normal_incidence_fresnel"

    scalar_allowed = not vector_required
    calibration_required = claim_type in _DIMENSIONAL
    if claim_type == "ring_radius" and beam in {"V1", "V3"}:
        reason = (
            "Phase 2C preserved vortex ring radius under both models; the reported value must name "
            "its scalar or vector reference."
        )
    elif vector_required:
        reason = f"Phase 2C requires a vector reference for {beam} {claim_type}."
    elif objective == "vector_debye":
        reason = "The requested quantitative-vector mode explicitly selects the validated Debye reference."
    else:
        reason = f"Phase 2C permits scalar screening for {beam} {claim_type}."

    return SolverDecision(
        requested_mode=cast(OpticalFidelityMode, fidelity_mode),
        selected_objective_solver=objective,
        selected_interface_solver=interface,
        claim_type=cast(ClaimType, claim_type),
        reason=reason,
        scalar_allowed=scalar_allowed,
        vector_required=vector_required,
        calibration_required=calibration_required,
        metric_reference_solver=objective,
    )


def solver_policy_rows() -> list[dict[str, object]]:
    """Return the complete machine-readable policy, including forbidden requests."""

    rows: list[dict[str, object]] = []
    for beam in BEAM_CASES:
        for claim in CLAIM_TYPES:
            for mode in FIDELITY_MODES:
                try:
                    decision = select_solver_for_claim(
                        beam,
                        cast(ClaimType, claim),
                        cast(OpticalFidelityMode, mode),
                    )
                except ValueError as exc:
                    rows.append({
                        "beam_case": beam,
                        "claim_type": claim,
                        "requested_mode": mode,
                        "request_status": "forbidden_lower_fidelity",
                        "selected_objective_solver": "",
                        "selected_interface_solver": "",
                        "scalar_allowed": False,
                        "vector_required": True,
                        "calibration_required": claim in _DIMENSIONAL,
                        "metric_reference_solver": "",
                        "reason": str(exc),
                    })
                else:
                    row = asdict(decision)
                    row.update({"beam_case": beam, "request_status": "allowed"})
                    rows.append(row)
    return rows
