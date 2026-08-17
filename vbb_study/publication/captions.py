"""Caption and caveat policies for publication-governed outputs.

These helpers do not decide physics. They make the model boundary explicit so
publication/export code cannot silently turn diagnostic, proxy, or legacy
artifacts into paper-ready claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

WARNING_LEVELS = {"green", "amber", "red"}

CAPTION_CONTEXTS = {
    "scalar_fast_preset",
    "scalar_publication_preset",
    "lab_realism",
    "interface_correction",
    "vector_current_lab",
    "digital_twin_vector_hexagon",
    "materials_proxy",
    "capsule_proxy",
    "advanced_hex_polygonal_discrete",
    "quicklook_diagnostic",
    "validation",
    "governance_registry",
    "diagnostic_legacy",
    "rejected",
}


@dataclass(frozen=True)
class CaptionGateResult:
    """Caption/caveat decision for one governed output family."""

    caption_text: str
    caveat_text: str
    export_allowed: bool
    warning_level: str


def _normalise_power_label(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    if text in {"pass", "marginal", "fail"}:
        return text
    return "unknown"


def _result(caption_text: str, caveat_text: str, export_allowed: bool, warning_level: str) -> CaptionGateResult:
    if warning_level not in WARNING_LEVELS:
        raise ValueError(f"warning_level must be one of {sorted(WARNING_LEVELS)}, got {warning_level!r}")
    return CaptionGateResult(
        caption_text=str(caption_text).strip(),
        caveat_text=str(caveat_text).strip(),
        export_allowed=bool(export_allowed),
        warning_level=warning_level,
    )


def caption_gate(context: str, **kwargs: Any) -> CaptionGateResult:
    """Return the standard caption/caveat decision for one output context.

    Parameters are deliberately open-ended so notebook/export code can pass the
    available row metadata without needing a separate function for every stage.
    Unknown context names are rejected.
    """

    key = str(context).strip()
    if key not in CAPTION_CONTEXTS:
        raise ValueError(f"Unknown caption context: {context!r}")

    if key == "scalar_fast_preset":
        label = _normalise_power_label(kwargs.get("propagation_power_label"))
        export_allowed = label not in {"fail", "unknown"}
        warning = "red" if label == "fail" else "amber"
        return _result(
            "Scalar Bessel/vortex diagnostic fast-preset output.",
            "Diagnostic fast preset; not publication-grade where propagation_power_label is fail.",
            export_allowed,
            warning,
        )

    if key == "scalar_publication_preset":
        label = _normalise_power_label(kwargs.get("propagation_power_label"))
        warning = "red" if label == "fail" else "amber" if label == "marginal" else "green"
        return _result(
            "Scalar Bessel/vortex publication-preset output.",
            "Publication preset, but fail/marginal propagation-power labels remain visible and must not be hidden.",
            True,
            warning,
        )

    if key == "lab_realism":
        return _result(
            "Lab-realism optical route output.",
            "Scalar lab-realistic optical model; hardware route and plane labels shown.",
            True,
            "amber",
        )

    if key == "interface_correction":
        physically_implemented = bool(kwargs.get("physically_implemented", False))
        correction_label = str(kwargs.get("interface_correction_label") or "ideal_numerical_correction")
        return _result(
            "Through-sample interface-correction diagnostic output.",
            "Ideal numerical correction unless explicitly marked physically implemented.",
            physically_implemented and correction_label != "ideal_numerical_correction",
            "amber",
        )

    if key == "vector_current_lab":
        return _result(
            "Vector/Jones current-lab route output.",
            "Current lab Case 1 is a limited SOP-encoded approximation, not true radial/azimuthal vector-beam generation.",
            True,
            "amber",
        )

    if key == "digital_twin_vector_hexagon":
        return _result(
            "Nathan vector-hexagon digital-twin route output.",
            "Exploratory air-side optical prediction; no material response, camera model, Richards-Wolf focus, or bench-calibrated 4F stop validation.",
            False,
            "amber",
        )

    if key == "materials_proxy":
        calibrated = str(kwargs.get("material_model_status", "")).strip() == "experimentally_calibrated"
        caveat = (
            "Calibrated material-response output."
            if calibrated
            else "Planning proxy only; not calibrated material damage/modification prediction."
        )
        return _result("Material-facing optical fluence output.", caveat, True, "green" if calibrated else "amber")

    if key == "capsule_proxy":
        return _result(
            "Capsule/weld-feature application-planning output.",
            "Application-planning geometry proxy; not weld success prediction.",
            True,
            "amber",
        )

    if key == "advanced_hex_polygonal_discrete":
        focal_plane_only = bool(kwargs.get("focal_plane_only", False))
        return _result(
            "Advanced hexagonal/polygonal/discrete N-fold optical output.",
            "Focal-plane pattern is not automatically propagation-stable; stability requires accepted-depth metrics.",
            True,
            "amber" if focal_plane_only else "green",
        )

    if key == "quicklook_diagnostic":
        return _result(
            "Stage 8.7 quick-look diagnostic output.",
            "Quick-look diagnostic only; visual interpolation is display-only and material response remains a planning proxy.",
            False,
            "amber",
        )

    if key == "validation":
        return _result(
            "Validation/QA output.",
            "Validation output; numerical tolerance, retained-power, and resampling caveats must remain visible.",
            True,
            "green",
        )

    if key == "governance_registry":
        return _result(
            "Figure/output governance registry.",
            "Governance artifact; final export is allowed only through the final_export_allowed registry gate.",
            True,
            "green",
        )

    if key == "diagnostic_legacy":
        return _result(
            "Diagnostic or legacy artifact.",
            "Diagnostic/legacy output; not final-export safe without explicit caveats and registry allow-listing.",
            False,
            "red",
        )

    return _result(
        "Rejected artifact.",
        "Rejected or unknown output; must not be used for publication export.",
        False,
        "red",
    )


def caveat_for_context(context: str, **kwargs: Any) -> str:
    """Return only the caveat text for a governed output context."""

    return caption_gate(context, **kwargs).caveat_text


__all__ = [
    "CAPTION_CONTEXTS",
    "WARNING_LEVELS",
    "CaptionGateResult",
    "caption_gate",
    "caveat_for_context",
]
