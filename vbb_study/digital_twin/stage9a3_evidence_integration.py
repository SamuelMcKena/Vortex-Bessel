"""Stage 9A.3 verified bibliography and evidence-layer integration.

This module is an evidence/documentation post-processor for the Stage 9A.2
audit artifacts. It does not implement optical propagation, 4F filtering,
camera modelling, inverse correction, AI, or material-response physics.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any


STAGE = "9A.3"
FINAL_EXPORT_ALLOWED = False

SEED_BIB_REL = Path("references/incoming/structured_beam_methods_verified_seed.bib")
SEED_MAP_REL = Path("references/incoming/structured_beam_methods_verified_seed_map.json")

CLAIM_ALIAS_FROM_SEED = {
    "fused_silica_channel_future": "fused_silica_bessel_channel_or_tgv_future",
}

SPECIFIC_FUSED_SILICA_CLAIMS = [
    "fused_silica_bessel_channel_or_tgv_future",
    "fused_silica_waveguide_future",
    "fused_silica_welding_future",
]

SLM_PHASE_EVIDENCE_ITEMS = [
    "wavelength-specific phase stroke",
    "grayscale-to-phase LUT",
    "input polarisation requirement",
    "incidence-angle condition",
    "active area / pixel geometry",
    "spatial phase non-uniformity",
    "fill factor",
    "reflectivity",
]

CLAIM_SUPPORT_TEXT = {
    "angular_spectrum_or_bl_asm_propagation": "band-limited angular-spectrum propagation principle and sampling/FOV constraints",
    "finite_sampling_and_aliasing_control": "finite-grid sampling and alias-control principle for angular-spectrum propagation",
    "phase_only_slm_mask_generation": "SLM phase calibration can be spatially varying and must be calibrated",
    "phase_quantisation_and_grayscale_export": "phase command/calibration principle for spatially varying SLM response",
    "command_domain_carrier_grating": "pixelated SLM diffraction-order and zero-order phenomenon at principle level",
    "pixelated_slm_zero_order_and_unwanted_orders": "zero-order and unwanted-order behaviour of pixelated SLMs at principle level",
    "multi_plane_phase_retrieval_future": "Bessel-beam phase-front retrieval and correction as a future method principle",
    "effective_aberration_inference_future": "closed-loop/modal aberration-correction principle only",
    "zernike_or_phase_conjugate_correction_future": "Zernike/SLM aberration-correction principle only",
    "fused_silica_bessel_channel_or_tgv_future": "Bessel-beam high-aspect-ratio microchannel fabrication principle only",
    "fused_silica_welding_future": "Bessel-beam transparent/non-transparent welding principle only",
}

CLAIM_DOES_NOT_SUPPORT = {
    "fused_silica_bessel_channel_or_tgv_future": "this wavelength, pulse format, vortex configuration, sample geometry, etch result, or local fused-silica process outcome",
    "fused_silica_welding_future": "this interface preparation, focal geometry, pulse format, weld strength, feature symmetry, or local apparatus",
    "phase_only_slm_mask_generation": "calibrated 1030 nm SLM response, LUT accuracy, bench polarisation, order efficiency, or local SLM non-uniformity",
    "phase_quantisation_and_grayscale_export": "calibrated grayscale-to-phase response, physical diffraction efficiency, or local SLM non-uniformity",
    "command_domain_carrier_grating": "local SLM pixel pitch, carrier sign convention, Fourier-plane scale, stop position, order purity, lens geometry, or bench alignment",
    "pixelated_slm_zero_order_and_unwanted_orders": "local zero/+1/residual order fractions, selected-order purity, or physical 4F filtering",
}

CLAIM_CAN = {
    "phase_only_slm_mask_generation": "generate mathematically wrapped phase command maps for SLM1/SLM2 with principle-level SLM calibration literature attached",
    "phase_quantisation_and_grayscale_export": "export wrapped/quantised command maps while declaring physical phase response unresolved",
    "command_domain_carrier_grating": "generate command-domain carrier sweeps and cite the diffraction-order principle",
    "pixelated_slm_zero_order_and_unwanted_orders": "cite zero-order/unwanted-order behaviour as a future 4F/order-selection concern",
    "fused_silica_bessel_channel_or_tgv_future": "cite Bessel microchannel fabrication as a principle-level future application direction",
    "fused_silica_waveguide_future": "state only that a targeted fused-silica waveguide source search is still required",
    "fused_silica_welding_future": "cite Bessel welding as a principle-level future application direction",
    "fused_silica_application_boundary": "state only that the broad fused-silica boundary has been superseded by specific future application claims",
}

CLAIM_CANNOT = {
    "phase_only_slm_mask_generation": "claim calibrated physical phase response at 1030 nm, order efficiency, or local SLM non-uniformity correction",
    "phase_quantisation_and_grayscale_export": "claim grayscale commands produce calibrated physical phase without manufacturer and bench evidence",
    "command_domain_carrier_grating": "claim local carrier-to-Fourier-plane scaling, stop placement, selected-order purity, or direct Fourier-plane coordinates from downstream images",
    "pixelated_slm_zero_order_and_unwanted_orders": "claim an active validated order-efficiency or physical 4F model",
    "fused_silica_bessel_channel_or_tgv_future": "claim TGV/channel formation in this apparatus or any fused-silica process window",
    "fused_silica_waveguide_future": "claim waveguide writing, index change, loss, or mode quality",
    "fused_silica_welding_future": "claim weld strength, interface quality, symmetry, or local process validity",
    "fused_silica_application_boundary": "claim any fused-silica TGV/channel, waveguide, welding, or modification outcome",
}

P0_DELIVERABLE_TYPE = {
    "P0_SLM_SPEC": "manufacturer_datasheet",
    "P0_SLM_PHASE": "bench_measurement",
    "P0_LENS_GEOMETRY": "manufacturer_datasheet_and_bench_measurement",
    "P0_CARRIER_MAPPING": "first_carrier_session_image_dataset",
    "P0_FOURIER_STOP": "bench_measurement",
    "P0_CAMERA": "manufacturer_datasheet_and_bench_measurement",
    "P0_AXICON": "manufacturer_datasheet_and_bench_measurement",
    "P0_INPUT_BEAM": "bench_measurement",
}

P0_CONCRETE_DELIVERABLE = {
    "P0_SLM_SPEC": "manufacturer datasheet for actual SLM model plus bench orientation measurement",
    "P0_SLM_PHASE": "bench measurement of 1030 nm grayscale-to-phase or diffraction-efficiency response",
    "P0_LENS_GEOMETRY": "manufacturer datasheet plus bench measurement of lens positions and apertures",
    "P0_CARRIER_MAPPING": "downstream carrier-stop response dataset now; direct Fourier-plane mapping only with temporary Fourier-plane access",
    "P0_FOURIER_STOP": "bench measurement of Fourier-stop centre, radius, and adjustment convention",
    "P0_CAMERA": "manufacturer datasheet plus bench measurement of scale, orientation, and linearity",
    "P0_AXICON": "manufacturer datasheet plus bench measurement of physical axicon pose",
    "P0_INPUT_BEAM": "bench measurement of beam size, centring, ellipticity, and polarisation",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _bib_keys(text: str) -> list[str]:
    return re.findall(r"@\w+\{\s*([^,\s]+)", text)


def _load_seed(root: Path) -> tuple[str, dict[str, Any], dict[str, list[str]], list[str]]:
    seed_bib_path = root / SEED_BIB_REL
    seed_map_path = root / SEED_MAP_REL
    if not seed_bib_path.exists() or not seed_map_path.exists():
        raise FileNotFoundError("Stage 9A.3 requires the verified seed BibTeX and seed map files.")
    seed_bib = seed_bib_path.read_text(encoding="utf-8")
    seed_map = _read_json(seed_map_path)
    claim_to_keys: dict[str, list[str]] = {}
    for claim_id, keys in seed_map["claim_to_keys"].items():
        canonical_claim_id = CLAIM_ALIAS_FROM_SEED.get(claim_id, claim_id)
        claim_to_keys[canonical_claim_id] = list(keys)
    seed_keys = _bib_keys(seed_bib)
    return seed_bib, seed_map, claim_to_keys, seed_keys


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _claim_by_id(claims: list[dict[str, Any]], claim_id: str) -> dict[str, Any]:
    for claim in claims:
        if claim["claim_id"] == claim_id:
            return claim
    raise KeyError(claim_id)


def _set_claim_fields(claim: dict[str, Any], **updates: Any) -> None:
    for key, value in updates.items():
        claim[key] = value


def _apply_required_claim_reclassifications(claims: list[dict[str, Any]]) -> None:
    phase = _claim_by_id(claims, "phase_only_slm_mask_generation")
    _set_claim_fields(
        phase,
        claim_type="command_model",
        current_status="implemented_command_phase_model",
        physics_status="command_phase_model_unverified_physical_response",
        evidence_category="multi_layer_evidence",
        evidence_status="verified_principle_needs_manufacturer_and_bench_evidence",
        manufacturer_evidence_required=list(SLM_PHASE_EVIDENCE_ITEMS),
        bench_evidence_required=[
            "1030 nm grayscale-to-phase calibration",
            "diffraction-efficiency or interferometric phase response",
            "zero/order power split on the local bench",
        ],
        assumption_or_placeholder=(
            "Implemented command-phase model only; physical phase response at 1030 nm "
            "remains unverified until manufacturer specifications and bench calibration are attached."
        ),
        known_limitations=[
            "mathematical phase command is not calibrated physical SLM phase response",
            "local polarisation, incidence angle, active area, pixel geometry, and spatial non-uniformity are unresolved",
        ],
        what_this_does_not_prove=[
            "calibrated physical SLM phase response",
            "local order efficiency",
            "local zero-order suppression",
        ],
        validation_required=[
            "manufacturer SLM specification pack",
            "1030 nm grayscale-to-phase or diffraction-efficiency bench calibration",
        ],
        publication_claim_readiness="bench_validation_required",
    )

    quant = _claim_by_id(claims, "phase_quantisation_and_grayscale_export")
    _set_claim_fields(
        quant,
        claim_type="command_model",
        current_status="implemented_command_export_model",
        physics_status="command_quantisation_unverified_physical_response",
        evidence_category="multi_layer_evidence",
        evidence_status="verified_principle_needs_manufacturer_and_bench_evidence",
        manufacturer_evidence_required=list(SLM_PHASE_EVIDENCE_ITEMS),
        bench_evidence_required=[
            "1030 nm grayscale-to-phase LUT calibration",
            "displayed command map verification",
        ],
        assumption_or_placeholder=(
            "Wrapped and quantised command maps are generated, but grayscale-to-physical-phase "
            "response at 1030 nm is not yet calibrated."
        ),
        known_limitations=[
            "bit-depth export does not imply physical phase accuracy",
            "spatial phase non-uniformity and local LUT are unresolved",
        ],
        what_this_does_not_prove=[
            "calibrated grayscale-to-phase response",
            "physical diffraction efficiency",
        ],
        validation_required=[
            "manufacturer LUT or phase-stroke data",
            "bench grayscale-to-phase calibration",
        ],
        publication_claim_readiness="bench_validation_required",
    )

    carrier = _claim_by_id(claims, "command_domain_carrier_grating")
    _set_claim_fields(
        carrier,
        current_status="implemented_command_carrier_model",
        physics_status="command_domain_model_needs_bench_mapping",
        evidence_category="multi_layer_evidence",
        evidence_status="verified_principle_needs_bench_mapping",
        manufacturer_evidence_required=[
            "SLM display dimensions",
            "SLM orientation",
            "pixel pitch or active command area",
        ],
        bench_evidence_required=[
            "downstream response versus carrier command cycles and, separately, direct Fourier-plane order displacement only if temporary Fourier-plane access exists",
            "carrier sign convention",
            "selected-order purity and residual-order power",
        ],
        assumption_or_placeholder=(
            "Carrier units remain command cycles across the displayed area. The installed "
            "downstream camera measures empirical response only; direct Fourier-plane order "
            "positions require temporary access at or conjugate to the Fourier plane."
        ),
        what_this_does_not_prove=[
            "direct Fourier-plane coordinates from downstream images",
            "stop position",
            "selected-order purity",
            "lens geometry or bench alignment",
        ],
    )

    pixelated = _claim_by_id(claims, "pixelated_slm_zero_order_and_unwanted_orders")
    _set_claim_fields(
        pixelated,
        evidence_category="multi_layer_evidence",
        evidence_status="verified_principle_future_bench_required",
        known_limitations=[
            "literature supports the phenomenon only",
            "current code records no order efficiency and no zero-order rejection",
            "real zero/+1/residual-order fractions require bench measurement",
        ],
        what_this_does_not_prove=[
            "local order power fractions",
            "selected-order purity",
            "active physical 4F filter behaviour",
        ],
        publication_claim_readiness="bench_validation_required",
    )


def _make_fused_claims() -> list[dict[str, Any]]:
    base_paths = ["configs/materials/fused_silica_evidence_template.json", "docs/44_code_to_evidence_audit.md"]
    return [
        {
            "claim_id": "fused_silica_bessel_channel_or_tgv_future",
            "title": "Future fused-silica Bessel channel / TGV-like processing path",
            "project_component": "future fused-silica application branch",
            "implementation_paths": base_paths,
            "route_relevance": "future_planned",
            "claim_type": "material_application_hypothesis",
            "current_status": "planned_future",
            "physics_status": "future_not_implemented",
            "evidence_category": "multi_layer_evidence",
            "evidence_status": "verified_principle_needs_material_bench_evidence",
            "literature_reference_keys": [],
            "manufacturer_evidence_required": [],
            "bench_evidence_required": [
                "fused-silica sample grade/thickness/surface condition",
                "laser pulse conditions and calibrated optical field at sample entry",
                "channel/TGV morphology after processing",
                "etching or through-hole confirmation where relevant",
            ],
            "assumption_or_placeholder": "",
            "known_limitations": [
                "principle-level Bessel microchannel literature only",
                "no local fused-silica process model or outcome measurement",
            ],
            "what_this_does_not_prove": [
                "this wavelength or pulse format",
                "vortex-Bessel channel formation",
                "sample geometry validity",
                "TGV yield or etch selectivity",
            ],
            "validation_required": [
                "targeted fused-silica material/process literature review",
                "bench process trial",
                "microscopy and metrology",
            ],
            "publication_claim_readiness": "not_claimable",
            "implemented_route_components": [],
            "final_export_allowed": FINAL_EXPORT_ALLOWED,
            "research_question": "Can the calibrated apparatus produce Bessel-assisted channel/TGV-like features in fused silica?",
            "optical_target": "calibrated Bessel or vortex-Bessel optical field at the fused-silica sample entry plane",
            "material_process_unknowns": [
                "modification threshold",
                "etch selectivity",
                "crack formation",
                "debris/redeposition",
                "aspect ratio and taper",
            ],
            "required_material_metrology": ["optical microscopy", "SEM or confocal profile", "etch/through-hole verification"],
            "claim_boundary": "Bhuyan 2010 supports Bessel microchannel principle only; it does not validate this bench or process.",
            "literature_support_status": "verified_principle_only",
        },
        {
            "claim_id": "fused_silica_waveguide_future",
            "title": "Future fused-silica waveguide writing path",
            "project_component": "future fused-silica application branch",
            "implementation_paths": base_paths,
            "route_relevance": "future_planned",
            "claim_type": "material_application_hypothesis",
            "current_status": "planned_future",
            "physics_status": "future_not_implemented",
            "evidence_category": "unsupported_future_work",
            "evidence_status": "targeted_search_required",
            "literature_reference_keys": [],
            "literature_search_keys": ["search_fused_silica_waveguide_direct_source"],
            "manufacturer_evidence_required": [],
            "bench_evidence_required": [
                "fused-silica sample grade/thickness/surface condition",
                "written-track microscopy",
                "index-change or mode-guidance measurement",
                "propagation-loss measurement",
            ],
            "assumption_or_placeholder": "",
            "known_limitations": [
                "no verified direct waveguide source in the seed bibliography",
                "no local waveguide process model or metrology",
            ],
            "what_this_does_not_prove": [
                "waveguide writing",
                "index change",
                "mode quality",
                "loss or coupling efficiency",
            ],
            "validation_required": [
                "targeted primary-source search",
                "bench writing trial",
                "mode/loss metrology",
            ],
            "publication_claim_readiness": "not_claimable",
            "implemented_route_components": [],
            "final_export_allowed": FINAL_EXPORT_ALLOWED,
            "research_question": "Can a future calibrated optical route write useful waveguides in fused silica?",
            "optical_target": "application-specific intensity/fluence distribution for waveguide writing, not defined yet",
            "material_process_unknowns": [
                "index-change regime",
                "damage threshold",
                "mode confinement",
                "propagation loss",
                "stress/birefringence",
            ],
            "required_material_metrology": ["near-field/far-field mode image", "propagation loss", "microscopy", "index-profile estimate"],
            "claim_boundary": "No verified direct waveguide source is integrated; this branch remains unresolved.",
            "literature_support_status": "targeted_search_required",
        },
        {
            "claim_id": "fused_silica_welding_future",
            "title": "Future fused-silica / glass welding path",
            "project_component": "future fused-silica application branch",
            "implementation_paths": base_paths + ["vbb_study/equations/capsule_geometry.py"],
            "route_relevance": "future_planned",
            "claim_type": "material_application_hypothesis",
            "current_status": "planned_future",
            "physics_status": "future_not_implemented",
            "evidence_category": "multi_layer_evidence",
            "evidence_status": "verified_principle_needs_material_bench_evidence",
            "literature_reference_keys": [],
            "manufacturer_evidence_required": [],
            "bench_evidence_required": [
                "interface preparation and contact condition",
                "pulse conditions and calibrated optical field at interface",
                "weld cross-section microscopy",
                "mechanical or leak/strength test where relevant",
            ],
            "assumption_or_placeholder": "",
            "known_limitations": [
                "principle-level Bessel welding literature only",
                "capsule/weld branch remains geometry-only",
            ],
            "what_this_does_not_prove": [
                "interface preparation validity",
                "local focal geometry",
                "weld strength",
                "feature symmetry",
            ],
            "validation_required": [
                "targeted glass-welding literature review",
                "bench welding trial",
                "cross-section and strength/leak metrology",
            ],
            "publication_claim_readiness": "not_claimable",
            "implemented_route_components": [],
            "final_export_allowed": FINAL_EXPORT_ALLOWED,
            "research_question": "Can a future calibrated Bessel-like route weld transparent/non-transparent or glass interfaces?",
            "optical_target": "calibrated elongated Bessel-like focus at a prepared interface",
            "material_process_unknowns": [
                "interface absorption/contact",
                "heat affected zone",
                "void/crack formation",
                "joint strength",
                "feature symmetry",
            ],
            "required_material_metrology": ["cross-section microscopy", "void/crack inspection", "joint strength or leak test"],
            "claim_boundary": "Zhang 2018 supports Bessel welding principle only; it does not validate this apparatus or interface.",
            "literature_support_status": "verified_principle_only",
        },
    ]


def _split_fused_silica_claims(claims: list[dict[str, Any]]) -> None:
    broad = _claim_by_id(claims, "fused_silica_application_boundary")
    _set_claim_fields(
        broad,
        current_status="superseded_by_specific_application_claims",
        evidence_status="superseded_by_specific_application_claims",
        literature_reference_keys=[],
        manufacturer_evidence_required=[],
        bench_evidence_required=[],
        known_limitations=[
            "broad fused-silica boundary is retained only for backwards compatibility",
            "specific channel/TGV, waveguide, and welding claims carry application evidence requirements",
        ],
        validation_required=[],
        publication_claim_readiness="not_claimable",
        superseded_by=list(SPECIFIC_FUSED_SILICA_CLAIMS),
    )
    existing = {claim["claim_id"]: claim for claim in claims}
    for new_claim in _make_fused_claims():
        if new_claim["claim_id"] in existing:
            existing[new_claim["claim_id"]].clear()
            existing[new_claim["claim_id"]].update(new_claim)
        else:
            claims.append(new_claim)


def _add_evidence_layers(claim: dict[str, Any], verified_keys: list[str], search_keys: list[str]) -> None:
    layers: list[dict[str, Any]] = []
    if verified_keys:
        layers.append(
            {
                "layer_type": "peer_reviewed_primary_source",
                "status": "verified",
                "reference_keys": verified_keys,
                "what_it_supports": CLAIM_SUPPORT_TEXT.get(claim["claim_id"], "method principle only"),
                "what_it_does_not_support": CLAIM_DOES_NOT_SUPPORT.get(
                    claim["claim_id"],
                    "specific bench validation, manufacturer specifications, local alignment, or material/process outcome",
                ),
            }
        )
    elif search_keys:
        layers.append(
            {
                "layer_type": "targeted_literature_search",
                "status": "targeted_search_required",
                "search_keys": search_keys,
                "what_it_supports": "unresolved until a verified source is committed",
                "what_it_does_not_support": "any current publication claim",
            }
        )
    if claim.get("manufacturer_evidence_required"):
        layers.append(
            {
                "layer_type": "manufacturer_specification",
                "status": "needs_manufacturer_data",
                "required_items": claim["manufacturer_evidence_required"],
            }
        )
    if claim.get("bench_evidence_required"):
        layers.append(
            {
                "layer_type": "bench_measurement",
                "status": "needs_bench_measurement",
                "required_items": claim["bench_evidence_required"],
            }
        )
    if claim.get("evidence_status") == "assumption_declared" or claim.get("physics_status") == "diagnostic_placeholder":
        layers.append(
            {
                "layer_type": "declared_assumption_or_placeholder",
                "status": "declared",
                "statement": claim.get("assumption_or_placeholder") or "diagnostic placeholder; not validated",
            }
        )
    claim["evidence_layers"] = layers
    claim["evidence_layer_summary"] = {
        "literature_supported_principle": "verified" if verified_keys else ("targeted_search_required" if search_keys else "not_applicable"),
        "numerically_implemented_representation": (
            "implemented"
            if claim.get("current_status") in {"implemented_active", "implemented_command_phase_model", "implemented_command_export_model", "implemented_command_carrier_model", "implemented_benchmark_only"}
            else "not_implemented"
        ),
        "manufacturer_specified_component": "needs_manufacturer_data" if claim.get("manufacturer_evidence_required") else "not_required",
        "bench_validated_representation": "needs_bench_measurement" if claim.get("bench_evidence_required") else "not_bench_validated",
    }


def _claim_next_action(claim: dict[str, Any]) -> str:
    if claim.get("manufacturer_evidence_required"):
        return "Attach manufacturer specification and then run the required bench measurement."
    if claim.get("bench_evidence_required"):
        return "Acquire the required bench measurement before narrowing the claim."
    if claim.get("literature_status") == "targeted_search_required":
        return "Complete targeted source search without filling gaps from generic papers."
    if claim.get("current_status") == "superseded_by_specific_application_claims":
        return "Use the specific fused-silica application claims instead."
    return "Retain as documented boundary until a narrower evidence need is identified."


def integrate_claim_registry(registry: dict[str, Any], seed_map: dict[str, Any], claim_to_keys: dict[str, list[str]]) -> dict[str, Any]:
    registry = copy.deepcopy(registry)
    registry["stage"] = STAGE
    registry["purpose"] = "verified bibliography integration and multi-layer evidence claim-boundary register"
    registry["claim_boundary"] = (
        "evidence integration only; no new optical propagation, no physical 4F, "
        "no camera model, no inverse correction, no AI, no material response"
    )
    registry["evidence_layer_vocabulary"] = [
        "peer_reviewed_primary_source",
        "targeted_literature_search",
        "manufacturer_specification",
        "bench_measurement",
        "declared_assumption_or_placeholder",
        "material_process_metrology",
    ]
    registry["machine_readable_claim_distinctions"] = [
        "literature_supported_principle",
        "numerically_implemented_representation",
        "manufacturer_specified_component",
        "bench_validated_representation",
    ]
    registry["seed_scope"] = seed_map["seed_scope"]
    registry["verified_seed_claim_map"] = claim_to_keys

    claims = registry["claims"]
    _apply_required_claim_reclassifications(claims)
    _split_fused_silica_claims(claims)

    for claim in claims:
        old_refs = list(claim.get("literature_reference_keys", []))
        existing_search_keys = list(claim.get("literature_search_keys", []))
        search_keys = _unique(existing_search_keys + [key for key in old_refs if key.startswith("search_")])
        verified_keys = claim_to_keys.get(claim["claim_id"], [])
        claim["literature_reference_keys"] = verified_keys
        claim["verified_literature_keys"] = verified_keys
        if search_keys:
            claim["literature_search_keys"] = search_keys
        claim["literature_status"] = "verified_principle_only" if verified_keys else ("targeted_search_required" if search_keys else claim.get("literature_support_status", "not_applicable"))
        claim["manufacturer_specification_status"] = "needs_manufacturer_data" if claim.get("manufacturer_evidence_required") else "not_required"
        claim["bench_evidence_status"] = "needs_bench_measurement" if claim.get("bench_evidence_required") else "not_bench_validated"
        claim["local_bench_validation_status"] = "not_bench_validated"
        claim["what_can_currently_be_claimed"] = CLAIM_CAN.get(
            claim["claim_id"],
            "only the implemented diagnostic or method-boundary statement described by current_status and evidence layers",
        )
        claim["what_cannot_currently_be_claimed"] = CLAIM_CANNOT.get(
            claim["claim_id"],
            "local bench validation, manufacturer calibration, or material/process outcome unless separately evidenced",
        )
        claim["next_action"] = _claim_next_action(claim)
        _add_evidence_layers(claim, verified_keys, search_keys)

    registry["verified_reference_key_to_claim_ids"] = {
        key: sorted(claim["claim_id"] for claim in claims if key in claim.get("literature_reference_keys", []))
        for keys in claim_to_keys.values()
        for key in keys
    }
    return registry


def update_research_backlog(backlog: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    backlog = copy.deepcopy(backlog)
    backlog["stage"] = STAGE
    backlog["status"] = "verified_seed_integrated_open_validation_layers"
    claims = {claim["claim_id"]: claim for claim in registry["claims"]}
    for item in backlog["items"]:
        if "fused_silica_application_boundary" in item["linked_claim_ids"]:
            mapped = {
                "P3_FS_REGIMES": ["fused_silica_bessel_channel_or_tgv_future", "fused_silica_welding_future"],
                "P3_FS_INTERFACE": list(SPECIFIC_FUSED_SILICA_CLAIMS),
                "P3_TGV": ["fused_silica_bessel_channel_or_tgv_future"],
                "P3_WAVEGUIDE": ["fused_silica_waveguide_future"],
                "P3_WELDING": ["fused_silica_welding_future", "capsule_or_weld_feature_geometry_branch"],
            }.get(item["research_id"])
            if mapped:
                item["linked_claim_ids"] = mapped
        item["linked_code_paths"] = _unique(
            [
                path
                for claim_id in item["linked_claim_ids"]
                if claim_id in claims
                for path in claims[claim_id].get("implementation_paths", [])
            ]
        )[:8]
        layers = []
        if item.get("literature_needed"):
            layers.append("peer_reviewed_primary_source")
        if item.get("manufacturer_data_needed"):
            layers.append("manufacturer_specification")
        if item.get("bench_data_needed"):
            layers.append("bench_measurement")
        for claim_id in item["linked_claim_ids"]:
            for layer in claims.get(claim_id, {}).get("evidence_layers", []):
                layers.append(layer["layer_type"])
        item["required_evidence_layers"] = _unique(layers) or ["explicit_legacy_or_quarantine_label"]
        item["concrete_next_deliverable"] = P0_CONCRETE_DELIVERABLE.get(item["research_id"], item["expected_deliverable"])
        item["next_deliverable_type"] = P0_DELIVERABLE_TYPE.get(item["research_id"], "documented_evidence_artifact")
    return backlog


def _search_entry(claim_id: str, question: str, queries: list[str], source_types: list[str], *, keys: list[str] | None = None) -> dict[str, Any]:
    verified = bool(keys)
    return {
        "claim_id": claim_id,
        "research_question": question,
        "exact_search_queries": queries,
        "target_journals_or_source_types": source_types,
        "required_evidence_standard": "verified primary source or authoritative textbook/standard; DOI/publisher metadata must be checked before BibTeX entry is added",
        "candidate_source_status": "verified_seed_integrated" if verified else "targeted_search_required",
        "verified_reference_keys": keys or [],
        "why_this_reference_is_needed": "Literature supports method principles only and does not validate the local bench.",
    }


def update_literature_search_plan(search_plan: dict[str, Any], claim_to_keys: dict[str, list[str]]) -> dict[str, Any]:
    entries = [
        _search_entry("angular_spectrum_or_bl_asm_propagation", "What sampling/FOV rules are accepted for BL-ASM/ASM finite-aperture propagation?", ["band limited angular spectrum propagation aliasing finite aperture"], ["peer-reviewed optics methods paper"], keys=claim_to_keys.get("angular_spectrum_or_bl_asm_propagation")),
        _search_entry("finite_sampling_and_aliasing_control", "What source supports finite-grid sampling and alias control for angular-spectrum propagation?", ["band limited angular spectrum method sampling aliasing"], ["peer-reviewed optics methods paper"], keys=claim_to_keys.get("finite_sampling_and_aliasing_control")),
        _search_entry("phase_only_slm_mask_generation", "What source supports SLM phase calibration and spatially varying phase response?", ["LCOS SLM phase calibration spatially varying phase response"], ["peer-reviewed SLM calibration paper"], keys=claim_to_keys.get("phase_only_slm_mask_generation")),
        _search_entry("phase_quantisation_and_grayscale_export", "What source supports grayscale command/phase calibration boundaries for SLMs?", ["spatial light modulator grayscale phase calibration quantisation"], ["peer-reviewed SLM calibration paper"], keys=claim_to_keys.get("phase_quantisation_and_grayscale_export")),
        _search_entry("command_domain_carrier_grating", "How does a pixelated SLM carrier/grating create diffraction orders and zero-order artifacts?", ["pixelated spatial light modulator zero order carrier grating"], ["peer-reviewed SLM diffraction paper"], keys=claim_to_keys.get("command_domain_carrier_grating")),
        _search_entry("pixelated_slm_zero_order_and_unwanted_orders", "How do pixelated SLMs distribute zero, first, and unwanted orders?", ["pixelated spatial light modulator zero order Fourier filtering"], ["peer-reviewed SLM diffraction paper"], keys=claim_to_keys.get("pixelated_slm_zero_order_and_unwanted_orders")),
        _search_entry("physical_fourier_filtering_future_route", "What physical thin-lens/finite-aperture 4F model is appropriate for a measured SLM Fourier plane?", ["finite aperture 4F optical system scalar diffraction Fourier filter"], ["peer-reviewed Fourier optics paper", "authoritative Fourier optics textbook"]),
        _search_entry("physical_axicon_bessel_conversion", "How do physical axicon decentre, tilt, apex and finite aperture affect Bessel beams?", ["physical axicon decentration tilt Bessel beam tolerance"], ["peer-reviewed axicon tolerance paper", "manufacturer axicon datasheet"]),
        _search_entry("physical_axicon_aperture_and_decentre", "What source quantifies axicon aperture/decentre tolerances for Bessel beam quality?", ["axicon finite aperture decentre Bessel beam quality"], ["peer-reviewed axicon tolerance paper"]),
        _search_entry("camera_z_stack_acquisition", "What metrology standards support camera z-stack acquisition of annular/Bessel beams?", ["camera z stack Bessel beam profiling metrology"], ["peer-reviewed beam metrology paper"]),
        _search_entry("camera_coordinate_calibration", "How should camera scale/orientation/reference-plane calibration be validated for beam profiling?", ["camera calibration beam profiling magnification orientation"], ["peer-reviewed metrology paper", "camera manufacturer documentation"]),
        _search_entry("multi_plane_phase_retrieval_future", "Which multi-plane phase retrieval methods are suitable for Bessel intensity z-stacks?", ["multi plane phase retrieval Bessel beam intensity z stack"], ["peer-reviewed phase retrieval paper"], keys=claim_to_keys.get("multi_plane_phase_retrieval_future")),
        _search_entry("effective_aberration_inference_future", "What methods support effective aberration inference/correction from measured beam errors?", ["closed loop aberration correction Zernike wavefront sensor"], ["peer-reviewed adaptive optics paper"], keys=claim_to_keys.get("effective_aberration_inference_future")),
        _search_entry("zernike_or_phase_conjugate_correction_future", "What methods support Zernike or SLM correction of measured aberrations?", ["SLM aberration correction Shack Hartmann sensor holographic correction"], ["peer-reviewed adaptive optics paper"], keys=claim_to_keys.get("zernike_or_phase_conjugate_correction_future")),
        _search_entry("neural_fast_estimator_future", "What evidence standards are required for synthetic-to-real neural optical estimators?", ["synthetic to real optical alignment neural estimator uncertainty"], ["peer-reviewed ML uncertainty paper"]),
        _search_entry("fused_silica_bessel_channel_or_tgv_future", "What does Bessel-beam microchannel literature support, and what local evidence remains required?", ["femtosecond laser Bessel beam microchannel fused silica"], ["peer-reviewed ultrafast processing paper"], keys=claim_to_keys.get("fused_silica_bessel_channel_or_tgv_future")),
        _search_entry("fused_silica_waveguide_future", "What directly verified source should support Bessel/vortex-Bessel waveguide writing in fused silica?", ["ultrafast laser fused silica waveguide Bessel beam writing"], ["peer-reviewed waveguide-writing paper"]),
        _search_entry("fused_silica_welding_future", "What does Bessel-beam glass welding literature support, and what local evidence remains required?", ["femtosecond laser Bessel beam welding transparent materials"], ["peer-reviewed glass welding paper"], keys=claim_to_keys.get("fused_silica_welding_future")),
        _search_entry("legacy_crznse_material_proxy_branch", "Which Cr:ZnSe/ZnSe-family material proxy assumptions are material-specific and non-transferable?", ["CrZnSe femtosecond laser waveguide writing threshold", "ZnSe femtosecond laser modification threshold"], ["peer-reviewed material processing paper"]),
    ]
    return {
        "stage": STAGE,
        "status": "verified_seed_integrated_plus_targeted_open_searches",
        "entries": entries,
    }


def update_evidence_registers(manufacturer: dict[str, Any], bench: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    manufacturer = copy.deepcopy(manufacturer)
    bench = copy.deepcopy(bench)
    manufacturer["stage"] = STAGE
    bench["stage"] = STAGE
    for entry in manufacturer["entries"]:
        if entry["evidence_id"] in {"M_SLM1_SPEC", "M_SLM2_SPEC"}:
            entry["needed_fields"] = _unique(entry["needed_fields"] + SLM_PHASE_EVIDENCE_ITEMS)
        if entry["evidence_id"] == "M_AXICON":
            entry["required_for_claim_ids"] = _unique(entry["required_for_claim_ids"] + ["physical_axicon_aperture_and_decentre"])
    bench["entries"].extend(
        [
            {
                "evidence_id": "B_SLM_PHASE_RESPONSE",
                "measurement": "1030 nm SLM grayscale-to-phase and diffraction-efficiency calibration",
                "required_for_claim_ids": ["phase_only_slm_mask_generation", "phase_quantisation_and_grayscale_export"],
                "current_value_state": "unknown",
                "raw_data_path": None,
                "derived_data_path": None,
                "ready": False,
            },
            {
                "evidence_id": "B_ORDER_POWER_SPLIT",
                "measurement": "zero/+1/residual order fractions and selected-order purity",
                "required_for_claim_ids": ["command_domain_carrier_grating", "pixelated_slm_zero_order_and_unwanted_orders"],
                "current_value_state": "unknown",
                "raw_data_path": None,
                "derived_data_path": None,
                "ready": False,
            },
            {
                "evidence_id": "B_FUSED_SILICA_CHANNEL_TGV",
                "measurement": "future fused-silica channel/TGV morphology and material metrology",
                "required_for_claim_ids": ["fused_silica_bessel_channel_or_tgv_future"],
                "current_value_state": "unknown",
                "raw_data_path": None,
                "derived_data_path": None,
                "ready": False,
            },
            {
                "evidence_id": "B_FUSED_SILICA_WAVEGUIDE",
                "measurement": "future fused-silica waveguide mode/loss/material metrology",
                "required_for_claim_ids": ["fused_silica_waveguide_future"],
                "current_value_state": "unknown",
                "raw_data_path": None,
                "derived_data_path": None,
                "ready": False,
            },
            {
                "evidence_id": "B_FUSED_SILICA_WELDING",
                "measurement": "future fused-silica/glass weld cross-section and strength/leak metrology",
                "required_for_claim_ids": ["fused_silica_welding_future"],
                "current_value_state": "unknown",
                "raw_data_path": None,
                "derived_data_path": None,
                "ready": False,
            },
        ]
    )
    seen: set[str] = set()
    unique_entries = []
    for entry in bench["entries"]:
        if entry["evidence_id"] in seen:
            continue
        seen.add(entry["evidence_id"])
        unique_entries.append(entry)
    bench["entries"] = unique_entries
    return manufacturer, bench


def update_fused_silica_template(template: dict[str, Any]) -> dict[str, Any]:
    template = copy.deepcopy(template)
    template["stage"] = STAGE
    template["application_claims"] = {
        "channel_or_tgv": "fused_silica_bessel_channel_or_tgv_future",
        "waveguide": "fused_silica_waveguide_future",
        "welding": "fused_silica_welding_future",
    }
    template["properties"]["weld_strength"] = {"value": None, "unit": None, "evidence_status": "needs_bench_measurement"}
    template["properties"]["channel_or_tgv_morphology"] = {"value": None, "unit": None, "evidence_status": "needs_bench_measurement"}
    return template


def make_references_readme(registry: dict[str, Any]) -> str:
    key_rows = "\n".join(
        f"- `{key}` -> {', '.join(claim_ids)}"
        for key, claim_ids in sorted(registry["verified_reference_key_to_claim_ids"].items())
    )
    return f"""# References For Structured-Beam Methods

This directory stores verified bibliography records only.

- The bibliography supports method principles, not automatic validation of this bench.
- DOI/publisher metadata must be verified before a new entry is committed.
- Reviews may support orientation, but primary papers are preferred for implemented methods.
- Manufacturer documents belong in the manufacturer-evidence register, not BibTeX by default.
- Bench measurements belong in the bench-evidence register, not BibTeX.
- Every citation key must be linked to at least one project claim ID.

## Current Verified Claim Links

{key_rows}

The canonical registry is `configs/evidence/project_claim_registry.json`.
"""


def make_markdown_doc(registry: dict[str, Any], backlog: dict[str, Any], search_plan: dict[str, Any]) -> str:
    claims = registry["claims"]
    verified_claims = [claim for claim in claims if claim.get("verified_literature_keys")]
    unresolved = [
        claim
        for claim in claims
        if claim["claim_id"]
        in {
            "physical_axicon_bessel_conversion",
            "physical_axicon_aperture_and_decentre",
            "physical_fourier_filtering_future_route",
            "camera_z_stack_acquisition",
            "camera_coordinate_calibration",
            "neural_fast_estimator_future",
            "fused_silica_waveguide_future",
            "legacy_crznse_material_proxy_branch",
        }
    ]
    claim_rows = "\n".join(
        f"| `{claim['claim_id']}` | {claim['current_status']} | {', '.join(claim.get('verified_literature_keys', [])) or 'none'} | {claim['manufacturer_specification_status']} | {claim['bench_evidence_status']} | {claim['what_can_currently_be_claimed']} | {claim['what_cannot_currently_be_claimed']} |"
        for claim in claims
    )
    unresolved_rows = "\n".join(
        f"| `{claim['claim_id']}` | {claim.get('literature_status')} | {claim['manufacturer_specification_status']} | {claim['bench_evidence_status']} | {claim['next_action']} |"
        for claim in unresolved
    )
    fused_rows = "\n".join(
        f"| `{claim_id}` | {_claim_by_id(claims, claim_id).get('literature_support_status')} | {_claim_by_id(claims, claim_id).get('publication_claim_readiness')} | {_claim_by_id(claims, claim_id).get('claim_boundary')} |"
        for claim_id in SPECIFIC_FUSED_SILICA_CLAIMS
    )
    verified_rows = "\n".join(
        f"- `{claim['claim_id']}` -> {', '.join(claim['verified_literature_keys'])}"
        for claim in verified_claims
    )
    search_count = len(search_plan["entries"])
    return f"""# Stage 9A.3 Verified Methods Evidence and Open Validation Layers

Stage 9A.3 integrates the supplied verified seed bibliography into the Stage
9A.2 claim registry. It is documentation and evidence work only: no physical 4F
propagation, camera model, inverse correction, neural network, material-response
physics, or active CSLM/axicon physics has been added.

## Boundary

```text
fourier_filter_physics_available = False
camera_model_enabled = False
material_model_enabled = False
diagnostic_only = True
final_export_allowed = False
```

Literature support for a principle is not the same thing as validation of this
numerical implementation, calibration of this bench, or demonstration of a
fused-silica process outcome.

## Verified Seed References Linked To Claims

{verified_rows}

The canonical bibliography is `references/structured_beam_methods.bib`. It was
copied from `references/incoming/structured_beam_methods_verified_seed.bib`
without supplementing it from memory.

## Multi-Layer Claim Register

Full structured records are stored in
`configs/evidence/project_claim_registry.json`.

| claim ID | status | verified literature | manufacturer status | bench status | currently claimable | not claimable |
|---|---|---|---|---|---|---|
{claim_rows}

## Fused-Silica Application Split

The broad `fused_silica_application_boundary` record is retained only as a
backwards-compatible superseded boundary. Specific future branches now carry
their own evidence needs.

This branch contains CrZnSe-specific proxy assumptions and is not validated for
fused-silica TGV, waveguide, welding, or modification predictions. The legacy
Cr:ZnSe branch remains quarantined from fused-silica decisions.

| claim ID | literature status | readiness | boundary |
|---|---|---|---|
{fused_rows}

## Deliberately Unresolved Claims

These remain unresolved because the seed bibliography does not validate them and
they require targeted literature, manufacturer specifications, bench data, or a
legacy/quarantine label.

| claim ID | literature | manufacturer | bench | next action |
|---|---|---|---|---|
{unresolved_rows}

## Evidence Registers

- Literature search plan: `configs/evidence/literature_search_plan.json`
  ({search_count} entries)
- Manufacturer evidence register:
  `configs/evidence/manufacturer_evidence_register.json`
- Bench evidence register:
  `configs/evidence/bench_evidence_register.json`

## Immediate Lab Action

Run the downstream carrier-stop characterisation session from Stage 9A.1B:
record actual SLM/camera/lens/stop/axicon identifiers, capture dark and flat
references, then measure downstream response versus SLM2 command-domain carrier
cycles and Fourier-stop settings without moving the fixed downstream route.
"""


def make_summary(registry: dict[str, Any], backlog: dict[str, Any], search_plan: dict[str, Any], bib_keys: list[str]) -> str:
    verified_claims = [claim["claim_id"] for claim in registry["claims"] if claim.get("verified_literature_keys")]
    reclassified = [
        "phase_only_slm_mask_generation",
        "phase_quantisation_and_grayscale_export",
        "command_domain_carrier_grating",
        "pixelated_slm_zero_order_and_unwanted_orders",
    ]
    return f"""# Stage 9A.2 / 9A.3 Evidence Audit Summary

Starting checkpoint for Stage 9A.3: Stage 9A.2 code evidence audit and research
backlog (`6bbc210`).

Stage 9A.3 integrates the supplied verified seed bibliography and adds explicit
multi-layer evidence records. No optical propagation, physical 4F, camera model,
inverse correction, neural network, or material-response physics is implemented.

## Verified Bibliography

- `references/structured_beam_methods.bib`
- Verified entries: {len(bib_keys)}
- Source seed: `references/incoming/structured_beam_methods_verified_seed.bib`

## Claims Linked To Verified References

{chr(10).join(f"- `{claim_id}`" for claim_id in verified_claims)}

## Reclassified To Avoid Overstating Physical Validation

{chr(10).join(f"- `{claim_id}`" for claim_id in reclassified)}

## Fused-Silica Split

- `fused_silica_bessel_channel_or_tgv_future`
- `fused_silica_waveguide_future`
- `fused_silica_welding_future`

The broad `fused_silica_application_boundary` is retained only as
`superseded_by_specific_application_claims`.

## Backlog And Search Plan

- Backlog items: {len(backlog['items'])}
- Literature/search-plan entries: {len(search_plan['entries'])}

## Immediate Lab Action

Run the downstream carrier-stop characterisation session from Stage 9A.1B:
record actual SLM/camera/lens/stop/axicon identifiers, capture dark and flat
references, then measure downstream response versus SLM2 command-domain carrier
cycles and Fourier-stop settings without moving the fixed downstream route.
"""


def update_notebook(root: Path, registry: dict[str, Any]) -> None:
    notebook_path = root / "notebooks/digital_twin/00_full_beam_to_write_cockpit_MVP.ipynb"
    nb = _read_json(notebook_path)
    nb["cells"] = [cell for cell in nb["cells"] if cell.get("id") not in {"stage9a3-evidence-md", "stage9a3-evidence-tables"}]
    nb["cells"].append(
        {
            "cell_type": "markdown",
            "id": "stage9a3-evidence-md",
            "metadata": {},
            "source": [
                "## Stage 9A.3 - Verified Methods Evidence and Open Validation Layers\n",
                "\n",
                "Verified bibliography keys are now linked to claim IDs, while manufacturer and bench validation remain separate evidence layers. No physical 4F, camera model, correction engine, AI, or material response is enabled.\n",
            ],
        }
    )
    nb["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "stage9a3-evidence-tables",
            "metadata": {},
            "outputs": [],
            "source": [
                "import json\n",
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "from IPython.display import display\n",
                "\n",
                "registry = json.loads(Path('configs/evidence/project_claim_registry.json').read_text())\n",
                "claims = registry['claims']\n",
                "rows = []\n",
                "for claim in claims:\n",
                "    rows.append({\n",
                "        'claim ID': claim['claim_id'],\n",
                "        'implemented status': claim['current_status'],\n",
                "        'verified literature key(s)': ', '.join(claim.get('verified_literature_keys', [])) or 'none',\n",
                "        'manufacturer evidence status': claim.get('manufacturer_specification_status', 'not_recorded'),\n",
                "        'bench evidence status': claim.get('bench_evidence_status', 'not_recorded'),\n",
                "        'what can currently be claimed': claim.get('what_can_currently_be_claimed', ''),\n",
                "        'what cannot currently be claimed': claim.get('what_cannot_currently_be_claimed', ''),\n",
                "        'next action': claim.get('next_action', ''),\n",
                "    })\n",
                "evidence_df = pd.DataFrame(rows)\n",
                "display(evidence_df)\n",
                "\n",
                "fused_ids = [\n",
                "    'fused_silica_bessel_channel_or_tgv_future',\n",
                "    'fused_silica_waveguide_future',\n",
                "    'fused_silica_welding_future',\n",
                "]\n",
                "display(evidence_df[evidence_df['claim ID'].isin(fused_ids)])\n",
                "print('Stage 9A.3 boundary: literature principle support is not bench validation or material-process validation.')\n",
            ],
        }
    )
    _write_json(notebook_path, nb)


def write_stage9a3_artifacts(root: str | Path = ".") -> dict[str, Path]:
    root = Path(root)
    seed_bib, seed_map, claim_to_keys, seed_keys = _load_seed(root)
    registry = integrate_claim_registry(_read_json(root / "configs/evidence/project_claim_registry.json"), seed_map, claim_to_keys)
    backlog = update_research_backlog(_read_json(root / "configs/evidence/research_backlog.json"), registry)
    search_plan = update_literature_search_plan(_read_json(root / "configs/evidence/literature_search_plan.json"), claim_to_keys)
    manufacturer, bench = update_evidence_registers(
        _read_json(root / "configs/evidence/manufacturer_evidence_register.json"),
        _read_json(root / "configs/evidence/bench_evidence_register.json"),
    )
    fused_template = update_fused_silica_template(_read_json(root / "configs/materials/fused_silica_evidence_template.json"))

    outputs = {
        "registry": root / "configs/evidence/project_claim_registry.json",
        "backlog": root / "configs/evidence/research_backlog.json",
        "search_plan": root / "configs/evidence/literature_search_plan.json",
        "manufacturer": root / "configs/evidence/manufacturer_evidence_register.json",
        "bench": root / "configs/evidence/bench_evidence_register.json",
        "fused_silica_template": root / "configs/materials/fused_silica_evidence_template.json",
        "doc": root / "docs/44_code_to_evidence_audit.md",
        "summary": root / "STAGE9A2_CODE_TO_EVIDENCE_AUDIT_SUMMARY.md",
        "references_readme": root / "references/README.md",
        "bib": root / "references/structured_beam_methods.bib",
        "notebook": root / "notebooks/digital_twin/00_full_beam_to_write_cockpit_MVP.ipynb",
    }
    _write_json(outputs["registry"], registry)
    _write_json(outputs["backlog"], backlog)
    _write_json(outputs["search_plan"], search_plan)
    _write_json(outputs["manufacturer"], manufacturer)
    _write_json(outputs["bench"], bench)
    _write_json(outputs["fused_silica_template"], fused_template)
    outputs["doc"].write_text(make_markdown_doc(registry, backlog, search_plan), encoding="utf-8")
    outputs["summary"].write_text(make_summary(registry, backlog, search_plan, seed_keys), encoding="utf-8")
    outputs["references_readme"].write_text(make_references_readme(registry), encoding="utf-8")
    outputs["bib"].write_text(seed_bib, encoding="utf-8")
    update_notebook(root, registry)
    return outputs


if __name__ == "__main__":
    for key, value in write_stage9a3_artifacts(Path.cwd()).items():
        print(f"{key}: {value}")
