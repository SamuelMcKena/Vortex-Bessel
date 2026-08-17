"""Stage 9A.3 verified bibliography and evidence-layer tests."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "evidence" / "project_claim_registry.json"
BACKLOG_PATH = ROOT / "configs" / "evidence" / "research_backlog.json"
SEARCH_PATH = ROOT / "configs" / "evidence" / "literature_search_plan.json"
BIB_PATH = ROOT / "references" / "structured_beam_methods.bib"
SEED_BIB_PATH = ROOT / "references" / "incoming" / "structured_beam_methods_verified_seed.bib"
SEED_MAP_PATH = ROOT / "references" / "incoming" / "structured_beam_methods_verified_seed_map.json"
NOTEBOOK_PATH = ROOT / "notebooks" / "digital_twin" / "00_full_beam_to_write_cockpit_MVP.ipynb"


EXPECTED_BIB_KEYS = {
    "matsushima2009bandlimited",
    "engstrom2013slmcalibration",
    "zhang2009zeroorder",
    "miao2022besselretrieval",
    "neil2000closedloop",
    "lopezquesada2009slmcorrection",
    "bhuyan2010microchannels",
    "zhang2018besselwelding",
}


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _bib_keys(path: Path = BIB_PATH) -> set[str]:
    return set(re.findall(r"@\w+\{\s*([^,\s]+)", path.read_text(encoding="utf-8")))


def _claims() -> dict[str, dict]:
    return {claim["claim_id"]: claim for claim in _json(REGISTRY_PATH)["claims"]}


def _all_registry_reference_keys() -> set[str]:
    keys: set[str] = set()
    for claim in _claims().values():
        keys.update(claim.get("literature_reference_keys", []))
        keys.update(claim.get("verified_literature_keys", []))
        for layer in claim.get("evidence_layers", []):
            keys.update(layer.get("reference_keys", []))
    return keys


def test_verified_seed_bibliography_is_canonical_and_referenced():
    assert BIB_PATH.read_text(encoding="utf-8") == SEED_BIB_PATH.read_text(encoding="utf-8")
    assert _bib_keys() == EXPECTED_BIB_KEYS
    assert _all_registry_reference_keys().issubset(_bib_keys())

    registry = _json(REGISTRY_PATH)
    for key in EXPECTED_BIB_KEYS:
        assert registry["verified_reference_key_to_claim_ids"][key]


def test_seed_map_claim_links_are_preserved_with_channel_alias():
    seed_map = _json(SEED_MAP_PATH)["claim_to_keys"]
    claims = _claims()
    direct_claims = {
        "angular_spectrum_or_bl_asm_propagation",
        "finite_sampling_and_aliasing_control",
        "phase_only_slm_mask_generation",
        "phase_quantisation_and_grayscale_export",
        "command_domain_carrier_grating",
        "pixelated_slm_zero_order_and_unwanted_orders",
        "multi_plane_phase_retrieval_future",
        "effective_aberration_inference_future",
        "zernike_or_phase_conjugate_correction_future",
        "fused_silica_welding_future",
    }
    for claim_id in direct_claims:
        assert claims[claim_id]["verified_literature_keys"] == seed_map[claim_id]
    assert claims["fused_silica_bessel_channel_or_tgv_future"]["verified_literature_keys"] == seed_map["fused_silica_channel_future"]


def test_active_canonical_claims_have_a_valid_evidence_pathway():
    active_types = {"physical_model", "numerical_method", "derived_metric", "measurement_method", "command_model", "diagnostic_boundary"}
    for claim in _claims().values():
        if claim["route_relevance"] != "canonical_active" or claim["claim_type"] not in active_types:
            continue
        layer_statuses = [layer["status"] for layer in claim.get("evidence_layers", [])]
        has_pathway = any(
            [
                "verified" in layer_statuses,
                claim.get("manufacturer_evidence_required"),
                claim.get("bench_evidence_required"),
                claim.get("evidence_status") == "assumption_declared",
                claim.get("physics_status") == "diagnostic_placeholder",
            ]
        )
        assert has_pathway, claim["claim_id"]


def test_slm_command_claims_are_not_marked_physically_calibrated():
    claims = _claims()
    for claim_id in ("phase_only_slm_mask_generation", "phase_quantisation_and_grayscale_export"):
        claim = claims[claim_id]
        status_blob = json.dumps(claim).lower()
        assert claim["current_status"].startswith("implemented_command")
        assert "unverified_physical_response" in claim["physics_status"]
        assert claim["manufacturer_specification_status"] == "needs_manufacturer_data"
        assert claim["bench_evidence_status"] == "needs_bench_measurement"
        assert "physically_calibrated" not in status_blob
        assert "bench_validated" not in claim["evidence_layer_summary"]["bench_validated_representation"]


def test_command_carrier_has_literature_principle_and_bench_mapping_requirement():
    claim = _claims()["command_domain_carrier_grating"]
    assert claim["verified_literature_keys"] == ["zhang2009zeroorder"]
    assert claim["evidence_status"] == "verified_principle_needs_bench_mapping"
    assert any("carrier command cycles" in item for item in claim["bench_evidence_required"])
    assert any(layer["layer_type"] == "peer_reviewed_primary_source" for layer in claim["evidence_layers"])
    assert any(layer["layer_type"] == "bench_measurement" for layer in claim["evidence_layers"])


def test_pixelated_slm_order_behaviour_remains_future_4f_realism():
    claim = _claims()["pixelated_slm_zero_order_and_unwanted_orders"]
    assert claim["current_status"] == "planned_future"
    assert claim["physics_status"] == "future_not_implemented"
    assert claim["verified_literature_keys"] == ["zhang2009zeroorder"]
    assert "active validated order-efficiency" in claim["what_cannot_currently_be_claimed"]
    assert claim["implemented_route_components"] == []


def test_fused_silica_application_claims_are_split_and_broad_boundary_superseded():
    claims = _claims()
    split = {
        "fused_silica_bessel_channel_or_tgv_future",
        "fused_silica_waveguide_future",
        "fused_silica_welding_future",
    }
    assert split.issubset(claims)
    broad = claims["fused_silica_application_boundary"]
    assert broad["current_status"] == "superseded_by_specific_application_claims"
    assert set(broad["superseded_by"]) == split
    assert claims["fused_silica_bessel_channel_or_tgv_future"]["verified_literature_keys"] == ["bhuyan2010microchannels"]
    assert claims["fused_silica_welding_future"]["verified_literature_keys"] == ["zhang2018besselwelding"]


def test_waveguide_branch_remains_unresolved_without_direct_verified_source():
    claim = _claims()["fused_silica_waveguide_future"]
    assert claim["verified_literature_keys"] == []
    assert claim["literature_support_status"] == "targeted_search_required"
    assert claim["publication_claim_readiness"] == "not_claimable"
    assert any(layer["layer_type"] == "targeted_literature_search" for layer in claim["evidence_layers"])


def test_deliberately_unresolved_claims_remain_unfilled_without_seed_support():
    seed_map = _json(SEED_MAP_PATH)
    claims = _claims()
    for claim_id in seed_map["claims_deliberately_not_filled_by_this_seed"]:
        if claim_id == "fused_silica_waveguide_future":
            assert claims[claim_id]["literature_status"] == "targeted_search_required"
        elif claim_id == "legacy_crznse_material_proxy_branch":
            assert claims[claim_id]["route_relevance"] == "legacy_retained"
        else:
            assert not claims[claim_id]["verified_literature_keys"], claim_id

    neural = claims["neural_fast_estimator_future"]
    assert neural["current_status"] == "not_implemented"
    assert neural["physics_status"] == "future_not_implemented"
    assert neural["publication_claim_readiness"] == "not_claimable"


def test_no_claim_says_literature_alone_validates_the_local_bench():
    forbidden_claim_phrases = ("literature validates this bench", "literature alone validates", "bench validated by literature")
    for claim in _claims().values():
        claim_blob = json.dumps(claim).lower()
        assert not any(phrase in claim_blob for phrase in forbidden_claim_phrases), claim["claim_id"]
        assert claim.get("local_bench_validation_status") != "bench_validated"
        for layer in claim.get("evidence_layers", []):
            if layer["layer_type"] == "peer_reviewed_primary_source":
                does_not_support = layer["what_it_does_not_support"].lower()
                assert any(token in does_not_support for token in ("bench", "local", "specific", "apparatus")), claim["claim_id"]


def test_research_backlog_has_evidence_layers_and_direct_lab_deliverables():
    backlog = _json(BACKLOG_PATH)["items"]
    for item in backlog:
        assert item["linked_claim_ids"], item["research_id"]
        assert item["linked_code_paths"], item["research_id"]
        assert item["required_evidence_layers"], item["research_id"]
        assert item["concrete_next_deliverable"], item["research_id"]

    first_lab_items = {
        "P0_SLM_SPEC",
        "P0_SLM_PHASE",
        "P0_CARRIER_MAPPING",
        "P0_FOURIER_STOP",
        "P0_LENS_GEOMETRY",
        "P0_CAMERA",
        "P0_AXICON",
    }
    allowed = {"manufacturer_datasheet", "bench_measurement", "manufacturer_datasheet_and_bench_measurement", "first_carrier_session_image_dataset"}
    by_id = {item["research_id"]: item for item in backlog}
    for research_id in first_lab_items:
        assert by_id[research_id]["next_deliverable_type"] in allowed
        assert "code" not in by_id[research_id]["concrete_next_deliverable"].lower()


def test_literature_search_plan_uses_verified_or_targeted_statuses_only():
    entries = _json(SEARCH_PATH)["entries"]
    assert entries
    for entry in entries:
        assert entry["candidate_source_status"] in {"verified_seed_integrated", "targeted_search_required"}
        assert set(entry["verified_reference_keys"]).issubset(_bib_keys())


def test_notebook_json_valid_and_includes_stage9a3():
    nb = _json(NOTEBOOK_PATH)
    source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
    assert "Stage 9A.3" in source
    assert "Verified Methods Evidence and Open Validation Layers" in source
    assert "verified literature key(s)" in source
