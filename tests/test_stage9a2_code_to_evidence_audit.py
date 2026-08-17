"""Stage 9A.2 code-to-evidence audit tests."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "evidence" / "project_claim_registry.json"
BACKLOG_PATH = ROOT / "configs" / "evidence" / "research_backlog.json"
SEARCH_PATH = ROOT / "configs" / "evidence" / "literature_search_plan.json"
MANUFACTURER_PATH = ROOT / "configs" / "evidence" / "manufacturer_evidence_register.json"
BENCH_PATH = ROOT / "configs" / "evidence" / "bench_evidence_register.json"
NOTEBOOK_PATH = ROOT / "notebooks" / "digital_twin" / "00_full_beam_to_write_cockpit_MVP.ipynb"
DOC_PATH = ROOT / "docs" / "44_code_to_evidence_audit.md"
MATERIAL_DOC_PATH = ROOT / "docs" / "03_materials_application.md"
FIGURE_PATH = ROOT / "outputs" / "figures" / "digital_twin" / "stage9a2_code_to_evidence_roadmap.png"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _claims():
    return _json(REGISTRY_PATH)["claims"]


def test_all_canonical_active_route_components_have_claim_register_entry():
    registry = _json(REGISTRY_PATH)
    component_to_claims = {}
    for claim in registry["claims"]:
        for component in claim.get("implemented_route_components", []):
            component_to_claims.setdefault(component, []).append(claim["claim_id"])

    missing = [
        component
        for component in registry["active_route_components"]
        if component not in component_to_claims
    ]
    assert not missing


def test_active_physical_or_numerical_claims_have_evidence_pathway():
    active_types = {"physical_model", "numerical_method", "derived_metric", "measurement_method"}
    for claim in _claims():
        if claim["current_status"] != "implemented_active" or claim["claim_type"] not in active_types:
            continue
        has_pathway = any(
            [
                claim.get("literature_reference_keys"),
                claim.get("manufacturer_evidence_required"),
                claim.get("bench_evidence_required"),
                claim.get("assumption_or_placeholder"),
                claim["evidence_status"] == "assumption_declared",
            ]
        )
        assert has_pathway, claim["claim_id"]


def test_future_4f_camera_inverse_ai_and_material_claims_not_implemented_active():
    forbidden_claim_ids = {
        "physical_fourier_filtering_future_route",
        "camera_coordinate_calibration",
        "multi_plane_phase_retrieval_future",
        "effective_aberration_inference_future",
        "zernike_or_phase_conjugate_correction_future",
        "neural_fast_estimator_future",
        "fused_silica_application_boundary",
    }
    by_id = {claim["claim_id"]: claim for claim in _claims()}
    for claim_id in forbidden_claim_ids:
        assert by_id[claim_id]["current_status"] != "implemented_active"


def test_crznse_paths_classified_and_not_fused_silica_ready():
    registry = _json(REGISTRY_PATH)
    cr_claim = next(c for c in registry["claims"] if c["claim_id"] == "legacy_crznse_material_proxy_branch")
    assert cr_claim["current_status"] == "implemented_legacy"
    assert cr_claim["publication_claim_readiness"] == "not_claimable"
    assert "not valid for fused silica" in cr_claim["assumption_or_placeholder"].lower()

    material_doc = MATERIAL_DOC_PATH.read_text(encoding="utf-8").lower()
    assert "crznse-specific proxy assumptions" in material_doc
    assert "not validated" in material_doc
    assert "fused-silica" in material_doc


def test_every_research_backlog_item_links_to_claim_or_code_path():
    backlog = _json(BACKLOG_PATH)["items"]
    for item in backlog:
        assert item.get("linked_claim_ids") or item.get("linked_code_paths"), item["research_id"]


def test_p0_items_are_level_c_or_d_readiness_blockers():
    for item in _json(BACKLOG_PATH)["items"]:
        if item["priority"] == "P0":
            assert item.get("readiness_blocker_level") in {"Level-C", "Level-D"}, item["research_id"]


def test_literature_search_entries_contain_no_fabricated_citations():
    entries = _json(SEARCH_PATH)["entries"]
    assert entries
    bib_text = (ROOT / "references" / "structured_beam_methods.bib").read_text(encoding="utf-8").lower()
    bib_keys = set(re.findall(r"@\w+\{\s*([^,\s]+)", bib_text))
    forbidden = ("doi:", "@article", "journal =", "author =", "year =")
    for entry in entries:
        blob = json.dumps(entry).lower()
        status = entry["candidate_source_status"]
        assert status in {"search_needed_no_verified_citation", "targeted_search_required", "verified_seed_integrated"}
        if status == "verified_seed_integrated":
            assert set(entry["verified_reference_keys"]).issubset(bib_keys), entry["claim_id"]
        else:
            assert not any(token in blob for token in forbidden), entry["claim_id"]

    assert bib_keys


def test_manufacturer_and_bench_registers_distinguish_value_states():
    expected = {"unknown", "placeholder", "estimated", "manufacturer_specified", "measured"}
    manufacturer = _json(MANUFACTURER_PATH)
    bench = _json(BENCH_PATH)
    assert set(manufacturer["value_state_vocabulary"]) == expected
    assert set(bench["value_state_vocabulary"]) == expected
    assert all(entry["current_value_state"] in expected for entry in manufacturer["entries"])
    assert all(entry["current_value_state"] in expected for entry in bench["entries"])
    assert all(entry["verified"] is False for entry in manufacturer["entries"])
    assert all(entry["ready"] is False for entry in bench["entries"])


def test_notebook_json_valid_and_includes_stage9a2():
    nb = _json(NOTEBOOK_PATH)
    source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
    assert "Stage 9A.2" in source
    assert "Evidence, Research Gaps, and Claim Boundaries" in source
    assert "project_claim_registry.json" in source
    assert "stage9a2_code_to_evidence_roadmap.png" in source


def test_required_audit_artifacts_exist_and_boundary_text_present():
    for path in (REGISTRY_PATH, BACKLOG_PATH, SEARCH_PATH, MANUFACTURER_PATH, BENCH_PATH, DOC_PATH, FIGURE_PATH):
        assert path.exists(), path
    doc = DOC_PATH.read_text(encoding="utf-8")
    assert "fourier_filter_physics_available = False" in doc
    assert "Literature support for a principle is not the same thing" in doc
    assert "CrZnSe-specific proxy assumptions" in doc
