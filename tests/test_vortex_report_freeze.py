from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "outputs" / "validation" / "report_freeze"
CLAIM_COLUMNS = [
    "claim_id",
    "claim_text",
    "beam_case",
    "solver",
    "mapping_mode",
    "maturity",
    "status",
    "quantitative_valid",
    "calibration_required",
    "source_code_path",
    "data_path",
    "figure_path",
    "test_path",
    "notes",
]


def _csv(name: str) -> list[dict[str, str]]:
    with (FREEZE / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_frozen_claim_schema_is_exact_and_ids_are_unique() -> None:
    path = FREEZE / "vortex_report_claims.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == CLAIM_COLUMNS
    ids = [row["claim_id"] for row in rows]
    assert ids
    assert len(ids) == len(set(ids))


def test_each_claim_evidence_path_exists() -> None:
    for row in _csv("vortex_report_claims.csv"):
        for column in ("source_code_path", "data_path", "figure_path", "test_path"):
            for value in filter(None, (part.strip() for part in row[column].split(";"))):
                assert (ROOT / value).is_file(), f"{row['claim_id']}: missing {column} {value}"


def test_selected_evidence_hashes_match_disk() -> None:
    rows = _csv("vortex_report_files.csv")
    assert rows
    for row in rows:
        path = ROOT / row["path"]
        assert path.is_file()
        assert _sha256(path) == row["sha256"]


def test_manifest_indexes_match_their_frozen_files() -> None:
    manifest = json.loads((FREEZE / "vortex_report_manifest.json").read_text(encoding="utf-8"))
    claims = ROOT / manifest["claims_path"]
    files = ROOT / manifest["files_index_path"]
    assert _sha256(claims) == manifest["claims_sha256"]
    assert _sha256(files) == manifest["files_index_sha256"]
    assert manifest["selected_evidence_count"] == len(_csv("vortex_report_files.csv"))


def test_freeze_has_no_selected_large_binary_or_secret_signature() -> None:
    assert all(int(row["size_bytes"]) <= 90 * 1024 * 1024 for row in _csv("vortex_report_files.csv"))
    manifest = json.loads((FREEZE / "vortex_report_manifest.json").read_text(encoding="utf-8"))
    assert manifest["audit"]["secret_signature_findings"] == []


def test_report_ready_references_are_resolved() -> None:
    broken = [
        row for row in _csv("path_reference_audit.csv")
        if row["scope"] == "report_ready" and row["exists"].lower() == "false"
    ]
    assert broken == []


def test_scope_forbids_experimental_and_material_claims() -> None:
    manifest = json.loads((FREEZE / "vortex_report_manifest.json").read_text(encoding="utf-8"))
    scope = manifest["scope_boundary"].lower()
    assert "no experimental validation" in scope
    assert "no" in scope and "material-modification claim" in scope


def test_phase_outcomes_are_reported_without_promotion() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for outcome in ("PHASE1-B", "PHASE1R-B", "PHASE2A-B", "PHASE2B-A", "PHASE2C-B"):
        assert outcome in readme
