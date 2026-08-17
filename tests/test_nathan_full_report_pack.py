from __future__ import annotations

import csv
import json
from pathlib import Path

from vbb_study.digital_twin import REPORT_ROOT, build_nathan_full_report_pack

ROOT = Path(REPORT_ROOT)


def _ensure_report() -> Path:
    required = [
        ROOT / "nathan_hexagonal_bessel_full_report.tex",
        ROOT / "nathan_hexagonal_bessel_full_report.pdf",
        ROOT / "build_report_summary.json",
        ROOT / "figure_manifest.csv",
    ]
    if not all(path.exists() for path in required):
        build_nathan_full_report_pack(report_root=ROOT)
    return ROOT


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _json_load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_report_tex_pdf_and_readme_exist_with_declared_build_method() -> None:
    root = _ensure_report()
    summary = _json_load(root / "build_report_summary.json")
    readme = (root / "README.md").read_text(encoding="utf-8")

    assert (root / "nathan_hexagonal_bessel_full_report.tex").stat().st_size > 20_000
    assert (root / "nathan_hexagonal_bessel_full_report.pdf").stat().st_size > 100_000
    assert summary["page_count"] >= 30
    assert summary["pdf_build_method"] in {"pdflatex", "xelatex", "lualatex", "tectonic", "matplotlib_fallback"}
    assert summary["pdf_build_status"] in {"compiled", "fallback_pdf_created"}
    assert summary["compile_command"] in readme
    assert "Required LaTeX packages" in readme
    assert "Figure Source Mapping" in readme
    assert "figure_manifest.csv" in readme
    assert "graphicx" in readme


def test_report_pack_has_expected_registry_and_manifest_counts() -> None:
    root = _ensure_report()
    summary = _json_load(root / "build_report_summary.json")

    assert summary["section_count"] >= 18
    assert summary["equation_count"] >= 15
    assert summary["claim_count"] >= 10
    assert summary["number_count"] >= 20
    assert summary["figure_count"] >= 15
    assert len(_csv_rows(root / "equation_registry.csv")) == summary["equation_count"]
    assert len(_csv_rows(root / "claim_registry.csv")) == summary["claim_count"]
    assert len(_csv_rows(root / "number_registry.csv")) == summary["number_count"]
    assert len(_csv_rows(root / "figure_manifest.csv")) == summary["figure_count"]


def test_required_evidence_pack_subfolders_exist() -> None:
    root = _ensure_report()
    required = {
        "equations",
        "figures",
        "tables",
        "claims",
        "provenance",
        "sequential_architecture",
        "source_validation",
        "inverse_design",
        "realism",
        "correction",
        "hardware",
        "superseded",
    }

    assert required.issubset({path.name for path in (root / "evidence_pack").iterdir() if path.is_dir()})


def test_every_manifest_figure_is_copied_with_traceable_source() -> None:
    root = _ensure_report()
    rows = _csv_rows(root / "figure_manifest.csv")

    assert rows
    assert all(row["source_file"] for row in rows)
    assert all(row["source_stage"] for row in rows)
    assert all(row["copy_status"] == "copied" for row in rows)
    assert all((root.parent / row["copied_to"]).exists() for row in rows)


def test_primary_figures_use_publication_or_native_sampling_not_lowres_hero_arrays() -> None:
    root = _ensure_report()
    rows = _csv_rows(root / "figure_manifest.csv")
    primary = [row for row in rows if row["primary_hero"] == "True"]

    assert primary
    assert all(row["N"] != "384" for row in primary)
    assert any(row["N"] == "1536" for row in primary)
    assert any("1920x1080" in row["N"] for row in primary)


def test_report_explicitly_uses_sequential_architecture_and_supersedes_split_arm_pbs() -> None:
    root = _ensure_report()
    tex = (root / "nathan_hexagonal_bessel_full_report.tex").read_text(encoding="utf-8").lower()
    superseded = _csv_rows(root / "superseded_material.csv")
    split_rows = [row for row in superseded if "split" in row["material"].lower() or "pbs" in row["material"].lower()]

    assert "no canonical pbs split/recombine interferometer arms are used" in tex
    assert "the accepted implementation is collinear and sequential" in tex
    assert split_rows
    assert all(row["status"] == "superseded" for row in split_rows)


def test_forbidden_old_optimizer_candidate_is_not_rehabilitated() -> None:
    root = _ensure_report()
    claims = _csv_rows(root / "claim_registry.csv")
    superseded = _csv_rows(root / "superseded_material.csv")
    forbidden_id = "m2u2_opt_003_c5.75_i0.32_q-0.25_r+0.0_p0.10"

    assert any(forbidden_id in row["claim"] or forbidden_id in row["numerical_result"] for row in claims)
    assert any(row["material"] == forbidden_id and row["status"] == "forbidden" for row in superseded)


def test_realistic_sequential_z60_metric_is_traceable_in_claim_and_number_registries() -> None:
    root = _ensure_report()
    claims = _csv_rows(root / "claim_registry.csv")
    numbers = _csv_rows(root / "number_registry.csv")
    tex = (root / "nathan_hexagonal_bessel_full_report.tex").read_text(encoding="utf-8")

    assert any(row["number_id"] == "N019" and row["value"] == "0.9936209493" for row in numbers)
    assert any("0.9936209493" in row["numerical_result"] for row in claims)
    assert "0.9936209493" in tex


def test_claims_and_numbers_have_evidence_or_source_files() -> None:
    root = _ensure_report()
    claims = _csv_rows(root / "claim_registry.csv")
    numbers = _csv_rows(root / "number_registry.csv")

    assert all(row["evidence_files"] for row in claims)
    assert all(row["data_source"] for row in claims)
    assert all(row["source_file"] for row in numbers)


def test_microfabrication_and_sample_plane_success_are_not_claimed() -> None:
    root = _ensure_report()
    summary = _json_load(root / "build_report_summary.json")
    tex = (root / "nathan_hexagonal_bessel_full_report.tex").read_text(encoding="utf-8").lower()
    claims = _csv_rows(root / "claim_registry.csv")

    assert summary["microfabrication_sample_plane_claim"] is False
    assert "microfabrication/sample-plane success is not claimed" in tex
    assert any(row["claim"] == "Microfabrication/sample-plane success is not claimed by this report." for row in claims)


def test_power_ledger_is_sequential_not_split_arm_bookkeeping() -> None:
    root = _ensure_report()
    rows = _csv_rows(root / "evidence_pack" / "tables" / "mode2w_fix_sequential_power_ledger.csv")
    text = " ".join(f"{row['stage']} {row['note']}" for row in rows).lower()

    assert rows
    assert "recombination" not in text
    assert "h_arm" not in text
    assert "v_arm" not in text
    assert all(row["split_arm_stage"] == "False" for row in rows)


def test_report_includes_sampling_audit_and_sas_zoom_evidence() -> None:
    root = _ensure_report()
    audit = _csv_rows(root / "evidence_pack" / "tables" / "mode2w_fix_numerical_source_audit.csv")
    tex = (root / "nathan_hexagonal_bessel_full_report.tex").read_text(encoding="utf-8")
    sas = [row for row in audit if row.get("render_method") == "scalable_angular_spectrum_zoom"]

    assert sas
    assert min(float(row["samples_per_radial_fringe"]) for row in sas) > 9.0
    assert "N=1536" in tex
    assert "SAS-scaled focus crops" in tex
