from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from vbb_study.digital_twin.nathan_mode2u3_hardware_closure import assert_not_forbidden
from vbb_study.digital_twin.nathan_mode2u2_fix_strict_hexagon_optimisation import OLD_BEST_COMPROMISE_ID
from vbb_study.digital_twin.nathan_report_visual_refinement import (
    PRIORITY_1,
    REFINED_FIGURE_ROOT,
    REFINED_PDF,
    REFINED_TEX,
    VISUAL_AUDIT_CSV,
    VISUAL_AUDIT_JSON,
    _REFINED_MAIN_FIGURES,
)

REPORT_ROOT = Path("report")


def _audit_rows() -> list[dict[str, str]]:
    with VISUAL_AUDIT_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _audit_json() -> dict:
    return json.loads(VISUAL_AUDIT_JSON.read_text(encoding="utf-8"))


def _manifest_ids() -> set[str]:
    with (REPORT_ROOT / "figure_manifest.csv").open(newline="", encoding="utf-8") as f:
        return {row["figure_id"] for row in csv.DictReader(f)}


def test_every_report_figure_has_an_audit_row() -> None:
    audited = {row["report_figure_id"] for row in _audit_rows()}

    missing = _manifest_ids() - audited
    assert not missing, f"figures without audit rows: {sorted(missing)}"


def test_every_priority_1_figure_has_explicit_visual_inspection_result() -> None:
    rows = {row["report_figure_id"]: row for row in _audit_rows()}
    for fid in PRIORITY_1:
        row = rows[fid]

        assert row["severity"] in {"critical", "major", "moderate", "minor", "pass"}
        assert "multimodal" in row["inspection_evidence"]
        assert row["recommended_action"]


def test_no_low_n_source_backs_primary_hero_figures() -> None:
    payload = _audit_json()
    manifest = payload["refined_manifest"]

    assert int(manifest["hero_grid_n"]) >= 1536
    assert int(manifest["propagation_grid_n"]) >= 1024
    for row in _audit_rows():
        if row["report_figure_id"] in PRIORITY_1 and row["numerical_N"].isdigit():
            assert int(row["numerical_N"]) >= 1024


def test_slm_masks_preserve_native_panel_aspect() -> None:
    payload = _audit_json()
    mask_shape = payload["refined_manifest"]["figures"]["F2"]["mask_shape"]

    assert mask_shape == [1080, 1920]


def test_xz_yz_figures_are_not_forced_square() -> None:
    payload = _audit_json()
    f4b = payload["refined_manifest"]["figures"]["F4B"]

    assert float(f4b["transverse_crop_mm"]) <= 2.0  # beam-scale crop, not the full +/-5 mm window
    # the audit records the old F13 aspect failure explicitly
    row = next(r for r in _audit_rows() if r["report_figure_id"] == "F13")
    assert row["wrong_aspect_ratio"] == "True"
    assert row["replacement_source"] == "F4B"


def test_sequential_architecture_is_canonical() -> None:
    payload = _audit_json()

    assert payload["sequential_architecture_canonical"] is True
    tex = REFINED_TEX.read_text(encoding="utf-8")
    assert "F1_sequential_architecture" in tex


def test_split_arm_diagram_is_not_used_as_final_architecture() -> None:
    payload = _audit_json()

    assert payload["split_arm_used_as_final_architecture"] is False
    assert payload["refined_manifest"]["split_arm_used"] is False


def test_power_ledger_is_sequential_only() -> None:
    ledger = Path("outputs/figures/digital_twin/nathan_mode2w_fix_sequential_master/07_power/mode2w_fix_sequential_power_ledger.csv")
    with ledger.open(newline="", encoding="utf-8") as f:
        stages = [row["stage"] for row in csv.DictReader(f)]

    joined = " ".join(stages).lower()
    assert "arm_h" not in joined and "arm_v" not in joined and "recombination" not in joined
    assert any("slm1" in s for s in joined.split())


def test_forbidden_old_optimiser_candidate_remains_forbidden() -> None:
    with pytest.raises(ValueError, match="strict hexagon gate"):
        assert_not_forbidden(OLD_BEST_COMPROMISE_ID)


def test_all_refined_figures_exist() -> None:
    for fid, filename, _ in _REFINED_MAIN_FIGURES:
        png = REFINED_FIGURE_ROOT / filename

        assert png.exists(), f"missing refined figure {fid}: {png}"
        assert png.with_suffix(".pdf").exists(), f"missing vector sibling for {fid}"


def test_refined_pdf_exists() -> None:
    assert REFINED_PDF.exists()
    assert REFINED_TEX.exists()
    payload = _audit_json()
    assert payload["refined_pdf"]["page_count"] > 10
    # honesty requirement: fallback status is stated, not hidden
    assert payload["latex_engine_available"] is False


def test_priority_1_figures_pass_final_gate_for_vqa_a() -> None:
    payload = _audit_json()
    gate = {row["figure_id"]: row for row in payload["second_pass_gate"]}
    fields = ("visually_sharp", "aspect_correct", "text_readable", "colourbars_readable",
              "no_visible_pixelation", "scientifically_faithful")
    if payload["outcome"] == "VQA-A":
        for row in gate.values():
            for field in fields:
                assert row[field] is True, f"{row['figure_id']}: {field} failed but outcome is VQA-A"
    else:
        assert payload["outcome"] in {"VQA-B", "VQA-C", "VQA-D"}


def test_audit_severity_and_actions_use_allowed_vocabulary() -> None:
    allowed_sev = {"critical", "major", "moderate", "minor", "pass"}
    allowed_actions = {"keep", "re-render", "replace_with_higher_N", "replace_with_vector_output",
                       "correct_aspect_ratio", "split_figure", "enlarge_text", "crop_dead_space",
                       "redesign_layout", "remove", "move_to_appendix"}
    for row in _audit_rows():

        assert row["severity"] in allowed_sev
        assert row["recommended_action"] in allowed_actions
