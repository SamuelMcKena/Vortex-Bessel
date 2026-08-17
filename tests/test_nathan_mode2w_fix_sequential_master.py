from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from vbb_study.digital_twin import (
    MODE2WF_DEFAULT_OUTPUT_ROOT,
    OLD_BEST_COMPROMISE_ID,
    assert_not_forbidden,
    write_mode2w_fix_sequential_master,
)

ROOT = Path(MODE2WF_DEFAULT_OUTPUT_ROOT)


def _ensure_outputs() -> Path:
    manifest = ROOT / "10_final_status" / "mode2w_fix_manifest.json"
    if not manifest.exists():
        write_mode2w_fix_sequential_master(output_dir=ROOT)
    return ROOT


def _json_load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_canonical_architecture_contains_no_pbs_split_recombine_arms() -> None:
    root = _ensure_outputs()
    rows = _csv_rows(root / "00_architecture" / "mode2w_fix_sequential_architecture.csv")
    text = " ".join(f"{row['component']} {row['role']}" for row in rows).lower()

    assert "pbs" not in text
    assert "recombine" not in text
    assert "spatial arm" not in text
    assert [row["component"] for row in rows].index("SLM1") < [row["component"] for row in rows].index("SLM2")


def test_sequential_ideal_route_reproduces_pre_qwp_target_channels() -> None:
    root = _ensure_outputs()
    eq = _json_load(root / "00_architecture" / "mode2w_fix_sequential_equivalence.json")

    assert eq["sequential_pre_qwp_overlap_to_abstract"] >= 0.999999


def test_sequential_post_qwp_field_reproduces_validated_target() -> None:
    root = _ensure_outputs()
    eq = _json_load(root / "00_architecture" / "mode2w_fix_sequential_equivalence.json")

    assert eq["sequential_post_qwp_overlap_to_target"] >= 0.999999


def test_sequential_z60_output_preserves_strict_hexagon() -> None:
    root = _ensure_outputs()
    eq = _json_load(root / "00_architecture" / "mode2w_fix_sequential_equivalence.json")
    status = _json_load(root / "10_final_status" / "m2wf_outcome_report.json")

    assert eq["realistic_sequential_strict_hexagon"] is True
    assert eq["realistic_sequential_strict_class"] == "visual_hexagonal_field"
    assert status["strict_hexagon_result"] is True


def test_slm1_uses_positive_alpha_convention() -> None:
    root = _ensure_outputs()
    rows = _csv_rows(root / "00_architecture" / "mode2w_fix_phase_conventions.csv")
    slm1 = next(row for row in rows if row["element"] == "SLM1")

    assert "phi_H = +alpha + carrier" in slm1["phase_convention"]


def test_slm2_uses_negative_alpha_plus_pi_over_2_convention() -> None:
    root = _ensure_outputs()
    rows = _csv_rows(root / "00_architecture" / "mode2w_fix_phase_conventions.csv")
    slm2 = next(row for row in rows if row["element"] == "SLM2")

    assert "phi_V = -alpha + pi/2 + carrier" in slm2["phase_convention"]


def test_swap_hwps_are_conditional_on_lc_director_orientation() -> None:
    root = _ensure_outputs()
    rows = _csv_rows(root / "00_architecture" / "mode2w_fix_sequential_variants.csv")
    variants = {row["variant"]: row for row in rows}

    assert variants["A_same_panel_orientation"]["swap_hwps_required"] == "True"
    assert variants["B_orthogonally_mounted_slm2"]["swap_hwps_required"] == "False"
    assert variants["B_orthogonally_mounted_slm2"]["rotated_panel_valid"] == "True"


def test_source_audit_records_n_and_dx_for_every_primary_field_panel() -> None:
    root = _ensure_outputs()
    rows = _csv_rows(root / "01_source_audit" / "mode2w_fix_numerical_source_audit.csv")

    assert rows
    for row in rows:
        assert int(row["numerical_N"]) > 0
        assert float(row["numerical_dx_m"]) > 0.0
        if row["route_or_case"] != "native_panel_phase_mask":
            assert float(row["samples_per_radial_fringe"]) > 0.0


def test_no_n384_array_is_used_for_primary_hero_comparison() -> None:
    root = _ensure_outputs()
    rows = _csv_rows(root / "01_source_audit" / "mode2w_fix_numerical_source_audit.csv")
    hero = [row for row in rows if row["figure_id"] == "fig3A"]

    assert hero
    assert all(int(row["numerical_N"]) != 384 for row in hero)


def test_primary_ideal_vs_realistic_uses_n1536_where_available() -> None:
    root = _ensure_outputs()
    rows = _csv_rows(root / "01_source_audit" / "mode2w_fix_numerical_source_audit.csv")
    hero = [row for row in rows if row["figure_id"] == "fig3A"]

    assert {int(row["numerical_N"]) for row in hero} == {1536}


def test_primary_zoom_panels_use_sas_scaled_output_grid() -> None:
    root = _ensure_outputs()
    rows = _csv_rows(root / "01_source_audit" / "mode2w_fix_numerical_source_audit.csv")
    hero = [row for row in rows if row["figure_id"] == "fig3A"]
    status = _json_load(root / "10_final_status" / "m2wf_outcome_report.json")

    assert status["sas_zoom_rendering_enabled"] is True
    assert hero
    assert all(row["render_method"] == "scalable_angular_spectrum_zoom" for row in hero)
    assert all(float(row["numerical_dx_m"]) < float(row["native_input_dx_m"]) for row in hero)
    assert all(float(row["samples_per_radial_fringe"]) >= 20.0 for row in hero)
    assert all(row["metrics_computed_on_native_data"] == "True" for row in hero)


def test_propagation_focus_crops_use_sas_zoom_not_dead_full_window() -> None:
    root = _ensure_outputs()
    rows = _csv_rows(root / "01_source_audit" / "mode2w_fix_numerical_source_audit.csv")
    zoom = [row for row in rows if row["figure_id"] == "fig4" and row.get("render_method") == "scalable_angular_spectrum_zoom"]

    assert {row["panel_id"] for row in zoom} == {
        "sas_focus_crop_z30mm",
        "sas_focus_crop_z60mm",
        "sas_focus_crop_z90mm",
        "sas_focus_crop_z150mm",
        "sas_focus_crop_z200mm",
    }
    assert all(float(row["numerical_dx_m"]) < float(row["native_input_dx_m"]) for row in zoom)
    assert min(float(row["samples_per_radial_fringe"]) for row in zoom) > 9.0


def test_power_ledger_contains_no_split_arm_or_pbs_recombination_stages() -> None:
    root = _ensure_outputs()
    rows = _csv_rows(root / "07_power" / "mode2w_fix_sequential_power_ledger.csv")
    text = " ".join(f"{row['stage']} {row['note']}" for row in rows).lower()

    assert "pbs" not in text
    assert "recombination" not in text
    assert "h_arm" not in text
    assert "v_arm" not in text
    assert all(row["split_arm_stage"] == "False" for row in rows)


def test_figure2_uses_native_1920_by_1080_masks() -> None:
    root = _ensure_outputs()
    metadata = _json_load(root / "02_target_masks" / "mode2w_fix_slm_mask_metadata.json")
    audit = _csv_rows(root / "01_source_audit" / "mode2w_fix_numerical_source_audit.csv")
    mask_rows = [row for row in audit if row["route_or_case"] == "native_panel_phase_mask"]

    assert metadata["panel"]["width_px"] == 1920
    assert metadata["panel"]["height_px"] == 1080
    assert metadata["carrier_period_slm_pixels"] == pytest.approx(20.0)
    assert len(mask_rows) == 2
    assert all(int(row["numerical_N"]) == 1920 for row in mask_rows)
    assert all(int(row["numerical_M_y"]) == 1080 for row in mask_rows)


def test_old_forbidden_optimiser_candidates_remain_forbidden() -> None:
    root = _ensure_outputs()
    manifest = _json_load(root / "10_final_status" / "mode2w_fix_manifest.json")

    with pytest.raises(ValueError):
        assert_not_forbidden(OLD_BEST_COMPROMISE_ID)
    assert manifest["canonical_operating_point"] != OLD_BEST_COMPROMISE_ID
    assert manifest["secondary_operating_point"] != OLD_BEST_COMPROMISE_ID


def test_repaired_strict_gate_remains_in_force() -> None:
    root = _ensure_outputs()
    metrics = _csv_rows(root / "03_ideal_vs_realistic" / "mode2w_fix_ideal_vs_realistic_metrics.csv")
    realistic = next(row for row in metrics if row["route_id"] == "realistic_sequential_dual_slm_4f")
    status = _json_load(root / "10_final_status" / "m2wf_outcome_report.json")

    assert realistic["classifier_label"] == "visual_hexagonal_field"
    assert realistic["strict_hexagon_eligible"] == "True"
    assert status["selected_outcome"] == "M2WF-A"


def test_supplementary_figure_is_compact_not_blank_canvas() -> None:
    root = _ensure_outputs()
    rows = _csv_rows(root / "09_supplementary" / "mode2w_fix_supplementary_table.csv")

    assert 6 <= len(rows) <= 14
    for suffix in (".png", ".pdf"):
        path = root / "09_supplementary" / f"figA1_supplementary_parameter_table{suffix}"
        assert path.exists()
        assert path.stat().st_size > 1000


def test_no_microfabrication_sample_plane_success_claim_is_made() -> None:
    root = _ensure_outputs()
    status = _json_load(root / "10_final_status" / "m2wf_outcome_report.json")
    manifest = _json_load(root / "10_final_status" / "mode2w_fix_manifest.json")
    doc = Path("docs/83_nathan_mode2w_fix_sequential_architecture.md").read_text(encoding="utf-8").lower()

    assert status["microfabrication_sample_plane_claim"] is False
    assert manifest["microfabrication_sample_plane_claim"] is False
    assert "no microfabrication/sample-plane success claim" in doc


def test_all_redesigned_figures_are_created() -> None:
    root = _ensure_outputs()
    manifest = _json_load(root / "10_final_status" / "mode2w_fix_manifest.json")

    assert set(manifest["figures"]) == {"fig1", "fig2", "fig3A", "fig3B", "fig3C", "fig4A", "fig4B", "fig4C", "fig5A", "fig5B", "figA1"}
    for paths in manifest["figures"].values():
        for raw in paths:
            path = Path(raw)
            assert path.exists()
            assert path.stat().st_size > 1000
