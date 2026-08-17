from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from vbb_study.digital_twin.phase2b_visual_cases import (
    PHASE2B_3D_CASE_IDS,
    PHASE2B_CASE_IDS,
    Phase2BConfig,
    phase2b_case_registry,
)
from vbb_study.digital_twin.phase2b_visual_diagnostics import (
    PHASE2B_ALLOWED_OUTCOMES,
    PHASE2B_SUBDIRS,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures" / "phase2b_visual_diagnostics"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_phase2b_registry_and_publication_sampling_are_fixed() -> None:
    registry = phase2b_case_registry()
    assert tuple(row["case_id"] for row in registry) == PHASE2B_CASE_IDS
    assert {row["case_id"] for row in registry if row["needs_3d"]} == {"B0", "V1", "V3", "H1"}
    config = Phase2BConfig()
    config.validate()
    assert config.scalar_grid_n >= 512
    assert config.hex_grid_n >= 1024
    assert config.hero_grid_n >= 1536
    assert config.z_values_m().size == 101


def test_phase2b_output_tree_and_required_machine_artifacts_exist() -> None:
    for subdir in PHASE2B_SUBDIRS:
        assert (OUT / subdir).is_dir()
    for relative in (
        "00_manifests/figure_provenance.csv",
        "00_manifests/figure_provenance.json",
        "00_manifests/phase2b_final_manifest.json",
        "08_summary_tables/phase2b_case_summary.csv",
        "08_summary_tables/phase2b_endpoint_reproduction_audit.csv",
        "09_final_reports/phase2b_outcome_report.json",
    ):
        assert (OUT / relative).is_file()
    assert (ROOT / "docs" / "91_phase2b_visual_diagnostics_and_beam_volume_maps.md").is_file()


def test_phase2b_outcome_is_a_and_upstream_outputs_are_unchanged() -> None:
    report = _json(OUT / "09_final_reports" / "phase2b_outcome_report.json")
    assert report["outcome"] in PHASE2B_ALLOWED_OUTCOMES
    assert report["outcome"] == "PHASE2B-A"
    assert report["upstream_outputs_unchanged"] is True
    assert report["upstream_hashes_before"] == report["upstream_hashes_after"]
    assert report["split_arm_architecture_reintroduced"] is False
    assert report["microfabrication_sample_plane_success_claim"] is False


def test_every_selected_case_has_native_xy_slice_and_profile_rendering() -> None:
    rows = _rows(OUT / "00_manifests" / "figure_provenance.csv")
    ids = {row["figure_id"] for row in rows}
    for case_id in ("G0", "B0", "V1", "V3", "H1_REALISTIC"):
        assert f"{case_id}_xy_bundle" in ids
        assert f"{case_id}_xz_yz" in ids
        assert f"{case_id}_profiles" in ids


def test_mandatory_3d_outputs_and_render_provenance_are_complete() -> None:
    rows = _rows(OUT / "00_manifests" / "figure_provenance.csv")
    by_id = {row["figure_id"]: row for row in rows}
    for case_id in PHASE2B_3D_CASE_IDS:
        row = by_id[f"{case_id}_3d_intensity"]
        assert Path(row["png_path"]).is_file()
        assert Path(row["pdf_path"]).is_file()
        assert "surface mesh" in row["render_downsampling"]
        assert "z60" in row["render_method"]
        assert "height and colour encode the same intensity" in row["normalisation_policy"]
        assert row["display_interpolation"] == "none"


def test_all_metrics_are_native_and_interpolation_is_display_only() -> None:
    provenance = _rows(OUT / "00_manifests" / "figure_provenance.csv")
    summary = _rows(OUT / "08_summary_tables" / "phase2b_case_summary.csv")
    assert provenance
    assert all(row["metrics_computed_on_native_arrays"].lower() == "true" for row in provenance)
    assert all(row["display_interpolation_used_for_metrics"].lower() == "false" for row in provenance)
    assert all(row["native_metrics_only"].lower() == "true" for row in summary)
    assert all(row["display_interpolation_used_for_metrics"].lower() == "false" for row in summary)


def test_paired_hex_figures_enforce_explicit_common_crop_and_colour_rules() -> None:
    rows = _rows(OUT / "00_manifests" / "figure_provenance.csv")
    paired = [row for row in rows if row["figure_id"].startswith("H1_") and row["paired_colour_crop_rule_id"]]
    assert {row["figure_id"] for row in paired} >= {
        "H1_continuous_averaged_early_mid_late",
        "H1_continuous_averaged_profiles",
        "H1_highn_continuous_averaged_z60",
        "H1_cross_route_realism_correction",
    }
    assert all("common" in row["paired_colour_crop_rule_id"] or "profiles" in row["paired_colour_crop_rule_id"] for row in paired)
    assert all(row["crop_rule"] for row in paired)
    assert all(row["normalisation_policy"] for row in paired)


def test_highn_hex_hero_uses_n1536_and_sas_only_for_display() -> None:
    rows = _rows(OUT / "00_manifests" / "figure_provenance.csv")
    hero = next(row for row in rows if row["figure_id"] == "H1_highn_continuous_averaged_z60")
    report = _json(OUT / "09_final_reports" / "phase2b_outcome_report.json")
    assert int(hero["native_grid_n"]) == 1536
    assert "scalable angular spectrum" in hero["render_method"]
    assert hero["display_interpolation_used_for_metrics"].lower() == "false"
    assert report["sas_used_for_focus_rendering_only"] is True


def test_endpoint_cases_reproduce_phase2a_mode2y_and_mode2z_exactly() -> None:
    rows = _rows(OUT / "08_summary_tables" / "phase2b_endpoint_reproduction_audit.csv")
    sources = {row["source"] for row in rows}
    assert sources == {
        "PHASE 2A stored scalar endpoint",
        "MODE 2Y N=1024 stored endpoint",
        "MODE 2Z-HN N=1536 stored endpoint",
    }
    assert len(rows) == 34
    assert all(row["reproduced"].lower() == "true" for row in rows)


def test_energy_plot_data_agrees_numerically_with_phase2a_ledger() -> None:
    source = _rows(ROOT / "outputs" / "validation" / "phase2a" / "canonical_power_ledgers.csv")
    plotted = _rows(OUT / "07_energy_ledgers" / "phase2a_energy_ledger_plot_data.csv")
    assert len(plotted) == len(source)
    source_by_key = {
        (row["case_id"], row["route_variant"], int(row["row_index"])): row for row in source
    }
    for row in plotted:
        key = (row["case_id"], row["route_variant"], int(row["row_index"]))
        expected = source_by_key[key]
        assert float(row["stage_efficiency"]) == float(expected["stage_efficiency"])
        assert float(row["cumulative_efficiency"]) == float(expected["cumulative_efficiency"])
        assert float(row["pulse_energy_J"]) == float(expected["pulse_energy_J"])
        assert row["first_order_efficiency_reapplied"].lower() == "false"


def test_continuous_hex_improves_all_three_reported_sharpness_observables() -> None:
    summary = _json(OUT / "06_hex_comparisons" / "continuous_vs_averaged_metric_summary.json")
    assert summary["continuous_improves_three_predeclared_sharpness_observables"] is True
    assert summary["continuous_edge_gradient_relative_improvement"] > 0.0
    assert summary["continuous_transition_width_relative_improvement"] > 0.0
    assert summary["continuous_ridge_fwhm_relative_improvement"] > 0.0


def test_publication_pngs_have_nontrivial_pixel_dimensions() -> None:
    rows = _rows(OUT / "00_manifests" / "figure_provenance.csv")
    assert len(rows) == 27
    for row in rows:
        with Image.open(row["png_path"]) as image:
            width, height = image.size
        assert width >= 2400
        assert height >= 1200
