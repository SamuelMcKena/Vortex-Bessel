from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from vbb_study.digital_twin import (
    CANONICAL_OPERATING_POINT_ID,
    MODE2W_DEFAULT_OUTPUT_ROOT,
    OLD_BEST_COMPROMISE_ID,
    assert_not_forbidden,
    write_mode2w_annotated_master_figure_pack,
)

ROOT = Path(MODE2W_DEFAULT_OUTPUT_ROOT)


def _ensure_outputs() -> Path:
    manifest = ROOT / "mode2w_manifest.json"
    if not manifest.exists():
        write_mode2w_annotated_master_figure_pack(output_dir=ROOT, grid_n=256, z_planes=11)
    return ROOT


def _json_load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_all_five_main_figure_files_are_created() -> None:
    root = _ensure_outputs()

    for name in [
        "fig1_optical_system_annotated",
        "fig2_target_and_masks_annotated",
        "fig3_ideal_vs_actual_outputs",
        "fig4_propagation_and_power",
        "fig5_tolerance_correction_build_readiness",
    ]:
        for suffix in [".png", ".pdf"]:
            path = root / f"{name}{suffix}"
            assert path.exists()
            assert path.stat().st_size > 1000


def test_appendix_figure_is_created() -> None:
    root = _ensure_outputs()

    for suffix in [".png", ".pdf"]:
        path = root / f"figA1_appendix_metrics_and_provenance{suffix}"
        assert path.exists()
        assert path.stat().st_size > 1000


def test_canonical_operating_point_is_realistic_4f_reference() -> None:
    root = _ensure_outputs()
    rows = _json_load(root / "mode2w_operating_point_summary.json")

    assert rows[0]["candidate_id"] == CANONICAL_OPERATING_POINT_ID
    assert rows[0]["candidate_id"] == "REALISTIC_4F_HEXAGON_REFERENCE"
    assert rows[0]["strict_hexagon_eligible"] is True


def test_forbidden_old_optima_are_not_used() -> None:
    root = _ensure_outputs()
    manifest = _json_load(root / "mode2w_manifest.json")
    operating = _json_load(root / "mode2w_operating_point_summary.json")

    with pytest.raises(ValueError, match="strict hexagon gate"):
        assert_not_forbidden(OLD_BEST_COMPROMISE_ID)
    assert all(row["candidate_id"] != OLD_BEST_COMPROMISE_ID for row in operating)
    assert manifest["canonical_operating_point"] != OLD_BEST_COMPROMISE_ID
    assert manifest["secondary_operating_point"] != OLD_BEST_COMPROMISE_ID


def test_slm_mask_metadata_records_phase_convention() -> None:
    root = _ensure_outputs()
    metadata = _json_load(root / "mode2w_slm_mask_metadata.json")

    assert metadata["candidate_id"] == CANONICAL_OPERATING_POINT_ID
    assert "phi_H = +alpha" in metadata["phase_convention"]
    assert "phi_V = -alpha + pi/2" in metadata["phase_convention"]
    assert metadata["lut_applied"] is False
    assert metadata["hardware_ready"] is False


def test_optical_settings_table_contains_required_physical_settings() -> None:
    root = _ensure_outputs()
    rows = {row["setting"]: row for row in _csv_rows(root / "mode2w_optical_element_settings.csv")}

    assert rows["display_carrier"]["value"] == "6.25 lp/mm"
    assert rows["fourf_focal_length"]["value"] == "300 mm"
    assert "1.54" in rows["iris_diameter"]["value"]
    assert rows["axicon"]["value"] == "2 deg base, n = 1.458"


def test_power_ledger_closes_within_tolerance() -> None:
    root = _ensure_outputs()
    rows = {row["stage"]: row for row in _csv_rows(root / "mode2w_power_ledger.csv")}
    f = lambda stage: float(rows[stage]["model_fraction_of_input"])

    assert f("09_selected_plus1_order_h") + f("11_rejected_power_h") == pytest.approx(f("07_after_slm_h"))
    assert f("10_selected_plus1_order_v") + f("12_rejected_power_v") == pytest.approx(f("08_after_slm_v"))
    assert f("13_zero_order_total") + f("07_after_slm_h") + f("08_after_slm_v") == pytest.approx(1.0)
    assert f("20_useful_central_hexagon_power") + f("21_power_outside_useful_region") == pytest.approx(f("19_total_power_at_z60"))


def test_output_comparison_table_includes_ideal_and_realistic_routes() -> None:
    root = _ensure_outputs()
    route_ids = {row["route_id"] for row in _csv_rows(root / "mode2w_output_comparison_metrics.csv")}

    assert "M2P_M2N_ideal_dual_slm_qwp" in route_ids
    assert "realistic_dual_slm_4f" in route_ids
    assert CANONICAL_OPERATING_POINT_ID in route_ids


def test_build_readiness_summary_is_created() -> None:
    root = _ensure_outputs()
    summary = _json_load(root / "mode2w_build_readiness_summary.json")

    assert summary["selected_outcome"] == "M2W-A"
    assert summary["architecture_valid"] is True
    assert summary["source_scale_build_authorised"] is True
    assert summary["native_masks_exported"] is True
    assert summary["fourf_geometry_defined"] is True


def test_no_microfabrication_sample_plane_claim_is_made() -> None:
    root = _ensure_outputs()
    summary = _json_load(root / "mode2w_build_readiness_summary.json")
    manifest = _json_load(root / "mode2w_manifest.json")
    doc = Path("docs/82_nathan_mode2w_annotated_master_figure_pack.md").read_text(encoding="utf-8")

    assert summary["microfabrication_sample_plane_claim"] is False
    assert manifest["microfabrication_sample_plane_claim"] is False
    assert "no microfabrication/sample-plane success claim" in doc.lower()
