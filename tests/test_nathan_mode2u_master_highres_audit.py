from __future__ import annotations

import csv
import json
from pathlib import Path

from vbb_study.digital_twin import (
    MODE2U_DEFAULT_OUTPUT_ROOT,
    mode2u_visual_acceptability,
    write_mode2u_master_highres_audit,
)


ROOT = Path(MODE2U_DEFAULT_OUTPUT_ROOT)


def _ensure_outputs() -> Path:
    manifest = ROOT / "nathan_master_highres_manifest.json"
    if not manifest.exists():
        write_mode2u_master_highres_audit(
            output_dir=ROOT,
            grid_n=192,
            z_planes=5,
            run_compensation=False,
            optimisation_max_cases=8,
            run_high_n=False,
        )
    return ROOT


def test_mode2u_output_root_is_separate_from_earlier_outputs() -> None:
    root = _ensure_outputs()
    assert root.name == "nathan_mode2u_master_highres_audit"
    assert root != Path("outputs/figures/digital_twin/nathan_mode2s_lab_realism_tolerance")


def test_mode2u_master_manifest_exists_and_names_required_stages() -> None:
    root = _ensure_outputs()
    manifest = json.loads((root / "nathan_master_highres_manifest.json").read_text(encoding="utf-8"))

    assert manifest["stage"] == "nathan_mode2u_master_highres_audit"
    for stage in ["V0", "M2P", "M2N", "M2Q", "M2S", "MODE1_CONTRAST"]:
        assert stage in manifest["included_stages"]
    assert manifest["physics_changed"] is False
    assert manifest["render_interpolation"] == "lanczos"
    assert manifest["publication_sampling_min_fringe_samples"] >= 8.0


def test_mode2u_energy_ledger_files_exist_and_have_stable_schema() -> None:
    root = _ensure_outputs()
    csv_path = root / "06_energy_ledgers" / "energy_ledger_routes.csv"
    json_path = root / "06_energy_ledgers" / "energy_ledger_routes.json"

    assert csv_path.exists()
    assert json_path.exists()
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    required = {"route_id", "branch", "stage_index", "stage", "input_power_norm", "output_power_norm", "loss_reason"}
    assert required.issubset(rows[0])


def test_mode2u_publication_sampling_audit_separates_nyquist_from_publication() -> None:
    root = _ensure_outputs()
    rows = json.loads((root / "publication_sampling_audit.json").read_text(encoding="utf-8"))
    by_n = {int(row["grid_n"]): row for row in rows}

    assert by_n[1024]["nyquist_pass"] is True
    assert by_n[1024]["publication_recommended"] is False
    assert by_n[1536]["nyquist_pass"] is True
    assert by_n[1536]["publication_recommended"] is True
    assert by_n[1536]["samples_per_radial_fringe"] >= 8.0
    assert by_n[1536]["ring_diameter_pixels"] >= 30.0


def test_mode2u_optimal_hexagon_pareto_files_exist() -> None:
    root = _ensure_outputs()
    opt = root / "08_optimal_hexagon_sweep"
    for name in [
        "optimal_hexagon_pareto.csv",
        "optimal_hexagon_pareto.json",
        "optimal_hexagon_pareto_plot.png",
        "optimal_hexagon_best_shape.png",
        "optimal_hexagon_best_power.png",
        "optimal_hexagon_best_compromise.png",
        "optimal_hexagon_parameter_table.csv",
    ]:
        assert (opt / name).exists()


def test_mode2u_realistic_build_recommendation_json_exists() -> None:
    root = _ensure_outputs()
    rec = json.loads((root / "nathan_master_recommendation.json").read_text(encoding="utf-8"))

    assert rec["recommended_build_route"].startswith("dual-SLM")
    assert rec["six_polarizer_route_needed"] is False
    assert rec["source_scale_branch_validated"] is True


def test_mode2u_visual_audit_cannot_accept_triangular_dark_core() -> None:
    root = _ensure_outputs()
    assert mode2u_visual_acceptability("triangular_dark_core") is False
    assert mode2u_visual_acceptability("triangular_lobed_field") is False
    audit = json.loads((root / "nathan_master_visual_audit.json").read_text(encoding="utf-8"))
    for row in audit:
        if str(row["strict_class"]) in {"triangular_dark_core", "triangular_lobed_field"}:
            assert row["acceptable_hexagon"] is False


def test_mode2u_recommendation_does_not_claim_microfabrication_success() -> None:
    root = _ensure_outputs()
    manifest = json.loads((root / "nathan_master_highres_manifest.json").read_text(encoding="utf-8"))
    rec = json.loads((root / "nathan_master_recommendation.json").read_text(encoding="utf-8"))

    assert manifest["source_scale_branch_validated"] is True
    assert manifest["microfabrication_branch_validated"] is False
    assert manifest["microfabrication_sample_plane_claim"] is False
    assert rec["microfabrication_branch_validated"] is False


def test_mode2u_highres_regeneration_includes_required_stage_outputs() -> None:
    root = _ensure_outputs()
    expected = [
        root / "00_v0_reference" / "v0_reference_z60_highres.png",
        root / "01_m2p_preaxicon" / "m2p_dual_slm_qwp_vs_target_highres.png",
        root / "02_m2n_source_replication" / "m2n_route_dual_slm_4f_z60_highres.png",
        root / "03_m2q_inverse_masks" / "m2q_forward_verification_z60_highres.png",
        root / "04_m2s_lab_realism" / "m2s_clean_baseline_highres.png",
        root / "05_mode1_microfabrication_contrast" / "source_vs_micro_branch_contrast_highres.png",
    ]
    for path in expected:
        assert path.exists()
