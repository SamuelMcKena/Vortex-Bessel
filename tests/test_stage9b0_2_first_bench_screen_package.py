"""Stage 9B.0.2 first bench screen package tests."""

from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

from vbb_study.digital_twin.downstream_carrier_stop import (
    CAPTURE_FAMILIES,
    DownstreamCarrierStopConfig,
)
from vbb_study.digital_twin.first_bench_screen import (
    DEFAULT_FIRST_SCREEN_CANDIDATE_IDS,
    FIRST_BENCH_CAPTURE_COLUMNS,
    OPERATOR_SET_REQUIRED_FIELDS,
    OPTIONAL_EXTENSION_CANDIDATE_IDS,
    FirstBenchScreenConfig,
    build_first_bench_capture_plan_rows,
    build_first_bench_screen_config,
    create_first_bench_screen_package,
    plot_first_bench_screen_mask_atlas,
    plot_first_bench_screen_overview,
    validate_first_bench_capture_rows,
)


ROOT = Path(__file__).resolve().parents[1]


def _runtime_root() -> Path:
    root = ROOT / "outputs" / "pytest_runtime" / f"stage9b02_{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _demo_package():
    root = _runtime_root()
    return create_first_bench_screen_package(
        "stage9b02_test",
        config=FirstBenchScreenConfig(),
        downstream_config=DownstreamCarrierStopConfig.demo(),
        output_root=root / "outputs",
        data_root=root / "data",
    )


def test_static_contract_limits_default_candidates_and_keeps_extensions_optional():
    contract = build_first_bench_screen_config()
    assert contract["stage"] == "9B.0.2"
    assert tuple(contract["default_first_screen_candidate_ids"]) == DEFAULT_FIRST_SCREEN_CANDIDATE_IDS
    assert DEFAULT_FIRST_SCREEN_CANDIDATE_IDS == (
        "gaussian_reference",
        "vortex_ell_1",
        "vortex_ell_2",
    )
    assert tuple(contract["optional_later_extension_candidate_ids"]) == OPTIONAL_EXTENSION_CANDIDATE_IDS
    assert contract["include_optional_extensions_by_default"] is False
    assert "vortex_ell_3" not in contract["default_first_screen_candidate_ids"]
    assert "vortex_ell_4" not in contract["default_first_screen_candidate_ids"]


def test_capture_plan_preserves_downstream_baseline_families_and_default_candidates():
    rows = build_first_bench_capture_plan_rows(downstream_config=DownstreamCarrierStopConfig.demo())
    assert validate_first_bench_capture_rows(rows) == []
    families = {row["capture_family"] for row in rows}
    assert set(CAPTURE_FAMILIES).issubset(families)
    candidate_ids = [row["candidate_id"] for row in rows if row["capture_phase"] == "C_candidate_screen"]
    assert set(candidate_ids) == set(DEFAULT_FIRST_SCREEN_CANDIDATE_IDS)
    assert "vortex_ell_3" not in candidate_ids
    assert "vortex_ell_4" not in candidate_ids


def test_candidate_rows_show_slm1_to_slm2_bridge_and_slm2_carrier_only():
    rows = build_first_bench_capture_plan_rows(downstream_config=DownstreamCarrierStopConfig.demo())
    candidate_rows = [row for row in rows if row["capture_phase"] == "C_candidate_screen"]
    assert candidate_rows
    for row in candidate_rows:
        assert row["upstream source mode"] == "existing_cslm_component_route"
        assert row["slm1_phase_applied_at_slm1"] == "true"
        assert row["slm1_to_slm2_propagation_included"] == "true"
        assert row["slm2_carrier_applied_at_slm2"] == "true"
        assert row["slm2_contains_vortex"] == "false"
        assert row["slm2_contains_axicon"] == "false"
        assert row["carrier realism label"] == "ideal_continuous_phase_ramp"


def test_package_structure_and_required_templates_exist():
    paths = _demo_package()
    package_dir = Path(paths["run_dir"])
    for key in (
        "run_manifest",
        "first_bench_screen_manifest",
        "capture_plan",
        "hardware_profile_snapshot",
        "candidate_atlas_snapshot",
        "nominal_4f_profile_snapshot",
        "claim_boundary",
        "raw_policy",
        "overview_figure",
        "mask_atlas_figure",
    ):
        assert Path(paths[key]).is_file(), key
    assert (package_dir / "phase_masks" / "slm1").is_dir()
    assert (package_dir / "phase_masks" / "slm2").is_dir()
    exp = package_dir / "experiment_package"
    for name in (
        "LAB_README_FIRST_BENCH_SCREEN.md",
        "operator_checklist.md",
        "as_found_bench_record.csv",
        "carrier_stop_baseline_log.csv",
        "candidate_screen_log.csv",
        "camera_capture_log.csv",
        "candidate_observation_template.csv",
        "operator_notes_template.md",
    ):
        assert (exp / name).is_file(), name
    for candidate_id in DEFAULT_FIRST_SCREEN_CANDIDATE_IDS:
        assert (exp / "candidate_summaries" / f"{candidate_id}.md").is_file()


def test_capture_plan_links_actual_mask_files_for_planned_non_dark_captures():
    paths = _demo_package()
    package_dir = Path(paths["run_dir"])
    with open(paths["capture_plan"], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    for field in FIRST_BENCH_CAPTURE_COLUMNS:
        assert field in rows[0]
    for row in rows:
        if row["SLM1 mask ID"] == "shuttered":
            continue
        assert (package_dir / row["SLM1 phase file"]).is_file(), row["SLM1 phase file"]
        assert (package_dir / row["SLM1 grayscale PNG"]).is_file(), row["SLM1 grayscale PNG"]
        assert (package_dir / row["SLM2 phase file"]).is_file(), row["SLM2 phase file"]
        assert (package_dir / row["SLM2 grayscale PNG"]).is_file(), row["SLM2 grayscale PNG"]


def test_candidate_metadata_labels_command_masks_unvalidated_and_no_axicon_on_slm2():
    paths = _demo_package()
    package_dir = Path(paths["run_dir"])
    manifest = json.loads(Path(paths["first_bench_screen_manifest"]).read_text(encoding="utf-8"))
    for candidate_id in DEFAULT_FIRST_SCREEN_CANDIDATE_IDS:
        candidate = manifest["candidate_manifests"][candidate_id]
        assert candidate["hardware_command_export_status"] == "command_masks_exportable_unvalidated"
        assert candidate["bench_validation_status"] == "not_bench_validated"
        assert candidate["physical_4f_readiness"] == "blocked"
        assert candidate["slm1_phase_applied_at_slm1"] is True
        assert candidate["slm1_to_slm2_propagation_included"] is True
        assert candidate["slm2_carrier_mode"] == "ideal_continuous_phase_ramp"
    slm2_meta_files = list((package_dir / "phase_masks" / "slm2").glob("*candidate_carrier*_metadata.json"))
    assert slm2_meta_files
    for path in slm2_meta_files:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        assert metadata["contains_vortex"] is False
        assert metadata["contains_axicon"] is False
        assert metadata["hardware_command_export_status"] == "command_masks_exportable_unvalidated"


def test_operator_unknowns_and_raw_policy_are_explicit():
    paths = _demo_package()
    package_dir = Path(paths["run_dir"])
    with open(package_dir / "experiment_package" / "as_found_bench_record.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {row["required_field"] for row in rows} == set(OPERATOR_SET_REQUIRED_FIELDS)
    assert all(row["value_status"] == "unknown_recorded" for row in rows)
    raw_policy = Path(paths["raw_policy"]).read_text(encoding="utf-8")
    assert "does not create fake raw images" in raw_policy
    assert "do not commit raw images" in raw_policy
    assert list((Path(paths["data_dir"]) / "raw").iterdir()) == []
    assert list((package_dir / "raw").iterdir()) == [package_dir / "raw" / "README_RAW_DATA_POLICY.md"]


def test_package_does_not_activate_forbidden_models():
    paths = _demo_package()
    manifest = json.loads(Path(paths["run_manifest"]).read_text(encoding="utf-8"))
    gov = manifest["governance"]
    assert gov["new_optical_physics_added"] is False
    assert gov["physical_4f_filter_modelled"] is False
    assert gov["physical_4f_readiness_ready"] is False
    assert gov["camera_model_enabled"] is False
    assert gov["inverse_correction_enabled"] is False
    assert gov["phase_diversity_enabled"] is False
    assert gov["zernike_fitting_enabled"] is False
    assert gov["ai_enabled"] is False
    assert gov["material_model_enabled"] is False
    assert gov["pixelated_slm_order_physics_modelled"] is False
    assert gov["raw_camera_images_processed"] is False
    assert gov["final_export_allowed"] is False


def test_static_config_docs_and_notebook_surface_stage9b02_boundary():
    cfg = json.loads((ROOT / "configs/studies/cslm_first_bench_screen_v1.json").read_text(encoding="utf-8"))
    assert cfg["stage"] == "9B.0.2"
    assert cfg["default_first_screen_candidate_ids"] == list(DEFAULT_FIRST_SCREEN_CANDIDATE_IDS)
    assert "command masks exportable but unvalidated" in cfg["claim_boundary"]
    doc = (ROOT / "docs/49_first_bench_screen_package.md").read_text(encoding="utf-8")
    assert "Stage 9B.0.2 First Bench Screen Package" in doc
    assert "vortex_ell_3` and `vortex_ell_4` are optional later extensions only" in doc
    notebook = json.loads((ROOT / "notebooks/digital_twin/02_first_bench_screen_package.ipynb").read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "Stage 9B.0.2 - First Bench Screen Package" in text
    assert "SLM2 owns the carrier-only command mask" in text


def test_stage9b02_figures_can_be_generated():
    root = _runtime_root()
    overview = plot_first_bench_screen_overview(output_path=root / "overview.png")
    atlas = plot_first_bench_screen_mask_atlas(output_path=root / "atlas.png")
    assert overview.is_file()
    assert atlas.is_file()


def test_optional_extension_requires_explicit_config():
    default_rows = build_first_bench_capture_plan_rows(downstream_config=DownstreamCarrierStopConfig.demo())
    assert not any(row["candidate_id"] in OPTIONAL_EXTENSION_CANDIDATE_IDS for row in default_rows)
    extended_rows = build_first_bench_capture_plan_rows(
        config=FirstBenchScreenConfig(include_optional_extensions=True),
        downstream_config=DownstreamCarrierStopConfig.demo(),
    )
    assert any(row["candidate_id"] == "vortex_ell_3" for row in extended_rows)
    assert any(row["candidate_id"] == "vortex_ell_4" for row in extended_rows)
