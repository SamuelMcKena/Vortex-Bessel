"""Stage 9A.1B downstream carrier/stop characterisation tests."""

import csv
import json
from pathlib import Path

import numpy as np

from vbb_study.digital_twin.bench_inventory import (
    DOWNSTREAM_EMPIRICAL_EVIDENCE_STATE,
    downstream_empirical_carrier_stop_evidence_effect,
    evaluate_physical_4f_readiness,
)
from vbb_study.digital_twin.downstream_carrier_stop import (
    CAPTURE_FAMILIES,
    DOWNSTREAM_CAMERA_WARNING,
    OPERATING_POINT_BOUNDARY_LABELS,
    DownstreamCarrierStopConfig,
    build_calibration_access_modes,
    build_downstream_carrier_stop_study,
    build_downstream_operating_point_summary,
    compute_downstream_image_metrics,
    create_downstream_carrier_stop_characterisation_session,
    downstream_capture_plan_rows,
    validate_downstream_capture_rows,
)
from vbb_study.digital_twin.slm_calibration_masks import (
    DIRECT_CALIBRATION_MODE,
    build_carrier_calibration_study,
)


def test_calibration_modes_exist_with_distinct_conclusions():
    modes = build_calibration_access_modes()
    assert set(modes) == {"direct_fourier_plane_access", "downstream_focus_empirical"}
    direct = modes["direct_fourier_plane_access"]
    downstream = modes["downstream_focus_empirical"]
    assert direct["mode_id"] == DIRECT_CALIBRATION_MODE
    assert "observed zero/+1/-1 order positions in camera pixels" in direct["what_is_observable"]
    assert "final output centroid in camera pixels" in downstream["what_is_observable"]
    assert direct["camera_access_status"] != downstream["camera_access_status"]


def test_downstream_mode_cannot_claim_direct_fourier_plane_coordinates():
    downstream = build_calibration_access_modes()["downstream_focus_empirical"]
    blocked = "\n".join(downstream["what_is_not_observable"])
    assert "direct Fourier-plane order positions" in blocked
    assert "physical Fourier-plane x/y coordinates" in blocked
    assert "direct stop radius in Fourier-plane mm" in blocked
    assert "order positions in camera pixels" not in "\n".join(downstream["what_is_observable"])


def test_downstream_evidence_does_not_mark_physical_4f_ready():
    readiness = evaluate_physical_4f_readiness()
    assert readiness["fourier_filter_physics_available"] is False
    assert readiness["C_initial_scalar_4f_model"]["ready"] is False
    effect = readiness[DOWNSTREAM_EMPIRICAL_EVIDENCE_STATE]
    assert effect["physical_4f_readiness_effect"] == "does_not_mark_ready"
    assert "physical_4f_readiness_ready" in effect["cannot_support"]
    assert downstream_empirical_carrier_stop_evidence_effect(True)["available"] is True


def test_downstream_package_complete_and_templates_exist(tmp_path):
    paths = create_downstream_carrier_stop_characterisation_session(
        "stage9a1b_test",
        config=DownstreamCarrierStopConfig.demo(),
        output_root=tmp_path / "outputs",
        data_root=tmp_path / "data",
        save_overview_to=tmp_path / "overview.png",
    )
    for key in (
        "run_manifest",
        "acquisition_plan",
        "capture_manifest_template",
        "hardware_profile_snapshot",
        "bench_inventory_snapshot",
        "coordinate_contract_snapshot",
        "mask_atlas_figure",
    ):
        assert Path(paths[key]).is_file(), key
    exp = Path(paths["experiment_package_dir"])
    for name in (
        "LAB_README_DOWNSTREAM_CARRIER_STOP_SESSION.md",
        "bench_setup_sheet.md",
        "bench_setup_sheet.csv",
        "camera_capture_checklist.csv",
        "downstream_carrier_sweep_log.csv",
        "downstream_stop_sweep_log.csv",
        "downstream_response_observation_template.csv",
        "operator_notes_template.md",
    ):
        assert (exp / name).is_file(), name
    manifest = json.loads(Path(paths["run_manifest"]).read_text(encoding="utf-8"))
    assert manifest["calibration_mode"] == "downstream_focus_empirical"
    assert manifest["governance"]["fourier_filter_physics_available"] is False
    assert manifest["governance"]["final_export_allowed"] is False
    assert (Path(paths["data_dir"]) / "raw").is_dir()
    assert list((Path(paths["data_dir"]) / "raw").iterdir()) == []


def test_capture_manifest_requires_downstream_metadata(tmp_path):
    config = DownstreamCarrierStopConfig.demo()
    rows = downstream_capture_plan_rows(config, data_dir=tmp_path / "data")
    assert validate_downstream_capture_rows(rows) == []
    required = {
        "calibration_mode",
        "camera_plane_label",
        "camera_plane_relationship_to_fourier_plane",
        "physical_axicon_state",
        "downstream_optics_state",
        "fourier_stop_state",
        "fourier_stop_centre_command_or_stage_x",
        "fourier_stop_centre_command_or_stage_y",
        "fourier_stop_radius_command_or_aperture_label",
        "carrier_cycles_x",
        "carrier_cycles_y",
        "SLM1 mask ID",
        "SLM2 mask ID",
        "camera position",
        "exposure",
        "gain",
        "neutral-density/filter state",
        "laser energy setting where available",
        "manual notes",
    }
    assert required.issubset(rows[0].keys())
    broken = [dict(rows[0])]
    broken[0].pop("camera_plane_relationship_to_fourier_plane")
    assert validate_downstream_capture_rows(broken)


def test_capture_families_and_carriers_remain_command_domain(tmp_path):
    study = build_downstream_carrier_stop_study(DownstreamCarrierStopConfig.demo())
    assert [f["family_id"] for f in study["capture_families"]] == list(CAPTURE_FAMILIES)
    assert study["carrier_sweep_definition"]["units"] == "command_cycles_across_displayed_area"
    assert study["carrier_sweep_definition"]["physical_frequency_status"] == "uncalibrated_command_domain"
    rows = downstream_capture_plan_rows(DownstreamCarrierStopConfig.demo(), data_dir=tmp_path / "data")
    assert {r["capture_family"] for r in rows} == set(CAPTURE_FAMILIES)
    blob = json.dumps({"study": study, "rows": rows}).lower()
    assert "cycles/mm" not in blob
    assert "cycles_per_mm" not in blob


def test_downstream_metrics_and_operating_point_summary_are_pixel_only():
    image = np.zeros((64, 64), dtype=float)
    yy, xx = np.mgrid[:64, :64]
    image += 100.0 * np.exp(-(((xx - 35.0) ** 2 + (yy - 30.0) ** 2) / 40.0))
    metrics = compute_downstream_image_metrics(image, bit_depth=16)
    assert "centroid_x_px" in metrics
    assert "major_axis_second_moment_px2" in metrics
    assert metrics["claim_boundary_labels"] == list(OPERATING_POINT_BOUNDARY_LABELS)
    assert not any(k.endswith("_um") or k.endswith("_mm") for k in metrics)

    rows = [
        {
            "carrier_cycles_x": 8,
            "carrier_cycles_y": 0,
            "fourier_stop_centre_command_or_stage_x": "baseline",
            "fourier_stop_centre_command_or_stage_y": "baseline",
            "fourier_stop_radius_command_or_aperture_label": "baseline",
            "total_camera_counts": 10.0,
            "centroid_x_px": 31.0,
            "centroid_y_px": 32.0,
            "spot_or_ring_classification": "spot_or_non_annular",
        }
    ]
    summary = build_downstream_operating_point_summary(rows)
    assert summary[0]["claim_boundary_labels"] == list(OPERATING_POINT_BOUNDARY_LABELS)
    assert summary[0]["empirical_ranking"] == 1


def test_existing_stage9a1_direct_mode_remains_available():
    study = build_carrier_calibration_study()
    assert study["calibration_mode"] == "direct_fourier_plane_access"
    assert "temporary diagnostic camera/profiler/IR card/power meter" in "\n".join(
        study["operator_setup_requirements"]
    )
    assert study["governance"]["physical_4f_filter_modelled"] is False
    assert study["governance"]["final_export_allowed"] is False


def test_no_unsafe_models_become_active():
    study = build_downstream_carrier_stop_study(DownstreamCarrierStopConfig.demo())
    gov = study["governance"]
    assert gov["physical_4f_filter_modelled"] is False
    assert gov["fourier_filter_physics_available"] is False
    assert gov["camera_model_enabled"] is False
    assert gov["material_model_enabled"] is False
    assert gov["inverse_correction_enabled"] is False
    assert gov["ai_enabled"] is False
    assert gov["final_export_allowed"] is False

    blob = json.dumps(study).lower()
    forbidden_active = ("zernike", "phase diversity", "neural", "fused-silica prediction")
    assert not any(token in blob for token in forbidden_active)


def test_config_and_notebook_surface_stage9a1b():
    cfg = Path("configs/studies/cslm_carrier_stop_characterisation_downstream_v1.json")
    assert cfg.is_file()
    study = json.loads(cfg.read_text(encoding="utf-8"))
    assert study["calibration_mode"] == "downstream_focus_empirical"
    assert study["camera_plane_label"] == "downstream_final_focus"
    assert study["fourier_stop_state"] == "recorded_and_user_editable"
    assert study["physical_axicon_state"] == "recorded_and_user_editable"

    nb = json.loads(Path("notebooks/digital_twin/00_full_beam_to_write_cockpit_MVP.ipynb").read_text(encoding="utf-8"))
    text = "\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if "stage9a1b" in cell.get("metadata", {}).get("id", "").lower()
    )
    assert "Stage 9A.1B" in text
    assert DOWNSTREAM_CAMERA_WARNING in text


def test_capture_manifest_csv_headers_match_required_fields(tmp_path):
    paths = create_downstream_carrier_stop_characterisation_session(
        "stage9a1b_csv",
        config=DownstreamCarrierStopConfig.demo(),
        output_root=tmp_path / "outputs",
        data_root=tmp_path / "data",
    )
    with open(paths["capture_manifest_template"], newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
    for field in (
        "calibration_mode",
        "camera_plane_relationship_to_fourier_plane",
        "fourier_stop_state",
        "physical_axicon_state",
        "downstream_optics_state",
    ):
        assert field in headers
