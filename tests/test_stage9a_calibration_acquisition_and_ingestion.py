"""Stage 9A calibration acquisition + measured-data ingestion tests.

All image arrays here are synthetic_unit_test_only / not_laboratory_data and are
written under pytest tmp_path (never the real calibration-run data directory).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from vbb_study.digital_twin.calibration_acquisition import (
    GOVERNANCE,
    build_calibration_campaign_v1,
    create_calibration_acquisition_package,
    generate_run_id,
    ingest_calibration_capture,
    write_capture_manifest,
    validate_capture_manifest,
    save_derived_artifact,
    sha256_of_file,
)
from vbb_study.digital_twin.measured_image_metrics import (
    compute_measured_image_metrics,
    image_quality_report,
    compare_measured_to_model,
)


# --- synthetic_unit_test_only fixtures (not_laboratory_data) ----------------

def _dark_frame(n=64):
    return (np.zeros((n, n)) + 3.0).astype(np.uint16)


def _centred_annulus(n=80, r0=12.0, w=3.0, amp=200.0):
    y, x = np.mgrid[0:n, 0:n]
    r = np.hypot(x - (n - 1) / 2.0, y - (n - 1) / 2.0)
    return (amp * np.exp(-((r - r0) ** 2) / (2 * w ** 2))).astype(np.uint16)


def _offcentre_annulus(n=80, cx=30, cy=50, r0=12.0, w=3.0, amp=200.0):
    y, x = np.mgrid[0:n, 0:n]
    r = np.hypot(x - cx, y - cy)
    return (amp * np.exp(-((r - r0) ** 2) / (2 * w ** 2))).astype(np.uint16)


def _saturated_annulus(n=80):
    y, x = np.mgrid[0:n, 0:n]
    r = np.hypot(x - (n - 1) / 2.0, y - (n - 1) / 2.0)
    a = 200000.0 * np.exp(-((r - 12.0) ** 2) / (2 * 3.0 ** 2))  # float, exceeds 16-bit range
    return np.clip(a, 0, 65535).astype(np.uint16)               # clip BEFORE cast (no overflow)


def _non_annular(n=80, w=10.0, amp=200.0):
    y, x = np.mgrid[0:n, 0:n]
    r = np.hypot(x - (n - 1) / 2.0, y - (n - 1) / 2.0)
    return (amp * np.exp(-(r ** 2) / (2 * w ** 2))).astype(np.uint16)  # solid Gaussian


# 1. Acquisition package creates required manifests and lab templates.
def test_acquisition_package_creates_required_files(tmp_path):
    rid = generate_run_id("t")
    paths = create_calibration_acquisition_package(
        rid, output_root=tmp_path / "outputs", data_root=tmp_path / "data")
    for key in ("run_manifest", "acquisition_plan", "capture_manifest_template",
                "hardware_profile_snapshot", "bench_inventory_snapshot",
                "coordinate_contract_snapshot"):
        assert Path(paths[key]).is_file(), key
    exp = paths["experiment_package_dir"]
    for name in ("bench_setup_sheet.md", "bench_setup_sheet.csv", "camera_capture_checklist.csv",
                 "energy_measurement_log.csv", "physical_axicon_alignment_log.csv",
                 "fused_silica_pilot_observation_template.csv", "operator_notes_template.md"):
        assert (exp / name).is_file(), name
    manifest = json.loads(Path(paths["run_manifest"]).read_text(encoding="utf-8"))
    assert manifest["physical_4f_status"] == "not_implemented_blocked"
    assert manifest["camera_status"].startswith("no_camera_model")
    assert manifest["governance"]["fourier_filter_physics_available"] is False


# 2. Run-id paths are unique; existing runs are not overwritten.
def test_run_not_overwritten(tmp_path):
    rid = generate_run_id("t")
    create_calibration_acquisition_package(rid, output_root=tmp_path / "o", data_root=tmp_path / "d")
    with pytest.raises(FileExistsError):
        create_calibration_acquisition_package(rid, output_root=tmp_path / "o", data_root=tmp_path / "d")
    assert generate_run_id() != generate_run_id()


# 3. Capture manifest validates required metadata; rejects missing critical fields.
def test_capture_manifest_validation():
    good = [{"capture_id": "c1", "capture_kind": "dark_frame", "run_id": "r", "file_path": "x.npy",
             "camera_frame_id": "camera_sensor_pixel_frame", "image_units": "pixel",
             "profile_name": "p", "route_mode": "holographic_cslm", "order_handoff_mode": "none",
             "capture_status": "ingested"}]
    assert validate_capture_manifest(good) == []
    bad_kind = [dict(good[0], capture_kind="bogus")]
    assert any("invalid capture_kind" in i for i in validate_capture_manifest(bad_kind))
    missing = [dict(good[0], file_path="")]
    assert any("missing required field" in i for i in validate_capture_manifest(missing))
    dup = good + [dict(good[0])]
    assert any("duplicate capture_id" in i for i in validate_capture_manifest(dup))


# 4. Raw ingestion preserves checksum and raw source evidence.
def test_ingestion_preserves_checksum(tmp_path):
    src = tmp_path / "synthetic_unit_test_only.npy"
    np.save(src, _centred_annulus())
    src_sha = sha256_of_file(src)
    data_dir = tmp_path / "data" / "run1"
    cap = ingest_calibration_capture(data_dir, src, capture_id="c1",
                                     capture_kind="post_axicon_xy", run_id="run1")
    assert cap.raw_file_sha256 == src_sha
    assert Path(cap.file_path).is_file()
    assert sha256_of_file(cap.file_path) == src_sha   # raw copy is byte-identical
    assert cap.capture_status == "ingested"
    # re-ingest must not overwrite
    with pytest.raises(FileExistsError):
        ingest_calibration_capture(data_dir, src, capture_id="c1",
                                   capture_kind="post_axicon_xy", run_id="run1")


# 5. Derived preprocessing never overwrites raw data.
def test_derived_does_not_touch_raw(tmp_path):
    src = tmp_path / "syn.npy"; np.save(src, _centred_annulus())
    data_dir = tmp_path / "data" / "run2"
    cap = ingest_calibration_capture(data_dir, src, capture_id="c1",
                                     capture_kind="post_axicon_xy", run_id="run2")
    raw_sha_before = sha256_of_file(cap.file_path)
    bg_sub = np.load(cap.file_path).astype(float) - 3.0
    out = save_derived_artifact(data_dir, "c1", "background_subtract", bg_sub, params={"bg": 3.0})
    assert Path(out).parent.name == "derived"
    assert sha256_of_file(cap.file_path) == raw_sha_before   # raw unchanged
    assert (data_dir / "derived" / "processing_manifest.csv").is_file()


# 6. Pixel metrics remain labelled in pixels without a calibrated mapping.
def test_pixel_metrics_labelled_pixels():
    m = compute_measured_image_metrics(_centred_annulus().astype(float), bit_depth=16)
    assert m["units"] == "pixel"
    assert m["coordinate_status"] == "pixel_only_uncalibrated"
    assert "ring_radius_px" in m and "ring_radius_um" not in m


# 7. Physical-unit metrics blocked when coordinate calibration unknown.
def test_physical_metrics_blocked_uncalibrated():
    m = compute_measured_image_metrics(_centred_annulus().astype(float), bit_depth=16,
                                       coordinate_calibrated=False, camera_scale_um_per_px=0.5)
    assert m["physical_metrics_status"] == "blocked_coordinate_uncalibrated"
    assert "ring_radius_um" not in m
    m2 = compute_measured_image_metrics(_centred_annulus().astype(float), bit_depth=16,
                                        coordinate_calibrated=True, camera_scale_um_per_px=0.5)
    assert m2["physical_metrics_status"] == "calibrated_camera_scale_declared"
    assert "ring_radius_um" in m2
    cmp = compare_measured_to_model(m, {"ring_radius_um": 4.0})
    assert cmp["comparison_status"] == "comparison_not_physically_calibrated"


# 8. Synthetic annulus produces valid ring metrics.
def test_synthetic_annulus_ring_metrics():
    m = compute_measured_image_metrics(_centred_annulus(n=80, r0=12.0).astype(float), bit_depth=16)
    assert m["is_annular"] is True
    assert abs(m["ring_centre_x_px"] - 39.5) < 2.0
    assert abs(m["ring_radius_px"] - 12.0) < 2.0
    off = compute_measured_image_metrics(_offcentre_annulus(cx=30, cy=50).astype(float), bit_depth=16)
    assert abs(off["ring_centre_x_px"] - 30) < 3.0 and abs(off["ring_centre_y_px"] - 50) < 3.0


# 9. Non-annular / saturated images receive appropriate quality flags.
def test_quality_flags_for_bad_images():
    nonann = image_quality_report(_non_annular().astype(float), bit_depth=16, expect_annular=True)
    assert nonann["is_annular"] is False
    assert "not_annular" in nonann["flags"] or "expected_annular_but_not_detected" in nonann["flags"]
    sat = image_quality_report(_saturated_annulus().astype(float), bit_depth=16)
    assert "saturated" in sat["flags"]
    assert sat["quality_status"] == "rejected"
    dark = compute_measured_image_metrics(_dark_frame().astype(float), bit_depth=16)
    assert dark["is_annular"] is False


# 11. No new code enables physical 4F / camera physics / material / inverse / AI.
def test_no_unsafe_enabling():
    assert GOVERNANCE["fourier_filter_physics_available"] is False
    assert GOVERNANCE["camera_model_enabled"] is False
    assert GOVERNANCE["material_model_enabled"] is False
    assert GOVERNANCE["final_export_allowed"] is False
    blob = json.dumps(build_calibration_campaign_v1()).lower()
    for token in ("neural", "inverse_correction", "thin_lens_applied", "fourier_filter_physics_available\": true",
                  "dose", "plasma", "thermal", "ablation", "material_prediction"):
        assert token not in blob
    # family 6 perturbations are planned-only
    fam6 = next(f for f in build_calibration_campaign_v1()["families"] if f["family_id"] == 6)
    assert fam6["status"] == "planned_future_calibration"
    assert fam6["implementation"] == "not_implemented_in_current_stage"
