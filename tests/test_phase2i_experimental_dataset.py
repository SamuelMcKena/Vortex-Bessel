from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vbb_study.calibration.schema import CalibrationBundle, canonical_calibration_template, measurement
from vbb_study.experimental.bench_dataset import (
    dataset_readiness,
    load_experimental_dataset,
    load_intensity_frame,
    sha256_file,
    verify_dataset_hashes,
)


def _write_manifest(root: Path, frames: list[dict], *, classification: str = "synthetic_not_experimental") -> Path:
    path = root / "dataset.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_id": "phase2i_test",
                "data_classification": classification,
                "canonical_z_ref_m": 0.030,
                "frames": frames,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _npy_frame(root: Path, name: str, values: np.ndarray) -> tuple[str, str]:
    path = root / name
    np.save(path, np.asarray(values, dtype=float))
    return path.name, sha256_file(path)


def test_dataset_ingests_hash_verified_lossless_numeric_frame(tmp_path: Path) -> None:
    rel, digest = _npy_frame(tmp_path, "frame.npy", np.arange(16, dtype=float).reshape(4, 4))
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "frame_id": "cam_z30",
                "case_id": "B0",
                "role": "camera_intensity",
                "path": rel,
                "sha256": digest,
                "z_m": 0.030,
                "quantitative": True,
            }
        ],
    )
    dataset = load_experimental_dataset(manifest)
    assert verify_dataset_hashes(dataset)["all_match"] is True
    image = load_intensity_frame(dataset, dataset.frame("cam_z30"))
    assert image.shape == (4, 4)
    assert np.array_equal(image, np.arange(16, dtype=float).reshape(4, 4))


def test_dataset_detects_file_tampering(tmp_path: Path) -> None:
    rel, digest = _npy_frame(tmp_path, "frame.npy", np.ones((4, 4)))
    manifest = _write_manifest(
        tmp_path,
        [{"frame_id": "f", "case_id": "B0", "role": "camera_intensity", "path": rel, "sha256": digest, "z_m": 0.030}],
    )
    dataset = load_experimental_dataset(manifest)
    np.save(tmp_path / rel, np.zeros((4, 4)))
    assert verify_dataset_hashes(dataset)["all_match"] is False


def test_quantitative_png_screenshot_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "screenshot.png"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(path)
    manifest = _write_manifest(
        tmp_path,
        [{"frame_id": "screen", "case_id": "B0", "role": "camera_intensity", "path": path.name, "sha256": sha256_file(path), "z_m": 0.030, "quantitative": True}],
    )
    with pytest.raises(ValueError, match="allowed quantitative"):
        load_experimental_dataset(manifest)


def test_dataset_path_cannot_escape_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_phase2i.npy"
    np.save(outside, np.ones((3, 3)))
    manifest = _write_manifest(
        tmp_path,
        [{"frame_id": "escape", "case_id": "B0", "role": "camera_intensity", "path": f"../{outside.name}", "sha256": sha256_file(outside), "z_m": 0.030}],
    )
    try:
        with pytest.raises(ValueError, match="escapes dataset root"):
            load_experimental_dataset(manifest)
    finally:
        outside.unlink(missing_ok=True)


def test_laboratory_readiness_requires_real_camera_coordinates_background_and_zref(tmp_path: Path) -> None:
    frame_rel, frame_hash = _npy_frame(tmp_path, "z30.npy", np.ones((4, 4)))
    frame2_rel, frame2_hash = _npy_frame(tmp_path, "z40.npy", np.ones((4, 4)) * 2.0)
    bg_rel, bg_hash = _npy_frame(tmp_path, "background.npy", np.zeros((4, 4)))
    manifest = _write_manifest(
        tmp_path,
        [
            {"frame_id": "bg", "case_id": "B0", "role": "camera_background", "path": bg_rel, "sha256": bg_hash, "quantitative": True},
            {"frame_id": "z30", "case_id": "B0", "role": "camera_intensity", "path": frame_rel, "sha256": frame_hash, "z_m": 0.030, "background_frame_id": "bg", "quantitative": True},
            {"frame_id": "z40", "case_id": "B0", "role": "camera_intensity", "path": frame2_rel, "sha256": frame2_hash, "z_m": 0.040, "background_frame_id": "bg", "quantitative": True},
        ],
        classification="laboratory_measurement",
    )
    dataset = load_experimental_dataset(manifest)
    template = canonical_calibration_template()
    template["data_classification"] = "laboratory_measurement"
    template["camera"]["object_plane_scale_m_per_pixel"] = measurement(5.5e-6, 0.1e-6, "measured_target", "m/pixel")
    template["camera"]["rotation_deg"] = measurement(0.2, 0.05, "measured_target", "deg")
    template["camera"]["centre_pixel"] = [1.5, 1.5]
    readiness = dataset_readiness(dataset, calibration_bundle=CalibrationBundle(template))
    assert readiness["absolute_calibrated_comparison_ready"] is True
    assert readiness["gates"]["measured_longitudinal_stack"]["complete"] is True
    assert readiness["agreement_acceptance_defined"] is False
