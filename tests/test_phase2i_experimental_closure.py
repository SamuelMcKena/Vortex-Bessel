from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vbb_study.calibration.schema import CalibrationBundle, canonical_calibration_template, measurement
from vbb_study.experimental.bench_dataset import load_experimental_dataset, sha256_file
from vbb_study.experimental.closure import compare_record_to_simulation, write_closure_evidence


def test_identical_synthetic_camera_plane_closes_without_fitting_geometry(tmp_path: Path) -> None:
    coord = (np.arange(21, dtype=float) - 10.0) * 5.0e-6
    X, Y = np.meshgrid(coord, coord, indexing="xy")
    simulated = np.exp(-2.0 * ((X / 28e-6) ** 2 + (Y / 36e-6) ** 2))
    measured = simulated * 20000.0
    background = np.zeros_like(measured)
    np.save(tmp_path / "measured.npy", measured)
    np.save(tmp_path / "background.npy", background)
    manifest = {
        "schema_version": "1.0",
        "dataset_id": "synthetic_closure",
        "data_classification": "synthetic_not_experimental",
        "canonical_z_ref_m": 0.030,
        "frames": [
            {
                "frame_id": "bg",
                "case_id": "B0",
                "role": "camera_background",
                "path": "background.npy",
                "sha256": sha256_file(tmp_path / "background.npy"),
                "quantitative": True
            },
            {
                "frame_id": "camera_z30",
                "case_id": "B0",
                "role": "camera_intensity",
                "path": "measured.npy",
                "sha256": sha256_file(tmp_path / "measured.npy"),
                "z_m": 0.030,
                "background_frame_id": "bg",
                "quantitative": True
            }
        ]
    }
    manifest_path = tmp_path / "dataset.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    dataset = load_experimental_dataset(manifest_path)

    calibration = canonical_calibration_template()
    calibration["data_classification"] = "synthetic_not_experimental"
    calibration["camera"]["object_plane_scale_m_per_pixel"] = measurement(5.0e-6, 0.0, "synthetic_measurement", "m/pixel")
    calibration["camera"]["rotation_deg"] = measurement(0.0, 0.0, "synthetic_measurement", "deg")
    calibration["camera"]["centre_pixel"] = [10.0, 10.0]
    evidence = compare_record_to_simulation(
        dataset,
        dataset.frame("camera_z30"),
        CalibrationBundle(calibration),
        simulated_intensity=simulated,
        simulated_x_m=coord,
        simulated_y_m=coord,
    )
    metrics = evidence.comparison.metrics
    assert metrics["energy_normalised_correlation"] > 1.0 - 1e-12
    assert metrics["energy_normalised_l2"] < 1e-12
    assert metrics["centroid_error_m"] < 1e-15
    assert evidence.metadata["camera_geometry_fitted_to_image"] is False

    payload = write_closure_evidence(evidence, tmp_path / "closure")
    assert payload["agreement_acceptance_applied"] is False
    assert payload["agreement_acceptance_passed"] is None
    assert (tmp_path / "closure" / "camera_z30__camera_closure.png").is_file()
    assert (tmp_path / "closure" / "camera_z30__camera_closure.npz").is_file()
    assert (tmp_path / "closure" / "camera_z30__camera_closure.json").is_file()
