from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vbb_study.calibration.schema import CalibrationBundle, canonical_calibration_template, measurement
from vbb_study.experimental.bench_dataset import load_experimental_dataset, sha256_file
from vbb_study.experimental.longitudinal import (
    build_measured_longitudinal_evidence,
    select_longitudinal_stack,
    write_measured_longitudinal_evidence,
)


def _dataset(tmp_path: Path, *, exposures: tuple[float | None, ...]) -> tuple[Path, list[str]]:
    n = 21
    coord = np.arange(n, dtype=float) - n // 2
    X, Y = np.meshgrid(coord, coord, indexing="xy")
    np.save(tmp_path / "bg.npy", np.zeros((n, n)))
    bg_hash = sha256_file(tmp_path / "bg.npy")
    frames = [
        {
            "frame_id": "bg",
            "case_id": "B0",
            "role": "camera_background",
            "path": "bg.npy",
            "sha256": bg_hash,
            "quantitative": True,
        }
    ]
    ids: list[str] = []
    for index, (z_m, exposure) in enumerate(zip((0.020, 0.030, 0.040), exposures)):
        image = np.exp(-0.08 * ((X - index) ** 2 + Y ** 2)) * 10000.0
        name = f"z{index}.npy"
        np.save(tmp_path / name, image)
        frame_id = f"z{index}"
        ids.append(frame_id)
        row = {
            "frame_id": frame_id,
            "case_id": "B0",
            "role": "camera_intensity",
            "path": name,
            "sha256": sha256_file(tmp_path / name),
            "z_m": z_m,
            "background_frame_id": "bg",
            "quantitative": True,
        }
        if exposure is not None:
            row["exposure_s"] = exposure
        frames.append(row)
    manifest = tmp_path / "dataset.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_id": "longitudinal_test",
                "data_classification": "synthetic_not_experimental",
                "canonical_z_ref_m": 0.030,
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )
    return manifest, ids


def _calibration() -> CalibrationBundle:
    data = canonical_calibration_template()
    data["data_classification"] = "synthetic_not_experimental"
    data["camera"]["object_plane_scale_m_per_pixel"] = measurement(5e-6, 0.0, "synthetic_measurement", "m/pixel")
    data["camera"]["rotation_deg"] = measurement(0.0, 0.0, "synthetic_measurement", "deg")
    data["camera"]["centre_pixel"] = [10.0, 10.0]
    return CalibrationBundle(data)


def test_measured_longitudinal_requires_one_consistent_exposure(tmp_path: Path) -> None:
    manifest, _ = _dataset(tmp_path, exposures=(0.001, 0.001, 0.002))
    dataset = load_experimental_dataset(manifest)
    with pytest.raises(ValueError, match="one exposure"):
        select_longitudinal_stack(dataset, case_id="B0")


def test_measured_longitudinal_refuses_unknown_exposure(tmp_path: Path) -> None:
    manifest, _ = _dataset(tmp_path, exposures=(0.001, None, 0.001))
    dataset = load_experimental_dataset(manifest)
    with pytest.raises(ValueError, match="exposure_s"):
        select_longitudinal_stack(dataset, case_id="B0")


def test_measured_longitudinal_builds_fixed_camera_axis_heatmap(tmp_path: Path) -> None:
    manifest, ids = _dataset(tmp_path, exposures=(0.001, 0.001, 0.001))
    dataset = load_experimental_dataset(manifest)
    evidence = build_measured_longitudinal_evidence(dataset, _calibration(), case_id="B0")
    assert evidence.frame_ids == tuple(ids)
    assert evidence.xz_intensity.shape == (3, 21)
    assert evidence.yz_intensity.shape == (3, 21)
    assert np.all(np.diff(evidence.z_m) > 0.0)
    assert evidence.metadata["radiometric_policy"].startswith("identical exposure required")
    payload = write_measured_longitudinal_evidence(
        evidence,
        tmp_path / "out",
        canonical_z_ref_m=0.030,
    )
    assert payload["canonical_z_ref_m"] == 0.030
    assert (tmp_path / "out" / "B0__measured_longitudinal.png").is_file()
    assert (tmp_path / "out" / "B0__measured_longitudinal.npz").is_file()
