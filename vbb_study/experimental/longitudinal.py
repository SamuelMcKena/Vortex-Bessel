"""Measured fixed-camera-axis longitudinal evidence from a calibrated z stack."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from vbb_study.calibration.camera_comparison import camera_axes, preprocess_camera_image
from vbb_study.calibration.schema import CalibrationBundle
from vbb_study.experimental.bench_dataset import (
    ExperimentalDataset,
    ExperimentalFrameRecord,
    load_intensity_frame,
    sha256_file,
)
from vbb_study.experimental.closure import camera_calibration_from_bundle
from vbb_study.reporting.evidence_conventions import DEFAULT_FIGURE_DPI, INTENSITY_CMAP, common_positive_peak


@dataclass(frozen=True)
class MeasuredLongitudinalEvidence:
    dataset_id: str
    case_id: str
    role: str
    analyzer_nominal_deg: float | None
    frame_ids: tuple[str, ...]
    z_m: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    xz_intensity: np.ndarray
    yz_intensity: np.ndarray
    exposure_s: float
    fixed_camera_row: int
    fixed_camera_col: int
    metadata: dict[str, Any]


def select_longitudinal_stack(
    dataset: ExperimentalDataset,
    *,
    case_id: str,
    role: str = "camera_intensity",
    analyzer_nominal_deg: float | None = None,
) -> tuple[ExperimentalFrameRecord, ...]:
    """Select one internally comparable measured z stack.

    A longitudinal map is not assembled across different beam cases, different
    analyzer states, unknown exposure, or repeated/ambiguous z positions.
    """

    if role not in {"camera_intensity", "analyzer_intensity"}:
        raise ValueError("longitudinal evidence requires camera_intensity or analyzer_intensity")
    records = [
        f
        for f in dataset.frames
        if f.quantitative and f.case_id == case_id and f.role == role
    ]
    if role == "analyzer_intensity":
        if analyzer_nominal_deg is None:
            raise ValueError("analyzer_nominal_deg is required for analyzer longitudinal evidence")
        records = [
            f for f in records
            if f.analyzer_nominal_deg is not None
            and math.isclose(float(f.analyzer_nominal_deg), float(analyzer_nominal_deg), abs_tol=1e-9)
        ]
    if len(records) < 2:
        raise ValueError("measured longitudinal evidence requires at least two frames in one case/state")
    if any(f.z_m is None for f in records):
        raise ValueError("every longitudinal frame requires physical z_m")
    records.sort(key=lambda f: float(f.z_m))
    z = np.asarray([float(f.z_m) for f in records], dtype=float)
    if np.any(np.diff(z) <= 0.0):
        raise ValueError("longitudinal stack z_m values must be unique and strictly increasing")
    if any(f.exposure_s is None for f in records):
        raise ValueError("longitudinal intensity comparison requires exposure_s on every frame")
    exposure = np.asarray([float(f.exposure_s) for f in records], dtype=float)
    if not np.allclose(exposure, exposure[0], rtol=1e-9, atol=0.0):
        raise ValueError("longitudinal stack must use one exposure or an explicit radiometric correction model")
    if any(f.background_frame_id is None for f in records):
        raise ValueError("every longitudinal frame must reference quantitative background evidence")
    return tuple(records)


def build_measured_longitudinal_evidence(
    dataset: ExperimentalDataset,
    calibration_bundle: CalibrationBundle,
    *,
    case_id: str,
    role: str = "camera_intensity",
    analyzer_nominal_deg: float | None = None,
) -> MeasuredLongitudinalEvidence:
    records = select_longitudinal_stack(
        dataset,
        case_id=case_id,
        role=role,
        analyzer_nominal_deg=analyzer_nominal_deg,
    )
    camera = camera_calibration_from_bundle(calibration_bundle)
    images: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    hashes: dict[str, str] = {}
    shape: tuple[int, int] | None = None
    for record in records:
        actual_hash = sha256_file(dataset.resolved_path(record))
        if actual_hash.lower() != record.sha256.lower():
            raise ValueError(f"{record.frame_id}: SHA-256 mismatch")
        measured = load_intensity_frame(dataset, record)
        background = load_intensity_frame(dataset, dataset.frame(str(record.background_frame_id)))
        if background.shape != measured.shape:
            raise ValueError(f"{record.frame_id}: background shape mismatch")
        corrected, valid, _ = preprocess_camera_image(
            measured,
            background=background,
            saturation_level=camera.saturation_level,
        )
        if shape is None:
            shape = corrected.shape
        elif corrected.shape != shape:
            raise ValueError("all longitudinal frames must have identical camera array shape")
        images.append(corrected)
        valid_masks.append(valid)
        hashes[record.frame_id] = actual_hash

    assert shape is not None
    _, _, X, Y = camera_axes(shape, camera)
    iy0, ix0 = np.unravel_index(int(np.argmin(X * X + Y * Y)), X.shape)
    x_m = np.asarray(X[iy0, :], dtype=float)
    y_m = np.asarray(Y[:, ix0], dtype=float)
    xz: list[np.ndarray] = []
    yz: list[np.ndarray] = []
    saturated_line_samples = 0
    for image, valid in zip(images, valid_masks):
        x_line = np.asarray(image[iy0, :], dtype=float)
        y_line = np.asarray(image[:, ix0], dtype=float)
        xv = np.asarray(valid[iy0, :], dtype=bool)
        yv = np.asarray(valid[:, ix0], dtype=bool)
        saturated_line_samples += int(np.count_nonzero(~xv)) + int(np.count_nonzero(~yv))
        xz.append(np.where(xv, x_line, np.nan))
        yz.append(np.where(yv, y_line, np.nan))

    return MeasuredLongitudinalEvidence(
        dataset_id=dataset.dataset_id,
        case_id=case_id,
        role=role,
        analyzer_nominal_deg=analyzer_nominal_deg,
        frame_ids=tuple(f.frame_id for f in records),
        z_m=np.asarray([float(f.z_m) for f in records], dtype=float),
        x_m=x_m,
        y_m=y_m,
        xz_intensity=np.asarray(xz, dtype=float),
        yz_intensity=np.asarray(yz, dtype=float),
        exposure_s=float(records[0].exposure_s),
        fixed_camera_row=int(iy0),
        fixed_camera_col=int(ix0),
        metadata={
            "data_classification": dataset.data_classification,
            "camera_axis_policy": "nearest measured camera row/column to calibrated lab origin; no invented subpixel camera data",
            "frame_sha256": hashes,
            "saturated_or_invalid_line_sample_count": saturated_line_samples,
            "radiometric_policy": "identical exposure required; no uncalibrated frame-by-frame gain normalization",
        },
    )


def write_measured_longitudinal_evidence(
    evidence: MeasuredLongitudinalEvidence,
    output_dir: str | Path,
    *,
    canonical_z_ref_m: float | None = None,
) -> dict[str, Any]:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    common_peak = common_positive_peak([evidence.xz_intensity, evidence.yz_intensity])
    xz = evidence.xz_intensity / common_peak
    yz = evidence.yz_intensity / common_peak
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.2), constrained_layout=True)
    im = axes[0].imshow(
        xz,
        origin="lower",
        extent=[float(np.nanmin(evidence.x_m)) * 1e3, float(np.nanmax(evidence.x_m)) * 1e3, float(evidence.z_m[0]) * 1e3, float(evidence.z_m[-1]) * 1e3],
        cmap=INTENSITY_CMAP,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
    )
    axes[1].imshow(
        yz,
        origin="lower",
        extent=[float(np.nanmin(evidence.y_m)) * 1e3, float(np.nanmax(evidence.y_m)) * 1e3, float(evidence.z_m[0]) * 1e3, float(evidence.z_m[-1]) * 1e3],
        cmap=INTENSITY_CMAP,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
    )
    if canonical_z_ref_m is not None and evidence.z_m[0] <= canonical_z_ref_m <= evidence.z_m[-1]:
        for ax in axes:
            ax.axhline(float(canonical_z_ref_m) * 1e3, color="white", linestyle="--", linewidth=0.8, alpha=0.85)
    axes[0].set_title("measured fixed-camera-row x-z")
    axes[1].set_title("measured fixed-camera-column y-z")
    axes[0].set_xlabel("physical camera-row coordinate (mm)")
    axes[1].set_xlabel("physical camera-column coordinate (mm)")
    axes[0].set_ylabel("z (mm)")
    axes[1].set_ylabel("z (mm)")
    fig.colorbar(im, ax=axes, shrink=0.82, label="measured I / common stack peak")
    fig.suptitle(
        f"Phase 2I measured longitudinal evidence: {evidence.case_id}\n"
        f"{len(evidence.frame_ids)} calibrated z planes; exposure={evidence.exposure_s:g} s",
        fontsize=13,
    )
    stem = f"{evidence.case_id}__measured_longitudinal"
    png = outdir / f"{stem}.png"
    fig.savefig(png, dpi=DEFAULT_FIGURE_DPI)
    plt.close(fig)

    npz = outdir / f"{stem}.npz"
    np.savez_compressed(
        npz,
        z_m=evidence.z_m,
        x_m=evidence.x_m,
        y_m=evidence.y_m,
        xz_intensity=evidence.xz_intensity,
        yz_intensity=evidence.yz_intensity,
    )
    payload = {
        "outcome": "PHASE2I-MEASURED-LONGITUDINAL-EVIDENCE",
        "dataset_id": evidence.dataset_id,
        "data_classification": evidence.metadata.get("data_classification"),
        "case_id": evidence.case_id,
        "role": evidence.role,
        "analyzer_nominal_deg": evidence.analyzer_nominal_deg,
        "frame_ids": list(evidence.frame_ids),
        "z_m": evidence.z_m.tolist(),
        "canonical_z_ref_m": canonical_z_ref_m,
        "exposure_s": evidence.exposure_s,
        "common_stack_peak_counts": common_peak,
        "camera_axis_policy": evidence.metadata.get("camera_axis_policy"),
        "radiometric_policy": evidence.metadata.get("radiometric_policy"),
        "saturated_or_invalid_line_sample_count": evidence.metadata.get("saturated_or_invalid_line_sample_count"),
        "files": {"figure": png.name, "arrays": npz.name},
    }
    (outdir / f"{stem}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "MeasuredLongitudinalEvidence",
    "build_measured_longitudinal_evidence",
    "select_longitudinal_stack",
    "write_measured_longitudinal_evidence",
]
