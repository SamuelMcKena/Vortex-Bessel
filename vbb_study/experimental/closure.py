"""Quantitative measured-to-simulation closure on calibrated camera coordinates."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from vbb_study.calibration.camera_comparison import (
    CameraCalibration,
    CameraComparison,
    compare_simulation_to_camera,
)
from vbb_study.calibration.schema import CalibrationBundle, source_at, value_at
from vbb_study.experimental.bench_dataset import (
    ExperimentalDataset,
    ExperimentalFrameRecord,
    load_intensity_frame,
    sha256_file,
)
from vbb_study.reporting.evidence_conventions import (
    DEFAULT_FIGURE_DPI,
    INTENSITY_CMAP,
    SIGNED_CMAP,
    common_positive_peak,
)
from vbb_study.vector_field import VectorField


EPS = np.finfo(float).tiny
_UNCALIBRATED_SOURCES = {
    "",
    "missing",
    "unspecified",
    "assumed",
    "calibration_required",
    "template",
    "unknown",
}


@dataclass(frozen=True)
class ExperimentalClosureEvidence:
    dataset_id: str
    frame_id: str
    case_id: str
    z_m: float
    analyzer_nominal_deg: float | None
    comparison: CameraComparison
    measured_sha256: str
    background_frame_id: str | None
    metadata: Mapping[str, Any]


def _require_calibrated(bundle: CalibrationBundle, path: str) -> Any:
    value = value_at(bundle, path)
    source = str(source_at(bundle, path)).strip().lower()
    if value in (None, "") or source in _UNCALIBRATED_SOURCES:
        raise ValueError(f"Phase 2I requires calibrated {path}; source={source!r}")
    return value


def camera_calibration_from_bundle(bundle: CalibrationBundle) -> CameraCalibration:
    """Build the existing Phase 2G camera model without fitting geometry to data."""

    scale = float(_require_calibrated(bundle, "camera.object_plane_scale_m_per_pixel"))
    rotation_deg = float(_require_calibrated(bundle, "camera.rotation_deg"))
    centre = value_at(bundle, "camera.centre_pixel")
    if centre is None or not isinstance(centre, (list, tuple)) or len(centre) != 2:
        raise ValueError("Phase 2I requires camera.centre_pixel=[x,y] from calibration")
    cx, cy = map(float, centre)
    if not np.isfinite(cx) or not np.isfinite(cy):
        raise ValueError("camera.centre_pixel must contain finite values")
    saturation = value_at(bundle, "camera.saturation_level")
    calibration = CameraCalibration(
        object_plane_scale_m_per_pixel=scale,
        rotation_rad=math.radians(rotation_deg),
        centre_pixel_x=cx,
        centre_pixel_y=cy,
        saturation_level=None if saturation is None else float(saturation),
    )
    calibration.validate()
    return calibration


def _background_for_record(
    dataset: ExperimentalDataset,
    record: ExperimentalFrameRecord,
) -> np.ndarray | None:
    if record.background_frame_id is None:
        return None
    return load_intensity_frame(dataset, dataset.frame(record.background_frame_id))


def compare_record_to_simulation(
    dataset: ExperimentalDataset,
    record: ExperimentalFrameRecord,
    calibration_bundle: CalibrationBundle,
    *,
    simulated_intensity: np.ndarray,
    simulated_x_m: np.ndarray,
    simulated_y_m: np.ndarray,
) -> ExperimentalClosureEvidence:
    """Compare one hash-verified quantitative measured frame to one physical plane."""

    if not record.quantitative or record.role not in {"camera_intensity", "analyzer_intensity"}:
        raise ValueError("Phase 2I quantitative closure requires camera_intensity or analyzer_intensity")
    actual_hash = sha256_file(dataset.resolved_path(record))
    if actual_hash.lower() != record.sha256.lower():
        raise ValueError(f"{record.frame_id}: measured file SHA-256 does not match manifest")
    measured = load_intensity_frame(dataset, record)
    background = _background_for_record(dataset, record)
    if background is not None and background.shape != measured.shape:
        raise ValueError(f"{record.frame_id}: background frame shape does not match measurement")
    camera = camera_calibration_from_bundle(calibration_bundle)
    comparison = compare_simulation_to_camera(
        measured,
        camera,
        simulated_intensity,
        simulated_x_m,
        simulated_y_m,
        background=background,
    )
    return ExperimentalClosureEvidence(
        dataset_id=dataset.dataset_id,
        frame_id=record.frame_id,
        case_id=record.case_id,
        z_m=float(record.z_m),
        analyzer_nominal_deg=record.analyzer_nominal_deg,
        comparison=comparison,
        measured_sha256=actual_hash,
        background_frame_id=record.background_frame_id,
        metadata={
            "data_classification": dataset.data_classification,
            "camera_geometry_fitted_to_image": False,
            "comparison_normalisation": "energy_normalised_morphology_for_non-radiometric_camera_closure",
            "agreement_acceptance_applied": False,
            "agreement_acceptance_note": "Metrics are reported; no pass threshold is invented by Phase 2I.",
        },
    )


def compare_vector_field_record(
    dataset: ExperimentalDataset,
    record: ExperimentalFrameRecord,
    calibration_bundle: CalibrationBundle,
    simulated_field: VectorField,
) -> ExperimentalClosureEvidence:
    x = np.asarray(simulated_field.grid["x"], dtype=float)
    y = np.asarray(simulated_field.grid.get("y", simulated_field.grid["x"]), dtype=float)
    return compare_record_to_simulation(
        dataset,
        record,
        calibration_bundle,
        simulated_intensity=np.asarray(simulated_field.intensity, dtype=float),
        simulated_x_m=x,
        simulated_y_m=y,
    )


def _energy_normalised(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    arr = np.where(valid, np.maximum(np.asarray(values, dtype=float), 0.0), 0.0)
    return arr / max(float(np.sum(arr)), EPS)


def write_closure_evidence(
    evidence: ExperimentalClosureEvidence,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write one auditable measured/simulated morphology comparison package."""

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    comparison = evidence.comparison
    valid = np.asarray(comparison.valid_mask, dtype=bool)
    measured = _energy_normalised(comparison.measured_intensity, valid)
    simulated = _energy_normalised(comparison.simulated_on_camera, valid)
    residual = simulated - measured
    common_peak = common_positive_peak([measured, simulated])
    residual_scale = max(float(np.max(np.abs(residual[valid]))) if np.any(valid) else 0.0, EPS)

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2), constrained_layout=True)
    extent = [
        float(np.min(comparison.x_m)) * 1e3,
        float(np.max(comparison.x_m)) * 1e3,
        float(np.min(comparison.y_m)) * 1e3,
        float(np.max(comparison.y_m)) * 1e3,
    ]
    im0 = axes[0, 0].imshow(
        measured / common_peak,
        origin="lower",
        extent=extent,
        cmap=INTENSITY_CMAP,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    axes[0, 1].imshow(
        simulated / common_peak,
        origin="lower",
        extent=extent,
        cmap=INTENSITY_CMAP,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    imr = axes[0, 2].imshow(
        residual / residual_scale,
        origin="lower",
        extent=extent,
        cmap=SIGNED_CMAP,
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    axes[0, 0].set_title("measured")
    axes[0, 1].set_title("simulation on calibrated camera grid")
    axes[0, 2].set_title("signed morphology residual")
    for ax in axes[0, :]:
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
    fig.colorbar(im0, ax=axes[0, :2], shrink=0.78, label="energy-normalised I / common peak")
    fig.colorbar(imr, ax=axes[0, 2], shrink=0.78, label="residual / max |residual|")

    measured_total = max(float(np.sum(np.maximum(comparison.measured_intensity, 0.0))), EPS)
    simulated_total = max(float(np.sum(np.maximum(comparison.simulated_on_camera, 0.0))), EPS)
    mx = np.asarray(comparison.measured_x_profile, dtype=float) / measured_total
    sx = np.asarray(comparison.simulated_x_profile, dtype=float) / simulated_total
    my = np.asarray(comparison.measured_y_profile, dtype=float) / measured_total
    sy = np.asarray(comparison.simulated_y_profile, dtype=float) / simulated_total
    profile_nominal = max(float(np.max(mx)), float(np.max(my)), EPS)
    axes[1, 0].plot(np.asarray(comparison.x_m) * 1e3, mx / profile_nominal, label="measured")
    axes[1, 0].plot(np.asarray(comparison.x_m) * 1e3, sx / profile_nominal, label="simulation")
    axes[1, 0].set_title("fixed camera-origin x profile")
    axes[1, 0].set_xlabel("x (mm)")
    axes[1, 0].set_ylabel("I / measured nominal line scale")
    axes[1, 0].grid(alpha=0.20)
    axes[1, 0].legend()

    axes[1, 1].plot(np.asarray(comparison.y_m) * 1e3, my / profile_nominal, label="measured")
    axes[1, 1].plot(np.asarray(comparison.y_m) * 1e3, sy / profile_nominal, label="simulation")
    axes[1, 1].set_title("fixed camera-origin y profile")
    axes[1, 1].set_xlabel("y (mm)")
    axes[1, 1].set_ylabel("I / measured nominal line scale")
    axes[1, 1].grid(alpha=0.20)

    metric_lines = [
        f"corr = {comparison.metrics['energy_normalised_correlation']:.6f}",
        f"L2 = {comparison.metrics['energy_normalised_l2']:.6f}",
        f"centroid error = {comparison.metrics['centroid_error_m'] * 1e6:.3f} µm",
        f"covariance rel. error = {comparison.metrics['covariance_relative_frobenius_error']:.6f}",
        f"valid overlap = {comparison.metrics['valid_overlap_fraction']:.4f}",
        "acceptance threshold: NOT DEFINED",
    ]
    axes[1, 2].axis("off")
    axes[1, 2].text(0.02, 0.96, "\n".join(metric_lines), va="top", ha="left", family="monospace")

    analyzer = "none" if evidence.analyzer_nominal_deg is None else f"{evidence.analyzer_nominal_deg:g} deg nominal"
    fig.suptitle(
        f"Phase 2I measured-vs-sim closure: {evidence.case_id} / {evidence.frame_id}\n"
        f"z={evidence.z_m * 1e3:.3f} mm; analyzer={analyzer}; no fitted camera geometry",
        fontsize=13,
    )
    png = outdir / f"{evidence.frame_id}__camera_closure.png"
    fig.savefig(png, dpi=DEFAULT_FIGURE_DPI)
    plt.close(fig)

    npz = outdir / f"{evidence.frame_id}__camera_closure.npz"
    np.savez_compressed(
        npz,
        measured_intensity=np.asarray(comparison.measured_intensity, dtype=float),
        simulated_on_camera=np.asarray(comparison.simulated_on_camera, dtype=float),
        valid_mask=valid,
        x_m=np.asarray(comparison.x_m, dtype=float),
        y_m=np.asarray(comparison.y_m, dtype=float),
        measured_x_profile=np.asarray(comparison.measured_x_profile, dtype=float),
        measured_y_profile=np.asarray(comparison.measured_y_profile, dtype=float),
        simulated_x_profile=np.asarray(comparison.simulated_x_profile, dtype=float),
        simulated_y_profile=np.asarray(comparison.simulated_y_profile, dtype=float),
    )
    payload = {
        "outcome": "PHASE2I-CAMERA-CLOSURE-EVIDENCE",
        "dataset_id": evidence.dataset_id,
        "data_classification": evidence.metadata.get("data_classification"),
        "frame_id": evidence.frame_id,
        "case_id": evidence.case_id,
        "z_m": evidence.z_m,
        "analyzer_nominal_deg": evidence.analyzer_nominal_deg,
        "measured_sha256": evidence.measured_sha256,
        "background_frame_id": evidence.background_frame_id,
        "camera_geometry_fitted_to_image": False,
        "comparison_normalisation": evidence.metadata.get("comparison_normalisation"),
        "metrics": {k: float(v) for k, v in comparison.metrics.items()},
        "agreement_acceptance_applied": False,
        "agreement_acceptance_passed": None,
        "files": {"figure": png.name, "arrays": npz.name},
    }
    json_path = outdir / f"{evidence.frame_id}__camera_closure.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "ExperimentalClosureEvidence",
    "camera_calibration_from_bundle",
    "compare_record_to_simulation",
    "compare_vector_field_record",
    "write_closure_evidence",
]
