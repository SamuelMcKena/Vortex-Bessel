"""Versioned measured-bench dataset ingestion with provenance and readiness gates.

The dataset layer deliberately separates *files that exist* from *evidence that is
ready for a physical claim*.  Quantitative comparison frames must be lossless
numeric/radiometric arrays, carry SHA-256 hashes, and remain bound to calibrated
physical coordinates.  Screenshots may be retained as qualitative provenance but
cannot silently enter quantitative camera comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

from vbb_study.calibration.schema import CalibrationBundle, source_at, value_at


DATASET_SCHEMA_VERSION = "1.0"
DATA_CLASSIFICATIONS = {
    "laboratory_measurement",
    "synthetic_not_experimental",
    "template_unpopulated",
}
FRAME_ROLES = {
    "camera_intensity",
    "camera_background",
    "analyzer_intensity",
    "qualitative_reference",
}
QUANTITATIVE_ROLES = {"camera_intensity", "camera_background", "analyzer_intensity"}
QUANTITATIVE_EXTENSIONS = {".npy", ".npz", ".csv", ".txt", ".tif", ".tiff"}
QUALITATIVE_EXTENSIONS = QUANTITATIVE_EXTENSIONS | {".png", ".jpg", ".jpeg"}
_HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")
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
class ExperimentalFrameRecord:
    frame_id: str
    case_id: str
    role: str
    relative_path: str
    sha256: str
    z_m: float | None = None
    analyzer_nominal_deg: float | None = None
    array_key: str | None = None
    quantitative: bool = True
    exposure_s: float | None = None
    background_frame_id: str | None = None
    notes: str = ""

    def validate(self) -> None:
        if not self.frame_id.strip():
            raise ValueError("frame_id must be non-empty")
        if not self.case_id.strip():
            raise ValueError("case_id must be non-empty")
        if self.role not in FRAME_ROLES:
            raise ValueError(f"unsupported frame role {self.role!r}")
        if not self.relative_path.strip():
            raise ValueError(f"{self.frame_id}: relative_path must be non-empty")
        if not _HASH_RE.fullmatch(self.sha256):
            raise ValueError(f"{self.frame_id}: sha256 must contain exactly 64 hex characters")
        if self.quantitative and self.role not in QUANTITATIVE_ROLES:
            raise ValueError(f"{self.frame_id}: qualitative_reference cannot be quantitative")
        if self.role in {"camera_intensity", "analyzer_intensity"}:
            if self.z_m is None or not np.isfinite(float(self.z_m)):
                raise ValueError(f"{self.frame_id}: quantitative observation requires finite z_m")
        if self.role == "analyzer_intensity":
            if self.analyzer_nominal_deg is None or not np.isfinite(float(self.analyzer_nominal_deg)):
                raise ValueError(f"{self.frame_id}: analyzer_intensity requires analyzer_nominal_deg")
        if self.exposure_s is not None and (not np.isfinite(float(self.exposure_s)) or float(self.exposure_s) <= 0.0):
            raise ValueError(f"{self.frame_id}: exposure_s must be positive when supplied")


@dataclass(frozen=True)
class ExperimentalDataset:
    root: Path
    manifest_path: Path
    dataset_id: str
    schema_version: str
    data_classification: str
    canonical_z_ref_m: float | None
    frames: tuple[ExperimentalFrameRecord, ...]
    metadata: Mapping[str, Any]

    @property
    def is_laboratory(self) -> bool:
        return self.data_classification == "laboratory_measurement"

    def frame(self, frame_id: str) -> ExperimentalFrameRecord:
        matches = [f for f in self.frames if f.frame_id == frame_id]
        if len(matches) != 1:
            raise KeyError(frame_id)
        return matches[0]

    def frames_for_case(self, case_id: str) -> tuple[ExperimentalFrameRecord, ...]:
        return tuple(f for f in self.frames if f.case_id == case_id)

    def frames_by_role(self, role: str) -> tuple[ExperimentalFrameRecord, ...]:
        return tuple(f for f in self.frames if f.role == role)

    def resolved_path(self, record: ExperimentalFrameRecord) -> Path:
        return _safe_resolve(self.root, record.relative_path)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(int(chunk_size)), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_resolve(root: Path, relative_path: str) -> Path:
    root_resolved = root.resolve()
    candidate = (root_resolved / relative_path).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"dataset path escapes dataset root: {relative_path!r}") from exc
    return candidate


def _record_from_mapping(row: Mapping[str, Any]) -> ExperimentalFrameRecord:
    record = ExperimentalFrameRecord(
        frame_id=str(row.get("frame_id", "")),
        case_id=str(row.get("case_id", "")),
        role=str(row.get("role", "")),
        relative_path=str(row.get("path", row.get("relative_path", ""))),
        sha256=str(row.get("sha256", "")),
        z_m=None if row.get("z_m") is None else float(row["z_m"]),
        analyzer_nominal_deg=(
            None if row.get("analyzer_nominal_deg") is None else float(row["analyzer_nominal_deg"])
        ),
        array_key=None if row.get("array_key") in (None, "") else str(row.get("array_key")),
        quantitative=bool(row.get("quantitative", True)),
        exposure_s=None if row.get("exposure_s") is None else float(row["exposure_s"]),
        background_frame_id=(
            None if row.get("background_frame_id") in (None, "") else str(row.get("background_frame_id"))
        ),
        notes=str(row.get("notes", "")),
    )
    record.validate()
    return record


def load_experimental_dataset(
    manifest_path: str | Path,
    *,
    require_files: bool = True,
) -> ExperimentalDataset:
    path = Path(manifest_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("experimental dataset manifest must be a JSON object")
    version = str(payload.get("schema_version", ""))
    if version != DATASET_SCHEMA_VERSION:
        raise ValueError(f"unsupported experimental dataset schema {version!r}; expected {DATASET_SCHEMA_VERSION}")
    dataset_id = str(payload.get("dataset_id", "")).strip()
    if not dataset_id:
        raise ValueError("dataset_id must be non-empty")
    classification = str(payload.get("data_classification", "template_unpopulated"))
    if classification not in DATA_CLASSIFICATIONS:
        raise ValueError(f"invalid data_classification {classification!r}")
    z_ref = payload.get("canonical_z_ref_m")
    z_ref_m = None if z_ref is None else float(z_ref)
    if z_ref_m is not None and (not np.isfinite(z_ref_m) or z_ref_m < 0.0):
        raise ValueError("canonical_z_ref_m must be finite and non-negative when supplied")
    raw_frames = payload.get("frames", [])
    if not isinstance(raw_frames, list):
        raise ValueError("frames must be a list")
    frames = tuple(_record_from_mapping(row) for row in raw_frames)
    ids = [f.frame_id for f in frames]
    if len(ids) != len(set(ids)):
        raise ValueError("frame_id values must be unique")

    dataset = ExperimentalDataset(
        root=path.parent,
        manifest_path=path,
        dataset_id=dataset_id,
        schema_version=version,
        data_classification=classification,
        canonical_z_ref_m=z_ref_m,
        frames=frames,
        metadata={k: v for k, v in payload.items() if k != "frames"},
    )

    known = set(ids)
    for frame in frames:
        resolved = dataset.resolved_path(frame)
        extension = resolved.suffix.lower()
        allowed = QUANTITATIVE_EXTENSIONS if frame.quantitative else QUALITATIVE_EXTENSIONS
        if extension not in allowed:
            category = "quantitative" if frame.quantitative else "qualitative"
            raise ValueError(f"{frame.frame_id}: {extension!r} is not an allowed {category} frame format")
        if require_files and not resolved.is_file():
            raise FileNotFoundError(resolved)
        if frame.background_frame_id is not None and frame.background_frame_id not in known:
            raise ValueError(f"{frame.frame_id}: unknown background_frame_id {frame.background_frame_id!r}")
        if frame.background_frame_id is not None:
            background = dataset.frame(frame.background_frame_id)
            if background.role != "camera_background":
                raise ValueError(f"{frame.frame_id}: background_frame_id must point to camera_background")
    return dataset


def verify_dataset_hashes(dataset: ExperimentalDataset) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    all_match = True
    for record in dataset.frames:
        path = dataset.resolved_path(record)
        actual = sha256_file(path)
        match = actual.lower() == record.sha256.lower()
        all_match &= match
        rows.append(
            {
                "frame_id": record.frame_id,
                "path": record.relative_path,
                "expected_sha256": record.sha256.lower(),
                "actual_sha256": actual,
                "match": bool(match),
                "bytes": int(path.stat().st_size),
            }
        )
    return {"all_match": bool(all_match), "frames": rows}


def load_intensity_frame(dataset: ExperimentalDataset, record: ExperimentalFrameRecord) -> np.ndarray:
    path = dataset.resolved_path(record)
    extension = path.suffix.lower()
    if record.quantitative and extension not in QUANTITATIVE_EXTENSIONS:
        raise ValueError(f"{record.frame_id}: non-quantitative image format cannot enter numerical comparison")
    if extension == ".npy":
        values = np.load(path, allow_pickle=False)
    elif extension == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            keys = list(archive.files)
            key = record.array_key
            if key is None:
                if len(keys) != 1:
                    raise ValueError(f"{record.frame_id}: npz contains multiple arrays; array_key is required")
                key = keys[0]
            if key not in archive:
                raise ValueError(f"{record.frame_id}: array_key {key!r} not present in npz")
            values = archive[key]
    elif extension == ".csv":
        values = np.loadtxt(path, delimiter=",")
    elif extension == ".txt":
        values = np.loadtxt(path)
    elif extension in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
        with Image.open(path) as image:
            values = np.asarray(image)
    else:
        raise ValueError(f"unsupported frame extension {extension!r}")
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"{record.frame_id}: intensity frame must be a non-empty 2-D array")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{record.frame_id}: intensity frame contains non-finite values")
    if np.any(arr < 0.0):
        raise ValueError(f"{record.frame_id}: raw quantitative intensity may not contain negative values")
    return arr


def _calibrated(bundle: CalibrationBundle | None, path: str) -> bool:
    if bundle is None:
        return False
    value = value_at(bundle, path)
    source = str(source_at(bundle, path)).strip().lower()
    return value not in (None, "") and source not in _UNCALIBRATED_SOURCES


def _gate(complete: bool, missing: Iterable[str] = ()) -> dict[str, Any]:
    missing_list = [str(item) for item in missing]
    return {"complete": bool(complete), "missing": [] if complete else missing_list}


def dataset_readiness(
    dataset: ExperimentalDataset,
    *,
    calibration_bundle: CalibrationBundle | None = None,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    quantitative = tuple(f for f in dataset.frames if f.quantitative and f.role in {"camera_intensity", "analyzer_intensity"})
    backgrounds = tuple(f for f in dataset.frames if f.quantitative and f.role == "camera_background")
    z_values = sorted({float(f.z_m) for f in quantitative if f.z_m is not None})
    analyzer_angles = sorted({float(f.analyzer_nominal_deg) for f in quantitative if f.role == "analyzer_intensity" and f.analyzer_nominal_deg is not None})
    hash_report = verify_dataset_hashes(dataset) if verify_hashes else {"all_match": None, "frames": []}

    zref_frames: tuple[ExperimentalFrameRecord, ...] = ()
    if dataset.canonical_z_ref_m is not None:
        zref_frames = tuple(
            f for f in quantitative
            if f.z_m is not None and math.isclose(float(f.z_m), float(dataset.canonical_z_ref_m), abs_tol=1e-9)
        )

    camera_missing = [
        path
        for path in ("camera.object_plane_scale_m_per_pixel", "camera.rotation_deg")
        if not _calibrated(calibration_bundle, path)
    ]
    if calibration_bundle is not None and value_at(calibration_bundle, "camera.centre_pixel") is None:
        camera_missing.append("camera.centre_pixel")

    analyzer_required = any(f.role == "analyzer_intensity" for f in quantitative)
    analyzer_missing: list[str] = []
    if analyzer_required:
        for angle in (0, 45, 90, 135):
            if float(angle) not in analyzer_angles:
                analyzer_missing.append(f"measured analyzer frame nominal {angle} deg")
            if not _calibrated(calibration_bundle, f"polarization.analyzer_{angle}_actual_deg"):
                analyzer_missing.append(f"polarization.analyzer_{angle}_actual_deg")
        for path in ("polarization.analyzer_extinction_ratio", "polarization.analyzer_transmission"):
            if not _calibrated(calibration_bundle, path):
                analyzer_missing.append(path)

    gates = {
        "laboratory_classification": _gate(dataset.is_laboratory, ["data_classification=laboratory_measurement"]),
        "quantitative_camera_evidence": _gate(bool(quantitative), ["at least one quantitative camera/analyzer frame"]),
        "file_integrity": _gate(hash_report.get("all_match") is True, ["all frame SHA-256 hashes must match"]),
        "camera_physical_coordinates": _gate(not camera_missing, camera_missing),
        "background_evidence": _gate(bool(backgrounds), ["quantitative camera_background frame"]),
        "canonical_zref_frame": _gate(
            dataset.canonical_z_ref_m is not None and bool(zref_frames),
            ["canonical_z_ref_m and at least one quantitative frame at that exact physical z"],
        ),
        "measured_longitudinal_stack": _gate(
            len(z_values) >= 2,
            ["at least two distinct calibrated z positions; denser sampling is required for useful longitudinal evidence"],
        ),
        "vector_analyzer_stack": _gate((not analyzer_required) or (not analyzer_missing), analyzer_missing),
    }
    absolute_ready = all(
        gates[name]["complete"]
        for name in (
            "laboratory_classification",
            "quantitative_camera_evidence",
            "file_integrity",
            "camera_physical_coordinates",
            "background_evidence",
            "canonical_zref_frame",
        )
    )
    return {
        "outcome": (
            "PHASE2I-EXPERIMENTAL-DATASET-READY-FOR-CALIBRATED-COMPARISON"
            if absolute_ready
            else "PHASE2I-EXPERIMENTAL-DATASET-INCOMPLETE"
        ),
        "dataset_id": dataset.dataset_id,
        "schema_version": dataset.schema_version,
        "data_classification": dataset.data_classification,
        "canonical_z_ref_m": dataset.canonical_z_ref_m,
        "quantitative_frame_count": len(quantitative),
        "background_frame_count": len(backgrounds),
        "distinct_z_positions_m": z_values,
        "analyzer_nominal_angles_deg": analyzer_angles,
        "hash_report": hash_report,
        "gates": gates,
        "absolute_calibrated_comparison_ready": bool(absolute_ready),
        "agreement_acceptance_defined": False,
        "agreement_acceptance_note": "No numerical sim-to-experiment pass thresholds are invented by the ingestion layer.",
    }


__all__ = [
    "DATASET_SCHEMA_VERSION",
    "ExperimentalDataset",
    "ExperimentalFrameRecord",
    "dataset_readiness",
    "load_experimental_dataset",
    "load_intensity_frame",
    "sha256_file",
    "verify_dataset_hashes",
]
