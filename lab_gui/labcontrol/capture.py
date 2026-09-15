"""Formal, provenance-complete quantitative capture service."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .controller import LabController
from .metrics import BeamMetrics, MetricEngine
from .phase_service import PhaseBundle, phase_sha256
from .state import AcquisitionState, ExperimentState, utc_now


class CaptureError(RuntimeError):
    pass


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class MeasurementRecord:
    session_id: str
    trial_id: str
    trial_root: Path
    state_revision: int
    phase_hashes: dict[str, str]
    frame_ids: tuple[str, ...]
    metrics: tuple[BeamMetrics, ...]
    data_kind: str


class SessionRepository:
    """On-disk session with immutable trials and content-addressed phases."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.objects = self.root / "objects" / "phase"
        self.trials = self.root / "trials"

    def initialise(self, state: ExperimentState) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        self.objects.mkdir(parents=True, exist_ok=True)
        self.trials.mkdir(parents=True, exist_ok=True)
        manifest_path = self.root / "session.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            return str(existing["session_id"])
        session_id = state.session.session_id or f"session-{utc_now().replace(':', '').replace('+', '_')}-{uuid.uuid4().hex[:8]}"
        _write_json(
            manifest_path,
            {
                "schema_version": 1,
                "session_id": session_id,
                "created_utc": utc_now(),
                "data_kind": state.system.data_kind.value,
                "experiment_label": state.system.experiment_label,
                "physical_configuration": state.system.physical_configuration,
                "trials": [],
            },
        )
        return session_id

    def store_phase(self, phase: np.ndarray, expected_hash: str) -> Path:
        actual_hash = phase_sha256(phase)
        if actual_hash != expected_hash:
            raise CaptureError("Generated phase changed before provenance storage.")
        destination = self.objects / f"{expected_hash}.npy"
        if not destination.exists():
            temporary = destination.with_suffix(".npy.tmp")
            with temporary.open("wb") as handle:
                wrapped = np.ascontiguousarray(np.mod(phase, 2.0 * np.pi), dtype="<f4")
                wrapped[wrapped >= np.float32(2.0 * np.pi)] = 0.0
                np.save(handle, wrapped, allow_pickle=False)
            os.replace(temporary, destination)
        return destination

    def reserve_trial(self, trial_id: str) -> Path:
        if not trial_id or Path(trial_id).name != trial_id or trial_id in {".", ".."}:
            raise CaptureError("Trial id must be a non-empty filename-safe value.")
        destination = self.trials / trial_id
        if destination.exists():
            raise CaptureError(f"Trial {trial_id!r} already exists; formal data are never overwritten.")
        destination.mkdir(parents=True)
        return destination

    def register_trial(self, trial_id: str, manifest_sha256: str) -> None:
        path = self.root / "session.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["trials"].append(
            {
                "trial_id": trial_id,
                "manifest": f"trials/{trial_id}/measurement.json",
                "manifest_sha256": manifest_sha256,
            }
        )
        _write_json(path, manifest)


class FormalCaptureService:
    """Capture fresh frames only after exact complete-phase cast is known."""

    def __init__(
        self,
        controller: LabController,
        repository: SessionRepository,
        *,
        metric_engine: MetricEngine | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.controller = controller
        self.repository = repository
        self.metric_engine = metric_engine or MetricEngine()
        self.sleeper = sleeper

    def _ensure_cast(self, names: tuple[str, ...], *, auto_cast: bool) -> PhaseBundle:
        bundle = self.controller.generate()
        state = self.controller.store.snapshot()
        stale = [
            name
            for name in names
            if getattr(state, name.lower()).last_cast_sha256 != bundle.hashes[name]
        ]
        if stale:
            if not auto_cast:
                raise CaptureError(
                    "Formal capture refused: configured phase is not the last successful cast on "
                    + ", ".join(stale)
                )
            self.controller.cast(stale)
            bundle = self.controller.last_bundle or bundle
        verified = self.controller.store.snapshot()
        for name in names:
            if getattr(verified, name.lower()).last_cast_sha256 != bundle.hashes[name]:
                raise CaptureError(f"Formal capture refused: {name} cast hash could not be verified.")
        return bundle

    def capture(
        self,
        trial_id: str,
        *,
        repeats: int = 1,
        recipe: str | None = None,
        role: str = "CURRENT",
        settle_s: float = 0.0,
        required_slms: Iterable[str] = ("SLM1", "SLM2"),
        auto_cast: bool = True,
        calibration_ids: Iterable[str] = (),
    ) -> MeasurementRecord:
        if repeats < 1:
            raise ValueError("Formal capture requires at least one fresh frame.")
        if settle_s < 0:
            raise ValueError("Settling time cannot be negative.")
        names = tuple(dict.fromkeys(name.upper() for name in required_slms))
        bundle = self._ensure_cast(names, auto_cast=auto_cast)
        state = self.controller.store.snapshot()
        session_id = self.repository.initialise(state)
        trial_root = self.repository.reserve_trial(trial_id)
        camera_root = trial_root / "camera"
        camera_root.mkdir()
        phase_objects = {
            name: self.repository.store_phase(bundle.results[name].phase_rad, bundle.hashes[name])
            for name in names
        }
        _write_json(trial_root / "state.json", state.to_dict())
        _write_json(
            trial_root / "phase_hashes.json",
            {
                name: {
                    "sha256": bundle.hashes[name],
                    "object": str(path.relative_to(self.repository.root)),
                    "units": "radians",
                    "wrapped_interval": "[0, 2pi)",
                }
                for name, path in phase_objects.items()
            },
        )

        self.controller.store.update(
            lambda current: (
                setattr(current.camera, "acquisition", AcquisitionState.CAPTURING),
                setattr(current.session, "session_id", session_id),
                setattr(current.session, "session_root", str(self.repository.root)),
                setattr(current.session, "recipe", recipe),
                setattr(current.session, "current_trial", trial_id),
            ),
            source="formal_capture",
            reason=f"Started formal capture {trial_id}",
        )
        if settle_s:
            self.sleeper(float(settle_s))
        frames = []
        metrics = []
        try:
            for index in range(1, repeats + 1):
                acquired = self.controller.acquire_frame(fresh=True)
                raw_path = camera_root / f"frame_{index:03d}.npy"
                np.save(raw_path, acquired.data, allow_pickle=False)
                analysed = self.metric_engine.analyse(acquired, state)
                frames.append(
                    {
                        "index": index,
                        "frame_id": acquired.frame_id,
                        "timestamp_utc": acquired.timestamp_utc,
                        "raw_path": str(raw_path.relative_to(self.repository.root)),
                        "raw_file_sha256": _file_sha256(raw_path),
                        "raw_pixels_sha256": acquired.pixel_sha256,
                        "shape_yx": list(acquired.shape_yx),
                        "provider": acquired.provider,
                        "data_kind": acquired.data_kind,
                        "exposure_us": acquired.exposure_us,
                        "gain": acquired.gain,
                        "full_scale": acquired.full_scale,
                        "z_mm": acquired.z_mm,
                        "metadata": acquired.metadata,
                    }
                )
                metrics.append(analysed)
        except Exception:
            self.controller.store.update(
                lambda current: setattr(current.camera, "acquisition", AcquisitionState.ERROR),
                source="formal_capture",
                reason=f"Formal capture {trial_id} failed",
            )
            raise
        finally:
            # A formal capture does not implicitly restart or stop a provider;
            # it returns the state to LIVE if the camera worker owns a stream.
            provider_running = bool(getattr(self.controller.camera_provider, "_running", False))
            self.controller.store.update(
                lambda current: setattr(
                    current.camera,
                    "acquisition",
                    AcquisitionState.LIVE if provider_running else AcquisitionState.STOPPED,
                ),
                source="formal_capture",
                reason=f"Formal capture {trial_id} finished",
            )

        kinds = {str(item["data_kind"]) for item in frames}
        data_kind = kinds.pop() if len(kinds) == 1 else "MIXED"
        metrics_payload = {
            "analysis_version": self.metric_engine.analysis_version,
            "per_frame": [item.to_dict() for item in metrics],
            "repeat_stability": self.metric_engine.repeat_stability(metrics),
        }
        _write_json(trial_root / "metrics.json", metrics_payload)
        measurement = {
            "schema_version": 1,
            "session_id": session_id,
            "trial_id": trial_id,
            "role": role,
            "recipe": recipe,
            "created_utc": utc_now(),
            "data_kind": data_kind,
            "state_revision": state.revision,
            "state_file": "state.json",
            "phase_hashes": dict(bundle.hashes),
            "required_slms": list(names),
            "slm_serials": {"SLM1": state.slm1.phase.serial, "SLM2": state.slm2.phase.serial},
            "wavelength_nm": state.system.wavelength_nm,
            "physical_configuration": state.system.physical_configuration,
            "camera": {
                "provider": state.camera.provider,
                "implementation_status": state.camera.implementation_status,
                "settings_id": state.camera.settings_id,
                "exposure_us": state.camera.exposure_us,
                "gain": state.camera.gain,
                "pixel_size_um": state.camera.pixel_size_um,
                "z_mm": state.camera.current_z_mm,
                "z_reference": state.camera.z_reference,
            },
            "repeats": repeats,
            "settle_s": settle_s,
            "calibration_ids": list(calibration_ids),
            "analysis_version": self.metric_engine.analysis_version,
            "frames": frames,
            "metrics_file": "metrics.json",
        }
        measurement_path = trial_root / "measurement.json"
        _write_json(measurement_path, measurement)
        self.repository.register_trial(trial_id, _file_sha256(measurement_path))
        return MeasurementRecord(
            session_id=session_id,
            trial_id=trial_id,
            trial_root=trial_root,
            state_revision=state.revision,
            phase_hashes=dict(bundle.hashes),
            frame_ids=tuple(item["frame_id"] for item in frames),
            metrics=tuple(metrics),
            data_kind=data_kind,
        )
