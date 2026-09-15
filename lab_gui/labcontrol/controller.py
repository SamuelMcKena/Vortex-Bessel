"""Headless application service shared by the compact and advanced clients."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from slm_lab_control.config import AppConfig
from slm_lab_control.logging_utils import create_cast_folder, save_cast_bundle

from .devices.camera import CameraFrame, CameraProvider
from .devices.slm import BackendSlmProvider, SlmProvider, slm_provider_from_config
from .phase_service import PhaseBundle, PhaseService, phase_sha256
from .state import (
    AcquisitionState,
    ConnectionState,
    DataKind,
    ExperimentState,
    ExperimentStore,
    utc_now,
)


@dataclass(frozen=True)
class CastReceipt:
    folder: Path
    phase_hashes: dict[str, str]
    messages: tuple[str, ...]
    transfer_mode: str


class LabController:
    """Own providers and coordinate safe mutations of ExperimentStore.

    This is deliberately Qt-free.  A later broker can expose the same methods
    over IPC while keeping this single-owner hardware boundary.
    """

    def __init__(
        self,
        store: ExperimentStore,
        project_root: str | Path,
        *,
        phase_service: PhaseService | None = None,
        slm_provider: SlmProvider | None = None,
        camera_provider: CameraProvider | None = None,
    ):
        self.store = store
        self.project_root = Path(project_root)
        self.phase_service = phase_service or PhaseService()
        self.slm_provider = slm_provider
        self.camera_provider = camera_provider
        self.last_bundle: PhaseBundle | None = None
        self._provider_signature: tuple[object, ...] | None = None
        self._lock = threading.RLock()
        if camera_provider is not None:
            self._publish_camera_selection(camera_provider, camera_provider.name)

    @property
    def backend(self):
        """Compatibility view for the v0.6 measurement workbench."""

        provider = self.slm_provider
        return provider.backend if isinstance(provider, BackendSlmProvider) else None

    def app_config(self) -> AppConfig:
        return self.store.snapshot().to_app_config()

    def replace_app_config(
        self, config: AppConfig, *, source: str, reason: str = "SLM controls changed"
    ) -> None:
        self.store.replace_app_config(config, source=source, reason=reason)

    def generate(self) -> PhaseBundle:
        with self._lock:
            bundle = self.phase_service.generate(self.store.snapshot())
            self.last_bundle = bundle

            def record(state: ExperimentState) -> None:
                state.slm1.complete_phase_sha256 = bundle.hashes["SLM1"]
                state.slm2.complete_phase_sha256 = bundle.hashes["SLM2"]

            self.store.update(
                record,
                source="phase_engine",
                reason=f"Generated complete phases from revision {bundle.source_revision}",
            )
            return bundle

    def _desired_provider_signature(self, config: AppConfig) -> tuple[object, ...]:
        return (config.backend, config.sdk_major, config.sdk_minor, str(config.resolve_output_root(self.project_root)))

    def _ensure_slm_provider(self) -> SlmProvider:
        config = self.app_config()
        desired = self._desired_provider_signature(config)
        if self.slm_provider is None or (
            isinstance(self.slm_provider, BackendSlmProvider)
            and self._provider_signature is not None
            and self._provider_signature != desired
        ):
            if self.slm_provider is not None:
                self.slm_provider.close()
            self.slm_provider = slm_provider_from_config(config, self.project_root)
            self._provider_signature = desired
        elif isinstance(self.slm_provider, BackendSlmProvider) and self._provider_signature is None:
            self._provider_signature = desired
        return self.slm_provider

    def connect_slms(self) -> tuple[str, ...]:
        with self._lock:
            config = self.app_config()
            provider = self._ensure_slm_provider()
            self.store.update(
                lambda state: (
                    setattr(state.slm1, "connection", ConnectionState.CONNECTING),
                    setattr(state.slm2, "connection", ConnectionState.CONNECTING),
                ),
                source="slm_provider",
                reason="Connecting both SLMs",
            )
            try:
                messages = tuple(provider.connect(config))

                def connected(state: ExperimentState) -> None:
                    for name in ("SLM1", "SLM2"):
                        status = provider.status(name)
                        target = getattr(state, name.lower())
                        target.connection = status.connection
                        backend_state = getattr(provider, "backend", None)
                        device = getattr(backend_state, "devices", {}).get(name) if backend_state else None
                        target.phase_mode_verified = bool(device and device.phase_mode_verified)
                        target.last_hardware_command = status.message

                self.store.update(
                    connected,
                    source="slm_provider",
                    reason="Both SLM connection attempts completed",
                )
                return messages
            except Exception as exc:
                self.store.update(
                    lambda state: (
                        setattr(state.slm1, "connection", ConnectionState.ERROR),
                        setattr(state.slm2, "connection", ConnectionState.ERROR),
                        setattr(state.slm1, "last_hardware_command", str(exc)),
                        setattr(state.slm2, "last_hardware_command", str(exc)),
                    ),
                    source="slm_provider",
                    reason="SLM connection failed",
                )
                raise

    def cast(self, names: Iterable[str]) -> CastReceipt:
        selected_names = tuple(dict.fromkeys(str(name).upper() for name in names))
        if not selected_names or any(name not in {"SLM1", "SLM2"} for name in selected_names):
            raise ValueError("Select SLM1 and/or SLM2 for casting.")
        with self._lock:
            bundle = self.generate()
            provider = self._ensure_slm_provider()
            if any(provider.status(name).connection != ConnectionState.CONNECTED for name in selected_names):
                self.connect_slms()
            config = self.app_config()
            folder = create_cast_folder(config.resolve_output_root(self.project_root), "slm_cast")
            selected = {name: bundle.results[name] for name in selected_names}
            written = save_cast_bundle(folder, config, selected)
            messages = []
            for name in selected_names:
                messages.append(
                    provider.cast(name, bundle.results[name], config.transfer_mode, Path(written[name]))
                )
            stamp = utc_now()

            def record(state: ExperimentState) -> None:
                for name, message in zip(selected_names, messages):
                    target = getattr(state, name.lower())
                    target.last_cast_sha256 = bundle.hashes[name]
                    target.last_cast_utc = stamp
                    target.last_hardware_command = message
                    status = provider.status(name)
                    target.connection = status.connection
                    backend_state = getattr(provider, "backend", None)
                    device = getattr(backend_state, "devices", {}).get(name) if backend_state else None
                    target.phase_mode_verified = bool(device and device.phase_mode_verified)

            self.store.update(
                record,
                source="slm_provider",
                reason=f"Cast complete phase to {', '.join(selected_names)}",
            )
            return CastReceipt(folder, dict(bundle.hashes), tuple(messages), config.transfer_mode)

    def blank(self, names: Iterable[str] = ("SLM1", "SLM2")) -> tuple[str, ...]:
        selected_names = tuple(dict.fromkeys(str(name).upper() for name in names))
        if not selected_names or any(name not in {"SLM1", "SLM2"} for name in selected_names):
            raise ValueError("Select SLM1 and/or SLM2 for blanking.")
        with self._lock:
            config = self.app_config()
            provider = self._ensure_slm_provider()
            if any(provider.status(name).connection != ConnectionState.CONNECTED for name in selected_names):
                self.connect_slms()
            folder = create_cast_folder(config.resolve_output_root(self.project_root), "blank")
            messages = tuple(provider.blank(config, selected_names, folder))
            stamp = utc_now()

            def record(state: ExperimentState) -> None:
                for name, message in zip(selected_names, messages):
                    target = getattr(state, name.lower())
                    shape = (
                        target.phase.geometry.height_px,
                        target.phase.geometry.width_px,
                    )
                    target.last_cast_sha256 = phase_sha256(np.zeros(shape, dtype=np.float64))
                    target.last_cast_utc = stamp
                    target.last_hardware_command = message

            self.store.update(
                record,
                source="slm_provider",
                reason=f"Blanked {', '.join(selected_names)}",
            )
            return messages

    def close_slms(self) -> str:
        with self._lock:
            if self.slm_provider is None:
                message = "SLM provider already closed."
            else:
                message = self.slm_provider.close()
            self.store.update(
                lambda state: (
                    setattr(state.slm1, "connection", ConnectionState.DISCONNECTED),
                    setattr(state.slm2, "connection", ConnectionState.DISCONNECTED),
                ),
                source="slm_provider",
                reason="Closed SLM provider",
            )
            return message

    def _publish_camera_selection(self, provider: CameraProvider, provider_name: str) -> None:
        declared = DataKind(getattr(provider, "declared_data_kind", "EXPERIMENT"))
        configurable = bool(getattr(provider, "supports_configuration", True))
        self.store.update(
            lambda state: (
                setattr(state.camera, "provider", provider_name),
                setattr(state.camera, "implementation_status", provider.implementation_status),
                setattr(state.camera, "connection", ConnectionState.DISCONNECTED),
                setattr(state.camera, "acquisition", AcquisitionState.STOPPED),
                setattr(state.camera, "device_id", None),
                setattr(state.camera, "exposure_control", "SUPPORTED" if configurable else "CONTROLLED_IN_PC_BEAMAGE"),
                setattr(state.camera, "gain_control", "SUPPORTED" if configurable else "CONTROLLED_IN_PC_BEAMAGE"),
                setattr(state.camera, "frame_quality", "QUANTITATIVE" if provider_name != "beamage" else "LIVE_PREVIEW_UNVERIFIED"),
                setattr(state.system, "data_kind", declared),
            ),
            source="camera_provider",
            reason=f"Selected {provider_name} camera provider",
        )

    def set_camera_provider(self, provider: CameraProvider, *, provider_name: str | None = None) -> None:
        with self._lock:
            if self.camera_provider is not None and self.camera_provider.connected:
                self.camera_provider.disconnect()
            self.camera_provider = provider
            self._publish_camera_selection(provider, provider_name or provider.name)

    def connect_camera(self) -> str:
        if self.camera_provider is None:
            raise RuntimeError("No camera provider has been selected.")
        try:
            status = self.camera_provider.connect()
            snapshot = self.store.snapshot()
            if getattr(self.camera_provider, "supports_configuration", True):
                self.camera_provider.configure(
                    exposure_us=snapshot.camera.exposure_us,
                    gain=snapshot.camera.gain,
                )
            self.store.update(
                lambda state: (
                    setattr(state.camera, "connection", status.connection),
                    setattr(state.camera, "implementation_status", status.implementation_status),
                    setattr(state.camera, "device_id", getattr(self.camera_provider, "device_serial", None)),
                    setattr(state.camera, "last_error", None),
                ),
                source="camera_provider",
                reason="Camera connected",
            )
            return status.message
        except Exception as exc:
            self.store.update(
                lambda state: (
                    setattr(state.camera, "connection", ConnectionState.ERROR),
                    setattr(state.camera, "acquisition", AcquisitionState.ERROR),
                    setattr(state.camera, "last_error", str(exc)),
                ),
                source="camera_provider",
                reason="Camera connection failed",
            )
            raise

    def start_camera(self) -> None:
        if self.camera_provider is None:
            raise RuntimeError("No camera provider has been selected.")
        if not self.camera_provider.connected:
            self.connect_camera()
        self.camera_provider.start()
        self.store.update(
            lambda state: setattr(state.camera, "acquisition", AcquisitionState.LIVE),
            source="camera_provider",
            reason="Live camera acquisition started",
        )

    def stop_camera(self) -> None:
        if self.camera_provider is not None:
            self.camera_provider.stop()
        self.store.update(
            lambda state: setattr(state.camera, "acquisition", AcquisitionState.STOPPED),
            source="camera_provider",
            reason="Live camera acquisition stopped",
        )

    def acquire_frame(self, *, fresh: bool = True, timeout_s: float = 2.0) -> CameraFrame:
        if self.camera_provider is None:
            raise RuntimeError("No camera provider has been selected.")
        frame = self.camera_provider.acquire_frame(fresh=fresh, timeout_s=timeout_s)
        self.store.update(
            lambda state: (
                setattr(state.camera, "last_frame_id", frame.frame_id),
                setattr(state.camera, "last_frame_utc", frame.timestamp_utc),
                setattr(state.camera, "shape_yx", frame.shape_yx),
                setattr(state.camera, "full_scale", frame.full_scale),
                setattr(state.camera, "current_z_mm", frame.z_mm),
                setattr(
                    state.camera,
                    "frame_quality",
                    "QUANTITATIVE" if frame.metadata.get("quantitative_valid", True) else "LIVE_PREVIEW_UNVERIFIED",
                ),
                setattr(state.camera, "last_error", None),
                setattr(state.system, "data_kind", DataKind(frame.data_kind)),
            ),
            source="camera_provider",
            reason=f"Acquired quantitative frame {frame.frame_id}",
        )
        return frame

    def close(self) -> None:
        if self.camera_provider is not None:
            self.camera_provider.disconnect()
        self.close_slms()
