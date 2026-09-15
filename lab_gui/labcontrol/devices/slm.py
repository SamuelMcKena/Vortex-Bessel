"""SLM provider backed by the existing, phase-safe v0.6 HEDS implementation."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from slm_lab_control.config import AppConfig
from slm_lab_control.hardware import make_backend
from slm_lab_control.hardware.backends import BaseSlmBackend
from slm_lab_control.phase import PhaseResult

from .base import DeviceStatus, ProviderError
from labcontrol.state import ConnectionState


class SlmProvider(ABC):
    """Owns both panels so separate clients cannot compete for HEDS handles."""

    @abstractmethod
    def connect(self, config: AppConfig) -> list[str]: ...

    @abstractmethod
    def cast(
        self,
        name: str,
        result: PhaseResult,
        transfer_mode: str,
        log_png: Path,
    ) -> str: ...

    @abstractmethod
    def blank(self, config: AppConfig, names: Iterable[str], output_folder: Path) -> list[str]: ...

    @abstractmethod
    def close(self) -> str: ...

    @abstractmethod
    def disconnect(self) -> str: ...

    @abstractmethod
    def status(self, name: str) -> DeviceStatus: ...


class BackendSlmProvider(SlmProvider):
    """Thin adapter; no phase conversion or HEDS semantics are reimplemented."""

    def __init__(self, backend: BaseSlmBackend):
        self.backend = backend
        self._lock = threading.RLock()

    @classmethod
    def from_config(cls, config: AppConfig, project_root: Path) -> "BackendSlmProvider":
        output = config.resolve_output_root(project_root)
        backend = make_backend(
            config.backend,
            output,
            sdk_major=config.sdk_major,
            sdk_minor=config.sdk_minor,
        )
        return cls(backend)

    def connect(self, config: AppConfig) -> list[str]:
        with self._lock:
            messages = [self.backend.init_sdk()]
            for name, phase in (("SLM1", config.slm1), ("SLM2", config.slm2)):
                messages.append(
                    self.backend.connect(
                        name,
                        phase.serial,
                        phase.geometry.wavelength_nm,
                    )
                )
            return messages

    def cast(
        self,
        name: str,
        result: PhaseResult,
        transfer_mode: str,
        log_png: Path,
    ) -> str:
        with self._lock:
            if transfer_mode == "png_file":
                return self.backend.show_png(name, log_png)
            if transfer_mode == "phase_file":
                return self.backend.show_phase_file(name, log_png)
            if transfer_mode == "direct_gray_array":
                return self.backend.show_gray_array(
                    name, result.gray_uint8, log_png, allow_png_fallback=False
                )
            if transfer_mode == "direct_phase_array":
                return self.backend.show_phase_array(
                    name,
                    result.phase_rad,
                    result.gray_uint8,
                    log_png,
                    allow_png_fallback=False,
                )
            if transfer_mode == "auto":
                return self.backend.show_phase_array(
                    name,
                    result.phase_rad,
                    result.gray_uint8,
                    log_png,
                    allow_png_fallback=True,
                )
            raise ProviderError(f"Unknown SLM transfer mode: {transfer_mode}")

    def blank(self, config: AppConfig, names: Iterable[str], output_folder: Path) -> list[str]:
        with self._lock:
            messages: list[str] = []
            for name in names:
                phase = config.slm1 if name == "SLM1" else config.slm2
                transfer = config.transfer_mode
                messages.append(
                    self.backend.blank(
                        name,
                        output_folder / f"{name.lower()}_blank.png",
                        shape=(phase.geometry.height_px, phase.geometry.width_px),
                        gray=0,
                        use_direct=transfer != "png_file",
                        allow_png_fallback=transfer not in {
                            "direct_gray_array",
                            "direct_phase_array",
                        },
                    )
                )
            return messages

    def close(self) -> str:
        with self._lock:
            return self.backend.close()

    def disconnect(self) -> str:
        return self.close()

    def status(self, name: str) -> DeviceStatus:
        state = self.backend.devices.get(name)
        connected = bool(state and state.connected)
        verified = bool(state and state.phase_mode_verified)
        message = state.last_transfer if state else "not initialised"
        if connected and not verified and self.backend.name == "heds":
            message = "connected, phase wavelength not verified"
        return DeviceStatus(
            provider=self.backend.name,
            connection=ConnectionState.CONNECTED if connected else ConnectionState.DISCONNECTED,
            # A successful SDK connection verifies the runtime phase path, not the
            # optical bench.  Physical validation is recorded separately through
            # calibration evidence and must never be inferred from connection state.
            implementation_status="HARDWARE_UNVERIFIED" if self.backend.name == "heds" else "SOFTWARE_TESTED",
            message=message,
        )


class HedsSlmProvider(BackendSlmProvider):
    """Named provider for the locked, wavelength-verified HEDS backend."""

    @classmethod
    def from_config(cls, config: AppConfig, project_root: Path) -> "HedsSlmProvider":
        if config.backend != "heds":
            raise ValueError("HedsSlmProvider requires config.backend='heds'.")
        return cls(
            make_backend(
                "heds",
                config.resolve_output_root(project_root),
                sdk_major=config.sdk_major,
                sdk_minor=config.sdk_minor,
            )
        )


class DummySlmProvider(BackendSlmProvider):
    """File-preserving software provider used for development and CI."""

    @classmethod
    def from_config(cls, config: AppConfig, project_root: Path) -> "DummySlmProvider":
        if config.backend != "dummy":
            raise ValueError("DummySlmProvider requires config.backend='dummy'.")
        return cls(
            make_backend(
                "dummy",
                config.resolve_output_root(project_root),
                sdk_major=config.sdk_major,
                sdk_minor=config.sdk_minor,
            )
        )


def slm_provider_from_config(config: AppConfig, project_root: Path) -> BackendSlmProvider:
    return (
        HedsSlmProvider.from_config(config, project_root)
        if config.backend == "heds"
        else DummySlmProvider.from_config(config, project_root)
    )
