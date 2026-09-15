"""Quantitative camera providers for dummy, replay and Beamage operation."""

from __future__ import annotations

import hashlib
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from labcontrol.state import ExperimentState, utc_now
from .base import DeviceStatus, ProviderError, ProviderUnavailable
from labcontrol.state import ConnectionState


SUPPORTED_FRAME_SUFFIXES = {".bmg", ".npy", ".txt", ".csv", ".bmp", ".png", ".tif", ".tiff"}


@dataclass
class CameraFrame:
    data: np.ndarray
    frame_id: str
    timestamp_utc: str
    provider: str
    exposure_us: float
    gain: float
    full_scale: float
    z_mm: float | None
    data_kind: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        array = np.asarray(self.data)
        if array.ndim != 2 or min(array.shape) < 4 or not np.isrealobj(array):
            raise ValueError("CameraFrame data must be a real two-dimensional matrix.")
        if not np.isfinite(array).all():
            raise ValueError("CameraFrame contains NaN or infinity.")
        self.data = np.ascontiguousarray(array, dtype=np.float64)

    @property
    def shape_yx(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.data.shape)

    @property
    def pixel_sha256(self) -> str:
        return hashlib.sha256(self.data.tobytes()).hexdigest()


class CameraProvider(ABC):
    name = "camera"
    implementation_status = "HARDWARE_UNVERIFIED"
    declared_data_kind = "EXPERIMENT"

    @abstractmethod
    def connect(self) -> DeviceStatus: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def acquire_frame(self, *, fresh: bool = True, timeout_s: float = 2.0) -> CameraFrame: ...

    @abstractmethod
    def configure(self, *, exposure_us: float, gain: float) -> None: ...

    @property
    @abstractmethod
    def connected(self) -> bool: ...


class DummyCameraProvider(CameraProvider):
    """Deterministic state-aware synthetic camera for development and CI."""

    name = "dummy"
    implementation_status = "SOFTWARE_TESTED"
    declared_data_kind = "SYNTHETIC"

    def __init__(
        self,
        state_supplier: Callable[[], ExperimentState],
        shape_yx: tuple[int, int] = (512, 512),
        seed: int = 20260915,
    ):
        self._state_supplier = state_supplier
        self._shape = tuple(int(v) for v in shape_yx)
        self._seed = int(seed)
        self._counter = 0
        self._connected = False
        self._running = False
        self._exposure_us = 1000.0
        self._gain = 0.0
        self._lock = threading.RLock()

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> DeviceStatus:
        self._connected = True
        return DeviceStatus(
            self.name,
            ConnectionState.CONNECTED,
            self.implementation_status,
            "Synthetic quantitative frames; never experimental evidence",
        )

    def disconnect(self) -> None:
        with self._lock:
            self._running = False
            self._connected = False

    def start(self) -> None:
        if not self._connected:
            self.connect()
        self._running = True

    def stop(self) -> None:
        self._running = False

    def configure(self, *, exposure_us: float, gain: float) -> None:
        if exposure_us <= 0 or gain < 0:
            raise ValueError("Exposure must be positive and gain non-negative.")
        self._exposure_us = float(exposure_us)
        self._gain = float(gain)

    def acquire_frame(self, *, fresh: bool = True, timeout_s: float = 2.0) -> CameraFrame:
        del fresh, timeout_s
        with self._lock:
            if not self._connected:
                raise ProviderError("Dummy camera is disconnected.")
            state = self._state_supplier()
            self._counter += 1
            y, x = np.indices(self._shape, dtype=np.float64)
            z = state.camera.current_z_mm if state.camera.current_z_mm is not None else 38.0
            q = state.effective_vortex_charge()
            slm2 = state.slm2.phase
            # Visible movement with z and steering makes replay/demo workflows
            # exercise the real analysis path without claiming propagation truth.
            cx = (self._shape[1] - 1) / 2 + 0.08 * (z - 38.0) + 0.8 * slm2.steering_x_mrad
            cy = (self._shape[0] - 1) / 2 - 0.05 * (z - 38.0) + 0.8 * slm2.steering_y_mrad
            r = np.hypot(x - cx, y - cy)
            theta = np.arctan2(y - cy, x - cx)
            if q:
                radius = 24.0 + 0.45 * abs(q) + 0.05 * (z - 38.0)
                radial = np.exp(-0.5 * ((r - radius) / 3.2) ** 2)
                asymmetry = 0.24 + 0.7 * slm2.z22_cos_amp_waves
                signal = radial * np.clip(1.0 + asymmetry * np.cos(2 * theta), 0.05, None)
            else:
                sx = 10.0 * (1.0 + 0.25 * slm2.z22_cos_amp_waves)
                sy = 10.0 * (1.0 - 0.25 * slm2.z22_cos_amp_waves)
                signal = np.exp(-0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))
            rng = np.random.default_rng(self._seed + self._counter)
            scale = min(3600.0, 1800.0 * self._exposure_us / 1000.0 * (1.0 + 0.05 * self._gain))
            data = 18.0 + scale * signal + rng.normal(0.0, 1.2, self._shape)
            frame_id = f"dummy-{self._counter:08d}"
            return CameraFrame(
                data=data,
                frame_id=frame_id,
                timestamp_utc=utc_now(),
                provider=self.name,
                exposure_us=self._exposure_us,
                gain=self._gain,
                full_scale=4095.0,
                z_mm=float(z),
                data_kind="SYNTHETIC",
                metadata={
                    "warning": "Synthetic development frame; not optical propagation or lab evidence.",
                    "effective_vortex_charge": q,
                },
            )


class ReplayCameraProvider(CameraProvider):
    """Replays quantitative BMG/numeric images through the CameraProvider API."""

    name = "replay"
    implementation_status = "REPLAY_VALIDATED"

    def __init__(
        self,
        sources: str | Path | Sequence[str | Path],
        *,
        loop: bool = True,
        exposure_us: float = 0.0,
        gain: float = 0.0,
        full_scale: float = 4095.0,
        z_by_name: dict[str, float] | None = None,
        source_data_kind: str = "REPLAY",
    ):
        if isinstance(sources, (str, Path)):
            source = Path(sources)
            paths = (
                sorted(p for p in source.iterdir() if p.suffix.lower() in SUPPORTED_FRAME_SUFFIXES)
                if source.is_dir()
                else [source]
            )
        else:
            paths = [Path(p) for p in sources]
        self._paths = [p for p in paths if p.suffix.lower() in SUPPORTED_FRAME_SUFFIXES]
        if not self._paths:
            raise ValueError("ReplayCamera requires at least one supported quantitative frame.")
        self._loop = bool(loop)
        self._index = 0
        self._connected = False
        self._running = False
        self._exposure_us = float(exposure_us)
        self._gain = float(gain)
        self._full_scale = float(full_scale)
        self._z_by_name = dict(z_by_name or {})
        if source_data_kind not in {"REPLAY", "EXPERIMENT", "SYNTHETIC"}:
            raise ValueError("Replay source_data_kind must be REPLAY, EXPERIMENT or SYNTHETIC.")
        self._source_data_kind = source_data_kind
        self.declared_data_kind = source_data_kind
        self._lock = threading.RLock()

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> DeviceStatus:
        self._connected = True
        return DeviceStatus(
            self.name,
            ConnectionState.CONNECTED,
            self.implementation_status,
            f"{len(self._paths)} quantitative replay frame(s)",
        )

    def disconnect(self) -> None:
        with self._lock:
            self._running = False
            self._connected = False

    def start(self) -> None:
        if not self._connected:
            self.connect()
        self._running = True

    def stop(self) -> None:
        self._running = False

    def configure(self, *, exposure_us: float, gain: float) -> None:
        self._exposure_us = float(exposure_us)
        self._gain = float(gain)

    def acquire_frame(self, *, fresh: bool = True, timeout_s: float = 2.0) -> CameraFrame:
        del fresh, timeout_s
        with self._lock:
            if not self._connected:
                raise ProviderError("Replay camera is disconnected.")
            if self._index >= len(self._paths):
                if not self._loop:
                    raise EOFError("Replay camera reached the final frame.")
                self._index = 0
            path = self._paths[self._index]
            self._index += 1
            from vbb_study.lab.core import load_array

            data = load_array(path)
            z = self._z_by_name.get(path.name, self._z_by_name.get(path.stem))
            return CameraFrame(
                data=data,
                frame_id=f"replay-{self._index:08d}-{path.stem}",
                timestamp_utc=utc_now(),
                provider=self.name,
                exposure_us=self._exposure_us,
                gain=self._gain,
                full_scale=self._full_scale,
                z_mm=z,
                data_kind=self._source_data_kind,
                metadata={
                    "source_path": str(path),
                    "source_suffix": path.suffix.lower(),
                    "acquisition_route": "REPLAY",
                },
            )


class BeamageBridge(Protocol):
    """Contract to be implemented against Gentec's official .NET named-pipe example."""

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def configure(self, exposure_us: float, gain: float) -> None: ...
    def read_quantitative_frame(self, timeout_s: float) -> tuple[np.ndarray, dict[str, Any]]: ...


class BeamageCameraProvider(CameraProvider):
    """Hardware adapter boundary for PC-Beamage's official named-pipe route.

    Gentec publishes a .NET named-pipe example and states that PC-Beamage must be
    installed and running.  The vendor bridge/protocol is not bundled here, so a
    concrete bridge must be injected on the Windows lab PC.  Refusing to connect
    without it is deliberate; screenshots are never treated as camera matrices.
    """

    name = "beamage"
    implementation_status = "HARDWARE_UNVERIFIED"
    declared_data_kind = "EXPERIMENT"

    def __init__(self, bridge: BeamageBridge | None = None):
        self._bridge = bridge
        self._connected = False
        self._running = False
        self._exposure_us = 1000.0
        self._gain = 0.0
        self._counter = 0
        self._lock = threading.RLock()

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> DeviceStatus:
        if self._bridge is None:
            raise ProviderUnavailable(
                "Beamage bridge is not configured. Install/run PC-Beamage and bind a bridge built from "
                "Gentec's official BEAMAGE .NET named-pipe example; screen capture is not supported."
            )
        self._bridge.connect()
        self._connected = True
        return DeviceStatus(
            self.name,
            ConnectionState.CONNECTED,
            self.implementation_status,
            "Official named-pipe bridge connected; live bench behaviour still requires validation",
        )

    def disconnect(self) -> None:
        with self._lock:
            try:
                if self._bridge is not None:
                    self._bridge.disconnect()
            finally:
                self._running = False
                self._connected = False

    def start(self) -> None:
        if not self._connected:
            self.connect()
        assert self._bridge is not None
        self._bridge.start()
        self._running = True

    def stop(self) -> None:
        if self._bridge is not None and self._connected:
            self._bridge.stop()
        self._running = False

    def configure(self, *, exposure_us: float, gain: float) -> None:
        if exposure_us <= 0 or gain < 0:
            raise ValueError("Exposure must be positive and gain non-negative.")
        self._exposure_us = float(exposure_us)
        self._gain = float(gain)
        if self._bridge is not None and self._connected:
            self._bridge.configure(self._exposure_us, self._gain)

    def acquire_frame(self, *, fresh: bool = True, timeout_s: float = 2.0) -> CameraFrame:
        del fresh
        with self._lock:
            if not self._connected or self._bridge is None:
                raise ProviderError("Beamage provider is disconnected.")
            data, metadata = self._bridge.read_quantitative_frame(float(timeout_s))
            self._counter += 1
            return CameraFrame(
                data=data,
                frame_id=f"beamage-{self._counter:08d}",
                timestamp_utc=utc_now(),
                provider=self.name,
                exposure_us=float(metadata.get("exposure_us", self._exposure_us)),
                gain=float(metadata.get("gain", self._gain)),
                full_scale=float(metadata.get("full_scale", 4095.0)),
                z_mm=metadata.get("z_mm"),
                data_kind="EXPERIMENT",
                metadata=dict(metadata),
            )
