"""Typed, serialisable and observable authoritative experiment state."""

from __future__ import annotations

import copy
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable

from slm_lab_control.config import AppConfig, SlmPhaseConfig


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class ConnectionState(StrEnum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    ERROR = "ERROR"


class AcquisitionState(StrEnum):
    STOPPED = "STOPPED"
    LIVE = "LIVE"
    CAPTURING = "CAPTURING"
    ERROR = "ERROR"


class PhysicalAxiconState(StrEnum):
    UNKNOWN = "UNKNOWN"
    OUT = "OUT"
    IN = "IN"


class DataKind(StrEnum):
    EXPERIMENT = "EXPERIMENT"
    REPLAY = "REPLAY"
    SYNTHETIC = "SYNTHETIC"


@dataclass
class ApplicationSettings:
    backend: str = "dummy"
    sdk_major: int = 4
    sdk_minor: int = 2
    transfer_mode: str = "png_file"
    output_root: str = "outputs/casts"
    preset_root: str = "presets"
    auto_regenerate: bool = True


@dataclass
class SystemState:
    experiment_label: str = "Untitled experiment"
    wavelength_nm: float = 1030.0
    physical_axicon: PhysicalAxiconState = PhysicalAxiconState.UNKNOWN
    physical_configuration: str = "Dual SLM + 4F + physical axicon"
    data_kind: DataKind = DataKind.EXPERIMENT
    notes: str = ""


@dataclass
class SlmState:
    phase: SlmPhaseConfig
    connection: ConnectionState = ConnectionState.DISCONNECTED
    phase_mode_verified: bool = False
    complete_phase_sha256: str | None = None
    last_cast_sha256: str | None = None
    last_cast_utc: str | None = None
    last_hardware_command: str | None = None
    accepted_correction_id: str | None = None


@dataclass
class CameraState:
    provider: str = "dummy"
    implementation_status: str = "SOFTWARE_TESTED"
    device_id: str | None = None
    connection: ConnectionState = ConnectionState.DISCONNECTED
    acquisition: AcquisitionState = AcquisitionState.STOPPED
    exposure_us: float = 1000.0
    gain: float = 0.0
    pixel_size_um: float = 5.5
    shape_yx: tuple[int, int] = (2048, 2048)
    full_scale: float = 4095.0
    settings_id: str = "UNSET"
    current_z_mm: float | None = None
    z_reference: str = "UNSET"
    last_frame_id: str | None = None
    last_frame_utc: str | None = None
    last_error: str | None = None
    exposure_control: str = "SUPPORTED"
    gain_control: str = "SUPPORTED"
    frame_quality: str = "QUANTITATIVE"


@dataclass
class GeometryState:
    camera_travel_axis_calibrated: bool = False
    camera_travel_calibration_id: str | None = None
    optical_axis_reference: str = "camera-relative"


@dataclass
class SessionState:
    session_id: str | None = None
    session_root: str | None = None
    recipe: str | None = None
    recipe_run_id: str | None = None
    current_step: str | None = None
    current_trial: str | None = None
    accepted_trial: str | None = None
    golden_reference_id: str | None = None


@dataclass(frozen=True)
class StateDelta:
    path: str
    before: Any
    after: Any


@dataclass(frozen=True)
class StateChangeEvent:
    event_id: str
    revision: int
    timestamp_utc: str
    source: str
    reason: str
    changes: tuple[StateDelta, ...]

    @property
    def changed_paths(self) -> tuple[str, ...]:
        return tuple(change.path for change in self.changes)


@dataclass
class ExperimentState:
    """One complete description of what the application currently believes.

    Presets, cast state, measurements, calibrations and accepted corrections are
    represented separately.  In particular, configuring a phase is not treated
    as proof that it was cast, and casting is not treated as a measurement.
    """

    schema_version: int = 1
    revision: int = 0
    updated_utc: str = field(default_factory=utc_now)
    application: ApplicationSettings = field(default_factory=ApplicationSettings)
    system: SystemState = field(default_factory=SystemState)
    slm1: SlmState = field(
        default_factory=lambda: SlmState(SlmPhaseConfig(name="SLM1"))
    )
    slm2: SlmState = field(
        default_factory=lambda: SlmState(SlmPhaseConfig(name="SLM2"))
    )
    camera: CameraState = field(default_factory=CameraState)
    geometry: GeometryState = field(default_factory=GeometryState)
    session: SessionState = field(default_factory=SessionState)
    calibration_statuses: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_app_config(cls, config: AppConfig) -> "ExperimentState":
        cfg = copy.deepcopy(config)
        cfg.apply_locked_hardware()
        return cls(
            application=ApplicationSettings(
                backend=cfg.backend,
                sdk_major=cfg.sdk_major,
                sdk_minor=cfg.sdk_minor,
                transfer_mode=cfg.transfer_mode,
                output_root=cfg.output_root,
                preset_root=cfg.preset_root,
                auto_regenerate=cfg.auto_regenerate,
            ),
            system=SystemState(wavelength_nm=cfg.slm1.geometry.wavelength_nm),
            slm1=SlmState(copy.deepcopy(cfg.slm1)),
            slm2=SlmState(copy.deepcopy(cfg.slm2)),
        )

    def to_app_config(self) -> AppConfig:
        cfg = AppConfig(
            backend=self.application.backend,
            sdk_major=self.application.sdk_major,
            sdk_minor=self.application.sdk_minor,
            slm1=copy.deepcopy(self.slm1.phase),
            slm2=copy.deepcopy(self.slm2.phase),
            output_root=self.application.output_root,
            preset_root=self.application.preset_root,
            auto_regenerate=self.application.auto_regenerate,
            transfer_mode=self.application.transfer_mode,
        )
        cfg.apply_locked_hardware()
        return cfg

    def active_vortex_contributions(self) -> dict[str, int]:
        active: dict[str, int] = {}
        for name, slm in (("SLM1", self.slm1), ("SLM2", self.slm2)):
            if slm.phase.switches.vortex and int(slm.phase.vortex_charge) != 0:
                active[name] = int(slm.phase.vortex_charge)
        return active

    def effective_vortex_charge(self) -> int:
        return sum(self.active_vortex_contributions().values())

    def analysis_family(self) -> str:
        return "vortex_bessel" if self.effective_vortex_charge() != 0 else "central_beam"

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported ExperimentState schema {self.schema_version}.")
        if self.system.wavelength_nm <= 0:
            raise ValueError("System wavelength must be positive.")
        for name, slm in (("SLM1", self.slm1), ("SLM2", self.slm2)):
            slm.phase.name = name
            slm.phase.apply_locked_hardware()
            if abs(slm.phase.geometry.wavelength_nm - self.system.wavelength_nm) > 1e-9:
                raise ValueError(
                    f"{name} wavelength {slm.phase.geometry.wavelength_nm:g} nm does not match "
                    f"system wavelength {self.system.wavelength_nm:g} nm."
                )
        if self.camera.pixel_size_um <= 0 or self.camera.full_scale <= 0:
            raise ValueError("Camera pixel size and full scale must be positive.")
        if len(self.camera.shape_yx) != 2 or min(self.camera.shape_yx) < 4:
            raise ValueError("Camera shape must be a two-dimensional [rows, columns] size.")

    def to_dict(self) -> dict[str, Any]:
        return _primitive(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentState":
        a = data.get("application", {})
        s = data.get("system", {})
        c = data.get("camera", {})
        g = data.get("geometry", {})
        session = data.get("session", {})

        def slm(name: str) -> SlmState:
            raw = data.get(name.lower(), {})
            return SlmState(
                phase=SlmPhaseConfig.from_dict(raw.get("phase", {"name": name})),
                connection=ConnectionState(raw.get("connection", "DISCONNECTED")),
                phase_mode_verified=bool(raw.get("phase_mode_verified", False)),
                complete_phase_sha256=raw.get("complete_phase_sha256"),
                last_cast_sha256=raw.get("last_cast_sha256"),
                last_cast_utc=raw.get("last_cast_utc"),
                last_hardware_command=raw.get("last_hardware_command"),
                accepted_correction_id=raw.get("accepted_correction_id"),
            )

        state = cls(
            schema_version=int(data.get("schema_version", 1)),
            revision=int(data.get("revision", 0)),
            updated_utc=str(data.get("updated_utc", utc_now())),
            application=ApplicationSettings(**a),
            system=SystemState(
                experiment_label=s.get("experiment_label", "Untitled experiment"),
                wavelength_nm=float(s.get("wavelength_nm", 1030.0)),
                physical_axicon=PhysicalAxiconState(s.get("physical_axicon", "UNKNOWN")),
                physical_configuration=s.get(
                    "physical_configuration", "Dual SLM + 4F + physical axicon"
                ),
                data_kind=DataKind(s.get("data_kind", "EXPERIMENT")),
                notes=s.get("notes", ""),
            ),
            slm1=slm("SLM1"),
            slm2=slm("SLM2"),
            camera=CameraState(
                provider=c.get("provider", "dummy"),
                implementation_status=c.get("implementation_status", "SOFTWARE_TESTED"),
                device_id=c.get("device_id"),
                connection=ConnectionState(c.get("connection", "DISCONNECTED")),
                acquisition=AcquisitionState(c.get("acquisition", "STOPPED")),
                exposure_us=float(c.get("exposure_us", 1000.0)),
                gain=float(c.get("gain", 0.0)),
                pixel_size_um=float(c.get("pixel_size_um", 5.5)),
                shape_yx=tuple(c.get("shape_yx", (2048, 2048))),
                full_scale=float(c.get("full_scale", 4095.0)),
                settings_id=c.get("settings_id", "UNSET"),
                current_z_mm=c.get("current_z_mm"),
                z_reference=c.get("z_reference", "UNSET"),
                last_frame_id=c.get("last_frame_id"),
                last_frame_utc=c.get("last_frame_utc"),
                last_error=c.get("last_error"),
                exposure_control=c.get("exposure_control", "SUPPORTED"),
                gain_control=c.get("gain_control", "SUPPORTED"),
                frame_quality=c.get("frame_quality", "QUANTITATIVE"),
            ),
            geometry=GeometryState(**g),
            session=SessionState(**session),
            calibration_statuses=dict(data.get("calibration_statuses", {})),
        )
        state.validate()
        return state


Subscriber = Callable[[StateChangeEvent, ExperimentState], None]


class ExperimentStore:
    """Thread-safe owner of the current state and its structured change log."""

    def __init__(self, initial: ExperimentState | None = None, event_limit: int = 1000):
        self._lock = threading.RLock()
        self._state = copy.deepcopy(initial or ExperimentState())
        self._state.validate()
        self._subscribers: list[Subscriber] = []
        self._events: list[StateChangeEvent] = []
        self._event_limit = max(1, int(event_limit))

    def snapshot(self) -> ExperimentState:
        with self._lock:
            return copy.deepcopy(self._state)

    @property
    def state(self) -> ExperimentState:
        """A defensive snapshot; callers cannot mutate authoritative state in place."""

        return self.snapshot()

    def subscribe(self, callback: Subscriber) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def update(
        self,
        mutator: Callable[[ExperimentState], None],
        *,
        source: str,
        reason: str,
    ) -> StateChangeEvent | None:
        with self._lock:
            before = copy.deepcopy(self._state)
            after = copy.deepcopy(self._state)
            mutator(after)
            after.validate()
            changes = tuple(_diff(_primitive(before), _primitive(after)))
            if not changes:
                return None
            after.revision = before.revision + 1
            after.updated_utc = utc_now()
            event = StateChangeEvent(
                event_id=f"state-{after.revision:08d}",
                revision=after.revision,
                timestamp_utc=after.updated_utc,
                source=source,
                reason=reason,
                changes=changes,
            )
            self._state = after
            self._events.append(event)
            self._events = self._events[-self._event_limit :]
            subscribers = tuple(self._subscribers)
            snapshot = copy.deepcopy(after)
        for callback in subscribers:
            callback(event, copy.deepcopy(snapshot))
        return event

    def replace_app_config(
        self, config: AppConfig, *, source: str = "slm_gui", reason: str = "SLM controls changed"
    ) -> StateChangeEvent | None:
        cfg = copy.deepcopy(config)
        cfg.apply_locked_hardware()

        def apply(state: ExperimentState) -> None:
            state.application = ApplicationSettings(
                backend=cfg.backend,
                sdk_major=cfg.sdk_major,
                sdk_minor=cfg.sdk_minor,
                transfer_mode=cfg.transfer_mode,
                output_root=cfg.output_root,
                preset_root=cfg.preset_root,
                auto_regenerate=cfg.auto_regenerate,
            )
            state.slm1.phase = copy.deepcopy(cfg.slm1)
            state.slm2.phase = copy.deepcopy(cfg.slm2)
            state.system.wavelength_nm = cfg.slm1.geometry.wavelength_nm

        return self.update(apply, source=source, reason=reason)

    def set_vortex(
        self, slm_name: str, enabled: bool, charge: int, *, source: str = "api"
    ) -> StateChangeEvent | None:
        key = slm_name.lower()
        if key not in {"slm1", "slm2"}:
            raise ValueError("slm_name must be SLM1 or SLM2.")

        def apply(state: ExperimentState) -> None:
            slm = getattr(state, key)
            slm.phase.switches.vortex = bool(enabled)
            slm.phase.vortex_charge = int(charge)

        return self.update(
            apply,
            source=source,
            reason=f"Set {slm_name.upper()} vortex enabled={bool(enabled)} charge={int(charge)}",
        )

    def set_camera_z(self, z_mm: float | None, *, source: str = "api") -> StateChangeEvent | None:
        if z_mm is not None and not isinstance(z_mm, (int, float)):
            raise TypeError("z_mm must be numeric or None.")
        return self.update(
            lambda state: setattr(state.camera, "current_z_mm", None if z_mm is None else float(z_mm)),
            source=source,
            reason="Camera z updated",
        )

    def recent_events(self) -> tuple[StateChangeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.snapshot().to_dict(), indent=2, allow_nan=False)
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text(payload, encoding="utf-8")
        temp.replace(target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentStore":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(ExperimentState.from_dict(data))


def _primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {name: _primitive(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {str(k): _primitive(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_primitive(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _diff(before: Any, after: Any, prefix: str = "") -> Iterable[StateDelta]:
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in before:
                yield StateDelta(path, None, after[key])
            elif key not in after:
                yield StateDelta(path, before[key], None)
            else:
                yield from _diff(before[key], after[key], path)
        return
    if before != after:
        yield StateDelta(prefix, before, after)
