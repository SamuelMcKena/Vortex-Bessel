"""Calibration dependency graph, physical-change invalidation and readiness."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from .state import ExperimentStore, PhysicalAxiconState, utc_now


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class CalibrationStatus(StrEnum):
    UNKNOWN = "UNKNOWN"
    VALID = "VALID"
    STALE = "STALE"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    FAILED = "FAILED"


class CalibrationKey(StrEnum):
    CAMERA_PIXEL_SCALE = "camera_pixel_scale"
    CAMERA_SENSOR_GEOMETRY = "camera_sensor_geometry"
    CAMERA_RESPONSE = "camera_response"
    CAMERA_TRAVEL_AXIS = "camera_travel_axis"
    SLM_SERIAL_IDENTITY = "slm_serial_identity"
    SLM_PANEL_GEOMETRY = "slm_panel_geometry"
    SLM_PHASE_PATH = "slm_phase_path"
    SLM_PHASE_RESPONSE = "slm_phase_response"
    SLM_NATIVE_ORIENTATION = "slm_native_orientation"
    SLM_TO_OPTICAL_MAPPING = "slm_to_optical_mapping"
    CARRIER_CONVENTION = "carrier_convention"
    RELAY_GEOMETRY = "relay_geometry"
    AXICON_PLACEMENT = "axicon_placement"
    Q0_PROPAGATION_REFERENCE = "q0_propagation_reference"
    VORTEX_PROPAGATION_REFERENCE = "vortex_propagation_reference"
    LOW_ORDER_CORRECTION = "low_order_correction"
    RETRIEVED_RESIDUAL_MAP = "retrieved_residual_map"
    GOLDEN_VORTEX_REFERENCE = "golden_vortex_reference"
    VORTEX_VERIFICATION = "vortex_verification"


class PhysicalChange(StrEnum):
    AXICON_REMOVED_REINSERTED = "AXICON_REMOVED_REINSERTED"
    AXICON_XY_CHANGED = "AXICON_XY_CHANGED"
    CAMERA_REMOUNTED = "CAMERA_REMOUNTED"
    CAMERA_SETTINGS_CHANGED = "CAMERA_SETTINGS_CHANGED"
    PINHOLE_CHANGED = "PINHOLE_CHANGED"
    RELAY_OPTIC_MOVED = "RELAY_OPTIC_MOVED"
    SLM_REPLACED = "SLM_REPLACED"
    LASER_ALIGNMENT_CHANGED = "LASER_ALIGNMENT_CHANGED"
    WAVELENGTH_CHANGED = "WAVELENGTH_CHANGED"


DEPENDENCIES: dict[CalibrationKey, tuple[CalibrationKey, ...]] = {
    CalibrationKey.CAMERA_PIXEL_SCALE: (),
    CalibrationKey.CAMERA_SENSOR_GEOMETRY: (),
    CalibrationKey.CAMERA_RESPONSE: (CalibrationKey.CAMERA_SENSOR_GEOMETRY,),
    CalibrationKey.CAMERA_TRAVEL_AXIS: (
        CalibrationKey.CAMERA_PIXEL_SCALE,
        CalibrationKey.CAMERA_SENSOR_GEOMETRY,
    ),
    CalibrationKey.SLM_SERIAL_IDENTITY: (),
    CalibrationKey.SLM_PANEL_GEOMETRY: (CalibrationKey.SLM_SERIAL_IDENTITY,),
    CalibrationKey.SLM_PHASE_PATH: (
        CalibrationKey.SLM_SERIAL_IDENTITY,
        CalibrationKey.SLM_PANEL_GEOMETRY,
    ),
    CalibrationKey.SLM_PHASE_RESPONSE: (CalibrationKey.SLM_PHASE_PATH,),
    CalibrationKey.SLM_NATIVE_ORIENTATION: (CalibrationKey.SLM_PANEL_GEOMETRY,),
    CalibrationKey.CARRIER_CONVENTION: (
        CalibrationKey.SLM_PANEL_GEOMETRY,
        CalibrationKey.SLM_NATIVE_ORIENTATION,
    ),
    CalibrationKey.RELAY_GEOMETRY: (),
    CalibrationKey.SLM_TO_OPTICAL_MAPPING: (
        CalibrationKey.SLM_NATIVE_ORIENTATION,
        CalibrationKey.RELAY_GEOMETRY,
    ),
    CalibrationKey.AXICON_PLACEMENT: (CalibrationKey.RELAY_GEOMETRY,),
    CalibrationKey.Q0_PROPAGATION_REFERENCE: (
        CalibrationKey.CAMERA_PIXEL_SCALE,
        CalibrationKey.CAMERA_SENSOR_GEOMETRY,
        CalibrationKey.AXICON_PLACEMENT,
    ),
    CalibrationKey.VORTEX_PROPAGATION_REFERENCE: (
        CalibrationKey.Q0_PROPAGATION_REFERENCE,
        CalibrationKey.SLM_TO_OPTICAL_MAPPING,
        CalibrationKey.CARRIER_CONVENTION,
    ),
    CalibrationKey.LOW_ORDER_CORRECTION: (
        CalibrationKey.VORTEX_PROPAGATION_REFERENCE,
        CalibrationKey.SLM_TO_OPTICAL_MAPPING,
    ),
    CalibrationKey.RETRIEVED_RESIDUAL_MAP: (
        CalibrationKey.VORTEX_PROPAGATION_REFERENCE,
        CalibrationKey.SLM_TO_OPTICAL_MAPPING,
    ),
    CalibrationKey.GOLDEN_VORTEX_REFERENCE: (
        CalibrationKey.VORTEX_PROPAGATION_REFERENCE,
        CalibrationKey.LOW_ORDER_CORRECTION,
    ),
    CalibrationKey.VORTEX_VERIFICATION: (
        CalibrationKey.GOLDEN_VORTEX_REFERENCE,
        CalibrationKey.LOW_ORDER_CORRECTION,
    ),
}


DIRECT_INVALIDATION: dict[PhysicalChange, tuple[CalibrationKey, ...]] = {
    PhysicalChange.AXICON_REMOVED_REINSERTED: (CalibrationKey.AXICON_PLACEMENT,),
    PhysicalChange.AXICON_XY_CHANGED: (CalibrationKey.AXICON_PLACEMENT,),
    PhysicalChange.CAMERA_REMOUNTED: (
        CalibrationKey.CAMERA_SENSOR_GEOMETRY,
        CalibrationKey.CAMERA_TRAVEL_AXIS,
    ),
    PhysicalChange.CAMERA_SETTINGS_CHANGED: (
        CalibrationKey.CAMERA_RESPONSE,
        CalibrationKey.Q0_PROPAGATION_REFERENCE,
        CalibrationKey.VORTEX_PROPAGATION_REFERENCE,
    ),
    PhysicalChange.PINHOLE_CHANGED: (
        CalibrationKey.Q0_PROPAGATION_REFERENCE,
        CalibrationKey.VORTEX_PROPAGATION_REFERENCE,
    ),
    PhysicalChange.RELAY_OPTIC_MOVED: (CalibrationKey.RELAY_GEOMETRY,),
    PhysicalChange.SLM_REPLACED: (CalibrationKey.SLM_SERIAL_IDENTITY,),
    PhysicalChange.LASER_ALIGNMENT_CHANGED: (
        CalibrationKey.Q0_PROPAGATION_REFERENCE,
        CalibrationKey.VORTEX_PROPAGATION_REFERENCE,
    ),
    PhysicalChange.WAVELENGTH_CHANGED: (
        CalibrationKey.SLM_PHASE_PATH,
        CalibrationKey.SLM_PHASE_RESPONSE,
        CalibrationKey.Q0_PROPAGATION_REFERENCE,
    ),
}


@dataclass
class CalibrationRecord:
    key: CalibrationKey
    status: CalibrationStatus = CalibrationStatus.UNKNOWN
    calibration_id: str | None = None
    updated_utc: str | None = None
    evidence: tuple[str, ...] = ()
    source_session: str | None = None
    hardware_fingerprint: dict[str, str] = field(default_factory=dict)
    quality_metric: dict[str, float] = field(default_factory=dict)
    notes: str = ""
    stale_reason: str | None = None

    @property
    def dependencies(self) -> tuple[CalibrationKey, ...]:
        return DEPENDENCIES[self.key]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.value,
            "status": self.status.value,
            "calibration_id": self.calibration_id,
            "updated_utc": self.updated_utc,
            "dependencies": [key.value for key in self.dependencies],
            "evidence": list(self.evidence),
            "source_session": self.source_session,
            "hardware_fingerprint": dict(self.hardware_fingerprint),
            "quality_metric": dict(self.quality_metric),
            "notes": self.notes,
            "stale_reason": self.stale_reason,
        }


@dataclass(frozen=True)
class ReadinessReport:
    ready: bool
    statuses: dict[str, str]
    blockers: tuple[str, ...]
    recommended_action: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "statuses": dict(self.statuses),
            "blockers": list(self.blockers),
            "recommended_action": self.recommended_action,
        }


PROCESSING_REQUIREMENTS: tuple[CalibrationKey, ...] = (
    CalibrationKey.SLM_PHASE_PATH,
    CalibrationKey.CAMERA_PIXEL_SCALE,
    CalibrationKey.CAMERA_SENSOR_GEOMETRY,
    CalibrationKey.CAMERA_TRAVEL_AXIS,
    CalibrationKey.AXICON_PLACEMENT,
    CalibrationKey.Q0_PROPAGATION_REFERENCE,
    CalibrationKey.VORTEX_PROPAGATION_REFERENCE,
    CalibrationKey.LOW_ORDER_CORRECTION,
    CalibrationKey.VORTEX_VERIFICATION,
)


class CalibrationRegistry:
    def __init__(self, records: Iterable[CalibrationRecord] | None = None):
        self.records = {key: CalibrationRecord(key) for key in CalibrationKey}
        for record in records or ():
            self.records[record.key] = record

    def mark(
        self,
        key: CalibrationKey | str,
        status: CalibrationStatus | str,
        *,
        calibration_id: str | None = None,
        evidence: Iterable[str] = (),
        source_session: str | None = None,
        hardware_fingerprint: dict[str, str] | None = None,
        quality_metric: dict[str, float] | None = None,
        notes: str = "",
    ) -> CalibrationRecord:
        key = CalibrationKey(key)
        status = CalibrationStatus(status)
        record = self.records[key]
        record.status = status
        record.calibration_id = calibration_id or record.calibration_id
        record.updated_utc = utc_now()
        record.evidence = tuple(str(item) for item in evidence)
        record.source_session = source_session
        record.hardware_fingerprint = dict(hardware_fingerprint or {})
        record.quality_metric = {str(k): float(v) for k, v in (quality_metric or {}).items()}
        record.notes = notes
        record.stale_reason = None if status != CalibrationStatus.STALE else record.stale_reason
        return record

    def status(self, key: CalibrationKey | str) -> CalibrationStatus:
        return self.records[CalibrationKey(key)].status

    def dependent_closure(self, roots: Iterable[CalibrationKey]) -> set[CalibrationKey]:
        affected = set(roots)
        changed = True
        while changed:
            changed = False
            for key, dependencies in DEPENDENCIES.items():
                if key not in affected and any(dependency in affected for dependency in dependencies):
                    affected.add(key)
                    changed = True
        return affected

    def report_physical_change(self, change: PhysicalChange | str) -> tuple[CalibrationKey, ...]:
        change = PhysicalChange(change)
        affected = self.dependent_closure(DIRECT_INVALIDATION[change])
        stamp = utc_now()
        for key in affected:
            record = self.records[key]
            if record.status != CalibrationStatus.UNKNOWN:
                record.status = CalibrationStatus.STALE
            record.updated_utc = stamp
            record.stale_reason = change.value
        return tuple(sorted(affected, key=lambda item: item.value))

    def readiness(
        self, requirements: Iterable[CalibrationKey] = PROCESSING_REQUIREMENTS
    ) -> ReadinessReport:
        required = tuple(requirements)
        blockers = tuple(
            key.value for key in required if self.records[key].status != CalibrationStatus.VALID
        )
        if not blockers:
            action = "No calibration action required."
        elif any("slm_phase" in key or "slm_serial" in key for key in blockers):
            action = "Verify the SLM identity, wavelength and phase-data path before any processing run."
        elif any(key.startswith("camera_") for key in blockers):
            action = "Run camera geometry and travel-axis calibration."
        elif any(
            key in blockers
            for key in (
                CalibrationKey.AXICON_PLACEMENT.value,
                CalibrationKey.Q0_PROPAGATION_REFERENCE.value,
                CalibrationKey.VORTEX_PROPAGATION_REFERENCE.value,
                CalibrationKey.LOW_ORDER_CORRECTION.value,
                CalibrationKey.VORTEX_VERIFICATION.value,
            )
        ):
            action = "Run quick vortex recovery, then fresh independent verification."
        else:
            action = "Calibrate the listed dependencies and repeat readiness evaluation."
        return ReadinessReport(
            ready=not blockers,
            statuses={key.value: self.records[key].status.value for key in required},
            blockers=blockers,
            recommended_action=action,
        )

    def publish(self, store: ExperimentStore, *, reason: str = "Calibration registry updated") -> None:
        statuses = {key.value: record.status.value for key, record in self.records.items()}
        store.update(
            lambda state: setattr(state, "calibration_statuses", statuses),
            source="calibration",
            reason=reason,
        )

    def apply_physical_change(
        self, store: ExperimentStore, change: PhysicalChange | str
    ) -> tuple[CalibrationKey, ...]:
        change = PhysicalChange(change)
        affected = self.report_physical_change(change)

        def update_state(state) -> None:
            state.calibration_statuses = {
                key.value: record.status.value for key, record in self.records.items()
            }
            if change in {
                PhysicalChange.AXICON_REMOVED_REINSERTED,
                PhysicalChange.AXICON_XY_CHANGED,
            }:
                state.system.physical_axicon = PhysicalAxiconState.IN
            if change == PhysicalChange.WAVELENGTH_CHANGED:
                state.slm1.phase_mode_verified = False
                state.slm2.phase_mode_verified = False

        store.update(
            update_state,
            source="physical_change",
            reason=f"Operator reported {change.value}",
        )
        return affected

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "records": {key.value: record.to_dict() for key, record in self.records.items()},
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        temporary.replace(target)
        return target

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationRegistry":
        if int(data.get("schema_version", 1)) != 1:
            raise ValueError("Unsupported calibration registry schema.")
        records = []
        for key_text, raw in data.get("records", {}).items():
            key = CalibrationKey(key_text)
            records.append(
                CalibrationRecord(
                    key=key,
                    status=CalibrationStatus(raw.get("status", "UNKNOWN")),
                    calibration_id=raw.get("calibration_id"),
                    updated_utc=raw.get("updated_utc"),
                    evidence=tuple(raw.get("evidence", ())),
                    source_session=raw.get("source_session"),
                    hardware_fingerprint=dict(raw.get("hardware_fingerprint", {})),
                    quality_metric={
                        str(k): float(v) for k, v in raw.get("quality_metric", {}).items()
                    },
                    notes=raw.get("notes", ""),
                    stale_reason=raw.get("stale_reason"),
                )
            )
        return cls(records)

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationRegistry":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
