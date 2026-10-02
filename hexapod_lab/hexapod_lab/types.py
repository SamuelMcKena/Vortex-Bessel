from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class Pose6D:
    """HXP Cartesian pose in millimetres/degrees: X, Y, Z, U, V, W."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    u: float = 0.0
    v: float = 0.0
    w: float = 0.0

    def as_tuple(self) -> tuple[float, float, float, float, float, float]:
        return (self.x, self.y, self.z, self.u, self.v, self.w)

    @classmethod
    def from_iterable(cls, values: Any) -> "Pose6D":
        vals = tuple(float(v) for v in values)
        if len(vals) != 6:
            raise ValueError(f"Pose6D requires 6 values, got {len(vals)}")
        return cls(*vals)

    def plus(self, other: "Pose6D") -> "Pose6D":
        return Pose6D(*(a + b for a, b in zip(self.as_tuple(), other.as_tuple())))

    def lerp(self, other: "Pose6D", fraction: float) -> "Pose6D":
        t = max(0.0, min(1.0, float(fraction)))
        return Pose6D(*(a + (b - a) * t for a, b in zip(self.as_tuple(), other.as_tuple())))

    def max_abs_delta(self, other: "Pose6D") -> float:
        return max(abs(a - b) for a, b in zip(self.as_tuple(), other.as_tuple()))


class MotionState(str, Enum):
    DISCONNECTED = "disconnected"
    IDLE = "idle"
    MOVING = "moving"
    FAULT = "fault"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class HexapodSnapshot:
    timestamp_s: float
    actual: Pose6D
    setpoint: Pose6D | None = None
    target: Pose6D | None = None
    state: MotionState = MotionState.IDLE
    status_code: int | None = None
    status_text: str = ""
    connected: bool = True
    provider: str = "virtual"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LaserSnapshot:
    """State of the process-beam Pockels-cell command path.

    ``gate_enabled=True`` means the software has requested the Pockels cell OPEN
    (process beam enabled). It does *not* mean the PHAROS laser source itself is
    powered on.
    """

    timestamp_s: float
    gate_enabled: bool
    connected: bool = True
    provider: str = "virtual"
    connector_name: str = "LX13"
    readback_known: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def pockels_open(self) -> bool:
        return self.gate_enabled


@dataclass(frozen=True, slots=True)
class AttenuatorSnapshot:
    """Normalized attenuator state expressed as requested transmission percent."""

    timestamp_s: float
    transmission_percent: float
    connected: bool = True
    provider: str = "virtual"
    readback_known: bool = True
    device_name: str = "Virtual attenuator"
    metadata: Mapping[str, Any] = field(default_factory=dict)
