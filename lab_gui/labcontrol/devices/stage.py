"""Stage providers with a manual-prompt path that recipes can later automate."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .base import DeviceStatus
from labcontrol.state import ConnectionState


@dataclass(frozen=True)
class MoveRequest:
    target_z_mm: float
    prompt: str
    requires_operator_confirmation: bool


class StageProvider(ABC):
    @abstractmethod
    def connect(self) -> DeviceStatus: ...

    @abstractmethod
    def request_move(self, z_mm: float) -> MoveRequest: ...

    @abstractmethod
    def confirm_position(self, z_mm: float) -> float: ...

    @property
    @abstractmethod
    def position_mm(self) -> float | None: ...


class ManualStageProvider(StageProvider):
    name = "manual"

    def __init__(self):
        self._position: float | None = None

    def connect(self) -> DeviceStatus:
        return DeviceStatus(
            self.name,
            ConnectionState.CONNECTED,
            "SOFTWARE_TESTED",
            "Operator-guided z positioning",
        )

    def request_move(self, z_mm: float) -> MoveRequest:
        z = float(z_mm)
        return MoveRequest(
            target_z_mm=z,
            prompt=f"Move the camera to z = {z:.3f} mm, then confirm Ready.",
            requires_operator_confirmation=True,
        )

    def confirm_position(self, z_mm: float) -> float:
        self._position = float(z_mm)
        return self._position

    @property
    def position_mm(self) -> float | None:
        return self._position


class DummyStageProvider(ManualStageProvider):
    name = "dummy-stage"

    def request_move(self, z_mm: float) -> MoveRequest:
        self._position = float(z_mm)
        return MoveRequest(
            target_z_mm=self._position,
            prompt=f"Synthetic stage moved to z = {self._position:.3f} mm.",
            requires_operator_confirmation=False,
        )
