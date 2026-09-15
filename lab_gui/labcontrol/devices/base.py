"""Shared device-provider types."""

from __future__ import annotations

from dataclasses import dataclass

from labcontrol.state import ConnectionState


class ProviderError(RuntimeError):
    """A recoverable provider failure suitable for display to an operator."""


class ProviderUnavailable(ProviderError):
    """The requested provider cannot run in the current environment."""


@dataclass(frozen=True)
class DeviceStatus:
    provider: str
    connection: ConnectionState
    implementation_status: str
    message: str = ""
