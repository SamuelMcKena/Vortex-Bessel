from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from labcontrol.devices.beamage_pipe import (
    COMMAND_WCHAR_COUNTS,
    PIPE_PATH,
    BeamagePipeClient,
    decode_response,
    encode_command,
)
from labcontrol.devices.camera import BeamageCameraProvider
from labcontrol.devices.base import ProviderError


def wide(text: str) -> bytes:
    return (text + "\0").encode("utf-16-le")


class FakePipe:
    def __init__(self, responses: dict[str, str], *, image_path: Path | None = None):
        self.responses = dict(responses)
        self.image_path = image_path
        self.opened = False
        self.closed = False
        self.path = None
        self.payloads: list[bytes] = []

    def open(self, path: str, timeout_s: float) -> None:
        assert timeout_s > 0
        self.path = path
        self.opened = True

    def close(self) -> None:
        self.closed = True
        self.opened = False

    def exchange(self, payload: bytes, timeout_s: float) -> bytes:
        assert timeout_s > 0
        self.payloads.append(payload)
        command = decode_response(payload)
        if command.startswith("*CTLBMPSAVE") and self.image_path is not None:
            return wide(str(self.image_path))
        response = self.responses.get(command)
        if response is None:
            raise TimeoutError(command)
        return wide(response)


def responses() -> dict[str, str]:
    return {
        "*MEASNM": "BEAMAGE-4M-TEST",
        "*GETIMGWID": "2048",
        "*GETIMGHGT": "2048",
        "*CTLSTART": "ACK",
        "*CTLSTOP": "ACK",
        "*GETCAPTURE": "1",
        "*GETALLMEA": "1;2;3;4;5;0.8;12;0.1;0.2;3.5",
        "*GETALLPOS": "101;102;103;104;88;0.9;0.8",
    }


def test_vendor_unicode_command_framing_matches_declared_tchar_counts() -> None:
    for command, count in COMMAND_WCHAR_COUNTS.items():
        payload = encode_command(command)
        assert len(payload) == count * 2
        assert decode_response(payload) == command


def test_connect_identity_start_stop_and_measurement_parsing() -> None:
    transport = FakePipe(responses())
    client = BeamagePipeClient(transport=transport)
    client.connect()
    assert transport.payloads == []
    client.probe_identity()
    assert transport.path == PIPE_PATH
    assert client.identity is not None
    assert client.identity.serial == "BEAMAGE-4M-TEST"
    assert (client.identity.width_px, client.identity.height_px) == (2048, 2048)
    client.start()
    assert client.running
    assert client.measurements()["exposure_ms"] == 3.5
    assert client.positions()["centroid_x"] == 101.0
    client.stop()
    assert not client.running
    client.disconnect()
    assert transport.closed


def test_named_pipe_bmp_is_preview_only_until_hardware_validation(tmp_path: Path) -> None:
    image_path = tmp_path / "beamage.bmp"
    Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8)).save(image_path)
    client = BeamagePipeClient(transport=FakePipe(responses(), image_path=image_path))
    client.connect()
    client.probe_identity()
    provider = BeamageCameraProvider(client)
    status = provider.connect()
    assert status.connection.value == "CONNECTED"
    provider.start()
    frame = provider.acquire_frame()
    assert frame.shape_yx == (8, 8)
    assert frame.metadata["quantitative_valid"] is False
    assert frame.metadata["measurement_scope"] == "LIVE_PREVIEW_ONLY"
    assert frame.metadata["device_serial"] == "BEAMAGE-4M-TEST"


def test_timeout_and_malformed_error_are_recoverable() -> None:
    transport = FakePipe({"*MEASNM": "BEAMAGE-4M-TEST", "*GETIMGWID": "2048", "*GETIMGHGT": "2048"})
    client = BeamagePipeClient(transport=transport)
    client.connect()
    with pytest.raises(TimeoutError):
        client.start()
    assert client.connected is False
    assert transport.closed

    bad = FakePipe({"*MEASNM": "ERROR pipeline disabled"})
    client = BeamagePipeClient(transport=bad)
    client.connect()
    with pytest.raises(ProviderError, match="pipeline disabled"):
        client.probe_identity()
