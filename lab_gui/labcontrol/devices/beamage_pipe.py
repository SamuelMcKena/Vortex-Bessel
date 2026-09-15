"""PC-Beamage named-pipe client derived from Gentec's supplied C++ example.

The vendor example is a Unicode MFC application.  It opens
``\\\\.\\pipe\\pipe_beamage`` for duplex I/O and writes UTF-16LE ``TCHAR``
command buffers.  The explicit character counts below mirror that source.  For
commands where the example writes beyond the terminating NUL, this client pads
with deterministic NULs instead of reproducing undefined memory reads.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from .base import ProviderError, ProviderUnavailable


PIPE_PATH = r"\\.\pipe\pipe_beamage"

# Exact TCHAR counts used by NamedPipeClient-V1.00.12/NamedPipeClientDlg.cpp.
COMMAND_WCHAR_COUNTS: dict[str, int] = {
    "*MEASNM": 8,
    "*GETCAPTURE": 11,
    "*CTLSTART": 9,
    "*CTLSTOP": 8,
    "*GETALLMEA": 10,
    "*GETALLPOS": 10,
    "*GETIMGWID": 12,
    "*GETIMGHGT": 12,
    "*CTLBMPSAVE": 12,
    "*CTLBMPSAVE110": 14,
}


class PipeTransport(Protocol):
    def open(self, path: str, timeout_s: float) -> None: ...
    def close(self) -> None: ...
    def exchange(self, payload: bytes, timeout_s: float) -> bytes: ...


class Win32NamedPipeTransport:
    """Small pywin32 transport with polling reads and bounded timeouts."""

    def __init__(self, *, poll_interval_s: float = 0.01, max_response_bytes: int = 65536):
        self.poll_interval_s = float(poll_interval_s)
        self.max_response_bytes = int(max_response_bytes)
        self._handle = None
        self._win32file = None
        self._win32pipe = None

    def open(self, path: str, timeout_s: float) -> None:
        if os.name != "nt":
            raise ProviderUnavailable("The PC-Beamage named pipe is available only on Windows.")
        try:
            import win32file
            import win32pipe
        except ImportError as exc:
            raise ProviderUnavailable(
                "pywin32 is required for PC-Beamage. Install it in the active Anaconda environment."
            ) from exc
        try:
            win32pipe.WaitNamedPipe(path, max(1, int(float(timeout_s) * 1000)))
            handle = win32file.CreateFile(
                path,
                win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                0,
                None,
                win32file.OPEN_EXISTING,
                0,
                None,
            )
            try:
                win32pipe.SetNamedPipeHandleState(handle, win32pipe.PIPE_READMODE_MESSAGE, None, None)
            except Exception:
                # Some server configurations expose byte mode; request/response
                # still works because the vendor messages are short.
                pass
        except Exception as exc:
            raise ProviderUnavailable(
                f"Could not open {path}. Start PC-Beamage, connect the camera, and enable Pipeline. ({exc})"
            ) from exc
        self._handle = handle
        self._win32file = win32file
        self._win32pipe = win32pipe

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            try:
                handle.Close()
            except Exception:
                pass

    def exchange(self, payload: bytes, timeout_s: float) -> bytes:
        if self._handle is None or self._win32file is None or self._win32pipe is None:
            raise ProviderError("PC-Beamage named pipe is not open.")
        try:
            self._win32file.WriteFile(self._handle, payload)
        except Exception as exc:
            self.close()
            raise ProviderError(f"PC-Beamage pipe write failed: {exc}") from exc

        deadline = time.monotonic() + max(0.01, float(timeout_s))
        while time.monotonic() < deadline:
            try:
                _peek_data, available, _remaining = self._win32pipe.PeekNamedPipe(self._handle, 0)
                if available:
                    _status, data = self._win32file.ReadFile(
                        self._handle, min(self.max_response_bytes, max(2, int(available)))
                    )
                    return bytes(data)
            except Exception as exc:
                self.close()
                raise ProviderError(f"PC-Beamage pipe read failed: {exc}") from exc
            time.sleep(self.poll_interval_s)
        raise TimeoutError(f"PC-Beamage did not respond within {float(timeout_s):.2f} s.")


def encode_command(command: str) -> bytes:
    """Encode one command using the vendor example's Unicode TCHAR framing."""

    if not command.startswith("*") or any(ch in command for ch in "\r\n"):
        raise ValueError("Invalid PC-Beamage command.")
    count = COMMAND_WCHAR_COUNTS.get(command, len(command) + 1)
    encoded = (command + "\0").encode("utf-16-le")
    return encoded[: count * 2].ljust(count * 2, b"\0")


def decode_response(payload: bytes) -> str:
    """Decode the NUL-terminated UTF-16LE response used by the MFC client."""

    if len(payload) % 2:
        payload = payload[:-1]
    text = payload.decode("utf-16-le", errors="replace")
    return text.split("\0", 1)[0].strip()


@dataclass(frozen=True)
class BeamageIdentity:
    serial: str
    width_px: int | None
    height_px: int | None


class BeamagePipeClient:
    """Thread-safe high-level client for the official PC-Beamage command set."""

    def __init__(
        self,
        transport: PipeTransport | None = None,
        *,
        pipe_path: str = PIPE_PATH,
        timeout_s: float = 2.0,
    ):
        self.transport = transport or Win32NamedPipeTransport()
        self.pipe_path = str(pipe_path)
        self.timeout_s = float(timeout_s)
        self.identity: BeamageIdentity | None = None
        self._connected = False
        self._running = False
        self._lock = threading.RLock()

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def running(self) -> bool:
        return self._running

    def command(self, command: str, *, timeout_s: float | None = None) -> str:
        with self._lock:
            if not self._connected:
                raise ProviderError("PC-Beamage named pipe is disconnected.")
            try:
                payload = self.transport.exchange(
                    encode_command(command), timeout_s or self.timeout_s
                )
            except Exception:
                # A broken duplex exchange invalidates the handle.  Require an
                # explicit reconnect instead of leaving a false CONNECTED state.
                self.transport.close()
                self._connected = False
                self._running = False
                raise
            response = decode_response(payload)
            if not response:
                raise ProviderError(f"PC-Beamage returned an empty response to {command}.")
            if response.upper().startswith(("ERR", "ERROR", "NACK")):
                raise ProviderError(f"PC-Beamage rejected {command}: {response}")
            return response

    def connect(self) -> None:
        with self._lock:
            if self._connected:
                return
            self.transport.open(self.pipe_path, self.timeout_s)
            self._connected = True
            try:
                serial = self.command("*MEASNM")
                width = self._query_optional_int("*GETIMGWID")
                height = self._query_optional_int("*GETIMGHGT")
                self.identity = BeamageIdentity(serial=serial, width_px=width, height_px=height)
            except Exception:
                self.disconnect()
                raise

    def disconnect(self) -> None:
        with self._lock:
            self.transport.close()
            self._connected = False
            self._running = False

    def _query_optional_int(self, command: str) -> int | None:
        try:
            return int(float(self.command(command)))
        except (ProviderError, TimeoutError, ValueError):
            return None

    def start(self) -> None:
        self.command("*CTLSTART")
        self._running = True

    def stop(self) -> None:
        if self._connected:
            self.command("*CTLSTOP")
        self._running = False

    def get_capture_state(self) -> str:
        return self.command("*GETCAPTURE")

    def measurements(self) -> dict[str, float | str]:
        values = self.command("*GETALLMEA").split(";")
        names = (
            "sigma_x",
            "sigma_y",
            "major",
            "minor",
            "effective_diameter",
            "ellipticity",
            "orientation",
            "divergence_x",
            "divergence_y",
            "exposure_ms",
        )
        return _parse_fields(names, values)

    def positions(self) -> dict[str, float | str]:
        values = self.command("*GETALLPOS").split(";")
        names = ("centroid_x", "centroid_y", "peak_x", "peak_y", "peak", "peak_ratio_x", "peak_ratio_y")
        return _parse_fields(names, values)

    def image_dimensions(self) -> tuple[int | None, int | None]:
        width = self._query_optional_int("*GETIMGWID")
        height = self._query_optional_int("*GETIMGHGT")
        return width, height

    def request_bmp_path(self, *, continuous: bool, timeout_s: float | None = None) -> Path:
        command = "*CTLBMPSAVE110" if continuous else "*CTLBMPSAVE"
        path_text = self.command(command, timeout_s=timeout_s).strip().strip('"')
        path = Path(path_text)
        deadline = time.monotonic() + (timeout_s or self.timeout_s)
        while not path.is_file() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not path.is_file():
            raise ProviderError(f"PC-Beamage returned an image path that is not available: {path_text}")
        return path

    def read_quantitative_frame(self, timeout_s: float) -> tuple[np.ndarray, dict[str, Any]]:
        """Read the vendor-requested BMP while explicitly marking its limits.

        The supplied client loads this BMP for display but does not establish that
        it is an untouched sensor matrix.  It is therefore valid for live preview,
        but not for formal quantitative evidence until compared with a documented
        numeric export on the physical system.
        """

        path = self.request_bmp_path(continuous=True, timeout_s=timeout_s)
        with Image.open(path) as image:
            mode = image.mode
            array = np.asarray(image.copy())
        if array.ndim == 3:
            # Preserve a deterministic intensity view without claiming raw data.
            array = np.mean(array[..., :3].astype(np.float64), axis=2)
        if array.ndim != 2:
            raise ProviderError(f"Unsupported PC-Beamage BMP shape {array.shape!r}.")
        maximum = float(np.iinfo(array.dtype).max) if np.issubdtype(array.dtype, np.integer) else float(np.max(array) or 1.0)
        try:
            measured = self.measurements()
        except Exception:
            measured = {}
        try:
            positions = self.positions()
        except Exception:
            positions = {}
        return np.asarray(array, dtype=np.float64), {
            "source_path": str(path),
            "source_format": "PC-Beamage named-pipe BMP",
            "bmp_mode": mode,
            "full_scale": maximum,
            "quantitative_valid": False,
            "measurement_scope": "LIVE_PREVIEW_ONLY",
            "warning": "Vendor BMP route is not yet validated as an untouched sensor matrix.",
            "measurements": measured,
            "positions": positions,
            "exposure_us": float(measured.get("exposure_ms", 0.0)) * 1000.0,
            "device_serial": self.identity.serial if self.identity else None,
            "pipe_path": self.pipe_path,
        }

    def configure(self, exposure_us: float, gain: float) -> None:
        del exposure_us, gain
        raise ProviderUnavailable(
            "Exposure and gain commands are not present in the supplied named-pipe example; control them in PC-Beamage."
        )


def _parse_fields(names: tuple[str, ...], values: list[str]) -> dict[str, float | str]:
    result: dict[str, float | str] = {}
    for index, name in enumerate(names):
        text = values[index].strip() if index < len(values) else ""
        try:
            result[name] = float(text)
        except ValueError:
            result[name] = text
    return result
