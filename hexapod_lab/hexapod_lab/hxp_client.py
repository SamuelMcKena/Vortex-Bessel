from __future__ import annotations

from dataclasses import dataclass
import socket
import threading
from typing import Iterable

from .types import Pose6D


class HXPError(RuntimeError):
    def __init__(self, code: int, command: str, response: str = "") -> None:
        super().__init__(f"HXP error {code} for {command}: {response}")
        self.code = int(code)
        self.command = command
        self.response = response


class HXPProtocolError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class HXPConnectionConfig:
    host: str
    port: int = 5001
    timeout_s: float = 2.0


class HXPConnection:
    """Minimal HXP TCP/API connection.

    HXP API replies are terminated by `,EndOfAPI`. A connection is intentionally
    single-purpose because HXP sockets are blocking; higher-level code opens
    independent control/poll/I/O sockets so position polling can continue while
    a motion command is in flight.
    """

    def __init__(self, config: HXPConnectionConfig) -> None:
        self.config = config
        self._sock: socket.socket | None = None
        self._lock = threading.RLock()

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def connect(self) -> None:
        with self._lock:
            self.close()
            sock = socket.create_connection((self.config.host, self.config.port), timeout=self.config.timeout_s)
            sock.settimeout(self.config.timeout_s)
            self._sock = sock

    def close(self) -> None:
        with self._lock:
            sock, self._sock = self._sock, None
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass

    def request(self, command: str, *, raise_on_error: bool = True) -> tuple[int, str]:
        with self._lock:
            if self._sock is None:
                raise ConnectionError("HXP socket is not connected")
            payload = command.encode("ascii", errors="strict")
            self._sock.sendall(payload)
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = self._sock.recv(4096)
                if not chunk:
                    raise ConnectionError("HXP closed the socket before EndOfAPI")
                chunks.append(chunk)
                total += len(chunk)
                if total > 8 * 1024 * 1024:
                    raise HXPProtocolError("HXP response exceeded 8 MiB safety limit")
                if b",EndOfAPI" in b"".join(chunks[-2:]):
                    break

            raw = b"".join(chunks).decode("ascii", errors="replace")
            marker = raw.find(",EndOfAPI")
            if marker < 0:
                raise HXPProtocolError(f"Malformed HXP reply: {raw[:200]!r}")
            body = raw[:marker]
            if "," in body:
                code_text, response = body.split(",", 1)
            else:
                code_text, response = body, ""
            try:
                code = int(code_text.strip())
            except ValueError as exc:
                raise HXPProtocolError(f"Malformed HXP error code: {code_text!r}") from exc
            if code != 0 and raise_on_error:
                raise HXPError(code, command, response)
            return code, response.strip()


class HXPClient:
    """Small, auditable subset of the Newport HXP API needed by the GUI."""

    def __init__(self, host: str, port: int = 5001, timeout_s: float = 2.0) -> None:
        cfg = HXPConnectionConfig(host=host, port=int(port), timeout_s=float(timeout_s))
        self.control = HXPConnection(cfg)
        self.poll = HXPConnection(cfg)
        self.io = HXPConnection(cfg)

    def connect(self) -> None:
        opened: list[HXPConnection] = []
        try:
            for conn in (self.control, self.poll, self.io):
                conn.connect()
                opened.append(conn)
        except Exception:
            for conn in reversed(opened):
                conn.close()
            raise

    def close(self) -> None:
        for conn in (self.io, self.poll, self.control):
            conn.close()

    @property
    def connected(self) -> bool:
        return self.control.connected and self.poll.connected and self.io.connected

    @staticmethod
    def _float_list(response: str, count: int) -> list[float]:
        fields = [item.strip() for item in response.split(",") if item.strip() != ""]
        if len(fields) < count:
            raise HXPProtocolError(f"Expected {count} numeric values, got {len(fields)}: {response!r}")
        try:
            return [float(v) for v in fields[:count]]
        except ValueError as exc:
            raise HXPProtocolError(f"Non-numeric HXP response: {response!r}") from exc

    @staticmethod
    def _outputs(type_name: str, count: int) -> str:
        return ",".join([f"{type_name} *"] * int(count))

    def firmware_version(self) -> str:
        _, response = self.poll.request("FirmwareVersionGet(char *)")
        return response

    def error_string(self, code: int) -> str:
        _, response = self.poll.request(f"ErrorStringGet({int(code)},char *)", raise_on_error=False)
        return response

    def current_pose(self, group: str = "HEXAPOD") -> Pose6D:
        _, response = self.poll.request(f"GroupPositionCurrentGet({group},{self._outputs('double', 6)})")
        return Pose6D.from_iterable(self._float_list(response, 6))

    def setpoint_pose(self, group: str = "HEXAPOD") -> Pose6D:
        _, response = self.poll.request(f"GroupPositionSetpointGet({group},{self._outputs('double', 6)})")
        return Pose6D.from_iterable(self._float_list(response, 6))

    def target_pose(self, group: str = "HEXAPOD") -> Pose6D:
        _, response = self.poll.request(f"GroupPositionTargetGet({group},{self._outputs('double', 6)})")
        return Pose6D.from_iterable(self._float_list(response, 6))

    def group_status(self, group: str = "HEXAPOD") -> int:
        _, response = self.poll.request(f"GroupStatusGet({group},int *)")
        fields = [x.strip() for x in response.split(",") if x.strip()]
        if not fields:
            raise HXPProtocolError("GroupStatusGet returned no status code")
        return int(float(fields[0]))

    def group_status_text(self, status_code: int) -> str:
        _, response = self.poll.request(f"GroupStatusStringGet({int(status_code)},char *)")
        return response

    def move_absolute(self, pose: Pose6D, group: str = "HEXAPOD", coordinate_system: str = "Work") -> None:
        args = ",".join(f"{v:.12g}" for v in pose.as_tuple())
        self.control.request(f"HexapodMoveAbsolute({group},{coordinate_system},{args})")

    def move_incremental(self, delta: Pose6D, group: str = "HEXAPOD", coordinate_system: str = "Work") -> None:
        args = ",".join(f"{v:.12g}" for v in delta.as_tuple())
        self.control.request(f"HexapodMoveIncremental({group},{coordinate_system},{args})")

    def move_line_incremental_with_target_velocity(
        self,
        dx_mm: float,
        dy_mm: float,
        dz_mm: float,
        velocity_mm_s: float,
        *,
        group: str = "HEXAPOD",
        coordinate_system: str = "Work",
    ) -> None:
        """Execute the HXP line move used by the legacy lab TCL scripts.

        Legacy form:
        HexapodMoveIncrementalControlWithTargetVelocity
            HEXAPOD Work Line dX dY dZ velocity
        """
        velocity = float(velocity_mm_s)
        if velocity <= 0:
            raise ValueError("target velocity must be > 0 mm/s")
        self.control.request(
            "HexapodMoveIncrementalControlWithTargetVelocity("
            f"{group},{coordinate_system},Line,"
            f"{float(dx_mm):.12g},{float(dy_mm):.12g},"
            f"{float(dz_mm):.12g},{velocity:.12g})"
        )

    def abort(self, group: str = "HEXAPOD") -> None:
        self.io.request(f"GroupMoveAbort({group})")

    def initialize(self, group: str = "HEXAPOD") -> None:
        self.control.request(f"GroupInitialize({group})")

    def home(self, group: str = "HEXAPOD") -> None:
        self.control.request(f"GroupHomeSearch({group})")

    def digital_get(self, gpio_name: str) -> int:
        _, response = self.io.request(f"GPIODigitalGet({gpio_name},unsigned short *)")
        return int(float(response.split(",", 1)[0].strip()))

    def digital_set(self, gpio_name: str, mask: int, value: int) -> None:
        self.io.request(f"GPIODigitalSet({gpio_name},{int(mask)},{int(value)})")

    def analog_get(self, gpio_name: str) -> float:
        """Read an HXP analogue GPIO value.

        The LabVIEW v3 front panel reads GPIO2.DAC1 with GPIOAnalogGet.
        """
        _, response = self.io.request(
            f"GPIOAnalogGet({gpio_name},double *)"
        )
        return float(response.split(",", 1)[0].strip())

    def analog_set(self, gpio_name: str, value: float) -> None:
        """Set an HXP analogue output.

        This exists because the legacy lab TCL uses GPIOAnalogSet for the
        power/attenuation path. No channel or calibration is assumed here.
        """
        self.io.request(
            f"GPIOAnalogSet({gpio_name},{float(value):.12g})"
        )

    def gathering_configure_cartesian_current(self, group: str = "HEXAPOD") -> None:
        names = [f"{group}.{axis}.CurrentPosition" for axis in "XYZUVW"]
        self.control.request("GatheringConfigurationSet(" + ",".join(names) + ")")

    def gathering_run(self, samples: int, divisor: int) -> None:
        self.control.request(f"GatheringRun({int(samples)},{int(divisor)})")

    def gathering_stop(self) -> None:
        self.control.request("GatheringStop()")
