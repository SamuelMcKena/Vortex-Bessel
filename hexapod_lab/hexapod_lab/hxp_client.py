from __future__ import annotations

from dataclasses import dataclass
import math
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

    def request(
        self,
        command: str,
        *,
        raise_on_error: bool = True,
        response_timeout_s: float | None = None,
    ) -> tuple[int, str]:
        """Send one synchronous HXP API command and read through EndOfAPI.

        response_timeout_s overrides the normal socket timeout for commands
        that legitimately block until a long motion or homing operation has
        completed. A transport timeout or malformed reply invalidates the TCP
        stream, so the socket is closed rather than risking a stale reply being
        mistaken for the next command.
        """
        with self._lock:
            if self._sock is None:
                raise ConnectionError("HXP socket is not connected")
            sock = self._sock
            timeout = (
                self.config.timeout_s
                if response_timeout_s is None
                else float(response_timeout_s)
            )
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError(
                    "HXP response timeout must be finite and positive"
                )
            payload = command.encode("ascii", errors="strict")
            try:
                sock.settimeout(timeout)
                sock.sendall(payload)
                chunks: list[bytes] = []
                total = 0
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        raise ConnectionError(
                            "HXP closed the socket before EndOfAPI"
                        )
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > 8 * 1024 * 1024:
                        raise HXPProtocolError(
                            "HXP response exceeded 8 MiB safety limit"
                        )
                    if b",EndOfAPI" in b"".join(chunks[-2:]):
                        break

                raw = b"".join(chunks).decode("ascii", errors="replace")
                marker = raw.find(",EndOfAPI")
                if marker < 0:
                    raise HXPProtocolError(
                        f"Malformed HXP reply: {raw[:200]!r}"
                    )
                body = raw[:marker]
                if "," in body:
                    code_text, response = body.split(",", 1)
                else:
                    code_text, response = body, ""
                try:
                    code = int(code_text.strip())
                except ValueError as exc:
                    raise HXPProtocolError(
                        f"Malformed HXP error code: {code_text!r}"
                    ) from exc
                if code != 0 and raise_on_error:
                    raise HXPError(code, command, response)
                return code, response.strip()
            except socket.timeout as exc:
                self.close()
                raise TimeoutError(
                    f"HXP command timed out after {timeout:.1f} s: {command}"
                ) from exc
            except (ConnectionError, HXPProtocolError, OSError):
                self.close()
                raise
            finally:
                if self._sock is sock:
                    try:
                        sock.settimeout(self.config.timeout_s)
                    except OSError:
                        self.close()


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
        fields = [
            item.strip()
            for item in response.split(",")
            if item.strip() != ""
        ]
        if len(fields) < count:
            raise HXPProtocolError(
                f"Expected {count} numeric values, got {len(fields)}: {response!r}"
            )
        try:
            values = [float(v) for v in fields[:count]]
        except ValueError as exc:
            raise HXPProtocolError(
                f"Non-numeric HXP response: {response!r}"
            ) from exc
        if not all(math.isfinite(v) for v in values):
            raise HXPProtocolError(
                f"Non-finite HXP response: {response!r}"
            )
        return values

    @staticmethod
    def _finite_values(
        label: str,
        values: Iterable[float],
    ) -> tuple[float, ...]:
        out = tuple(float(v) for v in values)
        if not all(math.isfinite(v) for v in out):
            raise ValueError(f"{label} values must be finite")
        return out

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

    def positioner_user_travel_limits(
        self,
        positioner: str,
    ) -> tuple[float, float]:
        """Read the configured HXP user travel limits for one positioner.

        This is deliberately read-only. On the lab controller this API applies
        to physical positioners such as HEXAPOD.1 through HEXAPOD.6. The
        virtual Cartesian channels HEXAPOD.X through HEXAPOD.W do not expose
        PositionerUserTravelLimitsGet.
        """
        name = str(positioner).strip()
        if not name or any(char in name for char in ",()"):
            raise ValueError("invalid HXP positioner name")
        _, response = self.poll.request(
            f"PositionerUserTravelLimitsGet({name},double *,double *)"
        )
        minimum, maximum = self._float_list(response, 2)
        if minimum >= maximum:
            raise HXPProtocolError(
                f"Invalid travel limits for {name}: {minimum}, {maximum}"
            )
        return minimum, maximum

    def positioner_current_position(self, positioner: str) -> float:
        """Read one actuator/strut position, e.g. ``HEXAPOD.1``."""
        name = str(positioner).strip()
        if not name or any(char in name for char in ",()"):
            raise ValueError("invalid HXP positioner name")
        _, response = self.poll.request(
            f"GroupPositionCurrentGet({name},double *)"
        )
        return self._float_list(response, 1)[0]

    def move_absolute(
        self,
        pose: Pose6D,
        group: str = "HEXAPOD",
        coordinate_system: str = "Work",
    ) -> None:
        values = self._finite_values("absolute pose", pose.as_tuple())
        args = ",".join(f"{v:.12g}" for v in values)
        self.control.request(
            f"HexapodMoveAbsolute({group},{coordinate_system},{args})",
            response_timeout_s=max(self.control.config.timeout_s, 180.0),
        )

    def move_incremental(
        self,
        delta: Pose6D,
        group: str = "HEXAPOD",
        coordinate_system: str = "Work",
    ) -> None:
        values = self._finite_values("incremental pose", delta.as_tuple())
        args = ",".join(f"{v:.12g}" for v in values)
        self.control.request(
            f"HexapodMoveIncremental({group},{coordinate_system},{args})",
            response_timeout_s=max(self.control.config.timeout_s, 180.0),
        )

    def line_incremental_control_limits(
        self,
        dx_mm: float,
        dy_mm: float,
        dz_mm: float,
        *,
        group: str = "HEXAPOD",
        coordinate_system: str = "Work",
    ) -> tuple[float, float]:
        """Ask the HXP to preflight an incremental Line trajectory.

        Returns ``(maximum_velocity_carriage_mm_s, trajectory_percent)`` from
        ``HexapodMoveIncrementalControlLimitGet``. This is the authoritative
        controller-side feasibility check for real translation-only Line moves.
        """
        dx, dy, dz = self._finite_values(
            "Line displacement",
            (dx_mm, dy_mm, dz_mm),
        )
        if max(abs(dx), abs(dy), abs(dz)) <= 1e-15:
            raise ValueError("Line trajectory must have a non-zero displacement")
        _, response = self.poll.request(
            "HexapodMoveIncrementalControlLimitGet("
            f"{group},{coordinate_system},Line,"
            f"{dx:.12g},{dy:.12g},{dz:.12g},double *,double *)"
        )
        maximum_velocity, trajectory_fraction = self._float_list(response, 2)
        if maximum_velocity <= 0:
            raise HXPProtocolError(
                f"HXP returned invalid maximum Line velocity {maximum_velocity!r}"
            )
        if trajectory_fraction < -1e-9 or trajectory_fraction > 1.000001:
            raise HXPProtocolError(
                "HXP returned invalid executable-trajectory fraction "
                f"{trajectory_fraction!r}; expected 0..1"
            )
        return float(maximum_velocity), float(
            max(0.0, min(1.0, trajectory_fraction))
        )

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
        dx, dy, dz, velocity = self._finite_values(
            "Line move",
            (dx_mm, dy_mm, dz_mm, velocity_mm_s),
        )
        if max(abs(dx), abs(dy), abs(dz)) <= 1e-15:
            raise ValueError("Line move must have a non-zero displacement")
        if velocity <= 0:
            raise ValueError("target velocity must be > 0 mm/s")
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        expected_motion_s = distance / velocity
        response_timeout = max(
            self.control.config.timeout_s,
            30.0 + 2.0 * expected_motion_s,
        )
        self.control.request(
            "HexapodMoveIncrementalControlWithTargetVelocity("
            f"{group},{coordinate_system},Line,"
            f"{dx:.12g},{dy:.12g},{dz:.12g},{velocity:.12g})",
            response_timeout_s=response_timeout,
        )

    def abort(self, group: str = "HEXAPOD") -> None:
        self.io.request(f"GroupMoveAbort({group})")

    def initialize(self, group: str = "HEXAPOD") -> None:
        self.control.request(
            f"GroupInitialize({group})",
            response_timeout_s=max(self.control.config.timeout_s, 60.0),
        )

    def home(self, group: str = "HEXAPOD") -> None:
        self.control.request(
            f"GroupHomeSearch({group})",
            response_timeout_s=max(self.control.config.timeout_s, 300.0),
        )

    def coordinate_system_get(
        self,
        coordinate_system: str,
        group: str = "HEXAPOD",
    ) -> Pose6D:
        system = str(coordinate_system).strip()
        if system not in {"Work", "Tool"}:
            raise ValueError("coordinate system must be Work or Tool")
        _, response = self.poll.request(
            f"HexapodCoordinateSystemGet({group},{system},"
            f"{self._outputs('double', 6)})"
        )
        return Pose6D.from_iterable(self._float_list(response, 6))

    def require_ready_for_motion(
        self,
        group: str = "HEXAPOD",
    ) -> tuple[int, str]:
        """Require a referenced HXP group before issuing a normal move.

        Newport documents states 11 (Ready state from homing) and 12 (Ready
        state from motion) as the normal referenced ready states.
        """
        status = self.group_status(group)
        try:
            status_text = self.group_status_text(status)
        except Exception:
            status_text = f"status {status}"
        if status not in (11, 12):
            raise RuntimeError(
                f"HXP group {group} is not ready for motion: "
                f"{status} ({status_text})"
            )
        return status, status_text

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
        value = float(response.split(",", 1)[0].strip())
        if not math.isfinite(value):
            raise HXPProtocolError(
                f"Non-finite analogue readback for {gpio_name}: {response!r}"
            )
        return value

    def analog_set(self, gpio_name: str, value: float) -> None:
        """Set an HXP analogue output.

        This exists because the legacy lab TCL uses GPIOAnalogSet for the
        power/attenuation path. No channel or calibration is assumed here.
        """
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("analogue output value must be finite")
        self.io.request(
            f"GPIOAnalogSet({gpio_name},{numeric:.12g})"
        )

    def gathering_configure_cartesian_current(self, group: str = "HEXAPOD") -> None:
        names = [f"{group}.{axis}.CurrentPosition" for axis in "XYZUVW"]
        self.control.request("GatheringConfigurationSet(" + ",".join(names) + ")")

    def gathering_run(self, samples: int, divisor: int) -> None:
        self.control.request(f"GatheringRun({int(samples)},{int(divisor)})")

    def gathering_stop(self) -> None:
        self.control.request("GatheringStop()")
