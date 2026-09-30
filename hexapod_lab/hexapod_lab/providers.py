from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import threading
import time
from typing import Callable

from .hxp_client import HXPClient
from .types import AttenuatorSnapshot, HexapodSnapshot, LaserSnapshot, MotionState, Pose6D


class HexapodProvider(ABC):
    name: str = "provider"

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def snapshot(self) -> HexapodSnapshot: ...

    @abstractmethod
    def move_absolute(self, pose: Pose6D) -> None: ...

    @abstractmethod
    def move_incremental(self, delta: Pose6D) -> None: ...

    def move_line_incremental_with_target_velocity(
        self,
        dx_mm: float,
        dy_mm: float,
        dz_mm: float,
        velocity_mm_s: float,
    ) -> None:
        """Translation-only line move at an explicit target velocity."""
        raise NotImplementedError

    @abstractmethod
    def abort(self) -> None: ...

    def initialize(self) -> None:
        raise NotImplementedError

    def home(self) -> None:
        raise NotImplementedError

    def tick(self, dt_s: float) -> None:
        """Advance providers that need a host-side simulation clock."""

    def is_busy(self) -> bool:
        return self.snapshot().state == MotionState.MOVING


class VirtualHexapodProvider(HexapodProvider):
    name = "virtual"

    def __init__(
        self,
        *,
        linear_speed_mm_s: float = 8.0,
        angular_speed_deg_s: float = 8.0,
        motion_validator: Callable[[Pose6D, Pose6D], None] | None = None,
    ) -> None:
        self.linear_speed_mm_s = float(linear_speed_mm_s)
        self.angular_speed_deg_s = float(angular_speed_deg_s)
        self._motion_validator = motion_validator
        self._connected = False
        self._actual = Pose6D()
        self._start = Pose6D()
        self._target = Pose6D()
        self._elapsed = 0.0
        self._duration = 0.0
        self._state = MotionState.DISCONNECTED
        self._lock = threading.RLock()

    def connect(self) -> None:
        with self._lock:
            self._connected = True
            self._state = MotionState.IDLE

    def disconnect(self) -> None:
        with self._lock:
            self._connected = False
            self._state = MotionState.DISCONNECTED

    def _plan_to(self, target: Pose6D) -> None:
        if not self._connected:
            raise ConnectionError("virtual hexapod is not connected")
        if self._motion_validator is not None:
            self._motion_validator(self._actual, target)
        linear = max(abs(a - b) for a, b in zip(self._actual.as_tuple()[:3], target.as_tuple()[:3]))
        angular = max(abs(a - b) for a, b in zip(self._actual.as_tuple()[3:], target.as_tuple()[3:]))
        t_linear = linear / max(self.linear_speed_mm_s, 1e-9)
        t_angular = angular / max(self.angular_speed_deg_s, 1e-9)
        self._start = self._actual
        self._target = target
        self._elapsed = 0.0
        self._duration = max(t_linear, t_angular, 0.05)
        self._state = MotionState.MOVING

    def move_absolute(self, pose: Pose6D) -> None:
        with self._lock:
            self._plan_to(pose)

    def move_incremental(self, delta: Pose6D) -> None:
        with self._lock:
            self._plan_to(
                self._target.plus(delta)
                if self._state == MotionState.MOVING
                else self._actual.plus(delta)
            )

    def move_line_incremental_with_target_velocity(
        self,
        dx_mm: float,
        dy_mm: float,
        dz_mm: float,
        velocity_mm_s: float,
    ) -> None:
        with self._lock:
            if not self._connected:
                raise ConnectionError("virtual hexapod is not connected")
            velocity = float(velocity_mm_s)
            if velocity <= 0:
                raise ValueError("target velocity must be > 0 mm/s")
            base = (
                self._target
                if self._state == MotionState.MOVING
                else self._actual
            )
            delta = Pose6D(
                x=float(dx_mm),
                y=float(dy_mm),
                z=float(dz_mm),
            )
            target = base.plus(delta)
            if self._motion_validator is not None:
                self._motion_validator(self._actual, target)
            distance = math.sqrt(
                float(dx_mm) ** 2
                + float(dy_mm) ** 2
                + float(dz_mm) ** 2
            )
            self._start = self._actual
            self._target = target
            self._elapsed = 0.0
            self._duration = max(distance / velocity, 0.05)
            self._state = MotionState.MOVING

    def abort(self) -> None:
        with self._lock:
            self._target = self._actual
            self._state = MotionState.ABORTED if self._connected else MotionState.DISCONNECTED

    def initialize(self) -> None:
        if not self._connected:
            raise ConnectionError("virtual hexapod is not connected")

    def home(self) -> None:
        self.move_absolute(Pose6D())

    def tick(self, dt_s: float) -> None:
        with self._lock:
            if not self._connected or self._state != MotionState.MOVING:
                return
            self._elapsed += max(0.0, float(dt_s))
            f = min(1.0, self._elapsed / max(self._duration, 1e-9))
            eased = f * f * (3.0 - 2.0 * f)
            self._actual = self._start.lerp(self._target, eased)
            if f >= 1.0:
                self._actual = self._target
                self._state = MotionState.IDLE

    def is_busy(self) -> bool:
        with self._lock:
            return self._state == MotionState.MOVING

    def snapshot(self) -> HexapodSnapshot:
        with self._lock:
            return HexapodSnapshot(
                timestamp_s=time.time(),
                actual=self._actual,
                setpoint=self._actual,
                target=self._target,
                state=self._state,
                connected=self._connected,
                provider=self.name,
                status_text="VIRTUAL" if self._connected else "DISCONNECTED",
            )


@dataclass(frozen=True, slots=True)
class HXPProviderConfig:
    host: str
    port: int = 5001
    group: str = "HEXAPOD"
    coordinate_system: str = "Work"
    timeout_s: float = 2.0


class HXPProvider(HexapodProvider):
    name = "hxp-real"

    def __init__(self, config: HXPProviderConfig) -> None:
        self.config = config
        self.client = HXPClient(
            config.host,
            config.port,
            config.timeout_s,
        )
        self._connected = False
        self._move_thread: threading.Thread | None = None
        self._move_error: BaseException | None = None
        self._abort_requested = False
        self._last_snapshot: HexapodSnapshot | None = None
        self._lock = threading.RLock()

    def connect(self) -> None:
        try:
            self.client.connect()
            self._connected = True
            self._move_error = None
            self._abort_requested = False
            self._last_snapshot = self.snapshot()
        except Exception:
            self._connected = False
            self.client.close()
            raise

    def disconnect(self) -> None:
        with self._lock:
            self._connected = False
            self._abort_requested = True
            self.client.close()

    def _start_blocking_call(
        self,
        fn: Callable[[], None],
        *,
        require_ready: bool = True,
    ) -> None:
        with self._lock:
            if not self._connected or not self.client.connected:
                raise ConnectionError("HXP is not connected")
            if self._move_thread is not None and self._move_thread.is_alive():
                raise RuntimeError(
                    "a blocking HXP motion command is already in progress"
                )
            if require_ready:
                self.client.require_ready_for_motion(self.config.group)
            self._move_error = None
            self._abort_requested = False

            def worker() -> None:
                try:
                    fn()
                except BaseException as exc:
                    with self._lock:
                        if not self._abort_requested:
                            self._move_error = exc

            self._move_thread = threading.Thread(
                target=worker,
                name="hxp-motion",
                daemon=True,
            )
            self._move_thread.start()

    def move_absolute(self, pose: Pose6D) -> None:
        self._start_blocking_call(
            lambda: self.client.move_absolute(
                pose,
                self.config.group,
                self.config.coordinate_system,
            )
        )

    def move_incremental(self, delta: Pose6D) -> None:
        self._start_blocking_call(
            lambda: self.client.move_incremental(
                delta,
                self.config.group,
                self.config.coordinate_system,
            )
        )

    def move_line_incremental_with_target_velocity(
        self,
        dx_mm: float,
        dy_mm: float,
        dz_mm: float,
        velocity_mm_s: float,
    ) -> None:
        self._start_blocking_call(
            lambda: self.client.move_line_incremental_with_target_velocity(
                dx_mm,
                dy_mm,
                dz_mm,
                velocity_mm_s,
                group=self.config.group,
                coordinate_system=self.config.coordinate_system,
            )
        )

    def abort(self) -> None:
        with self._lock:
            self._abort_requested = True
        try:
            self.client.abort(self.config.group)
        finally:
            # STOP/abort is also the acknowledgement path for a failed GUI
            # motion worker; retain the failure in logs/UI before this point,
            # but do not leave the provider permanently latched.
            with self._lock:
                self._move_error = None

    def initialize(self) -> None:
        self._start_blocking_call(
            lambda: self.client.initialize(self.config.group),
            require_ready=False,
        )

    def home(self) -> None:
        self._start_blocking_call(
            lambda: self.client.home(self.config.group),
            require_ready=False,
        )

    def is_busy(self) -> bool:
        with self._lock:
            if self._move_error is not None:
                exc, self._move_error = self._move_error, None
                raise RuntimeError("HXP motion command failed") from exc
            if self._move_thread is not None and self._move_thread.is_alive():
                return True
            return bool(
                self._last_snapshot is not None
                and self._last_snapshot.state == MotionState.MOVING
            )

    def snapshot(self) -> HexapodSnapshot:
        if not self._connected or not self.client.connected:
            return HexapodSnapshot(
                timestamp_s=time.time(),
                actual=(
                    self._last_snapshot.actual
                    if self._last_snapshot is not None
                    else Pose6D()
                ),
                state=MotionState.DISCONNECTED,
                connected=False,
                provider=self.name,
                status_text="DISCONNECTED",
            )

        actual = self.client.current_pose(self.config.group)
        setpoint = self.client.setpoint_pose(self.config.group)
        target = self.client.target_pose(self.config.group)
        status = self.client.group_status(self.config.group)
        try:
            status_text = self.client.group_status_text(status)
        except Exception:
            status_text = f"status {status}"

        moving_thread = (
            self._move_thread is not None
            and self._move_thread.is_alive()
        )
        with self._lock:
            pending_error = self._move_error

        if moving_thread:
            state = MotionState.MOVING
        elif pending_error is not None:
            state = MotionState.FAULT
            status_text = (
                f"{status_text} • last GUI motion failed: "
                f"{type(pending_error).__name__}: {pending_error}"
            )
        elif status in (11, 12):
            state = MotionState.IDLE
        else:
            # A connected controller is not automatically motion-ready.
            # Treat unreferenced, disabled and error states as non-ready.
            state = MotionState.FAULT

        snap = HexapodSnapshot(
            timestamp_s=time.time(),
            actual=actual,
            setpoint=setpoint,
            target=target,
            state=state,
            status_code=status,
            status_text=status_text,
            connected=True,
            provider=self.name,
        )
        self._last_snapshot = snap
        return snap


class LaserGateProvider(ABC):
    name: str = "laser"

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def set_gate(self, enabled: bool) -> None: ...

    @abstractmethod
    def snapshot(self) -> LaserSnapshot: ...

    def safe_off(self) -> None:
        """Request the safe/CLOSED state and surface any failure to the caller."""
        self.set_gate(False)


class VirtualLaserGate(LaserGateProvider):
    name = "virtual-lx13"

    def __init__(
        self,
        *,
        profile_label: str = "Generic virtual LX13",
        gpio_name: str = "",
        mask: int = 0,
        open_value: int | None = None,
        closed_value: int | None = None,
    ) -> None:
        self._connected = False
        self._enabled = False
        self.profile_label = str(profile_label)
        self.gpio_name = str(gpio_name)
        self.mask = int(mask)
        self.open_value = open_value
        self.closed_value = closed_value

    def configure_profile(
        self,
        *,
        profile_label: str,
        gpio_name: str,
        mask: int,
        open_value: int | None,
        closed_value: int | None,
    ) -> None:
        # Mock-only metadata. Logical OPEN/CLOSED remains deterministic even
        # when a historical profile has unresolved physical polarity.
        self.profile_label = str(profile_label)
        self.gpio_name = str(gpio_name)
        self.mask = int(mask)
        self.open_value = open_value
        self.closed_value = closed_value

    def connect(self) -> None:
        self._connected = True
        self._enabled = False

    def disconnect(self) -> None:
        self._enabled = False
        self._connected = False

    def set_gate(self, enabled: bool) -> None:
        if not self._connected:
            raise ConnectionError("virtual laser gate is not connected")
        self._enabled = bool(enabled)

    def snapshot(self) -> LaserSnapshot:
        raw_value = self.open_value if self._enabled else self.closed_value
        return LaserSnapshot(
            timestamp_s=time.time(),
            gate_enabled=self._enabled,
            connected=self._connected,
            provider=self.name,
            connector_name=f"LX13 mock • {self.profile_label}",
            readback_known=True,
            metadata={
                "profile_label": self.profile_label,
                "gpio_name": self.gpio_name,
                "mask": self.mask,
                "raw_value": raw_value,
                "polarity_known": (
                    self.open_value is not None
                    and self.closed_value is not None
                ),
            },
        )


@dataclass(frozen=True, slots=True)
class HXPDigitalLaserConfig:
    gpio_name: str
    mask: int
    enabled_value: int
    disabled_value: int
    connector_name: str = "PHAROS LX13"
    wiring_verified: bool = False


class HXPDigitalLaserGate(LaserGateProvider):
    """HXP digital-output gate for the external PHAROS LX13 interface.

    No pinout, active level, or electrical compatibility is assumed. Construction
    requires an explicit, user-supplied verified mapping. This class controls a
    digital output only; it does not bypass the laser's physical safety chain.
    """

    name = "hxp-lx13-real"

    def __init__(self, client: HXPClient, config: HXPDigitalLaserConfig) -> None:
        self.client = client
        self.config = config
        self._connected = False
        self._enabled = False
        self._last_raw_readback: int | None = None
        self._readback_known = False

    def connect(self) -> None:
        if not self.config.wiring_verified:
            raise RuntimeError("LX13/HXP wiring is not marked verified in hardware configuration")
        if not self.config.gpio_name or self.config.mask <= 0:
            raise ValueError("real LX13 gating requires a GPIO name and non-zero mask")
        if self.config.enabled_value == self.config.disabled_value:
            raise ValueError(
                "Pockels OPEN and CLOSED digital values are identical; "
                "verify the LX13/HXP mapping before arming"
            )
        if not self.client.connected:
            raise ConnectionError("HXP must be connected before real LX13 gating can be enabled")
        self._connected = True
        try:
            # Fail closed on connection and verify the HXP output register if
            # the controller permits readback.
            self.set_gate(False)
        except Exception:
            self._connected = False
            raise

    def disconnect(self) -> None:
        try:
            if self._connected:
                self.safe_off()
        finally:
            self._connected = False

    def set_gate(self, enabled: bool) -> None:
        if not self._connected:
            raise ConnectionError("real LX13 gate is not connected/armed")
        value = self.config.enabled_value if enabled else self.config.disabled_value
        self.client.digital_set(self.config.gpio_name, self.config.mask, value)

        self._readback_known = False
        self._last_raw_readback = None
        try:
            raw = self.client.digital_get(self.config.gpio_name)
            self._last_raw_readback = int(raw)
            self._readback_known = True
            if (raw & self.config.mask) != (value & self.config.mask):
                # An OPEN-command mismatch is treated as an uncertain beam
                # state. Immediately request the configured CLOSED value before
                # surfacing the error.
                if enabled:
                    try:
                        self.client.digital_set(
                            self.config.gpio_name,
                            self.config.mask,
                            self.config.disabled_value,
                        )
                    finally:
                        self._enabled = False
                raise RuntimeError(
                    "HXP digital output readback does not match the requested "
                    f"Pockels state: raw={raw}, mask={self.config.mask}, "
                    f"requested={value}; CLOSED was requested"
                )
        except RuntimeError:
            raise
        except Exception:
            # Some installations may not expose useful DO readback. The command
            # state remains available but is labelled as such in the GUI.
            self._readback_known = False

        self._enabled = bool(enabled)

    def snapshot(self) -> LaserSnapshot:
        return LaserSnapshot(
            timestamp_s=time.time(),
            gate_enabled=self._enabled,
            connected=self._connected,
            provider=self.name,
            connector_name=self.config.connector_name,
            readback_known=self._readback_known,
            metadata={
                "gpio_name": self.config.gpio_name,
                "mask": self.config.mask,
                "raw_readback": self._last_raw_readback,
                "open_value": self.config.enabled_value,
                "closed_value": self.config.disabled_value,
            },
        )


class AttenuatorProvider(ABC):
    """Normalized optical-attenuator interface.

    The GUI speaks in requested transmission percent. A device-specific real
    driver may later translate that setpoint through its calibration curve to
    motor angle, voltage, or another native quantity.
    """

    name: str = "attenuator"

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def set_transmission_percent(self, value: float) -> None: ...

    @abstractmethod
    def snapshot(self) -> AttenuatorSnapshot: ...


class VirtualAttenuatorProvider(AttenuatorProvider):
    name = "virtual-attenuator"

    def __init__(self, initial_transmission_percent: float = 0.0) -> None:
        self._connected = False
        self._transmission_percent = self._clamp(initial_transmission_percent)

    @staticmethod
    def _clamp(value: float) -> float:
        value = float(value)
        if not 0.0 <= value <= 100.0:
            raise ValueError("attenuator transmission must be between 0 and 100 %")
        return value

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def set_transmission_percent(self, value: float) -> None:
        if not self._connected:
            raise ConnectionError("virtual attenuator is not connected")
        self._transmission_percent = self._clamp(value)

    def snapshot(self) -> AttenuatorSnapshot:
        return AttenuatorSnapshot(
            timestamp_s=time.time(),
            transmission_percent=self._transmission_percent,
            connected=self._connected,
            provider=self.name,
            readback_known=True,
            device_name="Virtual attenuator",
        )


@dataclass(frozen=True, slots=True)
class HXPAnalogAttenuatorConfig:
    gpio_name: str = "GPIO2.DAC1"
    raw_min: float = 0.0
    raw_max: float = 10.0
    transmission_min_percent: float = 0.0
    transmission_max_percent: float = 100.0
    inverted: bool = True
    readback_tolerance_raw: float = 0.05
    device_name: str = "HXP GPIO2.DAC1 attenuator"


class HXPAnalogAttenuatorProvider(AttenuatorProvider):
    """Real attenuator path confirmed on the current lab HXP.

    Current lab confirmation establishes ``GPIO2.DAC1`` as the attenuator
    command with inverse polarity: raw 0 = 100 % transmission and raw 10 = 0 %
    transmission. The GUI therefore uses an inverted linear 0–10 DAC mapping.
    """

    name = "hxp-analog-attenuator"

    def __init__(self, client: HXPClient, config: HXPAnalogAttenuatorConfig | None = None) -> None:
        self.client = client
        self.config = config or HXPAnalogAttenuatorConfig()
        self._connected = False
        self._transmission_percent = 0.0
        self._raw_readback: float | None = None
        self._readback_known = False

    def _raw_to_percent(self, raw: float) -> float:
        c = self.config
        raw_span = c.raw_max - c.raw_min
        pct_span = c.transmission_max_percent - c.transmission_min_percent
        raw = float(raw)
        if not math.isfinite(raw):
            raise ValueError("attenuator readback must be finite")
        if raw_span <= 0 or pct_span <= 0:
            raise ValueError("invalid attenuator calibration span")
        tolerance = max(0.0, float(c.readback_tolerance_raw))
        if raw < c.raw_min - tolerance or raw > c.raw_max + tolerance:
            raise RuntimeError(
                f"attenuator raw readback {raw:.4f} is outside calibrated "
                f"range [{c.raw_min:.4f}, {c.raw_max:.4f}]"
            )
        raw = max(c.raw_min, min(c.raw_max, raw))
        fraction = (raw - c.raw_min) / raw_span
        if c.inverted:
            pct = c.transmission_max_percent - fraction * pct_span
        else:
            pct = c.transmission_min_percent + fraction * pct_span
        return max(
            c.transmission_min_percent,
            min(c.transmission_max_percent, pct),
        )

    def _percent_to_raw(self, percent: float) -> float:
        c = self.config
        percent = float(percent)
        if not math.isfinite(percent):
            raise ValueError("attenuator transmission must be finite")
        if not c.transmission_min_percent <= percent <= c.transmission_max_percent:
            raise ValueError(
                f"attenuator transmission must be between {c.transmission_min_percent:g} "
                f"and {c.transmission_max_percent:g} %"
            )
        pct_span = c.transmission_max_percent - c.transmission_min_percent
        if pct_span <= 0:
            raise ValueError("invalid attenuator transmission calibration span")
        if c.inverted:
            fraction = (c.transmission_max_percent - percent) / pct_span
        else:
            fraction = (percent - c.transmission_min_percent) / pct_span
        return c.raw_min + fraction * (c.raw_max - c.raw_min)

    def connect(self) -> None:
        if not self.client.connected:
            raise ConnectionError("HXP must be connected before the real attenuator")
        raw = float(self.client.analog_get(self.config.gpio_name))
        self._raw_readback = raw
        self._transmission_percent = self._raw_to_percent(raw)
        self._readback_known = True
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def set_transmission_percent(self, value: float) -> None:
        if not self._connected:
            raise ConnectionError("real HXP attenuator is not connected")
        raw = self._percent_to_raw(float(value))
        self.client.analog_set(self.config.gpio_name, raw)
        self._readback_known = False
        try:
            readback = float(self.client.analog_get(self.config.gpio_name))
            self._raw_readback = readback
            self._readback_known = True
            if abs(readback - raw) > self.config.readback_tolerance_raw:
                raise RuntimeError(
                    f"attenuator DAC readback {readback:.4f} does not match "
                    f"requested {raw:.4f}"
                )
            self._transmission_percent = self._raw_to_percent(readback)
        except RuntimeError:
            raise
        except Exception:
            self._raw_readback = raw
            self._transmission_percent = float(value)

    def snapshot(self) -> AttenuatorSnapshot:
        return AttenuatorSnapshot(
            timestamp_s=time.time(),
            transmission_percent=self._transmission_percent,
            connected=self._connected,
            provider=self.name,
            readback_known=self._readback_known,
            device_name=self.config.device_name,
            metadata={
                "gpio_name": self.config.gpio_name,
                "raw_readback": self._raw_readback,
                "raw_min": self.config.raw_min,
                "raw_max": self.config.raw_max,
                "mapping": "raw 0 = 100 % transmission; raw 10 = 0 % transmission",
                "inverted": self.config.inverted,
            },
        )


class UnconfiguredAttenuatorProvider(AttenuatorProvider):
    """Safe placeholder until the physical attenuator hardware is identified.

    This deliberately refuses commands instead of guessing a serial protocol,
    motor angle convention, calibration curve, voltage range, or controller.
    """

    name = "real-attenuator-unconfigured"

    def __init__(self, device_name: str = "Real attenuator — driver not configured") -> None:
        self.device_name = device_name

    def connect(self) -> None:
        raise RuntimeError(
            "Real attenuator control is not bound yet. Add the device-specific "
            "driver/calibration before enabling hardware attenuation commands."
        )

    def disconnect(self) -> None:
        return

    def set_transmission_percent(self, value: float) -> None:
        raise RuntimeError(
            "Real attenuator control is unavailable until its hardware driver is configured"
        )

    def snapshot(self) -> AttenuatorSnapshot:
        return AttenuatorSnapshot(
            timestamp_s=time.time(),
            transmission_percent=0.0,
            connected=False,
            provider=self.name,
            readback_known=False,
            device_name=self.device_name,
        )
