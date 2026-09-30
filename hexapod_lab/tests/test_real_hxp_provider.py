import threading
import time

import pytest

from hexapod_lab.providers import HXPProvider, HXPProviderConfig
from hexapod_lab.types import MotionState, Pose6D


class FakeHXPClient:
    def __init__(self, status=12, *, fail_snapshot=False):
        self.connected = False
        self.status = status
        self.fail_snapshot = fail_snapshot
        self.closed = False
        self.ready_checks = 0

    def connect(self):
        self.connected = True

    def close(self):
        self.connected = False
        self.closed = True

    def current_pose(self, group):
        if self.fail_snapshot:
            raise RuntimeError("snapshot failed")
        return Pose6D(z=-14.0)

    def setpoint_pose(self, group):
        return Pose6D(z=-14.0)

    def target_pose(self, group):
        return Pose6D(z=-14.0)

    def group_status(self, group):
        return self.status

    def group_status_text(self, status):
        return f"status-{status}"

    def require_ready_for_motion(self, group):
        self.ready_checks += 1
        if self.status not in (11, 12):
            raise RuntimeError("not ready")
        return self.status, f"status-{self.status}"

    def move_incremental(self, delta, group, coordinate_system):
        return None

    def abort(self, group):
        return None


def _provider(status=12, *, fail_snapshot=False):
    provider = HXPProvider(HXPProviderConfig(host="127.0.0.1"))
    provider.client = FakeHXPClient(status, fail_snapshot=fail_snapshot)
    return provider


@pytest.mark.parametrize("status", [11, 12])
def test_real_provider_reports_ready_states_as_idle(status):
    provider = _provider(status)
    provider.connect()
    snap = provider.snapshot()
    assert snap.connected is True
    assert snap.state == MotionState.IDLE


def test_real_provider_reports_nonready_connected_state_as_fault():
    provider = _provider(42)
    provider.connect()
    snap = provider.snapshot()
    assert snap.connected is True
    assert snap.state == MotionState.FAULT


def test_real_provider_connect_failure_closes_partial_client():
    provider = _provider(12, fail_snapshot=True)
    with pytest.raises(RuntimeError, match="snapshot failed"):
        provider.connect()
    assert provider.client.connected is False
    assert provider.client.closed is True


def test_real_motion_checks_ready_state_before_worker_start():
    provider = _provider(42)
    # Avoid provider.connect(), which is allowed to show the non-ready state.
    provider.client.connect()
    provider._connected = True
    with pytest.raises(RuntimeError, match="not ready"):
        provider.move_incremental(Pose6D(x=0.01))
    assert provider.client.ready_checks == 1


def test_explicit_abort_suppresses_worker_abort_exception():
    provider = _provider(12)
    provider.connect()

    release = threading.Event()

    def blocking():
        release.wait(1.0)
        raise RuntimeError("motion interrupted")

    provider._start_blocking_call(blocking)
    provider.abort()
    release.set()
    provider._move_thread.join(timeout=1.0)
    # An intentional abort must not later poison polling/recipe state.
    assert provider._move_error is None
