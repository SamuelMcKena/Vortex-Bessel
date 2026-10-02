import pytest

from hexapod_lab.hxp_client import HXPClient, HXPProtocolError


class FakeEndpoint:
    connected = True

    def __init__(self, response="12.5,1.0"):
        self.commands = []
        self.kwargs = []
        self.response = response

    def request(self, command, **kwargs):
        self.commands.append(command)
        self.kwargs.append(kwargs)
        return 0, self.response


def _client(poll_response="12.5,1.0"):
    client = object.__new__(HXPClient)
    client.poll = FakeEndpoint(poll_response)
    client.control = FakeEndpoint("")
    client.io = FakeEndpoint("")
    return client


def test_line_control_limit_get_uses_native_hxp_api():
    client = _client()
    vmax, fraction = client.line_incremental_control_limits(
        -7.0,
        0.02,
        0.0,
        group="HEXAPOD",
        coordinate_system="Work",
    )
    assert vmax == 12.5
    assert fraction == 1.0
    assert client.poll.commands == [
        "HexapodMoveIncrementalControlLimitGet(HEXAPOD,Work,Line,-7,0.02,0,double *,double *)"
    ]


def test_full_hxp_trajectory_fraction_is_one():
    client = _client()
    vmax, fraction = client.line_incremental_control_limits(
        0.05,
        0.0,
        0.0,
        group="HEXAPOD",
        coordinate_system="Work",
    )
    assert vmax == 12.5
    assert fraction == 1.0


@pytest.mark.parametrize("response", ["12.5,1.1", "12.5,-0.1", "0,1"])
def test_line_control_limit_rejects_impossible_controller_outputs(response):
    client = _client(response)
    with pytest.raises(HXPProtocolError):
        client.line_incremental_control_limits(0.05, 0.0, 0.0)


def test_line_control_limit_rejects_zero_length():
    client = _client()
    with pytest.raises(ValueError, match="non-zero"):
        client.line_incremental_control_limits(0.0, 0.0, 0.0)


def test_long_line_move_gets_motion_sized_socket_timeout():
    client = _client()
    client.control.config = type("Cfg", (), {"timeout_s": 10.0})()
    client.move_line_incremental_with_target_velocity(
        -7.0,
        0.0,
        0.0,
        0.2,
    )
    assert client.control.commands[-1] == (
        "HexapodMoveIncrementalControlWithTargetVelocity("
        "HEXAPOD,Work,Line,-7,0,0,0.2)"
    )
    # 7 mm / 0.2 mm/s = 35 s; implementation allows 30 s overhead and 2x motion.
    assert client.control.kwargs[-1]["response_timeout_s"] >= 100.0


class ReadyPoll:
    connected = True

    def __init__(self, status):
        self.status = status

    def request(self, command, **kwargs):
        if command.startswith("GroupStatusGet"):
            return 0, str(self.status)
        if command.startswith("GroupStatusStringGet"):
            return 0, f"status-{self.status}"
        raise AssertionError(command)


@pytest.mark.parametrize("status", [11, 12])
def test_require_ready_for_motion_accepts_referenced_ready_states(status):
    client = _client()
    client.poll = ReadyPoll(status)
    returned, text = client.require_ready_for_motion()
    assert returned == status
    assert text == f"status-{status}"


def test_require_ready_for_motion_rejects_not_referenced_state():
    client = _client()
    client.poll = ReadyPoll(42)
    with pytest.raises(RuntimeError, match="not ready for motion"):
        client.require_ready_for_motion()


def test_coordinate_system_get_uses_hexapod_api():
    client = _client("0,0,209,0,0,0")
    pose = client.coordinate_system_get("Work")
    assert pose.z == 209.0
    assert client.poll.commands[-1] == (
        "HexapodCoordinateSystemGet("
        "HEXAPOD,Work,double *,double *,double *,double *,double *,double *)"
    )
