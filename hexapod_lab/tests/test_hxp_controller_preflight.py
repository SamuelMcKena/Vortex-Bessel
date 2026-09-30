from hexapod_lab.hxp_client import HXPClient


class FakeEndpoint:
    connected = True

    def __init__(self):
        self.commands = []

    def request(self, command, **kwargs):
        self.commands.append(command)
        return 0, "12.5,1.0"


def test_line_control_limit_get_uses_native_hxp_api():
    client = object.__new__(HXPClient)
    client.poll = FakeEndpoint()
    client.control = FakeEndpoint()
    client.io = FakeEndpoint()
    vmax, percent = client.line_incremental_control_limits(
        -7.0, 0.02, 0.0, group="HEXAPOD", coordinate_system="Work"
    )
    assert vmax == 12.5
    assert percent == 1.0
    assert client.poll.commands == [
        "HexapodMoveIncrementalControlLimitGet(HEXAPOD,Work,Line,-7,0.02,0,double *,double *)"
    ]


def test_full_hxp_trajectory_fraction_is_one():
    client = object.__new__(HXPClient)
    client.poll = FakeEndpoint()
    client.control = FakeEndpoint()
    client.io = FakeEndpoint()
    vmax, fraction = client.line_incremental_control_limits(
        0.05, 0.0, 0.0, group="HEXAPOD", coordinate_system="Work"
    )
    assert vmax == 12.5
    assert fraction == 1.0
