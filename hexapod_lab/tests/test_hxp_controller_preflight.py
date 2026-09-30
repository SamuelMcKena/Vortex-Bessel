from hexapod_lab.hxp_client import HXPClient


class FakeEndpoint:
    connected = True

    def __init__(self):
        self.commands = []

    def request(self, command, **kwargs):
        self.commands.append(command)
        return 0, "12.5,100.0"


def test_line_control_limit_get_uses_native_hxp_api():
    client = object.__new__(HXPClient)
    client.poll = FakeEndpoint()
    client.control = FakeEndpoint()
    client.io = FakeEndpoint()
    vmax, percent = client.line_incremental_control_limits(
        -7.0, 0.02, 0.0, group="HEXAPOD", coordinate_system="Work"
    )
    assert vmax == 12.5
    assert percent == 100.0
    assert client.poll.commands == [
        "HexapodMoveIncrementalControlLimitGet(HEXAPOD,Work,Line,-7,0.02,0,double *,double *)"
    ]
