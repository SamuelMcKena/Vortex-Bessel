from hexapod_lab.providers import HXPDigitalLaserConfig, HXPDigitalLaserGate


class FakeHXPClient:
    def __init__(self):
        self.connected = True
        self.value = 0
        self.writes = []

    def digital_set(self, gpio_name, mask, value):
        self.writes.append((gpio_name, mask, value))
        self.value = (self.value & ~mask) | (value & mask)

    def digital_get(self, gpio_name):
        return self.value


class BadReadbackClient(FakeHXPClient):
    def digital_get(self, gpio_name):
        return 0


def config():
    return HXPDigitalLaserConfig(
        gpio_name="GPIO4.DO",
        mask=1,
        enabled_value=1,
        disabled_value=0,
        wiring_verified=True,
    )


def test_real_pockels_provider_closes_on_connect_and_reads_back():
    client = FakeHXPClient()
    gate = HXPDigitalLaserGate(client, config())
    gate.connect()
    snap = gate.snapshot()
    assert snap.connected is True
    assert snap.pockels_open is False
    assert snap.readback_known is True
    assert client.writes[-1] == ("GPIO4.DO", 1, 0)

    gate.set_gate(True)
    snap = gate.snapshot()
    assert snap.pockels_open is True
    assert snap.metadata["raw_readback"] & 1 == 1


def test_real_pockels_provider_rejects_readback_mismatch():
    client = BadReadbackClient()
    gate = HXPDigitalLaserGate(client, config())
    gate.connect()  # CLOSED=0 matches the bad readback.
    try:
        gate.set_gate(True)
    except RuntimeError as exc:
        assert "readback" in str(exc).lower()
        assert client.writes[-1] == ("GPIO4.DO", 1, 0)
        assert gate.snapshot().pockels_open is False
    else:
        raise AssertionError("readback mismatch was not rejected")
