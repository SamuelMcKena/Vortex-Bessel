from hexapod_lab.providers import (
    HXPAnalogAttenuatorConfig,
    HXPAnalogAttenuatorProvider,
)


class FakeClient:
    connected = True

    def __init__(self, raw=4.0):
        self.raw = float(raw)
        self.writes = []

    def analog_get(self, gpio):
        assert gpio == "GPIO2.DAC1"
        return self.raw

    def analog_set(self, gpio, value):
        assert gpio == "GPIO2.DAC1"
        self.raw = float(value)
        self.writes.append((gpio, float(value)))


def test_real_attenuator_reads_4_as_60_percent():
    client = FakeClient(4.0)
    att = HXPAnalogAttenuatorProvider(client, HXPAnalogAttenuatorConfig())
    att.connect()
    snap = att.snapshot()
    assert snap.connected is True
    assert snap.readback_known is True
    assert snap.transmission_percent == 60.0
    assert snap.metadata["raw_readback"] == 4.0


def test_real_attenuator_writes_percent_as_dac_and_checks_readback():
    client = FakeClient(4.0)
    att = HXPAnalogAttenuatorProvider(client, HXPAnalogAttenuatorConfig())
    att.connect()
    att.set_transmission_percent(25.0)
    assert client.writes[-1] == ("GPIO2.DAC1", 7.5)
    assert att.snapshot().transmission_percent == 25.0


def test_real_attenuator_endpoint_mapping_is_inverted():
    client = FakeClient(0.0)
    att = HXPAnalogAttenuatorProvider(client, HXPAnalogAttenuatorConfig())
    att.connect()
    assert att.snapshot().transmission_percent == 100.0
    att.set_transmission_percent(0.0)
    assert client.writes[-1] == ("GPIO2.DAC1", 10.0)
    att.set_transmission_percent(100.0)
    assert client.writes[-1] == ("GPIO2.DAC1", 0.0)
