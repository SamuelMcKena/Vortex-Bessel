from hexapod_lab.providers import (
    VirtualAttenuatorProvider,
    VirtualHexapodProvider,
    VirtualLaserGate,
)
from hexapod_lab.types import MotionState, Pose6D


def test_virtual_hexapod_reaches_target():
    stage = VirtualHexapodProvider(linear_speed_mm_s=10.0)
    stage.connect()
    stage.move_absolute(Pose6D(x=2.0, z=1.0))
    assert stage.snapshot().state == MotionState.MOVING
    assert stage.is_busy() is True
    for _ in range(200):
        stage.tick(0.02)
    snap = stage.snapshot()
    assert snap.state == MotionState.IDLE
    assert stage.is_busy() is False
    assert abs(snap.actual.x - 2.0) < 1e-9
    assert abs(snap.actual.z - 1.0) < 1e-9


def test_virtual_laser_fail_closed():
    gate = VirtualLaserGate()
    gate.connect()
    assert gate.snapshot().pockels_open is False
    gate.set_gate(True)
    assert gate.snapshot().pockels_open is True
    gate.disconnect()
    assert gate.snapshot().pockels_open is False


def test_virtual_attenuator_setpoint():
    attenuator = VirtualAttenuatorProvider(0.0)
    attenuator.connect()
    attenuator.set_transmission_percent(37.5)
    snap = attenuator.snapshot()
    assert snap.connected is True
    assert snap.readback_known is True
    assert snap.transmission_percent == 37.5


def test_virtual_attenuator_rejects_out_of_range():
    attenuator = VirtualAttenuatorProvider()
    attenuator.connect()
    try:
        attenuator.set_transmission_percent(-0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative transmission setpoint was accepted")


def test_virtual_target_velocity_line_move():
    stage = VirtualHexapodProvider()
    stage.connect()
    stage.move_line_incremental_with_target_velocity(
        3.0,
        4.0,
        0.0,
        5.0,
    )
    assert stage.is_busy() is True
    # 3-4-5 line at 5 mm/s should take 1 s in the virtual model.
    for _ in range(49):
        stage.tick(0.02)
    assert stage.is_busy() is True
    stage.tick(0.03)
    snap = stage.snapshot()
    assert stage.is_busy() is False
    assert abs(snap.actual.x - 3.0) < 1e-9
    assert abs(snap.actual.y - 4.0) < 1e-9
