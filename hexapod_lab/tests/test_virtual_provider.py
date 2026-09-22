from hexapod_lab.providers import VirtualHexapodProvider, VirtualLaserGate
from hexapod_lab.types import MotionState, Pose6D


def test_virtual_hexapod_reaches_target():
    stage = VirtualHexapodProvider(linear_speed_mm_s=10.0)
    stage.connect()
    stage.move_absolute(Pose6D(x=2.0, z=1.0))
    assert stage.snapshot().state == MotionState.MOVING
    for _ in range(200):
        stage.tick(0.02)
    snap = stage.snapshot()
    assert snap.state == MotionState.IDLE
    assert abs(snap.actual.x - 2.0) < 1e-9
    assert abs(snap.actual.z - 1.0) < 1e-9


def test_virtual_laser_fail_closed():
    gate = VirtualLaserGate()
    gate.connect()
    assert gate.snapshot().gate_enabled is False
    gate.set_gate(True)
    assert gate.snapshot().gate_enabled is True
    gate.disconnect()
    assert gate.snapshot().gate_enabled is False
