from __future__ import annotations

import numpy as np

from labcontrol.controller import LabController
from labcontrol.devices.camera import DummyCameraProvider
from labcontrol.phase_service import PhaseService, phase_sha256
from labcontrol.state import ConnectionState, ExperimentState, ExperimentStore
from slm_lab_control.phase import compose_phase


def test_phase_service_is_exact_v06_composer_adapter() -> None:
    state = ExperimentState()
    state.slm1.phase.switches.vortex = True
    state.slm1.phase.vortex_charge = 4
    state.slm2.phase.switches.steering = True
    state.slm2.phase.steering_x_mrad = 0.25
    expected1 = compose_phase(state.slm1.phase)
    expected2 = compose_phase(state.slm2.phase)
    bundle = PhaseService().generate(state)
    np.testing.assert_array_equal(bundle.results["SLM1"].phase_rad, expected1.phase_rad)
    np.testing.assert_array_equal(bundle.results["SLM2"].phase_rad, expected2.phase_rad)
    assert bundle.hashes["SLM1"] == phase_sha256(expected1.phase_rad)


def test_controller_records_configured_generated_and_cast_state_separately(tmp_path) -> None:
    state = ExperimentState()
    state.application.output_root = str(tmp_path / "casts")
    store = ExperimentStore(state)
    controller = LabController(store, tmp_path)
    bundle = controller.generate()
    generated = store.snapshot()
    assert generated.slm1.complete_phase_sha256 == bundle.hashes["SLM1"]
    assert generated.slm1.last_cast_sha256 is None

    receipt = controller.cast(("SLM1",))
    cast = store.snapshot()
    assert cast.slm1.last_cast_sha256 == receipt.phase_hashes["SLM1"]
    assert cast.slm2.last_cast_sha256 is None
    assert cast.slm1.connection == ConnectionState.CONNECTED
    assert receipt.folder.is_dir()


def test_controller_camera_updates_same_store(tmp_path) -> None:
    store = ExperimentStore()
    camera = DummyCameraProvider(store.snapshot, shape_yx=(64, 80))
    controller = LabController(store, tmp_path, camera_provider=camera)
    controller.connect_camera()
    controller.start_camera()
    acquired = controller.acquire_frame()
    assert store.snapshot().camera.last_frame_id == acquired.frame_id
    assert store.snapshot().camera.shape_yx == (64, 80)
    controller.stop_camera()
