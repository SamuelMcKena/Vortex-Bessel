from __future__ import annotations

import numpy as np
import pytest

from slm_lab_control.config import AppConfig

from labcontrol.devices.base import ProviderError
from labcontrol.devices.camera import DummyCameraProvider, ReplayCameraProvider
from labcontrol.mode_controller import ModeAwareLabController
from labcontrol.state import ExperimentState, ExperimentStore
from labcontrol.virtual_lab import (
    DataOrigin,
    OperatingMode,
    RecordedReadOnlySlmProvider,
    ScenarioKind,
    VirtualBenchEngine,
    VirtualLabGeometry,
    VirtualSlmProvider,
    capture_virtual_z_stack,
)


def make_controller(tmp_path, *, grid_n: int = 96) -> ModeAwareLabController:
    state = ExperimentState.from_app_config(AppConfig())
    store = ExperimentStore(state)
    camera = DummyCameraProvider(store.snapshot, shape_yx=(64, 64))
    engine = VirtualBenchEngine(
        VirtualLabGeometry(preview_grid_n=grid_n, validation_grid_n=max(128, grid_n)),
        quality="preview",
    )
    return ModeAwareLabController(
        store,
        tmp_path,
        camera_provider=camera,
        virtual_engine=engine,
    )


def test_seeded_hidden_scenario_is_reproducible_without_public_truth(tmp_path) -> None:
    controller = make_controller(tmp_path)
    engine = controller.virtual_engine
    first = engine.generate_scenario(ScenarioKind.MIXED, 1048)
    engine.reset_scenario()
    second = engine.generate_scenario(ScenarioKind.MIXED, 1048)
    assert first.truth_hash == second.truth_hash
    assert first.scenario_id == second.scenario_id
    assert not hasattr(first, "hidden_truth")
    public = engine.public_scenario_record()
    assert "hidden_truth" not in public
    assert set(public) == {"scenario_id", "kind", "seed", "truth_hash", "created_utc"}


def test_virtual_mode_uses_dry_run_slms_and_same_camera_frame_contract(tmp_path) -> None:
    controller = make_controller(tmp_path)
    controller.set_operating_mode(OperatingMode.VIRTUAL_LAB)
    assert isinstance(controller.slm_provider, VirtualSlmProvider)
    assert controller.camera_provider is not None
    assert controller.camera_provider.name == "virtual"

    controller.store.set_vortex("SLM1", True, 40, source="test")
    controller.cast(("SLM1", "SLM2"), persist=False)
    controller.connect_camera()
    controller.start_camera()
    controller.move_stage(31.0)
    frame = controller.acquire_frame(fresh=True)

    assert frame.provider == "virtual"
    assert frame.data_kind == "SYNTHETIC"
    assert frame.data.shape == (96, 96)
    assert np.isfinite(frame.data).all()
    assert frame.metadata["data_origin"] == DataOrigin.SIMULATED.value
    assert frame.metadata["operating_mode"] == OperatingMode.VIRTUAL_LAB.value
    assert frame.metadata["hidden_truth_exposed"] is False
    assert "hidden_truth" not in frame.metadata
    assert frame.metadata["geometry_status"] == "PARTIALLY_BOUND_NOT_BENCH_CALIBRATED"


def test_virtual_z_stack_uses_one_frame_contract_and_changes_with_z(tmp_path) -> None:
    controller = make_controller(tmp_path)
    controller.set_operating_mode(OperatingMode.VIRTUAL_LAB)
    controller.store.set_vortex("SLM1", True, 5, source="test")
    controller.cast(("SLM1", "SLM2"), persist=False)
    result = capture_virtual_z_stack(controller, (31.0, 34.0, 37.0))

    assert result.z_mm == (31.0, 34.0, 37.0)
    assert len(result.frames) == 3
    assert np.isfinite(result.objective.total)
    assert all(frame.metadata["data_origin"] == "simulated" for frame in result.frames)
    assert not np.allclose(result.frames[0].data, result.frames[-1].data)


def test_switching_virtual_to_live_invalidates_virtual_cast_receipts(tmp_path) -> None:
    controller = make_controller(tmp_path)
    controller.set_operating_mode(OperatingMode.VIRTUAL_LAB)
    controller.cast(("SLM1", "SLM2"), persist=False)
    before = controller.store.snapshot()
    assert before.slm1.last_cast_sha256
    assert before.slm2.last_cast_sha256

    controller.set_operating_mode(OperatingMode.LIVE_LAB)
    after = controller.store.snapshot()
    assert after.slm1.last_cast_sha256 is None
    assert after.slm2.last_cast_sha256 is None
    assert controller.camera_provider is not None
    assert controller.camera_provider.name == "beamage"
    assert not controller.camera_provider.connected


def test_recorded_lab_is_read_only_and_frames_keep_measured_replay_provenance(tmp_path) -> None:
    controller = make_controller(tmp_path)
    source = tmp_path / "recorded.npy"
    np.save(source, np.ones((32, 32), dtype=np.float64), allow_pickle=False)
    replay = ReplayCameraProvider([source], loop=True, source_data_kind="REPLAY")
    controller.set_operating_mode(OperatingMode.RECORDED_LAB, replay_provider=replay)
    assert isinstance(controller.slm_provider, RecordedReadOnlySlmProvider)

    controller.connect_camera()
    controller.start_camera()
    frame = controller.acquire_frame(fresh=True)
    assert frame.provider == "replay"
    assert frame.metadata["data_origin"] == DataOrigin.RECORDED_MEASURED.value
    assert frame.metadata["operating_mode"] == OperatingMode.RECORDED_LAB.value

    with pytest.raises(ProviderError):
        controller.cast(("SLM1",), persist=False)


def test_reported_geometry_is_not_silently_converted_to_ideal_f300_4f() -> None:
    geometry = VirtualLabGeometry()
    summary = geometry.public_summary()
    assert geometry.lens1_focal_length_mm == 300.0
    assert geometry.lens2_focal_length_mm == 300.0
    assert geometry.lens_to_lens_separation_mm == 300.0
    assert geometry.pinhole_axial_position_mm is None
    assert geometry.relay_mode == "ideal_selected_order_surrogate"
    assert "PARTIALLY_BOUND" in summary["geometry_status"]
    assert geometry.axicon_reported_label == "Thorlabs 20° axicon"
    assert "not_bound" in geometry.axicon_model_angle_source
