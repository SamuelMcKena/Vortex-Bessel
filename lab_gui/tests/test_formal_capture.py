from __future__ import annotations

import json
from pathlib import Path

from labcontrol.capture import CaptureError, FormalCaptureService, SessionRepository
from labcontrol.controller import LabController
from labcontrol.devices.camera import DummyCameraProvider
from labcontrol.state import ExperimentState, ExperimentStore


def make_controller(tmp_path: Path) -> LabController:
    state = ExperimentState()
    state.application.output_root = str(tmp_path / "casts")
    state.camera.settings_id = "dummy-fixed-1000us-g0"
    state.camera.z_reference = "synthetic-stage-zero"
    state.camera.current_z_mm = 38.0
    store = ExperimentStore(state)
    camera = DummyCameraProvider(store.snapshot, shape_yx=(64, 64), seed=3)
    controller = LabController(store, tmp_path, camera_provider=camera)
    controller.connect_camera()
    controller.start_camera()
    return controller


def test_formal_capture_refuses_uncast_state(tmp_path: Path) -> None:
    controller = make_controller(tmp_path)
    service = FormalCaptureService(controller, SessionRepository(tmp_path / "session"))
    try:
        service.capture("trial_refused", required_slms=("SLM1",), auto_cast=False)
    except CaptureError as exc:
        assert "not the last successful cast" in str(exc)
    else:
        raise AssertionError("uncast formal state was accepted")


def test_formal_capture_links_raw_frame_state_phase_and_metrics(tmp_path: Path) -> None:
    controller = make_controller(tmp_path)
    repository = SessionRepository(tmp_path / "session")
    service = FormalCaptureService(controller, repository, sleeper=lambda _seconds: None)
    record = service.capture(
        "trial_0001",
        repeats=2,
        recipe="test_capture",
        role="VERIFICATION",
        settle_s=0.01,
        required_slms=("SLM1",),
    )
    assert record.data_kind == "SYNTHETIC"
    assert len(record.frame_ids) == 2
    measurement = json.loads(
        (record.trial_root / "measurement.json").read_text(encoding="utf-8")
    )
    assert measurement["data_kind"] == "SYNTHETIC"
    assert measurement["camera"]["settings_id"] == "dummy-fixed-1000us-g0"
    assert measurement["camera"]["z_mm"] == 38.0
    assert measurement["phase_hashes"]["SLM1"] == record.phase_hashes["SLM1"]
    assert (record.trial_root / "state.json").is_file()
    assert (record.trial_root / "camera" / "frame_001.npy").is_file()
    assert (repository.objects / f"{record.phase_hashes['SLM1']}.npy").is_file()
    metrics = json.loads((record.trial_root / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["analysis_version"] == "labcontrol-metrics-1"
    assert metrics["repeat_stability"]["repeat_count"] == 2.0

    session = json.loads((repository.root / "session.json").read_text(encoding="utf-8"))
    assert session["data_kind"] == "SYNTHETIC"
    assert session["trials"][0]["trial_id"] == "trial_0001"
