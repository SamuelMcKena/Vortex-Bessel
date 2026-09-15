from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from labcontrol.beam_walk import PlaneObservation, compare_beam_walk, fit_beam_walk
from labcontrol.devices.camera import (
    BeamageCameraProvider,
    CameraFrame,
    DummyCameraProvider,
    ReplayCameraProvider,
)
from labcontrol.devices.base import ProviderUnavailable
from labcontrol.metrics import MetricEngine
from labcontrol.state import ExperimentState, ExperimentStore


def frame(data: np.ndarray, *, frame_id: str = "test") -> CameraFrame:
    return CameraFrame(
        data=data,
        frame_id=frame_id,
        timestamp_utc="2026-09-15T00:00:00+00:00",
        provider="test",
        exposure_us=1000.0,
        gain=0.0,
        full_scale=4095.0,
        z_mm=38.0,
        data_kind="SYNTHETIC",
    )


def test_dummy_camera_is_deterministic_state_aware_and_labelled() -> None:
    store = ExperimentStore()
    camera = DummyCameraProvider(store.snapshot, shape_yx=(128, 128), seed=9)
    status = camera.connect()
    assert status.implementation_status == "SOFTWARE_TESTED"
    first = camera.acquire_frame()
    assert first.data_kind == "SYNTHETIC"
    assert "not optical propagation" in first.metadata["warning"]
    assert first.shape_yx == (128, 128)
    assert len(first.pixel_sha256) == 64

    store.set_vortex("SLM1", True, 7)
    second = camera.acquire_frame()
    assert second.metadata["effective_vortex_charge"] == 7
    assert not np.array_equal(first.data, second.data)


def test_replay_camera_reads_quantitative_npy_without_rendering(tmp_path: Path) -> None:
    source = tmp_path / "z38.npy"
    expected = np.arange(64, dtype=np.float32).reshape(8, 8)
    np.save(source, expected)
    camera = ReplayCameraProvider(source, z_by_name={"z38": 38.0})
    camera.connect()
    actual = camera.acquire_frame()
    np.testing.assert_array_equal(actual.data, expected)
    assert actual.data_kind == "REPLAY"
    assert actual.z_mm == 38.0
    assert actual.metadata["source_path"] == str(source)


def test_beamage_refuses_to_fabricate_connection_without_live_windows_pipe() -> None:
    with pytest.raises(ProviderUnavailable, match="named pipe is available only on Windows"):
        BeamageCameraProvider().connect()


def test_metric_family_tracks_authoritative_vortex_state() -> None:
    size = 161
    y, x = np.indices((size, size), dtype=float)
    cy, cx = 78.4, 83.1
    gaussian = 12.0 + 3000.0 * np.exp(
        -0.5 * (((x - cx) / 8.0) ** 2 + ((y - cy) / 12.0) ** 2)
    )
    state = ExperimentState()
    central = MetricEngine().analyse(frame(gaussian), state)
    assert central.family == "central_beam"
    assert central.centre_yx_px == pytest.approx((cy, cx), abs=0.1)
    assert central.values["ellipticity"] == pytest.approx(1.5, rel=0.03)

    radius = np.hypot(x - cx, y - cy)
    ring = 12.0 + 2500.0 * np.exp(-0.5 * ((radius - 29.0) / 3.0) ** 2)
    state.slm1.phase.switches.vortex = True
    state.slm1.phase.vortex_charge = 6
    vortex = MetricEngine().analyse(frame(ring), state)
    assert vortex.family == "vortex_bessel"
    assert vortex.centre_yx_px == pytest.approx((cy, cx), abs=0.5)
    assert vortex.values["principal_ring_radius_px"] == pytest.approx(29.0, abs=1.5)
    assert vortex.values["dark_core_fraction"] < 0.05


def test_beam_walk_fit_reports_relative_mismatch_and_repeat_scatter() -> None:
    observations = []
    for z in (31.0, 38.0, 45.0):
        for repeat, noise in enumerate((-0.1, 0.1), 1):
            observations.append(
                PlaneObservation(
                    frame_id=f"z{z}-{repeat}",
                    z_mm=z,
                    centre_y_px=80.0 - 0.2 * z + noise,
                    centre_x_px=60.0 + 0.3 * z - noise,
                    ring_radius_px=25.0 + 0.1 * z,
                )
            )
    result = fit_beam_walk(observations, pixel_size_um=5.5)
    assert result.fit["x_mrad"] == pytest.approx(1.65)
    assert result.fit["y_mrad"] == pytest.approx(-1.10)
    assert result.fit["r2_x"] == pytest.approx(1.0)
    assert "not an absolute laser angle" in result.interpretation
    assert result.fit["repeat_scatter_px_rms"] > 0

    other = fit_beam_walk(
        [
            PlaneObservation("a", 0.0, 0.0, 0.0),
            PlaneObservation("b", 1.0, 0.0, 1.0),
        ],
        pixel_size_um=1.0,
    )
    comparison = compare_beam_walk(result, other, material_difference_mrad=0.2)
    assert comparison["materially_different"] is True
