from __future__ import annotations

import time

import numpy as np
from PySide6.QtWidgets import QApplication

from labcontrol.ui.advanced import AdvancedLabWindow
from labcontrol.ui.beam_walk_view import BeamWalkPlot
from labcontrol.ui.image_view import render_preview
from labcontrol.beam_walk import PlaneObservation, fit_beam_walk
from slm_lab_control.app import MainWindow


def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_advanced_gui_starts_with_six_focused_sections() -> None:
    qt = app()
    window = AdvancedLabWindow()
    try:
        assert window.pages.count() == 6
        assert window.image_view is not None
        assert window.pages.widget(window.PAGE_MEASURE) is not None
        assert window.store.snapshot().camera.provider == "dummy"
    finally:
        window.close()
        qt.processEvents()


def test_compact_and_advanced_clients_observe_same_vortex_state() -> None:
    qt = app()
    advanced = AdvancedLabWindow()
    compact = MainWindow(controller=advanced.controller)
    try:
        advanced.store.set_vortex("SLM1", True, 11, source="test")
        for _ in range(4):
            qt.processEvents()
        assert advanced.store.snapshot().effective_vortex_charge() == 11
        assert advanced.slm_state_controls["SLM1"][1].value() == 11
        assert compact.slm1_panel.vortex_charge.value() == 11
        assert compact.slm1_panel.grp_vortex.isChecked()
    finally:
        compact.close()
        advanced.close()
        qt.processEvents()


def test_camera_worker_stops_cleanly_and_delivers_quantitative_frame() -> None:
    qt = app()
    window = AdvancedLabWindow()
    try:
        window.start_live()
        deadline = time.monotonic() + 2.0
        while window.current_frame is None and time.monotonic() < deadline:
            qt.processEvents()
            time.sleep(0.01)
        assert window.current_frame is not None
        assert window.current_frame.data.ndim == 2
        window.stop_live()
        assert window._camera_thread is None
        assert window._camera_worker is None
    finally:
        window.close()
        qt.processEvents()


def test_display_transform_never_mutates_quantitative_input() -> None:
    raw = np.linspace(-2.0, 50.0, 100).reshape(10, 10)
    before = raw.copy()
    preview = render_preview(raw, colour="false colour", scale="log", gamma=1.4)
    np.testing.assert_array_equal(raw, before)
    assert preview.shape == (10, 10, 3)
    assert preview.dtype == np.uint8


def test_propagation_plot_renders_fitted_camera_relative_data() -> None:
    qt = app()
    result = fit_beam_walk(
        [
            PlaneObservation("near", 31.0, 100.0, 90.0, 24.0),
            PlaneObservation("mid", 38.0, 101.0, 92.0, 25.0),
            PlaneObservation("far", 45.0, 102.0, 94.0, 26.0),
        ],
        pixel_size_um=5.5,
    )
    plot = BeamWalkPlot()
    plot.resize(900, 240)
    plot.set_result(result)
    plot.show()
    qt.processEvents()
    assert not plot.grab().isNull()
    plot.close()
