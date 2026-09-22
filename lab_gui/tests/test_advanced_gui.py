from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from labcontrol.ui.advanced import AdvancedLabWindow
from labcontrol.ui.beam_walk_view import BeamWalkPlot
from labcontrol.ui.image_view import render_preview
from labcontrol.beam_walk import PlaneObservation, fit_beam_walk
from slm_lab_control.app import MainWindow, ManualDoubleSpinBox


def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_advanced_gui_starts_on_nine_section_home_cockpit() -> None:
    qt = app()
    window = AdvancedLabWindow()
    try:
        assert window.pages.count() == 9
        assert window.pages.currentIndex() == window.PAGE_HOME
        assert window.image_view is not None
        assert window.home_camera_view is not None
        assert set(window.slm_quick_cards) == {"SLM1", "SLM2"}
        assert set(window.slm_detail_views) == {"SLM1", "SLM2"}
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
        assert advanced.slm_quick_cards["SLM1"].charge.value() == 11
        assert advanced.slm_detail_views["SLM1"].panel.vortex_charge.value() == 11
        assert compact.slm1_panel.vortex_charge.value() == 11
        assert compact.slm1_panel.grp_vortex.isChecked()
    finally:
        compact.close()
        advanced.close()
        qt.processEvents()


def test_in_progress_numeric_text_survives_unrelated_refresh_and_tab_switch() -> None:
    qt = app()
    window = AdvancedLabWindow()
    try:
        editor = window.slm_detail_views["SLM1"].panel.center_x.lineEdit()
        editor.setFocus(Qt.OtherFocusReason)
        editor.selectAll()
        editor.insert("1234.5")

        window.store.update(
            lambda state: setattr(state.camera, "last_frame_id", "unrelated-frame"),
            source="test",
            reason="Unrelated camera refresh",
        )
        qt.processEvents()
        assert editor.text() == "1234.5"

        window.set_page(window.PAGE_MEASURE)
        window.set_page(window.PAGE_SLM1)
        qt.processEvents()
        assert editor.text() == "1234.5"
    finally:
        window.close()
        qt.processEvents()


def test_programmatic_state_update_reaches_home_detail_and_compact_views() -> None:
    qt = app()
    advanced = AdvancedLabWindow()
    compact = MainWindow(controller=advanced.controller)
    try:
        advanced.store.update(
            lambda state: setattr(state.slm2.phase, "center_x_px", 777.25),
            source="test",
            reason="Programmatic SLM2 update",
        )
        qt.processEvents()
        assert advanced.slm_detail_views["SLM2"].panel.center_x.value() == 777.25
        assert compact.slm2_panel.center_x.value() == 777.25
    finally:
        compact.close()
        advanced.close()
        qt.processEvents()


class _WheelEvent:
    def __init__(self) -> None:
        self.ignored = False

    def ignore(self) -> None:
        self.ignored = True


def test_numeric_controls_ignore_wheel_but_keep_normal_stepping() -> None:
    advanced = AdvancedLabWindow()
    try:
        for spin in (
            advanced.home_display_gamma,
            advanced.slm_quick_cards["SLM1"].charge,
            ManualDoubleSpinBox(),
        ):
            before = spin.value()
            event = _WheelEvent()
            spin.wheelEvent(event)
            assert event.ignored is True
            assert spin.value() == before
            spin.stepBy(1)
            assert spin.value() != before
    finally:
        advanced.close()


def test_camera_worker_stops_cleanly_and_delivers_quantitative_frame() -> None:
    qt = app()
    window = AdvancedLabWindow()
    from PySide6.QtCore import QThread

    delivered_on_gui_thread = []
    original_on_frame = window._on_frame

    def checked_on_frame(frame):
        delivered_on_gui_thread.append(QThread.currentThread() == window.thread())
        original_on_frame(frame)

    window._on_frame = checked_on_frame
    try:
        window.start_live()
        deadline = time.monotonic() + 2.0
        while window.current_frame is None and time.monotonic() < deadline:
            qt.processEvents()
            time.sleep(0.01)
        assert window.current_frame is not None
        assert window.current_frame.data.ndim == 2
        assert delivered_on_gui_thread == [True], "Qt widgets must never be updated from the camera worker"
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


def test_beamage_preview_keeps_every_acquired_pixel_and_can_fit_signal() -> None:
    qt = app()
    from labcontrol.devices.camera import CameraFrame
    from labcontrol.state import utc_now
    from labcontrol.ui.image_view import QuantitativeImageView

    raw = np.zeros((900, 1301), dtype=np.float64)
    raw[410:490, 610:690] = 1000.0
    frame = CameraFrame(
        data=raw,
        frame_id="full-resolution-test",
        timestamp_utc=utc_now(),
        provider="beamage",
        exposure_us=2500.0,
        gain=0.0,
        full_scale=4095.0,
        z_mm=None,
        data_kind="EXPERIMENT",
        metadata={"quantitative_valid": False},
    )
    view = QuantitativeImageView()
    try:
        view.set_quantitative_frame(
            frame,
            None,
            colour="inferno",
            scale="percentile",
            gamma=1.0,
        )
        # Every acquired pixel is kept; only a zoomed-out *screen* rendering is
        # area-averaged so the display cannot invent moire.
        assert view._display_rgb.shape[:2] == (900, 1301)
        assert (view.scene().sceneRect().width(), view.scene().sceneRect().height()) == (1301, 900)
        rect = view.signal_rect()
        assert rect is not None
        assert rect.width() < 1301
        assert rect.height() < 900
        view.resize(1400, 1000)
        view.resetTransform()
        view.scale(2.0, 2.0)
        qt.processEvents()
        assert view._pixmap_item.pixmap().width() == 1301, "zoomed in, every camera pixel must be drawn"
        view.scale(0.2, 0.2)
        assert view._pixmap_item.pixmap().width() < 1301
        assert view._pixmap_item.scale() > 1.0
        view.fit_signal()
        qt.processEvents()
    finally:
        view.close()


def test_scientific_colour_maps_are_distinct_and_preserve_shape() -> None:
    raw = np.linspace(0.0, 4095.0, 400).reshape(20, 20)
    previews = {
        name: render_preview(raw, colour=name, scale="sensor range", gamma=1.0, full_scale=4095.0)
        for name in ("inferno", "gentec-like", "turbo", "viridis")
    }
    assert all(preview.shape == (20, 20, 3) for preview in previews.values())
    assert len({preview.tobytes() for preview in previews.values()}) == len(previews)


def test_camera_ui_offers_full_frame_beam_fit_and_beamage_control_route() -> None:
    qt = app()
    window = AdvancedLabWindow()
    try:
        assert window.home_auto_fit.isChecked()
        assert window.auto_fit.isChecked()
        assert "inferno" in [window.home_colour_mode.itemText(i) for i in range(window.home_colour_mode.count())]
        assert "sensor range" in [window.display_scale.itemText(i) for i in range(window.display_scale.count())]
        window._select_camera_provider("beamage")
        qt.processEvents()
        assert not window.pc_beamage_controls.isHidden()
        assert "official Pipeline" in window.camera_control_note.text()
    finally:
        window.close()
        qt.processEvents()
