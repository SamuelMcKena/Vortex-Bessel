"""Regressions for defects that made the assembled GUI unusable in practice."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from labcontrol.calibration import CalibrationKey, CalibrationStatus
from labcontrol.recipes import RecipePrerequisiteError, builtin_recipes
from labcontrol.ui.advanced import ADVANCED_QSS
from labcontrol.ui.virtual_advanced_v3 import GUIDE_HTML, VirtualLabResolutionWindow
from labcontrol.virtual_lab import OperatingMode
from labcontrol.virtual_lab_resolution import (
    ResolutionAwareVirtualBenchEngine,
    VirtualOutputResolution,
)
from slm_lab_control.app import SLMControlPanel
from slm_lab_control.config import SlmPhaseConfig
from slm_lab_control.ui.style import APP_QSS


def _app() -> QApplication:
    app = QApplication.instance() or QApplication([])
    # Layout assertions are only meaningful against the stylesheet the launcher
    # installs; the default style has noticeably wider control metrics.
    if not app.styleSheet():
        app.setStyleSheet(APP_QSS + ADVANCED_QSS)
    return app


def test_editing_any_slm_parameter_emits_changed_without_raising() -> None:
    """``changed`` is a zero-argument signal fed by one-argument Qt signals.

    Connecting ``changed.emit`` straight to ``valueChanged``/``toggled`` made
    PySide raise "changed() only accepts 0 argument(s), 1 given!" on every edit,
    so no SLM parameter change ever reached the state store.
    """

    app = _app()
    panel = SLMControlPanel(SlmPhaseConfig(name="SLM1"))
    emissions: list[int] = []
    panel.changed.connect(lambda: emissions.append(1))
    try:
        panel.vortex_charge.setValue(7)
        panel.background_gray.setValue(11)
        panel.grp_vortex.setChecked(not panel.grp_vortex.isChecked())
        app.processEvents()
        assert len(emissions) >= 3
        assert panel.to_config("SLM1").vortex_charge == 7
    finally:
        panel.deleteLater()
        app.processEvents()


def test_slm_detail_edits_reach_the_authoritative_state(tmp_path) -> None:
    app = _app()
    window = VirtualLabResolutionWindow()
    try:
        detail = window.slm_detail_views["SLM1"]
        detail.panel.vortex_charge.setValue(9)
        app.processEvents()
        assert window.store.snapshot().slm1.phase.vortex_charge == 9
    finally:
        window.close()
        app.processEvents()


def test_live_model_grid_resolves_rings_without_the_multi_second_grid() -> None:
    engine = ResolutionAwareVirtualBenchEngine(beamage_n=64, maximum_grid_n=128)
    engine.set_output_resolution(VirtualOutputResolution.LIVE_1M)
    assert engine.grid_n == 32  # SLM relay grid
    # The camera always reports true Beamage pixels, whatever the relay grid.
    assert engine.shape_yx == (64, 64)

def test_gui_opens_on_the_interactive_grid() -> None:
    app = _app()
    window = VirtualLabResolutionWindow()
    try:
        # 256/512 alias the q=20 ring family and 2048 costs seconds per camera
        # move, so the window must open on the grid in between.
        assert window.resolution_engine.output_resolution is VirtualOutputResolution.LIVE_1M
        assert window.resolution_engine.grid_n == 1024
        assert window.resolution_engine.live_n == 1024
    finally:
        window.close()
        app.processEvents()


def test_recipe_prerequisites_are_reported_before_the_operator_presses_start() -> None:
    app = _app()
    window = VirtualLabResolutionWindow()
    try:
        window.recipe_choice.setCurrentText("recover_q20")
        window._refresh_recipe_description()
        missing = window._missing_recipe_prerequisites()
        assert missing, "a fresh registry has no VALID calibrations"
        assert "Cannot start yet" in window.recipe_prerequisite_status.text()
        for name in missing:
            assert name.split(" [")[0] in window.recipe_prerequisite_status.text()
    finally:
        window.close()
        app.processEvents()


def test_simulated_calibration_unblocks_recipes_only_on_the_virtual_bench(monkeypatch) -> None:
    app = _app()
    window = VirtualLabResolutionWindow()
    shown: list[tuple] = []
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: shown.append(args))
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: shown.append(args))
    try:
        window.recipe_choice.setCurrentText("recover_q20")
        window._refresh_recipe_description()
        assert not window.recipe_simulated_calibration.isVisible()

        # Live lab keeps the dependency guard: nothing is marked.
        window._mark_simulated_calibrations()
        assert shown, "the live-lab path must explain why it refuses"
        assert window.calibration.status(CalibrationKey.CAMERA_PIXEL_SCALE) != CalibrationStatus.VALID
        try:
            window.recipe_engine.start("recover_q20")
        except RecipePrerequisiteError:
            pass
        else:
            raise AssertionError("prerequisites must still block a live-lab recipe")

        window._ensure_virtual_mode()
        window._refresh_state(window.store.snapshot())
        window._mark_simulated_calibrations()
        assert window.mode_controller.operating_mode is OperatingMode.VIRTUAL_LAB
        assert not window._missing_recipe_prerequisites()

        run = window.recipe_engine.start("recover_q20")
        assert run.recipe_name == "recover_q20"
        record = window.calibration.records[CalibrationKey.CAMERA_PIXEL_SCALE]
        assert record.calibration_id.startswith("SIMULATED::")
        assert any("SIMULATED" in item for item in record.evidence)
        assert record.hardware_fingerprint["bench"] == "VIRTUAL"
    finally:
        window.close()
        app.processEvents()


def test_every_builtin_recipe_can_be_rehearsed_on_the_virtual_bench() -> None:
    app = _app()
    window = VirtualLabResolutionWindow()
    try:
        window._ensure_virtual_mode()
        window._mark_simulated_calibrations()
        for name in sorted(builtin_recipes()):
            run = window.recipe_engine.start(name)
            assert window.recipe_engine.current_step(run) is not None
    finally:
        window.close()
        app.processEvents()


def test_window_opens_inside_the_available_screen() -> None:
    """Regression: the window opened larger than the work area and overhung it.

    resize() sets the client area, so sizing to the full work area still left
    the title bar and borders off the bottom and right of a 1280x800 display.
    """

    app = _app()
    window = VirtualLabResolutionWindow()
    try:
        available = app.primaryScreen().availableGeometry()
        assert window.width() <= available.width() - 16
        assert window.height() <= available.height() - 48
    finally:
        window.close()
        app.processEvents()


def test_instructions_page_is_reachable_and_describes_the_workflow() -> None:
    app = _app()
    window = VirtualLabResolutionWindow()
    try:
        window.resize(1280, 713)
        window.show()
        window.set_page(window.PAGE_GUIDE)
        app.processEvents()
        assert window.pages.currentIndex() == window.PAGE_GUIDE
        assert window.page_title.text() == "How to use this"
        scroll = window.pages.widget(window.PAGE_GUIDE)
        assert scroll.horizontalScrollBar().maximum() == 0
        for phrase in ("VIRTUAL LAB", "Cast masks + go live", "LIVE MODEL", "prerequisites", "Next iteration", "4.83"):
            assert phrase in GUIDE_HTML
    finally:
        window.close()
        app.processEvents()


def test_home_shows_both_slm_cards_beside_a_dominant_camera_on_a_laptop_screen() -> None:
    """The camera is the instrument; the two SLM cards stack beside it."""

    app = _app()
    window = VirtualLabResolutionWindow()
    try:
        window.resize(1280, 713)
        window.show()
        window.set_page(window.PAGE_HOME)
        app.processEvents()
        app.processEvents()
        assert window.home_split.orientation() == Qt.Horizontal
        slm_column, camera_column = window.home_split.sizes()
        assert camera_column > slm_column, "the camera must take the larger share"
        scroll = window.pages.widget(window.PAGE_HOME)
        assert scroll.horizontalScrollBar().maximum() == 0
        assert scroll.verticalScrollBar().maximum() < 120
        # Both quick cards must be laid out, not one pushed off the column.
        for card in window.slm_quick_cards.values():
            assert card.height() > 150
    finally:
        window.close()
        app.processEvents()


def test_worker_signals_reach_overridden_handlers_on_the_gui_thread() -> None:
    """Regression for the intermittent crash.

    PySide6 6.11 runs a queued slot that a subclass overrides on the emitting
    worker thread.  The v3 window overrides _virtual_task_finished,
    _virtual_progress_event and _live_stopped, so those touched widgets from
    worker threads.  They must now always run on the GUI thread.
    """

    import threading
    import time

    from PySide6.QtCore import QObject, QThread, Signal

    from labcontrol.ui.controls import GuiThreadDispatcher

    app = _app()
    window = VirtualLabResolutionWindow()
    seen: list[str] = []
    window._virtual_task_finished = lambda result: seen.append(threading.current_thread().name)

    class Emitter(QObject):
        done = Signal(object)

        def run(self):
            self.done.emit({"kind": "x"})

    thread = QThread()
    emitter = Emitter()
    emitter.moveToThread(thread)
    emitter.done.connect(window._gui_thread_dispatcher().forward("_virtual_task_finished"), Qt.DirectConnection)
    emitter.done.connect(thread.quit)
    thread.started.connect(emitter.run)
    try:
        thread.start()
        deadline = time.monotonic() + 5
        while not seen and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert seen == ["MainThread"]
        assert isinstance(window._gui_thread_dispatcher(), GuiThreadDispatcher)
    finally:
        thread.wait(2000)
        window.close()
        app.processEvents()
