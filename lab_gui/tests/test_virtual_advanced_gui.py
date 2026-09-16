from __future__ import annotations

from PySide6.QtWidgets import QApplication

from labcontrol.ui.virtual_advanced import VirtualLabWindow
from labcontrol.virtual_lab import OperatingMode, VirtualSlmProvider


def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_virtual_gui_preserves_home_cockpit_and_adds_tenth_page() -> None:
    qt = app()
    window = VirtualLabWindow()
    try:
        assert window.pages.count() == 10
        assert window.pages.currentIndex() == window.PAGE_HOME
        assert window.home_camera_view is not None
        assert set(window.slm_quick_cards) == {"SLM1", "SLM2"}
        assert window.pages.widget(window.PAGE_VIRTUAL) is not None
        assert window.mode_choice.currentText() == OperatingMode.LIVE_LAB.value
        assert window.home_provider_choice.findText("virtual") >= 0
        assert window.camera_provider_choice.findText("virtual") >= 0
    finally:
        window.close()
        qt.processEvents()


def test_virtual_mode_installs_only_virtual_slm_target() -> None:
    qt = app()
    window = VirtualLabWindow()
    try:
        # Even if the normal application backend were configured for HEDS, the
        # explicit Virtual Lab mode must override the cast target.
        window.store.update(
            lambda state: setattr(state.application, "backend", "heds"),
            source="test",
            reason="Pretend real backend was selected",
        )
        window.mode_choice.setCurrentText(OperatingMode.VIRTUAL_LAB.value)
        for _ in range(4):
            qt.processEvents()
        assert window.mode_controller.operating_mode is OperatingMode.VIRTUAL_LAB
        assert window.controller.camera_provider is not None
        assert window.controller.camera_provider.name == "virtual"
        assert isinstance(window.controller.slm_provider, VirtualSlmProvider)
        assert "SIMULATED" in window.mode_badge.text()
    finally:
        window.close()
        qt.processEvents()


def test_recorded_mode_is_visibly_read_only() -> None:
    qt = app()
    window = VirtualLabWindow()
    try:
        window.mode_choice.setCurrentText(OperatingMode.RECORDED_LAB.value)
        for _ in range(4):
            qt.processEvents()
        assert window.mode_controller.operating_mode is OperatingMode.RECORDED_LAB
        assert "READ" in window.mode_badge.text().upper()
        assert window.slm_quick_cards["SLM1"].connection.text() == "READ ONLY"
        assert window.slm_quick_cards["SLM2"].connection.text() == "READ ONLY"
    finally:
        window.close()
        qt.processEvents()
