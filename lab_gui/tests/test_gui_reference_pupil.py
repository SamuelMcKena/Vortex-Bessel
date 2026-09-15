from PySide6.QtWidgets import QApplication

from slm_lab_control.app import SLMControlPanel
from slm_lab_control.config import SlmPhaseConfig


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_reference_pupil_is_an_always_available_setting_not_a_phase_switch():
    _application()
    cfg = SlmPhaseConfig(name="SLM1")
    cfg.switches.circular_pupil = True  # Simulate a config made by the old GUI.

    panel = SLMControlPanel(cfg)

    assert panel.grp_pupil.isCheckable() is False
    assert panel.pupil_mm.isEnabled() is True
    assert "never masks the SLM" in panel.grp_pupil.title()
    assert "never blanks pixels" in panel.pupil_note.text()
    assert panel.to_config("SLM1").switches.circular_pupil is False
