"""Resolution-selection extension for the Virtual Lab experiment builder."""

from __future__ import annotations

import faulthandler
import json
import sys
import traceback
from pathlib import Path

from PySide6.QtWidgets import QApplication, QFormLayout, QLabel, QPushButton

from slm_lab_control.config import AppConfig
from slm_lab_control.ui.style import APP_QSS

from ..devices.camera import DummyCameraProvider
from ..mode_controller import ModeAwareLabController
from ..state import ExperimentState, ExperimentStore
from ..virtual_lab_resolution import ResolutionAwareVirtualBenchEngine, VirtualOutputResolution
from .advanced import ADVANCED_QSS, PROJECT_ROOT, panel
from .controls import NoWheelComboBox
from .virtual_advanced_v2 import VirtualLabExperimentWindow


_FAULT_LOG_HANDLE = None


class VirtualLabResolutionWindow(VirtualLabExperimentWindow):
    """Full Virtual Lab GUI with explicit detector/model resolution choices."""

    def __init__(self, store=None, controller=None, calibration=None):
        if controller is None:
            initial = ExperimentState.from_app_config(AppConfig())
            initial.system.experiment_label = "Optical lab session"
            initial.camera.shape_yx = (512, 512)
            store = store or ExperimentStore(initial)
            camera = DummyCameraProvider(store.snapshot, shape_yx=(512, 512))
            engine = ResolutionAwareVirtualBenchEngine()
            controller = ModeAwareLabController(
                store,
                PROJECT_ROOT,
                camera_provider=camera,
                virtual_engine=engine,
            )
        elif not isinstance(controller.virtual_engine, ResolutionAwareVirtualBenchEngine):
            raise TypeError(
                "VirtualLabResolutionWindow requires ResolutionAwareVirtualBenchEngine so output sampling "
                "and optical-grid provenance remain authoritative."
            )
        super().__init__(store=store, controller=controller, calibration=calibration)

    @property
    def resolution_engine(self) -> ResolutionAwareVirtualBenchEngine:
        engine = self.mode_controller.virtual_engine
        if not isinstance(engine, ResolutionAwareVirtualBenchEngine):
            raise TypeError("Resolution-aware virtual engine is not installed.")
        return engine

    def _build_virtual_page(self) -> None:
        super()._build_virtual_page()
        if self.virtual_quality.findText("maximum") < 0:
            self.virtual_quality.addItem("maximum")

        page = self.pages.widget(self.PAGE_VIRTUAL)
        layout = page.layout()
        if layout is None:
            raise RuntimeError("Virtual Lab page has no layout.")

        card, box = panel(
            "Virtual camera resolution",
            "Choose Beamage pixel-count equivalence or retain the higher-resolution numerical model. Physical Beamage FOV/pixel pitch are not yet calibrated, so 2048×2048 here is a pixel-count match only.",
        )
        form = QFormLayout()
        self.virtual_output_resolution = NoWheelComboBox()
        self.virtual_output_resolution.addItems([item.value for item in VirtualOutputResolution])
        self.virtual_output_resolution.setCurrentText(self.resolution_engine.output_resolution.value)
        form.addRow("Output / propagation mode", self.virtual_output_resolution)
        box.addLayout(form)

        apply_button = QPushButton("Apply resolution mode")
        apply_button.setObjectName("Accent")
        apply_button.clicked.connect(self._apply_virtual_output_resolution)
        box.addWidget(apply_button)

        self.virtual_resolution_status = QLabel()
        self.virtual_resolution_status.setObjectName("Muted")
        self.virtual_resolution_status.setWordWrap(True)
        box.addWidget(self.virtual_resolution_status)

        warning = QLabel(
            "MAXIMUM MODEL uses a 4096×4096 propagation/output grid and is computationally expensive. "
            "MAX MODEL → BEAMAGE 4M uses that same 4096×4096 propagation, then 2×2 area-integrates to 2048×2048. "
            "That paired comparison is the cleanest way to isolate virtual detector sampling from propagation-grid resolution."
        )
        warning.setObjectName("WarnChip")
        warning.setWordWrap(True)
        box.addWidget(warning)

        # Put resolution immediately before the existing mock camera/correction body.
        insert_at = max(2, layout.count() - 2)
        layout.insertWidget(insert_at, card)
        self._refresh_virtual_resolution_status()

    def _refresh_virtual_resolution_status(self) -> None:
        summary = self.resolution_engine.resolution_summary()
        self.virtual_resolution_status.setText(
            f"Propagation grid: {summary['propagation_grid_yx'][1]}×{summary['propagation_grid_yx'][0]} • "
            f"camera frame: {summary['output_frame_yx'][1]}×{summary['output_frame_yx'][0]}\n"
            f"{summary['physical_sampling_status']}"
        )

    def _apply_virtual_output_resolution(self) -> None:
        try:
            self._ensure_virtual_mode()
            mode = VirtualOutputResolution(self.virtual_output_resolution.currentText())
            self.resolution_engine.set_output_resolution(mode)
            if mode in {VirtualOutputResolution.MAXIMUM, VirtualOutputResolution.MAXIMUM_TO_BEAMAGE}:
                self.virtual_quality.setCurrentText("maximum")
            elif mode is VirtualOutputResolution.BEAMAGE_4M:
                # Pixel-count-equivalent propagation at the real camera's nominal
                # 4M shape.  This is deliberately not called a physical pixel-pitch match.
                if self.virtual_quality.currentText() == "maximum":
                    self.virtual_quality.setCurrentText("validation")

            def update_camera_state(state):
                state.camera.shape_yx = self.resolution_engine.shape_yx
                state.camera.pixel_size_um = self.resolution_engine.pixel_size_um
                state.camera.frame_quality = (
                    "SIMULATED_NUMERICAL_INTENSITY • " + self.resolution_engine.output_resolution.value
                )

            self.store.update(
                update_camera_state,
                source="virtual_resolution",
                reason=f"Virtual output resolution changed to {mode.value}",
            )
            self.current_frame = None
            self.current_metrics = None
            self._refresh_virtual_resolution_status()
            self.virtual_progress.appendPlainText(
                "\nResolution mode changed. Capture a fresh stack before comparing metrics.\n"
                + json.dumps(self.resolution_engine.resolution_summary(), indent=2)
            )
        except Exception as exc:
            self._show_error("Could not change virtual resolution", exc)


def main() -> int:
    global _FAULT_LOG_HANDLE
    crash_path = Path.cwd() / "lab_gui_crash.log"
    _FAULT_LOG_HANDLE = crash_path.open("a", encoding="utf-8", buffering=1)
    faulthandler.enable(_FAULT_LOG_HANDLE, all_threads=True)

    def log_uncaught(error_type, error, error_traceback) -> None:
        print("\n--- uncaught GUI exception ---", file=_FAULT_LOG_HANDLE)
        traceback.print_exception(error_type, error, error_traceback, file=_FAULT_LOG_HANDLE)
        sys.__excepthook__(error_type, error, error_traceback)

    sys.excepthook = log_uncaught
    application = QApplication.instance() or QApplication(sys.argv)
    application.setStyleSheet(APP_QSS + ADVANCED_QSS)
    window = VirtualLabResolutionWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
