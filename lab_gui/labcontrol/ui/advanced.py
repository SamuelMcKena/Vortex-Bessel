"""Unified, state-driven lab operator interface."""

from __future__ import annotations

import json
import faulthandler
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from slm_lab_control.config import AppConfig
from slm_lab_control.presets import load_preset
from slm_lab_control.ui.style import APP_QSS

from ..calibration import (
    CalibrationKey,
    CalibrationRegistry,
    CalibrationStatus,
    PhysicalChange,
)
from ..beam_walk import PlaneObservation, fit_beam_walk
from ..capture import FormalCaptureService, SessionRepository
from ..controller import LabController
from ..devices.camera import (
    BeamageCameraProvider,
    CameraFrame,
    DummyCameraProvider,
    ReplayCameraProvider,
)
from ..devices.beamage_pipe import focus_pc_beamage_window
from ..metrics import BeamMetrics, MetricEngine
from ..recipes import RecipeEngine, RecipeRun, builtin_recipes
from ..state import AcquisitionState, ConnectionState, ExperimentState, ExperimentStore, PhysicalAxiconState
from .camera_worker import CameraAcquisitionWorker, stop_worker_thread
from .beam_walk_view import BeamWalkPlot
from .controls import NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSpinBox, blocked_signals
from .image_view import QuantitativeImageView, render_preview
from .slm_views import SlmDetailView, SlmQuickCard


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FAULT_LOG_HANDLE = None


ADVANCED_QSS = """
QMainWindow, QWidget { background: #0b1218; color: #e8f0f5; }
#LabSidebar { background: #081016; border-right: 1px solid #20313d; }
#LabBrand { color: #75e6c4; font-size: 23px; font-weight: 800; letter-spacing: 1px; }
QPushButton#LabNav { text-align: left; padding: 11px 13px; border: 0; border-radius: 7px; color: #aebdc7; }
QPushButton#LabNav:checked { background: #173342; color: #ffffff; font-weight: 700; }
QPushButton#Accent { background: #27b58b; color: #06120f; font-weight: 800; padding: 9px 14px; border-radius: 6px; }
QPushButton#Danger { background: #482630; color: #ffb6c0; border: 1px solid #7a3445; padding: 8px 12px; }
#Panel { background: #101c24; border: 1px solid #263a47; border-radius: 9px; }
#Title { font-size: 24px; font-weight: 750; color: #f2f7fa; }
#Section { font-size: 15px; font-weight: 750; color: #8ce5cd; }
#Muted { color: #8fa2ae; }
#GoodChip { background: #14392f; color: #78efc2; padding: 6px 10px; border-radius: 9px; font-weight: 700; }
#WarnChip { background: #422f19; color: #ffd58a; padding: 6px 10px; border-radius: 9px; font-weight: 700; }
#BadChip { background: #482630; color: #ff9cac; padding: 6px 10px; border-radius: 9px; font-weight: 700; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget, QPlainTextEdit, QListWidget {
  background: #0b151c; border: 1px solid #314653; border-radius: 5px; padding: 5px; selection-background-color: #277f70;
}
QHeaderView::section { background: #152630; color: #bcd0da; padding: 7px; border: 0; }
"""


def panel(title: str, subtitle: str = "") -> tuple[QFrame, QVBoxLayout]:
    card = QFrame()
    card.setObjectName("Panel")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(15, 14, 15, 14)
    heading = QLabel(title)
    heading.setObjectName("Section")
    layout.addWidget(heading)
    if subtitle:
        note = QLabel(subtitle)
        note.setObjectName("Muted")
        note.setWordWrap(True)
        layout.addWidget(note)
    return card, layout


class StateSignalBridge(QObject):
    changed = Signal(object, object)


class AdvancedLabWindow(QMainWindow):
    PAGE_HOME = 0
    PAGE_SLM1 = 1
    PAGE_SLM2 = 2
    PAGE_MEASURE = 3
    PAGE_RECOVER = 4
    PAGE_SESSIONS = 5
    PAGE_PRESETS = 6
    PAGE_CALIBRATION = 7
    PAGE_SYSTEM = 8

    def __init__(
        self,
        store: ExperimentStore | None = None,
        controller: LabController | None = None,
        calibration: CalibrationRegistry | None = None,
    ):
        super().__init__()
        self.setWindowTitle("Unified Optical Lab Control")
        self.resize(1680, 1020)
        if controller is None:
            initial = ExperimentState.from_app_config(AppConfig())
            initial.system.experiment_label = "Optical lab session"
            initial.camera.shape_yx = (512, 512)
            store = store or ExperimentStore(initial)
            camera = DummyCameraProvider(store.snapshot, shape_yx=(512, 512))
            controller = LabController(store, PROJECT_ROOT, camera_provider=camera)
        self.controller = controller
        self.store = controller.store
        if self.controller.camera_provider is None:
            self.controller.set_camera_provider(
                DummyCameraProvider(self.store.snapshot, shape_yx=(512, 512))
            )
        self.calibration = calibration or CalibrationRegistry()
        self.metric_engine = MetricEngine()
        self.recipe_engine = RecipeEngine(self.store, self.calibration)
        self.recipe_run: RecipeRun | None = None
        self.current_frame: CameraFrame | None = None
        self.current_metrics: BeamMetrics | None = None
        self._frame_counter = 0
        self._camera_thread: QThread | None = None
        self._camera_worker: CameraAcquisitionWorker | None = None
        self._compact_window = None
        self._history: list[dict[str, Any]] = []
        self._syncing = False
        self._build()
        self._state_bridge = StateSignalBridge(self)
        self._state_bridge.changed.connect(self._on_state_change)
        self._unsubscribe = self.store.subscribe(
            lambda event, state: self._state_bridge.changed.emit(event, state)
        )
        try:
            # Populate both Home cards immediately.  This is computation only;
            # opening the GUI never connects to or casts to physical hardware.
            self.controller.generate()
        except Exception as exc:
            self._home_message(f"Initial phase preview unavailable: {exc}")
        self._refresh_state(self.store.snapshot())
        self._refresh_presets()
        self._refresh_recipe_description()

    # ------------------------------------------------------------------ layout

    def _build(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("LabSidebar")
        sidebar.setFixedWidth(255)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(16, 20, 16, 18)
        brand = QLabel("LAB CONTROL")
        brand.setObjectName("LabBrand")
        side.addWidget(brand)
        description = QLabel("state → image → evidence")
        description.setObjectName("Muted")
        side.addWidget(description)
        side.addSpacing(20)
        self.pages = QStackedWidget()
        self.nav_buttons = []
        for title, index in (
            ("Home", self.PAGE_HOME),
            ("SLM 1", self.PAGE_SLM1),
            ("SLM 2", self.PAGE_SLM2),
            ("Measure / camera", self.PAGE_MEASURE),
            ("Optimise / recover", self.PAGE_RECOVER),
            ("Sessions / compare", self.PAGE_SESSIONS),
            ("Presets", self.PAGE_PRESETS),
            ("Calibration / readiness", self.PAGE_CALIBRATION),
            ("System", self.PAGE_SYSTEM),
        ):
            button = QPushButton(title)
            button.setObjectName("LabNav")
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, page=index: self.set_page(page))
            self.nav_buttons.append(button)
            side.addWidget(button)
        side.addStretch(1)
        self.sidebar_status = QLabel("Camera: stopped\nSLMs: disconnected")
        self.sidebar_status.setObjectName("Muted")
        self.sidebar_status.setWordWrap(True)
        side.addWidget(self.sidebar_status)
        root.addWidget(sidebar)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(20, 14, 20, 18)
        header = QHBoxLayout()
        titles = QVBoxLayout()
        self.page_title = QLabel("System / readiness")
        self.page_title.setObjectName("Title")
        self.optical_summary = QLabel()
        self.optical_summary.setObjectName("Muted")
        titles.addWidget(self.page_title)
        titles.addWidget(self.optical_summary)
        header.addLayout(titles, 1)
        self.source_chip = QLabel("SYNTHETIC")
        self.source_chip.setObjectName("WarnChip")
        self.readiness_chip = QLabel("NOT READY")
        self.readiness_chip.setObjectName("BadChip")
        header.addWidget(self.source_chip)
        header.addWidget(self.readiness_chip)
        body_layout.addLayout(header)
        body_layout.addWidget(self.pages, 1)
        root.addWidget(body, 1)

        self._build_home_page()
        self._build_slm_detail_page("SLM1")
        self._build_slm_detail_page("SLM2")
        self._build_measure_page()
        self._build_recover_page()
        self._build_sessions_page()
        self._build_presets_page()
        self._build_system_page()
        self._build_system_info_page()
        self.set_page(self.PAGE_HOME)

    def _page(self, title: str, subtitle: str) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        heading = QLabel(title)
        heading.setObjectName("Section")
        note = QLabel(subtitle)
        note.setObjectName("Muted")
        note.setWordWrap(True)
        layout.addWidget(heading)
        layout.addWidget(note)
        self.pages.addWidget(page)
        return page, layout

    def _build_system_page(self) -> None:
        _, layout = self._page(
            "System readiness",
            "Readiness is derived from recorded calibration dependencies. Unknown is not silently treated as valid.",
        )
        split = QSplitter(Qt.Horizontal)
        calibration_card, calibration_layout = panel("Calibration dependency state")
        self.calibration_table = QTableWidget(0, 3)
        self.calibration_table.setHorizontalHeaderLabels(["Calibration", "Status", "Reason / evidence"])
        self.calibration_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.calibration_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.calibration_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        calibration_layout.addWidget(self.calibration_table)
        registry_row = QHBoxLayout()
        load = QPushButton("Load registry…")
        save = QPushButton("Save registry…")
        load.clicked.connect(self._load_calibration)
        save.clicked.connect(self._save_calibration)
        registry_row.addWidget(load)
        registry_row.addWidget(save)
        calibration_layout.addLayout(registry_row)
        split.addWidget(calibration_card)

        actions_card, actions = panel(
            "Physical change event",
            "Report what physically changed; only its dependency closure becomes stale.",
        )
        self.physical_change = NoWheelComboBox()
        self.physical_change.addItems([change.value for change in PhysicalChange])
        actions.addWidget(self.physical_change)
        report = QPushButton("Record change and invalidate dependencies")
        report.setObjectName("Danger")
        report.clicked.connect(self._report_physical_change)
        actions.addWidget(report)
        self.readiness_detail = QPlainTextEdit()
        self.readiness_detail.setReadOnly(True)
        actions.addWidget(self.readiness_detail, 1)
        hardware = QPushButton("Connect both SLMs")
        hardware.setObjectName("Accent")
        hardware.clicked.connect(self._connect_slms)
        actions.addWidget(hardware)
        close = QPushButton("Close SLM provider")
        close.clicked.connect(self._close_slms)
        actions.addWidget(close)
        split.addWidget(actions_card)
        split.setSizes([1000, 520])
        layout.addWidget(split, 1)

    def _build_home_page(self) -> None:
        _, layout = self._page(
            "Lab cockpit",
            "Immediate SLM command state on the left; the camera is the dominant live instrument view on the right.",
        )
        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)
        self.slm_quick_cards: dict[str, SlmQuickCard] = {}
        for name, page in (("SLM1", self.PAGE_SLM1), ("SLM2", self.PAGE_SLM2)):
            card = SlmQuickCard(name, self.controller)
            card.details_requested.connect(lambda _name, target=page: self.set_page(target))
            card.preset_requested.connect(lambda: self.set_page(self.PAGE_PRESETS))
            card.operation_message.connect(self._home_message)
            self.slm_quick_cards[name] = card
            left_layout.addWidget(card, 1)
        split.addWidget(left)

        camera_card, camera_box = panel("Live camera", "The same acquired frame feeds Home, Measure, metrics and storage.")
        self.home_camera_view = QuantitativeImageView()
        self.home_camera_view.setMinimumSize(650, 500)
        camera_box.addWidget(self.home_camera_view, 1)
        camera_row = QHBoxLayout()
        self.home_provider_choice = NoWheelComboBox()
        self.home_provider_choice.addItems(["dummy", "replay", "beamage"])
        self.home_provider_choice.currentTextChanged.connect(self._select_camera_provider)
        connect = QPushButton("Connect")
        connect.clicked.connect(self._connect_camera)
        start = QPushButton("Start live")
        start.setObjectName("Accent")
        start.clicked.connect(self.start_live)
        stop = QPushButton("Stop")
        stop.clicked.connect(self.stop_live)
        capture = QPushButton("Formal capture")
        capture.clicked.connect(self._formal_capture)
        camera_row.addWidget(QLabel("Provider"))
        camera_row.addWidget(self.home_provider_choice)
        camera_row.addWidget(connect)
        camera_row.addWidget(start)
        camera_row.addWidget(stop)
        camera_row.addWidget(capture)
        camera_box.addLayout(camera_row)
        display_row = QHBoxLayout()
        self.home_colour_mode = NoWheelComboBox()
        self.home_colour_mode.addItems(["inferno", "gentec-like", "turbo", "viridis", "grayscale"])
        self.home_display_scale = NoWheelComboBox()
        self.home_display_scale.addItems(["percentile", "log", "sensor range", "full range"])
        self.home_display_gamma = NoWheelDoubleSpinBox()
        self.home_display_gamma.setRange(0.1, 5.0)
        self.home_display_gamma.setValue(1.0)
        self.home_display_gamma.setSingleStep(0.1)
        for widget in (self.home_colour_mode, self.home_display_scale, self.home_display_gamma):
            if isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(self._rerender)
            else:
                widget.valueChanged.connect(self._rerender)
        fit_beam = QPushButton("Fit beam")
        fit_beam.setToolTip("Zoom to the signal without cropping or resampling the camera frame.")
        fit_beam.clicked.connect(self.home_camera_view.fit_signal)
        reset = QPushButton("Full frame")
        reset.clicked.connect(self.home_camera_view.fit_full_frame)
        zoom_in = QPushButton("+")
        zoom_in.setFixedWidth(32)
        zoom_in.clicked.connect(self.home_camera_view.zoom_in)
        zoom_out = QPushButton("−")
        zoom_out.setFixedWidth(32)
        zoom_out.clicked.connect(self.home_camera_view.zoom_out)
        self.home_auto_fit = QCheckBox("Auto-fit beam")
        self.home_auto_fit.setChecked(True)
        display_row.addWidget(QLabel("Colour"))
        display_row.addWidget(self.home_colour_mode)
        display_row.addWidget(QLabel("Scale"))
        display_row.addWidget(self.home_display_scale)
        display_row.addWidget(QLabel("Gamma"))
        display_row.addWidget(self.home_display_gamma)
        display_row.addStretch(1)
        display_row.addWidget(self.home_auto_fit)
        display_row.addWidget(fit_beam)
        display_row.addWidget(reset)
        display_row.addWidget(zoom_out)
        display_row.addWidget(zoom_in)
        camera_box.addLayout(display_row)
        self.home_camera_status = QLabel("Camera disconnected")
        self.home_camera_status.setObjectName("Muted")
        self.home_camera_status.setWordWrap(True)
        self.home_metric_summary = QLabel("No live frame")
        self.home_metric_summary.setObjectName("Muted")
        self.home_metric_summary.setWordWrap(True)
        self.home_operation_status = QLabel("")
        self.home_operation_status.setObjectName("Muted")
        self.home_operation_status.setWordWrap(True)
        camera_box.addWidget(self.home_camera_status)
        camera_box.addWidget(self.home_metric_summary)
        camera_box.addWidget(self.home_operation_status)
        split.addWidget(camera_card)
        split.setSizes([520, 980])
        layout.addWidget(split, 1)

    def _build_slm_detail_page(self, name: str) -> None:
        _, layout = self._page(
            f"{name} complete control",
            "Full legacy-parity phase control backed by the authoritative ExperimentState and the hardware phase composer.",
        )
        view = SlmDetailView(name, self.controller, PROJECT_ROOT)
        view.operation_message.connect(self._home_message)
        if not hasattr(self, "slm_detail_views"):
            self.slm_detail_views: dict[str, SlmDetailView] = {}
        self.slm_detail_views[name] = view
        layout.addWidget(view, 1)

    def _build_system_info_page(self) -> None:
        _, layout = self._page(
            "System and hardware",
            "Choose hardware routes explicitly. Editing configuration never casts automatically.",
        )
        row = QHBoxLayout()
        slm_card, slm_box = panel("SLM hardware route")
        form = QFormLayout()
        self.system_backend = NoWheelComboBox()
        self.system_backend.addItems(["dummy", "heds"])
        self.system_transfer = NoWheelComboBox()
        self.system_transfer.addItems(["png_file", "phase_file", "direct_gray_array", "direct_phase_array", "auto"])
        form.addRow("Backend", self.system_backend)
        form.addRow("Transfer mode", self.system_transfer)
        slm_box.addLayout(form)
        slm_buttons = QHBoxLayout()
        connect = QPushButton("Connect both SLMs")
        connect.setObjectName("Accent")
        connect.clicked.connect(self._connect_slms)
        close = QPushButton("Disconnect")
        close.clicked.connect(self._close_slms)
        slm_buttons.addWidget(connect)
        slm_buttons.addWidget(close)
        slm_box.addLayout(slm_buttons)
        row.addWidget(slm_card, 1)

        context_card, context_box = panel("Physical measurement context")
        form = QFormLayout()
        self.axicon_state = NoWheelComboBox()
        self.axicon_state.addItems([item.value for item in PhysicalAxiconState])
        self.current_z = NoWheelDoubleSpinBox()
        self.current_z.setRange(-10000.0, 10000.0)
        self.current_z.setDecimals(3)
        self.current_z.setSuffix(" mm")
        self.current_z.setKeyboardTracking(False)
        self.z_reference = QLineEdit("UNSET")
        form.addRow("Physical axicon", self.axicon_state)
        form.addRow("Camera z", self.current_z)
        form.addRow("z reference", self.z_reference)
        context_box.addLayout(form)
        row.addWidget(context_card, 1)
        layout.addLayout(row)
        self.system_backend.currentTextChanged.connect(self._apply_system_state)
        self.system_transfer.currentTextChanged.connect(self._apply_system_state)
        self.axicon_state.currentTextChanged.connect(self._apply_system_state)
        self.current_z.editingFinished.connect(self._apply_system_state)
        self.z_reference.editingFinished.connect(self._apply_system_state)
        diagnostics, diagnostics_box = panel(
            "Hardware diagnostics",
            "The GUI remains available without HEDS or PC-Beamage. Run tools/test_beamage_pipe.py before first camera use.",
        )
        self.system_detail = QPlainTextEdit()
        self.system_detail.setReadOnly(True)
        diagnostics_box.addWidget(self.system_detail)
        layout.addWidget(diagnostics, 1)

    def _build_measure_page(self) -> None:
        _, layout = self._page(
            "Quantitative camera",
            "One acquired matrix feeds display, metrics and formal storage. Colour, log, gamma and zoom are display-only.",
        )
        causal = QFrame()
        causal.setObjectName("Panel")
        causal_row = QHBoxLayout(causal)
        self.causal_phase = QLabel("PHASE STATE")
        self.causal_image = QLabel("QUANTITATIVE FRAME")
        self.causal_metrics = QLabel("STATE-AWARE METRICS")
        for index, widget in enumerate((self.causal_phase, self.causal_image, self.causal_metrics)):
            widget.setAlignment(Qt.AlignCenter)
            widget.setObjectName("Section" if index != 1 else "GoodChip")
            causal_row.addWidget(widget, 1)
            if index < 2:
                causal_row.addWidget(QLabel("→"))
        layout.addWidget(causal)

        split = QSplitter(Qt.Horizontal)
        image_card, image_layout = panel("Live beam image")
        self.image_view = QuantitativeImageView()
        image_layout.addWidget(self.image_view, 1)
        view_row = QHBoxLayout()
        self.colour_mode = NoWheelComboBox()
        self.colour_mode.addItems(["inferno", "gentec-like", "turbo", "viridis", "grayscale"])
        self.display_scale = NoWheelComboBox()
        self.display_scale.addItems(["percentile", "log", "sensor range", "full range"])
        self.display_gamma = NoWheelDoubleSpinBox()
        self.display_gamma.setRange(0.1, 5.0)
        self.display_gamma.setValue(1.0)
        self.display_gamma.setSingleStep(0.1)
        fit_beam = QPushButton("Fit beam")
        fit_beam.setToolTip("Zoom to the signal without cropping or resampling the camera frame.")
        fit_beam.clicked.connect(self.image_view.fit_signal)
        reset = QPushButton("Full frame")
        reset.clicked.connect(self.image_view.fit_full_frame)
        zoom_in = QPushButton("+")
        zoom_in.setFixedWidth(32)
        zoom_in.clicked.connect(self.image_view.zoom_in)
        zoom_out = QPushButton("−")
        zoom_out.setFixedWidth(32)
        zoom_out.clicked.connect(self.image_view.zoom_out)
        self.auto_fit = QCheckBox("Auto-fit beam")
        self.auto_fit.setChecked(True)
        for widget in (self.colour_mode, self.display_scale, self.display_gamma):
            if isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(self._rerender)
            else:
                widget.valueChanged.connect(self._rerender)
        view_row.addWidget(QLabel("Colour"))
        view_row.addWidget(self.colour_mode)
        view_row.addWidget(QLabel("Scale"))
        view_row.addWidget(self.display_scale)
        view_row.addWidget(QLabel("Gamma"))
        view_row.addWidget(self.display_gamma)
        view_row.addStretch(1)
        view_row.addWidget(self.auto_fit)
        view_row.addWidget(fit_beam)
        view_row.addWidget(reset)
        view_row.addWidget(zoom_out)
        view_row.addWidget(zoom_in)
        image_layout.addLayout(view_row)
        overlay_row = QHBoxLayout()
        self.overlay_centre = QCheckBox("Detected centre")
        self.overlay_ring = QCheckBox("Principal ring / dark core")
        self.overlay_roi = QCheckBox("Analysis ROI")
        for checkbox in (self.overlay_centre, self.overlay_ring, self.overlay_roi):
            checkbox.setChecked(True)
            checkbox.toggled.connect(self._rerender)
            overlay_row.addWidget(checkbox)
        overlay_row.addStretch(1)
        image_layout.addLayout(overlay_row)
        split.addWidget(image_card)

        controls = QWidget()
        controls.setMinimumWidth(390)
        controls.setMaximumWidth(480)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(6, 0, 0, 0)
        camera_card, camera_box = panel("Camera provider")
        form = QFormLayout()
        self.camera_provider_choice = NoWheelComboBox()
        self.camera_provider_choice.addItems(["dummy", "replay", "beamage"])
        self.camera_provider_choice.currentTextChanged.connect(self._select_camera_provider)
        self.camera_status = QLabel("Disconnected")
        self.camera_status.setWordWrap(True)
        self.camera_status.setObjectName("Muted")
        self.exposure = NoWheelDoubleSpinBox()
        self.exposure.setRange(1.0, 10_000_000.0)
        self.exposure.setValue(1000.0)
        self.exposure.setSuffix(" µs")
        self.gain = NoWheelDoubleSpinBox()
        self.gain.setRange(0.0, 100.0)
        self.gain.setValue(0.0)
        form.addRow("Provider", self.camera_provider_choice)
        form.addRow("Status", self.camera_status)
        form.addRow("Exposure readback", self.exposure)
        form.addRow("Gain", self.gain)
        camera_box.addLayout(form)
        self.camera_control_note = QLabel("")
        self.camera_control_note.setObjectName("Muted")
        self.camera_control_note.setWordWrap(True)
        camera_box.addWidget(self.camera_control_note)
        provider_row = QHBoxLayout()
        browse = QPushButton("Choose replay…")
        browse.clicked.connect(self._browse_replay)
        connect = QPushButton("Connect")
        connect.clicked.connect(self._connect_camera)
        self.pc_beamage_controls = QPushButton("PC-Beamage controls")
        self.pc_beamage_controls.clicked.connect(self._focus_beamage_controls)
        provider_row.addWidget(browse)
        provider_row.addWidget(connect)
        provider_row.addWidget(self.pc_beamage_controls)
        camera_box.addLayout(provider_row)
        live_row = QHBoxLayout()
        self.start_live_button = QPushButton("Start live")
        self.start_live_button.setObjectName("Accent")
        self.stop_live_button = QPushButton("Stop")
        self.start_live_button.clicked.connect(self.start_live)
        self.stop_live_button.clicked.connect(self.stop_live)
        live_row.addWidget(self.start_live_button)
        live_row.addWidget(self.stop_live_button)
        camera_box.addLayout(live_row)
        controls_layout.addWidget(camera_card)

        metric_card, metric_box = panel("Live metrics", "Heavy analysis updates at a lower rate than display.")
        self.saturation_warning = QLabel("No frame")
        self.saturation_warning.setObjectName("Muted")
        self.saturation_warning.setWordWrap(True)
        self.live_metrics = QPlainTextEdit()
        self.live_metrics.setReadOnly(True)
        self.live_metrics.setMaximumHeight(215)
        metric_box.addWidget(self.saturation_warning)
        metric_box.addWidget(self.live_metrics)
        controls_layout.addWidget(metric_card)

        capture_card, capture_box = panel("Formal capture", "Fresh, cast-verified, numeric and provenance-linked.")
        capture_form = QFormLayout()
        self.trial_id = QLineEdit("trial_0001")
        self.capture_role = NoWheelComboBox()
        self.capture_role.addItems(["CURRENT", "BASELINE", "ACCEPTED", "VERIFICATION"])
        self.capture_repeats = NoWheelSpinBox()
        self.capture_repeats.setRange(1, 100)
        self.capture_repeats.setValue(3)
        self.settle_s = NoWheelDoubleSpinBox()
        self.settle_s.setRange(0.0, 60.0)
        self.settle_s.setValue(0.1)
        self.settle_s.setSuffix(" s")
        self.session_path = QLineEdit(str(PROJECT_ROOT / "outputs" / "sessions" / "active_session"))
        capture_form.addRow("Trial", self.trial_id)
        capture_form.addRow("Role", self.capture_role)
        capture_form.addRow("Repeats", self.capture_repeats)
        capture_form.addRow("Settle", self.settle_s)
        capture_form.addRow("Session folder", self.session_path)
        capture_box.addLayout(capture_form)
        capture = QPushButton("Cast state + capture fresh frames")
        capture.setObjectName("Accent")
        capture.clicked.connect(self._formal_capture)
        capture_box.addWidget(capture)
        controls_layout.addWidget(capture_card)
        controls_layout.addStretch(1)
        split.addWidget(controls)
        split.setSizes([1120, 420])
        layout.addWidget(split, 1)

    def _build_recover_page(self) -> None:
        _, layout = self._page(
            "Optimise & recover",
            "Recipe steps live in the headless core. Commands remain provisional until fresh verification passes.",
        )
        row = QHBoxLayout()
        recipe_card, recipe_box = panel("Recipe catalogue")
        self.recipe_choice = NoWheelComboBox()
        self.recipe_choice.addItems(sorted(builtin_recipes()))
        self.recipe_choice.setCurrentText("recover_q20")
        self.recipe_choice.currentTextChanged.connect(self._refresh_recipe_description)
        self.recipe_description = QLabel()
        self.recipe_description.setObjectName("Muted")
        self.recipe_description.setWordWrap(True)
        self.recipe_steps = QListWidget()
        recipe_box.addWidget(self.recipe_choice)
        recipe_box.addWidget(self.recipe_description)
        recipe_box.addWidget(self.recipe_steps, 1)
        start = QPushButton("Start selected recipe")
        start.setObjectName("Accent")
        start.clicked.connect(self._start_recipe)
        recipe_box.addWidget(start)
        row.addWidget(recipe_card, 1)

        progress_card, progress = panel("Current run")
        self.recipe_status = QLabel("No active recipe")
        self.recipe_status.setObjectName("Muted")
        self.recipe_instruction = QLabel("Select a recipe to inspect its steps.")
        self.recipe_instruction.setWordWrap(True)
        self.recipe_log = QPlainTextEdit()
        self.recipe_log.setReadOnly(True)
        advance = QPushButton("Confirm step complete")
        advance.clicked.connect(self._advance_recipe)
        demo = QPushButton("Run synthetic closed-loop demonstration")
        demo.clicked.connect(self._run_demo)
        progress.addWidget(self.recipe_status)
        progress.addWidget(self.recipe_instruction)
        progress.addWidget(self.recipe_log, 1)
        progress.addWidget(advance)
        progress.addWidget(demo)
        row.addWidget(progress_card, 1)
        layout.addLayout(row, 1)

    def _build_sessions_page(self) -> None:
        _, layout = self._page(
            "Recent captures",
            "Compare raw quantitative frames. Difference rendering is derived from raw arrays, never from coloured previews.",
        )
        selectors = QHBoxLayout()
        self.baseline_choice = NoWheelComboBox()
        self.current_choice = NoWheelComboBox()
        self.baseline_choice.currentIndexChanged.connect(self._update_comparison)
        self.current_choice.currentIndexChanged.connect(self._update_comparison)
        selectors.addWidget(QLabel("Baseline"))
        selectors.addWidget(self.baseline_choice, 1)
        selectors.addWidget(QLabel("Current"))
        selectors.addWidget(self.current_choice, 1)
        layout.addLayout(selectors)
        views = QHBoxLayout()
        self.baseline_view = QuantitativeImageView()
        self.current_view = QuantitativeImageView()
        self.difference_view = QuantitativeImageView()
        for title, view in (
            ("BASELINE", self.baseline_view),
            ("CURRENT", self.current_view),
            ("NORMALISED DIFFERENCE", self.difference_view),
        ):
            card, box = panel(title)
            view.setMinimumSize(260, 260)
            box.addWidget(view)
            views.addWidget(card, 1)
        layout.addLayout(views, 1)
        self.walk_plot = BeamWalkPlot()
        layout.addWidget(self.walk_plot)
        self.comparison_metrics = QPlainTextEdit()
        self.comparison_metrics.setReadOnly(True)
        self.comparison_metrics.setMaximumHeight(150)
        layout.addWidget(self.comparison_metrics)

    def _build_presets_page(self) -> None:
        _, layout = self._page(
            "Phase presets",
            "Existing v0.6 JSON presets remain loadable; locked hardware values and the non-gating pupil migration are reapplied.",
        )
        card, box = panel("Preset library")
        self.preset_list = QListWidget()
        self.preset_list.itemDoubleClicked.connect(lambda _item: self._load_selected_preset())
        box.addWidget(self.preset_list, 1)
        buttons = QHBoxLayout()
        load = QPushButton("Load selected")
        load.setObjectName("Accent")
        load.clicked.connect(self._load_selected_preset)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._refresh_presets)
        compact = QPushButton("Open full dual-SLM editor")
        compact.clicked.connect(self._open_compact)
        buttons.addWidget(load)
        buttons.addWidget(refresh)
        buttons.addWidget(compact)
        box.addLayout(buttons)
        layout.addWidget(card, 1)

    # ----------------------------------------------------------- state / status

    def set_page(self, index: int) -> None:
        labels = [
            "Home",
            "SLM 1",
            "SLM 2",
            "Measure / camera",
            "Optimise / recover",
            "Sessions / compare",
            "Presets",
            "Calibration / readiness",
            "System",
        ]
        self.pages.setCurrentIndex(index)
        self.page_title.setText(labels[index])
        for position, button in enumerate(self.nav_buttons):
            button.setChecked(position == index)
        self._rerender()

    def _on_state_change(self, event, state: ExperimentState) -> None:
        refresh_slms: set[str] = set()
        for path in event.changed_paths:
            if path.startswith("application"):
                refresh_slms.update(("SLM1", "SLM2"))
            elif path.startswith("slm1"):
                refresh_slms.add("SLM1")
            elif path.startswith("slm2"):
                refresh_slms.add("SLM2")
        self._refresh_state(state, refresh_slms=refresh_slms)

    def _refresh_state(
        self,
        state: ExperimentState,
        *,
        refresh_slms: set[str] | None = None,
    ) -> None:
        # ``None`` means a complete initial/model refresh.  Event-driven updates
        # pass the exact SLM names that changed so a camera frame or an SLM2
        # status update cannot overwrite text currently being typed into SLM1.
        slms_to_refresh = {"SLM1", "SLM2"} if refresh_slms is None else refresh_slms
        self._syncing = True
        try:
            q = state.effective_vortex_charge()
            contributions = state.active_vortex_contributions()
            self.optical_summary.setText(
                f"q={q:+d} from {contributions or 'no active vortex'}  •  axicon {state.system.physical_axicon.value}  •  "
                f"z={state.camera.current_z_mm if state.camera.current_z_mm is not None else 'unset'} mm  •  revision {state.revision}"
            )
            self.source_chip.setText(state.system.data_kind.value)
            self.source_chip.setObjectName(
                "WarnChip" if state.system.data_kind.value != "EXPERIMENT" else "GoodChip"
            )
            self.source_chip.style().unpolish(self.source_chip)
            self.source_chip.style().polish(self.source_chip)
            self.causal_phase.setText(f"PHASE STATE\nq={q:+d} • rev {state.revision}")
            self.causal_image.setText(
                f"QUANTITATIVE FRAME\n{state.camera.last_frame_id or 'waiting'}"
            )
            self.causal_metrics.setText(f"STATE-AWARE METRICS\n{state.analysis_family()}")
            self.sidebar_status.setText(
                f"Camera: {state.camera.connection.value.lower()} / {state.camera.acquisition.value.lower()}\n"
                f"SLM1: {state.slm1.connection.value.lower()}\nSLM2: {state.slm2.connection.value.lower()}"
            )
            self.camera_status.setText(
                f"{state.camera.connection.value} • {state.camera.implementation_status}"
                + (f"\n{state.camera.last_error}" if state.camera.last_error else "")
            )
            camera_widgets = (
                self.exposure,
                self.gain,
                self.camera_provider_choice,
                self.home_provider_choice,
                self.system_backend,
                self.system_transfer,
                self.axicon_state,
                self.current_z,
                self.z_reference,
            )
            with blocked_signals(camera_widgets):
                self.exposure.setValue(state.camera.exposure_us)
                self.gain.setValue(state.camera.gain)
                self.camera_provider_choice.setCurrentText(state.camera.provider)
                self.home_provider_choice.setCurrentText(state.camera.provider)
                self.system_backend.setCurrentText(state.application.backend)
                self.system_transfer.setCurrentText(state.application.transfer_mode)
                self.axicon_state.setCurrentText(state.system.physical_axicon.value)
                self.current_z.setValue(state.camera.current_z_mm or 0.0)
                self.z_reference.setText(state.camera.z_reference)
            configurable = state.camera.exposure_control == "SUPPORTED"
            self.exposure.setEnabled(configurable)
            self.gain.setEnabled(state.camera.gain_control == "SUPPORTED")
            is_beamage = state.camera.provider == "beamage"
            self.pc_beamage_controls.setVisible(is_beamage)
            if is_beamage:
                self.camera_control_note.setText(
                    "Live exposure is read back here. The official Pipeline does not provide an exposure-write command; "
                    "use PC-Beamage controls (the button brings its window forward)."
                )
            else:
                self.camera_control_note.setText("Exposure and gain are editable for this provider.")
            camera_text = (
                f"{state.camera.provider.upper()} • {state.camera.connection.value} • {state.camera.acquisition.value}"
                f" • {state.camera.implementation_status}"
                + (f" • device {state.camera.device_id}" if state.camera.device_id else "")
                + f"\nFrame route: {state.camera.frame_quality}"
                + (f"\n{state.camera.last_error}" if state.camera.last_error else "")
            )
            self.home_camera_status.setText(camera_text)
            for name in slms_to_refresh:
                self.slm_quick_cards[name].refresh(state)
                self.slm_detail_views[name].refresh(state)
            self.system_detail.setPlainText(
                f"SLM backend: {state.application.backend}\n"
                f"Transfer mode: {state.application.transfer_mode}\n"
                f"SLM1: {state.slm1.connection.value} • phase mode verified={state.slm1.phase_mode_verified}\n"
                f"SLM2: {state.slm2.connection.value} • phase mode verified={state.slm2.phase_mode_verified}\n"
                f"Camera: {camera_text}"
            )
            self._refresh_calibration_table()
        finally:
            self._syncing = False

    def _apply_system_state(self) -> None:
        if self._syncing:
            return

        def apply(state: ExperimentState) -> None:
            state.application.backend = self.system_backend.currentText()
            state.application.transfer_mode = self.system_transfer.currentText()
            state.system.physical_axicon = PhysicalAxiconState(self.axicon_state.currentText())
            state.camera.current_z_mm = self.current_z.value()
            state.camera.z_reference = self.z_reference.text().strip() or "UNSET"

        self.store.update(apply, source="advanced_system", reason="Operator changed system context")

    def _home_message(self, message: str) -> None:
        self.home_operation_status.setText(message)

    # --------------------------------------------------------------- providers

    def _configure_camera(self) -> None:
        exposure, gain = self.exposure.value(), self.gain.value()
        if self.controller.camera_provider is None:
            raise RuntimeError("Choose a camera provider first.")
        if not getattr(self.controller.camera_provider, "supports_configuration", True):
            self.camera_status.setText("Exposure and gain are controlled in PC-Beamage for this provider.")
            return
        self.controller.camera_provider.configure(exposure_us=exposure, gain=gain)
        self.store.update(
            lambda state: (
                setattr(state.camera, "exposure_us", exposure),
                setattr(state.camera, "gain", gain),
                setattr(state.camera, "settings_id", f"{state.camera.provider}-{exposure:g}us-g{gain:g}"),
            ),
            source="advanced_gui",
            reason="Camera acquisition settings changed",
        )

    def _select_camera_provider(self, name: str) -> None:
        if self._syncing:
            return
        self.stop_live()
        try:
            if name == "dummy":
                provider = DummyCameraProvider(self.store.snapshot, shape_yx=(512, 512))
            elif name == "beamage":
                provider = BeamageCameraProvider()
            else:
                self.camera_status.setText("Choose one or more quantitative replay files.")
                return
            self.controller.set_camera_provider(provider)
            with blocked_signals((self.camera_provider_choice, self.home_provider_choice)):
                self.camera_provider_choice.setCurrentText(name)
                self.home_provider_choice.setCurrentText(name)
        except Exception as exc:
            self._show_error("Camera selection failed", exc)

    def _browse_replay(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Choose quantitative camera frames",
            str(PROJECT_ROOT),
            "Quantitative frames (*.bmg *.npy *.txt *.csv *.bmp *.png *.tif *.tiff)",
        )
        if not paths:
            return
        try:
            self.stop_live()
            provider = ReplayCameraProvider(paths, loop=True)
            self.controller.set_camera_provider(provider)
            self._syncing = True
            with blocked_signals((self.camera_provider_choice, self.home_provider_choice)):
                self.camera_provider_choice.setCurrentText("replay")
                self.home_provider_choice.setCurrentText("replay")
            self._syncing = False
            self.camera_status.setText(f"Replay selected: {len(paths)} numeric frame(s)")
        except Exception as exc:
            self._syncing = False
            self._show_error("Replay selection failed", exc)

    def _connect_camera(self) -> None:
        try:
            self._configure_camera()
            message = self.controller.connect_camera()
            self.camera_status.setText(message)
        except Exception as exc:
            self._show_error("Camera connection failed", exc)

    def _focus_beamage_controls(self) -> None:
        if focus_pc_beamage_window():
            self.camera_status.setText("PC-Beamage brought to front; adjust exposure there and the readback will update here.")
        else:
            self.camera_status.setText("PC-Beamage window was not found. Start it and connect the camera first.")

    def start_live(self) -> None:
        if self._camera_thread is not None and self._camera_thread.isRunning():
            return
        try:
            self._configure_camera()
            thread = QThread(self)
            # PC-Beamage writes and reloads a full 2048×2048 BMP.  A modest
            # preview rate is much more stable than treating it like an in-memory
            # camera SDK, while dummy/replay remain responsive at 15 fps.
            target_fps = 3.0 if isinstance(self.controller.camera_provider, BeamageCameraProvider) else 15.0
            worker = CameraAcquisitionWorker(self.controller, target_fps=target_fps)
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.frame_ready.connect(self._on_frame)
            worker.error.connect(self._camera_error)
            worker.stopped.connect(thread.quit, Qt.DirectConnection)
            thread.finished.connect(self._live_stopped)
            self._camera_thread = thread
            self._camera_worker = worker
            self.start_live_button.setEnabled(False)
            thread.start()
        except Exception as exc:
            self._show_error("Could not start live camera", exc)

    def stop_live(self) -> None:
        thread, worker = self._camera_thread, self._camera_worker
        if thread is None:
            return
        if not stop_worker_thread(thread, worker):
            self.camera_status.setText("Camera worker did not stop within 3 seconds; disconnect hardware safely.")
        self._camera_thread = None
        self._camera_worker = None
        self.start_live_button.setEnabled(True)

    def _live_stopped(self) -> None:
        self.start_live_button.setEnabled(True)

    def _camera_error(self, message: str) -> None:
        self.camera_status.setText(f"Acquisition error: {message}")
        self.store.update(
            lambda state: (
                setattr(state.camera, "connection", ConnectionState.ERROR),
                setattr(state.camera, "acquisition", AcquisitionState.ERROR),
                setattr(state.camera, "last_error", message),
            ),
            source="camera_worker",
            reason="Live acquisition failed",
        )

    def _on_frame(self, frame: CameraFrame) -> None:
        self.current_frame = frame
        self._frame_counter += 1
        if not frame.metadata.get("quantitative_valid", True):
            self.current_metrics = None
            measured = frame.metadata.get("measurements", {})
            self.live_metrics.setPlainText(
                "LIVE PREVIEW ONLY — named-pipe BMP is not yet validated as raw quantitative data.\n"
                + "\n".join(f"PC-Beamage {key}: {value}" for key, value in measured.items())
            )
            self.saturation_warning.setText("Preview route hardware-unverified • formal capture disabled")
            self.saturation_warning.setObjectName("WarnChip")
            self.home_metric_summary.setText("LIVE PREVIEW ONLY • use PC-Beamage values until BMP validation")
        elif self.current_metrics is None or self._frame_counter % 4 == 0:
            try:
                self.current_metrics = self.metric_engine.analyse(frame, self.store.snapshot())
                self._show_metrics(self.current_metrics)
            except Exception as exc:
                self.live_metrics.setPlainText(f"Analysis unavailable: {exc}")
                self.current_metrics = None
        self._rerender()
        if self.pages.currentIndex() == self.PAGE_MEASURE and self.auto_fit.isChecked():
            self.image_view.fit_signal()
        elif self.pages.currentIndex() == self.PAGE_HOME and self.home_auto_fit.isChecked():
            self.home_camera_view.fit_signal()

    def _show_metrics(self, metrics: BeamMetrics) -> None:
        keys = (
            "total_signal",
            "saturation_fraction",
            "edge_power_fraction",
            "centre_y_px",
            "centre_x_px",
            "fwhm_major_px",
            "fwhm_minor_px",
            "ellipticity",
            "principal_ring_radius_px",
            "ring_eccentricity",
            "azimuthal_cv",
            "dark_core_fraction",
        )
        values = [f"family: {metrics.family}"]
        values.extend(
            f"{key}: {metrics.values[key]:.6g}"
            for key in keys
            if key in metrics.values and isinstance(metrics.values[key], (int, float))
        )
        self.live_metrics.setPlainText("\n".join(values))
        centre = metrics.centre_yx_px
        self.home_metric_summary.setText(
            f"{metrics.family} • centre (y,x)=({centre[0]:.2f}, {centre[1]:.2f}) px • "
            f"signal={float(metrics.values.get('total_signal', 0.0)):.5g}"
        )
        saturation = float(metrics.values.get("saturation_fraction") or 0.0)
        clipped = bool(metrics.values.get("clipped"))
        if saturation > 0 or clipped:
            self.saturation_warning.setText(
                f"WARNING • saturation {100*saturation:.3f}% • edge clipping {'YES' if clipped else 'no'}"
            )
            self.saturation_warning.setObjectName("BadChip")
        else:
            self.saturation_warning.setText("Signal in range • sensor edge clear")
            self.saturation_warning.setObjectName("GoodChip")
        self.saturation_warning.style().unpolish(self.saturation_warning)
        self.saturation_warning.style().polish(self.saturation_warning)

    def _rerender(self, *_args) -> None:
        if self.current_frame is None:
            return
        current_page = self.pages.currentIndex()
        if current_page == self.PAGE_MEASURE:
            self.image_view.set_quantitative_frame(
                self.current_frame,
                self.current_metrics,
                colour=self.colour_mode.currentText(),
                scale=self.display_scale.currentText(),
                gamma=self.display_gamma.value(),
                show_centre=self.overlay_centre.isChecked(),
                show_ring=self.overlay_ring.isChecked(),
                show_roi=self.overlay_roi.isChecked(),
            )
        elif current_page == self.PAGE_HOME:
            self.home_camera_view.set_quantitative_frame(
                self.current_frame,
                self.current_metrics,
                colour=self.home_colour_mode.currentText(),
                scale=self.home_display_scale.currentText(),
                gamma=self.home_display_gamma.value(),
                show_centre=True,
                show_ring=True,
                show_roi=True,
            )

    # ---------------------------------------------------------- SLM operations

    def _connect_slms(self) -> None:
        try:
            messages = self.controller.connect_slms()
            QMessageBox.information(self, "SLM connection", "\n".join(messages))
        except Exception as exc:
            self._show_error("SLM connection failed", exc)

    def _close_slms(self) -> None:
        try:
            self.controller.close_slms()
        except Exception as exc:
            self._show_error("SLM close failed", exc)

    def _cast(self, names: tuple[str, ...]) -> None:
        try:
            receipt = self.controller.cast(names)
            QMessageBox.information(
                self,
                "Cast complete",
                f"{receipt.transfer_mode}\n{receipt.folder}\n" + "\n".join(receipt.messages),
            )
        except Exception as exc:
            self._show_error("Cast failed", exc)

    def _blank_both(self) -> None:
        answer = QMessageBox.question(
            self,
            "Blank both SLMs?",
            "This sends a zero gray command to both panels and replaces the current displayed phase.",
        )
        if answer != QMessageBox.Yes:
            return
        try:
            self.controller.blank(("SLM1", "SLM2"))
        except Exception as exc:
            self._show_error("Blank failed", exc)

    def _open_compact(self) -> None:
        from slm_lab_control.app import MainWindow

        if self._compact_window is None:
            self._compact_window = MainWindow(controller=self.controller)
        self._compact_window.show()
        self._compact_window.raise_()

    # -------------------------------------------------------- formal / history

    def _formal_capture(self) -> None:
        was_live = self._camera_thread is not None and self._camera_thread.isRunning()
        self.stop_live()
        try:
            self._configure_camera()
            if self.controller.camera_provider is None:
                raise RuntimeError("Choose a camera provider.")
            if not self.controller.camera_provider.connected:
                self.controller.connect_camera()
            self.controller.start_camera()
            repository = SessionRepository(self.session_path.text().strip())
            service = FormalCaptureService(self.controller, repository, metric_engine=self.metric_engine)
            record = service.capture(
                self.trial_id.text().strip(),
                repeats=self.capture_repeats.value(),
                recipe=self.recipe_run.recipe_name if self.recipe_run else None,
                role=self.capture_role.currentText(),
                settle_s=self.settle_s.value(),
            )
            first_path = record.trial_root / "camera" / "frame_001.npy"
            self._history.append(
                {
                    "trial_id": record.trial_id,
                    "role": self.capture_role.currentText(),
                    "frame": np.load(first_path, allow_pickle=False),
                    "metrics": record.metrics[0],
                    "data_kind": record.data_kind,
                    "z_mm": self.store.snapshot().camera.current_z_mm,
                    "root": str(record.trial_root),
                }
            )
            self._refresh_history()
            self.trial_id.setText(f"trial_{len(self._history)+1:04d}")
            QMessageBox.information(
                self,
                "Formal capture complete",
                f"{record.trial_id}: {len(record.frame_ids)} fresh frame(s)\n{record.trial_root}",
            )
        except Exception as exc:
            self._show_error("Formal capture failed", exc)
        finally:
            try:
                self.controller.stop_camera()
            except Exception:
                pass
            if was_live:
                self.start_live()

    def _refresh_history(self) -> None:
        for combo in (self.baseline_choice, self.current_choice):
            combo.blockSignals(True)
            combo.clear()
            for index, item in enumerate(self._history):
                combo.addItem(f"{item['role']} • {item['trial_id']} • {item['data_kind']}", index)
            combo.blockSignals(False)
        if self._history:
            baseline_index = next(
                (index for index, item in enumerate(self._history) if item["role"] == "BASELINE"),
                0,
            )
            self.baseline_choice.setCurrentIndex(baseline_index)
            self.current_choice.setCurrentIndex(len(self._history) - 1)
            self._update_comparison()
            self._update_propagation()

    def _update_propagation(self) -> None:
        observations = []
        for item in self._history:
            if item.get("z_mm") is None:
                continue
            metrics = item["metrics"]
            observations.append(
                PlaneObservation(
                    frame_id=item["trial_id"],
                    z_mm=float(item["z_mm"]),
                    centre_y_px=metrics.centre_yx_px[0],
                    centre_x_px=metrics.centre_yx_px[1],
                    ring_radius_px=metrics.values.get("principal_ring_radius_px"),
                    power_proxy=float(metrics.values["total_signal"]),
                )
            )
        if len({observation.z_mm for observation in observations}) < 2:
            self.walk_plot.set_result(None)
            return
        state = self.store.snapshot()
        self.walk_plot.set_result(
            fit_beam_walk(
                observations,
                pixel_size_um=state.camera.pixel_size_um,
                camera_axis_calibrated=state.geometry.camera_travel_axis_calibrated,
            )
        )

    def _update_comparison(self, *_args) -> None:
        if not self._history or self.baseline_choice.currentIndex() < 0 or self.current_choice.currentIndex() < 0:
            return
        baseline = self._history[int(self.baseline_choice.currentData())]
        current = self._history[int(self.current_choice.currentData())]
        if baseline["frame"].shape != current["frame"].shape:
            self.comparison_metrics.setPlainText("Frame shapes differ; numerical difference is unavailable.")
            return
        kwargs = {
            "colour": self.colour_mode.currentText(),
            "scale": self.display_scale.currentText(),
            "gamma": self.display_gamma.value(),
        }
        self.baseline_view.set_preview_array(render_preview(baseline["frame"], **kwargs), metrics=baseline["metrics"])
        self.current_view.set_preview_array(render_preview(current["frame"], **kwargs), metrics=current["metrics"])
        b = np.asarray(baseline["frame"], dtype=float)
        c = np.asarray(current["frame"], dtype=float)
        b_norm = b / max(float(np.sum(b)), 1e-12)
        c_norm = c / max(float(np.sum(c)), 1e-12)
        difference = c_norm - b_norm
        limit = float(np.percentile(np.abs(difference), 99.5)) or 1.0
        unit = np.clip(0.5 + 0.5 * difference / limit, 0.0, 1.0)
        rgb = np.stack((unit, 1.0 - np.abs(2.0 * unit - 1.0), 1.0 - unit), axis=-1)
        self.difference_view.set_preview_array(np.asarray(np.rint(rgb * 255), dtype=np.uint8))
        b_values, c_values = baseline["metrics"].values, current["metrics"].values
        self.comparison_metrics.setPlainText(
            f"Baseline: {baseline['trial_id']} ({baseline['metrics'].family})\n"
            f"Current: {current['trial_id']} ({current['metrics'].family})\n"
            f"Total-signal ratio: {float(c_values['total_signal']) / max(float(b_values['total_signal']), 1e-12):.6g}\n"
            f"Centre shift (y,x): ({float(c_values['centre_y_px'])-float(b_values['centre_y_px']):+.4f}, "
            f"{float(c_values['centre_x_px'])-float(b_values['centre_x_px']):+.4f}) px"
        )

    # ----------------------------------------------------- calibration / recipe

    def _refresh_calibration_table(self) -> None:
        records = list(self.calibration.records.values())
        self.calibration_table.setRowCount(len(records))
        for row, record in enumerate(records):
            reason = record.stale_reason or "; ".join(record.evidence) or record.notes
            for column, value in enumerate((record.key.value, record.status.value, reason)):
                item = QTableWidgetItem(value)
                if column == 1:
                    item.setForeground(
                        QColor("#79edbe")
                        if record.status == CalibrationStatus.VALID
                        else QColor("#ffd58a")
                        if record.status in {CalibrationStatus.UNKNOWN, CalibrationStatus.PARTIALLY_VERIFIED}
                        else QColor("#ff9cac")
                    )
                self.calibration_table.setItem(row, column, item)
        readiness = self.calibration.readiness()
        self.readiness_chip.setText("READY" if readiness.ready else "NOT READY")
        self.readiness_chip.setObjectName("GoodChip" if readiness.ready else "BadChip")
        self.readiness_chip.style().unpolish(self.readiness_chip)
        self.readiness_chip.style().polish(self.readiness_chip)
        self.readiness_detail.setPlainText(
            ("SYSTEM READY FOR PROCESSING" if readiness.ready else "SYSTEM NOT READY")
            + "\n\n"
            + "\n".join(f"{key}: {value}" for key, value in readiness.statuses.items())
            + f"\n\nRecommended:\n{readiness.recommended_action}"
        )

    def _report_physical_change(self) -> None:
        try:
            affected = self.calibration.apply_physical_change(
                self.store, PhysicalChange(self.physical_change.currentText())
            )
            QMessageBox.information(
                self,
                "Dependencies updated",
                "Affected calibration closure:\n" + "\n".join(key.value for key in affected),
            )
        except Exception as exc:
            self._show_error("Could not record physical change", exc)

    def _load_calibration(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load calibration registry", str(PROJECT_ROOT), "JSON (*.json)")
        if not path:
            return
        try:
            self.calibration = CalibrationRegistry.load(path)
            self.recipe_engine.calibration = self.calibration
            self.calibration.publish(self.store, reason=f"Loaded calibration registry {path}")
            self._refresh_calibration_table()
        except Exception as exc:
            self._show_error("Calibration load failed", exc)

    def _save_calibration(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save calibration registry", str(PROJECT_ROOT / "calibration_registry.json"), "JSON (*.json)")
        if path:
            try:
                self.calibration.save(path)
            except Exception as exc:
                self._show_error("Calibration save failed", exc)

    def _refresh_recipe_description(self, *_args) -> None:
        definition = builtin_recipes()[self.recipe_choice.currentText()]
        prerequisites = ", ".join(key.value for key in definition.prerequisites) or "none"
        self.recipe_description.setText(f"{definition.description}\nPrerequisites: {prerequisites}")
        self.recipe_steps.clear()
        for index, step in enumerate(definition.steps, 1):
            self.recipe_steps.addItem(f"{index}. {step.title}  •  {step.kind.value}")

    def _start_recipe(self) -> None:
        try:
            self.recipe_run = self.recipe_engine.start(self.recipe_choice.currentText())
            self._show_recipe_step()
        except Exception as exc:
            self._show_error("Recipe cannot start", exc)

    def _advance_recipe(self) -> None:
        if self.recipe_run is None:
            QMessageBox.information(self, "Recipe", "Start a recipe first.")
            return
        try:
            completed = self.recipe_engine.current_step(self.recipe_run)
            next_step = self.recipe_engine.complete_step(
                self.recipe_run, {"operator_confirmed": True}
            )
            if completed:
                self.recipe_log.appendPlainText(f"Completed: {completed.title}")
            self._save_recipe_checkpoint()
            self._show_recipe_step()
            if next_step is None:
                QMessageBox.information(self, "Recipe complete", self.recipe_run.recipe_name)
        except Exception as exc:
            self._show_error("Recipe progression failed", exc)

    def _show_recipe_step(self) -> None:
        if self.recipe_run is None:
            return
        step = self.recipe_engine.current_step(self.recipe_run)
        self.recipe_status.setText(f"{self.recipe_run.recipe_name} • {self.recipe_run.status.value}")
        self.recipe_instruction.setText(
            f"{step.title}\n{step.instructions}" if step else "All recipe steps complete."
        )

    def _save_recipe_checkpoint(self) -> None:
        if self.recipe_run is None:
            return
        path = Path(self.session_path.text().strip()) / "recipe_run.json"
        self.recipe_engine.save(self.recipe_run, path)

    def _run_demo(self) -> None:
        try:
            from ..demo import run_closed_loop_demo

            output = Path(self.session_path.text().strip()).parent / "synthetic_closed_loop_demo"
            result = run_closed_loop_demo(output)
            self.recipe_log.setPlainText(json.dumps(result, indent=2))
            QMessageBox.information(self, "Synthetic demo complete", str(output))
        except Exception as exc:
            self._show_error("Synthetic demo failed", exc)

    # --------------------------------------------------------------- presets

    def _refresh_presets(self) -> None:
        self.preset_list.clear()
        root = self.store.snapshot().to_app_config().resolve_preset_root(PROJECT_ROOT)
        root.mkdir(parents=True, exist_ok=True)
        for path in sorted(root.rglob("*.json")):
            self.preset_list.addItem(str(path.relative_to(root)))
            self.preset_list.item(self.preset_list.count() - 1).setData(Qt.UserRole, str(path))

    def _load_selected_preset(self) -> None:
        item = self.preset_list.currentItem()
        if item is None:
            QMessageBox.information(self, "Presets", "Select a preset first.")
            return
        try:
            config = load_preset(Path(item.data(Qt.UserRole)))
            self.controller.replace_app_config(
                config, source="advanced_gui", reason=f"Loaded preset {item.text()}"
            )
            self.controller.generate()
        except Exception as exc:
            self._show_error("Preset load failed", exc)

    # ---------------------------------------------------------------- cleanup

    def _show_error(self, title: str, error: Exception) -> None:
        QMessageBox.warning(self, title, str(error))

    def closeEvent(self, event):  # noqa: N802 - Qt API
        self.stop_live()
        try:
            if self.controller.camera_provider is not None:
                self.controller.camera_provider.disconnect()
        finally:
            self._unsubscribe()
        event.accept()


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
    window = AdvancedLabWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
