from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import json
import time

from PySide6 import QtCore, QtGui, QtWidgets

from .hardware_profiles import (
    LegacyHardwareProfile,
    PockelsCandidate,
    load_legacy_hardware_profile,
)
from .kinematics import RigKinematics, RigProfile
from .movement_map import MovementMap2D
from .providers import (
    AttenuatorProvider,
    HXPDigitalLaserConfig,
    HXPDigitalLaserGate,
    HXPProvider,
    HXPProviderConfig,
    HexapodProvider,
    LaserGateProvider,
    UnconfiguredAttenuatorProvider,
    VirtualAttenuatorProvider,
    VirtualHexapodProvider,
    VirtualLaserGate,
)
from .recipe import Recipe, RecipeStep, StepKind, preflight_recipe
from .sweeps import RasterSweepSpec, build_raster_sweep
from .types import (
    AttenuatorSnapshot,
    HexapodSnapshot,
    LaserSnapshot,
    MotionState,
    Pose6D,
)
from .viewer import Hexapod3DViewer


PACKAGE_DIR = Path(__file__).resolve().parent
APP_ROOT = PACKAGE_DIR.parent
ASSETS_DIR = APP_ROOT / "assets"
DEFAULT_PROFILE = ASSETS_DIR / "cad_profile.json"
DEFAULT_CONFIG = APP_ROOT / "hardware_config.example.json"
LEGACY_EVIDENCE = ASSETS_DIR / "legacy_hardware_evidence.json"


class RecipeList(QtWidgets.QListWidget):
    orderChanged = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        self.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.setAlternatingRowColors(True)
        self.setSpacing(2)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        super().dropEvent(event)
        self.orderChanged.emit()


class MainWindow(QtWidgets.QMainWindow):
    """Standalone manual controller + script builder for HXP and process beam."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hexapod + Laser Lab — Standalone Controller")
        self.resize(1720, 1020)
        self.setMinimumSize(1250, 760)

        self.profile = RigProfile.from_json(DEFAULT_PROFILE)
        self.kinematics = RigKinematics(self.profile)
        self.legacy_profile: LegacyHardwareProfile = (
            load_legacy_hardware_profile(LEGACY_EVIDENCE)
        )
        self._selected_legacy_candidate_key = "labview_v3"

        self.virtual_stage = VirtualHexapodProvider()
        self.virtual_stage.connect()
        self.real_stage: HXPProvider | None = None

        self.virtual_laser = VirtualLaserGate()
        self.virtual_laser.connect()
        self.real_laser: HXPDigitalLaserGate | None = None

        self.virtual_attenuator = VirtualAttenuatorProvider(0.0)
        self.virtual_attenuator.connect()
        self.real_attenuator: AttenuatorProvider = (
            UnconfiguredAttenuatorProvider()
        )

        self._last_stage_snapshot = self.virtual_stage.snapshot()
        self._last_tick_monotonic = time.monotonic()
        self._poll_pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="hxp-poll",
        )
        self._poll_future: Future | None = None
        self._last_real_poll = 0.0

        self.recipe = Recipe()
        self._recipe_running = False
        self._recipe_index = 0
        self._recipe_step_issued = False
        self._wait_until = 0.0

        # Manual "LINE Move_While Write" mirrors the recovered LabVIEW workflow.
        self._manual_write_line_active = False
        self._manual_write_line_started = False
        self._manual_write_line_description = ""

        self._build_ui()
        self._apply_style()
        self._load_default_config()
        self._apply_mock_profile()
        self._update_recipe_preflight_view()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self.statusBar().showMessage(
            "Virtual stage, Pockels cell and attenuator ready"
        )

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(7)

        root.addLayout(self._build_header())

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._build_manual_tab(), "CONTROL")
        self.tabs.addTab(self._build_script_tab(), "SCRIPT BUILDER")
        self.tabs.addTab(self._build_commissioning_tab(), "COMMISSIONING")
        self.tabs.addTab(self._build_setup_tab(), "SETUP + DIAGNOSTICS")
        root.addWidget(self.tabs, 1)

    def _build_header(self) -> QtWidgets.QHBoxLayout:
        header = QtWidgets.QHBoxLayout()
        header.setSpacing(8)

        title_box = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("HEXAPOD + LASER LAB")
        title.setObjectName("title")
        subtitle = QtWidgets.QLabel(
            "Manual control • digital twin • process-beam control"
        )
        subtitle.setObjectName("muted")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)

        header.addStretch(1)

        mode_label = QtWidgets.QLabel("Lab mode")
        mode_label.setObjectName("headerLabel")
        header.addWidget(mode_label)
        self.lab_mode = QtWidgets.QComboBox()
        self.lab_mode.addItems(["MOCK LAB", "REAL LAB"])
        self.lab_mode.setMinimumWidth(105)
        self.lab_mode.currentIndexChanged.connect(self._lab_mode_changed)
        header.addWidget(self.lab_mode)

        self.mode_chip = QtWidgets.QLabel("MOCK • SAFE")
        self.mode_chip.setObjectName("chipSafe")
        header.addWidget(self.mode_chip)

        profile_label = QtWidgets.QLabel("Legacy profile")
        profile_label.setObjectName("headerLabel")
        header.addWidget(profile_label)
        self.legacy_profile_combo = QtWidgets.QComboBox()
        for candidate in self.legacy_profile.pockels_candidates:
            self.legacy_profile_combo.addItem(candidate.label, candidate.key)
        self.legacy_profile_combo.currentIndexChanged.connect(
            self._legacy_profile_changed
        )
        self.legacy_profile_combo.setMinimumWidth(240)
        header.addWidget(self.legacy_profile_combo)

        stage_label = QtWidgets.QLabel("Stage")
        stage_label.setObjectName("headerLabel")
        header.addWidget(stage_label)
        self.stage_mode = QtWidgets.QComboBox()
        self.stage_mode.addItems(["Virtual", "Real Newport HXP"])
        self.stage_mode.setMinimumWidth(150)
        self.stage_mode.currentIndexChanged.connect(
            self._stage_mode_changed
        )
        header.addWidget(self.stage_mode)

        self.stage_chip = QtWidgets.QLabel("STAGE IDLE")
        self.stage_chip.setObjectName("chipNeutral")
        header.addWidget(self.stage_chip)

        self.beam_chip = QtWidgets.QLabel("BEAM CLOSED")
        self.beam_chip.setObjectName("chipSafe")
        header.addWidget(self.beam_chip)

        self.attenuator_chip = QtWidgets.QLabel("ATT 0.0 %")
        self.attenuator_chip.setObjectName("chipNeutral")
        header.addWidget(self.attenuator_chip)

        self.close_beam_header = QtWidgets.QPushButton("CLOSE BEAM")
        self.close_beam_header.setObjectName("safeAction")
        self.close_beam_header.setToolTip(
            "Request Pockels cell CLOSED. This does not power off the PHAROS."
        )
        self.close_beam_header.clicked.connect(self._close_all_pockels)
        header.addWidget(self.close_beam_header)

        self.stop_header = QtWidgets.QPushButton(
            "STOP MOTION + CLOSE BEAM"
        )
        self.stop_header.setObjectName("danger")
        self.stop_header.setToolTip(
            "Software abort: request Pockels CLOSED and send HXP GroupMoveAbort. "
            "This is not a hardware emergency stop."
        )
        self.stop_header.clicked.connect(self._abort_all)
        header.addWidget(self.stop_header)

        return header

    def _build_manual_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.addWidget(self._scroll_wrap(self._build_motion_panel()))
        split.addWidget(self._build_visual_panel())
        split.addWidget(self._scroll_wrap(self._build_process_panel()))
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 0)
        split.setSizes([365, 930, 355])
        layout.addWidget(split)
        return tab

    def _build_motion_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(9)

        stage_box = QtWidgets.QGroupBox("STAGE CONNECTION")
        stage_layout = QtWidgets.QVBoxLayout(stage_box)
        self.manual_stage_status = QtWidgets.QLabel(
            "Virtual stage connected"
        )
        self.manual_stage_status.setObjectName("statusPill")
        stage_layout.addWidget(self.manual_stage_status)
        row = QtWidgets.QHBoxLayout()
        self.connect_stage_btn = QtWidgets.QPushButton("Connect")
        self.connect_stage_btn.clicked.connect(self._connect_stage)
        self.disconnect_stage_btn = QtWidgets.QPushButton("Disconnect")
        self.disconnect_stage_btn.clicked.connect(self._disconnect_stage)
        row.addWidget(self.connect_stage_btn)
        row.addWidget(self.disconnect_stage_btn)
        stage_layout.addLayout(row)
        hint = QtWidgets.QLabel(
            "Real-HXP network settings live in Setup + Diagnostics."
        )
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        stage_layout.addWidget(hint)
        layout.addWidget(stage_box)

        target_box = QtWidgets.QGroupBox("TARGET POSE")
        target_layout = QtWidgets.QVBoxLayout(target_box)
        pose_grid = QtWidgets.QGridLayout()
        self.pose_boxes: dict[str, QtWidgets.QDoubleSpinBox] = {}
        for i, axis in enumerate("XYZUVW"):
            box = self._pose_spinbox(i)
            self.pose_boxes[axis] = box
            r = i % 3
            c = (i // 3) * 2
            pose_grid.addWidget(QtWidgets.QLabel(axis), r, c)
            pose_grid.addWidget(box, r, c + 1)
        target_layout.addLayout(pose_grid)

        move_row = QtWidgets.QHBoxLayout()
        self.move_abs_btn = QtWidgets.QPushButton("MOVE TO TARGET")
        self.move_abs_btn.setObjectName("primary")
        self.move_abs_btn.clicked.connect(self._move_absolute)
        copy_btn = QtWidgets.QPushButton("Target ← actual")
        copy_btn.clicked.connect(self._target_from_actual)
        move_row.addWidget(self.move_abs_btn, 2)
        move_row.addWidget(copy_btn, 1)
        target_layout.addLayout(move_row)
        layout.addWidget(target_box)

        jog_box = QtWidgets.QGroupBox("JOG")
        jog_layout = QtWidgets.QVBoxLayout(jog_box)
        steps = QtWidgets.QFormLayout()
        self.jog_mm = QtWidgets.QDoubleSpinBox()
        self.jog_mm.setRange(0.0001, 100.0)
        self.jog_mm.setDecimals(4)
        self.jog_mm.setValue(0.1)
        self.jog_mm.setSuffix(" mm")
        self.jog_deg = QtWidgets.QDoubleSpinBox()
        self.jog_deg.setRange(0.0001, 30.0)
        self.jog_deg.setDecimals(4)
        self.jog_deg.setValue(0.1)
        self.jog_deg.setSuffix(" °")
        steps.addRow("XYZ step", self.jog_mm)
        steps.addRow("UVW step", self.jog_deg)
        jog_layout.addLayout(steps)

        jog_grid = QtWidgets.QGridLayout()
        self.jog_buttons: list[QtWidgets.QPushButton] = []
        for row, axis in enumerate("XYZUVW"):
            minus = QtWidgets.QPushButton(f"{axis} −")
            plus = QtWidgets.QPushButton(f"{axis} +")
            minus.clicked.connect(
                lambda _=False, a=axis: self._jog(a, -1.0)
            )
            plus.clicked.connect(
                lambda _=False, a=axis: self._jog(a, +1.0)
            )
            self.jog_buttons.extend([minus, plus])
            jog_grid.addWidget(minus, row, 0)
            jog_grid.addWidget(plus, row, 1)
        jog_layout.addLayout(jog_grid)
        layout.addWidget(jog_box)

        quick_box = QtWidgets.QGroupBox("QUICK LINE / WRITING MOVE")
        quick_layout = QtWidgets.QVBoxLayout(quick_box)
        quick_note = QtWidgets.QLabel(
            "Mirrors the recovered LabVIEW LINE Move / LINE Move_While Write "
            "workflow using native HXP Line + target velocity."
        )
        quick_note.setObjectName("muted")
        quick_note.setWordWrap(True)
        quick_layout.addWidget(quick_note)

        quick_form = QtWidgets.QFormLayout()
        self.quick_dx = QtWidgets.QDoubleSpinBox()
        self.quick_dx.setRange(-1000.0, 1000.0)
        self.quick_dx.setDecimals(4)
        self.quick_dx.setValue(-7.0)
        self.quick_dx.setSuffix(" mm")
        self.quick_dy = QtWidgets.QDoubleSpinBox()
        self.quick_dy.setRange(-1000.0, 1000.0)
        self.quick_dy.setDecimals(4)
        self.quick_dy.setValue(0.0)
        self.quick_dy.setSuffix(" mm")
        self.quick_dz = QtWidgets.QDoubleSpinBox()
        self.quick_dz.setRange(-1000.0, 1000.0)
        self.quick_dz.setDecimals(4)
        self.quick_dz.setValue(0.0)
        self.quick_dz.setSuffix(" mm")
        self.quick_velocity = QtWidgets.QDoubleSpinBox()
        self.quick_velocity.setRange(0.001, 100.0)
        self.quick_velocity.setDecimals(3)
        self.quick_velocity.setValue(1.0)
        self.quick_velocity.setSuffix(" mm/s")
        self.quick_row_pitch = QtWidgets.QDoubleSpinBox()
        self.quick_row_pitch.setRange(-100.0, 100.0)
        self.quick_row_pitch.setDecimals(4)
        self.quick_row_pitch.setValue(0.02)
        self.quick_row_pitch.setSuffix(" mm")
        quick_form.addRow("dX", self.quick_dx)
        quick_form.addRow("dY", self.quick_dy)
        quick_form.addRow("dZ", self.quick_dz)
        quick_form.addRow("Velocity", self.quick_velocity)
        quick_form.addRow("Return row pitch", self.quick_row_pitch)
        quick_layout.addLayout(quick_form)

        quick_buttons = QtWidgets.QGridLayout()
        self.quick_move_btn = QtWidgets.QPushButton("MOVE LINE")
        self.quick_move_btn.clicked.connect(
            lambda: self._start_quick_line(write=False)
        )
        self.quick_write_btn = QtWidgets.QPushButton(
            "WRITE LINE\nPOCKELS OPEN DURING MOVE"
        )
        self.quick_write_btn.setObjectName("beamOnSmall")
        self.quick_write_btn.clicked.connect(
            lambda: self._start_quick_line(write=True)
        )
        self.quick_return_btn = QtWidgets.QPushButton(
            "RETURN + ROW\nBEAM CLOSED"
        )
        self.quick_return_btn.setObjectName("safeAction")
        self.quick_return_btn.clicked.connect(self._quick_return_row)
        quick_buttons.addWidget(self.quick_move_btn, 0, 0)
        quick_buttons.addWidget(self.quick_write_btn, 0, 1)
        quick_buttons.addWidget(self.quick_return_btn, 1, 0, 1, 2)
        quick_layout.addLayout(quick_buttons)

        self.quick_line_status = QtWidgets.QLabel("Ready")
        self.quick_line_status.setObjectName("statusPill")
        self.quick_line_status.setWordWrap(True)
        quick_layout.addWidget(self.quick_line_status)
        layout.addWidget(quick_box)

        actions = QtWidgets.QGroupBox("STAGE ACTIONS")
        actions_layout = QtWidgets.QHBoxLayout(actions)
        self.home_btn = QtWidgets.QPushButton("HOME")
        self.home_btn.clicked.connect(self._home)
        self.stop_motion_btn = QtWidgets.QPushButton(
            "STOP + CLOSE BEAM"
        )
        self.stop_motion_btn.setObjectName("danger")
        self.stop_motion_btn.clicked.connect(self._abort_all)
        actions_layout.addWidget(self.home_btn)
        actions_layout.addWidget(self.stop_motion_btn)
        layout.addWidget(actions)
        layout.addStretch(1)
        return panel

    def _build_visual_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(5)

        topbar = QtWidgets.QHBoxLayout()
        topbar.addWidget(self._section("LIVE DIGITAL TWIN"))
        topbar.addStretch(1)
        self.cad_status = QtWidgets.QLabel("CAD-derived rig")
        self.cad_status.setObjectName("muted")
        topbar.addWidget(self.cad_status)
        load_cad = QtWidgets.QPushButton("Load STEP…")
        load_cad.clicked.connect(self._load_step_dialog)
        topbar.addWidget(load_cad)
        clear = QtWidgets.QPushButton("Clear paths")
        clear.clicked.connect(self._clear_visual_traces)
        topbar.addWidget(clear)
        layout.addLayout(topbar)

        self.viewer = Hexapod3DViewer(panel, self.profile)
        self.viewer.set_status_callback(self._cad_message)

        map_container = QtWidgets.QWidget()
        map_layout = QtWidgets.QVBoxLayout(map_container)
        map_layout.setContentsMargins(0, 0, 0, 0)
        map_layout.setSpacing(3)
        map_bar = QtWidgets.QHBoxLayout()
        map_bar.addWidget(self._section("2D MOVEMENT MAP"))
        map_bar.addStretch(1)
        self.map_mode = QtWidgets.QComboBox()
        self.map_mode.addItems(
            ["Laser path on sample", "HXP XY carriage path"]
        )
        self.map_span = QtWidgets.QDoubleSpinBox()
        self.map_span.setRange(2.0, 500.0)
        self.map_span.setValue(40.0)
        self.map_span.setSuffix(" mm span")
        map_bar.addWidget(self.map_mode)
        map_bar.addWidget(self.map_span)
        map_layout.addLayout(map_bar)

        self.movement_map = MovementMap2D(map_container)
        self.map_mode.currentIndexChanged.connect(
            lambda i: self.movement_map.set_mode(
                "sample" if i == 0 else "hxp"
            )
        )
        self.map_span.valueChanged.connect(
            self.movement_map.set_span_mm
        )
        map_layout.addWidget(self.movement_map, 1)

        vertical = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        vertical.addWidget(self.viewer)
        vertical.addWidget(map_container)
        vertical.setStretchFactor(0, 4)
        vertical.setStretchFactor(1, 2)
        vertical.setSizes([650, 260])
        layout.addWidget(vertical, 1)
        return panel

    def _build_process_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(9)

        beam_box = QtWidgets.QGroupBox("PROCESS BEAM / POCKELS CELL")
        beam_layout = QtWidgets.QVBoxLayout(beam_box)
        self.manual_beam_status = QtWidgets.QLabel(
            "LASER OFF • POCKELS CLOSED"
        )
        self.manual_beam_status.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.manual_beam_status.setObjectName("laserOff")
        beam_layout.addWidget(self.manual_beam_status)

        explanation = QtWidgets.QLabel(
            "LASER ON/OFF here means opening/closing the process-beam "
            "Pockels cell. It does not power-cycle the PHAROS source."
        )
        explanation.setObjectName("muted")
        explanation.setWordWrap(True)
        beam_layout.addWidget(explanation)

        self.manual_beam_arm = QtWidgets.QCheckBox(
            "ARM MANUAL REAL-BEAM CONTROL"
        )
        self.manual_beam_arm.setObjectName("dangerCheck")
        self.manual_beam_arm.setToolTip(
            "Required only when the real LX13/Pockels provider is selected."
        )
        beam_layout.addWidget(self.manual_beam_arm)

        self.laser_on_btn = QtWidgets.QPushButton(
            "LASER ON\nOPEN POCKELS CELL"
        )
        self.laser_on_btn.setObjectName("beamOn")
        self.laser_on_btn.setMinimumHeight(62)
        self.laser_on_btn.clicked.connect(
            lambda: self._set_pockels(True, manual=True)
        )
        beam_layout.addWidget(self.laser_on_btn)

        self.laser_off_btn = QtWidgets.QPushButton(
            "LASER OFF\nCLOSE POCKELS CELL"
        )
        self.laser_off_btn.setObjectName("beamOff")
        self.laser_off_btn.setMinimumHeight(58)
        self.laser_off_btn.clicked.connect(
            lambda: self._set_pockels(False, manual=True)
        )
        beam_layout.addWidget(self.laser_off_btn)

        provider_note = QtWidgets.QLabel(
            "Pockels provider is configured in Setup + Diagnostics."
        )
        provider_note.setObjectName("muted")
        provider_note.setWordWrap(True)
        beam_layout.addWidget(provider_note)
        layout.addWidget(beam_box)

        att_box = QtWidgets.QGroupBox("ATTENUATOR")
        att_layout = QtWidgets.QVBoxLayout(att_box)
        self.attenuator_status = QtWidgets.QLabel(
            "Transmission setpoint: 0.0 %"
        )
        self.attenuator_status.setObjectName("statusPill")
        att_layout.addWidget(self.attenuator_status)

        slider_row = QtWidgets.QHBoxLayout()
        self.attenuator_slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        self.attenuator_slider.setRange(0, 1000)
        self.attenuator_slider.setValue(0)
        self.attenuator_spin = QtWidgets.QDoubleSpinBox()
        self.attenuator_spin.setRange(0.0, 100.0)
        self.attenuator_spin.setDecimals(1)
        self.attenuator_spin.setSuffix(" %")
        self.attenuator_spin.setValue(0.0)
        self.attenuator_slider.valueChanged.connect(
            lambda v: self.attenuator_spin.setValue(v / 10.0)
        )
        self.attenuator_spin.valueChanged.connect(
            lambda v: self.attenuator_slider.setValue(round(v * 10.0))
        )
        slider_row.addWidget(self.attenuator_slider, 1)
        slider_row.addWidget(self.attenuator_spin)
        att_layout.addLayout(slider_row)

        presets = QtWidgets.QHBoxLayout()
        for value in (0, 25, 50, 75, 100):
            b = QtWidgets.QPushButton(f"{value}%")
            b.clicked.connect(
                lambda _=False, v=value: self._attenuator_preset(v)
            )
            presets.addWidget(b)
        att_layout.addLayout(presets)

        self.set_attenuator_btn = QtWidgets.QPushButton(
            "SET ATTENUATOR"
        )
        self.set_attenuator_btn.setObjectName("primary")
        self.set_attenuator_btn.clicked.connect(
            self._set_attenuator_manual
        )
        att_layout.addWidget(self.set_attenuator_btn)
        att_note = QtWidgets.QLabel(
            "The GUI uses normalized transmission %. A real attenuator "
            "driver/calibration will map this to the device's native control."
        )
        att_note.setObjectName("muted")
        att_note.setWordWrap(True)
        att_layout.addWidget(att_note)
        layout.addWidget(att_box)

        state_box = QtWidgets.QGroupBox("LIVE POSITION")
        state_layout = QtWidgets.QVBoxLayout(state_box)
        self.state_labels: dict[str, QtWidgets.QLabel] = {}
        grid = QtWidgets.QGridLayout()
        for i, axis in enumerate("XYZUVW"):
            grid.addWidget(QtWidgets.QLabel(axis), i, 0)
            lab = QtWidgets.QLabel("0.0000")
            lab.setObjectName("mono")
            self.state_labels[axis] = lab
            grid.addWidget(lab, i, 1)
        state_layout.addLayout(grid)
        self.motion_status = QtWidgets.QLabel("IDLE")
        self.motion_status.setObjectName("statusPill")
        state_layout.addWidget(self.motion_status)
        layout.addWidget(state_box)

        leg_box = QtWidgets.QGroupBox("ACTUATOR GEOMETRY")
        leg_layout = QtWidgets.QGridLayout(leg_box)
        self.leg_labels: list[QtWidgets.QLabel] = []
        for i in range(6):
            leg_layout.addWidget(QtWidgets.QLabel(f"Strut {i + 1}"), i, 0)
            val = QtWidgets.QLabel("— mm")
            val.setObjectName("mono")
            self.leg_labels.append(val)
            leg_layout.addWidget(val, i, 1)
        layout.addWidget(leg_box)
        layout.addStretch(1)
        return panel

    def _build_script_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.addWidget(self._scroll_wrap(self._build_script_palette()))
        split.addWidget(self._build_script_sequence())
        split.addWidget(self._build_script_run_panel())
        split.setSizes([360, 760, 390])
        layout.addWidget(split)
        return tab

    def _build_script_palette(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        move_box = QtWidgets.QGroupBox("ADD MOTION STEP")
        move_layout = QtWidgets.QVBoxLayout(move_box)
        self.script_move_kind = QtWidgets.QComboBox()
        self.script_move_kind.addItems(
            [
                "Absolute pose",
                "Relative move",
                "Line move at target velocity",
            ]
        )
        move_layout.addWidget(self.script_move_kind)
        pose_grid = QtWidgets.QGridLayout()
        self.script_pose_boxes: dict[str, QtWidgets.QDoubleSpinBox] = {}
        for i, axis in enumerate("XYZUVW"):
            box = self._pose_spinbox(i)
            self.script_pose_boxes[axis] = box
            r = i % 3
            c = (i // 3) * 2
            pose_grid.addWidget(QtWidgets.QLabel(axis), r, c)
            pose_grid.addWidget(box, r, c + 1)
        move_layout.addLayout(pose_grid)

        speed_row = QtWidgets.QFormLayout()
        self.script_line_velocity = QtWidgets.QDoubleSpinBox()
        self.script_line_velocity.setRange(0.001, 100.0)
        self.script_line_velocity.setDecimals(3)
        self.script_line_velocity.setValue(1.0)
        self.script_line_velocity.setSuffix(" mm/s")
        self.script_line_velocity.setToolTip(
            "Used only for 'Line move at target velocity'. "
            "This maps to the legacy HXP "
            "HexapodMoveIncrementalControlWithTargetVelocity command."
        )
        speed_row.addRow("Line velocity", self.script_line_velocity)
        move_layout.addLayout(speed_row)

        row = QtWidgets.QHBoxLayout()
        add_move = QtWidgets.QPushButton("+ ADD MOVE")
        add_move.setObjectName("primary")
        add_move.clicked.connect(self._recipe_add_move)
        copy_live = QtWidgets.QPushButton("Pose ← live")
        copy_live.clicked.connect(self._script_pose_from_live)
        row.addWidget(add_move, 2)
        row.addWidget(copy_live, 1)
        move_layout.addLayout(row)
        layout.addWidget(move_box)

        beam_box = QtWidgets.QGroupBox("ADD BEAM STEP")
        beam_layout = QtWidgets.QVBoxLayout(beam_box)
        on = QtWidgets.QPushButton("+ LASER ON / POCKELS OPEN")
        on.setObjectName("beamOnSmall")
        on.clicked.connect(
            lambda: self._append_recipe_step(
                RecipeStep.pockels_cell(True)
            )
        )
        off = QtWidgets.QPushButton("+ LASER OFF / POCKELS CLOSED")
        off.setObjectName("beamOffSmall")
        off.clicked.connect(
            lambda: self._append_recipe_step(
                RecipeStep.pockels_cell(False)
            )
        )
        beam_layout.addWidget(on)
        beam_layout.addWidget(off)
        layout.addWidget(beam_box)

        att_box = QtWidgets.QGroupBox("ADD ATTENUATOR STEP")
        att_layout = QtWidgets.QHBoxLayout(att_box)
        self.script_attenuator = QtWidgets.QDoubleSpinBox()
        self.script_attenuator.setRange(0.0, 100.0)
        self.script_attenuator.setDecimals(1)
        self.script_attenuator.setSuffix(" %")
        add_att = QtWidgets.QPushButton("+ ADD")
        add_att.clicked.connect(self._recipe_add_attenuator)
        att_layout.addWidget(self.script_attenuator)
        att_layout.addWidget(add_att)
        layout.addWidget(att_box)

        wait_box = QtWidgets.QGroupBox("ADD WAIT")
        wait_layout = QtWidgets.QHBoxLayout(wait_box)
        self.script_wait = QtWidgets.QDoubleSpinBox()
        self.script_wait.setRange(0.0, 3600.0)
        self.script_wait.setDecimals(3)
        self.script_wait.setValue(1.0)
        self.script_wait.setSuffix(" s")
        add_wait = QtWidgets.QPushButton("+ ADD")
        add_wait.clicked.connect(self._recipe_add_wait)
        wait_layout.addWidget(self.script_wait)
        wait_layout.addWidget(add_wait)
        layout.addWidget(wait_box)

        sweep_box = QtWidgets.QGroupBox("RASTER / PARAMETER SWEEP")
        sweep_layout = QtWidgets.QFormLayout(sweep_box)

        self.sweep_write_dx = QtWidgets.QDoubleSpinBox()
        self.sweep_write_dx.setRange(-1000.0, 1000.0)
        self.sweep_write_dx.setDecimals(3)
        self.sweep_write_dx.setValue(-7.0)
        self.sweep_write_dx.setSuffix(" mm")

        self.sweep_row_pitch = QtWidgets.QDoubleSpinBox()
        self.sweep_row_pitch.setRange(-100.0, 100.0)
        self.sweep_row_pitch.setDecimals(4)
        self.sweep_row_pitch.setValue(0.02)
        self.sweep_row_pitch.setSuffix(" mm")

        self.sweep_series_spacing = QtWidgets.QDoubleSpinBox()
        self.sweep_series_spacing.setRange(0.0, 100.0)
        self.sweep_series_spacing.setDecimals(4)
        self.sweep_series_spacing.setValue(0.10)
        self.sweep_series_spacing.setSuffix(" mm")

        self.sweep_v_start = QtWidgets.QDoubleSpinBox()
        self.sweep_v_start.setRange(0.001, 100.0)
        self.sweep_v_start.setDecimals(3)
        self.sweep_v_start.setValue(0.2)
        self.sweep_v_start.setSuffix(" mm/s")

        self.sweep_v_stop = QtWidgets.QDoubleSpinBox()
        self.sweep_v_stop.setRange(0.001, 100.0)
        self.sweep_v_stop.setDecimals(3)
        self.sweep_v_stop.setValue(2.1)
        self.sweep_v_stop.setSuffix(" mm/s")

        self.sweep_v_step = QtWidgets.QDoubleSpinBox()
        self.sweep_v_step.setRange(-100.0, 100.0)
        self.sweep_v_step.setDecimals(3)
        self.sweep_v_step.setValue(0.1)
        self.sweep_v_step.setSuffix(" mm/s")

        self.sweep_return_velocity = QtWidgets.QDoubleSpinBox()
        self.sweep_return_velocity.setRange(0.001, 100.0)
        self.sweep_return_velocity.setDecimals(3)
        self.sweep_return_velocity.setValue(10.0)
        self.sweep_return_velocity.setSuffix(" mm/s")

        self.sweep_use_attenuation = QtWidgets.QCheckBox(
            "Sweep attenuator transmission"
        )
        self.sweep_att_start = QtWidgets.QDoubleSpinBox()
        self.sweep_att_start.setRange(0.0, 100.0)
        self.sweep_att_start.setValue(25.0)
        self.sweep_att_start.setSuffix(" %")
        self.sweep_att_stop = QtWidgets.QDoubleSpinBox()
        self.sweep_att_stop.setRange(0.0, 100.0)
        self.sweep_att_stop.setValue(25.0)
        self.sweep_att_stop.setSuffix(" %")
        self.sweep_att_step = QtWidgets.QDoubleSpinBox()
        self.sweep_att_step.setRange(-100.0, 100.0)
        self.sweep_att_step.setValue(5.0)
        self.sweep_att_step.setSuffix(" %")

        sweep_layout.addRow("Write dX", self.sweep_write_dx)
        sweep_layout.addRow("Row pitch", self.sweep_row_pitch)
        sweep_layout.addRow("Series spacing", self.sweep_series_spacing)
        sweep_layout.addRow("Velocity start", self.sweep_v_start)
        sweep_layout.addRow("Velocity stop", self.sweep_v_stop)
        sweep_layout.addRow("Velocity step", self.sweep_v_step)
        sweep_layout.addRow("Return velocity", self.sweep_return_velocity)
        sweep_layout.addRow(self.sweep_use_attenuation)
        sweep_layout.addRow("Atten start", self.sweep_att_start)
        sweep_layout.addRow("Atten stop", self.sweep_att_stop)
        sweep_layout.addRow("Atten step", self.sweep_att_step)

        self.sweep_summary = QtWidgets.QLabel(
            "Legacy-compatible raster generator: Pockels OPEN only for "
            "the writing line; return moves are CLOSED."
        )
        self.sweep_summary.setObjectName("muted")
        self.sweep_summary.setWordWrap(True)
        sweep_layout.addRow(self.sweep_summary)

        build_sweep = QtWidgets.QPushButton(
            "GENERATE SWEEP RECIPE"
        )
        build_sweep.setObjectName("primary")
        build_sweep.clicked.connect(self._generate_sweep_recipe)
        sweep_layout.addRow(build_sweep)
        layout.addWidget(sweep_box)

        tip = QtWidgets.QLabel(
            "Build the recipe here; ordinary stage/laser control remains on "
            "the Control tab. Drag steps in the sequence to reorder them."
        )
        tip.setObjectName("muted")
        tip.setWordWrap(True)
        layout.addWidget(tip)
        layout.addStretch(1)
        return panel

    def _build_script_sequence(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QtWidgets.QHBoxLayout()
        header.addWidget(self._section("SEQUENCE"))
        self.recipe_name = QtWidgets.QLineEdit("Untitled recipe")
        self.recipe_name.setPlaceholderText("Recipe name")
        header.addWidget(self.recipe_name, 1)
        layout.addLayout(header)

        self.recipe_list = RecipeList()
        self.recipe_list.setMinimumHeight(420)
        self.recipe_list.orderChanged.connect(
            self._update_recipe_preflight_view
        )
        layout.addWidget(self.recipe_list, 1)

        edit_row = QtWidgets.QHBoxLayout()
        remove = QtWidgets.QPushButton("Remove selected")
        remove.clicked.connect(self._recipe_remove)
        duplicate = QtWidgets.QPushButton("Duplicate selected")
        duplicate.clicked.connect(self._recipe_duplicate)
        clear = QtWidgets.QPushButton("Clear recipe")
        clear.clicked.connect(self._recipe_clear)
        edit_row.addWidget(remove)
        edit_row.addWidget(duplicate)
        edit_row.addWidget(clear)
        edit_row.addStretch(1)
        layout.addLayout(edit_row)

        file_row = QtWidgets.QHBoxLayout()
        save = QtWidgets.QPushButton("Save recipe…")
        save.clicked.connect(self._save_recipe)
        load = QtWidgets.QPushButton("Load recipe…")
        load.clicked.connect(self._load_recipe)
        file_row.addWidget(save)
        file_row.addWidget(load)
        file_row.addStretch(1)
        layout.addLayout(file_row)
        return panel

    def _build_script_run_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        summary_box = QtWidgets.QGroupBox("EXECUTION TARGET")
        summary_layout = QtWidgets.QVBoxLayout(summary_box)
        self.execution_summary = QtWidgets.QLabel("")
        self.execution_summary.setWordWrap(True)
        self.execution_summary.setObjectName("statusPill")
        summary_layout.addWidget(self.execution_summary)
        layout.addWidget(summary_box)

        preflight_box = QtWidgets.QGroupBox("PREFLIGHT")
        preflight_layout = QtWidgets.QVBoxLayout(preflight_box)
        self.preflight_text = QtWidgets.QPlainTextEdit()
        self.preflight_text.setReadOnly(True)
        self.preflight_text.setMinimumHeight(220)
        preflight_layout.addWidget(self.preflight_text)
        refresh = QtWidgets.QPushButton("REFRESH PREFLIGHT")
        refresh.clicked.connect(self._update_recipe_preflight_view)
        preflight_layout.addWidget(refresh)
        layout.addWidget(preflight_box, 1)

        self.real_script_arm = QtWidgets.QCheckBox(
            "ARM REAL SCRIPT EXECUTION"
        )
        self.real_script_arm.setObjectName("dangerCheck")
        self.real_script_arm.setToolTip(
            "Required if any step will command real hardware."
        )
        layout.addWidget(self.real_script_arm)

        self.run_recipe_btn = QtWidgets.QPushButton(
            "▶ RUN SCRIPT ON CURRENT PROVIDERS"
        )
        self.run_recipe_btn.setObjectName("primary")
        self.run_recipe_btn.setMinimumHeight(44)
        self.run_recipe_btn.clicked.connect(self._run_recipe)
        layout.addWidget(self.run_recipe_btn)

        self.stop_recipe_btn = QtWidgets.QPushButton(
            "■ STOP SCRIPT + CLOSE BEAM"
        )
        self.stop_recipe_btn.setObjectName("danger")
        self.stop_recipe_btn.clicked.connect(self._stop_recipe)
        layout.addWidget(self.stop_recipe_btn)

        self.recipe_progress = QtWidgets.QLabel("Ready")
        self.recipe_progress.setObjectName("statusPill")
        self.recipe_progress.setWordWrap(True)
        layout.addWidget(self.recipe_progress)

        note = QtWidgets.QLabel(
            "A script always starts by requesting the Pockels cell CLOSED. "
            "Real-hardware execution is blocked unless all required providers "
            "are connected and the real-script arm box is checked."
        )
        note.setObjectName("muted")
        note.setWordWrap(True)
        layout.addWidget(note)
        return panel

    def _build_setup_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)

        hxp_box = QtWidgets.QGroupBox("NEWPORT HXP")
        hxp = QtWidgets.QFormLayout(hxp_box)
        self.hxp_host = QtWidgets.QLineEdit("192.168.0.254")
        self.hxp_port = QtWidgets.QSpinBox()
        self.hxp_port.setRange(1, 65535)
        self.hxp_port.setValue(5001)
        self.hxp_timeout = QtWidgets.QDoubleSpinBox()
        self.hxp_timeout.setRange(0.1, 60.0)
        self.hxp_timeout.setDecimals(1)
        self.hxp_timeout.setValue(10.0)
        self.hxp_timeout.setSuffix(" s")
        self.hxp_group = QtWidgets.QLineEdit("HEXAPOD")
        self.hxp_coords = QtWidgets.QComboBox()
        self.hxp_coords.addItems(["Work", "Tool"])
        hxp.addRow("IP address", self.hxp_host)
        hxp.addRow("Port", self.hxp_port)
        hxp.addRow("Timeout", self.hxp_timeout)
        hxp.addRow("Group", self.hxp_group)
        hxp.addRow("Move frame", self.hxp_coords)
        hxp_buttons = QtWidgets.QHBoxLayout()
        cb = QtWidgets.QPushButton("Connect HXP")
        cb.clicked.connect(self._connect_real_hxp)
        db = QtWidgets.QPushButton("Disconnect")
        db.clicked.connect(self._disconnect_real_hxp)
        hxp_buttons.addWidget(cb)
        hxp_buttons.addWidget(db)
        hxp.addRow(hxp_buttons)
        left_layout.addWidget(hxp_box)

        pockels_box = QtWidgets.QGroupBox("POCKELS CELL / PHAROS LX13")
        pockels = QtWidgets.QFormLayout(pockels_box)
        self.laser_mode = QtWidgets.QComboBox()
        self.laser_mode.addItems(
            ["Virtual Pockels cell", "Real HXP GPIO → PHAROS LX13"]
        )
        self.laser_mode.currentIndexChanged.connect(
            self._laser_mode_changed
        )
        self.gpio_name = QtWidgets.QLineEdit("")
        self.gpio_mask = QtWidgets.QSpinBox()
        self.gpio_mask.setRange(0, 65535)
        self.gpio_open = QtWidgets.QSpinBox()
        self.gpio_open.setRange(0, 65535)
        self.gpio_closed = QtWidgets.QSpinBox()
        self.gpio_closed.setRange(0, 65535)
        pockels.addRow("Provider", self.laser_mode)
        pockels.addRow("HXP GPIO name", self.gpio_name)
        pockels.addRow("Mask", self.gpio_mask)
        pockels.addRow("OPEN value", self.gpio_open)
        pockels.addRow("CLOSED value", self.gpio_closed)
        self.wiring_verified = QtWidgets.QCheckBox(
            "I verified LX13 pinout, active level and HXP electrical compatibility"
        )
        self.wiring_verified.setWordWrap(True)
        pockels.addRow(self.wiring_verified)
        pockels_buttons = QtWidgets.QHBoxLayout()
        arm = QtWidgets.QPushButton("Connect / arm provider")
        arm.clicked.connect(self._connect_laser)
        close = QtWidgets.QPushButton("Request CLOSED")
        close.setObjectName("safeAction")
        close.clicked.connect(self._close_all_pockels)
        pockels_buttons.addWidget(arm)
        pockels_buttons.addWidget(close)
        pockels.addRow(pockels_buttons)
        left_layout.addWidget(pockels_box)

        att_box = QtWidgets.QGroupBox("ATTENUATOR HARDWARE")
        att = QtWidgets.QFormLayout(att_box)
        self.attenuator_mode = QtWidgets.QComboBox()
        self.attenuator_mode.addItems(
            [
                "Virtual attenuator",
                "Real attenuator — driver not configured",
            ]
        )
        self.attenuator_mode.currentIndexChanged.connect(
            self._attenuator_mode_changed
        )
        self.attenuator_model = QtWidgets.QLineEdit("")
        self.attenuator_model.setPlaceholderText(
            "Enter make/model/controller when known"
        )
        self.attenuator_driver_status = QtWidgets.QLabel(
            "Virtual normalized transmission control is ready. "
            "Real device binding is intentionally blocked until the "
            "attenuator hardware/protocol and calibration are known."
        )
        self.attenuator_driver_status.setWordWrap(True)
        self.attenuator_driver_status.setObjectName("muted")
        att.addRow("Provider", self.attenuator_mode)
        att.addRow("Hardware", self.attenuator_model)
        att.addRow(self.attenuator_driver_status)
        connect_att = QtWidgets.QPushButton("Connect attenuator")
        connect_att.clicked.connect(self._connect_attenuator)
        att.addRow(connect_att)
        left_layout.addWidget(att_box)

        config_box = QtWidgets.QGroupBox("CONFIGURATION")
        config_layout = QtWidgets.QHBoxLayout(config_box)
        load_cfg = QtWidgets.QPushButton("Load config…")
        load_cfg.clicked.connect(self._load_hardware_config_dialog)
        save_cfg = QtWidgets.QPushButton("Save config…")
        save_cfg.clicked.connect(self._save_hardware_config_dialog)
        config_layout.addWidget(load_cfg)
        config_layout.addWidget(save_cfg)
        left_layout.addWidget(config_box)
        left_layout.addStretch(1)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)

        cad_box = QtWidgets.QGroupBox("CAD / DIGITAL TWIN")
        cad_layout = QtWidgets.QVBoxLayout(cad_box)
        self.setup_cad_status = QtWidgets.QLabel(
            "Using CAD-derived rig profile. Load the supplied STEP for exact surfaces."
        )
        self.setup_cad_status.setWordWrap(True)
        self.setup_cad_status.setObjectName("muted")
        cad_layout.addWidget(self.setup_cad_status)
        load_step = QtWidgets.QPushButton(
            "Load Stewart Platform STEP…"
        )
        load_step.clicked.connect(self._load_step_dialog)
        cad_layout.addWidget(load_step)
        right_layout.addWidget(cad_box)

        diagnostics = QtWidgets.QGroupBox("DIAGNOSTICS")
        diag_layout = QtWidgets.QVBoxLayout(diagnostics)
        diag_buttons = QtWidgets.QHBoxLayout()
        firmware = QtWidgets.QPushButton("Read HXP firmware")
        firmware.clicked.connect(self._diagnostic_firmware)
        poll = QtWidgets.QPushButton("Poll HXP pose")
        poll.clicked.connect(self._diagnostic_pose)
        clear_log = QtWidgets.QPushButton("Clear log")
        clear_log.clicked.connect(lambda: self.diag_log.clear())
        diag_buttons.addWidget(firmware)
        diag_buttons.addWidget(poll)
        diag_buttons.addWidget(clear_log)
        diag_layout.addLayout(diag_buttons)
        self.diag_log = QtWidgets.QPlainTextEdit()
        self.diag_log.setReadOnly(True)
        diag_layout.addWidget(self.diag_log, 1)
        right_layout.addWidget(diagnostics, 1)

        safety = QtWidgets.QGroupBox("SAFETY BOUNDARY")
        safety_layout = QtWidgets.QVBoxLayout(safety)
        safety_text = QtWidgets.QLabel(
            "This GUI can request motion and process-beam state, but it is not "
            "a safety PLC, laser interlock, or hardware emergency stop. "
            "The physical PHAROS interlock/shutter chain and the hardware E-stop "
            "remain authoritative. Real LX13 control is intentionally impossible "
            "until its wiring values are explicitly verified."
        )
        safety_text.setWordWrap(True)
        safety_text.setObjectName("muted")
        safety_layout.addWidget(safety_text)
        right_layout.addWidget(safety)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.addWidget(self._scroll_wrap(left))
        split.addWidget(right)
        split.setSizes([650, 850])
        layout.addWidget(split)
        return tab

    def _scroll_wrap(self, widget: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
        area = QtWidgets.QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        area.setWidget(widget)
        return area

    def _section(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("section")
        return label

    def _pose_spinbox(self, index: int) -> QtWidgets.QDoubleSpinBox:
        box = QtWidgets.QDoubleSpinBox()
        box.setDecimals(4)
        if index < 3:
            box.setRange(-1000.0, 1000.0)
            box.setSingleStep(0.1)
            box.setSuffix(" mm")
        else:
            box.setRange(-180.0, 180.0)
            box.setSingleStep(0.1)
            box.setSuffix(" °")
        return box

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #0f151c;
                color: #e8edf2;
                font-size: 13px;
            }
            QLabel#title {
                font-size: 22px;
                font-weight: 800;
                letter-spacing: 1px;
            }
            QLabel#headerLabel, QLabel#section {
                color: #93a2b0;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 1px;
            }
            QLabel#muted { color: #8794a0; }
            QLabel#mono {
                font-family: Consolas, 'Courier New';
                font-size: 12px;
            }
            QLabel#statusPill {
                background:#1c2630;
                border:1px solid #34414d;
                border-radius:6px;
                padding:7px;
            }
            QLabel#chipNeutral, QLabel#chipSafe, QLabel#chipLive,
            QLabel#chipWarn {
                border-radius:7px;
                padding:7px 10px;
                font-weight:700;
            }
            QLabel#chipNeutral {
                background:#1c2630;
                border:1px solid #3a4855;
                color:#d4dce3;
            }
            QLabel#chipSafe {
                background:#183427;
                border:1px solid #326c49;
                color:#b8efca;
            }
            QLabel#chipLive {
                background:#5b361a;
                border:1px solid #b16b2f;
                color:#ffe0a9;
            }
            QLabel#chipWarn {
                background:#492426;
                border:1px solid #8e4549;
                color:#ffc4c4;
            }
            QLabel#laserOff {
                background:#183427;
                border:1px solid #326c49;
                color:#b8efca;
                border-radius:7px;
                padding:10px;
                font-weight:800;
            }
            QLabel#laserOn {
                background:#5b361a;
                border:1px solid #b16b2f;
                color:#ffe0a9;
                border-radius:7px;
                padding:10px;
                font-weight:800;
            }
            QPushButton, QComboBox, QLineEdit, QSpinBox,
            QDoubleSpinBox, QPlainTextEdit {
                background:#18212a;
                border:1px solid #35424f;
                border-radius:5px;
                padding:6px;
                min-height:23px;
            }
            QPushButton:hover { border-color:#687a8a; }
            QPushButton:disabled {
                color:#5f6972;
                background:#151b21;
                border-color:#252d35;
            }
            QPushButton#primary {
                background:#214d69;
                border-color:#3f80a6;
                font-weight:800;
            }
            QPushButton#danger {
                background:#592627;
                border-color:#9a4446;
                color:#ffd0d0;
                font-weight:800;
            }
            QPushButton#safeAction {
                background:#1c402d;
                border-color:#397a55;
                color:#c8f4d7;
                font-weight:800;
            }
            QPushButton#beamOn, QPushButton#beamOnSmall {
                background:#603818;
                border:1px solid #b97131;
                color:#ffe1a9;
                font-weight:900;
            }
            QPushButton#beamOff, QPushButton#beamOffSmall {
                background:#1b432e;
                border:1px solid #3c8059;
                color:#c5f1d4;
                font-weight:900;
            }
            QGroupBox {
                border:1px solid #2e3944;
                border-radius:8px;
                margin-top:11px;
                padding-top:11px;
                font-weight:800;
                color:#a9b6c2;
            }
            QTabWidget::pane {
                border:1px solid #28333d;
                border-radius:6px;
            }
            QTabBar::tab {
                background:#151d25;
                border:1px solid #29343f;
                padding:9px 20px;
                margin-right:2px;
                font-weight:700;
                color:#8f9da9;
            }
            QTabBar::tab:selected {
                background:#22303d;
                color:#eef3f7;
                border-bottom:2px solid #4d8bb3;
            }
            QListWidget {
                background:#0c1218;
                border:1px solid #303b46;
                alternate-background-color:#121a22;
            }
            QListWidget::item {
                padding:8px;
                border-bottom:1px solid #202a33;
            }
            QListWidget::item:selected {
                background:#23445a;
                color:#f2f7fb;
            }
            QCheckBox#dangerCheck {
                color:#ffb5ae;
                font-weight:800;
            }
            QSplitter::handle {
                background:#26313b;
                width:2px;
                height:2px;
            }
            QSlider::groove:horizontal {
                height:6px;
                background:#26313b;
                border-radius:3px;
            }
            QSlider::handle:horizontal {
                width:15px;
                margin:-5px 0;
                background:#73a9ce;
                border-radius:7px;
            }
            """
        )

    # ------------------------------------------------------------- providers
    def _stage_provider(self) -> HexapodProvider:
        if self.stage_mode.currentIndex() == 0:
            return self.virtual_stage
        if self.real_stage is None or not self.real_stage.client.connected:
            raise ConnectionError(
                "Real HXP is selected but is not connected"
            )
        return self.real_stage

    def _laser_provider(self) -> LaserGateProvider:
        if self.laser_mode.currentIndex() == 0:
            return self.virtual_laser
        if self.real_laser is None:
            raise ConnectionError(
                "Real PHAROS LX13/Pockels provider is selected but not armed"
            )
        snap = self.real_laser.snapshot()
        if not snap.connected:
            raise ConnectionError(
                "Real PHAROS LX13/Pockels provider is not connected"
            )
        return self.real_laser

    def _attenuator_provider(self) -> AttenuatorProvider:
        if self.attenuator_mode.currentIndex() == 0:
            return self.virtual_attenuator
        return self.real_attenuator

    def _laser_snapshot(self) -> LaserSnapshot:
        if self.laser_mode.currentIndex() == 0:
            return self.virtual_laser.snapshot()
        if self.real_laser is None:
            return LaserSnapshot(
                timestamp_s=time.time(),
                gate_enabled=False,
                connected=False,
                provider="hxp-lx13-real",
                connector_name="PHAROS LX13",
                readback_known=False,
            )
        return self.real_laser.snapshot()

    def _attenuator_snapshot(self) -> AttenuatorSnapshot:
        if self.attenuator_mode.currentIndex() == 0:
            return self.virtual_attenuator.snapshot()
        return self.real_attenuator.snapshot()

    # ---------------------------------------------------------- configuration
    def _config_dict(self) -> dict:
        return {
            "hxp": {
                "host": self.hxp_host.text().strip(),
                "port": self.hxp_port.value(),
                "timeout_s": self.hxp_timeout.value(),
                "group": self.hxp_group.text().strip() or "HEXAPOD",
                "coordinate_system": self.hxp_coords.currentText(),
            },
            "pockels": {
                "connector": "PHAROS LX13",
                "provider": (
                    "virtual"
                    if self.laser_mode.currentIndex() == 0
                    else "hxp_gpio"
                ),
                "gpio_name": self.gpio_name.text().strip(),
                "mask": self.gpio_mask.value(),
                "open_value": self.gpio_open.value(),
                "closed_value": self.gpio_closed.value(),
                "wiring_verified": self.wiring_verified.isChecked(),
            },
            "attenuator": {
                "provider": (
                    "virtual"
                    if self.attenuator_mode.currentIndex() == 0
                    else "unconfigured_real"
                ),
                "model": self.attenuator_model.text().strip(),
                "transmission_percent": self.attenuator_spin.value(),
            },
        }

    def _load_config_dict(self, cfg: dict) -> None:
        hxp = cfg.get("hxp", {})
        self.hxp_host.setText(str(hxp.get("host", self.hxp_host.text())))
        self.hxp_port.setValue(int(hxp.get("port", self.hxp_port.value())))
        self.hxp_timeout.setValue(
            float(
                hxp.get(
                    "timeout_s",
                    float(hxp.get("timeout_ms", 10000)) / 1000.0,
                )
            )
        )
        self.hxp_group.setText(
            str(hxp.get("group", self.hxp_group.text()))
        )
        coord = str(
            hxp.get("coordinate_system", self.hxp_coords.currentText())
        )
        idx = self.hxp_coords.findText(coord)
        if idx >= 0:
            self.hxp_coords.setCurrentIndex(idx)

        pockels = cfg.get("pockels", cfg.get("laser", {}))
        provider = str(pockels.get("provider", "virtual"))
        self.laser_mode.setCurrentIndex(
            1 if provider in {"hxp_gpio", "real"} else 0
        )
        self.gpio_name.setText(str(pockels.get("gpio_name", "")))
        self.gpio_mask.setValue(int(pockels.get("mask", 0)))
        self.gpio_open.setValue(
            int(
                pockels.get(
                    "open_value",
                    pockels.get("enabled_value", 0),
                )
            )
        )
        self.gpio_closed.setValue(
            int(
                pockels.get(
                    "closed_value",
                    pockels.get("disabled_value", 0),
                )
            )
        )
        self.wiring_verified.setChecked(
            bool(pockels.get("wiring_verified", False))
        )

        att = cfg.get("attenuator", {})
        self.attenuator_mode.setCurrentIndex(
            1 if att.get("provider") == "unconfigured_real" else 0
        )
        self.attenuator_model.setText(str(att.get("model", "")))
        transmission = float(att.get("transmission_percent", 0.0))
        self.attenuator_spin.setValue(
            max(0.0, min(100.0, transmission))
        )
        if self.attenuator_mode.currentIndex() == 0:
            self.virtual_attenuator.set_transmission_percent(
                self.attenuator_spin.value()
            )

    def _load_default_config(self) -> None:
        if not DEFAULT_CONFIG.is_file():
            return
        try:
            self._load_config_dict(
                json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
            )
        except Exception as exc:
            self.statusBar().showMessage(f"Config warning: {exc}")

    def _load_hardware_config_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load hardware configuration",
            str(APP_ROOT),
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            cfg = json.loads(Path(path).read_text(encoding="utf-8"))
            self._load_config_dict(cfg)
            self.statusBar().showMessage(
                f"Loaded configuration: {Path(path).name}",
                6000,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Configuration load failed",
                str(exc),
            )

    def _save_hardware_config_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save hardware configuration",
            str(APP_ROOT / "hardware_config.local.json"),
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            Path(path).write_text(
                json.dumps(self._config_dict(), indent=2),
                encoding="utf-8",
            )
            self.statusBar().showMessage(
                f"Saved configuration: {Path(path).name}",
                6000,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Configuration save failed",
                str(exc),
            )

    # --------------------------------------------------------------- HXP
    def _stage_mode_changed(self, _index: int) -> None:
        self.real_script_arm.setChecked(False)
        self.manual_beam_arm.setChecked(False)
        if self.stage_mode.currentIndex() == 0:
            if not self.virtual_stage.snapshot().connected:
                self.virtual_stage.connect()
            self._last_stage_snapshot = self.virtual_stage.snapshot()
        elif (
            self.real_stage is not None
            and self.real_stage.client.connected
        ):
            try:
                self._last_stage_snapshot = self.real_stage.snapshot()
            except Exception:
                pass
        else:
            self._last_stage_snapshot = HexapodSnapshot(
                timestamp_s=time.time(),
                actual=self._last_stage_snapshot.actual,
                state=MotionState.DISCONNECTED,
                connected=False,
                provider="hxp-real",
                status_text="REAL HXP NOT CONNECTED",
            )
        self._update_recipe_preflight_view()

    def _connect_stage(self) -> None:
        if self.stage_mode.currentIndex() == 0:
            self.virtual_stage.connect()
            self._last_stage_snapshot = self.virtual_stage.snapshot()
            self.statusBar().showMessage(
                "Virtual hexapod connected",
                4000,
            )
            return
        self._connect_real_hxp()

    def _connect_real_hxp(self) -> None:
        try:
            cfg = HXPProviderConfig(
                host=self.hxp_host.text().strip(),
                port=self.hxp_port.value(),
                group=self.hxp_group.text().strip() or "HEXAPOD",
                coordinate_system=self.hxp_coords.currentText(),
                timeout_s=self.hxp_timeout.value(),
            )
            QtWidgets.QApplication.setOverrideCursor(
                QtCore.Qt.CursorShape.WaitCursor
            )

            # A real Pockels provider owns HXP I/O sockets. Tear it down before
            # replacing/reconnecting the HXP client so no stale client survives.
            if self.real_laser is not None:
                try:
                    self.real_laser.disconnect()
                except Exception:
                    pass
                self.real_laser = None

            if self.real_stage is not None:
                try:
                    self.real_stage.disconnect()
                except Exception:
                    pass

            self.real_stage = HXPProvider(cfg)
            self.real_stage.connect()
            if self.stage_mode.currentIndex() == 1:
                self._last_stage_snapshot = self.real_stage.snapshot()
            self._diag(
                f"Connected HXP {cfg.host}:{cfg.port}; "
                f"group={cfg.group}, frame={cfg.coordinate_system}"
            )
            self.statusBar().showMessage(
                f"Connected to HXP {cfg.host}:{cfg.port}",
                5000,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "HXP connection failed",
                str(exc),
            )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._update_recipe_preflight_view()

    def _disconnect_stage(self) -> None:
        if self.stage_mode.currentIndex() == 0:
            self.virtual_stage.disconnect()
            self.statusBar().showMessage(
                "Virtual stage disconnected",
                4000,
            )
            return
        self._disconnect_real_hxp()

    def _disconnect_real_hxp(self) -> None:
        try:
            if self.real_laser is not None:
                try:
                    self.real_laser.disconnect()
                finally:
                    self.real_laser = None
            if self.real_stage is not None:
                self.real_stage.disconnect()
            self.statusBar().showMessage(
                "Real HXP disconnected; real Pockels provider disarmed",
                5000,
            )
        except Exception as exc:
            self.statusBar().showMessage(
                f"Disconnect warning: {exc}",
                6000,
            )
        self._update_recipe_preflight_view()

    def _target_pose(self) -> Pose6D:
        return Pose6D(
            *(self.pose_boxes[axis].value() for axis in "XYZUVW")
        )

    def _move_absolute(self) -> None:
        try:
            self._stage_provider().move_absolute(self._target_pose())
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Move failed",
                str(exc),
            )

    def _target_from_actual(self) -> None:
        for axis, value in zip(
            "XYZUVW",
            self._last_stage_snapshot.actual.as_tuple(),
        ):
            self.pose_boxes[axis].setValue(value)

    def _jog(self, axis: str, sign: float) -> None:
        values = [0.0] * 6
        idx = "XYZUVW".index(axis)
        values[idx] = sign * (
            self.jog_mm.value() if idx < 3 else self.jog_deg.value()
        )
        try:
            self._stage_provider().move_incremental(
                Pose6D.from_iterable(values)
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Jog failed",
                str(exc),
            )

    def _home(self) -> None:
        real = self.stage_mode.currentIndex() == 1
        if real:
            answer = QtWidgets.QMessageBox.warning(
                self,
                "Home real HXP?",
                "A real HXP home search can move all six struts through a "
                "substantial path. Confirm the physical setup is clear.",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        try:
            self._stage_provider().home()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Home failed",
                str(exc),
            )

    def _abort_all(self) -> None:
        self._close_all_pockels()
        try:
            self._stage_provider().abort()
        except Exception as exc:
            self.statusBar().showMessage(
                f"Motion-abort warning: {exc}",
                7000,
            )
        self._recipe_running = False
        self.recipe_progress.setText(
            "STOPPED — close-beam command issued"
        )

    # ------------------------------------------------------ Pockels / laser
    def _laser_mode_changed(self, _index: int) -> None:
        # Mode changes must never leave a previously-selected real Pockels
        # command OPEN and then hide that provider from the operator.
        self._close_all_pockels()
        self.manual_beam_arm.setChecked(False)
        self.real_script_arm.setChecked(False)
        if self.laser_mode.currentIndex() == 0:
            if not self.virtual_laser.snapshot().connected:
                self.virtual_laser.connect()
        self._update_recipe_preflight_view()

    def _connect_laser(self) -> None:
        try:
            if self.laser_mode.currentIndex() == 0:
                self.virtual_laser.connect()
                self.virtual_laser.set_gate(False)
                self.statusBar().showMessage(
                    "Virtual Pockels provider connected",
                    4000,
                )
                return

            if (
                self.real_stage is None
                or not self.real_stage.client.connected
            ):
                raise RuntimeError(
                    "Connect the real HXP first. The current LX13 implementation "
                    "uses an HXP digital-output socket."
                )

            cfg = HXPDigitalLaserConfig(
                gpio_name=self.gpio_name.text().strip(),
                mask=self.gpio_mask.value(),
                enabled_value=self.gpio_open.value(),
                disabled_value=self.gpio_closed.value(),
                connector_name="PHAROS LX13 / Pockels cell",
                wiring_verified=self.wiring_verified.isChecked(),
            )
            self.real_laser = HXPDigitalLaserGate(
                self.real_stage.client,
                cfg,
            )
            self.real_laser.connect()
            self._diag(
                "Real LX13/Pockels provider armed; CLOSED state requested"
            )
            self.statusBar().showMessage(
                "Real LX13/Pockels provider armed; initial state CLOSED",
                6000,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Pockels provider not armed",
                str(exc),
            )
        self._update_recipe_preflight_view()

    def _set_pockels(
        self,
        open_: bool,
        *,
        manual: bool = False,
        script: bool = False,
    ) -> None:
        try:
            if (
                open_
                and self.laser_mode.currentIndex() == 1
                and manual
                and not self.manual_beam_arm.isChecked()
            ):
                raise RuntimeError(
                    "Real manual beam control is not armed. "
                    "Check ARM MANUAL REAL-BEAM CONTROL first."
                )
            if (
                open_
                and self.laser_mode.currentIndex() == 1
                and script
                and not self.real_script_arm.isChecked()
            ):
                raise RuntimeError(
                    "Real script execution is not armed"
                )
            self._laser_provider().set_gate(bool(open_))
            self.statusBar().showMessage(
                "Pockels OPEN — process beam enabled"
                if open_
                else "Pockels CLOSED — process beam disabled",
                3500,
            )
        except Exception as exc:
            if script:
                raise
            QtWidgets.QMessageBox.critical(
                self,
                "Pockels command failed",
                str(exc),
            )

    def _close_all_pockels(self) -> None:
        # Request closure on every provider we may have touched. This prevents
        # mode switching from leaving a previously-selected provider open.
        for provider in (self.virtual_laser, self.real_laser):
            if provider is None:
                continue
            try:
                provider.safe_off()
            except Exception:
                pass
        self.statusBar().showMessage(
            "Close-beam request issued",
            3500,
        )

    # ----------------------------------------------------------- attenuator
    def _attenuator_mode_changed(self, _index: int) -> None:
        self.real_script_arm.setChecked(False)
        if self.attenuator_mode.currentIndex() == 0:
            if not self.virtual_attenuator.snapshot().connected:
                self.virtual_attenuator.connect()
        self._update_recipe_preflight_view()

    def _connect_attenuator(self) -> None:
        try:
            self._attenuator_provider().connect()
            self.statusBar().showMessage(
                "Attenuator provider connected",
                4000,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.information(
                self,
                "Attenuator driver required",
                str(exc),
            )
        self._update_recipe_preflight_view()

    def _attenuator_preset(self, value: float) -> None:
        self.attenuator_spin.setValue(float(value))

    def _set_attenuator_manual(self) -> None:
        try:
            self._set_attenuator(
                self.attenuator_spin.value(),
                script=False,
            )
        except Exception:
            pass

    def _set_attenuator(
        self,
        value: float,
        *,
        script: bool = False,
    ) -> None:
        if (
            self.attenuator_mode.currentIndex() == 1
            and script
            and not self.real_script_arm.isChecked()
        ):
            raise RuntimeError(
                "Real script execution is not armed"
            )
        provider = self._attenuator_provider()
        provider.set_transmission_percent(float(value))
        self.statusBar().showMessage(
            f"Attenuator setpoint {float(value):.1f} %",
            3000,
        )

    # --------------------------------------------------------------- CAD
    def _load_step_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Stewart-platform STEP",
            str(APP_ROOT),
            "STEP files (*.step *.stp *.STEP *.STP)",
        )
        if not path:
            return
        try:
            QtWidgets.QApplication.setOverrideCursor(
                QtCore.Qt.CursorShape.WaitCursor
            )
            self.viewer.load_step(path)
            label = f"Exact STEP CAD • {Path(path).name}"
            self.cad_status.setText(label)
            self.setup_cad_status.setText(label)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "CAD load failed",
                str(exc),
            )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _cad_message(self, message: str) -> None:
        self.cad_status.setText(message)
        self.setup_cad_status.setText(message)
        self.statusBar().showMessage(message, 8000)

    def _clear_visual_traces(self) -> None:
        self.viewer.clear_traces()
        self.movement_map.clear()

    # -------------------------------------------------------------- recipes
    def _sync_recipe_from_list(self) -> None:
        self.recipe.name = self.recipe_name.text().strip() or "Untitled recipe"
        steps: list[RecipeStep] = []
        for i in range(self.recipe_list.count()):
            item = self.recipe_list.item(i)
            data = item.data(QtCore.Qt.ItemDataRole.UserRole)
            steps.append(RecipeStep.from_dict(data))
        self.recipe.steps = steps

    def _refresh_recipe_list(self) -> None:
        self.recipe_list.clear()
        for step in self.recipe.steps:
            item = QtWidgets.QListWidgetItem(step.describe())
            item.setData(
                QtCore.Qt.ItemDataRole.UserRole,
                step.to_dict(),
            )
            self.recipe_list.addItem(item)
        self._update_recipe_preflight_view()

    def _append_recipe_step(self, step: RecipeStep) -> None:
        self._sync_recipe_from_list()
        self.recipe.steps.append(step)
        self._refresh_recipe_list()
        self.recipe_list.setCurrentRow(self.recipe_list.count() - 1)

    def _recipe_add_move(self) -> None:
        pose = Pose6D(
            *(
                self.script_pose_boxes[axis].value()
                for axis in "XYZUVW"
            )
        )
        kind = self.script_move_kind.currentIndex()
        if kind == 0:
            self._append_recipe_step(
                RecipeStep.move_absolute(pose)
            )
        elif kind == 1:
            self._append_recipe_step(
                RecipeStep.move_incremental(pose)
            )
        else:
            self._append_recipe_step(
                RecipeStep.move_line_velocity(
                    pose.x,
                    pose.y,
                    pose.z,
                    self.script_line_velocity.value(),
                )
            )

    def _script_pose_from_live(self) -> None:
        for axis, value in zip(
            "XYZUVW",
            self._last_stage_snapshot.actual.as_tuple(),
        ):
            self.script_pose_boxes[axis].setValue(value)

    def _recipe_add_attenuator(self) -> None:
        self._append_recipe_step(
            RecipeStep.attenuator_set(
                self.script_attenuator.value()
            )
        )

    def _recipe_add_wait(self) -> None:
        self._append_recipe_step(
            RecipeStep.wait(self.script_wait.value())
        )

    def _recipe_remove(self) -> None:
        self._sync_recipe_from_list()
        row = self.recipe_list.currentRow()
        if row < 0:
            return
        self.recipe.steps.pop(row)
        self._refresh_recipe_list()
        if self.recipe.steps:
            self.recipe_list.setCurrentRow(
                min(row, len(self.recipe.steps) - 1)
            )

    def _recipe_duplicate(self) -> None:
        self._sync_recipe_from_list()
        row = self.recipe_list.currentRow()
        if row < 0:
            return
        clone = RecipeStep.from_dict(
            self.recipe.steps[row].to_dict()
        )
        self.recipe.steps.insert(row + 1, clone)
        self._refresh_recipe_list()
        self.recipe_list.setCurrentRow(row + 1)

    def _recipe_clear(self) -> None:
        self._sync_recipe_from_list()
        if not self.recipe.steps and self.recipe_list.count() == 0:
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            "Clear recipe?",
            "Remove every step from the current recipe?",
        )
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            self.recipe.steps.clear()
            self._refresh_recipe_list()

    def _recipe_hardware_issues(self) -> list[tuple[str, str]]:
        self._sync_recipe_from_list()
        kinds = {step.kind for step in self.recipe.steps}
        issues: list[tuple[str, str]] = []

        uses_stage = bool(
            kinds
            & {
                StepKind.MOVE_ABSOLUTE,
                StepKind.MOVE_INCREMENTAL,
                StepKind.MOVE_LINE_VELOCITY,
            }
        )
        uses_pockels = StepKind.POCKELS_CELL in kinds
        uses_attenuator = StepKind.ATTENUATOR_SET in kinds

        if uses_stage and self.stage_mode.currentIndex() == 1:
            if (
                self.real_stage is None
                or not self.real_stage.client.connected
            ):
                issues.append(
                    ("error", "Real HXP is selected but not connected")
                )

        if uses_pockels and self.laser_mode.currentIndex() == 1:
            if (
                self.real_laser is None
                or not self.real_laser.snapshot().connected
            ):
                issues.append(
                    (
                        "error",
                        "Real LX13/Pockels provider is selected but not armed",
                    )
                )

        if uses_attenuator and self.attenuator_mode.currentIndex() == 1:
            issues.append(
                (
                    "error",
                    "Real attenuator is selected but no device-specific driver "
                    "is configured yet",
                )
            )

        real_paths = []
        virtual_paths = []
        if uses_stage:
            (
                real_paths
                if self.stage_mode.currentIndex() == 1
                else virtual_paths
            ).append("stage")
        if uses_pockels:
            (
                real_paths
                if self.laser_mode.currentIndex() == 1
                else virtual_paths
            ).append("Pockels")
        if uses_attenuator:
            (
                real_paths
                if self.attenuator_mode.currentIndex() == 1
                else virtual_paths
            ).append("attenuator")

        if real_paths and virtual_paths:
            issues.append(
                (
                    "warning",
                    "Mixed execution providers: real "
                    + ", ".join(real_paths)
                    + " with virtual "
                    + ", ".join(virtual_paths),
                )
            )

        return issues

    def _script_uses_real_hardware(self) -> bool:
        self._sync_recipe_from_list()
        for step in self.recipe.steps:
            if (
                step.kind
                in {
                    StepKind.MOVE_ABSOLUTE,
                    StepKind.MOVE_INCREMENTAL,
                    StepKind.MOVE_LINE_VELOCITY,
                }
                and self.stage_mode.currentIndex() == 1
            ):
                return True
            if (
                step.kind == StepKind.POCKELS_CELL
                and self.laser_mode.currentIndex() == 1
            ):
                return True
            if (
                step.kind == StepKind.ATTENUATOR_SET
                and self.attenuator_mode.currentIndex() == 1
            ):
                return True
        return False

    def _update_recipe_preflight_view(self) -> None:
        if not hasattr(self, "preflight_text"):
            return
        self._sync_recipe_from_list()
        issues = preflight_recipe(self.recipe)
        hardware = self._recipe_hardware_issues()

        lines: list[str] = []
        if not issues and not hardware:
            lines.append("PASS — recipe and selected providers are ready.")
        for issue in issues:
            prefix = issue.severity.upper()
            location = (
                f"step {issue.step_index + 1}: "
                if issue.step_index is not None
                else ""
            )
            lines.append(f"{prefix}: {location}{issue.message}")
        for severity, message in hardware:
            lines.append(f"{severity.upper()}: {message}")

        self.preflight_text.setPlainText("\n".join(lines))
        self.execution_summary.setText(
            "Stage: "
            + ("REAL HXP" if self.stage_mode.currentIndex() else "VIRTUAL")
            + "\nPockels: "
            + (
                "REAL LX13"
                if self.laser_mode.currentIndex()
                else "VIRTUAL"
            )
            + "\nAttenuator: "
            + (
                "REAL / UNBOUND"
                if self.attenuator_mode.currentIndex()
                else "VIRTUAL"
            )
        )

    def _run_recipe(self) -> None:
        self._sync_recipe_from_list()
        if not self.recipe.steps:
            QtWidgets.QMessageBox.information(
                self,
                "Empty script",
                "Add at least one step before running the script.",
            )
            return
        issues = preflight_recipe(self.recipe)
        hw = self._recipe_hardware_issues()
        errors = [
            issue
            for issue in issues
            if issue.severity == "error"
        ]
        errors.extend(
            message
            for severity, message in hw
            if severity == "error"
        )
        if errors:
            self._update_recipe_preflight_view()
            QtWidgets.QMessageBox.critical(
                self,
                "Script preflight failed",
                "Fix the preflight errors before running the script.",
            )
            return

        if (
            self._script_uses_real_hardware()
            and not self.real_script_arm.isChecked()
        ):
            QtWidgets.QMessageBox.warning(
                self,
                "Real script not armed",
                "This script will command real hardware. Check "
                "ARM REAL SCRIPT EXECUTION after reviewing preflight.",
            )
            return

        # Known safe beam state before the first recipe step.
        self._close_all_pockels()
        self._recipe_running = True
        self._recipe_index = 0
        self._recipe_step_issued = False
        self._wait_until = 0.0
        self._clear_visual_traces()
        self.recipe_progress.setText(
            f"Running {self.recipe.name}…"
        )
        self.tabs.setCurrentIndex(1)

    def _stop_recipe(self) -> None:
        self._recipe_running = False
        self._close_all_pockels()
        try:
            stage = self._stage_provider()
            if stage.is_busy():
                stage.abort()
        except Exception:
            pass
        self.recipe_progress.setText(
            "Stopped — close-beam command issued"
        )

    def _advance_recipe(self) -> None:
        self._recipe_index += 1
        self._recipe_step_issued = False
        self._wait_until = 0.0
        if self._recipe_index >= len(self.recipe.steps):
            self._recipe_running = False
            self._close_all_pockels()
            self.recipe_progress.setText(
                "Complete — Pockels CLOSED"
            )

    def _recipe_tick(self) -> None:
        if not self._recipe_running:
            return
        if self._recipe_index >= len(self.recipe.steps):
            self._advance_recipe()
            return

        step = self.recipe.steps[self._recipe_index]
        self.recipe_progress.setText(
            f"Step {self._recipe_index + 1}/{len(self.recipe.steps)}  •  "
            f"{step.describe()}"
        )

        try:
            if step.kind == StepKind.MOVE_ABSOLUTE:
                stage = self._stage_provider()
                if not self._recipe_step_issued:
                    stage.move_absolute(
                        Pose6D.from_iterable(step.payload["pose"])
                    )
                    self._recipe_step_issued = True
                elif not stage.is_busy():
                    self._advance_recipe()

            elif step.kind == StepKind.MOVE_INCREMENTAL:
                stage = self._stage_provider()
                if not self._recipe_step_issued:
                    stage.move_incremental(
                        Pose6D.from_iterable(step.payload["delta"])
                    )
                    self._recipe_step_issued = True
                elif not stage.is_busy():
                    self._advance_recipe()

            elif step.kind == StepKind.MOVE_LINE_VELOCITY:
                stage = self._stage_provider()
                if not self._recipe_step_issued:
                    dx, dy, dz = (
                        float(v)
                        for v in step.payload["delta_xyz_mm"]
                    )
                    stage.move_line_incremental_with_target_velocity(
                        dx,
                        dy,
                        dz,
                        float(step.payload["velocity_mm_s"]),
                    )
                    self._recipe_step_issued = True
                elif not stage.is_busy():
                    self._advance_recipe()

            elif step.kind == StepKind.POCKELS_CELL:
                if not self._recipe_step_issued:
                    self._set_pockels(
                        bool(step.payload.get("open")),
                        script=True,
                    )
                    self._recipe_step_issued = True
                    self._advance_recipe()

            elif step.kind == StepKind.ATTENUATOR_SET:
                if not self._recipe_step_issued:
                    self._set_attenuator(
                        float(step.payload["transmission_percent"]),
                        script=True,
                    )
                    self._recipe_step_issued = True
                    self._advance_recipe()

            elif step.kind == StepKind.WAIT:
                if not self._recipe_step_issued:
                    self._wait_until = (
                        time.monotonic()
                        + float(step.payload.get("seconds", 0.0))
                    )
                    self._recipe_step_issued = True
                elif time.monotonic() >= self._wait_until:
                    self._advance_recipe()

        except Exception as exc:
            self._recipe_running = False
            self._close_all_pockels()
            self.recipe_progress.setText(
                f"FAILED at step {self._recipe_index + 1}: {exc}"
            )
            QtWidgets.QMessageBox.critical(
                self,
                "Script failed",
                str(exc),
            )

    def _save_recipe(self) -> None:
        self._sync_recipe_from_list()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save recipe",
            f"{self.recipe.name.replace(' ', '_')}.json",
            "JSON (*.json)",
        )
        if path:
            self.recipe.save(path)

    def _load_recipe(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load recipe",
            "",
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            self.recipe = Recipe.load(path)
            self.recipe_name.setText(self.recipe.name)
            self._refresh_recipe_list()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Recipe load failed",
                str(exc),
            )

    # ----------------------------------------------------------- diagnostics
    def _diag(self, message: str) -> None:
        if not hasattr(self, "diag_log"):
            return
        stamp = time.strftime("%H:%M:%S")
        self.diag_log.appendPlainText(f"[{stamp}] {message}")

    def _diagnostic_firmware(self) -> None:
        try:
            if (
                self.real_stage is None
                or not self.real_stage.client.connected
            ):
                raise ConnectionError("Real HXP is not connected")
            version = self.real_stage.client.firmware_version()
            self._diag(f"HXP firmware: {version}")
        except Exception as exc:
            self._diag(f"Firmware read failed: {exc}")

    def _diagnostic_pose(self) -> None:
        try:
            if (
                self.real_stage is None
                or not self.real_stage.client.connected
            ):
                raise ConnectionError("Real HXP is not connected")
            snap = self.real_stage.snapshot()
            values = ", ".join(
                f"{axis}={value:.5f}"
                for axis, value in zip(
                    "XYZUVW",
                    snap.actual.as_tuple(),
                )
            )
            self._diag(
                f"Current pose: {values}; status={snap.status_text}"
            )
        except Exception as exc:
            self._diag(f"Pose poll failed: {exc}")

    # --------------------------------------------------------------- polling
    def _poll_stage(self, now: float, dt: float) -> None:
        if self.stage_mode.currentIndex() == 0:
            if (
                self._poll_future is not None
                and self._poll_future.done()
            ):
                self._poll_future = None
            if self.virtual_stage.snapshot().connected:
                self.virtual_stage.tick(dt)
                self._last_stage_snapshot = (
                    self.virtual_stage.snapshot()
                )
            else:
                self._last_stage_snapshot = HexapodSnapshot(
                    timestamp_s=time.time(),
                    actual=self._last_stage_snapshot.actual,
                    state=MotionState.DISCONNECTED,
                    connected=False,
                    provider="virtual",
                    status_text="DISCONNECTED",
                )
            return

        if (
            self.real_stage is None
            or not self.real_stage.client.connected
        ):
            self._last_stage_snapshot = HexapodSnapshot(
                timestamp_s=time.time(),
                actual=self._last_stage_snapshot.actual,
                state=MotionState.DISCONNECTED,
                connected=False,
                provider="hxp-real",
                status_text="NOT CONNECTED",
            )
            return

        if self._poll_future is not None and self._poll_future.done():
            try:
                self._last_stage_snapshot = (
                    self._poll_future.result()
                )
            except Exception as exc:
                self.statusBar().showMessage(
                    f"HXP polling error: {exc}",
                    5000,
                )
            self._poll_future = None

        if (
            self._poll_future is None
            and now - self._last_real_poll >= 0.15
        ):
            self._last_real_poll = now
            self._poll_future = self._poll_pool.submit(
                self.real_stage.snapshot
            )

    def _set_object_style(
        self,
        widget: QtWidgets.QWidget,
        object_name: str,
    ) -> None:
        if widget.objectName() == object_name:
            return
        widget.setObjectName(object_name)
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _update_readouts(self) -> None:
        snap = self._last_stage_snapshot
        for axis, value in zip(
            "XYZUVW",
            snap.actual.as_tuple(),
        ):
            unit = " mm" if axis in "XYZ" else " °"
            self.state_labels[axis].setText(
                f"{value:+.4f}{unit}"
            )

        self.motion_status.setText(
            f"{snap.state.value.upper()}  •  "
            f"{snap.status_text or snap.provider}"
        )
        self.manual_stage_status.setText(
            (
                "Connected"
                if snap.connected
                else "NOT CONNECTED"
            )
            + f"  •  {snap.provider}"
        )

        lengths = self.kinematics.leg_lengths_mm(snap.actual)
        home = self.kinematics.leg_lengths_mm(Pose6D())
        for label, length, reference in zip(
            self.leg_labels,
            lengths,
            home,
        ):
            label.setText(
                f"{length:8.3f} mm   Δ {length-reference:+.3f}"
            )

        laser = self._laser_snapshot()
        if laser.pockels_open:
            self.manual_beam_status.setText(
                "LASER ON • POCKELS OPEN"
                + (
                    ""
                    if laser.readback_known
                    else " • COMMAND STATE ONLY"
                )
            )
            self._set_object_style(
                self.manual_beam_status,
                "laserOn",
            )
            self.beam_chip.setText("BEAM OPEN")
            self._set_object_style(
                self.beam_chip,
                "chipLive",
            )
        else:
            suffix = (
                ""
                if laser.connected
                else " • PROVIDER NOT CONNECTED"
            )
            self.manual_beam_status.setText(
                "LASER OFF • POCKELS CLOSED" + suffix
            )
            self._set_object_style(
                self.manual_beam_status,
                "laserOff",
            )
            self.beam_chip.setText(
                "BEAM CLOSED"
                if laser.connected
                else "POCKELS NOT CONNECTED"
            )
            self._set_object_style(
                self.beam_chip,
                "chipSafe" if laser.connected else "chipWarn",
            )

        attenuator = self._attenuator_snapshot()
        if attenuator.connected:
            self.attenuator_status.setText(
                "Transmission setpoint: "
                f"{attenuator.transmission_percent:.1f} %"
                + (
                    ""
                    if attenuator.readback_known
                    else " • command state only"
                )
            )
            self.attenuator_chip.setText(
                f"ATT {attenuator.transmission_percent:.1f} %"
            )
            self._set_object_style(
                self.attenuator_chip,
                "chipNeutral",
            )
        else:
            self.attenuator_status.setText(
                "Real attenuator not bound to a hardware driver"
            )
            self.attenuator_chip.setText("ATT UNBOUND")
            self._set_object_style(
                self.attenuator_chip,
                "chipWarn",
            )

        if snap.state == MotionState.MOVING:
            self.stage_chip.setText("STAGE MOVING")
            self._set_object_style(
                self.stage_chip,
                "chipLive",
            )
        elif snap.connected:
            self.stage_chip.setText("STAGE READY")
            self._set_object_style(
                self.stage_chip,
                "chipSafe",
            )
        else:
            self.stage_chip.setText("STAGE NOT CONNECTED")
            self._set_object_style(
                self.stage_chip,
                "chipWarn",
            )

        self.viewer.update_state(
            snap.actual,
            laser.pockels_open,
        )
        self.movement_map.update_state(
            snap.actual,
            laser.pockels_open,
            self.viewer.beam_hit_sample_xy,
        )

        operator_free = not self._recipe_running
        stage_ready = snap.connected
        motion_ready = stage_ready and operator_free
        self.move_abs_btn.setEnabled(motion_ready)
        self.home_btn.setEnabled(motion_ready)
        for button in self.jog_buttons:
            button.setEnabled(motion_ready)

        laser_ready = laser.connected
        real_manual_ok = (
            self.laser_mode.currentIndex() == 0
            or self.manual_beam_arm.isChecked()
        )
        self.laser_on_btn.setEnabled(
            laser_ready and real_manual_ok and operator_free
        )
        # Closing the process beam remains available even while a script runs.
        self.laser_off_btn.setEnabled(laser_ready)

        att_ready = attenuator.connected
        self.set_attenuator_btn.setEnabled(
            att_ready and operator_free
        )

        # Do not allow provider switching while motion or a script is active.
        provider_switch_safe = (
            not self._recipe_running
            and snap.state != MotionState.MOVING
        )
        self.stage_mode.setEnabled(provider_switch_safe)
        self.laser_mode.setEnabled(not self._recipe_running)
        self.attenuator_mode.setEnabled(not self._recipe_running)
        self.connect_stage_btn.setEnabled(not self._recipe_running)
        self.disconnect_stage_btn.setEnabled(not self._recipe_running)
        self.real_script_arm.setEnabled(not self._recipe_running)

    def _tick(self) -> None:
        now = time.monotonic()
        dt = min(
            0.25,
            max(0.0, now - self._last_tick_monotonic),
        )
        self._last_tick_monotonic = now
        self._poll_stage(now, dt)
        self._recipe_tick()
        self._update_readouts()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            self._close_all_pockels()
            if self.real_stage is not None:
                try:
                    self.real_stage.disconnect()
                except Exception:
                    pass
            self.virtual_stage.disconnect()
            self.virtual_laser.disconnect()
            self.virtual_attenuator.disconnect()
            try:
                self.real_attenuator.disconnect()
            except Exception:
                pass
        finally:
            self._poll_pool.shutdown(
                wait=False,
                cancel_futures=True,
            )
        super().closeEvent(event)
