from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import json
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .kinematics import RigKinematics, RigProfile
from .providers import (
    HXPDigitalLaserConfig,
    HXPDigitalLaserGate,
    HXPProvider,
    HXPProviderConfig,
    HexapodProvider,
    LaserGateProvider,
    VirtualHexapodProvider,
    VirtualLaserGate,
)
from .recipe import Recipe, RecipeStep, StepKind, preflight_recipe
from .types import HexapodSnapshot, MotionState, Pose6D
from .viewer import Hexapod3DViewer


PACKAGE_DIR = Path(__file__).resolve().parent
APP_ROOT = PACKAGE_DIR.parent
ASSETS_DIR = APP_ROOT / "assets"
DEFAULT_PROFILE = ASSETS_DIR / "cad_profile.json"
DEFAULT_CONFIG = APP_ROOT / "hardware_config.example.json"


class RecipeList(QtWidgets.QListWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.setAlternatingRowColors(True)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hexapod + Laser Lab — Standalone Controller")
        self.resize(1650, 1000)

        self.profile = RigProfile.from_json(DEFAULT_PROFILE)
        self.kinematics = RigKinematics(self.profile)
        self.virtual_stage = VirtualHexapodProvider()
        self.virtual_stage.connect()
        self.real_stage: HXPProvider | None = None
        self.stage: HexapodProvider = self.virtual_stage

        self.virtual_laser = VirtualLaserGate()
        self.virtual_laser.connect()
        self.real_laser: HXPDigitalLaserGate | None = None
        self.laser: LaserGateProvider = self.virtual_laser

        self._last_stage_snapshot = self.stage.snapshot()
        self._last_tick_monotonic = time.monotonic()
        self._poll_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hxp-poll")
        self._poll_future: Future | None = None
        self._last_real_poll = 0.0

        self.recipe = Recipe()
        self._recipe_running = False
        self._recipe_index = 0
        self._recipe_step_issued = False
        self._wait_until = 0.0
        self._recipe_real_armed = False

        self._build_ui()
        self._apply_style()
        self._load_default_config()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self.statusBar().showMessage("Virtual stage + virtual LX13 ready")

    # ---------------- UI ----------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("HEXAPOD + LASER LAB")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch(1)
        self.mode_badge = QtWidgets.QLabel("VIRTUAL LAB — NO HARDWARE COMMANDS")
        self.mode_badge.setObjectName("virtualBadge")
        self.mode_badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        header.addWidget(self.mode_badge)
        root.addLayout(header)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.addWidget(self._build_motion_panel())
        split.addWidget(self._build_viewer_panel())
        split.addWidget(self._build_state_panel())
        split.setSizes([380, 880, 360])
        root.addWidget(split, 1)

        root.addWidget(self._build_recipe_panel(), 0)

    def _section(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("section")
        return label

    def _build_motion_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        layout.addWidget(self._section("MODE + CONNECTION"))
        self.stage_mode = QtWidgets.QComboBox()
        self.stage_mode.addItems(["Virtual hexapod", "Real Newport HXP"])
        self.stage_mode.currentIndexChanged.connect(self._stage_mode_changed)
        layout.addWidget(self.stage_mode)

        form = QtWidgets.QFormLayout()
        self.hxp_host = QtWidgets.QLineEdit("192.168.33.3")
        self.hxp_port = QtWidgets.QSpinBox(); self.hxp_port.setRange(1, 65535); self.hxp_port.setValue(5001)
        self.hxp_group = QtWidgets.QLineEdit("HEXAPOD")
        self.hxp_coords = QtWidgets.QComboBox(); self.hxp_coords.addItems(["Work", "Tool"])
        form.addRow("HXP IP", self.hxp_host)
        form.addRow("Port", self.hxp_port)
        form.addRow("Group", self.hxp_group)
        form.addRow("Move frame", self.hxp_coords)
        layout.addLayout(form)

        conn_row = QtWidgets.QHBoxLayout()
        self.connect_stage_btn = QtWidgets.QPushButton("Connect")
        self.connect_stage_btn.clicked.connect(self._connect_stage)
        self.disconnect_stage_btn = QtWidgets.QPushButton("Disconnect")
        self.disconnect_stage_btn.clicked.connect(self._disconnect_stage)
        conn_row.addWidget(self.connect_stage_btn); conn_row.addWidget(self.disconnect_stage_btn)
        layout.addLayout(conn_row)

        layout.addWidget(self._section("TARGET POSE"))
        grid = QtWidgets.QGridLayout()
        self.pose_boxes: dict[str, QtWidgets.QDoubleSpinBox] = {}
        for i, axis in enumerate("XYZUVW"):
            label = QtWidgets.QLabel(axis)
            box = QtWidgets.QDoubleSpinBox()
            box.setDecimals(4)
            box.setSingleStep(0.1 if i < 3 else 0.1)
            box.setRange(-1000.0 if i < 3 else -180.0, 1000.0 if i < 3 else 180.0)
            box.setSuffix(" mm" if i < 3 else " °")
            self.pose_boxes[axis] = box
            grid.addWidget(label, i, 0)
            grid.addWidget(box, i, 1)
        layout.addLayout(grid)

        move_row = QtWidgets.QHBoxLayout()
        move_btn = QtWidgets.QPushButton("MOVE ABSOLUTE")
        move_btn.setObjectName("primary")
        move_btn.clicked.connect(self._move_absolute)
        capture_btn = QtWidgets.QPushButton("Target = actual")
        capture_btn.clicked.connect(self._target_from_actual)
        move_row.addWidget(move_btn, 2); move_row.addWidget(capture_btn, 1)
        layout.addLayout(move_row)

        layout.addWidget(self._section("JOG"))
        jog_form = QtWidgets.QFormLayout()
        self.jog_mm = QtWidgets.QDoubleSpinBox(); self.jog_mm.setRange(0.0001, 100); self.jog_mm.setDecimals(4); self.jog_mm.setValue(0.1); self.jog_mm.setSuffix(" mm")
        self.jog_deg = QtWidgets.QDoubleSpinBox(); self.jog_deg.setRange(0.0001, 30); self.jog_deg.setDecimals(4); self.jog_deg.setValue(0.1); self.jog_deg.setSuffix(" °")
        jog_form.addRow("Linear step", self.jog_mm); jog_form.addRow("Angular step", self.jog_deg)
        layout.addLayout(jog_form)

        jog_grid = QtWidgets.QGridLayout()
        for row, axis in enumerate("XYZUVW"):
            minus = QtWidgets.QPushButton(f"{axis} −")
            plus = QtWidgets.QPushButton(f"{axis} +")
            minus.clicked.connect(lambda _=False, a=axis: self._jog(a, -1.0))
            plus.clicked.connect(lambda _=False, a=axis: self._jog(a, +1.0))
            jog_grid.addWidget(minus, row, 0); jog_grid.addWidget(plus, row, 1)
        layout.addLayout(jog_grid)

        action_row = QtWidgets.QHBoxLayout()
        self.home_btn = QtWidgets.QPushButton("HOME")
        self.home_btn.clicked.connect(self._home)
        self.abort_btn = QtWidgets.QPushButton("ABORT")
        self.abort_btn.setObjectName("danger")
        self.abort_btn.clicked.connect(self._abort_all)
        action_row.addWidget(self.home_btn); action_row.addWidget(self.abort_btn)
        layout.addLayout(action_row)
        layout.addStretch(1)
        return panel

    def _build_viewer_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(self._section("DIGITAL TWIN + PATH MAP"))
        bar.addStretch(1)
        self.cad_status = QtWidgets.QLabel("CAD-derived rig")
        bar.addWidget(self.cad_status)
        load_cad = QtWidgets.QPushButton("Load Stewart Platform.STEP…")
        load_cad.clicked.connect(self._load_step_dialog)
        bar.addWidget(load_cad)
        clear = QtWidgets.QPushButton("Clear traces")
        clear.clicked.connect(self._clear_visual_traces)
        bar.addWidget(clear)
        layout.addLayout(bar)

        self.viewer = Hexapod3DViewer(panel, self.profile)
        self.viewer.set_status_callback(self._cad_message)

        map_container = QtWidgets.QWidget(panel)
        map_layout = QtWidgets.QVBoxLayout(map_container)
        map_layout.setContentsMargins(0, 0, 0, 0)
        map_layout.setSpacing(3)

        map_bar = QtWidgets.QHBoxLayout()
        map_bar.addWidget(self._section("2D MOVEMENT MAP"))
        map_bar.addStretch(1)
        self.map_mode = QtWidgets.QComboBox()
        self.map_mode.addItems(["Laser path on sample", "HXP XY carriage path"])
        self.map_span = QtWidgets.QDoubleSpinBox()
        self.map_span.setRange(2.0, 500.0)
        self.map_span.setValue(40.0)
        self.map_span.setSuffix(" mm span")
        map_bar.addWidget(self.map_mode)
        map_bar.addWidget(self.map_span)
        map_layout.addLayout(map_bar)

        self.movement_map = MovementMap2D(map_container)
        self.map_mode.currentIndexChanged.connect(
            lambda i: self.movement_map.set_mode("sample" if i == 0 else "hxp")
        )
        self.map_span.valueChanged.connect(self.movement_map.set_span_mm)
        map_layout.addWidget(self.movement_map, 1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(self.viewer)
        splitter.addWidget(map_container)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([620, 250])
        layout.addWidget(splitter, 1)
        return panel

    def _clear_visual_traces(self) -> None:
        self.viewer.clear_traces()
        self.movement_map.clear()

    def _build_state_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        layout.addWidget(self._section("LIVE STATE"))
        self.state_labels: dict[str, QtWidgets.QLabel] = {}
        grid = QtWidgets.QGridLayout()
        for i, axis in enumerate("XYZUVW"):
            grid.addWidget(QtWidgets.QLabel(axis), i, 0)
            lab = QtWidgets.QLabel("0.0000")
            lab.setObjectName("mono")
            self.state_labels[axis] = lab
            grid.addWidget(lab, i, 1)
        layout.addLayout(grid)
        self.motion_status = QtWidgets.QLabel("IDLE")
        self.motion_status.setObjectName("statusPill")
        layout.addWidget(self.motion_status)

        layout.addWidget(self._section("ACTUATOR GEOMETRY"))
        self.leg_labels: list[QtWidgets.QLabel] = []
        for i in range(6):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel(f"Strut {i + 1}"))
            val = QtWidgets.QLabel("— mm"); val.setObjectName("mono")
            row.addStretch(1); row.addWidget(val)
            self.leg_labels.append(val)
            layout.addLayout(row)

        layout.addWidget(self._section("LASER / LX13"))
        self.laser_mode = QtWidgets.QComboBox()
        self.laser_mode.addItems(["Virtual LX13", "Real HXP GPIO → PHAROS LX13"])
        self.laser_mode.currentIndexChanged.connect(self._laser_mode_changed)
        layout.addWidget(self.laser_mode)
        self.laser_status = QtWidgets.QLabel("GATE CLOSED (SIMULATED)")
        self.laser_status.setObjectName("laserOff")
        layout.addWidget(self.laser_status)

        self.laser_connect_btn = QtWidgets.QPushButton("Connect / arm laser gate")
        self.laser_connect_btn.clicked.connect(self._connect_laser)
        layout.addWidget(self.laser_connect_btn)
        gate_row = QtWidgets.QHBoxLayout()
        self.gate_on_btn = QtWidgets.QPushButton("GATE ON")
        self.gate_on_btn.clicked.connect(lambda: self._set_gate(True))
        self.gate_off_btn = QtWidgets.QPushButton("GATE OFF")
        self.gate_off_btn.clicked.connect(lambda: self._set_gate(False))
        gate_row.addWidget(self.gate_on_btn); gate_row.addWidget(self.gate_off_btn)
        layout.addLayout(gate_row)

        laser_form = QtWidgets.QFormLayout()
        self.gpio_name = QtWidgets.QLineEdit("")
        self.gpio_mask = QtWidgets.QSpinBox(); self.gpio_mask.setRange(0, 65535); self.gpio_mask.setValue(0)
        self.gpio_on = QtWidgets.QSpinBox(); self.gpio_on.setRange(0, 65535); self.gpio_on.setValue(0)
        self.gpio_off = QtWidgets.QSpinBox(); self.gpio_off.setRange(0, 65535); self.gpio_off.setValue(0)
        laser_form.addRow("HXP GPIO", self.gpio_name)
        laser_form.addRow("Mask", self.gpio_mask)
        laser_form.addRow("Gate-ON value", self.gpio_on)
        laser_form.addRow("Gate-OFF value", self.gpio_off)
        layout.addLayout(laser_form)
        self.wiring_verified = QtWidgets.QCheckBox("I have verified the PHAROS LX13 pinout, active level and HXP electrical compatibility")
        self.wiring_verified.setWordWrap(True)
        layout.addWidget(self.wiring_verified)

        layout.addWidget(self._section("SAFETY / MODE"))
        self.real_arm = QtWidgets.QCheckBox("ARM REAL RECIPE EXECUTION")
        self.real_arm.setObjectName("dangerCheck")
        layout.addWidget(self.real_arm)
        hint = QtWidgets.QLabel("The GUI is not a substitute for the laser interlock or emergency-stop chain. Real mode starts fail-closed and never assumes LX13 logic levels.")
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        layout.addWidget(hint)
        layout.addStretch(1)
        return panel

    def _build_recipe_panel(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("EXPERIMENT RECIPE — drag rows to reorder")
        layout = QtWidgets.QVBoxLayout(box)
        self.recipe_list = RecipeList()
        self.recipe_list.setMinimumHeight(155)
        layout.addWidget(self.recipe_list)

        add_row = QtWidgets.QHBoxLayout()
        for text, cb in [
            ("+ ABS target", self._recipe_add_abs),
            ("+ REL jog", self._recipe_add_rel),
            ("+ Laser ON", lambda: self._recipe_add_gate(True)),
            ("+ Laser OFF", lambda: self._recipe_add_gate(False)),
            ("+ Wait", self._recipe_add_wait),
        ]:
            b = QtWidgets.QPushButton(text); b.clicked.connect(cb); add_row.addWidget(b)
        remove = QtWidgets.QPushButton("Remove selected"); remove.clicked.connect(self._recipe_remove)
        add_row.addWidget(remove)
        layout.addLayout(add_row)

        run_row = QtWidgets.QHBoxLayout()
        self.preflight_btn = QtWidgets.QPushButton("PREFLIGHT")
        self.preflight_btn.clicked.connect(self._preflight_dialog)
        self.run_recipe_btn = QtWidgets.QPushButton("▶ RUN RECIPE")
        self.run_recipe_btn.setObjectName("primary")
        self.run_recipe_btn.clicked.connect(self._run_recipe)
        self.stop_recipe_btn = QtWidgets.QPushButton("■ STOP + GATE OFF")
        self.stop_recipe_btn.setObjectName("danger")
        self.stop_recipe_btn.clicked.connect(self._stop_recipe)
        save = QtWidgets.QPushButton("Save…"); save.clicked.connect(self._save_recipe)
        load = QtWidgets.QPushButton("Load…"); load.clicked.connect(self._load_recipe)
        run_row.addWidget(self.preflight_btn); run_row.addWidget(self.run_recipe_btn); run_row.addWidget(self.stop_recipe_btn)
        run_row.addStretch(1); run_row.addWidget(save); run_row.addWidget(load)
        layout.addLayout(run_row)
        self.recipe_progress = QtWidgets.QLabel("Ready")
        self.recipe_progress.setObjectName("muted")
        layout.addWidget(self.recipe_progress)
        return box

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #11161d; color: #e8edf2; font-size: 13px; }
            QLabel#title { font-size: 21px; font-weight: 700; letter-spacing: 1px; }
            QLabel#section { color: #9ba8b5; font-size: 11px; font-weight: 700; letter-spacing: 1px; padding-top: 4px; }
            QLabel#virtualBadge { background:#183426; color:#b9f0cb; border:1px solid #2b6342; border-radius:6px; padding:7px 12px; font-weight:700; }
            QLabel#realBadge { background:#51211f; color:#ffd1cc; border:1px solid #a74b45; border-radius:6px; padding:7px 12px; font-weight:700; }
            QLabel#mono { font-family: Consolas, 'Courier New'; font-size: 13px; }
            QLabel#muted { color:#8d98a4; }
            QLabel#statusPill { background:#26303a; border-radius:5px; padding:6px; font-weight:700; }
            QLabel#laserOff { background:#26303a; border:1px solid #47515c; padding:8px; border-radius:5px; font-weight:700; }
            QLabel#laserOn { background:#5a3518; border:1px solid #b66f2f; color:#ffe0ad; padding:8px; border-radius:5px; font-weight:700; }
            QPushButton, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox { background:#1a222c; border:1px solid #384451; border-radius:5px; padding:6px; min-height:23px; }
            QPushButton:hover { border-color:#6d7c8b; }
            QPushButton#primary { background:#234b68; border-color:#37779f; font-weight:700; }
            QPushButton#danger { background:#5d2525; border-color:#963f3f; font-weight:700; }
            QGroupBox { border:1px solid #303b46; border-radius:6px; margin-top:10px; padding-top:10px; font-weight:700; }
            QListWidget { background:#0d1218; border:1px solid #303b46; alternate-background-color:#121a22; }
            QCheckBox#dangerCheck { color:#ffaaa3; font-weight:700; }
            QSplitter::handle { background:#26303a; width:2px; }
            """
        )

    # ---------------- configuration ----------------
    def _load_default_config(self) -> None:
        if not DEFAULT_CONFIG.is_file():
            return
        try:
            cfg = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
            hxp = cfg.get("hxp", {})
            self.hxp_host.setText(str(hxp.get("host", self.hxp_host.text())))
            self.hxp_port.setValue(int(hxp.get("port", self.hxp_port.value())))
            self.hxp_group.setText(str(hxp.get("group", self.hxp_group.text())))
            coord = str(hxp.get("coordinate_system", self.hxp_coords.currentText()))
            index = self.hxp_coords.findText(coord)
            if index >= 0: self.hxp_coords.setCurrentIndex(index)
            laser = cfg.get("laser", {})
            self.gpio_name.setText(str(laser.get("gpio_name", "")))
            self.gpio_mask.setValue(int(laser.get("mask", 0)))
            self.gpio_on.setValue(int(laser.get("enabled_value", 0)))
            self.gpio_off.setValue(int(laser.get("disabled_value", 0)))
            self.wiring_verified.setChecked(bool(laser.get("wiring_verified", False)))
        except Exception as exc:
            self.statusBar().showMessage(f"Config warning: {exc}")

    # ---------------- stage ----------------
    def _stage_mode_changed(self, index: int) -> None:
        if index == 0:
            self.stage = self.virtual_stage
            if not self.stage.snapshot().connected:
                self.virtual_stage.connect()
            self.mode_badge.setText("VIRTUAL LAB — NO HARDWARE COMMANDS")
            self.mode_badge.setObjectName("virtualBadge")
        else:
            if self.real_stage is not None:
                self.stage = self.real_stage
            self.mode_badge.setText("REAL LAB SELECTED — HARDWARE COMMANDS POSSIBLE")
            self.mode_badge.setObjectName("realBadge")
        self.mode_badge.style().unpolish(self.mode_badge); self.mode_badge.style().polish(self.mode_badge)

    def _connect_stage(self) -> None:
        try:
            if self.stage_mode.currentIndex() == 0:
                self.virtual_stage.connect(); self.stage = self.virtual_stage
                self.statusBar().showMessage("Virtual hexapod connected")
                return
            cfg = HXPProviderConfig(
                host=self.hxp_host.text().strip(),
                port=self.hxp_port.value(),
                group=self.hxp_group.text().strip() or "HEXAPOD",
                coordinate_system=self.hxp_coords.currentText(),
                timeout_s=2.0,
            )
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            self.real_stage = HXPProvider(cfg)
            self.real_stage.connect()
            self.stage = self.real_stage
            self.statusBar().showMessage(f"Connected to HXP {cfg.host}:{cfg.port}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "HXP connection failed", str(exc))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _disconnect_stage(self) -> None:
        try:
            if self.stage is self.real_stage and self.real_laser is not None:
                self.real_laser.safe_off()
            self.stage.disconnect()
        except Exception as exc:
            self.statusBar().showMessage(f"Disconnect warning: {exc}")

    def _target_pose(self) -> Pose6D:
        return Pose6D(*(self.pose_boxes[a].value() for a in "XYZUVW"))

    def _move_absolute(self) -> None:
        try:
            self.stage.move_absolute(self._target_pose())
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Move failed", str(exc))

    def _target_from_actual(self) -> None:
        for axis, value in zip("XYZUVW", self._last_stage_snapshot.actual.as_tuple()):
            self.pose_boxes[axis].setValue(value)

    def _jog(self, axis: str, sign: float) -> None:
        vals = [0.0] * 6
        idx = "XYZUVW".index(axis)
        vals[idx] = sign * (self.jog_mm.value() if idx < 3 else self.jog_deg.value())
        try:
            self.stage.move_incremental(Pose6D.from_iterable(vals))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Jog failed", str(exc))

    def _home(self) -> None:
        if self.stage is not self.virtual_stage:
            ans = QtWidgets.QMessageBox.warning(
                self,
                "Home real HXP?",
                "A real HXP home search can move all six struts through a substantial path. Confirm the physical setup is clear before continuing.",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if ans != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        try:
            self.stage.home()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Home failed", str(exc))

    def _abort_all(self) -> None:
        try:
            self.laser.safe_off()
        finally:
            try:
                self.stage.abort()
            except Exception as exc:
                self.statusBar().showMessage(f"Abort warning: {exc}")
        self._recipe_running = False
        self.recipe_progress.setText("ABORTED — gate-off requested")

    # ---------------- laser ----------------
    def _laser_mode_changed(self, index: int) -> None:
        if index == 0:
            self.laser = self.virtual_laser
            if not self.virtual_laser.snapshot().connected:
                self.virtual_laser.connect()
        elif self.real_laser is not None:
            self.laser = self.real_laser

    def _connect_laser(self) -> None:
        try:
            if self.laser_mode.currentIndex() == 0:
                self.virtual_laser.connect(); self.laser = self.virtual_laser
                return
            if self.real_stage is None or not self.real_stage.client.connected:
                raise RuntimeError("Connect the real HXP first; the current LX13 implementation uses an HXP digital-output socket")
            cfg = HXPDigitalLaserConfig(
                gpio_name=self.gpio_name.text().strip(),
                mask=self.gpio_mask.value(),
                enabled_value=self.gpio_on.value(),
                disabled_value=self.gpio_off.value(),
                wiring_verified=self.wiring_verified.isChecked(),
            )
            self.real_laser = HXPDigitalLaserGate(self.real_stage.client, cfg)
            self.real_laser.connect()
            self.laser = self.real_laser
            self.statusBar().showMessage("Real LX13 gate armed through verified HXP GPIO mapping; initial state forced OFF")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Laser gate not armed", str(exc))

    def _set_gate(self, enabled: bool) -> None:
        try:
            self.laser.set_gate(enabled)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Laser gate command failed", str(exc))

    # ---------------- CAD ----------------
    def _load_step_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Stewart-platform STEP", str(APP_ROOT), "STEP files (*.step *.stp *.STEP *.STP)")
        if not path:
            return
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            self.viewer.load_step(path)
            self.cad_status.setText("Exact STEP CAD • articulated")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "CAD load failed", str(exc))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _cad_message(self, message: str) -> None:
        self.cad_status.setText(message)
        self.statusBar().showMessage(message, 8000)

    # ---------------- recipe ----------------
    def _sync_recipe_from_list(self) -> None:
        steps = []
        for i in range(self.recipe_list.count()):
            item = self.recipe_list.item(i)
            data = item.data(QtCore.Qt.ItemDataRole.UserRole)
            steps.append(RecipeStep.from_dict(data))
        self.recipe.steps = steps

    def _append_recipe_step(self, step: RecipeStep) -> None:
        item = QtWidgets.QListWidgetItem(step.describe())
        item.setData(QtCore.Qt.ItemDataRole.UserRole, step.to_dict())
        self.recipe_list.addItem(item)
        self._sync_recipe_from_list()

    def _recipe_add_abs(self) -> None:
        self._append_recipe_step(RecipeStep.move_absolute(self._target_pose()))

    def _recipe_add_rel(self) -> None:
        dlg = QtWidgets.QDialog(self); dlg.setWindowTitle("Add relative move")
        lay = QtWidgets.QFormLayout(dlg)
        boxes = []
        for i, axis in enumerate("XYZUVW"):
            b = QtWidgets.QDoubleSpinBox(); b.setDecimals(4); b.setRange(-1000 if i < 3 else -180, 1000 if i < 3 else 180)
            b.setSuffix(" mm" if i < 3 else " °"); boxes.append(b); lay.addRow(f"d{axis}", b)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept); buttons.rejected.connect(dlg.reject); lay.addRow(buttons)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._append_recipe_step(RecipeStep.move_incremental(Pose6D.from_iterable([b.value() for b in boxes])))

    def _recipe_add_gate(self, enabled: bool) -> None:
        self._append_recipe_step(RecipeStep.laser_gate(enabled))

    def _recipe_add_wait(self) -> None:
        value, ok = QtWidgets.QInputDialog.getDouble(self, "Wait step", "Seconds", 1.0, 0.0, 3600.0, 3)
        if ok:
            self._append_recipe_step(RecipeStep.wait(value))

    def _recipe_remove(self) -> None:
        row = self.recipe_list.currentRow()
        if row >= 0:
            self.recipe_list.takeItem(row)
            self._sync_recipe_from_list()

    def _preflight(self) -> list:
        self._sync_recipe_from_list()
        return preflight_recipe(self.recipe)

    def _preflight_dialog(self) -> None:
        issues = self._preflight()
        if not issues:
            text = "Preflight passed. Recipe ends gate-OFF and all step payloads are valid."
            icon = QtWidgets.QMessageBox.Icon.Information
        else:
            text = "\n".join(f"{x.severity.upper()}: " + (f"step {x.step_index + 1}: " if x.step_index is not None else "") + x.message for x in issues)
            icon = QtWidgets.QMessageBox.Icon.Warning if not any(x.severity == "error" for x in issues) else QtWidgets.QMessageBox.Icon.Critical
        msg = QtWidgets.QMessageBox(icon, "Recipe preflight", text, parent=self); msg.exec()

    def _run_recipe(self) -> None:
        issues = self._preflight()
        errors = [x for x in issues if x.severity == "error"]
        if errors:
            self._preflight_dialog(); return
        real_stage = self.stage is self.real_stage
        real_laser = self.laser is self.real_laser and self.real_laser is not None
        if (real_stage or real_laser) and not self.real_arm.isChecked():
            QtWidgets.QMessageBox.warning(self, "Real run not armed", "Check ARM REAL RECIPE EXECUTION only after reviewing the preflight and physical setup.")
            return
        self._recipe_running = True
        self._recipe_index = 0
        self._recipe_step_issued = False
        self._wait_until = 0.0
        self.viewer.clear_traces()
        self.recipe_progress.setText("Recipe started")

    def _stop_recipe(self) -> None:
        self._recipe_running = False
        self.laser.safe_off()
        try:
            if self._last_stage_snapshot.state == MotionState.MOVING:
                self.stage.abort()
        except Exception:
            pass
        self.recipe_progress.setText("Stopped — gate-off requested")

    def _advance_recipe(self) -> None:
        self._recipe_index += 1
        self._recipe_step_issued = False
        self._wait_until = 0.0
        if self._recipe_index >= len(self.recipe.steps):
            self._recipe_running = False
            self.laser.safe_off()
            self.recipe_progress.setText("Recipe complete — gate OFF")

    def _recipe_tick(self) -> None:
        if not self._recipe_running:
            return
        if self._recipe_index >= len(self.recipe.steps):
            self._advance_recipe(); return
        step = self.recipe.steps[self._recipe_index]
        self.recipe_progress.setText(f"Step {self._recipe_index + 1}/{len(self.recipe.steps)} — {step.describe()}")
        try:
            if step.kind == StepKind.MOVE_ABSOLUTE:
                if not self._recipe_step_issued:
                    self.stage.move_absolute(Pose6D.from_iterable(step.payload["pose"])); self._recipe_step_issued = True
                elif self._last_stage_snapshot.state != MotionState.MOVING:
                    self._advance_recipe()
            elif step.kind == StepKind.MOVE_INCREMENTAL:
                if not self._recipe_step_issued:
                    self.stage.move_incremental(Pose6D.from_iterable(step.payload["delta"])); self._recipe_step_issued = True
                elif self._last_stage_snapshot.state != MotionState.MOVING:
                    self._advance_recipe()
            elif step.kind == StepKind.LASER_GATE:
                if not self._recipe_step_issued:
                    self.laser.set_gate(bool(step.payload.get("enabled"))); self._recipe_step_issued = True
                    self._advance_recipe()
            elif step.kind == StepKind.WAIT:
                if not self._recipe_step_issued:
                    self._wait_until = time.monotonic() + float(step.payload.get("seconds", 0.0)); self._recipe_step_issued = True
                elif time.monotonic() >= self._wait_until:
                    self._advance_recipe()
        except Exception as exc:
            self._recipe_running = False
            self.laser.safe_off()
            self.recipe_progress.setText(f"FAILED at step {self._recipe_index + 1}: {exc}")
            QtWidgets.QMessageBox.critical(self, "Recipe failed", str(exc))

    def _save_recipe(self) -> None:
        self._sync_recipe_from_list()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save recipe", "recipe.json", "JSON (*.json)")
        if path:
            self.recipe.save(path)

    def _load_recipe(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load recipe", "", "JSON (*.json)")
        if not path: return
        try:
            self.recipe = Recipe.load(path)
            self.recipe_list.clear()
            for step in self.recipe.steps:
                item = QtWidgets.QListWidgetItem(step.describe()); item.setData(QtCore.Qt.ItemDataRole.UserRole, step.to_dict()); self.recipe_list.addItem(item)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Recipe load failed", str(exc))

    # ---------------- update loop ----------------
    def _poll_stage(self, now: float, dt: float) -> None:
        if self.stage is self.virtual_stage:
            self.virtual_stage.tick(dt)
            self._last_stage_snapshot = self.virtual_stage.snapshot()
            return

        if not isinstance(self.stage, HXPProvider):
            return
        if self._poll_future is not None and self._poll_future.done():
            try:
                self._last_stage_snapshot = self._poll_future.result()
            except Exception as exc:
                self.statusBar().showMessage(f"HXP polling error: {exc}")
            self._poll_future = None
        if self._poll_future is None and now - self._last_real_poll >= 0.15:
            self._last_real_poll = now
            self._poll_future = self._poll_pool.submit(self.stage.snapshot)

    def _update_readouts(self) -> None:
        snap = self._last_stage_snapshot
        for axis, value in zip("XYZUVW", snap.actual.as_tuple()):
            unit = " mm" if axis in "XYZ" else " °"
            self.state_labels[axis].setText(f"{value:+.4f}{unit}")
        self.motion_status.setText(f"{snap.state.value.upper()}  •  {snap.status_text or snap.provider}")
        lengths = self.kinematics.leg_lengths_mm(snap.actual)
        home = self.kinematics.leg_lengths_mm(Pose6D())
        for lab, length, reference in zip(self.leg_labels, lengths, home):
            lab.setText(f"{length:8.3f} mm   Δ {length-reference:+.3f}")

        laser = self.laser.snapshot()
        if laser.gate_enabled:
            self.laser_status.setText(f"GATE ON  •  {laser.connector_name}" + ("" if laser.readback_known else "  •  command state only"))
            self.laser_status.setObjectName("laserOn")
        else:
            self.laser_status.setText(f"GATE CLOSED  •  {laser.connector_name}" + ("" if laser.readback_known else "  •  command state only"))
            self.laser_status.setObjectName("laserOff")
        self.laser_status.style().unpolish(self.laser_status); self.laser_status.style().polish(self.laser_status)
        self.viewer.update_state(snap.actual, laser.gate_enabled)

    def _tick(self) -> None:
        now = time.monotonic()
        dt = min(0.25, max(0.0, now - self._last_tick_monotonic))
        self._last_tick_monotonic = now
        self._poll_stage(now, dt)
        self._recipe_tick()
        self._update_readouts()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            self.laser.safe_off()
            if self.real_stage is not None:
                self.real_stage.disconnect()
            self.virtual_stage.disconnect()
            self.virtual_laser.disconnect()
        finally:
            self._poll_pool.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)
