"""Shared SLM quick cards and complete detailed editors for the unified GUI."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from slm_lab_control.app import SLMControlPanel
from slm_lab_control.config import SlmPhaseConfig
from slm_lab_control.presets import load_preset, save_preset

from ..controller import LabController
from ..state import ExperimentState
from .controls import NoWheelSpinBox, blocked_signals


def phase_pixmap(array: np.ndarray, width: int = 360, height: int = 190) -> QPixmap:
    values = np.ascontiguousarray(array, dtype=np.uint8)
    rows, columns = values.shape
    image = QImage(values.data, columns, rows, columns, QImage.Format_Grayscale8).copy()
    return QPixmap.fromImage(image).scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def active_layers(phase: SlmPhaseConfig) -> list[str]:
    layers = [name for name, enabled in phase.switches.__dict__.items() if enabled and name != "circular_pupil"]
    return layers


class SlmQuickCard(QFrame):
    """Compact Home card editing the same ExperimentState as the detail page."""

    details_requested = Signal(str)
    preset_requested = Signal()
    operation_message = Signal(str)

    def __init__(self, name: str, controller: LabController, parent: QWidget | None = None):
        super().__init__(parent)
        self.name = name.upper()
        self.controller = controller
        self.setObjectName("Panel")
        self._syncing = False
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(7)
        heading = QHBoxLayout()
        title = QLabel(self.name)
        title.setObjectName("Section")
        self.connection = QLabel("DISCONNECTED")
        self.connection.setObjectName("BadChip")
        heading.addWidget(title)
        heading.addStretch(1)
        heading.addWidget(self.connection)
        root.addLayout(heading)

        self.preview = QLabel("Generate phase to preview")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(145)
        self.preview.setObjectName("ImageWell")
        root.addWidget(self.preview, 1)

        controls = QHBoxLayout()
        self.vortex = QCheckBox("Vortex")
        self.charge = NoWheelSpinBox()
        self.charge.setRange(-250, 250)
        self.charge.setKeyboardTracking(False)
        self.correction = QCheckBox("Retrieved correction")
        controls.addWidget(self.vortex)
        controls.addWidget(QLabel("ℓ"))
        controls.addWidget(self.charge)
        controls.addWidget(self.correction)
        controls.addStretch(1)
        root.addLayout(controls)

        self.layers = QLabel("Active: none")
        self.layers.setObjectName("Muted")
        self.layers.setWordWrap(True)
        self.cast_state = QLabel("Configured phase has not been cast")
        self.cast_state.setObjectName("Muted")
        self.cast_state.setWordWrap(True)
        root.addWidget(self.layers)
        root.addWidget(self.cast_state)

        actions = QHBoxLayout()
        for label, handler, object_name in (
            ("Connect", self._connect, "Quiet"),
            ("Cast", self._cast, "Accent"),
            ("Blank", self._blank, "Danger"),
            ("Preset", self.preset_requested.emit, "Quiet"),
            ("Details", lambda: self.details_requested.emit(self.name), "Quiet"),
        ):
            button = QPushButton(label)
            button.setObjectName(object_name)
            button.clicked.connect(handler)
            actions.addWidget(button)
        root.addLayout(actions)

        self.vortex.toggled.connect(self._apply_quick_state)
        self.charge.editingFinished.connect(self._apply_quick_state)
        self.correction.toggled.connect(self._apply_quick_state)

    def _apply_quick_state(self) -> None:
        if self._syncing:
            return

        def apply(state: ExperimentState) -> None:
            phase = getattr(state, self.name.lower()).phase
            phase.switches.vortex = self.vortex.isChecked()
            phase.vortex_charge = self.charge.value()
            phase.switches.retrieved_correction = self.correction.isChecked()

        self.controller.store.update(
            apply,
            source="advanced_home",
            reason=f"Changed {self.name} Home quick controls",
        )
        try:
            self.controller.generate()
        except Exception as exc:
            self.operation_message.emit(f"{self.name} generation failed: {exc}")

    def refresh(self, state: ExperimentState) -> None:
        slm = getattr(state, self.name.lower())
        self._syncing = True
        try:
            with blocked_signals((self.vortex, self.charge, self.correction)):
                self.vortex.setChecked(slm.phase.switches.vortex)
                self.charge.setValue(slm.phase.vortex_charge)
                self.correction.setChecked(slm.phase.switches.retrieved_correction)
        finally:
            self._syncing = False
        self.connection.setText(slm.connection.value)
        self.connection.setObjectName("GoodChip" if slm.connection.value == "CONNECTED" else "BadChip")
        self.connection.style().unpolish(self.connection)
        self.connection.style().polish(self.connection)
        layers = active_layers(slm.phase)
        self.layers.setText("Active: " + (", ".join(layers) if layers else "none"))
        matches = bool(slm.complete_phase_sha256 and slm.complete_phase_sha256 == slm.last_cast_sha256)
        self.cast_state.setText(
            "CAST MATCHES CONFIGURED PHASE" if matches else "CONFIGURED PHASE DIFFERS FROM LAST CAST"
        )
        self.cast_state.setObjectName("StatusGood" if matches else "StatusWarn")
        self.cast_state.style().unpolish(self.cast_state)
        self.cast_state.style().polish(self.cast_state)
        bundle = self.controller.last_bundle
        if bundle is not None and bundle.hashes.get(self.name) == slm.complete_phase_sha256:
            self.preview.setPixmap(phase_pixmap(bundle.results[self.name].gray_uint8))

    def _connect(self) -> None:
        try:
            self.operation_message.emit("\n".join(self.controller.connect_slms()))
        except Exception as exc:
            self.operation_message.emit(f"SLM connection failed: {exc}")

    def _cast(self) -> None:
        try:
            receipt = self.controller.cast((self.name,))
            self.operation_message.emit("\n".join(receipt.messages))
        except Exception as exc:
            self.operation_message.emit(f"{self.name} cast failed: {exc}")

    def _blank(self) -> None:
        try:
            self.operation_message.emit("\n".join(self.controller.blank((self.name,))))
        except Exception as exc:
            self.operation_message.emit(f"{self.name} blank failed: {exc}")


class SlmDetailView(QWidget):
    """Full legacy-parity SLM editor embedded in the unified application."""

    operation_message = Signal(str)

    def __init__(self, name: str, controller: LabController, project_root: Path, parent: QWidget | None = None):
        super().__init__(parent)
        self.name = name.upper()
        self.controller = controller
        self.project_root = Path(project_root)
        self._syncing = False
        state = controller.store.snapshot()
        self.panel = SLMControlPanel(copy.deepcopy(getattr(state, self.name.lower()).phase))
        self._build()
        self.panel.changed.connect(self._apply_panel_state)

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        self.status = QLabel()
        self.status.setObjectName("Muted")
        self.status.setWordWrap(True)
        header.addWidget(self.status, 1)
        for label, handler, object_name in (
            ("Connect SLMs", self._connect, "Quiet"),
            ("Generate", self._generate, "Quiet"),
            (f"Cast {self.name}", self._cast, "Accent"),
            (f"Blank {self.name}", self._blank, "Danger"),
            ("Load preset…", self._load_preset, "Quiet"),
            ("Save preset…", self._save_preset, "Quiet"),
            ("Reset phase", self._reset_phase, "Danger"),
        ):
            button = QPushButton(label)
            button.setObjectName(object_name)
            button.clicked.connect(handler)
            header.addWidget(button)
        root.addLayout(header)
        body = QHBoxLayout()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.panel)
        body.addWidget(scroll, 3)
        preview_box = QFrame()
        preview_box.setObjectName("Panel")
        preview_layout = QVBoxLayout(preview_box)
        title = QLabel("Authoritative complete phase")
        title.setObjectName("Section")
        self.preview = QLabel("Generate phase to preview")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(360, 260)
        self.preview.setObjectName("ImageWell")
        self.hashes = QLabel()
        self.hashes.setObjectName("Muted")
        self.hashes.setWordWrap(True)
        preview_layout.addWidget(title)
        preview_layout.addWidget(self.preview, 1)
        preview_layout.addWidget(self.hashes)
        body.addWidget(preview_box, 2)
        root.addLayout(body, 1)

    def _apply_panel_state(self) -> None:
        if self._syncing:
            return
        phase = self.panel.to_config(self.name)
        self.controller.store.update(
            lambda state: setattr(getattr(state, self.name.lower()), "phase", copy.deepcopy(phase)),
            source="advanced_slm_detail",
            reason=f"Edited complete {self.name} phase configuration",
        )
        self._generate()

    def refresh(self, state: ExperimentState) -> None:
        slm = getattr(state, self.name.lower())
        self._syncing = True
        try:
            self.panel.set_config(copy.deepcopy(slm.phase))
        finally:
            self._syncing = False
        matches = bool(slm.complete_phase_sha256 and slm.complete_phase_sha256 == slm.last_cast_sha256)
        self.status.setText(
            f"{slm.phase.serial} • {slm.connection.value} • {slm.phase.geometry.wavelength_nm:g} nm • "
            + ("cast matches configured phase" if matches else "configured phase differs from last cast")
        )
        self.hashes.setText(
            f"Configured: {slm.complete_phase_sha256 or 'not generated'}\n"
            f"Last cast: {slm.last_cast_sha256 or 'never cast'}\n"
            f"Last command: {slm.last_hardware_command or 'none'}"
        )
        bundle = self.controller.last_bundle
        if bundle is not None and bundle.hashes.get(self.name) == slm.complete_phase_sha256:
            self.preview.setPixmap(phase_pixmap(bundle.results[self.name].gray_uint8, 520, 420))

    def _generate(self) -> None:
        try:
            self.controller.generate()
        except Exception as exc:
            self.operation_message.emit(f"{self.name} generation failed: {exc}")

    def _connect(self) -> None:
        try:
            self.operation_message.emit("\n".join(self.controller.connect_slms()))
        except Exception as exc:
            self.operation_message.emit(f"SLM connection failed: {exc}")

    def _cast(self) -> None:
        try:
            self.operation_message.emit("\n".join(self.controller.cast((self.name,)).messages))
        except Exception as exc:
            self.operation_message.emit(f"{self.name} cast failed: {exc}")

    def _blank(self) -> None:
        try:
            self.operation_message.emit("\n".join(self.controller.blank((self.name,))))
        except Exception as exc:
            self.operation_message.emit(f"{self.name} blank failed: {exc}")

    def _load_preset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load dual-SLM preset", str(self.project_root), "JSON preset (*.json)")
        if not path:
            return
        try:
            self.controller.replace_app_config(load_preset(Path(path)), source="advanced_slm_detail", reason=f"Loaded preset {Path(path).name}")
            self.controller.generate()
        except Exception as exc:
            QMessageBox.warning(self, "Preset load failed", str(exc))

    def _save_preset(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save dual-SLM preset", str(self.project_root / "presets"), "JSON preset (*.json)")
        if not path:
            return
        destination = Path(path)
        if destination.suffix.lower() != ".json":
            destination = destination.with_suffix(".json")
        save_preset(destination, self.controller.app_config())
        self.operation_message.emit(f"Saved preset: {destination}")

    def _reset_phase(self) -> None:
        answer = QMessageBox.question(
            self,
            f"Reset {self.name}",
            "Reset this configured phase to the locked hardware defaults? This does not cast automatically.",
        )
        if answer != QMessageBox.Yes:
            return
        fresh = SlmPhaseConfig(name=self.name)
        fresh.apply_locked_hardware(preserve_alignment_centre=False)
        self.controller.store.update(
            lambda state: setattr(getattr(state, self.name.lower()), "phase", fresh),
            source="advanced_slm_detail",
            reason=f"Reset configured {self.name} phase",
        )
        self.controller.generate()
