from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .config import AppConfig, SlmPhaseConfig, TermSwitches
from .hardware import BackendError, make_backend
from .hardware_profiles import profile_for
from .logging_utils import create_cast_folder, save_cast_bundle, timestamp
from .phase import PhaseResult, compose_phase
from .presets import load_preset, save_preset
from .ui.style import APP_QSS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# Input controls
# -----------------------------------------------------------------------------


class ManualDoubleSpinBox(QDoubleSpinBox):
    """Numeric field that changes only when the user types a value.

    Mouse wheel, arrow buttons and keyboard stepping are deliberately disabled so
    scrolling the SLM editor cannot silently change an experimental parameter.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.setKeyboardTracking(False)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):  # noqa: N802 - Qt API
        event.ignore()

    def stepBy(self, steps: int) -> None:  # noqa: N802 - Qt API
        return


class ManualSpinBox(QSpinBox):
    """Integer equivalent of :class:`ManualDoubleSpinBox`."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.setKeyboardTracking(False)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):  # noqa: N802 - Qt API
        event.ignore()

    def stepBy(self, steps: int) -> None:  # noqa: N802 - Qt API
        return


class NoWheelComboBox(QComboBox):
    """Combo box that does not accidentally cycle while the editor is scrolled."""

    def wheelEvent(self, event):  # noqa: N802 - Qt API
        event.ignore()


def _spin(
    minimum: float,
    maximum: float,
    value: float,
    step: float,
    decimals: int = 3,
) -> ManualDoubleSpinBox:
    w = ManualDoubleSpinBox()
    w.setRange(minimum, maximum)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setValue(value)
    return w


def _ispin(minimum: int, maximum: int, value: int, step: int = 1) -> ManualSpinBox:
    w = ManualSpinBox()
    w.setRange(minimum, maximum)
    w.setSingleStep(step)
    w.setValue(value)
    return w


def _card(object_name: str = "Card") -> QFrame:
    f = QFrame()
    f.setObjectName(object_name)
    return f


def _section_label(text: str, description: str = "") -> QWidget:
    box = QWidget()
    layout = QVBoxLayout(box)
    layout.setContentsMargins(2, 10, 2, 2)
    layout.setSpacing(2)
    title = QLabel(text)
    title.setObjectName("SectionTitle")
    layout.addWidget(title)
    if description:
        sub = QLabel(description)
        sub.setObjectName("SectionSubtitle")
        sub.setWordWrap(True)
        layout.addWidget(sub)
    return box


def _arr_to_pixmap(arr: np.ndarray, max_width: int = 700) -> QPixmap:
    if arr.dtype != np.uint8:
        arr = np.asarray(np.clip(arr, 0, 255), dtype=np.uint8)
    h, w = arr.shape
    qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()
    pix = QPixmap.fromImage(qimg)
    if w > max_width:
        return pix.scaledToWidth(max_width, Qt.SmoothTransformation)
    return pix


def _enable_children(widget: QWidget, enabled: bool) -> None:
    """Enable/disable child controls without disabling the group checkbox itself."""
    for child in widget.findChildren(QWidget):
        if child is widget:
            continue
        child.setEnabled(enabled)


class ComponentGroup(QGroupBox):
    """Checkable phase-term card; checked means the term enters the final phase."""

    def __init__(self, title: str, checked: bool = False, parent: QWidget | None = None):
        super().__init__(title, parent)
        self.setObjectName("ComponentGroup")
        self.setCheckable(True)
        self.setChecked(checked)
        self.toggled.connect(lambda state: _enable_children(self, state))

    def finish(self) -> None:
        _enable_children(self, self.isChecked())


# -----------------------------------------------------------------------------
# Per-SLM editor
# -----------------------------------------------------------------------------


class SLMControlPanel(QWidget):
    changed = Signal()

    def __init__(self, config: SlmPhaseConfig, parent: QWidget | None = None):
        super().__init__(parent)
        self._config = config
        self._build()
        self.set_config(config)
        self._connect_changed_signals()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 18)
        root.setSpacing(10)

        # Locked hardware summary. These values deliberately have no input fields.
        locked = _card("LockedCard")
        locked_layout = QVBoxLayout(locked)
        locked_layout.setContentsMargins(16, 12, 16, 12)
        title = QLabel("Locked lab hardware")
        title.setObjectName("CardTitle")
        self.hardware_summary = QLabel()
        self.hardware_summary.setObjectName("Muted")
        self.hardware_summary.setWordWrap(True)
        self.carrier_summary = QLabel()
        self.carrier_summary.setObjectName("LockedValue")
        self.carrier_summary.setWordWrap(True)
        note = QLabel(
            "Panel geometry, serial, wavelength, bit depth and carrier convention are fixed by "
            "slm_lab_control/hardware_profiles.py and cannot be changed by scrolling, presets or this editor."
        )
        note.setObjectName("TinyMuted")
        note.setWordWrap(True)
        locked_layout.addWidget(title)
        locked_layout.addWidget(self.hardware_summary)
        locked_layout.addWidget(self.carrier_summary)
        locked_layout.addWidget(note)
        root.addWidget(locked)

        root.addWidget(
            _section_label(
                "Beam placement",
                "These coordinates define the origin used by vortex, focus, digital axicon and correction terms.",
            )
        )
        beam = QGroupBox("Beam centre / computational origin")
        beam.setObjectName("ComponentGroup")
        form = QFormLayout(beam)
        self.center_x = _spin(-8192.0, 8192.0, 960.0, 1.0, 2)
        self.center_y = _spin(-8192.0, 8192.0, 540.0, 1.0, 2)
        self.rotation_deg = _spin(-180.0, 180.0, 0.0, 0.1, 3)
        form.addRow("Centre x / px", self.center_x)
        form.addRow("Centre y / px", self.center_y)
        form.addRow("Phase-term rotation / deg", self.rotation_deg)
        root.addWidget(beam)

        self.grp_pupil = ComponentGroup("Circular pupil gate — BLANKS outside pupil", False)
        form = QFormLayout(self.grp_pupil)
        self.pupil_mm = _spin(0.1, 50.0, 9.0, 0.1, 3)
        form.addRow("Pupil diameter / mm", self.pupil_mm)
        pupil_note = QLabel("Leave OFF for normal additive phase composition. When ON, everything outside this pupil is deliberately replaced by the background.")
        pupil_note.setObjectName("TinyMuted")
        pupil_note.setWordWrap(True)
        form.addRow("", pupil_note)
        root.addWidget(self.grp_pupil)

        root.addWidget(
            _section_label(
                "Core beam shaping",
                "Use only the terms you actually want encoded on this SLM. The locked carrier is added automatically.",
            )
        )
        self.grp_vortex = ComponentGroup("Vortex / OAM phase", False)
        form = QFormLayout(self.grp_vortex)
        self.vortex_charge = _ispin(-250, 250, 0)
        form.addRow("Topological charge ℓ", self.vortex_charge)
        root.addWidget(self.grp_vortex)

        self.grp_axicon = ComponentGroup("Digital axicon / Bessel phase", False)
        form = QFormLayout(self.grp_axicon)
        self.axicon_period = _spin(2.0, 5000.0, 80.0, 1.0, 3)
        self.axicon_sign = _spin(-1.0, 1.0, 1.0, 1.0, 0)
        form.addRow("Radial phase period / px", self.axicon_period)
        form.addRow("Sign", self.axicon_sign)
        ax_note = QLabel("Leave this OFF when you only want the physical axicon to provide the conical phase.")
        ax_note.setObjectName("TinyMuted")
        ax_note.setWordWrap(True)
        form.addRow("", ax_note)
        root.addWidget(self.grp_axicon)

        self.grp_focus = ComponentGroup("Lens / focus phase", False)
        form = QFormLayout(self.grp_focus)
        self.focus_mm = _spin(-1_000_000.0, 1_000_000.0, 0.0, 10.0, 3)
        self.focus_sign = _spin(-1.0, 1.0, 1.0, 1.0, 0)
        form.addRow("Focal length / mm", self.focus_mm)
        form.addRow("Sign", self.focus_sign)
        root.addWidget(self.grp_focus)

        root.addWidget(
            _section_label(
                "Wavefront & aberration correction",
                "Correction maps and low-order terms are explicit here so they can be tested independently in the lab.",
            )
        )
        self.grp_wavefront = ComponentGroup("Measured wavefront compensation map", False)
        form = QFormLayout(self.grp_wavefront)
        self.wavefront_path = QLineEdit()
        self.wavefront_path.setPlaceholderText("Select the calibrated grayscale compensation map…")
        browse = QPushButton("Browse…")
        browse.setObjectName("Quiet")
        browse.clicked.connect(lambda: self._browse_phase_file("wavefront"))
        wave_row = QWidget()
        wave_layout = QHBoxLayout(wave_row)
        wave_layout.setContentsMargins(0, 0, 0, 0)
        wave_layout.addWidget(self.wavefront_path, 1)
        wave_layout.addWidget(browse)
        self.wavefront_gain = _spin(-5.0, 5.0, 1.0, 0.05, 3)
        form.addRow("Compensation image", wave_row)
        form.addRow("Gain", self.wavefront_gain)
        root.addWidget(self.grp_wavefront)

        self.grp_zernike = ComponentGroup("Low-order Zernike correction", False)
        form = QFormLayout(self.grp_zernike)
        self.z20_amp = _spin(-20.0, 20.0, 0.0, 0.01, 4)
        self.z22_cos_amp = _spin(-20.0, 20.0, 0.0, 0.01, 4)
        self.z22_sin_amp = _spin(-20.0, 20.0, 0.0, 0.01, 4)
        self.z31_cos_amp = _spin(-20.0, 20.0, 0.0, 0.01, 4)
        self.z31_sin_amp = _spin(-20.0, 20.0, 0.0, 0.01, 4)
        self.z40_amp = _spin(-20.0, 20.0, 0.0, 0.01, 4)
        form.addRow("Defocus Z₂⁰ / waves", self.z20_amp)
        form.addRow("Astigmatism Z₂² cos / waves", self.z22_cos_amp)
        form.addRow("Astigmatism Z₂² sin / waves", self.z22_sin_amp)
        form.addRow("Coma Z₃¹ cos / waves", self.z31_cos_amp)
        form.addRow("Coma Z₃¹ sin / waves", self.z31_sin_amp)
        form.addRow("Spherical Z₄⁰ / waves", self.z40_amp)
        z_note = QLabel(
            "Useful for controlled correction sweeps. Start at zero; do not treat these as calibrated corrections until measured."
        )
        z_note.setObjectName("TinyMuted")
        z_note.setWordWrap(True)
        form.addRow("", z_note)
        root.addWidget(self.grp_zernike)

        self.grp_spherical = ComponentGroup("Sample-interface spherical correction", False)
        form = QFormLayout(self.grp_spherical)
        self.interface_NA = _spin(0.0, 1.4, 0.4, 0.01, 4)
        self.interface_depth = _spin(-100000.0, 100000.0, 0.0, 1.0, 3)
        self.interface_n1 = _spin(0.5, 3.0, 1.0, 0.01, 4)
        self.interface_n2 = _spin(0.5, 3.0, 1.45, 0.01, 4)
        self.interface_sign = _spin(-1.0, 1.0, -1.0, 1.0, 0)
        form.addRow("Interface NA", self.interface_NA)
        form.addRow("Depth / µm", self.interface_depth)
        form.addRow("n₁", self.interface_n1)
        form.addRow("n₂", self.interface_n2)
        form.addRow("Correction sign", self.interface_sign)
        sph_note = QLabel(
            "ADDITIVE term only: outside its computational pupil the contribution is zero, so the existing wavefront map and 20 px blaze remain present."
        )
        sph_note.setObjectName("TinyMuted")
        sph_note.setWordWrap(True)
        form.addRow("", sph_note)
        root.addWidget(self.grp_spherical)

        self.grp_retrieved = ComponentGroup("Retrieved / inverse correction map", False)
        form = QFormLayout(self.grp_retrieved)
        self.retrieved_correction_path = QLineEdit()
        self.retrieved_correction_path.setPlaceholderText("Prefer a native 1920×1080 .npy phase map in radians…")
        browse_retrieved = QPushButton("Browse…")
        browse_retrieved.setObjectName("Quiet")
        browse_retrieved.clicked.connect(lambda: self._browse_phase_file("retrieved"))
        retrieved_row = QWidget()
        retrieved_layout = QHBoxLayout(retrieved_row)
        retrieved_layout.setContentsMargins(0, 0, 0, 0)
        retrieved_layout.addWidget(self.retrieved_correction_path, 1)
        retrieved_layout.addWidget(browse_retrieved)
        self.retrieved_correction_gain = _spin(-2.0, 2.0, 0.05, 0.01, 4)
        form.addRow("Correction phase file", retrieved_row)
        form.addRow("Gain", self.retrieved_correction_gain)
        retrieved_note = QLabel(
            "Added in radians to wavefront + locked 20 px blaze + any other enabled terms, then wrapped ONCE at the end. "
            "Use negative gain for a sign test. Current q=20 maps remain uncalibrated until branch/mapping are resolved."
        )
        retrieved_note.setObjectName("TinyMuted")
        retrieved_note.setWordWrap(True)
        form.addRow("", retrieved_note)
        root.addWidget(self.grp_retrieved)

        root.addWidget(
            _section_label(
                "Experimental overlays",
                "Keep unusual or one-off terms here so the normal beam-shaping controls stay clean.",
            )
        )
        self.grp_custom = ComponentGroup("Custom phase overlay", False)
        form = QFormLayout(self.grp_custom)
        self.custom_phase_path = QLineEdit()
        self.custom_phase_path.setPlaceholderText("Load an arbitrary 8-bit phase image…")
        browse_custom = QPushButton("Browse…")
        browse_custom.setObjectName("Quiet")
        browse_custom.clicked.connect(lambda: self._browse_phase_file("custom"))
        custom_row = QWidget()
        custom_layout = QHBoxLayout(custom_row)
        custom_layout.setContentsMargins(0, 0, 0, 0)
        custom_layout.addWidget(self.custom_phase_path, 1)
        custom_layout.addWidget(browse_custom)
        self.custom_phase_gain = _spin(-10.0, 10.0, 1.0, 0.05, 3)
        form.addRow("Phase image", custom_row)
        form.addRow("Gain", self.custom_phase_gain)
        root.addWidget(self.grp_custom)

        self.grp_nfold = ComponentGroup("N-fold / polygonal perturbation", False)
        form = QFormLayout(self.grp_nfold)
        self.nfold_order = _ispin(1, 64, 6)
        self.nfold_amp = _spin(-20.0, 20.0, 0.0, 0.05, 4)
        form.addRow("Order", self.nfold_order)
        form.addRow("Amplitude / waves", self.nfold_amp)
        root.addWidget(self.grp_nfold)

        base = QGroupBox("Output scaling")
        base.setObjectName("ComponentGroup")
        form = QFormLayout(base)
        self.background_gray = _spin(0.0, 255.0, 0.0, 1.0, 2)
        self.global_gain = _spin(-10.0, 10.0, 1.0, 0.05, 4)
        form.addRow("Background gray", self.background_gray)
        form.addRow("Global phase gain", self.global_gain)
        root.addWidget(base)

        for group in self._term_groups():
            group.finish()
        root.addStretch(1)

    def _term_groups(self) -> List[ComponentGroup]:
        return [
            self.grp_pupil,
            self.grp_vortex,
            self.grp_axicon,
            self.grp_focus,
            self.grp_wavefront,
            self.grp_zernike,
            self.grp_spherical,
            self.grp_retrieved,
            self.grp_custom,
            self.grp_nfold,
        ]

    def _all_widgets(self) -> Iterable[QWidget]:
        return [
            self.center_x,
            self.center_y,
            self.rotation_deg,
            self.grp_pupil,
            self.pupil_mm,
            self.grp_vortex,
            self.vortex_charge,
            self.grp_axicon,
            self.axicon_period,
            self.axicon_sign,
            self.grp_focus,
            self.focus_mm,
            self.focus_sign,
            self.grp_wavefront,
            self.wavefront_path,
            self.wavefront_gain,
            self.grp_zernike,
            self.z20_amp,
            self.z22_cos_amp,
            self.z22_sin_amp,
            self.z31_cos_amp,
            self.z31_sin_amp,
            self.z40_amp,
            self.grp_spherical,
            self.interface_NA,
            self.interface_depth,
            self.interface_n1,
            self.interface_n2,
            self.interface_sign,
            self.grp_retrieved,
            self.retrieved_correction_path,
            self.retrieved_correction_gain,
            self.grp_custom,
            self.custom_phase_path,
            self.custom_phase_gain,
            self.grp_nfold,
            self.nfold_order,
            self.nfold_amp,
            self.background_gray,
            self.global_gain,
        ]

    def _connect_changed_signals(self) -> None:
        for w in self._all_widgets():
            if isinstance(w, QLineEdit):
                w.editingFinished.connect(self.changed.emit)
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                w.valueChanged.connect(self.changed.emit)
            elif isinstance(w, QComboBox):
                w.currentTextChanged.connect(self.changed.emit)
            elif isinstance(w, QCheckBox):
                w.stateChanged.connect(self.changed.emit)
            elif isinstance(w, QGroupBox) and w.isCheckable():
                w.toggled.connect(self.changed.emit)

    def _browse_phase_file(self, kind: str) -> None:
        if kind == "wavefront":
            title = "Select wavefront compensation image"
            filter_text = "Phase images (*.png *.bmp *.jpg *.jpeg *.tif *.tiff);;All files (*.*)"
        elif kind == "retrieved":
            title = "Select retrieved / inverse correction phase map"
            filter_text = "Phase maps (*.npy *.png *.bmp *.tif *.tiff);;All files (*.*)"
        else:
            title = "Select custom phase image"
            filter_text = "Phase images (*.png *.bmp *.jpg *.jpeg *.tif *.tiff);;All files (*.*)"
        path, _ = QFileDialog.getOpenFileName(self, title, str(PROJECT_ROOT), filter_text)
        if not path:
            return
        if kind == "wavefront":
            self.wavefront_path.setText(path)
            self.grp_wavefront.setChecked(True)
        elif kind == "retrieved":
            self.retrieved_correction_path.setText(path)
            self.grp_retrieved.setChecked(True)
        else:
            self.custom_phase_path.setText(path)
            self.grp_custom.setChecked(True)
        self.changed.emit()

    def set_config(self, config: SlmPhaseConfig) -> None:
        config.apply_locked_hardware()
        self._config = config
        profile = profile_for(config.name)
        self.hardware_summary.setText(profile.short_summary)
        self.carrier_summary.setText(profile.carrier_summary)

        for w in self._all_widgets():
            w.blockSignals(True)
        try:
            sw = config.switches
            self.grp_pupil.setChecked(sw.circular_pupil)
            self.grp_vortex.setChecked(sw.vortex)
            self.grp_axicon.setChecked(sw.axicon)
            self.grp_focus.setChecked(sw.focus)
            self.grp_wavefront.setChecked(sw.wavefront)
            self.grp_zernike.setChecked(sw.zernike_z40)
            self.grp_spherical.setChecked(sw.spherical_interface)
            self.grp_retrieved.setChecked(getattr(sw, "retrieved_correction", False))
            self.grp_custom.setChecked(getattr(sw, "custom_phase", False))
            self.grp_nfold.setChecked(sw.n_fold)

            self.center_x.setValue(config.center_x_px)
            self.center_y.setValue(config.center_y_px)
            self.rotation_deg.setValue(config.term_rotation_deg)
            self.pupil_mm.setValue(config.pupil_diameter_mm)
            self.vortex_charge.setValue(config.vortex_charge)
            self.axicon_period.setValue(config.axicon_period_px)
            self.axicon_sign.setValue(config.axicon_sign)
            self.focus_mm.setValue(config.focus_focal_length_mm)
            self.focus_sign.setValue(config.focus_sign)
            self.wavefront_path.setText(config.wavefront_path)
            self.wavefront_gain.setValue(config.wavefront_gain)
            self.z20_amp.setValue(getattr(config, "z20_amp_waves", 0.0))
            self.z22_cos_amp.setValue(getattr(config, "z22_cos_amp_waves", 0.0))
            self.z22_sin_amp.setValue(getattr(config, "z22_sin_amp_waves", 0.0))
            self.z31_cos_amp.setValue(getattr(config, "z31_cos_amp_waves", 0.0))
            self.z31_sin_amp.setValue(getattr(config, "z31_sin_amp_waves", 0.0))
            self.z40_amp.setValue(config.z40_amp_waves)
            self.interface_NA.setValue(config.interface_NA)
            self.interface_depth.setValue(config.interface_depth_um)
            self.interface_n1.setValue(config.interface_n1)
            self.interface_n2.setValue(config.interface_n2)
            self.interface_sign.setValue(config.interface_sign)
            self.retrieved_correction_path.setText(getattr(config, "retrieved_correction_path", ""))
            self.retrieved_correction_gain.setValue(getattr(config, "retrieved_correction_gain", 0.05))
            self.custom_phase_path.setText(getattr(config, "custom_phase_path", ""))
            self.custom_phase_gain.setValue(getattr(config, "custom_phase_gain", 1.0))
            self.nfold_order.setValue(config.n_fold_order)
            self.nfold_amp.setValue(config.n_fold_amp_waves)
            self.background_gray.setValue(config.background_gray)
            self.global_gain.setValue(config.global_phase_gain)
        finally:
            for w in self._all_widgets():
                w.blockSignals(False)
            for group in self._term_groups():
                group.finish()

    def to_config(self, name: str) -> SlmPhaseConfig:
        # Begin from the existing config so unknown/new fields survive round trips.
        cfg = self._config
        cfg.name = name
        cfg.switches = TermSwitches(
            wavefront=self.grp_wavefront.isChecked(),
            blaze=True,  # overwritten from locked hardware profile below
            focus=self.grp_focus.isChecked(),
            axicon=self.grp_axicon.isChecked(),
            vortex=self.grp_vortex.isChecked(),
            spherical_interface=self.grp_spherical.isChecked(),
            zernike_z40=self.grp_zernike.isChecked(),
            n_fold=self.grp_nfold.isChecked(),
            circular_pupil=self.grp_pupil.isChecked(),
            retrieved_correction=self.grp_retrieved.isChecked(),
            custom_phase=self.grp_custom.isChecked(),
        )

        cfg.center_x_px = self.center_x.value()
        cfg.center_y_px = self.center_y.value()
        cfg.term_rotation_deg = self.rotation_deg.value()
        cfg.pupil_diameter_mm = self.pupil_mm.value()
        cfg.vortex_charge = int(self.vortex_charge.value())
        cfg.axicon_period_px = self.axicon_period.value()
        cfg.axicon_sign = self.axicon_sign.value()
        cfg.focus_focal_length_mm = self.focus_mm.value()
        cfg.focus_sign = self.focus_sign.value()
        cfg.wavefront_path = self.wavefront_path.text().strip()
        cfg.wavefront_gain = self.wavefront_gain.value()
        cfg.z20_amp_waves = self.z20_amp.value()
        cfg.z22_cos_amp_waves = self.z22_cos_amp.value()
        cfg.z22_sin_amp_waves = self.z22_sin_amp.value()
        cfg.z31_cos_amp_waves = self.z31_cos_amp.value()
        cfg.z31_sin_amp_waves = self.z31_sin_amp.value()
        cfg.z40_amp_waves = self.z40_amp.value()
        cfg.interface_NA = self.interface_NA.value()
        cfg.interface_depth_um = self.interface_depth.value()
        cfg.interface_n1 = self.interface_n1.value()
        cfg.interface_n2 = self.interface_n2.value()
        cfg.interface_sign = self.interface_sign.value()
        cfg.retrieved_correction_path = self.retrieved_correction_path.text().strip()
        cfg.retrieved_correction_gain = self.retrieved_correction_gain.value()
        cfg.custom_phase_path = self.custom_phase_path.text().strip()
        cfg.custom_phase_gain = self.custom_phase_gain.value()
        cfg.n_fold_order = int(self.nfold_order.value())
        cfg.n_fold_amp_waves = self.nfold_amp.value()
        cfg.background_gray = self.background_gray.value()
        cfg.global_phase_gain = self.global_gain.value()
        cfg.apply_locked_hardware()
        self._config = cfg
        return cfg

    def active_term_names(self) -> List[str]:
        labels: List[str] = ["carrier"]
        if self.grp_wavefront.isChecked():
            labels.append("wavefront")
        if self.grp_vortex.isChecked():
            labels.append(f"vortex ℓ={self.vortex_charge.value()}")
        if self.grp_axicon.isChecked():
            labels.append("digital axicon")
        if self.grp_focus.isChecked():
            labels.append("focus")
        if self.grp_zernike.isChecked():
            labels.append("Zernike correction")
        if self.grp_spherical.isChecked():
            labels.append("interface correction")
        if self.grp_retrieved.isChecked():
            labels.append(f"retrieved correction ×{self.retrieved_correction_gain.value():g}")
        if self.grp_pupil.isChecked():
            labels.append("pupil gate")
        if self.grp_custom.isChecked():
            labels.append("custom overlay")
        if self.grp_nfold.isChecked():
            labels.append(f"{self.nfold_order.value()}-fold perturbation")
        return labels


# -----------------------------------------------------------------------------
# Preview / overview widgets
# -----------------------------------------------------------------------------


class PreviewCard(QWidget):
    def __init__(self, title: str, compact: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.compact = compact
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        heading = QLabel(title)
        heading.setObjectName("CardTitle")
        self.image = QLabel("No phase generated")
        self.image.setObjectName("PreviewLabel")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumSize(260 if compact else 420, 165 if compact else 250)
        self.image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stats = QLabel("")
        self.stats.setObjectName("TinyMuted")
        self.stats.setWordWrap(True)
        layout.addWidget(heading)
        layout.addWidget(self.image, 1)
        layout.addWidget(self.stats)

    def set_result(self, result: PhaseResult) -> None:
        width = 430 if self.compact else 720
        pix = _arr_to_pixmap(result.gray_uint8, max_width=width)
        self.image.setPixmap(
            pix.scaled(self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        components = ", ".join(result.components.keys())
        self.stats.setText(
            f"{len(result.components)} components  •  gray μ={result.stats['gray_mean']:.1f}, "
            f"σ={result.stats['gray_std']:.1f}\n{components}"
        )

    def resizeEvent(self, event):  # noqa: N802 - Qt API
        super().resizeEvent(event)


class OverviewSlmCard(QFrame):
    edit_requested = Signal(str)
    cast_requested = Signal(str)

    def __init__(self, name: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.name = name
        self.setObjectName("Card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(9)

        top = QHBoxLayout()
        title = QLabel(name)
        title.setObjectName("PanelTitle")
        self.locked = QLabel()
        self.locked.setObjectName("TinyMuted")
        self.locked.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top.addWidget(title)
        top.addWidget(self.locked, 1)
        layout.addLayout(top)

        self.preview = PreviewCard(f"{name} phase preview", compact=True)
        layout.addWidget(self.preview, 1)
        self.terms = QLabel("Active: carrier")
        self.terms.setObjectName("ActiveTerms")
        self.terms.setWordWrap(True)
        layout.addWidget(self.terms)

        actions = QHBoxLayout()
        edit = QPushButton(f"Edit {name}")
        edit.setObjectName("Quiet")
        cast = QPushButton(f"Cast {name}")
        cast.setObjectName("Primary")
        edit.clicked.connect(lambda: self.edit_requested.emit(self.name))
        cast.clicked.connect(lambda: self.cast_requested.emit(self.name))
        actions.addWidget(edit)
        actions.addWidget(cast)
        layout.addLayout(actions)

    def set_state(self, config: SlmPhaseConfig, result: PhaseResult | None, terms: List[str]) -> None:
        profile = profile_for(config.name)
        self.locked.setText(profile.serial)
        self.terms.setText("Active: " + "  •  ".join(terms))
        if result is not None:
            self.preview.set_result(result)


class SlmEditorPage(QWidget):
    def __init__(self, name: str, panel: SLMControlPanel, preview: PreviewCard, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel(f"{name} phase editor")
        title.setObjectName("PageTitle")
        profile = profile_for(name)
        sub = QLabel(f"{profile.short_summary}  •  {profile.carrier_summary}")
        sub.setObjectName("Muted")
        header_text = QVBoxLayout()
        header_text.addWidget(title)
        header_text.addWidget(sub)
        header.addLayout(header_text, 1)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(panel)
        splitter.addWidget(scroll)

        right = _card()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(14, 14, 14, 14)
        right_l.addWidget(preview, 1)
        tip = QLabel(
            "Tip: the phase preview is the exact wrapped 8-bit mask that will be logged/cast. "
            "Experimental controls are on the left; locked panel/carrier values are not editable here."
        )
        tip.setObjectName("TinyMuted")
        tip.setWordWrap(True)
        right_l.addWidget(tip)
        splitter.addWidget(right)
        splitter.setSizes([620, 800])
        layout.addWidget(splitter, 1)


# -----------------------------------------------------------------------------
# Main window
# -----------------------------------------------------------------------------


class MainWindow(QMainWindow):
    PAGE_HOME = 0
    PAGE_SLM1 = 1
    PAGE_SLM2 = 2
    PAGE_PRESETS = 3

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dual-SLM Lab Control Cockpit v0.6 — measurement sessions")
        self.resize(1760, 1040)
        self.app_config = self._default_config()
        self.backend = None
        self.results: Dict[str, PhaseResult] = {}
        self._build()
        self._sync_panels_from_config()
        self.refresh_preset_library()
        self.regenerate()

    def _default_config(self) -> AppConfig:
        cfg = AppConfig()
        cfg.transfer_mode = "png_file"
        cfg.slm1.switches.wavefront = False
        cfg.slm1.switches.vortex = False
        cfg.slm1.switches.axicon = False
        cfg.slm1.center_x_px = 960.0
        cfg.slm1.center_y_px = 540.0

        cfg.slm2.switches.wavefront = False
        cfg.slm2.switches.vortex = False
        cfg.slm2.switches.axicon = False
        cfg.slm2.center_x_px = 960.0
        cfg.slm2.center_y_px = 540.0
        cfg.apply_locked_hardware()
        return cfg

    def _build(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Persistent navigation keeps SLM switching one click away.
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(190)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(14, 18, 14, 18)
        side.setSpacing(8)
        brand = QLabel("SLM\nCOCKPIT")
        brand.setObjectName("Brand")
        side.addWidget(brand)
        brand_sub = QLabel("dual-panel lab control")
        brand_sub.setObjectName("TinyMuted")
        side.addWidget(brand_sub)
        side.addSpacing(18)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_buttons: List[QPushButton] = []
        nav_items = [
            ("Overview", self.PAGE_HOME),
            ("SLM1", self.PAGE_SLM1),
            ("SLM2", self.PAGE_SLM2),
            ("Presets", self.PAGE_PRESETS),
        ]
        for text, index in nav_items:
            btn = QPushButton(text)
            btn.setObjectName("NavButton")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked=False, i=index: self.set_page(i))
            self.nav_group.addButton(btn)
            self.nav_buttons.append(btn)
            side.addWidget(btn)
        self.nav_buttons[0].setChecked(True)
        side.addStretch(1)

        locked_note = QLabel("Hardware geometry + carrier are locked in code")
        locked_note.setObjectName("TinyMuted")
        locked_note.setWordWrap(True)
        side.addWidget(locked_note)
        outer.addWidget(sidebar)

        shell = QWidget()
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(18, 14, 18, 16)
        shell_layout.setSpacing(12)

        # Global header/actions appear on every page.
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("Dual-SLM Lab Control Cockpit")
        title.setObjectName("Title")
        subtitle = QLabel("Experiment controls up front; panel-specific constants locked away")
        subtitle.setObjectName("Subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box, 1)

        self.status = QLabel("Dummy mode ready")
        self.status.setObjectName("StatusGood")
        header.addWidget(self.status)
        self.btn_lab = QPushButton("Lab measurements")
        self.btn_lab.clicked.connect(self.open_lab_workbench)
        header.addWidget(self.btn_lab)
        self.btn_regen = QPushButton("Regenerate")
        self.btn_regen.setObjectName("Quiet")
        self.btn_cast_both = QPushButton("Cast both")
        self.btn_cast_both.setObjectName("Primary")
        self.btn_blank = QPushButton("Emergency blank")
        self.btn_blank.setObjectName("Danger")
        header.addWidget(self.btn_regen)
        header.addWidget(self.btn_cast_both)
        header.addWidget(self.btn_blank)
        shell_layout.addLayout(header)

        self.pages = QStackedWidget()
        shell_layout.addWidget(self.pages, 1)
        outer.addWidget(shell, 1)

        # Per-SLM panels are reused by overview and editor pages.
        self.slm1_panel = SLMControlPanel(self.app_config.slm1)
        self.slm2_panel = SLMControlPanel(self.app_config.slm2)
        self.editor_preview1 = PreviewCard("SLM1 phase mask")
        self.editor_preview2 = PreviewCard("SLM2 phase mask")

        self._build_home_page()
        self.pages.addWidget(SlmEditorPage("SLM1", self.slm1_panel, self.editor_preview1))
        self.pages.addWidget(SlmEditorPage("SLM2", self.slm2_panel, self.editor_preview2))
        self._build_presets_page()

        self.btn_regen.clicked.connect(self.regenerate)
        self.btn_cast_both.clicked.connect(lambda: self.cast(["SLM1", "SLM2"]))
        self.btn_blank.clicked.connect(self.blank_both)
        self.slm1_panel.changed.connect(self._maybe_regenerate)
        self.slm2_panel.changed.connect(self._maybe_regenerate)

    def open_lab_workbench(self) -> None:
        from .lab_workbench import LabWorkbench
        dialog = LabWorkbench(self)
        dialog.exec()

    def _build_home_page(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        top = QGridLayout()
        top.setHorizontalSpacing(12)
        top.setVerticalSpacing(12)

        connection = _card()
        c = QVBoxLayout(connection)
        c.setContentsMargins(16, 14, 16, 14)
        heading = QLabel("Lab connection")
        heading.setObjectName("CardTitle")
        c.addWidget(heading)
        form = QFormLayout()
        self.backend_kind = NoWheelComboBox()
        self.backend_kind.addItems(["dummy", "heds"])
        self.transfer_mode = NoWheelComboBox()
        self.transfer_mode.addItems(["png_file", "phase_file", "direct_gray_array", "direct_phase_array", "auto"])
        self.auto_regen = QCheckBox("Auto-regenerate after manual edits")
        self.auto_regen.setChecked(True)
        form.addRow("Backend", self.backend_kind)
        form.addRow("Transfer", self.transfer_mode)
        form.addRow("", self.auto_regen)
        c.addLayout(form)
        buttons = QHBoxLayout()
        self.btn_connect = QPushButton("Connect both")
        self.btn_connect.setObjectName("Primary")
        self.btn_close = QPushButton("Close SLM windows")
        self.btn_close.setObjectName("Quiet")
        buttons.addWidget(self.btn_connect)
        buttons.addWidget(self.btn_close)
        c.addLayout(buttons)
        sdk = QLabel("HEDS SDK target: 4.2  •  change only in config.py if the lab SDK changes")
        sdk.setObjectName("TinyMuted")
        sdk.setWordWrap(True)
        c.addWidget(sdk)
        top.addWidget(connection, 0, 0)

        preset_quick = _card()
        q = QVBoxLayout(preset_quick)
        q.setContentsMargins(16, 14, 16, 14)
        h = QLabel("Preset quick-load")
        h.setObjectName("CardTitle")
        q.addWidget(h)
        self.home_preset_combo = NoWheelComboBox()
        self.home_preset_combo.setPlaceholderText("Choose a preset…")
        q.addWidget(self.home_preset_combo)
        row = QHBoxLayout()
        self.btn_home_load_preset = QPushButton("Load selected")
        self.btn_home_load_preset.setObjectName("Primary")
        self.btn_home_presets = QPushButton("Preset library")
        self.btn_home_presets.setObjectName("Quiet")
        row.addWidget(self.btn_home_load_preset)
        row.addWidget(self.btn_home_presets)
        q.addLayout(row)
        info = QLabel("Preset loading cannot alter locked SLM geometry or carrier values.")
        info.setObjectName("TinyMuted")
        info.setWordWrap(True)
        q.addWidget(info)
        top.addWidget(preset_quick, 0, 1)
        top.setColumnStretch(0, 1)
        top.setColumnStretch(1, 1)
        layout.addLayout(top)

        cards = QHBoxLayout()
        cards.setSpacing(12)
        self.overview1 = OverviewSlmCard("SLM1")
        self.overview2 = OverviewSlmCard("SLM2")
        self.overview1.edit_requested.connect(lambda _name: self.set_page(self.PAGE_SLM1))
        self.overview2.edit_requested.connect(lambda _name: self.set_page(self.PAGE_SLM2))
        self.overview1.cast_requested.connect(lambda name: self.cast([name]))
        self.overview2.cast_requested.connect(lambda name: self.cast([name]))
        cards.addWidget(self.overview1, 1)
        cards.addWidget(self.overview2, 1)
        layout.addLayout(cards, 1)

        diagnostics = _card()
        diag_layout = QVBoxLayout(diagnostics)
        diag_layout.setContentsMargins(14, 12, 14, 12)
        diag_title = QLabel("Warnings / activity log")
        diag_title.setObjectName("CardTitle")
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(125)
        diag_layout.addWidget(diag_title)
        diag_layout.addWidget(self.log, 1)
        layout.addWidget(diagnostics)
        self.pages.addWidget(page)

        self.btn_connect.clicked.connect(self.connect_backend)
        self.btn_close.clicked.connect(self.close_backend)
        self.backend_kind.currentTextChanged.connect(self._update_config_from_controls)
        self.transfer_mode.currentTextChanged.connect(self._update_config_from_controls)
        self.auto_regen.stateChanged.connect(self._update_config_from_controls)
        self.btn_home_presets.clicked.connect(lambda: self.set_page(self.PAGE_PRESETS))
        self.btn_home_load_preset.clicked.connect(self.load_home_selected_preset)

    def _build_presets_page(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        title = QLabel("Preset library")
        title.setObjectName("PageTitle")
        sub = QLabel(
            "JSON presets live in the project presets folder. Double-click one to load it. "
            "Locked hardware constants are re-applied after every load."
        )
        sub.setObjectName("Muted")
        sub.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(sub)

        body = QSplitter(Qt.Horizontal)
        left = _card()
        l = QVBoxLayout(left)
        l.setContentsMargins(14, 14, 14, 14)
        self.preset_list = QListWidget()
        self.preset_list.setObjectName("PresetList")
        l.addWidget(self.preset_list, 1)
        actions = QGridLayout()
        self.btn_preset_load = QPushButton("Load selected")
        self.btn_preset_load.setObjectName("Primary")
        self.btn_preset_save = QPushButton("Save current as…")
        self.btn_preset_refresh = QPushButton("Refresh")
        self.btn_preset_open = QPushButton("Open folder")
        self.btn_preset_open.setObjectName("Quiet")
        actions.addWidget(self.btn_preset_load, 0, 0)
        actions.addWidget(self.btn_preset_save, 0, 1)
        actions.addWidget(self.btn_preset_refresh, 1, 0)
        actions.addWidget(self.btn_preset_open, 1, 1)
        l.addLayout(actions)
        body.addWidget(left)

        right = _card()
        r = QVBoxLayout(right)
        r.setContentsMargins(16, 14, 16, 14)
        hw = QLabel("Locked system profile")
        hw.setObjectName("CardTitle")
        r.addWidget(hw)
        for name in ("SLM1", "SLM2"):
            profile = profile_for(name)
            label = QLabel(
                f"{name}\n{profile.short_summary}\n{profile.carrier_summary}"
            )
            label.setObjectName("LockedProfile")
            label.setWordWrap(True)
            r.addWidget(label)
        note = QLabel(
            "Why this is locked: presets should describe an experiment, not redefine the panel resolution, "
            "pixel pitch, serial number or carrier. If the physical lab configuration changes, update "
            "hardware_profiles.py once and every preset follows it."
        )
        note.setObjectName("Muted")
        note.setWordWrap(True)
        r.addWidget(note)
        r.addStretch(1)
        body.addWidget(right)
        body.setSizes([850, 550])
        layout.addWidget(body, 1)
        self.pages.addWidget(page)

        self.preset_list.itemDoubleClicked.connect(lambda _item: self.load_selected_preset())
        self.btn_preset_load.clicked.connect(self.load_selected_preset)
        self.btn_preset_save.clicked.connect(self.save_named_preset)
        self.btn_preset_refresh.clicked.connect(self.refresh_preset_library)
        self.btn_preset_open.clicked.connect(self.open_preset_folder)

    # ------------------------------------------------------------------
    # Navigation / config synchronisation
    # ------------------------------------------------------------------

    def set_page(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        if 0 <= index < len(self.nav_buttons):
            self.nav_buttons[index].setChecked(True)
        if index == self.PAGE_PRESETS:
            self.refresh_preset_library()

    def _append_log(self, message: str) -> None:
        self.log.append(f"[{timestamp()}] {message}")

    def _set_status(self, message: str, warning: bool = False) -> None:
        self.status.setText(message)
        self.status.setObjectName("StatusWarn" if warning else "StatusGood")
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)

    def _sync_panels_from_config(self) -> None:
        self.app_config.apply_locked_hardware()
        self.backend_kind.blockSignals(True)
        self.transfer_mode.blockSignals(True)
        self.auto_regen.blockSignals(True)
        try:
            self.backend_kind.setCurrentText(self.app_config.backend)
            self.transfer_mode.setCurrentText(getattr(self.app_config, "transfer_mode", "png_file"))
            self.auto_regen.setChecked(self.app_config.auto_regenerate)
        finally:
            self.backend_kind.blockSignals(False)
            self.transfer_mode.blockSignals(False)
            self.auto_regen.blockSignals(False)
        self.slm1_panel.set_config(self.app_config.slm1)
        self.slm2_panel.set_config(self.app_config.slm2)

    def _update_config_from_controls(self) -> None:
        self.app_config.backend = self.backend_kind.currentText()
        self.app_config.transfer_mode = self.transfer_mode.currentText()
        self.app_config.auto_regenerate = self.auto_regen.isChecked()
        self.app_config.slm1 = self.slm1_panel.to_config("SLM1")
        self.app_config.slm2 = self.slm2_panel.to_config("SLM2")
        self.app_config.apply_locked_hardware()

    def _maybe_regenerate(self) -> None:
        self._update_config_from_controls()
        if self.auto_regen.isChecked():
            self.regenerate()

    # ------------------------------------------------------------------
    # Phase generation / casting
    # ------------------------------------------------------------------

    def regenerate(self) -> None:
        try:
            self._update_config_from_controls()
            r1 = compose_phase(self.app_config.slm1)
            r2 = compose_phase(self.app_config.slm2)
            self.results = {"SLM1": r1, "SLM2": r2}
            self.editor_preview1.set_result(r1)
            self.editor_preview2.set_result(r2)
            self.overview1.set_state(
                self.app_config.slm1, r1, self.slm1_panel.active_term_names()
            )
            self.overview2.set_state(
                self.app_config.slm2, r2, self.slm2_panel.active_term_names()
            )
            warnings = r1.warnings + r2.warnings
            if warnings:
                self._set_status(f"Generated with {len(warnings)} warning(s)", warning=True)
                self._append_log("Warnings:\n" + "\n".join(f"  - {w}" for w in warnings))
            else:
                self._set_status("Phase masks generated")
            return True
        except Exception as exc:
            self._set_status("Generation failed", warning=True)
            self._append_log(f"Generation error: {exc}")
            QMessageBox.critical(self, "Generation failed", str(exc))
            self.results = {}
            return False

    def output_root(self) -> Path:
        return self.app_config.resolve_output_root(PROJECT_ROOT)

    def connect_backend(self) -> None:
        try:
            self._update_config_from_controls()
            self.backend = make_backend(
                self.app_config.backend,
                self.output_root(),
                sdk_major=self.app_config.sdk_major,
                sdk_minor=self.app_config.sdk_minor,
            )
            msg = self.backend.init_sdk()
            self._append_log(msg)
            self._append_log(self.backend.connect("SLM1", self.app_config.slm1.serial, self.app_config.slm1.geometry.wavelength_nm))
            self._append_log(self.backend.connect("SLM2", self.app_config.slm2.serial, self.app_config.slm2.geometry.wavelength_nm))
            self._set_status(f"Connected via {self.app_config.backend}")
        except Exception as exc:
            self._set_status("Connection failed", warning=True)
            self._append_log(f"Connection error: {exc}")
            QMessageBox.critical(self, "Connection failed", str(exc))

    def _ensure_backend(self):
        if self.backend is None:
            self.connect_backend()
        if self.backend is None:
            raise BackendError("Backend was not created.")
        return self.backend

    def cast(self, names: Iterable[str]) -> None:
        try:
            if not self.regenerate():
                raise BackendError("Phase generation failed; no stale phase will be cast.")
            backend = self._ensure_backend()
            folder = create_cast_folder(self.output_root(), "slm_cast")
            selected = {name: self.results[name] for name in names}
            written = save_cast_bundle(folder, self.app_config, selected)
            transfer = self.app_config.transfer_mode
            for name in names:
                if transfer == "png_file":
                    # Legacy image-data path retained unchanged for backwards compatibility.
                    msg = backend.show_png(name, Path(written[name]))
                elif transfer == "phase_file":
                    # HEDS interprets 8-bit file values as phase over a 2*pi unit.
                    msg = backend.show_phase_file(name, Path(written[name]))
                elif transfer == "direct_gray_array":
                    msg = backend.show_gray_array(
                        name,
                        self.results[name].gray_uint8,
                        Path(written[name]),
                        allow_png_fallback=False,
                    )
                elif transfer == "direct_phase_array":
                    msg = backend.show_phase_array(
                        name,
                        self.results[name].phase_rad,
                        self.results[name].gray_uint8,
                        Path(written[name]),
                        allow_png_fallback=False,
                    )
                else:
                    msg = backend.show_phase_array(
                        name,
                        self.results[name].phase_rad,
                        self.results[name].gray_uint8,
                        Path(written[name]),
                        allow_png_fallback=True,
                    )
                self._append_log(msg)
            self._append_log(f"Metadata: {written['metadata']}")
            self._set_status(f"Cast complete ({transfer})")
        except Exception as exc:
            self._set_status("Cast failed", warning=True)
            self._append_log(f"Cast error: {exc}")
            QMessageBox.critical(self, "Cast failed", str(exc))

    def blank_both(self) -> None:
        try:
            backend = self._ensure_backend()
            folder = create_cast_folder(self.output_root(), "blank")
            s1 = self.app_config.slm1.geometry
            s2 = self.app_config.slm2.geometry
            transfer = self.app_config.transfer_mode
            allow_fallback = transfer not in ("direct_gray_array", "direct_phase_array")
            self._append_log(
                backend.blank(
                    "SLM1",
                    folder / "slm1_blank.png",
                    shape=(s1.height_px, s1.width_px),
                    gray=0,
                    use_direct=(transfer != "png_file"),
                    allow_png_fallback=allow_fallback,
                )
            )
            self._append_log(
                backend.blank(
                    "SLM2",
                    folder / "slm2_blank.png",
                    shape=(s2.height_px, s2.width_px),
                    gray=0,
                    use_direct=(transfer != "png_file"),
                    allow_png_fallback=allow_fallback,
                )
            )
            self._set_status("Both SLMs blanked")
        except Exception as exc:
            self._set_status("Blank failed", warning=True)
            self._append_log(f"Blank error: {exc}")
            QMessageBox.critical(self, "Blank failed", str(exc))

    def close_backend(self) -> None:
        try:
            if self.backend is not None:
                self._append_log(self.backend.close())
                self.backend = None
            self._set_status("Backend closed")
        except Exception as exc:
            self._set_status("Close failed", warning=True)
            self._append_log(f"Close error: {exc}")

    # ------------------------------------------------------------------
    # Preset library
    # ------------------------------------------------------------------

    def preset_root(self) -> Path:
        p = self.app_config.resolve_preset_root(PROJECT_ROOT)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def preset_files(self) -> List[Path]:
        return sorted(self.preset_root().rglob("*.json"), key=lambda p: str(p).lower())

    def refresh_preset_library(self) -> None:
        if not hasattr(self, "preset_list"):
            return
        current = self.home_preset_combo.currentData() if hasattr(self, "home_preset_combo") else None
        files = self.preset_files()
        self.preset_list.clear()
        self.home_preset_combo.blockSignals(True)
        self.home_preset_combo.clear()
        self.home_preset_combo.addItem("Choose a preset…", None)
        for path in files:
            rel = path.relative_to(self.preset_root())
            item = QListWidgetItem(str(rel))
            item.setData(Qt.UserRole, str(path))
            self.preset_list.addItem(item)
            self.home_preset_combo.addItem(str(rel), str(path))
        if current:
            idx = self.home_preset_combo.findData(current)
            if idx >= 0:
                self.home_preset_combo.setCurrentIndex(idx)
        self.home_preset_combo.blockSignals(False)

    def _load_preset_path(self, path: Path) -> None:
        try:
            cfg = load_preset(path)
            cfg.apply_locked_hardware()
            self.app_config = cfg
            self._sync_panels_from_config()
            self.regenerate()
            self._append_log(f"Preset loaded: {path}")
            self._set_status(f"Loaded preset: {path.stem}")
        except Exception as exc:
            QMessageBox.critical(self, "Load preset failed", str(exc))

    def load_selected_preset(self) -> None:
        item = self.preset_list.currentItem()
        if item is None:
            QMessageBox.information(self, "Preset library", "Select a preset first.")
            return
        path = Path(item.data(Qt.UserRole))
        self._load_preset_path(path)

    def load_home_selected_preset(self) -> None:
        path = self.home_preset_combo.currentData()
        if not path:
            QMessageBox.information(self, "Preset quick-load", "Choose a preset first.")
            return
        self._load_preset_path(Path(path))

    def save_named_preset(self) -> None:
        self._update_config_from_controls()
        name, ok = QInputDialog.getText(
            self,
            "Save preset",
            "Preset name:",
            QLineEdit.Normal,
            f"experiment_{timestamp()}",
        )
        if not ok or not name.strip():
            return
        safe = "".join(c if (c.isalnum() or c in "-_ ") else "_" for c in name.strip()).strip()
        safe = safe.replace(" ", "_")
        path = self.preset_root() / f"{safe}.json"
        if path.exists():
            answer = QMessageBox.question(
                self,
                "Overwrite preset?",
                f"{path.name} already exists. Overwrite it?",
            )
            if answer != QMessageBox.Yes:
                return
        save_preset(path, self.app_config)
        self._append_log(f"Preset saved: {path}")
        self.refresh_preset_library()
        self._set_status(f"Saved preset: {path.stem}")

    def open_preset_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.preset_root().resolve())))


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_QSS)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
