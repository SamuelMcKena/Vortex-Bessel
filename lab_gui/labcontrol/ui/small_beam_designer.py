"""Small, read-only design workspace using the repository's original designer."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy.special import jv
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QLabel,
                              QPushButton, QPlainTextEdit, QScrollArea, QWidget,
                              QFileDialog, QHBoxLayout, QMessageBox)
from .controls import NoWheelDoubleSpinBox, NoWheelSpinBox, NoWheelComboBox
from ..small_beam_design import design_small_beam, ideal_radial_reference
from ..sample_plane import bench_ring_radius_um, demagnification_for_ring


class SmallBeamDesigner(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Small-beam designer — ideal sample-plane reference, not the live bench")
        self.resize(1000, 800)
        root = QVBoxLayout(self)
        area = QScrollArea()
        area.setWidgetResizable(True)
        body = QWidget()
        box = QVBoxLayout(body)
        note = QLabel("Choose a size and length to find the required cone and objective mapping. "
                      "This does not alter the bench, cast SLM masks, or simulate correction. "
                      "For q=0, size is the first-dark-ring diameter (not FWHM); for q≠0, actual size is the bright-ring diameter.")
        note.setWordWrap(True)
        box.addWidget(note)
        form = QFormLayout()
        self.diameter = self._number(0.1, 10000, 3, " µm")
        self.length = self._number(1, 100000, 150, " µm")
        self.charge = NoWheelSpinBox()
        self.charge.setRange(-100, 100)
        self.definition = NoWheelComboBox()
        self.definition.addItems(["Actual spot / vortex-ring diameter", "Equivalent q=0 first-null diameter (legacy target)"])
        self.wavelength = self._number(200, 20000, 1030, " nm")
        self.radius = self._number(0.05, 20, 2, " mm")
        self.index = self._number(1, 4, 1, "")
        for label, widget in [("Target diameter", self.diameter), ("Diameter convention", self.definition),
                              ("Reference Bessel length — not FWHM", self.length), ("Charge q", self.charge),
                              ("Wavelength", self.wavelength), ("Input SLM beam radius", self.radius),
                              ("Sample refractive index (air = 1)", self.index)]:
            form.addRow(label, widget)
        box.addLayout(form)
        actions = QHBoxLayout()
        calculate = QPushButton("Calculate ideal design")
        calculate.clicked.connect(self.calculate)
        legacy = QPushButton("Load old repository target")
        legacy.clicked.connect(self.load_legacy)
        self.export = QPushButton("Export figure + design…")
        self.export.clicked.connect(self.export_design)
        self.use_on_bench = QPushButton("Set this objective on the virtual bench")
        self.use_on_bench.setToolTip(
            "Puts the required demagnification into Settings -> Objective. The post-axicon pattern scales "
            "exactly, so the simulator then reports sample-plane microns for the same propagation."
        )
        self.use_on_bench.clicked.connect(self.apply_to_bench)
        self.use_on_bench.setEnabled(False)
        for button in (calculate, legacy, self.export, self.use_on_bench):
            actions.addWidget(button)
        box.addLayout(actions)
        self.bench_note = QLabel()
        self.bench_note.setWordWrap(True)
        box.addWidget(self.bench_note)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMinimumHeight(185)
        box.addWidget(self.report)
        self.figure = Figure(figsize=(9, 3.8), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(350)
        box.addWidget(self.canvas)
        area.setWidget(body)
        root.addWidget(area)
        self.result = None
        # Edits must never leave an old figure looking like the new design.
        for widget in (self.diameter, self.length, self.charge, self.wavelength, self.radius, self.index):
            widget.valueChanged.connect(self.invalidate)
        self.definition.currentIndexChanged.connect(self.invalidate)
        self.calculate()

    @staticmethod
    def _number(low, high, value, suffix):
        box = NoWheelDoubleSpinBox()
        box.setRange(low, high)
        box.setDecimals(3)
        box.setValue(value)
        box.setSuffix(suffix)
        return box

    def invalidate(self, *_args):
        self.result = None
        self.bench_demagnification = None
        self.export.setEnabled(False)
        if hasattr(self, "use_on_bench"):
            self.use_on_bench.setEnabled(False)
            self.bench_note.setText("")
        self.report.setPlainText("Inputs edited — press Calculate ideal design.")
        self.figure.clear()
        self.canvas.draw_idle()

    def load_legacy(self):
        self.diameter.setValue(3)
        self.length.setValue(150)
        self.charge.setValue(3)
        self.definition.setCurrentIndex(1)
        self.wavelength.setValue(1029)
        self.radius.setValue(2)
        self.index.setValue(2.44)
        self.calculate()

    def calculate(self):
        self.invalidate()
        try:
            d = design_small_beam(
                diameter_um=self.diameter.value(), length_um=self.length.value(),
                charge=self.charge.value(), wavelength_nm=self.wavelength.value(),
                input_radius_mm=self.radius.value(), medium_index=self.index.value(),
                diameter_definition="actual" if self.definition.currentIndex() == 0 else "equivalent_q0",
            )
        except ValueError as exc:
            self.report.setPlainText(str(exc))
            return
        self.result = d
        self.export.setEnabled(True)
        ring = d["vortex_main_ring_diameter_m"] * 1e6
        self.report.setPlainText(
            f"IDEAL DESIGN ONLY — no live-bench change\n"
            f"Equivalent q=0 first-null diameter: {d['equivalent_l0_core_diameter_m'] * 1e6:.3f} µm\n"
            + (f"Actual q={d['ell']} bright-ring diameter: {ring:.3f} µm\n" if d['ell'] else "q=0: bright central spot, no dark vortex core\n")
            + f"Reference length: {d['predicted_bessel_length_m'] * 1e6:.2f} µm (NOT axial FWHM)\n"
            f"Required transverse magnification sample/SLM: {d['objective_map_demag']:.6f}\n"
            f"Required sample beam radius: {d['w0_sample_m'] * 1e6:.3f} µm; NA: {d['required_NA']:.3f}\n"
            f"Sample k⊥: {d['kr_sample_m_inv']:.3g} m⁻¹; cone: {d['cone_half_angle_deg']:.2f}°\n"
            f"{d['warning']}\nA micron-scale target needs sample-plane optics/sampling; the direct Beamage view cannot resolve it."
        )
        self._report_bench_route(d)
        radial_um, intensity = ideal_radial_reference(d)
        extent = radial_um[-1]
        axis = np.linspace(-extent, extent, 512)
        rr = np.hypot(axis[:, None], axis[None, :]) * 1e-6
        image = jv(abs(d['ell']), d['kr_sample_m_inv'] * rr) ** 2
        ax, profile = self.figure.subplots(1, 2)
        ax.imshow(image / image.max(), extent=(-extent, extent, -extent, extent),
                  origin="lower", cmap="inferno", vmin=0, vmax=1)
        ax.set(xlabel="x (µm)", ylabel="y (µm)", title=f"Ideal J{abs(d['ell'])} reference — not propagated")
        profile.plot(radial_um, intensity)
        profile.set(xlabel="Radius (µm)", ylabel="Normalised intensity", title="Ideal radial profile")
        profile.grid(alpha=0.3)
        self.canvas.draw_idle()

    def _report_bench_route(self, design) -> None:
        """What this target needs from *this* axicon, which the bench cannot change.

        The 4F relay is 1:1 and the axicon's k_perp is fixed, so the bench ring
        size is fixed too: only a demagnifying objective after the axicon can
        reach a micron-scale target.
        """

        engine = getattr(self.parent(), "resolution_engine", None)
        if engine is None:
            self.bench_note.setText("")
            return
        try:
            plan = engine.axicon_plan()
            charge = int(design["ell"])
            bench_ring_um = bench_ring_radius_um(k_perp_m_inv=plan.k_perp_m_inv, charge=charge)
            target_diameter_m = (
                design["vortex_main_ring_diameter_m"] if charge else design["equivalent_l0_core_diameter_m"]
            )
            demagnification = demagnification_for_ring(
                bench_ring_radius_um=bench_ring_um,
                target_ring_radius_um=0.5 * target_diameter_m * 1e6,
            )
            fitted = engine.sample_plane(charge=charge)
        except Exception as exc:
            self.bench_note.setText(f"Cannot map this target onto the current axicon: {exc}")
            return
        self.bench_demagnification = demagnification
        ratio = 1.0 / demagnification
        length_um = (plan.bessel_zone_mm * 1e3) * demagnification * demagnification
        self.use_on_bench.setEnabled(ratio > 1.0009)
        already = "" if fitted.demagnification >= 1.0 else f" The bench is set to {fitted.ratio_label} now."
        target_length_um = float(self.length.value())
        beam_radius_mm = float(getattr(self.parent(), "resolution_engine").canonical_beam_radius_mm)
        needed_radius_mm = beam_radius_mm * target_length_um / length_um if length_um > 0 else float("nan")
        self.bench_note.setText(
            f"ON THIS BENCH, KEEPING THIS AXICON (k⊥ fixed, so the magnification quoted above - which lets the "
            f"cone be chosen freely - does not apply): the axicon makes a q={design['ell']} ring of diameter "
            f"{2 * bench_ring_um:.1f} um, "
            f"so this target needs a 1:{ratio:.1f} objective after the axicon. At 1:{ratio:.1f} the Bessel region "
            f"becomes {length_um:.0f} um long for the {beam_radius_mm:g} mm input beam. For your {target_length_um:.0f} um "
            f"length target, set the input beam radius to about {needed_radius_mm:.3f} mm: ring size follows the "
            f"objective, length follows the beam radius.{already}"
        )

    def apply_to_bench(self) -> None:
        window = self.parent()
        demagnification = getattr(self, "bench_demagnification", None)
        if window is None or demagnification is None or not hasattr(window, "virtual_objective"):
            return
        window.virtual_objective.setValue(max(1, int(round(1.0 / demagnification))))
        if hasattr(window, "virtual_tabs") and hasattr(window, "virtual_tab_indices"):
            window.virtual_tabs.setCurrentIndex(window.virtual_tab_indices["settings"])
        self.bench_note.setText(
            self.bench_note.text()
            + "  Sent to Settings -> Objective: press Apply geometry there to commit it."
        )

    def export_design(self):
        if self.result is None:
            return
        name, _ = QFileDialog.getSaveFileName(self, "Export ideal design", "small_beam_design.json", "JSON (*.json)")
        if not name:
            return
        target = Path(name).with_suffix(".json")
        figure = target.with_suffix(".png")
        if figure.exists() and QMessageBox.question(self, "Replace figure?", f"Replace {figure}?") != QMessageBox.Yes:
            return
        try:
            target.write_text(json.dumps(self.result, indent=2), encoding="utf-8")
            self.figure.savefig(figure, dpi=400)
        except OSError as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
