"""Experiment-builder extension for the Unified Virtual Lab GUI.

This keeps the existing Home cockpit and Virtual Lab page, then adds explicit
multi-fault perturbation controls, measurement-budget controls and a repeated
SLM-only auto-convergence workflow.
"""

from __future__ import annotations

import faulthandler
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from slm_lab_control.config import AppConfig
from slm_lab_control.ui.style import APP_QSS

from ..devices.camera import DummyCameraProvider
from ..metrics import MetricEngine
from ..mode_controller import ModeAwareLabController
from ..state import ExperimentState, ExperimentStore
from ..virtual_lab import BlindCorrectionRunner, OperatingMode
from ..virtual_lab_experiments import (
    AutoConvergeRunner,
    DEFAULT_SLM_CORRECTION_PARAMETERS,
    ExperimentalVirtualBenchEngine,
    ManualPerturbationSpec,
    build_even_z_plan,
    estimated_blind_cycle_frames,
    expand_measurement_plan,
    random_manual_perturbation,
)
from .advanced import ADVANCED_QSS, PROJECT_ROOT, panel
from .controls import NoWheelDoubleSpinBox, NoWheelSpinBox
from .virtual_advanced import VirtualLabWindow


_FAULT_LOG_HANDLE = None


def _dspin(
    minimum: float,
    maximum: float,
    value: float,
    step: float,
    decimals: int = 4,
    suffix: str = "",
) -> NoWheelDoubleSpinBox:
    box = NoWheelDoubleSpinBox()
    box.setRange(float(minimum), float(maximum))
    box.setDecimals(int(decimals))
    box.setSingleStep(float(step))
    box.setValue(float(value))
    if suffix:
        box.setSuffix(suffix)
    return box


def _ispin(minimum: int, maximum: int, value: int) -> NoWheelSpinBox:
    box = NoWheelSpinBox()
    box.setRange(int(minimum), int(maximum))
    box.setValue(int(value))
    return box


class VirtualLabExperimentWindow(VirtualLabWindow):
    """VirtualLabWindow plus explicit perturbation and convergence experiments."""

    def __init__(
        self,
        store: ExperimentStore | None = None,
        controller: ModeAwareLabController | None = None,
        calibration=None,
    ):
        if controller is None:
            initial = ExperimentState.from_app_config(AppConfig())
            initial.system.experiment_label = "Optical lab session"
            initial.camera.shape_yx = (512, 512)
            store = store or ExperimentStore(initial)
            camera = DummyCameraProvider(store.snapshot, shape_yx=(512, 512))
            engine = ExperimentalVirtualBenchEngine()
            controller = ModeAwareLabController(
                store,
                PROJECT_ROOT,
                camera_provider=camera,
                virtual_engine=engine,
            )
        elif not isinstance(controller.virtual_engine, ExperimentalVirtualBenchEngine):
            raise TypeError(
                "VirtualLabExperimentWindow requires ExperimentalVirtualBenchEngine so manual perturbations "
                "and the correction model share one authoritative virtual bench."
            )
        super().__init__(store=store, controller=controller, calibration=calibration)

    @property
    def experiment_engine(self) -> ExperimentalVirtualBenchEngine:
        engine = self.mode_controller.virtual_engine
        if not isinstance(engine, ExperimentalVirtualBenchEngine):
            raise TypeError("Experimental virtual engine is not installed.")
        return engine

    def _build_virtual_page(self) -> None:
        super()._build_virtual_page()
        page = self.pages.widget(self.PAGE_VIRTUAL)
        layout = page.layout()
        if layout is None:
            raise RuntimeError("Virtual Lab page has no layout.")

        perturb_card = self._build_perturbation_builder()
        measurement_card = self._build_measurement_and_convergence()

        # Existing page layout: heading, note, scenario/geometry, camera/correction,
        # results.  Put experiment controls between geometry and acquisition.
        insert_at = max(2, layout.count() - 2)
        layout.insertWidget(insert_at, perturb_card)
        layout.insertWidget(insert_at + 1, measurement_card)
        self._update_frame_budget_preview()

    # ------------------------------------------------------------------
    # Perturbation builder
    # ------------------------------------------------------------------

    def _build_perturbation_builder(self) -> QWidget:
        card, box = panel(
            "Perturbation Builder — combine faults deliberately",
            "These values are hidden bench truth. They can all coexist. The SLM optimiser sees only intensity frames and the commands it issued.",
        )
        grid = QGridLayout()

        beam = QWidget()
        beam_form = QFormLayout(beam)
        self.perturb_beam_enabled = QCheckBox("Enable input-beam perturbations")
        self.perturb_beam_enabled.setChecked(True)
        self.perturb_beam_radius_mm = _dspin(0.05, 20.0, 2.0, 0.05, 3, " mm")
        self.perturb_radius_x = _dspin(0.5, 1.5, 1.0, 0.01, 4)
        self.perturb_radius_y = _dspin(0.5, 1.5, 1.0, 0.01, 4)
        self.perturb_dx = _dspin(-1000.0, 1000.0, 0.0, 5.0, 2, " µm")
        self.perturb_dy = _dspin(-1000.0, 1000.0, 0.0, 5.0, 2, " µm")
        self.perturb_tx = _dspin(-10.0, 10.0, 0.0, 0.01, 4, " mrad")
        self.perturb_ty = _dspin(-10.0, 10.0, 0.0, 0.01, 4, " mrad")
        self.perturb_curv_x = _dspin(-1000.0, 1000.0, 0.0, 0.1, 3, " m")
        self.perturb_curv_y = _dspin(-1000.0, 1000.0, 0.0, 0.1, 3, " m")
        self.perturb_curv_x.setToolTip("0 = collimated / infinite radius")
        self.perturb_curv_y.setToolTip("0 = collimated / infinite radius")
        beam_form.addRow(self.perturb_beam_enabled)
        beam_form.addRow("Nominal 1/e radius", self.perturb_beam_radius_mm)
        beam_form.addRow("Radius X scale", self.perturb_radius_x)
        beam_form.addRow("Radius Y scale", self.perturb_radius_y)
        beam_form.addRow("Beam X offset", self.perturb_dx)
        beam_form.addRow("Beam Y offset", self.perturb_dy)
        beam_form.addRow("Pointing X", self.perturb_tx)
        beam_form.addRow("Pointing Y", self.perturb_ty)
        beam_form.addRow("Curvature Rx", self.perturb_curv_x)
        beam_form.addRow("Curvature Ry", self.perturb_curv_y)

        wave = QWidget()
        wave_form = QFormLayout(wave)
        self.perturb_wave_enabled = QCheckBox("Enable hidden wavefront aberration")
        self.perturb_wave_enabled.setChecked(True)
        self.perturb_defocus = _dspin(-2.0, 2.0, 0.0, 0.01, 4, " waves")
        self.perturb_astig_x = _dspin(-2.0, 2.0, 0.0, 0.01, 4, " waves")
        self.perturb_astig_y = _dspin(-2.0, 2.0, 0.0, 0.01, 4, " waves")
        self.perturb_coma_x = _dspin(-2.0, 2.0, 0.0, 0.01, 4, " waves")
        self.perturb_coma_y = _dspin(-2.0, 2.0, 0.0, 0.01, 4, " waves")
        self.perturb_spherical = _dspin(-2.0, 2.0, 0.0, 0.01, 4, " waves")
        wave_form.addRow(self.perturb_wave_enabled)
        wave_form.addRow("Defocus", self.perturb_defocus)
        wave_form.addRow("Astigmatism X", self.perturb_astig_x)
        wave_form.addRow("Astigmatism XY", self.perturb_astig_y)
        wave_form.addRow("Coma X", self.perturb_coma_x)
        wave_form.addRow("Coma Y", self.perturb_coma_y)
        wave_form.addRow("Spherical", self.perturb_spherical)

        alignment = QWidget()
        align_form = QFormLayout(alignment)
        self.perturb_axicon_enabled = QCheckBox("Enable physical-axicon decentre")
        self.perturb_axicon_enabled.setChecked(True)
        self.perturb_axicon_x = _dspin(-1000.0, 1000.0, 0.0, 5.0, 2, " µm")
        self.perturb_axicon_y = _dspin(-1000.0, 1000.0, 0.0, 5.0, 2, " µm")
        align_form.addRow(self.perturb_axicon_enabled)
        align_form.addRow("Axicon X offset", self.perturb_axicon_x)
        align_form.addRow("Axicon Y offset", self.perturb_axicon_y)
        alignment_note = QLabel(
            "This is a real mechanical-type fault in the model. SLM-only correction is still tried first; "
            "only the residual is left for optional Alignment Assist."
        )
        alignment_note.setObjectName("Muted")
        alignment_note.setWordWrap(True)
        align_form.addRow("", alignment_note)

        grid.addWidget(beam, 0, 0)
        grid.addWidget(wave, 0, 1)
        grid.addWidget(alignment, 0, 2)
        box.addLayout(grid)

        row = QHBoxLayout()
        apply_button = QPushButton("Apply manual perturbations")
        apply_button.setObjectName("Accent")
        apply_button.clicked.connect(self._apply_manual_perturbations)
        random_button = QPushButton("Randomise multi-fault case")
        random_button.clicked.connect(self._randomise_manual_perturbations)
        nominal_button = QPushButton("Clear perturbations")
        nominal_button.clicked.connect(self._clear_manual_perturbations)
        row.addWidget(apply_button)
        row.addWidget(random_button)
        row.addWidget(nominal_button)
        box.addLayout(row)

        self.manual_perturbation_status = QLabel(
            "Zero/1.0 values are nominal. Multiple non-zero groups are intentionally allowed at once."
        )
        self.manual_perturbation_status.setObjectName("Muted")
        self.manual_perturbation_status.setWordWrap(True)
        box.addWidget(self.manual_perturbation_status)
        return card

    def _manual_spec_from_controls(self) -> ManualPerturbationSpec:
        beam = self.perturb_beam_enabled.isChecked()
        wave = self.perturb_wave_enabled.isChecked()
        axicon = self.perturb_axicon_enabled.isChecked()
        return ManualPerturbationSpec(
            radius_x_scale=self.perturb_radius_x.value() if beam else 1.0,
            radius_y_scale=self.perturb_radius_y.value() if beam else 1.0,
            beam_decentre_x_um=self.perturb_dx.value() if beam else 0.0,
            beam_decentre_y_um=self.perturb_dy.value() if beam else 0.0,
            beam_pointing_x_mrad=self.perturb_tx.value() if beam else 0.0,
            beam_pointing_y_mrad=self.perturb_ty.value() if beam else 0.0,
            curvature_radius_x_m=self.perturb_curv_x.value() if beam else 0.0,
            curvature_radius_y_m=self.perturb_curv_y.value() if beam else 0.0,
            defocus_waves=self.perturb_defocus.value() if wave else 0.0,
            astigmatism_x_waves=self.perturb_astig_x.value() if wave else 0.0,
            astigmatism_y_waves=self.perturb_astig_y.value() if wave else 0.0,
            coma_x_waves=self.perturb_coma_x.value() if wave else 0.0,
            coma_y_waves=self.perturb_coma_y.value() if wave else 0.0,
            spherical_waves=self.perturb_spherical.value() if wave else 0.0,
            axicon_decentre_x_um=self.perturb_axicon_x.value() if axicon else 0.0,
            axicon_decentre_y_um=self.perturb_axicon_y.value() if axicon else 0.0,
        )

    def _write_spec_to_controls(self, spec: ManualPerturbationSpec) -> None:
        values = {
            self.perturb_radius_x: spec.radius_x_scale,
            self.perturb_radius_y: spec.radius_y_scale,
            self.perturb_dx: spec.beam_decentre_x_um,
            self.perturb_dy: spec.beam_decentre_y_um,
            self.perturb_tx: spec.beam_pointing_x_mrad,
            self.perturb_ty: spec.beam_pointing_y_mrad,
            self.perturb_curv_x: spec.curvature_radius_x_m,
            self.perturb_curv_y: spec.curvature_radius_y_m,
            self.perturb_defocus: spec.defocus_waves,
            self.perturb_astig_x: spec.astigmatism_x_waves,
            self.perturb_astig_y: spec.astigmatism_y_waves,
            self.perturb_coma_x: spec.coma_x_waves,
            self.perturb_coma_y: spec.coma_y_waves,
            self.perturb_spherical: spec.spherical_waves,
            self.perturb_axicon_x: spec.axicon_decentre_x_um,
            self.perturb_axicon_y: spec.axicon_decentre_y_um,
        }
        for widget, value in values.items():
            widget.setValue(float(value))

    def _apply_manual_perturbations(self) -> None:
        try:
            self._ensure_virtual_mode()
            self.experiment_engine.set_quality(self.virtual_quality.currentText())
            self.experiment_engine.realistic_camera = self.virtual_realistic_camera.isChecked()
            self.experiment_engine.set_canonical_beam_radius_mm(self.perturb_beam_radius_mm.value())
            spec = self._manual_spec_from_controls()
            handle = self.experiment_engine.apply_manual_perturbations(
                spec,
                seed=self.virtual_seed.value(),
                label="MANUAL_MULTI_FAULT",
            )
            self.current_frame = None
            self.current_metrics = None
            self.manual_perturbation_status.setText(
                f"Applied {handle.scenario_id} • truth hash {handle.truth_hash[:16]}… • optimiser remains blind to exact values."
            )
            self.virtual_scenario_status.setText(
                f"{handle.scenario_id}\ntruth hash: {handle.truth_hash}\n"
                "MANUAL MULTI-FAULT: exact hidden values are not passed into the optimiser."
            )
            self.virtual_results.clear()
        except Exception as exc:
            self._show_error("Could not apply manual perturbations", exc)

    def _randomise_manual_perturbations(self) -> None:
        spec = random_manual_perturbation(self.virtual_seed.value(), severity=1.0)
        self._write_spec_to_controls(spec)
        self._apply_manual_perturbations()

    def _clear_manual_perturbations(self) -> None:
        self._write_spec_to_controls(ManualPerturbationSpec())
        self.perturb_beam_radius_mm.setValue(2.0)
        self._apply_manual_perturbations()

    # ------------------------------------------------------------------
    # Measurement plan / correction authority / convergence
    # ------------------------------------------------------------------

    def _build_measurement_and_convergence(self) -> QWidget:
        card, box = panel(
            "Measurement budget + iterative SLM recovery",
            "Choose how many mock camera planes/repeats are measured, what phase authority the SLMs may use, and when repeated correction should stop.",
        )
        outer = QHBoxLayout()

        measure = QWidget()
        measure_form = QFormLayout(measure)
        self.measure_z_start = _dspin(-5000.0, 5000.0, 31.0, 0.5, 3, " mm")
        self.measure_z_stop = _dspin(-5000.0, 5000.0, 46.0, 0.5, 3, " mm")
        self.measure_z_count = _ispin(2, 101, 6)
        self.measure_repeats = _ispin(1, 20, 1)
        build = QPushButton("Build evenly spaced z plan")
        build.clicked.connect(self._build_z_plan_from_controls)
        measure_form.addRow("z start", self.measure_z_start)
        measure_form.addRow("z stop", self.measure_z_stop)
        measure_form.addRow("Distinct z planes", self.measure_z_count)
        measure_form.addRow("Repeats per plane", self.measure_repeats)
        measure_form.addRow("", build)

        authority = QWidget()
        authority_form = QFormLayout(authority)
        self.correction_checks: dict[str, QCheckBox] = {}
        labels = {
            "tip_x": "Tip / tilt X (alignment-like)",
            "tip_y": "Tip / tilt Y (alignment-like)",
            "defocus": "Defocus",
            "astig_x": "Astigmatism X",
            "astig_xy": "Astigmatism XY",
            "coma_x": "Coma X",
            "coma_y": "Coma Y",
            "spherical": "Spherical",
        }
        for key in DEFAULT_SLM_CORRECTION_PARAMETERS:
            cb = QCheckBox(labels[key])
            cb.setChecked(True)
            self.correction_checks[key] = cb
            authority_form.addRow(cb)
        authority_note = QLabel(
            "Tip/tilt is intentionally SLM authority: beam pointing / some apparent alignment walk should be tried optically before mechanical Alignment Assist."
        )
        authority_note.setObjectName("Muted")
        authority_note.setWordWrap(True)
        authority_form.addRow("", authority_note)

        converge = QWidget()
        converge_form = QFormLayout(converge)
        self.converge_max_cycles = _ispin(1, 20, 5)
        self.converge_tolerance_pct = _dspin(0.0, 20.0, 0.5, 0.1, 3, " %")
        self.converge_patience = _ispin(1, 10, 2)
        self.converge_max_frames = _ispin(10, 100000, 5000)
        converge_form.addRow("Maximum correction cycles", self.converge_max_cycles)
        converge_form.addRow("Stop below improvement", self.converge_tolerance_pct)
        converge_form.addRow("Consecutive low-gain cycles", self.converge_patience)
        converge_form.addRow("Maximum camera frames", self.converge_max_frames)

        outer.addWidget(measure)
        outer.addWidget(authority)
        outer.addWidget(converge)
        box.addLayout(outer)

        action_row = QHBoxLayout()
        self.auto_converge_button = QPushButton("Start auto-converge SLM-only")
        self.auto_converge_button.setObjectName("Accent")
        self.auto_converge_button.clicked.connect(self._start_auto_converge)
        estimate_button = QPushButton("Recalculate frame budget")
        estimate_button.clicked.connect(self._update_frame_budget_preview)
        action_row.addWidget(self.auto_converge_button)
        action_row.addWidget(estimate_button)
        box.addLayout(action_row)

        self.frame_budget_preview = QPlainTextEdit()
        self.frame_budget_preview.setReadOnly(True)
        self.frame_budget_preview.setMaximumHeight(105)
        box.addWidget(self.frame_budget_preview)

        for widget in (
            self.measure_z_start,
            self.measure_z_stop,
            self.measure_z_count,
            self.measure_repeats,
            self.converge_max_cycles,
            self.converge_max_frames,
            self.converge_tolerance_pct,
        ):
            widget.valueChanged.connect(lambda _value: self._update_frame_budget_preview())
        for cb in self.correction_checks.values():
            cb.toggled.connect(lambda _checked: self._update_frame_budget_preview())
        self.virtual_target.currentTextChanged.connect(lambda _text: self._update_frame_budget_preview())
        return card

    def _build_z_plan_from_controls(self) -> None:
        plan = build_even_z_plan(
            self.measure_z_start.value(),
            self.measure_z_stop.value(),
            self.measure_z_count.value(),
        )
        self.virtual_z_plan.setText(", ".join(f"{value:.6g}" for value in plan))
        self._update_frame_budget_preview()

    def _selected_correction_parameters(self) -> tuple[str, ...]:
        return tuple(key for key, cb in self.correction_checks.items() if cb.isChecked())

    def _selected_targets(self) -> tuple[str, ...]:
        text = self.virtual_target.currentText()
        return ("SLM1", "SLM2") if text == "BOTH" else (text,)

    def _update_frame_budget_preview(self) -> None:
        if not hasattr(self, "frame_budget_preview"):
            return
        try:
            z_count = self.measure_z_count.value()
            repeats = self.measure_repeats.value()
            parameters = self._selected_correction_parameters()
            targets = self._selected_targets()
            one_cycle = estimated_blind_cycle_frames(z_count, repeats, len(parameters), len(targets))
            max_cycles = self.converge_max_cycles.value()
            worst_case = one_cycle * max_cycles
            text = (
                f"One full correction cycle: ~{one_cycle:,} numerical camera frames\n"
                f"Worst case at {max_cycles} cycle(s): ~{worst_case:,} frames\n"
                f"Configured hard budget: {self.converge_max_frames.value():,} frames • "
                f"{len(parameters)} SLM mode(s) • {len(targets)} target SLM(s)"
            )
            if one_cycle > self.converge_max_frames.value():
                text += "\nWARNING: budget is too small for even one complete verified cycle."
            self.frame_budget_preview.setPlainText(text)
        except Exception as exc:
            self.frame_budget_preview.setPlainText(f"Budget preview unavailable: {exc}")

    def _expanded_z_plan(self) -> tuple[float, ...]:
        return expand_measurement_plan(self._parse_z_plan(), self.measure_repeats.value())

    def _start_blind_virtual_correction(self) -> None:
        """Enhanced single-run button: selected SLM modes + measurement repeats."""
        try:
            self._ensure_virtual_mode()
            base_z = self._parse_z_plan()
            expanded_z = expand_measurement_plan(base_z, self.measure_repeats.value())
            targets = self._selected_targets()
            parameters = self._selected_correction_parameters()
            if not parameters:
                raise ValueError("Select at least one SLM correction mode.")
            runner = BlindCorrectionRunner(self.mode_controller)
            amplitude = self.virtual_probe.value()
            passes = self.virtual_passes.value()

            def task(**callbacks):
                return runner.run(
                    expanded_z,
                    targets=targets,
                    parameters=parameters,
                    probe_amplitude_waves=amplitude,
                    passes=passes,
                    seed=self.virtual_seed.value(),
                    **callbacks,
                )

            self.virtual_progress.setPlainText(
                f"Starting BLIND SLM correction • targets={targets} • modes={parameters}\n"
                f"z={base_z} • repeats/plane={self.measure_repeats.value()}\n"
                "Tip/tilt is allowed when selected, so alignment-like pointing errors are attempted with SLMs first.\n"
                "Hidden truth is not passed to the correction runner."
            )
            self._start_virtual_task(task)
        except Exception as exc:
            self._show_error("Could not start blind correction", exc)

    def _start_auto_converge(self) -> None:
        try:
            self._ensure_virtual_mode()
            base_z = self._parse_z_plan()
            targets = self._selected_targets()
            parameters = self._selected_correction_parameters()
            if not parameters:
                raise ValueError("Select at least one SLM correction mode.")
            runner = AutoConvergeRunner(self.mode_controller, metric_engine=MetricEngine())

            def task(**callbacks):
                return runner.run(
                    base_z,
                    repeats_per_plane=self.measure_repeats.value(),
                    targets=targets,
                    parameters=parameters,
                    initial_probe_amplitude_waves=self.virtual_probe.value(),
                    max_cycles=self.converge_max_cycles.value(),
                    convergence_fraction=self.converge_tolerance_pct.value() / 100.0,
                    patience=self.converge_patience.value(),
                    max_frames=self.converge_max_frames.value(),
                    seed=self.virtual_seed.value(),
                    **callbacks,
                )

            self.virtual_progress.setPlainText(
                "AUTO-CONVERGE SLM-ONLY STARTED\n"
                f"targets={targets}\n"
                f"modes={parameters}\n"
                f"z={base_z} • repeats/plane={self.measure_repeats.value()}\n"
                f"max cycles={self.converge_max_cycles.value()} • "
                f"stop below={self.converge_tolerance_pct.value():g}% for "
                f"{self.converge_patience.value()} cycle(s) • frame budget={self.converge_max_frames.value()}\n"
                "Mechanical Alignment Assist remains disabled during this run."
            )
            self._start_virtual_task(task)
        except Exception as exc:
            self._show_error("Could not start auto-converge SLM correction", exc)

    def _virtual_progress_event(self, payload: dict[str, Any]) -> None:
        kind = payload.get("kind")
        if kind == "cycle_start":
            self.virtual_progress.appendPlainText(
                f"\nCYCLE {payload['cycle']}/{payload['max_cycles']} • "
                f"probe={payload['probe_amplitude_waves']:.5g} waves • "
                f"~{payload['estimated_cycle_frames']} frames"
            )
            return
        if kind == "cycle_complete":
            self.virtual_progress.appendPlainText(
                f"CYCLE {payload['cycle']} COMPLETE • J {payload['objective_before']:.6g} → "
                f"{payload['objective_after']:.6g} • improvement {100.0 * payload['fractional_improvement']:.3f}% • "
                f"frames {payload['total_frames']}"
            )
            return
        super()._virtual_progress_event(payload)


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
    window = VirtualLabExperimentWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
