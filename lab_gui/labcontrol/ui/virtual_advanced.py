"""Unified lab GUI with an additional Virtual Lab / blind-correction workspace.

The existing AdvancedLabWindow Home cockpit is deliberately preserved.  This
subclass adds a first-class source mode and a tenth page for simulation control.
"""

from __future__ import annotations

import faulthandler
import json
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
)

from slm_lab_control.config import AppConfig
from slm_lab_control.ui.style import APP_QSS

from ..devices.camera import DummyCameraProvider, ReplayCameraProvider
from ..metrics import MetricEngine
from ..mode_controller import ModeAwareLabController
from ..state import ExperimentState, ExperimentStore
from ..virtual_lab import (
    AlignmentAssistRunner,
    BlindCorrectionRunner,
    OperatingMode,
    ScenarioKind,
    VirtualStackResult,
    capture_virtual_z_stack,
)
from .advanced import ADVANCED_QSS, AdvancedLabWindow, PROJECT_ROOT, panel
from .controls import NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSpinBox, blocked_signals
from .image_view import QuantitativeImageView


_FAULT_LOG_HANDLE = None


class _VirtualTaskWorker(QObject):
    frame = Signal(object)
    progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, task: Callable[..., Any]):
        super().__init__()
        self.task = task
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            result = self.task(
                frame_callback=self.frame.emit,
                progress_callback=self.progress.emit,
                cancelled=self.cancel_event.is_set,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class VirtualLabWindow(AdvancedLabWindow):
    PAGE_VIRTUAL = 9

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
            controller = ModeAwareLabController(store, PROJECT_ROOT, camera_provider=camera)
        elif not isinstance(controller, ModeAwareLabController):
            raise TypeError(
                "VirtualLabWindow requires ModeAwareLabController so virtual mode cannot bypass hardware safety."
            )
        self._virtual_thread: QThread | None = None
        self._virtual_worker: _VirtualTaskWorker | None = None
        self._last_virtual_stack: VirtualStackResult | None = None
        super().__init__(store=store, controller=controller, calibration=calibration)

    @property
    def mode_controller(self) -> ModeAwareLabController:
        return self.controller  # type: ignore[return-value]

    def _build(self) -> None:
        super()._build()

        # First-class source selector in the existing sidebar.  The Home cockpit
        # itself is intentionally not rearranged.
        sidebar = self.findChild(QFrame, "LabSidebar")
        if sidebar is None or sidebar.layout() is None:
            raise RuntimeError("Could not locate existing LabSidebar.")
        side = sidebar.layout()

        source_label = QLabel("EXPERIMENT SOURCE")
        source_label.setObjectName("Section")
        self.mode_choice = NoWheelComboBox()
        self.mode_choice.addItems([mode.value for mode in OperatingMode])
        self.mode_choice.setCurrentText(self.mode_controller.operating_mode.value)
        self.mode_badge = QLabel("LIVE LAB • providers disconnected until explicitly connected")
        self.mode_badge.setObjectName("WarnChip")
        self.mode_badge.setWordWrap(True)

        side.insertWidget(2, source_label)
        side.insertWidget(3, self.mode_choice)
        side.insertWidget(4, self.mode_badge)

        # Add a tenth page and nav item without disturbing the current 0..8 layout.
        self._build_virtual_page()
        virtual_nav = QPushButton("Virtual Lab")
        virtual_nav.setObjectName("LabNav")
        virtual_nav.setCheckable(True)
        virtual_nav.clicked.connect(lambda _checked=False: self.set_page(self.PAGE_VIRTUAL))
        # Insert before the stretch/status at the bottom.
        insert_at = max(0, side.count() - 2)
        side.insertWidget(insert_at, virtual_nav)
        self.nav_buttons.append(virtual_nav)

        for combo in (self.home_provider_choice, self.camera_provider_choice):
            if combo.findText("virtual") < 0:
                combo.addItem("virtual")

        self.mode_choice.currentTextChanged.connect(self._change_operating_mode)

    def _build_virtual_page(self) -> None:
        _, layout = self._page(
            "Virtual Lab / blind correction",
            "The digital twin impersonates the camera at the CameraFrame boundary. Hidden truth is not available to the optimiser.",
        )

        top = QSplitter()
        scenario_card, scenario_box = panel(
            "Blind scenario",
            "Seeded hidden errors are reproducible. Exact truth remains hidden until you explicitly reveal it.",
        )
        scenario_form = QFormLayout()
        self.virtual_scenario = NoWheelComboBox()
        self.virtual_scenario.addItems([item.value for item in ScenarioKind])
        self.virtual_scenario.setCurrentText(ScenarioKind.MIXED.value)
        self.virtual_seed = NoWheelSpinBox()
        self.virtual_seed.setRange(0, 2_000_000_000)
        self.virtual_seed.setValue(1048)
        self.virtual_quality = NoWheelComboBox()
        self.virtual_quality.addItems(["preview", "validation"])
        self.virtual_quality.setCurrentText("preview")
        self.virtual_realistic_camera = QCheckBox("Add deterministic detector noise")
        scenario_form.addRow("Scenario", self.virtual_scenario)
        scenario_form.addRow("Seed", self.virtual_seed)
        scenario_form.addRow("Numerical quality", self.virtual_quality)
        scenario_form.addRow("Camera model", self.virtual_realistic_camera)
        scenario_box.addLayout(scenario_form)
        buttons = QHBoxLayout()
        generate = QPushButton("Generate hidden scenario")
        generate.setObjectName("Accent")
        generate.clicked.connect(self._generate_virtual_scenario)
        reset = QPushButton("Reset nominal")
        reset.clicked.connect(self._reset_virtual_scenario)
        reveal = QPushButton("Reveal truth")
        reveal.setObjectName("Danger")
        reveal.clicked.connect(self._reveal_virtual_truth)
        buttons.addWidget(generate)
        buttons.addWidget(reset)
        buttons.addWidget(reveal)
        scenario_box.addLayout(buttons)
        self.virtual_scenario_status = QLabel("No blind scenario generated in this GUI session.")
        self.virtual_scenario_status.setObjectName("Muted")
        self.virtual_scenario_status.setWordWrap(True)
        scenario_box.addWidget(self.virtual_scenario_status)
        top.addWidget(scenario_card)

        geometry_card, geometry_box = panel(
            "Bench geometry / claim boundary",
            "Reported values and modelling placeholders are deliberately shown separately.",
        )
        self.virtual_geometry = QPlainTextEdit()
        self.virtual_geometry.setReadOnly(True)
        self.virtual_geometry.setMaximumHeight(260)
        geometry_box.addWidget(self.virtual_geometry)
        geom_form = QFormLayout()
        self.virtual_slm_sep = NoWheelDoubleSpinBox()
        self.virtual_slm_sep.setRange(0.0, 5000.0)
        self.virtual_slm_sep.setDecimals(4)
        self.virtual_slm_sep.setSuffix(" mm")
        self.virtual_slm_sep.setValue(self.mode_controller.virtual_engine.geometry.slm1_to_slm2_mm)
        self.virtual_axicon_angle = NoWheelDoubleSpinBox()
        self.virtual_axicon_angle.setRange(0.01, 45.0)
        self.virtual_axicon_angle.setDecimals(4)
        self.virtual_axicon_angle.setSuffix("° model base angle")
        self.virtual_axicon_angle.setValue(
            self.mode_controller.virtual_engine.geometry.axicon_model_base_angle_deg
        )
        geom_form.addRow("SLM1 → SLM2 model", self.virtual_slm_sep)
        geom_form.addRow("Axicon model angle", self.virtual_axicon_angle)
        geometry_box.addLayout(geom_form)
        apply_geometry = QPushButton("Apply model assumptions")
        apply_geometry.clicked.connect(self._apply_virtual_geometry)
        geometry_box.addWidget(apply_geometry)
        top.addWidget(geometry_card)
        top.setSizes([650, 850])
        layout.addWidget(top)

        body = QSplitter()
        image_card, image_box = panel(
            "Mock camera / z acquisition",
            "Same numerical frame contract as the camera pipeline; no false-colour pixels enter the metrics.",
        )
        self.virtual_camera_view = QuantitativeImageView()
        self.virtual_camera_view.setMinimumSize(520, 420)
        image_box.addWidget(self.virtual_camera_view, 1)
        z_form = QFormLayout()
        self.virtual_z_plan = QLineEdit("31, 34, 37, 40, 43, 46")
        self.virtual_z_plan.setToolTip(
            "Comma-separated virtual millimetres from the post-axicon model plane. Physical camera z=0 remains uncalibrated."
        )
        z_form.addRow("z planes (mm)", self.virtual_z_plan)
        image_box.addLayout(z_form)
        acquisition_buttons = QHBoxLayout()
        single = QPushButton("Capture current z")
        single.clicked.connect(self._capture_virtual_current)
        stack = QPushButton("Capture Z stack")
        stack.setObjectName("Accent")
        stack.clicked.connect(self._capture_virtual_stack)
        acquisition_buttons.addWidget(single)
        acquisition_buttons.addWidget(stack)
        image_box.addLayout(acquisition_buttons)
        self.virtual_stack_status = QPlainTextEdit()
        self.virtual_stack_status.setReadOnly(True)
        self.virtual_stack_status.setMaximumHeight(170)
        image_box.addWidget(self.virtual_stack_status)
        body.addWidget(image_card)

        correction_card, correction_box = panel(
            "Blind correction",
            "SLM-only sensorless modal probing runs first. Alignment Assist is an explicit second phase.",
        )
        correction_form = QFormLayout()
        self.virtual_target = NoWheelComboBox()
        self.virtual_target.addItems(["SLM1", "SLM2", "BOTH"])
        self.virtual_target.setCurrentText("BOTH")
        self.virtual_probe = NoWheelDoubleSpinBox()
        self.virtual_probe.setRange(0.005, 1.0)
        self.virtual_probe.setDecimals(3)
        self.virtual_probe.setValue(0.15)
        self.virtual_probe.setSuffix(" waves")
        self.virtual_passes = NoWheelSpinBox()
        self.virtual_passes.setRange(1, 5)
        self.virtual_passes.setValue(2)
        correction_form.addRow("Correction target", self.virtual_target)
        correction_form.addRow("Initial modal probe", self.virtual_probe)
        correction_form.addRow("Passes", self.virtual_passes)
        correction_box.addLayout(correction_form)

        run_buttons = QHBoxLayout()
        optimise = QPushButton("Start blind SLM correction")
        optimise.setObjectName("Accent")
        optimise.clicked.connect(self._start_blind_virtual_correction)
        align = QPushButton("Alignment Assist")
        align.setToolTip("Separate virtual x/y axicon alignment search after SLM-only correction.")
        align.clicked.connect(self._start_virtual_alignment)
        cancel = QPushButton("Cancel task")
        cancel.setObjectName("Danger")
        cancel.clicked.connect(self._cancel_virtual_task)
        run_buttons.addWidget(optimise)
        run_buttons.addWidget(align)
        run_buttons.addWidget(cancel)
        correction_box.addLayout(run_buttons)
        self.virtual_progress = QPlainTextEdit()
        self.virtual_progress.setReadOnly(True)
        correction_box.addWidget(self.virtual_progress, 1)
        body.addWidget(correction_card)
        body.setSizes([850, 650])
        layout.addWidget(body, 1)

        truth_card, truth_box = panel(
            "Results / truth reveal",
            "Truth reveal is validation-only. Similar output intensity does not prove unique physical parameter identification.",
        )
        self.virtual_results = QPlainTextEdit()
        self.virtual_results.setReadOnly(True)
        self.virtual_results.setMaximumHeight(190)
        truth_box.addWidget(self.virtual_results)
        layout.addWidget(truth_card)

        self._refresh_virtual_geometry_summary()

    def set_page(self, index: int) -> None:
        if index != self.PAGE_VIRTUAL:
            super().set_page(index)
            return
        if not hasattr(self, "pages") or self.pages.count() <= self.PAGE_VIRTUAL:
            return
        self.pages.setCurrentIndex(self.PAGE_VIRTUAL)
        self.page_title.setText("Virtual Lab")
        for position, button in enumerate(self.nav_buttons):
            button.setChecked(position == self.PAGE_VIRTUAL)
        self._rerender_virtual()

    def _refresh_state(self, state: ExperimentState, *, refresh_slms: set[str] | None = None) -> None:
        super()._refresh_state(state, refresh_slms=refresh_slms)
        if not hasattr(self, "mode_choice"):
            return
        mode = self.mode_controller.operating_mode
        with blocked_signals((self.mode_choice,)):
            self.mode_choice.setCurrentText(mode.value)

        if mode is OperatingMode.VIRTUAL_LAB:
            text = "VIRTUAL LAB • SIMULATED DATA • virtual SLM/camera/stage"
            object_name = "WarnChip"
            for card in self.slm_quick_cards.values():
                card.connection.setText(
                    "VIRTUAL" if card.connection.text() == "CONNECTED" else "VIRTUAL OFF"
                )
        elif mode is OperatingMode.RECORDED_LAB:
            text = "RECORDED LAB • MEASURED REPLAY • SLM commands disabled"
            object_name = "WarnChip"
            for card in self.slm_quick_cards.values():
                card.connection.setText("READ ONLY")
        else:
            text = "LIVE LAB • physical commands require explicit connection/cast"
            object_name = "GoodChip" if state.camera.provider == "beamage" else "WarnChip"

        self.mode_badge.setText(text)
        self.mode_badge.setObjectName(object_name)
        self.mode_badge.style().unpolish(self.mode_badge)
        self.mode_badge.style().polish(self.mode_badge)
        self.home_camera_status.setText(text + "\n" + self.home_camera_status.text())

    def _change_operating_mode(self, text: str) -> None:
        if self._syncing:
            return
        try:
            self.stop_live()
            message = self.mode_controller.set_operating_mode(OperatingMode(text))
            self.current_frame = None
            self.current_metrics = None
            self.home_operation_status.setText(message)
            with blocked_signals((self.home_provider_choice, self.camera_provider_choice)):
                provider = {
                    OperatingMode.LIVE_LAB: "beamage",
                    OperatingMode.RECORDED_LAB: "replay",
                    OperatingMode.VIRTUAL_LAB: "virtual",
                }[self.mode_controller.operating_mode]
                self.home_provider_choice.setCurrentText(provider)
                self.camera_provider_choice.setCurrentText(provider)
            self._refresh_state(self.store.snapshot())
        except Exception as exc:
            self._show_error("Operating mode switch failed", exc)

    def _select_camera_provider(self, name: str) -> None:
        if self._syncing:
            return
        try:
            if name == "virtual":
                self.stop_live()
                self.mode_controller.set_operating_mode(OperatingMode.VIRTUAL_LAB)
                self._refresh_state(self.store.snapshot())
                return
            if name == "beamage":
                self.stop_live()
                self.mode_controller.set_operating_mode(OperatingMode.LIVE_LAB)
                self._refresh_state(self.store.snapshot())
                return
            if name == "replay":
                self.stop_live()
                self.mode_controller.set_operating_mode(OperatingMode.RECORDED_LAB)
                self.camera_status.setText("RECORDED LAB selected. Choose quantitative replay files.")
                self._refresh_state(self.store.snapshot())
                return
            if name == "dummy":
                self.stop_live()
                self.mode_controller.set_operating_mode(OperatingMode.VIRTUAL_LAB)
                self.controller.set_camera_provider(
                    DummyCameraProvider(self.store.snapshot, shape_yx=(512, 512)),
                    provider_name="dummy",
                )
                self._refresh_state(self.store.snapshot())
                return
            super()._select_camera_provider(name)
        except Exception as exc:
            self._show_error("Camera selection failed", exc)

    def _browse_replay(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Choose recorded laboratory camera frames",
            str(PROJECT_ROOT),
            "Quantitative frames (*.bmg *.npy *.txt *.csv *.bmp *.png *.tif *.tiff)",
        )
        if not paths:
            return
        try:
            self.stop_live()
            provider = ReplayCameraProvider(
                paths,
                loop=True,
                source_data_kind="REPLAY",
            )
            self.mode_controller.set_operating_mode(
                OperatingMode.RECORDED_LAB,
                replay_provider=provider,
            )
            with blocked_signals((self.camera_provider_choice, self.home_provider_choice, self.mode_choice)):
                self.camera_provider_choice.setCurrentText("replay")
                self.home_provider_choice.setCurrentText("replay")
                self.mode_choice.setCurrentText(OperatingMode.RECORDED_LAB.value)
            self.camera_status.setText(
                f"RECORDED LAB • {len(paths)} measured replay frame(s) selected • READ ONLY"
            )
            self._refresh_state(self.store.snapshot())
        except Exception as exc:
            self._show_error("Recorded replay selection failed", exc)

    def _formal_capture(self) -> None:
        if self.mode_controller.operating_mode is OperatingMode.RECORDED_LAB:
            QMessageBox.information(
                self,
                "Recorded Lab is read-only",
                "Historical data cannot respond to a fresh SLM cast/capture. Use Sessions / Compare for re-analysis.",
            )
            return
        super()._formal_capture()

    def _on_frame(self, frame) -> None:
        super()._on_frame(frame)
        if (
            hasattr(self, "virtual_camera_view")
            and self.mode_controller.operating_mode is OperatingMode.VIRTUAL_LAB
        ):
            self._rerender_virtual()

    def _rerender_virtual(self) -> None:
        if not hasattr(self, "virtual_camera_view") or self.current_frame is None:
            return
        self.virtual_camera_view.set_quantitative_frame(
            self.current_frame,
            self.current_metrics,
            colour=self.home_colour_mode.currentText(),
            scale=self.home_display_scale.currentText(),
            gamma=self.home_display_gamma.value(),
            show_centre=True,
            show_ring=True,
            show_roi=True,
        )

    def _ensure_virtual_mode(self) -> None:
        if self.mode_controller.operating_mode is not OperatingMode.VIRTUAL_LAB:
            self.mode_controller.set_operating_mode(OperatingMode.VIRTUAL_LAB)
            with blocked_signals((self.mode_choice, self.home_provider_choice, self.camera_provider_choice)):
                self.mode_choice.setCurrentText(OperatingMode.VIRTUAL_LAB.value)
                self.home_provider_choice.setCurrentText("virtual")
                self.camera_provider_choice.setCurrentText("virtual")

    def _refresh_virtual_geometry_summary(self) -> None:
        if not hasattr(self, "virtual_geometry"):
            return
        summary = self.mode_controller.virtual_engine.geometry.public_summary()
        self.virtual_geometry.setPlainText(json.dumps(summary, indent=2, ensure_ascii=False))

    def _apply_virtual_geometry(self) -> None:
        try:
            self.mode_controller.virtual_engine.set_geometry(
                slm1_to_slm2_mm=self.virtual_slm_sep.value(),
                axicon_model_base_angle_deg=self.virtual_axicon_angle.value(),
            )
            self._refresh_virtual_geometry_summary()
            self.virtual_scenario_status.setText(
                "Model assumptions updated. These values remain explicitly uncalibrated until bound to bench measurements."
            )
        except Exception as exc:
            self._show_error("Virtual geometry update failed", exc)

    def _generate_virtual_scenario(self) -> None:
        try:
            self._ensure_virtual_mode()
            engine = self.mode_controller.virtual_engine
            engine.set_quality(self.virtual_quality.currentText())
            engine.realistic_camera = self.virtual_realistic_camera.isChecked()
            handle = engine.generate_scenario(
                self.virtual_scenario.currentText(),
                self.virtual_seed.value(),
            )
            self.current_frame = None
            self.current_metrics = None
            self.virtual_scenario_status.setText(
                f"{handle.scenario_id}\ntruth hash: {handle.truth_hash}\n"
                "BLIND: exact hidden coefficients are not exposed to the optimiser."
            )
            self.virtual_results.clear()
        except Exception as exc:
            self._show_error("Could not generate virtual scenario", exc)

    def _reset_virtual_scenario(self) -> None:
        try:
            self._ensure_virtual_mode()
            handle = self.mode_controller.virtual_engine.reset_scenario()
            self.virtual_scenario_status.setText(
                f"{handle.scenario_id} • nominal hidden-error state • still SIMULATED"
            )
            self.virtual_results.clear()
        except Exception as exc:
            self._show_error("Could not reset virtual scenario", exc)

    def _reveal_virtual_truth(self) -> None:
        truth = self.mode_controller.virtual_engine.reveal_truth()
        self.virtual_results.setPlainText(json.dumps(truth, indent=2, ensure_ascii=False))

    def _parse_z_plan(self) -> tuple[float, ...]:
        text = self.virtual_z_plan.text().strip()
        values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
        if len(values) < 2:
            raise ValueError("Provide at least two comma-separated z planes.")
        return values

    def _ensure_virtual_cast(self) -> None:
        # Explicit dry-run cast.  This can never reach HEDS because the mode-aware
        # controller has installed VirtualSlmProvider.
        self.mode_controller.cast(("SLM1", "SLM2"), persist=False)

    def _capture_virtual_current(self) -> None:
        try:
            self._ensure_virtual_mode()
            self._ensure_virtual_cast()
            if self.controller.camera_provider is None or not self.controller.camera_provider.connected:
                self.controller.connect_camera()
            self.controller.start_camera()
            z_plan = self._parse_z_plan()
            self.mode_controller.move_stage(z_plan[0])
            frame = self.controller.acquire_frame(fresh=True)
            self._on_frame(frame)
            self.virtual_stack_status.setPlainText(
                f"Captured {frame.frame_id}\nz={frame.z_mm} mm\n"
                f"data_origin={frame.metadata.get('data_origin')}\n"
                f"relay={frame.metadata.get('relay_mode')}"
            )
        except Exception as exc:
            self._show_error("Virtual capture failed", exc)

    def _start_virtual_task(self, task: Callable[..., Any]) -> None:
        if self._virtual_thread is not None and self._virtual_thread.isRunning():
            raise RuntimeError("A Virtual Lab task is already running.")
        self.stop_live()
        thread = QThread(self)
        worker = _VirtualTaskWorker(task)
        worker.moveToThread(thread)
        worker.frame.connect(self._on_frame)
        worker.progress.connect(self._virtual_progress_event)
        worker.finished.connect(self._virtual_task_finished)
        worker.failed.connect(self._virtual_task_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.started.connect(worker.run)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._virtual_task_cleanup)
        self._virtual_thread = thread
        self._virtual_worker = worker
        thread.start()

    def _capture_virtual_stack(self) -> None:
        try:
            self._ensure_virtual_mode()
            self._ensure_virtual_cast()
            z_plan = self._parse_z_plan()

            def task(**callbacks):
                return capture_virtual_z_stack(
                    self.mode_controller,
                    z_plan,
                    metric_engine=MetricEngine(),
                    **callbacks,
                )

            self.virtual_progress.setPlainText("Starting virtual z-stack…")
            self._start_virtual_task(task)
        except Exception as exc:
            self._show_error("Could not start virtual z-stack", exc)

    def _start_blind_virtual_correction(self) -> None:
        try:
            self._ensure_virtual_mode()
            z_plan = self._parse_z_plan()
            target_text = self.virtual_target.currentText()
            targets = ("SLM1", "SLM2") if target_text == "BOTH" else (target_text,)
            amplitude = self.virtual_probe.value()
            passes = self.virtual_passes.value()
            runner = BlindCorrectionRunner(self.mode_controller)

            def task(**callbacks):
                return runner.run(
                    z_plan,
                    targets=targets,
                    probe_amplitude_waves=amplitude,
                    passes=passes,
                    seed=self.virtual_seed.value(),
                    **callbacks,
                )

            self.virtual_progress.setPlainText(
                f"Starting BLIND SLM correction • targets={targets} • z={z_plan}\n"
                "Hidden truth is not passed to the correction runner."
            )
            self._start_virtual_task(task)
        except Exception as exc:
            self._show_error("Could not start blind correction", exc)

    def _start_virtual_alignment(self) -> None:
        try:
            self._ensure_virtual_mode()
            z_plan = self._parse_z_plan()
            runner = AlignmentAssistRunner(self.mode_controller)

            def task(**callbacks):
                return runner.run(z_plan, **callbacks)

            self.virtual_progress.appendPlainText(
                "\nALIGNMENT ASSIST explicitly enabled: virtual axicon x/y only."
            )
            self._start_virtual_task(task)
        except Exception as exc:
            self._show_error("Could not start Alignment Assist", exc)

    def _cancel_virtual_task(self) -> None:
        if self._virtual_worker is not None:
            self._virtual_worker.cancel()
            self.virtual_progress.appendPlainText("Cancellation requested…")

    def _virtual_progress_event(self, payload: dict[str, Any]) -> None:
        kind = payload.get("kind", "progress")
        if kind == "z_plane":
            line = f"z {payload['index']}/{payload['count']}: {payload['z_mm']:.4g} mm"
        elif kind == "mode_start":
            line = (
                f"pass {payload['pass']}/{payload['passes']} • {payload['slm']} • "
                f"{payload['parameter']} • probe {payload['probe_amplitude_waves']:.4g} waves"
            )
        elif kind == "candidate":
            line = (
                f"  {payload['slm']} {payload['parameter']} {payload['command']:+.4g} "
                f"→ J={payload['score']:.6g}"
            )
        elif kind == "alignment_candidate":
            line = (
                f"  alignment {payload['axis']}={payload['command_um']:+.3g} µm "
                f"→ J={payload['objective']:.6g}"
            )
        else:
            line = json.dumps(payload)
        self.virtual_progress.appendPlainText(line)

    def _virtual_task_finished(self, result: Any) -> None:
        if isinstance(result, VirtualStackResult):
            self._last_virtual_stack = result
            payload = result.summary()
            self.virtual_stack_status.setPlainText(json.dumps(payload, indent=2))
        elif hasattr(result, "to_dict"):
            payload = result.to_dict()
            self.virtual_results.setPlainText(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            self.virtual_results.setPlainText(str(result))
        self.virtual_progress.appendPlainText("Task complete.")

    def _virtual_task_failed(self, message: str) -> None:
        self.virtual_progress.appendPlainText("FAILED: " + message)
        QMessageBox.warning(self, "Virtual Lab task failed", message)

    def _virtual_task_cleanup(self) -> None:
        self._virtual_thread = None
        self._virtual_worker = None

    def closeEvent(self, event):  # noqa: N802
        if self._virtual_worker is not None:
            self._virtual_worker.cancel()
        thread = self._virtual_thread
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait(3000)
        super().closeEvent(event)


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
    window = VirtualLabWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
