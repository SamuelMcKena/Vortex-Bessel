from __future__ import annotations

import math

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from slm_lab_control.config import AppConfig

from labcontrol.devices.camera import DummyCameraProvider
from labcontrol.mode_controller import ModeAwareLabController
from labcontrol.state import ExperimentState, ExperimentStore
from labcontrol.ui.virtual_advanced_v2 import VirtualLabExperimentWindow
from labcontrol.virtual_lab import OperatingMode, VirtualLabGeometry
from labcontrol.virtual_lab_experiments import (
    AutoConvergeRunner,
    DEFAULT_SLM_CORRECTION_PARAMETERS,
    ExperimentalVirtualBenchEngine,
    ManualPerturbationSpec,
    build_even_z_plan,
    estimated_blind_cycle_frames,
    expand_measurement_plan,
)


def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_manual_perturbations_can_combine_beam_wavefront_and_alignment_faults() -> None:
    engine = ExperimentalVirtualBenchEngine()
    spec = ManualPerturbationSpec(
        radius_x_scale=1.08,
        radius_y_scale=0.94,
        beam_decentre_x_um=75.0,
        beam_decentre_y_um=-40.0,
        beam_pointing_x_mrad=0.11,
        beam_pointing_y_mrad=-0.07,
        curvature_radius_x_m=4.5,
        curvature_radius_y_m=-6.0,
        defocus_waves=0.12,
        astigmatism_x_waves=-0.08,
        coma_y_waves=0.09,
        axicon_decentre_x_um=35.0,
        axicon_decentre_y_um=-20.0,
    )
    handle = engine.apply_manual_perturbations(spec, seed=99)
    truth = engine.reveal_truth()["hidden_truth"]

    assert handle.kind == "MANUAL"
    assert truth["radius_x_scale"] == pytest.approx(1.08)
    assert truth["radius_y_scale"] == pytest.approx(0.94)
    assert truth["beam_decentre_m"] == pytest.approx([75e-6, -40e-6])
    assert truth["beam_pointing_rad"] == pytest.approx([0.11e-3, -0.07e-3])
    assert truth["axicon_decentre_m"] == pytest.approx([35e-6, -20e-6])
    assert truth["input_zernike_waves_rms"]["defocus"] == pytest.approx(0.12)
    assert truth["input_zernike_waves_rms"]["astigmatism_x"] == pytest.approx(-0.08)
    assert truth["input_zernike_waves_rms"]["coma_y"] == pytest.approx(0.09)
    assert truth["curvature_radius_x_m"] == pytest.approx(4.5)
    assert truth["curvature_radius_y_m"] == pytest.approx(-6.0)


def test_zero_curvature_is_collimated() -> None:
    engine = ExperimentalVirtualBenchEngine()
    engine.apply_manual_perturbations(ManualPerturbationSpec())
    hidden = engine._hidden  # test-only inspection of truth-side object
    assert math.isinf(hidden.curvature_radius_x_m)
    assert math.isinf(hidden.curvature_radius_y_m)


def test_measurement_plan_controls_plane_count_repeats_and_frame_budget() -> None:
    plan = build_even_z_plan(31.0, 46.0, 6)
    assert plan == (31.0, 34.0, 37.0, 40.0, 43.0, 46.0)
    expanded = expand_measurement_plan(plan, 3)
    assert len(expanded) == 18
    assert expanded[:3] == (31.0, 31.0, 31.0)
    assert expanded[-3:] == (46.0, 46.0, 46.0)

    frames = estimated_blind_cycle_frames(
        z_plane_count=6,
        repeats_per_plane=3,
        parameter_count=8,
        target_count=2,
    )
    assert frames == (2 + 7 * 8 * 2) * 6 * 3


def test_slm_correction_authority_includes_alignment_like_tip_tilt() -> None:
    assert DEFAULT_SLM_CORRECTION_PARAMETERS[:2] == ("tip_x", "tip_y")
    assert "defocus" in DEFAULT_SLM_CORRECTION_PARAMETERS
    assert "coma_x" in DEFAULT_SLM_CORRECTION_PARAMETERS


def test_auto_converge_executes_verified_slm_only_cycle_on_small_virtual_bench(tmp_path) -> None:
    state = ExperimentState.from_app_config(AppConfig())
    store = ExperimentStore(state)
    camera = DummyCameraProvider(store.snapshot, shape_yx=(48, 48))
    engine = ExperimentalVirtualBenchEngine(
        VirtualLabGeometry(preview_grid_n=48, validation_grid_n=64),
        quality="preview",
    )
    controller = ModeAwareLabController(
        store,
        tmp_path,
        camera_provider=camera,
        virtual_engine=engine,
    )
    controller.set_operating_mode(OperatingMode.VIRTUAL_LAB)
    engine.apply_manual_perturbations(
        ManualPerturbationSpec(defocus_waves=0.12, beam_pointing_x_mrad=0.03),
        seed=7,
    )
    result = AutoConvergeRunner(controller).run(
        (1.0, 2.0),
        targets=("SLM2",),
        parameters=("defocus",),
        initial_probe_amplitude_waves=0.10,
        max_cycles=1,
        convergence_fraction=0.0,
        patience=1,
        max_frames=100,
        seed=7,
    )
    assert len(result.cycles) == 1
    assert result.total_frames == estimated_blind_cycle_frames(2, 1, 1, 1)
    assert np.isfinite(result.initial_objective)
    assert np.isfinite(result.final_objective)
    # This run never enables mechanical Alignment Assist; all accepted changes
    # come through the virtual SLM provider.
    assert controller.virtual_engine.alignment_command.axicon_x_um == 0.0
    assert controller.virtual_engine.alignment_command.axicon_y_um == 0.0


def test_extended_gui_exposes_fault_builder_measurement_budget_and_autoconverge() -> None:
    qt = app()
    window = VirtualLabExperimentWindow()
    try:
        assert isinstance(window.mode_controller.virtual_engine, ExperimentalVirtualBenchEngine)
        assert window.perturb_beam_enabled.isChecked()
        assert window.perturb_wave_enabled.isChecked()
        assert window.perturb_axicon_enabled.isChecked()
        assert window.measure_z_count.value() == 6
        assert window.measure_repeats.value() == 1
        assert window.auto_converge_button.text() == "Start auto-converge SLM-only"
        assert window.correction_checks["tip_x"].isChecked()
        assert window.correction_checks["tip_y"].isChecked()
        assert "frames" in window.frame_budget_preview.toPlainText().lower()
    finally:
        window.close()
        qt.processEvents()
