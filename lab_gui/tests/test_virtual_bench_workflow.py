from __future__ import annotations

import copy
import time

import numpy as np
import pytest
from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import QLabel, QApplication, QMessageBox, QPushButton, QScrollArea

from labcontrol.devices.camera import DummyCameraProvider
from labcontrol.mode_controller import ModeAwareLabController
from labcontrol.state import ExperimentState, ExperimentStore
from labcontrol.ui.advanced import ADVANCED_QSS
from labcontrol.ui.virtual_advanced_v3 import VirtualLabResolutionWindow
from labcontrol.virtual_lab import OperatingMode, VirtualLabGeometry
from labcontrol.virtual_lab_resolution import ResolutionAwareVirtualBenchEngine, VirtualOutputResolution
from slm_lab_control.config import AppConfig
from slm_lab_control.ui.style import APP_QSS


def _app() -> QApplication:
    app = QApplication.instance() or QApplication([])
    # Layout assertions are only meaningful against the stylesheet the launcher
    # installs; the default style has noticeably wider control metrics.
    if not app.styleSheet():
        app.setStyleSheet(APP_QSS + ADVANCED_QSS)
    return app


def _small_window(tmp_path) -> VirtualLabResolutionWindow:
    state = ExperimentState.from_app_config(AppConfig())
    store = ExperimentStore(state)
    engine = ResolutionAwareVirtualBenchEngine(
        VirtualLabGeometry(preview_grid_n=48, validation_grid_n=64, axicon_k_perp_m_inv=None),
        beamage_n=64,
        maximum_grid_n=128,
    )
    controller = ModeAwareLabController(
        store,
        tmp_path,
        camera_provider=DummyCameraProvider(store.snapshot, shape_yx=(48, 48)),
        virtual_engine=engine,
    )
    return VirtualLabResolutionWindow(store=store, controller=controller)


def test_virtual_bench_cockpit_fits_without_scrolling_and_exposes_controls(tmp_path) -> None:
    """Camera beside controls, nothing reachable only by sideways scrolling.

    Checked at both the operator's 1280x800 laptop (client ~1264x704) and a
    1920x1080 monitor.
    """

    app = _app()
    window = _small_window(tmp_path)
    try:
        window.show()
        window.set_page(window.PAGE_VIRTUAL)
        for width, height in ((1264, 704), (1920, 1040)):
            window.resize(width, height)
            app.processEvents()
            app.processEvents()
            page = window.pages.widget(window.PAGE_VIRTUAL)
            assert isinstance(page, QScrollArea)
            assert page.horizontalScrollBar().maximum() == 0
            assert page.verticalScrollBar().maximum() == 0
            assert window.virtual_cockpit_split.orientation() == Qt.Horizontal
            for index in window.virtual_tab_indices.values():
                window.virtual_tabs.setCurrentIndex(index)
                app.processEvents()
                assert window.virtual_tabs.widget(index).horizontalScrollBar().maximum() == 0, (width, index)
        assert window.virtual_camera_z.value() == 31.0
        assert window.virtual_realistic_camera.isChecked()
        assert not window.virtual_apply_button.isEnabled()
        assert window.slm_detail_views["SLM1"] is not None
        assert window.slm_detail_views["SLM2"] is not None
    finally:
        window.close()
        app.processEvents()

def test_camera_card_rows_are_never_drawn_over_the_image(tmp_path) -> None:
    """Regression: squeezed rows overlapped the camera view on a laptop screen."""

    app = _app()
    window = _small_window(tmp_path)
    try:
        window.resize(1264, 704)
        window.show()
        window.set_page(window.PAGE_VIRTUAL)
        app.processEvents()
        app.processEvents()
        view = window.virtual_camera_view
        card = view.parentWidget()
        assert card.height() >= card.minimumSizeHint().height()
        below = view.geometry().bottom()
        for widget in (window.virtual_auto_fit, window.virtual_camera_metrics):
            assert widget.geometry().top() > below
        for label in card.findChildren(QLabel):
            if label.text() in ("Colour", "Scale"):
                assert not label.isVisible(), "a detached label was left painted over the camera"
    finally:
        window.close()
        app.processEvents()

def test_window_can_be_narrower_than_a_laptop_screen() -> None:
    """Regression: an unwrapped header label pinned the window to ~1700 px.

    The window refused to be resized below that, so on a smaller display the
    right-hand side of every page was simply off screen.
    """

    app = _app()
    window = VirtualLabResolutionWindow()
    try:
        assert window.minimumSizeHint().width() <= 1100
        window.resize(1280, 800)
        window.show()
        app.processEvents()
        assert window.width() == 1280
    finally:
        window.close()
        app.processEvents()


def test_moving_the_camera_during_a_live_view_reaches_the_new_plane(tmp_path) -> None:
    """Regression: a frame already in flight used to restore the previous z.

    The synthetic camera reads z out of the state, so writing the frame's z back
    into that same state raced every stage command and pinned the live view to
    whichever plane it started on.
    """

    app = _app()
    window = _small_window(tmp_path)
    errors = []
    window._show_error = lambda title, exc: errors.append((title, str(exc)))
    try:
        window._apply_virtual_vortex_route()
        window.virtual_camera_z.setValue(31.0)
        window._start_virtual_live(recast=True)
        deadline = time.monotonic() + 60
        while window.current_frame is None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert window.current_frame is not None
        assert window.current_frame.z_mm == 31.0

        window.virtual_camera_step.setValue(4.0)
        window._step_virtual_camera(1)
        assert window.virtual_camera_z.value() == 35.0
        assert window.store.snapshot().camera.current_z_mm == 35.0

        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
            if window.current_frame is not None and window.current_frame.z_mm == 35.0:
                break
        assert window.current_frame.z_mm == 35.0, "live view never reached the commanded plane"
        assert not errors
    finally:
        window.stop_live()
        window.close()
        app.processEvents()


def test_virtual_cockpit_has_one_primary_path_for_each_action(tmp_path) -> None:
    app = _app()
    window = _small_window(tmp_path)
    try:
        window.show()
        window.set_page(window.PAGE_VIRTUAL)
        titles = [window.virtual_tabs.tabText(i) for i in range(window.virtual_tabs.count())]
        assert titles == ["1 Beam", "2 Errors", "3 Correct", "Captures", "Settings"]
        assert window.virtual_optimise_button.isHidden(), "direct run-and-apply duplicates review/apply"
        assert all(button.isHidden() for button in window.virtual_z_planner.findChildren(QPushButton))
        window._show_virtual_tab("correction")
        app.processEvents()
        # Find, review and apply all live on the one Correct tab.
        assert window.virtual_preview_button.isVisible()
        assert window.virtual_apply_button.isVisible()
        assert window.correct_table.isVisible()
        assert window.virtual_tab_indices["results"] == window.virtual_tab_indices["correction"]
    finally:
        window.close()
        app.processEvents()


def test_guided_correction_review_discard_and_next_iteration(tmp_path) -> None:
    """The proposal table, Discard and Next iteration keep the loop legible."""

    import copy
    from types import SimpleNamespace

    app = _app()
    window = _small_window(tmp_path)
    try:
        window._ensure_virtual_mode()
        window._ensure_virtual_cast()
        baseline = window.store.snapshot()
        candidate = copy.deepcopy(baseline)
        candidate.slm2.phase.z31_cos_amp_waves = 0.3
        trial = SimpleNamespace(
            initial_objective=1.66, final_objective=1.50, improvement_fraction=0.095, cancelled=False,
            accepted_runs=[
                {"slm": "SLM2", "parameter": "coma_x", "accepted": True, "accepted_command": 0.3},
                {"slm": "SLM2", "parameter": "coma_y", "accepted": False, "accepted_command": 0.0},
            ],
        )
        result = {"kind": "correction_preview", "result": trial, "candidate": candidate,
                  "baseline_signature": window._preview_signature(baseline), "z_plan": (31.0, 45.0),
                  "before_metrics": None, "original_z": 31.0}
        window._virtual_task_finished(result)
        app.processEvents()
        assert window.correct_table.rowCount() == 2
        assert window.correct_table.item(0, 2).text() == "+0.3"
        assert window.correct_table.item(1, 2).text() == "no change"
        assert "better" in window.correct_score.text()
        assert window.virtual_apply_button.isEnabled() and window.correct_discard.isEnabled()

        window.correct_discard.click()
        assert not window.virtual_apply_button.isEnabled()
        assert window.correct_table.rowCount() == 0
        assert window.correct_history.item(0, 4).text() == "Discarded"
        assert window.store.snapshot().slm2.phase.z31_cos_amp_waves == 0.0

        window.correct_next.click()
        assert "iteration 2" in window.correct_verdict.text()
    finally:
        window.close()
        app.processEvents()

def test_virtual_live_pauses_off_camera_pages_and_resumes_without_reset(tmp_path) -> None:
    app = _app()
    window = _small_window(tmp_path)
    try:
        window.set_page(window.PAGE_VIRTUAL)
        window.virtual_camera_z.setValue(37.0)
        window.virtual_trial_name.setText("keep_this_trial_name")
        window._apply_virtual_vortex_route()
        window._start_virtual_live(recast=True)
        deadline = time.monotonic() + 30
        while window.current_frame is None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert window.current_frame is not None
        first_frame_id = window.current_frame.frame_id
        cast_hash = window.resolution_engine.cast_hash("SLM1")

        window.set_page(window.PAGE_SLM1)
        deadline = time.monotonic() + 10
        while window._camera_thread is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert window._camera_thread is None
        assert window._resume_virtual_live_after_navigation
        assert window.virtual_camera_view.frame is None
        assert window.virtual_camera_z.value() == 37.0
        assert window.virtual_trial_name.text() == "keep_this_trial_name"
        assert window.resolution_engine.cast_hash("SLM1") == cast_hash
        assert window.store.snapshot().slm1.phase.vortex_charge == 20

        window.set_page(window.PAGE_VIRTUAL)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            app.processEvents()
            if window.current_frame is not None and window.current_frame.frame_id != first_frame_id:
                break
            time.sleep(0.01)
        assert window.current_frame.frame_id != first_frame_id
        assert window._camera_thread is not None
        assert not window._resume_virtual_live_after_navigation

        assert window.stop_live()
        window.set_page(window.PAGE_SLM2)
        window.set_page(window.PAGE_VIRTUAL)
        app.processEvents()
        assert window._camera_thread is None, "explicit Stop must not auto-resume"
    finally:
        window.close()
        app.processEvents()


def test_virtual_correction_preview_is_delivered_on_gui_thread(tmp_path) -> None:
    app = _app()
    window = _small_window(tmp_path)
    delivered_on_gui_thread = []
    window._on_frame = lambda _frame: delivered_on_gui_thread.append(
        QThread.currentThread() == window.thread()
    )
    try:
        window.set_page(window.PAGE_VIRTUAL)

        def task(**callbacks):
            callbacks["frame_callback"](object())
            return "preview complete"

        window._start_virtual_task(task)
        deadline = time.monotonic() + 10
        while window._virtual_thread is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert delivered_on_gui_thread == [True]
        assert window._virtual_thread is None
    finally:
        window.close()
        app.processEvents()


def test_live_frames_fluctuate_like_a_real_detector(tmp_path) -> None:
    app = _app()
    window = _small_window(tmp_path)
    try:
        assert window.virtual_realistic_camera.isChecked()
        window._apply_virtual_vortex_route()
        window._capture_virtual_current()
        app.processEvents()
        first = np.array(window.current_frame.data)
        window._capture_virtual_current()
        app.processEvents()
        second = np.array(window.current_frame.data)
        assert not np.array_equal(first, second), "identical frames are not a believable live view"
        spread = abs(second.sum() - first.sum()) / max(first.sum(), 1.0)
        assert spread < 0.05, "fluctuation should be small, not a different beam"
    finally:
        window.close()
        app.processEvents()


def test_mask_previews_report_whether_an_edit_has_been_cast(tmp_path) -> None:
    app = _app()
    window = _small_window(tmp_path)
    try:
        window._apply_virtual_vortex_route()
        app.processEvents()
        _, status = window.virtual_mask_previews["SLM1"]
        assert "never cast" in status.text()
        window._ensure_virtual_mode()
        window._ensure_virtual_cast()
        window._refresh_state(window.store.snapshot())
        assert "cast on the virtual bench" in status.text()
        window.store.set_vortex("SLM1", True, 12, source="test")
        window.controller.generate()
        window._refresh_state(window.store.snapshot())
        assert "EDITED SINCE CAST" in status.text()
    finally:
        window.close()
        app.processEvents()


def test_vortex_route_is_configure_then_cast_and_camera_uses_selected_z(tmp_path) -> None:
    app = _app()
    window = _small_window(tmp_path)
    errors = []
    window._show_error = lambda title, exc: errors.append((title, str(exc)))
    try:
        window.virtual_vortex_route.setCurrentText("SLM1 only")
        window.virtual_route_charge.setValue(20)
        window._apply_virtual_vortex_route()
        assert not errors
        assert window.mode_controller.operating_mode is OperatingMode.VIRTUAL_LAB
        assert window.store.snapshot().active_vortex_contributions() == {"SLM1": 20}
        assert window.resolution_engine.cast_hash("SLM1") is None
        assert window.resolution_engine.cast_vortex_charge() == 0

        window.virtual_camera_z.setValue(35.0)
        window._capture_virtual_current()
        app.processEvents()
        assert not errors
        assert window.current_frame is not None
        assert window.current_frame.z_mm == 35.0
        assert window.current_frame.data_kind == "SYNTHETIC"
        assert window.current_frame.metadata["virtual_power_scale"] != 1.0
        assert window.resolution_engine.cast_hash("SLM1") is not None
        assert window.resolution_engine.cast_hash("SLM2") is not None
        assert window.resolution_engine.cast_vortex_charge() == 20
    finally:
        window.close()
        app.processEvents()


def test_preview_restores_masks_and_never_silently_applies_proposal(tmp_path) -> None:
    app = _app()
    window = _small_window(tmp_path)
    errors = []
    window._show_error = lambda title, exc: errors.append((title, str(exc)))
    window._virtual_task_failed = lambda message: errors.append(("task", message))
    try:
        window._apply_virtual_vortex_route()
        window.virtual_z_plan.setText("31, 32")
        window.virtual_target.setCurrentText("SLM2")
        window.virtual_passes.setValue(1)
        for key, checkbox in window.correction_checks.items():
            checkbox.setChecked(key == "defocus")
        before = window.store.snapshot()
        window._preview_virtual_correction()
        deadline = time.monotonic() + 90.0
        while window._virtual_thread is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()
        assert window._virtual_thread is None, "Preview worker did not finish"
        assert not errors
        after = window.store.snapshot()
        assert after.slm1.phase.as_dict() == before.slm1.phase.as_dict()
        assert after.slm2.phase.as_dict() == before.slm2.phase.as_dict()
        assert after.slm1.last_cast_sha256 == window.resolution_engine.cast_hash("SLM1")
        assert after.slm2.last_cast_sha256 == window.resolution_engine.cast_hash("SLM2")
        assert "NOT APPLIED" in window.virtual_results.toPlainText()
        assert len(window._virtual_correction_history) == 1
        assert "#1" in window.virtual_iteration_history.toPlainText()
    finally:
        window.close()
        app.processEvents()


def test_operator_can_save_repeated_frames_at_chosen_z(tmp_path, monkeypatch) -> None:
    app = _app()
    window = _small_window(tmp_path)
    errors = []
    window._show_error = lambda title, exc: errors.append((title, str(exc)))
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.Ok)
    try:
        window.session_path.setText(str(tmp_path / "session"))
        window.virtual_camera_z.setValue(33.0)
        window.virtual_formal_repeats.setValue(2)
        window.virtual_trial_name.setText("z33_repeats_2")
        window._save_virtual_capture_at_z()
        assert not errors
        assert len(window._history) == 1
        assert window._history[0]["z_mm"] == 33.0
        root = tmp_path / "session" / "trials" / "z33_repeats_2" / "camera"
        files = sorted(root.glob("frame_*.npy"))
        assert len(files) == 2
        assert all(np.load(path, allow_pickle=False).shape == window.resolution_engine.shape_yx for path in files)
    finally:
        window.close()
        app.processEvents()


def test_reviewed_proposal_requires_explicit_apply_and_recasts(tmp_path) -> None:
    app = _app()
    window = _small_window(tmp_path)
    errors = []
    window._show_error = lambda title, exc: errors.append((title, str(exc)))
    try:
        window._ensure_virtual_mode()
        window._ensure_virtual_cast()
        baseline = window.store.snapshot()
        candidate = copy.deepcopy(baseline)
        candidate.slm2.phase.switches.zernike_z40 = True
        candidate.slm2.phase.z31_cos_amp_waves = 0.12
        window._pending_virtual_proposal = {
            "candidate": candidate,
            "baseline_signature": window._preview_signature(baseline),
        }
        assert window.store.snapshot().slm2.phase.z31_cos_amp_waves == 0.0
        window._start_virtual_live = lambda *, recast: None
        window._apply_reviewed_virtual_correction()
        assert not errors
        after = window.store.snapshot()
        assert after.slm2.phase.z31_cos_amp_waves == 0.12
        assert after.slm2.last_cast_sha256 == window.resolution_engine.cast_hash("SLM2")
        assert window._pending_virtual_proposal is None
    finally:
        window.close()
        app.processEvents()


def test_input_radius_changes_propagated_camera_and_invalidates_preview(tmp_path) -> None:
    app = _app()
    window = _small_window(tmp_path)
    errors = []
    window._show_error = lambda title, exc: errors.append((title, str(exc)))
    try:
        window._apply_virtual_vortex_route()
        window.virtual_camera_z.setValue(31.0)
        window._capture_virtual_current()
        before = window.current_frame.data.copy()
        assert window.current_frame.metadata["canonical_beam_radius_mm"] == 2.0
        window._pending_virtual_proposal = {"stale": True}
        window.virtual_apply_button.setEnabled(True)
        window.perturb_beam_radius_mm.setValue(1.0)
        assert window.resolution_engine.canonical_beam_radius_mm == 2.0
        window._apply_input_beam_radius()
        assert window.resolution_engine.canonical_beam_radius_mm == 1.0
        assert window._pending_virtual_proposal is None
        assert not window.virtual_apply_button.isEnabled()
        window._capture_virtual_current()
        assert window.current_frame.metadata["canonical_beam_radius_mm"] == 1.0
        assert not np.allclose(before, window.current_frame.data)
        assert not errors
    finally:
        window.close()
        app.processEvents()


def test_strong_fault_example_waits_for_apply_and_camera_overlay_is_optional(tmp_path) -> None:
    app = _app()
    window = _small_window(tmp_path)
    try:
        original_hash = window.resolution_engine.scenario.truth_hash
        window._load_strong_manual_perturbations()
        assert window.resolution_engine.scenario.truth_hash == original_hash
        assert window.perturb_coma_x.value() == 0.65
        window._apply_manual_perturbations()
        assert window.resolution_engine.scenario.truth_hash != original_hash
        assert window.virtual_overlay_ring.isChecked()
        window.virtual_overlay_ring.setChecked(False)
        assert not window.virtual_overlay_ring.isChecked()
        window._setup_quick_correction()
        assert window.virtual_target.currentText() == "SLM2"
        assert window.virtual_passes.value() == 1
        assert set(window._selected_correction_parameters()) == {"coma_x", "coma_y"}
        assert window._expanded_z_plan() == (31.0, 45.0)
    finally:
        window.close()
        app.processEvents()


def test_axicon_preflight_rejects_unmodellable_values_and_keeps_previous_axicon(tmp_path) -> None:
    app = _app()
    window = _small_window(tmp_path)
    errors = []
    window._show_error = lambda title, exc: errors.append((title, str(exc)))
    try:
        engine = window.resolution_engine
        # A literal 20° base angle is twice as steep as the measured bench optic
        # and would need more than the 4096² axicon grid over the model window.
        report = window._axicon_angle_report(20.0, grid_n=1024)
        assert not report["valid"]
        assert "NOT APPLIED" in report["message"]

        window.virtual_axicon_mode.setCurrentText(window.AXICON_ANGLE)
        old = engine.geometry
        window.virtual_axicon_angle.setValue(20.0)
        window._apply_virtual_geometry()
        assert engine.geometry == old
        assert "NOT APPLIED" in errors[-1][1]

        window.virtual_axicon_angle.setValue(45.0)
        window._apply_virtual_geometry()
        assert engine.geometry == old
        assert "Snell branch" in errors[-1][1]

        window.virtual_axicon_angle.setValue(0.5)
        window._apply_virtual_geometry()
        assert engine.geometry.axicon_k_perp_m_inv is None
        assert engine.geometry.axicon_model_base_angle_deg == 0.5
        assert window._pending_virtual_proposal is None
    finally:
        window.close()
        app.processEvents()


def test_bench_axicon_uses_measured_k_perp_and_true_beamage_pixels() -> None:
    """The bench optic, sampled so its intensity fringes are resolved, seen by 5.5 µm pixels."""

    from labcontrol.axicon_propagation import MEASURED_BENCH_AXICON_K_PERP_M_INV
    engine = ResolutionAwareVirtualBenchEngine(output_resolution=VirtualOutputResolution.LIVE_1M)
    geometry = engine.geometry
    assert geometry.axicon_k_perp_m_inv == MEASURED_BENCH_AXICON_K_PERP_M_INV
    assert geometry.simulation_window_mm == pytest.approx(2048 * 5.5e-3)  # Beamage-4M sensor
    assert engine.pixel_size_um == pytest.approx(5.5)
    plan = engine.axicon_plan()
    assert plan.valid
    assert 9.0 < plan.base_angle_deg < 10.5  # label "20°" is not the base angle
    assert plan.samples_per_period >= 4.0  # two samples per intensity fringe
    assert plan.propagation_grid_n % engine.detector_n == 0
    assert plan.propagation_grid_n == 4096

def test_resolved_axicon_stage_matches_the_reference_digital_twin_helpers() -> None:
    """The lean single-precision stage must reproduce the vbb_study reference route."""

    import math

    from labcontrol.axicon_propagation import (
        base_angle_deg_from_k_perp,
        build_resolved_axicon_spectrum,
        fourier_resample_fixed_window,
        resolved_field_at_z,
    )
    from vbb_study.digital_twin.vortex_continuous_propagation import build_fixed_support_spectrum, native_field_at_z
    from vbb_study.digital_twin.vortex_system_route import AxiconError, physical_axicon_on_own_plane
    from vbb_study.equations.fields import make_xy_grid

    lam, window, relay, fine, k_perp = 1.03e-6, 6e-3, 128, 512, 1.2e5
    decentre, z = (70e-6, -40e-6), 0.02
    x = (np.arange(relay) - relay / 2 + 0.5) * window / relay
    X, Y = np.meshgrid(x, x)
    field = np.exp(-(X**2 + Y**2) / (1.1e-3) ** 2) * np.exp(5j * np.arctan2(Y, X))
    grid = make_xy_grid(fine, window / fine)
    base = base_angle_deg_from_k_perp(k_perp, wavelength_m=lam, refractive_index=1.458, external_index=1.0)
    transmission, meta = physical_axicon_on_own_plane(
        grid, wavelength_m=lam, base_angle_rad=math.radians(base), refractive_index=1.458,
        external_index=1.0, error=AxiconError(decentre_m=decentre),
    )
    reference = native_field_at_z(
        build_fixed_support_spectrum(
            fourier_resample_fixed_window(field, fine) * transmission, grid,
            wavelength_m=lam, z_max_m=0.04, minimum_retained_spectral_power=0.98,
        ),
        z,
    )
    fast = resolved_field_at_z(
        build_resolved_axicon_spectrum(
            field, window_m=window, fine_n=fine, wavelength_m=lam, k_perp_m_inv=meta["exact_kr_m_inv"],
            decentre_m=decentre, z_support_m=0.04,
        ),
        z,
    )
    ref_i, fast_i = np.abs(reference) ** 2, np.abs(fast) ** 2
    assert np.corrcoef(ref_i.ravel(), fast_i.ravel())[0, 1] > 0.99999
    assert abs(ref_i.sum() - fast_i.sum()) / ref_i.sum() < 1e-5

def test_hand_edited_slm_mask_reaches_the_live_virtual_camera(tmp_path) -> None:
    """Editing a mask by hand, then casting, must change what the camera shows."""

    app = _app()
    window = _small_window(tmp_path)
    errors = []
    window._show_error = lambda title, exc: errors.append((title, str(exc)))
    try:
        window._apply_virtual_vortex_route()
        window.virtual_camera_z.setValue(31.0)
        window._start_virtual_live(recast=True)
        deadline = time.monotonic() + 60
        while window.current_frame is None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        before = np.array(window.current_frame.data)
        before_id = window.current_frame.frame_id

        # Edit SLM2 through the same full editor the operator uses.
        detail = window.slm_detail_views["SLM2"]
        detail.panel.grp_zernike.setChecked(True)
        detail.panel.z20_amp.setValue(0.45)
        app.processEvents()
        assert window.store.snapshot().slm2.phase.z20_amp_waves == 0.45

        # An uncast edit must not silently change the optical field.
        _, status = window.virtual_mask_previews["SLM2"]
        window._refresh_state(window.store.snapshot())
        assert "EDITED SINCE CAST" in status.text()

        window._start_virtual_live(recast=True)
        deadline = time.monotonic() + 60
        after = before
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
            if window.current_frame is not None and window.current_frame.frame_id != before_id:
                after = np.array(window.current_frame.data)
                break
        assert not np.allclose(before, after), "cast mask edit never reached the virtual camera"
        assert not errors
    finally:
        window.stop_live()
        window.close()
        app.processEvents()


def _axicon_window(tmp_path) -> VirtualLabResolutionWindow:
    """Small bench with a real axicon stage: 64 camera pixels, 256 native samples."""

    state = ExperimentState.from_app_config(AppConfig())
    store = ExperimentStore(state)
    engine = ResolutionAwareVirtualBenchEngine(
        VirtualLabGeometry(
            preview_grid_n=48, validation_grid_n=64, axicon_k_perp_m_inv=30000.0, max_propagation_grid_n=256
        ),
        beamage_n=64,
        maximum_grid_n=128,
    )
    controller = ModeAwareLabController(
        store,
        tmp_path,
        camera_provider=DummyCameraProvider(store.snapshot, shape_yx=(48, 48)),
        virtual_engine=engine,
    )
    return VirtualLabResolutionWindow(store=store, controller=controller)


def test_native_view_is_what_the_camera_pixels_integrate_and_toggles_in_place(tmp_path) -> None:
    app = _app()
    window = _axicon_window(tmp_path)
    errors: list[str] = []
    window._show_error = lambda title, exc: errors.append(f"{title}: {exc}")
    try:
        window.set_page(window.PAGE_VIRTUAL)
        window._apply_virtual_vortex_route()
        window._capture_virtual_current()
        app.processEvents()
        frame = window.current_frame
        assert frame is not None and frame.shape_yx == (64, 64)
        engine = window.resolution_engine
        camera, _metadata = engine._model_intensity(window.store.snapshot(), frame.z_mm)
        native, pixel_scale = engine.native_intensity(window.store.snapshot(), frame.z_mm)
        assert native.shape == (256, 256) and pixel_scale == pytest.approx(0.25)
        # The camera is exactly the native model averaged over each 5.5 µm-equivalent pixel.
        binned = native.reshape(64, 4, 64, 4).mean(axis=(1, 3))
        np.testing.assert_allclose(binned, camera, rtol=1e-5, atol=1e-6 * float(camera.max()))
        assert engine.native_intensity(window.store.snapshot(), frame.z_mm)[0] is native  # cached

        view = window.virtual_camera_view
        window.virtual_native_toggle.setChecked(True)
        app.processEvents()
        assert window.home_native_toggle.isChecked(), "one switch shown in two places"
        assert window.virtual_native_toggle.text() == window.NATIVE_ON_TEXT
        assert view._pixel_scale == pytest.approx(0.25)
        assert view._display_rgb.shape[:2] == (256, 256)
        assert view.scene().sceneRect().getRect() == (0.0, 0.0, 64.0, 64.0), "same zoom and overlays"
        assert window.virtual_frame_chip.text().startswith("NATIVE")
        assert np.array_equal(window.current_frame.data, frame.data), "stored data stays the camera frame"

        window.home_native_toggle.setChecked(False)
        app.processEvents()
        assert not window.virtual_native_toggle.isChecked()
        assert not engine.keep_native_intensity
        assert view._pixel_scale == 1.0 and view._display_rgb.shape[:2] == (64, 64)
        assert view.scene().sceneRect().getRect() == (0.0, 0.0, 64.0, 64.0)
        assert errors == []
    finally:
        window.close()
        app.processEvents()


def test_native_toggle_is_only_offered_for_the_virtual_lab(tmp_path) -> None:
    app = _app()
    window = _small_window(tmp_path)
    try:
        window._apply_virtual_vortex_route()  # a cast selects the Virtual Lab
        app.processEvents()
        assert window.virtual_native_toggle.isEnabled() and window.home_native_toggle.isEnabled()
        window.virtual_native_toggle.setChecked(True)
        window.mode_choice.setCurrentText(OperatingMode.RECORDED_LAB.value)
        app.processEvents()
        assert not window.virtual_native_toggle.isEnabled()
        assert not window.home_native_toggle.isEnabled()
        assert not window.virtual_native_toggle.isChecked()
        assert not window.resolution_engine.keep_native_intensity
    finally:
        window.close()
        app.processEvents()


def test_thin_ring_is_not_clipped_by_the_display_level() -> None:
    from labcontrol.ui.image_view import _colour_lut, render_preview

    y, x = np.mgrid[:2048, :2048]
    radius = np.hypot(x - 1024.0, y - 1024.0)
    raw = (4000.0 * np.exp(-0.5 * ((radius - 8.0) / 0.9) ** 2)).astype(np.float32)  # q=20 ring, ~2 px wide
    raw += np.float32(40.0)
    top = _colour_lut("inferno")[-1]
    for scale in ("percentile", "full range"):
        preview = render_preview(raw, colour="inferno", scale=scale)
        saturated = np.all(preview == top, axis=2)
        # Before: 0.05 % of a 4M sensor (~2000 px) was allowed to clip, i.e. the whole ring.
        assert int(np.count_nonzero(saturated)) <= 64, scale  # symmetric ties and LUT rounding
        # Clipping is limited to the ring crest; one pixel off the crest is never drawn white.
        assert not saturated[np.abs(radius - 8.0) > 1.0].any(), scale
