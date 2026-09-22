"""The bench 4F relay, its Fourier stop, and the objective route to tiny beams."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from labcontrol.devices.camera import DummyCameraProvider
from labcontrol.mode_controller import ModeAwareLabController
from labcontrol.relay_4f import (
    apply_fourier_plane_aperture,
    fourier_aperture_report,
    fourier_plane_cutoff_cycles_per_m,
    rotate_180,
)
from labcontrol.sample_plane import (
    bench_ring_radius_um,
    demagnification_for_ring,
    sample_plane_scale,
)
from labcontrol.state import ExperimentState, ExperimentStore
from labcontrol.ui.advanced import ADVANCED_QSS
from labcontrol.ui.virtual_advanced_v3 import VirtualLabResolutionWindow
from labcontrol.virtual_lab import VirtualLabGeometry
from labcontrol.virtual_lab_experiments import ManualPerturbationSpec
from labcontrol.virtual_lab_resolution import ResolutionAwareVirtualBenchEngine
from slm_lab_control.config import AppConfig
from slm_lab_control.ui.style import APP_QSS


def _app() -> QApplication:
    app = QApplication.instance() or QApplication([])
    if not app.styleSheet():
        app.setStyleSheet(APP_QSS + ADVANCED_QSS)
    return app


def _toy_engine(base_angle_deg: float = 0.5, **geometry) -> ResolutionAwareVirtualBenchEngine:
    """A cheap axicon bench: the relay, not the 4096-square bench optic."""

    engine = ResolutionAwareVirtualBenchEngine(
        VirtualLabGeometry(
            preview_grid_n=256,
            validation_grid_n=256,
            axicon_k_perp_m_inv=None,
            axicon_model_base_angle_deg=base_angle_deg,
            max_propagation_grid_n=512,
            **geometry,
        ),
        beamage_n=256,
        maximum_grid_n=512,
    )
    flat = np.zeros((256, 256))
    engine.set_cast_phase("SLM1", flat, flat)
    engine.set_cast_phase("SLM2", flat, flat)
    return engine


def _centroid_x_px(engine, state) -> float:
    intensity = np.asarray(engine._model_intensity(state, 25.0)[0], dtype=float)
    weight = intensity - intensity.min()
    x = np.indices(intensity.shape)[1]
    return float((x * weight).sum() / weight.sum()) - intensity.shape[1] / 2 + 0.5


def test_bench_4f_is_the_reported_symmetric_relay() -> None:
    """SLM-300-L1-300-stop-300-L2-300-axicon: 1:1, inverting, stop in the shared focal plane."""

    geometry = VirtualLabGeometry()
    assert geometry.lens1_focal_length_mm == geometry.lens2_focal_length_mm == 300.0
    assert geometry.slm_to_lens1_mm == 300.0
    assert geometry.lens_to_lens_separation_mm == 600.0
    assert geometry.lens2_to_axicon_mm == 300.0
    assert geometry.relay_magnification == 1.0
    assert geometry.relay_inverts_image is True
    assert geometry.pinhole_axial_position_mm == 300.0
    assert geometry.relay_mode == "symmetric_4f_unit_magnification_order_select"
    # The stop diameter is still unmeasured, so no filtering is invented.
    assert geometry.fourier_aperture_diameter_mm is None
    notes = " ".join(geometry.public_summary()["known_fact_notes"])
    assert "symmetric 4F" in notes and "180 degrees" in notes


def test_unit_magnification_relay_lands_an_slm_decentre_on_the_far_side() -> None:
    """A 1:1 4F rotates the SLM plane by 180 degrees, so odd errors change sign."""

    state = ExperimentState.from_app_config(AppConfig())
    walk = {}
    for inverts in (False, True):
        engine = _toy_engine(relay_inverts_image=inverts)
        engine.apply_manual_perturbations(ManualPerturbationSpec(beam_decentre_x_um=300.0), seed=3)
        walk[inverts] = _centroid_x_px(engine, state)
    assert walk[False] > 1.0, "a decentred beam must move the camera spot"
    assert walk[True] == pytest.approx(-walk[False], rel=1e-3)


def test_rotate_180_is_exact_on_the_centred_grid() -> None:
    field = np.arange(16, dtype=float).reshape(4, 4) + 1j
    np.testing.assert_array_equal(rotate_180(rotate_180(field)), field)
    np.testing.assert_array_equal(rotate_180(field)[0, 0], field[-1, -1])


def test_fourier_stop_passes_the_selected_order_until_it_is_a_pinhole() -> None:
    """At f=300 mm a few mm passes everything; only a sub-millimetre stop filters."""

    cutoff = fourier_plane_cutoff_cycles_per_m(
        diameter_m=3e-3, wavelength_m=1030e-9, focal_length_m=0.3
    )
    assert cutoff == pytest.approx(0.5 * 3e-3 / (1030e-9 * 0.3))
    report = fourier_aperture_report(
        diameter_mm=3.0, wavelength_nm=1030.0, focal_length_mm=300.0, window_mm=11.264, grid_n=2048
    )
    assert report["cutoff_cycles_per_mm"] == pytest.approx(4.85, abs=0.05)
    assert report["smallest_relayed_feature_um"] == pytest.approx(103.0, abs=1.0)

    n = 512
    dx = 11.264e-3 / n
    x = (np.arange(n) - n / 2 + 0.5) * dx
    X, Y = np.meshgrid(x, x)
    beam = np.exp(-(X**2 + Y**2) / (2e-3) ** 2) * np.exp(1j * 20 * np.arctan2(Y, X))
    passed = {}
    for diameter_mm in (12.0, 3.0, 1.0):
        _filtered, meta = apply_fourier_plane_aperture(
            beam,
            dx_m=dx,
            wavelength_m=1030e-9,
            focal_length_m=0.3,
            diameter_m=diameter_mm * 1e-3,
        )
        passed[diameter_mm] = meta["fourier_aperture_power_fraction"]
    # A *pure phase* vortex winds q times around its core, so its spectrum
    # reaches well past a few cycles/mm.  Converged against grid 256..4096 and
    # reproduced by the engine's own frame metadata.
    assert passed[12.0] == pytest.approx(0.99, abs=0.01)
    assert passed[3.0] == pytest.approx(0.81, abs=0.01)
    assert passed[1.0] == pytest.approx(0.14, abs=0.01)

    flat = np.exp(-(X**2 + Y**2) / (2e-3) ** 2)
    _out, meta = apply_fourier_plane_aperture(
        flat, dx_m=dx, wavelength_m=1030e-9, focal_length_m=0.3, diameter_m=3e-3
    )
    assert meta["fourier_aperture_power_fraction"] > 0.9999, "a plain Gaussian passes a 3 mm stop untouched"


def test_sample_plane_scales_rings_by_m_and_length_by_m_squared() -> None:
    bench = sample_plane_scale(
        demagnification=1.0, camera_pixel_um=5.5, ring_radius_um=46.0, bessel_length_mm=25.0,
        k_perp_m_inv=482741.41419101483,
    )
    assert bench.ratio_label == "1:1" and bench.z_scale == 1.0
    assert bench.numerical_aperture == pytest.approx(0.0791, abs=1e-4)

    small = sample_plane_scale(
        demagnification=0.2, camera_pixel_um=5.5, native_pixel_um=2.75, ring_radius_um=46.0,
        bessel_length_mm=25.0, k_perp_m_inv=482741.41419101483,
    )
    assert small.ratio_label == "1:5"
    assert small.ring_radius_um == pytest.approx(46.0 * 0.2)
    assert small.bessel_length_um == pytest.approx(25.0e3 * 0.04)
    assert small.pixel_um == pytest.approx(1.1) and small.native_sample_um == pytest.approx(0.55)
    assert small.sample_z_um(10.0) == pytest.approx(10.0 * 0.04 * 1e3)
    assert small.bench_z_mm(small.sample_z_um(10.0)) == pytest.approx(10.0)
    assert small.valid and "paraxial" in small.message

    impossible = sample_plane_scale(
        demagnification=1 / 20, camera_pixel_um=5.5, k_perp_m_inv=482741.41419101483,
    )
    assert not impossible.valid and "no propagating cone" in impossible.message
    with pytest.raises(ValueError):
        sample_plane_scale(demagnification=2.0, camera_pixel_um=5.5)


def test_bench_ring_radius_follows_the_bessel_root() -> None:
    k_perp = 482741.41419101483
    assert bench_ring_radius_um(k_perp_m_inv=k_perp, charge=0) == pytest.approx(4.98, abs=0.02)
    assert bench_ring_radius_um(k_perp_m_inv=k_perp, charge=20) == pytest.approx(46.03, abs=0.05)
    m = demagnification_for_ring(bench_ring_radius_um=46.03, target_ring_radius_um=9.2)
    assert m == pytest.approx(0.2, abs=1e-3)
    assert demagnification_for_ring(bench_ring_radius_um=10.0, target_ring_radius_um=50.0) == 1.0


def test_engine_reports_the_sample_plane_for_the_bench_axicon() -> None:
    engine = ResolutionAwareVirtualBenchEngine()
    bench = engine.sample_plane(charge=20)
    assert bench.ring_radius_um == pytest.approx(46.0, abs=0.5)
    assert bench.pixel_um == pytest.approx(5.5)
    assert bench.native_sample_um == pytest.approx(2.75)
    fitted = engine.sample_plane(
        charge=20, geometry=dataclasses.replace(engine.geometry, objective_demagnification=0.25)
    )
    assert fitted.ring_radius_um == pytest.approx(46.0 * 0.25, abs=0.2)
    assert fitted.pixel_um == pytest.approx(1.375)


def test_gui_applies_a_possible_objective_and_refuses_an_impossible_one(tmp_path) -> None:
    app = _app()
    state = ExperimentState.from_app_config(AppConfig())
    store = ExperimentStore(state)
    engine = _toy_engine(base_angle_deg=1.0)
    controller = ModeAwareLabController(
        store,
        tmp_path,
        camera_provider=DummyCameraProvider(store.snapshot, shape_yx=(48, 48)),
        virtual_engine=engine,
    )
    window = VirtualLabResolutionWindow(store=store, controller=controller)
    errors: list[str] = []
    window._show_error = lambda title, exc: errors.append(str(exc))
    try:
        window.set_page(window.PAGE_VIRTUAL)
        window.virtual_route_charge.setValue(20)
        window.virtual_objective.setValue(3)
        app.processEvents()
        assert "1:3" in window.virtual_sample_status.text()
        window._apply_virtual_geometry()
        app.processEvents()
        assert engine.geometry.objective_demagnification == pytest.approx(1 / 3)
        assert errors == []

        window.virtual_objective.setValue(200)
        app.processEvents()
        window._apply_virtual_geometry()
        app.processEvents()
        assert errors and "no propagating cone" in errors[-1]
        assert engine.geometry.objective_demagnification == pytest.approx(1 / 3), "refused, so unchanged"
    finally:
        window.close()
        app.processEvents()

def test_auto_expose_fits_a_bright_plane_and_lifts_a_faint_one(tmp_path) -> None:
    """The bench answer to clipping is the exposure, not a rescaled display."""

    app = _app()
    state = ExperimentState.from_app_config(AppConfig())
    store = ExperimentStore(state)
    engine = _toy_engine()
    controller = ModeAwareLabController(
        store,
        tmp_path,
        camera_provider=DummyCameraProvider(store.snapshot, shape_yx=(48, 48)),
        virtual_engine=engine,
    )
    window = VirtualLabResolutionWindow(store=store, controller=controller)
    errors: list[str] = []
    window._show_error = lambda title, exc: errors.append(str(exc))
    try:
        window.set_page(window.PAGE_VIRTUAL)
        window._apply_virtual_vortex_route()
        window._capture_virtual_current()
        app.processEvents()
        peak = engine.peak_model_intensity(store.snapshot(), window.virtual_camera_z.value())
        assert peak > 0.0

        # Start far too bright, then far too dim: both land near the target.
        for exposure in (50_000.0, 5.0):
            window.exposure.setValue(exposure)
            window._configure_camera()
            window._auto_expose_virtual()
            app.processEvents()
            counts = float(np.asarray(window.current_frame.data).max())
            full_scale = float(window.current_frame.full_scale)
            assert counts == pytest.approx(window.AUTO_EXPOSE_TARGET * full_scale, rel=0.15)
            assert not window.current_metrics.values.get("clipped")
            assert "Exposure" in window.virtual_bench_status.text()
        assert errors == []
    finally:
        window.close()
        app.processEvents()


def test_peak_model_intensity_is_the_noise_free_peak() -> None:
    """Exposure arithmetic must not read a clipped frame or chase detector noise."""

    state = ExperimentState.from_app_config(AppConfig())
    engine = _toy_engine()
    peak = engine.peak_model_intensity(state, 25.0)
    intensity, _metadata = engine._model_intensity(state, 25.0)
    assert peak == pytest.approx(float(np.asarray(intensity).max()))
    assert engine.peak_model_intensity(state, 25.0) == peak, "repeatable, unlike a noisy frame"
