from __future__ import annotations

import numpy as np

from vbb_study.short_bessel import ShortBesselDesignInput, design_short_bessel
from vbb_study.short_bessel_bench_candidate import BenchCandidateConfig, run_bench_candidate


def _design():
    return design_short_bessel(ShortBesselDesignInput(target_fwhm_um=5.0, target_length_um=500.0))


def test_canonical_20_pixel_carrier_predicts_about_1p93_mm_shift_for_f300() -> None:
    run = run_bench_candidate(
        _design(),
        ell=0,
        config=BenchCandidateConfig(
            grid_n=257,
            z_points=31,
            z_max_um=700.0,
            pinhole_radius_mm=0.25,
            lens_clear_radius_mm=7.0,
            measured_physical_only_fwhm_um=3.0,
        ),
    )
    assert np.isclose(abs(run.metrics.predicted_order_shift_mm), 1.929375, rtol=5e-4)
    assert abs(run.metrics.measured_order_peak_x_mm) > 1.0
    assert np.isclose(run.metrics.slm_active_width_mm, 15.36, rtol=0.0, atol=1e-12)
    assert np.isclose(run.metrics.slm_active_height_mm, 8.64, rtol=0.0, atol=1e-12)


def test_3um_physical_core_requires_cancelling_slm_radial_phase() -> None:
    run = run_bench_candidate(
        _design(),
        ell=0,
        config=BenchCandidateConfig(
            grid_n=257,
            z_points=31,
            pinhole_radius_mm=0.35,
            measured_physical_only_fwhm_um=3.0,
        ),
    )
    assert run.metrics.residual_slm_kr_sample_m_inv < 0.0
    assert run.metrics.residual_slm_kr_pre_m_inv < 0.0
    assert 0.0 < run.metrics.slm1_capture_fraction <= 1.0
    assert 0.0 < run.metrics.slm2_capture_fraction <= 1.0
    assert 0.0 <= run.metrics.pinhole_transmitted_fraction <= 1.0


def test_vortex_charge_magnitude_survives_candidate_route_at_generous_filter_radius() -> None:
    """The nominal route may reverse the reported winding sign by convention.

    Until relay/camera orientation is bench-calibrated, the robust topological
    invariant here is |winding|=|ell|, not the sign reported by this numerical
    coordinate convention.
    """
    design = _design()
    for ell in (1, 3):
        run = run_bench_candidate(
            design,
            ell=ell,
            config=BenchCandidateConfig(
                grid_n=257,
                z_points=41,
                pinhole_radius_mm=0.50,
                lens_clear_radius_mm=7.0,
                measured_physical_only_fwhm_um=3.0,
            ),
        )
        winding = run.metrics.measured_phase_winding
        assert np.isfinite(winding)
        assert abs(abs(winding) - abs(ell)) < 0.35
        assert run.metrics.measured_ring_diameter_um is not None
        assert run.metrics.measured_ring_diameter_um > 0.0


def test_candidate_route_keeps_calibration_boundary_explicit() -> None:
    run = run_bench_candidate(
        _design(),
        ell=0,
        config=BenchCandidateConfig(grid_n=257, z_points=31, pinhole_radius_mm=0.35),
    )
    text = run.metrics.claim_boundary.lower()
    assert "candidate feasibility" in text
    assert "uncalibrated" in text
    assert "measured physical-axicon-only fwhm" in text
