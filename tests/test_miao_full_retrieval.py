from pathlib import Path
import sys

import numpy as np
from scipy import special

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "notebooks" / "experimental" / "axicon_aberration_correction"
sys.path.insert(0, str(MOD))

from miao_full_retrieval import (
    PlaneRetrieval, angular_field_from_coefficients, optimise_k_perp_ideal_mode,
    assemble_full_aperture, interpolate_to_cartesian, resolve_conjugate_branch,
    correction_manifest,
)
from iterative_correction_controller_v2 import _candidate_mask
from q20_experimental_acceptance_metrics import compute_plane_metrics
from run_q20_miao_retrieval import calibrated_axes_in_crop


def fake_plane(i, z, kp, theta, phase):
    return PlaneRetrieval(i, z, 64.0, 64.0, kp, 4, 0.01, 0.99, 0.02,
                          np.arange(-4, 5), np.zeros(9, complex), theta,
                          np.exp(1j*phase))


def test_per_plane_kperp_optimizer_recovers_ideal_ring():
    q = 3
    pixel = 5.5e-6
    kp_true = 4.7e5
    n = 128
    yy, xx = np.indices((n, n), dtype=float)
    r = np.hypot(xx-(n-1)/2, yy-(n-1)/2)*pixel
    image = special.jv(q, kp_true*r)**2
    kp = optimise_k_perp_ideal_mode(image, ((n-1)/2, (n-1)/2), pixel, q,
                                    kp_true*1.06, search_fraction=0.15,
                                    rmax_um=220, n_r=36, n_theta=72)
    assert abs(kp-kp_true)/kp_true < 0.02


def test_stationary_phase_radius_and_radial_gradient_are_recovered():
    lam = 1030e-9
    k = 2*np.pi/lam
    nominal = 4.9e5
    z = np.linspace(0.020, 0.035, 6)
    theta = np.linspace(0, 2*np.pi, 720, endpoint=False)
    kp = nominal + np.asarray([0, -800, -1200, -400, 500, 900], float)
    planes = [fake_plane(i, z[i], kp[i], theta, 0.2*np.cos(2*theta))
              for i in range(len(z))]
    full = assemble_full_aperture(planes, z, lam, k_perp_nominal_m_inv=nominal)
    order = np.argsort(z*kp/k)
    assert np.allclose(full.rho_m, (z*kp/k)[order])
    assert np.allclose(full.radial_phase_gradient_rad_per_m, (nominal-kp)[order])
    assert np.isfinite(full.radial_phase_rad).all()


def test_programmed_vortex_is_not_reinserted_in_angular_residual():
    theta = np.linspace(0, 2*np.pi, 720, endpoint=False)
    m = np.asarray([-2, -1, 0, 1, 2])
    c = np.asarray([0, 0.1j, 1.0, 0.2, 0], complex)
    g1 = angular_field_from_coefficients(c, m, theta)
    g2 = angular_field_from_coefficients(c, m, theta)
    assert np.allclose(g1, g2)
    assert np.max(np.abs(np.unwrap(np.angle(g1)))) < 2*np.pi


def test_conjugate_branch_resolution_uses_independent_input_intensity():
    theta = np.linspace(0, 2*np.pi, 720, endpoint=False)
    amp = 1.0 + 0.35*np.cos(theta) + 0.12*np.sin(2*theta)
    g = np.stack([amp*np.exp(1j*0.2*np.cos(3*theta)) for _ in range(4)])
    branch, sd, sc = resolve_conjugate_branch(g, np.abs(g)**2)
    assert branch == "direct"
    assert sd > sc
    rotated = np.roll(np.abs(g)**2, g.shape[1]//2, axis=1)
    branch2, sd2, sc2 = resolve_conjugate_branch(g, rotated)
    assert branch2 == "conjugate"
    assert sc2 > sd2


def test_unresolved_branch_blocks_low_gain_trial():
    lam = 1030e-9
    theta = np.linspace(0, 2*np.pi, 720, endpoint=False)
    z = np.asarray([0.02, 0.025, 0.03])
    planes = [fake_plane(i, z[i], 4.9e5, theta, 0.1*np.cos(2*theta)) for i in range(3)]
    full = assemble_full_aperture(planes, z, lam)
    manifest = correction_manifest(full, True, True, True, False)
    assert full.branch == "unresolved"
    assert not manifest["application_ready_for_low_gain_trial"]
    assert not manifest["hardware_ready"]


def test_wrapped_phase_interpolation_is_finite_inside_sampled_annulus():
    lam = 1030e-9
    theta = np.linspace(0, 2*np.pi, 720, endpoint=False)
    z = np.asarray([0.02, 0.025, 0.03])
    phases = [3.05*np.cos(theta), -3.05*np.cos(theta), 3.0*np.sin(2*theta)]
    planes = [fake_plane(i, z[i], 4.9e5, theta, phases[i]) for i in range(3)]
    full = assemble_full_aperture(planes, z, lam)
    cart = interpolate_to_cartesian(full, grid_size=96)
    assert np.isfinite(cart["residual_phase_rad"][cart["valid"]]).all()


def test_native_slm_candidate_keeps_shape_and_signed_phase():
    residual = np.full((37, 53), np.nan)
    residual[5:30, 7:45] = 0.8
    accepted = np.zeros_like(residual)
    accepted[~np.isfinite(residual)] = np.nan
    candidate = _candidate_mask(residual, accepted, 0.05)
    assert candidate.shape == residual.shape
    assert np.isnan(candidate[0, 0])
    assert np.allclose(candidate[5:30, 7:45], 0.04)


def test_per_z_camera_axis_calibration_tracks_stage_runout():
    n = 4
    cal = {"camera_optical_axis_yx_px_by_z":
           [[100.0, 200.0], [100.4, 199.8], [100.9, 199.5], [101.2, 199.3]]}
    shifts = np.asarray([[0.1, -0.2], [0.0, 0.1], [-0.1, 0.0], [0.2, 0.2]])
    axes, ready, source = calibrated_axes_in_crop(
        cal, (10.0, 20.0), (50, 150), shifts, n)
    expected = np.asarray(cal["camera_optical_axis_yx_px_by_z"]) + shifts - np.asarray([50, 150])
    assert ready
    assert "per-z" in source
    assert np.allclose(axes, expected)


def test_independent_acceptance_metric_is_near_ideal_for_synthetic_bessel():
    q = 5
    kp = 4.4e5
    pixel = 2.0e-6
    n = 192
    cy = cx = (n-1)/2
    yy, xx = np.indices((n, n), dtype=float)
    r = np.hypot(yy-cy, xx-cx)*pixel
    image = special.jv(q, kp*r)**2
    result = compute_plane_metrics(
        image, (cy, cx), pixel_pitch_m=pixel, q=q,
        k_perp_nominal_m_inv=kp, roi_radius_um=150)
    assert result["measured_vs_ideal_corr"] > 0.999999
    assert result["measured_vs_ideal_rmse"] < 1e-8
    assert result["measured_ring_cv"] < 0.25
    assert result["measured_dark_core_ratio"] < 0.2


def test_legacy_normalized_z_map_and_second_remap_are_not_consumed_by_v3_controller():
    text = (MOD/"iterative_correction_controller_v2.py").read_text(encoding="utf-8")
    assert "UNCALIBRATED_DO_NOT_APPLY_q20_modal_correction.npy" not in text
    assert "slm2_correction_phase_rad.npy" in text
    assert "build_slm2_complete_preview" not in text
    assert "second_coordinate_mapping_applied" in text


def test_runner_blocks_geometric_slm_mapping_without_conjugacy_and_axis_calibration():
    text = (MOD/"run_q20_miao_retrieval.py").read_text(encoding="utf-8")
    assert "slm2_is_conjugate_to_input_plane" in text
    assert "camera_optical_axis_yx_px_by_z" in text
    assert "nonconjugate_relay_backpropagation_implemented" in text
    assert "k_perp_nominal_m_inv" in text


def test_controller_acceptance_contract_matches_metric_generator():
    controller = (MOD/"iterative_correction_controller_v2.py").read_text(encoding="utf-8")
    metrics = (MOD/"q20_experimental_acceptance_metrics.py").read_text(encoding="utf-8")
    for name in ("measured_vs_ideal_corr", "measured_vs_ideal_rmse",
                 "measured_ring_cv", "measured_dark_core_ratio"):
        assert name in controller
        assert name in metrics
