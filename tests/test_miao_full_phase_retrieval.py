from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORR = ROOT / "notebooks" / "experimental" / "axicon_aberration_correction"
if str(CORR) not in sys.path:
    sys.path.insert(0, str(CORR))

MODULE = CORR / "miao_full_phase_retrieval.py"
spec = importlib.util.spec_from_file_location("miao_full_phase_retrieval", MODULE)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

from modal_vortex_bessel import modal_basis


def _polar_from_field(q, m_values, coeffs, kperp, *, n_r=38, n_theta=64):
    radii_m = np.linspace(8e-6, 210e-6, n_r)
    theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    rr, tt = np.meshgrid(radii_m, theta, indexing="ij")
    B, _ = modal_basis(q, np.asarray(m_values), kperp, rr.ravel(), tt.ravel())
    intensity = np.abs(B @ np.asarray(coeffs))**2
    intensity = intensity.reshape(n_r, n_theta)
    intensity /= intensity.max()
    y = intensity.ravel()
    rflat = rr.ravel()
    w = rflat/rflat.max() * (0.25 + 0.75*np.sqrt(np.clip(y, 0, 1)))
    return {
        "center_y_px": 0.0,
        "center_x_px": 0.0,
        "core_score": 1.0,
        "ring_radius_px": 20.0,
        "radii_px": radii_m/5.5e-6,
        "theta": theta,
        "measured": intensity,
        "rflat_m": rflat,
        "phiflat_rad": tt.ravel(),
        "y": y,
        "weights": w,
    }


def test_variable_kperp_fit_recovers_synthetic_plane():
    q = 4
    k_true = 4.82e5
    m_values = np.array([-3, -2, 0, 2, 3])
    coeffs = np.array([0.05+0.02j, -0.08+0.04j, 1.0+0j, 0.11-0.03j, -0.04+0.06j])
    polar = _polar_from_field(q, m_values, coeffs, k_true)
    fit = mod.fit_polar_plane_variable_kperp(
        polar,
        plane_index=0,
        z_relative_m=0.0,
        q=q,
        k_seed_m_inv=4.65e5,
        search_fraction=0.10,
        coarse_points=7,
        m_max_search=3,
        m_max_final=3,
        include_first_order=False,
        search_maxiter=45,
        final_maxiter=100,
    )
    assert abs(fit.k_perp_opt_m_inv-k_true)/k_true < 0.008
    assert fit.correlation > 0.995


def _dummy_fit(i, z, kopt, coeffs, m_values):
    return mod.MiaoPlaneFit(
        plane_index=i,
        z_relative_m=z,
        center_y_px=0.0,
        center_x_px=0.0,
        core_score=1.0,
        ring_radius_px=20.0,
        k_seed_m_inv=kopt,
        k_perp_opt_m_inv=kopt,
        k_search_lo_m_inv=.9*kopt,
        k_search_hi_m_inv=1.1*kopt,
        m_values=np.asarray(m_values, int),
        coeffs=np.asarray(coeffs, complex),
        data_cost=0.0,
        correlation=1.0,
        nrmse=0.0,
        k_at_search_boundary=False,
    )


def test_radial_phase_is_recovered_from_kperp_variation():
    k0 = 4.9e5
    # Choose a known constant radial phase gradient.  Miao gives kopt=k0-grad
    # and rho=z*kopt/k0, so the recovered phase must be linear in rho.
    grad = 2200.0  # rad/m
    kopt = k0-grad
    z_abs = np.array([20, 24, 28, 32, 36], float)*1e-3
    z_rel = z_abs-z_abs[-1]
    m_values = np.array([-2, 0, 2])
    coeffs = np.array([0.08+0.02j, 1+0j, 0.05-0.03j])
    fits = [_dummy_fit(i, z, kopt, coeffs, m_values) for i, z in enumerate(z_rel)]
    wf = mod.reconstruct_transverse_wavefront(
        fits,
        q=8,
        z_at_relative_zero_from_reference_m=float(z_abs[-1]),
        nominal_k_tan_alpha_m_inv=k0,
        n_theta=128,
    )
    slope = np.polyfit(wf.rho_m, wf.radial_phase_rad, 1)[0]
    assert abs(slope-grad) < 1e-8*max(1.0, abs(grad))
    assert np.allclose(wf.radial_gradient_rad_per_m, grad)
    assert wf.nominal_cone_calibrated


def test_programmed_qtheta_is_not_reinserted_into_residual():
    k0 = 4.9e5
    z_abs = np.array([20, 25, 30], float)*1e-3
    z_rel = z_abs-z_abs[-1]
    # Pure ideal angular field: only m=0 coefficient.  Regardless of q, the
    # residual phase should be radially constant (global piston only), not q spokes.
    fits = [
        _dummy_fit(i, z, k0, np.array([1+0j]), np.array([0]))
        for i, z in enumerate(z_rel)
    ]
    wf = mod.reconstruct_transverse_wavefront(
        fits,
        q=20,
        z_at_relative_zero_from_reference_m=float(z_abs[-1]),
        nominal_k_tan_alpha_m_inv=k0,
        n_theta=256,
    )
    phase = np.angle(np.exp(1j*wf.residual_phase_rows_rad))
    assert np.max(np.abs(phase-phase[:, :1])) < 1e-12
    assert wf.metadata["programmed_qtheta_in_residual"] is False


def test_default_mode_set_blocks_unreliable_first_order_terms():
    modes = mod._mode_indices(8, include_first_order=False)
    assert -1 not in modes and 1 not in modes
    assert 0 in modes
    modes_with = mod._mode_indices(8, include_first_order=True)
    assert -1 in modes_with and 1 in modes_with


def test_cartesian_interpolation_uses_complex_field_and_keeps_hardware_blocked():
    k0 = 4.9e5
    z_abs = np.array([20, 25, 30], float)*1e-3
    z_rel = z_abs-z_abs[-1]
    m = np.array([-2, 0, 2])
    coeffs = np.array([0.12+0.03j, 1+0j, -0.05+0.04j])
    fits = [_dummy_fit(i, z, k0, coeffs, m) for i, z in enumerate(z_rel)]
    wf = mod.reconstruct_transverse_wavefront(
        fits,
        q=20,
        z_at_relative_zero_from_reference_m=float(z_abs[-1]),
        nominal_k_tan_alpha_m_inv=k0,
        n_theta=128,
    )
    cart = mod.assemble_cartesian_residual(wf, grid_size=128)
    assert np.isfinite(cart.residual_phase_rad[cart.valid_mask]).all()
    assert np.allclose(
        cart.correction_phase_rad[cart.valid_mask],
        -cart.residual_phase_rad[cart.valid_mask],
    )
    assert cart.metadata["hardware_ready"] is False
