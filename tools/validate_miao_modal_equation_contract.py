"""Direct contract test of the Miao et al. Eq. (3) modal inversion.

This test deliberately removes the digital twin and propagation engine.  A q=20
focal intensity is synthesized directly from the same stationary-phase modal
model used in Miao et al., Opt. Express 30, 11360-11371 (2022), Eq. (3):

    U(r,phi) = sum_n (-i)^n c_n J_n(k_perp r) exp(-i n phi)

with

    c_n^(q) = integral E_tilde(theta) exp[i(n+q)theta] dtheta.

The code parameterises m=n+q, so n=m-q and E_tilde(theta) is reconstructed as
sum_m c_m exp(-i m theta).  A small known amplitude asymmetry is included only
to resolve the intensity-only direct/conjugate ambiguity in the same way as the
paper: compare the retrieved input intensity with an independently known input
intensity profile.

If this test fails, the modal optimizer/indexing is wrong.  If it passes while a
full propagated-axicon test fails, the discrepancy is in the stationary-phase
forward-model assumptions or the coupling to the physical route, not Eq. (3)
itself.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
from scipy import special

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "notebooks" / "experimental" / "axicon_aberration_correction"
sys.path.insert(0, str(MOD))

from miao_full_retrieval import (  # noqa: E402
    fit_plane_adaptive,
    angular_field_from_coefficients,
    resolve_conjugate_branch,
)

OUT = ROOT / "outputs" / "validation" / "miao_modal_equation_contract"
Q = 20
KP = 9.90e4
PIXEL_M = 2.0e-6
N = 512
THETA_N = 4096
EPS = np.finfo(float).tiny


def _truth(theta: np.ndarray) -> np.ndarray:
    # The amplitude marker is deliberately weak and smooth.  It is not an
    # aberration correction term; it exists only to make the conjugate branch
    # observable from independent input intensity, as in Miao et al.
    amp = 1.0 + 0.16*np.cos(theta) + 0.05*np.sin(2.0*theta)
    phase = 0.42*np.cos(2.0*theta) + 0.24*np.sin(3.0*theta) + 0.12*np.cos(5.0*theta)
    return amp*np.exp(1j*phase)


def _fourier_coefficients(field: np.ndarray, theta: np.ndarray, m_values: np.ndarray) -> np.ndarray:
    # c_m = (1/2pi) integral E(theta) exp(+i m theta) dtheta.
    # The common 2pi factor is irrelevant to intensity fitting and omitted.
    return np.asarray([
        np.mean(field*np.exp(1j*int(m)*theta)) for m in np.asarray(m_values, int)
    ], np.complex128)


def _synth_image() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0*np.pi, THETA_N, endpoint=False)
    truth = _truth(theta)
    m_values = np.arange(-12, 13, dtype=int)
    coeffs = _fourier_coefficients(truth, theta, m_values)

    c = 0.5*(N-1)
    yy, xx = np.indices((N, N), dtype=float)
    x = (xx-c)*PIXEL_M
    y = (yy-c)*PIXEL_M
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    u = np.zeros((N, N), np.complex128)
    for cm, m in zip(coeffs, m_values):
        n = int(m)-Q
        u += ((-1j)**n)*cm*special.jv(n, KP*r)*np.exp(-1j*n*phi)
    image = np.abs(u)**2
    image /= max(float(np.max(image)), EPS)
    return image, theta, truth, m_values


def _circular_rms(a: np.ndarray, b: np.ndarray) -> float:
    d = np.angle(np.exp(1j*(np.asarray(a)-np.asarray(b))))
    piston = np.angle(np.mean(np.exp(1j*d)))
    d = np.angle(np.exp(1j*(d-piston)))
    return float(np.sqrt(np.mean(d*d)))


def build(out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    image, theta_truth, truth, _ = _synth_image()
    center = (0.5*(N-1), 0.5*(N-1))

    fit = fit_plane_adaptive(
        image, 0, 0.1, center, PIXEL_M, Q, KP,
        max_aberration_order=30,
        order_step=2,
        cost_threshold=1e-5,
        min_fractional_improvement=1e-4,
        rmax_um=460,
        n_r=96,
        n_theta=192,
    )

    theta = np.asarray(fit.theta_rad, float)
    truth_on_fit = _truth(theta)
    recovered = angular_field_from_coefficients(fit.coeffs, fit.m_values, theta)
    branch, sd, sc = resolve_conjugate_branch(
        recovered[None, :], np.abs(truth_on_fit[None, :])**2, min_score_margin=0.01
    )
    if branch == "conjugate":
        recovered = np.conj(np.roll(recovered, len(recovered)//2))

    # Complex scale is unobservable in the intensity fit.  Compare amplitude
    # after peak normalization and phase after removing one global piston.
    amp_truth = np.abs(truth_on_fit); amp_truth /= max(float(amp_truth.max()), EPS)
    amp_rec = np.abs(recovered); amp_rec /= max(float(amp_rec.max()), EPS)
    amp_rmse = float(np.sqrt(np.mean((amp_truth-amp_rec)**2)))
    phase_rms = _circular_rms(np.angle(recovered), np.angle(truth_on_fit))

    summary = {
        "study": "direct Miao Eq3 q20 modal contract",
        "q": Q,
        "k_perp_truth_m_inv": KP,
        "k_perp_recovered_m_inv": float(fit.k_perp_m_inv),
        "k_perp_fractional_error": float(abs(fit.k_perp_m_inv-KP)/KP),
        "fit_cost": float(fit.fit_cost),
        "fit_corr": float(fit.fit_corr),
        "fit_nrmse": float(fit.fit_nrmse),
        "selected_residual_order_max": int(fit.aberration_order_max),
        "branch": branch,
        "branch_score_direct": sd,
        "branch_score_conjugate": sc,
        "retrieved_input_amplitude_rmse": amp_rmse,
        "retrieved_input_phase_rms_rad": phase_rms,
    }
    np.save(out/"synthetic_focal_intensity.npy", image.astype(np.float32))
    np.save(out/"truth_input_complex.npy", truth.astype(np.complex64))
    np.save(out/"retrieved_input_complex.npy", recovered.astype(np.complex64))
    (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    build()
