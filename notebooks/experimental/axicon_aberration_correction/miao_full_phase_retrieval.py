"""Full Miao-style phase-front retrieval for the measured q=20 Bessel z-scan.

This module implements the physics used in Miao et al., Optics Express 30,
11360-11371 (2022), DOI 10.1364/OE.454796, in the notation already used by
``modal_vortex_bessel.py``.

For every measured z plane we fit BOTH

    * the complex angular Bessel-mode coefficients c_n^(q), and
    * an optimum transverse wavenumber k_perp_opt(z).

The latter is essential.  In the stationary-phase model

    d psi_rho / d rho = k tan(alpha) - k_perp_opt,
    rho_z = z * k_perp_opt / (k tan(alpha)).

Therefore the z scan supplies radial diversity for one transverse input
wavefront.  It is not an independently programmable longitudinal phase.  The
radial phase is obtained by integrating the recovered radial gradient, while the
angular residual is reconstructed from the modal coefficients.  The programmed
q*theta vortex is factored out and is never put into the aberration correction.

Important limits
----------------
* A global phase piston remains unobservable and is fixed by gauge.
* If the absolute distance from the input/axicon reference to each measured z is
  not calibrated, only a model-registered/relative radial coordinate is available.
* The present morphology-centred acquisition cannot support a quantitative coma
  (m=+-1) claim.  Those modes are excluded by default; use the separate full-sensor
  trajectory analysis for pointing and calibrate the optical axis before enabling
  them.
* The complex-conjugate / pi-rotated twin ambiguity described by Miao et al. is
  reported as unresolved unless independent input-plane orientation information
  is supplied.
* A reconstructed transverse correction is NOT an SLM2 command until the
  camera/input-to-SLM transform and 1030-nm phase LUT are calibrated and a fresh
  post-correction z scan closes the loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy import integrate, ndimage, optimize, special

from modal_vortex_bessel import (
    EPS,
    aberration_field_theta,
    find_dark_core_center,
    fit_coefficients,
    modal_basis,
    principal_inner_ring,
    sample_polar,
)


@dataclass
class MiaoPlaneFit:
    """One measured z plane retrieved with variable k_perp."""

    plane_index: int
    z_relative_m: float
    center_y_px: float
    center_x_px: float
    core_score: float
    ring_radius_px: float
    k_seed_m_inv: float
    k_perp_opt_m_inv: float
    k_search_lo_m_inv: float
    k_search_hi_m_inv: float
    m_values: np.ndarray
    coeffs: np.ndarray
    data_cost: float
    correlation: float
    nrmse: float
    k_at_search_boundary: bool


@dataclass
class MiaoWavefront:
    """Recovered transverse wavefront sampled on input annuli."""

    z_relative_m: np.ndarray
    z_from_reference_m: np.ndarray
    rho_m: np.ndarray
    k_perp_opt_m_inv: np.ndarray
    radial_gradient_rad_per_m: np.ndarray
    radial_phase_rad: np.ndarray
    theta_rad: np.ndarray
    angular_complex_rows: np.ndarray
    residual_complex_rows: np.ndarray
    residual_phase_rows_rad: np.ndarray
    residual_amplitude_rows: np.ndarray
    nominal_k_tan_alpha_m_inv: float
    absolute_z_calibrated: bool
    nominal_cone_calibrated: bool
    global_piston_fixed_by_gauge: bool
    first_order_azimuthal_modes_retrieved: bool
    twin_ambiguity_resolved: bool
    metadata: dict


@dataclass
class CartesianResidual:
    x_m: np.ndarray
    y_m: np.ndarray
    residual_phase_rad: np.ndarray
    residual_amplitude: np.ndarray
    correction_phase_rad: np.ndarray
    valid_mask: np.ndarray
    metadata: dict


def _mode_indices(m_max: int, *, include_first_order: bool) -> np.ndarray:
    values = np.arange(-int(m_max), int(m_max) + 1, dtype=int)
    if not include_first_order:
        values = values[~np.isin(values, (-1, 1))]
    if 0 not in values:
        raise ValueError("m=0 ideal mode must be present")
    return values


def _polar_measurement(
    image: np.ndarray,
    *,
    pixel_pitch_m: float,
    rmax_um: float,
    n_r: int,
    n_theta: int,
    center_yx: tuple[float, float] | None = None,
) -> dict:
    image = np.asarray(image, float)
    if center_yx is None:
        cy, cx, core_score = find_dark_core_center(image)
    else:
        cy, cx = map(float, center_yx)
        core_score = np.nan
    ring_radius_px, _, _ = principal_inner_ring(image, cy, cx)
    max_radius_px = min(float(rmax_um) * 1e-6 / float(pixel_pitch_m), 70.0)
    radii_px = np.linspace(1.5, max_radius_px, int(n_r))
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False)
    measured = sample_polar(image, cy, cx, radii_px, theta)
    measured = np.clip(measured, 0.0, None)
    measured /= max(float(measured.max()), EPS)

    rr, tt = np.meshgrid(radii_px * float(pixel_pitch_m), theta, indexing="ij")
    rflat = rr.ravel()
    phiflat = tt.ravel()
    y = measured.ravel()
    # Eq. (5) contains r dr dphi.  The signal weighting is only a soft SNR term.
    weights = rflat / max(float(rflat.max()), EPS)
    weights *= 0.25 + 0.75 * np.sqrt(np.clip(y, 0.0, 1.0))
    return {
        "center_y_px": float(cy),
        "center_x_px": float(cx),
        "core_score": float(core_score),
        "ring_radius_px": float(ring_radius_px),
        "radii_px": radii_px,
        "theta": theta,
        "measured": measured,
        "rflat_m": rflat,
        "phiflat_rad": phiflat,
        "y": y,
        "weights": weights,
    }


def _data_cost_from_coeffs(B, coeffs, y, weights) -> tuple[float, float, float]:
    y = np.asarray(y, float)
    weights = np.asarray(weights, float)
    yn = y / max(float(y.max()), EPS)
    prediction = np.abs(np.asarray(B) @ np.asarray(coeffs)) ** 2
    pn = prediction / max(float(prediction.max()), EPS)
    denom = max(float(np.sum(weights * yn * yn)), EPS)
    data_cost = float(np.sum(weights * (pn - yn) ** 2) / denom)
    corr = 0.0 if np.std(yn) <= EPS or np.std(pn) <= EPS else float(np.corrcoef(yn, pn)[0, 1])
    nrmse = float(np.sqrt(np.mean((pn - yn) ** 2)) / max(np.sqrt(np.mean(yn**2)), EPS))
    return data_cost, corr, nrmse


def fit_polar_plane_variable_kperp(
    polar: dict,
    *,
    plane_index: int,
    z_relative_m: float,
    q: int,
    k_seed_m_inv: float,
    search_fraction: float = 0.12,
    coarse_points: int = 9,
    m_max_search: int = 4,
    m_max_final: int = 8,
    include_first_order: bool = False,
    search_maxiter: int = 35,
    final_maxiter: int = 140,
    reg: float = 2e-4,
) -> MiaoPlaneFit:
    """Fit k_perp and angular coefficients for one intensity plane.

    The first loop follows the paper's logic: search k_perp with a small number
    of modes.  A final coefficient fit at k_perp_opt uses the larger mode set.
    """

    k_seed = abs(float(k_seed_m_inv))
    if k_seed <= 0:
        raise ValueError("k_seed_m_inv must be positive")
    frac = float(search_fraction)
    if not 0.01 <= frac <= 0.40:
        raise ValueError("search_fraction outside a reasonable perturbative range")
    lo = k_seed * (1.0 - frac)
    hi = k_seed * (1.0 + frac)
    m_search = _mode_indices(m_max_search, include_first_order=include_first_order)
    m_final = _mode_indices(m_max_final, include_first_order=include_first_order)
    rflat = np.asarray(polar["rflat_m"], float)
    phiflat = np.asarray(polar["phiflat_rad"], float)
    y = np.asarray(polar["y"], float)
    weights = np.asarray(polar["weights"], float)

    cache: dict[float, tuple[float, np.ndarray]] = {}

    def evaluate(kperp: float) -> float:
        key = float(kperp)
        if key in cache:
            return cache[key][0]
        B, _ = modal_basis(int(q), m_search, key, rflat, phiflat)
        coeffs, _, _ = fit_coefficients(
            B, y, weights, m_search, maxiter=int(search_maxiter), reg=float(reg)
        )
        cost, _, _ = _data_cost_from_coeffs(B, coeffs, y, weights)
        cache[key] = (cost, coeffs)
        return cost

    coarse = np.linspace(lo, hi, max(5, int(coarse_points)))
    coarse_cost = np.asarray([evaluate(k) for k in coarse])
    ibest = int(np.argmin(coarse_cost))
    left = coarse[max(0, ibest - 1)]
    right = coarse[min(len(coarse) - 1, ibest + 1)]
    if not left < right:
        left, right = lo, hi

    result = optimize.minimize_scalar(
        evaluate,
        bounds=(float(left), float(right)),
        method="bounded",
        options={"xatol": max(2.0, 1.0e-5 * k_seed), "maxiter": 30},
    )
    kopt = float(result.x if result.success else coarse[ibest])

    B_final, _ = modal_basis(int(q), m_final, kopt, rflat, phiflat)
    coeffs, _, _ = fit_coefficients(
        B_final, y, weights, m_final, maxiter=int(final_maxiter), reg=float(reg)
    )
    data_cost, corr, nrmse = _data_cost_from_coeffs(B_final, coeffs, y, weights)
    edge_tol = 0.01 * (hi - lo)
    boundary = bool(kopt <= lo + edge_tol or kopt >= hi - edge_tol)
    return MiaoPlaneFit(
        plane_index=int(plane_index),
        z_relative_m=float(z_relative_m),
        center_y_px=float(polar["center_y_px"]),
        center_x_px=float(polar["center_x_px"]),
        core_score=float(polar["core_score"]),
        ring_radius_px=float(polar["ring_radius_px"]),
        k_seed_m_inv=k_seed,
        k_perp_opt_m_inv=kopt,
        k_search_lo_m_inv=lo,
        k_search_hi_m_inv=hi,
        m_values=m_final,
        coeffs=np.asarray(coeffs, complex),
        data_cost=data_cost,
        correlation=corr,
        nrmse=nrmse,
        k_at_search_boundary=boundary,
    )


def retrieve_variable_kperp_stack(
    images: Sequence[np.ndarray],
    z_relative_m: Iterable[float],
    *,
    pixel_pitch_m: float,
    q: int,
    rmax_um: float = 220.0,
    n_r: int = 48,
    n_theta: int = 96,
    search_fraction: float = 0.12,
    m_max_search: int = 4,
    m_max_final: int = 8,
    include_first_order: bool = False,
) -> list[MiaoPlaneFit]:
    """Run the two-stage variable-k_perp retrieval on every measured plane."""

    z = np.asarray(list(z_relative_m), float)
    if len(images) != len(z):
        raise ValueError("one z coordinate is required per image")
    if np.any(np.diff(z) <= 0):
        raise ValueError("z_relative_m must be strictly increasing")

    jprime = float(special.jnp_zeros(abs(int(q)), 1)[0])
    fits: list[MiaoPlaneFit] = []
    for i, (image, zi) in enumerate(zip(images, z)):
        polar = _polar_measurement(
            image,
            pixel_pitch_m=float(pixel_pitch_m),
            rmax_um=float(rmax_um),
            n_r=int(n_r),
            n_theta=int(n_theta),
        )
        k_seed = jprime / max(float(polar["ring_radius_px"]) * float(pixel_pitch_m), EPS)
        fits.append(
            fit_polar_plane_variable_kperp(
                polar,
                plane_index=i,
                z_relative_m=float(zi),
                q=int(q),
                k_seed_m_inv=k_seed,
                search_fraction=float(search_fraction),
                m_max_search=int(m_max_search),
                m_max_final=int(m_max_final),
                include_first_order=bool(include_first_order),
            )
        )
    return fits


def _resolve_nominal_k(
    fits: Sequence[MiaoPlaneFit],
    nominal_k_tan_alpha_m_inv: float | None,
) -> tuple[float, bool, str]:
    if nominal_k_tan_alpha_m_inv is not None:
        value = abs(float(nominal_k_tan_alpha_m_inv))
        if value <= 0:
            raise ValueError("nominal_k_tan_alpha_m_inv must be positive")
        return value, True, "externally supplied nominal k*tan(alpha)"
    values = np.asarray([f.k_perp_opt_m_inv for f in fits], float)
    value = float(np.median(values))
    return value, False, "robust median fitted k_perp; radial phase is relative to this reference"


def reconstruct_transverse_wavefront(
    fits: Sequence[MiaoPlaneFit],
    *,
    q: int,
    z_at_relative_zero_from_reference_m: float | None,
    nominal_k_tan_alpha_m_inv: float | None = None,
    n_theta: int = 720,
    include_first_order: bool = False,
    twin_ambiguity_resolved: bool = False,
) -> MiaoWavefront:
    """Assemble radial + angular residual phase according to Miao Eqs. (3)-(6).

    If the absolute axial offset is unavailable, a positive model coordinate is
    created by translating the first scan plane to a small positive distance.
    That supports relative radial reconstruction/diagnostics only and is marked
    as uncalibrated.  It must never be used as an SLM radius map.
    """

    if not fits:
        raise ValueError("no plane fits supplied")
    z_rel = np.asarray([f.z_relative_m for f in fits], float)
    if np.any(np.diff(z_rel) <= 0):
        raise ValueError("plane fits must be ordered by increasing z")
    kopt = np.asarray([f.k_perp_opt_m_inv for f in fits], float)
    k_nom, nominal_calibrated, nominal_source = _resolve_nominal_k(
        fits, nominal_k_tan_alpha_m_inv
    )

    if z_at_relative_zero_from_reference_m is None:
        # A gauge for relative radial spacing only.  The offset is explicitly
        # not interpreted as a physical input-plane distance.
        dz = float(np.median(np.diff(z_rel)))
        z_abs = z_rel - float(z_rel.min()) + max(abs(dz), 1.0e-6)
        absolute_z_calibrated = False
        z_source = "relative scan coordinate shifted positive; not a calibrated input distance"
    else:
        z_abs = z_rel + float(z_at_relative_zero_from_reference_m)
        if np.any(z_abs <= 0):
            raise ValueError("absolute z mapping places a measured plane at z<=0")
        absolute_z_calibrated = True
        z_source = "externally/model supplied distance from the input/axicon reference"

    # Miao: rho_z = z * k_perp_opt / (k tan alpha)
    rho = z_abs * kopt / k_nom
    order = np.argsort(rho)
    if np.any(np.diff(rho[order]) <= 0):
        raise ValueError("retrieved annulus radii are not unique/monotonic")

    # Miao: d psi_rho / d rho = k tan alpha - k_perp_opt
    grad = k_nom - kopt
    rho_sorted = rho[order]
    grad_sorted = grad[order]
    radial_phase_sorted = integrate.cumulative_trapezoid(
        grad_sorted, rho_sorted, initial=0.0
    )
    # Global piston is unobservable; choose weighted mean zero.
    radial_phase_sorted -= float(np.mean(radial_phase_sorted))

    theta = np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False)
    angular_rows_sorted: list[np.ndarray] = []
    for idx in order:
        fit = fits[int(idx)]
        g = aberration_field_theta(fit.coeffs, fit.m_values, theta)
        # fit_coefficients fixes the ideal m=0 coefficient real-positive.  This
        # supplies the angular-row phase gauge; radial phase is restored separately.
        angular_rows_sorted.append(np.asarray(g, complex))
    angular = np.stack(angular_rows_sorted)
    radial_factor = np.exp(1j * radial_phase_sorted)[:, None]
    residual_complex = radial_factor * angular
    residual_phase = np.angle(residual_complex)
    residual_amp = np.abs(residual_complex)
    residual_amp /= max(float(np.nanmedian(residual_amp)), EPS)

    metadata = {
        "method": "Miao-style variable-k_perp annular phase retrieval",
        "equations": {
            "radial_gradient": "dpsi_rho/drho = k*tan(alpha) - k_perp_opt",
            "annulus_radius": "rho_z = z*k_perp_opt/(k*tan(alpha))",
            "angular_field": "g(theta)=sum_m c_m exp(-i m theta), with m=n+q",
        },
        "q_target": int(q),
        "programmed_qtheta_in_residual": False,
        "nominal_k_source": nominal_source,
        "z_mapping_source": z_source,
        "global_piston": "fixed to zero-mean gauge",
        "m_plus_minus_1": "included" if include_first_order else "excluded: centred morphology data do not calibrate coma/pointing",
        "twin_ambiguity": "resolved externally" if twin_ambiguity_resolved else "unresolved without independent input-plane orientation/intensity",
        "hardware_ready": False,
        "hardware_blocker": (
            "requires absolute input-to-camera radial mapping, relay magnification, camera-to-SLM "
            "rotation/parity, illuminated SLM footprint, 1030-nm phase LUT, and a new post-correction z scan"
        ),
    }
    return MiaoWavefront(
        z_relative_m=z_rel[order],
        z_from_reference_m=z_abs[order],
        rho_m=rho_sorted,
        k_perp_opt_m_inv=kopt[order],
        radial_gradient_rad_per_m=grad_sorted,
        radial_phase_rad=radial_phase_sorted,
        theta_rad=theta,
        angular_complex_rows=angular,
        residual_complex_rows=residual_complex,
        residual_phase_rows_rad=residual_phase,
        residual_amplitude_rows=residual_amp,
        nominal_k_tan_alpha_m_inv=float(k_nom),
        absolute_z_calibrated=absolute_z_calibrated,
        nominal_cone_calibrated=nominal_calibrated,
        global_piston_fixed_by_gauge=True,
        first_order_azimuthal_modes_retrieved=bool(include_first_order),
        twin_ambiguity_resolved=bool(twin_ambiguity_resolved),
        metadata=metadata,
    )


def _interp_complex_rows(
    rows: np.ndarray,
    rho_rows: np.ndarray,
    R: np.ndarray,
    TH: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation of the complex residual field in rho/theta."""

    rows = np.asarray(rows, complex)
    rho_rows = np.asarray(rho_rows, float)
    if rows.ndim != 2 or len(rho_rows) != rows.shape[0]:
        raise ValueError("row dimensions disagree")
    if np.any(np.diff(rho_rows) <= 0):
        raise ValueError("rho_rows must increase")

    hi = np.searchsorted(rho_rows, R, side="right")
    hi = np.clip(hi, 1, len(rho_rows) - 1)
    lo = hi - 1
    wr = (R - rho_rows[lo]) / np.maximum(rho_rows[hi] - rho_rows[lo], EPS)

    ntheta = rows.shape[1]
    tf = np.mod(TH, 2.0 * np.pi) / (2.0 * np.pi) * ntheta
    j0 = np.floor(tf).astype(int) % ntheta
    j1 = (j0 + 1) % ntheta
    wt = tf - np.floor(tf)
    lower = (1.0 - wt) * rows[lo, j0] + wt * rows[lo, j1]
    upper = (1.0 - wt) * rows[hi, j0] + wt * rows[hi, j1]
    return (1.0 - wr) * lower + wr * upper


def assemble_cartesian_residual(
    wavefront: MiaoWavefront,
    *,
    grid_size: int = 700,
    padding_fraction: float = 0.08,
) -> CartesianResidual:
    """Map the recovered complex residual to a Cartesian input-plane grid."""

    rho_min = float(wavefront.rho_m[0])
    rho_max = float(wavefront.rho_m[-1])
    span = max(rho_max - rho_min, EPS)
    extent = rho_max + float(padding_fraction) * span
    axis = np.linspace(-extent, extent, int(grid_size))
    X, Y = np.meshgrid(axis, axis, indexing="xy")
    R = np.hypot(X, Y)
    TH = np.arctan2(Y, X)
    complex_map = _interp_complex_rows(
        wavefront.residual_complex_rows, wavefront.rho_m, R, TH
    )
    valid = (R >= rho_min) & (R <= rho_max)
    phase = np.full(R.shape, np.nan, float)
    amplitude = np.full(R.shape, np.nan, float)
    correction = np.full(R.shape, np.nan, float)
    phase[valid] = np.angle(complex_map[valid])
    amplitude[valid] = np.abs(complex_map[valid])
    correction[valid] = -phase[valid]
    if np.any(valid):
        amplitude[valid] /= max(float(np.nanmedian(amplitude[valid])), EPS)
    return CartesianResidual(
        x_m=axis,
        y_m=axis,
        residual_phase_rad=phase,
        residual_amplitude=amplitude,
        correction_phase_rad=correction,
        valid_mask=valid,
        metadata={
            **wavefront.metadata,
            "cartesian_interpolation": "complex residual field; wrapped angles are never interpolated as scalars",
            "correction_type": "phase-only conjugate of retrieved transverse residual",
            "hardware_ready": False,
        },
    )


def plane_fits_to_table(fits: Sequence[MiaoPlaneFit]):
    """Return a pandas DataFrame without making pandas a hard module dependency."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "plane_index": f.plane_index,
                "z_relative_mm": f.z_relative_m * 1e3,
                "center_y_px": f.center_y_px,
                "center_x_px": f.center_x_px,
                "core_score": f.core_score,
                "ring_radius_px": f.ring_radius_px,
                "k_seed_rad_per_um": f.k_seed_m_inv * 1e-6,
                "k_perp_opt_rad_per_um": f.k_perp_opt_m_inv * 1e-6,
                "data_cost": f.data_cost,
                "correlation": f.correlation,
                "nrmse": f.nrmse,
                "k_at_search_boundary": f.k_at_search_boundary,
            }
            for f in fits
        ]
    )
