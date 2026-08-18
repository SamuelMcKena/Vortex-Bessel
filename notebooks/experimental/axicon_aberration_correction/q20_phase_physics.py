"""Physics-safe reconstruction helpers for the q=20 vortex--Bessel z-scan.

The measured z-scan does *not* provide an independent longitudinal phase mask.
Instead, each camera plane samples a different annulus of one transverse input
wavefront.  For a conical wave, the stationary-phase mapping is

    rho_z = z tan(alpha),     k_perp = k tan(alpha)  (paraxial)

or, using the exact decomposition k**2 = k_perp**2 + k_z**2,

    tan(alpha) = k_perp / k_z.

This module therefore treats the z-stack as radial diversity for reconstructing
one transverse residual phase psi(rho, theta).  It deliberately removes the
unobservable annulus-to-annulus piston and never re-inserts the programmed
q*theta vortex phase into the *aberration* correction.

Reference
---------
B. Miao, L. Feder, J. E. Shrock, H. M. Milchberg,
"Phase front retrieval and correction of Bessel beams," Optics Express 30,
11360--11371 (2022), doi:10.1364/OE.454796.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

EPS = 1.0e-12


@dataclass(frozen=True)
class ConeGeometry:
    wavelength_m: float
    k_m_inv: float
    k_perp_m_inv: float
    k_z_m_inv: float
    tan_alpha: float
    alpha_rad: float


@dataclass(frozen=True)
class AnnulusMapping:
    z_relative_m: np.ndarray
    z_from_axicon_m: np.ndarray | None
    rho_m: np.ndarray
    radius_reference: Literal["absolute", "relative"]
    absolute_radius_calibrated: bool


def cone_geometry(wavelength_m: float, k_perp_m_inv: float) -> ConeGeometry:
    """Return conical-ray geometry from wavelength and transverse wavenumber.

    ``k_perp`` must be smaller than the free-space wavenumber.  The exact
    k-vector decomposition is used for the angle; at the small angles relevant
    to an axicon this reduces to k_perp ~= k*tan(alpha).
    """
    wavelength_m = float(wavelength_m)
    k_perp_m_inv = abs(float(k_perp_m_inv))
    if wavelength_m <= 0:
        raise ValueError("wavelength_m must be positive")
    k = 2.0 * np.pi / wavelength_m
    if not (0.0 < k_perp_m_inv < k):
        raise ValueError("k_perp_m_inv must satisfy 0 < k_perp < 2*pi/lambda")
    k_z = float(np.sqrt(k * k - k_perp_m_inv * k_perp_m_inv))
    tan_alpha = k_perp_m_inv / k_z
    return ConeGeometry(
        wavelength_m=wavelength_m,
        k_m_inv=k,
        k_perp_m_inv=k_perp_m_inv,
        k_z_m_inv=k_z,
        tan_alpha=tan_alpha,
        alpha_rad=float(np.arctan(tan_alpha)),
    )


def annulus_mapping_from_z(
    z_relative_m: np.ndarray,
    *,
    wavelength_m: float,
    k_perp_m_inv: float,
    z_at_relative_zero_from_axicon_m: float | None = None,
) -> AnnulusMapping:
    """Map measured z planes to sampled input-wavefront annuli.

    If the distance from the axicon/input reference to ``z_relative == 0`` is
    known, absolute radii are returned.  Otherwise only *relative* annulus
    radius is physically determined by the scan, and the innermost sampled
    plane is assigned rho=0 for diagnostic plotting only.

    The relative mapping is scientifically useful for visualising how the
    recovered *non-axisymmetric* residual evolves across the scanned aperture,
    but it is not a hardware coordinate map.
    """
    z = np.asarray(z_relative_m, dtype=float)
    if z.ndim != 1 or z.size < 2:
        raise ValueError("z_relative_m must be a 1-D array containing >=2 planes")
    if not np.all(np.isfinite(z)):
        raise ValueError("z_relative_m contains non-finite values")
    geom = cone_geometry(wavelength_m, k_perp_m_inv)

    if z_at_relative_zero_from_axicon_m is None:
        rho = (z - np.min(z)) * geom.tan_alpha
        z_abs = None
        reference: Literal["absolute", "relative"] = "relative"
        calibrated = False
    else:
        z_abs = z + float(z_at_relative_zero_from_axicon_m)
        if np.any(z_abs <= 0):
            raise ValueError(
                "Absolute camera distances must be positive; check the z offset/sign"
            )
        rho = z_abs * geom.tan_alpha
        reference = "absolute"
        calibrated = True

    order = np.argsort(rho)
    return AnnulusMapping(
        z_relative_m=z[order],
        z_from_axicon_m=None if z_abs is None else z_abs[order],
        rho_m=np.asarray(rho[order], dtype=float),
        radius_reference=reference,
        absolute_radius_calibrated=calibrated,
    )


def unit_phasor(phase_rad: np.ndarray) -> np.ndarray:
    phase = np.asarray(phase_rad, dtype=float)
    if not np.all(np.isfinite(phase)):
        raise ValueError("phase_rad contains non-finite values")
    return np.exp(1j * phase)


def gauge_annular_phase_rows(phase_rows_rad: np.ndarray) -> np.ndarray:
    """Remove unobservable annulus-to-annulus piston without unwrapping phase.

    Intensity at one z plane is invariant to a global complex phase.  Independent
    modal fits therefore do not measure the radial piston between annuli.  We
    choose a smooth gauge by aligning each row's unit phasor to the preceding
    row.  This makes the gauge choice explicit rather than accidentally turning
    arbitrary wrapped row pistons into radial structure.

    The returned map contains only the recoverable angular/non-axisymmetric
    phase in this gauge.  It is *not* a recovered axisymmetric radial phase.
    """
    phase = np.asarray(phase_rows_rad, dtype=float)
    if phase.ndim != 2 or phase.shape[0] < 2 or phase.shape[1] < 8:
        raise ValueError("phase_rows_rad must have shape (n_z>=2, n_theta>=8)")
    u = unit_phasor(phase).astype(complex, copy=True)

    ref = np.mean(u[0])
    if abs(ref) < 1e-8:
        ref = u[0, 0]
    u[0] *= np.exp(-1j * np.angle(ref))

    for i in range(1, len(u)):
        overlap = np.vdot(u[i - 1], u[i])
        if abs(overlap) < 1e-8:
            delta = np.angle(u[i, 0])
        else:
            delta = np.angle(overlap)
        u[i] *= np.exp(-1j * delta)
    return np.angle(u)


def _bilinear_phasor_interpolation(
    phasor_rows: np.ndarray,
    rho_rows_m: np.ndarray,
    rho_query_m: np.ndarray,
    theta_query_rad: np.ndarray,
) -> np.ndarray:
    """Interpolate complex unit phasors in (rho, theta), including theta wrap."""
    u = np.asarray(phasor_rows, dtype=complex)
    rho_rows = np.asarray(rho_rows_m, dtype=float)
    if u.ndim != 2 or rho_rows.ndim != 1 or len(rho_rows) != u.shape[0]:
        raise ValueError("phasor_rows and rho_rows_m dimensions are inconsistent")
    if np.any(np.diff(rho_rows) <= 0):
        raise ValueError("rho_rows_m must be strictly increasing")

    rq = np.asarray(rho_query_m, dtype=float)
    tq = np.mod(np.asarray(theta_query_rad, dtype=float), 2.0 * np.pi)
    if rq.shape != tq.shape:
        raise ValueError("rho_query_m and theta_query_rad must have the same shape")

    hi = np.searchsorted(rho_rows, rq, side="right")
    hi = np.clip(hi, 1, len(rho_rows) - 1)
    lo = hi - 1
    denom = np.maximum(rho_rows[hi] - rho_rows[lo], EPS)
    wr = (rq - rho_rows[lo]) / denom

    ntheta = u.shape[1]
    tf = tq / (2.0 * np.pi) * ntheta
    j0 = np.floor(tf).astype(int) % ntheta
    j1 = (j0 + 1) % ntheta
    wt = tf - np.floor(tf)

    u00 = u[lo, j0]
    u01 = u[lo, j1]
    u10 = u[hi, j0]
    u11 = u[hi, j1]
    low = (1.0 - wt) * u00 + wt * u01
    high = (1.0 - wt) * u10 + wt * u11
    out = (1.0 - wr) * low + wr * high
    mag = np.abs(out)
    out = np.where(mag > EPS, out / mag, 1.0 + 0.0j)
    return out


def assemble_transverse_residual_phase(
    phase_rows_rad: np.ndarray,
    z_relative_m: np.ndarray,
    *,
    wavelength_m: float,
    k_perp_m_inv: float,
    z_at_relative_zero_from_axicon_m: float | None = None,
    grid_size: int = 512,
    padding_fraction: float = 0.08,
) -> dict[str, np.ndarray | float | str | bool | None]:
    """Assemble a 2-D *transverse residual* phase from z-sampled annuli.

    The interpolation is performed through complex unit phasors, never directly
    on wrapped phase values.  The programmed vortex phase q*theta is intentionally
    absent: the output is an aberration residual only.

    Without an absolute z-to-input-plane distance, the radial coordinate is only
    relative and the result is diagnostic.  Even with an absolute annulus radius,
    camera-to-SLM magnification/rotation/parity and the SLM phase LUT are still
    required before hardware use.
    """
    phase = np.asarray(phase_rows_rad, dtype=float)
    z = np.asarray(z_relative_m, dtype=float)
    if phase.shape[0] != z.size:
        raise ValueError("one phase row is required for every z plane")
    if int(grid_size) < 64:
        raise ValueError("grid_size must be >=64")

    mapping = annulus_mapping_from_z(
        z,
        wavelength_m=wavelength_m,
        k_perp_m_inv=k_perp_m_inv,
        z_at_relative_zero_from_axicon_m=z_at_relative_zero_from_axicon_m,
    )
    geom = cone_geometry(wavelength_m, k_perp_m_inv)
    original_rho = ((z - np.min(z)) * geom.tan_alpha
                    if z_at_relative_zero_from_axicon_m is None
                    else (z + float(z_at_relative_zero_from_axicon_m)) * geom.tan_alpha)
    order = np.argsort(original_rho)
    phase = phase[order]
    phase_gauge = gauge_annular_phase_rows(phase)
    u = unit_phasor(phase_gauge)

    rho_min = float(mapping.rho_m[0])
    rho_max = float(mapping.rho_m[-1])
    span = max(rho_max - rho_min, EPS)
    radius_extent = rho_max + float(padding_fraction) * span
    if mapping.radius_reference == "relative":
        rho_min_display = max(0.06 * span, EPS)
        rho_rows_for_display = mapping.rho_m + rho_min_display
        rho_min = float(rho_rows_for_display[0])
        rho_max = float(rho_rows_for_display[-1])
        radius_extent = rho_max + float(padding_fraction) * span
    else:
        rho_rows_for_display = mapping.rho_m

    axis_m = np.linspace(-radius_extent, radius_extent, int(grid_size))
    X, Y = np.meshgrid(axis_m, axis_m, indexing="xy")
    R = np.hypot(X, Y)
    TH = np.arctan2(Y, X)
    mapped = _bilinear_phasor_interpolation(u, rho_rows_for_display, R, TH)
    valid = (R >= rho_min) & (R <= rho_max)
    residual = np.full(R.shape, np.nan, dtype=float)
    residual[valid] = np.angle(mapped[valid])
    correction = np.full(R.shape, np.nan, dtype=float)
    correction[valid] = np.angle(np.conj(mapped[valid]))

    return {
        "x_m": axis_m,
        "y_m": axis_m,
        "rho_rows_m": rho_rows_for_display,
        "rho_rows_physical_m": mapping.rho_m,
        "z_relative_m": mapping.z_relative_m,
        "residual_phase_rad": residual,
        "conjugate_correction_phase_rad": correction,
        "gauge_fixed_phase_rows_rad": phase_gauge,
        "k_perp_m_inv": geom.k_perp_m_inv,
        "tan_alpha": geom.tan_alpha,
        "alpha_rad": geom.alpha_rad,
        "radius_reference": mapping.radius_reference,
        "absolute_radius_calibrated": mapping.absolute_radius_calibrated,
        "radial_piston_recovered": False,
        "contains_programmed_vortex_phase": False,
        "hardware_ready": False,
        "hardware_blocker": (
            "annulus-to-annulus piston is not measured by independent intensity fits; "
            "absolute camera-to-input radius and camera/SLM magnification, rotation, parity "
            "and phase LUT must be calibrated and then validated with a new measured z-stack"
        ),
    }


def central_band_sections(
    stack: np.ndarray,
    *,
    half_width_px: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return noise-robust x-z and y-z central sections from a (z,y,x) stack."""
    a = np.asarray(stack, dtype=float)
    if a.ndim != 3:
        raise ValueError("stack must have shape (z, y, x)")
    cy = a.shape[1] // 2
    cx = a.shape[2] // 2
    h = int(max(0, half_width_px))
    xz = a[:, max(0, cy - h):min(a.shape[1], cy + h + 1), :].mean(axis=1)
    yz = a[:, :, max(0, cx - h):min(a.shape[2], cx + h + 1)].mean(axis=2)
    return xz, yz
