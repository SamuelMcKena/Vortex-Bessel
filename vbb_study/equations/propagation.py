"""Propagation and sampling equations for scalar-field checks.

The heavy propagators live in ``bessel_twin_core``.  Here I keep the compact
expressions for medium wavelength, ASM longitudinal wavenumber, Matsushima-style
band limiting, and Nyquist margins so the notebooks can report the same
quantities with one vocabulary.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from .fields import fft2c, ifft2c, make_xy_grid

TWOPI = 2.0 * math.pi
EPS = 1.0e-30
um = 1e-6


def medium_wavelength_m(wavelength0_m: float, n_medium: float = 1.0) -> float:
    """Return the wavelength inside a medium in metres."""

    return float(wavelength0_m) / max(float(n_medium), EPS)


def medium_wavenumber_m_inv(wavelength0_m: float, n_medium: float = 1.0) -> float:
    """Return ``k = 2*pi/lambda_medium`` in rad/m."""

    return TWOPI / max(medium_wavelength_m(wavelength0_m, n_medium), EPS)


def fft_frequency_grid(N: int, dx_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Return centered FFT frequency grids in cycles/m."""

    freq = np.fft.fftshift(np.fft.fftfreq(int(N), d=float(dx_m)))
    return np.meshgrid(freq, freq, indexing="xy")


def asm_longitudinal_wavenumber_m_inv(
    FX_cpm: Any,
    FY_cpm: Any,
    *,
    wavelength_m: float,
    include_evanescent: bool = True,
) -> np.ndarray:
    """Return the ASM ``k_z`` array in rad/m for frequency grids in cycles/m."""

    FX = np.asarray(FX_cpm, dtype=float)
    FY = np.asarray(FY_cpm, dtype=float)
    arg = TWOPI**2 * ((1.0 / max(float(wavelength_m), EPS)) ** 2 - FX**2 - FY**2)
    root = np.sqrt(np.abs(arg))
    if include_evanescent:
        return np.where(arg >= 0.0, root, 1j * root)
    return np.where(arg >= 0.0, root, 0.0)


def matsushima_bandlimit_mask(
    FX_cpm: Any,
    FY_cpm: Any,
    *,
    wavelength_m: float,
    z_m: float,
    N: int,
    dx_m: float,
) -> np.ndarray:
    """Return the square band-limit mask used by BL-ASM.

    This matches the compact sampling condition used in the engine:
    ``u_lim = 1/(lambda * sqrt((2*du*z)^2 + 1))`` with
    ``du = 1/(N*dx)``.
    """

    FX = np.asarray(FX_cpm, dtype=float)
    FY = np.asarray(FY_cpm, dtype=float)
    du = 1.0 / (int(N) * float(dx_m))
    u_lim = 1.0 / (max(float(wavelength_m), EPS) * math.sqrt((2.0 * du * abs(float(z_m))) ** 2 + 1.0))
    return (np.abs(FX) <= u_lim) & (np.abs(FY) <= u_lim)


def nyquist_spatial_frequency_cpm(dx_m: float) -> float:
    """Return the grid Nyquist spatial frequency in cycles/m."""

    return 0.5 / max(float(dx_m), EPS)


def radial_period_m(kr_m_inv: float) -> float:
    """Return the radial Bessel fringe period ``2*pi/k_r`` in metres."""

    return TWOPI / max(float(kr_m_inv), EPS)


def samples_per_radial_period(dx_m: float, kr_m_inv: float) -> float:
    """Return how many transverse samples cover one ``2*pi/k_r`` period."""

    return radial_period_m(float(kr_m_inv)) / max(float(dx_m), EPS)


def sampling_margin_for_transverse_wavevector(dx_m: float, kr_m_inv: float) -> float:
    """Return a Nyquist margin for a transverse wavevector.

    Values greater than one mean ``k_r/(2*pi)`` lies inside the sampling
    Nyquist limit; smaller values are a warning that the radial oscillation is
    under-sampled.
    """

    kr_cpm = float(kr_m_inv) / TWOPI
    return nyquist_spatial_frequency_cpm(float(dx_m)) / max(abs(kr_cpm), EPS)


def discrete_power(values: Any, dx_m: float) -> float:
    """Return ``sum(|U|^2) dx^2`` for a sampled transverse field."""

    U = np.asarray(values)
    return float(np.sum(np.abs(U) ** 2) * float(dx_m) * float(dx_m))


def _kz_medium(
    grid: dict[str, Any],
    wavelength_m: float,
    n_medium: float,
    include_evanescent: bool = True,
) -> np.ndarray:
    k = TWOPI * float(n_medium) / float(wavelength_m)
    kx = TWOPI * grid["FX"]
    ky = TWOPI * grid["FY"]
    arg = k**2 - kx**2 - ky**2
    if include_evanescent:
        return np.where(arg >= 0.0, np.sqrt(np.maximum(arg, 0.0)), 1j * np.sqrt(np.maximum(-arg, 0.0)))
    return np.where(arg >= 0.0, np.sqrt(np.maximum(arg, 0.0)), 0.0)


def _transfer_function_medium(
    grid: dict[str, Any],
    wavelength_m: float,
    z_m: float,
    n_medium: float,
    include_evanescent: bool = True,
) -> np.ndarray:
    """Stable medium transfer function.

    Propagating components use `exp(i*kz*z)`. Evanescent components decay as
    `exp(-alpha*abs(z))`, which avoids unphysical blow-up during negative-z
    focal scans.
    """

    k = TWOPI * float(n_medium) / float(wavelength_m)
    kx = TWOPI * grid["FX"]
    ky = TWOPI * grid["FY"]
    arg = k**2 - kx**2 - ky**2
    prop = arg >= 0.0
    H = np.zeros_like(arg, dtype=complex)
    H[prop] = np.exp(1j * np.sqrt(arg[prop]) * float(z_m))
    if include_evanescent:
        H[~prop] = np.exp(-np.sqrt(np.maximum(-arg[~prop], 0.0)) * abs(float(z_m)))
    return H


def bandlimit_mask_matsushima(
    grid: dict[str, Any],
    wavelength_m: float,
    z_m: float,
    n_medium: float = 1.0,
) -> np.ndarray:
    """Matsushima-Shimobaba rectangular transfer-function bandlimit."""

    lam = float(wavelength_m) / max(float(n_medium), EPS)
    du = 1.0 / (int(grid["N"]) * float(grid["dx"]))
    zz = abs(float(z_m))
    u_lim = 1.0 / (lam * math.sqrt((2.0 * du * zz) ** 2 + 1.0))
    return (np.abs(grid["FX"]) <= u_lim) & (np.abs(grid["FY"]) <= u_lim)


def angular_spectrum_propagate_bl(
    U0: np.ndarray,
    grid: dict[str, Any],
    wavelength_m: float,
    z_m: float,
    n_medium: float = 1.0,
    bandlimit: bool = True,
    include_evanescent: bool = True,
) -> np.ndarray:
    """Medium-aware band-limited angular-spectrum propagation."""

    H = _transfer_function_medium(grid, wavelength_m, z_m, n_medium, include_evanescent=include_evanescent)
    if bandlimit:
        H = H * bandlimit_mask_matsushima(grid, wavelength_m, z_m, n_medium=n_medium)
    return ifft2c(fft2c(U0) * H)


def make_bl_asm_propagator(
    U0: np.ndarray,
    grid: dict[str, Any],
    wavelength_m: float,
    n_medium: float = 1.0,
    bandlimit: bool = True,
    include_evanescent: bool = True,
) -> Callable[[float], np.ndarray]:
    """Precompute the angular spectrum and return a propagation closure."""

    A0 = fft2c(U0)
    def prop(z_m: float) -> np.ndarray:
        H = _transfer_function_medium(grid, wavelength_m, z_m, n_medium, include_evanescent=include_evanescent)
        if bandlimit:
            H = H * bandlimit_mask_matsushima(grid, wavelength_m, z_m, n_medium=n_medium)
        return ifft2c(A0 * H)

    return prop


def _zero_pad_center(U: np.ndarray, pad_factor: int = 2) -> np.ndarray:
    """Center-pad a square complex field by an integer factor."""

    arr = np.asarray(U, complex)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("SAS propagation expects one square 2D field.")
    pf = int(pad_factor)
    if pf < 1:
        raise ValueError("pad_factor must be >= 1.")
    if pf == 1:
        return arr.copy()
    N = int(arr.shape[0])
    out = np.zeros((pf * N, pf * N), dtype=complex)
    y0 = (out.shape[0] - N) // 2
    x0 = (out.shape[1] - N) // 2
    out[y0 : y0 + N, x0 : x0 + N] = arr
    return out


def _zero_unpad_center(U: np.ndarray, original_shape: tuple[int, int]) -> np.ndarray:
    """Return the centered window matching `original_shape`."""

    ny, nx = int(original_shape[0]), int(original_shape[1])
    y0 = (U.shape[0] - ny) // 2
    x0 = (U.shape[1] - nx) // 2
    return U[y0 : y0 + ny, x0 : x0 + nx]


def _sas_z_limit_m(side_length_m: float, N: int, wavelength_medium_m: float) -> float:
    """Distance limit from the reference SAS implementation."""

    L = float(side_length_m)
    n = float(N)
    lam = float(wavelength_medium_m)
    if L <= 0.0 or n <= 0.0 or lam <= 0.0:
        return 0.0
    a = math.sqrt(8.0 * L**2 / n**2 + lam**2)
    b = math.sqrt(L**2 / (8.0 * L**2 + n**2 * lam**2))
    denom = lam * (-1.0 + 2.0 * math.sqrt(2.0) * b)
    if abs(denom) <= EPS:
        return float("inf")
    return float((-4.0 * L * a * b) / denom)


def sas_validity_report(
    grid: dict[str, Any],
    wavelength_m: float,
    z_m: float,
    n_medium: float = 1.0,
    pad_factor: int = 2,
) -> dict[str, float | bool]:
    """Return SAS output sampling and z-limit diagnostics."""

    N = int(grid["N"])
    dx = float(grid["dx"])
    pf = max(1, int(pad_factor))
    lam = float(wavelength_m) / max(float(n_medium), EPS)
    side = N * dx
    z_abs = abs(float(z_m))
    z_limit = _sas_z_limit_m(side, N, lam)
    if z_abs <= EPS:
        out_dx = dx
        out_side = side
        magnification = 1.0
    else:
        out_dx = lam * z_abs / max(pf * side, EPS)
        out_side = N * out_dx
        magnification = out_side / max(side, EPS)
    return {
        "valid": bool(z_abs <= z_limit + EPS),
        "z_abs_m": float(z_abs),
        "z_limit_m": float(z_limit),
        "z_limit_margin_m": float(z_limit - z_abs),
        "z_over_limit": float(z_abs / (z_limit + EPS)),
        "input_dx_m": float(dx),
        "input_side_length_m": float(side),
        "medium_wavelength_m": float(lam),
        "pad_factor": int(pf),
        "output_dx_m": float(out_dx),
        "output_side_length_m": float(out_side),
        "output_magnification": float(magnification),
    }


def scalable_angular_spectrum_propagate(
    U0: np.ndarray,
    grid: dict[str, Any],
    wavelength_m: float,
    z_m: float,
    n_medium: float = 1.0,
    pad_factor: int = 2,
    bandlimit: bool = True,
    skip_final_phase: bool = True,
    allow_invalid: bool = False,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Scalable angular-spectrum propagation with a scaled output grid.

    This NumPy implementation follows the three-FFT SAS structure from the
    public PyTorch reference notebook and uses the medium wavelength
    `wavelength_m / n_medium`.  The returned field is the central unpadded
    window on the scaled output grid.
    """

    U = np.asarray(U0, complex)
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("SAS propagation expects one square 2D field.")
    N = int(U.shape[0])
    dx = float(grid["dx"])
    pf = max(1, int(pad_factor))
    z = float(z_m)
    validity = sas_validity_report(grid, wavelength_m, z, n_medium=n_medium, pad_factor=pf)
    if abs(z) <= EPS:
        meta = {
            **validity,
            "method": "sas",
            "bandlimit": bool(bandlimit),
            "skip_final_phase": bool(skip_final_phase),
            "input_power": float(np.sum(np.abs(U) ** 2) * dx * dx),
            "full_output_power": float(np.sum(np.abs(U) ** 2) * dx * dx),
            "retained_power_fraction": 1.0,
        }
        return U.copy(), make_xy_grid(N, dx), meta
    if not bool(validity["valid"]) and not allow_invalid:
        raise ValueError(
            "SAS propagation distance exceeds the reference z-limit: "
            f"|z|={abs(z)/um:.3g} um, limit={float(validity['z_limit_m'])/um:.3g} um. "
            "Set sas_allow_invalid=True to override."
        )

    lam = float(validity["medium_wavelength_m"])
    side = float(validity["input_side_length_m"])
    Np = pf * N
    side_padded = pf * side
    k = TWOPI / max(lam, EPS)

    U_p = _zero_pad_center(U, pf)
    padded_grid = make_xy_grid(Np, dx)
    U_un = np.fft.ifftshift(U_p)

    fx = np.fft.fftfreq(Np, d=dx)
    fy = fx
    FX = fx.reshape(1, Np)
    FY = fy.reshape(Np, 1)

    cx = lam * FX
    cy = lam * FY
    if bandlimit:
        tx = side_padded / (2.0 * abs(z)) + np.abs(cx)
        ty = side_padded / (2.0 * abs(z)) + np.abs(cy)
        W = ((cx**2 * (1.0 + tx**2) / (tx**2 + EPS) + cy**2) <= 1.0) & (
            (cy**2 * (1.0 + ty**2) / (ty**2 + EPS) + cx**2) <= 1.0
        )
    else:
        W = np.ones((Np, Np), dtype=bool)

    H_as = np.sqrt(0j + 1.0 - (FX * lam) ** 2 - (FY * lam) ** 2)
    H_fr = 1.0 - 0.5 * (FX * lam) ** 2 - 0.5 * (FY * lam) ** 2
    delta_H = W * np.exp(1j * k * z * (H_as - H_fr))
    U_pre = np.fft.ifft2(np.fft.fft2(U_un) * delta_H)

    x_un = np.fft.ifftshift(padded_grid["x"])
    X_un = x_un.reshape(1, Np)
    Y_un = x_un.reshape(Np, 1)
    out_dx = float(validity["output_dx_m"])
    out_grid = make_xy_grid(N, out_dx)
    q_un = np.fft.ifftshift(make_xy_grid(Np, out_dx)["x"])
    QX_un = q_un.reshape(1, Np)
    QY_un = q_un.reshape(Np, 1)

    H1 = np.exp(1j * k * (X_un**2 + Y_un**2) / (2.0 * z))
    F = np.fft.fft2(H1 * U_pre)
    amp_scale = dx * dx / (1j * lam * z)
    if skip_final_phase:
        U_full = np.fft.fftshift(amp_scale * F)
    else:
        H2 = np.exp(1j * k * z) * np.exp(1j * k * (QX_un**2 + QY_un**2) / (2.0 * z))
        U_full = np.fft.fftshift(amp_scale * H2 * F)

    U_out = _zero_unpad_center(U_full, U.shape)
    input_power = float(np.sum(np.abs(U) ** 2) * dx * dx)
    full_power = float(np.sum(np.abs(U_full) ** 2) * out_dx * out_dx)
    retained_power = float(np.sum(np.abs(U_out) ** 2) * out_dx * out_dx)
    meta = {
        **validity,
        "method": "sas",
        "bandlimit": bool(bandlimit),
        "skip_final_phase": bool(skip_final_phase),
        "input_power": input_power,
        "full_output_power": full_power,
        "retained_output_power": retained_power,
        "retained_power_fraction": float(retained_power / (full_power + EPS)),
        "full_power_ratio": float(full_power / (input_power + EPS)),
        "returned_power_ratio": float(retained_power / (input_power + EPS)),
        "bandlimit_retained_fraction": float(np.count_nonzero(W) / W.size),
    }
    return U_out, out_grid, meta


def focus_to_focal_plane(
    U_pupil: np.ndarray,
    pupil_grid: dict[str, Any],
    laser: Any,
    objective: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Ideal scalar lens focal-plane field by a single FFT."""

    N = int(pupil_grid["N"])
    dx = float(pupil_grid["dx"])
    dx_f = laser.wavelength_m * objective.f_eff_m / max(N * dx, EPS)
    U_f = fft2c(U_pupil) * (dx * dx) / (1j * laser.wavelength_m * objective.f_eff_m)
    focal_grid = make_xy_grid(N, dx_f)
    return U_f, focal_grid
