"""Spectral vector Fresnel transmission at a planar +z interface.

The operator preserves ``kx`` and ``ky``, decomposes every incident spectral
component into its local s/p basis, applies electric-field Fresnel
coefficients once, and reconstructs the field in the transmitted-medium
basis.  This is independent of the scalar Phase 2A surface-energy ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from vbb_study.equations.fields import fft2c, ifft2c


TWOPI = 2.0 * np.pi
EPS = np.finfo(float).tiny


@dataclass(frozen=True)
class FresnelInterfaceConfig:
    wavelength_m: float
    n_incident: complex
    n_transmitted: complex
    include_evanescent: bool = False
    normal_direction: Literal["+z"] = "+z"

    def validate(self) -> None:
        if not np.isfinite(self.wavelength_m) or self.wavelength_m <= 0.0:
            raise ValueError("wavelength_m must be finite and positive")
        for value, name in ((self.n_incident, "n_incident"), (self.n_transmitted, "n_transmitted")):
            index = complex(value)
            if not np.isfinite(index.real) or not np.isfinite(index.imag) or index.real <= 0.0:
                raise ValueError(f"{name} must have a finite positive real part")
            if index.imag < 0.0:
                raise ValueError(f"{name} must use the passive Im(n)>=0 convention")
        if self.normal_direction != "+z":
            raise ValueError("the canonical interface normal is +z")


@dataclass
class FresnelInterfaceResult:
    Ex: np.ndarray
    Ey: np.ndarray
    Ez: np.ndarray
    reflected_Ex: np.ndarray | None
    reflected_Ey: np.ndarray | None
    reflected_Ez: np.ndarray | None
    propagating_mask: np.ndarray
    evanescent_mask: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def intensity(self) -> np.ndarray:
        return np.abs(self.Ex) ** 2 + np.abs(self.Ey) ** 2 + np.abs(self.Ez) ** 2


def _outgoing_sqrt(values: np.ndarray) -> np.ndarray:
    """Square root with the outgoing/decaying branch for passive media."""

    root = np.sqrt(np.asarray(values, dtype=np.complex128))
    flip = (np.real(root) < 0.0) | ((np.abs(np.real(root)) <= 1e-14) & (np.imag(root) < 0.0))
    root = np.where(flip, -root, root)
    if np.any(np.real(root) < -1e-12) or np.any(np.imag(root) < -1e-12):
        raise FloatingPointError("could not enforce outgoing kz branch")
    return root


def fresnel_coefficients(
    n_incident: complex,
    n_transmitted: complex,
    theta_incident_rad: float,
) -> dict[str, complex | float]:
    """Return analytic s/p electric-field and power coefficients."""

    n1 = complex(n_incident)
    n2 = complex(n_transmitted)
    theta1 = float(theta_incident_rad)
    sin2 = n1 * np.sin(theta1) / n2
    cos1 = complex(np.cos(theta1))
    cos2 = complex(_outgoing_sqrt(np.asarray([1.0 - sin2 * sin2]))[0])
    ts = 2.0 * n1 * cos1 / (n1 * cos1 + n2 * cos2)
    tp = 2.0 * n1 * cos1 / (n2 * cos1 + n1 * cos2)
    rs = (n1 * cos1 - n2 * cos2) / (n1 * cos1 + n2 * cos2)
    rp = (n2 * cos1 - n1 * cos2) / (n2 * cos1 + n1 * cos2)
    flux = float(np.real(n2 * cos2) / max(float(np.real(n1 * cos1)), EPS))
    return {
        "cos_theta_incident": cos1,
        "cos_theta_transmitted": cos2,
        "t_s": ts,
        "t_p": tp,
        "r_s": rs,
        "r_p": rp,
        "T_s": float(flux * abs(ts) ** 2),
        "T_p": float(flux * abs(tp) ** 2),
        "R_s": float(abs(rs) ** 2),
        "R_p": float(abs(rp) ** 2),
        "snell_residual": float(abs(n1 * np.sin(theta1) - n2 * sin2)),
    }


def _relative_transversality(
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    kz: np.ndarray,
    k_magnitude: complex,
    active: np.ndarray,
) -> float:
    amplitude = np.sqrt(np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2)
    residual = np.abs(kx * ex + ky * ey + kz * ez)
    denominator = abs(k_magnitude) * amplitude
    ratio = np.divide(residual, denominator, out=np.zeros_like(residual, dtype=float), where=denominator > EPS)
    selected = active & (amplitude > 1e-13 * max(float(np.max(amplitude)), EPS))
    return float(np.max(ratio[selected])) if np.any(selected) else 0.0


def transmit_vector_field_planar_interface(
    Ex: np.ndarray,
    Ey: np.ndarray,
    Ez: np.ndarray,
    dx_m: float,
    dy_m: float,
    config: FresnelInterfaceConfig,
) -> FresnelInterfaceResult:
    """Transmit a sampled vector field through a planar dielectric interface."""

    config.validate()
    if not np.isfinite(dx_m) or not np.isfinite(dy_m) or dx_m <= 0.0 or dy_m <= 0.0:
        raise ValueError("dx_m and dy_m must be finite and positive")
    ex = np.asarray(Ex, dtype=np.complex128)
    ey = np.asarray(Ey, dtype=np.complex128)
    ez = np.asarray(Ez, dtype=np.complex128)
    if ex.ndim != 2 or ex.shape != ey.shape or ex.shape != ez.shape:
        raise ValueError("Ex, Ey and Ez must be same-shaped 2D arrays")
    ny, nx = ex.shape
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=float(dx_m)))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=float(dy_m)))
    FX, FY = np.meshgrid(fx, fy, indexing="xy")
    kx = TWOPI * FX
    ky = TWOPI * FY
    kt = np.hypot(kx, ky)
    k0 = TWOPI / float(config.wavelength_m)
    n1 = complex(config.n_incident)
    n2 = complex(config.n_transmitted)
    k1 = n1 * k0
    k2 = n2 * k0
    kz1 = _outgoing_sqrt(k1 * k1 - kt * kt)
    kz2 = _outgoing_sqrt(k2 * k2 - kt * kt)
    tolerance = 1e-12 * max(abs(k1), abs(k2), 1.0)
    incident_propagating = (np.abs(np.imag(kz1)) <= tolerance) & (np.real(kz1) >= -tolerance)
    transmitted_propagating = (np.abs(np.imag(kz2)) <= tolerance) & (np.real(kz2) >= -tolerance)
    propagating = incident_propagating & transmitted_propagating
    evanescent = ~transmitted_propagating

    ax = fft2c(ex)
    ay = fft2c(ey)
    az = fft2c(ez)
    safe_kt = np.where(kt > 0.0, kt, 1.0)
    sx = -ky / safe_kt
    sy = kx / safe_kt
    sx = np.where(kt > 0.0, sx, 0.0)
    sy = np.where(kt > 0.0, sy, 1.0)
    p1x = sy * kz1 / k1
    p1y = -sx * kz1 / k1
    p1z = (sx * ky - sy * kx) / k1
    p2x = sy * kz2 / k2
    p2y = -sx * kz2 / k2
    p2z = (sx * ky - sy * kx) / k2
    es = ax * sx + ay * sy
    ep = ax * p1x + ay * p1y + az * p1z
    cos1 = kz1 / k1
    cos2 = kz2 / k2
    ts = 2.0 * n1 * cos1 / (n1 * cos1 + n2 * cos2)
    tp = 2.0 * n1 * cos1 / (n2 * cos1 + n1 * cos2)
    rs = (n1 * cos1 - n2 * cos2) / (n1 * cos1 + n2 * cos2)
    rp = (n2 * cos1 - n1 * cos2) / (n2 * cos1 + n1 * cos2)
    active = np.ones_like(propagating, dtype=bool) if config.include_evanescent else propagating
    tx = np.where(active, ts * es * sx + tp * ep * p2x, 0.0)
    ty = np.where(active, ts * es * sy + tp * ep * p2y, 0.0)
    tz = np.where(active, tp * ep * p2z, 0.0)

    krz = -kz1
    prx = sy * krz / k1
    pry = -sx * krz / k1
    prz = (sx * ky - sy * kx) / k1
    rx = np.where(incident_propagating, rs * es * sx + rp * ep * prx, 0.0)
    ry = np.where(incident_propagating, rs * es * sy + rp * ep * pry, 0.0)
    rz = np.where(incident_propagating, rp * ep * prz, 0.0)

    incident_flux_density = np.maximum(np.real(n1 * cos1), 0.0) * (np.abs(es) ** 2 + np.abs(ep) ** 2)
    transmitted_flux_density = np.maximum(np.real(n2 * cos2), 0.0) * (
        np.abs(ts * es) ** 2 + np.abs(tp * ep) ** 2
    )
    reflected_flux_density = np.maximum(np.real(n1 * cos1), 0.0) * (
        np.abs(rs * es) ** 2 + np.abs(rp * ep) ** 2
    )
    incident_flux = float(np.sum(np.where(incident_propagating, incident_flux_density, 0.0)))
    transmitted_flux = float(np.sum(np.where(propagating, transmitted_flux_density, 0.0)))
    reflected_flux = float(np.sum(np.where(incident_propagating, reflected_flux_density, 0.0)))
    evanescent_incident_flux = float(
        np.sum(np.where(incident_propagating & ~transmitted_propagating, incident_flux_density, 0.0))
    )
    t_fraction = transmitted_flux / max(incident_flux, EPS)
    r_fraction = reflected_flux / max(incident_flux, EPS)
    theta1 = np.arctan2(kt, np.maximum(np.real(kz1), 0.0))
    weighted_theta = float(
        np.sum(np.where(incident_propagating, incident_flux_density * theta1, 0.0)) / max(incident_flux, EPS)
    )
    maximum_angle_power_threshold = 1e-10
    active_incident = incident_propagating & (
        incident_flux_density
        > maximum_angle_power_threshold * max(float(np.max(incident_flux_density)), EPS)
    )
    max_theta = float(np.max(theta1[active_incident])) if np.any(active_incident) else 0.0
    s_flux = float(np.sum(np.where(incident_propagating, np.maximum(np.real(n1 * cos1), 0.0) * np.abs(es) ** 2, 0.0)))
    p_flux = float(np.sum(np.where(incident_propagating, np.maximum(np.real(n1 * cos1), 0.0) * np.abs(ep) ** 2, 0.0)))
    t_residual = _relative_transversality(tx, ty, tz, kx, ky, kz2, k2, active)
    i_residual = _relative_transversality(ax, ay, az, kx, ky, kz1, k1, incident_propagating)
    identity = bool(abs(n1 - n2) <= 1e-14 * max(abs(n1), abs(n2), 1.0))
    if identity:
        out_ex, out_ey, out_ez = ex.copy(), ey.copy(), ez.copy()
        ref_ex = np.zeros_like(ex)
        ref_ey = np.zeros_like(ey)
        ref_ez = np.zeros_like(ez)
        t_fraction, r_fraction = 1.0, 0.0
    else:
        out_ex, out_ey, out_ez = ifft2c(tx), ifft2c(ty), ifft2c(tz)
        ref_ex, ref_ey, ref_ez = ifft2c(rx), ifft2c(ry), ifft2c(rz)
    diagnostics: dict[str, Any] = {
        "method": "centred_fft_local_sp_vector_fresnel",
        "phase_convention": "exp(+i kz z - i omega t)",
        "kz_branch": "Re(kz)>=0 and Im(kz)>=0",
        "normal_incidence_basis": "s=+y, p=+x for the incident +z wave",
        "electric_field_coefficients_applied_once": True,
        "phase2a_energy_ledger_factor_applied": False,
        "transverse_wavevector_preserved": True,
        "incident_flux_arb": incident_flux,
        "transmitted_flux_arb": transmitted_flux,
        "reflected_flux_arb": reflected_flux,
        "transmitted_power_fraction": float(t_fraction),
        "reflected_power_fraction": float(r_fraction),
        "lossless_R_plus_T": float(r_fraction + t_fraction),
        "s_incident_power_fraction": float(s_flux / max(s_flux + p_flux, EPS)),
        "p_incident_power_fraction": float(p_flux / max(s_flux + p_flux, EPS)),
        "evanescent_incident_power_fraction": float(evanescent_incident_flux / max(incident_flux, EPS)),
        "maximum_incidence_angle_rad": max_theta,
        "maximum_incidence_power_support_relative_threshold": maximum_angle_power_threshold,
        "mean_power_weighted_incidence_angle_rad": weighted_theta,
        "incident_transversality_residual": i_residual,
        "transmitted_transversality_residual": t_residual,
        "incident_propagating_bin_count": int(np.count_nonzero(incident_propagating)),
        "transmitted_propagating_bin_count": int(np.count_nonzero(propagating)),
        "evanescent_bin_count": int(np.count_nonzero(evanescent)),
        "physically_incident_air_bins_marked_tir": int(
            np.count_nonzero(incident_propagating & ~transmitted_propagating) if n1.real <= n2.real else 0
        ),
        "normal_index_identity": identity,
    }
    return FresnelInterfaceResult(
        Ex=out_ex,
        Ey=out_ey,
        Ez=out_ez,
        reflected_Ex=ref_ex,
        reflected_Ey=ref_ey,
        reflected_Ez=ref_ez,
        propagating_mask=propagating,
        evanescent_mask=evanescent,
        diagnostics=diagnostics,
    )


__all__ = [
    "FresnelInterfaceConfig",
    "FresnelInterfaceResult",
    "fresnel_coefficients",
    "transmit_vector_field_planar_interface",
]
