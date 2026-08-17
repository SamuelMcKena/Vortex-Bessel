"""Nathan-style reconstructed source-model port (V0A) for the V0 parity control.

This is a project re-implementation of the vector-axicon algorithm published in
Nathan Marco's report appendix (``Laser_Manufacturing.pdf``, Listing 1: grid,
``make_segmented_ra_input``, ``apply_axicon`` with Fresnel s/p split, and
``propagate_vector_asm`` with the ``P = I - ss^T`` transversality projection).  It
is *not* execution of Nathan's original ``.py`` file; it is a reconstruction that
has been validated to reproduce his verbatim appendix code to machine precision
for the Figure-4 hexagon case (see docs/53).

This module is intentionally isolated from the Digital Twin path.  It does not
import the project vector axicon, ObjectiveMap, focusing, or Nathan wrapper
helpers.  It exists so V0 can compare this reconstructed source-style port (V0A)
against the current project implementation with identical source parameters
(V0B).  Both V0 paths use Nathan's axis-sampled grid centring (``make_source_grid``);
the reason this matters is documented in docs/53.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

EPS = 1.0e-30
TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class NathanLiteralSourceConfig:
    """Nathan Figure-4 source-scale parameters."""

    wavelength_m: float = 1030e-9
    beam_radius_m: float = 2.0e-3
    axicon_n: float = 1.458
    medium_n: float = 1.0
    axicon_apex_angle_deg: float = 176.0
    grid_n: int = 1024
    window_m: float = 10.0e-3
    n_pairs: int = 3
    sector_theta_rad: float = np.pi / 3.0
    sector_rotation_rad: float = 0.0
    z_start_m: float = 0.1e-3
    z_end_m: float = 290.0e-3
    z_reference_m: float = 60.0e-3
    z_planes: int = 61

    @property
    def axicon_base_angle_rad(self) -> float:
        return 0.5 * np.deg2rad(max(0.0, 180.0 - float(self.axicon_apex_angle_deg)))


def make_source_grid(grid_n: int, window_m: float) -> dict[str, Any]:
    """Return Nathan's square source grid with centred FFT frequencies.

    The axis is sampled exactly (``x = -L + arange*dx``, so ``x=0`` lands on the
    index ``n//2`` sample), matching Nathan's appendix ``make_space_grid``.  This
    on-axis sampling is required: on a zero-straddling grid (``arange - n/2 + 0.5``)
    the r=0 radial/azimuthal polarisation singularity is represented by four
    uncancelled near-axis pixels, which inject a spurious bright on-axis core into
    the propagated segmented field (see docs/53).  Nathan's centred convention
    preserves the analytic m=0 cancellation, giving the reported dark-core
    hexagonal Bessel beam.
    """

    n = int(grid_n)
    dx = float(window_m) / max(n, 1)
    x = (np.arange(n, dtype=float) - n // 2) * dx
    X, Y = np.meshgrid(x, x, indexing="xy")
    fx = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    FX, FY = np.meshgrid(fx, fx, indexing="xy")
    return {
        "N": n,
        "dx": dx,
        "x": x,
        "y": x,
        "X": X,
        "Y": Y,
        "R": np.hypot(X, Y),
        "PHI": np.arctan2(Y, X),
        "FX": FX,
        "FY": FY,
    }


def z_values(config: NathanLiteralSourceConfig) -> np.ndarray:
    """Return z planes with the reference plane inserted exactly."""

    z = np.linspace(float(config.z_start_m), float(config.z_end_m), max(2, int(config.z_planes)))
    ref = float(config.z_reference_m)
    if float(np.min(z)) - EPS <= ref <= float(np.max(z)) + EPS and not np.any(np.isclose(z, ref, rtol=0.0, atol=1e-15)):
        z = np.sort(np.append(z, ref))
    return z.astype(float)


def make_segmented_ra_input(
    grid: Mapping[str, Any],
    config: NathanLiteralSourceConfig,
) -> dict[str, np.ndarray]:
    """Literal three-pair alternating radial/azimuthal source field."""

    theta = np.asarray(grid["PHI"], dtype=float)
    R = np.asarray(grid["R"], dtype=float)
    cell_angle = TWOPI / float(config.n_pairs)
    local = np.mod(theta - float(config.sector_rotation_rad), cell_angle)
    radial_mask = local >= (cell_angle - float(config.sector_theta_rad))
    phi0_map = np.where(radial_mask, 0.0, 0.5 * np.pi)
    gaussian = np.exp(-(R**2) / max(float(config.beam_radius_m), EPS) ** 2)
    return {
        "ex": (gaussian * np.cos(theta + phi0_map)).astype(np.complex128),
        "ey": (gaussian * np.sin(theta + phi0_map)).astype(np.complex128),
        "ez": np.zeros_like(R, dtype=np.complex128),
        "radial_mask": radial_mask,
        "gaussian": gaussian,
    }


def _fft2c(values: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(values)))


def _ifft2c(values: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(values)))


def _fresnel_sp_amplitudes(n_axicon: float, n_medium: float, beta_rad: float) -> tuple[complex, complex, complex]:
    n_ax = float(n_axicon)
    n_med = float(n_medium)
    beta = float(beta_rad)
    cos_i = np.cos(beta)
    t_entry = 2.0 * n_med / (n_med + n_ax)
    sin_t = (n_ax / n_med) * np.sin(beta) + 0j
    cos_t = np.sqrt(1.0 - sin_t * sin_t)
    if np.real(cos_t) < 0.0:
        cos_t = -cos_t
    if np.imag(cos_t) < 0.0:
        cos_t = -cos_t
    t_s = 2.0 * n_ax * cos_i / (n_ax * cos_i + n_med * cos_t)
    t_p = 2.0 * n_ax * cos_i / (n_med * cos_i + n_ax * cos_t)
    return complex(t_entry), complex(t_p), complex(t_s)


def apply_source_axicon(
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    grid: Mapping[str, Any],
    config: NathanLiteralSourceConfig,
) -> dict[str, np.ndarray | dict[str, Any]]:
    """Apply source-scale vector axicon phase and local p/s treatment."""

    R = np.asarray(grid["R"], dtype=float)
    phi = np.asarray(grid["PHI"], dtype=float)
    er_x = np.cos(phi)
    er_y = np.sin(phi)
    ep_x = -np.sin(phi)
    ep_y = np.cos(phi)
    er = ex * er_x + ey * er_y
    ephi = ex * ep_x + ey * ep_y
    t_entry, t_p, t_s = _fresnel_sp_amplitudes(config.axicon_n, config.medium_n, config.axicon_base_angle_rad)
    k_r = TWOPI / float(config.wavelength_m) * (float(config.axicon_n) - float(config.medium_n)) * np.tan(config.axicon_base_angle_rad)
    phase = np.exp(-1j * abs(k_r) * R)
    er_out = t_entry * t_p * er * phase
    ephi_out = t_entry * t_s * ephi * phase
    return {
        "ex": er_out * er_x + ephi_out * ep_x,
        "ey": er_out * er_y + ephi_out * ep_y,
        "ez": ez * phase * t_entry * 0.5 * (t_p + t_s),
        "metadata": {
            "n_axicon": float(config.axicon_n),
            "n_medium": float(config.medium_n),
            "base_angle_rad": float(config.axicon_base_angle_rad),
            "base_angle_deg": float(np.rad2deg(config.axicon_base_angle_rad)),
            "k_r_m_inv": float(k_r),
            "t_entry_abs": float(abs(t_entry)),
            "t_p_abs": float(abs(t_p)),
            "t_s_abs": float(abs(t_s)),
        },
    }


def propagate_source_vector_asm(
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    grid: Mapping[str, Any],
    config: NathanLiteralSourceConfig,
    z_planes_m: Sequence[float],
) -> np.ndarray:
    """Vector angular-spectrum propagation with one transversality projection."""

    k = TWOPI * float(config.medium_n) / float(config.wavelength_m)
    kx = TWOPI * np.asarray(grid["FX"], dtype=float)
    ky = TWOPI * np.asarray(grid["FY"], dtype=float)
    kz = np.sqrt((k * k - kx * kx - ky * ky) + 0j)
    kz = np.where(np.imag(kz) < 0.0, -kz, kz)
    ax = _fft2c(ex)
    ay = _fft2c(ey)
    az = _fft2c(ez)
    sx = kx / max(float(k), EPS)
    sy = ky / max(float(k), EPS)
    sz = kz / max(float(k), EPS)
    dot = sx * ax + sy * ay + sz * az
    ax_p = ax - sx * dot
    ay_p = ay - sy * dot
    az_p = az - sz * dot
    z_arr = np.asarray(z_planes_m, dtype=float)
    stack = np.empty((z_arr.size,) + np.asarray(ex).shape, dtype=np.float32)
    for idx, z_m in enumerate(z_arr):
        transfer = np.exp(1j * kz * float(z_m))
        ex_z = _ifft2c(ax_p * transfer)
        ey_z = _ifft2c(ay_p * transfer)
        ez_z = _ifft2c(az_p * transfer)
        stack[idx] = (np.abs(ex_z) ** 2 + np.abs(ey_z) ** 2 + np.abs(ez_z) ** 2).astype(np.float32)
    return stack


def run_literal_source_port(config: NathanLiteralSourceConfig | None = None) -> dict[str, Any]:
    """Run V0A: literal Nathan source-port propagation."""

    cfg = config or NathanLiteralSourceConfig()
    grid = make_source_grid(cfg.grid_n, cfg.window_m)
    source = make_segmented_ra_input(grid, cfg)
    axicon = apply_source_axicon(source["ex"], source["ey"], source["ez"], grid, cfg)
    z = z_values(cfg)
    stack = propagate_source_vector_asm(axicon["ex"], axicon["ey"], axicon["ez"], grid, cfg, z)
    reference_index = int(np.argmin(np.abs(z - float(cfg.z_reference_m))))
    return {
        "config": asdict(cfg),
        "grid": grid,
        "source": source,
        "axicon": axicon["metadata"],
        "z_values_m": z,
        "reference_index": reference_index,
        "intensity_stack": stack,
    }


__all__ = [
    "NathanLiteralSourceConfig",
    "apply_source_axicon",
    "make_segmented_ra_input",
    "make_source_grid",
    "propagate_source_vector_asm",
    "run_literal_source_port",
    "z_values",
]
