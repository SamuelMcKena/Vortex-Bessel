"""vbb_study/viz_fields.py — Reusable field-rendering layer (Stage 6A).

Three renderers, each operating on plain NumPy arrays or engine result dicts.
No engine imports; no physics changes.

  complex_field_image    domain-coloured complex field.
                         HUE = phase via cyclic colourmap (colorcet CET_C2 or
                         matplotlib twilight), VALUE/brightness = normalised
                         amplitude.  Makes ℓ=3 winding visibly wrap 3 times.

  linked_field_views     Four coupled panels from a real engine result dict:
                           ax1 — transverse |U|² at a selectable z,
                           ax2 — transverse domain-coloured phase (surface plane),
                           ax3 — longitudinal x–z intensity slice,
                           ax4 — radial profile + analytic |J_ℓ|² overlay +
                                 measured ring marker.
                         Physical units (µm), origin lower, perceptually-uniform
                         intensity cmap.

  azimuthal_order_panel  Wraps the validated A4 _azimuthal_power_spectrum tool
                         from tests/test_physics_validation.py.
                         Polar plot of ring intensity + bar chart of azimuthal
                         order power with order-6 highlighted (hexagon readiness).

Note on the HTML toy:  ``bessel_beam_simulator.html`` is an idealised analytic
intuition aid and is NOT engine output.  Do not use it for quantitative figures.
"""
from __future__ import annotations

import math
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy import special as sp
from scipy.ndimage import map_coordinates

try:
    import colorcet as cc
    _CYCLIC_CMAP = cc.cm.CET_C2      # perceptually-uniform isoluminant cyclic
    _CYCLIC_CMAP_NAME = "colorcet CET_C2"
except ImportError:
    _CYCLIC_CMAP = plt.get_cmap("twilight")
    _CYCLIC_CMAP_NAME = "twilight"

_TWOPI = 2.0 * math.pi
_UM = 1e-6            # metres per micrometre
_INTENSITY_CMAP = "inferno"

__all__ = [
    "complex_field_image",
    "linked_field_views",
    "azimuthal_order_panel",
    "measured_charge_label",
]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers (also importable for unit tests)
# ─────────────────────────────────────────────────────────────────────────────

def _field_to_rgba(U: np.ndarray) -> np.ndarray:
    """2-D complex field → H×W×4 uint8 RGBA.

    Encoding: phase is mapped to the cyclic colourmap (_CYCLIC_CMAP) so that
    a full 2π phase rotation visits every colour exactly once.  The RGB
    channels are then multiplied by the normalised amplitude (|U|/max|U|),
    making zero-amplitude regions black.

    Convention: phase = −π maps to colourmap position 0; phase = +π maps to
    colourmap position 1 (same colour, completing the cycle).
    """
    U = np.asarray(U, dtype=complex)
    amp = np.abs(U)
    peak = float(amp.max())
    brightness = (amp / peak) if peak > 0.0 else np.zeros_like(amp)   # [0, 1]
    phase_norm = (np.angle(U) + math.pi) / _TWOPI % 1.0               # [0, 1)

    # Evaluate cyclic colourmap at each pixel's phase
    rgba_cmap = _CYCLIC_CMAP(phase_norm)                               # H×W×4 float [0,1]
    rgba_out = rgba_cmap.copy()

    # Multiply RGB by amplitude (alpha stays 1.0)
    rgba_out[..., :3] *= brightness[..., np.newaxis]
    rgba_out[..., 3] = 1.0

    return (rgba_out * 255).astype(np.uint8)


def _azimuthal_power_spectrum(intensities: np.ndarray) -> np.ndarray:
    """One-sided power per azimuthal order — validated A4 tool (re-exported).

    Identical to the function in tests/test_physics_validation.py.
    """
    N = len(intensities)
    spectrum = np.fft.rfft(np.asarray(intensities, dtype=float))
    return np.abs(spectrum) ** 2 / N ** 2


def _radial_profile_4cut(I_2d: np.ndarray, N: int, dx_m: float
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthal average over 4 radial cuts (0°, 45°, 90°, 135°)."""
    n_r = N // 2
    r_um = np.arange(n_r) * dx_m * 1e6
    cx, cy = N // 2, N // 2
    profiles = []
    for ang_deg in (0, 45, 90, 135):
        theta = math.radians(ang_deg)
        rs = np.arange(n_r)
        cols = np.clip((cx + rs * math.cos(theta)).astype(int), 0, N - 1)
        rows = np.clip((cy + rs * math.sin(theta)).astype(int), 0, N - 1)
        profiles.append(np.asarray(I_2d, dtype=float)[rows, cols])
    return r_um, np.mean(profiles, axis=0)


def _add_phase_colorbar(ax: plt.Axes, fig: plt.Figure) -> None:
    """Attach a cyclic phase colourbar to *ax*."""
    norm = mcolors.Normalize(vmin=-math.pi, vmax=math.pi)
    sm = plt.cm.ScalarMappable(cmap=_CYCLIC_CMAP, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Phase (rad)")
    cb.set_ticks([-math.pi, -math.pi / 2, 0.0, math.pi / 2, math.pi])
    cb.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])


def _crop_to_grid(field_2d: np.ndarray, N_full: int, N_crop: int) -> np.ndarray:
    """Centre-crop a N_full×N_full complex field to N_crop×N_crop."""
    centre = N_full // 2
    half = N_crop // 2
    s = slice(centre - half, centre + half)
    return field_2d[s, s]


def _phase_winding(
    field_2d: np.ndarray,
    grid: dict[str, Any],
    sample_radius_m: float,
    n_phi: int = 256,
) -> float:
    """Return phase winding in turns (closed-loop incremental phase accumulation).

    Ported from tests/test_physics_validation.py A3.
    """
    x = np.asarray(grid["x"], dtype=float)
    dx = float(grid["dx"])
    x0 = float(x[0])
    phis = np.linspace(0.0, _TWOPI, n_phi, endpoint=False)
    col = (sample_radius_m * np.cos(phis) - x0) / dx
    row = (sample_radius_m * np.sin(phis) - x0) / dx
    arr = np.asarray(field_2d, dtype=complex)
    E_r = map_coordinates(np.real(arr), [row, col], order=1, mode="nearest")
    E_i = map_coordinates(np.imag(arr), [row, col], order=1, mode="nearest")
    E_ring = E_r + 1j * E_i
    steps = np.angle(np.conj(E_ring[:-1]) * E_ring[1:])
    closing = np.angle(np.conj(E_ring[-1]) * E_ring[0])
    return float(np.sum(steps) + closing) / _TWOPI


def phase_winding(
    field_2d: np.ndarray,
    grid: dict[str, Any],
    sample_radius_m: float,
    n_phi: int = 256,
) -> float:
    """Measure complex-field winding around a closed circular contour."""

    return _phase_winding(field_2d, grid, sample_radius_m, n_phi=n_phi)


def measured_charge_label(
    field_2d: np.ndarray,
    grid: dict[str, Any],
    sample_radius_m: float,
    *,
    design_ell: int,
    conjugate_mode: str | None = None,
    n_phi: int = 256,
) -> str:
    """Measured phase winding as a human-readable label string.

    Charge is noted as preserved when winding ≈ design_ell (within 0.15 turns);
    otherwise the discrepancy is flagged with the conjugate_mode culprit if supplied.
    """
    winding = phase_winding(field_2d, grid, sample_radius_m, n_phi=n_phi)
    if abs(winding - design_ell) < 0.15:
        return (
            f"measured winding = {winding:.2f}"
            f" (design ℓ={design_ell}; ✓ charge preserved)"
        )
    mode_str = (
        f"; SLM2 conjugate_mode='{conjugate_mode}' strips the helical phase"
        if conjugate_mode
        else ""
    )
    return (
        f"measured winding = {winding:.2f}"
        f" (design ℓ={design_ell}; charge stripped{mode_str})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# A — Domain-coloured complex-field image
# ─────────────────────────────────────────────────────────────────────────────

def complex_field_image(
    U: np.ndarray,
    grid: dict[str, Any] | None = None,
    *,
    ax: plt.Axes | None = None,
    title: str = "",
    return_rgba: bool = False,
) -> tuple[plt.Figure, plt.Axes] | tuple[plt.Figure, np.ndarray]:
    """Domain-coloured complex-field image.

    HUE encodes phase via a cyclic perceptually-uniform colourmap; BRIGHTNESS
    encodes normalised amplitude.  An ℓ=3 vortex beam will show exactly 3 full
    colour cycles around any ring centred on the dark core.

    Parameters
    ----------
    U:
        2-D complex array (the transverse field).
    grid:
        Optional engine grid dict with key ``'x'`` (metres) for axis labels.
    ax:
        Optional existing Axes.  A new figure is created if None.
    title:
        Axes title.  Defaults to a standard label.
    return_rgba:
        If True, return ``(fig, rgba_uint8)`` instead of ``(fig, ax)``.
        Use this in unit tests that inspect the rendered pixels directly.

    Returns
    -------
    ``(fig, ax)`` normally; ``(fig, rgba)`` when *return_rgba=True*.
    """
    U = np.asarray(U, dtype=complex)
    rgba = _field_to_rgba(U)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.0), constrained_layout=True)
    else:
        fig = ax.get_figure()

    extent = None
    if grid is not None:
        x = np.asarray(grid["x"], dtype=float) / _UM
        extent = [float(x[0]), float(x[-1]), float(x[0]), float(x[-1])]

    ax.imshow(rgba, origin="lower", extent=extent, interpolation="nearest")
    if extent is not None:
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
    ax.set_title(title or f"Phase (hue · {_CYCLIC_CMAP_NAME}) · amplitude (brightness)")

    _add_phase_colorbar(ax, fig)

    if return_rgba:
        return fig, rgba
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# B — Linked field views (4 panels from a real engine result)
# ─────────────────────────────────────────────────────────────────────────────

def linked_field_views(
    result: dict[str, Any],
    *,
    z_idx: int | None = None,
    fig: plt.Figure | None = None,
    title_prefix: str = "",
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    """Four coupled panels from a real engine result dict.

    Panels
    ------
    ax1 — transverse |U|² intensity at the selected z plane (crop grid).
    ax2 — transverse domain-coloured phase from the surface-plane complex field
          (cropped to the same crop-grid window; labelled "surface plane").
    ax3 — longitudinal x–z intensity slice through the Bessel zone.
    ax4 — azimuthally-averaged radial intensity at peak z, overlaid with the
          analytic |J_ℓ(k_r r)|² prediction (independently computed) and the
          engine-measured ring radius marked.

    All axes use physical µm units with origin at lower-left.

    Parameters
    ----------
    result:
        Engine result dict from ``bt.run_case()``.  Required keys:
        ``surface_field`` (attributes ``Ex``, ``grid``),
        ``volume`` (keys ``z``, ``xz``, ``intensity_stack``, ``crop_grid``,
        ``peak_index``),
        ``design`` (attributes ``ell``, ``kr_sample_m_inv``,
        ``vortex_main_ring_radius_m``),
        ``metrics`` (key ``ring_radius_um``).
    z_idx:
        Z-plane index for panel 1.  Defaults to the engine ``peak_index``.
    fig:
        Optional existing Figure.  A new 4-panel figure is created if None.
    title_prefix:
        Prepended to the figure suptitle.

    Returns
    -------
    ``(fig, (ax_intensity, ax_phase, ax_xz, ax_radial))``
    """
    sf = result["surface_field"]
    vol = result["volume"]
    design = result["design"]
    metrics = result["metrics"]

    # ── Unpack engine arrays ─────────────────────────────────────────────────
    z_arr = np.asarray(vol["z"], dtype=float)
    n_z = len(z_arr)
    peak_idx = int(vol.get("peak_index", 0))
    z_idx = int(np.clip(z_idx if z_idx is not None else peak_idx, 0, n_z - 1))

    crop_grid = vol["crop_grid"]
    N_crop = int(crop_grid["N"])
    dx_crop_m = float(crop_grid["dx"])
    x_crop_um = np.asarray(crop_grid["x"], dtype=float) / _UM
    z_um = z_arr / _UM

    intensity_stack = np.asarray(vol["intensity_stack"], dtype=float)  # (n_z, N_crop, N_crop)
    I_transverse = intensity_stack[z_idx]
    I_peak = intensity_stack[peak_idx]
    xz = np.asarray(vol["xz"], dtype=float)                            # (N_crop, n_z)

    # ── Surface-plane complex field (cropped to crop-grid size) ──────────────
    Ex_full = np.asarray(sf.Ex, dtype=complex)
    N_full = Ex_full.shape[0]
    Ex_crop = _crop_to_grid(Ex_full, N_full, N_crop)

    # ── Design parameters ────────────────────────────────────────────────────
    ell = int(design.ell)
    kr = float(design.kr_sample_m_inv)
    ring_r_um = float(metrics.get("ring_radius_um",
                                   float(design.vortex_main_ring_radius_m) / _UM))

    # ── Build figure ─────────────────────────────────────────────────────────
    if fig is None:
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), constrained_layout=True)
    else:
        axes = fig.get_axes()
    ax1, ax2, ax3, ax4 = axes[:4]

    ext_crop = [float(x_crop_um[0]), float(x_crop_um[-1]),
                float(x_crop_um[0]), float(x_crop_um[-1])]

    # ── Panel 1: transverse intensity ────────────────────────────────────────
    im1 = ax1.imshow(I_transverse, origin="lower", extent=ext_crop,
                     cmap=_INTENSITY_CMAP, interpolation="bilinear")
    fig.colorbar(im1, ax=ax1, label="|U|² [a.u.]", fraction=0.046, pad=0.04)
    ax1.set_xlabel("x (µm)")
    ax1.set_ylabel("y (µm)")
    ax1.set_title(f"|U|²  z = {z_um[z_idx]:.1f} µm")

    # ── Panel 2: domain-coloured phase (surface plane, cropped) ─────────────
    rgba = _field_to_rgba(Ex_crop)
    ax2.imshow(rgba, origin="lower", extent=ext_crop, interpolation="nearest")
    ax2.set_xlabel("x (µm)")
    ax2.set_ylabel("y (µm)")
    ax2.set_title(f"Phase · amplitude  [surface plane, ℓ={ell}]")
    _add_phase_colorbar(ax2, fig)

    # ── Panel 3: longitudinal x–z slice ─────────────────────────────────────
    ext_xz = [float(z_um[0]), float(z_um[-1]),
               float(x_crop_um[0]), float(x_crop_um[-1])]
    im3 = ax3.imshow(xz, origin="lower", extent=ext_xz, cmap=_INTENSITY_CMAP,
                     aspect="auto", interpolation="bilinear")
    fig.colorbar(im3, ax=ax3, label="|U|² [a.u.]", fraction=0.046, pad=0.04)
    ax3.axvline(x=z_um[z_idx], color="cyan", lw=0.9, ls="--",
                label=f"z = {z_um[z_idx]:.0f} µm")
    ax3.set_xlabel("z (µm)")
    ax3.set_ylabel("x (µm)")
    ax3.set_title("Longitudinal x–z")
    ax3.legend(fontsize=7, loc="upper right")

    # ── Panel 4: radial profile + analytic J_ℓ overlay ──────────────────────
    r_um, I_radial = _radial_profile_4cut(I_peak, N_crop, dx_crop_m)
    r_fine_um = np.linspace(0.0, float(r_um[-1]), 4096)
    J_analytic = np.abs(sp.jn(ell, kr * r_fine_um * _UM)) ** 2
    peak_I = float(I_radial.max()) or 1.0
    peak_J = float(J_analytic.max()) or 1.0
    J_norm = J_analytic / peak_J * peak_I

    ax4.plot(r_um, I_radial, color="#0072B2", lw=1.5, label="Engine (4-cut avg)")
    ax4.plot(r_fine_um, J_norm, color="#D55E00", lw=1.0, ls="--",
             label=f"|J_{{{ell}}}(k_r r)|² (analytic)")
    ax4.axvline(x=ring_r_um, color="#009E73", lw=1.2, ls=":",
                label=f"Ring  r = {ring_r_um:.2f} µm")
    ax4.set_xlabel("r (µm)")
    ax4.set_ylabel("|U|² [a.u.]")
    ax4.set_title(f"Radial profile  z = {z_um[peak_idx]:.1f} µm (peak)")
    ax4.legend(fontsize=7)

    # ── Suptitle ─────────────────────────────────────────────────────────────
    tag = f"ℓ = {ell}"
    fig.suptitle(f"{title_prefix}  {tag}".strip() if title_prefix else tag,
                 fontsize=12)
    return fig, (ax1, ax2, ax3, ax4)


# ─────────────────────────────────────────────────────────────────────────────
# C — Azimuthal-order panel (wraps validated A4 tool)
# ─────────────────────────────────────────────────────────────────────────────

def azimuthal_order_panel(
    field_on_ring: np.ndarray,
    *,
    ell: int | None = None,
    n_order_highlight: int = 6,
    title: str = "",
    fig: plt.Figure | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Polar ring plot + azimuthal-order power bar chart.

    Wraps the validated A4 ``_azimuthal_power_spectrum`` tool from
    ``tests/test_physics_validation.py``.

    Parameters
    ----------
    field_on_ring:
        1-D array sampled at equally-spaced azimuthal angles.  Complex values
        are converted to intensity (|·|²) before analysis.
    ell:
        Topological charge (label only; no physics change).
    n_order_highlight:
        Azimuthal order highlighted in the bar chart.  Default 6 flags
        readiness for the hexagonal beam work.
    title:
        Optional figure suptitle.

    Returns
    -------
    ``(fig, (ax_polar, ax_bar))``
    """
    field_on_ring = np.asarray(field_on_ring)
    if np.iscomplexobj(field_on_ring):
        intensities = np.abs(field_on_ring) ** 2
    else:
        intensities = np.asarray(field_on_ring, dtype=float)

    n_phi = len(intensities)
    phis = np.linspace(0.0, _TWOPI, n_phi, endpoint=False)
    power = _azimuthal_power_spectrum(intensities)
    n_orders = len(power)
    total_power = float(np.sum(power)) or 1.0
    frac = power / total_power

    if fig is None:
        fig = plt.figure(figsize=(10, 4.5), constrained_layout=True)
    ax_polar = fig.add_subplot(1, 2, 1, projection="polar")
    ax_bar = fig.add_subplot(1, 2, 2)

    # ── Left: polar ring ─────────────────────────────────────────────────────
    I_norm = intensities / max(float(intensities.max()), 1e-30)
    ax_polar.plot(phis, I_norm, color="#0072B2", lw=1.4)
    ax_polar.fill(phis, I_norm, alpha=0.25, color="#0072B2")
    ax_polar.set_theta_zero_location("E")
    ax_polar.set_theta_direction(1)
    ax_polar.set_title("Ring intensity (azimuthal)", fontsize=9, pad=12)

    # ── Right: power bar chart ────────────────────────────────────────────────
    orders = np.arange(n_orders)
    colors = [
        "#D55E00" if m == n_order_highlight else "#0072B2"
        for m in orders
    ]
    max_show = min(max(20, n_order_highlight + 6), n_orders)
    ax_bar.bar(orders[:max_show], frac[:max_show], color=colors[:max_show], width=0.75)
    if n_order_highlight < n_orders:
        ax_bar.axvline(x=n_order_highlight, color="#D55E00", lw=1.0, ls="--",
                       label=f"m = {n_order_highlight} (hex readiness)")
        ax_bar.legend(fontsize=8)
    ax_bar.set_xlabel("Azimuthal order m")
    ax_bar.set_ylabel("Power fraction")
    ell_tag = f"  (ℓ = {ell})" if ell is not None else ""
    ax_bar.set_title(f"Azimuthal order spectrum{ell_tag}", fontsize=9)
    ax_bar.set_xlim(-0.5, max_show - 0.5)

    if title:
        fig.suptitle(title, fontsize=11)
    return fig, (ax_polar, ax_bar)
