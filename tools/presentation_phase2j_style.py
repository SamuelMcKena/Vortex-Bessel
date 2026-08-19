"""Shared Phase 2J presentation-rendering style.

This module changes presentation rendering only. It does not change optical
fields, propagation, calibration assumptions, or scientific claim boundaries.

The intensity palette is deliberately *not* Matplotlib ``inferno``. The first
Phase 2J pass still looked purple/blue at low intensity, which was not the
requested presentation look. ``PHASE2J_THERMAL`` is an explicit monotonic
black -> deep red -> red -> orange -> amber -> yellow palette with no blue,
cyan, green or purple segment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

EPS = np.finfo(float).tiny

FIG_BG = "#05070a"
AX_BG = "#07090c"
TEXT = "#f4f4f1"
MUTED = "#c8ced3"
GRID = "#30363d"
RED = "#ff453a"
GREEN = "#39d6ad"
GOLD = "#f2c14e"
BORDER = "#334252"

# Exact requested intensity look: black -> red -> orange -> yellow.
# Keep this palette explicit so a future Matplotlib default cannot silently
# reintroduce purple/blue tones.
THERMAL_HEX = (
    "#000000",
    "#180000",
    "#4d0000",
    "#8f0800",
    "#cf1d00",
    "#f04400",
    "#ff7600",
    "#ffab00",
    "#ffd13a",
    "#fff176",
)
CMAP_NAME = "phase2j_thermal"
CMAP = LinearSegmentedColormap.from_list(CMAP_NAME, THERMAL_HEX, N=256)
SAVE_DPI = 480
DISPLAY_INTERPOLATION = "lanczos"


def normalise(values: np.ndarray, peak: float | None = None) -> np.ndarray:
    arr = np.maximum(np.asarray(values, dtype=float), 0.0)
    scale = float(np.max(arr)) if peak is None else float(peak)
    return arr / max(scale, EPS)


def style_fig(fig: plt.Figure) -> None:
    fig.patch.set_facecolor(FIG_BG)


def style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(AX_BG)
    for spine in ax.spines.values():
        spine.set_color("#aeb7bf")
        spine.set_linewidth(0.75)
    ax.tick_params(colors=MUTED, labelsize=9, length=2.5)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    ax.grid(color=GRID, alpha=0.28, linewidth=0.45)


def save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def draw_xy(
    ax: plt.Axes,
    values: np.ndarray,
    extent_mm: Sequence[float],
    title: str,
    *,
    peak: float | None = None,
    ylabel: bool = False,
    show_y: bool | None = None,
) -> None:
    if show_y is not None:
        ylabel = bool(show_y)
    style_ax(ax)
    ax.imshow(
        normalise(values, peak),
        origin="lower",
        extent=list(map(float, extent_mm)),
        cmap=CMAP,
        vmin=0.0,
        vmax=1.0,
        interpolation=DISPLAY_INTERPOLATION,
        aspect="equal",
    )
    ax.set_title(title, fontsize=13.0, weight="bold", pad=7)
    ax.set_xlabel("x (mm)", fontsize=9)
    if ylabel:
        ax.set_ylabel("y (mm)", fontsize=9)
    else:
        ax.tick_params(labelleft=False)
    ax.axhline(0.0, color="white", alpha=0.12, linewidth=0.42)
    ax.axvline(0.0, color="white", alpha=0.12, linewidth=0.42)


def draw_xz(
    ax: plt.Axes,
    values: np.ndarray,
    coord_m: np.ndarray,
    z_values_m: np.ndarray,
    *,
    peak: float | None = None,
    ylabel: bool = True,
    show_y: bool | None = None,
    z_ref_m: float = 60e-3,
) -> None:
    if show_y is not None:
        ylabel = bool(show_y)
    style_ax(ax)
    ax.imshow(
        normalise(np.asarray(values, dtype=float).T, peak),
        origin="lower",
        extent=[
            float(z_values_m[0]) * 1e3,
            float(z_values_m[-1]) * 1e3,
            float(coord_m[0]) * 1e3,
            float(coord_m[-1]) * 1e3,
        ],
        cmap=CMAP,
        vmin=0.0,
        vmax=1.0,
        interpolation=DISPLAY_INTERPOLATION,
        aspect="auto",
    )
    ax.set_xlabel("z from axicon (mm)", fontsize=9)
    if ylabel:
        ax.set_ylabel("x at fixed y=0 (mm)", fontsize=9)
    else:
        ax.tick_params(labelleft=False)
    ax.axhline(0.0, color="white", alpha=0.12, linewidth=0.42)
    ax.axvline(float(z_ref_m) * 1e3, color="white", alpha=0.28, linestyle="--", linewidth=0.7)


def presentation_crop_halfwidth(requested_halfwidth_m: float) -> float:
    """Map legacy presentation crops to visibly tighter Phase 2J framing.

    The crop is display framing only. It never changes the propagated field.
    Wide axicon-plane illumination panels remain wide because they show pupil
    loading rather than the output beam itself.
    """
    h = float(requested_halfwidth_m)
    if h >= 2.0e-3:
        return h
    if h >= 0.8e-3:
        return 0.52e-3  # decentre / realistic-error output fingerprints
    if h >= 0.34e-3:
        return 0.24e-3  # non-ideal tip output
    if h >= 0.30e-3:
        return 0.18e-3  # ideal output / route output
    return h


def validate_palette_has_no_cool_segment() -> None:
    """Fail CI if the intensity palette ever regains blue/cyan/green segments."""
    samples = np.asarray(CMAP(np.linspace(0.0, 1.0, 256)))[:, :3]
    # The thermal palette may have tiny blue in the pale-yellow endpoint, but
    # blue must never dominate red and green must never dominate red by a large
    # margin. This rejects turbo/viridis/inferno-like cool low-intensity bands.
    if np.any(samples[:, 2] > samples[:, 0] + 1e-9):
        raise RuntimeError("Phase 2J thermal palette contains a blue-dominant segment")
    if np.any(samples[:, 1] > samples[:, 0] + 0.08):
        raise RuntimeError("Phase 2J thermal palette contains a green/cyan-dominant segment")


# Presentation-only touch: forces the nominal-constraint workflow to rebuild
# figure 03 with the corrected physical-stage renderer introduced on 2026-08-19.
