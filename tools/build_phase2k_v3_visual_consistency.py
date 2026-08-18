"""Render a visually consistent V3 figure set for presentation use.

This is a presentation-only renderer that keeps the underlying numerical data
unchanged while exporting a harmonised visual family:

* transverse V3 intensity (2D)
* transverse V3 intensity (3D surface)
* ideal fixed-lab V3 propagation (2D x-z)
* ideal fixed-lab V3 propagation (3D landscape)

All outputs share:
* the same square canvas size
* the same thermal colour map
* consistent normalisation styling
* dark presentation background

No synthetic data are introduced; all panels are rendered directly from the
current repository optical model.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import presentation_phase2j_style as style
from vbb_study.digital_twin.vortex_continuous_propagation import (
    build_fixed_plane_longitudinal_map,
    build_fixed_support_spectrum,
    native_field_at_z,
)
from vbb_study.digital_twin.vortex_system_route import build_system_route

Z_REF_M = 60.0e-3
Z_VALUES_M = np.linspace(5.0e-3, 120.0e-3, 120)
PROP_COORD_M = np.linspace(-0.18e-3, 0.18e-3, 601)
XY_HALFWIDTH_M = 0.235e-3
CANVAS_SIZE = (8.0, 8.0)
CMAP = style.CMAP
TEXT = style.TEXT
MUTED = style.MUTED
FIG_BG = style.FIG_BG
GAMMA = 0.55
DPI = 520
DISPLAY_INTERPOLATION = getattr(style, "DISPLAY_INTERPOLATION", "nearest")


def _normalise(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    peak = float(np.max(arr))
    if peak <= 0.0:
        return np.zeros_like(arr)
    return arr / peak


def _propagator(route: Mapping[str, Any], z_max_m: float = 120e-3):
    return build_fixed_support_spectrum(
        np.asarray(route["post_axicon"], dtype=np.complex128),
        dict(route["grid"]),
        wavelength_m=float(route["metadata"]["wavelength_m"]),
        z_max_m=float(z_max_m),
        minimum_retained_spectral_power=0.995,
    )


def _square_crop(values: np.ndarray, grid: Mapping[str, Any], halfwidth_m: float):
    x = np.asarray(grid["x"], dtype=float)
    ids = np.flatnonzero(np.abs(x) <= float(halfwidth_m))
    if ids.size < 70:
        raise RuntimeError(
            f"presentation crop contains only {ids.size} native samples; increase grid_n"
        )
    return np.asarray(values)[np.ix_(ids, ids)], x[ids]


def _clean_3d_axis(ax: Any) -> None:
    ax.set_facecolor(FIG_BG)
    ax.tick_params(colors=MUTED, labelsize=8, pad=0)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.zaxis.label.set_color(MUTED)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((*colors.to_rgb(FIG_BG), 0.0))
        axis.pane.set_edgecolor((*colors.to_rgb(MUTED), 0.10))
        try:
            axis._axinfo["grid"]["color"] = (*colors.to_rgb(MUTED), 0.10)
            axis._axinfo["grid"]["linewidth"] = 0.42
        except Exception:
            pass


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep exact exported pixel dimensions identical for the whole set.  Do
    # not use bbox_inches='tight' here because that changes final canvas size.
    fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())


def _square_2d_axis(fig: plt.Figure) -> plt.Axes:
    ax = fig.add_axes([0.12, 0.17, 0.76, 0.76])
    style.style_ax(ax)
    return ax


def render_v3_xy_2d(output_dir: Path, grid_n: int) -> Path:
    route = build_system_route("V3", grid_n=int(grid_n))
    prop = _propagator(route, Z_REF_M)
    field = np.asarray(native_field_at_z(prop, Z_REF_M), dtype=np.complex128)
    intensity = _normalise(np.abs(field) ** 2)
    crop, xc = _square_crop(intensity, route["grid"], XY_HALFWIDTH_M)
    extent = [xc[0] * 1e3, xc[-1] * 1e3, xc[0] * 1e3, xc[-1] * 1e3]
    norm = colors.PowerNorm(gamma=GAMMA, vmin=0.0, vmax=1.0)

    fig = plt.figure(figsize=CANVAS_SIZE, facecolor=FIG_BG)
    ax = _square_2d_axis(fig)
    im = ax.imshow(
        crop,
        origin="lower",
        extent=extent,
        cmap=CMAP,
        norm=norm,
        interpolation=DISPLAY_INTERPOLATION,
        aspect="equal",
    )
    ax.set_title("V3 vortex–Bessel intensity  |  z = 60 mm", color=TEXT, fontsize=15.5, weight="bold", pad=12)
    ax.set_xlabel("x (mm)", fontsize=9)
    ax.set_ylabel("y (mm)", fontsize=9)
    ax.axhline(0.0, color="white", alpha=0.10, linewidth=0.45)
    ax.axvline(0.0, color="white", alpha=0.10, linewidth=0.45)
    cax = fig.add_axes([0.18, 0.08, 0.56, 0.020])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(colors=MUTED, labelsize=7, length=2)
    cb.outline.set_edgecolor((*colors.to_rgb(MUTED), 0.28))
    cb.set_label("normalised intensity", color=MUTED, fontsize=8, labelpad=2)
    out = output_dir / "00_V3_xy_intensity_square.png"
    _savefig(fig, out)
    plt.close(fig)
    del route, prop, field, intensity, crop
    gc.collect()
    return out


def render_v3_xy_3d(output_dir: Path, grid_n: int) -> Path:
    route = build_system_route("V3", grid_n=int(grid_n))
    prop = _propagator(route, Z_REF_M)
    field = np.asarray(native_field_at_z(prop, Z_REF_M), dtype=np.complex128)
    intensity = np.abs(field) ** 2
    crop, xc = _square_crop(intensity, route["grid"], XY_HALFWIDTH_M)
    crop = _normalise(crop)

    stride = max(1, int(np.ceil(max(crop.shape) / 300)))
    surface_i = crop[::stride, ::stride]
    x_mm = xc[::stride] * 1e3
    X, Y = np.meshgrid(x_mm, x_mm)
    norm = colors.PowerNorm(gamma=GAMMA, vmin=0.0, vmax=1.0)

    fig = plt.figure(figsize=CANVAS_SIZE, facecolor=FIG_BG)
    ax = fig.add_subplot(111, projection="3d")
    _clean_3d_axis(ax)
    surf = ax.plot_surface(
        X, Y, surface_i, cmap=CMAP, norm=norm,
        rstride=1, cstride=1, linewidth=0.0, antialiased=True,
        shade=False, alpha=0.99, rasterized=True,
    )
    floor = -0.07
    ax.contour(
        X, Y, surface_i, zdir="z", offset=floor,
        levels=[0.018, 0.035, 0.065, 0.11, 0.18, 0.30, 0.48, 0.72],
        cmap=CMAP, norm=norm, linewidths=0.8, alpha=0.52,
    )
    ax.set_zlim(floor, 1.03)
    ax.set_box_aspect((1.0, 1.0, 0.47))
    ax.view_init(elev=27, azim=-56)
    ax.set_xlabel("x (mm)", labelpad=7, fontsize=9)
    ax.set_ylabel("y (mm)", labelpad=7, fontsize=9)
    ax.set_zlabel("normalised intensity", labelpad=6, fontsize=9)
    ax.set_title("V3 vortex–Bessel intensity  |  z = 60 mm", color=TEXT, fontsize=15.5, weight="bold", pad=18)

    cax = fig.add_axes([0.18, 0.08, 0.56, 0.020])
    cb = fig.colorbar(surf, cax=cax, orientation="horizontal")
    cb.ax.tick_params(colors=MUTED, labelsize=7, length=2)
    cb.outline.set_edgecolor((*colors.to_rgb(MUTED), 0.28))
    cb.set_label("normalised intensity", color=MUTED, fontsize=8, labelpad=2)

    out = output_dir / "01_V3_3d_intensity_square.png"
    _savefig(fig, out)
    plt.close(fig)
    del route, prop, field, intensity, crop, surface_i
    gc.collect()
    return out


def _build_v3_xz(grid_n: int) -> np.ndarray:
    route = build_system_route("V3", grid_n=int(grid_n))
    prop = _propagator(route, float(Z_VALUES_M[-1]))
    mapped = build_fixed_plane_longitudinal_map(
        prop,
        z_values_m=Z_VALUES_M,
        x_coordinates_m=PROP_COORD_M,
        y_coordinates_m=PROP_COORD_M,
        fixed_x_m=0.0,
        fixed_y_m=0.0,
        source_label="phase2k-v3-visual-consistency-fixed-lab",
    )
    xz = _normalise(np.asarray(mapped.xz_intensity, dtype=float))
    del route, prop, mapped
    gc.collect()
    return xz


def render_v3_xz_2d(output_dir: Path, grid_n: int) -> Path:
    xz = _build_v3_xz(grid_n)
    x_mm = PROP_COORD_M * 1e3
    z_mm = Z_VALUES_M * 1e3
    norm = colors.PowerNorm(gamma=GAMMA, vmin=0.0, vmax=1.0)

    fig = plt.figure(figsize=CANVAS_SIZE, facecolor=FIG_BG)
    ax = fig.add_axes([0.12, 0.19, 0.76, 0.70])
    style.style_ax(ax)
    image = ax.imshow(
        xz.T,
        origin="lower",
        extent=[z_mm[0], z_mm[-1], x_mm[0], x_mm[-1]],
        cmap=CMAP,
        norm=norm,
        interpolation=DISPLAY_INTERPOLATION,
        aspect="auto",
    )
    ax.set_title("Ideal V3 vortex–Bessel propagation", color=TEXT, fontsize=15.5, weight="bold", pad=12)
    ax.set_xlabel("z from physical axicon (mm)", fontsize=9)
    ax.set_ylabel("x at fixed y = 0 (mm)", fontsize=9)
    ax.axhline(0.0, color="white", alpha=0.10, linewidth=0.45)
    ax.axvline(Z_REF_M * 1e3, color="white", alpha=0.24, linestyle="--", linewidth=0.8)
    ax.text(Z_REF_M * 1e3 + 2.0, x_mm[-1] - 0.012, "z = 60 mm", color=MUTED, fontsize=8.2, rotation=90, va="top")
    cax = fig.add_axes([0.18, 0.08, 0.56, 0.020])
    cb = fig.colorbar(image, cax=cax, orientation="horizontal")
    cb.ax.tick_params(colors=MUTED, labelsize=7, length=2)
    cb.outline.set_edgecolor((*colors.to_rgb(MUTED), 0.28))
    cb.set_label("normalised intensity", color=MUTED, fontsize=8, labelpad=2)
    out = output_dir / "02_V3_ideal_propagation_xz_square.png"
    _savefig(fig, out)
    plt.close(fig)
    del xz
    gc.collect()
    return out


def render_v3_xz_3d(output_dir: Path, grid_n: int) -> Path:
    xz = _build_v3_xz(grid_n)
    x_mm = PROP_COORD_M * 1e3
    z_mm = Z_VALUES_M * 1e3
    norm = colors.PowerNorm(gamma=GAMMA, vmin=0.0, vmax=1.0)
    stride_z = 2
    stride_x = 3
    Z, X = np.meshgrid(z_mm[::stride_z], x_mm[::stride_x], indexing="ij")
    I = xz[::stride_z, ::stride_x]

    fig = plt.figure(figsize=CANVAS_SIZE, facecolor=FIG_BG)
    ax = fig.add_subplot(111, projection="3d")
    _clean_3d_axis(ax)
    surf = ax.plot_surface(
        Z, X, I, cmap=CMAP, norm=norm,
        rstride=1, cstride=1, linewidth=0.0, antialiased=True,
        shade=False, alpha=0.985, rasterized=True,
    )
    ax.set_xlabel("z from axicon (mm)", labelpad=8, fontsize=9)
    ax.set_ylabel("x (mm)", labelpad=8, fontsize=9)
    ax.set_zlabel("normalised intensity", labelpad=6, fontsize=9)
    ax.set_zlim(0.0, 1.02)
    ax.set_box_aspect((1.15, 0.88, 0.46))
    ax.view_init(elev=29, azim=-64)
    ax.set_title("Ideal V3 propagation landscape", color=TEXT, fontsize=15.5, weight="bold", pad=16)
    cax = fig.add_axes([0.18, 0.08, 0.56, 0.020])
    cb = fig.colorbar(surf, cax=cax, orientation="horizontal")
    cb.ax.tick_params(colors=MUTED, labelsize=7, length=2)
    cb.outline.set_edgecolor((*colors.to_rgb(MUTED), 0.28))
    cb.set_label("normalised intensity", color=MUTED, fontsize=8, labelpad=2)
    out = output_dir / "03_V3_ideal_propagation_landscape_square.png"
    _savefig(fig, out)
    plt.close(fig)
    del xz, Z, X, I
    gc.collect()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-n", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "v3_xy_2d": str(render_v3_xy_2d(output_dir, args.grid_n).name),
        "v3_xy_3d": str(render_v3_xy_3d(output_dir, args.grid_n).name),
        "v3_xz_2d": str(render_v3_xz_2d(output_dir, args.grid_n).name),
        "v3_xz_3d": str(render_v3_xz_3d(output_dir, args.grid_n).name),
        "canvas_inches": list(CANVAS_SIZE),
        "export_pixel_size": [int(CANVAS_SIZE[0] * DPI), int(CANVAS_SIZE[1] * DPI)],
        "colour_map": str(CMAP),
        "gamma": GAMMA,
        "z_ref_mm": Z_REF_M * 1e3,
        "status": "presentation_only_consistent_visual_set",
    }
    (output_dir / "v3_visual_consistency_manifest.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
