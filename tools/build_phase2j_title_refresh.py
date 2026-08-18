"""Presentation-only refresh for the title slide and computational-route slide.

Outputs are regenerated from the current repository optical model; no AI or
synthetic bench imagery is used.

The renderer deliberately separates two ideas that were previously conflated
in the route labels:

* bench role: SLM1 is the beam-shaping phase plane and SLM2 is the wavefront-
  correction phase plane; in the laboratory both holograms may also contain a
  blaze/carrier term for diffraction-order management;
* current ideal numerical reference: the plotted SLM1 active term is the
  shaping/vortex term and the plotted SLM2 active term is the carrier, while
  the SLM2 correction term is zero.

This is a presentation clarification, not a silent change to the forward model.
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
TITLE_HALFWIDTH_M = 0.235e-3
ROUTE_PUPIL_HALFWIDTH_M = 2.4e-3
ROUTE_OUTPUT_HALFWIDTH_M = 0.18e-3


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
            f"presentation crop contains only {ids.size} native samples; "
            "increase grid_n rather than hiding sparse sampling with export DPI"
        )
    return np.asarray(values)[np.ix_(ids, ids)], x[ids]


def _clean_3d_axis(ax: Any) -> None:
    ax.set_facecolor(style.FIG_BG)
    ax.tick_params(colors=style.MUTED, labelsize=8, pad=0)
    ax.xaxis.label.set_color(style.MUTED)
    ax.yaxis.label.set_color(style.MUTED)
    ax.zaxis.label.set_color(style.MUTED)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((*colors.to_rgb(style.FIG_BG), 0.0))
        axis.pane.set_edgecolor((*colors.to_rgb(style.MUTED), 0.10))
        try:
            axis._axinfo["grid"]["color"] = (*colors.to_rgb(style.MUTED), 0.10)
            axis._axinfo["grid"]["linewidth"] = 0.42
        except Exception:
            pass


def build_cinematic_v3_surface(output_dir: Path, grid_n: int) -> Path:
    """Dark cinematic 3D transverse V3 intensity at the canonical z=60 mm."""
    route = build_system_route("V3", grid_n=int(grid_n))
    prop = _propagator(route, Z_REF_M)
    field = np.asarray(native_field_at_z(prop, Z_REF_M), dtype=np.complex128)
    intensity = np.abs(field) ** 2
    crop, xc = _square_crop(intensity, route["grid"], TITLE_HALFWIDTH_M)
    crop = style.normalise(crop)

    stride = max(1, int(np.ceil(max(crop.shape) / 300)))
    surface_i = crop[::stride, ::stride]
    x_mm = xc[::stride] * 1e3
    X, Y = np.meshgrid(x_mm, x_mm)
    norm = colors.PowerNorm(gamma=0.50, vmin=0.0, vmax=1.0)

    fig = plt.figure(figsize=(10.2, 8.0), facecolor=style.FIG_BG)
    ax = fig.add_subplot(111, projection="3d")
    _clean_3d_axis(ax)
    surf = ax.plot_surface(
        X,
        Y,
        surface_i,
        cmap=style.CMAP,
        norm=norm,
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=True,
        shade=False,
        alpha=0.99,
        rasterized=True,
    )

    floor = -0.07
    ax.contour(
        X,
        Y,
        surface_i,
        zdir="z",
        offset=floor,
        levels=[0.018, 0.035, 0.065, 0.11, 0.18, 0.30, 0.48, 0.72],
        cmap=style.CMAP,
        norm=norm,
        linewidths=0.8,
        alpha=0.52,
    )
    ax.set_zlim(floor, 1.03)
    ax.set_box_aspect((1.0, 1.0, 0.47))
    ax.view_init(elev=27, azim=-56)
    ax.set_xlabel("x (mm)", labelpad=7, fontsize=9)
    ax.set_ylabel("y (mm)", labelpad=7, fontsize=9)
    ax.set_zlabel("normalised intensity", labelpad=6, fontsize=9)
    ax.set_title(
        "V3 vortex–Bessel intensity  |  z = 60 mm",
        color=style.TEXT,
        fontsize=15.5,
        weight="bold",
        pad=18,
    )

    cax = fig.add_axes([0.31, 0.078, 0.38, 0.020])
    cb = fig.colorbar(surf, cax=cax, orientation="horizontal")
    cb.ax.tick_params(colors=style.MUTED, labelsize=7, length=2)
    cb.outline.set_edgecolor((*colors.to_rgb(style.MUTED), 0.28))
    cb.set_label("normalised intensity", color=style.MUTED, fontsize=8, labelpad=2)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "00_V3_3d_intensity_cinematic.png"
    fig.savefig(out, dpi=520, bbox_inches="tight", facecolor=fig.get_facecolor(), pad_inches=0.06)
    plt.close(fig)
    del route, prop, field, intensity, crop, surface_i
    gc.collect()
    return out


def build_v3_propagation_assets(output_dir: Path, grid_n: int) -> tuple[Path, Path]:
    """Create clean fixed-lab V3 x-z and cinematic 3D propagation landscapes."""
    route = build_system_route("V3", grid_n=int(grid_n))
    prop = _propagator(route, float(Z_VALUES_M[-1]))
    mapped = build_fixed_plane_longitudinal_map(
        prop,
        z_values_m=Z_VALUES_M,
        x_coordinates_m=PROP_COORD_M,
        y_coordinates_m=PROP_COORD_M,
        fixed_x_m=0.0,
        fixed_y_m=0.0,
        source_label="phase2j-title-refresh-V3-fixed-lab",
    )
    xz = style.normalise(np.asarray(mapped.xz_intensity, dtype=float))
    x_mm = PROP_COORD_M * 1e3
    z_mm = Z_VALUES_M * 1e3
    norm = colors.PowerNorm(gamma=0.57, vmin=0.0, vmax=1.0)

    # A clean, high-resolution x-z map: no sample proxy, no material overlay,
    # no tracking/recentring, and no decorative geometry that could be mistaken
    # for measured apparatus.
    fig, ax = plt.subplots(figsize=(10.7, 6.3))
    style.style_fig(fig)
    style.style_ax(ax)
    image = ax.imshow(
        xz.T,
        origin="lower",
        extent=[z_mm[0], z_mm[-1], x_mm[0], x_mm[-1]],
        cmap=style.CMAP,
        norm=norm,
        interpolation=style.DISPLAY_INTERPOLATION,
        aspect="auto",
    )
    ax.set_title("Ideal V3 vortex–Bessel propagation", color=style.TEXT, fontsize=16, weight="bold", pad=14)
    ax.set_xlabel("z from physical axicon (mm)", fontsize=10)
    ax.set_ylabel("x at fixed y = 0 (mm)", fontsize=10)
    ax.axhline(0.0, color="white", alpha=0.10, linewidth=0.45)
    ax.axvline(Z_REF_M * 1e3, color="white", alpha=0.24, linestyle="--", linewidth=0.8)
    ax.text(
        Z_REF_M * 1e3 + 2.0,
        x_mm[-1] - 0.012,
        "z = 60 mm",
        color=style.MUTED,
        fontsize=8.2,
        rotation=90,
        va="top",
    )
    cb = fig.colorbar(image, ax=ax, pad=0.018, shrink=0.86)
    cb.ax.tick_params(colors=style.MUTED, labelsize=8)
    cb.outline.set_edgecolor((*colors.to_rgb(style.MUTED), 0.25))
    cb.set_label("global-peak-normalised intensity", color=style.MUTED, fontsize=9)
    fig.tight_layout()
    out_xz = output_dir / "01_V3_ideal_propagation_cinematic_xz.png"
    fig.savefig(out_xz, dpi=500, bbox_inches="tight", facecolor=fig.get_facecolor(), pad_inches=0.05)
    plt.close(fig)

    # A title-slide alternative: the same fixed-lab x-z data rendered as a
    # perspective intensity landscape.  Height and colour encode the same
    # intensity, so the view is cinematic without inventing a new quantity.
    stride_z = 2
    stride_x = 3
    Z, X = np.meshgrid(z_mm[::stride_z], x_mm[::stride_x], indexing="ij")
    I = xz[::stride_z, ::stride_x]
    fig = plt.figure(figsize=(10.5, 7.7), facecolor=style.FIG_BG)
    ax = fig.add_subplot(111, projection="3d")
    _clean_3d_axis(ax)
    surf = ax.plot_surface(
        Z,
        X,
        I,
        cmap=style.CMAP,
        norm=norm,
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=True,
        shade=False,
        alpha=0.985,
        rasterized=True,
    )
    ax.set_xlabel("z from axicon (mm)", labelpad=8, fontsize=9)
    ax.set_ylabel("x (mm)", labelpad=8, fontsize=9)
    ax.set_zlabel("normalised intensity", labelpad=6, fontsize=9)
    ax.set_zlim(0.0, 1.02)
    ax.set_box_aspect((1.45, 0.72, 0.44))
    ax.view_init(elev=29, azim=-64)
    ax.set_title("Ideal V3 propagation landscape", color=style.TEXT, fontsize=16, weight="bold", pad=16)
    cax = fig.add_axes([0.32, 0.075, 0.36, 0.020])
    cb = fig.colorbar(surf, cax=cax, orientation="horizontal")
    cb.ax.tick_params(colors=style.MUTED, labelsize=7, length=2)
    cb.outline.set_edgecolor((*colors.to_rgb(style.MUTED), 0.28))
    cb.set_label("normalised intensity", color=style.MUTED, fontsize=8, labelpad=2)
    out_landscape = output_dir / "02_V3_ideal_propagation_landscape.png"
    fig.savefig(out_landscape, dpi=500, bbox_inches="tight", facecolor=fig.get_facecolor(), pad_inches=0.06)
    plt.close(fig)

    del route, prop, mapped, xz
    gc.collect()
    return out_xz, out_landscape


def build_computational_route_roles(output_dir: Path, grid_n: int) -> Path:
    """Clarify experimental SLM roles without changing the plotted forward model."""
    route = build_system_route("V1", grid_n=int(grid_n))
    grid = dict(route["grid"])
    x = np.asarray(grid["x"], dtype=float)
    ids = np.flatnonzero(np.abs(x) <= ROUTE_PUPIL_HALFWIDTH_M)
    extent = [x[ids[0]] * 1e3, x[ids[-1]] * 1e3, x[ids[0]] * 1e3, x[ids[-1]] * 1e3]

    input_i = np.abs(np.asarray(route["input_beam"])) ** 2
    slm1_phase = np.angle(np.asarray(route["post_slm1"]) * np.conj(np.asarray(route["input_beam"])))
    slm2_phase = np.angle(np.asarray(route["post_slm2"]) * np.conj(np.asarray(route["post_slm1"])))
    fourier_i = np.abs(np.asarray(route["fourier_plane_before_iris"])) ** 2
    axicon_phase = np.angle(np.asarray(route["post_axicon_local"]) * np.conj(np.asarray(route["field_on_axicon_plane"])))
    output_i = np.abs(np.asarray(native_field_at_z(_propagator(route, Z_REF_M), Z_REF_M))) ** 2

    fig, axes = plt.subplots(1, 6, figsize=(16.5, 4.35), constrained_layout=False)
    style.style_fig(fig)
    fig.subplots_adjust(left=0.043, right=0.992, bottom=0.18, top=0.67, wspace=0.055)

    panels = [
        (input_i, "input beam", style.CMAP, False),
        (slm1_phase, "SLM1: shaping term", "twilight", True),
        (slm2_phase, "SLM2: active carrier term", "twilight", True),
        (fourier_i, "4F: selected order", style.CMAP, False),
        (axicon_phase, "physical axicon", "twilight", True),
    ]
    for i, (data, title, cmap, is_phase) in enumerate(panels):
        ax = axes[i]
        style.style_ax(ax)
        sub = np.asarray(data)[np.ix_(ids, ids)]
        if is_phase:
            ax.imshow(sub, origin="lower", extent=extent, cmap=cmap, vmin=-np.pi, vmax=np.pi, interpolation="nearest", aspect="equal")
        else:
            ax.imshow(style.normalise(sub), origin="lower", extent=extent, cmap=style.CMAP, vmin=0.0, vmax=1.0, interpolation=style.DISPLAY_INTERPOLATION, aspect="equal")
        ax.set_title(title, fontsize=10.2, color=style.TEXT, weight="bold", pad=6)
        ax.set_xlabel("x (mm)", fontsize=7.5)
        if i == 0:
            ax.set_ylabel("y (mm)", fontsize=7.5)
        else:
            ax.tick_params(labelleft=False)

    out_crop, out_x = _square_crop(output_i, grid, ROUTE_OUTPUT_HALFWIDTH_M)
    out_extent = [out_x[0] * 1e3, out_x[-1] * 1e3, out_x[0] * 1e3, out_x[-1] * 1e3]
    ax = axes[5]
    style.style_ax(ax)
    ax.imshow(style.normalise(out_crop), origin="lower", extent=out_extent, cmap=style.CMAP, vmin=0.0, vmax=1.0, interpolation=style.DISPLAY_INTERPOLATION, aspect="equal")
    ax.set_title("propagated field\nz = 60 mm", fontsize=10.2, color=style.TEXT, weight="bold", pad=6)
    ax.set_xlabel("x (mm)", fontsize=7.5)
    ax.tick_params(labelleft=False)

    fig.suptitle(
        "Current numerical route: dual SLM → 4F → physical axicon → propagated field",
        color=style.TEXT,
        fontsize=15.4,
        weight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.855,
        "Bench roles: SLM1 = beam-shaping phase + blaze/carrier   •   SLM2 = wavefront-correction phase + blaze/carrier",
        color=style.TEXT,
        fontsize=10.2,
        weight="bold",
        ha="center",
    )
    fig.text(
        0.5,
        0.800,
        "Ideal reference plotted here: SLM2 correction = 0; panel labels identify the phase terms currently active in the numerical route.",
        color=style.MUTED,
        fontsize=8.8,
        ha="center",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "03_computational_route_dual_slm_roles.png"
    fig.savefig(out, dpi=480, bbox_inches="tight", facecolor=fig.get_facecolor(), pad_inches=0.05)
    plt.close(fig)
    del route
    gc.collect()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-n", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/presentation_title_refresh"))
    args = parser.parse_args()
    if args.grid_n < 2048:
        raise ValueError("presentation title refresh requires grid_n >= 2048")
    style.validate_palette_has_no_cool_segment()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    p0 = build_cinematic_v3_surface(args.output_dir, args.grid_n)
    p1, p2 = build_v3_propagation_assets(args.output_dir, args.grid_n)
    p3 = build_computational_route_roles(args.output_dir, args.grid_n)

    manifest = {
        "outcome": "PHASE2J-TITLE-REFRESH",
        "grid_n": int(args.grid_n),
        "z_reference_mm": Z_REF_M * 1e3,
        "fixed_lab_longitudinal": True,
        "per_z_recentering": False,
        "intensity_palette": style.CMAP_NAME,
        "physics_change": False,
        "bench_role_note": "SLM1 beam shaping + blaze/carrier; SLM2 wavefront correction + blaze/carrier",
        "current_model_note": "ideal plotted route has zero SLM2 correction; active plotted terms remain shaping on SLM1 and carrier on SLM2",
        "figures": [str(p0), str(p1), str(p2), str(p3)],
    }
    (args.output_dir / "title_refresh_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
