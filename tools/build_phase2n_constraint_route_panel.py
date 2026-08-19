"""Build the corrected V3 nominal experimental-constraint route panel.

This renderer is intentionally separate from the older compact ladder figure.
It shows the actual conceptual/physical planes in the nominal scalar route:

continuous V3 target -> native SLM1 pixels -> 8-bit SLM1 command ->
SLM2 carrier -> Fourier plane before iris -> selected order after iris ->
physical axicon -> propagated V3 intensity.

Important display policy
------------------------
* Continuous phase is wrapped to [0, 2*pi) before plotting.
* SLM command panels are drawn on the native 8 um pixel lattice rather than
  pretending the 2048-point / 10 mm propagation grid resolves pixel borders.
* Fourier-plane panels use physical millimetre coordinates and explicitly draw
  the finite iris at the nominal +1 order.
* The final panel is labelled intensity, not complex field.
* This is a nominal fixed-parameter model, not a calibrated bench prediction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Circle
import numpy as np

import presentation_phase2j_style as style
from build_phase2l_nominal_constraints import XY_HALF_M, _crop, _norm, _xy
from vbb_study.digital_twin.phase2a_canonical import _panel_from_manifest
from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.vortex_beam_slm_errors import quantise_commanded_phase
from vbb_study.digital_twin.vortex_system_route import (
    AxiconError,
    build_system_route,
    physical_axicon_on_own_plane,
)

TWOPI = 2.0 * np.pi
PHASE_CMAP = "twilight"
INTENSITY_NORM = colors.PowerNorm(gamma=0.38, vmin=0.0, vmax=1.0)
SLM_HALF_M = 0.080e-3
FOURIER_X_M = (-0.55e-3, 3.20e-3)
FOURIER_Y_M = (-1.35e-3, 1.35e-3)


def _native_slm_grid(panel, half_m: float = SLM_HALF_M):
    """Return a central native-pixel SLM coordinate window."""
    pitch = float(panel.pitch_m)
    x_all = (np.arange(int(panel.n_x), dtype=float) - 0.5 * (int(panel.n_x) - 1)) * pitch
    y_all = (np.arange(int(panel.n_y), dtype=float) - 0.5 * (int(panel.n_y) - 1)) * pitch
    x = x_all[np.abs(x_all) <= float(half_m)]
    y = y_all[np.abs(y_all) <= float(half_m)]
    X, Y = np.meshgrid(x, y, indexing="xy")
    extent = [
        (x[0] - 0.5 * pitch) * 1e3,
        (x[-1] + 0.5 * pitch) * 1e3,
        (y[0] - 0.5 * pitch) * 1e3,
        (y[-1] + 0.5 * pitch) * 1e3,
    ]
    return X, Y, extent


def _continuous_display_grid(half_m: float = SLM_HALF_M, n: int = 601):
    x = np.linspace(-float(half_m), float(half_m), int(n))
    X, Y = np.meshgrid(x, x, indexing="xy")
    extent = [x[0] * 1e3, x[-1] * 1e3, x[0] * 1e3, x[-1] * 1e3]
    return X, Y, extent


def _crop_rect(values: np.ndarray, x: np.ndarray, xlim_m, ylim_m):
    xi = np.flatnonzero((x >= float(xlim_m[0])) & (x <= float(xlim_m[1])))
    yi = np.flatnonzero((x >= float(ylim_m[0])) & (x <= float(ylim_m[1])))
    if xi.size == 0 or yi.size == 0:
        raise RuntimeError("requested Fourier-plane crop is outside the simulation grid")
    crop = np.asarray(values)[np.ix_(yi, xi)]
    extent = [x[xi[0]] * 1e3, x[xi[-1]] * 1e3, x[yi[0]] * 1e3, x[yi[-1]] * 1e3]
    return crop, extent


def _phase_panel(ax, values, extent, title: str, *, native_pixels: bool = False):
    style.style_ax(ax)
    im = ax.imshow(
        np.mod(np.asarray(values, dtype=float), TWOPI),
        origin="lower",
        extent=extent,
        cmap=PHASE_CMAP,
        vmin=0.0,
        vmax=TWOPI,
        interpolation="nearest" if native_pixels else "bilinear",
        aspect="equal",
    )
    ax.set_title(title, color=style.TEXT, fontsize=11.2, weight="bold", pad=8)
    ax.set_xlabel("x (mm)", fontsize=8)
    ax.set_ylabel("y (mm)", fontsize=8)
    return im


def _intensity_panel(ax, values, extent, title: str):
    style.style_ax(ax)
    im = ax.imshow(
        _norm(values),
        origin="lower",
        extent=extent,
        cmap=style.CMAP,
        norm=INTENSITY_NORM,
        interpolation=style.DISPLAY_INTERPOLATION,
        aspect="equal",
    )
    ax.set_title(title, color=style.TEXT, fontsize=11.2, weight="bold", pad=8)
    ax.set_xlabel("x (mm)", fontsize=8)
    ax.set_ylabel("y (mm)", fontsize=8)
    return im


def build_panel(output_dir: Path, grid_n: int) -> tuple[Path, Path]:
    manifest = canonical_hardware_manifest()
    panel = _panel_from_manifest(manifest)
    route = build_system_route("V3", grid_n=int(grid_n))

    carrier_cpm = float(hardware_value(manifest, "carrier_frequency_cpm"))
    phase_levels = int(hardware_value(manifest, "slm_phase_bits"))
    levels = 2 ** phase_levels
    pixel_pitch_m = float(hardware_value(manifest, "slm_pixel_pitch_m"))
    base_angle_deg = float(hardware_value(manifest, "axicon_base_angle_deg"))

    # 1) Continuous wrapped V3 target on a dense display grid.
    Xc, Yc, continuous_extent = _continuous_display_grid()
    continuous_v3 = np.mod(3.0 * np.arctan2(Yc, Xc), TWOPI)

    # 2-4) Native SLM pixel visualisations.  These are deliberately rendered on
    # the physical 8 um pixel lattice, independently of the propagation grid.
    Xp, Yp, slm_extent = _native_slm_grid(panel)
    slm1_sampled = np.mod(3.0 * np.arctan2(Yp, Xp), TWOPI)
    slm1_8bit = quantise_commanded_phase(slm1_sampled, levels)
    slm2_carrier = quantise_commanded_phase(np.mod(TWOPI * carrier_cpm * Xp, TWOPI), levels)

    # 5-6) Fourier-plane intensity before and immediately after the actual
    # finite iris.  Keep physical coordinates and the same crop in both panels.
    fourier_before = np.abs(np.asarray(route["fourier_plane_before_iris"], dtype=np.complex128)) ** 2
    iris_mask = np.asarray(route["fourier_iris_mask"], dtype=float)
    fourier_after = fourier_before * iris_mask
    x_grid = np.asarray(route["grid"]["x"], dtype=float)
    fourier_before_crop, fourier_extent = _crop_rect(fourier_before, x_grid, FOURIER_X_M, FOURIER_Y_M)
    fourier_after_crop, _ = _crop_rect(fourier_after, x_grid, FOURIER_X_M, FOURIER_Y_M)

    fourf_meta = route["metadata"]["fourf"]
    iris_centre_m = tuple(map(float, fourf_meta["physical_iris_centre_m"]))
    iris_radius_m = float(fourf_meta["iris_radius_m"])
    nominal_order_m = tuple(map(float, fourf_meta["nominal_selected_order_centre_m"]))
    selected_fraction = float(fourf_meta["iris_selected_power_fraction"])

    # 7) Physical axicon phase itself, using the same nominal axicon model as the
    # route.  This prevents the diagram from implying that the 4F creates the
    # Bessel field.
    g = route["grid"]
    axicon_t, _ = physical_axicon_on_own_plane(
        g,
        wavelength_m=float(hardware_value(manifest, "wavelength_m")),
        base_angle_rad=math.radians(base_angle_deg),
        refractive_index=float(hardware_value(manifest, "axicon_refractive_index")),
        external_index=float(hardware_value(manifest, "axicon_external_medium_index")),
        error=AxiconError(),
    )
    axicon_phase = np.mod(np.angle(axicon_t), TWOPI)
    axicon_crop, axicon_x = _crop(axicon_phase, g, half=XY_HALF_M)
    axicon_extent = [axicon_x[0] * 1e3, axicon_x[-1] * 1e3, axicon_x[0] * 1e3, axicon_x[-1] * 1e3]

    # 8) Propagated intensity, explicitly labelled as intensity.
    final_intensity = _xy(route)
    final_crop, final_x = _crop(final_intensity, g, half=XY_HALF_M)
    final_extent = [final_x[0] * 1e3, final_x[-1] * 1e3, final_x[0] * 1e3, final_x[-1] * 1e3]

    fig, axes = plt.subplots(2, 4, figsize=(16.6, 8.6), facecolor=style.FIG_BG)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.12, top=0.79, wspace=0.26, hspace=0.42)
    ax = axes.ravel()

    phase_im = _phase_panel(ax[0], continuous_v3, continuous_extent, "Continuous V3 target\nwrapped phase", native_pixels=False)
    _phase_panel(ax[1], slm1_sampled, slm_extent, "SLM1 native command\n8 µm pixels", native_pixels=True)
    _phase_panel(ax[2], slm1_8bit, slm_extent, "SLM1 quantised command\n8-bit / 256 levels", native_pixels=True)
    _phase_panel(ax[3], slm2_carrier, slm_extent, "SLM2 carrier / blaze\n20-pixel period", native_pixels=True)

    _intensity_panel(ax[4], fourier_before_crop, fourier_extent, "Fourier plane before iris\nphysical coordinates")
    _intensity_panel(ax[5], fourier_after_crop, fourier_extent, "Selected +1 order\nafter finite iris")

    # Explicit physical iris overlay on both Fourier-plane panels.
    for q in (ax[4], ax[5]):
        q.axvline(0.0, color=style.MUTED, alpha=0.34, linewidth=0.8, linestyle="--")
        q.axhline(0.0, color=style.MUTED, alpha=0.20, linewidth=0.6, linestyle="--")
        q.add_patch(
            Circle(
                (iris_centre_m[0] * 1e3, iris_centre_m[1] * 1e3),
                iris_radius_m * 1e3,
                fill=False,
                edgecolor=style.TEXT,
                linewidth=1.4,
                alpha=0.92,
            )
        )
        q.plot(
            nominal_order_m[0] * 1e3,
            nominal_order_m[1] * 1e3,
            marker="+",
            markersize=8,
            markeredgewidth=1.2,
            color=style.TEXT,
        )
    ax[4].text(0.02, 0.03, "optical axis", transform=ax[4].transAxes, color=style.MUTED, fontsize=7.3)
    ax[5].text(
        0.03,
        0.03,
        f"selected power = {100.0 * selected_fraction:.1f}%",
        transform=ax[5].transAxes,
        color=style.MUTED,
        fontsize=7.3,
    )

    _phase_panel(
        ax[6],
        axicon_crop,
        axicon_extent,
        f"Physical axicon\nnominal sharp cone, {base_angle_deg:g}°",
        native_pixels=False,
    )
    _intensity_panel(ax[7], final_crop, final_extent, "Normalised V3 intensity\nz = 60 mm")

    fig.suptitle(
        "How nominal experimental constraints enter the V3 computational route",
        color=style.TEXT,
        fontsize=18.5,
        weight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.905,
        "continuous target → SLM1 sampling / 8-bit command → SLM2 carrier → physical 4F + finite +1 iris → physical axicon → propagation",
        ha="center",
        color=style.MUTED,
        fontsize=10.3,
    )
    fig.text(
        0.5,
        0.855,
        f"Native SLM display uses {pixel_pitch_m * 1e6:.0f} µm pixels; Fourier iris centre = {iris_centre_m[0] * 1e3:.3f} mm, radius = {iris_radius_m * 1e3:.3f} mm",
        ha="center",
        color=style.MUTED,
        fontsize=9.0,
    )
    fig.text(
        0.5,
        0.825,
        "Nominal fixed-parameter model only — no measured LUT, static wavefront map, fringing calibration, deliberate misalignment or axicon surface map",
        ha="center",
        color=style.MUTED,
        fontsize=8.6,
    )

    # One shared cyclic phase colourbar for all phase panels.
    cax = fig.add_axes([0.305, 0.055, 0.39, 0.018])
    cb = fig.colorbar(phase_im, cax=cax, orientation="horizontal")
    cb.set_ticks([0.0, np.pi, TWOPI])
    cb.set_ticklabels(["0", "π", "2π"])
    cb.ax.tick_params(colors=style.MUTED, labelsize=8, length=2)
    cb.outline.set_edgecolor((*colors.to_rgb(style.MUTED), 0.35))
    cb.set_label("wrapped phase (rad)", color=style.MUTED, fontsize=8.5, labelpad=2)

    png_path = output_dir / "03_constraint_ladder_SLM_4F.png"
    fig.savefig(png_path, dpi=500, bbox_inches="tight", facecolor=style.FIG_BG, pad_inches=0.06)
    plt.close(fig)

    metadata = {
        "claim": "nominal_fixed_parameter_route_visualisation_not_calibrated_bench_prediction",
        "case_id": "V3",
        "vortex_charge": 3,
        "continuous_phase_wrapped_for_display": True,
        "slm_display_grid": "native_pixel_centres",
        "slm_pixel_pitch_m": pixel_pitch_m,
        "slm_phase_levels": levels,
        "carrier_frequency_cpm": carrier_cpm,
        "carrier_period_px": 20,
        "fourier_iris_centre_m": iris_centre_m,
        "fourier_nominal_order_centre_m": nominal_order_m,
        "fourier_iris_radius_m": iris_radius_m,
        "fourier_selected_power_fraction": selected_fraction,
        "axicon_base_angle_deg": base_angle_deg,
        "final_plane_z_m": 60.0e-3,
        "panels": [
            "continuous_wrapped_v3_target",
            "slm1_native_8um_sampled_command",
            "slm1_8bit_quantised_command",
            "slm2_8bit_carrier_command",
            "fourier_plane_before_iris_with_physical_iris_overlay",
            "fourier_plane_after_iris",
            "physical_axicon_wrapped_phase",
            "normalised_v3_intensity_z_60mm",
        ],
    }
    json_path = output_dir / "03_constraint_route_panel_manifest.json"
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return png_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-n", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    png, manifest = build_panel(args.output_dir, args.grid_n)
    print(png)
    print(manifest)


if __name__ == "__main__":
    main()
