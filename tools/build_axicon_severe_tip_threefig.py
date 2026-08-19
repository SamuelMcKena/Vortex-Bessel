"""Render only the three presentation figures for severe rounded-tip avoidance.

The optimisation is delegated to build_axicon_severe_tip_comparison, which uses
an 800 um rounded apex and matched sharp-tip controls.  This wrapper keeps the
scientific selection but presents only what is needed on slides:

  1. XY: incident fields at the axicon and z=60 mm outputs;
  2. XZ: fixed-laboratory propagation through the severe rounded axicon;
  3. 1D: z=60 mm transverse lineouts and rounded/sharp on-axis modulation.

No sweep plot, manifest, or extra presentation image is emitted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

import build_axicon_presentation_evidence_v3 as base
import build_axicon_severe_tip_comparison as severe
import build_phase2j_presentation_suite as suite
import presentation_phase2j_style as style

EPS = np.finfo(float).tiny
TIP_RADIUS_M = severe.SEVERE_TIP_RADIUS_M
Z_REF_M = severe.Z_REF_M
COLORS = ("#ff9d00", "#39d6ad")


def _choose(result: dict) -> dict:
    rows = list(result["rows"])
    admissible = [r for r in rows if (
        r["tip_fraction"] <= 5e-4
        and r["mean_sharp_onaxis_vs_B0"] >= 0.20
        and r["z60_centre_to_peak_rounded"] >= 0.80
    )]
    if admissible:
        return min(admissible, key=lambda r: (r["rms_vs_ordinary_B0"], r["clear_radius_um"]))
    return min(rows, key=lambda r: (r["rms_vs_ordinary_B0"] + 5.0*r["tip_fraction"], -r["z60_centre_to_peak_rounded"]))


def render(out: Path, sweep_grid_n: int, render_grid_n: int) -> None:
    sweep = severe.select_clear_radius(sweep_grid_n)
    chosen = _choose(sweep)
    selected_um = int(chosen["clear_radius_um"])

    wavelength, grid, b0, hollow, sharp_t, round_t = severe._build_fields(render_grid_n, selected_um)
    assert hollow is not None
    coord = np.asarray(suite.TIP_COORD_M, float)
    z = np.asarray(suite.Z_VALUES_M, float)

    cases = []
    for label, incident in (
        ("ordinary B0", b0),
        (f"annular ell=0\n{selected_um} um clear radius", hollow),
    ):
        xy_round = base._xy_from_post_axicon(incident * round_t, grid, wavelength)
        xy_sharp = base._xy_from_post_axicon(incident * sharp_t, grid, wavelength)
        xz_round, _ = base._xz_from_post_axicon(incident * round_t, grid, wavelength, coord, f"severe-threefig-{label}-round")
        xz_sharp, _ = base._xz_from_post_axicon(incident * sharp_t, grid, wavelength, coord, f"severe-threefig-{label}-sharp")
        axis_round = severe._onaxis_from_xz(xz_round, coord)
        axis_sharp = severe._onaxis_from_xz(xz_sharp, coord)
        rms = severe._rms_ratio_error(axis_round, axis_sharp, z)
        cases.append({
            "label": label,
            "incident": incident,
            "xy_round": xy_round,
            "xy_sharp": xy_sharp,
            "xz_round": xz_round,
            "xz_sharp": xz_sharp,
            "axis_round": axis_round,
            "axis_sharp": axis_sharp,
            "tip_fraction": severe._fraction_inside(incident, grid, TIP_RADIUS_M),
            "rms": rms,
            "centre_ratio": severe._centre_to_peak(xy_round, grid),
        })

    improvement = cases[1]["rms"] / max(cases[0]["rms"], EPS)
    print(f"SELECTED_CLEAR_RADIUS_UM={selected_um}")
    print(f"ORDINARY_B0_APEX_POWER_PERCENT={100*cases[0]['tip_fraction']:.6f}")
    print(f"ANNULAR_APEX_POWER_PERCENT={100*cases[1]['tip_fraction']:.6f}")
    print(f"ORDINARY_B0_ONAXIS_RMS={cases[0]['rms']:.8f}")
    print(f"ANNULAR_ONAXIS_RMS={cases[1]['rms']:.8f}")
    print(f"ANNULAR_TO_B0_RMS_RATIO={improvement:.8f}")
    print(f"ANNULAR_Z60_CENTRE_TO_PEAK={cases[1]['centre_ratio']:.8f}")

    # ---------- XY ----------
    # Top row: the fields actually incident on the severe rounded apex.
    # Bottom row: the output at the common z=60 mm camera plane.
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 8.3), constrained_layout=True)
    style.style_fig(fig)
    incident_peak = max(float(np.max(np.abs(c["incident"])**2)) for c in cases)
    for col, case in enumerate(cases):
        inc = np.abs(case["incident"])**2
        crop, extent = base._fixed_crop(inc, grid, 1.40e-3)
        title = "ordinary B0 directly illuminates apex" if col == 0 else "annular ell=0 clears apex"
        style.draw_xy(axes[0, col], crop, extent, title, peak=incident_peak, show_y=(col == 0))
        axes[0, col].add_patch(Circle((0, 0), TIP_RADIUS_M*1e3, fill=False, edgecolor="#39d6ad", lw=1.4, ls="--"))
        axes[0, col].text(
            0.03, 0.04,
            f"power inside 800 um region: {100*case['tip_fraction']:.3f}%",
            transform=axes[0, col].transAxes, color=style.TEXT, fontsize=8,
            bbox=dict(facecolor=style.FIG_BG, edgecolor=style.BORDER, alpha=0.88, pad=2.5),
        )

    output_peak = max(float(np.max(c["xy_round"])) for c in cases)
    for col, case in enumerate(cases):
        crop, extent = base._fixed_crop(case["xy_round"], grid, 0.42e-3)
        style.draw_xy(axes[1, col], crop, extent, "output at z = 60 mm", peak=output_peak, show_y=(col == 0))
        axes[1, col].text(
            0.03, 0.04,
            f"centre / peak = {case['centre_ratio']:.3f}",
            transform=axes[1, col].transAxes, color=style.TEXT, fontsize=8,
            bbox=dict(facecolor=style.FIG_BG, edgecolor=style.BORDER, alpha=0.88, pad=2.5),
        )
    fig.suptitle("800 um rounded axicon tip: direct illumination vs annular avoidance", color=style.TEXT, fontsize=16)
    fig.text(0.5, -0.008, "Dashed circle marks the severe rounded-apex footprint. Both inputs have equal pre-axicon power; annular input is non-vortex (ell=0).", ha="center", color=style.MUTED, fontsize=9)
    pxy = out / "09_severe_tip_avoidance_XY.png"
    style.save(fig, pxy)

    # ---------- XZ ----------
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9), constrained_layout=True)
    style.style_fig(fig)
    xz_peak = max(float(np.max(c["xz_round"])) for c in cases)
    for col, case in enumerate(cases):
        style.draw_xz(axes[col], case["xz_round"], coord, z, peak=xz_peak, show_y=(col == 0), z_ref_m=Z_REF_M)
        axes[col].set_title(case["label"], color=style.TEXT, fontsize=11)
        axes[col].text(
            0.03, 0.04,
            f"rounded/sharp on-axis RMS = {100*case['rms']:.1f}%",
            transform=axes[col].transAxes, color=style.TEXT, fontsize=8,
            bbox=dict(facecolor=style.FIG_BG, edgecolor=style.BORDER, alpha=0.88, pad=2.5),
        )
    fig.suptitle("Longitudinal field through the same severely rounded axicon", color=style.TEXT, fontsize=15)
    pxz = out / "09b_severe_tip_avoidance_XZ.png"
    style.save(fig, pxz)

    # ---------- 1D ----------
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    style.style_fig(fig)
    for ax in axes:
        base._style_line_axis(ax)

    shared = max(float(np.max(c["xy_round"])) for c in cases)
    for case, colour in zip(cases, COLORS):
        x, line = base._lineout_x(case["xy_round"], grid, 0.0)
        keep = np.abs(x) <= 0.45e-3
        axes[0].plot(x[keep]*1e3, line[keep]/shared, color=colour, lw=1.9, label=case["label"].replace("\n", " - "))
    axes[0].set_xlabel("x at fixed y = 0 (mm)")
    axes[0].set_ylabel("intensity / shared rounded-tip maximum")
    axes[0].set_title("Transverse intensity at z = 60 mm", color=style.TEXT, fontsize=12)

    for case, colour in zip(cases, COLORS):
        ratio = case["axis_round"] / np.maximum(case["axis_sharp"], EPS)
        axes[1].plot(z*1e3, ratio, color=colour, lw=1.9, label=case["label"].replace("\n", " - "))
    axes[1].axhline(1.0, color=style.MUTED, lw=0.9, ls="--", alpha=0.75)
    axes[1].set_xlabel("z from axicon (mm)")
    axes[1].set_ylabel("on-axis intensity: rounded / matched sharp")
    axes[1].set_title("Rounded-tip interference signature", color=style.TEXT, fontsize=12)

    for ax in axes:
        leg = ax.legend(frameon=False, fontsize=8)
        for t in leg.get_texts():
            t.set_color(style.TEXT)
    fig.suptitle(f"Severe-tip mitigation: annular rounded/sharp RMS is {improvement:.2f}x the ordinary-B0 value", color=style.TEXT, fontsize=14)
    p1d = out / "09c_severe_tip_avoidance_1D.png"
    style.save(fig, p1d)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-grid-n", type=int, default=1024)
    parser.add_argument("--render-grid-n", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/axicon_severe_tip"))
    args = parser.parse_args()
    if args.sweep_grid_n < 1024 or args.render_grid_n < 1024:
        raise ValueError("severe-tip figures require grids >= 1024")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render(args.output_dir, args.sweep_grid_n, args.render_grid_n)


if __name__ == "__main__":
    main()
