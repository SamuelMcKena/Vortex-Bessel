"""Severe rounded-tip axicon avoidance: normal B0 versus annular ell=0 input.

This renderer is intentionally focused on the presentation comparison only.
A severe 800 um radial rounding is used because the mild 200 um case produced
only a subtle perturbation for the present beam.  Two incident fields are
compared through the SAME rounded axicon:

  1. ordinary routed B0 illumination, which overlaps the rounded apex;
  2. a non-vortex annular (ell=0) version of the routed B0 field whose central
     dark region clears the rounded apex.

Sharp-tip versions of both incident fields are computed internally as matched
controls and used for quantitative optimisation, but are not shown in the
presentation figures.  The annular field is an axicon-plane planning target;
it is not claimed as a calibrated phase-only SLM command.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

import build_axicon_presentation_evidence_v3 as base
import build_phase2j_presentation_suite as suite
import presentation_phase2j_style as style

from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.vortex_system_route import build_system_route

EPS = np.finfo(float).tiny
TIP_RADIUS_M = 800e-6
Z_REF_M = 60e-3
CLEAR_RADII_UM = (820, 860, 900, 940, 980, 1020, 1080, 1140, 1200)
EDGE_WIDTH_M = 55e-6
XZ_COORD_M = np.linspace(-0.34e-3, 0.34e-3, 681)
COLORS = ("#ff9d00", "#39d6ad")


def _annular_l0(field_b0: np.ndarray, grid, clear_radius_m: float, edge_width_m: float = EDGE_WIDTH_M) -> np.ndarray:
    """Smooth, explicitly cleared ell=0 annular field, power matched to B0."""
    X = np.asarray(grid["X"], float)
    Y = np.asarray(grid["Y"], float)
    R = np.hypot(X, Y)
    # Logistic/tanh wall: essentially dark throughout the rounded-apex region,
    # but without a hard binary edge that would add an artificial diffraction ring.
    notch = 0.5 * (1.0 + np.tanh((R - float(clear_radius_m)) / max(float(edge_width_m), EPS)))
    target = np.asarray(field_b0, np.complex128) * notch
    p0 = float(np.sum(np.abs(field_b0) ** 2))
    p1 = float(np.sum(np.abs(target) ** 2))
    if p1 > EPS:
        target *= math.sqrt(p0 / p1)
    return target


def _tip_fraction(field: np.ndarray, grid) -> float:
    R = np.hypot(np.asarray(grid["X"], float), np.asarray(grid["Y"], float))
    p = np.abs(np.asarray(field, np.complex128)) ** 2
    return float(np.sum(p[R <= TIP_RADIUS_M]) / max(float(np.sum(p)), EPS))


def _peak_vs_z(xz: np.ndarray) -> np.ndarray:
    return np.max(np.asarray(xz, float), axis=1)


def _rms_ratio(round_xz: np.ndarray, sharp_xz: np.ndarray, z0: float, z1: float) -> tuple[float, float]:
    z = np.asarray(suite.Z_VALUES_M, float)
    ratio = _peak_vs_z(round_xz) / np.maximum(_peak_vs_z(sharp_xz), EPS)
    keep = (z >= z0) & (z <= z1)
    dev = ratio[keep] - 1.0
    return float(np.sqrt(np.mean(dev**2))), float(np.max(np.abs(dev)))


def _centre_to_peak(xy: np.ndarray, grid) -> float:
    x = np.asarray(grid["x"], float)
    i0 = int(np.argmin(np.abs(x)))
    return float(np.asarray(xy, float)[i0, i0] / max(float(np.max(xy)), EPS))


def _crop_xy(values: np.ndarray, grid, halfwidth_m: float):
    return base._fixed_crop(values, grid, halfwidth_m)


def _evaluate_candidate(field_b0, grid, wavelength, sharp_t, round_t, clear_um: int, b0_sharp_xz, b0_round_xz, b0_mean_peak: float) -> dict:
    ann = _annular_l0(field_b0, grid, clear_um * 1e-6)
    sharp_xz, _ = base._xz_from_post_axicon(ann * sharp_t, grid, wavelength, XZ_COORD_M, f"severe-ann-{clear_um}-sharp")
    round_xz, _ = base._xz_from_post_axicon(ann * round_t, grid, wavelength, XZ_COORD_M, f"severe-ann-{clear_um}-round")
    round_xy = base._xy_from_post_axicon(ann * round_t, grid, wavelength)
    dev_rms, dev_max = _rms_ratio(round_xz, sharp_xz, 55e-3, 115e-3)
    early_rms, _ = _rms_ratio(round_xz, sharp_xz, 20e-3, 55e-3)
    z = np.asarray(suite.Z_VALUES_M, float)
    developed = (z >= 55e-3) & (z <= 115e-3)
    mean_peak = float(np.mean(_peak_vs_z(sharp_xz)[developed]))
    return {
        "clear_radius_um": int(clear_um),
        "fraction_power_inside_800um_tip": _tip_fraction(ann, grid),
        "developed_55_to_115mm_rms_rounded_vs_sharp": dev_rms,
        "developed_55_to_115mm_max_abs_rounded_vs_sharp": dev_max,
        "early_20_to_55mm_rms_rounded_vs_sharp": early_rms,
        "mean_sharp_peak_55_to_115mm_vs_B0": mean_peak / max(b0_mean_peak, EPS),
        "z60_on_axis_to_peak_rounded": _centre_to_peak(round_xy, grid),
        "field": ann,
        "sharp_xz": sharp_xz,
        "round_xz": round_xz,
        "round_xy": round_xy,
    }


def build(out: Path, grid_n: int) -> dict:
    hw = canonical_hardware_manifest()
    wavelength = float(hardware_value(hw, "wavelength_m"))
    gamma = math.radians(float(hardware_value(hw, "axicon_base_angle_deg")))
    n_ax = float(hardware_value(hw, "axicon_refractive_index"))
    n_ext = float(hardware_value(hw, "axicon_external_medium_index"))

    route = build_system_route("B0", grid_n=grid_n)
    grid = dict(route["grid"])
    field_b0 = np.asarray(route["field_on_axicon_plane"], np.complex128)
    sharp_t = base._axicon_transmission(grid, wavelength, gamma, n_ax, n_ext, 0.0)
    round_t = base._axicon_transmission(grid, wavelength, gamma, n_ax, n_ext, TIP_RADIUS_M)

    b0_sharp_xz, _ = base._xz_from_post_axicon(field_b0 * sharp_t, grid, wavelength, XZ_COORD_M, "severe-b0-sharp")
    b0_round_xz, _ = base._xz_from_post_axicon(field_b0 * round_t, grid, wavelength, XZ_COORD_M, "severe-b0-round")
    b0_round_xy = base._xy_from_post_axicon(field_b0 * round_t, grid, wavelength)
    b0_dev_rms, b0_dev_max = _rms_ratio(b0_round_xz, b0_sharp_xz, 55e-3, 115e-3)
    z = np.asarray(suite.Z_VALUES_M, float)
    developed = (z >= 55e-3) & (z <= 115e-3)
    b0_mean_peak = float(np.mean(_peak_vs_z(b0_sharp_xz)[developed]))

    candidates = [
        _evaluate_candidate(field_b0, grid, wavelength, sharp_t, round_t, r, b0_sharp_xz, b0_round_xz, b0_mean_peak)
        for r in CLEAR_RADII_UM
    ]

    # We explicitly require a strong improvement over ordinary B0 for the
    # severe rounded tip, while retaining a useful bright-centred Bessel beam.
    passing = [c for c in candidates if (
        c["fraction_power_inside_800um_tip"] <= 1.0e-3
        and c["developed_55_to_115mm_rms_rounded_vs_sharp"] <= 0.45 * b0_dev_rms
        and c["mean_sharp_peak_55_to_115mm_vs_B0"] >= 0.25
        and c["z60_on_axis_to_peak_rounded"] >= 0.75
    )]
    if passing:
        selected = min(passing, key=lambda c: c["clear_radius_um"])
        mode = "smallest_radius_with_clear_severe_tip_improvement"
    else:
        def penalty(c):
            return (
                3.0 * c["developed_55_to_115mm_rms_rounded_vs_sharp"] / max(b0_dev_rms, 1e-6)
                + 0.5 * c["early_20_to_55mm_rms_rounded_vs_sharp"]
                + 0.5 * max(0.0, 0.25 - c["mean_sharp_peak_55_to_115mm_vs_B0"])
                + 0.5 * max(0.0, 0.75 - c["z60_on_axis_to_peak_rounded"])
                + 0.4 * min(10.0, c["fraction_power_inside_800um_tip"] / 1e-3)
            )
        selected = min(candidates, key=penalty)
        mode = "minimum_penalty_fallback"

    ann = selected["field"]
    ann_round_xy = selected["round_xy"]
    ann_round_xz = selected["round_xz"]

    # ---------- XY only ----------
    # Top row shows what physically reaches the rounded apex; bottom row shows
    # the corresponding z=60 mm beam.  Same scale within each row.
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 8.2), constrained_layout=True)
    style.style_fig(fig)
    inc_b0 = np.abs(field_b0) ** 2
    inc_ann = np.abs(ann) ** 2
    inc_peak = max(float(np.max(inc_b0)), float(np.max(inc_ann)), EPS)
    for col, (inc, title, frac) in enumerate([
        (inc_b0, "ordinary B0 -> 800 um rounded tip", _tip_fraction(field_b0, grid)),
        (inc_ann, f"annular ell=0 -> same tip\nclear radius = {selected['clear_radius_um']} um", selected["fraction_power_inside_800um_tip"]),
    ]):
        crop, extent = _crop_xy(inc, grid, 1.35e-3)
        style.draw_xy(axes[0, col], crop, extent, title, peak=inc_peak, show_y=(col == 0))
        axes[0, col].add_patch(Circle((0, 0), TIP_RADIUS_M * 1e3, fill=False, edgecolor="#39d6ad", lw=1.4, ls="--"))
        axes[0, col].text(0.03, 0.04, f"power in rounded apex: {100*frac:.3f}%", transform=axes[0,col].transAxes, color=style.TEXT, fontsize=8, bbox=dict(facecolor=style.FIG_BG, edgecolor=style.BORDER, alpha=0.88, pad=2.5))
    out_peak = max(float(np.max(b0_round_xy)), float(np.max(ann_round_xy)), EPS)
    for col, (xy, title) in enumerate([(b0_round_xy, "output at z = 60 mm"), (ann_round_xy, "output at z = 60 mm")]):
        crop, extent = _crop_xy(xy, grid, 0.34e-3)
        style.draw_xy(axes[1, col], crop, extent, title, peak=out_peak, show_y=(col == 0))
    fig.suptitle("Severely rounded axicon tip: direct illumination versus annular tip avoidance", color=style.TEXT, fontsize=16)
    pxy = out / "09_severe_rounded_tip_avoidance_XY.png"
    style.save(fig, pxy)

    # ---------- XZ only ----------
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    style.style_fig(fig)
    xz_peak = max(float(np.max(b0_round_xz)), float(np.max(ann_round_xz)), EPS)
    style.draw_xz(axes[0], b0_round_xz, XZ_COORD_M, suite.Z_VALUES_M, peak=xz_peak, show_y=True, z_ref_m=Z_REF_M)
    axes[0].set_title("ordinary B0 -> 800 um rounded tip", color=style.TEXT, fontsize=11)
    style.draw_xz(axes[1], ann_round_xz, XZ_COORD_M, suite.Z_VALUES_M, peak=xz_peak, show_y=False, z_ref_m=Z_REF_M)
    axes[1].set_title(f"annular ell=0 -> same tip\nclear radius = {selected['clear_radius_um']} um", color=style.TEXT, fontsize=11)
    fig.suptitle("Longitudinal field: severe rounded-tip interference and avoidance", color=style.TEXT, fontsize=15)
    pxz = out / "09b_severe_rounded_tip_avoidance_XZ.png"
    style.save(fig, pxz)

    # ---------- 1D only ----------
    fig, ax = plt.subplots(figsize=(9.8, 4.8), constrained_layout=True)
    style.style_fig(fig); base._style_line_axis(ax)
    ref = max(float(np.max(b0_round_xy)), float(np.max(ann_round_xy)), EPS)
    for xy, label, colour in [
        (b0_round_xy, "ordinary B0 through 800 um rounded tip", COLORS[0]),
        (ann_round_xy, f"annular ell=0 through same tip ({selected['clear_radius_um']} um clear)", COLORS[1]),
    ]:
        x, line = base._lineout_x(xy, grid, 0.0)
        keep = np.abs(x) <= 0.38e-3
        ax.plot(x[keep]*1e3, line[keep]/ref, lw=1.9, color=colour, label=label)
    ax.set_xlabel("x at fixed y = 0 (mm)")
    ax.set_ylabel("intensity / shared maximum")
    ax.set_title("Transverse intensity at z = 60 mm", color=style.TEXT, fontsize=13)
    leg = ax.legend(frameon=False, fontsize=8)
    for t in leg.get_texts(): t.set_color(style.TEXT)
    p1d = out / "09c_severe_rounded_tip_avoidance_1D.png"
    style.save(fig, p1d)

    # Strip large arrays before JSON serialisation.
    rows = []
    for c in candidates:
        rows.append({k: v for k, v in c.items() if k not in {"field", "sharp_xz", "round_xz", "round_xy"}})
    result = {
        "outcome": "SEVERE-800UM-ROUNDED-TIP-ANNULAR-L0-AVOIDANCE",
        "grid_n": int(grid_n),
        "tip_radius_um": 800,
        "selected_clear_radius_um": int(selected["clear_radius_um"]),
        "selection_mode": mode,
        "ordinary_B0": {
            "fraction_power_inside_800um_tip": _tip_fraction(field_b0, grid),
            "developed_55_to_115mm_rms_rounded_vs_sharp": b0_dev_rms,
            "developed_55_to_115mm_max_abs_rounded_vs_sharp": b0_dev_max,
            "z60_on_axis_to_peak_rounded": _centre_to_peak(b0_round_xy, grid),
        },
        "selected_annular_l0": {k: v for k, v in selected.items() if k not in {"field", "sharp_xz", "round_xz", "round_xy"}},
        "sweep_rows": rows,
        "figures": [str(pxy), str(pxz), str(p1d)],
        "claim_boundary": "annular ell=0 field is an axicon-plane planning target derived from routed B0; calibrated phase-only SLM realisation is not claimed",
    }
    (out / "severe_rounded_tip_avoidance_manifest.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-n", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/severe_rounded_tip_avoidance"))
    args = parser.parse_args()
    if args.grid_n < 1024:
        raise ValueError("severe rounded-tip evidence requires grid_n >= 1024")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    build(args.output_dir, args.grid_n)


if __name__ == "__main__":
    main()
