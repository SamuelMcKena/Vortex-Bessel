"""Focused axicon evidence with a physically matched hollow-beam tip-avoidance test.

The decentre and rounded-tip presentation panels are inherited from
``build_axicon_presentation_evidence_v3``.  Only the tip-avoidance experiment is
replaced here.

For avoidance, the ordinary routed B0 field and a hollow *non-vortex* (ell=0)
version of that same routed field are each propagated through both a sharp and
a 200 um rounded axicon.  This gives matched sharp/rounded controls for each
incident field and directly tests whether removing illumination from the
rounded apex suppresses the rounded-tip perturbation while allowing a normal
on-axis Bessel core to reform downstream.

The hollow field is an axicon-plane planning target.  It preserves the routed
B0 phase and applies only a smooth central amplitude notch.  It is not claimed
as a calibrated phase-only SLM command.
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
TIP_RADIUS_M = 200e-6
HOLLOW_CORE_M = 260e-6
LINE_COLORS = ("#fff176", "#ff9d00", "#39d6ad", "#ff453a")


def _hollow_l0_target(field_b0: np.ndarray, grid, core_radius_m: float) -> np.ndarray:
    """Smooth hollow ell=0 target with the same pre-axicon total power as B0."""
    X = np.asarray(grid["X"], float)
    Y = np.asarray(grid["Y"], float)
    R = np.hypot(X, Y)
    notch = 1.0 - np.exp(-(R / max(float(core_radius_m), EPS)) ** 6)
    # Preserve the routed B0 phase; there is no helical exp(i*ell*phi) factor.
    target = np.asarray(field_b0, np.complex128) * notch
    p0 = float(np.sum(np.abs(field_b0) ** 2))
    p1 = float(np.sum(np.abs(target) ** 2))
    if p1 > EPS:
        target *= math.sqrt(p0 / p1)
    return target


def _tip_fraction(field: np.ndarray, grid, radius_m: float) -> float:
    R = np.hypot(np.asarray(grid["X"], float), np.asarray(grid["Y"], float))
    p = np.abs(np.asarray(field, np.complex128)) ** 2
    return float(np.sum(p[R <= radius_m]) / max(float(np.sum(p)), EPS))


def build_tip_avoidance(out: Path, grid_n: int):
    manifest = canonical_hardware_manifest()
    wavelength = float(hardware_value(manifest, "wavelength_m"))
    gamma = math.radians(float(hardware_value(manifest, "axicon_base_angle_deg")))
    n_ax = float(hardware_value(manifest, "axicon_refractive_index"))
    n_ext = float(hardware_value(manifest, "axicon_external_medium_index"))

    route = build_system_route("B0", grid_n=grid_n)
    grid = dict(route["grid"])
    field_b0 = np.asarray(route["field_on_axicon_plane"], np.complex128)
    field_hollow = _hollow_l0_target(field_b0, grid, HOLLOW_CORE_M)

    sharp_t = base._axicon_transmission(grid, wavelength, gamma, n_ax, n_ext, 0.0)
    round_t = base._axicon_transmission(grid, wavelength, gamma, n_ax, n_ext, TIP_RADIUS_M)

    cases = [
        {"label": "B0 input -> sharp tip\nmatched reference", "family": "B0", "tip": "sharp", "incident": field_b0, "post": field_b0 * sharp_t},
        {"label": "B0 input -> 200 um rounded tip\ntip-overlap case", "family": "B0", "tip": "rounded", "incident": field_b0, "post": field_b0 * round_t},
        {"label": "annular ell=0 input -> sharp tip\nmatched hollow reference", "family": "annular_l0", "tip": "sharp", "incident": field_hollow, "post": field_hollow * sharp_t},
        {"label": "annular ell=0 input -> 200 um rounded tip\ntip-avoidance case", "family": "annular_l0", "tip": "rounded", "incident": field_hollow, "post": field_hollow * round_t},
    ]

    for i, case in enumerate(cases):
        case["inside_tip_fraction"] = _tip_fraction(case["incident"], grid, TIP_RADIUS_M)
        case["xy"] = base._xy_from_post_axicon(case["post"], grid, wavelength)
        case["xz"], case["retained"] = base._xz_from_post_axicon(
            case["post"], grid, wavelength, suite.TIP_COORD_M, f"hollow-tip-avoidance-{i}"
        )

    ref_xy = max(float(np.max(cases[0]["xy"])), EPS)
    ref_xz = max(float(np.max(cases[0]["xz"])), EPS)
    incident_ref = max(float(np.max(np.abs(c["incident"]) ** 2)) for c in cases)

    # Main matched-control visual: input, z=60 mm output, and fixed-lab XZ.
    fig, axes = plt.subplots(3, 4, figsize=(15.8, 10.0), constrained_layout=True)
    style.style_fig(fig)
    for col, case in enumerate(cases):
        inc = np.abs(case["incident"]) ** 2
        crop, extent = base._fixed_crop(inc, grid, 0.65e-3)
        style.draw_xy(axes[0, col], crop, extent, case["label"], peak=incident_ref, show_y=(col == 0))
        axes[0, col].add_patch(Circle((0, 0), TIP_RADIUS_M * 1e3, fill=False, edgecolor="#39d6ad", lw=1.2, ls="--"))
        axes[0, col].text(
            0.03, 0.04, f"power inside 200 um: {100*case['inside_tip_fraction']:.3f}%",
            transform=axes[0, col].transAxes, color=style.TEXT, fontsize=8,
            bbox=dict(facecolor=style.FIG_BG, edgecolor=style.BORDER, alpha=0.88, pad=2.5),
        )

        xy_crop, xy_extent = base._fixed_crop(case["xy"], grid, 0.34e-3)
        style.draw_xy(axes[1, col], xy_crop, xy_extent, "output at z = 60 mm", peak=ref_xy, show_y=(col == 0))
        axes[1, col].text(
            0.03, 0.04, f"peak / B0-sharp = {np.max(case['xy'])/ref_xy:.3f}",
            transform=axes[1, col].transAxes, color=style.TEXT, fontsize=8,
            bbox=dict(facecolor=style.FIG_BG, edgecolor=style.BORDER, alpha=0.88, pad=2.5),
        )
        style.draw_xz(
            axes[2, col], case["xz"], suite.TIP_COORD_M, suite.Z_VALUES_M,
            peak=ref_xz, show_y=(col == 0), z_ref_m=base.Z_REF_M,
        )

    axes[0, 0].set_ylabel("axicon-plane y (mm)")
    axes[1, 0].set_ylabel("y at 60 mm (mm)")
    axes[2, 0].set_ylabel("x at fixed y=0 (mm)")
    fig.suptitle("Rounded-tip axicon: matched B0 vs annular ell=0 tip-avoidance test", color=style.TEXT, fontsize=17, y=1.01)
    fig.text(
        0.5, -0.008,
        "Each input is propagated through sharp and 200 um rounded versions of the same axicon. The annular input is non-vortex (ell=0), so a normal on-axis Bessel core may reform downstream. Dashed circle = rounded-apex region.",
        ha="center", color=style.GOLD, fontsize=9,
    )
    pmain = out / "09_tip_avoidance_three_way_audit.png"
    style.save(fig, pmain)

    # Direct rounded/sharp perturbation for each matched incident field.
    b0_sharp = np.max(cases[0]["xz"], axis=1)
    b0_round = np.max(cases[1]["xz"], axis=1)
    hollow_sharp = np.max(cases[2]["xz"], axis=1)
    hollow_round = np.max(cases[3]["xz"], axis=1)
    b0_ratio = b0_round / np.maximum(b0_sharp, EPS)
    hollow_ratio = hollow_round / np.maximum(hollow_sharp, EPS)

    fig, ax = plt.subplots(figsize=(10.4, 5.0), constrained_layout=True)
    style.style_fig(fig); base._style_line_axis(ax)
    ax.plot(suite.Z_VALUES_M * 1e3, b0_ratio, color=LINE_COLORS[1], lw=1.9, label="ordinary B0: rounded / sharp")
    ax.plot(suite.Z_VALUES_M * 1e3, hollow_ratio, color=LINE_COLORS[2], lw=1.9, label="annular ell=0: rounded / sharp")
    ax.axhline(1.0, color=style.MUTED, lw=0.9, ls="--", alpha=0.7)
    ax.set_xlabel("z from axicon (mm)")
    ax.set_ylabel("matched peak-intensity ratio")
    ax.set_title("Rounded-tip perturbation versus z - direct matched-input comparison", color=style.TEXT, fontsize=13)
    leg = ax.legend(frameon=False, fontsize=9)
    for text in leg.get_texts(): text.set_color(style.TEXT)
    pax = out / "09b_tip_avoidance_three_way_1D_intensity.png"
    style.save(fig, pax)

    # Camera-plane transverse lineouts.
    fig, ax = plt.subplots(figsize=(10.4, 5.0), constrained_layout=True)
    style.style_fig(fig); base._style_line_axis(ax)
    for case, colour in zip(cases, LINE_COLORS):
        x, line = base._lineout_x(case["xy"], grid, 0.0)
        keep = np.abs(x) <= 0.38e-3
        ax.plot(x[keep] * 1e3, line[keep] / ref_xy, color=colour, lw=1.7, label=case["label"].replace("\n", " - "))
    ax.set_xlabel("x at fixed y = 0 (mm)")
    ax.set_ylabel("intensity / B0-sharp z=60 mm peak")
    ax.set_title("Tip avoidance - transverse 1D intensity at z = 60 mm", color=style.TEXT, fontsize=13)
    leg = ax.legend(frameon=False, fontsize=8)
    for text in leg.get_texts(): text.set_color(style.TEXT)
    ptrans = out / "09c_tip_avoidance_transverse_1D_intensity.png"
    style.save(fig, ptrans)

    z = np.asarray(suite.Z_VALUES_M, float)
    window = (z >= 20e-3) & (z <= 100e-3)
    b0_dev = b0_ratio[window] - 1.0
    hollow_dev = hollow_ratio[window] - 1.0
    audit = [
        {
            "input_family": "B0",
            "fraction_incident_power_inside_200um_tip": cases[1]["inside_tip_fraction"],
            "z60_rounded_to_sharp_peak_ratio": float(np.max(cases[1]["xy"]) / max(float(np.max(cases[0]["xy"])), EPS)),
            "rounded_vs_sharp_rms_fractional_deviation_20_to_100mm": float(np.sqrt(np.mean(b0_dev ** 2))),
            "rounded_vs_sharp_max_abs_fractional_deviation_20_to_100mm": float(np.max(np.abs(b0_dev))),
        },
        {
            "input_family": "annular_l0",
            "fraction_incident_power_inside_200um_tip": cases[3]["inside_tip_fraction"],
            "z60_rounded_to_sharp_peak_ratio": float(np.max(cases[3]["xy"]) / max(float(np.max(cases[2]["xy"])), EPS)),
            "rounded_vs_sharp_rms_fractional_deviation_20_to_100mm": float(np.sqrt(np.mean(hollow_dev ** 2))),
            "rounded_vs_sharp_max_abs_fractional_deviation_20_to_100mm": float(np.max(np.abs(hollow_dev))),
        },
    ]
    return pmain, pax, ptrans, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-n", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/axicon_presentation_evidence"))
    args = parser.parse_args()
    if args.grid_n < 1024:
        raise ValueError("presentation evidence requires grid_n >= 1024")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    p04, p04b, dec = base.build_decentre(out, args.grid_n)
    p05, p05b, tip = base.build_rounded_tip(out, args.grid_n)
    p09, p09b, p09c, audit = build_tip_avoidance(out, args.grid_n)

    manifest = {
        "outcome": "AXICON-PRESENTATION-EVIDENCE-V5-HOLLOW-L0-MATCHED-CONTROL",
        "grid_n": int(args.grid_n),
        "z_ref_m": base.Z_REF_M,
        "figures": [str(p) for p in (p04, p04b, p05, p05b, p09, p09b, p09c)],
        "decentre_xy_policy": {
            "x_halfwidth_m": base.DECENTRE_X_HALF_M,
            "y_halfwidth_m": base.DECENTRE_Y_HALF_M,
            "centres_m": list(base.DECENTRES_M),
            "axes_are_absolute_lab_coordinates": True,
            "xz_fixed_lab_y0_unchanged": True,
        },
        "decentre_1d": dec,
        "rounded_tip_1d": tip,
        "tip_avoidance_audit": audit,
        "tip_radius_m": TIP_RADIUS_M,
        "hollow_core_m": HOLLOW_CORE_M,
        "literature_basis": [
            "Brzobohaty et al., Optics Express 16, 12688-12700 (2008), DOI 10.1364/OE.16.012688",
            "Rao and Samanta, Optics Letters 43, 3029-3032 (2018), hollow-input suppression of axicon-tip modulation",
        ],
        "claim_boundary": "annular ell=0 input is an axicon-plane planning target derived from the routed B0 field; calibrated phase-only SLM realisation is not claimed",
    }
    (out / "axicon_presentation_evidence_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
