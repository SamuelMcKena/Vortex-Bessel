"""Final axicon evidence v4: matched sharp/rounded controls for tip avoidance.

This renderer keeps the validated v3 decentre and rounded-tip figures and
replaces the avoidance audit with a matched-control experiment.  For each
incident field (B0, routed V1, wide-core V1 planning target), the same field is
propagated through both a sharp and a 200 um rounded axicon.  The rounded/sharp
ratio therefore isolates the perturbation introduced by the rounded apex from
the ordinary Bessel-zone envelope.

Literature basis:
* Brzobohaty, Cizmar & Zemanek, Opt. Express 16, 12688-12700 (2008),
  DOI 10.1364/OE.16.012688: rounded-tip refracted light interferes with the
  quasi-Bessel field and causes longitudinal intensity modulation.
* Rao & Samanta, Opt. Lett. 43, 3029-3032 (2018),
  DOI 10.1364/OL.43.003029: a hollow input with a central dark core suppresses
  the effect of axicon-tip imperfection.

The wide-core V1 case remains an axicon-plane complex-field planning target;
this file does not claim that the current phase-only SLM route has been
calibrated to realise that exact amplitude profile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

import build_axicon_presentation_evidence_v3 as v3
import presentation_phase2j_style as style

from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.vortex_system_route import build_system_route

EPS = np.finfo(float).tiny


def _safe_ratio(num: np.ndarray, den: np.ndarray, *, floor_fraction: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Return num/den only where the matched sharp reference is meaningful."""
    d = np.asarray(den, float)
    n = np.asarray(num, float)
    valid = d >= float(floor_fraction) * max(float(np.max(d)), EPS)
    ratio = np.full_like(d, np.nan, dtype=float)
    ratio[valid] = n[valid] / np.maximum(d[valid], EPS)
    return ratio, valid


def build_tip_avoidance_matched(out: Path, grid_n: int):
    manifest = canonical_hardware_manifest()
    wavelength = float(hardware_value(manifest, "wavelength_m"))
    gamma = np.deg2rad(float(hardware_value(manifest, "axicon_base_angle_deg")))
    n_ax = float(hardware_value(manifest, "axicon_refractive_index"))
    n_ext = float(hardware_value(manifest, "axicon_external_medium_index"))

    b0 = build_system_route("B0", grid_n=grid_n)
    v1 = build_system_route("V1", grid_n=grid_n)
    grid = dict(b0["grid"])
    field_b0 = np.asarray(b0["field_on_axicon_plane"], np.complex128)
    field_v1 = np.asarray(v1["field_on_axicon_plane"], np.complex128)
    field_v1_wide = v3._wide_core_v1_target(field_v1, grid, v3.WIDE_VORTEX_CORE_M)

    sharp_t = v3._axicon_transmission(grid, wavelength, gamma, n_ax, n_ext, 0.0)
    round_t = v3._axicon_transmission(grid, wavelength, gamma, n_ax, n_ext, v3.TIP_RADIUS_M)

    cases = [
        {"label": "B0\nfull apex overlap", "kind": "B0", "incident": field_b0, "colour": v3.LINE_COLORS[0]},
        {"label": "V1\nnatural vortex null", "kind": "natural V1", "incident": field_v1, "colour": v3.LINE_COLORS[2]},
        {"label": "wide-core V1\napex-avoidance target", "kind": "wide-core V1", "incident": field_v1_wide, "colour": v3.LINE_COLORS[3]},
    ]

    for i, c in enumerate(cases):
        c["inside_tip_fraction"] = v3._fraction_inside(c["incident"], grid, v3.TIP_RADIUS_M)
        c["sharp_xy"] = v3._xy_from_post_axicon(c["incident"] * sharp_t, grid, wavelength)
        c["round_xy"] = v3._xy_from_post_axicon(c["incident"] * round_t, grid, wavelength)
        c["sharp_xz"], c["sharp_retained"] = v3._xz_from_post_axicon(
            c["incident"] * sharp_t, grid, wavelength, v3.suite.TIP_COORD_M, f"matched-sharp-{i}"
        )
        c["round_xz"], c["round_retained"] = v3._xz_from_post_axicon(
            c["incident"] * round_t, grid, wavelength, v3.suite.TIP_COORD_M, f"matched-rounded-{i}"
        )
        c["sharp_peak_z"] = np.max(c["sharp_xz"], axis=1)
        c["round_peak_z"] = np.max(c["round_xz"], axis=1)
        c["ratio_z"], c["ratio_valid"] = _safe_ratio(c["round_peak_z"], c["sharp_peak_z"])

    # Main audit: three incident fields, same rounded axicon, plus difference to matched sharp control.
    global_incident_ref = max(float(np.max(np.abs(c["incident"])**2)) for c in cases)
    global_output_ref = max(float(np.max(c["sharp_xy"])) for c in cases)
    global_diff_ref = max(float(np.max(np.abs(c["round_xy"] - c["sharp_xy"]))) for c in cases)

    fig, axes = plt.subplots(3, 3, figsize=(13.8, 10.2), constrained_layout=True)
    style.style_fig(fig)
    for col, c in enumerate(cases):
        inc = np.abs(c["incident"]) ** 2
        crop, extent = v3._fixed_crop(inc, grid, 0.65e-3)
        style.draw_xy(axes[0, col], crop, extent, c["label"], peak=global_incident_ref, show_y=(col == 0))
        axes[0, col].add_patch(Circle((0, 0), v3.TIP_RADIUS_M*1e3, fill=False, edgecolor="#39d6ad", lw=1.3, ls="--"))
        axes[0, col].text(
            0.03, 0.04, f"power inside 200 µm = {100*c['inside_tip_fraction']:.3f}%",
            transform=axes[0, col].transAxes, color=style.TEXT, fontsize=8,
            bbox=dict(facecolor=style.FIG_BG, edgecolor=style.BORDER, alpha=0.88, pad=2.5),
        )

        xy_crop, xy_extent = v3._fixed_crop(c["round_xy"], grid, 0.34e-3)
        style.draw_xy(axes[1, col], xy_crop, xy_extent, "rounded-tip output at z = 60 mm", peak=global_output_ref, show_y=(col == 0))
        matched_peak = max(float(np.max(c["sharp_xy"])), EPS)
        axes[1, col].text(
            0.03, 0.04, f"rounded/sharp matched peak = {np.max(c['round_xy'])/matched_peak:.4f}",
            transform=axes[1, col].transAxes, color=style.TEXT, fontsize=8,
            bbox=dict(facecolor=style.FIG_BG, edgecolor=style.BORDER, alpha=0.88, pad=2.5),
        )

        delta = np.abs(c["round_xy"] - c["sharp_xy"])
        d_crop, d_extent = v3._fixed_crop(delta, grid, 0.34e-3)
        style.draw_xy(axes[2, col], d_crop, d_extent, "|rounded − matched sharp|", peak=global_diff_ref, show_y=(col == 0))

    axes[0,0].set_ylabel("axicon-plane y (mm)")
    axes[1,0].set_ylabel("y at 60 mm (mm)")
    axes[2,0].set_ylabel("y at 60 mm (mm)")
    fig.suptitle("Rounded-tip axicon perturbation and vortex-based apex avoidance", color=style.TEXT, fontsize=17, y=1.035)
    fig.text(
        0.5, -0.006,
        "Dashed circle = 200 µm rounded-apex region. Each rounded result is compared against a sharp axicon with the identical incident field; no no-axicon control is used. Wide-core V1 is a planning target, not a calibrated phase-only SLM command.",
        ha="center", color=style.GOLD, fontsize=9,
    )
    pmain = out / "09_tip_avoidance_three_way_audit.png"
    style.save(fig, pmain)

    # Direct rounded-apex perturbation signature: matched rounded/sharp ratio vs z.
    fig, ax = plt.subplots(figsize=(10.6, 5.0), constrained_layout=True)
    style.style_fig(fig); v3._style_line_axis(ax)
    ax.axhline(1.0, color=style.MUTED, lw=1.0, ls="--", alpha=0.65)
    audit = []
    z = np.asarray(v3.suite.Z_VALUES_M, float)
    eval_zone = (z >= 20e-3) & (z <= 100e-3)
    for c in cases:
        ax.plot(z*1e3, c["ratio_z"], color=c["colour"], lw=1.9, label=c["kind"])
        valid = c["ratio_valid"] & eval_zone & np.isfinite(c["ratio_z"])
        residual = c["ratio_z"][valid] - 1.0
        rms = float(np.sqrt(np.mean(residual**2))) if np.any(valid) else float("nan")
        max_abs = float(np.max(np.abs(residual))) if np.any(valid) else float("nan")
        z60_sharp = max(float(np.max(c["sharp_xy"])), EPS)
        z60_ratio = float(np.max(c["round_xy"]) / z60_sharp)
        audit.append({
            "label": c["kind"],
            "fraction_incident_power_inside_200um_tip": c["inside_tip_fraction"],
            "rounded_to_sharp_peak_ratio_z60": z60_ratio,
            "rounded_to_sharp_peak_ratio_rms_deviation_20_to_100mm": rms,
            "rounded_to_sharp_peak_ratio_max_abs_deviation_20_to_100mm": max_abs,
            "sharp_fixed_support_retained_power_fraction": c["sharp_retained"],
            "rounded_fixed_support_retained_power_fraction": c["round_retained"],
        })
    ax.set_xlabel("z from axicon (mm)")
    ax.set_ylabel("matched peak ratio: rounded / sharp")
    ax.set_title("Rounded-apex perturbation isolated from the ordinary Bessel envelope", color=style.TEXT, fontsize=13)
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts(): t.set_color(style.TEXT)
    pax = out / "09b_tip_avoidance_three_way_1D_intensity.png"
    style.save(fig, pax)

    # Camera-plane perturbation: signed difference versus the matched sharp control.
    fig, ax = plt.subplots(figsize=(10.6, 5.0), constrained_layout=True)
    style.style_fig(fig); v3._style_line_axis(ax)
    ax.axhline(0.0, color=style.MUTED, lw=0.9, ls="--", alpha=0.55)
    for c in cases:
        x, rline = v3._lineout_x(c["round_xy"], grid, 0.0)
        _, sline = v3._lineout_x(c["sharp_xy"], grid, 0.0)
        ref = max(float(np.max(c["sharp_xy"])), EPS)
        keep = np.abs(x) <= 0.38e-3
        ax.plot(x[keep]*1e3, (rline[keep]-sline[keep])/ref, color=c["colour"], lw=1.8, label=c["kind"])
    ax.set_xlabel("x at fixed y = 0 (mm)")
    ax.set_ylabel("(rounded − sharp) / matched sharp peak")
    ax.set_title("Rounded-tip perturbation in the z = 60 mm transverse profile", color=style.TEXT, fontsize=13)
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts(): t.set_color(style.TEXT)
    ptrans = out / "09c_tip_avoidance_transverse_1D_intensity.png"
    style.save(fig, ptrans)
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

    p04, p04b, dec = v3.build_decentre(out, args.grid_n)
    p05, p05b, tip = v3.build_rounded_tip(out, args.grid_n)
    p09, p09b, p09c, audit = build_tip_avoidance_matched(out, args.grid_n)

    manifest = {
        "outcome": "AXICON-PRESENTATION-EVIDENCE-V4-MATCHED-SHARP-ROUNDED-CONTROLS",
        "grid_n": int(args.grid_n),
        "z_ref_m": v3.Z_REF_M,
        "figures": [str(p) for p in (p04, p04b, p05, p05b, p09, p09b, p09c)],
        "decentre_xy_policy": {
            "x_halfwidth_m": v3.DECENTRE_X_HALF_M,
            "y_halfwidth_m": v3.DECENTRE_Y_HALF_M,
            "centres_m": list(v3.DECENTRES_M),
            "axes_are_absolute_lab_coordinates": True,
            "xz_fixed_lab_y0_unchanged": True,
        },
        "decentre_1d": dec,
        "rounded_tip_1d": tip,
        "tip_avoidance_matched_control_audit": audit,
        "tip_radius_m": v3.TIP_RADIUS_M,
        "wide_vortex_core_m": v3.WIDE_VORTEX_CORE_M,
        "literature_basis": [
            "Brzobohaty, Cizmar and Zemanek, Optics Express 16, 12688-12700 (2008), DOI 10.1364/OE.16.012688",
            "Rao and Samanta, Optics Letters 43, 3029-3032 (2018), DOI 10.1364/OL.43.003029",
        ],
        "claim_boundary": "wide-core V1 is an axicon-plane complex-field planning target; calibrated phase-only SLM realisation is not claimed",
    }
    (out / "axicon_presentation_evidence_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
