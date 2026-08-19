"""Optimise the hollow ell=0 input radius for a 200 um rounded axicon tip.

The sweep is deliberately evaluated against matched sharp-tip controls for the
same incident field.  It separates the early formation region from the
established Bessel region so that a large hollow core cannot look artificially
'good' simply by delaying beam formation.

Selection policy
----------------
For each hollow-core scale r0, record:
  * incident-power fraction inside the physical 200 um apex region;
  * RMS and maximum rounded/sharp peak-intensity perturbation in 20-50 mm;
  * RMS and maximum rounded/sharp perturbation in 50-100 mm;
  * mean sharp-tip peak intensity in 50-100 mm relative to ordinary B0;
  * z=60 mm on-axis / transverse-peak ratio for the rounded case.

The selected radius is the smallest tested radius that:
  1. puts <=0.10% of incident power inside the 200 um apex;
  2. keeps the developed-region rounded/sharp RMS perturbation <= the ordinary
     B0 rounded-tip perturbation;
  3. preserves >=50% of the ordinary B0 mean developed-region peak intensity;
  4. gives an on-axis-bright, non-vortex output at z=60 mm (centre/peak >=0.8).
If no radius satisfies all four conditions, the minimum penalty score is used
and the manifest marks the selection as fallback rather than threshold-passing.

The hollow field is an axicon-plane planning target derived from the routed B0
field.  No calibrated phase-only SLM realisation is claimed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import build_axicon_hollow_avoidance as avoid
import build_axicon_presentation_evidence_v3 as base
import build_phase2j_presentation_suite as suite
import presentation_phase2j_style as style

from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.vortex_system_route import build_system_route

EPS = np.finfo(float).tiny
TIP_RADIUS_M = 200e-6
DEFAULT_RADII_UM = (160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360)


def _window_metrics(ratio: np.ndarray, z: np.ndarray, z0: float, z1: float) -> tuple[float, float]:
    keep = (z >= z0) & (z <= z1)
    dev = np.asarray(ratio[keep], float) - 1.0
    return float(np.sqrt(np.mean(dev**2))), float(np.max(np.abs(dev)))


def _central_ratio(xy: np.ndarray, grid) -> float:
    x = np.asarray(grid["x"], float)
    ix = int(np.argmin(np.abs(x)))
    centre = float(np.asarray(xy, float)[ix, ix])
    peak = max(float(np.max(xy)), EPS)
    return centre / peak


def run_sweep(out: Path, grid_n: int, radii_um: list[int]) -> dict:
    manifest = canonical_hardware_manifest()
    wavelength = float(hardware_value(manifest, "wavelength_m"))
    gamma = math.radians(float(hardware_value(manifest, "axicon_base_angle_deg")))
    n_ax = float(hardware_value(manifest, "axicon_refractive_index"))
    n_ext = float(hardware_value(manifest, "axicon_external_medium_index"))

    route = build_system_route("B0", grid_n=grid_n)
    grid = dict(route["grid"])
    field_b0 = np.asarray(route["field_on_axicon_plane"], np.complex128)
    sharp_t = base._axicon_transmission(grid, wavelength, gamma, n_ax, n_ext, 0.0)
    round_t = base._axicon_transmission(grid, wavelength, gamma, n_ax, n_ext, TIP_RADIUS_M)

    b0_sharp_xz, _ = base._xz_from_post_axicon(field_b0 * sharp_t, grid, wavelength, suite.TIP_COORD_M, "sweep-b0-sharp")
    b0_round_xz, _ = base._xz_from_post_axicon(field_b0 * round_t, grid, wavelength, suite.TIP_COORD_M, "sweep-b0-round")
    z = np.asarray(suite.Z_VALUES_M, float)
    b0_sharp_peak = np.max(b0_sharp_xz, axis=1)
    b0_round_peak = np.max(b0_round_xz, axis=1)
    b0_ratio = b0_round_peak / np.maximum(b0_sharp_peak, EPS)
    b0_early_rms, b0_early_max = _window_metrics(b0_ratio, z, 20e-3, 50e-3)
    b0_dev_rms, b0_dev_max = _window_metrics(b0_ratio, z, 50e-3, 100e-3)
    developed = (z >= 50e-3) & (z <= 100e-3)
    b0_mean_developed_peak = float(np.mean(b0_sharp_peak[developed]))

    rows = []
    for radius_um in radii_um:
        r0 = float(radius_um) * 1e-6
        hollow = avoid._hollow_l0_target(field_b0, grid, r0)
        h_sharp = hollow * sharp_t
        h_round = hollow * round_t
        sharp_xz, _ = base._xz_from_post_axicon(h_sharp, grid, wavelength, suite.TIP_COORD_M, f"sweep-hollow-{radius_um}-sharp")
        round_xz, _ = base._xz_from_post_axicon(h_round, grid, wavelength, suite.TIP_COORD_M, f"sweep-hollow-{radius_um}-round")
        sharp_peak = np.max(sharp_xz, axis=1)
        round_peak = np.max(round_xz, axis=1)
        ratio = round_peak / np.maximum(sharp_peak, EPS)
        early_rms, early_max = _window_metrics(ratio, z, 20e-3, 50e-3)
        dev_rms, dev_max = _window_metrics(ratio, z, 50e-3, 100e-3)
        sharp_mean = float(np.mean(sharp_peak[developed]))
        xy_round = base._xy_from_post_axicon(h_round, grid, wavelength)
        centre_ratio = _central_ratio(xy_round, grid)
        tip_fraction = avoid._tip_fraction(hollow, grid, TIP_RADIUS_M)
        rows.append({
            "core_radius_um": int(radius_um),
            "fraction_incident_power_inside_200um_tip": float(tip_fraction),
            "early_20_to_50mm_rms_rounded_vs_sharp": early_rms,
            "early_20_to_50mm_max_abs_rounded_vs_sharp": early_max,
            "developed_50_to_100mm_rms_rounded_vs_sharp": dev_rms,
            "developed_50_to_100mm_max_abs_rounded_vs_sharp": dev_max,
            "mean_sharp_peak_50_to_100mm_vs_B0": sharp_mean / max(b0_mean_developed_peak, EPS),
            "z60_on_axis_to_transverse_peak_rounded": centre_ratio,
        })

    for row in rows:
        row["passes_thresholds"] = bool(
            row["fraction_incident_power_inside_200um_tip"] <= 1e-3
            and row["developed_50_to_100mm_rms_rounded_vs_sharp"] <= b0_dev_rms
            and row["mean_sharp_peak_50_to_100mm_vs_B0"] >= 0.50
            and row["z60_on_axis_to_transverse_peak_rounded"] >= 0.80
        )
        row["penalty_score"] = float(
            2.0 * row["developed_50_to_100mm_rms_rounded_vs_sharp"] / max(b0_dev_rms, 1e-6)
            + 0.35 * row["early_20_to_50mm_rms_rounded_vs_sharp"] / max(b0_early_rms, 1e-6)
            + 0.35 * max(0.0, 0.50 - row["mean_sharp_peak_50_to_100mm_vs_B0"]) / 0.50
            + 0.50 * max(0.0, 0.80 - row["z60_on_axis_to_transverse_peak_rounded"]) / 0.80
            + 0.25 * min(10.0, row["fraction_incident_power_inside_200um_tip"] / 1e-3)
        )

    passing = [r for r in rows if r["passes_thresholds"]]
    if passing:
        selected = min(passing, key=lambda r: r["core_radius_um"])
        selection_mode = "smallest_radius_passing_all_thresholds"
    else:
        selected = min(rows, key=lambda r: r["penalty_score"])
        selection_mode = "minimum_penalty_fallback"

    result = {
        "outcome": "HOLLOW-L0-ROUNDED-TIP-RADIUS-SWEEP",
        "grid_n": int(grid_n),
        "tip_radius_um": 200,
        "radii_um": list(map(int, radii_um)),
        "ordinary_B0_reference": {
            "early_20_to_50mm_rms_rounded_vs_sharp": b0_early_rms,
            "early_20_to_50mm_max_abs_rounded_vs_sharp": b0_early_max,
            "developed_50_to_100mm_rms_rounded_vs_sharp": b0_dev_rms,
            "developed_50_to_100mm_max_abs_rounded_vs_sharp": b0_dev_max,
        },
        "selection_mode": selection_mode,
        "selected_core_radius_um": int(selected["core_radius_um"]),
        "selected_row": selected,
        "rows": rows,
        "claim_boundary": "hollow ell=0 field is an axicon-plane planning target derived from routed B0; calibrated phase-only SLM realisation is not claimed",
    }

    # Four-panel sweep diagnostic.
    radii = np.array([r["core_radius_um"] for r in rows], float)
    tip_pct = 100*np.array([r["fraction_incident_power_inside_200um_tip"] for r in rows], float)
    dev_rms = np.array([r["developed_50_to_100mm_rms_rounded_vs_sharp"] for r in rows], float)
    early_rms = np.array([r["early_20_to_50mm_rms_rounded_vs_sharp"] for r in rows], float)
    useful = np.array([r["mean_sharp_peak_50_to_100mm_vs_B0"] for r in rows], float)
    centre = np.array([r["z60_on_axis_to_transverse_peak_rounded"] for r in rows], float)

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), constrained_layout=True)
    style.style_fig(fig)
    for ax in axes.ravel():
        base._style_line_axis(ax)
        ax.axvline(selected["core_radius_um"], color=style.GOLD, lw=1.0, ls="--", alpha=0.8)

    axes[0,0].plot(radii, tip_pct, marker="o")
    axes[0,0].axhline(0.10, color=style.MUTED, lw=0.8, ls=":")
    axes[0,0].set_ylabel("incident power inside 200 um (%)")
    axes[0,0].set_title("Apex loading", color=style.TEXT)

    axes[0,1].plot(radii, early_rms, marker="o", label="20-50 mm")
    axes[0,1].plot(radii, dev_rms, marker="s", label="50-100 mm")
    axes[0,1].axhline(b0_dev_rms, color=style.MUTED, lw=0.8, ls=":", label="B0 developed RMS")
    axes[0,1].set_ylabel("RMS rounded/sharp fractional deviation")
    axes[0,1].set_title("Rounded-tip perturbation", color=style.TEXT)
    leg=axes[0,1].legend(frameon=False, fontsize=8)
    for t in leg.get_texts(): t.set_color(style.TEXT)

    axes[1,0].plot(radii, useful, marker="o")
    axes[1,0].axhline(0.50, color=style.MUTED, lw=0.8, ls=":")
    axes[1,0].set_ylabel("mean sharp peak / B0 (50-100 mm)")
    axes[1,0].set_xlabel("hollow-core scale r0 (um)")
    axes[1,0].set_title("Useful developed-beam strength", color=style.TEXT)

    axes[1,1].plot(radii, centre, marker="o")
    axes[1,1].axhline(0.80, color=style.MUTED, lw=0.8, ls=":")
    axes[1,1].set_ylabel("z=60 mm on-axis / transverse peak")
    axes[1,1].set_xlabel("hollow-core scale r0 (um)")
    axes[1,1].set_title("Normal on-axis Bessel recovery", color=style.TEXT)

    fig.suptitle(f"Hollow ell=0 rounded-tip sweep - selected r0 = {selected['core_radius_um']} um", color=style.TEXT, fontsize=16)
    sweep_png = out / "10_hollow_tip_avoidance_radius_sweep.png"
    style.save(fig, sweep_png)
    result["figure"] = str(sweep_png)

    (out / "hollow_tip_avoidance_sweep_manifest.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    (out / "selected_hollow_core_um.txt").write_text(f"{selected['core_radius_um']}\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-n", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/axicon_hollow_sweep"))
    parser.add_argument("--radii-um", type=int, nargs="*", default=list(DEFAULT_RADII_UM))
    args = parser.parse_args()
    if args.grid_n < 1024:
        raise ValueError("sweep requires grid_n >= 1024")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run_sweep(args.output_dir, args.grid_n, list(args.radii_um))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
