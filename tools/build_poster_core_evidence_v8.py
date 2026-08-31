"""Poster figure quality gate v8.

This pass deliberately does *not* rebuild the poster.  It converts the current
figure curation into an explicit scientific + visual gate:

* self-healing is rerendered in a tighter poster composition using the same
  simulated field, with a black->red->orange->yellow intensity language;
* the simultaneous axicon/4F-iris benchmark is reported as an identifiability
  test rather than as two successful recovered parameters;
* q=20 experimental diagnostic rasters are audited but are not promoted to
  poster-ready status merely because they are canonical/tracked outputs.

The important methodological rule is that a low optimiser/grid-search loss is
not sufficient to report a physical bench parameter.  A parameter must also be
supported by the explored cost landscape and by the synthetic truth test before
it is allowed through the reporting gate.
"""

from __future__ import annotations

from pathlib import Path
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np

import build_poster_core_evidence_v4 as v4
import build_poster_core_evidence_v7 as v7

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "poster" / "core_evidence_v8"
EPS = np.finfo(float).tiny

# Deliberately avoids the purple low-intensity floor of inferno.  This is only a
# display mapping; no simulated or measured values are modified.
POSTER_THERMAL = LinearSegmentedColormap.from_list(
    "poster_thermal",
    [
        (0.00, "#000000"),
        (0.10, "#080000"),
        (0.28, "#430000"),
        (0.50, "#a51600"),
        (0.72, "#f05a00"),
        (0.88, "#ffb000"),
        (1.00, "#fff2a8"),
    ],
    N=256,
)


def _style(ax: plt.Axes) -> None:
    v4._style(ax)
    ax.grid(False)


def build_self_healing_quality(out: Path) -> tuple[Path, dict]:
    """Tighter self-healing composition using the unchanged v4 propagation."""
    cfg = v4.vbb_studies.beam_air_config(v4.bt.default_config("balanced"))
    cfg = v4.replace(cfg, target=v4.replace(cfg.target, ell=0))
    design = v4.bt.compute_design_from_config(cfg)
    first_zero = float(design.equivalent_l0_first_zero_radius_m)
    obstacle_radius = 3.5 * first_zero

    bundle = v4.pdiag.build_self_healing_bundle(
        config=cfg,
        preset="balanced",
        path="ideal",
        case_id="poster_B0_self_healing_quality_gate_v8",
        obstacle_kind="disk",
        obstacle_radius_m=obstacle_radius,
        axial_points=151,
    )

    vol = bundle["obstructed_volume"]
    cgrid = vol["crop_grid"]
    x_um = np.asarray(cgrid["x"], float) / v4.bt.um
    z_um = np.asarray(bundle["z_relative"], float) / v4.bt.um
    xz = np.asarray(vol["xz"], float)
    ref = v4._crop_plane(bundle["reference_plane"], bundle["grid"], cgrid)
    blocked = v4._crop_plane(bundle["obstructed_plane"], bundle["grid"], cgrid)
    stack = np.asarray(vol["intensity_stack"], float)
    recovery = np.asarray(bundle["onaxis_recovery"], float)

    ids = np.flatnonzero((recovery >= 0.80) & np.isfinite(recovery))
    if ids.size:
        recovered_idx = int(ids[0])
    else:
        search0 = max(1, int(0.08 * len(recovery)))
        recovered_idx = int(search0 + np.nanargmax(recovery[search0:]))
    recovered = stack[recovered_idx]

    shared = float(np.max(ref))
    extent_xy = [x_um[0], x_um[-1], x_um[0], x_um[-1]]
    extent_xz = [z_um[0], z_um[-1], x_um[0], x_um[-1]]

    k0 = 2.0 * np.pi / float(cfg.laser.wavelength_m)
    theta = math.asin(min(0.999999, abs(float(design.kr_sample_m_inv)) / k0))
    z_geom_um = float(obstacle_radius / max(math.tan(theta), EPS) / v4.bt.um)
    r_um = obstacle_radius / v4.bt.um

    fig = plt.figure(figsize=(15.4, 6.45), facecolor=v4.DARK)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.58, 1.0], wspace=0.15,
                             left=0.055, right=0.985, top=0.865, bottom=0.115)

    ax_xz = fig.add_subplot(outer[0, 0])
    _style(ax_xz)
    ax_xz.imshow(v4._norm(xz) ** 0.48, origin="lower", aspect="auto",
                 extent=extent_xz, cmap=POSTER_THERMAL, vmin=0, vmax=1,
                 interpolation="nearest")
    ax_xz.set_xlabel("distance after obstruction, z (µm)", fontsize=10)
    ax_xz.set_ylabel("x at fixed y = 0 (µm)", fontsize=10)
    ax_xz.set_title("Longitudinal reconstruction", fontsize=13.2, weight="bold", pad=7)

    z_span = max(float(z_um[-1] - z_um[0]), 1.0)
    obs_width = 0.020 * z_span
    ax_xz.add_patch(Rectangle((float(z_um[0]), -r_um), obs_width, 2 * r_um,
                              facecolor="black", edgecolor="white", linewidth=1.25, zorder=6))
    ax_xz.text(float(z_um[0]) + 0.032*z_span, 1.20*r_um,
               f"opaque disk  Ø {2*r_um:.1f} µm", color=v4.FG, fontsize=9.0,
               ha="left", va="center")
    z_guide = min(float(z_um[0] + z_geom_um), float(z_um[-1]))
    ax_xz.plot([z_um[0], z_guide], [r_um, 0], color=v4.CYAN, ls="--", lw=1.45, alpha=.85)
    ax_xz.plot([z_um[0], z_guide], [-r_um, 0], color=v4.CYAN, ls="--", lw=1.45, alpha=.85)
    ax_xz.scatter([z_guide], [0], s=20, color=v4.CYAN, zorder=7)
    ax_xz.axvline(z_um[recovered_idx], color=v4.GREEN, lw=1.0, ls=":", alpha=.85)
    ax_xz.text(z_um[recovered_idx], 0.92*x_um[-1], "selected recovery",
               color=v4.GREEN, fontsize=8.4, ha="center", va="top")

    right = outer[0, 1].subgridspec(2, 3, height_ratios=[1.0, 0.34],
                                     hspace=0.28, wspace=0.10)
    transverse = [
        (ref, "reference"),
        (blocked, "blocked"),
        (recovered, f"recovered\n{z_um[recovered_idx]:.0f} µm"),
    ]
    for col, (plane, title) in enumerate(transverse):
        ax = fig.add_subplot(right[0, col])
        _style(ax)
        ax.imshow(v4._norm(plane, shared) ** 0.48, origin="lower", extent=extent_xy,
                  cmap=POSTER_THERMAL, vmin=0, vmax=1, interpolation="nearest")
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10.8, weight="bold", pad=5)
        ax.set_xlabel("x (µm)", fontsize=8.5)
        if col == 0:
            ax.set_ylabel("y (µm)", fontsize=8.5)
        else:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelsize=7.3)

    ax_r = fig.add_subplot(right[1, :])
    _style(ax_r)
    ax_r.plot(z_um, bundle["onaxis_recovery"], color=v4.GOLD, lw=2.0,
              label="on-axis recovery")
    ax_r.axhline(1.0, color=v4.MUTED, lw=.9, ls=":")
    ax_r.axvline(z_um[recovered_idx], color=v4.GREEN, lw=1.0, ls="--")
    ax_r.set_xlabel("z after obstruction (µm)", fontsize=8.7)
    ax_r.set_ylabel("I / reference", fontsize=8.7)
    ax_r.set_ylim(0, min(1.35, max(1.10, 1.04*float(np.nanmax(recovery)))))
    ax_r.tick_params(labelsize=7.3)
    ax_r.legend(frameon=False, labelcolor=v4.FG, fontsize=7.8, loc="lower right")

    fig.suptitle("Bessel self-healing after a finite central obstruction",
                 color=v4.FG, fontsize=18.0, weight="bold", y=0.976)
    fig.text(0.5, 0.915,
             "Simulated field only — black disk and dashed conical guides are annotations; intensity data are unchanged.",
             ha="center", color=v4.MUTED, fontsize=9.4)

    path = v4._save(fig, out / "01_self_healing_poster_quality.png", dpi=330)
    meta = {
        "status": "PASS_POSTER_CANDIDATE",
        "first_zero_radius_um": float(first_zero / v4.bt.um),
        "obstacle_radius_um": float(r_um),
        "obstacle_diameter_um": float(2*r_um),
        "obstacle_radius_in_first_zero_radii": 3.5,
        "geometric_reconstruction_distance_um": z_geom_um,
        "selected_recovered_plane_um": float(z_um[recovered_idx]),
        "selected_onaxis_recovery": float(recovery[recovered_idx]),
        "display_change_only": "custom black-red-orange-yellow colormap and tighter composition",
        "claim_boundary": "heatmap is simulated wave propagation; obstruction/cone/recovery markers are annotations",
    }
    return path, meta


def _profile_delta(profile: np.ndarray) -> np.ndarray:
    p = np.asarray(profile, float)
    return (p - float(np.min(p))) / max(float(np.min(p)), EPS)


def build_identifiability_gate(out: Path) -> tuple[Path, dict, Path]:
    """Turn v7 into an honest report/withhold identifiability result."""
    raw_path, summary, csv_path = v7.build_metric_aware_recovery(out)
    fine = summary["fine_fit"]
    x = np.asarray(fine["values_x"], float)
    y = np.asarray(fine["values_y"], float)
    costs = np.asarray(fine["costs"], float)
    iy, ix = [int(v) for v in fine["best_index_yx"]]

    truth_x = float(summary["truth"]["axicon_decentre_x_um"])
    truth_y = float(summary["truth"]["fourf_iris_offset_x_radii"])
    best_x = float(fine["best_x"])
    best_y = float(fine["best_y"])
    margin = float(fine["relative_cost_margin"])

    prof_x = np.min(costs, axis=0)
    prof_y = np.min(costs, axis=1)
    dy = _profile_delta(prof_y)
    dx = _profile_delta(prof_x)

    x_step = float(np.median(np.diff(x))) if len(x) > 1 else np.inf
    axicon_grid_consistent = abs(best_x - truth_x) <= 0.5*abs(x_step) + 1e-9
    iris_on_boundary = iy in (0, len(y)-1)
    iris_truth_missed = abs(best_y - truth_y) > 0.5*abs(float(np.median(np.diff(y)))) + 1e-9
    weak_global_separation = margin < 0.02
    iris_reportable = not (iris_on_boundary or iris_truth_missed or weak_global_separation)

    fig = plt.figure(figsize=(15.4, 7.8), facecolor=v4.DARK)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 0.88, 1.05],
                          height_ratios=[1.0, 1.0], hspace=0.30, wspace=0.26,
                          left=0.06, right=0.985, top=0.865, bottom=0.10)

    ax_cost = fig.add_subplot(gs[:, 0])
    _style(ax_cost)
    im = ax_cost.imshow(costs, origin="lower", aspect="auto",
                        extent=[x[0], x[-1], y[0], y[-1]], cmap=v4.DIFF)
    ax_cost.scatter([truth_x], [truth_y], marker="x", s=100, color=v4.RED,
                    linewidths=2.2, label="injected truth")
    ax_cost.scatter([best_x], [best_y], marker="o", s=78, facecolors="none",
                    edgecolors=v4.FG, linewidths=1.8, label="grid minimum")
    ax_cost.set_xlabel("axicon decentre x (µm)")
    ax_cost.set_ylabel("4F iris offset x (iris radii)")
    ax_cost.set_title("Joint synthetic inverse landscape", fontsize=13, weight="bold")
    ax_cost.legend(frameon=False, labelcolor=v4.FG, fontsize=8.5, loc="upper left")
    cb = fig.colorbar(im, ax=ax_cost, fraction=0.047, pad=0.025)
    cb.set_label("composite loss", color=v4.MUTED, fontsize=8.5)
    cb.ax.tick_params(colors=v4.MUTED, labelsize=7.5)
    cb.outline.set_edgecolor("#51606d")

    ax_x = fig.add_subplot(gs[0, 1])
    _style(ax_x)
    ax_x.plot(x, 100*dx, "o-", color=v4.CYAN, lw=2.0, ms=4.8)
    ax_x.axvline(truth_x, color=v4.RED, ls="--", lw=1.1, label="truth")
    ax_x.axvline(best_x, color=v4.FG, ls=":", lw=1.1, label="minimum")
    ax_x.set_title("Profiled axicon direction", fontsize=11.6, weight="bold")
    ax_x.set_xlabel("decentre x (µm)")
    ax_x.set_ylabel("loss above minimum (%)")
    ax_x.legend(frameon=False, labelcolor=v4.FG, fontsize=7.8)

    ax_y = fig.add_subplot(gs[1, 1])
    _style(ax_y)
    ax_y.plot(y, 100*dy, "o-", color=v4.GOLD, lw=2.0, ms=4.8)
    ax_y.axvline(truth_y, color=v4.RED, ls="--", lw=1.1, label="truth")
    ax_y.axvline(best_y, color=v4.FG, ls=":", lw=1.1, label="minimum")
    ax_y.set_title("Profiled iris direction", fontsize=11.6, weight="bold")
    ax_y.set_xlabel("iris offset x (R)")
    ax_y.set_ylabel("loss above minimum (%)")
    ax_y.legend(frameon=False, labelcolor=v4.FG, fontsize=7.8)

    ax_gate = fig.add_subplot(gs[:, 2])
    ax_gate.set_facecolor(v4.DARK)
    ax_gate.axis("off")
    ax_gate.text(0.02, 0.97, "IDENTIFIABILITY GATE", color=v4.FG,
                 fontsize=14.2, weight="bold", va="top")

    # Card 1: axicon survives this synthetic benchmark.
    ax_gate.add_patch(Rectangle((0.01, 0.55), 0.98, 0.31, transform=ax_gate.transAxes,
                                facecolor="#0d1b16", edgecolor=v4.GREEN, linewidth=1.4))
    ax_gate.text(0.05, 0.815, "REPORT", transform=ax_gate.transAxes,
                 color=v4.GREEN, fontsize=12.2, weight="bold", va="top")
    ax_gate.text(0.05, 0.754, "Axicon lateral decentre", transform=ax_gate.transAxes,
                 color=v4.FG, fontsize=11.4, weight="bold", va="top")
    ax_gate.text(0.05, 0.685,
                 f"injected  {truth_x:+.0f} µm\nrecovered {best_x:+.0f} µm\ngrid error   {abs(best_x-truth_x):.0f} µm",
                 transform=ax_gate.transAxes, color=v4.MUTED, fontsize=10.0,
                 va="top", linespacing=1.45)
    ax_gate.text(0.05, 0.575,
                 "Discrete synthetic estimate — not a confidence interval.",
                 transform=ax_gate.transAxes, color=v4.CYAN, fontsize=8.3, va="bottom")

    # Card 2: iris is deliberately withheld.
    ax_gate.add_patch(Rectangle((0.01, 0.13), 0.98, 0.34, transform=ax_gate.transAxes,
                                facecolor="#21100d", edgecolor=v4.RED, linewidth=1.4))
    ax_gate.text(0.05, 0.425, "WITHHOLD", transform=ax_gate.transAxes,
                 color=v4.RED, fontsize=12.2, weight="bold", va="top")
    ax_gate.text(0.05, 0.365, "4F iris lateral offset", transform=ax_gate.transAxes,
                 color=v4.FG, fontsize=11.4, weight="bold", va="top")
    reasons = []
    if iris_on_boundary:
        reasons.append("minimum lies on search boundary")
    if iris_truth_missed:
        reasons.append(f"truth {truth_y:+.2f} R -> fit {best_y:+.2f} R")
    if weak_global_separation:
        reasons.append(f"best/2nd separation only {100*margin:.1f}%")
    ax_gate.text(0.05, 0.300, "\n".join("• " + r for r in reasons),
                 transform=ax_gate.transAxes, color=v4.MUTED, fontsize=9.1,
                 va="top", linespacing=1.45)
    ax_gate.text(0.05, 0.155,
                 "Result: NOT IDENTIFIABLE FROM THIS BENCHMARK",
                 transform=ax_gate.transAxes, color=v4.GOLD, fontsize=8.7,
                 weight="bold", va="bottom")

    ax_gate.text(0.02, 0.045,
                 "Rule: optimiser minimum ≠ physical diagnosis.\nOnly parameters surviving the gate proceed to reporting.",
                 transform=ax_gate.transAxes, color=v4.MUTED, fontsize=8.8,
                 va="bottom", linespacing=1.35)

    fig.suptitle("Physical-error inference must pass an identifiability gate",
                 color=v4.FG, fontsize=18.0, weight="bold", y=0.975)
    fig.text(0.5, 0.918,
             "17-plane V1 synthetic stack, simultaneous off-grid errors + noise; morphology and calibrated-throughput signatures are combined only for this synthetic benchmark.",
             ha="center", color=v4.MUTED, fontsize=9.2)

    path = v4._save(fig, out / "02_physical_identifiability_gate.png", dpi=330)
    gate = {
        "status": "PASS_METHOD_FIGURE",
        "source_raw_v7_figure": str(raw_path),
        "axicon": {
            "decision": "REPORT_SYNTHETIC_ESTIMATE" if axicon_grid_consistent else "WITHHOLD",
            "truth_um": truth_x,
            "recovered_um": best_x,
            "absolute_grid_error_um": abs(best_x-truth_x),
        },
        "fourf_iris": {
            "decision": "REPORT_SYNTHETIC_ESTIMATE" if iris_reportable else "WITHHOLD_NOT_IDENTIFIABLE",
            "truth_radii": truth_y,
            "recovered_radii": best_y,
            "minimum_on_search_boundary": bool(iris_on_boundary),
            "truth_missed_by_more_than_half_grid_step": bool(iris_truth_missed),
        },
        "joint_best_second_relative_margin": margin,
        "claim_boundary": "synthetic identifiability demonstration only; no experimental parameter or statistical uncertainty is claimed",
    }
    return path, gate, csv_path


def write_audit(out: Path, self_meta: dict, gate: dict) -> Path:
    audit = {
        "outcome": "POSTER-FIGURE-QUALITY-GATE-V8",
        "poster_rebuild_allowed": False,
        "poster_rebuild_blocker": "q20 experimental figures still require poster-specific rerender/claim review from source arrays; diagnostic rasters are not promoted automatically",
        "candidates": [
            {
                "figure": "01_self_healing_poster_quality.png",
                "decision": "PASS_POSTER_CANDIDATE",
                "science": "PASS",
                "visual": "PASS",
                "note": "same propagation data; tighter composition and display-only thermal remap",
            },
            {
                "figure": "02_physical_identifiability_gate.png",
                "decision": "PASS_METHOD_FIGURE",
                "science": "PASS_WITH_CLAIM_BOUNDARY",
                "visual": "PASS",
                "note": "axicon is reportable in the synthetic truth test; iris is explicitly withheld",
            },
            {
                "figure": "figures/experimental/q20_aberration/reconstruction/annular_aberration_phase.png",
                "decision": "RERENDER_REQUIRED",
                "science": "PASS_DIAGNOSTIC",
                "visual": "FAIL_POSTER",
                "note": "theta-vs-z wrapped phase diagnostic is legitimate but visually reads as a striped raster; build a poster-specific transverse residual-phase view from source arrays",
            },
            {
                "figure": "figures/experimental/q20_aberration/single_mask/single_z_double_confirmation_minus10.png",
                "decision": "RERENDER_AND_CLAIM_REVIEW",
                "science": "DIAGNOSTIC_ONLY_PENDING_INTERPRETATION",
                "visual": "FAIL_POSTER",
                "note": "six-panel diagnostic is crowded; its own source reports that inverse phase recreation does not reproduce the laboratory angular/fan error structure, so it must not be described as a successful reverse confirmation",
            },
            {
                "figure": "figures/experimental/q20_aberration/phase_error_recreation/phase_error_recreation_signed_xz_yz.png",
                "decision": "APPENDIX_ONLY",
                "science": "PASS_DIAGNOSTIC",
                "visual": "FAIL_MAIN_POSTER",
                "note": "useful longitudinal diagnostic but too dense for a hero poster panel",
            },
            {
                "figure_family": "existing presentation route / B0-V1-V3 / fixed-lab axicon decentre",
                "decision": "PRESERVE_AS_BENCHMARK",
                "science": "PASS_PER_EXISTING_PRESENTATION_AUDIT",
                "visual": "PASS_BENCHMARK",
                "note": "do not replace strong presentation-quality figures merely to show more modules",
            },
        ],
        "self_healing": self_meta,
        "physical_identifiability": gate,
        "quality_rule": "one obvious scientific message per figure; CI success is necessary but not sufficient for poster inclusion",
    }
    path = out / "figure_quality_audit_v8.json"
    path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return path


def contact_sheet(out: Path, self_path: Path, gate_path: Path) -> Path:
    """Only show figures that survived this pass; rejected diagnostics stay out."""
    fig = plt.figure(figsize=(17.8, 9.8), facecolor="#15191d")
    gs = fig.add_gridspec(1, 2, wspace=0.08, left=0.025, right=0.975,
                          top=0.90, bottom=0.05)
    for ax, path, label in zip(
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [self_path, gate_path],
        ["PASS — self-healing", "PASS — identifiability-gated diagnosis"],
    ):
        ax.set_facecolor("#15191d")
        ax.imshow(plt.imread(path))
        ax.set_title(label, color="white", fontsize=13.2, weight="bold", pad=8)
        ax.axis("off")
    fig.suptitle("Poster figure quality gate v8 — only surviving figures shown",
                 color="white", fontsize=20, weight="bold", y=0.975)
    return v4._save(fig, out / "00_quality_gate_v8_contact_sheet.png", dpi=190)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p1, m1 = build_self_healing_quality(OUT)
    p2, m2, csv2 = build_identifiability_gate(OUT)
    audit = write_audit(OUT, m1, m2)
    sheet = contact_sheet(OUT, p1, p2)
    manifest = {
        "outcome": "POSTER-CORE-EVIDENCE-V8",
        "story": "visual/scientific quality gate before any poster rebuild",
        "poster_rebuild_allowed": False,
        "files": [str(sheet), str(p1), str(p2), str(csv2), str(audit)],
        "excluded_from_contact_sheet": [
            "raw v7 recovery figure",
            "q20 annular wrapped-phase diagnostic",
            "q20 six-panel double-confirmation diagnostic",
            "q20 dense XZ/YZ recreation diagnostic",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
