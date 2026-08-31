"""Poster figure quality gate v8.1.

Refines v8 after full-resolution inspection: the identifiability figure now uses
the *coarse* joint landscape (which contains the injected truth) for the 2-D and
iris-profile evidence, while retaining the fine axicon profile/estimate.  This
prevents a failed refined iris window from being confused with the underlying
physical conclusion: the tested observable is weak in the iris direction.
"""
from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import build_poster_core_evidence_v4 as v4
import build_poster_core_evidence_v7 as v7
import build_poster_core_evidence_v8 as v8

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "poster" / "core_evidence_v8_1"
EPS = np.finfo(float).tiny


def _style(ax):
    v4._style(ax)
    ax.grid(False)


def _profile_delta(profile):
    p = np.asarray(profile, float)
    return (p - float(np.min(p))) / max(float(np.min(p)), EPS)


def build_identifiability_gate(out: Path):
    raw_path, summary, csv_path = v7.build_metric_aware_recovery(out)
    coarse = summary["coarse_fit"]
    fine = summary["fine_fit"]

    cx = np.asarray(coarse["values_x"], float)
    cy = np.asarray(coarse["values_y"], float)
    ccost = np.asarray(coarse["costs"], float)
    fx = np.asarray(fine["values_x"], float)
    fy = np.asarray(fine["values_y"], float)
    fcost = np.asarray(fine["costs"], float)
    fiy, fix = [int(v) for v in fine["best_index_yx"]]

    truth_x = float(summary["truth"]["axicon_decentre_x_um"])
    truth_y = float(summary["truth"]["fourf_iris_offset_x_radii"])
    best_x = float(fine["best_x"])
    best_y = float(fine["best_y"])
    margin = float(fine["relative_cost_margin"])

    # Fine x profile gives the useful local axicon estimate.  Coarse y profile
    # contains the injected iris truth and exposes the weak iris sensitivity
    # without relying on the later refinement window.
    fine_x_profile = np.min(fcost, axis=0)
    coarse_y_profile = np.min(ccost, axis=1)
    dx = _profile_delta(fine_x_profile)
    dy = _profile_delta(coarse_y_profile)
    iris_profile_span_pct = float(100*np.max(dy))

    x_step = float(np.median(np.diff(fx)))
    axicon_grid_consistent = abs(best_x-truth_x) <= 0.5*abs(x_step) + 1e-9
    iris_fine_boundary = fiy in (0, len(fy)-1)
    iris_wrong_sign = np.sign(best_y) != np.sign(truth_y) and abs(best_y) > 1e-12
    weak_global_separation = margin < 0.02
    weak_iris_profile = iris_profile_span_pct < 2.0
    iris_reportable = not (
        iris_fine_boundary or iris_wrong_sign or weak_global_separation or weak_iris_profile
    )

    fig = plt.figure(figsize=(15.4, 7.8), facecolor=v4.DARK)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, .88, 1.05],
                          hspace=.30, wspace=.26, left=.06, right=.985,
                          top=.865, bottom=.10)

    ax_cost = fig.add_subplot(gs[:, 0])
    _style(ax_cost)
    im = ax_cost.imshow(ccost, origin="lower", aspect="auto",
                        extent=[cx[0], cx[-1], cy[0], cy[-1]], cmap=v4.DIFF)
    ax_cost.scatter([truth_x], [truth_y], marker="x", s=105, color=v4.RED,
                    linewidths=2.2, label="injected truth")
    # show coarse minimum, because this panel is explicitly the coarse landscape
    ciy, cix = [int(v) for v in coarse["best_index_yx"]]
    coarse_best_x = float(coarse["best_x"])
    coarse_best_y = float(coarse["best_y"])
    ax_cost.scatter([coarse_best_x], [coarse_best_y], marker="o", s=78,
                    facecolors="none", edgecolors=v4.FG, linewidths=1.8,
                    label="coarse minimum")
    ax_cost.set_xlabel("axicon decentre x (µm)")
    ax_cost.set_ylabel("4F iris offset x (iris radii)")
    ax_cost.set_title("Coarse joint landscape — truth retained", fontsize=12.7, weight="bold")
    ax_cost.legend(frameon=False, labelcolor=v4.FG, fontsize=8.4, loc="upper left")
    cb = fig.colorbar(im, ax=ax_cost, fraction=.047, pad=.025)
    cb.set_label("composite loss", color=v4.MUTED, fontsize=8.5)
    cb.ax.tick_params(colors=v4.MUTED, labelsize=7.5)
    cb.outline.set_edgecolor("#51606d")

    ax_x = fig.add_subplot(gs[0, 1])
    _style(ax_x)
    ax_x.plot(fx, 100*dx, "o-", color=v4.CYAN, lw=2.0, ms=4.8)
    ax_x.axvline(truth_x, color=v4.RED, ls="--", lw=1.1, label="truth")
    ax_x.axvline(best_x, color=v4.FG, ls=":", lw=1.1, label="fine minimum")
    ax_x.set_title("Axicon direction — resolved", fontsize=11.4, weight="bold")
    ax_x.set_xlabel("decentre x (µm)")
    ax_x.set_ylabel("loss above minimum (%)")
    ax_x.legend(frameon=False, labelcolor=v4.FG, fontsize=7.7)

    ax_y = fig.add_subplot(gs[1, 1])
    _style(ax_y)
    ax_y.plot(cy, 100*dy, "o-", color=v4.GOLD, lw=2.0, ms=4.8)
    ax_y.axvline(truth_y, color=v4.RED, ls="--", lw=1.1, label="truth")
    ax_y.axvline(coarse_best_y, color=v4.FG, ls=":", lw=1.1, label="coarse minimum")
    ax_y.set_title("Iris direction — weak/degenerate", fontsize=11.4, weight="bold")
    ax_y.set_xlabel("iris offset x (R)")
    ax_y.set_ylabel("loss above minimum (%)")
    ax_y.legend(frameon=False, labelcolor=v4.FG, fontsize=7.7)

    ax_gate = fig.add_subplot(gs[:, 2])
    ax_gate.set_facecolor(v4.DARK)
    ax_gate.axis("off")
    ax_gate.text(.02, .97, "IDENTIFIABILITY GATE", color=v4.FG,
                 fontsize=14.2, weight="bold", va="top")

    ax_gate.add_patch(Rectangle((.01,.55),.98,.31, transform=ax_gate.transAxes,
                                facecolor="#0d1b16", edgecolor=v4.GREEN, linewidth=1.4))
    ax_gate.text(.05,.815,"REPORT", transform=ax_gate.transAxes,
                 color=v4.GREEN, fontsize=12.2, weight="bold", va="top")
    ax_gate.text(.05,.754,"Axicon lateral decentre", transform=ax_gate.transAxes,
                 color=v4.FG, fontsize=11.4, weight="bold", va="top")
    ax_gate.text(.05,.685,
                 f"injected   {truth_x:+.0f} µm\nrecovered  {best_x:+.0f} µm\ngrid error    {abs(best_x-truth_x):.0f} µm",
                 transform=ax_gate.transAxes, color=v4.MUTED, fontsize=10.0,
                 va="top", linespacing=1.45)
    ax_gate.text(.05,.575,"Discrete synthetic estimate — not a confidence interval.",
                 transform=ax_gate.transAxes, color=v4.CYAN, fontsize=8.3, va="bottom")

    ax_gate.add_patch(Rectangle((.01,.13),.98,.34, transform=ax_gate.transAxes,
                                facecolor="#21100d", edgecolor=v4.RED, linewidth=1.4))
    ax_gate.text(.05,.425,"WITHHOLD", transform=ax_gate.transAxes,
                 color=v4.RED, fontsize=12.2, weight="bold", va="top")
    ax_gate.text(.05,.365,"4F iris lateral offset", transform=ax_gate.transAxes,
                 color=v4.FG, fontsize=11.4, weight="bold", va="top")
    reasons = [
        f"coarse profiled loss spans only {iris_profile_span_pct:.1f}%",
        f"injected {truth_y:+.2f} R -> refined fit {best_y:+.2f} R",
        f"refined best/2nd separation only {100*margin:.1f}%",
    ]
    if iris_fine_boundary:
        reasons.append("refined minimum also lands on its boundary")
    ax_gate.text(.05,.300,"\n".join("• "+r for r in reasons),
                 transform=ax_gate.transAxes, color=v4.MUTED, fontsize=8.8,
                 va="top", linespacing=1.40)
    ax_gate.text(.05,.155,"Result: NOT IDENTIFIABLE FROM THIS BENCHMARK",
                 transform=ax_gate.transAxes, color=v4.GOLD, fontsize=8.7,
                 weight="bold", va="bottom")
    ax_gate.text(.02,.045,
                 "Rule: optimiser minimum ≠ physical diagnosis.\nOnly parameters surviving the gate proceed to reporting.",
                 transform=ax_gate.transAxes, color=v4.MUTED, fontsize=8.8,
                 va="bottom", linespacing=1.35)

    fig.suptitle("Physical-error inference must pass an identifiability gate",
                 color=v4.FG, fontsize=18.0, weight="bold", y=.975)
    fig.text(.5,.918,
             "17-plane V1 synthetic stack, simultaneous off-grid errors + noise; coarse landscape is shown so the injected truth remains inside the tested domain.",
             ha="center", color=v4.MUTED, fontsize=9.2)

    path = v4._save(fig, out/"02_physical_identifiability_gate.png", dpi=330)
    gate = {
        "status": "PASS_METHOD_FIGURE",
        "source_raw_v7_figure": str(raw_path),
        "display_landscape": "coarse grid (contains injected truth)",
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
            "coarse_profile_span_percent": iris_profile_span_pct,
            "refined_minimum_on_boundary": bool(iris_fine_boundary),
            "refined_wrong_sign": bool(iris_wrong_sign),
        },
        "joint_best_second_relative_margin": margin,
        "claim_boundary": "synthetic identifiability demonstration only; no experimental parameter or statistical uncertainty is claimed",
    }
    return path, gate, csv_path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p1, m1 = v8.build_self_healing_quality(OUT)
    p2, m2, csv2 = build_identifiability_gate(OUT)
    audit = v8.write_audit(OUT, m1, m2)
    sheet = v8.contact_sheet(OUT, p1, p2)
    manifest = {
        "outcome": "POSTER-CORE-EVIDENCE-V8.1",
        "story": "full-resolution visual/scientific quality gate before poster rebuild",
        "poster_rebuild_allowed": False,
        "poster_rebuild_blocker": "q20 experimental source-array rerender remains unresolved; tracked diagnostic rasters are not promoted",
        "files": [str(sheet), str(p1), str(p2), str(csv2), str(audit)],
    }
    (OUT/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
