"""Presentation refinements for the axicon-error slides.

Produces three presentation assets:
  * clean lateral-decentre schematic with no pseudo ray-trace glow;
  * clean rounded-apex schematic with a restrained interference annotation;
  * longitudinal on-axis rounded/sharp comparison for the mitigation slide.

The longitudinal plot reuses the validated localized rounded-cap stress-test
model from build_axicon_localized_round_tip_threefig.py and the selected
600 um annular clear radius from GitHub Actions run 32256466021.  It is a
presentation mechanism study, not a metrology-fitted model of the lab axicon.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Polygon, Rectangle
import numpy as np

import build_axicon_localized_round_tip_threefig as severe
import build_axicon_presentation_evidence_v3 as base
import build_phase2j_presentation_suite as suite
import presentation_phase2j_style as style

EPS = np.finfo(float).tiny
SELECTED_CLEAR_RADIUS_UM = 600

BG = "#05080b"
FG = "#f2f2f2"
MUTED = "#aeb8c2"
CYAN = "#64d9ff"
TEAL = "#39d6ad"
RED = "#ff4d57"
ORANGE = "#ff9d00"
GRID = "#34414d"


def _clean_canvas(figsize=(10.0, 6.2)):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(-3.1, 3.1)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def build_decentre_schematic(path: Path) -> None:
    """Clean geometry-only schematic of axicon lateral decentre."""
    fig, ax = _clean_canvas()

    # Fixed laboratory optical axis and input beam.
    ax.plot([0.35, 9.65], [0, 0], ls=(0, (5, 5)), lw=1.2, color=MUTED, alpha=0.9)
    ax.add_patch(Rectangle((0.45, -0.34), 3.0, 0.68, facecolor=RED, edgecolor="none", alpha=0.14))
    ax.annotate("", xy=(3.45, 0), xytext=(0.65, 0), arrowprops=dict(arrowstyle="->", lw=2.0, color=RED))
    ax.text(0.62, 0.56, "input beam", color=FG, fontsize=15)
    ax.text(8.45, 0.20, "laboratory optical axis", color=MUTED, fontsize=12)

    # Axicon is deliberately displaced relative to the fixed beam/axis.
    dy = 0.82
    axicon = Polygon([[3.85, -2.05 + dy], [3.85, 2.05 + dy], [5.05, dy]], closed=True,
                     facecolor="#1a3444", edgecolor=CYAN, lw=2.0)
    ax.add_patch(axicon)
    ax.plot([3.55, 5.45], [dy, dy], ls=(0, (3, 4)), lw=1.0, color=CYAN, alpha=0.75)
    ax.text(3.72, 2.55 + dy*0.2, "axicon", color=FG, fontsize=15)

    # Explicitly show the imposed offset rather than invented asymmetric rays.
    ax.annotate("", xy=(3.35, dy), xytext=(3.35, 0),
                arrowprops=dict(arrowstyle="<->", lw=1.6, color=FG))
    ax.text(2.02, 0.36, r"imposed lateral decentre  $\Delta x$", color=FG, fontsize=13)

    # Predicted output centroid is indicated as a translated beam marker only.
    cx, cy = 7.35, dy
    for r, alpha in [(0.58, 0.18), (0.40, 0.28), (0.23, 0.5)]:
        ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor=ORANGE, lw=1.2, alpha=alpha))
    ax.add_patch(Circle((cx, cy), 0.085, facecolor="#ffe47a", edgecolor="none"))
    ax.plot([5.35, cx], [dy, dy], lw=1.4, color=ORANGE, alpha=0.85)
    ax.text(6.26, 1.55, "predicted beam\ncentroid shift", color=FG, fontsize=14, ha="center")
    ax.annotate("", xy=(cx, 0.02), xytext=(cx, dy-0.02),
                arrowprops=dict(arrowstyle="<->", lw=1.3, color=ORANGE))

    ax.text(5.0, -2.72,
            "Only the axicon position is changed; propagation is evaluated in fixed laboratory coordinates.",
            color=MUTED, fontsize=11.5, ha="center")
    fig.savefig(path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def build_rounded_apex_schematic(path: Path) -> None:
    """Clean schematic of a localized rounded axicon apex and its signature."""
    fig, ax = _clean_canvas()

    ax.plot([0.35, 9.65], [0, 0], ls=(0, (5, 5)), lw=1.2, color=MUTED, alpha=0.9)
    ax.add_patch(Rectangle((0.45, -0.38), 3.0, 0.76, facecolor=RED, edgecolor="none", alpha=0.14))
    ax.annotate("", xy=(3.45, 0), xytext=(0.65, 0), arrowprops=dict(arrowstyle="->", lw=2.0, color=RED))
    ax.text(0.62, 0.60, "input beam", color=FG, fontsize=15)

    # Simplified axicon outline with a visibly rounded local apex.
    ax.plot([3.7, 3.7], [-2.05, 2.05], color=CYAN, lw=2.0)
    ax.plot([3.7, 4.77], [2.05, 0.46], color=CYAN, lw=2.0)
    ax.plot([3.7, 4.77], [-2.05, -0.46], color=CYAN, lw=2.0)
    ax.add_patch(Arc((4.78, 0), 0.92, 0.92, theta1=-90, theta2=90, color=CYAN, lw=2.0))
    ax.fill([3.7, 3.7, 4.77, 4.77], [-2.05, 2.05, 0.46, -0.46], color="#1a3444", alpha=0.75)
    ax.text(3.35, 2.45, "rounded-tip axicon", color=FG, fontsize=15)

    # Local defect footprint callout, not a glowing pseudo-ray picture.
    ax.add_patch(Circle((4.78, 0), 0.58, fill=False, edgecolor=TEAL, lw=1.5, ls="--"))
    ax.annotate("localized apex defect", xy=(4.82, -0.52), xytext=(5.45, -1.60), color=TEAL, fontsize=13,
                arrowprops=dict(arrowstyle="->", lw=1.3, color=TEAL))

    # Two physically motivated contributions are labelled schematically.
    ax.plot([5.1, 8.95], [0.80, 0.80], color=ORANGE, lw=1.3, alpha=0.9)
    ax.text(6.05, 1.06, "conical contribution", color=ORANGE, fontsize=12)
    ax.plot([5.1, 8.95], [0.28, 0.28], color=RED, lw=1.3, alpha=0.9)
    ax.text(6.05, 0.48, "rounded-apex contribution", color=RED, fontsize=12)

    # Minimal axial modulation trace to show the observable consequence.
    xs = np.linspace(5.15, 9.15, 300)
    env = np.exp(-0.12*(xs-5.15))
    ys = -0.62 + 0.20*env*np.sin(10.5*(xs-5.15))
    ax.plot(xs, ys, color=FG, lw=1.5)
    ax.plot([5.15, 9.15], [-0.62, -0.62], color=GRID, lw=0.8)
    ax.text(6.05, -1.04, "interference → longitudinal intensity modulation", color=FG, fontsize=12.5)

    ax.text(5.0, -2.72,
            "The rounded central region adds a second refracted contribution that interferes with the Bessel-forming field.",
            color=MUTED, fontsize=11.5, ha="center")
    fig.savefig(path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def build_longitudinal_mitigation(path: Path) -> None:
    """Single slide-ready matched sharp/rounded longitudinal comparison."""
    wavelength, grid, b0, sharp_t, round_t = severe._fields(2048)
    hollow = severe._hollow_l0(b0, grid, SELECTED_CLEAR_RADIUS_UM*1e-6)
    coord = np.asarray(suite.TIP_COORD_M, float)
    z = np.asarray(suite.Z_VALUES_M, float)

    cases = []
    for label, incident, colour in (
        ("ordinary B0", b0, ORANGE),
        (rf"annular $\ell=0$ ({SELECTED_CLEAR_RADIUS_UM} µm clear radius)", hollow, TEAL),
    ):
        xs, _ = base._xz_from_post_axicon(incident*sharp_t, grid, wavelength, coord, f"slide-refine-{label}-sharp")
        xr, _ = base._xz_from_post_axicon(incident*round_t, grid, wavelength, coord, f"slide-refine-{label}-round")
        sharp_axis = severe._onaxis(xs, coord)
        round_axis = severe._onaxis(xr, coord)
        ratio = round_axis / np.maximum(sharp_axis, EPS)
        cases.append((label, incident, ratio, colour))

    b0_tip = 100*severe._fraction_inside(b0, grid)
    h_tip = 100*severe._fraction_inside(hollow, grid)

    fig, ax = plt.subplots(figsize=(9.8, 5.3), constrained_layout=True)
    style.style_fig(fig)
    base._style_line_axis(ax)
    for label, _incident, ratio, colour in cases:
        ax.plot(z*1e3, ratio, lw=2.2, color=colour, label=label)
    ax.axhline(1.0, color=style.MUTED, lw=1.0, ls="--", alpha=0.75)
    ax.set_xlim(float(np.min(z))*1e3, float(np.max(z))*1e3)
    ax.set_xlabel("z from axicon (mm)")
    ax.set_ylabel("on-axis intensity  $I_{rounded}(0,z)/I_{sharp}(0,z)$")
    ax.set_title("Longitudinal on-axis response to the same localized rounded-apex defect",
                 color=style.TEXT, fontsize=13)

    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for t in leg.get_texts():
        t.set_color(style.TEXT)

    ax.text(0.025, 0.95,
            f"APEX ILLUMINATION\n{b0_tip:.1f}%  →  {h_tip:.3g}%",
            transform=ax.transAxes, va="top", ha="left", color=style.TEXT, fontsize=12,
            bbox=dict(facecolor=style.FIG_BG, edgecolor=TEAL, linewidth=1.5, alpha=0.94, pad=7))
    ax.text(0.025, 0.73,
            "ordinary B0: strong rounded/sharp modulation\nannular input: rounded response ≈ matched sharp reference",
            transform=ax.transAxes, va="top", ha="left", color=style.MUTED, fontsize=9.5)

    fig.savefig(path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/figures/presentation_refinements"))
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    build_decentre_schematic(a.output_dir / "schematic_axicon_lateral_decentre_clean.png")
    build_rounded_apex_schematic(a.output_dir / "schematic_axicon_rounded_apex_clean.png")
    build_longitudinal_mitigation(a.output_dir / "09c_severe_tip_avoidance_1D_longitudinal.png")
    print("WROTE", a.output_dir)


if __name__ == "__main__":
    main()
