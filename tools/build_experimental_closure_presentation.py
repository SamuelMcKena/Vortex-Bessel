"""Render the final slide-ready experimental model–measurement closure schematic.

This is the canonical presentation version of the closure loop.  It deliberately
uses generic experimental language rather than internal development-phase names.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath

BG = "#05080b"
PANEL = "#0b1015"
TEXT = "#f2f4f5"
MUTED = "#b7bec5"
RED = "#ff4b42"
CYAN = "#35d9c0"
WHITE = "#d7dde2"


def render(output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(16, 5.2), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    ax.text(8, 4.78, "Experimental model–measurement closure",
            ha="center", va="center", color=TEXT, fontsize=24, weight="bold")
    ax.text(8, 4.40, "Experimental measurements enter the loop at this stage",
            ha="center", va="center", color=MUTED, fontsize=13)

    items = [
        (0.55, 2.02, 3.15, 1.62, RED,  "simulate",       "predicted XY / XZ\nsignatures"),
        (4.38, 2.02, 3.15, 1.62, CYAN, "measure",        "calibrated camera\nz-stack"),
        (8.21, 2.02, 3.45, 1.62, RED,  "compare / infer", "fit residuals and\nphysical parameters"),
        (12.20, 2.02, 3.25, 1.62, CYAN,"update",         "calibration +\nSLM correction"),
    ]

    for x, y, w, h, colour, title, body in items:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.016,rounding_size=0.13",
            facecolor=PANEL, edgecolor=colour, linewidth=1.8,
        ))
        ax.text(x + w/2, y + h*0.66, title, ha="center", va="center",
                color=TEXT, fontsize=18, weight="bold")
        ax.text(x + w/2, y + h*0.39, body, ha="center", va="center",
                color=MUTED, fontsize=11.5)

    for i in range(3):
        x1 = items[i][0] + items[i][2] + 0.04
        y1 = items[i][1] + items[i][3] / 2
        x2 = items[i+1][0] - 0.04
        y2 = items[i+1][1] + items[i+1][3] / 2
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
            linewidth=1.8, color=RED,
        ))

    # Smooth return path underneath the forward chain; this avoids text/arrow collisions.
    verts = [(13.8, 1.86), (11.4, 0.78), (4.65, 0.78), (2.10, 1.86)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor="none",
                           edgecolor=WHITE, linewidth=1.6))
    ax.add_patch(FancyArrowPatch(
        (2.26, 1.88), (2.05, 1.88), arrowstyle="-|>", mutation_scale=15,
        linewidth=1.6, color=WHITE,
    ))

    ax.text(8, 0.58,
            "repeat until model and measured beam agree within calibrated uncertainty",
            ha="center", va="center", color=TEXT, fontsize=13)
    ax.text(8, 0.18, "simulate → measure → infer → update → validate",
            ha="center", va="center", color=MUTED, fontsize=11.5)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=BG, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return output


if __name__ == "__main__":
    render(Path("figures/presentation/06_simulation_experiment_closure_phase2j.png"))
