"""Final presentation-only cleanup for two figures used in the live deck.

This does not alter the optical model or any numerical field. It only changes
presentation rendering:

1. The V3 ideal-vs-nominal *absolute difference* panel uses a perceptually
   distinct sequential colormap (cividis) rather than the thermal intensity
   palette, so it cannot be mistaken for another beam-intensity map.
2. The axicon-decentre XY panels suppress the hollow circular marker that was
   previously drawn at the imposed axicon position. The marker was annotation,
   not simulated beam structure.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

import build_axicon_presentation_evidence_v3 as axicon
import build_phase2l_nominal_constraints as nominal
import presentation_phase2j_style as style

DIFF_CMAP = "cividis"


def render_nominal_difference(out: Path, grid_n: int) -> Path:
    """Re-render the V3 propagation comparison with a distinct residual palette."""
    ideal_route = nominal.continuous_ideal("V3", grid_n)
    nominal_route = nominal.build_system_route("V3", grid_n=grid_n)
    ideal_xz = nominal._norm(nominal._xz(ideal_route, "presentation-cleanup-ideal-v3"))
    nominal_xz = nominal._norm(nominal._xz(nominal_route, "presentation-cleanup-nominal-v3"))
    difference = np.abs(nominal_xz - ideal_xz)

    z = nominal.Z_VALUES_M * 1e3
    x = nominal.PROP_COORD_M * 1e3
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), facecolor=nominal.BG)
    fig.subplots_adjust(left=.055, right=.985, bottom=.16, top=.78, wspace=.13)
    for axis in axes:
        style.style_ax(axis)

    for axis, values, title in (
        (axes[0], ideal_xz, "Continuous ideal"),
        (axes[1], nominal_xz, "Nominal bench-constrained"),
    ):
        axis.imshow(
            values.T,
            origin="lower",
            extent=[z[0], z[-1], x[0], x[-1]],
            cmap=nominal.CMAP,
            norm=nominal.PNORM,
            interpolation=style.DISPLAY_INTERPOLATION,
            aspect="auto",
        )
        axis.set_title(title, color=nominal.TEXT, fontsize=12.5, weight="bold")
        axis.set_xlabel("z from axicon (mm)")
        axis.axvline(60, color="white", alpha=.25, ls="--", lw=.8)

    axes[0].set_ylabel("x at fixed y=0 (mm)")
    im = axes[2].imshow(
        difference.T,
        origin="lower",
        extent=[z[0], z[-1], x[0], x[-1]],
        cmap=DIFF_CMAP,
        vmin=0,
        vmax=max(float(np.max(difference)), 1e-12),
        interpolation=style.DISPLAY_INTERPOLATION,
        aspect="auto",
    )
    axes[2].set_title("Absolute morphology difference", color=nominal.TEXT, fontsize=12.5, weight="bold")
    axes[2].set_xlabel("z from axicon (mm)")
    cb = fig.colorbar(im, ax=axes[2], pad=.02, shrink=.88)
    cb.ax.tick_params(colors=nominal.MUTED, labelsize=7)
    cb.set_label("|I_nominal − I_ideal|", color=nominal.MUTED, fontsize=8)

    fig.suptitle(
        "V3 propagation: ideal vs nominal experimental constraints",
        color=nominal.TEXT,
        fontsize=17,
        weight="bold",
        y=.95,
    )
    fig.text(
        .5,
        .865,
        "Same physical coordinates and propagation range; main panels peak-normalised for morphology",
        ha="center",
        color=nominal.MUTED,
        fontsize=9.2,
    )

    path = out / "02_V3_propagation_ideal_vs_nominal.png"
    fig.savefig(path, dpi=480, bbox_inches="tight", facecolor=nominal.BG, pad_inches=.06)
    plt.close(fig)
    return path


def render_clean_decentre(out: Path, grid_n: int) -> Path:
    """Render the existing decentre evidence without the annotation circles."""
    original_scatter = Axes.scatter

    def _suppress_scatter(self, *args, **kwargs):
        # build_decentre uses scatter only for the hollow axicon-position marker
        # on the three XY presentation panels. Suppressing it leaves the
        # simulated data and all axes/normalisation untouched.
        return None

    Axes.scatter = _suppress_scatter
    try:
        p2d, _, _ = axicon.build_decentre(out, grid_n)
    finally:
        Axes.scatter = original_scatter
    return p2d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-n", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.grid_n < 1024:
        raise ValueError("presentation cleanup requires grid_n >= 1024")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(render_nominal_difference(args.output_dir, args.grid_n))
    print(render_clean_decentre(args.output_dir, args.grid_n))


if __name__ == "__main__":
    main()
