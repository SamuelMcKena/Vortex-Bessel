"""Render a slide-ready rounded-apex avoidance schematic.

This is an explanatory presentation schematic, not a quantitative ray trace.
The axicon tip is drawn as an integrated rounded geometry (not a circular insert):
ordinary B0 illumination overlaps the rounded apex and admits an additional
central contribution, while a hollow l=0 input clears the apex and preserves
the conical rays needed to reconstruct a bright-centred Bessel field.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, FancyArrowPatch
from matplotlib import patheffects as pe

BG = "#020407"
TEXT = "#eef3f6"
MUTED = "#b8c0c7"
BLUE = "#55b9ff"
CYAN = "#33e0d0"
RED = "#ff3f4a"
ORANGE = "#ff9d00"
GOLD = "#ffd95c"


def _hexrgb(value: str):
    value = value.lstrip("#")
    return tuple(int(value[i:i+2], 16)/255 for i in (0, 2, 4))


def render(output: Path):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=180)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    def glow_line(x, y, color, lw=2.0, glow=7, alpha=1.0, z=5, ls="-"):
        for width, a in ((glow, .05), (glow*.65, .09), (glow*.35, .16)):
            ax.plot(x, y, color=color, lw=lw+width, alpha=a,
                    solid_capstyle="round", zorder=z-1, ls=ls)
        ax.plot(x, y, color=color, lw=lw, alpha=alpha,
                solid_capstyle="round", zorder=z, ls=ls)

    def glow_text(x, y, text, color=TEXT, fontsize=13,
                  ha="center", va="center", weight=None):
        artist = ax.text(x, y, text, color=color, fontsize=fontsize,
                         ha=ha, va=va, weight=weight, zorder=20)
        artist.set_path_effects([
            pe.withStroke(linewidth=4, foreground=BG, alpha=.75)
        ])
        return artist

    def rounded_axicon(cx, cy, h=3.15, body=1.55, nose=.33):
        """Draw a cone cross-section with a genuinely rounded integrated apex."""
        x_left = cx - body/2
        x_neck = cx + body/2 - .15
        y_top = cy + h/2
        y_bottom = cy - h/2
        verts = [
            (x_left, y_top),
            (x_neck-.12, cy+.23),
            (x_neck+.02, cy+.17),
            (x_neck+nose, cy+.06),
            (x_neck+nose, cy),
            (x_neck+nose, cy-.06),
            (x_neck+.02, cy-.17),
            (x_neck-.12, cy-.23),
            (x_left, y_bottom),
            (x_left, y_top),
        ]
        codes = [
            MplPath.MOVETO,
            MplPath.LINETO,
            MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.LINETO,
            MplPath.CLOSEPOLY,
        ]
        patch = PathPatch(MplPath(verts, codes), facecolor="#173a5a",
                          edgecolor=BLUE, lw=1.7, alpha=.93, zorder=8)
        patch.set_path_effects([
            pe.withStroke(linewidth=10, foreground=BLUE, alpha=.04),
            pe.withStroke(linewidth=6, foreground=BLUE, alpha=.08),
            pe.withStroke(linewidth=3.5, foreground=BLUE, alpha=.14),
        ])
        ax.add_patch(patch)
        glow_line([x_left, x_left], [y_bottom, y_top], BLUE,
                  lw=1.6, glow=5, z=9)
        return x_neck + nose

    def gaussian_beam(x0, x1, y0, sigma, color_rgb, amp=.75, cutout=0):
        nx, ny = 900, 240
        ys = np.linspace(y0-1.2, y0+1.2, ny)
        X, Y = np.meshgrid(np.linspace(x0, x1, nx), ys)
        intensity = np.exp(-.5*((Y-y0)/sigma)**2)
        if cutout > 0:
            intensity *= 1 - np.exp(-.5*((Y-y0)/cutout)**2)
        rgba = np.zeros((ny, nx, 4))
        rgba[..., 0] = color_rgb[0]
        rgba[..., 1] = color_rgb[1]
        rgba[..., 2] = color_rgb[2]
        rgba[..., 3] = amp * intensity
        ax.imshow(rgba, extent=[x0, x1, ys[0], ys[-1]],
                  origin="lower", aspect="auto", zorder=2)

    def arrow(x0, y0, x1, y1, color, lw=1.6, ms=16, ls="-"):
        artist = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=ms, color=color, lw=lw,
                                 linestyle=ls, zorder=18)
        artist.set_path_effects([
            pe.withStroke(linewidth=6, foreground=color, alpha=.08)
        ])
        ax.add_patch(artist)

    def ordinary_panel(xc):
        yc = 4.25
        glow_text(xc, 7.85, "ordinary B0 illumination", fontsize=19, weight="bold")
        glow_text(xc, 7.46, "rounded apex is illuminated", color=RED, fontsize=12)
        tip = rounded_axicon(xc-.35, yc)

        gaussian_beam(xc-3.6, xc-1.08, yc, .62, _hexrgb(RED), amp=.45)
        arrow(xc-3.2, yc, xc-1.25, yc, RED, 1.5, 15)
        glow_text(xc-2.35, 5.25, "Gaussian / B0 input", fontsize=11.5)

        glow_line([tip, xc+2.55], [yc, yc+1.15], RED, 1.4, 4, z=9)
        glow_line([tip, xc+2.55], [yc, yc-1.15], RED, 1.4, 4, z=9)
        glow_line([tip, xc+2.35], [yc, yc], ORANGE, 1.5, 4, z=9, ls="--")
        glow_text(xc+1.15, 4.65, "apex contribution", color=ORANGE, fontsize=10.5)

        xs = np.linspace(tip+.18, xc+2.15, 8)
        amps = [.42, .52, .68, .56, .78, .50, .70, .42]
        for xpos, amplitude in zip(xs, amps):
            yy = np.linspace(yc-.45, yc+.45, 150)
            xx = np.linspace(xpos-.12, xpos+.12, 40)
            X, Y = np.meshgrid(xx, yy)
            intensity = np.exp(-((X-xpos)/.06)**2
                               -((Y-yc)/(.10+.05*amplitude))**2) * amplitude
            rgba = np.zeros((*intensity.shape, 4))
            rgba[..., :3] = _hexrgb(RED)
            rgba[..., 3] = .62 * intensity
            ax.imshow(rgba, extent=[xpos-.12, xpos+.12, yc-.45, yc+.45],
                      origin="lower", aspect="auto", zorder=4)

        glow_text(xc+1.1, 6.35,
                  "rounded-apex contribution\ninterferes with the conical field",
                  fontsize=12.2)
        glow_text(xc, 1.00, "longitudinal modulation",
                  color=RED, fontsize=13, weight="bold")
        arrow(xc-.15, 1.65, xc-.15, 3.62, RED, 1.2, 12)

    def annular_panel(xc):
        yc = 4.25
        glow_text(xc, 7.85, "annular ℓ = 0 illumination", fontsize=19, weight="bold")
        glow_text(xc, 7.46, "dark centre clears the rounded apex", color=CYAN, fontsize=12)
        tip = rounded_axicon(xc-.35, yc)

        gaussian_beam(xc-3.6, xc-1.08, yc, .70, _hexrgb(CYAN), amp=.40, cutout=.33)
        arrow(xc-3.2, yc+.57, xc-1.25, yc+.57, CYAN, 1.4, 13)
        arrow(xc-3.2, yc-.57, xc-1.25, yc-.57, CYAN, 1.4, 13)
        glow_text(xc-2.38, 5.38,
                  "hollow / annular input\n(no helical phase)", fontsize=11.2)

        glow_line([tip, xc+2.55], [yc, yc+1.05], CYAN, 1.4, 4, z=9)
        glow_line([tip, xc+2.55], [yc, yc-1.05], CYAN, 1.4, 4, z=9)
        glow_line([tip+.35, xc+2.28], [yc, yc], GOLD, 1.9, 6, z=10)
        for offset, alpha in ((.28, .75), (-.28, .75), (.48, .42), (-.48, .42)):
            glow_line([tip+.45, xc+2.25], [yc+offset, yc+offset],
                      CYAN, .9, 2.5, alpha=alpha, z=7)

        glow_text(xc+1.12, 6.35,
                  "conical field reconstructs a\nbright-centred Bessel beam",
                  fontsize=12.2)
        glow_text(xc, 1.00, "reduced apex sensitivity",
                  color=CYAN, fontsize=13, weight="bold")
        arrow(xc-.05, 1.65, xc-.05, 3.62, CYAN, 1.2, 12)

    ordinary_panel(4.0)
    annular_panel(12.0)
    glow_line([8, 8], [.75, 8.05], "#29323a", .8, 0, z=1)
    glow_text(
        8, .35,
        "same rounded-tip axicon • suppress illumination of the imperfect apex while preserving Bessel-forming conical rays",
        color=MUTED, fontsize=11.2,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=BG, bbox_inches="tight", pad_inches=.03)
    plt.close(fig)


if __name__ == "__main__":
    render(Path("figures/presentation/schematic_axicon_tip_avoidance_clean.png"))
