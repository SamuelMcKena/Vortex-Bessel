"""Visual-first poster figure for the q=20 correction concept.

The poster should explain the method through the data, not through status boxes or
long prose.  The figure therefore shows a short visual chain:

    measured field -> physical-system fit -> residual phase -> correction
                   -> corrected transverse field / propagation -> ideal

The physical-error fit is represented compactly by the four model planes already
implemented in the digital twin (beam, SLM, 4F and axicon).  The transverse and
longitudinal correction comparisons are cropped from the tracked q20 analysis
figures so the poster retains the experimentally anchored/model-predicted data
products rather than inventing substitute artwork.

Poster outputs are 500-dpi PNG and PDF.  A lower-resolution review image and the
source/crop panels are included in the Actions artifact for visual inspection.
"""
from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle
import numpy as np
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "poster" / "q20_inverse_v4"

SINGLE_Z = ROOT / "figures/experimental/q20_aberration/single_mask/single_z_double_confirmation_minus10.png"
XZ_COMPARE = ROOT / "figures/experimental/q20_aberration/validation/realigned_signed_xz_yz_measured_corrected_ideal.png"
PHASE = ROOT / "figures/experimental/q20_aberration/reconstruction/annular_aberration_phase.png"
MEASURED_XZ_ATLAS = ROOT / "figures/experimental/q20_aberration/validation/all_z_full_signed_xz_yz_maps.png"

BG = "#070a0d"
PANEL = "#0b1015"
FG = "#f5f6f7"
MUTED = "#a8b1ba"
CYAN = "#4dd9d5"
GOLD = "#ffbd4a"
BORDER = "#3f4a55"


def _trim_white(im: Image.Image, threshold: int = 16, pad: int = 8) -> Image.Image:
    rgb = im.convert("RGB")
    diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white")).convert("L")
    diff = diff.point(lambda p: 255 if p > threshold else 0)
    box = diff.getbbox()
    if box is None:
        return rgb
    l, t, r, b = box
    return rgb.crop((max(0,l-pad), max(0,t-pad), min(rgb.width,r+pad), min(rgb.height,b+pad)))


def _fcrop(im: Image.Image, box: tuple[float,float,float,float]) -> Image.Image:
    w, h = im.size
    l, t, r, b = box
    return im.crop((round(l*w), round(t*h), round(r*w), round(b*h)))


def _read_sources() -> dict[str, Image.Image]:
    for p in (SINGLE_Z, XZ_COMPARE, PHASE, MEASURED_XZ_ATLAS):
        if not p.exists():
            raise FileNotFoundError(p)
    single = Image.open(SINGLE_Z).convert("RGB")
    xz = Image.open(XZ_COMPARE).convert("RGB")
    phase = _trim_white(Image.open(PHASE).convert("RGB"), pad=6)
    atlas = Image.open(MEASURED_XZ_ATLAS).convert("RGB")

    # make_single_z_double_confirmation.py: 2 x 3 layout.  These crops retain
    # the three top-row transverse panels while removing most of the old title /
    # explanatory text.  We add clean poster labels ourselves below.
    transverse = {
        "xy_measured": _trim_white(_fcrop(single, (0.015, 0.105, 0.325, 0.505)), pad=4),
        "xy_corrected": _trim_white(_fcrop(single, (0.335, 0.105, 0.655, 0.505)), pad=4),
        "xy_ideal": _trim_white(_fcrop(single, (0.665, 0.105, 0.985, 0.505)), pad=4),
    }

    # realigned_signed_xz_yz_measured_corrected_ideal.png is the tracked
    # measured/corrected/ideal longitudinal comparison.  Use its XZ (top) row.
    propagation = {
        "xz_measured": _trim_white(_fcrop(xz, (0.010, 0.055, 0.330, 0.505)), pad=4),
        "xz_corrected": _trim_white(_fcrop(xz, (0.335, 0.055, 0.665, 0.505)), pad=4),
        "xz_ideal": _trim_white(_fcrop(xz, (0.670, 0.055, 0.995, 0.505)), pad=4),
    }

    # A clean full-stack experimental XZ crop used in the existing chat-export
    # workflow; included as a fallback/reference and in the artifact audit.
    measured_stack_xz = _trim_white(_fcrop(atlas, (0.002, 0.015, 0.218, 0.53)), pad=4)

    return {**transverse, **propagation, "phase": phase,
            "measured_stack_xz": measured_stack_xz,
            "single_full": single, "xz_full": xz}


def _panel(ax: plt.Axes, image: Image.Image, title: str, *, label_size=9.2) -> None:
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(BORDER); s.set_linewidth(0.8)
    ax.imshow(np.asarray(image), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, color=FG, fontsize=label_size, weight="bold", pad=5)


def _physical_fit(ax: plt.Axes) -> None:
    """Compact visual bridge from the forward error library into phase retrieval."""
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(BORDER); s.set_linewidth(0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.91, "Fit physical system", color=FG, fontsize=10.3,
            weight="bold", ha="center", va="top")

    names = ("Beam", "SLM", "4F", "Axicon")
    xs = (0.16, 0.39, 0.62, 0.85)
    for x, name in zip(xs, names):
        ax.add_patch(Circle((x, 0.58), 0.075, facecolor="#111a22",
                            edgecolor=CYAN, lw=1.2))
        if name == "Beam":
            for r, a in ((0.047, .9), (0.029, .65), (0.013, .4)):
                ax.add_patch(Circle((x,0.58), r, fill=False, ec=GOLD, lw=1.0, alpha=a))
        elif name == "SLM":
            for d in np.linspace(-0.045,0.045,4):
                ax.plot([x+d,x+d],[0.535,0.625], color=GOLD, lw=.75)
            for d in np.linspace(-0.045,0.045,4):
                ax.plot([x-0.045,x+0.045],[0.58+d,0.58+d], color=GOLD, lw=.75)
        elif name == "4F":
            ax.add_patch(Circle((x,0.58),0.045, fill=False, ec=GOLD, lw=1.1))
            ax.add_patch(Circle((x+0.025,0.58),0.020, fill=False, ec=FG, lw=.8))
        else:
            ax.plot([x-0.05,x,x+0.05],[0.535,0.625,0.535], color=GOLD, lw=1.3)
        ax.text(x, 0.43, name, color=MUTED, fontsize=7.7, ha="center", va="top")

    ax.text(0.5, 0.20,
            r"$\hat{\theta}_{\rm phys}=\arg\min_{\theta}\,\sum_z\mathcal{L}[I_{\rm model}(z;\theta),I_{\rm meas}(z)]$",
            color=FG, fontsize=9.0, ha="center", va="center")
    ax.text(0.5, 0.075, "best-fit forward field", color=CYAN, fontsize=7.7,
            ha="center", va="center")


def _phase_panel(ax: plt.Axes, image: Image.Image) -> None:
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(BORDER); s.set_linewidth(0.8)
    ax.imshow(np.asarray(image), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Residual phase", color=FG, fontsize=10.3, weight="bold", pad=5)
    ax.text(0.5, -0.07, r"$\phi_{\rm corr}=-\psi_{\rm res}$",
            transform=ax.transAxes, color=CYAN, fontsize=9.5, weight="bold",
            ha="center", va="top")


def _arrow(fig: plt.Figure, ax1: plt.Axes, ax2: plt.Axes, yfrac: float = .50) -> None:
    a, b = ax1.get_position(), ax2.get_position()
    y = a.y0 + yfrac*(a.y1-a.y0)
    fig.add_artist(FancyArrowPatch((a.x1+0.004, y), (b.x0-0.004, y),
                                   transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=13, lw=1.25, color=CYAN))


def _save_debug_sources(out: Path, src: dict[str, Image.Image]) -> list[str]:
    d = out / "review_crops"
    d.mkdir(parents=True, exist_ok=True)
    keep = ("xy_measured","xy_corrected","xy_ideal","xz_measured","xz_corrected","xz_ideal",
            "phase","measured_stack_xz")
    paths = []
    for name in keep:
        p = d / f"{name}.png"
        src[name].save(p)
        paths.append(str(p))
    return paths


def build(out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    src = _read_sources()
    review_crops = _save_debug_sources(out, src)

    # Main composition: data dominates the page.  The two small centre panels are
    # the only algorithmic bridge; there are no status/disclaimer cards.
    fig = plt.figure(figsize=(16.6, 7.5), facecolor=BG)
    outer = fig.add_gridspec(2, 5,
        width_ratios=[1.22, 0.80, 0.82, 1.15, 1.15],
        height_ratios=[1.0, 1.0],
        left=0.035, right=0.985, bottom=0.085, top=0.80,
        wspace=0.14, hspace=0.25)

    # Measurement column.
    ax_xy_m = fig.add_subplot(outer[0,0]); _panel(ax_xy_m, src["xy_measured"], "Measured transverse field")
    ax_xz_m = fig.add_subplot(outer[1,0]); _panel(ax_xz_m, src["xz_measured"], "Measured propagation")

    # Compact algorithm bridge spans both rows.
    mid = outer[:,1:3].subgridspec(2,1, height_ratios=[0.78,1.22], hspace=0.22)
    ax_fit = fig.add_subplot(mid[0,0]); _physical_fit(ax_fit)
    ax_phase = fig.add_subplot(mid[1,0]); _phase_panel(ax_phase, src["phase"])

    # Corrected / ideal results: two rows, two columns.
    ax_xy_c = fig.add_subplot(outer[0,3]); _panel(ax_xy_c, src["xy_corrected"], "Corrected")
    ax_xy_i = fig.add_subplot(outer[0,4]); _panel(ax_xy_i, src["xy_ideal"], "Ideal")
    ax_xz_c = fig.add_subplot(outer[1,3]); _panel(ax_xz_c, src["xz_corrected"], "Corrected propagation")
    ax_xz_i = fig.add_subplot(outer[1,4]); _panel(ax_xz_i, src["xz_ideal"], "Ideal propagation")

    _arrow(fig, ax_xy_m, ax_fit, .52)
    _arrow(fig, ax_phase, ax_xy_c, .56)

    # The correction relation is shown once between residual phase and results.
    p1, p2 = ax_fit.get_position(), ax_phase.get_position()
    fig.add_artist(FancyArrowPatch(((p1.x0+p1.x1)/2, p1.y0-0.008),
                                   ((p2.x0+p2.x1)/2, p2.y1+0.008),
                                   transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=13, lw=1.15, color=CYAN))

    # Group labels make the two before/after rows readable from several metres.
    fig.text(0.035, 0.945, "MEASUREMENT-DRIVEN CORRECTION", color=CYAN,
             fontsize=9.0, weight="bold", ha="left")
    fig.text(0.035, 0.895, "Fit the system. Retrieve the residual. Correct the beam.",
             color=FG, fontsize=21.5, weight="bold", ha="left")
    fig.text(0.035, 0.845,
             "Measured q=20 propagation → physical forward-model fit → residual phase → corrected transverse and longitudinal field",
             color=MUTED, fontsize=9.2, ha="left")

    fig.text(0.077, 0.815, "MEASURED", color=MUTED, fontsize=8.2, weight="bold", ha="center")
    fig.text(0.355, 0.815, "INVERSE MODEL", color=MUTED, fontsize=8.2, weight="bold", ha="center")
    fig.text(0.795, 0.815, "CORRECTION RESULT", color=MUTED, fontsize=8.2, weight="bold", ha="center")

    # Small loop-back arrow: the method is iterative, without turning it into a
    # prose-heavy fifth panel.
    fig.add_artist(FancyArrowPatch((0.94, 0.055), (0.08, 0.055),
                                   transform=fig.transFigure,
                                   connectionstyle="arc3,rad=-0.06", arrowstyle="-|>",
                                   mutation_scale=12, lw=1.0, color="#66727d"))
    fig.text(0.51, 0.027, "repeat measurement", color="#7f8a94",
             fontsize=7.7, ha="center")

    png = out / "poster_q20_correction_visual.png"
    pdf = out / "poster_q20_correction_visual.pdf"
    fig.savefig(png, dpi=500, bbox_inches="tight", facecolor=BG)
    fig.savefig(pdf, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    with Image.open(png) as im:
        preview = im.convert("RGB")
        preview.thumbnail((2500, 1400), Image.Resampling.LANCZOS)
        preview_path = out / "poster_q20_correction_visual.preview.jpg"
        preview.save(preview_path, quality=91, subsampling=0)
        final_size = list(im.size)
        dpi = list(im.info.get("dpi", (0,0)))

    manifest = {
        "outcome": "POSTER-Q20-INVERSE-V4",
        "visual_story": [
            "measured transverse field and XZ propagation",
            "physical forward-model fit across beam/SLM/4F/axicon",
            "retrieved residual phase and conjugate correction",
            "corrected transverse field compared with ideal",
            "corrected XZ propagation compared with ideal",
            "repeat measurement",
        ],
        "final_assets": {"png_500dpi": str(png), "pdf": str(pdf),
                         "preview_review_only": str(preview_path)},
        "final_png_pixel_size": final_size,
        "final_png_dpi": dpi,
        "review_crops": review_crops,
        "source_files": [str(p.relative_to(ROOT)) for p in (SINGLE_Z, XZ_COMPARE, PHASE, MEASURED_XZ_ATLAS)],
        # Scientific provenance remains machine-readable but is intentionally not
        # rendered as a wall of status language on the poster.
        "scope": {
            "measured_input": "experimental q20 z-stack",
            "correction_comparison": "model-predicted correction from current q20 retrieval figures",
            "programmed_qtheta_removed_from_residual": True,
        },
    }
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    build()
