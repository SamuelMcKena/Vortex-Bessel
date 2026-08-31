"""Build the poster-ready q=20 inverse/correction story without overstating validation.

This figure is deliberately different from the old giant q20 atlases.  It uses
tracked experimental figures for what has actually been measured/inferred, then
switches to a vector workflow for the hardware-correction and validation stages
that have not yet been completed experimentally.

Scientific scope:
- measured longitudinal evolution is experimental camera data;
- the displayed residual phase is the current non-axisymmetric diagnostic;
- the programmed q*theta phase is not treated as an aberration;
- no simulated/model-corrected beam is labelled as an experimental result;
- a post-SLM correction claim is blocked until a new identically sampled camera
  stack has actually been acquired.

Final export: 500-dpi PNG plus PDF (vector text/schematic; embedded source images
remain at their native tracked resolution).
"""
from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle
import numpy as np
from PIL import Image, ImageChops, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "poster" / "q20_inverse_v3"

MEASURED_SOURCE = ROOT / "figures/experimental/q20_aberration/validation/all_z_full_signed_xz_yz_maps.png"
PHASE_SOURCE = ROOT / "figures/experimental/q20_aberration/reconstruction/annular_aberration_phase.png"

BG = "#070a0d"
CARD = "#0d1218"
CARD2 = "#111820"
FG = "#f4f6f7"
MUTED = "#aab4be"
CYAN = "#4dd9d5"
AMBER = "#ffb54a"
RED = "#ff6b6b"
GREEN = "#61d095"
BORDER = "#3b4855"


def _trim_white(im: Image.Image, threshold: int = 18, pad: int = 18) -> Image.Image:
    rgb = im.convert("RGB")
    bg = Image.new("RGB", rgb.size, "white")
    diff = ImageChops.difference(rgb, bg).convert("L")
    diff = diff.point(lambda p: 255 if p > threshold else 0)
    box = diff.getbbox()
    if box is None:
        return rgb
    l, t, r, b = box
    return rgb.crop((max(0, l-pad), max(0, t-pad), min(rgb.width, r+pad), min(rgb.height, b+pad)))


def _fraction_crop(im: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    w, h = im.size
    l, t, r, b = box
    return im.crop((int(l*w), int(t*h), int(r*w), int(b*h)))


def _source_images() -> tuple[Image.Image, Image.Image, dict]:
    if not MEASURED_SOURCE.exists():
        raise FileNotFoundError(MEASURED_SOURCE)
    if not PHASE_SOURCE.exists():
        raise FileNotFoundError(PHASE_SOURCE)

    measured_full = Image.open(MEASURED_SOURCE).convert("RGB")
    # This exact crop is already used by export_q20_figures_for_chat.yml and
    # isolates the complete measured XZ stack from the signed XZ/YZ atlas.
    measured_xz = _fraction_crop(measured_full, (0.002, 0.015, 0.218, 0.53))
    measured_xz = _trim_white(measured_xz, pad=12)

    phase = _trim_white(Image.open(PHASE_SOURCE).convert("RGB"), pad=14)
    meta = {
        "measured_source": str(MEASURED_SOURCE.relative_to(ROOT)),
        "measured_source_pixel_size": list(measured_full.size),
        "measured_crop_fraction": [0.002, 0.015, 0.218, 0.53],
        "measured_crop_pixel_size": list(measured_xz.size),
        "phase_source": str(PHASE_SOURCE.relative_to(ROOT)),
        "phase_source_pixel_size": list(Image.open(PHASE_SOURCE).size),
    }
    return measured_xz, phase, meta


def _card(ax: plt.Axes, title: str, kicker: str, index: int, *, status: str | None = None,
          status_color: str = CYAN) -> None:
    ax.set_facecolor(CARD)
    for s in ax.spines.values():
        s.set_color(BORDER)
        s.set_linewidth(1.0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.055, 0.94, f"{index}", color=BG, ha="center", va="center",
            fontsize=10.5, weight="bold",
            bbox=dict(boxstyle="circle,pad=0.35", fc=CYAN, ec="none"))
    ax.text(0.12, 0.955, title, transform=ax.transAxes, color=FG,
            fontsize=12.6, weight="bold", ha="left", va="top")
    ax.text(0.12, 0.905, kicker, transform=ax.transAxes, color=MUTED,
            fontsize=8.0, ha="left", va="top")
    if status:
        ax.text(0.945, 0.945, status, color=status_color, fontsize=7.3, weight="bold",
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.27", fc="#0a0e13", ec=status_color,
                          lw=0.8, alpha=0.98))


def _place_image(ax: plt.Axes, im: Image.Image, box=(0.055, 0.12, 0.89, 0.70),
                 background="#ffffff") -> None:
    x, y, w, h = box
    ax.add_patch(FancyBboxPatch((x, y), w, h, transform=ax.transAxes,
                                boxstyle="round,pad=0.008", facecolor=background,
                                edgecolor="#59636d", linewidth=0.7, zorder=1))
    arr = np.asarray(im)
    ih, iw = arr.shape[:2]
    target_ratio = w / h
    image_ratio = iw / ih
    if image_ratio >= target_ratio:
        draw_w = w
        draw_h = w / image_ratio
        x0 = x
        y0 = y + (h-draw_h)/2
    else:
        draw_h = h
        draw_w = h * image_ratio
        x0 = x + (w-draw_w)/2
        y0 = y
    inset = ax.inset_axes([x0+0.01, y0+0.01, max(draw_w-0.02, 0.02), max(draw_h-0.02, 0.02)])
    inset.imshow(arr, interpolation="nearest")
    inset.axis("off")


def _gate_row(ax: plt.Axes, y: float, text: str, *, done: bool = False) -> None:
    c = GREEN if done else AMBER
    ax.add_patch(Circle((0.12, y), 0.018, transform=ax.transAxes,
                        facecolor=c if done else "none", edgecolor=c, lw=1.5))
    if done:
        ax.text(0.12, y-0.002, "✓", color=BG, fontsize=8.5, weight="bold",
                ha="center", va="center")
    ax.text(0.165, y, text, transform=ax.transAxes, color=FG if done else "#d8dde1",
            fontsize=8.5, ha="left", va="center")


def _correction_schematic(ax: plt.Axes) -> None:
    # Equation is the scientifically meaningful centrepiece; the graphics are
    # explicitly schematic rather than pretending to be a retrieved phase map.
    ax.text(0.5, 0.76, r"$\phi_{\mathrm{corr}}(\rho,\theta)=-\psi_{\mathrm{res}}(\rho,\theta)$",
            transform=ax.transAxes, color=FG, fontsize=15.2, weight="bold",
            ha="center", va="center")
    ax.text(0.5, 0.695, "programmed qθ is excluded from the correction",
            transform=ax.transAxes, color=CYAN, fontsize=8.2, ha="center")

    # SLM icon.
    x0, y0, w, h = 0.30, 0.48, 0.40, 0.13
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, transform=ax.transAxes,
                                boxstyle="round,pad=0.012", facecolor="#151d26",
                                edgecolor="#708090", lw=1.0))
    for i in range(8):
        ax.plot([x0 + w*i/8, x0 + w*i/8], [y0, y0+h], transform=ax.transAxes,
                color="#293440", lw=0.45)
    for j in range(4):
        ax.plot([x0, x0+w], [y0 + h*j/4, y0 + h*j/4], transform=ax.transAxes,
                color="#293440", lw=0.45)
    ax.text(0.5, 0.545, "SLM2", transform=ax.transAxes, color=FG,
            fontsize=11, weight="bold", ha="center", va="center")

    ax.text(0.08, 0.405, "Hardware-map gates", transform=ax.transAxes,
            color=MUTED, fontsize=8.2, weight="bold")
    _gate_row(ax, 0.345, "absolute z / annulus-radius calibration", done=False)
    _gate_row(ax, 0.285, "direct vs conjugate branch resolved", done=False)
    _gate_row(ax, 0.225, "input plane → SLM2 scale / centre / parity", done=False)
    _gate_row(ax, 0.165, "1030-nm SLM phase LUT / stroke", done=False)
    ax.text(0.5, 0.075, "No hardware-ready correction is exported while these gates are open.",
            transform=ax.transAxes, color=AMBER, fontsize=8.0, ha="center", va="center")


def _validation_schematic(ax: plt.Axes) -> None:
    ax.text(0.5, 0.76, "LOW-GAIN TRIAL", transform=ax.transAxes, color=FG,
            fontsize=13, weight="bold", ha="center")
    ax.add_patch(FancyArrowPatch((0.5, 0.70), (0.5, 0.62), transform=ax.transAxes,
                                 arrowstyle="-|>", mutation_scale=13, lw=1.25,
                                 color=CYAN))

    # Stack of camera frames.
    for k in range(4, -1, -1):
        dx, dy = k*0.012, k*0.011
        ax.add_patch(Rectangle((0.27+dx, 0.43+dy), 0.46, 0.16,
                               transform=ax.transAxes, facecolor="#10151b",
                               edgecolor="#6a7680", lw=0.8))
        ax.add_patch(Circle((0.50+dx, 0.51+dy), 0.045, transform=ax.transAxes,
                            fill=False, ec=AMBER, lw=1.25))
    ax.text(0.5, 0.385, "new 18-plane × 4-repeat BMG stack",
            transform=ax.transAxes, color=FG, fontsize=8.8, ha="center", weight="bold")

    ax.add_patch(FancyArrowPatch((0.5, 0.35), (0.5, 0.28), transform=ax.transAxes,
                                 arrowstyle="-|>", mutation_scale=13, lw=1.25,
                                 color=CYAN))
    ax.text(0.5, 0.23, "MEASURED before → after acceptance",
            transform=ax.transAxes, color=FG, fontsize=9.0, weight="bold", ha="center")
    ax.text(0.5, 0.17, "core / annular uniformity / propagation metrics",
            transform=ax.transAxes, color=MUTED, fontsize=8.1, ha="center")
    ax.text(0.5, 0.085,
            "A model prediction cannot accept its own correction.",
            transform=ax.transAxes, color=RED, fontsize=8.1, weight="bold", ha="center")


def build(out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    measured, phase, source_meta = _source_images()

    fig = plt.figure(figsize=(14.7, 6.15), facecolor=BG)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.27, 1.14, 1.00, 1.00],
                          left=0.028, right=0.985, bottom=0.205, top=0.78, wspace=0.045)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

    _card(axes[0], "Measure the distorted field", "complete q=20 longitudinal camera scan", 1,
          status="EXPERIMENTAL", status_color=GREEN)
    _place_image(axes[0], measured, box=(0.045, 0.12, 0.91, 0.70), background="#ffffff")
    axes[0].text(0.5, 0.075, "Measured X–Z evolution from the 72-frame acquisition",
                 transform=axes[0].transAxes, color=MUTED, fontsize=7.7, ha="center")

    _card(axes[1], "Infer the residual phase", "fit physical errors first; retrieve what remains", 2,
          status="MODEL-INFERRED", status_color=CYAN)
    _place_image(axes[1], phase, box=(0.045, 0.13, 0.91, 0.68), background="#ffffff")
    axes[1].text(0.5, 0.075,
                 "Current non-axisymmetric q=20 residual diagnostic\n(programmed vortex removed)",
                 transform=axes[1].transAxes, color=MUTED, fontsize=7.5,
                 ha="center", va="center")

    _card(axes[2], "Build the SLM2 correction", "Miao-style full-aperture path", 3,
          status="CALIBRATION-GATED", status_color=AMBER)
    _correction_schematic(axes[2])

    _card(axes[3], "Close the loop experimentally", "validation must use a new camera stack", 4,
          status="PENDING", status_color=RED)
    _validation_schematic(axes[3])

    # Flow arrows in figure coordinates.
    for a, b in zip(axes[:-1], axes[1:]):
        pa, pb = a.get_position(), b.get_position()
        y = 0.49*(pa.y0+pa.y1)
        fig.add_artist(FancyArrowPatch((pa.x1+0.003, y), (pb.x0-0.003, y),
                                       transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=14, lw=1.25, color=CYAN))

    fig.text(0.035, 0.942, "INTENSITY-ONLY ABERRATION IDENTIFICATION → CLOSED-LOOP CORRECTION",
             color=CYAN, fontsize=8.7, weight="bold", ha="left")
    fig.text(0.035, 0.885, "From measured q=20 propagation to a defensible SLM correction",
             color=FG, fontsize=21.0, weight="bold", ha="left")
    fig.text(0.035, 0.835,
             "Explicit beam / 4F / axicon errors are fitted in the digital twin first; the phase-retrieval stage targets the residual wavefront that the physical model cannot explain.",
             color=MUTED, fontsize=9.3, ha="left")

    # Bottom scientific-scope strip.
    x0, y0, w, h = 0.035, 0.055, 0.93, 0.10
    fig.add_artist(FancyBboxPatch((x0, y0), w, h, transform=fig.transFigure,
                                  boxstyle="round,pad=0.008", facecolor=CARD2,
                                  edgecolor=BORDER, lw=0.8))
    fig.text(0.055, 0.120, "CURRENT CLAIM", color=CYAN, fontsize=8.3, weight="bold", ha="left")
    fig.text(0.145, 0.120,
             "experimental z-stack + model-inferred residual phase + implemented calibration-gated correction pipeline",
             color=FG, fontsize=8.6, ha="left")
    fig.text(0.055, 0.082, "NOT YET CLAIMED", color=RED, fontsize=8.3, weight="bold", ha="left")
    fig.text(0.145, 0.082,
             "experimentally corrected q=20 beam — this requires applying the mask and acquiring a new before/after stack",
             color=MUTED, fontsize=8.5, ha="left")
    fig.text(0.955, 0.082, "Miao et al., Opt. Express 30, 11360–11371 (2022)",
             color="#7f8992", fontsize=7.1, ha="right")

    png = out / "poster_q20_inverse_correction_story.png"
    pdf = out / "poster_q20_inverse_correction_story.pdf"
    fig.savefig(png, dpi=500, facecolor=BG, bbox_inches="tight")
    fig.savefig(pdf, facecolor=BG, bbox_inches="tight")
    plt.close(fig)

    # A small review copy for chat inspection; never use this on the poster.
    with Image.open(png) as im:
        preview = im.convert("RGB")
        preview.thumbnail((2400, 1200), Image.Resampling.LANCZOS)
        preview_path = out / "poster_q20_inverse_correction_story.preview.jpg"
        preview.save(preview_path, quality=90, subsampling=0)

    manifest = {
        "outcome": "POSTER-Q20-INVERSE-V3",
        "final_assets": {
            "png_500dpi": str(png),
            "pdf_vector_text_and_schematic": str(pdf),
            "review_preview_only": str(preview_path),
        },
        "scientific_scope": {
            "measured_longitudinal_stack": "experimental",
            "displayed_phase": "current model-inferred non-axisymmetric residual diagnostic",
            "programmed_vortex_in_correction": False,
            "hardware_correction_status": "calibration-gated",
            "post_slm_camera_stack_available": False,
            "simulated_corrected_beam_labelled_experimental": False,
        },
        "source_provenance": source_meta,
        "reference": "B. Miao et al., Optics Express 30, 11360-11371 (2022), DOI 10.1364/OE.454796",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    build()
