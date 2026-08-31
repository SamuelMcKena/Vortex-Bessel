"""Build a poster-ready q=20 inverse/correction story without overstating validation.

Tracked experimental images are used only for stages that have actually been
measured/inferred.  Hardware correction and post-SLM validation are drawn as a
vector workflow because those measurements have not yet been completed.

Final export: 500-dpi PNG plus PDF.  The PDF keeps all labels/schematic elements
as vector graphics while the tracked experimental source figures are embedded at
native resolution.
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
    diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white")).convert("L")
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
    # Same experimentally reviewed XZ crop used by export_q20_figures_for_chat.yml.
    measured_xz = _trim_white(
        _fraction_crop(measured_full, (0.002, 0.015, 0.218, 0.53)), pad=12)
    phase_full = Image.open(PHASE_SOURCE).convert("RGB")
    phase = _trim_white(phase_full, pad=14)
    return measured_xz, phase, {
        "measured_source": str(MEASURED_SOURCE.relative_to(ROOT)),
        "measured_source_pixel_size": list(measured_full.size),
        "measured_crop_fraction": [0.002, 0.015, 0.218, 0.53],
        "measured_crop_pixel_size": list(measured_xz.size),
        "phase_source": str(PHASE_SOURCE.relative_to(ROOT)),
        "phase_source_pixel_size": list(phase_full.size),
    }


def _card(ax: plt.Axes, title: str, kicker: str, index: int, *, status: str,
          status_color: str) -> None:
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(1.0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax.text(0.055, 0.947, f"{index}", color=BG, ha="center", va="center",
            fontsize=10.2, weight="bold",
            bbox=dict(boxstyle="circle,pad=0.34", fc=CYAN, ec="none"))
    ax.text(0.12, 0.965, title, transform=ax.transAxes, color=FG,
            fontsize=11.2, weight="bold", ha="left", va="top")
    ax.text(0.12, 0.914, kicker, transform=ax.transAxes, color=MUTED,
            fontsize=7.6, ha="left", va="top")
    # Put status on its own row so it never collides with the card title.
    ax.text(0.945, 0.862, status, color=status_color, fontsize=7.0, weight="bold",
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="#0a0e13", ec=status_color,
                      lw=0.8, alpha=0.98))


def _place_image(ax: plt.Axes, im: Image.Image, box=(0.05, 0.13, 0.90, 0.67)) -> None:
    x, y, w, h = box
    ax.add_patch(FancyBboxPatch((x, y), w, h, transform=ax.transAxes,
                                boxstyle="round,pad=0.008", facecolor="white",
                                edgecolor="#59636d", linewidth=0.7, zorder=1))
    arr = np.asarray(im)
    ih, iw = arr.shape[:2]
    target_ratio, image_ratio = w/h, iw/ih
    if image_ratio >= target_ratio:
        dw, dh = w, w/image_ratio
        x0, y0 = x, y + (h-dh)/2
    else:
        dh, dw = h, h*image_ratio
        x0, y0 = x + (w-dw)/2, y
    inset = ax.inset_axes([x0+0.01, y0+0.01, max(dw-0.02, 0.02), max(dh-0.02, 0.02)])
    inset.imshow(arr, interpolation="nearest")
    inset.axis("off")


def _gate_row(ax: plt.Axes, y: float, text: str) -> None:
    ax.add_patch(Circle((0.12, y), 0.017, transform=ax.transAxes,
                        facecolor="none", edgecolor=AMBER, lw=1.45))
    ax.text(0.165, y, text, transform=ax.transAxes, color="#d8dde1",
            fontsize=7.9, ha="left", va="center")


def _correction_schematic(ax: plt.Axes) -> None:
    ax.text(0.5, 0.735, r"$\phi_{\mathrm{corr}}(\rho,\theta)=-\psi_{\mathrm{res}}(\rho,\theta)$",
            transform=ax.transAxes, color=FG, fontsize=14.5, weight="bold",
            ha="center", va="center")
    ax.text(0.5, 0.675, "programmed qθ is excluded from the correction",
            transform=ax.transAxes, color=CYAN, fontsize=7.7, ha="center")

    x0, y0, w, h = 0.30, 0.485, 0.40, 0.12
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, transform=ax.transAxes,
                                boxstyle="round,pad=0.012", facecolor="#151d26",
                                edgecolor="#708090", lw=1.0))
    for i in range(9):
        xx = x0 + w*i/8
        ax.plot([xx, xx], [y0, y0+h], transform=ax.transAxes, color="#293440", lw=0.4)
    for j in range(5):
        yy = y0 + h*j/4
        ax.plot([x0, x0+w], [yy, yy], transform=ax.transAxes, color="#293440", lw=0.4)
    ax.text(0.5, 0.545, "SLM2", transform=ax.transAxes, color=FG,
            fontsize=10.5, weight="bold", ha="center", va="center")

    ax.text(0.08, 0.415, "Hardware-map gates", transform=ax.transAxes,
            color=MUTED, fontsize=8.0, weight="bold")
    _gate_row(ax, 0.355, "absolute z / annulus-radius calibration")
    _gate_row(ax, 0.300, "direct vs conjugate branch resolved")
    _gate_row(ax, 0.245, "input plane → SLM2 scale / centre / parity")
    _gate_row(ax, 0.190, "1030-nm SLM phase LUT / stroke")
    ax.text(0.5, 0.095, "Hardware output remains blocked\nuntil every gate is closed.",
            transform=ax.transAxes, color=AMBER, fontsize=7.4,
            ha="center", va="center")


def _validation_schematic(ax: plt.Axes) -> None:
    ax.text(0.5, 0.735, "LOW-GAIN TRIAL", transform=ax.transAxes, color=FG,
            fontsize=12.8, weight="bold", ha="center")
    ax.add_patch(FancyArrowPatch((0.5, 0.685), (0.5, 0.61), transform=ax.transAxes,
                                 arrowstyle="-|>", mutation_scale=13, lw=1.2, color=CYAN))
    for k in range(4, -1, -1):
        dx, dy = k*0.012, k*0.011
        ax.add_patch(Rectangle((0.27+dx, 0.43+dy), 0.46, 0.15,
                               transform=ax.transAxes, facecolor="#10151b",
                               edgecolor="#6a7680", lw=0.8))
        ax.add_patch(Circle((0.50+dx, 0.505+dy), 0.043, transform=ax.transAxes,
                            fill=False, ec=AMBER, lw=1.2))
    ax.text(0.5, 0.382, "new 18-plane × 4-repeat BMG stack",
            transform=ax.transAxes, color=FG, fontsize=8.2, ha="center", weight="bold")
    ax.add_patch(FancyArrowPatch((0.5, 0.345), (0.5, 0.285), transform=ax.transAxes,
                                 arrowstyle="-|>", mutation_scale=13, lw=1.2, color=CYAN))
    ax.text(0.5, 0.235, "MEASURED before → after acceptance",
            transform=ax.transAxes, color=FG, fontsize=8.4, weight="bold", ha="center")
    ax.text(0.5, 0.183, "core / annular uniformity / propagation metrics",
            transform=ax.transAxes, color=MUTED, fontsize=7.4, ha="center")
    ax.text(0.5, 0.095, "Model prediction ≠ experimental acceptance.",
            transform=ax.transAxes, color=RED, fontsize=7.5, weight="bold", ha="center")


def build(out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    measured, phase, source_meta = _source_images()

    fig = plt.figure(figsize=(15.6, 6.15), facecolor=BG)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.28, 1.14, 1.00, 1.00],
                          left=0.025, right=0.987, bottom=0.205, top=0.78, wspace=0.045)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

    _card(axes[0], "Measure the distorted field", "complete q=20 longitudinal camera scan", 1,
          status="EXPERIMENTAL", status_color=GREEN)
    _place_image(axes[0], measured, box=(0.045, 0.135, 0.91, 0.65))
    axes[0].text(0.5, 0.078, "Measured X–Z evolution from the 72-frame acquisition",
                 transform=axes[0].transAxes, color=MUTED, fontsize=7.3, ha="center")

    _card(axes[1], "Infer the residual phase", "fit physical errors first; retrieve what remains", 2,
          status="MODEL-INFERRED", status_color=CYAN)
    _place_image(axes[1], phase, box=(0.045, 0.145, 0.91, 0.64))
    axes[1].text(0.5, 0.078,
                 "Current non-axisymmetric q=20 residual diagnostic\n(programmed vortex removed)",
                 transform=axes[1].transAxes, color=MUTED, fontsize=7.0,
                 ha="center", va="center")

    _card(axes[2], "Build the SLM2 correction", "Miao-style full-aperture path", 3,
          status="CALIBRATION-GATED", status_color=AMBER)
    _correction_schematic(axes[2])

    _card(axes[3], "Close the loop", "new post-SLM camera stack required", 4,
          status="PENDING", status_color=RED)
    _validation_schematic(axes[3])

    for a, b in zip(axes[:-1], axes[1:]):
        pa, pb = a.get_position(), b.get_position()
        y = 0.49*(pa.y0+pa.y1)
        fig.add_artist(FancyArrowPatch((pa.x1+0.002, y), (pb.x0-0.002, y),
                                       transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=14, lw=1.2, color=CYAN))

    fig.text(0.032, 0.942, "INTENSITY-ONLY ABERRATION IDENTIFICATION → CLOSED-LOOP CORRECTION",
             color=CYAN, fontsize=8.6, weight="bold", ha="left")
    fig.text(0.032, 0.885, "From measured q=20 propagation to a defensible SLM correction",
             color=FG, fontsize=20.4, weight="bold", ha="left")
    fig.text(0.032, 0.835,
             "Explicit beam / 4F / axicon errors are fitted in the digital twin first; phase retrieval then targets the residual wavefront the physical model cannot explain.",
             color=MUTED, fontsize=9.0, ha="left")

    x0, y0, w, h = 0.032, 0.055, 0.936, 0.10
    fig.add_artist(FancyBboxPatch((x0, y0), w, h, transform=fig.transFigure,
                                  boxstyle="round,pad=0.008", facecolor=CARD2,
                                  edgecolor=BORDER, lw=0.8))
    fig.text(0.052, 0.120, "CURRENT CLAIM", color=CYAN, fontsize=8.2, weight="bold", ha="left")
    fig.text(0.142, 0.120,
             "experimental z-stack + model-inferred residual phase + implemented calibration-gated correction pipeline",
             color=FG, fontsize=8.3, ha="left")
    fig.text(0.052, 0.082, "NOT YET CLAIMED", color=RED, fontsize=8.2, weight="bold", ha="left")
    fig.text(0.142, 0.082,
             "experimentally corrected q=20 beam — this requires applying the mask and acquiring a new before/after stack",
             color=MUTED, fontsize=8.2, ha="left")
    fig.text(0.958, 0.082, "Miao et al., Opt. Express 30, 11360–11371 (2022)",
             color="#7f8992", fontsize=6.9, ha="right")

    png = out / "poster_q20_inverse_correction_story.png"
    pdf = out / "poster_q20_inverse_correction_story.pdf"
    fig.savefig(png, dpi=500, facecolor=BG, bbox_inches="tight")
    fig.savefig(pdf, facecolor=BG, bbox_inches="tight")
    plt.close(fig)

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
