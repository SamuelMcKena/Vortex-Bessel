"""Clean data-led q=20 correction figure for the PhD poster.

This pass removes the status/disclaimer-card language from the previous concept
figures.  The visual grammar is simply:

    MEASUREMENT -> PHYSICAL FIT -> RESIDUAL PHASE -> CORRECTED FIELD
                                                    compared with IDEAL

Both transverse and longitudinal correction results are shown.  The physical-fit
column explicitly ties the existing beam/SLM/4F/axicon error library into the
residual-phase retrieval without turning the poster into an optimiser diagram.
"""
from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
import numpy as np
from PIL import Image, ImageDraw

import build_poster_q20_inverse_v4 as v4

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "poster" / "q20_inverse_v5"

BG = "#070a0d"
PANEL = "#0b1015"
FG = "#f5f6f7"
MUTED = "#a9b2bb"
CYAN = "#4dd9d5"
GOLD = "#ffbd4a"
BORDER = "#394550"


def _crop(im: Image.Image, box: tuple[float,float,float,float]) -> Image.Image:
    w,h = im.size
    l,t,r,b = box
    return im.crop((round(l*w), round(t*h), round(r*w), round(b*h)))


def _clean_panels(src: dict[str,Image.Image]) -> dict[str,Image.Image]:
    # Second-stage crops remove old matplotlib margins, neighbouring axes,
    # colorbars and legacy labels from the tracked comparison figures.
    boxes = {
        "xy_measured":  (0.135, 0.000, 0.915, 0.935),
        "xy_corrected": (0.057, 0.000, 0.807, 0.935),
        "xy_ideal":     (0.000, 0.000, 0.704, 0.935),
        "xz_measured":  (0.100, 0.014, 0.907, 0.905),
        "xz_corrected": (0.020, 0.010, 0.800, 0.900),
        "xz_ideal":     (0.000, 0.010, 0.710, 0.900),
        "phase":        (0.045, 0.075, 0.890, 0.895),
    }
    out = {name:_crop(src[name], box) for name,box in boxes.items()}

    # Remove two annotations inherited from the old single-plane diagnostic.
    # They sit in dark corners, so this does not remove useful beam structure.
    for name, frac_w in (("xy_measured",0.40),("xy_ideal",0.42)):
        im = out[name].copy()
        draw = ImageDraw.Draw(im)
        w,h = im.size
        draw.rectangle((0, int(0.885*h), int(frac_w*w), h), fill=(0,0,0))
        out[name] = im
    return out


def _image_panel(ax: plt.Axes, im: Image.Image) -> None:
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_edgecolor(BORDER); s.set_linewidth(0.8)
    ax.imshow(np.asarray(im), interpolation="nearest", aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])


def _physical_fit(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_edgecolor(BORDER); s.set_linewidth(0.8)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xticks([]); ax.set_yticks([])

    labels = ("Beam", "SLM", "4F", "Axicon")
    ys = (0.78,0.60,0.42,0.24)
    for y,name in zip(ys,labels):
        ax.add_patch(Circle((0.28,y),0.070, facecolor="#111a22", edgecolor=CYAN, lw=1.25))
        if name == "Beam":
            ax.add_patch(Circle((0.28,y),0.038,fill=False,ec=GOLD,lw=1.15))
            ax.add_patch(Circle((0.28,y),0.020,fill=False,ec=GOLD,lw=.9))
        elif name == "SLM":
            for d in (-.032,0,.032):
                ax.plot([.248+d,.312+d],[y-.032,y+.032],color=GOLD,lw=.7)
            for d in (-.032,0,.032):
                ax.plot([.248,.312],[y+d,y+d],color=GOLD,lw=.7)
        elif name == "4F":
            ax.add_patch(Circle((.28,y),.038,fill=False,ec=GOLD,lw=1.1))
            ax.add_patch(Circle((.300,y),.017,fill=False,ec=FG,lw=.8))
        else:
            ax.plot([.235,.28,.325],[y-.04,y+.045,y-.04],color=GOLD,lw=1.25)
        ax.text(.43,y,name,color=FG,fontsize=9.0,weight="bold",ha="left",va="center")

    ax.text(.5,.085,"fit across z",color=CYAN,fontsize=8.2,weight="bold",ha="center")


def _phase_panel(ax: plt.Axes, im: Image.Image) -> None:
    _image_panel(ax,im)
    ax.text(.5,-.065,r"$\phi_{\mathrm{corr}}=-\psi_{\mathrm{res}}$",
            transform=ax.transAxes,color=CYAN,fontsize=10.2,weight="bold",ha="center",va="top")


def _arrow(fig: plt.Figure, left: plt.Axes, right: plt.Axes, y: float=.50) -> None:
    a,b=left.get_position(),right.get_position()
    yy=a.y0+y*(a.y1-a.y0)
    fig.add_artist(FancyArrowPatch((a.x1+.004,yy),(b.x0-.004,yy),
                                   transform=fig.transFigure,arrowstyle="-|>",
                                   mutation_scale=14,lw=1.25,color=CYAN))


def _save_review(out:Path, clean:dict[str,Image.Image]) -> list[str]:
    d=out/"review_crops"; d.mkdir(parents=True,exist_ok=True)
    paths=[]
    for name,im in clean.items():
        p=d/f"{name}.png"; im.save(p); paths.append(str(p))
    return paths


def build(out:Path=OUT) -> dict:
    out.mkdir(parents=True,exist_ok=True)
    raw=v4._read_sources()
    clean=_clean_panels(raw)
    review=_save_review(out,clean)

    fig=plt.figure(figsize=(16.5,7.2),facecolor=BG)
    gs=fig.add_gridspec(2,5,
        width_ratios=[1.16,.58,.88,1.05,1.05],
        height_ratios=[1,1],
        left=.04,right=.985,bottom=.09,top=.78,wspace=.12,hspace=.16)

    ax_mxy=fig.add_subplot(gs[0,0]); _image_panel(ax_mxy,clean["xy_measured"])
    ax_mxz=fig.add_subplot(gs[1,0]); _image_panel(ax_mxz,clean["xz_measured"])

    ax_fit=fig.add_subplot(gs[:,1]); _physical_fit(ax_fit)
    ax_phase=fig.add_subplot(gs[:,2]); _phase_panel(ax_phase,clean["phase"])

    ax_cxy=fig.add_subplot(gs[0,3]); _image_panel(ax_cxy,clean["xy_corrected"])
    ax_cxz=fig.add_subplot(gs[1,3]); _image_panel(ax_cxz,clean["xz_corrected"])
    ax_ixy=fig.add_subplot(gs[0,4]); _image_panel(ax_ixy,clean["xy_ideal"])
    ax_ixz=fig.add_subplot(gs[1,4]); _image_panel(ax_ixz,clean["xz_ideal"])

    _arrow(fig,ax_mxy,ax_fit,.52)
    _arrow(fig,ax_fit,ax_phase,.50)
    _arrow(fig,ax_phase,ax_cxy,.52)

    # Minimal academic labels only.
    fig.text(.04,.94,"Measurement-driven aberration correction",color=FG,
             fontsize=22.0,weight="bold",ha="left")
    fig.text(.04,.887,"Physical-error fitting + residual phase retrieval",color=MUTED,
             fontsize=10.2,ha="left")

    headings=(
        (ax_mxy,"MEASUREMENT"),
        (ax_fit,"PHYSICAL FIT"),
        (ax_phase,"RESIDUAL PHASE"),
        (ax_cxy,"CORRECTED"),
        (ax_ixy,"IDEAL"),
    )
    for ax,text in headings:
        p=ax.get_position()
        fig.text((p.x0+p.x1)/2,.816,text,color=MUTED,fontsize=8.6,
                 weight="bold",ha="center")

    # Row labels identify the two observables without adding captions to every panel.
    ptop=ax_mxy.get_position(); pbot=ax_mxz.get_position()
    fig.text(.018,(ptop.y0+ptop.y1)/2,"TRANSVERSE",color="#7f8a94",fontsize=7.8,
             weight="bold",ha="center",va="center",rotation=90)
    fig.text(.018,(pbot.y0+pbot.y1)/2,"PROPAGATION",color="#7f8a94",fontsize=7.8,
             weight="bold",ha="center",va="center",rotation=90)

    # A thin return arrow communicates iteration without a separate prose panel.
    fig.add_artist(FancyArrowPatch((.91,.055),(.10,.055),transform=fig.transFigure,
                                   connectionstyle="arc3,rad=-.045",arrowstyle="-|>",
                                   mutation_scale=11,lw=.95,color="#606c76"))
    fig.text(.505,.028,"repeat",color="#717c86",fontsize=7.4,ha="center")

    png=out/"poster_q20_correction_visual.png"
    pdf=out/"poster_q20_correction_visual.pdf"
    fig.savefig(png,dpi=500,bbox_inches="tight",facecolor=BG)
    fig.savefig(pdf,bbox_inches="tight",facecolor=BG)
    plt.close(fig)

    with Image.open(png) as im:
        preview=im.convert("RGB"); preview.thumbnail((2500,1300),Image.Resampling.LANCZOS)
        prev=out/"poster_q20_correction_visual.preview.jpg"
        preview.save(prev,quality=92,subsampling=0)
        size=list(im.size); dpi=list(im.info.get("dpi",(0,0)))

    manifest={
        "outcome":"POSTER-Q20-INVERSE-V5",
        "visual_flow":["measurement","physical error fit","residual phase","corrected field","ideal reference","repeat"],
        "physical_fit_families_shown":["beam","SLM","4F","axicon"],
        "observables_shown":["transverse intensity","longitudinal XZ propagation"],
        "assets":{"png_500dpi":str(png),"pdf":str(pdf),"preview":str(prev)},
        "png_pixel_size":size,"png_dpi":dpi,"review_crops":review,
        "scientific_provenance":{
            "input":"tracked experimental q20 camera products",
            "corrected_field":"tracked q20 correction-model products",
            "programmed_qtheta_removed_from_residual":True,
        },
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest,indent=2))
    return manifest


if __name__=="__main__":
    build()
