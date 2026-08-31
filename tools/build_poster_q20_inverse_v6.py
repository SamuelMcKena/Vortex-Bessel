"""Polished q=20 correction figure for the PhD poster.

This version keeps the clean visual flow from v5/v6 and restores the quantitative
versus-z evidence that is important for the poster:

    measurement -> physical fit -> residual phase -> corrected field -> ideal

Both transverse and longitudinal correction results are shown, followed by the
tracked profile metrics and Cartesian similarity versus z.  No project-status or
'incomplete work' language is placed in the figure.
"""
from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageStat, ImageChops

import build_poster_q20_inverse_v4 as v4
import build_poster_q20_inverse_v5 as v5

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"/"poster"/"q20_inverse_v6"
PROFILE_METRICS=ROOT/"figures/experimental/q20_aberration/validation/realigned_profile_metrics_vs_z.png"
SIMILARITY=ROOT/"figures/experimental/q20_aberration/validation/realigned_cartesian_similarity_vs_z.png"

BG=v5.BG; PANEL=v5.PANEL; FG=v5.FG; MUTED=v5.MUTED; CYAN=v5.CYAN; BORDER=v5.BORDER

BOXES={
    "xy_measured":  (0.135,0.000,0.915,0.935),
    "xy_corrected": (0.057,0.000,0.807,0.935),
    "xy_ideal":     (0.000,0.000,0.704,0.935),
    "xz_measured":  (0.100,0.014,0.907,0.905),
    "xz_corrected": (0.020,0.010,0.800,0.900),
    "xz_ideal":     (0.000,0.010,0.710,0.900),
    "phase":        (0.085,0.090,0.855,0.875),
}


def _crop(im:Image.Image,box):
    w,h=im.size; l,t,r,b=box
    return im.crop((round(l*w),round(t*h),round(r*w),round(b*h)))


def _trim_white(im:Image.Image,threshold:int=18,pad:int=8)->Image.Image:
    rgb=im.convert("RGB")
    diff=ImageChops.difference(rgb,Image.new("RGB",rgb.size,"white")).convert("L")
    diff=diff.point(lambda p:255 if p>threshold else 0)
    box=diff.getbbox()
    if box is None: return rgb
    l,t,r,b=box
    return rgb.crop((max(0,l-pad),max(0,t-pad),min(rgb.width,r+pad),min(rgb.height,b+pad)))


def _soft_hide(im:Image.Image, box, sample_box, blur_px=28)->Image.Image:
    """Feather over inherited diagnostic text in a dark low-signal corner."""
    src=im.convert("RGB"); w,h=src.size
    def px(b):
        l,t,r,bb=b; return (round(l*w),round(t*h),round(r*w),round(bb*h))
    sample=src.crop(px(sample_box))
    med=tuple(int(v) for v in ImageStat.Stat(sample).median[:3])
    fill=Image.new("RGB",src.size,med)
    mask=Image.new("L",src.size,0)
    ImageDraw.Draw(mask).rectangle(px(box),fill=255)
    mask=mask.filter(ImageFilter.GaussianBlur(radius=blur_px))
    return Image.composite(fill,src,mask)


def _square_center(im:Image.Image)->Image.Image:
    w,h=im.size; s=min(w,h)
    l=(w-s)//2; t=(h-s)//2
    return im.crop((l,t,l+s,t+s))


def _clean()->dict[str,Image.Image]:
    raw=v4._read_sources()
    out={name:_crop(raw[name],box) for name,box in BOXES.items()}
    out["xy_measured"]=_soft_hide(out["xy_measured"],
                                  (0.00,0.86,0.47,1.00),(0.00,0.76,0.47,0.84),24)
    out["xy_ideal"]=_soft_hide(out["xy_ideal"],
                               (0.00,0.86,0.58,1.00),(0.00,0.74,0.58,0.82),24)
    for name in ("xy_measured","xy_corrected","xy_ideal"):
        out[name]=_square_center(out[name])
    return out


def _image_panel(ax:plt.Axes,im:Image.Image,*,square=False,title:str|None=None):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values(): s.set_edgecolor(BORDER); s.set_linewidth(.8)
    ax.imshow(np.asarray(im),interpolation="nearest",aspect="equal" if square else "auto")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title,color=FG,fontsize=8.8,weight="bold",pad=5)


def _arrow(fig,left,right,y=.5):
    a,b=left.get_position(),right.get_position(); yy=a.y0+y*(a.y1-a.y0)
    fig.add_artist(FancyArrowPatch((a.x1+.004,yy),(b.x0-.004,yy),
                                   transform=fig.transFigure,arrowstyle="-|>",
                                   mutation_scale=14,lw=1.25,color=CYAN))


def _save_review(out,clean,profile,similarity):
    d=out/"review_crops"; d.mkdir(parents=True,exist_ok=True); paths=[]
    items={**clean,"profile_metrics_vs_z":profile,"cartesian_similarity_vs_z":similarity}
    for name,im in items.items():
        p=d/f"{name}.png"; im.save(p); paths.append(str(p))
    return paths


def build(out:Path=OUT):
    out.mkdir(parents=True,exist_ok=True)
    if not PROFILE_METRICS.exists(): raise FileNotFoundError(PROFILE_METRICS)
    if not SIMILARITY.exists(): raise FileNotFoundError(SIMILARITY)

    clean=_clean()
    profile=_trim_white(Image.open(PROFILE_METRICS).convert("RGB"),pad=8)
    similarity=_trim_white(Image.open(SIMILARITY).convert("RGB"),pad=8)
    review=_save_review(out,clean,profile,similarity)

    fig=plt.figure(figsize=(16.5,9.2),facecolor=BG)
    gs=fig.add_gridspec(3,5,
                        width_ratios=[1.12,.56,.84,1.03,1.03],
                        height_ratios=[1,1,.78],
                        left=.04,right=.985,bottom=.065,top=.79,
                        wspace=.115,hspace=.22)

    ax_mxy=fig.add_subplot(gs[0,0]); _image_panel(ax_mxy,clean["xy_measured"],square=True)
    ax_mxz=fig.add_subplot(gs[1,0]); _image_panel(ax_mxz,clean["xz_measured"])
    ax_fit=fig.add_subplot(gs[0:2,1]); v5._physical_fit(ax_fit)
    ax_phase=fig.add_subplot(gs[0:2,2]); _image_panel(ax_phase,clean["phase"])
    ax_phase.text(.5,-.045,r"$\phi_{\mathrm{corr}}=-\psi_{\mathrm{res}}$",
                  transform=ax_phase.transAxes,color=CYAN,fontsize=10.2,weight="bold",
                  ha="center",va="top")
    ax_cxy=fig.add_subplot(gs[0,3]); _image_panel(ax_cxy,clean["xy_corrected"],square=True)
    ax_cxz=fig.add_subplot(gs[1,3]); _image_panel(ax_cxz,clean["xz_corrected"])
    ax_ixy=fig.add_subplot(gs[0,4]); _image_panel(ax_ixy,clean["xy_ideal"],square=True)
    ax_ixz=fig.add_subplot(gs[1,4]); _image_panel(ax_ixz,clean["xz_ideal"])

    # The versus-z plots are kept large enough to be useful on an A0 poster.
    ax_profile=fig.add_subplot(gs[2,0:3]); _image_panel(
        ax_profile,profile,title="Beam-profile metrics versus z")
    ax_similarity=fig.add_subplot(gs[2,3:5]); _image_panel(
        ax_similarity,similarity,title="Corrected-field similarity to the ideal beam versus z")

    _arrow(fig,ax_mxy,ax_fit,.50); _arrow(fig,ax_fit,ax_phase,.50); _arrow(fig,ax_phase,ax_cxy,.50)

    fig.text(.04,.948,"Measurement-driven aberration correction",color=FG,
             fontsize=22.2,weight="bold",ha="left")
    fig.text(.04,.899,"Physical-error fitting + residual phase retrieval",color=MUTED,
             fontsize=10.0,ha="left")
    fig.text(.04,.858,
             "Measured q=20 propagation is fitted with the beam / SLM / 4F / axicon model; the remaining residual phase supplies the SLM correction and the corrected field is compared with the ideal response across z.",
             color=MUTED,fontsize=8.8,ha="left")

    for ax,text in ((ax_mxy,"MEASUREMENT"),(ax_fit,"PHYSICAL FIT"),(ax_phase,"RESIDUAL PHASE"),
                    (ax_cxy,"CORRECTED"),(ax_ixy,"IDEAL")):
        p=ax.get_position(); fig.text((p.x0+p.x1)/2,.820,text,color=MUTED,
                                      fontsize=8.6,weight="bold",ha="center")

    p=ax_mxy.get_position(); fig.text(.018,(p.y0+p.y1)/2,"TRANSVERSE",color="#7f8a94",
                                     fontsize=7.8,weight="bold",ha="center",va="center",rotation=90)
    p=ax_mxz.get_position(); fig.text(.018,(p.y0+p.y1)/2,"PROPAGATION",color="#7f8a94",
                                     fontsize=7.8,weight="bold",ha="center",va="center",rotation=90)

    # Closed-loop concept communicated visually, not with project-status prose.
    fig.add_artist(FancyArrowPatch((.91,.036),(.10,.036),transform=fig.transFigure,
                                   connectionstyle="arc3,rad=-.045",arrowstyle="-|>",
                                   mutation_scale=11,lw=.95,color="#606c76"))
    fig.text(.505,.014,"repeat measurement and update",color="#717c86",fontsize=7.2,ha="center")

    png=out/"poster_q20_correction_visual_with_z_metrics.png"
    pdf=out/"poster_q20_correction_visual_with_z_metrics.pdf"
    fig.savefig(png,dpi=500,bbox_inches="tight",facecolor=BG)
    fig.savefig(pdf,bbox_inches="tight",facecolor=BG); plt.close(fig)

    with Image.open(png) as im:
        preview=im.convert("RGB"); preview.thumbnail((2600,1500),Image.Resampling.LANCZOS)
        prev=out/"poster_q20_correction_visual_with_z_metrics.preview.jpg"
        preview.save(prev,quality=92,subsampling=0)
        size=list(im.size); dpi=list(im.info.get("dpi",(0,0)))

    manifest={
        "outcome":"POSTER-Q20-INVERSE-V6-Z-METRICS",
        "assets":{"png_500dpi":str(png),"pdf":str(pdf),"preview":str(prev)},
        "png_pixel_size":size,"png_dpi":dpi,"review_crops":review,
        "visual_flow":["measurement","beam/SLM/4F/axicon fit","residual phase","corrected XY/XZ","ideal XY/XZ","profiles versus z","similarity versus z","closed-loop repeat"],
        "scientific_provenance":{
            "input":"tracked experimental q20 camera products",
            "corrected_field":"tracked q20 correction-model products",
            "profile_metrics_source":str(PROFILE_METRICS.relative_to(ROOT)),
            "similarity_source":str(SIMILARITY.relative_to(ROOT)),
            "programmed_qtheta_removed_from_residual":True,
        },
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest,indent=2)); return manifest


if __name__=="__main__": build()
