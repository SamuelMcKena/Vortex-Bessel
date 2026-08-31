"""Second integrated poster layout for the model-assisted correction workflow.

Keeps the real physical-fit validation from v1, but stops the legacy annular
phase raster from dominating the page.  The residual-phase diagnostic is shown at
its natural aspect inside a compact card; corrected/ideal propagation and the
z-dependent agreement plot receive the main result space.
"""
from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image

import build_poster_hybrid_correction_v1 as v1
import build_poster_q20_inverse_v6 as q6

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"/"poster"/"hybrid_correction_v2"
SIMILARITY=v1.SIMILARITY
BG=v1.BG; CARD=v1.CARD; FG=v1.FG; MUTED=v1.MUTED; CYAN=v1.CYAN; BORDER=v1.BORDER


def _phase_card(ax,phase:Image.Image):
    ax.set_facecolor(CARD);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
    ax.add_patch(FancyBboxPatch((.01,.01),.98,.98,boxstyle="round,pad=.012",
                                transform=ax.transAxes,fc=CARD,ec=BORDER,lw=.9))
    ax.text(.06,.94,"Residual phase",color=FG,fontsize=10.7,weight="bold",ha="left",va="top")
    arr=np.asarray(phase)
    # Natural image aspect: do not stretch the theta-vs-z diagnostic vertically.
    inset=ax.inset_axes([.055,.49,.89,.34])
    inset.imshow(arr,interpolation="nearest",aspect="equal")
    inset.set_xticks([]);inset.set_yticks([])
    for s in inset.spines.values():s.set_color(BORDER);s.set_linewidth(.65)
    ax.text(.5,.365,r"$\phi_{\rm corr}(\rho,\theta)=-\psi_{\rm res}(\rho,\theta)$",
            color=CYAN,fontsize=10.4,weight="bold",ha="center")
    ax.text(.5,.265,"retrieve only the wavefront not explained\nby the fitted physical model",
            color=MUTED,fontsize=7.3,ha="center",va="center",linespacing=1.35)
    ax.text(.5,.105,"programmed qθ excluded",color="#7f8a94",fontsize=6.8,ha="center")


def _arrow(fig,left,right,y=.51):
    a,b=left.get_position(),right.get_position();yy=a.y0+y*(a.y1-a.y0)
    fig.add_artist(FancyArrowPatch((a.x1+.004,yy),(b.x0-.004,yy),transform=fig.transFigure,
                                   arrowstyle="-|>",mutation_scale=14,lw=1.25,color=CYAN))


def build(out:Path=OUT):
    out.mkdir(parents=True,exist_ok=True)
    q=q6._clean()
    target,fitted,axis,fit,reg=v1._fit_validation()
    similarity=v1._trim_white(Image.open(SIMILARITY).convert("RGB"),pad=4)

    fig=plt.figure(figsize=(17.2,8.35),facecolor=BG)
    gs=fig.add_gridspec(3,5,width_ratios=[1.02,1.08,.72,1.13,1.13],
                        height_ratios=[.62,1.00,.80],left=.035,right=.987,
                        bottom=.065,top=.80,wspace=.105,hspace=.17)

    # 1. Measured q=20 intensity.
    ax_mxy=fig.add_subplot(gs[0,0]);v1._image(ax_mxy,q["xy_measured"],square=True,title="Measured transverse field")
    ax_mxz=fig.add_subplot(gs[1:,0]);v1._image(ax_mxz,q["xz_measured"],title="Measured propagation")

    # 2. The actual simulation library is searched here; the inset is a synthetic
    # validation so it is not mistaken for a fit to the tracked q20 BMG data.
    ax_fit=fig.add_subplot(gs[:,1]);v1._fit_card(ax_fit,target,fitted,axis,fit,reg)

    # 3. Existing q20 residual retrieval, compact and unstretched.
    ax_phase=fig.add_subplot(gs[:,2]);_phase_card(ax_phase,q["phase"])

    # 4. Put the useful correction evidence at the centre of attention.
    ax_cxy=fig.add_subplot(gs[0,3]);v1._image(ax_cxy,q["xy_corrected"],square=True,title="Corrected")
    ax_ixy=fig.add_subplot(gs[0,4]);v1._image(ax_ixy,q["xy_ideal"],square=True,title="Ideal")
    ax_cxz=fig.add_subplot(gs[1,3]);v1._image(ax_cxz,q["xz_corrected"],title="Corrected propagation")
    ax_ixz=fig.add_subplot(gs[1,4]);v1._image(ax_ixz,q["xz_ideal"],title="Ideal propagation")
    ax_metric=fig.add_subplot(gs[2,3:5]);v1._image(ax_metric,similarity,title="Agreement across the z-stack")

    _arrow(fig,ax_mxy,ax_fit);_arrow(fig,ax_fit,ax_phase);_arrow(fig,ax_phase,ax_cxy)

    fig.text(.035,.944,"Model-assisted intensity-only correction",color=FG,fontsize=22.5,weight="bold",ha="left")
    fig.text(.035,.891,"Measured propagation  →  physical-system fit  →  residual wavefront  →  SLM correction",
             color=MUTED,fontsize=9.8,ha="left")

    for ax,text in ((ax_mxy,"MEASUREMENT"),(ax_fit,"DIGITAL TWIN"),(ax_phase,"PHASE RETRIEVAL"),(ax_cxy,"CORRECTION RESULT")):
        p=ax.get_position();fig.text((p.x0+p.x1)/2,.825,text,color=MUTED,fontsize=8.1,weight="bold",ha="center")

    fig.text(.985,.025,"physical parameters first  ·  residual phase second  ·  compare the full propagation, not one camera plane",
             color="#7f8a94",fontsize=7.5,ha="right")

    png=out/"poster_hybrid_model_assisted_correction.png";pdf=out/"poster_hybrid_model_assisted_correction.pdf"
    fig.savefig(png,dpi=500,bbox_inches="tight",facecolor=BG)
    fig.savefig(pdf,bbox_inches="tight",facecolor=BG);plt.close(fig)

    with Image.open(png) as im:
        preview=im.convert("RGB");preview.thumbnail((2500,1450),Image.Resampling.LANCZOS)
        prev=out/"poster_hybrid_model_assisted_correction.preview.jpg";preview.save(prev,quality=92,subsampling=0)
        size=list(im.size);dpi=list(im.info.get("dpi",(0,0)))
    manifest={"outcome":"POSTER-HYBRID-CORRECTION-V2","model_registry_total":len(reg),
              "physical_fit_validation":fit.as_dict(),
              "flow":["q20 measured intensity","physical digital-twin fit","residual phase","corrected/ideal XY and XZ","similarity versus z"],
              "scope":{"physical_fit_panel":"synthetic validation of the real registry-driven fitter","q20_panels":"tracked q20 experimental/retrieval and model-correction products"},
              "assets":{"png_500dpi":str(png),"pdf":str(pdf),"preview":str(prev)},"png_pixel_size":size,"png_dpi":dpi}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest,indent=2));return manifest


if __name__=="__main__":build()
