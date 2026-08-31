"""Integrated poster figure: digital-twin fitting -> residual phase -> correction.

The purpose of this renderer is to make the algorithmic connection visible.
The q=20 camera/retrieval products remain the experimental correction example;
a compact synthetic validation beside them demonstrates that the *same* forward
beam/SLM/4F/axicon error registry is genuinely searched by the physical-fitting
stage.  The two are not silently conflated.

Visual flow:
    measured intensity evolution
      -> physical-system fit using the forward simulation library
      -> residual phase retrieval
      -> corrected vs ideal transverse / longitudinal model response
      -> quantitative similarity versus z
"""
from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image, ImageChops, ImageOps

import build_poster_physical_fit_v1 as pf
import build_poster_q20_inverse_v6 as q6
from vbb_study.digital_twin.hierarchical_physical_fit import (
    apply_registry_family,
    hierarchical_physical_fit,
    registry_family_groups,
)
from vbb_study.digital_twin.vortex_system_error_sweeps import system_sweep_registry
from vbb_study.digital_twin.vortex_system_route import SystemErrorConfig

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"/"poster"/"hybrid_correction_v1"
SIMILARITY=ROOT/"figures/experimental/q20_aberration/validation/realigned_cartesian_similarity_vs_z.png"
BG="#070a0d"; CARD="#0b1015"; FG="#f5f6f7"; MUTED="#aab4be"; CYAN="#4dd9d5"; BORDER="#44515d"; GOLD="#ffbd4a"


def _trim_white(im:Image.Image,threshold=16,pad=6)->Image.Image:
    rgb=im.convert("RGB")
    diff=ImageChops.difference(rgb,Image.new("RGB",rgb.size,"white")).convert("L")
    diff=diff.point(lambda p:255 if p>threshold else 0)
    box=diff.getbbox()
    if box is None:return rgb
    l,t,r,b=box
    return rgb.crop((max(0,l-pad),max(0,t-pad),min(rgb.width,r+pad),min(rgb.height,b+pad)))


def _image(ax,im:Image.Image,*,square=False,title=None):
    ax.set_facecolor(CARD)
    for s in ax.spines.values(): s.set_color(BORDER); s.set_linewidth(.8)
    ax.imshow(np.asarray(im),interpolation="nearest",aspect="equal" if square else "auto")
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title,color=FG,fontsize=9.2,weight="bold",pad=5)


def _fit_validation():
    reg=system_sweep_registry(); simulator=pf.OpticalStackSimulator()
    truth=SystemErrorConfig()
    truth=apply_registry_family(truth,"beam_radius_scale",0.85,registry=reg)
    truth=apply_registry_family(truth,"axicon_lateral_decentre_x",250e-6,registry=reg)
    clean=simulator(truth)
    rng=np.random.default_rng(7261)
    target=np.maximum(clean+rng.normal(size=clean.shape)*pf.NOISE_SIGMA*np.max(clean,axis=(1,2))[:,None,None],0.0)
    fit=hierarchical_physical_fit(target_stack=target,simulate_config=simulator,
                                  families=pf.FIT_FAMILIES,registry=reg,max_stages=2,
                                  min_improvement_fraction=.003)
    fitted=simulator(fit.final_config)
    if simulator.axis_m is None: raise RuntimeError("fit simulator axis missing")
    return target,fitted,simulator.axis_m,fit,reg


def _fit_card(ax,target,fitted,axis,fit,reg):
    ax.set_facecolor(CARD); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    for s in ax.spines.values():s.set_visible(False)
    groups=registry_family_groups(); total=len(reg)
    ax.add_patch(FancyBboxPatch((.01,.01),.98,.98,boxstyle="round,pad=.012",
                                transform=ax.transAxes,fc=CARD,ec=BORDER,lw=.9))
    ax.text(.05,.94,"Physical-system fit",color=FG,fontsize=11.0,weight="bold",ha="left",va="top")
    ax.text(.05,.885,f"{total} simulated error families",color=CYAN,fontsize=8.0,weight="bold",ha="left")
    ax.text(.05,.84,"7 beam  ·  11 SLM  ·  15 4F  ·  8 axicon",color=MUTED,fontsize=7.2,ha="left")

    # Real replayed optical benchmark, displayed compactly.
    xz_t=pf._xz(target); xz_f=pf._xz(fitted)
    extent=[pf.Z_M[0]*1e3,pf.Z_M[-1]*1e3,axis[0]*1e3,axis[-1]*1e3]
    a1=ax.inset_axes([.05,.48,.40,.28]); a2=ax.inset_axes([.55,.48,.40,.28])
    for a,arr,title in ((a1,xz_t,"input"),(a2,xz_f,"fit")):
        a.imshow(np.maximum(arr,0)**.48,origin="lower",aspect="auto",extent=extent,
                 cmap=pf.THERMAL,vmin=0,vmax=1,interpolation="nearest")
        a.set_title(title,color=FG,fontsize=7.2,weight="bold",pad=2)
        a.set_xticks([]);a.set_yticks([])
        for s in a.spines.values():s.set_color(BORDER);s.set_linewidth(.55)

    y=.375
    for step in fit.steps:
        if step.accepted and step.selected_family and step.selected_value is not None:
            fam=pf.DISPLAY.get(step.selected_family,step.selected_family)
            val=pf._format_value(step.selected_family,step.selected_value)
            ax.text(.07,y,f"{step.stage}",color=BG,fontsize=7.0,weight="bold",ha="center",va="center",
                    bbox=dict(boxstyle="circle,pad=.26",fc=CYAN,ec="none"))
            ax.text(.12,y,f"{fam}: {val}",color=FG,fontsize=7.7,weight="bold",ha="left",va="center")
            ax.text(.93,y,f"−{100*step.improvement_fraction:.1f}% loss",color=CYAN,fontsize=7.0,ha="right",va="center")
            y-=.075
    ax.text(.05,.155,f"stack loss  {fit.initial_cost:.4f}  →  {fit.final_cost:.4f}",color=MUTED,fontsize=7.6,ha="left")
    ax.text(.05,.085,"synthetic two-error validation",color="#7f8a94",fontsize=6.7,ha="left")


def _phase_card(ax,phase:Image.Image):
    _image(ax,phase,title="Residual phase")
    ax.text(.5,-.06,r"$\phi_{\rm corr}=-\psi_{\rm res}$",transform=ax.transAxes,
            color=CYAN,fontsize=10.0,weight="bold",ha="center",va="top")


def _arrow(fig,left,right,y=.53):
    a,b=left.get_position(),right.get_position(); yy=a.y0+y*(a.y1-a.y0)
    fig.add_artist(FancyArrowPatch((a.x1+.004,yy),(b.x0-.004,yy),transform=fig.transFigure,
                                   arrowstyle="-|>",mutation_scale=14,lw=1.25,color=CYAN))


def build(out:Path=OUT):
    out.mkdir(parents=True,exist_ok=True)
    q=q6._clean()
    target,fitted,axis,fit,reg=_fit_validation()
    if not SIMILARITY.exists(): raise FileNotFoundError(SIMILARITY)
    similarity=_trim_white(Image.open(SIMILARITY).convert("RGB"),pad=4)

    fig=plt.figure(figsize=(17.2,9.15),facecolor=BG)
    gs=fig.add_gridspec(3,5,width_ratios=[1.05,1.05,.88,1.0,1.0],
                        height_ratios=[.72,1.0,.78],left=.035,right=.987,
                        bottom=.065,top=.81,wspace=.12,hspace=.21)

    # Measurement column: actual q=20 tracked camera/model-analysis product.
    ax_mxy=fig.add_subplot(gs[0,0]); _image(ax_mxy,q["xy_measured"],square=True,title="Measured transverse field")
    ax_mxz=fig.add_subplot(gs[1:,0]); _image(ax_mxz,q["xz_measured"],title="Measured propagation")

    # Physical fit: actual forward-model replay benchmark, not icons.
    ax_fit=fig.add_subplot(gs[:,1]); _fit_card(ax_fit,target,fitted,axis,fit,reg)

    # Existing q=20 residual retrieval.
    ax_phase=fig.add_subplot(gs[:,2]); _phase_card(ax_phase,q["phase"])

    # Model-predicted correction and ideal reference.
    ax_cxy=fig.add_subplot(gs[0,3]); _image(ax_cxy,q["xy_corrected"],square=True,title="Corrected")
    ax_ixy=fig.add_subplot(gs[0,4]); _image(ax_ixy,q["xy_ideal"],square=True,title="Ideal")
    ax_cxz=fig.add_subplot(gs[1,3]); _image(ax_cxz,q["xz_corrected"],title="Corrected propagation")
    ax_ixz=fig.add_subplot(gs[1,4]); _image(ax_ixz,q["xz_ideal"],title="Ideal propagation")
    ax_metric=fig.add_subplot(gs[2,3:5]); _image(ax_metric,similarity,title="Agreement across the z-stack")

    _arrow(fig,ax_mxy,ax_fit,.50); _arrow(fig,ax_fit,ax_phase,.50); _arrow(fig,ax_phase,ax_cxy,.50)

    fig.text(.035,.945,"Model-assisted intensity-only correction",color=FG,fontsize=22.5,weight="bold",ha="left")
    fig.text(.035,.892,
             "Measured propagation → fit the physical optical system → retrieve the remaining wavefront error → apply the SLM correction",
             color=MUTED,fontsize=9.7,ha="left")

    for ax,text in ((ax_mxy,"MEASUREMENT"),(ax_fit,"DIGITAL TWIN"),(ax_phase,"RESIDUAL WAVEFRONT"),(ax_cxy,"CORRECTION")):
        p=ax.get_position(); fig.text((p.x0+p.x1)/2,.835,text,color=MUTED,fontsize=8.2,weight="bold",ha="center")

    fig.text(.985,.026,
             "The physical fit removes explainable beam / SLM / 4F / axicon errors; phase retrieval is reserved for the remaining aberration.",
             color="#7f8a94",fontsize=7.7,ha="right")

    png=out/"poster_hybrid_model_assisted_correction.png"; pdf=out/"poster_hybrid_model_assisted_correction.pdf"
    fig.savefig(png,dpi=500,bbox_inches="tight",facecolor=BG)
    fig.savefig(pdf,bbox_inches="tight",facecolor=BG);plt.close(fig)

    with Image.open(png) as im:
        preview=im.convert("RGB");preview.thumbnail((2500,1500),Image.Resampling.LANCZOS)
        prev=out/"poster_hybrid_model_assisted_correction.preview.jpg";preview.save(prev,quality=92,subsampling=0)
        size=list(im.size);dpi=list(im.info.get("dpi",(0,0)))
    manifest={
        "outcome":"POSTER-HYBRID-CORRECTION-V1",
        "model_registry_total":len(reg),
        "physical_fit_validation":fit.as_dict(),
        "flow":["q20 measured intensity","physical digital-twin fit","residual phase","SLM correction prediction","ideal comparison","similarity versus z"],
        "scope":{"physical_fit_panel":"synthetic validation of the real forward-model fitter","q20_panels":"tracked q20 experimental/retrieval and model-correction products"},
        "assets":{"png_500dpi":str(png),"pdf":str(pdf),"preview":str(prev)},
        "png_pixel_size":size,"png_dpi":dpi,
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest,indent=2));return manifest


if __name__=="__main__":build()
