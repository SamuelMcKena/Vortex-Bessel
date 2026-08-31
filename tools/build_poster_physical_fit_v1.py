"""Poster evidence that the forward error simulations are part of the inverse fit.

This is a synthetic validation of the *physical fitting stage*, not a substitute
for the experimental q=20 residual-phase result.  The same SystemErrorConfig and
``system_sweep_registry`` used by the forward error studies are replayed against a
noisy multi-plane V1 intensity stack.  Accepted perturbations are accumulated in
the digital twin; the remaining mismatch is what the experimental pipeline hands
to residual-phase retrieval.

For CI/runtime the demonstrator ranks eight representative, physically distinct
families spanning beam, SLM, 4F and axicon planes.  The fitting API itself accepts
all 41 declared registry families (``families=None``).
"""
from __future__ import annotations

from pathlib import Path
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image

from vbb_study.digital_twin.hierarchical_physical_fit import (
    apply_registry_family,
    hierarchical_physical_fit,
    registry_family_groups,
)
from vbb_study.digital_twin.physical_error_inference import plane_normalise_stack
from vbb_study.digital_twin.vortex_continuous_propagation import (
    build_fixed_support_spectrum,
    native_field_at_z,
)
from vbb_study.digital_twin.vortex_system_error_sweeps import system_sweep_registry
from vbb_study.digital_twin.vortex_system_route import SystemErrorConfig, build_system_route

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "poster" / "physical_fit_v1"
BG = "#070a0d"
AXBG = "#0b1015"
FG = "#f5f6f7"
MUTED = "#aab4be"
CYAN = "#4dd9d5"
GOLD = "#ffbd4a"
BORDER = "#44515d"
EPS = np.finfo(float).tiny

THERMAL = LinearSegmentedColormap.from_list(
    "poster_thermal_fit",
    [(0.00,"#000000"),(0.12,"#110000"),(0.35,"#650000"),
     (0.58,"#cf2700"),(0.78,"#ff7b00"),(0.92,"#ffd229"),(1.00,"#fff6bd")],
    N=256,
)

Z_M = np.linspace(25e-3, 85e-3, 9)
GRID_N = 288
CROP_HALF_M = 1.20e-3
NOISE_SIGMA = 0.0035

# Representative fit bank for the real optical smoke benchmark.  The underlying
# fitter accepts all 41 registry families; these eight give broad plane coverage
# without making every poster CI run hundreds of complete 4F propagations.
FIT_FAMILIES = (
    "beam_lateral_decentre_x",
    "beam_radius_scale",
    "slm1_hologram_offset_x",
    "fourf_iris_offset_x",
    "fourf_iris_radius_scale",
    "fourf_lens1_despace",
    "axicon_lateral_decentre_x",
    "axicon_round_tip",
)

DISPLAY = {
    "beam_lateral_decentre_x": "beam decentre",
    "beam_radius_scale": "beam radius",
    "slm1_hologram_offset_x": "SLM1 registration",
    "fourf_iris_offset_x": "4F iris offset",
    "fourf_iris_radius_scale": "4F iris opening",
    "fourf_lens1_despace": "4F lens despace",
    "axicon_lateral_decentre_x": "axicon decentre",
    "axicon_round_tip": "rounded axicon tip",
}


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(AXBG)
    for s in ax.spines.values():
        s.set_color(BORDER); s.set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(FG)
    ax.grid(False)


def _format_value(family: str, value: float) -> str:
    if family in {"beam_lateral_decentre_x", "slm1_hologram_offset_x", "axicon_lateral_decentre_x"}:
        return f"{value*1e6:+.0f} µm"
    if family == "fourf_iris_offset_x": return f"{value*1e3:+.2f} mm"
    if family == "fourf_lens1_despace": return f"{value*1e3:+.0f} mm"
    if family == "axicon_round_tip": return f"{value*1e6:.0f} µm"
    return f"{value:.2f}×"


class OpticalStackSimulator:
    def __init__(self) -> None:
        self.cache: dict[SystemErrorConfig, np.ndarray] = {}
        self.axis_m: np.ndarray | None = None

    def __call__(self, config: SystemErrorConfig) -> np.ndarray:
        if config in self.cache:
            return self.cache[config]
        route = build_system_route("V1", grid_n=GRID_N, config=config)
        prop = build_fixed_support_spectrum(
            np.asarray(route["post_axicon"], np.complex128),
            dict(route["grid"]),
            wavelength_m=float(route["metadata"]["wavelength_m"]),
            z_max_m=float(Z_M[-1]),
            minimum_retained_spectral_power=0.995,
        )
        x = np.asarray(route["grid"]["x"], float)
        ids = np.flatnonzero(np.abs(x) <= CROP_HALF_M)
        if ids.size < 24:
            raise RuntimeError("poster fit crop is too small for the simulation grid")
        self.axis_m = x[ids]
        planes = []
        for z in Z_M:
            field = native_field_at_z(prop, float(z))
            intensity = np.abs(np.asarray(field, np.complex128))**2
            planes.append(intensity[np.ix_(ids, ids)])
        stack = np.asarray(planes, float)
        self.cache[config] = stack
        return stack


def _xz(stack: np.ndarray) -> np.ndarray:
    a = plane_normalise_stack(stack)
    iy = a.shape[1]//2
    # transpose so transverse coordinate is vertical and z horizontal
    return np.asarray(a[:, iy, :], float).T


def _ranking_bar(ax: plt.Axes, step, *, title: str, topn: int = 6) -> None:
    _style(ax)
    ranking = list(step.rankings)[:topn]
    names = [DISPLAY.get(r.family, r.family) for r in ranking][::-1]
    improvements = [100.0*r.improvement_fraction for r in ranking][::-1]
    bars = ax.barh(np.arange(len(names)), improvements, color=[CYAN if r.family == step.selected_family else "#65717c" for r in ranking][::-1])
    ax.set_yticks(np.arange(len(names)), names, fontsize=7.2)
    ax.set_xlabel("loss reduction from current model (%)", fontsize=7.2)
    ax.set_title(title, fontsize=9.5, weight="bold", pad=6)
    ax.axvline(0, color="#65717c", lw=.6)
    for bar, val in zip(bars, improvements):
        ax.text(max(val,0)+0.35, bar.get_y()+bar.get_height()/2, f"{val:.1f}",
                color=MUTED, fontsize=6.8, va="center")


def _imshow_xz(ax: plt.Axes, arr: np.ndarray, axis_m: np.ndarray, title: str, *, power=.48) -> None:
    _style(ax)
    extent=[Z_M[0]*1e3,Z_M[-1]*1e3,axis_m[0]*1e3,axis_m[-1]*1e3]
    ax.imshow(np.maximum(arr,0)**power, origin="lower", aspect="auto", extent=extent,
              cmap=THERMAL, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=10.0, weight="bold", pad=6)
    ax.set_xlabel("z from axicon (mm)", fontsize=7.5)
    ax.set_ylabel("x at y≈0 (mm)", fontsize=7.5)


def _library_strip(ax: plt.Axes, groups: dict[str, tuple[str,...]]) -> None:
    ax.set_facecolor(AXBG); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    total=sum(len(v) for v in groups.values())
    ax.text(.03,.86,f"{total} parameterised error families",color=FG,fontsize=10.5,
            weight="bold",ha="left",va="center")
    labels=(("beam","Beam"),("slm","SLM"),("4f","4F relay"),("axicon","Axicon"))
    xs=(.14,.38,.63,.87)
    for (key,label),x in zip(labels,xs):
        n=len(groups.get(key,()))
        ax.text(x,.54,str(n),color=CYAN,fontsize=18,weight="bold",ha="center",va="center")
        ax.text(x,.30,label,color=MUTED,fontsize=8.0,weight="bold",ha="center",va="center")
    ax.text(.03,.08,"same SystemErrorConfig + optical route as the forward simulations",
            color="#7f8a94",fontsize=7.2,ha="left")


def build(out: Path=OUT) -> dict:
    out.mkdir(parents=True,exist_ok=True)
    reg=system_sweep_registry(); groups=registry_family_groups()
    simulator=OpticalStackSimulator()

    truth=SystemErrorConfig()
    truth=apply_registry_family(truth,"beam_radius_scale",0.85,registry=reg)
    truth=apply_registry_family(truth,"axicon_lateral_decentre_x",250e-6,registry=reg)
    clean_target=simulator(truth)
    rng=np.random.default_rng(7261)
    target=np.maximum(clean_target + rng.normal(size=clean_target.shape)*NOISE_SIGMA*np.max(clean_target,axis=(1,2))[:,None,None],0.0)

    fit=hierarchical_physical_fit(
        target_stack=target,
        simulate_config=simulator,
        families=FIT_FAMILIES,
        registry=reg,
        max_stages=2,
        min_improvement_fraction=0.003,
    )
    nominal=simulator(SystemErrorConfig())
    fitted=simulator(fit.final_config)
    if simulator.axis_m is None: raise RuntimeError("simulator did not expose crop axis")
    axis=simulator.axis_m

    # Figure: the simulation library is visibly upstream of the fit, and the
    # ranked candidate families are real replayed forward models rather than icons.
    fig=plt.figure(figsize=(16.8,8.25),facecolor=BG)
    gs=fig.add_gridspec(3,5,
        width_ratios=[1.10,.92,.92,1.08,1.08],
        height_ratios=[.45,1.0,1.0],
        left=.045,right=.985,bottom=.075,top=.79,wspace=.18,hspace=.30)

    axlib=fig.add_subplot(gs[0,0:3]); _library_strip(axlib,groups)
    axtruth=fig.add_subplot(gs[1:,0]); _imshow_xz(axtruth,_xz(target),axis,"Distorted z-stack")
    axnom=fig.add_subplot(gs[1:,1]); _imshow_xz(axnom,_xz(nominal),axis,"Nominal model")
    axr1=fig.add_subplot(gs[1,2]); _ranking_bar(axr1,fit.steps[0],title="Fit stage 1")
    axr2=fig.add_subplot(gs[2,2]); _ranking_bar(axr2,fit.steps[1],title="Fit stage 2")
    axfit=fig.add_subplot(gs[1:,3]); _imshow_xz(axfit,_xz(fitted),axis,"Fitted physical model")

    # Residual is plotted in normalized intensity units, not phase.
    residual=np.abs(plane_normalise_stack(target)-plane_normalise_stack(fitted))
    rxz=_xz(residual)
    axres=fig.add_subplot(gs[1:,4]); _style(axres)
    extent=[Z_M[0]*1e3,Z_M[-1]*1e3,axis[0]*1e3,axis[-1]*1e3]
    im=axres.imshow(rxz,origin="lower",aspect="auto",extent=extent,cmap="magma",
                    vmin=0,vmax=max(float(np.percentile(rxz,99)),1e-4),interpolation="nearest")
    axres.set_title("Remaining intensity mismatch",fontsize=10.0,weight="bold",pad=6)
    axres.set_xlabel("z from axicon (mm)",fontsize=7.5); axres.set_ylabel("x at y≈0 (mm)",fontsize=7.5)
    cb=fig.colorbar(im,ax=axres,fraction=.045,pad=.025); cb.ax.tick_params(labelsize=6.5,colors=MUTED)
    cb.outline.set_edgecolor(BORDER); cb.set_label("|ΔI|, plane-normalized",color=MUTED,fontsize=6.8)

    # Normal academic copy: no report/withhold/gate cards.
    fig.text(.045,.945,"Fitting physical system errors from the measured z-stack",color=FG,
             fontsize=21.5,weight="bold",ha="left")
    fig.text(.045,.892,
             "The forward beam / SLM / 4F / axicon simulations are replayed inside the inverse fit before residual phase retrieval.",
             color=MUTED,fontsize=9.5,ha="left")

    selected=[]
    for s in fit.steps:
        if s.accepted and s.selected_family is not None and s.selected_value is not None:
            selected.append(f"{DISPLAY.get(s.selected_family,s.selected_family)} {_format_value(s.selected_family,s.selected_value)}")
    fig.text(.64,.837," + ".join(selected),color=CYAN,fontsize=8.4,weight="bold",ha="center")
    fig.text(.64,.810,f"fit loss {fit.initial_cost:.4f} → {fit.final_cost:.4f}",color=MUTED,fontsize=8.0,ha="center")
    fig.text(.965,.035,"remaining mismatch → residual phase retrieval → SLM correction",color=CYAN,
             fontsize=8.2,weight="bold",ha="right")

    png=out/"physical_model_fit_to_zstack.png"; pdf=out/"physical_model_fit_to_zstack.pdf"
    fig.savefig(png,dpi=500,bbox_inches="tight",facecolor=BG)
    fig.savefig(pdf,bbox_inches="tight",facecolor=BG); plt.close(fig)

    with Image.open(png) as imf:
        preview=imf.convert("RGB"); preview.thumbnail((2500,1500),Image.Resampling.LANCZOS)
        prev=out/"physical_model_fit_to_zstack.preview.jpg"; preview.save(prev,quality=92,subsampling=0)
        pixels=list(imf.size); dpi=list(imf.info.get("dpi",(0,0)))

    manifest={
        "outcome":"POSTER-PHYSICAL-FIT-V1",
        "registry_total_families":len(reg),
        "registry_groups":{k:len(v) for k,v in groups.items()},
        "benchmark_candidate_families":list(FIT_FAMILIES),
        "benchmark_truth":{"beam_radius_scale":0.85,"axicon_lateral_decentre_x_m":250e-6},
        "noise_sigma_fraction_of_plane_peak":NOISE_SIGMA,
        "z_planes":len(Z_M),
        "fit":fit.as_dict(),
        "selected_display":selected,
        "assets":{"png_500dpi":str(png),"pdf":str(pdf),"preview":str(prev)},
        "png_pixel_size":pixels,"png_dpi":dpi,
        "handoff":"remaining optical mismatch is the input to residual-phase retrieval; intensity residual itself is not a phase map",
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest,indent=2))
    return manifest


if __name__=="__main__": build()
