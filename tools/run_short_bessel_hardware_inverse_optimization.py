from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np

from vbb_study.short_bessel import ShortBesselDesignInput, design_short_bessel
from vbb_study.short_bessel_bench_candidate import BenchCandidateConfig
from vbb_study.short_bessel_selected_order_frame import run_selected_order_frame


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def _score(row: dict[str, object]) -> float:
    f = float(row["central_fwhm_um"])
    l = float(row["sample_halfmax_zone_length_um"])
    capture = min(float(row["slm1_capture_fraction"]), float(row["slm2_capture_fraction"]))
    r = float(row["input_pre_radius_mm"])
    score = abs(f - 5.0) / 0.15 + abs(l - 500.0) / 15.0
    if capture < 0.90:
        score += 20.0 * (0.90 - capture)
    # Very weak preference for less clipping / smaller illuminated radius when
    # optical performance is otherwise essentially tied.
    score += 0.03 * max(r - 4.32, 0.0) / 0.1
    return float(score)


def _sweep_figure(rows: list[dict[str, object]], path: Path) -> None:
    programmed = sorted({float(r["programmed_target_fwhm_um"]) for r in rows})
    radii = sorted({float(r["input_pre_radius_mm"]) for r in rows})
    F = np.full((len(programmed), len(radii)), np.nan)
    L = np.full_like(F, np.nan)
    C = np.full_like(F, np.nan)
    S = np.full_like(F, np.nan)
    pi = {v:i for i,v in enumerate(programmed)}; ri={v:i for i,v in enumerate(radii)}
    for row in rows:
        i=pi[float(row["programmed_target_fwhm_um"])]; j=ri[float(row["input_pre_radius_mm"])]
        F[i,j]=float(row["central_fwhm_um"]); L[i,j]=float(row["sample_halfmax_zone_length_um"])
        C[i,j]=float(row["slm1_capture_fraction"]); S[i,j]=float(row["score"])
    extent=(min(radii),max(radii),min(programmed),max(programmed))
    fig,axes=plt.subplots(2,2,figsize=(13.5,9.5))
    for ax,data,title,cblabel in [
        (axes[0,0],F,"Propagated B0 FWHM","FWHM (um)"),
        (axes[0,1],L,"Propagated B0 L50","L50 (um)"),
        (axes[1,0],C,"SLM1 Gaussian capture","power fraction inside panel"),
        (axes[1,1],S,"Inverse-design score","lower is better"),
    ]:
        im=ax.imshow(data,extent=extent,origin="lower",aspect="auto",cmap="viridis")
        fig.colorbar(im,ax=ax,label=cblabel)
        ax.set_xlabel("input 1/e field radius on SLM (mm)")
        ax.set_ylabel("programmed equivalent B0 FWHM target (um)")
        ax.set_title(title)
    axes[0,0].contour(radii,programmed,F,levels=[5.0],linewidths=1.5)
    axes[0,1].contour(radii,programmed,L,levels=[500.0],linewidths=1.5)
    fig.suptitle("Hardware-aware inverse sweep | rectangular SLM + selected-order 4F frame | pinhole r=0.50 mm",fontsize=14)
    fig.tight_layout(rect=(0,0,1,0.94)); fig.savefig(path,dpi=200); plt.close(fig)


def _convergence_figure(rows: list[dict[str, object]], path: Path) -> None:
    n=np.array([int(r["grid_n"]) for r in rows]); f=np.array([float(r["central_fwhm_um"]) for r in rows]); l=np.array([float(r["sample_halfmax_zone_length_um"]) for r in rows])
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2))
    axes[0].plot(n,f,marker="o"); axes[0].axhline(5.0,linestyle="--",linewidth=1); axes[0].set_ylabel("B0 FWHM (um)")
    axes[1].plot(n,l,marker="o"); axes[1].axhline(500.0,linestyle="--",linewidth=1); axes[1].set_ylabel("B0 L50 (um)")
    for ax in axes: ax.set_xlabel("square numerical grid N"); ax.grid(True,alpha=.25)
    fig.suptitle("Numerical convergence of the hardware-aware candidate",fontsize=14); fig.tight_layout(rect=(0,0,1,.91)); fig.savefig(path,dpi=200); plt.close(fig)


def _final_figure(runs: dict[int, object], path: Path) -> None:
    def sn(a):
        a=np.maximum(np.asarray(a,float),0); m=float(np.max(a)); return np.sqrt(a/m) if m>0 else np.zeros_like(a)
    def nn(a):
        a=np.maximum(np.asarray(a,float),0); m=float(np.max(a)); return a/m if m>0 else np.zeros_like(a)
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    for j,ell in enumerate((0,1,3)):
        r=runs[ell]; m=r.metrics; mask=np.abs(r.sample_x_um)<=30
        axes[0,j].imshow(sn(r.sample_xz_intensity[:,mask]),extent=(r.sample_x_um[mask][0],r.sample_x_um[mask][-1],r.sample_z_um[0],r.sample_z_um[-1]),origin="lower",aspect="auto",cmap="inferno",vmin=0,vmax=1)
        axes[0,j].axhline(m.sample_halfmax_zone_start_um,linestyle="--",linewidth=1); axes[0,j].axhline(m.sample_halfmax_zone_end_um,linestyle="--",linewidth=1)
        axes[0,j].set_title(f"ell={ell} | L50={m.sample_halfmax_zone_length_um:.0f} um"); axes[0,j].set_xlabel("x (um)"); axes[0,j].set_ylabel("z (um)")
        xm=np.abs(r.sample_x_um)<=30; xx=r.sample_x_um[xm]; xy=nn(r.sample_peak_xy_intensity)[np.ix_(xm,xm)]
        axes[1,j].imshow(xy,extent=(xx[0],xx[-1],xx[0],xx[-1]),origin="lower",cmap="inferno",vmin=0,vmax=1)
        size=m.central_fwhm_um if ell==0 else m.measured_ring_diameter_um
        axes[1,j].set_title(f"peak XY | size={size:.2f} um | |w|={abs(m.measured_phase_winding):.2f}"); axes[1,j].set_xlabel("x (um)"); axes[1,j].set_ylabel("y (um)"); axes[1,j].set_aspect("equal")
    fig.suptitle("Hardware-inverse-designed B0 / V1 / V3 candidate",fontsize=14); fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(path,dpi=200); plt.close(fig)


def main() -> None:
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-dir",type=Path,default=Path("outputs/short_bessel_hardware_inverse"))
    ap.add_argument("--physical-fwhm-um",type=float,default=3.0)
    ap.add_argument("--final-grid-n",type=int,default=768)
    args=ap.parse_args(); out=args.output_dir; out.mkdir(parents=True,exist_ok=True)

    programmed_targets=(4.8,5.0,5.2,5.4,5.6)
    beam_radii=(3.8,4.1,4.3,4.5,4.7,4.9,5.2,5.5)
    sweep=[]
    for ptarget in programmed_targets:
        design=design_short_bessel(ShortBesselDesignInput(target_fwhm_um=ptarget,target_length_um=500.0))
        for radius in beam_radii:
            cfg=BenchCandidateConfig(grid_n=384,input_pre_radius_mm=radius,pinhole_radius_mm=0.50,measured_physical_only_fwhm_um=float(args.physical_fwhm_um),lens_clear_radius_mm=7.0)
            run=run_selected_order_frame(design,ell=0,config=cfg,sample_z_max_um=850.0,sample_z_points=111)
            row=run.metrics.as_dict(); row["programmed_target_fwhm_um"]=ptarget; row["score"]=_score(row); sweep.append(row)
    _write_csv(out/"hardware_inverse_sweep.csv",sweep); _sweep_figure(sweep,out/"01_hardware_inverse_sweep.png")
    best=min(sweep,key=lambda r:float(r["score"])); best_target=float(best["programmed_target_fwhm_um"]); best_radius=float(best["input_pre_radius_mm"])

    # Resolution convergence at the chosen candidate.
    convergence=[]
    design=design_short_bessel(ShortBesselDesignInput(target_fwhm_um=best_target,target_length_um=500.0))
    for n in (384,512,768,1024):
        cfg=BenchCandidateConfig(grid_n=n,input_pre_radius_mm=best_radius,pinhole_radius_mm=0.50,measured_physical_only_fwhm_um=float(args.physical_fwhm_um),lens_clear_radius_mm=7.0)
        run=run_selected_order_frame(design,ell=0,config=cfg,sample_z_max_um=850.0,sample_z_points=141)
        row=run.metrics.as_dict(); row["grid_n"]=n; convergence.append(row)
    _write_csv(out/"hardware_candidate_convergence.csv",convergence); _convergence_figure(convergence,out/"02_hardware_candidate_convergence.png")

    # Check a few filter radii at high resolution and keep the best B0/V1/V3 compromise.
    filter_candidates=[]; filter_runs={}
    for pr in (0.35,0.50,0.65,0.80):
        group=[]; runs={}
        for ell in (0,1,3):
            cfg=BenchCandidateConfig(grid_n=int(args.final_grid_n),input_pre_radius_mm=best_radius,pinhole_radius_mm=pr,measured_physical_only_fwhm_um=float(args.physical_fwhm_um),lens_clear_radius_mm=7.0)
            r=run_selected_order_frame(design,ell=ell,config=cfg,sample_z_max_um=850.0,sample_z_points=171); runs[ell]=r; group.append(r.metrics.as_dict())
        b0,v1,v3=runs[0].metrics,runs[1].metrics,runs[3].metrics
        score=abs(b0.central_fwhm_um-5.0)/0.15+abs(b0.sample_halfmax_zone_length_um-500.0)/15.0+abs(abs(v1.measured_phase_winding)-1.0)/.25+abs(abs(v3.measured_phase_winding)-3.0)/.25
        filter_candidates.append({"pinhole_radius_mm":pr,"score":float(score),"B0_fwhm_um":b0.central_fwhm_um,"B0_L50_um":b0.sample_halfmax_zone_length_um,"V1_ring_um":v1.measured_ring_diameter_um,"V1_winding":v1.measured_phase_winding,"V3_ring_um":v3.measured_ring_diameter_um,"V3_winding":v3.measured_phase_winding})
        filter_runs[pr]=runs
    _write_csv(out/"final_filter_candidates.csv",filter_candidates)
    chosen=min(filter_candidates,key=lambda r:float(r["score"])); chosen_pr=float(chosen["pinhole_radius_mm"]); final_runs=filter_runs[chosen_pr]
    _final_figure(final_runs,out/"03_hardware_inverse_final_B0_V1_V3.png")

    report={
        "requested_output":{"B0_FWHM_um":5.0,"B0_L50_um":500.0},
        "coarse_best":{"programmed_target_fwhm_um":best_target,"input_pre_radius_mm":best_radius,"score":float(best["score"]),"metrics":best},
        "selected_pinhole_radius_mm":chosen_pr,
        "final_high_resolution":{str(e):final_runs[e].metrics.as_dict() for e in (0,1,3)},
        "convergence":convergence,
        "filter_candidates":filter_candidates,
        "claim_boundary":final_runs[0].metrics.claim_boundary,
    }
    (out/"hardware_inverse_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
