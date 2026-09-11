from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np

from tools.run_short_bessel_4mm_telescope_lab_candidate import galilean_telescope
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
    c = float(row["slm1_capture_fraction"])
    s = abs(f - 5.0) / 0.10 + abs(l - 500.0) / 15.0
    if c < 0.90:
        s += 25.0 * (0.90 - c)
    return float(s)


def _figure(rows: list[dict[str, object]], path: Path) -> None:
    mags = sorted({float(r["telescope_magnification"]) for r in rows})
    targets = sorted({float(r["programmed_target_fwhm_um"]) for r in rows})
    F=np.full((len(targets),len(mags)),np.nan); L=np.full_like(F,np.nan); C=np.full_like(F,np.nan)
    mi={v:i for i,v in enumerate(mags)}; ti={v:i for i,v in enumerate(targets)}
    for r in rows:
        i=ti[float(r["programmed_target_fwhm_um"])]; j=mi[float(r["telescope_magnification"])]
        F[i,j]=float(r["central_fwhm_um"]); L[i,j]=float(r["sample_halfmax_zone_length_um"]); C[i,j]=float(r["slm1_capture_fraction"])
    extent=(min(mags),max(mags),min(targets),max(targets))
    fig,axes=plt.subplots(1,3,figsize=(14.5,4.5))
    for ax,data,title,label in [(axes[0],F,"B0 FWHM","um"),(axes[1],L,"B0 L50","um"),(axes[2],C,"SLM1 capture","fraction")]:
        im=ax.imshow(data,extent=extent,origin="lower",aspect="auto",cmap="viridis"); fig.colorbar(im,ax=ax,label=label)
        ax.set_xlabel("Galilean magnification"); ax.set_ylabel("programmed target FWHM (um)"); ax.set_title(title)
    axes[0].contour(mags,targets,F,levels=[5.0],linewidths=1.4)
    axes[1].contour(mags,targets,L,levels=[500.0],linewidths=1.4)
    fig.suptitle("High-resolution 4 mm input telescope verification",fontsize=14)
    fig.tight_layout(rect=(0,0,1,.93)); fig.savefig(path,dpi=200); plt.close(fig)


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
    fig.suptitle("High-resolution 4 mm-input telescope candidate",fontsize=14); fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(path,dpi=200); plt.close(fig)


def main() -> None:
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-dir",type=Path,default=Path("outputs/short_bessel_4mm_telescope_highres"))
    ap.add_argument("--grid-n",type=int,default=768)
    ap.add_argument("--physical-fwhm-um",type=float,default=3.0)
    args=ap.parse_args(); out=args.output_dir; out.mkdir(parents=True,exist_ok=True)

    f1=-50.0
    f2s=(90.0,95.0,100.0,105.0,105.5)
    telescopes={f2:galilean_telescope(input_diameter_1e2_mm=4.0,wavelength_nm=1029.0,f1_mm=f1,f2_mm=f2,lens_clear_radius_mm=10.0) for f2 in f2s}
    targets=(5.1,5.2,5.3,5.4,5.5)
    rows=[]
    for f2,tel in telescopes.items():
        for target in targets:
            design=design_short_bessel(ShortBesselDesignInput(target_fwhm_um=target,target_length_um=500.0))
            cfg=BenchCandidateConfig(grid_n=int(args.grid_n),input_pre_radius_mm=tel.output_radius_1e_field_mm,pinhole_radius_mm=0.50,measured_physical_only_fwhm_um=float(args.physical_fwhm_um),lens_clear_radius_mm=7.0)
            run=run_selected_order_frame(design,ell=0,config=cfg,sample_z_max_um=850.0,sample_z_points=171)
            row=run.metrics.as_dict(); row.update({"telescope_f1_mm":f1,"telescope_f2_mm":f2,"telescope_magnification":tel.paraxial_magnification,"telescope_separation_mm":tel.separation_mm,"telescope_output_diameter_mm":tel.output_diameter_1e2_mm,"telescope_output_wavefront_radius_m":tel.output_wavefront_radius_m,"programmed_target_fwhm_um":target}); row["score"]=_score(row); rows.append(row)
    _write_csv(out/"highres_telescope_target_sweep.csv",rows); _figure(rows,out/"01_highres_telescope_target_sweep.png")
    best=min(rows,key=lambda r:float(r["score"])); f2=float(best["telescope_f2_mm"]); target=float(best["programmed_target_fwhm_um"]); tel=telescopes[f2]
    design=design_short_bessel(ShortBesselDesignInput(target_fwhm_um=target,target_length_um=500.0))

    filter_rows=[]; filter_runs={}
    for pr in (0.35,0.50,0.65,0.80):
        runs={}
        for ell in (0,1,3):
            cfg=BenchCandidateConfig(grid_n=int(args.grid_n),input_pre_radius_mm=tel.output_radius_1e_field_mm,pinhole_radius_mm=pr,measured_physical_only_fwhm_um=float(args.physical_fwhm_um),lens_clear_radius_mm=7.0)
            runs[ell]=run_selected_order_frame(design,ell=ell,config=cfg,sample_z_max_um=850.0,sample_z_points=171)
        b0,v1,v3=runs[0].metrics,runs[1].metrics,runs[3].metrics
        score=abs(b0.central_fwhm_um-5.0)/.10+abs(b0.sample_halfmax_zone_length_um-500.0)/15.0+abs(abs(v1.measured_phase_winding)-1.0)/.25+abs(abs(v3.measured_phase_winding)-3.0)/.25
        filter_rows.append({"pinhole_radius_mm":pr,"score":float(score),"B0_FWHM_um":b0.central_fwhm_um,"B0_L50_um":b0.sample_halfmax_zone_length_um,"V1_ring_um":v1.measured_ring_diameter_um,"V1_winding":v1.measured_phase_winding,"V3_ring_um":v3.measured_ring_diameter_um,"V3_winding":v3.measured_phase_winding}); filter_runs[pr]=runs
    _write_csv(out/"highres_filter_sweep.csv",filter_rows)
    chosen=min(filter_rows,key=lambda r:float(r["score"])); pr=float(chosen["pinhole_radius_mm"]); runs=filter_runs[pr]
    _final_figure(runs,out/"02_highres_final_B0_V1_V3.png")

    report={"input":{"diameter_1e2_mm":4.0,"wavelength_nm":1029.0},"selected_telescope":tel.as_dict(),"selected_programmed_target_fwhm_um":target,"selected_pinhole_radius_mm":pr,"highres_best":best,"final":{str(e):runs[e].metrics.as_dict() for e in (0,1,3)},"filter_sweep":filter_rows,"claim_boundary":runs[0].metrics.claim_boundary}
    (out/"highres_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8"); print(json.dumps(report,indent=2))

if __name__=="__main__": main()
