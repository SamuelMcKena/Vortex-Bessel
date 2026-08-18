"""Presentation figures comparing continuous-ideal and nominal bench-constrained B0/V1/V3,
now including explicit absolute-difference panels for the transverse comparison.

This is a drop-in replacement / patch variant for tools/build_phase2l_nominal_constraints.py.
"""
from __future__ import annotations
import argparse, gc, json, math
from pathlib import Path
from typing import Any, Mapping
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import presentation_phase2j_style as style
from vbb_study.digital_twin.phase2a_canonical import _panel_from_manifest
from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.vortex_beam_slm_errors import GaussianBeamError, gaussian_input_field, quantise_commanded_phase
from vbb_study.digital_twin.vortex_continuous_propagation import build_fixed_plane_longitudinal_map, build_fixed_support_spectrum, native_field_at_z
from vbb_study.digital_twin.vortex_system_route import AxiconError, build_system_route, physical_axicon_on_own_plane
from vbb_study.equations.fields import make_xy_grid
from vbb_study.slm_model import pixelate

TWOPI=2*np.pi; Z_REF_M=60e-3; Z_VALUES_M=np.linspace(5e-3,120e-3,120); PROP_COORD_M=np.linspace(-0.18e-3,0.18e-3,601)
WINDOW_M=10e-3; XY_HALF_M=0.18e-3; CMAP=style.CMAP; TEXT=style.TEXT; MUTED=style.MUTED; BG=style.FIG_BG
PNORM=colors.PowerNorm(gamma=.55,vmin=0,vmax=1)

def _ell(c): return {'B0':0,'V1':1,'V3':3}[c]
def _norm(a):
    a=np.maximum(np.asarray(a,float),0); return a/max(float(np.max(a)),np.finfo(float).tiny)
def _crop(a,g,half=XY_HALF_M):
    x=np.asarray(g['x'],float); ids=np.flatnonzero(np.abs(x)<=half); return np.asarray(a)[np.ix_(ids,ids)],x[ids]
def _prop(post,g):
    m=canonical_hardware_manifest(); return build_fixed_support_spectrum(np.asarray(post,np.complex128),dict(g),wavelength_m=float(hardware_value(m,'wavelength_m')),z_max_m=float(Z_VALUES_M[-1]),minimum_retained_spectral_power=.995)
def _xy(route):
    p=_prop(route['post_axicon'],route['grid']); return np.abs(np.asarray(native_field_at_z(p,Z_REF_M)))**2
def _xz(route,label):
    p=_prop(route['post_axicon'],route['grid']); q=build_fixed_plane_longitudinal_map(p,z_values_m=Z_VALUES_M,x_coordinates_m=PROP_COORD_M,y_coordinates_m=PROP_COORD_M,fixed_x_m=0,fixed_y_m=0,source_label=label); return np.asarray(q.xz_intensity,float)

def continuous_ideal(cid,n):
    m=canonical_hardware_manifest(); wl=float(hardware_value(m,'wavelength_m')); w=float(hardware_value(m,'beam_radius_on_slm_m')); na=float(hardware_value(m,'axicon_refractive_index')); ne=float(hardware_value(m,'axicon_external_medium_index')); ga=math.radians(float(hardware_value(m,'axicon_base_angle_deg')))
    g=make_xy_grid(int(n),WINDOW_M/int(n)); b,_=gaussian_input_field(g,wavelength_m=wl,canonical_radius_m=w,error=GaussianBeamError()); ph=float(_ell(cid))*np.arctan2(np.asarray(g['Y']),np.asarray(g['X'])); at,_=physical_axicon_on_own_plane(g,wavelength_m=wl,base_angle_rad=ga,refractive_index=na,external_index=ne,error=AxiconError()); return {'grid':g,'post_axicon':b*np.exp(1j*ph)*at,'phase':ph}

def ideal_vs_nominal(out,n):
    cases=[('B0','B0 — ℓ=0'),('V1','V1 — ℓ=1'),('V3','V3 — ℓ=3')]
    fig,ax=plt.subplots(3,3,figsize=(13.4,12.0),facecolor=BG)
    fig.subplots_adjust(left=.06,right=.985,bottom=.08,top=.865,wspace=.10,hspace=.28)
    max_diff = 0.0
    prepared=[]
    for cid,t in cases:
        a=continuous_ideal(cid,n); b=build_system_route(cid,grid_n=n); ia=_xy(a); ib=_xy(b)
        ia_n=_norm(ia); ib_n=_norm(ib)
        diff=np.abs(ib_n-ia_n)
        max_diff=max(max_diff,float(np.max(diff)))
        prepared.append((cid,t,a,b,ia_n,ib_n,diff))
    max_diff=max(max_diff,1e-12)
    dnorm=colors.PowerNorm(gamma=.75,vmin=0,vmax=max_diff)
    diff_mappable=None

    for c,(_,t,a,b,ia_n,ib_n,diff) in enumerate(prepared):
        for r,(v,route,label,norm,cmap_) in enumerate([
            (ia_n,a,'continuous ideal',PNORM,CMAP),
            (ib_n,b,'nominal bench-constrained',PNORM,CMAP),
            (diff,b,'absolute difference',dnorm,CMAP),
        ]):
            q=ax[r,c]; style.style_ax(q); cr,x=_crop(v,route['grid'])
            im=q.imshow(cr,origin='lower',extent=[x[0]*1e3,x[-1]*1e3,x[0]*1e3,x[-1]*1e3],cmap=cmap_,norm=norm,interpolation=style.DISPLAY_INTERPOLATION,aspect='equal')
            if r<2:
                q.set_title(f'{t}\n{label}',color=TEXT,fontsize=11.2,weight='bold')
            else:
                q.set_title(f'{t}\n|I_nominal − I_ideal|',color=TEXT,fontsize=11.2,weight='bold')
                diff_mappable=im
            q.set_xlabel('x (mm)',fontsize=8)
            if c==0:q.set_ylabel('y (mm)',fontsize=8)
            else:q.tick_params(labelleft=False)
        del a,b,ia_n,ib_n,diff
        gc.collect()
    fig.suptitle('Ideal beam family → nominal experimental constraints',color=TEXT,fontsize=18,weight='bold',y=.972)
    fig.text(.5,.920,'SLM pixelation + 8-bit phase + fill-factor throughput + carrier/blaze + finite 4F order selection',ha='center',color=MUTED,fontsize=10.2)
    fig.text(.5,.890,'Third row isolates the morphology change caused by nominal hardware constraints only',ha='center',color=MUTED,fontsize=9.3)
    fig.text(.5,.867,'No deliberate misalignment, wavefront error, axicon defect or measured correction map',ha='center',color=MUTED,fontsize=9.1)
    if diff_mappable is not None:
        cax=fig.add_axes([0.25,0.045,0.50,0.015])
        cb=fig.colorbar(diff_mappable,cax=cax,orientation='horizontal')
        cb.ax.tick_params(colors=MUTED,labelsize=7,length=2)
        cb.outline.set_edgecolor((*colors.to_rgb(MUTED),0.28))
        cb.set_label('|I_nominal − I_ideal|',color=MUTED,fontsize=8,labelpad=2)
    p=out/'01_ideal_vs_nominal_constraints_B0_V1_V3.png'; fig.savefig(p,dpi=480,bbox_inches='tight',facecolor=BG,pad_inches=.06); plt.close(fig); return p

def v3_prop(out,n):
    a=continuous_ideal('V3',n); b=build_system_route('V3',grid_n=n); ia=_norm(_xz(a,'phase2l-ideal-v3')); ib=_norm(_xz(b,'phase2l-nominal-v3')); d=np.abs(ib-ia); z=Z_VALUES_M*1e3; x=PROP_COORD_M*1e3
    fig,ax=plt.subplots(1,3,figsize=(14.5,4.8),facecolor=BG); fig.subplots_adjust(left=.055,right=.985,bottom=.16,top=.78,wspace=.13)
    for q in ax:style.style_ax(q)
    for q,v,t in [(ax[0],ia,'Continuous ideal'),(ax[1],ib,'Nominal bench-constrained')]:
        q.imshow(v.T,origin='lower',extent=[z[0],z[-1],x[0],x[-1]],cmap=CMAP,norm=PNORM,interpolation=style.DISPLAY_INTERPOLATION,aspect='auto');q.set_title(t,color=TEXT,fontsize=12.5,weight='bold');q.set_xlabel('z from axicon (mm)');q.axvline(60,color='white',alpha=.25,ls='--',lw=.8)
    ax[0].set_ylabel('x at fixed y=0 (mm)'); im=ax[2].imshow(d.T,origin='lower',extent=[z[0],z[-1],x[0],x[-1]],cmap=CMAP,vmin=0,vmax=max(float(np.max(d)),1e-12),interpolation=style.DISPLAY_INTERPOLATION,aspect='auto'); ax[2].set_title('Absolute morphology difference',color=TEXT,fontsize=12.5,weight='bold'); ax[2].set_xlabel('z from axicon (mm)'); cb=fig.colorbar(im,ax=ax[2],pad=.02,shrink=.88); cb.ax.tick_params(colors=MUTED,labelsize=7); cb.set_label('|I_nominal − I_ideal|',color=MUTED,fontsize=8)
    fig.suptitle('V3 propagation: ideal vs nominal experimental constraints',color=TEXT,fontsize=17,weight='bold',y=.95); fig.text(.5,.865,'Same physical coordinates and propagation range; main panels peak-normalised for morphology',ha='center',color=MUTED,fontsize=9.2)
    p=out/'02_V3_propagation_ideal_vs_nominal.png';fig.savefig(p,dpi=480,bbox_inches='tight',facecolor=BG,pad_inches=.06);plt.close(fig);return p

def ladder(out,n):
    m=canonical_hardware_manifest(); f=float(hardware_value(m,'carrier_frequency_cpm')); panel=_panel_from_manifest(m); g=make_xy_grid(int(n),WINDOW_M/int(n)); X=np.asarray(g['X']);Y=np.asarray(g['Y']); cont=3*np.arctan2(Y,X); pix=pixelate(cont,g,panel); q=quantise_commanded_phase(pix,256); blaze=quantise_commanded_phase(pixelate(TWOPI*f*X,g,panel),256); route=build_system_route('V3',grid_n=n); four=np.abs(np.asarray(route['fourier_plane_before_iris']))**2; final=_xy(route)
    xx=np.asarray(g['x']); ids=np.flatnonzero(np.abs(xx)<=.08e-3); ext=[xx[ids[0]]*1e3,xx[ids[-1]]*1e3,xx[ids[0]]*1e3,xx[ids[-1]]*1e3]
    fig,ax=plt.subplots(1,6,figsize=(16,3.6),facecolor=BG);fig.subplots_adjust(left=.025,right=.99,bottom=.16,top=.74,wspace=.12)
    for a in ax:style.style_ax(a)
    for a,(v,t) in zip(ax[:4],[(cont,'continuous V3 phase'),(pix,'SLM pixelation\n8 µm pitch'),(q,'8-bit phase\n256 levels'),(blaze,'SLM2 carrier/blaze\n20 px period')]):a.imshow(np.asarray(v)[np.ix_(ids,ids)],origin='lower',extent=ext,cmap='twilight',vmin=0,vmax=TWOPI,interpolation='nearest',aspect='equal');a.set_title(t,color=TEXT,fontsize=10.2,weight='bold');a.set_xlabel('x (mm)',fontsize=7);a.tick_params(labelleft=False)
    ax[4].imshow(_norm(four),origin='lower',cmap=CMAP,norm=PNORM,interpolation='nearest',aspect='equal');ax[4].set_title('4F order selection\nfinite iris',color=TEXT,fontsize=10.2,weight='bold');ax[4].set_xticks([]);ax[4].set_yticks([])
    cr,x=_crop(final,route['grid']);ax[5].imshow(_norm(cr),origin='lower',extent=[x[0]*1e3,x[-1]*1e3,x[0]*1e3,x[-1]*1e3],cmap=CMAP,norm=PNORM,interpolation=style.DISPLAY_INTERPOLATION,aspect='equal');ax[5].set_title('resulting V3 field\nz=60 mm',color=TEXT,fontsize=10.2,weight='bold');ax[5].set_xlabel('x (mm)',fontsize=7);ax[5].tick_params(labelleft=False)
    fig.suptitle('How nominal experimental constraints enter the computational route',color=TEXT,fontsize=16.5,weight='bold',y=.96);fig.text(.5,.835,'continuous target → pixelated command → quantised phase → carrier/blaze → 4F selection → propagated field',ha='center',color=MUTED,fontsize=9.7)
    p=out/'03_constraint_ladder_SLM_4F.png';fig.savefig(p,dpi=500,bbox_inches='tight',facecolor=BG,pad_inches=.06);plt.close(fig);return p

def write_manifest(out):
    m=canonical_hardware_manifest(); data={'claim':'nominal_fixed_parameter_simulation_not_calibrated_bench_prediction','nominal_fixed_parameter_simulation_ready':m['nominal_fixed_parameter_simulation_ready'],'fixed_bench_prediction_ready':m['fixed_bench_prediction_ready'],'fixed_bench_prediction_blocker':m['fixed_bench_prediction_blocker'],'included':{'slm_pixel_pitch_um':float(hardware_value(m,'slm_pixel_pitch_m'))*1e6,'slm_phase_bits':int(hardware_value(m,'slm_phase_bits')),'slm_fill_factor':float(hardware_value(m,'slm_fill_factor')),'carrier_period_px':20,'carrier_frequency_cpm':float(hardware_value(m,'carrier_frequency_cpm')),'fourf_focal_length_m':float(hardware_value(m,'fourf_focal_length_m')),'fourier_iris_radius_m':float(hardware_value(m,'fourier_iris_radius_m')),'physical_axicon':True},'missing_measurements':['SLM phase LUT at operating wavelength','SLM phase stroke/nonlinearity','static SLM wavefront maps','calibrated fringing response','per-panel orientation/parity','measured 4F separations and iris centering/size','physical axicon clear aperture/surface map','camera scale/absolute observation-plane calibration'],'important_note':'Existing build_system_route presentation figures already contain the nominal hardware constraints; they are ideal only in the sense of zero deliberate errors.'};(out/'nominal_constraints_manifest.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--grid-n',type=int,default=2048);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);print(ideal_vs_nominal(a.output_dir,a.grid_n));print(v3_prop(a.output_dir,a.grid_n));print(ladder(a.output_dir,a.grid_n));write_manifest(a.output_dir)
if __name__=='__main__':main()
