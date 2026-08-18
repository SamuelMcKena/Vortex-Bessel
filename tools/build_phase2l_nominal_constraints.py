"""Presentation figures comparing a continuous ideal beam with the repository's nominal bench-constrained route.

Important claim boundary: build_system_route already includes the known nominal
hardware constraints (SLM pixelation, 256 phase levels, fill-factor throughput,
carrier/blaze, explicit finite-iris 4F filtering, and the physical axicon).  It
does not invent missing measured LUTs, phase-stroke maps, fringing kernels,
static SLM maps, or unmeasured bench geometry.  Therefore the constrained row
is a nominal fixed-parameter simulation, not a calibrated bench prediction.
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
WINDOW_M=10e-3; XY_HALFWIDTH_M=0.18e-3; CMAP=style.CMAP; TEXT=style.TEXT; MUTED=style.MUTED; FIG_BG=style.FIG_BG
NORM=colors.PowerNorm(gamma=0.55,vmin=0,vmax=1)

def _ell(case_id): return {'B0':0,'V1':1,'V3':3}[case_id]
def _norm(a):
    a=np.maximum(np.asarray(a,float),0); return a/max(float(np.max(a)),np.finfo(float).tiny)
def _crop(a,grid,half=XY_HALFWIDTH_M):
    x=np.asarray(grid['x'],float); ids=np.flatnonzero(np.abs(x)<=half); return np.asarray(a)[np.ix_(ids,ids)],x[ids]
def _prop(post,grid):
    m=canonical_hardware_manifest(); return build_fixed_support_spectrum(np.asarray(post,np.complex128),dict(grid),wavelength_m=float(hardware_value(m,'wavelength_m')),z_max_m=float(Z_VALUES_M[-1]),minimum_retained_spectral_power=0.995)

def continuous_ideal(case_id,grid_n):
    m=canonical_hardware_manifest(); wl=float(hardware_value(m,'wavelength_m')); w=float(hardware_value(m,'beam_radius_on_slm_m'))
    n_ax=float(hardware_value(m,'axicon_refractive_index')); n_ext=float(hardware_value(m,'axicon_external_medium_index')); gamma=math.radians(float(hardware_value(m,'axicon_base_angle_deg')))
    g=make_xy_grid(int(grid_n),WINDOW_M/int(grid_n)); beam,_=gaussian_input_field(g,wavelength_m=wl,canonical_radius_m=w,error=GaussianBeamError())
    phi=float(_ell(case_id))*np.arctan2(np.asarray(g['Y']),np.asarray(g['X'])); shaped=beam*np.exp(1j*phi)
    at,_=physical_axicon_on_own_plane(g,wavelength_m=wl,base_angle_rad=gamma,refractive_index=n_ax,external_index=n_ext,error=AxiconError())
    return {'grid':g,'post_axicon':shaped*at,'phase':phi}

def xy_xz(route,label):
    p=_prop(route['post_axicon'],route['grid']); f=native_field_at_z(p,Z_REF_M); xy=np.abs(np.asarray(f))**2
    mapped=build_fixed_plane_longitudinal_map(p,z_values_m=Z_VALUES_M,x_coordinates_m=PROP_COORD_M,y_coordinates_m=PROP_COORD_M,fixed_x_m=0.0,fixed_y_m=0.0,source_label=label)
    return xy,np.asarray(mapped.xz_intensity,float)

def figure_ideal_vs_nominal(outdir,grid_n):
    cases=[('B0','B0 — ℓ=0'),('V1','V1 — ℓ=1'),('V3','V3 — ℓ=3')]
    fig,axes=plt.subplots(2,3,figsize=(13.2,8.3),facecolor=FIG_BG); fig.subplots_adjust(left=.06,right=.985,bottom=.08,top=.82,wspace=.10,hspace=.27)
    for c,(cid,title) in enumerate(cases):
        ideal=continuous_ideal(cid,grid_n); nominal=build_system_route(cid,grid_n=grid_n)
        ixy,_=xy_xz(ideal,f'phase2l-ideal-{cid}'); nxy,_=xy_xz(nominal,f'phase2l-nominal-{cid}')
        for r,(vals,route,prefix) in enumerate([(ixy,ideal,'continuous ideal'),(nxy,nominal,'nominal bench-constrained')]):
            ax=axes[r,c]; style.style_ax(ax); cr,x=_crop(vals,route['grid'])
            ax.imshow(_norm(cr),origin='lower',extent=[x[0]*1e3,x[-1]*1e3,x[0]*1e3,x[-1]*1e3],cmap=CMAP,norm=NORM,interpolation=style.DISPLAY_INTERPOLATION,aspect='equal')
            ax.set_title(f'{title}\n{prefix}',color=TEXT,fontsize=11.5,weight='bold',pad=7); ax.set_xlabel('x (mm)',fontsize=8)
            if c==0: ax.set_ylabel('y (mm)',fontsize=8)
            else: ax.tick_params(labelleft=False)
        del ideal,nominal,ixy,nxy; gc.collect()
    fig.suptitle('Ideal beam family → nominal experimental constraints',color=TEXT,fontsize=18,weight='bold',y=.965)
    fig.text(.5,.905,'Bottom row adds SLM pixelation + 8-bit phase levels + fill-factor throughput + blaze/carrier + finite 4F order selection',ha='center',color=MUTED,fontsize=10.2)
    fig.text(.5,.865,'No deliberate misalignment, wavefront error, axicon defect or measured correction map is applied',ha='center',color=MUTED,fontsize=9.3)
    p=outdir/'01_ideal_vs_nominal_constraints_B0_V1_V3.png'; fig.savefig(p,dpi=480,bbox_inches='tight',facecolor=FIG_BG,pad_inches=.06); plt.close(fig); return p

def figure_v3_propagation(outdir,grid_n):
    ideal=continuous_ideal('V3',grid_n); nominal=build_system_route('V3',grid_n=grid_n); _,ixz=xy_xz(ideal,'phase2l-i-v3'); _,nxz=xy_xz(nominal,'phase2l-n-v3')
    ii=_norm(ixz); nn=_norm(nxz); dd=np.abs(nn-ii); z=Z_VALUES_M*1e3; x=PROP_COORD_M*1e3
    fig,axes=plt.subplots(1,3,figsize=(14.5,4.8),facecolor=FIG_BG); fig.subplots_adjust(left=.055,right=.985,bottom=.16,top=.78,wspace=.13)
    for ax in axes: style.style_ax(ax)
    for ax,d,t in [(axes[0],ii,'Continuous ideal'),(axes[1],nn,'Nominal bench-constrained')]:
        ax.imshow(d.T,origin='lower',extent=[z[0],z[-1],x[0],x[-1]],cmap=CMAP,norm=NORM,interpolation=style.DISPLAY_INTERPOLATION,aspect='auto'); ax.set_title(t,color=TEXT,fontsize=12.5,weight='bold'); ax.set_xlabel('z from axicon (mm)'); ax.axvline(60,color='white',alpha=.25,ls='--',lw=.8)
    axes[0].set_ylabel('x at fixed y=0 (mm)'); im=axes[2].imshow(dd.T,origin='lower',extent=[z[0],z[-1],x[0],x[-1]],cmap=CMAP,vmin=0,vmax=max(float(np.max(dd)),1e-12),interpolation=style.DISPLAY_INTERPOLATION,aspect='auto'); axes[2].set_title('Absolute morphology difference',color=TEXT,fontsize=12.5,weight='bold'); axes[2].set_xlabel('z from axicon (mm)')
    cb=fig.colorbar(im,ax=axes[2],pad=.02,shrink=.88); cb.ax.tick_params(colors=MUTED,labelsize=7); cb.set_label('|I_nominal − I_ideal|',color=MUTED,fontsize=8)
    fig.suptitle('V3 propagation: continuous ideal vs nominal experimental constraints',color=TEXT,fontsize=17,weight='bold',y=.95); fig.text(.5,.865,'Same physical coordinates and propagation range; main panels are peak-normalised for morphology comparison',ha='center',color=MUTED,fontsize=9.2)
    p=outdir/'02_V3_propagation_ideal_vs_nominal.png'; fig.savefig(p,dpi=480,bbox_inches='tight',facecolor=FIG_BG,pad_inches=.06); plt.close(fig); return p

def figure_constraint_ladder(outdir,grid_n):
    m=canonical_hardware_manifest(); carrier=float(hardware_value(m,'carrier_frequency_cpm')); panel=_panel_from_manifest(m); g=make_xy_grid(int(grid_n),WINDOW_M/int(grid_n)); X=np.asarray(g['X']); Y=np.asarray(g['Y'])
    cont=3*np.arctan2(Y,X); pix=pixelate(cont,g,panel); quant=quantise_commanded_phase(pix,256); blaze=quantise_commanded_phase(pixelate(TWOPI*carrier*X,g,panel),256)
    route=build_system_route('V3',grid_n=grid_n); fourier=np.abs(np.asarray(route['fourier_plane_before_iris']))**2; final,_=xy_xz(route,'phase2l-ladder')
    xx=np.asarray(g['x']); ids=np.flatnonzero(np.abs(xx)<=.08e-3); ext=[xx[ids[0]]*1e3,xx[ids[-1]]*1e3,xx[ids[0]]*1e3,xx[ids[-1]]*1e3]
    fig,axes=plt.subplots(1,6,figsize=(16,3.6),facecolor=FIG_BG); fig.subplots_adjust(left=.025,right=.99,bottom=.16,top=.74,wspace=.12)
    for ax in axes: style.style_ax(ax)
    for ax,(d,t) in zip(axes[:4],[(cont,'continuous V3 phase'),(pix,'SLM pixelation\n8 µm pitch'),(quant,'8-bit phase\n256 levels'),(blaze,'SLM2 blaze/carrier\n20 px period')]):
        ax.imshow(np.asarray(d)[np.ix_(ids,ids)],origin='lower',extent=ext,cmap='twilight',vmin=0,vmax=TWOPI,interpolation='nearest',aspect='equal'); ax.set_title(t,color=TEXT,fontsize=10.2,weight='bold',pad=7); ax.set_xlabel('x (mm)',fontsize=7); ax.tick_params(labelleft=False)
    axes[4].imshow(_norm(fourier),origin='lower',cmap=CMAP,norm=NORM,interpolation='nearest',aspect='equal'); axes[4].set_title('4F order selection\nfinite Fourier iris',color=TEXT,fontsize=10.2,weight='bold',pad=7); axes[4].set_xticks([]); axes[4].set_yticks([])
    cr,x=_crop(final,route['grid']); axes[5].imshow(_norm(cr),origin='lower',extent=[x[0]*1e3,x[-1]*1e3,x[0]*1e3,x[-1]*1e3],cmap=CMAP,norm=NORM,interpolation=style.DISPLAY_INTERPOLATION,aspect='equal'); axes[5].set_title('resulting V3 field\nz = 60 mm',color=TEXT,fontsize=10.2,weight='bold',pad=7); axes[5].set_xlabel('x (mm)',fontsize=7); axes[5].tick_params(labelleft=False)
    fig.suptitle('How nominal experimental constraints enter the computational route',color=TEXT,fontsize=16.5,weight='bold',y=.96); fig.text(.5,.835,'continuous target → sampled SLM command → quantised phase → diffraction carrier → finite 4F selection → propagated field',ha='center',color=MUTED,fontsize=9.7)
    p=outdir/'03_constraint_ladder_SLM_4F.png'; fig.savefig(p,dpi=500,bbox_inches='tight',facecolor=FIG_BG,pad_inches=.06); plt.close(fig); return p

def manifest(outdir):
    m=canonical_hardware_manifest(); data={'claim':'nominal_fixed_parameter_simulation_not_calibrated_bench_prediction','nominal_fixed_parameter_simulation_ready':m['nominal_fixed_parameter_simulation_ready'],'fixed_bench_prediction_ready':m['fixed_bench_prediction_ready'],'fixed_bench_prediction_blocker':m['fixed_bench_prediction_blocker'],'included':{'slm_pixel_pitch_um':float(hardware_value(m,'slm_pixel_pitch_m'))*1e6,'slm_phase_bits':int(hardware_value(m,'slm_phase_bits')),'slm_fill_factor':float(hardware_value(m,'slm_fill_factor')),'carrier_period_px':20,'carrier_frequency_cpm':float(hardware_value(m,'carrier_frequency_cpm')),'fourf_focal_length_m':float(hardware_value(m,'fourf_focal_length_m')),'fourier_iris_radius_m':float(hardware_value(m,'fourier_iris_radius_m')),'physical_axicon':True},'missing_measurements':['SLM phase LUT at operating wavelength','SLM phase stroke/nonlinearity','static SLM wavefront maps','calibrated fringing-field response','per-panel orientation/parity','measured 4F separations and iris centering/size','physical axicon clear aperture/surface map','camera scale/absolute observation-plane calibration'],'important_note':'Existing build_system_route presentation figures already contain nominal hardware constraints; they are ideal only in the sense of zero deliberate errors.'}; (outdir/'nominal_constraints_manifest.json').write_text(json.dumps(data,indent=2),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--grid-n',type=int,default=2048); ap.add_argument('--output-dir',type=Path,required=True); a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True); print(figure_ideal_vs_nominal(a.output_dir,a.grid_n)); print(figure_v3_propagation(a.output_dir,a.grid_n)); print(figure_constraint_ladder(a.output_dir,a.grid_n)); manifest(a.output_dir)
if __name__=='__main__': main()
