"""q=20 correction benchmark v4: physical fit + illuminated-region retrieval.

The direct Eq. (3) inversion is accurate when its assumptions are met.  A
separate compatibility test of the complete dual-SLM/4F/axicon route showed,
however, that the stationary-phase approximation is poor at early z planes whose
mapped input annuli carry negligible q=20 field amplitude.  Those planes are not
useful samples of the Bessel focal line and should not be forced into the modal
retrieval.

v4 keeps the wide scan for physical parameter estimation, but chooses the subset
used for Miao phase retrieval from the independently known/measured intensity at
the axicon-input plane.  This is consistent with the Miao method, which already
requires an independent input-intensity measurement to resolve its
conjugate/180-degree ambiguity.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
from scipy import ndimage
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/'tools') not in sys.path: sys.path.insert(0,str(ROOT/'tools'))
MOD=ROOT/'notebooks'/'experimental'/'axicon_aberration_correction'
if str(MOD) not in sys.path: sys.path.insert(0,str(MOD))

import benchmark_q20_miao_vs_digital_twin as v1  # noqa: E402
import benchmark_q20_method_physics_v2 as v2  # noqa: E402
import benchmark_q20_method_physics_v3 as v3  # noqa: E402
from miao_full_retrieval import assemble_full_aperture, fit_plane_adaptive, interpolate_to_cartesian  # noqa: E402
from vbb_study.digital_twin.hierarchical_physical_fit import hierarchical_physical_fit  # noqa: E402
from vbb_study.digital_twin.vortex_system_error_sweeps import system_sweep_registry  # noqa: E402
from vbb_study.digital_twin.vortex_system_route import SystemErrorConfig  # noqa: E402

OUT=ROOT/'outputs'/'poster'/'q20_method_comparison_v4'
Z_FIT_M=v3.Z_FIT_M
Z_DISPLAY_M=v3.Z_DISPLAY_M
ILLUMINATION_AMPLITUDE_THRESHOLD=0.15
EPS=np.finfo(float).tiny

# Explicitly restore one shared numerical contract after importing older modules.
v1.GRID_N=v3.GRID_N; v1.WINDOW_M=v3.WINDOW_M
v1.Z_FIT_M=Z_FIT_M; v1.Z_DISPLAY_M=Z_DISPLAY_M
v1.NOISE_SIGMA=v3.NOISE_SIGMA; v1.FIT_CROP_HALF_M=v3.FIT_CROP_HALF_M
v1.METRIC_RADIUS_M=v2.METRIC_RADIUS_M
v2.Z_FIT_M=Z_FIT_M; v2.Z_DISPLAY_M=Z_DISPLAY_M


def _annulus_mean_amplitude(route: dict, z_values: np.ndarray) -> np.ndarray:
    field=np.asarray(route['field_on_axicon_plane'],np.complex128)
    amp=np.abs(field)
    x=np.asarray(route['grid']['x'],float); dx=float(route['grid']['dx'])
    wavelength=float(route['metadata']['wavelength_m']); k=2*np.pi/wavelength
    kp=float(route['metadata']['axicon']['exact_kr_m_inv'])
    theta=np.linspace(0,2*np.pi,720,endpoint=False)
    out=[]
    for z in np.asarray(z_values,float):
        rho=float(z)*kp/k
        xs=rho*np.cos(theta); ys=rho*np.sin(theta)
        xc=(xs-x[0])/dx; yc=(ys-x[0])/dx
        row=ndimage.map_coordinates(amp,[yc,xc],order=1,mode='constant',cval=0.0)
        out.append(float(np.mean(row)))
    out=np.asarray(out,float)
    return out/max(float(np.max(out)),EPS)


def illuminated_plane_indices(route: dict, z_values: np.ndarray,
                              threshold: float=ILLUMINATION_AMPLITUDE_THRESHOLD) -> tuple[np.ndarray,np.ndarray]:
    """Select z planes whose stationary-phase input annuli are appreciably lit."""
    rel=_annulus_mean_amplitude(route,z_values)
    ids=np.flatnonzero(rel>=float(threshold))
    if len(ids)<5:
        # Five annuli is a minimum useful radial diversity for this benchmark.
        ids=np.argsort(rel)[-min(5,len(rel)):]
        ids=np.sort(ids)
    return np.asarray(ids,int),rel


def miao_correction_selected(route: dict, noisy_stack: np.ndarray,
                             z_values: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray,dict]:
    grid=route['grid']; pixel=float(grid['dx'])
    nominal_kp=float(route['metadata']['axicon']['exact_kr_m_inv'])
    center=v2._known_axis_center(route)
    ids=np.asarray(indices,int); zsub=np.asarray(z_values,float)[ids]
    retrievals=[]
    for j,i in enumerate(ids):
        image=np.asarray(noisy_stack[int(i)],float); z=float(z_values[int(i)])
        retrievals.append(fit_plane_adaptive(
            image,j,z,center,pixel,v3.v1.Q,nominal_kp,
            max_aberration_order=30,order_step=2,cost_threshold=0.05,
            min_fractional_improvement=0.006,rmax_um=720,n_r=52,n_theta=144,
        ))
    refs=v2._reference_rows(route,retrievals,zsub)
    full=assemble_full_aperture(
        retrievals,zsub,float(route['metadata']['wavelength_m']),
        k_perp_nominal_m_inv=nominal_kp,reference_intensity_rows=refs,
    )
    cart=interpolate_to_cartesian(full,grid_size=640,padding_fraction=0.02)
    correction,valid=v1._map_cartesian_phase_to_grid(cart,grid)
    return correction,{
        'branch':full.branch,
        'branch_score_direct':full.branch_score_direct,
        'branch_score_conjugate':full.branch_score_conjugate,
        'valid_grid_fraction':float(np.mean(valid)),
        'median_plane_fit_corr':float(np.median([r.fit_corr for r in retrievals])),
        'median_plane_fit_nrmse':float(np.median([r.fit_nrmse for r in retrievals])),
        'median_k_perp_m_inv':float(np.median([r.k_perp_m_inv for r in retrievals])),
        'median_mode_order':float(np.median([r.aberration_order_max for r in retrievals])),
        'rho_min_mm':float(np.min(full.rho_m)*1e3),'rho_max_mm':float(np.max(full.rho_m)*1e3),
        'selected_plane_indices':[int(i) for i in ids],
        'selected_z_mm':[float(z*1e3) for z in zsub],
    }


def _selected_from_hierarchical(fit) -> dict[str,float]:
    out={}
    for step in fit.steps:
        if step.accepted and step.selected_family is not None:
            out[str(step.selected_family)]=float(step.selected_value)
    return out


def build(out: Path=OUT) -> dict:
    out.mkdir(parents=True,exist_ok=True)
    registry=system_sweep_registry(); nominal=SystemErrorConfig()
    nominal_route=v1._route(nominal,None); residual=v1._residual_phase(nominal_route['grid'])
    truth=v1.apply_registry_family(nominal,'beam_radius_scale',0.85,registry=registry)
    truth=v1.apply_registry_family(truth,'axicon_lateral_decentre_x',250e-6,registry=registry)
    ideal_route=v1._route(nominal,None); distorted_route=v1._route(truth,residual)
    distorted_clean=v1._propagate(distorted_route,Z_FIT_M)
    rng=np.random.default_rng(v3.SEED); noise=rng.normal(size=distorted_clean.shape)
    distorted=v1._add_noise(distorted_clean,noise)

    x=np.asarray(ideal_route['grid']['x'],float); ids=np.flatnonzero(np.abs(x)<=v3.FIT_CROP_HALF_M)
    target=distorted[:,ids[:,None],ids]; simulator=v1.PhysicalSimulator(ids)

    beam_cfg,beam_fit=v3._fit_single_registry_parameter(target,simulator,nominal,'beam_radius_scale',registry)
    iris_fit=hierarchical_physical_fit(
        target_stack=target,simulate_config=simulator,initial_config=beam_cfg,
        families=('fourf_iris_radius_scale',),registry=registry,max_stages=1,
        min_improvement_fraction=0.004,loss_fn=v2.axisymmetric_shape_error)
    lateral_fit=hierarchical_physical_fit(
        target_stack=target,simulate_config=simulator,initial_config=iris_fit.final_config,
        families=v2.LATERAL_PARAMETERS,registry=registry,max_stages=1,
        min_improvement_fraction=0.004,loss_fn=v2.lateral_trajectory_error)
    estimated=lateral_fit.final_config
    compensated=v1._compensate_physical_parameters(truth,estimated)
    compensated_route=v1._route(compensated,residual)
    compensated_clean=v1._propagate(compensated_route,Z_FIT_M)
    compensated_noisy=v1._add_noise(compensated_clean,noise)

    # The baseline and hybrid are both allowed the independent input-intensity
    # information assumed by the Miao method; their useful focal-line subsets may
    # differ because the physical compensation changes the field incident on the
    # axicon.
    miao_ids,miao_illum=illuminated_plane_indices(distorted_route,Z_FIT_M)
    hybrid_ids,hybrid_illum=illuminated_plane_indices(compensated_route,Z_FIT_M)
    miao_phase,miao_diag=miao_correction_selected(distorted_route,distorted,Z_FIT_M,miao_ids)
    hybrid_phase,hybrid_diag=miao_correction_selected(compensated_route,compensated_noisy,Z_FIT_M,hybrid_ids)

    compensated_nores=v1._route(compensated,None)
    oracle_phase,oracle_valid=v2._oracle_phase(compensated_route,compensated_nores)
    ideal_disp=v1._propagate(ideal_route,Z_DISPLAY_M); distorted_disp=v1._propagate(distorted_route,Z_DISPLAY_M)
    miao_disp=v1._propagate(distorted_route,Z_DISPLAY_M,miao_phase)
    hybrid_disp=v1._propagate(compensated_route,Z_DISPLAY_M,hybrid_phase)
    oracle_disp=v1._propagate(compensated_route,Z_DISPLAY_M,oracle_phase)
    metrics=v2._metric_rows(Z_DISPLAY_M,ideal_disp,distorted_disp,miao_disp,hybrid_disp,oracle_disp,ideal_route['grid'])
    metrics.to_csv(out/'comparison_metrics_vs_z.csv',index=False)

    selected={}
    if beam_cfg!=nominal: selected['beam_radius_scale']=float(beam_fit.best_value)
    selected.update(_selected_from_hierarchical(iris_fit)); selected.update(_selected_from_hierarchical(lateral_fit))
    mvalid=oracle_valid&(np.abs(miao_phase)>0); hvalid=oracle_valid&(np.abs(hybrid_phase)>0)
    summary={
        'study':'q20 physical-observable fit + illuminated-focal-region Miao retrieval v4',
        'truth':{'beam_radius_scale':0.85,'axicon_lateral_decentre_x_m':250e-6,
                 'residual_phase':'0.42 cos2theta+0.24 sin3theta+0.12 cos5theta'},
        'physical_parameter_estimation':{
            'selected':selected,'beam_radius_fit':beam_fit.as_dict(),
            'fourier_iris_fit':iris_fit.as_dict(),'lateral_fit':lateral_fit.as_dict(),
            'method':'beam radius: longitudinal Bessel envelope; iris: radial morphology; lateral errors: centroid trajectory'},
        'retrieval_plane_selection':{
            'criterion':'mean axicon-input annulus amplitude >= 0.15 of scan maximum',
            'miao_relative_annulus_amplitude':[float(v) for v in miao_illum],
            'hybrid_relative_annulus_amplitude':[float(v) for v in hybrid_illum],
        },
        'miao_only':{**miao_diag,'phase_rms_to_oracle_rad':v2._circular_phase_rms(miao_phase,oracle_phase,mvalid)},
        'digital_twin_plus_miao':{**hybrid_diag,'phase_rms_to_oracle_rad':v2._circular_phase_rms(hybrid_phase,oracle_phase,hvalid)},
    }
    for key,col in (('aberrated','distorted'),('miao_only','miao_only'),
                    ('digital_twin_plus_miao','digital_twin_plus_miao'),('oracle_phase_only','oracle_phase_only')):
        summary.setdefault(key,{})
        summary[key].update({'mean_pearson_r':float(metrics[f'{col}_pearson_r'].mean()),
                             'mean_nrmse':float(metrics[f'{col}_nrmse'].mean())})
    summary['hybrid_minus_miao']={'mean_pearson_r':summary['digital_twin_plus_miao']['mean_pearson_r']-summary['miao_only']['mean_pearson_r'],
                                  'mean_nrmse':summary['digital_twin_plus_miao']['mean_nrmse']-summary['miao_only']['mean_nrmse']}
    summary['hybrid_minus_aberrated']={'mean_pearson_r':summary['digital_twin_plus_miao']['mean_pearson_r']-summary['aberrated']['mean_pearson_r'],
                                       'mean_nrmse':summary['digital_twin_plus_miao']['mean_nrmse']-summary['aberrated']['mean_nrmse']}

    np.save(out/'miao_correction_rad.npy',miao_phase.astype(np.float32)); np.save(out/'hybrid_correction_rad.npy',hybrid_phase.astype(np.float32))
    png,pdf,preview=v2._build_diagnostic_figure(out,ideal_route['grid'],metrics,summary,ideal_disp,distorted_disp,miao_disp,hybrid_disp,oracle_disp)
    p4=out/'q20_method_physics_v4.png'; d4=out/'q20_method_physics_v4.pdf'; r4=out/'q20_method_physics_v4.preview.jpg'
    Path(png).replace(p4); Path(pdf).replace(d4); Path(preview).replace(r4)
    with Image.open(p4) as im: summary['assets']={'png':str(p4),'pdf':str(d4),'preview':str(r4),'pixel_size':list(im.size),'dpi':list(im.info.get('dpi',(0,0)))}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2)); return summary


if __name__=='__main__': build()
