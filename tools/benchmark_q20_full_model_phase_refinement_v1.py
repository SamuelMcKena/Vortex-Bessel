"""Full-forward-model residual phase refinement for the q=20 correction study.

This benchmark is the next step beyond Miao-only residual retrieval.  The Miao
modal inversion is retained as the analytical baseline, but the digital twin is
allowed to refine a low-dimensional residual phase by evaluating candidate
wavefronts through the *complete* Gaussian -> SLM1 -> SLM2 -> 4F/iris -> axicon
route.

The controlled sequence is:

1. generate a wide noisy z-stack containing known physical errors + unknown
   angular phase;
2. estimate physically distinct errors using the observables established in v3;
3. apply those inferred physical adjustments and acquire a second synthetic
   stack (closed-loop analogue);
4. compare Miao residual correction with a full-forward-model estimate of the
   same residual phase;
5. fit the model only on alternating illuminated z planes and report held-out
   planes independently.

The full-model residual is represented by Fourier angular phase terms m=1..6.
The programmed q=20 phase is never part of the fitted correction.  The synthetic
truth contains m=2,3,5 but the optimizer is not told which modes are non-zero.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy import optimize
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/'tools') not in sys.path: sys.path.insert(0,str(ROOT/'tools'))

import benchmark_q20_miao_vs_digital_twin as v1  # noqa: E402
import benchmark_q20_method_physics_v2 as v2  # noqa: E402
import benchmark_q20_method_physics_v3 as v3  # noqa: E402
import benchmark_q20_method_physics_v4 as v4  # noqa: E402
from vbb_study.digital_twin.hierarchical_physical_fit import hierarchical_physical_fit  # noqa: E402
from vbb_study.digital_twin.vortex_system_error_sweeps import system_sweep_registry  # noqa: E402
from vbb_study.digital_twin.vortex_system_route import SystemErrorConfig  # noqa: E402

OUT=ROOT/'outputs'/'validation'/'q20_full_model_phase_refinement_v1'
Z_WIDE=v3.Z_FIT_M
Z_EVAL=v3.Z_DISPLAY_M
MODES=np.arange(1,7,dtype=int)
PHASE_BOUND_RAD=1.0
REGULARIZATION=2e-4
FIT_CROP_HALF_M=1.15e-3
EPS=np.finfo(float).tiny

v1.GRID_N=v3.GRID_N; v1.WINDOW_M=v3.WINDOW_M
v1.Z_FIT_M=Z_WIDE; v1.Z_DISPLAY_M=Z_EVAL
v1.NOISE_SIGMA=v3.NOISE_SIGMA; v1.METRIC_RADIUS_M=v2.METRIC_RADIUS_M
v2.Z_FIT_M=Z_WIDE; v2.Z_DISPLAY_M=Z_EVAL


def phase_from_coefficients(grid: dict, coeff: np.ndarray) -> np.ndarray:
    c=np.asarray(coeff,float)
    if c.shape!=(2*len(MODES),): raise ValueError('coefficient vector has wrong shape')
    theta=np.asarray(grid['PHI'],float)
    phase=np.zeros_like(theta,float)
    for j,m in enumerate(MODES):
        phase += c[2*j]*np.cos(int(m)*theta) + c[2*j+1]*np.sin(int(m)*theta)
    return phase


def _truth_coefficients() -> np.ndarray:
    c=np.zeros(2*len(MODES),float)
    # [cos(m theta), sin(m theta)] pairs.
    c[2*(2-1)] = 0.42
    c[2*(3-1)+1] = 0.24
    c[2*(5-1)] = 0.12
    return c


def _normalised_crop(stack: np.ndarray, ids: np.ndarray) -> np.ndarray:
    a=np.asarray(stack,float)[:,ids[:,None],ids]
    peak=np.max(a,axis=(1,2),keepdims=True)
    return a/np.maximum(peak,EPS)


def _stack_rmse(model: np.ndarray, data: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(model,float)-np.asarray(data,float))**2)))


def _fit_physical_parameters(target: np.ndarray, simulator, nominal, registry):
    beam_cfg,beam_fit=v3._fit_single_registry_parameter(target,simulator,nominal,'beam_radius_scale',registry)
    iris_fit=hierarchical_physical_fit(
        target_stack=target,simulate_config=simulator,initial_config=beam_cfg,
        families=('fourf_iris_radius_scale',),registry=registry,max_stages=1,
        min_improvement_fraction=0.004,loss_fn=v2.axisymmetric_shape_error)
    lateral_fit=hierarchical_physical_fit(
        target_stack=target,simulate_config=simulator,initial_config=iris_fit.final_config,
        families=v2.LATERAL_PARAMETERS,registry=registry,max_stages=1,
        min_improvement_fraction=0.004,loss_fn=v2.lateral_trajectory_error)
    selected={}
    if beam_cfg!=nominal: selected['beam_radius_scale']=float(beam_fit.best_value)
    selected.update(v4._selected_from_hierarchical(iris_fit)); selected.update(v4._selected_from_hierarchical(lateral_fit))
    return lateral_fit.final_config, selected, beam_fit, iris_fit, lateral_fit


def refine_phase_full_model(config: SystemErrorConfig, measured_stack: np.ndarray,
                            z_values: np.ndarray, train_indices: np.ndarray,
                            *, initial: np.ndarray|None=None) -> tuple[np.ndarray,dict]:
    """Estimate SLM1 residual phase coefficients with the complete forward route."""
    route0=v1._route(config,None); grid=route0['grid']
    x=np.asarray(grid['x'],float); crop=np.flatnonzero(np.abs(x)<=FIT_CROP_HALF_M)
    train=np.asarray(train_indices,int)
    target=_normalised_crop(np.asarray(measured_stack,float)[train],crop)
    ztrain=np.asarray(z_values,float)[train]
    x0=np.zeros(2*len(MODES),float) if initial is None else np.asarray(initial,float).copy()
    bounds=[(-PHASE_BOUND_RAD,PHASE_BOUND_RAD)]*len(x0)
    history=[]

    def objective(c):
        phase=phase_from_coefficients(grid,c)
        route=v1._route(config,phase)
        pred=v1._propagate(route,ztrain)
        predn=_normalised_crop(pred,crop)
        data_term=_stack_rmse(predn,target)
        reg=REGULARIZATION*float(np.mean(np.asarray(c,float)**2))
        value=data_term+reg
        history.append({'cost':float(value),'data_rmse':float(data_term),'coefficients':[float(v) for v in c]})
        return value

    initial_cost=float(objective(x0))
    result=optimize.minimize(
        objective,x0,method='L-BFGS-B',bounds=bounds,
        options={'maxiter':18,'ftol':2e-7,'gtol':2e-5,'maxls':15},
    )
    coeff=np.asarray(result.x,float)
    # Independent held-out score is evaluated outside this function.
    return coeff,{
        'success':bool(result.success),'message':str(result.message),
        'iterations':int(result.nit),'function_evaluations':int(result.nfev),
        'initial_cost':initial_cost,'final_cost':float(result.fun),
        'coefficients':[float(v) for v in coeff],
        'modes':[int(m) for m in MODES],
        'history_tail':history[-20:],
    }


def _phase_coeff_table(coeff: np.ndarray, truth: np.ndarray) -> list[dict]:
    rows=[]
    for j,m in enumerate(MODES):
        rows.append({'m':int(m),'cos_est_rad':float(coeff[2*j]),'sin_est_rad':float(coeff[2*j+1]),
                     'cos_truth_rad':float(truth[2*j]),'sin_truth_rad':float(truth[2*j+1])})
    return rows


def _heldout_model_score(config, coeff, target_stack, z_values, indices) -> dict:
    route0=v1._route(config,None); x=np.asarray(route0['grid']['x'],float)
    crop=np.flatnonzero(np.abs(x)<=FIT_CROP_HALF_M); ids=np.asarray(indices,int)
    phase=phase_from_coefficients(route0['grid'],coeff)
    pred=v1._propagate(v1._route(config,phase),np.asarray(z_values,float)[ids])
    pn=_normalised_crop(pred,crop); tn=_normalised_crop(np.asarray(target_stack,float)[ids],crop)
    per=[]
    for i in range(len(ids)):
        a=pn[i].ravel(); b=tn[i].ravel()
        per.append({'z_mm':float(z_values[ids[i]]*1e3),
                    'pearson_r':float(np.corrcoef(a,b)[0,1]),
                    'rmse':float(np.sqrt(np.mean((a-b)**2)))})
    return {'mean_pearson_r':float(np.mean([r['pearson_r'] for r in per])),
            'mean_rmse':float(np.mean([r['rmse'] for r in per])),'planes':per}


def build(out: Path=OUT) -> dict:
    out.mkdir(parents=True,exist_ok=True)
    registry=system_sweep_registry(); nominal=SystemErrorConfig()
    ideal_route=v1._route(nominal,None); grid=ideal_route['grid']
    truth_coeff=_truth_coefficients(); residual=phase_from_coefficients(grid,truth_coeff)
    # Use the same physical truth as v2-v4.
    truth=v1.apply_registry_family(nominal,'beam_radius_scale',0.85,registry=registry)
    truth=v1.apply_registry_family(truth,'axicon_lateral_decentre_x',250e-6,registry=registry)
    distorted_route=v1._route(truth,residual); distorted_clean=v1._propagate(distorted_route,Z_WIDE)
    rng=np.random.default_rng(v3.SEED); noise=rng.normal(size=distorted_clean.shape)
    distorted=v1._add_noise(distorted_clean,noise)

    # Physical estimation uses the wide scan.
    x=np.asarray(grid['x'],float); phys_crop=np.flatnonzero(np.abs(x)<=v3.FIT_CROP_HALF_M)
    target_phys=distorted[:,phys_crop[:,None],phys_crop]; simulator=v1.PhysicalSimulator(phys_crop)
    estimated,selected,beam_fit,iris_fit,lateral_fit=_fit_physical_parameters(target_phys,simulator,nominal,registry)

    # Closed-loop physical adjustment followed by a new stack.  This mirrors the
    # intended experimental sequence: estimate component errors, adjust, then
    # retrieve the residual wavefront from a fresh z scan.
    adjusted=v1._compensate_physical_parameters(truth,estimated)
    adjusted_route=v1._route(adjusted,residual)
    adjusted_clean=v1._propagate(adjusted_route,Z_WIDE)
    noise2=np.random.default_rng(v3.SEED+1).normal(size=adjusted_clean.shape)
    adjusted_measured=v1._add_noise(adjusted_clean,noise2)

    illuminated,illum=v4.illuminated_plane_indices(adjusted_route,Z_WIDE)
    train=illuminated[::2]; held=illuminated[1::2]
    if len(held)<2:
        train=illuminated[:-2]; held=illuminated[-2:]

    # Paper residual retrieval on the adjusted stack.
    miao_phase,miao_diag=v4.miao_correction_selected(adjusted_route,adjusted_measured,Z_WIDE,illuminated)

    # Full-forward-model refinement.  Start from zero deliberately: success must
    # come from the measured intensity stack and full route, not synthetic truth.
    coeff,refine_diag=refine_phase_full_model(adjusted,adjusted_measured,Z_WIDE,train)
    full_phase=phase_from_coefficients(grid,coeff)
    held_score=_heldout_model_score(adjusted,coeff,adjusted_measured,Z_WIDE,held)
    truth_held=_heldout_model_score(adjusted,truth_coeff,adjusted_measured,Z_WIDE,held)

    # Apply negative fitted residual on the same SLM1 plane as the synthetic
    # aberration.  Miao correction remains applied at its native axicon-input
    # plane, which is a favourable baseline for the paper method.
    full_corrected_route=v1._route(adjusted,residual-full_phase)

    ideal_eval=v1._propagate(ideal_route,Z_EVAL)
    distorted_eval=v1._propagate(distorted_route,Z_EVAL)
    adjusted_uncorrected_eval=v1._propagate(adjusted_route,Z_EVAL)
    miao_eval=v1._propagate(adjusted_route,Z_EVAL,miao_phase)
    full_eval=v1._propagate(full_corrected_route,Z_EVAL)

    # Exact phase-only oracle after physical adjustment.
    adjusted_nores=v1._route(adjusted,None)
    oracle_phase,_=v2._oracle_phase(adjusted_route,adjusted_nores)
    oracle_eval=v1._propagate(adjusted_route,Z_EVAL,oracle_phase)

    # Reuse metric definition against the independent nominal route.
    metrics=v2._metric_rows(Z_EVAL,ideal_eval,adjusted_uncorrected_eval,miao_eval,full_eval,oracle_eval,grid)
    # Rename the v2 'digital_twin_plus_miao' columns to full-model refinement.
    metrics=metrics.rename(columns={
        'distorted_pearson_r':'after_physical_adjustment_pearson_r',
        'distorted_nrmse':'after_physical_adjustment_nrmse',
        'digital_twin_plus_miao_pearson_r':'full_model_refinement_pearson_r',
        'digital_twin_plus_miao_nrmse':'full_model_refinement_nrmse',
    })
    # Add original distorted beam metrics for context.
    orig=v2._metric_rows(Z_EVAL,ideal_eval,distorted_eval,distorted_eval,distorted_eval,oracle_eval,grid)
    metrics['original_distorted_pearson_r']=orig['distorted_pearson_r']
    metrics['original_distorted_nrmse']=orig['distorted_nrmse']
    metrics.to_csv(out/'metrics_vs_z.csv',index=False)

    def avg(prefix):
        return {'mean_pearson_r':float(metrics[f'{prefix}_pearson_r'].mean()),
                'mean_nrmse':float(metrics[f'{prefix}_nrmse'].mean())}
    phase_rms=float(np.sqrt(np.mean(np.angle(np.exp(1j*(full_phase-residual)))**2)))
    summary={
        'study':'q20 full-forward-model residual phase refinement v1',
        'physical_truth':{'beam_radius_scale':0.85,'axicon_lateral_decentre_x_m':250e-6},
        'physical_estimation':{'selected':selected,'beam_radius_fit':beam_fit.as_dict(),
            'iris_fit':iris_fit.as_dict(),'lateral_fit':lateral_fit.as_dict()},
        'residual_phase_truth':_phase_coeff_table(truth_coeff,truth_coeff),
        'illuminated_z_mm':[float(Z_WIDE[i]*1e3) for i in illuminated],
        'fit_z_mm':[float(Z_WIDE[i]*1e3) for i in train],
        'heldout_z_mm':[float(Z_WIDE[i]*1e3) for i in held],
        'full_model_refinement':{**refine_diag,'coefficient_table':_phase_coeff_table(coeff,truth_coeff),
            'phase_rms_to_truth_rad':phase_rms,'heldout':held_score,'truth_model_heldout':truth_held},
        'miao_after_physical_adjustment':miao_diag,
        'original_distorted':avg('original_distorted'),
        'after_physical_adjustment':avg('after_physical_adjustment'),
        'miao_after_physical_adjustment_metrics':avg('miao_only'),
        'full_model_refinement_metrics':avg('full_model_refinement'),
        'oracle_phase_only':avg('oracle_phase_only'),
    }
    summary['full_model_minus_miao']={
        'mean_pearson_r':summary['full_model_refinement_metrics']['mean_pearson_r']-summary['miao_after_physical_adjustment_metrics']['mean_pearson_r'],
        'mean_nrmse':summary['full_model_refinement_metrics']['mean_nrmse']-summary['miao_after_physical_adjustment_metrics']['mean_nrmse'],
    }
    np.save(out/'estimated_residual_phase_slm1_rad.npy',full_phase.astype(np.float32))
    np.save(out/'truth_residual_phase_slm1_rad.npy',residual.astype(np.float32))
    np.save(out/'miao_correction_axicon_input_rad.npy',miao_phase.astype(np.float32))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__': build()
