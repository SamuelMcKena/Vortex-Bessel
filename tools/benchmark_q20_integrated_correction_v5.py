"""Integrated q=20 correction benchmark using physically observable system errors.

This replaces the earlier synthetic beam-radius requirement.  The canonical
hardware manifest already treats the SLM-plane Gaussian radius as a calibration
quantity, so it is not scientifically sensible to require a downstream z-stack
to identify it uniquely in the presence of unknown phase.

The controlled hidden state used here is instead:

* Fourier-plane iris radius = 0.85 x nominal;
* axicon lateral displacement = +250 um;
* unknown residual phase = 0.42 cos(2theta) + 0.24 sin(3theta)
  + 0.12 cos(5theta).

These physical perturbations are deliberately chosen because they act on
different observables available in the camera propagation scan: iris opening
changes the centered radial morphology, whereas axicon displacement changes the
beam trajectory.  The algorithm then compares three correction paths:

A. Miao et al. residual-phase retrieval applied directly to the distorted stack;
B. digital-twin physical parameter estimation -> physical adjustment -> the
   same Miao retrieval on a new stack;
C. the same physical adjustment -> residual phase estimated by propagating
   candidates through the complete digital twin, with alternating illuminated
   z planes held out from the phase fit.

This is a synthetic truth-controlled method study.  No optimizer is allowed to
use the hidden parameter values or hidden phase coefficients.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'tools') not in sys.path:
    sys.path.insert(0, str(ROOT/'tools'))

import benchmark_q20_miao_vs_digital_twin as base  # noqa: E402
import benchmark_q20_method_physics_v2 as p2  # noqa: E402
import benchmark_q20_method_physics_v3 as p3  # noqa: E402
import benchmark_q20_method_physics_v4 as p4  # noqa: E402
import benchmark_q20_full_model_phase_refinement_v1 as full  # noqa: E402
from vbb_study.digital_twin.hierarchical_physical_fit import hierarchical_physical_fit  # noqa: E402
from vbb_study.digital_twin.physical_observable_fit import (  # noqa: E402
    axisymmetric_radial_morphology,
    centroid_trajectory,
)
from vbb_study.digital_twin.vortex_system_error_sweeps import system_sweep_registry  # noqa: E402
from vbb_study.digital_twin.vortex_system_route import SystemErrorConfig  # noqa: E402

OUT = ROOT/'outputs'/'validation'/'q20_integrated_correction_v5'
Z_FIT = p3.Z_FIT_M
Z_EVAL = p3.Z_DISPLAY_M
GRID_N = 512
WINDOW_M = 8e-3
PHYSICAL_CROP_HALF_M = 1.65e-3
NOISE_SIGMA = 0.0015
SEED = 2041
EPS = np.finfo(float).tiny

# One shared numerical contract for all imported helpers.
base.GRID_N = GRID_N; base.WINDOW_M = WINDOW_M
base.Z_FIT_M = Z_FIT; base.Z_DISPLAY_M = Z_EVAL
base.NOISE_SIGMA = NOISE_SIGMA; base.FIT_CROP_HALF_M = PHYSICAL_CROP_HALF_M
base.METRIC_RADIUS_M = p2.METRIC_RADIUS_M
p2.Z_FIT_M = Z_FIT; p2.Z_DISPLAY_M = Z_EVAL
p4.Z_FIT_M = Z_FIT; p4.Z_DISPLAY_M = Z_EVAL
full.Z_WIDE = Z_FIT; full.Z_EVAL = Z_EVAL


def _truth_phase(grid: dict) -> tuple[np.ndarray, np.ndarray]:
    coeff = full._truth_coefficients()
    return full.phase_from_coefficients(grid, coeff), coeff


def _selected(fit) -> dict[str, float]:
    out = {}
    for step in fit.steps:
        if step.accepted and step.selected_family is not None:
            out[str(step.selected_family)] = float(step.selected_value)
    return out


def _physical_fit(target_crop: np.ndarray, simulator, registry) -> tuple[SystemErrorConfig, dict]:
    nominal = SystemErrorConfig()
    # Aperture opening first: centered azimuthal morphology discards translation
    # and strongly suppresses the non-axisymmetric residual phase.
    iris = hierarchical_physical_fit(
        target_stack=target_crop,
        simulate_config=simulator,
        initial_config=nominal,
        families=('fourf_iris_radius_scale',),
        registry=registry,
        max_stages=1,
        min_improvement_fraction=0.004,
        loss_fn=axisymmetric_radial_morphology,
    )
    # Then estimate lateral optical displacement from the first spatial moment
    # versus z while keeping the inferred aperture state fixed.
    lateral = hierarchical_physical_fit(
        target_stack=target_crop,
        simulate_config=simulator,
        initial_config=iris.final_config,
        families=p2.LATERAL_PARAMETERS,
        registry=registry,
        max_stages=1,
        min_improvement_fraction=0.004,
        loss_fn=centroid_trajectory,
    )
    selected = {}
    selected.update(_selected(iris)); selected.update(_selected(lateral))
    return lateral.final_config, {
        'selected': selected,
        'iris_fit': iris.as_dict(),
        'lateral_fit': lateral.as_dict(),
        'measurement_assignment': {
            'fourf_iris_radius_scale': 'azimuthally averaged transverse intensity across z',
            'axicon_lateral_decentre_x': 'intensity-centroid trajectory across z',
        },
    }


def _averages(metrics, prefix: str) -> dict:
    return {
        'mean_pearson_r': float(metrics[f'{prefix}_pearson_r'].mean()),
        'mean_nrmse': float(metrics[f'{prefix}_nrmse'].mean()),
    }


def build(out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    registry = system_sweep_registry()
    nominal = SystemErrorConfig()
    ideal_route = base._route(nominal, None)
    residual, truth_coeff = _truth_phase(ideal_route['grid'])

    truth = base.apply_registry_family(nominal, 'fourf_iris_radius_scale', 0.85, registry=registry)
    truth = base.apply_registry_family(truth, 'axicon_lateral_decentre_x', 250e-6, registry=registry)
    distorted_route = base._route(truth, residual)
    distorted_clean = base._propagate(distorted_route, Z_FIT)
    rng = np.random.default_rng(SEED)
    distorted_measured = base._add_noise(distorted_clean, rng.normal(size=distorted_clean.shape))

    x = np.asarray(ideal_route['grid']['x'], float)
    crop = np.flatnonzero(np.abs(x) <= PHYSICAL_CROP_HALF_M)
    target_crop = distorted_measured[:, crop[:, None], crop]
    simulator = base.PhysicalSimulator(crop)
    estimated, physical_diag = _physical_fit(target_crop, simulator, registry)

    # Apply the inferred physical changes to the hidden system, then acquire a
    # genuinely new noisy stack before residual-phase correction.
    adjusted = base._compensate_physical_parameters(truth, estimated)
    adjusted_route = base._route(adjusted, residual)
    adjusted_clean = base._propagate(adjusted_route, Z_FIT)
    rng2 = np.random.default_rng(SEED+1)
    adjusted_measured = base._add_noise(adjusted_clean, rng2.normal(size=adjusted_clean.shape))

    # Paper baseline: apply Miao retrieval directly to the original distorted
    # stack, using only stationary-phase annuli with appreciable input power.
    direct_ids, direct_illum = p4.illuminated_plane_indices(distorted_route, Z_FIT)
    miao_direct_phase, miao_direct_diag = p4.miao_correction_selected(
        distorted_route, distorted_measured, Z_FIT, direct_ids)

    # Model-assisted paper path: same Miao implementation after the inferred
    # physical adjustment and a new independent z-stack.
    adjusted_ids, adjusted_illum = p4.illuminated_plane_indices(adjusted_route, Z_FIT)
    miao_hybrid_phase, miao_hybrid_diag = p4.miao_correction_selected(
        adjusted_route, adjusted_measured, Z_FIT, adjusted_ids)

    # Full digital-twin residual fit on alternating illuminated planes, with the
    # other planes withheld until after optimization.
    train = adjusted_ids[::2]
    held = adjusted_ids[1::2]
    if len(held) < 2:
        train = adjusted_ids[:-2]; held = adjusted_ids[-2:]
    coeff, refine_diag = full.refine_phase_full_model(
        adjusted, adjusted_measured, Z_FIT, train)
    estimated_phase = full.phase_from_coefficients(ideal_route['grid'], coeff)
    held_score = full._heldout_model_score(adjusted, coeff, adjusted_measured, Z_FIT, held)
    truth_held = full._heldout_model_score(adjusted, truth_coeff, adjusted_measured, Z_FIT, held)

    full_corrected_route = base._route(adjusted, residual-estimated_phase)

    # Evaluation is against the independently generated nominal route on a much
    # denser z grid than was used by either inverse stage.
    ideal_eval = base._propagate(ideal_route, Z_EVAL)
    distorted_eval = base._propagate(distorted_route, Z_EVAL)
    adjusted_eval = base._propagate(adjusted_route, Z_EVAL)
    miao_direct_eval = base._propagate(distorted_route, Z_EVAL, miao_direct_phase)
    miao_hybrid_eval = base._propagate(adjusted_route, Z_EVAL, miao_hybrid_phase)
    full_eval = base._propagate(full_corrected_route, Z_EVAL)

    # Build one explicit metric table rather than overloading old column names.
    grid = ideal_route['grid']; X=np.asarray(grid['X'],float); Y=np.asarray(grid['Y'],float)
    roi=np.hypot(X,Y) <= p2.METRIC_RADIUS_M
    def pair(a,b):
        av=np.asarray(a,float)[roi]; bv=np.asarray(b,float)[roi]
        av/=max(float(np.max(av)),EPS); bv/=max(float(np.max(bv)),EPS)
        return float(np.corrcoef(av,bv)[0,1]), float(np.sqrt(np.mean((av-bv)**2)))
    rows=[]
    stacks={
        'distorted':distorted_eval,
        'physical_adjustment_only':adjusted_eval,
        'miao_only':miao_direct_eval,
        'physical_fit_plus_miao':miao_hybrid_eval,
        'physical_fit_plus_full_model':full_eval,
    }
    for iz,z in enumerate(Z_EVAL):
        row={'z_mm':float(z*1e3)}
        for name,stack in stacks.items():
            r,e=pair(stack[iz],ideal_eval[iz]); row[f'{name}_pearson_r']=r; row[f'{name}_nrmse']=e
        rows.append(row)
    import pandas as pd
    metrics=pd.DataFrame(rows); metrics.to_csv(out/'metrics_vs_z.csv',index=False)

    phase_delta=np.angle(np.exp(1j*(estimated_phase-residual)))
    phase_rms=float(np.sqrt(np.mean(phase_delta**2)))
    summary={
        'study':'integrated q20 physical-model and residual-wavefront correction v5',
        'truth':{
            'fourf_iris_radius_scale':0.85,
            'axicon_lateral_decentre_x_m':250e-6,
            'residual_phase_coefficients':full._phase_coeff_table(truth_coeff,truth_coeff),
        },
        'physical_parameter_estimation':physical_diag,
        'retrieval_plane_selection':{
            'criterion':'stationary-phase input-annulus amplitude >= 0.15 of scan maximum',
            'direct_miao_z_mm':[float(Z_FIT[i]*1e3) for i in direct_ids],
            'model_assisted_z_mm':[float(Z_FIT[i]*1e3) for i in adjusted_ids],
            'full_model_train_z_mm':[float(Z_FIT[i]*1e3) for i in train],
            'full_model_heldout_z_mm':[float(Z_FIT[i]*1e3) for i in held],
            'direct_relative_annulus_amplitude':[float(v) for v in direct_illum],
            'adjusted_relative_annulus_amplitude':[float(v) for v in adjusted_illum],
        },
        'miao_only':miao_direct_diag,
        'physical_fit_plus_miao':miao_hybrid_diag,
        'full_model_residual_fit':{
            **refine_diag,
            'phase_rms_to_truth_rad':phase_rms,
            'coefficient_table':full._phase_coeff_table(coeff,truth_coeff),
            'heldout':held_score,
            'truth_model_heldout':truth_held,
        },
    }
    for name in stacks:
        summary[name]={
            'mean_pearson_r':float(metrics[f'{name}_pearson_r'].mean()),
            'mean_nrmse':float(metrics[f'{name}_nrmse'].mean()),
        }
    summary['gain_full_model_over_miao_only']={
        'mean_pearson_r':summary['physical_fit_plus_full_model']['mean_pearson_r']-summary['miao_only']['mean_pearson_r'] if 'mean_pearson_r' in summary['miao_only'] else summary['physical_fit_plus_full_model']['mean_pearson_r']-float(metrics.miao_only_pearson_r.mean()),
        'mean_nrmse':summary['physical_fit_plus_full_model']['mean_nrmse']-float(metrics.miao_only_nrmse.mean()),
    }
    # miao diagnostics and metric summaries are intentionally separate.
    summary['miao_only_metrics']={'mean_pearson_r':float(metrics.miao_only_pearson_r.mean()),'mean_nrmse':float(metrics.miao_only_nrmse.mean())}
    summary['physical_fit_plus_miao_metrics']={'mean_pearson_r':float(metrics.physical_fit_plus_miao_pearson_r.mean()),'mean_nrmse':float(metrics.physical_fit_plus_miao_nrmse.mean())}

    np.save(out/'estimated_residual_phase_rad.npy',estimated_phase.astype(np.float32))
    np.save(out/'truth_residual_phase_rad.npy',residual.astype(np.float32))
    np.save(out/'miao_only_correction_rad.npy',miao_direct_phase.astype(np.float32))
    np.save(out/'physical_fit_plus_miao_correction_rad.npy',miao_hybrid_phase.astype(np.float32))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    build()
