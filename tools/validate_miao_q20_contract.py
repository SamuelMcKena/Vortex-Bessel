"""Synthetic contract test for the Miao q=20 retrieval implementation.

A known non-axisymmetric phase is applied directly to the field incident on the
axicon, matching the physical object reconstructed in Miao et al.  No beam/SLM/
4F/axicon parameter errors are present.  The test therefore isolates the Miao
retrieval, annulus mapping, correction sign and correction application plane from
the digital-twin parameter-estimation problem.
"""
from __future__ import annotations
from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import benchmark_q20_miao_vs_digital_twin as base  # noqa: E402
import benchmark_q20_method_physics_v2 as phys  # noqa: E402
from vbb_study.digital_twin.vortex_system_route import SystemErrorConfig  # noqa: E402

OUT=ROOT/'outputs'/'validation'/'miao_q20_contract'
Z_FIT=np.linspace(28e-3,138e-3,15)
Z_EVAL=np.linspace(28e-3,138e-3,111)
EPS=np.finfo(float).tiny


def phase_truth(grid):
    X=np.asarray(grid['X'],float); Y=np.asarray(grid['Y'],float)
    R=np.hypot(X,Y); T=np.arctan2(Y,X)
    p=0.42*np.cos(2*T)+0.24*np.sin(3*T)+0.12*np.cos(5*T)
    p*=1-np.exp(-(R/90e-6)**2)
    return p


def phase_rms(a,b,mask):
    d=np.angle(np.exp(1j*(np.asarray(a)-np.asarray(b))))[mask]
    piston=np.angle(np.mean(np.exp(1j*d))) if d.size else 0.0
    d=np.angle(np.exp(1j*(d-piston)))
    return float(np.sqrt(np.mean(d*d))) if d.size else float('nan')


def metrics(ideal,test,grid):
    X=np.asarray(grid['X']); Y=np.asarray(grid['Y']); roi=np.hypot(X,Y)<=0.9e-3
    cs=[]; es=[]
    for a,b in zip(test,ideal):
        av=np.asarray(a)[roi]; bv=np.asarray(b)[roi]
        av=av/max(float(av.max()),EPS); bv=bv/max(float(bv.max()),EPS)
        cs.append(float(np.corrcoef(av,bv)[0,1])); es.append(float(np.sqrt(np.mean((av-bv)**2))))
    return float(np.mean(cs)),float(np.mean(es))


def build(out=OUT):
    out.mkdir(parents=True,exist_ok=True)
    # Preserve v2 scan geometry but use a larger alias-safe numerical window.
    base.GRID_N=512; base.WINDOW_M=8e-3; base.Z_FIT_M=Z_FIT; base.Z_DISPLAY_M=Z_EVAL
    phys.Z_FIT_M=Z_FIT; phys.Z_DISPLAY_M=Z_EVAL

    route=base._route(SystemErrorConfig(),None)
    truth=phase_truth(route['grid'])
    ideal_fit=base._propagate(route,Z_FIT)
    aberr_fit_clean=base._propagate(route,Z_FIT,truth)
    rng=np.random.default_rng(2032); noise=rng.normal(size=aberr_fit_clean.shape)
    base.NOISE_SIGMA=0.0015
    aberr_fit=base._add_noise(aberr_fit_clean,noise)

    correction,diag=phys.miao_correction_calibrated_axis(route,aberr_fit)
    ideal=base._propagate(route,Z_EVAL)
    aberr=base._propagate(route,Z_EVAL,truth)
    corrected=base._propagate(route,Z_EVAL,truth+correction)
    oracle=base._propagate(route,Z_EVAL,np.zeros_like(truth))

    amp=np.abs(np.asarray(route['field_on_axicon_plane']))
    illum=amp>0.02*float(amp.max())
    retrieved=np.abs(correction)>0
    valid=illum&retrieved
    corr0,err0=metrics(ideal,aberr,route['grid'])
    corr1,err1=metrics(ideal,corrected,route['grid'])
    corro,erro=metrics(ideal,oracle,route['grid'])
    rms=phase_rms(correction,-truth,valid)
    # Sign diagnostic is not used to choose the answer; it identifies a convention
    # error if the implemented conjugate is systematically reversed.
    rms_wrong=phase_rms(-correction,-truth,valid)
    summary={
        'study':'Miao q20 synthetic contract validation',
        'phase_truth':'0.42 cos2theta + 0.24 sin3theta + 0.12 cos5theta at axicon input',
        'fit_z_mm':[float(Z_FIT[0]*1e3),float(Z_FIT[-1]*1e3)],
        'retrieval':diag,
        'phase_rms_to_exact_conjugate_rad':rms,
        'phase_rms_if_sign_reversed_rad':rms_wrong,
        'aberrated':{'mean_pearson_r':corr0,'mean_nrmse':err0},
        'retrieved_correction':{'mean_pearson_r':corr1,'mean_nrmse':err1},
        'oracle':{'mean_pearson_r':corro,'mean_nrmse':erro},
        'improvement':{'pearson_r':corr1-corr0,'nrmse':err1-err0},
    }
    np.save(out/'truth_phase_rad.npy',truth.astype(np.float32)); np.save(out/'retrieved_correction_rad.npy',correction.astype(np.float32))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2)); return summary


if __name__=='__main__': build()
