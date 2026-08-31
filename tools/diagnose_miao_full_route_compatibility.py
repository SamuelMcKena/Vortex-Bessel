"""Quantify where the Miao stationary-phase model departs from the full route.

The inverse modal optimizer has already been verified directly against Miao
et al. Eq. (3).  This diagnostic therefore asks the complementary question:
if the *true complex field* incident on the simulated axicon is known, how well
does the Miao stationary-phase expression predict the intensity produced by the
full angular-spectrum route at the corresponding z plane?

For each z, the stationary-phase annulus rho=z*k_perp/k is sampled from the true
axicon-input field.  The programmed q*theta phase is removed, Fourier
coefficients c_m are calculated directly (no inverse fitting), and Miao Eq. (3)
is evaluated on the same polar camera coordinates used for the measured/full
propagated field.  A discrepancy here is a forward-model approximation error,
not an optimizer error.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "notebooks" / "experimental" / "axicon_aberration_correction"
sys.path.insert(0, str(ROOT/"tools"))
sys.path.insert(0, str(MOD))

import benchmark_q20_miao_vs_digital_twin as base  # noqa: E402
from miao_full_retrieval import modal_basis, sample_polar  # noqa: E402
from vbb_study.digital_twin.vortex_system_route import SystemErrorConfig  # noqa: E402

OUT = ROOT / "outputs" / "validation" / "miao_full_route_compatibility"
Q = 20
GRID_N = 512
WINDOW_M = 8.0e-3
Z_M = np.linspace(32e-3, 132e-3, 7)
M_VALUES = np.arange(-30, 31, dtype=int)
N_THETA_INPUT = 720
N_R = 72
N_THETA_CAMERA = 144
RMAX_UM = 720.0
EPS = np.finfo(float).tiny

base.GRID_N = GRID_N
base.WINDOW_M = WINDOW_M


def _residual_phase(grid: dict) -> np.ndarray:
    X=np.asarray(grid["X"],float); Y=np.asarray(grid["Y"],float)
    R=np.hypot(X,Y); T=np.arctan2(Y,X)
    p=0.42*np.cos(2*T)+0.24*np.sin(3*T)+0.12*np.cos(5*T)
    return p*(1-np.exp(-(R/90e-6)**2))


def _sample_input_annulus(route: dict, rho_m: float, theta: np.ndarray) -> np.ndarray:
    field=np.asarray(route["field_on_axicon_plane"],np.complex128)
    x=np.asarray(route["grid"]["x"],float); dx=float(route["grid"]["dx"])
    xs=rho_m*np.cos(theta); ys=rho_m*np.sin(theta)
    xc=(xs-x[0])/dx; yc=(ys-x[0])/dx
    return ndimage.map_coordinates(field,[yc,xc],order=1,mode="constant",cval=0.0)


def _miao_prediction(route: dict, z_m: float, image: np.ndarray) -> dict:
    grid=route["grid"]; pixel=float(grid["dx"])
    kp=float(route["metadata"]["axicon"]["exact_kr_m_inv"])
    wavelength=float(route["metadata"]["wavelength_m"]); k=2*np.pi/wavelength
    rho=float(z_m)*kp/k

    theta_in=np.linspace(0,2*np.pi,N_THETA_INPUT,endpoint=False)
    input_ring=_sample_input_annulus(route,rho,theta_in)
    # The route programs +q*theta on SLM1; remove that known phase before the
    # c_m Fourier decomposition used by the q-residual form of Eq. (3).
    e_tilde=input_ring*np.exp(-1j*Q*theta_in)
    coeff=np.asarray([np.mean(e_tilde*np.exp(1j*int(m)*theta_in)) for m in M_VALUES],complex)

    center=(0.5*(image.shape[0]-1),0.5*(image.shape[1]-1))
    rmax_px=min(RMAX_UM*1e-6/pixel,0.47*min(image.shape))
    radii_px=np.linspace(1.5,rmax_px,N_R)
    theta_cam=np.linspace(0,2*np.pi,N_THETA_CAMERA,endpoint=False)
    measured=sample_polar(image,center,radii_px,theta_cam)
    rr,pp=np.meshgrid(radii_px*pixel,theta_cam,indexing="ij")
    B=modal_basis(Q,M_VALUES,kp,rr.ravel(),pp.ravel())
    pred=np.abs(B@coeff)**2
    data=np.asarray(measured,float).ravel()
    pred/=max(float(pred.max()),EPS); data/=max(float(data.max()),EPS)
    corr=float(np.corrcoef(pred,data)[0,1])
    rmse=float(np.sqrt(np.mean((pred-data)**2)))
    return {
        "z_mm":float(z_m*1e3),"rho_mm":rho*1e3,
        "pearson_r":corr,"rmse_peak_normalized":rmse,
        "input_annulus_mean_amplitude":float(np.mean(np.abs(input_ring))),
    }


def _case(label: str, route: dict) -> dict:
    stack=base._propagate(route,Z_M)
    rows=[_miao_prediction(route,float(z),stack[i]) for i,z in enumerate(Z_M)]
    return {
        "label":label,
        "rows":rows,
        "median_pearson_r":float(np.median([r["pearson_r"] for r in rows])),
        "mean_pearson_r":float(np.mean([r["pearson_r"] for r in rows])),
        "median_rmse":float(np.median([r["rmse_peak_normalized"] for r in rows])),
    }


def build(out: Path=OUT) -> dict:
    out.mkdir(parents=True,exist_ok=True)
    nominal=base._route(SystemErrorConfig(),None)
    residual=_residual_phase(nominal["grid"])
    aberrated=base._route(SystemErrorConfig(),residual)
    summary={
        "study":"Miao stationary-phase Eq3 compatibility with full q20 optical route",
        "interpretation":"coefficients are computed from the true axicon-input complex field; no inverse fitting is used",
        "grid":{"N":GRID_N,"window_mm":WINDOW_M*1e3,"dx_um":WINDOW_M/GRID_N*1e6},
        "nominal":_case("nominal q20 route",nominal),
        "known_residual_phase":_case("same route plus known m=2,3,5 input phase",aberrated),
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))
    return summary


if __name__=="__main__": build()
