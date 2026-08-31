"""Physics-separated q=20 correction benchmark.

This is a diagnostic successor to benchmark_q20_miao_vs_digital_twin.py.
It addresses two confounders exposed by v1:

1. Miao et al. map each focal-plane measurement to one annulus of the field
   incident on the axicon.  The z scan must therefore span the useful input
   aperture; a short scan cannot produce a full-aperture correction.
2. Unknown higher-order wavefront phase must not be allowed to masquerade as a
   physical translation/scale parameter.  Physical estimation is split into
   observables with the appropriate azimuthal symmetry before residual-phase
   retrieval is attempted.

The synthetic truth remains deliberately known.  An oracle phase correction is
also computed directly from the complex field at the axicon-input plane.  It is
not used by either tested method; it only diagnoses correction-plane/sign errors
and gives the phase-only upper bound for this controlled case.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
MIAO_DIR = ROOT / "notebooks" / "experimental" / "axicon_aberration_correction"
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))
if str(MIAO_DIR) not in sys.path:
    sys.path.insert(0, str(MIAO_DIR))

import benchmark_q20_miao_vs_digital_twin as v1  # noqa: E402
from miao_full_retrieval import assemble_full_aperture, fit_plane_adaptive, interpolate_to_cartesian  # noqa: E402
from vbb_study.digital_twin.hierarchical_physical_fit import hierarchical_physical_fit  # noqa: E402
from vbb_study.digital_twin.vortex_system_error_sweeps import system_sweep_registry  # noqa: E402
from vbb_study.digital_twin.vortex_system_route import SystemErrorConfig  # noqa: E402

OUT = ROOT / "outputs" / "poster" / "q20_method_comparison_v2"
Q = 20
# For the canonical 2-mm input beam and current axicon k_perp, this range maps
# approximately 0.4--1.9 mm of the axicon-input radius, rather than stopping at
# ~1.2 mm as v1 did.
Z_FIT_M = np.linspace(28e-3, 138e-3, 15)
Z_DISPLAY_M = np.linspace(28e-3, 138e-3, 111)
NOISE_SIGMA = 0.0015
SEED = 2031
FIT_CROP_HALF_M = 1.65e-3
METRIC_RADIUS_M = 0.90e-3
EPS = np.finfo(float).tiny

# Keep v1 propagation helpers on the same physical range.
v1.Z_FIT_M = Z_FIT_M
v1.Z_DISPLAY_M = Z_DISPLAY_M
v1.NOISE_SIGMA = NOISE_SIGMA
v1.FIT_CROP_HALF_M = FIT_CROP_HALF_M
v1.METRIC_RADIUS_M = METRIC_RADIUS_M

AXISYMMETRIC_PARAMETERS = ("beam_radius_scale", "fourf_iris_radius_scale")
LATERAL_PARAMETERS = (
    "beam_lateral_decentre_x",
    "slm1_hologram_offset_x",
    "fourf_iris_offset_x",
    "axicon_lateral_decentre_x",
)


def _threshold_weights(image: np.ndarray) -> np.ndarray:
    a = np.asarray(image, float)
    a = a / max(float(np.max(a)), EPS)
    # Suppress clipped positive noise floor while preserving rings.
    w = np.clip((a - 0.01) / 0.99, 0.0, 1.0)
    return w**1.35


def _centroid(image: np.ndarray) -> tuple[float, float]:
    w = _threshold_weights(image)
    yy, xx = np.indices(w.shape, dtype=float)
    s = max(float(np.sum(w)), EPS)
    return float(np.sum(yy*w)/s), float(np.sum(xx*w)/s)


def _radial_profile_about_own_centroid(image: np.ndarray, bins: int = 72) -> np.ndarray:
    a = np.asarray(image, float)
    a = a / max(float(np.max(a)), EPS)
    cy, cx = _centroid(a)
    yy, xx = np.indices(a.shape, dtype=float)
    rr = np.hypot(yy-cy, xx-cx)
    rmax = 0.47*min(a.shape)
    edges = np.linspace(0.0, rmax, bins+1)
    ids = np.clip(np.digitize(rr.ravel(), edges)-1, 0, bins-1)
    mask = rr.ravel() <= rmax
    sums = np.bincount(ids[mask], weights=a.ravel()[mask], minlength=bins)
    nums = np.bincount(ids[mask], minlength=bins)
    prof = sums/np.maximum(nums, 1)
    # Shape, not absolute power, is used here; physical throughput is a separate
    # observable and is not assumed measured in this benchmark.
    return prof/max(float(np.max(prof)), EPS)


def axisymmetric_shape_error(model: np.ndarray, data: np.ndarray) -> float:
    """Compare azimuthally averaged radial morphology after removing translation.

    Higher azimuthal phase terms are intentionally marginalized.  This is the
    appropriate observable for beam-radius / aperture-radius estimation.
    """
    mp = np.stack([_radial_profile_about_own_centroid(p) for p in np.asarray(model)])
    dp = np.stack([_radial_profile_about_own_centroid(p) for p in np.asarray(data)])
    return float(np.sqrt(np.mean((mp-dp)**2)))


def lateral_trajectory_error(model: np.ndarray, data: np.ndarray) -> float:
    """Compare only first-order lateral beam trajectory through the z stack.

    The imposed synthetic residual contains m=2,3,5 angular terms, so it should
    not be fitted as a translation.  Normalized centroid trajectories retain the
    m=1 information needed to discriminate lateral alignment parameters.
    """
    mc = np.asarray([_centroid(p) for p in np.asarray(model)], float)
    dc = np.asarray([_centroid(p) for p in np.asarray(data)], float)
    scale = 0.15*min(np.asarray(model).shape[-2:])
    # Include both absolute trajectory and its z-dependent slope/curvature.
    pos = np.sqrt(np.mean(((mc-dc)/scale)**2))
    if len(mc) > 2:
        dm = np.diff(mc, axis=0)
        dd = np.diff(dc, axis=0)
        slope = np.sqrt(np.mean(((dm-dd)/scale)**2))
    else:
        slope = 0.0
    return float(pos + 0.35*slope)


def _known_axis_center(route: dict) -> tuple[float, float]:
    n = int(np.asarray(route["grid"]["X"]).shape[0])
    c = 0.5*(n-1)
    return c, c


def _reference_rows(route: dict, retrievals: list, z_abs_m: np.ndarray) -> np.ndarray:
    field = np.asarray(route["field_on_axicon_plane"], np.complex128)
    intensity = np.abs(field)**2
    x = np.asarray(route["grid"]["x"], float)
    dx = float(route["grid"]["dx"])
    wavelength = float(route["metadata"]["wavelength_m"])
    k = 2*np.pi/wavelength
    kp = np.asarray([r.k_perp_m_inv for r in retrievals], float)
    rho = np.asarray(z_abs_m, float)*kp/k
    theta = np.asarray(retrievals[0].theta_rad, float)
    rows = []
    for rr in rho:
        xs = rr*np.cos(theta)
        ys = rr*np.sin(theta)
        xc = (xs-x[0])/dx
        yc = (ys-x[0])/dx
        rows.append(ndimage.map_coordinates(intensity, [yc, xc], order=1,
                                             mode="constant", cval=0.0))
    return np.asarray(rows, float)[np.argsort(rho)]


def miao_correction_calibrated_axis(route: dict, noisy_stack: np.ndarray) -> tuple[np.ndarray, dict]:
    """Miao retrieval using the known laboratory optical axis in the synthetic test.

    Miao et al. explicitly note that optical-axis calibration is required to
    recover first-order translation/coma terms.  Searching for the beam's own
    dark-core centre, as v1 did, discards that information and is not the fair
    synthetic analogue of a calibrated experiment.
    """
    grid = route["grid"]
    pixel = float(grid["dx"])
    nominal_kp = float(route["metadata"]["axicon"]["exact_kr_m_inv"])
    center = _known_axis_center(route)
    retrievals = []
    for i, (image, z) in enumerate(zip(np.asarray(noisy_stack, float), Z_FIT_M)):
        retrievals.append(fit_plane_adaptive(
            image, i, float(z), center, pixel, Q, nominal_kp,
            max_aberration_order=30,
            order_step=2,
            cost_threshold=0.05,
            min_fractional_improvement=0.006,
            rmax_um=720,
            n_r=52,
            n_theta=144,
        ))
    refs = _reference_rows(route, retrievals, Z_FIT_M)
    full = assemble_full_aperture(
        retrievals, Z_FIT_M, float(route["metadata"]["wavelength_m"]),
        k_perp_nominal_m_inv=nominal_kp,
        reference_intensity_rows=refs,
    )
    cart = interpolate_to_cartesian(full, grid_size=640, padding_fraction=0.02)
    correction, valid = v1._map_cartesian_phase_to_grid(cart, grid)
    return correction, {
        "branch": full.branch,
        "branch_score_direct": full.branch_score_direct,
        "branch_score_conjugate": full.branch_score_conjugate,
        "valid_grid_fraction": float(np.mean(valid)),
        "median_plane_fit_corr": float(np.median([r.fit_corr for r in retrievals])),
        "median_plane_fit_nrmse": float(np.median([r.fit_nrmse for r in retrievals])),
        "median_k_perp_m_inv": float(np.median([r.k_perp_m_inv for r in retrievals])),
        "median_mode_order": float(np.median([r.aberration_order_max for r in retrievals])),
        "rho_min_mm": float(np.min(full.rho_m)*1e3),
        "rho_max_mm": float(np.max(full.rho_m)*1e3),
    }


def _oracle_phase(route_with_residual: dict, route_without_residual: dict) -> tuple[np.ndarray, np.ndarray]:
    """Phase-only field-ratio correction at the actual axicon-input plane."""
    a = np.asarray(route_with_residual["field_on_axicon_plane"], np.complex128)
    b = np.asarray(route_without_residual["field_on_axicon_plane"], np.complex128)
    amp = np.maximum(np.abs(a), np.abs(b))
    valid = amp > 0.02*float(np.max(amp))
    corr = np.zeros(a.shape, float)
    corr[valid] = np.angle(b[valid]*np.conj(a[valid]))
    return corr, valid


def _circular_phase_rms(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    d = np.angle(np.exp(1j*(np.asarray(a)-np.asarray(b))))[valid]
    if d.size == 0:
        return float("nan")
    piston = np.angle(np.mean(np.exp(1j*d)))
    d = np.angle(np.exp(1j*(d-piston)))
    return float(np.sqrt(np.mean(d*d)))


def _metric_rows(z_m, ideal, distorted, miao, hybrid, oracle, grid):
    X = np.asarray(grid["X"], float)
    Y = np.asarray(grid["Y"], float)
    roi = np.hypot(X, Y) <= METRIC_RADIUS_M
    def met(a, b):
        av = np.asarray(a, float)[roi]; bv = np.asarray(b, float)[roi]
        av /= max(float(np.max(av)), EPS); bv /= max(float(np.max(bv)), EPS)
        return float(np.corrcoef(av, bv)[0,1]), float(np.sqrt(np.mean((av-bv)**2)))
    rows=[]
    for i,z in enumerate(np.asarray(z_m,float)):
        row={"z_mm":float(z*1e3)}
        for key,stack in (("distorted",distorted),("miao_only",miao),
                          ("digital_twin_plus_miao",hybrid),("oracle_phase_only",oracle)):
            c,e=met(stack[i],ideal[i]); row[f"{key}_pearson_r"]=c; row[f"{key}_nrmse"]=e
        rows.append(row)
    return pd.DataFrame(rows)


def _combine_fit_summary(axis_fit, lateral_fit) -> dict:
    selected = {}
    for fit in (axis_fit, lateral_fit):
        for step in fit.steps:
            if step.accepted and step.selected_family is not None:
                selected[step.selected_family] = float(step.selected_value)
    return selected


def _build_diagnostic_figure(out, grid, metrics, summary, ideal, distorted, miao, hybrid, oracle):
    fig, axs = plt.subplots(2, 4, figsize=(17.2, 8.5), facecolor=v1.BG, constrained_layout=True)
    for ax, stack, title in zip(axs[0], (distorted,miao,hybrid,oracle),
            ("Distorted", "Miao retrieval", "Digital twin + Miao", "Oracle phase-only")):
        v1._imshow_xz(ax, stack, grid, title)
    ax=axs[1,0]; v1._style(ax)
    for key,label in (("distorted","distorted"),("miao_only","Miao"),
                      ("digital_twin_plus_miao","digital twin + Miao"),("oracle_phase_only","oracle")):
        ax.plot(metrics.z_mm,metrics[f"{key}_pearson_r"],label=label,lw=1.4)
    ax.set(title="Agreement with nominal beam",xlabel="z from axicon (mm)",ylabel="Pearson r",ylim=(-.05,1.05)); ax.grid(alpha=.18); ax.legend(fontsize=7)
    ax=axs[1,1]; v1._style(ax)
    for key,label in (("distorted","distorted"),("miao_only","Miao"),
                      ("digital_twin_plus_miao","digital twin + Miao"),("oracle_phase_only","oracle")):
        ax.plot(metrics.z_mm,metrics[f"{key}_nrmse"],label=label,lw=1.4)
    ax.set(title="Normalized transverse RMSE",xlabel="z from axicon (mm)",ylabel="NRMSE"); ax.grid(alpha=.18)
    ax=axs[1,2]; v1._style(ax); ax.axis("off")
    ax.text(.03,.92,"Physical estimation",color=v1.FG,fontsize=11,weight="bold",va="top")
    y=.76
    for k,val in summary["physical_parameter_estimation"]["selected"].items():
        if "decentre" in k or "offset" in k:
            text=f"{k.replace('_',' ')}: {val*1e6:+.0f} µm"
        else: text=f"{k.replace('_',' ')}: {val:.3g}"
        ax.text(.03,y,text,color=v1.FG,fontsize=8.2); y-=.12
    ax.text(.03,.25,"axisymmetric fit: radial profile",color=v1.CYAN,fontsize=7.5)
    ax.text(.03,.14,"lateral fit: centroid trajectory",color=v1.CYAN,fontsize=7.5)
    ax=axs[1,3]; v1._style(ax); ax.axis("off")
    ax.text(.03,.92,"Retrieval diagnostics",color=v1.FG,fontsize=11,weight="bold",va="top")
    lines=[
        f"Miao phase RMS to oracle: {summary['miao_only']['phase_rms_to_oracle_rad']:.3f} rad",
        f"Hybrid phase RMS to oracle: {summary['digital_twin_plus_miao']['phase_rms_to_oracle_rad']:.3f} rad",
        f"Miao median focal fit r: {summary['miao_only']['median_plane_fit_corr']:.3f}",
        f"Hybrid median focal fit r: {summary['digital_twin_plus_miao']['median_plane_fit_corr']:.3f}",
        f"retrieved radius: {summary['digital_twin_plus_miao']['rho_min_mm']:.2f}–{summary['digital_twin_plus_miao']['rho_max_mm']:.2f} mm",
    ]
    y=.76
    for line in lines: ax.text(.03,y,line,color=v1.MUTED,fontsize=7.9); y-=.12
    fig.suptitle("q = 20 correction benchmark — physics diagnostics v2",color=v1.FG,fontsize=19,weight="bold")
    png=out/"q20_method_physics_v2.png"; pdf=out/"q20_method_physics_v2.pdf"
    fig.savefig(png,dpi=500,facecolor=v1.BG,bbox_inches="tight"); fig.savefig(pdf,facecolor=v1.BG,bbox_inches="tight"); plt.close(fig)
    with Image.open(png) as im:
        prev=im.convert("RGB"); prev.thumbnail((2600,1500),Image.Resampling.LANCZOS); preview=out/"q20_method_physics_v2.preview.jpg"; prev.save(preview,quality=92,subsampling=0)
    return png,pdf,preview


def build(out: Path = OUT) -> dict:
    out.mkdir(parents=True,exist_ok=True)
    registry=system_sweep_registry(); nominal=SystemErrorConfig()
    nominal_route=v1._route(nominal,None); residual=v1._residual_phase(nominal_route["grid"])
    truth=v1.apply_registry_family(nominal,"beam_radius_scale",0.85,registry=registry)
    truth=v1.apply_registry_family(truth,"axicon_lateral_decentre_x",250e-6,registry=registry)
    ideal_route=v1._route(nominal,None); distorted_route=v1._route(truth,residual)
    distorted_clean=v1._propagate(distorted_route,Z_FIT_M)
    rng=np.random.default_rng(SEED); noise=rng.normal(size=distorted_clean.shape); distorted=v1._add_noise(distorted_clean,noise)

    x=np.asarray(ideal_route["grid"]["x"],float); ids=np.flatnonzero(np.abs(x)<=FIT_CROP_HALF_M)
    target=distorted[:,ids[:,None],ids]; simulator=v1.PhysicalSimulator(ids)

    axis_fit=hierarchical_physical_fit(target_stack=target,simulate_config=simulator,
        families=AXISYMMETRIC_PARAMETERS,registry=registry,max_stages=1,
        min_improvement_fraction=0.002,loss_fn=axisymmetric_shape_error)
    lateral_fit=hierarchical_physical_fit(target_stack=target,simulate_config=simulator,
        initial_config=axis_fit.final_config,families=LATERAL_PARAMETERS,registry=registry,
        max_stages=1,min_improvement_fraction=0.002,loss_fn=lateral_trajectory_error)
    estimated=lateral_fit.final_config
    compensated=v1._compensate_physical_parameters(truth,estimated)
    compensated_route=v1._route(compensated,residual)
    compensated_clean=v1._propagate(compensated_route,Z_FIT_M)
    compensated_noisy=v1._add_noise(compensated_clean,noise)

    miao_phase,miao_diag=miao_correction_calibrated_axis(distorted_route,distorted)
    hybrid_phase,hybrid_diag=miao_correction_calibrated_axis(compensated_route,compensated_noisy)

    # Oracle uses the same compensated physical system, with and without the
    # imposed residual, to isolate phase-retrieval/application errors.
    compensated_nores=v1._route(compensated,None)
    oracle_phase,oracle_valid=_oracle_phase(compensated_route,compensated_nores)

    ideal_disp=v1._propagate(ideal_route,Z_DISPLAY_M)
    distorted_disp=v1._propagate(distorted_route,Z_DISPLAY_M)
    miao_disp=v1._propagate(distorted_route,Z_DISPLAY_M,miao_phase)
    hybrid_disp=v1._propagate(compensated_route,Z_DISPLAY_M,hybrid_phase)
    oracle_disp=v1._propagate(compensated_route,Z_DISPLAY_M,oracle_phase)
    metrics=_metric_rows(Z_DISPLAY_M,ideal_disp,distorted_disp,miao_disp,hybrid_disp,oracle_disp,ideal_route["grid"])
    metrics.to_csv(out/"comparison_metrics_vs_z.csv",index=False)

    selected=_combine_fit_summary(axis_fit,lateral_fit)
    # Compare retrieved correction with the physically exact phase-only field
    # ratio on the same illuminated pixels.  This is diagnostic only.
    mvalid=oracle_valid & (np.abs(miao_phase)>0)
    hvalid=oracle_valid & (np.abs(hybrid_phase)>0)
    summary={
        "study":"physics-separated q20 Miao versus digital-twin-assisted Miao benchmark v2",
        "truth":{"beam_radius_scale":0.85,"axicon_lateral_decentre_x_m":250e-6,
                 "residual_phase":"0.42 cos(2theta)+0.24 sin(3theta)+0.12 cos(5theta)"},
        "z_sampling":{"fit_planes":len(Z_FIT_M),"z_min_mm":float(Z_FIT_M[0]*1e3),"z_max_mm":float(Z_FIT_M[-1]*1e3)},
        "physical_parameter_estimation":{"selected":selected,
            "axisymmetric_fit":axis_fit.as_dict(),"lateral_fit":lateral_fit.as_dict(),
            "method":"axisymmetric radial morphology followed by first-order centroid trajectory"},
        "miao_only":{**miao_diag,"phase_rms_to_oracle_rad":_circular_phase_rms(miao_phase,oracle_phase,mvalid)},
        "digital_twin_plus_miao":{**hybrid_diag,"phase_rms_to_oracle_rad":_circular_phase_rms(hybrid_phase,oracle_phase,hvalid)},
    }
    for key,col in (("aberrated","distorted"),("miao_only","miao_only"),
                    ("digital_twin_plus_miao","digital_twin_plus_miao"),("oracle_phase_only","oracle_phase_only")):
        summary.setdefault(key,{})
        summary[key].update({"mean_pearson_r":float(metrics[f"{col}_pearson_r"].mean()),
                             "mean_nrmse":float(metrics[f"{col}_nrmse"].mean())})
    summary["hybrid_minus_miao"]={"mean_pearson_r":summary["digital_twin_plus_miao"]["mean_pearson_r"]-summary["miao_only"]["mean_pearson_r"],
                                  "mean_nrmse":summary["digital_twin_plus_miao"]["mean_nrmse"]-summary["miao_only"]["mean_nrmse"]}
    summary["hybrid_minus_aberrated"]={"mean_pearson_r":summary["digital_twin_plus_miao"]["mean_pearson_r"]-summary["aberrated"]["mean_pearson_r"],
                                       "mean_nrmse":summary["digital_twin_plus_miao"]["mean_nrmse"]-summary["aberrated"]["mean_nrmse"]}

    np.save(out/"miao_correction_rad.npy",miao_phase.astype(np.float32)); np.save(out/"hybrid_correction_rad.npy",hybrid_phase.astype(np.float32)); np.save(out/"oracle_correction_rad.npy",oracle_phase.astype(np.float32))
    png,pdf,preview=_build_diagnostic_figure(out,ideal_route["grid"],metrics,summary,ideal_disp,distorted_disp,miao_disp,hybrid_disp,oracle_disp)
    with Image.open(png) as im: summary["assets"]={"png":str(png),"pdf":str(pdf),"preview":str(preview),"pixel_size":list(im.size),"dpi":list(im.info.get("dpi",(0,0)))}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2)); return summary


if __name__=="__main__": build()
