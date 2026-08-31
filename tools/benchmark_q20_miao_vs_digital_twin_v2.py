"""Refined controlled q=20 method comparison.

This version addresses two problems exposed by the first comparison run:

1. the second injected physical perturbation is a 4F iris-radius error rather
   than a beam-radius error, so it cannot be easily mimicked by the unknown
   angular wavefront phase; and
2. the Miao input-plane correction is given one synthetic known-sign/parity
   calibration before either method is compared.  The same fixed mapping is
   then used for the Miao-only and digital-twin-assisted branches.

The comparison remains falsifiable: no CI assertion requires the digital-twin
route to outperform the paper method.
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

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
MIAO_DIR = ROOT / "notebooks" / "experimental" / "axicon_aberration_correction"
for p in (TOOLS, MIAO_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import benchmark_q20_miao_vs_digital_twin as base  # noqa: E402
from miao_full_retrieval import (  # noqa: E402
    assemble_full_aperture,
    fit_plane_adaptive,
    interpolate_to_cartesian,
)
from modal_vortex_bessel import find_dark_core_center  # noqa: E402
from vbb_study.digital_twin.hierarchical_physical_fit import (  # noqa: E402
    apply_registry_family,
    hierarchical_physical_fit,
)
from vbb_study.digital_twin.vortex_system_error_sweeps import system_sweep_registry  # noqa: E402
from vbb_study.digital_twin.vortex_system_route import SystemErrorConfig  # noqa: E402

OUT = ROOT / "outputs" / "poster" / "q20_method_comparison_v2"
Q = base.Q
Z_FIT_M = base.Z_FIT_M
Z_DISPLAY_M = base.Z_DISPLAY_M
NOISE_SIGMA = 0.0010
SEED = 2042
FIT_PARAMETERS = base.FIT_PARAMETERS
EPS = base.EPS

# Keep the shared helpers on exactly the same numerical settings as this study.
base.NOISE_SIGMA = NOISE_SIGMA
base.OUT = OUT


def residual_phase(grid: dict) -> np.ndarray:
    X = np.asarray(grid["X"], float)
    Y = np.asarray(grid["Y"], float)
    R = np.hypot(X, Y)
    TH = np.arctan2(Y, X)
    phase = 0.30*np.cos(2.0*TH) + 0.18*np.sin(3.0*TH) + 0.09*np.cos(5.0*TH)
    phase *= 1.0 - np.exp(-(R/90e-6)**2)
    return np.asarray(phase, float)


def retrieve_raw_correction(route: dict, noisy_stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run the authoritative Miao retrieval and keep its sampled-annulus mask."""
    grid = route["grid"]
    pixel = float(grid["dx"])
    nominal_kp = float(route["metadata"]["axicon"]["exact_kr_m_inv"])
    retrievals = []
    for i, (image, z) in enumerate(zip(np.asarray(noisy_stack, float), Z_FIT_M)):
        cy, cx, _ = find_dark_core_center(image)
        retrievals.append(fit_plane_adaptive(
            image,
            i,
            float(z),
            (float(cy), float(cx)),
            pixel,
            Q,
            nominal_kp,
            max_aberration_order=14,
            order_step=2,
            cost_threshold=0.035,
            min_fractional_improvement=0.006,
            rmax_um=440,
            n_r=42,
            n_theta=96,
        ))
    refs = base._reference_rows(route, retrievals, Z_FIT_M)
    full = assemble_full_aperture(
        retrievals,
        Z_FIT_M,
        float(route["metadata"]["wavelength_m"]),
        k_perp_nominal_m_inv=nominal_kp,
        reference_intensity_rows=refs,
    )
    cart = interpolate_to_cartesian(full, grid_size=512, padding_fraction=0.03)
    correction, valid = base._map_cartesian_phase_to_grid(cart, grid)
    diag = {
        "branch_from_intensity_reference": full.branch,
        "branch_score_direct": full.branch_score_direct,
        "branch_score_conjugate": full.branch_score_conjugate,
        "sampled_annulus_grid_fraction": float(np.mean(valid)),
        "rho_min_um": float(np.min(full.rho_m)*1e6),
        "rho_max_um": float(np.max(full.rho_m)*1e6),
        "median_plane_fit_corr": float(np.median([r.fit_corr for r in retrievals])),
        "median_plane_fit_nrmse": float(np.median([r.fit_nrmse for r in retrievals])),
        "median_k_perp_m_inv": float(np.median([r.k_perp_m_inv for r in retrievals])),
    }
    return correction, valid, diag


def transform_phase(phase: np.ndarray, name: str) -> np.ndarray:
    p = np.asarray(phase, float)
    if name == "identity":
        return p
    if name == "opposite_sign":
        return -p
    if name == "rotate_180":
        return np.rot90(p, 2)
    if name == "opposite_sign_rotate_180":
        return -np.rot90(p, 2)
    raise KeyError(name)


def effective_phase_at_axicon(reference_route: dict, aberrated_route: dict) -> tuple[np.ndarray, np.ndarray]:
    u0 = np.asarray(reference_route["field_on_axicon_plane"], np.complex128)
    u1 = np.asarray(aberrated_route["field_on_axicon_plane"], np.complex128)
    cross = u1*np.conj(u0)
    phase = np.angle(cross)
    amp = np.minimum(np.abs(u0), np.abs(u1))
    mask = amp >= 0.08*float(np.max(amp))
    return phase, mask


def circular_rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    d = np.angle(np.exp(1j*(np.asarray(a, float)-np.asarray(b, float))))
    return float(np.sqrt(np.mean(d[np.asarray(mask, bool)]**2)))


def calibrate_mapping(nominal_route: dict, phase_truth_slm: np.ndarray) -> tuple[str, dict]:
    """Resolve the synthetic sign/180° convention once using a known phase test."""
    calibration_route = base._route(SystemErrorConfig(), phase_truth_slm)
    stack = base._propagate(calibration_route, Z_FIT_M)
    raw, valid, diag = retrieve_raw_correction(calibration_route, stack)
    effective, amp_mask = effective_phase_at_axicon(nominal_route, calibration_route)
    desired = -effective
    names = ("identity", "opposite_sign", "rotate_180", "opposite_sign_rotate_180")
    scores = {}
    mask = valid & amp_mask
    for name in names:
        scores[name] = circular_rmse(transform_phase(raw, name), desired, mask)
    selected = min(scores, key=scores.get)
    return selected, {
        "purpose": "known-sign synthetic input-plane phase test; fixed before method comparison",
        "selected_transform": selected,
        "circular_phase_rmse_rad_by_transform": scores,
        "retrieval": diag,
        "comparison_pixels": int(np.sum(mask)),
    }


def metric_table(z_m: np.ndarray, ideal: np.ndarray, stages: dict[str, np.ndarray], grid: dict) -> pd.DataFrame:
    X = np.asarray(grid["X"], float)
    Y = np.asarray(grid["Y"], float)
    roi = np.hypot(X, Y) <= base.METRIC_RADIUS_M

    def one(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        av = np.asarray(a, float)[roi]
        bv = np.asarray(b, float)[roi]
        av = av/max(float(np.max(av)), EPS)
        bv = bv/max(float(np.max(bv)), EPS)
        return float(np.corrcoef(av, bv)[0, 1]), float(np.sqrt(np.mean((av-bv)**2)))

    rows = []
    for iz, z in enumerate(z_m):
        row = {"z_mm": float(z*1e3)}
        for key, stack in stages.items():
            r, e = one(stack[iz], ideal[iz])
            row[f"{key}_pearson_r"] = r
            row[f"{key}_nrmse"] = e
        rows.append(row)
    return pd.DataFrame(rows)


def fit_display(fit) -> list[str]:
    out = []
    for step in fit.steps:
        if not step.accepted or step.selected_family is None:
            continue
        v = float(step.selected_value)
        if step.selected_family == "axicon_lateral_decentre_x":
            out.append(f"Axicon lateral position: {v*1e6:+.0f} µm")
        elif step.selected_family == "fourf_iris_radius_scale":
            out.append(f"Fourier iris radius: {v:.2f} × nominal")
        elif step.selected_family == "fourf_iris_offset_x":
            out.append(f"Fourier iris position: {v*1e3:+.2f} mm")
        elif step.selected_family == "slm1_hologram_offset_x":
            out.append(f"SLM1 pattern position: {v*1e6:+.0f} µm")
        elif step.selected_family == "beam_radius_scale":
            out.append(f"Input beam radius: {v:.2f} × nominal")
        elif step.selected_family == "beam_lateral_decentre_x":
            out.append(f"Input beam position: {v*1e6:+.0f} µm")
        else:
            out.append(f"{step.selected_family}: {v:g}")
    return out


def summary_method(metrics: pd.DataFrame, prefix: str) -> dict:
    return {
        "mean_pearson_r": float(metrics[f"{prefix}_pearson_r"].mean()),
        "median_pearson_r": float(metrics[f"{prefix}_pearson_r"].median()),
        "mean_nrmse": float(metrics[f"{prefix}_nrmse"].mean()),
        "median_nrmse": float(metrics[f"{prefix}_nrmse"].median()),
    }


def build_figure(out: Path, grid: dict, fit, metrics: pd.DataFrame,
                 ideal: np.ndarray, distorted: np.ndarray, aligned: np.ndarray,
                 miao: np.ndarray, hybrid: np.ndarray, summary: dict) -> tuple[Path, Path, Path]:
    fig = plt.figure(figsize=(17.3, 10.0), facecolor=base.BG)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.02, .72, .78],
                          left=.045, right=.985, bottom=.075, top=.82,
                          hspace=.31, wspace=.18)

    for col, (stack, title) in enumerate(((distorted, "Aberrated"),
                                          (miao, "Miao et al."),
                                          (hybrid, "Digital twin + Miao"),
                                          (ideal, "Nominal q = 20"))):
        base._imshow_xz(fig.add_subplot(gs[0, col]), stack, grid, title)

    ax = fig.add_subplot(gs[1, 0]); base._style(ax)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Physical parameter estimation", fontsize=10.4, weight="bold", pad=7)
    ax.text(.055, .82, "Estimated directly from the intensity z-stack", color=base.MUTED, fontsize=7.7)
    y = .62
    for line in fit_display(fit)[:3]:
        ax.text(.06, y, line, color=base.FG, fontsize=8.4, weight="bold")
        y -= .17
    ax.text(.06, .14, r"$E_I=\langle\mathrm{RMSE}(\tilde I_{model},\tilde I_{data})\rangle_z$",
            color=base.CYAN, fontsize=8.1)
    ax.text(.06, .045, f"mean transverse error  {fit.initial_cost:.4f} → {fit.final_cost:.4f}",
            color=base.MUTED, fontsize=7.4)

    ax = fig.add_subplot(gs[1, 1]); base._style(ax)
    for key, label, lw in (("distorted", "aberrated", 1.15),
                           ("physical_update", "after parameter update", 1.25),
                           ("miao_only", "Miao", 1.35),
                           ("hybrid", "digital twin + Miao", 1.55)):
        ax.plot(metrics.z_mm, metrics[f"{key}_pearson_r"], lw=lw, label=label)
    ax.set(title="Agreement with the nominal beam", xlabel="z from axicon (mm)", ylabel="Pearson r", ylim=(-.05, 1.05))
    ax.grid(alpha=.18, color="#66727d"); ax.legend(fontsize=6.8, frameon=False, labelcolor=base.FG)

    ax = fig.add_subplot(gs[1, 2]); base._style(ax)
    for key, label, lw in (("distorted", "aberrated", 1.15),
                           ("physical_update", "after parameter update", 1.25),
                           ("miao_only", "Miao", 1.35),
                           ("hybrid", "digital twin + Miao", 1.55)):
        ax.plot(metrics.z_mm, metrics[f"{key}_nrmse"], lw=lw, label=label)
    ax.set(title="Transverse intensity error", xlabel="z from axicon (mm)", ylabel="normalized RMSE")
    ax.grid(alpha=.18, color="#66727d")

    ax = fig.add_subplot(gs[1, 3]); base._style(ax)
    names = ["Miao", "Twin + Miao"]
    vals = [summary["miao_only"]["mean_pearson_r"], summary["hybrid"]["mean_pearson_r"]]
    bars = ax.bar(np.arange(2), vals, width=.55)
    ax.set_xticks(np.arange(2), names); ax.set_ylim(0, 1.05); ax.set_ylabel("mean Pearson r")
    ax.set_title("Mean agreement across the Bessel region", fontsize=9.7, weight="bold")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+.025, f"{val:.3f}", color=base.FG, fontsize=7.3, ha="center")
    ax.text(.5, .06,
            f"mean NRMSE  {summary['miao_only']['mean_nrmse']:.3f}  →  {summary['hybrid']['mean_nrmse']:.3f}",
            transform=ax.transAxes, color=base.MUTED, fontsize=7.1, ha="center")

    iz = len(Z_DISPLAY_M)//2
    ax = fig.add_subplot(gs[2, :3]); base._style(ax)
    for stack, label, lw in ((distorted, "aberrated", 1.0), (aligned, "parameter update only", 1.15),
                             (miao, "Miao", 1.25), (hybrid, "digital twin + Miao", 1.5),
                             (ideal, "nominal", 1.3)):
        x_mm, cut = base._xcut(stack, grid, iz)
        keep = np.abs(x_mm) <= 1.15
        ax.plot(x_mm[keep], cut[keep], lw=lw, label=label)
    ax.set(title=f"Transverse profile at z = {Z_DISPLAY_M[iz]*1e3:.0f} mm",
           xlabel="x at y = 0 (mm)", ylabel="normalized intensity", ylim=(-.03, 1.08))
    ax.grid(alpha=.18, color="#66727d"); ax.legend(fontsize=7.0, ncol=5, frameon=False, labelcolor=base.FG)

    ax = fig.add_subplot(gs[2, 3]); base._style(ax)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Two correction strategies", fontsize=10.2, weight="bold", pad=7)
    ax.text(.07, .80, "Miao et al.", color=base.GOLD, fontsize=9.0, weight="bold")
    ax.text(.07, .68, "intensity z-stack → residual phase → phase correction", color=base.MUTED, fontsize=7.2)
    ax.text(.07, .45, "Digital twin + Miao", color=base.CYAN, fontsize=9.0, weight="bold")
    ax.text(.07, .33, "intensity z-stack → physical parameters → system update", color=base.MUTED, fontsize=7.2)
    ax.text(.07, .23, "→ residual phase → phase correction", color=base.MUTED, fontsize=7.2)
    ax.text(.07, .06, "same q, z-range, noise and evaluation in both branches", color=base.FG, fontsize=7.0)

    fig.text(.045, .952, "Digital-twin-assisted correction of a q = 20 Bessel beam",
             color=base.FG, fontsize=22.0, weight="bold", ha="left")
    fig.text(.045, .904,
             "The Miao phase-retrieval method is tested alone and after estimating physically meaningful system parameters from the same intensity z-stack.",
             color=base.MUTED, fontsize=9.4, ha="left")
    fig.text(.045, .858,
             "Controlled test: axicon offset = +250 µm, Fourier-iris radius = 0.85 × nominal, plus a non-axisymmetric phase aberration.",
             color=base.CYAN, fontsize=8.5, ha="left")

    png = out/"q20_miao_vs_digital_twin_v2.png"
    pdf = out/"q20_miao_vs_digital_twin_v2.pdf"
    fig.savefig(png, dpi=500, bbox_inches="tight", facecolor=base.BG)
    fig.savefig(pdf, bbox_inches="tight", facecolor=base.BG)
    plt.close(fig)
    with Image.open(png) as im:
        preview = im.convert("RGB"); preview.thumbnail((2600, 1600), Image.Resampling.LANCZOS)
        prev = out/"q20_miao_vs_digital_twin_v2.preview.jpg"; preview.save(prev, quality=92, subsampling=0)
    return png, pdf, prev


def build(out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    registry = system_sweep_registry()
    nominal = SystemErrorConfig()
    nominal_route = base._route(nominal, None)
    phase_truth = residual_phase(nominal_route["grid"])

    mapping_name, mapping_diag = calibrate_mapping(nominal_route, phase_truth)

    truth = apply_registry_family(nominal, "fourf_iris_radius_scale", 0.85, registry=registry)
    truth = apply_registry_family(truth, "axicon_lateral_decentre_x", 250e-6, registry=registry)
    distorted_route = base._route(truth, phase_truth)
    clean_fit = base._propagate(distorted_route, Z_FIT_M)
    rng = np.random.default_rng(SEED)
    noise = rng.normal(size=clean_fit.shape)
    measured_fit = base._add_noise(clean_fit, noise)

    x = np.asarray(nominal_route["grid"]["x"], float)
    ids = np.flatnonzero(np.abs(x) <= base.FIT_CROP_HALF_M)
    target = measured_fit[:, ids[:, None], ids]
    simulator = base.PhysicalSimulator(ids)
    fit = hierarchical_physical_fit(
        target_stack=target,
        simulate_config=simulator,
        families=FIT_PARAMETERS,
        registry=registry,
        max_stages=2,
        min_improvement_fraction=0.003,
    )

    updated_config = base._compensate_physical_parameters(truth, fit.final_config)
    updated_route = base._route(updated_config, phase_truth)
    updated_fit_clean = base._propagate(updated_route, Z_FIT_M)
    updated_fit = base._add_noise(updated_fit_clean, noise)

    raw_miao, _, miao_diag = retrieve_raw_correction(distorted_route, measured_fit)
    raw_hybrid, _, hybrid_diag = retrieve_raw_correction(updated_route, updated_fit)
    miao_phase = transform_phase(raw_miao, mapping_name)
    hybrid_phase = transform_phase(raw_hybrid, mapping_name)

    ideal = base._propagate(nominal_route, Z_DISPLAY_M)
    distorted = base._propagate(distorted_route, Z_DISPLAY_M)
    aligned = base._propagate(updated_route, Z_DISPLAY_M)
    miao = base._propagate(distorted_route, Z_DISPLAY_M, miao_phase)
    hybrid = base._propagate(updated_route, Z_DISPLAY_M, hybrid_phase)

    metrics = metric_table(Z_DISPLAY_M, ideal, {
        "distorted": distorted,
        "physical_update": aligned,
        "miao_only": miao,
        "hybrid": hybrid,
    }, nominal_route["grid"])
    metrics.to_csv(out/"comparison_metrics_vs_z.csv", index=False)

    selected = {}
    for step in fit.steps:
        if step.accepted and step.selected_family is not None:
            selected[step.selected_family] = float(step.selected_value)

    summary = {
        "study": "q20 Miao correction alone versus digital-twin physical parameter estimation followed by the same Miao correction",
        "q": Q,
        "truth": {
            "axicon_lateral_decentre_x_m": 250e-6,
            "fourf_iris_radius_scale": 0.85,
            "residual_phase": "0.30 cos(2theta) + 0.18 sin(3theta) + 0.09 cos(5theta)",
        },
        "noise_sigma_fraction_of_plane_peak": NOISE_SIGMA,
        "fit_z_planes": len(Z_FIT_M),
        "evaluation_z_planes": len(Z_DISPLAY_M),
        "synthetic_coordinate_calibration": mapping_diag,
        "physical_parameter_estimation": {
            "parameters_tested": list(FIT_PARAMETERS),
            "selected": selected,
            "selected_display": fit_display(fit),
            "E_I_definition": "mean over z of pixel RMSE between independently peak-normalized model and data planes on fixed laboratory coordinates",
            "E_I_before": float(fit.initial_cost),
            "E_I_after": float(fit.final_cost),
            "trace": fit.as_dict(),
        },
        "aberrated": summary_method(metrics, "distorted"),
        "physical_update_only": summary_method(metrics, "physical_update"),
        "miao_only": {**miao_diag, **summary_method(metrics, "miao_only")},
        "hybrid": {**hybrid_diag, **summary_method(metrics, "hybrid")},
    }
    summary["hybrid_minus_miao"] = {
        "mean_pearson_r": summary["hybrid"]["mean_pearson_r"]-summary["miao_only"]["mean_pearson_r"],
        "mean_nrmse": summary["hybrid"]["mean_nrmse"]-summary["miao_only"]["mean_nrmse"],
    }

    np.save(out/"miao_only_correction_phase_axicon_input_rad.npy", miao_phase.astype(np.float32))
    np.save(out/"digital_twin_plus_miao_correction_phase_axicon_input_rad.npy", hybrid_phase.astype(np.float32))
    np.save(out/"synthetic_residual_phase_truth_slm1_rad.npy", phase_truth.astype(np.float32))

    png, pdf, preview = build_figure(out, nominal_route["grid"], fit, metrics,
                                     ideal, distorted, aligned, miao, hybrid, summary)
    with Image.open(png) as im:
        summary["assets"] = {
            "png_500dpi": str(png), "pdf": str(pdf), "preview": str(preview),
            "png_pixel_size": list(im.size), "png_dpi": list(im.info.get("dpi", (0, 0))),
            "metrics_csv": str(out/"comparison_metrics_vs_z.csv"),
        }
    (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    build()
