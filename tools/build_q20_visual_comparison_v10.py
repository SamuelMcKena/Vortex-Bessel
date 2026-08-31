"""Poster-resolution visual comparison for the final q=20 correction study.

Produces one deliberately simple four-column figure:

    target | original distorted | Miao corrected | complete model corrected

Rows show (1) transverse intensity at one held-out plane, (2) fixed-laboratory
x-z propagation, and (3) a linear intensity profile through the same transverse
plane.  All intensity images share the target scale; profiles are linear and
normalised by the target peak.  The displayed heatmaps use a disclosed power-law
gamma only to make low-intensity Bessel structure visible on an A0 poster.

The selected transverse plane is chosen only from the full-model held-out set,
using the largest target-plane peak.  Correction performance is not used to pick
that plane.
"""
from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
Q20 = ROOT / "outputs" / "validation" / "q20_integrated_correction_v6"
OUT = ROOT / "outputs" / "poster" / "final_evidence_v9"

BG = "#06080b"
AXBG = "#090d12"
FG = "#f4f6f8"
MUTED = "#a8b2bd"
BORDER = "#45515d"
TARGET = "#f4f6f8"
ORIGINAL = "#ff6b5e"
MIAO = "#ffbe4d"
COMPLETE = "#57e0b0"
EPS = np.finfo(float).tiny
DISPLAY_GAMMA = 0.55
XY_HALF_M = 0.70e-3
XZ_HALF_M = 0.70e-3
Z_MIN_M = 60e-3
Z_MAX_M = 138e-3

THERMAL = LinearSegmentedColormap.from_list(
    "poster_q20_thermal",
    [(0.00, "#000000"), (0.12, "#100000"), (0.30, "#520000"),
     (0.52, "#b32100"), (0.72, "#ef6200"), (0.88, "#ffb800"),
     (1.00, "#fff3ad")],
    N=256,
)


def style(ax: plt.Axes) -> None:
    ax.set_facecolor(AXBG)
    for s in ax.spines.values():
        s.set_color(BORDER)
        s.set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(FG)
    ax.grid(False)


def display_intensity(a: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.maximum(np.asarray(a, float), 0.0) / max(float(scale), EPS), 0.0, 1.0) ** DISPLAY_GAMMA


def pair_metrics(candidate: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    a = np.asarray(candidate, float).ravel()
    b = np.asarray(target, float).ravel()
    scale = max(float(np.max(b)), EPS)
    an = a / scale
    bn = b / scale
    return float(np.corrcoef(an, bn)[0, 1]), float(np.sqrt(np.mean((an - bn) ** 2)))


def choose_heldout_plane(z: np.ndarray, ideal: np.ndarray, summary: dict) -> int:
    held = [float(p["z_mm"]) * 1e-3 for p in summary["full_model_residual_fit"]["heldout"]["planes"]]
    ids = [int(np.argmin(np.abs(z - zz))) for zz in held]
    # Target only: select the visually strongest held-out target plane.
    peaks = [float(np.max(ideal[i])) for i in ids]
    return int(ids[int(np.argmax(peaks))])


def build(out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    stack_path = Q20 / "comparison_eval_stacks.npz"
    summary_path = Q20 / "summary.json"
    if not stack_path.exists() or not summary_path.exists():
        raise FileNotFoundError("run benchmark_q20_integrated_correction_v6.py first")

    dat = np.load(stack_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    x = np.asarray(dat["x_m"], float)
    z = np.asarray(dat["z_m"], float)
    ideal = np.asarray(dat["ideal"], float)

    cases = [
        ("Target", "ideal", TARGET),
        ("Original", "distorted", ORIGINAL),
        ("Miao corrected", "miao_only", MIAO),
        ("Complete model", "full_model_slm2", COMPLETE),
    ]
    iz = choose_heldout_plane(z, ideal, summary)
    iy0 = int(np.argmin(np.abs(x)))
    xy_ids = np.flatnonzero(np.abs(x) <= XY_HALF_M)
    xz_ids = np.flatnonzero(np.abs(x) <= XZ_HALF_M)
    z_ids = np.flatnonzero((z >= Z_MIN_M) & (z <= Z_MAX_M))
    if len(xy_ids) < 32 or len(xz_ids) < 32 or len(z_ids) < 10:
        raise RuntimeError("comparison crop is under-sampled")

    xy_extent = [x[xy_ids[0]]*1e3, x[xy_ids[-1]]*1e3,
                 x[xy_ids[0]]*1e3, x[xy_ids[-1]]*1e3]
    xz_extent = [z[z_ids[0]]*1e3, z[z_ids[-1]]*1e3,
                 x[xz_ids[0]]*1e3, x[xz_ids[-1]]*1e3]

    target_xy = ideal[iz][np.ix_(xy_ids, xy_ids)]
    target_xy_peak = float(np.max(target_xy))
    target_xz = ideal[np.ix_(z_ids, [iy0], xz_ids)][:, 0, :]
    target_xz_peak = float(np.max(target_xz))
    target_profile = ideal[iz, iy0, xy_ids]
    profile_scale = max(float(np.max(target_profile)), EPS)

    fig = plt.figure(figsize=(19.2, 13.2), facecolor=BG)
    gs = fig.add_gridspec(
        3, 4,
        height_ratios=[1.00, 1.02, 0.72],
        hspace=0.30, wspace=0.11,
        left=0.055, right=0.985, bottom=0.075, top=0.885,
    )

    metrics = {}
    for col, (label, key, colour) in enumerate(cases):
        stack = np.asarray(dat[key], float)
        plane = stack[iz][np.ix_(xy_ids, xy_ids)]
        xz = stack[np.ix_(z_ids, [iy0], xz_ids)][:, 0, :]
        profile = stack[iz, iy0, xy_ids] / profile_scale

        ax = fig.add_subplot(gs[0, col]); style(ax)
        ax.imshow(
            display_intensity(plane, target_xy_peak),
            origin="lower", extent=xy_extent, cmap=THERMAL,
            vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal",
        )
        ax.set_title(label, fontsize=15.2, weight="bold", color=colour, pad=8)
        ax.set_xlabel("x (mm)", fontsize=10)
        if col == 0:
            ax.set_ylabel("y (mm)", fontsize=10)
        else:
            ax.tick_params(labelleft=False)
        ax.axhline(0, color="white", alpha=0.14, lw=0.6)
        ax.axvline(0, color="white", alpha=0.14, lw=0.6)

        axz = fig.add_subplot(gs[1, col]); style(axz)
        axz.imshow(
            display_intensity(xz.T, target_xz_peak),
            origin="lower", extent=xz_extent, cmap=THERMAL,
            vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto",
        )
        axz.axvline(float(z[iz])*1e3, color="white", alpha=0.70, ls="--", lw=1.0)
        axz.axhline(0, color="white", alpha=0.14, lw=0.6)
        axz.set_xlabel("z from axicon (mm)", fontsize=10)
        if col == 0:
            axz.set_ylabel("x at fixed y = 0 (mm)", fontsize=10)
        else:
            axz.tick_params(labelleft=False)

        ap = fig.add_subplot(gs[2, col]); style(ap)
        target_line = target_profile / profile_scale
        if key == "ideal":
            ap.plot(x[xy_ids]*1e3, target_line, lw=2.2, color=TARGET)
        else:
            ap.plot(x[xy_ids]*1e3, target_line, lw=1.5, ls="--", color=TARGET, alpha=0.74, label="target")
            ap.plot(x[xy_ids]*1e3, profile, lw=2.2, color=colour, label=label.lower())
            ap.legend(frameon=False, labelcolor=FG, fontsize=8.5, loc="upper right")
        ap.set_xlim(x[xy_ids[0]]*1e3, x[xy_ids[-1]]*1e3)
        ap.set_ylim(-0.03, max(1.05, 1.05*float(np.max(profile))))
        ap.set_xlabel("x (mm)", fontsize=10)
        if col == 0:
            ap.set_ylabel("intensity / target peak", fontsize=10)
        else:
            ap.tick_params(labelleft=False)

        r, rmse = pair_metrics(plane, target_xy)
        metrics[key] = {"pearson_r_at_selected_plane": r, "nrmse_to_target_peak": rmse}
        if key != "ideal":
            ap.text(0.03, 0.91, f"r = {r:.4f}\nNRMSE = {rmse:.4f}", transform=ap.transAxes,
                    ha="left", va="top", color=MUTED, fontsize=8.7)

    fig.suptitle(
        "q = 20 correction: target, measured distortion and two reconstruction routes",
        color=FG, fontsize=21.5, weight="bold", y=0.972,
    )
    fig.text(
        0.5, 0.925,
        f"Transverse plane: z = {float(z[iz])*1e3:.1f} mm (held out from the full-model residual fit).  "
        "All heatmaps use the same target intensity scale.",
        ha="center", color=MUTED, fontsize=11.2,
    )
    fig.text(
        0.5, 0.032,
        f"Heatmap display only: (I / I_target)^{{{DISPLAY_GAMMA:.2f}}}.  Profiles and reported metrics use linear intensity.  "
        "Miao is the direct analytical axicon-input baseline; the complete-model correction is applied as an additive SLM2 phase layer before the 4F relay.",
        ha="center", color=MUTED, fontsize=9.4,
    )

    png = out / "09_q20_target_original_miao_complete.png"
    pdf = out / "09_q20_target_original_miao_complete.pdf"
    fig.savefig(png, dpi=520, facecolor=BG, bbox_inches="tight")
    fig.savefig(pdf, dpi=520, facecolor=BG, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "study": "q20 target-original-Miao-complete visual comparison",
        "selected_z_mm": float(z[iz]*1e3),
        "selected_plane_rule": "largest target peak among full-model held-out z planes; correction performance not used",
        "heatmap_common_normalisation": "target intensity peak",
        "heatmap_display_gamma": DISPLAY_GAMMA,
        "profile_normalisation": "linear intensity / target profile peak",
        "xz_plane": "fixed laboratory y=0; no z-dependent recentering",
        "metrics": metrics,
        "source_summary": {
            "distorted": summary["distorted"],
            "miao_only": summary["miao_only_native_input"],
            "complete_model": summary["physical_fit_plus_full_model_slm2"],
            "heldout_fit": summary["full_model_residual_fit"]["heldout"],
            "phase_rms_to_truth_rad": summary["full_model_residual_fit"]["phase_rms_to_truth_rad"],
        },
        "files": [png.name, pdf.name],
    }
    (out / "09_q20_target_original_miao_complete.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    build()
