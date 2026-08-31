"""Build the final poster-facing scientific figure set.

This renderer reuses the current dual-SLM/4F/axicon forward model and the final
q=20 synthetic correction benchmark. It deliberately keeps figures image-led and
uses plain scientific labels suitable for an A0 PhD poster.

Outputs are collected in outputs/poster/final_evidence_v9.
"""
from __future__ import annotations

from pathlib import Path
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

import build_phase2i_presentation_figures as phase2i
import build_presentation_extended_evidence as extended
import build_poster_core_evidence_v8 as core

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "poster" / "final_evidence_v9"
Q20 = ROOT / "outputs" / "validation" / "q20_integrated_correction_v6"

BG = "#070a0d"
AXBG = "#0b1015"
FG = "#f5f6f7"
MUTED = "#aab4be"
CYAN = "#4dd9d5"
GOLD = "#ffbd4a"
GREEN = "#45e0a8"
RED = "#ff5a52"
BORDER = "#44515d"
EPS = np.finfo(float).tiny
THERMAL = LinearSegmentedColormap.from_list(
    "poster_final_thermal",
    [(0.00, "#000000"), (0.10, "#090000"), (0.28, "#460000"),
     (0.50, "#a91a00"), (0.72, "#ef5b00"), (0.88, "#ffb300"),
     (1.00, "#fff2a8")], N=256,
)

# Re-render the presentation science in the preferred poster intensity language.
phase2i.CMAP = THERMAL
extended.CMAP = THERMAL


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(AXBG)
    for s in ax.spines.values():
        s.set_color(BORDER)
        s.set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(FG)
    ax.grid(False)


def _save(fig: plt.Figure, path: Path, dpi: int = 330) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _norm(a: np.ndarray, peak: float | None = None) -> np.ndarray:
    arr = np.maximum(np.asarray(a, float), 0.0)
    scale = float(np.max(arr)) if peak is None else float(peak)
    return arr / max(scale, EPS)


def _corr_rmse(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    av = np.asarray(a, float).ravel(); bv = np.asarray(b, float).ravel()
    av = av / max(float(np.max(av)), EPS); bv = bv / max(float(np.max(bv)), EPS)
    return float(np.corrcoef(av, bv)[0, 1]), float(np.sqrt(np.mean((av-bv)**2)))


def build_q20_comparison(out: Path) -> tuple[Path, dict]:
    stack_path = Q20 / "comparison_eval_stacks.npz"
    metrics_path = Q20 / "metrics_vs_z.csv"
    summary_path = Q20 / "summary.json"
    if not stack_path.exists() or not metrics_path.exists() or not summary_path.exists():
        raise FileNotFoundError("run benchmark_q20_integrated_correction_v6.py before poster rendering")

    dat = np.load(stack_path)
    metrics = pd.read_csv(metrics_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    x = np.asarray(dat["x_m"], float)
    z = np.asarray(dat["z_m"], float)
    iz = int(np.argmin(np.abs(z - 110e-3)))
    ids = np.flatnonzero(np.abs(x) <= 0.62e-3)
    extent = [x[ids[0]]*1e3, x[ids[-1]]*1e3, x[ids[0]]*1e3, x[ids[-1]]*1e3]

    ideal = np.asarray(dat["ideal"][iz], float)[np.ix_(ids, ids)]
    panels = [
        ("ideal", "Target", ideal),
        ("distorted", "Distorted system", np.asarray(dat["distorted"][iz], float)[np.ix_(ids, ids)]),
        ("physical_adjustment", "Physical adjustment", np.asarray(dat["physical_adjustment"][iz], float)[np.ix_(ids, ids)]),
        ("physical_plus_miao", "Miao residual correction", np.asarray(dat["physical_plus_miao"][iz], float)[np.ix_(ids, ids)]),
        ("full_model_slm2", "Full model + SLM2", np.asarray(dat["full_model_slm2"][iz], float)[np.ix_(ids, ids)]),
    ]
    common_peak = float(np.max(ideal))

    fig = plt.figure(figsize=(16.0, 8.7), facecolor=BG)
    gs = fig.add_gridspec(2, 5, height_ratios=[1.04, 0.82], hspace=0.34, wspace=0.10,
                          left=0.055, right=0.985, top=0.885, bottom=0.09)
    panel_metrics = {}
    for col, (key, title, plane) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col]); _style(ax)
        ax.imshow(_norm(plane, common_peak)**0.55, origin="lower", extent=extent,
                  cmap=THERMAL, vmin=0, vmax=1, interpolation="nearest", aspect="equal")
        ax.set_title(title, fontsize=11.3, weight="bold", pad=6)
        ax.set_xlabel("x (mm)", fontsize=8)
        if col == 0: ax.set_ylabel("y (mm)", fontsize=8)
        else: ax.tick_params(labelleft=False)
        r, e = _corr_rmse(plane, ideal)
        panel_metrics[key] = {"pearson_r_at_selected_plane": r, "nrmse_at_selected_plane": e}
        if key != "ideal":
            ax.text(0.5, -0.19, f"r = {r:.3f}   NRMSE = {e:.3f}", transform=ax.transAxes,
                    ha="center", va="top", color=MUTED, fontsize=8.3)

    axr = fig.add_subplot(gs[1, :3]); _style(axr)
    curves = [
        ("distorted_pearson_r", "distorted", RED),
        ("physical_adjustment_only_pearson_r", "physical adjustment", GOLD),
        ("physical_fit_plus_miao_native_input_pearson_r", "physical fit + Miao", CYAN),
        ("physical_fit_plus_full_model_slm2_pearson_r", "full model + SLM2", GREEN),
    ]
    for col, label, colour in curves:
        axr.plot(metrics["z_mm"], metrics[col], lw=2.15, label=label, color=colour)
    axr.set_ylim(0.0, 1.02)
    axr.set_xlabel("z from axicon (mm)")
    axr.set_ylabel("Pearson correlation to nominal q=20 field")
    axr.set_title("Agreement across the complete evaluation scan", fontsize=12.2, weight="bold")
    axr.legend(frameon=False, labelcolor=FG, fontsize=8.8, ncol=2, loc="lower right")

    axe = fig.add_subplot(gs[1, 3:]); _style(axe)
    names = ["distorted", "physical_adjustment_only", "physical_fit_plus_miao_native_input", "physical_fit_plus_full_model_slm2"]
    labels = ["distorted", "physical\nadjustment", "physical +\nMiao", "full model\n+ SLM2"]
    vals = [float(summary[n]["mean_pearson_r"]) for n in names]
    bars = axe.bar(np.arange(len(vals)), vals, width=0.66)
    for b, val in zip(bars, vals):
        axe.text(b.get_x()+b.get_width()/2, val+0.018, f"{val:.3f}", ha="center", color=FG, fontsize=9.2)
    axe.set_xticks(np.arange(len(vals)), labels)
    axe.set_ylim(0, 1.08)
    axe.set_ylabel("mean Pearson r")
    axe.set_title("Mean performance", fontsize=12.2, weight="bold")

    fig.suptitle("Closing the loop: physical system fit and residual-wavefront correction",
                 color=FG, fontsize=18.4, weight="bold", y=0.973)
    fig.text(0.5, 0.915,
             "Synthetic q=20 truth study. Full-model residual fitted on alternating illuminated planes; correction applied as an additive SLM2 phase layer.",
             ha="center", color=MUTED, fontsize=9.6)
    path = _save(fig, out / "07_q20_closed_loop_correction.png")
    meta = {
        "selected_z_mm": float(z[iz]*1e3),
        "panels": panel_metrics,
        "summary": {n: summary[n] for n in names},
        "heldout": summary["full_model_residual_fit"]["heldout"],
        "phase_rms_to_truth_rad": summary["full_model_residual_fit"]["phase_rms_to_truth_rad"],
        "claim_boundary": summary["correction_planes"],
    }
    return path, meta


def build_q20_phase_recovery(out: Path) -> tuple[Path, dict]:
    summary = json.loads((Q20 / "summary.json").read_text(encoding="utf-8"))
    truth = np.load(Q20 / "truth_residual_phase_slm1_rad.npy")
    est = np.load(Q20 / "estimated_residual_phase_slm1_rad.npy")
    stacks = np.load(Q20 / "comparison_eval_stacks.npz")
    x = np.asarray(stacks["x_m"], float)
    ids = np.flatnonzero(np.abs(x) <= 2.15e-3)
    extent = [x[ids[0]]*1e3, x[ids[-1]]*1e3, x[ids[0]]*1e3, x[ids[-1]]*1e3]
    t = np.asarray(truth, float)[np.ix_(ids, ids)]
    e = np.asarray(est, float)[np.ix_(ids, ids)]
    d = np.angle(np.exp(1j*(e-t)))
    rms = float(summary["full_model_residual_fit"]["phase_rms_to_truth_rad"])
    held = summary["full_model_residual_fit"]["heldout"]
    coeff = summary["full_model_residual_fit"]["coefficient_table"]

    fig = plt.figure(figsize=(15.8, 7.0), facecolor=BG)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.72], hspace=0.30, wspace=0.18,
                          left=0.055, right=0.985, top=0.86, bottom=0.105)
    maps = [(t, "Injected residual phase"), (e, "Recovered residual phase")]
    for col, (arr, title) in enumerate(maps):
        ax = fig.add_subplot(gs[0, col]); _style(ax)
        im = ax.imshow(arr, origin="lower", extent=extent, cmap="twilight", vmin=-np.pi, vmax=np.pi,
                       interpolation="nearest", aspect="equal")
        ax.set_title(title, fontsize=11.6, weight="bold")
        ax.set_xlabel("x (mm)")
        if col == 0: ax.set_ylabel("y (mm)")
        else: ax.tick_params(labelleft=False)

    axd = fig.add_subplot(gs[0, 2]); _style(axd)
    lim = max(0.005, float(np.nanpercentile(np.abs(d), 99.5)))
    imd = axd.imshow(d, origin="lower", extent=extent, cmap="coolwarm", vmin=-lim, vmax=lim,
                     interpolation="nearest", aspect="equal")
    axd.set_title("Recovered - truth", fontsize=11.6, weight="bold")
    axd.set_xlabel("x (mm)"); axd.tick_params(labelleft=False)
    cb = fig.colorbar(imd, ax=axd, fraction=0.047, pad=0.03)
    cb.set_label("phase error (rad)", color=MUTED, fontsize=8)
    cb.ax.tick_params(colors=MUTED, labelsize=7.5)

    axtext = fig.add_subplot(gs[0, 3]); axtext.set_facecolor(BG); axtext.axis("off")
    axtext.text(0.02, 0.92, "Held-out validation", color=FG, fontsize=13.0, weight="bold", va="top")
    axtext.text(0.02, 0.70, f"Phase RMS error\n{rms:.4f} rad", color=GREEN, fontsize=15.0, weight="bold", va="top")
    axtext.text(0.02, 0.43, f"Unused z planes\nmean r = {float(held['mean_pearson_r']):.5f}\nmean RMSE = {float(held['mean_rmse']):.4f}",
                color=FG, fontsize=11.0, va="top", linespacing=1.45)
    axtext.text(0.02, 0.10, "The programmed q = 20 vortex term is excluded from the fitted residual.",
                color=MUTED, fontsize=9.0, va="bottom", wrap=True)

    axc = fig.add_subplot(gs[1, :]); _style(axc)
    modes = [int(r["m"]) for r in coeff]
    truth_c = []; est_c = []; labels = []
    for r in coeff:
        truth_c.extend([float(r["cos_truth_rad"]), float(r["sin_truth_rad"])])
        est_c.extend([float(r["cos_est_rad"]), float(r["sin_est_rad"])])
        labels.extend([f"c{int(r['m'])}", f"s{int(r['m'])}"])
    xx = np.arange(len(labels)); w = 0.38
    axc.bar(xx-w/2, truth_c, width=w, label="truth")
    axc.bar(xx+w/2, est_c, width=w, label="recovered")
    axc.axhline(0, color=MUTED, lw=0.8)
    axc.set_xticks(xx, labels)
    axc.set_ylabel("angular Fourier coefficient (rad)")
    axc.set_title("Residual phase basis:  ψ(θ) = Σ [cₘ cos(mθ) + sₘ sin(mθ)]", fontsize=12.0, weight="bold")
    axc.legend(frameon=False, labelcolor=FG, fontsize=9)

    fig.suptitle("Full-model residual phase retrieval", color=FG, fontsize=18.0, weight="bold", y=0.965)
    fig.text(0.5, 0.902,
             "Candidate phases are propagated through the complete numerical optical route; validation planes are not used by the optimizer.",
             ha="center", color=MUTED, fontsize=9.5)
    path = _save(fig, out / "08_q20_phase_recovery_and_validation.png")
    return path, {"phase_rms_rad": rms, "heldout": held, "coefficient_table": coeff}


def build_contact_sheet(out: Path, files: list[Path]) -> Path:
    thumbs = []
    for p in files:
        im = Image.open(p).convert("RGB")
        im.thumbnail((1100, 620), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (1140, 700), (7, 10, 13))
        canvas.paste(im, ((1140-im.width)//2, 18))
        d = ImageDraw.Draw(canvas)
        d.text((24, 650), p.name, fill=(235, 238, 242))
        thumbs.append(canvas)
    cols = 2; rows = int(math.ceil(len(thumbs)/cols))
    sheet = Image.new("RGB", (cols*1140, rows*700), (7, 10, 13))
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i%cols)*1140, (i//cols)*700))
    path = out / "00_final_evidence_contact_sheet.png"
    sheet.save(path, quality=95)
    return path


def build(out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    route = phase2i.build_computational_route(out, 768); files.append(route)
    hero, hero_meta = extended.build_beam_profile_hero(out, 768); files.append(hero)
    errors, error_meta = extended.build_real_error_fingerprints(out, 768); files.append(errors)
    dec, dec_meta = phase2i.build_v1_decentre(out, 768); files.append(dec)
    tip, tip_meta = phase2i.build_v1_tip(out, 1024); files.append(tip)
    heal, heal_meta = core.build_self_healing_quality(out); files.append(heal)
    qcomp, qcomp_meta = build_q20_comparison(out); files.append(qcomp)
    qphase, qphase_meta = build_q20_phase_recovery(out); files.append(qphase)
    contact = build_contact_sheet(out, files)

    manifest = {
        "study": "final poster evidence v9",
        "files": [p.name for p in files],
        "contact_sheet": contact.name,
        "beam_profile": hero_meta,
        "system_errors": error_meta,
        "axicon_decentre": dec_meta,
        "axicon_tip": tip_meta,
        "self_healing": heal_meta,
        "q20_correction": qcomp_meta,
        "q20_phase_recovery": qphase_meta,
        "poster_claim_boundary": [
            "physical perturbation magnitudes are illustrative sensitivity cases unless explicitly measured",
            "q20 closed-loop figures are synthetic truth-controlled validation, not a completed experimental SLM correction",
            "real SLM2 application remains conditional on measured coordinate mapping and 1030-nm LUT calibration",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    build()
