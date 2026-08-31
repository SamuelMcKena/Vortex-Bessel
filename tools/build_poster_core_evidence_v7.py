"""Poster core evidence v7.

This pass turns the simulated physical-error recovery into a genuinely
metric-aware diagnostic rather than fitting peak-normalised morphology alone.
The bounded search uses two dimensionless terms with fixed equal weights:

* full-XY plane-normalised morphology RMSE across 17 z planes;
* relative total-power / throughput RMSE across the same planes.

The throughput term is useful for 4F iris errors, while morphology retains the
spatial signature needed for axicon decentre.  On real camera data the throughput
term is only valid when exposure/gain and acquisition scaling are stable or
calibrated; the code and figure say this explicitly.
"""

from __future__ import annotations

from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import build_poster_core_evidence_v4 as v4
import build_poster_core_evidence_v5 as v5
from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.physical_error_inference import (
    grid_search_two_parameters,
    morphology_rmse,
    plane_normalise_stack,
)
from vbb_study.digital_twin.vortex_explicit_4f import FourFError
from vbb_study.digital_twin.vortex_system_route import AxiconError, SystemErrorConfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "poster" / "core_evidence_v7"
EPS = np.finfo(float).tiny


def throughput_rmse(candidate: np.ndarray, target: np.ndarray) -> float:
    """Relative total-power trace error, preserving the common source scale."""
    c = np.maximum(np.asarray(candidate, float), 0.0)
    t = np.maximum(np.asarray(target, float), 0.0)
    if c.shape != t.shape:
        raise ValueError("candidate and target stacks must share one camera grid")
    pc = np.sum(c, axis=(1, 2))
    pt = np.sum(t, axis=(1, 2))
    scale = max(float(np.mean(pt)), EPS)
    return float(np.sqrt(np.mean((pc - pt) ** 2)) / scale)


def composite_loss(candidate: np.ndarray, target: np.ndarray) -> float:
    return 0.5 * morphology_rmse(candidate, target) + 0.5 * throughput_rmse(candidate, target)


def build_metric_aware_recovery(out: Path, *, grid_n: int = 256) -> tuple[Path, dict, Path]:
    z = np.linspace(20e-3, 100e-3, 17)
    halfwidth_m = 0.95e-3
    truth_dec_um = 275.0
    truth_iris_r = 0.27
    noise_fraction = 0.006

    manifest = canonical_hardware_manifest()
    iris_radius_m = float(hardware_value(manifest, "fourier_iris_radius_m"))

    def config(dec_um: float, iris_r: float) -> SystemErrorConfig:
        return SystemErrorConfig(
            fourf=FourFError(iris_offset_m=(float(iris_r) * iris_radius_m, 0.0)),
            axicon=AxiconError(decentre_m=(float(dec_um) * 1e-6, 0.0)),
        )

    truth_clean, coords = v5._xy_stack(
        config(truth_dec_um, truth_iris_r), grid_n=grid_n, z_m=z,
        halfwidth_m=halfwidth_m, label="poster-v7-truth",
    )
    target = v4._add_plane_relative_noise(truth_clean, noise_fraction, seed=20260831)

    def simulate(dec_um: float, iris_r: float) -> np.ndarray:
        return v5._xy_stack(
            config(dec_um, iris_r), grid_n=grid_n, z_m=z,
            halfwidth_m=halfwidth_m, label=f"poster-v7-{dec_um:g}-{iris_r:g}",
        )[0]

    coarse_dec = np.asarray([-400., -200., 0., 200., 400.])
    coarse_iris = np.asarray([-0.50, -0.25, 0.0, 0.25, 0.50])
    coarse = grid_search_two_parameters(
        parameter_x="axicon decentre x", units_x="µm", values_x=coarse_dec,
        parameter_y="4F iris offset x", units_y="iris radii", values_y=coarse_iris,
        target_stack=target, simulate=simulate, loss_fn=composite_loss,
    )

    fine_dec = np.arange(coarse.best_x - 150.0, coarse.best_x + 150.1, 50.0)
    fine_iris = np.arange(coarse.best_y - 0.18, coarse.best_y + 0.1801, 0.06)
    fine = grid_search_two_parameters(
        parameter_x="axicon decentre x", units_x="µm", values_x=fine_dec,
        parameter_y="4F iris offset x", units_y="iris radii", values_y=fine_iris,
        target_stack=target, simulate=simulate, loss_fn=composite_loss,
    )

    best, _ = v5._xy_stack(
        config(fine.best_x, fine.best_y), grid_n=grid_n, z_m=z,
        halfwidth_m=halfwidth_m, label="poster-v7-best",
    )
    target_n = plane_normalise_stack(target)
    best_n = plane_normalise_stack(best)
    residual = np.abs(best_n - target_n)
    iy0 = int(np.argmin(np.abs(coords)))
    target_xz = target_n[:, iy0, :].T
    best_xz = best_n[:, iy0, :].T
    residual_xz = residual[:, iy0, :].T
    extent = [float(z[0] * 1e3), float(z[-1] * 1e3),
              float(coords[0] * 1e3), float(coords[-1] * 1e3)]

    best_morph = morphology_rmse(best, target)
    best_power = throughput_rmse(best, target)
    truth_noise_morph = morphology_rmse(truth_clean, target)
    truth_noise_power = throughput_rmse(truth_clean, target)

    zero_iris_row = int(np.argmin(np.abs(coarse.values_y)))
    naive_ix = int(np.argmin(coarse.costs[zero_iris_row]))
    naive_dec_um = float(coarse.values_x[naive_ix])

    fig = plt.figure(figsize=(15.4, 8.6), facecolor=v4.DARK)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.05, 1.0],
                          height_ratios=[1.0, 0.94], hspace=0.30, wspace=0.27)

    for col, (arr, title) in enumerate([
        (target_xz, "synthetic measurement slice\nfull XY stack + 0.6% noise"),
        (best_xz, "best forward-model fit\nmetric-aware objective"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        v4._style(ax)
        ax.imshow(arr ** 0.50, origin="lower", aspect="auto", extent=extent,
                  cmap=v4.THERMAL, vmin=0, vmax=1)
        ax.set_title(title, fontsize=11.5, weight="bold")
        ax.set_xlabel("z from axicon (mm)")
        if col == 0:
            ax.set_ylabel("x at fixed y = 0 (mm)")
        else:
            ax.tick_params(labelleft=False)

    ax_res = fig.add_subplot(gs[0, 2])
    v4._style(ax_res)
    vmax = max(float(np.quantile(residual_xz, 0.995)), EPS)
    im_res = ax_res.imshow(residual_xz, origin="lower", aspect="auto", extent=extent,
                           cmap=v4.DIFF, vmin=0, vmax=vmax)
    ax_res.set_title("absolute morphology residual\n(display slice only)", fontsize=11.5, weight="bold")
    ax_res.set_xlabel("z from axicon (mm)")
    ax_res.tick_params(labelleft=False)
    c1 = fig.colorbar(im_res, ax=ax_res, fraction=0.048, pad=0.025)
    c1.ax.tick_params(labelsize=7.5, colors=v4.MUTED)
    c1.outline.set_edgecolor("#51606d")

    ax_cost = fig.add_subplot(gs[1, :2])
    v4._style(ax_cost)
    im = ax_cost.imshow(
        fine.costs, origin="lower", aspect="auto",
        extent=[fine.values_x[0], fine.values_x[-1], fine.values_y[0], fine.values_y[-1]],
        cmap=v4.DIFF,
    )
    ax_cost.scatter([truth_dec_um], [truth_iris_r], marker="x", s=95,
                    color=v4.RED, linewidths=2.0, label="injected truth")
    ax_cost.scatter([fine.best_x], [fine.best_y], marker="o", s=70, facecolors="none",
                    edgecolors=v4.FG, linewidths=1.8, label="recovered grid minimum")
    ax_cost.set_xlabel("axicon decentre x (µm)")
    ax_cost.set_ylabel("4F iris offset x (iris radii)")
    ax_cost.set_title("joint inverse cost = 0.5 morphology + 0.5 throughput",
                      fontsize=11.8, weight="bold")
    ax_cost.legend(frameon=False, labelcolor=v4.FG, fontsize=8.5, loc="upper left")
    c2 = fig.colorbar(im, ax=ax_cost, fraction=0.025, pad=0.018)
    c2.set_label("dimensionless composite loss", color=v4.MUTED, fontsize=8.2)
    c2.ax.tick_params(labelsize=7.5, colors=v4.MUTED)
    c2.outline.set_edgecolor("#51606d")

    ax_txt = fig.add_subplot(gs[1, 2])
    ax_txt.set_facecolor(v4.DARK)
    ax_txt.axis("off")
    ax_txt.text(0.02, 0.95, "recovered physical metrics", color=v4.CYAN,
                fontsize=12.5, weight="bold", va="top")
    lines = [
        ("Axicon: injected", f"{truth_dec_um:+.0f} µm"),
        ("Axicon: recovered", f"{fine.best_x:+.0f} µm"),
        ("absolute error", f"{abs(fine.best_x-truth_dec_um):.0f} µm"),
        ("4F iris: injected", f"{truth_iris_r:+.2f} R"),
        ("4F iris: recovered", f"{fine.best_y:+.2f} R"),
        ("absolute error", f"{abs(fine.best_y-truth_iris_r):.2f} R"),
        ("morphology RMSE", f"{best_morph:.4f}"),
        ("throughput RMSE", f"{best_power:.4f}"),
        ("best/2nd-best margin", f"{100*fine.relative_cost_margin:.1f}%"),
    ]
    y = 0.82
    for label, value in lines:
        ax_txt.text(0.03, y, label, color=v4.MUTED, fontsize=9.0, va="center")
        ax_txt.text(0.97, y, value, color=v4.FG, fontsize=9.8, weight="bold",
                    ha="right", va="center")
        y -= 0.074
    ax_txt.text(0.03, 0.105,
                f"If the iris were assumed perfect,\nthe coarse fit would prefer ~{naive_dec_um:+.0f} µm decentre.",
                color=v4.GOLD, fontsize=9.0, linespacing=1.35)
    ax_txt.text(0.03, 0.018,
                "Throughput requires stable/calibrated camera scaling.\nSynthetic validation — not experimental certainty.",
                color=v4.RED, fontsize=8.1, linespacing=1.25)

    fig.suptitle("Metric-aware recovery of simulated bench errors",
                 color=v4.FG, fontsize=18.3, weight="bold", y=0.985)
    fig.text(0.5, 0.944,
             "The digital twin now returns interpretable physical parameters using both field morphology and a calibrated throughput signature, before any residual phase correction is attempted.",
             ha="center", color=v4.MUTED, fontsize=9.6)

    png = v4._save(fig, out / "02_metric_aware_axicon_iris_recovery.png")
    summary = {
        "benchmark_scope": "synthetic metric-aware model-to-model validation of simultaneous axicon decentre and 4F iris offset",
        "objective": {
            "morphology_weight": 0.5,
            "throughput_weight": 0.5,
            "throughput_experimental_requirement": "stable or calibrated camera exposure/gain and common acquisition scaling",
        },
        "grid_n": int(grid_n),
        "z_planes": int(len(z)),
        "xy_crop_halfwidth_um": float(halfwidth_m * 1e6),
        "xy_crop_samples_per_axis": int(len(coords)),
        "noise_sigma_fraction_of_each_plane_peak": float(noise_fraction),
        "truth": {
            "axicon_decentre_x_um": truth_dec_um,
            "fourf_iris_offset_x_radii": truth_iris_r,
        },
        "coarse_fit": coarse.as_dict(),
        "fine_fit": fine.as_dict(),
        "recovered": {
            "axicon_decentre_x_um": float(fine.best_x),
            "fourf_iris_offset_x_radii": float(fine.best_y),
            "axicon_abs_error_um": float(abs(fine.best_x - truth_dec_um)),
            "iris_abs_error_radii": float(abs(fine.best_y - truth_iris_r)),
            "best_composite_loss": float(fine.best_cost),
            "morphology_rmse": float(best_morph),
            "throughput_rmse": float(best_power),
            "relative_cost_margin": float(fine.relative_cost_margin),
        },
        "noise_floor_at_true_clean_model": {
            "morphology_rmse": float(truth_noise_morph),
            "throughput_rmse": float(truth_noise_power),
        },
        "naive_coarse_fit_with_iris_fixed_zero": {
            "preferred_axicon_decentre_x_um": naive_dec_um,
        },
        "claim_boundary": "synthetic validation; parameter reporting in experiment requires identifiability screening and calibration provenance",
    }

    csv_path = out / "02_metric_aware_axicon_iris_recovery_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value", "units"])
        writer.writerow(["truth_axicon_decentre_x", truth_dec_um, "um"])
        writer.writerow(["recovered_axicon_decentre_x", fine.best_x, "um"])
        writer.writerow(["axicon_absolute_error", abs(fine.best_x-truth_dec_um), "um"])
        writer.writerow(["truth_4f_iris_offset_x", truth_iris_r, "iris_radius"])
        writer.writerow(["recovered_4f_iris_offset_x", fine.best_y, "iris_radius"])
        writer.writerow(["iris_absolute_error", abs(fine.best_y-truth_iris_r), "iris_radius"])
        writer.writerow(["best_composite_loss", fine.best_cost, "dimensionless"])
        writer.writerow(["best_morphology_rmse", best_morph, "dimensionless"])
        writer.writerow(["best_throughput_rmse", best_power, "fraction"])
        writer.writerow(["relative_best_second_margin", fine.relative_cost_margin, "fraction"])
        writer.writerow(["noise_sigma", noise_fraction, "fraction_of_plane_peak"])
        writer.writerow(["z_planes", len(z), "count"])

    return png, summary, csv_path


def contact_sheet(out: Path, paths: list[Path]) -> Path:
    fig = plt.figure(figsize=(18, 14.5), facecolor="#171b1f")
    gs = fig.add_gridspec(3, 2, hspace=0.18, wspace=0.09)
    for i, p in enumerate(paths[:6]):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.set_facecolor("#171b1f")
        ax.imshow(plt.imread(p))
        ax.set_title(p.stem.replace("_", " "), color="white", fontsize=12, weight="bold")
        ax.axis("off")
    fig.suptitle("Poster core evidence v7 — self-healing + metric-aware diagnosis + experimental retrieval",
                 color="white", fontsize=20, weight="bold", y=0.995)
    return v4._save(fig, out / "00_core_evidence_v7_contact_sheet.png", dpi=180)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p1, m1 = v4.build_self_healing_path(OUT)
    p2, m2, csv2 = build_metric_aware_recovery(OUT)
    exp = v4.copy_experimental(OUT)
    sheet = contact_sheet(OUT, [p1, p2] + exp)
    manifest = {
        "outcome": "POSTER-CORE-EVIDENCE-V7",
        "story": "physical simulation -> self-healing -> metric-aware physical diagnosis -> experimental residual-phase retrieval",
        "self_healing": m1,
        "metric_aware_physical_error_recovery": m2,
        "experimental_figures": [str(p) for p in exp],
        "files": [str(sheet), str(p1), str(p2), str(csv2)] + [str(p) for p in exp],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
