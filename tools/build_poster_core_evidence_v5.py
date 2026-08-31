"""Poster core evidence v5.

The v4 simultaneous-error benchmark exposed an important limitation: fitting only
one fixed-y longitudinal line does not cleanly separate axicon decentre from beam
pointing.  This pass keeps that result as a useful diagnostic lesson and upgrades
the benchmark to fit the *full cropped XY intensity at every z plane*.

The synthetic truth remains deliberately off-grid and noisy.  The output reports
only discrete-grid estimates and cost separation; it does not invent statistical
confidence intervals or claim unique experimental diagnosis.
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
from vbb_study.digital_twin.physical_error_inference import (
    grid_search_two_parameters,
    plane_normalise_stack,
)
from vbb_study.digital_twin.vortex_beam_slm_errors import GaussianBeamError
from vbb_study.digital_twin.vortex_continuous_propagation import (
    build_fixed_support_spectrum,
    native_field_at_z,
)
from vbb_study.digital_twin.vortex_system_route import (
    AxiconError,
    SystemErrorConfig,
    build_system_route,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "poster" / "core_evidence_v5"


def _xy_stack(
    config: SystemErrorConfig,
    *,
    grid_n: int,
    z_m: np.ndarray,
    halfwidth_m: float,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return full cropped XY intensity for every requested fixed-lab z plane."""

    route = build_system_route("V1", grid_n=int(grid_n), config=config)
    prop = build_fixed_support_spectrum(
        np.asarray(route["post_axicon"], dtype=np.complex128),
        dict(route["grid"]),
        wavelength_m=float(route["metadata"]["wavelength_m"]),
        z_max_m=float(np.max(z_m)),
        minimum_retained_spectral_power=0.995,
    )
    x = np.asarray(route["grid"]["x"], dtype=float)
    ids = np.flatnonzero(np.abs(x) <= float(halfwidth_m))
    if ids.size < 21:
        raise RuntimeError("full-XY inference crop is under-sampled")
    planes = []
    for z in np.asarray(z_m, float):
        field = native_field_at_z(prop, float(z))
        intensity = np.abs(np.asarray(field, np.complex128)) ** 2
        planes.append(intensity[np.ix_(ids, ids)])
    return np.stack(planes, axis=0), x[ids]


def build_joint_parameter_recovery_xy(out: Path, *, grid_n: int = 256) -> tuple[Path, dict, Path]:
    z = np.linspace(20e-3, 100e-3, 17)
    halfwidth_m = 0.95e-3
    truth_dec_um = 275.0
    truth_pointing_mrad = 0.47
    noise_fraction = 0.010

    def config(dec_um: float, pointing_mrad: float) -> SystemErrorConfig:
        return SystemErrorConfig(
            beam=GaussianBeamError(pointing_rad=(float(pointing_mrad) * 1e-3, 0.0)),
            axicon=AxiconError(decentre_m=(float(dec_um) * 1e-6, 0.0)),
        )

    truth_clean, coords = _xy_stack(
        config(truth_dec_um, truth_pointing_mrad),
        grid_n=grid_n,
        z_m=z,
        halfwidth_m=halfwidth_m,
        label="poster-v5-fullxy-truth",
    )
    target = v4._add_plane_relative_noise(truth_clean, noise_fraction, seed=20260831)

    def simulate(dec_um: float, pointing_mrad: float) -> np.ndarray:
        return _xy_stack(
            config(dec_um, pointing_mrad),
            grid_n=grid_n,
            z_m=z,
            halfwidth_m=halfwidth_m,
            label=f"poster-v5-fullxy-{dec_um:g}-{pointing_mrad:g}",
        )[0]

    coarse_dec = np.asarray([-400., -200., 0., 200., 400.])
    coarse_point = np.asarray([-0.8, -0.4, 0.0, 0.4, 0.8])
    coarse = grid_search_two_parameters(
        parameter_x="axicon decentre x", units_x="µm", values_x=coarse_dec,
        parameter_y="input pointing x", units_y="mrad", values_y=coarse_point,
        target_stack=target, simulate=simulate,
    )

    fine_dec = np.arange(coarse.best_x - 150.0, coarse.best_x + 150.1, 50.0)
    fine_point = np.arange(coarse.best_y - 0.30, coarse.best_y + 0.3001, 0.10)
    fine = grid_search_two_parameters(
        parameter_x="axicon decentre x", units_x="µm", values_x=fine_dec,
        parameter_y="input pointing x", units_y="mrad", values_y=fine_point,
        target_stack=target, simulate=simulate,
    )

    best, _ = _xy_stack(
        config(fine.best_x, fine.best_y),
        grid_n=grid_n,
        z_m=z,
        halfwidth_m=halfwidth_m,
        label="poster-v5-fullxy-best",
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

    zero_point_row = int(np.argmin(np.abs(coarse.values_y)))
    naive_ix = int(np.argmin(coarse.costs[zero_point_row]))
    naive_dec_um = float(coarse.values_x[naive_ix])

    fig = plt.figure(figsize=(15.4, 8.5), facecolor=v4.DARK)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.05, 1.0],
                          height_ratios=[1.0, 0.92], hspace=0.30, wspace=0.27)

    for col, (arr, title) in enumerate([
        (target_xz, "synthetic measurement slice\nfull XY stack used in fit + 1% noise"),
        (best_xz, "best forward-model fit\nshown at the same y = 0 slice"),
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
    vmax = max(float(np.quantile(residual_xz, 0.995)), v4.EPS)
    im_res = ax_res.imshow(residual_xz, origin="lower", aspect="auto", extent=extent,
                           cmap=v4.DIFF, vmin=0, vmax=vmax)
    ax_res.set_title("absolute residual\n(display slice only)", fontsize=11.5, weight="bold")
    ax_res.set_xlabel("z from axicon (mm)")
    ax_res.tick_params(labelleft=False)
    c1 = fig.colorbar(im_res, ax=ax_res, fraction=0.048, pad=0.025)
    c1.ax.tick_params(labelsize=7.5, colors=v4.MUTED)
    c1.outline.set_edgecolor("#51606d")

    ax_cost = fig.add_subplot(gs[1, :2])
    v4._style(ax_cost)
    im = ax_cost.imshow(
        fine.costs,
        origin="lower",
        aspect="auto",
        extent=[fine.values_x[0], fine.values_x[-1], fine.values_y[0], fine.values_y[-1]],
        cmap=v4.DIFF,
    )
    ax_cost.scatter([truth_dec_um], [truth_pointing_mrad], marker="x", s=95,
                    color=v4.RED, linewidths=2.0, label="injected truth")
    ax_cost.scatter([fine.best_x], [fine.best_y], marker="o", s=70, facecolors="none",
                    edgecolors=v4.FG, linewidths=1.8, label="recovered grid minimum")
    ax_cost.set_xlabel("axicon decentre x (µm)")
    ax_cost.set_ylabel("input pointing x (mrad)")
    ax_cost.set_title("joint inverse cost — full XY morphology across 17 planes",
                      fontsize=11.8, weight="bold")
    ax_cost.legend(frameon=False, labelcolor=v4.FG, fontsize=8.5, loc="upper left")
    c2 = fig.colorbar(im, ax=ax_cost, fraction=0.025, pad=0.018)
    c2.set_label("plane-normalised morphology RMSE", color=v4.MUTED, fontsize=8.2)
    c2.ax.tick_params(labelsize=7.5, colors=v4.MUTED)
    c2.outline.set_edgecolor("#51606d")

    ax_txt = fig.add_subplot(gs[1, 2])
    ax_txt.set_facecolor(v4.DARK)
    ax_txt.axis("off")
    ax_txt.text(0.02, 0.94, "interpretable physical metrics", color=v4.CYAN,
                fontsize=12.5, weight="bold", va="top")
    lines = [
        ("Injected axicon decentre", f"{truth_dec_um:+.0f} µm"),
        ("Recovered axicon decentre", f"{fine.best_x:+.0f} µm"),
        ("absolute error", f"{abs(fine.best_x-truth_dec_um):.0f} µm"),
        ("Injected beam pointing", f"{truth_pointing_mrad:+.2f} mrad"),
        ("Recovered beam pointing", f"{fine.best_y:+.2f} mrad"),
        ("absolute error", f"{abs(fine.best_y-truth_pointing_mrad):.2f} mrad"),
        ("best-fit RMSE", f"{fine.best_cost:.4f}"),
        ("best/2nd-best margin", f"{100*fine.relative_cost_margin:.1f}%"),
    ]
    y = 0.80
    for label, value in lines:
        ax_txt.text(0.03, y, label, color=v4.MUTED, fontsize=9.2, va="center")
        ax_txt.text(0.97, y, value, color=v4.FG, fontsize=10.0, weight="bold",
                    ha="right", va="center")
        y -= 0.085
    ax_txt.text(0.03, 0.105,
                f"If pointing were incorrectly fixed to zero,\nthe coarse fit would prefer ~{naive_dec_um:+.0f} µm decentre.",
                color=v4.GOLD, fontsize=9.1, linespacing=1.35)
    ax_txt.text(0.03, 0.018,
                "Synthetic validation only — discrete grid estimates,\nnot statistical confidence intervals.",
                color=v4.RED, fontsize=8.3, linespacing=1.25)

    fig.suptitle("Physical-error recovery using the full simulated camera stack",
                 color=v4.FG, fontsize=18.2, weight="bold", y=0.985)
    fig.text(0.5, 0.944,
             "Off-grid simultaneous errors + deterministic noise; the optimizer sees full cropped XY intensity at every z plane, not only one longitudinal line.",
             ha="center", color=v4.MUTED, fontsize=9.6)

    png = v4._save(fig, out / "02_joint_physical_error_recovery_fullxy.png")

    summary = {
        "benchmark_scope": "synthetic model-to-model validation using full cropped XY intensity at 17 z planes",
        "grid_n": int(grid_n),
        "z_planes": int(len(z)),
        "xy_crop_halfwidth_um": float(halfwidth_m * 1e6),
        "xy_crop_samples_per_axis": int(len(coords)),
        "noise_sigma_fraction_of_each_plane_peak": float(noise_fraction),
        "truth": {
            "axicon_decentre_x_um": truth_dec_um,
            "input_pointing_x_mrad": truth_pointing_mrad,
        },
        "coarse_fit": coarse.as_dict(),
        "fine_fit": fine.as_dict(),
        "recovered": {
            "axicon_decentre_x_um": float(fine.best_x),
            "input_pointing_x_mrad": float(fine.best_y),
            "axicon_abs_error_um": float(abs(fine.best_x - truth_dec_um)),
            "pointing_abs_error_mrad": float(abs(fine.best_y - truth_pointing_mrad)),
            "best_cost": float(fine.best_cost),
            "relative_cost_margin": float(fine.relative_cost_margin),
        },
        "naive_coarse_fit_with_pointing_fixed_zero": {
            "preferred_axicon_decentre_x_um": naive_dec_um,
        },
        "claim_boundary": "synthetic full-XY validation; discrete-grid cost separation is not a statistical confidence interval and does not establish unique experimental causality",
    }

    csv_path = out / "02_joint_physical_error_recovery_fullxy_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value", "units"])
        writer.writerow(["truth_axicon_decentre_x", truth_dec_um, "um"])
        writer.writerow(["recovered_axicon_decentre_x", fine.best_x, "um"])
        writer.writerow(["axicon_absolute_error", abs(fine.best_x-truth_dec_um), "um"])
        writer.writerow(["truth_input_pointing_x", truth_pointing_mrad, "mrad"])
        writer.writerow(["recovered_input_pointing_x", fine.best_y, "mrad"])
        writer.writerow(["pointing_absolute_error", abs(fine.best_y-truth_pointing_mrad), "mrad"])
        writer.writerow(["best_fit_rmse", fine.best_cost, "a.u."])
        writer.writerow(["relative_best_second_margin", fine.relative_cost_margin, "fraction"])
        writer.writerow(["noise_sigma", noise_fraction, "fraction_of_plane_peak"])
        writer.writerow(["z_planes", len(z), "count"])
        writer.writerow(["xy_crop_samples_per_axis", len(coords), "count"])

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
    fig.suptitle("Poster core evidence v5 — simulation → physical metrics → experimental phase retrieval",
                 color="white", fontsize=20, weight="bold", y=0.995)
    return v4._save(fig, out / "00_core_evidence_v5_contact_sheet.png", dpi=180)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p1, m1 = v4.build_self_healing_path(OUT)
    p2, m2, csv2 = build_joint_parameter_recovery_xy(OUT)
    exp = v4.copy_experimental(OUT)
    sheet = contact_sheet(OUT, [p1, p2] + exp)
    manifest = {
        "outcome": "POSTER-CORE-EVIDENCE-V5",
        "story": "physical simulation -> self-healing -> full-XY physical-error inference -> experimental residual-phase retrieval",
        "self_healing": m1,
        "joint_physical_error_recovery_fullxy": m2,
        "experimental_figures": [str(p) for p in exp],
        "files": [str(sheet), str(p1), str(p2), str(csv2)] + [str(p) for p in exp],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
