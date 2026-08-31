"""Build focused poster evidence v4.

This pass implements two specific improvements requested after visual/scientific
review:

1. self-healing is shown as a longitudinal path around a deliberately visible
   central obstruction, using the same inferno/thermal intensity language as the
   main beam figures;
2. physical-error inference is stress-tested with *simultaneous*, off-grid
   axicon-decentre and beam-pointing errors plus deterministic camera-like noise.

The inverse benchmark remains synthetic model-to-model validation.  It is useful
for proving that the forward-model fitting machinery can return interpretable
physical metrics, but it is not evidence that an experimental z-stack uniquely
identifies those two physical causes.

Canonical q=20 experimental phase-retrieval figures are copied unchanged so
measured-data provenance is preserved.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import csv
import json
import math
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import bessel_twin_core as bt
import publication_diagnostics as pdiag
from vbb_study import vbb_studies
from vbb_study.digital_twin.physical_error_inference import (
    grid_search_two_parameters,
    plane_normalise_stack,
)
from vbb_study.digital_twin.vortex_beam_slm_errors import GaussianBeamError
from vbb_study.digital_twin.vortex_continuous_propagation import (
    build_fixed_plane_longitudinal_map,
    build_fixed_support_spectrum,
)
from vbb_study.digital_twin.vortex_system_route import (
    AxiconError,
    SystemErrorConfig,
    build_system_route,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "poster" / "core_evidence_v4"
DARK = "#070a0d"
AX = "#0b1015"
FG = "#f3f5f6"
MUTED = "#aab5bf"
CYAN = "#59e0d5"
GOLD = "#ffd166"
GREEN = "#65df9b"
RED = "#ff6b6b"
THERMAL = "inferno"
DIFF = "cividis"
EPS = np.finfo(float).tiny


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(AX)
    for spine in ax.spines.values():
        spine.set_color("#51606d")
        spine.set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(FG)


def _save(fig: plt.Figure, path: Path, dpi: int = 330) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor(), pad_inches=0.05)
    plt.close(fig)
    return path


def _norm(a: np.ndarray, scale: float | None = None) -> np.ndarray:
    arr = np.maximum(np.asarray(a, dtype=float), 0.0)
    denom = float(np.max(arr)) if scale is None else float(scale)
    return arr / max(denom, EPS)


def _crop_plane(plane: np.ndarray, full_grid: dict, crop_grid: dict) -> np.ndarray:
    x_full = np.asarray(full_grid["x"], float)
    x_crop = np.asarray(crop_grid["x"], float)
    ids = np.flatnonzero((x_full >= x_crop[0]) & (x_full <= x_crop[-1]))
    if ids.size < 4:
        return np.asarray(plane, float)
    return np.asarray(plane, float)[np.ix_(ids, ids)]


def build_self_healing_path(out: Path) -> tuple[Path, dict]:
    """Make the obstruction and the reconstruction path visually explicit."""

    cfg = vbb_studies.beam_air_config(bt.default_config("balanced"))
    cfg = replace(cfg, target=replace(cfg.target, ell=0))
    design = bt.compute_design_from_config(cfg)

    # The previous poster figure used 0.9 x the first-zero radius, which was
    # physically valid but visually tiny.  Here we deliberately obscure several
    # central rings while retaining the outer conical reservoir.
    first_zero = float(design.equivalent_l0_first_zero_radius_m)
    obstacle_radius = 3.5 * first_zero
    bundle = pdiag.build_self_healing_bundle(
        config=cfg,
        preset="balanced",
        path="ideal",
        case_id="poster_B0_self_healing_visible_obstacle",
        obstacle_kind="disk",
        obstacle_radius_m=obstacle_radius,
        axial_points=151,
    )

    vol = bundle["obstructed_volume"]
    cgrid = vol["crop_grid"]
    x_um = np.asarray(cgrid["x"], float) / bt.um
    z_um = np.asarray(bundle["z_relative"], float) / bt.um
    xz = np.asarray(vol["xz"], float)
    ref = _crop_plane(bundle["reference_plane"], bundle["grid"], cgrid)
    blocked = _crop_plane(bundle["obstructed_plane"], bundle["grid"], cgrid)
    stack = np.asarray(vol["intensity_stack"], float)
    recovery = np.asarray(bundle["onaxis_recovery"], float)

    # Choose the first substantial recovery; if the larger obstruction never
    # reaches 80%, use the strongest recovered plane and report its actual ratio.
    ids = np.flatnonzero((recovery >= 0.80) & np.isfinite(recovery))
    if ids.size:
        recovered_idx = int(ids[0])
    else:
        search0 = max(1, int(0.08 * len(recovery)))
        recovered_idx = int(search0 + np.nanargmax(recovery[search0:]))
    recovered = stack[recovered_idx]

    shared = float(np.max(ref))
    extent_xy = [x_um[0], x_um[-1], x_um[0], x_um[-1]]
    extent_xz = [z_um[0], z_um[-1], x_um[0], x_um[-1]]

    # Geometric guide derived from the conical transverse wavevector.  This is
    # annotation only; the field underneath is full wave propagation.
    k0 = 2.0 * np.pi / float(cfg.laser.wavelength_m)
    theta = math.asin(min(0.999999, abs(float(design.kr_sample_m_inv)) / k0))
    z_geom_um = float(obstacle_radius / max(math.tan(theta), EPS) / bt.um)

    fig = plt.figure(figsize=(16.2, 7.5), facecolor=DARK)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.72, 1.0, 1.0, 1.0], height_ratios=[1.0, 0.48], wspace=0.18, hspace=0.31)

    ax_xz = fig.add_subplot(gs[:, 0])
    _style(ax_xz)
    ax_xz.imshow(_norm(xz) ** 0.42, origin="lower", aspect="auto", extent=extent_xz,
                 cmap=THERMAL, vmin=0, vmax=1)
    ax_xz.set_title("wave propagation after the obstruction", fontsize=13, weight="bold", pad=8)
    ax_xz.set_xlabel("distance after obstacle, z (µm)")
    ax_xz.set_ylabel("x at fixed y = 0 (µm)")

    z_span = max(float(z_um[-1] - z_um[0]), 1.0)
    obs_width = 0.024 * z_span
    r_um = obstacle_radius / bt.um
    ax_xz.add_patch(Rectangle((float(z_um[0]), -r_um), obs_width, 2 * r_um,
                              facecolor="black", edgecolor="white", linewidth=1.35, zorder=6))
    ax_xz.text(float(z_um[0]) + 0.034 * z_span, 1.17 * r_um,
               f"opaque disk\nØ {2*r_um:.1f} µm", color=FG, fontsize=9.2,
               va="center", ha="left")

    z_guide = min(float(z_um[0] + z_geom_um), float(z_um[-1]))
    ax_xz.plot([z_um[0], z_guide], [r_um, 0], color=CYAN, ls="--", lw=1.6, alpha=0.9)
    ax_xz.plot([z_um[0], z_guide], [-r_um, 0], color=CYAN, ls="--", lw=1.6, alpha=0.9)
    ax_xz.scatter([z_guide], [0], s=22, color=CYAN, zorder=7)
    ax_xz.text(0.61, 0.965, "conical-wave guide", transform=ax_xz.transAxes,
               color=CYAN, fontsize=8.8, ha="center", va="top")

    transverse = [
        (ref, "unobstructed reference"),
        (blocked, "immediately after disk"),
        (recovered, f"reconstructed field\nz = {z_um[recovered_idx]:.0f} µm"),
    ]
    for col, (plane, title) in enumerate(transverse, start=1):
        ax = fig.add_subplot(gs[0, col])
        _style(ax)
        ax.imshow(_norm(plane, shared) ** 0.42, origin="lower", extent=extent_xy,
                  cmap=THERMAL, vmin=0, vmax=1)
        ax.set_title(title, fontsize=11.2, weight="bold", pad=7)
        ax.set_xlabel("x (µm)")
        if col == 1:
            ax.set_ylabel("y (µm)")
        else:
            ax.tick_params(labelleft=False)
        ax.grid(False)

    ax_r = fig.add_subplot(gs[1, 1:])
    _style(ax_r)
    ax_r.plot(z_um, bundle["peak_recovery"], color=CYAN, lw=2.5, label="peak recovery")
    ax_r.plot(z_um, bundle["onaxis_recovery"], color=GOLD, lw=2.2, ls="--", label="on-axis recovery")
    ax_r.axhline(1.0, color=MUTED, lw=1.0, ls=":")
    ax_r.axvline(z_um[recovered_idx], color=GREEN, lw=1.1, ls="--")
    ax_r.set_xlabel("distance after obstruction (µm)")
    ax_r.set_ylabel("obstructed / reference")
    ymax = max(1.15, 1.04 * float(np.nanmax(bundle["peak_recovery"])))
    ax_r.set_ylim(0, min(1.55, ymax))
    ax_r.legend(frameon=False, ncol=2, labelcolor=FG, loc="lower right")
    ax_r.grid(alpha=0.12)

    fig.suptitle("Self-healing of a Bessel field around a finite obstruction", color=FG,
                 fontsize=18.5, weight="bold", y=0.989)
    fig.text(0.5, 0.942,
             "Thermal heatmap = simulated wave field.  Black disk and dashed cone lines are explicit geometric annotations.",
             ha="center", color=MUTED, fontsize=9.8)

    meta = {
        "first_zero_radius_um": float(first_zero / bt.um),
        "obstacle_radius_um": float(r_um),
        "obstacle_diameter_um": float(2 * r_um),
        "obstacle_radius_in_first_zero_radii": 3.5,
        "geometric_cone_half_angle_deg": float(np.degrees(theta)),
        "geometric_reconstruction_distance_um": z_geom_um,
        "selected_recovered_plane_um": float(z_um[recovered_idx]),
        "selected_onaxis_recovery": float(recovery[recovered_idx]),
        "claim_boundary": "heatmap is simulated wave propagation; obstacle and dashed conical guide are annotations",
    }
    return _save(fig, out / "01_self_healing_visible_obstacle.png"), meta


def _xz_stack(config: SystemErrorConfig, *, grid_n: int, z_m: np.ndarray,
              coord_m: np.ndarray, label: str) -> np.ndarray:
    route = build_system_route("V1", grid_n=int(grid_n), config=config)
    prop = build_fixed_support_spectrum(
        np.asarray(route["post_axicon"], dtype=np.complex128),
        dict(route["grid"]),
        wavelength_m=float(route["metadata"]["wavelength_m"]),
        z_max_m=float(np.max(z_m)),
        minimum_retained_spectral_power=0.995,
    )
    mapped = build_fixed_plane_longitudinal_map(
        prop,
        z_values_m=np.asarray(z_m, float),
        x_coordinates_m=np.asarray(coord_m, float),
        y_coordinates_m=np.asarray(coord_m, float),
        fixed_x_m=0.0,
        fixed_y_m=0.0,
        source_label=label,
    )
    xz = np.asarray(mapped.xz_intensity, float)
    if xz.shape[0] == len(z_m):
        return xz[:, :, None]
    if xz.shape[1] == len(z_m):
        return xz.T[:, :, None]
    raise RuntimeError(f"unexpected XZ shape {xz.shape} for {len(z_m)} z planes")


def _add_plane_relative_noise(stack: np.ndarray, sigma_fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    arr = np.maximum(np.asarray(stack, float), 0.0).copy()
    flat = arr.reshape(arr.shape[0], -1)
    peaks = np.maximum(np.max(flat, axis=1), EPS)
    noise = rng.normal(size=arr.shape) * peaks[:, None, None] * float(sigma_fraction)
    return np.maximum(arr + noise, 0.0)


def build_joint_parameter_recovery(out: Path, *, grid_n: int = 256) -> tuple[Path, dict, Path]:
    """Off-grid, noisy, simultaneous-error model-to-model benchmark."""

    z = np.linspace(20e-3, 100e-3, 17)
    coord = np.linspace(-0.82e-3, 0.82e-3, 181)

    truth_dec_um = 275.0
    truth_pointing_mrad = 0.47
    noise_fraction = 0.015

    def config(dec_um: float, pointing_mrad: float) -> SystemErrorConfig:
        return SystemErrorConfig(
            beam=GaussianBeamError(pointing_rad=(float(pointing_mrad) * 1e-3, 0.0)),
            axicon=AxiconError(decentre_m=(float(dec_um) * 1e-6, 0.0)),
        )

    truth_clean = _xz_stack(config(truth_dec_um, truth_pointing_mrad), grid_n=grid_n,
                            z_m=z, coord_m=coord, label="poster-v4-joint-truth")
    target = _add_plane_relative_noise(truth_clean, noise_fraction, seed=20260831)

    coarse_dec = np.asarray([-400., -200., 0., 200., 400.])
    coarse_point = np.asarray([-0.8, -0.4, 0.0, 0.4, 0.8])
    coarse = grid_search_two_parameters(
        parameter_x="axicon decentre x", units_x="µm", values_x=coarse_dec,
        parameter_y="input pointing x", units_y="mrad", values_y=coarse_point,
        target_stack=target,
        simulate=lambda d, p: _xz_stack(config(d, p), grid_n=grid_n, z_m=z,
                                         coord_m=coord, label=f"poster-v4-coarse-{d:g}-{p:g}"),
    )

    fine_dec = np.arange(coarse.best_x - 150.0, coarse.best_x + 150.1, 50.0)
    fine_point = np.arange(coarse.best_y - 0.30, coarse.best_y + 0.3001, 0.10)
    fine = grid_search_two_parameters(
        parameter_x="axicon decentre x", units_x="µm", values_x=fine_dec,
        parameter_y="input pointing x", units_y="mrad", values_y=fine_point,
        target_stack=target,
        simulate=lambda d, p: _xz_stack(config(d, p), grid_n=grid_n, z_m=z,
                                         coord_m=coord, label=f"poster-v4-fine-{d:g}-{p:g}"),
    )

    best = _xz_stack(config(fine.best_x, fine.best_y), grid_n=grid_n, z_m=z,
                     coord_m=coord, label="poster-v4-best-fit")
    target_n = plane_normalise_stack(target)[:, :, 0]
    best_n = plane_normalise_stack(best)[:, :, 0]
    residual = np.abs(best_n - target_n)

    # Naive decentre-only diagnostic from the coarse row where pointing is fixed
    # to zero.  This is useful for showing why simultaneous fitting matters.
    zero_point_row = int(np.argmin(np.abs(coarse.values_y - 0.0)))
    naive_ix = int(np.argmin(coarse.costs[zero_point_row]))
    naive_dec_um = float(coarse.values_x[naive_ix])
    nominal_iy = zero_point_row
    nominal_ix = int(np.argmin(np.abs(coarse.values_x - 0.0)))
    nominal_cost = float(coarse.costs[nominal_iy, nominal_ix])

    extent = [float(z[0] * 1e3), float(z[-1] * 1e3),
              float(coord[0] * 1e3), float(coord[-1] * 1e3)]

    fig = plt.figure(figsize=(15.4, 8.5), facecolor=DARK)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.05, 1.0], height_ratios=[1.0, 0.92],
                          hspace=0.30, wspace=0.27)

    for col, (arr, title) in enumerate([
        (target_n, "synthetic z-stack\nunknown simultaneous errors + 1.5% noise"),
        (best_n, "best forward-model fit"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        _style(ax)
        ax.imshow(arr.T ** 0.50, origin="lower", aspect="auto", extent=extent,
                  cmap=THERMAL, vmin=0, vmax=1)
        ax.set_title(title, fontsize=11.6, weight="bold")
        ax.set_xlabel("z from axicon (mm)")
        if col == 0:
            ax.set_ylabel("x at fixed y = 0 (mm)")
        else:
            ax.tick_params(labelleft=False)

    ax_res = fig.add_subplot(gs[0, 2])
    _style(ax_res)
    im_res = ax_res.imshow(residual.T, origin="lower", aspect="auto", extent=extent,
                           cmap=DIFF, vmin=0, vmax=max(float(np.quantile(residual, 0.995)), EPS))
    ax_res.set_title("absolute morphology residual", fontsize=11.6, weight="bold")
    ax_res.set_xlabel("z from axicon (mm)")
    ax_res.tick_params(labelleft=False)
    cb = fig.colorbar(im_res, ax=ax_res, fraction=0.048, pad=0.025)
    cb.ax.tick_params(labelsize=7.5, colors=MUTED)
    cb.outline.set_edgecolor("#51606d")

    ax_cost = fig.add_subplot(gs[1, :2])
    _style(ax_cost)
    image = ax_cost.imshow(
        fine.costs,
        origin="lower",
        aspect="auto",
        extent=[fine.values_x[0], fine.values_x[-1], fine.values_y[0], fine.values_y[-1]],
        cmap=DIFF,
    )
    ax_cost.scatter([truth_dec_um], [truth_pointing_mrad], marker="x", s=95,
                    color=RED, linewidths=2.0, label="injected truth")
    ax_cost.scatter([fine.best_x], [fine.best_y], marker="o", s=68, facecolors="none",
                    edgecolors=FG, linewidths=1.8, label="recovered grid minimum")
    ax_cost.set_xlabel("axicon decentre x (µm)")
    ax_cost.set_ylabel("input pointing x (mrad)")
    ax_cost.set_title("joint inverse cost landscape — 17 fixed-lab z planes", fontsize=11.8, weight="bold")
    ax_cost.legend(frameon=False, labelcolor=FG, fontsize=8.5, loc="upper left")
    c2 = fig.colorbar(image, ax=ax_cost, fraction=0.025, pad=0.018)
    c2.set_label("plane-normalised morphology RMSE", color=MUTED, fontsize=8.2)
    c2.ax.tick_params(labelsize=7.5, colors=MUTED)
    c2.outline.set_edgecolor("#51606d")

    ax_txt = fig.add_subplot(gs[1, 2])
    ax_txt.set_facecolor(DARK)
    ax_txt.axis("off")
    ax_txt.text(0.02, 0.94, "interpretable physical metrics", color=CYAN,
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
        ax_txt.text(0.03, y, label, color=MUTED, fontsize=9.2, va="center")
        ax_txt.text(0.97, y, value, color=FG, fontsize=10.0, weight="bold", ha="right", va="center")
        y -= 0.085
    ax_txt.text(0.03, 0.105,
                f"If pointing were incorrectly fixed to 0,\nthe coarse fit would prefer ~{naive_dec_um:+.0f} µm decentre.",
                color=GOLD, fontsize=9.1, linespacing=1.35)
    ax_txt.text(0.03, 0.018,
                "Synthetic validation only — not a claim of unique\nexperimental diagnosis or statistical uncertainty.",
                color=RED, fontsize=8.2, linespacing=1.25)

    fig.suptitle("Can the digital twin recover physical errors from a propagation signature?",
                 color=FG, fontsize=18.0, weight="bold", y=0.985)
    fig.text(0.5, 0.944,
             "Off-grid truth + simultaneous axicon/pointing errors + deterministic camera-like noise; all candidate fields are replayed through the same physical forward route.",
             ha="center", color=MUTED, fontsize=9.6)

    png = _save(fig, out / "02_joint_physical_error_recovery.png")

    summary = {
        "benchmark_scope": "synthetic model-to-model validation, simultaneous off-grid errors with deterministic noise",
        "grid_n": int(grid_n),
        "z_planes": int(len(z)),
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
            "nominal_zero_zero_cost": nominal_cost,
        },
        "claim_boundary": "discrete grid estimates and cost separation are not statistical confidence intervals; experimental identifiability requires calibration and uncertainty analysis",
    }

    csv_path = out / "02_joint_physical_error_recovery_summary.csv"
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

    return png, summary, csv_path


def copy_experimental(out: Path) -> list[Path]:
    sources = [
        ("03_experimental_retrieved_residual_phase.png",
         ROOT / "figures/experimental/q20_aberration/reconstruction/annular_aberration_phase.png"),
        ("04_experimental_inverse_confirmation.png",
         ROOT / "figures/experimental/q20_aberration/single_mask/single_z_double_confirmation_minus10.png"),
        ("05_experimental_phase_recreation_xz_yz.png",
         ROOT / "figures/experimental/q20_aberration/phase_error_recreation/phase_error_recreation_signed_xz_yz.png"),
    ]
    copied = []
    for name, src in sources:
        if not src.exists():
            raise FileNotFoundError(src)
        dst = out / name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def contact_sheet(out: Path, paths: list[Path]) -> Path:
    fig = plt.figure(figsize=(18, 14.5), facecolor="#171b1f")
    gs = fig.add_gridspec(3, 2, hspace=0.18, wspace=0.09)
    for i, p in enumerate(paths[:6]):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.set_facecolor("#171b1f")
        ax.imshow(plt.imread(p))
        ax.set_title(p.stem.replace("_", " "), color="white", fontsize=12, weight="bold")
        ax.axis("off")
    fig.suptitle("Poster core evidence v4 — focused simulation → inference → experimental retrieval",
                 color="white", fontsize=20, weight="bold", y=0.995)
    return _save(fig, out / "00_core_evidence_v4_contact_sheet.png", dpi=180)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p1, m1 = build_self_healing_path(OUT)
    p2, m2, csv2 = build_joint_parameter_recovery(OUT)
    exp = copy_experimental(OUT)
    sheet = contact_sheet(OUT, [p1, p2] + exp)
    manifest = {
        "outcome": "POSTER-CORE-EVIDENCE-V4",
        "story": "simulation -> visible self-healing/physics -> physical-error inference -> experimental residual-phase retrieval",
        "self_healing": m1,
        "joint_physical_error_recovery": m2,
        "experimental_figures": [str(p) for p in exp],
        "files": [str(sheet), str(p1), str(p2), str(csv2)] + [str(p) for p in exp],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
