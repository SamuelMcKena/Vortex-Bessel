"""Poster-core figure curation: propagation, physical diagnosis, phase retrieval.

This pass follows the focused project story rather than catalogue breadth:

    physical optical model -> nominal constraints -> controlled errors
    -> physical-parameter diagnosis -> experimental residual-phase retrieval
    -> correction / independent remeasurement.

Two new poster-facing figures are generated from current code:

1. a self-healing x-z map that explicitly shows the obstruction plane and the
   reconstruction region in the same thermal palette as the main beam figures;
2. a synthetic physical-parameter diagnosis in which a hidden axicon x-decentre
   is recovered from a 17-plane forward-model stack and reported as an actual
   physical metric.

The experimental phase-retrieval panel is composed only from canonical tracked
q=20 outputs.  No experimental image or correction result is fabricated.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

import bessel_twin_core as bt
import publication_diagnostics as pdiag
from vbb_study import vbb_studies
from vbb_study.digital_twin.physical_parameter_inference import fit_scalar_parameter
from vbb_study.digital_twin.vortex_system_route import AxiconError, SystemErrorConfig, build_system_route

import build_presentation_extended_evidence as ext

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "poster" / "figure_shortlist_v2"
DARK = "#070a0d"
AX_BG = "#0c1015"
FG = "#f2f4f6"
MUTED = "#aab5bf"
CYAN = "#4ddad1"
GOLD = "#ffd166"
RED = "#ff5a52"
GREEN = "#45d6a8"
THERMAL = "inferno"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#5b6670")
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(FG)


def _norm(values: np.ndarray, scale: float | None = None) -> np.ndarray:
    a = np.maximum(np.asarray(values, dtype=float), 0.0)
    denom = float(np.max(a)) if scale is None else float(scale)
    return a / max(denom, np.finfo(float).tiny)


def _crop_plane(plane: np.ndarray, full_grid: dict, crop_grid: dict) -> np.ndarray:
    x_full = np.asarray(full_grid["x"], float)
    x_crop = np.asarray(crop_grid["x"], float)
    ids = np.flatnonzero((x_full >= x_crop[0]) & (x_full <= x_crop[-1]))
    return np.asarray(plane, float)[np.ix_(ids, ids)] if ids.size >= 4 else np.asarray(plane, float)


def build_self_healing_path() -> tuple[Path, dict]:
    """Show the obstructing object directly in the longitudinal propagation map."""
    cfg = bt.default_config("balanced")
    cfg = vbb_studies.beam_air_config(cfg)
    cfg = replace(cfg, target=replace(cfg.target, ell=0))
    design = bt.compute_design_from_config(cfg)
    radius_m = 0.90 * float(design.equivalent_l0_first_zero_radius_m)

    bundle = pdiag.build_self_healing_bundle(
        config=cfg,
        preset="balanced",
        path="ideal",
        case_id="poster_B0_self_healing_path_v3",
        obstacle_kind="disk",
        obstacle_radius_m=radius_m,
        axial_points=101,
    )
    vol = bundle["obstructed_volume"]
    cgrid = vol["crop_grid"]
    x_um = np.asarray(cgrid["x"], float) / bt.um
    z_um = np.asarray(bundle["z_relative"], float) / bt.um
    xz = np.asarray(vol["xz"], float)
    ref = _crop_plane(bundle["reference_plane"], bundle["grid"], cgrid)
    blocked = _crop_plane(bundle["obstructed_plane"], bundle["grid"], cgrid)
    stack = np.asarray(vol["intensity_stack"], float)

    radius_um = float(radius_m / bt.um)
    half_um = min(float(np.max(np.abs(x_um))), max(12.0, 8.0 * radius_um))
    xid = np.flatnonzero(np.abs(x_um) <= half_um)
    if xid.size >= 8:
        x_plot = x_um[xid]
        xz_plot = xz[xid, :]
        blocked_plot = blocked[np.ix_(xid, xid)]
        ref_plot = ref[np.ix_(xid, xid)]
        stack_plot = stack[:, xid][:, :, xid]
    else:
        x_plot, xz_plot, blocked_plot, ref_plot, stack_plot = x_um, xz, blocked, ref, stack

    onaxis = np.asarray(bundle["onaxis_recovery"], float)
    peak = np.asarray(bundle["peak_recovery"], float)
    good = np.flatnonzero(onaxis >= 0.80)
    recovery_idx = int(good[0]) if good.size else int(np.nanargmax(onaxis))
    recovery_z_um = float(z_um[recovery_idx])
    recovered_xy = np.asarray(stack_plot[recovery_idx], float)
    shared = float(np.max(ref_plot))

    fig = plt.figure(figsize=(15.8, 8.3), facecolor=DARK)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.30, 0.80], width_ratios=[2.15, 0.80, 0.80, 0.80], hspace=0.28, wspace=0.18)

    ax = fig.add_subplot(gs[0, 0])
    _style(ax)
    ax.imshow(
        _norm(xz_plot, shared) ** 0.43,
        origin="lower",
        extent=[float(z_um[0]), float(z_um[-1]), float(x_plot[0]), float(x_plot[-1])],
        cmap=THERMAL,
        vmin=0,
        vmax=1,
        interpolation="bilinear",
        aspect="auto",
    )
    object_width = max(0.018 * (float(z_um[-1]) - float(z_um[0])), 0.5)
    object_z = float(z_um[0])
    ax.add_patch(Rectangle((object_z, -radius_um), object_width, 2.0 * radius_um,
                           facecolor="#9aa2a8", edgecolor="white", linewidth=1.0, alpha=0.95, zorder=8))
    ax.annotate(
        "opaque disk\n(mask plane)",
        xy=(object_z + 0.5 * object_width, radius_um),
        xytext=(object_z + 0.13 * (z_um[-1]-z_um[0]), 0.78 * half_um),
        color=FG,
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color=FG, lw=1.1),
    )
    # These dashed lines are an explanatory reconstruction envelope, not ray traces.
    ax.plot([object_z + object_width, recovery_z_um], [radius_um, 0.0], ls="--", lw=1.2, color=CYAN, alpha=0.9)
    ax.plot([object_z + object_width, recovery_z_um], [-radius_um, 0.0], ls="--", lw=1.2, color=CYAN, alpha=0.9)
    ax.axvline(recovery_z_um, color=GOLD, ls=":", lw=1.3, alpha=0.9)
    ax.text(recovery_z_um, 0.92*half_um, "80% on-axis\nrecovery", color=GOLD, fontsize=9, ha="center", va="top")
    ax.set_xlabel("distance after obstruction (µm)")
    ax.set_ylabel("x at fixed y=0 (µm)")
    ax.set_title("Longitudinal field: energy bypasses the blocked core and reconstructs", fontsize=12, weight="bold")
    ax.grid(False)

    extent_xy = [float(x_plot[0]), float(x_plot[-1]), float(x_plot[0]), float(x_plot[-1])]
    for col, (plane, title) in enumerate([
        (ref_plot, "before obstruction"),
        (blocked_plot, "at obstruction"),
        (recovered_xy, f"recovered\nz={recovery_z_um:.0f} µm"),
    ], start=1):
        q = fig.add_subplot(gs[0, col])
        _style(q)
        q.imshow(_norm(plane, shared) ** 0.43, origin="lower", extent=extent_xy,
                 cmap=THERMAL, vmin=0, vmax=1, interpolation="bilinear")
        if col == 2:
            q.add_patch(plt.Circle((0, 0), radius_um, facecolor="#9aa2a8", edgecolor="white", lw=1.0, alpha=0.95))
        q.set_title(title, fontsize=10.5, weight="bold")
        q.set_xlabel("x (µm)")
        if col == 1:
            q.set_ylabel("y (µm)")
        else:
            q.tick_params(labelleft=False)
        q.grid(False)

    curve = fig.add_subplot(gs[1, :])
    _style(curve)
    curve.plot(z_um, peak, color=CYAN, lw=2.6, label="peak-intensity recovery")
    curve.plot(z_um, onaxis, color=GOLD, lw=2.3, ls="--", label="on-axis recovery")
    curve.axhline(0.80, color=GOLD, lw=0.9, ls=":", alpha=0.75)
    curve.axhline(1.0, color=MUTED, lw=0.9, ls=":", alpha=0.55)
    curve.axvline(recovery_z_um, color=GOLD, lw=0.9, ls=":", alpha=0.75)
    curve.set_xlabel("distance after obstruction (µm)")
    curve.set_ylabel("obstructed / unobstructed")
    curve.set_ylim(0, min(1.45, max(1.12, 1.05*float(np.nanmax(peak)))))
    curve.grid(alpha=0.12)
    curve.legend(frameon=False, ncol=2, labelcolor=FG, loc="lower right")

    fig.suptitle("Bessel self-healing after a local opaque obstruction", color=FG, fontsize=18, weight="bold", y=0.985)
    fig.text(0.5, 0.943,
             "Thermal scale matches the main beam figures. Grey object thickness is exaggerated only for visibility; dashed cyan lines are reconstruction-envelope guides, not ray traces.",
             ha="center", color=MUTED, fontsize=9.6)
    out = OUT / "01_self_healing_object_path.png"
    fig.savefig(out, dpi=360, bbox_inches="tight", facecolor=DARK, pad_inches=0.06)
    plt.close(fig)
    return out, {
        "obstacle_kind": "disk",
        "obstacle_radius_um": radius_um,
        "onaxis_recovery_threshold": 0.80,
        "first_80pct_onaxis_recovery_um": recovery_z_um,
        "claim_boundary": "thin opaque mask plane; displayed z-thickness exaggerated; dashed guides are explanatory envelope only",
    }


def build_synthetic_physical_diagnosis(grid_n: int = 512) -> tuple[Path, dict]:
    """Recover a hidden axicon x-decentre and expose the numerical fit metrics."""
    z_m = np.linspace(20e-3, 100e-3, 17)
    coord_m = np.linspace(-0.75e-3, 0.75e-3, 241)
    truth_um = 300.0
    candidates_um = np.arange(-500.0, 500.1, 100.0)

    truth_route = build_system_route(
        "V1", grid_n=int(grid_n),
        config=SystemErrorConfig(axicon=AxiconError(decentre_m=(truth_um*1e-6, 0.0))),
    )
    target_xz, retained_truth = ext._xz(truth_route, coord_m, z_values_m=z_m, label="poster-v3-hidden-axicon-decentre")

    candidate_maps = []
    retained = []
    for value_um in candidates_um:
        route = build_system_route(
            "V1", grid_n=int(grid_n),
            config=SystemErrorConfig(axicon=AxiconError(decentre_m=(float(value_um)*1e-6, 0.0))),
        )
        xz, frac = ext._xz(route, coord_m, z_values_m=z_m, label=f"poster-v3-candidate-{value_um:+.0f}um")
        candidate_maps.append(xz)
        retained.append(float(frac))

    fit = fit_scalar_parameter(
        target_xz,
        candidate_maps,
        candidates_um,
        parameter_name="axicon x-decentre",
        parameter_unit="µm",
        nominal_value=0.0,
    )
    best_idx = int(np.argmin(np.asarray(fit.costs)))
    nominal_idx = int(np.argmin(np.abs(candidates_um)))
    abs_error_um = abs(float(fit.best_value) - truth_um)

    fig = plt.figure(figsize=(14.2, 7.0), facecolor=DARK)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.25, 1.0], hspace=0.30, wspace=0.22)

    def draw_xz(ax, values, title, peak=None):
        _style(ax)
        scale = float(np.max(values)) if peak is None else float(peak)
        ax.imshow(_norm(values, scale).T ** 0.42, origin="lower",
                  extent=[z_m[0]*1e3, z_m[-1]*1e3, coord_m[0]*1e3, coord_m[-1]*1e3],
                  cmap=THERMAL, vmin=0, vmax=1, interpolation="bilinear", aspect="auto")
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xlabel("z from axicon (mm)")
        ax.set_ylabel("x at fixed y=0 (mm)")
        ax.grid(False)

    shared = float(np.max(target_xz))
    draw_xz(fig.add_subplot(gs[0, 0]), target_xz, "synthetic target\nhidden error", shared)
    draw_xz(fig.add_subplot(gs[0, 1]), candidate_maps[nominal_idx], "nominal model\n0 µm", shared)
    draw_xz(fig.add_subplot(gs[1, 0]), candidate_maps[best_idx], f"best physical fit\n{fit.best_value:+.0f} µm", shared)

    residual = _norm(candidate_maps[best_idx]) - _norm(target_xz)
    rax = fig.add_subplot(gs[1, 1])
    _style(rax)
    lim = max(float(np.max(np.abs(residual))), 1e-12)
    rax.imshow(residual.T, origin="lower",
               extent=[z_m[0]*1e3, z_m[-1]*1e3, coord_m[0]*1e3, coord_m[-1]*1e3],
               cmap="coolwarm", vmin=-lim, vmax=lim, aspect="auto")
    rax.set_title("best-fit residual", fontsize=11, weight="bold")
    rax.set_xlabel("z from axicon (mm)")
    rax.set_ylabel("x (mm)")

    cost_ax = fig.add_subplot(gs[:, 2])
    _style(cost_ax)
    cost_ax.plot(candidates_um, fit.costs, marker="o", lw=2.0, color=CYAN)
    cost_ax.axvline(truth_um, color=GREEN, lw=1.5, ls="--", label="hidden truth")
    cost_ax.axvline(fit.best_value, color=RED, lw=1.5, ls=":", label="recovered")
    cost_ax.set_xlabel("candidate axicon x-decentre (µm)")
    cost_ax.set_ylabel("multi-plane morphology RMSE")
    cost_ax.set_title("17-plane physical-parameter fit", fontsize=12, weight="bold")
    cost_ax.grid(alpha=0.12)
    cost_ax.legend(frameon=False, labelcolor=FG, fontsize=9)

    info = fig.add_subplot(gs[:, 3])
    info.set_facecolor(DARK)
    info.axis("off")
    info.text(0.02, 0.92, "PHYSICAL DIAGNOSIS", color=RED, fontsize=11, weight="bold")
    info.text(0.02, 0.80, f"hidden decentre\n{truth_um:+.0f} µm", color=FG, fontsize=16, weight="bold")
    info.text(0.02, 0.64, f"recovered\n{fit.best_value:+.0f} µm", color=GREEN, fontsize=18, weight="bold")
    info.text(0.02, 0.51, f"absolute error\n{abs_error_um:.0f} µm", color=FG, fontsize=13)
    info.text(0.02, 0.39, f"search resolution\n{fit.grid_step:.0f} µm", color=MUTED, fontsize=12)
    info.text(0.02, 0.27, f"cost reduction vs nominal\n{100*fit.improvement_fraction_vs_nominal:.1f}%", color=CYAN, fontsize=12, weight="bold")
    info.text(0.02, 0.12,
              "Candidate-grid estimate, not a\nstatistical uncertainty. The same\nlayer can be extended to pointing,\nSLM registration and iris offset.",
              color=MUTED, fontsize=9.5, linespacing=1.4)

    fig.suptitle("Digital-twin error diagnosis: recover a physical error from its z-stack signature", color=FG, fontsize=18, weight="bold", y=0.98)
    fig.text(0.5, 0.935,
             "Model-to-model validation: the inverse sees only the synthetic intensity stack; the hidden +300 µm axicon decentre is used only for the final truth check.",
             ha="center", color=MUTED, fontsize=9.6)
    out = OUT / "08_synthetic_axicon_decentre_diagnosis.png"
    fig.savefig(out, dpi=360, bbox_inches="tight", facecolor=DARK, pad_inches=0.06)
    plt.close(fig)

    metrics = fit.as_dict()
    metrics.update({
        "synthetic_truth_um": truth_um,
        "absolute_recovery_error_um": abs_error_um,
        "z_plane_count": int(len(z_m)),
        "fixed_support_retained_power_fraction_truth": float(retained_truth),
        "fixed_support_retained_power_fraction_candidates": retained,
        "scope": "synthetic physical-parameter recovery; not a measured bench tolerance",
    })
    (OUT / "08_synthetic_axicon_decentre_diagnosis.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame({"candidate_axicon_x_decentre_um": candidates_um, "multi_plane_rmse": fit.costs}).to_csv(
        OUT / "08_synthetic_axicon_decentre_cost_curve.csv", index=False
    )
    return out, metrics


def build_experimental_phase_retrieval_panel() -> Path:
    """Compose canonical measured q=20 retrieval outputs without changing data."""
    phase_path = ROOT / "figures/experimental/q20_aberration/reconstruction/annular_aberration_phase.png"
    spectrum_path = ROOT / "figures/experimental/q20_aberration/reconstruction/modal_spectrum.png"
    if not phase_path.exists() or not spectrum_path.exists():
        raise FileNotFoundError("canonical q20 retrieval figures are missing")

    fig = plt.figure(figsize=(13.6, 5.9), facecolor=DARK)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.85], wspace=0.06, left=0.025, right=0.985, bottom=0.08, top=0.80)
    for ax, path, label in (
        (fig.add_subplot(gs[0, 0]), phase_path, "retrieved residual phase"),
        (fig.add_subplot(gs[0, 1]), spectrum_path, "retrieved modal content"),
    ):
        ax.imshow(plt.imread(path))
        ax.axis("off")
        ax.set_title(label, color=FG, fontsize=12, weight="bold", pad=8)

    fig.suptitle("Experimental q=20 phase retrieval from the measured BMG z-stack", color=FG, fontsize=17, weight="bold", y=0.96)
    fig.text(0.5, 0.865,
             "The programmed q=20 vortex is treated as desired beam structure; the inverse reconstructs the residual aberration to be corrected.",
             ha="center", color=MUTED, fontsize=9.8)
    fig.text(0.5, 0.025,
             "Canonical tracked retrieval outputs. Hardware application remains gated by branch, camera-axis, relay/conjugacy and SLM-LUT calibration.",
             ha="center", color="#ffb4ae", fontsize=9.0)
    out = OUT / "09_experimental_phase_retrieval.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=DARK, pad_inches=0.06)
    plt.close(fig)
    return out


def copy_core_figures() -> list[Path]:
    mapping = [
        (ROOT / "figures/presentation/01_computational_route_phase2j.png", "02_computational_route.png"),
        (ROOT / "figures/presentation/02_beam_profile_shaping_B0_V1_V3_thermal_tight.png", "03_ideal_beam_family.png"),
        (ROOT / "figures/presentation/04_V1_axicon_decentre_fixed_lab_thermal_tight.png", "05_axicon_decentre_example.png"),
        (ROOT / "figures/presentation/05_V1_nonideal_tip_fixed_lab_thermal_tight.png", "06_nonideal_apex_example.png"),
        (ROOT / "figures/experimental/q20_aberration/single_mask/single_z_double_confirmation_minus10.png", "10_q20_single_z_confirmation.png"),
        (ROOT / "figures/experimental/q20_aberration/phase_error_recreation/phase_error_recreation_signed_xz_yz.png", "11_q20_phase_error_recreation_xz_yz.png"),
    ]
    copied = []
    for src, name in mapping:
        if not src.exists():
            raise FileNotFoundError(src)
        dst = OUT / name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def contact_sheet(paths: list[Path]) -> Path:
    fig = plt.figure(figsize=(19, 22), facecolor="#15191d")
    gs = fig.add_gridspec(4, 3, hspace=0.18, wspace=0.08)
    for i, path in enumerate(paths):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ax.set_facecolor("#15191d")
        ax.imshow(plt.imread(path))
        ax.set_title(path.stem.replace("_", " "), color="white", fontsize=11, weight="bold")
        ax.axis("off")
    fig.suptitle("Poster core-story figure audit v3", color="white", fontsize=22, weight="bold", y=0.995)
    fig.text(0.5, 0.008,
             "Core story only: optical model → nominal constraints → controlled errors → physical diagnosis → measured phase retrieval → correction validation",
             ha="center", color="#4ddad1", fontsize=12, weight="bold")
    out = OUT / "00_core_story_contact_sheet_v3.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Remove the interface figure from the active shortlist: it is outside the
    # focused poster claim and has not been a major experimental workstream.
    for old in (
        "02_sample_interface_comparison.png",
        "01_self_healing_sequence.png",
        "00_shortlist_contact_sheet.png",
        "00_shortlist_contact_sheet_v21.png",
    ):
        p = OUT / old
        if p.exists():
            p.unlink()

    self_heal, heal_meta = build_self_healing_path()
    diagnosis, diagnosis_meta = build_synthetic_physical_diagnosis()
    phase = build_experimental_phase_retrieval_panel()
    copied = copy_core_figures()

    # Nominal-constraint figure is produced by the current presentation cleanup
    # workflow and may be promoted/copied into the final poster pack separately;
    # do not silently recreate a stale version here.
    ordered = [
        self_heal,
        copied[0],
        copied[1],
        copied[2],
        copied[3],
        diagnosis,
        phase,
        copied[4],
        copied[5],
    ]
    sheet = contact_sheet(ordered)
    (OUT / "core_story_v3_manifest.json").write_text(json.dumps({
        "self_healing": heal_meta,
        "physical_parameter_diagnosis": diagnosis_meta,
        "experimental_phase_retrieval_source": "canonical tracked q20 reconstruction outputs",
        "excluded_from_core": ["through-sample/interface propagation"],
        "contact_sheet": str(sheet),
        "figures": [str(p) for p in ordered],
    }, indent=2) + "\n", encoding="utf-8")
    print(sheet)
    for p in ordered:
        print(p)


if __name__ == "__main__":
    main()
