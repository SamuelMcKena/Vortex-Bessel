"""Polished poster-core figures for the focused digital-twin/correction story.

This pass intentionally leaves the optical physics unchanged.  It improves the
presentation of three pieces that deserve poster space:

* Bessel self-healing with the opaque obstruction visible in the x-z map;
* interpretable synthetic physical-error diagnosis (axicon x-decentre);
* measured q=20 residual-phase retrieval using canonical tracked outputs.

The beam-intensity panels use the same black -> red -> orange -> yellow visual
language as the main presentation figures.  Signed residuals use a diverging map.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle
import numpy as np
import pandas as pd

import bessel_twin_core as bt
import publication_diagnostics as pdiag
from vbb_study import vbb_studies
from vbb_study.digital_twin.physical_error_diagnostics import (
    parameter_definition,
    system_error_config_for_parameter,
)
from vbb_study.digital_twin.physical_parameter_inference import fit_scalar_parameter
from vbb_study.digital_twin.vortex_system_route import build_system_route

import build_presentation_extended_evidence as ext

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "poster" / "figure_shortlist_v2"
DARK = "#07090c"
AX_BG = "#0a0d11"
FG = "#f3f5f6"
MUTED = "#aeb7bf"
CYAN = "#55ddd5"
GOLD = "#ffd166"
RED = "#ff5349"
GREEN = "#45d6a8"

# Presentation thermal: deliberately avoids the purple low-intensity region of
# stock inferno so it visually matches the existing deck's black/red/orange/yellow maps.
THERMAL = LinearSegmentedColormap.from_list(
    "vbb_presentation_thermal",
    [
        (0.00, "#000000"),
        (0.10, "#120000"),
        (0.24, "#4a0000"),
        (0.40, "#9d0800"),
        (0.56, "#e22b00"),
        (0.72, "#ff6a00"),
        (0.86, "#ffb000"),
        (1.00, "#fff36a"),
    ],
)


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#59636d")
        spine.set_linewidth(0.8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(FG)


def _norm(values: np.ndarray, scale: float | None = None) -> np.ndarray:
    arr = np.maximum(np.asarray(values, dtype=float), 0.0)
    denom = float(np.max(arr)) if scale is None else float(scale)
    return arr / max(denom, np.finfo(float).tiny)


def _crop_plane(plane: np.ndarray, full_grid: dict, crop_grid: dict) -> np.ndarray:
    x_full = np.asarray(full_grid["x"], float)
    x_crop = np.asarray(crop_grid["x"], float)
    ids = np.flatnonzero((x_full >= x_crop[0]) & (x_full <= x_crop[-1]))
    return np.asarray(plane, float)[np.ix_(ids, ids)] if ids.size >= 4 else np.asarray(plane, float)


def build_self_healing_v4() -> tuple[Path, dict]:
    """Render obstruction -> bypass -> reconstruction in one readable figure."""
    cfg = bt.default_config("balanced")
    cfg = vbb_studies.beam_air_config(cfg)
    cfg = replace(cfg, target=replace(cfg.target, ell=0))
    design = bt.compute_design_from_config(cfg)
    obstacle_radius_m = 0.90 * float(design.equivalent_l0_first_zero_radius_m)

    bundle = pdiag.build_self_healing_bundle(
        config=cfg,
        preset="balanced",
        path="ideal",
        case_id="poster_B0_self_healing_v4",
        obstacle_kind="disk",
        obstacle_radius_m=obstacle_radius_m,
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

    obstacle_radius_um = float(obstacle_radius_m / bt.um)
    half_um = min(float(np.max(np.abs(x_um))), max(12.0, 8.0 * obstacle_radius_um))
    ids = np.flatnonzero(np.abs(x_um) <= half_um)
    if ids.size >= 8:
        xp = x_um[ids]
        xz_plot = xz[ids, :]
        ref_plot = ref[np.ix_(ids, ids)]
        blocked_plot = blocked[np.ix_(ids, ids)]
        stack_plot = stack[:, ids][:, :, ids]
    else:
        xp, xz_plot, ref_plot, blocked_plot, stack_plot = x_um, xz, ref, blocked, stack

    onaxis = np.asarray(bundle["onaxis_recovery"], float)
    peak = np.asarray(bundle["peak_recovery"], float)
    recovered = np.flatnonzero(onaxis >= 0.80)
    recovery_idx = int(recovered[0]) if recovered.size else int(np.nanargmax(onaxis))
    recovery_z_um = float(z_um[recovery_idx])
    recovered_xy = np.asarray(stack_plot[recovery_idx], float)
    shared = float(np.max(ref_plot))

    fig = plt.figure(figsize=(15.9, 8.1), facecolor=DARK)
    gs = fig.add_gridspec(
        2, 4,
        height_ratios=[1.34, 0.72],
        width_ratios=[2.25, 0.82, 0.82, 0.82],
        hspace=0.30,
        wspace=0.18,
    )

    ax = fig.add_subplot(gs[0, 0])
    _style(ax)
    ax.imshow(
        _norm(xz_plot, shared) ** 0.43,
        origin="lower",
        extent=[float(z_um[0]), float(z_um[-1]), float(xp[0]), float(xp[-1])],
        cmap=THERMAL,
        vmin=0,
        vmax=1,
        interpolation="bilinear",
        aspect="auto",
    )
    object_z = float(z_um[0])
    object_width = max(0.016 * (float(z_um[-1]) - object_z), 0.45)
    ax.add_patch(
        Rectangle(
            (object_z, -obstacle_radius_um), object_width, 2.0 * obstacle_radius_um,
            facecolor="#9da4aa", edgecolor="white", linewidth=1.0, alpha=0.96, zorder=8,
        )
    )
    # Geometry guides make the self-healing mechanism immediately visible but
    # are explicitly explanatory, not geometrical-optics ray traces.
    z0 = object_z + object_width
    ax.fill(
        [z0, recovery_z_um, z0],
        [obstacle_radius_um, 0.0, -obstacle_radius_um],
        facecolor=CYAN, alpha=0.055, edgecolor="none", zorder=5,
    )
    ax.plot([z0, recovery_z_um], [obstacle_radius_um, 0.0], ls="--", lw=1.25, color=CYAN, alpha=0.92)
    ax.plot([z0, recovery_z_um], [-obstacle_radius_um, 0.0], ls="--", lw=1.25, color=CYAN, alpha=0.92)
    ax.axvline(recovery_z_um, color=GOLD, ls=":", lw=1.25, alpha=0.85)

    ax.annotate(
        "opaque obstruction",
        xy=(object_z + 0.5*object_width, -obstacle_radius_um),
        xytext=(0.05, 0.12),
        textcoords="axes fraction",
        color=FG,
        fontsize=9.5,
        weight="bold",
        arrowprops=dict(arrowstyle="->", color=FG, lw=1.0),
        bbox=dict(boxstyle="round,pad=0.25", facecolor=DARK, edgecolor="#56616b", alpha=0.86),
    )
    ax.annotate(
        "reconstruction region",
        xy=(recovery_z_um, 0.0),
        xytext=(0.62, 0.88),
        textcoords="axes fraction",
        color=GOLD,
        fontsize=9.5,
        weight="bold",
        arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.0),
        bbox=dict(boxstyle="round,pad=0.25", facecolor=DARK, edgecolor="#7f6b28", alpha=0.86),
    )
    ax.set_xlabel("distance after obstruction (µm)")
    ax.set_ylabel("x at fixed y=0 (µm)")
    ax.set_title("Longitudinal field: the blocked core reconstructs from the surviving conical spectrum", fontsize=11.6, weight="bold")
    ax.grid(False)

    extent_xy = [float(xp[0]), float(xp[-1]), float(xp[0]), float(xp[-1])]
    panels = [
        (ref_plot, "before"),
        (blocked_plot, "obstruction plane"),
        (recovered_xy, f"recovered\n{recovery_z_um:.0f} µm later"),
    ]
    for col, (plane, title) in enumerate(panels, start=1):
        q = fig.add_subplot(gs[0, col])
        _style(q)
        q.imshow(
            _norm(plane, shared) ** 0.43,
            origin="lower",
            extent=extent_xy,
            cmap=THERMAL,
            vmin=0,
            vmax=1,
            interpolation="bilinear",
        )
        if col == 2:
            q.add_patch(Circle((0, 0), obstacle_radius_um, facecolor="#9da4aa", edgecolor="white", lw=1.0, alpha=0.96))
        q.set_title(title, fontsize=10.2, weight="bold")
        q.set_xlabel("x (µm)")
        if col == 1:
            q.set_ylabel("y (µm)")
        else:
            q.tick_params(labelleft=False)
        q.grid(False)

    curve = fig.add_subplot(gs[1, :])
    _style(curve)
    curve.plot(z_um, peak, color=CYAN, lw=2.4, label="peak recovery")
    curve.plot(z_um, onaxis, color=GOLD, lw=2.2, ls="--", label="on-axis recovery")
    curve.axhline(0.80, color=GOLD, lw=0.85, ls=":", alpha=0.75)
    curve.axhline(1.00, color=MUTED, lw=0.85, ls=":", alpha=0.55)
    curve.axvline(recovery_z_um, color=GOLD, lw=0.85, ls=":", alpha=0.75)
    curve.text(
        recovery_z_um, 0.05,
        f"first ≥80% on-axis recovery: {recovery_z_um:.0f} µm",
        color=GOLD, fontsize=9, ha="left", va="bottom",
    )
    curve.set_xlabel("distance after obstruction (µm)")
    curve.set_ylabel("obstructed / unobstructed")
    curve.set_ylim(0, min(1.45, max(1.13, 1.05 * float(np.nanmax(peak)))))
    curve.grid(alpha=0.12)
    curve.legend(frameon=False, ncol=2, labelcolor=FG, loc="lower right")

    fig.suptitle("Bessel self-healing after a local opaque obstruction", color=FG, fontsize=18, weight="bold", y=0.985)
    fig.text(
        0.5, 0.945,
        "Same black→red→orange→yellow thermal language as the main beam figures. Grey mask thickness is exaggerated for visibility; cyan guides are explanatory envelopes, not ray traces.",
        ha="center", color=MUTED, fontsize=9.4,
    )
    path = OUT / "01_self_healing_object_path_v4.png"
    fig.savefig(path, dpi=380, bbox_inches="tight", facecolor=DARK, pad_inches=0.06)
    plt.close(fig)
    meta = {
        "obstacle_kind": "disk",
        "obstacle_radius_um": obstacle_radius_um,
        "first_80pct_onaxis_recovery_um": recovery_z_um,
        "display_colormap": "custom black-red-orange-yellow thermal",
        "claim_boundary": "thin opaque mask plane; z-thickness exaggerated; cyan guides are explanatory envelope only",
    }
    return path, meta


def build_physical_diagnosis_v4(grid_n: int = 512) -> tuple[Path, dict]:
    """Recover a hidden physical axicon offset and report an engineering-unit metric."""
    parameter_name = "axicon_decentre_x_um"
    definition = parameter_definition(parameter_name)
    z_m = np.linspace(20e-3, 100e-3, 17)
    coord_m = np.linspace(-0.75e-3, 0.75e-3, 241)
    truth_um = 300.0
    candidates_um = np.asarray(definition.recommended_screen, dtype=float)

    truth_route = build_system_route(
        "V1", grid_n=int(grid_n),
        config=system_error_config_for_parameter(parameter_name, truth_um),
    )
    target_xz, retained_truth = ext._xz(
        truth_route, coord_m, z_values_m=z_m, label="poster-v4-hidden-axicon-decentre"
    )

    candidate_maps: list[np.ndarray] = []
    retained: list[float] = []
    for value_um in candidates_um:
        route = build_system_route(
            "V1", grid_n=int(grid_n),
            config=system_error_config_for_parameter(parameter_name, float(value_um)),
        )
        xz, fraction = ext._xz(
            route, coord_m, z_values_m=z_m,
            label=f"poster-v4-axicon-decentre-{value_um:+.0f}um",
        )
        candidate_maps.append(xz)
        retained.append(float(fraction))

    fit = fit_scalar_parameter(
        target_xz,
        candidate_maps,
        candidates_um,
        parameter_name=parameter_name,
        parameter_unit="µm",
        nominal_value=0.0,
    )
    best_idx = int(np.argmin(np.asarray(fit.costs)))
    nominal_idx = int(np.argmin(np.abs(candidates_um)))
    abs_error_um = abs(float(fit.best_value) - truth_um)

    fig = plt.figure(figsize=(14.6, 6.9), facecolor=DARK)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.02, 1.02, 1.26, 0.92], hspace=0.28, wspace=0.22)

    def draw_xz(axis: plt.Axes, values: np.ndarray, title: str, shared_peak: float) -> None:
        _style(axis)
        axis.imshow(
            _norm(values, shared_peak).T ** 0.42,
            origin="lower",
            extent=[z_m[0]*1e3, z_m[-1]*1e3, coord_m[0]*1e3, coord_m[-1]*1e3],
            cmap=THERMAL, vmin=0, vmax=1, interpolation="bilinear", aspect="auto",
        )
        axis.set_title(title, fontsize=10.8, weight="bold")
        axis.set_xlabel("z from axicon (mm)")
        axis.set_ylabel("x at fixed y=0 (mm)")
        axis.grid(False)

    shared = float(np.max(target_xz))
    draw_xz(fig.add_subplot(gs[0, 0]), target_xz, "synthetic z-stack\n(hidden error)", shared)
    draw_xz(fig.add_subplot(gs[0, 1]), candidate_maps[nominal_idx], "nominal model\n0 µm", shared)
    draw_xz(fig.add_subplot(gs[1, 0]), candidate_maps[best_idx], f"best physical fit\n{fit.best_value:+.0f} µm", shared)

    # Show the *nominal* mismatch, not the exact-fit residual.  The latter is
    # intentionally near zero in this model-to-model validation and is visually uninformative.
    nominal_residual = _norm(candidate_maps[nominal_idx]) - _norm(target_xz)
    rax = fig.add_subplot(gs[1, 1])
    _style(rax)
    lim = max(float(np.max(np.abs(nominal_residual))), 1e-12)
    im = rax.imshow(
        nominal_residual.T,
        origin="lower",
        extent=[z_m[0]*1e3, z_m[-1]*1e3, coord_m[0]*1e3, coord_m[-1]*1e3],
        cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="bilinear", aspect="auto",
    )
    rax.set_title("nominal − target residual\n(error signature)", fontsize=10.8, weight="bold")
    rax.set_xlabel("z from axicon (mm)")
    rax.set_ylabel("x (mm)")
    cbar = fig.colorbar(im, ax=rax, pad=0.02, shrink=0.78)
    cbar.ax.tick_params(labelsize=7, colors=MUTED)
    cbar.set_label("signed normalised residual", color=MUTED, fontsize=8)

    cax = fig.add_subplot(gs[:, 2])
    _style(cax)
    cax.plot(candidates_um, fit.costs, marker="o", ms=5, lw=2.0, color=CYAN)
    cax.axvline(truth_um, color=GREEN, lw=1.5, ls="--", label="hidden truth")
    cax.axvline(fit.best_value, color=RED, lw=1.5, ls=":", label="recovered")
    cax.set_xlabel("candidate axicon x-decentre (µm)")
    cax.set_ylabel("17-plane morphology RMSE")
    cax.set_title("fit the physical parameter\nthrough the forward model", fontsize=11.5, weight="bold")
    cax.grid(alpha=0.12)
    cax.legend(frameon=False, labelcolor=FG, fontsize=9)

    info = fig.add_subplot(gs[:, 3])
    info.set_facecolor(DARK)
    info.axis("off")
    info.text(0.02, 0.93, "INTERPRETABLE OUTPUT", color=RED, fontsize=10.5, weight="bold")
    info.text(0.02, 0.80, "axicon x-decentre", color=MUTED, fontsize=10)
    info.text(0.02, 0.70, f"{fit.best_value:+.0f} µm", color=GREEN, fontsize=24, weight="bold")
    info.text(0.02, 0.57, f"truth check: {truth_um:+.0f} µm", color=FG, fontsize=11.5)
    info.text(0.02, 0.49, f"recovery error: {abs_error_um:.0f} µm", color=FG, fontsize=11.5)
    info.text(0.02, 0.41, f"search step: {fit.grid_step:.0f} µm", color=MUTED, fontsize=10.5)
    info.text(0.02, 0.31, f"nominal cost reduction\n{100*fit.improvement_fraction_vs_nominal:.1f}%", color=CYAN, fontsize=11.5, weight="bold")
    info.text(
        0.02, 0.08,
        "Same diagnostic layer can screen\ninput pointing, SLM1 registration\nand Fourier-iris offset before the\nremaining residual phase is retrieved.",
        color=MUTED, fontsize=9.2, linespacing=1.45,
    )

    fig.suptitle("Digital-twin diagnosis: recover a physical bench parameter from its z-stack signature", color=FG, fontsize=17.5, weight="bold", y=0.98)
    fig.text(
        0.5, 0.935,
        "Model-to-model validation only: the +300 µm truth is hidden from the fit and used solely to verify recovery. A real measured estimate still requires calibration and identifiability checks.",
        ha="center", color=MUTED, fontsize=9.35,
    )
    path = OUT / "08_synthetic_axicon_decentre_diagnosis_v4.png"
    fig.savefig(path, dpi=380, bbox_inches="tight", facecolor=DARK, pad_inches=0.06)
    plt.close(fig)

    metrics = fit.as_dict()
    metrics.update({
        "synthetic_truth_um": truth_um,
        "absolute_recovery_error_um": abs_error_um,
        "z_plane_count": int(len(z_m)),
        "physical_plane": definition.physical_plane,
        "diagnostic_description": definition.description,
        "fixed_support_retained_power_fraction_truth": float(retained_truth),
        "fixed_support_retained_power_fraction_candidates": retained,
        "scope": "synthetic physical-parameter recovery; not a calibrated measured bench tolerance",
    })
    (OUT / "08_synthetic_axicon_decentre_diagnosis_v4.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame({
        "candidate_axicon_x_decentre_um": candidates_um,
        "multi_plane_rmse": fit.costs,
    }).to_csv(OUT / "08_synthetic_axicon_decentre_cost_curve_v4.csv", index=False)
    return path, metrics


def build_experimental_phase_retrieval_v4() -> Path:
    """Make the canonical retrieved residual phase the visual hero."""
    phase_path = ROOT / "figures/experimental/q20_aberration/reconstruction/annular_aberration_phase.png"
    polar_path = ROOT / "figures/experimental/q20_aberration/reconstruction/polar_measured_fit_corrected.png"
    if not phase_path.exists() or not polar_path.exists():
        raise FileNotFoundError("canonical q20 phase-retrieval evidence is missing")

    phase_img = plt.imread(phase_path)
    polar_img = plt.imread(polar_path)

    fig = plt.figure(figsize=(14.4, 6.5), facecolor=DARK)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.52, 0.88, 0.62], wspace=0.06, left=0.025, right=0.985, bottom=0.09, top=0.80)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(phase_img)
    ax0.axis("off")
    ax0.set_title("retrieved annular residual phase", color=FG, fontsize=12.5, weight="bold", pad=9)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(polar_img)
    ax1.axis("off")
    ax1.set_title("measured field → modal fit → predicted correction", color=FG, fontsize=10.5, weight="bold", pad=9)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(DARK)
    ax2.axis("off")
    ax2.text(0.02, 0.93, "WHAT IS RETRIEVED?", color=RED, fontsize=10.2, weight="bold")
    ax2.text(0.02, 0.80, "angular residual", color=FG, fontsize=12, weight="bold")
    ax2.text(0.02, 0.72, "+", color=MUTED, fontsize=12)
    ax2.text(0.02, 0.64, "radial phase term", color=FG, fontsize=12, weight="bold")
    ax2.text(0.02, 0.53, "=", color=MUTED, fontsize=12)
    ax2.text(0.02, 0.44, "residual phase\nfor correction", color=GREEN, fontsize=14, weight="bold")
    ax2.text(
        0.02, 0.22,
        "The programmed q=20 vortex is\nkept as desired beam structure —\nit is not removed as an aberration.",
        color=MUTED, fontsize=9.4, linespacing=1.4,
    )
    ax2.text(
        0.02, 0.05,
        "Hardware use remains gated by\nbranch, camera-axis, relay/conjugacy\nand measured SLM-LUT calibration.",
        color="#ffb4ae", fontsize=8.7, linespacing=1.35,
    )

    fig.suptitle("Experimental q=20 phase retrieval from the measured BMG z-stack", color=FG, fontsize=17.5, weight="bold", y=0.96)
    fig.text(
        0.5, 0.865,
        "Canonical tracked experimental reconstruction: use the measured multi-plane intensity to estimate the residual field, then predict an additive SLM correction.",
        ha="center", color=MUTED, fontsize=9.6,
    )
    path = OUT / "09_experimental_phase_retrieval_v4.png"
    fig.savefig(path, dpi=330, bbox_inches="tight", facecolor=DARK, pad_inches=0.06)
    plt.close(fig)
    return path


def build_contact_sheet(paths: list[Path]) -> Path:
    fig = plt.figure(figsize=(18, 10.8), facecolor="#15191d")
    gs = fig.add_gridspec(2, 2, hspace=0.16, wspace=0.08)
    for i, path in enumerate(paths):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.set_facecolor("#15191d")
        ax.imshow(plt.imread(path))
        ax.set_title(path.stem.replace("_", " "), color="white", fontsize=11.5, weight="bold")
        ax.axis("off")
    fig.suptitle("Focused poster additions v4 — inspect before poster layout", color="white", fontsize=20, weight="bold", y=0.99)
    fig.text(
        0.5, 0.012,
        "self-healing mechanism · interpretable physical-error metric · measured q20 residual-phase retrieval",
        ha="center", color=CYAN, fontsize=11.5, weight="bold",
    )
    out = OUT / "00_focused_additions_contact_sheet_v4.png"
    fig.savefig(out, dpi=190, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    self_heal, self_meta = build_self_healing_v4()
    diagnosis, diag_meta = build_physical_diagnosis_v4()
    phase = build_experimental_phase_retrieval_v4()
    sheet = build_contact_sheet([self_heal, diagnosis, phase])
    (OUT / "focused_additions_v4_manifest.json").write_text(
        json.dumps({
            "self_healing": self_meta,
            "physical_diagnosis": diag_meta,
            "phase_retrieval_source": "canonical tracked q20 experimental reconstruction outputs",
            "contact_sheet": str(sheet),
            "figures": [str(self_heal), str(diagnosis), str(phase)],
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(sheet)
    print(self_heal)
    print(diagnosis)
    print(phase)


if __name__ == "__main__":
    main()
