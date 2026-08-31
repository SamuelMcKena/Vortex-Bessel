"""Build the focused poster evidence requested after figure curation.

This builder deliberately follows the narrowed research story:

    physical-system simulation -> nominal constraints -> controlled errors ->
    synthetic physical-parameter recovery -> experimental phase retrieval/correction

It does NOT include the through-sample/interface figure.

New poster-facing outputs
-------------------------
01_self_healing_path.png
    A B0 self-healing figure dominated by a longitudinal XZ map.  The opaque
    obstruction is drawn explicitly at z=0 and a geometric cone-ray guide is
    overlaid only as a visual aid; the heatmap itself is simulated data.

02_physical_parameter_recovery.png
    Model-to-model validation of a low-dimensional diagnostic layer.  Four
    separate synthetic tests inject one known physical error at a time and fit
    that same parameter back through the current system route.  This is not a
    claim of unique experimental diagnosis.

03_experimental_phase_retrieval.png
    Canonical measured q=20 retrieved residual-phase output copied without
    alteration from the experimental reconstruction folder.

04_experimental_inverse_confirmation.png
    Canonical single-z measured/ideal/inverse confirmation copied without
    alteration.
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
from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.physical_error_inference import grid_search_parameter
from vbb_study.digital_twin.vortex_beam_slm_errors import GaussianBeamError, SLMError
from vbb_study.digital_twin.vortex_continuous_propagation import (
    build_fixed_plane_longitudinal_map,
    build_fixed_support_spectrum,
)
from vbb_study.digital_twin.vortex_explicit_4f import FourFError
from vbb_study.digital_twin.vortex_system_route import AxiconError, SystemErrorConfig, build_system_route

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "poster" / "core_evidence_v3"
DARK = "#070a0d"
AX = "#0b1015"
FG = "#f3f5f6"
MUTED = "#aab5bf"
CYAN = "#59e0d5"
GOLD = "#ffd166"
MAGENTA = "#ff6eb4"
GREEN = "#65df9b"
RED = "#ff6b6b"
THERMAL = "inferno"
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


def _save(fig: plt.Figure, path: Path, dpi: int = 320) -> Path:
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
    cfg = vbb_studies.beam_air_config(bt.default_config("balanced"))
    cfg = replace(cfg, target=replace(cfg.target, ell=0))
    design = bt.compute_design_from_config(cfg)
    obstacle_radius = 0.90 * float(design.equivalent_l0_first_zero_radius_m)

    bundle = pdiag.build_self_healing_bundle(
        config=cfg,
        preset="balanced",
        path="ideal",
        case_id="poster_B0_self_healing_path",
        obstacle_kind="disk",
        obstacle_radius_m=obstacle_radius,
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

    # Pick a visibly reconstructed plane from the recovery trace rather than an
    # arbitrary fraction of the z range.
    recovery = np.asarray(bundle["onaxis_recovery"], float)
    candidates = np.flatnonzero((recovery >= 0.85) & np.isfinite(recovery))
    recovered_idx = int(candidates[0]) if candidates.size else int(0.75 * (len(z_um) - 1))
    recovered = stack[recovered_idx]

    shared = float(np.max(ref))
    extent_xy = [x_um[0], x_um[-1], x_um[0], x_um[-1]]
    extent_xz = [z_um[0], z_um[-1], x_um[0], x_um[-1]]

    fig = plt.figure(figsize=(15.6, 7.4), facecolor=DARK)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.48, 1.0, 1.0, 1.0], height_ratios=[1.0, 0.52], wspace=0.20, hspace=0.30)

    ax_xz = fig.add_subplot(gs[:, 0])
    _style(ax_xz)
    ax_xz.imshow(_norm(xz) ** 0.44, origin="lower", aspect="auto", extent=extent_xz, cmap=THERMAL, vmin=0, vmax=1)
    ax_xz.set_title("longitudinal reconstruction", fontsize=13, weight="bold", pad=8)
    ax_xz.set_xlabel("distance after obstruction, z (µm)")
    ax_xz.set_ylabel("x at fixed y = 0 (µm)")

    # Explicit obstruction at the left boundary.  The rectangle is an annotation,
    # not simulated intensity, and matches the actual disk radius used above.
    z_span = max(float(z_um[-1] - z_um[0]), 1.0)
    obs_width = 0.018 * z_span
    ax_xz.add_patch(Rectangle((float(z_um[0]), -obstacle_radius / bt.um), obs_width, 2 * obstacle_radius / bt.um,
                              facecolor="black", edgecolor="white", linewidth=1.1, zorder=5))
    ax_xz.text(float(z_um[0]) + 0.028 * z_span, 1.25 * obstacle_radius / bt.um, "opaque disk", color=FG, fontsize=9, va="center")

    # Geometric conical-ray guide derived from kr/k.  It is deliberately dashed
    # and labelled as a guide so it is not confused with traced rays from the wave solver.
    k0 = 2.0 * np.pi / float(cfg.laser.wavelength_m)
    theta = math.asin(min(0.999999, abs(float(design.kr_sample_m_inv)) / k0))
    z_geom_um = float(obstacle_radius / max(math.tan(theta), EPS) / bt.um)
    z_guide = min(z_geom_um, float(z_um[-1]))
    r0_um = float(obstacle_radius / bt.um)
    ax_xz.plot([z_um[0], z_guide], [r0_um, 0.0], color=CYAN, ls="--", lw=1.4, alpha=0.85)
    ax_xz.plot([z_um[0], z_guide], [-r0_um, 0.0], color=CYAN, ls="--", lw=1.4, alpha=0.85)
    ax_xz.text(0.54, 0.95, "conical-wave guide", transform=ax_xz.transAxes, color=CYAN, fontsize=8.5, ha="center", va="top")

    transverse = [(ref, "before obstruction"), (blocked, "at obstruction"), (recovered, f"recovered\nz = {z_um[recovered_idx]:.0f} µm")]
    for col, (plane, title) in enumerate(transverse, start=1):
        ax = fig.add_subplot(gs[0, col])
        _style(ax)
        ax.imshow(_norm(plane, shared) ** 0.44, origin="lower", extent=extent_xy, cmap=THERMAL, vmin=0, vmax=1)
        ax.set_title(title, fontsize=11.5, weight="bold", pad=7)
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
    ax_r.axvline(z_um[recovered_idx], color=GREEN, lw=1.0, ls="--", alpha=0.8)
    ax_r.set_xlabel("distance after obstruction (µm)")
    ax_r.set_ylabel("obstructed / unobstructed")
    ax_r.set_ylim(0, min(1.5, max(1.15, 1.05 * float(np.nanmax(bundle["peak_recovery"])))))
    ax_r.legend(frameon=False, ncol=2, labelcolor=FG, loc="lower right")
    ax_r.grid(alpha=0.12)

    fig.suptitle("Bessel-beam self-healing: the field reconstructs behind a local obstruction", color=FG, fontsize=18, weight="bold", y=0.988)
    fig.text(0.5, 0.94, "Simulated intensity uses the same thermal palette as the main beam figures; the black obstacle and dashed cone lines are explicit annotations.", ha="center", color=MUTED, fontsize=9.7)

    meta = {
        "obstacle_radius_um": float(obstacle_radius / bt.um),
        "geometric_cone_half_angle_deg": float(np.degrees(theta)),
        "geometric_reconstruction_distance_um": z_geom_um,
        "selected_recovered_plane_um": float(z_um[recovered_idx]),
        "selected_onaxis_recovery": float(recovery[recovered_idx]),
        "claim_boundary": "heatmap is wave-propagation output; obstacle block and dashed conical-wave lines are annotations",
    }
    path = _save(fig, out / "01_self_healing_path.png")
    return path, meta


def _xz_stack(config: SystemErrorConfig, *, grid_n: int, z_m: np.ndarray, coord_m: np.ndarray, label: str) -> np.ndarray:
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
    # The inference helper expects a 3-D stack.  Here each z plane is represented
    # by its fixed-y line profile, with a singleton third dimension.
    xz = np.asarray(mapped.xz_intensity, float)
    if xz.shape[0] == len(z_m):
        return xz[:, :, None]
    if xz.shape[1] == len(z_m):
        return xz.T[:, :, None]
    raise RuntimeError(f"unexpected XZ shape {xz.shape} for {len(z_m)} z planes")


def build_parameter_recovery(out: Path, *, grid_n: int = 384) -> tuple[Path, list[dict]]:
    z = np.linspace(20e-3, 100e-3, 17)
    coord = np.linspace(-0.8e-3, 0.8e-3, 241)
    manifest = canonical_hardware_manifest()
    iris_radius = float(hardware_value(manifest, "fourier_iris_radius_m"))

    tests = [
        {
            "name": "axicon decentre x",
            "units": "µm",
            "truth": 300.0,
            "values": np.arange(-500.0, 500.1, 100.0),
            "make": lambda v: SystemErrorConfig(axicon=AxiconError(decentre_m=(float(v) * 1e-6, 0.0))),
            "color": CYAN,
        },
        {
            "name": "input pointing x",
            "units": "mrad",
            "truth": 0.6,
            "values": np.arange(-1.0, 1.0001, 0.2),
            "make": lambda v: SystemErrorConfig(beam=GaussianBeamError(pointing_rad=(float(v) * 1e-3, 0.0))),
            "color": GOLD,
        },
        {
            "name": "SLM1 registration x",
            "units": "µm",
            "truth": 200.0,
            "values": np.arange(-400.0, 400.1, 100.0),
            "make": lambda v: SystemErrorConfig(slm1=SLMError(pattern_offset_m=(float(v) * 1e-6, 0.0))),
            "color": MAGENTA,
        },
        {
            "name": "4F iris offset x",
            "units": "iris radii",
            "truth": 0.3,
            "values": np.arange(-0.5, 0.5001, 0.1),
            "make": lambda v: SystemErrorConfig(fourf=FourFError(iris_offset_m=(float(v) * iris_radius, 0.0))),
            "color": GREEN,
        },
    ]

    results = []
    fig = plt.figure(figsize=(13.8, 8.2), facecolor=DARK)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.82], hspace=0.34, wspace=0.28)

    for i, test in enumerate(tests):
        target = _xz_stack(test["make"](test["truth"]), grid_n=grid_n, z_m=z, coord_m=coord, label=f"poster-param-truth-{i}")
        fit = grid_search_parameter(
            parameter=test["name"],
            units=test["units"],
            candidate_values=test["values"],
            target_stack=target,
            simulate=lambda value, t=test, ii=i: _xz_stack(t["make"](value), grid_n=grid_n, z_m=z, coord_m=coord, label=f"poster-param-candidate-{ii}-{value:g}"),
        )
        row = fit.as_dict()
        row["truth_value"] = float(test["truth"])
        row["absolute_error"] = float(abs(fit.best_value - test["truth"]))
        row["scope"] = "synthetic one-parameter-at-a-time model-to-model recovery"
        results.append(row)

        ax = fig.add_subplot(gs[i // 2, i % 2])
        _style(ax)
        ax.plot(fit.candidate_values, fit.costs, marker="o", lw=1.7, color=test["color"])
        ax.axvline(test["truth"], color=FG, ls="--", lw=1.2, label="injected")
        ax.axvline(fit.best_value, color=RED, ls=":", lw=1.5, label="recovered")
        ax.set_title(test["name"], fontsize=11.5, weight="bold")
        ax.set_xlabel(f"candidate ({test['units']})")
        ax.set_ylabel("17-plane morphology RMSE")
        ax.grid(alpha=0.12)
        ax.text(0.04, 0.93, f"injected  {test['truth']:+g} {test['units']}\nrecovered {fit.best_value:+g} {test['units']}", transform=ax.transAxes, va="top", color=FG, fontsize=9.2,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=DARK, edgecolor="#42515d", alpha=0.90))
        if i == 0:
            ax.legend(frameon=False, labelcolor=FG, fontsize=8)

    ax_text = fig.add_subplot(gs[:, 2])
    ax_text.set_facecolor(DARK)
    ax_text.axis("off")
    ax_text.text(0.02, 0.96, "PHYSICAL DIAGNOSIS LAYER", color=CYAN, fontsize=12, weight="bold", va="top")
    ax_text.text(0.02, 0.86, "The same forward model can return\ninterpretable bench parameters — not\nonly a phase correction.", color=FG, fontsize=13, weight="bold", linespacing=1.35, va="top")
    ax_text.text(0.02, 0.67,
                 "For each synthetic benchmark, one\nknown error is hidden in a 17-plane V1\nintensity stack. The candidate physical\nparameter is replayed through the full\nSLM → 4F → axicon route and fitted from\nits longitudinal morphology.",
                 color=MUTED, fontsize=10.5, linespacing=1.45, va="top")
    ax_text.text(0.02, 0.37,
                 "Experimental extension:\nfit a bounded parameter vector such as\nbeam pointing, SLM registration, iris\noffset, axicon decentre/tilt and selected\nlow-order phase terms; report uncertainty\nand degeneracy before applying correction.",
                 color=FG, fontsize=10.5, linespacing=1.45, va="top")
    ax_text.text(0.02, 0.12, "Important: these four panels are\nmodel-to-model validation, not measured\nbench estimates.", color=GOLD, fontsize=9.5, weight="bold", linespacing=1.35, va="top")

    fig.suptitle("Can the simulated system identify the physical error that generated the beam?", color=FG, fontsize=18, weight="bold", y=0.985)
    fig.text(0.5, 0.94, "One-parameter synthetic recovery establishes the diagnostic concept before multi-parameter fitting of real z-stack measurements.", ha="center", color=MUTED, fontsize=9.7)

    path = _save(fig, out / "02_physical_parameter_recovery.png")
    with (out / "02_physical_parameter_recovery.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with (out / "02_physical_parameter_recovery.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["parameter", "units", "truth_value", "best_value", "absolute_error", "best_cost", "second_best_cost", "relative_cost_margin", "scope"])
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in writer.fieldnames})
    return path, results


def copy_experimental_phase_outputs(out: Path) -> list[Path]:
    files = [
        (ROOT / "figures/experimental/q20_aberration/reconstruction/annular_aberration_phase.png", out / "03_experimental_phase_retrieval.png"),
        (ROOT / "figures/experimental/q20_aberration/single_mask/single_z_double_confirmation_minus10.png", out / "04_experimental_inverse_confirmation.png"),
        (ROOT / "figures/experimental/q20_aberration/phase_error_recreation/phase_error_recreation_signed_xz_yz.png", out / "05_experimental_phase_recreation_xz_yz.png"),
    ]
    copied = []
    for src, dst in files:
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def build_contact_sheet(out: Path, paths: list[Path]) -> Path:
    fig = plt.figure(figsize=(17, 13), facecolor="#171b1f")
    gs = fig.add_gridspec(2, 3, hspace=0.20, wspace=0.10)
    for i, p in enumerate(paths[:6]):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ax.imshow(plt.imread(p))
        ax.set_title(p.stem.replace("_", " "), color="white", fontsize=11.5, weight="bold")
        ax.axis("off")
    fig.suptitle("Poster core evidence v3 — focused on system modelling, diagnosis and correction", color="white", fontsize=19, weight="bold", y=0.992)
    return _save(fig, out / "00_core_evidence_contact_sheet.png", dpi=180)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    self_path, self_meta = build_self_healing_path(OUT)
    recovery_path, recovery_meta = build_parameter_recovery(OUT)
    experimental = copy_experimental_phase_outputs(OUT)
    ordered = [self_path, recovery_path, *experimental]
    sheet = build_contact_sheet(OUT, ordered)
    manifest = {
        "story": "simulate physical system -> nominal constraints -> controlled errors -> infer physical metrics -> retrieve residual phase -> correction",
        "excluded": ["through-sample/interface figure"],
        "self_healing": self_meta,
        "synthetic_parameter_recovery": recovery_meta,
        "experimental_phase_outputs_copied_without_data_modification": [str(p.relative_to(ROOT)) for p in experimental],
        "figures": [str(p.relative_to(ROOT)) for p in ordered],
        "contact_sheet": str(sheet.relative_to(ROOT)),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
