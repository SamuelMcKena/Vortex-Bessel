"""Curate simulated system errors and connect them to the correction workflow.

This is a figure/benchmark pass, not a poster compositor.  It does two things:

1. render a compact set of physically distinct perturbations from across the
   actual source-scale route; and
2. use those same registered perturbation families as candidate models in a
   multi-plane physical fit before handing the remaining problem to the existing
   q=20 residual-phase retrieval.

No intensity residual is interpreted as phase.  The q=20 Miao-style retrieval
remains the authoritative residual phase / SLM2 correction path.
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

import build_poster_core_evidence_v5 as v5
from vbb_study.digital_twin.physical_error_dictionary import (
    combine_error_configs,
    correction_handoff_manifest,
    greedy_fit_error_dictionary,
)
from vbb_study.digital_twin.physical_error_inference import morphology_rmse, plane_normalise_stack
from vbb_study.digital_twin.vortex_system_error_sweeps import system_sweep_registry
from vbb_study.digital_twin.vortex_system_route import SystemErrorConfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "poster" / "error_dictionary_v1"
EPS = np.finfo(float).tiny

BG = "#070a0d"
AXBG = "#0b0f14"
FG = "#f2f3f4"
MUTED = "#aab4be"
CYAN = "#4dd9d5"
GOLD = "#ffca58"
RED = "#ff665e"
GREEN = "#58d99d"

THERMAL = LinearSegmentedColormap.from_list(
    "poster_thermal",
    [
        (0.00, "#000000"),
        (0.10, "#090000"),
        (0.30, "#4d0000"),
        (0.52, "#b51d00"),
        (0.74, "#f56c00"),
        (0.90, "#ffc21a"),
        (1.00, "#fff5bd"),
    ],
    N=256,
)

Z_SIG = np.linspace(20e-3, 100e-3, 13)
Z_FIT = np.linspace(20e-3, 100e-3, 17)
HALFWIDTH_M = 0.95e-3


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(AXBG)
    for s in ax.spines.values():
        s.set_color("#53606d")
        s.set_linewidth(0.7)
    ax.tick_params(colors=MUTED, labelsize=7.8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(FG)
    ax.grid(False)


def _save(fig: plt.Figure, path: Path, dpi: int = 280) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _add_noise(stack: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    arr = np.maximum(np.asarray(stack, float), 0.0).copy()
    rng = np.random.default_rng(seed)
    for iz in range(arr.shape[0]):
        sigma = float(fraction) * max(float(np.max(arr[iz])), EPS)
        arr[iz] = np.maximum(arr[iz] + rng.normal(0.0, sigma, arr[iz].shape), 0.0)
    return arr


def _power_trace_shape_rmse(candidate: np.ndarray, target: np.ndarray) -> float:
    c = np.maximum(np.asarray(candidate, float), 0.0)
    t = np.maximum(np.asarray(target, float), 0.0)
    pc = np.sum(c, axis=(1, 2)); pt = np.sum(t, axis=(1, 2))
    pc /= max(float(np.mean(pc)), EPS)
    pt /= max(float(np.mean(pt)), EPS)
    return float(np.sqrt(np.mean((pc - pt) ** 2)))


def fit_loss(candidate: np.ndarray, target: np.ndarray) -> float:
    # Relative z-throughput shape is global-scale invariant, so the synthetic
    # benchmark does not depend on an arbitrary absolute camera gain.
    return 0.85 * morphology_rmse(candidate, target) + 0.15 * _power_trace_shape_rmse(candidate, target)


def _xz(stack: np.ndarray, coords: np.ndarray) -> np.ndarray:
    iy0 = int(np.argmin(np.abs(coords)))
    return np.asarray(stack)[:, iy0, :].T


def _extent(z: np.ndarray, coords: np.ndarray) -> list[float]:
    return [float(z[0]*1e3), float(z[-1]*1e3), float(coords[0]*1e3), float(coords[-1]*1e3)]


def _format_value(family: str, value: float) -> str:
    if family in {"beam_lateral_decentre_x", "slm1_hologram_offset_x", "axicon_lateral_decentre_x"}:
        return f"{value*1e6:+.0f} µm"
    if family == "fourf_iris_offset_x":
        return f"{value*1e3:+.2f} mm"
    if family == "fourf_lens1_despace":
        return f"{value*1e3:+.0f} mm"
    if family == "axicon_rigid_tilt_x":
        return f"{math.degrees(value):+.2f}°"
    if family in {"axicon_round_tip", "axicon_flat_tip"}:
        return f"{value*1e6:.0f} µm"
    if family == "beam_radius_scale":
        return f"{value:.2f} × nominal"
    if family == "fourf_iris_radius_scale":
        return f"{value:.2f} × nominal"
    return f"{value:g}"


SIGNATURE_CASES = [
    ("beam_lateral_decentre_x", 500e-6, "Input beam decentre"),
    ("beam_radius_scale", 0.70, "Input beam radius"),
    ("slm1_hologram_offset_x", 200e-6, "SLM1 hologram offset"),
    ("fourf_iris_offset_x", 0.30e-3, "4F iris offset"),
    ("fourf_lens1_despace", 10e-3, "4F lens 1 despace"),
    ("axicon_lateral_decentre_x", 500e-6, "Axicon decentre"),
    ("axicon_rigid_tilt_x", math.radians(0.50), "Axicon tilt"),
    ("axicon_round_tip", 20e-6, "Rounded axicon tip"),
]

FIT_FAMILIES = [
    "beam_lateral_decentre_x",
    "beam_radius_scale",
    "slm1_hologram_offset_x",
    "fourf_iris_offset_x",
    "fourf_lens1_despace",
    "axicon_lateral_decentre_x",
    "axicon_round_tip",
]


def build_signature_figures(out: Path, *, grid_n: int = 320) -> tuple[list[Path], list[dict]]:
    reg = system_sweep_registry()
    nominal, coords = v5._xy_stack(
        SystemErrorConfig(), grid_n=grid_n, z_m=Z_SIG,
        halfwidth_m=HALFWIDTH_M, label="error-dictionary-nominal",
    )
    nominal_n = plane_normalise_stack(nominal)
    nominal_xy = nominal_n[len(Z_SIG)//2]
    nominal_xz = _xz(nominal_n, coords)
    ext_xy = [coords[0]*1e3, coords[-1]*1e3, coords[0]*1e3, coords[-1]*1e3]
    ext_xz = _extent(Z_SIG, coords)

    paths: list[Path] = []
    records: list[dict] = []
    for family, value, title in SIGNATURE_CASES:
        config = reg[family]["builder"](float(value))
        stack, c = v5._xy_stack(
            config, grid_n=grid_n, z_m=Z_SIG,
            halfwidth_m=HALFWIDTH_M, label=f"error-dictionary-{family}",
        )
        if not np.allclose(c, coords):
            raise RuntimeError("signature cases do not share fixed camera coordinates")
        stack_n = plane_normalise_stack(stack)
        err_xy = stack_n[len(Z_SIG)//2]
        err_xz = _xz(stack_n, coords)

        fig = plt.figure(figsize=(9.5, 7.0), facecolor=BG)
        gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.18,
                              left=0.08, right=0.97, top=0.84, bottom=0.09)
        panels = [
            (nominal_xy, "Nominal", ext_xy, "xy"),
            (err_xy, _format_value(family, value), ext_xy, "xy"),
            (nominal_xz, "Nominal propagation", ext_xz, "xz"),
            (err_xz, "Perturbed propagation", ext_xz, "xz"),
        ]
        for i, (arr, ptitle, ext, kind) in enumerate(panels):
            ax = fig.add_subplot(gs[i//2, i%2]); _style(ax)
            ax.imshow(np.maximum(arr,0)**0.48, origin="lower", aspect="equal" if kind=="xy" else "auto",
                      extent=ext, cmap=THERMAL, vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(ptitle, fontsize=10.8, weight="bold")
            if kind == "xy":
                ax.set_xlabel("x (mm)")
                if i % 2 == 0: ax.set_ylabel("y (mm)")
                else: ax.tick_params(labelleft=False)
            else:
                ax.set_xlabel("z from axicon (mm)")
                if i % 2 == 0: ax.set_ylabel("x at y = 0 (mm)")
                else: ax.tick_params(labelleft=False)
        fig.suptitle(title, color=FG, fontsize=16.5, weight="bold", y=0.965)
        fig.text(0.5, 0.905,
                 "V1 forward model; identical fixed laboratory coordinates and display normalisation by plane.",
                 color=MUTED, ha="center", fontsize=8.6)
        p = _save(fig, out / f"signature_{family}.png")
        paths.append(p)
        records.append({
            "family": family,
            "display_name": title,
            "value": float(value),
            "display_value": _format_value(family, value),
            "units": str(reg[family].get("units", "")),
            "fidelity": str(reg[family].get("fidelity", "")),
        })
    return paths, records


def build_signature_contact_sheet(out: Path, paths: list[Path], records: list[dict]) -> Path:
    fig = plt.figure(figsize=(17.6, 15.2), facecolor="#15191d")
    gs = fig.add_gridspec(4, 2, hspace=0.11, wspace=0.055,
                          left=0.025, right=0.975, top=0.94, bottom=0.025)
    for i, (path, rec) in enumerate(zip(paths, records)):
        ax = fig.add_subplot(gs[i//2, i%2])
        ax.set_facecolor("#15191d")
        ax.imshow(plt.imread(path))
        ax.set_title(f"{rec['display_name']}  ·  {rec['display_value']}",
                     color="white", fontsize=12.2, weight="bold", pad=5)
        ax.axis("off")
    fig.suptitle("Simulated perturbations across the optical system",
                 color="white", fontsize=20, weight="bold", y=0.985)
    return _save(fig, out / "00_error_signature_contact_sheet.png", dpi=190)


def build_dictionary_fit(out: Path, *, grid_n: int = 224) -> tuple[Path, dict, Path]:
    reg = system_sweep_registry()

    # Deliberately use two different physical planes.  Both truth values are in
    # the declared screening registry so the first benchmark tests family
    # selection/accumulation rather than interpolation quality.
    truth = combine_error_configs(
        reg["axicon_lateral_decentre_x"]["builder"](250e-6),
        reg["beam_radius_scale"]["builder"](0.85),
    )

    cache: dict[SystemErrorConfig, np.ndarray] = {}
    coords_holder: list[np.ndarray] = []
    def simulate(config: SystemErrorConfig) -> np.ndarray:
        if config not in cache:
            stack, coords = v5._xy_stack(
                config, grid_n=grid_n, z_m=Z_FIT,
                halfwidth_m=HALFWIDTH_M, label=f"dictionary-fit-{len(cache):03d}",
            )
            cache[config] = stack
            if not coords_holder:
                coords_holder.append(coords)
        return cache[config]

    truth_clean = simulate(truth)
    target = _add_noise(truth_clean, 0.005, seed=20260831)
    fit = greedy_fit_error_dictionary(
        families=FIT_FAMILIES,
        target_stack=target,
        simulate_config=simulate,
        max_stages=2,
        minimum_improvement_fraction=0.01,
        loss_fn=fit_loss,
    )
    best = simulate(fit.fitted_config)
    coords = coords_holder[0]

    accepted = [s.accepted_family for s in fit.stages if s.accepted_family]
    expected = {"axicon_lateral_decentre_x", "beam_radius_scale"}
    recovery_pass = set(accepted) == expected

    tn = plane_normalise_stack(target); bn = plane_normalise_stack(best)
    txz = _xz(tn, coords); bxz = _xz(bn, coords)
    rxz = np.abs(txz - bxz)
    ext = _extent(Z_FIT, coords)

    fig = plt.figure(figsize=(15.4, 7.9), facecolor=BG)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.80],
                          hspace=0.33, wspace=0.30,
                          left=0.055, right=0.985, top=0.86, bottom=0.10)
    for col, (arr, title) in enumerate([
        (txz, "Synthetic z-stack"),
        (bxz, "Fitted physical model"),
        (rxz / max(float(np.max(rxz)), EPS), "Residual intensity"),
    ]):
        ax = fig.add_subplot(gs[0, col]); _style(ax)
        ax.imshow(np.maximum(arr,0)**0.48, origin="lower", aspect="auto", extent=ext,
                  cmap=THERMAL if col < 2 else "magma", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_title(title, fontsize=11.3, weight="bold")
        ax.set_xlabel("z from axicon (mm)")
        if col == 0: ax.set_ylabel("x at y = 0 (mm)")
        else: ax.tick_params(labelleft=False)

    ax_text = fig.add_subplot(gs[0, 3]); ax_text.set_facecolor(BG); ax_text.axis("off")
    ax_text.text(0.02, 0.95, "Synthetic test", color=FG, fontsize=12.4, weight="bold", va="top")
    ax_text.text(0.02, 0.82,
                 "Hidden model:\n  axicon decentre  +250 µm\n  beam radius       0.85 × nominal",
                 color=MUTED, fontsize=10.0, va="top", linespacing=1.45)
    y = 0.55
    for stage in fit.stages:
        if stage.accepted_family is None:
            continue
        label = next((t for f, _, t in SIGNATURE_CASES if f == stage.accepted_family), stage.accepted_family)
        ax_text.text(0.02, y, f"Fit {stage.stage}: {label}", color=CYAN, fontsize=9.6, weight="bold")
        ax_text.text(0.02, y-0.075, _format_value(stage.accepted_family, float(stage.accepted_value)),
                     color=FG, fontsize=12.2, weight="bold")
        y -= 0.20
    ax_text.text(0.02, 0.08,
                 f"fit loss: {fit.initial_cost:.4f} → {fit.final_cost:.4f}",
                 color=GREEN if recovery_pass else RED, fontsize=9.4)

    # Candidate family ranking from each accepted stage: plain scientific labels,
    # no report/withhold/gate language.
    for stage_col, stage in enumerate(fit.stages[:2]):
        ax = fig.add_subplot(gs[1, stage_col*2:(stage_col+1)*2]); _style(ax)
        rankings = list(stage.rankings)
        names = []
        vals = []
        for r in rankings:
            names.append(next((t for f, _, t in SIGNATURE_CASES if f == r.family), r.family.replace("_", " ")))
            vals.append(float(r.best_cost))
        order = np.argsort(vals)[::-1]
        names = [names[i] for i in order]
        vals = [vals[i] for i in order]
        ypos = np.arange(len(names))
        ax.barh(ypos, vals)
        ax.set_yticks(ypos, labels=names, fontsize=7.8)
        ax.set_xlabel("best multi-plane fit loss")
        ax.set_title(f"Candidate models after fit {stage.stage-1}", fontsize=10.8, weight="bold")

    fig.suptitle("Fitting physical system errors to a multi-plane intensity stack",
                 color=FG, fontsize=18.0, weight="bold", y=0.975)
    fig.text(0.5, 0.917,
             "The same forward models used for the perturbation study are searched directly against the z-stack; the remaining discrepancy is passed to the separate residual-phase retrieval.",
             ha="center", color=MUTED, fontsize=9.2)
    png = _save(fig, out / "09_physical_error_dictionary_fit.png", dpi=310)

    handoff = correction_handoff_manifest(fit)
    handoff.update({
        "benchmark": {
            "type": "synthetic two-error dictionary fit",
            "case": "V1",
            "z_planes": int(len(Z_FIT)),
            "noise_fraction_of_plane_peak": 0.005,
            "truth": {
                "axicon_lateral_decentre_x_m": 250e-6,
                "beam_radius_scale": 0.85,
            },
            "candidate_families": FIT_FAMILIES,
            "accepted_families": accepted,
            "expected_families": sorted(expected),
            "recovery_pass": bool(recovery_pass),
            "fit": fit.as_dict(),
        }
    })
    handoff_path = out / "10_correction_handoff_manifest.json"
    handoff_path.write_text(json.dumps(handoff, indent=2) + "\n", encoding="utf-8")

    if not recovery_pass:
        raise RuntimeError(
            "synthetic physical-error dictionary benchmark did not recover the two injected families; "
            f"accepted={accepted}, expected={sorted(expected)}"
        )
    return png, handoff, handoff_path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sig_paths, records = build_signature_figures(OUT)
    sheet = build_signature_contact_sheet(OUT, sig_paths, records)
    fit_png, handoff, handoff_path = build_dictionary_fit(OUT)
    manifest = {
        "outcome": "POSTER-ERROR-DICTIONARY-V1",
        "purpose": "curate simulated error signatures and make them candidate models in the physical fitting stage before q20 residual-phase correction",
        "poster_rebuild_allowed": False,
        "signature_contact_sheet": str(sheet),
        "signature_figures": [str(p) for p in sig_paths],
        "dictionary_fit_figure": str(fit_png),
        "correction_handoff_manifest": str(handoff_path),
        "error_cases": records,
        "q20_handoff": handoff["residual_phase_stage"],
        "figure_policy": "individual plots are curation candidates; no error family is included on the poster solely because it exists in the registry",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
