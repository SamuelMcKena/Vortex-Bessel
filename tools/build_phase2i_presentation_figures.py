"""Build slide-ready figures from the current Phase 2I optical model.

This renderer is intentionally separate from the historical conference figure
branch.  It uses the current integrated dual-SLM -> explicit 4F -> physical
axicon route and the fixed-support/fixed-laboratory longitudinal propagator.

The figures are presentation assets, but the optical data are regenerated from
current repository code.  No synthetic 'experimental' images or AI-generated
bench imagery are inserted.

Outputs
-------
01_computational_route.png
02_beam_profile_shaping_B0_V1_V3.png
03_realistic_error_scope.png
04_V1_axicon_decentre_fixed_lab.png
05_V1_nonideal_tip_fixed_lab.png
06_simulation_experiment_closure.png
presentation_figure_manifest.json

Conventions
-----------
* x-z maps use fixed laboratory coordinates; no per-z recentring/tracking.
* one frozen angular-spectrum support is used for each complete z sweep.
* comparative perturbation panels use the nominal-case peak as the common
  normalisation, so power redistribution is visible rather than hidden.
* ideal B0/V1/V3 morphology panels are normalised per beam family because the
  purpose is beam-profile comparison, not absolute radiometry.
* the experiment panel is a measurement *input* schematic until real Phase 2I
  camera data are supplied; no fake measurement is rendered.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

from vbb_study.digital_twin.phase2a_contracts import (
    canonical_hardware_manifest,
    hardware_value,
)
from vbb_study.digital_twin.vortex_axicon_tip_reference import tip_resolution
from vbb_study.digital_twin.vortex_continuous_propagation import (
    build_fixed_plane_longitudinal_map,
    build_fixed_support_spectrum,
    native_field_at_z,
)
from vbb_study.digital_twin.vortex_system_route import (
    AxiconError,
    SystemErrorConfig,
    build_system_route,
)


EPS = np.finfo(float).tiny
Z_REF_M = 60.0e-3
Z_VALUES_M = np.linspace(5.0e-3, 120.0e-3, 32)
IDEAL_COORD_M = np.linspace(-0.32e-3, 0.32e-3, 321)
DECENTRE_COORD_M = np.linspace(-1.00e-3, 1.00e-3, 401)
TIP_COORD_M = np.linspace(-0.36e-3, 0.36e-3, 321)

FIG_BG = "#080b0f"
AX_BG = "#0c1117"
TEXT = "#f1f3f4"
MUTED = "#aeb8c4"
RED = "#ff3b30"
BORDER = "#334252"
GREEN = "#39d6ad"
CMAP = "turbo"


def _normalise(values: np.ndarray, peak: float | None = None) -> np.ndarray:
    arr = np.maximum(np.asarray(values, dtype=float), 0.0)
    scale = float(np.max(arr)) if peak is None else float(peak)
    return arr / max(scale, EPS)


def _style_figure(fig: plt.Figure) -> None:
    fig.patch.set_facecolor(FIG_BG)


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(AX_BG)
    for spine in ax.spines.values():
        spine.set_color("#697684")
        spine.set_linewidth(0.7)
    ax.tick_params(colors=MUTED, labelsize=8, length=2.5)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=260, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _ell(case_id: str) -> int:
    return {"B0": 0, "V1": 1, "V3": 3}[case_id]


def _propagator(route: Mapping[str, Any], z_max_m: float | None = None):
    return build_fixed_support_spectrum(
        np.asarray(route["post_axicon"], dtype=np.complex128),
        dict(route["grid"]),
        wavelength_m=float(route["metadata"]["wavelength_m"]),
        z_max_m=float(Z_VALUES_M[-1] if z_max_m is None else z_max_m),
        minimum_retained_spectral_power=0.995,
    )


def _xy_at_z(route: Mapping[str, Any], z_m: float = Z_REF_M):
    prop = _propagator(route)
    field = native_field_at_z(prop, float(z_m))
    return np.asarray(field, dtype=np.complex128), prop


def _longitudinal(route: Mapping[str, Any], coordinate_m: np.ndarray, label: str):
    prop = _propagator(route)
    mapped = build_fixed_plane_longitudinal_map(
        prop,
        z_values_m=Z_VALUES_M,
        x_coordinates_m=coordinate_m,
        y_coordinates_m=coordinate_m,
        fixed_x_m=0.0,
        fixed_y_m=0.0,
        source_label=label,
    )
    return mapped, prop


def _fixed_lab_crop(
    intensity: np.ndarray,
    grid: Mapping[str, Any],
    halfwidth_m: float,
) -> tuple[np.ndarray, list[float]]:
    x = np.asarray(grid["x"], dtype=float)
    ids = np.flatnonzero(np.abs(x) <= float(halfwidth_m))
    if ids.size < 24:
        raise RuntimeError("presentation fixed-lab crop is under-sampled")
    crop = np.asarray(intensity)[np.ix_(ids, ids)]
    extent = [x[ids[0]] * 1e3, x[ids[-1]] * 1e3, x[ids[0]] * 1e3, x[ids[-1]] * 1e3]
    return crop, extent


def _centroid(intensity: np.ndarray, grid: Mapping[str, Any]) -> tuple[float, float]:
    values = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    total = float(np.sum(values))
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    return (
        float(np.sum(values * X) / max(total, EPS)),
        float(np.sum(values * Y) / max(total, EPS)),
    )


def _draw_xy(
    ax: plt.Axes,
    values: np.ndarray,
    extent_mm: Sequence[float],
    *,
    title: str,
    peak: float | None = None,
    show_y: bool = False,
) -> None:
    _style_axis(ax)
    ax.imshow(
        _normalise(values, peak),
        origin="lower",
        extent=list(map(float, extent_mm)),
        cmap=CMAP,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title(title, fontsize=12, pad=7)
    ax.set_xlabel("x (mm)", fontsize=8)
    if show_y:
        ax.set_ylabel("y (mm)", fontsize=8)
    else:
        ax.tick_params(labelleft=False)
    ax.axhline(0.0, color="white", alpha=0.18, linewidth=0.5)
    ax.axvline(0.0, color="white", alpha=0.18, linewidth=0.5)


def _draw_xz(
    ax: plt.Axes,
    values: np.ndarray,
    coordinate_m: np.ndarray,
    *,
    peak: float | None = None,
    show_y: bool = False,
) -> None:
    _style_axis(ax)
    ax.imshow(
        _normalise(np.asarray(values, dtype=float).T, peak),
        origin="lower",
        extent=[
            Z_VALUES_M[0] * 1e3,
            Z_VALUES_M[-1] * 1e3,
            float(coordinate_m[0]) * 1e3,
            float(coordinate_m[-1]) * 1e3,
        ],
        cmap=CMAP,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_xlabel("z from axicon (mm)", fontsize=8)
    if show_y:
        ax.set_ylabel("x at fixed y=0 (mm)", fontsize=8)
    else:
        ax.tick_params(labelleft=False)
    ax.axhline(0.0, color="white", alpha=0.18, linewidth=0.5)
    ax.axvline(Z_REF_M * 1e3, color="white", alpha=0.30, linestyle="--", linewidth=0.7)


def build_computational_route(output_dir: Path, grid_n: int) -> Path:
    route = build_system_route("V1", grid_n=int(grid_n))
    grid = dict(route["grid"])
    x = np.asarray(grid["x"], dtype=float)
    ids = np.flatnonzero(np.abs(x) <= 2.4e-3)
    extent = [x[ids[0]] * 1e3, x[ids[-1]] * 1e3, x[ids[0]] * 1e3, x[ids[-1]] * 1e3]

    input_i = np.abs(np.asarray(route["input_beam"])) ** 2
    slm1_phase = np.angle(np.asarray(route["post_slm1"]) * np.conj(np.asarray(route["input_beam"])))
    slm2_phase = np.angle(np.asarray(route["post_slm2"]) * np.conj(np.asarray(route["post_slm1"])))
    fourier_i = np.abs(np.asarray(route["fourier_plane_before_iris"])) ** 2
    axicon_phase = np.angle(np.asarray(route["post_axicon_local"]) * np.conj(np.asarray(route["field_on_axicon_plane"])))
    output_field, _ = _xy_at_z(route)
    output_i = np.abs(output_field) ** 2

    fig, axes = plt.subplots(1, 6, figsize=(15.6, 3.2), constrained_layout=True)
    _style_figure(fig)
    panels = [
        (input_i, "input beam", CMAP, False),
        (slm1_phase, "SLM1: vortex", "twilight", True),
        (slm2_phase, "SLM2: carrier", "twilight", True),
        (fourier_i, "4F: order selection", CMAP, False),
        (axicon_phase, "physical axicon", "twilight", True),
    ]
    for i, (data, title, cmap, phase) in enumerate(panels):
        ax = axes[i]
        _style_axis(ax)
        sub = np.asarray(data)[np.ix_(ids, ids)]
        if phase:
            ax.imshow(sub, origin="lower", extent=extent, cmap=cmap, vmin=-np.pi, vmax=np.pi, interpolation="nearest")
        else:
            ax.imshow(_normalise(sub), origin="lower", extent=extent, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(title, fontsize=10, pad=6)
        ax.set_xlabel("x (mm)", fontsize=7)
        if i == 0:
            ax.set_ylabel("y (mm)", fontsize=7)
        else:
            ax.tick_params(labelleft=False)

    crop, out_extent = _fixed_lab_crop(output_i, grid, 0.32e-3)
    _draw_xy(axes[5], crop, out_extent, title=f"field at z={Z_REF_M*1e3:.0f} mm", show_y=False)
    fig.suptitle("Current numerical route: dual SLM → 4F → physical axicon → propagated field", color=TEXT, fontsize=15, y=1.04)
    path = output_dir / "01_computational_route.png"
    _save(fig, path)
    return path


def build_ideal_family(output_dir: Path, grid_n: int) -> tuple[Path, list[dict[str, Any]]]:
    cases = (("B0", "B0 — bright-core Bessel"), ("V1", "V1 — vortex-Bessel ℓ=1"), ("V3", "V3 — vortex-Bessel ℓ=3"))
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.4), constrained_layout=True)
    _style_figure(fig)
    metrics: list[dict[str, Any]] = []

    for col, (case_id, label) in enumerate(cases):
        route = build_system_route(case_id, grid_n=int(grid_n))
        field, prop_xy = _xy_at_z(route)
        intensity = np.abs(field) ** 2
        crop, extent = _fixed_lab_crop(intensity, route["grid"], 0.32e-3)
        _draw_xy(axes[0, col], crop, extent, title=label, show_y=(col == 0))

        mapped, prop = _longitudinal(route, IDEAL_COORD_M, f"presentation-ideal-{case_id}")
        _draw_xz(axes[1, col], mapped.xz_intensity, IDEAL_COORD_M, show_y=(col == 0))
        cx, cy = _centroid(intensity, route["grid"])
        metrics.append({
            "case_id": case_id,
            "vortex_charge": _ell(case_id),
            "z_ref_m": Z_REF_M,
            "xy_centroid_x_m": cx,
            "xy_centroid_y_m": cy,
            "fixed_support_retained_power_fraction": float(prop.retained_spectral_power_fraction),
            "normalisation": "per_case_peak_for_morphology_only",
            "fixed_lab_longitudinal": True,
        })

    fig.suptitle("Beam profile shaping — ideal simulated outputs", color=TEXT, fontsize=16, y=1.03)
    path = output_dir / "02_beam_profile_shaping_B0_V1_V3.png"
    _save(fig, path)
    return path, metrics


def build_error_scope(output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14.2, 4.0))
    _style_figure(fig)
    ax.set_facecolor(FIG_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    blocks = [
        (0.025, 0.17, "ideal model", "dual SLM\n4F + axicon"),
        (0.245, 0.16, "alignment", "beam / axicon\ndecentre & tilt"),
        (0.445, 0.16, "SLM / 4F", "registration\nphase & iris errors"),
        (0.645, 0.16, "axicon", "tip geometry\nsurface / angle"),
        (0.845, 0.13, "observable", "x–y + x–z\nsignature"),
    ]
    for idx, (x0, width, title, body) in enumerate(blocks):
        ax.add_patch(FancyBboxPatch((x0, 0.33), width, 0.42, boxstyle="round,pad=0.012,rounding_size=0.018", linewidth=1.2, edgecolor=RED if idx in {1,2,3} else BORDER, facecolor=AX_BG))
        ax.text(x0 + width/2, 0.61, title, color=TEXT, ha="center", va="center", fontsize=13, weight="bold")
        ax.text(x0 + width/2, 0.46, body, color=MUTED, ha="center", va="center", fontsize=10, linespacing=1.4)
    for left, right in zip(blocks[:-1], blocks[1:]):
        x1 = left[0] + left[1]
        x2 = right[0]
        ax.add_patch(FancyArrowPatch((x1+0.008,0.54),(x2-0.008,0.54),arrowstyle="-|>",mutation_scale=15,linewidth=1.4,color=RED))
    ax.text(0.5, 0.17, "change a physical parameter → predict the signature that should be visible on the bench", color=TEXT, ha="center", fontsize=13)
    ax.text(0.5, 0.08, "examples next: V1 axicon lateral decentre and non-ideal tip geometry", color=MUTED, ha="center", fontsize=10)
    path = output_dir / "03_realistic_error_scope.png"
    _save(fig, path)
    return path


def build_v1_decentre(output_dir: Path, grid_n: int) -> tuple[Path, list[dict[str, Any]]]:
    values = (-500e-6, 0.0, +500e-6)
    labels = ("−500 µm", "aligned", "+500 µm")
    cases: list[dict[str, Any]] = []

    for value, label in zip(values, labels):
        route = build_system_route("V1", grid_n=int(grid_n), config=SystemErrorConfig(axicon=AxiconError(decentre_m=(float(value), 0.0))))
        field, prop_xy = _xy_at_z(route)
        intensity = np.abs(field) ** 2
        mapped, prop = _longitudinal(route, DECENTRE_COORD_M, f"presentation-V1-axicon-decentre-{value:g}")
        cx, cy = _centroid(intensity, route["grid"])
        line = np.maximum(np.asarray(mapped.xz_intensity, dtype=float), 0.0)
        weights = np.sum(line, axis=1)
        xcent = np.sum(line * DECENTRE_COORD_M[None, :], axis=1) / np.maximum(weights, EPS)
        slope = float(np.polyfit(Z_VALUES_M, xcent, 1)[0])
        cases.append({
            "decentre_m": float(value), "label": label, "route": route, "intensity": intensity,
            "mapped": mapped, "centroid_x_m": cx, "centroid_y_m": cy,
            "fixed_lab_line_centroid_slope_rad": slope,
            "fixed_support_retained_power_fraction": float(prop.retained_spectral_power_fraction),
        })

    nominal_xy_peak = float(np.max(cases[1]["intensity"]))
    nominal_xz_peak = float(np.max(cases[1]["mapped"].xz_intensity))
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.4), constrained_layout=True)
    _style_figure(fig)
    metrics: list[dict[str, Any]] = []
    for col, item in enumerate(cases):
        crop, extent = _fixed_lab_crop(item["intensity"], item["route"]["grid"], 1.0e-3)
        _draw_xy(axes[0, col], crop, extent, title=item["label"], peak=nominal_xy_peak, show_y=(col==0))
        _draw_xz(axes[1, col], item["mapped"].xz_intensity, DECENTRE_COORD_M, peak=nominal_xz_peak, show_y=(col==0))
        metrics.append({k: v for k, v in item.items() if k not in {"route", "intensity", "mapped"}})
    fig.suptitle("V1 axicon lateral decentre — fixed laboratory coordinates", color=TEXT, fontsize=16, y=1.03)
    path = output_dir / "04_V1_axicon_decentre_fixed_lab.png"
    _save(fig, path)
    return path, metrics


def _rounded_tip_error(radius_m: float, base_angle_rad: float) -> AxiconError:
    if radius_m == 0.0:
        return AxiconError(tip_model="sharp")
    # The integrated system route stores the hyperboloidal vertical parameter a.
    # The presentation label is the corresponding radial rounding scale r_h.
    return AxiconError(
        tip_model="hyperboloidal_round",
        rounding_parameter_m=float(radius_m) * math.tan(float(base_angle_rad)),
    )


def build_v1_tip(output_dir: Path, grid_n: int) -> tuple[Path, list[dict[str, Any]]]:
    manifest = canonical_hardware_manifest()
    gamma = math.radians(float(hardware_value(manifest, "axicon_base_angle_deg")))
    radii = (0.0, 200e-6, 800e-6)
    labels = ("ideal sharp tip", "200 µm radial rounding", "800 µm radial rounding")
    cases: list[dict[str, Any]] = []

    for radius, label in zip(radii, labels):
        route = build_system_route("V1", grid_n=int(grid_n), config=SystemErrorConfig(axicon=_rounded_tip_error(float(radius), gamma)))
        resolution = tip_resolution(float(radius), float(route["grid"]["dx"]), minimum_pixels=12.0)
        if radius != 0.0 and not resolution.resolved:
            raise RuntimeError(f"tip case {radius*1e6:.1f} µm is only {resolution.radius_pixels:.2f} pixels on N={grid_n}")
        field, prop_xy = _xy_at_z(route)
        intensity = np.abs(field) ** 2
        mapped, prop = _longitudinal(route, TIP_COORD_M, f"presentation-V1-tip-{radius:g}")
        cases.append({
            "radial_rounding_scale_m": float(radius), "label": label, "route": route,
            "intensity": intensity, "mapped": mapped,
            "tip_radius_pixels": float(resolution.radius_pixels),
            "passes_12_pixel_resolution_gate": bool(resolution.resolved),
            "fixed_support_retained_power_fraction": float(prop.retained_spectral_power_fraction),
        })

    nominal_xy_peak = float(np.max(cases[0]["intensity"]))
    nominal_xz_peak = float(np.max(cases[0]["mapped"].xz_intensity))
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.4), constrained_layout=True)
    _style_figure(fig)
    metrics: list[dict[str, Any]] = []
    for col, item in enumerate(cases):
        crop, extent = _fixed_lab_crop(item["intensity"], item["route"]["grid"], 0.36e-3)
        _draw_xy(axes[0, col], crop, extent, title=item["label"], peak=nominal_xy_peak, show_y=(col==0))
        _draw_xz(axes[1, col], item["mapped"].xz_intensity, TIP_COORD_M, peak=nominal_xz_peak, show_y=(col==0))
        metrics.append({k: v for k, v in item.items() if k not in {"route", "intensity", "mapped"}})
    fig.suptitle("V1 non-ideal axicon tip — common nominal normalisation", color=TEXT, fontsize=16, y=1.03)
    path = output_dir / "05_V1_nonideal_tip_fixed_lab.png"
    _save(fig, path)
    return path, metrics


def build_closure(output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13.5, 4.3))
    _style_figure(fig)
    ax.set_facecolor(FIG_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.035, 0.28, 0.20, 0.46, "simulate", "predict x–y / x–z\nand error signatures", RED),
        (0.285, 0.28, 0.20, 0.46, "measure", "camera z-stack\n+ background / scale", GREEN),
        (0.535, 0.28, 0.20, 0.46, "compare / infer", "residuals + metrics\nfit physical parameters", RED),
        (0.785, 0.28, 0.18, 0.46, "update", "calibration / additive\nSLM correction", GREEN),
    ]
    for x0, y0, w, h, title, body, colour in boxes:
        ax.add_patch(FancyBboxPatch((x0,y0),w,h,boxstyle="round,pad=0.014,rounding_size=0.018",linewidth=1.3,edgecolor=colour,facecolor=AX_BG))
        ax.text(x0+w/2,y0+0.31,title,color=TEXT,ha="center",fontsize=15,weight="bold")
        ax.text(x0+w/2,y0+0.16,body,color=MUTED,ha="center",fontsize=10,linespacing=1.4)
    for a,b in zip(boxes[:-1], boxes[1:]):
        ax.add_patch(FancyArrowPatch((a[0]+a[2]+0.008,0.51),(b[0]-0.008,0.51),arrowstyle="-|>",mutation_scale=16,linewidth=1.5,color=RED))
    ax.add_patch(FancyArrowPatch((0.87,0.24),(0.13,0.24),connectionstyle="arc3,rad=-0.22",arrowstyle="-|>",mutation_scale=15,linewidth=1.3,color=MUTED))
    ax.text(0.50,0.06,"repeat until the model and measured beam agree within calibrated uncertainty",color=TEXT,ha="center",fontsize=12)
    ax.text(0.50,0.88,"Phase 2I experimental closure",color=TEXT,ha="center",fontsize=17,weight="bold")
    ax.text(0.50,0.81,"real measurements enter here; no synthetic measurement is substituted",color=MUTED,ha="center",fontsize=10)
    path = output_dir / "06_simulation_experiment_closure.png"
    _save(fig, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build current Phase 2I presentation figures from repository physics.")
    parser.add_argument("--grid-n", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/presentation_phase2i"))
    args = parser.parse_args()

    if int(args.grid_n) < 1024:
        # The 200 µm rounded-tip panel must pass the 12-pixel local-feature gate
        # on the canonical 10 mm route window.  N=1024 is the safe slide-render default.
        raise ValueError("presentation renderer requires --grid-n >= 1024 so the 200 µm tip case is resolved")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    generated.append(build_computational_route(out, int(args.grid_n)))
    ideal_path, ideal_metrics = build_ideal_family(out, int(args.grid_n)); generated.append(ideal_path)
    generated.append(build_error_scope(out))
    dec_path, dec_metrics = build_v1_decentre(out, int(args.grid_n)); generated.append(dec_path)
    tip_path, tip_metrics = build_v1_tip(out, int(args.grid_n)); generated.append(tip_path)
    generated.append(build_closure(out))

    payload = {
        "outcome": "PHASE2I-PRESENTATION-FIGURES",
        "source_branch_contract": "phase2i-experimental-bench-closure",
        "grid_n": int(args.grid_n),
        "z_ref_m": Z_REF_M,
        "z_values_m": Z_VALUES_M.tolist(),
        "figures": [str(path) for path in generated],
        "ideal_family_metrics": ideal_metrics,
        "V1_axicon_decentre_metrics": dec_metrics,
        "V1_tip_metrics": tip_metrics,
        "evidence_conventions": {
            "longitudinal_coordinates": "fixed laboratory x-z; y=0; no per-z recentring",
            "propagator": "fixed-support angular spectrum over complete z sweep",
            "perturbation_normalisation": "common nominal peak",
            "ideal_family_normalisation": "per-case morphology peak",
            "experimental_images": "none fabricated; measurement shown only as an input stage",
        },
        "claim_boundary": (
            "These are numerical presentation figures. Absolute experimental agreement and material-response claims remain calibration dependent."
        ),
    }
    (out / "presentation_figure_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
