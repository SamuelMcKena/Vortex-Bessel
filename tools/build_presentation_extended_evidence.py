"""Build presentation-only evidence missing from the current Phase 2I figure pack.

This module is deliberately isolated on the presentation audit branch.  It uses
current repository physics and does not alter accepted Phase 2A--2I outputs.
No experimental image is fabricated.

Outputs
-------
07_beam_profile_shaping_hero.png
    B0/V1/V3 transverse profiles with one large representative V1 x-z map.
08_V1_real_error_fingerprints.png
    Nominal versus input pointing, SLM1 registration and 4F iris-offset
    signatures, all produced by the current explicit system route.
09_tip_avoidance_planning_proxy.png
    A shape-only annular-illumination planning proxy at the axicon plane.  This
    is *not* claimed as a calibrated phase-only SLM implementation.
10_synthetic_zstack_inverse_recovery.png
    A one-parameter synthetic inverse proof-of-principle: a multi-plane V1
    intensity stack is generated with a known axicon decentre and recovered by
    sweeping that physical parameter through the same forward model.

Scientific boundaries
---------------------
* x-z evidence uses fixed laboratory coordinates with no per-z recentring.
* perturbation comparisons use a common nominal peak where practical.
* the tip-avoidance panel equalises incident power only for a morphology/tip-
  loading comparison; it is not an efficiency prediction.
* the inverse-recovery panel is synthetic model-to-model evidence, not an
  experimental aberration reconstruction.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.vortex_beam_slm_errors import GaussianBeamError, SLMError
from vbb_study.digital_twin.vortex_continuous_propagation import (
    build_fixed_plane_longitudinal_map,
    build_fixed_support_spectrum,
    native_field_at_z,
)
from vbb_study.digital_twin.vortex_explicit_4f import FourFError
from vbb_study.digital_twin.vortex_system_route import (
    AxiconError,
    SystemErrorConfig,
    build_system_route,
    physical_axicon_on_own_plane,
)
from vbb_study.vbb_materials_study import axial_flatten_apodization

EPS = np.finfo(float).tiny
FIG_BG = "#080b0f"
AX_BG = "#0c1117"
TEXT = "#f1f3f4"
MUTED = "#aeb8c4"
RED = "#ff3b30"
GREEN = "#39d6ad"
BORDER = "#334252"
CMAP = "turbo"
Z_REF_M = 60e-3
Z_VALUES_M = np.linspace(5e-3, 120e-3, 32)
IDEAL_COORD_M = np.linspace(-0.34e-3, 0.34e-3, 341)
ERROR_COORD_M = np.linspace(-1.0e-3, 1.0e-3, 401)


def _style_fig(fig: plt.Figure) -> None:
    fig.patch.set_facecolor(FIG_BG)


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(AX_BG)
    for spine in ax.spines.values():
        spine.set_color("#667380")
        spine.set_linewidth(0.7)
    ax.tick_params(colors=MUTED, labelsize=8, length=2.5)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=260, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _norm(a: np.ndarray, peak: float | None = None) -> np.ndarray:
    a = np.maximum(np.asarray(a, dtype=float), 0.0)
    scale = float(np.max(a)) if peak is None else float(peak)
    return a / max(scale, EPS)


def _prop(route: Mapping[str, Any], z_max_m: float = float(Z_VALUES_M[-1])):
    return build_fixed_support_spectrum(
        np.asarray(route["post_axicon"], dtype=np.complex128),
        dict(route["grid"]),
        wavelength_m=float(route["metadata"]["wavelength_m"]),
        z_max_m=float(z_max_m),
        minimum_retained_spectral_power=0.995,
    )


def _xy(route: Mapping[str, Any], z_m: float = Z_REF_M) -> np.ndarray:
    return np.abs(native_field_at_z(_prop(route), float(z_m))) ** 2


def _xz(
    route: Mapping[str, Any],
    coord_m: np.ndarray,
    *,
    z_values_m: np.ndarray = Z_VALUES_M,
    label: str,
) -> tuple[np.ndarray, float]:
    prop = _prop(route, z_max_m=float(np.max(z_values_m)))
    mapped = build_fixed_plane_longitudinal_map(
        prop,
        z_values_m=np.asarray(z_values_m, dtype=float),
        x_coordinates_m=np.asarray(coord_m, dtype=float),
        y_coordinates_m=np.asarray(coord_m, dtype=float),
        fixed_x_m=0.0,
        fixed_y_m=0.0,
        source_label=label,
    )
    return np.asarray(mapped.xz_intensity, dtype=float), float(prop.retained_spectral_power_fraction)


def _crop(values: np.ndarray, grid: Mapping[str, Any], halfwidth_m: float) -> tuple[np.ndarray, list[float]]:
    x = np.asarray(grid["x"], dtype=float)
    ids = np.flatnonzero(np.abs(x) <= float(halfwidth_m))
    if ids.size < 24:
        raise RuntimeError("presentation crop is under-sampled")
    return np.asarray(values)[np.ix_(ids, ids)], [
        x[ids[0]] * 1e3, x[ids[-1]] * 1e3, x[ids[0]] * 1e3, x[ids[-1]] * 1e3
    ]


def _draw_xy(ax: plt.Axes, values: np.ndarray, extent_mm: Sequence[float], title: str, *, peak: float | None = None, ylabel: bool = False) -> None:
    _style_ax(ax)
    ax.imshow(_norm(values, peak), origin="lower", extent=list(extent_mm), cmap=CMAP, vmin=0, vmax=1, interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=12, pad=6)
    ax.set_xlabel("x (mm)", fontsize=8)
    if ylabel:
        ax.set_ylabel("y (mm)", fontsize=8)
    else:
        ax.tick_params(labelleft=False)
    ax.axhline(0, color="white", alpha=0.18, lw=0.5)
    ax.axvline(0, color="white", alpha=0.18, lw=0.5)


def _draw_xz(ax: plt.Axes, values: np.ndarray, coord_m: np.ndarray, z_values_m: np.ndarray, *, peak: float | None = None, ylabel: bool = True) -> None:
    _style_ax(ax)
    ax.imshow(
        _norm(values.T, peak), origin="lower",
        extent=[float(z_values_m[0])*1e3, float(z_values_m[-1])*1e3, float(coord_m[0])*1e3, float(coord_m[-1])*1e3],
        cmap=CMAP, vmin=0, vmax=1, interpolation="nearest", aspect="auto"
    )
    ax.set_xlabel("z from axicon (mm)", fontsize=8)
    if ylabel:
        ax.set_ylabel("x at fixed y=0 (mm)", fontsize=8)
    else:
        ax.tick_params(labelleft=False)
    ax.axhline(0, color="white", alpha=0.18, lw=0.5)
    ax.axvline(Z_REF_M*1e3, color="white", alpha=0.3, ls="--", lw=0.7)


def build_beam_profile_hero(out: Path, grid_n: int) -> tuple[Path, dict[str, Any]]:
    cases = [
        ("B0", "B0 — Bessel, ℓ = 0"),
        ("V1", "V1 — vortex-Bessel, ℓ = 1"),
        ("V3", "V3 — vortex-Bessel, ℓ = 3"),
    ]
    fig = plt.figure(figsize=(12.2, 7.0), constrained_layout=True)
    _style_fig(fig)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.80])
    metrics: dict[str, Any] = {}
    v1_route = None
    for col, (case, title) in enumerate(cases):
        route = build_system_route(case, grid_n=int(grid_n))
        intensity = _xy(route)
        crop, extent = _crop(intensity, route["grid"], 0.34e-3)
        _draw_xy(fig.add_subplot(gs[0, col]), crop, extent, title, ylabel=(col == 0))
        metrics[case] = {"z_ref_m": Z_REF_M, "peak": float(np.max(intensity))}
        if case == "V1":
            v1_route = route
    assert v1_route is not None
    xz, retained = _xz(v1_route, IDEAL_COORD_M, label="presentation-hero-V1")
    ax = fig.add_subplot(gs[1, :])
    _draw_xz(ax, xz, IDEAL_COORD_M, Z_VALUES_M)
    ax.set_title("Representative V1 longitudinal propagation — fixed laboratory coordinates", fontsize=11, color=TEXT, pad=5)
    fig.suptitle("Beam profile shaping: B0 → V1 → V3", color=TEXT, fontsize=17, y=1.02)
    fig.text(0.5, -0.015, "Transverse structure is programmable; the longitudinal field is propagated through the same current system model.", color=MUTED, ha="center", fontsize=10)
    metrics["V1_fixed_support_retained_power_fraction"] = retained
    return _save(fig, out / "07_beam_profile_shaping_hero.png"), metrics


def build_real_error_fingerprints(out: Path, grid_n: int) -> tuple[Path, list[dict[str, Any]]]:
    manifest = canonical_hardware_manifest()
    iris_radius = float(hardware_value(manifest, "fourier_iris_radius_m"))
    cases: list[tuple[str, SystemErrorConfig]] = [
        ("nominal", SystemErrorConfig()),
        ("beam pointing\n+0.75 mrad", SystemErrorConfig(beam=GaussianBeamError(pointing_rad=(0.75e-3, 0.0)))),
        ("SLM1 registration\n+250 µm", SystemErrorConfig(slm1=SLMError(pattern_offset_m=(250e-6, 0.0)))),
        ("4F iris offset\n+0.35 R", SystemErrorConfig(fourf=FourFError(iris_offset_m=(0.35*iris_radius, 0.0)))),
    ]
    data: list[dict[str, Any]] = []
    for label, config in cases:
        route = build_system_route("V1", grid_n=int(grid_n), config=config)
        intensity = _xy(route)
        xz, retained = _xz(route, ERROR_COORD_M, label=f"presentation-error-{label}")
        data.append({"label": label, "route": route, "xy": intensity, "xz": xz, "retained": retained})
    nominal_xy_peak = float(np.max(data[0]["xy"]))
    nominal_xz_peak = float(np.max(data[0]["xz"]))
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 6.4), constrained_layout=True)
    _style_fig(fig)
    summary: list[dict[str, Any]] = []
    for col, item in enumerate(data):
        crop, extent = _crop(item["xy"], item["route"]["grid"], 1.0e-3)
        _draw_xy(axes[0, col], crop, extent, item["label"], peak=nominal_xy_peak, ylabel=(col == 0))
        _draw_xz(axes[1, col], item["xz"], ERROR_COORD_M, Z_VALUES_M, peak=nominal_xz_peak, ylabel=(col == 0))
        yy, xx = np.indices(item["xy"].shape)
        power = np.maximum(np.asarray(item["xy"], float), 0.0)
        total = float(np.sum(power))
        grid = item["route"]["grid"]
        X = np.asarray(grid["X"], float); Y = np.asarray(grid["Y"], float)
        cx = float(np.sum(power*X)/max(total, EPS)); cy = float(np.sum(power*Y)/max(total, EPS))
        summary.append({"label": item["label"].replace("\n", " "), "centroid_x_m": cx, "centroid_y_m": cy, "fixed_support_retained_power_fraction": float(item["retained"])})
    fig.suptitle("V1: realistic system errors leave different signatures through z", color=TEXT, fontsize=16, y=1.03)
    fig.text(0.5, -0.015, "Same forward model, same fixed lab coordinates, different physical error planes — a basis for experimental diagnosis.", color=MUTED, ha="center", fontsize=10)
    return _save(fig, out / "08_V1_real_error_fingerprints.png"), summary


def _power_fraction_in_radius(field: np.ndarray, grid: Mapping[str, Any], radius_m: float) -> float:
    intensity = np.abs(np.asarray(field, np.complex128))**2
    R = np.hypot(np.asarray(grid["X"], float), np.asarray(grid["Y"], float))
    return float(np.sum(intensity[R <= float(radius_m)]) / max(float(np.sum(intensity)), EPS))


def build_tip_avoidance_proxy(out: Path, grid_n: int) -> tuple[Path, dict[str, Any]]:
    route = build_system_route("V1", grid_n=int(grid_n))
    grid = route["grid"]
    field_ax = np.asarray(route["field_on_axicon_plane"], np.complex128)
    manifest = canonical_hardware_manifest()
    beam_radius = float(hardware_value(manifest, "beam_radius_on_slm_m"))
    wavelength = float(hardware_value(manifest, "wavelength_m"))
    gamma = math.radians(float(hardware_value(manifest, "axicon_base_angle_deg")))
    n_ax = float(hardware_value(manifest, "axicon_refractive_index"))
    n_ext = float(hardware_value(manifest, "axicon_external_medium_index"))
    tip_radius = 200e-6

    R = np.hypot(np.asarray(grid["X"], float), np.asarray(grid["Y"], float))
    plan = axial_flatten_apodization(0.75)
    mask = np.asarray(plan["mask_fn"](R, beam_radius_m=beam_radius), float)
    field_ann = field_ax * mask
    p0 = float(np.sum(np.abs(field_ax)**2)); p1 = float(np.sum(np.abs(field_ann)**2))
    # Equal total power for shape/tip-loading comparison only; not an efficiency claim.
    if p1 > EPS:
        field_ann *= math.sqrt(p0/p1)

    ax_error = AxiconError(
        tip_model="hyperboloidal_round",
        rounding_parameter_m=tip_radius*math.tan(gamma),
    )
    axicon_t, _ = physical_axicon_on_own_plane(
        grid, wavelength_m=wavelength, base_angle_rad=gamma,
        refractive_index=n_ax, external_index=n_ext, error=ax_error,
    )
    post_nom = field_ax * axicon_t
    post_ann = field_ann * axicon_t

    def map_field(field: np.ndarray, label: str) -> tuple[np.ndarray, float]:
        prop = build_fixed_support_spectrum(
            field, dict(grid), wavelength_m=wavelength,
            z_max_m=float(Z_VALUES_M[-1]), minimum_retained_spectral_power=0.995,
        )
        mapped = build_fixed_plane_longitudinal_map(
            prop, z_values_m=Z_VALUES_M,
            x_coordinates_m=IDEAL_COORD_M, y_coordinates_m=IDEAL_COORD_M,
            fixed_x_m=0.0, fixed_y_m=0.0, source_label=label,
        )
        return np.asarray(mapped.xz_intensity, float), float(prop.retained_spectral_power_fraction)

    xz_nom, ret0 = map_field(post_nom, "tip-proxy-nominal")
    xz_ann, ret1 = map_field(post_ann, "tip-proxy-annular")
    frac0 = _power_fraction_in_radius(field_ax, grid, tip_radius)
    frac1 = _power_fraction_in_radius(field_ann, grid, tip_radius)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0), constrained_layout=True)
    _style_fig(fig)
    for col, (field, title, frac) in enumerate([
        (field_ax, "current nominal illumination", frac0),
        (field_ann, "annular target — planning proxy", frac1),
    ]):
        intensity = np.abs(field)**2
        crop, extent = _crop(intensity, grid, 2.5e-3)
        _draw_xy(axes[0, col], crop, extent, title, ylabel=(col == 0))
        axes[0, col].add_patch(Circle((0,0), tip_radius*1e3, fill=False, edgecolor=RED, linewidth=1.5, linestyle="--"))
        axes[0, col].text(0.03, 0.04, f"power inside 200 µm tip: {100*frac:.2f}%", transform=axes[0,col].transAxes, color=TEXT, fontsize=9, bbox=dict(facecolor=FIG_BG, edgecolor=BORDER, alpha=0.85, pad=3))
    common = float(np.max(xz_nom))
    _draw_xz(axes[1,0], xz_nom, IDEAL_COORD_M, Z_VALUES_M, peak=common, ylabel=True)
    _draw_xz(axes[1,1], xz_ann, IDEAL_COORD_M, Z_VALUES_M, peak=common, ylabel=False)
    axes[1,0].set_title("same 200 µm rounded tip", color=TEXT, fontsize=11)
    axes[1,1].set_title("same rounded tip, annular target", color=TEXT, fontsize=11)
    fig.suptitle("Avoiding the axicon tip — target-illumination planning proxy", color=TEXT, fontsize=16, y=1.03)
    fig.text(0.5, -0.015, "Boundary: annular amplitude is imposed at the axicon plane for a shape-only test; a calibrated phase-only SLM encoding is not yet claimed.", color="#f2c14e", ha="center", fontsize=9)
    meta = {
        "tip_radius_m": tip_radius,
        "annular_proxy_strength": float(plan["strength"]),
        "nominal_fractional_power_inside_tip": frac0,
        "annular_fractional_power_inside_tip": frac1,
        "annular_mask_light_cost_before_equal_power_rescale": float(plan["light_cost"]),
        "equal_total_power_for_shape_comparison_only": True,
        "fixed_support_retained_power_fraction": [ret0, ret1],
        "claim_boundary": plan["proxy_note"],
    }
    return _save(fig, out / "09_tip_avoidance_planning_proxy.png"), meta


def _stack_cost(candidate: np.ndarray, target: np.ndarray) -> float:
    c = _norm(candidate)
    t = _norm(target)
    return float(np.sqrt(np.mean((c-t)**2)))


def build_synthetic_inverse_recovery(out: Path, grid_n: int) -> tuple[Path, dict[str, Any]]:
    z = np.linspace(20e-3, 100e-3, 9)
    coord = np.linspace(-0.75e-3, 0.75e-3, 241)
    truth = 300e-6
    candidates = np.arange(-500e-6, 500.1e-6, 100e-6)
    target_route = build_system_route("V1", grid_n=int(grid_n), config=SystemErrorConfig(axicon=AxiconError(decentre_m=(truth,0.0))))
    target, retained_truth = _xz(target_route, coord, z_values_m=z, label="synthetic-inverse-truth")
    costs: list[float] = []
    maps: list[np.ndarray] = []
    retained: list[float] = []
    for value in candidates:
        route = build_system_route("V1", grid_n=int(grid_n), config=SystemErrorConfig(axicon=AxiconError(decentre_m=(float(value),0.0))))
        xz, ret = _xz(route, coord, z_values_m=z, label=f"synthetic-inverse-{value:g}")
        maps.append(xz); retained.append(ret); costs.append(_stack_cost(xz, target))
    best = int(np.argmin(costs))
    recovered = float(candidates[best])
    nominal_idx = int(np.argmin(np.abs(candidates)))

    fig = plt.figure(figsize=(12.6, 6.3), constrained_layout=True)
    _style_fig(fig)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0,1.0,1.15])
    ax0 = fig.add_subplot(gs[0,0]); _draw_xz(ax0,target,coord,z, ylabel=True); ax0.set_title("synthetic target\nunknown to inverse", color=TEXT, fontsize=11)
    ax1 = fig.add_subplot(gs[0,1]); _draw_xz(ax1,maps[nominal_idx],coord,z, ylabel=False); ax1.set_title("nominal model\n0 µm", color=TEXT, fontsize=11)
    ax2 = fig.add_subplot(gs[0,2]); _style_ax(ax2)
    ax2.plot(candidates*1e6, costs, marker="o", lw=1.4)
    ax2.axvline(truth*1e6, color=GREEN, ls="--", lw=1.2, label="truth")
    ax2.axvline(recovered*1e6, color=RED, ls=":", lw=1.5, label="recovered")
    ax2.set_xlabel("candidate axicon decentre (µm)", fontsize=9); ax2.set_ylabel("multi-plane normalized L2", fontsize=9)
    ax2.set_title("inverse cost over 9 z planes", fontsize=11); ax2.legend(frameon=False, labelcolor=TEXT, fontsize=8)

    ax3 = fig.add_subplot(gs[1,0]); _style_ax(ax3)
    residual_nom = _norm(maps[nominal_idx]) - _norm(target)
    vmax = max(float(np.max(np.abs(residual_nom))), EPS)
    ax3.imshow(residual_nom.T, origin="lower", extent=[z[0]*1e3,z[-1]*1e3,coord[0]*1e3,coord[-1]*1e3], cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax3.set_title("nominal − target residual", fontsize=11); ax3.set_xlabel("z (mm)", fontsize=8); ax3.set_ylabel("x (mm)", fontsize=8)
    ax4 = fig.add_subplot(gs[1,1]); _draw_xz(ax4,maps[best],coord,z, ylabel=False); ax4.set_title(f"best fit\n{recovered*1e6:+.0f} µm", color=TEXT, fontsize=11)
    ax5 = fig.add_subplot(gs[1,2]); ax5.set_facecolor(FIG_BG); ax5.axis("off")
    ax5.text(0.02,0.82,"proof of principle",color=RED,fontsize=11,weight="bold")
    ax5.text(0.02,0.67,f"hidden decentre: {truth*1e6:+.0f} µm",color=TEXT,fontsize=13)
    ax5.text(0.02,0.54,f"recovered: {recovered*1e6:+.0f} µm",color=GREEN,fontsize=13,weight="bold")
    ax5.text(0.02,0.36,"many z planes constrain one\nphysical bench parameter better\nthan one camera plane alone.",color=MUTED,fontsize=11,linespacing=1.5)
    ax5.text(0.02,0.10,"Next layer: fit a vector of beam / SLM / 4F /\nlow-order wavefront parameters, then compute\nan additive correction map.",color=TEXT,fontsize=9,linespacing=1.4)
    fig.suptitle("Synthetic z-stack inverse recovery — model-to-model validation", color=TEXT, fontsize=16, y=1.03)
    fig.text(0.5,-0.015,"This is not an experimental reconstruction: the target stack is generated by the same forward-model family with a hidden physical parameter.",color="#f2c14e",ha="center",fontsize=9)
    meta = {
        "truth_axicon_decentre_m": truth,
        "candidate_axicon_decentre_m": candidates.tolist(),
        "costs": costs,
        "recovered_axicon_decentre_m": recovered,
        "z_planes_m": z.tolist(),
        "grid_n": int(grid_n),
        "fixed_support_retained_power_fraction_truth": retained_truth,
        "fixed_support_retained_power_fraction_candidates": retained,
        "claim_boundary": "synthetic model-to-model inverse proof-of-principle; not experimental aberration recovery",
    }
    return _save(fig, out / "10_synthetic_zstack_inverse_recovery.png"), meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-n", type=int, default=1024)
    parser.add_argument("--inverse-grid-n", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/presentation_phase2i"))
    args = parser.parse_args()
    if args.grid_n < 1024:
        raise ValueError("main extended presentation figures require grid_n >= 1024")
    if args.inverse_grid_n < 384:
        raise ValueError("inverse screening grid must be >= 384")
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    p7,m7 = build_beam_profile_hero(out,args.grid_n)
    p8,m8 = build_real_error_fingerprints(out,args.grid_n)
    p9,m9 = build_tip_avoidance_proxy(out,args.grid_n)
    p10,m10 = build_synthetic_inverse_recovery(out,args.inverse_grid_n)
    manifest = {
        "outcome":"PRESENTATION-EXTENDED-EVIDENCE",
        "source_branch_contract":"phase2i-experimental-bench-closure",
        "figures":[str(p) for p in (p7,p8,p9,p10)],
        "beam_profile_hero":m7,
        "real_error_fingerprints":m8,
        "tip_avoidance_planning_proxy":m9,
        "synthetic_inverse_recovery":m10,
    }
    (out/"presentation_extended_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest,indent=2))


if __name__ == "__main__":
    main()
