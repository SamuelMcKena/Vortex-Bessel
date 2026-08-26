"""Build a portrait PhD progress poster from the current Vortex-Bessel codebase.

The poster is intentionally broader than the short presentation.  It combines:

* the current dual-SLM -> 4F -> physical-axicon numerical route;
* scalar Bessel/vortex-Bessel propagation and a self-healing benchmark;
* a current-model sensitivity ranking;
* a source-to-sample interface calculation;
* a current Phase-2B V3 transverse 3-D intensity surface;
* exploratory discrete N-fold structured-beam families;
* the committed measured q=20 inverse-reconstruction / falsification evidence.

No experimental image is fabricated.  The q=20 camera data are represented only
through the canonical committed figures.  Model-only panels are regenerated from
the maintained current code.  Material threshold/application calculations remain
planning proxies and are not presented as measured modification outcomes.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import textwrap
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
import numpy as np
import pandas as pd

import bessel_twin_core as bt
import publication_diagnostics as pdiag
from vbb_study import vbb_discrete, vbb_studies
from vbb_study.config import MaterialConfig
from vbb_study.digital_twin.phase2b_visual_cases import Phase2BConfig, build_scalar_cases


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "poster" / "current_progress"

# Poster palette: intentionally brighter and more editorial than the presentation.
PAPER = "#f5f2ea"
INK = "#14212b"
NAVY = "#102c3d"
TEAL = "#0f8f8d"
TEAL_LIGHT = "#d9eeeb"
CORAL = "#e45f56"
CORAL_LIGHT = "#f6ded9"
GOLD = "#d8a12e"
GOLD_LIGHT = "#f5ead0"
PLUM = "#7b5a8e"
PLUM_LIGHT = "#e9deee"
MUTED = "#53636e"
LINE = "#c9d0d1"
WHITE = "#ffffff"
DARKPLOT = "#070b0f"


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _save(fig: plt.Figure, path: Path, *, dpi: int = 260, transparent: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor(), transparent=transparent)
    plt.close(fig)
    return path


def _norm(a: np.ndarray) -> np.ndarray:
    arr = np.maximum(np.asarray(a, dtype=float), 0.0)
    return arr / max(float(np.max(arr)), np.finfo(float).tiny)


def _style_dark_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(DARKPLOT)
    for spine in ax.spines.values():
        spine.set_color("#66737e")
        spine.set_linewidth(0.7)
    ax.tick_params(colors="#d8dee4", labelsize=8)
    ax.xaxis.label.set_color("#d8dee4")
    ax.yaxis.label.set_color("#d8dee4")
    ax.title.set_color(WHITE)


def build_self_healing(out: Path) -> tuple[Path, dict[str, Any]]:
    """Regenerate a compact self-healing benchmark from the maintained scalar engine."""

    base = bt.default_config("fast")
    base = vbb_studies.beam_air_config(base)
    base = replace(base, target=replace(base.target, ell=0))
    design = bt.compute_design_from_config(base)
    obstacle_radius = max(1.5 * bt.um, 0.60 * design.equivalent_l0_first_zero_radius_m)
    bundle = pdiag.build_self_healing_bundle(
        config=base,
        preset="fast",
        path="ideal",
        case_id="B0_self_healing_poster",
        obstacle_kind="disk",
        obstacle_radius_m=obstacle_radius,
        axial_points=51,
    )

    ref0 = np.asarray(bundle["reference_plane"], float)
    obs0 = np.asarray(bundle["obstructed_plane"], float)
    vol = bundle["obstructed_volume"]
    x_um = np.asarray(vol["crop_grid"]["x"], float) / bt.um
    z_um = np.asarray(bundle["z_relative"], float) / bt.um
    extent_xy = [x_um[0], x_um[-1], x_um[0], x_um[-1]]

    # Crop the obstacle-plane views to the same crop as the propagated volume.
    full_x = np.asarray(bundle["grid"]["x"], float)
    ids = np.flatnonzero((full_x >= x_um[0] * bt.um) & (full_x <= x_um[-1] * bt.um))
    if ids.size > 8:
        ref0 = ref0[np.ix_(ids, ids)]
        obs0 = obs0[np.ix_(ids, ids)]

    fig = plt.figure(figsize=(13.6, 6.4), facecolor=DARKPLOT)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.70], width_ratios=[1, 1, 1.55, 1.05], hspace=0.34, wspace=0.25)
    ax_ref = fig.add_subplot(gs[0, 0])
    ax_obs = fig.add_subplot(gs[0, 1])
    ax_xz = fig.add_subplot(gs[0, 2:])
    ax_rec = fig.add_subplot(gs[1, :])
    for ax in (ax_ref, ax_obs, ax_xz, ax_rec):
        _style_dark_axis(ax)

    im = ax_ref.imshow(_norm(ref0) ** 0.45, origin="lower", extent=extent_xy, cmap="magma", vmin=0, vmax=1)
    ax_ref.set(title="before obstruction", xlabel="x (µm)", ylabel="y (µm)")
    ax_obs.imshow(_norm(obs0) ** 0.45, origin="lower", extent=extent_xy, cmap="magma", vmin=0, vmax=1)
    ax_obs.set(title="immediately after", xlabel="x (µm)", ylabel="y (µm)")

    xz = np.asarray(vol["xz"], float)
    ax_xz.imshow(
        _norm(xz) ** 0.45,
        origin="lower",
        aspect="auto",
        extent=[z_um[0], z_um[-1], x_um[0], x_um[-1]],
        cmap="magma",
        vmin=0,
        vmax=1,
    )
    ax_xz.set(title="field reconstruction after the obstruction", xlabel="distance after obstacle (µm)", ylabel="x (µm)")

    peak = np.asarray(bundle["peak_recovery"], float)
    onaxis = np.asarray(bundle["onaxis_recovery"], float)
    ax_rec.plot(z_um, peak, lw=2.3, label="peak recovery")
    ax_rec.plot(z_um, onaxis, lw=2.0, ls="--", label="on-axis recovery")
    ax_rec.axhline(1.0, lw=1.0, ls=":", color="#ccd2d7")
    ax_rec.set(xlabel="distance after obstacle (µm)", ylabel="obstructed / reference", ylim=(0, max(1.15, float(np.nanpercentile(np.r_[peak, onaxis], 98)) * 1.05)))
    ax_rec.grid(alpha=0.16)
    ax_rec.legend(frameon=False, ncol=2, labelcolor=WHITE, loc="lower right")
    fig.colorbar(im, ax=[ax_ref, ax_obs], fraction=0.045, pad=0.03, label="display-normalised intensity")
    fig.suptitle("Self-healing benchmark — the propagated field reconstructs after a local obstruction", color=WHITE, fontsize=16, weight="bold", y=0.99)

    path = _save(fig, out / "03_self_healing_benchmark.png", dpi=300)
    metrics = dict(bundle["metrics"])
    metrics["obstacle_radius_um"] = float(obstacle_radius / bt.um)
    _write_json(out / "03_self_healing_metrics.json", metrics)
    return path, metrics


def _normalised_sensitivity(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if hasattr(bt, "_normalised_sensitivities"):
        try:
            return bt._normalised_sensitivities(df, metric=metric)  # type: ignore[attr-defined]
        except Exception:
            pass
    rows: list[dict[str, Any]] = []
    for knob, group in df.groupby("knob", dropna=False):
        vals = pd.to_numeric(group["knob_value"], errors="coerce")
        mets = pd.to_numeric(group[metric], errors="coerce")
        ok = vals.notna() & mets.notna()
        if ok.sum() >= 2 and float(np.nanmean(np.abs(vals[ok]))) > 0 and float(np.nanmean(np.abs(mets[ok]))) > 0:
            slope = float(np.polyfit(vals[ok], mets[ok], 1)[0])
            sens = slope * float(np.nanmean(vals[ok])) / (float(np.nanmean(mets[ok])) + bt.EPS)
            rows.append({"knob": str(knob), "normalised_sensitivity": sens, "abs_rank_value": abs(sens)})
    return pd.DataFrame(rows).sort_values("abs_rank_value", ascending=False)


def build_sensitivity(out: Path) -> tuple[Path, dict[str, Any]]:
    """Regenerate a current-model OAT sensitivity ranking."""

    base = bt.default_config("fast")
    base = replace(
        base,
        material=MaterialConfig.fused_silica(write_depth_m=300 * bt.um),
        target=replace(base.target, ell=1),
    )
    raw = bt.run_oat_sensitivity(base, preset="fast", save=False)
    metric = "canonical_zone_um" if "canonical_zone_um" in raw.columns else "bessel_zone_um"
    ranked = _normalised_sensitivity(raw, metric)
    if ranked.empty:
        raise RuntimeError("OAT sensitivity run returned no rankable rows")

    top = ranked.head(8).sort_values("abs_rank_value", ascending=True)
    fig, ax = plt.subplots(figsize=(8.4, 5.6), facecolor=WHITE, constrained_layout=True)
    vals = top["normalised_sensitivity"].to_numpy(float)
    labels = [str(v).replace("_", " ") for v in top["knob"]]
    colors = [CORAL if v < 0 else TEAL for v in vals]
    ax.barh(np.arange(len(top)), vals, color=colors, alpha=0.90)
    ax.axvline(0.0, color=INK, lw=1.0)
    ax.set_yticks(np.arange(len(top)), labels)
    ax.set_xlabel(f"normalised sensitivity of {metric.replace('_', ' ')}")
    ax.set_title("Which model parameters most strongly change the Bessel region?", color=INK, fontsize=14, weight="bold")
    ax.grid(axis="x", alpha=0.18)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    path = _save(fig, out / "04_parameter_sensitivity.png", dpi=300)
    raw.to_csv(out / "04_parameter_sensitivity_raw.csv", index=False)
    ranked.to_csv(out / "04_parameter_sensitivity_ranked.csv", index=False)
    return path, {"metric": metric, "top_ranked": ranked.head(8).to_dict(orient="records")}


def build_source_to_sample(out: Path) -> tuple[Path, dict[str, Any]]:
    """Regenerate the maintained air -> interface -> fused-silica planning route."""

    base = bt.default_config("fast")
    cfg = replace(
        base,
        material=MaterialConfig.fused_silica(write_depth_m=300 * bt.um),
        target=replace(base.target, ell=1),
        apply_interface=True,
        correct_interface=False,
        study_kind="full_source_to_sample",
    )
    journey = vbb_studies.run_full_source_to_sample(cfg, correct_interface=False, path="ideal")
    vol = journey.volume
    x_um = np.asarray(vol["crop_grid"]["x"], float) / bt.um
    z_um = np.asarray(vol["z"], float) / bt.um
    xz = np.asarray(vol["xz"], float)
    display = _norm(xz) ** 0.43

    fig, ax = plt.subplots(figsize=(12.8, 5.4), facecolor=DARKPLOT, constrained_layout=True)
    _style_dark_axis(ax)
    ax.imshow(
        display,
        origin="lower",
        aspect="auto",
        extent=[z_um[0], z_um[-1], x_um[0], x_um[-1]],
        cmap="magma",
        vmin=0,
        vmax=1,
    )
    ax.axvline(0.0, color="#6fe5dd", lw=2.0)
    target_um = float(journey.metrics.get("target_write_depth_um", 300.0))
    ax.axvline(target_um, color="#ffd36a", lw=1.4, ls="--")
    ax.text(0.012, 0.93, "AIR", transform=ax.transAxes, color="#c7dce8", fontsize=11, weight="bold")
    ax.text(0.72, 0.93, "FUSED SILICA (planning model)", transform=ax.transAxes, color="#ffd36a", fontsize=11, weight="bold")
    ax.set(title="Continuous source-to-sample field hand-off", xlabel="z relative to sample surface (µm)", ylabel="x (µm)")
    ax.text(0.505, 0.03, "interface", transform=ax.transAxes, color="#6fe5dd", fontsize=9, ha="center")
    path = _save(fig, out / "05_source_to_sample_interface.png", dpi=300)
    return path, {"metrics": dict(journey.metrics), "continuity": dict(journey.continuity)}


def build_v3_3d(out: Path) -> tuple[Path, dict[str, Any]]:
    """Build one current-route V3 z=60 mm transverse intensity surface."""

    study = Phase2BConfig(
        scalar_grid_n=256,
        hex_grid_n=256,
        hero_grid_n=256,
        z_start_m=0.0,
        z_end_m=0.16,
        z_step_m=0.01,
        render_xy_max=180,
        render_z_stride=1,
        sas_pad_factor=2,
        publication_quality=False,
        highn_hero=False,
    )
    study.validate()
    cases = build_scalar_cases(study)
    result = cases["V3"]
    keys = np.asarray(list(result.selected_planes.keys()), float)
    z_key = float(keys[int(np.argmin(np.abs(keys - 60e-3)))])
    plane = np.asarray(result.selected_planes[z_key], float)
    axis = (np.arange(result.native_grid_n, dtype=float) - result.native_grid_n // 2) * float(result.native_dx_m)
    half = float(np.clip(2.7 * max(result.ring_radius_m, 0.08e-3), 0.24e-3, 0.42e-3))
    ids = np.flatnonzero(np.abs(axis) <= half)
    plane = plane[np.ix_(ids, ids)]
    axis = axis[ids]
    plane = _norm(plane)
    stride = max(1, int(math.ceil(max(plane.shape) / 180)))
    plane = plane[::stride, ::stride]
    axis = axis[::stride]
    X, Y = np.meshgrid(axis * 1e3, axis * 1e3)

    fig = plt.figure(figsize=(8.6, 7.2), facecolor=DARKPLOT)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, plane, cmap="magma", linewidth=0, antialiased=True, shade=False, rasterized=True)
    ax.set_facecolor(DARKPLOT)
    ax.set_xlabel("x (mm)", color="#d9e0e5")
    ax.set_ylabel("y (mm)", color="#d9e0e5")
    ax.set_zlabel("normalised intensity", color="#d9e0e5")
    ax.tick_params(colors="#d9e0e5", labelsize=7)
    ax.set_zlim(0, 1.03)
    ax.view_init(elev=47, azim=-52)
    ax.set_box_aspect((1.0, 1.0, 0.54))
    ax.xaxis.pane.set_alpha(0.02)
    ax.yaxis.pane.set_alpha(0.02)
    ax.zaxis.pane.set_alpha(0.02)
    ax.set_title("V3 field morphology at z = 60 mm", color=WHITE, fontsize=15, weight="bold", pad=16)
    fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.07, label="normalised intensity")
    path = _save(fig, out / "06_V3_3d_intensity.png", dpi=320)
    return path, {
        "case_id": "V3",
        "z_m": z_key,
        "native_grid_n": int(result.native_grid_n),
        "native_dx_m": float(result.native_dx_m),
        "source_contract": str(result.metadata.get("source_contract", "")),
    }


def build_structured_gallery(out: Path) -> tuple[Path, dict[str, Any]]:
    """Show maintained discrete N-fold beam families as a secondary research arm."""

    cfg = bt.default_config("fast")
    design = bt.compute_design_from_config(cfg)
    grid = bt.make_xy_grid(320, 0.45 * bt.um)
    patterns = [
        ("triangular", "3-wave triangular"),
        ("square", "4-wave square"),
        ("hexagonal", "6-wave hexagonal"),
        ("honeycomb", "6-wave honeycomb"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.6), facecolor=DARKPLOT, constrained_layout=True)
    extent = [float(grid["x"][0] / bt.um), float(grid["x"][-1] / bt.um)] * 2
    summary: list[dict[str, Any]] = []
    for ax, (name, label) in zip(axes, patterns):
        _style_dark_axis(ax)
        pattern = vbb_discrete.pattern_preset(name)
        field = vbb_discrete.n_wave_complex_field(grid, design.kr_sample_m_inv, pattern, waist_m=38 * bt.um)
        intensity = _norm(np.abs(field) ** 2)
        ax.imshow(intensity, origin="lower", extent=extent, cmap="turbo", vmin=0, vmax=1)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("x (µm)")
        if ax is axes[0]:
            ax.set_ylabel("y (µm)")
        else:
            ax.tick_params(labelleft=False)
        summary.append({"name": name, "N": int(pattern.N)})
    fig.suptitle("Structured-beam extensions already represented in the codebase", color=WHITE, fontsize=14, weight="bold")
    path = _save(fig, out / "07_structured_beam_extensions.png", dpi=300)
    return path, {"patterns": summary, "claim_scope": "exploratory discrete N-fold model family"}


def _existing_asset(path: str) -> Path | None:
    candidate = ROOT / path
    return candidate if candidate.is_file() else None


def _add_figure_image(fig: plt.Figure, rect: tuple[float, float, float, float], path: Path | None, *, facecolor: str = "none", border: bool = False) -> plt.Axes:
    ax = fig.add_axes(rect)
    ax.set_facecolor(facecolor)
    ax.axis("off")
    if path is None or not path.is_file():
        ax.text(0.5, 0.5, "figure unavailable in this build", ha="center", va="center", color=MUTED, fontsize=18)
    else:
        img = plt.imread(path)
        ax.imshow(img)
    if border:
        patch = Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, edgecolor=LINE, linewidth=1.0)
        ax.add_patch(patch)
    return ax


def _caption(fig: plt.Figure, x: float, y: float, width: float, number: int, text: str, *, color: str = MUTED, size: float = 19) -> None:
    wrapped = textwrap.fill(text, width=max(40, int(width * 170)))
    fig.text(x, y, rf"$\bf{{Fig.\ {number}.}}$ {wrapped}", ha="left", va="top", color=color, fontsize=size, linespacing=1.15)


def _section_label(fig: plt.Figure, x: float, y: float, text: str, color: str) -> None:
    fig.text(x, y, text.upper(), ha="left", va="bottom", color=color, fontsize=23, weight="bold", family="DejaVu Sans")
    fig.lines.append(plt.Line2D([x, min(0.965, x + 0.16)], [y - 0.004, y - 0.004], transform=fig.transFigure, color=color, lw=5, solid_capstyle="round"))


def _rounded_block(fig: plt.Figure, rect: tuple[float, float, float, float], face: str, *, edge: str | None = None, radius: float = 0.012, alpha: float = 1.0) -> FancyBboxPatch:
    x, y, w, h = rect
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        transform=fig.transFigure,
        facecolor=face,
        edgecolor=edge or face,
        linewidth=1.0,
        alpha=alpha,
        zorder=-2,
    )
    fig.patches.append(patch)
    return patch


def build_poster(out: Path, generated: Mapping[str, Path]) -> tuple[Path, Path]:
    """Compose an A0 portrait academic poster with an intentionally asymmetric layout."""

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
    })
    fig = plt.figure(figsize=(33.1, 46.8), facecolor=PAPER)

    # Editorial background accents rather than a uniform grid of identical cards.
    fig.patches.append(Rectangle((0, 0.885), 1, 0.115, transform=fig.transFigure, color=NAVY, zorder=-5))
    fig.patches.append(Rectangle((0, 0), 0.012, 1, transform=fig.transFigure, color=TEAL, zorder=-5))
    fig.patches.append(Polygon([[0.64, 0.885], [1.0, 0.885], [1.0, 0.78], [0.82, 0.82]], closed=True, transform=fig.transFigure, facecolor="#173f50", edgecolor="none", zorder=-4))
    fig.patches.append(Polygon([[0.0, 0.292], [0.60, 0.318], [1.0, 0.282], [1.0, 0.255], [0.34, 0.274], [0.0, 0.255]], closed=True, transform=fig.transFigure, facecolor="#e7ece8", edgecolor="none", zorder=-5))

    # Header.
    fig.text(0.045, 0.956, "DIGITAL-TWIN MODELLING AND INVERSE CORRECTION", color=WHITE, fontsize=54, weight="bold", va="top")
    fig.text(0.045, 0.918, "OF PROGRAMMABLE VORTEX–BESSEL BEAMLINES", color="#9de2dc", fontsize=48, weight="bold", va="top")
    fig.text(0.048, 0.891, "Samuel McKenna  •  Applied Optics & Photonics Group  •  Heriot-Watt University", color="#dbe6eb", fontsize=24, va="top")

    # Abstract and central question.
    _section_label(fig, 0.045, 0.852, "Research aim", TEAL)
    abstract = (
        "Programmable vortex–Bessel beams offer extended propagation and controllable transverse structure for ultrafast laser processing, "
        "but the field delivered by a real optical system can differ substantially from an ideal analytical beam. This work develops a physics-based "
        "numerical framework that follows the experimental route through two phase SLMs, explicit 4F order selection and a physical axicon. The same "
        "model is used to explore beam formation, robustness, parameter sensitivity and propagation through an optical interface. Crucially, the forward "
        "model is also used in reverse: multi-plane camera measurements are fitted to the model to infer residual phase/system error, construct a candidate "
        "SLM correction and forward-test that correction before hardware validation. The aim is therefore not only to simulate a Bessel beam, but to build a "
        "measurement-linked digital twin capable of diagnosis, correction and ultimately controlled beam delivery for processing."
    )
    _rounded_block(fig, (0.04, 0.722, 0.43, 0.112), WHITE, edge="#d8dfdc", radius=0.010)
    fig.text(0.058, 0.820, textwrap.fill(abstract, 88), color=INK, fontsize=22.5, va="top", linespacing=1.28)
    _rounded_block(fig, (0.058, 0.735, 0.39, 0.022), TEAL_LIGHT, radius=0.008)
    fig.text(0.073, 0.751, "DESIGN  →  SIMULATE  →  MEASURE  →  INFER  →  CORRECT  →  VALIDATE", color=TEAL, fontsize=20, weight="bold", va="center")

    # Equation / model block.
    _section_label(fig, 0.515, 0.852, "Field model", CORAL)
    _rounded_block(fig, (0.505, 0.718, 0.455, 0.116), CORAL_LIGHT, edge="#edc7c2", radius=0.010)
    eq1 = r"$U_1(x,y)=U_G(x,y)\,\exp[i\ell\,\phi(x,y)]$"
    eq2 = r"$U_2(x,y)=U_1(x,y)\,\exp[i\,2\pi f_c x]$"
    eq3 = r"$U(z)=\mathcal{F}^{-1}\{\mathcal{F}[U(0)]\,e^{iz\sqrt{k^2-k_x^2-k_y^2}}\}$"
    eq4 = r"$\hat{\mathbf p}=\arg\min_{\mathbf p}\sum_j d\!\left(\mathcal N[I_j^{\rm meas}],\,\mathcal N[I_j^{\rm model}(\mathbf p)]\right),\qquad \phi_{\rm corr}\approx-\widehat{\Delta\phi}$"
    fig.text(0.528, 0.816, eq1, color=INK, fontsize=28, va="top")
    fig.text(0.935, 0.816, "(1)", color=CORAL, fontsize=22, ha="right", va="top")
    fig.text(0.528, 0.790, eq2, color=INK, fontsize=28, va="top")
    fig.text(0.935, 0.790, "(2)", color=CORAL, fontsize=22, ha="right", va="top")
    fig.text(0.528, 0.764, eq3, color=INK, fontsize=26, va="top")
    fig.text(0.935, 0.764, "(3)", color=CORAL, fontsize=22, ha="right", va="top")
    fig.text(0.528, 0.737, eq4, color=INK, fontsize=21.5, va="top")
    fig.text(0.935, 0.737, "(4)", color=CORAL, fontsize=22, ha="right", va="top")
    fig.text(0.528, 0.723,
             "Eqs. (1–2) represent the programmed vortex and carrier phases; Eq. (3) is the angular-spectrum propagator used for complex-field propagation. "
             "Eq. (4) summarises the measurement-driven inverse step: model parameters / phase structure are fitted across multiple z-planes and the recovered residual phase is tested as a conjugate correction.",
             color=MUTED, fontsize=17.5, va="bottom", linespacing=1.17)

    # Current numerical route.
    _section_label(fig, 0.045, 0.685, "1  Physics-based optical route", TEAL)
    route = _existing_asset("figures/presentation/01_computational_route_phase2j.png")
    _add_figure_image(fig, (0.045, 0.565, 0.91, 0.108), route)
    _caption(fig, 0.055, 0.558, 0.90, 1,
             "Current numerical route. The complex field is tracked through the same logical planes as the bench: input beam, SLM1, SLM2, explicit 4F order selection, physical refractive axicon and downstream propagation. Hardware effects are introduced at the plane where they physically occur.", size=18.5)

    # Middle: self-healing and 3D hero, deliberately unequal widths/heights.
    _section_label(fig, 0.045, 0.522, "2  Beam behaviour beyond the presentation", GOLD)
    _add_figure_image(fig, (0.045, 0.397, 0.52, 0.112), generated.get("self_healing"))
    _caption(fig, 0.055, 0.391, 0.50, 2,
             "Self-healing benchmark from the maintained scalar engine. A local obstruction suppresses the field at one plane, after which the conical spectrum progressively reconstructs the beam. The recovery curves compare the obstructed field with an unobstructed reference rather than relying on a single attractive image.", size=17.5)

    _rounded_block(fig, (0.59, 0.385, 0.365, 0.126), NAVY, radius=0.012)
    _add_figure_image(fig, (0.605, 0.399, 0.335, 0.097), generated.get("v3_3d"))
    _caption(fig, 0.607, 0.387, 0.33, 3,
             "Current-route V3 transverse intensity at z=60 mm, rendered as a genuine intensity surface. Height and colour encode the same normalised intensity; the plot is not a propagation-volume isosurface.", color="#dfe9ed", size=16.5)

    # Sensitivity + source to sample.
    _add_figure_image(fig, (0.045, 0.296, 0.34, 0.083), generated.get("sensitivity"), border=True)
    _caption(fig, 0.052, 0.291, 0.33, 4,
             "One-at-a-time model sensitivity ranking. This is used to identify which parameters deserve tighter calibration or experimental measurement; it is not a claim that the highest-ranked terms are necessarily the dominant errors in the current bench.", size=15.8)

    _add_figure_image(fig, (0.415, 0.297, 0.54, 0.080), generated.get("source_to_sample"))
    _caption(fig, 0.423, 0.291, 0.52, 5,
             "Maintained source-to-sample calculation. The air-side field is handed explicitly through an interface into fused silica and propagated in the material. This remains a planning-level optical model: no ablation, index change or weld success is inferred from intensity alone.", size=15.8)

    # Inverse section: visually dominant, because it is the main current development.
    _section_label(fig, 0.045, 0.245, "3  Measurement-driven inverse correction", PLUM)
    _rounded_block(fig, (0.035, 0.083, 0.93, 0.148), "#eee8f0", edge="#d6c8dc", radius=0.014)
    q20_quad = _existing_asset("figures/experimental/q20_aberration/reconstruction/realigned_cartesian_xy_measured_fit_corrected_ideal.png")
    q20_phase = _existing_asset("figures/experimental/q20_aberration/reconstruction/annular_aberration_phase.png")
    q20_test = _existing_asset("figures/experimental/q20_aberration/single_mask/single_mask_metrics_vs_z.png")
    q20_recreate = _existing_asset("figures/experimental/q20_aberration/phase_error_recreation/phase_error_recreation_agreement_vs_z.png")
    _add_figure_image(fig, (0.050, 0.128, 0.53, 0.091), q20_quad)
    _add_figure_image(fig, (0.600, 0.148, 0.17, 0.066), q20_phase)
    _add_figure_image(fig, (0.785, 0.148, 0.16, 0.066), q20_test)
    fig.text(0.600, 0.137, "retrieved phase", color=PLUM, fontsize=16.5, weight="bold", ha="left")
    fig.text(0.785, 0.137, "single-mask forward test", color=PLUM, fontsize=16.5, weight="bold", ha="left")
    _caption(fig, 0.052, 0.119, 0.52, 6,
             "Measured q=20 inverse workflow. The measured z-stack is real experimental evidence; the best-fit field and corrected field are model inference. The corrected panel is therefore a prediction of the recovered correction, not a post-correction camera measurement.", color=INK, size=16.8)
    _add_figure_image(fig, (0.608, 0.090, 0.325, 0.042), q20_recreate)
    fig.text(0.608, 0.086,
             "The recovered error is not accepted only because it fits one image: the phase-error recreation and single-mask forward tests ask whether the inferred structure reproduces the measured behaviour across z. Fresh post-SLM measurements remain the final closure step.",
             color=INK, fontsize=15.8, va="top", linespacing=1.15)

    # Bottom, asymmetrical extension/conclusions/reference area.
    _section_label(fig, 0.045, 0.061, "4  Extensions & next steps", CORAL)
    _add_figure_image(fig, (0.045, 0.012, 0.36, 0.043), generated.get("structured"))
    _caption(fig, 0.048, 0.009, 0.35, 7,
             "A secondary research arm explores discrete N-fold / polygonal beam families using the same numerical infrastructure.", size=13.8)

    fig.text(0.435, 0.057, "WHAT THE FRAMEWORK NOW DOES", color=TEAL, fontsize=20, weight="bold", va="top")
    bullet_text = (
        "• reproduces the optical route rather than inserting an ideal beam at the output\n"
        "• predicts qualitative and quantitative signatures of controlled perturbations\n"
        "• evaluates robustness, sensitivity and interface propagation\n"
        "• ingests measured multi-plane data for model fitting and phase reconstruction\n"
        "• forward-tests candidate corrections before experimental application"
    )
    fig.text(0.435, 0.047, bullet_text, color=INK, fontsize=14.6, va="top", linespacing=1.30)

    fig.text(0.715, 0.057, "REFERENCES", color=GOLD, fontsize=20, weight="bold", va="top")
    refs = (
        "[1] J. Durnin, JOSA A 4, 651–654 (1987).\n"
        "[2] J. Durnin, J. J. Miceli Jr. & J. H. Eberly, Phys. Rev. Lett. 58, 1499–1501 (1987).\n"
        "[3] O. Brzobohatý, T. Čižmár & P. Zemánek, Opt. Express 16, 12688–12700 (2008).\n"
        "[4] S. Rao & G. K. Samanta, Opt. Lett. 43, 3029–3032 (2018).\n\n"
        "Code and numerical provenance: SamuelMcKena/Vortex-Bessel"
    )
    fig.text(0.715, 0.047, refs, color=INK, fontsize=12.8, va="top", linespacing=1.22)

    # Footer claim boundary.
    fig.text(0.50, 0.0025,
             "Claim boundary: model-only panels are numerical predictions; q=20 measured images are experimental evidence; retrieved phase/corrected fields are model inference until verified by a fresh post-correction z-stack.",
             ha="center", va="bottom", color=MUTED, fontsize=11.5)

    pdf = out / "Vortex_Bessel_PhD_Progress_Poster_A0_portrait.pdf"
    png = out / "Vortex_Bessel_PhD_Progress_Poster_A0_portrait_preview.png"
    fig.savefig(pdf, format="pdf", bbox_inches="tight", facecolor=PAPER)
    fig.savefig(png, dpi=115, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return pdf, png


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quick", action="store_true", help="kept for CI compatibility; current poster build is already bounded")
    args = parser.parse_args()
    out = args.out
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    generated: dict[str, Path] = {}
    manifest: dict[str, Any] = {
        "poster_title": "Digital-twin modelling and inverse correction of programmable vortex–Bessel beamlines",
        "repository": "SamuelMcKena/Vortex-Bessel",
        "claim_boundary": {
            "model_panels": "numerical/model evidence",
            "q20_measured_panels": "experimental measured evidence where explicitly labelled",
            "q20_retrieval_and_corrected_panels": "model inference / forward prediction",
            "material_panels": "planning-level optical model only",
        },
        "generated": {},
        "errors": {},
    }

    builders = [
        ("self_healing", build_self_healing),
        ("sensitivity", build_sensitivity),
        ("source_to_sample", build_source_to_sample),
        ("v3_3d", build_v3_3d),
        ("structured", build_structured_gallery),
    ]
    for key, builder in builders:
        try:
            path, meta = builder(out)
            generated[key] = path
            manifest["generated"][key] = {"path": str(path.relative_to(ROOT)), "metadata": meta}
            print(f"[poster] built {key}: {path}")
        except Exception as exc:  # poster still builds, but provenance records the failure.
            manifest["errors"][key] = f"{type(exc).__name__}: {exc}"
            print(f"[poster] WARNING {key} failed: {type(exc).__name__}: {exc}")

    pdf, preview = build_poster(out, generated)
    manifest["poster_pdf"] = str(pdf.relative_to(ROOT))
    manifest["poster_preview"] = str(preview.relative_to(ROOT))
    _write_json(out / "poster_manifest.json", manifest)

    readme = out / "README.md"
    readme.write_text(
        "# Current PhD progress poster\n\n"
        "This directory is generated by `tools/build_phd_progress_poster.py`.\n\n"
        "The poster deliberately combines current numerical work that is broader than the short presentation: "
        "self-healing, parameter sensitivity, source-to-sample propagation, current-route 3-D morphology, structured-beam extensions, "
        "and the canonical q=20 inverse-reconstruction evidence.\n\n"
        "## Evidence boundary\n\n"
        "- Model-only panels are numerical predictions.\n"
        "- q=20 measured camera panels are experimental evidence.\n"
        "- q=20 retrieved phase and predicted-corrected fields are model inference, not a completed hardware correction.\n"
        "- Fused-silica/interface panels are optical planning calculations only and do not predict material modification.\n",
        encoding="utf-8",
    )
    print(f"[poster] PDF: {pdf}")
    print(f"[poster] preview: {preview}")


if __name__ == "__main__":
    main()
