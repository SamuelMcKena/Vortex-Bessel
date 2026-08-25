"""Final presentation evidence for axicon decentre and rounded-tip mitigation.

Physics policy
--------------
1. Axicon-decentre XY panels are *presentation crops only*.  Each panel is
   centred on the imposed axicon displacement so the displaced beam is in the
   middle of the frame, but the axis labels remain absolute laboratory x/y.
   The x window is deliberately wider than the y window so important rings are
   not clipped.  XZ maps remain fixed-laboratory y=0 with no recentering.
2. Rounded-tip avoidance follows the established physical mechanism: the
   rounded central region acts lens-like and its transmitted field interferes
   with the conically refracted field, producing longitudinal modulation.
   We therefore compare the SAME axicon under four illuminations/conditions:
      A sharp-tip B0 reference,
      B rounded-tip B0 (tip-interference case),
      C rounded-tip V1 from the current system route,
      D rounded-tip wide-core V1 planning target whose central dark region
        exceeds the rounded-apex radius.
   No no-axicon control is used.
3. The wide-core V1 field is an axicon-plane optical target used to test the
   mitigation concept.  Its pre-axicon total power is matched to the natural V1
   field, but output peaks are never renormalised to equality.  It is not yet a
   calibrated phase-only SLM implementation.

Literature basis
----------------
Brzobohaty et al., Optics Express 16, 12688-12700 (2008),
DOI 10.1364/OE.16.012688: rounded-tip refracted light interferes with the
quasi-Bessel field and produces unwanted longitudinal intensity modulation.
Rao & Samanta, Optics Letters 43, 3029-3032 (2018): hollow input beams with a
central dark region suppress the effect of the imperfect axicon tip.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

import build_phase2i_presentation_figures as core
import build_phase2j_presentation_suite as suite
import presentation_phase2j_style as style

from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.vortex_continuous_propagation import (
    build_fixed_plane_longitudinal_map,
    build_fixed_support_spectrum,
    native_field_at_z,
)
from vbb_study.digital_twin.vortex_system_route import (
    AxiconError,
    SystemErrorConfig,
    build_system_route,
    physical_axicon_on_own_plane,
)

EPS = np.finfo(float).tiny
Z_REF_M = 60e-3
DECENTRES_M = (-500e-6, 0.0, 500e-6)
DECENTRE_LABELS = ("−500 µm", "aligned", "+500 µm")
TIP_RADII_M = (0.0, 200e-6, 800e-6)
TIP_LABELS = ("ideal sharp tip", "200 µm radial rounding", "800 µm radial rounding")

# Deliberately rectangular: preserve more displaced-ring data along x.
DECENTRE_X_HALF_M = 0.38e-3
DECENTRE_Y_HALF_M = 0.24e-3
DECENTRE_1D_LAB_HALF_M = 0.90e-3
TIP_XY_HALF_M = 0.22e-3
TIP_1D_HALF_M = 0.34e-3
TIP_RADIUS_M = 200e-6
WIDE_VORTEX_CORE_M = 260e-6
LINE_COLORS = ("#fff176", "#ff9d00", "#ff453a", "#39d6ad")


def _style_line_axis(ax: plt.Axes) -> None:
    style.style_ax(ax)
    ax.grid(alpha=0.14, linewidth=0.55)


def _centroid(intensity: np.ndarray, grid: Mapping[str, Any]) -> tuple[float, float]:
    p = np.maximum(np.asarray(intensity, float), 0.0)
    total = float(np.sum(p))
    X = np.asarray(grid["X"], float)
    Y = np.asarray(grid["Y"], float)
    return float(np.sum(p * X) / max(total, EPS)), float(np.sum(p * Y) / max(total, EPS))


def _rect_crop(values: np.ndarray, grid: Mapping[str, Any], cx: float, cy: float, xhalf: float, yhalf: float):
    x = np.asarray(grid["x"], float)
    ix = np.flatnonzero(np.abs(x - float(cx)) <= float(xhalf))
    iy = np.flatnonzero(np.abs(x - float(cy)) <= float(yhalf))
    if ix.size < 70 or iy.size < 70:
        raise RuntimeError(f"under-sampled presentation crop: {ix.size} x {iy.size}")
    return np.asarray(values)[np.ix_(iy, ix)], [
        x[ix[0]] * 1e3, x[ix[-1]] * 1e3, x[iy[0]] * 1e3, x[iy[-1]] * 1e3
    ]


def _fixed_crop(values: np.ndarray, grid: Mapping[str, Any], halfwidth: float):
    return _rect_crop(values, grid, 0.0, 0.0, halfwidth, halfwidth)


def _lineout_x(intensity: np.ndarray, grid: Mapping[str, Any], y_m: float = 0.0):
    x = np.asarray(grid["x"], float)
    iy = int(np.argmin(np.abs(x - float(y_m))))
    return x, np.asarray(intensity, float)[iy]


def _prop(field: np.ndarray, grid: Mapping[str, Any], wavelength_m: float, zmax: float | None = None):
    return build_fixed_support_spectrum(
        np.asarray(field, np.complex128), dict(grid), wavelength_m=float(wavelength_m),
        z_max_m=float(suite.Z_VALUES_M[-1] if zmax is None else zmax),
        minimum_retained_spectral_power=0.995,
    )


def _xy_from_post_axicon(field: np.ndarray, grid: Mapping[str, Any], wavelength_m: float):
    prop = _prop(field, grid, wavelength_m, Z_REF_M)
    return np.abs(native_field_at_z(prop, Z_REF_M)) ** 2


def _xz_from_post_axicon(field: np.ndarray, grid: Mapping[str, Any], wavelength_m: float, coord_m: np.ndarray, label: str):
    prop = _prop(field, grid, wavelength_m)
    mapped = build_fixed_plane_longitudinal_map(
        prop,
        z_values_m=np.asarray(suite.Z_VALUES_M, float),
        x_coordinates_m=np.asarray(coord_m, float),
        y_coordinates_m=np.asarray(coord_m, float),
        fixed_x_m=0.0,
        fixed_y_m=0.0,
        source_label=label,
    )
    return np.asarray(mapped.xz_intensity, float), float(prop.retained_spectral_power_fraction)


def _rounded_tip_error(radius_m: float, gamma: float) -> AxiconError:
    if float(radius_m) <= 0.0:
        return AxiconError(tip_model="sharp")
    return AxiconError(
        tip_model="hyperboloidal_round",
        rounding_parameter_m=float(radius_m) * math.tan(float(gamma)),
    )


def _axicon_transmission(grid, wavelength, gamma, n_ax, n_ext, radius_m):
    return physical_axicon_on_own_plane(
        grid,
        wavelength_m=wavelength,
        base_angle_rad=gamma,
        refractive_index=n_ax,
        external_index=n_ext,
        error=_rounded_tip_error(radius_m, gamma),
    )[0]


def _wide_core_v1_target(field_v1: np.ndarray, grid: Mapping[str, Any], core_radius_m: float) -> np.ndarray:
    """Enlarge the V1 central null while preserving the routed V1 phase.

    This is deliberately an axicon-plane target, not a claim that the current
    phase-only SLM pair already realises this exact complex amplitude.
    """
    X = np.asarray(grid["X"], float)
    Y = np.asarray(grid["Y"], float)
    R = np.hypot(X, Y)
    r0 = max(float(core_radius_m), EPS)
    # Smooth high-order notch: ~zero inside the apex and approaches unity
    # outside it without introducing a hard diffracting edge.
    notch = 1.0 - np.exp(-(R / r0) ** 6)
    target = np.asarray(field_v1, np.complex128) * notch
    p0 = float(np.sum(np.abs(field_v1) ** 2))
    p1 = float(np.sum(np.abs(target) ** 2))
    if p1 > EPS:
        target *= math.sqrt(p0 / p1)
    return target


def _fraction_inside(field: np.ndarray, grid: Mapping[str, Any], radius_m: float) -> float:
    R = np.hypot(np.asarray(grid["X"], float), np.asarray(grid["Y"], float))
    p = np.abs(np.asarray(field, np.complex128)) ** 2
    return float(np.sum(p[R <= radius_m]) / max(float(np.sum(p)), EPS))


def build_decentre(out: Path, grid_n: int):
    cases = []
    for dec, label in zip(DECENTRES_M, DECENTRE_LABELS):
        route = build_system_route(
            "V1", grid_n=grid_n,
            config=SystemErrorConfig(axicon=AxiconError(decentre_m=(dec, 0.0))),
        )
        field, _ = core._xy_at_z(route)
        intensity = np.abs(field) ** 2
        mapped, prop = core._longitudinal(route, suite.DECENTRE_COORD_M, f"final-decentre-{dec:g}")
        cx, cy = _centroid(intensity, route["grid"])
        cases.append({
            "dec": dec, "label": label, "route": route, "intensity": intensity,
            "xz": np.asarray(mapped.xz_intensity, float), "centroid_x_m": cx,
            "centroid_y_m": cy, "retained": float(prop.retained_spectral_power_fraction),
        })

    ref_xy = float(np.max(cases[1]["intensity"]))
    ref_xz = float(np.max(cases[1]["xz"]))
    fig, axes = plt.subplots(2, 3, figsize=(12.7, 6.8), constrained_layout=True)
    style.style_fig(fig)
    for col, c in enumerate(cases):
        crop, extent = _rect_crop(
            c["intensity"], c["route"]["grid"], c["dec"], 0.0,
            DECENTRE_X_HALF_M, DECENTRE_Y_HALF_M,
        )
        style.draw_xy(axes[0, col], crop, extent, c["label"], peak=ref_xy, show_y=(col == 0))
        style.draw_xz(
            axes[1, col], c["xz"], suite.DECENTRE_COORD_M, suite.Z_VALUES_M,
            peak=ref_xz, show_y=(col == 0), z_ref_m=Z_REF_M,
        )
    fig.suptitle("V1 axicon lateral decentre — expanded XY field, fixed-lab XZ", color=style.TEXT, fontsize=16, y=1.025)
    fig.text(
        0.5, -0.012,
        "XY windows are centred on the imposed axicon displacement and widened along x; ticks remain absolute laboratory coordinates. XZ is unchanged at fixed y=0.",
        ha="center", color=style.MUTED, fontsize=9,
    )
    p2d = out / "04_V1_axicon_decentre_fixed_lab_thermal_tight.png"
    style.save(fig, p2d)

    # Full common laboratory x range for all three traces: no disjoint/clipped curves.
    _, aligned = _lineout_x(cases[1]["intensity"], cases[1]["route"]["grid"], 0.0)
    ref = max(float(np.max(aligned)), EPS)
    fig, ax = plt.subplots(figsize=(10.2, 4.9), constrained_layout=True)
    style.style_fig(fig); _style_line_axis(ax)
    rows = []
    for c, colour in zip(cases, LINE_COLORS[:3]):
        x, line = _lineout_x(c["intensity"], c["route"]["grid"], 0.0)
        keep = np.abs(x) <= DECENTRE_1D_LAB_HALF_M
        ax.plot(x[keep] * 1e3, line[keep] / ref, color=colour, lw=1.7, label=c["label"])
        ax.axvline(c["dec"] * 1e3, color=colour, alpha=0.28, lw=0.8, ls="--")
        rows.append({
            "label": c["label"], "imposed_decentre_m": c["dec"],
            "centroid_x_m": c["centroid_x_m"], "peak_ratio_to_aligned": float(np.max(line) / ref),
        })
    ax.set_xlim(-DECENTRE_1D_LAB_HALF_M * 1e3, DECENTRE_1D_LAB_HALF_M * 1e3)
    ax.set_xlabel("laboratory x at fixed y = 0 (mm)")
    ax.set_ylabel("intensity / aligned peak")
    ax.set_title("V1 axicon decentre — full 1D transverse intensity at z = 60 mm", color=style.TEXT, fontsize=13)
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts(): t.set_color(style.TEXT)
    p1d = out / "04b_V1_axicon_decentre_1D_intensity.png"
    style.save(fig, p1d)
    return p2d, p1d, rows


def build_rounded_tip(out: Path, grid_n: int):
    manifest = canonical_hardware_manifest()
    gamma = math.radians(float(hardware_value(manifest, "axicon_base_angle_deg")))
    cases = []
    for radius, label in zip(TIP_RADII_M, TIP_LABELS):
        route = build_system_route(
            "V1", grid_n=grid_n,
            config=SystemErrorConfig(axicon=_rounded_tip_error(radius, gamma)),
        )
        field, _ = core._xy_at_z(route)
        intensity = np.abs(field) ** 2
        mapped, prop = core._longitudinal(route, suite.TIP_COORD_M, f"final-tip-{radius:g}")
        cases.append({
            "radius": radius, "label": label, "route": route, "intensity": intensity,
            "xz": np.asarray(mapped.xz_intensity, float), "retained": float(prop.retained_spectral_power_fraction),
        })
    ref_xy = float(np.max(cases[0]["intensity"]))
    ref_xz = float(np.max(cases[0]["xz"]))
    fig, axes = plt.subplots(2, 3, figsize=(11.9, 6.6), constrained_layout=True)
    style.style_fig(fig)
    for col, c in enumerate(cases):
        crop, extent = _fixed_crop(c["intensity"], c["route"]["grid"], TIP_XY_HALF_M)
        style.draw_xy(axes[0, col], crop, extent, c["label"], peak=ref_xy, show_y=(col == 0))
        style.draw_xz(axes[1, col], c["xz"], suite.TIP_COORD_M, suite.Z_VALUES_M, peak=ref_xz, show_y=(col == 0), z_ref_m=Z_REF_M)
    fig.suptitle("V1 non-ideal axicon tip — shared sharp-tip intensity reference", color=style.TEXT, fontsize=16, y=1.025)
    p2d = out / "05_V1_nonideal_tip_fixed_lab_thermal_tight.png"
    style.save(fig, p2d)

    _, sharp = _lineout_x(cases[0]["intensity"], cases[0]["route"]["grid"], 0.0)
    ref = max(float(np.max(sharp)), EPS)
    fig, ax = plt.subplots(figsize=(10.0, 4.8), constrained_layout=True)
    style.style_fig(fig); _style_line_axis(ax)
    rows = []
    for c, colour in zip(cases, LINE_COLORS[:3]):
        x, line = _lineout_x(c["intensity"], c["route"]["grid"], 0.0)
        keep = np.abs(x) <= TIP_1D_HALF_M
        ax.plot(x[keep] * 1e3, line[keep] / ref, color=colour, lw=1.8, label=c["label"])
        rows.append({"label": c["label"], "peak_ratio_to_sharp": float(np.max(line) / ref)})
    ax.set_xlabel("x at fixed y = 0 (mm)")
    ax.set_ylabel("intensity / sharp-tip peak")
    ax.set_title("V1 rounded-tip axicon — 1D transverse intensity at z = 60 mm", color=style.TEXT, fontsize=13)
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts(): t.set_color(style.TEXT)
    p1d = out / "05b_V1_nonideal_tip_1D_intensity.png"
    style.save(fig, p1d)
    return p2d, p1d, rows


def build_tip_avoidance(out: Path, grid_n: int):
    manifest = canonical_hardware_manifest()
    wavelength = float(hardware_value(manifest, "wavelength_m"))
    gamma = math.radians(float(hardware_value(manifest, "axicon_base_angle_deg")))
    n_ax = float(hardware_value(manifest, "axicon_refractive_index"))
    n_ext = float(hardware_value(manifest, "axicon_external_medium_index"))

    b0 = build_system_route("B0", grid_n=grid_n)
    v1 = build_system_route("V1", grid_n=grid_n)
    grid = dict(b0["grid"])
    field_b0 = np.asarray(b0["field_on_axicon_plane"], np.complex128)
    field_v1 = np.asarray(v1["field_on_axicon_plane"], np.complex128)
    field_v1_wide = _wide_core_v1_target(field_v1, grid, WIDE_VORTEX_CORE_M)

    sharp_t = _axicon_transmission(grid, wavelength, gamma, n_ax, n_ext, 0.0)
    round_t = _axicon_transmission(grid, wavelength, gamma, n_ax, n_ext, TIP_RADIUS_M)

    cases = [
        {"label": "B0 → sharp tip\nideal reference", "incident": field_b0, "post": field_b0 * sharp_t, "kind": "sharp B0"},
        {"label": "B0 → 200 µm rounded tip\ntip-interference case", "incident": field_b0, "post": field_b0 * round_t, "kind": "rounded B0"},
        {"label": "V1 → same rounded tip\nnatural vortex null", "incident": field_v1, "post": field_v1 * round_t, "kind": "rounded V1"},
        {"label": "wide-core V1 → same rounded tip\napex-avoidance target", "incident": field_v1_wide, "post": field_v1_wide * round_t, "kind": "rounded wide V1"},
    ]

    for i, c in enumerate(cases):
        c["inside_tip_fraction"] = _fraction_inside(c["incident"], grid, TIP_RADIUS_M)
        c["xy"] = _xy_from_post_axicon(c["post"], grid, wavelength)
        c["xz"], c["retained"] = _xz_from_post_axicon(c["post"], grid, wavelength, suite.TIP_COORD_M, f"tip-avoidance-final-{i}")

    # Shared physical output scale referenced to sharp B0; do not peak-normalise each case.
    ref_xy = max(float(np.max(cases[0]["xy"])), EPS)
    ref_xz = max(float(np.max(cases[0]["xz"])), EPS)

    fig, axes = plt.subplots(3, 4, figsize=(15.8, 10.0), constrained_layout=True)
    style.style_fig(fig)
    incident_ref = max(float(np.max(np.abs(field_b0) ** 2)), float(np.max(np.abs(field_v1) ** 2)), float(np.max(np.abs(field_v1_wide) ** 2)), EPS)
    for col, c in enumerate(cases):
        inc = np.abs(c["incident"]) ** 2
        crop, extent = _fixed_crop(inc, grid, 0.65e-3)
        style.draw_xy(axes[0, col], crop, extent, c["label"], peak=incident_ref, show_y=(col == 0))
        axes[0, col].add_patch(Circle((0, 0), TIP_RADIUS_M * 1e3, fill=False, edgecolor="#39d6ad", lw=1.2, ls="--"))
        axes[0, col].text(
            0.03, 0.04, f"power inside 200 µm: {100*c['inside_tip_fraction']:.3f}%",
            transform=axes[0, col].transAxes, color=style.TEXT, fontsize=8,
            bbox=dict(facecolor=style.FIG_BG, edgecolor=style.BORDER, alpha=0.88, pad=2.5),
        )

        xy_crop, xy_extent = _fixed_crop(c["xy"], grid, 0.34e-3)
        style.draw_xy(axes[1, col], xy_crop, xy_extent, "output at z = 60 mm", peak=ref_xy, show_y=(col == 0))
        axes[1, col].text(
            0.03, 0.04, f"peak / sharp-B0 = {np.max(c['xy'])/ref_xy:.3f}",
            transform=axes[1, col].transAxes, color=style.TEXT, fontsize=8,
            bbox=dict(facecolor=style.FIG_BG, edgecolor=style.BORDER, alpha=0.88, pad=2.5),
        )

        style.draw_xz(axes[2, col], c["xz"], suite.TIP_COORD_M, suite.Z_VALUES_M, peak=ref_xz, show_y=(col == 0), z_ref_m=Z_REF_M)

    axes[0,0].set_ylabel("axicon-plane y (mm)")
    axes[1,0].set_ylabel("y at 60 mm (mm)")
    axes[2,0].set_ylabel("x at fixed y=0 (mm)")
    fig.suptitle("Rounded-tip axicon: interference mechanism and vortex-based tip avoidance", color=style.TEXT, fontsize=17, y=1.01)
    fig.text(
        0.5, -0.008,
        "Dashed circle = 200 µm rounded-apex region. All rounded cases use the same physical axicon; output panels share the sharp-B0 intensity scale. Wide-core V1 is an axicon-plane planning target, not yet a calibrated phase-only SLM command.",
        ha="center", color=style.GOLD, fontsize=9,
    )
    pmain = out / "09_tip_avoidance_three_way_audit.png"
    style.save(fig, pmain)

    # Axial peak-intensity diagnostic: this is the direct signature of rounded-tip interference.
    fig, ax = plt.subplots(figsize=(10.4, 5.0), constrained_layout=True)
    style.style_fig(fig); _style_line_axis(ax)
    for c, colour in zip(cases, LINE_COLORS):
        peak_z = np.max(c["xz"], axis=1) / ref_xz
        ax.plot(suite.Z_VALUES_M * 1e3, peak_z, color=colour, lw=1.8, label=c["label"].replace("\n", " — "))
    ax.set_xlabel("z from axicon (mm)")
    ax.set_ylabel("transverse peak intensity / sharp-B0 peak")
    ax.set_title("Rounded-tip interference audit — peak intensity versus propagation distance", color=style.TEXT, fontsize=13)
    leg = ax.legend(frameon=False, fontsize=8)
    for t in leg.get_texts(): t.set_color(style.TEXT)
    pax = out / "09b_tip_avoidance_three_way_1D_intensity.png"
    style.save(fig, pax)

    # z=60 transverse lineouts retained as a separate practical camera-plane diagnostic.
    fig, ax = plt.subplots(figsize=(10.4, 5.0), constrained_layout=True)
    style.style_fig(fig); _style_line_axis(ax)
    for c, colour in zip(cases, LINE_COLORS):
        x, line = _lineout_x(c["xy"], grid, 0.0)
        keep = np.abs(x) <= 0.38e-3
        ax.plot(x[keep] * 1e3, line[keep] / ref_xy, color=colour, lw=1.7, label=c["label"].replace("\n", " — "))
    ax.set_xlabel("x at fixed y = 0 (mm)")
    ax.set_ylabel("intensity / sharp-B0 z=60 mm peak")
    ax.set_title("Rounded-tip avoidance — transverse 1D intensity at z = 60 mm", color=style.TEXT, fontsize=13)
    leg = ax.legend(frameon=False, fontsize=8)
    for t in leg.get_texts(): t.set_color(style.TEXT)
    ptrans = out / "09c_tip_avoidance_transverse_1D_intensity.png"
    style.save(fig, ptrans)

    audit = []
    for c in cases:
        peak_z = np.max(c["xz"], axis=1)
        # Modulation metric over 20-100 mm to avoid edge transients.
        z = np.asarray(suite.Z_VALUES_M, float)
        keep = (z >= 20e-3) & (z <= 100e-3)
        vals = peak_z[keep]
        modulation_cv = float(np.std(vals) / max(float(np.mean(vals)), EPS))
        audit.append({
            "label": c["kind"],
            "fraction_incident_power_inside_200um_tip": c["inside_tip_fraction"],
            "z60_peak_ratio_to_sharp_B0": float(np.max(c["xy"]) / ref_xy),
            "peak_vs_z_modulation_cv_20_to_100mm": modulation_cv,
            "fixed_support_retained_power_fraction": c["retained"],
        })
    return pmain, pax, ptrans, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-n", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/axicon_presentation_evidence"))
    args = parser.parse_args()
    if args.grid_n < 1024:
        raise ValueError("presentation evidence requires grid_n >= 1024")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    p04, p04b, dec = build_decentre(out, args.grid_n)
    p05, p05b, tip = build_rounded_tip(out, args.grid_n)
    p09, p09b, p09c, audit = build_tip_avoidance(out, args.grid_n)

    manifest = {
        "outcome": "AXICON-PRESENTATION-EVIDENCE-V3-ROUNDED-TIP-PHYSICS-AUDIT",
        "grid_n": int(args.grid_n),
        "z_ref_m": Z_REF_M,
        "figures": [str(p) for p in (p04, p04b, p05, p05b, p09, p09b, p09c)],
        "decentre_xy_policy": {
            "x_halfwidth_m": DECENTRE_X_HALF_M,
            "y_halfwidth_m": DECENTRE_Y_HALF_M,
            "centres_m": list(DECENTRES_M),
            "axes_are_absolute_lab_coordinates": True,
            "xz_fixed_lab_y0_unchanged": True,
        },
        "decentre_1d": dec,
        "rounded_tip_1d": tip,
        "tip_avoidance_audit": audit,
        "tip_radius_m": TIP_RADIUS_M,
        "wide_vortex_core_m": WIDE_VORTEX_CORE_M,
        "literature_basis": [
            "Brzobohaty et al., Optics Express 16, 12688-12700 (2008), DOI 10.1364/OE.16.012688",
            "Rao and Samanta, Optics Letters 43, 3029-3032 (2018), hollow-input suppression of axicon-tip modulation",
        ],
        "claim_boundary": "wide-core V1 is an axicon-plane complex-field planning target; calibrated phase-only SLM realisation is not claimed",
    }
    (out / "axicon_presentation_evidence_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
