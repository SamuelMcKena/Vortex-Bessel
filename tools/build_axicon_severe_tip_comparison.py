"""Severe rounded-axicon comparison: ordinary B0 versus annular non-vortex input.

This is a presentation-focused stress test of the rounded-tip mechanism.  A
large illustrative 800 um radial rounding is used so that the ordinary B0 beam
strongly illuminates the lens-like central region.  A hollow ell=0 field is
then optimised to clear that same apex while preserving a bright on-axis
zero-order Bessel output.

The optimisation metric follows the literature mechanism directly: the
rounded/sharp *on-axis* intensity ratio versus z.  This is the modulation that
Brzobohaty et al. identify as a signature of round-tip interference, and the
quantity Rao & Samanta seek to suppress with hollow input illumination.

Only two presentation files are written:
  * 09_severe_tip_avoidance_xy_xz.png  -- XY at z=60 mm and fixed-lab XZ
  * 09b_severe_tip_avoidance_1D.png    -- transverse lineout + on-axis z trace

The annular field is an axicon-plane planning target derived from the routed B0
field.  It is not claimed as a calibrated phase-only SLM command.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import build_axicon_presentation_evidence_v3 as base
import build_phase2j_presentation_suite as suite
import presentation_phase2j_style as style

from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.vortex_system_route import build_system_route

EPS = np.finfo(float).tiny
SEVERE_TIP_RADIUS_M = 800e-6
Z_REF_M = 60e-3
TRANSITION_M = 45e-6
CANDIDATE_CLEAR_RADII_UM = (820, 860, 900, 950, 1000, 1050, 1100, 1200)
LINE_COLORS = ("#ff9d00", "#39d6ad")


def _hollow_clear_target(field_b0: np.ndarray, grid, clear_radius_m: float) -> np.ndarray:
    """Smooth annular ell=0 field whose dark centre clears the rounded apex."""
    X = np.asarray(grid["X"], float)
    Y = np.asarray(grid["Y"], float)
    R = np.hypot(X, Y)
    # Smooth near-top-hat radial gate.  At R << clear_radius the amplitude is
    # essentially zero; outside the edge it tends smoothly to the routed B0.
    gate = 0.5 * (1.0 + np.tanh((R - float(clear_radius_m)) / TRANSITION_M))
    target = np.asarray(field_b0, np.complex128) * gate
    # Equal pre-axicon power makes the ordinary and hollow inputs comparable.
    p0 = float(np.sum(np.abs(field_b0) ** 2))
    p1 = float(np.sum(np.abs(target) ** 2))
    if p1 > EPS:
        target *= math.sqrt(p0 / p1)
    return target


def _fraction_inside(field: np.ndarray, grid, radius_m: float) -> float:
    R = np.hypot(np.asarray(grid["X"], float), np.asarray(grid["Y"], float))
    p = np.abs(np.asarray(field, np.complex128)) ** 2
    return float(np.sum(p[R <= float(radius_m)]) / max(float(np.sum(p)), EPS))


def _onaxis_from_xz(xz: np.ndarray, coord_m: np.ndarray) -> np.ndarray:
    ix = int(np.argmin(np.abs(np.asarray(coord_m, float))))
    return np.asarray(xz, float)[:, ix]


def _centre_to_peak(xy: np.ndarray, grid) -> float:
    x = np.asarray(grid["x"], float)
    ix = int(np.argmin(np.abs(x)))
    arr = np.asarray(xy, float)
    return float(arr[ix, ix] / max(float(np.max(arr)), EPS))


def _rms_ratio_error(round_axis: np.ndarray, sharp_axis: np.ndarray, z: np.ndarray, z0=40e-3, z1=100e-3) -> float:
    keep = (z >= z0) & (z <= z1)
    ratio = np.asarray(round_axis, float)[keep] / np.maximum(np.asarray(sharp_axis, float)[keep], EPS)
    return float(np.sqrt(np.mean((ratio - 1.0) ** 2)))


def _build_fields(grid_n: int, clear_radius_um: float | None):
    manifest = canonical_hardware_manifest()
    wavelength = float(hardware_value(manifest, "wavelength_m"))
    gamma = math.radians(float(hardware_value(manifest, "axicon_base_angle_deg")))
    n_ax = float(hardware_value(manifest, "axicon_refractive_index"))
    n_ext = float(hardware_value(manifest, "axicon_external_medium_index"))

    route = build_system_route("B0", grid_n=grid_n)
    grid = dict(route["grid"])
    b0 = np.asarray(route["field_on_axicon_plane"], np.complex128)
    hollow = None if clear_radius_um is None else _hollow_clear_target(b0, grid, float(clear_radius_um) * 1e-6)
    sharp_t = base._axicon_transmission(grid, wavelength, gamma, n_ax, n_ext, 0.0)
    round_t = base._axicon_transmission(grid, wavelength, gamma, n_ax, n_ext, SEVERE_TIP_RADIUS_M)
    return wavelength, grid, b0, hollow, sharp_t, round_t


def select_clear_radius(grid_n: int) -> dict:
    wavelength, grid, b0, _, sharp_t, round_t = _build_fields(grid_n, None)
    z = np.asarray(suite.Z_VALUES_M, float)
    coord = np.asarray(suite.TIP_COORD_M, float)

    b0_sharp_xz, _ = base._xz_from_post_axicon(b0 * sharp_t, grid, wavelength, coord, "severe-sweep-b0-sharp")
    b0_round_xz, _ = base._xz_from_post_axicon(b0 * round_t, grid, wavelength, coord, "severe-sweep-b0-round")
    b0_sharp_axis = _onaxis_from_xz(b0_sharp_xz, coord)
    b0_round_axis = _onaxis_from_xz(b0_round_xz, coord)
    b0_rms = _rms_ratio_error(b0_round_axis, b0_sharp_axis, z)
    developed = (z >= 40e-3) & (z <= 100e-3)
    b0_mean_sharp = float(np.mean(b0_sharp_axis[developed]))

    rows = []
    for radius_um in CANDIDATE_CLEAR_RADII_UM:
        hollow = _hollow_clear_target(b0, grid, radius_um * 1e-6)
        h_sharp_xz, _ = base._xz_from_post_axicon(hollow * sharp_t, grid, wavelength, coord, f"severe-sweep-{radius_um}-sharp")
        h_round_xz, _ = base._xz_from_post_axicon(hollow * round_t, grid, wavelength, coord, f"severe-sweep-{radius_um}-round")
        h_sharp_axis = _onaxis_from_xz(h_sharp_xz, coord)
        h_round_axis = _onaxis_from_xz(h_round_xz, coord)
        rms = _rms_ratio_error(h_round_axis, h_sharp_axis, z)
        xy_round = base._xy_from_post_axicon(hollow * round_t, grid, wavelength)
        centre_ratio = _centre_to_peak(xy_round, grid)
        mean_sharp = float(np.mean(h_sharp_axis[developed])) / max(b0_mean_sharp, EPS)
        tip_fraction = _fraction_inside(hollow, grid, SEVERE_TIP_RADIUS_M)
        rows.append({
            "clear_radius_um": int(radius_um),
            "tip_fraction": tip_fraction,
            "developed_onaxis_rms_rounded_vs_sharp": rms,
            "rms_vs_ordinary_B0": rms / max(b0_rms, EPS),
            "mean_sharp_onaxis_vs_B0": mean_sharp,
            "z60_centre_to_peak_rounded": centre_ratio,
        })

    # Require real apex clearance and a genuinely normal, useful on-axis beam.
    admissible = [r for r in rows if r["tip_fraction"] <= 5e-4 and r["mean_sharp_onaxis_vs_B0"] >= 0.20 and r["z60_centre_to_peak_rounded"] >= 0.80]
    if admissible:
        selected = min(admissible, key=lambda r: (r["developed_onaxis_rms_rounded_vs_sharp"], r["clear_radius_um"]))
        mode = "minimum_onaxis_modulation_among_apex-clearing_bright-centre_candidates"
    else:
        selected = min(rows, key=lambda r: (r["developed_onaxis_rms_rounded_vs_sharp"] + 5.0*r["tip_fraction"], -r["z60_centre_to_peak_rounded"]))
        mode = "fallback_minimum_penalty"

    print("Severe-tip sweep (800 um apex):")
    print(f"ordinary B0 developed on-axis rounded/sharp RMS = {b0_rms:.6f}")
    for row in rows:
        print(row)
    print("selected:", selected, mode)
    return {"selected": selected, "rows": rows, "ordinary_b0_rms": b0_rms, "mode": mode}


def render_final(out: Path, grid_n: int, selected_um: int) -> None:
    wavelength, grid, b0, hollow, sharp_t, round_t = _build_fields(grid_n, selected_um)
    assert hollow is not None
    coord = np.asarray(suite.TIP_COORD_M, float)
    z = np.asarray(suite.Z_VALUES_M, float)

    cases = []
    for label, incident in (
        ("ordinary B0 through severe rounded tip", b0),
        (f"annular ell=0, clear radius {selected_um} um", hollow),
    ):
        xy_round = base._xy_from_post_axicon(incident * round_t, grid, wavelength)
        xz_round, _ = base._xz_from_post_axicon(incident * round_t, grid, wavelength, coord, f"severe-final-{label}-round")
        xz_sharp, _ = base._xz_from_post_axicon(incident * sharp_t, grid, wavelength, coord, f"severe-final-{label}-sharp")
        xy_sharp = base._xy_from_post_axicon(incident * sharp_t, grid, wavelength)
        cases.append({
            "label": label,
            "incident": incident,
            "xy_round": xy_round,
            "xy_sharp": xy_sharp,
            "xz_round": xz_round,
            "xz_sharp": xz_sharp,
            "tip_fraction": _fraction_inside(incident, grid, SEVERE_TIP_RADIUS_M),
        })

    # Shared absolute scales, with equal power at the axicon plane.
    xy_ref = max(float(np.max(c["xy_sharp"])) for c in cases)
    xz_ref = max(float(np.max(c["xz_sharp"])) for c in cases)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), constrained_layout=True)
    style.style_fig(fig)
    for col, case in enumerate(cases):
        crop, extent = base._fixed_crop(case["xy_round"], grid, 0.42e-3)
        style.draw_xy(axes[0, col], crop, extent, case["label"], peak=xy_ref, show_y=(col == 0))
        axes[0, col].text(
            0.03, 0.04,
            f"power through 800 um apex region: {100*case['tip_fraction']:.3f}%",
            transform=axes[0, col].transAxes, color=style.TEXT, fontsize=8,
            bbox=dict(facecolor=style.FIG_BG, edgecolor=style.BORDER, alpha=0.88, pad=2.5),
        )
        style.draw_xz(axes[1, col], case["xz_round"], coord, z, peak=xz_ref, show_y=(col == 0), z_ref_m=Z_REF_M)
    fig.suptitle("Severely rounded axicon (800 um): ordinary B0 vs annular tip avoidance", color=style.TEXT, fontsize=16, y=1.02)
    fig.text(0.5, -0.01, "Both inputs have equal pre-axicon power and pass through the same 800 um rounded-tip model. Annular field is non-vortex (ell=0).", ha="center", color=style.MUTED, fontsize=9)
    p_xyxz = out / "09_severe_tip_avoidance_xy_xz.png"
    style.save(fig, p_xyxz)

    # 1D figure: direct camera-plane lineout plus the literature-relevant
    # rounded/sharp on-axis modulation versus propagation distance.
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    style.style_fig(fig)
    for ax in axes:
        base._style_line_axis(ax)

    ref_line = max(float(np.max(cases[0]["xy_sharp"])), EPS)
    for case, colour in zip(cases, LINE_COLORS):
        x, line = base._lineout_x(case["xy_round"], grid, 0.0)
        keep = np.abs(x) <= 0.45e-3
        axes[0].plot(x[keep]*1e3, line[keep]/ref_line, color=colour, lw=1.8, label=case["label"])
    axes[0].set_xlabel("x at fixed y = 0 (mm)")
    axes[0].set_ylabel("intensity / B0 sharp-tip peak")
    axes[0].set_title("Transverse intensity at z = 60 mm", color=style.TEXT, fontsize=12)

    for case, colour in zip(cases, LINE_COLORS):
        sharp_axis = _onaxis_from_xz(case["xz_sharp"], coord)
        round_axis = _onaxis_from_xz(case["xz_round"], coord)
        ratio = round_axis / np.maximum(sharp_axis, EPS)
        axes[1].plot(z*1e3, ratio, color=colour, lw=1.9, label=case["label"])
    axes[1].axhline(1.0, color=style.MUTED, lw=0.9, ls="--", alpha=0.75)
    axes[1].set_xlabel("z from axicon (mm)")
    axes[1].set_ylabel("on-axis intensity: rounded / sharp")
    axes[1].set_title("Rounded-tip on-axis modulation", color=style.TEXT, fontsize=12)

    for ax in axes:
        leg = ax.legend(frameon=False, fontsize=8)
        for text in leg.get_texts():
            text.set_color(style.TEXT)
    p_1d = out / "09b_severe_tip_avoidance_1D.png"
    style.save(fig, p_1d)

    # Console-only audit; no extra presentation files are emitted.
    for case in cases:
        sharp_axis = _onaxis_from_xz(case["xz_sharp"], coord)
        round_axis = _onaxis_from_xz(case["xz_round"], coord)
        rms = _rms_ratio_error(round_axis, sharp_axis, z)
        print(case["label"])
        print(f"  apex power fraction = {case['tip_fraction']:.8f}")
        print(f"  developed 40-100 mm on-axis rounded/sharp RMS = {rms:.8f}")
        print(f"  z60 centre/peak = {_centre_to_peak(case['xy_round'], grid):.8f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-grid-n", type=int, default=1024)
    parser.add_argument("--render-grid-n", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/axicon_severe_tip"))
    args = parser.parse_args()
    if args.sweep_grid_n < 1024 or args.render_grid_n < 1024:
        raise ValueError("severe-tip evidence requires grids >= 1024")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = select_clear_radius(args.sweep_grid_n)
    selected_um = int(result["selected"]["clear_radius_um"])
    render_final(args.output_dir, args.render_grid_n, selected_um)
    print(f"SELECTED_CLEAR_RADIUS_UM={selected_um}")


if __name__ == "__main__":
    main()
