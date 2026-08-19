"""Localized severe rounded-cap axicon stress test with only XY, XZ and 1D outputs.

Why this model exists
---------------------
The repository's hyperboloidal_round parameter is a curvature scale, not a hard
radial footprint: its phase defect relaxes gradually well beyond the nominal
parameter.  Therefore drawing a circle at that parameter and claiming that a
slightly larger annulus 'avoids the tip' is not a valid avoidance test.

For the presentation mechanism we instead use an explicit *localized* rounded
cap defect on top of the exact sharp-cone transmission.  The defect has a
finite radial footprint and smoothly joins the sharp cone with zero value and
zero radial derivative at the footprint edge.  Outside that footprint the
transmission is exactly the sharp axicon transmission up to no extra phase.
This makes the causal experiment well posed:

  ordinary B0 overlaps the rounded-cap defect -> central contribution interferes;
  hollow ell=0 clears the same defect -> only the unaffected conical region is used.

This is an illustrative severe-tip stress-test, not a calibrated surface map of
the laboratory axicon and not a calibrated phase-only SLM implementation.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

import build_axicon_presentation_evidence_v3 as base
import build_phase2j_presentation_suite as suite
import presentation_phase2j_style as style

from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.vortex_system_route import build_system_route

EPS = np.finfo(float).tiny
TIP_FOOTPRINT_M = 400e-6
Z_REF_M = 80e-3
ANNULAR_EDGE_M = 35e-6
CANDIDATE_CLEAR_RADII_UM = (420, 440, 460, 480, 500, 530, 560, 600)
COLORS = ("#ff9d00", "#39d6ad")


def _hollow_l0(field_b0: np.ndarray, grid, clear_radius_m: float) -> np.ndarray:
    X = np.asarray(grid["X"], float)
    Y = np.asarray(grid["Y"], float)
    R = np.hypot(X, Y)
    gate = 0.5 * (1.0 + np.tanh((R - float(clear_radius_m)) / ANNULAR_EDGE_M))
    target = np.asarray(field_b0, np.complex128) * gate
    p0 = float(np.sum(np.abs(field_b0)**2))
    p1 = float(np.sum(np.abs(target)**2))
    if p1 > EPS:
        target *= math.sqrt(p0 / p1)
    return target


def _localized_round_transmission(grid, sharp_t: np.ndarray, wavelength: float, gamma: float, n_ax: float, n_ext: float) -> np.ndarray:
    """Sharp axicon plus a smooth, finite-support rounded-cap OPD defect."""
    X = np.asarray(grid["X"], float)
    Y = np.asarray(grid["Y"], float)
    R = np.hypot(X, Y)
    rho = R / TIP_FOOTPRINT_M
    support = rho < 1.0

    # Relative cap height.  At the centre this equals the cone height that would
    # occur at the footprint radius, giving a deliberately severe rounded-apex
    # stress test.  The quartic edge factor makes both defect and radial slope
    # vanish smoothly where the cap rejoins the sharp cone.
    h0 = TIP_FOOTPRINT_M * math.tan(float(gamma))
    delta_sag = np.zeros_like(R)
    delta_sag[support] = h0 * (1.0 - rho[support]**2)**2
    k0 = 2.0 * np.pi / float(wavelength)
    defect_phase = -k0 * (float(n_ax) - float(n_ext)) * delta_sag
    return np.asarray(sharp_t, np.complex128) * np.exp(1j * defect_phase)


def _fraction_inside(field: np.ndarray, grid) -> float:
    R = np.hypot(np.asarray(grid["X"], float), np.asarray(grid["Y"], float))
    p = np.abs(np.asarray(field, np.complex128))**2
    return float(np.sum(p[R <= TIP_FOOTPRINT_M]) / max(float(np.sum(p)), EPS))


def _onaxis(xz: np.ndarray, coord: np.ndarray) -> np.ndarray:
    i0 = int(np.argmin(np.abs(np.asarray(coord, float))))
    return np.asarray(xz, float)[:, i0]


def _centre_to_peak(xy: np.ndarray, grid) -> float:
    x = np.asarray(grid["x"], float)
    i0 = int(np.argmin(np.abs(x)))
    arr = np.asarray(xy, float)
    return float(arr[i0, i0] / max(float(np.max(arr)), EPS))


def _rms_ratio(round_axis: np.ndarray, sharp_axis: np.ndarray, z: np.ndarray, z0: float = 45e-3, z1: float = 115e-3) -> float:
    keep = (z >= z0) & (z <= z1)
    ratio = np.asarray(round_axis, float)[keep] / np.maximum(np.asarray(sharp_axis, float)[keep], EPS)
    return float(np.sqrt(np.mean((ratio - 1.0)**2)))


def _fields(grid_n: int):
    hw = canonical_hardware_manifest()
    wavelength = float(hardware_value(hw, "wavelength_m"))
    gamma = math.radians(float(hardware_value(hw, "axicon_base_angle_deg")))
    n_ax = float(hardware_value(hw, "axicon_refractive_index"))
    n_ext = float(hardware_value(hw, "axicon_external_medium_index"))
    route = build_system_route("B0", grid_n=grid_n)
    grid = dict(route["grid"])
    b0 = np.asarray(route["field_on_axicon_plane"], np.complex128)
    sharp_t = base._axicon_transmission(grid, wavelength, gamma, n_ax, n_ext, 0.0)
    round_t = _localized_round_transmission(grid, sharp_t, wavelength, gamma, n_ax, n_ext)
    return wavelength, grid, b0, sharp_t, round_t


def select_clear_radius(grid_n: int) -> dict:
    wavelength, grid, b0, sharp_t, round_t = _fields(grid_n)
    coord = np.asarray(suite.TIP_COORD_M, float)
    z = np.asarray(suite.Z_VALUES_M, float)

    b0_s, _ = base._xz_from_post_axicon(b0*sharp_t, grid, wavelength, coord, "localized-round-b0-sharp")
    b0_r, _ = base._xz_from_post_axicon(b0*round_t, grid, wavelength, coord, "localized-round-b0-round")
    b0_sa = _onaxis(b0_s, coord)
    b0_ra = _onaxis(b0_r, coord)
    b0_rms = _rms_ratio(b0_ra, b0_sa, z)
    developed = (z >= 60e-3) & (z <= 115e-3)
    b0_mean = float(np.mean(b0_sa[developed]))

    rows = []
    for clear_um in CANDIDATE_CLEAR_RADII_UM:
        hollow = _hollow_l0(b0, grid, clear_um*1e-6)
        hs, _ = base._xz_from_post_axicon(hollow*sharp_t, grid, wavelength, coord, f"localized-round-{clear_um}-sharp")
        hr, _ = base._xz_from_post_axicon(hollow*round_t, grid, wavelength, coord, f"localized-round-{clear_um}-round")
        hsa = _onaxis(hs, coord)
        hra = _onaxis(hr, coord)
        xy = np.abs(base.native_field_at_z(base._prop(hollow*round_t, grid, wavelength, Z_REF_M), Z_REF_M))**2 if False else base._xy_from_post_axicon(hollow*round_t, grid, wavelength)
        # _xy_from_post_axicon is fixed at the repository's 60 mm reference.  We
        # use it only as a bright-centre sanity gate during the low-res sweep;
        # final XY is evaluated explicitly at 80 mm below.
        rms = _rms_ratio(hra, hsa, z)
        mean_sharp = float(np.mean(hsa[developed])) / max(b0_mean, EPS)
        rows.append({
            "clear_radius_um": int(clear_um),
            "tip_fraction": _fraction_inside(hollow, grid),
            "rounded_sharp_onaxis_rms": rms,
            "rms_vs_ordinary_B0": rms / max(b0_rms, EPS),
            "mean_sharp_onaxis_vs_B0": mean_sharp,
            "z60_centre_to_peak": _centre_to_peak(xy, grid),
        })

    admissible = [r for r in rows if (
        r["tip_fraction"] <= 1e-4
        and r["mean_sharp_onaxis_vs_B0"] >= 0.25
        and r["z60_centre_to_peak"] >= 0.70
    )]
    selected = min(admissible if admissible else rows, key=lambda r: (r["rms_vs_ordinary_B0"], r["clear_radius_um"]))
    print(f"LOCALIZED_ROUND_B0_RMS={b0_rms:.8f}")
    for row in rows:
        print(row)
    print("SELECTED", selected)
    return {"selected": selected, "ordinary_b0_rms": b0_rms, "rows": rows}


def _xy_at_z(post: np.ndarray, grid, wavelength: float, z_m: float) -> np.ndarray:
    prop = base._prop(post, grid, wavelength, z_m)
    from vbb_study.digital_twin.vortex_continuous_propagation import native_field_at_z
    return np.abs(native_field_at_z(prop, float(z_m)))**2


def render(out: Path, sweep_grid_n: int, render_grid_n: int) -> None:
    sel = select_clear_radius(sweep_grid_n)
    selected_um = int(sel["selected"]["clear_radius_um"])
    wavelength, grid, b0, sharp_t, round_t = _fields(render_grid_n)
    hollow = _hollow_l0(b0, grid, selected_um*1e-6)
    coord = np.asarray(suite.TIP_COORD_M, float)
    z = np.asarray(suite.Z_VALUES_M, float)

    cases = []
    for label, incident in (("ordinary B0", b0), (f"annular ell=0\n{selected_um} um clear radius", hollow)):
        xz_s, _ = base._xz_from_post_axicon(incident*sharp_t, grid, wavelength, coord, f"localized-final-{label}-sharp")
        xz_r, _ = base._xz_from_post_axicon(incident*round_t, grid, wavelength, coord, f"localized-final-{label}-round")
        xy_r = _xy_at_z(incident*round_t, grid, wavelength, Z_REF_M)
        xy_s = _xy_at_z(incident*sharp_t, grid, wavelength, Z_REF_M)
        axis_s = _onaxis(xz_s, coord)
        axis_r = _onaxis(xz_r, coord)
        cases.append({
            "label": label,
            "incident": incident,
            "xy_r": xy_r,
            "xy_s": xy_s,
            "xz_r": xz_r,
            "xz_s": xz_s,
            "axis_s": axis_s,
            "axis_r": axis_r,
            "rms": _rms_ratio(axis_r, axis_s, z),
            "tip_fraction": _fraction_inside(incident, grid),
            "centre_ratio": _centre_to_peak(xy_r, grid),
        })

    improvement = cases[1]["rms"] / max(cases[0]["rms"], EPS)
    print(f"SELECTED_CLEAR_RADIUS_UM={selected_um}")
    print(f"B0_APEX_POWER_PERCENT={100*cases[0]['tip_fraction']:.6f}")
    print(f"ANNULAR_APEX_POWER_PERCENT={100*cases[1]['tip_fraction']:.6f}")
    print(f"B0_ROUNDED_SHARP_RMS={cases[0]['rms']:.8f}")
    print(f"ANNULAR_ROUNDED_SHARP_RMS={cases[1]['rms']:.8f}")
    print(f"ANNULAR_TO_B0_RMS_RATIO={improvement:.8f}")
    print(f"ANNULAR_Z80_CENTRE_TO_PEAK={cases[1]['centre_ratio']:.8f}")

    # XY: mechanism + final transverse output.
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 8.3), constrained_layout=True)
    style.style_fig(fig)
    inc_peak = max(float(np.max(np.abs(c["incident"])**2)) for c in cases)
    for col, case in enumerate(cases):
        inc = np.abs(case["incident"])**2
        crop, extent = base._fixed_crop(inc, grid, 0.82e-3)
        title = "ordinary B0 directly illuminates defect" if col == 0 else "annular ell=0 clears defect"
        style.draw_xy(axes[0,col], crop, extent, title, peak=inc_peak, show_y=(col==0))
        axes[0,col].add_patch(Circle((0,0), TIP_FOOTPRINT_M*1e3, fill=False, edgecolor="#39d6ad", lw=1.4, ls="--"))
        axes[0,col].text(0.03,0.04,f"power inside cap: {100*case['tip_fraction']:.3f}%", transform=axes[0,col].transAxes, color=style.TEXT, fontsize=8, bbox=dict(facecolor=style.FIG_BG, edgecolor=style.BORDER, alpha=0.88, pad=2.5))
    out_peak = max(float(np.max(c["xy_r"])) for c in cases)
    for col, case in enumerate(cases):
        crop, extent = base._fixed_crop(case["xy_r"], grid, 0.38e-3)
        style.draw_xy(axes[1,col], crop, extent, "output at z = 80 mm", peak=out_peak, show_y=(col==0))
        axes[1,col].text(0.03,0.04,f"centre / peak = {case['centre_ratio']:.3f}", transform=axes[1,col].transAxes, color=style.TEXT, fontsize=8, bbox=dict(facecolor=style.FIG_BG, edgecolor=style.BORDER, alpha=0.88, pad=2.5))
    fig.suptitle("Severe localized rounded-cap axicon defect: direct illumination vs tip avoidance", color=style.TEXT, fontsize=16)
    fig.text(0.5,-0.008,"Dashed circle = explicit 400 um defect footprint. Outside this footprint the axicon is exactly the sharp cone; annular input is non-vortex (ell=0).",ha="center",color=style.MUTED,fontsize=9)
    style.save(fig, out/"09_severe_tip_avoidance_XY.png")

    # XZ: only the two rounded-cap cases, shared scale.
    fig, axes = plt.subplots(1,2,figsize=(10.8,4.9),constrained_layout=True)
    style.style_fig(fig)
    xz_peak=max(float(np.max(c["xz_r"])) for c in cases)
    for col, case in enumerate(cases):
        style.draw_xz(axes[col],case["xz_r"],coord,z,peak=xz_peak,show_y=(col==0),z_ref_m=Z_REF_M)
        axes[col].set_title(case["label"],color=style.TEXT,fontsize=11)
        axes[col].text(0.03,0.04,f"rounded/sharp on-axis RMS: {100*case['rms']:.1f}%",transform=axes[col].transAxes,color=style.TEXT,fontsize=8,bbox=dict(facecolor=style.FIG_BG,edgecolor=style.BORDER,alpha=0.88,pad=2.5))
    fig.suptitle("XZ propagation through the same localized rounded-cap defect",color=style.TEXT,fontsize=15)
    style.save(fig,out/"09b_severe_tip_avoidance_XZ.png")

    # 1D: output lineout and direct rounded/sharp interference signature.
    fig,axes=plt.subplots(1,2,figsize=(12.0,4.8),constrained_layout=True)
    style.style_fig(fig)
    for ax in axes: base._style_line_axis(ax)
    shared=max(float(np.max(c["xy_r"])) for c in cases)
    x=np.asarray(grid["x"],float)
    i0=int(np.argmin(np.abs(x)))
    for case,colour in zip(cases,COLORS):
        line=np.asarray(case["xy_r"],float)[i0]
        keep=np.abs(x)<=0.42e-3
        axes[0].plot(x[keep]*1e3,line[keep]/shared,color=colour,lw=1.9,label=case["label"].replace("\n"," - "))
    axes[0].set_xlabel("x at fixed y = 0 (mm)")
    axes[0].set_ylabel("intensity / shared maximum")
    axes[0].set_title("Transverse intensity at z = 80 mm",color=style.TEXT,fontsize=12)
    for case,colour in zip(cases,COLORS):
        ratio=case["axis_r"]/np.maximum(case["axis_s"],EPS)
        axes[1].plot(z*1e3,ratio,color=colour,lw=1.9,label=case["label"].replace("\n"," - "))
    axes[1].axhline(1.0,color=style.MUTED,lw=0.9,ls="--",alpha=0.75)
    axes[1].set_xlabel("z from axicon (mm)")
    axes[1].set_ylabel("on-axis intensity: rounded / matched sharp")
    axes[1].set_title("Rounded-cap interference signature",color=style.TEXT,fontsize=12)
    for ax in axes:
        leg=ax.legend(frameon=False,fontsize=8)
        for t in leg.get_texts(): t.set_color(style.TEXT)
    fig.suptitle(f"Tip avoidance reduces rounded-cap modulation to {improvement:.2f}x the ordinary-B0 value",color=style.TEXT,fontsize=14)
    style.save(fig,out/"09c_severe_tip_avoidance_1D.png")


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-grid-n",type=int,default=1024)
    p.add_argument("--render-grid-n",type=int,default=2048)
    p.add_argument("--output-dir",type=Path,default=Path("outputs/figures/axicon_severe_tip"))
    a=p.parse_args()
    if a.sweep_grid_n<1024 or a.render_grid_n<1024: raise ValueError("grids must be >=1024")
    a.output_dir.mkdir(parents=True,exist_ok=True)
    render(a.output_dir,a.sweep_grid_n,a.render_grid_n)


if __name__=="__main__": main()
