"""Focused presentation evidence for axicon decentre and tip physics.

This script uses the current repository forward model and produces:
  04_V1_axicon_decentre_fixed_lab_thermal_tight.png
  04b_V1_axicon_decentre_1D_intensity.png
  05_V1_nonideal_tip_fixed_lab_thermal_tight.png
  05b_V1_nonideal_tip_1D_intensity.png
  09_tip_avoidance_three_way_audit.png
  09b_tip_avoidance_three_way_1D_intensity.png
  axicon_presentation_evidence_manifest.json

Conventions
-----------
* Decentre XY windows follow the z=60 mm intensity centroid only for framing;
  the displayed ticks remain absolute laboratory coordinates.  XZ maps remain
  fixed-laboratory y=0 with no per-z recentering.
* Rounded-tip comparisons use the sharp-tip case as the shared intensity
  reference.
* Tip-avoidance audit compares (i) centred beam through a 200 um rounded-tip
  axicon, (ii) the same incident field propagated with no axicon as a straight-
  through control, and (iii) a same-total-power vortex target whose dark core
  exceeds the 200 um rounded apex before the same rounded-tip axicon.
* The wide-core vortex is an axicon-plane target field used to audit the optical
  idea.  It is not yet claimed as a calibrated phase-only SLM implementation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
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
DECENTRE_LABELS = ("-500 µm", "aligned", "+500 µm")
TIP_RADII_M = (0.0, 200e-6, 800e-6)
TIP_LABELS = ("ideal sharp tip", "200 µm radial rounding", "800 µm radial rounding")
DECENTRE_XY_HALF_M = 0.24e-3
TIP_XY_HALF_M = 0.24e-3
TIP_RADIUS_M = 200e-6
VORTEX_DARK_CORE_RADIUS_M = 260e-6
LINE_COLORS = ("#fff176", "#ff7600", "#ff3b30")


def _style_line_axis(ax: plt.Axes) -> None:
    style.style_ax(ax)
    ax.grid(alpha=0.13, linewidth=0.55)


def _centroid(intensity: np.ndarray, grid: Mapping[str, Any]) -> tuple[float, float]:
    p = np.maximum(np.asarray(intensity, float), 0.0)
    total = float(np.sum(p))
    X = np.asarray(grid["X"], float)
    Y = np.asarray(grid["Y"], float)
    return float(np.sum(p * X) / max(total, EPS)), float(np.sum(p * Y) / max(total, EPS))


def _centred_crop(intensity: np.ndarray, grid: Mapping[str, Any], cx: float, cy: float, halfwidth: float):
    x = np.asarray(grid["x"], float)
    ix = np.flatnonzero(np.abs(x - float(cx)) <= float(halfwidth))
    iy = np.flatnonzero(np.abs(x - float(cy)) <= float(halfwidth))
    if ix.size < 70 or iy.size < 70:
        raise RuntimeError(f"under-sampled XY crop: {ix.size} x {iy.size}")
    return np.asarray(intensity)[np.ix_(iy, ix)], [x[ix[0]]*1e3, x[ix[-1]]*1e3, x[iy[0]]*1e3, x[iy[-1]]*1e3]


def _fixed_crop(intensity: np.ndarray, grid: Mapping[str, Any], halfwidth: float):
    x = np.asarray(grid["x"], float)
    ids = np.flatnonzero(np.abs(x) <= float(halfwidth))
    if ids.size < 70:
        raise RuntimeError(f"under-sampled XY crop: {ids.size}")
    return np.asarray(intensity)[np.ix_(ids, ids)], [x[ids[0]]*1e3, x[ids[-1]]*1e3, x[ids[0]]*1e3, x[ids[-1]]*1e3]


def _lineout_x(intensity: np.ndarray, grid: Mapping[str, Any], y_m: float = 0.0):
    x = np.asarray(grid["x"], float)
    iy = int(np.argmin(np.abs(x - float(y_m))))
    return x, np.asarray(intensity, float)[iy]


def _propagator(field: np.ndarray, grid: Mapping[str, Any], wavelength_m: float, zmax: float | None = None):
    return build_fixed_support_spectrum(
        np.asarray(field, np.complex128), dict(grid), wavelength_m=float(wavelength_m),
        z_max_m=float(suite.Z_VALUES_M[-1] if zmax is None else zmax),
        minimum_retained_spectral_power=0.995,
    )


def _xy_from_field(field: np.ndarray, grid: Mapping[str, Any], wavelength_m: float):
    return np.abs(native_field_at_z(_propagator(field, grid, wavelength_m, Z_REF_M), Z_REF_M))**2


def _xz_from_field(field: np.ndarray, grid: Mapping[str, Any], wavelength_m: float, coord_m: np.ndarray, label: str):
    prop = _propagator(field, grid, wavelength_m)
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


def _rounded_tip_error(radius_m: float, gamma: float):
    if radius_m == 0.0:
        return AxiconError(tip_model="sharp")
    return AxiconError(tip_model="hyperboloidal_round", rounding_parameter_m=float(radius_m)*math.tan(float(gamma)))


def build_decentre_2d(out: Path, grid_n: int):
    cases = []
    for dec, label in zip(DECENTRES_M, DECENTRE_LABELS):
        route = build_system_route("V1", grid_n=grid_n, config=SystemErrorConfig(axicon=AxiconError(decentre_m=(dec, 0.0))))
        field, _ = core._xy_at_z(route)
        intensity = np.abs(field)**2
        cx, cy = _centroid(intensity, route["grid"])
        mapped, prop = core._longitudinal(route, suite.DECENTRE_COORD_M, f"focused-decentre-{dec:g}")
        cases.append(dict(dec=dec, label=label, route=route, intensity=intensity, cx=cx, cy=cy, xz=np.asarray(mapped.xz_intensity,float), retained=float(prop.retained_spectral_power_fraction)))

    ref_xy = float(np.max(cases[1]["intensity"]))
    ref_xz = float(np.max(cases[1]["xz"]))
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.6), constrained_layout=True)
    style.style_fig(fig)
    for col, c in enumerate(cases):
        crop, extent = _centred_crop(c["intensity"], c["route"]["grid"], c["cx"], c["cy"], DECENTRE_XY_HALF_M)
        style.draw_xy(axes[0,col], crop, extent, title=c["label"], peak=ref_xy, show_y=(col==0))
        axes[0,col].scatter([c["cx"]*1e3], [c["cy"]*1e3], s=24, facecolors="none", edgecolors="white", linewidths=0.8)
        style.draw_xz(axes[1,col], c["xz"], suite.DECENTRE_COORD_M, suite.Z_VALUES_M, peak=ref_xz, show_y=(col==0), z_ref_m=Z_REF_M)
    fig.suptitle("V1 axicon lateral decentre — tight XY framing, fixed-lab XZ", color=style.TEXT, fontsize=16, y=1.025)
    fig.text(0.5,-0.012,"XY crop follows each z=60 mm centroid for visibility; axis values remain absolute laboratory coordinates. XZ stays fixed y=0.",ha="center",color=style.MUTED,fontsize=9)
    path = out/"04_V1_axicon_decentre_fixed_lab_thermal_tight.png"
    style.save(fig,path)
    return path, cases


def build_decentre_1d(out: Path, cases):
    ref = max(float(np.max(_lineout_x(cases[1]["intensity"], cases[1]["route"]["grid"], cases[1]["cy"])[1])), EPS)
    fig, ax = plt.subplots(figsize=(9.6,4.8), constrained_layout=True)
    style.style_fig(fig); _style_line_axis(ax)
    rows=[]
    for c, colour in zip(cases, LINE_COLORS):
        x,line = _lineout_x(c["intensity"], c["route"]["grid"], c["cy"])
        keep = np.abs(x-c["cx"]) <= DECENTRE_XY_HALF_M
        ax.plot(x[keep]*1e3, line[keep]/ref, color=colour, lw=1.8, label=c["label"])
        ax.axvline(c["cx"]*1e3, color=colour, alpha=.3, lw=.8, ls="--")
        rows.append(dict(label=c["label"], centroid_x_m=c["cx"], peak_ratio=float(np.max(line)/ref)))
    ax.set_xlabel("laboratory x (mm)"); ax.set_ylabel("intensity / aligned peak")
    ax.set_title("V1 axicon decentre — 1D transverse intensity at z = 60 mm", color=style.TEXT, fontsize=13)
    leg=ax.legend(frameon=False,fontsize=9)
    for t in leg.get_texts(): t.set_color(style.TEXT)
    path=out/"04b_V1_axicon_decentre_1D_intensity.png"; style.save(fig,path)
    return path, rows


def build_tip_2d(out: Path, grid_n: int):
    manifest=canonical_hardware_manifest(); gamma=math.radians(float(hardware_value(manifest,"axicon_base_angle_deg")))
    cases=[]
    for radius,label in zip(TIP_RADII_M,TIP_LABELS):
        route=build_system_route("V1",grid_n=grid_n,config=SystemErrorConfig(axicon=_rounded_tip_error(radius,gamma)))
        field,_=core._xy_at_z(route); intensity=np.abs(field)**2
        mapped,prop=core._longitudinal(route,suite.TIP_COORD_M,f"focused-tip-{radius:g}")
        cases.append(dict(radius=radius,label=label,route=route,intensity=intensity,xz=np.asarray(mapped.xz_intensity,float),retained=float(prop.retained_spectral_power_fraction)))
    ref_xy=float(np.max(cases[0]["intensity"])); ref_xz=float(np.max(cases[0]["xz"]))
    fig,axes=plt.subplots(2,3,figsize=(11.8,6.6),constrained_layout=True); style.style_fig(fig)
    for col,c in enumerate(cases):
        crop,extent=_fixed_crop(c["intensity"],c["route"]["grid"],TIP_XY_HALF_M)
        style.draw_xy(axes[0,col],crop,extent,title=c["label"],peak=ref_xy,show_y=(col==0))
        style.draw_xz(axes[1,col],c["xz"],suite.TIP_COORD_M,suite.Z_VALUES_M,peak=ref_xz,show_y=(col==0),z_ref_m=Z_REF_M)
    fig.suptitle("V1 non-ideal axicon tip — common sharp-tip reference",color=style.TEXT,fontsize=16,y=1.025)
    path=out/"05_V1_nonideal_tip_fixed_lab_thermal_tight.png"; style.save(fig,path)
    return path,cases


def build_tip_1d(out: Path, cases):
    x0,l0=_lineout_x(cases[0]["intensity"],cases[0]["route"]["grid"],0.0); ref=max(float(np.max(l0)),EPS)
    fig,ax=plt.subplots(figsize=(9.6,4.8),constrained_layout=True); style.style_fig(fig); _style_line_axis(ax)
    rows=[]
    for c,colour in zip(cases,LINE_COLORS):
        x,line=_lineout_x(c["intensity"],c["route"]["grid"],0.0); keep=np.abs(x)<=TIP_XY_HALF_M
        ax.plot(x[keep]*1e3,line[keep]/ref,color=colour,lw=1.8,label=c["label"])
        rows.append(dict(label=c["label"],peak_ratio=float(np.max(line)/ref)))
    ax.set_xlabel("x at y = 0 (mm)"); ax.set_ylabel("intensity / sharp-tip peak")
    ax.set_title("V1 rounded-tip axicon — 1D transverse intensity at z = 60 mm",color=style.TEXT,fontsize=13)
    leg=ax.legend(frameon=False,fontsize=9)
    for t in leg.get_texts(): t.set_color(style.TEXT)
    path=out/"05b_V1_nonideal_tip_1D_intensity.png"; style.save(fig,path)
    return path,rows


def _wide_vortex_target(field_ax: np.ndarray, grid: Mapping[str, Any], dark_core_radius_m: float):
    X=np.asarray(grid["X"],float); Y=np.asarray(grid["Y"],float); R=np.hypot(X,Y); phi=np.arctan2(Y,X)
    # Smooth l=1 vortex target.  The amplitude reaches 50% at approximately
    # dark_core_radius_m and approaches the original envelope outside it.
    r0=max(float(dark_core_radius_m),EPS)
    amp=1.0-np.exp(-(R/r0)**4)
    target=np.abs(field_ax)*amp*np.exp(1j*(np.angle(field_ax)+phi))
    p0=float(np.sum(np.abs(field_ax)**2)); p1=float(np.sum(np.abs(target)**2))
    if p1>EPS: target*=math.sqrt(p0/p1)  # same incident power, not same output peak
    return target


def build_tip_avoidance(out: Path, grid_n: int):
    # Use the B0 route to define one common centred field arriving at the axicon.
    # The wide-vortex case then modifies that same axicon-plane field into an l=1
    # target with a dark core wider than the 200 um rounded apex.
    base=build_system_route("B0",grid_n=grid_n)
    grid=dict(base["grid"]); field_ax=np.asarray(base["field_on_axicon_plane"],np.complex128)
    manifest=canonical_hardware_manifest(); wavelength=float(hardware_value(manifest,"wavelength_m")); gamma=math.radians(float(hardware_value(manifest,"axicon_base_angle_deg"))); n_ax=float(hardware_value(manifest,"axicon_refractive_index")); n_ext=float(hardware_value(manifest,"axicon_external_medium_index"))
    axicon_t,_=physical_axicon_on_own_plane(grid,wavelength_m=wavelength,base_angle_rad=gamma,refractive_index=n_ax,external_index=n_ext,error=_rounded_tip_error(TIP_RADIUS_M,gamma))
    vortex_field=_wide_vortex_target(field_ax,grid,VORTEX_DARK_CORE_RADIUS_M)
    cases=[
        dict(label="centred beam → 200 µm rounded tip", field=field_ax*axicon_t, incident=field_ax),
        dict(label="straight-through control (no axicon)", field=field_ax, incident=field_ax),
        dict(label="wide-core vortex → same rounded tip", field=vortex_field*axicon_t, incident=vortex_field),
    ]
    R=np.hypot(np.asarray(grid["X"],float),np.asarray(grid["Y"],float))
    for i,c in enumerate(cases):
        c["xy"]=_xy_from_field(c["field"],grid,wavelength)
        c["xz"],c["retained"]=_xz_from_field(c["field"],grid,wavelength,suite.TIP_COORD_M,f"tip-avoidance-{i}")
        pin=float(np.sum(np.abs(c["incident"])**2)); c["fraction_incident_power_inside_tip"]=float(np.sum(np.abs(c["incident"][R<=TIP_RADIUS_M])**2)/max(pin,EPS))
    ref_xy=max(float(np.max(cases[0]["xy"])),EPS); ref_xz=max(float(np.max(cases[0]["xz"])),EPS)
    fig,axes=plt.subplots(2,3,figsize=(12.2,6.8),constrained_layout=True); style.style_fig(fig)
    for col,c in enumerate(cases):
        crop,extent=_fixed_crop(c["xy"],grid,0.34e-3)
        style.draw_xy(axes[0,col],crop,extent,title=c["label"],peak=ref_xy,show_y=(col==0))
        axes[0,col].text(.03,.04,f"incident power within 200 µm: {100*c['fraction_incident_power_inside_tip']:.2f}%",transform=axes[0,col].transAxes,color=style.TEXT,fontsize=7.8,bbox=dict(facecolor=style.FIG_BG,edgecolor=style.BORDER,alpha=.86,pad=3))
        style.draw_xz(axes[1,col],c["xz"],suite.TIP_COORD_M,suite.Z_VALUES_M,peak=ref_xz,show_y=(col==0),z_ref_m=Z_REF_M)
    fig.suptitle("Rounded-tip axicon avoidance audit — straight-through and wide-vortex controls",color=style.TEXT,fontsize=16,y=1.025)
    fig.text(.5,-.012,"All three start from the same pre-axicon power. The wide-vortex target is power-normalised before the axicon; output peaks are not renormalised.",ha="center",color="#f2c14e",fontsize=9)
    p2d=out/"09_tip_avoidance_three_way_audit.png"; style.save(fig,p2d)

    fig1,ax=plt.subplots(figsize=(9.8,4.9),constrained_layout=True); style.style_fig(fig1); _style_line_axis(ax)
    rows=[]
    for c,colour in zip(cases,LINE_COLORS):
        x,line=_lineout_x(c["xy"],grid,0.0); keep=np.abs(x)<=0.34e-3
        ax.plot(x[keep]*1e3,line[keep]/ref_xy,color=colour,lw=1.8,label=c["label"])
        rows.append(dict(label=c["label"],peak_ratio_to_rounded_tip=float(np.max(line)/ref_xy),fraction_incident_power_inside_tip=c["fraction_incident_power_inside_tip"]))
    ax.set_xlabel("x at y = 0 (mm)"); ax.set_ylabel("intensity / rounded-tip peak")
    ax.set_title("Tip avoidance audit — 1D transverse intensity at z = 60 mm",color=style.TEXT,fontsize=13)
    leg=ax.legend(frameon=False,fontsize=8.5)
    for t in leg.get_texts(): t.set_color(style.TEXT)
    p1d=out/"09b_tip_avoidance_three_way_1D_intensity.png"; style.save(fig1,p1d)
    return p2d,p1d,rows


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--grid-n",type=int,default=2048); ap.add_argument("--output-dir",type=Path,default=Path("outputs/figures/axicon_presentation_evidence")); args=ap.parse_args()
    if args.grid_n<2048: raise ValueError("focused axicon presentation evidence requires grid_n >= 2048")
    args.output_dir.mkdir(parents=True,exist_ok=True); suite._patch_core_renderer()
    p04,dec_cases=build_decentre_2d(args.output_dir,args.grid_n); p04b,dec1=build_decentre_1d(args.output_dir,dec_cases)
    p05,tip_cases=build_tip_2d(args.output_dir,args.grid_n); p05b,tip1=build_tip_1d(args.output_dir,tip_cases)
    p09,p09b,avoid=build_tip_avoidance(args.output_dir,args.grid_n)
    payload={
        "outcome":"AXICON-PRESENTATION-EVIDENCE-AUDIT-V1",
        "grid_n":args.grid_n,
        "z_ref_m":Z_REF_M,
        "figures":[str(p) for p in (p04,p04b,p05,p05b,p09,p09b)],
        "decentre_1d":dec1,
        "rounded_tip_1d":tip1,
        "tip_avoidance":avoid,
        "tip_avoidance_dark_core_radius_m":VORTEX_DARK_CORE_RADIUS_M,
        "tip_radius_m":TIP_RADIUS_M,
        "claim_boundary":"wide-core vortex is an axicon-plane target field for optical auditing; calibrated phase-only SLM generation is not claimed",
    }
    (args.output_dir/"axicon_presentation_evidence_manifest.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(payload,indent=2))


if __name__=="__main__": main()
