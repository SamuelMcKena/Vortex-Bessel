"""Physics-safe q=20 presentation rebuild from the complete BeamGage BMG stack.

Authoritative outputs:
  01 measured all-plane BMG contact sheet
  02 measured XZ
  03 measured YZ
  04 measured XZ/YZ combined
  05 retrieved *residual* transverse phase (q*theta excluded)
  06 single-transverse-phase forward-model diagnostic

The script refuses incomplete acquisitions.  For the current dataset it requires
18 z planes x 4 repeats = 72 BMG files.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from modal_vortex_bessel import load_first_scan, read_bmg
from q20_phase_physics import (
    assemble_transverse_residual_phase,
    central_band_sections,
    cone_geometry,
)
from single_transverse_phase_forward_test import run_single_transverse_phase_test

THERMAL = "inferno"
PHASE = "twilight_shifted"
EPS = 1e-12


def _z_index(path: Path) -> int:
    m = re.match(r"z(\d+)_", path.stem.lower())
    if not m:
        raise ValueError(f"Cannot parse z index from {path.name}")
    return int(m.group(1))


def _bmg_inventory(data_dir: Path) -> dict[int, list[Path]]:
    groups: dict[int, list[Path]] = {}
    for p in sorted(Path(data_dir).glob("z*_*.bmg")):
        groups.setdefault(_z_index(p), []).append(p)
    return dict(sorted(groups.items()))


def _normalise_planes(stack: np.ndarray) -> np.ndarray:
    a = np.asarray(stack, float)
    p = np.maximum(a.reshape(a.shape[0], -1).max(axis=1), EPS)
    return a / p[:, None, None]


def _tight(stack: np.ndarray, pixel_pitch_m: float, limit_um: float):
    a = np.asarray(stack)
    cy, cx = a.shape[1] // 2, a.shape[2] // 2
    h = max(8, int(round(limit_um * 1e-6 / pixel_pitch_m)))
    cut = a[:, cy-h:cy+h+1, cx-h:cx+h+1]
    axis_um = (np.arange(cut.shape[2]) - (cut.shape[2]-1)/2) * pixel_pitch_m * 1e6
    return cut, axis_um


def _save_measured(stack, z_mm, output_dir, pixel_pitch_m, view_limit_um, repeats, saturated):
    shape_stack, axis_um = _tight(_normalise_planes(stack), pixel_pitch_m, view_limit_um)
    n = len(z_mm)
    extent_xy = [axis_um[0], axis_um[-1], axis_um[0], axis_um[-1]]

    fig, axes = plt.subplots(3, 6, figsize=(18, 9.4), constrained_layout=True)
    for iz, ax in enumerate(axes.ravel()):
        im = ax.imshow(shape_stack[iz], origin="lower", cmap=THERMAL,
                       vmin=0, vmax=1, extent=extent_xy, interpolation="nearest")
        ax.set_aspect("equal")
        ax.set(title=f"z = {z_mm[iz]:g} mm", xlabel="x (µm)", ylabel="y (µm)")
        note = f"{repeats[iz]} BMG repeats"
        if iz in saturated:
            note += "\nADC ≥ 98%"
        ax.text(.02, .02, note, transform=ax.transAxes, color="white", fontsize=7.5,
                bbox=dict(facecolor="black", alpha=.6, edgecolor="none", pad=2))
    fig.colorbar(im, ax=axes, label="plane-normalized measured intensity", shrink=.82)
    fig.suptitle("Measured q=20 vortex–Bessel evolution — complete 72-frame BMG acquisition",
                 fontsize=15)
    p1 = output_dir / "01_measured_q20_BMG_stack_all_planes.png"
    fig.savefig(p1, dpi=400, bbox_inches="tight")
    plt.close(fig)

    xz, yz = central_band_sections(shape_stack, half_width_px=2)
    paths = {"stack": str(p1)}
    for arr, coord, name in ((xz, "x", "02_measured_q20_XZ_all_planes.png"),
                             (yz, "y", "03_measured_q20_YZ_all_planes.png")):
        fig, ax = plt.subplots(figsize=(10.5, 6.8), constrained_layout=True)
        im = ax.imshow(arr, origin="lower", aspect="auto", cmap=THERMAL, vmin=0, vmax=1,
                       extent=[axis_um[0], axis_um[-1], z_mm[0], z_mm[-1]],
                       interpolation="nearest")
        ax.axvline(0, color="cyan", lw=.6, alpha=.65)
        ax.set(title=f"Measured q=20 {coord}–z evolution — all {n} BMG planes",
               xlabel=f"signed {coord} (µm)", ylabel="relative z (mm)")
        fig.colorbar(im, ax=ax, label="plane-normalized measured intensity")
        p = output_dir / name
        fig.savefig(p, dpi=450, bbox_inches="tight")
        plt.close(fig)
        paths[coord + "z"] = str(p)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.7), constrained_layout=True, sharey=True)
    for ax, arr, title, coord in zip(axes, (xz, yz), ("x–z at y≈0", "y–z at x≈0"), ("x", "y")):
        im = ax.imshow(arr, origin="lower", aspect="auto", cmap=THERMAL, vmin=0, vmax=1,
                       extent=[axis_um[0], axis_um[-1], z_mm[0], z_mm[-1]],
                       interpolation="nearest")
        ax.axvline(0, color="cyan", lw=.6, alpha=.65)
        ax.set(title=title, xlabel=f"signed {coord} (µm)", ylabel="relative z (mm)")
    fig.colorbar(im, ax=axes, label="plane-normalized measured intensity", shrink=.88)
    fig.suptitle("Measured q=20 longitudinal evolution from the complete BMG stack", fontsize=15)
    p4 = output_dir / "04_measured_q20_XZ_YZ_combined_all_planes.png"
    fig.savefig(p4, dpi=450, bbox_inches="tight")
    plt.close(fig)
    paths["xz_yz"] = str(p4)
    return paths


def _kr_from_modal_summary(modal_dir: Path, fallback=489678.1594027835):
    p = modal_dir / "summary.json"
    if p.exists():
        try:
            value = json.loads(p.read_text(encoding="utf-8")).get("kr_rad_per_um")
            if value is not None:
                return float(value) * 1e6
        except Exception:
            pass
    return float(fallback)


def _save_phase(modal_dir, z_mm, output_dir, wavelength_m, kr_m_inv,
                z_at_relative_zero_from_axicon_m):
    phase_rows = np.load(modal_dir / "annular_aberration_phase.npy")
    if phase_rows.shape[0] != len(z_mm):
        raise ValueError("annular_aberration_phase.npy does not match the measured z stack")
    r = assemble_transverse_residual_phase(
        phase_rows, z_mm*1e-3, wavelength_m=wavelength_m,
        k_perp_m_inv=kr_m_inv,
        z_at_relative_zero_from_axicon_m=z_at_relative_zero_from_axicon_m,
        grid_size=600)
    fixed = np.asarray(r["gauge_fixed_phase_rows_rad"])
    residual = np.asarray(r["residual_phase_rad"])
    correction = np.asarray(r["conjugate_correction_phase_rad"])
    x_mm = np.asarray(r["x_m"]) * 1e3

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), constrained_layout=True)
    im = axes[0].imshow(fixed, origin="lower", aspect="auto", cmap=PHASE,
                        vmin=-np.pi, vmax=np.pi,
                        extent=[0, 360, z_mm[0], z_mm[-1]])
    axes[0].set(title="Residual phase on sampled annuli",
                xlabel="azimuth θ (deg)", ylabel="relative z (mm)")
    fig.colorbar(im, ax=axes[0], label="wrapped residual phase (rad)", shrink=.84)
    for ax, arr, title in ((axes[1], residual, "Assembled transverse residual ψ(ρ,θ)"),
                           (axes[2], correction, "Conjugate transverse correction −ψ(ρ,θ)")):
        im = ax.imshow(arr, origin="lower", cmap=PHASE, vmin=-np.pi, vmax=np.pi,
                       extent=[x_mm[0], x_mm[-1], x_mm[0], x_mm[-1]])
        ax.set_aspect("equal")
        ax.set(title=title, xlabel="input-plane x (mm)", ylabel="input-plane y (mm)")
        fig.colorbar(im, ax=ax, label="wrapped phase (rad)", shrink=.84)
    radius = "absolute annulus radius" if r["absolute_radius_calibrated"] else "relative annulus radius only"
    fig.suptitle(
        "q=20 residual-phase reconstruction — programmed vortex removed\n"
        f"z supplies annular/radial diversity through ρz=z tanα; {radius}; radial piston not claimed",
        fontsize=14)
    p = output_dir / "05_retrieved_residual_phase_physics.png"
    fig.savefig(p, dpi=420, bbox_inches="tight")
    plt.close(fig)
    return r, p


def _save_forward_copy(forward_dir: Path, output_dir: Path):
    src = forward_dir / "single_transverse_phase_forward_model.png"
    if not src.exists():
        raise FileNotFoundError(src)
    d = np.load(forward_dir / "single_transverse_phase_stacks.npz")
    measured = d["measured"]; error = d["error_model"]; corrected = d["corrected_model"]
    z_mm = d["z_relative_mm"]; axis_um = d["x_um"]; d.close()
    stacks = (measured, error, corrected)
    titles = ("LAB MEASURED", "MODEL + RETRIEVED RESIDUAL", "MODEL AFTER CONJUGATE CORRECTION")
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9), constrained_layout=True, sharex=True, sharey=True)
    for col, (stack, title) in enumerate(zip(stacks, titles)):
        xz, yz = central_band_sections(stack, half_width_px=2)
        for row, (arr, section) in enumerate(((xz, "x–z"), (yz, "y–z"))):
            im = axes[row, col].imshow(arr, origin="lower", aspect="auto", cmap=THERMAL,
                                       vmin=0, vmax=1,
                                       extent=[axis_um[0], axis_um[-1], z_mm[0], z_mm[-1]],
                                       interpolation="nearest")
            axes[row, col].axvline(0, color="cyan", lw=.55, alpha=.6)
            axes[row, col].set(title=f"{section} | {title}",
                               xlabel="signed transverse coordinate (µm)", ylabel="relative z (mm)")
    fig.colorbar(im, ax=axes, label="plane-normalized intensity", shrink=.82)
    fig.suptitle("Single-transverse-phase physics check — all z evolution comes from propagation", fontsize=14)
    dst = output_dir / "06_single_transverse_phase_forward_model.png"
    fig.savefig(dst, dpi=420, bbox_inches="tight")
    plt.close(fig)
    return dst


def rebuild(data_dir: Path, *, modal_dir: Path, forward_dir: Path, output_dir: Path,
            z_start_mm=-17.0, z_step_mm=1.0, expected_planes=18, expected_repeats=4,
            wavelength_m=1030e-9, pixel_pitch_m=5.5e-6, q=20,
            view_limit_um=180.0, z_at_relative_zero_from_axicon_m=None):
    data_dir = Path(data_dir); modal_dir = Path(modal_dir)
    forward_dir = Path(forward_dir); output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = _bmg_inventory(data_dir)
    if len(groups) != expected_planes:
        raise RuntimeError(f"Expected {expected_planes} z planes, found {len(groups)}")
    wrong = {z: len(v) for z, v in groups.items() if len(v) != expected_repeats}
    if wrong:
        raise RuntimeError(f"Expected {expected_repeats} BMG repeats per plane; mismatches: {wrong}")
    if sum(map(len, groups.values())) != expected_planes * expected_repeats:
        raise RuntimeError("BMG acquisition is incomplete")

    stack = np.stack(load_first_scan(data_dir))
    z_mm = z_start_mm + np.arange(expected_planes) * z_step_mm
    saturated = set()
    max_adc_frac = 0.0
    for zi, paths in groups.items():
        for p in paths:
            frac = float(np.max(read_bmg(p))) / 4095.0
            max_adc_frac = max(max_adc_frac, frac)
            if frac >= .98:
                saturated.add(zi)
    repeats = {zi: len(paths) for zi, paths in groups.items()}
    measured = _save_measured(stack, z_mm, output_dir, pixel_pitch_m, view_limit_um,
                              repeats, saturated)

    phase_path = modal_dir / "annular_aberration_phase.npy"
    if not phase_path.exists():
        from q20_modal_analysis import run_modal_q20
        run_modal_q20(data_dir, modal_dir, pixel_pitch_m=pixel_pitch_m,
                      q=q, z_positions_mm=z_mm)
    kr = _kr_from_modal_summary(modal_dir)
    phase_info, phase_fig = _save_phase(
        modal_dir, z_mm, output_dir, wavelength_m, kr,
        z_at_relative_zero_from_axicon_m)

    if not (forward_dir / "single_transverse_phase_stacks.npz").exists():
        run_single_transverse_phase_test(
            data_dir, phase_path, forward_dir, z_relative_mm=z_mm,
            wavelength_m=wavelength_m, pixel_pitch_m=pixel_pitch_m, q=q,
            absolute_z_at_relative_zero_mm=(None if z_at_relative_zero_from_axicon_m is None
                                            else z_at_relative_zero_from_axicon_m*1e3))
    forward_fig = _save_forward_copy(forward_dir, output_dir)

    provenance = {
        "raw_bmg_files_used": 72,
        "z_planes_used": 18,
        "repeats_per_plane": 4,
        "z_mm": z_mm.tolist(),
        "effective_q": q,
        "wavelength_nm": wavelength_m*1e9,
        "camera_pixel_um": pixel_pitch_m*1e6,
        "kr_rad_per_um": kr*1e-6,
        "cone_alpha_deg": float(np.degrees(cone_geometry(wavelength_m, kr).alpha_rad)),
        "maximum_raw_adc_fraction": max_adc_frac,
        "near_saturation_warning": bool(max_adc_frac >= .98),
        "phase_interpretation": "single transverse residual reconstructed from z-sampled annuli; no longitudinal correction map",
        "programmed_qtheta_removed_from_residual": True,
        "radial_piston_recovered": False,
        "hardware_ready": False,
        "hardware_blocker": phase_info["hardware_blocker"],
        "measured_figures": measured,
        "phase_figure": str(phase_fig),
        "forward_model_figure": str(forward_fig),
    }
    (output_dir / "q20_presentation_rebuild_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def _args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path,
                   default=Path(os.environ.get("BESSEL_ZSCAN_DATA_DIR", here / "z-scan 2 1010")))
    p.add_argument("--modal-dir", type=Path,
                   default=here / "outputs" / "slm_closed_loop_alignment" / "modal_q20")
    p.add_argument("--forward-dir", type=Path,
                   default=here / "outputs" / "slm_closed_loop_alignment" / "modal_q20" /
                           "single_transverse_phase_forward_test")
    p.add_argument("--output-dir", type=Path,
                   default=here.parents[2] / "figures" / "experimental" /
                           "q20_aberration" / "presentation_rebuild")
    p.add_argument("--absolute-z-at-relative-zero-mm", type=float, default=None)
    return p.parse_args()


if __name__ == "__main__":
    a = _args()
    report = rebuild(
        a.data_dir, modal_dir=a.modal_dir, forward_dir=a.forward_dir,
        output_dir=a.output_dir,
        z_at_relative_zero_from_axicon_m=(None if a.absolute_z_at_relative_zero_mm is None
                                          else a.absolute_z_at_relative_zero_mm*1e-3))
    print(json.dumps(report, indent=2))
