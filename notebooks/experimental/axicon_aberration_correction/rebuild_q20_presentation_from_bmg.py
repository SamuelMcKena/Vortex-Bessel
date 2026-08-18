"""Rebuild the q=20 experimental presentation figures from the complete BMG stack.

This script replaces the misleading compact chat figures with provenance-explicit
outputs built from *all* raw BeamGage repeats in ``z-scan 2 1010``.

Outputs
-------
01_measured_q20_BMG_stack_all_planes.png
    Every measured z plane, each panel the registered mean of all repeats.
02_measured_q20_XZ_all_planes.png
03_measured_q20_YZ_all_planes.png
04_measured_q20_XZ_YZ_combined_all_planes.png
    Measured-only longitudinal morphology from the full z stack.
05_retrieved_residual_phase_physics.png
    The q=20 target phase is removed; z is used only as annular/radial diversity.
    No independent "longitudinal correction phase" is invented.
06_single_transverse_phase_forward_model.png
    Lab data vs a single input-plane residual-phase model vs the same model after
    conjugate correction, all propagated through one common forward model.

The final two figures are model diagnostics, not post-correction camera data.
No generated phase map is hardware-ready without SLM coordinate/LUT calibration
and a new measured validation z-scan.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from modal_vortex_bessel import read_bmg, preprocess, find_dark_core_center
from q20_phase_physics import (
    assemble_transverse_residual_phase,
    central_band_sections,
    cone_geometry,
)

EPS = 1.0e-12
THERMAL_CMAP = "inferno"
PHASE_CMAP = "twilight_shifted"


def _z_index(path: Path) -> int:
    match = re.match(r"z(\d+)_", path.stem.lower())
    if not match:
        raise ValueError(f"Cannot parse z index from {path.name}")
    return int(match.group(1))


def discover_bmg_groups(data_dir: Path) -> dict[int, list[Path]]:
    groups: dict[int, list[Path]] = {}
    for path in sorted(Path(data_dir).glob("z*_*.bmg")):
        groups.setdefault(_z_index(path), []).append(path)
    return dict(sorted(groups.items()))


def _crop_about(img: np.ndarray, cy: float, cx: float, size: int) -> np.ndarray:
    size = int(size)
    h = size // 2
    yi, xi = int(round(cy)), int(round(cx))
    y0 = max(0, min(img.shape[0] - size, yi - h))
    x0 = max(0, min(img.shape[1] - size, xi - h))
    return img[y0:y0 + size, x0:x0 + size]


def load_complete_registered_stack(
    data_dir: Path,
    *,
    roi_size: int = 768,
    adc_full_scale: float = 4095.0,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict[int, int]]:
    """Decode every BMG repeat once and return core-centred plane means.

    Repeat frames are registered within each z plane to the dark-core centre.
    The inter-plane beam-axis motion is retained separately in ``plane_centres``
    so that shape and pointing are not conflated.
    """
    groups = discover_bmg_groups(data_dir)
    if not groups:
        raise FileNotFoundError(f"No z*_*.bmg files found in {Path(data_dir).resolve()}")

    stack = []
    plane_centres = []
    qc_rows: list[dict] = []
    repeat_counts: dict[int, int] = {}

    target = np.array([(roi_size - 1) / 2.0, (roi_size - 1) / 2.0])
    for zi, paths in groups.items():
        repeat_counts[zi] = len(paths)
        crops = []
        centres_full = []
        for path in paths:
            raw = read_bmg(path)
            bg = float(np.percentile(raw, 35.0))
            proc = preprocess(raw)
            cy, cx, core_score = find_dark_core_center(proc)
            centres_full.append((cy, cx))
            crop = _crop_about(proc, cy, cx, roi_size)
            ccy, ccx, _ = find_dark_core_center(crop)
            shifted = ndimage.shift(
                crop,
                (target[0] - ccy, target[1] - ccx),
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            crops.append(shifted)
            raw_peak = float(np.max(raw))
            qc_rows.append({
                "z_index": int(zi),
                "file": path.name,
                "background_counts": bg,
                "raw_peak": raw_peak,
                "saturation_fraction": raw_peak / float(adc_full_scale),
                "core_score": float(core_score),
                "centre_y_px_full_sensor": float(cy),
                "centre_x_px_full_sensor": float(cx),
            })
        stack.append(np.mean(np.stack(crops), axis=0).astype(np.float32))
        plane_centres.append(np.median(np.asarray(centres_full), axis=0))

    return (np.stack(stack), np.asarray(plane_centres, dtype=float),
            qc_rows, repeat_counts)


def _normalise_planes(stack: np.ndarray) -> np.ndarray:
    a = np.asarray(stack, dtype=float)
    peak = np.maximum(a.reshape(a.shape[0], -1).max(axis=1), EPS)
    return a / peak[:, None, None]


def _tight_square(stack: np.ndarray, pixel_pitch_m: float, limit_um: float) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(stack)
    mid_y, mid_x = a.shape[1] // 2, a.shape[2] // 2
    half = max(8, int(round(limit_um * 1e-6 / pixel_pitch_m)))
    ys = slice(max(0, mid_y - half), min(a.shape[1], mid_y + half + 1))
    xs = slice(max(0, mid_x - half), min(a.shape[2], mid_x + half + 1))
    cut = a[:, ys, xs]
    axis_um = (np.arange(cut.shape[2]) - (cut.shape[2] - 1) / 2) * pixel_pitch_m * 1e6
    return cut, axis_um


def _save_measured_figures(
    stack: np.ndarray,
    z_mm: np.ndarray,
    output_dir: Path,
    *,
    pixel_pitch_m: float,
    view_limit_um: float,
    repeat_counts: dict[int, int],
    qc_rows: list[dict],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shape_stack, axis_um = _tight_square(_normalise_planes(stack), pixel_pitch_m, view_limit_um)
    n = len(z_mm)
    if n == 0:
        raise ValueError("empty z stack")

    ncols = 6
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.05*nrows),
                             constrained_layout=True, squeeze=False)
    extent = [axis_um[0], axis_um[-1], axis_um[0], axis_um[-1]]
    saturated_z = {
        int(row["z_index"]) for row in qc_rows if row["saturation_fraction"] >= 0.98
    }
    for iz, ax in enumerate(axes.ravel()):
        if iz >= n:
            ax.axis("off")
            continue
        im = ax.imshow(shape_stack[iz], origin="lower", cmap=THERMAL_CMAP,
                       vmin=0, vmax=1, extent=extent, interpolation="nearest")
        ax.set_aspect("equal")
        ax.set_title(f"z = {z_mm[iz]:g} mm", fontsize=10)
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        repeats = repeat_counts.get(iz, 0)
        label = f"{repeats}-frame registered mean"
        if iz in saturated_z:
            label += "\nADC ≥ 98%"
        ax.text(.02, .02, label, transform=ax.transAxes, color="white", fontsize=7.5,
                va="bottom", bbox=dict(facecolor="black", alpha=.58,
                                        edgecolor="none", pad=2))
    fig.colorbar(im, ax=axes, label="plane-normalized measured intensity", shrink=.82)
    fig.suptitle("Measured q=20 vortex–Bessel evolution — complete BeamGage BMG stack",
                 fontsize=16)
    p_stack = output_dir / "01_measured_q20_BMG_stack_all_planes.png"
    fig.savefig(p_stack, dpi=400, bbox_inches="tight")
    plt.close(fig)

    xz, yz = central_band_sections(shape_stack, half_width_px=2)
    files = {"stack": str(p_stack)}
    for arr, name, transverse in (
        (xz, "02_measured_q20_XZ_all_planes.png", "x"),
        (yz, "03_measured_q20_YZ_all_planes.png", "y"),
    ):
        fig, ax = plt.subplots(figsize=(10.5, 6.8), constrained_layout=True)
        shown = ax.imshow(arr, origin="lower", aspect="auto", cmap=THERMAL_CMAP,
                          vmin=0, vmax=1,
                          extent=[axis_um[0], axis_um[-1], z_mm[0], z_mm[-1]],
                          interpolation="nearest")
        ax.axvline(0, color="cyan", lw=.65, alpha=.65)
        ax.set(xlabel=f"signed {transverse} (µm)", ylabel="relative z (mm)",
               title=(f"Measured q=20 {transverse}–z evolution — all {n} BMG planes\n"
                      "core-centred morphology; ±2-pixel orthogonal band average"))
        fig.colorbar(shown, ax=ax, label="plane-normalized measured intensity")
        path = output_dir / name
        fig.savefig(path, dpi=450, bbox_inches="tight")
        plt.close(fig)
        files[transverse + "z"] = str(path)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.7), constrained_layout=True,
                             sharey=True)
    for ax, arr, title, coord in zip(
        axes, (xz, yz), ("x–z at y≈0", "y–z at x≈0"), ("x", "y")
    ):
        shown = ax.imshow(arr, origin="lower", aspect="auto", cmap=THERMAL_CMAP,
                          vmin=0, vmax=1,
                          extent=[axis_um[0], axis_um[-1], z_mm[0], z_mm[-1]],
                          interpolation="nearest")
        ax.axvline(0, color="cyan", lw=.65, alpha=.65)
        ax.set(title=title, xlabel=f"signed {coord} (µm)", ylabel="relative z (mm)")
    fig.colorbar(shown, ax=axes, label="plane-normalized measured intensity", shrink=.88)
    fig.suptitle("Measured q=20 longitudinal evolution from the complete BMG stack",
                 fontsize=15)
    p_xy = output_dir / "04_measured_q20_XZ_YZ_combined_all_planes.png"
    fig.savefig(p_xy, dpi=450, bbox_inches="tight")
    plt.close(fig)
    files["xz_yz"] = str(p_xy)
    return files


def _load_kr(modal_dir: Path, fallback_kr: float) -> float:
    summary = modal_dir / "summary.json"
    if summary.exists():
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
            if data.get("kr_rad_per_um") is not None:
                return float(data["kr_rad_per_um"]) * 1e6
        except Exception:
            pass
    return float(fallback_kr)


def _save_phase_physics_figure(
    modal_dir: Path,
    z_mm: np.ndarray,
    output_dir: Path,
    *,
    wavelength_m: float,
    kr_m_inv: float,
    z_at_relative_zero_from_axicon_m: float | None,
) -> dict:
    phase_path = modal_dir / "annular_aberration_phase.npy"
    if not phase_path.exists():
        raise FileNotFoundError(f"Missing modal annular phase rows: {phase_path}")
    phase_rows = np.load(phase_path)
    if phase_rows.shape[0] != len(z_mm):
        raise ValueError("annular phase file does not contain one row per measured z plane")

    assembled = assemble_transverse_residual_phase(
        phase_rows,
        z_mm * 1e-3,
        wavelength_m=wavelength_m,
        k_perp_m_inv=kr_m_inv,
        z_at_relative_zero_from_axicon_m=z_at_relative_zero_from_axicon_m,
        grid_size=600,
    )
    fixed = np.asarray(assembled["gauge_fixed_phase_rows_rad"])
    x_mm = np.asarray(assembled["x_m"]) * 1e3
    residual = np.asarray(assembled["residual_phase_rad"])
    correction = np.asarray(assembled["conjugate_correction_phase_rad"])

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), constrained_layout=True)
    im0 = axes[0].imshow(fixed, origin="lower", aspect="auto", cmap=PHASE_CMAP,
                         vmin=-np.pi, vmax=np.pi,
                         extent=[0, 360, z_mm[0], z_mm[-1]])
    axes[0].set(title="Recovered residual on each sampled annulus",
                xlabel="azimuth θ (deg)", ylabel="relative z (mm)")
    fig.colorbar(im0, ax=axes[0], label="wrapped residual phase (rad)", shrink=.84)

    for ax, arr, title in (
        (axes[1], residual, "Assembled transverse residual ψ(ρ,θ)"),
        (axes[2], correction, "Conjugate transverse correction −ψ(ρ,θ)"),
    ):
        shown = ax.imshow(arr, origin="lower", cmap=PHASE_CMAP,
                          vmin=-np.pi, vmax=np.pi,
                          extent=[x_mm[0], x_mm[-1], x_mm[0], x_mm[-1]])
        ax.set_aspect("equal")
        ax.set(title=title, xlabel="input-plane x (mm)", ylabel="input-plane y (mm)")
        fig.colorbar(shown, ax=ax, label="wrapped phase (rad)", shrink=.84)

    radius_text = ("absolute annulus radius" if assembled["absolute_radius_calibrated"]
                   else "relative annulus radius only")
    fig.suptitle(
        "q=20 residual-phase reconstruction — target vortex phase removed\n"
        f"z supplies radial annulus diversity (ρz = z tan α); {radius_text}; "
        "annular piston is not claimed",
        fontsize=14,
    )
    out = output_dir / "05_retrieved_residual_phase_physics.png"
    fig.savefig(out, dpi=420, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        output_dir / "05_retrieved_residual_phase_physics_data.npz",
        x_m=assembled["x_m"],
        rho_rows_m=assembled["rho_rows_m"],
        rho_rows_physical_m=assembled["rho_rows_physical_m"],
        z_relative_m=assembled["z_relative_m"],
        residual_phase_rad=residual,
        conjugate_correction_phase_rad=correction,
        gauge_fixed_phase_rows_rad=fixed,
    )
    return {k: v for k, v in assembled.items() if not isinstance(v, np.ndarray)} | {
        "figure": str(out)
    }


def _save_single_transverse_forward_figure(
    single_mask_dir: Path,
    output_dir: Path,
) -> str:
    stack_path = single_mask_dir / "single_mask_forward_stacks.npz"
    if not stack_path.exists():
        raise FileNotFoundError(f"Missing single-mask forward stacks: {stack_path}")
    d = np.load(stack_path)
    measured = np.asarray(d["measured"], float)
    ideal = np.asarray(d["ideal"], float)
    error_model = np.asarray(d["physical_annulus_inverse"], float)
    z_mm = np.asarray(d["z_relative_mm"], float)
    axis_um = np.asarray(d["x_um"], float)
    d.close()

    stacks = (measured, error_model, ideal)
    titles = ("LAB MEASURED", "MODEL + RETRIEVED RESIDUAL",
              "MODEL AFTER CONJUGATE CORRECTION")
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9), constrained_layout=True,
                             sharex=True, sharey=True)
    for col, (stack, title) in enumerate(zip(stacks, titles)):
        xz, yz = central_band_sections(stack, half_width_px=2)
        for row, (arr, section) in enumerate(((xz, "x–z"), (yz, "y–z"))):
            shown = axes[row, col].imshow(
                arr, origin="lower", aspect="auto", cmap=THERMAL_CMAP,
                vmin=0, vmax=1,
                extent=[axis_um[0], axis_um[-1], z_mm[0], z_mm[-1]],
                interpolation="nearest",
            )
            axes[row, col].axvline(0, color="cyan", lw=.55, alpha=.6)
            axes[row, col].set(title=f"{section} | {title}",
                               xlabel="signed transverse coordinate (µm)",
                               ylabel="relative z (mm)")
    fig.colorbar(shown, ax=axes, label="plane-normalized intensity", shrink=.82)
    fig.suptitle(
        "Single-transverse-phase physics check\n"
        "The correction exists only at the input plane; longitudinal behaviour is generated by propagation",
        fontsize=14,
    )
    out = output_dir / "06_single_transverse_phase_forward_model.png"
    fig.savefig(out, dpi=420, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def rebuild(
    data_dir: Path,
    *,
    modal_dir: Path,
    single_mask_dir: Path,
    output_dir: Path,
    z_start_mm: float = -17.0,
    z_step_mm: float = 1.0,
    expected_planes: int = 18,
    expected_repeats: int = 4,
    wavelength_m: float = 1030e-9,
    pixel_pitch_m: float = 5.5e-6,
    q: int = 20,
    view_limit_um: float = 180.0,
    z_at_relative_zero_from_axicon_m: float | None = None,
    run_modal_if_missing: bool = True,
    run_single_mask_if_missing: bool = True,
) -> dict:
    data_dir = Path(data_dir)
    modal_dir = Path(modal_dir)
    single_mask_dir = Path(single_mask_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stack, plane_centres, qc_rows, repeats = load_complete_registered_stack(
        data_dir, roi_size=768)
    z_indices = np.array(sorted(repeats), dtype=int)
    z_mm = z_start_mm + np.arange(len(z_indices), dtype=float) * z_step_mm

    if len(z_indices) != expected_planes:
        raise RuntimeError(
            f"Expected {expected_planes} BMG z planes but found {len(z_indices)}: {z_indices.tolist()}"
        )
    wrong_repeats = {zi: n for zi, n in repeats.items() if n != expected_repeats}
    if wrong_repeats:
        raise RuntimeError(
            f"Expected {expected_repeats} BMG repeats at every plane; mismatches: {wrong_repeats}"
        )

    measured_files = _save_measured_figures(
        stack, z_mm, output_dir, pixel_pitch_m=pixel_pitch_m,
        view_limit_um=view_limit_um, repeat_counts=repeats, qc_rows=qc_rows)

    if not (modal_dir / "annular_aberration_phase.npy").exists():
        if not run_modal_if_missing:
            raise FileNotFoundError(modal_dir / "annular_aberration_phase.npy")
        from q20_modal_analysis import run_modal_q20
        run_modal_q20(
            data_dir, modal_dir, pixel_pitch_m=pixel_pitch_m, q=q,
            z_positions_mm=z_mm)

    kr_m_inv = _load_kr(modal_dir, 489678.1594027835)
    phase_info = _save_phase_physics_figure(
        modal_dir, z_mm, output_dir, wavelength_m=wavelength_m,
        kr_m_inv=kr_m_inv,
        z_at_relative_zero_from_axicon_m=z_at_relative_zero_from_axicon_m)

    stack_path = single_mask_dir / "single_mask_forward_stacks.npz"
    if not stack_path.exists() and run_single_mask_if_missing:
        from single_mask_inverse_forward_test import run_single_mask_inverse_test
        legacy_map = modal_dir / "UNCALIBRATED_DO_NOT_APPLY_q20_modal_correction.npy"
        if not legacy_map.exists():
            raise FileNotFoundError(
                "The current single-mask validator expects the legacy diagnostic map file "
                f"to exist alongside annular_aberration_phase.npy: {legacy_map}"
            )
        run_single_mask_inverse_test(
            data_dir, legacy_map, single_mask_dir,
            z_relative_mm=z_mm, wavelength_m=wavelength_m,
            pixel_pitch_m=pixel_pitch_m, q=q)
    forward_figure = _save_single_transverse_forward_figure(single_mask_dir, output_dir)

    saturation_max = max(float(r["saturation_fraction"]) for r in qc_rows)
    provenance = {
        "data_dir": str(data_dir.resolve()),
        "raw_bmg_files_used": int(sum(repeats.values())),
        "z_planes_used": int(len(repeats)),
        "repeats_per_plane": {str(k): int(v) for k, v in repeats.items()},
        "z_mm": z_mm.tolist(),
        "wavelength_nm": wavelength_m * 1e9,
        "camera_pixel_um": pixel_pitch_m * 1e6,
        "effective_q": int(q),
        "kr_rad_per_um": kr_m_inv * 1e-6,
        "cone_alpha_deg": float(np.degrees(cone_geometry(wavelength_m, kr_m_inv).alpha_rad)),
        "maximum_raw_adc_fraction": saturation_max,
        "near_saturation_warning": bool(saturation_max >= 0.98),
        "phase_interpretation": (
            "one transverse residual psi(rho,theta) reconstructed from z-sampled annuli; "
            "no independent longitudinal correction phase"
        ),
        "radial_piston_recovered": False,
        "target_qtheta_removed_from_residual": True,
        "hardware_ready": False,
        "hardware_blocker": phase_info["hardware_blocker"],
        "measured_figures": measured_files,
        "phase_figure": phase_info["figure"],
        "forward_model_figure": forward_figure,
    }
    (output_dir / "q20_presentation_rebuild_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_data = Path(os.environ.get("BESSEL_ZSCAN_DATA_DIR", here / "z-scan 2 1010"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--modal-dir", type=Path,
                        default=here / "outputs" / "slm_closed_loop_alignment" / "modal_q20")
    parser.add_argument("--single-mask-dir", type=Path,
                        default=here / "outputs" / "slm_closed_loop_alignment" / "modal_q20" /
                                "single_mask_inverse_forward_test")
    parser.add_argument("--output-dir", type=Path,
                        default=here.parents[2] / "figures" / "experimental" /
                                "q20_aberration" / "presentation_rebuild")
    parser.add_argument("--z-start-mm", type=float, default=-17.0)
    parser.add_argument("--z-step-mm", type=float, default=1.0)
    parser.add_argument("--absolute-z-at-relative-zero-mm", type=float, default=None,
                        help=("Distance from the axicon/input reference to relative z=0. "
                              "Omit to keep the radial phase diagnostic relative and non-hardware."))
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    report = rebuild(
        args.data_dir,
        modal_dir=args.modal_dir,
        single_mask_dir=args.single_mask_dir,
        output_dir=args.output_dir,
        z_start_mm=args.z_start_mm,
        z_step_mm=args.z_step_mm,
        z_at_relative_zero_from_axicon_m=(
            None if args.absolute_z_at_relative_zero_mm is None
            else args.absolute_z_at_relative_zero_mm * 1e-3
        ),
    )
    print(json.dumps(report, indent=2))
