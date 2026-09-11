from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from vbb_study.short_bessel import ShortBesselDesignInput, design_short_bessel
from vbb_study.short_bessel_propagation import ShortBesselPropagationConfig
from vbb_study.short_bessel_realizability import (
    combined_slm_phase_preview,
    optimize_radius_for_halfmax_zone,
    propagate_vortex_case,
)


UM = 1.0e-6
MM = 1.0e-3


def _sqrt_norm(values: np.ndarray) -> np.ndarray:
    arr = np.maximum(np.asarray(values, dtype=float), 0.0)
    vmax = float(np.max(arr))
    return np.sqrt(arr / vmax) if vmax > 0.0 else np.zeros_like(arr)


def _norm(values: np.ndarray) -> np.ndarray:
    arr = np.maximum(np.asarray(values, dtype=float), 0.0)
    vmax = float(np.max(arr))
    return arr / vmax if vmax > 0.0 else np.zeros_like(arr)


def _write_search_csv(path: Path, optimization) -> None:
    rows = [p.as_dict() for p in optimization.search_points]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _radius_search_figure(optimization, path: Path) -> None:
    rows = sorted(optimization.search_points, key=lambda p: p.pre_radius_mm)
    x = np.array([p.pre_radius_mm for p in rows])
    y = np.array([p.halfmax_zone_um for p in rows])
    f = np.array([p.core_fwhm_um for p in rows])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    fig.suptitle("Numerical radius search for a 5 µm / 500 µm useful Bessel zone", fontsize=15)

    ax = axes[0]
    order = np.argsort(x)
    ax.plot(x[order], y[order], "o-")
    ax.axhline(optimization.requested_halfmax_zone_um, linestyle="--", linewidth=1.2, label="requested L50")
    ax.axvline(optimization.safe_pre_radius_mm, linestyle=":", linewidth=1.4, label="90% SLM-safe radius")
    ax.axvline(optimization.full_half_height_pre_radius_mm, linestyle="-.", linewidth=1.2, label="SLM half-height")
    ax.plot(
        [optimization.optimized_pre_radius_mm],
        [optimization.achieved_halfmax_zone_um],
        marker="*",
        markersize=15,
        label="optimized radius",
    )
    ax.set_xlabel("equivalent pre-objective / SLM 1/e field radius (mm)")
    ax.set_ylabel("propagated contiguous L50 (µm)")
    ax.set_title("Useful axial zone versus illuminated radius")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(x[order], f[order], "o-")
    ax.axhline(5.0, linestyle="--", linewidth=1.2, label="5 µm target")
    ax.axhspan(4.9, 5.1, alpha=0.12, label="±0.1 µm")
    ax.axvline(optimization.safe_pre_radius_mm, linestyle=":", linewidth=1.4)
    ax.axvline(optimization.full_half_height_pre_radius_mm, linestyle="-.", linewidth=1.2)
    ax.plot(
        [optimization.optimized_pre_radius_mm],
        [optimization.achieved_core_fwhm_um],
        marker="*",
        markersize=15,
    )
    ax.set_xlabel("equivalent pre-objective / SLM 1/e field radius (mm)")
    ax.set_ylabel("propagated B0 FWHM at peak (µm)")
    ax.set_title("Core-size stability while extending the Bessel zone")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    text = (
        f"optimized pre radius = {optimization.optimized_pre_radius_mm:.3f} mm\n"
        f"achieved L50 = {optimization.achieved_halfmax_zone_um:.1f} µm\n"
        f"core FWHM = {optimization.achieved_core_fwhm_um:.3f} µm\n"
        f"90% safe limit L50 = {optimization.safe_limit_halfmax_zone_um:.1f} µm\n"
        f"full half-height limit L50 = {optimization.full_limit_halfmax_zone_um:.1f} µm\n"
        f"status: {optimization.recommended_status}"
    )
    fig.text(0.5, 0.01, text, ha="center", va="bottom", family="monospace", fontsize=9)
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _b0_figure(case, optimization, path: Path, x_half_um: float = 30.0) -> None:
    m = case.metrics
    xmask = np.abs(case.x_um) <= float(x_half_um)
    centre = int(np.argmin(np.abs(case.x_um)))
    line = _norm(case.peak_xy_intensity[centre])
    line_mask = np.abs(case.x_um) <= 20.0

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle("Optimized B0: propagated 5 µm core with ~500 µm L50 target", fontsize=15)

    ax = axes[0]
    im = ax.imshow(
        _sqrt_norm(case.xz_intensity[:, xmask]),
        extent=(case.x_um[xmask][0], case.x_um[xmask][-1], case.z_um[0], case.z_um[-1]),
        origin="lower",
        aspect="auto",
        cmap="inferno",
        vmin=0,
        vmax=1,
    )
    ax.axhline(m.halfmax_zone_start_um, linestyle="--", linewidth=1)
    ax.axhline(m.halfmax_zone_end_um, linestyle="--", linewidth=1)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("z in fused silica (µm)")
    ax.set_title(f"XZ | L50={m.halfmax_zone_length_um:.1f} µm")
    fig.colorbar(im, ax=ax, label="sqrt(norm intensity)")

    ax = axes[1]
    axial = _norm(case.axial_metric)
    ax.plot(case.z_um, axial)
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.axvspan(m.halfmax_zone_start_um, m.halfmax_zone_end_um, alpha=0.12)
    ax.set_xlabel("z (µm)")
    ax.set_ylabel("normalised on-axis intensity")
    ax.set_title(f"Peak z={m.peak_z_um:.0f} µm")
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    ax.plot(case.x_um[line_mask], line[line_mask])
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.axvline(-2.5, linestyle=":", linewidth=1)
    ax.axvline(+2.5, linestyle=":", linewidth=1)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("normalised intensity")
    ax.set_title(f"Peak-plane FWHM={m.central_fwhm_um:.3f} µm")
    ax.grid(True, alpha=0.25)

    fig.text(
        0.5,
        0.015,
        (
            f"sample 1/e radius={m.sample_beam_radius_um:.2f} µm | pre radius={m.pre_beam_radius_mm:.3f} mm | "
            f"{optimization.recommended_status}"
        ),
        ha="center",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _vortex_xz_figure(cases, path: Path, x_half_um: float = 34.0) -> None:
    fig, axes = plt.subplots(len(cases), 2, figsize=(13, 4.2 * len(cases)), squeeze=False)
    fig.suptitle("Same optimized conical spectrum with B0 / V1 / V3 helical phase", fontsize=15)

    for row, case in enumerate(cases):
        m = case.metrics
        xmask = np.abs(case.x_um) <= float(x_half_um)
        ax = axes[row, 0]
        im = ax.imshow(
            _sqrt_norm(case.xz_intensity[:, xmask]),
            extent=(case.x_um[xmask][0], case.x_um[xmask][-1], case.z_um[0], case.z_um[-1]),
            origin="lower",
            aspect="auto",
            cmap="inferno",
            vmin=0,
            vmax=1,
        )
        ax.axhline(m.halfmax_zone_start_um, linestyle="--", linewidth=1)
        ax.axhline(m.halfmax_zone_end_um, linestyle="--", linewidth=1)
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("z (µm)")
        ax.set_title(f"ℓ={m.ell}: XZ | L50({m.axial_metric_name})={m.halfmax_zone_length_um:.0f} µm")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

        ax = axes[row, 1]
        ax.plot(case.z_um, _norm(case.axial_metric))
        ax.axhline(0.5, linestyle="--", linewidth=1)
        ax.axvspan(m.halfmax_zone_start_um, m.halfmax_zone_end_um, alpha=0.12)
        ax.set_xlabel("z (µm)")
        ax.set_ylabel("normalised axial metric")
        label = f"winding={m.measured_phase_winding:.3f} (error {m.winding_error_turns:+.3f})"
        if m.ell == 0:
            label += f" | FWHM={m.central_fwhm_um:.2f} µm"
        else:
            label += f" | ring D={m.measured_main_ring_diameter_um:.2f} µm"
        ax.set_title(label)
        ax.grid(True, alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _vortex_peak_xy_figure(cases, path: Path, half_width_um: float = 28.0) -> None:
    fig, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 5.2), squeeze=False)
    fig.suptitle("Peak-plane transverse structure and preserved topological charge", fontsize=15)

    for ax, case in zip(axes.ravel(), cases):
        m = case.metrics
        mask = np.abs(case.x_um) <= float(half_width_um)
        cropped = case.peak_xy_intensity[np.ix_(mask, mask)]
        xc = case.x_um[mask]
        im = ax.imshow(
            _norm(cropped),
            extent=(xc[0], xc[-1], xc[0], xc[-1]),
            origin="lower",
            cmap="inferno",
            vmin=0,
            vmax=1,
        )
        if m.ell == 0:
            subtitle = f"B0 | FWHM {m.central_fwhm_um:.2f} µm"
        else:
            subtitle = (
                f"V{m.ell} | ring D {m.measured_main_ring_diameter_um:.2f} µm\n"
                f"theory {m.predicted_main_ring_diameter_um:.2f} µm | winding {m.measured_phase_winding:.3f}"
            )
        ax.set_title(subtitle)
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _phase_figure(design, previews, path: Path) -> None:
    fig, axes = plt.subplots(1, len(previews), figsize=(5.2 * len(previews), 5.0), squeeze=False)
    fig.suptitle(
        "SLM radial + vortex phase topology previews\n(no blaze, wavefront correction, or physical-axicon residual calibration)",
        fontsize=14,
    )
    for ax, preview in zip(axes.ravel(), previews):
        phase = preview["phase_wrapped_rad"]
        x_mm = preview["x_m"] / MM
        y_mm = preview["y_m"] / MM
        im = ax.imshow(
            phase,
            extent=(x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]),
            origin="lower",
            cmap="twilight",
            vmin=0,
            vmax=2 * np.pi,
        )
        ax.set_title(f"ℓ={preview['ell']}")
        ax.set_xlabel("SLM x (mm)")
        ax.set_ylabel("SLM y (mm)")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="wrapped phase (rad)")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_outputs(*, output_dir: Path, fwhm_um: float = 5.0, requested_l50_um: float = 500.0) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = ShortBesselDesignInput()
    design_input = ShortBesselDesignInput(
        target_fwhm_um=float(fwhm_um),
        target_length_um=float(requested_l50_um),
        wavelength_nm=float(base.wavelength_nm),
        sample_index=1.45,
        relay_demag=float(base.relay_demag),
        objective_na=float(base.objective_na),
        slm_pixel_pitch_um=float(base.slm_pixel_pitch_um),
        slm_resolution_x=int(base.slm_resolution_x),
        slm_resolution_y=int(base.slm_resolution_y),
        slm_safe_radius_fraction=float(base.slm_safe_radius_fraction),
        minimum_pixels_per_phase_wrap=float(base.minimum_pixels_per_phase_wrap),
        current_pre_beam_radius_mm=float(base.current_pre_beam_radius_mm),
    )
    design = design_short_bessel(design_input)

    search_cfg = ShortBesselPropagationConfig(
        grid_n=257,
        dx_um=0.35,
        z_min_um=0.0,
        z_max_um=900.0,
        z_points=181,
        bandlimit=True,
        include_evanescent=True,
    )
    optimization = optimize_radius_for_halfmax_zone(
        design,
        requested_halfmax_zone_um=float(requested_l50_um),
        config=search_cfg,
        lower_sample_radius_um=20.0,
        upper_sample_radius_um=45.0,
        tolerance_um=2.5,
        max_iterations=10,
    )

    final_cfg = ShortBesselPropagationConfig(
        grid_n=513,
        dx_um=0.25,
        z_min_um=0.0,
        z_max_um=850.0,
        z_points=171,
        bandlimit=True,
        include_evanescent=True,
    )
    optimized_radius_m = optimization.optimized_sample_radius_um * UM
    cases = [
        propagate_vortex_case(
            design,
            ell=ell,
            sample_beam_radius_m=optimized_radius_m,
            config=final_cfg,
            snapshot_z_um=(100.0, 250.0, 400.0, 550.0, 700.0),
        )
        for ell in (0, 1, 3)
    ]

    report = {
        "design": design.as_dict(),
        "radius_optimization": optimization.as_dict(),
        "high_resolution_cases": {str(case.ell): case.metrics.as_dict() for case in cases},
        "validation_flags": {
            "optimized_B0_core_within_0p15_um": bool(abs((cases[0].metrics.central_fwhm_um or 999.0) - fwhm_um) <= 0.15),
            "optimized_B0_L50_within_25_um": bool(abs(cases[0].metrics.halfmax_zone_length_um - requested_l50_um) <= 25.0),
            "V1_winding_preserved": bool(abs(cases[1].metrics.winding_error_turns) <= 0.1),
            "V3_winding_preserved": bool(abs(cases[2].metrics.winding_error_turns) <= 0.1),
            "V1_ring_exists": bool((cases[1].metrics.measured_main_ring_diameter_um or 0.0) > 0.0),
            "V3_ring_exists": bool((cases[2].metrics.measured_main_ring_diameter_um or 0.0) > 0.0),
        },
        "claim_boundary": (
            "Equivalent sample-plane scalar propagation and SLM geometry/sampling feasibility only. "
            "The complete SLM + 4F + physical axicon + objective + sample-interface bench is not yet explicitly calibrated."
        ),
    }
    report_path = output_dir / "short_bessel_realizability_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    search_csv = output_dir / "radius_search.csv"
    _write_search_csv(search_csv, optimization)

    fig1 = output_dir / "01_radius_optimization.png"
    fig2 = output_dir / "02_optimized_B0_propagation.png"
    fig3 = output_dir / "03_B0_V1_V3_propagation.png"
    fig4 = output_dir / "04_B0_V1_V3_peak_XY.png"
    fig5 = output_dir / "05_SLM_radial_plus_vortex_phase.png"
    _radius_search_figure(optimization, fig1)
    _b0_figure(cases[0], optimization, fig2)
    _vortex_xz_figure(cases, fig3)
    _vortex_peak_xy_figure(cases, fig4)

    # Downsample phase topology plots for compact figures, but export full SLM masks separately.
    previews_small = [combined_slm_phase_preview(design, ell=ell, nx=640, ny=360) for ell in (0, 1, 3)]
    _phase_figure(design, previews_small, fig5)

    mask_paths: list[str] = []
    for ell in (0, 1, 3):
        preview = combined_slm_phase_preview(design, ell=ell)
        mask_path = output_dir / f"slm_radial_vortex_ell{ell}_target_only.png"
        Image.fromarray(preview["gray_uint8"], mode="L").save(mask_path)
        mask_paths.append(str(mask_path))

    arrays_path = output_dir / "short_bessel_realizability_arrays.npz"
    np.savez_compressed(
        arrays_path,
        **{
            f"ell{case.ell}_x_um": case.x_um
            for case in cases
        },
        **{
            f"ell{case.ell}_z_um": case.z_um
            for case in cases
        },
        **{
            f"ell{case.ell}_xz_intensity": case.xz_intensity
            for case in cases
        },
        **{
            f"ell{case.ell}_axial_metric": case.axial_metric
            for case in cases
        },
        **{
            f"ell{case.ell}_peak_xy": case.peak_xy_intensity
            for case in cases
        },
    )

    return {
        "report": str(report_path),
        "radius_search_csv": str(search_csv),
        "radius_figure": str(fig1),
        "optimized_B0_figure": str(fig2),
        "vortex_propagation_figure": str(fig3),
        "vortex_peak_xy_figure": str(fig4),
        "slm_phase_figure": str(fig5),
        "slm_masks": mask_paths,
        "arrays": str(arrays_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize a propagated 5 µm Bessel beam for ~500 µm L50 and test V1/V3 topology.")
    parser.add_argument("--fwhm-um", type=float, default=5.0)
    parser.add_argument("--l50-um", type=float, default=500.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/short_bessel_realizability"))
    args = parser.parse_args()
    outputs = build_outputs(output_dir=args.output_dir, fwhm_um=args.fwhm_um, requested_l50_um=args.l50_um)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
