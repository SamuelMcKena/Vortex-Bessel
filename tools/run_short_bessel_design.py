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
from scipy import special

from vbb_study.short_bessel import (
    ShortBesselDesignInput,
    design_short_bessel,
    design_sweep,
    slm_phase_preview,
)


UM = 1.0e-6


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _grid_from_rows(rows: list[dict[str, object]], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fwhm = np.array(sorted({float(row["target_fwhm_um"]) for row in rows}), dtype=float)
    lengths = np.array(sorted({float(row["target_length_um"]) for row in rows}), dtype=float)
    z = np.full((lengths.size, fwhm.size), np.nan, dtype=float)
    f_index = {value: i for i, value in enumerate(fwhm)}
    l_index = {value: i for i, value in enumerate(lengths)}
    for row in rows:
        z[l_index[float(row["target_length_um"])], f_index[float(row["target_fwhm_um"])]] = float(row[key])
    return fwhm, lengths, z


def _summary_text(result) -> str:
    physical = "not supplied"
    residual = "not available"
    if result.physical_kr_sample_m_inv is not None:
        physical = f"{result.physical_kr_sample_m_inv:,.0f} m^-1"
        residual = (
            f"{result.residual_slm_kr_pre_m_inv:,.1f} m^-1 "
            f"({result.residual_phase_sign})"
        )
    return "\n".join(
        [
            f"Target: {result.target_fwhm_um:.2f} um FWHM x {result.target_length_um:.0f} um",
            f"Equivalent J0 first-zero diameter: {result.target_first_zero_diameter_um:.3f} um",
            f"k_r(sample): {result.target_kr_sample_m_inv:,.0f} m^-1",
            f"Cone angle in sample: {result.target_cone_angle_in_sample_deg:.3f} deg",
            f"Required NA: {result.required_objective_na:.4f} / available {result.objective_na_available:.3f}",
            "",
            f"Required sample 1/e field radius: {result.required_sample_beam_radius_um:.2f} um",
            f"Required pre-objective radius: {result.required_pre_beam_radius_mm:.3f} mm",
            f"Current pre radius: {result.current_pre_beam_radius_mm:.3f} mm",
            f"Predicted length at current radius: {result.predicted_length_with_current_beam_um:.1f} um",
            f"Beam-radius scale factor: {result.required_beam_radius_scale_factor:.3f}x",
            f"SLM safe radius: {result.slm_safe_radius_mm:.3f} mm",
            f"Aperture margin: {result.slm_aperture_margin_mm:+.3f} mm",
            "",
            f"Target k_r(pre): {result.target_kr_pre_m_inv:,.1f} m^-1",
            f"Target phase wrap: {result.target_phase_wrap_period_um:.1f} um / {result.target_pixels_per_phase_wrap:.1f} px",
            f"Physical-only inferred k_r: {physical}",
            f"SLM residual command: {residual}",
            "",
            f"First-order feasibility gate: {'PASS' if result.hard_feasible else 'FAIL'}",
            "Inverse-design feasibility only; not a calibrated bench prediction.",
        ]
    )


def build_outputs(
    *,
    config: ShortBesselDesignInput,
    output_dir: Path,
    sweep_fwhm_min_um: float = 3.0,
    sweep_fwhm_max_um: float = 8.0,
    sweep_length_min_um: float = 200.0,
    sweep_length_max_um: float = 1000.0,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = design_short_bessel(config)

    report_path = output_dir / "short_bessel_design_report.json"
    report_path.write_text(json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8")

    fwhm_values = np.linspace(float(sweep_fwhm_min_um), float(sweep_fwhm_max_um), 21)
    length_values = np.linspace(float(sweep_length_min_um), float(sweep_length_max_um), 17)
    rows = design_sweep(fwhm_values, length_values, base=config)
    sweep_path = output_dir / "short_bessel_design_sweep.csv"
    _write_csv(sweep_path, rows)

    preview = slm_phase_preview(result, use_full_resolution=True)
    phase_mask_path = output_dir / f"slm_axicon_phase_{preview['mode']}.png"
    Image.fromarray(np.asarray(preview["gray_uint8"], dtype=np.uint8), mode="L").save(phase_mask_path)

    f_grid, l_grid, pre_radius = _grid_from_rows(rows, "required_pre_beam_radius_mm")
    _, _, feasible = _grid_from_rows(rows, "hard_feasible")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Short-Bessel programmable-axicon inverse design\n"
        "target geometry + SLM/aperture/NA feasibility; not a calibrated bench prediction",
        fontsize=14,
    )

    ax = axes[0, 0]
    mesh = ax.pcolormesh(f_grid, l_grid, pre_radius, shading="auto")
    fig.colorbar(mesh, ax=ax, label="required pre-objective 1/e field radius (mm)")
    ax.contour(
        f_grid,
        l_grid,
        pre_radius,
        levels=[result.slm_safe_radius_mm],
        linewidths=1.5,
    )
    ax.contour(
        f_grid,
        l_grid,
        feasible,
        levels=[0.5],
        linestyles="--",
        linewidths=1.2,
    )
    ax.plot(result.target_fwhm_um, result.target_length_um, marker="x", markersize=10, mew=2)
    ax.set_xlabel("target J0 core FWHM (um)")
    ax.set_ylabel("target Bessel reference length (um)")
    ax.set_title("Required upstream beam radius")

    ax = axes[0, 1]
    radial_um = np.linspace(-2.0 * result.target_first_zero_diameter_um, 2.0 * result.target_first_zero_diameter_um, 1200)
    radial_m = radial_um * UM
    intensity = special.j0(result.target_kr_sample_m_inv * np.abs(radial_m)) ** 2
    ax.plot(radial_um, intensity)
    ax.axhline(0.5, linestyle="--", linewidth=1.0)
    ax.axvline(-0.5 * result.target_fwhm_um, linestyle=":", linewidth=1.0)
    ax.axvline(+0.5 * result.target_fwhm_um, linestyle=":", linewidth=1.0)
    ax.axvline(-0.5 * result.target_first_zero_diameter_um, linestyle="--", linewidth=1.0)
    ax.axvline(+0.5 * result.target_first_zero_diameter_um, linestyle="--", linewidth=1.0)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("sample-plane radius coordinate (um)")
    ax.set_ylabel("normalised J0^2 intensity")
    ax.set_title("Target transverse Bessel profile")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    phase = np.asarray(preview["phase_wrapped_rad"], dtype=float)
    mid = phase.shape[0] // 2
    x_um = np.asarray(preview["x_m"], dtype=float) / UM
    ax.plot(x_um, phase[mid])
    ax.set_xlabel("SLM x coordinate (um)")
    ax.set_ylabel("wrapped phase (rad)")
    ax.set_ylim(0.0, 2.0 * np.pi)
    ax.set_title(f"SLM radial phase preview: {preview['mode']}")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    ax.axis("off")
    ax.text(0.0, 1.0, _summary_text(result), va="top", ha="left", family="monospace", fontsize=9.5)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure_path = output_dir / "short_bessel_design_summary.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    return {
        "report": str(report_path),
        "sweep": str(sweep_path),
        "phase_mask": str(phase_mask_path),
        "figure": str(figure_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inverse-design a short Bessel target and generate a signed SLM axicon phase preview. "
            "Supplying --physical-fwhm-um converts the phase into the residual needed on top of "
            "the measured physical-axicon-only beam."
        )
    )
    parser.add_argument("--fwhm-um", type=float, default=5.0)
    parser.add_argument("--length-um", type=float, default=500.0)
    parser.add_argument("--wavelength-nm", type=float, default=1029.0)
    parser.add_argument("--sample-index", type=float, default=1.45)
    parser.add_argument("--relay-demag", type=float, default=None)
    parser.add_argument("--objective-na", type=float, default=0.45)
    parser.add_argument("--current-pre-radius-mm", type=float, default=2.0)
    parser.add_argument(
        "--physical-fwhm-um",
        type=float,
        default=None,
        help="Measured physical-axicon-only J0 core FWHM at the sample plane.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/short_bessel_design"),
    )
    args = parser.parse_args()

    base = ShortBesselDesignInput()
    cfg = ShortBesselDesignInput(
        target_fwhm_um=float(args.fwhm_um),
        target_length_um=float(args.length_um),
        wavelength_nm=float(args.wavelength_nm),
        sample_index=float(args.sample_index),
        relay_demag=float(base.relay_demag if args.relay_demag is None else args.relay_demag),
        objective_na=float(args.objective_na),
        slm_pixel_pitch_um=base.slm_pixel_pitch_um,
        slm_resolution_x=base.slm_resolution_x,
        slm_resolution_y=base.slm_resolution_y,
        slm_safe_radius_fraction=base.slm_safe_radius_fraction,
        minimum_pixels_per_phase_wrap=base.minimum_pixels_per_phase_wrap,
        current_pre_beam_radius_mm=float(args.current_pre_radius_mm),
        measured_physical_fwhm_um=None if args.physical_fwhm_um is None else float(args.physical_fwhm_um),
    )
    result = design_short_bessel(cfg)
    outputs = build_outputs(config=cfg, output_dir=args.output_dir)
    print(json.dumps({"design": result.as_dict(), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
