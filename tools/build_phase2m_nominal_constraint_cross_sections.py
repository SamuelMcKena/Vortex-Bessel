"""Build throughput-aware 1D transverse overlay comparisons.

For B0, V1 and V3 this renderer compares the same z=60 mm, fixed-laboratory
y=0 cross-section for a continuous ideal field and the nominal bench-constrained
route.  The presentation figure uses ONE shared intensity reference per beam:
both curves are divided by the continuous-ideal 2D peak.  The nominal curve is
therefore not independently renormalised, so nominal SLM/4F throughput and any
redistribution of the peak remain visible.

A second, independently peak-normalised morphology comparison is retained only
in the CSV diagnostics.  This separates two physically different questions:
(1) how much intensity survives the nominal route, and (2) how much the shape
changes after removing an overall scale factor.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import presentation_phase2j_style as style
from build_phase2l_nominal_constraints import (
    XY_HALF_M,
    _crop,
    _xy,
    continuous_ideal,
)
from vbb_study.digital_twin.vortex_system_route import build_system_route


CASES = (("B0", "B0 — ℓ=0"), ("V1", "V1 — ℓ=1"), ("V3", "V3 — ℓ=3"))
EPS = np.finfo(float).tiny


def _lineout_y0(values: np.ndarray, route: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return x coordinate and nearest fixed-lab y=0 line from a cropped field."""
    crop, x = _crop(values, route["grid"], half=XY_HALF_M)
    y0 = int(np.argmin(np.abs(x)))
    return np.asarray(x, dtype=float), np.asarray(crop[y0, :], dtype=float)


def _peak_normalise(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    return values / max(float(np.max(values)), EPS)


def build_cross_sections(output_dir: Path, grid_n: int) -> tuple[Path, Path]:
    rows: list[dict[str, float | str]] = []
    prepared = []

    for case_id, case_label in CASES:
        ideal = continuous_ideal(case_id, grid_n)
        nominal = build_system_route(case_id, grid_n=grid_n)

        ideal_raw = np.maximum(np.asarray(_xy(ideal), dtype=float), 0.0)
        nominal_raw = np.maximum(np.asarray(_xy(nominal), dtype=float), 0.0)

        # Presentation scaling: one reference only.  This preserves the nominal
        # route's intensity loss/redistribution instead of forcing both peaks to 1.
        ideal_peak = max(float(np.max(ideal_raw)), EPS)
        ideal_shared = ideal_raw / ideal_peak
        nominal_shared = nominal_raw / ideal_peak

        # Shape-only diagnostic retained in the CSV for comparison with the old
        # independently normalised figure.
        ideal_shape = _peak_normalise(ideal_raw)
        nominal_shape = _peak_normalise(nominal_raw)

        x_i, ideal_line = _lineout_y0(ideal_shared, ideal)
        x_n, nominal_line = _lineout_y0(nominal_shared, nominal)
        x_si, ideal_shape_line = _lineout_y0(ideal_shape, ideal)
        x_sn, nominal_shape_line = _lineout_y0(nominal_shape, nominal)
        if not (
            np.allclose(x_i, x_n, rtol=0.0, atol=1e-15)
            and np.allclose(x_i, x_si, rtol=0.0, atol=1e-15)
            and np.allclose(x_i, x_sn, rtol=0.0, atol=1e-15)
        ):
            raise RuntimeError(f"{case_id}: ideal and nominal lineout coordinates differ")

        shared_difference = nominal_line - ideal_line
        shape_difference = np.abs(nominal_shape_line - ideal_shape_line)
        nominal_peak_ratio = float(np.max(nominal_raw)) / ideal_peak

        ideal_crop, _ = _crop(ideal_raw, ideal["grid"], half=XY_HALF_M)
        nominal_crop, _ = _crop(nominal_raw, nominal["grid"], half=XY_HALF_M)
        roi_integrated_ratio = float(np.sum(nominal_crop)) / max(float(np.sum(ideal_crop)), EPS)
        max_shape_difference = float(np.max(shape_difference))

        prepared.append(
            (
                case_id,
                case_label,
                x_i,
                ideal_line,
                nominal_line,
                nominal_peak_ratio,
                roi_integrated_ratio,
                max_shape_difference,
            )
        )

        for (
            x_m,
            i_ideal_shared,
            i_nominal_shared,
            delta_shared,
            i_ideal_shape,
            i_nominal_shape,
            delta_shape,
        ) in zip(
            x_i,
            ideal_line,
            nominal_line,
            shared_difference,
            ideal_shape_line,
            nominal_shape_line,
            shape_difference,
        ):
            rows.append(
                {
                    "case_id": case_id,
                    "x_mm": float(x_m * 1e3),
                    "ideal_shared_reference_intensity": float(i_ideal_shared),
                    "nominal_shared_reference_intensity": float(i_nominal_shared),
                    "shared_reference_difference": float(delta_shared),
                    # Backward-compatible shape-only columns from the previous
                    # renderer.  These are no longer what the PNG displays.
                    "ideal_normalised_intensity": float(i_ideal_shape),
                    "nominal_normalised_intensity": float(i_nominal_shape),
                    "absolute_difference": float(delta_shape),
                    "nominal_peak_over_ideal_peak": nominal_peak_ratio,
                    "nominal_roi_integrated_over_ideal": roi_integrated_ratio,
                    "max_shape_only_absolute_difference": max_shape_difference,
                }
            )

    fig, axes = plt.subplots(3, 1, figsize=(11.8, 9.2), facecolor=style.FIG_BG, sharex=True)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.09, top=0.82, hspace=0.38)

    for r, (
        case_id,
        case_label,
        x_m,
        ideal_line,
        nominal_line,
        nominal_peak_ratio,
        roi_integrated_ratio,
        max_shape_difference,
    ) in enumerate(prepared):
        x_mm = x_m * 1e3
        ax = axes[r]
        style.style_ax(ax)

        ax.plot(
            x_mm,
            ideal_line,
            color=style.TEXT,
            linewidth=2.2,
            label="continuous ideal",
            zorder=3,
        )
        ax.plot(
            x_mm,
            nominal_line,
            color=style.RED,
            linewidth=1.9,
            label="nominal constrained",
            zorder=4,
        )
        ax.fill_between(
            x_mm,
            ideal_line,
            nominal_line,
            color=style.RED,
            alpha=0.10,
            linewidth=0.0,
            zorder=2,
        )

        ymax = max(1.04, 1.06 * float(np.max(nominal_line)))
        ax.set_ylim(-0.02, ymax)
        ax.set_ylabel("intensity / ideal peak", fontsize=9.5, weight="bold")
        ax.axvline(0.0, color=style.MUTED, alpha=0.18, linewidth=0.7)
        ax.set_title(
            f"{case_label}   ·   z = 60 mm   ·   fixed-lab y = 0",
            fontsize=11.8,
            weight="bold",
            pad=8,
        )
        ax.text(
            0.985,
            0.72,
            (
                f"nominal peak / ideal peak = {nominal_peak_ratio:.3f}\n"
                f"central-ROI integral ratio = {roi_integrated_ratio:.3f}\n"
                f"shape-only max |ΔI| = {max_shape_difference:.3f}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=style.MUTED,
            fontsize=8.4,
        )
        ax.legend(
            frameon=False,
            fontsize=8.8,
            loc="upper right",
            labelcolor=style.MUTED,
        )

    axes[-1].set_xlabel("x (mm)", fontsize=9.5)

    fig.suptitle(
        "1D transverse cross-sections: continuous ideal vs nominal constrained",
        color=style.TEXT,
        fontsize=17.5,
        weight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.902,
        "Shared reference: both curves are divided by the continuous-ideal peak for that beam",
        ha="center",
        color=style.MUTED,
        fontsize=10.0,
    )
    fig.text(
        0.5,
        0.867,
        "The nominal curve is NOT renormalised; SLM/4F intensity loss and peak redistribution therefore remain visible",
        ha="center",
        color=style.MUTED,
        fontsize=9.1,
    )

    png_path = output_dir / "04_nominal_constraint_cross_sections.png"
    fig.savefig(png_path, dpi=500, bbox_inches="tight", facecolor=style.FIG_BG, pad_inches=0.06)
    plt.close(fig)

    csv_path = output_dir / "04_nominal_constraint_cross_sections.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case_id",
                "x_mm",
                "ideal_shared_reference_intensity",
                "nominal_shared_reference_intensity",
                "shared_reference_difference",
                "ideal_normalised_intensity",
                "nominal_normalised_intensity",
                "absolute_difference",
                "nominal_peak_over_ideal_peak",
                "nominal_roi_integrated_over_ideal",
                "max_shape_only_absolute_difference",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    return png_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-n", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path, csv_path = build_cross_sections(args.output_dir, args.grid_n)
    print(png_path)
    print(csv_path)


if __name__ == "__main__":
    main()
