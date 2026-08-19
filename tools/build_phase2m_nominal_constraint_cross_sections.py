"""Build 1D transverse overlay comparisons for nominal hardware constraints.

This presentation diagnostic complements ``build_phase2l_nominal_constraints``.
For B0, V1 and V3 it takes the same z=60 mm transverse planes used by the 2D
ideal-versus-nominal comparison, extracts the fixed laboratory y=0 line, and
plots continuous ideal and nominal bench-constrained intensity directly on the
same axes for immediate visual comparison.

The two parent transverse fields are independently peak-normalised exactly as in
the 2D morphology comparison.  This is therefore a morphology diagnostic, not
an absolute-throughput comparison and not a calibrated bench prediction.  The
absolute difference is retained in the CSV export for numerical inspection but
is intentionally not shown as a separate presentation panel.
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
    _norm,
    _xy,
    continuous_ideal,
)
from vbb_study.digital_twin.vortex_system_route import build_system_route


CASES = (("B0", "B0 — ℓ=0"), ("V1", "V1 — ℓ=1"), ("V3", "V3 — ℓ=3"))


def _lineout_y0(values: np.ndarray, route: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return x coordinate and nearest fixed-lab y=0 line from a cropped field."""
    crop, x = _crop(values, route["grid"], half=XY_HALF_M)
    y0 = int(np.argmin(np.abs(x)))
    return np.asarray(x, dtype=float), np.asarray(crop[y0, :], dtype=float)


def build_cross_sections(output_dir: Path, grid_n: int) -> tuple[Path, Path]:
    rows: list[dict[str, float | str]] = []
    prepared = []

    for case_id, case_label in CASES:
        ideal = continuous_ideal(case_id, grid_n)
        nominal = build_system_route(case_id, grid_n=grid_n)

        ideal_xy = _norm(_xy(ideal))
        nominal_xy = _norm(_xy(nominal))

        x_i, ideal_line = _lineout_y0(ideal_xy, ideal)
        x_n, nominal_line = _lineout_y0(nominal_xy, nominal)
        if not np.allclose(x_i, x_n, rtol=0.0, atol=1e-15):
            raise RuntimeError(f"{case_id}: ideal and nominal lineout coordinates differ")

        difference = np.abs(nominal_line - ideal_line)
        prepared.append((case_id, case_label, x_i, ideal_line, nominal_line))

        for x_m, i_ideal, i_nominal, d_abs in zip(x_i, ideal_line, nominal_line, difference):
            rows.append(
                {
                    "case_id": case_id,
                    "x_mm": float(x_m * 1e3),
                    "ideal_normalised_intensity": float(i_ideal),
                    "nominal_normalised_intensity": float(i_nominal),
                    "absolute_difference": float(d_abs),
                }
            )

    fig, axes = plt.subplots(3, 1, figsize=(11.8, 9.2), facecolor=style.FIG_BG, sharex=True)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.09, top=0.83, hspace=0.34)

    for r, (case_id, case_label, x_m, ideal_line, nominal_line) in enumerate(prepared):
        x_mm = x_m * 1e3
        ax = axes[r]
        style.style_ax(ax)

        ax.plot(
            x_mm,
            ideal_line,
            color=style.TEXT,
            linewidth=2.2,
            label="continuous ideal",
        )
        ax.plot(
            x_mm,
            nominal_line,
            color=style.RED,
            linewidth=1.8,
            label="nominal constrained",
        )
        ax.set_ylim(-0.02, 1.04)
        ax.set_ylabel("normalised intensity", fontsize=9.5, weight="bold")
        ax.axvline(0.0, color=style.MUTED, alpha=0.18, linewidth=0.7)
        ax.set_title(
            f"{case_label}   ·   z = 60 mm   ·   fixed-lab y = 0",
            fontsize=11.8,
            weight="bold",
            pad=8,
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
        0.905,
        "B0, V1 and V3 overlaid directly on the same fixed laboratory cross-section",
        ha="center",
        color=style.MUTED,
        fontsize=10.0,
    )
    fig.text(
        0.5,
        0.872,
        "Each parent field is independently peak-normalised: comparison isolates transverse morphology rather than throughput",
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
                "ideal_normalised_intensity",
                "nominal_normalised_intensity",
                "absolute_difference",
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
