"""Resolved scalar ideal/nominal-device comparisons from the same optical chain.

This is a source-scale *simulation*, not a calibrated sample-plane prediction.
The 10 mm input window must resolve the conical phase before scaled-output
propagation; resizing an under-resolved FFT image never satisfies that test.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vbb_study.digital_twin.phase2b_visual_cases import _scalar_seed, _sas_scalar_hero
from vbb_study.equations.propagation import samples_per_radial_period

MIN_SOURCE_SAMPLES = 6.0
MIN_OUTPUT_SAMPLES = 12.0
ROUTES = ("ideal_optical_route", "realistic_fixed_bench_route")
CASES = (("B0", 0), ("V1", 1), ("V3", 3))


def _crop(a: np.ndarray, grid: dict, halfwidth_m: float):
    x = np.asarray(grid["x"])
    keep = np.flatnonzero(np.abs(x) <= halfwidth_m)
    if keep.size < 24:
        raise ValueError("fewer than 24 physical output pixels across the requested ROI")
    s = slice(int(keep[0]), int(keep[-1]) + 1)
    return a[s, s], x[s]


def calculate(case_id: str, ell: int, *, grid_n: int, z_m: float, pad_factor: int):
    results = []
    for route in ROUTES:
        field, grid, seed = _scalar_seed(case_id, ell, grid_n=grid_n, variant=route)
        source_samples = samples_per_radial_period(grid["dx"], seed["radial_wavevector_m_inv"])
        if source_samples < MIN_SOURCE_SAMPLES:
            raise ValueError(
                f"{case_id}: input phase under-resolved ({source_samples:.2f} samples/period); "
                "increase grid_n at fixed 10 mm window, not display interpolation"
            )
        zoom = _sas_scalar_hero(
            field, grid, seed["wavelength_m"], z_m=z_m, pad_factor=pad_factor
        )
        output_samples = samples_per_radial_period(
            zoom["output_dx_m"], seed["radial_wavevector_m_inv"]
        )
        if not zoom["sas_valid"] or output_samples < MIN_OUTPUT_SAMPLES:
            raise ValueError(f"{case_id}: SAS validity/output sampling gate failed")
        intensity = np.asarray(zoom["intensity"], dtype=float)
        power = float(intensity.sum() * zoom["output_dx_m"] ** 2)
        if not np.isfinite(power) or power <= 0:
            raise ValueError(f"{case_id}: no finite power in the propagated output")
        results.append({
            "case_id": case_id, "route": route, "intensity": intensity,
            "grid": zoom["grid"], "source_grid_n": grid_n,
            "source_dx_um": grid["dx"] * 1e6,
            "output_dx_um": zoom["output_dx_m"] * 1e6,
            "source_samples_per_period": source_samples,
            "output_samples_per_period": output_samples,
            "z_mm": z_m * 1e3, "kr_rad_per_m": seed["radial_wavevector_m_inv"],
            "retained_power_au": power, "first_order_fraction": seed["first_order_efficiency"],
            "bench_calibrated": seed["bench_calibrated"],
        })
    return results


def render(output_dir: Path, *, grid_n: int = 1024, z_m: float = 0.06, pad_factor: int = 2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 10.8), layout="constrained")
    rows = []
    halfwidth = 0.24e-3
    for row, (case_id, ell) in enumerate(CASES):
        cases = calculate(case_id, ell, grid_n=grid_n, z_m=z_m, pad_factor=pad_factor)
        # Normalize by integrated retained power for morphology. Keep throughput
        # separate in the manifest; do not claim equal delivered pulse energies.
        crops = []
        for entry in cases:
            normalized = entry["intensity"] / entry["retained_power_au"]
            crop, x = _crop(normalized, entry["grid"], halfwidth)
            crops.append(crop)
        scale = max(float(np.max(a)) for a in crops)
        difference = np.abs(crops[1] - crops[0]) / scale
        arrays = [crops[0] / scale, crops[1] / scale, difference]
        for col, a in enumerate(arrays):
            ax = axes[row, col]
            vmax = 1.0 if col < 2 else max(float(np.max(difference)), 1e-12)
            ax.imshow(a, origin="lower", extent=[x[0]*1e3, x[-1]*1e3]*2,
                      cmap="magma" if col != 2 else "viridis", vmin=0, vmax=vmax,
                      interpolation="nearest")
            ax.set_aspect("equal")
            if col == 2:
                ax.text(.02, .02, f"max $\\Delta I$ / peak = {vmax:.2g}",
                        transform=ax.transAxes, color="white", fontsize=8,
                        bbox={"facecolor": "black", "alpha": .6, "edgecolor": "none"})
            if row == 0:
                ax.set_title(("ideal optical", "nominal throughput-only + 4F", "difference (own scale)")[col])
            ax.set_xlabel("x (mm)")
            if col == 0:
                ax.set_ylabel(f"{case_id} ($\\ell={ell}$)\ny (mm)")
        overlap = float(
            np.sum(crops[0] * crops[1]) /
            np.sqrt(np.sum(crops[0] ** 2) * np.sum(crops[1] ** 2))
        )
        rows.extend({k: v for k, v in entry.items() if k not in {"intensity", "grid"}}
                    | {"roi_morphology_overlap": overlap, "roi_halfwidth_mm": halfwidth * 1e3}
                    for entry in cases)
    fig.suptitle(
        f"Matched source-scale scalar routes at z = {z_m*1e3:g} mm\n"
        "SAS physical output grid; equal retained-power morphology; no measured calibration",
        fontsize=13,
    )
    fig.savefig(output_dir / "matched_scalar_routes.png", dpi=260)
    fig.savefig(output_dir / "matched_scalar_routes.pdf")
    plt.close(fig)
    with (output_dir / "matched_scalar_routes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("outputs/figures/matched_scalar_audit"))
    p.add_argument("--grid-n", type=int, default=1024)
    p.add_argument("--z-mm", type=float, default=60.0)
    p.add_argument("--pad-factor", type=int, default=2)
    args = p.parse_args()
    render(args.output_dir, grid_n=args.grid_n, z_m=args.z_mm / 1e3,
           pad_factor=args.pad_factor)


if __name__ == "__main__":
    main()
