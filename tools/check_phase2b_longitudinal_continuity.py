from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from vbb_study.digital_twin.phase2b_visual_cases import _scalar_seed
from vbb_study.digital_twin.propagation_audit import (
    central_roi_mask,
    compare_intensity_fields,
    scalar_padded_reference,
)
from vbb_study.equations.propagation import angular_spectrum_propagate_bl


EPS = np.finfo(float).tiny


def _normalised_row_changes(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=float)
    scale = max(float(np.max(values)), EPS)
    delta = np.diff(values / scale, axis=0)
    return np.sqrt(np.mean(delta * delta, axis=1))


def _case_audit(case_id: str, ell: int, grid_n: int) -> dict[str, Any]:
    field0, grid, meta = _scalar_seed(case_id, ell, grid_n=int(grid_n))
    wavelength = float(meta["wavelength_m"])
    coarse_z = np.arange(0.0, 200e-3 + 1e-12, 2e-3)
    n = int(grid["N"])
    mid = n // 2
    xz = np.empty((coarse_z.size, n), dtype=float)
    yz = np.empty_like(xz)
    for iz, z in enumerate(coarse_z):
        field = field0 if np.isclose(z, 0.0) else angular_spectrum_propagate_bl(
            field0,
            dict(grid),
            wavelength,
            float(z),
            n_medium=1.0,
            bandlimit=True,
            include_evanescent=True,
        )
        intensity = np.abs(field) ** 2
        xz[iz] = intensity[mid, :]
        yz[iz] = intensity[:, mid]

    cx = _normalised_row_changes(xz)
    cy = _normalised_row_changes(yz)
    combined = np.maximum(cx, cy)
    worst = int(np.argmax(combined))
    z0 = float(coarse_z[worst])
    z1 = float(coarse_z[worst + 1])

    fine_start = max(0.0, z0 - 1.0e-3)
    fine_end = min(float(coarse_z[-1]), z1 + 1.0e-3)
    fine_z = np.arange(fine_start, fine_end + 0.125e-3, 0.25e-3)
    fine_xz = np.empty((fine_z.size, n), dtype=float)
    fine_yz = np.empty_like(fine_xz)
    comparisons: list[dict[str, float]] = []
    roi = central_roi_mask(grid, 1.5e-3 if case_id != "G0" else 2.5e-3)
    for iz, z in enumerate(fine_z):
        candidate = field0 if np.isclose(z, 0.0) else angular_spectrum_propagate_bl(
            field0,
            dict(grid),
            wavelength,
            float(z),
            n_medium=1.0,
            bandlimit=True,
            include_evanescent=True,
        )
        intensity = np.abs(candidate) ** 2
        fine_xz[iz] = intensity[mid, :]
        fine_yz[iz] = intensity[:, mid]
        reference = scalar_padded_reference(
            field0,
            grid,
            wavelength_m=wavelength,
            z_m=float(z),
            n_medium=1.0,
            pad_factor=2,
        )
        cmp = compare_intensity_fields(
            candidate,
            reference,
            roi_mask=roi,
            dx_m=float(grid["dx"]),
        )
        comparisons.append(
            {
                "z_m": float(z),
                "intensity_correlation": float(cmp.intensity_correlation),
                "normalised_relative_l2": float(cmp.normalised_relative_l2),
                "peak_ratio": float(cmp.peak_ratio_candidate_to_reference),
                "roi_power_ratio": float(cmp.roi_power_ratio_candidate_to_reference),
            }
        )

    fx = _normalised_row_changes(fine_xz)
    fy = _normalised_row_changes(fine_yz)
    fine_max = float(max(np.max(fx), np.max(fy)))
    coarse_max = float(combined[worst])
    step_ratio = 0.25e-3 / 2.0e-3
    observed_ratio = float(fine_max / max(coarse_max, EPS))
    min_corr = float(min(row["intensity_correlation"] for row in comparisons))
    max_l2 = float(max(row["normalised_relative_l2"] for row in comparisons))

    return {
        "case_id": case_id,
        "grid_n": int(grid_n),
        "coarse_step_m": 2e-3,
        "fine_step_m": 0.25e-3,
        "step_ratio": step_ratio,
        "worst_coarse_interval_start_m": z0,
        "worst_coarse_interval_end_m": z1,
        "coarse_max_normalised_row_rms_change": coarse_max,
        "fine_max_normalised_row_rms_change": fine_max,
        "fine_to_coarse_change_ratio": observed_ratio,
        "fine_change_reduces_with_step": bool(observed_ratio < 0.75),
        "minimum_padded_reference_correlation_in_fine_window": min_corr,
        "maximum_padded_reference_l2_in_fine_window": max_l2,
        "padded_reference_pass": bool(min_corr >= 0.995 and max_l2 <= 0.03),
        "fine_z_comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve the worst Phase2B scalar longitudinal interval on a finer z grid."
    )
    parser.add_argument("--grid-n", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/validation/propagation_audit/phase2b_fine_z_continuity.json"),
    )
    args = parser.parse_args()

    cases = [
        _case_audit("G0", 0, int(args.grid_n)),
        _case_audit("B0", 0, int(args.grid_n)),
        _case_audit("V1", 1, int(args.grid_n)),
        _case_audit("V3", 3, int(args.grid_n)),
    ]
    failures: list[str] = []
    for row in cases:
        if not row["fine_change_reduces_with_step"]:
            failures.append(
                f"{row['case_id']} worst longitudinal change does not reduce sufficiently on 8x finer z sampling: "
                f"ratio={row['fine_to_coarse_change_ratio']:.4f}"
            )
        if not row["padded_reference_pass"]:
            failures.append(
                f"{row['case_id']} fine-z BL-ASM disagrees with padded reference: "
                f"min corr={row['minimum_padded_reference_correlation_in_fine_window']:.6f}, "
                f"max L2={row['maximum_padded_reference_l2_in_fine_window']:.6f}"
            )

    payload = {
        "outcome": "PHASE2B-FINE-Z-CONTINUITY-AUDIT",
        "hard_pass": not failures,
        "grid_n": int(args.grid_n),
        "cases": cases,
        "hard_failures": failures,
        "interpretation": (
            "A genuine rapidly forming field may have a large coarse adjacent-row change. The audit requires that the change shrink when dz is reduced by 8x and that every fine-z plane agree with the independent 2x-padded unbandlimited ASM reference."
        ),
        "report_figures_authorised": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
