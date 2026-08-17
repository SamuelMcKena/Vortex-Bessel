from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vbb_study.digital_twin.vortex_continuous_propagation import (
    build_fixed_support_spectrum,
    native_field_at_z,
)
from vbb_study.digital_twin.vortex_system_route import SystemErrorConfig, build_system_route
from vbb_study.equations.fields import make_xy_grid
from vbb_study.equations.propagation import angular_spectrum_propagate_bl


EPS = np.finfo(float).tiny


def _pad_center(field: np.ndarray, factor: int = 2) -> np.ndarray:
    n = field.shape[0]
    out = np.zeros((factor * n, factor * n), dtype=np.complex128)
    start = (out.shape[0] - n) // 2
    out[start : start + n, start : start + n] = field
    return out


def _crop_center(field: np.ndarray, n: int) -> np.ndarray:
    start = (field.shape[0] - n) // 2
    return field[start : start + n, start : start + n]


def _intensity_correlation(a: np.ndarray, b: np.ndarray, roi: np.ndarray) -> float:
    aa = (np.abs(a) ** 2)[roi].astype(float)
    bb = (np.abs(b) ** 2)[roi].astype(float)
    aa -= float(np.mean(aa))
    bb -= float(np.mean(bb))
    return float(np.dot(aa, bb) / max(np.linalg.norm(aa) * np.linalg.norm(bb), EPS))


def _relative_l2(a: np.ndarray, b: np.ndarray, roi: np.ndarray) -> float:
    aa = (np.abs(a) ** 2)[roi].astype(float)
    bb = (np.abs(b) ** 2)[roi].astype(float)
    scale_a = max(float(np.max(aa)), EPS)
    scale_b = max(float(np.max(bb)), EPS)
    aa /= scale_a
    bb /= scale_b
    return float(np.linalg.norm(aa - bb) / max(np.linalg.norm(bb), EPS))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-n", type=int, default=768)
    parser.add_argument("--output", type=Path, default=Path("outputs/validation/axicon_continuous_propagation_benchmark.json"))
    args = parser.parse_args()

    n = int(args.grid_n)
    z_values = (20e-3, 40e-3, 60e-3, 100e-3, 140e-3)
    rows: list[dict[str, float | str]] = []
    failures: list[str] = []
    for case_id in ("B0", "V1", "V3"):
        route = build_system_route(case_id, grid_n=n, config=SystemErrorConfig())
        grid = dict(route["grid"])
        wavelength = float(route["metadata"]["wavelength_m"])
        source = np.asarray(route["post_axicon"], dtype=np.complex128)
        fixed = build_fixed_support_spectrum(
            source,
            grid,
            wavelength_m=wavelength,
            z_max_m=max(z_values),
            minimum_retained_spectral_power=0.99,
        )

        padded_source = _pad_center(source, 2)
        padded_grid = make_xy_grid(2 * n, float(grid["dx"]))
        X = np.asarray(grid["X"], dtype=float)
        Y = np.asarray(grid["Y"], dtype=float)
        roi = (np.abs(X) <= 1.5e-3) & (np.abs(Y) <= 1.5e-3)

        for z in z_values:
            candidate = native_field_at_z(fixed, z)
            padded = angular_spectrum_propagate_bl(
                padded_source,
                padded_grid,
                wavelength,
                z,
                n_medium=1.0,
                bandlimit=False,
                include_evanescent=True,
            )
            reference = _crop_center(padded, n)
            corr = _intensity_correlation(candidate, reference, roi)
            rel = _relative_l2(candidate, reference, roi)
            rows.append(
                {
                    "case_id": case_id,
                    "z_m": float(z),
                    "retained_spectral_power_fraction": float(fixed.retained_spectral_power_fraction),
                    "central_roi_intensity_correlation": corr,
                    "central_roi_normalised_relative_l2": rel,
                }
            )
            if corr < 0.985:
                failures.append(f"{case_id} z={z*1e3:.0f} mm correlation={corr:.6f} < 0.985")
            if rel > 0.20:
                failures.append(f"{case_id} z={z*1e3:.0f} mm relative L2={rel:.6f} > 0.20")

    outcome = {
        "outcome": "AXICON-CONTINUOUS-PROPAGATION-BENCHMARK",
        "hard_pass": not failures,
        "candidate": "single max-z Matsushima support applied once; continuous exp(i*kz*z)",
        "reference": "2x centre-zero-padded unbandlimited angular spectrum; central 3 mm ROI",
        "grid_n": n,
        "rows": rows,
        "hard_failures": failures,
        "report_figures_authorised": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    print(json.dumps(outcome, indent=2))
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
