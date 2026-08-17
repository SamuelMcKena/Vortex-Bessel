from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.phase2e_spectral_propagation import build_dense_spectral_propagation
from vbb_study.digital_twin.vortex_axicon_oblique_reference import oblique_refractive_axicon_rays
from vbb_study.digital_twin.vortex_system_route import AxiconError, SystemErrorConfig, build_system_route


EPS = np.finfo(float).tiny


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float).ravel().copy()
    bb = np.asarray(b, dtype=float).ravel().copy()
    aa -= float(np.mean(aa))
    bb -= float(np.mean(bb))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= EPS:
        return 1.0 if np.allclose(aa, bb) else 0.0
    return float(np.dot(aa, bb) / denom)


def _mean_line_width(
    intensity_zq: np.ndarray,
    coordinate_m: np.ndarray,
) -> float:
    I = np.maximum(np.asarray(intensity_zq, dtype=float), 0.0)
    q = np.asarray(coordinate_m, dtype=float)
    line_power = np.sum(I, axis=1)
    centre = np.sum(I * q[None, :], axis=1) / np.maximum(line_power, EPS)
    variance = (
        np.sum(I * (q[None, :] - centre[:, None]) ** 2, axis=1)
        / np.maximum(line_power, EPS)
    )
    width = np.sqrt(np.maximum(variance, 0.0))
    peak = np.max(I, axis=1)
    active = peak >= 0.15 * max(float(np.max(peak)), EPS)
    if not np.any(active):
        return float("nan")
    return float(np.mean(width[active]))


def _axicon_transform_min_ratio(route: dict[str, Any]) -> float:
    meta = route["metadata"]
    return float(
        min(
            float(meta.get("lab_to_tilted", {}).get("spectral_power_ratio", 1.0)),
            float(meta.get("tilted_to_lab", {}).get("spectral_power_ratio", 1.0)),
        )
    )


def run_benchmark(
    *,
    case_id: str,
    grid_n: int,
    angles_deg: list[float],
) -> dict[str, Any]:
    manifest = canonical_hardware_manifest()
    wavelength = float(hardware_value(manifest, "wavelength_m"))
    gamma = math.radians(float(hardware_value(manifest, "axicon_base_angle_deg")))
    n_ax = float(hardware_value(manifest, "axicon_refractive_index"))
    n_ext = float(hardware_value(manifest, "axicon_external_medium_index"))

    # The current tilted-plane backend restores the destination carrier on the
    # sampled grid.  Reject benchmark angles that are not actually representable
    # rather than accepting aliased large-angle results.
    window_m = 10.0e-3
    dx_m = window_m / int(grid_n)
    nyquist_cpm = 0.5 / dx_m

    z = np.arange(5e-3, 140e-3 + 2.5e-3, 2.5e-3)
    transverse = np.linspace(-1.25e-3, 1.25e-3, 401)

    wave: dict[float, dict[str, Any]] = {}
    ray: dict[float, dict[str, float]] = {}

    for angle_deg in angles_deg:
        angle = math.radians(float(angle_deg))
        carrier_cpm = abs(math.sin(angle)) / wavelength
        if carrier_cpm >= 0.8 * nyquist_cpm:
            raise ValueError(
                f"{angle_deg:g} deg oblique benchmark requires {carrier_cpm:.1f} cpm, "
                f"too close to/exceeding 80% of grid Nyquist {nyquist_cpm:.1f} cpm"
            )

        route = build_system_route(
            case_id,
            grid_n=int(grid_n),
            config=SystemErrorConfig(axicon=AxiconError(tilt_rad=(0.0, angle))),
        )
        dense = build_dense_spectral_propagation(
            grid=route["grid"],
            wavelength_m=wavelength,
            z_values_m=z,
            transverse_coordinates_m=transverse,
            scalar_field=route["post_axicon"],
            source_label=f"{case_id}:oblique_axicon_y={angle_deg:g}deg",
        )
        xz = np.asarray(dense.xz_intensity, dtype=float)
        yz = np.asarray(dense.yz_intensity, dtype=float)
        wave[float(angle_deg)] = {
            "xz": xz,
            "yz": yz,
            "xz_mean_width_m": _mean_line_width(xz, transverse),
            "yz_mean_width_m": _mean_line_width(yz, transverse),
            "min_rotated_plane_spectral_power_ratio": _axicon_transform_min_ratio(route),
        }

        ref = oblique_refractive_axicon_rays(
            base_angle_rad=gamma,
            refractive_index=n_ax,
            external_index=n_ext,
            tilt_y_rad=angle,
            azimuth_samples=720,
        )
        ray[float(angle_deg)] = {
            "cone_radius_mean": float(ref.cone_radius_mean),
            "cone_radius_anisotropy_fraction": float(ref.cone_radius_anisotropy_fraction),
            "second_harmonic_fraction": float(ref.second_harmonic_fraction),
        }

    zero_key = min(wave, key=lambda value: abs(value))
    if abs(zero_key) > 1e-12:
        raise ValueError("angles_deg must include 0 deg")

    nominal_xz = wave[zero_key]["xz"]
    nominal_yz = wave[zero_key]["yz"]
    rows: list[dict[str, Any]] = []
    for angle_deg in sorted(wave):
        item = wave[angle_deg]
        wx = float(item["xz_mean_width_m"])
        wy = float(item["yz_mean_width_m"])
        width_anisotropy = abs(wx - wy) / max(0.5 * (wx + wy), EPS)
        rows.append(
            {
                "angle_deg": float(angle_deg),
                "required_carrier_cpm": float(abs(math.sin(math.radians(angle_deg))) / wavelength),
                "xz_corr_zero": _correlation(item["xz"], nominal_xz),
                "yz_corr_zero": _correlation(item["yz"], nominal_yz),
                "xz_yz_corr": _correlation(item["xz"], item["yz"]),
                "xz_mean_width_m": wx,
                "yz_mean_width_m": wy,
                "width_anisotropy_fraction": float(width_anisotropy),
                "min_rotated_plane_spectral_power_ratio": float(
                    item["min_rotated_plane_spectral_power_ratio"]
                ),
                **ray[angle_deg],
            }
        )

    nonzero = [row for row in rows if abs(float(row["angle_deg"])) > 1e-12]
    hard_failures: list[str] = []

    min_numerical_ratio = min(
        float(row["min_rotated_plane_spectral_power_ratio"]) for row in rows
    )
    if min_numerical_ratio < 0.985:
        hard_failures.append(
            "tilted-plane numerical power gate failed in oblique benchmark: "
            f"minimum={min_numerical_ratio:.6f}"
        )

    ray_anisotropy = [float(row["cone_radius_anisotropy_fraction"]) for row in rows]
    if any(b + 1e-12 < a for a, b in zip(ray_anisotropy, ray_anisotropy[1:])):
        hard_failures.append("two-interface Snell reference anisotropy is not monotonic with tilt")

    if nonzero:
        endpoint = max(nonzero, key=lambda row: abs(float(row["angle_deg"])))
        morphology_delta = max(
            1.0 - float(endpoint["xz_corr_zero"]),
            1.0 - float(endpoint["yz_corr_zero"]),
            1.0 - float(endpoint["xz_yz_corr"]),
            float(endpoint["width_anisotropy_fraction"]),
        )
        # This is intentionally an anti-invariance gate, not a fitted
        # quantitative threshold.  Thaning et al. and Dudutis et al. establish
        # that oblique axicon illumination introduces astigmatic broadening;
        # therefore a moderate, resolvable tilt must not remain numerically
        # indistinguishable from the zero-tilt field.
        if morphology_delta <= 1e-4:
            hard_failures.append(
                "oblique axicon wave model is effectively invariant at the largest "
                f"tested tilt ({endpoint['angle_deg']:g} deg); astigmatic response absent"
            )
    else:
        morphology_delta = 0.0
        hard_failures.append("oblique benchmark requires at least one non-zero tilt")

    return {
        "outcome": "OBLIQUE-AXICON-REFERENCE-BENCHMARK",
        "hard_pass": not hard_failures,
        "report_figures_authorised": False,
        "case_id": case_id,
        "grid_n": int(grid_n),
        "window_m": window_m,
        "nyquist_cpm": float(nyquist_cpm),
        "axicon_base_angle_deg": float(math.degrees(gamma)),
        "axicon_base_angle_provenance": "Phase 2A canonical manifest; calibration_required",
        "rows": rows,
        "minimum_rotated_plane_spectral_power_ratio": float(min_numerical_ratio),
        "endpoint_morphology_delta": float(morphology_delta),
        "hard_failures": hard_failures,
        "interpretation": (
            "The two-interface Snell ray bundle is an independent qualitative reference. "
            "Passing proves only that the current scalar wave route is numerically stable "
            "and not spuriously invariant under resolvable oblique illumination. It does "
            "not validate an absolute tilted refractive-axicon prediction."
        ),
        "references": [
            "Bin & Zhu, Applied Optics 37, 2563-2568 (1998)",
            "Thaning, Jaroszewicz & Friberg, Applied Optics 42, 9-17 (2003)",
            "Dudutis et al., Optics Express 26, 3627-3637 (2018)",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check oblique axicon physics against an independent Snell reference.")
    parser.add_argument("--case", default="B0")
    parser.add_argument("--grid-n", type=int, default=1024)
    parser.add_argument("--angles-deg", nargs="+", type=float, default=[0.0, 1.0, 2.0])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/validation/oblique_axicon_benchmark/benchmark.json"),
    )
    args = parser.parse_args()

    result = run_benchmark(
        case_id=str(args.case),
        grid_n=int(args.grid_n),
        angles_deg=[float(v) for v in args.angles_deg],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["hard_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
