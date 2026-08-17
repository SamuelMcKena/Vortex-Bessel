from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from vbb_study.digital_twin.vortex_axicon_oblique_wave import (
    build_carrier_tracked_oblique_axicon_route,
)
from vbb_study.digital_twin.vortex_following_propagation import (
    build_beam_following_propagation,
)
from vbb_study.digital_twin.vortex_system_route import AxiconError, SystemErrorConfig


EPS = np.finfo(float).tiny


def _mean_width(intensity: np.ndarray, coordinate: np.ndarray) -> float:
    arr = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    power = np.sum(arr, axis=1)
    centre = np.sum(arr * coordinate[None, :], axis=1) / np.maximum(power, EPS)
    variance = np.sum(arr * (coordinate[None, :] - centre[:, None]) ** 2, axis=1) / np.maximum(power, EPS)
    width = np.sqrt(np.maximum(variance, 0.0))
    peak = np.max(arr, axis=1)
    active = peak >= 0.15 * max(float(np.max(peak)), EPS)
    return float(np.mean(width[active]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark carrier-tracked oblique thin-axicon wave morphology against independent Snell rays.")
    parser.add_argument("--grid-n", type=int, default=768)
    parser.add_argument("--output", type=Path, default=Path("outputs/validation/axicon_oblique_wave_benchmark_v2.json"))
    args = parser.parse_args()

    z = np.arange(10e-3, 130e-3 + 3e-3, 3e-3)
    offset = np.linspace(-220e-6, 220e-6, 321)
    rows: list[dict[str, float]] = []
    hard_failures: list[str] = []
    for degrees in (0.0, 5.0, 10.0):
        angle = math.radians(degrees)
        route = build_carrier_tracked_oblique_axicon_route(
            "B0",
            grid_n=int(args.grid_n),
            config=SystemErrorConfig(axicon=AxiconError(tilt_rad=(0.0, angle))),
        )
        follow = build_beam_following_propagation(
            grid=dict(route["grid"]),
            wavelength_m=float(route["metadata"]["wavelength_m"]),
            z_values_m=z,
            transverse_offsets_m=offset,
            scalar_field=route["post_axicon"],
            x_axis_m=0.0,
            y_axis_m=0.0,
            source_label=f"B0-oblique-{degrees:g}deg",
        )
        wx = _mean_width(follow.xz_intensity, offset)
        wy = _mean_width(follow.yz_intensity, offset)
        wave_anisotropy = abs(wx - wy) / max(0.5 * (wx + wy), EPS)
        ray = route["metadata"].get("independent_snell_ray_reference", {})
        ray_anisotropy = float(ray.get("cone_radius_anisotropy_fraction", 0.0))
        to_meta = route["metadata"].get("lab_to_tilted", {})
        from_meta = route["metadata"].get("tilted_to_lab", {})
        to_l2 = float(to_meta.get("spectral_power_ratio", 1.0))
        from_l2 = float(from_meta.get("spectral_power_ratio", 1.0))
        to_flux = float(to_meta.get("normal_flux_power_ratio", 1.0))
        from_flux = float(from_meta.get("normal_flux_power_ratio", 1.0))
        rows.append(
            {
                "tilt_deg": float(degrees),
                "xz_mean_width_m": wx,
                "yz_mean_width_m": wy,
                "wave_width_anisotropy_fraction": float(wave_anisotropy),
                "snell_ray_cone_anisotropy_fraction": ray_anisotropy,
                "lab_to_tilted_spectral_l2_ratio": to_l2,
                "tilted_to_lab_spectral_l2_ratio": from_l2,
                "lab_to_tilted_normal_flux_ratio": to_flux,
                "tilted_to_lab_normal_flux_ratio": from_flux,
                "roundtrip_raw_l2_product": to_l2 * from_l2,
            }
        )
        if min(to_flux, from_flux) < 0.985:
            hard_failures.append(
                f"{degrees:g} deg carrier-tracked normal-flux ratio < 0.985"
            )
        if abs(to_l2 * from_l2 - 1.0) > 0.01:
            hard_failures.append(
                f"{degrees:g} deg roundtrip raw-L2 projection product differs from unity by >1%"
            )

    ray_values = np.asarray([row["snell_ray_cone_anisotropy_fraction"] for row in rows])
    wave_values = np.asarray([row["wave_width_anisotropy_fraction"] for row in rows])
    if not (ray_values[2] > ray_values[1] > ray_values[0] - 1e-12):
        hard_failures.append("independent Snell-ray anisotropy is not monotonic with tilt")
    if wave_values[2] <= wave_values[0] + 1e-3:
        hard_failures.append(
            "10 deg wave morphology is effectively invariant relative to zero tilt"
        )
    if wave_values[2] < 0.002:
        hard_failures.append(
            f"10 deg wave width anisotropy={wave_values[2]:.6f} is too small to demonstrate oblique morphology"
        )

    outcome = {
        "outcome": "AXICON-OBLIQUE-WAVE-BENCHMARK-V2",
        "hard_pass": not hard_failures,
        "report_figures_authorised": False,
        "grid_n": int(args.grid_n),
        "angles_deg": [0.0, 5.0, 10.0],
        "power_invariant": "normal_flux_integral_|A|^2_fnormal",
        "rows": rows,
        "hard_failures": hard_failures,
        "policy": (
            "Passing establishes a non-invariant, carrier-safe scalar thin-axicon oblique response with the same trend as the independent Snell reference. "
            "Raw spectral L2 projection is recorded but is not treated as a conserved finite-tilt power. "
            "This does not replace full refractive-surface/vector validation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    print(json.dumps(outcome, indent=2))
    if hard_failures:
        raise SystemExit("; ".join(hard_failures))


if __name__ == "__main__":
    main()
