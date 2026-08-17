from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from vbb_study.digital_twin.phase2a_contracts import (
    canonical_hardware_manifest,
    hardware_value,
)
from vbb_study.digital_twin.phase2e_spectral_propagation import on_axis_spectral_intensity
from vbb_study.digital_twin.vortex_axicon_tip_reference import (
    normalised_intensity,
    production_style_tip_phase_rad,
    radial_fresnel_field,
    shallow_exact_phase_gradient_relative_error,
    tip_resolution,
)
from vbb_study.equations.fields import make_xy_grid


EPS = np.finfo(float).tiny


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float).copy()
    bb = np.asarray(b, dtype=float).copy()
    aa -= float(np.mean(aa))
    bb -= float(np.mean(bb))
    return float(np.dot(aa, bb) / max(np.linalg.norm(aa) * np.linalg.norm(bb), EPS))


def _two_dimensional_trace(
    *,
    grid_n: int,
    window_m: float,
    wavelength_m: float,
    beam_radius_m: float,
    gamma: float,
    n_ax: float,
    n_ext: float,
    z_m: np.ndarray,
    tip_model: str,
    tip_radius_m: float,
) -> tuple[np.ndarray, float]:
    grid = make_xy_grid(int(grid_n), float(window_m) / int(grid_n))
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    R = np.hypot(X, Y)
    phase = production_style_tip_phase_rad(
        R,
        wavelength_m=wavelength_m,
        base_angle_rad=gamma,
        refractive_index=n_ax,
        external_index=n_ext,
        tip_model=tip_model,
        tip_radius_m=tip_radius_m,
    )
    field = np.exp(-(R * R) / (beam_radius_m * beam_radius_m)) * np.exp(1j * phase)
    trace = on_axis_spectral_intensity(
        grid=grid,
        wavelength_m=wavelength_m,
        z_values_m=z_m,
        scalar_field=field,
        n_medium=n_ext,
        bandlimit=True,
    )
    return normalised_intensity(np.sqrt(np.maximum(trace, 0.0))), float(grid["dx"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate 2-D axicon-tip ASM against an independent radial Fresnel integral.")
    parser.add_argument("--grid-n", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=Path("outputs/validation/axicon_tip_reference_benchmark.json"))
    args = parser.parse_args()

    manifest = canonical_hardware_manifest()
    wavelength = float(hardware_value(manifest, "wavelength_m"))
    beam_radius = float(hardware_value(manifest, "beam_radius_on_slm_m"))
    gamma = math.radians(float(hardware_value(manifest, "axicon_base_angle_deg")))
    n_ax = float(hardware_value(manifest, "axicon_refractive_index"))
    n_ext = float(hardware_value(manifest, "axicon_external_medium_index"))
    window = 10.0e-3
    z = np.linspace(20e-3, 120e-3, 81)

    cases = (
        ("sharp", 0.0),
        ("hyperbolic_round", 200e-6),
        ("flat_blunt", 200e-6),
    )
    rows: list[dict[str, float | str | bool]] = []
    hard_failures: list[str] = []
    for tip_model, radius in cases:
        two_d, dx = _two_dimensional_trace(
            grid_n=int(args.grid_n),
            window_m=window,
            wavelength_m=wavelength,
            beam_radius_m=beam_radius,
            gamma=gamma,
            n_ax=n_ax,
            n_ext=n_ext,
            z_m=z,
            tip_model=tip_model,
            tip_radius_m=radius,
        )
        radial = radial_fresnel_field(
            radial_observation_m=[0.0],
            z_values_m=z,
            wavelength_m=wavelength,
            beam_radius_m=beam_radius,
            base_angle_rad=gamma,
            refractive_index=n_ax,
            external_index=n_ext,
            vortex_charge=0,
            tip_model=tip_model,
            tip_radius_m=radius,
            radial_step_m=0.5e-6,
            phase_model="production_style",
        )[:, 0]
        radial_i = normalised_intensity(radial)
        corr = _corr(two_d, radial_i)
        rms = float(np.sqrt(np.mean((two_d - radial_i) ** 2)))
        resolution = tip_resolution(radius, dx, minimum_pixels=8.0)
        rows.append(
            {
                "tip_model": tip_model,
                "tip_radius_m": float(radius),
                "tip_radius_pixels": float(resolution.radius_pixels),
                "resolved_at_8px": bool(resolution.resolved),
                "axial_trace_correlation": float(corr),
                "axial_trace_rms_difference": float(rms),
            }
        )
        if radius != 0.0 and not resolution.resolved:
            hard_failures.append(f"{tip_model} radius is under-resolved on the benchmark grid")
        if corr < 0.97:
            hard_failures.append(f"{tip_model} radial-Fresnel/2-D-ASM correlation={corr:.6f} < 0.97")
        if rms > 0.15:
            hard_failures.append(f"{tip_model} radial-Fresnel/2-D-ASM RMS={rms:.6f} > 0.15")

    shallow_error = shallow_exact_phase_gradient_relative_error(
        base_angle_rad=gamma,
        refractive_index=n_ax,
        external_index=n_ext,
    )
    if shallow_error > 0.01:
        hard_failures.append(
            f"declared axicon angle is outside the authorised shallow-tip regime: gradient error={shallow_error:.4%}"
        )

    outcome = {
        "outcome": "AXICON-TIP-REFERENCE-BENCHMARK",
        "hard_pass": not hard_failures,
        "report_figures_authorised": False,
        "grid_n": int(args.grid_n),
        "window_m": window,
        "wavelength_m": wavelength,
        "beam_radius_m": beam_radius,
        "axicon_base_angle_deg": math.degrees(gamma),
        "shallow_exact_cone_gradient_relative_error": float(shallow_error),
        "cases": rows,
        "hard_failures": hard_failures,
        "policy": (
            "This validates the shallow-angle scalar tip operator only. Physical tip radii remain sensitivity values "
            "until profilometry is supplied; high-angle refractive use remains blocked."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    print(json.dumps(outcome, indent=2))
    if hard_failures:
        raise SystemExit("; ".join(hard_failures))


if __name__ == "__main__":
    main()
