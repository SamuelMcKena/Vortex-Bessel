from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from vbb_study.digital_twin.vector_refractive_axicon import sample_vector_field_on_tilted_entrance
from vbb_study.digital_twin.vector_refractive_axicon_eikonal import (
    build_tilted_vector_refractive_axicon_field,
    estimate_common_vector_eikonal,
)
from vbb_study.digital_twin.vortex_refractive_axicon import (
    RefractiveAxiconGeometry,
    trace_refractive_axicon_bundle,
)
from vbb_study.digital_twin.vortex_rotated_plane import rotation_matrix
from vbb_study.equations.fields import make_xy_grid
from vbb_study.vector_field import VectorField


WAVELENGTH_M = 1.029e-6
AXICON_INDEX = 1.458
EXTERNAL_INDEX = 1.0
BASE_ANGLE_DEG = 2.0
CLEAR_RADIUS_M = 1.2e-3
CENTRE_THICKNESS_M = 2.0e-3
GRID_N = 128
WINDOW_M = 3.0e-3
OUTPUT_N = 256
OUTPUT_WINDOW_M = 3.0e-3
PUPIL_GUARD_FRACTION = 0.90
TILT_DEG = (0.0, 5.0, 10.0)


def _geometry() -> RefractiveAxiconGeometry:
    return RefractiveAxiconGeometry(
        base_angle_rad=math.radians(BASE_ANGLE_DEG),
        clear_radius_m=CLEAR_RADIUS_M,
        centre_thickness_m=CENTRE_THICKNESS_M,
        refractive_index=AXICON_INDEX,
        external_index=EXTERNAL_INDEX,
    )


def _plane_wave() -> VectorField:
    grid = make_xy_grid(GRID_N, WINDOW_M / GRID_N)
    ex = np.full((GRID_N, GRID_N), 1.0 / np.sqrt(2.0), dtype=np.complex128)
    ey = 1j * ex
    return VectorField(
        ex=ex,
        ey=ey,
        ez=np.zeros_like(ex),
        grid=grid,
        wavelength_m=WAVELENGTH_M,
        medium_index=EXTERNAL_INDEX,
    )


def _stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95.0)),
        "p99": float(np.percentile(arr, 99.0)),
        "max": float(np.max(arr)),
    }


def _entrance_eikonal(source: VectorField, tilt_rad: float):
    envelope, carrier, poynting, surface_meta = sample_vector_field_on_tilted_entrance(
        source,
        tilt_x_rad=0.0,
        tilt_y_rad=float(tilt_rad),
    )
    electric = np.stack([envelope.ex, envelope.ey, envelope.ez], axis=-1)
    rotation = rotation_matrix(0.0, float(tilt_rad))
    estimate = estimate_common_vector_eikonal(
        electric,
        poynting,
        envelope.grid,
        carrier_local_cpm=(float(carrier[0]), float(carrier[1])),
        rotation_local_to_lab=rotation,
        wavelength_m=WAVELENGTH_M,
        medium_index=EXTERNAL_INDEX,
    )
    exact_local = rotation.T @ np.asarray([0.0, 0.0, 1.0])
    return estimate, np.asarray(exact_local, dtype=float), surface_meta


def run_benchmark() -> dict[str, object]:
    geometry = _geometry()
    source = _plane_wave()
    X = np.asarray(source.grid["X"], dtype=float)
    Y = np.asarray(source.grid["Y"], dtype=float)
    R = np.asarray(source.grid["R"], dtype=float)
    guarded_pupil = R <= PUPIL_GUARD_FRACTION * geometry.clear_radius_m
    incident_lab = np.asarray([0.0, 0.0, 1.0])

    cases: list[dict[str, object]] = []
    hard_failures: list[str] = []
    for tilt_deg in TILT_DEG:
        tilt = math.radians(tilt_deg)
        estimate, exact_local, surface_meta = _entrance_eikonal(source, tilt)
        entrance_valid = np.asarray(estimate.valid_mask, dtype=bool) & guarded_pupil
        entrance_error = np.linalg.norm(
            np.asarray(estimate.direction_local, dtype=float) - exact_local,
            axis=-1,
        )[entrance_valid]

        vector = build_tilted_vector_refractive_axicon_field(
            source,
            geometry=geometry,
            tilt_x_rad=0.0,
            tilt_y_rad=tilt,
            reference_gap_m=0.20e-3,
            output_n=OUTPUT_N,
            output_window_m=OUTPUT_WINDOW_M,
        )
        scalar = trace_refractive_axicon_bundle(
            X,
            Y,
            geometry=geometry,
            tilt_x_rad=0.0,
            tilt_y_rad=tilt,
            incident_direction_lab=incident_lab,
            reference_gap_m=0.20e-3,
            apex_exclusion_radius_m=0.0,
        )
        common = (
            np.asarray(estimate.valid_mask, dtype=bool)
            & np.asarray(vector.geometry_bundle.valid, dtype=bool)
            & np.asarray(scalar.valid, dtype=bool)
        )
        vector_out = np.asarray(vector.outgoing_direction_lab, dtype=float)[common]
        scalar_out = np.asarray(scalar.outgoing_lab, dtype=float)[common]
        outgoing_error = np.linalg.norm(vector_out - scalar_out, axis=1)
        vector_exit = np.asarray(vector.geometry_bundle.exit_point_lab_m, dtype=float)[common]
        scalar_exit = np.asarray(scalar.exit_point_lab_m, dtype=float)[common]
        exit_error_m = np.linalg.norm(vector_exit - scalar_exit, axis=1)
        vector_distance = np.asarray(vector.geometry_bundle.internal_distance_m, dtype=float)[common]
        scalar_distance = np.asarray(scalar.internal_distance_m, dtype=float)[common]
        glass_path_error_m = np.abs(vector_distance - scalar_distance)
        entrance_common_error = np.linalg.norm(
            np.asarray(estimate.direction_local, dtype=float) - exact_local,
            axis=-1,
        )[common]

        entrance_stats = _stats(entrance_error)
        common_entrance_stats = _stats(entrance_common_error)
        outgoing_stats = _stats(outgoing_error)
        exit_stats = _stats(exit_error_m)
        path_stats = _stats(glass_path_error_m)

        floor = 2e-11
        if outgoing_stats["median"] > 2.0 * common_entrance_stats["median"] + floor:
            hard_failures.append(f"{tilt_deg:g}deg outgoing median exceeds 2x entrance error")
        if outgoing_stats["p99"] > 2.0 * common_entrance_stats["p99"] + floor:
            hard_failures.append(f"{tilt_deg:g}deg outgoing p99 exceeds 2x entrance error")
        if outgoing_stats["max"] > 2.0 * common_entrance_stats["max"] + floor:
            hard_failures.append(f"{tilt_deg:g}deg outgoing max exceeds 2x entrance error")

        cases.append(
            {
                "tilt_deg": tilt_deg,
                "guarded_entrance_sample_count": int(np.count_nonzero(entrance_valid)),
                "common_ray_sample_count": int(np.count_nonzero(common)),
                "entrance_eikonal_unit_vector_error": entrance_stats,
                "entrance_eikonal_error_on_common_rays": common_entrance_stats,
                "outgoing_vector_vs_scalar_unit_vector_error": outgoing_stats,
                "exit_point_error_m": exit_stats,
                "glass_path_error_m": path_stats,
                "common_eikonal_metadata": dict(estimate.metadata),
                "surface_sampling": surface_meta,
                "vector_solver_selected_metrics": {
                    "final_flux_closure_ratio": float(vector.metadata["final_flux_closure_ratio"]),
                    "final_transversality_residual": float(vector.metadata["final_transversality_residual"]),
                    "required_nyquist_fraction": float(vector.metadata["required_nyquist_fraction"]),
                    "ray_flux_closure_ratio": float(vector.metadata["ray_flux_closure_ratio"]),
                },
            }
        )

    return {
        "outcome": "PHASE2H-VECTOR-REFRACTIVE-REFERENCE-BENCHMARK",
        "hard_pass": not hard_failures,
        "hard_failures": hard_failures,
        "policy": {
            "ray_direction": "common_eikonal_grad_Phi_not_total_structured_Poynting",
            "reference": "independent_existing_scalar_two_surface_tracer_given_exact_incident_plane_wave",
            "entrance_eikonal_domain": "guarded_physical_axicon_pupil",
            "guarded_pupil_fraction": PUPIL_GUARD_FRACTION,
            "outgoing_error_gate": "median/p99/max <= 2x same-ray entrance-eikonal error + 2e-11",
            "report_figures_authorised": False,
        },
        "configuration": {
            "wavelength_m": WAVELENGTH_M,
            "axicon_index": AXICON_INDEX,
            "external_index": EXTERNAL_INDEX,
            "base_angle_deg": BASE_ANGLE_DEG,
            "clear_radius_m": CLEAR_RADIUS_M,
            "centre_thickness_m": CENTRE_THICKNESS_M,
            "grid_n": GRID_N,
            "window_m": WINDOW_M,
            "output_n": OUTPUT_N,
            "output_window_m": OUTPUT_WINDOW_M,
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/validation/phase2h/vector_refractive_reference_benchmark.json"),
    )
    args = parser.parse_args()
    result = run_benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["hard_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
