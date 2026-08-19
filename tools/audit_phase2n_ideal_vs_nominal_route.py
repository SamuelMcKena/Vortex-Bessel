"""Numerically audit that the continuous-ideal and nominal bench routes differ.

This is a validation diagnostic, not a presentation renderer.  It quantifies
where the nominal route departs from the continuous ideal for B0/V1/V3:
SLM transfer, finite 4F order selection, field overlap at the axicon plane, and
raw versus shape-only differences at z=60 mm.

The nominal route is deliberately described as *bench-constrained*, not a fully
calibrated bench prediction: measured SLM LUT/static maps, exact 4F geometry,
lens OPD, axicon surface/clear aperture and camera calibration are still absent.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from build_phase2l_nominal_constraints import XY_HALF_M, _crop, _xy, continuous_ideal
from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.vortex_system_route import build_system_route
from vbb_study.slm_model import field_power


CASES = ("B0", "V1", "V3")
EPS = np.finfo(float).tiny


def _power(field: np.ndarray, route: dict) -> float:
    return field_power(np.asarray(field, dtype=np.complex128), route["grid"])


def _complex_overlap(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.complex128).ravel()
    bb = np.asarray(b, dtype=np.complex128).ravel()
    den = float(np.vdot(aa, aa).real * np.vdot(bb, bb).real)
    if den <= EPS:
        return 0.0
    return float(abs(np.vdot(aa, bb)) ** 2 / den)


def _peak_normalise(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    return values / max(float(np.max(values)), EPS)


def audit_case(case_id: str, grid_n: int) -> dict[str, float | str]:
    ideal = continuous_ideal(case_id, grid_n)
    nominal = build_system_route(case_id, grid_n=grid_n)

    p_input = _power(nominal["input_beam"], nominal)
    p_slm1 = _power(nominal["post_slm1"], nominal)
    p_slm2 = _power(nominal["post_slm2"], nominal)
    p_4f = _power(nominal["post_4f_selected_order"], nominal)
    p_ax_nominal = _power(nominal["post_axicon"], nominal)
    p_ax_ideal = _power(ideal["post_axicon"], ideal)

    # Continuous ideal immediately before the physical axicon: same Gaussian
    # input field, continuous vortex phase, no SLM or 4F constraints.
    ideal_pre_axicon = np.asarray(nominal["input_beam"], dtype=np.complex128) * np.exp(
        1j * np.asarray(ideal["phase"], dtype=float)
    )
    nominal_pre_axicon = np.asarray(nominal["field_on_axicon_plane"], dtype=np.complex128)

    ideal_xy_raw = np.maximum(np.asarray(_xy(ideal), dtype=float), 0.0)
    nominal_xy_raw = np.maximum(np.asarray(_xy(nominal), dtype=float), 0.0)
    ideal_peak = max(float(np.max(ideal_xy_raw)), EPS)
    nominal_peak = float(np.max(nominal_xy_raw))

    ideal_crop, _ = _crop(ideal_xy_raw, ideal["grid"], half=XY_HALF_M)
    nominal_crop, _ = _crop(nominal_xy_raw, nominal["grid"], half=XY_HALF_M)
    roi_ratio = float(np.sum(nominal_crop)) / max(float(np.sum(ideal_crop)), EPS)

    ideal_shape = _peak_normalise(ideal_xy_raw)
    nominal_shape = _peak_normalise(nominal_xy_raw)
    shape_max_abs = float(np.max(np.abs(nominal_shape - ideal_shape)))
    shape_rms = float(np.sqrt(np.mean((nominal_shape - ideal_shape) ** 2)))

    fourf_meta = nominal["metadata"]["fourf"]

    return {
        "case_id": case_id,
        "grid_n": int(grid_n),
        "grid_dx_um": float(nominal["grid"]["dx"]) * 1e6,
        "slm_pixel_pitch_um": float(hardware_value(canonical_hardware_manifest(), "slm_pixel_pitch_m")) * 1e6,
        "input_power": p_input,
        "post_slm1_power": p_slm1,
        "post_slm2_power": p_slm2,
        "post_4f_selected_power": p_4f,
        "post_axicon_nominal_power": p_ax_nominal,
        "post_axicon_ideal_power": p_ax_ideal,
        "slm1_power_ratio_to_input": p_slm1 / max(p_input, EPS),
        "slm2_power_ratio_to_slm1": p_slm2 / max(p_slm1, EPS),
        "two_slm_power_ratio_to_input": p_slm2 / max(p_input, EPS),
        "fourf_power_ratio_to_slm2": p_4f / max(p_slm2, EPS),
        "fourf_iris_selected_fraction": float(fourf_meta["iris_selected_power_fraction"]),
        "nominal_pre_axicon_power_ratio_to_continuous_ideal": p_4f / max(p_input, EPS),
        "pre_axicon_complex_overlap_with_continuous_ideal": _complex_overlap(
            ideal_pre_axicon, nominal_pre_axicon
        ),
        "post_axicon_power_ratio_nominal_over_ideal": p_ax_nominal / max(p_ax_ideal, EPS),
        "z60_peak_ratio_nominal_over_ideal": nominal_peak / ideal_peak,
        "z60_central_roi_integrated_ratio_nominal_over_ideal": roi_ratio,
        "z60_shape_only_max_abs_difference": shape_max_abs,
        "z60_shape_only_rms_difference": shape_rms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-n", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [audit_case(case_id, args.grid_n) for case_id in CASES]
    manifest = canonical_hardware_manifest()
    payload = {
        "claim": "continuous_ideal_and_nominal_bench_constrained_routes_are_physically_distinct",
        "continuous_ideal_route": [
            "Gaussian input",
            "continuous vortex phase",
            "ideal sharp physical axicon",
            "free-space propagation",
        ],
        "nominal_bench_constrained_route": [
            "Gaussian input",
            "8 um SLM pixelation",
            "256-level phase quantisation",
            "93% fill-factor throughput on each SLM",
            "20-pixel carrier/blaze",
            "explicit 300 mm 4F relay",
            "finite +1-order Fourier iris",
            "carrier removal in selected-order frame",
            "same ideal sharp physical axicon",
            "free-space propagation",
        ],
        "calibration_status": {
            "nominal_fixed_parameter_simulation_ready": manifest["nominal_fixed_parameter_simulation_ready"],
            "fixed_bench_prediction_ready": manifest["fixed_bench_prediction_ready"],
            "blocker": manifest["fixed_bench_prediction_blocker"],
            "calibration_required_parameters": manifest["calibration_required_parameters"],
        },
        "cases": rows,
    }

    json_path = args.output_dir / "05_ideal_vs_nominal_route_audit.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = args.output_dir / "05_ideal_vs_nominal_route_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
