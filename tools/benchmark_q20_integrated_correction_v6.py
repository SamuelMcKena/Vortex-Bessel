"""Final synthetic q=20 correction benchmark for poster evidence.

The hidden state contains two physically observable system errors plus a smooth
residual angular wavefront on SLM1. Physical parameters are estimated from
parameter-specific intensity observables. The remaining wavefront is fitted on
alternating illuminated z planes by propagating candidate phases through the
complete digital twin.

Crucially, the recovered residual is not removed on the same synthetic plane on
which it was injected. The compensating phase is applied as an additive SLM2
phase layer in the actual dual-SLM -> explicit 4F -> axicon route. In the current
concatenated-SLM numerical contract SLM1 and SLM2 share the same transverse
coordinates before the 4F relay, so -psi on SLM2 is the model-space hardware
correction for +psi on SLM1. Experimental use still requires the measured SLM2
coordinate/LUT calibration described by the q20 hardware pipeline.

The Miao result is retained as a favourable analytical baseline at its native
axicon-input plane; it is not relabelled as an already calibrated SLM2 mask.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import benchmark_q20_integrated_correction_v5 as v5  # noqa: E402
import benchmark_q20_miao_vs_digital_twin as base  # noqa: E402
import benchmark_q20_method_physics_v2 as p2  # noqa: E402
import benchmark_q20_method_physics_v4 as p4  # noqa: E402
import benchmark_q20_full_model_phase_refinement_v1 as full  # noqa: E402
from vbb_study.digital_twin.vortex_system_error_sweeps import system_sweep_registry  # noqa: E402
from vbb_study.digital_twin.vortex_system_route import SystemErrorConfig, build_system_route  # noqa: E402

OUT = ROOT / "outputs" / "validation" / "q20_integrated_correction_v6"
Z_FIT = v5.Z_FIT
Z_EVAL = v5.Z_EVAL
GRID_N = v5.GRID_N
WINDOW_M = v5.WINDOW_M
PHYSICAL_CROP_HALF_M = v5.PHYSICAL_CROP_HALF_M
NOISE_SIGMA = v5.NOISE_SIGMA
SEED = 2061
EPS = np.finfo(float).tiny

# Keep imported benchmark helpers on the same numerical contract.
base.GRID_N = GRID_N
base.WINDOW_M = WINDOW_M
base.Z_FIT_M = Z_FIT
base.Z_DISPLAY_M = Z_EVAL
base.NOISE_SIGMA = NOISE_SIGMA
base.FIT_CROP_HALF_M = PHYSICAL_CROP_HALF_M
base.METRIC_RADIUS_M = p2.METRIC_RADIUS_M
full.Z_WIDE = Z_FIT
full.Z_EVAL = Z_EVAL


def route_with_layers(
    config: SystemErrorConfig,
    *,
    residual_slm1_rad: np.ndarray | None = None,
    correction_slm2_rad: np.ndarray | None = None,
) -> dict:
    return build_system_route(
        "V20",
        grid_n=GRID_N,
        window_m=WINDOW_M,
        config=config,
        slm1_static_phase_map_rad=residual_slm1_rad,
        slm2_static_phase_map_rad=correction_slm2_rad,
    )


def _pair(a: np.ndarray, b: np.ndarray, roi: np.ndarray) -> tuple[float, float]:
    av = np.asarray(a, float)[roi]
    bv = np.asarray(b, float)[roi]
    av = av / max(float(np.max(av)), EPS)
    bv = bv / max(float(np.max(bv)), EPS)
    return float(np.corrcoef(av, bv)[0, 1]), float(np.sqrt(np.mean((av - bv) ** 2)))


def build(out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    registry = system_sweep_registry()
    nominal = SystemErrorConfig()

    ideal_route = route_with_layers(nominal)
    grid = ideal_route["grid"]
    residual, truth_coeff = v5._truth_phase(grid)

    truth = base.apply_registry_family(nominal, "fourf_iris_radius_scale", 0.85, registry=registry)
    truth = base.apply_registry_family(truth, "axicon_lateral_decentre_x", 250e-6, registry=registry)

    distorted_route = route_with_layers(truth, residual_slm1_rad=residual)
    distorted_clean = base._propagate(distorted_route, Z_FIT)
    rng = np.random.default_rng(SEED)
    distorted_measured = base._add_noise(distorted_clean, rng.normal(size=distorted_clean.shape))

    x = np.asarray(grid["x"], float)
    crop = np.flatnonzero(np.abs(x) <= PHYSICAL_CROP_HALF_M)
    target_crop = distorted_measured[:, crop[:, None], crop]
    simulator = base.PhysicalSimulator(crop)
    estimated, physical_diag = v5._physical_fit(target_crop, simulator, registry)

    # Physical alignment/aperture correction followed by a fresh scan.
    adjusted = base._compensate_physical_parameters(truth, estimated)
    adjusted_route = route_with_layers(adjusted, residual_slm1_rad=residual)
    adjusted_clean = base._propagate(adjusted_route, Z_FIT)
    rng2 = np.random.default_rng(SEED + 1)
    adjusted_measured = base._add_noise(adjusted_clean, rng2.normal(size=adjusted_clean.shape))

    direct_ids, direct_illum = p4.illuminated_plane_indices(distorted_route, Z_FIT)
    miao_direct_phase, miao_direct_diag = p4.miao_correction_selected(
        distorted_route, distorted_measured, Z_FIT, direct_ids
    )
    adjusted_ids, adjusted_illum = p4.illuminated_plane_indices(adjusted_route, Z_FIT)
    miao_hybrid_phase, miao_hybrid_diag = p4.miao_correction_selected(
        adjusted_route, adjusted_measured, Z_FIT, adjusted_ids
    )

    train = adjusted_ids[::2]
    held = adjusted_ids[1::2]
    if len(held) < 2:
        train = adjusted_ids[:-2]
        held = adjusted_ids[-2:]

    coeff, refine_diag = full.refine_phase_full_model(adjusted, adjusted_measured, Z_FIT, train)
    estimated_phase = full.phase_from_coefficients(grid, coeff)
    held_score = full._heldout_model_score(adjusted, coeff, adjusted_measured, Z_FIT, held)
    truth_held = full._heldout_model_score(adjusted, truth_coeff, adjusted_measured, Z_FIT, held)

    # Final hardware-like model path: residual stays on SLM1 and the negative
    # recovered residual is added to SLM2 before the 4F relay.
    full_corrected_route = route_with_layers(
        adjusted,
        residual_slm1_rad=residual,
        correction_slm2_rad=-estimated_phase,
    )
    oracle_slm2_route = route_with_layers(
        adjusted,
        residual_slm1_rad=residual,
        correction_slm2_rad=-residual,
    )

    ideal_eval = base._propagate(ideal_route, Z_EVAL)
    distorted_eval = base._propagate(distorted_route, Z_EVAL)
    adjusted_eval = base._propagate(adjusted_route, Z_EVAL)
    miao_direct_eval = base._propagate(distorted_route, Z_EVAL, miao_direct_phase)
    miao_hybrid_eval = base._propagate(adjusted_route, Z_EVAL, miao_hybrid_phase)
    full_eval = base._propagate(full_corrected_route, Z_EVAL)
    oracle_eval = base._propagate(oracle_slm2_route, Z_EVAL)

    X = np.asarray(grid["X"], float)
    Y = np.asarray(grid["Y"], float)
    roi = np.hypot(X, Y) <= p2.METRIC_RADIUS_M
    stacks = {
        "distorted": distorted_eval,
        "physical_adjustment_only": adjusted_eval,
        "miao_only_native_input": miao_direct_eval,
        "physical_fit_plus_miao_native_input": miao_hybrid_eval,
        "physical_fit_plus_full_model_slm2": full_eval,
        "oracle_slm2": oracle_eval,
    }
    rows = []
    for iz, z in enumerate(Z_EVAL):
        row = {"z_mm": float(z * 1e3)}
        for name, stack in stacks.items():
            r, e = _pair(stack[iz], ideal_eval[iz], roi)
            row[f"{name}_pearson_r"] = r
            row[f"{name}_nrmse"] = e
        rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics_vs_z.csv", index=False)

    phase_delta = np.angle(np.exp(1j * (estimated_phase - residual)))
    phase_rms = float(np.sqrt(np.mean(phase_delta**2)))

    def avg(name: str) -> dict:
        return {
            "mean_pearson_r": float(metrics[f"{name}_pearson_r"].mean()),
            "mean_nrmse": float(metrics[f"{name}_nrmse"].mean()),
        }

    summary = {
        "study": "integrated q20 physical-model fit and SLM2 residual correction v6",
        "truth": {
            "fourf_iris_radius_scale": 0.85,
            "axicon_lateral_decentre_x_m": 250e-6,
            "residual_phase_coefficients": full._phase_coeff_table(truth_coeff, truth_coeff),
        },
        "physical_parameter_estimation": physical_diag,
        "correction_planes": {
            "full_model_estimation_plane": "SLM1/input transverse coordinates in current digital twin",
            "full_model_application_plane": "additive SLM2 phase layer before explicit 4F relay",
            "miao_baseline_application_plane": "native axicon-input analytical baseline; not claimed as calibrated SLM2 map",
            "experimental_boundary": "real SLM2 deployment requires measured coordinate mapping, parity/rotation/scale and 1030-nm phase LUT",
        },
        "retrieval_plane_selection": {
            "criterion": "stationary-phase input-annulus amplitude >= 0.15 of scan maximum",
            "direct_miao_z_mm": [float(Z_FIT[i] * 1e3) for i in direct_ids],
            "model_assisted_z_mm": [float(Z_FIT[i] * 1e3) for i in adjusted_ids],
            "full_model_train_z_mm": [float(Z_FIT[i] * 1e3) for i in train],
            "full_model_heldout_z_mm": [float(Z_FIT[i] * 1e3) for i in held],
            "direct_relative_annulus_amplitude": [float(v) for v in direct_illum],
            "adjusted_relative_annulus_amplitude": [float(v) for v in adjusted_illum],
        },
        "miao_only": miao_direct_diag,
        "physical_fit_plus_miao": miao_hybrid_diag,
        "full_model_residual_fit": {
            **refine_diag,
            "phase_rms_to_truth_rad": phase_rms,
            "coefficient_table": full._phase_coeff_table(coeff, truth_coeff),
            "heldout": held_score,
            "truth_model_heldout": truth_held,
        },
    }
    for name in stacks:
        summary[name] = avg(name)
    summary["gain_full_model_slm2_over_miao_only"] = {
        "mean_pearson_r": summary["physical_fit_plus_full_model_slm2"]["mean_pearson_r"]
        - summary["miao_only_native_input"]["mean_pearson_r"],
        "mean_nrmse": summary["physical_fit_plus_full_model_slm2"]["mean_nrmse"]
        - summary["miao_only_native_input"]["mean_nrmse"],
    }

    np.save(out / "truth_residual_phase_slm1_rad.npy", residual.astype(np.float32))
    np.save(out / "estimated_residual_phase_slm1_rad.npy", estimated_phase.astype(np.float32))
    np.save(out / "slm2_correction_phase_rad.npy", (-estimated_phase).astype(np.float32))
    np.save(out / "miao_only_correction_axicon_input_rad.npy", miao_direct_phase.astype(np.float32))
    np.save(out / "physical_fit_plus_miao_correction_axicon_input_rad.npy", miao_hybrid_phase.astype(np.float32))
    np.savez_compressed(
        out / "comparison_eval_stacks.npz",
        x_m=np.asarray(grid["x"], float),
        z_m=np.asarray(Z_EVAL, float),
        ideal=ideal_eval.astype(np.float32),
        distorted=distorted_eval.astype(np.float32),
        physical_adjustment=adjusted_eval.astype(np.float32),
        miao_only=miao_direct_eval.astype(np.float32),
        physical_plus_miao=miao_hybrid_eval.astype(np.float32),
        full_model_slm2=full_eval.astype(np.float32),
        oracle_slm2=oracle_eval.astype(np.float32),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    build()
