"""Piston-invariant coordinate calibration for the q=20 method comparison.

The v2 benchmark correctly recovered the injected 4F-iris radius and axicon
position, but its known-phase coordinate check compared absolute phase.  Global
phase piston is unobservable and physically irrelevant, so it must be removed
before judging sign/rotation/parity.  This wrapper replaces that calibration
with a piston-invariant circular phase-shape error and tests the complete square
rotation/parity set, with both phase signs.

The selected transform is fixed once on a separate known-phase calibration case
and then used unchanged for both Miao-only and digital-twin-assisted branches.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import benchmark_q20_miao_vs_digital_twin_v2 as v2  # noqa: E402
from vbb_study.digital_twin.vortex_system_route import SystemErrorConfig  # noqa: E402

OUT = ROOT / "outputs" / "poster" / "q20_method_comparison_v3"


def _parse_transform(name: str) -> tuple[int, bool, float]:
    parts = name.split("_")
    rot = int(parts[0][1:])
    mirror = "mirror" in parts
    sign = -1.0 if "neg" in parts else 1.0
    return rot, mirror, sign


def transform_phase(phase: np.ndarray, name: str) -> np.ndarray:
    rot, mirror, sign = _parse_transform(name)
    p = np.rot90(np.asarray(phase, float), k=(rot//90) % 4)
    if mirror:
        p = np.fliplr(p)
    return sign*p


def _phase_shape_rmse(candidate: np.ndarray, desired: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Circular phase RMSE after removing one global phase piston."""
    use = np.asarray(mask, bool)
    delta = np.angle(np.exp(1j*(np.asarray(candidate, float)-np.asarray(desired, float))))
    phasor_mean = np.mean(np.exp(1j*delta[use]))
    piston = float(np.angle(phasor_mean)) if abs(phasor_mean) > 1e-12 else 0.0
    residual = np.angle(np.exp(1j*(delta-piston)))
    return float(np.sqrt(np.mean(residual[use]**2))), piston


def _mean_metrics(ideal: np.ndarray, test: np.ndarray, grid: dict) -> dict:
    table = v2.metric_table(v2.Z_FIT_M, ideal, {"test": test}, grid)
    return {
        "mean_pearson_r": float(table.test_pearson_r.mean()),
        "mean_nrmse": float(table.test_nrmse.mean()),
    }


def calibrate_mapping(nominal_route: dict, phase_truth_slm: np.ndarray) -> tuple[str, dict]:
    calibration_route = v2.base._route(SystemErrorConfig(), phase_truth_slm)
    measured = v2.base._propagate(calibration_route, v2.Z_FIT_M)
    raw, valid, retrieval_diag = v2.retrieve_raw_correction(calibration_route, measured)

    effective, amp_mask = v2.effective_phase_at_axicon(nominal_route, calibration_route)
    desired = -effective
    use = valid & amp_mask

    names = []
    for rot in (0, 90, 180, 270):
        for mirror in (False, True):
            for neg in (False, True):
                name = f"r{rot}" + ("_mirror" if mirror else "") + ("_neg" if neg else "")
                names.append(name)

    scores = {}
    pistons = {}
    for name in names:
        score, piston = _phase_shape_rmse(transform_phase(raw, name), desired, use)
        scores[name] = score
        pistons[name] = piston
    selected = min(scores, key=scores.get)
    selected_phase = transform_phase(raw, selected)

    ideal_stack = v2.base._propagate(nominal_route, v2.Z_FIT_M)
    corrected_stack = v2.base._propagate(calibration_route, v2.Z_FIT_M, selected_phase)
    before = _mean_metrics(ideal_stack, measured, nominal_route["grid"])
    after = _mean_metrics(ideal_stack, corrected_stack, nominal_route["grid"])

    return selected, {
        "purpose": "separate known-phase synthetic calibration of correction sign, rotation and parity",
        "global_phase_piston_removed_for_shape_comparison": True,
        "selected_transform": selected,
        "selected_phase_shape_rmse_rad": float(scores[selected]),
        "selected_global_piston_rad": float(pistons[selected]),
        "phase_shape_rmse_rad_by_transform": scores,
        "retrieval": retrieval_diag,
        "comparison_pixels": int(np.sum(use)),
        "known_phase_closure_before_correction": before,
        "known_phase_closure_after_correction": after,
        "known_phase_closure_change": {
            "pearson_r": float(after["mean_pearson_r"]-before["mean_pearson_r"]),
            "nrmse": float(after["mean_nrmse"]-before["mean_nrmse"]),
        },
    }


def build() -> dict:
    # Replace only the coordinate/sign calibration.  The optical model, Miao
    # retrieval, physical fitting and comparison metrics remain exactly v2.
    v2.transform_phase = transform_phase
    v2.calibrate_mapping = calibrate_mapping
    summary = v2.build(out=OUT)
    summary["study_version"] = 3
    summary["coordinate_calibration_note"] = (
        "global phase piston removed; all 4 right-angle rotations, one parity reflection, "
        "and both phase signs tested on a separate known-phase case"
    )
    (OUT/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print("V3_CALIBRATION", json.dumps(summary["synthetic_coordinate_calibration"], indent=2))
    return summary


if __name__ == "__main__":
    build()
