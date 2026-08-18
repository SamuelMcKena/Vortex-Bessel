"""Safe closed-loop controller for the Miao-style q=20 correction.

The old controller consumed the legacy normalized-z correction map.  This one
accepts only `slm2_correction_phase_rad.npy` produced by the calibrated full
retrieval and refuses to propose a mask while any pre-trial blocker remains.
A model prediction is never experimental acceptance.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from slm2_complete_mask_preview import build_slm2_complete_preview

EPS = 1e-12


def _candidate_mask(residual_phase, accepted_phase, gain):
    residual = np.asarray(residual_phase, float)
    accepted = np.asarray(accepted_phase, float)
    valid = np.isfinite(residual)
    if accepted.shape != residual.shape:
        raise ValueError("accepted and residual phase maps must have the same shape")
    rs = np.zeros_like(residual)
    rs[valid] = np.angle(np.exp(1j*residual[valid]))
    ac = np.zeros_like(accepted)
    av = np.isfinite(accepted)
    ac[av] = np.angle(np.exp(1j*accepted[av]))
    out = np.mod(ac + float(gain)*rs, 2*np.pi)
    out[~valid] = np.nan
    return out


def _append_history(path, row):
    p = Path(path)
    frame = pd.DataFrame([row])
    if p.exists():
        frame = pd.concat([pd.read_csv(p), frame], ignore_index=True)
    frame.to_csv(p, index=False)


def _blocked_state(retrieval_dir, loop_dir, manifest, reason):
    state = {
        "schema_version": 2,
        "status": "BLOCKED_BY_RETRIEVAL_OR_CALIBRATION",
        "experimental_accepted": False,
        "retrieval_dir": str(Path(retrieval_dir).resolve()),
        "candidate_phase_path": None,
        "hardware_ready": False,
        "hardware_blockers": manifest.get("pretrial_blockers",
                                           manifest.get("hardware_blockers", [reason])),
        "reason": reason,
        "next_action": "Complete the listed calibration/retrieval blockers; do not apply a correction mask yet.",
    }
    Path(loop_dir).mkdir(parents=True, exist_ok=True)
    (Path(loop_dir)/"closed_loop_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    return pd.DataFrame(), state


def propose_iteration(retrieval_dir, loop_dir, *, data_dir=None,
                      gains=(0.01, 0.02, 0.05, 0.10, 0.20), q=20,
                      kr_m_inv=None, force_recompute=False, trial_gain=0.05):
    """Create one conservative low-gain trial only from the calibrated full map.

    `q` and `kr_m_inv` are retained in the signature for compatibility with the
    old controller but are not used to reconstruct a correction here.
    """
    retrieval_dir, loop_dir = Path(retrieval_dir), Path(loop_dir)
    loop_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = retrieval_dir/"correction_manifest.json"
    if not manifest_path.exists():
        return _blocked_state(retrieval_dir, loop_dir, {},
                              "authoritative correction_manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not bool(manifest.get("application_ready_for_low_gain_trial", False)):
        return _blocked_state(retrieval_dir, loop_dir, manifest,
                              "full retrieval is not calibrated for a low-gain SLM2 trial")

    correction_path = retrieval_dir/"slm2_correction_phase_rad.npy"
    if not correction_path.exists():
        return _blocked_state(retrieval_dir, loop_dir, manifest,
                              "calibrated SLM2 correction phase file is missing")

    state_path = loop_dir/"closed_loop_state.json"
    previous = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
    if previous and not force_recompute and previous.get("status") == "AWAITING_EXPERIMENTAL_MEASUREMENT":
        return pd.DataFrame(), previous
    iteration = 0
    if previous and previous.get("status") == "EXPERIMENTALLY_ACCEPTED":
        iteration = int(previous.get("iteration", 0)) + 1
    elif previous:
        iteration = int(previous.get("iteration", 0))

    allowed = sorted(float(g) for g in gains if 0 < float(g) <= 0.20)
    if not allowed:
        raise ValueError("at least one gain in (0, 0.20] is required")
    gain = min(allowed, key=lambda g: abs(g-float(trial_gain)))

    residual = np.load(correction_path)
    if previous and previous.get("status") == "EXPERIMENTALLY_ACCEPTED" and previous.get("accepted_cumulative_phase_path"):
        accepted_path = loop_dir/previous["accepted_cumulative_phase_path"]
        accepted = np.load(accepted_path)
    else:
        accepted_path = loop_dir/"accepted_cumulative_phase_iteration_minus1.npy"
        if accepted_path.exists():
            accepted = np.load(accepted_path)
        else:
            accepted = np.zeros_like(residual)
            accepted[~np.isfinite(residual)] = np.nan
            np.save(accepted_path, accepted)

    candidate = _candidate_mask(residual, accepted, gain)
    candidate_name = f"iteration_{iteration:03d}_candidate_gain_{gain:.2f}_phase.npy"
    np.save(loop_dir/candidate_name, candidate)

    preview_dir = loop_dir/f"iteration_{iteration:03d}_slm2_preview"
    _, _, preview_manifest = build_slm2_complete_preview(
        loop_dir/candidate_name, preview_dir, ell_slm2=0,
        correction_gain=1.0, filename_tag=f"ITERATION_{iteration:03d}_MIAO_CANDIDATE")

    state = {
        "schema_version": 2,
        "iteration": iteration,
        "status": "AWAITING_EXPERIMENTAL_MEASUREMENT",
        "experimental_accepted": False,
        "baseline_data_dir": str(Path(data_dir).resolve()) if data_dir else None,
        "retrieval_dir": str(retrieval_dir.resolve()),
        "retrieval_manifest": str(manifest_path.resolve()),
        "candidate_gain": gain,
        "candidate_phase_path": candidate_name,
        "accepted_cumulative_phase_path": accepted_path.name,
        "preview_manifest": preview_manifest,
        "programmed_q_in_correction": False,
        "legacy_normalized_z_map_used": False,
        "model_prediction_is_not_acceptance": True,
        "hardware_ready": False,
        "next_action": "Apply only this low-gain trial, capture an identical new 18x4 z-stack, rerun the full retrieval, and evaluate the measured before/after metrics.",
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    _append_history(loop_dir/"iteration_history.csv", {
        "iteration": iteration, "event": "LOW_GAIN_TRIAL_PROPOSED",
        "status": state["status"], "gain": gain,
        "experimental_accepted": False, "candidate_phase_path": candidate_name,
    })
    return pd.DataFrame([{"iteration": iteration, "gain": gain,
                          "status": state["status"]}]), state


def evaluate_experimental_update(before_metrics_csv, after_metrics_csv, state_path):
    """Accept/reject only from a genuinely new camera z-stack."""
    state_path = Path(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "AWAITING_EXPERIMENTAL_MEASUREMENT":
        raise RuntimeError("controller is not waiting for a new experimental measurement")
    before, after = pd.read_csv(before_metrics_csv), pd.read_csv(after_metrics_csv)
    required = {"z_mm", "measured_vs_ideal_corr", "measured_vs_ideal_rmse",
                "measured_ring_cv", "measured_dark_core_ratio"}
    if not required.issubset(before) or not required.issubset(after):
        raise ValueError(f"missing experimental gate columns: {required-(set(before)&set(after))}")
    cols = sorted(required)
    merged = before[cols].merge(after[cols], on="z_mm", suffixes=("_before", "_after"))
    if len(merged) != len(before) or len(merged) != len(after):
        raise ValueError("before/after z planes do not match exactly")

    corr_gain = float(merged.measured_vs_ideal_corr_after.median() -
                      merged.measured_vs_ideal_corr_before.median())
    rmse_reduction = float(merged.measured_vs_ideal_rmse_before.median() -
                           merged.measured_vs_ideal_rmse_after.median())
    cv_fraction = float((merged.measured_ring_cv_before.median() -
                         merged.measured_ring_cv_after.median()) /
                        max(float(merged.measured_ring_cv_before.median()), EPS))
    dark_change = float(merged.measured_dark_core_ratio_after.max() -
                        merged.measured_dark_core_ratio_before.max())
    accepted = bool(corr_gain >= 0.01 and rmse_reduction > 0 and
                    cv_fraction >= 0.05 and dark_change <= 0.01)
    result = {
        "accepted": accepted, "cartesian_correlation_gain": corr_gain,
        "cartesian_rmse_reduction": rmse_reduction,
        "ring_cv_reduction_fraction": cv_fraction,
        "maximum_dark_core_ratio_change": dark_change,
        "reason": "EXPERIMENTALLY ACCEPTED" if accepted else
                  "REJECTED - new camera stack did not pass all gates",
    }
    state["experimental_accepted"] = accepted
    state["status"] = "EXPERIMENTALLY_ACCEPTED" if accepted else "EXPERIMENTALLY_REJECTED"
    state["experimental_evaluation"] = result
    if accepted:
        candidate = state_path.parent/state["candidate_phase_path"]
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        accepted_name = f"accepted_cumulative_phase_iteration_{int(state['iteration']):03d}.npy"
        np.save(state_path.parent/accepted_name, np.load(candidate))
        state["accepted_cumulative_phase_path"] = accepted_name
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    _append_history(state_path.parent/"iteration_history.csv", {
        "iteration": int(state["iteration"]), "event": "EXPERIMENTAL_EVALUATION",
        "status": state["status"], "gain": state.get("candidate_gain"),
        "experimental_accepted": accepted,
        "candidate_phase_path": state.get("candidate_phase_path"),
    })
    return result
