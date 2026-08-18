"""Safe closed-loop controller for the calibrated Miao-style q=20 correction.

The old controller consumed a legacy normalized-z correction map and then passed
its candidate through a second normalized-coordinate remapping step. Both are
forbidden here. This controller accepts only `slm2_correction_phase_rad.npy`
produced by the calibrated full retrieval; that array is already in native SLM2
pixel coordinates and is kept there without another geometric transform.

The output is an additive phase layer in radians, NOT a greyscale hardware
raster. The lab GUI/driver must combine it with the existing programmed phase,
wrap once, and use the independently calibrated 1030-nm SLM LUT.

Experimental acceptance requires two independently generated metric CSVs with
identical target/calibration provenance and DIFFERENT SHA-256 BMG dataset
fingerprints. Reusing the same camera stack therefore cannot accept a candidate.
"""
from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EPS = 1e-12


def _candidate_mask(residual_phase, accepted_phase, gain):
    """Accumulate signed phase through unit phasors, preserving native SLM pixels."""
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
    out = np.full_like(residual, np.nan)
    out[valid] = np.angle(np.exp(1j*(ac[valid] + float(gain)*rs[valid])))
    return out


def _append_history(path, row):
    p = Path(path)
    frame = pd.DataFrame([row])
    if p.exists():
        frame = pd.concat([pd.read_csv(p), frame], ignore_index=True)
    frame.to_csv(p, index=False)


def _write_native_slm2_preview(candidate, output_dir, iteration, gain):
    """Preview a native-pixel correction layer without resampling or LUT encoding."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phase = np.asarray(candidate, float)
    valid = np.isfinite(phase)

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    shown = np.ma.masked_invalid(phase)
    im = ax.imshow(shown, origin="upper", cmap="twilight", vmin=-np.pi, vmax=np.pi,
                   interpolation="nearest", aspect="equal")
    ax.set(title=(f"Iteration {iteration}: calibrated native-SLM2 correction layer, gain={gain:.2f}\n"
                  "phase radians only — no second coordinate mapping, no greyscale LUT export"),
           xlabel="SLM2 x pixel", ylabel="SLM2 y pixel")
    fig.colorbar(im, ax=ax, label="signed correction phase (rad)")
    png = output_dir/"native_slm2_correction_layer_preview.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "status": "CALIBRATED_PHASE_LAYER_PREVIEW_NOT_DIRECT_HARDWARE_RASTER",
        "coordinate_space": "native SLM2 pixels",
        "shape_yx": list(map(int, phase.shape)),
        "valid_pixel_fraction": float(np.mean(valid)),
        "phase_units": "radians",
        "phase_range_convention": "signed [-pi, pi] on valid pixels",
        "geometric_remapping_applied_here": False,
        "linear_greyscale_conversion_applied_here": False,
        "application_contract": (
            "Add this phase layer to the existing SLM2 programmed phase in the lab GUI, "
            "wrap the combined phase once, then encode with the measured 1030-nm LUT."
        ),
        "preview_png": png.name,
    }
    (output_dir/"native_slm2_correction_layer_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _blocked_state(retrieval_dir, loop_dir, manifest, reason):
    state = {
        "schema_version": 4,
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
    """Create one conservative low-gain trial from the calibrated native-SLM2 map.

    `q` and `kr_m_inv` remain in the signature only for compatibility with older
    callers; this controller never reconstructs its own correction from them.
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
                              "calibrated native-SLM2 correction phase file is missing")

    residual = np.load(correction_path)
    expected_shape = manifest.get("slm2_shape_yx")
    if expected_shape is not None and tuple(map(int, expected_shape)) != residual.shape:
        return _blocked_state(retrieval_dir, loop_dir, manifest,
                              "SLM2 correction array shape disagrees with calibrated manifest")

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
    candidate_name = f"iteration_{iteration:03d}_candidate_gain_{gain:.2f}_phase_rad.npy"
    np.save(loop_dir/candidate_name, candidate.astype(np.float32))

    preview_dir = loop_dir/f"iteration_{iteration:03d}_native_slm2_preview"
    preview_manifest = _write_native_slm2_preview(candidate, preview_dir, iteration, gain)

    state = {
        "schema_version": 4,
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
        "candidate_coordinate_space": "native SLM2 pixels",
        "candidate_phase_units": "radians",
        "second_coordinate_mapping_applied": False,
        "programmed_q_in_correction": False,
        "legacy_normalized_z_map_used": False,
        "direct_greyscale_hardware_export_created": False,
        "model_prediction_is_not_acceptance": True,
        "experimental_acceptance_requires_distinct_dataset_sha256": True,
        "hardware_ready": False,
        "next_action": (
            "Load the candidate as an additive PHASE correction layer in the calibrated SLM2 GUI, "
            "not as a direct bitmap; apply the measured LUT through the normal driver, capture an "
            "identical new 18x4 z-stack, generate independent acceptance metrics, then evaluate before/after."
        ),
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    _append_history(loop_dir/"iteration_history.csv", {
        "iteration": iteration, "event": "LOW_GAIN_TRIAL_PROPOSED",
        "status": state["status"], "gain": gain,
        "experimental_accepted": False, "candidate_phase_path": candidate_name,
    })
    return pd.DataFrame([{"iteration": iteration, "gain": gain,
                          "status": state["status"]}]), state


def _unique_constant(frame, column):
    values = frame[column].dropna().unique()
    if len(values) != 1:
        raise ValueError(f"{column} must contain exactly one provenance value per dataset")
    return values[0]


def evaluate_experimental_update(before_metrics_csv, after_metrics_csv, state_path):
    """Accept/reject only from a genuinely new camera z-stack with identical target provenance."""
    state_path = Path(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "AWAITING_EXPERIMENTAL_MEASUREMENT":
        raise RuntimeError("controller is not waiting for a new experimental measurement")
    before_path, after_path = Path(before_metrics_csv), Path(after_metrics_csv)
    if before_path.resolve() == after_path.resolve():
        raise ValueError("before and after metrics must be different files from different camera stacks")
    before, after = pd.read_csv(before_path), pd.read_csv(after_path)

    metric_cols = {"z_mm", "measured_vs_ideal_corr", "measured_vs_ideal_rmse",
                   "measured_ring_cv", "measured_dark_core_ratio"}
    provenance_cols = {"dataset_sha256", "q_target", "k_perp_nominal_m_inv",
                       "wavelength_m", "pixel_pitch_m", "roi_radius_um",
                       "camera_axis_source"}
    required = metric_cols | provenance_cols
    if not required.issubset(before) or not required.issubset(after):
        missing = required-(set(before)&set(after))
        raise ValueError(f"missing experimental gate/provenance columns: {missing}")

    before_digest = str(_unique_constant(before, "dataset_sha256"))
    after_digest = str(_unique_constant(after, "dataset_sha256"))
    if before_digest == after_digest:
        raise ValueError("before and after metrics resolve to the same BMG dataset SHA-256")

    # The target and calibration defining 'improvement' must be identical.
    for column in sorted(provenance_cols-{"dataset_sha256"}):
        b = _unique_constant(before, column)
        a = _unique_constant(after, column)
        if column == "camera_axis_source":
            if str(a) != str(b):
                raise ValueError(f"before/after {column} differs: {b!r} vs {a!r}")
        else:
            if not np.isclose(float(a), float(b), rtol=0, atol=1e-12):
                raise ValueError(f"before/after {column} differs: {b!r} vs {a!r}")

    cols = sorted(metric_cols)
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
        "accepted": accepted,
        "before_dataset_sha256": before_digest,
        "after_dataset_sha256": after_digest,
        "cartesian_correlation_gain": corr_gain,
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
        "before_dataset_sha256": before_digest,
        "after_dataset_sha256": after_digest,
    })
    return result
