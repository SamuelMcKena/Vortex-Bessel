"""Preview-only closed-loop controller for iterative SLM2 correction.

The controller makes a conservative candidate from the latest residual phase,
scores several gains on interleaved held-out z planes, and persists its state.
Crucially, a model-predicted improvement can only create an
``AWAITING_EXPERIMENTAL_MEASUREMENT`` proposal.  Acceptance requires a new lab
z-stack and never follows from the same data used to retrieve the correction.
"""
from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from modal_vortex_bessel import modal_basis, aberration_field_theta
from slm2_complete_mask_preview import build_slm2_complete_preview


EPS = 1e-12


def _normalise(a):
    a = np.asarray(a, float)
    return a / max(float(np.max(a)), EPS)


def _coefficients_from_row(row, m_values):
    return np.asarray([
        float(row[f"cm{m:+d}_abs_rel"]) * np.exp(1j * float(row[f"cm{m:+d}_phase"]))
        for m in m_values
    ], complex)


def _coefficients_from_angular_field(g_theta, theta, m_values):
    return np.asarray([
        np.mean(g_theta * np.exp(1j * m * theta)) for m in m_values
    ], complex)


def _interpolate_unit_phasor(z_query, z_train, phasor_rows):
    """Interpolate a wrapped phase in the complex plane at one held-out z."""
    z_train = np.asarray(z_train, float)
    phasor_rows = np.asarray(phasor_rows, complex)
    real = np.asarray([np.interp(z_query, z_train, phasor_rows[:, j].real)
                       for j in range(phasor_rows.shape[1])])
    imag = np.asarray([np.interp(z_query, z_train, phasor_rows[:, j].imag)
                       for j in range(phasor_rows.shape[1])])
    value = real + 1j * imag
    return value / np.maximum(np.abs(value), EPS)


def _corr_rmse(a, b, mask):
    av, bv = a[mask], b[mask]
    return (float(np.corrcoef(av, bv)[0, 1]),
            float(np.sqrt(np.mean((av - bv) ** 2))))


def _ring_profile(image, axis_um, radii_um=np.linspace(36, 55, 12)):
    theta = np.linspace(0, 2*np.pi, 720, endpoint=False)
    rr, tt = np.meshgrid(radii_um, theta, indexing="ij")
    px_um = float(axis_um[1] - axis_um[0])
    centre = (len(axis_um)-1)/2
    yy = centre + rr*np.sin(tt)/px_um
    xx = centre + rr*np.cos(tt)/px_um
    return ndimage.map_coordinates(image, [yy, xx], order=1).mean(axis=0)


def _dark_ratio(image, radius_um):
    core = radius_um < 20
    ring = (radius_um >= 36) & (radius_um <= 55)
    return float(np.mean(image[core]) / max(float(np.mean(image[ring])), EPS))


def _candidate_mask(full_correction, accepted_phase, gain):
    """Compose phases through unit phasors; never scale a wrapped 0..2pi array."""
    full = np.asarray(full_correction, float)
    valid = np.isfinite(full)
    residual_signed = np.zeros_like(full)
    residual_signed[valid] = np.angle(np.exp(1j * full[valid]))
    accepted_signed = np.zeros_like(full)
    accepted_valid = np.isfinite(accepted_phase)
    accepted_signed[accepted_valid] = np.angle(np.exp(1j * accepted_phase[accepted_valid]))
    candidate = np.mod(accepted_signed + float(gain) * residual_signed, 2*np.pi)
    candidate[~valid] = np.nan
    return candidate


def _append_history(path, row):
    path = Path(path)
    new = pd.DataFrame([row])
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new.to_csv(path, index=False)


def propose_iteration(modal_dir, loop_dir, *, data_dir=None,
                      gains=(0.01, 0.02, 0.05, 0.10, 0.20), q=20,
                      kr_m_inv=489678.1594027835, force_recompute=False):
    modal_dir, loop_dir = Path(modal_dir), Path(loop_dir)
    loop_dir.mkdir(parents=True, exist_ok=True)
    state_path = loop_dir / "closed_loop_state.json"
    previous_state = None
    iteration = 0
    if state_path.exists():
        previous_state = json.loads(state_path.read_text(encoding="utf-8"))
        previous_status = previous_state.get("status")
        if not force_recompute and previous_status in {
                "AWAITING_EXPERIMENTAL_MEASUREMENT", "MODEL_REJECTED_ON_HELD_OUT_Z",
                "EXPERIMENTALLY_REJECTED"}:
            return pd.read_csv(loop_dir / previous_state["gain_sweep_metrics_csv"]), previous_state
        if previous_status == "EXPERIMENTALLY_ACCEPTED" and not force_recompute:
            iteration = int(previous_state["iteration"]) + 1
        elif force_recompute:
            iteration = int(previous_state.get("iteration", 0))
    prefix = f"iteration_{iteration:03d}"

    recreation = np.load(modal_dir / "phase_error_recreation" /
                         "phase_error_recreation_stack.npz")
    axis = recreation["x_um"]
    z_mm = recreation["z_mm"]
    ideal = _normalise(recreation["ideal"])
    table = pd.read_csv(modal_dir / "modal_fit_metrics.csv")
    m_values = np.arange(-8, 9, dtype=int)
    theta = np.linspace(0, 2*np.pi, 2048, endpoint=False)

    X_m, Y_m = np.meshgrid(axis*1e-6, axis*1e-6, indexing="xy")
    R_m = np.hypot(X_m, Y_m)
    PHI = np.arctan2(Y_m, X_m)
    basis, _ = modal_basis(q, m_values, kr_m_inv, R_m.ravel(), PHI.ravel())
    roi = R_m <= 160e-6

    coeff_rows = []
    g_rows = []
    for z in z_mm:
        row = table.iloc[int(np.argmin(np.abs(table.z_mm-z)))]
        coeffs = _coefficients_from_row(row, m_values)
        coeff_rows.append(coeffs)
        g_rows.append(aberration_field_theta(coeffs, m_values, theta))
    coeff_rows = np.stack(coeff_rows)
    g_rows = np.stack(g_rows)
    residual_correction_phasor = np.exp(-1j*np.angle(g_rows))

    # Interleaving means every validation plane lies between training annuli.
    # Include the final endpoint in training so interpolation never extrapolates.
    train = np.zeros(len(z_mm), dtype=bool)
    train[::2] = True
    train[-1] = True
    validation = ~train
    correction_at_plane = []
    for iz, z in enumerate(z_mm):
        if train[iz]:
            correction_at_plane.append(residual_correction_phasor[iz])
        else:
            correction_at_plane.append(_interpolate_unit_phasor(
                z, z_mm[train], residual_correction_phasor[train]))
    correction_at_plane = np.stack(correction_at_plane)

    rows = []
    preview_stacks = {}
    for gain in (0.0, *tuple(float(g) for g in gains)):
        stack = []
        for iz in range(len(z_mm)):
            applied = np.exp(1j * gain * np.angle(correction_at_plane[iz]))
            after_g = g_rows[iz] * applied
            after_coeffs = _coefficients_from_angular_field(after_g, theta, m_values)
            intensity = (np.abs(basis @ after_coeffs) ** 2).reshape(len(axis), len(axis))
            intensity = _normalise(intensity)
            stack.append(intensity)
            corr, rmse = _corr_rmse(intensity, ideal, roi)
            ring = _ring_profile(intensity, axis)
            rows.append({
                "gain": gain, "z_mm": float(z_mm[iz]),
                "split": "train" if train[iz] else "validation",
                "ideal_corr": corr, "ideal_rmse": rmse,
                "ring_cv": float(np.std(ring)/max(float(np.mean(ring)), EPS)),
                "dark_core_ratio": _dark_ratio(intensity, R_m*1e6),
            })
        preview_stacks[gain] = np.stack(stack)
    sweep = pd.DataFrame(rows)
    metrics_name = f"{prefix}_model_gain_sweep_metrics.csv"
    sweep.to_csv(loop_dir / metrics_name, index=False)

    baseline = sweep[sweep.gain == 0.0]
    aggregate_rows = []
    for gain in gains:
        item = {"gain": float(gain)}
        for split in ("train", "validation"):
            before = baseline[baseline.split == split]
            after = sweep[(sweep.gain == float(gain)) & (sweep.split == split)]
            item[f"{split}_corr_gain"] = float(after.ideal_corr.median() - before.ideal_corr.median())
            item[f"{split}_rmse_reduction"] = float(before.ideal_rmse.median() - after.ideal_rmse.median())
            item[f"{split}_ring_cv_reduction"] = float(before.ring_cv.median() - after.ring_cv.median())
            item[f"{split}_dark_core_change"] = float(after.dark_core_ratio.median() - before.dark_core_ratio.median())
        item["validation_pass"] = bool(
            item["validation_corr_gain"] >= .005 and
            item["validation_rmse_reduction"] > 0 and
            item["validation_ring_cv_reduction"] >= -.01 and
            item["validation_dark_core_change"] <= .002)
        item["score"] = (item["validation_corr_gain"] +
                         2*item["validation_rmse_reduction"] +
                         .5*item["validation_ring_cv_reduction"])
        aggregate_rows.append(item)
    aggregate = pd.DataFrame(aggregate_rows)
    gain_summary_name = f"{prefix}_gain_summary.csv"
    aggregate.to_csv(loop_dir / gain_summary_name, index=False)
    passing = aggregate[aggregate.validation_pass]
    recommended_gain = (float(passing.loc[passing.score.idxmax(), "gain"])
                        if len(passing) else None)

    def write_gain_figure(selected_gain=None):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
        for split, marker in (("train", "o"), ("validation", "s")):
            axes[0].plot(aggregate.gain, aggregate[f"{split}_corr_gain"], marker+"-", label=split)
            axes[1].plot(aggregate.gain, aggregate[f"{split}_rmse_reduction"], marker+"-", label=split)
            axes[2].plot(aggregate.gain, aggregate[f"{split}_ring_cv_reduction"], marker+"-", label=split)
        axes[0].set(title="Ideal-correlation improvement", ylabel="correlation gain")
        axes[1].set(title="Ideal intensity-error reduction", ylabel="RMSE reduction")
        axes[2].set(title="Ring non-uniformity reduction", ylabel="CV reduction")
        for ax in axes:
            ax.axhline(0, color="black", lw=.7)
            if selected_gain is not None:
                ax.axvline(selected_gain, color="red", ls="--", lw=1,
                           label=f"proposed gain={selected_gain:.2f}")
            ax.set_xlabel("incremental correction gain")
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        verdict = ("candidate awaits experimental acceptance" if selected_gain is not None
                   else "ALL GAINS REJECTED ON HELD-OUT z PLANES — NO MASK PROPOSED")
        fig.suptitle(f"Iteration {iteration} model-only gain selection — {verdict}")
        fig.savefig(loop_dir / f"{prefix}_gain_sweep.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    hardware_blockers = [
        "measured SLM2 1030-nm LUT/phase stroke", "SLM2 display rotation/parity",
        "measured beam centre and radius on SLM2", "camera-to-SLM2 transform",
        "physical z-to-input-annulus map", "experimental single-mask full-stack validation",
    ]
    if recommended_gain is None:
        write_gain_figure(None)
        state = {
            "schema_version": 1, "iteration": iteration,
            "status": "MODEL_REJECTED_ON_HELD_OUT_Z",
            "experimental_accepted": False,
            "baseline_data_dir": str(Path(data_dir).resolve()) if data_dir else None,
            "modal_dir": str(modal_dir.resolve()),
            "training_z_mm": z_mm[train].tolist(),
            "validation_z_mm": z_mm[validation].tolist(),
            "candidate_gains": list(map(float, gains)),
            "recommended_gain": None, "candidate_phase_path": None,
            "gain_sweep_metrics_csv": metrics_name,
            "gain_summary_csv": gain_summary_name,
            "hardware_ready": False, "hardware_blockers": hardware_blockers,
            "model_prediction_is_not_acceptance": True,
            "reason": "Every incremental gain worsened at least one held-out validation criterion.",
            "next_action": "Do not apply a mask. Calibrate SLM2 coordinates/LUT and improve the physical forward model before proposing another iteration.",
        }
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        _append_history(loop_dir / "iteration_history.csv", {
            "iteration": iteration, "event": "MODEL_PROPOSAL",
            "status": state["status"],
            "recommended_gain": np.nan, "experimental_accepted": False,
            "baseline_data_dir": state["baseline_data_dir"],
            "candidate_phase_path": None,
        })
        return sweep, state

    full_correction_path = (modal_dir /
        "UNCALIBRATED_DO_NOT_APPLY_q20_modal_correction.npy")
    full_correction = np.load(full_correction_path)
    if (previous_state and previous_state.get("status") == "EXPERIMENTALLY_ACCEPTED" and
            previous_state.get("accepted_cumulative_phase_path")):
        accepted_path = loop_dir / previous_state["accepted_cumulative_phase_path"]
    else:
        accepted_path = loop_dir / "accepted_cumulative_phase_iteration_minus1.npy"
    if accepted_path.exists():
        accepted_phase = np.load(accepted_path)
    else:
        accepted_phase = np.zeros_like(full_correction)
        accepted_phase[~np.isfinite(full_correction)] = np.nan
        np.save(accepted_path, accepted_phase)
    candidate = _candidate_mask(full_correction, accepted_phase, recommended_gain)
    candidate_name = f"{prefix}_candidate_gain_{recommended_gain:.2f}_phase.npy"
    np.save(loop_dir / candidate_name, candidate)

    preview_dir = loop_dir / f"{prefix}_slm2_preview"
    _, _, preview_manifest = build_slm2_complete_preview(
        loop_dir / candidate_name, preview_dir, ell_slm2=0,
        correction_gain=1.0, filename_tag=f"ITERATION_{iteration:03d}_CANDIDATE")

    write_gain_figure(recommended_gain)

    state = {
        "schema_version": 1,
        "iteration": iteration,
        "status": "AWAITING_EXPERIMENTAL_MEASUREMENT",
        "experimental_accepted": False,
        "baseline_data_dir": str(Path(data_dir).resolve()) if data_dir else None,
        "modal_dir": str(modal_dir.resolve()),
        "training_z_mm": z_mm[train].tolist(),
        "validation_z_mm": z_mm[validation].tolist(),
        "candidate_gains": list(map(float, gains)),
        "recommended_gain": recommended_gain,
        "candidate_phase_path": candidate_name,
        "accepted_cumulative_phase_path": accepted_path.name,
        "gain_sweep_metrics_csv": metrics_name,
        "gain_summary_csv": gain_summary_name,
        "preview_manifest": preview_manifest,
        "hardware_ready": False,
        "hardware_blockers": preview_manifest["missing_before_hardware"],
        "model_prediction_is_not_acceptance": True,
        "next_action": "Calibrate SLM2 mapping/LUT, apply only an approved low-gain candidate, capture an identical new z-stack, then run experimental acceptance.",
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    _append_history(loop_dir / "iteration_history.csv", {
        "iteration": iteration, "event": "MODEL_PROPOSAL",
        "status": state["status"],
        "recommended_gain": recommended_gain,
        "experimental_accepted": False,
        "baseline_data_dir": state["baseline_data_dir"],
        "candidate_phase_path": candidate_name,
    })
    return sweep, state


def evaluate_experimental_update(before_metrics_csv, after_metrics_csv, state_path):
    """Accept/reject a proposed iteration using genuinely new camera data."""
    state_path = Path(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "AWAITING_EXPERIMENTAL_MEASUREMENT":
        raise RuntimeError("Controller is not waiting for a new experimental measurement")
    before = pd.read_csv(before_metrics_csv)
    after = pd.read_csv(after_metrics_csv)
    required = {"z_mm", "measured_vs_ideal_corr", "measured_vs_ideal_rmse",
                "measured_ring_cv", "measured_dark_core_ratio"}
    if not required.issubset(before) or not required.issubset(after):
        raise ValueError(f"Missing experimental gate columns: {required-(set(before)&set(after))}")
    merged = before[list(required)].merge(after[list(required)], on="z_mm",
                                           suffixes=("_before", "_after"))
    corr_gain = float(merged.measured_vs_ideal_corr_after.median() -
                      merged.measured_vs_ideal_corr_before.median())
    rmse_reduction = float(merged.measured_vs_ideal_rmse_before.median() -
                           merged.measured_vs_ideal_rmse_after.median())
    cv_reduction_fraction = float(
        (merged.measured_ring_cv_before.median()-merged.measured_ring_cv_after.median()) /
        max(float(merged.measured_ring_cv_before.median()), EPS))
    dark_change = float(merged.measured_dark_core_ratio_after.max() -
                        merged.measured_dark_core_ratio_before.max())
    accepted = bool(corr_gain >= .01 and rmse_reduction > 0 and
                    cv_reduction_fraction >= .05 and dark_change <= .01)
    result = {
        "accepted": accepted, "cartesian_correlation_gain": corr_gain,
        "cartesian_rmse_reduction": rmse_reduction,
        "ring_cv_reduction_fraction": cv_reduction_fraction,
        "maximum_dark_core_ratio_change": dark_change,
        "reason": "EXPERIMENTALLY ACCEPTED" if accepted else
                  "REJECTED — NEW CAMERA STACK DID NOT PASS ALL GATES",
    }
    iteration = int(state["iteration"])
    prefix = f"iteration_{iteration:03d}"
    state["experimental_accepted"] = accepted
    state["status"] = "EXPERIMENTALLY_ACCEPTED" if accepted else "EXPERIMENTALLY_REJECTED"
    state["experimental_evaluation"] = result
    if accepted:
        candidate_path = state_path.parent / state["candidate_phase_path"]
        if not candidate_path.exists():
            raise FileNotFoundError(f"Accepted candidate phase is missing: {candidate_path}")
        accepted_name = f"accepted_cumulative_phase_iteration_{iteration:03d}.npy"
        np.save(state_path.parent / accepted_name, np.load(candidate_path))
        state["accepted_cumulative_phase_path"] = accepted_name
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    result_path = state_path.with_name(f"{prefix}_experimental_evaluation.json")
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _append_history(state_path.parent / "iteration_history.csv", {
        "iteration": iteration, "event": "EXPERIMENTAL_EVALUATION",
        "status": state["status"], "recommended_gain": state.get("recommended_gain"),
        "experimental_accepted": accepted,
        "baseline_data_dir": state.get("baseline_data_dir"),
        "candidate_phase_path": state.get("candidate_phase_path"),
    })
    return result


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    modal = here / "outputs" / "slm_closed_loop_alignment" / "modal_q20"
    _, report = propose_iteration(
        modal, modal / "iterative_closed_loop",
        data_dir=here / "z-scan 2 1010", force_recompute=True)
    print(json.dumps(report, indent=2))
