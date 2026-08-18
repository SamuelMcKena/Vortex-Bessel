"""Run the physics-consistent q=20 multi-plane wavefront reconstruction.

This is the canonical intensity-stack wavefront diagnostic for the measured
axicon data.  It replaces the legacy notebook figures that:
  1. displayed the total wrapped q=20 phase as though it were aberration, and
  2. propagated a field from the first measured plane using the absolute
     z labels (-17 ... 0 mm), which double-counted the reference offset.

The present workflow reconstructs one complex field at one measured reference
plane, uses every training z plane through a common BL-ASM forward model, and
evaluates held-out planes without refitting.  Residual phase is formed relative
to the *complete nominal complex field* at that same plane.

Any conjugate phase shown here is a CAMERA-PLANE counterfactual.  Hardware SLM2
correction remains blocked until the camera<->SLM optical transform and phase
calibration are measured.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for CODE_ROOT in Path(__file__).resolve().parents:
    if (CODE_ROOT / "vbb_study").is_dir():
        if str(CODE_ROOT) not in sys.path:
            sys.path.insert(0, str(CODE_ROOT))
        break

from modal_vortex_bessel import load_first_scan, estimate_global_kr, find_dark_core_center
from physics_consistent_multiplane import (
    RetrievalGrid,
    apply_phase_screen,
    ideal_relative_phase,
    multiplane_phase_retrieval,
    normalise_intensity,
    phase_screen_from_residual,
    prepare_amplitude_stack,
    propagate_stack,
    signed_longitudinal_sections,
)
from vbb_study.equations.propagation import make_bl_asm_propagator


EPS = 1.0e-12


def _corr_rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    av = np.asarray(a, float)[mask]
    bv = np.asarray(b, float)[mask]
    if np.std(av) <= EPS or np.std(bv) <= EPS:
        corr = 0.0
    else:
        corr = float(np.corrcoef(av, bv)[0, 1])
    rmse = float(np.sqrt(np.mean((av - bv) ** 2)))
    return corr, rmse


def _normalised_stack_from_fields(fields) -> np.ndarray:
    return np.stack([normalise_intensity(np.abs(field) ** 2) for field in fields])


def _build_nominal_input(grid: dict, q: int, kr_m_inv: float, beam_radius_m: float) -> np.ndarray:
    """Nominal selected-order conical Gaussian at the model input plane."""

    amplitude = np.exp(-(grid["R"] / float(beam_radius_m)) ** 2)
    return amplitude * np.exp(1j * (int(q) * grid["PHI"] - float(kr_m_inv) * grid["R"]))


def _fit_ideal_absolute_z(
    nominal_input: np.ndarray,
    measured_intensity: np.ndarray,
    z_relative_m: np.ndarray,
    retrieval_grid: RetrievalGrid,
    *,
    end_search_mm=(18.0, 30.0),
    roi_radius_m=250e-6,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Fit one nuisance camera-to-input distance using the ideal model only.

    The recovered phase is not involved in this registration.  End-point and
    central anchor planes are used so the nuisance fit cannot mimic an arbitrary
    z-dependent aberration.
    """

    grid = retrieval_grid.grid()
    propagator = make_bl_asm_propagator(
        nominal_input,
        grid,
        retrieval_grid.wavelength_m,
        bandlimit=True,
        include_evanescent=False,
    )
    roi = grid["R"] <= float(roi_radius_m)
    n = len(z_relative_m)
    anchors = np.unique(np.asarray([0, n // 2, n - 1], int))

    coarse = np.arange(float(end_search_mm[0]), float(end_search_mm[1]) + 1e-9, 1.0)
    rows: list[dict] = []

    def evaluate(end_mm: float) -> None:
        cors = []
        rmses = []
        for idx in anchors:
            z_abs_m = end_mm * 1e-3 + float(z_relative_m[idx])
            model = normalise_intensity(np.abs(propagator(z_abs_m)) ** 2)
            corr, rmse = _corr_rmse(measured_intensity[idx], model, roi)
            cors.append(corr)
            rmses.append(rmse)
        rows.append(
            {
                "absolute_z_at_relative_zero_mm": float(end_mm),
                "mean_anchor_corr": float(np.mean(cors)),
                "mean_anchor_rmse": float(np.mean(rmses)),
            }
        )

    for value in coarse:
        evaluate(float(value))
    frame = pd.DataFrame(rows)
    best = float(frame.loc[frame.mean_anchor_corr.idxmax(), "absolute_z_at_relative_zero_mm"])

    fine = np.arange(
        max(float(end_search_mm[0]), best - 1.0),
        min(float(end_search_mm[1]), best + 1.0) + 1e-9,
        0.1,
    )
    for value in fine:
        if not np.any(np.isclose(frame.absolute_z_at_relative_zero_mm.to_numpy(), value)):
            evaluate(float(value))
    frame = pd.DataFrame(rows).sort_values("absolute_z_at_relative_zero_mm")
    best = float(frame.loc[frame.mean_anchor_corr.idxmax(), "absolute_z_at_relative_zero_mm"])
    return best * 1e-3 + np.asarray(z_relative_m, float), frame


def _train_holdout_weights(n_planes: int) -> tuple[np.ndarray, np.ndarray]:
    """Use deterministic interleaved held-out planes for a genuine z test."""

    holdout = np.arange(2, n_planes, 4, dtype=int)
    train = np.ones(n_planes, bool)
    train[holdout] = False
    if train.sum() < 3:
        train[:] = True
        holdout = np.array([], dtype=int)
    weights = train.astype(float)
    weights /= weights.sum()
    return weights, holdout


def _confidence_support(amplitudes: np.ndarray, reference_index: int, floor=0.02) -> np.ndarray:
    amp = amplitudes[int(reference_index)]
    return amp >= float(floor) * max(float(amp.max()), EPS)


def _metric_rows(
    measured: np.ndarray,
    reconstructed: np.ndarray,
    nominal: np.ndarray,
    error_recreation: np.ndarray,
    virtual_corrected: np.ndarray,
    z_mm: np.ndarray,
    roi: np.ndarray,
    holdout: np.ndarray,
) -> pd.DataFrame:
    rows = []
    holdout_set = set(map(int, holdout))
    for i, z in enumerate(z_mm):
        rec_c, rec_e = _corr_rmse(measured[i], reconstructed[i], roi)
        nom_c, nom_e = _corr_rmse(measured[i], nominal[i], roi)
        err_c, err_e = _corr_rmse(measured[i], error_recreation[i], roi)
        cor_c, cor_e = _corr_rmse(virtual_corrected[i], nominal[i], roi)
        rows.append(
            {
                "plane_index": i,
                "z_relative_mm": float(z),
                "split": "holdout" if i in holdout_set else "train",
                "reconstruction_vs_measured_corr": rec_c,
                "reconstruction_vs_measured_rmse": rec_e,
                "nominal_vs_measured_corr": nom_c,
                "nominal_vs_measured_rmse": nom_e,
                "phase_error_recreation_vs_measured_corr": err_c,
                "phase_error_recreation_vs_measured_rmse": err_e,
                "virtual_corrected_vs_nominal_corr": cor_c,
                "virtual_corrected_vs_nominal_rmse": cor_e,
            }
        )
    return pd.DataFrame(rows)


def run_global_multiplane(
    data_dir,
    output_dir,
    *,
    z_relative_mm=None,
    wavelength_m=1030e-9,
    pixel_pitch_m=5.5e-6,
    q=20,
    output_n=384,
    iterations=160,
    relaxation=0.65,
    beam_radius_m=2.0e-3,
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = load_first_scan(data_dir)
    if not images:
        raise FileNotFoundError(f"No z*_*.bmg files found in {data_dir}")

    if z_relative_mm is None:
        z_relative_mm = np.arange(-(len(images) - 1), 1, dtype=float)
    z_relative_mm = np.asarray(z_relative_mm, float)
    if len(z_relative_mm) != len(images):
        raise ValueError("z_relative_mm must contain one coordinate per measured plane")
    if np.any(np.diff(z_relative_mm) <= 0):
        raise ValueError("z_relative_mm must be strictly increasing")

    centers = [find_dark_core_center(image)[:2] for image in images]
    amplitudes, dx_m, removed_power = prepare_amplitude_stack(
        images,
        output_n=output_n,
        native_pixel_pitch_m=pixel_pitch_m,
        centers_yx=centers,
    )
    measured = np.stack([normalise_intensity(amp**2) for amp in amplitudes])
    z_relative_m = z_relative_mm * 1e-3
    retrieval_grid = RetrievalGrid(output_n, dx_m, wavelength_m)
    grid = retrieval_grid.grid()

    kr_m_inv, geometry = estimate_global_kr(images, pixel_pitch_m, q, .55)
    geometry.insert(1, "z_relative_mm", z_relative_mm)
    geometry.to_csv(output_dir / "ring_geometry_for_global_retrieval.csv", index=False)

    nominal_input = _build_nominal_input(grid, q, kr_m_inv, beam_radius_m)
    z_absolute_m, registration = _fit_ideal_absolute_z(
        nominal_input,
        measured,
        z_relative_m,
        retrieval_grid,
    )
    registration.to_csv(output_dir / "ideal_absolute_z_registration.csv", index=False)

    nominal_prop = make_bl_asm_propagator(
        nominal_input,
        grid,
        wavelength_m,
        bandlimit=True,
        include_evanescent=False,
    )
    nominal_fields = [nominal_prop(float(z)) for z in z_absolute_m]
    nominal_intensity = _normalised_stack_from_fields(nominal_fields)

    weights, holdout = _train_holdout_weights(len(images))
    train_indices = np.flatnonzero(weights > 0)

    # Pick a central TRAIN plane, not a held-out plane.
    median_z = np.median(z_relative_m[train_indices])
    reference_index = int(train_indices[np.argmin(np.abs(z_relative_m[train_indices] - median_z))])
    nominal_reference = nominal_fields[reference_index]
    initial_phase = np.angle(nominal_reference)
    support = _confidence_support(amplitudes, reference_index)

    retrieval = multiplane_phase_retrieval(
        amplitudes,
        z_relative_m,
        retrieval_grid,
        reference_index=reference_index,
        initial_phase=initial_phase,
        iterations=iterations,
        relaxation=relaxation,
        support=support,
        plane_weights=weights,
    )
    retrieval.normalization_scales = removed_power
    reconstructed = retrieval.predicted_intensity

    residual_phase, confidence = ideal_relative_phase(
        retrieval.reference_field,
        nominal_reference,
        support=support,
        smooth_sigma_px=1.5,
        confidence_floor=0.02,
    )
    correction_phase = phase_screen_from_residual(residual_phase, gain=1.0)

    # A camera/reference-plane virtual correction.  This does NOT pretend that
    # the same array can be copied to SLM2.
    camera_corrected_reference = apply_phase_screen(
        retrieval.reference_field, correction_phase
    )
    camera_corrected_fields = propagate_stack(
        camera_corrected_reference, z_relative_m, reference_index, retrieval_grid
    )
    camera_corrected_intensity = _normalised_stack_from_fields(camera_corrected_fields)

    # Strong falsification: if the residual phase really describes the lab
    # phase error, adding it to the NOMINAL reference field should recreate the
    # measured distortion on held-out z planes.
    residual_for_screen = np.where(np.isfinite(residual_phase), residual_phase, 0.0)
    error_reference = nominal_reference * np.exp(1j * residual_for_screen)
    error_fields = propagate_stack(
        error_reference, z_relative_m, reference_index, retrieval_grid
    )
    error_intensity = _normalised_stack_from_fields(error_fields)

    roi = grid["R"] <= 250e-6
    metrics = _metric_rows(
        measured,
        reconstructed,
        nominal_intensity,
        error_intensity,
        camera_corrected_intensity,
        z_relative_mm,
        roi,
        holdout,
    )
    metrics.to_csv(output_dir / "global_multiplane_metrics.csv", index=False)

    hold = metrics[metrics.split == "holdout"]
    if hold.empty:
        hold = metrics
    rec_gain = float(
        hold.reconstruction_vs_measured_corr.median()
        - hold.nominal_vs_measured_corr.median()
    )
    recreation_gain = float(
        hold.phase_error_recreation_vs_measured_corr.median()
        - hold.nominal_vs_measured_corr.median()
    )
    recreation_rmse_gain = float(
        hold.nominal_vs_measured_rmse.median()
        - hold.phase_error_recreation_vs_measured_rmse.median()
    )
    supports_phase = bool(recreation_gain >= 0.03 and recreation_rmse_gain > 0)

    summary = {
        "method": "one global complex reference field constrained by multi-plane intensity",
        "propagation": "Matsushima-style band-limited angular spectrum",
        "q": int(q),
        "kr_rad_per_um": float(kr_m_inv * 1e-6),
        "planes": len(images),
        "training_planes": train_indices.tolist(),
        "held_out_planes": holdout.tolist(),
        "reference_plane_index": int(reference_index),
        "reference_z_relative_mm": float(z_relative_mm[reference_index]),
        "reference_z_absolute_model_mm": float(z_absolute_m[reference_index] * 1e3),
        "effective_retrieval_pixel_um": float(dx_m * 1e6),
        "camera_plane_correction_only": True,
        "hardware_ready": False,
        "hardware_blocker": (
            "camera-to-SLM optical transform, relay magnification, parity/rotation, "
            "illuminated footprint and 1030-nm phase LUT not calibrated"
        ),
        "total_phase_is_aberration": False,
        "residual_definition": "angle(U_retrieved * conj(U_complete_nominal_reference))",
        "heldout_reconstruction_corr_gain_over_nominal": rec_gain,
        "heldout_phase_error_recreation_corr_gain_over_nominal": recreation_gain,
        "heldout_phase_error_recreation_rmse_reduction": recreation_rmse_gain,
        "supports_residual_phase_as_cause_of_measured_distortion": supports_phase,
        "claim": (
            "A hardware correction is NOT validated. The residual phase is supported "
            "as an explanatory camera-plane phase error only if the held-out inverse-error "
            "recreation gate passes."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    np.savez_compressed(
        output_dir / "global_multiplane_retrieval.npz",
        z_relative_mm=z_relative_mm,
        z_absolute_model_mm=z_absolute_m * 1e3,
        measured=measured,
        reconstructed=reconstructed,
        nominal=nominal_intensity,
        phase_error_recreation=error_intensity,
        virtual_camera_plane_corrected=camera_corrected_intensity,
        reference_field=retrieval.reference_field.astype(np.complex64),
        nominal_reference_field=nominal_reference.astype(np.complex64),
        residual_phase_rad=residual_phase.astype(np.float32),
        residual_confidence=confidence.astype(np.float32),
        camera_plane_conjugate_phase_rad=correction_phase.astype(np.float32),
        loss_history=retrieval.loss_history,
        removed_plane_power=removed_power,
    )

    _write_phase_figure(
        output_dir,
        retrieval.reference_field,
        nominal_reference,
        residual_phase,
        confidence,
        dx_m,
        z_relative_mm[reference_index],
    )
    _write_longitudinal_figure(
        output_dir,
        measured,
        reconstructed,
        nominal_intensity,
        error_intensity,
        camera_corrected_intensity,
        z_relative_mm,
        dx_m,
    )
    _write_selected_planes(
        output_dir,
        measured,
        reconstructed,
        nominal_intensity,
        error_intensity,
        camera_corrected_intensity,
        z_relative_mm,
        dx_m,
        holdout,
    )
    _write_metrics(output_dir, metrics, retrieval.loss_history)
    return metrics, summary


def _phase_extent_mm(n, dx_m):
    half = n * dx_m * 0.5 * 1e3
    return [-half, half, -half, half]


def _write_phase_figure(
    output_dir,
    retrieved_field,
    nominal_reference,
    residual_phase,
    confidence,
    dx_m,
    z_ref_mm,
):
    extent = _phase_extent_mm(retrieved_field.shape[0], dx_m)
    total = np.angle(retrieved_field)
    nominal = np.angle(nominal_reference)
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), constrained_layout=True)
    entries = (
        (total, "TOTAL RETRIEVED PHASE\ncontains q=20 + nominal propagation", "twilight", -np.pi, np.pi),
        (nominal, "COMPLETE NOMINAL PHASE\nreference model", "twilight", -np.pi, np.pi),
        (residual_phase, "RESIDUAL PHASE\nretrieved x nominal*", "twilight", -np.pi, np.pi),
        (confidence, "RESIDUAL CONFIDENCE\nphase display validity", "viridis", 0.0, 1.0),
    )
    for ax, (array, title, cmap, lo, hi) in zip(axes, entries):
        im = ax.imshow(array, origin="lower", extent=extent, cmap=cmap, vmin=lo, vmax=hi)
        ax.set(title=title, xlabel="x (mm)", ylabel="y (mm)")
        fig.colorbar(im, ax=ax, shrink=.80)
    fig.suptitle(
        f"Physics-consistent phase decomposition at reference z={z_ref_mm:g} mm\n"
        "Do not interpret the wrapped total q=20 phase as aberration",
        fontsize=14,
    )
    fig.savefig(output_dir / "phase_decomposition_residual_not_total.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_longitudinal_figure(
    output_dir,
    measured,
    reconstructed,
    nominal,
    error_recreation,
    virtual_corrected,
    z_mm,
    dx_m,
):
    stacks = (measured, reconstructed, nominal, error_recreation, virtual_corrected)
    titles = (
        "MEASURED MORPHOLOGY",
        "ONE-FIELD RECONSTRUCTION",
        "NOMINAL MODEL",
        "NOMINAL + RETRIEVED RESIDUAL",
        "VIRTUAL CAMERA-PLANE CONJUGATE",
    )
    n = measured.shape[-1]
    half_mm = n * dx_m * .5 * 1e3
    extent = [-half_mm, half_mm, float(z_mm[0]), float(z_mm[-1])]
    fig, axes = plt.subplots(2, len(stacks), figsize=(22, 8.5), constrained_layout=True)
    for col, (stack, title) in enumerate(zip(stacks, titles)):
        xz, yz = signed_longitudinal_sections(stack, band_px=2)
        for row, (section, plane) in enumerate(((xz, "X-Z"), (yz, "Y-Z"))):
            im = axes[row, col].imshow(
                section,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="inferno",
                vmin=0,
                vmax=1,
            )
            axes[row, col].set(
                title=f"{title}\n{plane}",
                xlabel="signed transverse coordinate (mm)",
                ylabel="relative z (mm)",
                xlim=(-0.55, 0.55),
            )
    fig.colorbar(im, ax=axes, label="plane-normalized intensity", shrink=.75)
    fig.suptitle(
        "One reference field -> all z planes (no per-plane phase refit; no accidental absolute-z double count)\n"
        "virtual conjugate is a camera-plane diagnostic, not an SLM result",
        fontsize=14,
    )
    fig.savefig(output_dir / "global_one_field_signed_xz_yz.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_selected_planes(
    output_dir,
    measured,
    reconstructed,
    nominal,
    error_recreation,
    virtual_corrected,
    z_mm,
    dx_m,
    holdout,
):
    if len(holdout):
        chosen = np.unique(np.asarray([holdout[0], holdout[len(holdout)//2], holdout[-1]], int))
    else:
        chosen = np.unique(np.asarray([0, len(z_mm)//2, len(z_mm)-1], int))
    stacks = (measured, reconstructed, nominal, error_recreation, virtual_corrected)
    titles = ("MEASURED", "ONE-FIELD RECON", "NOMINAL", "ERROR RECREATION", "VIRTUAL CONJUGATE")
    n = measured.shape[-1]
    half_mm = n * dx_m * .5 * 1e3
    extent = [-half_mm, half_mm, -half_mm, half_mm]
    fig, axes = plt.subplots(len(chosen), len(stacks), figsize=(18, 3.6*len(chosen)),
                             constrained_layout=True, squeeze=False)
    holdout_set = set(map(int, holdout))
    for row, idx in enumerate(chosen):
        for col, (stack, title) in enumerate(zip(stacks, titles)):
            im = axes[row, col].imshow(
                stack[idx], origin="lower", extent=extent, cmap="inferno", vmin=0, vmax=1
            )
            split = "HELD OUT" if idx in holdout_set else "TRAIN"
            axes[row, col].set(
                title=title,
                xlabel="x (mm)",
                ylabel="y (mm)",
                xlim=(-0.25, 0.25),
                ylim=(-0.25, 0.25),
            )
            axes[row, col].text(
                .02, .97, f"z={z_mm[idx]:g} mm | {split}",
                transform=axes[row, col].transAxes,
                va="top", color="white", fontsize=8,
                bbox=dict(facecolor="black", alpha=.55, edgecolor="none", pad=2),
            )
    fig.colorbar(im, ax=axes, label="plane-normalized intensity", shrink=.7)
    fig.suptitle("Held-out multi-plane reconstruction and phase-error falsification")
    fig.savefig(output_dir / "heldout_selected_planes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_metrics(output_dir, metrics, loss_history):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    z = metrics.z_relative_mm
    axes[0].plot(z, metrics.nominal_vs_measured_corr, "o-", label="nominal vs measured")
    axes[0].plot(z, metrics.reconstruction_vs_measured_corr, "o-", label="one-field recon vs measured")
    axes[0].plot(z, metrics.phase_error_recreation_vs_measured_corr, "o-", label="error recreation vs measured")
    axes[0].set(title="Measured agreement", ylabel="correlation", ylim=(-.05, 1.05))
    axes[1].plot(z, metrics.nominal_vs_measured_rmse, "o-", label="nominal")
    axes[1].plot(z, metrics.reconstruction_vs_measured_rmse, "o-", label="one-field recon")
    axes[1].plot(z, metrics.phase_error_recreation_vs_measured_rmse, "o-", label="error recreation")
    axes[1].set(title="Measured intensity error", ylabel="normalized RMSE")
    axes[2].plot(np.arange(1, len(loss_history)+1), loss_history)
    axes[2].set(title="Global retrieval convergence", xlabel="iteration", ylabel="mean amplitude-shape loss")
    for ax in axes[:2]:
        for zhold in metrics.loc[metrics.split == "holdout", "z_relative_mm"]:
            ax.axvline(zhold, color="0.75", lw=.7, ls=":")
        ax.set_xlabel("relative z (mm)")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    axes[2].grid(alpha=.2)
    fig.suptitle("Physics-consistent multi-plane retrieval metrics (vertical dotted = held out)")
    fig.savefig(output_dir / "global_multiplane_metrics_vs_z.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("BESSEL_ZSCAN_DATA_DIR", HERE / "z-scan 2 1010")),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "outputs" / "physics_consistent_global_multiplane",
    )
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--grid-n", type=int, default=384)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    _, summary = run_global_multiplane(
        args.data_dir,
        args.output_dir,
        iterations=args.iterations,
        output_n=args.grid_n,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
