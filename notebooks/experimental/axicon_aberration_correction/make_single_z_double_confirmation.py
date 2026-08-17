"""Focused one-plane double check of the recovered q=20 correction phase.

Row 1 applies the correction to the retrieved laboratory field and compares the
phase-only prediction with a normal circular vortex--Bessel target.  Row 2
applies the exact inverse phase to that ideal target and compares the recreated
error with the laboratory camera image.
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from modal_vortex_bessel import modal_basis, corrected_coefficients_phase_only


def _normalise(a):
    a = np.asarray(a, float)
    return a / max(float(np.max(a)), 1e-12)


def _similarity(a, b, mask):
    av, bv = a[mask], b[mask]
    return (float(np.corrcoef(av, bv)[0, 1]),
            float(np.sqrt(np.mean((av - bv) ** 2))))


def _radial_residual(image, radius_m, bin_width_m=2e-6):
    """Remove the circular radial mean so correlation tests only error structure."""
    bins = np.floor(radius_m / bin_width_m).astype(int)
    count = np.bincount(bins.ravel())
    mean = (np.bincount(bins.ravel(), weights=image.ravel()) /
            np.maximum(count, 1))
    return image - mean[bins]


def _coefficients_from_metrics(row, m_values):
    coeffs = []
    for m in m_values:
        amplitude = float(row[f"cm{m:+d}_abs_rel"])
        phase = float(row[f"cm{m:+d}_phase"])
        coeffs.append(amplitude * np.exp(1j * phase))
    return np.asarray(coeffs, complex)


def make_double_confirmation(modal_dir, output_path, *, z_relative_mm=-10.0,
                             q=20, kr_m_inv=489678.1594027835):
    modal_dir = Path(modal_dir)
    recreation_path = modal_dir / "phase_error_recreation" / "phase_error_recreation_stack.npz"
    data = np.load(recreation_path)
    z_values = data["z_mm"]
    index = int(np.argmin(np.abs(z_values - float(z_relative_mm))))
    selected_z = float(z_values[index])
    axis = data["x_um"]
    measured = _normalise(data["measured"][index])
    ideal = _normalise(data["ideal"])
    recreated_error = _normalise(data["ideal_with_retrieved_phase_error"][index])
    full_fit = _normalise(data["full_modal_fit"][index])

    metrics_table = pd.read_csv(modal_dir / "modal_fit_metrics.csv")
    row = metrics_table.iloc[int(np.argmin(np.abs(metrics_table.z_mm - selected_z)))]
    m_values = np.arange(-8, 9, dtype=int)
    coeffs = _coefficients_from_metrics(row, m_values)
    corrected_coeffs, _, _ = corrected_coefficients_phase_only(coeffs, m_values)

    x_m = axis * 1e-6
    X, Y = np.meshgrid(x_m, x_m, indexing="xy")
    R = np.hypot(X, Y)
    PHI = np.arctan2(Y, X)
    basis, _ = modal_basis(q, m_values, kr_m_inv, R.ravel(), PHI.ravel())
    corrected = _normalise(
        (np.abs(basis @ corrected_coeffs) ** 2).reshape(len(axis), len(axis)))

    roi = R <= 160e-6
    measured_ideal_corr, measured_ideal_rmse = _similarity(measured, ideal, roi)
    corrected_ideal_corr, corrected_ideal_rmse = _similarity(corrected, ideal, roi)
    recreated_measured_corr, recreated_measured_rmse = _similarity(
        recreated_error, measured, roi)
    recreated_fit_corr, recreated_fit_rmse = _similarity(recreated_error, full_fit, roi)
    structure_roi = (R >= 25e-6) & (R <= 160e-6)
    measured_residual = _radial_residual(measured, R)
    recreated_residual = _radial_residual(recreated_error, R)
    residual_structure_corr, _ = _similarity(
        measured_residual, recreated_residual, structure_roi)

    # The angular intensity around the principal ring is especially sensitive
    # to whether the recovered aberration points in the same directions as the
    # laboratory lobes; common ring radius/background cannot inflate this test.
    theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    radii_um = np.linspace(36, 55, 12)
    rr, tt = np.meshgrid(radii_um, theta, indexing="ij")
    pixel_um = float(axis[1] - axis[0])
    centre = (len(axis) - 1) / 2
    yy = centre + rr * np.sin(tt) / pixel_um
    xx = centre + rr * np.cos(tt) / pixel_um
    measured_ring = ndimage.map_coordinates(measured, [yy, xx], order=1).mean(axis=0)
    recreated_ring = ndimage.map_coordinates(recreated_error, [yy, xx], order=1).mean(axis=0)
    ring_angular_corr = float(np.corrcoef(measured_ring, recreated_ring)[0, 1])

    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.45), constrained_layout=True)
    rows = (
        ((measured, "LAB MEASURED ERROR"),
         (corrected, "+ RECOVERED CORRECTION\n(model prediction)"),
         (ideal, "NORMAL IDEAL q=20\nVORTEX–BESSEL")),
        ((ideal, "NORMAL IDEAL q=20\nVORTEX–BESSEL"),
         (recreated_error, "+ INVERSE CORRECTION\n(recovered error phase)"),
         (measured, "LAB MEASURED ERROR")),
    )
    for plot_row, items in zip(axes, rows):
        for ax, (image, title) in zip(plot_row, items):
            shown = ax.imshow(image, origin="lower", extent=extent, cmap="inferno",
                              vmin=0, vmax=1, interpolation="nearest")
            ax.axhline(0, color="white", lw=.35, alpha=.45)
            ax.axvline(0, color="white", lw=.35, alpha=.45)
            ax.set(title=title, xlabel="x (um)", ylabel="y (um)")
            ax.set_aspect("equal")

    axes[0, 0].text(.02, .03, "CORRECTION TEST →", transform=axes[0, 0].transAxes,
                    color="cyan", fontsize=9, weight="bold")
    axes[1, 0].text(.02, .03, "ERROR RECREATION TEST →", transform=axes[1, 0].transAxes,
                    color="cyan", fontsize=9, weight="bold")
    axes[0, 2].text(.02, .03, f"corrected vs ideal r={corrected_ideal_corr:.3f}",
                    transform=axes[0, 2].transAxes, color="white", fontsize=9,
                    bbox=dict(facecolor="black", alpha=.55, edgecolor="none", pad=2))
    axes[1, 2].text(.02, .03,
                    f"global r={recreated_measured_corr:.3f}\n"
                    f"error residual r={residual_structure_corr:.3f}\n"
                    f"ring angular r={ring_angular_corr:.3f} — NO MATCH",
                    transform=axes[1, 2].transAxes, color="white", fontsize=9,
                    bbox=dict(facecolor="black", alpha=.55, edgecolor="none", pad=2))
    fig.colorbar(shown, ax=axes, label="plane-normalized intensity", shrink=.86)
    fig.suptitle(f"Double confirmation of recovered phase at relative z={selected_z:g} mm\n"
                 "same phase map, opposite signs; ideal is the circular analytic vortex–Bessel field",
                 fontsize=14)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)

    result = {
        "z_relative_mm": selected_z,
        "measured_vs_ideal_corr_before": measured_ideal_corr,
        "measured_vs_ideal_rmse_before": measured_ideal_rmse,
        "corrected_model_vs_ideal_corr": corrected_ideal_corr,
        "corrected_model_vs_ideal_rmse": corrected_ideal_rmse,
        "ideal_plus_inverse_correction_vs_measured_corr": recreated_measured_corr,
        "ideal_plus_inverse_correction_vs_measured_rmse": recreated_measured_rmse,
        "ideal_plus_inverse_correction_vs_full_modal_fit_corr": recreated_fit_corr,
        "ideal_plus_inverse_correction_vs_full_modal_fit_rmse": recreated_fit_rmse,
        "ideal_plus_inverse_correction_vs_lab_nonaxisymmetric_residual_corr": residual_structure_corr,
        "ideal_plus_inverse_correction_vs_lab_ring_angular_corr": ring_angular_corr,
        "visual_verdict": "Does not reproduce the laboratory angular/fan error structure.",
        "important_scope": "Correction result is a per-plane phase-retrieval prediction, not a post-SLM camera measurement.",
    }
    output_path.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    modal = here / "outputs" / "slm_closed_loop_alignment" / "modal_q20"
    result = make_double_confirmation(
        modal,
        modal / "single_z_double_confirmation_minus10.png",
        z_relative_mm=-10.0)
    print(json.dumps(result, indent=2))
