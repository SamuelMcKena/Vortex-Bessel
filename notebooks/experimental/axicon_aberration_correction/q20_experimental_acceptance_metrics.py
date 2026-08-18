"""Independent measured beam-quality metrics for q=20 correction acceptance.

This module does not consume a retrieved phase map. It compares each measured
BMG plane to a fixed analytical q-th order Bessel target defined by the calibrated
nominal k_perp, in the calibrated camera optical-axis coordinate system. The
same target and geometry are used before and after a correction trial.

The output columns are the contract consumed by
`iterative_correction_controller_v2.evaluate_experimental_update`:
    z_mm
    measured_vs_ideal_corr
    measured_vs_ideal_rmse
    measured_ring_cv
    measured_dark_core_ratio

`measured_ring_cv` is explicitly an AZIMUTHAL coefficient of variation: the
intensity is sampled as a function of theta around the target principal ring,
with a small radial average at every theta. It is not the standard deviation of
all pixels in a thick annulus, which would incorrectly count the ideal radial
Bessel profile itself as non-uniformity.

Additional diagnostic columns are included but do not alter acceptance logic.
"""
from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, special

from modal_vortex_bessel import find_dark_core_center
from run_q20_miao_retrieval import (
    load_scan_preserve_plane_shift, calibrated_axes_in_crop,
)

EPS = 1e-12


def _normalise_roi(image, roi):
    a = np.asarray(image, float)
    scale = max(float(np.max(a[roi])), EPS)
    return a/scale


def _azimuthal_ring_profile(image, center_yx_px, ring_radius_m, pixel_pitch_m,
                            radial_fraction=0.15, n_r=9, n_theta=720):
    """Return I(theta), averaging the same radial samples at every angle."""
    cy, cx = map(float, center_yx_px)
    frac = float(np.clip(radial_fraction, 0.01, 0.30))
    radii_m = ring_radius_m*np.linspace(1.0-frac, 1.0+frac, int(n_r))
    theta = np.linspace(0, 2*np.pi, int(n_theta), endpoint=False)
    rr, tt = np.meshgrid(radii_m/float(pixel_pitch_m), theta, indexing="ij")
    yy = cy + rr*np.sin(tt)
    xx = cx + rr*np.cos(tt)
    sampled = ndimage.map_coordinates(np.asarray(image, float), [yy, xx],
                                       order=1, mode="constant", cval=np.nan)
    profile = np.nanmean(sampled, axis=0)
    if np.sum(np.isfinite(profile)) < 0.9*len(profile):
        raise ValueError("principal-ring profile leaves the camera ROI; check axis/k_perp")
    return profile


def compute_plane_metrics(image, optical_axis_yx_px, *, pixel_pitch_m,
                          q, k_perp_nominal_m_inv, roi_radius_um=160.0,
                          ring_band_fraction=0.15):
    """Compare one measured plane to a fixed ideal J_q(k_perp*r)^2 target."""
    image = np.asarray(image, float)
    cy, cx = map(float, optical_axis_yx_px)
    yy, xx = np.indices(image.shape, dtype=float)
    R_m = np.hypot(yy-cy, xx-cx)*float(pixel_pitch_m)
    roi = R_m <= float(roi_radius_um)*1e-6
    if int(np.sum(roi)) < 100:
        raise ValueError("analysis ROI contains too few pixels; check optical-axis calibration")

    kp = abs(float(k_perp_nominal_m_inv))
    if kp <= 0:
        raise ValueError("k_perp_nominal_m_inv must be positive")
    ideal = special.jv(int(q), kp*R_m)**2
    ideal = _normalise_roi(ideal, roi)
    measured = _normalise_roi(image, roi)

    mv, iv = measured[roi], ideal[roi]
    corr = float(np.corrcoef(mv, iv)[0, 1])
    rmse = float(np.sqrt(np.mean((mv-iv)**2)))

    # First radial intensity maximum of J_q occurs at the first positive zero of J'_q.
    ring_radius_m = float(special.jnp_zeros(int(q), 1)[0]/kp)
    ring_theta = _azimuthal_ring_profile(
        image, (cy, cx), ring_radius_m, pixel_pitch_m,
        radial_fraction=ring_band_fraction, n_r=9, n_theta=720)
    ring_mean = max(float(np.nanmean(ring_theta)), EPS)
    ring_cv = float(np.nanstd(ring_theta)/ring_mean)

    core = R_m < 0.35*ring_radius_m
    if int(np.sum(core)) < 10:
        raise ValueError("target core mask is too small; check pixel pitch and k_perp")
    dark_ratio = float(np.mean(image[core])/ring_mean)

    detected_cy, detected_cx, core_score = find_dark_core_center(image)
    core_offset_um = float(np.hypot(detected_cy-cy, detected_cx-cx)*pixel_pitch_m*1e6)

    # Measured radial location of the strongest annular band near the target ring.
    radii = np.linspace(max(2e-6, 0.55*ring_radius_m), 1.55*ring_radius_m, 120)
    radial_mean = []
    dr = max(1.25*pixel_pitch_m, 1.5e-6)
    for r in radii:
        band = np.abs(R_m-r) <= dr
        radial_mean.append(float(np.mean(image[band])) if np.any(band) else 0.0)
    measured_ring_radius_um = float(radii[int(np.argmax(radial_mean))]*1e6)

    return {
        "measured_vs_ideal_corr": corr,
        "measured_vs_ideal_rmse": rmse,
        "measured_ring_cv": ring_cv,
        "measured_dark_core_ratio": dark_ratio,
        "target_ring_radius_um": ring_radius_m*1e6,
        "measured_ring_radius_um": measured_ring_radius_um,
        "measured_core_offset_from_optical_axis_um": core_offset_um,
        "dark_core_detection_score": float(core_score),
    }


def run(data_dir, calibration_json, output_dir, *,
        z_relative_mm=np.arange(-17.0, 1.0), wavelength_m=1030e-9,
        pixel_pitch_m=5.5e-6, q=20, roi_radius_um=160.0):
    """Generate a before/after-compatible experimental metrics CSV from BMG data."""
    del wavelength_m  # retained in signature/provenance for the 1030-nm experiment
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(Path(calibration_json).read_text(encoding="utf-8"))
    kp = calibration.get("k_perp_nominal_m_inv")
    if kp is None:
        raise RuntimeError("k_perp_nominal_m_inv is required for independent acceptance metrics")

    (images, z_keys, estimated_axis, crop_origin, mean_shifts,
     sensor_shape, qc) = load_scan_preserve_plane_shift(data_dir)
    del z_keys, qc
    z_relative_mm = np.asarray(z_relative_mm, float)
    if len(images) != len(z_relative_mm):
        raise ValueError("z_relative_mm must contain one value per measured plane")
    axes, axis_calibrated, axis_source = calibrated_axes_in_crop(
        calibration, estimated_axis, crop_origin, mean_shifts, len(images))
    if not axis_calibrated:
        raise RuntimeError(
            "Independent acceptance metrics require measured camera optical-axis calibration; "
            "the median beam-core diagnostic axis is not accepted")

    rows = []
    for i, (image, zmm) in enumerate(zip(images, z_relative_mm)):
        row = {"z_mm": float(zmm), "z_index": int(i),
               "optical_axis_y_px": float(axes[i, 0]),
               "optical_axis_x_px": float(axes[i, 1])}
        row.update(compute_plane_metrics(
            image, axes[i], pixel_pitch_m=pixel_pitch_m, q=q,
            k_perp_nominal_m_inv=float(kp), roi_radius_um=roi_radius_um))
        rows.append(row)
    metrics = pd.DataFrame(rows)
    csv_path = out/"experimental_acceptance_metrics.csv"
    metrics.to_csv(csv_path, index=False)

    summary = {
        "method": "measured BMG versus fixed analytical ideal; independent of retrieved correction phase",
        "ring_cv_definition": "azimuthal CV of radially averaged principal-ring I(theta)",
        "planes": int(len(metrics)),
        "q": int(q),
        "k_perp_nominal_m_inv": float(kp),
        "pixel_pitch_m": float(pixel_pitch_m),
        "roi_radius_um": float(roi_radius_um),
        "camera_axis_source": axis_source,
        "sensor_shape_yx": list(map(int, sensor_shape)),
        "median_measured_vs_ideal_corr": float(metrics.measured_vs_ideal_corr.median()),
        "median_measured_vs_ideal_rmse": float(metrics.measured_vs_ideal_rmse.median()),
        "median_measured_ring_cv": float(metrics.measured_ring_cv.median()),
        "maximum_measured_dark_core_ratio": float(metrics.measured_dark_core_ratio.max()),
        "metrics_csv": csv_path.name,
    }
    (out/"experimental_acceptance_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes_plot = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes_plot[0, 0].plot(metrics.z_mm, metrics.measured_vs_ideal_corr, "o-")
    axes_plot[0, 0].set(title="Measured vs ideal", ylabel="correlation")
    axes_plot[0, 1].plot(metrics.z_mm, metrics.measured_vs_ideal_rmse, "o-")
    axes_plot[0, 1].set(title="Measured vs ideal", ylabel="RMSE")
    axes_plot[1, 0].plot(metrics.z_mm, metrics.measured_ring_cv, "o-")
    axes_plot[1, 0].set(title="Principal-ring uniformity", ylabel="azimuthal CV")
    axes_plot[1, 1].plot(metrics.z_mm, metrics.measured_dark_core_ratio, "o-")
    axes_plot[1, 1].set(title="Dark-core preservation", ylabel="core / ring mean")
    for ax in axes_plot.ravel():
        ax.set_xlabel("relative z (mm)")
        ax.grid(alpha=.25)
    fig.suptitle("q=20 experimental acceptance metrics — measured data only vs fixed target")
    fig.savefig(out/"experimental_acceptance_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return metrics, summary


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    calibration = here/"q20_hardware_calibration.json"
    if not calibration.exists():
        raise SystemExit("Create q20_hardware_calibration.json from the template first")
    run(here/"z-scan 2 1010", calibration,
        here/"outputs"/"miao_full_q20"/"experimental_metrics_before")
