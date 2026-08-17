from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vbb_study.calibration.schema import CalibrationBundle, canonical_calibration_template, measurement
from vbb_study.calibration.slm_phase import SLMPhaseCalibration
from vbb_study.digital_twin.bench_calibrated_vector_route import BenchCalibratedVectorInputs
from vbb_study.digital_twin.bench_calibrated_vector_tilt_route import build_calibrated_segmented_vector_tilt_route
from vbb_study.digital_twin.nathan_vector_hexagon import NathanHexagonConfig
from vbb_study.digital_twin.vector_tilt_study import beam_moment_metrics, vector_line_intensity
from vbb_study.vector_field import VectorField, propagate_vector_asm


TWOPI = 2.0 * np.pi
PRIMARY_TILTS_DEG = (-2.0, -1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
STRESS_TILTS_DEG = (-10.0, -5.0, 5.0, 10.0)
ALL_TILTS_DEG = PRIMARY_TILTS_DEG + STRESS_TILTS_DEG
REPRESENTATIVE_DEG = (-2.0, -1.0, 0.0, 1.0, 2.0)
AZIMUTH_MAGNITUDES_DEG = (1.0, 2.0)
AZIMUTH_DEG = tuple(float(v) for v in range(0, 360, 30))
Z_REF_MM = 30.0
Z_LONG_MM = np.arange(0.0, 60.0 + 1e-9, 2.0)


def _lut(panel: str) -> SLMPhaseCalibration:
    grey = np.arange(256, dtype=float)
    return SLMPhaseCalibration(
        panel_id=panel,
        wavelength_m=1029e-9,
        grey_levels=grey,
        phase_rad=TWOPI * grey / 255.0,
        calibration_date="synthetic_phase2h_systematic_study",
    )


def _bundle() -> CalibrationBundle:
    data = canonical_calibration_template()
    data["calibration_id"] = "synthetic_phase2h_systematic_vector_tilt"
    data["data_classification"] = "synthetic_not_experimental"
    data["laser"]["beam_radius_on_slm_m"] = measurement(2.0e-3, 0.0, "synthetic_measurement", "m")
    data["fourier_filter"]["focal_length_m"] = measurement(0.300, 0.0, "synthetic_measurement", "m")
    data["fourier_filter"]["iris_radius_m"] = measurement(0.70e-3, 0.0, "synthetic_measurement", "m")
    axicon = data["axicon"]
    axicon["base_angle_deg"] = measurement(2.0, 0.0, "synthetic_measurement", "deg")
    axicon["refractive_index"] = measurement(1.458, 0.0, "synthetic_measurement", "1")
    axicon["clear_radius_m"] = measurement(3.0e-3, 0.0, "synthetic_measurement", "m")
    axicon["centre_thickness_m"] = measurement(3.0e-3, 0.0, "synthetic_measurement", "m")
    axicon["angle_convention"] = "base_angle_from_flat_face"
    axicon["flat_face_upstream_verified"] = True
    pol = data["polarization"]
    pol["input_linear_angle_deg"] = measurement(45.0, 0.0, "synthetic_measurement", "deg")
    pol["input_degree_linear_polarization"] = measurement(1.0, 0.0, "synthetic_measurement", "1")
    pol["input_relative_phase_rad"] = measurement(0.0, 0.0, "synthetic_measurement", "rad")
    pol["slm_director_axis_deg"] = measurement(0.0, 0.0, "synthetic_measurement", "deg")
    pol["segmented_vector_hwp_retardance_rad"] = measurement(math.pi, 0.0, "synthetic_measurement", "rad")
    pol["segmented_vector_hwp_fast_axis_deg"] = measurement(45.0, 0.0, "synthetic_measurement", "deg")
    pol["segmented_vector_qwp_retardance_rad"] = measurement(0.5 * math.pi, 0.0, "synthetic_measurement", "rad")
    pol["segmented_vector_qwp_fast_axis_deg"] = measurement(-45.0, 0.0, "synthetic_measurement", "deg")
    return CalibrationBundle(data)


def _inputs_pair(bundle: CalibrationBundle, tilt_x_deg: float, tilt_y_deg: float) -> BenchCalibratedVectorInputs:
    return BenchCalibratedVectorInputs(
        calibration_bundle=bundle,
        slm1_phase_calibration=_lut("synthetic_slm1"),
        slm2_phase_calibration=_lut("synthetic_slm2"),
        axicon_tilt_rad=(math.radians(float(tilt_x_deg)), math.radians(float(tilt_y_deg))),
    )


def _build_case_pair(bundle: CalibrationBundle, config: NathanHexagonConfig, tilt_x_deg: float, tilt_y_deg: float) -> dict:
    return build_calibrated_segmented_vector_tilt_route(
        config,
        calibrated=_inputs_pair(bundle, tilt_x_deg, tilt_y_deg),
        vector_axicon_output_n=512,
        vector_axicon_output_window_m=7.2e-3,
        reference_gap_m=0.25e-3,
    )


def _build_axis_case(bundle: CalibrationBundle, config: NathanHexagonConfig, direction: str, tilt_deg: float) -> dict:
    if direction == "x":
        return _build_case_pair(bundle, config, tilt_deg, 0.0)
    if direction == "y":
        return _build_case_pair(bundle, config, 0.0, tilt_deg)
    raise ValueError("direction must be x or y")


def _plane(field: VectorField, z_mm: float) -> VectorField:
    return field if abs(float(z_mm)) < 1e-15 else propagate_vector_asm(field, float(z_mm) * 1e-3)


def _axes_mm(field: VectorField) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(field.grid["x"], dtype=float) * 1e3
    y = np.asarray(field.grid.get("y", field.grid["x"]), dtype=float) * 1e3
    return x, y


def _longitudinal(field: VectorField) -> dict[str, np.ndarray]:
    x_mm, y_mm = _axes_mm(field)
    ix0 = int(np.argmin(np.abs(x_mm)))
    iy0 = int(np.argmin(np.abs(y_mm)))
    xz, yz, cx, cy, peak, power = [], [], [], [], [], []
    for z_mm in Z_LONG_MM:
        f = _plane(field, float(z_mm))
        I = np.asarray(f.intensity, dtype=float)
        m = beam_moment_metrics(f)
        xz.append(I[iy0, :].copy())
        yz.append(I[:, ix0].copy())
        cx.append(m.centroid_x_m * 1e3)
        cy.append(m.centroid_y_m * 1e3)
        peak.append(m.peak_intensity)
        power.append(m.power_au_m2)
    return {
        "x_mm": x_mm,
        "y_mm": y_mm,
        "xz": np.asarray(xz),
        "yz": np.asarray(yz),
        "centroid_x_mm": np.asarray(cx),
        "centroid_y_mm": np.asarray(cy),
        "peak": np.asarray(peak),
        "power": np.asarray(power),
    }


def _centroid_slopes(long: dict[str, np.ndarray]) -> tuple[float, float]:
    z_m = Z_LONG_MM * 1e-3
    cx_m = np.asarray(long["centroid_x_mm"], dtype=float) * 1e-3
    cy_m = np.asarray(long["centroid_y_mm"], dtype=float) * 1e-3
    return float(np.polyfit(z_m, cx_m, 1)[0]), float(np.polyfit(z_m, cy_m, 1)[0])


def _metric_row(
    direction: str,
    tilt_deg: float,
    route: dict,
    long: dict,
    *,
    zero_m,
    zero_slope_x: float,
    zero_slope_y: float,
) -> dict[str, float | str | bool]:
    zref_field = _plane(route["post_axicon"], Z_REF_MM)
    m = beam_moment_metrics(zref_field)
    sx, sy = _centroid_slopes(long)
    dsx = sx - float(zero_slope_x)
    dsy = sy - float(zero_slope_y)
    raw_steering = math.atan(math.hypot(sx, sy))
    induced_steering = math.atan(math.hypot(dsx, dsy))
    if direction == "x":
        cross_delta_mm = (m.centroid_y_m - zero_m.centroid_y_m) * 1e3
        cross_slope = dsy
        cross_axis = "y"
    else:
        cross_delta_mm = (m.centroid_x_m - zero_m.centroid_x_m) * 1e3
        cross_slope = dsx
        cross_axis = "x"
    stable = Z_LONG_MM >= 20.0
    peaks = np.asarray(long["peak"], dtype=float)[stable]
    peak_cv = float(np.std(peaks) / max(float(np.mean(peaks)), np.finfo(float).tiny))
    meta = route["vector_refractive_axicon_result"].metadata
    return {
        "direction": direction,
        "rotation_axis": direction,
        "dominant_steering_coordinate": cross_axis,
        "tilt_deg": float(tilt_deg),
        "stress_case": bool(abs(float(tilt_deg)) > 2.0),
        "z_ref_mm": Z_REF_MM,
        "centroid_x_mm": m.centroid_x_m * 1e3,
        "centroid_y_mm": m.centroid_y_m * 1e3,
        "delta_centroid_x_from_0deg_mm": (m.centroid_x_m - zero_m.centroid_x_m) * 1e3,
        "delta_centroid_y_from_0deg_mm": (m.centroid_y_m - zero_m.centroid_y_m) * 1e3,
        "tilt_induced_cross_axis_centroid_mm": cross_delta_mm,
        "sigma_major_mm": m.sigma_major_m * 1e3,
        "sigma_minor_mm": m.sigma_minor_m * 1e3,
        "ellipticity": m.ellipticity,
        "major_axis_angle_deg": math.degrees(m.major_axis_angle_rad),
        "peak_over_nominal_0deg": m.peak_intensity / max(zero_m.peak_intensity, np.finfo(float).tiny),
        "power_over_nominal_0deg": m.power_au_m2 / max(zero_m.power_au_m2, np.finfo(float).tiny),
        "centroid_slope_x_m_per_m": sx,
        "centroid_slope_y_m_per_m": sy,
        "delta_centroid_slope_x_from_0deg_m_per_m": dsx,
        "delta_centroid_slope_y_from_0deg_m_per_m": dsy,
        "raw_steering_angle_mrad": raw_steering * 1e3,
        "tilt_induced_steering_magnitude_mrad": induced_steering * 1e3,
        "signed_tilt_induced_cross_axis_steering_mrad": math.atan(cross_slope) * 1e3,
        "peak_cv_z20_to_60": peak_cv,
        "final_flux_closure_ratio": float(meta["final_flux_closure_ratio"]),
        "final_transversality_residual": float(meta["final_transversality_residual"]),
        "required_nyquist_fraction": float(meta["required_nyquist_fraction"]),
        "common_eikonal_p95_component_disagreement": float(meta["common_eikonal"]["p95_component_wavevector_disagreement_fraction"]),
        "common_eikonal_p95_gradient_error": float(meta["common_eikonal"]["p95_reconstructed_gradient_error_fraction"]),
    }


def _azimuth_row(
    magnitude_deg: float,
    azimuth_deg: float,
    route: dict,
    *,
    zero_m,
) -> dict[str, float]:
    field = _plane(route["post_axicon"], Z_REF_MM)
    m = beam_moment_metrics(field)
    phi = math.radians(float(azimuth_deg))
    tx = float(magnitude_deg) * math.cos(phi)
    ty = float(magnitude_deg) * math.sin(phi)
    meta = route["vector_refractive_axicon_result"].metadata
    dx = (m.centroid_x_m - zero_m.centroid_x_m) * 1e3
    dy = (m.centroid_y_m - zero_m.centroid_y_m) * 1e3
    return {
        "tilt_magnitude_deg": float(magnitude_deg),
        "tilt_azimuth_deg": float(azimuth_deg),
        "tilt_x_deg": tx,
        "tilt_y_deg": ty,
        "z_ref_mm": Z_REF_MM,
        "delta_centroid_x_mm": dx,
        "delta_centroid_y_mm": dy,
        "delta_centroid_radius_mm": math.hypot(dx, dy),
        "ellipticity": m.ellipticity,
        "peak_over_nominal_0deg": m.peak_intensity / max(zero_m.peak_intensity, np.finfo(float).tiny),
        "power_over_nominal_0deg": m.power_au_m2 / max(zero_m.power_au_m2, np.finfo(float).tiny),
        "final_flux_closure_ratio": float(meta["final_flux_closure_ratio"]),
        "required_nyquist_fraction": float(meta["required_nyquist_fraction"]),
    }


def _figure_response(rows: list[dict], outdir: Path) -> None:
    primary = [r for r in rows if not r["stress_case"]]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.5), constrained_layout=True)
    fields = (
        ("tilt_induced_cross_axis_centroid_mm", "cross-axis centroid shift from 0° at 30 mm (mm)"),
        ("signed_tilt_induced_cross_axis_steering_mrad", "signed tilt-induced cross-axis steering (mrad)"),
        ("tilt_induced_steering_magnitude_mrad", "tilt-induced steering magnitude (mrad)"),
        ("ellipticity", "2nd-moment ellipticity"),
        ("peak_over_nominal_0deg", "peak / 0° peak"),
        ("peak_cv_z20_to_60", "peak CV, z=20–60 mm"),
    )
    for ax, (key, ylabel) in zip(axes.flat, fields):
        for direction in ("x", "y"):
            subset = sorted((r for r in primary if r["direction"] == direction), key=lambda r: float(r["tilt_deg"]))
            ax.plot([r["tilt_deg"] for r in subset], [r[key] for r in subset], marker="o", label=f"rotation about {direction}")
        ax.axvline(0.0, linewidth=0.8)
        if key.startswith("tilt_induced") or key.startswith("signed_tilt"):
            ax.axhline(0.0, linewidth=0.7)
        ax.set_xlabel("rigid axicon tilt (deg)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("Phase 2H systematic six-sector vector tilt response\nzero-tilt drift subtracted for steering observables; zref=30 mm", fontsize=13)
    fig.savefig(outdir / "01_systematic_six_sector_tilt_response.png", dpi=190)
    plt.close(fig)


def _figure_azimuth(rows: list[dict], outdir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    for magnitude in AZIMUTH_MAGNITUDES_DEG:
        subset = sorted((r for r in rows if r["tilt_magnitude_deg"] == magnitude), key=lambda r: r["tilt_azimuth_deg"])
        az = [r["tilt_azimuth_deg"] for r in subset]
        axes[0, 0].plot(az, [r["delta_centroid_x_mm"] for r in subset], marker="o", label=f"|tilt|={magnitude:g}°")
        axes[0, 1].plot(az, [r["delta_centroid_y_mm"] for r in subset], marker="o", label=f"|tilt|={magnitude:g}°")
        axes[1, 0].plot(az, [r["ellipticity"] for r in subset], marker="o", label=f"|tilt|={magnitude:g}°")
        axes[1, 1].plot(az, [r["peak_over_nominal_0deg"] for r in subset], marker="o", label=f"|tilt|={magnitude:g}°")
    axes[0, 0].set_ylabel("Δ centroid x at zref (mm)")
    axes[0, 1].set_ylabel("Δ centroid y at zref (mm)")
    axes[1, 0].set_ylabel("2nd-moment ellipticity")
    axes[1, 1].set_ylabel("peak / 0° peak")
    for ax in axes.flat:
        ax.set_xlabel("tilt-vector azimuth in (tilt_x, tilt_y) plane (deg)")
        ax.set_xticks(np.arange(0, 360, 60))
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("Phase 2H six-sector fixed-magnitude tilt-azimuth scan\ncloses directional sensitivity beyond x/y axes; zref=30 mm", fontsize=13)
    fig.savefig(outdir / "04_six_sector_tilt_azimuth_scan.png", dpi=190)
    plt.close(fig)


def _figure_longitudinal(direction: str, representative: dict[float, dict], outdir: Path) -> None:
    global_peak = max(float(np.max(v["long"]["xz"])) for v in representative.values())
    global_peak = max(global_peak, max(float(np.max(v["long"]["yz"])) for v in representative.values()))
    fig, axes = plt.subplots(len(REPRESENTATIVE_DEG), 2, figsize=(11.0, 13.5), constrained_layout=True)
    for row, tilt in enumerate(REPRESENTATIVE_DEG):
        case = representative[tilt]["long"]
        x, y = case["x_mm"], case["y_mm"]
        axes[row, 0].imshow(case["xz"] / global_peak, origin="lower", extent=[x[0], x[-1], Z_LONG_MM[0], Z_LONG_MM[-1]], vmin=0, vmax=1, aspect="auto")
        axes[row, 1].imshow(case["yz"] / global_peak, origin="lower", extent=[y[0], y[-1], Z_LONG_MM[0], Z_LONG_MM[-1]], vmin=0, vmax=1, aspect="auto")
        axes[row, 0].axhline(Z_REF_MM, linestyle="--", linewidth=0.9)
        axes[row, 1].axhline(Z_REF_MM, linestyle="--", linewidth=0.9)
        axes[row, 0].set_ylabel(f"tilt {tilt:+.1f}°\nz (mm)")
        axes[row, 0].set_xlabel("x (mm)")
        axes[row, 1].set_xlabel("y (mm)")
    axes[0, 0].set_title("fixed-lab x-z")
    axes[0, 1].set_title("fixed-lab y-z")
    fig.suptitle(f"Phase 2H six-sector rotation-about-{direction} longitudinal evidence\ncommon scale; dashed line is canonical zref={Z_REF_MM:.0f} mm", fontsize=13)
    fig.savefig(outdir / f"02_longitudinal_{direction}_tilt_representatives.png", dpi=190)
    plt.close(fig)


def _figure_zref_profiles(direction: str, representative: dict[float, dict], outdir: Path) -> None:
    fields = {tilt: _plane(data["route"]["post_axicon"], Z_REF_MM) for tilt, data in representative.items()}
    nominal_peak = beam_moment_metrics(fields[0.0]).peak_intensity
    coord = np.linspace(-1.2e-3, 1.2e-3, 1201)
    fig, axes = plt.subplots(len(REPRESENTATIVE_DEG), 3, figsize=(14.0, 14.0), constrained_layout=True)
    for row, tilt in enumerate(REPRESENTATIVE_DEG):
        field = fields[tilt]
        m = beam_moment_metrics(field)
        x = np.asarray(field.grid["x"]) * 1e3
        y = np.asarray(field.grid.get("y", field.grid["x"])) * 1e3
        I = field.intensity / nominal_peak
        axes[row, 0].imshow(I, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], vmin=0, vmax=max(1.0, float(np.max(I))), aspect="equal")
        axes[row, 0].plot(m.centroid_x_m * 1e3, m.centroid_y_m * 1e3, "+", markersize=9)
        lab_x, lab_y = vector_line_intensity(field, coord, fixed_x_m=0.0, fixed_y_m=0.0)
        cen_x, cen_y = vector_line_intensity(field, coord, fixed_x_m=m.centroid_x_m, fixed_y_m=m.centroid_y_m)
        axes[row, 1].plot(coord * 1e3, lab_x / nominal_peak, label="lab x through y=0")
        axes[row, 1].plot(coord * 1e3, lab_y / nominal_peak, label="lab y through x=0")
        axes[row, 2].plot(coord * 1e3, cen_x / nominal_peak, label="centred x")
        axes[row, 2].plot(coord * 1e3, cen_y / nominal_peak, label="centred y")
        axes[row, 0].set_ylabel(f"tilt {tilt:+.1f}°\ny (mm)")
        for col in (1, 2):
            axes[row, col].set_xlabel("coordinate (mm)")
            axes[row, col].set_ylabel("I / nominal 0° 2D peak")
            axes[row, col].grid(alpha=0.25)
    axes[0, 0].set_title(f"2D intensity at z={Z_REF_MM:.0f} mm")
    axes[0, 1].set_title("fixed-lab exact profiles")
    axes[0, 2].set_title("centroid-frame exact profiles")
    axes[0, 1].legend(fontsize=8)
    axes[0, 2].legend(fontsize=8)
    fig.suptitle(f"Phase 2H six-sector rotation-about-{direction} canonical profile evidence\nall curves use the same 0° peak normalisation", fontsize=13)
    fig.savefig(outdir / f"03_profiles_zref_{direction}_tilt_representatives.png", dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    bundle = _bundle()
    config = NathanHexagonConfig.fast(grid_n=128)
    rows: list[dict] = []
    representative: dict[str, dict[float, dict]] = {"x": {}, "y": {}}

    zero_route = _build_case_pair(bundle, config, 0.0, 0.0)
    zero_long = _longitudinal(zero_route["post_axicon"])
    zero_zref = _plane(zero_route["post_axicon"], Z_REF_MM)
    zero_m = beam_moment_metrics(zero_zref)
    zero_sx, zero_sy = _centroid_slopes(zero_long)

    for direction in ("x", "y"):
        for tilt in ALL_TILTS_DEG:
            if tilt == 0.0:
                route, long = zero_route, zero_long
            else:
                route = _build_axis_case(bundle, config, direction, tilt)
                long = _longitudinal(route["post_axicon"])
            rows.append(
                _metric_row(
                    direction,
                    tilt,
                    route,
                    long,
                    zero_m=zero_m,
                    zero_slope_x=zero_sx,
                    zero_slope_y=zero_sy,
                )
            )
            if tilt in REPRESENTATIVE_DEG:
                representative[direction][tilt] = {"route": route, "long": long}

    with (outdir / "systematic_vector_tilt_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    azimuth_rows: list[dict] = []
    for magnitude in AZIMUTH_MAGNITUDES_DEG:
        for azimuth in AZIMUTH_DEG:
            phi = math.radians(azimuth)
            tx = magnitude * math.cos(phi)
            ty = magnitude * math.sin(phi)
            route = _build_case_pair(bundle, config, tx, ty)
            azimuth_rows.append(_azimuth_row(magnitude, azimuth, route, zero_m=zero_m))
    with (outdir / "systematic_vector_tilt_azimuth_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(azimuth_rows[0].keys()))
        writer.writeheader()
        writer.writerows(azimuth_rows)

    payload = {
        "outcome": "PHASE2H-SYSTEMATIC-VECTOR-TILT-STUDY-SYNTHETIC",
        "data_classification": "synthetic_not_experimental",
        "report_figures_authorised": False,
        "vector_family": "six_sector_dual_slm_segmented_vector",
        "primary_tilts_deg": list(PRIMARY_TILTS_DEG),
        "stress_tilts_deg": list(STRESS_TILTS_DEG),
        "tilt_directions": ["rotation_about_x", "rotation_about_y"],
        "tilt_azimuth_scan_magnitudes_deg": list(AZIMUTH_MAGNITUDES_DEG),
        "tilt_azimuth_scan_deg": list(AZIMUTH_DEG),
        "canonical_z_ref_mm": Z_REF_MM,
        "z_ref_policy": "a_priori_fixed_physical_plane_shared_by_all_cases",
        "longitudinal_z_mm": Z_LONG_MM.tolist(),
        "zero_tilt_centroid_slope_x_m_per_m": zero_sx,
        "zero_tilt_centroid_slope_y_m_per_m": zero_sy,
        "steering_policy": "retain raw trajectory but use zero_tilt_subtracted_slopes for tilt-induced steering",
        "profile_sampling": "direct_complex_Fourier_series_componentwise_then_sum_intensity",
        "normalisation": "fixed_0deg_peak_at_same_zref",
        "rows": rows,
        "azimuth_rows": azimuth_rows,
    }
    (outdir / "systematic_vector_tilt_study.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _figure_response(rows, outdir)
    _figure_azimuth(azimuth_rows, outdir)
    for direction in ("x", "y"):
        _figure_longitudinal(direction, representative[direction], outdir)
        _figure_zref_profiles(direction, representative[direction], outdir)
    print(
        json.dumps(
            {
                "output_dir": str(outdir),
                "axis_sweep_case_count": len(rows),
                "azimuth_scan_case_count": len(azimuth_rows),
                "canonical_z_ref_mm": Z_REF_MM,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
