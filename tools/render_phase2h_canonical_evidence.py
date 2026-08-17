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
from vbb_study.reporting.evidence_conventions import (
    CanonicalEvidenceSpec,
    DEFAULT_FIGURE_DPI,
    INTENSITY_CMAP,
    canonical_evidence_filenames,
    common_positive_peak,
    nominal_profile_scale,
)
from vbb_study.vector_field import VectorField, propagate_vector_asm


TWOPI = 2.0 * np.pi
Z_REF_MM = 30.0
Z_MM = np.arange(0.0, 60.0 + 1e-12, 1.0)
REPRESENTATIVE_TILTS_DEG = (-2.0, -1.0, 0.0, 1.0, 2.0)
DETAIL_HALF_WIDTH_MM = 1.20
PROFILE_HALF_WIDTH_MM = 1.20
PROFILE_SAMPLES = 1201


def _lut(panel: str) -> SLMPhaseCalibration:
    grey = np.arange(256, dtype=float)
    return SLMPhaseCalibration(
        panel_id=panel,
        wavelength_m=1029e-9,
        grey_levels=grey,
        phase_rad=TWOPI * grey / 255.0,
        calibration_date="synthetic_phase2h_canonical_evidence",
    )


def _bundle() -> CalibrationBundle:
    data = canonical_calibration_template()
    data["calibration_id"] = "synthetic_phase2h_canonical_evidence"
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


def _inputs(bundle: CalibrationBundle, direction: str, tilt_deg: float) -> BenchCalibratedVectorInputs:
    angle = math.radians(float(tilt_deg))
    tilt = (angle, 0.0) if direction == "x" else (0.0, angle)
    return BenchCalibratedVectorInputs(
        calibration_bundle=bundle,
        slm1_phase_calibration=_lut("synthetic_slm1"),
        slm2_phase_calibration=_lut("synthetic_slm2"),
        axicon_tilt_rad=tilt,
    )


def _build_source(bundle: CalibrationBundle, direction: str, tilt_deg: float) -> tuple[VectorField, dict]:
    route = build_calibrated_segmented_vector_tilt_route(
        NathanHexagonConfig.fast(grid_n=128),
        calibrated=_inputs(bundle, direction, tilt_deg),
        vector_axicon_output_n=512,
        vector_axicon_output_window_m=7.2e-3,
        reference_gap_m=0.25e-3,
    )
    return route["post_axicon"], route


def _at_z(source: VectorField, z_mm: float) -> VectorField:
    if abs(float(z_mm)) < 1e-14:
        return source
    return propagate_vector_asm(source, float(z_mm) * 1e-3)


def _longitudinal(source: VectorField) -> dict[str, np.ndarray | float]:
    x_mm = np.asarray(source.grid["x"], dtype=float) * 1e3
    y_mm = np.asarray(source.grid.get("y", source.grid["x"]), dtype=float) * 1e3
    ix = int(np.argmin(np.abs(x_mm)))
    iy = int(np.argmin(np.abs(y_mm)))
    fixed_x_mm = float(x_mm[ix])
    fixed_y_mm = float(y_mm[iy])
    xz: list[np.ndarray] = []
    yz: list[np.ndarray] = []
    for z in Z_MM:
        field = _at_z(source, float(z))
        intensity = np.asarray(field.intensity, dtype=float)
        xz.append(intensity[iy, :].copy())
        yz.append(intensity[:, ix].copy())
    return {
        "x_mm": x_mm,
        "y_mm": y_mm,
        "xz": np.asarray(xz),
        "yz": np.asarray(yz),
        "fixed_x_mm": fixed_x_mm,
        "fixed_y_mm": fixed_y_mm,
    }


def _profile_data(field: VectorField) -> dict[str, np.ndarray | float]:
    coord = np.linspace(-PROFILE_HALF_WIDTH_MM, PROFILE_HALF_WIDTH_MM, PROFILE_SAMPLES) * 1e-3
    moments = beam_moment_metrics(field)
    lab_x, lab_y = vector_line_intensity(field, coord, fixed_x_m=0.0, fixed_y_m=0.0)
    centred_x, centred_y = vector_line_intensity(
        field,
        coord,
        fixed_x_m=moments.centroid_x_m,
        fixed_y_m=moments.centroid_y_m,
    )
    return {
        "coord_m": coord,
        "lab_x": lab_x,
        "lab_y": lab_y,
        "centred_x": centred_x,
        "centred_y": centred_y,
        "centroid_x_m": moments.centroid_x_m,
        "centroid_y_m": moments.centroid_y_m,
        "peak_2d": moments.peak_intensity,
        "power": moments.power_au_m2,
    }


def _render_longitudinal(
    family: str,
    direction: str,
    data: dict[float, dict[str, np.ndarray | float]],
    outdir: Path,
    filenames: dict[str, str],
) -> float:
    common_peak = common_positive_peak(
        [case[axis] for case in data.values() for axis in ("xz", "yz")]
    )
    fig, axes = plt.subplots(len(REPRESENTATIVE_TILTS_DEG), 2, figsize=(10.8, 13.4), constrained_layout=True)
    last = None
    for row, tilt in enumerate(REPRESENTATIVE_TILTS_DEG):
        case = data[tilt]
        x = np.asarray(case["x_mm"], dtype=float)
        y = np.asarray(case["y_mm"], dtype=float)
        xz = np.asarray(case["xz"], dtype=float) / common_peak
        yz = np.asarray(case["yz"], dtype=float) / common_peak
        last = axes[row, 0].imshow(
            xz,
            origin="lower",
            extent=[float(x[0]), float(x[-1]), float(Z_MM[0]), float(Z_MM[-1])],
            cmap=INTENSITY_CMAP,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            aspect="auto",
        )
        axes[row, 1].imshow(
            yz,
            origin="lower",
            extent=[float(y[0]), float(y[-1]), float(Z_MM[0]), float(Z_MM[-1])],
            cmap=INTENSITY_CMAP,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            aspect="auto",
        )
        for col in (0, 1):
            axes[row, col].set_ylim(float(Z_MM[0]), float(Z_MM[-1]))
            axes[row, col].axhline(Z_REF_MM, linewidth=0.8, linestyle="--", color="white", alpha=0.85)
        axes[row, 0].set_xlim(-DETAIL_HALF_WIDTH_MM, DETAIL_HALF_WIDTH_MM)
        axes[row, 1].set_xlim(-DETAIL_HALF_WIDTH_MM, DETAIL_HALF_WIDTH_MM)
        axes[row, 0].set_ylabel(f"tilt {tilt:+g}°\nz (mm)")
        axes[row, 0].set_xlabel("x (mm)")
        axes[row, 1].set_xlabel("y (mm)")
    axes[0, 0].set_title("fixed-lab x-z")
    axes[0, 1].set_title("fixed-lab y-z")
    if last is not None:
        fig.colorbar(last, ax=axes, shrink=0.82, label="I / common longitudinal peak")
    fig.suptitle(
        f"{family}: canonical fixed-laboratory longitudinal evidence\n"
        f"rotation about {direction}; linear common scale; dashed line zref={Z_REF_MM:g} mm; SYNTHETIC",
        fontsize=13,
    )
    fig.savefig(outdir / filenames["longitudinal"], dpi=DEFAULT_FIGURE_DPI)
    plt.close(fig)
    return common_peak


def _render_zref(
    family: str,
    direction: str,
    fields: dict[float, VectorField],
    profiles: dict[float, dict[str, np.ndarray | float]],
    outdir: Path,
    filenames: dict[str, str],
) -> tuple[float, float]:
    heatmap_peak = common_positive_peak([field.intensity for field in fields.values()])
    nominal_peak = nominal_profile_scale(fields[0.0].intensity)
    fig, axes = plt.subplots(len(REPRESENTATIVE_TILTS_DEG), 3, figsize=(13.2, 13.5), constrained_layout=True)
    last = None
    for row, tilt in enumerate(REPRESENTATIVE_TILTS_DEG):
        field = fields[tilt]
        profile = profiles[tilt]
        x = np.asarray(field.grid["x"], dtype=float) * 1e3
        y = np.asarray(field.grid.get("y", field.grid["x"]), dtype=float) * 1e3
        last = axes[row, 0].imshow(
            np.asarray(field.intensity, dtype=float) / heatmap_peak,
            origin="lower",
            extent=[float(x[0]), float(x[-1]), float(y[0]), float(y[-1])],
            cmap=INTENSITY_CMAP,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            aspect="equal",
        )
        axes[row, 0].set_xlim(-DETAIL_HALF_WIDTH_MM, DETAIL_HALF_WIDTH_MM)
        axes[row, 0].set_ylim(-DETAIL_HALF_WIDTH_MM, DETAIL_HALF_WIDTH_MM)
        axes[row, 0].set_xlabel("x (mm)")
        axes[row, 0].set_ylabel(f"tilt {tilt:+g}°\ny (mm)")

        coord_mm = np.asarray(profile["coord_m"], dtype=float) * 1e3
        axes[row, 1].plot(coord_mm, np.asarray(profile["lab_x"]) / nominal_peak, label="I(x, y=0)")
        axes[row, 1].plot(coord_mm, np.asarray(profile["lab_y"]) / nominal_peak, label="I(x=0, y)")
        axes[row, 1].set_xlabel("laboratory coordinate (mm)")
        axes[row, 1].set_ylabel("I / nominal 0° 2D peak")
        axes[row, 1].grid(alpha=0.20)

        axes[row, 2].plot(coord_mm, np.asarray(profile["centred_x"]) / nominal_peak, label="x through centroid")
        axes[row, 2].plot(coord_mm, np.asarray(profile["centred_y"]) / nominal_peak, label="y through centroid")
        axes[row, 2].set_xlabel("coordinate relative to beam centroid (mm)")
        axes[row, 2].set_ylabel("I / nominal 0° 2D peak")
        axes[row, 2].grid(alpha=0.20)
    axes[0, 0].set_title(f"2-D intensity at zref={Z_REF_MM:g} mm")
    axes[0, 1].set_title("exact fixed-lab profiles")
    axes[0, 2].set_title("exact beam-centred profiles")
    axes[0, 1].legend(fontsize=8)
    axes[0, 2].legend(fontsize=8)
    if last is not None:
        fig.colorbar(last, ax=axes[:, 0], shrink=0.80, label="I / common zref heatmap peak")
    fig.suptitle(
        f"{family}: canonical transverse/profile evidence\n"
        "2-D heatmaps share one 0–1 scale; profiles share nominal 0° 2-D peak; SYNTHETIC",
        fontsize=13,
    )
    fig.savefig(outdir / filenames["zref_profiles"], dpi=DEFAULT_FIGURE_DPI)
    plt.close(fig)
    return heatmap_peak, nominal_peak


def _write_csv(path: Path, profiles: dict[float, dict[str, np.ndarray | float]], nominal_peak: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "tilt_deg", "coordinate_frame", "profile_axis", "coordinate_m",
            "intensity_au", "intensity_over_nominal_0deg_2d_peak",
            "centroid_x_m", "centroid_y_m", "power_au_m2",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for tilt, p in profiles.items():
            coord = np.asarray(p["coord_m"], dtype=float)
            for frame, xvals, yvals in (
                ("laboratory", p["lab_x"], p["lab_y"]),
                ("centroid", p["centred_x"], p["centred_y"]),
            ):
                for axis, values in (("x", xvals), ("y", yvals)):
                    for coordinate_m, intensity in zip(coord, np.asarray(values, dtype=float)):
                        writer.writerow({
                            "tilt_deg": float(tilt),
                            "coordinate_frame": frame,
                            "profile_axis": axis,
                            "coordinate_m": float(coordinate_m),
                            "intensity_au": float(intensity),
                            "intensity_over_nominal_0deg_2d_peak": float(intensity / nominal_peak),
                            "centroid_x_m": float(p["centroid_x_m"]),
                            "centroid_y_m": float(p["centroid_y_m"]),
                            "power_au_m2": float(p["power"]),
                        })


def _render_direction(direction: str, outdir: Path) -> dict[str, object]:
    family = f"phase2h_six_sector_rot_{direction}"
    filenames = canonical_evidence_filenames(family, Z_REF_MM)
    spec = CanonicalEvidenceSpec(z_ref_mm=Z_REF_MM)
    spec.validate()
    bundle = _bundle()
    sources: dict[float, VectorField] = {}
    routes: dict[float, dict] = {}
    longitudinal: dict[float, dict[str, np.ndarray | float]] = {}
    zref_fields: dict[float, VectorField] = {}
    profiles: dict[float, dict[str, np.ndarray | float]] = {}
    for tilt in REPRESENTATIVE_TILTS_DEG:
        source, route = _build_source(bundle, direction, tilt)
        sources[tilt] = source
        routes[tilt] = route
        longitudinal[tilt] = _longitudinal(source)
        zref_fields[tilt] = _at_z(source, Z_REF_MM)
        profiles[tilt] = _profile_data(zref_fields[tilt])

    long_peak = _render_longitudinal(family, direction, longitudinal, outdir, filenames)
    heatmap_peak, nominal_peak = _render_zref(family, direction, zref_fields, profiles, outdir, filenames)
    _write_csv(outdir / filenames["metrics_csv"], profiles, nominal_peak)

    npz: dict[str, np.ndarray] = {
        "tilts_deg": np.asarray(REPRESENTATIVE_TILTS_DEG, dtype=float),
        "z_mm": np.asarray(Z_MM, dtype=float),
        "z_ref_mm": np.asarray([Z_REF_MM], dtype=float),
    }
    for tilt in REPRESENTATIVE_TILTS_DEG:
        tag = str(tilt).replace("-", "m").replace(".", "p")
        case = longitudinal[tilt]
        npz[f"tilt_{tag}_xz"] = np.asarray(case["xz"], dtype=float)
        npz[f"tilt_{tag}_yz"] = np.asarray(case["yz"], dtype=float)
        npz[f"tilt_{tag}_x_mm"] = np.asarray(case["x_mm"], dtype=float)
        npz[f"tilt_{tag}_y_mm"] = np.asarray(case["y_mm"], dtype=float)
        npz[f"tilt_{tag}_zref_intensity"] = np.asarray(zref_fields[tilt].intensity, dtype=float)
        npz[f"tilt_{tag}_profile_coordinate_m"] = np.asarray(profiles[tilt]["coord_m"], dtype=float)
        npz[f"tilt_{tag}_profile_lab_x"] = np.asarray(profiles[tilt]["lab_x"], dtype=float)
        npz[f"tilt_{tag}_profile_lab_y"] = np.asarray(profiles[tilt]["lab_y"], dtype=float)
        npz[f"tilt_{tag}_profile_centred_x"] = np.asarray(profiles[tilt]["centred_x"], dtype=float)
        npz[f"tilt_{tag}_profile_centred_y"] = np.asarray(profiles[tilt]["centred_y"], dtype=float)
    np.savez_compressed(outdir / filenames["raw_npz"], **npz)

    first = sources[0.0]
    manifest = {
        "outcome": "PHASE2H-CANONICAL-EVIDENCE-PAIR-SYNTHETIC",
        "family": family,
        "data_classification": "synthetic_not_experimental",
        "report_figures_authorised": False,
        "z_ref_mm": Z_REF_MM,
        "z_ref_policy": spec.z_ref_policy,
        "representative_tilts_deg": list(REPRESENTATIVE_TILTS_DEG),
        "longitudinal_z_mm": Z_MM.tolist(),
        "longitudinal_dz_mm": float(Z_MM[1] - Z_MM[0]),
        "longitudinal_frame": spec.longitudinal_frame,
        "primary_intensity_colormap": INTENSITY_CMAP,
        "primary_intensity_scale": spec.primary_intensity_scale,
        "longitudinal_common_peak_au": long_peak,
        "zref_heatmap_common_peak_au": heatmap_peak,
        "profile_nominal_0deg_2d_peak_au": nominal_peak,
        "profile_sampling": spec.profile_sampling,
        "primary_display_half_width_mm": DETAIL_HALF_WIDTH_MM,
        "field_n": int(first.ex.shape[0]),
        "field_dx_um": float(first.grid["dx"]) * 1e6,
        "field_window_mm": float(first.grid["dx"]) * int(first.ex.shape[0]) * 1e3,
        "png_dpi": DEFAULT_FIGURE_DPI,
        "rendered_image_interpolation": "nearest_display_only",
        "files": filenames,
    }
    (outdir / filenames["manifest"]).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = [_render_direction(direction, args.output_dir) for direction in ("x", "y")]
    (args.output_dir / "phase2h_canonical_evidence_index.json").write_text(
        json.dumps({"manifests": manifests}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "families": [m["family"] for m in manifests]}, indent=2))


if __name__ == "__main__":
    main()
