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
from vbb_study.digital_twin.vortex_profile_evidence import spectral_line_fields
from vbb_study.vector_field import VectorField, propagate_vector_asm

TWOPI = 2.0 * np.pi
DEFAULT_TILTS_DEG = (0.0, 5.0, 10.0)
DEFAULT_Z_MM = (0.0, 5.0, 10.0, 20.0, 30.0, 60.0)


def _lut(panel: str) -> SLMPhaseCalibration:
    grey = np.arange(256, dtype=float)
    return SLMPhaseCalibration(
        panel_id=panel,
        wavelength_m=1029e-9,
        grey_levels=grey,
        phase_rad=TWOPI * grey / 255.0,
        calibration_date="synthetic_phase2h_profile_preview",
    )


def _bundle() -> CalibrationBundle:
    data = canonical_calibration_template()
    data["calibration_id"] = "synthetic_phase2h_profile_preview"
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


def _inputs(bundle: CalibrationBundle, tilt_deg: float) -> BenchCalibratedVectorInputs:
    return BenchCalibratedVectorInputs(
        calibration_bundle=bundle,
        slm1_phase_calibration=_lut("synthetic_slm1"),
        slm2_phase_calibration=_lut("synthetic_slm2"),
        axicon_tilt_rad=(0.0, math.radians(float(tilt_deg))),
    )


def _build_sources(tilts_deg: tuple[float, ...]) -> dict[float, VectorField]:
    bundle = _bundle()
    config = NathanHexagonConfig.fast(grid_n=128)
    out: dict[float, VectorField] = {}
    for tilt in tilts_deg:
        route = build_calibrated_segmented_vector_tilt_route(
            config,
            calibrated=_inputs(bundle, tilt),
            vector_axicon_output_n=512,
            vector_axicon_output_window_m=7.2e-3,
            reference_gap_m=0.25e-3,
        )
        out[tilt] = route["post_axicon"]
    return out


def _at_z(source: VectorField, z_mm: float) -> VectorField:
    return source if abs(float(z_mm)) < 1e-15 else propagate_vector_asm(source, float(z_mm) * 1e-3)


def _centroid_m(field: VectorField) -> tuple[float, float]:
    intensity = np.asarray(field.intensity, dtype=float)
    x = np.asarray(field.grid["x"], dtype=float)
    y = np.asarray(field.grid["y"], dtype=float)
    X, Y = np.meshgrid(x, y, indexing="xy")
    total = float(np.sum(intensity))
    if total <= 0.0:
        return 0.0, 0.0
    return float(np.sum(X * intensity) / total), float(np.sum(Y * intensity) / total)


def _vector_line_intensity(
    field: VectorField,
    *,
    x_coordinates_m: np.ndarray,
    y_coordinates_m: np.ndarray,
    fixed_x_m: float,
    fixed_y_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_total = np.zeros_like(x_coordinates_m, dtype=float)
    y_total = np.zeros_like(y_coordinates_m, dtype=float)
    for component in (field.ex, field.ey, field.ez):
        x_line, y_line = spectral_line_fields(
            component,
            field.grid,
            x_coordinates_m=x_coordinates_m,
            y_coordinates_m=y_coordinates_m,
            fixed_x_m=float(fixed_x_m),
            fixed_y_m=float(fixed_y_m),
        )
        x_total += np.abs(x_line) ** 2
        y_total += np.abs(y_line) ** 2
    return x_total, y_total


def _plane_profiles(field: VectorField, *, span_m: float, samples: int) -> dict[str, np.ndarray | float]:
    lab = np.linspace(-float(span_m), float(span_m), int(samples))
    rel = np.linspace(-float(span_m), float(span_m), int(samples))
    cx, cy = _centroid_m(field)
    lab_x, lab_y = _vector_line_intensity(
        field,
        x_coordinates_m=lab,
        y_coordinates_m=lab,
        fixed_x_m=0.0,
        fixed_y_m=0.0,
    )
    centred_x, centred_y = _vector_line_intensity(
        field,
        x_coordinates_m=cx + rel,
        y_coordinates_m=cy + rel,
        fixed_x_m=cx,
        fixed_y_m=cy,
    )
    return {
        "lab_coordinate_m": lab,
        "relative_coordinate_m": rel,
        "lab_x": lab_x,
        "lab_y": lab_y,
        "centred_x": centred_x,
        "centred_y": centred_y,
        "centroid_x_m": cx,
        "centroid_y_m": cy,
        "peak_2d": float(np.max(field.intensity)),
        "power": float(field.power),
    }


def _extent_mm(field: VectorField) -> list[float]:
    x = np.asarray(field.grid["x"], dtype=float) * 1e3
    y = np.asarray(field.grid["y"], dtype=float) * 1e3
    return [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]


def _write_csv(path: Path, z_mm: float, fields: dict[float, VectorField], profiles: dict[float, dict], nominal_peak: float) -> None:
    rows: list[dict[str, float | str]] = []
    for tilt, profile in profiles.items():
        lab = np.asarray(profile["lab_coordinate_m"], dtype=float)
        rel = np.asarray(profile["relative_coordinate_m"], dtype=float)
        for frame, coord, xv, yv in (
            ("laboratory", lab, profile["lab_x"], profile["lab_y"]),
            ("centroid", rel, profile["centred_x"], profile["centred_y"]),
        ):
            for axis_name, values in (("x", xv), ("y", yv)):
                for coordinate_m, intensity in zip(coord, np.asarray(values, dtype=float)):
                    rows.append({
                        "z_mm": float(z_mm),
                        "tilt_deg": float(tilt),
                        "coordinate_frame": frame,
                        "profile_axis": axis_name,
                        "coordinate_m": float(coordinate_m),
                        "intensity_au": float(intensity),
                        "intensity_over_nominal_0deg_peak": float(intensity / nominal_peak),
                        "centroid_x_m": float(profile["centroid_x_m"]),
                        "centroid_y_m": float(profile["centroid_y_m"]),
                        "power_au_m2": float(profile["power"]),
                    })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _render_plane(outdir: Path, z_mm: float, fields: dict[float, VectorField], *, span_m: float, samples: int) -> dict:
    tilts = tuple(fields.keys())
    profiles = {tilt: _plane_profiles(field, span_m=span_m, samples=samples) for tilt, field in fields.items()}
    nominal_peak = max(float(profiles[tilts[0]]["peak_2d"]), np.finfo(float).tiny)
    common_image_peak = nominal_peak

    fig, axes = plt.subplots(len(tilts), 4, figsize=(16.5, 4.0 * len(tilts)), constrained_layout=True)
    if len(tilts) == 1:
        axes = np.asarray([axes])

    for row, tilt in enumerate(tilts):
        field = fields[tilt]
        p = profiles[tilt]
        image = np.asarray(field.intensity, dtype=float) / common_image_peak
        im = axes[row, 0].imshow(
            image,
            origin="lower",
            extent=_extent_mm(field),
            vmin=0.0,
            vmax=max(1.0, float(np.nanpercentile(image, 99.95))),
            aspect="equal",
        )
        cx_mm = float(p["centroid_x_m"]) * 1e3
        cy_mm = float(p["centroid_y_m"]) * 1e3
        axes[row, 0].axvline(0.0, linewidth=0.8, linestyle="--", alpha=0.7)
        axes[row, 0].axhline(0.0, linewidth=0.8, linestyle="--", alpha=0.7)
        axes[row, 0].plot([cx_mm], [cy_mm], marker="x", markersize=7)
        axes[row, 0].set_xlim(-span_m * 1e3, span_m * 1e3)
        axes[row, 0].set_ylim(-span_m * 1e3, span_m * 1e3)
        axes[row, 0].set_xlabel("x (mm)")
        axes[row, 0].set_ylabel(f"tilt = {tilt:g}°\ny (mm)")
        axes[row, 0].set_title(f"2D intensity at z={z_mm:g} mm\ncentroid=({cx_mm:.3f}, {cy_mm:.3f}) mm")

        lab_mm = np.asarray(p["lab_coordinate_m"]) * 1e3
        rel_mm = np.asarray(p["relative_coordinate_m"]) * 1e3
        axes[row, 1].plot(lab_mm, np.asarray(p["lab_x"]) / nominal_peak, label="x cut: y=0")
        axes[row, 1].plot(lab_mm, np.asarray(p["lab_y"]) / nominal_peak, label="y cut: x=0")
        axes[row, 1].set_title("Fixed lab-frame cuts")
        axes[row, 1].set_xlabel("lab coordinate (mm)")
        axes[row, 1].set_ylabel("I / nominal 0° 2D peak")
        axes[row, 1].legend(fontsize=8)
        axes[row, 1].grid(alpha=0.2)

        axes[row, 2].plot(rel_mm, np.asarray(p["centred_x"]) / nominal_peak, label="x cut through centroid")
        axes[row, 2].plot(rel_mm, np.asarray(p["centred_y"]) / nominal_peak, label="y cut through centroid")
        axes[row, 2].set_title("Beam-centred morphology cuts")
        axes[row, 2].set_xlabel("coordinate relative to centroid (mm)")
        axes[row, 2].set_ylabel("I / nominal 0° 2D peak")
        axes[row, 2].legend(fontsize=8)
        axes[row, 2].grid(alpha=0.2)

        axes[row, 3].plot(lab_mm, np.asarray(p["lab_x"]) / nominal_peak, label="lab x")
        axes[row, 3].plot(rel_mm + cx_mm, np.asarray(p["centred_x"]) / nominal_peak, linestyle="--", label="centred x placed in lab")
        axes[row, 3].axvline(cx_mm, linewidth=0.9, linestyle=":", label="centroid x")
        axes[row, 3].set_title("Steering vs morphology — x direction")
        axes[row, 3].set_xlabel("x (mm)")
        axes[row, 3].set_ylabel("I / nominal 0° 2D peak")
        axes[row, 3].legend(fontsize=8)
        axes[row, 3].grid(alpha=0.2)

    fig.colorbar(im, ax=axes[:, 0], shrink=0.78, label="I / nominal 0° 2D peak")
    fig.suptitle(
        f"Phase 2H true-line profile evidence at z={z_mm:g} mm\n"
        "direct complex Fourier-series line evaluation; common 0° scale; SYNTHETIC / NOT LAB CALIBRATED",
        fontsize=14,
    )
    tag = str(z_mm).replace(".", "p")
    png = outdir / f"phase2h_profiles_z{tag}mm.png"
    csv_path = outdir / f"phase2h_profiles_z{tag}mm.csv"
    fig.savefig(png, dpi=190)
    plt.close(fig)
    _write_csv(csv_path, z_mm, fields, profiles, nominal_peak)
    return {
        "z_mm": float(z_mm),
        "png": png.name,
        "csv": csv_path.name,
        "nominal_0deg_peak": nominal_peak,
        "cases": {
            str(tilt): {
                "centroid_x_mm": float(profiles[tilt]["centroid_x_m"]) * 1e3,
                "centroid_y_mm": float(profiles[tilt]["centroid_y_m"]) * 1e3,
                "peak_2d_over_nominal": float(profiles[tilt]["peak_2d"]) / nominal_peak,
                "power_au_m2": float(profiles[tilt]["power"]),
            }
            for tilt in tilts
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Render true arbitrary-z Phase 2H vector profile evidence.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/validation/phase2h/profile_planes"))
    parser.add_argument("--z-mm", type=float, nargs="+", default=list(DEFAULT_Z_MM))
    parser.add_argument("--tilts-deg", type=float, nargs="+", default=list(DEFAULT_TILTS_DEG))
    parser.add_argument("--span-mm", type=float, default=1.5)
    parser.add_argument("--samples", type=int, default=1201)
    args = parser.parse_args()

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    tilts = tuple(float(v) for v in args.tilts_deg)
    z_values = tuple(float(v) for v in args.z_mm)
    sources = _build_sources(tilts)
    manifest = {
        "outcome": "PHASE2H-ARBITRARY-Z-PROFILE-EVIDENCE",
        "data_classification": "synthetic_not_experimental",
        "report_figures_authorised": False,
        "profile_sampling": "direct_discrete_Fourier_series_on_Ex_Ey_Ez_then_sum_component_intensities",
        "intensity_image_interpolation": False,
        "normalisation": "all curves at each z divided by 0-degree 2D peak at same z",
        "lab_frame": "fixed x=0/y=0 lines preserve steering",
        "centred_frame": "lines through intensity centroid separate morphology from steering",
        "tilts_deg": list(tilts),
        "z_mm": list(z_values),
        "planes": [],
    }
    for z_mm in z_values:
        fields = {tilt: _at_z(source, z_mm) for tilt, source in sources.items()}
        manifest["planes"].append(
            _render_plane(outdir, z_mm, fields, span_m=float(args.span_mm) * 1e-3, samples=int(args.samples))
        )
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
