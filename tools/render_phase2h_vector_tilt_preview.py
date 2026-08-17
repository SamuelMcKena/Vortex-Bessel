from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vbb_study.calibration.schema import CalibrationBundle, canonical_calibration_template, measurement
from vbb_study.calibration.slm_phase import SLMPhaseCalibration
from vbb_study.digital_twin.bench_calibrated_vector_route import (
    BenchCalibratedVectorInputs,
    calibrated_vector_route_to_sample,
)
from vbb_study.digital_twin.bench_calibrated_vector_tilt_route import (
    build_calibrated_segmented_vector_tilt_route,
)
from vbb_study.digital_twin.nathan_vector_hexagon import NathanHexagonConfig
from vbb_study.digital_twin.objective_pupil_mapping import ObjectivePupilMappingConfig
from vbb_study.digital_twin.objective_sample_route import ObjectiveSampleConfig
from vbb_study.vector_field import VectorField, propagate_vector_asm


TWOPI = 2.0 * np.pi
TILT_DEG = (0.0, 5.0, 10.0)


def _lut(panel: str) -> SLMPhaseCalibration:
    grey = np.arange(256, dtype=float)
    return SLMPhaseCalibration(
        panel_id=panel,
        wavelength_m=1029e-9,
        grey_levels=grey,
        phase_rad=TWOPI * grey / 255.0,
        calibration_date="synthetic_preview",
    )


def _bundle() -> CalibrationBundle:
    data = canonical_calibration_template()
    data["calibration_id"] = "synthetic_phase2h_preview"
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


def _axes_mm(field: VectorField) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(field.grid["x"], dtype=float) * 1e3
    y = np.asarray(field.grid["y"], dtype=float) * 1e3
    return x, y


def _extent_mm(field: VectorField) -> list[float]:
    x, y = _axes_mm(field)
    return [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]


def _centroid_mm(field: VectorField) -> tuple[float, float]:
    x, y = _axes_mm(field)
    X, Y = np.meshgrid(x, y, indexing="xy")
    intensity = np.asarray(field.intensity, dtype=float)
    total = float(np.sum(intensity))
    if total <= 0.0:
        return float("nan"), float("nan")
    return float(np.sum(X * intensity) / total), float(np.sum(Y * intensity) / total)


def _normalized_stokes(field: VectorField, key: str) -> np.ndarray:
    stokes = field.stokes()
    s0 = np.asarray(stokes["S0"], dtype=float)
    values = np.asarray(stokes[key], dtype=float)
    out = np.divide(values, s0, out=np.zeros_like(values), where=s0 > 1e-20 * max(float(np.max(s0)), 1.0))
    out = np.where(s0 >= 0.015 * max(float(np.max(s0)), 1e-30), out, np.nan)
    return out


def _line_profiles(field: VectorField) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y = _axes_mm(field)
    ix0 = int(np.argmin(np.abs(x)))
    iy0 = int(np.argmin(np.abs(y)))
    intensity = np.asarray(field.intensity, dtype=float)
    return x, intensity[iy0, :], y, intensity[:, ix0]


def _build_routes() -> dict[float, dict]:
    bundle = _bundle()
    config = NathanHexagonConfig.fast(grid_n=128)
    routes: dict[float, dict] = {}
    for tilt in TILT_DEG:
        routes[tilt] = build_calibrated_segmented_vector_tilt_route(
            config,
            calibrated=_inputs(bundle, tilt),
            vector_axicon_output_n=512,
            vector_axicon_output_window_m=7.2e-3,
            reference_gap_m=0.25e-3,
        )
    return routes


def _sample_routes(routes: dict[float, dict]) -> dict[float, dict]:
    out: dict[float, dict] = {}
    mapping = ObjectivePupilMappingConfig(
        free_space_distance_m=0.0,
        output_window_m=4.0e-3,
        output_n=96,
        pupil_radius_m=1.6e-3,
    )
    objective = ObjectiveSampleConfig(
        wavelength_m=1029e-9,
        numerical_aperture=0.45,
        objective_focal_length_m=4.0e-3,
        objective_pupil_radius_m=1.6e-3,
        sample_refractive_index=1.45,
        sample_depth_m=2.0e-6,
        fft_pad_factor=1,
    )
    for tilt, route in routes.items():
        out[tilt] = calibrated_vector_route_to_sample(
            route,
            mapping_config=mapping,
            objective_config=objective,
        )
    return out


def _propagation(routes: dict[float, dict]) -> tuple[np.ndarray, dict[float, dict]]:
    z_mm = np.linspace(0.0, 30.0, 31)
    data: dict[float, dict] = {}
    for tilt, route in routes.items():
        source: VectorField = route["post_axicon"]
        x, y = _axes_mm(source)
        ix0 = int(np.argmin(np.abs(x)))
        iy0 = int(np.argmin(np.abs(y)))
        xz: list[np.ndarray] = []
        yz: list[np.ndarray] = []
        cx: list[float] = []
        cy: list[float] = []
        peak: list[float] = []
        for z in z_mm:
            field = source if z == 0.0 else propagate_vector_asm(source, float(z) * 1e-3)
            intensity = np.asarray(field.intensity, dtype=float)
            xz.append(intensity[iy0, :].copy())
            yz.append(intensity[:, ix0].copy())
            cxi, cyi = _centroid_mm(field)
            cx.append(cxi)
            cy.append(cyi)
            peak.append(float(np.max(intensity)))
        data[tilt] = {
            "x_mm": x,
            "y_mm": y,
            "xz": np.asarray(xz),
            "yz": np.asarray(yz),
            "centroid_x_mm": np.asarray(cx),
            "centroid_y_mm": np.asarray(cy),
            "peak": np.asarray(peak),
        }
    return z_mm, data


def _figure_transverse(routes: dict[float, dict], outdir: Path) -> None:
    post4f_peak = max(float(np.max(route["post_4f_selected_order"].intensity)) for route in routes.values())
    post_peak = max(float(np.max(route["post_axicon"].intensity)) for route in routes.values())
    fig, axes = plt.subplots(len(TILT_DEG), 4, figsize=(15.2, 10.8), constrained_layout=True)
    for row, tilt in enumerate(TILT_DEG):
        route = routes[tilt]
        pre: VectorField = route["post_4f_selected_order"]
        post: VectorField = route["post_axicon"]
        im0 = axes[row, 0].imshow(pre.intensity / post4f_peak, origin="lower", extent=_extent_mm(pre), vmin=0.0, vmax=1.0, aspect="equal")
        im1 = axes[row, 1].imshow(post.intensity / post_peak, origin="lower", extent=_extent_mm(post), vmin=0.0, vmax=1.0, aspect="equal")
        im2 = axes[row, 2].imshow(_normalized_stokes(post, "S3"), origin="lower", extent=_extent_mm(post), vmin=-1.0, vmax=1.0, aspect="equal", cmap="coolwarm")
        frac = np.divide(np.abs(post.ez) ** 2, post.intensity, out=np.zeros_like(post.intensity, dtype=float), where=post.intensity > 1e-30)
        upper = max(float(np.nanpercentile(frac[post.intensity > 0.01 * np.max(post.intensity)], 99.0)), 1e-6)
        im3 = axes[row, 3].imshow(frac, origin="lower", extent=_extent_mm(post), vmin=0.0, vmax=upper, aspect="equal")
        for ax in axes[row, :]:
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
        axes[row, 0].set_ylabel(f"tilt = {tilt:.0f}°\ny (mm)")
        if row == 0:
            axes[row, 0].set_title("Post-4F selected order\ncommon intensity scale")
            axes[row, 1].set_title("Post-axicon\ncommon intensity scale")
            axes[row, 2].set_title("Post-axicon normalized S3")
            axes[row, 3].set_title("Post-axicon |Ez|² / I")
    fig.colorbar(im0, ax=axes[:, 0], shrink=0.75, label="I / global post-4F peak")
    fig.colorbar(im1, ax=axes[:, 1], shrink=0.75, label="I / global post-axicon peak")
    fig.colorbar(im2, ax=axes[:, 2], shrink=0.75, label="S3 / S0")
    fig.colorbar(im3, ax=axes[:, 3], shrink=0.75, label="longitudinal fraction")
    fig.suptitle("Phase 2H preview — synthetic six-sector vector beam through physical two-surface tilted axicon\nNOT lab calibrated / NOT report-authorised", fontsize=14)
    fig.savefig(outdir / "01_phase2h_transverse_tilt_sweep.png", dpi=190)
    plt.close(fig)


def _figure_longitudinal(z_mm: np.ndarray, prop: dict[float, dict], outdir: Path) -> None:
    global_peak = max(float(np.max(case["xz"])) for case in prop.values())
    global_peak = max(global_peak, max(float(np.max(case["yz"])) for case in prop.values()))
    fig, axes = plt.subplots(len(TILT_DEG), 4, figsize=(16.0, 10.2), constrained_layout=True)
    for row, tilt in enumerate(TILT_DEG):
        case = prop[tilt]
        x = case["x_mm"]
        y = case["y_mm"]
        xz = case["xz"] / global_peak
        yz = case["yz"] / global_peak
        extent_x = [float(x[0]), float(x[-1]), float(z_mm[0]), float(z_mm[-1])]
        extent_y = [float(y[0]), float(y[-1]), float(z_mm[0]), float(z_mm[-1])]
        im0 = axes[row, 0].imshow(xz, origin="lower", extent=extent_x, vmin=0.0, vmax=1.0, aspect="auto")
        im1 = axes[row, 1].imshow(yz, origin="lower", extent=extent_y, vmin=0.0, vmax=1.0, aspect="auto")
        log_xz = np.log10(np.maximum(xz, 1e-6))
        log_yz = np.log10(np.maximum(yz, 1e-6))
        im2 = axes[row, 2].imshow(log_xz, origin="lower", extent=extent_x, vmin=-6.0, vmax=0.0, aspect="auto")
        im3 = axes[row, 3].imshow(log_yz, origin="lower", extent=extent_y, vmin=-6.0, vmax=0.0, aspect="auto")
        axes[row, 0].set_ylabel(f"tilt = {tilt:.0f}°\nz (mm)")
        for col in range(4):
            axes[row, col].set_ylabel(f"tilt = {tilt:.0f}°\nz (mm)")
        axes[row, 0].set_xlabel("x (mm)")
        axes[row, 1].set_xlabel("y (mm)")
        axes[row, 2].set_xlabel("x (mm)")
        axes[row, 3].set_xlabel("y (mm)")
        if row == 0:
            axes[row, 0].set_title("fixed lab x-z, linear")
            axes[row, 1].set_title("fixed lab y-z, linear")
            axes[row, 2].set_title("fixed lab x-z, log10")
            axes[row, 3].set_title("fixed lab y-z, log10")
    fig.colorbar(im0, ax=axes[:, :2], shrink=0.75, label="I / global peak")
    fig.colorbar(im2, ax=axes[:, 2:], shrink=0.75, label="log10(I / global peak)")
    fig.suptitle("Phase 2H preview — fixed physical laboratory planes (no z-dependent recentering)\nNOT lab calibrated / NOT report-authorised", fontsize=14)
    fig.savefig(outdir / "02_phase2h_fixed_lab_longitudinal.png", dpi=190)
    plt.close(fig)


def _figure_profiles_centroids(routes: dict[float, dict], z_mm: np.ndarray, prop: dict[float, dict], outdir: Path) -> None:
    nominal_peak = float(np.max(routes[0.0]["post_axicon"].intensity))
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    for tilt in TILT_DEG:
        post: VectorField = routes[tilt]["post_axicon"]
        x, px, y, py = _line_profiles(post)
        axes[0, 0].plot(x, px / nominal_peak, label=f"{tilt:.0f}°")
        axes[0, 1].plot(y, py / nominal_peak, label=f"{tilt:.0f}°")
        axes[1, 0].plot(z_mm, prop[tilt]["centroid_x_mm"], label=f"{tilt:.0f}°")
        axes[1, 1].plot(z_mm, prop[tilt]["centroid_y_mm"], label=f"{tilt:.0f}°")
    axes[0, 0].set_title("Fixed lab I(x, y≈0) immediately after axicon")
    axes[0, 1].set_title("Fixed lab I(y, x≈0) immediately after axicon")
    axes[1, 0].set_title("Intensity centroid x(z)")
    axes[1, 1].set_title("Intensity centroid y(z)")
    axes[0, 0].set_xlabel("x (mm)")
    axes[0, 1].set_xlabel("y (mm)")
    axes[0, 0].set_ylabel("I / nominal 0° post-axicon peak")
    axes[0, 1].set_ylabel("I / nominal 0° post-axicon peak")
    axes[1, 0].set_xlabel("z (mm)")
    axes[1, 1].set_xlabel("z (mm)")
    axes[1, 0].set_ylabel("centroid x (mm)")
    axes[1, 1].set_ylabel("centroid y (mm)")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.legend(title="axicon tilt")
    fig.suptitle("Phase 2H preview — common-scale line profiles and physical steering\nNOT lab calibrated / NOT report-authorised", fontsize=14)
    fig.savefig(outdir / "03_phase2h_profiles_and_centroids.png", dpi=190)
    plt.close(fig)


def _figure_sample(sample_routes: dict[float, dict], outdir: Path) -> None:
    fields = {tilt: result["sample_result"].field_in_sample for tilt, result in sample_routes.items()}
    peak = max(float(np.max(field.intensity)) for field in fields.values())
    fig, axes = plt.subplots(len(TILT_DEG), 4, figsize=(15.2, 10.8), constrained_layout=True)
    for row, tilt in enumerate(TILT_DEG):
        field: VectorField = fields[tilt]
        im0 = axes[row, 0].imshow(field.intensity / peak, origin="lower", extent=_extent_mm(field), vmin=0.0, vmax=1.0, aspect="equal")
        im1 = axes[row, 1].imshow(_normalized_stokes(field, "S1"), origin="lower", extent=_extent_mm(field), vmin=-1.0, vmax=1.0, aspect="equal", cmap="coolwarm")
        im2 = axes[row, 2].imshow(_normalized_stokes(field, "S2"), origin="lower", extent=_extent_mm(field), vmin=-1.0, vmax=1.0, aspect="equal", cmap="coolwarm")
        im3 = axes[row, 3].imshow(_normalized_stokes(field, "S3"), origin="lower", extent=_extent_mm(field), vmin=-1.0, vmax=1.0, aspect="equal", cmap="coolwarm")
        for ax in axes[row, :]:
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
        axes[row, 0].set_ylabel(f"tilt = {tilt:.0f}°\ny (mm)")
        if row == 0:
            axes[row, 0].set_title("Sample-plane intensity\ncommon scale")
            axes[row, 1].set_title("sample s1")
            axes[row, 2].set_title("sample s2")
            axes[row, 3].set_title("sample s3")
    fig.colorbar(im0, ax=axes[:, 0], shrink=0.75, label="I / global sample peak")
    fig.colorbar(im3, ax=axes[:, 1:], shrink=0.75, label="normalized Stokes")
    fig.suptitle("Phase 2H preview — tilted segmented-vector route at the objective/sample plane\nNOT lab calibrated / NOT report-authorised", fontsize=14)
    fig.savefig(outdir / "04_phase2h_sample_plane_tilt_sweep.png", dpi=190)
    plt.close(fig)


def _figure_ray_geometry(routes: dict[float, dict], outdir: Path) -> None:
    fig, axes = plt.subplots(2, len(TILT_DEG), figsize=(14.2, 8.5), constrained_layout=True)
    for col, tilt in enumerate(TILT_DEG):
        result = routes[tilt]["vector_refractive_axicon_result"]
        valid = np.asarray(result.geometry_bundle.valid, dtype=bool)
        outgoing = np.asarray(result.outgoing_direction_lab, dtype=float)[valid]
        exit_xy = np.asarray(result.geometry_bundle.exit_point_lab_m, dtype=float)[valid, :2] * 1e3
        step = max(1, outgoing.shape[0] // 12000)
        axes[0, col].scatter(outgoing[::step, 0], outgoing[::step, 1], s=2, alpha=0.35)
        axes[0, col].set_aspect("equal", adjustable="box")
        axes[0, col].set_title(f"tilt = {tilt:.0f}°\noutgoing transverse direction")
        axes[0, col].set_xlabel("s_x")
        axes[0, col].set_ylabel("s_y")
        axes[1, col].scatter(exit_xy[::step, 0], exit_xy[::step, 1], s=2, alpha=0.35)
        axes[1, col].set_aspect("equal", adjustable="box")
        axes[1, col].set_title("cone-surface exit footprint")
        axes[1, col].set_xlabel("x (mm)")
        axes[1, col].set_ylabel("y (mm)")
    fig.suptitle("Phase 2H preview — physical two-surface ray geometry behind the vector boundary field\nNOT lab calibrated / NOT report-authorised", fontsize=14)
    fig.savefig(outdir / "05_phase2h_ray_geometry.png", dpi=190)
    plt.close(fig)


def _write_metrics(routes: dict[float, dict], sample_routes: dict[float, dict], z_mm: np.ndarray, prop: dict[float, dict], outdir: Path) -> None:
    cases = []
    for tilt in TILT_DEG:
        result = routes[tilt]["vector_refractive_axicon_result"]
        post: VectorField = routes[tilt]["post_axicon"]
        sample: VectorField = sample_routes[tilt]["sample_result"].field_in_sample
        cx0, cy0 = _centroid_mm(post)
        cases.append(
            {
                "tilt_deg": tilt,
                "post_axicon_power": post.power,
                "post_axicon_centroid_mm": [cx0, cy0],
                "sample_power": sample.power,
                "final_flux_closure_ratio": float(result.metadata["final_flux_closure_ratio"]),
                "final_transversality_residual": float(result.metadata["final_transversality_residual"]),
                "required_nyquist_fraction": float(result.metadata["required_nyquist_fraction"]),
                "centroid_x_mm_over_z": prop[tilt]["centroid_x_mm"].tolist(),
                "centroid_y_mm_over_z": prop[tilt]["centroid_y_mm"].tolist(),
            }
        )
    payload = {
        "outcome": "PHASE2H-VECTOR-TILT-PREVIEW",
        "report_figures_authorised": False,
        "data_classification": "synthetic_not_experimental",
        "warning": "Preview/sensitivity figures only. Do not use as calibrated prediction of the laboratory axicon.",
        "z_mm": z_mm.tolist(),
        "cases": cases,
    }
    (outdir / "phase2h_preview_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (outdir / "README.txt").write_text(
        "Phase 2H preview figure pack\n\n"
        "These figures use the synthetic calibration fixture already used by the Phase 2H regression tests.\n"
        "They are for visual inspection of the validated model path only. They are NOT lab calibrated and are NOT report-authorised.\n\n"
        "01: transverse post-4F / post-axicon intensity, S3 and longitudinal-field fraction.\n"
        "02: fixed laboratory x-z/y-z propagation, linear and log scale, with no row recentering.\n"
        "03: common-scale line profiles plus physical centroid steering.\n"
        "04: objective/sample intensity and normalized Stokes maps.\n"
        "05: outgoing ray-direction and conical-exit geometry diagnostics.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render synthetic Phase 2H vector refractive axicon tilt preview figures.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/validation/phase2h/preview_figures"))
    args = parser.parse_args()
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    routes = _build_routes()
    sample_routes = _sample_routes(routes)
    z_mm, prop = _propagation(routes)

    _figure_transverse(routes, outdir)
    _figure_longitudinal(z_mm, prop, outdir)
    _figure_profiles_centroids(routes, z_mm, prop, outdir)
    _figure_sample(sample_routes, outdir)
    _figure_ray_geometry(routes, outdir)
    _write_metrics(routes, sample_routes, z_mm, prop, outdir)

    print(json.dumps({"output_dir": str(outdir), "files": sorted(path.name for path in outdir.iterdir())}, indent=2))


if __name__ == "__main__":
    main()
