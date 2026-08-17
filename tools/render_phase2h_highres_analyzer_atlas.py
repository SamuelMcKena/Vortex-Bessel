from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vbb_study.digital_twin.vector_refractive_axicon_eikonal import build_tilted_vector_refractive_axicon_field
from vbb_study.digital_twin.vector_tilt_study import (
    beam_moment_metrics,
    centered_coordinate_maps,
    higher_order_cylindrical_vector_input,
    ideal_linear_analyzer_frames,
    well_sampled_petal_observable,
)
from vbb_study.digital_twin.vortex_refractive_axicon import RefractiveAxiconGeometry
from vbb_study.reporting.evidence_conventions import HIGH_RES_FIGURE_DPI, INTENSITY_CMAP, common_positive_peak
from vbb_study.vector_field import VectorField, propagate_vector_asm


MODES = ("radial", "azimuthal")
ELLS = (1, 3)
ANALYZERS_DEG = (0, 45, 90, 135)
DISPLAY_TILTS_DEG = (-2.0, 0.0, 2.0)
Z_REF_MM = 30.0
SOURCE_N = 192
SOURCE_WINDOW_M = 3.0e-3
OUTPUT_N = 768
OUTPUT_WINDOW_M = 4.5e-3
DISPLAY_HALF_WIDTH_MM = 0.42
# Preserve the validated *physical* annulus policy when changing sampling.
# The earlier 12-pixel gate on a 512 / 7.2 mm grid was 168.75 um.
MINIMUM_ANALYSIS_RADIUS_M = 170.0e-6
GEOMETRY = RefractiveAxiconGeometry(
    base_angle_rad=math.radians(2.0),
    clear_radius_m=3.0e-3,
    centre_thickness_m=3.0e-3,
    refractive_index=1.458,
    external_index=1.0,
)


def _build(mode: str, ell: int, tilt_deg: float) -> tuple[VectorField, dict]:
    source = higher_order_cylindrical_vector_input(
        ell=ell,
        mode=mode,
        n=SOURCE_N,
        window_m=SOURCE_WINDOW_M,
        waist_m=0.90e-3,
    )
    result = build_tilted_vector_refractive_axicon_field(
        source,
        geometry=GEOMETRY,
        tilt_x_rad=math.radians(float(tilt_deg)),
        tilt_y_rad=0.0,
        reference_gap_m=0.25e-3,
        output_n=OUTPUT_N,
        output_window_m=OUTPUT_WINDOW_M,
    )
    return propagate_vector_asm(result.field, Z_REF_MM * 1e-3), dict(result.metadata)


def _render_state(mode: str, ell: int, outdir: Path) -> dict[str, object]:
    cases: dict[float, tuple[VectorField, dict[int, np.ndarray], dict]] = {}
    rows: list[dict[str, object]] = []
    expected = 2 * abs(int(ell))
    for tilt in DISPLAY_TILTS_DEG:
        field, metadata = _build(mode, ell, tilt)
        frames = ideal_linear_analyzer_frames(field, angles_deg=ANALYZERS_DEG)
        Xc, Yc, moments = centered_coordinate_maps(field)
        pixel_pitch_m = float(field.grid["dx"])
        minimum_radius_pixels = MINIMUM_ANALYSIS_RADIUS_M / pixel_pitch_m
        for analyzer in ANALYZERS_DEG:
            petals = well_sampled_petal_observable(
                frames[analyzer],
                Xc,
                Yc,
                pixel_pitch_m=pixel_pitch_m,
                minimum_radius_pixels=minimum_radius_pixels,
            )
            if int(petals.petal_count) != expected:
                raise RuntimeError(
                    f"high-resolution {mode} ell={ell} tilt={tilt:g} analyzer={analyzer:g} "
                    f"resolved {petals.petal_count} petals, expected {expected}; "
                    f"physical annulus floor={MINIMUM_ANALYSIS_RADIUS_M * 1e6:.1f} um"
                )
            rows.append(
                {
                    "mode": mode,
                    "ell": int(ell),
                    "tilt_deg": float(tilt),
                    "analyzer_deg": int(analyzer),
                    "expected_petals": expected,
                    "measured_petals": int(petals.petal_count),
                    "ring_radius_um": float(petals.ring_radius_m) * 1e6,
                    "ring_sample_count": int(petals.ring_sample_count),
                    "minimum_analysis_radius_um": MINIMUM_ANALYSIS_RADIUS_M * 1e6,
                    "minimum_analysis_radius_pixels": float(minimum_radius_pixels),
                    "modulation_cv": float(petals.modulation_fraction),
                    "centroid_x_mm": moments.centroid_x_m * 1e3,
                    "centroid_y_mm": moments.centroid_y_m * 1e3,
                    "required_nyquist_fraction": float(metadata["required_nyquist_fraction"]),
                    "final_flux_closure_ratio": float(metadata["final_flux_closure_ratio"]),
                    "final_transversality_residual": float(metadata["final_transversality_residual"]),
                }
            )
        cases[tilt] = (field, frames, metadata)

    common_peak = common_positive_peak(
        [frame for _field, frames, _meta in cases.values() for frame in frames.values()]
    )
    fig, axes = plt.subplots(len(DISPLAY_TILTS_DEG), len(ANALYZERS_DEG), figsize=(12.6, 9.6), constrained_layout=True)
    last = None
    for row, tilt in enumerate(DISPLAY_TILTS_DEG):
        field, frames, _meta = cases[tilt]
        moments = beam_moment_metrics(field)
        cx_mm = moments.centroid_x_m * 1e3
        cy_mm = moments.centroid_y_m * 1e3
        x = np.asarray(field.grid["x"], dtype=float) * 1e3
        y = np.asarray(field.grid.get("y", field.grid["x"]), dtype=float) * 1e3
        extent = [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]
        for col, analyzer in enumerate(ANALYZERS_DEG):
            last = axes[row, col].imshow(
                np.asarray(frames[analyzer], dtype=float) / common_peak,
                origin="lower",
                extent=extent,
                cmap=INTENSITY_CMAP,
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
                aspect="equal",
            )
            axes[row, col].set_xlim(cx_mm - DISPLAY_HALF_WIDTH_MM, cx_mm + DISPLAY_HALF_WIDTH_MM)
            axes[row, col].set_ylim(cy_mm - DISPLAY_HALF_WIDTH_MM, cy_mm + DISPLAY_HALF_WIDTH_MM)
            axes[row, col].set_xlabel("x (mm)")
            axes[row, col].set_ylabel("y (mm)")
            if row == 0:
                axes[row, col].set_title(f"analyzer {analyzer}°")
        axes[row, 0].set_ylabel(f"tilt {tilt:+g}°\ny (mm)")
    if last is not None:
        fig.colorbar(last, ax=axes, shrink=0.82, label="I / common state-atlas peak")
    dx_um = float(cases[0.0][0].grid["dx"]) * 1e6
    fig.suptitle(
        f"Phase 2H high-resolution {mode} analyzer spots, ell={ell} ({expected} petals)\n"
        f"recomputed N={OUTPUT_N}, dx={dx_um:.2f} µm, zref={Z_REF_MM:g} mm; centroid-following display crop; SYNTHETIC",
        fontsize=13,
    )
    filename = f"highres_analyzer_atlas_{mode}_ell{ell}.png"
    fig.savefig(outdir / filename, dpi=HIGH_RES_FIGURE_DPI)
    plt.close(fig)
    return {
        "mode": mode,
        "ell": int(ell),
        "expected_petals": expected,
        "filename": filename,
        "common_heatmap_peak_au": common_peak,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    states = [_render_state(mode, ell, args.output_dir) for mode in MODES for ell in ELLS]
    dx_um = OUTPUT_WINDOW_M / OUTPUT_N * 1e6
    manifest = {
        "outcome": "PHASE2H-HIGH-RESOLUTION-ANALYZER-ATLAS-SYNTHETIC",
        "data_classification": "synthetic_not_experimental",
        "report_figures_authorised": False,
        "purpose": "supplementary morphology display; full validated 224-frame systematic metrics remain separate",
        "optical_resolution_policy": "field_recomputed_at_higher_N_not_image_upscaled",
        "observable_resolution_policy": "minimum analyzer annulus radius held fixed in physical units when N/window changes",
        "minimum_analysis_radius_um": MINIMUM_ANALYSIS_RADIUS_M * 1e6,
        "rendered_image_interpolation": "nearest_display_only",
        "intensity_colormap": INTENSITY_CMAP,
        "heatmap_normalisation": "one_common_peak_per_mode_ell_atlas",
        "source_n": SOURCE_N,
        "source_window_mm": SOURCE_WINDOW_M * 1e3,
        "output_n": OUTPUT_N,
        "output_window_mm": OUTPUT_WINDOW_M * 1e3,
        "output_dx_um": dx_um,
        "z_ref_mm": Z_REF_MM,
        "display_tilts_deg": list(DISPLAY_TILTS_DEG),
        "analyzer_angles_deg": list(ANALYZERS_DEG),
        "display_half_width_mm": DISPLAY_HALF_WIDTH_MM,
        "png_dpi": HIGH_RES_FIGURE_DPI,
        "states": states,
    }
    (args.output_dir / "highres_analyzer_atlas_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "output_n": OUTPUT_N, "output_dx_um": dx_um}, indent=2))


if __name__ == "__main__":
    main()
