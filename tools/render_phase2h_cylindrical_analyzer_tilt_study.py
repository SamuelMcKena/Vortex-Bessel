from __future__ import annotations

import argparse
import csv
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
from vbb_study.vector_field import VectorField, propagate_vector_asm


TILTS_DEG = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
ANALYZERS_DEG = (0, 45, 90, 135)
MODES = ("radial", "azimuthal")
ELLS = (1, 3)
Z_REF_MM = 30.0
DISPLAY_HALF_WIDTH_MM = 0.55
GEOMETRY = RefractiveAxiconGeometry(
    base_angle_rad=math.radians(2.0),
    clear_radius_m=3.0e-3,
    centre_thickness_m=3.0e-3,
    refractive_index=1.458,
    external_index=1.0,
)


def _tilt_pair(direction: str, tilt_deg: float) -> tuple[float, float]:
    angle = math.radians(float(tilt_deg))
    if direction == "x":
        return angle, 0.0
    if direction == "y":
        return 0.0, angle
    raise ValueError("direction must be x or y")


def _build_output(mode: str, ell: int, direction: str, tilt_deg: float) -> tuple[VectorField, dict]:
    source = higher_order_cylindrical_vector_input(
        ell=ell,
        mode=mode,
        n=128,
        window_m=3.0e-3,
        waist_m=0.90e-3,
    )
    tx, ty = _tilt_pair(direction, tilt_deg)
    result = build_tilted_vector_refractive_axicon_field(
        source,
        geometry=GEOMETRY,
        tilt_x_rad=tx,
        tilt_y_rad=ty,
        reference_gap_m=0.25e-3,
        output_n=512,
        output_window_m=7.2e-3,
    )
    field = propagate_vector_asm(result.field, Z_REF_MM * 1e-3)
    return field, dict(result.metadata)


def _frame_metrics(field: VectorField, mode: str, ell: int, direction: str, tilt_deg: float) -> tuple[list[dict], dict[int, np.ndarray]]:
    frames = ideal_linear_analyzer_frames(field, angles_deg=ANALYZERS_DEG)
    Xc, Yc, moments = centered_coordinate_maps(field)
    q = float(field.grid["dx"])
    rows: list[dict] = []
    for angle in ANALYZERS_DEG:
        petals = well_sampled_petal_observable(
            frames[angle],
            Xc,
            Yc,
            pixel_pitch_m=q,
            minimum_radius_pixels=12.0,
        )
        expected = 2 * abs(int(ell))
        rows.append(
            {
                "mode": mode,
                "ell": int(ell),
                "direction": direction,
                "tilt_deg": float(tilt_deg),
                "z_ref_mm": Z_REF_MM,
                "analyzer_deg": int(angle),
                "expected_petals": expected,
                "dominant_harmonic": int(petals.harmonic),
                "petal_count": int(petals.petal_count),
                "expected_harmonic_retained": bool(petals.petal_count == expected),
                "petal_orientation_deg": math.degrees(float(petals.orientation_rad)),
                "orientation_symmetry_period_deg": 360.0 / expected,
                "modulation_cv": float(petals.modulation_fraction),
                "ring_radius_um": float(petals.ring_radius_m) * 1e6,
                "ring_sample_count": int(petals.ring_sample_count),
                "petal_ring_minimum_radius_pixels": 12.0,
                "centroid_x_mm": moments.centroid_x_m * 1e3,
                "centroid_y_mm": moments.centroid_y_m * 1e3,
                "beam_ellipticity": moments.ellipticity,
                "beam_peak_intensity": moments.peak_intensity,
                "beam_power_au_m2": moments.power_au_m2,
            }
        )
    return rows, frames


def _add_wrapped_orientation_shift(rows: list[dict]) -> None:
    refs: dict[tuple[str, int, str, int], float] = {}
    for row in rows:
        if float(row["tilt_deg"]) == 0.0:
            refs[(str(row["mode"]), int(row["ell"]), str(row["direction"]), int(row["analyzer_deg"]))] = float(row["petal_orientation_deg"])
    for row in rows:
        key = (str(row["mode"]), int(row["ell"]), str(row["direction"]), int(row["analyzer_deg"]))
        ref = refs[key]
        period = float(row["orientation_symmetry_period_deg"])
        delta = ((float(row["petal_orientation_deg"]) - ref + 0.5 * period) % period) - 0.5 * period
        row["petal_orientation_shift_from_0deg_wrapped_deg"] = float(delta)


def _figure_atlas(mode: str, ell: int, cases: dict[float, tuple[VectorField, dict[int, np.ndarray]]], outdir: Path) -> None:
    # Use one scale for every tilt/analyzer of a given state, but zoom to the
    # actual ~0.2 mm Bessel/analyzer region rather than the 7.2 mm FFT support.
    global_peak = max(float(np.max(frame)) for _, frames in cases.values() for frame in frames.values())
    fig, axes = plt.subplots(len(TILTS_DEG), len(ANALYZERS_DEG), figsize=(12.5, 18.0), constrained_layout=True)
    for row, tilt in enumerate(TILTS_DEG):
        field, frames = cases[tilt]
        m = beam_moment_metrics(field)
        cx = m.centroid_x_m * 1e3
        cy = m.centroid_y_m * 1e3
        x = np.asarray(field.grid["x"]) * 1e3
        y = np.asarray(field.grid.get("y", field.grid["x"])) * 1e3
        extent = [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]
        for col, angle in enumerate(ANALYZERS_DEG):
            axes[row, col].imshow(
                frames[angle] / max(global_peak, np.finfo(float).tiny),
                origin="lower",
                extent=extent,
                vmin=0.0,
                vmax=1.0,
                aspect="equal",
            )
            axes[row, col].plot(cx, cy, "+", markersize=7)
            axes[row, col].set_xlim(cx - DISPLAY_HALF_WIDTH_MM, cx + DISPLAY_HALF_WIDTH_MM)
            axes[row, col].set_ylim(cy - DISPLAY_HALF_WIDTH_MM, cy + DISPLAY_HALF_WIDTH_MM)
            axes[row, col].set_xlabel("x (mm)")
            axes[row, col].set_ylabel("y (mm)")
            if row == 0:
                axes[row, col].set_title(f"analyzer {angle}°")
        axes[row, 0].set_ylabel(f"tilt {tilt:+.1f}°\ny (mm)")
    fig.suptitle(
        f"Phase 2H {mode} generalized cylindrical-vector analyzer atlas, ell={ell}\n"
        f"rotation about x; zref={Z_REF_MM:.0f} mm; centroid-following display crop; common intensity scale; SIMULATION ONLY",
        fontsize=13,
    )
    fig.savefig(outdir / f"analyzer_atlas_{mode}_ell{ell}_x_tilt.png", dpi=190)
    plt.close(fig)


def _figure_summary(rows: list[dict], outdir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True)
    for mode in MODES:
        for ell in ELLS:
            for direction in ("x", "y"):
                subset = [
                    r
                    for r in rows
                    if r["mode"] == mode
                    and r["ell"] == ell
                    and r["direction"] == direction
                    and r["analyzer_deg"] == 0
                ]
                subset = sorted(subset, key=lambda r: float(r["tilt_deg"]))
                label = f"{mode}, ell={ell}, rot-{direction}"
                axes[0, 0].plot([r["tilt_deg"] for r in subset], [r["petal_count"] for r in subset], marker="o", label=label)
                axes[0, 1].plot([r["tilt_deg"] for r in subset], [r["modulation_cv"] for r in subset], marker="o", label=label)
                axes[1, 0].plot(
                    [r["tilt_deg"] for r in subset],
                    [r["petal_orientation_shift_from_0deg_wrapped_deg"] for r in subset],
                    marker="o",
                    label=label,
                )
                axes[1, 1].plot([r["tilt_deg"] for r in subset], [r["ring_radius_um"] for r in subset], marker="o", label=label)
    axes[0, 0].axhline(2.0, linewidth=0.7)
    axes[0, 0].axhline(6.0, linewidth=0.7)
    axes[0, 0].set_ylabel("dominant analyzer harmonic / petal count")
    axes[0, 1].set_ylabel("annular intensity coefficient of variation")
    axes[1, 0].set_ylabel("wrapped orientation shift from 0° case (deg)")
    axes[1, 1].set_ylabel("resolved analysis-ring radius (µm)")
    for ax in axes.flat:
        ax.axvline(0.0, linewidth=0.7)
        ax.set_xlabel("rigid axicon tilt (deg)")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(
        "Phase 2H cylindrical-vector analyzer response to axicon tilt\n"
        "0° analyzer summary; full 0/45/90/135 data retained in CSV/JSON",
        fontsize=13,
    )
    fig.savefig(outdir / "analyzer_tilt_summary.png", dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    x_atlas_cases: dict[tuple[str, int], dict[float, tuple[VectorField, dict[int, np.ndarray]]]] = {
        (mode, ell): {} for mode in MODES for ell in ELLS
    }
    zero_cache: dict[tuple[str, int], tuple[VectorField, dict, list[dict], dict[int, np.ndarray]]] = {}

    for mode in MODES:
        for ell in ELLS:
            for direction in ("x", "y"):
                for tilt in TILTS_DEG:
                    if tilt == 0.0 and (mode, ell) in zero_cache:
                        field, meta, case_rows, frames = zero_cache[(mode, ell)]
                        case_rows = [dict(r, direction=direction) for r in case_rows]
                    else:
                        field, meta = _build_output(mode, ell, direction, tilt)
                        case_rows, frames = _frame_metrics(field, mode, ell, direction, tilt)
                        if tilt == 0.0:
                            zero_cache[(mode, ell)] = (field, meta, [dict(r) for r in case_rows], frames)
                    for r in case_rows:
                        r["final_flux_closure_ratio"] = float(meta["final_flux_closure_ratio"])
                        r["final_transversality_residual"] = float(meta["final_transversality_residual"])
                        r["required_nyquist_fraction"] = float(meta["required_nyquist_fraction"])
                        r["common_eikonal_p95_component_disagreement"] = float(meta["common_eikonal"]["p95_component_wavevector_disagreement_fraction"])
                        r["common_eikonal_p95_gradient_error"] = float(meta["common_eikonal"]["p95_reconstructed_gradient_error_fraction"])
                    rows.extend(case_rows)
                    if direction == "x":
                        x_atlas_cases[(mode, ell)][tilt] = (field, frames)

    _add_wrapped_orientation_shift(rows)

    with (outdir / "cylindrical_analyzer_tilt_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    retention_by_tilt = {
        str(tilt): float(np.mean([bool(r["expected_harmonic_retained"]) for r in rows if float(r["tilt_deg"]) == tilt]))
        for tilt in TILTS_DEG
    }
    retention_all = float(np.mean([bool(r["expected_harmonic_retained"]) for r in rows]))
    payload = {
        "outcome": "PHASE2H-CYLINDRICAL-VECTOR-ANALYZER-TILT-STUDY-SYNTHETIC",
        "data_classification": "synthetic_not_experimental",
        "report_figures_authorised": False,
        "simulation_only_requires_polarization_converter": True,
        "modes": list(MODES),
        "ells": list(ELLS),
        "expected_petals": {"1": 2, "3": 6},
        "analyzer_angles_deg": list(ANALYZERS_DEG),
        "tilts_deg": list(TILTS_DEG),
        "tilt_directions": ["rotation_about_x", "rotation_about_y"],
        "canonical_z_ref_mm": Z_REF_MM,
        "petal_annulus_policy": "strongest radial annulus beyond 12 output pixels, then calibrated-pixel angular Fourier harmonic",
        "modulation_metric": "std(annular intensity) / mean(annular intensity); may exceed 1 for strongly modulated patterns",
        "orientation_reporting": "orientation shift is wrapped by 360/petal_count because equivalent petal axes must not create false jumps",
        "expected_harmonic_retention_fraction_all_224_frames": retention_all,
        "expected_harmonic_retention_fraction_by_tilt": retention_by_tilt,
        "rows": rows,
    }
    (outdir / "cylindrical_analyzer_tilt_study.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for (mode, ell), cases in x_atlas_cases.items():
        _figure_atlas(mode, ell, cases, outdir)
    _figure_summary(rows, outdir)
    print(
        json.dumps(
            {
                "output_dir": str(outdir),
                "row_count": len(rows),
                "all_tilt_expected_harmonic_retention_fraction": retention_all,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
