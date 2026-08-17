"""Rewrite the Stage 8 advanced hex/polygonal/discrete notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NB_HEX = ROOT / "notebooks" / "advanced" / "02_hexagonal_polygonal_beams.ipynb"
NB_DISCRETE = ROOT / "notebooks" / "advanced" / "03_discrete_nfold_beams.ipynb"


def _source(text: str) -> list[str]:
    body = dedent(text).strip("\n")
    return [line + "\n" for line in body.splitlines()]


def md(text: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def code(text: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


def notebook(cells: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


hex_cells = [
    md(
        """
        # Stage 8 Hexagonal And Polygonal Optical Geometry

        This notebook separates four different ideas that old outputs grouped
        under "hexagon": a focal-plane polygon target, a hollow polygonal
        outline, a phase-only focal-plane approximation, and a numerically
        propagated hollow-polygon candidate.

        A hexagonal or polygonal focal-plane pattern is not automatically a
        propagation-stable Bessel-like beam. Propagation stability must be
        measured with z-dependent metrics such as accepted depth, symmetry
        retention, outline fidelity, core suppression, and side-lobe
        contamination.

        Phase-only SLM compatibility is not assumed for complex-amplitude
        polygonal targets. If complex amplitude is required, the case is
        labelled future_hardware_required or simulation_only unless a tested
        encoding route is provided.
        """
    ),
    code(
        """
        from pathlib import Path

        import numpy as np
        import pandas as pd
        from IPython.display import display

        import bessel_twin_core as bt
        from vbb_study import setup_study
        from vbb_study.equations import polygonal
        from vbb_study.publication import advanced as advanced_schema
        from vbb_study.studies import polygonal_cases

        PATHS = setup_study.bootstrap(Path.cwd())
        RUN_ID = PATHS.get("run_id") or None
        CSV_OUT = PATHS["csv"] / "advanced"
        CSV_OUT.mkdir(parents=True, exist_ok=True)
        for folder in ("stage_h2", "polygonal_hex_ring", "hex_outline"):
            (PATHS["csv"] / folder).mkdir(parents=True, exist_ok=True)
        pd.set_option("display.max_columns", 120)
        """
    ),
    md(
        """
        ## Geometry And Metric Helpers

        The focal-plane target masks below are optical geometry targets. The
        only row that is propagation-tested is explicitly propagated over z and
        receives an accepted-depth metric. No row in this notebook claims
        material writing, ablation, bonding, void formation, or a stable written
        channel.
        """
    ),
    code(
        """
        grid = bt.make_xy_grid(160, 0.30 * bt.um)
        flat_radius_m = 7.0 * bt.um
        line_width_m = 0.75 * bt.um
        target_order = 6
        z_values_m = np.linspace(0.0, 80.0 * bt.um, 9)

        R = np.asarray(grid["R"], dtype=float)
        PHI = np.asarray(grid["PHI"], dtype=float)
        polygon_radius = polygonal.polygon_radius_function(PHI, flat_radius_m, target_order)
        filled_mask = polygonal.polygonal_target_mask(
            R,
            PHI,
            flat_radius_m=flat_radius_m,
            N=target_order,
            hollow=False,
        )
        outline_mask = polygonal.polygonal_target_mask(
            R,
            PHI,
            flat_radius_m=flat_radius_m,
            N=target_order,
            line_width_m=line_width_m,
            hollow=True,
        )
        eval_mask = R <= (flat_radius_m / np.cos(np.pi / target_order) + 5.0 * bt.um)
        target_outline_intensity = np.exp(-0.5 * ((R - polygon_radius) / line_width_m) ** 2)
        target_outline_intensity = target_outline_intensity / (float(np.max(target_outline_intensity)) + bt.EPS)

        def angular_order_metrics(intensity, grid, expected_order):
            I = np.asarray(intensity, dtype=float)
            Rg = np.asarray(grid["R"], dtype=float)
            PHIg = np.asarray(grid["PHI"], dtype=float) % (2.0 * np.pi)
            annulus = (Rg >= flat_radius_m - 2.0 * line_width_m) & (Rg <= polygonal.polygon_tip_radius_m(flat_radius_m, expected_order) + 2.0 * line_width_m)
            bins = np.linspace(0.0, 2.0 * np.pi, 361)
            profile = np.zeros(360, dtype=float)
            counts = np.zeros(360, dtype=float)
            idx = np.clip(np.digitize(PHIg[annulus], bins) - 1, 0, 359)
            np.add.at(profile, idx, I[annulus])
            np.add.at(counts, idx, 1.0)
            filled = counts > 0
            profile[filled] /= counts[filled]
            amps = np.abs(np.fft.rfft(profile - float(np.mean(profile))))
            if amps.size:
                amps[0] = 0.0
            measured = int(np.argmax(amps)) if amps.size else 0
            expected_amp = float(amps[int(expected_order)]) if int(expected_order) < amps.size else 0.0
            other = np.array(amps, copy=True)
            if int(expected_order) < other.size:
                other[int(expected_order)] = 0.0
            score = expected_amp / (expected_amp + float(np.max(other)) + bt.EPS)
            return measured, float(np.clip(score, 0.0, 1.0))

        def focal_metrics(intensity, target_mask):
            I = np.asarray(intensity, dtype=float)
            norm = I / (float(np.max(I)) + bt.EPS)
            predicted = (norm >= 0.35) & eval_mask
            measured_order, symmetry = angular_order_metrics(norm, grid, target_order)
            return {
                "measured_symmetry_order": measured_order,
                "symmetry_score": symmetry,
                "outline_fidelity_score": polygonal.outline_fidelity_score(predicted, target_mask),
                "edge_uniformity_score": polygonal.edge_uniformity_score(norm, target_mask),
                "core_suppression_score": polygonal.core_suppression_score(
                    norm,
                    R,
                    core_radius_m=0.45 * flat_radius_m,
                    reference_mask=target_mask,
                ),
                "side_lobe_contamination_score": polygonal.side_lobe_contamination_score(
                    norm,
                    target_mask,
                    evaluation_mask=eval_mask,
                ),
            }
        """
    ),
    md(
        """
        ## Focal-Plane Targets

        These rows intentionally stop at the focal plane. They can be useful
        target definitions or hardware-routing diagnostics, but they are not
        propagation-stable beam claims.
        """
    ),
    code(
        """
        rows = []
        for case in polygonal_cases.polygonal_stage8_cases():
            if case["case_id"] == "hexagonal_focal_plane_target":
                intensity = filled_mask.astype(float)
                metrics = focal_metrics(intensity, filled_mask)
            elif case["case_id"] == "hollow_hexagonal_outline_target":
                intensity = target_outline_intensity
                metrics = focal_metrics(intensity, outline_mask)
            elif case["case_id"] == "phase_only_polygonal_approximation":
                phase_only_proxy = target_outline_intensity * (0.88 + 0.12 * np.cos(target_order * PHI) ** 2)
                intensity = phase_only_proxy / (float(np.max(phase_only_proxy)) + bt.EPS)
                metrics = focal_metrics(intensity, outline_mask)
            else:
                continue
            row = {
                **case,
                **metrics,
                "preset": "stage8_hexagonal_polygonal",
                "path": "focal_plane",
                "optical_model_status": "geometry_proxy",
                "material_model_status": "optical_only",
                "calibration_status": "uncalibrated",
                "accepted_depth_um": 0.0,
                "accepted_depth_definition": "not_applicable_focal_plane_only",
                "accepted_depth_fraction": 0.0,
                "canonical_zone_um": pd.NA,
                "strict_bessel_region_um": pd.NA,
            }
            rows.append(advanced_schema.annotate_advanced_beam_row(row, run_id=RUN_ID))
        display(advanced_schema.ordered_advanced_beam_frame(rows)[[
            "case_id",
            "beam_family",
            "generation_method",
            "hardware_status",
            "propagation_stability_status",
            "outline_fidelity_score",
            "phase_only_compatible",
            "complex_amplitude_required",
        ]])
        """
    ),
    md(
        """
        ## Propagation-Tested Hollow Polygon Candidate

        The field below is a simulation-only amplitude/phase target propagated
        numerically. Its acceptance is based on z-dependent outline overlap,
        sixfold retention, dark-core score, and side-lobe contamination. A
        failure or marginal result remains useful because it prevents the
        focal-plane target from being misread as a stable channel.
        """
    ),
    code(
        """
        propagation_case = next(
            case for case in polygonal_cases.polygonal_stage8_cases()
            if case["case_id"] == "propagation_tested_hollow_polygon_candidate"
        )
        U0 = target_outline_intensity * np.exp(1j * 2.0 * PHI)
        volume = bt.propagate_volume(
            U0,
            grid,
            1030.0 * bt.nm,
            z_values_m,
            n_medium=1.0,
            crop_pixels=160,
            bandlimit=True,
            method="bl_asm",
        )
        crop_grid = volume["crop_grid"]
        Rc = np.asarray(crop_grid["R"], dtype=float)
        PHIc = np.asarray(crop_grid["PHI"], dtype=float)
        poly_c = polygonal.polygon_radius_function(PHIc, flat_radius_m, target_order)
        outline_c = polygonal.polygonal_target_mask(
            Rc,
            PHIc,
            flat_radius_m=flat_radius_m,
            N=target_order,
            line_width_m=line_width_m,
            hollow=True,
        )
        eval_c = Rc <= (flat_radius_m / np.cos(np.pi / target_order) + 5.0 * bt.um)

        z_rows = []
        accepted = []
        for idx, z_m in enumerate(z_values_m):
            plane = np.asarray(volume["intensity_stack"][idx], dtype=float)
            norm = plane / (float(np.max(plane)) + bt.EPS)
            predicted = (norm >= 0.35) & eval_c
            measured_order, symmetry = angular_order_metrics(norm, crop_grid, target_order)
            outline = polygonal.outline_fidelity_score(predicted, outline_c)
            edge = polygonal.edge_uniformity_score(norm, outline_c)
            core = polygonal.core_suppression_score(norm, Rc, core_radius_m=0.45 * flat_radius_m, reference_mask=outline_c)
            side = polygonal.side_lobe_contamination_score(norm, outline_c, evaluation_mask=eval_c)
            pass_plane = bool(outline >= 0.30 and symmetry >= 0.30 and core >= 0.50 and side <= 0.85)
            accepted.append(pass_plane)
            z_rows.append({
                **propagation_case,
                "case_id": f"{propagation_case['case_id']}_z{idx:02d}",
                "preset": "stage8_hexagonal_polygonal",
                "path": "simulation_z_profile",
                "z_um": float(z_m / bt.um),
                "accepted": pass_plane,
                "optical_model_status": "numerical_propagation",
                "material_model_status": "optical_only",
                "calibration_status": "uncalibrated",
                "measured_symmetry_order": measured_order,
                "symmetry_score": symmetry,
                "outline_fidelity_score": outline,
                "edge_uniformity_score": edge,
                "core_suppression_score": core,
                "side_lobe_contamination_score": side,
                "accepted_depth_definition": "z planes passing outline>=0.30, symmetry>=0.30, core>=0.50, side<=0.85",
            })

        depth = polygonal.accepted_depth_from_metric_stack(z_values_m, accepted)
        propagation_metrics = {
            "measured_symmetry_order": int(round(pd.Series([r["measured_symmetry_order"] for r in z_rows]).mode().iloc[0])),
            "symmetry_score": float(np.mean([r["symmetry_score"] for r in z_rows])),
            "outline_fidelity_score": float(np.mean([r["outline_fidelity_score"] for r in z_rows])),
            "edge_uniformity_score": float(np.mean([r["edge_uniformity_score"] for r in z_rows])),
            "core_suppression_score": float(np.mean([r["core_suppression_score"] for r in z_rows])),
            "side_lobe_contamination_score": float(np.mean([r["side_lobe_contamination_score"] for r in z_rows])),
            "accepted_depth_um": depth["accepted_depth_um"],
            "accepted_depth_fraction": depth["accepted_depth_fraction"],
            "accepted_plane_count": depth["accepted_plane_count"],
            "accepted_z_start_um": depth["accepted_z_start_um"],
            "accepted_z_end_um": depth["accepted_z_end_um"],
            "accepted_depth_definition": "longest contiguous z span passing outline/symmetry/core/side-lobe gate",
            "propagation_power_drift_fraction": float(
                (np.max(volume["total_power"]) - np.min(volume["total_power"]))
                / (np.mean(volume["total_power"]) + bt.EPS)
            ),
        }
        rows.append(advanced_schema.annotate_advanced_beam_row({
            **propagation_case,
            **propagation_metrics,
            "preset": "stage8_hexagonal_polygonal",
            "path": "simulation",
            "optical_model_status": "numerical_propagation",
            "material_model_status": "optical_only",
            "calibration_status": "uncalibrated",
        }, run_id=RUN_ID))

        z_profile = advanced_schema.ordered_advanced_beam_frame([
            advanced_schema.annotate_advanced_beam_row({
                **r,
                "accepted_depth_um": depth["accepted_depth_um"],
                "accepted_depth_fraction": depth["accepted_depth_fraction"],
                "propagation_power_drift_fraction": propagation_metrics["propagation_power_drift_fraction"],
            }, run_id=RUN_ID)
            for r in z_rows
        ])
        display(z_profile[[
            "case_id",
            "z_um",
            "accepted",
            "outline_fidelity_score",
            "symmetry_score",
            "core_suppression_score",
            "side_lobe_contamination_score",
            "propagation_stability_status",
        ]])
        """
    ),
    md(
        """
        ## Canonical Outputs

        The canonical Stage 8 CSVs are written under `outputs/csv/advanced`.
        Old `stage_h2`, `polygonal_hex_ring`, and `hex_outline` filenames are
        refreshed as compatibility copies with the same native metadata.
        """
    ),
    code(
        """
        summary = advanced_schema.ordered_advanced_beam_frame(rows)
        acceptance = summary.copy()
        acceptance["acceptance_check"] = acceptance["advanced_acceptance_label"]
        acceptance["acceptance_pass"] = (
            acceptance["propagation_stability_status"].isin([
                "propagation_tested_pass",
                "propagation_tested_marginal",
            ])
            & ~acceptance["focal_plane_only"].astype(bool)
        )

        summary_path = CSV_OUT / "hexagonal_polygonal_beam_summary.csv"
        acceptance_path = CSV_OUT / "hexagonal_polygonal_acceptance_summary.csv"
        summary.to_csv(summary_path, index=False)
        acceptance.to_csv(acceptance_path, index=False)

        compatibility_summary = [
            PATHS["csv"] / "stage_h2" / "H2_air_knob_sweep.csv",
            PATHS["csv"] / "stage_h2" / "H2_survival_summary.csv",
            PATHS["csv"] / "stage_h2" / "H2_transient_hexlike_scan.csv",
            PATHS["csv"] / "polygonal_hex_ring" / "11_polygonal_hex_ring_acceptance_metrics.csv",
            PATHS["csv"] / "polygonal_hex_ring" / "11_polygonal_hex_ring_materials_proxy.csv",
            PATHS["csv"] / "polygonal_hex_ring" / "12_hollow_hex_sidelobe_ideal_sweep.csv",
            PATHS["csv"] / "polygonal_hex_ring" / "12_hollow_hex_sidelobe_lab_shortlist.csv",
            PATHS["csv"] / "hex_outline" / "13_hollow_hex_outline_checkpoint.csv",
            PATHS["csv"] / "hex_outline" / "14_hexlike_transient_vs_outline.csv",
            PATHS["csv"] / "hex_outline" / "15_hybrid_transient_seed_lab_gate.csv",
            PATHS["csv"] / "hex_outline" / "16_hex_bessel_like_summary.csv",
            PATHS["csv"] / "hex_outline" / "17_zernike_hex_bessel_sweep.csv",
        ]
        for path in compatibility_summary:
            summary.to_csv(path, index=False)
        z_profile.to_csv(PATHS["csv"] / "polygonal_hex_ring" / "11_polygonal_hex_ring_z_stability.csv", index=False)
        z_profile.to_csv(PATHS["csv"] / "hex_outline" / "16_hex_bessel_like_z_profile.csv", index=False)

        assert not summary["material_writing_success_claimed"].astype(bool).any()
        assert not summary["stable_written_channel_claimed"].astype(bool).any()
        assert not ((summary["focal_plane_only"].astype(bool)) & (summary["propagation_tested"].astype(bool))).any()
        assert not (
            summary["complex_amplitude_required"].astype(bool)
            & summary["phase_only_compatible"].astype(bool)
        ).any()
        display(summary[[
            "case_id",
            "beam_family",
            "model_level",
            "generation_method",
            "hardware_status",
            "propagation_stability_status",
            "advanced_acceptance_label",
            "material_writing_success_claimed",
            "stable_written_channel_claimed",
        ]])
        print(summary_path)
        print(acceptance_path)
        """
    ),
]


discrete_cells = [
    md(
        """
        # Stage 8 Discrete N-Fold Optical Fields

        This notebook treats finite N-fold beams as discrete plane-wave
        superpositions on one transverse-k ring. These are optical fields and
        geometry diagnostics only. A clean N-fold transverse pattern is not a
        material-writing result, and a focal or short-z symmetry metric is not
        a stable written channel.

        Ideal N-wave rows are simulation-only or future-hardware targets.
        Phase-only proxy rows are labelled current-lab-realizable only for the
        encoded phase-only optical command, not for material modification.
        """
    ),
    code(
        """
        from pathlib import Path

        import numpy as np
        import pandas as pd
        from IPython.display import display

        import bessel_twin_core as bt
        from vbb_study import setup_study
        from vbb_study.equations import polygonal
        from vbb_study.publication import advanced as advanced_schema
        from vbb_study.studies import discrete_nfold_cases

        PATHS = setup_study.bootstrap(Path.cwd())
        RUN_ID = PATHS.get("run_id") or None
        CSV_OUT = PATHS["csv"] / "advanced"
        CSV_OUT.mkdir(parents=True, exist_ok=True)
        (PATHS["csv"] / "stage10_discrete").mkdir(parents=True, exist_ok=True)
        pd.set_option("display.max_columns", 120)
        """
    ),
    md(
        """
        ## Field And Metric Helpers

        The ideal branch uses a finite N-wave complex field. The phase-only
        branch keeps only the phase of that target under a Gaussian envelope.
        Both branches are propagated numerically and scored with symmetry,
        side-lobe, core, and accepted-depth metrics.
        """
    ),
    code(
        """
        grid = bt.make_xy_grid(128, 0.35 * bt.um)
        R = np.asarray(grid["R"], dtype=float)
        PHI = np.asarray(grid["PHI"], dtype=float)
        kr_m_inv = 1.55 / bt.um
        waist_m = 28.0 * bt.um
        wavelength_m = 1030.0 * bt.nm
        z_values_m = np.linspace(0.0, 70.0 * bt.um, 8)
        gaussian = bt.gaussian_amplitude(R, waist_m)

        def angular_order_metrics(intensity, grid, expected_order):
            I = np.asarray(intensity, dtype=float)
            Rg = np.asarray(grid["R"], dtype=float)
            PHIg = np.asarray(grid["PHI"], dtype=float) % (2.0 * np.pi)
            r0 = 2.0 * np.pi / kr_m_inv
            annulus = (Rg >= 0.45 * r0) & (Rg <= 2.6 * r0)
            bins = np.linspace(0.0, 2.0 * np.pi, 361)
            profile = np.zeros(360, dtype=float)
            counts = np.zeros(360, dtype=float)
            idx = np.clip(np.digitize(PHIg[annulus], bins) - 1, 0, 359)
            np.add.at(profile, idx, I[annulus])
            np.add.at(counts, idx, 1.0)
            filled = counts > 0
            profile[filled] /= counts[filled]
            amps = np.abs(np.fft.rfft(profile - float(np.mean(profile))))
            if amps.size:
                amps[0] = 0.0
            measured = int(np.argmax(amps)) if amps.size else 0
            expected_amp = float(amps[int(expected_order)]) if int(expected_order) < amps.size else 0.0
            other = np.array(amps, copy=True)
            if int(expected_order) < other.size:
                other[int(expected_order)] = 0.0
            score = expected_amp / (expected_amp + float(np.max(other)) + bt.EPS)
            return measured, float(np.clip(score, 0.0, 1.0))

        def score_plane(plane, grid, expected_order):
            I = np.asarray(plane, dtype=float)
            norm = I / (float(np.max(I)) + bt.EPS)
            Rg = np.asarray(grid["R"], dtype=float)
            r0 = 2.0 * np.pi / kr_m_inv
            signal = (Rg >= 0.45 * r0) & (Rg <= 2.6 * r0)
            eval_mask = Rg <= 6.0 * r0
            measured, symmetry = angular_order_metrics(norm, grid, expected_order)
            return {
                "measured_symmetry_order": measured,
                "symmetry_score": symmetry,
                "outline_fidelity_score": pd.NA,
                "edge_uniformity_score": polygonal.edge_uniformity_score(norm, signal),
                "core_suppression_score": polygonal.core_suppression_score(
                    norm,
                    Rg,
                    core_radius_m=0.45 * r0,
                    reference_mask=signal,
                ),
                "side_lobe_contamination_score": polygonal.side_lobe_contamination_score(
                    norm,
                    signal,
                    evaluation_mask=eval_mask,
                ),
            }
        """
    ),
    md(
        """
        ## N-Fold Propagation Suite

        The accepted depth is the longest contiguous z range where the measured
        angular order matches the requested order and the symmetry score stays
        above the explicit threshold. Passing this gate is an optical
        propagation statement only.
        """
    ),
    code(
        """
        rows = []
        z_detail_rows = []
        for case in discrete_nfold_cases.discrete_nfold_stage8_cases():
            order = int(case["target_symmetry_order"])
            ell = int(case.get("ell", 0))
            target_field = polygonal.discrete_nfold_field(
                R,
                PHI,
                kr_m_inv=kr_m_inv,
                N=order,
                ell=ell,
            ) * gaussian
            if case["path"] == "phase_only_proxy":
                U0 = np.exp(1j * np.angle(target_field)) * gaussian
                hardware_status = "current_lab_realizable"
            else:
                U0 = target_field
                hardware_status = "simulation_only"
            volume = bt.propagate_volume(
                U0,
                grid,
                wavelength_m,
                z_values_m,
                n_medium=1.0,
                crop_pixels=128,
                bandlimit=True,
                method="bl_asm",
            )
            accepted = []
            per_z = []
            for idx, z_m in enumerate(z_values_m):
                metrics = score_plane(volume["intensity_stack"][idx], volume["crop_grid"], order)
                pass_plane = bool(
                    int(metrics["measured_symmetry_order"]) == order
                    and float(metrics["symmetry_score"]) >= 0.18
                )
                accepted.append(pass_plane)
                per_z.append({**metrics, "z_um": float(z_m / bt.um), "accepted": pass_plane})
            depth = polygonal.accepted_depth_from_metric_stack(z_values_m, accepted)
            summary_metrics = {
                "measured_symmetry_order": int(round(pd.Series([p["measured_symmetry_order"] for p in per_z]).mode().iloc[0])),
                "symmetry_score": float(np.mean([p["symmetry_score"] for p in per_z])),
                "edge_uniformity_score": float(np.mean([p["edge_uniformity_score"] for p in per_z])),
                "core_suppression_score": float(np.mean([p["core_suppression_score"] for p in per_z])),
                "side_lobe_contamination_score": float(np.mean([p["side_lobe_contamination_score"] for p in per_z])),
                "accepted_depth_um": depth["accepted_depth_um"],
                "accepted_depth_fraction": depth["accepted_depth_fraction"],
                "accepted_plane_count": depth["accepted_plane_count"],
                "accepted_z_start_um": depth["accepted_z_start_um"],
                "accepted_z_end_um": depth["accepted_z_end_um"],
                "accepted_depth_definition": "longest contiguous z span with measured order N and symmetry_score >= 0.18",
                "canonical_zone_um": float(depth["accepted_depth_um"]),
                "strict_bessel_region_um": float(depth["accepted_depth_um"]) if depth["accepted_depth_fraction"] >= 0.65 else 0.0,
                "propagation_power_drift_fraction": float(
                    (np.max(volume["total_power"]) - np.min(volume["total_power"]))
                    / (np.mean(volume["total_power"]) + bt.EPS)
                ),
            }
            row = advanced_schema.annotate_advanced_beam_row({
                **case,
                **summary_metrics,
                "hardware_status": hardware_status,
                "optical_model_status": "numerical_propagation",
                "material_model_status": "optical_only",
                "calibration_status": "uncalibrated",
            }, run_id=RUN_ID)
            rows.append(row)
            for idx, detail in enumerate(per_z):
                z_detail_rows.append(advanced_schema.annotate_advanced_beam_row({
                    **case,
                    **detail,
                    "case_id": f"{case['case_id']}_z{idx:02d}",
                    "hardware_status": hardware_status,
                    "optical_model_status": "numerical_propagation",
                    "material_model_status": "optical_only",
                    "calibration_status": "uncalibrated",
                    "accepted_depth_um": depth["accepted_depth_um"],
                    "accepted_depth_fraction": depth["accepted_depth_fraction"],
                    "accepted_depth_definition": "parent case accepted-depth gate",
                    "propagation_power_drift_fraction": summary_metrics["propagation_power_drift_fraction"],
                }, run_id=RUN_ID))

        summary = advanced_schema.ordered_advanced_beam_frame(rows)
        z_profile = advanced_schema.ordered_advanced_beam_frame(z_detail_rows)
        display(summary[[
            "case_id",
            "target_symmetry_order",
            "path",
            "generation_method",
            "hardware_status",
            "symmetry_score",
            "accepted_depth_um",
            "propagation_stability_status",
            "phase_only_compatible",
            "complex_amplitude_required",
        ]])
        """
    ),
    md(
        """
        ## Canonical Outputs

        Canonical Stage 8 discrete CSVs are written under
        `outputs/csv/advanced`. Older `stage10_discrete` names are refreshed as
        schema-native compatibility copies. The old material-threshold view is
        intentionally not regenerated as a material claim.
        """
    ),
    code(
        """
        acceptance = summary.copy()
        acceptance["acceptance_check"] = acceptance["advanced_acceptance_label"]
        acceptance["acceptance_pass"] = acceptance["propagation_stability_status"].isin([
            "propagation_tested_pass",
            "propagation_tested_marginal",
        ])
        summary_path = CSV_OUT / "discrete_nfold_beam_summary.csv"
        acceptance_path = CSV_OUT / "discrete_nfold_acceptance_summary.csv"
        summary.to_csv(summary_path, index=False)
        acceptance.to_csv(acceptance_path, index=False)

        compat_dir = PATHS["csv"] / "stage10_discrete"
        for name in (
            "10_discrete_pattern_summary.csv",
            "10_discrete_cgh_exports.csv",
            "10_discrete_encoding_comparison.csv",
            "10_discrete_continuous_limit.csv",
        ):
            summary.to_csv(compat_dir / name, index=False)
        acceptance.to_csv(compat_dir / "10_discrete_acceptance_summary.csv", index=False)

        assert not summary["material_writing_success_claimed"].astype(bool).any()
        assert not summary["stable_written_channel_claimed"].astype(bool).any()
        assert not (summary["simulation_only"].astype(bool) & summary["current_lab_realizable"].astype(bool)).any()
        assert not (
            summary["complex_amplitude_required"].astype(bool)
            & summary["phase_only_compatible"].astype(bool)
        ).any()
        display(acceptance[[
            "case_id",
            "acceptance_pass",
            "advanced_acceptance_label",
            "hardware_status",
            "propagation_stability_status",
            "material_writing_success_claimed",
            "stable_written_channel_claimed",
        ]])
        print(summary_path)
        print(acceptance_path)
        """
    ),
]


NB_HEX.write_text(json.dumps(notebook(hex_cells), indent=1) + "\n", encoding="utf-8")
NB_DISCRETE.write_text(json.dumps(notebook(discrete_cells), indent=1) + "\n", encoding="utf-8")
print(f"Updated {NB_HEX.relative_to(ROOT)}")
print(f"Updated {NB_DISCRETE.relative_to(ROOT)}")
