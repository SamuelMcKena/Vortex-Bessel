"""Rewrite the Stage 7 capsule notebook with clean proxy-only content."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "advanced" / "01_capsule_weld_feature_design.ipynb"


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


cells = [
    md(
        """
        # Stage 7 Capsule / Weld-Feature Planning Geometry

        This notebook compares simulated optical/proxy geometry with a
        capsule-like application target. These capsule/weld-feature results
        are application-planning geometry proxies. They do not predict actual
        weld success, bonding, void formation, ablation, or refractive-index
        change without experimental calibration.

        A pass score means the simulated optical/proxy geometry matches the
        requested target under the model assumptions. It does not mean the
        material will physically produce that feature.
        """
    ),
    code(
        """
        from dataclasses import replace
        from pathlib import Path

        import numpy as np
        import pandas as pd
        from IPython.display import display

        import bessel_twin_core as bt
        from vbb_study import setup_study, vbb_capsule
        from vbb_study.publication import capsule as capsule_schema

        PATHS = setup_study.bootstrap(Path.cwd())
        CSV_OUT = PATHS["csv"] / "capsule"
        CSV_COMPAT = PATHS["csv"] / "stage9_capsule"
        FIG_OUT = PATHS["figures"] / "capsule"
        CSV_OUT.mkdir(parents=True, exist_ok=True)
        CSV_COMPAT.mkdir(parents=True, exist_ok=True)
        FIG_OUT.mkdir(parents=True, exist_ok=True)
        pd.set_option("display.max_columns", 90)
        """
    ),
    md(
        """
        ## Target And QA Boundary

        The target is a capsule-like XZ outline with a transverse width near
        4.2 um and length/depth near 200 um. The comparison uses optical
        geometry, thresholded fluence proxies, and propagation QA labels.
        `fail` propagation rows are not design-ready. `marginal` rows are
        exploratory. `pass` rows are still planning proxies unless
        experimentally calibrated.
        """
    ),
    code(
        """
        TARGET_WIDTH_UM = 4.2
        TARGET_LENGTH_UM = 200.0
        TARGET_DEPTH_UM = 200.0

        base = bt.default_config("fast")
        capsule_config = replace(
            base,
            grid=replace(
                base.grid,
                N=256,
                ideal_N=160,
                ideal_dx_m=0.30 * bt.um,
                crop_pixels=112,
                axial_points=9,
                axial_range_m=120.0 * bt.um,
                axial_target_factor=1.4,
                device_downsample=8,
                label="fast",
            ),
            target=replace(
                base.target,
                ell=0,
                target_core_diameter_m=TARGET_WIDTH_UM * bt.um,
                target_bessel_length_m=TARGET_LENGTH_UM * bt.um,
            ),
            energy=replace(base.energy, pulse_energy_in_J=20.0 * bt.uJ),
        )

        print(f"Target width={TARGET_WIDTH_UM:.1f} um, length={TARGET_LENGTH_UM:.0f} um")
        """
    ),
    md(
        """
        ## Optical Candidate Ranking

        The inverse-design sweep is optical-only. It ranks candidate spot/zone
        geometry and hardware reachability before any thresholded material
        proxy is used.
        """
    ),
    code(
        """
        design_df = vbb_capsule.design_solver(
            capsule_config,
            spot_range_um=(2.0, 5.0),
            target_zone_um=TARGET_LENGTH_UM,
            wavelength_m=capsule_config.laser.wavelength_m,
            ell_values=(0, 1, 2, 3),
            spot_samples_um=(2.0, 3.0, TARGET_WIDTH_UM, 5.0),
            energy_samples_uJ=(10.0, 20.0, 40.0),
        )

        ranking = vbb_capsule.candidate_ranking_from_design_solver(
            design_df,
            target_width_um=TARGET_WIDTH_UM,
            target_length_um=TARGET_LENGTH_UM,
            target_depth_um=TARGET_DEPTH_UM,
        )
        ranking_path = CSV_OUT / "capsule_candidate_ranking.csv"
        ranking.to_csv(ranking_path, index=False)
        ranking.to_csv(CSV_COMPAT / "09_capsule_design_feasible_set.csv", index=False)

        display_cols = [
            "planning_rank",
            "case_id",
            "geometry_model_status",
            "material_model_status",
            "calibration_status",
            "predicted_width_um",
            "predicted_length_um",
            "capsule_fit_score",
            "capsule_acceptance_label",
            "hardware_status",
        ]
        display(ranking[display_cols].head(12))
        print(ranking_path)
        """
    ),
    md(
        """
        ## Thresholded Capsule Geometry Proxy

        The sweep compares ideal and lab-realistic scalar propagation paths.
        The XZ map is a normalised planning visualisation that preserves the
        axial envelope for geometry comparison. It is not an energy-conserving
        3D deposition model and not a weld/material-response predictor.
        """
    ),
    code(
        """
        z_values = np.linspace(0.0, 250.0 * bt.um, 9)
        strengths = (0.0, 0.35, 0.70, 1.00)
        sweep_df, capsule_cases = vbb_capsule.sweep_capsule_apodization(
            capsule_config,
            strengths=strengths,
            z_values_m=z_values,
        )

        summary = vbb_capsule.capsule_summary_from_cases(
            capsule_cases,
            target_width_um=TARGET_WIDTH_UM,
            target_length_um=TARGET_LENGTH_UM,
            target_depth_um=TARGET_DEPTH_UM,
        )
        summary_path = CSV_OUT / "capsule_weld_feature_design_summary.csv"
        summary.to_csv(summary_path, index=False)
        summary.to_csv(CSV_COMPAT / "09_capsule_apodization_sweep.csv", index=False)

        display_cols = [
            "case_id",
            "path",
            "geometry_model_status",
            "material_model_status",
            "calibration_status",
            "propagation_power_label",
            "predicted_width_um",
            "predicted_length_um",
            "overlap_score",
            "capsule_fit_score",
            "capsule_acceptance_label",
            "xz_energy_conservation_status",
        ]
        display(summary[display_cols])
        print(summary_path)
        """
    ),
    md(
        """
        ## Acceptance Summary

        Acceptance is a planning label. It means the simulated proxy geometry
        is useful for comparing designs under the current assumptions. It does
        not mean that a physical weld, bond, void, ablation mark, or refractive
        index change will occur.
        """
    ),
    code(
        """
        acceptance = vbb_capsule.capsule_acceptance_summary(summary)
        acceptance_path = CSV_OUT / "capsule_acceptance_summary.csv"
        acceptance.to_csv(acceptance_path, index=False)
        acceptance.to_csv(CSV_COMPAT / "09_capsule_acceptance_summary.csv", index=False)

        display_cols = [
            "case_id",
            "acceptance_pass",
            "capsule_acceptance_label",
            "propagation_power_label",
            "geometry_model_status",
            "material_model_status",
            "calibration_status",
            "actual_weld_success_claimed",
        ]
        display(acceptance[display_cols])
        assert not acceptance["actual_weld_success_claimed"].astype(bool).any()
        assert not (
            (acceptance["calibration_status"] == "uncalibrated")
            & (acceptance["material_model_status"] == "experimentally_calibrated")
        ).any()
        print(acceptance_path)
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(f"Updated {NB.relative_to(ROOT)}")
