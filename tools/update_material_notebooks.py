"""Rewrite Stage 6 material notebooks with clean proxy-only content."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks" / "materials"


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


def write_notebook(path: Path, cells: list[dict[str, object]]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


COMMON_IMPORTS = """
from pathlib import Path

import pandas as pd
from IPython.display import display

import bessel_twin_core as bt
from Publication_Study import publication_diagnostics as pdiag
from vbb_study import setup_study, vbb_materials
from vbb_study.publication import materials as material_schema

PATHS = setup_study.bootstrap(Path.cwd())
CSV_OUT = PATHS["csv"] / "materials"
CSV_OUT.mkdir(parents=True, exist_ok=True)
"""


BUILD_OR_READ_SUMMARY = """
summary_path = CSV_OUT / "material_proxy_fluence_threshold_summary.csv"
if summary_path.exists():
    summary = pd.read_csv(summary_path)
else:
    summary, _cases = vbb_materials.build_shortlist_design_table(
        pdiag.DEFAULT_SHORTLIST,
        preset="fast",
        path="realistic",
    )
    summary = material_schema.ordered_material_frame(summary.to_dict("records"))
    summary.to_csv(summary_path, index=False)
"""


write_notebook(
    NB_DIR / "01_material_proxy_fluence_and_thresholds.ipynb",
    [
        md(
            """
            # Stage 6 Materials Proxy: Fluence And Thresholds

            This notebook computes optical fluence and threshold-comparison
            planning metrics from simulated scalar/vortex beam cases.

            These material-facing results are planning proxies unless explicitly
            marked experimentally_calibrated. A thresholded fluence map is not a
            calibrated prediction of ablation, void formation, refractive-index
            change, or weld success.
            """
        ),
        code(COMMON_IMPORTS),
        md(
            """
            ## Optical Fluence Versus Material Response

            The optical field supplies intensity, pulse energy, propagation QA,
            and Bessel-zone metrics. The material layer only compares that
            optical fluence with configured threshold proxies. No row here is
            experimentally calibrated.
            """
        ),
        code(
            """
            summary, cases = vbb_materials.build_shortlist_design_table(
                pdiag.DEFAULT_SHORTLIST,
                preset="fast",
                path="realistic",
            )
            summary = material_schema.ordered_material_frame(summary.to_dict("records"))

            route_notes = {
                "scalar_bessel": "ell=0 scalar Bessel optical field",
                "vortex_bessel": "ell>0 vortex Bessel optical field",
                "vector": "vector optical route can be joined later through shared optical metrics",
            }
            summary["source_optical_route"] = summary["beam_family"].map(route_notes).fillna(summary["beam_family"])
            summary["output_category"] = "planning_proxy"
            summary["safe_for_design_comparison"] = True

            summary = material_schema.ordered_material_frame(summary.to_dict("records"))
            summary_path = CSV_OUT / "material_proxy_fluence_threshold_summary.csv"
            summary.to_csv(summary_path, index=False)

            compatibility_path = CSV_OUT / "07_materials_design_table.csv"
            summary.to_csv(compatibility_path, index=False)

            display_cols = [
                "case_id",
                "beam_family",
                "material_model_status",
                "calibration_status",
                "threshold_source",
                "peak_fluence_J_cm2",
                "fluence_to_threshold_ratio",
                "thresholded_area_um2",
                "xz_energy_conservation_status",
            ]
            display(summary[display_cols])
            print(summary_path)
            print(compatibility_path)
            """
        ),
        md(
            """
            ## Reading The Proxy

            The threshold columns are safe for relative planning comparison
            between optical cases. They are not material-response predictions.
            Line-fluence XZ maps are diagnostic planning visualisations unless
            explicitly generated from an energy-conserving 3D deposition model.
            """
        ),
    ],
)


write_notebook(
    NB_DIR / "02_material_calibration_template.ipynb",
    [
        md(
            """
            # Stage 6 Materials Proxy: Calibration Template

            This notebook writes the table a lab user would fill after real
            calibration shots. The blank measured columns are intentional and
            prevent uncalibrated threshold proxies from being presented as
            material-response models.
            """
        ),
        code(COMMON_IMPORTS),
        code(BUILD_OR_READ_SUMMARY),
        md(
            """
            ## Required Measurements

            A calibrated material row needs material, wavelength, pulse
            duration, repetition rate, pulse count, NA or cone angle, measured
            modification threshold, measured line width or depth, microscope or
            etch method, and uncertainty.
            """
        ),
        code(
            """
            template = vbb_materials.calibration_template_from_proxy_summary(summary)
            cfg = bt.default_config("fast")
            template["wavelength_nm"] = cfg.laser.wavelength_m / bt.nm
            template["pulse_duration_fs"] = cfg.laser.pulse_duration_s / bt.fs
            template["repetition_rate_Hz"] = cfg.laser.rep_rate_Hz
            template["NA"] = cfg.objective.NA
            if "gamma_slm_deg" in template:
                template["cone_angle_deg"] = template["gamma_slm_deg"]
            template["calibration_status"] = "uncalibrated"
            template["material_model_status"] = "planning_proxy"
            template["material_response_model"] = "incubation_threshold_proxy"
            template = material_schema.ordered_material_frame(template.to_dict("records"))

            template_path = CSV_OUT / "material_calibration_template.csv"
            template.to_csv(template_path, index=False)

            display_cols = [
                "case_id",
                "material_name",
                "wavelength_nm",
                "pulse_duration_fs",
                "repetition_rate_Hz",
                "pulse_count",
                "measured_threshold_fluence_J_cm2",
                "measured_line_width_um",
                "microscope_or_etch_method",
                "measurement_uncertainty_um",
                "calibration_status",
            ]
            display(template[display_cols])
            print(template_path)
            """
        ),
        md(
            """
            ## Calibration Boundary

            Filling a measured threshold alone is not enough to claim a full
            ablation, void, refractive-index-change, or weld model. Fully
            calibrated rows require calibration evidence and a calibrated
            response-model label.
            """
        ),
    ],
)


write_notebook(
    NB_DIR / "03_application_design_tables.ipynb",
    [
        md(
            """
            # Stage 6 Materials Proxy: Application Design Tables

            This notebook writes application-facing planning tables for
            comparing candidate optical cases. It does not start capsule,
            weld-feature, hexagon, polygonal, or discrete N-fold studies.
            """
        ),
        code(COMMON_IMPORTS),
        code(BUILD_OR_READ_SUMMARY),
        md(
            """
            ## Design Comparison

            Rows are ranked by optical fluence margin and kept in the
            planning_proxy category. The rank is a design-planning aid, not a
            prediction of material modification success.
            """
        ),
        code(
            """
            design = vbb_materials.application_design_table_from_proxy_summary(summary)
            design["source_optical_route"] = design["beam_family"].map({
                "scalar_bessel": "scalar Bessel optical metric",
                "vortex_bessel": "vortex Bessel optical metric",
            }).fillna(design["beam_family"])
            design["qa_gate"] = design.apply(
                lambda row: "compare_proxy_only"
                if row["material_model_status"] == "planning_proxy"
                and row["calibration_status"] == "uncalibrated"
                else "review_required",
                axis=1,
            )
            design = material_schema.ordered_material_frame(design.to_dict("records"))

            design_path = CSV_OUT / "material_application_design_table.csv"
            design.to_csv(design_path, index=False)

            display_cols = [
                "planning_rank",
                "case_id",
                "beam_family",
                "qa_gate",
                "material_model_status",
                "calibration_status",
                "fluence_to_threshold_ratio",
                "thresholded_equivalent_diameter_um",
                "xz_proxy_length_um",
                "xz_energy_conservation_status",
            ]
            display(design[display_cols])
            print(design_path)
            """
        ),
        md(
            """
            ## Safe Use

            The table is safe for design comparison because every row carries
            native schema metadata and remains uncalibrated. Experimental
            calibration must be joined before making material-response claims.
            """
        ),
    ],
)

print("Updated material notebooks:")
for path in sorted(NB_DIR.glob("*.ipynb")):
    print(path.relative_to(ROOT))
