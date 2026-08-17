"""Update Phase 9 vector notebooks with native schema metadata.

This script edits only ``notebooks/vector`` and creates the requested hardware
routes notebook. It does not alter scalar, lab-realism, materials, or advanced
notebooks.
"""

from __future__ import annotations

from pathlib import Path

import nbformat

_pub = Path(__file__).resolve().parent.parent
_nb_dir = _pub / "notebooks" / "vector"


def _write(name: str, nb) -> None:
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    nbformat.write(nb, _nb_dir / name)
    print(f"updated {name}")


def _code(src: str):
    return nbformat.v4.new_code_cell(src.rstrip() + "\n")


def _markdown(src: str):
    return nbformat.v4.new_markdown_cell(src.rstrip() + "\n")


BOOTSTRAP = """\
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bessel_twin_core as bt
from vbb_study import setup_study, vbb_vector, vbb_style
from vbb_study.publication import vector as vector_schema

PATHS = setup_study.bootstrap(Path.cwd())
PRESET = "fast"
RUN_ID = PATHS.get("run_id") or None
out_csv = PATHS["csv"] / "vector"
out_fig = PATHS["figures"] / "vector"
compat_csv = PATHS["csv"] / "publication_study"
out_csv.mkdir(parents=True, exist_ok=True)
out_fig.mkdir(parents=True, exist_ok=True)
compat_csv.mkdir(parents=True, exist_ok=True)
vbb_style.apply_style()

cfg = bt.default_config(PRESET)
design = bt.compute_design_from_targets(cfg.laser, cfg.target, cfg.material)
grid = bt.make_xy_grid(256, 0.18 * bt.um)
KR = 0.95 / bt.um
WAIST = 48.0 * bt.um
ELL_VALUES = (1, 3)
"""


NOTEBOOK_01_HELPERS = """\
def _stamp(row):
    vector_schema.annotate_vector_row(row, run_id=RUN_ID, qa_status="exploratory")
    return row

def _case_metrics(case, *, target=None):
    radius = vbb_vector.predicted_ring_radius(int(case["ell"]), float(case["kr_m_inv"]))
    roi = vbb_vector.ring_roi(case["grid"], radius, rel_width=0.25)
    stokes = case["stokes"]
    circ = vbb_vector.circularity_residual(stokes, roi)
    row = {
        "ell": int(case["ell"]),
        "ring_radius_um": float(radius / bt.um),
        "petal_count_0deg": vbb_vector.petal_count_and_orientation(case["analyzer"][0], case["grid"], radius)["petal_count"],
        "petal_count_45deg": vbb_vector.petal_count_and_orientation(case["analyzer"][45], case["grid"], radius)["petal_count"],
        "mean_abs_S3_over_S0_roi": circ["mean_abs_s3_over_s0"],
        "max_abs_S3_over_S0_roi": circ["max_abs_s3_over_s0"],
        "total_power_au": float(np.sum(case["total_intensity"]) * float(case["grid"]["dx"]) ** 2),
    }
    if target in {"radial", "azimuthal"}:
        row["orientation_rms_error_rad"] = vbb_vector.mode_orientation_error(
            case["ellipse"]["psi"], case["grid"], roi, target=target, order=max(1, abs(int(case["ell"])))
        )
    return row
"""


NOTEBOOK_01_RUN = """\
rows = []
scalar_rows = []
vector_cases = []

for ell in ELL_VALUES:
    envelope = vbb_vector.scalar_bg_envelope(grid, ell=ell, kr_m_inv=KR, waist_m=WAIST)
    scalar_intensity = np.abs(envelope) ** 2
    scalar_row = _stamp({
        "case_id": f"scalar_reference_ell{ell}",
        "path": "vector_theory_atlas",
        "beam_family": "scalar_reference",
        "model_level": "scalar_reference",
        "generation_method": "scalar_reference",
        "vector_mode": "scalar_reference",
        "vector_model": "scalar_sas_with_jones_overlay",
        "vector_program": "scalar_reference",
        "vector_method": "not_applicable",
        "vector_encoder_hardware": "scalar_holographic_reference",
        "lab_realizable": True,
        "simulation_only": False,
        "requires_element": "none",
        "uses_waveplates": False,
        "uses_two_slm": False,
        "uses_shared_director_axis": False,
        "ell": ell,
        "target_core_diameter_um": float(cfg.target.target_core_diameter_m / bt.um),
        "vortex_main_ring_diameter_um": float(2.0 * vbb_vector.predicted_ring_radius(ell, KR) / bt.um),
        "retained_power_fraction": 1.0,
        "peak_intensity_au": float(np.max(scalar_intensity)),
    })
    scalar_rows.append(scalar_row)
    rows.append(dict(scalar_row))

    for mode in ("radial", "azimuthal"):
        case = vbb_vector.build_analytic_vector_mode(grid, ell=ell, kr_m_inv=KR, waist_m=WAIST, mode=mode)
        vector_cases.append(case)
        row = {
            "case_id": f"ideal_{mode}_ell{ell}",
            "path": "vector_theory_atlas",
            "beam_family": "vector",
            "model_level": "ideal_target",
            "generation_method": "qplate_or_vector_converter",
            "vector_mode": mode,
            "vector_model": "ideal_jones_target",
            "vector_program": "ideal_cylindrical_jones_basis",
            "vector_method": "analytic_reference",
            "vector_encoder_hardware": "not_current_bench",
            "lab_realizable": False,
            "simulation_only": False,
            "requires_element": "qplate_or_vector_mode_converter",
            "uses_waveplates": False,
            "uses_two_slm": False,
            "uses_shared_director_axis": False,
            "ell": ell,
            "target_core_diameter_um": float(cfg.target.target_core_diameter_m / bt.um),
            "vortex_main_ring_diameter_um": float(2.0 * vbb_vector.predicted_ring_radius(ell, KR) / bt.um),
            **_case_metrics(case, target=mode),
        }
        rows.append(_stamp(row))

    rows.append(_stamp({
        "case_id": f"ideal_hybrid_ell{ell}",
        "path": "vector_theory_atlas",
        "beam_family": "vector",
        "model_level": "ideal_target",
        "generation_method": "qplate_or_vector_converter",
        "vector_mode": "hybrid",
        "vector_model": "ideal_jones_target",
        "vector_program": "hybrid_reference_placeholder",
        "vector_method": "analytic_reference",
        "vector_encoder_hardware": "not_current_bench",
        "lab_realizable": False,
        "simulation_only": False,
        "requires_element": "independent_polarisation_axis_modulation",
        "uses_waveplates": False,
        "uses_two_slm": False,
        "uses_shared_director_axis": False,
        "ell": ell,
        "target_core_diameter_um": float(cfg.target.target_core_diameter_m / bt.um),
        "vortex_main_ring_diameter_um": float(2.0 * vbb_vector.predicted_ring_radius(ell, KR) / bt.um),
    }))

atlas = vector_schema.ordered_vector_frame(rows)
atlas.to_csv(out_csv / "vector_beam_theory_atlas.csv", index=False)
pd.DataFrame(scalar_rows).to_csv(out_csv / "vector_atlas_scalar_sas_summary.csv", index=False)
pd.DataFrame([row for row in rows if row["vector_mode"] in {"radial", "azimuthal", "hybrid"}]).to_csv(
    out_csv / "vector_atlas_jones_summary.csv", index=False
)

# Compatibility copies for older publication export paths.
pd.DataFrame(scalar_rows).to_csv(compat_csv / "vector_atlas_scalar_sas_summary.csv", index=False)
pd.DataFrame([row for row in rows if row["vector_mode"] in {"radial", "azimuthal", "hybrid"}]).to_csv(
    compat_csv / "vector_atlas_jones_summary.csv", index=False
)
atlas
"""


NOTEBOOK_01_FIGURES = """\
fig = vbb_vector.plot_analyzer_family_grid(vector_cases[:4])
vbb_style.save_figure(
    fig,
    out_fig / "vector_beam_theory_atlas_analyzer_panels.png",
    "Ideal radial and azimuthal vector Bessel reference targets. These are mathematical Jones targets; under the current same-axis two-SLM bench they are not claimed as current lab generated beams.",
    metadata={"stage": "vector", "figure": "vector_beam_theory_atlas_analyzer_panels"},
)
plt.close(fig)

fig = vbb_vector.plot_polarization_quiver(vector_cases[0], step=12)
vbb_style.save_figure(
    fig,
    out_fig / "vector_beam_theory_atlas_quiver.png",
    "Polarisation quiver for an ideal radial vector target. Hardware status is future_hardware_required under the current bench assumptions.",
    metadata={"stage": "vector", "figure": "vector_beam_theory_atlas_quiver"},
)
plt.close(fig)
"""


NOTEBOOK_02_HELPERS = """\
def _stamp(row):
    vector_schema.annotate_vector_row(row, run_id=RUN_ID, qa_status="exploratory")
    return row

def _scalar_metrics():
    small_grid = replace(cfg.grid, ideal_N=160, N=160, crop_pixels=96, axial_points=7, coarse_scan_points=7)
    run_cfg = replace(cfg, grid=small_grid, target=replace(cfg.target, ell=1))
    z_values = np.linspace(0.0, 100.0 * bt.um, 7)
    result = bt.run_case(run_cfg, preset=PRESET, path="ideal", case_id="vector_scalar_reference_ell1", z_values_m=z_values)
    return result["metrics"]

scalar_metrics = _scalar_metrics()

def _row_from_case(case_id, case, *, vector_mode, vector_model, model_level, generation_method, lab_realizable, simulation_only, requires_element, vector_method, vector_encoder_hardware, uses_waveplates, uses_two_slm, uses_shared_director_axis, scalar_reference_case_id="vector_scalar_reference_ell1"):
    row = {
        "case_id": case_id,
        "preset": PRESET,
        "path": "vector_ideal_vs_lab_case1",
        "beam_family": "vector",
        "model_level": model_level,
        "generation_method": generation_method,
        "vector_mode": vector_mode,
        "vector_model": vector_model,
        "vector_program": case.get("target", case.get("mode", vector_mode)),
        "vector_method": vector_method,
        "vector_encoder_hardware": vector_encoder_hardware,
        "lab_realizable": lab_realizable,
        "simulation_only": simulation_only,
        "requires_element": requires_element,
        "uses_waveplates": uses_waveplates,
        "uses_two_slm": uses_two_slm,
        "uses_shared_director_axis": uses_shared_director_axis,
        "encoded_power_fraction": case.get("encoded_power_fraction", np.nan),
        "scalar_reference_case_id": scalar_reference_case_id,
        "ell": int(case.get("ell", 1)),
        "target_core_diameter_um": float(cfg.target.target_core_diameter_m / bt.um),
        "vortex_main_ring_diameter_um": float(2.0 * vbb_vector.predicted_ring_radius(int(case.get("ell", 1)), KR) / bt.um),
        "canonical_zone_um": scalar_metrics.get("canonical_zone_um", scalar_metrics.get("bessel_zone_um")),
        "strict_bessel_region_um": scalar_metrics.get("strict_bessel_region_um", scalar_metrics.get("bessel_region_um")),
        "propagation_power_drift_fraction": scalar_metrics.get("propagation_power_drift_fraction", np.nan),
        "propagation_power_label": scalar_metrics.get("propagation_power_label", "unknown"),
        "hardware_note": case.get("hardware_note", ""),
    }
    if case.get("field") is not None:
        row["total_power_au"] = float(np.sum(case["total_intensity"]) * float(case["grid"]["dx"]) ** 2)
    return _stamp(row)
"""


NOTEBOOK_02_RUN = """\
ideal_radial = vbb_vector.build_analytic_vector_mode(grid, ell=1, kr_m_inv=KR, waist_m=WAIST, mode="radial")
ideal_azimuthal = vbb_vector.build_analytic_vector_mode(grid, ell=1, kr_m_inv=KR, waist_m=WAIST, mode="azimuthal")
case1 = vbb_vector.build_actual_lab_vector_case(
    grid,
    ell=1,
    kr_m_inv=KR,
    waist_m=WAIST,
    target="achievable_sop",
    method="B",
    carrier_lpmm=2.5,
)
paper_replica = vbb_vector.build_analytic_vector_mode(grid, ell=1, kr_m_inv=KR, waist_m=WAIST, mode="radial")

rows = [
    _stamp({
        "case_id": "vector_scalar_reference_ell1",
        "preset": PRESET,
        "path": "vector_ideal_vs_lab_case1",
        "beam_family": "scalar_reference",
        "model_level": "scalar_reference",
        "generation_method": "scalar_reference",
        "vector_mode": "scalar_reference",
        "vector_model": "scalar_sas_with_jones_overlay",
        "vector_program": "scalar_reference",
        "vector_method": "not_applicable",
        "vector_encoder_hardware": "scalar_holographic_reference",
        "lab_realizable": True,
        "simulation_only": False,
        "requires_element": "none",
        "uses_waveplates": False,
        "uses_two_slm": False,
        "uses_shared_director_axis": False,
        "ell": 1,
        "target_core_diameter_um": float(cfg.target.target_core_diameter_m / bt.um),
        "vortex_main_ring_diameter_um": float(2.0 * vbb_vector.predicted_ring_radius(1, KR) / bt.um),
        "canonical_zone_um": scalar_metrics.get("canonical_zone_um", scalar_metrics.get("bessel_zone_um")),
        "strict_bessel_region_um": scalar_metrics.get("strict_bessel_region_um", scalar_metrics.get("bessel_region_um")),
        "propagation_power_drift_fraction": scalar_metrics.get("propagation_power_drift_fraction", np.nan),
        "propagation_power_label": scalar_metrics.get("propagation_power_label", "unknown"),
    }),
    _row_from_case(
        "ideal_radial_target_ell1",
        ideal_radial,
        vector_mode="radial",
        vector_model="ideal_jones_target",
        model_level="ideal_target",
        generation_method="qplate_or_vector_converter",
        lab_realizable=False,
        simulation_only=False,
        requires_element="qplate_or_vector_mode_converter",
        vector_method="analytic_reference",
        vector_encoder_hardware="not_current_bench",
        uses_waveplates=False,
        uses_two_slm=False,
        uses_shared_director_axis=False,
    ),
    _row_from_case(
        "ideal_azimuthal_target_ell1",
        ideal_azimuthal,
        vector_mode="azimuthal",
        vector_model="ideal_jones_target",
        model_level="ideal_target",
        generation_method="qplate_or_vector_converter",
        lab_realizable=False,
        simulation_only=False,
        requires_element="qplate_or_vector_mode_converter",
        vector_method="analytic_reference",
        vector_encoder_hardware="not_current_bench",
        uses_waveplates=False,
        uses_two_slm=False,
        uses_shared_director_axis=False,
    ),
    _row_from_case(
        "current_lab_case1_sop_method_b",
        case1,
        vector_mode="sop_encoded_case1",
        vector_model="current_lab_case1_sop_encoded",
        model_level="current_lab_approximation",
        generation_method="two_slm_same_axis_sop",
        lab_realizable=True,
        simulation_only=False,
        requires_element="none",
        vector_method="method_B_complex_amplitude_proxy",
        vector_encoder_hardware="case1_same_axis_no_waveplates",
        uses_waveplates=False,
        uses_two_slm=True,
        uses_shared_director_axis=True,
    ),
    _row_from_case(
        "paper_replica_radial_diagnostic_ell1",
        paper_replica,
        vector_mode="paper_replica",
        vector_model="paper_replica_baliyan_nishchal",
        model_level="paper_replica",
        generation_method="paper_replica_simulation",
        lab_realizable=False,
        simulation_only=True,
        requires_element="waveplate_chain",
        vector_method="paper_jones_chain",
        vector_encoder_hardware="paper_qwp_hwp_chain",
        uses_waveplates=True,
        uses_two_slm=True,
        uses_shared_director_axis=False,
    ),
]

summary = vector_schema.ordered_vector_frame(rows)
summary.to_csv(out_csv / "vector_ideal_vs_lab_case1_summary.csv", index=False)

compat_ladder = summary.copy()
compat_ladder.to_csv(compat_csv / "stage6_fidelity_ladder_summary.csv", index=False)
compat_ladder.to_csv(out_csv / "stage6_fidelity_ladder_summary.csv", index=False)
summary[summary["case_id"] == "current_lab_case1_sop_method_b"].to_csv(
    compat_csv / "stage6_slm_encoded_vector_summary.csv", index=False
)
summary[summary["case_id"] == "current_lab_case1_sop_method_b"].to_csv(
    out_csv / "stage6_slm_encoded_vector_summary.csv", index=False
)
summary[summary["case_id"] == "paper_replica_radial_diagnostic_ell1"].to_csv(
    compat_csv / "stage6_paper_replica_vector_summary.csv", index=False
)
summary[summary["case_id"] == "paper_replica_radial_diagnostic_ell1"].to_csv(
    out_csv / "stage6_paper_replica_vector_summary.csv", index=False
)

baseline = summary.loc[summary["case_id"] == "vector_scalar_reference_ell1"].iloc[0]
delta_rows = []
for _, row in summary.iterrows():
    if row["case_id"] == "vector_scalar_reference_ell1":
        continue
    delta_rows.append(_stamp({
        "case_id": row["case_id"],
        "preset": PRESET,
        "path": "vector_ideal_vs_lab_case1",
        "beam_family": row["beam_family"],
        "model_level": "diagnostic",
        "generation_method": row["generation_method"],
        "vector_mode": row["vector_mode"],
        "vector_model": "diagnostic_only",
        "vector_program": row["vector_program"],
        "vector_method": row["vector_method"],
        "vector_encoder_hardware": row["vector_encoder_hardware"],
        "lab_realizable": bool(row["lab_realizable"]),
        "simulation_only": bool(row["simulation_only"]),
        "requires_element": row["requires_element"],
        "uses_waveplates": bool(row["uses_waveplates"]),
        "uses_two_slm": bool(row["uses_two_slm"]),
        "uses_shared_director_axis": bool(row["uses_shared_director_axis"]),
        "encoded_power_fraction": row.get("encoded_power_fraction", np.nan),
        "scalar_reference_case_id": "vector_scalar_reference_ell1",
        "ell": int(row["ell"]),
        "canonical_zone_um": row["canonical_zone_um"],
        "strict_bessel_region_um": row["strict_bessel_region_um"],
        "canonical_zone_delta_um": float(row["canonical_zone_um"] - baseline["canonical_zone_um"]),
        "strict_region_delta_um": float(row["strict_bessel_region_um"] - baseline["strict_bessel_region_um"]),
    }))
delta_table = vector_schema.ordered_vector_frame(delta_rows)
delta_table.to_csv(out_csv / "stage6_fidelity_delta_table.csv", index=False)
delta_table.to_csv(compat_csv / "stage6_fidelity_delta_table.csv", index=False)
summary
"""


NOTEBOOK_02_FIGURES = """\
fig = vbb_vector.plot_total_and_analyzer_panel(case1["field"], grid, title="Current lab Case 1 SOP-encoded approximation")
vbb_style.save_figure(
    fig,
    out_fig / "vector_current_lab_case1_sop_panel.png",
    "Current laboratory Case 1: limited SOP-encoded approximation using two phase-only SLMs with shared director axis and no waveplates. This is not a claim of true radial or azimuthal vector-beam generation.",
    metadata={"stage": "vector", "figure": "vector_current_lab_case1_sop_panel"},
)
plt.close(fig)

fig = vbb_vector.plot_total_and_analyzer_panel(paper_replica["field"], grid, title="Paper-replica radial diagnostic")
vbb_style.save_figure(
    fig,
    out_fig / "vector_paper_replica_radial_panel.png",
    "Paper-replica Baliyan-Nishchal style radial diagnostic. This route is labelled simulation_only under the current bench assumptions because it requires a waveplate chain not present in Case 1.",
    metadata={"stage": "vector", "figure": "vector_paper_replica_radial_panel"},
)
plt.close(fig)
"""


NOTEBOOK_03_RUN = """\
def _stamp(row):
    vector_schema.annotate_vector_row(row, run_id=RUN_ID, qa_status="route_catalogued")
    return row

routes = [
    {
        "case_id": "current_lab_case1_sop",
        "path": "vector_hardware_routes",
        "beam_family": "vector",
        "model_level": "current_lab_approximation",
        "generation_method": "two_slm_same_axis_sop",
        "vector_mode": "sop_encoded_case1",
        "vector_model": "current_lab_case1_sop_encoded",
        "vector_program": "H_shaped_plus_V_reference",
        "vector_method": "method_A_or_B",
        "vector_encoder_hardware": "case1_same_axis_no_waveplates",
        "lab_realizable": True,
        "simulation_only": False,
        "requires_element": "none",
        "uses_waveplates": False,
        "uses_two_slm": True,
        "uses_shared_director_axis": True,
        "route_note": "Limited SOP-encoded approximation only; not true radial/azimuthal vector generation.",
    },
    {
        "case_id": "ideal_radial_current_bench_request",
        "path": "vector_hardware_routes",
        "beam_family": "vector",
        "model_level": "future_hardware_route",
        "generation_method": "qplate_or_vector_converter",
        "vector_mode": "radial",
        "vector_model": "future_true_vector_route",
        "vector_program": "true_radial_target",
        "vector_method": "requires_extra_hardware",
        "vector_encoder_hardware": "not_current_bench",
        "lab_realizable": False,
        "simulation_only": False,
        "requires_element": "qplate_or_vector_mode_converter",
        "uses_waveplates": False,
        "uses_two_slm": False,
        "uses_shared_director_axis": False,
        "route_note": "Ideal radial target is useful, but current Case 1 does not implement it.",
    },
    {
        "case_id": "qplate_or_vector_converter_route",
        "path": "vector_hardware_routes",
        "beam_family": "vector",
        "model_level": "future_hardware_route",
        "generation_method": "qplate_or_vector_converter",
        "vector_mode": "radial",
        "vector_model": "future_true_vector_route",
        "vector_program": "polarisation_converter",
        "vector_method": "future_qplate_or_converter",
        "vector_encoder_hardware": "qplate_or_vector_mode_converter",
        "lab_realizable": False,
        "simulation_only": False,
        "requires_element": "qplate_or_vector_mode_converter",
        "uses_waveplates": False,
        "uses_two_slm": True,
        "uses_shared_director_axis": False,
        "route_note": "Future route for true vector modes if added and aligned.",
    },
    {
        "case_id": "interferometric_combiner_route",
        "path": "vector_hardware_routes",
        "beam_family": "vector",
        "model_level": "future_hardware_route",
        "generation_method": "interferometric_vector_combiner",
        "vector_mode": "hybrid",
        "vector_model": "future_true_vector_route",
        "vector_program": "independent_axes_recombined",
        "vector_method": "periscope_sagnac_or_common_path",
        "vector_encoder_hardware": "interferometric_combiner",
        "lab_realizable": False,
        "simulation_only": False,
        "requires_element": "interferometric_combiner",
        "uses_waveplates": True,
        "uses_two_slm": True,
        "uses_shared_director_axis": False,
        "route_note": "Requires independent polarisation-axis modulation and stable recombination.",
    },
    {
        "case_id": "paper_replica_baliyan_nishchal",
        "path": "vector_hardware_routes",
        "beam_family": "vector",
        "model_level": "paper_replica",
        "generation_method": "paper_replica_simulation",
        "vector_mode": "paper_replica",
        "vector_model": "paper_replica_baliyan_nishchal",
        "vector_program": "paper_jones_chain",
        "vector_method": "diagnostic_benchmark",
        "vector_encoder_hardware": "paper_qwp_hwp_chain",
        "lab_realizable": False,
        "simulation_only": True,
        "requires_element": "waveplate_chain",
        "uses_waveplates": True,
        "uses_two_slm": True,
        "uses_shared_director_axis": False,
        "route_note": "Simulation-only paper-replica diagnostic under current bench assumptions.",
    },
]

summary = vector_schema.ordered_vector_frame(_stamp(dict(row)) for row in routes)
summary.to_csv(out_csv / "vector_hardware_routes_summary.csv", index=False)

readme = PATHS["csv"] / "README.md"
readme.write_text(
    "# CSV output naming\\n\\n"
    "Stage 5 vector notebooks write canonical outputs under `outputs/csv/vector/`.\\n"
    "Compatibility copies for older publication-export names are kept under `outputs/csv/publication_study/` where useful.\\n\\n"
    "| Old name | Canonical Stage 5 name |\\n"
    "| --- | --- |\\n"
    "| `vector_atlas_scalar_sas_summary.csv` | `vector/vector_atlas_scalar_sas_summary.csv` and `vector/vector_beam_theory_atlas.csv` |\\n"
    "| `vector_atlas_jones_summary.csv` | `vector/vector_atlas_jones_summary.csv` and `vector/vector_beam_theory_atlas.csv` |\\n"
    "| `stage6_fidelity_ladder_summary.csv` | `vector/vector_ideal_vs_lab_case1_summary.csv` |\\n"
    "| `stage6_fidelity_delta_table.csv` | `vector/stage6_fidelity_delta_table.csv` compatibility diagnostic |\\n"
    "| `stage6_slm_encoded_vector_summary.csv` | `vector/vector_ideal_vs_lab_case1_summary.csv` filtered to current Case 1 |\\n"
    "| `stage6_paper_replica_vector_summary.csv` | `vector/vector_ideal_vs_lab_case1_summary.csv` filtered to paper-replica diagnostics |\\n",
    encoding="utf-8",
)
summary
"""


def build_notebook_01() -> None:
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        _markdown(
            "# Vector Beam Theory Atlas\n\n"
            "Ideal radial/azimuthal vector Bessel beams are reference Jones targets. "
            "They are not labelled as current-lab outputs under the current two-SLM "
            "same-axis/no-waveplate bench."
        ),
        _code(BOOTSTRAP),
        _code(NOTEBOOK_01_HELPERS),
        _code(NOTEBOOK_01_RUN),
        _code(NOTEBOOK_01_FIGURES),
    ]
    _write("01_vector_beam_theory_atlas.ipynb", nb)


def build_notebook_02() -> None:
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        _markdown(
            "# Vector Ideal Versus Current-Lab Case 1\n\n"
            "The current laboratory vector case is not a claim of full true radial or "
            "azimuthal vector-beam generation. It is a limited current-lab SOP-encoded "
            "approximation unless additional polarisation-conversion or independent-axis "
            "hardware is introduced."
        ),
        _code(BOOTSTRAP),
        _code(NOTEBOOK_02_HELPERS),
        _code(NOTEBOOK_02_RUN),
        _code(NOTEBOOK_02_FIGURES),
    ]
    _write("02_vector_ideal_vs_lab_case1.ipynb", nb)


def build_notebook_03() -> None:
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        _markdown(
            "# Vector Hardware Routes\n\n"
            "Ideal radial/azimuthal vector Bessel beams remain useful reference targets, "
            "but their hardware status is future_hardware_required or simulation_only "
            "under the current bench assumptions."
        ),
        _code(BOOTSTRAP),
        _code(NOTEBOOK_03_RUN),
    ]
    _write("03_vector_hardware_routes.ipynb", nb)


def main() -> None:
    build_notebook_01()
    build_notebook_02()
    build_notebook_03()


if __name__ == "__main__":
    main()
