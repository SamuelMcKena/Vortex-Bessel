"""Update Phase 8 lab-realism notebooks for native metadata and plane labels.

This script is intentionally narrow: it edits only notebooks under
``notebooks/lab_realism`` and creates the missing objective/pupil geometry
notebook. It does not touch scalar, vector, materials, or advanced notebooks.
"""

from __future__ import annotations

from pathlib import Path

import nbformat

_pub = Path(__file__).resolve().parent.parent
_nb_dir = _pub / "notebooks" / "lab_realism"


def _read(name: str):
    return nbformat.read(_nb_dir / name, as_version=4)


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


BOOTSTRAP_STAGE_C_HOLOGRAPHIC = """\
from dataclasses import replace
from pathlib import Path

import pandas as pd

import bessel_twin_core as bt
from vbb_study import setup_study, vbb_regime, vbb_train_viz
from vbb_study.equations import objective_pupil as objp
from vbb_study.publication import lab_realism as lab_schema

PATHS = setup_study.bootstrap(Path.cwd())
PRESET = "fast"
RUN_ID = PATHS.get("run_id") or None
base = replace(bt.default_config(PRESET), generation_method="holographic")
out_fig = PATHS["figures"] / "stage_c"
out_csv = PATHS["csv"] / "stage_c"
out_fig.mkdir(parents=True, exist_ok=True)
out_csv.mkdir(parents=True, exist_ok=True)

def _objective_fields(cfg):
    return {
        "objective_NA": float(cfg.objective.NA),
        "objective_f_eff_mm": float(cfg.objective.f_eff_m / bt.mm),
        "pupil_radius_mm": float(cfg.objective.pupil_radius_m / bt.mm),
        "pupil_clipped_fraction": objp.gaussian_clipping_power_fraction(
            cfg.laser.beam_radius_on_slm_m,
            cfg.objective.pupil_radius_m,
        ),
    }

def _hardware_status(path_label):
    return "current_lab_realizable" if path_label == "lab" else "simulation_only"
"""


HOLOGRAPHIC_SUMMARY = """\
rows = []
for regime in ("general", "limits"):
    cfg = vbb_regime.config_for_regime(base, regime)
    for label, path_name in [("ideal", "ideal"), ("lab", "realistic")]:
        result = bt.run_case(cfg, preset=PRESET, path=path_name, case_id=f"{regime}_holographic_{label}")
        m = result["metrics"]
        row = {
            "case_id": f"{regime}_holographic_{label}",
            "regime": regime,
            "path": path_name,
            "route_variant": label,
            "canonical_zone_um": m["canonical_zone_um"],
            "strict_bessel_region_um": m["strict_bessel_region_um"],
            "feature_diameter_um": m["feature_diameter_um"],
            "peak_fluence_J_cm2": m["peak_fluence_J_cm2"],
            "side_to_core_peak_ratio": m["side_to_core_peak_ratio"],
            "first_order_selected_fraction": m.get("first_order_selected_fraction"),
            "propagation_power_drift_fraction": m.get("propagation_power_drift_fraction"),
            "propagation_power_label": m.get("propagation_power_label"),
            "validity_valid": result["validity_report"]["valid"],
            **_objective_fields(cfg),
        }
        lab_schema.annotate_lab_realism_row(
            row,
            generation_method="holographic_axicon",
            model_level="lab_realistic" if label == "lab" else "ideal_target",
            hardware_status=_hardware_status(label),
            plane_label="surface_plane",
            coordinate_frame="surface_plane_air_um",
            run_id=RUN_ID,
            preset=PRESET,
            path=path_name,
        )
        rows.append(row)
summary = lab_schema.ordered_lab_realism_frame(rows)
summary.to_csv(out_csv / "holographic_axicon_route_summary.csv", index=False)
summary
"""


HOLOGRAPHIC_SUMMARY_PIVOT = """\
delta = summary.pivot(
    index="regime",
    columns="path",
    values=[
        "canonical_zone_um",
        "feature_diameter_um",
        "peak_fluence_J_cm2",
        "side_to_core_peak_ratio",
    ],
)
delta
"""


HOLOGRAPHIC_CARRIER_SWEEP = """\
# Fourier/filter-plane carrier and first-order stop sweep.
carrier_sweep = vbb_train_viz.holographic_carrier_filter_sweep(base)
carrier_sweep["case_id"] = [
    f"carrier_px{int(row.blaze_period_px)}_filter{float(row.configured_filter_radius_lpmm):.2f}".replace(".", "p")
    for row in carrier_sweep.itertuples()
]
carrier_sweep = lab_schema.with_lab_realism_metadata(
    carrier_sweep,
    generation_method="holographic_axicon",
    model_level="hardware_route",
    hardware_status="current_lab_realizable",
    plane_label="fourier_filter_plane",
    coordinate_frame="fourier_filter_plane_spatial_frequency_lpmm",
    run_id=RUN_ID,
    preset=PRESET,
    path="realistic",
)
carrier_sweep.to_csv(out_csv / "holographic_first_order_filter_sweep.csv", index=False)
vbb_train_viz.plot_holographic_carrier_filter_tradeoff(carrier_sweep, base, output_dir=out_fig)
carrier_sweep.loc[carrier_sweep["is_optimum"]].reset_index(drop=True)
"""


HOLOGRAPHIC_FAIR_COMPARISON = """\
# Fair rerun: use the optimized holographic carrier/filter in the method comparison.
optimum = carrier_sweep.loc[carrier_sweep["is_optimum"]].iloc[0]
optimised_base = replace(
    base,
    slm=replace(
        base.slm,
        blaze_period_px=int(optimum["blaze_period_px"]),
        first_order_filter_radius_lpmm=float(optimum["configured_filter_radius_lpmm"]),
    ),
)
fair_raw = vbb_train_viz.method_comparison_table(
    optimised_base,
    regimes=("general",),
    methods=("holographic", "physical"),
)
rows = []
for row in fair_raw.to_dict("records"):
    method = str(row.get("method", "")).lower()
    lab_schema.annotate_lab_realism_row(
        row,
        generation_method="physical_axicon" if method == "physical" else "holographic_axicon",
        model_level="hardware_route",
        hardware_status="future_hardware_required" if method == "physical" else "current_lab_realizable",
        plane_label="surface_plane",
        coordinate_frame="surface_plane_air_um",
        run_id=RUN_ID,
        preset=PRESET,
        path=str(row.get("path", "realistic")),
    )
    rows.append(row)
fair_comparison = lab_schema.ordered_lab_realism_frame(rows)
fair_comparison.to_csv(out_csv / "holographic_optimised_method_comparison.csv", index=False)
fair_comparison
"""


BOOTSTRAP_STAGE_C_PHYSICAL = BOOTSTRAP_STAGE_C_HOLOGRAPHIC.replace(
    'generation_method="holographic"', 'generation_method="physical"'
).replace(
    'return "current_lab_realizable" if path_label == "lab" else "simulation_only"',
    'return "future_hardware_required" if path_label == "lab" else "simulation_only"',
)


PHYSICAL_SUMMARY = """\
rows = []
for regime in ("general", "limits"):
    cfg = vbb_regime.config_for_regime(base, regime)
    ideal_cfg = replace(cfg, physical_axicon=replace(cfg.physical_axicon, slm2_stroke_levels=None, slm2_conjugate_mode="preserve_vortex"))
    lab_cfg = replace(cfg, physical_axicon=replace(cfg.physical_axicon, slm2_stroke_levels=256, slm2_conjugate_mode="preserve_vortex"))
    for label, run_cfg in [("ideal", ideal_cfg), ("lab", lab_cfg)]:
        result = bt.run_case(run_cfg, preset=PRESET, path="ideal", case_id=f"{regime}_physical_{label}")
        m = result["metrics"]
        meta = result["axicon_metadata"]
        design = bt.compute_design_from_targets(run_cfg.laser, run_cfg.target, run_cfg.material)
        row = {
            "case_id": f"{regime}_physical_{label}",
            "regime": regime,
            "path": "ideal",
            "route_variant": label,
            "physical_axicon_phase": meta.get("physical_axicon_phase"),
            "physical_axicon_pixelated": meta.get("physical_axicon_pixelated"),
            "physical_axicon_base_angle_deg": meta.get("gamma_deg", design.gamma_slm_deg),
            "equivalent_kr_m_inv": result["axicon_result"].k_r,
            "predicted_bessel_length_um": design.target_bessel_length_m / bt.um,
            "canonical_zone_um": m["canonical_zone_um"],
            "strict_bessel_region_um": m["strict_bessel_region_um"],
            "feature_diameter_um": m["feature_diameter_um"],
            "peak_fluence_J_cm2": m["peak_fluence_J_cm2"],
            "side_to_core_peak_ratio": m["side_to_core_peak_ratio"],
            "first_order_selected_fraction": m.get("first_order_selected_fraction"),
            "propagation_power_drift_fraction": m.get("propagation_power_drift_fraction"),
            "propagation_power_label": m.get("propagation_power_label"),
            "slm2_residual_phase_rms_before_rad": meta.get("slm2_residual_phase_rms_before_rad"),
            "slm2_residual_phase_rms_after_rad": meta.get("slm2_residual_phase_rms_after_rad"),
            "validity_valid": result["validity_report"]["valid"],
            **_objective_fields(run_cfg),
        }
        lab_schema.annotate_lab_realism_row(
            row,
            generation_method="physical_axicon",
            model_level="hardware_route",
            hardware_status=_hardware_status(label),
            plane_label="surface_plane",
            coordinate_frame="surface_plane_air_um",
            run_id=RUN_ID,
            preset=PRESET,
            path="ideal",
        )
        rows.append(row)
summary = lab_schema.ordered_lab_realism_frame(rows)
summary.to_csv(out_csv / "physical_axicon_design_summary.csv", index=False)
summary
"""


BOOTSTRAP_STAGE_C_COMPARISON = """\
from pathlib import Path

import pandas as pd

import bessel_twin_core as bt
from vbb_study import setup_study, vbb_train_viz
from vbb_study.publication import lab_realism as lab_schema

PATHS = setup_study.bootstrap(Path.cwd())
PRESET = "fast"
RUN_ID = PATHS.get("run_id") or None
base = bt.default_config(PRESET)
out_fig = PATHS["figures"] / "stage_c"
out_csv = PATHS["csv"] / "stage_c"
out_fig.mkdir(parents=True, exist_ok=True)
out_csv.mkdir(parents=True, exist_ok=True)
"""


COMPARISON_SUMMARY = """\
raw = vbb_train_viz.method_comparison_table(base)
rows = []
for row in raw.to_dict("records"):
    method = str(row.get("method", "")).lower()
    label = str(row.get("variant", row.get("path", "lab"))).lower()
    lab_schema.annotate_lab_realism_row(
        row,
        generation_method="physical_axicon" if method == "physical" else "holographic_axicon",
        model_level="hardware_route",
        hardware_status="future_hardware_required" if method == "physical" else "current_lab_realizable",
        plane_label="surface_plane",
        coordinate_frame="surface_plane_air_um",
        run_id=RUN_ID,
        preset=PRESET,
        path=str(row.get("path", "realistic")),
    )
    rows.append(row)
comparison = lab_schema.ordered_lab_realism_frame(rows)
comparison.to_csv(out_csv / "holographic_physical_method_comparison.csv", index=False)
comparison
"""


BOOTSTRAP_STAGE_D = """\
from dataclasses import replace
from pathlib import Path

import pandas as pd

import bessel_twin_core as bt
from vbb_study import setup_study, vbb_regime, vbb_sample_study
from vbb_study.publication import lab_realism as lab_schema

PATHS = setup_study.bootstrap(Path.cwd())
PRESET = "fast"
RUN_ID = PATHS.get("run_id") or None
out_fig = PATHS["figures"] / "stage_d"
out_csv = PATHS["csv"] / "stage_d"
out_fig.mkdir(parents=True, exist_ok=True)
out_csv.mkdir(parents=True, exist_ok=True)

base0 = bt.default_config(PRESET)
notebook_grid = replace(base0.grid, axial_points=25, coarse_scan_points=17, crop_pixels=160)
base0 = replace(base0, grid=notebook_grid)

def _route_method(method):
    return "physical_axicon" if method == "physical" else "holographic_axicon"

def _route_hardware_status(method, variant):
    if variant == "ideal":
        return "simulation_only"
    return "future_hardware_required" if method == "physical" else "current_lab_realizable"
"""


THROUGH_SAMPLE_FUNCTIONS = """\
def configured_case(method, regime, variant):
    cfg = replace(base0, generation_method=method)
    cfg = vbb_regime.config_for_regime(cfg, regime)
    if method == "physical":
        levels = 256 if variant == "lab" else None
        cfg = replace(cfg, physical_axicon=replace(cfg.physical_axicon, slm2_stroke_levels=levels, slm2_conjugate_mode="preserve_vortex"))
        path = "ideal"
    else:
        path = "realistic" if variant == "lab" else "ideal"
    return cfg, path

def row_for(label, regime, method, variant, beam, uncorrected, corrected):
    cm = corrected.metrics
    um = uncorrected.metrics
    return {
        "case_id": label,
        "regime": regime,
        "route_generation_method": _route_method(method),
        "method": method,
        "variant": variant,
        "path": "realistic" if method == "holographic" and variant == "lab" else "ideal",
        "beam_air_zone_um": beam["metrics"]["canonical_zone_um"],
        "sample_uncorrected_zone_um": um["canonical_zone_um"],
        "sample_corrected_zone_um": cm["canonical_zone_um"],
        "strict_bessel_region_um": cm["strict_bessel_region_um"],
        "sample_corrected_peak_z_um": cm["peak_z_um"],
        "ring_or_core_um": cm["ring_radius_um"] if int(cm["ell"]) else cm["core_radius_um"],
        "peak_fluence_J_cm2": cm["peak_fluence_J_cm2"],
        "side_to_core_peak_ratio": cm["side_to_core_peak_ratio"],
        "medium_n": cm["medium_n"],
        "surface_transmission": cm["surface_transmission"],
        "interface_correction_label_uncorrected": um["interface_correction_label"],
        "interface_correction_label_corrected": cm["interface_correction_label"],
        "interface_correction_implementation": cm["interface_correction_implementation"],
        "corrected_rel_l2_to_no_interface": cm["corrected_rel_l2_to_no_interface"],
        "uncorrected_rel_l2_to_no_interface": cm["uncorrected_rel_l2_to_no_interface"],
        "phase_only_power_relative_error": cm["phase_only_power_relative_error"],
        "spherical_after_waves": cm["interface_spherical_after_waves"],
    }
"""


THROUGH_SAMPLE_RUN = """\
rows = []
correction_rows = []
results = {}
for regime in ("general", "limits"):
    for method in ("holographic", "physical"):
        for variant in ("ideal", "lab"):
            cfg, path = configured_case(method, regime, variant)
            label = f"{regime}_{method}_{variant}"
            beam = bt.run_case(cfg, preset=PRESET, path=path, case_id=f"{label}_beam")
            uncorrected = vbb_sample_study.run_through_sample(beam["surface_field"], cfg, correct_interface=False)
            corrected = vbb_sample_study.run_through_sample(beam["surface_field"], cfg, correct_interface=True)
            rows.append(row_for(label, regime, method, variant, beam, uncorrected, corrected))
            for sample in (uncorrected, corrected):
                sm = sample.metrics
                correction_rows.append({
                    "case_id": f"{label}_{sm['interface_correction_label']}",
                    "regime": regime,
                    "method": method,
                    "variant": variant,
                    "path": path,
                    "route_generation_method": _route_method(method),
                    "canonical_zone_um": sm["canonical_zone_um"],
                    "strict_bessel_region_um": sm["strict_bessel_region_um"],
                    "interface_correction_label": sm["interface_correction_label"],
                    "interface_correction_implementation": sm["interface_correction_implementation"],
                    "corrected_rel_l2_to_no_interface": sm["corrected_rel_l2_to_no_interface"],
                    "uncorrected_rel_l2_to_no_interface": sm["uncorrected_rel_l2_to_no_interface"],
                    "phase_only_power_relative_error": sm["phase_only_power_relative_error"],
                    "medium_n": sm["medium_n"],
                    "surface_transmission": sm["surface_transmission"],
                })
            results[label] = {"beam": beam, "uncorrected": uncorrected, "corrected": corrected}

summary = pd.DataFrame(rows)
summary = lab_schema.with_lab_realism_metadata(
    summary,
    generation_method="interface_corrected_numerical",
    model_level="interface_model",
    hardware_status="diagnostic_only",
    plane_label="in_medium_plane",
    coordinate_frame="sample_plane_um_z_from_surface_in_medium",
    run_id=RUN_ID,
    preset=PRESET,
    path="through_sample",
)
summary.to_csv(out_csv / "through_sample_summary.csv", index=False)

correction_rows_stamped = []
for row in correction_rows:
    label = row["interface_correction_label"]
    lab_schema.annotate_lab_realism_row(
        row,
        generation_method="interface_corrected_numerical" if label == "ideal_numerical_correction" else "interface_uncorrected",
        model_level="interface_model",
        hardware_status="diagnostic_only" if label == "ideal_numerical_correction" else _route_hardware_status(row["method"], row["variant"]),
        plane_label="in_medium_plane",
        coordinate_frame="sample_plane_um_z_from_surface_in_medium",
        run_id=RUN_ID,
        preset=PRESET,
        path=row["path"],
    )
    correction_rows_stamped.append(row)
interface_correction_summary = lab_schema.ordered_lab_realism_frame(correction_rows_stamped)
interface_correction_summary.to_csv(out_csv / "interface_correction_summary.csv", index=False)
corrected_vs_uncorrected_metrics = interface_correction_summary.copy()
corrected_vs_uncorrected_metrics.to_csv(out_csv / "corrected_vs_uncorrected_metrics.csv", index=False)
summary
"""


THROUGH_SAMPLE_PLOTS = """\
for label in ("general_holographic_lab", "general_physical_lab"):
    vbb_sample_study.plot_sample_result_comparison(
        results[label]["uncorrected"],
        results[label]["corrected"],
        out_fig / f"through_sample_{label}.png",
        title=f"Through-sample {label.replace('_', ' ')}",
    )
"""


BOOTSTRAP_STAGE_E = """\
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import bessel_twin_core as bt
from vbb_study import setup_study, vbb_metrics, vbb_regime, vbb_studies, vbb_style
from vbb_study.publication import lab_realism as lab_schema

PATHS = setup_study.bootstrap(Path.cwd())
PRESET = "fast"
RUN_ID = PATHS.get("run_id") or None
out_fig = PATHS["figures"] / "stage_e"
out_csv = PATHS["csv"] / "stage_e"
out_fig.mkdir(parents=True, exist_ok=True)
out_csv.mkdir(parents=True, exist_ok=True)
vbb_style.apply_style()

base0 = bt.default_config(PRESET)
general_grid = replace(base0.grid, axial_points=61, coarse_scan_points=25, crop_pixels=160)
limits_grid = replace(base0.grid, axial_points=151, coarse_scan_points=37, crop_pixels=160)
base0 = replace(base0, grid=general_grid)

def _route_method(method):
    return "physical_axicon" if method == "physical" else "holographic_axicon"

def _route_hardware_status(method, variant):
    if variant == "ideal":
        return "simulation_only"
    return "future_hardware_required" if method == "physical" else "current_lab_realizable"
"""


def update_existing() -> None:
    nb = _read("01_holographic_axicon_route.ipynb")
    nb.cells[0].source = "# Holographic Axicon Beam Study\n\nStage C notebook for the `slm_plane -> fourier_filter_plane -> objective_pupil_plane -> surface_plane` holographic route. The blaze carrier separates diffraction orders; first-order filtering happens in the Fourier/filter plane before objective-pupil clipping and focusing."
    nb.cells[1].source = BOOTSTRAP_STAGE_C_HOLOGRAPHIC
    nb.cells[2].source = HOLOGRAPHIC_SUMMARY
    nb.cells[3].source = HOLOGRAPHIC_SUMMARY_PIVOT
    nb.cells[6].source = HOLOGRAPHIC_CARRIER_SWEEP
    nb.cells[7].source = HOLOGRAPHIC_FAIR_COMPARISON
    _write("01_holographic_axicon_route.ipynb", nb)

    nb = _read("02_physical_axicon_route.ipynb")
    nb.cells[0].source = "# Physical Axicon Beam Study\n\nStage C notebook for the physical axicon route. This route does not use a holographic blaze carrier or first-order selection in the same sense as the SLM hologram route; it has separate aperture, efficiency, alignment, and hardware assumptions."
    nb.cells[1].source = BOOTSTRAP_STAGE_C_PHYSICAL
    nb.cells[2].source = PHYSICAL_SUMMARY
    _write("02_physical_axicon_route.ipynb", nb)

    nb = _read("03_holographic_vs_physical_axicon.ipynb")
    nb.cells[0].source = "# Holographic vs Physical Axicon\n\nStage C route comparison. Holographic and physical axicons may target similar conical wavevectors, but they differ in blaze/order filtering, aperture, efficiency, aberrations, alignment, and hardware status."
    nb.cells[1].source = BOOTSTRAP_STAGE_C_COMPARISON
    nb.cells[2].source = COMPARISON_SUMMARY
    _write("03_holographic_vs_physical_axicon.ipynb", nb)

    nb = _read("05_through_sample_interface.ipynb")
    nb.cells[0].source = "# Through-Sample Interface Study\n\nStage D notebook. The input is an air-side `surface_plane` field. The sample branch uses a planar interface, explicit in-medium coordinates, and labels the correction branch as `ideal_numerical_correction` unless a hardware route implements it."
    nb.cells[1].source = BOOTSTRAP_STAGE_D
    nb.cells[2].source = THROUGH_SAMPLE_FUNCTIONS
    nb.cells[3].source = THROUGH_SAMPLE_RUN
    nb.cells[4].source = THROUGH_SAMPLE_PLOTS
    _write("05_through_sample_interface.ipynb", nb)

    nb = _read("06_full_source_to_sample_journey.ipynb")
    nb.cells[0].source = "# Full Source-to-Sample Journey\n\nStage E notebook. Stitches `slm_plane`, `fourier_filter_plane`, `objective_pupil_plane`, `surface_plane`, and in-medium `sample_plane` propagation. Corrected-interface panels are ideal numerical diagnostics unless hardware correction is explicitly implemented."
    nb.cells[1].source = BOOTSTRAP_STAGE_E
    src2 = nb.cells[2].source
    src2 = src2.replace("'correction': correction,", "'correction': correction, 'interface_correction_label': correction,")
    nb.cells[2].source = src2
    src3 = nb.cells[3].source
    src3 = src3.replace("for c, corr in enumerate(('uncorrected', 'corrected')):", "for c, corr in enumerate(('uncorrected_interface', 'ideal_numerical_correction')):")
    src3 = src3.replace("ax.set_title(f'{variant} | {corr}')", "ax.set_title(f'{variant} | {corr.replace('_', ' ')}')")
    src3 = src3.replace(
        "lab_corr = journeys[('lab', 'corrected')]",
        "lab_corr = journeys[('lab', 'ideal_numerical_correction')]",
    )
    src3 = src3.replace(
        "lab_valid = validities[('lab', 'corrected')]",
        "lab_valid = validities[('lab', 'ideal_numerical_correction')]",
    )
    nb.cells[3].source = src3
    src4 = nb.cells[4].source
    src4 = src4.replace("(('uncorrected', False), ('corrected', True))", "(('uncorrected_interface', False), ('ideal_numerical_correction', True))")
    src4 = src4.replace("out_fig/f'NB_full_journey_{regime}_{method}.png'", "out_fig / f'full_source_to_sample_journey_{regime}_{method}.png'")
    src4 = src4.replace("summary = pd.DataFrame(rows)\nsummary.to_csv(out_csv/'NB_full_journey_summary.csv', index=False)\nsummary", """raw_summary = pd.DataFrame(rows)
stamped_rows = []
for row in raw_summary.to_dict("records"):
    label = row["interface_correction_label"]
    method = row["method"]
    variant = row["variant"]
    lab_schema.annotate_lab_realism_row(
        row,
        generation_method="interface_corrected_numerical" if label == "ideal_numerical_correction" else "interface_uncorrected",
        model_level="interface_model",
        hardware_status="diagnostic_only" if label == "ideal_numerical_correction" else _route_hardware_status(method, variant),
        plane_label="propagation_axis_z",
        coordinate_frame="stitched_air_surface_and_in_medium_z_um",
        run_id=RUN_ID,
        preset=PRESET,
        path="full_source_to_sample",
    )
    row["route_generation_method"] = _route_method(method)
    stamped_rows.append(row)
summary = lab_schema.ordered_lab_realism_frame(stamped_rows)
summary.to_csv(out_csv / "full_source_to_sample_journey_summary.csv", index=False)
summary""")
    nb.cells[4].source = src4
    _write("06_full_source_to_sample_journey.ipynb", nb)


def create_objective_pupil_notebook() -> None:
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        _markdown(
            "# Objective Pupil and First-Order Filtering\n\n"
            "Stage C geometry notebook for the `slm_plane -> fourier_filter_plane -> "
            "objective_pupil_plane -> surface_plane` route. It separates objective/NA "
            "geometry from full propagation so clipping and first-order limits are visible."
        ),
        _code(
            """\
from dataclasses import replace
from pathlib import Path

import pandas as pd

import bessel_twin_core as bt
from vbb_study import setup_study, vbb_regime, vbb_studies
from vbb_study.equations import objective_pupil as objp
from vbb_study.publication import lab_realism as lab_schema

PATHS = setup_study.bootstrap(Path.cwd())
PRESET = "fast"
RUN_ID = PATHS.get("run_id") or None
out_csv = PATHS["csv"] / "stage_c"
out_csv.mkdir(parents=True, exist_ok=True)
base = replace(bt.default_config(PRESET), generation_method="holographic")

def _common(case_id, cfg, plane_label, coordinate_frame, **extra):
    row = {
        "case_id": case_id,
        "objective_NA": float(cfg.objective.NA),
        "objective_f_eff_mm": float(cfg.objective.f_eff_m / bt.mm),
        "pupil_radius_mm": float(cfg.objective.pupil_radius_m / bt.mm),
        **extra,
    }
    lab_schema.annotate_lab_realism_row(
        row,
        generation_method="objective_pupil_limited",
        model_level="hardware_route",
        hardware_status="current_lab_realizable",
        plane_label=plane_label,
        coordinate_frame=coordinate_frame,
        run_id=RUN_ID,
        preset=PRESET,
        path="geometry_only",
    )
    return row
"""
        ),
        _code(
            """\
rows = []
for regime in ("general", "limits"):
    cfg = vbb_regime.config_for_regime(base, regime)
    design = bt.compute_design_from_targets(cfg.laser, cfg.target, cfg.material)
    pupil_radius = objp.pupil_radius_m(cfg.objective.f_eff_m, cfg.objective.NA, cfg.objective.immersion_n)
    fill = objp.gaussian_pupil_fill_fraction(cfg.laser.beam_radius_on_slm_m, pupil_radius)
    rows.append(_common(
        f"{regime}_objective_pupil",
        cfg,
        "objective_pupil_plane",
        "objective_pupil_plane_mm",
        regime=regime,
        pupil_radius_formula="f_eff_m * NA / immersion_n",
        pupil_fill_fraction=fill,
        pupil_clipped_fraction=1.0 - fill,
        pupil_fill_ratio=objp.pupil_fill_ratio(cfg.laser.beam_radius_on_slm_m, pupil_radius),
        fourier_ring_radius_mm=objp.fourier_plane_ring_radius_m(
            design.kr_slm_m_inv,
            cfg.objective.f_eff_m,
            cfg.laser.wavelength_m,
        ) / bt.mm,
    ))
objective_pupil_geometry = lab_schema.ordered_lab_realism_frame(rows)
objective_pupil_geometry.to_csv(out_csv / "objective_pupil_geometry_summary.csv", index=False)
objective_pupil_geometry
"""
        ),
        _code(
            """\
rows = []
for regime in ("general", "limits"):
    cfg = vbb_regime.config_for_regime(base, regime)
    air_cfg = vbb_studies.beam_air_config(cfg)
    design = bt.compute_design_from_targets(air_cfg.laser, air_cfg.target, air_cfg.material)
    grid = bt.make_xy_grid(int(air_cfg.grid.N), float(air_cfg.slm.pixel_pitch_m) * float(air_cfg.grid.device_downsample))
    for blaze_px in (12, 20, 32):
        test_cfg = replace(air_cfg, slm=replace(air_cfg.slm, blaze_period_px=blaze_px))
        geom = bt.first_order_filter_geometry(grid, test_cfg.slm, design)
        status = "current_lab_realizable" if geom["first_order_geometry_valid"] else "diagnostic_only"
        row = _common(
            f"{regime}_blaze{blaze_px}",
            test_cfg,
            "fourier_filter_plane",
            "fourier_filter_plane_spatial_frequency_lpmm",
            regime=regime,
            blaze_period_px=blaze_px,
            **geom,
        )
        row["hardware_status"] = status
        rows.append(row)
first_order_filter_geometry = lab_schema.ordered_lab_realism_frame(rows)
first_order_filter_geometry.to_csv(out_csv / "first_order_filter_geometry_summary.csv", index=False)
first_order_filter_geometry
"""
        ),
        _code(
            """\
rows = []
for beam_radius_mm in (1.0, 2.0, 3.0):
    cfg = replace(base, laser=replace(base.laser, beam_radius_on_slm_m=beam_radius_mm * bt.mm))
    pupil_radius = cfg.objective.pupil_radius_m
    fill = objp.gaussian_pupil_fill_fraction(cfg.laser.beam_radius_on_slm_m, pupil_radius)
    rows.append(_common(
        f"beam_radius_{beam_radius_mm:g}mm",
        cfg,
        "objective_pupil_plane",
        "objective_pupil_plane_mm",
        beam_radius_on_slm_mm=beam_radius_mm,
        pupil_fill_fraction=fill,
        pupil_clipped_fraction=1.0 - fill,
        pupil_fill_ratio=objp.pupil_fill_ratio(cfg.laser.beam_radius_on_slm_m, pupil_radius),
    ))
pupil_clipping_summary = lab_schema.ordered_lab_realism_frame(rows)
pupil_clipping_summary.to_csv(out_csv / "pupil_clipping_summary.csv", index=False)
pupil_clipping_summary
"""
        ),
    ]
    _write("04_objective_pupil_and_first_order_filtering.ipynb", nb)


def main() -> None:
    update_existing()
    create_objective_pupil_notebook()


if __name__ == "__main__":
    main()
