"""Add Phase 1R evidence fields to the affected lab-realism notebooks.

This is deliberately narrower than ``update_lab_realism_notebooks.py``. It
updates only the source cells that generate repaired physical-route artifacts,
then clears their saved execution state so a controlled rerun cannot retain
pre-repair output cells.
"""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = ROOT / "notebooks" / "lab_realism"


def _load(name: str):
    return nbformat.read(NOTEBOOK_DIR / name, as_version=4)


def _source_cell(nb, marker: str):
    matches = [cell for cell in nb.cells if cell.cell_type == "code" and marker in cell.source]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one code cell containing {marker!r}, found {len(matches)}")
    return matches[0]


def _replace_once(source: str, old: str, new: str) -> str:
    if new in source:
        return source
    if source.count(old) != 1:
        raise RuntimeError(f"Expected one occurrence of {old!r}, found {source.count(old)}")
    return source.replace(old, new)


def _write(name: str, nb) -> None:
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    nbformat.write(nb, NOTEBOOK_DIR / name)
    print(f"prepared {name}")


def _prepare_physical_route() -> None:
    name = "02_physical_axicon_route.ipynb"
    nb = _load(name)
    bootstrap = _source_cell(nb, "BOOTSTRAP_STAGE_C_PHYSICAL") if any(
        cell.cell_type == "code" and "BOOTSTRAP_STAGE_C_PHYSICAL" in cell.source for cell in nb.cells
    ) else _source_cell(nb, "base = replace(bt.default_config(PRESET), generation_method=\"physical\")")
    bootstrap.source = _replace_once(
        bootstrap.source,
        "from vbb_study import setup_study, vbb_regime, vbb_train_viz",
        "from vbb_study import setup_study, vbb_regime, vbb_train_viz, viz_fields",
    )
    run = _source_cell(nb, 'summary.to_csv(out_csv / "physical_axicon_design_summary.csv"')
    run.source = _replace_once(
        run.source,
        "        design = bt.compute_design_from_targets(run_cfg.laser, run_cfg.target, run_cfg.material)\n",
        "        design = bt.compute_design_from_config(run_cfg)\n"
        "        surface = result[\"surface_field\"]\n"
        "        measured_winding = viz_fields.phase_winding(\n"
        "            surface.Ex, surface.grid, float(design.vortex_main_ring_radius_m)\n"
        "        )\n"
        "        winding_error = abs(measured_winding - int(design.ell))\n"
        "        if winding_error >= 0.1:\n"
        "            raise RuntimeError(\n"
        "                f\"{regime}/{label} physical winding {measured_winding:.6g} \"\n"
        "                f\"does not match requested ell={design.ell}\"\n"
        "            )\n",
    )
    run.source = _replace_once(
        run.source,
        '            "propagation_power_drift_fraction": m.get("propagation_power_drift_fraction"),\n',
        '            "requested_vortex_charge": int(design.ell),\n'
        '            "measured_winding": float(measured_winding),\n'
        '            "winding_error": float(winding_error),\n'
        '            "winding_pass": bool(winding_error < 0.1),\n'
        '            "slm2_conjugate_mode": str(run_cfg.physical_axicon.slm2_conjugate_mode),\n'
        '            "vortex_removal_acknowledged": bool(run_cfg.physical_axicon.allow_vortex_removal),\n'
        '            "propagation_power_drift_fraction": m.get("propagation_power_drift_fraction"),\n',
    )
    run.source = _replace_once(
        run.source,
        '            "propagation_power_label": m.get("propagation_power_label"),\n',
        '            "propagation_power_label": m.get("propagation_power_label"),\n'
        '            "quantitative_metrics_valid": m.get("quantitative_metrics_valid"),\n'
        '            "quantitative_metrics_invalid_reason": m.get("quantitative_metrics_invalid_reason"),\n',
    )
    _write(name, nb)


def _prepare_through_sample() -> None:
    name = "05_through_sample_interface.ipynb"
    nb = _load(name)
    bootstrap = _source_cell(nb, "base0 = bt.default_config(PRESET)")
    bootstrap.source = _replace_once(
        bootstrap.source,
        "from vbb_study import setup_study, vbb_regime, vbb_sample_study",
        "from vbb_study import setup_study, vbb_regime, vbb_sample_study, viz_fields",
    )
    funcs = _source_cell(nb, "def row_for(")
    funcs.source = _replace_once(
        funcs.source,
        "def row_for(label, regime, method, variant, beam, uncorrected, corrected):",
        "def winding_fields(beam, method, cfg):\n"
        "    surface = beam[\"surface_field\"]\n"
        "    design = beam[\"design\"]\n"
        "    measured = viz_fields.phase_winding(\n"
        "        surface.Ex, surface.grid, float(design.vortex_main_ring_radius_m)\n"
        "    )\n"
        "    error = abs(measured - int(design.ell))\n"
        "    if method == \"physical\" and error >= 0.1:\n"
        "        raise RuntimeError(\n"
        "            f\"physical winding {measured:.6g} does not match requested ell={design.ell}\"\n"
        "        )\n"
        "    return {\n"
        "        \"requested_vortex_charge\": int(design.ell),\n"
        "        \"measured_winding\": float(measured),\n"
        "        \"winding_error\": float(error),\n"
        "        \"winding_pass\": bool(error < 0.1),\n"
        "        \"slm2_conjugate_mode\": (\n"
        "            str(cfg.physical_axicon.slm2_conjugate_mode) if method == \"physical\" else \"not_applicable\"\n"
        "        ),\n"
        "        \"vortex_removal_acknowledged\": (\n"
        "            bool(cfg.physical_axicon.allow_vortex_removal) if method == \"physical\" else False\n"
        "        ),\n"
        "        \"propagation_power_drift_fraction\": beam[\"metrics\"].get(\"propagation_power_drift_fraction\"),\n"
        "        \"propagation_power_label\": beam[\"metrics\"].get(\"propagation_power_label\"),\n"
        "        \"quantitative_metrics_valid\": beam[\"metrics\"].get(\"quantitative_metrics_valid\"),\n"
        "        \"quantitative_metrics_invalid_reason\": beam[\"metrics\"].get(\"quantitative_metrics_invalid_reason\"),\n"
        "    }\n\n"
        "def row_for(label, regime, method, variant, beam, uncorrected, corrected, winding):",
    )
    funcs.source = _replace_once(
        funcs.source,
        '        "spherical_after_waves": cm["interface_spherical_after_waves"],\n',
        '        "spherical_after_waves": cm["interface_spherical_after_waves"],\n'
        '        **winding,\n',
    )
    run = _source_cell(nb, "correction_rows = []")
    run.source = _replace_once(
        run.source,
        '            rows.append(row_for(label, regime, method, variant, beam, uncorrected, corrected))\n',
        '            winding = winding_fields(beam, method, cfg)\n'
        '            rows.append(row_for(label, regime, method, variant, beam, uncorrected, corrected, winding))\n',
    )
    run.source = _replace_once(
        run.source,
        '                    "surface_transmission": sm["surface_transmission"],\n',
        '                    "surface_transmission": sm["surface_transmission"],\n'
        '                    **winding,\n',
    )
    _write(name, nb)


def _prepare_full_journey() -> None:
    name = "06_full_source_to_sample_journey.ipynb"
    nb = _load(name)
    bootstrap = _source_cell(nb, "general_grid = replace(base0.grid")
    bootstrap.source = _replace_once(
        bootstrap.source,
        "from vbb_study import setup_study, vbb_metrics, vbb_regime, vbb_studies, vbb_style",
        "from vbb_study import setup_study, vbb_metrics, vbb_regime, vbb_studies, vbb_style, viz_fields",
    )
    funcs = _source_cell(nb, "def journey_row(")
    funcs.source = _replace_once(
        funcs.source,
        "def journey_row(label, regime, method, variant, correction, journey, validity):",
        "def journey_winding_fields(journey, method, cfg):\n"
        "    air = journey.air_result\n"
        "    surface = air[\"surface_field\"]\n"
        "    design = air[\"design\"]\n"
        "    measured = viz_fields.phase_winding(\n"
        "        surface.Ex, surface.grid, float(design.vortex_main_ring_radius_m)\n"
        "    )\n"
        "    error = abs(measured - int(design.ell))\n"
        "    if method == \"physical\" and error >= 0.1:\n"
        "        raise RuntimeError(\n"
        "            f\"physical winding {measured:.6g} does not match requested ell={design.ell}\"\n"
        "        )\n"
        "    return {\n"
        "        \"requested_vortex_charge\": int(design.ell),\n"
        "        \"measured_winding\": float(measured),\n"
        "        \"winding_error\": float(error),\n"
        "        \"winding_pass\": bool(error < 0.1),\n"
        "        \"slm2_conjugate_mode\": (\n"
        "            str(cfg.physical_axicon.slm2_conjugate_mode) if method == \"physical\" else \"not_applicable\"\n"
        "        ),\n"
        "        \"vortex_removal_acknowledged\": (\n"
        "            bool(cfg.physical_axicon.allow_vortex_removal) if method == \"physical\" else False\n"
        "        ),\n"
        "        \"propagation_power_drift_fraction\": air[\"volume\"].get(\"propagation_power_drift_fraction\"),\n"
        "        \"propagation_power_label\": air[\"volume\"].get(\"propagation_power_label\"),\n"
        "        \"quantitative_metrics_valid\": air[\"volume\"].get(\"quantitative_metrics_valid\"),\n"
        "        \"quantitative_metrics_invalid_reason\": air[\"volume\"].get(\"quantitative_metrics_invalid_reason\"),\n"
        "    }\n\n"
        "def journey_row(label, regime, method, variant, correction, journey, validity, winding):",
    )
    funcs.source = _replace_once(
        funcs.source,
        "        'interface_spherical_after_waves': sm['interface_spherical_after_waves'],\n",
        "        'interface_spherical_after_waves': sm['interface_spherical_after_waves'],\n"
        "        **winding,\n",
    )
    run = _source_cell(nb, "all_results = {}")
    run.source = _replace_once(
        run.source,
        "                rows.append(journey_row(label, regime, method, variant, correction, journey, validity))\n",
        "                winding = journey_winding_fields(journey, method, cfg)\n"
        "                rows.append(journey_row(label, regime, method, variant, correction, journey, validity, winding))\n",
    )
    _write(name, nb)


def main() -> None:
    _prepare_physical_route()
    _prepare_through_sample()
    _prepare_full_journey()


if __name__ == "__main__":
    main()
