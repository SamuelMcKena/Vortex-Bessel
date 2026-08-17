"""Update scalar notebooks for Phase 7 canonical schema.

This script makes three kinds of changes to the four scalar study notebooks:

1. Renames ``*_publication`` CSV output names to ``*_atlas`` (Task 3 from Phase 7).
2. Inserts a schema-import cell near the top of each notebook.
3. Appends a power-QA diagnostic cell and a canonical-zone vs. strict-region
   distinction cell at the end of each notebook.

Run from the repo root:

    python Publication_Study/tools/update_scalar_notebooks.py

This script is idempotent: running it twice on the same notebook will not
duplicate cells because it checks for marker strings before inserting.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap so vbb_study is importable.
# ---------------------------------------------------------------------------

_here = Path(__file__).resolve().parent
_pub = _here.parent
_root = _pub.parent
for _p in (_root, _pub):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import nbformat
except ImportError:
    print("ERROR: nbformat is required. Install with: pip install nbformat")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Cell source templates
# ---------------------------------------------------------------------------

SCHEMA_IMPORT_MARKER = "# PHASE7: canonical schema import"
SCHEMA_IMPORT_SOURCE = """\
# PHASE7: canonical schema import
# Imports the canonical scalar output schema helpers.
# source_schema_version is embedded in every row written to CSV.
from vbb_study.publication.tables import (
    SCALAR_OUTPUT_SCHEMA_VERSION,
    SCALAR_SUMMARY_COLUMNS,
    annotate_scalar_row,
    ordered_row,
    propagation_power_label as _power_label,
)
print(f"Scalar output schema version: {SCALAR_OUTPUT_SCHEMA_VERSION}")
print(f"Canonical columns: {len(SCALAR_SUMMARY_COLUMNS)} defined in tables.py")
"""

POWER_QA_MARKER = "# PHASE7: propagation power QA"
POWER_QA_SOURCE = """\
# PHASE7: propagation power QA
# Check propagation power drift for all cases found in notebook global scope.
# pass <= 5%, marginal <= 20%, fail > 20%.
# Fast preset is diagnostic/debug only. Publication preset improves drift, but
# does not make every case publication-ready; fail rows stay visible and should
# be treated as numerical diagnostics or rerun with finer grid/sampling/model setup.
import pandas as pd  # noqa: F811 (safe re-import)
_found_qa = False
for _name in [
    "summary",
    "df",
    "shortlist_df",
    "summary_df",
    "comparison_table",
    "preset_comparison",
    "oat",
    "tradeoff",
    "ell_family",
    "device_realism",
    "interface_depth",
]:
    _df = globals().get(_name)
    if _df is not None and isinstance(_df, pd.DataFrame) and "propagation_power_label" in _df.columns:
        _qa_cols = ["preset", "case_id", "propagation_power_drift_fraction", "propagation_power_label"]
        _qa = _df[[c for c in _qa_cols if c in _df.columns]].copy()
        if "preset" not in _qa.columns:
            _qa["preset"] = globals().get("PRESET", "unknown")
        _summary = (
            _qa.groupby("preset", dropna=False)["propagation_power_label"]
            .value_counts()
            .unstack(fill_value=0)
            .reindex(columns=["pass", "marginal", "fail"], fill_value=0)
            .reset_index()
        )
        print(f"Power QA counts by preset for '{_name}':")
        print(_summary.to_string(index=False))
        _pass = int((_qa["propagation_power_label"] == "pass").sum())
        _marginal = int((_qa["propagation_power_label"] == "marginal").sum())
        _fail = int((_qa["propagation_power_label"] == "fail").sum())
        print(f"Power QA for '{_name}': {_pass} pass, {_marginal} marginal, {_fail} fail")
        if _fail > 0:
            print("WARNING: FAIL cases (>20% drift) are NOT suitable for publication figures.")
        if _marginal > 0:
            print("NOTE: MARGINAL cases (5-20% drift) — review before publication.")
        _found_qa = True
        break
if not _found_qa:
    print("NOTE: power QA cell ran — no scalar summary DataFrame found in scope yet.")
    print("      Re-run this cell AFTER the cell that builds 'summary' or 'shortlist_df'.")
"""

ZONE_REGION_MARKER = "# PHASE7: canonical zone vs strict region"
FAST_PRESET_CAUTION_MARKER = "# PHASE7: fast-preset caution"
FAST_PRESET_CAUTION_SOURCE = """\
# PHASE7: fast-preset caution
# This notebook was last executed with the 'fast' grid preset.
# The fast preset is a DIAGNOSTIC / DEBUG speed mode:
#   - small grid (N=512), coarse axial sampling
#   - may show propagation_power_label='fail' (>20% power drift)
#   - NOT suitable as the primary source for publication figures
#
# Publication preset improves propagation-power drift, but does NOT certify
# every case as publication-ready. Preserve propagation_power_label and inspect
# fail rows as numerical diagnostics or rerun with finer grid/sampling/model setup.
# See: tools/regen_scalar_shortlist.py --preset publication
#
# Cases with propagation_power_label='fail' MUST NOT be reported as
# publication-ready or hidden by threshold relaxation.
import os as _os
_rid = _os.environ.get("STRUCTURED_BEAM_RUN_ID", "interactive")
_preset_env = _os.environ.get("STRUCTURED_BEAM_PRESET", "fast")
print(f"[Phase7 caution] run_id={_rid!r}  preset={_preset_env!r}")
print("[Phase7 caution] Fast preset: results are diagnostic quality only.")
print("[Phase7 caution] Publication preset improves drift but may still leave fail cases.")
"""

ZONE_REGION_SOURCE = """\
# PHASE7: canonical zone vs strict region
# These are TWO DIFFERENT metrics. Never plot or report them interchangeably.
#
#   canonical_zone_um       = axial peak FWHM (single observable, broader)
#   strict_bessel_region_um = triple-intersection fabrication region (narrower)
#
# Use strict_bessel_region_um for fabrication planning and heatmaps.
# Use canonical_zone_um for optical diagnostic comparisons.
import pandas as pd  # noqa: F811
_zone_found = False
for _name in ["summary", "df", "shortlist_df", "summary_df", "oat", "tradeoff"]:
    _df = globals().get(_name)
    if _df is not None and isinstance(_df, pd.DataFrame):
        _c, _s = "canonical_zone_um", "strict_bessel_region_um"
        if _c in _df.columns and _s in _df.columns:
            _id = "case_id" if "case_id" in _df.columns else _df.columns[0]
            _z = _df[[_id, _c, _s]].copy()
            _z["zone_margin_um"] = _z[_c] - _z[_s]
            _z["strict_fraction"] = (_z[_s] / _z[_c].replace(0, float("nan"))).round(3)
            print(f"Zone/region comparison for '{_name}' ({len(_z)} rows):")
            print(_z.to_string(index=False))
            print()
            print("  canonical_zone_um       = axial_peak_fwhm  (from bessel_zone_metrics)")
            print("  strict_bessel_region_um = strict_intersection_peak_power_radius (narrower)")
            _zone_found = True
            break
if not _zone_found:
    print("NOTE: zone/region cell — no scalar summary DataFrame with both zone columns found.")
    print("      Re-run after the cell that builds 'summary' or 'shortlist_df'.")
"""

# ---------------------------------------------------------------------------
# CSV name replacements for notebooks
# ---------------------------------------------------------------------------

# Map of (old_source_fragment → new_source_fragment) per notebook stem.
CSV_RENAMES: dict[str, list[tuple[str, str]]] = {
    "02_scalar_ideal_vs_lab_diagnostics": [
        ('"shortlist_realistic_summary.csv"', '"scalar_shortlist_realistic_summary.csv"'),
    ],
    "04_scalar_parameter_sweeps": [
        ('"oat_sensitivity_publication"', '"oat_sensitivity_atlas"'),
        ('"tradeoff_publication"',        '"tradeoff_atlas"'),
        ('"ell_family_publication"',       '"ell_family_atlas"'),
        ('"device_realism_publication"',   '"device_realism_atlas"'),
        ('"interface_depth_publication"',  '"interface_depth_atlas"'),
        ('"sampling_publication"',         '"sampling_atlas"'),
    ],
    "05_scalar_validation_suite": [
        ('"sas_validation_summary.csv"', '"scalar_validation_summary.csv"'),
    ],
}

# ---------------------------------------------------------------------------
# Notebook paths
# ---------------------------------------------------------------------------

NOTEBOOKS: list[Path] = [
    _pub / "notebooks" / "scalar" / "02_scalar_ideal_vs_lab_diagnostics.ipynb",
    _pub / "notebooks" / "scalar" / "03_scalar_robustness_and_self_healing.ipynb",
    _pub / "notebooks" / "scalar" / "04_scalar_parameter_sweeps.ipynb",
    _pub / "notebooks" / "scalar" / "05_scalar_validation_suite.ipynb",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_code_cell(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source=source)


def _cell_has_marker(cell: dict, marker: str) -> bool:
    src = cell.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    return marker in src


def _has_marker_in_notebook(nb, marker: str) -> bool:
    return any(_cell_has_marker(c, marker) for c in nb.get("cells", []))


def _apply_csv_renames(source: str, renames: list[tuple[str, str]]) -> str:
    for old, new in renames:
        # Only replace if the old string is present AND the new string is not
        # already present (prevents double-rename if the script runs twice).
        if old in source and new not in source:
            source = source.replace(old, new)
    return source


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def process_notebook(path: Path) -> None:
    if not path.exists():
        print(f"  SKIP (not found): {path.name}")
        return

    nb = nbformat.read(str(path), as_version=4)
    changed = False

    # ---- 1. Apply CSV renames to all code cells --------------------------
    stem = path.stem
    renames = CSV_RENAMES.get(stem, [])
    if renames:
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            src = cell["source"]
            if isinstance(src, list):
                src = "".join(src)
            new_src = _apply_csv_renames(src, renames)
            if new_src != src:
                cell["source"] = new_src
                changed = True
                print(f"  CSV rename applied in {path.name}")

    # ---- 2. Insert schema-import cell near the top (after cell 2) --------
    if not _has_marker_in_notebook(nb, SCHEMA_IMPORT_MARKER):
        insert_idx = min(2, len(nb["cells"]))
        nb["cells"].insert(insert_idx, _make_code_cell(SCHEMA_IMPORT_SOURCE))
        changed = True
        print(f"  Schema import cell added to {path.name}")

    # ---- 3. Append or update fast-preset caution cell --------------------
    if not _has_marker_in_notebook(nb, FAST_PRESET_CAUTION_MARKER):
        # Insert after the schema import cell (position 3).
        insert_pos = min(3, len(nb["cells"]))
        nb["cells"].insert(insert_pos, _make_code_cell(FAST_PRESET_CAUTION_SOURCE))
        changed = True
        print(f"  Fast-preset caution cell added to {path.name}")
    else:
        for cell in nb["cells"]:
            if cell["cell_type"] == "code" and _cell_has_marker(cell, FAST_PRESET_CAUTION_MARKER):
                current = cell.get("source", "")
                if isinstance(current, list):
                    current = "".join(current)
                if current.strip() != FAST_PRESET_CAUTION_SOURCE.strip():
                    cell["source"] = FAST_PRESET_CAUTION_SOURCE
                    changed = True
                    print(f"  Fast-preset caution cell updated in {path.name}")

    # ---- 5. Append or update power-QA cell --------------------------------
    if not _has_marker_in_notebook(nb, POWER_QA_MARKER):
        nb["cells"].append(_make_code_cell(POWER_QA_SOURCE))
        changed = True
        print(f"  Power QA cell appended to {path.name}")
    else:
        # Update existing cell if source has changed.
        for cell in nb["cells"]:
            if cell["cell_type"] == "code" and _cell_has_marker(cell, POWER_QA_MARKER):
                current = cell.get("source", "")
                if isinstance(current, list):
                    current = "".join(current)
                if current.strip() != POWER_QA_SOURCE.strip():
                    cell["source"] = POWER_QA_SOURCE
                    changed = True
                    print(f"  Power QA cell updated in {path.name}")

    # ---- 6. Append or update zone/region distinction cell ----------------
    if not _has_marker_in_notebook(nb, ZONE_REGION_MARKER):
        nb["cells"].append(_make_code_cell(ZONE_REGION_SOURCE))
        changed = True
        print(f"  Zone/region cell appended to {path.name}")
    else:
        for cell in nb["cells"]:
            if cell["cell_type"] == "code" and _cell_has_marker(cell, ZONE_REGION_MARKER):
                current = cell.get("source", "")
                if isinstance(current, list):
                    current = "".join(current)
                if current.strip() != ZONE_REGION_SOURCE.strip():
                    cell["source"] = ZONE_REGION_SOURCE
                    changed = True
                    print(f"  Zone/region cell updated in {path.name}")

    if changed:
        nbformat.write(nb, str(path))
        print(f"  SAVED: {path.name}")
    else:
        print(f"  No changes needed: {path.name}")


def main() -> None:
    print(f"[update_scalar_notebooks] updating {len(NOTEBOOKS)} notebooks")
    for nb_path in NOTEBOOKS:
        print(f"\n{nb_path.relative_to(_pub)}")
        process_notebook(nb_path)
    print("\n[update_scalar_notebooks] done")


if __name__ == "__main__":
    main()
