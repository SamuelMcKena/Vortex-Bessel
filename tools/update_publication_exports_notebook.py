"""Update the publication_exports notebook for Phase 7 canonical naming.

Changes:
1. Insert a schema-metadata cell after the bootstrap cell.
2. Redirect CSV outputs to outputs/csv/publication_exports/ (not publication_study/).
3. Rename old CSV output filenames to new canonical study names.
4. Add a cell documenting source study CSVs and schema version.
5. Ensure (mkdir) for the publication_exports/ output subdirectory.

Run:
    python Publication_Study/tools/update_publication_exports_notebook.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_pub = _here.parent
_root = _pub.parent
for _p in (_root, _pub):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import nbformat
except ImportError:
    print("ERROR: nbformat required.  pip install nbformat")
    sys.exit(1)

NB_PATH = _pub / "notebooks" / "publication_exports" / "03_report_export.ipynb"

# ---------------------------------------------------------------------------
# Markers and cell sources
# ---------------------------------------------------------------------------

SCHEMA_HEADER_MARKER = "# PHASE7-PUBEXP: schema and source documentation"
SCHEMA_HEADER_SOURCE = """\
# PHASE7-PUBEXP: schema and source documentation
# This notebook is the PAPER-FACING export layer.
# It regenerates outputs from the scalar engine.
# Paper-facing CSVs are written to:
#   outputs/csv/publication_exports/
# They are DISTINCT from the study-wide CSVs in:
#   outputs/csv/publication_study/
#
# Canonical schema version (source_schema_version in every CSV):
from vbb_study.publication.tables import (
    SCALAR_OUTPUT_SCHEMA_VERSION,
    annotate_scalar_row,
    propagation_power_label as _power_label,
)
import os as _os
_run_id = _os.environ.get("STRUCTURED_BEAM_RUN_ID", "")
print(f"Schema version:  {SCALAR_OUTPUT_SCHEMA_VERSION}")
print(f"Run ID:          {_run_id or '(interactive)'}")
print()
print("Source study CSVs (study-generated inputs, if using pre-computed results):")
print("  outputs/csv/publication_study/scalar_shortlist_realistic_summary.csv")
print("  outputs/csv/publication_study/scalar_shortlist_ideal_vs_lab_comparison.csv")
print()
print("Paper-facing output location: outputs/csv/publication_exports/")
"""

MKDIR_MARKER = "# PHASE7-PUBEXP: create publication_exports output directory"
MKDIR_SOURCE = """\
# PHASE7-PUBEXP: create publication_exports output directory
# Write paper-facing CSVs here, not into publication_study/ (the study layer).
PUB_EXP_CSV = PATHS["csv"] / "publication_exports"
PUB_EXP_CSV.mkdir(parents=True, exist_ok=True)
print(f"Publication export CSV dir: {PUB_EXP_CSV}")
"""

# ---------------------------------------------------------------------------
# CSV name renames
# ---------------------------------------------------------------------------

CSV_RENAMES = [
    # Old name (in publication_study/) → New name (in publication_exports/ via PUB_EXP_CSV)
    # We change the save path variable AND the filename.
    (
        'PUB_OUT["csv"] / "shortlist_realistic_summary.csv"',
        'PUB_EXP_CSV / "scalar_shortlist_realistic_summary.csv"',
    ),
    (
        'PUB_OUT["csv"] / "shortlist_extended_comparison.csv"',
        'PUB_EXP_CSV / "scalar_shortlist_ideal_vs_lab_comparison.csv"',
    ),
    (
        'PUB_OUT["csv"] / "shortlist_artifact_manifest.csv"',
        'PUB_EXP_CSV / "scalar_shortlist_artifact_manifest.csv"',
    ),
    (
        'PUB_OUT["csv"] / "vortex_charge_sweep.csv"',
        'PUB_EXP_CSV / "scalar_vortex_charge_sweep.csv"',
    ),
    (
        'PUB_OUT["csv"] / "vortex_charge_length_sensitivity.csv"',
        'PUB_EXP_CSV / "scalar_vortex_charge_length_sensitivity.csv"',
    ),
    (
        'PUB_OUT["csv"] / "ell_family_comparison.csv"',
        'PUB_EXP_CSV / "scalar_ell_family_comparison.csv"',
    ),
    (
        'PUB_OUT["csv"] / "oat_sensitivity_raw.csv"',
        'PUB_EXP_CSV / "scalar_oat_sensitivity_raw.csv"',
    ),
    (
        'PUB_OUT["csv"] / "oat_sensitivity_ranked.csv"',
        'PUB_EXP_CSV / "scalar_oat_sensitivity_ranked.csv"',
    ),
    (
        'PUB_OUT["csv"] / "calibration_template.csv"',
        'PUB_EXP_CSV / "scalar_calibration_template.csv"',
    ),
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


def _has_marker(nb, marker: str) -> bool:
    return any(_cell_has_marker(c, marker) for c in nb.get("cells", []))


def _apply_renames(src: str) -> tuple[str, bool]:
    changed = False
    for old, new in CSV_RENAMES:
        if old in src and new not in src:
            src = src.replace(old, new)
            changed = True
    return src, changed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def process() -> None:
    if not NB_PATH.exists():
        print(f"ERROR: notebook not found at {NB_PATH}")
        sys.exit(1)

    nb = nbformat.read(str(NB_PATH), as_version=4)
    changed = False

    # 1. CSV renames in all code cells.
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        new_src, cell_changed = _apply_renames(src)
        if cell_changed:
            cell["source"] = new_src
            changed = True
            print("  CSV rename applied")

    # 2. Insert schema header after cell 1 (bootstrap).
    if not _has_marker(nb, SCHEMA_HEADER_MARKER):
        nb["cells"].insert(2, _make_code_cell(SCHEMA_HEADER_SOURCE))
        changed = True
        print("  Schema header cell inserted")
    else:
        for cell in nb["cells"]:
            if cell["cell_type"] == "code" and _cell_has_marker(cell, SCHEMA_HEADER_MARKER):
                cur = cell.get("source", "")
                if isinstance(cur, list):
                    cur = "".join(cur)
                if cur.strip() != SCHEMA_HEADER_SOURCE.strip():
                    cell["source"] = SCHEMA_HEADER_SOURCE
                    changed = True
                    print("  Schema header cell updated")

    # 3. Insert mkdir cell after cell 3 (just after the schema header).
    if not _has_marker(nb, MKDIR_MARKER):
        nb["cells"].insert(4, _make_code_cell(MKDIR_SOURCE))
        changed = True
        print("  mkdir cell inserted")
    else:
        for cell in nb["cells"]:
            if cell["cell_type"] == "code" and _cell_has_marker(cell, MKDIR_MARKER):
                cur = cell.get("source", "")
                if isinstance(cur, list):
                    cur = "".join(cur)
                if cur.strip() != MKDIR_SOURCE.strip():
                    cell["source"] = MKDIR_SOURCE
                    changed = True
                    print("  mkdir cell updated")

    if changed:
        nbformat.write(nb, str(NB_PATH))
        print(f"  SAVED: {NB_PATH.relative_to(_pub)}")
    else:
        print("  No changes needed.")


if __name__ == "__main__":
    print(f"[update_publication_exports] {NB_PATH.name}")
    process()
    print("[update_publication_exports] done")
