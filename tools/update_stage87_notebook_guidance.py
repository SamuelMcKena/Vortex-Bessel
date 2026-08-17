"""Insert Stage 8.7 quick-look guidance cells into adjustable notebooks.

The script is intentionally markdown-only.  It does not modify execution logic,
physics parameters, output labels, or publication gates in the target notebooks.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import nbformat
except ImportError:
    print("ERROR: nbformat is required to update notebooks.")
    sys.exit(1)

HERE = Path(__file__).resolve().parent
PUB = HERE.parent

MARKER = "STAGE87: adjustable quicklook guidance"
GUIDANCE = f"""\
## Stage 8.7 Adjustable Quick-Look Guidance

<!-- {MARKER} -->

For fast parameter scouting, use `notebooks/quicklook/00_quick_beam_to_sample_simulator.ipynb`. This notebook remains on its locked stage path: existing execution logic, propagation-power labels, material-proxy caveats, and governance routing are unchanged.

Safe local edits are the explicit config variables already exposed by this notebook, or a copied exploratory run. Keep `fail` and `marginal` labels visible. If a displayed image is visually smoothed, treat that as display interpolation only; rerun balanced/publication sampling before numerical interpretation.
"""

NOTEBOOKS = [
    PUB / "notebooks" / "scalar" / "02_scalar_ideal_vs_lab_diagnostics.ipynb",
    PUB / "notebooks" / "scalar" / "04_scalar_parameter_sweeps.ipynb",
    PUB / "notebooks" / "lab_realism" / "04_objective_pupil_and_first_order_filtering.ipynb",
    PUB / "notebooks" / "lab_realism" / "06_full_source_to_sample_journey.ipynb",
    PUB / "notebooks" / "materials" / "01_material_proxy_fluence_and_thresholds.ipynb",
    PUB / "notebooks" / "advanced" / "02_hexagonal_polygonal_beams.ipynb",
    PUB / "notebooks" / "advanced" / "03_discrete_nfold_beams.ipynb",
]


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def update_notebook(path: Path) -> bool:
    nb = nbformat.read(path, as_version=4)
    if any(MARKER in _cell_source(cell) for cell in nb.cells):
        return False
    nb.cells.insert(1 if nb.cells else 0, nbformat.v4.new_markdown_cell(GUIDANCE))
    nbformat.write(nb, path)
    return True


def main() -> int:
    changed = 0
    for path in NOTEBOOKS:
        if not path.exists():
            print(f"missing: {path}")
            return 1
        did_change = update_notebook(path)
        changed += int(did_change)
        print(f"{'updated' if did_change else 'unchanged'}: {path.relative_to(PUB)}")
    print(f"Stage 8.7 guidance cells inserted: {changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
