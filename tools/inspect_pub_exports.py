"""Quick inspection of the publication_exports notebook structure."""
import json, sys
from pathlib import Path

nb_path = Path(__file__).parent.parent / "notebooks/publication_exports/03_report_export.ipynb"
nb = json.load(open(nb_path, encoding="utf-8"))
print(f"Total cells: {len(nb['cells'])}")

keywords = ["shortlist_realistic", "shortlist_extended", "sweep", "PUB_OUT", "bootstrap", "csv_dir", "read_csv"]
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if any(k in src for k in keywords):
        print(f"\n--- Cell {i} ({cell['cell_type']}) ---")
        print(src[:500])
