"""Repair legacy parameter-sweep atlas CSVs missing canonical metadata.

Notebook 04 now writes run_id, generated_at_utc, and source_schema_version
natively.  This script remains as a repair utility for older generated CSVs.

Run:
    python Publication_Study/tools/stamp_atlas_csvs.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_pub = Path(__file__).resolve().parent.parent
_root = _pub.parent
for _p in (_root, _pub):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd
from vbb_study.publication.tables import SCALAR_OUTPUT_SCHEMA_VERSION

CSV_DIR = _pub / "outputs" / "csv" / "publication_study"

# Atlas sweep CSVs produced by notebook 04 that need schema stamping.
ATLAS_CSVS = [
    "03_sweep_tradeoff_atlas.csv",
    "03_sweep_sampling_atlas.csv",
    "03_sweep_device_realism_atlas.csv",
    "03_sweep_ell_family_atlas.csv",
    "03_sweep_oat_sensitivity_atlas.csv",
    "03_sweep_interface_depth_atlas.csv",
]


def stamp(path: Path, run_id: str, generated_at: str) -> bool:
    if not path.exists():
        print(f"  SKIP (not found): {path.name}")
        return False
    df = pd.read_csv(path)
    meta = {
        "run_id": run_id,
        "generated_at_utc": generated_at,
        "source_schema_version": SCALAR_OUTPUT_SCHEMA_VERSION,
    }
    needs_stamp = any(
        col not in df.columns or df[col].isna().any() or (df[col] == "").any()
        for col in meta
    )
    if not needs_stamp:
        print(f"  OK (already stamped): {path.name}")
        return False
    # Build new column order: meta first, then remaining columns.
    for col, val in meta.items():
        if col not in df.columns:
            df[col] = val
        else:
            df[col] = df[col].fillna(val).replace("", val)
    new_cols = list(meta.keys()) + [c for c in df.columns if c not in meta]
    df = df[new_cols].copy()
    df.to_csv(path, index=False)
    print(f"  STAMPED: {path.name}  ({len(df)} rows)")
    return True


def main() -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    generated_at = datetime.now(timezone.utc).isoformat()
    print(f"[stamp_atlas_csvs] run_id={run_id}")
    for name in ATLAS_CSVS:
        stamp(CSV_DIR / name, run_id, generated_at)
    print("[stamp_atlas_csvs] done")


if __name__ == "__main__":
    main()
