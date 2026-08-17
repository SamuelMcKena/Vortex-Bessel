"""Regenerate scalar shortlist CSVs for fast and publication presets.

Produces:
  outputs/csv/publication_study/scalar_shortlist_realistic_summary.csv
  outputs/csv/publication_study/scalar_shortlist_realistic_summary_publication.csv
  outputs/csv/publication_study/scalar_preset_comparison_fast_vs_publication.csv

Run:
    python Publication_Study/tools/regen_scalar_shortlist.py
    python Publication_Study/tools/regen_scalar_shortlist.py --preset publication
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_pub = Path(__file__).resolve().parent.parent
_root = _pub.parent
for _p in (_root, _pub):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from vbb_study import setup_study
from vbb_study.studies.scalar_cases import (
    DEFAULT_SHORTLIST,
    run_shortlist,
    save_shortlist_csv,
)
from vbb_study.publication.tables import (
    SCALAR_OUTPUT_SCHEMA_VERSION,
    ordered_row,
    propagation_power_label,
)

import pandas as pd


DEFAULT_CASES = DEFAULT_SHORTLIST


def run_and_save(
    preset: str,
    path: str = "realistic",
    cases: list | None = None,
    *,
    output_dir: Path,
    run_id: str | None = None,
) -> pd.DataFrame:
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"  Running preset={preset!r}, path={path!r}, {len(cases or DEFAULT_CASES)} cases ...")
    df = run_shortlist(cases, preset=preset, path=path, run_id=run_id, qa_status="exploratory")

    suffix = "" if preset == "fast" else f"_{preset}"
    filename = f"scalar_shortlist_realistic_summary{suffix}.csv"
    csv_path = save_shortlist_csv(df, output_dir, filename=filename)
    print(f"  Saved: {csv_path.relative_to(_pub)}")

    # Print power QA summary.
    if "propagation_power_label" in df.columns:
        counts = df["propagation_power_label"].value_counts().to_dict()
        print(f"  Power QA: {counts}")
        if counts.get("fail", 0) > 0:
            print(f"  WARNING: {counts['fail']} case(s) labelled 'fail' "
                  f"(>20% power drift). Not publication-ready at this preset.")
    return df


def build_comparison(
    fast_df: pd.DataFrame,
    pub_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Build a fast-vs-publication preset comparison CSV."""

    compare_cols = [
        "run_id",
        "generated_at_utc",
        "source_schema_version",
        "case_id",
        "preset",
        "canonical_zone_um",
        "strict_bessel_region_um",
        "propagation_power_drift_fraction",
        "propagation_power_label",
        "qa_status",
        "output_dx_at_peak_um",
        "focal_samples_per_radial_period",
    ]
    rows = []
    for df in (fast_df, pub_df):
        for _, row in df.iterrows():
            r = {c: row.get(c, float("nan")) for c in compare_cols}
            rows.append(r)
    cmp = pd.DataFrame(rows)
    path = output_dir / "scalar_preset_comparison_fast_vs_publication.csv"
    cmp.to_csv(path, index=False)
    print(f"  Comparison saved: {path.relative_to(_pub)}")
    return cmp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate scalar shortlist CSVs for one or both presets."
    )
    parser.add_argument(
        "--preset",
        choices=["fast", "publication", "both"],
        default="both",
        help="Which preset to run (default: both).",
    )
    parser.add_argument(
        "--path",
        choices=["realistic", "ideal"],
        default="realistic",
    )
    args = parser.parse_args()

    paths = setup_study.bootstrap(_pub, apply_plot_style=False)
    output_dir = paths["csv"] / "publication_study"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"[regen_scalar_shortlist] run_id={run_id}")

    fast_df = pub_df = None

    if args.preset in ("fast", "both"):
        print("\n--- fast preset ---")
        fast_df = run_and_save(
            "fast", path=args.path, output_dir=output_dir, run_id=run_id
        )

    if args.preset in ("publication", "both"):
        print("\n--- publication preset ---")
        pub_df = run_and_save(
            "publication", path=args.path, output_dir=output_dir, run_id=run_id
        )

    if fast_df is not None and pub_df is not None:
        print("\n--- comparison ---")
        build_comparison(fast_df, pub_df, output_dir)

    print("\n[regen_scalar_shortlist] done")


if __name__ == "__main__":
    main()
