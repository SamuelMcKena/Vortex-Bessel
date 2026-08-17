"""Regenerate validation_suite.csv from the current engine."""

import sys
from pathlib import Path

_pub = Path(__file__).resolve().parent.parent
_root = _pub.parent
for _p in (_root, _pub):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from vbb_study import vbb_validation, setup_study

paths = setup_study.bootstrap(apply_plot_style=False)
output_dir = paths["outputs"] / "csv" / "publication_study"
output_dir.mkdir(parents=True, exist_ok=True)

df = vbb_validation.run_validation_suite(output_dir=str(output_dir), save=True)
all_pass = bool(df["pass"].all())
n = len(df)
print(f"Validation suite: all_pass={all_pass}, n={n}")
for group in df["group"].unique():
    sub = df[df["group"] == group]
    passed = int(sub["pass"].sum())
    print(f"  {group}: {passed}/{len(sub)} pass")
