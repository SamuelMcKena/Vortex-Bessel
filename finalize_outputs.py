"""Compatibility wrapper — the canonical finalizer is finalize_publication_outputs.py.

This thin wrapper exists so that new scripts can call::

    python finalize_outputs.py

while the underlying implementation stays in
``finalize_publication_outputs.py``, which is imported by the runner and
must keep its current name for backward compatibility.
"""

from __future__ import annotations

import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_root = _here.parent
for _p in (_root, _here):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from finalize_publication_outputs import finalize_outputs  # noqa: E402

__all__ = ["finalize_outputs"]

if __name__ == "__main__":
    import argparse
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(description="Collect and record artifacts under outputs/.")
    parser.add_argument("--run-id", default=None, help="Run ID to associate with this collection.")
    args = parser.parse_args()

    from vbb_study import setup_study

    paths = setup_study.bootstrap(Path(__file__), apply_plot_style=False)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    finalize_outputs(paths["outputs"], run_id=run_id)
    print(f"[study] finalization complete — run_id: {run_id}")
