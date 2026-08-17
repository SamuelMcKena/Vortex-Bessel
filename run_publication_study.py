"""Compatibility wrapper — the canonical runner is now run_study.py.

This script preserves the original command-line interface so that existing
calls, scripts, and documentation that reference ``run_publication_study.py``
continue to work unchanged.  All logic now lives in ``run_study.py``.

For new work use::

    python Publication_Study/run_study.py [options]

or with stage selection::

    python Publication_Study/run_study.py --stage scalar

The wrapper re-exports the key objects so that code that imports from this
module (e.g. ``from Publication_Study.run_publication_study import
ORDERED_NOTEBOOKS``) also keeps working.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure both the repo root and Publication_Study are on sys.path so that
# run_study can import vbb_study regardless of caller location.
_here = Path(__file__).resolve().parent
_root = _here.parent
for _p in (_root, _here):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_study import (  # noqa: E402 — must follow path setup
    DEFAULT_CLEAN_OUTPUTS,
    ORDERED_NOTEBOOKS,
    PROJECT_SCHEMA_VERSION,
    RunResult,
    STAGE_NOTEBOOKS,
    STUDY_OVERVIEW_NOTEBOOK,
    clean_output_folders,
    main,
    notebooks_for_stage,
    print_notebook_order,
    run_notebooks,
    selected_notebooks,
)

__all__ = [
    "DEFAULT_CLEAN_OUTPUTS",
    "ORDERED_NOTEBOOKS",
    "PROJECT_SCHEMA_VERSION",
    "RunResult",
    "STAGE_NOTEBOOKS",
    "STUDY_OVERVIEW_NOTEBOOK",
    "clean_output_folders",
    "main",
    "notebooks_for_stage",
    "print_notebook_order",
    "run_notebooks",
    "selected_notebooks",
]

if __name__ == "__main__":
    main()
