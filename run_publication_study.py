"""Backward-compatible entry point for the old Publication_Study runner.

New work should use ``python run_study.py``.  This file remains so old commands
and imports continue to resolve while all execution logic lives in one place.
"""

from __future__ import annotations

from run_study import (  # re-export the stable compatibility surface
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
    raise SystemExit(main())
