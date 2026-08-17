"""Backward-compatible entry point for the old Publication_Study runner.

Running this file with no explicit stage now executes the maintained
``publication`` suite: the current replacements for the original numbered
Publication_Study notebooks. All execution logic still lives in ``run_study``.
"""

from __future__ import annotations

import sys

from run_study import (  # re-export the stable compatibility surface
    DEFAULT_CLEAN_OUTPUTS,
    ORDERED_NOTEBOOKS,
    PROJECT_SCHEMA_VERSION,
    PUBLICATION_NOTEBOOKS,
    RunResult,
    STAGE_NOTEBOOKS,
    STUDY_OVERVIEW_NOTEBOOK,
    clean_output_folders,
    main as _study_main,
    notebooks_for_stage,
    print_notebook_order,
    run_notebooks,
    selected_notebooks,
)


def main(argv: list[str] | None = None) -> int:
    """Run the current publication suite unless the caller selects another mode."""

    args = list(sys.argv[1:] if argv is None else argv)
    has_explicit_selection = any(
        arg == "--list" or arg == "--only" or arg.startswith("--only=")
        or arg == "--stage" or arg.startswith("--stage=")
        for arg in args
    )
    if not has_explicit_selection:
        args = ["--stage", "publication", *args]
    return _study_main(args)


__all__ = [
    "DEFAULT_CLEAN_OUTPUTS",
    "ORDERED_NOTEBOOKS",
    "PROJECT_SCHEMA_VERSION",
    "PUBLICATION_NOTEBOOKS",
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
