"""Canonical runner for the Vortex-Bessel structured-beam study.

The old Publication_Study grew several generations of notebook names and runner
scripts. This runner deliberately discovers the notebooks that actually exist
in this repository, so there is one place to run the current code without
remembering which historical file was newest.

Examples
--------
List the available stages and notebooks::

    python run_study.py --list

Run the current replacement for the original numbered Publication_Study::

    python run_study.py --stage publication

Run the scalar workflow::

    python run_study.py --stage scalar

Run the physical/holographic bench-realism workflow::

    python run_study.py --stage lab_realism

Run every numerical/model notebook (experimental data-dependent notebooks are
excluded from ``all`` on purpose)::

    python run_study.py --stage all

Run the measured axicon-aberration notebook after setting its data directory::

    set BESSEL_ZSCAN_DATA_DIR=D:\\path\\to\\z-scan
    python run_study.py --stage experimental

Executed notebooks are written beneath ``outputs/executed_notebooks`` rather
than modifying the source notebooks in place.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

PROJECT_SCHEMA_VERSION = "3.1.0"
ROOT = Path(__file__).resolve().parent
NOTEBOOK_ROOT = ROOT / "notebooks"
OUTPUT_ROOT = ROOT / "outputs"
EXECUTED_ROOT = OUTPUT_ROOT / "executed_notebooks"
STUDY_OVERVIEW_NOTEBOOK = "notebooks/00_study_overview_and_conventions.ipynb"

# Canonical topic directories. Their contents are discovered at runtime so a
# renamed/added notebook cannot silently leave the runner pointing at a dead
# path. Experimental is intentionally opt-in because it depends on local
# measurement data that should not be committed to Git.
STAGE_DIRS: OrderedDict[str, str] = OrderedDict([
    ("scalar", "scalar"),
    ("lab_realism", "lab_realism"),
    ("vector", "vector"),
    ("materials", "materials"),
    ("advanced", "advanced"),
    ("digital_twin", "digital_twin"),
    ("experimental", "experimental"),
])
DEFAULT_STAGES = tuple(name for name in STAGE_DIRS if name != "experimental")
DEFAULT_CLEAN_OUTPUTS = ("executed_notebooks", "jupyter_runtime")

# Current replacements for the original numbered 00--10 Publication_Study
# workflow. Some historical notebooks have been split into two clearer current
# notebooks, so this intentionally has more than eleven entries. The mapping is
# documented in docs/PUBLICATION_STUDY_MAP.md.
PUBLICATION_NOTEBOOKS = [
    STUDY_OVERVIEW_NOTEBOOK,
    "notebooks/scalar/02_scalar_ideal_vs_lab_diagnostics.ipynb",
    "notebooks/scalar/03_scalar_robustness_and_self_healing.ipynb",
    "notebooks/scalar/04_scalar_parameter_sweeps.ipynb",
    "notebooks/scalar/05_scalar_validation_suite.ipynb",
    "notebooks/vector/01_vector_beam_theory_atlas.ipynb",
    "notebooks/vector/02_vector_ideal_vs_lab_case1.ipynb",
    "notebooks/vector/03_vector_hardware_routes.ipynb",
    "notebooks/materials/01_material_proxy_fluence_and_thresholds.ipynb",
    "notebooks/materials/02_material_calibration_template.ipynb",
    "notebooks/materials/03_application_design_tables.ipynb",
    "notebooks/advanced/01_capsule_weld_feature_design.ipynb",
    "notebooks/advanced/03_discrete_nfold_beams.ipynb",
]


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _discover_stage(stage: str) -> list[str]:
    if stage not in STAGE_DIRS:
        raise ValueError(f"Unknown stage {stage!r}. Choose from: {', '.join(STAGE_DIRS)}, publication, all")
    folder = NOTEBOOK_ROOT / STAGE_DIRS[stage]
    if not folder.exists():
        return []
    return [_relative(path) for path in sorted(folder.rglob("*.ipynb"))]


def _build_registry() -> OrderedDict[str, list[str]]:
    return OrderedDict((stage, _discover_stage(stage)) for stage in STAGE_DIRS)


STAGE_NOTEBOOKS = _build_registry()
ORDERED_NOTEBOOKS = [STUDY_OVERVIEW_NOTEBOOK] + [
    nb for stage in DEFAULT_STAGES for nb in STAGE_NOTEBOOKS[stage]
]


def notebooks_for_stage(stage: str) -> list[str]:
    """Return the current notebook list for one stage.

    ``publication`` is the maintained replacement for the original numbered
    Publication_Study workflow. ``all`` means all numerical/model topic stages
    and deliberately excludes the measured experimental stage. The overview
    notebook is prepended to normal numerical topic stages when it exists.
    """
    if stage == "publication":
        return list(PUBLICATION_NOTEBOOKS)
    if stage == "all":
        notebooks = [nb for name in DEFAULT_STAGES for nb in STAGE_NOTEBOOKS[name]]
    else:
        if stage not in STAGE_NOTEBOOKS:
            raise ValueError(f"Unknown stage {stage!r}")
        notebooks = list(STAGE_NOTEBOOKS[stage])
    overview = ROOT / STUDY_OVERVIEW_NOTEBOOK
    if overview.exists() and stage != "experimental":
        return [STUDY_OVERVIEW_NOTEBOOK] + notebooks
    return notebooks


def selected_notebooks(*, stage: str | None = None, only: str | None = None, **_: object) -> list[str]:
    """Compatibility helper retained for older imports.

    New code should use ``--stage`` or ``--only``. Historical start/stop slice
    arguments are intentionally no longer part of the public CLI because they
    encouraged dependence on stale notebook orderings.
    """
    if only:
        path = Path(only)
        candidate = path if path.is_absolute() else ROOT / path
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        return [_relative(candidate)]
    return notebooks_for_stage(stage or "all")


def print_notebook_order() -> None:
    print("Vortex-Bessel runnable notebook map")
    print(f"  overview: {STUDY_OVERVIEW_NOTEBOOK}")
    print("\n[publication] (current replacement for original numbered Publication_Study)")
    for notebook in PUBLICATION_NOTEBOOKS:
        print(f"  {notebook}")
    for stage, notebooks in STAGE_NOTEBOOKS.items():
        suffix = " (local measurement data required)" if stage == "experimental" else ""
        print(f"\n[{stage}]{suffix}")
        if not notebooks:
            print("  <none>")
        for notebook in notebooks:
            print(f"  {notebook}")


def clean_output_folders(*args: object, dry_run: bool = False, **kwargs: object) -> list[Path]:
    """Clean only transient notebook-execution folders.

    Governed validation evidence under ``outputs/validation`` is never touched.
    The loose signature preserves compatibility with older imports.
    """
    requested = kwargs.get("requested")
    if requested is None and args:
        # Old signature was (paths, requested); new direct use can be (requested,).
        requested = args[-1] if isinstance(args[-1], (list, tuple)) else None
    names = list(requested or DEFAULT_CLEAN_OUTPUTS)
    cleaned: list[Path] = []
    for name in names:
        if str(name) not in DEFAULT_CLEAN_OUTPUTS:
            raise ValueError(
                f"Refusing to clean {name!r}; allowed transient folders are: "
                + ", ".join(DEFAULT_CLEAN_OUTPUTS)
            )
        target = OUTPUT_ROOT / str(name)
        cleaned.append(target)
        if dry_run:
            print(f"[study] would clean {target.relative_to(ROOT)}")
            continue
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        print(f"[study] cleaned {target.relative_to(ROOT)}")
    return cleaned


def _execute_notebook(relative_name: str, timeout_s: int) -> None:
    source = ROOT / relative_name
    stage_name = source.parent.name
    if "experimental" in source.parts:
        stage_name = "experimental"
    destination_dir = EXECUTED_ROOT / stage_name
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    runtime_dir = OUTPUT_ROOT / "jupyter_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    env["JUPYTER_RUNTIME_DIR"] = str(runtime_dir)
    env["JUPYTER_ALLOW_INSECURE_WRITES"] = "1"

    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(source),
        "--output",
        str(destination),
        f"--ExecutePreprocessor.timeout={int(timeout_s)}",
        "--ExecutePreprocessor.kernel_name=python3",
    ]
    print(f"[study] executing {relative_name}", flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def run_notebooks(
    *,
    timeout_s: int = 1800,
    stage: str | None = None,
    only: str | None = None,
    clean_output: list[str] | None = None,
    continue_on_error: bool = False,
    **_: object,
) -> dict[str, object]:
    """Execute a clean stage selection and return a compact run summary."""
    requested = selected_notebooks(stage=stage, only=only)
    if clean_output is not None:
        clean_output_folders(clean_output)

    completed: list[str] = []
    failed: list[dict[str, str]] = []
    for notebook in requested:
        try:
            _execute_notebook(notebook, timeout_s)
            completed.append(notebook)
        except Exception as exc:
            failed.append({"notebook": notebook, "error": f"{type(exc).__name__}: {exc}"})
            if not continue_on_error:
                raise
    return {"requested": requested, "completed": completed, "failed": failed}


# Compatibility alias: older code imported RunResult as a named object. The
# clean runner returns a plain serialisable summary instead.
RunResult = dict


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list current runnable notebooks")
    parser.add_argument(
        "--stage",
        choices=[*STAGE_DIRS.keys(), "publication", "all"],
        default="all",
        help=(
            "workflow to execute; 'publication' replaces the original numbered "
            "Publication_Study and 'all' excludes local-data experimental notebooks"
        ),
    )
    parser.add_argument("--only", help="execute one repository-relative notebook")
    parser.add_argument("--timeout-s", type=int, default=1800, help="per-notebook timeout")
    parser.add_argument("--dry-run", action="store_true", help="show the execution plan only")
    parser.add_argument("--clean", action="store_true", help="remove transient executed notebooks/runtime first")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print_notebook_order()
        return 0

    requested = selected_notebooks(stage=args.stage, only=args.only)
    missing = [name for name in requested if not (ROOT / name).exists()]
    if missing:
        parser.error("runner registry contains missing notebooks: " + ", ".join(missing))

    if args.dry_run:
        if args.clean:
            clean_output_folders(dry_run=True)
        print("[study] execution plan")
        for name in requested:
            print(f"  {name}")
        return 0

    if args.clean:
        clean_output_folders()

    summary = run_notebooks(
        stage=args.stage,
        only=args.only,
        timeout_s=args.timeout_s,
        continue_on_error=args.continue_on_error,
    )
    print(
        f"[study] completed {len(summary['completed'])}/{len(summary['requested'])} notebooks; "
        f"failures={len(summary['failed'])}"
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
