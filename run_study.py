"""Execute the structured-beam simulation atlas study.

This is the canonical runner for the Structured-Beam Simulation Atlas
(internally called VBB Study). It replaces ``run_publication_study.py``,
which is now a thin compatibility wrapper that delegates here.

Notebooks are organised into stages that match the study taxonomy:
  quicklook    - Stage 8.7 fast beam-to-sample diagnostic simulator
  scalar        — scalar Bessel-Gauss baselines and diagnostics
  lab_realism   — holographic/physical axicon, objective, interface
  vector        — polarisation-structured vector beams
  materials     — material proxy, fluence, calibration templates
  capsule       — capsule/weld-feature application geometry proxies
  advanced      — hexagonal, polygonal, discrete N-fold
  publication_exports — final paper figures, tables, report export

Usage examples
--------------
List canonical notebook order::

    python run_study.py --list

Preview a run without executing::

    python run_study.py --dry-run

Run all stages::

    python run_study.py --timeout-s 1800

Run one stage only::

    python run_study.py --stage scalar

Run one notebook::

    python run_study.py --only notebooks/scalar/04_scalar_parameter_sweeps.ipynb

Run a slice::

    python run_study.py --start-at notebooks/scalar/02_scalar_ideal_vs_lab_diagnostics.ipynb \\
                        --stop-after notebooks/scalar/05_scalar_validation_suite.ipynb

Clean outputs before run::

    python run_study.py --clean-output figures csv manifests --dry-run
    python run_study.py --clean-output figures csv manifests --stage scalar
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap — must happen before any vbb_study import
# ---------------------------------------------------------------------------

PUBLICATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = PUBLICATION_DIR.parent
for _candidate in (REPO_ROOT, PUBLICATION_DIR):
    _candidate_text = str(_candidate)
    if _candidate_text in sys.path:
        sys.path.remove(_candidate_text)
    sys.path.insert(0, _candidate_text)

from vbb_study import setup_study  # noqa: E402

# ---------------------------------------------------------------------------
# Study schema version — increment when outputs schema changes
# ---------------------------------------------------------------------------

PROJECT_SCHEMA_VERSION = "2.0.0"

# ---------------------------------------------------------------------------
# Notebook registry
# ---------------------------------------------------------------------------
# Every notebook path is relative to PUBLICATION_DIR.  Using forward slashes
# works on all platforms via pathlib.

STUDY_OVERVIEW_NOTEBOOK = "notebooks/00_study_overview_and_conventions.ipynb"

# Notebooks organised by study stage, in canonical execution order.
STAGE_NOTEBOOKS: OrderedDict[str, list[str]] = OrderedDict([
    ("quicklook", [
        "notebooks/quicklook/00_quick_beam_to_sample_simulator.ipynb",
    ]),
    ("scalar", [
        "notebooks/scalar/02_scalar_ideal_vs_lab_diagnostics.ipynb",
        "notebooks/scalar/03_scalar_robustness_and_self_healing.ipynb",
        "notebooks/scalar/04_scalar_parameter_sweeps.ipynb",
        "notebooks/scalar/05_scalar_validation_suite.ipynb",
    ]),
    ("lab_realism", [
        "notebooks/lab_realism/01_holographic_axicon_route.ipynb",
        "notebooks/lab_realism/02_physical_axicon_route.ipynb",
        "notebooks/lab_realism/03_holographic_vs_physical_axicon.ipynb",
        "notebooks/lab_realism/04_objective_pupil_and_first_order_filtering.ipynb",
        "notebooks/lab_realism/05_through_sample_interface.ipynb",
        "notebooks/lab_realism/06_full_source_to_sample_journey.ipynb",
    ]),
    ("vector", [
        "notebooks/vector/01_vector_beam_theory_atlas.ipynb",
        "notebooks/vector/02_vector_ideal_vs_lab_case1.ipynb",
        "notebooks/vector/03_vector_hardware_routes.ipynb",
    ]),
    ("materials", [
        "notebooks/materials/01_material_proxy_fluence_and_thresholds.ipynb",
        "notebooks/materials/02_material_calibration_template.ipynb",
        "notebooks/materials/03_application_design_tables.ipynb",
    ]),
    ("capsule", [
        "notebooks/advanced/01_capsule_weld_feature_design.ipynb",
    ]),
    ("advanced", [
        "notebooks/advanced/02_hexagonal_polygonal_beams.ipynb",
        "notebooks/advanced/03_discrete_nfold_beams.ipynb",
    ]),
    ("digital_twin", [
        "notebooks/digital_twin/03_nathan_vector_hexagon_target_field.ipynb",
        "notebooks/digital_twin/04_patterned_hwp_vector_route.ipynb",
        "notebooks/digital_twin/05_serial_dual_slm_vector_route.ipynb",
        "notebooks/digital_twin/06_shared_axicon_hexagon_propagation.ipynb",
        "notebooks/digital_twin/07_nathan_vector_hexagon_robustness_and_equivalence.ipynb",
    ]),
    ("publication_exports", [
        "notebooks/publication_exports/03_report_export.ipynb",
    ]),
])

# Full ordered sequence: study overview first, then all stages in order.
ORDERED_NOTEBOOKS: list[str] = (
    [STUDY_OVERVIEW_NOTEBOOK]
    + [nb for stage_nbs in STAGE_NOTEBOOKS.values() for nb in stage_nbs]
)

KNOWN_STAGES = list(STAGE_NOTEBOOKS.keys()) + ["all"]

DEFAULT_CLEAN_OUTPUTS = ("figures", "csv", "holograms", "manifests", "jupyter_runtime")


# ---------------------------------------------------------------------------
# Run result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    run_id: str
    stage: str | None
    requested_notebooks: list[str]
    completed_notebooks: list[str]
    failed_notebook: dict[str, object] | None
    start_manifest: Path
    finish_manifest: Path


# ---------------------------------------------------------------------------
# Notebook selection helpers
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_run_id() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _validate_notebook_name(name: str | None, option: str) -> None:
    if name is None:
        return
    if name not in ORDERED_NOTEBOOKS:
        known = "\n  ".join(ORDERED_NOTEBOOKS)
        raise ValueError(f"{option} names an unknown notebook: {name}\nKnown notebooks:\n  {known}")


def notebooks_for_stage(stage: str) -> list[str]:
    """Return notebooks for a given stage name, or all notebooks for 'all'."""

    if stage == "all":
        return list(ORDERED_NOTEBOOKS)
    if stage not in STAGE_NOTEBOOKS:
        options = ", ".join(KNOWN_STAGES)
        raise ValueError(f"Unknown stage {stage!r}. Choose from: {options}")
    # Include the study overview notebook when running any single stage.
    return [STUDY_OVERVIEW_NOTEBOOK] + list(STAGE_NOTEBOOKS[stage])


def selected_notebooks(
    *,
    stage: str | None = None,
    start_at: str | None = None,
    stop_after: str | None = None,
    only: str | None = None,
) -> list[str]:
    """Return the validated notebook slice requested by CLI options.

    ``--stage`` selects a predefined topic group.  ``--start-at``,
    ``--stop-after``, and ``--only`` operate on the full ordered list,
    same as the old runner.  They cannot be combined with ``--stage``.
    """

    if stage and (start_at or stop_after or only):
        raise ValueError("--stage cannot be combined with --start-at, --stop-after, or --only")
    if stage:
        return notebooks_for_stage(stage)

    _validate_notebook_name(start_at, "--start-at")
    _validate_notebook_name(stop_after, "--stop-after")
    _validate_notebook_name(only, "--only")

    if only and (start_at or stop_after):
        raise ValueError("--only cannot be combined with --start-at or --stop-after")
    if only:
        return [only]

    start_index = ORDERED_NOTEBOOKS.index(start_at) if start_at else 0
    stop_index = ORDERED_NOTEBOOKS.index(stop_after) if stop_after else len(ORDERED_NOTEBOOKS) - 1
    if start_index > stop_index:
        raise ValueError("--start-at must not come after --stop-after in the ordered notebook list")
    return ORDERED_NOTEBOOKS[start_index: stop_index + 1]


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _manifest_payload(
    *,
    run_id: str,
    phase: str,
    paths: dict[str, Path],
    stage: str | None,
    requested_notebooks: list[str],
    completed_notebooks: list[str],
    failed_notebook: dict[str, object] | None,
    started_at: datetime,
    clean_output: list[str] | None,
    finalize_requested: bool,
    continue_on_error: bool,
) -> dict[str, object]:
    return {
        "project_schema_version": PROJECT_SCHEMA_VERSION,
        "run_id": run_id,
        "phase": phase,
        "stage": stage,
        "timestamp_utc": _utc_now().isoformat(),
        "started_at_utc": started_at.isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": setup_study.code_version(paths["root"]),
        "requested_notebooks": requested_notebooks,
        "completed_notebooks": completed_notebooks,
        "failed_notebook": failed_notebook,
        "output_root": str(paths["outputs"]),
        "clean_output": clean_output,
        "finalize_requested": finalize_requested,
        "continue_on_error": continue_on_error,
    }


def _write_run_manifest(
    *,
    paths: dict[str, Path],
    run_id: str,
    phase: str,
    stage: str | None,
    requested_notebooks: list[str],
    completed_notebooks: list[str],
    failed_notebook: dict[str, object] | None,
    started_at: datetime,
    clean_output: list[str] | None,
    finalize_requested: bool,
    continue_on_error: bool,
) -> Path:
    manifest = _manifest_payload(
        run_id=run_id,
        phase=phase,
        paths=paths,
        stage=stage,
        requested_notebooks=requested_notebooks,
        completed_notebooks=completed_notebooks,
        failed_notebook=failed_notebook,
        started_at=started_at,
        clean_output=clean_output,
        finalize_requested=finalize_requested,
        continue_on_error=continue_on_error,
    )
    return _write_json(paths["manifests"] / f"{run_id}_{phase}.json", manifest)


# ---------------------------------------------------------------------------
# Output cleaning
# ---------------------------------------------------------------------------


def _resolve_output_child(output_root: Path, requested: str) -> Path:
    relative = Path(requested)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"--clean-output must name folders inside outputs/: {requested}")
    target = (output_root / relative).resolve()
    root = output_root.resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError(f"--clean-output refuses to clean outside outputs/: {requested}")
    return target


def clean_output_folders(
    paths: dict[str, Path],
    requested: list[str],
    *,
    dry_run: bool = False,
) -> list[Path]:
    """Remove and recreate selected generated folders under outputs/."""

    output_root = paths["outputs"]
    names = list(DEFAULT_CLEAN_OUTPUTS if not requested else requested)
    cleaned: list[Path] = []
    for name in names:
        target = _resolve_output_child(output_root, name)
        cleaned.append(target)
        if dry_run:
            print(f"[study] would clean outputs/{Path(name).as_posix()}")
            continue
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        print(f"[study] cleaned {target}")
    return cleaned


# ---------------------------------------------------------------------------
# Notebook execution
# ---------------------------------------------------------------------------


def _notebook_command(notebook: Path, timeout_s: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to", "notebook",
        "--execute",
        str(notebook),
        "--inplace",
        f"--ExecutePreprocessor.timeout={int(timeout_s)}",
        "--ExecutePreprocessor.kernel_name=python3",
    ]


def run_notebooks(
    *,
    timeout_s: int = 1800,
    stage: str | None = None,
    start_at: str | None = None,
    stop_after: str | None = None,
    only: str | None = None,
    clean_output: list[str] | None = None,
    no_finalize: bool = False,
    continue_on_error: bool = False,
) -> RunResult:
    """Execute the requested notebooks and write start/finish manifests."""

    paths = setup_study.bootstrap(Path(__file__), apply_plot_style=False)
    setup_study.validate_workspace(paths, strict=True)
    publication = paths["publication"]
    requested = selected_notebooks(
        stage=stage, start_at=start_at, stop_after=stop_after, only=only
    )
    if clean_output is not None:
        clean_output_folders(paths, clean_output)

    run_id = _new_run_id()
    started_at = _utc_now()
    completed: list[str] = []
    failed: dict[str, object] | None = None

    start_manifest = _write_run_manifest(
        paths=paths,
        run_id=run_id,
        phase="start",
        stage=stage,
        requested_notebooks=requested,
        completed_notebooks=completed,
        failed_notebook=failed,
        started_at=started_at,
        clean_output=clean_output,
        finalize_requested=not no_finalize,
        continue_on_error=continue_on_error,
    )

    for name in requested:
        notebook = publication / name
        if not notebook.exists():
            raise FileNotFoundError(notebook)
        command = _notebook_command(notebook, timeout_s)
        print(f"[study] executing {name}", flush=True)
        env = os.environ.copy()
        runtime_dir = paths["outputs"] / "jupyter_runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        env["JUPYTER_RUNTIME_DIR"] = str(runtime_dir)
        env["JUPYTER_ALLOW_INSECURE_WRITES"] = "1"
        # Propagate the run_id and stage into the notebook kernel so
        # annotate_scalar_row can stamp every row with a real identifier, and
        # the fast-preset caution cell can log the execution context.
        env["STRUCTURED_BEAM_RUN_ID"] = run_id
        env["STRUCTURED_BEAM_STAGE"] = stage or "all"
        # Ensure notebooks in subdirectories can find vbb_study and
        # bessel_twin_core regardless of the kernel's working directory.
        # Publication_Study/ comes FIRST so `import vbb_study` resolves to
        # the real package (Publication_Study/vbb_study/) which has all
        # subpackages (publication/, studies/, equations/).  The repo root
        # follows for backward-compatibility shims and Publication_Study
        # package access.
        _pp = os.pathsep.join([
            str(paths["publication"]),
            str(paths["root"]),
        ])
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = _pp + (os.pathsep + existing_pp if existing_pp else "")
        try:
            subprocess.run(command, cwd=paths["root"], check=True, env=env)
        except subprocess.CalledProcessError as exc:
            failed = {"name": name, "returncode": exc.returncode}
            print(f"[study] failed {name} with return code {exc.returncode}", flush=True)
            if not continue_on_error:
                break
        else:
            completed.append(name)

    if failed is None or continue_on_error:
        if no_finalize:
            print("[study] skipping final output collection (--no-finalize)", flush=True)
        else:
            from Publication_Study.finalize_publication_outputs import finalize_outputs

            finalize_outputs(paths["outputs"], run_id=run_id, run_started_at=started_at)

    finish_manifest = _write_run_manifest(
        paths=paths,
        run_id=run_id,
        phase="finish",
        stage=stage,
        requested_notebooks=requested,
        completed_notebooks=completed,
        failed_notebook=failed,
        started_at=started_at,
        clean_output=clean_output,
        finalize_requested=not no_finalize,
        continue_on_error=continue_on_error,
    )
    result = RunResult(
        run_id=run_id,
        stage=stage,
        requested_notebooks=requested,
        completed_notebooks=completed,
        failed_notebook=failed,
        start_manifest=start_manifest,
        finish_manifest=finish_manifest,
    )
    if failed is not None and not continue_on_error:
        raise RuntimeError(f"Notebook failed: {failed['name']} (return code {failed['returncode']})")
    return result


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------


def print_notebook_order(
    names: list[str] | None = None,
    *,
    by_stage: bool = False,
) -> None:
    if by_stage:
        print("[study] notebook order by stage:")
        idx = 1
        print(f"  [overview]")
        print(f"    {idx:02d}. {STUDY_OVERVIEW_NOTEBOOK}")
        idx += 1
        for stage_name, nbs in STAGE_NOTEBOOKS.items():
            print(f"  [{stage_name}]")
            for nb in nbs:
                print(f"    {idx:02d}. {nb}")
                idx += 1
    else:
        print("[study] notebook order:")
        for index, name in enumerate(names or ORDERED_NOTEBOOKS, start=1):
            print(f"  {index:02d}. {name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the structured-beam simulation atlas study notebooks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Stages: " + ", ".join(KNOWN_STAGES) + "\n\n"
            "Example:\n"
            "  python run_study.py --stage scalar\n"
            "  python run_study.py --only notebooks/scalar/04_scalar_parameter_sweeps.ipynb\n"
        ),
    )
    parser.add_argument(
        "--timeout-s", type=int, default=1800,
        help="Per-notebook execution timeout in seconds. Default: 1800.",
    )
    parser.add_argument(
        "--stage",
        choices=KNOWN_STAGES,
        default=None,
        help="Run only notebooks in a given study stage (or 'all').",
    )
    parser.add_argument("--start-at", default=None, help="Notebook filename to start from.")
    parser.add_argument("--stop-after", default=None, help="Notebook filename to stop after.")
    parser.add_argument("--only", default=None, help="Execute exactly one notebook by path.")
    parser.add_argument(
        "--list", action="store_true",
        help="Print the canonical notebook order (use --list --stage X to list one stage).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without executing.")
    parser.add_argument(
        "--clean-output",
        nargs="*",
        default=None,
        metavar="FOLDER",
        help=(
            "Remove/recreate selected folders under outputs/ before running. "
            "With no folder names cleans: " + ", ".join(DEFAULT_CLEAN_OUTPUTS) + "."
        ),
    )
    parser.add_argument("--no-finalize", action="store_true", help="Skip final artifact/caption collection.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue after a failed notebook. Default behaviour is fail-fast.",
    )
    args = parser.parse_args()

    try:
        requested = selected_notebooks(
            stage=args.stage,
            start_at=args.start_at,
            stop_after=args.stop_after,
            only=args.only,
        )
    except ValueError as exc:
        parser.error(str(exc))

    paths = setup_study.bootstrap(Path(__file__), apply_plot_style=False)
    try:
        setup_study.validate_workspace(paths, strict=True)
    except FileNotFoundError as exc:
        parser.exit(2, f"{exc}\n")

    if args.list:
        print_notebook_order(requested if args.stage else None, by_stage=bool(args.stage))
        return

    if args.dry_run:
        print_notebook_order(requested, by_stage=False)
        if args.clean_output is not None:
            clean_output_folders(paths, args.clean_output, dry_run=True)
        print(f"[study] output root:   {paths['outputs']}")
        print(f"[study] stage:         {args.stage or 'all'}")
        print(f"[study] finalization:  {'skip' if args.no_finalize else 'would run'}")
        return

    try:
        result = run_notebooks(
            timeout_s=args.timeout_s,
            stage=args.stage,
            start_at=args.start_at,
            stop_after=args.stop_after,
            only=args.only,
            clean_output=args.clean_output,
            no_finalize=args.no_finalize,
            continue_on_error=args.continue_on_error,
        )
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")

    print("[study] completed:")
    for name in result.completed_notebooks:
        print(f"  {name}")
    print(f"[study] run_id:    {result.run_id}")
    print(f"[study] manifests: {result.start_manifest.name}, {result.finish_manifest.name}")


if __name__ == "__main__":
    main()
