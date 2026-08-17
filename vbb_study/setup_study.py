"""Workspace bootstrap and manifest helpers for the structured-beam study.

The notebooks, scripts, and smoke tests all need the same path contract:

* ``root`` is the checkout root that contains ``Publication_Study``.
* ``publication`` is the study workspace (``Publication_Study/``).
* ``vbb_study`` and ``bessel_twin_core.py`` inside ``Publication_Study`` are
  the active source.
* ``docs`` is ``Publication_Study/docs``.
* ``outputs`` and its named children are generated folders.

Keeping the contract here prevents notebooks from drifting back to local
absolute paths.

Notebooks now live in ``Publication_Study/notebooks/`` subdirectories.  All
entries in ``REQUIRED_NOTEBOOKS`` are relative to the ``publication`` path,
e.g. ``"notebooks/scalar/02_scalar_ideal_vs_lab_diagnostics.ipynb"``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


# Canonical notebook list — paths relative to the publication directory.
# These are the minimum notebooks required for workspace validation.
# The full ordered execution sequence lives in run_study.ORDERED_NOTEBOOKS.
REQUIRED_NOTEBOOKS = [
    "notebooks/00_study_overview_and_conventions.ipynb",
    # quicklook branch
    "notebooks/quicklook/00_quick_beam_to_sample_simulator.ipynb",
    # scalar branch
    "notebooks/scalar/02_scalar_ideal_vs_lab_diagnostics.ipynb",
    "notebooks/scalar/03_scalar_robustness_and_self_healing.ipynb",
    "notebooks/scalar/04_scalar_parameter_sweeps.ipynb",
    "notebooks/scalar/05_scalar_validation_suite.ipynb",
    # lab-realism branch
    "notebooks/lab_realism/01_holographic_axicon_route.ipynb",
    "notebooks/lab_realism/02_physical_axicon_route.ipynb",
    "notebooks/lab_realism/03_holographic_vs_physical_axicon.ipynb",
    "notebooks/lab_realism/04_objective_pupil_and_first_order_filtering.ipynb",
    "notebooks/lab_realism/05_through_sample_interface.ipynb",
    "notebooks/lab_realism/06_full_source_to_sample_journey.ipynb",
    # vector branch
    "notebooks/vector/01_vector_beam_theory_atlas.ipynb",
    "notebooks/vector/02_vector_ideal_vs_lab_case1.ipynb",
    "notebooks/vector/03_vector_hardware_routes.ipynb",
    # materials branch
    "notebooks/materials/01_material_proxy_fluence_and_thresholds.ipynb",
    "notebooks/materials/02_material_calibration_template.ipynb",
    "notebooks/materials/03_application_design_tables.ipynb",
    # publication exports
    "notebooks/publication_exports/03_report_export.ipynb",
    # advanced branch
    "notebooks/advanced/01_capsule_weld_feature_design.ipynb",
    "notebooks/advanced/03_discrete_nfold_beams.ipynb",
    # digital-twin vector hexagon branch
    "notebooks/digital_twin/03_nathan_vector_hexagon_target_field.ipynb",
    "notebooks/digital_twin/04_patterned_hwp_vector_route.ipynb",
    "notebooks/digital_twin/05_serial_dual_slm_vector_route.ipynb",
    "notebooks/digital_twin/06_shared_axicon_hexagon_propagation.ipynb",
    "notebooks/digital_twin/07_nathan_vector_hexagon_robustness_and_equivalence.ipynb",
]

REQUIRED_DOCS = [
    # Original docs — kept under old names until Phase 7 renaming.
    "00_theory.md",
    "01_conventions.md",
    "02_validation.md",
    "03_materials_application.md",
    "04_actual_lab_vector_case1.md",
    "05_study_taxonomy.md",
    "16_figure_output_governance.md",
    # New docs added in Phase 2.
    "00_project_overview.md",
    "08_refactor_plan.md",
]

REQUIRED_SOURCE_FILES = [
    "Publication_Study/bessel_twin_core.py",
    # New canonical runner.
    "Publication_Study/run_study.py",
    # Compatibility wrapper (delegates to run_study.py).
    "Publication_Study/run_publication_study.py",
    "Publication_Study/finalize_publication_outputs.py",
    "Publication_Study/vbb_study/__init__.py",
    "Publication_Study/vbb_study/setup_study.py",
    "Publication_Study/vbb_study/publication/quicklook.py",
    "Publication_Study/vbb_study/publication/visuals.py",
    "Publication_Study/vbb_study/study_taxonomy.py",
    # Root-level compatibility shims.
    "bessel_twin_core.py",
    "vbb_study/__init__.py",
]


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the repository root by walking upward from ``start``.

    Anchors on ``Publication_Study/bessel_twin_core.py``, which is the
    authoritative scalar-physics source.  The root-level compatibility shim
    is also tolerated for older checkouts.
    """

    here = Path.cwd() if start is None else Path(start).resolve()
    if here.is_file():
        here = here.parent
    for candidate in (here, *here.parents):
        if (candidate / "Publication_Study" / "bessel_twin_core.py").exists():
            return candidate
        if candidate.name == "Publication_Study" and (candidate / "bessel_twin_core.py").exists():
            return candidate.parent
        if (candidate / "bessel_twin_core.py").exists() and (candidate / "Publication_Study").exists():
            return candidate
    raise FileNotFoundError("Could not find repository root containing Publication_Study/bessel_twin_core.py")


def bootstrap(
    start: str | Path | None = None,
    *,
    apply_plot_style: bool = True,
) -> dict[str, Path]:
    """Add study source folders to ``sys.path`` and return canonical paths.

    Returns
    -------
    dict
        Paths for source, documentation, and generated-output locations. The
        ``modular_lab`` key is a compatibility alias for legacy notebooks; it
        points at ``reference_kernels`` rather than an external lab folder.
    """

    root = find_repo_root(start)
    publication = root / "Publication_Study"
    reference_kernels = publication / "reference_kernels"
    outputs = publication / "outputs"
    paths = {
        # Source/workspace roots.
        "root": root,
        "publication": publication,
        "reference_kernels": reference_kernels,
        "modular_lab": reference_kernels,
        # Generated-output contract. These folders may be removed only by the
        # runner's explicit --clean-output option.
        "outputs": outputs,
        "figures": outputs / "figures",
        "csv": outputs / "csv",
        "holograms": outputs / "holograms",
        "manifests": outputs / "manifests",
        "jupyter_runtime": outputs / "jupyter_runtime",
        # Compatibility pointer for older exploratory notebooks. Publication
        # notebooks should prefer paths["outputs"] and paths["docs"].
        "root_outputs": root / "outputs",
        "docs": publication / "docs",
    }
    for key in ("root", "publication"):
        path_text = str(paths[key])
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
    # Expose run_id from the runner environment so notebooks can stamp outputs.
    # Notebooks should use paths["run_id"] when calling annotate_scalar_row.
    import os as _os
    paths["run_id"] = _os.environ.get("STRUCTURED_BEAM_RUN_ID", "")
    if apply_plot_style:
        from . import vbb_style

        vbb_style.apply_style()
    return paths


def _missing(paths: Mapping[str, Path], key: str, names: list[str]) -> list[str]:
    base = Path(paths[key])
    return [str(base / name) for name in names if not (base / name).exists()]


def validate_workspace(paths: Mapping[str, Path], strict: bool = True) -> dict[str, list[str]]:
    """Check that the publication workspace has its expected source skeleton.

    The validation is intentionally about reproducibility plumbing, not physics
    correctness: it checks that canonical notebooks, docs, active source files,
    and compatibility shims exist. Generated scientific outputs are not
    required because a fresh checkout may not include them.

    Parameters
    ----------
    paths:
        Mapping returned by :func:`bootstrap`.
    strict:
        When true, raise ``FileNotFoundError`` with a grouped message if any
        required item is missing. When false, only return the report.
    """

    root = Path(paths["root"])
    report = {
        "missing_notebooks": _missing(paths, "publication", REQUIRED_NOTEBOOKS),
        "missing_docs": _missing(paths, "docs", REQUIRED_DOCS),
        "missing_source_files": [str(root / name) for name in REQUIRED_SOURCE_FILES if not (root / name).exists()],
    }
    if strict and any(report.values()):
        lines = ["Publication_Study workspace validation failed:"]
        for label, items in report.items():
            if items:
                lines.append(f"- {label}:")
                lines.extend(f"  {item}" for item in items)
        raise FileNotFoundError("\n".join(lines))
    return report


def _jsonable(value: Any) -> Any:
    """Convert common research objects into stable JSON-like values."""

    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def config_hash(config: Any) -> str:
    """Return a short deterministic hash for a config-like object."""

    payload = json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def code_version(root: str | Path | None = None) -> str | None:
    """Return the current git commit when git is available."""

    try:
        repo = find_repo_root(root)
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    commit = proc.stdout.strip()
    return commit or None


def stage1_engine_git_fields(root: str | Path | None = None) -> dict[str, str | None]:
    """Return Stage 1 baseline provenance fields for future captures."""

    commit = code_version(root)
    if commit:
        return {
            "engine_git_commit": commit,
            "engine_git_commit_note": "recorded from git rev-parse HEAD",
        }
    return {
        "engine_git_commit": None,
        "engine_git_commit_note": "unavailable: git rev-parse HEAD returned no commit",
    }


def run_manifest(
    *,
    config: Any = None,
    paths: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Build the minimal manifest I want next to generated artefacts.

    The manifest records timestamp, config hash, Python/runtime details, code
    version when available, and output paths. It deliberately does not invent
    any metric values.
    """

    repo = find_repo_root(root)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo),
        "code_version": code_version(repo),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "config_hash": config_hash(config) if config is not None else None,
        "paths": _jsonable(dict(paths or {})),
        "extra": _jsonable(dict(extra or {})),
    }


def write_run_manifest(
    manifest_path: str | Path,
    *,
    config: Any = None,
    paths: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
) -> Path:
    """Write a JSON run manifest and return the path."""

    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = run_manifest(config=config, paths=paths, extra=extra, root=root)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


__all__ = [
    "REQUIRED_DOCS",
    "REQUIRED_NOTEBOOKS",
    "REQUIRED_SOURCE_FILES",
    "bootstrap",
    "code_version",
    "config_hash",
    "find_repo_root",
    "run_manifest",
    "stage1_engine_git_fields",
    "validate_workspace",
    "write_run_manifest",
]
