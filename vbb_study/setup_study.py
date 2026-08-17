"""Workspace bootstrap, validation and manifest helpers.

The current repository is self-contained: its checkout root is the study
workspace.  Older development checkouts placed the same material beneath a
``Publication_Study/`` directory, so the bootstrap still recognises that layout
for compatibility, but no current code should require it.

The path contract returned by :func:`bootstrap` is deliberately stable:

* ``root``: Git checkout root;
* ``publication``: active study workspace (the same as ``root`` in this repo);
* ``vbb_study`` source and ``bessel_twin_core.py`` live in that workspace;
* ``docs`` and ``reference_kernels`` live directly beneath the workspace;
* ``outputs`` contains generated artefacts and governed validation records.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


# Minimum current notebook skeleton.  The complete runnable registry is
# discovered dynamically by run_study.py; this list exists only for workspace
# health checks and therefore must not contain deleted historical notebooks.
REQUIRED_NOTEBOOKS = [
    "notebooks/00_study_overview_and_conventions.ipynb",
    "notebooks/scalar/02_scalar_ideal_vs_lab_diagnostics.ipynb",
    "notebooks/scalar/03_scalar_robustness_and_self_healing.ipynb",
    "notebooks/scalar/04_scalar_parameter_sweeps.ipynb",
    "notebooks/scalar/05_scalar_validation_suite.ipynb",
    "notebooks/lab_realism/01_holographic_axicon_route.ipynb",
    "notebooks/lab_realism/02_physical_axicon_route.ipynb",
    "notebooks/lab_realism/03_holographic_vs_physical_axicon.ipynb",
    "notebooks/lab_realism/04_objective_pupil_and_first_order_filtering.ipynb",
    "notebooks/lab_realism/05_through_sample_interface.ipynb",
    "notebooks/lab_realism/06_full_source_to_sample_journey.ipynb",
    "notebooks/vector/01_vector_beam_theory_atlas.ipynb",
    "notebooks/vector/02_vector_ideal_vs_lab_case1.ipynb",
    "notebooks/vector/03_vector_hardware_routes.ipynb",
    "notebooks/vector/04_vector_arm_hexagon.ipynb",
    "notebooks/materials/01_material_proxy_fluence_and_thresholds.ipynb",
    "notebooks/materials/02_material_calibration_template.ipynb",
    "notebooks/materials/03_application_design_tables.ipynb",
    "notebooks/advanced/01_capsule_weld_feature_design.ipynb",
    "notebooks/advanced/02_hexagonal_polygonal_beams.ipynb",
    "notebooks/advanced/03_discrete_nfold_beams.ipynb",
    "notebooks/digital_twin/00_full_beam_to_write_cockpit_MVP.ipynb",
    "notebooks/experimental/axicon_aberration_correction/Bessel_zscan_alignment_correction.ipynb",
]

REQUIRED_DOCS = [
    "00_theory.md",
    "01_conventions.md",
    "04_model_limitations.md",
]

# Paths are relative to the active study workspace, not to an assumed parent
# directory called Publication_Study.
REQUIRED_SOURCE_FILES = [
    "bessel_twin_core.py",
    "run_study.py",
    "run_publication_study.py",
    "finalize_publication_outputs.py",
    "vbb_study/__init__.py",
    "vbb_study/setup_study.py",
    "vbb_study/publication/visuals.py",
    "vbb_study/study_taxonomy.py",
]


def _looks_like_current_workspace(candidate: Path) -> bool:
    return (
        (candidate / "bessel_twin_core.py").is_file()
        and (candidate / "vbb_study" / "__init__.py").is_file()
        and (candidate / "notebooks").is_dir()
    )


def _looks_like_legacy_parent(candidate: Path) -> bool:
    workspace = candidate / "Publication_Study"
    return (
        (workspace / "bessel_twin_core.py").is_file()
        and (workspace / "vbb_study" / "__init__.py").is_file()
    )


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the checkout root for either the clean or legacy layout."""

    here = Path.cwd() if start is None else Path(start).expanduser().resolve()
    if here.is_file():
        here = here.parent

    for candidate in (here, *here.parents):
        # In an old checkout, prefer the parent of Publication_Study so git
        # provenance still refers to the actual repository root.
        if candidate.name == "Publication_Study" and _looks_like_current_workspace(candidate):
            parent = candidate.parent
            if (parent / ".git").exists() or _looks_like_legacy_parent(parent):
                return parent
        if _looks_like_current_workspace(candidate):
            return candidate
        if _looks_like_legacy_parent(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not find a Vortex-Bessel workspace containing "
        "bessel_twin_core.py, vbb_study/ and notebooks/."
    )


def _workspace_from_root(root: Path) -> Path:
    legacy = root / "Publication_Study"
    if _looks_like_current_workspace(legacy):
        return legacy
    if _looks_like_current_workspace(root):
        return root
    raise FileNotFoundError(f"No active structured-beam workspace found beneath {root}")


def bootstrap(
    start: str | Path | None = None,
    *,
    apply_plot_style: bool = True,
) -> dict[str, Any]:
    """Add source folders to ``sys.path`` and return canonical workspace paths."""

    root = find_repo_root(start)
    workspace = _workspace_from_root(root)
    reference_kernels = workspace / "reference_kernels"
    outputs = workspace / "outputs"

    paths: dict[str, Any] = {
        "root": root,
        # Historical name retained as an API key: in Vortex-Bessel this points
        # to the checkout root itself.
        "publication": workspace,
        "reference_kernels": reference_kernels,
        "modular_lab": reference_kernels,
        "outputs": outputs,
        "figures": outputs / "figures",
        "csv": outputs / "csv",
        "holograms": outputs / "holograms",
        "manifests": outputs / "manifests",
        "jupyter_runtime": outputs / "jupyter_runtime",
        "root_outputs": root / "outputs",
        "docs": workspace / "docs",
        "run_id": os.environ.get("STRUCTURED_BEAM_RUN_ID", ""),
    }

    # Workspace first so imports always resolve to this repository rather than
    # a stale sibling checkout on PYTHONPATH.
    for path in (root, workspace):
        text = str(path)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)

    if apply_plot_style:
        from . import vbb_style

        vbb_style.apply_style()
    return paths


def _missing(base: Path, names: list[str]) -> list[str]:
    return [str(base / name) for name in names if not (base / name).exists()]


def validate_workspace(paths: Mapping[str, Any], strict: bool = True) -> dict[str, list[str]]:
    """Validate the minimum current source/notebook/document skeleton.

    This checks reproducibility plumbing only; it does not assert experimental
    calibration or scientific validity of generated results.
    """

    workspace = Path(paths["publication"])
    report = {
        "missing_notebooks": _missing(workspace, REQUIRED_NOTEBOOKS),
        "missing_docs": _missing(Path(paths["docs"]), REQUIRED_DOCS),
        "missing_source_files": _missing(workspace, REQUIRED_SOURCE_FILES),
    }
    if strict and any(report.values()):
        lines = ["Vortex-Bessel workspace validation failed:"]
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
    """Return baseline engine provenance fields for future captures."""

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
    """Build a minimal provenance manifest for generated artefacts."""

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
    """Write a JSON run manifest and return its path."""

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
