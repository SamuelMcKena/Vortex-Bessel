"""Final output housekeeping for the structured-beam study.

The notebooks generate the science. This module records what is present under
``outputs/`` after a run, while avoiding the claim that old artifacts were
definitely produced by the latest run unless their timestamps tie them to the
current ``run_id``.

Manifests written by this module carry:
  - ``project_schema_version`` — top-level study schema version
  - ``source_schema_version`` — scalar CSV field schema version
  - ``run_id`` — run identifier
  - ``generated_at_utc`` — collection timestamp
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from vbb_study import vbb_style


def _project_schema_version() -> str:
    """Return the top-level study schema version; graceful if unavailable."""
    try:
        from run_study import PROJECT_SCHEMA_VERSION  # noqa: PLC0415
        return PROJECT_SCHEMA_VERSION
    except ImportError:
        return "unknown"


def _scalar_output_schema_version() -> str:
    """Return the scalar CSV field schema version; graceful if unavailable."""
    try:
        from vbb_study.publication.tables import SCALAR_OUTPUT_SCHEMA_VERSION  # noqa: PLC0415
        return SCALAR_OUTPUT_SCHEMA_VERSION
    except ImportError:
        return "unknown"


INVENTORY_SUFFIXES = {
    ".png",
    ".svg",
    ".pdf",
    ".csv",
    ".json",
    ".jsonl",
    ".html",
    ".txt",
    ".md",
    ".npy",
    ".npz",
}

HASH_MAX_BYTES = 50 * 1024 * 1024


def _caption_for(path: Path) -> str:
    """Return a conservative caption when a notebook did not write one itself."""

    title = path.stem.replace("_", " ")
    return (
        f"Publication study figure `{path.name}` ({title}). "
        "Axes, colorbars, display scaling, and units are shown in the figure "
        "or the corresponding notebook section."
    )


def _coerce_datetime(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_files(root: Path) -> Iterable[Path]:
    """Yield files below root while skipping protected runtime/temp folders."""

    def _ignore_error(_: OSError) -> None:
        return None

    for dirpath, _, filenames in os.walk(root, onerror=_ignore_error):
        directory = Path(dirpath)
        for filename in filenames:
            yield directory / filename


def _hash_file(path: Path, *, size: int, max_hash_bytes: int) -> str | None:
    if size > max_hash_bytes:
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _artifact_rows(
    paths: Iterable[Path],
    root: Path,
    *,
    run_id: str | None,
    run_started_at: datetime | None,
    max_hash_bytes: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(paths, key=lambda item: str(item).lower()):
        if path.is_file():
            try:
                stat = path.stat()
            except OSError:
                continue
            modified = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
            generated = bool(run_id and run_started_at and modified >= run_started_at)
            rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "suffix": path.suffix.lower(),
                    "bytes": stat.st_size,
                    "modified_utc": modified.isoformat(),
                    "sha256": _hash_file(path, size=stat.st_size, max_hash_bytes=max_hash_bytes),
                    "manifest_run_id": run_id or "",
                    "artifact_run_id": run_id if generated else "",
                    "generated_in_current_run": generated,
                    "run_association": "current_run" if generated else "preexisting_or_unknown",
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "path",
        "suffix",
        "bytes",
        "modified_utc",
        "sha256",
        "manifest_run_id",
        "artifact_run_id",
        "generated_in_current_run",
        "run_association",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finalize_outputs(
    output_root: str | Path,
    *,
    run_id: str | None = None,
    run_started_at: datetime | str | None = None,
    max_hash_bytes: int = HASH_MAX_BYTES,
) -> dict[str, Path]:
    """Write caption and artifact manifests for the output tree.

    Parameters
    ----------
    output_root:
        The ``Publication_Study/outputs`` folder.
    run_id:
        Optional current-run identifier from ``run_publication_study.py``.
    run_started_at:
        Optional run-start timestamp. Artifacts modified after this timestamp
        are tagged as tied to the current run; older files remain inventory
        entries only.
    """

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifests_dir = root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    started_at = _coerce_datetime(run_started_at)

    figures = sorted((path for path in _safe_files(root) if path.suffix.lower() == ".png"), key=lambda item: str(item).lower())
    central_manifest = manifests_dir / vbb_style.CAPTIONS_MANIFEST_NAME
    central_manifest.write_text("", encoding="utf-8")

    _proj_ver = _project_schema_version()
    _scalar_ver = _scalar_output_schema_version()

    for figure in figures:
        try:
            stat = figure.stat()
        except OSError:
            continue
        modified = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
        generated = bool(run_id and started_at and modified >= started_at)
        caption_path = vbb_style.caption_path_for(figure)
        if not caption_path.exists():
            caption_path.write_text(_caption_for(figure) + "\n", encoding="utf-8")
        record = {
            "artifact": str(figure.relative_to(root)),
            "figure": str(figure.relative_to(root)),
            "caption": str(caption_path.relative_to(root)),
            "caption_text": caption_path.read_text(encoding="utf-8").strip(),
            "figure_name_pattern": vbb_style.FIGURE_NAME_PATTERN,
            "saved_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_run_id": run_id,
            "artifact_run_id": run_id if generated else None,
            "generated_in_current_run": generated,
            "metadata": {
                "finalized_by": "Publication_Study.finalize_publication_outputs",
                "project_schema_version": _proj_ver,
                "source_schema_version": _scalar_ver,
                "run_association": "current_run" if generated else "preexisting_or_unknown",
            },
        }
        with central_manifest.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    artifact_paths = [
        path
        for path in _safe_files(root)
        if path.suffix.lower() in INVENTORY_SUFFIXES
    ]
    artifact_manifest = manifests_dir / "publication_artifact_manifest.csv"
    _write_csv(
        artifact_manifest,
        _artifact_rows(
            artifact_paths,
            root,
            run_id=run_id,
            run_started_at=started_at,
            max_hash_bytes=max_hash_bytes,
        ),
    )
    # Write a small JSON companion manifest with schema/run metadata.
    meta_manifest = manifests_dir / "finalize_meta.json"
    meta_manifest.write_text(
        json.dumps(
            {
                "finalized_by": "Publication_Study.finalize_publication_outputs",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id or "",
                "project_schema_version": _proj_ver,
                "source_schema_version": _scalar_ver,
                "output_root": str(root),
                "artifact_manifest": str(artifact_manifest),
                "captions_manifest": str(central_manifest),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "captions_manifest": central_manifest,
        "artifact_manifest": artifact_manifest,
        "meta_manifest": meta_manifest,
    }


__all__ = ["finalize_outputs"]
