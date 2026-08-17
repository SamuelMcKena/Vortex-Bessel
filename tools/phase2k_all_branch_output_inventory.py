"""Inventory generated/scientific artifacts across every Git branch.

The current ``outputs/`` tree is only one view of repository history.  This
script enumerates every remote branch available in a full clone, records files
that look like generated numerical/visual evidence, and deduplicates them by
Git blob SHA.  It therefore exposes historical figures/tables that disappeared
from the current branch without pretending that they remain scientifically
valid.

The inventory is provenance only.  Every unique payload remains quarantined
until its producer/reference/convergence/calibration gates have passed.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import subprocess
from typing import Any


GENERATED_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".pdf", ".svg", ".gif", ".mp4",
    ".csv", ".json", ".jsonl", ".txt", ".npy", ".npz", ".html",
    ".ipynb", ".zip",
}

GENERATED_PATH_TOKENS = (
    "outputs/",
    "figures/",
    "results/",
    "validation/",
    "reports/",
    "presentation",
    "publication",
)


def _run(*args: str) -> str:
    return subprocess.check_output(args, text=True, encoding="utf-8", errors="replace")


def remote_branches() -> list[str]:
    refs = _run(
        "git",
        "for-each-ref",
        "--format=%(refname)",
        "refs/remotes/origin",
    ).splitlines()
    return sorted(
        ref
        for ref in refs
        if ref and not ref.endswith("/HEAD")
    )


def looks_generated(path: str) -> bool:
    lower = path.lower()
    suffix = Path(lower).suffix
    if suffix not in GENERATED_SUFFIXES:
        return False
    if lower.startswith(("archive/", ".github/")):
        # Archive payloads are still represented through the branches that
        # created them; skip bulk source archives from classification here.
        return False
    return any(token in lower for token in GENERATED_PATH_TOKENS) or suffix in {
        ".png", ".jpg", ".jpeg", ".pdf", ".gif", ".mp4", ".npz", ".npy"
    }


def branch_tree(ref: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = _run("git", "ls-tree", "-r", "-l", ref)
    for line in text.splitlines():
        # <mode> <type> <sha> <size>\t<path>
        try:
            meta, path = line.split("\t", 1)
            mode, obj_type, sha, size = meta.split(maxsplit=3)
        except ValueError:
            continue
        if obj_type != "blob" or not looks_generated(path):
            continue
        rows.append(
            {
                "branch_ref": ref,
                "branch": ref.removeprefix("refs/remotes/origin/"),
                "path": path,
                "blob_sha": sha,
                "bytes": None if size == "-" else int(size),
                "suffix": Path(path).suffix.lower(),
                "scientific_use_allowed_now": False,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["branch"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/validation/phase2k_truth_audit"),
    )
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    branches = remote_branches()
    rows: list[dict[str, Any]] = []
    for ref in branches:
        rows.extend(branch_tree(ref))
    write_csv(out / "all_branch_generated_artifact_occurrences.csv", rows)

    by_blob: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_blob[row["blob_sha"]].append(row)

    unique_rows: list[dict[str, Any]] = []
    for sha, group in sorted(by_blob.items()):
        paths = sorted({row["path"] for row in group})
        branches_for_blob = sorted({row["branch"] for row in group})
        size_values = [row["bytes"] for row in group if row["bytes"] is not None]
        representative = group[0]
        unique_rows.append(
            {
                "blob_sha": sha,
                "bytes": max(size_values) if size_values else None,
                "suffix": representative["suffix"],
                "occurrence_count": len(group),
                "branch_count": len(branches_for_blob),
                "path_count": len(paths),
                "representative_path": paths[0],
                "branches": ";".join(branches_for_blob),
                "paths": ";".join(paths),
                "scientific_use_allowed_now": False,
                "reason": "historical/current Git artifact; producer truth and calibration status not established by existence",
            }
        )
    write_csv(out / "all_branch_unique_generated_payloads.csv", unique_rows)

    current = "phase2k-mathematical-physics-output-audit"
    current_shas = {
        row["blob_sha"] for row in rows if row["branch"] == current
    }
    historical_only = [row for row in unique_rows if row["blob_sha"] not in current_shas]
    write_csv(out / "all_branch_historical_only_generated_payloads.csv", historical_only)

    summary = {
        "branch_count": len(branches),
        "generated_artifact_occurrence_count": len(rows),
        "unique_generated_payload_count_by_blob_sha": len(unique_rows),
        "unique_payloads_present_on_current_audit_branch": len(current_shas),
        "historical_only_unique_payload_count": len(historical_only),
        "unique_payload_bytes": int(sum(row["bytes"] or 0 for row in unique_rows)),
        "historical_only_unique_payload_bytes": int(sum(row["bytes"] or 0 for row in historical_only)),
        "scientific_use_allowed_now_count": 0,
        "policy": "branch-wide provenance inventory only; no payload is validated by this scan",
    }
    (out / "all_branch_generated_artifact_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
