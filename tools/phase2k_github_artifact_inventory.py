"""Inventory every GitHub Actions artifact for the repository.

This is metadata/provenance inventory, not a physics validator.  It closes the
important gap between files committed under ``outputs/`` and large numerical
artifacts retained only by GitHub Actions.  The script paginates the complete
artifact API, records every artifact, groups duplicate names/digests, and emits
an explicit download plan for distinct non-expired artifact payloads.

It is intended to run in GitHub Actions with ``GITHUB_TOKEN`` and
``GITHUB_REPOSITORY`` available.  No artifact is declared scientifically valid
by this inventory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


API_VERSION = "2022-11-28"


def _request_json(url: str, token: str) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": "phase2k-mathematical-truth-audit",
        },
    )
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def fetch_all_artifacts(repository: str, token: str) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    page = 1
    while True:
        payload = _request_json(
            f"https://api.github.com/repos/{repository}/actions/artifacts?per_page=100&page={page}",
            token,
        )
        batch = list(payload.get("artifacts", []))
        artifacts.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return artifacts


def classify_name(name: str) -> str:
    n = str(name).lower()
    if "phase2k" in n:
        return "phase2k_audit_or_snapshot"
    if "presentation" in n:
        return "presentation_derivative"
    if "axicon" in n and "tip" in n:
        return "axicon_tip_study"
    if "axicon" in n and "alignment" in n:
        return "axicon_alignment_study"
    if "system-evidence" in n or "system_error" in n or "system-error" in n:
        return "system_error_study"
    if "propagation" in n or "profile" in n:
        return "propagation_or_profile_study"
    if "vector" in n or "debye" in n or "fresnel" in n:
        return "vector_or_interface_study"
    if "calibration" in n or "experimental" in n or "readiness" in n:
        return "calibration_or_experimental_closure"
    if "material" in n:
        return "material_proxy_study"
    if "hex" in n or "polygon" in n:
        return "polygonal_or_hexagonal_study"
    return "other"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["artifact_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/validation/phase2k_truth_audit"),
    )
    args = parser.parse_args()
    if not args.repository or not args.token:
        raise SystemExit("repository and GitHub token are required")

    artifacts = fetch_all_artifacts(args.repository, args.token)
    rows: list[dict[str, Any]] = []
    for item in artifacts:
        run = item.get("workflow_run") or {}
        rows.append(
            {
                "artifact_id": int(item["id"]),
                "name": str(item.get("name", "")),
                "classification": classify_name(str(item.get("name", ""))),
                "size_in_bytes": int(item.get("size_in_bytes", 0)),
                "expired": bool(item.get("expired", False)),
                "created_at": str(item.get("created_at", "")),
                "updated_at": str(item.get("updated_at", "")),
                "expires_at": str(item.get("expires_at", "")),
                "digest": str(item.get("digest", "")),
                "workflow_run_id": run.get("id"),
                "head_branch": str(run.get("head_branch", "")),
                "head_sha": str(run.get("head_sha", "")),
                "archive_download_url": str(item.get("archive_download_url", "")),
                "scientific_use_allowed_now": False,
                "reason": "Actions artifact must be opened, traced to its producer and pass Phase 2K truth gates before scientific reuse",
            }
        )

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "github_actions_artifact_inventory.csv", rows)

    # Distinct-payload plan: artifacts sharing a GitHub-provided digest can be
    # treated as byte-identical payloads for content-audit download purposes.
    by_digest: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = row["digest"] or f"no-digest:{row['artifact_id']}"
        by_digest[key].append(row)
    download_rows: list[dict[str, Any]] = []
    for digest, group in by_digest.items():
        candidates = [row for row in group if not row["expired"]]
        representative = max(candidates or group, key=lambda row: row["artifact_id"])
        download_rows.append(
            {
                "representative_artifact_id": representative["artifact_id"],
                "representative_name": representative["name"],
                "digest": digest,
                "duplicate_artifact_count": len(group),
                "payload_size_in_bytes": representative["size_in_bytes"],
                "expired": representative["expired"],
                "classification": representative["classification"],
                "head_branch": representative["head_branch"],
                "head_sha": representative["head_sha"],
                "content_audit_required": True,
            }
        )
    download_rows.sort(key=lambda row: (row["classification"], row["representative_name"], row["representative_artifact_id"]))
    _write_csv(out / "github_actions_distinct_payload_download_plan.csv", download_rows)

    name_counts = Counter(row["name"] for row in rows)
    class_counts = Counter(row["classification"] for row in rows)
    nonexpired = [row for row in rows if not row["expired"]]
    distinct_nonexpired = [row for row in download_rows if not row["expired"]]
    summary = {
        "repository": args.repository,
        "artifact_record_count": len(rows),
        "nonexpired_artifact_count": len(nonexpired),
        "expired_artifact_count": len(rows) - len(nonexpired),
        "total_recorded_bytes": int(sum(row["size_in_bytes"] for row in rows)),
        "total_nonexpired_bytes": int(sum(row["size_in_bytes"] for row in nonexpired)),
        "distinct_payload_count_by_digest": len(download_rows),
        "distinct_nonexpired_payload_count": len(distinct_nonexpired),
        "distinct_nonexpired_payload_bytes": int(sum(row["payload_size_in_bytes"] for row in distinct_nonexpired)),
        "classification_counts": dict(sorted(class_counts.items())),
        "artifact_name_counts": dict(sorted(name_counts.items())),
        "scientific_use_allowed_now_count": 0,
        "policy": "metadata inventory only; every Actions payload remains quarantined until content/producer validation",
    }
    (out / "github_actions_artifact_inventory_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
