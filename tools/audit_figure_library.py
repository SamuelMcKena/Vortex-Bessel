"""Validate the curated figure library and its scientific-status metadata."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
MANIFEST = FIGURES / "manifest.csv"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff"}
FORBIDDEN_CANONICAL_TOKENS = ("pre_realign", "postcorrectionoutput1")


def main() -> int:
    errors: list[str] = []
    if not MANIFEST.is_file():
        print("FAIL: figures/manifest.csv is missing")
        return 1

    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    paths = [row.get("path", "") for row in rows]
    if len(paths) != len(set(paths)):
        errors.append("manifest contains duplicate paths")

    for row in rows:
        rel = row.get("path", "")
        if not rel:
            errors.append("manifest row has no path")
            continue
        path = ROOT / rel
        if not path.is_file():
            errors.append(f"manifest points to missing file: {rel}")
        low = rel.lower()
        if row.get("status") == "canonical" and any(token in low for token in FORBIDDEN_CANONICAL_TOKENS):
            errors.append(f"legacy/superseded figure marked canonical: {rel}")
        if "rejected_do_not_use" in row.get("source_path", "").lower() or "candidate" in path.name.lower():
            if row.get("status") == "canonical":
                errors.append(f"rejected/controller candidate marked canonical: {rel}")
        if row.get("evidence_type") == "hardware_preview_not_validated" and row.get("presentation_preferred") == "yes":
            errors.append(f"unvalidated hardware preview marked presentation preferred: {rel}")

    committed_images = {
        path.relative_to(ROOT).as_posix()
        for path in FIGURES.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    manifested_images = {
        rel for rel in paths if Path(rel).suffix.lower() in IMAGE_SUFFIXES
    }
    unlisted = sorted(committed_images - manifested_images)
    if unlisted:
        errors.append("committed canonical-library images missing from manifest: " + ", ".join(unlisted))

    if not (FIGURES / "legacy_output_index.csv").is_file():
        errors.append("historical Publication_Study image index is missing")

    print(f"manifest rows: {len(rows)}")
    print(f"committed canonical-library images: {len(committed_images)}")
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("PASS: canonical figure library is internally consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
