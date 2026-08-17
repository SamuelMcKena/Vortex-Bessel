"""Inventory repository figure assets and figure-producing code for presentation review.

This is a metadata audit, not a scientific acceptance gate.  It enumerates tracked
figure-like assets across the repository and identifies Python sources that contain
rendering/export hooks.  The resulting CSV/JSON records make the presentation
selection auditable instead of relying on whichever PNG happens to be easy to find.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
from typing import Iterable

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional for metadata only
    Image = None


FIGURE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".svg", ".pdf", ".gif"}
RENDER_TOKENS = ("savefig", "plt.subplots", "plt.figure", "Image.save", "write_phase2", "render_")


def tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [root / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def classify(path: Path) -> str:
    text = path.as_posix().lower()
    name = path.name.lower()
    if "presentation_phase2i" in text:
        return "new_presentation_candidate"
    if "conference_workshop" in text:
        return "legacy_conference_candidate"
    if "phase2e_report_visualisation" in text:
        if any(token in name for token in ("canonical_propagation_primary", "transverse_snapshots")):
            return "primary_scalar_candidate"
        return "phase2e_diagnostic_or_appendix"
    if "phase2b_visual_diagnostics" in text:
        if any(token in text for token in ("02_xy_planes", "03_xz_yz_slices", "04_profiles")):
            return "primary_scalar_candidate"
        return "phase2b_diagnostic_or_appendix"
    if "phase2c" in text:
        return "vector_reference_appendix"
    if "phase2h" in text or "h1" in name or "nathan" in text:
        return "separate_vector_hexagonal_scope"
    if "publication_study" in text or "/stage_" in text or "/stage" in text or "archive" in text:
        return "historical_or_legacy"
    if "outputs/figures" in text:
        return "other_tracked_figure"
    return "other_figure_asset"


def raster_dimensions(path: Path) -> tuple[int | None, int | None]:
    if Image is None or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif"}:
        return None, None
    try:
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return None, None


def scan_renderers(files: Iterable[Path], root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in files:
        if path.suffix.lower() != ".py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        hits = sorted({token for token in RENDER_TOKENS if token in text})
        if not hits:
            continue
        records.append({
            "path": path.relative_to(root).as_posix(),
            "tokens": hits,
            "presentation_relevance_hint": (
                "high" if any(word in path.name.lower() for word in ("visual", "render", "figure", "propagation", "atlas")) else "review"
            ),
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/validation/presentation_figure_audit"))
    args = parser.parse_args()
    root = args.root.resolve()
    out = (root / args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    files = tracked_files(root)
    figure_records: list[dict[str, object]] = []
    for path in files:
        if path.suffix.lower() not in FIGURE_SUFFIXES or not path.exists():
            continue
        width, height = raster_dimensions(path)
        rel = path.relative_to(root)
        figure_records.append({
            "path": rel.as_posix(),
            "suffix": path.suffix.lower(),
            "bytes": int(path.stat().st_size),
            "width_px": width,
            "height_px": height,
            "classification": classify(rel),
        })

    figure_records.sort(key=lambda row: str(row["path"]))
    renderers = scan_renderers(files, root)
    renderers.sort(key=lambda row: str(row["path"]))

    csv_path = out / "tracked_figure_inventory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "suffix", "bytes", "width_px", "height_px", "classification"])
        writer.writeheader()
        writer.writerows(figure_records)

    renderer_path = out / "figure_renderer_inventory.json"
    renderer_path.write_text(json.dumps(renderers, indent=2) + "\n", encoding="utf-8")

    counts: dict[str, int] = {}
    for row in figure_records:
        key = str(row["classification"])
        counts[key] = counts.get(key, 0) + 1
    payload = {
        "outcome": "PRESENTATION-FIGURE-INVENTORY",
        "tracked_figure_assets": len(figure_records),
        "figure_producing_python_files": len(renderers),
        "classification_counts": dict(sorted(counts.items())),
        "inventory_csv": csv_path.relative_to(root).as_posix(),
        "renderer_inventory_json": renderer_path.relative_to(root).as_posix(),
        "note": "Classification is a presentation-review hint only; scientific authority is determined by phase sign-offs and evidence contracts.",
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
