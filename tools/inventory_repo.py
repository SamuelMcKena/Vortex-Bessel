"""Print a structured inventory of the Publication_Study workspace.

Useful for orientation after returning to the project, or for checking
that a refactor left the active tree in a known state.

Usage:

    python Publication_Study/tools/inventory_repo.py
    python Publication_Study/tools/inventory_repo.py --json
    python Publication_Study/tools/inventory_repo.py --stage notebooks
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

_here = Path(__file__).resolve().parent
_pub = _here.parent
_root = _pub.parent
for _p in (_root, _pub):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SKIP_DIRS = {
    "__pycache__",
    ".ipynb_checkpoints",
    ".pytest_cache",
    "pytest-cache-files",
    ".venv",
    "venv",
}


def _should_skip(name: str) -> bool:
    return name.startswith("pytest-cache-files") or name in SKIP_DIRS


def _size_kb(path: Path) -> float:
    try:
        return path.stat().st_size / 1024
    except OSError:
        return 0.0


def _gather(root: Path, *, max_depth: int = 6) -> list[dict]:
    results = []

    def _walk(dir_: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(dir_.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return
        for entry in entries:
            if _should_skip(entry.name):
                continue
            rel = entry.relative_to(root)
            if entry.is_dir():
                results.append({"type": "dir", "path": str(rel), "depth": depth})
                _walk(entry, depth + 1)
            else:
                results.append({
                    "type": "file",
                    "path": str(rel),
                    "depth": depth,
                    "ext": entry.suffix.lower(),
                    "size_kb": round(_size_kb(entry), 1),
                })

    _walk(root, 0)
    return results


def _category(item: dict) -> str:
    if item["type"] == "dir":
        return "directory"
    ext = item.get("ext", "")
    p = item["path"]
    if ext == ".ipynb":
        return "notebook"
    if ext == ".py":
        return "python"
    if ext == ".md":
        return "markdown"
    if ext in (".png", ".svg", ".pdf"):
        return "figure"
    if ext == ".csv":
        return "csv"
    if ext == ".json":
        return "json"
    if "archive" in p.replace("\\", "/"):
        return "archive"
    return "other"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def report(pub_root: Path, *, stage: str | None = None, as_json: bool = False) -> None:
    items = _gather(pub_root)

    # Filter by stage if requested
    if stage == "notebooks":
        items = [i for i in items if "notebook" in i["path"].replace("\\", "/") or i.get("ext") == ".ipynb"]
    elif stage == "source":
        items = [i for i in items if i.get("ext") == ".py"]
    elif stage == "docs":
        items = [i for i in items if "docs" in i["path"].replace("\\", "/")]
    elif stage == "outputs":
        items = [i for i in items if "outputs" in i["path"].replace("\\", "/")]
    elif stage == "archive":
        items = [i for i in items if "archive" in i["path"].replace("\\", "/")]

    if as_json:
        print(json.dumps(items, indent=2))
        return

    # Human-readable table
    notebooks = [i for i in items if i.get("ext") == ".ipynb"]
    py_files = [i for i in items if i.get("ext") == ".py"]
    md_files = [i for i in items if i.get("ext") == ".md"]
    dirs = [i for i in items if i["type"] == "dir"]

    print(f"\n[inventory] Publication_Study — {pub_root}")
    print(f"[inventory]   directories : {len(dirs)}")
    print(f"[inventory]   notebooks   : {len(notebooks)}")
    print(f"[inventory]   python files: {len(py_files)}")
    print(f"[inventory]   docs (md)   : {len(md_files)}")
    print()

    if not stage or stage == "notebooks":
        print("── Notebooks ──────────────────────────────────────────────")
        for nb in sorted(notebooks, key=lambda x: x["path"]):
            print(f"  {nb['path']:<70}  {nb['size_kb']:>8.1f} KB")

    if not stage or stage == "source":
        print("\n── Python source (active, non-archive) ────────────────────")
        active_py = [
            p for p in py_files
            if "archive" not in p["path"].replace("\\", "/")
            and "reference_kernels" not in p["path"].replace("\\", "/")
            and "__pycache__" not in p["path"].replace("\\", "/")
        ]
        for p in sorted(active_py, key=lambda x: x["path"]):
            print(f"  {p['path']:<70}  {p['size_kb']:>8.1f} KB")

    if not stage or stage == "docs":
        print("\n── Docs ────────────────────────────────────────────────────")
        doc_files = [i for i in md_files if "docs" in i["path"].replace("\\", "/")]
        for d in sorted(doc_files, key=lambda x: x["path"]):
            print(f"  {d['path']:<70}  {d['size_kb']:>8.1f} KB")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Print workspace inventory.")
    parser.add_argument(
        "--stage",
        choices=["notebooks", "source", "docs", "outputs", "archive"],
        default=None,
        help="Show only one category.",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON.")
    args = parser.parse_args()

    pub_root = _pub
    report(pub_root, stage=args.stage, as_json=args.json)


if __name__ == "__main__":
    main()
