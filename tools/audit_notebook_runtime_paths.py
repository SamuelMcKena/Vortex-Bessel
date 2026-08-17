"""Check current notebooks for executable references to the old nested workspace.

Markdown may discuss the historical ``Publication_Study`` folder, but code cells
in the standalone Vortex-Bessel repository must not require importing from it or
writing beneath a sibling ``Publication_Study/`` directory.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks"

PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("legacy import", re.compile(r"^\s*(?:from|import)\s+Publication_Study(?:\.|\s|$)")),
    ("legacy Path()", re.compile(r"Path\(\s*[\"']Publication_Study[\"']\s*\)")),
    ("legacy output path", re.compile(r"[\"']Publication_Study/outputs(?:/|[\"'])")),
    ("legacy sys.path", re.compile(r"sys\.path.*Publication_Study")),
)


def notebook_code(path: Path) -> list[tuple[int, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cells: list[tuple[int, str]] = []
    for index, cell in enumerate(payload.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(str(part) for part in source)
        cells.append((index, str(source)))
    return cells


def audit() -> list[str]:
    findings: list[str] = []
    for path in sorted(NOTEBOOK_ROOT.rglob("*.ipynb")):
        rel = path.relative_to(ROOT).as_posix()
        try:
            cells = notebook_code(path)
        except Exception as exc:
            findings.append(f"{rel}: could not parse notebook: {type(exc).__name__}: {exc}")
            continue
        for cell_index, source in cells:
            for line_number, line in enumerate(source.splitlines(), start=1):
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                for label, pattern in PATTERNS:
                    if pattern.search(line):
                        findings.append(
                            f"{rel}: cell {cell_index}, line {line_number}: {label}: {line.strip()}"
                        )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="print findings but return success (useful while repairing a migration)",
    )
    args = parser.parse_args(argv)

    notebooks = list(NOTEBOOK_ROOT.rglob("*.ipynb"))
    findings = audit()
    print(f"notebooks checked: {len(notebooks)}")
    if not findings:
        print("no stale executable Publication_Study paths found")
        return 0
    for finding in findings:
        print(f"STALE: {finding}")
    return 0 if args.report_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
