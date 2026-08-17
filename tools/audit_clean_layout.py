"""Audit active source for stale assumptions from the old Publication_Study layout.

This check is intentionally narrow.  It protects the code that should run from
the root of the standalone Vortex-Bessel repository while allowing provenance
documentation and explicitly compatibility-oriented files to mention the old
workspace name.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Files whose prose intentionally discusses the legacy layout.  Executable path
# logic inside setup_study.py is covered by direct behavioural tests instead.
ALLOW_LEGACY_NAME = {
    ROOT / "vbb_study" / "setup_study.py",
    ROOT / "run_publication_study.py",
}

CRITICAL_ROOT_FILES = [
    ROOT / "run_study.py",
    ROOT / "run_publication_study.py",
    ROOT / "bessel_twin_core.py",
    ROOT / "publication_diagnostics.py",
    ROOT / "interface_correction_diagnosis.py",
    ROOT / "finalize_outputs.py",
    ROOT / "finalize_publication_outputs.py",
]

STALE_PARENT_PATTERNS = [
    re.compile(r"\.parents\[2\]"),
    re.compile(r"\.parents\[3\]"),
]


def active_python_files() -> list[Path]:
    paths = [path for path in CRITICAL_ROOT_FILES if path.exists()]
    paths.extend(sorted((ROOT / "vbb_study").rglob("*.py")))
    paths.extend(sorted((ROOT / "notebooks" / "experimental").rglob("*.py")))
    return list(dict.fromkeys(paths))


def audit() -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for path in active_python_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(ROOT).as_posix()

        if "Publication_Study" in text and path not in ALLOW_LEGACY_NAME:
            warnings.append(f"{rel}: contains legacy workspace name 'Publication_Study'")

        # A fixed parents[2]/parents[3] assumption was correct only for selected
        # historical nesting depths.  Current active source should derive its
        # workspace explicitly instead.
        for pattern in STALE_PARENT_PATTERNS:
            if pattern.search(text):
                errors.append(f"{rel}: contains stale fixed-depth path expression {pattern.pattern}")

    return errors, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-legacy-name", action="store_true")
    args = parser.parse_args(argv)

    errors, warnings = audit()
    print(f"active source files checked: {len(active_python_files())}")
    for item in warnings:
        print(f"WARNING: {item}")
    for item in errors:
        print(f"ERROR: {item}")

    if args.strict_legacy_name and warnings:
        return 1
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
