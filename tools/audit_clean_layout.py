"""Audit active source for stale assumptions from the old Publication_Study layout.

The audit protects code that should run directly from the standalone
Vortex-Bessel checkout.  It is depth-aware: ``parents[2]`` is correct for a
module two directories below the repository root, but incorrect for a module
immediately inside ``vbb_study``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# These files intentionally mention the legacy workspace in compatibility prose
# or in the compatibility branch of the bootstrap logic.
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

PARENTS_INDEX = re.compile(r"\.parents\[(\d+)\]")
PARENT_CHAIN = re.compile(r"(?:\.parent){2,}")


def active_python_files() -> list[Path]:
    paths = [path for path in CRITICAL_ROOT_FILES if path.exists()]
    paths.extend(sorted((ROOT / "vbb_study").rglob("*.py")))
    paths.extend(sorted((ROOT / "notebooks" / "experimental").rglob("*.py")))
    return list(dict.fromkeys(paths))


def _directory_depth(path: Path) -> int:
    """Return the number of directories between ROOT and ``path``."""

    return len(path.relative_to(ROOT).parts) - 1


def _fixed_depth_errors(path: Path, text: str) -> list[str]:
    """Find expressions that walk above the standalone repository root."""

    depth = _directory_depth(path)
    errors: list[str] = []

    # For a root file, parents[0] is ROOT.  For vbb_study/file.py, parents[1]
    # is ROOT, etc.  Larger indices walk above the checkout and are almost
    # always remnants of the old extra Publication_Study nesting level.
    for match in PARENTS_INDEX.finditer(text):
        index = int(match.group(1))
        if index > depth:
            errors.append(f"{match.group(0)} walks above repository root (depth={depth})")

    # Count explicit .parent.parent... chains using the same rule.  One .parent
    # from a root-level file reaches ROOT, depth+1 parents is therefore the
    # maximum chain that still lands on ROOT.
    for match in PARENT_CHAIN.finditer(text):
        count = match.group(0).count(".parent")
        max_to_root = depth + 1
        if count > max_to_root:
            errors.append(
                f"{match.group(0)} walks above repository root "
                f"(chain={count}, maximum_to_root={max_to_root})"
            )

    return errors


def audit() -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for path in active_python_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(ROOT).as_posix()

        if "Publication_Study" in text and path not in ALLOW_LEGACY_NAME:
            warnings.append(f"{rel}: contains legacy workspace name 'Publication_Study'")

        for detail in _fixed_depth_errors(path, text):
            errors.append(f"{rel}: {detail}")

    return errors, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-legacy-name", action="store_true")
    args = parser.parse_args(argv)

    files = active_python_files()
    errors, warnings = audit()
    print(f"active source files checked: {len(files)}")
    for item in warnings:
        print(f"WARNING: {item}")
    for item in errors:
        print(f"ERROR: {item}")

    if args.strict_legacy_name and warnings:
        return 1
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
