"""Render the full axicon presentation set using a sweep-selected hollow radius."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import build_axicon_hollow_avoidance as avoid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-n", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hollow-core-um", type=float, required=True)
    args = parser.parse_args()

    avoid.HOLLOW_CORE_M = float(args.hollow_core_um) * 1e-6
    sys.argv = [
        "build_axicon_hollow_avoidance.py",
        "--grid-n", str(int(args.grid_n)),
        "--output-dir", str(args.output_dir),
    ]
    avoid.main()


if __name__ == "__main__":
    main()
