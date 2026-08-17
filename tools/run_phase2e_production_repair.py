from __future__ import annotations

import argparse
import json
from pathlib import Path

from vbb_study.digital_twin.phase2e_production_repair import run_production_repair


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Phase 2E nominal no-additional-aperture production repair checks."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/validation/phase2e_production_repair"),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["B0", "V1", "V3"],
    )
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=[2048, 2560, 3072],
        help="Fixed-10-mm-window transverse source-grid sizes.",
    )
    parser.add_argument(
        "--z-mm",
        type=float,
        nargs="+",
        default=[20.0, 40.0, 60.0, 80.0, 100.0],
        help="Selected source-scale free-space propagation distances in mm.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_production_repair(
        output_root=args.output_root,
        cases=tuple(args.cases),
        n_values=tuple(args.n_values),
        z_values_m=tuple(value * 1e-3 for value in args.z_mm),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
