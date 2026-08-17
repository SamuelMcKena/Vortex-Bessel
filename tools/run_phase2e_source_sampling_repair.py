from __future__ import annotations

import argparse
import json
from pathlib import Path

from vbb_study.digital_twin.phase2e_source_sampling_repair import run_quick_audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the bounded Phase 2E source-sampling and route audit."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/validation/phase2e_source_sampling_repair"),
    )
    parser.add_argument(
        "--figure-root",
        type=Path,
        default=Path("outputs/figures/phase2e_source_sampling_repair"),
    )
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=[512, 768, 1024, 1536],
        help="Fixed-10-mm-window source-grid sizes for the quick audit.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_quick_audit(
        output_root=args.output_root,
        figure_root=args.figure_root,
        n_values=tuple(args.n_values),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
