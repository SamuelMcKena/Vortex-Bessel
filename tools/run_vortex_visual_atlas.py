from __future__ import annotations

import argparse
import json
from pathlib import Path

from vbb_study.digital_twin.vortex_visual_atlas import run_atlas_screening


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run repaired-route vortex Bessel parameter/aberration screening atlas.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/validation/vortex_visual_atlas"))
    parser.add_argument("--cases", nargs="+", default=["B0", "V1", "V3"])
    parser.add_argument("--grid-n", type=int, default=1536)
    parser.add_argument("--z-mm", type=float, default=60.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_atlas_screening(
        output_root=args.output_root,
        cases=tuple(args.cases),
        grid_n=int(args.grid_n),
        z_m=float(args.z_mm) * 1e-3,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
