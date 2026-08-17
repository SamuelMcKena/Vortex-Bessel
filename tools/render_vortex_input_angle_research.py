from __future__ import annotations

import argparse
import json
from pathlib import Path

from vbb_study.digital_twin.vortex_error_research_figures import render_input_beam_angle_study


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render literature-backed input-pointing diagnostics for the vortex-Bessel route."
    )
    parser.add_argument("--case", choices=["B0", "V1", "V3"], default="B0")
    parser.add_argument("--grid-n", type=int, default=1536)
    parser.add_argument(
        "--angles-mrad",
        type=float,
        nargs="+",
        default=[-1.0, -0.5, 0.0, 0.5, 1.0],
    )
    parser.add_argument(
        "--figure-root",
        type=Path,
        default=Path("outputs/figures/vortex_error_physics_rebuild"),
    )
    parser.add_argument(
        "--validation-root",
        type=Path,
        default=Path("outputs/validation/vortex_error_physics_rebuild"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = render_input_beam_angle_study(
        case_id=args.case,
        grid_n=int(args.grid_n),
        angles_rad=tuple(float(v) * 1e-3 for v in args.angles_mrad),
        figure_root=args.figure_root,
        validation_root=args.validation_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
