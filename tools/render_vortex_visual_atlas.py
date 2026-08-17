from __future__ import annotations

import argparse
import json
from pathlib import Path

from vbb_study.digital_twin.vortex_visual_atlas_figures import run_visual_atlas_figures


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render vortex-Bessel visual atlas figures. Physical-error families are "
            "subject to the fidelity gates in vortex_error_reference_models."
        )
    )
    parser.add_argument(
        "--figure-root",
        type=Path,
        default=Path("outputs/figures/vortex_visual_atlas"),
    )
    parser.add_argument("--grid-n", type=int, default=3072)
    parser.add_argument("--cases", nargs="+", default=["B0", "V1", "V3"])
    parser.add_argument(
        "--family",
        choices=["parameter", "manufacturing", "aberration", "alignment"],
        default=None,
        help="Render only one sweep family.",
    )
    parser.add_argument(
        "--parameter",
        default=None,
        help="Render only one parameter within the selected family.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.family == "alignment" and args.parameter == "axicon_tilt_y_rad":
        raise SystemExit(
            "axicon_tilt_y_rad is blocked for report/atlas rendering on the physics-rebuild branch. "
            "The inherited rotated-thin-element OPD model is superseded as report evidence; "
            "a rotated-angular-spectrum/direct-diffraction validated backend is required first."
        )
    if args.family == "parameter" and args.parameter == "input_beam_angle_x_rad":
        raise SystemExit(
            "input_beam_angle_x_rad now has a dedicated research diagnostic because the Fourier "
            "grating order and fixed 4F iris must be shown explicitly. Run "
            "tools/render_vortex_input_angle_research.py instead."
        )
    result = run_visual_atlas_figures(
        figure_root=args.figure_root,
        grid_n=int(args.grid_n),
        cases=tuple(args.cases),
        family_filter=args.family,
        parameter_filter=args.parameter,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
