"""Current-repository entry point for the Fu et al. vector-Bessel diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vbb_study.digital_twin.fu_oscillating_vector_bessel import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ell", type=int, default=3)
    parser.add_argument("--grid-n", type=int, default=1024)
    parser.add_argument("--ha1-period-um", type=float, default=160.0)
    parser.add_argument("--ha2-period-um", type=float, default=1200.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/digital_twin/fu_oscillating_vector_bessel"))
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, ell=args.ell, grid_n=args.grid_n,
                         d_m=args.ha1_period_um * 1e-6,
                         D_m=args.ha2_period_um * 1e-6), indent=2))


if __name__ == "__main__":
    main()
