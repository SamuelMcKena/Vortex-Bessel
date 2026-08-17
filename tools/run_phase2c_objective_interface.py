"""Generate the PHASE 2C vector-objective/interface benchmark artifacts."""

from __future__ import annotations

import argparse

from vbb_study.digital_twin.phase2c_objective_interface import (
    Phase2CConfig,
    generate_phase2c_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pupil-grid-n", type=int, default=1024)
    parser.add_argument("--no-publication-quality", action="store_true")
    args = parser.parse_args()
    result = generate_phase2c_outputs(
        Phase2CConfig(
            pupil_grid_n=args.pupil_grid_n,
            publication_quality=not args.no_publication_quality,
        )
    )
    print(result.outcome)
    print(result.outcome_reason)


if __name__ == "__main__":
    main()
