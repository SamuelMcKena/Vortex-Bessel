"""Run the isolated Phase 2E report visualisation build."""

from __future__ import annotations

import argparse
import json

from vbb_study.digital_twin.phase2e_report_pipeline import generate_phase2e_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace only the Phase 2E output root; accepted upstream roots remain protected",
    )
    args = parser.parse_args()
    report = generate_phase2e_outputs(overwrite=bool(args.overwrite))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
