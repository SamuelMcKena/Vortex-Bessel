from __future__ import annotations

import argparse
import json
from pathlib import Path

from vbb_study.calibration.io import load_calibration_bundle
from vbb_study.experimental.bench_dataset import dataset_readiness, load_experimental_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Phase 2I measured-bench evidence readiness")
    parser.add_argument("--dataset", type=Path, required=True, help="experimental dataset manifest JSON")
    parser.add_argument("--calibration", type=Path, default=None, help="optional calibration bundle JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-verify-hashes", action="store_true")
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()

    dataset = load_experimental_dataset(args.dataset)
    calibration = None if args.calibration is None else load_calibration_bundle(args.calibration)
    readiness = dataset_readiness(
        dataset,
        calibration_bundle=calibration,
        verify_hashes=not args.no_verify_hashes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(readiness, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(readiness, indent=2))
    if args.require_ready and not readiness["absolute_calibrated_comparison_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
