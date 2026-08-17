"""Validate a Phase 2D calibration JSON bundle and print claim readiness."""

from __future__ import annotations

import argparse
from pathlib import Path

from vbb_study.calibration.io import load_calibration_bundle
from vbb_study.calibration.validation import calibration_readiness_for_claim, validate_calibration_bundle
from vbb_study.solver_policy import CLAIM_TYPES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    bundle = load_calibration_bundle(args.bundle)
    report = validate_calibration_bundle(bundle)
    print(f"schema_version: {bundle.schema_version}")
    print(f"valid_schema: {report.valid_schema}")
    print(f"data_classification: {bundle.data_classification}")
    print("missing_required_fields:")
    for item in report.missing_values:
        print(f"  - {item}")
    print("inconsistent_units:")
    for item in report.inconsistent_units:
        print(f"  - {item}")
    print("physically_impossible_values:")
    for item in report.physically_impossible_values:
        print(f"  - {item}")
    enabled: list[str] = []
    blocked: list[str] = []
    for claim in CLAIM_TYPES:
        readiness = calibration_readiness_for_claim(bundle, claim)
        (enabled if readiness.ready else blocked).append(claim)
    print("claims_currently_enabled: " + (", ".join(enabled) if enabled else "none"))
    print("claims_currently_blocked: " + (", ".join(blocked) if blocked else "none"))
    print("operator_review_warnings:")
    for item in report.warnings:
        print(f"  - {item}")
    if report.errors:
        print("errors:")
        for item in report.errors:
            print(f"  - {item}")
    return 0 if report.valid_schema else 1


if __name__ == "__main__":
    raise SystemExit(main())
