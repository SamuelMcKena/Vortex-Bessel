"""JSON serialisation for Phase 2D calibration bundles."""

from __future__ import annotations

import json
from pathlib import Path

from vbb_study.calibration.schema import CalibrationBundle


def load_calibration_bundle(path: Path) -> CalibrationBundle:
    source = Path(path)
    if source.suffix.lower() != ".json":
        raise ValueError("Phase 2D calibration bundles currently require versioned JSON")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("calibration bundle root must be a JSON object")
    return CalibrationBundle(payload)


def dump_calibration_bundle(bundle: CalibrationBundle, path: Path) -> Path:
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        raise ValueError("Phase 2D calibration bundles currently require a .json destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(bundle.data, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return destination
