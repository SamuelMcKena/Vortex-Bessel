"""Capture production-preset (paper) baselines for the Stage 5.5 lock.

Run from the Publication_Study directory:
  C:\\PhD\\.venv2\\Scripts\\python.exe tools\\capture_prod_baselines.py

Writes baselines_prod/<case_id>.json (same float.hex + sha256 scheme as
baselines/) plus baselines_prod/ENVIRONMENT.json and PROVENANCE.json.
Does NOT touch baselines/ or any engine files.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bessel_twin_core as bt
from vbb_study import vbb_regime

PRESET = "paper"
BASELINE_DIR = ROOT / "baselines_prod"
BASELINE_DIR.mkdir(exist_ok=True)

CASE_DEFINITIONS = [
    {"regime": "general", "route": "holographic", "variant": "ideal",
     "path": "ideal", "physical_slm2_stroke_levels": None, "physical_slm2_conjugate_mode": None},
    {"regime": "general", "route": "holographic", "variant": "lab",
     "path": "realistic", "physical_slm2_stroke_levels": None, "physical_slm2_conjugate_mode": None},
    {"regime": "general", "route": "physical", "variant": "ideal",
     "path": "ideal", "physical_slm2_stroke_levels": None, "physical_slm2_conjugate_mode": "full"},
    {"regime": "general", "route": "physical", "variant": "lab",
     "path": "ideal", "physical_slm2_stroke_levels": 256, "physical_slm2_conjugate_mode": "full"},
    {"regime": "limits", "route": "holographic", "variant": "ideal",
     "path": "ideal", "physical_slm2_stroke_levels": None, "physical_slm2_conjugate_mode": None},
    {"regime": "limits", "route": "holographic", "variant": "lab",
     "path": "realistic", "physical_slm2_stroke_levels": None, "physical_slm2_conjugate_mode": None},
    {"regime": "limits", "route": "physical", "variant": "ideal",
     "path": "ideal", "physical_slm2_stroke_levels": None, "physical_slm2_conjugate_mode": "full"},
    {"regime": "limits", "route": "physical", "variant": "lab",
     "path": "ideal", "physical_slm2_stroke_levels": 256, "physical_slm2_conjugate_mode": "full"},
]


def _encode(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _encode(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _encode(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        contiguous = np.ascontiguousarray(arr)
        return {
            "kind": "ndarray",
            "dtype": str(contiguous.dtype),
            "dtype_str": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "c_contiguous": bool(contiguous.flags.c_contiguous),
            "source_c_contiguous": bool(arr.flags.c_contiguous),
            "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return _encode(value.item())
    if isinstance(value, float):
        return {"kind": "float", "hex": float(value).hex()}
    if isinstance(value, bool):
        return {"kind": "bool", "value": bool(value)}
    if isinstance(value, int):
        return {"kind": "int", "value": int(value)}
    if value is None or isinstance(value, str):
        return value
    return {"kind": "repr", "value": repr(value)}


def _flat_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _flat_value(value.item())
    if isinstance(value, float):
        return float(value).hex()
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return [_flat_value(v) for v in value]
    return repr(value)


def _flatten_config(value: Any, prefix: str = "") -> dict[str, Any]:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, subvalue in sorted(value.items(), key=lambda item: str(item[0])):
            name = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_config(subvalue, name))
        return out
    return {prefix: _flat_value(value)}


def _payload_from_result(result: Mapping[str, Any]) -> dict[str, Any]:
    result_fields = {k: v for k, v in result.items() if k not in {"metrics", "design", "axicon_metadata"}}
    return {
        "metrics": _encode(result["metrics"]),
        "design": _encode(result["design"]),
        "axicon_metadata": _encode(result.get("axicon_metadata", {})),
        "result_fields": _encode(result_fields),
    }


def _build_case(case_def: dict) -> tuple:
    regime = str(case_def["regime"])
    route = str(case_def["route"])
    variant = str(case_def["variant"])
    path = str(case_def["path"])

    config = replace(bt.default_config(PRESET), generation_method=route)
    config = vbb_regime.config_for_regime(config, regime)
    if route == "physical":
        config = replace(
            config,
            physical_axicon=replace(
                config.physical_axicon,
                slm2_stroke_levels=case_def["physical_slm2_stroke_levels"],
                slm2_conjugate_mode=str(case_def["physical_slm2_conjugate_mode"]),
                allow_vortex_removal=True,
            ),
        )
    case_id = f"{regime}_{route}_{variant}"
    return config, path, case_id


def capture_all() -> None:
    capture_ts = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

    # ENVIRONMENT
    env = {
        "capture_utc": capture_ts,
        "preset": PRESET,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "sys_byteorder": sys.byteorder,
        "numpy_version": np.__version__,
    }
    (BASELINE_DIR / "ENVIRONMENT.json").write_text(
        json.dumps(env, indent=2), encoding="utf-8"
    )

    total_start = time.time()
    case_times = {}

    for case_def in CASE_DEFINITIONS:
        config, path, case_id = _build_case(case_def)
        print(f"  Running {case_id} ...", end="", flush=True)
        t0 = time.time()
        result = bt.run_case(config, preset=PRESET, path=path, case_id=case_id)
        elapsed = time.time() - t0
        case_times[case_id] = elapsed
        print(f" {elapsed:.1f}s")

        capture_info = {
            "created_utc": capture_ts,
            "encoder_version": "hex-float-array-sha256-v1",
            "exactness": "bit-exact for captured scalar values and array byte hashes",
            "array_hashing": "sha256(np.ascontiguousarray(a).tobytes())",
            "endianness_assumption": (
                "Hashes use native-endian NumPy bytes from this platform; "
                "dtype_str records byte order and ENVIRONMENT.json records sys.byteorder."
            ),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "sys_byteorder": sys.byteorder,
            "numpy_version": np.__version__,
        }
        baseline = {
            "capture": capture_info,
            "case_definition": {**case_def, "preset": PRESET},
            "case_id": case_id,
            "config_flat": _flatten_config(config),
            "engine_git_commit": "N/A",
            "engine_git_commit_note": "run without git introspection",
            "payload": _payload_from_result(result),
            "schema_version": "prod-v1",
        }
        out_path = BASELINE_DIR / f"{case_id}.json"
        out_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
        print(f"  Written: {out_path.name}")

    total_elapsed = time.time() - total_start

    # PROVENANCE
    prov = {
        "created_utc": capture_ts,
        "preset": PRESET,
        "total_wall_seconds": total_elapsed,
        "per_case_seconds": case_times,
        "note": (
            "Production-resolution (paper preset: N=2048, device_downsample=1, "
            "axial_points=181, ideal_N=1024, ideal_dx=0.18um) baselines. "
            "Physical full conjugation is explicitly acknowledged as an intentional "
            "zero-winding legacy diagnostic. This baseline locks that diagnostic "
            "behaviour bit-for-bit at production resolution."
        ),
    }
    (BASELINE_DIR / "PROVENANCE.json").write_text(
        json.dumps(prov, indent=2), encoding="utf-8"
    )
    print(f"\nDone. Total: {total_elapsed:.1f}s  ({total_elapsed/60:.1f}min)")
    print(f"Baselines written to: {BASELINE_DIR}")


if __name__ == "__main__":
    capture_all()
