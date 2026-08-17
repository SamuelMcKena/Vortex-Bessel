from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

import bessel_twin_core as bt
from vbb_study import vbb_regime


BASELINE_DIR = ROOT / "baselines"
NON_CASE_BASELINE_JSON = {"ENVIRONMENT.json", "PROVENANCE.json"}
EXPECTED_CASES = {
    "general_holographic_ideal",
    "general_holographic_lab",
    "general_physical_ideal",
    "general_physical_lab",
    "limits_holographic_ideal",
    "limits_holographic_lab",
    "limits_physical_ideal",
    "limits_physical_lab",
}
PHASE1_INTENTIONAL_PAYLOAD_PATHS = {
    "payload.metrics.objective_map_source",
    "payload.metrics.propagation_power_clipping_note",
    "payload.result_fields.metrics_config.physical_axicon.slm2_conjugate_mode",
    "payload.result_fields.sampling_report.propagation_power_clipping_note",
}


def _baseline_files() -> list[Path]:
    return sorted(
        path
        for path in BASELINE_DIR.glob("*.json")
        if path.name not in NON_CASE_BASELINE_JSON
    )


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


def _phase1_expected_config(baseline: Mapping[str, Any], config: bt.TwinConfig) -> dict[str, Any]:
    """Extend the immutable baseline schema with intentional Phase 1 fields."""

    expected = dict(baseline["config_flat"])
    route = str(baseline["case_definition"]["route"])
    expected["mapping_mode"] = "target_matched_inverse_design"
    expected["physical_axicon.allow_vortex_removal"] = route == "physical"
    expected["physical_axicon.slm2_conjugate_mode"] = (
        "full" if route == "physical" else "preserve_vortex"
    )
    assert config.mapping_mode == expected["mapping_mode"]
    return expected


def _project_to_baseline_schema(expected: Any, actual: Any) -> Any:
    """Retain every historical key while excluding separately tested additions."""

    if isinstance(expected, dict) and isinstance(actual, dict):
        return {
            key: _project_to_baseline_schema(value, actual[key])
            if key in actual
            else "<missing>"
            for key, value in expected.items()
        }
    if isinstance(expected, list) and isinstance(actual, list):
        projected = [
            _project_to_baseline_schema(left, right)
            for left, right in zip(expected, actual)
        ]
        return projected + actual[len(expected):]
    return actual


def _assert_phase1_additive_contract(result: Mapping[str, Any], route: str) -> None:
    design = result["design"]
    metrics = result["metrics"]
    volume = result["volume"]
    report = result["propagation_power_validity_report"]
    assert design.mapping_mode == "target_matched_inverse_design"
    assert metrics["mapping_mode"] == design.mapping_mode
    assert metrics["objective_map_source"] == design.objective_map_source
    assert metrics["objective_map_demag"] == design.objective_map_demag
    assert "Quantitative metrics require drift<=0.05" in metrics["propagation_power_clipping_note"]
    assert metrics["quantitative_metrics_valid"] == report["quantitative_metrics_valid"]
    assert metrics["propagation_power_drift_fraction"] == volume["propagation_power_drift_fraction"]
    expected_mode = "full" if route == "physical" else "preserve_vortex"
    assert result["metrics_config"].physical_axicon.slm2_conjugate_mode == expected_mode
    assert "Quantitative metrics require drift<=0.05" in result["sampling_report"]["propagation_power_clipping_note"]
    if route == "physical":
        assert result["axicon_metadata"]["vortex_preservation_contract"] == "explicitly_acknowledged_removal"


def _build_case(case_definition: Mapping[str, Any]) -> tuple[bt.TwinConfig, str, str]:
    preset = str(case_definition["preset"])
    regime = str(case_definition["regime"])
    route = str(case_definition["route"])
    variant = str(case_definition["variant"])
    config = replace(bt.default_config(preset), generation_method=route)
    config = vbb_regime.config_for_regime(config, regime)
    if route == "holographic":
        path = "ideal" if variant == "ideal" else "realistic"
    elif route == "physical":
        config = replace(
            config,
            physical_axicon=replace(
                config.physical_axicon,
                slm2_stroke_levels=case_definition["physical_slm2_stroke_levels"],
                slm2_conjugate_mode=str(case_definition["physical_slm2_conjugate_mode"]),
                allow_vortex_removal=True,
            ),
        )
        path = "ideal"
    else:
        raise ValueError(f"Unsupported route: {route!r}")
    case_id = f"{regime}_{route}_{variant}"
    assert path == case_definition["path"]
    return config, path, case_id


def _diff(expected: Any, actual: Any, prefix: str = "") -> list[tuple[str, Any, Any]]:
    if type(expected) is not type(actual):
        return [(prefix or "<root>", expected, actual)]
    if isinstance(expected, dict):
        out: list[tuple[str, Any, Any]] = []
        for key in sorted(set(expected) | set(actual)):
            name = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected:
                out.append((name, "<missing>", actual[key]))
            elif key not in actual:
                out.append((name, expected[key], "<missing>"))
            else:
                out.extend(_diff(expected[key], actual[key], name))
        return out
    if isinstance(expected, list):
        out = []
        if len(expected) != len(actual):
            out.append((f"{prefix}.length", len(expected), len(actual)))
        for index, (left, right) in enumerate(zip(expected, actual)):
            out.extend(_diff(left, right, f"{prefix}[{index}]"))
        return out
    if expected != actual:
        return [(prefix or "<root>", expected, actual)]
    return []


def _short(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, default=repr)
    if len(text) > 500:
        return text[:497] + "..."
    return text


def test_stage1_baseline_files_exist() -> None:
    case_ids = {path.stem for path in _baseline_files()}
    assert case_ids == EXPECTED_CASES


@pytest.mark.parametrize("baseline_path", _baseline_files(), ids=lambda path: path.stem)
def test_characterisation_lock_matches_baseline(baseline_path: Path) -> None:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    case_id = str(baseline["case_id"])
    config, path, rebuilt_case_id = _build_case(baseline["case_definition"])
    assert rebuilt_case_id == case_id

    expected_config = _phase1_expected_config(baseline, config)
    config_diffs = _diff(expected_config, _flatten_config(config), "config_flat")
    if config_diffs:
        key, expected, actual = config_diffs[0]
        pytest.fail(f"{case_id} config provenance changed at {key}: expected {_short(expected)}, got {_short(actual)}")

    result = bt.run_case(config, preset=baseline["case_definition"]["preset"], path=path, case_id=case_id)
    _assert_phase1_additive_contract(result, str(baseline["case_definition"]["route"]))
    actual_payload = _payload_from_result(result)
    projected_payload = _project_to_baseline_schema(baseline["payload"], actual_payload)
    payload_diffs = _diff(baseline["payload"], projected_payload, "payload")
    payload_diffs = [diff for diff in payload_diffs if diff[0] not in PHASE1_INTENTIONAL_PAYLOAD_PATHS]
    if payload_diffs:
        lines = [f"{case_id} changed at {len(payload_diffs)} exact key(s). First differences:"]
        for key, expected, actual in payload_diffs[:10]:
            lines.append(f"- {key}: expected {_short(expected)}, got {_short(actual)}")
        pytest.fail("\n".join(lines))
