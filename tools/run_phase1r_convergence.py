"""Run the bounded Phase 1R convergence-recovery campaign.

The campaign is intentionally selective. It covers the canonical scalar and
physical routes, the two report shortlist controls, a near-threshold wide-core
case, and two deliberately difficult high-k_r/long-zone cases. Historical
sweep points outside this bounded set remain blocked rather than being inferred
valid from a neighbouring case.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import bessel_twin_core as bt
from vbb_study import vbb_regime
from vbb_study.viz_fields import phase_winding


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "validation" / "phase1_reconciliation"
OUT.mkdir(parents=True, exist_ok=True)

POWER_DRIFT_MAX = 0.05
CONVERGENCE_TOLERANCES = {
    "ring_or_feature_radius": 0.03,
    "canonical_zone": 0.05,
    "strict_region": 0.05,
    "peak_intensity": 0.05,
    "side_lobe_ratio": 0.05,
}


@dataclass(frozen=True)
class RecoveryCase:
    case_id: str
    build: Callable[[], object]
    path: str
    resolutions: tuple[int, ...]
    rationale: str


def _base_case(
    *,
    ell: int = 3,
    core_um: float = 3.0,
    length_um: float = 150.0,
    method: str = "holographic",
    regime: str = "general",
):
    base = bt.default_config("fast")
    base = replace(
        base,
        generation_method=method,
        target=replace(
            base.target,
            ell=ell,
            target_core_diameter_m=core_um * bt.um,
            target_bessel_length_m=length_um * bt.um,
        ),
    )
    if regime in {"general", "limits"}:
        base = replace(base, regime=regime)
    return base


CASES = (
    RecoveryCase(
        "general_holographic_ideal",
        lambda: _base_case(),
        "ideal",
        (384, 512, 768, 1024),
        "canonical ideal scalar route",
    ),
    RecoveryCase(
        "general_holographic_realistic",
        lambda: _base_case(),
        "realistic",
        (512, 768, 1024),
        "canonical current holographic device route",
    ),
    RecoveryCase(
        "general_physical_ideal",
        lambda: _base_case(method="physical"),
        "ideal",
        (384, 512, 768, 1024),
        "repaired normal physical-axicon route",
    ),
    RecoveryCase(
        "limits_holographic_ideal",
        lambda: _base_case(core_um=2.0, length_um=300.0, regime="limits"),
        "ideal",
        (512, 768, 1024, 1536),
        "limits-regime high-k_r/long-zone control",
    ),
    RecoveryCase(
        "limits_physical_ideal",
        lambda: _base_case(core_um=2.0, length_um=300.0, method="physical", regime="limits"),
        "ideal",
        (512, 768, 1024, 1536),
        "limits-regime repaired physical route",
    ),
    RecoveryCase(
        "shortlist_ell0_core3_L150_realistic",
        lambda: _base_case(ell=0),
        "realistic",
        (512, 768, 1024),
        "historical scalar shortlist ell=0 control",
    ),
    RecoveryCase(
        "shortlist_ell5_core4_L200_realistic",
        lambda: _base_case(ell=5, core_um=4.0, length_um=200.0),
        "realistic",
        (512, 768, 1024),
        "historical scalar shortlist ell=5 control",
    ),
    RecoveryCase(
        "near_threshold_D8_L150_ideal",
        lambda: _base_case(core_um=8.0),
        "ideal",
        (384, 512, 768, 1024),
        "near-threshold wide-core sweep point",
    ),
    RecoveryCase(
        "extreme_D1_L800_ideal",
        lambda: _base_case(core_um=1.0, length_um=800.0),
        "ideal",
        (512, 768, 1024, 1536),
        "extreme high-k_r and long-zone stress point",
    ),
    RecoveryCase(
        "long_D3_L800_ideal",
        lambda: _base_case(core_um=3.0, length_um=800.0),
        "ideal",
        (512, 768, 1024, 1536),
        "long-zone stress point at canonical transverse scale",
    ),
)


def _rel_delta(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return float(abs(a - b) / max(abs(a), abs(b), bt.EPS))


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _run_one(
    case: RecoveryCase,
    n: int,
    method: str = "bl_asm",
    *,
    device_downsample: int | None = None,
) -> dict[str, object]:
    base = case.build()
    grid = replace(
        base.grid,
        N=n,
        ideal_N=n,
        device_downsample=(
            int(base.grid.device_downsample)
            if device_downsample is None
            else int(device_downsample)
        ),
        crop_pixels=min(int(base.grid.crop_pixels), n),
        label=f"phase1r_N{n}",
    )
    propagation = replace(base.propagation, method=method)
    config = replace(base, grid=grid, propagation=propagation, validity_on_violation="flag")
    started = time.perf_counter()
    try:
        result = bt.run_case(
            config,
            preset="fast",
            path=case.path,
            case_id=f"phase1r_{case.case_id}_{method}_N{n}",
        )
    except Exception as exc:  # retained as evidence, never silently dropped
        return {
            "case_id": case.case_id,
            "method": method,
            "N": n,
            "device_downsample": int(grid.device_downsample),
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "runtime_seconds": time.perf_counter() - started,
        }
    metrics = result["metrics"]
    design = result["design"]
    surface = result["surface_field"]
    winding = phase_winding(
        surface.Ex,
        surface.grid,
        float(design.vortex_main_ring_radius_m),
    )
    ring_or_feature = (
        float(metrics.get("ring_radius_um", np.nan))
        if int(design.ell) != 0
        else float(metrics.get("feature_radius_um", np.nan))
    )
    return {
        "case_id": case.case_id,
        "method": method,
        "N": n,
        "device_downsample": int(grid.device_downsample),
        "status": "completed",
        "runtime_seconds": time.perf_counter() - started,
        "path": case.path,
        "generation_method": str(config.generation_method),
        "ell": int(design.ell),
        "target_core_diameter_um": float(design.target_core_diameter_m / bt.um),
        "target_bessel_length_um": float(design.target_bessel_length_m / bt.um),
        "propagation_power_drift_fraction": float(metrics["propagation_power_drift_fraction"]),
        "power_drift_pass": bool(metrics["propagation_power_drift_fraction"] <= POWER_DRIFT_MAX),
        "ring_or_feature_radius_um": ring_or_feature,
        "canonical_zone_um": float(metrics.get("canonical_zone_um", np.nan)),
        "strict_bessel_region_um": float(metrics.get("strict_bessel_region_um", np.nan)),
        "peak_intensity_au": float(metrics.get("peak_in_plane", np.nan)),
        "side_lobe_ratio": float(metrics.get("side_to_core_peak_ratio", np.nan)),
        "measured_winding": float(winding),
        "winding_error": float(abs(winding - int(design.ell))),
        "winding_pass": bool(abs(winding - int(design.ell)) < 0.1),
    }


def _case_summary(case: RecoveryCase, rows: list[dict[str, object]]) -> dict[str, object]:
    completed = [row for row in rows if row.get("status") == "completed"]
    adequate = [row for row in completed if bool(row.get("power_drift_pass"))]
    final_pair = adequate[-2:] if len(adequate) >= 2 else []
    deltas: dict[str, float] = {}
    metric_map = {
        "ring_or_feature_radius": "ring_or_feature_radius_um",
        "canonical_zone": "canonical_zone_um",
        "strict_region": "strict_bessel_region_um",
        "peak_intensity": "peak_intensity_au",
        "side_lobe_ratio": "side_lobe_ratio",
    }
    if len(final_pair) == 2:
        for label, key in metric_map.items():
            deltas[label] = _rel_delta(float(final_pair[0][key]), float(final_pair[1][key]))
    metric_pass = bool(
        deltas
        and all(
            (not np.isfinite(delta)) or delta <= CONVERGENCE_TOLERANCES[label]
            for label, delta in deltas.items()
        )
    )
    winding_pass = bool(completed and all(bool(row.get("winding_pass")) for row in completed))
    recovered = bool(len(final_pair) == 2 and metric_pass and winding_pass)
    drifts = [float(row["propagation_power_drift_fraction"]) for row in completed]
    if recovered:
        likely_cause = "insufficient_spatial_window_causing_z_dependent_bl_asm_bandlimit_clipping"
    elif len(drifts) >= 2 and drifts[-1] < 0.5 * drifts[0]:
        likely_cause = "high_kr_or_long_zone_requires_window_beyond_practical_phase1r_limit"
    elif drifts:
        likely_cause = "persistent_bandlimit_or_retained_window_loss"
    else:
        likely_cause = "unknown_execution_failure"
    return {
        "case_id": case.case_id,
        "rationale": case.rationale,
        "resolutions": list(case.resolutions),
        "adequate_resolutions": [int(row["N"]) for row in adequate],
        "final_adequate_pair": [int(row["N"]) for row in final_pair],
        "metric_relative_deltas": deltas,
        "convergence_pass": recovered,
        "quantitative_status": "recovered" if recovered else "invalid_unconverged",
        "likely_cause": likely_cause,
    }


def main() -> None:
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for case in CASES:
        case_rows = []
        print(f"PHASE1R_CONVERGENCE_START {case.case_id}", flush=True)
        for n in case.resolutions:
            row = _run_one(case, n)
            rows.append(row)
            case_rows.append(row)
            print(
                f"  N={n} status={row.get('status')} "
                f"drift={row.get('propagation_power_drift_fraction', 'nan')}",
                flush=True,
            )
        summaries.append(_case_summary(case, case_rows))

    # Compare SAS only where it is explicitly allowed by the engine. Failure is
    # retained as evidence and cannot promote a BL-ASM case to recovered.
    sas_row = _run_one(CASES[0], 768, method="sas")
    rows.append(sas_row)

    pd.DataFrame(rows).to_csv(OUT / "phase1r_convergence_runs.csv", index=False)
    manifest = {
        "schema_version": "1.0.0",
        "power_drift_limit_fraction": POWER_DRIFT_MAX,
        "predeclared_metric_tolerances": CONVERGENCE_TOLERANCES,
        "selected_case_count": len(CASES),
        "recovered_case_count": sum(bool(row["convergence_pass"]) for row in summaries),
        "blocked_case_count": sum(not bool(row["convergence_pass"]) for row in summaries),
        "cases": summaries,
        "sas_comparison": sas_row,
        "runs_csv": "outputs/validation/phase1_reconciliation/phase1r_convergence_runs.csv",
    }
    manifest = _json_safe(manifest)
    (OUT / "phase1r_selected_convergence_results.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: manifest[key] for key in ("selected_case_count", "recovered_case_count", "blocked_case_count")}))


if __name__ == "__main__":
    main()
