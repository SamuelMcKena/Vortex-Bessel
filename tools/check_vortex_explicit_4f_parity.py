from __future__ import annotations

import argparse
import json

import numpy as np

from vbb_study.digital_twin.phase2a_canonical import _fourier_first_order
from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.vortex_system_route import SystemErrorConfig, build_system_route


EPS = np.finfo(float).tiny


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.complex128).ravel()
    bb = np.asarray(b, dtype=np.complex128).ravel()
    return float(abs(np.vdot(aa, bb)) / max(np.linalg.norm(aa) * np.linalg.norm(bb), EPS))


def intensity_correlation(a: np.ndarray, b: np.ndarray) -> float:
    ia = np.abs(np.asarray(a, dtype=complex)) ** 2
    ib = np.abs(np.asarray(b, dtype=complex)) ** 2
    aa = ia.ravel() - float(np.mean(ia))
    bb = ib.ravel() - float(np.mean(ib))
    return float(np.dot(aa, bb) / max(np.linalg.norm(aa) * np.linalg.norm(bb), EPS))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check nominal explicit 4F parity against accepted collapsed first-order filter.")
    parser.add_argument("--grid-n", type=int, default=1536)
    parser.add_argument("--cases", nargs="+", default=["B0", "V1", "V3"])
    args = parser.parse_args()

    manifest = canonical_hardware_manifest()
    carrier = float(hardware_value(manifest, "carrier_frequency_cpm"))
    iris_cpm = float(hardware_value(manifest, "fourier_iris_radius_cpm"))
    rows = []
    for case_id in args.cases:
        route = build_system_route(case_id, grid_n=int(args.grid_n), config=SystemErrorConfig())
        collapsed, eff = _fourier_first_order(
            route["post_slm2"],
            route["grid"],
            carrier,
            iris_cpm,
            0.0,
        )
        explicit = route["post_4f_selected_order"]
        rows.append({
            "case_id": case_id,
            "grid_n": int(args.grid_n),
            "field_overlap": overlap(explicit, collapsed),
            "intensity_correlation": intensity_correlation(explicit, collapsed),
            "collapsed_selected_fraction": float(eff),
            "explicit_iris_selected_fraction": float(route["metadata"]["fourf"]["iris_selected_power_fraction"]),
            "explicit_carrier_removal": route["metadata"]["selected_order_carrier_removal"],
        })

    outcome = {
        "outcome": "VORTEX-EXPLICIT-4F-PARITY",
        "report_error_sweeps_authorised": all(
            row["field_overlap"] >= 0.98 and row["intensity_correlation"] >= 0.99 for row in rows
        ),
        "acceptance": {
            "field_overlap_min": 0.98,
            "intensity_correlation_min": 0.99,
        },
        "rows": rows,
    }
    print(json.dumps(outcome, indent=2))
    if not outcome["report_error_sweeps_authorised"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
