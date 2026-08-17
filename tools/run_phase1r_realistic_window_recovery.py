"""Supplement Phase 1R with a larger realistic focal-plane window."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from run_phase1r_convergence import CASES, OUT, _case_summary, _json_safe, _run_one


TARGETS = {
    "general_holographic_realistic",
    "shortlist_ell0_core3_L150_realistic",
    "shortlist_ell5_core4_L200_realistic",
}


def _is_ds2(value) -> bool:
    try:
        return bool(np.isfinite(float(value)) and int(float(value)) == 2)
    except (TypeError, ValueError):
        return False


def main() -> None:
    csv_path = OUT / "phase1r_convergence_runs.csv"
    manifest_path = OUT / "phase1r_selected_convergence_results.json"
    existing = pd.read_csv(csv_path).to_dict("records")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summaries = {row["case_id"]: row for row in manifest["cases"]}

    for case in CASES:
        if case.case_id not in TARGETS:
            continue
        supplement = [
            row
            for row in existing
            if row.get("case_id") == case.case_id
            and str(row.get("method")) == "bl_asm"
            and _is_ds2(row.get("device_downsample"))
        ]
        print(f"PHASE1R_REALISTIC_WINDOW_START {case.case_id}", flush=True)
        completed_n = {int(float(row["N"])) for row in supplement}
        for n in (1024, 1536, 2048):
            if n in completed_n:
                continue
            row = _run_one(case, n, device_downsample=2)
            existing.append(row)
            supplement.append(row)
            print(
                f"  N={n} ds=2 status={row.get('status')} "
                f"drift={row.get('propagation_power_drift_fraction', 'nan')}",
                flush=True,
            )
        supplement.sort(key=lambda row: int(float(row["N"])))
        summaries[case.case_id] = _case_summary(case, supplement)
        summaries[case.case_id]["resolutions"] = [
            {"N": 1024, "device_downsample": 2},
            {"N": 1536, "device_downsample": 2},
            {"N": 2048, "device_downsample": 2},
        ]
        summaries[case.case_id]["window_repair"] = (
            "device_downsample 4 -> 2 doubles the focal-plane side length; "
            "N then tests resolution at the repaired window"
        )

    manifest["cases"] = [summaries[case.case_id] for case in CASES]
    manifest["recovered_case_count"] = sum(
        bool(row["convergence_pass"]) for row in manifest["cases"]
    )
    manifest["blocked_case_count"] = sum(
        not bool(row["convergence_pass"]) for row in manifest["cases"]
    )
    pd.DataFrame(existing).to_csv(csv_path, index=False)
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "recovered_case_count": manifest["recovered_case_count"],
                "blocked_case_count": manifest["blocked_case_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
