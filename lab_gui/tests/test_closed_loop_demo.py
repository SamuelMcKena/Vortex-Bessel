from __future__ import annotations

import json
from pathlib import Path

from labcontrol.demo import run_closed_loop_demo


def test_closed_loop_demo_replays_metrics_walk_verification_and_readiness(tmp_path: Path) -> None:
    result = run_closed_loop_demo(tmp_path / "demo")
    assert result["data_kind"] == "SYNTHETIC"
    assert result["accepted"] is True
    assert result["sensorless"]["status"] == "ACCEPTED"
    assert result["recipe"]["status"] == "COMPLETE"
    assert result["readiness"]["ready"] is True
    assert result["readiness"]["scope"] == "synthetic demonstration only"
    assert result["slm_cast_status"] == "NOT_CAST_SYNTHETIC_DEMO"
    assert result["walk_comparison"]["classification"] == (
        "primarily common post-axicon or measurement geometry"
    )
    run_root = Path(result["run_root"])
    assert (run_root / "DEMO_REPORT.md").is_file()
    state = json.loads((run_root / "final_state.json").read_text(encoding="utf-8"))
    assert state["system"]["data_kind"] == "SYNTHETIC"
    assert state["slm2"]["accepted_correction_id"] == result["sensorless"]["run_id"]
