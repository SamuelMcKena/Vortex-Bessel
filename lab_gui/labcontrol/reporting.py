"""Human-readable Markdown reporting for formal sessions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_session_report(session_root: str | Path) -> Path:
    root = Path(session_root)
    session = json.loads((root / "session.json").read_text(encoding="utf-8"))
    lines = [
        f"# Experiment report — {session.get('experiment_label', 'Untitled experiment')}",
        "",
        f"- Session: `{session['session_id']}`",
        f"- Created: {session['created_utc']}",
        f"- Data kind: **{session.get('data_kind', 'UNKNOWN')}**",
        f"- Physical configuration: {session.get('physical_configuration', 'unrecorded')}",
        "",
        "## Measurements",
        "",
    ]
    for entry in session.get("trials", []):
        trial_path = root / entry["manifest"]
        trial: dict[str, Any] = json.loads(trial_path.read_text(encoding="utf-8"))
        metrics = json.loads((trial_path.parent / trial["metrics_file"]).read_text(encoding="utf-8"))
        first = metrics["per_frame"][0] if metrics["per_frame"] else {"values": {}}
        values = first.get("values", {})
        lines.extend(
            [
                f"### {trial['trial_id']} — {trial.get('role', 'CURRENT')}",
                "",
                f"- Timestamp: {trial['created_utc']}",
                f"- Recipe: {trial.get('recipe') or 'manual capture'}",
                f"- Source data: **{trial['data_kind']}**",
                f"- z: {trial['camera'].get('z_mm')} mm ({trial['camera'].get('z_reference')})",
                f"- Camera: {trial['camera'].get('provider')} / {trial['camera'].get('settings_id')}",
                f"- Repeats: {trial['repeats']}",
                f"- Analysis: {trial['analysis_version']}",
                f"- Family: {first.get('family', 'unknown')}",
                f"- Total signal: {values.get('total_signal', 'n/a')}",
                f"- Centre (y, x): {first.get('centre_yx_px', 'n/a')} px",
                f"- Saturation fraction: {values.get('saturation_fraction', 'n/a')}",
                f"- Phase hashes: `{trial['phase_hashes']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Scientific interpretation guardrail",
            "",
            "Beam-walk slopes are beam-camera relative unless the camera travel axis is independently calibrated. "
            "A sensorless result is an accepted correction command, not automatically a measured physical aberration.",
            "",
        ]
    )
    output = root / "reports" / "session_report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
