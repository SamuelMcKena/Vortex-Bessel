from __future__ import annotations

import json
from pathlib import Path

from vbb_study.digital_twin import MODE2U_DEFAULT_OUTPUT_ROOT, write_mode2u_master_highres_audit


ROOT = Path(MODE2U_DEFAULT_OUTPUT_ROOT)


def _ensure_outputs() -> Path:
    if not (ROOT / "nathan_master_visual_audit.json").exists():
        write_mode2u_master_highres_audit(
            output_dir=ROOT,
            grid_n=192,
            z_planes=5,
            run_compensation=False,
            optimisation_max_cases=8,
            run_high_n=False,
        )
    return ROOT


def test_mode2s_highres_visual_audit_has_sweep_contact_sheets() -> None:
    root = _ensure_outputs()
    m2s = root / "04_m2s_lab_realism"

    for name in [
        "m2s_hv_piston_contact_sheet_highres.png",
        "m2s_qwp_angle_contact_sheet_highres.png",
        "m2s_iris_decentre_contact_sheet_highres.png",
        "m2s_hv_shift_contact_sheet_highres.png",
        "m2s_axicon_decentre_contact_sheet_highres.png",
        "m2s_combined_and_compensated_contact_sheet_highres.png",
    ]:
        assert (m2s / name).exists()


def test_mode2s_highres_visual_audit_records_bad_case_as_failure() -> None:
    root = _ensure_outputs()
    audit = json.loads((root / "nathan_master_visual_audit.json").read_text(encoding="utf-8"))
    bad_rows = [row for row in audit if "combined_bad_lab" in str(row["case_id"])]

    assert bad_rows
    assert all(row["acceptable_hexagon"] is False for row in bad_rows)
    assert any(str(row["strict_class"]) in {"triangular_lobed_field", "triangular_dark_core"} for row in bad_rows)


def test_mode2s_highres_visual_audit_keeps_source_scale_claim_boundary() -> None:
    root = _ensure_outputs()
    manifest = json.loads((root / "nathan_master_highres_manifest.json").read_text(encoding="utf-8"))

    assert manifest["source_scale_branch_validated"] is True
    assert manifest["microfabrication_branch_validated"] is False
    assert manifest["six_polarizer_route_needed"] is False
