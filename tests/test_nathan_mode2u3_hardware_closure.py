from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from vbb_study.digital_twin import (
    CANONICAL_OPERATING_POINT_ID,
    MODE2U3_DEFAULT_OUTPUT_ROOT,
    OLD_BEST_COMPROMISE_ID,
    STRICT_COMPROMISE_ID,
    assert_not_forbidden,
    camera_closure_rows,
    forbidden_operating_point_ids,
    hwp_requirement_rows,
    jones_axis_route_rows,
    physical_4f_rows,
    qwp_lab_axis_statement,
    resolve_axicon_index_scopes,
    resolve_m2u2_conflicts,
    resolve_slm_hardware,
    resolve_wavelength_scopes,
    write_mode2u3_hardware_closure,
)

ROOT = Path(MODE2U3_DEFAULT_OUTPUT_ROOT)


def _ensure_outputs() -> Path:
    manifest = ROOT / "07_final_status" / "nathan_mode2u3_manifest.json"
    if not manifest.exists():
        write_mode2u3_hardware_closure(output_dir=ROOT, grid_n=384, z_planes=9)
    return ROOT


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _json_load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_exact_slm_identity_is_not_silently_inferred_from_generic_metadata() -> None:
    slm, _ = resolve_slm_hardware()

    assert slm.model == "PLUTO-2.1 NIR-149"
    assert slm.provenance["model"] == "externally_supplied_lab_identity"
    assert "PLUTO" not in "HOLOEYE LCOS-NIR"  # generic twin record cannot supply the exact model
    assert "externally_supplied_lab_identity" in slm.provenance["make"]


def test_externally_supplied_lab_identity_is_provenance_labelled() -> None:
    root = _ensure_outputs()
    payload = _json_load(root / "00_slm" / "slm_hardware_closure.json")

    assert payload["external_lab_identity"]["provenance"] == "externally_supplied_lab_identity"
    model_rows = [r for r in payload["rows"] if r["field"] == "model"]
    assert model_rows
    assert all(r["provenance"] == "externally_supplied_lab_identity" for r in model_rows)


def test_unknown_phase_stroke_remains_unknown_without_documentary_evidence() -> None:
    slm, unknowns = resolve_slm_hardware()

    assert slm.phase_stroke_rad is None
    assert slm.provenance["phase_stroke_rad"] == "unknown"
    assert unknowns["unknown_fields"]["phase_stroke_rad"]["status"] == "unresolved_requires_calibration"
    root = _ensure_outputs()
    schema = _json_load(root / "01_phase_calibration" / "slm_phase_calibration_schema.json")
    assert schema["fabricated_values"] is False
    assert schema["record_fields"]["usable_phase_stroke_rad"] is None


def test_wavelength_conflict_is_explicitly_resolved_or_scoped() -> None:
    rows = resolve_wavelength_scopes()
    values = sorted({row["value_nm"] for row in rows})

    assert values == [1029.0, 1030.0]
    conflict = next(r for r in resolve_m2u2_conflicts() if r["parameter_family"] == "wavelength")
    assert conflict["status"] == "resolved_different_scopes"
    # No averaging: 1029.5 must not appear anywhere in the scope table.
    assert all(abs(row["value_nm"] - 1029.5) > 0.4 for row in rows)


def test_axicon_index_conflict_is_explicitly_resolved_or_scoped() -> None:
    rows = resolve_axicon_index_scopes()

    assert {row["value"] for row in rows} == {1.458, 1.5}
    assert all(row["resolution_status"] == "resolved_different_scopes" for row in rows)
    source_row = next(row for row in rows if row["value"] == 1.458)
    assert "source-scale" in source_row["scope"]
    conflict = next(r for r in resolve_m2u2_conflicts() if r["parameter_family"] == "axicon_refractive_index")
    assert conflict["status"] == "resolved_different_scopes"


def test_physical_4f_displacement_satisfies_lambda_f_carrier() -> None:
    rows = physical_4f_rows()
    for row in rows:
        expected_m = (row["wavelength_nm"] * 1e-9) * row["focal_length_m"] * (row["carrier_lpmm"] * 1e3)

        assert row["first_order_displacement_mm"] == pytest.approx(expected_m / 1e-3, rel=1e-12)


def test_fourier_iris_is_translated_into_physical_mm() -> None:
    rows = physical_4f_rows()
    recommended = next(r for r in rows if r["recommended_focal_length"] and abs(r["wavelength_nm"] - 1029.0) < 0.5)

    assert recommended["iris_radius_mm"] > 0.0
    assert recommended["iris_diameter_mm"] == pytest.approx(2.0 * recommended["iris_radius_mm"])
    # 0.40 x carrier separation at f = 300 mm, 1029 nm: ~0.77 mm radius.
    assert recommended["iris_radius_mm"] == pytest.approx(0.7718, abs=0.01)
    assert recommended["physically_plausible"] is True


def test_camera_scale_is_not_fabricated() -> None:
    rows = camera_closure_rows()

    assert rows
    assert all(row["value"] is None for row in rows)
    assert all(row["provenance"] == "unknown" for row in rows)
    assert all(row["status"] == "unresolved_requires_calibration" for row in rows)


def test_jones_axis_audit_records_reflection_flips_and_handedness() -> None:
    rows = jones_axis_route_rows()

    assert len(rows) >= 8
    for row in rows:
        assert row["reflection_flip"]
        assert row["handedness"]
        assert row["jones_basis"]
    slm_rows = [r for r in rows if "SLM" in r["stage"]]
    assert all("reflection" in str(r["reflection_flip"]) for r in slm_rows)


def test_qwp_minus_45_is_converted_into_explicit_lab_axis_language() -> None:
    statement = qwp_lab_axis_statement()

    text = statement["physical_statement"].lower()
    assert "fast axis" in text
    assert "clockwise" in text
    assert "lab-horizontal" in text or "horizontal" in text
    assert "looking" in statement["viewing_convention"].lower()
    assert statement["fast_or_slow"].startswith("beta marks the FAST axis")
    assert statement["deferrable_to_routine_calibration"] is True


def test_hwp_requirements_derived_from_slm_polarisation_compatibility() -> None:
    rows = hwp_requirement_rows()
    by_question = {row["question"]: row for row in rows}

    slm_v = by_question["polarisation reaching SLM-V"]
    assert "HWP" in str(slm_v["extra_component"])
    assert "director" in str(slm_v["requirement"]) or "director" in str(slm_v["answer"])
    input_hwp = by_question["input HWP before the PBS?"]
    assert str(input_hwp["answer"]).startswith("YES")
    qwp = by_question["is one final QWP sufficient?"]
    assert str(qwp["answer"]).startswith("YES")


def test_every_m2u2_conflict_receives_a_closure_status() -> None:
    rows = resolve_m2u2_conflicts()

    assert len(rows) >= 7
    allowed = {
        "resolved_same_physical_context",
        "resolved_different_scopes",
        "resolved_placeholder_removed",
        "unresolved_requires_measurement",
        "unresolved_missing_evidence",
    }
    assert all(row["status"] in allowed for row in rows)
    families = {row["parameter_family"] for row in rows}
    assert {"wavelength", "axicon_refractive_index", "4f_focal_length", "carrier_frequency",
            "camera_calibration", "slm_exact_model_and_phase_stroke"} <= families


def test_hardware_rebind_reruns_canonical_and_strict_compromise() -> None:
    root = _ensure_outputs()
    rows = _csv_rows(root / "06_rebind" / "resolved_hardware_rebind.csv")
    bindings = {row["binding"] for row in rows}
    ids = {row["candidate_id"] for row in rows}

    assert {"old_binding_1030nm", "resolved_binding_1029nm"} <= bindings
    assert CANONICAL_OPERATING_POINT_ID in ids
    assert STRICT_COMPROMISE_ID in ids
    assert "m2s_combined_moderate_lab" in ids
    assert "m2s_axicon_decentre_0p5mm_compensated" in ids


def test_discarded_old_non_hexagonal_optima_cannot_be_revived() -> None:
    forbidden = forbidden_operating_point_ids()

    assert OLD_BEST_COMPROMISE_ID in forbidden
    with pytest.raises(ValueError, match="strict hexagon gate"):
        assert_not_forbidden(OLD_BEST_COMPROMISE_ID)
    assert_not_forbidden(CANONICAL_OPERATING_POINT_ID)
    assert_not_forbidden(STRICT_COMPROMISE_ID)


def test_repaired_strict_hexagon_gate_is_used_in_hardware_rebind() -> None:
    root = _ensure_outputs()
    payload = _json_load(root / "06_rebind" / "resolved_hardware_rebind.json")

    assert "repaired M2U2-FIX strict hexagon" in payload["strict_gate_used"]
    assert "calibrated project-specific eligibility threshold" in payload["reference_drift_floor_note"]
    for row in payload["rows"]:
        assert "strict_hexagon_eligible" in row
        assert "corr_to_realistic_4f" in row
    new_canonical = next(
        r for r in payload["rows"]
        if r["binding"] == "resolved_binding_1029nm" and r["candidate_id"] == CANONICAL_OPERATING_POINT_ID
    )
    assert bool(new_canonical["strict_hexagon_eligible"]) is True


def test_m2v_is_authorised_only_under_m2u3_a() -> None:
    root = _ensure_outputs()
    outcome = _json_load(root / "07_final_status" / "m2u3_outcome_report.json")

    assert outcome["selected_outcome"] in outcome["allowed_outcomes"]
    assert outcome["m2v_authorised"] == (outcome["selected_outcome"] == "M2U3-A")
    assert "only under M2U3-A" in outcome["m2v_authorisation_condition"]


def test_no_microfabrication_sample_plane_success_is_claimed() -> None:
    root = _ensure_outputs()
    outcome = _json_load(root / "07_final_status" / "m2u3_outcome_report.json")
    manifest = _json_load(root / "07_final_status" / "nathan_mode2u3_manifest.json")

    assert outcome["microfabrication_sample_plane_claim"] is False
    assert manifest["microfabrication_sample_plane_claim"] is False
    assert "blocked" in outcome["micro_scale_note"]


def test_slm_unknowns_file_lists_calibration_routes_not_values() -> None:
    root = _ensure_outputs()
    unknowns = _json_load(root / "00_slm" / "slm_hardware_unknowns.json")

    for field, entry in unknowns["unknown_fields"].items():
        assert entry["status"] in {"unknown", "unresolved_requires_calibration", "externally_supplied_lab_identity"}
        assert entry["resolves_by"]
    assert "no HOLOEYE SDK" in unknowns["sdk_gui_evidence"]
