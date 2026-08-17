from __future__ import annotations

import csv
import json
from pathlib import Path

from vbb_study.digital_twin import (
    MODE2U2F_DEFAULT_OUTPUT_ROOT,
    OLD_BEST_COMPROMISE_ID,
    REALISTIC_4F_REFERENCE_ID,
    V0_REFERENCE_ID,
    write_mode2u2_fix_strict_hexagon_optimisation,
)


ROOT = Path(MODE2U2F_DEFAULT_OUTPUT_ROOT)


def _ensure_outputs() -> Path:
    manifest = ROOT / "nathan_mode2u2f_strict_manifest.json"
    if not manifest.exists():
        write_mode2u2_fix_strict_hexagon_optimisation(
            output_dir=ROOT,
            grid_n=384,
            z_planes=9,
            search_max_cases=36,
            highres_grid_n=384,
            run_highres=True,
        )
    return ROOT


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_old_x_shaped_best_compromise_cannot_pass_new_strict_eligibility() -> None:
    root = _ensure_outputs()
    rows = _csv_rows(root / "old_optima_strict_audit.csv")
    old = next(row for row in rows if row["candidate_id"] == OLD_BEST_COMPROMISE_ID)

    assert old["strict_hexagon_eligible"] == "False"
    assert "realistic-4F reference" in old["strict_fail_reasons"]


def test_v0_and_realistic_dual_slm_4f_baseline_pass_calibration() -> None:
    root = _ensure_outputs()
    rows = _csv_rows(root / "hexagon_classifier_calibration.csv")
    by_id = {row["case_id"]: row for row in rows}

    assert by_id[V0_REFERENCE_ID]["strict_hexagon_eligible"] == "True"
    assert by_id[REALISTIC_4F_REFERENCE_ID]["strict_hexagon_eligible"] == "True"


def test_old_triangular_failure_and_h4_dominant_output_fail() -> None:
    root = _ensure_outputs()
    rows = _csv_rows(root / "hexagon_classifier_calibration.csv")
    by_id = {row["case_id"]: row for row in rows}

    assert by_id["old_triangular_mode1_failure_proxy"]["strict_hexagon_eligible"] == "False"
    assert by_id["synthetic_h4_fourfold_failure"]["strict_hexagon_eligible"] == "False"
    assert by_id["synthetic_h4_fourfold_failure"]["fourfold_x_veto"] == "True"


def test_full_field_dark_background_correlation_alone_cannot_create_eligibility() -> None:
    root = _ensure_outputs()
    rows = _csv_rows(root / "old_optima_strict_audit.csv")
    old = next(row for row in rows if row["candidate_id"] == OLD_BEST_COMPROMISE_ID)

    assert float(old["corr_full"]) > 0.98
    assert old["strict_hexagon_eligible"] == "False"


def test_noneligible_candidate_cannot_be_selected_as_final_optimum() -> None:
    root = _ensure_outputs()
    candidate_rows = _csv_rows(root / "strict_hexagon_candidates.csv")
    eligible = {row["candidate_id"] for row in candidate_rows if row["strict_hexagon_eligible"] == "True"}
    optima = _csv_rows(root / "strict_optima_summary.csv")

    assert optima
    for row in optima:
        assert row["candidate_id"] in eligible
        assert row["strict_hexagon_eligible"] == "True"


def test_useful_region_mask_is_fixed_across_candidates() -> None:
    root = _ensure_outputs()
    region = json.loads((root / "strict_useful_region_definition.json").read_text(encoding="utf-8"))
    rows = _csv_rows(root / "strict_hexagon_candidates.csv")

    assert region["region_id"] == "fixed_regular_hexagon_radius_2p65_v0_ring"
    assert len({row["P_useful_over_P_total"] for row in rows}) > 1
    assert "hex_radius_m" in region


def test_peak_metric_is_not_single_pixel_only() -> None:
    root = _ensure_outputs()
    rows = _csv_rows(root / "strict_hexagon_candidates.csv")

    assert rows[0]["peak_metric_definition"].startswith("mean intensity in the 3x3")
    assert "local_3x3_peak_mean" in rows[0]
    assert "single_pixel_peak" in rows[0]


def test_all_final_strict_optima_have_highres_figures() -> None:
    root = _ensure_outputs()
    for name in [
        "strict_best_shape_highres.png",
        "strict_best_peak_highres.png",
        "strict_best_useful_energy_highres.png",
        "strict_best_compromise_highres.png",
    ]:
        assert (root / name).exists()


def test_m2u3_authorisation_is_outcome_gated() -> None:
    root = _ensure_outputs()
    outcome = json.loads((root / "m2u2f_outcome_report.json").read_text(encoding="utf-8"))

    assert outcome["selected_outcome"] in outcome["allowed_outcomes"]
    if outcome["selected_outcome"] in {"M2U2F-A", "M2U2F-B"}:
        assert outcome["m2u3_authorised"] is True
    else:
        assert outcome["m2u3_authorised"] is False
