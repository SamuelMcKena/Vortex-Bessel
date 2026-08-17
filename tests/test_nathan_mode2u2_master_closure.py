from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from vbb_study.digital_twin import (
    MODE2U2_DEFAULT_OUTPUT_ROOT,
    mode2u_visual_acceptability,
    native_panel_geometry,
    resolve_nathan_hardware_binding,
    write_mode2u2_master_closure,
)


ROOT = Path(MODE2U2_DEFAULT_OUTPUT_ROOT)


def _ensure_outputs() -> Path:
    manifest = ROOT / "nathan_mode2u2_master_manifest.json"
    if not manifest.exists():
        write_mode2u2_master_closure(
            output_dir=ROOT,
            grid_n=256,
            z_planes=5,
            optimisation_max_cases=8,
            interaction_samples=6,
        )
    return ROOT


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_hardware_provenance_exists_for_realistic_parameters() -> None:
    root = _ensure_outputs()
    rows = _csv_rows(root / "hardware_parameter_provenance.csv")
    names = {r["parameter_name"] for r in rows}

    for required in {
        "wavelength_m",
        "slm_resolution_x",
        "slm_resolution_y",
        "slm_pixel_pitch_m",
        "slm_phase_bits",
        "carrier_cycles_per_m",
        "iris_radius_cycles_per_m",
        "qwp_nominal_angle_rad",
        "axicon_refractive_index",
        "camera_pixel_pitch_m",
    }:
        assert required in names


def test_planning_assumptions_cannot_masquerade_as_measured_or_manufacturer_spec() -> None:
    root = _ensure_outputs()
    rows = _csv_rows(root / "hardware_parameter_provenance.csv")

    for row in rows:
        if row["category"] in {"planning_assumption", "placeholder", "unknown"}:
            assert row["measured_lab"] == "False"
            assert row["manufacturer_spec"] == "False"


def test_hardware_conflicts_are_explicitly_reported() -> None:
    root = _ensure_outputs()
    conflicts = json.loads((root / "hardware_parameter_conflicts.json").read_text(encoding="utf-8"))
    families = {row["parameter_family"] for row in conflicts}

    assert "axicon_refractive_index" in families
    assert "4f_focal_length" in families
    assert "camera_calibration" in families


def test_native_panel_slm_geometry_matches_binding_and_preserves_aspect_ratio() -> None:
    root = _ensure_outputs()
    binding = resolve_nathan_hardware_binding()
    geom = native_panel_geometry(binding)
    rows = json.loads((root / "native_panel_confirmation.json").read_text(encoding="utf-8"))

    assert geom["panel_width_px"] == 1920
    assert geom["panel_height_px"] == 1080
    assert geom["pixel_pitch_m"] == pytest.approx(8.0e-6)
    assert geom["active_aspect_ratio"] == pytest.approx(16.0 / 9.0)
    assert rows
    for row in rows:
        assert row["panel_width_px"] == 1920
        assert row["panel_height_px"] == 1080
        assert row["native_phase_rasterized_exact_panel"] is True
        assert row["native_mask_aspect_ratio"] == pytest.approx(16.0 / 9.0)


def test_power_ledger_closes_and_uses_fixed_region() -> None:
    root = _ensure_outputs()
    ledger = _csv_rows(root / "energy_ledger_full.csv")
    region = json.loads((root / "useful_region_definition.json").read_text(encoding="utf-8"))

    assert region["region_id"] == "fixed_regular_hexagon_radius_2p65_v0_ring"
    assert ledger
    assert max(float(row["numerical_power_closure_error"]) for row in ledger) <= 1e-9
    by_route = {}
    for row in ledger:
        by_route.setdefault(row["route_id"], set()).add(row["stage"])
    for stages in by_route.values():
        assert "useful_hexagon_region_power" in stages
        assert "peak_intensity_proxy" in stages


def test_optimal_shape_peak_useful_and_compromise_candidates_exist() -> None:
    root = _ensure_outputs()
    payload = json.loads((root / "optimal_hexagon_candidates.json").read_text(encoding="utf-8"))

    assert {"best_shape", "best_peak", "best_useful_power", "best_compromise"}.issubset(payload["best"])
    for name in [
        "optimal_hexagon_best_shape.png",
        "optimal_hexagon_best_peak.png",
        "optimal_hexagon_best_useful_power.png",
        "optimal_hexagon_best_compromise.png",
    ]:
        assert (root / name).exists()


def test_triangular_dark_core_cannot_pass_strict_visual_acceptance() -> None:
    assert mode2u_visual_acceptability("triangular_dark_core") is False
    assert mode2u_visual_acceptability("triangular_lobed_field") is False


def test_interaction_robustness_samples_are_reproducible_from_stored_seed() -> None:
    root = _ensure_outputs()
    summary = json.loads((root / "interaction_robustness_summary.json").read_text(encoding="utf-8"))
    rows = _csv_rows(root / "interaction_robustness_samples.csv")

    assert int(summary["seed"]) == 20260709
    assert len(rows) == int(summary["sample_count"])
    assert {int(r["seed"]) for r in rows} == {int(summary["seed"])}


def test_blind_correction_does_not_receive_hidden_injected_truth() -> None:
    root = _ensure_outputs()
    payload = json.loads((root / "blind_correction_results.json").read_text(encoding="utf-8"))

    assert payload["meta"]["uses_injected_truth"] is False
    for row in payload["rows"]:
        assert row["uses_injected_truth"] is False
        assert "camera_intensity" in row["algorithm_observables"]


def test_camera_shack_hartmann_and_stokes_roles_remain_distinct() -> None:
    root = _ensure_outputs()
    rows = json.loads((root / "correction_responsibility_matrix.json").read_text(encoding="utf-8"))
    instruments = {row["instrument"] for row in rows}

    assert {"camera", "Shack-Hartmann", "Stokes/polarimetry"}.issubset(instruments)
    camera = next(row for row in rows if row["instrument"] == "camera")
    sh = next(row for row in rows if row["instrument"] == "Shack-Hartmann")
    assert "final intensity" in camera["primary_observables"]
    assert "wavefront" in sh["primary_observables"]


def test_final_report_does_not_claim_microfabrication_or_authorise_m2v() -> None:
    root = _ensure_outputs()
    manifest = json.loads((root / "nathan_mode2u2_master_manifest.json").read_text(encoding="utf-8"))
    outcome = json.loads((root / "m2u2_outcome_report.json").read_text(encoding="utf-8"))
    doc = Path("docs/73_nathan_mode2u2_master_closure.md").read_text(encoding="utf-8").lower()

    assert manifest["microfabrication_sample_plane_claim"] is False
    assert outcome["selected_outcome"] in outcome["allowed_outcomes"]
    assert outcome["m2v_authorised"] is False
    assert "m2v authorised: `false`" in doc
