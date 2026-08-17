"""Stage 9B.0.1 upstream CSLM bridge and stop-sampling validity tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vbb_study.digital_twin.candidate_beam_atlas import (
    CANDIDATE_STATUS_LABELS,
    CandidateSpec,
    build_candidate_atlas_config,
    build_candidate_ranking_validity_rows,
    build_candidate_specs,
    candidate_manifest,
    evaluate_candidate_stop_sampling_convergence,
    export_candidate_package,
    plot_candidate_ranking_validity,
    plot_stop_sampling_convergence,
    plot_upstream_cslm_to_nominal_4f_chain,
    simulate_candidate,
)
from vbb_study.digital_twin.nominal_f300_4f import (
    CARRIER_REALISM,
    NominalF300Config,
    carrier_phase,
    phase_export_payload,
    run_nominal_f300_4f,
    run_to_manifest,
    stop_sampling_report,
)


ROOT = Path(__file__).resolve().parents[1]


def test_nominal_model_rejects_direct_slm1_phase_shortcut():
    cfg = NominalF300Config.exploratory()
    phase = np.zeros((cfg.simulation_grid_size, cfg.simulation_grid_size), dtype=float)
    with pytest.raises(ValueError, match="no longer accepts slm1_phase_rad"):
        run_nominal_f300_4f(cfg, slm1_phase_rad=phase)


def test_candidate_simulation_uses_existing_cslm_bridge():
    spec = CandidateSpec("vortex_ell_2", "vortex_charge_sweep", ell=2)
    run = simulate_candidate(spec, NominalF300Config.exploratory())
    assert run.upstream_source_mode == "existing_cslm_component_route"
    assert run.slm1_phase_applied_at_slm1 is True
    assert run.slm1_to_slm2_propagation_included is True
    assert "SLM1_to_SLM2_segment" in run.upstream_component_chain
    assert "field_arriving_at_SLM2" in run.upstream_component_chain
    assert np.ptp(run.slm1_phase_rad) > 0.0


def test_carrier_is_ideal_surrogate_not_pixelated_order_physics():
    run = simulate_candidate(CandidateSpec("vortex_ell_1", "vortex_charge_sweep", ell=1), NominalF300Config.exploratory())
    manifest = run_to_manifest(run)
    assert run.carrier_realism == CARRIER_REALISM
    assert run.ideal_blazed_carrier_shift_surrogate is True
    assert run.pixelated_slm_diffraction_orders_modelled is False
    assert run.zero_order_modelled is False
    assert run.physical_order_efficiency_modelled is False
    assert run.selected_order_purity_predicted is False
    assert manifest["carrier_boundary"]["pixelated_slm_diffraction_orders_modelled"] is False


def test_stop_sampling_profiles_gate_candidate_ranking():
    exploratory = stop_sampling_report(NominalF300Config.exploratory())
    standard = stop_sampling_report(NominalF300Config.standard())
    verified = stop_sampling_report(NominalF300Config.standard(), convergence_status="passed_for_nominal_scenario")
    assert exploratory["stop_sampling_status"] == "exploratory_only"
    assert exploratory["ranking_allowed"] is False
    assert standard["stop_sampling_status"] == "ranking_eligible"
    assert standard["ranking_allowed"] is False
    assert verified["stop_sampling_status"] == "convergence_verified"
    assert verified["ranking_allowed"] is True


def test_candidate_convergence_rows_are_ranking_eligible():
    rows = build_candidate_ranking_validity_rows()
    assert [row["candidate_id"] for row in rows] == [
        "gaussian_reference",
        "vortex_ell_1",
        "vortex_ell_2",
        "vortex_ell_3",
        "vortex_ell_4",
    ]
    assert all(row["convergence_status"] == "passed_for_nominal_scenario" for row in rows)
    assert all(row["stop_sampling_status"] == "convergence_verified" for row in rows)
    assert all(row["ranking_allowed"] is True for row in rows)
    assert sorted(row["robustness_rank"] for row in rows) == [1, 2, 3, 4, 5]


def test_initial_shortlist_is_gaussian_and_vortex_only():
    specs = build_candidate_specs()
    assert len(specs) == 5
    assert {spec.candidate_family for spec in specs} == {"gaussian_reference", "vortex_charge_sweep"}
    assert {spec.ell for spec in specs} == {0, 1, 2, 3, 4}


def test_candidate_manifest_contains_required_stage9b01_fields():
    spec = CandidateSpec("vortex_ell_2", "vortex_charge_sweep", ell=2)
    run = simulate_candidate(spec, NominalF300Config.standard())
    convergence = evaluate_candidate_stop_sampling_convergence(spec)
    manifest = candidate_manifest(spec, run, [], convergence)
    for key in (
        "candidate_id",
        "candidate_family",
        "topological_charge",
        "upstream_source_mode",
        "slm1_to_slm2_propagation_included",
        "slm2_carrier_mode",
        "carrier_realism",
        "stop_sampling_status",
        "convergence_status",
        "hardware_command_export_status",
        "bench_validation_status",
        "physical_4f_readiness",
    ):
        assert key in manifest
    assert manifest["hardware_command_export_status"] == "command_masks_exportable_unvalidated"
    assert manifest["bench_validation_status"] == "not_bench_validated"
    assert manifest["physical_4f_readiness"] == "blocked"


def test_export_package_uses_unvalidated_label_and_convergence_report(tmp_path):
    spec = CandidateSpec("vortex_ell_2", "vortex_charge_sweep", ell=2)
    paths = export_candidate_package(spec, run_id="stage9b0_1_test", output_root=tmp_path)
    package_dir = tmp_path / "stage9b0_1_test" / "vortex_ell_2"
    assert paths["stop_sampling_convergence_report"].is_file()
    manifest = json.loads((package_dir / "candidate_manifest.json").read_text(encoding="utf-8"))
    assert manifest["candidate_status_labels"] == list(CANDIDATE_STATUS_LABELS)
    assert manifest["hardware_command_export_status"] == "command_masks_exportable_unvalidated"
    assert manifest["convergence_status"] == "passed_for_nominal_scenario"
    boundary = (package_dir / "claim_boundary.md").read_text(encoding="utf-8")
    assert "pixelated_slm_diffraction_orders_modelled = `False`" in boundary


def test_static_study_contract_records_bridge_carrier_and_sampling():
    study = json.loads((ROOT / "configs/studies/cslm_nominal_4f_candidate_atlas_v1.json").read_text(encoding="utf-8"))
    assert study["stage"] == "9B.0.1"
    assert study["upstream_source_contract"]["candidate_default"] == "existing_cslm_component_route"
    assert study["carrier_boundary"]["carrier_realism"] == "ideal_continuous_phase_ramp"
    assert study["carrier_boundary"]["pixelated_slm_diffraction_orders_modelled"] is False
    assert study["stop_sampling_policy"]["standard_profile"]["stop_sampling_status"] == "ranking_eligible"
    assert study["initial_shortlist_candidate_ids"] == [
        "gaussian_reference",
        "vortex_ell_1",
        "vortex_ell_2",
        "vortex_ell_3",
        "vortex_ell_4",
    ]


def test_notebook_and_docs_expose_stage9b01_boundary():
    notebook = json.loads(
        (ROOT / "notebooks/digital_twin/01_nominal_f300_4f_virtual_bench_and_candidate_atlas.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "Stage 9B.0.1 - Upstream CSLM Bridge and Stop-Sampling Validity" in source
    assert "existing_cslm_component_route" in source
    doc = (ROOT / "docs/48_upstream_cslm_bridge_and_stop_sampling.md").read_text(encoding="utf-8")
    assert "Pixelated-SLM order physics not modelled" in doc
    assert "field_arriving_at_slm2" in doc


def test_stage9b01_figures_can_be_generated(tmp_path):
    bridge = plot_upstream_cslm_to_nominal_4f_chain(tmp_path / "bridge.png")
    convergence = plot_stop_sampling_convergence(tmp_path / "convergence.png")
    ranking = plot_candidate_ranking_validity(tmp_path / "ranking.png")
    assert bridge.is_file()
    assert convergence.is_file()
    assert ranking.is_file()


def test_slm2_phase_export_marks_unvalidated_carrier_surrogate():
    cfg = NominalF300Config.exploratory()
    grid = {"X": np.zeros((cfg.simulation_grid_size, cfg.simulation_grid_size)), "Y": np.zeros((cfg.simulation_grid_size, cfg.simulation_grid_size))}
    payload = phase_export_payload(carrier_phase(grid, cfg), mask_id="slm2_carrier", slm_id="SLM2", config=cfg)
    metadata = payload["metadata"]
    assert metadata["hardware_command_export_status"] == "command_masks_exportable_unvalidated"
    assert metadata["carrier_realism"] == "ideal_continuous_phase_ramp"
    assert metadata["pixelated_slm_diffraction_orders_modelled"] is False
