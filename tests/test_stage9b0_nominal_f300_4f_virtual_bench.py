"""Stage 9B.0 nominal F300 4F virtual-bench tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vbb_study.equations.fields import make_xy_grid
from vbb_study.digital_twin.bench_inventory import evaluate_physical_4f_readiness
from vbb_study.digital_twin.candidate_beam_atlas import (
    CANDIDATE_STATUS_LABELS,
    CandidateSpec,
    build_candidate_atlas_config,
    build_candidate_specs,
    candidate_manifest,
    export_candidate_package,
    relay_output_candidate_metrics,
    run_candidate_robustness,
    simulate_candidate,
)
from vbb_study.digital_twin.cslm_route import build_executed_cslm_component_chain
from vbb_study.digital_twin.nominal_f300_4f import (
    CLAIM_BOUNDARY_LABELS,
    FINAL_EXPORT_ALLOWED,
    NOMINAL_COMPONENT_SEQUENCE,
    NominalF300Config,
    carrier_phase,
    config_from_profile,
    fourier_plane_centroid_m,
    load_nominal_f300_profile,
    nominal_4f_sanity_report,
    phase_export_payload,
    run_nominal_f300_4f,
    run_to_manifest,
    vortex_phase,
)


ROOT = Path(__file__).resolve().parents[1]


def test_nominal_profile_records_only_f300_nominal_geometry_and_unknowns():
    profile = load_nominal_f300_profile(ROOT / "configs/hardware/cslm_f300_nominal_4f_profile.json")
    geom = profile["known_nominal_geometry"]
    for key in (
        "slm2_to_lens1_m",
        "lens1_focal_length_m",
        "lens1_to_fourier_plane_m",
        "fourier_plane_to_lens2_m",
        "lens2_focal_length_m",
        "lens2_to_nominal_relay_output_m",
    ):
        assert geom[key]["value"] == 0.3
        assert geom[key]["provenance"] == "nominal_from_bench_description"
    assert profile["physical_4f_readiness"] == "blocked"
    assert profile["final_export_allowed"] is False
    assert all(value is None for value in profile["unknown_or_unverified_hardware"].values())


def test_nominal_component_sequence_and_transform_flags_are_explicit():
    run = run_nominal_f300_4f(NominalF300Config.fast())
    manifest = run_to_manifest(run)
    assert manifest["component_sequence"] == list(NOMINAL_COMPONENT_SEQUENCE)
    rows = {row["component_id"]: row for row in manifest["component_manifest"]}
    assert rows["fourier_plane_field_pre_stop"]["transform_applied"] is False
    assert rows["fourier_stop_pinhole"]["transform_applied"] is True
    assert rows["nominal_relay_output_plane"]["transform_applied"] is False
    assert rows["SLM2_phase_plane"]["note"] == "SLM2 owns carrier only"
    assert run.nominal_4f_forward_model is True
    assert run.bench_calibrated is False
    assert run.final_export_allowed is FINAL_EXPORT_ALLOWED


def test_fourier_stop_is_plane_stop_not_output_crop():
    run = run_nominal_f300_4f(NominalF300Config.fast())
    sequence = [row["component_id"] for row in run.component_manifest]
    assert sequence.index("fourier_stop_pinhole") < sequence.index("fourier_plane_to_lens2_propagation")
    stop = next(row for row in run.component_manifest if row["component_id"] == "fourier_stop_pinhole")
    assert stop["component_type"] == "fourier_plane_circular_amplitude_stop"
    assert "Fourier plane" in stop["note"]
    assert run.fourier_plane_field_post_stop.shape == run.fourier_plane_field_pre_stop.shape


def test_smaller_pinhole_cannot_increase_transmitted_power():
    small = run_nominal_f300_4f(NominalF300Config.fast(pinhole_radius_m=0.00010))
    large = run_nominal_f300_4f(NominalF300Config.fast(pinhole_radius_m=0.00040))
    assert small.diagnostics["pinhole_transmitted_fraction"] <= large.diagnostics["pinhole_transmitted_fraction"] + 1e-9
    assert relay_output_candidate_metrics(small)["relative_transmitted_energy"] <= relay_output_candidate_metrics(large)["relative_transmitted_energy"] + 1e-9


def test_carrier_sign_mirrors_fourier_plane_displacement():
    common = dict(pinhole_radius_m=0.004, pinhole_offset_x_m=0.0, pinhole_offset_y_m=0.0, lens_clear_radius_m=0.0038)
    pos = run_nominal_f300_4f(
        NominalF300Config.fast(**common, command_domain_carrier_cycles_x=8, numerical_model_carrier_cycles_x=8)
    )
    neg = run_nominal_f300_4f(
        NominalF300Config.fast(**common, command_domain_carrier_cycles_x=-8, numerical_model_carrier_cycles_x=-8)
    )
    cx_pos, cy_pos = fourier_plane_centroid_m(pos)
    cx_neg, cy_neg = fourier_plane_centroid_m(neg)
    assert cx_pos > 0.0
    assert cx_neg < 0.0
    assert np.isclose(abs(cx_pos), abs(cx_neg), rtol=0.05)
    assert abs(cy_pos - cy_neg) < 1e-12


def test_lens_pupil_clipping_is_logged():
    run = run_nominal_f300_4f(NominalF300Config.fast(lens_clear_radius_m=0.0004))
    assert run.diagnostics["lens1_pupil_transmitted_fraction"] < 0.98
    assert run.diagnostics["lens2_pupil_transmitted_fraction"] < 0.98
    assert any("beam clipping at lens1 pupil" == warning for warning in run.warnings)
    assert any("beam clipping at lens2 pupil" == warning for warning in run.warnings)


def test_open_stop_relay_sanity_behaves_like_unit_magnification_4f():
    cfg = NominalF300Config.fast(
        pinhole_radius_m=0.0035,
        pinhole_offset_x_m=0.0,
        pinhole_offset_y_m=0.0,
        lens_clear_radius_m=0.0038,
        command_domain_carrier_cycles_x=0,
        numerical_model_carrier_cycles_x=0,
    )
    run = run_nominal_f300_4f(cfg)
    report = nominal_4f_sanity_report(run)
    assert report["open_stop_relay_intensity_correlation_with_inverted_input"] > 0.99
    assert report["final_export_allowed"] is False


def test_carrier_semantics_remain_nominal_not_physical_frequency():
    cfg = NominalF300Config.fast(command_domain_carrier_cycles_x=8, numerical_model_carrier_cycles_x=8)
    run = run_nominal_f300_4f(cfg)
    report = nominal_4f_sanity_report(run)
    assert report["carrier_coordinate_status"] == "nominal_model_not_bench_calibrated"
    assert report["carrier_cycles_across_model_width"] == [8.0, 0.0]
    blob = json.dumps(run_to_manifest(run)).lower()
    assert "cycles/mm" not in blob
    assert "physical_frequency_calibrated" not in blob


def test_slm1_vortex_and_slm2_carrier_only_no_slm2_axicon():
    cfg = NominalF300Config.fast()
    full_grid = make_xy_grid(cfg.simulation_grid_size, cfg.dx_m)
    slm1 = phase_export_payload(vortex_phase(full_grid, 2), mask_id="slm1_vortex_2", slm_id="SLM1", config=cfg)
    slm2 = phase_export_payload(carrier_phase(full_grid, cfg), mask_id="slm2_carrier", slm_id="SLM2", config=cfg)
    assert full_grid["X"].shape == slm1["phase_rad"].shape
    assert slm1["metadata"]["contains_axicon"] is False
    assert slm2["metadata"]["contains_axicon"] is False
    assert slm2["metadata"]["slm_id"] == "SLM2"

    spec = CandidateSpec("vortex_ell_2", "vortex_charge_sweep", ell=2)
    run = simulate_candidate(spec, cfg)
    manifest = candidate_manifest(spec, run, [])
    assert manifest["slm_role_contract"]["SLM2_contains_axicon_phase"] is False
    assert "carrier/future correction only" in manifest["slm_role_contract"]["SLM2"]


def test_candidate_package_contains_required_files_and_boundaries(tmp_path):
    spec = CandidateSpec("vortex_ell_2", "vortex_charge_sweep", ell=2)
    paths = export_candidate_package(spec, run_id="stage9b0_test", output_root=tmp_path)
    package_dir = tmp_path / "stage9b0_test" / "vortex_ell_2"
    required = build_candidate_atlas_config()["package_contract"]["required_files"]
    for name in required:
        assert (package_dir / name).is_file(), name
    manifest = json.loads((package_dir / "candidate_manifest.json").read_text(encoding="utf-8"))
    assert manifest["candidate_status_labels"] == list(CANDIDATE_STATUS_LABELS)
    assert manifest["simulation_status"] == "nominal_unvalidated"
    assert manifest["physical_4f_readiness"] == "blocked"
    assert manifest["camera_validation"] == "absent"
    assert manifest["material_prediction"] == "absent"
    assert manifest["final_export_allowed"] is False
    assert manifest["slm_role_contract"]["SLM2_contains_axicon_phase"] is False
    assert "not a material-response prediction" in (package_dir / "claim_boundary.md").read_text(encoding="utf-8")
    assert paths["candidate_manifest"].is_file()


def test_atlas_specs_cover_required_candidate_families():
    specs = build_candidate_specs()
    families = {spec.candidate_family for spec in specs}
    assert families == {"gaussian_reference", "vortex_charge_sweep"}
    assert [spec.candidate_id for spec in specs] == [
        "gaussian_reference",
        "vortex_ell_1",
        "vortex_ell_2",
        "vortex_ell_3",
        "vortex_ell_4",
    ]
    assert {spec.ell for spec in specs if spec.candidate_family == "vortex_charge_sweep"} == {1, 2, 3, 4}


def test_axicon_handoff_default_is_disabled():
    cfg = config_from_profile(load_nominal_f300_profile(ROOT / "configs/hardware/cslm_f300_nominal_4f_profile.json"))
    assert cfg.relay_output_to_axicon_mode == "unknown_not_simulated"
    run = run_nominal_f300_4f(NominalF300Config.fast())
    assert run.config.relay_output_to_axicon_mode == "unknown_not_simulated"
    assert run_to_manifest(run)["component_sequence"][-1] == "nominal_relay_output_plane"


def test_existing_active_cslm_route_and_readiness_are_not_changed():
    readiness = evaluate_physical_4f_readiness()
    assert readiness["fourier_filter_physics_available"] is False
    assert readiness["C_initial_scalar_4f_model"]["ready"] is False
    assert readiness["final_export_allowed"] is False
    chain_ids = [component.component_id for component in build_executed_cslm_component_chain()]
    assert "nominal_relay_output_plane" not in chain_ids
    assert "lens1_thin_phase_and_pupil" not in chain_ids


def test_static_artifacts_and_notebook_surface_stage9b0_contract():
    study = json.loads((ROOT / "configs/studies/cslm_nominal_4f_candidate_atlas_v1.json").read_text(encoding="utf-8"))
    assert study["study_name"] == "cslm_nominal_4f_candidate_atlas_v1"
    assert study["final_export_allowed"] is False
    assert study["slm_role_contract"]["SLM2_forbidden_content"][0] == "axicon phase"

    notebook = json.loads(
        (ROOT / "notebooks/digital_twin/01_nominal_f300_4f_virtual_bench_and_candidate_atlas.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "Stage 9B.0" in source
    assert "not_physical_4f_readiness_ready" in source

    for doc in (
        ROOT / "docs/46_nominal_f300_4f_virtual_bench.md",
        ROOT / "docs/47_candidate_beam_atlas_contract.md",
        ROOT / "STAGE9B0_NOMINAL_F300_4F_VIRTUAL_BENCH_SUMMARY.md",
    ):
        text = doc.read_text(encoding="utf-8")
        assert "final_export_allowed" in text
        assert "not_bench" in text or "not bench" in text.lower()


def test_no_final_or_material_ready_language_in_nominal_manifests():
    run = run_nominal_f300_4f(NominalF300Config.fast())
    manifest = run_to_manifest(run)
    assert manifest["final_export_allowed"] is False
    assert manifest["camera_validation"] == "absent"
    assert manifest["material_prediction"] == "absent"
    assert manifest["physical_4f_readiness"] != "ready"
    assert set(CLAIM_BOUNDARY_LABELS).issubset(set(manifest["claim_boundary_labels"]))
    blob = json.dumps(manifest).lower()
    for token in ("material_model_enabled\": true", "camera_model_enabled\": true", "\"physical_4f_readiness\": \"ready\"", "publication_ready"):
        assert token not in blob
