from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vbb_study.digital_twin.nathan_vector_hexagon import (
    NathanHexagonConfig,
    NathanMicroHexagonConfig,
    NathanSourceParityConfig,
    PatternedHWPConfig,
    build_visual_reproduction_ladder_report,
    build_control_suite,
    build_downstream_focus_validation_gate,
    build_downstream_model_comparison_gate,
    build_route_comparison_report,
    build_route_propagations,
    canonical_target_diagnostics,
    canonical_target_field,
    compare_vector_fields,
    default_nathan_grid,
    digital_twin_plane_map_rows,
    downstream_sampling_audit_rows,
    hwp_robustness_sweep,
    inherited_parameter_rows,
    lattice_control_report,
    nathan_specific_parameter_rows,
    nathan_literal_segmented_ra_input,
    plot_route_xy_xz_profiles,
    plot_visual_ladder_stage,
    run_v0_source_parity_visual_control,
    run_v1_inherited_preobjective_visual_gate,
    run_patterned_hwp_route,
    run_serial_slm_route,
    route_xy_xz_profile_arrays,
    serial_slm_robustness_sweep,
    source_convention_validation_control,
    source_parity_comparison,
    source_parity_grid,
    v0_numerical_resolution_status,
    v0_source_output_parity_report,
    v0_source_parameter_parity,
    vector_config_from_existing_twin,
    visual_ladder_stage_arrays,
)
from vbb_study.design import default_config
from vbb_study.publication.captions import caption_gate
from vbb_study.publication.figure_registry import registry_items
from vbb_study.vector_field import VectorField


def _cfg(n: int = 96) -> NathanHexagonConfig:
    return NathanHexagonConfig.fast(grid_n=n, z_planes=5, angular_samples=720)


def test_micro_hexagon_config_inherits_existing_digital_twin_baseline() -> None:
    twin = default_config("fast")
    cfg = NathanMicroHexagonConfig.from_existing_digital_twin_baseline(twin, grid_n=64, z_planes=5)
    assert cfg.twin is twin
    assert cfg.vector.wavelength_m == pytest.approx(twin.laser.wavelength_m)
    assert cfg.vector.pulse_duration_s == pytest.approx(twin.laser.pulse_duration_s)
    assert cfg.vector.waist_m == pytest.approx(twin.laser.beam_radius_on_slm_m)
    assert cfg.vector.slm1.n_x == twin.slm.resolution_x
    assert cfg.vector.slm1.n_y == twin.slm.resolution_y
    assert cfg.vector.slm1.pitch_m == pytest.approx(twin.slm.pixel_pitch_m)
    assert cfg.vector.slm1.carrier_lp_per_mm == pytest.approx(twin.slm.carrier_lpmm)
    assert cfg.vector.slm1.fill_factor == pytest.approx(twin.slm.fill_factor)
    assert cfg.vector.slm2.fill_factor == pytest.approx(twin.slm.fill_factor)
    assert cfg.baseline_preset == twin.grid.label


def test_vector_panel_config_explicitly_inherits_changed_slm_fill_factor() -> None:
    twin = default_config("fast")
    modified = replace(twin, slm=replace(twin.slm, fill_factor=0.81))
    cfg = NathanMicroHexagonConfig.from_existing_digital_twin_baseline(modified, grid_n=64, z_planes=3)
    assert cfg.vector.slm1.fill_factor == pytest.approx(0.81)
    assert cfg.vector.slm2.fill_factor == pytest.approx(0.81)
    resolved = vector_config_from_existing_twin(modified)
    assert resolved.slm1.fill_factor == pytest.approx(0.81)
    assert resolved.slm2.fill_factor == pytest.approx(0.81)


def test_scope_tables_and_plane_map_make_existing_geometry_explicit() -> None:
    cfg = _cfg(64)
    inherited = inherited_parameter_rows(cfg)
    nathan_rows = nathan_specific_parameter_rows(cfg)
    planes = digital_twin_plane_map_rows(cfg)
    inherited_names = {row["parameter"] for row in inherited}
    assert {"laser.wavelength_m", "laser.beam_radius_on_slm_m", "objective.NA", "target.target_bessel_length_m", "design.kr_sample_m_inv"} <= inherited_names
    assert any(row["parameter"] == "vector.n_pairs" and row["value"] == 3 for row in nathan_rows)
    assert [row["plane"] for row in planes] == ["P0", "P1", "P2", "P3", "P4", "P5", "P6+"]
    assert any("vector-generator output handoff" in row["meaning"] for row in planes)


def test_source_convention_validation_is_isolated_from_study_geometry() -> None:
    control = source_convention_validation_control(_cfg(64))
    assert control["scope"] == "Source-convention validation only; not the Digital Twin geometry."
    assert control["sector_count"] == 6
    assert control["sector_width_deg"] == pytest.approx(60.0)
    assert control["state_labels"] == {0: "radial", 1: "azimuthal"}
    assert control["radial_alpha_equals_theta"] is True
    assert control["azimuthal_alpha_equals_theta_plus_pi_over_2"] is True
    assert control["alternating_sixty_degree_states"] is True


def test_v0_literal_nathan_input_matches_canonical_arrays() -> None:
    cfg = NathanSourceParityConfig(grid_n=64, z_planes=5, window_m=8e-3)
    grid = source_parity_grid(cfg)
    literal, radial_mask = nathan_literal_segmented_ra_input(
        grid,
        wavelength_m=cfg.wavelength_m,
        beam_radius_m=cfg.beam_radius_m,
        n_pairs=cfg.n_pairs,
        sector_theta_rad=cfg.sector_theta_rad,
        sector_rotation_rad=cfg.sector_rotation_rad,
    )
    parity = source_parity_comparison(cfg)
    assert parity["status"] == "source_parity_exact"
    assert parity["radial_mask_equal"] is True
    assert parity["max_ex_abs_diff_away_from_centre"] < 1e-12
    assert parity["max_ey_abs_diff_away_from_centre"] < 1e-12
    assert literal.ex.shape == radial_mask.shape == (64, 64)

    rotated = NathanSourceParityConfig(grid_n=64, z_planes=5, window_m=8e-3, sector_rotation_rad=0.41)
    rotated_parity = source_parity_comparison(rotated)
    assert rotated_parity["status"] == "source_parity_exact"
    assert rotated_parity["sector_rotation_rad"] == pytest.approx(0.41)


def test_visual_reproduction_ladder_v0_uses_fixed_planes_and_stops_by_default() -> None:
    cfg = NathanSourceParityConfig(grid_n=48, z_planes=5, window_m=8e-3, z_span_m=20e-3, z_reference_m=10e-3)
    stage = run_v0_source_parity_visual_control(cfg)
    assert stage.stage_id == "V0"
    assert stage.reference_index == 2
    assert stage.metadata["plane_selection"].startswith("declared")
    assert stage.metadata["source_parity"]["status"] == "source_parity_exact"
    assert stage.comparison_stack is not None
    assert stage.intensity_stack.shape == stage.comparison_stack.shape == (5, 48, 48)
    assert stage.metadata["current_minus_literal_stack_rms"] < 1e-10

    arrays = visual_ladder_stage_arrays(stage)
    assert arrays["xy_full"].shape == (48, 48)
    assert arrays["xz"].shape == (5, 48)
    assert arrays["reference_z_m"] == pytest.approx(stage.z_values_m[stage.reference_index])

    report = build_visual_reproduction_ladder_report(source_config=cfg)
    assert [item.stage_id for item in report.stages] == ["V0"]
    assert report.status_report["input_array_parity"]["status"] == "source_parity_exact"
    assert report.status_report["propagated_output_verdict_is_separate_from_input_parity"] is True
    assert report.status_report["v2_allowed"] is False
    assert "V0 source-output parity status" in report.stopping_result


def test_v0_default_source_parameters_match_nathan_primary_sampling() -> None:
    cfg = NathanSourceParityConfig()
    parameter_parity = v0_source_parameter_parity(cfg)
    resolution = v0_numerical_resolution_status(cfg)
    assert parameter_parity["status"] == "source_parameters_match_nathan"
    assert parameter_parity["actual"]["grid_n"] == 1024
    assert parameter_parity["actual"]["window_m"] == pytest.approx(10e-3)
    assert parameter_parity["actual"]["dx_um"] == pytest.approx(9.765625)
    assert parameter_parity["actual"]["z_reference_m"] == pytest.approx(60e-3)
    assert resolution["status"] == "primary_v0_resolution"


def test_v0_report_keeps_input_and_propagated_output_verdicts_separate() -> None:
    cfg = NathanSourceParityConfig(grid_n=40, z_planes=5, window_m=8e-3, z_span_m=20e-3, z_reference_m=10e-3)
    stage = run_v0_source_parity_visual_control(cfg)
    report = v0_source_output_parity_report(stage)
    assert report["input_array_parity"]["status"] == "source_parity_exact"
    assert report["source_parameter_parity"]["status"] == "source_parameters_not_nathan_style"
    assert report["propagated_output_visual_verdict"] in {"PASS", "PARTIAL", "FAIL", "UNRESOLVED"}
    assert report["operator_visual_assessment"] is None
    assert report["propagated_output_verdict_is_separate_from_input_parity"] is True


def test_visual_ladder_plot_helper_creates_fixed_plane_figure() -> None:
    cfg = NathanSourceParityConfig(grid_n=40, z_planes=5, window_m=8e-3, z_span_m=20e-3, z_reference_m=10e-3)
    stage = run_v0_source_parity_visual_control(cfg)
    fig_path = Path("outputs/figures/digital_twin/_test_nathan_visual_ladder_v0.png")
    fig, _ = plot_visual_ladder_stage(stage, output_path=fig_path)
    import matplotlib.pyplot as plt

    plt.close(fig)
    assert fig_path.is_file()
    assert fig_path.stat().st_size > 0


def test_patterned_hwp_mosaic_and_seams_co_rotate_with_sector_target() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 1440, endpoint=False)
    grid = {
        "PHI": theta.reshape(1, -1),
        "R": np.ones((1, theta.size)),
        "X": np.cos(theta).reshape(1, -1),
        "Y": np.sin(theta).reshape(1, -1),
        "x": np.arange(theta.size, dtype=float),
        "y": np.asarray([0.0]),
        "dx": 1.0,
        "dy": 1.0,
    }
    base = _cfg(64)
    rotated = replace(base, vector=replace(base.vector, sector_rotation_rad=0.37))
    hwp = PatternedHWPConfig(case="mosaic", tiles_per_sector=4, seam_width_rad=0.025)
    base_route = run_patterned_hwp_route(base, grid=grid, hwp=hwp)
    rotated_route = run_patterned_hwp_route(rotated, grid=grid, hwp=hwp)
    assert rotated_route.comparison.normalized_rms_error == pytest.approx(base_route.comparison.normalized_rms_error, abs=2e-3)
    assert rotated_route.comparison.power_ratio == pytest.approx(base_route.comparison.power_ratio, abs=2e-3)


def test_downstream_model_comparison_gate_keeps_s0_s1_v1_distinct() -> None:
    cfg = NathanMicroHexagonConfig.fast(grid_n=48, z_planes=3, angular_samples=360)
    gate = build_downstream_model_comparison_gate(cfg, control_ids=("all_radial", "all_azimuthal", "nathan_six_sector"))
    assert gate["status"] == "diagnostic_gate_only_no_final_claim"
    results = {(result.route_id, result.control_id): result for result in gate["route_results"]}
    assert ("S0_existing_current_digital_twin_path", "nathan_six_sector") in results
    assert ("S1_scalar_component_surrogate", "nathan_six_sector") in results
    assert ("V1_vector_downstream_reference_current", "nathan_six_sector") in results

    s0 = results[("S0_existing_current_digital_twin_path", "nathan_six_sector")]
    s1 = results[("S1_scalar_component_surrogate", "nathan_six_sector")]
    v1 = results[("V1_vector_downstream_reference_current", "nathan_six_sector")]
    assert s0.intensity_stack.shape == s1.intensity_stack.shape == v1.intensity_stack.shape
    assert all(value == pytest.approx(0.0) for value in s1.ez_energy_fraction)
    assert s1.metadata["vector_asm_projection"] is False
    assert s1.metadata["ps_fresnel_axicon"] is False
    assert s0.metadata["vector_asm_projection"] is True
    assert s0.metadata["ps_fresnel_axicon"] is True
    assert v1.metadata["same_implementation_as_S0"] is True
    assert gate["comparisons"]["S0_minus_V1_canonical_intensity_rms"] == pytest.approx(0.0)
    assert "S1_minus_V1_canonical_intensity_rms" in gate["comparisons"]


def test_downstream_sampling_audit_exposes_grid_facts() -> None:
    rows = downstream_sampling_audit_rows(_cfg(64))
    by_item = {row["item"]: row for row in rows}
    assert by_item["handoff_grid_N"]["value"] == 64
    assert by_item["slm_pixel_pitch_m"]["value"] == pytest.approx(default_config("fast").slm.pixel_pitch_m)
    assert by_item["focal_dx_m"]["value"] > 0.0
    assert by_item["samples_per_radial_period_at_focus"]["value"] > 0.0
    assert "single_grid_warning" in by_item


def test_downstream_focus_validation_gate_runs_real_f2_reference() -> None:
    cfg = NathanMicroHexagonConfig.fast(grid_n=24, z_planes=3, angular_samples=240)
    gate = build_downstream_focus_validation_gate(
        cfg,
        control_ids=("all_radial", "all_azimuthal", "nathan_six_sector"),
        f2_chunk_size=96,
    )
    assert gate["status"] == "focus_validation_gate_unresolved_pending_converged_F2"
    assert gate["completion_gate"]["selected_conclusion"] is None
    assert gate["completion_gate"]["C_allowed"] is False
    assert any("scalar per-component focal transform has not been validated" in item for item in gate["model_boundary"])

    results = {(result.route_id, result.control_id): result for result in gate["route_results"]}
    f0 = results[("F0_current_scalar_focus_bridge", "nathan_six_sector")]
    f1 = results[("F1_scalar_component_surrogate", "nathan_six_sector")]
    f2 = results[("F2_vectorial_pupil_spectrum_reference", "nathan_six_sector")]
    assert f0.intensity_stack.shape == f1.intensity_stack.shape == f2.intensity_stack.shape
    assert f2.metadata["scalar_pupil_focus_mapping"] is False
    assert f2.metadata["objective_model"].startswith("F2 direct vectorial plane-wave")
    assert f2.metadata["focus_reference"]["solver"] == "fft"
    assert np.mean(f2.ez_energy_fraction) > 0.0
    assert all(value == pytest.approx(0.0) for value in f1.ez_energy_fraction)
    assert {"S0", "S1", "S2", "S3"} <= set(f2.stokes_maps)
    assert "F0_current_scalar_focus_bridge_minus_F2:nathan_six_sector" in gate["difference_maps"]
    assert "F1_scalar_component_surrogate_minus_F2:nathan_six_sector:full_field_equal_power_shape_rms" in gate["comparisons"]
    assert "F1_scalar_component_surrogate_minus_F2:nathan_six_sector:raw_intensity_rms_unscaled" in gate["comparisons"]
    assert "power_rows" in gate
    assert len(gate["multiscale_sampling_rows"]) > len(gate["sampling_audit_rows"])


def test_f2_physical_control_signatures_and_low_na_diagnostic() -> None:
    cfg = NathanMicroHexagonConfig.fast(grid_n=33, z_planes=11, angular_samples=240)
    gate = build_downstream_focus_validation_gate(
        cfg,
        control_ids=("all_radial", "all_azimuthal"),
        f2_solver="fft",
    )
    results = {(result.route_id, result.control_id): result for result in gate["route_results"]}
    f0_radial = results[("F0_current_scalar_focus_bridge", "all_radial")]
    f2_radial = results[("F2_vectorial_pupil_spectrum_reference", "all_radial")]
    f2_azimuthal = results[("F2_vectorial_pupil_spectrum_reference", "all_azimuthal")]

    assert f0_radial.z_values_m.shape == f2_radial.z_values_m.shape == (11,)
    assert np.allclose(f0_radial.z_values_m, f2_radial.z_values_m)
    assert f0_radial.metadata["output_grid_N"] == f2_radial.metadata["output_grid_N"]
    assert f0_radial.metadata["output_grid_dx_m"] == pytest.approx(f2_radial.metadata["output_grid_dx_m"])
    assert np.isfinite(f2_radial.intensity_stack).all()
    assert np.isfinite(f2_azimuthal.intensity_stack).all()
    assert f2_radial.transversality_residual is not None and f2_radial.transversality_residual < 1e-12
    assert f2_azimuthal.transversality_residual is not None and f2_azimuthal.transversality_residual < 1e-12

    radial_ez0 = f2_radial.metadata["on_axis_ez_intensity_by_z"][0]
    azimuthal_ez0 = f2_azimuthal.metadata["on_axis_ez_intensity_by_z"][0]
    assert radial_ez0 > 100.0 * max(azimuthal_ez0, 1e-30)
    assert np.mean(f2_radial.ez_energy_fraction) > 0.01
    assert np.mean(f2_azimuthal.ez_energy_fraction) < 1e-12

    for result in (f2_radial, f2_azimuthal):
        image = np.asarray(result.intensity_stack[0], dtype=float)
        rotated = np.rot90(image)
        image_n = image / max(float(np.sum(image)), 1e-30)
        rotated_n = rotated / max(float(np.sum(rotated)), 1e-30)
        rotational_rms = np.sqrt(np.sum((image_n - rotated_n) ** 2)) / max(np.sqrt(np.sum(image_n**2)), 1e-30)
        assert rotational_rms < 0.12

    base_twin = cfg.twin
    pupil_radius = base_twin.objective.pupil_radius_m
    low_na_twin = replace(
        base_twin,
        objective=replace(base_twin.objective, NA=0.08, f_eff_m=pupil_radius * base_twin.objective.immersion_n / 0.08),
    )
    high_na_gate = build_downstream_focus_validation_gate(cfg, control_ids=("scalar_bessel_gaussian_baseline",), f2_solver="fft")
    low_na_gate = build_downstream_focus_validation_gate(
        cfg,
        twin_config=low_na_twin,
        control_ids=("scalar_bessel_gaussian_baseline",),
        f2_solver="fft",
    )
    key = "F0_current_scalar_focus_bridge_minus_F2:scalar_bessel_gaussian_baseline:full_field_equal_power_shape_rms"
    assert np.isfinite(low_na_gate["comparisons"][key])
    assert np.isfinite(high_na_gate["comparisons"][key])
    assert low_na_gate["completion_gate"]["selected_conclusion"] is None
    assert high_na_gate["completion_gate"]["selected_conclusion"] is None


def test_canonical_target_field_and_controls_are_single_ground_truth() -> None:
    cfg = _cfg()
    grid = default_nathan_grid(cfg)
    diag = canonical_target_diagnostics(cfg, grid=grid)
    field = diag["field"]
    assert field.ex.shape == (cfg.grid_n, cfg.grid_n)
    assert set(np.unique(diag["sector_mask"])) == {0, 1}
    assert np.allclose(field.intensity, np.asarray(diag["Ex_amplitude"]) ** 2 + np.asarray(diag["Ey_amplitude"]) ** 2)
    assert {"S0", "S1", "S2", "S3"} <= set(diag["stokes"])
    plus = diag["circular_plus"]
    minus = diag["circular_minus"]
    assert plus.shape == minus.shape == field.ex.shape

    radial = diag["radial_control"]
    azimuthal = diag["azimuthal_control"]
    assert compare_vector_fields(radial, field).complex_overlap < 0.8
    assert compare_vector_fields(azimuthal, field).complex_overlap < 0.8


def test_patterned_hwp_continuous_reproduces_target_and_piston_only_comparison_has_teeth() -> None:
    cfg = _cfg()
    route = run_patterned_hwp_route(cfg, hwp=PatternedHWPConfig(case="continuous"))
    assert route.comparison.complex_overlap > 1.0 - 1e-12
    assert route.comparison.normalized_rms_error < 1e-12
    assert route.metadata["power_transmission"] == pytest.approx(1.0)

    target = canonical_target_field(cfg)
    ramp = np.exp(1j * 0.2 * np.asarray(target.grid["X"]) / float(target.grid["dx"]))
    spatially_rephased = VectorField(
        ex=target.ex * ramp,
        ey=target.ey * ramp,
        ez=target.ez,
        grid=target.grid,
        wavelength_m=target.wavelength_m,
    )
    comparison = compare_vector_fields(spatially_rephased, target)
    assert comparison.normalized_rms_error > 0.05
    assert comparison.complex_overlap < 0.999


def test_patterned_hwp_mosaic_converges_toward_continuous_target() -> None:
    cfg = _cfg(128)
    errors = [
        run_patterned_hwp_route(cfg, hwp=PatternedHWPConfig(case="mosaic", tiles_per_sector=t)).comparison.normalized_rms_error
        for t in (1, 2, 4, 8, 16)
    ]
    assert all(left > right for left, right in zip(errors, errors[1:]))
    assert errors[-1] < 0.025

    seam = run_patterned_hwp_route(cfg, hwp=PatternedHWPConfig(case="continuous", seam_width_rad=0.03))
    assert seam.comparison.power_ratio < 1.0
    defect = run_patterned_hwp_route(cfg, hwp=PatternedHWPConfig(case="continuous", central_defect_radius_m=0.2 * cfg.vector.waist_m))
    assert defect.comparison.power_ratio < 1.0


def test_serial_slm_ideal_reconstructs_target_and_wrong_variants_fail() -> None:
    cfg = _cfg()
    ideal = run_serial_slm_route(cfg, case="ideal")
    assert ideal.comparison.complex_overlap > 1.0 - 1e-12
    assert ideal.comparison.normalized_rms_error < 1e-12

    wrong_phase = run_serial_slm_route(cfg, case="ideal", naive_psi2=True)
    assert wrong_phase.comparison.normalized_rms_error > 0.1

    realistic = run_serial_slm_route(NathanHexagonConfig.fast(grid_n=96), case="panel_realistic")
    assert realistic.metadata["order_filter_applied"] is True
    assert realistic.metadata["iris_ledger"]["relative_error"] < 1e-12
    assert realistic.comparison.complex_overlap < 0.2


def test_shared_axicon_report_controls_and_lattice_rejection_execute() -> None:
    report = build_route_comparison_report(_cfg(64))
    assert len(report.input_rows) == 6
    assert len(report.output_rows) == 6
    assert all(np.isfinite(float(row["best_h6"])) for row in report.output_rows)
    by_id = {row["route_id"]: row for row in report.output_rows}
    assert by_id["canonical_target"]["best_h6"] == pytest.approx(by_id["patterned_hwp_continuous"]["best_h6"])
    assert by_id["serial_slm_ideal"]["best_h6"] == pytest.approx(by_id["canonical_target"]["best_h6"])

    controls = build_control_suite(_cfg(64))
    assert controls["wrong_serial_phase"]["normalized_rms_error"] > 0.1
    assert controls["six_lobe_lattice"]["wall_continuity_pass"] is False
    assert lattice_control_report(default_nathan_grid(_cfg(64)))["wall_continuity"] < 0.25


def test_xy_xz_visualisation_helpers_create_profile_gallery() -> None:
    cfg = _cfg(64)
    propagations = build_route_propagations(cfg, route_ids=("canonical_target", "serial_slm_panel_realistic"))
    assert len(propagations) == 2

    arrays = route_xy_xz_profile_arrays(propagations[0])
    assert arrays["xy_intensity"].shape == (cfg.grid_n, cfg.grid_n)
    assert arrays["xz_intensity"].shape[0] == cfg.z_planes
    assert arrays["xz_intensity"].shape[1] == cfg.grid_n
    assert np.isfinite(arrays["xy_intensity"]).all()
    assert np.isfinite(arrays["xz_intensity"]).all()

    fig_path = Path("outputs/figures/digital_twin/_test_nathan_vector_hexagon_xy_xz_profiles.png")
    fig, _ = plot_route_xy_xz_profiles(propagations, output_path=fig_path)
    import matplotlib.pyplot as plt

    plt.close(fig)
    assert fig_path.is_file()
    assert fig_path.stat().st_size > 0


def test_sweep_helpers_return_exploratory_rows_without_hidden_thresholds() -> None:
    cfg = _cfg(64)
    hwp_rows = hwp_robustness_sweep(cfg, family="tiles_per_sector", values=(1, 4))
    assert hwp_rows[0]["normalized_rms_error"] > hwp_rows[1]["normalized_rms_error"]

    slm_rows = serial_slm_robustness_sweep(cfg, family="fill_factor", values=(1.0, 0.93))
    assert len(slm_rows) == 2
    unsupported = serial_slm_robustness_sweep(cfg, family="registration_shift", values=(0.0,))
    assert unsupported[0]["status"] == "not_supported_by_current_geometry"


def test_publication_governance_marks_nathan_outputs_exploratory() -> None:
    gate = caption_gate("digital_twin_vector_hexagon")
    assert gate.export_allowed is False
    assert gate.warning_level == "amber"
    assert any("nathan_vector_hexagon" in item.output_path and item.status == "exploratory_only" for item in registry_items())
