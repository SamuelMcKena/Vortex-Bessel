"""Stage 8C.3 component-owned route architecture tests."""

from __future__ import annotations

import numpy as np

from vbb_study.digital_twin.route_aware_axicon import (
    EXECUTED_PHYSICAL_AXICON_COMPONENT_IDS,
    RouteAwareAxiconConfig,
    build_route_component_declarations,
    build_route_perturbation_records,
    route_inspection_rows,
    run_route_aware_axicon_pipeline,
)


CFG = RouteAwareAxiconConfig.fast(grid_N=112, dx_um=0.6, n_z=10)


def test_executed_route_is_ordered_component_and_segment_chain():
    run = run_route_aware_axicon_pipeline(config=CFG)
    ids = tuple(c.component_id for c in run.component_chain)
    assert ids == EXECUTED_PHYSICAL_AXICON_COMPONENT_IDS
    assert tuple(r.component_id for r in run.route_inspection_records) == ids
    assert any(c.component_type == "propagation_segment" for c in run.component_chain)
    assert any(c.distance_to_next_element_mm > 0 for c in run.component_chain)
    for row in route_inspection_rows(run):
        assert "incoming_field_metrics" in row
        assert "outgoing_field_metrics" in row
        assert "energy before (uJ)" in row
        assert "energy after (uJ)" in row
        assert "centroid before (um)" in row
        assert "angle after (mrad)" in row
        assert "model status" in row
        assert "transform_applied" in row
    assert not any("objective" in cid.lower() for cid in ids)
    assert not any("after_objective" in cid.lower() for cid in ids)


def test_component_declarations_separate_supported_and_unrepresented_optics():
    declarations = {c.component_id: c for c in build_route_component_declarations(CFG)}
    assert declarations["physical_axicon"].represented_by_current_engine is True
    assert declarations["physical_axicon"].physical_model_available is True
    assert "decentre_x_um" in declarations["physical_axicon"].misalignment_modes_currently_supported
    assert "axial_offset_um" in declarations["physical_axicon"].misalignment_modes_currently_supported
    assert declarations["SLM1"].represented_by_current_engine is False
    assert declarations["SLM1"].status == "warning_only"
    assert declarations["Fourier_filter"].physical_model_available is False
    assert declarations["objective_pupil"].status == "future_not_implemented"
    executed = {c.component_id for c in run_route_aware_axicon_pipeline(config=CFG).component_chain}
    for unsupported in ("SLM1", "SLM2", "Fourier_filter", "relay_lens", "objective_pupil", "objective"):
        assert unsupported in declarations
        assert unsupported not in executed


def test_physical_axicon_decentre_is_component_pose_error_at_axicon_plane():
    run = run_route_aware_axicon_pipeline({"physical_axicon_centre_x_um": 7.0}, config=CFG)
    axicon_row = [r for r in run.route_inspection_records if r.component_id == "physical_axicon"][0]
    assert axicon_row.actual_pose_error["decentre_x_um"] == 7.0
    assert axicon_row.component_type == "physical_axicon"
    assert "post_axicon_diagnostic_boundary" in axicon_row.downstream_consequences
    assert run.axicon_incidence_metrics["relative_beam_to_axicon_offset_x_um"] < -6.0
    rec = [r for r in build_route_perturbation_records(run.config) if r.parameter_name == "physical_axicon_centre_x_um"][0]
    assert rec.component_id == "physical_axicon"
    assert rec.implementation_plane == "physical_axicon_plane"


def test_aperture_decentre_is_applied_at_aperture_component_and_changes_energy():
    base = run_route_aware_axicon_pipeline(config=CFG)
    dec = run_route_aware_axicon_pipeline(
        {"input_aperture_radius_um": 18.0, "input_aperture_centre_x_um": 14.0},
        config=CFG,
    )
    ap_row = [r for r in dec.route_inspection_records if r.component_id == "input_aperture"][0]
    assert ap_row.actual_pose_error["decentre_x_um"] == 14.0
    assert ap_row.aperture_overlap is not None
    assert ap_row.aperture_overlap < 0.99
    assert dec.propagated_stack.transmitted_fraction < base.propagated_stack.transmitted_fraction


def test_axicon_axial_offset_edits_adjacent_segment_distances_and_preserves_total_path():
    base = run_route_aware_axicon_pipeline(config=CFG)
    run = run_route_aware_axicon_pipeline({"physical_axicon_axial_offset_um": 75.0}, config=CFG)
    base_rows = {r.component_id: r for r in base.route_inspection_records}
    rows = {r.component_id: r for r in run.route_inspection_records}
    assert abs(rows["source_to_physical_axicon"].distance_to_next_element_mm - (CFG.pre_axicon_distance_mm + 0.075)) < 1e-12
    assert abs(rows["post_axicon_free_space_segment"].distance_to_next_element_mm - (CFG.post_axicon_free_space_distance_mm - 0.075)) < 1e-12
    segment_ids = ("source_to_physical_axicon", "post_axicon_free_space_segment", "post_axicon_to_reference_segment")
    base_total = sum(base_rows[k].distance_to_next_element_mm for k in segment_ids)
    shifted_total = sum(rows[k].distance_to_next_element_mm for k in segment_ids)
    assert abs(base_total - shifted_total) < 1e-12
    axicon = rows["physical_axicon"]
    assert axicon.actual_pose_error["axial_offset_um"] == 75.0


def test_field_state_controls_are_boundary_conditions_with_approximation_labels():
    run = run_route_aware_axicon_pipeline(
        {"field_tilt_x_mrad": 9.0, "field_tilt_location": "before_physical_axicon"},
        config=CFG,
    )
    boundary = [r for r in run.route_inspection_records if r.component_id == "physical_axicon_input_boundary"][0]
    assert boundary.model_status == "boundary_condition_active"
    rec = [r for r in run.perturbation_records if r.parameter_name == "field_tilt_x_mrad"][0]
    assert rec.component_id == "physical_axicon_input_boundary"
    assert rec.boundary_plane == "before_physical_axicon"
    assert "phase ramp" in rec.physical_approximation
    assert "upstream" in rec.upstream_hardware_error_could_emulate
    assert "physical_axicon" in rec.downstream_components_consume


def test_post_axicon_diagnostic_boundary_changes_only_downstream_records():
    base = run_route_aware_axicon_pipeline(config=CFG)
    pert = run_route_aware_axicon_pipeline(
        {"field_tilt_x_mrad": 8.0, "field_tilt_location": "post_axicon_diagnostic_boundary"},
        config=CFG,
    )
    base_rows = {r.component_id: r for r in base.route_inspection_records}
    pert_rows = {r.component_id: r for r in pert.route_inspection_records}
    upstream = (
        "source_field",
        "source_boundary_condition",
        "input_aperture",
        "source_to_physical_axicon",
        "physical_axicon_input_boundary",
        "physical_axicon",
        "after_physical_axicon_boundary",
        "post_axicon_free_space_segment",
    )
    for cid in upstream:
        for key in ("energy_uJ", "centroid_x_um", "centroid_y_um", "angle_x_mrad", "angle_y_mrad"):
            assert np.isclose(
                base_rows[cid].outgoing_field_metrics[key],
                pert_rows[cid].outgoing_field_metrics[key],
            )
    assert base_rows["post_axicon_diagnostic_boundary"].transform_applied is False
    assert pert_rows["post_axicon_diagnostic_boundary"].transform_applied is True
    assert pert_rows["post_axicon_diagnostic_boundary"].angle_after_mrad[0] > 7.0
    assert not np.isclose(
        base_rows["reference_plane"].centroid_after_um[0],
        pert_rows["reference_plane"].centroid_after_um[0],
    )


def test_no_stage8d_or_material_response_in_component_route():
    run = run_route_aware_axicon_pipeline(config=CFG)
    blob = str(run.metadata).lower() + str(run.propagated_stack.metadata).lower()
    assert run.final_export_allowed is False
    assert run.model_status == "optical_prediction"
    assert run.config.n_medium == 1.0
    assert run.reference_plane_state.metadata["route_endpoint"] == "free_space"
    assert "free-space reference plane" in str(run.reference_plane_state.metadata["reference_plane"])
    assert run.reference_plane_state.metadata["no_material_model"] is True
    reference = [r for r in run.route_inspection_records if r.component_id == "reference_plane"][0]
    assert reference.model_status == "diagnostic_only"
    banned = ("stage8d", "plasma", "ablation", "damage", "dose_accumulation", "thermal")
    assert not any(term in blob for term in banned)
