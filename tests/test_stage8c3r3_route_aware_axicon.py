"""Stage 8C.3R.3 route-aware physical-axicon alignment tests."""

from __future__ import annotations

import numpy as np

from vbb_study.digital_twin.route_aware_axicon import (
    DIAGNOSTIC_GEOMETRY_NOTE,
    DIAGNOSTIC_SWEEP_LABEL,
    REPRESENTED_PHYSICAL_AXICON_LOCATIONS,
    RouteAwareAxiconConfig,
    build_axicon_alignment_sweep_families,
    build_route_perturbation_records,
    holographic_slm_route_declarations,
    physical_axicon_route_graph,
    run_axicon_alignment_sweep,
    run_route_aware_axicon_pipeline,
)

CFG = RouteAwareAxiconConfig.fast(grid_N=192, dx_um=0.5, n_z=16)


def test_zero_control_physical_axicon_route_reproduces_clean_baseline():
    a = run_route_aware_axicon_pipeline(config=CFG)
    b = run_route_aware_axicon_pipeline(config=CFG)
    assert np.allclose(a.propagated_stack.intensity_zyx, b.propagated_stack.intensity_zyx)
    assert a.route_mode == "physical_axicon"
    assert a.final_export_allowed is False
    assert a.config.n_medium == 1.0
    assert DIAGNOSTIC_GEOMETRY_NOTE in a.propagated_stack.metadata["diagnostic_geometry_note"]


def test_perturbation_records_are_location_aware_not_generic_beam_tilt():
    cfg = RouteAwareAxiconConfig.fast(field_tilt_x_mrad=8.0, field_tilt_location="before_physical_axicon")
    rows = [r.as_dict() for r in build_route_perturbation_records(cfg)]
    tilt = [r for r in rows if r["parameter_name"] == "field_tilt_x_mrad"][0]
    assert tilt["perturbation_type"] == "field_tilt"
    assert tilt["injection_location"] == "before_physical_axicon"
    assert "physical_axicon_plane" in tilt["downstream_elements_affected"]
    assert tilt["active / warning-only / future status"] == "physics_active"
    assert all("beam_tilt" not in r["parameter_name"] for r in rows)


def test_route_graph_declares_represented_locations_and_demo_distances():
    graph = physical_axicon_route_graph(CFG)
    locations = {n.physical_location for n in graph}
    for loc in REPRESENTED_PHYSICAL_AXICON_LOCATIONS:
        assert loc in locations
    assert any(n.kind == "propagation_segment" and n.distance_mm > 0 for n in graph)
    assert any(DIAGNOSTIC_GEOMETRY_NOTE in n.note for n in graph if n.kind == "propagation_segment")


def test_source_plane_tilt_walks_off_at_axicon_with_geometry_prediction():
    run = run_route_aware_axicon_pipeline(
        {"field_tilt_x_mrad": 12.0, "field_tilt_location": "source_plane"},
        config=CFG,
    )
    m = run.axicon_incidence_metrics
    assert m["walkoff_model_applies"] is True
    assert abs(m["upstream_tilt_predicted_walkoff_um"] - m["upstream_tilt_measured_walkoff_um"]) < 0.25
    assert m["incident_angle_x_mrad"] > 10.0
    assert m["relative_beam_to_axicon_offset_um"] > 5.0


def test_same_tilt_changes_meaning_with_injection_location():
    source = run_route_aware_axicon_pipeline(
        {"field_tilt_x_mrad": 12.0, "field_tilt_location": "source_plane"},
        config=CFG,
    )
    before = run_route_aware_axicon_pipeline(
        {"field_tilt_x_mrad": 12.0, "field_tilt_location": "before_physical_axicon"},
        config=CFG,
    )
    after = run_route_aware_axicon_pipeline(
        {"field_tilt_x_mrad": 12.0, "field_tilt_location": "after_physical_axicon"},
        config=CFG,
    )
    assert source.axicon_incidence_metrics["relative_beam_to_axicon_offset_um"] > 5.0
    assert before.axicon_incidence_metrics["relative_beam_to_axicon_offset_um"] < 0.5
    assert before.axicon_incidence_metrics["incident_angle_x_mrad"] > 10.0
    assert after.axicon_incidence_metrics["incident_angle_x_mrad"] < 1.0
    status = [r.status for r in after.perturbation_records if r.parameter_name == "field_tilt_x_mrad"][0]
    assert status == "post_axicon_steering_test"


def test_beam_decentre_at_axicon_input_changes_relative_offset():
    run = run_route_aware_axicon_pipeline(
        {"beam_decentre_x_um": 6.0, "beam_decentre_location": "before_physical_axicon"},
        config=CFG,
    )
    assert run.axicon_incidence_metrics["relative_beam_to_axicon_offset_x_um"] > 5.0
    rec = [r for r in run.perturbation_records if r.parameter_name == "beam_decentre_x_um"][0]
    assert rec.injection_location == "before_physical_axicon"


def test_axicon_lateral_offset_changes_relative_beam_axicon_offset():
    run = run_route_aware_axicon_pipeline({"physical_axicon_centre_x_um": 6.0}, config=CFG)
    m = run.axicon_incidence_metrics
    assert m["axicon_centre_x_um"] == 6.0
    assert m["relative_beam_to_axicon_offset_x_um"] < -5.0


def test_physical_axicon_aperture_clipping_reduces_energy_without_renormalisation():
    base = run_route_aware_axicon_pipeline(config=CFG)
    clipped = run_route_aware_axicon_pipeline(
        {"physical_axicon_clear_aperture_radius_um": 18.0},
        config=CFG,
    )
    assert clipped.propagated_stack.transmitted_fraction < base.propagated_stack.transmitted_fraction
    assert clipped.propagated_stack.reference_plane_pulse_energy_uJ < clipped.propagated_stack.input_pulse_energy_uJ
    assert clipped.axicon_incidence_metrics["axicon_transmitted_fraction"] < 0.99


def test_physical_axicon_mechanical_tilt_is_future_not_generic_field_tilt():
    base = run_route_aware_axicon_pipeline(config=CFG)
    run = run_route_aware_axicon_pipeline({"physical_axicon_mechanical_tilt_x_mrad": 4.0}, config=CFG)
    assert np.allclose(base.propagated_stack.intensity_zyx, run.propagated_stack.intensity_zyx)
    rec = [r for r in run.perturbation_records if r.parameter_name == "physical_axicon_mechanical_tilt_x_mrad"][0]
    assert rec.status == "future_not_implemented"
    assert rec.implementation_plane == "not_represented_by_current_engine"


def test_holographic_route_unrepresented_planes_stay_warning_only():
    rows = holographic_slm_route_declarations()
    assert rows
    assert all(r.status == "warning_only" for r in rows)
    assert all(r.implementation_plane == "not_represented_by_current_engine" for r in rows)


def test_axicon_alignment_sweeps_are_diagnostic_not_lab_tolerances():
    families = build_axicon_alignment_sweep_families()
    assert set(families) == {
        "source_tilt", "axicon_input_decentre", "axicon_lateral_offset",
        "input_radius", "axicon_aperture", "relative_source_axicon",
    }
    curve = run_axicon_alignment_sweep("source_tilt", config=RouteAwareAxiconConfig.fast(grid_N=96, n_z=10))
    assert curve.label == DIAGNOSTIC_SWEEP_LABEL
    assert all(row["diagnostic_label"] == DIAGNOSTIC_SWEEP_LABEL for row in curve.rows)


def test_no_material_response_or_stage8d_claim():
    run = run_route_aware_axicon_pipeline(config=CFG)
    blob = str(run.metadata).lower() + str(run.propagated_stack.metadata).lower()
    assert run.model_status == "optical_prediction"
    assert run.final_export_allowed is False
    assert "no_material_model" in str(run.reference_plane_state.metadata)
    banned = ("plasma", "ablation", "damage", "waveguide", "dose_accumulation", "stage8d")
    assert not any(term in blob for term in banned)
