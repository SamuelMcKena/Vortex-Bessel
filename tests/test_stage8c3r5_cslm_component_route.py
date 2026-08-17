from pathlib import Path

import numpy as np

from vbb_study.digital_twin.cslm_route import (
    CONCEPTUAL_CSLM_COMPONENT_IDS,
    EXECUTED_CSLM_COMPONENT_IDS,
    WARNING_ONLY_CSLM_COMPONENT_IDS,
    CSLMRouteConfig,
    evaluate_fourier_filter_feasibility,
    plot_cslm_fourier_order_selection_audit,
    plot_cslm_phase_and_field_baselines,
    plot_cslm_route_inspection,
    run_cslm_baseline_route,
)


def _run(**overrides):
    return run_cslm_baseline_route(CSLMRouteConfig.fast(**overrides))


def test_cslm_route_chain_is_ordered_and_component_owned():
    run = _run()

    assert run.conceptual_route_chain == CONCEPTUAL_CSLM_COMPONENT_IDS
    assert run.executed_route_chain == EXECUTED_CSLM_COMPONENT_IDS
    assert [record.component_id for record in run.inspection_records] == list(EXECUTED_CSLM_COMPONENT_IDS)

    rows = [record.as_dict() for record in run.inspection_records]
    for row in rows:
        assert "component identifier" in row
        assert "incoming_field_metrics" in row
        assert "outgoing_field_metrics" in row
        assert "transform_applied" in row
        assert "actual segment distance (mm)" in row


def test_slm1_and_slm2_are_distinct_declared_planes():
    run = _run()
    declarations = {component.component_id: component for component in run.route_declaration}

    slm1 = declarations["SLM1_phase_plane"]
    slm2 = declarations["SLM2_phase_plane"]

    assert slm1.physical_location == "SLM1_plane"
    assert slm2.physical_location == "SLM2_plane"
    assert slm1.component_specific_parameters["slm1_role"] == "phase_only_conditioning"
    assert slm1.component_specific_parameters["topological_charge"] == run.config.slm1_topological_charge
    assert slm2.component_specific_parameters["slm2_role"] == "phase_correction_and_carrier_preserve_vortex"
    assert slm2.component_specific_parameters["axicon_phase_produced_here"] is False
    assert slm1.component_id != slm2.component_id


def test_phase_only_slm_transforms_preserve_energy_before_filters():
    run = _run()
    records = {record.component_id: record for record in run.inspection_records}

    for component_id in ("SLM1_phase_plane", "SLM2_phase_plane"):
        record = records[component_id]
        assert record.transform_applied is True
        assert record.model_status.startswith("phase_only")
        assert record.energy_after_uJ == record.energy_before_uJ
        before_power = record.incoming_field_metrics["raw_power_arb_um2"]
        after_power = record.outgoing_field_metrics["raw_power_arb_um2"]
        assert np.isclose(after_power, before_power, rtol=1e-12, atol=1e-12)


def test_slm2_does_not_produce_axicon_phase_and_only_applies_own_terms_before_propagation():
    run = _run(slm2_carrier_frequency_cpm=0.0)

    expected_slm2_field = run.slm2_input_state.field * np.exp(1j * run.slm2_quantized_phase_rad)
    assert np.allclose(run.slm2_state.field, expected_slm2_field)
    assert set(run.slm2_phase_terms_rad) == {"carrier_phase_rad", "correction_phase_rad"}
    assert "axicon_phase_rad" not in run.slm2_phase_terms_rad
    assert "vortex_phase_rad" not in run.slm2_phase_terms_rad
    assert run.slm2_state.metadata["axicon_phase_produced_here"] is False
    assert run.slm2_state.metadata["phase_quantisation_before_propagation"] is True
    assert run.baseline_metrics["phase_quantisation_before_propagation"] is True
    assert run.baseline_metrics["slm2_axicon_phase_present"] is False

    zero_reference = run.baseline_fields["zero_reference_field"]
    assert not np.allclose(run.reference_plane_state.field, zero_reference)


def test_changing_topological_charge_changes_generated_field_measurably():
    run = _run(slm2_carrier_frequency_cpm=0.0)

    assert run.baseline_metrics["topological_charge_test_from"] == run.config.slm1_topological_charge
    assert run.baseline_metrics["topological_charge_owner"] == "SLM1_phase_plane"
    assert run.baseline_metrics["topological_charge_measurable_change"] > 0.05
    assert run.baseline_metrics["slm1_vortex_core_fraction_r4um"] < 0.75
    assert run.baseline_metrics["active_route_contains_physical_axicon"] is False


def test_carrier_order_audit_uses_explicit_spatial_units():
    config = CSLMRouteConfig.fast(slm2_carrier_frequency_cpm=45_000.0)
    audit = evaluate_fourier_filter_feasibility(config)

    assert audit.fourier_filter_physics_available is False
    assert audit.present_parameters["slm2_carrier_frequency_cpm"] == 45_000.0
    assert audit.present_parameters["carrier_frequency_cycles_per_mm"] == 45.0
    assert audit.present_parameters["carrier_spatial_period_um"] > 0.0
    assert "cycles/m" in audit.order_selection_result["spatial_units"]
    assert np.isfinite(audit.order_selection_result["order_angle_rad_from_grating_equation"])


def test_inactive_fourier_filter_returns_no_fake_filtered_field():
    run = _run()

    assert run.fourier_filter_physics_available is False
    assert run.order_selection_result["filter_executed"] is False
    assert run.order_selection_result["filtered_field"] is None
    for component_id in WARNING_ONLY_CSLM_COMPONENT_IDS:
        assert component_id not in run.executed_route_chain
    declarations = {component.component_id: component for component in run.route_declaration}
    assert declarations["plus_one_order_filter"].represented_by_current_engine is False
    assert declarations["plus_one_order_filter"].physical_model_available is False


def test_no_material_or_stage8d_claims_are_introduced():
    run = _run()

    assert run.final_export_allowed is False
    assert run.propagated_stack.final_export_allowed is False
    assert run.reference_plane_state.metadata["n_medium"] == 1.0
    assert run.reference_plane_state.metadata["diagnostic_only"] is True
    assert run.reference_plane_state.metadata["no_material_model"] is True
    assert run.reference_plane_state.metadata["final_export_allowed"] is False

    forbidden = ("material", "dose", "thermal", "plasma", "nonlinear")
    generated_text = " ".join(
        [
            " ".join(run.executed_route_chain),
            " ".join(component.component_id for component in run.executed_components),
        ]
    ).lower()
    for word in forbidden:
        assert word not in generated_text


def test_required_figures_are_diagnostic_only_outputs():
    run = _run()

    out = Path("outputs/figures/digital_twin/stage8c3r5_test_outputs")
    out.mkdir(parents=True, exist_ok=True)
    p1 = plot_cslm_route_inspection(run, out / "route.png")
    p2 = plot_cslm_phase_and_field_baselines(run, out / "baselines.png")
    p3 = plot_cslm_fourier_order_selection_audit(run, out / "audit.png")

    assert p1.exists()
    assert p2.exists()
    assert p3.exists()
    assert run.final_export_allowed is False
    assert run.fourier_feasibility.final_export_allowed is False
