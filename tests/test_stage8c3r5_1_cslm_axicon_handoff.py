from pathlib import Path

import numpy as np

from vbb_study.equations.fields import make_xy_grid
from vbb_study.digital_twin.cslm_route import (
    EXECUTED_CSLM_COMPONENT_IDS,
    IDEAL_AXICON_BENCHMARK_COMPONENT_IDS,
    CSLMRouteConfig,
    _angle_mrad,
    _compose_slm2_phase,
    generate_stage8c3r5_1_previews,
    plot_cslm_to_axicon_handoff_audit,
    plot_ideal_vortex_bessel_axicon_benchmark,
    plot_unfiltered_vs_ideal_handoff,
    run_cslm_baseline_route,
)


def _benchmark_run(**overrides):
    params = dict(
        order_handoff_mode="ideal_selected_order_surrogate",
        slm2_carrier_frequency_cpm=45_000.0,
        slm2_correction_phase_rad=0.73,
    )
    params.update(overrides)  # allow a test to override any default (e.g. correction phase)
    cfg = CSLMRouteConfig.fast(**params)
    return run_cslm_baseline_route(cfg)


def test_none_handoff_preserves_active_r5_route_behaviour():
    run = run_cslm_baseline_route(CSLMRouteConfig.fast(order_handoff_mode="none"))

    assert run.executed_route_chain == EXECUTED_CSLM_COMPONENT_IDS
    assert run.ideal_benchmark_route_chain == ()
    assert run.ideal_selected_order_surrogate_state is None
    assert run.axicon_benchmark is None
    assert run.order_handoff.mode == "none"
    assert run.order_handoff.output_component_id == "post_SLM2_pre_4F_diagnostic_plane"
    assert "physical_axicon" not in " ".join(run.executed_route_chain)


def test_ideal_surrogate_comes_from_slm2_input_plus_slm2_correction_phase():
    run = _benchmark_run()
    grid = make_xy_grid(run.config.grid_N, run.config.dx_m)
    _, _, _, correction_quantized = _compose_slm2_phase(
        grid,
        run.config,
        include_carrier=False,
    )

    expected = run.slm2_input_state.field * np.exp(1j * correction_quantized)
    assert np.allclose(run.ideal_selected_order_surrogate_state.field, expected)
    assert not np.allclose(run.ideal_selected_order_surrogate_state.field, run.slm2_input_state.field)
    assert run.ideal_selected_order_surrogate_state.plane_name == "ideal_selected_order_handoff_plane"


def test_ideal_surrogate_is_explicitly_non_physical_and_not_4f_filtered():
    run = _benchmark_run()
    state = run.ideal_selected_order_surrogate_state
    handoff = run.order_handoff

    assert handoff.physical_filter_modelled is False
    assert handoff.energy_selection_modelled is False
    assert handoff.order_efficiency_modelled is False
    assert handoff.zero_order_rejection_modelled is False
    assert handoff.claim_boundary == "carrier-free desired-order benchmark; not a physical 4F prediction"
    assert state.metadata["physical_filter_modelled"] is False
    assert state.metadata["zero_order_rejection_modelled"] is False
    assert state.metadata["order_efficiency_modelled"] is False
    assert state.metadata["selected_order_energy_uJ"] is None
    forbidden = ("filtered_field", "plus_one_field", "4F_output", "physical_selected_order")
    joined = " ".join([state.plane_name, *state.applied_components, state.metadata["claim_boundary"]])
    for token in forbidden:
        assert token not in joined


def test_actual_post_slm2_field_retains_carrier_term():
    run = _benchmark_run()

    assert not np.allclose(run.slm2_state.field, run.ideal_selected_order_surrogate_state.field)
    actual_angle_x = _angle_mrad(run.slm2_state.field, run.config)[0]
    ideal_angle_x = _angle_mrad(run.ideal_selected_order_surrogate_state.field, run.config)[0]
    assert abs(actual_angle_x - ideal_angle_x) > 1.0
    assert run.post_slm2_unfiltered_state.metadata["contains_carrier"] is True


def test_ideal_surrogate_excludes_carrier_term():
    run = _benchmark_run()

    input_angle = _angle_mrad(run.slm2_input_state.field, run.config)
    ideal_angle = _angle_mrad(run.ideal_selected_order_surrogate_state.field, run.config)
    actual_angle = _angle_mrad(run.slm2_state.field, run.config)

    assert np.allclose(ideal_angle, input_angle, atol=1e-9)
    assert abs(actual_angle[0] - input_angle[0]) > 1.0


def test_slm1_vortex_remains_present_in_ideal_surrogate():
    vortex = _benchmark_run(slm1_topological_charge=3)
    flat = _benchmark_run(slm1_topological_charge=0)

    assert vortex.slm1_state.metadata["topological_charge"] == 3
    assert not np.allclose(
        vortex.ideal_selected_order_surrogate_state.field,
        flat.ideal_selected_order_surrogate_state.field,
    )
    assert vortex.baseline_metrics["topological_charge_owner"] == "SLM1_phase_plane"
    assert vortex.slm2_state.metadata["axicon_phase_produced_here"] is False


def test_physical_axicon_acts_on_ideal_surrogate_not_raw_slm2_input():
    run = _benchmark_run(slm2_correction_phase_rad=0.91)
    benchmark = run.axicon_benchmark

    expected = run.ideal_selected_order_surrogate_state.field * benchmark.transmission
    raw_shortcut = run.slm2_input_state.field * benchmark.transmission
    assert np.allclose(benchmark.physical_axicon_state.field, expected)
    assert not np.allclose(benchmark.physical_axicon_state.field, raw_shortcut)
    assert benchmark.benchmark_route_chain == IDEAL_AXICON_BENCHMARK_COMPONENT_IDS


def test_axicon_finite_aperture_clipping_reduces_energy_without_hidden_renormalisation():
    run = _benchmark_run(physical_axicon_clear_aperture_radius_um=8.0)
    benchmark = run.axicon_benchmark
    metrics = benchmark.metrics

    assert metrics["axicon_aperture_overlap_fraction"] < 1.0
    assert metrics["energy_after_axicon_uJ"] < metrics["energy_entering_axicon_uJ"]
    assert np.isclose(
        benchmark.physical_axicon_state.pulse_energy_after_uJ,
        benchmark.physical_axicon_state.pulse_energy_before_uJ
        * metrics["axicon_aperture_overlap_fraction"],
    )
    assert np.isclose(
        benchmark.benchmark_stack.sample_pulse_energy_uJ,
        metrics["energy_after_axicon_uJ"],
    )


def test_no_order_selection_efficiency_or_zero_order_loss_is_reported():
    run = _benchmark_run()
    metrics = run.axicon_benchmark.metrics

    assert metrics["selected_order_energy_uJ"] is None
    assert metrics["order_efficiency_modelled"] is False
    assert metrics["zero_order_rejection_modelled"] is False
    assert metrics["physical_filter_modelled"] is False
    assert run.fourier_filter_physics_available is False
    assert run.order_selection_result["filter_executed"] is False
    assert run.order_selection_result["filtered_field"] is None


def test_no_material_response_claims_are_introduced_in_benchmark_outputs():
    run = _benchmark_run()
    benchmark = run.axicon_benchmark

    text = " ".join(
        [
            " ".join(run.executed_route_chain),
            " ".join(benchmark.benchmark_route_chain),
            str(benchmark.benchmark_reference_state.metadata),
            str(benchmark.benchmark_stack.metadata),
        ]
    ).lower()
    forbidden = ("dose", "thermal", "plasma", "nonlinear", "damage")
    for token in forbidden:
        assert token not in text
    assert benchmark.benchmark_reference_state.metadata["no_material_model"] is True


def test_every_benchmark_output_has_final_export_disabled():
    run = _benchmark_run()
    benchmark = run.axicon_benchmark

    assert run.final_export_allowed is False
    assert run.ideal_selected_order_surrogate_state.metadata["final_export_allowed"] is False
    assert benchmark.final_export_allowed is False
    assert benchmark.physical_axicon_state.metadata["final_export_allowed"] is False
    assert benchmark.benchmark_reference_state.metadata["final_export_allowed"] is False
    assert benchmark.benchmark_stack.final_export_allowed is False
    assert benchmark.metrics["final_export_allowed"] is False


def test_required_r5_1_figures_are_generated_as_opt_in_outputs():
    run = _benchmark_run()
    out = Path("outputs/figures/digital_twin/stage8c3r5_1_test_outputs")
    out.mkdir(parents=True, exist_ok=True)

    p1 = plot_cslm_to_axicon_handoff_audit(run, out / "handoff.png")
    p2 = plot_unfiltered_vs_ideal_handoff(run, out / "handoff_compare.png")
    p3 = plot_ideal_vortex_bessel_axicon_benchmark(run, out / "benchmark.png")
    generated = generate_stage8c3r5_1_previews(
        CSLMRouteConfig.fast(order_handoff_mode="ideal_selected_order_surrogate"),
        out / "generated",
    )

    assert p1.exists()
    assert p2.exists()
    assert p3.exists()
    assert set(generated) == {
        "handoff_audit",
        "unfiltered_vs_ideal_handoff",
        "ideal_vortex_bessel_axicon_benchmark",
    }
    assert all(path.exists() for path in generated.values())
