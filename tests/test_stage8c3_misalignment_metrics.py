"""Stage 8C.3 degradation metric and comparison-plot tests."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from pathlib import Path

from vbb_study.digital_twin.active_realism_metrics import (
    build_stage8c3_sensitivity_scenarios,
    compute_misalignment_sensitivity_sweep,
    compute_degradation_metrics,
    metric_delta_table,
    plot_baseline_vs_perturbed_comparison,
    plot_misalignment_sensitivity_sweep,
)
from vbb_study.digital_twin.field_coupling import stack_from_arrays
from vbb_study.digital_twin.field_fluence import scale_stack_to_fluence
from vbb_study.digital_twin.lab_perturbations import apply_lab_perturbations_to_stack


MINIMUM_METRICS = {
    "centroid_x_um",
    "centroid_y_um",
    "peak_x_um",
    "peak_y_um",
    "peak_z_um",
    "ring_circularity_score",
    "azimuthal_uniformity_score",
    "side_lobe_imbalance",
    "core_fill_fraction",
    "central_darkness_contrast",
    "peak_fluence_change_fraction",
    "target_depth_peak_fluence",
    "central_roi_peak_fluence",
    "captured_power_drift_fraction",
    "pupil_clipped_power_fraction",
    "first_order_selected_fraction",
    "symmetry_score",
    "baseline_similarity_score",
    "unregistered_similarity_score",
    "registered_similarity_score",
    "centroid_shift_um",
    "translation_dominated_boolean",
    "residual_shape_deformation_score",
    "input_pulse_energy_uJ",
    "energy_before_perturbation_uJ",
    "energy_after_passive_loss_uJ",
    "transmitted_fraction",
    "peak_to_total_energy_ratio",
    "renormalisation_factor_applied",
    "post_stack_power_ratio",
    "commanded_axis_x_um",
    "commanded_axis_y_um",
    "ring_centre_x_um",
    "ring_centre_y_um",
    "ring_axis_offset_um",
    "brightest_point_offset_um",
    "beam_axis_surface_x_um",
    "beam_axis_surface_y_um",
    "beam_axis_target_offset_um",
    "beam_steering_angle_mrad",
    "field_of_view_margin_um",
    "out_of_frame_fraction",
    "crop_edge_energy_fraction",
}


def _stack():
    x = np.linspace(-20.0, 20.0, 61)
    y = x.copy()
    z = np.linspace(-60.0, 180.0, 25)
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.hypot(X, Y)
    planes = []
    for zi in z:
        amp = np.exp(-((zi - 80.0) / 80.0) ** 2) + 0.2
        planes.append(amp * np.exp(-((R - 5.5) ** 2) / (2.0 * 1.8**2)) + 0.12 * amp * np.exp(-(R**2) / (2.0 * 10.0**2)))
    return stack_from_arrays(np.asarray(planes), x, y, z, source_status="unit_test_fixture")


def test_degradation_metrics_include_required_keys():
    stack = _stack()
    fluence = scale_stack_to_fluence(stack, 40.0)
    metrics = compute_degradation_metrics(stack, fluence, target_depth_um=80.0)
    assert MINIMUM_METRICS <= set(metrics)
    assert "peak_trajectory_x_span_um" in metrics


def test_metric_delta_table_reports_required_metrics():
    stack = _stack()
    fluence = scale_stack_to_fluence(stack, 40.0)
    baseline = compute_degradation_metrics(stack, fluence, target_depth_um=80.0)
    result = apply_lab_perturbations_to_stack(
        stack,
        {
            "enable_vortex_centre_offset": True,
            "vortex_centre_offset_x_um": 4.0,
            "enable_zero_order_leakage": True,
            "zero_order_leakage_fraction": 0.08,
        },
    )
    perturbed_fluence = scale_stack_to_fluence(result.perturbed_stack, 40.0)
    perturbed = compute_degradation_metrics(
        result.perturbed_stack,
        perturbed_fluence,
        baseline_stack=stack,
        baseline_fluence_result=fluence,
        target_depth_um=80.0,
        perturbation_metadata=result.metadata,
    )
    table = metric_delta_table(baseline, perturbed)
    assert MINIMUM_METRICS <= {row["metric"] for row in table}
    assert perturbed["baseline_similarity_score"] < 1.0


def test_baseline_vs_perturbed_plot_uses_shared_scaling():
    stack = _stack()
    fluence = scale_stack_to_fluence(stack, 40.0)
    result = apply_lab_perturbations_to_stack(
        stack,
        {
            "enable_beam_decentre": True,
            "beam_decentre_x_um": 3.0,
            "enable_zernike_aberrations": True,
            "zernike_coma_x_waves": 0.2,
        },
    )
    perturbed_fluence = scale_stack_to_fluence(result.perturbed_stack, 40.0)
    fig = plot_baseline_vs_perturbed_comparison(
        stack,
        result.perturbed_stack,
        fluence,
        perturbed_fluence,
        selected_plane_index=10,
        show_caveats=True,
    )
    clims = {ax.get_title(): ax.images[0].get_clim() for ax in fig.axes if ax.images}
    assert clims["baseline XY fluence"] == clims["perturbed XY fluence"]
    assert clims["baseline XZ fluence"] == clims["perturbed XZ fluence"]
    assert fig.stage8c3_metadata["final_export_allowed"] is False
    assert fig.stage8c3_metadata["figure_status"] == "diagnostic_allowed"
    plt.close(fig)


def test_sensitivity_scenarios_cover_required_stage8c3b_cases():
    scenarios = build_stage8c3_sensitivity_scenarios()
    assert {
        "relative_vortex_axicon_misregistration",
        "axicon_relative_misregistration",
        "beam_decentre_slm_aperture",
        "beam_tilt_finite_pupil",
        "pupil_decentre_clipping",
        "low_order_aberrations",
        "zero_order_leakage",
        "combined_lab_stress",
        "vortex_axicon_coshift_translation",
    } <= set(scenarios)
    assert "vortex_centre_offset_x_um" in scenarios["relative_vortex_axicon_misregistration"].severe_controls
    assert "axicon_centre_offset_x_um" in scenarios["axicon_relative_misregistration"].severe_controls
    assert "beam_decentre_x_um" in scenarios["beam_decentre_slm_aperture"].severe_controls
    assert "beam_tilt_x_mrad" in scenarios["beam_tilt_finite_pupil"].severe_controls
    assert "pupil_decentre_x_um" in scenarios["pupil_decentre_clipping"].severe_controls
    for key in [
        "zernike_coma_x_waves",
        "zernike_astig_0_waves",
        "zernike_defocus_waves",
        "zernike_spherical_waves",
    ]:
        assert key in scenarios["low_order_aberrations"].severe_controls
    assert "zero_order_leakage_fraction" in scenarios["zero_order_leakage"].severe_controls


def test_sensitivity_sweep_metrics_show_mild_and_severe_degradation():
    stack = _stack()
    scenarios = build_stage8c3_sensitivity_scenarios()
    genuine_count = 0
    for scenario_key, scenario_def in scenarios.items():
        sweep = compute_misalignment_sensitivity_sweep(
            stack,
            40.0,
            scenario=scenario_key,
            target_depth_um=80.0,
            central_roi_half_width_um=8.0,
        )
        metric = sweep.scenario.degradation_metric
        assert sweep.mild_metrics[metric] != sweep.baseline_metrics[metric]
        assert sweep.severe_metrics[metric] != sweep.baseline_metrics[metric]
        assert sweep.metadata["severe_worse_than_mild"] is True
        if scenario_def.counts_as_genuine_degradation:
            genuine_count += 1
            assert (
                sweep.severe_metrics["residual_shape_deformation_score"]
                >= sweep.mild_metrics["residual_shape_deformation_score"]
            )
            assert (
                sweep.severe_metrics["registered_similarity_score"]
                <= sweep.mild_metrics["registered_similarity_score"] + 1e-9
            )
            assert not sweep.severe_metrics["translation_dominated_boolean"]
    assert genuine_count >= 7


def test_coshifted_vortex_axicon_is_translation_dominated_after_registration():
    stack = _stack()
    sweep = compute_misalignment_sensitivity_sweep(
        stack,
        40.0,
        scenario="vortex_axicon_coshift_translation",
        target_depth_um=80.0,
        central_roi_half_width_um=8.0,
    )
    severe = sweep.severe_metrics
    assert severe["translation_dominated_boolean"] is True
    assert severe["registered_similarity_score"] > severe["unregistered_similarity_score"] + 0.10
    assert severe["registered_similarity_score"] > 0.90
    assert severe["residual_shape_deformation_score"] < 0.10


def test_relative_phase_misregistration_has_more_residual_deformation_than_coshift():
    stack = _stack()
    relative = compute_misalignment_sensitivity_sweep(
        stack,
        40.0,
        scenario="relative_vortex_axicon_misregistration",
        target_depth_um=80.0,
        central_roi_half_width_um=8.0,
    )
    coshift = compute_misalignment_sensitivity_sweep(
        stack,
        40.0,
        scenario="vortex_axicon_coshift_translation",
        target_depth_um=80.0,
        central_roi_half_width_um=8.0,
    )
    assert (
        relative.severe_metrics["residual_shape_deformation_score"]
        > coshift.severe_metrics["residual_shape_deformation_score"] + 0.05
    )
    assert relative.severe_metrics["translation_dominated_boolean"] is False


def test_zero_order_leakage_raises_core_fill_with_severity():
    stack = _stack()
    sweep = compute_misalignment_sensitivity_sweep(
        stack,
        40.0,
        scenario="zero_order_leakage",
        target_depth_um=80.0,
        central_roi_half_width_um=8.0,
    )
    assert sweep.mild_metrics["core_fill_fraction"] > sweep.baseline_metrics["core_fill_fraction"]
    assert sweep.severe_metrics["core_fill_fraction"] > sweep.mild_metrics["core_fill_fraction"] + 0.05


def test_passive_clipping_reduces_perturbed_fluence_energy_and_reports_audit():
    stack = _stack()
    sweep = compute_misalignment_sensitivity_sweep(
        stack,
        40.0,
        scenario="pupil_decentre_clipping",
        target_depth_um=80.0,
        central_roi_half_width_um=8.0,
    )
    assert sweep.baseline_fluence.pulse_energy_uJ == 40.0
    assert sweep.severe_fluence.pulse_energy_uJ < sweep.mild_fluence.pulse_energy_uJ < 40.0
    assert sweep.severe_metrics["energy_after_passive_loss_uJ"] == sweep.severe_fluence.pulse_energy_uJ
    assert sweep.severe_metrics["transmitted_fraction"] < sweep.mild_metrics["transmitted_fraction"]
    assert sweep.severe_metrics["renormalisation_factor_applied"] == sweep.severe_metrics["transmitted_fraction"]
    assert sweep.severe_metrics["silent_renormalisation_warning"] is False
    assert np.isfinite(sweep.severe_metrics["peak_to_total_energy_ratio"])


def test_axis_and_fov_metrics_report_commanded_vs_actual_beam_axis():
    stack = _stack()
    sweep = compute_misalignment_sensitivity_sweep(
        stack,
        40.0,
        scenario="beam_decentre_slm_aperture",
        target_depth_um=80.0,
        central_roi_half_width_um=8.0,
    )
    assert sweep.severe_metrics["commanded_axis_x_um"] == 0.0
    assert sweep.severe_metrics["commanded_axis_y_um"] == 0.0
    assert sweep.severe_metrics["ring_axis_offset_um"] > sweep.mild_metrics["ring_axis_offset_um"]
    assert sweep.severe_metrics["beam_axis_target_offset_um"] > sweep.mild_metrics["beam_axis_target_offset_um"]
    assert sweep.severe_metrics["field_of_view_margin_um"] < sweep.baseline_metrics["field_of_view_margin_um"]
    assert 0.0 <= sweep.severe_metrics["out_of_frame_fraction"] <= 1.0
    assert 0.0 <= sweep.severe_metrics["crop_edge_energy_fraction"] <= 1.0


def test_zero_order_leakage_conserves_passive_energy():
    stack = _stack()
    sweep = compute_misalignment_sensitivity_sweep(
        stack,
        40.0,
        scenario="zero_order_leakage",
        target_depth_um=80.0,
        central_roi_half_width_um=8.0,
    )
    assert sweep.mild_metrics["transmitted_fraction"] == 1.0
    assert sweep.severe_metrics["transmitted_fraction"] == 1.0
    assert sweep.severe_fluence.pulse_energy_uJ == 40.0
    assert sweep.severe_metrics["energy_after_passive_loss_uJ"] == 40.0


def test_sensitivity_sweep_plot_uses_shared_scales_and_saves():
    stack = _stack()
    out = Path("outputs/figures/digital_twin/_stage8c3_test_sensitivity_sweep_preview.png")
    fig = plot_misalignment_sensitivity_sweep(
        stack,
        40.0,
        scenario="combined_lab_stress",
        target_depth_um=80.0,
        central_roi_half_width_um=8.0,
        output_path=out,
        show_caveats=True,
        dpi=80,
    )
    assert out.is_file()
    images = [ax.images[0] for ax in fig.axes if ax.images]
    assert len(images) >= 12
    assert images[0].get_clim() == images[1].get_clim() == images[2].get_clim()
    assert images[4].get_clim() == images[5].get_clim() == images[6].get_clim()
    assert images[8].get_clim() == images[9].get_clim() == images[10].get_clim()
    assert fig.stage8c3_metadata["stage"] == "stage8c3d_conservation_axis_diagnostics"
    assert fig.stage8c3_metadata["final_export_allowed"] is False
    assert fig.stage8c3_metadata["severe_worse_than_mild"] is True
    assert fig.stage8c3_metadata["translation_dominated_severe"] is False
    assert fig.stage8c3_metadata["severe_transmitted_fraction"] < 1.0
    assert fig.stage8c3_metadata["severe_energy_after_passive_loss_uJ"] < 40.0
    assert np.isfinite(fig.stage8c3_metadata["severe_axis_offset_um"])
    assert fig.stage8c3_metadata["severe_post_engine_spatial_clipping_applied"] is False
    assert "no post-propagation spatial crop" in fig.stage8c3_metadata["severe_passive_clipping_visual_model"]
    plt.close(fig)
