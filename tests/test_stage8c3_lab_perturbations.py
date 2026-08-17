"""Stage 8C.3 active lab-realism perturbation tests."""

import numpy as np

from vbb_study.digital_twin.active_realism_metrics import compute_degradation_metrics
from vbb_study.digital_twin.exposure_bookkeeping import line_exposure_summary
from vbb_study.digital_twin.field_coupling import stack_from_arrays
from vbb_study.digital_twin.field_fluence import scale_stack_to_fluence
from vbb_study.digital_twin.lab_perturbations import apply_lab_perturbations_to_stack


def _ring_stack():
    x = np.linspace(-25.0, 25.0, 81)
    y = x.copy()
    z = np.linspace(-100.0, 250.0, 36)
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.hypot(X, Y)
    planes = []
    for zi in z:
        ring_r = 5.0 + 0.008 * zi
        width = 2.0 + 0.002 * abs(zi - 80.0)
        amp = np.exp(-((zi - 90.0) / 110.0) ** 2) + 0.2
        planes.append(
            amp * np.exp(-((R - ring_r) ** 2) / (2.0 * width**2))
            + 0.15 * amp * np.exp(-(R**2) / (2.0 * 14.0**2))
        )
    return stack_from_arrays(
        np.asarray(planes),
        x,
        y,
        z,
        field_label="stage8c3_test_ring",
        source_status="unit_test_fixture",
    )


def _baseline_metrics(stack):
    fluence = scale_stack_to_fluence(stack, 50.0)
    return fluence, compute_degradation_metrics(stack, fluence, target_depth_um=100.0)


def _perturbed_metrics(stack, controls):
    baseline_fluence, baseline = _baseline_metrics(stack)
    result = apply_lab_perturbations_to_stack(stack, controls)
    perturbed_fluence = scale_stack_to_fluence(result.perturbed_stack, 50.0)
    metrics = compute_degradation_metrics(
        result.perturbed_stack,
        perturbed_fluence,
        baseline_metrics=baseline,
        baseline_stack=stack,
        baseline_fluence_result=baseline_fluence,
        target_depth_um=100.0,
        perturbation_metadata=result.metadata,
    )
    return baseline, result, metrics


def test_beam_decentre_changes_centroid():
    stack = _ring_stack()
    baseline, _, metrics = _perturbed_metrics(
        stack,
        {"enable_beam_decentre": True, "beam_decentre_x_um": 4.0},
    )
    assert abs(metrics["centroid_x_um"] - baseline["centroid_x_um"]) > 2.0
    assert metrics["symmetry_score"] < baseline["symmetry_score"]


def test_beam_tilt_changes_projection_and_records_phase_ramp():
    stack = _ring_stack()
    baseline, result, metrics = _perturbed_metrics(
        stack,
        {"enable_beam_tilt": True, "beam_tilt_x_mrad": 20.0},
    )
    assert metrics["peak_trajectory_x_span_um"] > baseline["peak_trajectory_x_span_um"]
    assert result.metadata["phase_ramp_kx_rad_per_um"] > 0.0
    assert "E(x,y) exp[i(kx0 x + ky0 y)]" in result.metadata["beam_tilt_phase_ramp"]


def test_vortex_and_axicon_centre_offsets_degrade_uniformity_or_symmetry():
    stack = _ring_stack()
    baseline, _, vortex = _perturbed_metrics(
        stack,
        {"enable_vortex_centre_offset": True, "vortex_centre_offset_x_um": 5.0},
    )
    _, _, axicon = _perturbed_metrics(
        stack,
        {"enable_axicon_centre_offset": True, "axicon_centre_offset_y_um": 5.0},
    )
    assert vortex["azimuthal_uniformity_score"] < baseline["azimuthal_uniformity_score"] - 0.02
    assert axicon["symmetry_score"] < baseline["symmetry_score"]


def test_zero_order_leakage_increases_core_fill_fraction():
    stack = _ring_stack()
    baseline, _, metrics = _perturbed_metrics(
        stack,
        {"enable_zero_order_leakage": True, "zero_order_leakage_fraction": 0.15},
    )
    assert metrics["core_fill_fraction"] > baseline["core_fill_fraction"] + 0.05
    assert metrics["central_darkness_contrast"] < baseline["central_darkness_contrast"]


def test_pupil_clipping_reports_clipped_power_without_postprop_crop_artifact():
    stack = _ring_stack()
    baseline, result, metrics = _perturbed_metrics(
        stack,
        {
            "enable_pupil_clipping": True,
            "pupil_radius_um": 8.0,
            "pupil_decentre_x_um": 2.0,
        },
    )
    assert metrics["pupil_clipped_power_fraction"] > 0.1
    assert result.metadata["post_engine_spatial_clipping_applied"] is False
    assert "no post-propagation spatial crop" in result.metadata["passive_clipping_visual_model"]
    assert np.allclose(result.perturbed_stack.intensity_zyx, stack.intensity_zyx)
    assert metrics["symmetry_score"] == baseline["symmetry_score"]


def test_defocus_changes_peak_z_or_fluence():
    stack = _ring_stack()
    baseline, _, metrics = _perturbed_metrics(
        stack,
        {"enable_defocus": True, "focus_offset_um": 50.0},
    )
    changed_z = abs(metrics["peak_z_um"] - baseline["peak_z_um"]) > 1.0
    changed_fluence = abs(metrics["peak_fluence_change_fraction"]) > 1e-3
    assert changed_z or changed_fluence


def test_coma_changes_asymmetry_or_side_lobe_imbalance():
    stack = _ring_stack()
    baseline, _, metrics = _perturbed_metrics(
        stack,
        {"enable_zernike_aberrations": True, "zernike_coma_x_waves": 0.3},
    )
    assert (
        metrics["symmetry_score"] < baseline["symmetry_score"]
        or metrics["side_lobe_imbalance"] > baseline["side_lobe_imbalance"]
    )


def test_scan_speed_still_changes_exposure_bookkeeping():
    slow = line_exposure_summary(10.0, 25_000.0, 1.0, 500.0, 3.0)
    fast = line_exposure_summary(10.0, 25_000.0, 2.0, 500.0, 3.0)
    assert fast["pulse_spacing_um"] > slow["pulse_spacing_um"]
    assert fast["pulses_per_spot"] < slow["pulses_per_spot"]
