from __future__ import annotations

import math

import numpy as np
import pytest

from vbb_study.calibration.schema import CalibrationBundle, canonical_calibration_template, measurement
from vbb_study.calibration.slm_phase import SLMPhaseCalibration
from vbb_study.digital_twin.bench_calibrated_vector_route import (
    BenchCalibratedVectorInputs,
    build_calibrated_segmented_vector_route,
    calibrated_vector_route_to_sample,
)
from vbb_study.digital_twin.nathan_vector_hexagon import NathanHexagonConfig
from vbb_study.digital_twin.objective_pupil_mapping import ObjectivePupilMappingConfig
from vbb_study.digital_twin.objective_sample_route import (
    ObjectiveSampleConfig,
    focus_vector_pupil_into_sample,
)


TWOPI = 2.0 * np.pi


def _lut(panel: str) -> SLMPhaseCalibration:
    grey = np.arange(256, dtype=float)
    return SLMPhaseCalibration(
        panel_id=panel,
        wavelength_m=1029e-9,
        grey_levels=grey,
        phase_rad=TWOPI * grey / 255.0,
        calibration_date="synthetic",
    )


def _synthetic_bundle() -> CalibrationBundle:
    data = canonical_calibration_template()
    data["calibration_id"] = "synthetic_vector_phase2g"
    data["data_classification"] = "synthetic_not_experimental"
    data["laser"]["beam_radius_on_slm_m"] = measurement(2.0e-3, 0.0, "synthetic_measurement", "m")
    data["fourier_filter"]["focal_length_m"] = measurement(0.300, 0.0, "synthetic_measurement", "m")
    # Roughly equivalent to a 2-2.5 lp/mm spectral-radius selection at f=300 mm.
    data["fourier_filter"]["iris_radius_m"] = measurement(0.70e-3, 0.0, "synthetic_measurement", "m")
    data["axicon"]["base_angle_deg"] = measurement(2.0, 0.0, "synthetic_measurement", "deg")
    data["axicon"]["refractive_index"] = measurement(1.458, 0.0, "synthetic_measurement", "1")
    data["axicon"]["clear_radius_m"] = measurement(3.0e-3, 0.0, "synthetic_measurement", "m")

    pol = data["polarization"]
    pol["input_linear_angle_deg"] = measurement(45.0, 0.0, "synthetic_measurement", "deg")
    pol["input_degree_linear_polarization"] = measurement(1.0, 0.0, "synthetic_measurement", "1")
    pol["input_relative_phase_rad"] = measurement(0.0, 0.0, "synthetic_measurement", "rad")
    pol["slm_director_axis_deg"] = measurement(0.0, 0.0, "synthetic_measurement", "deg")
    pol["segmented_vector_hwp_retardance_rad"] = measurement(math.pi, 0.0, "synthetic_measurement", "rad")
    pol["segmented_vector_hwp_fast_axis_deg"] = measurement(45.0, 0.0, "synthetic_measurement", "deg")
    # retarder_jones uses the fast-axis convention for which -45 deg reproduces
    # the established qwp45_matrix used by vector_arm_chain.
    pol["segmented_vector_qwp_retardance_rad"] = measurement(0.5 * math.pi, 0.0, "synthetic_measurement", "rad")
    pol["segmented_vector_qwp_fast_axis_deg"] = measurement(-45.0, 0.0, "synthetic_measurement", "deg")
    return CalibrationBundle(data)


def test_spatially_varying_vector_pupil_reaches_vector_debye_solver() -> None:
    n = 64
    x = np.linspace(-1.8e-3, 1.8e-3, n)
    X, Y = np.meshgrid(x, x, indexing="xy")
    R = np.hypot(X, Y)
    phi = np.arctan2(Y, X)
    amp = np.exp(-(R / 1.0e-3) ** 2)
    ex = amp * np.cos(phi)
    ey = amp * np.sin(phi)
    result = focus_vector_pupil_into_sample(
        ex.astype(complex),
        ey.astype(complex),
        {"x": x, "y": x, "dx": float(x[1] - x[0])},
        config=ObjectiveSampleConfig(
            wavelength_m=1029e-9,
            numerical_aperture=0.45,
            objective_focal_length_m=4.0e-3,
            objective_pupil_radius_m=1.6e-3,
            sample_refractive_index=1.45,
            sample_depth_m=0.0,
            fft_pad_factor=1,
        ),
    )
    assert result.metadata["input_polarization_model"] == "spatially_varying_Ex_Ey"
    assert result.focal_plane_air.component_power_fractions["Ez_power_fraction"] > 0.0
    assert np.all(np.isfinite(result.field_in_sample.intensity))


def test_calibrated_segmented_vector_route_uses_20px_common_order() -> None:
    cfg = NathanHexagonConfig.fast(grid_n=128)
    route = build_calibrated_segmented_vector_route(
        cfg,
        calibrated=BenchCalibratedVectorInputs(
            calibration_bundle=_synthetic_bundle(),
            slm1_phase_calibration=_lut("synthetic_slm1"),
            slm2_phase_calibration=_lut("synthetic_slm2"),
        ),
    )
    meta = route["metadata"]
    assert meta["carrier_period_px"] == pytest.approx(20.0)
    assert meta["selected_common_carrier_cpm"] == pytest.approx(-6250.0)
    assert route["post_4f_selected_order"].power > 0.0
    assert route["post_axicon"].power > 0.0
    assert np.all(np.isfinite(route["post_axicon"].intensity))
    # This is a hardware-realistic quantised/fill-factor route, so it need not
    # be exactly the analytic target; it must nevertheless preserve the intended
    # vector encoder rather than collapse into an unrelated Jones field.
    assert meta["encoder_target_comparison"]["complex_overlap"] > 0.70


def test_calibrated_segmented_vector_route_can_reach_sample_without_scalarising_polarization() -> None:
    cfg = NathanHexagonConfig.fast(grid_n=128)
    route = build_calibrated_segmented_vector_route(
        cfg,
        calibrated=BenchCalibratedVectorInputs(
            calibration_bundle=_synthetic_bundle(),
            slm1_phase_calibration=_lut("synthetic_slm1"),
            slm2_phase_calibration=_lut("synthetic_slm2"),
        ),
    )
    result = calibrated_vector_route_to_sample(
        route,
        mapping_config=ObjectivePupilMappingConfig(
            free_space_distance_m=0.0,
            output_window_m=4.0e-3,
            output_n=64,
            pupil_radius_m=1.6e-3,
        ),
        objective_config=ObjectiveSampleConfig(
            wavelength_m=1029e-9,
            numerical_aperture=0.45,
            objective_focal_length_m=4.0e-3,
            objective_pupil_radius_m=1.6e-3,
            sample_refractive_index=1.45,
            sample_depth_m=2.0e-6,
            fft_pad_factor=1,
        ),
    )
    assert result["metadata"]["spatially_varying_vector_pupil_preserved"] is True
    sample = result["sample_result"]
    assert sample.metadata["input_polarization_model"] == "spatially_varying_Ex_Ey"
    assert sample.field_in_sample.power > 0.0
    assert np.all(np.isfinite(sample.field_in_sample.intensity))


def test_calibrated_vector_axicon_tilt_is_not_faked_with_scalar_reference() -> None:
    cfg = NathanHexagonConfig.fast(grid_n=64)
    with pytest.raises(ValueError, match="full vector two-surface"):
        build_calibrated_segmented_vector_route(
            cfg,
            calibrated=BenchCalibratedVectorInputs(
                calibration_bundle=_synthetic_bundle(),
                slm1_phase_calibration=_lut("synthetic_slm1"),
                slm2_phase_calibration=_lut("synthetic_slm2"),
                axicon_tilt_rad=(1e-3, 0.0),
            ),
        )
