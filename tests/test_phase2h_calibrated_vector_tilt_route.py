from __future__ import annotations

import math

import numpy as np
import pytest

from vbb_study.calibration.schema import CalibrationBundle, canonical_calibration_template, measurement
from vbb_study.calibration.slm_phase import SLMPhaseCalibration
from vbb_study.digital_twin.bench_calibrated_vector_route import (
    BenchCalibratedVectorInputs,
    calibrated_vector_route_to_sample,
)
from vbb_study.digital_twin.bench_calibrated_vector_tilt_route import (
    build_calibrated_segmented_vector_tilt_route,
    refractive_axicon_geometry_from_calibration,
)
from vbb_study.digital_twin.nathan_vector_hexagon import NathanHexagonConfig
from vbb_study.digital_twin.objective_pupil_mapping import ObjectivePupilMappingConfig
from vbb_study.digital_twin.objective_sample_route import ObjectiveSampleConfig


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


def _bundle(*, convention: str = "base_angle_from_flat_face", flat_first: bool = True) -> CalibrationBundle:
    data = canonical_calibration_template()
    data["calibration_id"] = "synthetic_phase2h_vector_tilt"
    data["data_classification"] = "synthetic_not_experimental"
    data["laser"]["beam_radius_on_slm_m"] = measurement(2.0e-3, 0.0, "synthetic_measurement", "m")
    data["fourier_filter"]["focal_length_m"] = measurement(0.300, 0.0, "synthetic_measurement", "m")
    data["fourier_filter"]["iris_radius_m"] = measurement(0.70e-3, 0.0, "synthetic_measurement", "m")

    axicon = data["axicon"]
    axicon["base_angle_deg"] = measurement(2.0, 0.0, "synthetic_measurement", "deg")
    axicon["refractive_index"] = measurement(1.458, 0.0, "synthetic_measurement", "1")
    axicon["clear_radius_m"] = measurement(3.0e-3, 0.0, "synthetic_measurement", "m")
    axicon["centre_thickness_m"] = measurement(3.0e-3, 0.0, "synthetic_measurement", "m")
    axicon["angle_convention"] = convention
    axicon["flat_face_upstream_verified"] = bool(flat_first)

    pol = data["polarization"]
    pol["input_linear_angle_deg"] = measurement(45.0, 0.0, "synthetic_measurement", "deg")
    pol["input_degree_linear_polarization"] = measurement(1.0, 0.0, "synthetic_measurement", "1")
    pol["input_relative_phase_rad"] = measurement(0.0, 0.0, "synthetic_measurement", "rad")
    pol["slm_director_axis_deg"] = measurement(0.0, 0.0, "synthetic_measurement", "deg")
    pol["segmented_vector_hwp_retardance_rad"] = measurement(math.pi, 0.0, "synthetic_measurement", "rad")
    pol["segmented_vector_hwp_fast_axis_deg"] = measurement(45.0, 0.0, "synthetic_measurement", "deg")
    pol["segmented_vector_qwp_retardance_rad"] = measurement(0.5 * math.pi, 0.0, "synthetic_measurement", "rad")
    pol["segmented_vector_qwp_fast_axis_deg"] = measurement(-45.0, 0.0, "synthetic_measurement", "deg")
    return CalibrationBundle(data)


def _inputs(bundle: CalibrationBundle, tilt_deg: float = 5.0) -> BenchCalibratedVectorInputs:
    return BenchCalibratedVectorInputs(
        calibration_bundle=bundle,
        slm1_phase_calibration=_lut("synthetic_slm1"),
        slm2_phase_calibration=_lut("synthetic_slm2"),
        axicon_tilt_rad=(0.0, math.radians(tilt_deg)),
    )


def test_phase2h_requires_explicit_base_angle_convention_and_surface_order() -> None:
    with pytest.raises(ValueError, match="base_angle_from_flat_face"):
        refractive_axicon_geometry_from_calibration(_inputs(_bundle(convention="apex_angle")))
    with pytest.raises(ValueError, match="flat_face_upstream_verified"):
        refractive_axicon_geometry_from_calibration(_inputs(_bundle(flat_first=False)))


def test_calibrated_six_sector_vector_field_passes_two_surface_tilt_solver() -> None:
    route = build_calibrated_segmented_vector_tilt_route(
        NathanHexagonConfig.fast(grid_n=128),
        calibrated=_inputs(_bundle(), tilt_deg=5.0),
        vector_axicon_output_n=512,
        vector_axicon_output_window_m=7.2e-3,
        reference_gap_m=0.25e-3,
    )
    result = route["vector_refractive_axicon_result"]
    assert route["metadata"]["vector_rigid_axicon_tilt_supported"] is True
    assert route["metadata"]["axicon_model"] == "phase2h_common_eikonal_two_surface_vector_refractive"
    assert result.metadata["ray_direction_definition"] == "local_wavevector_grad_Phi"
    assert result.metadata["poynting_role"] == "independent_energy_flux_diagnostic_only"
    assert result.metadata["common_eikonal"]["p95_component_wavevector_disagreement_fraction"] < 0.02
    assert result.metadata["common_eikonal"]["p95_reconstructed_gradient_error_fraction"] < 0.01
    assert abs(float(result.metadata["final_flux_closure_ratio"]) - 1.0) < 2e-12
    assert float(result.metadata["final_transversality_residual"]) < 1e-10
    assert float(result.metadata["required_nyquist_fraction"]) < 0.90
    assert route["post_axicon"].power > 0.0
    assert np.all(np.isfinite(route["post_axicon"].intensity))


def test_phase2h_tilted_segmented_vector_field_reaches_sample_without_scalarisation() -> None:
    route = build_calibrated_segmented_vector_tilt_route(
        NathanHexagonConfig.fast(grid_n=128),
        calibrated=_inputs(_bundle(), tilt_deg=5.0),
        vector_axicon_output_n=512,
        vector_axicon_output_window_m=7.2e-3,
        reference_gap_m=0.25e-3,
    )
    sample_route = calibrated_vector_route_to_sample(
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
    assert sample_route["metadata"]["spatially_varying_vector_pupil_preserved"] is True
    sample = sample_route["sample_result"]
    assert sample.metadata["input_polarization_model"] == "spatially_varying_Ex_Ey"
    assert sample.field_in_sample.power > 0.0
    assert np.all(np.isfinite(sample.field_in_sample.intensity))
