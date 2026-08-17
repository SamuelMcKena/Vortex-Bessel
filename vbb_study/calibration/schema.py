"""Schema objects and null-preserving canonical calibration template."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping


CALIBRATION_SCHEMA_VERSION = "1.1"
DATA_CLASSIFICATIONS = {"laboratory_measurement", "synthetic_not_experimental", "template_unpopulated"}


@dataclass(frozen=True)
class CalibrationBundle:
    data: dict[str, Any]

    @property
    def schema_version(self) -> str:
        return str(self.data.get("schema_version", ""))

    @property
    def calibration_id(self) -> str:
        return str(self.data.get("calibration_id", ""))

    @property
    def data_classification(self) -> str:
        return str(self.data.get("data_classification", "template_unpopulated"))

    @property
    def is_synthetic(self) -> bool:
        return self.data_classification == "synthetic_not_experimental"

    def copy_data(self) -> dict[str, Any]:
        return deepcopy(self.data)


def measurement(value: Any, uncertainty: Any, source: str, unit: str, **extra: Any) -> dict[str, Any]:
    row = {"value": value, "uncertainty": uncertainty, "source": source, "unit": unit}
    row.update(extra)
    return row


def canonical_calibration_template() -> dict[str, Any]:
    """Return an editable bundle; nulls are deliberate and never auto-filled.

    Version 1.1 keeps the original Phase 2D keys but adds the calibration state
    required by the Phase 2G bench-calibrated digital twin. The physical bench
    carrier is fixed at 20 pixels on the 8 um SLM, i.e. 6.25 lp/mm.
    """

    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "calibration_id": "lab_YYYYMMDD",
        "created_utc": None,
        "operator": "",
        "data_classification": "template_unpopulated",
        "laser": {
            "wavelength_m": measurement(1.029e-6, None, "manufacturer", "m"),
            "pulse_energy_J": measurement(None, None, "calibration_required", "J"),
            "pulse_duration_s": measurement(None, None, "calibration_required", "s"),
            "repetition_rate_Hz": measurement(None, None, "calibration_required", "Hz"),
            "beam_radius_on_slm_m": measurement(
                None,
                None,
                "calibration_required",
                "m",
                definition="1/e^2 intensity radius = 1/e field-amplitude radius",
            ),
            "beam_radius_y_on_slm_m": measurement(None, None, "calibration_required", "m"),
            "beam_M2_x": measurement(None, None, "calibration_required", "1"),
            "beam_M2_y": measurement(None, None, "calibration_required", "1"),
        },
        "slm": {
            "make": "HOLOEYE",
            "model": "PLUTO-2.1 NIR-149",
            "panel_ids": [None, None],
            "active_resolution": [1920, 1080],
            "pixel_pitch_m": 8e-6,
            "fill_factor": 0.93,
            "phase_bits": 8,
            "carrier_period_px": 20,
            "carrier_lp_per_mm": 6.25,
            "slm1_phase_lut_path": None,
            "slm2_phase_lut_path": None,
            "slm1_phase_stroke_rad": None,
            "slm2_phase_stroke_rad": None,
            # Backward-compatible aggregate keys retained for Phase 2D readers.
            "phase_lut_path": None,
            "phase_stroke_rad": None,
            "phase_stroke_uncertainty_rad": None,
            "panel_orientation_verified": False,
            "slm1_panel_orientation_verified": False,
            "slm2_panel_orientation_verified": False,
            "calibration_date": None,
            "slm1_static_phase_map_path": None,
            "slm2_static_phase_map_path": None,
            "slm1_fringe_calibration_path": None,
            "slm2_fringe_calibration_path": None,
        },
        "polarization": {
            "input_linear_angle_deg": measurement(None, None, "calibration_required", "deg"),
            "input_degree_linear_polarization": measurement(None, None, "calibration_required", "1"),
            "input_relative_phase_rad": measurement(None, None, "calibration_required", "rad"),
            "slm_director_axis_deg": measurement(None, None, "calibration_required", "deg"),
            "analyzer_0_actual_deg": measurement(None, None, "calibration_required", "deg"),
            "analyzer_45_actual_deg": measurement(None, None, "calibration_required", "deg"),
            "analyzer_90_actual_deg": measurement(None, None, "calibration_required", "deg"),
            "analyzer_135_actual_deg": measurement(None, None, "calibration_required", "deg"),
            "analyzer_extinction_ratio": measurement(None, None, "calibration_required", "1"),
            "analyzer_transmission": measurement(None, None, "calibration_required", "1"),
            "linear_analyzer_evidence_path": None,
            "full_stokes_qwp_present": None,
            "full_stokes_qwp_retardance_rad": measurement(None, None, "calibration_required", "rad"),
            "full_stokes_qwp_fast_axis_deg": measurement(None, None, "calibration_required", "deg"),
            "segmented_vector_hwp_retardance_rad": measurement(None, None, "calibration_required", "rad"),
            "segmented_vector_hwp_fast_axis_deg": measurement(None, None, "calibration_required", "deg"),
            "segmented_vector_qwp_retardance_rad": measurement(None, None, "calibration_required", "rad"),
            "segmented_vector_qwp_fast_axis_deg": measurement(None, None, "calibration_required", "deg"),
            "segmented_vector_optics_evidence_path": None,
        },
        "wavefront_sensor": {
            "make": None,
            "model": None,
            "lenslet_pitch_m": measurement(None, None, "calibration_required", "m"),
            "lenslet_focal_length_m": measurement(None, None, "calibration_required", "m"),
            "sensor_pixel_pitch_m": measurement(None, None, "calibration_required", "m"),
            "reference_centroids_path": None,
            "latest_centroids_path": None,
            "latest_slopes_path": None,
            "slm_registration_rotation_deg": measurement(None, None, "calibration_required", "deg"),
            "slm_registration_scale_x": measurement(None, None, "calibration_required", "1"),
            "slm_registration_scale_y": measurement(None, None, "calibration_required", "1"),
            "slm_registration_offset_x_m": measurement(None, None, "calibration_required", "m"),
            "slm_registration_offset_y_m": measurement(None, None, "calibration_required", "m"),
            "residual_opd_rms_m": measurement(None, None, "calibration_required", "m"),
            "correction_map_path": None,
            "correction_iteration": 0,
        },
        "fourier_filter": {
            "focal_length_m": measurement(0.300, None, "assumed", "m"),
            "iris_radius_m": measurement(None, None, "calibration_required", "m"),
            "plus_one_position_m": measurement(None, None, "calibration_required", "m"),
            "zero_order_position_m": measurement(None, None, "calibration_required", "m"),
            "selected_order_power_fraction": measurement(None, None, "calibration_required", "1"),
        },
        "objective": {
            "make": None,
            "model": None,
            "numerical_aperture": measurement(0.45, None, "manufacturer", "1"),
            "focal_length_m": measurement(None, None, "calibration_required", "m"),
            "effective_pupil_radius_m": measurement(None, None, "calibration_required", "m"),
            "pupil_fill_fraction": measurement(None, None, "calibration_required", "1"),
            "entrance_pupil_mapping_magnification": measurement(None, None, "calibration_required", "1"),
            "entrance_pupil_rotation_deg": measurement(None, None, "calibration_required", "deg"),
            "entrance_pupil_offset_x_m": measurement(None, None, "calibration_required", "m"),
            "entrance_pupil_offset_y_m": measurement(None, None, "calibration_required", "m"),
        },
        "relay": {
            "magnification": measurement(None, None, "calibration_required", "1"),
            "component_ids": [],
            "distances_m": [],
        },
        "axicon": {
            "make": None,
            "model": None,
            "angle_convention": None,
            "base_angle_deg": measurement(2.0, None, "assumed", "deg"),
            "clear_aperture_m": measurement(None, None, "calibration_required", "m"),
            "clear_radius_m": measurement(None, None, "calibration_required", "m"),
            "centre_thickness_m": measurement(None, None, "calibration_required", "m"),
            "refractive_index": measurement(1.458, None, "assumed", "1"),
            "flat_face_upstream_verified": False,
            "coating_state": "unknown",
            "tip_profile_path": None,
            "surface_map_path": None,
        },
        "camera": {
            "make": "Gentec-EO",
            "model": "Beamage 4M",
            "pixel_pitch_m": measurement(None, None, "calibration_required", "m"),
            "magnification": measurement(None, None, "calibration_required", "1"),
            "object_plane_scale_m_per_pixel": measurement(None, None, "calibration_required", "m/pixel"),
            "rotation_deg": measurement(None, None, "calibration_required", "deg"),
            "centre_pixel": None,
            "background_frame_path": None,
            "saturation_level": None,
            "exposure_s": None,
            "attenuation_stack": [],
        },
        "temporal": {
            "spectrum_path": None,
            "spectral_phase_path": None,
            "measured_intensity_fwhm_s": measurement(None, None, "calibration_required", "s"),
            "gdd_s2": measurement(None, None, "calibration_required", "s^2"),
            "tod_s3": measurement(None, None, "calibration_required", "s^3"),
            "measurement_method": None,
        },
        "transmissions": {
            name: measurement(None, None, "calibration_required", "1")
            for name in ("slm1", "slm2", "four_f", "axicon", "objective", "interface")
        },
        "material": {
            "name": "fused silica",
            "refractive_index": measurement(None, None, "calibration_required", "1"),
            "coating_state": "unknown",
            "surface_orientation_verified": False,
            "response_dataset_path": None,
            "response_model_path": None,
        },
        "experimental_comparison": {
            "performed": False,
            "evidence_path": None,
            "acceptance_passed": False,
            "case_ids": [],
            "z_positions_m": [],
            "vector_analyzer_evidence_path": None,
            "vector_case_ids": [],
        },
    }


def value_at(bundle: CalibrationBundle | Mapping[str, Any], path: str) -> Any:
    data: Mapping[str, Any] = bundle.data if isinstance(bundle, CalibrationBundle) else bundle
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    if isinstance(current, Mapping) and "value" in current:
        return current.get("value")
    return current


def uncertainty_at(bundle: CalibrationBundle | Mapping[str, Any], path: str) -> float | None:
    data: Mapping[str, Any] = bundle.data if isinstance(bundle, CalibrationBundle) else bundle
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    if isinstance(current, Mapping):
        value = current.get("uncertainty")
        return None if value is None else float(value)
    return None


def source_at(bundle: CalibrationBundle | Mapping[str, Any], path: str) -> str:
    data: Mapping[str, Any] = bundle.data if isinstance(bundle, CalibrationBundle) else bundle
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return "missing"
        current = current[part]
    if isinstance(current, Mapping):
        return str(current.get("source", "unspecified"))
    if current is None:
        return "missing"
    return "supplied"
