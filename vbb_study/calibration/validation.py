"""Validation and claim-dependency governance for calibration bundles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from vbb_study.calibration.schema import (
    CALIBRATION_SCHEMA_VERSION,
    DATA_CLASSIFICATIONS,
    CalibrationBundle,
    source_at,
    value_at,
)
from vbb_study.solver_policy import CLAIM_TYPES


SUPPORTED_CALIBRATION_SCHEMA_VERSIONS = frozenset({"1.0", CALIBRATION_SCHEMA_VERSION})


@dataclass(frozen=True)
class CalibrationValidationReport:
    valid_schema: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    missing_values: tuple[str, ...]
    inconsistent_units: tuple[str, ...]
    physically_impossible_values: tuple[str, ...]


@dataclass(frozen=True)
class CalibrationReadiness:
    claim_type: str
    ready: bool
    status: str
    missing_measurements: tuple[str, ...]
    non_calibrated_measurements: tuple[str, ...]
    satisfied_measurements: tuple[str, ...]


@dataclass(frozen=True)
class CalibrationDependency:
    path: str
    required_or_optional: str
    notes: str


_DIMENSION_SCALE = (
    CalibrationDependency("laser.wavelength_m", "required", "Optical length scale."),
    CalibrationDependency("fourier_filter.focal_length_m", "required", "Measured 4F pupil mapping geometry."),
    CalibrationDependency("fourier_filter.iris_radius_m", "required", "Measured selected-order spatial support."),
    CalibrationDependency("objective.numerical_aperture", "required", "Focal spatial-frequency support."),
    CalibrationDependency("objective.focal_length_m", "required", "Objective focal geometry."),
    CalibrationDependency("objective.effective_pupil_radius_m", "required", "Effective rather than nominal pupil."),
    CalibrationDependency("objective.pupil_fill_fraction", "required", "Measured pupil fill controls focal detail."),
    CalibrationDependency("relay.magnification", "required", "Maps objective and observation planes."),
    CalibrationDependency("camera.object_plane_scale_m_per_pixel", "required", "Measured object-plane image scale."),
)

CLAIM_DEPENDENCIES: dict[str, tuple[CalibrationDependency, ...]] = {
    "global_transverse_morphology": (),
    "feature_radius": _DIMENSION_SCALE,
    "ring_radius": _DIMENSION_SCALE,
    "peak_location": _DIMENSION_SCALE,
    "edge_sharpness": _DIMENSION_SCALE,
    "ridge_width": _DIMENSION_SCALE,
    "transition_width": _DIMENSION_SCALE,
    "longitudinal_field": (
        CalibrationDependency("laser.wavelength_m", "required", "Vector Debye wavelength."),
        CalibrationDependency("objective.numerical_aperture", "required", "Sets longitudinal coupling."),
        CalibrationDependency("objective.pupil_fill_fraction", "required", "Sets illuminated angular support."),
    ),
    "polarisation_component": (
        CalibrationDependency("laser.wavelength_m", "required", "Vector propagation wavelength."),
        CalibrationDependency("objective.numerical_aperture", "required", "Vector objective geometry."),
        CalibrationDependency("objective.pupil_fill_fraction", "required", "Illuminated vector pupil."),
    ),
    "interface_power": (
        CalibrationDependency("laser.wavelength_m", "required", "Interface wavelength."),
        CalibrationDependency("material.refractive_index", "required", "Fresnel material binding."),
        CalibrationDependency("material.coating_state", "required", "Unknown coating changes transmission."),
        CalibrationDependency("material.surface_orientation_verified", "required", "Defines the interface normal."),
    ),
    "absolute_dimensions": _DIMENSION_SCALE,
    "absolute_fluence": (
        CalibrationDependency("laser.pulse_energy_J", "required", "Measured incident pulse energy."),
        CalibrationDependency("camera.object_plane_scale_m_per_pixel", "required", "Physical pixel area."),
        CalibrationDependency("transmissions.slm1", "required", "Measured once in the power ledger."),
        CalibrationDependency("transmissions.slm2", "required", "Measured once in the power ledger."),
        CalibrationDependency("transmissions.four_f", "required", "Common 4F throughput excluding selected-order factor."),
        CalibrationDependency("transmissions.objective", "required", "Objective throughput."),
        CalibrationDependency("transmissions.interface", "required", "Interface throughput."),
    ),
}

EXTRA_DEPENDENCIES: dict[str, tuple[CalibrationDependency, ...]] = {
    "fourier_filter_geometry": (
        CalibrationDependency("laser.wavelength_m", "required", "Carrier displacement wavelength."),
        CalibrationDependency("fourier_filter.focal_length_m", "required", "Measured carrier-displacement slope."),
        CalibrationDependency("fourier_filter.iris_radius_m", "required", "Measured selected-order aperture."),
        CalibrationDependency("fourier_filter.plus_one_position_m", "required", "Measured selected-order centre."),
    ),
    "slm_phase_fidelity": (
        CalibrationDependency("slm.slm1_phase_lut_path", "required", "1029-nm measured SLM1 LUT."),
        CalibrationDependency("slm.slm2_phase_lut_path", "required", "1029-nm measured SLM2 LUT."),
        CalibrationDependency("slm.slm1_phase_stroke_rad", "required", "Measured SLM1 optical phase stroke."),
        CalibrationDependency("slm.slm2_phase_stroke_rad", "required", "Measured SLM2 optical phase stroke."),
        CalibrationDependency("slm.slm1_panel_orientation_verified", "required", "SLM1 LC/director orientation."),
        CalibrationDependency("slm.slm2_panel_orientation_verified", "required", "SLM2 LC/director orientation."),
        CalibrationDependency("slm.calibration_date", "required", "Calibration provenance date."),
        CalibrationDependency("laser.wavelength_m", "required", "LUT operating wavelength."),
    ),
    "slm_phase_fidelity_legacy": (
        CalibrationDependency("slm.phase_lut_path", "required", "Wavelength-specific panel LUT."),
        CalibrationDependency("slm.phase_stroke_rad", "required", "Measured phase stroke."),
        CalibrationDependency("slm.panel_orientation_verified", "required", "LC director orientation."),
        CalibrationDependency("slm.calibration_date", "required", "Calibration provenance date."),
        CalibrationDependency("laser.wavelength_m", "required", "LUT operating wavelength."),
    ),
    "wavefront_correction": (
        CalibrationDependency("wavefront_sensor.lenslet_pitch_m", "required", "Shack-Hartmann sampling geometry."),
        CalibrationDependency("wavefront_sensor.lenslet_focal_length_m", "required", "Converts spot displacement to slope."),
        CalibrationDependency("wavefront_sensor.reference_centroids_path", "required", "Reference centroid map."),
        CalibrationDependency("wavefront_sensor.latest_slopes_path", "required", "Measured residual slopes."),
        CalibrationDependency("wavefront_sensor.slm_registration_rotation_deg", "required", "SH-to-SLM rotation."),
        CalibrationDependency("wavefront_sensor.slm_registration_scale_x", "required", "SH-to-SLM x scale."),
        CalibrationDependency("wavefront_sensor.slm_registration_scale_y", "required", "SH-to-SLM y scale."),
        CalibrationDependency("wavefront_sensor.slm_registration_offset_x_m", "required", "SH-to-SLM x offset."),
        CalibrationDependency("wavefront_sensor.slm_registration_offset_y_m", "required", "SH-to-SLM y offset."),
        CalibrationDependency("wavefront_sensor.correction_map_path", "required", "Frozen correction applied before structured phase."),
    ),
    "bessel_zone_length": (
        CalibrationDependency("laser.beam_radius_on_slm_m", "required", "Measured input radius."),
        CalibrationDependency("axicon.base_angle_deg", "required", "Measured axicon angle."),
        CalibrationDependency("axicon.refractive_index", "required", "Index at operating wavelength."),
        CalibrationDependency("axicon.clear_aperture_m", "required", "Clipping bound."),
    ),
    "refractive_axicon_tilt": (
        CalibrationDependency("axicon.base_angle_deg", "required", "Physical base angle using declared convention."),
        CalibrationDependency("axicon.clear_radius_m", "required", "Finite refractive surface radius."),
        CalibrationDependency("axicon.centre_thickness_m", "required", "Second surface location."),
        CalibrationDependency("axicon.refractive_index", "required", "Material index at 1029 nm."),
        CalibrationDependency("axicon.angle_convention", "required", "Prevents apex/base-angle ambiguity."),
        CalibrationDependency("axicon.flat_face_upstream_verified", "required", "Defines surface order."),
    ),
    "objective_sample_vector_field": (
        CalibrationDependency("objective.numerical_aperture", "required", "Richards-Wolf angular support."),
        CalibrationDependency("objective.focal_length_m", "required", "Objective geometry."),
        CalibrationDependency("objective.effective_pupil_radius_m", "required", "Entrance-pupil radius."),
        CalibrationDependency("objective.entrance_pupil_mapping_magnification", "required", "Maps source field to objective pupil."),
        CalibrationDependency("material.refractive_index", "required", "Vector Fresnel and in-material propagation."),
        CalibrationDependency("material.surface_orientation_verified", "required", "Planar-interface normal."),
    ),
    "vector_analyzer_spots": (
        CalibrationDependency("camera.object_plane_scale_m_per_pixel", "required", "Physical scale for ring/petal radius."),
        CalibrationDependency("camera.rotation_deg", "required", "Camera-to-lab orientation for petal angle."),
        CalibrationDependency("polarization.analyzer_0_actual_deg", "required", "Measured 0-degree analyzer axis."),
        CalibrationDependency("polarization.analyzer_45_actual_deg", "required", "Measured 45-degree analyzer axis."),
        CalibrationDependency("polarization.analyzer_90_actual_deg", "required", "Measured 90-degree analyzer axis."),
        CalibrationDependency("polarization.analyzer_135_actual_deg", "required", "Measured 135-degree analyzer axis."),
        CalibrationDependency("polarization.analyzer_extinction_ratio", "required", "Finite analyzer leakage."),
        CalibrationDependency("polarization.analyzer_transmission", "required", "Analyzer throughput."),
        CalibrationDependency("polarization.linear_analyzer_evidence_path", "required", "0/45/90/135-degree calibration evidence."),
        CalibrationDependency("experimental_comparison.vector_analyzer_evidence_path", "required", "Measured vector analyzer frames."),
    ),
    "vector_case1_hardware": (
        CalibrationDependency("laser.beam_radius_on_slm_m", "required", "Physical input Gaussian on the SLMs."),
        CalibrationDependency("slm.slm1_phase_lut_path", "required", "Measured phase response for the shaped channel."),
        CalibrationDependency("slm.slm1_phase_stroke_rad", "required", "Measured optical phase stroke."),
        CalibrationDependency("polarization.input_linear_angle_deg", "required", "Measured input H/V amplitude split."),
        CalibrationDependency("polarization.input_degree_linear_polarization", "required", "Input polarization purity."),
        CalibrationDependency("polarization.slm_director_axis_deg", "required", "Physical shaped SLM axis."),
        CalibrationDependency("fourier_filter.iris_radius_m", "required", "Selected-order support."),
        CalibrationDependency("fourier_filter.plus_one_position_m", "required", "Selected-order centre."),
    ),
    "segmented_vector_hexagon": (
        CalibrationDependency("slm.slm1_phase_lut_path", "required", "Measured SLM1 phase response."),
        CalibrationDependency("slm.slm2_phase_lut_path", "required", "Measured SLM2 phase response."),
        CalibrationDependency("polarization.input_linear_angle_deg", "required", "Input polarization state."),
        CalibrationDependency("polarization.slm_director_axis_deg", "required", "SLM/polarization basis registration."),
        CalibrationDependency("polarization.segmented_vector_hwp_retardance_rad", "required", "Measured HWP retardance."),
        CalibrationDependency("polarization.segmented_vector_hwp_fast_axis_deg", "required", "Measured HWP orientation."),
        CalibrationDependency("polarization.segmented_vector_qwp_retardance_rad", "required", "Measured QWP retardance."),
        CalibrationDependency("polarization.segmented_vector_qwp_fast_axis_deg", "required", "Measured QWP orientation."),
        CalibrationDependency("polarization.segmented_vector_optics_evidence_path", "required", "Vector encoder optics calibration evidence."),
        CalibrationDependency("fourier_filter.iris_radius_m", "required", "Vector selected-order filtering."),
        CalibrationDependency("camera.object_plane_scale_m_per_pixel", "required", "Measured hexagon/spot scale."),
    ),
    "full_stokes_vector": (
        CalibrationDependency("polarization.full_stokes_qwp_present", "required", "A QWP analyzer is required for S3."),
        CalibrationDependency("polarization.full_stokes_qwp_retardance_rad", "required", "Measured QWP retardance for S3."),
        CalibrationDependency("polarization.full_stokes_qwp_fast_axis_deg", "required", "Measured QWP fast-axis orientation."),
    ),
    "spatiotemporal_field": (
        CalibrationDependency("laser.pulse_duration_s", "required", "Pulse temporal scale."),
        CalibrationDependency("temporal.spectrum_path", "required", "Measured spectral amplitude."),
        CalibrationDependency("temporal.gdd_s2", "required", "Measured/characterised second-order spectral phase."),
        CalibrationDependency("temporal.measurement_method", "required", "Temporal calibration provenance."),
    ),
    "experimental_agreement": (
        CalibrationDependency("camera.object_plane_scale_m_per_pixel", "required", "Physical camera scale."),
        CalibrationDependency("camera.rotation_deg", "required", "Lab/camera coordinate registration."),
        CalibrationDependency("camera.background_frame_path", "required", "Background subtraction."),
        CalibrationDependency("experimental_comparison.evidence_path", "required", "Measured G0/B0/V1/V3 evidence."),
        CalibrationDependency("experimental_comparison.performed", "required", "Comparison was actually executed."),
        CalibrationDependency("experimental_comparison.acceptance_passed", "required", "Predeclared comparison gate passed."),
    ),
    "material_response": (
        CalibrationDependency("material.response_dataset_path", "required", "Measured processing response dataset."),
        CalibrationDependency("material.response_model_path", "required", "Frozen fitted response model with validation."),
    ),
}

EXPECTED_UNITS = {
    "laser.wavelength_m": "m",
    "laser.pulse_energy_J": "J",
    "laser.beam_radius_on_slm_m": "m",
    "fourier_filter.focal_length_m": "m",
    "fourier_filter.iris_radius_m": "m",
    "fourier_filter.plus_one_position_m": "m",
    "objective.numerical_aperture": "1",
    "objective.focal_length_m": "m",
    "objective.effective_pupil_radius_m": "m",
    "objective.pupil_fill_fraction": "1",
    "relay.magnification": "1",
    "axicon.base_angle_deg": "deg",
    "axicon.clear_aperture_m": "m",
    "axicon.refractive_index": "1",
    "camera.pixel_pitch_m": "m",
    "camera.magnification": "1",
    "camera.object_plane_scale_m_per_pixel": "m/pixel",
    "camera.rotation_deg": "deg",
    "polarization.input_linear_angle_deg": "deg",
    "polarization.input_degree_linear_polarization": "1",
    "polarization.input_relative_phase_rad": "rad",
    "polarization.slm_director_axis_deg": "deg",
    "polarization.analyzer_0_actual_deg": "deg",
    "polarization.analyzer_45_actual_deg": "deg",
    "polarization.analyzer_90_actual_deg": "deg",
    "polarization.analyzer_135_actual_deg": "deg",
    "polarization.analyzer_extinction_ratio": "1",
    "polarization.analyzer_transmission": "1",
    "polarization.full_stokes_qwp_retardance_rad": "rad",
    "polarization.full_stokes_qwp_fast_axis_deg": "deg",
    "polarization.segmented_vector_hwp_retardance_rad": "rad",
    "polarization.segmented_vector_hwp_fast_axis_deg": "deg",
    "polarization.segmented_vector_qwp_retardance_rad": "rad",
    "polarization.segmented_vector_qwp_fast_axis_deg": "deg",
    "material.refractive_index": "1",
    **{f"transmissions.{name}": "1" for name in ("slm1", "slm2", "four_f", "objective", "interface")},
}

_REQUIRED_SECTIONS = {
    "laser", "slm", "fourier_filter", "objective", "relay", "axicon", "camera",
    "transmissions", "material", "experimental_comparison",
}
_CALIBRATED_SOURCES = {"measured", "manufacturer", "derived", "synthetic_measurement"}


def _raw_at(data: Mapping[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _dependency_satisfied(bundle: CalibrationBundle, path: str) -> tuple[bool, str]:
    value = value_at(bundle, path)
    if path.endswith("coating_state"):
        return value not in (None, "", "unknown"), source_at(bundle, path)
    if path.endswith("_verified") or path in {
        "experimental_comparison.performed",
        "experimental_comparison.acceptance_passed",
    }:
        return value is True, source_at(bundle, path)
    if value in (None, ""):
        return False, source_at(bundle, path)
    source = source_at(bundle, path)
    if source in {"assumed", "placeholder", "calibration_required", "missing"}:
        return False, source
    return source in _CALIBRATED_SOURCES or not isinstance(_raw_at(bundle.data, path), Mapping), source


def calibration_readiness_for_claim(bundle: CalibrationBundle, claim_type: str) -> CalibrationReadiness:
    if claim_type not in CLAIM_DEPENDENCIES and claim_type not in EXTRA_DEPENDENCIES:
        raise ValueError(f"unknown calibration claim type: {claim_type!r}")
    dependencies = CLAIM_DEPENDENCIES.get(claim_type, EXTRA_DEPENDENCIES.get(claim_type, ()))
    missing: list[str] = []
    non_calibrated: list[str] = []
    satisfied: list[str] = []
    for dependency in dependencies:
        ok, source = _dependency_satisfied(bundle, dependency.path)
        if ok:
            satisfied.append(dependency.path)
        elif value_at(bundle, dependency.path) in (None, "", "unknown", False):
            missing.append(dependency.path)
        else:
            non_calibrated.append(f"{dependency.path} [{source}]")
    ready = not missing and not non_calibrated
    if not dependencies:
        status = "ready_without_calibration"
    elif ready and bundle.is_synthetic:
        status = "synthetic_complete_not_experimental"
    elif ready:
        status = "calibrated_inputs_available"
    else:
        status = "blocked_missing_or_uncalibrated_measurements"
    return CalibrationReadiness(
        claim_type=claim_type,
        ready=ready,
        status=status,
        missing_measurements=tuple(missing),
        non_calibrated_measurements=tuple(non_calibrated),
        satisfied_measurements=tuple(satisfied),
    )


def validate_calibration_bundle(bundle: CalibrationBundle) -> CalibrationValidationReport:
    data = bundle.data
    errors: list[str] = []
    warnings: list[str] = []
    missing: list[str] = []
    unit_errors: list[str] = []
    impossible: list[str] = []

    if bundle.schema_version not in SUPPORTED_CALIBRATION_SCHEMA_VERSIONS:
        errors.append(
            f"unsupported schema_version {bundle.schema_version!r}; supported={sorted(SUPPORTED_CALIBRATION_SCHEMA_VERSIONS)!r}"
        )
    absent_sections = sorted(_REQUIRED_SECTIONS - set(data))
    errors.extend(f"missing required section: {section}" for section in absent_sections)
    if bundle.data_classification not in DATA_CLASSIFICATIONS:
        errors.append(f"invalid data_classification: {bundle.data_classification!r}")
    if not bundle.calibration_id:
        errors.append("calibration_id is required")
    created = data.get("created_utc")
    if created not in (None, ""):
        try:
            datetime.fromisoformat(str(created).replace("Z", "+00:00"))
        except ValueError:
            errors.append("created_utc must be ISO-8601 or null")

    for path, expected_unit in EXPECTED_UNITS.items():
        raw = _raw_at(data, path)
        if raw is None:
            missing.append(path)
            continue
        if isinstance(raw, Mapping):
            value = raw.get("value")
            unit = raw.get("unit")
            uncertainty = raw.get("uncertainty")
            if unit != expected_unit:
                message = f"{path} uses unit {unit!r}; expected {expected_unit!r}"
                unit_errors.append(message)
                errors.append(message)
            if uncertainty is not None and (not isinstance(uncertainty, (int, float)) or uncertainty < 0):
                message = f"{path} uncertainty must be a non-negative number or null"
                impossible.append(message)
                errors.append(message)
        else:
            value = raw
        if value is None:
            missing.append(path)
            continue
        if not isinstance(value, (int, float)):
            message = f"{path} value must be numeric or null"
            impossible.append(message)
            errors.append(message)
            continue
        numeric = float(value)
        if path.startswith("transmissions.") or path in {
            "objective.numerical_aperture",
            "objective.pupil_fill_fraction",
            "polarization.analyzer_transmission",
        }:
            valid = 0.0 < numeric <= 1.0
        elif path == "polarization.input_degree_linear_polarization":
            valid = 0.0 <= numeric <= 1.0
        elif path == "polarization.analyzer_extinction_ratio":
            valid = numeric >= 1.0
        elif path.endswith("refractive_index"):
            valid = numeric >= 1.0
        elif path == "axicon.base_angle_deg":
            valid = 0.0 < numeric < 90.0
        elif path.endswith("_deg") or path.endswith("_rad"):
            valid = np.isfinite(numeric)
        else:
            valid = numeric > 0.0
        if not valid:
            message = f"physically impossible value for {path}: {numeric}"
            impossible.append(message)
            errors.append(message)

    resolution = _raw_at(data, "slm.active_resolution")
    if not (
        isinstance(resolution, list)
        and len(resolution) == 2
        and all(isinstance(value, int) and value > 0 for value in resolution)
    ):
        message = "slm.active_resolution must contain two positive integer pixel counts"
        impossible.append(message)
        errors.append(message)
    pixel_pitch = _raw_at(data, "slm.pixel_pitch_m")
    if not isinstance(pixel_pitch, (int, float)) or pixel_pitch <= 0:
        message = "slm.pixel_pitch_m must be positive SI metres"
        impossible.append(message)
        errors.append(message)
    period = _raw_at(data, "slm.carrier_period_px")
    if period is not None and not np.isclose(float(period), 20.0, rtol=0.0, atol=1e-12):
        message = f"physical bench carrier period must be 20 px, got {period!r}"
        impossible.append(message)
        errors.append(message)

    if data.get("data_classification") == "laboratory_measurement":
        if not data.get("operator"):
            warnings.append("laboratory bundle has no operator identity")
        if created in (None, ""):
            warnings.append("laboratory bundle has no creation timestamp")
    if bundle.is_synthetic:
        warnings.append("synthetic_not_experimental: populated values cannot establish experimental maturity")

    return CalibrationValidationReport(
        valid_schema=not errors,
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings)),
        missing_values=tuple(dict.fromkeys(missing)),
        inconsistent_units=tuple(dict.fromkeys(unit_errors)),
        physically_impossible_values=tuple(dict.fromkeys(impossible)),
    )


def calibration_dependency_rows(bundle: CalibrationBundle) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for claim, dependencies in {**CLAIM_DEPENDENCIES, **EXTRA_DEPENDENCIES}.items():
        if not dependencies:
            rows.append({
                "claim_type": claim,
                "required_measurement": "none",
                "required_or_optional": "not_applicable",
                "current_status": "ready_without_calibration",
                "current_source": "Phase 2C solver evidence",
                "blocks_claim": False,
                "notes": "Relative morphology only.",
            })
            continue
        for dependency in dependencies:
            ok, source = _dependency_satisfied(bundle, dependency.path)
            rows.append({
                "claim_type": claim,
                "required_measurement": dependency.path,
                "required_or_optional": dependency.required_or_optional,
                "current_status": "available" if ok else "missing_or_uncalibrated",
                "current_source": source,
                "blocks_claim": dependency.required_or_optional == "required" and not ok,
                "notes": dependency.notes,
            })
    return rows


assert set(CLAIM_TYPES) == set(CLAIM_DEPENDENCIES)


import numpy as np
