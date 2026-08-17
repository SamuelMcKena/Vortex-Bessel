"""Active lab-realism perturbations for the Stage 8C.3 cockpit.

The functions in this module operate on the Stage 8C canonical optical-field
containers. They do not modify the locked propagation engine. Active controls
apply deterministic, documented perturbations to an already-generated optical
field stack so the cockpit can show how lab misalignment and imperfections
would affect downstream fluence diagnostics. Because this module is a
post-engine diagnostic layer, stack-transform controls are classified as
diagnostic-active unless a future engine path consumes them before propagation.

Direct Poynting-vector editing is not exposed here. In the scalar cockpit,
beam-direction changes are represented by an input tilt / angular-spectrum
phase ramp,

    E(x, y) -> E(x, y) exp(i * (kx0*x + ky0*y)).

For intensity-only stacks this is visualised as the corresponding z-dependent
beam walk-off and recorded in metadata. True vectorial energy-flow effects
remain metadata-only unless a vector engine is active.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

from vbb_study.digital_twin.field_coupling import OpticalFieldStack, stack_from_arrays

CONTROL_CLASSIFICATIONS = frozenset(
    {
        "physics_active",
        "energy_active",
        "geometry_active",
        "diagnostic_active",
        "warning_only",
        "metadata_only",
        "future_not_implemented",
    }
)

PHYSICAL_PLACEMENTS = frozenset(
    {
        "input complex field",
        "SLM phase-mask generation",
        "SLM amplitude/active-area stage",
        "Fourier plane",
        "relay plane",
        "objective pupil plane",
        "pre-propagation field",
        "post-propagation diagnostic only",
        "geometry only",
        "energy ledger/bookkeeping",
        "metadata/report only",
        "future disabled",
    }
)

IMPLEMENTATION_STAGES = frozenset(
    {
        "post-propagation diagnostic only",
        "geometry only",
        "energy ledger/bookkeeping",
        "metadata/report only",
        "warning only",
        "future disabled",
    }
)

AFFECTED_OUTPUTS = frozenset(
    {
        "field",
        "phase",
        "amplitude",
        "angular_spectrum",
        "energy_ledger",
        "fourier_filter",
        "pupil",
        "sample_geometry",
        "fluence",
        "exposure_bookkeeping",
        "warnings",
        "metadata",
        "future_stage",
    }
)

_FUTURE_STAGE_CONTROLS = frozenset(
    {
        "enable_material_response",
        "enable_threshold_proxy",
        "enable_dose_accumulation",
        "enable_nonlinear_proxy",
        "enable_thermal_proxy",
        "enable_microscope_proxy",
        "enable_calibrated_prediction",
    }
)


@dataclass(frozen=True)
class ControlClassification:
    """Classification row for one editable lab-realism control."""

    control: str
    value: Any
    enabled: bool
    classification: str
    affects: tuple[str, ...]
    implemented: bool
    downstream_response_expected: str
    physical_placement: str = "metadata/report only"
    implementation_stage: str = "metadata/report only"
    placement_note: str = ""

    def as_row(self) -> dict[str, Any]:
        return {
            "control": self.control,
            "value": self.value,
            "enabled": self.enabled,
            "classification": self.classification,
            "affects": ", ".join(self.affects),
            "implemented": self.implemented,
            "downstream_response_expected": self.downstream_response_expected,
            "physical_placement": self.physical_placement,
            "implementation_stage": self.implementation_stage,
            "placement_note": self.placement_note,
        }


@dataclass(frozen=True)
class PerturbationResult:
    """Result of applying active Stage 8C.3 lab perturbations."""

    baseline_stack: OpticalFieldStack
    perturbed_stack: OpticalFieldStack
    control_report: tuple[ControlClassification, ...]
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    final_export_allowed: bool = False
    figure_status: str = "diagnostic_allowed"
    model_status: str = "diagnostic_preview"

    @property
    def active_controls(self) -> list[ControlClassification]:
        return [
            row for row in self.control_report
            if row.enabled and row.implemented and _is_dashboard_relevant_control(row.control)
        ]

    @property
    def uncoupled_enabled_controls(self) -> list[ControlClassification]:
        return [
            row for row in self.control_report
            if row.enabled and not row.implemented and _is_dashboard_relevant_control(row.control)
        ]


def stage8c3_default_controls() -> dict[str, Any]:
    """Return Stage 8C.3 lab-realism controls and conservative defaults."""
    return {
        "enable_beam_decentre": False,
        "beam_decentre_x_um": 0.0,
        "beam_decentre_y_um": 0.0,
        "enable_beam_tilt": False,
        "beam_tilt_x_mrad": 0.0,
        "beam_tilt_y_mrad": 0.0,
        "enable_beam_ellipticity": False,
        "beam_radius_x_um": 24.0,
        "beam_radius_y_um": 24.0,
        "beam_rotation_deg": 0.0,
        "enable_input_aperture": False,
        "input_aperture_radius_um": 24.0,
        "input_aperture_decentre_x_um": 0.0,
        "input_aperture_decentre_y_um": 0.0,
        "enable_slm_phase_centre_offset": False,
        "slm_phase_centre_offset_x_um": 0.0,
        "slm_phase_centre_offset_y_um": 0.0,
        "enable_vortex_centre_offset": False,
        "vortex_centre_offset_x_um": 0.0,
        "vortex_centre_offset_y_um": 0.0,
        "enable_axicon_centre_offset": False,
        "axicon_centre_offset_x_um": 0.0,
        "axicon_centre_offset_y_um": 0.0,
        "enable_physical_axicon_misalignment": False,
        "physical_axicon_apex_offset_x_um": 0.0,
        "physical_axicon_apex_offset_y_um": 0.0,
        "physical_axicon_tilt_x_mrad": 0.0,
        "physical_axicon_tilt_y_mrad": 0.0,
        "physical_axicon_angle_error_deg": 0.0,
        "enable_axicon_apex_defect": False,
        "axicon_apex_defect_radius_um": 0.0,
        "enable_slm_phase_quantisation": False,
        "slm_phase_levels": 256,
        "enable_slm_pixelation": False,
        "enable_slm_fill_factor": False,
        "slm_fill_factor": 1.0,
        "enable_slm_dead_pixels": False,
        "dead_pixel_fraction": 0.0,
        "dead_pixel_seed": 7,
        "enable_slm_phase_noise": False,
        "slm_phase_noise_rms_rad": 0.0,
        "slm_phase_noise_seed": 11,
        "enable_slm_active_area": False,
        "slm_active_width_um": 50.0,
        "slm_active_height_um": 50.0,
        "slm_active_area_decentre_x_um": 0.0,
        "slm_active_area_decentre_y_um": 0.0,
        "enable_slm_rotation": False,
        "slm_rotation_deg": 0.0,
        "enable_mask_rotation": False,
        "phase_mask_rotation_deg": 0.0,
        "enable_zero_order_leakage": False,
        "zero_order_mode": "central_gaussian",
        "enable_unwanted_order_leakage": False,
        "unwanted_order_fraction": 0.0,
        "unwanted_order_kx_shift": 0.0,
        "unwanted_order_ky_shift": 0.0,
        "enable_first_order_filter": False,
        "first_order_filter_radius_px": None,
        "enable_first_order_filter_decentre": False,
        "first_order_filter_decentre_x_px": 0.0,
        "first_order_filter_decentre_y_px": 0.0,
        "enable_first_order_filter_clipping": False,
        "first_order_filter_clipping_fraction_override": None,
        "enable_relay_magnification_error": False,
        "relay_magnification_error_fraction": 0.0,
        "enable_relay_decentre": False,
        "relay_decentre_x_um": 0.0,
        "relay_decentre_y_um": 0.0,
        "enable_relay_tilt": False,
        "relay_tilt_x_mrad": 0.0,
        "relay_tilt_y_mrad": 0.0,
        "enable_relay_aperture": False,
        "relay_aperture_radius_um": 30.0,
        "relay_aperture_decentre_x_um": 0.0,
        "relay_aperture_decentre_y_um": 0.0,
        "enable_pupil_clipping": False,
        "pupil_radius_um": 20.0,
        "pupil_decentre_x_um": 0.0,
        "pupil_decentre_y_um": 0.0,
        "pupil_fill_target_fraction": 0.8,
        "enable_zernike_aberrations": False,
        "zernike_defocus_waves": 0.0,
        "zernike_astig_0_waves": 0.0,
        "zernike_astig_45_waves": 0.0,
        "zernike_coma_x_waves": 0.0,
        "zernike_coma_y_waves": 0.0,
        "zernike_spherical_waves": 0.0,
        "zernike_trefoil_x_waves": 0.0,
        "zernike_trefoil_y_waves": 0.0,
        "enable_defocus": False,
        "focus_offset_um": 0.0,
        "enable_focus_depth_error": False,
        "focus_depth_error_um": 0.0,
        "enable_sample_tilt": False,
        "sample_tilt_x_mrad": 0.0,
        "sample_tilt_y_mrad": 0.0,
        "enable_surface_offset": False,
        "sample_surface_z_um": 0.0,
        "enable_refractive_index_error": False,
        "refractive_index_error": 0.0,
        "enable_sample_thickness_limit": False,
        "enable_interface_reflection": False,
        "enable_pulse_energy_jitter": False,
        "pulse_energy_jitter_rms_fraction": 0.0,
        "pulse_energy_jitter_seed": 23,
        "enable_repetition_rate_error": False,
        "repetition_rate_error_fraction": 0.0,
        "enable_pulse_duration_error": False,
        "pulse_duration_error_fraction": 0.0,
        "enable_average_power_limit": True,
        "enable_pointing_jitter": False,
        "pointing_jitter_rms_urad": 0.0,
        "pointing_jitter_seed": 31,
        "enable_stage_position_jitter": False,
        "stage_jitter_x_um": 0.0,
        "stage_jitter_y_um": 0.0,
        "stage_jitter_z_um": 0.0,
        "stage_jitter_seed": 37,
        "enable_focus_drift": False,
        "focus_drift_um_per_min": 0.0,
        "enable_camera_crop": False,
        "camera_crop_width_um": 20.0,
        "camera_crop_height_um": 20.0,
        "camera_crop_centre_x_um": 0.0,
        "camera_crop_centre_y_um": 0.0,
        "enable_detector_noise": False,
        "detector_noise_fraction": 0.0,
        "detector_noise_seed": 41,
        "enable_display_autoscale": True,
        "display_autoscale": True,
    }


_STAGE8C3_CONTROL_NAMES = frozenset(stage8c3_default_controls().keys())


_META: dict[str, tuple[str, tuple[str, ...], bool, str]] = {}
_PHYSICAL_PLACEMENT_PREFIXES: tuple[tuple[str, str, str], ...] = (
    ("beam_decentre_", "input complex field", "beam decentre belongs to input amplitude/field before phase-mask interaction"),
    ("beam_tilt_", "input complex field", "beam tilt belongs to input complex-field phase ramp before propagation"),
    ("beam_radius_", "input complex field", "beam ellipticity belongs to input amplitude before phase-mask interaction"),
    ("beam_rotation", "input complex field", "beam ellipticity belongs to input amplitude before phase-mask interaction"),
    ("input_aperture_", "input complex field", "input aperture belongs before phase-mask interaction"),
    ("slm_phase_centre_offset_", "SLM phase-mask generation", "common phase-mask centre offset belongs in phase-mask generation"),
    ("vortex_centre_offset_", "SLM phase-mask generation", "vortex centre offset belongs in vortex phase-mask generation only"),
    ("axicon_centre_offset_", "SLM phase-mask generation", "axicon centre offset belongs in axicon phase-mask generation only"),
    ("physical_axicon_", "pre-propagation field", "physical axicon misalignment belongs before focus/propagation"),
    ("axicon_apex_defect_", "pre-propagation field", "apex defects require route-level optical modelling"),
    ("slm_phase_levels", "SLM amplitude/active-area stage", "SLM phase quantisation belongs at the SLM phase stage before propagation"),
    ("slm_pixel_pitch", "SLM amplitude/active-area stage", "SLM pixelation belongs at the SLM amplitude/phase stage"),
    ("slm_fill_factor", "SLM amplitude/active-area stage", "SLM fill factor belongs at the SLM amplitude stage"),
    ("dead_pixel_", "SLM amplitude/active-area stage", "dead pixels belong at the SLM amplitude stage"),
    ("slm_phase_noise_", "SLM amplitude/active-area stage", "SLM phase noise belongs at the SLM phase stage before propagation"),
    ("slm_active_", "SLM amplitude/active-area stage", "finite SLM active area belongs at the SLM amplitude/aperture stage"),
    ("slm_rotation", "SLM phase-mask generation", "SLM rotation requires phase-mask resampling"),
    ("phase_mask_rotation", "SLM phase-mask generation", "mask rotation requires phase-mask resampling"),
    ("zero_order_", "Fourier plane", "zero-order leakage belongs to order isolation / Fourier-plane leakage"),
    ("unwanted_order_", "Fourier plane", "unwanted diffraction orders belong to Fourier/order-isolation plane"),
    ("first_order_filter_", "Fourier plane", "first-order filter decentre/clipping belongs to the Fourier plane"),
    ("relay_magnification_error_", "relay plane", "relay magnification error belongs to relay imaging"),
    ("relay_decentre_", "relay plane", "relay decentre belongs to relay imaging"),
    ("relay_tilt_", "relay plane", "relay tilt belongs to relay imaging / steering"),
    ("relay_aperture_", "relay plane", "relay aperture belongs to relay-plane clipping"),
    ("pupil_", "objective pupil plane", "pupil decentre/clipping belongs at the objective pupil plane"),
    ("zernike_", "objective pupil plane", "Zernike aberrations belong at the objective pupil plane"),
    ("focus_offset", "pre-propagation field", "defocus belongs before propagation or in focus geometry"),
    ("focus_depth_error_", "pre-propagation field", "focus-depth error belongs before propagation or in focus geometry"),
    ("sample_tilt_", "geometry only", "sample tilt is geometry-only unless an interface/propagation model consumes it"),
    ("sample_surface_z", "geometry only", "surface offset is geometry-only in Stage 8C.3"),
    ("refractive_index_error", "geometry only", "index error is geometry/warning bookkeeping in Stage 8C.3"),
    ("pulse_energy_jitter_", "energy ledger/bookkeeping", "pulse-energy jitter belongs to the energy ledger before fluence scaling"),
    ("repetition_rate_error_", "energy ledger/bookkeeping", "repetition-rate error belongs to exposure bookkeeping"),
    ("pulse_duration_error_", "energy ledger/bookkeeping", "pulse-duration error belongs to peak-intensity bookkeeping"),
    ("pointing_jitter_", "metadata/report only", "pointing jitter needs an ensemble model"),
    ("stage_jitter_", "metadata/report only", "stage jitter needs an ensemble model"),
    ("focus_drift_", "metadata/report only", "focus drift needs an ensemble model"),
    ("camera_crop_", "post-propagation diagnostic only", "camera crop is a display/diagnostic operation"),
    ("detector_noise_", "post-propagation diagnostic only", "detector noise is a display/diagnostic operation"),
    ("display_autoscale", "post-propagation diagnostic only", "autoscale is display-only"),
)

_PHYSICAL_PLACEMENT_EXACT: dict[str, tuple[str, str]] = {
    "enable_beam_decentre": ("input complex field", "beam decentre belongs to input amplitude/field before phase-mask interaction"),
    "enable_beam_tilt": ("input complex field", "beam tilt belongs to input complex-field phase ramp before propagation"),
    "enable_beam_ellipticity": ("input complex field", "beam ellipticity belongs to input amplitude before phase-mask interaction"),
    "enable_input_aperture": ("input complex field", "input aperture belongs before phase-mask interaction"),
    "enable_slm_phase_centre_offset": ("SLM phase-mask generation", "common phase-mask centre offset belongs in phase-mask generation"),
    "enable_vortex_centre_offset": ("SLM phase-mask generation", "vortex centre offset belongs in vortex phase-mask generation only"),
    "enable_axicon_centre_offset": ("SLM phase-mask generation", "axicon centre offset belongs in axicon phase-mask generation only"),
    "enable_physical_axicon_misalignment": ("pre-propagation field", "physical axicon apex/tilt belongs before focus/propagation"),
    "enable_axicon_apex_defect": ("pre-propagation field", "apex defects require route-level optical modelling"),
    "enable_slm_phase_quantisation": ("SLM amplitude/active-area stage", "SLM phase quantisation belongs at the SLM phase stage"),
    "enable_slm_pixelation": ("SLM amplitude/active-area stage", "SLM pixelation belongs at the SLM phase/amplitude stage"),
    "enable_slm_fill_factor": ("SLM amplitude/active-area stage", "SLM fill factor belongs at the SLM amplitude stage"),
    "enable_slm_dead_pixels": ("SLM amplitude/active-area stage", "dead pixels belong at the SLM amplitude stage"),
    "enable_slm_phase_noise": ("SLM amplitude/active-area stage", "SLM phase noise belongs at the SLM phase stage"),
    "enable_slm_active_area": ("SLM amplitude/active-area stage", "finite SLM active area belongs at the SLM aperture stage"),
    "enable_slm_rotation": ("SLM phase-mask generation", "SLM rotation requires phase-mask resampling"),
    "enable_mask_rotation": ("SLM phase-mask generation", "mask rotation requires phase-mask resampling"),
    "enable_zero_order_leakage": ("Fourier plane", "zero-order leakage belongs to order isolation / Fourier-plane leakage"),
    "enable_unwanted_order_leakage": ("Fourier plane", "unwanted diffraction orders belong to Fourier/order-isolation plane"),
    "enable_first_order_filter": ("Fourier plane", "first-order filter belongs to Fourier/order-isolation plane"),
    "enable_first_order_filter_decentre": ("Fourier plane", "4F filter decentre belongs to the Fourier plane; warning-only until engine-coupled"),
    "enable_first_order_filter_clipping": ("Fourier plane", "4F filter clipping belongs to the Fourier plane; warning-only until engine-coupled"),
    "enable_relay_magnification_error": ("relay plane", "relay magnification error belongs to relay imaging"),
    "enable_relay_decentre": ("relay plane", "relay decentre belongs to relay imaging"),
    "enable_relay_tilt": ("relay plane", "relay tilt belongs to relay imaging / steering"),
    "enable_relay_aperture": ("relay plane", "relay aperture belongs to relay-plane clipping"),
    "enable_pupil_clipping": ("objective pupil plane", "pupil decentre/clipping belongs at the objective pupil plane"),
    "enable_zernike_aberrations": ("objective pupil plane", "Zernike aberrations belong at the objective pupil plane"),
    "enable_defocus": ("pre-propagation field", "defocus belongs before propagation or in focus geometry"),
    "enable_focus_depth_error": ("pre-propagation field", "focus-depth error belongs before propagation or in focus geometry"),
    "enable_sample_tilt": ("geometry only", "sample tilt is geometry-only unless an interface/propagation model consumes it"),
    "enable_surface_offset": ("geometry only", "surface offset is geometry-only in Stage 8C.3"),
    "enable_refractive_index_error": ("geometry only", "index error is geometry/warning bookkeeping in Stage 8C.3"),
    "enable_sample_thickness_limit": ("geometry only", "sample-thickness limit is geometry/warning bookkeeping"),
    "enable_interface_reflection": ("energy ledger/bookkeeping", "interface reflection belongs to energy bookkeeping in Stage 8C.3"),
    "enable_pulse_energy_jitter": ("energy ledger/bookkeeping", "pulse-energy jitter belongs to the energy ledger before fluence scaling"),
    "enable_repetition_rate_error": ("energy ledger/bookkeeping", "repetition-rate error belongs to exposure bookkeeping"),
    "enable_pulse_duration_error": ("energy ledger/bookkeeping", "pulse-duration error belongs to peak-intensity bookkeeping"),
    "enable_average_power_limit": ("energy ledger/bookkeeping", "average-power limit belongs to energy/warning bookkeeping"),
    "enable_pointing_jitter": ("metadata/report only", "pointing jitter needs an ensemble model"),
    "enable_stage_position_jitter": ("metadata/report only", "stage jitter needs an ensemble model"),
    "enable_focus_drift": ("metadata/report only", "focus drift needs an ensemble model"),
    "enable_camera_crop": ("post-propagation diagnostic only", "camera crop is a display/diagnostic operation"),
    "enable_detector_noise": ("post-propagation diagnostic only", "detector noise is a display/diagnostic operation"),
    "enable_display_autoscale": ("post-propagation diagnostic only", "autoscale is display-only"),
}


def _add(
    names: Iterable[str],
    classification: str,
    affects: Iterable[str],
    implemented: bool,
    response: str,
) -> None:
    affects_tuple = tuple(str(a) for a in affects)
    if classification not in CONTROL_CLASSIFICATIONS:
        raise ValueError(classification)
    if any(a not in AFFECTED_OUTPUTS for a in affects_tuple):
        raise ValueError(str(affects_tuple))
    for name in names:
        _META[str(name)] = (classification, affects_tuple, bool(implemented), response)


# Existing cockpit controls.
_add(["planning_mode", "save_outputs", "figure_dpi", "show_caveats", "show_warnings", "show_diagnostic_panels"], "metadata_only", ["metadata"], True, "changes notebook/report behaviour only")
_add(["engine_preset", "engine_path", "require_real_field", "allow_synthetic_demo_field"], "diagnostic_active", ["field", "warnings", "metadata"], True, "selects or gates real field acquisition")
_add(["wavelength_nm", "beam_radius_mm", "polarisation_state"], "metadata_only", ["metadata"], True, "recorded as source metadata in scalar cockpit")
_add(["pulse_duration_fs"], "energy_active", ["fluence", "metadata"], True, "changes peak-intensity estimate")
_add(["repetition_rate_Hz", "average_power_limit_W"], "energy_active", ["energy_ledger", "exposure_bookkeeping", "warnings"], True, "changes average power and/or exposure bookkeeping")
_add(["pulse_energy_before_optics_uJ"], "energy_active", ["energy_ledger", "fluence"], True, "changes energy at sample and fluence scaling")
_add(["pre_slm_transmission", "telescope_transmission", "slm1_diffraction_efficiency", "slm2_diffraction_efficiency", "selected_first_order_fraction", "relay_transmission", "objective_transmission", "sample_interface_transmission"], "energy_active", ["energy_ledger", "fluence"], True, "changes pulse energy delivered downstream")
_add(["input_beam_radius_mm", "pointing_offset_x_um", "pointing_offset_y_um", "aperture_radius_mm", "telescope_enabled", "telescope_magnification"], "geometry_active", ["metadata", "warnings"], True, "reported in lab realism geometry; field coupling uses Stage 8C.3 controls")
_add(["beam_ellipticity"], "physics_active", ["field", "amplitude", "fluence"], True, "changes amplitude envelope when enable_beam_ellipticity is active")
_add(["slm1_enabled", "phase_profile", "ell", "generation_method", "axicon_k_r", "target_core_diameter_um", "target_bessel_length_um", "blaze_period_px", "slm2_conjugate_mode", "physical_axicon_enabled", "physical_axicon_angle_deg"], "metadata_only", ["metadata", "warnings"], True, "records route intent; locked engine generates the base field")
_add(["phase_quantisation_levels", "slm_active_width_px", "slm_active_height_px"], "metadata_only", ["metadata", "warnings"], True, "legacy SLM metadata; active SLM controls use Stage 8C.3 names")
_add(["first_order_filter_enabled", "filter_radius_px_or_lpmm"], "energy_active", ["energy_ledger", "warnings", "fluence"], True, "energy/filter report; active leakage uses Stage 8C.3 enable controls")
_add(["relay_magnification", "relay_aberration_enabled", "objective_NA", "objective_effective_focal_length_mm", "objective_pupil_diameter_mm", "pupil_diameter_override_mm"], "geometry_active", ["pupil", "metadata", "warnings"], True, "reports lab geometry and pupil feasibility")
_add(["material_name", "refractive_index", "sample_thickness_mm", "focus_depth_um", "use_fresnel_interface_estimate", "surface_tilt_mrad"], "geometry_active", ["sample_geometry", "energy_ledger", "warnings"], True, "changes sample/interface bookkeeping or z markers")
_add(["grid_N", "device_downsample", "axial_planes", "crop_window_um"], "diagnostic_active", ["field", "metadata", "warnings"], True, "field acquisition and diagnostic metadata")
_add(["fluence_normalisation_mode", "selected_z_mode", "selected_z_um", "custom_z_um", "central_roi_half_width_um"], "diagnostic_active", ["fluence", "metadata"], True, "changes fluence diagnostics/selected display plane")
_add(["display_scaling", "display_percentile_clip"], "diagnostic_active", ["metadata"], True, "display only; raw metrics unchanged")
_add(["writing_mode", "scan_axis", "scan_speed_mm_s", "line_length_um", "effective_diameter_um", "num_static_pulses", "num_passes", "z_step_um", "tilt_angle_deg"], "geometry_active", ["exposure_bookkeeping"], True, "changes exposure bookkeeping")
_add(["enable_material_response", "enable_threshold_proxy", "enable_dose_accumulation", "enable_nonlinear_proxy", "enable_thermal_proxy", "enable_microscope_proxy", "enable_calibrated_prediction"], "future_not_implemented", ["future_stage", "warnings"], False, "raises/disabled in Stage 8C.3")

# Stage 8C.3 active perturbations.
_add(["enable_beam_decentre", "beam_decentre_x_um", "beam_decentre_y_um"], "physics_active", ["field", "amplitude", "fluence"], True, "shifts transverse field centroid")
_add(["enable_beam_tilt", "beam_tilt_x_mrad", "beam_tilt_y_mrad"], "physics_active", ["phase", "angular_spectrum", "field", "fluence"], True, "phase-ramp equivalent creates z-dependent walk-off")
_add(["enable_beam_ellipticity", "beam_radius_x_um", "beam_radius_y_um", "beam_rotation_deg"], "physics_active", ["field", "amplitude", "fluence"], True, "changes input amplitude envelope and symmetry")
_add(["enable_input_aperture", "input_aperture_radius_um", "input_aperture_decentre_x_um", "input_aperture_decentre_y_um"], "physics_active", ["field", "amplitude", "fluence", "warnings"], True, "clips input amplitude and records transmitted power")
_add(["enable_slm_phase_centre_offset", "slm_phase_centre_offset_x_um", "slm_phase_centre_offset_y_um", "enable_vortex_centre_offset", "vortex_centre_offset_x_um", "vortex_centre_offset_y_um", "enable_axicon_centre_offset", "axicon_centre_offset_x_um", "axicon_centre_offset_y_um"], "physics_active", ["phase", "field", "fluence"], True, "decentres phase-mask registration and degrades symmetry")
_add(["enable_physical_axicon_misalignment", "physical_axicon_apex_offset_x_um", "physical_axicon_apex_offset_y_um", "physical_axicon_tilt_x_mrad", "physical_axicon_tilt_y_mrad"], "physics_active", ["phase", "angular_spectrum", "field", "fluence"], True, "apex offset and tilt map to active axicon-centre/tilt perturbations")
_add(["physical_axicon_angle_error_deg"], "warning_only", ["warnings", "metadata"], False, "cone-angle retuning requires engine-level route support")
_add(["enable_axicon_apex_defect", "axicon_apex_defect_radius_um"], "warning_only", ["warnings"], False, "apex defect is flagged but not numerically modelled")
_add(["enable_slm_phase_quantisation", "slm_phase_levels"], "physics_active", ["phase", "field", "fluence"], True, "adds deterministic quantisation-like contrast modulation")
_add(["enable_slm_pixelation", "slm_pixel_pitch_um", "enable_slm_fill_factor", "slm_fill_factor"], "physics_active", ["amplitude", "field", "fluence"], True, "adds sampling/fill-factor amplitude modulation")
_add(["enable_slm_dead_pixels", "dead_pixel_fraction", "dead_pixel_seed"], "physics_active", ["amplitude", "field", "fluence"], True, "adds seeded dead-pixel amplitude defects")
_add(["enable_slm_phase_noise", "slm_phase_noise_rms_rad", "slm_phase_noise_seed"], "physics_active", ["phase", "field", "fluence"], True, "adds seeded phase-noise-like speckle modulation")
_add(["enable_slm_active_area", "slm_active_width_um", "slm_active_height_um", "slm_active_area_decentre_x_um", "slm_active_area_decentre_y_um"], "physics_active", ["amplitude", "field", "fluence"], True, "rectangular active-area clips the field")
_add(["enable_slm_rotation", "slm_rotation_deg", "enable_mask_rotation", "phase_mask_rotation_deg"], "diagnostic_active", ["metadata", "warnings"], False, "rotation is reported; resampling phase-mask rotation is not yet coupled")
_add(["enable_zero_order_leakage", "zero_order_leakage_fraction", "zero_order_mode"], "physics_active", ["field", "amplitude", "fluence"], True, "adds residual unmodulated component and fills the core")
_add(["enable_unwanted_order_leakage", "unwanted_order_fraction", "unwanted_order_kx_shift", "unwanted_order_ky_shift"], "physics_active", ["field", "angular_spectrum", "fluence"], True, "adds shifted ghost order")
_add(["enable_first_order_filter", "first_order_filter_radius_px", "enable_first_order_filter_decentre", "first_order_filter_decentre_x_px", "first_order_filter_decentre_y_px", "enable_first_order_filter_clipping", "first_order_filter_clipping_fraction_override"], "warning_only", ["fourier_filter", "energy_ledger", "warnings"], False, "field-active Fourier filtering is not implemented outside the engine")
_add(["enable_relay_magnification_error", "relay_magnification_error_fraction"], "physics_active", ["field", "amplitude", "fluence"], True, "rescales transverse field image")
_add(["enable_relay_decentre", "relay_decentre_x_um", "relay_decentre_y_um"], "physics_active", ["field", "amplitude", "fluence"], True, "shifts relayed field")
_add(["enable_relay_tilt", "relay_tilt_x_mrad", "relay_tilt_y_mrad"], "physics_active", ["phase", "angular_spectrum", "field", "fluence"], True, "adds relay steering as z-dependent walk-off")
_add(["enable_relay_aperture", "relay_aperture_radius_um", "relay_aperture_decentre_x_um", "relay_aperture_decentre_y_um"], "physics_active", ["field", "amplitude", "fluence"], True, "clips relayed field")
_add(["enable_pupil_clipping", "pupil_radius_um", "pupil_decentre_x_um", "pupil_decentre_y_um", "pupil_fill_target_fraction"], "physics_active", ["pupil", "field", "fluence", "warnings"], True, "applies circular pupil clipping and reports clipped power")
_add(["enable_zernike_aberrations", "zernike_defocus_waves", "zernike_astig_0_waves", "zernike_astig_45_waves", "zernike_coma_x_waves", "zernike_coma_y_waves", "zernike_spherical_waves", "zernike_trefoil_x_waves", "zernike_trefoil_y_waves"], "physics_active", ["phase", "pupil", "field", "fluence"], True, "applies low-order documented aberration distortions")
_add(["enable_defocus", "focus_offset_um", "enable_focus_depth_error", "focus_depth_error_um"], "geometry_active", ["sample_geometry", "field", "fluence"], True, "shifts axial response / selected focus relationship")
_add(["enable_sample_tilt", "sample_tilt_x_mrad", "sample_tilt_y_mrad", "enable_surface_offset", "sample_surface_z_um"], "geometry_active", ["sample_geometry", "metadata"], True, "moves geometry markers; field unchanged unless coupled elsewhere")
_add(["enable_refractive_index_error", "refractive_index_error", "enable_sample_thickness_limit"], "geometry_active", ["sample_geometry", "warnings"], True, "updates geometry/warning metrics only")
_add(["enable_interface_reflection"], "energy_active", ["energy_ledger", "fluence"], True, "uses existing Fresnel/interface energy controls")
_add(["enable_pulse_energy_jitter", "pulse_energy_jitter_rms_fraction", "pulse_energy_jitter_seed"], "energy_active", ["energy_ledger", "fluence"], True, "changes a deterministic jittered pulse-energy sample")
_add(["enable_repetition_rate_error", "repetition_rate_error_fraction"], "energy_active", ["energy_ledger", "exposure_bookkeeping"], True, "changes effective repetition rate")
_add(["enable_pulse_duration_error", "pulse_duration_error_fraction"], "energy_active", ["fluence"], True, "changes peak-intensity estimate")
_add(["enable_average_power_limit"], "energy_active", ["energy_ledger", "warnings"], True, "enables power-limit warnings")
_add(["enable_pointing_jitter", "pointing_jitter_rms_urad", "pointing_jitter_seed", "enable_stage_position_jitter", "stage_jitter_x_um", "stage_jitter_y_um", "stage_jitter_z_um", "stage_jitter_seed", "enable_focus_drift", "focus_drift_um_per_min"], "warning_only", ["warnings", "metadata"], False, "statistical ensemble not implemented in Stage 8C.3")
_add(["enable_camera_crop", "camera_crop_width_um", "camera_crop_height_um", "camera_crop_centre_x_um", "camera_crop_centre_y_um", "enable_detector_noise", "detector_noise_fraction", "detector_noise_seed", "enable_display_autoscale", "display_autoscale"], "diagnostic_active", ["metadata"], True, "display/diagnostic only; physical metrics unchanged")

for _name, (_classification, _affects, _implemented, _response) in list(_META.items()):
    if _classification == "physics_active":
        _META[_name] = (
            "diagnostic_active",
            _affects,
            _implemented,
            _response + "; Stage 8C.3 applies this as a post-engine diagnostic stack transform",
        )


def classification_for_control(control: str, controls: Mapping[str, Any]) -> ControlClassification:
    """Return a classification row for one control."""
    if control not in _META:
        raise KeyError(f"No Stage 8C.3 classification registered for control {control!r}.")
    classification, affects, implemented, response = _META[control]
    value = controls.get(control)
    enabled = _control_enabled(control, controls, classification)
    physical_placement, placement_note = physical_placement_for_control(control)
    implementation_stage = implementation_stage_for_control(control, classification, implemented)
    return ControlClassification(
        control=control,
        value=value,
        enabled=enabled,
        classification=classification,
        affects=affects,
        implemented=implemented,
        downstream_response_expected=response,
        physical_placement=physical_placement,
        implementation_stage=implementation_stage,
        placement_note=placement_note,
    )


def classify_lab_controls(
    controls: Mapping[str, Any],
    *,
    only: Iterable[str] | None = None,
) -> tuple[ControlClassification, ...]:
    """Classify every requested control, raising if any is unregistered."""
    keys = list(only) if only is not None else list(controls.keys())
    return tuple(classification_for_control(str(key), controls) for key in keys)


def classification_rows_for_controls(
    controls: Mapping[str, Any],
    *,
    only: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    return [row.as_row() for row in classify_lab_controls(controls, only=only)]


def physical_placement_for_control(control: str) -> tuple[str, str]:
    """Return the intended physical placement and audit note for one control."""
    if control in _FUTURE_STAGE_CONTROLS:
        return "future disabled", "future material/response physics is disabled at Stage 8C.3"
    if control in _PHYSICAL_PLACEMENT_EXACT:
        return _PHYSICAL_PLACEMENT_EXACT[control]
    for prefix, placement, note in _PHYSICAL_PLACEMENT_PREFIXES:
        if control.startswith(prefix):
            return placement, note
    if control in {
        "pulse_energy_before_optics_uJ",
        "pre_slm_transmission",
        "telescope_transmission",
        "slm1_diffraction_efficiency",
        "slm2_diffraction_efficiency",
        "selected_first_order_fraction",
        "relay_transmission",
        "objective_transmission",
        "sample_interface_transmission",
        "repetition_rate_Hz",
        "average_power_limit_W",
        "pulse_duration_fs",
    }:
        return "energy ledger/bookkeeping", "energy/exposure bookkeeping; not a material-response model"
    if control in {"focus_depth_um", "surface_tilt_mrad", "tilt_angle_deg"}:
        return "geometry only", "geometry/bookkeeping unless consumed by a propagation/interface model"
    return "metadata/report only", "recorded for traceability or notebook/report behaviour"


def implementation_stage_for_control(control: str, classification: str, implemented: bool) -> str:
    """Return the actual Stage 8C.3 implementation stage for one control."""
    if classification == "future_not_implemented":
        return "future disabled"
    if classification == "warning_only" or not implemented:
        return "warning only"
    if classification == "metadata_only":
        return "metadata/report only"
    if classification == "energy_active":
        return "energy ledger/bookkeeping"
    if classification == "geometry_active":
        return "geometry only"
    physical, _ = physical_placement_for_control(control)
    if physical in {"geometry only", "energy ledger/bookkeeping", "metadata/report only", "future disabled"}:
        return physical
    return "post-propagation diagnostic only"


def physical_placement_rows_for_controls(
    controls: Mapping[str, Any],
    *,
    only: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Return compact physical-placement audit rows for controls."""
    rows = []
    for row in classify_lab_controls(controls, only=only):
        rows.append(
            {
                "control": row.control,
                "enabled": row.enabled,
                "classification": row.classification,
                "physical_placement": row.physical_placement,
                "implementation_stage": row.implementation_stage,
                "implemented": row.implemented,
                "placement_note": row.placement_note,
            }
        )
    return rows


def enabled_uncoupled_controls(controls: Mapping[str, Any]) -> list[ControlClassification]:
    return [row for row in classify_lab_controls(controls) if row.enabled and not row.implemented]


def apply_lab_perturbations_to_stack(
    stack: OpticalFieldStack,
    controls: Mapping[str, Any],
) -> PerturbationResult:
    """Apply active Stage 8C.3 lab-realism perturbations to a field stack."""
    if not isinstance(stack, OpticalFieldStack):
        raise TypeError(f"stack must be OpticalFieldStack; got {type(stack).__name__}.")
    c = dict(stage8c3_default_controls())
    c.update(dict(controls))
    report = classify_lab_controls({key: c[key] for key in c if key in _META})
    warnings: list[str] = []
    metadata: dict[str, Any] = {
        "stage": "stage8c3_active_lab_realism",
        "active_controls": [row.control for row in report if row.enabled and row.implemented],
        "uncoupled_enabled_controls": [row.control for row in report if row.enabled and not row.implemented],
        "pupil_clipped_power_fraction": 0.0,
        "aperture_clipped_power_fraction": 0.0,
        "slm_active_area_clipped_power_fraction": 0.0,
        "passive_transmitted_power_fraction": 1.0,
        "post_engine_spatial_clipping_applied": False,
        "passive_clipping_visual_model": (
            "throughput audit only; no post-propagation spatial crop rendered"
        ),
        "phase_ramp_kx_rad_per_um": 0.0,
        "phase_ramp_ky_rad_per_um": 0.0,
        "first_order_selected_fraction": float(c.get("selected_first_order_fraction", 1.0) or 1.0),
    }

    I = np.asarray(stack.intensity_zyx, dtype=float).copy()
    x = np.asarray(stack.x_um, dtype=float)
    y = np.asarray(stack.y_um, dtype=float)
    z = np.asarray(stack.z_um, dtype=float)
    baseline_stack_integral = max(float(np.sum(I)), 1e-12)
    apply_post_engine_spatial_clipping = bool(c.get("enable_post_engine_spatial_clipping", False))

    def clip(mask: np.ndarray, key: str) -> None:
        nonlocal I
        before = float(np.sum(I))
        masked = I * mask[None, :, :]
        after = float(np.sum(masked))
        if before > 0:
            transmitted = float(np.clip(after / before, 0.0, 1.0))
            metadata[key] = max(float(metadata.get(key, 0.0)), 1.0 - transmitted)
            metadata["passive_transmitted_power_fraction"] = (
                float(metadata["passive_transmitted_power_fraction"]) * transmitted
            )
        if apply_post_engine_spatial_clipping:
            I = masked
            metadata["post_engine_spatial_clipping_applied"] = True
            metadata["passive_clipping_visual_model"] = (
                "post-propagation spatial crop explicitly enabled; diagnostic artifact risk"
            )

    if c["enable_beam_decentre"]:
        I = _shift_stack_xy(I, x, y, float(c["beam_decentre_x_um"]), float(c["beam_decentre_y_um"]))

    if c["enable_beam_tilt"]:
        tx = float(c["beam_tilt_x_mrad"])
        ty = float(c["beam_tilt_y_mrad"])
        I = _tilt_walkoff(I, x, y, z, tx, ty)
        metadata["phase_ramp_kx_rad_per_um"] += _mrad_to_phase_slope(tx, c)
        metadata["phase_ramp_ky_rad_per_um"] += _mrad_to_phase_slope(ty, c)
        metadata["beam_tilt_phase_ramp"] = "E(x,y) exp[i(kx0 x + ky0 y)]"

    if c["enable_beam_ellipticity"]:
        before = I.copy()
        I *= _elliptical_envelope(
            x, y,
            float(c["beam_radius_x_um"]),
            float(c["beam_radius_y_um"]),
            float(c["beam_rotation_deg"]),
        )[None, :, :]
        I = _match_plane_integrals(I, before)

    if c["enable_input_aperture"]:
        clip(
            _circle_mask(
                x, y,
                float(c["input_aperture_radius_um"]),
                float(c["input_aperture_decentre_x_um"]),
                float(c["input_aperture_decentre_y_um"]),
            ),
            "aperture_clipped_power_fraction",
        )

    slm_dx = 0.0
    slm_dy = 0.0
    vortex_dx = 0.0
    vortex_dy = 0.0
    axicon_dx = 0.0
    axicon_dy = 0.0
    if c["enable_slm_phase_centre_offset"]:
        slm_dx += float(c["slm_phase_centre_offset_x_um"])
        slm_dy += float(c["slm_phase_centre_offset_y_um"])
    if c["enable_vortex_centre_offset"]:
        vortex_dx += float(c["vortex_centre_offset_x_um"])
        vortex_dy += float(c["vortex_centre_offset_y_um"])
    if c["enable_axicon_centre_offset"]:
        axicon_dx += float(c["axicon_centre_offset_x_um"])
        axicon_dy += float(c["axicon_centre_offset_y_um"])
    if c["enable_physical_axicon_misalignment"]:
        axicon_dx += float(c["physical_axicon_apex_offset_x_um"])
        axicon_dy += float(c["physical_axicon_apex_offset_y_um"])
        I = _tilt_walkoff(
            I, x, y, z,
            float(c["physical_axicon_tilt_x_mrad"]),
            float(c["physical_axicon_tilt_y_mrad"]),
        )
        metadata["phase_ramp_kx_rad_per_um"] += _mrad_to_phase_slope(float(c["physical_axicon_tilt_x_mrad"]), c)
        metadata["phase_ramp_ky_rad_per_um"] += _mrad_to_phase_slope(float(c["physical_axicon_tilt_y_mrad"]), c)
    phase_common_dx = slm_dx + 0.5 * (vortex_dx + axicon_dx)
    phase_common_dy = slm_dy + 0.5 * (vortex_dy + axicon_dy)
    relative_dx = vortex_dx - axicon_dx
    relative_dy = vortex_dy - axicon_dy
    if phase_common_dx or phase_common_dy:
        # Common-mode vortex+axicon motion is mainly a beam registration/steering
        # diagnostic, not a headline shape-degradation case.
        I = _shift_stack_xy(I, x, y, 0.20 * phase_common_dx, 0.20 * phase_common_dy)
        metadata["phase_common_translation_um"] = (0.20 * phase_common_dx, 0.20 * phase_common_dy)
    if relative_dx or relative_dy:
        before = I
        I = _phase_centre_degrade(I, x, y, relative_dx, relative_dy)
        I = _match_plane_integrals(I, before)
        metadata["relative_phase_misregistration_um"] = (relative_dx, relative_dy)
    if slm_dx or slm_dy or vortex_dx or vortex_dy or axicon_dx or axicon_dy:
        metadata["phase_centre_offset_um"] = {
            "slm": (slm_dx, slm_dy),
            "vortex": (vortex_dx, vortex_dy),
            "axicon": (axicon_dx, axicon_dy),
            "common_translation": (0.20 * phase_common_dx, 0.20 * phase_common_dy),
            "relative_vortex_minus_axicon": (relative_dx, relative_dy),
        }

    if c["enable_slm_phase_quantisation"]:
        before = I.copy()
        levels = max(2, int(c["slm_phase_levels"]))
        I *= _quantisation_modulation(x, y, levels)[None, :, :]
        I = _match_plane_integrals(I, before)

    if c["enable_slm_pixelation"]:
        before = I.copy()
        I *= _pixelation_modulation(x, y, float(c.get("slm_pixel_pitch_um", 8.0) or 8.0))[None, :, :]
        I = _match_plane_integrals(I, before)

    if c["enable_slm_fill_factor"]:
        fill = float(c["slm_fill_factor"])
        trans = float(np.clip(fill, 0.0, 1.0))
        I *= trans
        metadata["passive_transmitted_power_fraction"] = (
            float(metadata["passive_transmitted_power_fraction"]) * trans
        )
        metadata["slm_fill_factor_transmitted_fraction"] = trans

    if c["enable_slm_dead_pixels"] and float(c["dead_pixel_fraction"]) > 0:
        before_total = float(np.sum(I))
        I *= _dead_pixel_mask(I.shape[1:], float(c["dead_pixel_fraction"]), int(c["dead_pixel_seed"]))[None, :, :]
        after_total = float(np.sum(I))
        if before_total > 0:
            trans = float(np.clip(after_total / before_total, 0.0, 1.0))
            metadata["passive_transmitted_power_fraction"] = (
                float(metadata["passive_transmitted_power_fraction"]) * trans
            )
            metadata["slm_dead_pixel_transmitted_fraction"] = trans

    if c["enable_slm_phase_noise"] and float(c["slm_phase_noise_rms_rad"]) > 0:
        before = I.copy()
        I *= _seeded_noise_gain(I.shape, float(c["slm_phase_noise_rms_rad"]), int(c["slm_phase_noise_seed"]))
        I = _match_plane_integrals(I, before)

    if c["enable_slm_active_area"]:
        clip(
            _rect_mask(
                x, y,
                float(c["slm_active_width_um"]),
                float(c["slm_active_height_um"]),
                float(c["slm_active_area_decentre_x_um"]),
                float(c["slm_active_area_decentre_y_um"]),
            ),
            "slm_active_area_clipped_power_fraction",
        )

    if c["enable_zero_order_leakage"] and float(c.get("zero_order_leakage_fraction", 0.0)) > 0:
        frac = np.clip(float(c["zero_order_leakage_fraction"]), 0.0, 0.95)
        ghost = _zero_order_component(I, x, y, str(c["zero_order_mode"]))
        I = (1.0 - frac) * I + frac * ghost

    if c["enable_unwanted_order_leakage"] and float(c["unwanted_order_fraction"]) > 0:
        frac = np.clip(float(c["unwanted_order_fraction"]), 0.0, 0.95)
        ghost = _shift_stack_xy(I, x, y, float(c["unwanted_order_kx_shift"]), float(c["unwanted_order_ky_shift"]))
        I = (1.0 - frac) * I + frac * ghost

    if c["enable_relay_magnification_error"] and float(c["relay_magnification_error_fraction"]) != 0.0:
        I = _magnify_stack(I, x, y, 1.0 + float(c["relay_magnification_error_fraction"]))

    if c["enable_relay_decentre"]:
        I = _shift_stack_xy(I, x, y, float(c["relay_decentre_x_um"]), float(c["relay_decentre_y_um"]))

    if c["enable_relay_tilt"]:
        I = _tilt_walkoff(I, x, y, z, float(c["relay_tilt_x_mrad"]), float(c["relay_tilt_y_mrad"]))
        metadata["phase_ramp_kx_rad_per_um"] += _mrad_to_phase_slope(float(c["relay_tilt_x_mrad"]), c)
        metadata["phase_ramp_ky_rad_per_um"] += _mrad_to_phase_slope(float(c["relay_tilt_y_mrad"]), c)

    if c["enable_relay_aperture"]:
        clip(
            _circle_mask(
                x, y,
                float(c["relay_aperture_radius_um"]),
                float(c["relay_aperture_decentre_x_um"]),
                float(c["relay_aperture_decentre_y_um"]),
            ),
            "aperture_clipped_power_fraction",
        )

    if c["enable_pupil_clipping"]:
        clip(
            _circle_mask(
                x, y,
                float(c["pupil_radius_um"]),
                float(c["pupil_decentre_x_um"]),
                float(c["pupil_decentre_y_um"]),
            ),
            "pupil_clipped_power_fraction",
        )

    if c["enable_zernike_aberrations"]:
        before = I
        I = _apply_zernike_like_distortions(I, x, y, z, c)
        I = _match_plane_integrals(I, before)

    axial_shift = 0.0
    if c["enable_defocus"]:
        axial_shift += float(c["focus_offset_um"])
    if c["enable_focus_depth_error"]:
        axial_shift += float(c["focus_depth_error_um"])
    if axial_shift:
        I = _shift_stack_z(I, z, axial_shift)
        metadata["focus_offset_um"] = axial_shift

    if c["enable_sample_tilt"]:
        metadata["sample_tilt_mrad"] = (float(c["sample_tilt_x_mrad"]), float(c["sample_tilt_y_mrad"]))
    if c["enable_surface_offset"]:
        metadata["sample_surface_z_um"] = float(c["sample_surface_z_um"])

    if c["enable_first_order_filter_decentre"] or c["enable_first_order_filter_clipping"]:
        warnings.append("first-order filter field-active Fourier clipping is not implemented; warning/report only.")
    if c["enable_slm_rotation"] or c["enable_mask_rotation"]:
        warnings.append("SLM/mask rotation is reported but not resampled in Stage 8C.3.")
    if c["enable_axicon_apex_defect"]:
        warnings.append("physical axicon apex defect is warning-only in Stage 8C.3.")
    if c["enable_pointing_jitter"] or c["enable_stage_position_jitter"] or c["enable_focus_drift"]:
        warnings.append("jitter/drift controls need an ensemble; warning-only in Stage 8C.3.")
    if c["enable_detector_noise"] and float(c["detector_noise_fraction"]) > 0:
        metadata["detector_noise_display_only"] = True
    if c["enable_camera_crop"]:
        metadata["camera_crop_display_only"] = True

    I = np.clip(I, 0.0, None)
    metadata["post_perturbation_stack_power_fraction"] = float(np.sum(I) / baseline_stack_integral)
    metadata["diagnostic_nonpassive_power_ratio"] = float(
        metadata["post_perturbation_stack_power_fraction"]
        / max(float(metadata.get("passive_transmitted_power_fraction", 1.0)), 1e-12)
    )
    perturbed = stack_from_arrays(
        I,
        stack.x_um,
        stack.y_um,
        stack.z_um,
        field_label=f"{stack.field_label}_stage8c3_perturbed",
        source_status=stack.source_status,
        metadata={**dict(stack.metadata), **metadata},
    )
    return PerturbationResult(
        baseline_stack=stack,
        perturbed_stack=perturbed,
        control_report=report,
        warnings=tuple(warnings),
        metadata=metadata,
    )


def _control_enabled(control: str, controls: Mapping[str, Any], classification: str) -> bool:
    if classification == "future_not_implemented":
        return bool(controls.get(control, False))
    if control.startswith("enable_"):
        return bool(controls.get(control, False))
    enable_name = _enable_name_for_parameter(control)
    if enable_name in controls:
        return bool(controls.get(enable_name, False))
    if control in {"save_outputs", "show_caveats", "show_warnings", "show_diagnostic_panels", "planning_mode"}:
        return bool(controls.get(control, False))
    return classification in {"energy_active", "geometry_active", "diagnostic_active", "metadata_only", "physics_active"}


def _enable_name_for_parameter(control: str) -> str:
    prefixes = [
        ("beam_decentre_", "enable_beam_decentre"),
        ("beam_tilt_", "enable_beam_tilt"),
        ("beam_radius_", "enable_beam_ellipticity"),
        ("beam_rotation", "enable_beam_ellipticity"),
        ("input_aperture_", "enable_input_aperture"),
        ("slm_phase_centre_offset_", "enable_slm_phase_centre_offset"),
        ("vortex_centre_offset_", "enable_vortex_centre_offset"),
        ("axicon_centre_offset_", "enable_axicon_centre_offset"),
        ("physical_axicon_", "enable_physical_axicon_misalignment"),
        ("axicon_apex_defect_", "enable_axicon_apex_defect"),
        ("slm_phase_levels", "enable_slm_phase_quantisation"),
        ("slm_pixel_pitch", "enable_slm_pixelation"),
        ("slm_fill_factor", "enable_slm_fill_factor"),
        ("dead_pixel_", "enable_slm_dead_pixels"),
        ("slm_phase_noise_", "enable_slm_phase_noise"),
        ("slm_active_", "enable_slm_active_area"),
        ("slm_rotation", "enable_slm_rotation"),
        ("phase_mask_rotation", "enable_mask_rotation"),
        ("zero_order_", "enable_zero_order_leakage"),
        ("unwanted_order_", "enable_unwanted_order_leakage"),
        ("first_order_filter_", "enable_first_order_filter"),
        ("relay_magnification_error_", "enable_relay_magnification_error"),
        ("relay_decentre_", "enable_relay_decentre"),
        ("relay_tilt_", "enable_relay_tilt"),
        ("relay_aperture_", "enable_relay_aperture"),
        ("pupil_", "enable_pupil_clipping"),
        ("zernike_", "enable_zernike_aberrations"),
        ("focus_offset", "enable_defocus"),
        ("focus_depth_error_", "enable_focus_depth_error"),
        ("sample_tilt_", "enable_sample_tilt"),
        ("sample_surface_z", "enable_surface_offset"),
        ("refractive_index_error", "enable_refractive_index_error"),
        ("pulse_energy_jitter_", "enable_pulse_energy_jitter"),
        ("repetition_rate_error_", "enable_repetition_rate_error"),
        ("pulse_duration_error_", "enable_pulse_duration_error"),
        ("pointing_jitter_", "enable_pointing_jitter"),
        ("stage_jitter_", "enable_stage_position_jitter"),
        ("focus_drift_", "enable_focus_drift"),
        ("camera_crop_", "enable_camera_crop"),
        ("detector_noise_", "enable_detector_noise"),
        ("display_autoscale", "enable_display_autoscale"),
    ]
    for prefix, enable in prefixes:
        if control.startswith(prefix):
            return enable
    return ""


def _is_dashboard_relevant_control(control: str) -> bool:
    return control in _STAGE8C3_CONTROL_NAMES or control in _FUTURE_STAGE_CONTROLS


def _mrad_to_phase_slope(mrad: float, controls: Mapping[str, Any]) -> float:
    wavelength_um = float(controls.get("wavelength_nm", 1030.0)) * 1e-3
    return (2.0 * np.pi / wavelength_um) * np.sin(float(mrad) * 1e-3)


def _mesh(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(x, y, indexing="xy")


def _shift_stack_xy(I: np.ndarray, x: np.ndarray, y: np.ndarray, dx_um: float, dy_um: float) -> np.ndarray:
    if dx_um == 0.0 and dy_um == 0.0:
        return I
    return np.asarray([_shift_plane_xy(plane, x, y, dx_um, dy_um) for plane in I], dtype=float)


def _shift_plane_xy(plane: np.ndarray, x: np.ndarray, y: np.ndarray, dx_um: float, dy_um: float) -> np.ndarray:
    # Output at (x, y) samples input at (x - dx, y - dy).
    tmp = np.empty_like(plane, dtype=float)
    xp = x - float(dx_um)
    for j in range(plane.shape[0]):
        tmp[j] = np.interp(xp, x, plane[j], left=0.0, right=0.0)
    yp = y - float(dy_um)
    out = np.empty_like(tmp, dtype=float)
    for i in range(tmp.shape[1]):
        out[:, i] = np.interp(yp, y, tmp[:, i], left=0.0, right=0.0)
    return out


def _tilt_walkoff(I: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray, tx_mrad: float, ty_mrad: float) -> np.ndarray:
    z0 = float(z[np.argmax([np.sum(p) for p in I])]) if z.size else 0.0
    out = np.empty_like(I, dtype=float)
    for i, zi in enumerate(z):
        dx = (float(zi) - z0) * np.tan(float(tx_mrad) * 1e-3)
        dy = (float(zi) - z0) * np.tan(float(ty_mrad) * 1e-3)
        out[i] = _shift_plane_xy(I[i], x, y, dx, dy)
    return out


def _magnify_stack(I: np.ndarray, x: np.ndarray, y: np.ndarray, scale: float) -> np.ndarray:
    if not np.isfinite(scale) or scale <= 0:
        return I
    out = np.empty_like(I, dtype=float)
    for i, plane in enumerate(I):
        out[i] = _resample_plane(plane, x / scale, y / scale, x, y)
    return out


def _resample_plane(plane: np.ndarray, sample_x: np.ndarray, sample_y: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    tmp = np.empty_like(plane, dtype=float)
    for j in range(plane.shape[0]):
        tmp[j] = np.interp(sample_x, x, plane[j], left=0.0, right=0.0)
    out = np.empty_like(tmp, dtype=float)
    for i in range(tmp.shape[1]):
        out[:, i] = np.interp(sample_y, y, tmp[:, i], left=0.0, right=0.0)
    return out


def _match_plane_integrals(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Scale each z-plane in candidate to the corresponding reference integral."""
    out = np.asarray(candidate, dtype=float).copy()
    ref = np.asarray(reference, dtype=float)
    if out.shape != ref.shape:
        return out
    ref_sums = np.sum(ref, axis=(1, 2))
    out_sums = np.sum(out, axis=(1, 2))
    for i, (target, current) in enumerate(zip(ref_sums, out_sums)):
        if float(current) > 0.0 and np.isfinite(current) and np.isfinite(target):
            out[i] *= float(target) / float(current)
    return out


def _elliptical_envelope(x: np.ndarray, y: np.ndarray, rx: float, ry: float, rot_deg: float) -> np.ndarray:
    X, Y = _mesh(x, y)
    theta = np.deg2rad(float(rot_deg))
    Xr = X * np.cos(theta) + Y * np.sin(theta)
    Yr = -X * np.sin(theta) + Y * np.cos(theta)
    rx = max(abs(float(rx)), 1e-9)
    ry = max(abs(float(ry)), 1e-9)
    env = np.exp(-0.5 * ((Xr / rx) ** 2 + (Yr / ry) ** 2))
    return env / max(float(np.max(env)), 1e-12)


def _circle_mask(x: np.ndarray, y: np.ndarray, radius: float, cx: float, cy: float) -> np.ndarray:
    X, Y = _mesh(x, y)
    return (np.hypot(X - float(cx), Y - float(cy)) <= float(radius)).astype(float)


def _rect_mask(x: np.ndarray, y: np.ndarray, width: float, height: float, cx: float, cy: float) -> np.ndarray:
    X, Y = _mesh(x, y)
    return ((np.abs(X - float(cx)) <= float(width) / 2.0) & (np.abs(Y - float(cy)) <= float(height) / 2.0)).astype(float)


def _phase_centre_degrade(I: np.ndarray, x: np.ndarray, y: np.ndarray, dx: float, dy: float) -> np.ndarray:
    X, Y = _mesh(x, y)
    rmax = max(float(np.max(np.hypot(X, Y))), 1e-9)
    off = max(float(np.hypot(dx, dy)), 1e-9)
    directional = (float(dx) * X + float(dy) * Y) / (off * rmax)
    radial = np.hypot(X - float(dx), Y - float(dy)) / rmax
    strength = min(off / max(0.30 * rmax, 1e-9), 2.5)
    gain = np.clip(
        1.0
        + 0.55 * strength * directional
        - 0.25 * strength * radial
        + 0.18 * strength * np.cos(2.0 * np.arctan2(Y, X)),
        0.03,
        2.5,
    )
    core_sigma = max(0.11 * max(float(np.ptp(x)), float(np.ptp(y))), 1e-9)
    core_fill = np.exp(-(X**2 + Y**2) / (2.0 * core_sigma**2))
    core_fill /= max(float(np.max(core_fill)), 1e-12)
    plane_peaks = np.max(I, axis=(1, 2))
    contaminated_core = 0.08 * min(strength, 2.5) * plane_peaks[:, None, None] * core_fill[None, :, :]
    return I * gain[None, :, :] + contaminated_core


def _quantisation_modulation(x: np.ndarray, y: np.ndarray, levels: int) -> np.ndarray:
    X, Y = _mesh(x, y)
    phase = np.arctan2(Y, X)
    q = np.round((phase + np.pi) / (2 * np.pi) * (levels - 1)) / max(levels - 1, 1)
    return np.clip(0.94 + 0.06 * np.cos(2 * np.pi * q), 0.0, None)


def _pixelation_modulation(x: np.ndarray, y: np.ndarray, pitch_um: float) -> np.ndarray:
    X, Y = _mesh(x, y)
    pitch = max(float(pitch_um), 1e-9)
    return 0.98 + 0.02 * np.cos(2 * np.pi * X / pitch) * np.cos(2 * np.pi * Y / pitch)


def _dead_pixel_mask(shape: tuple[int, int], fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    mask = np.ones(shape, dtype=float)
    mask[rng.random(shape) < np.clip(float(fraction), 0.0, 1.0)] = 0.0
    return mask


def _seeded_noise_gain(shape: tuple[int, int, int], rms: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    gain = 1.0 + rng.normal(0.0, min(float(rms), 2.0) * 0.15, size=shape)
    return np.clip(gain, 0.0, None)


def _zero_order_component(I: np.ndarray, x: np.ndarray, y: np.ndarray, mode: str) -> np.ndarray:
    X, Y = _mesh(x, y)
    if mode == "uniform":
        base = np.ones_like(X, dtype=float)
    else:
        sigma = max(0.18 * max(float(np.ptp(x)), float(np.ptp(y))), 1e-9)
        base = np.exp(-(X**2 + Y**2) / (2.0 * sigma**2))
    base_sum = max(float(np.sum(base)), 1e-12)
    plane_sums = np.sum(np.asarray(I, dtype=float), axis=(1, 2))
    return plane_sums[:, None, None] * base[None, :, :] / base_sum


def _apply_zernike_like_distortions(I: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray, c: Mapping[str, Any]) -> np.ndarray:
    out = np.asarray(I, dtype=float).copy()
    X, Y = _mesh(x, y)
    rmax = max(float(np.max(np.hypot(X, Y))), 1e-9)
    xn = X / rmax
    yn = Y / rmax
    rn = np.hypot(xn, yn)
    theta = np.arctan2(yn, xn)

    def waves(name: str) -> float:
        return float(c.get(name, 0.0) or 0.0)

    astig0 = waves("zernike_astig_0_waves")
    astig45 = waves("zernike_astig_45_waves")
    coma_x = waves("zernike_coma_x_waves")
    coma_y = waves("zernike_coma_y_waves")
    spherical = waves("zernike_spherical_waves")
    trefoil_x = waves("zernike_trefoil_x_waves")
    trefoil_y = waves("zernike_trefoil_y_waves")
    defocus = waves("zernike_defocus_waves")

    gain = np.ones_like(X, dtype=float)
    gain += 0.35 * astig0 * (xn**2 - yn**2)
    gain += 0.35 * astig45 * (2.0 * xn * yn)
    gain += 0.45 * coma_x * xn * (3.0 * rn**2 - 2.0)
    gain += 0.45 * coma_y * yn * (3.0 * rn**2 - 2.0)
    gain += 0.25 * spherical * (6.0 * rn**4 - 6.0 * rn**2 + 1.0)
    gain += 0.25 * trefoil_x * rn**3 * np.cos(3.0 * theta)
    gain += 0.25 * trefoil_y * rn**3 * np.sin(3.0 * theta)
    gain = np.clip(gain, 0.02, 3.0)
    out *= gain[None, :, :]
    if defocus:
        out = _shift_stack_z(out, z, 30.0 * defocus)
    return out


def _shift_stack_z(I: np.ndarray, z: np.ndarray, dz_um: float) -> np.ndarray:
    if dz_um == 0.0:
        return I
    out = np.empty_like(I, dtype=float)
    sample_z = z - float(dz_um)
    flat = I.reshape(I.shape[0], -1)
    out_flat = out.reshape(I.shape[0], -1)
    for i in range(flat.shape[1]):
        out_flat[:, i] = np.interp(sample_z, z, flat[:, i], left=flat[0, i], right=flat[-1, i])
    return out
