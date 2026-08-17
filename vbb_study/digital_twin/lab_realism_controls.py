"""Editable lab-realism controls for the Stage 8C.1 cockpit MVP.

This module organises user-editable source-to-sample controls and stage-by-stage
diagnostic bookkeeping. It does not alter optical propagation physics and it
does not implement material response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from vbb_study.digital_twin.energy_accounting import (
    LaserSource,
    OpticalComponent,
    average_power_w,
    compute_energy_ledger,
    fresnel_normal_incidence_transmission,
)
from vbb_study.digital_twin.exposure_bookkeeping import line_exposure_summary
from vbb_study.digital_twin.lab_perturbations import (
    classification_rows_for_controls,
    stage8c3_default_controls,
)

MODEL_STATUS_DIAGNOSTIC = "diagnostic_preview"
MODEL_STATUS_OPTICAL = "optical_prediction"
MODEL_STATUS_ENERGY = "energy_accounting_prediction"
MODEL_STATUS_FLUENCE = "fluence_prediction"
MODEL_STATUS_EXPOSURE = "exposure_bookkeeping"

ALLOWED_STATUS_LEVELS = frozenset(
    {"pass", "caution", "fail", "diagnostic_only", "disabled_future", "missing"}
)

REQUIRED_STAGE_NAMES = [
    "laser_source",
    "pre_slm_beam_conditioning",
    "telescope_or_beam_expander",
    "slm1_phase",
    "slm2_phase_or_axicon",
    "first_order_filter",
    "relay_optics",
    "objective_and_pupil",
    "sample_interface",
    "in_sample_propagation",
    "field_to_fluence",
    "exposure_bookkeeping",
    "future_material_response_disabled",
]

FUTURE_PHYSICS_FLAGS = [
    "enable_material_response",
    "enable_threshold_proxy",
    "enable_dose_accumulation",
    "enable_nonlinear_proxy",
    "enable_thermal_proxy",
    "enable_microscope_proxy",
    "enable_calibrated_prediction",
]


@dataclass(frozen=True)
class LabStageControl:
    """Editable controls and diagnostics for one beam-path stage."""

    stage_name: str
    enabled: bool
    editable_inputs: Mapping[str, Any]
    computed_outputs: Mapping[str, Any] = field(default_factory=dict)
    units: Mapping[str, str] = field(default_factory=dict)
    model_status: str = MODEL_STATUS_DIAGNOSTIC
    warnings: list[str] = field(default_factory=list)
    handoff_to_next_stage: str = ""
    available_metrics: list[str] = field(default_factory=list)
    missing_metrics: list[str] = field(default_factory=list)
    control_classifications: list[Mapping[str, Any]] = field(default_factory=list)
    affected_outputs: list[str] = field(default_factory=list)
    status_level: str = "diagnostic_only"

    def __post_init__(self) -> None:
        if self.stage_name not in REQUIRED_STAGE_NAMES:
            raise ValueError(f"Unknown Stage 8C.1 lab stage: {self.stage_name!r}.")
        if self.status_level not in ALLOWED_STATUS_LEVELS:
            raise ValueError(f"Invalid status_level {self.status_level!r}.")
        object.__setattr__(self, "editable_inputs", dict(self.editable_inputs))
        object.__setattr__(self, "computed_outputs", dict(self.computed_outputs))
        object.__setattr__(self, "units", dict(self.units))
        object.__setattr__(self, "warnings", list(self.warnings))
        object.__setattr__(self, "available_metrics", list(self.available_metrics))
        object.__setattr__(self, "missing_metrics", list(self.missing_metrics))
        rows = [dict(r) for r in self.control_classifications]
        if not rows and self.editable_inputs:
            rows = classification_rows_for_controls(self.editable_inputs)
        affected = list(self.affected_outputs)
        if not affected:
            seen: list[str] = []
            for row in rows:
                for item in str(row.get("affects", "")).split(","):
                    name = item.strip()
                    if name and name not in seen:
                        seen.append(name)
            affected = seen
        object.__setattr__(self, "control_classifications", rows)
        object.__setattr__(self, "affected_outputs", affected)


@dataclass(frozen=True)
class LabStageResult:
    """Compact row used for report display."""

    stage_name: str
    enabled: bool
    key_inputs: str
    key_outputs: str
    model_status: str
    status_level: str
    warnings: str
    missing_metrics: str
    handoff_to_next_stage: str


@dataclass(frozen=True)
class LabRealismReport:
    """Stage-by-stage lab realism report for the integrated cockpit."""

    stages: tuple[LabStageControl, ...]
    model_status: str = MODEL_STATUS_DIAGNOSTIC
    final_export_allowed: bool = False

    def __post_init__(self) -> None:
        names = [s.stage_name for s in self.stages]
        missing = [s for s in REQUIRED_STAGE_NAMES if s not in names]
        if missing:
            raise ValueError(f"LabRealismReport missing stages: {missing}.")

    def to_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for stage in self.stages:
            rows.append(
                {
                    "stage_name": stage.stage_name,
                    "enabled": stage.enabled,
                    "key_inputs": _compact_mapping(stage.editable_inputs),
                    "key_outputs": _compact_mapping(stage.computed_outputs),
                    "model_status": stage.model_status,
                    "status_level": stage.status_level,
                    "warnings": "; ".join(stage.warnings),
                    "missing_metrics": "; ".join(stage.missing_metrics),
                    "control_classifications": _compact_classifications(stage.control_classifications),
                    "affected_outputs": ", ".join(stage.affected_outputs),
                    "handoff_to_next_stage": stage.handoff_to_next_stage,
                }
            )
        return rows

    def to_stage_results(self) -> list[LabStageResult]:
        return [LabStageResult(**row) for row in self.to_rows()]

    def to_dataframe(self) -> Any:
        try:
            import pandas as pd
        except Exception:
            return self.to_rows()
        return pd.DataFrame(self.to_rows())


def default_lab_controls() -> dict[str, Any]:
    """Return editable defaults used by the Stage 8C.1 cockpit notebook."""
    controls = {
        "planning_mode": True,
        "save_outputs": False,
        "figure_dpi": 180,
        "show_caveats": True,
        "show_warnings": True,
        "show_diagnostic_panels": True,
        "engine_preset": "fast",
        "engine_path": "ideal",
        "require_real_field": True,
        "allow_synthetic_demo_field": False,
        "wavelength_nm": 1030.0,
        "pulse_duration_fs": 260.0,
        "repetition_rate_Hz": 25_000.0,
        "pulse_energy_before_optics_uJ": 200.0,
        "average_power_limit_W": 10.0,
        "beam_radius_mm": 2.0,
        "polarisation_state": "linear",
        "pre_slm_transmission": 1.0,
        "input_beam_radius_mm": 2.0,
        "beam_ellipticity": 1.0,
        "pointing_offset_x_um": 0.0,
        "pointing_offset_y_um": 0.0,
        "aperture_radius_mm": None,
        "telescope_enabled": False,
        "telescope_magnification": 1.0,
        "telescope_transmission": 1.0,
        "slm1_enabled": True,
        "phase_profile": "vortex",
        "ell": 3,
        "phase_quantisation_levels": 256,
        "slm_pixel_pitch_um": 8.0,
        "slm_active_width_px": None,
        "slm_active_height_px": None,
        "slm1_diffraction_efficiency": 0.95,
        "generation_method": "holographic",
        "axicon_k_r": None,
        "target_core_diameter_um": 3.0,
        "target_bessel_length_um": 150.0,
        "blaze_period_px": 20,
        "slm2_diffraction_efficiency": 0.95,
        "slm2_conjugate_mode": "preserve_vortex",
        "physical_axicon_enabled": False,
        "physical_axicon_angle_deg": None,
        "first_order_filter_enabled": True,
        "selected_first_order_fraction": 0.73,
        "filter_radius_px_or_lpmm": None,
        "zero_order_leakage_fraction": 0.0,
        "relay_transmission": 0.90,
        "relay_magnification": 1.0,
        "relay_aberration_enabled": False,
        "objective_NA": 0.45,
        "objective_transmission": 0.85,
        "objective_effective_focal_length_mm": None,
        "objective_pupil_diameter_mm": None,
        "pupil_diameter_override_mm": None,
        "material_name": "Cr:ZnSe",
        "refractive_index": 2.44,
        "sample_thickness_mm": 5.0,
        "focus_depth_um": 100.0,
        "sample_interface_transmission": 0.95,
        "use_fresnel_interface_estimate": False,
        "surface_tilt_mrad": 0.0,
        "grid_N": None,
        "device_downsample": None,
        "axial_planes": None,
        "crop_window_um": None,
        "fluence_normalisation_mode": "per_plane_transverse_energy",
        "selected_z_mode": "target_depth",
        "selected_z_um": "target_depth",
        "custom_z_um": 100.0,
        "central_roi_half_width_um": 10.0,
        "display_scaling": "percentile",
        "display_percentile_clip": (0.5, 99.5),
        "writing_mode": "line_x",
        "scan_axis": "x",
        "scan_speed_mm_s": 1.0,
        "line_length_um": 500.0,
        "effective_diameter_um": 3.0,
        "num_static_pulses": 100,
        "num_passes": 1,
        "z_step_um": 0.0,
        "tilt_angle_deg": 0.0,
        "enable_material_response": False,
        "enable_threshold_proxy": False,
        "enable_dose_accumulation": False,
        "enable_nonlinear_proxy": False,
        "enable_thermal_proxy": False,
        "enable_microscope_proxy": False,
        "enable_calibrated_prediction": False,
    }
    controls.update(stage8c3_default_controls())
    return controls


def validate_future_physics_disabled(controls: Mapping[str, Any]) -> None:
    """Raise if any future-physics toggle is enabled in Stage 8C.1."""
    enabled = [name for name in FUTURE_PHYSICS_FLAGS if bool(controls.get(name, False))]
    if enabled:
        raise NotImplementedError(
            "Stage 8C.1 is optical/energy/exposure bookkeeping only. "
            f"These future physics controls are not implemented here: {', '.join(enabled)}."
        )


def _effective_pulse_energy_uJ(c: Mapping[str, Any]) -> float:
    energy = float(c["pulse_energy_before_optics_uJ"])
    if c.get("enable_pulse_energy_jitter") and float(c.get("pulse_energy_jitter_rms_fraction", 0.0)) > 0:
        rng = np.random.default_rng(int(c.get("pulse_energy_jitter_seed", 23)))
        energy *= max(0.0, 1.0 + float(rng.normal(0.0, float(c["pulse_energy_jitter_rms_fraction"]))))
    return float(energy)


def _effective_repetition_rate_hz(c: Mapping[str, Any]) -> float:
    rep_rate = float(c["repetition_rate_Hz"])
    if c.get("enable_repetition_rate_error"):
        rep_rate *= max(0.0, 1.0 + float(c.get("repetition_rate_error_fraction", 0.0)))
    return float(rep_rate)


def _effective_pulse_duration_fs(c: Mapping[str, Any]) -> float:
    duration = float(c["pulse_duration_fs"])
    if c.get("enable_pulse_duration_error"):
        duration *= max(1e-9, 1.0 + float(c.get("pulse_duration_error_fraction", 0.0)))
    return float(duration)


def _effective_average_power_limit_w(c: Mapping[str, Any]) -> float | None:
    if not c.get("enable_average_power_limit", True):
        return None
    limit = c.get("average_power_limit_W")
    return None if limit is None else float(limit)


def build_laser_source_from_controls(controls: Mapping[str, Any]) -> LaserSource:
    c = _with_defaults(controls)
    return LaserSource(
        wavelength_nm=float(c["wavelength_nm"]),
        pulse_duration_fs=_effective_pulse_duration_fs(c),
        repetition_rate_Hz=_effective_repetition_rate_hz(c),
        pulse_energy_before_optics_uJ=_effective_pulse_energy_uJ(c),
        average_power_limit_W=_effective_average_power_limit_w(c),
        beam_radius_mm=float(c["beam_radius_mm"]),
        polarisation_state=str(c["polarisation_state"]),
    )


def build_energy_components_from_controls(controls: Mapping[str, Any]) -> list[OpticalComponent]:
    """Build the editable Stage 8C.1 energy chain without altering physics."""
    c = _with_defaults(controls)
    return [
        OpticalComponent(
            "pre_slm_beam_conditioning",
            "passive_optics",
            float(c["pre_slm_transmission"]),
            notes="pre-SLM conditioning / telescope input",
        ),
        OpticalComponent(
            "telescope_or_beam_expander",
            "passive_optics",
            float(c["telescope_transmission"]),
            enabled=bool(c["telescope_enabled"]),
            notes="editable telescope/beam-expander transmission",
        ),
        OpticalComponent(
            "slm1_phase",
            "diffractive_optics",
            1.0,
            float(c["slm1_diffraction_efficiency"]),
            enabled=bool(c["slm1_enabled"]),
            notes=f"SLM1 phase profile {c['phase_profile']}",
        ),
        OpticalComponent(
            "slm2_phase_or_axicon",
            "diffractive_or_physical_axicon",
            1.0,
            float(c["slm2_diffraction_efficiency"]),
            notes=f"generation method {c['generation_method']}",
        ),
        OpticalComponent(
            "first_order_filter",
            "fourier_filter",
            float(c["selected_first_order_fraction"]),
            enabled=bool(c["first_order_filter_enabled"]),
            notes="selected first diffraction order",
        ),
        OpticalComponent(
            "relay_optics",
            "passive_optics",
            float(c["relay_transmission"]),
            notes="relay optics transmission",
        ),
        OpticalComponent(
            "objective_and_pupil",
            "objective",
            float(c["objective_transmission"]),
            notes="objective transmission",
        ),
        OpticalComponent(
            "sample_interface",
            "interface",
            _sample_interface_transmission(c),
            notes="sample entry interface",
        ),
    ]


def build_energy_ledger_from_controls(controls: Mapping[str, Any]):
    c = _with_defaults(controls)
    source = build_laser_source_from_controls(c)
    components = build_energy_components_from_controls(c)
    return compute_energy_ledger(
        source.pulse_energy_before_optics_uJ,
        source.repetition_rate_Hz,
        components,
        average_power_limit_W=source.average_power_limit_W,
        source=source,
    )


def build_exposure_summary_from_controls(
    controls: Mapping[str, Any],
    pulse_energy_at_sample_uJ: float,
    repetition_rate_Hz: float | None = None,
) -> dict[str, Any]:
    c = _with_defaults(controls)
    rep_rate = float(repetition_rate_Hz if repetition_rate_Hz is not None else _effective_repetition_rate_hz(c))
    return dict(
        line_exposure_summary(
            pulse_energy_at_sample_uJ=float(pulse_energy_at_sample_uJ),
            repetition_rate_Hz=rep_rate,
            scan_speed_mm_s=float(c["scan_speed_mm_s"]),
            line_length_um=float(c["line_length_um"]),
            effective_diameter_um=float(c["effective_diameter_um"]),
        )
    )


def build_lab_realism_report(
    controls: Mapping[str, Any],
    *,
    energy_ledger: Any | None = None,
    exposure_summary: Mapping[str, Any] | None = None,
    field_summary: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> LabRealismReport:
    """Build the required stage-by-stage lab realism report."""
    c = _with_defaults(controls)
    validate_future_physics_disabled(c)

    if energy_ledger is None:
        energy_ledger = build_energy_ledger_from_controls(c)

    if exposure_summary is None:
        source = build_laser_source_from_controls(c)
        exposure_summary = build_exposure_summary_from_controls(
            c,
            pulse_energy_at_sample_uJ=float(energy_ledger.energy_at_sample_uJ),
            repetition_rate_Hz=float(source.repetition_rate_Hz),
        )

    stages = [
        _laser_source_stage(c),
        _pre_slm_stage(c),
        _telescope_stage(c),
        _slm1_stage(c),
        _slm2_stage(c),
        _first_order_stage(c),
        _relay_stage(c),
        _objective_stage(c),
        _sample_stage(c),
        _propagation_stage(c, field_summary),
        _field_to_fluence_stage(c, diagnostics),
        _exposure_stage(c, exposure_summary),
        _future_disabled_stage(c),
    ]
    return LabRealismReport(stages=tuple(stages))


def _laser_source_stage(c: Mapping[str, Any]) -> LabStageControl:
    pulse_energy = _effective_pulse_energy_uJ(c)
    repetition_rate = _effective_repetition_rate_hz(c)
    pulse_duration = _effective_pulse_duration_fs(c)
    avg_power = average_power_w(pulse_energy, repetition_rate)
    limit = _effective_average_power_limit_w(c)
    warnings: list[str] = []
    status = "pass"
    if limit is not None and avg_power > float(limit):
        warnings.append("average power before optics exceeds configured limit")
        status = "fail"
    return LabStageControl(
        stage_name="laser_source",
        enabled=True,
        editable_inputs=_pick(
            c,
            "wavelength_nm",
            "pulse_duration_fs",
            "repetition_rate_Hz",
            "pulse_energy_before_optics_uJ",
            "enable_pulse_energy_jitter",
            "pulse_energy_jitter_rms_fraction",
            "pulse_energy_jitter_seed",
            "enable_repetition_rate_error",
            "repetition_rate_error_fraction",
            "enable_pulse_duration_error",
            "pulse_duration_error_fraction",
            "enable_average_power_limit",
            "average_power_limit_W",
            "beam_radius_mm",
            "polarisation_state",
        ),
        computed_outputs={
            "average_power_before_optics_W": avg_power,
            "power_limit_status": "below_limit" if not warnings else "above_limit",
            "effective_pulse_energy_before_optics_uJ": pulse_energy,
            "effective_repetition_rate_Hz": repetition_rate,
            "effective_pulse_duration_fs": pulse_duration,
        },
        units={
            "wavelength_nm": "nm",
            "pulse_duration_fs": "fs",
            "repetition_rate_Hz": "Hz",
            "pulse_energy_before_optics_uJ": "uJ",
            "beam_radius_mm": "mm",
        },
        model_status=MODEL_STATUS_ENERGY,
        warnings=warnings,
        handoff_to_next_stage="pulse energy and source metadata to pre-SLM conditioning",
        available_metrics=["average_power_before_optics_W", "power_limit_status"],
        status_level=status,
    )


def _pre_slm_stage(c: Mapping[str, Any]) -> LabStageControl:
    energy_after = c["pulse_energy_before_optics_uJ"] * c["pre_slm_transmission"]
    aperture = c.get("aperture_radius_mm")
    warnings: list[str] = []
    status = "pass"
    clipping = "not available from current engine"
    if aperture is not None:
        clipping = "clear"
        if float(aperture) < float(c["input_beam_radius_mm"]):
            clipping = "aperture clips input beam"
            warnings.append("aperture radius is smaller than input beam radius")
            status = "caution"
    return LabStageControl(
        stage_name="pre_slm_beam_conditioning",
        enabled=True,
        editable_inputs=_pick(
            c,
            "pre_slm_transmission",
            "input_beam_radius_mm",
            "beam_ellipticity",
            "pointing_offset_x_um",
            "pointing_offset_y_um",
            "aperture_radius_mm",
            "enable_beam_decentre",
            "beam_decentre_x_um",
            "beam_decentre_y_um",
            "enable_beam_tilt",
            "beam_tilt_x_mrad",
            "beam_tilt_y_mrad",
            "enable_beam_ellipticity",
            "beam_radius_x_um",
            "beam_radius_y_um",
            "beam_rotation_deg",
            "enable_input_aperture",
            "input_aperture_radius_um",
            "input_aperture_decentre_x_um",
            "input_aperture_decentre_y_um",
        ),
        computed_outputs={
            "energy_after_pre_slm_uJ": energy_after,
            "beam_radius_at_slm_mm": c["input_beam_radius_mm"],
            "aperture_clipping_warning": clipping,
        },
        model_status=MODEL_STATUS_ENERGY,
        warnings=warnings,
        handoff_to_next_stage="conditioned beam radius and energy to telescope/SLM",
        available_metrics=["energy_after_pre_slm_uJ", "beam_radius_at_slm_mm"],
        missing_metrics=[] if aperture is not None else ["aperture clipping not available from current engine"],
        status_level=status if aperture is not None else "missing",
    )


def _telescope_stage(c: Mapping[str, Any]) -> LabStageControl:
    radius = float(c["input_beam_radius_mm"])
    if c["telescope_enabled"]:
        radius *= float(c["telescope_magnification"])
    energy_after = c["pulse_energy_before_optics_uJ"] * c["pre_slm_transmission"] * (
        float(c["telescope_transmission"]) if c["telescope_enabled"] else 1.0
    )
    return LabStageControl(
        stage_name="telescope_or_beam_expander",
        enabled=bool(c["telescope_enabled"]),
        editable_inputs=_pick(c, "telescope_enabled", "telescope_magnification", "telescope_transmission"),
        computed_outputs={
            "beam_radius_after_telescope_mm": radius,
            "energy_after_telescope_uJ": energy_after,
        },
        model_status=MODEL_STATUS_ENERGY,
        handoff_to_next_stage="beam radius and energy to SLM1",
        available_metrics=["beam_radius_after_telescope_mm", "energy_after_telescope_uJ"],
        status_level="pass" if c["telescope_enabled"] else "diagnostic_only",
    )


def _slm1_stage(c: Mapping[str, Any]) -> LabStageControl:
    warnings = []
    status = "pass"
    if int(c["phase_quantisation_levels"]) < 16:
        warnings.append("phase quantisation levels are low")
        status = "caution"
    sampling = "active area not available from current engine"
    if c["slm_active_width_px"] and c["slm_active_height_px"]:
        sampling = "configured"
    return LabStageControl(
        stage_name="slm1_phase",
        enabled=bool(c["slm1_enabled"]),
        editable_inputs=_pick(
            c,
            "slm1_enabled",
            "phase_profile",
            "ell",
            "phase_quantisation_levels",
            "slm_pixel_pitch_um",
            "slm_active_width_px",
            "slm_active_height_px",
            "slm1_diffraction_efficiency",
            "enable_slm_phase_centre_offset",
            "slm_phase_centre_offset_x_um",
            "slm_phase_centre_offset_y_um",
            "enable_vortex_centre_offset",
            "vortex_centre_offset_x_um",
            "vortex_centre_offset_y_um",
            "enable_slm_phase_quantisation",
            "slm_phase_levels",
            "enable_slm_pixelation",
            "enable_slm_fill_factor",
            "slm_fill_factor",
            "enable_slm_dead_pixels",
            "dead_pixel_fraction",
            "dead_pixel_seed",
            "enable_slm_phase_noise",
            "slm_phase_noise_rms_rad",
            "slm_phase_noise_seed",
            "enable_slm_active_area",
            "slm_active_width_um",
            "slm_active_height_um",
            "slm_active_area_decentre_x_um",
            "slm_active_area_decentre_y_um",
            "enable_slm_rotation",
            "slm_rotation_deg",
            "enable_mask_rotation",
            "phase_mask_rotation_deg",
        ),
        computed_outputs={
            "slm_sampling_status": sampling,
            "slm_phase_quantisation_status": "ok" if not warnings else "coarse",
            "energy_after_slm1_uJ": "computed in energy ledger",
        },
        model_status=MODEL_STATUS_OPTICAL,
        warnings=warnings,
        handoff_to_next_stage="vortex/phase state and energy to SLM2 route",
        available_metrics=["slm_phase_quantisation_status"],
        missing_metrics=[] if sampling == "configured" else ["SLM active area not available from current engine"],
        status_level=status if c["slm1_enabled"] else "diagnostic_only",
    )


def _slm2_stage(c: Mapping[str, Any]) -> LabStageControl:
    warnings = []
    method = str(c["generation_method"])
    route_status = "holographic route"
    if method == "physical" or c["physical_axicon_enabled"]:
        route_status = "physical axicon route selected"
        if c["physical_axicon_angle_deg"] is None:
            warnings.append("physical axicon angle is not configured")
    return LabStageControl(
        stage_name="slm2_phase_or_axicon",
        enabled=True,
        editable_inputs=_pick(
            c,
            "generation_method",
            "axicon_k_r",
            "target_core_diameter_um",
            "target_bessel_length_um",
            "blaze_period_px",
            "slm2_diffraction_efficiency",
            "slm2_conjugate_mode",
            "physical_axicon_enabled",
            "physical_axicon_angle_deg",
            "enable_axicon_centre_offset",
            "axicon_centre_offset_x_um",
            "axicon_centre_offset_y_um",
            "enable_physical_axicon_misalignment",
            "physical_axicon_apex_offset_x_um",
            "physical_axicon_apex_offset_y_um",
            "physical_axicon_tilt_x_mrad",
            "physical_axicon_tilt_y_mrad",
            "physical_axicon_angle_error_deg",
            "enable_axicon_apex_defect",
            "axicon_apex_defect_radius_um",
        ),
        computed_outputs={
            "route_status": route_status,
            "measured_topological_winding_if_available": "not available from current engine",
            "physical_route_charge_warning_if_applicable": "; ".join(warnings) if warnings else "none",
            "energy_after_slm2_uJ": "computed in energy ledger",
        },
        model_status=MODEL_STATUS_OPTICAL,
        warnings=warnings,
        handoff_to_next_stage="routed diffracted field to first-order filtering",
        available_metrics=["route_status"],
        missing_metrics=["measured topological winding not available from current engine"],
        status_level="caution" if warnings else "pass",
    )


def _first_order_stage(c: Mapping[str, Any]) -> LabStageControl:
    warnings = []
    valid = bool(c["first_order_filter_enabled"]) and float(c["selected_first_order_fraction"]) > 0
    if float(c["zero_order_leakage_fraction"]) > 0.02:
        warnings.append("zero-order leakage fraction is non-negligible")
    return LabStageControl(
        stage_name="first_order_filter",
        enabled=bool(c["first_order_filter_enabled"]),
        editable_inputs=_pick(
            c,
            "first_order_filter_enabled",
            "selected_first_order_fraction",
            "filter_radius_px_or_lpmm",
            "zero_order_leakage_fraction",
            "enable_zero_order_leakage",
            "zero_order_mode",
            "enable_unwanted_order_leakage",
            "unwanted_order_fraction",
            "unwanted_order_kx_shift",
            "unwanted_order_ky_shift",
            "enable_first_order_filter",
            "first_order_filter_radius_px",
            "enable_first_order_filter_decentre",
            "first_order_filter_decentre_x_px",
            "first_order_filter_decentre_y_px",
            "enable_first_order_filter_clipping",
            "first_order_filter_clipping_fraction_override",
        ),
        computed_outputs={
            "carrier_frequency_lpmm": "not available from current engine",
            "cone_frequency_lpmm": "not available from current engine",
            "first_order_geometry_valid": valid,
            "first_order_selected_energy_uJ": "computed in energy ledger",
            "zero_order_warning": "; ".join(warnings) if warnings else "none",
        },
        model_status=MODEL_STATUS_OPTICAL,
        warnings=warnings,
        handoff_to_next_stage="selected diffraction order to relay optics",
        available_metrics=["first_order_geometry_valid", "zero_order_warning"],
        missing_metrics=["carrier/cone frequency not available from current engine"],
        status_level="pass" if valid and not warnings else "caution",
    )


def _relay_stage(c: Mapping[str, Any]) -> LabStageControl:
    warnings = ["relay aberration enabled as label only"] if c["relay_aberration_enabled"] else []
    return LabStageControl(
        stage_name="relay_optics",
        enabled=True,
        editable_inputs=_pick(
            c,
            "relay_transmission",
            "relay_magnification",
            "relay_aberration_enabled",
            "enable_relay_magnification_error",
            "relay_magnification_error_fraction",
            "enable_relay_decentre",
            "relay_decentre_x_um",
            "relay_decentre_y_um",
            "enable_relay_tilt",
            "relay_tilt_x_mrad",
            "relay_tilt_y_mrad",
            "enable_relay_aperture",
            "relay_aperture_radius_um",
            "relay_aperture_decentre_x_um",
            "relay_aperture_decentre_y_um",
        ),
        computed_outputs={
            "energy_after_relay_uJ": "computed in energy ledger",
            "effective_beam_scale": c["relay_magnification"],
        },
        model_status=MODEL_STATUS_ENERGY,
        warnings=warnings,
        handoff_to_next_stage="relayed beam to objective pupil",
        available_metrics=["effective_beam_scale"],
        status_level="caution" if warnings else "pass",
    )


def _objective_stage(c: Mapping[str, Any]) -> LabStageControl:
    pupil_diam = c.get("pupil_diameter_override_mm") or c.get("objective_pupil_diameter_mm")
    beam_radius = float(c["input_beam_radius_mm"])
    if c["telescope_enabled"]:
        beam_radius *= float(c["telescope_magnification"])
    fill = clipping = None
    missing = []
    warnings = []
    status = "pass"
    if pupil_diam:
        pupil_radius = float(pupil_diam) / 2.0
        fill = beam_radius / pupil_radius if pupil_radius > 0 else np.inf
        clipping = max(0.0, fill - 1.0)
        if clipping > 0:
            warnings.append("beam overfills objective pupil")
            status = "caution"
    else:
        pupil_radius = "not available from current engine"
        missing.append("objective pupil diameter not available from current engine")
        status = "missing"
    return LabStageControl(
        stage_name="objective_and_pupil",
        enabled=True,
        editable_inputs=_pick(
            c,
            "objective_NA",
            "objective_transmission",
            "objective_effective_focal_length_mm",
            "objective_pupil_diameter_mm",
            "pupil_diameter_override_mm",
            "enable_pupil_clipping",
            "pupil_radius_um",
            "pupil_decentre_x_um",
            "pupil_decentre_y_um",
            "pupil_fill_target_fraction",
            "enable_zernike_aberrations",
            "zernike_defocus_waves",
            "zernike_astig_0_waves",
            "zernike_astig_45_waves",
            "zernike_coma_x_waves",
            "zernike_coma_y_waves",
            "zernike_spherical_waves",
            "zernike_trefoil_x_waves",
            "zernike_trefoil_y_waves",
        ),
        computed_outputs={
            "pupil_radius_mm": pupil_radius,
            "pupil_fill_fraction": "not available from current engine" if fill is None else fill,
            "pupil_clipping_fraction": "not available from current engine" if clipping is None else clipping,
            "pupil_status": "missing pupil diameter" if fill is None else ("overfilled" if clipping else "ok"),
            "energy_after_objective_uJ": "computed in energy ledger",
        },
        model_status=MODEL_STATUS_OPTICAL,
        warnings=warnings,
        handoff_to_next_stage="focused field to sample interface",
        available_metrics=["pupil_status"],
        missing_metrics=missing,
        status_level=status,
    )


def _sample_stage(c: Mapping[str, Any]) -> LabStageControl:
    fresnel = fresnel_normal_incidence_transmission(1.0, float(c["refractive_index"]))
    target = float(c["focus_depth_um"])
    thickness_um = float(c["sample_thickness_mm"]) * 1000.0
    warnings = []
    status = "pass"
    if target < 0 or target > thickness_um:
        warnings.append("target focus depth is outside sample thickness")
        status = "fail"
    return LabStageControl(
        stage_name="sample_interface",
        enabled=True,
        editable_inputs=_pick(
            c,
            "material_name",
            "refractive_index",
            "sample_thickness_mm",
            "focus_depth_um",
            "sample_interface_transmission",
            "use_fresnel_interface_estimate",
            "surface_tilt_mrad",
            "enable_defocus",
            "focus_offset_um",
            "enable_focus_depth_error",
            "focus_depth_error_um",
            "enable_sample_tilt",
            "sample_tilt_x_mrad",
            "sample_tilt_y_mrad",
            "enable_surface_offset",
            "sample_surface_z_um",
            "enable_refractive_index_error",
            "refractive_index_error",
            "enable_sample_thickness_limit",
            "enable_interface_reflection",
        ),
        computed_outputs={
            "fresnel_normal_incidence_transmission": fresnel,
            "energy_entering_sample_uJ": "computed in energy ledger",
            "surface_z_um": 0.0,
            "target_depth_z_um": target,
            "sample_bounds_status": "inside sample" if not warnings else "outside sample",
        },
        model_status=MODEL_STATUS_ENERGY,
        warnings=warnings,
        handoff_to_next_stage="sample-entered optical field to propagation stack",
        available_metrics=["fresnel_normal_incidence_transmission", "sample_bounds_status"],
        status_level=status,
    )


def _propagation_stage(c: Mapping[str, Any], field_summary: Mapping[str, Any] | None) -> LabStageControl:
    missing = []
    warnings = []
    outputs = {
        "field_source_status": "not available from current engine",
        "grid_sampling_status": "not available from current engine",
        "crop_window_status": "not available from current engine",
        "raw_captured_power_drift": "not available from current engine",
        "bessel_zone_length_if_available": "not available from current engine",
        "strict_bessel_region_if_available": "not available from current engine",
        "core_or_ring_diameter_if_available": "not available from current engine",
        "side_lobe_ratio_if_available": "not available from current engine",
    }
    status = "missing"
    if field_summary:
        outputs.update(
            {
                "field_source_status": field_summary.get("source_status", "available"),
                "grid_sampling_status": f"dx={field_summary.get('dx_um', 'na')} um, dy={field_summary.get('dy_um', 'na')} um",
                "raw_captured_power_drift": field_summary.get("propagation_energy_drift_fraction", "not available from current engine"),
            }
        )
        status = "diagnostic_only"
    else:
        missing.append("real optical field summary not available from current engine")
        if c["require_real_field"]:
            warnings.append("real field is required by controls")
            status = "fail"
    return LabStageControl(
        stage_name="in_sample_propagation",
        enabled=True,
        editable_inputs=_pick(
            c,
            "engine_preset",
            "engine_path",
            "grid_N",
            "device_downsample",
            "axial_planes",
            "crop_window_um",
            "require_real_field",
            "allow_synthetic_demo_field",
            "enable_pointing_jitter",
            "pointing_jitter_rms_urad",
            "pointing_jitter_seed",
            "enable_stage_position_jitter",
            "stage_jitter_x_um",
            "stage_jitter_y_um",
            "stage_jitter_z_um",
            "stage_jitter_seed",
            "enable_focus_drift",
            "focus_drift_um_per_min",
        ),
        computed_outputs=outputs,
        model_status=MODEL_STATUS_OPTICAL,
        warnings=warnings,
        handoff_to_next_stage="real optical intensity stack to fluence scaling",
        available_metrics=["field_source_status", "grid_sampling_status", "raw_captured_power_drift"],
        missing_metrics=missing + [
            "bessel zone length not available from current engine",
            "side-lobe ratio not available from current engine",
        ],
        status_level=status,
    )


def _field_to_fluence_stage(c: Mapping[str, Any], diagnostics: Mapping[str, Any] | None) -> LabStageControl:
    outputs = {
        "selected_plane_z_um": "not available from current engine",
        "selected_plane_reason": "not available from current engine",
        "peak_fluence_j_cm2": "not available from current engine",
        "central_roi_peak_fluence_j_cm2": "not available from current engine",
        "target_depth_peak_fluence_j_cm2": "not available from current engine",
        "peak_intensity_w_cm2": "not available from current engine",
        "edge_peak_warning": "not available from current engine",
    }
    warnings = []
    status = "missing"
    if diagnostics:
        outputs.update(
            {
                "selected_plane_z_um": diagnostics.get("selected_plane_z_um"),
                "selected_plane_reason": diagnostics.get("selected_plane_reason"),
                "peak_fluence_j_cm2": diagnostics.get("global_peak_value"),
                "central_roi_peak_fluence_j_cm2": diagnostics.get("central_roi_peak_value"),
                "target_depth_peak_fluence_j_cm2": diagnostics.get("target_depth_peak_value"),
                "peak_intensity_w_cm2": diagnostics.get("peak_intensity_w_cm2", "not available from current engine"),
                "edge_peak_warning": diagnostics.get("global_peak_near_boundary"),
            }
        )
        if diagnostics.get("global_peak_near_boundary"):
            warnings.append("global peak is near crop boundary")
            status = "caution"
        else:
            status = "pass"
    return LabStageControl(
        stage_name="field_to_fluence",
        enabled=True,
        editable_inputs=_pick(
            c,
            "fluence_normalisation_mode",
            "selected_z_mode",
            "selected_z_um",
            "custom_z_um",
            "central_roi_half_width_um",
            "display_scaling",
            "display_percentile_clip",
            "enable_camera_crop",
            "camera_crop_width_um",
            "camera_crop_height_um",
            "camera_crop_centre_x_um",
            "camera_crop_centre_y_um",
            "enable_detector_noise",
            "detector_noise_fraction",
            "detector_noise_seed",
            "enable_display_autoscale",
            "display_autoscale",
        ),
        computed_outputs=outputs,
        model_status=MODEL_STATUS_FLUENCE,
        warnings=warnings,
        handoff_to_next_stage="selected optical fluence metrics to exposure/interpretation panels",
        available_metrics=["selected_plane_z_um", "peak_fluence_j_cm2", "central_roi_peak_fluence_j_cm2"],
        missing_metrics=[] if diagnostics else ["fluence diagnostics not available from current engine"],
        status_level=status,
    )


def _exposure_stage(c: Mapping[str, Any], exposure_summary: Mapping[str, Any]) -> LabStageControl:
    warnings = list(exposure_summary.get("warnings", []) or [])
    if float(exposure_summary.get("pulses_per_spot", 0.0)) > 100:
        warnings.append("high-overlap heat-accumulation warning label only")
    return LabStageControl(
        stage_name="exposure_bookkeeping",
        enabled=str(c["writing_mode"]) != "disabled",
        editable_inputs=_pick(
            c,
            "writing_mode",
            "scan_axis",
            "scan_speed_mm_s",
            "line_length_um",
            "effective_diameter_um",
            "num_static_pulses",
            "num_passes",
            "z_step_um",
            "tilt_angle_deg",
        ),
        computed_outputs={
            "pulse_spacing_um": exposure_summary.get("pulse_spacing_um"),
            "pulses_per_spot": exposure_summary.get("pulses_per_spot"),
            "line_duration_s": exposure_summary.get("line_duration_s"),
            "total_pulses_on_line": exposure_summary.get("total_pulses_on_line"),
            "dose_per_unit_length_proxy": exposure_summary.get("dose_per_unit_length_J_m"),
            "overlap_fraction": exposure_summary.get("overlap_fraction"),
            "continuity_warning": "; ".join(exposure_summary.get("warnings", []) or []) or "none",
            "heat_accumulation_warning_label_only": "present" if any("heat" in w.lower() for w in warnings) else "none",
        },
        model_status=MODEL_STATUS_EXPOSURE,
        warnings=warnings,
        handoff_to_next_stage="exposure geometry summary; material-response modules remain disabled",
        available_metrics=["pulse_spacing_um", "pulses_per_spot", "overlap_fraction"],
        status_level="caution" if warnings else "pass",
    )


def _future_disabled_stage(c: Mapping[str, Any]) -> LabStageControl:
    return LabStageControl(
        stage_name="future_material_response_disabled",
        enabled=False,
        editable_inputs={name: bool(c.get(name, False)) for name in FUTURE_PHYSICS_FLAGS},
        computed_outputs={
            "disabled_reason": "Stage 8C.1 is optical/energy/exposure only",
            "required_future_stage": "Stage 8E+ for exposure trajectory and calibrated material modules",
            "required_calibration_data": "material constants, absorption, thresholds, thermal data, and measured calibration",
        },
        model_status=MODEL_STATUS_DIAGNOSTIC,
        warnings=[],
        handoff_to_next_stage="future controls are intentionally disabled",
        available_metrics=["disabled_reason"],
        missing_metrics=[
            "material calibration not implemented",
            "threshold maps are future control / not implemented",
            "thermal accumulation is future control / not implemented",
        ],
        status_level="disabled_future",
    )


def _with_defaults(controls: Mapping[str, Any]) -> dict[str, Any]:
    base = default_lab_controls()
    base.update(dict(controls))
    return base


def _pick(mapping: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    return {k: mapping.get(k) for k in keys}


def _compact_mapping(mapping: Mapping[str, Any], limit: int = 6) -> str:
    items = list(mapping.items())
    parts = [f"{k}={_fmt_value(v)}" for k, v in items[:limit]]
    if len(items) > limit:
        parts.append(f"... +{len(items) - limit}")
    return "; ".join(parts)


def _compact_classifications(rows: list[Mapping[str, Any]], limit: int = 4) -> str:
    parts = [
        f"{row.get('control')}:{row.get('classification')}"
        for row in rows[:limit]
    ]
    if len(rows) > limit:
        parts.append(f"... +{len(rows) - limit}")
    return "; ".join(parts)


def _fmt_value(value: Any) -> str:
    if isinstance(value, float):
        if np.isfinite(value):
            return f"{value:.4g}"
        return str(value)
    if isinstance(value, (list, tuple)) and len(value) <= 3:
        return "(" + ", ".join(_fmt_value(v) for v in value) + ")"
    return str(value)


def _sample_interface_transmission(c: Mapping[str, Any]) -> float:
    if c.get("use_fresnel_interface_estimate") or c.get("enable_interface_reflection"):
        refractive_index = float(c["refractive_index"])
        if c.get("enable_refractive_index_error"):
            refractive_index += float(c.get("refractive_index_error", 0.0))
        return fresnel_normal_incidence_transmission(1.0, refractive_index)
    return float(c["sample_interface_transmission"])
