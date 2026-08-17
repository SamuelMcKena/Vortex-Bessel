"""Stage 8C.3R.5.2 editable hardware / geometry / numerical control contract.

This is a *metadata layer* over the existing Stage 8C.3R.5 / R5.1 config
dataclasses (``CSLMRouteConfig`` is the single source of model values).  It does
NOT add new optical physics and does NOT change the R5.1 active-route or
benchmark-route meaning.

Every user-facing parameter is described by an :class:`EditableControl` with an
explicit status and provenance, so the notebook can present one coherent control
table and a versioned hardware profile without inventing measured laboratory
geometry.

Hard boundary (unchanged): n = 1.0 free-space optical-field / fluence diagnostics
only; ``fourier_filter_physics_available = False``; ``diagnostic_only``;
``final_export_allowed = False``.  No material / 4F / camera / GUI / 3D physics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from vbb_study.digital_twin.cslm_route import (
    CSLMRouteConfig,
    FOUR_F_REQUIRED_PARAMETERS,
    FOUR_F_BLOCKING_MODEL_GAPS,
)

ControlStatus = Literal[
    "physics_active",
    "benchmark_only",
    "warning_only",
    "future_not_implemented",
    "numerical_advanced",
    "derived_read_only",
]

ControlProvenance = Literal[
    "measured",
    "manufacturer_specification",
    "estimated",
    "diagnostic_placeholder",
    "unknown",
    "derived",
]

VALID_STATUSES: frozenset[str] = frozenset(
    {"physics_active", "benchmark_only", "warning_only",
     "future_not_implemented", "numerical_advanced", "derived_read_only"}
)
VALID_PROVENANCE: frozenset[str] = frozenset(
    {"measured", "manufacturer_specification", "estimated",
     "diagnostic_placeholder", "unknown", "derived"}
)
NON_MEASURED_PROVENANCE: frozenset[str] = frozenset({"unknown", "diagnostic_placeholder"})

GOVERNANCE_LOCKED_VALUES: dict[str, Any] = {
    "diagnostic_only": True,
    "final_export_allowed": False,
    "fourier_filter_physics_available": False,
}

DEMO_PROFILE_NAME = "cslm_physical_axicon_demo_profile"
DEMO_PROFILE_STATUS = "diagnostic_demo_not_measured_bench"
TEMPLATE_PROFILE_NAME = "cslm_physical_axicon_measured_bench_template"
TEMPLATE_PROFILE_STATUS = "measured_bench_template_unfilled"


@dataclass(frozen=True)
class EditableControl:
    control_id: str
    config_field: str           # "" if profile-only (no CSLMRouteConfig field)
    display_name: str
    group: str
    value: Any
    unit: str
    component_id: str
    physical_location: str
    status: ControlStatus
    provenance: ControlProvenance
    editable: bool
    affects_active_model: bool
    affects_benchmark_branch: bool
    required_for_measured_bench_mode: bool
    description: str
    minimum: float | None = None
    maximum: float | None = None
    allowed_values: tuple[Any, ...] = ()

    def as_row(self) -> dict[str, Any]:
        return {
            "control": self.control_id,
            "current value": self.value,
            "unit": self.unit,
            "component/location": f"{self.component_id} / {self.physical_location}",
            "status": self.status,
            "provenance": self.provenance,
            "editable": self.editable,
            "affects active model": self.affects_active_model,
            "affects benchmark": self.affects_benchmark_branch,
            "required for measured bench": self.required_for_measured_bench_mode,
        }


# ---------------------------------------------------------------------------
# Control specifications (metadata only; values pulled from CSLMRouteConfig)
# ---------------------------------------------------------------------------
# spec = (control_id, config_field, display, group, unit, component, location,
#         status, provenance, editable, affects_active, affects_bench,
#         req_measured, description, minimum, maximum, allowed)

_P, _B, _W, _F, _N, _D = (
    "physics_active", "benchmark_only", "warning_only",
    "future_not_implemented", "numerical_advanced", "derived_read_only",
)

_SPECS: tuple[tuple, ...] = (
    # 1. route and governance ------------------------------------------------
    ("route_mode", "route_mode", "Route mode", "route_and_governance", "", "route", "route_selector",
     _P, "diagnostic_placeholder", True, True, False, False,
     "Active route family selector.", None, None, ("holographic_cslm",)),
    ("order_handoff_mode", "order_handoff_mode", "Order handoff mode", "route_and_governance", "",
     "post_SLM2_branch", "post_SLM2_pre_4F_diagnostic_plane", _P, "diagnostic_placeholder",
     True, True, True, False,
     "none = active CSLM diagnostic route; ideal_selected_order_surrogate = opt-in benchmark branch.",
     None, None, ("none", "ideal_selected_order_surrogate", "physical_4f_filter")),
    ("physical_axicon_enabled_for_benchmark", "physical_axicon_enabled_for_benchmark",
     "Physical axicon enabled (benchmark)", "route_and_governance", "", "physical_axicon_benchmark_plane",
     "benchmark_branch", _B, "diagnostic_placeholder", True, False, True, False,
     "Whether the benchmark branch applies the physical axicon.", None, None, (True, False)),
    ("wavelength_nm", "wavelength_nm", "Wavelength", "route_and_governance", "nm", "source_field",
     "source", _P, "manufacturer_specification", True, True, True, True,
     "Laser wavelength used by all free-space propagation.", 200.0, 20000.0, ()),
    ("n_medium", "n_medium", "Propagation index n", "route_and_governance", "", "free_space",
     "all_segments", _P, "diagnostic_placeholder", True, True, True, False,
     "Free-space reference index; keep 1.0 for this stage (no material model).", 1.0, 1.0, ()),
    ("diagnostic_only", "", "diagnostic_only", "route_and_governance", "", "governance", "governance",
     _D, "derived", False, False, False, False,
     "Governance flag; locked True.", None, None, (True,)),
    ("final_export_allowed", "", "final_export_allowed", "route_and_governance", "", "governance",
     "governance", _D, "derived", False, False, False, False,
     "Governance flag; locked False.", None, None, (False,)),
    ("fourier_filter_physics_available", "", "fourier_filter_physics_available",
     "route_and_governance", "", "4F_route", "Fourier_plane", _D, "derived", False, False, False, False,
     "Derived; physical 4F filter physics is not implemented -> always False.", None, None, (False,)),
    # 2. source and input beam ----------------------------------------------
    ("input_pulse_energy_uJ", "input_pulse_energy_uJ", "Input pulse energy", "source_and_input_beam",
     "uJ", "source_field", "source", _P, "estimated", True, True, True, True,
     "Pulse energy entering the route.", 0.0, None, ()),
    ("input_beam_radius_um", "input_beam_radius_um", "Input beam radius (1/e^2-ish)",
     "source_and_input_beam", "um", "source_field", "source", _P, "estimated", True, True, True, True,
     "Gaussian source radius.", 0.0, None, ()),
    ("input_beam_ellipticity", "", "Input beam ellipticity", "source_and_input_beam", "ratio",
     "source_field", "source", _F, "unknown", True, False, False, True,
     "Not applied by the current CSLM source (circular Gaussian only).", None, None, ()),
    ("input_beam_rotation_deg", "", "Input beam rotation", "source_and_input_beam", "deg",
     "source_field", "source", _F, "unknown", True, False, False, True,
     "Not applied by the current CSLM source.", None, None, ()),
    ("input_beam_centre_x_um", "", "Input beam centre x", "source_and_input_beam", "um",
     "source_field", "source", _F, "unknown", True, False, False, True,
     "Beam decentre not modelled in the current CSLM source.", None, None, ()),
    ("input_beam_centre_y_um", "", "Input beam centre y", "source_and_input_beam", "um",
     "source_field", "source", _F, "unknown", True, False, False, True,
     "Beam decentre not modelled in the current CSLM source.", None, None, ()),
    ("input_aperture_radius_um", "", "Input aperture radius", "source_and_input_beam", "um",
     "input_conditioning_boundary", "input_conditioning_boundary", _F, "unknown", True, False, False, True,
     "Input aperture not modelled in the current CSLM source.", None, None, ()),
    # 3. SLM1 ----------------------------------------------------------------
    ("slm1_phase_mode", "slm1_phase_mode", "SLM1 phase mode", "slm1", "", "SLM1_phase_plane",
     "SLM1_phase_plane", _P, "diagnostic_placeholder", True, True, True, False,
     "SLM1 phase pattern family.", None, None, ("vortex", "flat", "zero", "linear_ramp")),
    ("slm1_topological_charge", "slm1_topological_charge", "SLM1 topological charge", "slm1", "",
     "SLM1_phase_plane", "SLM1_phase_plane", _P, "estimated", True, True, True, False,
     "Vortex charge l owned by SLM1.", None, None, ()),
    ("slm1_linear_ramp_cpm", "slm1_linear_ramp_cpm", "SLM1 linear ramp", "slm1", "cycles/m",
     "SLM1_phase_plane", "SLM1_phase_plane", _P, "estimated", True, True, True, False,
     "Optional SLM1 linear ramp frequency.", None, None, ()),
    ("slm1_to_slm2_distance_mm", "slm1_to_slm2_distance_mm", "SLM1->SLM2 distance", "slm1", "mm",
     "SLM1_to_SLM2_segment", "SLM1_to_SLM2_segment", _P, "diagnostic_placeholder", True, True, True, True,
     "Free-space distance from SLM1 to SLM2 (active propagation).", 0.0, None, ()),
    ("slm1_pixel_pitch_um", "", "SLM1 pixel pitch", "slm1", "um", "SLM1_phase_plane", "SLM1_phase_plane",
     _W, "unknown", True, False, False, True,
     "Hardware record; not consumed by the continuous-field model yet.", None, None, ()),
    ("slm1_active_area_mm", "", "SLM1 active area", "slm1", "mm", "SLM1_phase_plane", "SLM1_phase_plane",
     _W, "unknown", True, False, False, True,
     "Hardware record; not consumed yet.", None, None, ()),
    ("slm1_resolution_px", "", "SLM1 resolution", "slm1", "px", "SLM1_phase_plane", "SLM1_phase_plane",
     _W, "unknown", True, False, False, True,
     "Hardware record; not consumed yet.", None, None, ()),
    ("slm1_fill_factor", "", "SLM1 fill factor", "slm1", "ratio", "SLM1_phase_plane", "SLM1_phase_plane",
     _W, "unknown", True, False, False, True,
     "Hardware record; not consumed yet.", None, None, ()),
    ("slm1_phase_calibration_status", "", "SLM1 phase calibration status", "slm1", "", "SLM1_phase_plane",
     "SLM1_phase_plane", _W, "unknown", True, False, False, True,
     "Hardware record; phase response calibration state.", None, None,
     ("unknown", "uncalibrated", "factory", "measured")),
    # 4. SLM2 ----------------------------------------------------------------
    ("slm2_conjugate_mode", "slm2_conjugate_mode", "SLM2 conjugate mode", "slm2", "", "SLM2_phase_plane",
     "SLM2_phase_plane", _P, "diagnostic_placeholder", True, True, True, False,
     "SLM2 vortex-handling mode (preserves SLM1 vortex).", None, None, ("preserve_vortex", "conjugate")),
    ("slm2_to_pre_4f_diagnostic_distance_mm", "slm2_to_pre_4f_diagnostic_distance_mm",
     "SLM2->pre-4F diagnostic distance", "slm2", "mm", "SLM2_to_pre_4F_diagnostic_segment",
     "SLM2_to_pre_4F_diagnostic_segment", _P, "diagnostic_placeholder", True, True, True, True,
     "Active free-space distance to the post-SLM2 pre-4F diagnostic plane.", 0.0, None, ()),
    ("slm2_carrier_frequency_cpm", "slm2_carrier_frequency_cpm", "SLM2 carrier frequency", "slm2",
     "cycles/m", "SLM2_phase_plane", "SLM2_phase_plane", _P, "estimated", True, True, True, False,
     "SLM2 blaze/carrier spatial frequency.", None, None, ()),
    ("slm2_correction_phase_rad", "slm2_correction_phase_rad", "SLM2 uniform piston placeholder",
     "slm2", "rad", "SLM2_phase_plane", "SLM2_phase_plane", _P, "diagnostic_placeholder", True, True, True, False,
     "Uniform piston placeholder (scalar). NOT an aberration-correction map; a true correction "
     "needs a spatial phase map (future).", None, None, ()),
    ("slm_phase_quantisation_levels", "slm_phase_quantisation_levels", "SLM phase quantisation levels",
     "slm2", "levels", "SLM2_phase_plane", "SLM2_phase_plane", _P, "estimated", True, True, True, False,
     "Phase quantisation levels applied before propagation.", 1, None, ()),
    ("slm2_active_area_mm", "", "SLM2 active area", "slm2", "mm", "SLM2_phase_plane", "SLM2_phase_plane",
     _W, "unknown", True, False, False, True, "Hardware record; not consumed yet.", None, None, ()),
    ("slm2_resolution_px", "", "SLM2 resolution", "slm2", "px", "SLM2_phase_plane", "SLM2_phase_plane",
     _W, "unknown", True, False, False, True, "Hardware record; not consumed yet.", None, None, ()),
    ("slm2_pixel_pitch_um", "", "SLM2 pixel pitch", "slm2", "um", "SLM2_phase_plane", "SLM2_phase_plane",
     _W, "unknown", True, False, False, True, "Hardware record; not consumed yet.", None, None, ()),
    ("slm2_fill_factor", "", "SLM2 fill factor", "slm2", "ratio", "SLM2_phase_plane", "SLM2_phase_plane",
     _W, "unknown", True, False, False, True, "Hardware record; not consumed yet.", None, None, ()),
    ("slm2_phase_calibration_status", "", "SLM2 phase calibration status", "slm2", "",
     "SLM2_phase_plane", "SLM2_phase_plane", _W, "unknown", True, False, False, True,
     "Hardware record; phase response calibration state.", None, None,
     ("unknown", "uncalibrated", "factory", "measured")),
    ("spatial_correction_map_source", "", "Spatial correction map source", "slm2", "",
     "SLM2_phase_plane", "SLM2_phase_plane", _F, "unknown", True, False, False, False,
     "Future spatial SLM2 correction map source. No effect on the field in this stage.",
     None, None, ("none", "Zernike coefficients", "imported phase array", "calibrated correction mask")),
    # 5. inter-SLM and 4F geometry (warning-only hardware records) -----------
    ("slm2_to_lens1_distance_mm", "slm2_to_lens1_distance_mm", "SLM2->lens1 distance",
     "inter_slm_and_4f_geometry", "mm", "SLM2_to_fourier_lens_segment", "SLM2_to_fourier_lens_segment",
     _W, "diagnostic_placeholder", True, False, False, True,
     "4F hardware record; does not affect the active field (no 4F physics).", 0.0, None, ()),
    ("fourier_lens1_focal_length_mm", "fourier_lens1_focal_length_mm", "Lens 1 focal length",
     "inter_slm_and_4f_geometry", "mm", "Fourier_lens_1", "Fourier_lens_1", _W, "diagnostic_placeholder",
     True, False, False, True, "4F hardware record; no active effect.", 0.0, None, ()),
    ("fourier_lens1_clear_aperture_mm", "", "Lens 1 clear aperture", "inter_slm_and_4f_geometry", "mm",
     "Fourier_lens_1", "Fourier_lens_1", _W, "unknown", True, False, False, True,
     "4F hardware record; no active effect.", 0.0, None, ()),
    ("lens1_to_fourier_plane_distance_mm", "lens1_to_fourier_plane_distance_mm",
     "Lens1->Fourier plane distance", "inter_slm_and_4f_geometry", "mm", "Fourier_plane", "Fourier_plane",
     _W, "diagnostic_placeholder", True, False, False, True, "4F hardware record; no active effect.",
     0.0, None, ()),
    ("fourier_filter_centre_x_um", "fourier_filter_centre_x_um", "Fourier stop centre x",
     "inter_slm_and_4f_geometry", "um", "plus_one_order_filter", "Fourier_plane", _W, "diagnostic_placeholder",
     True, False, False, True, "4F stop record; no active effect.", None, None, ()),
    ("fourier_filter_centre_y_um", "fourier_filter_centre_y_um", "Fourier stop centre y",
     "inter_slm_and_4f_geometry", "um", "plus_one_order_filter", "Fourier_plane", _W, "diagnostic_placeholder",
     True, False, False, True, "4F stop record; no active effect.", None, None, ()),
    ("fourier_filter_radius_um", "fourier_filter_radius_um", "Fourier stop radius",
     "inter_slm_and_4f_geometry", "um", "plus_one_order_filter", "Fourier_plane", _W, "diagnostic_placeholder",
     True, False, False, True, "4F stop record; no active effect.", 0.0, None, ()),
    ("fourier_filter_shape", "fourier_filter_shape", "Fourier stop shape", "inter_slm_and_4f_geometry",
     "", "plus_one_order_filter", "Fourier_plane", _W, "diagnostic_placeholder", True, False, False, True,
     "4F stop record; no active effect.", None, None, ("circular", "rectangular")),
    ("fourier_plane_to_lens2_distance_mm", "fourier_plane_to_lens2_distance_mm",
     "Fourier plane->lens2 distance", "inter_slm_and_4f_geometry", "mm", "Fourier_lens_2", "Fourier_lens_2",
     _W, "diagnostic_placeholder", True, False, False, True, "4F hardware record; no active effect.",
     0.0, None, ()),
    ("fourier_lens2_focal_length_mm", "fourier_lens2_focal_length_mm", "Lens 2 focal length",
     "inter_slm_and_4f_geometry", "mm", "Fourier_lens_2", "Fourier_lens_2", _W, "diagnostic_placeholder",
     True, False, False, True, "4F hardware record; no active effect.", 0.0, None, ()),
    ("fourier_lens2_clear_aperture_mm", "", "Lens 2 clear aperture", "inter_slm_and_4f_geometry", "mm",
     "Fourier_lens_2", "Fourier_lens_2", _W, "unknown", True, False, False, True,
     "4F hardware record; no active effect.", 0.0, None, ()),
    ("lens2_to_output_plane_distance_mm", "lens2_to_output_plane_distance_mm",
     "Lens2->output plane distance", "inter_slm_and_4f_geometry", "mm", "4F_output_plane", "4F_output_plane",
     _W, "diagnostic_placeholder", True, False, False, True, "4F hardware record; no active effect.",
     0.0, None, ()),
    # 6. physical axicon benchmark ------------------------------------------
    ("physical_axicon_cone_parameter_rad_per_um", "physical_axicon_cone_parameter_rad_per_um",
     "Axicon cone parameter", "physical_axicon_benchmark", "rad/um", "physical_axicon_benchmark_plane",
     "physical_axicon_benchmark_plane", _B, "estimated", True, False, True, True,
     "Physical axicon scalar cone parameter (benchmark branch only).", 0.0, None, ()),
    ("physical_axicon_clear_aperture_radius_um", "physical_axicon_clear_aperture_radius_um",
     "Axicon clear aperture radius", "physical_axicon_benchmark", "um", "physical_axicon_benchmark_plane",
     "physical_axicon_benchmark_plane", _B, "diagnostic_placeholder", True, False, True, True,
     "Axicon clear aperture (benchmark branch only).", 0.0, None, ()),
    ("physical_axicon_centre_x_um", "physical_axicon_centre_x_um", "Axicon centre x",
     "physical_axicon_benchmark", "um", "physical_axicon_benchmark_plane", "physical_axicon_benchmark_plane",
     _B, "diagnostic_placeholder", True, False, True, True, "Axicon centre (benchmark only).", None, None, ()),
    ("physical_axicon_centre_y_um", "physical_axicon_centre_y_um", "Axicon centre y",
     "physical_axicon_benchmark", "um", "physical_axicon_benchmark_plane", "physical_axicon_benchmark_plane",
     _B, "diagnostic_placeholder", True, False, True, True, "Axicon centre (benchmark only).", None, None, ()),
    ("physical_axicon_to_benchmark_reference_distance_mm", "physical_axicon_to_benchmark_reference_distance_mm",
     "Axicon->benchmark reference distance", "physical_axicon_benchmark", "mm", "post_axicon_benchmark_segment",
     "post_axicon_benchmark_segment", _B, "diagnostic_placeholder", True, False, True, True,
     "Free-space distance to the benchmark reference plane (benchmark only).", 0.0, None, ()),
    ("physical_axicon_axial_offset_um", "", "Axicon axial offset", "physical_axicon_benchmark", "um",
     "physical_axicon_benchmark_plane", "physical_axicon_benchmark_plane", _F, "unknown", True, False, False, True,
     "Mechanical axicon axial offset; not modelled yet.", None, None, ()),
    ("physical_axicon_mechanical_tip_tilt_mrad", "", "Axicon mechanical tip/tilt",
     "physical_axicon_benchmark", "mrad", "physical_axicon_benchmark_plane", "physical_axicon_benchmark_plane",
     _F, "unknown", True, False, False, True, "Mechanical axicon tip/tilt; not modelled yet.", None, None, ()),
    # 7. camera / reference-plane record (warning-only) ----------------------
    ("camera_model", "", "Camera model", "camera_reference_plane", "", "camera_plane", "camera_plane",
     _W, "unknown", True, False, False, True, "Future experiment-comparison record; no camera physics.",
     None, None, ()),
    ("camera_pixel_pitch_um", "", "Camera pixel pitch", "camera_reference_plane", "um", "camera_plane",
     "camera_plane", _W, "unknown", True, False, False, True, "Camera record; no camera physics.", 0.0, None, ()),
    ("camera_sensor_resolution_px", "", "Camera sensor resolution", "camera_reference_plane", "px",
     "camera_plane", "camera_plane", _W, "unknown", True, False, False, True,
     "Camera record; no camera physics.", None, None, ()),
    ("camera_magnification", "", "Camera magnification", "camera_reference_plane", "x", "camera_plane",
     "camera_plane", _W, "unknown", True, False, False, True, "Camera record; no camera physics.", None, None, ()),
    ("camera_plane_location_mm", "", "Camera-plane location", "camera_reference_plane", "mm",
     "camera_plane", "camera_plane", _W, "unknown", True, False, False, True,
     "Camera record; no camera physics.", None, None, ()),
    ("reference_plane_definition", "", "Reference-plane definition", "camera_reference_plane", "",
     "reference_plane", "reference_plane", _W, "diagnostic_placeholder", True, False, False, True,
     "Definition of the reference plane for future experimental comparison.", None, None, ()),
    ("camera_calibration_status", "", "Camera calibration status", "camera_reference_plane", "",
     "camera_plane", "camera_plane", _W, "unknown", True, False, False, True,
     "Camera calibration state record.", None, None, ("unknown", "uncalibrated", "measured")),
    # 8. advanced numerical --------------------------------------------------
    ("grid_N", "grid_N", "Grid N", "advanced_numerical", "px", "numerical", "numerical_grid",
     _N, "derived", True, True, True, False, "Transverse grid size (not laboratory hardware).", 16, None, ()),
    ("dx_um", "dx_um", "Grid pitch dx", "advanced_numerical", "um", "numerical", "numerical_grid",
     _N, "derived", True, True, True, False, "Transverse sample pitch (not laboratory hardware).", 0.0, None, ()),
    ("n_z", "n_z", "Number of z planes", "advanced_numerical", "planes", "numerical", "numerical_grid",
     _N, "derived", True, True, True, False, "Number of propagation planes (not laboratory hardware).", 2, None, ()),
    ("z_max_um", "z_max_um", "z range max", "advanced_numerical", "um", "numerical", "numerical_grid",
     _N, "derived", True, True, False, False, "Active-route propagation z range (not laboratory hardware).",
     0.0, None, ()),
    ("bandlimit", "bandlimit", "Band-limit propagator", "advanced_numerical", "", "numerical",
     "numerical_grid", _N, "derived", True, True, True, False,
     "Band-limited angular-spectrum flag (not laboratory hardware).", None, None, (True, False)),
)


def _spec_value(config: CSLMRouteConfig, config_field: str) -> Any:
    if config_field and hasattr(config, config_field):
        return getattr(config, config_field)
    return None


def build_cslm_editable_control_registry(
    config: CSLMRouteConfig | None = None,
    *,
    profile: Mapping[str, Any] | None = None,
) -> tuple[EditableControl, ...]:
    """Build the editable-control registry mapped onto an existing config.

    Values come from ``config`` for mapped fields; profile-only controls take
    their value from ``profile`` (if given) else ``None``.  Governance flags use
    the locked governance values.
    """
    config = config or CSLMRouteConfig()
    prof_controls = dict((profile or {}).get("controls", {}))
    controls: list[EditableControl] = []
    for spec in _SPECS:
        (cid, field, display, group, unit, comp, loc, status, prov, editable,
         aam, abb, reqm, desc, mn, mx, allowed) = spec
        if cid in GOVERNANCE_LOCKED_VALUES:
            value: Any = GOVERNANCE_LOCKED_VALUES[cid]
        elif field:
            value = _spec_value(config, field)
        else:
            entry = prof_controls.get(cid)
            value = entry.get("value") if isinstance(entry, Mapping) else None
        prov_eff = prov
        entry = prof_controls.get(cid)
        if isinstance(entry, Mapping) and entry.get("provenance") in VALID_PROVENANCE:
            prov_eff = entry["provenance"]
        controls.append(EditableControl(
            control_id=cid, config_field=field, display_name=display, group=group,
            value=value, unit=unit, component_id=comp, physical_location=loc,
            status=status, provenance=prov_eff, editable=editable,
            affects_active_model=aam, affects_benchmark_branch=abb,
            required_for_measured_bench_mode=reqm, description=desc,
            minimum=mn, maximum=mx, allowed_values=tuple(allowed),
        ))
    return tuple(controls)


def editable_control_rows(
    registry: Sequence[EditableControl] | None = None,
    *,
    config: CSLMRouteConfig | None = None,
    group: str | None = None,
) -> list[dict[str, Any]]:
    """Display rows for an editable-control DataFrame."""
    registry = registry if registry is not None else build_cslm_editable_control_registry(config)
    return [c.as_row() for c in registry if group is None or c.group == group]


def apply_editable_control_overrides(
    config: CSLMRouteConfig,
    overrides: Mapping[str, Any],
) -> CSLMRouteConfig:
    """Apply control-id overrides to the real CSLMRouteConfig fields.

    Profile-only controls (no config field) are accepted but do not change the
    model config.  Governance flags cannot be set to unsafe values, and there is
    no path to enable physical 4F filter physics.
    """
    registry = {c.control_id: c for c in build_cslm_editable_control_registry(config)}
    kwargs: dict[str, Any] = {}
    for cid, val in overrides.items():
        if cid not in registry:
            raise KeyError(f"unknown control_id: {cid!r}")
        ctrl = registry[cid]
        if cid in GOVERNANCE_LOCKED_VALUES:
            if val != GOVERNANCE_LOCKED_VALUES[cid]:
                raise ValueError(
                    f"{cid} is a locked governance flag and cannot be set to {val!r}."
                )
            continue
        if not ctrl.editable:
            raise ValueError(f"control {cid!r} is not editable.")
        if ctrl.allowed_values and val not in ctrl.allowed_values:
            raise ValueError(f"{cid}={val!r} not in allowed values {ctrl.allowed_values}.")
        if ctrl.config_field and hasattr(config, ctrl.config_field):
            kwargs[ctrl.config_field] = val
        # profile-only controls: recorded in the profile, not the model config
    return replace(config, **kwargs) if kwargs else config


# ---------------------------------------------------------------------------
# Hardware profiles
# ---------------------------------------------------------------------------


def build_default_demo_profile(config: CSLMRouteConfig | None = None) -> dict[str, Any]:
    """Diagnostic demo profile: mapped values from config, unknowns left null."""
    registry = build_cslm_editable_control_registry(config)
    controls: dict[str, Any] = {}
    for c in registry:
        controls[c.control_id] = {
            "value": c.value,
            "unit": c.unit,
            "status": c.status,
            "provenance": c.provenance,
            "config_field": c.config_field,
        }
    return {
        "profile_name": DEMO_PROFILE_NAME,
        "profile_status": DEMO_PROFILE_STATUS,
        "claim_boundary": "diagnostic demo geometry only; not measured laboratory geometry; "
                          "n=1.0 free-space; no material/4F/camera physics; final_export_allowed=False",
        "controls": controls,
    }


def build_measured_bench_template(config: CSLMRouteConfig | None = None) -> dict[str, Any]:
    """Blank measured-bench template: every real parameter null/unknown (no invented values)."""
    registry = build_cslm_editable_control_registry(config)
    controls: dict[str, Any] = {}
    for c in registry:
        # Derived/governance values are kept; everything else is blanked for real measurement.
        if c.status == "derived_read_only":
            value, provenance = c.value, "derived"
        else:
            value, provenance = None, "unknown"
        controls[c.control_id] = {
            "value": value,
            "unit": c.unit,
            "status": c.status,
            "provenance": provenance,
            "config_field": c.config_field,
        }
    return {
        "profile_name": TEMPLATE_PROFILE_NAME,
        "profile_status": TEMPLATE_PROFILE_STATUS,
        "claim_boundary": "blank measured-bench template; fill from real measurements only; "
                          "no invented values; n=1.0 free-space; final_export_allowed=False",
        "controls": controls,
    }


def validate_hardware_profile(profile: Mapping[str, Any]) -> list[str]:
    """Return a list of validation issues (empty == valid)."""
    issues: list[str] = []
    if "profile_name" not in profile or "profile_status" not in profile:
        issues.append("profile missing profile_name/profile_status")
    controls = profile.get("controls", {})
    if not isinstance(controls, Mapping):
        return issues + ["profile 'controls' must be a mapping"]
    is_template = profile.get("profile_name") == TEMPLATE_PROFILE_NAME
    for cid, entry in controls.items():
        if not isinstance(entry, Mapping):
            issues.append(f"{cid}: entry must be a mapping")
            continue
        prov = entry.get("provenance")
        if prov not in VALID_PROVENANCE:
            issues.append(f"{cid}: invalid provenance {prov!r}")
        status = entry.get("status")
        if status is not None and status not in VALID_STATUSES:
            issues.append(f"{cid}: invalid status {status!r}")
        # A measured-bench template must not contain invented (non-null) real values.
        if is_template and status != "derived_read_only":
            if entry.get("value") is not None or prov != "unknown":
                issues.append(f"{cid}: measured-bench template must be null/unknown, found "
                              f"value={entry.get('value')!r} provenance={prov!r}")
        # Never present unknown/placeholder as measured.
        if prov == "measured" and entry.get("value") is None:
            issues.append(f"{cid}: provenance 'measured' but value is null")
    return issues


def save_hardware_profile(profile: Mapping[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2, sort_keys=False), encoding="utf-8")
    return out


def load_hardware_profile(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def config_from_profile(
    profile: Mapping[str, Any],
    base_config: CSLMRouteConfig | None = None,
) -> CSLMRouteConfig:
    """Load mapped profile values into a CSLMRouteConfig (config fields only)."""
    base = base_config or CSLMRouteConfig()
    registry = {c.control_id: c for c in build_cslm_editable_control_registry(base)}
    overrides: dict[str, Any] = {}
    for cid, entry in profile.get("controls", {}).items():
        ctrl = registry.get(cid)
        if ctrl is None or not ctrl.config_field or cid in GOVERNANCE_LOCKED_VALUES:
            continue
        value = entry.get("value") if isinstance(entry, Mapping) else None
        if value is not None:
            overrides[cid] = value
    return apply_editable_control_overrides(base, overrides)


# ---------------------------------------------------------------------------
# Completeness report
# ---------------------------------------------------------------------------


def hardware_profile_completeness_report(
    profile: Mapping[str, Any] | None = None,
    *,
    config: CSLMRouteConfig | None = None,
) -> dict[str, Any]:
    """Distinguish active-model / benchmark / physical-4F / measured-bench / camera readiness."""
    registry = build_cslm_editable_control_registry(config, profile=profile)
    by_id = {c.control_id: c for c in registry}

    def _value(cid: str) -> Any:
        return by_id[cid].value if cid in by_id else None

    active = [c for c in registry if c.status == "physics_active"]
    bench = [c for c in registry if c.status == "benchmark_only"]
    four_f = [c for c in registry if c.status == "warning_only"
              and c.group == "inter_slm_and_4f_geometry"]
    camera = [c for c in registry if c.group == "camera_reference_plane"]
    measured_required = [c for c in registry if c.required_for_measured_bench_mode]

    active_missing = [c.control_id for c in active if c.value is None]
    bench_missing = [c.control_id for c in bench if c.value is None]
    # Physical 4F is blocked by the model gap regardless of values.
    four_f_missing = [c.control_id for c in four_f
                      if c.value is None or c.provenance in NON_MEASURED_PROVENANCE]
    measured_missing = [c.control_id for c in measured_required
                        if c.provenance != "measured" or c.value is None]
    camera_missing = [c.control_id for c in camera
                      if c.value is None or c.provenance != "measured"]

    return {
        "active_cslm_diagnostic_branch": "complete" if not active_missing else "incomplete",
        "ideal_axicon_benchmark_branch": "complete" if not bench_missing else "incomplete",
        "physical_4f_route": "blocked",
        "measured_lab_route": "blocked" if measured_missing else "complete",
        "camera_comparison": "blocked" if camera_missing else "complete",
        "active_branch_missing": active_missing,
        "benchmark_branch_missing": bench_missing,
        "physical_4f_blocking_model_gaps": list(FOUR_F_BLOCKING_MODEL_GAPS),
        "physical_4f_required_parameters": list(FOUR_F_REQUIRED_PARAMETERS),
        "physical_4f_missing_or_unmeasured": four_f_missing,
        "measured_bench_missing": measured_missing,
        "camera_comparison_missing": camera_missing,
        "fourier_filter_physics_available": False,
        "diagnostic_only": True,
        "final_export_allowed": False,
        "claim_boundary": "n=1.0 free-space optical/fluence diagnostic; no material/4F/camera physics",
    }


def status_counts(registry: Sequence[EditableControl] | None = None) -> dict[str, int]:
    registry = registry if registry is not None else build_cslm_editable_control_registry()
    counts = {s: 0 for s in VALID_STATUSES}
    for c in registry:
        counts[c.status] += 1
    return counts


def provenance_counts(registry: Sequence[EditableControl] | None = None) -> dict[str, int]:
    registry = registry if registry is not None else build_cslm_editable_control_registry()
    counts = {p: 0 for p in VALID_PROVENANCE}
    for c in registry:
        counts[c.provenance] += 1
    return counts


def plot_hardware_profile_completeness(
    profile: Mapping[str, Any] | None = None,
    *,
    config: CSLMRouteConfig | None = None,
    output_path: str | Path | None = None,
    dpi: int = 160,
):
    """Diagnostic-only figure: control status/provenance counts + readiness gate."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    registry = build_cslm_editable_control_registry(config, profile=profile)
    rep = hardware_profile_completeness_report(profile, config=config)
    sc = status_counts(registry)
    pc = provenance_counts(registry)

    fig = plt.figure(figsize=(15.0, 8.4), facecolor="white")
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.97, top=0.86, bottom=0.08, hspace=0.45, wspace=0.22)
    fig.suptitle("Stage 8C.3R.5.2 Editable Hardware / Geometry Control Contract\n"
                 "n=1.0 free-space; no material/4F/camera physics; final_export_allowed=False",
                 x=0.04, y=0.975, ha="left", va="top", fontsize=14, fontweight="bold")

    ax = fig.add_subplot(gs[0, 0])
    ks = list(sc); ax.bar(ks, [sc[k] for k in ks], color="#1565c0")
    ax.set_title("Controls by status", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=25, labelsize=8)
    for i, k in enumerate(ks):
        ax.text(i, sc[k], str(sc[k]), ha="center", va="bottom", fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    kp = list(pc); ax.bar(kp, [pc[k] for k in kp], color="#ef6c00")
    ax.set_title("Controls by provenance", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=25, labelsize=8)
    for i, k in enumerate(kp):
        ax.text(i, pc[k], str(pc[k]), ha="center", va="bottom", fontsize=8)

    ax = fig.add_subplot(gs[1, 0]); ax.set_axis_off()
    cats = [
        ("Active CSLM diagnostic branch", rep["active_cslm_diagnostic_branch"]),
        ("Ideal axicon benchmark branch", rep["ideal_axicon_benchmark_branch"]),
        ("Physical 4F route", rep["physical_4f_route"]),
        ("Measured lab route", rep["measured_lab_route"]),
        ("Camera comparison", rep["camera_comparison"]),
    ]
    colour = {"complete": "#1b5e20", "blocked": "#b71c1c", "incomplete": "#ef6c00"}
    ax.text(0.0, 1.0, "Readiness gate", fontsize=12, fontweight="bold", va="top")
    for i, (name, st) in enumerate(cats):
        ax.text(0.02, 0.84 - i * 0.16, name, fontsize=10, va="top")
        ax.text(0.78, 0.84 - i * 0.16, st.upper(), fontsize=10, fontweight="bold",
                va="top", color=colour.get(st, "#333"))

    ax = fig.add_subplot(gs[1, 1]); ax.set_axis_off()
    lines = [
        f"physical 4F blocked: missing/unmeasured = {len(rep['physical_4f_missing_or_unmeasured'])}",
        f"measured-bench missing = {len(rep['measured_bench_missing'])}",
        f"camera-comparison missing = {len(rep['camera_comparison_missing'])}",
        f"fourier_filter_physics_available = {rep['fourier_filter_physics_available']}",
        "",
        "physical 4F model gaps:",
    ]
    for g in rep["physical_4f_blocking_model_gaps"][:4]:
        lines.append("  - " + g[:62])
    lines += ["", "CLAIM: " + rep["claim_boundary"]]
    ax.text(0.0, 1.0, "\n".join(lines), fontsize=8.4, va="top", family="monospace")

    if output_path is not None:
        out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight", metadata={
            "Title": "Stage 8C.3R.5.2 hardware profile completeness",
            "final_export_allowed": "False"})
    return fig
