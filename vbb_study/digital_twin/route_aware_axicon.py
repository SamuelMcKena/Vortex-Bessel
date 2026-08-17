"""Stage 8C.3 route-aware physical-axicon beamline diagnostics.

This module keeps the Stage 8C.3R study in free space (``n = 1.0``) while
representing the physical-axicon route as an ordered component/segment chain.
Supported lab-realism errors belong to named components and are applied in the
local component plane, then propagated through all downstream represented
segments/elements.

Field-state controls are retained only as explicitly labelled boundary
conditions at named planes.  They are not treated as generic stand-ins for every
possible upstream hardware error.

All default distances are diagnostic demo geometry, not measured bench
distances.  Outputs are optical/fluence diagnostics only;
``final_export_allowed=False``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from scipy.ndimage import shift as nd_shift

import matplotlib
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402

from vbb_study.equations.fields import fft2c, make_xy_grid
from vbb_study.equations.propagation import make_bl_asm_propagator
from vbb_study.digital_twin.component_plane_states import (
    ComponentPlaneState,
    PropagatedFieldStack,
    field_power,
)
from vbb_study.digital_twin.annular_axis_tracking import estimate_annular_axis
from vbb_study.digital_twin.component_plane_metrics import (
    DIAGNOSTIC_SWEEP_LABEL,
    classify_translation_vs_deformation,
    compute_axis_tracking,
    compute_energy_throughput,
    stack_to_fluence,
)

_UM = 1e-6
_MM = 1e-3

FINAL_EXPORT_ALLOWED = False
MODEL_STATUS = "optical_prediction"
DIAGNOSTIC_GEOMETRY_NOTE = (
    "diagnostic demo geometry only; no actual laboratory bench distance is assumed"
)

PHYSICAL_LOCATIONS: tuple[str, ...] = (
    "source_plane",
    "after_beam_conditioning",
    "before_SLM1",
    "after_SLM1",
    "before_SLM2",
    "after_SLM2",
    "before_physical_axicon",
    "physical_axicon_plane",
    "after_physical_axicon",
    "post_axicon_diagnostic_boundary",
    "fourier_plane",
    "after_fourier_filter",
    "relay_plane",
    "objective_pupil_plane",
    "after_objective",
    "free_space_reference_plane",
    "diagnostic_only",
    "not_represented_by_current_engine",
)

REPRESENTED_PHYSICAL_AXICON_LOCATIONS: tuple[str, ...] = (
    "source_plane",
    "after_beam_conditioning",
    "before_physical_axicon",
    "physical_axicon_plane",
    "after_physical_axicon",
    "post_axicon_diagnostic_boundary",
    "free_space_reference_plane",
)

EXECUTED_PHYSICAL_AXICON_COMPONENT_IDS: tuple[str, ...] = (
    "source_field",
    "source_boundary_condition",
    "input_aperture",
    "source_to_physical_axicon",
    "physical_axicon_input_boundary",
    "physical_axicon",
    "after_physical_axicon_boundary",
    "post_axicon_free_space_segment",
    "post_axicon_diagnostic_boundary",
    "post_axicon_to_reference_segment",
    "reference_plane",
)


@dataclass(frozen=True)
class RouteGraphNode:
    name: str
    kind: str  # plane | propagation_segment | element | reference
    physical_location: str
    distance_mm: float = 0.0
    represented: bool = True
    note: str = ""


@dataclass(frozen=True)
class RoutePerturbationRecord:
    parameter_name: str
    perturbation_type: str
    magnitude: float | str
    units: str
    enabled: bool
    injection_location: str
    route: str
    implementation_plane: str
    downstream_elements_affected: tuple[str, ...]
    status: str
    note: str = ""
    component_id: str = ""
    boundary_plane: str = ""
    physical_approximation: str = ""
    upstream_hardware_error_could_emulate: str = ""
    downstream_components_consume: tuple[str, ...] = ()

    @property
    def physical_location(self) -> str:
        return self.injection_location

    @property
    def classification(self) -> str:
        return self.status

    def as_dict(self) -> dict[str, Any]:
        return {
            "parameter_name": self.parameter_name,
            "perturbation_type": self.perturbation_type,
            "magnitude": self.magnitude,
            "units": self.units,
            "enabled": bool(self.enabled),
            "injection_location": self.injection_location,
            "route": self.route,
            "implementation_plane": self.implementation_plane,
            "downstream_elements_affected": list(self.downstream_elements_affected),
            "active / warning-only / future status": self.status,
            "classification": self.status,
            "note": self.note,
            "component_id": self.component_id,
            "boundary_plane": self.boundary_plane,
            "physical_approximation": self.physical_approximation,
            "upstream_hardware_error_could_emulate": self.upstream_hardware_error_could_emulate,
            "downstream_components_consume": list(self.downstream_components_consume),
        }


@dataclass(frozen=True)
class ComponentPose:
    """Physical pose error for a represented component."""

    decentre_x_um: float = 0.0
    decentre_y_um: float = 0.0
    axial_offset_um: float = 0.0
    tip_x_mrad: float = 0.0
    tip_y_mrad: float = 0.0
    roll_deg: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "decentre_x_um": float(self.decentre_x_um),
            "decentre_y_um": float(self.decentre_y_um),
            "axial_offset_um": float(self.axial_offset_um),
            "tip_x_mrad": float(self.tip_x_mrad),
            "tip_y_mrad": float(self.tip_y_mrad),
            "roll_deg": float(self.roll_deg),
        }


@dataclass(frozen=True)
class BeamlineComponent:
    """One ordered component or propagation segment in the diagnostic route."""

    component_id: str
    component_type: str
    physical_location: str
    nominal_z_position_um: float
    distance_from_previous_component_mm: float
    distance_to_next_element_mm: float
    enabled: bool
    physical_pose: ComponentPose = field(default_factory=ComponentPose)
    component_specific_parameters: Mapping[str, Any] = field(default_factory=dict)
    clear_aperture: Mapping[str, Any] = field(default_factory=dict)
    status: str = "physics_active"
    represented_by_current_engine: bool = True
    physical_model_available: bool = True
    misalignment_modes_currently_supported: tuple[str, ...] = ()
    downstream_elements_affected: tuple[str, ...] = ()
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "component_type": self.component_type,
            "physical_location": self.physical_location,
            "nominal_z_position_um": float(self.nominal_z_position_um),
            "distance_from_previous_component_mm": float(self.distance_from_previous_component_mm),
            "distance_to_next_element_mm": float(self.distance_to_next_element_mm),
            "enabled": bool(self.enabled),
            "physical_pose": self.physical_pose.as_dict(),
            "component_specific_parameters": dict(self.component_specific_parameters),
            "clear_aperture": dict(self.clear_aperture),
            "active / warning-only / future status": self.status,
            "represented_by_current_engine": bool(self.represented_by_current_engine),
            "physical_model_available": bool(self.physical_model_available),
            "misalignment_modes_currently_supported": list(self.misalignment_modes_currently_supported),
            "downstream_elements_affected": list(self.downstream_elements_affected),
            "note": self.note,
        }


@dataclass(frozen=True)
class RouteInspectionRecord:
    """Per-component incoming/outgoing diagnostic state for the route view."""

    component_id: str
    component_name: str
    component_type: str
    nominal_location_um: float
    distance_from_previous_component_mm: float
    distance_to_next_element_mm: float
    actual_pose_error: Mapping[str, float]
    incoming_field_metrics: Mapping[str, float]
    outgoing_field_metrics: Mapping[str, float]
    energy_before_uJ: float
    energy_after_uJ: float
    centroid_before_um: tuple[float, float]
    centroid_after_um: tuple[float, float]
    angle_before_mrad: tuple[float, float]
    angle_after_mrad: tuple[float, float]
    aperture_overlap: float | None
    downstream_consequences: tuple[str, ...]
    model_status: str
    transform_applied: bool
    warnings: tuple[str, ...] = ()
    represented_by_current_engine: bool = True
    physical_model_available: bool = True
    misalignment_modes_currently_supported: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "component name": self.component_name,
            "component identifier": self.component_id,
            "component_id": self.component_id,
            "component_type": self.component_type,
            "nominal location (um)": float(self.nominal_location_um),
            "distance_from_previous_component_mm": float(self.distance_from_previous_component_mm),
            "distance_to_next_element_mm": float(self.distance_to_next_element_mm),
            "actual segment distance (mm)": float(self.distance_to_next_element_mm),
            "actual_pose_error": dict(self.actual_pose_error),
            "incoming_field_metrics": dict(self.incoming_field_metrics),
            "outgoing_field_metrics": dict(self.outgoing_field_metrics),
            "energy before (uJ)": float(self.energy_before_uJ),
            "energy after (uJ)": float(self.energy_after_uJ),
            "centroid before (um)": tuple(float(v) for v in self.centroid_before_um),
            "centroid after (um)": tuple(float(v) for v in self.centroid_after_um),
            "angle before (mrad)": tuple(float(v) for v in self.angle_before_mrad),
            "angle after (mrad)": tuple(float(v) for v in self.angle_after_mrad),
            "aperture overlap": None if self.aperture_overlap is None else float(self.aperture_overlap),
            "downstream consequences": list(self.downstream_consequences),
            "model status": self.model_status,
            "transform_applied": bool(self.transform_applied),
            "warnings": list(self.warnings),
            "represented_by_current_engine": bool(self.represented_by_current_engine),
            "physical_model_available": bool(self.physical_model_available),
            "misalignment_modes_currently_supported": list(self.misalignment_modes_currently_supported),
        }


@dataclass(frozen=True)
class RouteAwareAxiconConfig:
    """Editable route-aware physical-axicon diagnostic geometry."""

    route_mode: str = "physical_axicon"
    wavelength_nm: float = 1030.0
    n_medium: float = 1.0
    grid_N: int = 224
    dx_um: float = 0.5
    n_z: int = 36
    input_pulse_energy_uJ: float = 95.76

    # Diagnostic demo geometry; not actual bench distances.
    pre_axicon_distance_mm: float = 0.80
    post_axicon_distance_mm: float = 0.225
    post_axicon_free_space_distance_mm: float = 0.150
    post_axicon_diagnostic_to_reference_distance_mm: float = 0.075
    # Compatibility aliases from the first component-route draft. These are not
    # objective models and are normalised into the neutral post-axicon distances.
    axicon_to_objective_distance_mm: float = 0.150
    objective_to_reference_distance_mm: float = 0.075

    input_beam_radius_um: float = 22.0
    input_beam_ellipticity: float = 1.0
    input_beam_rotation_deg: float = 0.0
    input_aperture_radius_um: float = 48.0
    input_aperture_centre_x_um: float = 0.0
    input_aperture_centre_y_um: float = 0.0

    # Correct C3R.3 perturbation controls: type + magnitude + injection location.
    field_tilt_x_mrad: float = 0.0
    field_tilt_y_mrad: float = 0.0
    field_tilt_location: str = "source_plane"
    beam_decentre_x_um: float = 0.0
    beam_decentre_y_um: float = 0.0
    beam_decentre_location: str = "source_plane"

    # Compatibility aliases from the first C3R.3 draft / older C3R pipeline.
    # These are normalised into the generic location-aware controls.
    input_beam_decentre_x_um: float = 0.0
    input_beam_decentre_y_um: float = 0.0
    input_beam_tilt_x_mrad: float = 0.0
    input_beam_tilt_y_mrad: float = 0.0

    # Explicit decentre at the axicon input plane, distinct from source decentre.
    axicon_input_beam_decentre_x_um: float = 0.0
    axicon_input_beam_decentre_y_um: float = 0.0

    physical_axicon_centre_x_um: float = 0.0
    physical_axicon_centre_y_um: float = 0.0
    physical_axicon_axial_offset_um: float = 0.0
    physical_axicon_clear_aperture_radius_um: float = 42.0
    axicon_cone_parameter: float = 1.05  # rad / um
    physical_axicon_mechanical_tilt_x_mrad: float = 0.0
    physical_axicon_mechanical_tilt_y_mrad: float = 0.0

    enable_post_axicon_steering_test: bool = False
    post_axicon_steering_x_mrad: float = 0.0
    post_axicon_steering_y_mrad: float = 0.0
    bandlimit: bool = True

    @classmethod
    def fast(cls, **overrides: Any) -> "RouteAwareAxiconConfig":
        base = dict(grid_N=160, dx_um=0.6, n_z=24, input_beam_radius_um=20.0)
        base.update(overrides)
        return cls(**base)

    @property
    def wavelength_m(self) -> float:
        return float(self.wavelength_nm) * 1e-9

    @property
    def dx_m(self) -> float:
        return float(self.dx_um) * _UM

    @property
    def k_medium_rad_per_m(self) -> float:
        return 2.0 * np.pi * float(self.n_medium) / self.wavelength_m

    @property
    def pre_axicon_distance_um(self) -> float:
        return float(self.pre_axicon_distance_mm) * 1000.0

    @property
    def post_axicon_distance_um(self) -> float:
        return float(self.post_axicon_distance_mm) * 1000.0

    @property
    def post_axicon_free_space_distance_um(self) -> float:
        return float(self.post_axicon_free_space_distance_mm) * 1000.0

    @property
    def post_axicon_diagnostic_to_reference_distance_um(self) -> float:
        return float(self.post_axicon_diagnostic_to_reference_distance_mm) * 1000.0

    @property
    def axicon_to_objective_distance_um(self) -> float:
        return float(self.post_axicon_free_space_distance_mm) * 1000.0

    @property
    def objective_to_reference_distance_um(self) -> float:
        return float(self.post_axicon_diagnostic_to_reference_distance_mm) * 1000.0


@dataclass(frozen=True)
class RouteAwareAxiconRun:
    config: RouteAwareAxiconConfig
    route_mode: str
    component_chain: tuple[BeamlineComponent, ...]
    source_state: ComponentPlaneState
    axicon_incident_state: ComponentPlaneState
    physical_axicon_state: ComponentPlaneState
    reference_plane_state: ComponentPlaneState
    propagated_stack: PropagatedFieldStack
    axicon_incidence_metrics: Mapping[str, Any]
    route_inspection_records: tuple[RouteInspectionRecord, ...]
    perturbation_records: tuple[RoutePerturbationRecord, ...]
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    final_export_allowed: bool = FINAL_EXPORT_ALLOWED
    model_status: str = MODEL_STATUS


def _with_overrides(config: RouteAwareAxiconConfig, controls: Mapping[str, Any] | None) -> RouteAwareAxiconConfig:
    controls = dict(controls or {})
    if not controls:
        return _normalise_location_controls(config, {})
    valid = {f.name for f in RouteAwareAxiconConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    updates = {k: v for k, v in controls.items() if k in valid}
    return _normalise_location_controls(replace(config, **updates), controls)


def _valid_location(location: str) -> str:
    loc = str(location)
    if loc == "pre_physical_axicon":
        loc = "before_physical_axicon"
    if loc == "post_physical_axicon":
        loc = "after_physical_axicon"
    if loc == "after_objective":
        loc = "post_axicon_diagnostic_boundary"
    if loc == "post_objective_reference_plane":
        loc = "free_space_reference_plane"
    if loc not in PHYSICAL_LOCATIONS:
        raise ValueError(f"Unsupported perturbation injection location: {location!r}.")
    return loc


def _normalise_location_controls(
    config: RouteAwareAxiconConfig,
    controls: Mapping[str, Any],
) -> RouteAwareAxiconConfig:
    updates: dict[str, Any] = {
        "field_tilt_location": _valid_location(config.field_tilt_location),
        "beam_decentre_location": _valid_location(config.beam_decentre_location),
    }
    # Keep the editable total post-axicon distance coherent with the represented
    # downstream split unless the caller explicitly supplied segment distances.
    neutral_segment_supplied = (
        "post_axicon_free_space_distance_mm" in controls
        or "post_axicon_diagnostic_to_reference_distance_mm" in controls
    )
    legacy_segment_supplied = (
        "axicon_to_objective_distance_mm" in controls
        or "objective_to_reference_distance_mm" in controls
    )
    if "post_axicon_distance_mm" in controls and not neutral_segment_supplied and not legacy_segment_supplied:
        post = float(config.post_axicon_distance_mm)
        updates["post_axicon_free_space_distance_mm"] = 2.0 * post / 3.0
        updates["post_axicon_diagnostic_to_reference_distance_mm"] = post / 3.0
    else:
        if legacy_segment_supplied and not neutral_segment_supplied:
            updates["post_axicon_free_space_distance_mm"] = float(config.axicon_to_objective_distance_mm)
            updates["post_axicon_diagnostic_to_reference_distance_mm"] = float(config.objective_to_reference_distance_mm)
        post = (
            float(updates.get("post_axicon_free_space_distance_mm", config.post_axicon_free_space_distance_mm))
            + float(updates.get(
                "post_axicon_diagnostic_to_reference_distance_mm",
                config.post_axicon_diagnostic_to_reference_distance_mm,
            ))
        )
        updates["post_axicon_distance_mm"] = post
    updates["axicon_to_objective_distance_mm"] = float(
        updates.get("post_axicon_free_space_distance_mm", config.post_axicon_free_space_distance_mm)
    )
    updates["objective_to_reference_distance_mm"] = float(
        updates.get(
            "post_axicon_diagnostic_to_reference_distance_mm",
            config.post_axicon_diagnostic_to_reference_distance_mm,
        )
    )

    generic_tilt_supplied = (
        "field_tilt_x_mrad" in controls or "field_tilt_y_mrad" in controls or "field_tilt_location" in controls
    )
    if not generic_tilt_supplied:
        if bool(config.enable_post_axicon_steering_test):
            updates["field_tilt_x_mrad"] = float(config.post_axicon_steering_x_mrad)
            updates["field_tilt_y_mrad"] = float(config.post_axicon_steering_y_mrad)
            updates["field_tilt_location"] = "after_physical_axicon"
        elif abs(config.input_beam_tilt_x_mrad) > 0 or abs(config.input_beam_tilt_y_mrad) > 0:
            updates["field_tilt_x_mrad"] = float(config.input_beam_tilt_x_mrad)
            updates["field_tilt_y_mrad"] = float(config.input_beam_tilt_y_mrad)
            updates["field_tilt_location"] = "source_plane"

    generic_decentre_supplied = (
        "beam_decentre_x_um" in controls or "beam_decentre_y_um" in controls or "beam_decentre_location" in controls
    )
    if not generic_decentre_supplied:
        if abs(config.axicon_input_beam_decentre_x_um) > 0 or abs(config.axicon_input_beam_decentre_y_um) > 0:
            updates["beam_decentre_x_um"] = float(config.axicon_input_beam_decentre_x_um)
            updates["beam_decentre_y_um"] = float(config.axicon_input_beam_decentre_y_um)
            updates["beam_decentre_location"] = "before_physical_axicon"
        elif abs(config.input_beam_decentre_x_um) > 0 or abs(config.input_beam_decentre_y_um) > 0:
            updates["beam_decentre_x_um"] = float(config.input_beam_decentre_x_um)
            updates["beam_decentre_y_um"] = float(config.input_beam_decentre_y_um)
            updates["beam_decentre_location"] = "source_plane"
    updates["field_tilt_location"] = _valid_location(updates.get("field_tilt_location", config.field_tilt_location))
    updates["beam_decentre_location"] = _valid_location(updates.get("beam_decentre_location", config.beam_decentre_location))
    return replace(config, **updates)


def _grid(config: RouteAwareAxiconConfig) -> dict[str, Any]:
    return make_xy_grid(int(config.grid_N), float(config.dx_m))


def _xy_um(grid: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(grid["x"], float) / _UM
    return x, x.copy()


def physical_axicon_route_graph(config: RouteAwareAxiconConfig) -> tuple[RouteGraphNode, ...]:
    """Represented route graph for the physical-axicon diagnostic path."""
    source_to_axicon_mm, post_axicon_free_space_mm, diagnostic_to_ref_mm = _actual_segment_distances_mm(config)
    return (
        RouteGraphNode("source field", "element", "source_plane", note="complex source/input field"),
        RouteGraphNode("source boundary condition", "field_state_boundary", "source_plane",
                       note="explicit boundary approximation, not hardware misalignment"),
        RouteGraphNode("input aperture", "element", "after_beam_conditioning",
                       note="field-transforming aperture component"),
        RouteGraphNode("source to axicon", "propagation_segment", "before_physical_axicon",
                       distance_mm=float(source_to_axicon_mm),
                       note=DIAGNOSTIC_GEOMETRY_NOTE),
        RouteGraphNode("axicon input boundary condition", "field_state_boundary", "before_physical_axicon",
                       note="explicit boundary approximation consumed by downstream axicon"),
        RouteGraphNode("physical axicon", "element", "physical_axicon_plane",
                       note="fixed thin axicon phase and clear aperture; local pose controls apply here"),
        RouteGraphNode("after axicon boundary condition", "field_state_boundary", "after_physical_axicon",
                       note="post-axicon steering test boundary; not an upstream axicon fault"),
        RouteGraphNode("post-axicon free-space segment", "propagation_segment", "post_axicon_diagnostic_boundary",
                       distance_mm=float(post_axicon_free_space_mm),
                       note=DIAGNOSTIC_GEOMETRY_NOTE),
        RouteGraphNode("post-axicon diagnostic boundary", "field_state_boundary", "post_axicon_diagnostic_boundary",
                       note="downstream field-state diagnostic boundary; no objective model is present"),
        RouteGraphNode("post-axicon to reference segment", "propagation_segment", "free_space_reference_plane",
                       distance_mm=float(diagnostic_to_ref_mm),
                       note=DIAGNOSTIC_GEOMETRY_NOTE),
        RouteGraphNode("free-space reference", "reference", "free_space_reference_plane",
                       note="free-space reference plane in air; no material model"),
    )


def _route_location_order() -> list[str]:
    return list(REPRESENTED_PHYSICAL_AXICON_LOCATIONS)


def _downstream_elements(location: str) -> tuple[str, ...]:
    loc = _valid_location(location)
    order = _route_location_order()
    if loc not in order:
        return ()
    downstream = order[order.index(loc) + 1:]
    elements = []
    for item in downstream:
        if item in {
            "physical_axicon_plane",
            "after_physical_axicon",
            "post_axicon_diagnostic_boundary",
            "free_space_reference_plane",
        }:
            elements.append(item)
    return tuple(elements)


def _downstream_component_ids(component_id: str) -> tuple[str, ...]:
    if component_id not in EXECUTED_PHYSICAL_AXICON_COMPONENT_IDS:
        return ()
    i = EXECUTED_PHYSICAL_AXICON_COMPONENT_IDS.index(component_id)
    return EXECUTED_PHYSICAL_AXICON_COMPONENT_IDS[i + 1:]


def _boundary_component_id_for_location(location: str) -> str:
    loc = _valid_location(location)
    return {
        "source_plane": "source_boundary_condition",
        "after_beam_conditioning": "source_boundary_condition",
        "before_physical_axicon": "physical_axicon_input_boundary",
        "after_physical_axicon": "after_physical_axicon_boundary",
        "post_axicon_diagnostic_boundary": "post_axicon_diagnostic_boundary",
        "free_space_reference_plane": "reference_plane",
    }.get(loc, "not_represented_by_current_engine")


def _actual_segment_distances_mm(config: RouteAwareAxiconConfig) -> tuple[float, float, float]:
    ax_shift_mm = float(config.physical_axicon_axial_offset_um) / 1000.0
    source_to_axicon = float(config.pre_axicon_distance_mm) + ax_shift_mm
    post_axicon_free_space = float(config.post_axicon_free_space_distance_mm) - ax_shift_mm
    diagnostic_to_reference = float(config.post_axicon_diagnostic_to_reference_distance_mm)
    return max(source_to_axicon, 0.0), max(post_axicon_free_space, 0.0), max(diagnostic_to_reference, 0.0)


def build_physical_axicon_beamline(config: RouteAwareAxiconConfig) -> tuple[BeamlineComponent, ...]:
    """Return the executed ordered component/segment chain for the physical-axicon route."""
    source_to_axicon_mm, post_axicon_free_space_mm, diagnostic_to_ref_mm = _actual_segment_distances_mm(config)
    nominal_axicon_z_um = float(config.pre_axicon_distance_um)
    actual_axicon_z_um = nominal_axicon_z_um + float(config.physical_axicon_axial_offset_um)
    nominal_diagnostic_z_um = nominal_axicon_z_um + float(config.post_axicon_free_space_distance_um)
    reference_z_um = nominal_diagnostic_z_um + float(config.post_axicon_diagnostic_to_reference_distance_um)

    def downstream(cid: str) -> tuple[str, ...]:
        return _downstream_component_ids(cid)

    return (
        BeamlineComponent(
            "source_field", "source", "source_plane", 0.0, 0.0, 0.0, True,
            component_specific_parameters={
                "beam_radius_um": float(config.input_beam_radius_um),
                "ellipticity": float(config.input_beam_ellipticity),
                "rotation_deg": float(config.input_beam_rotation_deg),
            },
            misalignment_modes_currently_supported=("source_field_boundary_shape",),
            downstream_elements_affected=downstream("source_field"),
            note="complex scalar source field; no material response",
        ),
        BeamlineComponent(
            "source_boundary_condition", "field_state_boundary", "source_plane", 0.0, 0.0, 0.0, True,
            component_specific_parameters={
                "boundary_plane": "source_plane",
                "physical_approximation": "input field state supplied at source boundary",
                "could_emulate": "upstream source/steering error not explicitly represented",
            },
            status="boundary_condition",
            physical_model_available=False,
            misalignment_modes_currently_supported=("field_tilt", "beam_decentre"),
            downstream_elements_affected=downstream("source_boundary_condition"),
            note="not a component misalignment; consumes field-state controls only when targeted here",
        ),
        BeamlineComponent(
            "input_aperture", "aperture", "after_beam_conditioning", 0.0, 0.0,
            source_to_axicon_mm, True,
            ComponentPose(
                decentre_x_um=float(config.input_aperture_centre_x_um),
                decentre_y_um=float(config.input_aperture_centre_y_um),
            ),
            clear_aperture={"shape": "circular", "radius_um": float(config.input_aperture_radius_um)},
            misalignment_modes_currently_supported=("decentre_x_um", "decentre_y_um", "radius_um"),
            downstream_elements_affected=downstream("input_aperture"),
            note="active aperture element at the source/conditioning plane",
        ),
        BeamlineComponent(
            "source_to_physical_axicon", "propagation_segment", "before_physical_axicon",
            0.0, 0.0, source_to_axicon_mm, True,
            component_specific_parameters={"distance_to_next_element_mm": source_to_axicon_mm},
            misalignment_modes_currently_supported=("distance_to_next_element_mm",),
            downstream_elements_affected=downstream("source_to_physical_axicon"),
            note=DIAGNOSTIC_GEOMETRY_NOTE,
        ),
        BeamlineComponent(
            "physical_axicon_input_boundary", "field_state_boundary", "before_physical_axicon",
            actual_axicon_z_um, source_to_axicon_mm, 0.0, True,
            component_specific_parameters={
                "boundary_plane": "before_physical_axicon",
                "physical_approximation": "field state supplied immediately before axicon",
                "could_emulate": "upstream steering/decentre accumulated before axicon",
            },
            status="boundary_condition",
            physical_model_available=False,
            misalignment_modes_currently_supported=("field_tilt", "beam_decentre"),
            downstream_elements_affected=downstream("physical_axicon_input_boundary"),
            note="not a component misalignment; explicit axicon-input boundary condition",
        ),
        BeamlineComponent(
            "physical_axicon", "physical_axicon", "physical_axicon_plane",
            nominal_axicon_z_um, source_to_axicon_mm, post_axicon_free_space_mm, True,
            ComponentPose(
                decentre_x_um=float(config.physical_axicon_centre_x_um),
                decentre_y_um=float(config.physical_axicon_centre_y_um),
                axial_offset_um=float(config.physical_axicon_axial_offset_um),
                tip_x_mrad=float(config.physical_axicon_mechanical_tilt_x_mrad),
                tip_y_mrad=float(config.physical_axicon_mechanical_tilt_y_mrad),
            ),
            component_specific_parameters={"cone_parameter_rad_per_um": float(config.axicon_cone_parameter)},
            clear_aperture={"shape": "circular", "radius_um": float(config.physical_axicon_clear_aperture_radius_um)},
            misalignment_modes_currently_supported=(
                "decentre_x_um", "decentre_y_um", "axial_offset_um",
                "clear_aperture_radius_um", "cone_parameter_rad_per_um",
            ),
            downstream_elements_affected=downstream("physical_axicon"),
            note="thin scalar axicon transmission: aperture * exp(i phase); mechanical tilt is not active",
        ),
        BeamlineComponent(
            "after_physical_axicon_boundary", "field_state_boundary", "after_physical_axicon",
            actual_axicon_z_um, 0.0, post_axicon_free_space_mm, True,
            component_specific_parameters={
                "boundary_plane": "after_physical_axicon",
                "physical_approximation": "post-axicon steering test",
                "could_emulate": "downstream steering optic only if such optic is declared separately",
            },
            status="boundary_condition",
            physical_model_available=False,
            misalignment_modes_currently_supported=("field_tilt", "beam_decentre"),
            downstream_elements_affected=downstream("after_physical_axicon_boundary"),
            note="boundary condition, not an upstream axicon fault",
        ),
        BeamlineComponent(
            "post_axicon_free_space_segment", "propagation_segment", "post_axicon_diagnostic_boundary",
            actual_axicon_z_um, 0.0, post_axicon_free_space_mm, True,
            component_specific_parameters={"distance_to_next_element_mm": post_axicon_free_space_mm},
            misalignment_modes_currently_supported=("distance_to_next_element_mm",),
            downstream_elements_affected=downstream("post_axicon_free_space_segment"),
            note=DIAGNOSTIC_GEOMETRY_NOTE,
        ),
        BeamlineComponent(
            "post_axicon_diagnostic_boundary", "field_state_boundary", "post_axicon_diagnostic_boundary",
            nominal_diagnostic_z_um, post_axicon_free_space_mm, diagnostic_to_ref_mm, True,
            component_specific_parameters={
                "boundary_plane": "post_axicon_diagnostic_boundary",
                "physical_approximation": "post-axicon diagnostic field-state boundary",
                "could_emulate": "downstream steering if that optic is declared separately",
            },
            status="boundary_condition",
            physical_model_available=False,
            misalignment_modes_currently_supported=("field_tilt", "beam_decentre"),
            downstream_elements_affected=downstream("post_axicon_diagnostic_boundary"),
            note="diagnostic boundary only; no objective model is present",
        ),
        BeamlineComponent(
            "post_axicon_to_reference_segment", "propagation_segment", "free_space_reference_plane",
            nominal_diagnostic_z_um, 0.0, diagnostic_to_ref_mm, True,
            component_specific_parameters={"distance_to_next_element_mm": diagnostic_to_ref_mm},
            misalignment_modes_currently_supported=("distance_to_next_element_mm",),
            downstream_elements_affected=downstream("post_axicon_to_reference_segment"),
            note=DIAGNOSTIC_GEOMETRY_NOTE,
        ),
        BeamlineComponent(
            "reference_plane", "reference_plane", "free_space_reference_plane",
            reference_z_um, diagnostic_to_ref_mm, 0.0, True,
            status="diagnostic_only",
            physical_model_available=False,
            misalignment_modes_currently_supported=(),
            downstream_elements_affected=(),
            note="free-space optical diagnostic plane; no sample or material model",
        ),
    )


def build_route_component_declarations(config: RouteAwareAxiconConfig) -> tuple[BeamlineComponent, ...]:
    """Return represented components plus explicitly unsupported route declarations."""
    represented = list(build_physical_axicon_beamline(config))
    unsupported = [
        BeamlineComponent(
            "steering_mirror", "steering_mirror", "not_represented_by_current_engine",
            0.0, 0.0, 0.0, False, status="warning_only",
            represented_by_current_engine=False, physical_model_available=False,
            misalignment_modes_currently_supported=(),
            downstream_elements_affected=("input_aperture", "physical_axicon", "reference_plane"),
            note="mirror tilt requires an explicit mirror/reflection plane; otherwise use a labelled field boundary",
        ),
        BeamlineComponent(
            "SLM1", "SLM", "not_represented_by_current_engine",
            0.0, 0.0, 0.0, False, status="warning_only",
            represented_by_current_engine=False, physical_model_available=False,
            misalignment_modes_currently_supported=(),
            downstream_elements_affected=("Fourier_filter", "relay_lens", "objective_pupil"),
            note="holographic SLM route is not executed by the physical-axicon engine",
        ),
        BeamlineComponent(
            "SLM2", "SLM", "not_represented_by_current_engine",
            0.0, 0.0, 0.0, False, status="warning_only",
            represented_by_current_engine=False, physical_model_available=False,
            misalignment_modes_currently_supported=(),
            downstream_elements_affected=("Fourier_filter", "relay_lens", "objective_pupil"),
            note="holographic SLM route is not executed by the physical-axicon engine",
        ),
        BeamlineComponent(
            "Fourier_filter", "Fourier_filter", "fourier_plane",
            0.0, 0.0, 0.0, False, status="warning_only",
            represented_by_current_engine=False, physical_model_available=False,
            misalignment_modes_currently_supported=(),
            downstream_elements_affected=("relay_lens", "objective_pupil"),
            note="no explicit 4F Fourier filtering plane in the current engine",
        ),
        BeamlineComponent(
            "relay_lens", "relay_lens", "relay_plane",
            0.0, 0.0, 0.0, False, status="warning_only",
            represented_by_current_engine=False, physical_model_available=False,
            misalignment_modes_currently_supported=(),
            downstream_elements_affected=("objective_pupil",),
            note="relay imaging plane is not represented by the physical-axicon engine",
        ),
        BeamlineComponent(
            "objective_pupil", "objective_pupil", "objective_pupil_plane",
            0.0, 0.0, 0.0, False, status="future_not_implemented",
            represented_by_current_engine=False, physical_model_available=False,
            misalignment_modes_currently_supported=(),
            downstream_elements_affected=("objective", "reference_plane"),
            note="pupil clipping/Zernike model is not part of this free-space physical-axicon correction",
        ),
        BeamlineComponent(
            "objective", "objective", "after_objective",
            0.0, 0.0, 0.0, False, status="future_not_implemented",
            represented_by_current_engine=False, physical_model_available=False,
            misalignment_modes_currently_supported=(),
            downstream_elements_affected=("reference_plane",),
            note="objective physics is not modelled and is not in the executed physical-axicon route",
        ),
    ]
    return tuple(represented + unsupported)


def _location_is_upstream_of_axicon(location: str) -> bool:
    return _valid_location(location) in {"source_plane", "after_beam_conditioning", "before_physical_axicon"}


def _distance_from_location_to_axicon_um(location: str, config: RouteAwareAxiconConfig) -> float:
    loc = _valid_location(location)
    source_to_axicon_mm, _, _ = _actual_segment_distances_mm(config)
    if loc in {"source_plane", "after_beam_conditioning"}:
        return float(source_to_axicon_mm) * 1000.0
    if loc == "before_physical_axicon":
        return 0.0
    return 0.0


def _phase_ramp(field: np.ndarray, grid: Mapping[str, Any], config: RouteAwareAxiconConfig,
                tx_mrad: float, ty_mrad: float) -> np.ndarray:
    k = config.k_medium_rad_per_m
    kx = k * np.sin(float(tx_mrad) * 1e-3)
    ky = k * np.sin(float(ty_mrad) * 1e-3)
    return field * np.exp(1j * (kx * np.asarray(grid["X"]) + ky * np.asarray(grid["Y"])))


def _circular_mask(grid: Mapping[str, Any], radius_um: float, cx_um: float = 0.0, cy_um: float = 0.0) -> np.ndarray:
    r = np.hypot(np.asarray(grid["X"]) - float(cx_um) * _UM,
                 np.asarray(grid["Y"]) - float(cy_um) * _UM)
    return (r <= float(radius_um) * _UM).astype(float)


def _centroid(field: np.ndarray, x_um: np.ndarray, y_um: np.ndarray) -> tuple[float, float]:
    I = np.abs(field) ** 2
    s = float(np.sum(I))
    if s <= 0:
        return 0.0, 0.0
    X, Y = np.meshgrid(x_um, y_um)
    return float(np.sum(I * X) / s), float(np.sum(I * Y) / s)


def _beam_radii(field: np.ndarray, x_um: np.ndarray, y_um: np.ndarray,
                cx: float, cy: float) -> tuple[float, float, float]:
    I = np.abs(field) ** 2
    s = float(np.sum(I))
    if s <= 0:
        return 0.0, 0.0, 1.0
    X, Y = np.meshgrid(x_um, y_um)
    rx = float(2.0 * np.sqrt(np.sum(I * (X - cx) ** 2) / s))
    ry = float(2.0 * np.sqrt(np.sum(I * (Y - cy) ** 2) / s))
    ell = float(rx / max(ry, 1e-12))
    return rx, ry, ell


def _spectral_angle_mrad(field: np.ndarray, grid: Mapping[str, Any],
                         config: RouteAwareAxiconConfig) -> tuple[float, float]:
    spec = np.abs(fft2c(field)) ** 2
    total = float(np.sum(spec))
    if total <= 0:
        return 0.0, 0.0
    kx = 2.0 * np.pi * np.asarray(grid["FX"])
    ky = 2.0 * np.pi * np.asarray(grid["FY"])
    mx = float(np.sum(spec * kx) / total)
    my = float(np.sum(spec * ky) / total)
    k = config.k_medium_rad_per_m
    kz = np.sqrt(max(k * k - mx * mx - my * my, 1e-30))
    return float(np.arctan2(mx, kz) * 1000.0), float(np.arctan2(my, kz) * 1000.0)


def _field_metric_summary(
    field: np.ndarray,
    grid: Mapping[str, Any],
    config: RouteAwareAxiconConfig,
    energy_uJ: float,
) -> dict[str, float]:
    x_um, y_um = _xy_um(grid)
    cx, cy = _centroid(field, x_um, y_um)
    ax, ay = _spectral_angle_mrad(field, grid, config)
    rx, ry, ell = _beam_radii(field, x_um, y_um, cx, cy)
    return {
        "energy_uJ": float(energy_uJ),
        "field_power_integral": float(field_power(field, config.dx_um, config.dx_um)),
        "centroid_x_um": float(cx),
        "centroid_y_um": float(cy),
        "angle_x_mrad": float(ax),
        "angle_y_mrad": float(ay),
        "beam_radius_x_um": float(rx),
        "beam_radius_y_um": float(ry),
        "ellipticity": float(ell),
    }


def _energy_after_transform(
    energy_before_uJ: float,
    field_before: np.ndarray,
    field_after: np.ndarray,
    config: RouteAwareAxiconConfig,
) -> float:
    p0 = field_power(field_before, config.dx_um, config.dx_um)
    p1 = field_power(field_after, config.dx_um, config.dx_um)
    return float(energy_before_uJ) * p1 / max(p0, 1e-30)


def _make_inspection_record(
    component: BeamlineComponent,
    field_before: np.ndarray,
    field_after: np.ndarray,
    energy_before_uJ: float,
    energy_after_uJ: float,
    grid: Mapping[str, Any],
    config: RouteAwareAxiconConfig,
    *,
    aperture_overlap: float | None = None,
    transform_applied: bool = False,
    warnings: tuple[str, ...] = (),
    model_status: str | None = None,
) -> RouteInspectionRecord:
    incoming = _field_metric_summary(field_before, grid, config, energy_before_uJ)
    outgoing = _field_metric_summary(field_after, grid, config, energy_after_uJ)
    return RouteInspectionRecord(
        component_id=component.component_id,
        component_name=component.component_id.replace("_", " "),
        component_type=component.component_type,
        nominal_location_um=float(component.nominal_z_position_um),
        distance_from_previous_component_mm=float(component.distance_from_previous_component_mm),
        distance_to_next_element_mm=float(component.distance_to_next_element_mm),
        actual_pose_error=component.physical_pose.as_dict(),
        incoming_field_metrics=incoming,
        outgoing_field_metrics=outgoing,
        energy_before_uJ=float(energy_before_uJ),
        energy_after_uJ=float(energy_after_uJ),
        centroid_before_um=(incoming["centroid_x_um"], incoming["centroid_y_um"]),
        centroid_after_um=(outgoing["centroid_x_um"], outgoing["centroid_y_um"]),
        angle_before_mrad=(incoming["angle_x_mrad"], incoming["angle_y_mrad"]),
        angle_after_mrad=(outgoing["angle_x_mrad"], outgoing["angle_y_mrad"]),
        aperture_overlap=aperture_overlap,
        downstream_consequences=component.downstream_elements_affected,
        model_status=model_status or component.status,
        transform_applied=bool(transform_applied),
        warnings=warnings,
        represented_by_current_engine=component.represented_by_current_engine,
        physical_model_available=component.physical_model_available,
        misalignment_modes_currently_supported=component.misalignment_modes_currently_supported,
    )


def route_inspection_rows(run: RouteAwareAxiconRun) -> tuple[dict[str, Any], ...]:
    """Return GUI-ready route-inspection rows for an executed run."""
    return tuple(r.as_dict() for r in run.route_inspection_records)


def _field_of_view_margin_um(field: np.ndarray, x_um: np.ndarray, y_um: np.ndarray) -> float:
    cx, cy = _centroid(field, x_um, y_um)
    rx, ry, _ = _beam_radii(field, x_um, y_um, cx, cy)
    mx = min(cx - float(x_um.min()), float(x_um.max()) - cx) - 1.5 * rx
    my = min(cy - float(y_um.min()), float(y_um.max()) - cy) - 1.5 * ry
    return float(min(mx, my))


def _shift_field_um(field: np.ndarray, dx_um: float, dy_um: float, pixel_um: float) -> np.ndarray:
    if abs(dx_um) + abs(dy_um) <= 0:
        return field
    return nd_shift(
        field,
        shift=(float(dy_um) / float(pixel_um), float(dx_um) / float(pixel_um)),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def _make_source_field(grid: Mapping[str, Any], config: RouteAwareAxiconConfig) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(grid["X"], float) / _UM
    Y = np.asarray(grid["Y"], float) / _UM
    ell = max(float(config.input_beam_ellipticity), 1e-6)
    wx = float(config.input_beam_radius_um) * np.sqrt(ell)
    wy = float(config.input_beam_radius_um) / np.sqrt(ell)
    rot = np.deg2rad(float(config.input_beam_rotation_deg))
    Xr = X * np.cos(rot) + Y * np.sin(rot)
    Yr = -X * np.sin(rot) + Y * np.cos(rot)
    amp0 = np.exp(-((Xr / max(wx, 1e-9)) ** 2 + (Yr / max(wy, 1e-9)) ** 2))
    aperture = _circular_mask(grid, float(config.input_aperture_radius_um), 0.0, 0.0)
    field = amp0.astype(complex) * aperture
    return field, amp0.astype(complex)


def _apply_location_perturbations(
    field: np.ndarray,
    location: str,
    grid: Mapping[str, Any],
    config: RouteAwareAxiconConfig,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Apply generic field perturbations injected at a represented route location."""
    loc = _valid_location(location)
    out = field
    applied: list[str] = []
    if _valid_location(config.beam_decentre_location) == loc and (
        abs(config.beam_decentre_x_um) > 0 or abs(config.beam_decentre_y_um) > 0
    ):
        out = _shift_field_um(out, config.beam_decentre_x_um, config.beam_decentre_y_um, config.dx_um)
        applied.append(f"beam_decentre_at_{loc}")
    if _valid_location(config.field_tilt_location) == loc and (
        abs(config.field_tilt_x_mrad) > 0 or abs(config.field_tilt_y_mrad) > 0
    ):
        out = _phase_ramp(out, grid, config, config.field_tilt_x_mrad, config.field_tilt_y_mrad)
        label = "post_axicon_steering_test" if loc == "after_physical_axicon" else f"field_tilt_at_{loc}"
        applied.append(label)
    return out, tuple(applied)


def physical_axicon_transmission(grid: Mapping[str, Any], config: RouteAwareAxiconConfig) -> np.ndarray:
    X = np.asarray(grid["X"], float) / _UM
    Y = np.asarray(grid["Y"], float) / _UM
    xa = float(config.physical_axicon_centre_x_um)
    ya = float(config.physical_axicon_centre_y_um)
    r = np.hypot(X - xa, Y - ya)
    aperture = (r <= float(config.physical_axicon_clear_aperture_radius_um)).astype(float)
    phase = -float(config.axicon_cone_parameter) * r
    return aperture * np.exp(1j * phase)


def build_route_perturbation_records(config: RouteAwareAxiconConfig) -> tuple[RoutePerturbationRecord, ...]:
    route = str(config.route_mode)
    tilt_loc = _valid_location(config.field_tilt_location)
    dec_loc = _valid_location(config.beam_decentre_location)
    tilt_component = _boundary_component_id_for_location(tilt_loc)
    dec_component = _boundary_component_id_for_location(dec_loc)
    tilt_status = "post_axicon_steering_test" if tilt_loc == "after_physical_axicon" else "physics_active"
    if tilt_loc == "free_space_reference_plane":
        tilt_status = "diagnostic_only"
    records = [
        RoutePerturbationRecord(
            "field_tilt_x_mrad", "field_tilt", float(config.field_tilt_x_mrad), "mrad",
            abs(config.field_tilt_x_mrad) > 0, tilt_loc, route, tilt_loc,
            _downstream_elements(tilt_loc), tilt_status,
            "field-state boundary condition at a named plane, not a generic component misalignment",
            tilt_component, tilt_loc,
            "phase ramp applied to the complex field at the boundary plane",
            "upstream steering/mirror/source-angle error if that hardware is not explicitly represented",
            _downstream_component_ids(tilt_component),
        ),
        RoutePerturbationRecord(
            "field_tilt_y_mrad", "field_tilt", float(config.field_tilt_y_mrad), "mrad",
            abs(config.field_tilt_y_mrad) > 0, tilt_loc, route, tilt_loc,
            _downstream_elements(tilt_loc), tilt_status,
            "field-state boundary condition at a named plane, not a generic component misalignment",
            tilt_component, tilt_loc,
            "phase ramp applied to the complex field at the boundary plane",
            "upstream steering/mirror/source-angle error if that hardware is not explicitly represented",
            _downstream_component_ids(tilt_component),
        ),
        RoutePerturbationRecord(
            "beam_decentre_x_um", "beam_decentre", float(config.beam_decentre_x_um), "um",
            abs(config.beam_decentre_x_um) > 0, dec_loc, route, dec_loc,
            _downstream_elements(dec_loc), "physics_active",
            "field-state boundary condition at a named plane, not a generic component misalignment",
            dec_component, dec_loc,
            "transverse interpolation shift applied to the complex field at the boundary plane",
            "upstream decentre/walkoff if that hardware is not explicitly represented",
            _downstream_component_ids(dec_component),
        ),
        RoutePerturbationRecord(
            "beam_decentre_y_um", "beam_decentre", float(config.beam_decentre_y_um), "um",
            abs(config.beam_decentre_y_um) > 0, dec_loc, route, dec_loc,
            _downstream_elements(dec_loc), "physics_active",
            "field-state boundary condition at a named plane, not a generic component misalignment",
            dec_component, dec_loc,
            "transverse interpolation shift applied to the complex field at the boundary plane",
            "upstream decentre/walkoff if that hardware is not explicitly represented",
            _downstream_component_ids(dec_component),
        ),
        RoutePerturbationRecord(
            "input_beam_ellipticity", "beam_shape", float(config.input_beam_ellipticity), "ratio",
            abs(config.input_beam_ellipticity - 1.0) > 1e-12, "source_plane", route,
            "source_plane", _downstream_elements("source_plane"), "physics_active",
            "source-field component parameter",
            "source_field",
        ),
        RoutePerturbationRecord(
            "input_aperture_radius_um", "aperture", float(config.input_aperture_radius_um), "um",
            True, "after_beam_conditioning", route, "after_beam_conditioning",
            _downstream_elements("after_beam_conditioning"), "physics_active",
            "input-aperture clear radius at the aperture component plane",
            "input_aperture",
        ),
        RoutePerturbationRecord(
            "input_aperture_centre_x_um", "aperture_decentre", float(config.input_aperture_centre_x_um), "um",
            abs(config.input_aperture_centre_x_um) > 0, "after_beam_conditioning", route,
            "after_beam_conditioning", _downstream_elements("after_beam_conditioning"), "physics_active",
            "input-aperture pose decentre applied at its own component plane",
            "input_aperture",
        ),
        RoutePerturbationRecord(
            "input_aperture_centre_y_um", "aperture_decentre", float(config.input_aperture_centre_y_um), "um",
            abs(config.input_aperture_centre_y_um) > 0, "after_beam_conditioning", route,
            "after_beam_conditioning", _downstream_elements("after_beam_conditioning"), "physics_active",
            "input-aperture pose decentre applied at its own component plane",
            "input_aperture",
        ),
        RoutePerturbationRecord(
            "physical_axicon_centre_x_um", "mechanical_lateral_offset",
            float(config.physical_axicon_centre_x_um), "um",
            abs(config.physical_axicon_centre_x_um) > 0, "physical_axicon_plane", route,
            "physical_axicon_plane", _downstream_elements("physical_axicon_plane"), "physics_active",
            "mechanical lateral axicon displacement",
            "physical_axicon",
        ),
        RoutePerturbationRecord(
            "physical_axicon_centre_y_um", "mechanical_lateral_offset",
            float(config.physical_axicon_centre_y_um), "um",
            abs(config.physical_axicon_centre_y_um) > 0, "physical_axicon_plane", route,
            "physical_axicon_plane", _downstream_elements("physical_axicon_plane"), "physics_active",
            "mechanical lateral axicon displacement",
            "physical_axicon",
        ),
        RoutePerturbationRecord(
            "physical_axicon_axial_offset_um", "mechanical_axial_offset",
            float(config.physical_axicon_axial_offset_um), "um",
            abs(config.physical_axicon_axial_offset_um) > 0, "physical_axicon_plane", route,
            "physical_axicon_plane", _downstream_elements("physical_axicon_plane"), "physics_active",
            "axicon axial pose changes adjacent propagation segment distances",
            "physical_axicon",
        ),
        RoutePerturbationRecord(
            "physical_axicon_clear_aperture_radius_um", "aperture",
            float(config.physical_axicon_clear_aperture_radius_um), "um",
            True, "physical_axicon_plane", route, "physical_axicon_plane",
            _downstream_elements("physical_axicon_plane"), "physics_active",
            "physical axicon clear aperture",
            "physical_axicon",
        ),
        RoutePerturbationRecord(
            "axicon_cone_parameter", "phase_element", float(config.axicon_cone_parameter), "rad/um",
            True, "physical_axicon_plane", route, "physical_axicon_plane",
            _downstream_elements("physical_axicon_plane"), "physics_active",
            "fixed thin physical-axicon phase",
            "physical_axicon",
        ),
        RoutePerturbationRecord(
            "physical_axicon_mechanical_tilt_x_mrad", "physical_axicon_mechanical_tilt",
            float(config.physical_axicon_mechanical_tilt_x_mrad), "mrad",
            abs(config.physical_axicon_mechanical_tilt_x_mrad) > 0, "physical_axicon_plane", route,
            "not_represented_by_current_engine", _downstream_elements("physical_axicon_plane"),
            "future_not_implemented",
            "not silently represented as generic field tilt; thin-element approximation not yet implemented",
            "physical_axicon",
        ),
        RoutePerturbationRecord(
            "physical_axicon_mechanical_tilt_y_mrad", "physical_axicon_mechanical_tilt",
            float(config.physical_axicon_mechanical_tilt_y_mrad), "mrad",
            abs(config.physical_axicon_mechanical_tilt_y_mrad) > 0, "physical_axicon_plane", route,
            "not_represented_by_current_engine", _downstream_elements("physical_axicon_plane"),
            "future_not_implemented",
            "not silently represented as generic field tilt; thin-element approximation not yet implemented",
            "physical_axicon",
        ),
    ]
    return tuple(records)


def holographic_slm_route_declarations() -> tuple[RoutePerturbationRecord, ...]:
    return (
        RoutePerturbationRecord(
            "fourier_plane_filter", "fourier_filter", "not configured", "n/a", False,
            "fourier_plane", "holographic_slm", "not_represented_by_current_engine",
            ("after_fourier_filter", "relay_plane", "objective_pupil_plane"), "warning_only",
            "no explicit 4F Fourier plane in current engine",
        ),
        RoutePerturbationRecord(
            "relay_plane_errors", "relay_error", "not configured", "n/a", False,
            "relay_plane", "holographic_slm", "not_represented_by_current_engine",
            ("objective_pupil_plane",), "warning_only",
            "relay imaging plane not represented by current engine",
        ),
    )


def _axicon_metrics(
    source_field: np.ndarray,
    launch_field: np.ndarray,
    incident: np.ndarray,
    after_axicon: np.ndarray,
    grid: Mapping[str, Any],
    config: RouteAwareAxiconConfig,
    p_source_unapertured: float,
    p_source: float,
) -> dict[str, Any]:
    x_um, y_um = _xy_um(grid)
    sx, sy = _centroid(source_field, x_um, y_um)
    lx, ly = _centroid(launch_field, x_um, y_um)
    cx, cy = _centroid(incident, x_um, y_um)
    rx, ry, ell = _beam_radii(incident, x_um, y_um, cx, cy)
    ax_angle_x, ax_angle_y = _spectral_angle_mrad(incident, grid, config)
    ax_mask = _circular_mask(
        grid,
        config.physical_axicon_clear_aperture_radius_um,
        config.physical_axicon_centre_x_um,
        config.physical_axicon_centre_y_um,
    )
    p_incident = field_power(incident, config.dx_um, config.dx_um)
    p_after = field_power(after_axicon, config.dx_um, config.dx_um)
    overlap = float(np.sum((np.abs(incident) ** 2) * ax_mask) / max(np.sum(np.abs(incident) ** 2), 1e-30))
    tilt_loc = _valid_location(config.field_tilt_location)
    applies_before_axicon = _location_is_upstream_of_axicon(tilt_loc)
    if applies_before_axicon:
        tilt_distance_um = _distance_from_location_to_axicon_um(tilt_loc, config)
        expected_dx = float(tilt_distance_um * np.tan(config.field_tilt_x_mrad * 1e-3))
        expected_dy = float(tilt_distance_um * np.tan(config.field_tilt_y_mrad * 1e-3))
        measured_dx = float(cx - lx)
        measured_dy = float(cy - ly)
        walkoff_error = float(np.hypot(measured_dx - expected_dx, measured_dy - expected_dy))
    else:
        expected_dx = expected_dy = 0.0
        measured_dx = measured_dy = 0.0
        walkoff_error = 0.0
    rel_x = float(cx - config.physical_axicon_centre_x_um)
    rel_y = float(cy - config.physical_axicon_centre_y_um)
    return {
        "field_location": "physical_axicon_plane",
        "field_tilt_injection_location": tilt_loc,
        "beam_decentre_injection_location": _valid_location(config.beam_decentre_location),
        "source_centroid_x_um": sx,
        "source_centroid_y_um": sy,
        "pre_axicon_launch_centroid_x_um": lx,
        "pre_axicon_launch_centroid_y_um": ly,
        "incident_centroid_x_um": cx,
        "incident_centroid_y_um": cy,
        "incident_angle_x_mrad": ax_angle_x,
        "incident_angle_y_mrad": ax_angle_y,
        "incident_beam_radius_x_um": rx,
        "incident_beam_radius_y_um": ry,
        "incident_ellipticity": ell,
        "axicon_centre_x_um": float(config.physical_axicon_centre_x_um),
        "axicon_centre_y_um": float(config.physical_axicon_centre_y_um),
        "relative_beam_to_axicon_offset_x_um": rel_x,
        "relative_beam_to_axicon_offset_y_um": rel_y,
        "relative_beam_to_axicon_offset_um": float(np.hypot(rel_x, rel_y)),
        "axicon_aperture_overlap_fraction": overlap,
        "source_aperture_transmitted_fraction": float(p_source / max(p_source_unapertured, 1e-30)),
        "axicon_transmitted_fraction": float(p_after / max(p_incident, 1e-30)),
        "field_of_view_margin_um": _field_of_view_margin_um(incident, x_um, y_um),
        "upstream_tilt_predicted_walkoff_x_um": expected_dx,
        "upstream_tilt_predicted_walkoff_y_um": expected_dy,
        "upstream_tilt_predicted_walkoff_um": float(np.hypot(expected_dx, expected_dy)),
        "upstream_tilt_measured_walkoff_x_um": measured_dx,
        "upstream_tilt_measured_walkoff_y_um": measured_dy,
        "upstream_tilt_measured_walkoff_um": float(np.hypot(measured_dx, measured_dy)),
        "walkoff_model_error_um": walkoff_error,
        "walkoff_model_applies": bool(applies_before_axicon),
        "diagnostic_geometry_note": DIAGNOSTIC_GEOMETRY_NOTE,
    }


def run_route_aware_axicon_pipeline(
    controls: Mapping[str, Any] | None = None,
    *,
    config: RouteAwareAxiconConfig | None = None,
) -> RouteAwareAxiconRun:
    """Run the component-owned physical-axicon diagnostic pipeline."""
    config = _with_overrides(config or RouteAwareAxiconConfig(), controls)
    if config.route_mode != "physical_axicon":
        warnings = (
            "holographic_slm route declarations are available, but this physical-axicon "
            "pipeline does not fake unrepresented 4F/relay planes.",
        )
        config = replace(config, route_mode="physical_axicon")
    else:
        warnings = ()

    grid = _grid(config)
    x_um, y_um = _xy_um(grid)
    components = build_physical_axicon_beamline(config)
    component_by_id = {c.component_id: c for c in components}
    inspection: list[RouteInspectionRecord] = []

    source_unapertured, _ = _make_source_field(grid, replace(config, input_aperture_radius_um=1.0e9))
    source = source_unapertured.copy()
    p_unap = field_power(source_unapertured, config.dx_um, config.dx_um)
    source_energy_before = float(config.input_pulse_energy_uJ)
    inspection.append(_make_inspection_record(
        component_by_id["source_field"], source_unapertured, source,
        source_energy_before, source_energy_before, grid, config,
        transform_applied=True,
        model_status="physics_active",
    ))

    source_before_boundary = source.copy()
    source, a_source = _apply_location_perturbations(source, "source_plane", grid, config)
    source_boundary_energy = _energy_after_transform(source_energy_before, source_before_boundary, source, config)
    inspection.append(_make_inspection_record(
        component_by_id["source_boundary_condition"], source_before_boundary, source,
        source_energy_before, source_boundary_energy, grid, config,
        transform_applied=bool(a_source),
        model_status="boundary_condition_active" if a_source else "boundary_condition_available",
        warnings=tuple(a_source),
    ))

    aperture_component = component_by_id["input_aperture"]
    aperture_mask = _circular_mask(
        grid,
        float(config.input_aperture_radius_um),
        float(config.input_aperture_centre_x_um),
        float(config.input_aperture_centre_y_um),
    )
    before_aperture = source.copy()
    source = source * aperture_mask
    p_source = field_power(source, config.dx_um, config.dx_um)
    source_energy = _energy_after_transform(source_boundary_energy, before_aperture, source, config)
    aperture_overlap = float(
        np.sum((np.abs(before_aperture) ** 2) * aperture_mask)
        / max(np.sum(np.abs(before_aperture) ** 2), 1e-30)
    )
    inspection.append(_make_inspection_record(
        aperture_component, before_aperture, source,
        source_boundary_energy, source_energy, grid, config,
        aperture_overlap=aperture_overlap,
        transform_applied=True,
        model_status="physics_active",
    ))

    before_after_beam_conditioning = source.copy()
    source, a_after_conditioning = _apply_location_perturbations(source, "after_beam_conditioning", grid, config)
    if a_after_conditioning:
        source_after_conditioning_energy = _energy_after_transform(
            source_energy, before_after_beam_conditioning, source, config
        )
        inspection.append(_make_inspection_record(
            component_by_id["source_boundary_condition"], before_after_beam_conditioning, source,
            source_energy, source_after_conditioning_energy, grid, config,
            transform_applied=True,
            model_status="boundary_condition_active",
            warnings=tuple(a_after_conditioning),
        ))
        source_energy = source_after_conditioning_energy
    pre_axicon_launch = source.copy()

    pre_prop = make_bl_asm_propagator(
        source, grid, config.wavelength_m, n_medium=config.n_medium, bandlimit=config.bandlimit
    )
    source_to_axicon_mm, post_axicon_free_space_mm, diagnostic_to_ref_mm = _actual_segment_distances_mm(config)
    incident_pre_location = pre_prop(source_to_axicon_mm * _MM)
    inspection.append(_make_inspection_record(
        component_by_id["source_to_physical_axicon"], source, incident_pre_location,
        source_energy, source_energy, grid, config,
        transform_applied=True,
        model_status="physics_active",
    ))

    if _valid_location(config.field_tilt_location) in {"source_plane", "after_beam_conditioning"}:
        walkoff_reference = pre_axicon_launch
    else:
        walkoff_reference = incident_pre_location
    before_axicon_boundary = incident_pre_location.copy()
    incident, applied_before_axicon = _apply_location_perturbations(
        incident_pre_location, "before_physical_axicon", grid, config
    )
    incident_energy = _energy_after_transform(source_energy, before_axicon_boundary, incident, config)
    inspection.append(_make_inspection_record(
        component_by_id["physical_axicon_input_boundary"], before_axicon_boundary, incident,
        source_energy, incident_energy, grid, config,
        transform_applied=bool(applied_before_axicon),
        model_status="boundary_condition_active" if applied_before_axicon else "boundary_condition_available",
        warnings=tuple(applied_before_axicon),
    ))

    T = physical_axicon_transmission(grid, config)
    before_axicon = incident.copy()
    after_axicon_component = incident * T
    p_incident = field_power(incident, config.dx_um, config.dx_um)
    p_after = field_power(after_axicon_component, config.dx_um, config.dx_um)
    axicon_energy = incident_energy * p_after / max(p_incident, 1e-30)
    ax_mask = _circular_mask(
        grid,
        config.physical_axicon_clear_aperture_radius_um,
        config.physical_axicon_centre_x_um,
        config.physical_axicon_centre_y_um,
    )
    axicon_overlap = float(
        np.sum((np.abs(incident) ** 2) * ax_mask) / max(np.sum(np.abs(incident) ** 2), 1e-30)
    )
    axicon_warnings: list[str] = []
    if abs(config.physical_axicon_mechanical_tilt_x_mrad) > 0 or abs(config.physical_axicon_mechanical_tilt_y_mrad) > 0:
        axicon_warnings.append(
            "physical axicon mechanical tilt is future_not_implemented and was not converted to field tilt"
        )
    inspection.append(_make_inspection_record(
        component_by_id["physical_axicon"], before_axicon, after_axicon_component,
        incident_energy, axicon_energy, grid, config,
        aperture_overlap=axicon_overlap,
        transform_applied=True,
        warnings=tuple(axicon_warnings),
        model_status="physics_active",
    ))

    before_after_axicon_boundary = after_axicon_component.copy()
    after_axicon = after_axicon_component
    after_axicon, applied_after_axicon = _apply_location_perturbations(
        after_axicon, "after_physical_axicon", grid, config
    )
    post_axicon_energy = _energy_after_transform(axicon_energy, before_after_axicon_boundary, after_axicon, config)
    inspection.append(_make_inspection_record(
        component_by_id["after_physical_axicon_boundary"], before_after_axicon_boundary, after_axicon,
        axicon_energy, post_axicon_energy, grid, config,
        transform_applied=bool(applied_after_axicon),
        model_status="boundary_condition_active" if applied_after_axicon else "boundary_condition_available",
        warnings=tuple(applied_after_axicon),
    ))

    metrics = _axicon_metrics(source, walkoff_reference, incident, after_axicon_component, grid, config, p_unap, p_source)
    records = build_route_perturbation_records(config)
    warnings = tuple(list(warnings) + [
        r.note for r in records if r.enabled and r.status in {"future_not_implemented", "warning_only"}
    ])

    source_state = ComponentPlaneState(
        plane_name="source_plane",
        field=source,
        x_um=x_um,
        y_um=y_um,
        dx_um=config.dx_um,
        dy_um=config.dx_um,
        pulse_energy_before_uJ=config.input_pulse_energy_uJ,
        pulse_energy_after_uJ=source_energy,
        transmitted_fraction=source_energy / max(config.input_pulse_energy_uJ, 1e-30),
        applied_components=(
            "source_field",
            "input_aperture",
        ) + tuple(a_source) + tuple(a_after_conditioning),
        metadata={
            "route": "physical_axicon",
            "physical_location": "source_plane",
            "route_graph": [n.__dict__ for n in physical_axicon_route_graph(config)],
            "component_chain": [c.as_dict() for c in components],
        },
    )
    incident_state = ComponentPlaneState(
        plane_name="physical_axicon_incident_field",
        field=incident,
        x_um=x_um,
        y_um=y_um,
        dx_um=config.dx_um,
        dy_um=config.dx_um,
        pulse_energy_before_uJ=source_energy,
        pulse_energy_after_uJ=incident_energy,
        transmitted_fraction=incident_energy / max(source_energy, 1e-30),
        applied_components=("pre_axicon_free_space_propagation",) + tuple(applied_before_axicon),
        metadata={"route": "physical_axicon", "physical_location": "before_physical_axicon"},
    )
    axicon_state = ComponentPlaneState(
        plane_name="physical_axicon_plane",
        field=after_axicon,
        x_um=x_um,
        y_um=y_um,
        dx_um=config.dx_um,
        dy_um=config.dx_um,
        pulse_energy_before_uJ=incident_energy,
        pulse_energy_after_uJ=post_axicon_energy,
        transmitted_fraction=post_axicon_energy / max(incident_energy, 1e-30),
        applied_components=("physical_axicon_thin_element",) + tuple(applied_after_axicon),
        metadata={
            "route": "physical_axicon",
            "physical_location": "physical_axicon_plane",
            "axicon_transmission": "aperture * exp(i phi_axicon)",
        },
    )

    post_prop = make_bl_asm_propagator(
        after_axicon, grid, config.wavelength_m, n_medium=config.n_medium, bandlimit=config.bandlimit
    )
    actual_post_axicon_um = float(post_axicon_free_space_mm + diagnostic_to_ref_mm) * 1000.0
    z_um = np.linspace(0.0, actual_post_axicon_um, int(config.n_z))
    intensity = np.empty((len(z_um), len(x_um), len(x_um)), dtype=float)
    z_diagnostic_um = float(post_axicon_free_space_mm) * 1000.0
    field_at_diagnostic_boundary = post_prop(z_diagnostic_um * _UM)
    inspection.append(_make_inspection_record(
        component_by_id["post_axicon_free_space_segment"], after_axicon, field_at_diagnostic_boundary,
        post_axicon_energy, post_axicon_energy, grid, config,
        transform_applied=True,
        model_status="physics_active",
    ))
    before_post_axicon_diagnostic_boundary = field_at_diagnostic_boundary.copy()
    field_at_diagnostic_boundary, applied_post_axicon_diagnostic = _apply_location_perturbations(
        field_at_diagnostic_boundary, "post_axicon_diagnostic_boundary", grid, config
    )
    post_axicon_diagnostic_energy = _energy_after_transform(
        post_axicon_energy, before_post_axicon_diagnostic_boundary, field_at_diagnostic_boundary, config
    )
    inspection.append(_make_inspection_record(
        component_by_id["post_axicon_diagnostic_boundary"],
        before_post_axicon_diagnostic_boundary,
        field_at_diagnostic_boundary,
        post_axicon_energy, post_axicon_diagnostic_energy, grid, config,
        transform_applied=bool(applied_post_axicon_diagnostic),
        model_status="boundary_condition_active" if applied_post_axicon_diagnostic else "boundary_condition_available",
        warnings=tuple(applied_post_axicon_diagnostic),
    ))
    post_diagnostic_prop = make_bl_asm_propagator(
        field_at_diagnostic_boundary, grid, config.wavelength_m, n_medium=config.n_medium, bandlimit=config.bandlimit
    )
    for i, zz in enumerate(z_um):
        if float(zz) <= z_diagnostic_um + 1e-12:
            U = post_prop(float(zz) * _UM)
        else:
            U = post_diagnostic_prop((float(zz) - z_diagnostic_um) * _UM)
        intensity[i] = np.abs(U) ** 2
    before_reference_segment = field_at_diagnostic_boundary.copy()
    reference_pre_boundary = post_diagnostic_prop(diagnostic_to_ref_mm * _MM)
    inspection.append(_make_inspection_record(
        component_by_id["post_axicon_to_reference_segment"], before_reference_segment, reference_pre_boundary,
        post_axicon_diagnostic_energy, post_axicon_diagnostic_energy, grid, config,
        transform_applied=True,
        model_status="physics_active",
    ))
    reference_field = reference_pre_boundary
    before_reference_boundary = reference_field.copy()
    reference_field, applied_reference = _apply_location_perturbations(
        reference_field, "free_space_reference_plane", grid, config
    )
    reference_energy = _energy_after_transform(
        post_axicon_diagnostic_energy, before_reference_boundary, reference_field, config
    )
    inspection.append(_make_inspection_record(
        component_by_id["reference_plane"], before_reference_boundary, reference_field,
        post_axicon_diagnostic_energy, reference_energy, grid, config,
        transform_applied=bool(applied_reference),
        model_status="diagnostic_only",
        warnings=tuple(applied_reference),
    ))
    intensity[-1] = np.abs(reference_field) ** 2

    reference_state = ComponentPlaneState(
        plane_name="free_space_reference_plane",
        field=reference_field,
        x_um=x_um,
        y_um=y_um,
        dx_um=config.dx_um,
        dy_um=config.dx_um,
        pulse_energy_before_uJ=post_axicon_diagnostic_energy,
        pulse_energy_after_uJ=reference_energy,
        transmitted_fraction=reference_energy / max(post_axicon_diagnostic_energy, 1e-30),
        applied_components=("post_axicon_free_space_propagation",)
        + tuple(applied_post_axicon_diagnostic)
        + tuple(applied_reference),
        metadata={
            "reference_plane": "free-space reference plane after physical axicon route, n=1.0",
            "no_material_model": True,
            "physical_location": "free_space_reference_plane",
            "route_endpoint": "free_space",
        },
    )

    stack = PropagatedFieldStack(
        intensity_zyx=intensity,
        x_um=x_um,
        y_um=y_um,
        z_um=z_um,
        input_pulse_energy_uJ=config.input_pulse_energy_uJ,
        sample_pulse_energy_uJ=reference_energy,
        transmitted_fraction=reference_energy / max(config.input_pulse_energy_uJ, 1e-30),
        plane_states=(source_state, incident_state, axicon_state, reference_state),
        warnings=warnings,
        metadata={
            "stage": "stage8c3r3_route_aware_physical_axicon",
            "route_mode": "physical_axicon",
            "n_medium": float(config.n_medium),
            "diagnostic_geometry_note": DIAGNOSTIC_GEOMETRY_NOTE,
            "route_graph": [n.__dict__ for n in physical_axicon_route_graph(config)],
            "component_chain": [c.as_dict() for c in components],
            "route_inspection": [r.as_dict() for r in inspection],
            "perturbations": [r.as_dict() for r in records],
        },
    )
    return RouteAwareAxiconRun(
        config=config,
        route_mode="physical_axicon",
        component_chain=components,
        source_state=source_state,
        axicon_incident_state=incident_state,
        physical_axicon_state=axicon_state,
        reference_plane_state=reference_state,
        propagated_stack=stack,
        axicon_incidence_metrics=metrics,
        route_inspection_records=tuple(inspection),
        perturbation_records=records,
        warnings=warnings,
        metadata={
            "stage": "stage8c3_route_component_owned_physical_axicon",
            "final_export_allowed": False,
            "component_chain": [c.as_dict() for c in components],
            "route_inspection": [r.as_dict() for r in inspection],
        },
    )


@dataclass(frozen=True)
class AxiconAlignmentSweepFamily:
    key: str
    title: str
    param_label: str
    param_values: tuple[float, ...]
    control_fn: Callable[[float], Mapping[str, Any]]


@dataclass(frozen=True)
class AxiconAlignmentSweepResult:
    family: AxiconAlignmentSweepFamily
    rows: tuple[Mapping[str, Any], ...]
    label: str = DIAGNOSTIC_SWEEP_LABEL
    final_export_allowed: bool = False


def build_axicon_alignment_sweep_families() -> dict[str, AxiconAlignmentSweepFamily]:
    return {
        "source_tilt": AxiconAlignmentSweepFamily(
            "source_tilt", "Source-plane tilt through fixed axicon", "source tilt (mrad)",
            (0.0, 3.0, 6.0, 9.0, 12.0, 16.0),
            lambda v: {"field_tilt_x_mrad": v, "field_tilt_location": "source_plane", "pre_axicon_distance_mm": 0.80},
        ),
        "axicon_input_decentre": AxiconAlignmentSweepFamily(
            "axicon_input_decentre", "Beam decentre at axicon input", "input decentre at axicon (um)",
            (0.0, 2.0, 4.0, 6.0, 9.0, 12.0),
            lambda v: {"beam_decentre_x_um": v, "beam_decentre_location": "before_physical_axicon"},
        ),
        "axicon_lateral_offset": AxiconAlignmentSweepFamily(
            "axicon_lateral_offset", "Mechanical axicon lateral offset", "axicon offset (um)",
            (0.0, 2.0, 4.0, 6.0, 9.0, 12.0),
            lambda v: {"physical_axicon_centre_x_um": v},
        ),
        "input_radius": AxiconAlignmentSweepFamily(
            "input_radius", "Input beam radius / aperture overlap", "beam radius (um)",
            (12.0, 16.0, 20.0, 24.0, 30.0, 36.0),
            lambda v: {"input_beam_radius_um": v, "input_aperture_radius_um": 44.0},
        ),
        "axicon_aperture": AxiconAlignmentSweepFamily(
            "axicon_aperture", "Physical axicon clear aperture", "clear aperture radius (um)",
            (18.0, 22.0, 26.0, 32.0, 38.0, 44.0),
            lambda v: {"physical_axicon_clear_aperture_radius_um": v},
        ),
        "relative_source_axicon": AxiconAlignmentSweepFamily(
            "relative_source_axicon", "Relative source/axicon alignment", "opposed offset (um)",
            (0.0, 2.0, 4.0, 6.0, 9.0, 12.0),
            lambda v: {"beam_decentre_x_um": v, "beam_decentre_location": "source_plane",
                       "physical_axicon_centre_x_um": -v},
        ),
    }


def _route_reliability(metrics: Mapping[str, Any], axis: Mapping[str, Any]) -> str:
    if float(metrics["field_of_view_margin_um"]) < 0.0 or float(axis["out_of_frame_fraction"]) > 0.02:
        return "invalid_out_of_frame"
    if float(metrics["axicon_aperture_overlap_fraction"]) < 0.85 or float(axis["ring_fit_quality"]) < 0.45:
        return "caution_crop_limited"
    return "numerically_reliable"


def run_axicon_alignment_sweep(
    family: str | AxiconAlignmentSweepFamily,
    *,
    config: RouteAwareAxiconConfig | None = None,
) -> AxiconAlignmentSweepResult:
    if isinstance(family, str):
        family = build_axicon_alignment_sweep_families()[family]
    config = config or RouteAwareAxiconConfig.fast()
    baseline = run_route_aware_axicon_pipeline(config=config)
    rows: list[dict[str, Any]] = []
    for value in family.param_values:
        run = run_route_aware_axicon_pipeline(family.control_fn(value), config=config)
        st = run.propagated_stack
        sel = int(np.argmax(st.intensity_zyx.max(axis=(1, 2))))
        est = estimate_annular_axis(st.intensity_zyx[sel], st.x_um, st.y_um)
        axis = compute_axis_tracking(st, plane_index=sel, core_radius_um=2.0)
        cls = classify_translation_vs_deformation(
            baseline.propagated_stack, st, plane_index=sel, core_radius_um=2.0
        )
        energy = compute_energy_throughput(run)  # type: ignore[arg-type]
        rows.append({
            "param_value": float(value),
            "diagnostic_label": DIAGNOSTIC_SWEEP_LABEL,
            "axicon_plane_walkoff_um": float(run.axicon_incidence_metrics["relative_beam_to_axicon_offset_um"]),
            "axicon_aperture_overlap_fraction": float(run.axicon_incidence_metrics["axicon_aperture_overlap_fraction"]),
            "transmitted_fraction": float(st.transmitted_fraction),
            "reference_plane_pulse_energy_uJ": float(st.reference_plane_pulse_energy_uJ),
            "reference_plane_beam_axis_error_um": float(est["beam_axis_error_um"]),
            "ring_centre_x_um": float(est["ring_centre_x_um"]),
            "core_centre_x_um": float(est["core_centre_x_um"]),
            "azimuthal_uniformity": float(est["azimuthal_uniformity"]),
            "ring_fit_quality": float(est["ring_fit_quality"]),
            "residual_deformation": float(cls["residual_shape_deformation_score"]),
            "peak_fluence_per_transmitted_energy": float(energy["peak_to_reference_energy_ratio"]),
            "numerical_reliability": _route_reliability(run.axicon_incidence_metrics, axis),
        })
    return AxiconAlignmentSweepResult(family=family, rows=tuple(rows))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _extent(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())


def _save(fig: Any, output_path: str | Path | None, dpi: int, title: str, stage: str) -> None:
    if output_path is None:
        return
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", metadata={
        "Title": title,
        "stage": stage,
        "final_export_allowed": "False",
        "Description": "Stage 8C.3R.3 route-aware physical-axicon free-space diagnostic; no material response.",
    })


def _card(ax: Any, title: str, lines: list[str], face: str = "#fafafa", edge: str = "#455a64",
          fs: float = 7.4) -> None:
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor=face, edgecolor=edge, lw=1.2, clip_on=False))
    ax.text(0.04, 0.94, title, transform=ax.transAxes, va="top", ha="left",
            fontsize=fs + 1.5, fontweight="bold", color=edge)
    ax.text(0.04, 0.80, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
            fontsize=fs, family="monospace", color="#1a1a1a")


def _badges(fig: Any, y: float = 0.92) -> None:
    for x, txt, ec, fc in [
        (0.045, "DIAGNOSTIC ONLY", "#0d47a1", "#e3f2fd"),
        (0.180, "NO MATERIAL RESPONSE", "#4a148c", "#f3e5f5"),
        (0.365, "FREE-SPACE n=1.0", "#1b5e20", "#e8f5e9"),
        (0.505, "ROUTE-AWARE AXICON", "#bf360c", "#fff3e0"),
    ]:
        fig.text(x, y, txt, ha="left", va="center", fontsize=8.8, fontweight="bold",
                 color=ec, bbox=dict(boxstyle="round,pad=0.25", facecolor=fc, edgecolor=ec, lw=1.0))


def plot_route_aware_axicon_pipeline(
    *,
    config: RouteAwareAxiconConfig | None = None,
    controls: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
    dpi: int = 155,
) -> "matplotlib.figure.Figure":
    run = run_route_aware_axicon_pipeline(controls, config=config or RouteAwareAxiconConfig())
    st = run.propagated_stack
    fl = stack_to_fluence(st)
    x = np.asarray(st.x_um, float)
    z = np.asarray(st.z_um, float)
    ext = _extent(x, x)
    sel = int(np.argmax(st.intensity_zyx.max(axis=(1, 2))))
    y0 = int(np.argmin(np.abs(x)))
    axm = compute_axis_tracking(st, plane_index=sel, core_radius_um=2.0)
    T = physical_axicon_transmission(_grid(run.config), run.config)

    fig = plt.figure(figsize=(17.0, 12.0), facecolor="white")
    gs = fig.add_gridspec(3, 3, left=0.06, right=0.965, top=0.84, bottom=0.07,
                          hspace=0.42, wspace=0.28)
    fig.suptitle("Stage 8C.3R.3 Route-Aware Physical Axicon Pipeline\n"
                 "Perturbations are injected at declared route locations before downstream propagation/elements",
                 x=0.045, y=0.975, ha="left", va="top", fontsize=15.5, fontweight="bold")
    _badges(fig)

    panels = [
        (np.abs(run.source_state.field), "source/input amplitude", "source_plane", "viridis"),
        (np.angle(run.source_state.field), "source/input phase", "source_plane", "twilight"),
        (np.abs(run.axicon_incident_state.field), "field arriving at axicon", "physical_axicon_plane", "viridis"),
        (np.angle(T), "axicon aperture + phase centre", "physical_axicon_plane", "twilight"),
        (np.abs(run.physical_axicon_state.field), "field immediately after axicon", "physical_axicon_plane", "viridis"),
        (fl.fluence_zyx_j_cm2[-1], "reference-plane XY fluence", "free_space_reference_plane", "viridis"),
    ]
    for i, (arr, title, loc, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        im = ax.imshow(arr, origin="lower", extent=ext, cmap=cmap, aspect="equal")
        ax.set_title(f"{title}\n{loc}", fontsize=9.5, fontweight="bold")
        ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
        if "axicon" in title:
            ax.scatter([run.config.physical_axicon_centre_x_um], [run.config.physical_axicon_centre_y_um],
                       c="white", marker="+", s=90, linewidths=1.8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    axxz = fig.add_subplot(gs[2, 0])
    axxz.imshow(fl.fluence_zyx_j_cm2[:, y0, :].T, origin="lower",
                extent=(float(z.min()), float(z.max()), float(x.min()), float(x.max())),
                cmap="viridis", aspect="auto")
    axxz.plot(z, axm["axis_x_by_z_um"], color="#00e5ff", lw=1.8, label="fitted axis")
    axxz.set_title("reference-plane XZ fluence + axis", fontsize=9.5, fontweight="bold")
    axxz.set_xlabel("post-axicon z (um)"); axxz.set_ylabel("x (um)")
    axxz.legend(fontsize=7)

    energies = [
        run.source_state.pulse_energy_before_uJ,
        run.source_state.pulse_energy_after_uJ,
        run.axicon_incident_state.pulse_energy_after_uJ,
        run.physical_axicon_state.pulse_energy_after_uJ,
        st.reference_plane_pulse_energy_uJ,
    ]
    axen = fig.add_subplot(gs[2, 1])
    axen.plot(range(len(energies)), energies, "-o", color="#1565c0", lw=2)
    axen.set_xticks(range(len(energies)))
    axen.set_xticklabels(["input", "src ap", "ax in", "ax out", "ref"], rotation=15, fontsize=8)
    axen.set_ylabel("pulse energy (uJ)")
    axen.set_title("energy-throughput waterfall", fontsize=9.5, fontweight="bold")
    axen.grid(alpha=0.28)

    m = run.axicon_incidence_metrics
    _card(fig.add_subplot(gs[2, 2]), "Commanded vs actual axis / incidence", [
        f"pre distance        : {run.config.pre_axicon_distance_mm:.3f} mm",
        f"post distance       : {run.config.post_axicon_distance_mm:.3f} mm",
        f"incident centroid   : ({m['incident_centroid_x_um']:.2f}, {m['incident_centroid_y_um']:.2f}) um",
        f"incident angle      : {m['incident_angle_x_mrad']:.2f}, {m['incident_angle_y_mrad']:.2f} mrad",
        f"beam-axicon offset  : {m['relative_beam_to_axicon_offset_um']:.2f} um",
        f"axicon overlap      : {m['axicon_aperture_overlap_fraction']:.3f}",
        f"axis error @ ref    : {axm['beam_axis_error_um']:.2f} um",
        "geometry: diagnostic demo, not measured bench",
    ], "#e8f5e9", "#1b5e20")

    fig.c3r3_metadata = {  # type: ignore[attr-defined]
        "stage": "stage8c3r3_route_aware_axicon_pipeline",
        "final_export_allowed": False,
        "route_mode": "physical_axicon",
    }
    _save(fig, output_path, dpi, "Stage 8C.3R.3 Route-Aware Axicon Pipeline",
          "stage8c3r3_route_aware_axicon_pipeline")
    return fig


def plot_component_route_inspection(
    *,
    config: RouteAwareAxiconConfig | None = None,
    controls: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
    dpi: int = 150,
) -> "matplotlib.figure.Figure":
    """Plot a GUI-ready component/segment route-inspection view."""
    run = run_route_aware_axicon_pipeline(controls, config=config or RouteAwareAxiconConfig())
    rows = route_inspection_rows(run)
    fig = plt.figure(figsize=(17.5, 11.2), facecolor="white")
    gs = fig.add_gridspec(2, 2, left=0.045, right=0.97, top=0.82, bottom=0.16,
                          hspace=0.30, wspace=0.22, height_ratios=[1.65, 1.0])
    fig.suptitle("Stage 8C.3 Component-Owned Physical-Axicon Route Scaffold\n"
                 "misalign represented component -> apply local transform -> propagate through downstream elements",
                 x=0.045, y=0.972, ha="left", va="top", fontsize=15.2, fontweight="bold")
    _badges(fig)

    table_rows: list[list[str]] = []
    for r in rows:
        pose = r["actual_pose_error"]
        pose_bits = []
        if abs(float(pose.get("decentre_x_um", 0.0))) > 1e-12 or abs(float(pose.get("decentre_y_um", 0.0))) > 1e-12:
            pose_bits.append(f"dec=({pose['decentre_x_um']:.1f},{pose['decentre_y_um']:.1f})")
        if abs(float(pose.get("axial_offset_um", 0.0))) > 1e-12:
            pose_bits.append(f"dz={pose['axial_offset_um']:.1f}")
        if abs(float(pose.get("tip_x_mrad", 0.0))) > 1e-12 or abs(float(pose.get("tip_y_mrad", 0.0))) > 1e-12:
            pose_bits.append(f"tip=({pose['tip_x_mrad']:.1f},{pose['tip_y_mrad']:.1f})")
        pose_txt = ", ".join(pose_bits) if pose_bits else "nominal"
        cb = r["centroid before (um)"]
        ca = r["centroid after (um)"]
        ab = r["angle before (mrad)"]
        aa = r["angle after (mrad)"]
        overlap = r["aperture overlap"]
        table_rows.append([
            str(r["component_id"]),
            str(r["component_type"]),
            f"{float(r['distance_to_next_element_mm']):.3f}",
            pose_txt,
            f"{float(r['energy before (uJ)']):.2f}->{float(r['energy after (uJ)']):.2f}",
            f"({cb[0]:.1f},{cb[1]:.1f})->({ca[0]:.1f},{ca[1]:.1f})",
            f"({ab[0]:.1f},{ab[1]:.1f})->({aa[0]:.1f},{aa[1]:.1f})",
            "n/a" if overlap is None else f"{float(overlap):.3f}",
            "yes" if bool(r["transform_applied"]) else "no",
            str(r["model status"]),
        ])

    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis("off")
    cols = ["component", "type", "next mm", "pose error", "energy uJ",
            "centroid um", "angle mrad", "overlap", "xform", "status"]
    tbl = ax_table.table(cellText=table_rows, colLabels=cols, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.1)
    tbl.scale(1.0, 1.42)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#b0bec5")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#eceff1")
            cell.set_text_props(weight="bold", color="#263238")
        elif "boundary" in table_rows[row - 1][9]:
            cell.set_facecolor("#fffde7")
        elif "future" in table_rows[row - 1][9] or "warning" in table_rows[row - 1][9]:
            cell.set_facecolor("#ffebee")

    ids = [r["component_id"] for r in rows]
    e_after = np.array([float(r["energy after (uJ)"]) for r in rows], float)
    cx_after = np.array([float(r["centroid after (um)"][0]) for r in rows], float)
    ax0 = fig.add_subplot(gs[1, 0])
    ax0.plot(range(len(ids)), e_after, "-o", color="#1565c0", lw=1.8, ms=4)
    ax0.set_xticks(range(len(ids)))
    ax0.set_xticklabels(ids, rotation=55, ha="right", fontsize=6.7)
    ax0.set_ylabel("energy after stage (uJ)")
    ax0.set_title("cumulative energy through ordered route", fontsize=9.5, fontweight="bold")
    ax0.grid(alpha=0.25)

    ax1 = fig.add_subplot(gs[1, 1])
    ax1.plot(range(len(ids)), cx_after, "-s", color="#00695c", lw=1.8, ms=4)
    ax1.axhline(0.0, color="0.5", lw=0.8)
    ax1.set_xticks(range(len(ids)))
    ax1.set_xticklabels(ids, rotation=55, ha="right", fontsize=6.7)
    ax1.set_ylabel("x centroid after stage (um)")
    ax1.set_title("cumulative centroid consequence", fontsize=9.5, fontweight="bold")
    ax1.grid(alpha=0.25)

    fig.text(0.047, 0.045,
             "Field-state rows are labelled boundary conditions. Active component misalignments are local to represented optics.",
             fontsize=8.8, color="#1b5e20", fontweight="bold")
    fig.c3r3_metadata = {  # type: ignore[attr-defined]
        "stage": "stage8c3_component_owned_route_inspection",
        "final_export_allowed": False,
        "route_mode": "physical_axicon",
    }
    _save(fig, output_path, dpi, "Stage 8C.3 Component-Owned Route Inspection",
          "stage8c3_component_route_inspection")
    return fig


def plot_upstream_vs_post_axicon_tilt_comparison(
    *,
    config: RouteAwareAxiconConfig | None = None,
    tilt_mrad: float = 12.0,
    output_path: str | Path | None = None,
    dpi: int = 155,
) -> "matplotlib.figure.Figure":
    config = config or RouteAwareAxiconConfig()
    runs = [
        (
            "A. before physical axicon path\n(source-plane injection)",
            run_route_aware_axicon_pipeline(
                {"field_tilt_x_mrad": tilt_mrad, "field_tilt_location": "source_plane"},
                config=config,
            ),
            "upstream_field_tilt_location_source_plane",
        ),
        (
            "B. immediately after physical axicon",
            run_route_aware_axicon_pipeline(
                {"field_tilt_x_mrad": tilt_mrad, "field_tilt_location": "after_physical_axicon"},
                config=config,
            ),
            "post_axicon_steering_test",
        ),
        (
            "C. post-axicon diagnostic boundary",
            run_route_aware_axicon_pipeline(
                {"field_tilt_x_mrad": tilt_mrad, "field_tilt_location": "post_axicon_diagnostic_boundary"},
                config=config,
            ),
            "post_axicon_diagnostic_boundary",
        ),
    ]
    fig = plt.figure(figsize=(17.0, 13.2), facecolor="white")
    gs = fig.add_gridspec(3, 4, left=0.06, right=0.965, top=0.84, bottom=0.065,
                          hspace=0.45, wspace=0.28)
    fig.suptitle("Stage 8C.3R.3 Field-Tilt Injection Location Sweep\n"
                 "The same tilt produces different downstream behaviour depending on where it enters the route graph",
                 x=0.045, y=0.972, ha="left", va="top", fontsize=15.2, fontweight="bold")
    _badges(fig)
    for row, (label, run, classification) in enumerate(runs):
        st = run.propagated_stack
        fl = stack_to_fluence(st)
        x = np.asarray(st.x_um, float)
        z = np.asarray(st.z_um, float)
        y0 = int(np.argmin(np.abs(x)))
        ext = _extent(x, x)
        axis = compute_axis_tracking(st, plane_index=int(np.argmax(st.intensity_zyx.max(axis=(1, 2)))), core_radius_um=2.0)

        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(np.abs(run.axicon_incident_state.field), origin="lower", extent=ext, cmap="viridis", aspect="equal")
        ax0.scatter([run.config.physical_axicon_centre_x_um], [run.config.physical_axicon_centre_y_um],
                    c="white", marker="+", s=90, linewidths=1.8)
        ax0.set_title(f"{label}\nfield incident on axicon", fontsize=8.6, fontweight="bold")
        ax0.set_xlabel("x (um)"); ax0.set_ylabel("y (um)")

        ax1 = fig.add_subplot(gs[row, 1])
        ax1.imshow(fl.fluence_zyx_j_cm2[-1], origin="lower", extent=ext, cmap="viridis", aspect="equal")
        ax1.set_title("reference-plane XY fluence", fontsize=8.8, fontweight="bold")
        ax1.set_xlabel("x (um)"); ax1.set_ylabel("y (um)")

        ax2 = fig.add_subplot(gs[row, 2])
        ax2.imshow(fl.fluence_zyx_j_cm2[:, y0, :].T, origin="lower",
                   extent=(float(z.min()), float(z.max()), float(x.min()), float(x.max())),
                   cmap="viridis", aspect="auto")
        ax2.plot(z, axis["axis_x_by_z_um"], color="#00e5ff", lw=1.7)
        ax2.set_title("XZ propagation + axis trajectory", fontsize=8.8, fontweight="bold")
        ax2.set_xlabel("post-axicon z (um)"); ax2.set_ylabel("x (um)")

        m = run.axicon_incidence_metrics
        _card(fig.add_subplot(gs[row, 3]), "Classification / overlap / energy", [
            f"tilt location       : {m['field_tilt_injection_location']}",
            f"classification       : {classification}",
            f"incident centroid   : ({m['incident_centroid_x_um']:.2f}, {m['incident_centroid_y_um']:.2f}) um",
            f"pred walkoff        : {m['upstream_tilt_predicted_walkoff_um']:.2f} um",
            f"meas walkoff        : {m['upstream_tilt_measured_walkoff_um']:.2f} um",
            f"walkoff error       : {m['walkoff_model_error_um']:.2f} um",
            f"axicon overlap      : {m['axicon_aperture_overlap_fraction']:.3f}",
            f"transmitted frac    : {st.transmitted_fraction:.3f}",
            f"axis x steering     : {axis['beam_steering_angle_x_mrad']:.2f} mrad",
            "same type, different injection location",
        ], "#fffde7" if row == 0 else ("#e3f2fd" if row == 1 else "#e8f5e9"),
            "#f57f17" if row == 0 else ("#0d47a1" if row == 1 else "#1b5e20"))

    fig.c3r3_metadata = {  # type: ignore[attr-defined]
        "stage": "stage8c3r3_upstream_vs_post_axicon_tilt_comparison",
        "tilt_mrad": float(tilt_mrad),
        "comparison": "field_tilt_injection_location_sweep",
        "final_export_allowed": False,
    }
    _save(fig, output_path, dpi, "Stage 8C.3R.3 Field-Tilt Injection Location Sweep",
          "stage8c3r3_upstream_vs_post_axicon_tilt_comparison")
    return fig


def plot_axicon_alignment_sensitivity_atlas(
    *,
    config: RouteAwareAxiconConfig | None = None,
    sweep_results: Mapping[str, AxiconAlignmentSweepResult] | None = None,
    output_path: str | Path | None = None,
    dpi: int = 150,
) -> "matplotlib.figure.Figure":
    config = config or RouteAwareAxiconConfig.fast()
    families = build_axicon_alignment_sweep_families()
    results = dict(sweep_results or {})
    for key in families:
        if key not in results:
            results[key] = run_axicon_alignment_sweep(key, config=config)

    fig = plt.figure(figsize=(16.0, 11.2), facecolor="white")
    gs = fig.add_gridspec(3, 2, left=0.065, right=0.965, top=0.84, bottom=0.08,
                          hspace=0.55, wspace=0.32)
    fig.suptitle("Stage 8C.3R.3 Physical-Axicon Alignment Sensitivity Atlas\n"
                 + DIAGNOSTIC_SWEEP_LABEL,
                 x=0.045, y=0.972, ha="left", va="top", fontsize=15.2, fontweight="bold")
    _badges(fig)

    for i, key in enumerate(families):
        res = results[key]
        rows = list(res.rows)
        xv = np.array([r["param_value"] for r in rows], float)
        walk = np.array([r["axicon_plane_walkoff_um"] for r in rows], float)
        overlap = np.array([r["axicon_aperture_overlap_fraction"] for r in rows], float)
        trans = np.array([r["transmitted_fraction"] for r in rows], float)
        axis = np.array([r["reference_plane_beam_axis_error_um"] for r in rows], float)
        resid = np.array([r["residual_deformation"] for r in rows], float)
        fitq = np.array([r["ring_fit_quality"] for r in rows], float)
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax2 = ax.twinx()
        ax.plot(xv, walk, "-o", color="#1565c0", label="axicon walkoff (um)", lw=1.8, ms=4)
        ax.plot(xv, axis, "-s", color="#00838f", label="ref axis error (um)", lw=1.6, ms=3.5)
        ax2.plot(xv, overlap, "-^", color="#2e7d32", label="axicon overlap", lw=1.5, ms=3.5)
        ax2.plot(xv, trans, "-d", color="#ef6c00", label="transmitted fraction", lw=1.4, ms=3.5)
        ax2.plot(xv, resid, "-v", color="#6a1b9a", label="residual deformation", lw=1.3, ms=3.4)
        ax2.plot(xv, fitq, ":", color="#455a64", label="ring fit Q", lw=1.5)
        ax.set_title(res.family.title, fontsize=10.0, fontweight="bold")
        ax.set_xlabel(res.family.param_label, fontsize=8.5)
        ax.set_ylabel("um", fontsize=8.3, color="#1565c0")
        ax2.set_ylabel("fraction / score", fontsize=8.3)
        ax.grid(alpha=0.25)
        ax2.set_ylim(0.0, 1.05)
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [ln.get_label() for ln in lines], fontsize=6.8, loc="best", framealpha=0.82)
        ax.text(0.98, 0.04,
                f"last: U {rows[-1]['azimuthal_uniformity']:.2f}, "
                f"peak/E {rows[-1]['peak_fluence_per_transmitted_energy']:.2f}\n"
                f"{rows[-1]['numerical_reliability']}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.0,
                bbox=dict(facecolor="white", edgecolor="0.72", alpha=0.84, pad=2.5))

    fig.text(0.066, 0.035,
             "All curves are route-aware free-space diagnostics. They are not experimentally measured laboratory tolerances.",
             fontsize=9.0, color="#1b5e20", fontweight="bold")
    fig.c3r3_metadata = {  # type: ignore[attr-defined]
        "stage": "stage8c3r3_axicon_alignment_sensitivity_atlas",
        "families": list(families),
        "diagnostic_label": DIAGNOSTIC_SWEEP_LABEL,
        "final_export_allowed": False,
    }
    _save(fig, output_path, dpi, "Stage 8C.3R.3 Axicon Alignment Sensitivity Atlas",
          "stage8c3r3_axicon_alignment_sensitivity_atlas")
    return fig
