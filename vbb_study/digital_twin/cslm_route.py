"""Stage 8C.3R.5 component-owned concatenated-SLM route scaffold.

This module represents the programmable CSLM route as named component planes
and propagation segments while keeping the hard Stage 8C boundary:

    free-space optical-field and fluence diagnostics only
    n = 1.0
    diagnostic_only
    final_export_allowed = False

Active physics is limited to source complex field construction, SLM1
vortex/phase-only conditioning, SLM2 correction/carrier phase handling, phase
wrapping/quantisation where supported, and angular-spectrum free-space
propagation.  SLM2 does not produce an axicon phase in this route.  The real
laboratory 4F first-order filter is declared as part of the route contract, but
remains warning-only here because this code does not yet have a validated
lens/Fourier-plane/filter coordinate model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

import matplotlib
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402

from vbb_study.equations.fields import fft2c, make_xy_grid
from vbb_study.equations.holography import (
    TWOPI,
    blazed_carrier_phase_rad,
    quantize_phase_rad,
    spp_phase_rad,
    wrap_phase_rad,
)
from vbb_study.equations.propagation import make_bl_asm_propagator
from vbb_study.digital_twin.component_plane_metrics import stack_to_fluence
from vbb_study.digital_twin.component_plane_states import (
    ComponentPlaneState,
    PropagatedFieldStack,
    field_power,
)
from vbb_study.digital_twin.annular_axis_tracking import estimate_annular_axis
from vbb_study.digital_twin.route_aware_axicon import (
    BeamlineComponent,
    ComponentPose,
    RouteInspectionRecord,
    RouteAwareAxiconConfig,
    physical_axicon_transmission,
)

_UM = 1e-6
_MM = 1e-3
EPS = 1e-30

FINAL_EXPORT_ALLOWED = False
MODEL_STATUS = "optical_prediction"
FIGURE_STATUS = "diagnostic_only"
DIAGNOSTIC_GEOMETRY_NOTE = (
    "diagnostic demo geometry only; not measured laboratory geometry"
)

CONCEPTUAL_CSLM_COMPONENT_IDS: tuple[str, ...] = (
    "source_field",
    "input_conditioning_boundary",
    "SLM1_phase_plane",
    "SLM1_to_SLM2_segment",
    "SLM2_phase_plane",
    "SLM2_to_pre_4F_diagnostic_segment",
    "post_SLM2_pre_4F_diagnostic_plane",
    "SLM2_to_fourier_lens_segment",
    "Fourier_lens_1",
    "Fourier_plane",
    "plus_one_order_filter",
    "Fourier_lens_2",
    "4F_output_plane",
)

EXECUTED_CSLM_COMPONENT_IDS: tuple[str, ...] = (
    "source_field",
    "input_conditioning_boundary",
    "SLM1_phase_plane",
    "SLM1_to_SLM2_segment",
    "SLM2_phase_plane",
    "SLM2_to_pre_4F_diagnostic_segment",
    "post_SLM2_pre_4F_diagnostic_plane",
)

WARNING_ONLY_CSLM_COMPONENT_IDS: tuple[str, ...] = (
    "SLM2_to_fourier_lens_segment",
    "Fourier_lens_1",
    "Fourier_plane",
    "plus_one_order_filter",
    "Fourier_lens_2",
    "4F_output_plane",
)

IDEAL_AXICON_BENCHMARK_COMPONENT_IDS: tuple[str, ...] = (
    "ideal_selected_order_handoff_plane",
    "physical_axicon_benchmark_plane",
    "post_axicon_benchmark_segment",
    "ideal_axicon_benchmark_reference_plane",
)

OrderHandoffMode = Literal[
    "none",
    "ideal_selected_order_surrogate",
    "physical_4f_filter",
]

FOUR_F_REQUIRED_PARAMETERS: tuple[str, ...] = (
    "wavelength_nm",
    "slm_pixel_pitch_um_or_continuous_spatial_equivalent",
    "slm2_carrier_frequency_cpm",
    "fourier_lens1_focal_length_mm",
    "fourier_lens2_focal_length_mm",
    "slm2_to_lens1_distance_mm",
    "lens1_to_fourier_plane_distance_mm",
    "fourier_plane_to_lens2_distance_mm",
    "lens2_to_output_plane_distance_mm",
    "fourier_plane_coordinate_mapping",
    "filter_mask_centre_x_um",
    "filter_mask_centre_y_um",
    "filter_mask_radius_um",
    "filter_mask_shape",
)

FOUR_F_BLOCKING_MODEL_GAPS: tuple[str, ...] = (
    "no component-owned thin-lens transform is currently represented in the CSLM route",
    "no validated SLM2 carrier-to-Fourier-plane coordinate mapping is implemented",
    "no physically located Fourier-plane filter mask is applied to a field",
    "no lens-separation tolerance or measured relay geometry is consumed",
    "no separate zero/+1/residual order energy measurement exists in the current engine",
)


@dataclass(frozen=True)
class CSLMRouteConfig:
    """Editable component-owned CSLM baseline route controls."""

    route_mode: str = "holographic_cslm"
    wavelength_nm: float = 1030.0
    n_medium: float = 1.0
    grid_N: int = 192
    dx_um: float = 0.5
    n_z: int = 36
    z_max_um: float = 250.0
    input_pulse_energy_uJ: float = 95.76
    input_beam_radius_um: float = 24.0

    slm1_phase_mode: str = "vortex"
    slm1_topological_charge: int = 3
    slm1_linear_ramp_cpm: float = 0.0
    slm1_to_slm2_distance_mm: float = 0.040

    slm2_conjugate_mode: str = "preserve_vortex"
    slm2_carrier_frequency_cpm: float = 30_000.0
    # Uniform piston placeholder (a single scalar broadcast to a uniform phase
    # offset).  It is NOT an aberration-correction map.  A true future correction
    # requires a spatial phase map (Zernike coefficient map / imported phase
    # array / calibrated correction mask); that system is not implemented here.
    slm2_correction_phase_rad: float = 0.0
    slm_phase_quantisation_levels: int = 256
    slm2_to_pre_4f_diagnostic_distance_mm: float = 0.120

    order_handoff_mode: OrderHandoffMode = "none"
    physical_axicon_enabled_for_benchmark: bool = True
    physical_axicon_centre_x_um: float = 0.0
    physical_axicon_centre_y_um: float = 0.0
    physical_axicon_clear_aperture_radius_um: float = 42.0
    physical_axicon_cone_parameter_rad_per_um: float = 1.05
    physical_axicon_to_benchmark_reference_distance_mm: float = 0.120

    fourier_lens1_focal_length_mm: float = 100.0
    fourier_lens2_focal_length_mm: float = 100.0
    slm2_to_lens1_distance_mm: float = 100.0
    lens1_to_fourier_plane_distance_mm: float = 100.0
    fourier_plane_to_lens2_distance_mm: float = 100.0
    lens2_to_output_plane_distance_mm: float = 100.0
    fourier_filter_centre_x_um: float = 0.0
    fourier_filter_centre_y_um: float = 0.0
    fourier_filter_radius_um: float = 250.0
    fourier_filter_shape: str = "circular"

    bandlimit: bool = True

    @classmethod
    def fast(cls, **overrides: Any) -> "CSLMRouteConfig":
        """Smaller grid for tests and quick previews."""

        base = dict(grid_N=112, dx_um=0.7, n_z=18, input_beam_radius_um=20.0)
        base.update(overrides)
        return cls(**base)

    @property
    def wavelength_m(self) -> float:
        return float(self.wavelength_nm) * 1e-9

    @property
    def dx_m(self) -> float:
        return float(self.dx_um) * _UM

    @property
    def slm1_to_slm2_distance_um(self) -> float:
        return float(self.slm1_to_slm2_distance_mm) * 1000.0

    @property
    def slm2_to_pre_4f_diagnostic_distance_um(self) -> float:
        return float(self.slm2_to_pre_4f_diagnostic_distance_mm) * 1000.0

    @property
    def physical_axicon_to_benchmark_reference_distance_um(self) -> float:
        return float(self.physical_axicon_to_benchmark_reference_distance_mm) * 1000.0

    @property
    def slm2_carrier_frequency_cycles_per_mm(self) -> float:
        return float(self.slm2_carrier_frequency_cpm) / 1000.0

    @property
    def carrier_spatial_period_um(self) -> float:
        cpm = abs(float(self.slm2_carrier_frequency_cpm))
        if cpm <= EPS:
            return float("inf")
        return 1.0e6 / cpm


@dataclass(frozen=True)
class FourFFeasibilityAudit:
    """Dimensionally explicit audit before any 4F filter field is allowed."""

    fourier_filter_physics_available: bool
    required_parameters: tuple[str, ...]
    present_parameters: Mapping[str, Any]
    missing_or_unvalidated_parameters: tuple[str, ...]
    blocking_model_gaps: tuple[str, ...]
    warning_only_components: tuple[str, ...]
    order_selection_result: Mapping[str, Any]
    note: str
    final_export_allowed: bool = FINAL_EXPORT_ALLOWED
    model_status: str = FIGURE_STATUS

    def as_dict(self) -> dict[str, Any]:
        return {
            "fourier_filter_physics_available": bool(self.fourier_filter_physics_available),
            "required_parameters": list(self.required_parameters),
            "present_parameters": dict(self.present_parameters),
            "missing_or_unvalidated_parameters": list(self.missing_or_unvalidated_parameters),
            "blocking_model_gaps": list(self.blocking_model_gaps),
            "warning_only_components": list(self.warning_only_components),
            "order_selection_result": dict(self.order_selection_result),
            "note": self.note,
            "final_export_allowed": bool(self.final_export_allowed),
            "model_status": self.model_status,
        }


@dataclass(frozen=True)
class OrderSelectionHandoff:
    """Explicit provenance for any order-selection handoff branch."""

    mode: OrderHandoffMode
    input_component_id: str
    output_component_id: str
    physical_filter_modelled: bool
    energy_selection_modelled: bool
    order_efficiency_modelled: bool
    zero_order_rejection_modelled: bool
    field_provenance: str
    claim_boundary: str
    warnings: tuple[str, ...]
    final_export_allowed: bool = FINAL_EXPORT_ALLOWED

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "input_component_id": self.input_component_id,
            "output_component_id": self.output_component_id,
            "physical_filter_modelled": bool(self.physical_filter_modelled),
            "energy_selection_modelled": bool(self.energy_selection_modelled),
            "order_efficiency_modelled": bool(self.order_efficiency_modelled),
            "zero_order_rejection_modelled": bool(self.zero_order_rejection_modelled),
            "field_provenance": self.field_provenance,
            "claim_boundary": self.claim_boundary,
            "warnings": list(self.warnings),
            "final_export_allowed": bool(self.final_export_allowed),
        }


@dataclass(frozen=True)
class PhysicalAxiconBenchmarkResult:
    """Benchmark-only physical-axicon branch output."""

    handoff: OrderSelectionHandoff
    ideal_selected_order_state: ComponentPlaneState
    physical_axicon_state: ComponentPlaneState
    benchmark_reference_state: ComponentPlaneState
    benchmark_stack: PropagatedFieldStack
    benchmark_route_chain: tuple[str, ...]
    transmission: np.ndarray
    metrics: Mapping[str, Any]
    final_export_allowed: bool = FINAL_EXPORT_ALLOWED
    model_status: str = "benchmark_only"

    def as_dict(self) -> dict[str, Any]:
        return {
            "handoff": self.handoff.as_dict(),
            "benchmark_route_chain": list(self.benchmark_route_chain),
            "metrics": dict(self.metrics),
            "final_export_allowed": bool(self.final_export_allowed),
            "model_status": self.model_status,
        }


@dataclass(frozen=True)
class CSLMRouteRun:
    """Full CSLM baseline route output."""

    config: CSLMRouteConfig
    route_declaration: tuple[BeamlineComponent, ...]
    executed_components: tuple[BeamlineComponent, ...]
    inspection_records: tuple[RouteInspectionRecord, ...]
    source_state: ComponentPlaneState
    slm1_state: ComponentPlaneState
    slm2_input_state: ComponentPlaneState
    slm2_state: ComponentPlaneState
    reference_plane_state: ComponentPlaneState
    propagated_stack: PropagatedFieldStack
    slm1_phase_rad: np.ndarray
    slm2_phase_terms_rad: Mapping[str, np.ndarray]
    slm2_composite_phase_rad: np.ndarray
    slm2_wrapped_phase_rad: np.ndarray
    slm2_quantized_phase_rad: np.ndarray
    post_slm2_unfiltered_state: ComponentPlaneState
    order_handoff: OrderSelectionHandoff
    ideal_selected_order_surrogate_state: ComponentPlaneState | None
    axicon_benchmark: PhysicalAxiconBenchmarkResult | None
    baseline_fields: Mapping[str, Any]
    baseline_metrics: Mapping[str, Any]
    fourier_feasibility: FourFFeasibilityAudit
    warnings: tuple[str, ...]
    model_status: str = MODEL_STATUS
    final_export_allowed: bool = FINAL_EXPORT_ALLOWED

    @property
    def fourier_filter_physics_available(self) -> bool:
        return bool(self.fourier_feasibility.fourier_filter_physics_available)

    @property
    def executed_route_chain(self) -> tuple[str, ...]:
        return tuple(c.component_id for c in self.executed_components)

    @property
    def conceptual_route_chain(self) -> tuple[str, ...]:
        return tuple(c.component_id for c in self.route_declaration)

    @property
    def ideal_benchmark_route_chain(self) -> tuple[str, ...]:
        if self.axicon_benchmark is None:
            return ()
        return self.axicon_benchmark.benchmark_route_chain

    @property
    def order_selection_result(self) -> Mapping[str, Any]:
        return self.fourier_feasibility.order_selection_result


def _downstream(ids: tuple[str, ...], component_id: str) -> tuple[str, ...]:
    try:
        idx = ids.index(component_id)
    except ValueError:
        return ()
    return ids[idx + 1 :]


def _component(
    *,
    component_id: str,
    component_type: str,
    physical_location: str,
    nominal_z_position_um: float,
    distance_from_previous_component_mm: float,
    distance_to_next_element_mm: float,
    status: str,
    represented: bool,
    model_available: bool,
    parameters: Mapping[str, Any] | None = None,
    clear_aperture: Mapping[str, Any] | None = None,
    note: str,
    downstream_route: tuple[str, ...] = CONCEPTUAL_CSLM_COMPONENT_IDS,
) -> BeamlineComponent:
    return BeamlineComponent(
        component_id=component_id,
        component_type=component_type,
        physical_location=physical_location,
        nominal_z_position_um=float(nominal_z_position_um),
        distance_from_previous_component_mm=float(distance_from_previous_component_mm),
        distance_to_next_element_mm=float(distance_to_next_element_mm),
        enabled=True,
        physical_pose=ComponentPose(),
        component_specific_parameters=dict(parameters or {}),
        clear_aperture=dict(clear_aperture or {}),
        status=status,
        represented_by_current_engine=bool(represented),
        physical_model_available=bool(model_available),
        misalignment_modes_currently_supported=(),
        downstream_elements_affected=_downstream(downstream_route, component_id),
        note=note,
    )


def build_cslm_route_declaration(
    config: CSLMRouteConfig | None = None,
) -> tuple[BeamlineComponent, ...]:
    """Return the complete conceptual CSLM route declaration."""

    cfg = config or CSLMRouteConfig()
    z_slm1 = 0.0
    z_slm2 = cfg.slm1_to_slm2_distance_um
    z_lens1 = z_slm2 + cfg.slm2_to_lens1_distance_mm * 1000.0
    z_fourier = z_lens1 + cfg.lens1_to_fourier_plane_distance_mm * 1000.0
    z_lens2 = z_fourier + cfg.fourier_plane_to_lens2_distance_mm * 1000.0
    z_4f_out = z_lens2 + cfg.lens2_to_output_plane_distance_mm * 1000.0
    z_diag = z_slm2 + cfg.slm2_to_pre_4f_diagnostic_distance_um
    diag_to_lens1_mm = max(
        float(cfg.slm2_to_lens1_distance_mm) - float(cfg.slm2_to_pre_4f_diagnostic_distance_mm),
        0.0,
    )

    common_clear = {
        "aperture_model": "not applied in R5 baseline",
        "aperture_overlap": None,
    }

    return (
        _component(
            component_id="source_field",
            component_type="complex_field_state",
            physical_location="conditioned_input_field",
            nominal_z_position_um=0.0,
            distance_from_previous_component_mm=0.0,
            distance_to_next_element_mm=0.0,
            status="diagnostic_active",
            represented=True,
            model_available=True,
            parameters={
                "wavelength_nm": cfg.wavelength_nm,
                "n_medium": cfg.n_medium,
                "input_beam_radius_um": cfg.input_beam_radius_um,
            },
            clear_aperture=common_clear,
            note="source complex field in free space; diagnostic demo geometry",
        ),
        _component(
            component_id="input_conditioning_boundary",
            component_type="diagnostic_boundary",
            physical_location="before_SLM1",
            nominal_z_position_um=0.0,
            distance_from_previous_component_mm=0.0,
            distance_to_next_element_mm=0.0,
            status="diagnostic_active",
            represented=True,
            model_available=True,
            parameters={"boundary_condition": "identity"},
            clear_aperture=common_clear,
            note="named field-state boundary; no physical transform in the baseline",
        ),
        _component(
            component_id="SLM1_phase_plane",
            component_type="programmable_phase_plane",
            physical_location="SLM1_plane",
            nominal_z_position_um=z_slm1,
            distance_from_previous_component_mm=0.0,
            distance_to_next_element_mm=cfg.slm1_to_slm2_distance_mm,
            status="physics_active",
            represented=True,
            model_available=True,
            parameters={
                "slm1_role": slm1_role_for_config(cfg),
                "slm1_phase_mode": cfg.slm1_phase_mode,
                "topological_charge": cfg.slm1_topological_charge,
                "linear_ramp_cpm": cfg.slm1_linear_ramp_cpm,
                "phase_quantisation_levels": cfg.slm_phase_quantisation_levels,
            },
            clear_aperture=common_clear,
            note="phase-only vortex/conditioning plane; no independently validated amplitude shaping claim",
        ),
        _component(
            component_id="SLM1_to_SLM2_segment",
            component_type="free_space_propagation_segment",
            physical_location="between_SLM1_and_SLM2",
            nominal_z_position_um=0.5 * z_slm2,
            distance_from_previous_component_mm=0.0,
            distance_to_next_element_mm=cfg.slm1_to_slm2_distance_mm,
            status="physics_active",
            represented=True,
            model_available=True,
            parameters={"n_medium": cfg.n_medium, "distance_mm": cfg.slm1_to_slm2_distance_mm},
            clear_aperture=common_clear,
            note=DIAGNOSTIC_GEOMETRY_NOTE,
        ),
        _component(
            component_id="SLM2_phase_plane",
            component_type="phase_correction_carrier_plane",
            physical_location="SLM2_plane",
            nominal_z_position_um=z_slm2,
            distance_from_previous_component_mm=cfg.slm1_to_slm2_distance_mm,
            distance_to_next_element_mm=cfg.slm2_to_pre_4f_diagnostic_distance_mm,
            status="physics_active",
            represented=True,
            model_available=True,
            parameters={
                "slm2_role": "phase_correction_and_carrier_preserve_vortex",
                "conjugate_mode": cfg.slm2_conjugate_mode,
                "carrier_frequency_cpm": cfg.slm2_carrier_frequency_cpm,
                "carrier_frequency_cycles_per_mm": cfg.slm2_carrier_frequency_cycles_per_mm,
                "phase_quantisation_levels": cfg.slm_phase_quantisation_levels,
                "axicon_phase_produced_here": False,
            },
            clear_aperture=common_clear,
            note="SLM2 does not produce the axicon phase; it carries correction/carrier terms only",
        ),
        _component(
            component_id="SLM2_to_pre_4F_diagnostic_segment",
            component_type="free_space_propagation_segment",
            physical_location="SLM2_to_post_SLM2_pre_4F_diagnostic",
            nominal_z_position_um=z_slm2 + 0.5 * cfg.slm2_to_pre_4f_diagnostic_distance_um,
            distance_from_previous_component_mm=0.0,
            distance_to_next_element_mm=cfg.slm2_to_pre_4f_diagnostic_distance_mm,
            status="physics_active",
            represented=True,
            model_available=True,
            parameters={"n_medium": cfg.n_medium, "distance_mm": cfg.slm2_to_pre_4f_diagnostic_distance_mm},
            clear_aperture=common_clear,
            note="active unfiltered CSLM diagnostic propagation; not a final lab output",
        ),
        _component(
            component_id="post_SLM2_pre_4F_diagnostic_plane",
            component_type="post_slm2_unfiltered_diagnostic_plane",
            physical_location="post_SLM2_pre_4F_diagnostic_plane",
            nominal_z_position_um=z_diag,
            distance_from_previous_component_mm=cfg.slm2_to_pre_4f_diagnostic_distance_mm,
            distance_to_next_element_mm=diag_to_lens1_mm,
            status="diagnostic_active",
            represented=True,
            model_available=True,
            parameters={
                "n_medium": 1.0,
                "post_slm2_unfiltered_diagnostic_field": True,
                "contains_carrier": abs(float(cfg.slm2_carrier_frequency_cpm)) > 0,
                "not_final_lab_output": True,
                "diagnostic_only": True,
                "final_export_allowed": False,
            },
            clear_aperture=common_clear,
            note="post-SLM2 unfiltered field before any physical 4F/order-selection model",
        ),
        _component(
            component_id="SLM2_to_fourier_lens_segment",
            component_type="free_space_segment_to_4F",
            physical_location="post_SLM2_pre_4F_to_lens1",
            nominal_z_position_um=0.5 * (z_diag + z_lens1),
            distance_from_previous_component_mm=cfg.slm2_to_pre_4f_diagnostic_distance_mm,
            distance_to_next_element_mm=diag_to_lens1_mm,
            status="warning_only",
            represented=False,
            model_available=False,
            parameters={"distance_mm": diag_to_lens1_mm},
            clear_aperture=common_clear,
            note="declared laboratory route segment; not executed because 4F path is not represented",
        ),
        _component(
            component_id="Fourier_lens_1",
            component_type="thin_lens",
            physical_location="4F_lens1_plane",
            nominal_z_position_um=z_lens1,
            distance_from_previous_component_mm=cfg.slm2_to_lens1_distance_mm,
            distance_to_next_element_mm=cfg.lens1_to_fourier_plane_distance_mm,
            status="warning_only",
            represented=False,
            model_available=False,
            parameters={"focal_length_mm": cfg.fourier_lens1_focal_length_mm},
            clear_aperture=common_clear,
            note="lens transform not yet component-owned in this engine",
        ),
        _component(
            component_id="Fourier_plane",
            component_type="fourier_plane",
            physical_location="4F_fourier_plane",
            nominal_z_position_um=z_fourier,
            distance_from_previous_component_mm=cfg.lens1_to_fourier_plane_distance_mm,
            distance_to_next_element_mm=0.0,
            status="warning_only",
            represented=False,
            model_available=False,
            parameters={
                "coordinate_mapping": "not implemented",
                "order_position_source": "carrier frequency contract only",
            },
            clear_aperture=common_clear,
            note="physical Fourier-plane coordinates are not available in the current route",
        ),
        _component(
            component_id="plus_one_order_filter",
            component_type="fourier_plane_spatial_filter",
            physical_location="4F_fourier_plane",
            nominal_z_position_um=z_fourier,
            distance_from_previous_component_mm=0.0,
            distance_to_next_element_mm=cfg.fourier_plane_to_lens2_distance_mm,
            status="warning_only",
            represented=False,
            model_available=False,
            parameters={
                "order_label": "+1",
                "filter_centre_x_um": cfg.fourier_filter_centre_x_um,
                "filter_centre_y_um": cfg.fourier_filter_centre_y_um,
                "filter_radius_um": cfg.fourier_filter_radius_um,
                "filter_shape": cfg.fourier_filter_shape,
            },
            clear_aperture={"filter_mask_declared": True, "filter_mask_applied": False},
            note="declared but not applied; no fake filtered output field is generated",
        ),
        _component(
            component_id="Fourier_lens_2",
            component_type="thin_lens",
            physical_location="4F_lens2_plane",
            nominal_z_position_um=z_lens2,
            distance_from_previous_component_mm=cfg.fourier_plane_to_lens2_distance_mm,
            distance_to_next_element_mm=cfg.lens2_to_output_plane_distance_mm,
            status="warning_only",
            represented=False,
            model_available=False,
            parameters={"focal_length_mm": cfg.fourier_lens2_focal_length_mm},
            clear_aperture=common_clear,
            note="second 4F lens remains future/not represented",
        ),
        _component(
            component_id="4F_output_plane",
            component_type="4F_image_plane",
            physical_location="4F_output_plane",
            nominal_z_position_um=z_4f_out,
            distance_from_previous_component_mm=cfg.lens2_to_output_plane_distance_mm,
            distance_to_next_element_mm=0.0,
            status="future_not_implemented",
            represented=False,
            model_available=False,
            parameters={"output_field": "not generated"},
            clear_aperture=common_clear,
            note="no 4F output field until lens/filter mapping is implemented and validated",
        ),
    )


def build_executed_cslm_component_chain(
    config: CSLMRouteConfig | None = None,
) -> tuple[BeamlineComponent, ...]:
    """Return the executable subset of the CSLM route."""

    cfg = config or CSLMRouteConfig()
    declaration = {c.component_id: c for c in build_cslm_route_declaration(cfg)}
    z_slm2 = cfg.slm1_to_slm2_distance_um
    segment = _component(
        component_id="SLM2_to_pre_4F_diagnostic_segment",
        component_type="free_space_propagation_segment",
        physical_location="SLM2_to_post_SLM2_pre_4F_diagnostic",
        nominal_z_position_um=z_slm2 + 0.5 * cfg.slm2_to_pre_4f_diagnostic_distance_um,
        distance_from_previous_component_mm=0.0,
        distance_to_next_element_mm=cfg.slm2_to_pre_4f_diagnostic_distance_mm,
        status="physics_active",
        represented=True,
        model_available=True,
        parameters={"n_medium": cfg.n_medium, "distance_mm": cfg.slm2_to_pre_4f_diagnostic_distance_mm},
        clear_aperture={"aperture_model": "not applied in R5 baseline", "aperture_overlap": None},
        note="active unfiltered CSLM diagnostic propagation; not a final lab output",
        downstream_route=EXECUTED_CSLM_COMPONENT_IDS,
    )
    plane = _component(
        component_id="post_SLM2_pre_4F_diagnostic_plane",
        component_type="post_slm2_unfiltered_diagnostic_plane",
        physical_location="post_SLM2_pre_4F_diagnostic_plane",
        nominal_z_position_um=z_slm2 + cfg.slm2_to_pre_4f_diagnostic_distance_um,
        distance_from_previous_component_mm=cfg.slm2_to_pre_4f_diagnostic_distance_mm,
        distance_to_next_element_mm=0.0,
        status="diagnostic_active",
        represented=True,
        model_available=True,
        parameters={
            "n_medium": 1.0,
            "post_slm2_unfiltered_diagnostic_field": True,
            "not_final_lab_output": True,
            "diagnostic_only": True,
            "final_export_allowed": False,
        },
        clear_aperture={"aperture_model": "not applied in R5 baseline", "aperture_overlap": None},
        note="post-SLM2 unfiltered field before any physical 4F/order-selection model",
        downstream_route=EXECUTED_CSLM_COMPONENT_IDS,
    )
    execution_map = {
        **declaration,
        "SLM2_to_pre_4F_diagnostic_segment": segment,
        "post_SLM2_pre_4F_diagnostic_plane": plane,
    }
    return tuple(execution_map[name] for name in EXECUTED_CSLM_COMPONENT_IDS)


def slm1_role_for_config(config: CSLMRouteConfig) -> str:
    """Return the honest role label for SLM1."""

    if str(config.slm1_phase_mode).lower() in {"vortex", "flat", "zero", "linear_ramp"}:
        return "phase_only_conditioning"
    return "holographic_field_shaping_unvalidated"


def evaluate_fourier_filter_feasibility(
    config: CSLMRouteConfig | None = None,
) -> FourFFeasibilityAudit:
    """Audit whether a real 4F +1 filter can be executed."""

    cfg = config or CSLMRouteConfig()
    nyquist_cpm = 0.5 / cfg.dx_m
    carrier_cpm = float(cfg.slm2_carrier_frequency_cpm)
    order_angle_rad = np.nan
    if abs(carrier_cpm) < nyquist_cpm and abs(carrier_cpm) > EPS:
        order_angle_rad = float(np.arcsin(np.clip(cfg.wavelength_m * carrier_cpm, -1.0, 1.0)))

    present = {
        "wavelength_nm": cfg.wavelength_nm,
        "grid_dx_um_continuous_spatial_equivalent": cfg.dx_um,
        "slm2_carrier_frequency_cpm": carrier_cpm,
        "carrier_frequency_cycles_per_mm": cfg.slm2_carrier_frequency_cycles_per_mm,
        "carrier_spatial_period_um": cfg.carrier_spatial_period_um,
        "grid_nyquist_cpm": nyquist_cpm,
        "fourier_lens1_focal_length_mm": cfg.fourier_lens1_focal_length_mm,
        "fourier_lens2_focal_length_mm": cfg.fourier_lens2_focal_length_mm,
        "slm2_to_lens1_distance_mm": cfg.slm2_to_lens1_distance_mm,
        "lens1_to_fourier_plane_distance_mm": cfg.lens1_to_fourier_plane_distance_mm,
        "fourier_plane_to_lens2_distance_mm": cfg.fourier_plane_to_lens2_distance_mm,
        "lens2_to_output_plane_distance_mm": cfg.lens2_to_output_plane_distance_mm,
        "declared_filter_centre_um": (
            cfg.fourier_filter_centre_x_um,
            cfg.fourier_filter_centre_y_um,
        ),
        "declared_filter_radius_um": cfg.fourier_filter_radius_um,
        "declared_filter_shape": cfg.fourier_filter_shape,
        "carrier_order_angle_rad_from_grating_equation": order_angle_rad,
    }
    missing = (
        "measured SLM pixel pitch and active area for the actual device",
        "validated thin-lens operator attached to component planes",
        "carrier-order position mapping in Fourier-plane microns",
        "measured Fourier filter centre and radius in the 4F plane",
        "bench-measured SLM2/lens/filter/lens/output separations",
        "order-resolved energy calibration for zero, +1, and residual light",
    )
    order_selection = {
        "filter_executed": False,
        "filtered_field": None,
        "zero_order_energy_fraction": None,
        "selected_plus_one_order_energy_fraction": None,
        "rejected_energy_fraction": None,
        "carrier_frequency_cpm": carrier_cpm,
        "carrier_frequency_cycles_per_mm": cfg.slm2_carrier_frequency_cycles_per_mm,
        "carrier_spatial_period_um": cfg.carrier_spatial_period_um,
        "order_angle_rad_from_grating_equation": order_angle_rad,
        "spatial_units": "cycles/m for carrier; microns for declared filter plane",
        "status": "feasibility_only_no_filtered_field",
    }
    return FourFFeasibilityAudit(
        fourier_filter_physics_available=False,
        required_parameters=FOUR_F_REQUIRED_PARAMETERS,
        present_parameters=present,
        missing_or_unvalidated_parameters=missing,
        blocking_model_gaps=FOUR_F_BLOCKING_MODEL_GAPS,
        warning_only_components=WARNING_ONLY_CSLM_COMPONENT_IDS,
        order_selection_result=order_selection,
        note=(
            "Current code declares the 4F route but cannot execute a physical +1 "
            "filter without a component-owned lens and Fourier-plane coordinate model."
        ),
    )


def _build_source_field(grid: Mapping[str, Any], config: CSLMRouteConfig) -> np.ndarray:
    radius_m = max(float(config.input_beam_radius_um), EPS) * _UM
    R = np.asarray(grid["R"], dtype=float)
    amp = np.exp(-(R / radius_m) ** 2)
    return amp.astype(complex)


def _slm1_phase(grid: Mapping[str, Any], config: CSLMRouteConfig) -> np.ndarray:
    mode = str(config.slm1_phase_mode).lower()
    X = np.asarray(grid["X"], dtype=float)
    Phi = np.asarray(grid["PHI"], dtype=float)
    if mode in {"flat", "zero"}:
        phase = np.zeros_like(X, dtype=float)
    elif mode == "vortex":
        phase = spp_phase_rad(Phi, int(config.slm1_topological_charge))
    elif mode == "linear_ramp":
        phase = blazed_carrier_phase_rad(X, float(config.slm1_linear_ramp_cpm))
    else:
        phase = np.zeros_like(X, dtype=float)
    return _quantize_wrapped_phase(phase, config.slm_phase_quantisation_levels)


def _slm2_phase_terms(
    grid: Mapping[str, Any],
    config: CSLMRouteConfig,
    *,
    include_carrier: bool = True,
) -> dict[str, np.ndarray]:
    X = np.asarray(grid["X"], dtype=float)
    terms = {
        "carrier_phase_rad": (
            blazed_carrier_phase_rad(X, float(config.slm2_carrier_frequency_cpm))
            if include_carrier
            else np.zeros_like(X, dtype=float)
        ),
        "correction_phase_rad": float(config.slm2_correction_phase_rad) * np.ones_like(X),
    }
    return terms


def _compose_slm2_phase(
    grid: Mapping[str, Any],
    config: CSLMRouteConfig,
    *,
    include_carrier: bool = True,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    terms = _slm2_phase_terms(
        grid,
        config,
        include_carrier=include_carrier,
    )
    continuous = np.zeros_like(next(iter(terms.values())), dtype=float)
    for term in terms.values():
        continuous = continuous + np.asarray(term, dtype=float)
    wrapped = wrap_phase_rad(continuous)
    quantized = _quantize_wrapped_phase(wrapped, config.slm_phase_quantisation_levels)
    return terms, continuous, wrapped, quantized


def _quantize_wrapped_phase(phase_rad: np.ndarray, levels: int) -> np.ndarray:
    levels_i = int(levels)
    wrapped = wrap_phase_rad(phase_rad)
    if levels_i <= 1:
        return wrapped
    bits_float = np.log2(float(levels_i))
    bits_int = int(round(bits_float))
    if (1 << bits_int) == levels_i:
        return quantize_phase_rad(wrapped, bits_int)
    idx = np.floor(wrapped / TWOPI * levels_i + 0.5).astype(np.int64) % levels_i
    return idx.astype(float) * TWOPI / float(levels_i)


def _make_state(
    name: str,
    field_yx: np.ndarray,
    x_um: np.ndarray,
    energy_uJ: float,
    *,
    energy_after_uJ: float | None = None,
    applied_components: tuple[str, ...],
    warnings: tuple[str, ...] = (),
    metadata: Mapping[str, Any] | None = None,
) -> ComponentPlaneState:
    after = float(energy_uJ if energy_after_uJ is None else energy_after_uJ)
    return ComponentPlaneState(
        plane_name=name,
        field=field_yx,
        x_um=x_um,
        y_um=x_um,
        dx_um=float(np.mean(np.abs(np.diff(x_um)))),
        dy_um=float(np.mean(np.abs(np.diff(x_um)))),
        pulse_energy_before_uJ=float(energy_uJ),
        pulse_energy_after_uJ=after,
        transmitted_fraction=after / max(float(energy_uJ), EPS),
        applied_components=applied_components,
        warnings=warnings,
        representation="complex_field",
        metadata=dict(metadata or {}),
    )


def make_order_selection_handoff(mode: OrderHandoffMode) -> OrderSelectionHandoff:
    """Return the claim/provenance record for an order handoff mode."""

    if mode == "none":
        return OrderSelectionHandoff(
            mode="none",
            input_component_id="SLM2_phase_plane",
            output_component_id="post_SLM2_pre_4F_diagnostic_plane",
            physical_filter_modelled=False,
            energy_selection_modelled=False,
            order_efficiency_modelled=False,
            zero_order_rejection_modelled=False,
            field_provenance="no order-selection handoff; active branch remains post-SLM2 unfiltered diagnostic field",
            claim_boundary="active unfiltered CSLM diagnostic only; not final lab output",
            warnings=("4F filtering is inactive; no selected-order field is produced.",),
        )
    if mode == "ideal_selected_order_surrogate":
        return OrderSelectionHandoff(
            mode="ideal_selected_order_surrogate",
            input_component_id="SLM2_phase_plane",
            output_component_id="ideal_selected_order_handoff_plane",
            physical_filter_modelled=False,
            energy_selection_modelled=False,
            order_efficiency_modelled=False,
            zero_order_rejection_modelled=False,
            field_provenance=(
                "field arriving at SLM2 after SLM1 vortex propagation, multiplied by "
                "SLM2 correction phase only; carrier removed as an ideal desired-order surrogate"
            ),
            claim_boundary="carrier-free desired-order benchmark; not a physical 4F prediction",
            warnings=(
                "Surrogate excludes carrier by construction and is not a Fourier-plane crop.",
                "No order efficiency, zero-order rejection, or filter loss is modelled.",
            ),
        )
    if mode == "physical_4f_filter":
        return OrderSelectionHandoff(
            mode="physical_4f_filter",
            input_component_id="SLM2_phase_plane",
            output_component_id="4F_output_plane",
            physical_filter_modelled=False,
            energy_selection_modelled=False,
            order_efficiency_modelled=False,
            zero_order_rejection_modelled=False,
            field_provenance="requested physical 4F filter, but no component-owned Fourier-plane model exists",
            claim_boundary="warning-only; physical 4F filtering unavailable in Stage 8C.3R.5.1",
            warnings=("physical_4f_filter is not implemented; no filtered field is returned.",),
        )
    raise ValueError(f"Unsupported order_handoff_mode: {mode!r}")


def build_ideal_selected_order_surrogate(
    slm2_input_state: ComponentPlaneState,
    grid: Mapping[str, Any],
    config: CSLMRouteConfig,
) -> tuple[ComponentPlaneState, np.ndarray]:
    """Build the explicitly non-physical desired-order surrogate.

    The field starts from the field arriving at SLM2, so it retains SLM1 vortex
    conditioning and SLM1-to-SLM2 propagation.  It applies SLM2 correction phase
    only and removes the carrier as a benchmark surrogate.  It is not a physical
    Fourier filter or +1 selected-order field.
    """

    if slm2_input_state.field is None:
        raise ValueError("slm2_input_state must contain a complex field.")
    _, _, _, correction_phase = _compose_slm2_phase(grid, config, include_carrier=False)
    field = slm2_input_state.field * np.exp(1j * correction_phase)
    state = _make_state(
        "ideal_selected_order_handoff_plane",
        field,
        slm2_input_state.x_um,
        slm2_input_state.pulse_energy_after_uJ,
        applied_components=("ideal_selected_order_surrogate_no_physical_filter",),
        warnings=(
            "carrier-free desired-order benchmark only",
            "physical_filter_modelled=False",
            "zero_order_rejection_modelled=False",
            "order_efficiency_modelled=False",
        ),
        metadata={
            "handoff_mode": "ideal_selected_order_surrogate",
            "physical_filter_modelled": False,
            "zero_order_rejection_modelled": False,
            "order_efficiency_modelled": False,
            "energy_selection_modelled": False,
            "selected_order_energy_uJ": None,
            "claim_boundary": "carrier-free desired-order benchmark; not a physical 4F prediction",
            "field_provenance": (
                "SLM2 input field after SLM1 vortex and propagation; SLM2 correction phase applied; "
                "carrier omitted by non-physical surrogate"
            ),
            "final_export_allowed": False,
        },
    )
    return state, correction_phase


def _reference_field(
    field_yx: np.ndarray,
    grid: Mapping[str, Any],
    config: CSLMRouteConfig,
    distance_um: float | None = None,
) -> np.ndarray:
    prop = make_bl_asm_propagator(
        np.asarray(field_yx, dtype=complex),
        dict(grid),
        config.wavelength_m,
        n_medium=float(config.n_medium),
        bandlimit=bool(config.bandlimit),
    )
    default_distance_um = config.slm2_to_pre_4f_diagnostic_distance_um
    z_m = (default_distance_um if distance_um is None else float(distance_um)) * _UM
    return prop(z_m)


def _propagate_field_um(
    field_yx: np.ndarray,
    grid: Mapping[str, Any],
    config: CSLMRouteConfig,
    distance_um: float,
) -> np.ndarray:
    prop = make_bl_asm_propagator(
        np.asarray(field_yx, dtype=complex),
        dict(grid),
        config.wavelength_m,
        n_medium=float(config.n_medium),
        bandlimit=bool(config.bandlimit),
    )
    return prop(float(distance_um) * _UM)


def _propagated_stack(
    field_yx: np.ndarray,
    x_um: np.ndarray,
    grid: Mapping[str, Any],
    config: CSLMRouteConfig,
    *,
    states: tuple[ComponentPlaneState, ...],
    warnings: tuple[str, ...],
) -> PropagatedFieldStack:
    z_um = np.linspace(0.0, float(config.z_max_um), int(config.n_z))
    prop = make_bl_asm_propagator(
        np.asarray(field_yx, dtype=complex),
        dict(grid),
        config.wavelength_m,
        n_medium=float(config.n_medium),
        bandlimit=bool(config.bandlimit),
    )
    intensity = np.empty((z_um.size, x_um.size, x_um.size), dtype=float)
    for iz, z in enumerate(z_um):
        intensity[iz] = np.abs(prop(float(z) * _UM)) ** 2
    return PropagatedFieldStack(
        intensity_zyx=intensity,
        x_um=x_um,
        y_um=x_um,
        z_um=z_um,
        input_pulse_energy_uJ=float(config.input_pulse_energy_uJ),
        sample_pulse_energy_uJ=float(config.input_pulse_energy_uJ),
        transmitted_fraction=1.0,
        plane_states=states,
        warnings=warnings,
        representation="angular_spectrum_propagated_free_space",
        final_export_allowed=FINAL_EXPORT_ALLOWED,
        model_status=MODEL_STATUS,
        metadata={
            "stage": "8C.3R.5",
            "route_mode": config.route_mode,
            "n_medium": config.n_medium,
            "diagnostic_only": True,
            "no_material_model": True,
            "final_export_allowed": False,
        },
    )


def _propagated_stack_to_distance(
    field_yx: np.ndarray,
    x_um: np.ndarray,
    grid: Mapping[str, Any],
    config: CSLMRouteConfig,
    *,
    distance_um: float,
    sample_energy_uJ: float,
    states: tuple[ComponentPlaneState, ...],
    warnings: tuple[str, ...],
    representation: str,
) -> PropagatedFieldStack:
    z_um = np.linspace(0.0, float(distance_um), int(config.n_z))
    prop = make_bl_asm_propagator(
        np.asarray(field_yx, dtype=complex),
        dict(grid),
        config.wavelength_m,
        n_medium=float(config.n_medium),
        bandlimit=bool(config.bandlimit),
    )
    intensity = np.empty((z_um.size, x_um.size, x_um.size), dtype=float)
    for iz, z in enumerate(z_um):
        intensity[iz] = np.abs(prop(float(z) * _UM)) ** 2
    return PropagatedFieldStack(
        intensity_zyx=intensity,
        x_um=x_um,
        y_um=x_um,
        z_um=z_um,
        input_pulse_energy_uJ=float(config.input_pulse_energy_uJ),
        sample_pulse_energy_uJ=float(sample_energy_uJ),
        transmitted_fraction=float(sample_energy_uJ) / max(float(config.input_pulse_energy_uJ), EPS),
        plane_states=states,
        warnings=warnings,
        representation=representation,
        final_export_allowed=FINAL_EXPORT_ALLOWED,
        model_status=MODEL_STATUS,
        metadata={
            "stage": "8C.3R.5.1",
            "route_mode": config.route_mode,
            "n_medium": config.n_medium,
            "diagnostic_only": True,
            "benchmark_only": representation.startswith("benchmark"),
            "no_material_model": True,
            "final_export_allowed": False,
        },
    )


def _centroid(field_yx: np.ndarray | None, x_um: np.ndarray) -> tuple[float, float]:
    if field_yx is None:
        return (float("nan"), float("nan"))
    I = np.abs(np.asarray(field_yx)) ** 2
    total = float(np.sum(I))
    if total <= EPS:
        return (0.0, 0.0)
    X, Y = np.meshgrid(x_um, x_um, indexing="xy")
    return (float(np.sum(I * X) / total), float(np.sum(I * Y) / total))


def _angle_mrad(field_yx: np.ndarray | None, config: CSLMRouteConfig) -> tuple[float, float]:
    if field_yx is None:
        return (float("nan"), float("nan"))
    arr = np.asarray(field_yx, dtype=complex)
    spectrum = np.abs(fft2c(arr)) ** 2
    total = float(np.sum(spectrum))
    if total <= EPS:
        return (0.0, 0.0)
    freq = np.fft.fftshift(np.fft.fftfreq(arr.shape[1], d=config.dx_m))
    FX, FY = np.meshgrid(freq, freq, indexing="xy")
    fx_cpm = float(np.sum(spectrum * FX) / total)
    fy_cpm = float(np.sum(spectrum * FY) / total)
    sx = np.clip(config.wavelength_m * fx_cpm / max(float(config.n_medium), EPS), -1.0, 1.0)
    sy = np.clip(config.wavelength_m * fy_cpm / max(float(config.n_medium), EPS), -1.0, 1.0)
    return (float(np.arcsin(sx) * 1000.0), float(np.arcsin(sy) * 1000.0))


def _field_metrics(field_yx: np.ndarray | None, x_um: np.ndarray, config: CSLMRouteConfig) -> dict[str, float]:
    if field_yx is None:
        return {
            "raw_power_arb_um2": float("nan"),
            "peak_intensity_arb": float("nan"),
            "centroid_x_um": float("nan"),
            "centroid_y_um": float("nan"),
            "angle_x_mrad": float("nan"),
            "angle_y_mrad": float("nan"),
            "rms_radius_um": float("nan"),
        }
    I = np.abs(np.asarray(field_yx)) ** 2
    cx, cy = _centroid(field_yx, x_um)
    X, Y = np.meshgrid(x_um, x_um, indexing="xy")
    total = float(np.sum(I))
    rms = 0.0 if total <= EPS else float(np.sqrt(np.sum(I * ((X - cx) ** 2 + (Y - cy) ** 2)) / total))
    ax, ay = _angle_mrad(field_yx, config)
    return {
        "raw_power_arb_um2": field_power(field_yx, config.dx_um, config.dx_um),
        "peak_intensity_arb": float(np.max(I)),
        "centroid_x_um": cx,
        "centroid_y_um": cy,
        "angle_x_mrad": ax,
        "angle_y_mrad": ay,
        "rms_radius_um": rms,
    }


def _record(
    component: BeamlineComponent,
    incoming: ComponentPlaneState,
    outgoing: ComponentPlaneState,
    config: CSLMRouteConfig,
    *,
    transform_applied: bool,
    model_status: str,
    warnings: tuple[str, ...] = (),
    aperture_overlap: float | None = None,
) -> RouteInspectionRecord:
    x_um = incoming.x_um
    before_c = _centroid(incoming.field, x_um)
    after_c = _centroid(outgoing.field, x_um)
    before_a = _angle_mrad(incoming.field, config)
    after_a = _angle_mrad(outgoing.field, config)
    return RouteInspectionRecord(
        component_id=component.component_id,
        component_name=component.component_id,
        component_type=component.component_type,
        nominal_location_um=float(component.nominal_z_position_um),
        distance_from_previous_component_mm=float(component.distance_from_previous_component_mm),
        distance_to_next_element_mm=float(component.distance_to_next_element_mm),
        actual_pose_error=component.physical_pose.as_dict(),
        incoming_field_metrics=_field_metrics(incoming.field, x_um, config),
        outgoing_field_metrics=_field_metrics(outgoing.field, x_um, config),
        energy_before_uJ=float(incoming.pulse_energy_after_uJ),
        energy_after_uJ=float(outgoing.pulse_energy_after_uJ),
        centroid_before_um=before_c,
        centroid_after_um=after_c,
        angle_before_mrad=before_a,
        angle_after_mrad=after_a,
        aperture_overlap=aperture_overlap,
        downstream_consequences=component.downstream_elements_affected,
        model_status=model_status,
        transform_applied=bool(transform_applied),
        warnings=warnings,
        represented_by_current_engine=component.represented_by_current_engine,
        physical_model_available=component.physical_model_available,
        misalignment_modes_currently_supported=component.misalignment_modes_currently_supported,
    )


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=float).ravel()
    bv = np.asarray(b, dtype=float).ravel()
    na = float(np.sqrt(np.sum(av * av)))
    nb = float(np.sqrt(np.sum(bv * bv)))
    if na <= EPS or nb <= EPS:
        return 0.0
    return float(np.clip(np.sum(av * bv) / (na * nb), 0.0, 1.0))


def _core_fraction(field_yx: np.ndarray, x_um: np.ndarray, radius_um: float = 4.0) -> float:
    I = np.abs(np.asarray(field_yx)) ** 2
    pk = float(np.max(I))
    if pk <= EPS:
        return 0.0
    X, Y = np.meshgrid(x_um, x_um, indexing="xy")
    mask = np.hypot(X, Y) <= float(radius_um)
    if not np.any(mask):
        return 0.0
    return float(np.mean(I[mask]) / pk)


def _peak_radius_um(field_yx: np.ndarray, x_um: np.ndarray) -> float:
    I = np.abs(np.asarray(field_yx)) ** 2
    iy, ix = np.unravel_index(int(np.argmax(I)), I.shape)
    return float(np.hypot(x_um[ix], x_um[iy]))


def _route_axicon_config(config: CSLMRouteConfig) -> RouteAwareAxiconConfig:
    return RouteAwareAxiconConfig(
        wavelength_nm=config.wavelength_nm,
        n_medium=config.n_medium,
        grid_N=config.grid_N,
        dx_um=config.dx_um,
        n_z=config.n_z,
        input_pulse_energy_uJ=config.input_pulse_energy_uJ,
        physical_axicon_centre_x_um=config.physical_axicon_centre_x_um,
        physical_axicon_centre_y_um=config.physical_axicon_centre_y_um,
        physical_axicon_clear_aperture_radius_um=config.physical_axicon_clear_aperture_radius_um,
        axicon_cone_parameter=config.physical_axicon_cone_parameter_rad_per_um,
    )


def _axicon_aperture_overlap(field_yx: np.ndarray, transmission: np.ndarray, config: CSLMRouteConfig) -> float:
    before = field_power(field_yx, config.dx_um, config.dx_um)
    after = field_power(field_yx * transmission, config.dx_um, config.dx_um)
    return float(after / max(before, EPS))


def _field_momentum_delta_mrad(a: np.ndarray, b: np.ndarray, config: CSLMRouteConfig) -> tuple[float, float]:
    ax, ay = _angle_mrad(a, config)
    bx, by = _angle_mrad(b, config)
    return float(bx - ax), float(by - ay)


def build_physical_axicon_benchmark(
    surrogate_state: ComponentPlaneState,
    grid: Mapping[str, Any],
    config: CSLMRouteConfig,
    handoff: OrderSelectionHandoff,
) -> PhysicalAxiconBenchmarkResult:
    """Apply the represented physical axicon to the ideal surrogate branch."""

    if surrogate_state.field is None:
        raise ValueError("surrogate_state must contain a complex field.")
    x_um = surrogate_state.x_um
    transmission = physical_axicon_transmission(grid, _route_axicon_config(config))
    after_axicon_field = surrogate_state.field * transmission
    overlap = _axicon_aperture_overlap(surrogate_state.field, transmission, config)
    energy_before = float(surrogate_state.pulse_energy_after_uJ)
    energy_after = energy_before * overlap
    sx, sy = _centroid(surrogate_state.field, x_um)
    rel_x = float(sx - config.physical_axicon_centre_x_um)
    rel_y = float(sy - config.physical_axicon_centre_y_um)

    axicon_state = _make_state(
        "physical_axicon_plane",
        after_axicon_field,
        x_um,
        energy_before,
        energy_after_uJ=energy_after,
        applied_components=("physical_axicon_benchmark_transmission",),
        metadata={
            "benchmark_only": True,
            "not_physically_4F_filtered": True,
            "not_experimental_prediction": True,
            "physical_axicon_centre_x_um": config.physical_axicon_centre_x_um,
            "physical_axicon_centre_y_um": config.physical_axicon_centre_y_um,
            "physical_axicon_clear_aperture_radius_um": config.physical_axicon_clear_aperture_radius_um,
            "physical_axicon_cone_parameter_rad_per_um": config.physical_axicon_cone_parameter_rad_per_um,
            "axicon_aperture_overlap_fraction": overlap,
            "final_export_allowed": False,
        },
    )

    reference_field = _propagate_field_um(
        after_axicon_field,
        grid,
        config,
        config.physical_axicon_to_benchmark_reference_distance_um,
    )
    reference_state = _make_state(
        "ideal_axicon_benchmark_reference_plane",
        reference_field,
        x_um,
        energy_after,
        applied_components=("post_axicon_benchmark_free_space",),
        metadata={
            "benchmark_only": True,
            "not_physically_4F_filtered": True,
            "not_experimental_prediction": True,
            "n_medium": 1.0,
            "diagnostic_only": True,
            "no_material_model": True,
            "final_export_allowed": False,
        },
    )
    stack = _propagated_stack_to_distance(
        after_axicon_field,
        x_um,
        grid,
        config,
        distance_um=config.physical_axicon_to_benchmark_reference_distance_um,
        sample_energy_uJ=energy_after,
        states=(surrogate_state, axicon_state, reference_state),
        warnings=handoff.warnings
        + (
            "physical axicon benchmark only",
            "not physically 4F filtered",
            "not experimental prediction",
        ),
        representation="benchmark_physical_axicon_free_space",
    )
    I_ref = np.abs(reference_field) ** 2
    annular = estimate_annular_axis(I_ref, x_um, x_um)
    metrics = {
        "handoff_mode": handoff.mode,
        "energy_entering_axicon_uJ": energy_before,
        "energy_after_axicon_uJ": energy_after,
        "axicon_aperture_overlap_fraction": overlap,
        "relative_beam_to_axicon_offset_x_um": rel_x,
        "relative_beam_to_axicon_offset_y_um": rel_y,
        "relative_beam_to_axicon_offset_um": float(np.hypot(rel_x, rel_y)),
        "physical_axicon_centre_x_um": float(config.physical_axicon_centre_x_um),
        "physical_axicon_centre_y_um": float(config.physical_axicon_centre_y_um),
        "physical_axicon_clear_aperture_radius_um": float(config.physical_axicon_clear_aperture_radius_um),
        "ring_radius_um": float(annular["ring_radius_um"]),
        "dark_core_fraction": float(np.clip(1.0 - _core_fraction(reference_field, x_um, 4.0), 0.0, 1.0)),
        "azimuthal_uniformity": float(annular["azimuthal_uniformity"]),
        "selected_order_energy_uJ": None,
        "order_efficiency_modelled": False,
        "zero_order_rejection_modelled": False,
        "physical_filter_modelled": False,
        "final_export_allowed": False,
    }
    return PhysicalAxiconBenchmarkResult(
        handoff=handoff,
        ideal_selected_order_state=surrogate_state,
        physical_axicon_state=axicon_state,
        benchmark_reference_state=reference_state,
        benchmark_stack=stack,
        benchmark_route_chain=IDEAL_AXICON_BENCHMARK_COMPONENT_IDS,
        transmission=transmission,
        metrics=metrics,
    )


def _baseline_fields_and_metrics(
    grid: Mapping[str, Any],
    x_um: np.ndarray,
    source_field: np.ndarray,
    slm2_input_field: np.ndarray,
    config: CSLMRouteConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    slm1_to_slm2_prop = make_bl_asm_propagator(
        np.asarray(source_field, dtype=complex),
        dict(grid),
        config.wavelength_m,
        n_medium=float(config.n_medium),
        bandlimit=bool(config.bandlimit),
    )
    zero_slm2_input = slm1_to_slm2_prop(float(config.slm1_to_slm2_distance_mm) * _MM)
    zero_ref = _reference_field(zero_slm2_input, grid, config)

    vortex_ref = _reference_field(slm2_input_field, grid, config)

    next_charge = int(config.slm1_topological_charge) + 1
    changed_slm1_phase = _quantize_wrapped_phase(
        spp_phase_rad(np.asarray(grid["PHI"], dtype=float), next_charge),
        config.slm_phase_quantisation_levels,
    )
    changed_slm1_field = source_field * np.exp(1j * changed_slm1_phase)
    changed_prop = make_bl_asm_propagator(
        changed_slm1_field,
        dict(grid),
        config.wavelength_m,
        n_medium=float(config.n_medium),
        bandlimit=bool(config.bandlimit),
    )
    changed_slm2_input = changed_prop(float(config.slm1_to_slm2_distance_mm) * _MM)
    changed_ref = _reference_field(changed_slm2_input, grid, config)

    zero_I = np.abs(zero_ref) ** 2
    vortex_I = np.abs(vortex_ref) ** 2
    changed_I = np.abs(changed_ref) ** 2
    metrics = {
        "zero_phase_power_ratio": field_power(zero_ref, config.dx_um, config.dx_um)
        / max(field_power(zero_slm2_input, config.dx_um, config.dx_um), EPS),
        "slm1_vortex_peak_radius_um": _peak_radius_um(vortex_ref, x_um),
        "slm1_vortex_core_fraction_r4um": _core_fraction(vortex_ref, x_um, 4.0),
        "slm1_vortex_peak_intensity_arb": float(np.max(vortex_I)),
        "topological_charge_test_from": int(config.slm1_topological_charge),
        "topological_charge_test_to": next_charge,
        "topological_charge_owner": "SLM1_phase_plane",
        "topological_charge_intensity_similarity": _cosine_similarity(vortex_I, changed_I),
        "topological_charge_measurable_change": float(1.0 - _cosine_similarity(vortex_I, changed_I)),
        "slm2_axicon_phase_present": False,
        "active_route_contains_physical_axicon": False,
        "phase_quantisation_before_propagation": True,
        "energy_conserved_across_phase_only_elements": True,
        "normalisation_policy": "no hidden renormalisation after passive losses",
    }
    fields = {
        "zero_reference_field": zero_ref,
        "slm1_vortex_reference_field": vortex_ref,
        "changed_charge_reference_field": changed_ref,
        "zero_reference_intensity": zero_I,
        "slm1_vortex_reference_intensity": vortex_I,
        "changed_charge_reference_intensity": changed_I,
    }
    return fields, metrics


def run_cslm_baseline_route(
    config: CSLMRouteConfig | None = None,
) -> CSLMRouteRun:
    """Execute the represented subset of the CSLM baseline route."""

    cfg = config or CSLMRouteConfig()
    if cfg.route_mode != "holographic_cslm":
        raise ValueError("CSLMRouteConfig.route_mode must be 'holographic_cslm'.")
    grid = make_xy_grid(cfg.grid_N, cfg.dx_m)
    x_um = np.asarray(grid["x"], dtype=float) / _UM
    source = _build_source_field(grid, cfg)

    energy = float(cfg.input_pulse_energy_uJ)
    source_state = _make_state(
        "source_field",
        source,
        x_um,
        energy,
        applied_components=("source_complex_field",),
        metadata={
            "physical_location": "conditioned_input_field",
            "n_medium": cfg.n_medium,
            "diagnostic_only": True,
        },
    )
    boundary_state = _make_state(
        "input_conditioning_boundary",
        source,
        x_um,
        energy,
        applied_components=("identity_boundary",),
        metadata={"physical_location": "before_SLM1", "transform_applied": False},
    )

    slm1_phase = _slm1_phase(grid, cfg)
    slm1_field = boundary_state.field * np.exp(1j * slm1_phase)
    slm1_state = _make_state(
        "SLM1_phase_plane",
        slm1_field,
        x_um,
        energy,
        applied_components=("SLM1_phase_only_conditioning",),
        metadata={
            "slm1_role": slm1_role_for_config(cfg),
            "phase_mode": cfg.slm1_phase_mode,
            "topological_charge": cfg.slm1_topological_charge,
            "phase_quantisation_levels": cfg.slm_phase_quantisation_levels,
            "phase_quantisation_before_propagation": True,
        },
    )

    slm1_prop = make_bl_asm_propagator(
        slm1_state.field,
        dict(grid),
        cfg.wavelength_m,
        n_medium=float(cfg.n_medium),
        bandlimit=bool(cfg.bandlimit),
    )
    slm2_input_field = slm1_prop(float(cfg.slm1_to_slm2_distance_mm) * _MM)
    slm2_input_state = _make_state(
        "SLM2_input_plane",
        slm2_input_field,
        x_um,
        energy,
        applied_components=("SLM1_to_SLM2_free_space",),
        metadata={
            "distance_mm": cfg.slm1_to_slm2_distance_mm,
            "physical_location": "before_SLM2",
        },
    )

    slm2_terms, slm2_continuous, slm2_wrapped, slm2_quantized = _compose_slm2_phase(grid, cfg)
    slm2_field = slm2_input_state.field * np.exp(1j * slm2_quantized)
    slm2_state = _make_state(
        "SLM2_phase_plane",
        slm2_field,
        x_um,
        energy,
        applied_components=("SLM2_phase_correction_and_carrier",),
        metadata={
            "slm2_role": "phase_correction_and_carrier_preserve_vortex",
            "conjugate_mode": cfg.slm2_conjugate_mode,
            "carrier_frequency_cpm": cfg.slm2_carrier_frequency_cpm,
            "axicon_phase_produced_here": False,
            "slm2_correction_kind": "uniform_piston_placeholder",
            "slm2_correction_future_requirement": (
                "spatial phase map (Zernike coefficient map / imported phase array / "
                "calibrated correction mask); not implemented in this stage"
            ),
            "phase_quantisation_levels": cfg.slm_phase_quantisation_levels,
            "phase_quantisation_before_propagation": True,
        },
    )

    post_slm2_field = _propagate_field_um(
        slm2_state.field,
        grid,
        cfg,
        cfg.slm2_to_pre_4f_diagnostic_distance_um,
    )
    post_slm2_state = _make_state(
        "post_SLM2_pre_4F_diagnostic_plane",
        post_slm2_field,
        x_um,
        energy,
        applied_components=("SLM2_to_pre_4F_diagnostic_free_space",),
        metadata={
            "n_medium": 1.0,
            "diagnostic_only": True,
            "post_slm2_unfiltered_diagnostic_field": True,
            "contains_carrier": abs(float(cfg.slm2_carrier_frequency_cpm)) > 0,
            "not_final_lab_output": True,
            "no_material_model": True,
            "final_export_allowed": False,
            "distance_from_SLM2_mm": cfg.slm2_to_pre_4f_diagnostic_distance_mm,
        },
    )

    handoff = make_order_selection_handoff(cfg.order_handoff_mode)
    surrogate_state: ComponentPlaneState | None = None
    axicon_benchmark: PhysicalAxiconBenchmarkResult | None = None
    if cfg.order_handoff_mode == "ideal_selected_order_surrogate":
        surrogate_state, _surrogate_phase = build_ideal_selected_order_surrogate(
            slm2_input_state,
            grid,
            cfg,
        )
        if cfg.physical_axicon_enabled_for_benchmark:
            axicon_benchmark = build_physical_axicon_benchmark(
                surrogate_state,
                grid,
                cfg,
                handoff,
            )
    elif cfg.order_handoff_mode == "physical_4f_filter":
        surrogate_state = None
        axicon_benchmark = None

    warnings = (
        "4F lenses, Fourier plane, and +1 order filter are declared warning-only; no fake filtered field is generated.",
        "SLM2 does not produce an axicon phase in this route; the physical axicon is only used in the explicit R5.1 benchmark branch.",
        "All geometry is diagnostic demo geometry unless measured hardware values are supplied in a future stage.",
        "No material/interface/dose/nonlinear/thermal model is active.",
    )
    stack = _propagated_stack_to_distance(
        slm2_state.field,
        x_um,
        grid,
        cfg,
        distance_um=cfg.slm2_to_pre_4f_diagnostic_distance_um,
        sample_energy_uJ=energy,
        states=(source_state, slm1_state, slm2_input_state, slm2_state, post_slm2_state),
        warnings=warnings,
        representation="angular_spectrum_post_slm2_pre_4f_diagnostic",
    )

    fields, metrics = _baseline_fields_and_metrics(grid, x_um, source, slm2_input_state.field, cfg)
    route_declaration = build_cslm_route_declaration(cfg)
    executed = build_executed_cslm_component_chain(cfg)
    by_id = {c.component_id: c for c in executed}
    records = (
        _record(
            by_id["source_field"],
            source_state,
            source_state,
            cfg,
            transform_applied=False,
            model_status="complex_source_field",
        ),
        _record(
            by_id["input_conditioning_boundary"],
            source_state,
            boundary_state,
            cfg,
            transform_applied=False,
            model_status="diagnostic_boundary",
        ),
        _record(
            by_id["SLM1_phase_plane"],
            boundary_state,
            slm1_state,
            cfg,
            transform_applied=True,
            model_status="phase_only_transform",
        ),
        _record(
            by_id["SLM1_to_SLM2_segment"],
            slm1_state,
            slm2_input_state,
            cfg,
            transform_applied=True,
            model_status="free_space_propagation",
        ),
        _record(
            by_id["SLM2_phase_plane"],
            slm2_input_state,
            slm2_state,
            cfg,
            transform_applied=True,
            model_status="phase_only_correction_and_carrier_transform_no_axicon",
        ),
        _record(
            by_id["SLM2_to_pre_4F_diagnostic_segment"],
            slm2_state,
            post_slm2_state,
            cfg,
            transform_applied=True,
            model_status="free_space_propagation",
        ),
        _record(
            by_id["post_SLM2_pre_4F_diagnostic_plane"],
            post_slm2_state,
            post_slm2_state,
            cfg,
            transform_applied=False,
            model_status="post_slm2_unfiltered_diagnostic_only_not_final_lab_output",
        ),
    )
    feasibility = evaluate_fourier_filter_feasibility(cfg)
    return CSLMRouteRun(
        config=cfg,
        route_declaration=route_declaration,
        executed_components=executed,
        inspection_records=records,
        source_state=source_state,
        slm1_state=slm1_state,
        slm2_input_state=slm2_input_state,
        slm2_state=slm2_state,
        reference_plane_state=post_slm2_state,
        propagated_stack=stack,
        slm1_phase_rad=slm1_phase,
        slm2_phase_terms_rad=slm2_terms,
        slm2_composite_phase_rad=slm2_continuous,
        slm2_wrapped_phase_rad=slm2_wrapped,
        slm2_quantized_phase_rad=slm2_quantized,
        post_slm2_unfiltered_state=post_slm2_state,
        order_handoff=handoff,
        ideal_selected_order_surrogate_state=surrogate_state,
        axicon_benchmark=axicon_benchmark,
        baseline_fields=fields,
        baseline_metrics=metrics,
        fourier_feasibility=feasibility,
        warnings=warnings,
    )


def route_inspection_rows(run: CSLMRouteRun) -> list[dict[str, Any]]:
    """Return route-inspection rows as dictionaries."""

    return [record.as_dict() for record in run.inspection_records]


def save_cslm_phase_masks(
    run: CSLMRouteRun,
    output_path: str | Path = "outputs/figures/digital_twin/stage8c3r5_phase_masks.npz",
) -> Path:
    """Save inspectable SLM phase masks used by the run."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        slm1_phase_rad=run.slm1_phase_rad,
        slm2_composite_phase_rad=run.slm2_composite_phase_rad,
        slm2_wrapped_phase_rad=run.slm2_wrapped_phase_rad,
        slm2_quantized_phase_rad=run.slm2_quantized_phase_rad,
        slm1_topological_charge=np.array(run.config.slm1_topological_charge),
        carrier_frequency_cpm=np.array(run.config.slm2_carrier_frequency_cpm),
        slm2_axicon_phase_present=np.array(False),
        active_route_contains_physical_axicon=np.array(False),
    )
    return path


def _imshow(
    ax: plt.Axes,
    data: np.ndarray,
    *,
    extent: tuple[float, float, float, float],
    title: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Any:
    im = ax.imshow(
        np.asarray(data, dtype=float),
        extent=extent,
        origin="lower",
        cmap=cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    return im


def plot_cslm_route_inspection(
    run: CSLMRouteRun,
    output_path: str | Path = "outputs/figures/digital_twin/stage8c3r5_cslm_route_inspection.png",
) -> Path:
    """Plot the CSLM route chain and executed-stage bookkeeping."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(
        "Stage 8C.3R.5 component-owned CSLM route inspection\n"
        "free-space optical diagnostics only; no material model; final_export_allowed=False",
        fontsize=14,
    )

    ax = axes[0, 0]
    ax.axis("off")
    lines = ["Conceptual CSLM route declaration:"]
    for comp in run.route_declaration:
        tag = "ACTIVE" if comp.component_id in run.executed_route_chain else comp.status.upper()
        lines.append(f"{tag:24s} {comp.component_id}")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)

    ax = axes[0, 1]
    rows = route_inspection_rows(run)
    xs = np.arange(len(rows))
    energies = [r["energy after (uJ)"] for r in rows]
    ax.plot(xs, energies, marker="o")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["component_id"] for r in rows], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("energy after stage (uJ)")
    ax.set_title("Energy through executed stages")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    cx = [r["centroid after (um)"][0] for r in rows]
    cy = [r["centroid after (um)"][1] for r in rows]
    ax.plot(xs, cx, marker="o", label="centroid x")
    ax.plot(xs, cy, marker="s", label="centroid y")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["component_id"] for r in rows], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("centroid (um)")
    ax.set_title("Centroid through executed stages")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    ax.axis("off")
    status_lines = [
        "Executed stages:",
        "component_id | transform_applied | model_status",
    ]
    for record in run.inspection_records:
        status_lines.append(
            f"{record.component_id} | {record.transform_applied} | {record.model_status}"
        )
    status_lines.extend(
        [
            "",
            "4F decision:",
            f"fourier_filter_physics_available = {run.fourier_filter_physics_available}",
            "warning-only: Fourier_lens_1, Fourier_plane, +1 filter, Fourier_lens_2, 4F_output",
            "",
            "Active output: post-SLM2 pre-4F diagnostic plane",
            "free-space, n=1.0, diagnostic_only, no material model, not final lab output",
        ]
    )
    ax.text(0.0, 1.0, "\n".join(status_lines), va="top", family="monospace", fontsize=8.5)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.stage8c3r5_metadata = {
        "diagnostic_only": True,
        "final_export_allowed": False,
        "model_status": FIGURE_STATUS,
    }
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_cslm_phase_and_field_baselines(
    run: CSLMRouteRun,
    output_path: str | Path = "outputs/figures/digital_twin/stage8c3r5_slm_phase_and_field_baselines.png",
) -> Path:
    """Plot SLM phase masks and baseline reference-plane fields."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(run.reference_plane_state.x_um, dtype=float)
    z = np.asarray(run.propagated_stack.z_um, dtype=float)
    extent_xy = (float(x[0]), float(x[-1]), float(x[0]), float(x[-1]))
    y0 = int(np.argmin(np.abs(x)))

    vortex_xz = run.propagated_stack.intensity_zyx[:, y0, :].T

    vmax_xy = max(
        float(np.max(run.baseline_fields["slm1_vortex_reference_intensity"])),
        float(np.max(run.baseline_fields["changed_charge_reference_intensity"])),
        float(np.max(np.abs(run.post_slm2_unfiltered_state.field) ** 2)),
        EPS,
    )
    vmax_xz = max(float(np.max(vortex_xz)), EPS)

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle(
        "Stage 8C.3R.5 SLM phase and field baselines\n"
        "phase wrapping/quantisation before propagation; optical diagnostics only",
        fontsize=14,
    )
    im = _imshow(
        axes[0, 0],
        run.slm1_phase_rad,
        extent=extent_xy,
        title="SLM1 vortex / phase-only conditioning",
        cmap="twilight",
        vmin=0.0,
        vmax=TWOPI,
    )
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046)
    im = _imshow(
        axes[0, 1],
        np.abs(run.slm2_input_state.field) ** 2,
        extent=extent_xy,
        title="Field arriving at SLM2",
        vmax=float(np.max(np.abs(run.slm2_input_state.field) ** 2)),
    )
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046)
    im = _imshow(
        axes[0, 2],
        run.slm2_quantized_phase_rad,
        extent=extent_xy,
        title="SLM2 correction/carrier phase - no axicon",
        cmap="twilight",
        vmin=0.0,
        vmax=TWOPI,
    )
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046)

    im = _imshow(
        axes[1, 0],
        run.baseline_fields["slm1_vortex_reference_intensity"],
        extent=extent_xy,
        title="SLM1 vortex-only reference XY",
        vmax=vmax_xy,
    )
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)
    im = _imshow(
        axes[1, 1],
        np.abs(run.post_slm2_unfiltered_state.field) ** 2,
        extent=extent_xy,
        title="Post-SLM2 unfiltered diagnostic XY",
        vmax=vmax_xy,
    )
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)
    im = _imshow(
        axes[1, 2],
        run.baseline_fields["changed_charge_reference_intensity"],
        extent=extent_xy,
        title=f"SLM1 charge {run.config.slm1_topological_charge + 1} comparison XY",
        vmax=vmax_xy,
    )
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046)

    extent_xz = (float(z[0]), float(z[-1]), float(x[0]), float(x[-1]))
    im = _imshow(
        axes[2, 0],
        vortex_xz,
        extent=extent_xz,
        title="SLM1 vortex-only XZ free-space propagation",
        vmax=vmax_xz,
    )
    axes[2, 0].set_xlabel("z from SLM2 (um)")
    axes[2, 0].set_ylabel("x (um)")
    fig.colorbar(im, ax=axes[2, 0], fraction=0.046)
    axes[2, 1].axis("off")
    axes[2, 1].text(
        0.0,
        1.0,
        "Physical axicon benchmark is disabled in the default R5 route.\n"
        "Use order_handoff_mode='ideal_selected_order_surrogate'\n"
        "for the Stage 8C.3R.5.1 benchmark branch.",
        va="top",
        family="monospace",
        fontsize=9,
    )
    axes[2, 2].axis("off")
    metrics = run.baseline_metrics
    metric_lines = [
        "Baseline validations",
        f"zero-phase power ratio: {metrics['zero_phase_power_ratio']:.6f}",
        f"SLM1 vortex peak radius: {metrics['slm1_vortex_peak_radius_um']:.2f} um",
        f"SLM1 vortex core fraction r<4um: {metrics['slm1_vortex_core_fraction_r4um']:.3f}",
        (
            "charge change similarity "
            f"ell={metrics['topological_charge_test_from']}->{metrics['topological_charge_test_to']}: "
            f"{metrics['topological_charge_intensity_similarity']:.3f}"
        ),
        f"measurable charge delta: {metrics['topological_charge_measurable_change']:.3f}",
        "topological charge owner: SLM1",
        "SLM2 axicon phase present: FALSE",
        "active route contains physical axicon: FALSE",
        f"handoff mode: {run.order_handoff.mode}",
        "phase quantisation before propagation: TRUE",
        "material response: DISABLED",
    ]
    axes[2, 2].text(0.0, 1.0, "\n".join(metric_lines), va="top", family="monospace", fontsize=9)

    for ax in axes.ravel():
        if ax.has_data():
            ax.set_xlabel(ax.get_xlabel() or "x (um)")
            if not ax.get_ylabel():
                ax.set_ylabel("y (um)")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.stage8c3r5_metadata = {
        "diagnostic_only": True,
        "final_export_allowed": False,
        "model_status": FIGURE_STATUS,
    }
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_cslm_fourier_order_selection_audit(
    run: CSLMRouteRun,
    output_path: str | Path = "outputs/figures/digital_twin/stage8c3r5_fourier_order_selection_audit.png",
) -> Path:
    """Plot the 4F/+1 order feasibility result."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audit = run.fourier_feasibility
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(
        "Stage 8C.3R.5 4F first-order feasibility audit\n"
        "declared route only; no physical filtered field generated",
        fontsize=14,
    )
    axes[0].axis("off")
    route_lines = ["Conceptual 4F branch:"]
    for cid in CONCEPTUAL_CSLM_COMPONENT_IDS:
        if cid in EXECUTED_CSLM_COMPONENT_IDS:
            tag = "executed"
        elif cid in WARNING_ONLY_CSLM_COMPONENT_IDS:
            tag = "warning-only"
        else:
            tag = "declared"
        route_lines.append(f"{tag:14s} {cid}")
    route_lines.extend(
        [
            "",
            f"fourier_filter_physics_available = {audit.fourier_filter_physics_available}",
            f"filtered_field returned = {audit.order_selection_result['filtered_field']}",
            "",
            "Carrier/order units:",
            f"carrier = {run.config.slm2_carrier_frequency_cpm:.3g} cycles/m",
            f"carrier = {run.config.slm2_carrier_frequency_cycles_per_mm:.3g} cycles/mm",
            f"period = {run.config.carrier_spatial_period_um:.2f} um",
        ]
    )
    axes[0].text(0.0, 1.0, "\n".join(route_lines), va="top", family="monospace", fontsize=9)

    axes[1].axis("off")
    missing_lines = ["Missing/unvalidated before 4F can become active:"]
    missing_lines.extend(f"- {item}" for item in audit.missing_or_unvalidated_parameters)
    missing_lines.extend(["", "Blocking model gaps:"])
    missing_lines.extend(f"- {item}" for item in audit.blocking_model_gaps)
    axes[1].text(0.0, 1.0, "\n".join(missing_lines), va="top", ha="left", fontsize=9)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    fig.stage8c3r5_metadata = {
        "diagnostic_only": True,
        "final_export_allowed": False,
        "model_status": FIGURE_STATUS,
    }
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _require_axicon_benchmark(run: CSLMRouteRun) -> PhysicalAxiconBenchmarkResult:
    if run.order_handoff.mode != "ideal_selected_order_surrogate" or run.axicon_benchmark is None:
        raise ValueError(
            "Stage 8C.3R.5.1 benchmark figures require "
            "order_handoff_mode='ideal_selected_order_surrogate' and "
            "physical_axicon_enabled_for_benchmark=True."
        )
    return run.axicon_benchmark


def _phase_image(field_yx: np.ndarray) -> np.ndarray:
    return np.angle(np.asarray(field_yx, dtype=complex))


def _normalised_intensity(field_yx: np.ndarray) -> np.ndarray:
    intensity = np.abs(np.asarray(field_yx, dtype=complex)) ** 2
    return intensity / max(float(np.max(intensity)), EPS)


def plot_cslm_to_axicon_handoff_audit(
    run: CSLMRouteRun,
    output_path: str | Path = "outputs/figures/digital_twin/stage8c3r5_1_cslm_to_axicon_handoff_audit.png",
) -> Path:
    """Plot the active CSLM branch beside the explicit ideal axicon benchmark."""

    benchmark = _require_axicon_benchmark(run)
    surrogate = benchmark.ideal_selected_order_state
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(run.post_slm2_unfiltered_state.x_um, dtype=float)
    z = np.asarray(benchmark.benchmark_stack.z_um, dtype=float)
    extent_xy = (float(x[0]), float(x[-1]), float(x[0]), float(x[-1]))
    extent_xz = (float(z[0]), float(z[-1]), float(x[0]), float(x[-1]))
    y0 = int(np.argmin(np.abs(x)))
    axicon_phase = np.angle(benchmark.transmission)
    axicon_aperture = np.abs(benchmark.transmission) > 0.0
    benchmark_xz = benchmark.benchmark_stack.intensity_zyx[:, y0, :].T

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    fig.suptitle(
        "Stage 8C.3R.5.1 CSLM-to-physical-axicon handoff audit\n"
        "active post-SLM2 diagnostic branch plus opt-in ideal selected-order benchmark",
        fontsize=14,
    )

    im = _imshow(
        axes[0, 0],
        run.slm1_phase_rad,
        extent=extent_xy,
        title="active executed diagnostic: SLM1 vortex phase",
        cmap="twilight",
        vmin=0.0,
        vmax=TWOPI,
    )
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046)
    im = _imshow(
        axes[0, 1],
        _normalised_intensity(run.slm2_input_state.field),
        extent=extent_xy,
        title="active executed diagnostic: field arriving at SLM2",
    )
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046)
    im = _imshow(
        axes[0, 2],
        run.slm2_quantized_phase_rad,
        extent=extent_xy,
        title="active executed diagnostic: SLM2 correction + carrier, no axicon",
        cmap="twilight",
        vmin=0.0,
        vmax=TWOPI,
    )
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046)

    im = _imshow(
        axes[1, 0],
        _normalised_intensity(run.post_slm2_unfiltered_state.field),
        extent=extent_xy,
        title="active executed diagnostic: post-SLM2 pre-4F field",
    )
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)
    im = _imshow(
        axes[1, 1],
        _normalised_intensity(surrogate.field),
        extent=extent_xy,
        title="ideal selected-order surrogate: carrier removed non-physically",
    )
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)
    im = _imshow(
        axes[1, 2],
        axicon_phase,
        extent=extent_xy,
        title="physical axicon benchmark: scalar phase",
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046)
    X_um, Y_um = np.meshgrid(x, x, indexing="xy")
    axes[1, 2].contour(
        X_um,
        Y_um,
        axicon_aperture.astype(float),
        levels=[0.5],
        colors="white",
        linewidths=0.8,
    )

    im = _imshow(
        axes[2, 0],
        _normalised_intensity(benchmark.physical_axicon_state.field),
        extent=extent_xy,
        title="physical axicon benchmark: immediately after axicon",
    )
    fig.colorbar(im, ax=axes[2, 0], fraction=0.046)
    im = _imshow(
        axes[2, 1],
        benchmark_xz,
        extent=extent_xz,
        title="physical axicon benchmark: post-axicon XZ free-space",
    )
    axes[2, 1].set_xlabel("z after axicon (um)")
    axes[2, 1].set_ylabel("x (um)")
    fig.colorbar(im, ax=axes[2, 1], fraction=0.046)
    axes[2, 2].axis("off")
    lines = [
        "Branch/claim boundary",
        "A active: post_slm2_unfiltered_diagnostic_field",
        "  no 4F filter; not final lab output",
        "B benchmark: ideal_selected_order_surrogate_field",
        "  carrier-free desired-order surrogate",
        "  not a physical 4F prediction",
        "Axicon: physical scalar phase + clear aperture",
        "Material response: disabled",
        "final_export_allowed: False",
        "",
        f"handoff mode: {run.order_handoff.mode}",
        f"4F physics available: {run.fourier_filter_physics_available}",
    ]
    axes[2, 2].text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=8.5)

    for ax in axes.ravel():
        if ax.has_data():
            ax.set_xlabel(ax.get_xlabel() or "x (um)")
            if not ax.get_ylabel():
                ax.set_ylabel("y (um)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.stage8c3r5_1_metadata = {
        "diagnostic_only": True,
        "benchmark_only": True,
        "physical_filter_modelled": False,
        "final_export_allowed": False,
    }
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_unfiltered_vs_ideal_handoff(
    run: CSLMRouteRun,
    output_path: str | Path = "outputs/figures/digital_twin/stage8c3r5_1_unfiltered_vs_ideal_handoff.png",
) -> Path:
    """Compare actual unfiltered post-SLM2 field with the ideal surrogate."""

    benchmark = _require_axicon_benchmark(run)
    surrogate = benchmark.ideal_selected_order_state
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(run.post_slm2_unfiltered_state.x_um, dtype=float)
    extent_xy = (float(x[0]), float(x[-1]), float(x[0]), float(x[-1]))
    actual = run.post_slm2_unfiltered_state.field
    ideal = surrogate.field
    actual_i = _normalised_intensity(actual)
    ideal_i = _normalised_intensity(ideal)
    actual_metrics = _field_metrics(actual, x, run.config)
    ideal_metrics = _field_metrics(ideal, x, run.config)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        "Stage 8C.3R.5.1 unfiltered CSLM output versus ideal handoff\n"
        "the surrogate is not a physically simulated 4F selected order",
        fontsize=14,
    )
    vmax = max(float(np.max(actual_i)), float(np.max(ideal_i)), EPS)
    im = _imshow(
        axes[0, 0],
        actual_i,
        extent=extent_xy,
        title="active executed diagnostic: post-SLM2 unfiltered intensity",
        vmax=vmax,
    )
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046)
    im = _imshow(
        axes[0, 1],
        ideal_i,
        extent=extent_xy,
        title="ideal selected-order surrogate intensity",
        vmax=vmax,
    )
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046)
    axes[0, 2].axis("off")
    energy_lines = [
        "Energy and order boundary",
        f"post-SLM2 diagnostic energy: {run.post_slm2_unfiltered_state.pulse_energy_after_uJ:.3f} uJ",
        f"surrogate energy: {surrogate.pulse_energy_after_uJ:.3f} uJ",
        "selected_order_energy_uJ: None",
        "order_efficiency_modelled: False",
        "zero_order_rejection_modelled: False",
        "physical_filter_modelled: False",
        f"carrier period: {run.config.carrier_spatial_period_um:.2f} um",
    ]
    axes[0, 2].text(0.0, 1.0, "\n".join(energy_lines), va="top", family="monospace", fontsize=9)

    im = _imshow(
        axes[1, 0],
        _phase_image(actual),
        extent=extent_xy,
        title="active executed diagnostic: carrier-bearing phase",
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)
    im = _imshow(
        axes[1, 1],
        _phase_image(ideal),
        extent=extent_xy,
        title="ideal surrogate phase: correction only after SLM1 propagation",
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)
    axes[1, 2].axis("off")
    metric_lines = [
        "Centroid / momentum",
        (
            "actual centroid: "
            f"({actual_metrics['centroid_x_um']:.3f}, {actual_metrics['centroid_y_um']:.3f}) um"
        ),
        (
            "ideal centroid: "
            f"({ideal_metrics['centroid_x_um']:.3f}, {ideal_metrics['centroid_y_um']:.3f}) um"
        ),
        (
            "actual angle: "
            f"({actual_metrics['angle_x_mrad']:.3f}, {actual_metrics['angle_y_mrad']:.3f}) mrad"
        ),
        (
            "ideal angle: "
            f"({ideal_metrics['angle_x_mrad']:.3f}, {ideal_metrics['angle_y_mrad']:.3f}) mrad"
        ),
        "",
        "Carrier removal here is a labelled non-physical surrogate.",
        "It is not a Fourier-plane crop and not a +1 filtered field.",
    ]
    axes[1, 2].text(0.0, 1.0, "\n".join(metric_lines), va="top", family="monospace", fontsize=9)

    for ax in axes.ravel():
        if ax.has_data():
            ax.set_xlabel(ax.get_xlabel() or "x (um)")
            if not ax.get_ylabel():
                ax.set_ylabel("y (um)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    fig.stage8c3r5_1_metadata = {
        "diagnostic_only": True,
        "physical_filter_modelled": False,
        "final_export_allowed": False,
    }
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_ideal_vortex_bessel_axicon_benchmark(
    run: CSLMRouteRun,
    output_path: str | Path = "outputs/figures/digital_twin/stage8c3r5_1_ideal_vortex_bessel_axicon_benchmark.png",
) -> Path:
    """Plot the ideal selected-order plus physical-axicon benchmark branch."""

    benchmark = _require_axicon_benchmark(run)
    surrogate = benchmark.ideal_selected_order_state
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(surrogate.x_um, dtype=float)
    z = np.asarray(benchmark.benchmark_stack.z_um, dtype=float)
    extent_xy = (float(x[0]), float(x[-1]), float(x[0]), float(x[-1]))
    extent_xz = (float(z[0]), float(z[-1]), float(x[0]), float(x[-1]))
    y0 = int(np.argmin(np.abs(x)))
    axicon_phase = np.angle(benchmark.transmission)
    axicon_aperture = np.abs(benchmark.transmission) > 0.0
    after_axicon_i = _normalised_intensity(benchmark.physical_axicon_state.field)
    reference_i = np.abs(benchmark.benchmark_reference_state.field) ** 2
    reference_i_norm = reference_i / max(float(np.max(reference_i)), EPS)
    xz = benchmark.benchmark_stack.intensity_zyx[:, y0, :].T
    fluence = stack_to_fluence(benchmark.benchmark_stack).fluence_zyx_j_cm2[-1]

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    fig.suptitle(
        "Ideal selected-order plus physical-axicon benchmark\n"
        "Not a physical 4F-filtered CSLM prediction",
        fontsize=15,
    )
    im = _imshow(
        axes[0, 0],
        _normalised_intensity(surrogate.field),
        extent=extent_xy,
        title="ideal selected-order field incident on axicon",
    )
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046)
    im = _imshow(
        axes[0, 1],
        axicon_aperture.astype(float),
        extent=extent_xy,
        title="physical axicon clear aperture",
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046)
    im = _imshow(
        axes[0, 2],
        axicon_phase,
        extent=extent_xy,
        title="physical axicon scalar phase",
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046)

    im = _imshow(
        axes[1, 0],
        after_axicon_i,
        extent=extent_xy,
        title="post-axicon XY intensity",
    )
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)
    im = _imshow(
        axes[1, 1],
        xz,
        extent=extent_xz,
        title="post-axicon XZ propagation",
    )
    axes[1, 1].set_xlabel("z after axicon (um)")
    axes[1, 1].set_ylabel("x (um)")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)
    im = _imshow(
        axes[1, 2],
        reference_i_norm,
        extent=extent_xy,
        title="reference-plane XY intensity",
    )
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046)

    im = _imshow(
        axes[2, 0],
        fluence,
        extent=extent_xy,
        title="reference-plane optical fluence (J/cm^2)",
    )
    fig.colorbar(im, ax=axes[2, 0], fraction=0.046)
    axes[2, 1].axis("off")
    m = benchmark.metrics
    metric_lines = [
        "Benchmark metrics",
        f"ring radius: {m['ring_radius_um']:.3f} um",
        f"dark-core fraction: {m['dark_core_fraction']:.3f}",
        f"azimuthal uniformity: {m['azimuthal_uniformity']:.3f}",
        f"energy entering axicon: {m['energy_entering_axicon_uJ']:.3f} uJ",
        f"energy after axicon: {m['energy_after_axicon_uJ']:.3f} uJ",
        f"aperture overlap: {m['axicon_aperture_overlap_fraction']:.6f}",
        f"relative offset: {m['relative_beam_to_axicon_offset_um']:.3f} um",
    ]
    axes[2, 1].text(0.0, 1.0, "\n".join(metric_lines), va="top", family="monospace", fontsize=9)
    axes[2, 2].axis("off")
    claim_lines = [
        "Boundary",
        "benchmark_only: True",
        "not_physically_4F_filtered: True",
        "not_experimental_prediction: True",
        "order_efficiency_modelled: False",
        "zero_order_rejection_modelled: False",
        "material response: disabled",
        "final_export_allowed: False",
        "",
        "Only represented loss: physical axicon aperture clipping.",
    ]
    axes[2, 2].text(0.0, 1.0, "\n".join(claim_lines), va="top", family="monospace", fontsize=9)

    for ax in axes.ravel():
        if ax.has_data():
            ax.set_xlabel(ax.get_xlabel() or "x (um)")
            if not ax.get_ylabel():
                ax.set_ylabel("y (um)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.stage8c3r5_1_metadata = {
        "diagnostic_only": True,
        "benchmark_only": True,
        "not_physically_4F_filtered": True,
        "final_export_allowed": False,
    }
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def generate_stage8c3r5_1_previews(
    config: CSLMRouteConfig | None = None,
    output_dir: str | Path = "outputs/figures/digital_twin",
) -> dict[str, Path]:
    """Generate explicitly enabled R5.1 ideal selected-order benchmark previews."""

    if config is None:
        raise ValueError(
            "R5.1 previews require an explicit CSLMRouteConfig with "
            "order_handoff_mode='ideal_selected_order_surrogate'."
        )
    cfg = config
    if cfg.order_handoff_mode != "ideal_selected_order_surrogate":
        raise ValueError("R5.1 previews require order_handoff_mode='ideal_selected_order_surrogate'.")
    run = run_cslm_baseline_route(cfg)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "handoff_audit": plot_cslm_to_axicon_handoff_audit(
            run, out / "stage8c3r5_1_cslm_to_axicon_handoff_audit.png"
        ),
        "unfiltered_vs_ideal_handoff": plot_unfiltered_vs_ideal_handoff(
            run, out / "stage8c3r5_1_unfiltered_vs_ideal_handoff.png"
        ),
        "ideal_vortex_bessel_axicon_benchmark": plot_ideal_vortex_bessel_axicon_benchmark(
            run, out / "stage8c3r5_1_ideal_vortex_bessel_axicon_benchmark.png"
        ),
    }


def generate_stage8c3r5_previews(
    config: CSLMRouteConfig | None = None,
    output_dir: str | Path = "outputs/figures/digital_twin",
) -> dict[str, Path]:
    """Run the CSLM baseline and generate all required diagnostic previews."""

    run = run_cslm_baseline_route(config)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "route_inspection": plot_cslm_route_inspection(
            run, out / "stage8c3r5_cslm_route_inspection.png"
        ),
        "phase_and_field_baselines": plot_cslm_phase_and_field_baselines(
            run, out / "stage8c3r5_slm_phase_and_field_baselines.png"
        ),
        "fourier_order_selection_audit": plot_cslm_fourier_order_selection_audit(
            run, out / "stage8c3r5_fourier_order_selection_audit.png"
        ),
        "phase_masks": save_cslm_phase_masks(run, out / "stage8c3r5_phase_masks.npz"),
    }


__all__ = [
    "CONCEPTUAL_CSLM_COMPONENT_IDS",
    "EXECUTED_CSLM_COMPONENT_IDS",
    "WARNING_ONLY_CSLM_COMPONENT_IDS",
    "IDEAL_AXICON_BENCHMARK_COMPONENT_IDS",
    "FOUR_F_REQUIRED_PARAMETERS",
    "OrderHandoffMode",
    "OrderSelectionHandoff",
    "PhysicalAxiconBenchmarkResult",
    "CSLMRouteConfig",
    "CSLMRouteRun",
    "FourFFeasibilityAudit",
    "build_cslm_route_declaration",
    "build_executed_cslm_component_chain",
    "evaluate_fourier_filter_feasibility",
    "run_cslm_baseline_route",
    "route_inspection_rows",
    "save_cslm_phase_masks",
    "plot_cslm_route_inspection",
    "plot_cslm_phase_and_field_baselines",
    "plot_cslm_fourier_order_selection_audit",
    "plot_cslm_to_axicon_handoff_audit",
    "plot_unfiltered_vs_ideal_handoff",
    "plot_ideal_vortex_bessel_axicon_benchmark",
    "generate_stage8c3r5_previews",
    "generate_stage8c3r5_1_previews",
]
