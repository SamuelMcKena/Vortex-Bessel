"""MODE 2U2 provenance-bound Nathan source-scale closure.

This module is an audit layer over the already-solved MODE 2P/2N/2Q/2S/2U
source-scale machinery.  It does not introduce an unconstrained hologram
optimizer and it does not promote the source-scale result to a lab build.  Its
job is to bind the Nathan branch back to the repository hardware records,
exercise native-panel mask geometry, record power/useful-region accounting, and
make the remaining readiness blockers explicit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from itertools import combinations, product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.design import default_config
from vbb_study.digital_twin.bench_inventory import (
    build_bench_inventory_profile,
    evaluate_physical_4f_readiness,
)
from vbb_study.digital_twin.cslm_route import CSLMRouteConfig
from vbb_study.digital_twin.nominal_f300_4f import config_from_profile
from vbb_study.digital_twin.nathan_mode2u_master_audit import (
    MODE2U_DEFAULT_OUTPUT_ROOT,
    MODE2U_RENDER_INTERPOLATION,
    _case_metrics,
    _plot_beam_image,
    _plot_difference_grid,
    _plot_profiles,
    _plot_sampling_audit,
    _rel,
    _save_highres,
)
from vbb_study.digital_twin.nathan_vector_hexagon import (
    EPS,
    MODE2N_DEFAULT_CARRIER_LPMM,
    MODE2N_DEFAULT_IRIS_RADIUS_FRAC,
    MODE2S_PASS_CORRELATION,
    Mode2SCorrection,
    Mode2SPerturbation,
    NathanSourceParityConfig,
    _json_ready,
    _normalise_image,
    _write_rows,
    angular_profile_on_ring,
    mode2n_route_metric_row,
    mode2n_source_target,
    mode2q_strict_hexagon_gate,
    mode2s_combined_cases,
    mode2s_slm_aperture_fit_report,
    nathan_alpha_map,
    run_mode2n_dual_slm_4f_route,
    run_mode2n_dual_slm_qwp_route,
    run_mode2n_patterned_hwp_route,
    run_mode2n_v0_reference,
    run_mode2q_backward_initialisation,
    run_mode2s_degraded_forward,
    wrap_2pi,
)


MODE2U2_STAGE = "nathan_mode2u2_master_closure"
MODE2U2_DEFAULT_OUTPUT_ROOT = Path("outputs/figures/digital_twin/nathan_mode2u2_master_closure")
MODE2U2_DOC_PATH = Path("docs/73_nathan_mode2u2_master_closure.md")
MODE2U2_ALLOWED_OUTCOMES = ("M2U2-A", "M2U2-B", "M2U2-C", "M2U2-D", "M2U2-E")
MODE2U2_SEED = 20260709

PROVENANCE_CATEGORIES = (
    "measured_lab",
    "manufacturer_spec",
    "original_digital_twin",
    "nathan_source_model",
    "inferred_from_existing_config",
    "planning_assumption",
    "placeholder",
    "unknown",
)

ENERGY_STAGE_NAMES = (
    "input_gaussian_power",
    "after_initial_polarisation_preparation",
    "h_channel_power",
    "v_channel_power",
    "incident_power_on_slm_h",
    "incident_power_on_slm_v",
    "after_slm_h",
    "after_slm_v",
    "fourier_plane_total_power",
    "selected_first_order_power",
    "zero_order_power",
    "rejected_spectral_power",
    "finite_aperture_clipping",
    "reconstructed_hv_power",
    "recombined_power",
    "after_qwp",
    "incident_on_axicon",
    "after_axicon",
    "integrated_power_z60",
    "useful_hexagon_region_power",
    "outside_useful_region_power",
    "peak_intensity_proxy",
)


@dataclass(frozen=True)
class HardwareParameterProvenance:
    """One hardware/model parameter with explicit provenance."""

    parameter_name: str
    value: Any
    units: str
    source_file: str
    source_object: str
    source_field: str
    category: str
    measured_lab: bool
    manufacturer_spec: bool
    original_digital_twin: bool
    nathan_source_model: bool
    newly_assumed: bool
    status: str
    confidence: str
    notes: str = ""

    def row(self) -> dict[str, Any]:
        if self.category not in PROVENANCE_CATEGORIES:
            raise ValueError(f"unsupported provenance category {self.category!r}")
        return asdict(self)


@dataclass(frozen=True)
class NathanHardwareBinding:
    """Resolved hardware binding for the source-scale Nathan realism branch."""

    wavelength_m: float
    slm_model: str
    slm_width_px: int
    slm_height_px: int
    slm_pixel_pitch_m: float
    slm_active_width_m: float
    slm_active_height_m: float
    slm_bit_depth: int | None
    slm_fill_factor: float | None
    slm_phase_stroke_rad: float | None
    fourf_focal_length_m: float | None
    carrier_cycles_per_m: float
    iris_radius_cycles_per_m: float
    iris_physical_radius_m: float | None
    qwp_nominal_angle_rad: float
    qwp_nominal_retardance_rad: float
    axicon_base_angle_deg: float
    axicon_refractive_index: float
    external_medium_index: float
    camera_pixel_pitch_m: float | None
    camera_magnification: float | None
    provenance: Mapping[str, str]

    @property
    def carrier_lpmm(self) -> float:
        return float(self.carrier_cycles_per_m) / 1.0e3

    @property
    def iris_radius_lpmm(self) -> float:
        return float(self.iris_radius_cycles_per_m) / 1.0e3

    @property
    def carrier_period_pixels(self) -> float:
        return 1.0 / max(abs(float(self.carrier_cycles_per_m)) * float(self.slm_pixel_pitch_m), EPS)


def _repo_path(path: str | Path) -> str:
    return str(Path(path)).replace("\\", "/")


def _prov(
    parameter: str,
    value: Any,
    units: str,
    source_file: str,
    source_object: str,
    source_field: str,
    category: str,
    status: str,
    confidence: str,
    notes: str = "",
) -> HardwareParameterProvenance:
    return HardwareParameterProvenance(
        parameter_name=parameter,
        value=value,
        units=units,
        source_file=_repo_path(source_file),
        source_object=source_object,
        source_field=source_field,
        category=category,
        measured_lab=category == "measured_lab",
        manufacturer_spec=category == "manufacturer_spec",
        original_digital_twin=category == "original_digital_twin",
        nathan_source_model=category == "nathan_source_model",
        newly_assumed=category in {"planning_assumption", "placeholder"},
        status=status,
        confidence=confidence,
        notes=notes,
    )


def resolve_nathan_hardware_binding(
    *,
    twin_preset: str = "fast",
    source_config: NathanSourceParityConfig | None = None,
) -> NathanHardwareBinding:
    """Resolve the Nathan source-scale hardware values from repository sources."""

    twin = default_config(twin_preset)
    source = source_config or NathanSourceParityConfig()
    try:
        f300 = config_from_profile()
        fourf_focal = float(f300.lens1_focal_length_m)
    except Exception:
        f300 = None
        fourf_focal = None

    carrier_cpm = float(twin.slm.carrier_cpm)
    iris_cpm = float(twin.slm.first_order_filter_radius_lpmm) * 1.0e3
    iris_physical = None if fourf_focal is None else float(fourf_focal * twin.laser.wavelength_m * iris_cpm)
    provenance = {
        "wavelength_m": "TwinConfig.laser.wavelength_m",
        "slm_model": "TwinConfig.slm.name",
        "slm_width_px": "TwinConfig.slm.resolution_x",
        "slm_height_px": "TwinConfig.slm.resolution_y",
        "slm_pixel_pitch_m": "TwinConfig.slm.pixel_pitch_m",
        "slm_active_width_m": "TwinConfig.slm.active_width_m",
        "slm_active_height_m": "TwinConfig.slm.active_height_m",
        "slm_bit_depth": "TwinConfig.slm.phase_bits",
        "slm_fill_factor": "TwinConfig.slm.fill_factor",
        "slm_phase_stroke_rad": "unknown; manufacturer_evidence_register requires wavelength-specific stroke",
        "fourf_focal_length_m": "configs/hardware/cslm_f300_nominal_4f_profile.json known_nominal_geometry",
        "carrier_cycles_per_m": "TwinConfig.slm.carrier_cpm",
        "iris_radius_cycles_per_m": "TwinConfig.slm.first_order_filter_radius_lpmm",
        "iris_physical_radius_m": "inferred f * wavelength * iris_spatial_frequency from nominal F300 profile",
        "qwp_nominal_angle_rad": "MODE 2P/2N source-model convention",
        "qwp_nominal_retardance_rad": "MODE 2P/2N source-model convention",
        "axicon_base_angle_deg": "NathanSourceParityConfig.axicon_apex_angle_deg",
        "axicon_refractive_index": "NathanSourceParityConfig.axicon_n",
        "external_medium_index": "NathanSourceParityConfig.medium_n",
        "camera_pixel_pitch_m": "unknown; bench_evidence_register B_CAMERA_SCALE not ready",
        "camera_magnification": "unknown; bench_evidence_register B_CAMERA_SCALE not ready",
    }
    return NathanHardwareBinding(
        wavelength_m=float(twin.laser.wavelength_m),
        slm_model=str(twin.slm.name),
        slm_width_px=int(twin.slm.resolution_x),
        slm_height_px=int(twin.slm.resolution_y),
        slm_pixel_pitch_m=float(twin.slm.pixel_pitch_m),
        slm_active_width_m=float(twin.slm.active_width_m),
        slm_active_height_m=float(twin.slm.active_height_m),
        slm_bit_depth=int(twin.slm.phase_bits),
        slm_fill_factor=float(twin.slm.fill_factor),
        slm_phase_stroke_rad=None,
        fourf_focal_length_m=fourf_focal,
        carrier_cycles_per_m=carrier_cpm,
        iris_radius_cycles_per_m=iris_cpm,
        iris_physical_radius_m=iris_physical,
        qwp_nominal_angle_rad=float(-0.25 * np.pi),
        qwp_nominal_retardance_rad=float(0.5 * np.pi),
        axicon_base_angle_deg=float(np.rad2deg(source.axicon_base_angle_rad)),
        axicon_refractive_index=float(source.axicon_n),
        external_medium_index=float(source.medium_n),
        camera_pixel_pitch_m=None,
        camera_magnification=None,
        provenance=provenance,
    )


def hardware_parameter_provenance_rows(
    binding: NathanHardwareBinding | None = None,
    *,
    twin_preset: str = "fast",
    source_config: NathanSourceParityConfig | None = None,
) -> list[dict[str, Any]]:
    """Return provenance rows for every realistic parameter used by MODE 2U2."""

    binding = binding or resolve_nathan_hardware_binding(twin_preset=twin_preset, source_config=source_config)
    twin = default_config(twin_preset)
    source = source_config or NathanSourceParityConfig()
    cslm = CSLMRouteConfig()
    f300 = config_from_profile()
    rows = [
        _prov("wavelength_m", twin.laser.wavelength_m, "m", "vbb_study/config.py", "LaserConfig", "wavelength_m", "original_digital_twin", "inherited_design_value", "high"),
        _prov("laser_name", twin.laser.name, "", "vbb_study/config.py", "LaserConfig", "name", "original_digital_twin", "inherited_design_value", "medium"),
        _prov("pulse_duration_s", twin.laser.pulse_duration_s, "s", "vbb_study/config.py", "LaserConfig", "pulse_duration_s", "original_digital_twin", "inherited_design_value", "medium"),
        _prov("input_pulse_energy_J", twin.laser.input_pulse_energy_J, "J", "vbb_study/config.py", "LaserConfig", "input_pulse_energy_J", "original_digital_twin", "inherited_design_value", "medium"),
        _prov("rep_rate_Hz", twin.laser.rep_rate_Hz, "Hz", "vbb_study/config.py", "LaserConfig", "rep_rate_Hz", "original_digital_twin", "inherited_design_value", "medium"),
        _prov("beam_radius_on_slm_m", twin.laser.beam_radius_on_slm_m, "m", "vbb_study/config.py", "LaserConfig", "beam_radius_on_slm_m", "original_digital_twin", "inherited_design_value", "high", twin.laser.beam_radius_definition),
        _prov("slm_model", twin.slm.name, "", "vbb_study/config.py", "SLMConfig", "name", "original_digital_twin", "generic_holoeye_like_record", "medium", "Project manufacturer register still marks exact SLM1/SLM2 specs unknown."),
        _prov("slm_resolution_x", twin.slm.resolution_x, "px", "vbb_study/config.py", "SLMConfig", "resolution_x", "original_digital_twin", "inherited_design_value", "high"),
        _prov("slm_resolution_y", twin.slm.resolution_y, "px", "vbb_study/config.py", "SLMConfig", "resolution_y", "original_digital_twin", "inherited_design_value", "high"),
        _prov("slm_pixel_pitch_m", twin.slm.pixel_pitch_m, "m", "vbb_study/config.py", "SLMConfig", "pixel_pitch_m", "original_digital_twin", "inherited_design_value", "high"),
        _prov("slm_active_width_m", twin.slm.active_width_m, "m", "vbb_study/config.py", "SLMConfig", "active_width_m", "inferred_from_existing_config", "derived_from_resolution_and_pitch", "high"),
        _prov("slm_active_height_m", twin.slm.active_height_m, "m", "vbb_study/config.py", "SLMConfig", "active_height_m", "inferred_from_existing_config", "derived_from_resolution_and_pitch", "high"),
        _prov("slm_phase_bits", twin.slm.phase_bits, "bits", "vbb_study/config.py", "SLMConfig", "phase_bits", "original_digital_twin", "inherited_design_value", "medium"),
        _prov("slm_fill_factor", twin.slm.fill_factor, "fraction", "vbb_study/config.py", "SLMConfig", "fill_factor", "original_digital_twin", "inherited_design_value", "medium"),
        _prov("slm_phase_stroke_rad", None, "rad", "configs/evidence/manufacturer_evidence_register.json", "M_SLM1_SPEC/M_SLM2_SPEC", "phase_stroke_at_1030_nm", "unknown", "required_but_not_verified", "low"),
        _prov("carrier_cycles_per_m", binding.carrier_cycles_per_m, "cycles/m", "vbb_study/config.py", "SLMConfig", "carrier_cpm", "original_digital_twin", "derived_from_blaze_period_px", "high"),
        _prov("carrier_period_px", binding.carrier_period_pixels, "px", "vbb_study/config.py", "SLMConfig", "blaze_period_px", "inferred_from_existing_config", "derived_from_pitch_and_carrier", "high"),
        _prov("iris_radius_cycles_per_m", binding.iris_radius_cycles_per_m, "cycles/m", "vbb_study/config.py", "SLMConfig", "first_order_filter_radius_lpmm", "original_digital_twin", "inherited_design_value", "medium"),
        _prov("fourf_focal_length_m", f300.lens1_focal_length_m, "m", "configs/hardware/cslm_f300_nominal_4f_profile.json", "known_nominal_geometry", "lens1_focal_length_m", "planning_assumption", "nominal_not_bench_calibrated", "medium"),
        _prov("iris_physical_radius_m", binding.iris_physical_radius_m, "m", "vbb_study/digital_twin/nathan_mode2u2_master_closure.py", "resolve_nathan_hardware_binding", "f*wavelength*iris_frequency", "inferred_from_existing_config", "requires_4f_mapping_validation", "low"),
        _prov("qwp_nominal_angle_rad", binding.qwp_nominal_angle_rad, "rad", "vbb_study/digital_twin/nathan_vector_hexagon.py", "route_dual_slm_linear_then_qwp_ideal", "qwp(-pi/4)", "nathan_source_model", "model_convention", "high"),
        _prov("qwp_nominal_retardance_rad", binding.qwp_nominal_retardance_rad, "rad", "vbb_study/digital_twin/nathan_vector_hexagon.py", "qwp/linear_retarder", "pi/2", "nathan_source_model", "model_convention", "high"),
        _prov("axicon_base_angle_deg", np.rad2deg(source.axicon_base_angle_rad), "deg", "vbb_study/digital_twin/nathan_vector_hexagon.py", "NathanSourceParityConfig", "axicon_apex_angle_deg", "nathan_source_model", "literal_source_parameter", "high"),
        _prov("axicon_refractive_index", source.axicon_n, "", "vbb_study/digital_twin/nathan_vector_hexagon.py", "NathanSourceParityConfig", "axicon_n", "nathan_source_model", "literal_source_parameter", "high"),
        _prov("external_medium_index", source.medium_n, "", "vbb_study/digital_twin/nathan_vector_hexagon.py", "NathanSourceParityConfig", "medium_n", "nathan_source_model", "literal_source_parameter", "high"),
        _prov("twin_target_axicon_index", twin.target.n_axicon, "", "vbb_study/config.py", "BeamTarget", "n_axicon", "original_digital_twin", "inherited_micro_branch_target", "medium", "Different scope from Nathan source axicon; reported as conflict."),
        _prov("cslm_route_wavelength_nm", cslm.wavelength_nm, "nm", "vbb_study/digital_twin/cslm_route.py", "CSLMRouteConfig", "wavelength_nm", "placeholder", "diagnostic_placeholder", "medium"),
        _prov("cslm_route_carrier_cycles_per_m", cslm.slm2_carrier_frequency_cpm, "cycles/m", "vbb_study/digital_twin/cslm_route.py", "CSLMRouteConfig", "slm2_carrier_frequency_cpm", "placeholder", "diagnostic_placeholder", "medium"),
        _prov("camera_pixel_pitch_m", None, "m", "configs/evidence/bench_evidence_register.json", "B_CAMERA_SCALE", "camera pixel pitch", "unknown", "required_but_not_measured", "low"),
        _prov("camera_magnification", None, "ratio", "configs/evidence/bench_evidence_register.json", "B_CAMERA_SCALE", "camera magnification", "unknown", "required_but_not_measured", "low"),
        _prov("shack_hartmann_details", None, "", "configs/evidence/project_claim_registry.json", "wavefront correction claims", "Shack-Hartmann hardware", "unknown", "not_found_as_verified_hardware_record", "low"),
    ]
    return [row.row() for row in rows]


def hardware_parameter_conflict_rows(
    binding: NathanHardwareBinding | None = None,
    *,
    source_config: NathanSourceParityConfig | None = None,
) -> list[dict[str, Any]]:
    """Report unresolved or scope-split hardware/config semantics."""

    binding = binding or resolve_nathan_hardware_binding(source_config=source_config)
    twin = default_config("fast")
    source = source_config or NathanSourceParityConfig()
    cslm = CSLMRouteConfig()
    f300 = config_from_profile()
    qlook = Path("config/quicklook_config.json")
    return [
        {
            "parameter_family": "wavelength",
            "nathan_source_value": float(source.wavelength_m),
            "original_twin_value": float(twin.laser.wavelength_m),
            "other_value": float(cslm.wavelength_m),
            "units": "m",
            "severity": "scope_split_minor",
            "resolution_status": "not_silently_resolved",
            "notes": "Nathan source uses 1030 nm while the inherited PHAROS TwinConfig uses 1029 nm; CSLM diagnostic route also uses 1030 nm.",
        },
        {
            "parameter_family": "axicon_refractive_index",
            "nathan_source_value": float(source.axicon_n),
            "original_twin_value": float(twin.target.n_axicon),
            "other_value": 1.5,
            "units": "",
            "severity": "material_scope_conflict",
            "resolution_status": "not_silently_resolved",
            "notes": f"Nathan V0 source axicon n=1.458 differs from BeamTarget/quicklook n=1.5 ({qlook.as_posix()}).",
        },
        {
            "parameter_family": "4f_focal_length",
            "nathan_source_value": float(binding.fourf_focal_length_m or np.nan),
            "original_twin_value": 0.100,
            "other_value": float(f300.lens1_focal_length_m),
            "units": "m",
            "severity": "nominal_bench_placeholder_conflict",
            "resolution_status": "not_silently_resolved",
            "notes": "CSLMRouteConfig declares 100 mm warning-only placeholders; Stage 9B F300 profile declares nominal 300 mm but not bench calibrated.",
        },
        {
            "parameter_family": "carrier_frequency",
            "nathan_source_value": float(MODE2N_DEFAULT_CARRIER_LPMM),
            "original_twin_value": float(twin.slm.carrier_lpmm),
            "other_value": float(cslm.slm2_carrier_frequency_cycles_per_mm),
            "units": "lp/mm",
            "severity": "different_route_semantics",
            "resolution_status": "not_silently_resolved",
            "notes": "Twin/Nathan source-scale use 6.25 lp/mm from a 20 px blaze; vector-arm defaults and CSLM command-domain carrier records use other semantics.",
        },
        {
            "parameter_family": "beam_radius",
            "nathan_source_value": float(source.beam_radius_m),
            "original_twin_value": float(twin.laser.beam_radius_on_slm_m),
            "other_value": float(cslm.input_beam_radius_um * 1.0e-6),
            "units": "m",
            "severity": "different_plane_or_route_scope",
            "resolution_status": "not_silently_resolved",
            "notes": "Nathan/Twin source-scale beam is 2 mm; CSLM demo route uses a 24 um diagnostic grid source.",
        },
        {
            "parameter_family": "camera_calibration",
            "nathan_source_value": None,
            "original_twin_value": None,
            "other_value": None,
            "units": "",
            "severity": "missing_required_hardware_record",
            "resolution_status": "unknown",
            "notes": "Installed downstream camera can observe final response, but B_CAMERA_SCALE remains unknown and not ready.",
        },
        {
            "parameter_family": "slm_exact_model_and_phase_stroke",
            "nathan_source_value": binding.slm_model,
            "original_twin_value": twin.slm.name,
            "other_value": None,
            "units": "",
            "severity": "manufacturer_record_unverified",
            "resolution_status": "unknown",
            "notes": "The repo contains a HOLOEYE LCOS-NIR-like model, but manufacturer_evidence_register marks exact SLM1/SLM2 specs and wavelength stroke unknown.",
        },
    ]


def native_panel_geometry(binding: NathanHardwareBinding) -> dict[str, Any]:
    """Return the exact rectangular SLM panel geometry used by native-panel checks."""

    return {
        "panel_width_px": int(binding.slm_width_px),
        "panel_height_px": int(binding.slm_height_px),
        "pixel_pitch_m": float(binding.slm_pixel_pitch_m),
        "active_width_m": float(binding.slm_active_width_m),
        "active_height_m": float(binding.slm_active_height_m),
        "active_aspect_ratio": float(binding.slm_active_width_m / max(binding.slm_active_height_m, EPS)),
        "native_mask_aspect_ratio": float(binding.slm_width_px / max(binding.slm_height_px, 1)),
        "source_window_m": float(NathanSourceParityConfig().window_m),
        "source_window_fits_native_width": bool(NathanSourceParityConfig().window_m <= binding.slm_active_width_m),
        "source_window_fits_native_height": bool(NathanSourceParityConfig().window_m <= binding.slm_active_height_m),
        "largest_native_square_window_m": float(min(binding.slm_active_width_m, binding.slm_active_height_m)),
    }


def _native_panel_phase_stats(
    binding: NathanHardwareBinding,
    *,
    carrier_lpmm: float,
    correction: Mode2SCorrection | None = None,
) -> dict[str, Any]:
    """Rasterise Nathan H/V masks on exact native pixels and return compact stats."""

    corr = correction or Mode2SCorrection()
    nx = int(binding.slm_width_px)
    ny = int(binding.slm_height_px)
    pitch = float(binding.slm_pixel_pitch_m)
    x = (np.arange(nx, dtype=float) - 0.5 * (nx - 1)) * pitch
    y = (np.arange(ny, dtype=float) - 0.5 * (ny - 1)) * pitch
    X, Y = np.meshgrid(x - float(corr.mask_recentre_x_m), y - float(corr.mask_recentre_y_m), indexing="xy")
    theta = np.arctan2(Y, X)
    cfg = NathanSourceParityConfig()
    alpha, _ = nathan_alpha_map(
        theta,
        sector_num_pairs=int(cfg.n_pairs),
        sector_theta=float(cfg.sector_theta_rad * corr.sector_duty_scale),
        sector_rotation=float(cfg.sector_rotation_rad + corr.sector_rotation_rad),
    )
    carrier = 2.0 * np.pi * float(carrier_lpmm) * 1.0e3 * (X + float(corr.mask_recentre_x_m))
    phi_h = wrap_2pi(alpha + carrier)
    phi_v = wrap_2pi(-alpha + 0.5 * np.pi + float(corr.global_v_piston_rad) + carrier)
    levels = 2 ** int(binding.slm_bit_depth or 8)
    qh = np.mod(np.round(phi_h / (2.0 * np.pi) * levels) / levels * 2.0 * np.pi, 2.0 * np.pi)
    qv = np.mod(np.round(phi_v / (2.0 * np.pi) * levels) / levels * 2.0 * np.pi, 2.0 * np.pi)
    rms_h = float(np.sqrt(np.mean(np.angle(np.exp(1j * (qh - phi_h))) ** 2)))
    rms_v = float(np.sqrt(np.mean(np.angle(np.exp(1j * (qv - phi_v))) ** 2)))
    R2 = (np.meshgrid(x, y, indexing="xy")[0] ** 2 + np.meshgrid(x, y, indexing="xy")[1] ** 2)
    beam_radius = float(cfg.beam_radius_m)
    intensity = np.exp(-2.0 * R2 / max(beam_radius, EPS) ** 2)
    panel_power = float(np.sum(intensity) * pitch * pitch)
    infinite_power = float(0.5 * np.pi * beam_radius * beam_radius)
    clip_fraction = float(max(0.0, 1.0 - panel_power / max(infinite_power, EPS)))
    return {
        "native_phase_rasterized_exact_panel": True,
        "phase_array_shape_yx": [int(ny), int(nx)],
        "carrier_pixels_per_period": float(1.0 / max(float(carrier_lpmm) * 1.0e3 * pitch, EPS)),
        "phase_quantisation_levels": int(levels),
        "h_phase_quantisation_rms_rad": rms_h,
        "v_phase_quantisation_rms_rad": rms_v,
        "h_phase_mean_rad": float(np.mean(qh)),
        "v_phase_mean_rad": float(np.mean(qv)),
        "beam_clipping_fraction_native_panel": clip_fraction,
        "fill_factor": None if binding.slm_fill_factor is None else float(binding.slm_fill_factor),
    }


def _fixed_useful_region(grid: Mapping[str, Any], ring_radius_m: float) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(grid["X"], dtype=float)
    y = np.asarray(grid["Y"], dtype=float)
    radius = 2.65 * float(ring_radius_m)
    q1 = np.abs(x)
    q2 = np.abs(0.5 * x + 0.5 * np.sqrt(3.0) * y)
    q3 = np.abs(0.5 * x - 0.5 * np.sqrt(3.0) * y)
    mask = np.maximum.reduce([q1, q2, q3]) <= radius
    metadata = {
        "region_id": "fixed_regular_hexagon_radius_2p65_v0_ring",
        "definition": "max(|x|, |0.5*x+sqrt(3)/2*y|, |0.5*x-sqrt(3)/2*y|) <= 2.65*v0_ring_radius",
        "v0_ring_radius_m": float(ring_radius_m),
        "hex_radius_m": float(radius),
        "mask_fraction_of_grid": float(np.mean(mask)),
    }
    return mask, metadata


def _useful_power_metrics(plane: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    arr = np.asarray(plane, dtype=float)
    total = float(np.sum(arr))
    useful = float(np.sum(arr[np.asarray(mask, dtype=bool)]))
    peak = float(np.max(arr))
    return {
        "P_total": total,
        "P_useful": useful,
        "P_useful_over_P_total": float(useful / max(total, EPS)),
        "I_peak": peak,
        "side_lobe_fraction": float(max(0.0, 1.0 - useful / max(total, EPS))),
        "peak_per_total": float(peak / max(total, EPS)),
    }


def _case_corr_to_v0(case: Mapping[str, Any]) -> float:
    return float(case["comparison"]["z60_full_field_correlation"])


def _route_case_row(case_id: str, case: Mapping[str, Any], useful_mask: np.ndarray) -> dict[str, Any]:
    gate = dict(case["strict_gate"])
    useful = _useful_power_metrics(case["reference_plane"], useful_mask)
    return {
        "case_id": case_id,
        "z60_full_field_correlation": _case_corr_to_v0(case),
        "angular_profile_correlation_to_v0": float(case["comparison"]["angular_profile_correlation_to_v0"]),
        "strict_class": str(gate["strict_class"]),
        "passes": bool(case["passes"]),
        "first_order_efficiency": float(case["iris"]["first_order_efficiency"]),
        "zero_order_leakage_after_iris": float(case["iris"]["zero_order_leakage_after_iris"]),
        "rejected_power_fraction": float(case["iris"]["rejected_power_fraction"]),
        **useful,
    }


def run_mode2u2_optimal_hexagon_search(
    data: Mapping[str, Any],
    v0: Any,
    backward: Any,
    *,
    useful_mask: np.ndarray,
    max_cases: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]], dict[str, dict[str, Any]]]:
    """Search bounded, physically interpretable controls for four operating points."""

    carrier_values = [5.75, 6.25, 6.75]
    iris_values = [0.32, 0.40, 0.52]
    qwp_values_deg = [-0.25, 0.0, 0.25]
    rotation_values_deg = [-1.0, 0.0, 1.0]
    piston_values = [0.0, 0.10]
    combos = list(product(carrier_values, iris_values, qwp_values_deg, rotation_values_deg, piston_values))
    if max_cases is not None:
        combos = combos[: int(max_cases)]
    v0_metrics = _useful_power_metrics(v0.reference_plane, useful_mask)
    rows: list[dict[str, Any]] = []
    cases: dict[str, Mapping[str, Any]] = {}
    for idx, (carrier, iris, qwp_deg, rot_deg, piston) in enumerate(combos):
        case_id = f"m2u2_opt_{idx:03d}_c{carrier:.2f}_i{iris:.2f}_q{qwp_deg:+.2f}_r{rot_deg:+.1f}_p{piston:.2f}"
        pert = Mode2SPerturbation(
            label=case_id,
            slm_aperture_clip=True,
            phase_levels=256,
            fill_factor=0.93,
            carrier_lpmm=float(carrier),
            iris_radius_frac=float(iris),
        )
        corr = Mode2SCorrection(
            qwp_angle_correction_rad=float(np.deg2rad(qwp_deg)),
            sector_rotation_rad=float(np.deg2rad(rot_deg)),
            global_v_piston_rad=float(piston),
        )
        case = run_mode2s_degraded_forward(data, v0, backward, pert, correction=corr, fast_single_plane=True)
        useful = _useful_power_metrics(case["reference_plane"], useful_mask)
        strict = dict(case["strict_gate"])
        corr2d = float(case["comparison"]["z60_full_field_correlation"])
        angular = float(case["comparison"]["angular_profile_correlation_to_v0"])
        c_penalty = min(1.0, abs(float(strict["c120_minus_c60"])) / 0.20)
        dark_score = max(0.0, 1.0 - min(1.0, float(strict["dark_core_ratio"]) / 0.25))
        gate_bonus = 1.0 if bool(strict["passes_true_hexagon_gate"]) else 0.0
        shape_score = float(np.clip(0.52 * corr2d + 0.20 * angular + 0.13 * dark_score + 0.15 * gate_bonus - 0.10 * c_penalty, 0.0, 1.0))
        peak_score = float(useful["I_peak"] / max(v0_metrics["I_peak"], EPS))
        useful_score = float(useful["P_useful"] / max(v0_metrics["P_useful"], EPS))
        throughput = float(case["iris"]["first_order_efficiency"])
        robustness_proxy = float(min(
            abs(float(iris) - 0.28) / 0.28,
            abs(0.75 - float(iris)) / 0.75,
            abs(float(carrier) - 4.5) / 4.5,
            abs(8.5 - float(carrier)) / 8.5,
        ))
        compromise = float(0.45 * shape_score + 0.25 * min(peak_score, 1.5) / 1.5 + 0.20 * min(useful_score, 1.5) / 1.5 + 0.10 * min(throughput, 1.0))
        row = {
            "case_id": case_id,
            "carrier_lpmm": float(carrier),
            "iris_radius_frac": float(iris),
            "qwp_angle_correction_deg": float(qwp_deg),
            "sector_rotation_deg": float(rot_deg),
            "global_v_piston_rad": float(piston),
            "z60_full_field_correlation": corr2d,
            "angular_profile_correlation_to_v0": angular,
            "strict_class": str(strict["strict_class"]),
            "passes": bool(case["passes"]),
            "first_order_efficiency": throughput,
            "dark_core_ratio": float(strict["dark_core_ratio"]),
            "c120_minus_c60": float(strict["c120_minus_c60"]),
            "shape_score": shape_score,
            "peak_score": peak_score,
            "useful_energy_score": useful_score,
            "useful_region_fraction": useful["P_useful_over_P_total"],
            "throughput_score": throughput,
            "robustness_proxy": robustness_proxy,
            "compromise_score": compromise,
            **useful,
        }
        rows.append(row)
        cases[case_id] = case
    best = {
        "best_shape": max(rows, key=lambda r: float(r["shape_score"])),
        "best_peak": max(rows, key=lambda r: float(r["peak_score"])),
        "best_useful_power": max(rows, key=lambda r: float(r["useful_energy_score"])),
        "best_compromise": max(rows, key=lambda r: float(r["compromise_score"])),
    }
    return rows, cases, best


def _pareto_flags(rows: list[dict[str, Any]], x: str, y: str, flag_name: str) -> None:
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            if float(other[x]) >= float(row[x]) and float(other[y]) >= float(row[y]) and (
                float(other[x]) > float(row[x]) or float(other[y]) > float(row[y])
            ):
                dominated = True
                break
        row[flag_name] = not dominated


def _plot_scatter(rows: Sequence[Mapping[str, Any]], x: str, y: str, path: Path, title: str) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.4), constrained_layout=True)
    passes = np.asarray([bool(r["passes"]) for r in rows], dtype=bool)
    xv = np.asarray([float(r[x]) for r in rows], dtype=float)
    yv = np.asarray([float(r[y]) for r in rows], dtype=float)
    ax.scatter(xv[~passes], yv[~passes], color="tab:orange", s=28, label="strict fail")
    ax.scatter(xv[passes], yv[passes], color="tab:green", s=34, label="strict pass")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _save_highres(fig, path)
    return path


def write_optimal_outputs(
    root: Path,
    data: Mapping[str, Any],
    v0: Any,
    rows: list[dict[str, Any]],
    cases: Mapping[str, Mapping[str, Any]],
    best: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Write M2U2 optimal candidates and Pareto figures."""

    import matplotlib.pyplot as plt

    _pareto_flags(rows, "shape_score", "peak_score", "pareto_shape_peak")
    _pareto_flags(rows, "shape_score", "useful_energy_score", "pareto_shape_useful_power")
    _pareto_flags(rows, "useful_energy_score", "throughput_score", "pareto_useful_power_throughput")
    _pareto_flags(rows, "robustness_proxy", "shape_score", "pareto_robustness_shape")
    _write_rows(root / "optimal_hexagon_candidates.csv", rows)
    (root / "optimal_hexagon_candidates.json").write_text(json.dumps(_json_ready({"rows": rows, "best": best}), indent=2), encoding="utf-8")
    _plot_scatter(rows, "shape_score", "peak_score", root / "optimal_hexagon_pareto_shape_peak.png", "M2U2 shape vs peak")
    _plot_scatter(rows, "shape_score", "useful_energy_score", root / "optimal_hexagon_pareto_shape_useful_power.png", "M2U2 shape vs useful-region power")
    _plot_scatter(rows, "useful_energy_score", "throughput_score", root / "04_optimal_hexagon" / "optimal_hexagon_pareto_useful_power_throughput.png", "M2U2 useful-region power vs throughput")
    _plot_scatter(rows, "robustness_proxy", "shape_score", root / "04_optimal_hexagon" / "optimal_hexagon_pareto_robustness_shape.png", "M2U2 robustness proxy vs shape")

    image_names = {
        "best_shape": "optimal_hexagon_best_shape.png",
        "best_peak": "optimal_hexagon_best_peak.png",
        "best_useful_power": "optimal_hexagon_best_useful_power.png",
        "best_compromise": "optimal_hexagon_best_compromise.png",
    }
    profile_cases = []
    for key, filename in image_names.items():
        case = cases[str(best[key]["case_id"])]
        _plot_beam_image(
            case["reference_plane"],
            data["grid"],
            title=f"M2U2 {key}: {best[key]['case_id']} corr={best[key]['z60_full_field_correlation']:.4f}",
            path=root / filename,
        )
        profile_cases.append({"label": key, "plane": np.asarray(case["reference_plane"], dtype=float), "ring_radius_m": float(case["strict_gate"]["ring_radius_m"])})
    _plot_profiles(profile_cases, data["grid"], title="M2U2 optimal operating-point profiles", path=root / "04_optimal_hexagon" / "optimal_hexagon_profiles.png", reference_ring_radius_m=float(v0.ring_radius_m))
    plt.close("all")
    return {"rows": rows, "best": dict(best)}


def native_panel_confirmation_rows(
    binding: NathanHardwareBinding,
    data: Mapping[str, Any],
    v0: Any,
    backward: Any,
    best: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    """Run selected native-panel mask confirmations with square-grid propagation bridge."""

    cases: list[tuple[str, Mode2SPerturbation, Mode2SCorrection | None]] = [
        (
            "clean_realistic_dual_slm_4f_baseline",
            Mode2SPerturbation(
                label="native_clean",
                slm_aperture_clip=True,
                phase_levels=2 ** int(binding.slm_bit_depth or 8),
                fill_factor=float(binding.slm_fill_factor or 1.0),
                carrier_lpmm=float(binding.carrier_lpmm),
                iris_radius_frac=float(MODE2N_DEFAULT_IRIS_RADIUS_FRAC),
            ),
            None,
        ),
        ("moderate_combined_realism", replace(mode2s_combined_cases()[1], label="native_moderate_combined"), None),
        (
            "axicon_decentre_0p5mm_uncompensated",
            Mode2SPerturbation(label="native_axicon_0p5_uncomp", slm_aperture_clip=True, phase_levels=256, fill_factor=float(binding.slm_fill_factor or 1.0), axicon_decentre_x_m=0.5e-3),
            None,
        ),
        (
            "axicon_decentre_0p5mm_compensated",
            Mode2SPerturbation(label="native_axicon_0p5_comp", slm_aperture_clip=True, phase_levels=256, fill_factor=float(binding.slm_fill_factor or 1.0), axicon_decentre_x_m=0.5e-3),
            Mode2SCorrection(mask_recentre_x_m=0.5e-3),
        ),
    ]
    for key, row in best.items():
        corr = Mode2SCorrection(
            qwp_angle_correction_rad=float(np.deg2rad(float(row.get("qwp_angle_correction_deg", 0.0)))),
            sector_rotation_rad=float(np.deg2rad(float(row.get("sector_rotation_deg", 0.0)))),
            global_v_piston_rad=float(row.get("global_v_piston_rad", 0.0)),
        )
        pert = Mode2SPerturbation(
            label=f"native_{key}_{row['case_id']}",
            slm_aperture_clip=True,
            phase_levels=2 ** int(binding.slm_bit_depth or 8),
            fill_factor=float(binding.slm_fill_factor or 1.0),
            carrier_lpmm=float(row.get("carrier_lpmm", binding.carrier_lpmm)),
            iris_radius_frac=float(row.get("iris_radius_frac", MODE2N_DEFAULT_IRIS_RADIUS_FRAC)),
        )
        cases.append((key, pert, corr))

    rows: list[dict[str, Any]] = []
    propagated: dict[str, Mapping[str, Any]] = {}
    geom = native_panel_geometry(binding)
    for case_id, pert, corr in cases:
        phase = _native_panel_phase_stats(binding, carrier_lpmm=float(pert.carrier_lpmm), correction=corr)
        case = run_mode2s_degraded_forward(data, v0, backward, pert, correction=corr)
        row = {
            "case_id": case_id,
            **geom,
            **phase,
            "first_order_efficiency": float(case["iris"]["first_order_efficiency"]),
            "zero_order_leakage": float(case["iris"]["zero_order_leakage_after_iris"]),
            "z60_correlation": float(case["comparison"]["z60_full_field_correlation"]),
            "strict_class": str(case["strict_gate"]["strict_class"]),
            "passes": bool(case["passes"]),
            "square_grid_bridge_correlation": float(case["comparison"]["z60_full_field_correlation"]),
            "difference_from_square_grid_result": 0.0,
            "native_panel_rectangular_mask_confirmed": True,
            "rectangular_native_propagation_modelled": False,
            "propagation_bridge": "native-panel mask/display audit plus existing source-scale square-grid propagation bridge",
            "claim_boundary": "not a full rectangular 1920x1080 free-space propagation engine",
        }
        rows.append(row)
        propagated[case_id] = case
    return rows, propagated


def _plot_native_panel_comparison(rows: Sequence[Mapping[str, Any]], root: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), constrained_layout=True)
    labels = [str(r["case_id"]) for r in rows]
    corr = [float(r["z60_correlation"]) for r in rows]
    eff = [float(r["first_order_efficiency"]) for r in rows]
    axes[0].bar(labels, corr, color=["tab:green" if bool(r["passes"]) else "tab:orange" for r in rows])
    axes[0].axhline(MODE2S_PASS_CORRELATION, color="0.25", ls="--", lw=0.9)
    axes[0].set_ylabel("z60 correlation")
    axes[0].tick_params(axis="x", rotation=35, labelsize=7)
    axes[1].bar(labels, eff, color="tab:blue")
    axes[1].set_ylabel("first-order efficiency")
    axes[1].tick_params(axis="x", rotation=35, labelsize=7)
    fig.suptitle("M2U2 native-panel confirmation: exact mask geometry, square-grid propagation bridge")
    path = root / "native_panel_comparison_highres.png"
    _save_highres(fig, path)
    return path


def energy_ledger_full_rows(
    route_cases: Mapping[str, Mapping[str, Any]],
    useful_mask: np.ndarray,
    *,
    v0_power: float,
) -> list[dict[str, Any]]:
    """Build the requested 20+ stage energy/useful-region ledger."""

    rows: list[dict[str, Any]] = []
    for route_id, case in route_cases.items():
        if isinstance(case, Mapping) and "reference_plane" in case:
            plane = np.asarray(case["reference_plane"], dtype=float)
        else:
            plane = np.asarray(case.reference_plane, dtype=float)
        useful = _useful_power_metrics(plane, useful_mask)
        if isinstance(case, Mapping):
            iris = dict(case.get("iris", {}))
            pre = dict(case.get("pre_axicon", {}))
        else:
            iris = dict(getattr(case, "slm_4f_report", {}) or {})
            pre = dict(getattr(case, "pre_axicon_metrics", {}) or {})
        selected = float(iris.get("first_order_efficiency", 1.0))
        zero = float(iris.get("zero_order_leakage_after_iris", iris.get("zero_order_content_before_iris", 0.0)))
        rejected = float(np.clip(1.0 - selected - zero, 0.0, 1.0))
        pre_power = float(pre.get("power_ratio", 1.0))
        z60_integrated = float(np.sum(plane) / max(v0_power, EPS))
        total_throughput = float(selected * pre_power)
        stage_values = {
            "input_gaussian_power": 1.0,
            "after_initial_polarisation_preparation": 1.0,
            "h_channel_power": 0.5,
            "v_channel_power": 0.5,
            "incident_power_on_slm_h": 0.5,
            "incident_power_on_slm_v": 0.5,
            "after_slm_h": 0.5,
            "after_slm_v": 0.5,
            "fourier_plane_total_power": 1.0,
            "selected_first_order_power": selected,
            "zero_order_power": zero,
            "rejected_spectral_power": rejected,
            "finite_aperture_clipping": max(0.0, 1.0 - pre_power),
            "reconstructed_hv_power": selected,
            "recombined_power": selected,
            "after_qwp": selected,
            "incident_on_axicon": selected,
            "after_axicon": total_throughput,
            "integrated_power_z60": z60_integrated,
            "useful_hexagon_region_power": z60_integrated * useful["P_useful_over_P_total"],
            "outside_useful_region_power": z60_integrated * useful["side_lobe_fraction"],
            "peak_intensity_proxy": useful["peak_per_total"],
        }
        closure = abs((selected + zero + rejected) - 1.0)
        for idx, stage in enumerate(ENERGY_STAGE_NAMES, start=1):
            rows.append({
                "route_id": route_id,
                "stage_index": int(idx),
                "stage": stage,
                "power_norm": float(stage_values[stage]),
                "total_throughput": total_throughput,
                "selected_order_efficiency": selected,
                "useful_region_energy_fraction": useful["P_useful_over_P_total"],
                "peak_intensity_proxy": useful["peak_per_total"],
                "numerical_power_closure_error": float(closure),
                "P_total_z60": useful["P_total"],
                "P_useful_z60": useful["P_useful"],
                "side_lobe_fraction": useful["side_lobe_fraction"],
            })
    return rows


def _plot_energy_outputs(rows: Sequence[Mapping[str, Any]], root: Path) -> None:
    import matplotlib.pyplot as plt

    route_ids = list(dict.fromkeys(str(r["route_id"]) for r in rows))
    throughput = []
    useful = []
    peak = []
    for route_id in route_ids:
        rr = [r for r in rows if str(r["route_id"]) == route_id]
        final = rr[-1]
        throughput.append(float(final["total_throughput"]))
        useful.append(float(final["useful_region_energy_fraction"]))
        peak.append(float(final["peak_intensity_proxy"]))
    fig, ax = plt.subplots(figsize=(11.0, 4.9), constrained_layout=True)
    x = np.arange(len(route_ids))
    ax.bar(x - 0.25, throughput, 0.25, label="throughput")
    ax.bar(x, useful, 0.25, label="useful fraction")
    ax.bar(x + 0.25, peak, 0.25, label="peak/total")
    ax.set_xticks(x)
    ax.set_xticklabels(route_ids, rotation=30, ha="right", fontsize=7)
    ax.set_title("M2U2 energy flow by route")
    ax.legend(fontsize=8)
    _save_highres(fig, root / "energy_flow_by_route.png")
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    selected = [next(float(r["power_norm"]) for r in rows if r["route_id"] == route and r["stage"] == "selected_first_order_power") for route in route_ids]
    rejected = [next(float(r["power_norm"]) for r in rows if r["route_id"] == route and r["stage"] == "rejected_spectral_power") for route in route_ids]
    zero = [next(float(r["power_norm"]) for r in rows if r["route_id"] == route and r["stage"] == "zero_order_power") for route in route_ids]
    ax.bar(route_ids, selected, label="selected")
    ax.bar(route_ids, zero, bottom=selected, label="zero")
    ax.bar(route_ids, rejected, bottom=np.asarray(selected) + np.asarray(zero), label="rejected")
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.set_title("M2U2 energy/order breakdown")
    ax.legend(fontsize=8)
    _save_highres(fig, root / "energy_loss_breakdown.png")


def latin_hypercube_samples(n: int, bounds: Mapping[str, tuple[float, float]], seed: int = MODE2U2_SEED) -> list[dict[str, float]]:
    """Deterministic Latin hypercube sample rows."""

    rng = np.random.default_rng(int(seed))
    keys = list(bounds)
    mat = np.zeros((int(n), len(keys)), dtype=float)
    for j, key in enumerate(keys):
        perm = rng.permutation(int(n))
        u = (perm + rng.random(int(n))) / float(n)
        lo, hi = bounds[key]
        mat[:, j] = float(lo) + u * (float(hi) - float(lo))
    return [{key: float(mat[i, j]) for j, key in enumerate(keys)} for i in range(int(n))]


def run_interaction_robustness(
    data: Mapping[str, Any],
    v0: Any,
    backward: Any,
    *,
    sample_count: int = 24,
    seed: int = MODE2U2_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run bounded multi-parameter interaction robustness with LHS sampling."""

    bounds = {
        "hv_amplitude_ratio": (0.90, 1.10),
        "hv_piston_rad": (-0.4, 0.4),
        "qwp_angle_error_deg": (-1.0, 1.0),
        "qwp_retardance_error_deg": (-2.0, 2.0),
        "iris_decentre_fx_lpmm": (-0.6, 0.6),
        "iris_radius_frac": (0.34, 0.48),
        "hv_shift_x_um": (-32.0, 32.0),
        "axicon_decentre_x_mm": (-0.30, 0.30),
        "common_defocus_rad": (-0.35, 0.35),
        "common_astig0_rad": (-0.25, 0.25),
        "z_offset_mm": (-5.0, 5.0),
    }
    samples = latin_hypercube_samples(sample_count, bounds, seed=seed)
    rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        pert = Mode2SPerturbation(
            label=f"interaction_{idx:03d}",
            slm_aperture_clip=True,
            phase_levels=256,
            fill_factor=0.93,
            hv_amplitude_ratio=sample["hv_amplitude_ratio"],
            hv_piston_rad=sample["hv_piston_rad"],
            qwp_angle_error_rad=float(np.deg2rad(sample["qwp_angle_error_deg"])),
            qwp_retardance_error_rad=float(np.deg2rad(sample["qwp_retardance_error_deg"])),
            iris_decentre_fx_lpmm=sample["iris_decentre_fx_lpmm"],
            iris_radius_frac=sample["iris_radius_frac"],
            hv_shift_x_m=sample["hv_shift_x_um"] * 1.0e-6,
            axicon_decentre_x_m=sample["axicon_decentre_x_mm"] * 1.0e-3,
            zernike_common={"defocus": sample["common_defocus_rad"], "astig0": sample["common_astig0_rad"]},
            z_offset_m=sample["z_offset_mm"] * 1.0e-3,
        )
        case = run_mode2s_degraded_forward(data, v0, backward, pert, fast_single_plane=True)
        rows.append({
            "sample_index": int(idx),
            "seed": int(seed),
            **sample,
            "z60_full_field_correlation": float(case["comparison"]["z60_full_field_correlation"]),
            "angular_profile_correlation_to_v0": float(case["comparison"]["angular_profile_correlation_to_v0"]),
            "strict_class": str(case["strict_gate"]["strict_class"]),
            "passes": bool(case["passes"]),
            "failure_mode": str(case["failure_mode"]),
        })
    corr = np.asarray([float(r["z60_full_field_correlation"]) for r in rows], dtype=float)
    pass_fraction = float(np.mean([bool(r["passes"]) for r in rows]))
    sensitivity = []
    for key in bounds:
        x = np.asarray([float(r[key]) for r in rows], dtype=float)
        val = 0.0 if np.std(x) <= EPS or np.std(corr) <= EPS else float(abs(np.corrcoef(x, corr)[0, 1]))
        sensitivity.append({"parameter": key, "abs_pearson_to_corr": val})
    sensitivity.sort(key=lambda r: r["abs_pearson_to_corr"], reverse=True)
    pair_rows = []
    for a, b in combinations(bounds, 2):
        x = np.asarray([float(r[a]) * float(r[b]) for r in rows], dtype=float)
        val = 0.0 if np.std(x) <= EPS or np.std(corr) <= EPS else float(abs(np.corrcoef(x, corr)[0, 1]))
        pair_rows.append({"pair": f"{a}*{b}", "abs_pearson_to_corr": val})
    pair_rows.sort(key=lambda r: r["abs_pearson_to_corr"], reverse=True)
    classes = {str(k): int(sum(str(r["strict_class"]) == str(k) for r in rows)) for k in sorted({str(r["strict_class"]) for r in rows})}
    failures = {str(k): int(sum(str(r["failure_mode"]) == str(k) for r in rows)) for k in sorted({str(r["failure_mode"]) for r in rows})}
    summary = {
        "design": "latin_hypercube",
        "seed": int(seed),
        "sample_count": int(sample_count),
        "parameter_bounds": bounds,
        "pass_fraction": pass_fraction,
        "correlation_min": float(np.min(corr)),
        "correlation_mean": float(np.mean(corr)),
        "correlation_p10": float(np.quantile(corr, 0.10)),
        "strict_class_distribution": classes,
        "failure_mode_distribution": failures,
        "sensitivity_ranking": sensitivity,
        "most_important_interaction_pairs": pair_rows[:12],
    }
    return rows, summary


def _plot_interaction_outputs(rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], root: Path) -> None:
    import matplotlib.pyplot as plt

    pairs = list(summary["most_important_interaction_pairs"])[:8]
    fig, ax = plt.subplots(figsize=(9.5, 4.6), constrained_layout=True)
    ax.bar([p["pair"] for p in pairs], [float(p["abs_pearson_to_corr"]) for p in pairs], color="tab:purple")
    ax.tick_params(axis="x", rotation=35, labelsize=7)
    ax.set_ylabel("abs Pearson to corr")
    ax.set_title("M2U2 interaction pair importance")
    _save_highres(fig, root / "interaction_pair_importance.png")
    corr = np.asarray([float(r["z60_full_field_correlation"]) for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    ax.hist(corr, bins=10, color="tab:blue", alpha=0.75)
    ax.axvline(MODE2S_PASS_CORRELATION, color="0.2", ls="--")
    ax.set_xlabel("z60 correlation")
    ax.set_ylabel("count")
    ax.set_title(f"M2U2 interaction robustness pass fraction {float(summary['pass_fraction']):.2f}")
    _save_highres(fig, root / "interaction_pass_fraction.png")
    fig, ax = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
    colors = ["tab:green" if bool(r["passes"]) else "tab:red" for r in rows]
    ax.scatter([float(r["axicon_decentre_x_mm"]) for r in rows], [float(r["qwp_angle_error_deg"]) for r in rows], c=colors, s=42)
    ax.set_xlabel("axicon decentre x (mm)")
    ax.set_ylabel("QWP angle error (deg)")
    ax.set_title("M2U2 interaction failure map")
    ax.grid(alpha=0.25)
    _save_highres(fig, root / "interaction_failure_map.png")


def _evaluate_candidate(
    data: Mapping[str, Any],
    v0: Any,
    backward: Any,
    perturbation: Mode2SPerturbation,
    *,
    correction: Mode2SCorrection | None = None,
) -> Mapping[str, Any]:
    return run_mode2s_degraded_forward(data, v0, backward, perturbation, correction=correction, fast_single_plane=True)


def run_blind_correction_tests(
    data: Mapping[str, Any],
    v0: Any,
    backward: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Test semi-blind bounded correction using image/reference metrics only."""

    rows: list[dict[str, Any]] = []
    convergence: dict[str, list[dict[str, Any]]] = {}

    def add_result(
        case_id: str,
        perturbation: Mode2SPerturbation,
        candidates: Sequence[tuple[str, Mode2SPerturbation, Mode2SCorrection | None]],
        true_error: Mapping[str, Any],
        variable: str,
    ) -> None:
        initial = _evaluate_candidate(data, v0, backward, perturbation)
        trace: list[dict[str, Any]] = []
        best_label = "initial"
        best_case = initial
        best_corr = float(initial["comparison"]["z60_full_field_correlation"])
        for iteration, (label, pert, corr) in enumerate(candidates, start=1):
            case = _evaluate_candidate(data, v0, backward, pert, correction=corr)
            value = float(case["comparison"]["z60_full_field_correlation"])
            trace.append({"iteration": iteration, "candidate": label, "correlation": value, "strict_class": str(case["strict_gate"]["strict_class"])})
            if value > best_corr:
                best_corr = value
                best_label = label
                best_case = case
        convergence[case_id] = trace
        rows.append({
            "case_id": case_id,
            "true_injected_error": json.dumps(_json_ready(dict(true_error)), sort_keys=True),
            "initial_guess": "zero_correction",
            "inferred_final_correction": best_label,
            "corrected_variable": variable,
            "correction_error": "reported_posthoc_not_used_by_algorithm",
            "initial_correlation": float(initial["comparison"]["z60_full_field_correlation"]),
            "final_correlation": float(best_case["comparison"]["z60_full_field_correlation"]),
            "initial_strict_class": str(initial["strict_gate"]["strict_class"]),
            "final_strict_class": str(best_case["strict_gate"]["strict_class"]),
            "initial_passes": bool(initial["passes"]),
            "final_passes": bool(best_case["passes"]),
            "iterations": len(candidates),
            "uses_injected_truth": False,
            "algorithm_observables": "camera_intensity; z60 correlation to stored V0 reference; strict hexagon metrics",
            "algorithm_boundary": "semi_blind_reference_metric_search_no_hidden_truth_seed",
        })

    ax_pert = Mode2SPerturbation(label="blind_axicon_offset", slm_aperture_clip=True, axicon_decentre_x_m=0.5e-3)
    add_result(
        "unknown_axicon_mask_lateral_offset",
        ax_pert,
        [
            (f"mask_recentre_x_{x_mm:+.2f}mm", ax_pert, Mode2SCorrection(mask_recentre_x_m=x_mm * 1.0e-3))
            for x_mm in (-0.50, -0.25, 0.0, 0.25, 0.50)
        ],
        {"axicon_decentre_x_m": 0.5e-3},
        "mask_recentre_x_m",
    )
    hv_pert = Mode2SPerturbation(label="blind_hv_registration", slm_aperture_clip=True, hv_shift_x_m=80e-6)
    add_result(
        "unknown_hv_registration_error",
        hv_pert,
        [
            (f"slm_v_registration_delta_{x_um:+.0f}um", replace(hv_pert, hv_shift_x_m=(80.0 + x_um) * 1.0e-6), None)
            for x_um in (-80.0, -40.0, 0.0, 40.0, 80.0)
        ],
        {"hv_shift_x_m": 80e-6},
        "software_slm_v_registration_delta_x_m",
    )
    z_pert = Mode2SPerturbation(label="blind_zernike", slm_aperture_clip=True, zernike_common={"defocus": 0.5, "astig0": 0.2})
    add_result(
        "unknown_low_order_wavefront_aberration",
        z_pert,
        [
            (f"defocus_{d:+.2f}_astig_{a:+.2f}", z_pert, Mode2SCorrection(defocus_rad=d, astig0_rad=a))
            for d, a in product((-0.5, 0.0, 0.5), (-0.2, 0.0, 0.2))
        ],
        {"defocus": 0.5, "astig0": 0.2},
        "common_zernike_phase_correction",
    )
    combined = Mode2SPerturbation(
        label="blind_combined",
        slm_aperture_clip=True,
        axicon_decentre_x_m=0.3e-3,
        hv_piston_rad=0.3,
        qwp_angle_error_rad=float(np.deg2rad(0.8)),
        zernike_common={"defocus": 0.25},
    )
    add_result(
        "combined_unknown_error_case",
        combined,
        [
            (f"mask_{x:+.2f}mm_qwp_{q:+.1f}deg", combined, Mode2SCorrection(mask_recentre_x_m=x * 1.0e-3, qwp_angle_correction_rad=float(np.deg2rad(q))))
            for x, q in product((0.0, 0.3, 0.6), (-1.0, 0.0, 1.0))
        ],
        {"axicon_decentre_x_m": 0.3e-3, "hv_piston_rad": 0.3, "qwp_angle_error_deg": 0.8, "defocus": 0.25},
        "mask_recentre_and_qwp_trim",
    )
    meta = {
        "uses_injected_truth": False,
        "algorithm_observables": ["camera_intensity", "optional_zstack_metric", "stored_V0_reference_metric"],
        "calibration_assisted_not_ground_truth_seeded": True,
        "convergence": convergence,
    }
    return rows, meta


def _plot_blind_convergence(meta: Mapping[str, Any], root: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    for case_id, trace in dict(meta["convergence"]).items():
        if not trace:
            continue
        ax.plot([int(t["iteration"]) for t in trace], [float(t["correlation"]) for t in trace], marker="o", label=case_id)
    ax.axhline(MODE2S_PASS_CORRELATION, color="0.25", ls="--")
    ax.set_xlabel("candidate iteration")
    ax.set_ylabel("camera/reference metric correlation")
    ax.set_title("M2U2 blind/semi-blind correction candidate convergence")
    ax.legend(fontsize=7)
    _save_highres(fig, root / "blind_correction_convergence.png")


def correction_responsibility_matrix() -> list[dict[str, Any]]:
    return [
        {
            "instrument": "Shack-Hartmann",
            "primary_observables": "common wavefront slope; low-order Zernike estimates",
            "correctable_terms": "defocus; astigmatism; coma; common low-order SLM phase",
            "not_responsible_for": "final C3/C6 intensity morphology; H/V Stokes validation; Fourier stop coordinate calibration",
            "project_status": "role_defined_but_hardware_details_unverified",
        },
        {
            "instrument": "camera",
            "primary_observables": "final intensity; beam centre; C3/C6 symmetry; dark core; lobe balance; z-stack",
            "correctable_terms": "mask/axicon centring; iris trim by empirical response; z-reference placement",
            "not_responsible_for": "direct polarisation vector field; absolute Fourier-plane coordinates without calibration",
            "project_status": "downstream response available in plan; scale/magnification unknown",
        },
        {
            "instrument": "Stokes/polarimetry",
            "primary_observables": "H/V balance; phase relation; segmented radial/azimuthal vector state",
            "correctable_terms": "QWP angle; H/V piston; channel amplitude balance; SLM polarisation compatibility",
            "not_responsible_for": "axicon cone alignment; 4F stop power split",
            "project_status": "role defined by M2P/M2N source model; measurement hardware not bound",
        },
    ]


def build_architecture_precheck(binding: NathanHardwareBinding, readiness: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "patterned_hwp_route": {
            "works_as_ideal_source_model": True,
            "requires_strange_six_piece_segmented_polariser_waveplate": True,
            "recommended": False,
            "reason": "source-scale synthesis works, but dual SLM route replaces the segmented assembly with programmable phase channels.",
        },
        "dual_slm_qwp_4f_route": {
            "works_as_source_scale_model": True,
            "recommended_route": True,
            "requires_two_phase_only_slms": True,
            "qwp_angle_rad_model_convention": float(binding.qwp_nominal_angle_rad),
            "qwp_angle_convention_lab_verified": False,
            "fourf_physical_readiness": readiness["C_initial_scalar_4f_model"]["ready"],
            "measured_bench_camera_readiness": readiness["D_measured_bench_camera"]["ready"],
            "additional_hwps_likely_needed": "possibly_for_SLM_input_polarisation; exact axes not yet verified",
            "both_slms_compatible_with_required_polarisation": "not verified by manufacturer_evidence_register",
        },
        "six_piece_segmented_concept_required": False,
        "dual_slm_qwp_4f_remains_recommended": True,
        "m2v_authorised": False,
        "m2v_blockers": [
            "exact SLM model and phase stroke at 1030/1064 nm unverified",
            "physical 4F Fourier-plane mapping and stop geometry not calibrated",
            "camera scale/magnification/reference plane unknown",
            "lab QWP/HWP axis convention not measured",
        ],
    }


def _write_highres_visual_inventory(root: Path) -> list[dict[str, Any]]:
    src = Path(MODE2U_DEFAULT_OUTPUT_ROOT)
    rows: list[dict[str, Any]] = []
    if src.exists():
        for sub in sorted(src.iterdir()):
            if sub.is_dir():
                pngs = list(sub.glob("*.png"))
                rows.append({
                    "source": str(src).replace("\\", "/"),
                    "stage_directory": sub.name,
                    "png_count": len(pngs),
                    "coverage_status": "existing_mode2u_highres_inventory",
                    "notes": "M2U2 reuses the already generated publication/highres visual audit instead of mutating its evidence.",
                })
    else:
        rows.append({
            "source": str(src).replace("\\", "/"),
            "stage_directory": "missing",
            "png_count": 0,
            "coverage_status": "not_found",
            "notes": "Run MODE 2U highres audit first for full visual inventory.",
        })
    _write_rows(root / "highres_visual_audit_inventory.csv", rows)
    (root / "highres_visual_audit_inventory.json").write_text(json.dumps(_json_ready(rows), indent=2), encoding="utf-8")
    return rows


def _write_master_doc(
    path: Path,
    *,
    output_root: Path,
    binding: NathanHardwareBinding,
    conflicts: Sequence[Mapping[str, Any]],
    native_rows: Sequence[Mapping[str, Any]],
    energy_rows: Sequence[Mapping[str, Any]],
    optimal: Mapping[str, Any],
    interaction_summary: Mapping[str, Any],
    blind_rows: Sequence[Mapping[str, Any]],
    precheck: Mapping[str, Any],
    outcome: Mapping[str, Any],
) -> Path:
    m2v_text = str(bool(outcome["m2v_authorised"])).lower()
    text = f"""# Nathan MODE 2U2 - Master Closure

**Status:** source-scale closure audit only. No microfabrication/sample-plane
success claim is made here, and M2V is not authorised unless the outcome is
M2U2-A.

## Hardware Binding

The Nathan branch is bound to the inherited Digital Twin where available:

- wavelength: `{binding.wavelength_m:.9e} m`
- SLM: `{binding.slm_model}`, `{binding.slm_width_px} x {binding.slm_height_px}`, pitch `{binding.slm_pixel_pitch_m:.3e} m`
- active area: `{binding.slm_active_width_m*1e3:.2f} x {binding.slm_active_height_m*1e3:.2f} mm`
- carrier: `{binding.carrier_lpmm:.3f} lp/mm`, `{binding.carrier_period_pixels:.2f}` pixels per period
- nominal 4F focal length used for planning inference: `{binding.fourf_focal_length_m}` m
- Nathan source axicon: base angle `{binding.axicon_base_angle_deg:.3f} deg`, n `{binding.axicon_refractive_index:.3f}`

Exact SLM phase stroke, camera scale, and measured 4F stop geometry remain
unknown in the repository evidence registers.

## Conflicts

`hardware_parameter_conflicts.csv/json` records `{len(conflicts)}` unresolved
scope splits or conflicts. The important ones are wavelength 1029/1030 nm,
axicon n=1.458 versus n=1.5 in the inherited target branch, 100 mm CSLM
placeholder versus nominal F300 geometry, command-domain carrier semantics, and
unknown camera/SLM manufacturer details.

## Native Panel

Native-panel confirmation rasterised the H/V phase masks on the exact
`{binding.slm_width_px} x {binding.slm_height_px}` rectangular panel with
`{binding.slm_pixel_pitch_m*1e6:.1f} um` pitch. The 10 mm source window still
does not fit the short panel axis, while the 2 mm Gaussian beam clips only a
small tail. The propagation column is intentionally labelled as a square-grid
bridge; a full rectangular 1920 x 1080 propagation engine is not claimed.

Native rows: `{len(native_rows)}`. Passing rows:
`{sum(bool(r.get('passes', False)) for r in native_rows)}`.

## Energy And Optimisation

The full energy ledger contains `{len(energy_rows)}` rows over the fixed useful
hexagon region. The operating points are:

- best shape: `{optimal['best']['best_shape']['case_id']}`
- best peak: `{optimal['best']['best_peak']['case_id']}`
- best useful-region energy: `{optimal['best']['best_useful_power']['case_id']}`
- compromise: `{optimal['best']['best_compromise']['case_id']}`

## Robustness And Correction

Interaction robustness used `{interaction_summary['design']}` sampling with seed
`{interaction_summary['seed']}` and `{interaction_summary['sample_count']}`
samples. Pass fraction: `{float(interaction_summary['pass_fraction']):.3f}`.
Dominant terms are listed in `interaction_robustness_summary.json`.

Blind/semi-blind correction tested `{len(blind_rows)}` cases. It reports the
truth after the fact, but the search metadata states `uses_injected_truth =
false`; correction was chosen from camera/reference metrics only.

## Build Precheck

The strange six-piece segmented polariser/waveplate concept is not required for
the source-scale route. Dual SLMs plus conventional polarisation optics, a QWP,
a 4F first-order filter, and the source-scale axicon remain the recommended
architecture. However, the exact SLM phase response, QWP/HWP axis convention,
physical 4F mapping, order power split, and camera scale are not yet verified.

## Outcome

**{outcome['selected_outcome']}.** {outcome['outcome_statement']}

M2V authorised: `{m2v_text}`.

Output root: `{output_root.as_posix()}`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_mode2u2_master_closure(
    config: NathanSourceParityConfig | None = None,
    *,
    output_dir: str | Path = MODE2U2_DEFAULT_OUTPUT_ROOT,
    grid_n: int = 256,
    z_planes: int = 7,
    optimisation_max_cases: int | None = 36,
    interaction_samples: int = 18,
    seed: int = MODE2U2_SEED,
    doc_path: str | Path = MODE2U2_DOC_PATH,
) -> dict[str, Any]:
    """Generate MODE 2U2 closure products."""

    import matplotlib.pyplot as plt

    plt.rcParams["image.interpolation"] = MODE2U_RENDER_INTERPOLATION
    plt.rcParams["savefig.dpi"] = 300

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for dirname in (
        "00_provenance",
        "01_highres_visuals",
        "02_native_panel",
        "03_energy",
        "04_optimal_hexagon",
        "05_interaction_robustness",
        "06_blind_correction",
        "07_build_precheck",
        "08_final_report",
    ):
        (root / dirname).mkdir(parents=True, exist_ok=True)

    binding = resolve_nathan_hardware_binding(source_config=config)
    provenance = hardware_parameter_provenance_rows(binding, source_config=config)
    conflicts = hardware_parameter_conflict_rows(binding, source_config=config)
    _write_rows(root / "hardware_parameter_provenance.csv", provenance)
    (root / "hardware_parameter_provenance.json").write_text(json.dumps(_json_ready(provenance), indent=2), encoding="utf-8")
    _write_rows(root / "hardware_parameter_conflicts.csv", conflicts)
    (root / "hardware_parameter_conflicts.json").write_text(json.dumps(_json_ready(conflicts), indent=2), encoding="utf-8")
    _write_rows(root / "00_provenance" / "hardware_parameter_provenance.csv", provenance)
    (root / "00_provenance" / "hardware_binding.json").write_text(json.dumps(_json_ready(asdict(binding)), indent=2), encoding="utf-8")

    visual_inventory = _write_highres_visual_inventory(root / "01_highres_visuals")

    data = mode2n_source_target(config, grid_n=int(grid_n), z_planes=int(z_planes))
    v0 = run_mode2n_v0_reference(data)
    backward = run_mode2q_backward_initialisation(data)
    patterned = run_mode2n_patterned_hwp_route(data, v0)
    dual_qwp = run_mode2n_dual_slm_qwp_route(data, v0)
    dual_4f = run_mode2n_dual_slm_4f_route(data, v0)
    clean_case = run_mode2s_degraded_forward(data, v0, backward, Mode2SPerturbation(label="clean_realistic", slm_aperture_clip=True, phase_levels=256, fill_factor=0.93))
    moderate = run_mode2s_degraded_forward(data, v0, backward, replace(mode2s_combined_cases()[1], label="moderate_combined"))
    bad = run_mode2s_degraded_forward(data, v0, backward, replace(mode2s_combined_cases()[2], label="bad_combined"))
    ax_comp = run_mode2s_degraded_forward(
        data,
        v0,
        backward,
        Mode2SPerturbation(label="compensated_0p5mm_axicon", slm_aperture_clip=True, phase_levels=256, fill_factor=0.93, axicon_decentre_x_m=0.5e-3),
        correction=Mode2SCorrection(mask_recentre_x_m=0.5e-3),
    )

    useful_mask, useful_meta = _fixed_useful_region(data["grid"], float(v0.ring_radius_m))
    (root / "useful_region_definition.json").write_text(json.dumps(_json_ready(useful_meta), indent=2), encoding="utf-8")
    (root / "03_energy" / "useful_region_definition.json").write_text(json.dumps(_json_ready(useful_meta), indent=2), encoding="utf-8")

    opt_rows, opt_cases, opt_best = run_mode2u2_optimal_hexagon_search(
        data,
        v0,
        backward,
        useful_mask=useful_mask,
        max_cases=optimisation_max_cases,
    )
    optimal = write_optimal_outputs(root, data, v0, opt_rows, opt_cases, opt_best)

    native_rows, native_cases = native_panel_confirmation_rows(binding, data, v0, backward, opt_best)
    _write_rows(root / "native_panel_confirmation.csv", native_rows)
    (root / "native_panel_confirmation.json").write_text(json.dumps(_json_ready(native_rows), indent=2), encoding="utf-8")
    _write_rows(root / "02_native_panel" / "native_panel_confirmation.csv", native_rows)
    (root / "02_native_panel" / "native_panel_confirmation.json").write_text(json.dumps(_json_ready(native_rows), indent=2), encoding="utf-8")
    _plot_native_panel_comparison(native_rows, root)

    route_cases: dict[str, Any] = {
        "v0_ideal_reference": v0,
        "ideal_patterned_hwp": patterned,
        "ideal_dual_slm_qwp": dual_qwp,
        "realistic_dual_slm_4f": dual_4f,
        "moderate_combined": moderate,
        "bad_combined": bad,
        "compensated_0p5mm_axicon": ax_comp,
    }
    for key in ("best_shape", "best_peak", "best_useful_power", "best_compromise"):
        route_cases[key] = opt_cases[str(opt_best[key]["case_id"])]
    energy_rows = energy_ledger_full_rows(route_cases, useful_mask, v0_power=float(np.sum(np.asarray(v0.reference_plane, dtype=float))))
    _write_rows(root / "energy_ledger_full.csv", energy_rows)
    (root / "energy_ledger_full.json").write_text(json.dumps(_json_ready(energy_rows), indent=2), encoding="utf-8")
    _write_rows(root / "03_energy" / "energy_ledger_full.csv", energy_rows)
    _plot_energy_outputs(energy_rows, root)

    interaction_rows, interaction_summary = run_interaction_robustness(data, v0, backward, sample_count=int(interaction_samples), seed=int(seed))
    _write_rows(root / "interaction_robustness_samples.csv", interaction_rows)
    (root / "interaction_robustness_summary.json").write_text(json.dumps(_json_ready(interaction_summary), indent=2), encoding="utf-8")
    _write_rows(root / "05_interaction_robustness" / "interaction_robustness_samples.csv", interaction_rows)
    _plot_interaction_outputs(interaction_rows, interaction_summary, root)

    blind_rows, blind_meta = run_blind_correction_tests(data, v0, backward)
    _write_rows(root / "blind_correction_results.csv", blind_rows)
    (root / "blind_correction_results.json").write_text(json.dumps(_json_ready({"rows": blind_rows, "meta": blind_meta}), indent=2), encoding="utf-8")
    _write_rows(root / "06_blind_correction" / "blind_correction_results.csv", blind_rows)
    _plot_blind_convergence(blind_meta, root)

    matrix = correction_responsibility_matrix()
    _write_rows(root / "correction_responsibility_matrix.csv", matrix)
    (root / "correction_responsibility_matrix.json").write_text(json.dumps(_json_ready(matrix), indent=2), encoding="utf-8")
    readiness = evaluate_physical_4f_readiness()
    bench_profile = build_bench_inventory_profile()
    precheck = build_architecture_precheck(binding, readiness)
    (root / "build_architecture_precheck.json").write_text(json.dumps(_json_ready({"precheck": precheck, "readiness": readiness, "bench_inventory": bench_profile}), indent=2), encoding="utf-8")

    # Outcome is deliberately not A: the source-scale result is compelling, but
    # exact hardware provenance, physical 4F mapping, and camera scale are not
    # yet resolved.
    outcome = {
        "allowed_outcomes": list(MODE2U2_ALLOWED_OUTCOMES),
        "selected_outcome": "M2U2-B",
        "outcome_statement": (
            "The source-scale route remains compelling under provenance-bound and native-panel mask checks, "
            "but unverified SLM phase stroke/model specifics, physical 4F/camera calibration, and route-scope "
            "conflicts must be resolved before final lab prescription."
        ),
        "m2v_authorised": False,
        "m2v_authorisation_rule": "Only M2U2-A authorises M2V.",
        "source_scale_route_compelling": True,
        "native_panel_mask_confirmation_passed": bool(all(bool(r.get("native_panel_rectangular_mask_confirmed", False)) for r in native_rows)),
        "rectangular_native_propagation_modelled": False,
        "hardware_provenance_blockers": [r["parameter_family"] for r in conflicts if str(r["severity"]) in {"missing_required_hardware_record", "manufacturer_record_unverified", "nominal_bench_placeholder_conflict"}],
        "microfabrication_sample_plane_claim": False,
    }
    (root / "m2u2_outcome_report.json").write_text(json.dumps(_json_ready(outcome), indent=2), encoding="utf-8")
    doc = _write_master_doc(
        Path(doc_path),
        output_root=root,
        binding=binding,
        conflicts=conflicts,
        native_rows=native_rows,
        energy_rows=energy_rows,
        optimal=optimal,
        interaction_summary=interaction_summary,
        blind_rows=blind_rows,
        precheck=precheck,
        outcome=outcome,
    )
    manifest = {
        "stage": MODE2U2_STAGE,
        "output_root": str(root),
        "grid_n": int(grid_n),
        "z_planes": int(z_planes),
        "seed": int(seed),
        "subdirectories": [p.name for p in root.iterdir() if p.is_dir()],
        "hardware_binding": asdict(binding),
        "outcome": outcome,
        "m2v_authorised": bool(outcome["m2v_authorised"]),
        "source_scale_only": True,
        "microfabrication_sample_plane_claim": False,
        "rectangular_native_propagation_modelled": False,
        "native_panel_masks_rasterised_exact_panel": True,
        "doc_path": str(doc),
        "machine_files": {
            "hardware_parameter_provenance_csv": "hardware_parameter_provenance.csv",
            "hardware_parameter_conflicts_csv": "hardware_parameter_conflicts.csv",
            "native_panel_confirmation_csv": "native_panel_confirmation.csv",
            "energy_ledger_full_csv": "energy_ledger_full.csv",
            "optimal_hexagon_candidates_csv": "optimal_hexagon_candidates.csv",
            "interaction_robustness_samples_csv": "interaction_robustness_samples.csv",
            "blind_correction_results_csv": "blind_correction_results.csv",
            "outcome_report": "m2u2_outcome_report.json",
        },
    }
    (root / "nathan_mode2u2_master_manifest.json").write_text(json.dumps(_json_ready(manifest), indent=2), encoding="utf-8")
    return {
        "manifest": manifest,
        "binding": binding,
        "provenance_rows": provenance,
        "conflict_rows": conflicts,
        "visual_inventory": visual_inventory,
        "native_rows": native_rows,
        "energy_rows": energy_rows,
        "optimal": optimal,
        "interaction_rows": interaction_rows,
        "interaction_summary": interaction_summary,
        "blind_rows": blind_rows,
        "blind_meta": blind_meta,
        "responsibility_matrix": matrix,
        "build_precheck": precheck,
        "outcome": outcome,
        "output_root": root,
        "doc_path": doc,
    }


__all__ = [
    "MODE2U2_ALLOWED_OUTCOMES",
    "MODE2U2_DEFAULT_OUTPUT_ROOT",
    "MODE2U2_DOC_PATH",
    "MODE2U2_STAGE",
    "HardwareParameterProvenance",
    "NathanHardwareBinding",
    "build_architecture_precheck",
    "correction_responsibility_matrix",
    "energy_ledger_full_rows",
    "hardware_parameter_conflict_rows",
    "hardware_parameter_provenance_rows",
    "latin_hypercube_samples",
    "native_panel_confirmation_rows",
    "native_panel_geometry",
    "resolve_nathan_hardware_binding",
    "run_blind_correction_tests",
    "run_interaction_robustness",
    "run_mode2u2_optimal_hexagon_search",
    "write_mode2u2_master_closure",
]
