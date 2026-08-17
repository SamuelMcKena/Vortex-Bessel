"""PHASE 2A fixed-bench, SLM, energy, fluence, and error-plane contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from vbb_study.design import default_config
from vbb_study.digital_twin.energy_accounting import scale_intensity_to_fluence_j_cm2
from vbb_study.equations.fields import make_xy_grid
from vbb_study.slm_model import apply_slm, field_power
from vbb_study.vector_arm_config import SLMPanelConfig


PHASE2A_HARDWARE_ID = "PHAROS_DUAL_PLUTO_FIXED_BENCH_V1"
PHASE2A_MAPPING_MODE = "fixed_physical_optics"
PHASE2A_CANONICAL_SLM_MODEL = "throughput_only"
ALLOWED_HARDWARE_PROVENANCE = frozenset(
    {"measured", "manufacturer", "derived", "assumed", "placeholder", "calibration_required"}
)
ALLOWED_EFFICIENCY_SOURCES = frozenset({"simulated", "measured", "manufacturer", "assumed"})


def _hardware_parameter(
    name: str,
    value: Any,
    units: str,
    provenance: str,
    evidence: str,
    *,
    status: str = "active",
    notes: str = "",
) -> dict[str, Any]:
    if provenance not in ALLOWED_HARDWARE_PROVENANCE:
        raise ValueError(f"unsupported hardware provenance {provenance!r}")
    return {
        "parameter": name,
        "value": value,
        "units": units,
        "provenance": provenance,
        "evidence": evidence,
        "status": status,
        "notes": notes,
    }


def canonical_hardware_manifest() -> dict[str, Any]:
    """Return the one authoritative PHASE 2A fixed-bench binding.

    Values are inherited only where the repository already contains them.
    Missing measurements remain ``None`` and carry ``calibration_required``.
    """

    cfg = default_config("fast")
    laser = cfg.laser
    slm = cfg.slm
    objective = cfg.objective
    relay = cfg.relay
    relay_demag = float(objective.f_eff_m / relay.effective_relay_f_m)
    iris_radius_m = float(laser.wavelength_m * 0.300 * slm.first_order_filter_radius_lpmm * 1.0e3)
    parameters = [
        _hardware_parameter(
            "wavelength_m", 1029.0e-9, "m", "measured",
            "docs/78_nathan_mode2u3_final_hardware_closure.md question 6",
            notes="Actual PHAROS branch value; 1030 nm is retained only for Nathan source parity.",
        ),
        _hardware_parameter(
            "input_pulse_energy_J", laser.input_pulse_energy_J, "J", "assumed",
            "vbb_study.config.LaserConfig.input_pulse_energy_J",
            status="calibration_required",
            notes="Simulation operating value, not a shot-by-shot energy measurement.",
        ),
        _hardware_parameter(
            "pulse_duration_s", laser.pulse_duration_s, "s", "assumed",
            "vbb_study.config.LaserConfig.pulse_duration_s",
            status="calibration_required",
        ),
        _hardware_parameter(
            "beam_radius_on_slm_m", 2.0e-3, "m", "assumed",
            "vbb_study.config.LaserConfig.beam_radius_on_slm_m",
            status="calibration_required",
            notes="1/e field-amplitude radius; source-scale model value pending bench measurement.",
        ),
        _hardware_parameter("slm_count", 2, "panels", "manufacturer", "externally supplied lab identity"),
        _hardware_parameter("slm_make", "HOLOEYE", "", "manufacturer", "docs/78 question 1"),
        _hardware_parameter(
            "slm_model", "PLUTO-2.1 NIR-149", "", "manufacturer", "externally supplied lab identity",
            status="calibration_required",
            notes="Exact variant must still be confirmed from each physical panel label.",
        ),
        _hardware_parameter("slm_resolution_x_px", 1920, "px", "manufacturer", "docs/78 question 2"),
        _hardware_parameter("slm_resolution_y_px", 1080, "px", "manufacturer", "docs/78 question 2"),
        _hardware_parameter("slm_pixel_pitch_m", 8.0e-6, "m", "manufacturer", "docs/78 question 2"),
        _hardware_parameter("slm_active_width_m", 15.36e-3, "m", "derived", "1920 px * 8 um"),
        _hardware_parameter("slm_active_height_m", 8.64e-3, "m", "derived", "1080 px * 8 um"),
        _hardware_parameter("slm_phase_bits", 8, "bits", "manufacturer", "docs/78 question 2"),
        _hardware_parameter("slm_fill_factor", 0.93, "fraction", "manufacturer", "docs/78 question 2"),
        _hardware_parameter(
            "slm_phase_stroke_rad", None, "rad", "calibration_required", "docs/75 phase calibration",
            status="calibration_required",
        ),
        _hardware_parameter(
            "slm_phase_lut", None, "", "calibration_required", "docs/75 phase calibration",
            status="calibration_required",
        ),
        _hardware_parameter("carrier_frequency_cpm", 6250.0, "cycles/m", "derived", "20 px * 8 um period"),
        _hardware_parameter("fourf_focal_length_m", 0.300, "m", "assumed", "F300 nominal bench description", status="calibration_required"),
        _hardware_parameter("fourier_iris_radius_m", iris_radius_m, "m", "derived", "lambda * f * 2.5 lp/mm", status="calibration_required"),
        _hardware_parameter("fourier_iris_radius_cpm", 2500.0, "cycles/m", "assumed", "SLMConfig.first_order_filter_radius_lpmm"),
        _hardware_parameter("objective_NA", objective.NA, "", "assumed", "vbb_study.config.ObjectiveConfig.NA", status="calibration_required"),
        _hardware_parameter("objective_focal_length_m", objective.f_eff_m, "m", "assumed", "vbb_study.config.ObjectiveConfig.f_eff_m", status="calibration_required"),
        _hardware_parameter("objective_pupil_radius_m", objective.pupil_radius_m, "m", "derived", "f_eff * NA / immersion_n"),
        _hardware_parameter(
            "relay_effective_focal_length_m", relay.effective_relay_f_m, "m", "assumed",
            "vbb_study.config.RelayConfig.effective_relay_f_m", status="calibration_required",
        ),
        _hardware_parameter(
            "relay_magnification_to_sample", relay_demag, "ratio", "derived",
            "objective_f_eff / relay_effective_f", status="calibration_required",
            notes="Fixed for the canonical simulation but not bench-calibrated; sample-plane claims remain blocked.",
        ),
        _hardware_parameter("axicon_base_angle_deg", 2.0, "deg", "assumed", "NathanSourceParityConfig", status="calibration_required"),
        _hardware_parameter("axicon_refractive_index", 1.458, "", "assumed", "Nathan fused-silica source model"),
        _hardware_parameter("axicon_external_medium_index", 1.0, "", "assumed", "air-side source model"),
        _hardware_parameter(
            "axicon_clear_aperture_radius_m", None, "m", "calibration_required", "physical axicon inspection",
            status="calibration_required",
        ),
        _hardware_parameter("slm_reflectivity", 0.75, "fraction", "assumed", "vbb_study.config.EnergyBudget.slm_reflectivity", status="calibration_required"),
        _hardware_parameter("relay_transmission", 0.90, "fraction", "assumed", "vbb_study.config.EnergyBudget.relay_transmission", status="calibration_required"),
        _hardware_parameter("objective_transmission", 0.90, "fraction", "assumed", "vbb_study.config.EnergyBudget.focusing_transmission", status="calibration_required"),
        _hardware_parameter("axicon_transmission", 1.0, "fraction", "placeholder", "no measured axicon transmission", status="calibration_required"),
        _hardware_parameter("sample_surface_transmission", 0.96, "fraction", "assumed", "vbb_study.config.EnergyBudget.sample_surface_transmission", status="calibration_required"),
        _hardware_parameter(
            "camera_pixel_scale_m", None, "m/px", "calibration_required", "docs/77 camera calibration",
            status="calibration_required",
        ),
    ]
    unresolved = [row["parameter"] for row in parameters if row["status"] == "calibration_required"]
    return {
        "phase": "PHASE 2A",
        "hardware_id": PHASE2A_HARDWARE_ID,
        "mapping_mode": PHASE2A_MAPPING_MODE,
        "slm_fill_factor_model": PHASE2A_CANONICAL_SLM_MODEL,
        "slm_model_reason": (
            "The 10 mm canonical validation grid does not resolve 8 um pixel borders. "
            "The manufacturer fill factor is therefore an unresolved power throughput, "
            "while first-order selection is calculated separately from the propagated field."
        ),
        "parameter_provenance_labels": sorted(ALLOWED_HARDWARE_PROVENANCE),
        "parameters": parameters,
        "calibration_required_parameters": unresolved,
        "nominal_fixed_parameter_simulation_ready": True,
        "fixed_bench_prediction_ready": False,
        "fixed_bench_prediction_blocker": (
            "The optical route is fixed numerically, but critical bench geometry, SLM calibration, "
            "axicon geometry, camera scale and throughput remain unresolved. This is a nominal "
            "fixed-parameter model, not a calibrated bench prediction."
        ),
        "absolute_sample_plane_claim_ready": False,
        "absolute_energy_claim_ready": False,
    }


def hardware_value(manifest: Mapping[str, Any], name: str) -> Any:
    for row in manifest["parameters"]:
        if row["parameter"] == name:
            return row["value"]
    raise KeyError(name)


@dataclass(frozen=True)
class EnergyFactor:
    stage: str
    factor_id: str
    efficiency: float
    source: str
    loss_kind: str
    notes: str = ""
    efficiency_components: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.efficiency) <= 1.0:
            raise ValueError(f"{self.factor_id} efficiency must lie in [0, 1]")
        if self.source not in ALLOWED_EFFICIENCY_SOURCES:
            raise ValueError(f"unsupported efficiency source {self.source!r}")
        if self.loss_kind not in {"physical", "numerical"}:
            raise ValueError("loss_kind must be physical or numerical")


def compute_unified_power_ledger(
    case_id: str,
    route_variant: str,
    input_energy_J: float,
    factors: Sequence[EnergyFactor],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply each energy factor exactly once and return closure evidence."""

    energy = float(input_energy_J)
    cumulative = 1.0
    rows: list[dict[str, Any]] = []
    for index, factor in enumerate(factors):
        before = energy
        efficiency = float(factor.efficiency)
        energy = before * efficiency
        cumulative *= efficiency
        rows.append(
            {
                "case_id": case_id,
                "route_variant": route_variant,
                "row_index": index,
                "stage": factor.stage,
                "factor_id": factor.factor_id,
                "input_power_fraction": before / max(float(input_energy_J), np.finfo(float).tiny),
                "stage_efficiency": efficiency,
                "cumulative_efficiency": cumulative,
                "pulse_energy_J": energy,
                "source_of_efficiency": factor.source,
                "numerical_or_physical_loss": factor.loss_kind,
                "notes": factor.notes,
                "efficiency_components": dict(factor.efficiency_components or {}),
            }
        )
    product = float(np.prod([factor.efficiency for factor in factors], dtype=float))
    expected = float(input_energy_J) * product
    residual = abs(energy - expected) / max(abs(float(input_energy_J)), np.finfo(float).tiny)
    return rows, {
        "case_id": case_id,
        "route_variant": route_variant,
        "input_energy_J": float(input_energy_J),
        "final_energy_J": energy,
        "product_stage_efficiencies": product,
        "expected_final_energy_J": expected,
        "closure_relative_residual": residual,
        "closure_pass": bool(residual <= 1.0e-10),
    }


def canonical_case_factors(
    manifest: Mapping[str, Any],
    *,
    route_kind: str,
    physical_hardware: bool,
    input_aperture_fraction: float,
    first_order_fraction: float,
    objective_pupil_fraction: float,
) -> list[EnergyFactor]:
    """Build one route-aware factor sequence without first-order double counting."""

    ff = float(hardware_value(manifest, "slm_fill_factor"))
    reflectivity = float(hardware_value(manifest, "slm_reflectivity"))
    panel_transfer = ff * reflectivity if physical_hardware else 1.0
    factors = [
        EnergyFactor("laser", "laser_output", 1.0, "assumed", "physical", "Configured pulse energy before optics."),
        EnergyFactor(
            "input aperture", "input_aperture_capture", float(np.clip(input_aperture_fraction, 0.0, 1.0)),
            "simulated", "physical", "Calculated from the complex input field and panel aperture.",
        ),
    ]
    components = {
        "fill_factor": {"value": ff, "source": "manufacturer", "model": PHASE2A_CANONICAL_SLM_MODEL},
        "reflectivity": {"value": reflectivity, "source": "assumed"},
    }
    if route_kind == "parallel_vector":
        first_checkpoint = 0.5 * (1.0 + panel_transfer)
        second_checkpoint = panel_transfer / max(first_checkpoint, np.finfo(float).tiny)
        factors.extend(
            [
                EnergyFactor(
                    "SLM1", "slm1_h_arm_checkpoint", first_checkpoint, "assumed", "physical",
                    "H arm traverses SLM1 while the V arm is carried as the pending branch.", components,
                ),
                EnergyFactor(
                    "SLM2", "slm2_v_arm_checkpoint", second_checkpoint, "assumed", "physical",
                    "V arm traverses SLM2; the two checkpoint factors multiply to one panel transfer.", components,
                ),
            ]
        )
    else:
        factors.extend(
            [
                EnergyFactor("SLM1", "slm1_panel_transfer", panel_transfer, "assumed", "physical", "Sequential scalar panel.", components),
                EnergyFactor("SLM2", "slm2_panel_transfer", panel_transfer, "assumed", "physical", "Sequential scalar panel.", components),
            ]
        )
    factors.append(
        EnergyFactor(
            "first-order filter", "simulated_selected_first_order", float(np.clip(first_order_fraction, 0.0, 1.0)),
            "simulated", "physical",
            "Calculated selected fraction; no configured first-order efficiency is multiplied separately.",
        )
    )
    relay = float(hardware_value(manifest, "relay_transmission")) if physical_hardware else 1.0
    objective_t = float(hardware_value(manifest, "objective_transmission")) if physical_hardware else 1.0
    axicon_t = float(hardware_value(manifest, "axicon_transmission")) if physical_hardware else 1.0
    sample_t = float(hardware_value(manifest, "sample_surface_transmission")) if physical_hardware else 1.0
    factors.extend(
        [
            EnergyFactor("relay", "relay_transmission", relay, "assumed", "physical"),
            EnergyFactor(
                "objective pupil", "simulated_objective_pupil_capture",
                float(np.clip(objective_pupil_fraction, 0.0, 1.0)), "simulated", "physical",
            ),
            EnergyFactor("objective pupil", "objective_transmission", objective_t, "assumed", "physical"),
            EnergyFactor("surface", "axicon_transmission", axicon_t, "assumed", "physical"),
            EnergyFactor("sample", "sample_surface_transmission", sample_t, "assumed", "physical"),
        ]
    )
    return factors


def fluence_from_ledger_plane(
    intensity: np.ndarray,
    dx_m: float,
    final_energy_J: float,
) -> dict[str, float]:
    """Scale one model-plane intensity using the final ledger energy."""

    dx_um = float(dx_m) / 1.0e-6
    energy_uJ = float(final_energy_J) / 1.0e-6
    fluence = scale_intensity_to_fluence_j_cm2(intensity, dx_um, dx_um, energy_uJ)
    integrated_J = float(np.sum(fluence) * dx_um * dx_um * 1.0e-8)
    return {
        "model_plane_peak_fluence_J_cm2": float(np.max(fluence)),
        "model_plane_integrated_energy_J": integrated_J,
        "fluence_energy_residual_fraction": abs(integrated_J - final_energy_J)
        / max(abs(final_energy_J), np.finfo(float).tiny),
    }


def slm_model_comparison_rows(fill_factor: float = 0.93) -> list[dict[str, Any]]:
    """Return controlled numerical evidence for all explicit SLM models."""

    pitch = 8.0e-6
    grid = make_xy_grid(512, pitch / 64.0)
    panel = SLMPanelConfig(
        n_x=128,
        n_y=128,
        pitch_m=pitch,
        phase_levels=256,
        fill_factor=float(fill_factor),
        carrier_lp_per_mm=31.25,
    )
    field = np.ones((512, 512), dtype=np.complex128)
    phase = np.zeros_like(field, dtype=float)
    p_in = field_power(field, grid)
    rows: list[dict[str, Any]] = []
    equations = {
        "throughput_only": "Eout=sqrt(FF)*exp(i*phi)*Ein",
        "resolved_pixel_aperture": "Eout=M*exp(i*phi)*Ein",
        "coherent_unmodulated_deadspace": "Eout=[M*exp(i*phi)+(1-M)]*Ein",
    }
    for model in equations:
        result = apply_slm(
            field,
            phase,
            grid,
            panel,
            quantise_phase=False,
            apply_carrier=True,
            fill_factor_model=model,
        )
        spec_mod = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(result.modulated)))) ** 2
        spec_unmod = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(result.unmodulated)))) ** 2
        cy = spec_mod.shape[0] // 2
        cx = spec_mod.shape[1] // 2
        dc_unmod = float(spec_unmod[cy, cx] / max(float(np.sum(spec_unmod)), np.finfo(float).tiny))
        rows.append(
            {
                "slm_model": model,
                "equation": equations[model],
                "input_power": p_in,
                "output_power": result.ledger.total_power,
                "output_power_fraction": result.ledger.total_power / p_in,
                "modulated_power_fraction": result.ledger.modulated_power / p_in,
                "unmodulated_power_fraction": result.ledger.unmodulated_power / p_in,
                "zero_order_fraction_within_unmodulated_component": dc_unmod,
                "ledger_relative_error": result.ledger.relative_error,
                "fill_factor_sampling": result.metadata["fill_factor_sampling"],
                "expected_scope": (
                    "unresolved pixel borders; total throughput only"
                    if model == "throughput_only"
                    else "native or supersampled pixel grid"
                    if model == "resolved_pixel_aperture"
                    else "coherent zero-order/dead-space study"
                ),
                "fill_factor_loss_counted_in_external_energy_ledger": False,
            }
        )
    return rows


def error_injection_registry_rows() -> list[dict[str, Any]]:
    """Return the audited physical-plane registry for implemented perturbations."""

    columns = (
        "error_id", "module", "injection_plane", "mathematical_operator",
        "acts_on_complex_field", "acts_before_or_after_propagation", "physical_or_diagnostic",
        "supported_routes", "validation_test", "implementation_status", "notes",
    )
    raw = [
        ("input_beam_decentre", "phase2a_canonical/component_plane_pipeline", "before SLM1", "translate input amplitude E(x-dx,y-dy)", True, "before", "physical", "scalar,vector", "test_phase2a_error_registry_and_plane_operators", "active", "Never represented by a post-processed intensity shift."),
        ("input_tilt", "component_plane_pipeline", "before SLM1", "multiply by exp(i(kx*x+ky*y))", True, "before", "physical", "scalar,vector", "test_phase2a_error_registry_and_plane_operators", "active", "Input-plane phase ramp."),
        ("hologram_offset", "phase2a_canonical/nathan_mode2s", "corresponding SLM phase plane", "translate phase-mask coordinates", True, "before", "physical", "scalar,vector", "test_phase2a_error_registry_and_plane_operators", "active", "Separate from beam decentre."),
        ("slm_phase_error", "phase2a_canonical/component_plane_pipeline", "SLM phase plane", "add deterministic phase error delta_phi(x,y)", True, "before", "physical", "scalar,vector", "test_phase2a_error_registry_and_plane_operators", "active", "Acts on phase, not intensity."),
        ("slm_quantisation", "slm_model", "SLM phase plane", "nearest allowed phase level", True, "before", "physical", "scalar,vector", "test_phase2a_slm_models_are_explicit_and_power_consistent", "active", "Phase stroke/LUT remains calibration-required."),
        ("iris_offset_radius", "phase2a_canonical/nathan_mode2s", "4F Fourier plane", "translate/resize hard spectral aperture", True, "before", "physical", "scalar,vector", "test_phase2a_first_order_is_simulated_and_not_double_counted", "active", "Selected fraction is measured from the simulated spectrum."),
        ("axicon_decentre", "phase2a_canonical/nathan_mode2s", "axicon plane", "translate conical phase origin", True, "before", "physical", "scalar,vector", "test_phase2a_error_registry_and_plane_operators", "active", "Camera-frame recentering is not used as the physical operator."),
        ("axicon_tilt", "nathan_mode2s/component_plane_pipeline", "axicon plane", "multiply by linear phase ramp after conical phase", True, "before", "physical", "scalar,vector", "test_phase2a_error_registry_and_plane_operators", "active", "Tilt and decentre remain distinct."),
        ("objective_pupil_clipping", "phase2a_canonical/component_plane_pipeline", "objective pupil", "multiply by circular pupil mask", True, "before", "physical", "scalar,vector", "test_phase2a_error_registry_and_plane_operators", "active", "Captured fraction feeds the energy ledger."),
        ("low_order_aberration", "phase2a_canonical/component_plane_pipeline", "physical pupil/aberrating plane", "multiply by exp(i*2*pi*sum(c_j Z_j))", True, "before", "physical", "scalar,vector", "test_phase2a_error_registry_and_plane_operators", "active", "Plane must be declared by the route."),
        ("sample_interface_tilt", "component_plane_pipeline/lab_perturbations", "sample surface", "tilted interface coordinate/phase operator", True, "before", "physical", "scalar through-sample", "test_phase2a_error_registry_and_plane_operators", "calibration_limited", "Not applied to the source-scale H1 branch."),
        ("camera_noise", "lab_perturbations", "camera plane", "seeded detector-noise operator on measured intensity", False, "after", "diagnostic", "all observation routes", "test_phase2a_error_registry_and_plane_operators", "active", "Cannot alter upstream optical metrics."),
        ("post_processing_display_shift", "lab_perturbations", "display only", "translate rendered intensity array", False, "after", "diagnostic", "all display routes", "test_phase2a_error_registry_and_plane_operators", "active", "Forbidden as a surrogate for beam tilt or decentre."),
    ]
    rows = [dict(zip(columns, row, strict=True)) for row in raw]
    for row in rows:
        row["physical_operator"] = row["mathematical_operator"]
        row["affected_field"] = "complex_field" if row["acts_on_complex_field"] else "measured_or_display_intensity"
        row["diagnostic_only"] = row["physical_or_diagnostic"] == "diagnostic"
    return rows


def validate_error_registry(rows: Iterable[Mapping[str, Any]]) -> None:
    required = {
        "error_id", "module", "injection_plane", "mathematical_operator", "acts_on_complex_field",
        "acts_before_or_after_propagation", "physical_or_diagnostic", "supported_routes", "validation_test",
    }
    seen: set[str] = set()
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"error registry row is missing {sorted(missing)}")
        error_id = str(row["error_id"])
        if error_id in seen:
            raise ValueError(f"duplicate error_id {error_id}")
        seen.add(error_id)
        if row["physical_or_diagnostic"] == "diagnostic" and row["acts_before_or_after_propagation"] != "after":
            raise ValueError(f"diagnostic operation {error_id} cannot claim upstream injection")


__all__ = [
    "ALLOWED_EFFICIENCY_SOURCES",
    "ALLOWED_HARDWARE_PROVENANCE",
    "EnergyFactor",
    "PHASE2A_CANONICAL_SLM_MODEL",
    "PHASE2A_HARDWARE_ID",
    "PHASE2A_MAPPING_MODE",
    "canonical_case_factors",
    "canonical_hardware_manifest",
    "compute_unified_power_ledger",
    "error_injection_registry_rows",
    "fluence_from_ledger_plane",
    "hardware_value",
    "slm_model_comparison_rows",
    "validate_error_registry",
]
