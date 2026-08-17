"""Stage 9B.0/9B.0.1 nominal F300 scalar 4F virtual bench.

This module is an opt-in nominal forward model for the declared F300 relay:

    field arriving at SLM2 -> SLM2 carrier plane -> 300 mm
    -> Lens 1 f=300 mm -> 300 mm
    -> Fourier/pinhole plane -> 300 mm -> Lens 2 f=300 mm
    -> 300 mm -> nominal relay-output plane

It is not bench calibrated. It does not change the existing active CSLM route,
does not mark physical 4F readiness ready, and does not implement a camera
model, inverse correction, AI, nonlinear propagation, or material response.

Stage 9B.0.1 also makes the upstream ownership explicit: SLM1 phase must be
applied at SLM1 and propagated to SLM2 by a declared upstream route before this
nominal F300 model is called. A direct SLM1 phase argument at the SLM2 plane is
rejected.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.equations.fields import make_xy_grid
from vbb_study.equations.holography import (
    TWOPI,
    phase_to_gray,
    quantize_phase_rad,
    spp_phase_rad,
    wrap_phase_rad,
)
from vbb_study.equations.propagation import angular_spectrum_propagate_bl, discrete_power
from vbb_study.digital_twin.cslm_route import CSLMRouteConfig, run_cslm_baseline_route


STAGE = "9B.0.1"
PROFILE_PATH = Path("configs/hardware/cslm_f300_nominal_4f_profile.json")
MODEL_LABEL = "Nominal F300 4F scenario - not bench calibrated"
CLAIM_BOUNDARY_LABELS = (
    "nominal_4f_forward_model",
    "not_bench_calibrated",
    "not_physical_4f_readiness_ready",
    "not_camera_modelled",
    "not_material_modelled",
)
FINAL_EXPORT_ALLOWED = False
EPS = 1e-30

UPSTREAM_SOURCE_MODES = (
    "existing_cslm_component_route",
    "explicit_field_arriving_at_slm2",
    "synthetic_gaussian_unit_test_only",
)

CARRIER_REALISM = "ideal_continuous_phase_ramp"
IDEAL_CARRIER_BOUNDARY = (
    "The Fourier-plane stop selects a region of the ideal continuous-ramp spectrum. "
    "It is not a simulation of physical pixelated-SLM zero-order leakage, discrete "
    "diffraction-order power fractions, or measured selected-order purity."
)
CARRIER_BOUNDARY_FLAGS = {
    "carrier_realism": CARRIER_REALISM,
    "ideal_blazed_carrier_shift_surrogate": True,
    "pixelated_slm_diffraction_orders_modelled": False,
    "zero_order_modelled": False,
    "physical_order_efficiency_modelled": False,
    "selected_order_purity_predicted": False,
    "carrier_boundary": IDEAL_CARRIER_BOUNDARY,
}

STOP_SAMPLING_STATUSES = (
    "underresolved",
    "exploratory_only",
    "ranking_eligible",
    "convergence_verified",
)
CONVERGENCE_STATUSES = (
    "not_checked",
    "failed",
    "passed_for_nominal_scenario",
)

NOMINAL_COMPONENT_SEQUENCE = (
    "SLM2_phase_plane",
    "SLM2_to_lens1_propagation",
    "lens1_thin_phase_and_pupil",
    "lens1_to_fourier_plane_propagation",
    "fourier_plane_field_pre_stop",
    "fourier_stop_pinhole",
    "fourier_plane_field_post_stop",
    "fourier_plane_to_lens2_propagation",
    "lens2_thin_phase_and_pupil",
    "lens2_to_nominal_relay_output_propagation",
    "nominal_relay_output_plane",
)

UPSTREAM_TO_F300_COMPONENT_CHAIN = (
    "source_field",
    "input_conditioning_boundary",
    "SLM1_phase_plane",
    "SLM1_to_SLM2_segment",
    "field_arriving_at_SLM2",
    *NOMINAL_COMPONENT_SEQUENCE,
)


@dataclass(frozen=True)
class NominalF300Config:
    """Editable nominal F300 4F numerical scenario."""

    simulation_plane_width_m: float = 0.008
    simulation_grid_size: int = 128
    wavelength_m: float = 1.03e-6
    n_medium: float = 1.0
    input_beam_radius_m: float = 0.0008
    input_beam_decentre_x_m: float = 0.0
    input_beam_decentre_y_m: float = 0.0
    input_beam_ellipticity: float = 1.0
    input_beam_rotation_deg: float = 0.0
    lens_clear_radius_m: float = 0.0032
    pinhole_radius_m: float = 0.00018
    pinhole_offset_x_m: float = 0.00031
    pinhole_offset_y_m: float = 0.0
    command_domain_carrier_cycles_x: float = 8.0
    command_domain_carrier_cycles_y: float = 0.0
    numerical_model_carrier_cycles_x: float = 8.0
    numerical_model_carrier_cycles_y: float = 0.0
    slm2_to_lens1_m: float = 0.300
    lens1_focal_length_m: float = 0.300
    lens1_to_fourier_plane_m: float = 0.300
    fourier_plane_to_lens2_m: float = 0.300
    lens2_focal_length_m: float = 0.300
    lens2_to_nominal_relay_output_m: float = 0.300
    bandlimit: bool = True
    slm_phase_quantisation_levels: int = 256
    minimum_stop_diameter_pixels_for_exploration: float = 6.0
    minimum_stop_diameter_pixels_for_ranking: float = 10.0
    relay_output_to_axicon_mode: str = "unknown_not_simulated"
    relay_output_to_axicon_distance_m: float | None = None

    @classmethod
    def fast(cls, **overrides: Any) -> "NominalF300Config":
        return cls.exploratory(**overrides)

    @classmethod
    def exploratory(cls, **overrides: Any) -> "NominalF300Config":
        base = {
            "simulation_grid_size": 128,
            "simulation_plane_width_m": 0.008,
            "input_beam_radius_m": 0.0008,
            "lens_clear_radius_m": 0.0032,
        }
        base.update(overrides)
        return cls(**base)

    @classmethod
    def standard(cls, **overrides: Any) -> "NominalF300Config":
        base = {
            "simulation_grid_size": 256,
            "simulation_plane_width_m": 0.008,
            "input_beam_radius_m": 0.0008,
            "lens_clear_radius_m": 0.0032,
        }
        base.update(overrides)
        return cls(**base)

    @property
    def dx_m(self) -> float:
        return float(self.simulation_plane_width_m) / int(self.simulation_grid_size)


@dataclass(frozen=True)
class UpstreamSLM2FieldBridge:
    """Declared upstream field handoff into the nominal F300 model."""

    upstream_source_mode: str
    field_arriving_at_slm2: np.ndarray
    grid: Mapping[str, Any]
    slm1_phase_rad: np.ndarray
    source_field: np.ndarray
    post_slm1_field: np.ndarray
    component_chain: tuple[str, ...]
    diagnostics: Mapping[str, Any]
    slm1_phase_applied_at_slm1: bool
    slm1_to_slm2_propagation_included: bool
    slm2_carrier_applied_at_slm2: bool = False


@dataclass(frozen=True)
class NominalF300Run:
    config: NominalF300Config
    grid: Mapping[str, Any]
    slm1_phase_rad: np.ndarray
    slm2_phase_rad: np.ndarray
    slm2_input_field: np.ndarray
    post_lens1_field: np.ndarray
    fourier_plane_field_pre_stop: np.ndarray
    fourier_stop_transmission: np.ndarray
    fourier_plane_field_post_stop: np.ndarray
    post_lens2_field: np.ndarray
    nominal_relay_output_field: np.ndarray
    component_energy_ledger: tuple[Mapping[str, Any], ...]
    component_manifest: tuple[Mapping[str, Any], ...]
    warnings: tuple[str, ...]
    diagnostics: Mapping[str, Any]
    field_post_slm2: np.ndarray
    upstream_source_mode: str
    upstream_component_chain: tuple[str, ...]
    upstream_diagnostics: Mapping[str, Any]
    slm1_phase_applied_at_slm1: bool
    slm1_to_slm2_propagation_included: bool
    slm2_carrier_applied_at_slm2: bool
    stop_sampling_report: Mapping[str, Any]
    convergence_status: str = "not_checked"
    carrier_coordinate_status: str = "nominal_model_not_bench_calibrated"
    carrier_realism: str = CARRIER_REALISM
    ideal_blazed_carrier_shift_surrogate: bool = True
    pixelated_slm_diffraction_orders_modelled: bool = False
    zero_order_modelled: bool = False
    physical_order_efficiency_modelled: bool = False
    selected_order_purity_predicted: bool = False
    nominal_4f_forward_model: bool = True
    bench_calibrated: bool = False
    physical_4f_readiness: str = "blocked"
    final_export_allowed: bool = FINAL_EXPORT_ALLOWED

    @property
    def outputs(self) -> dict[str, Any]:
        return {
            "slm2_input_field": self.slm2_input_field,
            "post_lens1_field": self.post_lens1_field,
            "fourier_plane_field_pre_stop": self.fourier_plane_field_pre_stop,
            "fourier_stop_transmission": self.fourier_stop_transmission,
            "fourier_plane_field_post_stop": self.fourier_plane_field_post_stop,
            "field_post_slm2": self.field_post_slm2,
            "post_lens2_field": self.post_lens2_field,
            "nominal_relay_output_field": self.nominal_relay_output_field,
            "component_energy_ledger": list(self.component_energy_ledger),
            "component_manifest": list(self.component_manifest),
        }


def load_nominal_f300_profile(path: str | Path = PROFILE_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def config_from_profile(profile: Mapping[str, Any] | None = None) -> NominalF300Config:
    profile = profile or load_nominal_f300_profile()
    geom = profile["known_nominal_geometry"]
    conv = profile["model_convention"]

    def _v(section: Mapping[str, Any], key: str) -> Any:
        return section[key]["value"]

    dec = _v(conv, "input_beam_decentre_m")
    return NominalF300Config(
        simulation_plane_width_m=float(_v(conv, "simulation_plane_width_m")),
        simulation_grid_size=int(_v(conv, "simulation_grid_size")),
        wavelength_m=float(_v(conv, "wavelength_m")),
        input_beam_radius_m=float(_v(conv, "input_beam_radius_m")),
        input_beam_decentre_x_m=float(dec[0]),
        input_beam_decentre_y_m=float(dec[1]),
        input_beam_ellipticity=float(_v(conv, "input_beam_ellipticity")),
        lens_clear_radius_m=float(_v(conv, "lens_clear_radius_m")),
        pinhole_radius_m=float(_v(conv, "pinhole_radius_m")),
        pinhole_offset_x_m=float(_v(conv, "pinhole_offset_x_m")),
        pinhole_offset_y_m=float(_v(conv, "pinhole_offset_y_m")),
        slm2_to_lens1_m=float(_v(geom, "slm2_to_lens1_m")),
        lens1_focal_length_m=float(_v(geom, "lens1_focal_length_m")),
        lens1_to_fourier_plane_m=float(_v(geom, "lens1_to_fourier_plane_m")),
        fourier_plane_to_lens2_m=float(_v(geom, "fourier_plane_to_lens2_m")),
        lens2_focal_length_m=float(_v(geom, "lens2_focal_length_m")),
        lens2_to_nominal_relay_output_m=float(_v(geom, "lens2_to_nominal_relay_output_m")),
        minimum_stop_diameter_pixels_for_exploration=float(
            profile.get("stop_sampling_policy", {}).get("minimum_stop_diameter_pixels_for_exploration", 6.0)
        ),
        minimum_stop_diameter_pixels_for_ranking=float(
            profile.get("stop_sampling_policy", {}).get("minimum_stop_diameter_pixels_for_ranking", 10.0)
        ),
    )


def replace_config(config: NominalF300Config, **updates: Any) -> NominalF300Config:
    return replace(config, **updates)


def _grid(config: NominalF300Config) -> dict[str, Any]:
    return make_xy_grid(int(config.simulation_grid_size), config.dx_m)


def _power(field: np.ndarray, config: NominalF300Config) -> float:
    return discrete_power(field, config.dx_m)


def _gaussian_input(grid: Mapping[str, Any], config: NominalF300Config) -> np.ndarray:
    x0 = float(config.input_beam_decentre_x_m)
    y0 = float(config.input_beam_decentre_y_m)
    wx = max(float(config.input_beam_radius_m), EPS)
    wy = wx * max(float(config.input_beam_ellipticity), 0.05)
    rot = np.deg2rad(float(config.input_beam_rotation_deg))
    X = np.asarray(grid["X"], float) - x0
    Y = np.asarray(grid["Y"], float) - y0
    Xr = X * np.cos(rot) + Y * np.sin(rot)
    Yr = -X * np.sin(rot) + Y * np.cos(rot)
    return np.exp(-((Xr / wx) ** 2 + (Yr / wy) ** 2)).astype(complex)


def _validate_mode(mode: str) -> str:
    if mode not in UPSTREAM_SOURCE_MODES:
        raise ValueError(f"Unsupported upstream_source_mode {mode!r}; expected one of {UPSTREAM_SOURCE_MODES}.")
    return mode


def _zero_phase(config: NominalF300Config) -> np.ndarray:
    return np.zeros((int(config.simulation_grid_size), int(config.simulation_grid_size)), dtype=float)


def build_synthetic_gaussian_slm2_bridge(config: NominalF300Config | None = None) -> UpstreamSLM2FieldBridge:
    """Build a unit-test-only SLM2 input field with no SLM1 propagation claim."""

    cfg = config or NominalF300Config.exploratory()
    grid = _grid(cfg)
    field = _gaussian_input(grid, cfg)
    zero = _zero_phase(cfg)
    return UpstreamSLM2FieldBridge(
        upstream_source_mode="synthetic_gaussian_unit_test_only",
        field_arriving_at_slm2=field,
        grid=grid,
        slm1_phase_rad=zero,
        source_field=field,
        post_slm1_field=field,
        component_chain=("synthetic_gaussian_unit_test_only", "field_arriving_at_SLM2"),
        diagnostics={
            "upstream_source_mode": "synthetic_gaussian_unit_test_only",
            "slm1_phase_applied_at_slm1": False,
            "slm1_to_slm2_propagation_included": False,
            "not_for_candidate_package": True,
        },
        slm1_phase_applied_at_slm1=False,
        slm1_to_slm2_propagation_included=False,
    )


def build_existing_cslm_slm2_bridge(
    config: NominalF300Config | None = None,
    *,
    topological_charge: int = 0,
    slm1_phase_mode: str | None = None,
) -> UpstreamSLM2FieldBridge:
    """Use the existing component-owned CSLM route up to the SLM2 input plane."""

    cfg = config or NominalF300Config.standard()
    mode = slm1_phase_mode or ("flat" if int(topological_charge) == 0 else "vortex")
    cslm_cfg = CSLMRouteConfig(
        grid_N=int(cfg.simulation_grid_size),
        dx_um=float(cfg.dx_m) * 1e6,
        wavelength_nm=float(cfg.wavelength_m) * 1e9,
        n_medium=float(cfg.n_medium),
        input_beam_radius_um=float(cfg.input_beam_radius_m) * 1e6,
        slm1_phase_mode=mode,
        slm1_topological_charge=int(topological_charge),
        slm2_carrier_frequency_cpm=0.0,
        slm2_correction_phase_rad=0.0,
        slm_phase_quantisation_levels=int(cfg.slm_phase_quantisation_levels),
        n_z=4,
        z_max_um=1.0,
    )
    cslm_run = run_cslm_baseline_route(cslm_cfg)
    if cslm_run.slm2_input_state.field is None:
        raise ValueError("existing CSLM route did not produce a field arriving at SLM2.")
    return UpstreamSLM2FieldBridge(
        upstream_source_mode="existing_cslm_component_route",
        field_arriving_at_slm2=np.asarray(cslm_run.slm2_input_state.field, dtype=complex),
        grid=_grid(cfg),
        slm1_phase_rad=np.asarray(cslm_run.slm1_phase_rad, dtype=float),
        source_field=np.asarray(cslm_run.source_state.field, dtype=complex),
        post_slm1_field=np.asarray(cslm_run.slm1_state.field, dtype=complex),
        component_chain=(
            "source_field",
            "input_conditioning_boundary",
            "SLM1_phase_plane",
            "SLM1_to_SLM2_segment",
            "field_arriving_at_SLM2",
        ),
        diagnostics={
            "upstream_source_mode": "existing_cslm_component_route",
            "cslm_executed_route_chain": list(cslm_run.executed_route_chain),
            "slm1_phase_applied_at_slm1": True,
            "slm1_to_slm2_propagation_included": True,
            "slm1_phase_mode": mode,
            "topological_charge": int(topological_charge),
            "adapter_note": (
                "CSLM route grid is matched to the nominal 4F numerical grid; this is a declared "
                "nominal bridge, not measured SLM/4F calibration."
            ),
        },
        slm1_phase_applied_at_slm1=True,
        slm1_to_slm2_propagation_included=True,
    )


def vortex_phase(grid: Mapping[str, Any], ell: int) -> np.ndarray:
    if int(ell) == 0:
        return np.zeros_like(np.asarray(grid["X"], float))
    return wrap_phase_rad(spp_phase_rad(grid["PHI"], int(ell)))


def carrier_phase(grid: Mapping[str, Any], config: NominalF300Config) -> np.ndarray:
    width = float(config.simulation_plane_width_m)
    X = np.asarray(grid["X"], float)
    Y = np.asarray(grid["Y"], float)
    nx = float(config.numerical_model_carrier_cycles_x)
    ny = float(config.numerical_model_carrier_cycles_y)
    return wrap_phase_rad(TWOPI * (nx * X / width + ny * Y / width))


def carrier_boundary_record() -> dict[str, Any]:
    return dict(CARRIER_BOUNDARY_FLAGS)


def stop_sampling_report(config: NominalF300Config, *, convergence_status: str = "not_checked") -> dict[str, Any]:
    if convergence_status not in CONVERGENCE_STATUSES:
        raise ValueError(f"Unsupported convergence_status {convergence_status!r}.")
    radius_px = float(config.pinhole_radius_m) / max(float(config.dx_m), EPS)
    diameter_px = 2.0 * radius_px
    rounded_diameter_px = int(round(diameter_px))
    exploration_min = float(config.minimum_stop_diameter_pixels_for_exploration)
    ranking_min = float(config.minimum_stop_diameter_pixels_for_ranking)
    if rounded_diameter_px < exploration_min:
        status = "underresolved"
    elif diameter_px < ranking_min:
        status = "exploratory_only"
    else:
        status = "ranking_eligible"
    if status == "ranking_eligible" and convergence_status == "passed_for_nominal_scenario":
        status = "convergence_verified"
    return {
        "sampling_pitch_m": float(config.dx_m),
        "stop_radius_pixels": float(radius_px),
        "stop_diameter_pixels": float(diameter_px),
        "stop_diameter_pixels_rounded": int(rounded_diameter_px),
        "minimum_stop_diameter_pixels_for_exploration": exploration_min,
        "minimum_stop_diameter_pixels_for_ranking": ranking_min,
        "stop_sampling_status": status,
        "convergence_status": convergence_status,
        "ranking_allowed": bool(status in {"ranking_eligible", "convergence_verified"} and convergence_status == "passed_for_nominal_scenario"),
    }


def thin_lens_phase(grid: Mapping[str, Any], wavelength_m: float, focal_length_m: float) -> np.ndarray:
    k = TWOPI / float(wavelength_m)
    R2 = np.asarray(grid["X"], float) ** 2 + np.asarray(grid["Y"], float) ** 2
    return np.exp(-1j * k * R2 / (2.0 * float(focal_length_m)))


def circular_amplitude(grid: Mapping[str, Any], radius_m: float, x0_m: float = 0.0, y0_m: float = 0.0) -> np.ndarray:
    R = np.hypot(np.asarray(grid["X"], float) - float(x0_m), np.asarray(grid["Y"], float) - float(y0_m))
    return (R <= float(radius_m)).astype(float)


def _prop(field: np.ndarray, grid: Mapping[str, Any], config: NominalF300Config, distance_m: float) -> np.ndarray:
    return angular_spectrum_propagate_bl(
        np.asarray(field, complex),
        dict(grid),
        float(config.wavelength_m),
        float(distance_m),
        n_medium=float(config.n_medium),
        bandlimit=bool(config.bandlimit),
    )


def _field_metrics(field: np.ndarray, config: NominalF300Config) -> dict[str, Any]:
    I = np.abs(np.asarray(field, complex)) ** 2
    grid = _grid(config)
    X = np.asarray(grid["X"], float)
    Y = np.asarray(grid["Y"], float)
    total = float(np.sum(I))
    if total <= EPS:
        cx = cy = sx = sy = 0.0
    else:
        cx = float(np.sum(I * X) / total)
        cy = float(np.sum(I * Y) / total)
        sx = float(np.sqrt(max(np.sum(I * (X - cx) ** 2) / total, 0.0)))
        sy = float(np.sqrt(max(np.sum(I * (Y - cy) ** 2) / total, 0.0)))
    half = 0.5 * float(config.simulation_plane_width_m)
    margin = min(half - abs(cx), half - abs(cy))
    border = _border_power_fraction(I)
    return {
        "total_power_arb": float(total * config.dx_m * config.dx_m),
        "centroid_x_m": cx,
        "centroid_y_m": cy,
        "second_moment_width_x_m": sx,
        "second_moment_width_y_m": sy,
        "field_of_view_margin_m": float(margin),
        "border_power_fraction": float(border),
    }


def _border_power_fraction(intensity: np.ndarray, border_px: int = 2) -> float:
    I = np.asarray(intensity, float)
    total = float(np.sum(I))
    if total <= EPS:
        return 0.0
    b = max(1, int(border_px))
    edge = np.zeros_like(I, dtype=bool)
    edge[:b, :] = True
    edge[-b:, :] = True
    edge[:, :b] = True
    edge[:, -b:] = True
    return float(np.sum(I[edge]) / total)


def _resample_real_to_target(source: np.ndarray, source_x: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    arr = np.asarray(source, dtype=float)
    sx = np.asarray(source_x, dtype=float)
    tx = np.asarray(target_x, dtype=float)
    tmp = np.empty((arr.shape[0], tx.size), dtype=float)
    for iy in range(arr.shape[0]):
        tmp[iy] = np.interp(tx, sx, arr[iy], left=0.0, right=0.0)
    out = np.empty((tx.size, tx.size), dtype=float)
    for ix in range(tx.size):
        out[:, ix] = np.interp(tx, sx, tmp[:, ix], left=0.0, right=0.0)
    return out


def _normalised_intensity(field: np.ndarray) -> np.ndarray:
    intensity = np.abs(np.asarray(field, dtype=complex)) ** 2
    total = float(np.linalg.norm(intensity.ravel()))
    return intensity / max(total, EPS)


def evaluate_stop_sampling_convergence(
    exploratory_run: "NominalF300Run",
    standard_run: "NominalF300Run",
    *,
    energy_relative_tolerance: float = 0.35,
    centroid_tolerance_m: float = 2.5e-4,
    width_relative_tolerance: float = 0.35,
    intensity_correlation_minimum: float = 0.85,
) -> dict[str, Any]:
    """Compare exploratory and standard-grid stop sampling for one scenario."""

    exp_metrics = exploratory_run.diagnostics["nominal_relay_output_metrics"]
    std_metrics = standard_run.diagnostics["nominal_relay_output_metrics"]
    exp_energy = float(exploratory_run.component_energy_ledger[-1]["energy_after_arb_m2"])
    std_energy = float(standard_run.component_energy_ledger[-1]["energy_after_arb_m2"])
    energy_rel = abs(std_energy - exp_energy) / max(abs(std_energy), EPS)
    centroid_diff = float(
        np.hypot(
            float(std_metrics["centroid_x_m"]) - float(exp_metrics["centroid_x_m"]),
            float(std_metrics["centroid_y_m"]) - float(exp_metrics["centroid_y_m"]),
        )
    )
    exp_width = float(np.hypot(exp_metrics["second_moment_width_x_m"], exp_metrics["second_moment_width_y_m"]))
    std_width = float(np.hypot(std_metrics["second_moment_width_x_m"], std_metrics["second_moment_width_y_m"]))
    width_rel = abs(std_width - exp_width) / max(abs(std_width), EPS)

    exp_i = _normalised_intensity(exploratory_run.nominal_relay_output_field)
    std_i = _normalised_intensity(standard_run.nominal_relay_output_field)
    std_on_exp = _resample_real_to_target(std_i, np.asarray(standard_run.grid["x"], float), np.asarray(exploratory_run.grid["x"], float))
    denom = np.linalg.norm(exp_i.ravel()) * np.linalg.norm(std_on_exp.ravel()) + EPS
    corr = float(np.dot(exp_i.ravel(), std_on_exp.ravel()) / denom)
    exp_sampling = stop_sampling_report(exploratory_run.config)
    std_sampling = stop_sampling_report(standard_run.config)
    passed = (
        std_sampling["stop_sampling_status"] == "ranking_eligible"
        and energy_rel <= float(energy_relative_tolerance)
        and centroid_diff <= float(centroid_tolerance_m)
        and width_rel <= float(width_relative_tolerance)
        and corr >= float(intensity_correlation_minimum)
    )
    return {
        "convergence_status": "passed_for_nominal_scenario" if passed else "failed",
        "transmitted_energy_relative_difference": float(energy_rel),
        "relay_output_centroid_difference_m": float(centroid_diff),
        "second_moment_width_relative_difference": float(width_rel),
        "normalised_intensity_correlation": float(corr),
        "exploratory_stop_sampling_status": exp_sampling["stop_sampling_status"],
        "standard_stop_sampling_status": std_sampling["stop_sampling_status"],
        "exploratory_warnings": list(exploratory_run.warnings),
        "standard_warnings": list(standard_run.warnings),
        "warning_status_comparison": "same" if tuple(exploratory_run.warnings) == tuple(standard_run.warnings) else "different",
        "thresholds": {
            "energy_relative_tolerance": float(energy_relative_tolerance),
            "centroid_tolerance_m": float(centroid_tolerance_m),
            "width_relative_tolerance": float(width_relative_tolerance),
            "intensity_correlation_minimum": float(intensity_correlation_minimum),
        },
    }


def _ledger_row(component_id: str, component_type: str, before: float, after: float, note: str) -> dict[str, Any]:
    return {
        "component_id": component_id,
        "component_type": component_type,
        "energy_before_arb_m2": float(before),
        "energy_after_arb_m2": float(after),
        "transmitted_fraction": float(after / max(before, EPS)),
        "note": note,
    }


def _manifest_row(
    component_id: str,
    component_type: str,
    transform_applied: bool,
    distance_m: float | None,
    status: str,
    note: str,
) -> dict[str, Any]:
    return {
        "component_id": component_id,
        "component_type": component_type,
        "transform_applied": bool(transform_applied),
        "distance_m": None if distance_m is None else float(distance_m),
        "model_status": status,
        "downstream_components_affected": list(NOMINAL_COMPONENT_SEQUENCE[
            NOMINAL_COMPONENT_SEQUENCE.index(component_id) + 1 :
        ]) if component_id in NOMINAL_COMPONENT_SEQUENCE else [],
        "note": note,
    }


def run_nominal_f300_4f(
    config: NominalF300Config | None = None,
    *,
    field_arriving_at_slm2: np.ndarray | None = None,
    slm2_phase_rad: np.ndarray | None = None,
    field_post_slm2: np.ndarray | None = None,
    upstream_source_mode: str = "synthetic_gaussian_unit_test_only",
    upstream_bridge: UpstreamSLM2FieldBridge | None = None,
    slm1_phase_rad: np.ndarray | None = None,
    input_field: np.ndarray | None = None,
) -> NominalF300Run:
    """Run the nominal F300 scalar 4F model."""
    config = config or NominalF300Config()
    grid = _grid(config)
    if slm1_phase_rad is not None:
        raise ValueError(
            "run_nominal_f300_4f no longer accepts slm1_phase_rad at the SLM2 plane. "
            "Apply SLM1 phase at SLM1 and provide field_arriving_at_slm2 via the upstream CSLM bridge."
        )
    if input_field is not None:
        raise ValueError(
            "run_nominal_f300_4f no longer accepts input_field as a shortcut. "
            "Use field_arriving_at_slm2 or an UpstreamSLM2FieldBridge."
        )
    if field_arriving_at_slm2 is not None and field_post_slm2 is not None:
        raise ValueError("Provide either field_arriving_at_slm2 or field_post_slm2, not both.")

    upstream_source_mode = _validate_mode(upstream_source_mode)
    if upstream_bridge is not None:
        bridge = upstream_bridge
        upstream_source_mode = bridge.upstream_source_mode
        slm2_input = np.asarray(bridge.field_arriving_at_slm2, dtype=complex)
        slm1_phase = np.asarray(bridge.slm1_phase_rad, dtype=float)
        upstream_chain = tuple(bridge.component_chain)
        upstream_diag = dict(bridge.diagnostics)
        slm1_phase_applied = bool(bridge.slm1_phase_applied_at_slm1)
        slm1_to_slm2_included = bool(bridge.slm1_to_slm2_propagation_included)
    elif field_arriving_at_slm2 is not None:
        slm2_input = np.asarray(field_arriving_at_slm2, dtype=complex)
        slm1_phase = _zero_phase(config)
        upstream_chain = ("explicit_field_arriving_at_SLM2",)
        upstream_diag = {
            "upstream_source_mode": "explicit_field_arriving_at_slm2",
            "slm1_phase_applied_at_slm1": False,
            "slm1_to_slm2_propagation_included": False,
        }
        upstream_source_mode = "explicit_field_arriving_at_slm2"
        slm1_phase_applied = False
        slm1_to_slm2_included = False
    elif field_post_slm2 is not None:
        slm2_input = np.asarray(field_post_slm2, dtype=complex)
        slm1_phase = _zero_phase(config)
        upstream_chain = ("field_post_SLM2_supplied",)
        upstream_diag = {
            "upstream_source_mode": "explicit_field_arriving_at_slm2",
            "field_post_slm2_supplied": True,
            "slm1_phase_applied_at_slm1": False,
            "slm1_to_slm2_propagation_included": False,
        }
        upstream_source_mode = "explicit_field_arriving_at_slm2"
        slm1_phase_applied = False
        slm1_to_slm2_included = False
    elif upstream_source_mode == "synthetic_gaussian_unit_test_only":
        bridge = build_synthetic_gaussian_slm2_bridge(config)
        slm2_input = np.asarray(bridge.field_arriving_at_slm2, dtype=complex)
        slm1_phase = np.asarray(bridge.slm1_phase_rad, dtype=float)
        upstream_chain = tuple(bridge.component_chain)
        upstream_diag = dict(bridge.diagnostics)
        slm1_phase_applied = False
        slm1_to_slm2_included = False
    else:
        raise ValueError("field_arriving_at_slm2 or upstream_bridge is required for this upstream_source_mode.")
    if slm2_input.shape != (int(config.simulation_grid_size), int(config.simulation_grid_size)):
        raise ValueError("field_arriving_at_slm2 shape must match the configured square grid.")
    if slm1_phase.shape != slm2_input.shape:
        slm1_phase = np.zeros_like(np.real(slm2_input), dtype=float)

    if field_post_slm2 is not None:
        slm2_phase = np.zeros_like(np.real(slm2_input), dtype=float)
        field = np.asarray(field_post_slm2, dtype=complex)
        slm2_carrier_applied = False
    else:
        slm2_phase = carrier_phase(grid, config) if slm2_phase_rad is None else wrap_phase_rad(np.asarray(slm2_phase_rad, dtype=float))
        if slm2_phase.shape != slm2_input.shape:
            raise ValueError("slm2_phase_rad shape must match the configured square grid.")
        field = slm2_input * np.exp(1j * slm2_phase)
        slm2_carrier_applied = True
    lens1_field = _prop(field, grid, config, config.slm2_to_lens1_m)

    lens_pupil = circular_amplitude(grid, config.lens_clear_radius_m)
    lens1_before = _power(lens1_field, config)
    lens1_after_field = lens1_field * thin_lens_phase(grid, config.wavelength_m, config.lens1_focal_length_m) * lens_pupil
    lens1_after = _power(lens1_after_field, config)
    fourier_pre = _prop(lens1_after_field, grid, config, config.lens1_to_fourier_plane_m)

    stop = circular_amplitude(
        grid,
        config.pinhole_radius_m,
        config.pinhole_offset_x_m,
        config.pinhole_offset_y_m,
    )
    stop_before = _power(fourier_pre, config)
    fourier_post = fourier_pre * stop
    stop_after = _power(fourier_post, config)

    lens2_input = _prop(fourier_post, grid, config, config.fourier_plane_to_lens2_m)
    lens2_before = _power(lens2_input, config)
    lens2_after_field = lens2_input * thin_lens_phase(grid, config.wavelength_m, config.lens2_focal_length_m) * lens_pupil
    lens2_after = _power(lens2_after_field, config)
    relay_output = _prop(lens2_after_field, grid, config, config.lens2_to_nominal_relay_output_m)

    source_power = _power(slm2_input, config)
    after_slm2 = _power(field, config)
    after_prop_lens1 = _power(lens1_field, config)
    fourier_power = _power(fourier_pre, config)
    after_prop_lens2 = _power(lens2_input, config)
    output_power = _power(relay_output, config)
    ledger = (
        _ledger_row("SLM2_phase_plane", "phase_plane", source_power, after_slm2, "carrier phase only; no axicon on SLM2"),
        _ledger_row("SLM2_to_lens1_propagation", "free_space_propagation_segment", after_slm2, after_prop_lens1, "300 mm nominal propagation"),
        _ledger_row("lens1_thin_phase_and_pupil", "thin_lens_with_finite_pupil", lens1_before, lens1_after, "scalar thin-lens phase plus editable nominal pupil"),
        _ledger_row("lens1_to_fourier_plane_propagation", "free_space_propagation_segment", lens1_after, fourier_power, "300 mm nominal propagation"),
        _ledger_row("fourier_stop_pinhole", "fourier_plane_circular_amplitude_stop", stop_before, stop_after, "nominal_stop_parameter; not_measured_bench_geometry"),
        _ledger_row("fourier_plane_to_lens2_propagation", "free_space_propagation_segment", stop_after, after_prop_lens2, "300 mm nominal propagation"),
        _ledger_row("lens2_thin_phase_and_pupil", "thin_lens_with_finite_pupil", lens2_before, lens2_after, "scalar thin-lens phase plus editable nominal pupil"),
        _ledger_row("lens2_to_nominal_relay_output_propagation", "free_space_propagation_segment", lens2_after, output_power, "300 mm nominal propagation"),
    )
    manifest = (
        _manifest_row("SLM2_phase_plane", "phase_plane", True, None, "nominal_4f_forward_model", "SLM2 owns carrier only"),
        _manifest_row("SLM2_to_lens1_propagation", "free_space_propagation_segment", True, config.slm2_to_lens1_m, "nominal_4f_forward_model", "actual distance is nominal bench description"),
        _manifest_row("lens1_thin_phase_and_pupil", "thin_lens_with_finite_pupil", True, None, "nominal_4f_forward_model", "f=300 mm thin scalar lens"),
        _manifest_row("lens1_to_fourier_plane_propagation", "free_space_propagation_segment", True, config.lens1_to_fourier_plane_m, "nominal_4f_forward_model", "propagate to Fourier plane"),
        _manifest_row("fourier_plane_field_pre_stop", "diagnostic_field_plane", False, None, "diagnostic_boundary", "complex field before stop"),
        _manifest_row("fourier_stop_pinhole", "fourier_plane_circular_amplitude_stop", True, None, "nominal_stop_parameter", "applied at Fourier plane, not output crop"),
        _manifest_row("fourier_plane_field_post_stop", "diagnostic_field_plane", False, None, "diagnostic_boundary", "complex field after stop"),
        _manifest_row("fourier_plane_to_lens2_propagation", "free_space_propagation_segment", True, config.fourier_plane_to_lens2_m, "nominal_4f_forward_model", "propagate from Fourier plane to lens 2"),
        _manifest_row("lens2_thin_phase_and_pupil", "thin_lens_with_finite_pupil", True, None, "nominal_4f_forward_model", "f=300 mm thin scalar lens"),
        _manifest_row("lens2_to_nominal_relay_output_propagation", "free_space_propagation_segment", True, config.lens2_to_nominal_relay_output_m, "nominal_4f_forward_model", "propagate to nominal relay output"),
        _manifest_row("nominal_relay_output_plane", "diagnostic_field_plane", False, None, "diagnostic_boundary", "simulation stops here unless explicit axicon handoff scenario is enabled"),
    )
    sampling = stop_sampling_report(config)
    warnings = list(nominal_4f_warnings_from_fields(config, ledger, lens1_field, lens2_input, stop, relay_output))
    if sampling["stop_sampling_status"] == "underresolved":
        warnings.append("stop sampling underresolved: do not use for stop robustness ranking")
    elif sampling["stop_sampling_status"] == "exploratory_only":
        warnings.append("stop sampling exploratory_only: not_for_stop_robustness_ranking")
    warnings_tuple = tuple(warnings)
    diagnostics = {
        "slm2_input_metrics": _field_metrics(slm2_input, config),
        "field_post_slm2_metrics": _field_metrics(field, config),
        "fourier_plane_pre_stop_metrics": _field_metrics(fourier_pre, config),
        "nominal_relay_output_metrics": _field_metrics(relay_output, config),
        "pinhole_transmitted_fraction": float(stop_after / max(stop_before, EPS)),
        "lens1_pupil_transmitted_fraction": float(lens1_after / max(lens1_before, EPS)),
        "lens2_pupil_transmitted_fraction": float(lens2_after / max(lens2_before, EPS)),
        "carrier_boundary": carrier_boundary_record(),
        "stop_sampling": sampling,
        "upstream": upstream_diag,
    }
    return NominalF300Run(
        config=config,
        grid=grid,
        slm1_phase_rad=slm1_phase,
        slm2_phase_rad=slm2_phase,
        slm2_input_field=slm2_input,
        field_post_slm2=field,
        post_lens1_field=lens1_after_field,
        fourier_plane_field_pre_stop=fourier_pre,
        fourier_stop_transmission=stop,
        fourier_plane_field_post_stop=fourier_post,
        post_lens2_field=lens2_after_field,
        nominal_relay_output_field=relay_output,
        component_energy_ledger=ledger,
        component_manifest=manifest,
        warnings=warnings_tuple,
        diagnostics=diagnostics,
        upstream_source_mode=upstream_source_mode,
        upstream_component_chain=upstream_chain,
        upstream_diagnostics=upstream_diag,
        slm1_phase_applied_at_slm1=slm1_phase_applied,
        slm1_to_slm2_propagation_included=slm1_to_slm2_included,
        slm2_carrier_applied_at_slm2=slm2_carrier_applied,
        stop_sampling_report=sampling,
    )


def nominal_4f_warnings_from_fields(
    config: NominalF300Config,
    ledger: Sequence[Mapping[str, Any]],
    lens1_input_field: np.ndarray,
    lens2_input_field: np.ndarray,
    stop: np.ndarray,
    relay_output: np.ndarray,
) -> list[str]:
    warnings: list[str] = []
    carrier_cycles = max(abs(config.numerical_model_carrier_cycles_x), abs(config.numerical_model_carrier_cycles_y))
    pixels_per_cycle = float("inf") if carrier_cycles <= 0 else int(config.simulation_grid_size) / carrier_cycles
    if pixels_per_cycle < 8.0:
        warnings.append("carrier undersampling: fewer than 8 pixels per numerical carrier cycle")
    for label, field in (("lens1", lens1_input_field), ("lens2", lens2_input_field)):
        before = _power(field, config)
        after = _power(field * circular_amplitude(_grid(config), config.lens_clear_radius_m), config)
        if after / max(before, EPS) < 0.98:
            warnings.append(f"beam clipping at {label} pupil")
    half = 0.5 * config.simulation_plane_width_m
    if abs(config.pinhole_offset_x_m) + config.pinhole_radius_m > half or abs(config.pinhole_offset_y_m) + config.pinhole_radius_m > half:
        warnings.append("pinhole lying partly outside the simulation FOV")
    if _border_power_fraction(np.abs(relay_output) ** 2) > 0.02:
        warnings.append("output clipping: relay output has nontrivial border power")
    for row in ledger:
        ctype = str(row["component_type"])
        if ("pupil" in ctype or "stop" in ctype) and float(row["energy_after_arb_m2"]) > float(row["energy_before_arb_m2"]) * (1.0 + 1e-9):
            warnings.append(f"power increase after passive stop/pupil at {row['component_id']}")
    dx = config.dx_m
    if config.pinhole_radius_m < 2.0 * dx:
        warnings.append("insufficient Fourier-plane sampling: pinhole radius spans fewer than two samples")
    if not np.any(stop > 0):
        warnings.append("pinhole transmits no sampled pixels")
    return warnings


def nominal_4f_sanity_report(run: NominalF300Run) -> dict[str, Any]:
    cfg = run.config
    carrier_cycles = max(abs(cfg.numerical_model_carrier_cycles_x), abs(cfg.numerical_model_carrier_cycles_y))
    pixels_per_cycle = float("inf") if carrier_cycles <= 0 else int(cfg.simulation_grid_size) / carrier_cycles
    input_i = np.abs(run.slm2_input_field) ** 2
    output_i = np.abs(run.nominal_relay_output_field) ** 2
    flipped_input = np.flipud(np.fliplr(input_i))
    denom = np.linalg.norm(flipped_input.ravel()) * np.linalg.norm(output_i.ravel()) + EPS
    relay_corr = float(np.dot(flipped_input.ravel(), output_i.ravel()) / denom)
    return {
        "grid_size": int(cfg.simulation_grid_size),
        "simulation_plane_width_m": float(cfg.simulation_plane_width_m),
        "sampling_pitch_m": float(cfg.dx_m),
        "wavelength_m": float(cfg.wavelength_m),
        "maximum_spatial_frequency_content_cpm": float(carrier_cycles / max(cfg.simulation_plane_width_m, EPS)),
        "carrier_cycles_across_model_width": [float(cfg.numerical_model_carrier_cycles_x), float(cfg.numerical_model_carrier_cycles_y)],
        "pixels_per_carrier_cycle": float(pixels_per_cycle),
        "lens_pupil_radius_m": float(cfg.lens_clear_radius_m),
        "pinhole_radius_m": float(cfg.pinhole_radius_m),
        "propagation_distances_m": {
            "slm2_to_lens1_m": float(cfg.slm2_to_lens1_m),
            "lens1_to_fourier_plane_m": float(cfg.lens1_to_fourier_plane_m),
            "fourier_plane_to_lens2_m": float(cfg.fourier_plane_to_lens2_m),
            "lens2_to_nominal_relay_output_m": float(cfg.lens2_to_nominal_relay_output_m),
        },
        "energy_ledger": list(run.component_energy_ledger),
        "warnings": list(run.warnings),
        "open_stop_relay_intensity_correlation_with_inverted_input": relay_corr,
        "upstream_source_mode": run.upstream_source_mode,
        "upstream_component_chain": list(run.upstream_component_chain),
        "slm1_phase_applied_at_slm1": bool(run.slm1_phase_applied_at_slm1),
        "slm1_to_slm2_propagation_included": bool(run.slm1_to_slm2_propagation_included),
        "slm2_carrier_applied_at_slm2": bool(run.slm2_carrier_applied_at_slm2),
        "carrier_coordinate_status": run.carrier_coordinate_status,
        "carrier_boundary": carrier_boundary_record(),
        "stop_sampling": dict(run.stop_sampling_report),
        "convergence_status": run.convergence_status,
        "final_export_allowed": bool(run.final_export_allowed),
    }


def fourier_plane_centroid_m(run: NominalF300Run) -> tuple[float, float]:
    m = _field_metrics(run.fourier_plane_field_pre_stop, run.config)
    return float(m["centroid_x_m"]), float(m["centroid_y_m"])


def phase_export_payload(phase_rad: np.ndarray, *, mask_id: str, slm_id: str, config: NominalF300Config) -> dict[str, Any]:
    bits = max(1, int(round(np.log2(int(config.slm_phase_quantisation_levels)))))
    wrapped = wrap_phase_rad(phase_rad)
    gray = phase_to_gray(wrapped, bits=bits)
    return {
        "phase_rad": wrapped,
        "quantised_rad": quantize_phase_rad(wrapped, bits),
        "gray": gray,
        "metadata": {
            "mask_id": mask_id,
            "slm_id": slm_id,
            "stage": STAGE,
            "simulation_status": "nominal_unvalidated",
            "physical_4f_readiness": "blocked",
            "carrier_coordinate_status": "nominal_model_not_bench_calibrated",
            "carrier_realism": CARRIER_REALISM if slm_id == "SLM2" else "not_applicable",
            "ideal_blazed_carrier_shift_surrogate": bool(slm_id == "SLM2"),
            "pixelated_slm_diffraction_orders_modelled": False,
            "zero_order_modelled": False,
            "physical_order_efficiency_modelled": False,
            "selected_order_purity_predicted": False,
            "hardware_command_export_status": "command_masks_exportable_unvalidated",
            "contains_axicon": False,
            "final_export_allowed": False,
        },
    }


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    return out


def write_json(path: str | Path, obj: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    return out


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def run_to_manifest(run: NominalF300Run) -> dict[str, Any]:
    return {
        "stage": STAGE,
        "model_label": MODEL_LABEL,
        "claim_boundary_labels": list(CLAIM_BOUNDARY_LABELS),
        "simulation_status": "nominal_unvalidated",
        "upstream_source_mode": run.upstream_source_mode,
        "upstream_component_chain": list(run.upstream_component_chain),
        "slm1_phase_applied_at_slm1": bool(run.slm1_phase_applied_at_slm1),
        "slm1_to_slm2_propagation_included": bool(run.slm1_to_slm2_propagation_included),
        "slm2_carrier_applied_at_slm2": bool(run.slm2_carrier_applied_at_slm2),
        "carrier_realism": run.carrier_realism,
        "ideal_blazed_carrier_shift_surrogate": bool(run.ideal_blazed_carrier_shift_surrogate),
        "pixelated_slm_diffraction_orders_modelled": bool(run.pixelated_slm_diffraction_orders_modelled),
        "zero_order_modelled": bool(run.zero_order_modelled),
        "physical_order_efficiency_modelled": bool(run.physical_order_efficiency_modelled),
        "selected_order_purity_predicted": bool(run.selected_order_purity_predicted),
        "carrier_boundary": carrier_boundary_record(),
        "stop_sampling": dict(run.stop_sampling_report),
        "convergence_status": run.convergence_status,
        "physical_4f_readiness": run.physical_4f_readiness,
        "camera_validation": "absent",
        "material_prediction": "absent",
        "nominal_4f_forward_model": run.nominal_4f_forward_model,
        "bench_calibrated": run.bench_calibrated,
        "carrier_coordinate_status": run.carrier_coordinate_status,
        "component_sequence": list(NOMINAL_COMPONENT_SEQUENCE),
        "component_manifest": list(run.component_manifest),
        "component_energy_ledger": list(run.component_energy_ledger),
        "upstream_diagnostics": dict(run.upstream_diagnostics),
        "warnings": list(run.warnings),
        "config": asdict(run.config),
        "final_export_allowed": bool(run.final_export_allowed),
    }


def plot_component_sequence(
    run: NominalF300Run | None = None,
    output_path: str | Path = "outputs/figures/digital_twin/stage9b0_nominal_f300_4f_component_sequence.png",
):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    run = run or run_nominal_f300_4f(NominalF300Config.fast())
    x_mm = np.asarray(run.grid["x"], float) * 1e3
    extent = (float(x_mm.min()), float(x_mm.max()), float(x_mm.min()), float(x_mm.max()))
    panels = [
        (np.abs(run.slm2_input_field) ** 2, "SLM2 input intensity"),
        (np.abs(run.post_lens1_field) ** 2, "post Lens 1 pupil"),
        (np.abs(run.fourier_plane_field_pre_stop) ** 2, "Fourier plane pre-stop"),
        (run.fourier_stop_transmission, "Fourier stop transmission"),
        (np.abs(run.fourier_plane_field_post_stop) ** 2, "Fourier plane post-stop"),
        (np.abs(run.post_lens2_field) ** 2, "post Lens 2 pupil"),
        (np.abs(run.nominal_relay_output_field) ** 2, "nominal relay output"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(14.5, 11.0), facecolor="white")
    fig.suptitle(f"{MODEL_LABEL}\ncomponent-owned scalar propagation, lenses, Fourier stop, relay output",
                 x=0.04, y=0.98, ha="left", va="top", fontsize=14, fontweight="bold")
    for ax, (arr, title) in zip(axes.ravel(), panels):
        im = ax.imshow(arr, origin="lower", extent=extent, cmap="viridis")
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes.ravel()[7].axis("off")
    lines = [
        "Energy ledger",
        *[
            f"{row['component_id']}: {row['transmitted_fraction']:.3f}"
            for row in run.component_energy_ledger
        ],
    ]
    axes.ravel()[7].text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=7.2)
    axes.ravel()[8].axis("off")
    axes.ravel()[8].text(
        0.0,
        1.0,
        "Boundary\n"
        "nominal_4f_forward_model=True\n"
        "carrier_realism=ideal_continuous_phase_ramp\n"
        "pixelated_SLM_order_physics_modelled=False\n"
        "bench_calibrated=False\n"
        "physical_4f_readiness=blocked\n"
        "camera/material models absent\n"
        "final_export_allowed=False",
        va="top",
        family="monospace",
        fontsize=8.8,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=165, bbox_inches="tight", metadata={"Title": MODEL_LABEL, "final_export_allowed": "False"})
    plt.close(fig)
    return out


__all__ = [
    "CARRIER_BOUNDARY_FLAGS",
    "CARRIER_REALISM",
    "CLAIM_BOUNDARY_LABELS",
    "CONVERGENCE_STATUSES",
    "FINAL_EXPORT_ALLOWED",
    "IDEAL_CARRIER_BOUNDARY",
    "MODEL_LABEL",
    "NOMINAL_COMPONENT_SEQUENCE",
    "NominalF300Config",
    "NominalF300Run",
    "STOP_SAMPLING_STATUSES",
    "UPSTREAM_SOURCE_MODES",
    "UPSTREAM_TO_F300_COMPONENT_CHAIN",
    "UpstreamSLM2FieldBridge",
    "build_existing_cslm_slm2_bridge",
    "build_synthetic_gaussian_slm2_bridge",
    "carrier_boundary_record",
    "carrier_phase",
    "circular_amplitude",
    "config_from_profile",
    "evaluate_stop_sampling_convergence",
    "fourier_plane_centroid_m",
    "load_nominal_f300_profile",
    "nominal_4f_sanity_report",
    "phase_export_payload",
    "plot_component_sequence",
    "replace_config",
    "run_nominal_f300_4f",
    "run_to_manifest",
    "stop_sampling_report",
    "thin_lens_phase",
    "vortex_phase",
    "write_csv",
    "write_json",
]


if __name__ == "__main__":
    plot_component_sequence()
