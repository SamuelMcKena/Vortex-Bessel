"""Native-array case assembly for PHASE 2B visual diagnostics.

This module does not define a new optical model. It adapts the fixed PHASE 2A
scalar chain and the accepted Nathan MODE 2Y/2Z vector chain into memory-bounded
diagnostic volumes. Metrics remain tied to native arrays; spatial reduction is
used only for three-dimensional rendering.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.digital_twin.nathan_mode2u2_fix_strict_hexagon_optimisation import (
    evaluate_strict_hexagon_metrics,
)
from vbb_study.digital_twin.nathan_mode2w_fix_sequential_master import (
    _bench_from_config,
    _fixed_useful_region,
    _route_metrics,
    _sas_zoom_plane,
    _source_config,
)
from vbb_study.digital_twin.nathan_mode2y_continuous_vs_averaged import (
    Mode2YStudyConfig,
    _after_axicon,
    _intensity_from_prepared,
    _prepare_projected_spectrum,
    _radial_profile_native,
    _realistic_common_4f_field,
    build_mode2y_input_fields,
    propagated_shape_metrics,
)
from vbb_study.digital_twin.nathan_vector_hexagon import (
    Mode2SCorrection,
    Mode2SPerturbation,
    angular_profile_on_ring,
    mode2s_combined_cases,
    run_mode2n_dual_slm_qwp_route,
    run_mode2s_degraded_forward,
)
from vbb_study.digital_twin.phase2a_canonical import (
    _axicon_phase,
    _fourier_first_order,
    _normalised_power,
    _panel_from_manifest,
    _pupil_and_aberration,
    _radial_metrics,
    _variant_settings,
)
from vbb_study.digital_twin.phase2a_contracts import (
    PHASE2A_CANONICAL_SLM_MODEL,
    canonical_hardware_manifest,
    hardware_value,
)
from vbb_study.equations.fields import make_xy_grid
from vbb_study.equations.propagation import (
    angular_spectrum_propagate_bl,
    scalable_angular_spectrum_propagate,
)
from vbb_study.slm_model import apply_slm, slm_active_aperture


EPS = np.finfo(float).eps
PHASE2B_CASE_IDS = ("G0", "B0", "V1", "V3", "H1")
PHASE2B_3D_CASE_IDS = (
    "B0",
    "V1",
    "V3",
    "H1_REALISTIC",
    "H1_CONTINUOUS",
    "H1_AVERAGED",
)
PHASE2B_HEX_EARLY_M = 30.0e-3
PHASE2B_HEX_REFERENCE_M = 60.0e-3
PHASE2B_HEX_LATE_M = 150.0e-3


@dataclass(frozen=True)
class Phase2BConfig:
    """Sampling controls for the analysis layer, not for the optical contracts."""

    scalar_grid_n: int = 512
    hex_grid_n: int = 1024
    hero_grid_n: int = 1536
    z_start_m: float = 0.0
    z_end_m: float = 0.2
    z_step_m: float = 0.002
    render_xy_max: int = 224
    render_z_stride: int = 2
    sas_pad_factor: int = 2
    publication_quality: bool = True
    highn_hero: bool = True

    def z_values_m(self) -> np.ndarray:
        count = int(round((self.z_end_m - self.z_start_m) / self.z_step_m)) + 1
        values = np.linspace(self.z_start_m, self.z_end_m, count)
        for required in (PHASE2B_HEX_EARLY_M, PHASE2B_HEX_REFERENCE_M, PHASE2B_HEX_LATE_M):
            if not np.any(np.isclose(values, required, rtol=0.0, atol=1e-12)):
                raise ValueError(f"Phase 2B z grid does not include {required:g} m")
        return values

    def validate(self) -> None:
        if min(self.scalar_grid_n, self.hex_grid_n) < 96:
            raise ValueError("Phase 2B diagnostic grids must be at least N=96")
        if self.publication_quality:
            if self.scalar_grid_n < 512 or self.hex_grid_n < 1024:
                raise ValueError("publication Phase 2B requires scalar N>=512 and hex N>=1024")
            if self.highn_hero and self.hero_grid_n < 1536:
                raise ValueError("publication high-N hex heroes require N>=1536")
        if self.render_xy_max < 48 or self.render_z_stride < 1:
            raise ValueError("invalid 3D rendering reduction")
        self.z_values_m()


@dataclass(frozen=True)
class Phase2BCaseResult:
    """One native propagation result plus a display-only reduced volume."""

    case_id: str
    family: str
    route: str
    native_grid_n: int
    native_dx_m: float
    z_values_m: np.ndarray = field(repr=False, compare=False)
    selected_planes: Mapping[float, np.ndarray] = field(repr=False, compare=False)
    xz_map: np.ndarray = field(repr=False, compare=False)
    yz_map: np.ndarray = field(repr=False, compare=False)
    power_by_z: np.ndarray = field(repr=False, compare=False)
    render_volume: np.ndarray = field(repr=False, compare=False)
    render_x_m: np.ndarray = field(repr=False, compare=False)
    render_y_m: np.ndarray = field(repr=False, compare=False)
    render_z_m: np.ndarray = field(repr=False, compare=False)
    radial_radius_m: np.ndarray = field(repr=False, compare=False)
    radial_intensity: np.ndarray = field(repr=False, compare=False)
    angular_theta_rad: np.ndarray = field(repr=False, compare=False)
    angular_intensity: np.ndarray = field(repr=False, compare=False)
    focus_halfwidth_m: float
    ring_radius_m: float
    summary: Mapping[str, Any]
    metadata: Mapping[str, Any]
    sas_hero: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class Phase2BHexPackage:
    """Accepted continuous/averaged and cross-route hex diagnostic data."""

    realistic: Phase2BCaseResult
    continuous: Phase2BCaseResult
    averaged: Phase2BCaseResult
    cross_route_cases: tuple[Mapping[str, Any], ...]
    cross_route_metrics: tuple[Mapping[str, Any], ...]
    highn_hero: Mapping[str, Any]
    endpoint_audit: tuple[Mapping[str, Any], ...]
    bench: Mapping[str, Any] = field(repr=False, compare=False)


def phase2b_case_registry() -> tuple[dict[str, Any], ...]:
    """Return the fixed visual case registry."""

    return (
        {"case_id": "G0", "family": "Gaussian", "vortex_charge": 0, "needs_3d": False},
        {"case_id": "B0", "family": "Bessel", "vortex_charge": 0, "needs_3d": True},
        {"case_id": "V1", "family": "vortex Bessel", "vortex_charge": 1, "needs_3d": True},
        {"case_id": "V3", "family": "vortex Bessel", "vortex_charge": 3, "needs_3d": True},
        {"case_id": "H1", "family": "continuous vector hexagonal field", "vortex_charge": None, "needs_3d": True},
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _scalar_seed(case_id: str, ell: int, *, grid_n: int) -> tuple[np.ndarray, Mapping[str, Any], dict[str, Any]]:
    """Recreate the accepted PHASE 2A realistic scalar field at the axicon exit."""

    manifest = canonical_hardware_manifest()
    variant = "realistic_fixed_bench_route"
    settings = _variant_settings(variant)
    wavelength = float(hardware_value(manifest, "wavelength_m"))
    beam_radius = float(hardware_value(manifest, "beam_radius_on_slm_m"))
    grid = make_xy_grid(int(grid_n), 10.0e-3 / int(grid_n))
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    bx, by = settings["input_decentre_m"]
    raw_input = np.exp(-((X - float(bx)) ** 2 + (Y - float(by)) ** 2) / beam_radius**2)
    panel = _panel_from_manifest(manifest)
    panel_aperture = slm_active_aperture(grid, panel)
    input_aperture_fraction = _normalised_power(np.where(panel_aperture, raw_input, 0.0)) / max(
        _normalised_power(raw_input), EPS
    )
    hx, hy = settings["hologram_offset_m"]
    theta = np.arctan2(Y - float(hy), X - float(hx))
    radius_norm = np.hypot(X, Y) / max(2.0 * beam_radius, EPS)
    phase_error = float(settings["slm_phase_error_rms_rad"]) * (2.0 * radius_norm**2 - 1.0)
    phase1 = float(ell) * theta + phase_error
    phase2 = 0.5 * phase_error
    field0 = np.asarray(raw_input, dtype=np.complex128)
    slm1 = apply_slm(
        field0,
        phase1,
        grid,
        panel,
        quantise_phase=True,
        apply_fill_factor=True,
        apply_carrier=False,
        fill_factor_model=PHASE2A_CANONICAL_SLM_MODEL,
    )
    slm2 = apply_slm(
        slm1.total,
        phase2,
        grid,
        panel,
        quantise_phase=True,
        apply_fill_factor=True,
        apply_carrier=True,
        fill_factor_model=PHASE2A_CANONICAL_SLM_MODEL,
    )
    field_after_filter, first_order = _fourier_first_order(
        slm2.total,
        grid,
        float(hardware_value(manifest, "carrier_frequency_cpm")),
        float(hardware_value(manifest, "fourier_iris_radius_cpm")),
        0.0,
    )
    field_after_pupil, pupil_fraction = _pupil_and_aberration(
        field_after_filter,
        grid,
        float(hardware_value(manifest, "objective_pupil_radius_m")),
        settings,
    )
    kr = 0.0
    field_after_axicon = field_after_pupil
    if case_id != "G0":
        axicon, kr = _axicon_phase(grid, manifest, settings)
        field_after_axicon = field_after_pupil * axicon
    return np.asarray(field_after_axicon, dtype=np.complex128), grid, {
        "wavelength_m": wavelength,
        "input_aperture_fraction": float(input_aperture_fraction),
        "first_order_efficiency": float(first_order),
        "objective_pupil_fraction": float(pupil_fraction),
        "radial_wavevector_m_inv": float(kr),
        "mapping_mode": "fixed_physical_optics",
        "slm_fill_factor_model": PHASE2A_CANONICAL_SLM_MODEL,
        "source_contract": "PHASE 2A realistic_fixed_bench_route",
    }


def _focus_indices(grid: Mapping[str, Any], halfwidth_m: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(grid["x"], dtype=float)
    y = np.asarray(grid.get("y", x), dtype=float)
    xi = np.flatnonzero(np.abs(x) <= float(halfwidth_m))
    yi = np.flatnonzero(np.abs(y) <= float(halfwidth_m))
    if xi.size < 8 or yi.size < 8:
        raise ValueError("focus crop has too few native samples")
    return yi, xi


def _display_stride(length: int, maximum: int) -> int:
    return max(1, int(np.ceil(float(length) / float(maximum))))


def _profiles(
    plane: np.ndarray,
    grid: Mapping[str, Any],
    ring_radius_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    radial_r, radial_i = _radial_profile_native(plane, grid)
    if ring_radius_m > 0.0:
        theta, angular = angular_profile_on_ring(plane, grid, float(ring_radius_m), angular_bins=720)
    else:
        theta = np.linspace(-np.pi, np.pi, 720, endpoint=False)
        angular = np.full(theta.shape, np.nan)
    return radial_r, radial_i, np.asarray(theta), np.asarray(angular)


def _sas_scalar_hero(
    field: np.ndarray,
    grid: Mapping[str, Any],
    wavelength_m: float,
    *,
    z_m: float,
    pad_factor: int,
) -> dict[str, Any]:
    out, out_grid, meta = scalable_angular_spectrum_propagate(
        field,
        dict(grid),
        float(wavelength_m),
        float(z_m),
        n_medium=1.0,
        pad_factor=int(pad_factor),
        bandlimit=True,
        skip_final_phase=True,
        allow_invalid=False,
    )
    return {
        "intensity": np.asarray(np.abs(out) ** 2, dtype=np.float32),
        "grid": out_grid,
        "method": "scalable_angular_spectrum_zoom",
        "z_m": float(z_m),
        "input_N": int(grid["N"]),
        "input_dx_m": float(grid["dx"]),
        "output_N": int(out_grid["N"]),
        "output_dx_m": float(out_grid["dx"]),
        "sas_valid": bool(meta["valid"]),
        "component_meta": [meta],
    }


def _stream_scalar_case(
    case_id: str,
    family: str,
    ell: int,
    config: Phase2BConfig,
) -> Phase2BCaseResult:
    field0, grid, seed_meta = _scalar_seed(case_id, ell, grid_n=config.scalar_grid_n)
    wavelength = float(seed_meta["wavelength_m"])
    z_values = config.z_values_m()
    z60 = angular_spectrum_propagate_bl(field0, dict(grid), wavelength, PHASE2B_HEX_REFERENCE_M)
    plane60 = np.asarray(np.abs(z60) ** 2, dtype=np.float32)
    native_summary = _radial_metrics(plane60, grid)
    ring_radius = float(native_summary["dominant_off_axis_ring_radius_m"])
    if case_id == "G0":
        halfwidth = 3.0e-3
    else:
        halfwidth = float(np.clip(4.5 * ring_radius, 0.70e-3, 2.2e-3))
    yi, xi = _focus_indices(grid, halfwidth)
    sy = slice(int(yi[0]), int(yi[-1]) + 1)
    sx = slice(int(xi[0]), int(xi[-1]) + 1)
    xy_stride = max(_display_stride(yi.size, config.render_xy_max), _display_stride(xi.size, config.render_xy_max))
    render_indices = set(range(0, z_values.size, int(config.render_z_stride)))
    render_indices.add(int(np.argmin(np.abs(z_values - PHASE2B_HEX_REFERENCE_M))))
    selected_targets = (0.0, 20e-3, 60e-3, 120e-3, 200e-3)
    selected: dict[float, np.ndarray] = {}
    xz = np.empty((z_values.size, int(grid["N"])), dtype=np.float32)
    yz = np.empty_like(xz)
    powers = np.empty(z_values.size, dtype=float)
    render_planes: list[np.ndarray] = []
    render_z: list[float] = []
    mid = int(grid["N"]) // 2
    for index, z_m in enumerate(z_values):
        propagated = field0 if np.isclose(z_m, 0.0) else angular_spectrum_propagate_bl(
            field0, dict(grid), wavelength, float(z_m), n_medium=1.0
        )
        plane = np.asarray(np.abs(propagated) ** 2, dtype=np.float32)
        xz[index] = plane[mid, :]
        yz[index] = plane[:, mid]
        powers[index] = float(np.sum(plane, dtype=float) * float(grid["dx"]) ** 2)
        for target in selected_targets:
            if np.isclose(z_m, target, rtol=0.0, atol=1e-12):
                selected[float(target)] = plane
        if index in render_indices:
            render_planes.append(plane[sy, sx][::xy_stride, ::xy_stride])
            render_z.append(float(z_m))
    radial_r, radial_i, theta, angular = _profiles(plane60, grid, ring_radius if case_id != "G0" else 0.0)
    drift = float((np.max(powers) - np.min(powers)) / max(float(np.max(powers)), EPS))
    summary = {
        **native_summary,
        "case_id": case_id,
        "family": family,
        "route": "realistic_fixed_bench_route",
        "vortex_charge": int(ell),
        "propagation_power_drift_fraction": drift,
        "power_min": float(np.min(powers)),
        "power_max": float(np.max(powers)),
        "native_metrics": True,
    }
    hero = _sas_scalar_hero(
        field0,
        grid,
        wavelength,
        z_m=PHASE2B_HEX_REFERENCE_M,
        pad_factor=config.sas_pad_factor,
    )
    x = np.asarray(grid["x"], dtype=float)
    y = np.asarray(grid.get("y", x), dtype=float)
    return Phase2BCaseResult(
        case_id=case_id,
        family=family,
        route="realistic_fixed_bench_route",
        native_grid_n=int(grid["N"]),
        native_dx_m=float(grid["dx"]),
        z_values_m=z_values,
        selected_planes=selected,
        xz_map=xz,
        yz_map=yz,
        power_by_z=powers,
        render_volume=np.asarray(render_planes, dtype=np.float32),
        render_x_m=x[sx][::xy_stride],
        render_y_m=y[sy][::xy_stride],
        render_z_m=np.asarray(render_z, dtype=float),
        radial_radius_m=radial_r,
        radial_intensity=radial_i,
        angular_theta_rad=theta,
        angular_intensity=angular,
        focus_halfwidth_m=halfwidth,
        ring_radius_m=ring_radius,
        summary=summary,
        metadata={
            **seed_meta,
            "native_metrics_only": True,
            "display_interpolation_used_for_metrics": False,
            "render_spatial_stride": int(xy_stride),
            "render_z_stride": int(config.render_z_stride),
            "render_downsampling_method": "native index stride after focus crop",
            "metric_plane_z_m": PHASE2B_HEX_REFERENCE_M,
        },
        sas_hero=hero,
    )


def build_scalar_cases(config: Phase2BConfig) -> dict[str, Phase2BCaseResult]:
    """Build G0/B0/V1/V3 from the exact PHASE 2A realistic scalar chain."""

    config.validate()
    results: dict[str, Phase2BCaseResult] = {}
    for case_id, family, ell in (
        ("G0", "Gaussian", 0),
        ("B0", "Bessel", 0),
        ("V1", "vortex Bessel", 1),
        ("V3", "vortex Bessel", 3),
    ):
        results[case_id] = _stream_scalar_case(case_id, family, ell, config)
    return results


def _stream_vector_case(
    case_id: str,
    input_model: str,
    prepared: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    data: Mapping[str, Any],
    ring_radius_m: float,
    config: Phase2BConfig,
    *,
    first_order_efficiency: float,
) -> Phase2BCaseResult:
    grid = data["grid"]
    z_values = config.z_values_m()
    halfwidth = max(3.6 * float(ring_radius_m), 0.65e-3)
    yi, xi = _focus_indices(grid, halfwidth)
    sy = slice(int(yi[0]), int(yi[-1]) + 1)
    sx = slice(int(xi[0]), int(xi[-1]) + 1)
    xy_stride = max(_display_stride(yi.size, config.render_xy_max), _display_stride(xi.size, config.render_xy_max))
    render_indices = set(range(0, z_values.size, int(config.render_z_stride)))
    selected_targets = (
        0.0,
        PHASE2B_HEX_EARLY_M,
        PHASE2B_HEX_REFERENCE_M,
        90e-3,
        PHASE2B_HEX_LATE_M,
        200e-3,
    )
    selected: dict[float, np.ndarray] = {}
    n = int(grid["N"])
    xz = np.empty((z_values.size, n), dtype=np.float32)
    yz = np.empty_like(xz)
    powers = np.empty(z_values.size, dtype=float)
    render_planes: list[np.ndarray] = []
    render_z: list[float] = []
    mid = n // 2
    for index, z_m in enumerate(z_values):
        plane = np.asarray(_intensity_from_prepared(prepared, float(z_m)), dtype=np.float32)
        xz[index] = plane[mid, :]
        yz[index] = plane[:, mid]
        powers[index] = float(np.sum(plane, dtype=float) * float(grid["dx"]) ** 2)
        for target in selected_targets:
            if np.isclose(z_m, target, rtol=0.0, atol=1e-12):
                selected[float(target)] = plane
        if index in render_indices:
            render_planes.append(plane[sy, sx][::xy_stride, ::xy_stride])
            render_z.append(float(z_m))
    plane60 = selected[PHASE2B_HEX_REFERENCE_M]
    radial_r, radial_i, theta, angular = _profiles(plane60, grid, ring_radius_m)
    useful_mask, _ = _fixed_useful_region(grid, ring_radius_m)
    metrics = propagated_shape_metrics(
        plane60,
        grid,
        z_m=PHASE2B_HEX_REFERENCE_M,
        ring_radius_m=ring_radius_m,
        useful_mask=useful_mask,
    )
    drift = float((np.max(powers) - np.min(powers)) / max(float(np.max(powers)), EPS))
    summary = {
        "case_id": case_id,
        "family": "continuous vector hexagonal field" if input_model == "continuous" else "sector-averaged vector hexagonal surrogate",
        "route": "realistic sequential common-4F",
        "input_model": input_model,
        "first_order_efficiency": float(first_order_efficiency),
        "propagation_power_drift_fraction": drift,
        "power_min": float(np.min(powers)),
        "power_max": float(np.max(powers)),
        **metrics.__dict__,
        "native_metrics": True,
    }
    x = np.asarray(grid["x"], dtype=float)
    y = np.asarray(grid.get("y", x), dtype=float)
    return Phase2BCaseResult(
        case_id=case_id,
        family=str(summary["family"]),
        route="realistic sequential common-4F",
        native_grid_n=n,
        native_dx_m=float(grid["dx"]),
        z_values_m=z_values,
        selected_planes=selected,
        xz_map=xz,
        yz_map=yz,
        power_by_z=powers,
        render_volume=np.asarray(render_planes, dtype=np.float32),
        render_x_m=x[sx][::xy_stride],
        render_y_m=y[sy][::xy_stride],
        render_z_m=np.asarray(render_z, dtype=float),
        radial_radius_m=radial_r,
        radial_intensity=radial_i,
        angular_theta_rad=theta,
        angular_intensity=angular,
        focus_halfwidth_m=float(halfwidth),
        ring_radius_m=float(ring_radius_m),
        summary=summary,
        metadata={
            "source_contract": "MODE 2Y realistic continuous-versus-averaged",
            "native_metrics_only": True,
            "display_interpolation_used_for_metrics": False,
            "render_spatial_stride": int(xy_stride),
            "render_z_stride": int(config.render_z_stride),
            "render_downsampling_method": "native index stride after matched focus crop",
            "metric_plane_z_m": PHASE2B_HEX_REFERENCE_M,
        },
    )


def _mode2y_endpoint_audit(results: Sequence[Phase2BCaseResult]) -> tuple[dict[str, Any], ...]:
    old_rows = _read_csv(
        Path("outputs/figures/digital_twin/nathan_mode2y_continuous_vs_averaged/continuous_vs_averaged_summary.csv")
    )
    by_model = {
        str(row["input_model"]): row for row in old_rows if str(row["optical_route"]) == "realistic"
    }
    keys = (
        "peak_intensity",
        "useful_power_fraction",
        "edge_gradient_sharpness_mm_inv",
        "threshold_transition_width_mm",
        "bright_ridge_fwhm_mm",
    )
    rows: list[dict[str, Any]] = []
    for result in results:
        old = by_model[str(result.summary["input_model"])]
        for local_key in keys:
            old_key = "useful_region_power_fraction" if local_key == "useful_power_fraction" else local_key
            actual = float(result.summary[local_key])
            expected = float(old[old_key])
            rows.append({
                "source": "MODE 2Y N=1024 stored endpoint",
                "case_id": result.case_id,
                "metric": local_key,
                "expected": expected,
                "actual": actual,
                "absolute_difference": abs(actual - expected),
                "reproduced": bool(np.isclose(actual, expected, rtol=2e-7, atol=2e-7)),
            })
    return tuple(rows)


def _build_cross_route_cases(bench: Mapping[str, Any], config: Phase2BConfig) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    data = bench["data"]
    v0 = bench["v0"]
    backward = bench["backward"]
    ideal = run_mode2n_dual_slm_qwp_route(data, v0)
    mild = run_mode2s_degraded_forward(
        data, v0, backward, mode2s_combined_cases()[0], fast_single_plane=True
    )
    offset = Mode2SPerturbation(
        label="phase2b_axicon_decentre_0p5mm",
        slm_aperture_clip=True,
        phase_levels=256,
        fill_factor=0.93,
        axicon_decentre_x_m=0.5e-3,
    )
    correction = Mode2SCorrection(mask_recentre_x_m=0.5e-3)
    degraded = run_mode2s_degraded_forward(data, v0, backward, offset, fast_single_plane=True)
    corrected = run_mode2s_degraded_forward(
        data, v0, backward, offset, correction=correction, fast_single_plane=True
    )
    entries = (
        ("target_analytic", "target / analytic", np.asarray(v0.reference_plane), data["target"], None),
        ("ideal_sequential", "ideal sequential", np.asarray(ideal.reference_plane), ideal.pre_axicon_field, 1.0),
        (
            "realistic_sequential",
            "realistic sequential",
            np.asarray(bench["realistic"].reference_plane),
            bench["realistic"].pre_axicon_field,
            float(bench["realistic"].slm_4f_report["first_order_efficiency"]),
        ),
        (
            "mild_realism",
            "mild realism",
            np.asarray(mild["reference_plane"]),
            mild["pre_axicon_field"],
            float(mild["iris"]["first_order_efficiency"]),
        ),
        (
            "degraded_axicon_0p5mm",
            "degraded 0.5 mm offset",
            np.asarray(degraded["reference_plane"]),
            degraded["pre_axicon_field"],
            float(degraded["iris"]["first_order_efficiency"]),
        ),
        (
            "corrected_axicon_0p5mm",
            "corrected digital recentre",
            np.asarray(corrected["reference_plane"]),
            corrected["pre_axicon_field"],
            float(corrected["iris"]["first_order_efficiency"]),
        ),
    )
    cases: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for route_id, label, plane, components, efficiency in entries:
        sas = _sas_zoom_plane(
            components[0],
            components[1],
            bench,
            z_m=PHASE2B_HEX_REFERENCE_M,
            pad_factor=config.sas_pad_factor,
        )
        cases.append({
            "route_id": route_id,
            "label": label,
            "native_plane": np.asarray(plane, dtype=np.float32),
            "sas_intensity": np.asarray(sas["intensity"], dtype=np.float32),
            "sas_grid": sas["grid"],
            "sas_metadata": {key: value for key, value in sas.items() if key not in {"intensity", "grid"}},
            "first_order_efficiency": efficiency,
        })
        metrics.append(_route_metrics(
            route_id,
            plane,
            bench,
            role=label,
            first_order_efficiency=efficiency,
        ))
    return tuple(cases), tuple(metrics)


def _highn_hex_hero(config: Phase2BConfig) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    if not config.highn_hero:
        return {"enabled": False}, ()
    cfg = _source_config(
        grid_n=config.hero_grid_n,
        z_planes=2,
        z_start_m=0.0,
        z_end_m=PHASE2B_HEX_REFERENCE_M,
    )
    bench = _bench_from_config(cfg)
    data = bench["data"]
    inputs = build_mode2y_input_fields(data)
    amplitude = np.asarray(data["A"], dtype=float)
    prefields: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    iris_reports: dict[str, Mapping[str, Any]] = {}
    for label, alpha in (
        ("continuous", inputs.continuous_alpha_rad),
        ("sector_averaged", inputs.averaged_alpha_rad),
    ):
        prefields[label], iris_reports[label] = _realistic_common_4f_field(
            amplitude,
            alpha,
            data,
            carrier_lpmm=6.25,
            iris_radius_frac=0.40,
        )
    native_planes: dict[str, np.ndarray] = {}
    sas_planes: dict[str, np.ndarray] = {}
    sas_grids: dict[str, Mapping[str, Any]] = {}
    sas_meta: dict[str, Mapping[str, Any]] = {}
    local_metrics: dict[str, Mapping[str, Any]] = {}
    useful_mask, _ = _fixed_useful_region(data["grid"], float(bench["v0"].ring_radius_m))
    for label, components in prefields.items():
        after, _ = _after_axicon(components, data)
        plane = _intensity_from_prepared(_prepare_projected_spectrum(after), PHASE2B_HEX_REFERENCE_M)
        native_planes[label] = np.asarray(plane, dtype=np.float32)
        local_metrics[label] = propagated_shape_metrics(
            plane,
            data["grid"],
            z_m=PHASE2B_HEX_REFERENCE_M,
            ring_radius_m=float(bench["v0"].ring_radius_m),
            useful_mask=useful_mask,
        ).__dict__
        zoom = _sas_zoom_plane(
            components[0],
            components[1],
            bench,
            z_m=PHASE2B_HEX_REFERENCE_M,
            pad_factor=config.sas_pad_factor,
        )
        sas_planes[label] = np.asarray(zoom["intensity"], dtype=np.float32)
        sas_grids[label] = zoom["grid"]
        sas_meta[label] = {key: value for key, value in zoom.items() if key not in {"intensity", "grid"}}
    stored = _read_csv(
        Path("outputs/figures/digital_twin/nathan_mode2z_orientation_interpolation/07_highN_confirmation/mode2z_highn_summary.csv")
    )
    stored_by_eta = {float(row["eta"]): row for row in stored}
    audits: list[dict[str, Any]] = []
    for label, eta in (("sector_averaged", 0.0), ("continuous", 1.0)):
        expected = stored_by_eta[eta]
        for metric in (
            "peak_intensity",
            "edge_gradient_sharpness_mm_inv",
            "threshold_transition_width_mm",
            "bright_ridge_fwhm_mm",
        ):
            actual = float(local_metrics[label][metric])
            target = float(expected[metric])
            audits.append({
                "source": "MODE 2Z-HN N=1536 stored endpoint",
                "case_id": f"H1_{label.upper()}",
                "metric": metric,
                "expected": target,
                "actual": actual,
                "absolute_difference": abs(actual - target),
                "reproduced": bool(np.isclose(actual, target, rtol=2e-7, atol=2e-7)),
            })
    strict_cont = evaluate_strict_hexagon_metrics(
        native_planes["continuous"],
        grid=data["grid"],
        v0_plane=bench["v0"].reference_plane,
        realistic_plane=native_planes["continuous"],
        v0_ring_radius_m=float(bench["v0"].ring_radius_m),
        useful_mask=useful_mask,
    )
    strict_avg = evaluate_strict_hexagon_metrics(
        native_planes["sector_averaged"],
        grid=data["grid"],
        v0_plane=bench["v0"].reference_plane,
        realistic_plane=native_planes["continuous"],
        v0_ring_radius_m=float(bench["v0"].ring_radius_m),
        useful_mask=useful_mask,
    )
    return {
        "enabled": True,
        "native_grid_n": int(cfg.grid_n),
        "native_dx_m": float(data["grid"]["dx"]),
        "ring_radius_m": float(bench["v0"].ring_radius_m),
        "native_planes": native_planes,
        "native_grid": data["grid"],
        "sas_planes": sas_planes,
        "sas_grids": sas_grids,
        "sas_metadata": sas_meta,
        "local_metrics": local_metrics,
        "strict_metrics": {"continuous": strict_cont, "sector_averaged": strict_avg},
        "iris_reports": iris_reports,
        "metrics_use_native_arrays": True,
        "sas_used_for_display_only": True,
        "display_interpolation_used_for_metrics": False,
    }, tuple(audits)


def build_hex_package(config: Phase2BConfig) -> Phase2BHexPackage:
    """Build N=1024 volumes, cross routes, and the N=1536 z60 hero pair."""

    config.validate()
    cfg = _source_config(
        grid_n=config.hex_grid_n,
        z_planes=3,
        z_start_m=59e-3,
        z_end_m=61e-3,
    )
    bench = _bench_from_config(cfg)
    data = bench["data"]
    inputs = build_mode2y_input_fields(data)
    study = Mode2YStudyConfig(
        grid_n=config.hex_grid_n,
        z_start_m=config.z_start_m,
        z_end_m=config.z_end_m,
        z_step_m=config.z_step_m,
        publication_quality=config.publication_quality,
    )
    amplitude = np.asarray(data["A"], dtype=float)
    prefields: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    reports: dict[str, Mapping[str, Any]] = {}
    for label, alpha in (
        ("continuous", inputs.continuous_alpha_rad),
        ("sector_averaged", inputs.averaged_alpha_rad),
    ):
        prefields[label], reports[label] = _realistic_common_4f_field(
            amplitude,
            alpha,
            data,
            carrier_lpmm=float(study.carrier_lpmm),
            iris_radius_frac=float(study.iris_radius_frac),
        )
    prepared: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for label, components in prefields.items():
        after, _ = _after_axicon(components, data)
        prepared[label] = _prepare_projected_spectrum(after)
    continuous = _stream_vector_case(
        "H1_CONTINUOUS",
        "continuous",
        prepared["continuous"],
        data,
        float(bench["v0"].ring_radius_m),
        config,
        first_order_efficiency=float(reports["continuous"]["first_order_efficiency"]),
    )
    averaged = _stream_vector_case(
        "H1_AVERAGED",
        "sector_averaged",
        prepared["sector_averaged"],
        data,
        float(bench["v0"].ring_radius_m),
        config,
        first_order_efficiency=float(reports["sector_averaged"]["first_order_efficiency"]),
    )
    canonical_sas = _sas_zoom_plane(
        bench["realistic"].pre_axicon_field[0],
        bench["realistic"].pre_axicon_field[1],
        bench,
        z_m=PHASE2B_HEX_REFERENCE_M,
        pad_factor=config.sas_pad_factor,
    )
    realistic = replace(
        continuous,
        case_id="H1_REALISTIC",
        family="canonical realistic continuous vector hexagonal field",
        sas_hero=canonical_sas,
        metadata={
            **dict(continuous.metadata),
            "canonical_alias_of": "H1_CONTINUOUS",
            "canonical_source": "REALISTIC_4F_HEXAGON_REFERENCE",
        },
    )
    cross_cases, cross_metrics = _build_cross_route_cases(bench, config)
    endpoint_rows = list(_mode2y_endpoint_audit((continuous, averaged)))
    highn, highn_audit = _highn_hex_hero(config)
    endpoint_rows.extend(highn_audit)
    return Phase2BHexPackage(
        realistic=realistic,
        continuous=continuous,
        averaged=averaged,
        cross_route_cases=cross_cases,
        cross_route_metrics=cross_metrics,
        highn_hero=highn,
        endpoint_audit=tuple(endpoint_rows),
        bench=bench,
    )


def phase2a_scalar_endpoint_audit(results: Mapping[str, Phase2BCaseResult]) -> tuple[dict[str, Any], ...]:
    """Compare regenerated z60 scalar metrics to the frozen PHASE 2A CSV."""

    stored = _read_csv(Path("outputs/validation/phase2a/canonical_case_summary.csv"))
    by_case = {
        row["case_id"]: row
        for row in stored
        if row["route_variant"] == "realistic_fixed_bench_route" and row["case_id"] in results
    }
    keys = (
        "beam_second_moment_radius_m",
        "dominant_off_axis_ring_radius_m",
        "central_intensity_ratio",
        "peak_intensity_au",
    )
    rows: list[dict[str, Any]] = []
    for case_id, result in results.items():
        for key in keys:
            expected = float(by_case[case_id][key])
            actual = float(result.summary[key])
            rows.append({
                "source": "PHASE 2A stored scalar endpoint",
                "case_id": case_id,
                "metric": key,
                "expected": expected,
                "actual": actual,
                "absolute_difference": abs(actual - expected),
                "reproduced": bool(np.isclose(actual, expected, rtol=2e-7, atol=2e-7)),
            })
    return tuple(rows)


__all__ = [
    "PHASE2B_3D_CASE_IDS",
    "PHASE2B_CASE_IDS",
    "PHASE2B_HEX_EARLY_M",
    "PHASE2B_HEX_LATE_M",
    "PHASE2B_HEX_REFERENCE_M",
    "Phase2BCaseResult",
    "Phase2BConfig",
    "Phase2BHexPackage",
    "build_hex_package",
    "build_scalar_cases",
    "phase2a_scalar_endpoint_audit",
    "phase2b_case_registry",
]
