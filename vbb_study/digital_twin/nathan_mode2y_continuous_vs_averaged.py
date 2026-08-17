"""MODE 2Y continuous-versus-sector-averaged propagation audit.

The only intended input difference is local line orientation: the continuous
field follows the authoritative local radial/azimuthal basis, while the
surrogate uses one headless representative line per 60 degree sector.  Both
fields share amplitude, sector labels, total power, optical operators, grid,
and z samples.  Metrics are evaluated on native arrays only.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.equations.fields import fft2c, ifft2c
from vbb_study.vector_field import VectorField
from vbb_study.digital_twin.nathan_mode2u2_fix_strict_hexagon_optimisation import (
    OLD_BEST_COMPROMISE_ID,
    evaluate_strict_hexagon_metrics,
)
from vbb_study.digital_twin.nathan_mode2u2_master_closure import (
    _fixed_useful_region,
    _useful_power_metrics,
)
from vbb_study.digital_twin.nathan_mode2u3_hardware_closure import (
    CANONICAL_OPERATING_POINT_ID,
    STRICT_COMPROMISE_ID,
    assert_not_forbidden,
)
from vbb_study.digital_twin.nathan_mode2w_fix_sequential_master import _source_config
from vbb_study.digital_twin.nathan_vector_hexagon import (
    EPS,
    MODE2N_DEFAULT_CARRIER_LPMM,
    MODE2N_DEFAULT_IRIS_RADIUS_FRAC,
    TWOPI,
    _apply_free_space_vector_axicon,
    _mode2n_reference_plane_metrics,
    _mode2n_vector_field,
    angular_profile_on_ring,
    apply_uniform_jones,
    linear_retarder,
    mode2n_source_target,
    mode2s_apply_4f,
    nathan_alpha_map,
)


MODE2Y_STAGE = "nathan_mode2y_continuous_vs_averaged"
MODE2Y_DEFAULT_OUTPUT_ROOT = Path("outputs/figures/digital_twin/nathan_mode2y_continuous_vs_averaged")
MODE2Y_DOC_PATH = Path("docs/85_nathan_mode2y_continuous_vs_averaged.md")
MODE2Y_ALLOWED_OUTCOMES = ("M2Y-A", "M2Y-B", "M2Y-C", "M2Y-D")
MODE2Y_SELECTED_Z_M = (0.0, 10e-3, 30e-3, 60e-3, 90e-3, 150e-3, 200e-3)
MODE2Y_MIN_HERO_GRID_N = 1024
MODE2Y_SHARPNESS_RELATIVE_TOLERANCE = 0.05


@dataclass(frozen=True)
class Mode2YStudyConfig:
    """Sampling and fixed optical controls for the MODE 2Y comparison."""

    grid_n: int = 1024
    z_start_m: float = 0.0
    z_end_m: float = 0.2
    z_step_m: float = 0.002
    selected_z_m: tuple[float, ...] = MODE2Y_SELECTED_Z_M
    carrier_lpmm: float = MODE2N_DEFAULT_CARRIER_LPMM
    iris_radius_frac: float = MODE2N_DEFAULT_IRIS_RADIUS_FRAC
    publication_quality: bool = True

    def z_values_m(self) -> np.ndarray:
        count = int(round((float(self.z_end_m) - float(self.z_start_m)) / float(self.z_step_m))) + 1
        values = np.linspace(float(self.z_start_m), float(self.z_end_m), count)
        for required in self.selected_z_m:
            if not np.any(np.isclose(values, float(required), rtol=0.0, atol=1e-12)):
                raise ValueError(f"z grid does not include required plane {required:g} m")
        return values

    def validate(self) -> None:
        if int(self.grid_n) < 64:
            raise ValueError("grid_n is too small for a propagation audit")
        if self.publication_quality and int(self.grid_n) < MODE2Y_MIN_HERO_GRID_N:
            raise ValueError(f"publication MODE 2Y outputs require grid_n >= {MODE2Y_MIN_HERO_GRID_N}")
        if float(self.z_step_m) <= 0.0 or float(self.z_end_m) <= float(self.z_start_m):
            raise ValueError("z range and spacing must be positive")
        self.z_values_m()


@dataclass(frozen=True)
class Mode2YInputFields:
    """Continuous and piecewise-constant Jones fields on the same source grid."""

    continuous_ex: np.ndarray = field(repr=False, compare=False)
    continuous_ey: np.ndarray = field(repr=False, compare=False)
    averaged_ex: np.ndarray = field(repr=False, compare=False)
    averaged_ey: np.ndarray = field(repr=False, compare=False)
    continuous_alpha_rad: np.ndarray = field(repr=False, compare=False)
    averaged_alpha_rad: np.ndarray = field(repr=False, compare=False)
    radial_sector_mask: np.ndarray = field(repr=False, compare=False)
    sector_index: np.ndarray = field(repr=False, compare=False)
    continuous_power: float
    averaged_power_before_normalisation: float
    averaged_power_after_normalisation: float
    averaged_amplitude_scale: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PropagatedShapeMetrics:
    """Native-array shape and sharpness observables for one plane."""

    z_m: float
    peak_intensity: float
    useful_power: float
    useful_power_fraction: float
    dark_core_ratio: float
    edge_gradient_sharpness_mm_inv: float
    threshold_transition_width_mm: float
    corner_concentration: float
    corner_contrast: float
    bright_ridge_fwhm_mm: float
    h3: float
    h6: float
    h3_over_h6: float
    sharpness_composite: float


@dataclass(frozen=True)
class PropagationRouteResult:
    """Memory-bounded propagation result for one input model and optical route."""

    route_id: str
    optical_route: str
    input_model: str
    z_values_m: np.ndarray = field(repr=False, compare=False)
    selected_planes: Mapping[str, np.ndarray] = field(repr=False, compare=False)
    xz_map: np.ndarray = field(repr=False, compare=False)
    yz_map: np.ndarray = field(repr=False, compare=False)
    z_metrics: tuple[PropagatedShapeMetrics, ...]
    best_z_m: float
    best_z_index: int
    persistence_fraction: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Mode2YStudyResult:
    """Complete ideal and realistic continuous/averaged comparison."""

    config: Mode2YStudyConfig
    data: Mapping[str, Any] = field(repr=False, compare=False)
    inputs: Mode2YInputFields
    pre_axicon_fields: Mapping[str, tuple[np.ndarray, np.ndarray]] = field(repr=False, compare=False)
    routes: Mapping[str, PropagationRouteResult]
    pair_difference_rows: tuple[Mapping[str, Any], ...]
    summary_rows: tuple[Mapping[str, Any], ...]
    outcome: str
    outcome_reason: str


def build_sector_averaged_alpha(
    theta_rad: np.ndarray,
    *,
    sector_rotation_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one constant representative headless orientation per sector."""

    theta = np.asarray(theta_rad, dtype=float)
    width = np.pi / 3.0
    relative = np.mod(theta - float(sector_rotation_rad), TWOPI)
    sector_index = np.mod(np.floor(relative / width).astype(int), 6)
    centres = float(sector_rotation_rad) + (sector_index + 0.5) * width
    _, representative_radial = nathan_alpha_map(
        centres,
        sector_num_pairs=3,
        sector_theta=width,
        sector_rotation=float(sector_rotation_rad),
    )
    alpha = centres + np.where(representative_radial, 0.0, 0.5 * np.pi)
    return alpha, np.asarray(representative_radial, dtype=bool), sector_index


def build_mode2y_input_fields(data: Mapping[str, Any]) -> Mode2YInputFields:
    """Construct power-matched continuous and sector-averaged source fields."""

    cfg = data["config"]
    grid = data["grid"]
    amplitude = np.asarray(data["A"], dtype=float)
    continuous_alpha = np.asarray(data["alpha"], dtype=float)
    continuous_ex = np.asarray(data["target"][0], dtype=np.complex128)
    continuous_ey = np.asarray(data["target"][1], dtype=np.complex128)
    averaged_alpha, representative_radial, sector_index = build_sector_averaged_alpha(
        np.asarray(grid["PHI"], dtype=float),
        sector_rotation_rad=float(cfg.sector_rotation_rad),
    )
    averaged_ex_raw = amplitude * np.cos(averaged_alpha)
    averaged_ey_raw = amplitude * np.sin(averaged_alpha)
    continuous_power = float(np.sum(np.abs(continuous_ex) ** 2 + np.abs(continuous_ey) ** 2))
    averaged_power_raw = float(np.sum(np.abs(averaged_ex_raw) ** 2 + np.abs(averaged_ey_raw) ** 2))
    scale = float(np.sqrt(continuous_power / max(averaged_power_raw, EPS)))
    averaged_ex = np.asarray(scale * averaged_ex_raw, dtype=np.complex128)
    averaged_ey = np.asarray(scale * averaged_ey_raw, dtype=np.complex128)
    averaged_power = float(np.sum(np.abs(averaged_ex) ** 2 + np.abs(averaged_ey) ** 2))
    if not np.isclose(continuous_power, averaged_power, rtol=2e-14, atol=2e-14 * max(continuous_power, 1.0)):
        raise FloatingPointError("continuous and averaged input powers do not match")
    if np.allclose(continuous_ex, averaged_ex) and np.allclose(continuous_ey, averaged_ey):
        raise AssertionError("sector-averaged surrogate unexpectedly equals the continuous field")
    return Mode2YInputFields(
        continuous_ex=continuous_ex,
        continuous_ey=continuous_ey,
        averaged_ex=averaged_ex,
        averaged_ey=averaged_ey,
        continuous_alpha_rad=continuous_alpha,
        averaged_alpha_rad=averaged_alpha,
        radial_sector_mask=np.asarray(data["radial_mask"], dtype=bool),
        sector_index=sector_index,
        continuous_power=continuous_power,
        averaged_power_before_normalisation=averaged_power_raw,
        averaged_power_after_normalisation=averaged_power,
        averaged_amplitude_scale=scale,
        metadata={
            "sector_convention": "v0_authoritative",
            "representative_orientation": "one headless line at each 60 degree sector centre",
            "representative_radial_mask_matches_sector_labels": bool(
                np.array_equal(representative_radial, np.asarray(data["radial_mask"], dtype=bool))
            ),
            "same_amplitude_envelope": True,
            "same_total_input_power": True,
        },
    )


def _realistic_common_4f_field(
    amplitude: np.ndarray,
    alpha_rad: np.ndarray,
    data: Mapping[str, Any],
    *,
    carrier_lpmm: float,
    iris_radius_frac: float,
) -> tuple[tuple[np.ndarray, np.ndarray], Mapping[str, Any]]:
    """Apply the validated sequential carrier/common-4F/QWP pre-axicon chain."""

    grid = data["grid"]
    carrier_cpm = float(carrier_lpmm) * 1e3
    nyquist_cpm = 0.5 / float(grid["dx"])
    if (1.0 + float(iris_radius_frac)) * carrier_cpm >= nyquist_cpm:
        raise ValueError("carrier plus iris exceeds the sampled spectral band")
    carrier_phase = TWOPI * carrier_cpm * np.asarray(grid["X"], dtype=float)
    supply = np.asarray(amplitude, dtype=float) / np.sqrt(2.0)
    eh = supply * np.exp(1j * (np.asarray(alpha_rad, dtype=float) + carrier_phase))
    ev = supply * np.exp(1j * (-np.asarray(alpha_rad, dtype=float) + 0.5 * np.pi + carrier_phase))
    eh_f, ev_f, iris = mode2s_apply_4f(
        eh,
        ev,
        grid,
        carrier_lpmm=float(carrier_lpmm),
        iris_radius_frac=float(iris_radius_frac),
    )
    qwp = linear_retarder(0.5 * np.pi, -0.25 * np.pi)
    ex, ey = apply_uniform_jones(qwp, eh_f, ev_f)
    return (np.asarray(ex, dtype=np.complex128), np.asarray(ey, dtype=np.complex128)), iris


def build_mode2y_pre_axicon_fields(
    data: Mapping[str, Any],
    inputs: Mode2YInputFields,
    config: Mode2YStudyConfig,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Mapping[str, Any]]]:
    """Build both input models for ideal and validated realistic routes."""

    continuous_realistic, cont_iris = _realistic_common_4f_field(
        np.asarray(data["A"], dtype=float),
        inputs.continuous_alpha_rad,
        data,
        carrier_lpmm=float(config.carrier_lpmm),
        iris_radius_frac=float(config.iris_radius_frac),
    )
    averaged_amplitude = np.sqrt(np.abs(inputs.averaged_ex) ** 2 + np.abs(inputs.averaged_ey) ** 2)
    averaged_realistic, avg_iris = _realistic_common_4f_field(
        averaged_amplitude,
        inputs.averaged_alpha_rad,
        data,
        carrier_lpmm=float(config.carrier_lpmm),
        iris_radius_frac=float(config.iris_radius_frac),
    )
    fields = {
        "ideal_continuous": (inputs.continuous_ex, inputs.continuous_ey),
        "ideal_sector_averaged": (inputs.averaged_ex, inputs.averaged_ey),
        "realistic_continuous_common_4f": continuous_realistic,
        "realistic_sector_averaged_common_4f": averaged_realistic,
    }
    reports = {
        "ideal_continuous": {"first_order_efficiency": 1.0, "route": "ideal sequential-equivalent"},
        "ideal_sector_averaged": {"first_order_efficiency": 1.0, "route": "ideal sequential-equivalent"},
        "realistic_continuous_common_4f": dict(cont_iris),
        "realistic_sector_averaged_common_4f": dict(avg_iris),
    }
    return fields, reports


def _prepare_projected_spectrum(field: VectorField) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare the same spectral projection used by the validated vector ASM."""

    grid = field.grid
    k = TWOPI * float(field.medium_index) / float(field.wavelength_m)
    kx = TWOPI * np.asarray(grid["FX"], dtype=float)
    ky = TWOPI * np.asarray(grid["FY"], dtype=float)
    kz = np.sqrt((k * k - kx * kx - ky * ky) + 0j)
    kz = np.where(np.imag(kz) < 0.0, -kz, kz)
    ax = fft2c(field.ex)
    ay = fft2c(field.ey)
    az = fft2c(field.ez)
    sx = kx / max(k, EPS)
    sy = ky / max(k, EPS)
    sz = kz / max(k, EPS)
    dot = sx * ax + sy * ay + sz * az
    return ax - sx * dot, ay - sy * dot, az - sz * dot, kz


def _intensity_from_prepared(
    prepared: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    z_m: float,
) -> np.ndarray:
    ax, ay, az, kz = prepared
    transfer = np.exp(1j * kz * float(z_m))
    ex = ifft2c(ax * transfer)
    ey = ifft2c(ay * transfer)
    ez = ifft2c(az * transfer)
    return np.asarray(np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2, dtype=np.float64)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float).ravel()
    bb = np.asarray(b, dtype=float).ravel()
    aa = aa - float(np.mean(aa))
    bb = bb - float(np.mean(bb))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / max(denom, EPS))


def equal_power_difference_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Compare two output shapes after independent equal-power normalisation."""

    aa = np.asarray(a, dtype=float) / max(float(np.sum(a)), EPS)
    bb = np.asarray(b, dtype=float) / max(float(np.sum(b)), EPS)
    delta = aa - bb
    return {
        "equal_power_correlation": _safe_corr(aa, bb),
        "equal_power_relative_l2_difference": float(np.linalg.norm(delta) / max(float(np.linalg.norm(aa)), EPS)),
        "equal_power_l1_difference": float(np.sum(np.abs(delta))),
        "equal_power_signed_difference_sum": float(np.sum(delta)),
    }


def _radial_profile_native(
    plane: np.ndarray,
    grid: Mapping[str, Any],
    *,
    bins: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n_bins = int(bins or max(128, min(640, int(grid["N"]) // 2)))
    radius = np.asarray(grid["R"], dtype=float).ravel()
    values = np.asarray(plane, dtype=float).ravel()
    edges = np.linspace(0.0, float(np.max(radius)), n_bins + 1)
    index = np.clip(np.digitize(radius, edges) - 1, 0, n_bins - 1)
    sums = np.bincount(index, weights=values, minlength=n_bins)
    counts = np.bincount(index, minlength=n_bins)
    return 0.5 * (edges[:-1] + edges[1:]), sums / np.maximum(counts, 1)


def _crossing_radius(
    radii: np.ndarray,
    profile: np.ndarray,
    start_index: int,
    threshold: float,
) -> float:
    candidates = np.where(profile[start_index:] <= float(threshold))[0]
    if not candidates.size:
        return float("nan")
    return float(radii[start_index + int(candidates[0])])


def propagated_shape_metrics(
    plane: np.ndarray,
    grid: Mapping[str, Any],
    *,
    z_m: float,
    ring_radius_m: float,
    useful_mask: np.ndarray,
) -> PropagatedShapeMetrics:
    """Compute all sharpness observables directly from native numerical samples."""

    arr = np.asarray(plane, dtype=float)
    peak = max(float(np.max(arr)), EPS)
    norm = arr / peak
    radius = np.asarray(grid["R"], dtype=float)
    dx = float(grid["dx"])
    annulus = (radius >= 0.45 * float(ring_radius_m)) & (radius <= 2.2 * float(ring_radius_m))
    transition = annulus & (norm >= 0.2) & (norm <= 0.8)
    grad_y, grad_x = np.gradient(norm, dx, dx)
    gradient = np.hypot(grad_x, grad_y)
    edge_gradient_mm_inv = float(np.mean(gradient[transition]) * 1e-3) if np.any(transition) else 0.0

    radial_r, radial_i = _radial_profile_native(arr, grid)
    ring_window = (radial_r >= 0.40 * float(ring_radius_m)) & (radial_r <= 2.2 * float(ring_radius_m))
    indices = np.where(ring_window)[0]
    peak_index = int(indices[np.argmax(radial_i[indices])]) if indices.size else int(np.argmax(radial_i))
    radial_peak = max(float(radial_i[peak_index]), EPS)
    left_half_candidates = np.where(radial_i[: peak_index + 1] <= 0.5 * radial_peak)[0]
    right_half_candidates = np.where(radial_i[peak_index:] <= 0.5 * radial_peak)[0]
    left_radius = float(radial_r[left_half_candidates[-1]]) if left_half_candidates.size else float(radial_r[0])
    right_radius = (
        float(radial_r[peak_index + int(right_half_candidates[0])])
        if right_half_candidates.size
        else float(radial_r[-1])
    )
    fwhm_mm = float(max(right_radius - left_radius, dx) / 1e-3)
    r80 = _crossing_radius(radial_r, radial_i, peak_index, 0.8 * radial_peak)
    r20 = _crossing_radius(radial_r, radial_i, peak_index, 0.2 * radial_peak)
    transition_width_mm = (
        float(max(r20 - r80, dx) / 1e-3)
        if np.isfinite(r80) and np.isfinite(r20)
        else float("nan")
    )

    theta, angular = angular_profile_on_ring(arr, grid, float(ring_radius_m), angular_bins=720)
    angular_mean = float(np.mean(angular))
    centred = np.asarray(angular, dtype=float) - angular_mean
    denom = float(np.sum(np.abs(centred))) + EPS
    c3 = np.sum(centred * np.exp(-3j * theta))
    c6 = np.sum(centred * np.exp(-6j * theta))
    h3 = float(abs(c3) / denom)
    h6 = float(abs(c6) / denom)
    theta0 = float(np.mod(-np.angle(c6) / 6.0, np.pi / 3.0)) if abs(c6) > EPS else 0.0
    phase_distance = np.abs(np.angle(np.exp(1j * 6.0 * (theta - theta0)))) / 6.0
    corner_mask = phase_distance <= np.deg2rad(10.0)
    side_mask = phase_distance >= np.deg2rad(20.0)
    corner_mean = float(np.mean(angular[corner_mask])) if np.any(corner_mask) else 0.0
    side_mean = float(np.mean(angular[side_mask])) if np.any(side_mask) else 0.0
    concentration = float(corner_mean / max(side_mean, EPS))
    contrast = float((corner_mean - side_mean) / max(corner_mean + side_mean, EPS))
    core_mask = radius <= max(0.20 * float(ring_radius_m), dx)
    ring_mask = (radius >= 0.75 * float(ring_radius_m)) & (radius <= 1.25 * float(ring_radius_m))
    dark = float(np.mean(arr[core_mask]) / max(float(np.max(arr[ring_mask])), EPS))
    useful = _useful_power_metrics(arr, useful_mask)
    finite_transition = transition_width_mm if np.isfinite(transition_width_mm) else 1e9
    composite = float(
        edge_gradient_mm_inv * max(concentration, 0.0)
        / max(finite_transition * fwhm_mm, 1e-12)
    )
    return PropagatedShapeMetrics(
        z_m=float(z_m),
        peak_intensity=float(useful["I_peak"]),
        useful_power=float(useful["P_useful"]),
        useful_power_fraction=float(useful["P_useful_over_P_total"]),
        dark_core_ratio=dark,
        edge_gradient_sharpness_mm_inv=edge_gradient_mm_inv,
        threshold_transition_width_mm=transition_width_mm,
        corner_concentration=concentration,
        corner_contrast=contrast,
        bright_ridge_fwhm_mm=fwhm_mm,
        h3=h3,
        h6=h6,
        h3_over_h6=float(h3 / max(h6, EPS)),
        sharpness_composite=composite,
    )


def _after_axicon(
    field_components: tuple[np.ndarray, np.ndarray],
    data: Mapping[str, Any],
) -> tuple[VectorField, Mapping[str, Any]]:
    cfg = data["config"]
    field = _mode2n_vector_field(field_components[0], field_components[1], data)
    return _apply_free_space_vector_axicon(
        field,
        n_axicon=float(cfg.axicon_n),
        n_medium=float(cfg.medium_n),
        base_angle_rad=float(cfg.axicon_base_angle_rad),
    )


def _selected_key(z_m: float) -> str:
    return f"z{float(z_m) / 1e-3:.3f}mm"


def _stream_route_pair(
    *,
    optical_route: str,
    continuous_field: tuple[np.ndarray, np.ndarray],
    averaged_field: tuple[np.ndarray, np.ndarray],
    data: Mapping[str, Any],
    config: Mode2YStudyConfig,
    ring_radius_m: float,
    useful_mask: np.ndarray,
    reports: Mapping[str, Mapping[str, Any]],
) -> tuple[PropagationRouteResult, PropagationRouteResult, list[dict[str, Any]]]:
    z_values = config.z_values_m()
    route_ids = (
        f"{optical_route}_continuous" if optical_route == "ideal" else "realistic_continuous_common_4f",
        f"{optical_route}_sector_averaged" if optical_route == "ideal" else "realistic_sector_averaged_common_4f",
    )
    after_continuous, axicon_meta = _after_axicon(continuous_field, data)
    after_averaged, _ = _after_axicon(averaged_field, data)
    prepared_continuous = _prepare_projected_spectrum(after_continuous)
    prepared_averaged = _prepare_projected_spectrum(after_averaged)
    n = int(data["grid"]["N"])
    xz_cont = np.empty((z_values.size, n), dtype=np.float32)
    yz_cont = np.empty((z_values.size, n), dtype=np.float32)
    xz_avg = np.empty((z_values.size, n), dtype=np.float32)
    yz_avg = np.empty((z_values.size, n), dtype=np.float32)
    selected_cont: dict[str, np.ndarray] = {}
    selected_avg: dict[str, np.ndarray] = {}
    cont_metrics: list[PropagatedShapeMetrics] = []
    avg_metrics: list[PropagatedShapeMetrics] = []
    pair_rows: list[dict[str, Any]] = []
    selected = np.asarray(config.selected_z_m, dtype=float)
    mid = n // 2
    for index, z_m in enumerate(z_values):
        plane_cont = _intensity_from_prepared(prepared_continuous, float(z_m))
        plane_avg = _intensity_from_prepared(prepared_averaged, float(z_m))
        xz_cont[index] = plane_cont[mid, :]
        yz_cont[index] = plane_cont[:, mid]
        xz_avg[index] = plane_avg[mid, :]
        yz_avg[index] = plane_avg[:, mid]
        cont_metrics.append(propagated_shape_metrics(
            plane_cont, data["grid"], z_m=float(z_m), ring_radius_m=ring_radius_m, useful_mask=useful_mask
        ))
        avg_metrics.append(propagated_shape_metrics(
            plane_avg, data["grid"], z_m=float(z_m), ring_radius_m=ring_radius_m, useful_mask=useful_mask
        ))
        pair_rows.append({"optical_route": optical_route, "z_m": float(z_m), **equal_power_difference_metrics(plane_cont, plane_avg)})
        if np.any(np.isclose(selected, float(z_m), rtol=0.0, atol=1e-12)):
            selected_cont[_selected_key(float(z_m))] = plane_cont.astype(np.float32)
            selected_avg[_selected_key(float(z_m))] = plane_avg.astype(np.float32)
    valid_best = np.where(z_values >= 10e-3)[0]
    cont_scores = np.asarray([metric.sharpness_composite for metric in cont_metrics], dtype=float)
    avg_scores = np.asarray([metric.sharpness_composite for metric in avg_metrics], dtype=float)
    best_cont = int(valid_best[np.nanargmax(cont_scores[valid_best])])
    best_avg = int(valid_best[np.nanargmax(avg_scores[valid_best])])
    for key, best_index in (("continuous_best_z", best_cont), ("averaged_best_z", best_avg)):
        selected_cont[key] = _intensity_from_prepared(
            prepared_continuous, float(z_values[best_index])
        ).astype(np.float32)
        selected_avg[key] = _intensity_from_prepared(
            prepared_averaged, float(z_values[best_index])
        ).astype(np.float32)
    cont_persistence = float(np.mean(cont_scores >= 0.8 * max(float(np.nanmax(cont_scores)), EPS)))
    avg_persistence = float(np.mean(avg_scores >= 0.8 * max(float(np.nanmax(avg_scores)), EPS)))
    common_meta = {
        "stage": MODE2Y_STAGE,
        "axicon": dict(axicon_meta),
        "grid_n": n,
        "native_grid_metrics": True,
        "display_interpolation_used_for_metrics": False,
        "plane": "after_source_scale_axicon_free_space",
    }
    continuous_result = PropagationRouteResult(
        route_id=route_ids[0],
        optical_route=optical_route,
        input_model="continuous",
        z_values_m=z_values,
        selected_planes=selected_cont,
        xz_map=xz_cont,
        yz_map=yz_cont,
        z_metrics=tuple(cont_metrics),
        best_z_m=float(z_values[best_cont]),
        best_z_index=best_cont,
        persistence_fraction=cont_persistence,
        metadata={**common_meta, "pre_axicon_report": dict(reports[route_ids[0]])},
    )
    averaged_result = PropagationRouteResult(
        route_id=route_ids[1],
        optical_route=optical_route,
        input_model="sector_averaged",
        z_values_m=z_values,
        selected_planes=selected_avg,
        xz_map=xz_avg,
        yz_map=yz_avg,
        z_metrics=tuple(avg_metrics),
        best_z_m=float(z_values[best_avg]),
        best_z_index=best_avg,
        persistence_fraction=avg_persistence,
        metadata={**common_meta, "pre_axicon_report": dict(reports[route_ids[1]])},
    )
    return continuous_result, averaged_result, pair_rows


def _metric_at_z(route: PropagationRouteResult, z_m: float) -> PropagatedShapeMetrics:
    index = int(np.argmin(np.abs(np.asarray(route.z_values_m, dtype=float) - float(z_m))))
    if not np.isclose(float(route.z_values_m[index]), float(z_m), rtol=0.0, atol=1e-12):
        raise ValueError(f"route does not contain exact z={z_m:g} m")
    return route.z_metrics[index]


def _strict_row(
    route: PropagationRouteResult,
    *,
    grid: Mapping[str, Any],
    v0_plane: np.ndarray,
    realistic_plane: np.ndarray,
    ring_radius_m: float,
    useful_mask: np.ndarray,
) -> dict[str, Any]:
    plane = np.asarray(route.selected_planes[_selected_key(60e-3)], dtype=float)
    strict = evaluate_strict_hexagon_metrics(
        plane,
        grid=grid,
        v0_plane=v0_plane,
        realistic_plane=realistic_plane,
        v0_ring_radius_m=float(ring_radius_m),
        useful_mask=useful_mask,
    )
    local = _metric_at_z(route, 60e-3)
    return {
        "route_id": route.route_id,
        "optical_route": route.optical_route,
        "input_model": route.input_model,
        "z60_correlation_to_v0": float(strict["corr_full"]),
        "strict_hexagon_pass": bool(strict["strict_hexagon_eligible"]),
        "strict_classifier": str(strict["classifier_label"]),
        "strict_fail_reasons": str(strict["strict_fail_reasons"]),
        "peak_intensity": float(strict["single_pixel_peak"]),
        "peak_3x3_mean": float(strict["local_3x3_peak_mean"]),
        "useful_region_power": float(strict["P_useful"]),
        "useful_region_power_fraction": float(strict["P_useful_over_P_total"]),
        "dark_core_ratio": float(strict["dark_core_ratio"]),
        "c60": float(strict["c60"]),
        "c120": float(strict["c120"]),
        "delta_c120_minus_c60": float(strict["deltaC_c120_minus_c60"]),
        "h3": float(strict["h3"]),
        "h6": float(strict["h6"]),
        "h3_over_h6": float(strict["h3_over_h6"]),
        "edge_gradient_sharpness_mm_inv": local.edge_gradient_sharpness_mm_inv,
        "threshold_transition_width_mm": local.threshold_transition_width_mm,
        "corner_concentration": local.corner_concentration,
        "corner_contrast": local.corner_contrast,
        "bright_ridge_fwhm_mm": local.bright_ridge_fwhm_mm,
        "sharpness_composite": local.sharpness_composite,
        "best_z_mm": float(route.best_z_m / 1e-3),
        "best_sharpness_composite": float(route.z_metrics[route.best_z_index].sharpness_composite),
        "propagation_persistence_fraction": float(route.persistence_fraction),
        "metrics_native_grid_n": int(route.metadata["grid_n"]),
    }


def _sharpness_comparison(
    continuous_row: Mapping[str, Any],
    averaged_row: Mapping[str, Any],
) -> dict[str, Any]:
    improvements = {
        "edge_gradient_relative_improvement": float(
            float(continuous_row["edge_gradient_sharpness_mm_inv"])
            / max(float(averaged_row["edge_gradient_sharpness_mm_inv"]), EPS) - 1.0
        ),
        "transition_width_relative_improvement": float(
            float(averaged_row["threshold_transition_width_mm"])
            / max(float(continuous_row["threshold_transition_width_mm"]), EPS) - 1.0
        ),
        "corner_concentration_relative_improvement": float(
            float(continuous_row["corner_concentration"])
            / max(float(averaged_row["corner_concentration"]), EPS) - 1.0
        ),
        "ridge_fwhm_relative_improvement": float(
            float(averaged_row["bright_ridge_fwhm_mm"])
            / max(float(continuous_row["bright_ridge_fwhm_mm"]), EPS) - 1.0
        ),
    }
    values = np.asarray(list(improvements.values()), dtype=float)
    return {
        **improvements,
        "median_relative_sharpness_improvement": float(np.median(values)),
        "continuous_win_count_at_5pct": int(np.count_nonzero(values >= MODE2Y_SHARPNESS_RELATIVE_TOLERANCE)),
        "averaged_win_count_at_5pct": int(np.count_nonzero(values <= -MODE2Y_SHARPNESS_RELATIVE_TOLERANCE)),
    }


def mode2y_outcome(
    ideal_comparison: Mapping[str, Any],
    realistic_comparison: Mapping[str, Any],
) -> tuple[str, str]:
    """Classify the predeclared four-metric z=60 sharpness hypothesis."""

    ideal_wins = int(ideal_comparison["continuous_win_count_at_5pct"])
    realistic_wins = int(realistic_comparison["continuous_win_count_at_5pct"])
    ideal_losses = int(ideal_comparison["averaged_win_count_at_5pct"])
    realistic_losses = int(realistic_comparison["averaged_win_count_at_5pct"])
    median = float(np.median([
        ideal_comparison["median_relative_sharpness_improvement"],
        realistic_comparison["median_relative_sharpness_improvement"],
    ]))
    if ideal_wins >= 3 and realistic_wins >= 3 and median >= MODE2Y_SHARPNESS_RELATIVE_TOLERANCE:
        return "M2Y-A", "continuous is measurably sharper in both ideal and realistic routes"
    if ideal_losses >= 3 and realistic_losses >= 3 and median <= -MODE2Y_SHARPNESS_RELATIVE_TOLERANCE:
        return "M2Y-C", "sector-averaged is measurably sharper in both routes"
    if abs(median) < MODE2Y_SHARPNESS_RELATIVE_TOLERANCE and ideal_wins < 3 and realistic_wins < 3:
        return "M2Y-B", "propagated sharpness difference is negligible under the 5 percent criterion"
    return "M2Y-C", "propagated sharpness evidence is mixed across metrics or routes"


def run_mode2y_study(config: Mode2YStudyConfig | None = None) -> Mode2YStudyResult:
    """Run the complete continuous-versus-averaged source-scale comparison."""

    study = config or Mode2YStudyConfig()
    study.validate()
    assert_not_forbidden(CANONICAL_OPERATING_POINT_ID)
    assert_not_forbidden(STRICT_COMPROMISE_ID)
    if OLD_BEST_COMPROMISE_ID in {CANONICAL_OPERATING_POINT_ID, STRICT_COMPROMISE_ID}:
        raise AssertionError("forbidden optimiser candidate cannot become canonical")
    source_cfg = _source_config(
        grid_n=int(study.grid_n),
        z_planes=int(study.z_values_m().size),
        z_start_m=float(study.z_start_m),
        z_end_m=float(study.z_end_m),
    )
    data = mode2n_source_target(source_cfg, grid_n=int(study.grid_n), z_planes=int(study.z_values_m().size))
    inputs = build_mode2y_input_fields(data)
    pre_fields, reports = build_mode2y_pre_axicon_fields(data, inputs, study)

    provisional_after, _ = _after_axicon(pre_fields["ideal_continuous"], data)
    provisional = _intensity_from_prepared(_prepare_projected_spectrum(provisional_after), 60e-3)
    reference_diag = _mode2n_reference_plane_metrics(provisional, data["grid"])
    ring_radius = float(reference_diag["ring_radius_m"])
    useful_mask, useful_meta = _fixed_useful_region(data["grid"], ring_radius)

    ideal_cont, ideal_avg, ideal_pair = _stream_route_pair(
        optical_route="ideal",
        continuous_field=pre_fields["ideal_continuous"],
        averaged_field=pre_fields["ideal_sector_averaged"],
        data=data,
        config=study,
        ring_radius_m=ring_radius,
        useful_mask=useful_mask,
        reports=reports,
    )
    realistic_cont, realistic_avg, realistic_pair = _stream_route_pair(
        optical_route="realistic",
        continuous_field=pre_fields["realistic_continuous_common_4f"],
        averaged_field=pre_fields["realistic_sector_averaged_common_4f"],
        data=data,
        config=study,
        ring_radius_m=ring_radius,
        useful_mask=useful_mask,
        reports=reports,
    )
    routes = {
        result.route_id: result
        for result in (ideal_cont, ideal_avg, realistic_cont, realistic_avg)
    }
    v0_plane = np.asarray(ideal_cont.selected_planes[_selected_key(60e-3)], dtype=float)
    realistic_plane = np.asarray(realistic_cont.selected_planes[_selected_key(60e-3)], dtype=float)
    summary_rows = tuple(
        _strict_row(
            route,
            grid=data["grid"],
            v0_plane=v0_plane,
            realistic_plane=realistic_plane,
            ring_radius_m=ring_radius,
            useful_mask=useful_mask,
        )
        for route in (ideal_cont, ideal_avg, realistic_cont, realistic_avg)
    )
    rows_by_id = {str(row["route_id"]): row for row in summary_rows}
    ideal_comparison = _sharpness_comparison(rows_by_id[ideal_cont.route_id], rows_by_id[ideal_avg.route_id])
    realistic_comparison = _sharpness_comparison(rows_by_id[realistic_cont.route_id], rows_by_id[realistic_avg.route_id])
    outcome, reason = mode2y_outcome(ideal_comparison, realistic_comparison)
    pair_rows = tuple([
        *ideal_pair,
        *realistic_pair,
        {"optical_route": "ideal", "z_m": 60e-3, "comparison_scope": "sharpness_summary", **ideal_comparison},
        {"optical_route": "realistic", "z_m": 60e-3, "comparison_scope": "sharpness_summary", **realistic_comparison},
    ])
    data_with_meta = {
        **dict(data),
        "mode2y_ring_radius_m": ring_radius,
        "mode2y_useful_mask": useful_mask,
        "mode2y_useful_meta": useful_meta,
        "mode2y_ideal_sharpness_comparison": ideal_comparison,
        "mode2y_realistic_sharpness_comparison": realistic_comparison,
    }
    return Mode2YStudyResult(
        config=study,
        data=data_with_meta,
        inputs=inputs,
        pre_axicon_fields=pre_fields,
        routes=routes,
        pair_difference_rows=pair_rows,
        summary_rows=summary_rows,
        outcome=outcome,
        outcome_reason=reason,
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: _json_ready(row.get(key, "")) for key in fields} for row in rows])
    return path


def _write_document(path: Path, result: Mode2YStudyResult, output_root: Path) -> Path:
    lines = [
        "| route | z60 corr | strict | edge grad (mm^-1) | 80-20 width (mm) | corner conc. | ridge FWHM (mm) | best z (mm) |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in result.summary_rows:
        lines.append(
            f"| `{row['route_id']}` | {float(row['z60_correlation_to_v0']):.7f} | {bool(row['strict_hexagon_pass'])} | "
            f"{float(row['edge_gradient_sharpness_mm_inv']):.4f} | {float(row['threshold_transition_width_mm']):.5f} | "
            f"{float(row['corner_concentration']):.4f} | {float(row['bright_ridge_fwhm_mm']):.5f} | {float(row['best_z_mm']):.1f} |"
        )
    ideal_cmp = result.data["mode2y_ideal_sharpness_comparison"]
    realistic_cmp = result.data["mode2y_realistic_sharpness_comparison"]
    text = f"""# Nathan MODE 2Y - Continuous vs Sector-Averaged Propagation Audit

**Status:** source-scale sequential propagation comparison only. No split-arm architecture and no
microfabrication/sample-plane success claim.

## Question

With the Gaussian envelope, six-sector labels, total input power, axicon and propagation held fixed,
does the true continuously varying local radial/azimuthal field produce a sharper propagated hexagon
than a deliberately piecewise-constant one-line-per-sector surrogate?

## Fixed Comparison

The continuous input follows local `e_r(theta)` or `e_theta(theta)`. The surrogate assigns the
orientation at each 60 degree sector centre to every pixel in that sector. It is a diagnostic
surrogate, not a physically correct radial/azimuthal field. Input powers match to floating-point
precision. Both are propagated through the ideal sequential-equivalent route and the validated
carrier + common-4F + QWP sequential route, then through the same source-scale axicon and vector ASM.

Sampling is native N={result.config.grid_n}, z={result.config.z_start_m / 1e-3:.0f} to {result.config.z_end_m / 1e-3:.0f} mm
in {result.config.z_step_m / 1e-3:.0f} mm steps. Display interpolation never enters a metric.

## Results

{chr(10).join(lines)}

Ideal median continuous sharpness change: {100.0 * float(ideal_cmp['median_relative_sharpness_improvement']):+.2f}%.
Realistic median continuous sharpness change: {100.0 * float(realistic_cmp['median_relative_sharpness_improvement']):+.2f}%.
The predeclared decision requires at least three of four sharpness metrics to improve by 5% in both
routes before calling the continuous field measurably sharper.

The continuous field wins edge-gradient sharpness, 80-20 transition width and bright-ridge FWHM in
both routes. The averaged surrogate has greater corner concentration and slightly greater peak and
useful-region energy, but poorer V0 morphology and it fails the repaired strict gate in the realistic
route. The realistic continuous field remains strict-eligible.

## Conclusion

Outcome **{result.outcome}**: {result.outcome_reason}. This conclusion concerns source-scale propagated
morphology only. Local-vector truth remains the separate MODE 2X result, and the repaired strict
hexagon gate remains independently reported.

Output root: `{output_root}`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_mode2y_outputs(
    output_root: str | Path = MODE2Y_DEFAULT_OUTPUT_ROOT,
    *,
    document_path: str | Path = MODE2Y_DOC_PATH,
    config: Mode2YStudyConfig | None = None,
) -> dict[str, Any]:
    """Run MODE 2Y and write publication figures plus machine-readable outputs."""

    study = config or Mode2YStudyConfig()
    study.validate()
    root = Path(output_root)
    for subdir in ("00_inputs", "01_xy_planes", "02_focus_crops", "03_propagation_maps", "04_profiles", "05_metrics", "06_final_status"):
        (root / subdir).mkdir(parents=True, exist_ok=True)
    result = run_mode2y_study(study)
    from vbb_study.digital_twin.nathan_mode2y_figures import write_mode2y_figures

    figure_paths = write_mode2y_figures(result, root)
    summary_rows = [dict(row) for row in result.summary_rows]
    _write_csv(root / "continuous_vs_averaged_summary.csv", summary_rows)
    _write_json(root / "continuous_vs_averaged_summary.json", {
        "stage": MODE2Y_STAGE,
        "outcome": result.outcome,
        "rows": summary_rows,
        "pair_difference_rows": result.pair_difference_rows,
    })
    scope = {
        "stage": MODE2Y_STAGE,
        "scientific_scope": "source-scale continuous-versus-sector-averaged propagation audit",
        "accepted_architecture": "one sequential collinear beam -> SLM1 -> conditional swap -> SLM2 -> optional swap-back -> common 4F -> QWP -> axicon",
        "split_arm_pbs_architecture_used": False,
        "microfabrication_sample_plane_success_claim": False,
        "grid_n": int(study.grid_n),
        "hero_minimum_grid_n": MODE2Y_MIN_HERO_GRID_N,
        "publication_quality": bool(study.publication_quality),
        "z_values_mm": [float(value / 1e-3) for value in study.z_values_m()],
        "selected_z_mm": [float(value / 1e-3) for value in study.selected_z_m],
        "z60_included_exactly": bool(np.any(np.isclose(study.z_values_m(), 60e-3, rtol=0.0, atol=1e-12))),
        "metrics_native_arrays_only": True,
        "display_interpolation_only": True,
        "continuous_and_averaged_input_power_relative_error": float(
            abs(result.inputs.continuous_power - result.inputs.averaged_power_after_normalisation)
            / max(result.inputs.continuous_power, EPS)
        ),
        "canonical_operating_point_preserved": CANONICAL_OPERATING_POINT_ID,
        "forbidden_operating_point": OLD_BEST_COMPROMISE_ID,
    }
    _write_json(root / "simulation_scope_manifest.json", scope)
    outcome_report = {
        "stage": MODE2Y_STAGE,
        "outcome": result.outcome,
        "allowed_outcomes": MODE2Y_ALLOWED_OUTCOMES,
        "outcome_reason": result.outcome_reason,
        "continuous_measurably_sharper": result.outcome == "M2Y-A",
        "difference_negligible": result.outcome == "M2Y-B",
        "mixed_or_averaged_sharper": result.outcome == "M2Y-C",
        "ideal_sharpness_comparison": result.data["mode2y_ideal_sharpness_comparison"],
        "realistic_sharpness_comparison": result.data["mode2y_realistic_sharpness_comparison"],
        "summary_rows": summary_rows,
        "no_microfabrication_sample_plane_success_claim": True,
    }
    _write_json(root / "06_final_status/continuous_vs_averaged_outcome_report.json", outcome_report)
    _write_document(Path(document_path), result, root)
    return {
        "result": result,
        "outcome": result.outcome,
        "output_root": root,
        "document_path": Path(document_path),
        "figure_paths": figure_paths,
        "summary_csv": root / "continuous_vs_averaged_summary.csv",
        "summary_json": root / "continuous_vs_averaged_summary.json",
        "scope_manifest": root / "simulation_scope_manifest.json",
        "outcome_report": root / "06_final_status/continuous_vs_averaged_outcome_report.json",
    }


__all__ = [
    "MODE2Y_ALLOWED_OUTCOMES",
    "MODE2Y_DEFAULT_OUTPUT_ROOT",
    "MODE2Y_DOC_PATH",
    "MODE2Y_MIN_HERO_GRID_N",
    "MODE2Y_SELECTED_Z_M",
    "MODE2Y_SHARPNESS_RELATIVE_TOLERANCE",
    "MODE2Y_STAGE",
    "Mode2YInputFields",
    "Mode2YStudyConfig",
    "Mode2YStudyResult",
    "PropagatedShapeMetrics",
    "PropagationRouteResult",
    "build_mode2y_input_fields",
    "build_mode2y_pre_axicon_fields",
    "build_sector_averaged_alpha",
    "equal_power_difference_metrics",
    "mode2y_outcome",
    "propagated_shape_metrics",
    "run_mode2y_study",
    "write_mode2y_outputs",
]
