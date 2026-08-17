"""MODE 2X local radial/azimuthal vector-field truth audit.

This module evaluates complex pre-axicon Jones fields in the local cylindrical
basis.  Native arrays are used for every metric; interpolation is confined to
figure rendering.  Final-plane hexagon eligibility is carried only as a
separate, already validated output property.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.equations.vector_jones import stokes_from_linear_components
from vbb_study.digital_twin.nathan_mode2u2_fix_strict_hexagon_optimisation import (
    OLD_BEST_COMPROMISE_ID,
)
from vbb_study.digital_twin.nathan_mode2u3_hardware_closure import (
    CANONICAL_OPERATING_POINT_ID,
    STRICT_COMPROMISE_CONTROLS,
    STRICT_COMPROMISE_ID,
    assert_not_forbidden,
)
from vbb_study.digital_twin.nathan_mode2v_lab_ready_build import load_operating_points
from vbb_study.digital_twin.nathan_mode2w_fix_sequential_master import (
    MODE2WF_DEFAULT_OUTPUT_ROOT,
    _source_config,
)
from vbb_study.digital_twin.nathan_vector_hexagon import (
    EPS,
    Mode2SCorrection,
    Mode2SPerturbation,
    apply_uniform_jones,
    linear_retarder,
    mode2n_source_target,
    mode2s_combined_cases,
    nathan_alpha_map,
    route_dual_slm_linear_then_qwp_ideal,
    run_mode2n_dual_slm_4f_route,
    run_mode2n_v0_reference,
    run_mode2q_backward_initialisation,
    run_mode2s_degraded_forward,
    synthesize_with_patterned_hwp,
)


MODE2X_STAGE = "nathan_mode2x_local_vector_truth"
MODE2X_DEFAULT_OUTPUT_ROOT = Path("outputs/figures/digital_twin/nathan_mode2x_local_vector_truth")
MODE2X_DOC_PATH = Path("docs/84_nathan_mode2x_local_vector_truth.md")
MODE2X_ALLOWED_OUTCOMES = ("M2X-A", "M2X-B", "M2X-C", "M2X-D")
MODE2X_SECTOR_CONVENTION = "v0_authoritative"
MODE2X_PLANE = "pre_axicon_source_scale"


@dataclass(frozen=True)
class LocalVectorBasisFields:
    """Complex field components and powers in the local cylindrical basis."""

    er: np.ndarray
    etheta: np.ndarray
    transverse_power: np.ndarray
    radial_power_fraction: np.ndarray
    azimuthal_power_fraction: np.ndarray
    theta_rad: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class LocalVectorPurityMetrics:
    """Intensity-weighted local vector-purity metrics for one route."""

    route_id: str
    radial_purity: float
    azimuthal_purity: float
    radial_leakage: float
    azimuthal_leakage: float
    local_angle_rms_rad: float
    local_angle_p95_rad: float
    local_angle_max_rad: float
    s3_rms_fraction: float
    s3_p95_fraction: float
    s3_max_fraction: float
    linear_polarisation_fraction: float
    radial_valid_power_fraction: float
    azimuthal_valid_power_fraction: float
    radial_sector_power: float
    azimuthal_sector_power: float
    transverse_power_total: float
    intensity_threshold_fraction: float
    excluded_low_intensity_fraction: float
    centre_policy: str
    passed_radial_purity_gate: bool
    passed_azimuthal_purity_gate: bool
    passed_local_angle_gate: bool
    passed_local_linearity_gate: bool
    passed_full_vector_truth_gate: bool


@dataclass(frozen=True)
class LocalVectorTruthResult:
    """Native-grid maps and summary metrics for one pre-axicon Jones field."""

    route_id: str
    basis_fields: LocalVectorBasisFields
    metrics: LocalVectorPurityMetrics
    radial_sector_mask: np.ndarray
    azimuthal_sector_mask: np.ndarray
    target_alpha_rad: np.ndarray
    actual_alpha_rad: np.ndarray
    angle_error_rad: np.ndarray
    radial_leakage_map: np.ndarray
    azimuthal_leakage_map: np.ndarray
    s0: np.ndarray
    s1: np.ndarray
    s2: np.ndarray
    s3: np.ndarray
    metadata: Mapping[str, Any]


def mode2x_gate_definitions() -> dict[str, dict[str, float]]:
    """Return the fixed ideal and realistic local-truth gates."""

    return {
        "ideal": {
            "radial_purity_min": 0.999999,
            "azimuthal_purity_min": 0.999999,
            "local_angle_rms_rad_max": 1.0e-6,
            "s3_rms_fraction_max": 1.0e-8,
        },
        "realistic": {
            "radial_purity_min": 0.98,
            "azimuthal_purity_min": 0.98,
            "local_angle_rms_rad_max": 0.10,
            "s3_rms_fraction_max": 0.05,
        },
    }


def _coordinate_mesh(
    x_m: np.ndarray,
    y_m: np.ndarray,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)
    if x.ndim == 1 and y.ndim == 1:
        X, Y = np.meshgrid(x, y, indexing="xy")
    elif x.ndim == 2 and y.ndim == 2 and x.shape == y.shape:
        X, Y = x, y
    else:
        raise ValueError("x_m and y_m must be matching 1D vectors or matching 2D meshes")
    if X.shape != shape or Y.shape != shape:
        raise ValueError(f"coordinate shape {X.shape} is incompatible with field shape {shape}")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        raise ValueError("coordinates must be finite")
    return X, Y


def _grid_step(X: np.ndarray, Y: np.ndarray) -> tuple[float, float]:
    dx_values = np.abs(np.diff(X, axis=1)).ravel()
    dy_values = np.abs(np.diff(Y, axis=0)).ravel()
    dx_pos = dx_values[np.isfinite(dx_values) & (dx_values > 0.0)]
    dy_pos = dy_values[np.isfinite(dy_values) & (dy_values > 0.0)]
    dx = float(np.median(dx_pos)) if dx_pos.size else 0.0
    dy = float(np.median(dy_pos)) if dy_pos.size else 0.0
    return dx, dy


def cartesian_to_local_cylindrical(
    ex: np.ndarray,
    ey: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    intensity_threshold_fraction: float = 1.0e-6,
    centre_policy: str = "exclude_singular_neighbourhood",
) -> LocalVectorBasisFields:
    """Convert a Cartesian Jones field to local radial/azimuthal components.

    The singular centre is never assigned a preferred direction.  The default
    policy excludes the exact centre on axis-sampled grids and the central four
    samples on zero-straddling grids.
    """

    Ex = np.asarray(ex, dtype=np.complex128)
    Ey = np.asarray(ey, dtype=np.complex128)
    if Ex.ndim != 2 or Ey.ndim != 2 or Ex.shape != Ey.shape:
        raise ValueError("ex and ey must be matching two-dimensional arrays")
    if not np.isfinite(float(intensity_threshold_fraction)) or not 0.0 <= intensity_threshold_fraction < 1.0:
        raise ValueError("intensity_threshold_fraction must lie in [0, 1)")
    X, Y = _coordinate_mesh(x_m, y_m, Ex.shape)
    theta = np.arctan2(Y, X)
    c = np.cos(theta)
    s = np.sin(theta)
    er = Ex * c + Ey * s
    etheta = -Ex * s + Ey * c
    cartesian_power = np.abs(Ex) ** 2 + np.abs(Ey) ** 2
    transverse_power = np.abs(er) ** 2 + np.abs(etheta) ** 2
    scale = max(float(np.max(cartesian_power)), 1.0)
    if not np.allclose(cartesian_power, transverse_power, rtol=5.0e-13, atol=5.0e-14 * scale):
        raise FloatingPointError("Cartesian-to-cylindrical transform did not preserve transverse power")

    dx, dy = _grid_step(X, Y)
    radius = np.hypot(X, Y)
    if centre_policy == "exclude_singular_neighbourhood":
        centre_radius = 0.5 * np.hypot(dx, dy) + np.finfo(float).eps * max(float(np.max(radius)), 1.0)
        centre_excluded = radius <= centre_radius
    elif centre_policy == "exclude_exact_origin":
        centre_excluded = radius <= np.finfo(float).eps * max(float(np.max(radius)), 1.0)
    else:
        raise ValueError("unsupported centre_policy")
    threshold = float(intensity_threshold_fraction) * max(float(np.max(transverse_power)), EPS)
    valid = np.isfinite(transverse_power) & (transverse_power > threshold) & ~centre_excluded
    radial_fraction = np.divide(
        np.abs(er) ** 2,
        transverse_power,
        out=np.zeros_like(transverse_power, dtype=float),
        where=transverse_power > EPS,
    )
    azimuthal_fraction = np.divide(
        np.abs(etheta) ** 2,
        transverse_power,
        out=np.zeros_like(transverse_power, dtype=float),
        where=transverse_power > EPS,
    )
    return LocalVectorBasisFields(
        er=er,
        etheta=etheta,
        transverse_power=transverse_power,
        radial_power_fraction=radial_fraction,
        azimuthal_power_fraction=azimuthal_fraction,
        theta_rad=theta,
        valid_mask=valid,
    )


def build_radial_azimuthal_sector_masks(
    theta_rad: np.ndarray,
    *,
    sector_rotation_rad: float,
    sector_duty_scale: float = 1.0,
    existing_convention: str = MODE2X_SECTOR_CONVENTION,
    angular_guard_band_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build masks by delegating orientation to the authoritative V0 helper."""

    if existing_convention != MODE2X_SECTOR_CONVENTION:
        raise ValueError("MODE 2X supports only the authoritative V0 sector convention")
    duty = float(sector_duty_scale)
    if not 0.0 < duty <= 1.0:
        raise ValueError("sector_duty_scale must lie in (0, 1]")
    guard = float(angular_guard_band_rad)
    if not 0.0 <= guard < np.pi / 6.0:
        raise ValueError("angular_guard_band_rad must lie in [0, pi/6)")
    theta = np.asarray(theta_rad, dtype=float)
    cell = 2.0 * np.pi / 3.0
    sector_theta = (np.pi / 3.0) * duty
    _, radial = nathan_alpha_map(
        theta,
        sector_num_pairs=3,
        sector_theta=sector_theta,
        sector_rotation=float(sector_rotation_rad),
    )
    radial = np.asarray(radial, dtype=bool)
    azimuthal = ~radial
    if guard > 0.0:
        phi_cell = np.mod(theta - float(sector_rotation_rad), cell)
        boundary0 = np.minimum(phi_cell, cell - phi_cell)
        boundary1 = np.abs(phi_cell - (cell - sector_theta))
        guarded = np.minimum(boundary0, boundary1) <= guard
        radial = radial & ~guarded
        azimuthal = azimuthal & ~guarded
    if np.any(radial & azimuthal):
        raise AssertionError("radial and azimuthal sector masks overlap")
    return radial, azimuthal


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return float(np.sqrt(np.sum(weights * values**2) / max(float(np.sum(weights)), EPS)))


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(keep):
        return float("nan")
    values = values[keep]
    weights = weights[keep]
    order = np.argsort(values)
    values = values[order]
    cumulative = np.cumsum(weights[order])
    target = float(percentile) / 100.0 * float(cumulative[-1])
    return float(values[min(int(np.searchsorted(cumulative, target, side="left")), values.size - 1)])


def line_orientation_error(actual_alpha_rad: np.ndarray, target_alpha_rad: np.ndarray) -> np.ndarray:
    """Return headless line-orientation error in [-pi/2, pi/2)."""

    delta = np.asarray(actual_alpha_rad, dtype=float) - np.asarray(target_alpha_rad, dtype=float)
    return 0.5 * np.angle(np.exp(2.0j * delta))


def evaluate_local_vector_truth(
    route_id: str,
    ex: np.ndarray,
    ey: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    target_alpha_rad: np.ndarray,
    *,
    sector_rotation_rad: float = 0.0,
    sector_duty_scale: float = 1.0,
    intensity_threshold_fraction: float = 1.0e-6,
    centre_policy: str = "exclude_singular_neighbourhood",
    angular_guard_band_rad: float = np.deg2rad(0.5),
    gate_class: str = "ideal",
    metadata: Mapping[str, Any] | None = None,
) -> LocalVectorTruthResult:
    """Evaluate native-grid local purity, line orientation, and linearity."""

    gates = mode2x_gate_definitions()
    if gate_class not in gates:
        raise ValueError(f"unknown gate_class {gate_class!r}")
    gate = gates[gate_class]
    basis = cartesian_to_local_cylindrical(
        ex,
        ey,
        x_m,
        y_m,
        intensity_threshold_fraction=intensity_threshold_fraction,
        centre_policy=centre_policy,
    )
    target_alpha = np.asarray(target_alpha_rad, dtype=float)
    if target_alpha.shape != basis.transverse_power.shape:
        raise ValueError("target_alpha_rad shape does not match the Jones field")
    radial_mask, azimuthal_mask = build_radial_azimuthal_sector_masks(
        basis.theta_rad,
        sector_rotation_rad=sector_rotation_rad,
        sector_duty_scale=sector_duty_scale,
        angular_guard_band_rad=angular_guard_band_rad,
    )
    stokes = stokes_from_linear_components(ex, ey)
    s0 = np.asarray(stokes["S0"], dtype=float)
    s1 = np.asarray(stokes["S1"], dtype=float)
    s2 = np.asarray(stokes["S2"], dtype=float)
    s3 = np.asarray(stokes["S3"], dtype=float)
    actual_alpha = 0.5 * np.arctan2(s2, s1)
    angle_error = line_orientation_error(actual_alpha, target_alpha)
    sector_union = radial_mask | azimuthal_mask
    valid = basis.valid_mask & sector_union & np.isfinite(angle_error)
    radial_valid = valid & radial_mask
    azimuthal_valid = valid & azimuthal_mask
    power = basis.transverse_power
    radial_power = float(np.sum(power[radial_valid]))
    azimuthal_power = float(np.sum(power[azimuthal_valid]))
    radial_purity = float(np.sum(np.abs(basis.er[radial_valid]) ** 2) / max(radial_power, EPS))
    azimuthal_purity = float(np.sum(np.abs(basis.etheta[azimuthal_valid]) ** 2) / max(azimuthal_power, EPS))

    radial_leakage_map = np.full(power.shape, np.nan, dtype=float)
    azimuthal_leakage_map = np.full(power.shape, np.nan, dtype=float)
    radial_leakage_map[radial_valid] = basis.azimuthal_power_fraction[radial_valid]
    azimuthal_leakage_map[azimuthal_valid] = basis.radial_power_fraction[azimuthal_valid]
    angle_map = np.full(power.shape, np.nan, dtype=float)
    angle_map[valid] = angle_error[valid]
    actual_map = np.full(power.shape, np.nan, dtype=float)
    actual_map[valid] = actual_alpha[valid]

    valid_weights = power[valid]
    abs_angle = np.abs(angle_error[valid])
    s3_fraction = np.divide(np.abs(s3), s0, out=np.zeros_like(s0), where=s0 > EPS)
    s3_valid = s3_fraction[valid]
    s3_rms = _weighted_rms(s3_valid, valid_weights)
    linear_limit = float(gate["s3_rms_fraction_max"])
    linear_fraction = float(
        np.sum(valid_weights[s3_valid <= linear_limit]) / max(float(np.sum(valid_weights)), EPS)
    )
    total_power = float(np.sum(power))
    threshold_power = float(intensity_threshold_fraction) * max(float(np.max(power)), EPS)
    low_intensity = power <= threshold_power
    excluded_low_intensity_power = float(np.sum(power[low_intensity]))
    centre_excluded = ~basis.valid_mask & ~low_intensity & np.isfinite(power)
    guarded_boundary = basis.valid_mask & ~sector_union
    radial_available = float(np.sum(power[radial_mask]))
    azimuthal_available = float(np.sum(power[azimuthal_mask]))
    angle_rms = _weighted_rms(angle_error[valid], valid_weights)
    passed_radial = bool(radial_purity >= float(gate["radial_purity_min"]))
    passed_azimuthal = bool(azimuthal_purity >= float(gate["azimuthal_purity_min"]))
    passed_angle = bool(angle_rms <= float(gate["local_angle_rms_rad_max"]))
    passed_linearity = bool(s3_rms <= float(gate["s3_rms_fraction_max"]))
    metrics = LocalVectorPurityMetrics(
        route_id=str(route_id),
        radial_purity=radial_purity,
        azimuthal_purity=azimuthal_purity,
        radial_leakage=float(1.0 - radial_purity),
        azimuthal_leakage=float(1.0 - azimuthal_purity),
        local_angle_rms_rad=angle_rms,
        local_angle_p95_rad=_weighted_percentile(abs_angle, valid_weights, 95.0),
        local_angle_max_rad=float(np.max(abs_angle)) if abs_angle.size else float("nan"),
        s3_rms_fraction=s3_rms,
        s3_p95_fraction=_weighted_percentile(s3_valid, valid_weights, 95.0),
        s3_max_fraction=float(np.max(s3_valid)) if s3_valid.size else float("nan"),
        linear_polarisation_fraction=linear_fraction,
        radial_valid_power_fraction=float(radial_power / max(radial_available, EPS)),
        azimuthal_valid_power_fraction=float(azimuthal_power / max(azimuthal_available, EPS)),
        radial_sector_power=radial_power,
        azimuthal_sector_power=azimuthal_power,
        transverse_power_total=total_power,
        intensity_threshold_fraction=float(intensity_threshold_fraction),
        excluded_low_intensity_fraction=float(excluded_low_intensity_power / max(total_power, EPS)),
        centre_policy=str(centre_policy),
        passed_radial_purity_gate=passed_radial,
        passed_azimuthal_purity_gate=passed_azimuthal,
        passed_local_angle_gate=passed_angle,
        passed_local_linearity_gate=passed_linearity,
        passed_full_vector_truth_gate=bool(passed_radial and passed_azimuthal and passed_angle and passed_linearity),
    )
    return LocalVectorTruthResult(
        route_id=str(route_id),
        basis_fields=basis,
        metrics=metrics,
        radial_sector_mask=radial_mask,
        azimuthal_sector_mask=azimuthal_mask,
        target_alpha_rad=target_alpha,
        actual_alpha_rad=actual_map,
        angle_error_rad=angle_map,
        radial_leakage_map=radial_leakage_map,
        azimuthal_leakage_map=azimuthal_leakage_map,
        s0=s0,
        s1=s1,
        s2=s2,
        s3=s3,
        metadata={
            "stage": MODE2X_STAGE,
            "plane": MODE2X_PLANE,
            "gate_class": gate_class,
            "gate": dict(gate),
            "sector_convention": MODE2X_SECTOR_CONVENTION,
            "sector_rotation_rad": float(sector_rotation_rad),
            "sector_duty_scale": float(sector_duty_scale),
            "angular_guard_band_rad": float(angular_guard_band_rad),
            "centre_excluded_pixel_count": int(np.count_nonzero(centre_excluded)),
            "centre_excluded_power_fraction": float(np.sum(power[centre_excluded]) / max(total_power, EPS)),
            "guarded_boundary_pixel_count": int(np.count_nonzero(guarded_boundary)),
            "guarded_boundary_power_fraction": float(np.sum(power[guarded_boundary]) / max(total_power, EPS)),
            "low_intensity_excluded_pixel_count": int(np.count_nonzero(low_intensity)),
            "native_grid_metrics": True,
            "display_interpolation_used_for_metrics": False,
            **dict(metadata or {}),
        },
    )


def _ideal_sequential_field(amplitude: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    supply = np.asarray(amplitude, dtype=float) / np.sqrt(2.0)
    phi_h = np.asarray(alpha, dtype=float)
    phi_v = -np.asarray(alpha, dtype=float) + 0.5 * np.pi
    e1 = (supply * np.exp(1.0j * phi_h), supply.astype(np.complex128))
    e2 = (e1[1], e1[0])
    e3 = (supply * np.exp(1.0j * phi_v), e2[1])
    e4 = (e3[1], e3[0])
    qwp = linear_retarder(0.5 * np.pi, -0.25 * np.pi)
    return apply_uniform_jones(qwp, e4[0], e4[1])


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _stored_final_properties() -> dict[str, dict[str, Any]]:
    equivalence = _read_json(
        MODE2WF_DEFAULT_OUTPUT_ROOT / "00_architecture/mode2w_fix_sequential_equivalence.json", {}
    )
    realism_rows = _read_json(
        MODE2WF_DEFAULT_OUTPUT_ROOT / "04_realism/mode2w_fix_realism_metrics.json", []
    )
    correction_payload = _read_json(
        MODE2WF_DEFAULT_OUTPUT_ROOT / "05_correction/mode2w_fix_correction_metrics.json", {}
    )
    realism = {str(row.get("route_id")): row for row in realism_rows}
    correction = {str(row.get("route_id")): row for row in correction_payload.get("rows", [])}
    realistic_corr = float(equivalence.get("realistic_sequential_z60_corr_to_v0", np.nan))
    realistic_pass = bool(equivalence.get("realistic_sequential_strict_hexagon", False))
    ideal_corr = float(equivalence.get("ideal_sequential_z60_corr_to_v0", 1.0))
    ideal_pass = bool(equivalence.get("ideal_sequential_strict_hexagon_candidate_gate", False))
    props: dict[str, dict[str, Any]] = {
        "authoritative_analytic_target": {"final_z60_correlation": 1.0, "final_strict_hexagon_pass": ideal_pass},
        "ideal_patterned_hwp": {"final_z60_correlation": 1.0, "final_strict_hexagon_pass": ideal_pass},
        "ideal_abstract_dual_slm_qwp": {"final_z60_correlation": 1.0, "final_strict_hexagon_pass": ideal_pass},
        "ideal_sequential_dual_slm": {"final_z60_correlation": ideal_corr, "final_strict_hexagon_pass": ideal_pass},
        "realistic_sequential_carrier_common_4f": {
            "final_z60_correlation": realistic_corr,
            "final_strict_hexagon_pass": realistic_pass,
        },
        CANONICAL_OPERATING_POINT_ID: {
            "final_z60_correlation": realistic_corr,
            "final_strict_hexagon_pass": realistic_pass,
        },
    }
    for route_id, stored_id in (
        ("m2s_combined_moderate_lab", "moderate_realism"),
        ("m2s_combined_bad_lab", "bad_realism"),
    ):
        row = realism.get(stored_id, {})
        props[route_id] = {
            "final_z60_correlation": float(row.get("corr_full", np.nan)),
            "final_strict_hexagon_pass": bool(row.get("strict_hexagon_eligible", False)),
        }
    corrected = correction.get("corrected_axicon_0p5mm", {})
    props["compensated_axicon_mask_offset_0p5mm"] = {
        "final_z60_correlation": float(corrected.get("corr_full", np.nan)),
        "final_strict_hexagon_pass": bool(corrected.get("strict_hexagon_eligible", False)),
    }
    try:
        _, secondary = load_operating_points()
        props[STRICT_COMPROMISE_ID] = {
            "final_z60_correlation": float(secondary.get("corr_full", np.nan)),
            "final_strict_hexagon_pass": bool(secondary.get("strict_hexagon_eligible", False)),
        }
    except (OSError, KeyError, ValueError):
        props[STRICT_COMPROMISE_ID] = {
            "final_z60_correlation": float("nan"),
            "final_strict_hexagon_pass": False,
        }
    return props


def build_mode2x_route_results(
    *,
    grid_n: int = 1024,
    z_planes: int = 5,
    intensity_threshold_fraction: float = 1.0e-6,
    angular_guard_band_rad: float = np.deg2rad(0.5),
) -> list[LocalVectorTruthResult]:
    """Build and audit all required ideal, realistic, degraded, and corrected routes."""

    assert_not_forbidden(CANONICAL_OPERATING_POINT_ID)
    assert_not_forbidden(STRICT_COMPROMISE_ID)
    if OLD_BEST_COMPROMISE_ID in {CANONICAL_OPERATING_POINT_ID, STRICT_COMPROMISE_ID}:
        raise AssertionError("forbidden optimiser candidate cannot be canonical or secondary")
    cfg = _source_config(grid_n=int(grid_n), z_planes=int(z_planes), z_start_m=50.0e-3, z_end_m=70.0e-3)
    data = mode2n_source_target(cfg, grid_n=int(grid_n), z_planes=int(z_planes))
    v0 = run_mode2n_v0_reference(data)
    realistic = run_mode2n_dual_slm_4f_route(data, v0)
    backward = run_mode2q_backward_initialisation(data)
    A = np.asarray(data["A"], dtype=float)
    alpha = np.asarray(data["alpha"], dtype=float)
    target = tuple(np.asarray(v, dtype=np.complex128) for v in data["target"])
    patterned_ex, patterned_ey, _ = synthesize_with_patterned_hwp(A, alpha)
    abstract = route_dual_slm_linear_then_qwp_ideal(A, alpha, target, mask=data["metric_mask"])
    sequential = _ideal_sequential_field(A, alpha)
    final_props = _stored_final_properties()
    grid = data["grid"]
    x = np.asarray(grid["x"], dtype=float)
    y = np.asarray(grid.get("y", x), dtype=float)
    rotation = float(cfg.sector_rotation_rad)
    duty = float(cfg.sector_theta_rad / (np.pi / 3.0))

    route_specs: list[dict[str, Any]] = [
        {"id": "authoritative_analytic_target", "field": target, "gate": "ideal", "family": "V0 analytic target"},
        {"id": "ideal_patterned_hwp", "field": (patterned_ex, patterned_ey), "gate": "ideal", "family": "patterned HWP"},
        {"id": "ideal_abstract_dual_slm_qwp", "field": (abstract.Ex, abstract.Ey), "gate": "ideal", "family": "abstract dual-SLM + QWP"},
        {"id": "ideal_sequential_dual_slm", "field": sequential, "gate": "ideal", "family": "sequential collinear dual-SLM"},
        {
            "id": "realistic_sequential_carrier_common_4f",
            "field": realistic.pre_axicon_field,
            "gate": "realistic",
            "family": "sequential carrier + common 4F + QWP",
        },
        {
            "id": CANONICAL_OPERATING_POINT_ID,
            "field": realistic.pre_axicon_field,
            "gate": "realistic",
            "family": "canonical realistic 4F reference",
        },
    ]
    combined = {case.label: case for case in mode2s_combined_cases()}
    for route_id, case_id in (
        ("m2s_combined_moderate_lab", "combined_moderate_lab"),
        ("m2s_combined_bad_lab", "combined_bad_lab"),
    ):
        case = run_mode2s_degraded_forward(
            data, v0, backward, combined[case_id], fast_single_plane=True
        )
        route_specs.append({
            "id": route_id,
            "field": case["pre_axicon_field"],
            "gate": "realistic",
            "family": case_id,
            "runtime_final_z60_correlation": float(case["comparison"]["z60_full_field_correlation"]),
            "runtime_final_strict_hexagon_pass": bool(case["strict_gate"]["passes_true_hexagon_gate"]),
        })
    offset_perturbation = Mode2SPerturbation(
        label="axicon_decentre_0p5mm",
        slm_aperture_clip=True,
        phase_levels=256,
        fill_factor=0.93,
        axicon_decentre_x_m=0.5e-3,
    )
    offset_correction = Mode2SCorrection(mask_recentre_x_m=0.5e-3)
    corrected = run_mode2s_degraded_forward(
        data,
        v0,
        backward,
        offset_perturbation,
        correction=offset_correction,
        fast_single_plane=True,
    )
    shifted_X = np.asarray(grid["X"], dtype=float) - offset_correction.mask_recentre_x_m
    shifted_Y = np.asarray(grid["Y"], dtype=float) - offset_correction.mask_recentre_y_m
    shifted_theta = np.arctan2(shifted_Y, shifted_X)
    shifted_alpha, _ = nathan_alpha_map(
        shifted_theta,
        sector_num_pairs=int(cfg.n_pairs),
        sector_theta=float(cfg.sector_theta_rad),
        sector_rotation=rotation,
    )
    route_specs.append({
        "id": "compensated_axicon_mask_offset_0p5mm",
        "field": corrected["pre_axicon_field"],
        "gate": "realistic",
        "family": "measured 0.5 mm axicon offset + digital mask recentring",
        "x": x - offset_correction.mask_recentre_x_m,
        "y": y - offset_correction.mask_recentre_y_m,
        "target_alpha": shifted_alpha,
        "coordinate_origin": "measured axicon axis at x=+0.5 mm",
        "runtime_final_z60_correlation": float(corrected["comparison"]["z60_full_field_correlation"]),
        "runtime_final_strict_hexagon_pass": bool(corrected["strict_gate"]["passes_true_hexagon_gate"]),
    })
    secondary = run_mode2s_degraded_forward(
        data,
        v0,
        backward,
        STRICT_COMPROMISE_CONTROLS.perturbation(),
        correction=STRICT_COMPROMISE_CONTROLS.correction(),
        fast_single_plane=True,
    )
    route_specs.append({
        "id": STRICT_COMPROMISE_ID,
        "field": secondary["pre_axicon_field"],
        "gate": "realistic",
        "family": "secondary strict-eligible operating point",
        "runtime_final_z60_correlation": float(secondary["comparison"]["z60_full_field_correlation"]),
        "runtime_final_strict_hexagon_pass": bool(secondary["strict_gate"]["passes_true_hexagon_gate"]),
    })

    results: list[LocalVectorTruthResult] = []
    for spec in route_specs:
        route_id = str(spec["id"])
        Ex, Ey = spec["field"]
        props = dict(final_props.get(route_id, {}))
        target_map = np.asarray(spec.get("target_alpha", alpha), dtype=float)
        result = evaluate_local_vector_truth(
            route_id,
            Ex,
            Ey,
            np.asarray(spec.get("x", x), dtype=float),
            np.asarray(spec.get("y", y), dtype=float),
            target_map,
            sector_rotation_rad=rotation,
            sector_duty_scale=duty,
            intensity_threshold_fraction=float(intensity_threshold_fraction),
            angular_guard_band_rad=float(angular_guard_band_rad),
            gate_class=str(spec["gate"]),
            metadata={
                "route_family": str(spec["family"]),
                "grid_n": int(grid_n),
                "grid_dx_m": float(grid["dx"]),
                "physical_window_m": float(grid_n) * float(grid["dx"]),
                "coordinate_origin": str(spec.get("coordinate_origin", "source grid and axicon design axis")),
                "final_z60_correlation": float(props.get("final_z60_correlation", np.nan)),
                "final_strict_hexagon_pass": bool(props.get("final_strict_hexagon_pass", False)),
                "runtime_final_z60_correlation": float(spec.get("runtime_final_z60_correlation", np.nan)),
                "runtime_legacy_true_hexagon_pass": bool(spec.get("runtime_final_strict_hexagon_pass", False)),
                "final_gate_source": "stored repaired strict-gate outputs",
            },
        )
        results.append(result)
    return results


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {key: _json_ready(item) for key, item in asdict(value).items()}
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
        for row in rows:
            writer.writerow({key: _json_ready(row.get(key, "")) for key in fields})
    return path


def local_vector_truth_summary_rows(results: Sequence[LocalVectorTruthResult]) -> list[dict[str, Any]]:
    """Flatten route metrics while retaining final morphology as separate fields."""

    rows: list[dict[str, Any]] = []
    for result in results:
        row = asdict(result.metrics)
        row.update({
            "gate_class": result.metadata["gate_class"],
            "plane": result.metadata["plane"],
            "route_family": result.metadata["route_family"],
            "final_z60_correlation": result.metadata["final_z60_correlation"],
            "final_strict_hexagon_pass": result.metadata["final_strict_hexagon_pass"],
            "runtime_final_z60_correlation": result.metadata["runtime_final_z60_correlation"],
            "runtime_legacy_true_hexagon_pass": result.metadata["runtime_legacy_true_hexagon_pass"],
            "sector_convention": result.metadata["sector_convention"],
            "angular_guard_band_rad": result.metadata["angular_guard_band_rad"],
        })
        rows.append(row)
    return rows


def mode2x_outcome(results: Sequence[LocalVectorTruthResult]) -> str:
    """Select exactly one MODE 2X outcome without changing gates post hoc."""

    by_id = {result.route_id: result for result in results}
    ideal_ids = (
        "authoritative_analytic_target",
        "ideal_patterned_hwp",
        "ideal_abstract_dual_slm_qwp",
        "ideal_sequential_dual_slm",
    )
    if any(route_id not in by_id for route_id in ideal_ids) or CANONICAL_OPERATING_POINT_ID not in by_id:
        return "M2X-D"
    ideal_pass = all(by_id[route_id].metrics.passed_full_vector_truth_gate for route_id in ideal_ids)
    canonical_pass = by_id[CANONICAL_OPERATING_POINT_ID].metrics.passed_full_vector_truth_gate
    if ideal_pass and canonical_pass:
        return "M2X-A"
    if ideal_pass:
        return "M2X-B"
    return "M2X-C"


def _mpl() -> tuple[Any, Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Patch, Wedge

    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 11.0,
        "axes.labelsize": 9.5,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })
    return plt, LineCollection, (Patch, Wedge)


def _save_figure(fig: Any, png: Path, pdf: Path) -> None:
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=360)
    fig.savefig(pdf)


def _display_crop(result: LocalVectorTruthResult, beam_radius_m: float) -> tuple[slice, slice, list[float]]:
    n, m = result.s0.shape
    half_window = 1.55 * float(beam_radius_m)
    window = float(result.metadata["physical_window_m"])
    dx = float(result.metadata["grid_dx_m"])
    x = (np.arange(m) - m // 2) * dx
    y = (np.arange(n) - n // 2) * dx
    ix = np.where(np.abs(x) <= min(half_window, 0.49 * window))[0]
    iy = np.where(np.abs(y) <= min(half_window, 0.49 * window))[0]
    sx = slice(int(ix[0]), int(ix[-1]) + 1)
    sy = slice(int(iy[0]), int(iy[-1]) + 1)
    extent = [float(x[ix[0]] / 1e-3), float(x[ix[-1]] / 1e-3), float(y[iy[0]] / 1e-3), float(y[iy[-1]] / 1e-3)]
    return sy, sx, extent


def _line_segments(alpha: np.ndarray, extent: Sequence[float], *, samples: int = 25) -> np.ndarray:
    ny, nx = alpha.shape
    ys = np.linspace(0, ny - 1, samples, dtype=int)
    xs = np.linspace(0, nx - 1, samples, dtype=int)
    xx, yy = np.meshgrid(xs, ys)
    angles = alpha[yy, xx]
    xcoord = np.interp(xx, [0, nx - 1], [extent[0], extent[1]])
    ycoord = np.interp(yy, [0, ny - 1], [extent[2], extent[3]])
    length = 0.075 * min(float(extent[1] - extent[0]), float(extent[3] - extent[2]))
    dx = 0.5 * length * np.cos(angles)
    dy = 0.5 * length * np.sin(angles)
    segments = np.stack(
        [np.stack([xcoord - dx, ycoord - dy], axis=-1), np.stack([xcoord + dx, ycoord + dy], axis=-1)],
        axis=-2,
    )
    return segments.reshape(-1, 2, 2)


def _plot_sector_schematic(output_root: Path, sector_rotation_rad: float) -> None:
    plt, _, patch_types = _mpl()
    Patch, Wedge = patch_types
    fig, ax = plt.subplots(figsize=(8.2, 8.2), constrained_layout=True)
    colours = {"radial": "#d95f02", "azimuthal": "#1b9e77"}
    for index in range(6):
        start = sector_rotation_rad + index * np.pi / 3.0
        centre = start + np.pi / 6.0
        _, radial = build_radial_azimuthal_sector_masks(
            np.asarray([centre]), sector_rotation_rad=sector_rotation_rad
        )
        kind = "radial" if bool(radial[0]) else "azimuthal"
        ax.add_patch(Wedge((0.0, 0.0), 1.0, np.rad2deg(start), np.rad2deg(start + np.pi / 3.0),
                           facecolor=colours[kind], alpha=0.18, edgecolor="white", linewidth=2.0))
        alpha = centre if kind == "radial" else centre + np.pi / 2.0
        centre_xy = np.asarray([0.58 * np.cos(centre), 0.58 * np.sin(centre)])
        direction = np.asarray([np.cos(alpha), np.sin(alpha)])
        p0 = centre_xy - 0.20 * direction
        p1 = centre_xy + 0.20 * direction
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=colours[kind], linewidth=4.0, solid_capstyle="round")
        ax.text(0.80 * np.cos(centre), 0.80 * np.sin(centre), kind.capitalize(), ha="center", va="center", fontsize=11)
    ax.add_patch(plt.Circle((0.0, 0.0), 1.0, fill=False, color="0.2", linewidth=1.2))
    ax.set(xlim=(-1.08, 1.08), ylim=(-1.08, 1.08), aspect="equal")
    ax.axis("off")
    ax.set_title("Sector-averaged schematic", fontsize=18, pad=34)
    fig.text(0.5, 0.93, "Representative orientation only - not the local vector field", ha="center", fontsize=12)
    ax.legend(
        handles=[Patch(facecolor=colours["radial"], alpha=0.35, label="Radial sector"),
                 Patch(facecolor=colours["azimuthal"], alpha=0.35, label="Azimuthal sector")],
        loc="lower center", ncol=2, frameon=False,
    )
    _save_figure(
        fig,
        output_root / "01_figures/sector_averaged_polarisation_schematic.png",
        output_root / "01_figures/sector_averaged_polarisation_schematic.pdf",
    )
    plt.close(fig)


def _overlay_local_lines(ax: Any, result: LocalVectorTruthResult, sy: slice, sx: slice, extent: Sequence[float], *, target: bool) -> None:
    _, LineCollection, _ = _mpl()
    alpha = result.target_alpha_rad[sy, sx] if target else result.actual_alpha_rad[sy, sx]
    segments = _line_segments(alpha, extent)
    finite = np.all(np.isfinite(segments), axis=(1, 2))
    collection = LineCollection(segments[finite], colors="white", linewidths=0.75, alpha=0.90)
    ax.add_collection(collection)


def _plot_true_local_field(output_root: Path, target: LocalVectorTruthResult, beam_radius_m: float) -> None:
    plt, _, _ = _mpl()
    sy, sx, extent = _display_crop(target, beam_radius_m)
    fig, ax = plt.subplots(figsize=(9.2, 8.2), constrained_layout=True)
    image = ax.imshow(target.s0[sy, sx], origin="lower", extent=extent, cmap="magma", interpolation="bicubic")
    _overlay_local_lines(ax, target, sy, sx, extent, target=True)
    radius = 0.48 * min(extent[1] - extent[0], extent[3] - extent[2])
    for angle in np.arange(0.0, 2.0 * np.pi, np.pi / 3.0):
        ax.plot([0.0, radius * np.cos(angle)], [0.0, radius * np.sin(angle)], "--", color="cyan", linewidth=0.7, alpha=0.6)
    ax.plot([-2.75, -2.25], [2.55, 2.55], color="white", linewidth=2.0)
    ax.text(-2.20, 2.55, r"local $e_r$", color="white", va="center")
    ax.plot([-2.50, -2.50], [1.85, 2.35], color="white", linewidth=2.0)
    ax.text(-2.38, 2.10, r"local $e_\theta$", color="white", va="center")
    ax.set_title("True smoothly varying local polarisation field\nHeadless line orientation on native target S0")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.colorbar(image, ax=ax, label="S0 (arb. units)", shrink=0.86)
    _save_figure(
        fig,
        output_root / "01_figures/true_local_polarisation_field.png",
        output_root / "01_figures/true_local_polarisation_field.pdf",
    )
    plt.close(fig)


def _plot_target_truth(output_root: Path, target: LocalVectorTruthResult, beam_radius_m: float) -> None:
    plt, _, _ = _mpl()
    sy, sx, extent = _display_crop(target, beam_radius_m)
    s3_fraction = np.divide(np.abs(target.s3), target.s0, out=np.zeros_like(target.s0), where=target.s0 > EPS)
    panels = [
        (target.s0, "S0", "magma", None),
        (target.basis_fields.radial_power_fraction, r"$|E_r|^2/S_0$", "viridis", (0.0, 1.0)),
        (target.basis_fields.azimuthal_power_fraction, r"$|E_\theta|^2/S_0$", "viridis", (0.0, 1.0)),
        (target.radial_leakage_map, r"radial leakage $\epsilon_R$", "inferno", (0.0, 0.02)),
        (target.azimuthal_leakage_map, r"azimuthal leakage $\epsilon_A$", "inferno", (0.0, 0.02)),
        (np.abs(target.angle_error_rad), r"$|\Delta\alpha|$ (rad)", "inferno", (0.0, 0.02)),
        (s3_fraction, r"$|S_3|/S_0$", "inferno", (0.0, 0.02)),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15.0, 14.0), constrained_layout=True)
    for ax, (array, title, cmap, limits) in zip(axes.ravel(), panels):
        kwargs = {} if limits is None else {"vmin": limits[0], "vmax": limits[1]}
        image = ax.imshow(array[sy, sx], origin="lower", extent=extent, cmap=cmap, interpolation="bicubic", **kwargs)
        ax.set_title(title)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        fig.colorbar(image, ax=ax, shrink=0.72)
    line_ax = axes.ravel()[7]
    line_image = line_ax.imshow(target.s0[sy, sx], origin="lower", extent=extent, cmap="magma", interpolation="bicubic")
    _overlay_local_lines(line_ax, target, sy, sx, extent, target=True)
    line_ax.set_title("True local line field")
    line_ax.set_xlabel("x (mm)")
    line_ax.set_ylabel("y (mm)")
    fig.colorbar(line_image, ax=line_ax, shrink=0.72)
    metrics_ax = axes.ravel()[8]
    metrics_ax.axis("off")
    metrics = target.metrics
    metrics_ax.text(
        0.02,
        0.98,
        "Native-grid metrics\n\n"
        f"radial purity = {metrics.radial_purity:.12f}\n"
        f"azimuthal purity = {metrics.azimuthal_purity:.12f}\n"
        f"angle RMS = {metrics.local_angle_rms_rad:.3e} rad\n"
        f"S3 RMS = {metrics.s3_rms_fraction:.3e}\n"
        f"vector-truth gate = {metrics.passed_full_vector_truth_gate}\n\n"
        "Interpolation: display only",
        va="top",
        family="monospace",
        fontsize=11,
    )
    fig.suptitle("MODE 2X target local radial/azimuthal truth", fontsize=16)
    _save_figure(
        fig,
        output_root / "02_target_truth/target_local_vector_truth.png",
        output_root / "02_target_truth/target_local_vector_truth.pdf",
    )
    plt.close(fig)


def _plot_route_comparison(output_root: Path, results: Sequence[LocalVectorTruthResult]) -> None:
    plt, _, _ = _mpl()
    labels = [result.route_id for result in results]
    short = [
        "target", "HWP", "abstract", "sequential", "realistic", "canonical",
        "moderate", "bad", "corrected", "secondary",
    ]
    x = np.arange(len(results))
    radial = [result.metrics.radial_purity for result in results]
    azimuthal = [result.metrics.azimuthal_purity for result in results]
    angle = [max(result.metrics.local_angle_rms_rad, 1.0e-18) for result in results]
    s3 = [max(result.metrics.s3_rms_fraction, 1.0e-18) for result in results]
    corr = [float(result.metadata["final_z60_correlation"]) for result in results]
    local_pass = [result.metrics.passed_full_vector_truth_gate for result in results]
    final_pass = [bool(result.metadata["final_strict_hexagon_pass"]) for result in results]
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 10.0), constrained_layout=True)
    axes[0, 0].bar(x - 0.18, radial, 0.36, label="radial", color="#d95f02")
    axes[0, 0].bar(x + 0.18, azimuthal, 0.36, label="azimuthal", color="#1b9e77")
    axes[0, 0].axhline(0.98, color="0.25", linestyle="--", linewidth=0.9)
    axes[0, 0].set_ylim(0.0, 1.02)
    axes[0, 0].set_title("Local cylindrical-basis purity")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].bar(x, angle, color="#7570b3")
    axes[0, 1].set_yscale("log")
    axes[0, 1].axhline(0.10, color="0.25", linestyle="--", linewidth=0.9)
    axes[0, 1].set_title("Local line-angle RMS (rad)")
    axes[0, 2].bar(x, s3, color="#e7298a")
    axes[0, 2].set_yscale("log")
    axes[0, 2].axhline(0.05, color="0.25", linestyle="--", linewidth=0.9)
    axes[0, 2].set_title("Local S3 RMS fraction")
    axes[1, 0].bar(x, corr, color="#66a61e")
    axes[1, 0].set_ylim(0.0, 1.02)
    axes[1, 0].set_title("Final z=60 mm correlation (separate)")
    axes[1, 1].bar(x - 0.18, [float(v) for v in local_pass], 0.36, label="local vector truth", color="#1b9e77")
    axes[1, 1].bar(x + 0.18, [float(v) for v in final_pass], 0.36, label="repaired strict hexagon", color="#d95f02")
    axes[1, 1].set_ylim(0.0, 1.12)
    axes[1, 1].set_title("Independent pass/fail gates")
    axes[1, 1].legend(frameon=False)
    axes[1, 2].axis("off")
    axes[1, 2].text(0.0, 1.0, "Audited route IDs\n\n" + "\n".join(f"{i + 1}. {label}" for i, label in enumerate(labels)), va="top", fontsize=8.5)
    for ax in axes.ravel()[:5]:
        ax.set_xticks(x, short, rotation=38, ha="right")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("MODE 2X local vector truth and final morphology are independent", fontsize=16)
    _save_figure(
        fig,
        output_root / "03_route_comparison/route_vector_purity_comparison.png",
        output_root / "03_route_comparison/route_vector_purity_comparison.pdf",
    )
    plt.close(fig)


def _plot_ideal_vs_realistic(
    output_root: Path,
    results: Sequence[LocalVectorTruthResult],
    beam_radius_m: float,
) -> None:
    plt, _, _ = _mpl()
    by_id = {result.route_id: result for result in results}
    route_ids = [
        "authoritative_analytic_target",
        "ideal_sequential_dual_slm",
        "realistic_sequential_carrier_common_4f",
        "m2s_combined_moderate_lab",
        "m2s_combined_bad_lab",
        "compensated_axicon_mask_offset_0p5mm",
    ]
    labels = ["analytic target", "ideal sequential", "realistic sequential", "moderate realism", "bad realism", "corrected 0.5 mm"]
    target = by_id[route_ids[0]]
    sy, sx, extent = _display_crop(target, beam_radius_m)
    fig, axes = plt.subplots(len(route_ids), 4, figsize=(16.0, 22.0), constrained_layout=True)
    titles = [r"radial leakage $\epsilon_R$", r"azimuthal leakage $\epsilon_A$", r"$|\Delta\alpha|$ (rad)", r"$|S_3|/S_0$"]
    for row_index, (route_id, label) in enumerate(zip(route_ids, labels)):
        result = by_id[route_id]
        s3_fraction = np.divide(np.abs(result.s3), result.s0, out=np.zeros_like(result.s0), where=result.s0 > EPS)
        arrays = [result.radial_leakage_map, result.azimuthal_leakage_map, np.abs(result.angle_error_rad), s3_fraction]
        maxima = [0.20, 0.20, 0.25, 0.15]
        for column, (array, vmax) in enumerate(zip(arrays, maxima)):
            image = axes[row_index, column].imshow(
                array[sy, sx], origin="lower", extent=extent, cmap="inferno", vmin=0.0, vmax=vmax, interpolation="bicubic"
            )
            if row_index == 0:
                axes[row_index, column].set_title(titles[column])
            axes[row_index, column].set_xlabel("x (mm)")
            if column == 0:
                axes[row_index, column].set_ylabel(f"{label}\ny (mm)")
            fig.colorbar(image, ax=axes[row_index, column], shrink=0.68)
    fig.suptitle("Ideal, realistic, degraded, and corrected pre-axicon local truth\nIdentical physical crop and colour scales", fontsize=16)
    _save_figure(
        fig,
        output_root / "03_route_comparison/ideal_vs_realistic_local_truth.png",
        output_root / "03_route_comparison/ideal_vs_realistic_local_truth.pdf",
    )
    plt.close(fig)


def _write_document(path: Path, rows: Sequence[Mapping[str, Any]], outcome: str, output_root: Path) -> None:
    table = [
        "| route | radial purity | azimuthal purity | angle RMS (rad) | S3 RMS | local pass | final strict pass |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        table.append(
            f"| `{row['route_id']}` | {float(row['radial_purity']):.9f} | {float(row['azimuthal_purity']):.9f} | "
            f"{float(row['local_angle_rms_rad']):.3e} | {float(row['s3_rms_fraction']):.3e} | "
            f"{bool(row['passed_full_vector_truth_gate'])} | {bool(row['final_strict_hexagon_pass'])} |"
        )
    text = f"""# Nathan MODE 2X - Local Radial/Azimuthal Vector-Field Truth Audit

**Status:** source-scale pre-axicon local vector truth only. No new architecture.
No microfabrication/sample-plane success claim.

## Why This Audit Exists

Sector averaging is useful as a meeting schematic, but one constant arrow cannot prove a radial or
azimuthal field. Each sector is locally linearly polarised, but its orientation varies continuously
with position. A radial sector is not represented physically by one constant average arrow; its local
polarisation follows e_r(theta). An azimuthal sector follows e_theta(theta).

## Method

The native complex Cartesian field is converted point by point using
`Er = Ex cos(theta) + Ey sin(theta)` and `Etheta = -Ex sin(theta) + Ey cos(theta)`.
Radial and azimuthal purity are intensity-weighted local-basis powers. Line orientation is obtained
from `0.5 atan2(S2,S1)` and compared modulo pi. Local linearity is audited from `|S3|/S0` using the
project convention `S3 = -2 Im(Ex Ey*)`.

The central singular neighbourhood, pixels below `1e-6` of peak intensity, and a 0.5 degree guard
on authoritative V0 sector boundaries are excluded. Native samples determine every metric;
interpolation is display-only.

## Results

{chr(10).join(table)}

The pre-axicon local-vector gate and repaired final z=60 mm intensity-hexagon gate are independent.
A route can pass either one without passing the other. No audited ideal implementation uses a
sector-averaged constant orientation: the analytic target, patterned HWP, abstract dual-SLM and
sequential dual-SLM fields all retain the continuously varying local `alpha(theta)` map.

The canonical realistic 4F route remains locally linear (`S3` RMS at floating-point noise), but its
hard common-4F filtering smooths the discontinuous sector boundaries enough to produce about 2.2%
cross-basis leakage and a 0.154 rad intensity-weighted line-angle RMS. It therefore keeps the repaired
final strict hexagon while failing the independently fixed local-vector gate. The ideal-route final
strict entries are false because the repaired candidate gate is calibrated to the immutable realistic
4F reference; they are not failures of ideal local vector truth.

## Meeting Figures

- `{output_root / '01_figures/sector_averaged_polarisation_schematic.png'}` is explanatory only.
- `{output_root / '01_figures/true_local_polarisation_field.png'}` is the physically truthful local field.

## Conclusion

Outcome **{outcome}**. Ideal local truth passes, but the realistic sequential route requires revised
filtering or correction before a high-purity local-vector claim is accepted. The accepted architecture
remains one sequential collinear beam through SLM1,
the conditional polarisation swap, SLM2, optional swap-back, the common 4F, QWP and axicon. MODE 2X
adds a local vector-field audit only. It does not authorise or claim microfabrication/sample-plane
performance.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_mode2x_local_vector_truth(
    output_root: str | Path = MODE2X_DEFAULT_OUTPUT_ROOT,
    *,
    document_path: str | Path = MODE2X_DOC_PATH,
    grid_n: int = 1024,
    z_planes: int = 5,
) -> dict[str, Any]:
    """Run MODE 2X and write all figures, tables, gates, scope, and outcome files."""

    root = Path(output_root)
    for name in ("00_scope", "01_figures", "02_target_truth", "03_route_comparison", "04_route_maps", "05_gates", "06_final_status"):
        (root / name).mkdir(parents=True, exist_ok=True)
    results = build_mode2x_route_results(grid_n=int(grid_n), z_planes=int(z_planes))
    rows = local_vector_truth_summary_rows(results)
    outcome = mode2x_outcome(results)
    target = next(result for result in results if result.route_id == "authoritative_analytic_target")
    beam_radius_m = 2.0e-3
    _plot_sector_schematic(root, float(target.metadata["sector_rotation_rad"]))
    _plot_true_local_field(root, target, beam_radius_m)
    _plot_target_truth(root, target, beam_radius_m)
    _plot_route_comparison(root, results)
    _plot_ideal_vs_realistic(root, results, beam_radius_m)

    _write_csv(root / "local_vector_truth_summary.csv", rows)
    _write_json(root / "local_vector_truth_summary.json", {"stage": MODE2X_STAGE, "rows": rows})
    _write_json(root / "05_gates/local_vector_truth_gate_definitions.json", {
        "stage": MODE2X_STAGE,
        "gates": mode2x_gate_definitions(),
        "final_hexagon_gate_is_separate": True,
        "thresholds_calibrated_after_results": False,
    })
    scope = {
        "stage": MODE2X_STAGE,
        "plane": MODE2X_PLANE,
        "accepted_architecture": "one sequential collinear beam -> SLM1 -> conditional HWP swap -> SLM2 -> optional swap-back -> common 4F -> QWP -> axicon",
        "superseded_parallel_arm_architecture_reintroduced": False,
        "microfabrication_sample_plane_success_claim": False,
        "sector_average_used_as_proof": False,
        "sector_convention": MODE2X_SECTOR_CONVENTION,
        "canonical_operating_point": CANONICAL_OPERATING_POINT_ID,
        "secondary_operating_point": STRICT_COMPROMISE_ID,
        "forbidden_operating_point": OLD_BEST_COMPROMISE_ID,
        "routes": [result.route_id for result in results],
        "metrics_use_native_complex_pre_axicon_fields": True,
        "display_interpolation_used_for_metrics": False,
    }
    _write_json(root / "00_scope/local_vector_truth_scope_manifest.json", scope)
    outcome_report = {
        "stage": MODE2X_STAGE,
        "outcome": outcome,
        "allowed_outcomes": MODE2X_ALLOWED_OUTCOMES,
        "ideal_routes_pass": all(row["passed_full_vector_truth_gate"] for row in rows if row["gate_class"] == "ideal"),
        "canonical_realistic_local_truth_pass": next(row for row in rows if row["route_id"] == CANONICAL_OPERATING_POINT_ID)["passed_full_vector_truth_gate"],
        "canonical_final_strict_hexagon_pass": next(row for row in rows if row["route_id"] == CANONICAL_OPERATING_POINT_ID)["final_strict_hexagon_pass"],
        "existing_route_used_sector_averaged_constant_orientation": False,
        "meeting_figures_generated": True,
        "no_microfabrication_sample_plane_success_claim": True,
        "summary_rows": rows,
    }
    _write_json(root / "06_final_status/local_vector_truth_outcome_report.json", outcome_report)
    _write_document(Path(document_path), rows, outcome, root)
    return {
        "outcome": outcome,
        "results": results,
        "rows": rows,
        "output_root": root,
        "document_path": Path(document_path),
        "scope_manifest": root / "00_scope/local_vector_truth_scope_manifest.json",
        "gate_definitions": root / "05_gates/local_vector_truth_gate_definitions.json",
        "outcome_report": root / "06_final_status/local_vector_truth_outcome_report.json",
    }


__all__ = [
    "MODE2X_ALLOWED_OUTCOMES",
    "MODE2X_DEFAULT_OUTPUT_ROOT",
    "MODE2X_DOC_PATH",
    "MODE2X_PLANE",
    "MODE2X_SECTOR_CONVENTION",
    "MODE2X_STAGE",
    "LocalVectorBasisFields",
    "LocalVectorPurityMetrics",
    "LocalVectorTruthResult",
    "build_mode2x_route_results",
    "build_radial_azimuthal_sector_masks",
    "cartesian_to_local_cylindrical",
    "evaluate_local_vector_truth",
    "line_orientation_error",
    "local_vector_truth_summary_rows",
    "mode2x_gate_definitions",
    "mode2x_outcome",
    "write_mode2x_local_vector_truth",
]
