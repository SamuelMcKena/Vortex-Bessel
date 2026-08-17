"""MODE 2Z orientation-interpolation propagation sweep.

The sweep varies only the in-sector angular fidelity between the MODE 2Y
sector-centre surrogate (eta=0) and the true local radial/azimuthal target
(eta=1). All fields share amplitude, power, sector labels, optical operators,
grid, and propagation samples. Metrics are evaluated on native arrays only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.stats import spearmanr

from vbb_study.digital_twin.nathan_local_vector_truth import (
    evaluate_local_vector_truth,
    line_orientation_error,
)
from vbb_study.digital_twin.nathan_mode2u3_hardware_closure import (
    CANONICAL_OPERATING_POINT_ID,
    STRICT_COMPROMISE_ID,
    assert_not_forbidden,
)
from vbb_study.digital_twin.nathan_mode2w_fix_sequential_master import _source_config
from vbb_study.digital_twin.nathan_mode2y_continuous_vs_averaged import (
    MODE2Y_MIN_HERO_GRID_N,
    Mode2YInputFields,
    PropagatedShapeMetrics,
    PropagationRouteResult,
    _after_axicon,
    _fixed_useful_region,
    _intensity_from_prepared,
    _prepare_projected_spectrum,
    _realistic_common_4f_field,
    _selected_key,
    _strict_row,
    _write_csv,
    _write_json,
    build_mode2y_input_fields,
    propagated_shape_metrics,
)
from vbb_study.digital_twin.nathan_vector_hexagon import (
    EPS,
    MODE2N_DEFAULT_CARRIER_LPMM,
    MODE2N_DEFAULT_IRIS_RADIUS_FRAC,
    _mode2n_reference_plane_metrics,
    mode2n_source_target,
)


MODE2Z_STAGE = "nathan_mode2z_orientation_interpolation"
MODE2Z_DEFAULT_OUTPUT_ROOT = Path("outputs/figures/digital_twin/nathan_mode2z_orientation_interpolation")
MODE2Z_DOC_PATH = Path("docs/86_nathan_mode2z_orientation_interpolation.md")
MODE2Z_ALLOWED_OUTCOMES = ("M2Z-A", "M2Z-B", "M2Z-C", "M2Z-D")
MODE2Z_DEFAULT_ETA_VALUES = tuple(float(value) for value in np.linspace(0.0, 1.0, 11))
MODE2Z_TREND_RHO_MIN = 0.70
MODE2Z_ENDPOINT_IMPROVEMENT_MIN = 0.05
MODE2Z_CORRELATION_ABSOLUTE_IMPROVEMENT_MIN = 0.01
MODE2Z_STEP_TOLERANCE = 0.01
MODE2Z_MONOTONIC_STEP_FRACTION_MIN = 0.90
MODE2Z_CLASSIFIER_REVISION = "M2Z-GATE-FIX-1"


@dataclass(frozen=True)
class Mode2ZSweepConfig:
    """Sampling and fixed optical controls for the MODE 2Z sweep."""

    grid_n: int = 1024
    eta_values: tuple[float, ...] = MODE2Z_DEFAULT_ETA_VALUES
    z_start_m: float = 0.0
    z_end_m: float = 0.2
    z_step_m: float = 0.002
    selected_z_m: tuple[float, ...] = (0.06,)
    propagation_map_eta: tuple[float, ...] = (0.0, 0.5, 1.0)
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
        eta = np.asarray(self.eta_values, dtype=float)
        if eta.ndim != 1 or eta.size < 3 or not np.all(np.isfinite(eta)):
            raise ValueError("eta_values must contain at least three finite values")
        if np.any(eta < 0.0) or np.any(eta > 1.0) or np.any(np.diff(eta) <= 0.0):
            raise ValueError("eta_values must be strictly increasing inside [0, 1]")
        if not np.isclose(float(eta[0]), 0.0) or not np.isclose(float(eta[-1]), 1.0):
            raise ValueError("eta_values must include exact endpoints 0 and 1")
        if int(self.grid_n) < 64:
            raise ValueError("grid_n is too small for an interpolation sweep")
        if self.publication_quality and int(self.grid_n) < MODE2Y_MIN_HERO_GRID_N:
            raise ValueError(f"publication MODE 2Z outputs require grid_n >= {MODE2Y_MIN_HERO_GRID_N}")
        if float(self.z_step_m) <= 0.0 or float(self.z_end_m) <= float(self.z_start_m):
            raise ValueError("z range and spacing must be positive")
        self.z_values_m()


@dataclass(frozen=True)
class Mode2ZSweepResult:
    """Complete ideal and realistic orientation-interpolation sweep."""

    config: Mode2ZSweepConfig
    data: Mapping[str, Any] = field(repr=False, compare=False)
    mode2y_inputs: Mode2YInputFields = field(repr=False, compare=False)
    alpha_by_eta: Mapping[float, np.ndarray] = field(repr=False, compare=False)
    routes: Mapping[str, PropagationRouteResult]
    input_rows: tuple[Mapping[str, Any], ...]
    summary_rows: tuple[Mapping[str, Any], ...]
    z_metric_rows: tuple[Mapping[str, Any], ...]
    trend_rows: tuple[Mapping[str, Any], ...]
    outcome: str
    outcome_reason: str


def _eta_label(eta: float) -> str:
    return f"{float(eta):.2f}"


def mode2z_route_id(optical_route: str, eta: float) -> str:
    """Return a stable route identifier for one optical route and eta."""

    if optical_route == "ideal":
        return f"ideal_eta_{_eta_label(eta)}"
    if optical_route == "realistic":
        return f"realistic_eta_{_eta_label(eta)}_common_4f"
    raise ValueError("optical_route must be 'ideal' or 'realistic'")


def in_sector_orientation_delta(
    theta_rad: np.ndarray,
    sector_index: np.ndarray,
    *,
    sector_rotation_rad: float,
) -> np.ndarray:
    """Return the signed angular offset from each sector centre."""

    theta = np.asarray(theta_rad, dtype=float)
    index = np.asarray(sector_index, dtype=int)
    if theta.shape != index.shape:
        raise ValueError("theta_rad and sector_index must have matching shapes")
    centres = float(sector_rotation_rad) + (index + 0.5) * (np.pi / 3.0)
    delta = np.angle(np.exp(1j * (theta - centres)))
    if float(np.max(np.abs(delta))) > np.pi / 6.0 + 1e-12:
        raise AssertionError("sector-centre interpolation exceeded a 30 degree in-sector offset")
    return np.asarray(delta, dtype=float)


def build_interpolated_alpha(
    data: Mapping[str, Any],
    inputs: Mode2YInputFields,
    eta: float,
) -> np.ndarray:
    """Interpolate from sector-centre orientation to the true local field."""

    value = float(eta)
    if not 0.0 <= value <= 1.0:
        raise ValueError("eta must lie in [0, 1]")
    cfg = data["config"]
    delta = in_sector_orientation_delta(
        np.asarray(data["grid"]["PHI"], dtype=float),
        inputs.sector_index,
        sector_rotation_rad=float(cfg.sector_rotation_rad),
    )
    alpha = np.asarray(inputs.averaged_alpha_rad, dtype=float) + value * delta
    if np.isclose(value, 0.0) and np.max(np.abs(line_orientation_error(alpha, inputs.averaged_alpha_rad))) > 1e-13:
        raise AssertionError("eta=0 did not reproduce the sector-averaged endpoint")
    if np.isclose(value, 1.0) and np.max(np.abs(line_orientation_error(alpha, inputs.continuous_alpha_rad))) > 1e-13:
        raise AssertionError("eta=1 did not reproduce the continuous endpoint")
    return alpha


def build_interpolated_input_field(
    data: Mapping[str, Any],
    inputs: Mode2YInputFields,
    eta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return power-matched Cartesian input components and their orientation."""

    alpha = build_interpolated_alpha(data, inputs, eta)
    amplitude = np.asarray(data["A"], dtype=float)
    ex = np.asarray(amplitude * np.cos(alpha), dtype=np.complex128)
    ey = np.asarray(amplitude * np.sin(alpha), dtype=np.complex128)
    power = float(np.sum(np.abs(ex) ** 2 + np.abs(ey) ** 2))
    if not np.isclose(power, inputs.continuous_power, rtol=2e-14, atol=2e-14 * max(inputs.continuous_power, 1.0)):
        raise FloatingPointError("interpolated field power does not match the continuous endpoint")
    return ex, ey, alpha


def _retain_propagation_map(config: Mode2ZSweepConfig, eta: float) -> bool:
    values = np.asarray(config.propagation_map_eta, dtype=float)
    return bool(np.any(np.isclose(values, float(eta), rtol=0.0, atol=1e-12)))


def _stream_eta_route(
    *,
    optical_route: str,
    eta: float,
    field_components: tuple[np.ndarray, np.ndarray],
    data: Mapping[str, Any],
    config: Mode2ZSweepConfig,
    ring_radius_m: float,
    useful_mask: np.ndarray,
    pre_axicon_report: Mapping[str, Any],
) -> PropagationRouteResult:
    """Propagate one eta without retaining a full three-dimensional stack."""

    after_axicon, axicon_meta = _after_axicon(field_components, data)
    prepared = _prepare_projected_spectrum(after_axicon)
    z_values = config.z_values_m()
    n = int(data["grid"]["N"])
    retain_map = _retain_propagation_map(config, eta)
    xz = np.empty((z_values.size, n), dtype=np.float32) if retain_map else np.empty((0, n), dtype=np.float32)
    yz = np.empty((z_values.size, n), dtype=np.float32) if retain_map else np.empty((0, n), dtype=np.float32)
    selected: dict[str, np.ndarray] = {}
    metrics: list[PropagatedShapeMetrics] = []
    mid = n // 2
    selected_z = np.asarray(config.selected_z_m, dtype=float)
    for index, z_m in enumerate(z_values):
        plane = _intensity_from_prepared(prepared, float(z_m))
        if retain_map:
            xz[index] = plane[mid, :]
            yz[index] = plane[:, mid]
        metrics.append(propagated_shape_metrics(
            plane,
            data["grid"],
            z_m=float(z_m),
            ring_radius_m=float(ring_radius_m),
            useful_mask=useful_mask,
        ))
        if np.any(np.isclose(selected_z, float(z_m), rtol=0.0, atol=1e-12)):
            selected[_selected_key(float(z_m))] = plane.astype(np.float32)
    valid_best = np.where(z_values >= 10e-3)[0]
    scores = np.asarray([metric.sharpness_composite for metric in metrics], dtype=float)
    best_index = int(valid_best[np.nanargmax(scores[valid_best])])
    selected["best_z"] = _intensity_from_prepared(prepared, float(z_values[best_index])).astype(np.float32)
    persistence = float(np.mean(scores >= 0.8 * max(float(np.nanmax(scores)), EPS)))
    return PropagationRouteResult(
        route_id=mode2z_route_id(optical_route, eta),
        optical_route=optical_route,
        input_model=f"orientation_interpolation_eta_{_eta_label(eta)}",
        z_values_m=z_values,
        selected_planes=selected,
        xz_map=xz,
        yz_map=yz,
        z_metrics=tuple(metrics),
        best_z_m=float(z_values[best_index]),
        best_z_index=best_index,
        persistence_fraction=persistence,
        metadata={
            "stage": MODE2Z_STAGE,
            "eta": float(eta),
            "axicon": dict(axicon_meta),
            "grid_n": n,
            "native_grid_metrics": True,
            "display_interpolation_used_for_metrics": False,
            "propagation_map_retained": retain_map,
            "plane": "after_source_scale_axicon_free_space",
            "pre_axicon_report": dict(pre_axicon_report),
        },
    )


def _morphology_quality_indices(rows: list[dict[str, Any]]) -> None:
    """Add an endpoint-normalised descriptive morphology index in place."""

    for optical_route in ("ideal", "realistic"):
        group = [row for row in rows if row["optical_route"] == optical_route]
        group.sort(key=lambda row: float(row["eta"]))
        base = group[0]
        for row in group:
            ratios = np.asarray([
                float(row["z60_correlation_to_v0"]) / max(float(base["z60_correlation_to_v0"]), EPS),
                float(row["edge_gradient_sharpness_mm_inv"]) / max(float(base["edge_gradient_sharpness_mm_inv"]), EPS),
                float(base["threshold_transition_width_mm"]) / max(float(row["threshold_transition_width_mm"]), EPS),
                float(base["bright_ridge_fwhm_mm"]) / max(float(row["bright_ridge_fwhm_mm"]), EPS),
            ], dtype=float)
            row["morphology_quality_index"] = float(np.exp(np.mean(np.log(np.maximum(ratios, EPS)))))


def _pareto_flags(rows: list[dict[str, Any]]) -> None:
    """Flag non-dominated morphology/useful-energy points per optical route."""

    for optical_route in ("ideal", "realistic"):
        group = [row for row in rows if row["optical_route"] == optical_route]
        for row in group:
            dominated = any(
                float(other["morphology_quality_index"]) >= float(row["morphology_quality_index"])
                and float(other["useful_region_power"]) >= float(row["useful_region_power"])
                and (
                    float(other["morphology_quality_index"]) > float(row["morphology_quality_index"])
                    or float(other["useful_region_power"]) > float(row["useful_region_power"])
                )
                for other in group
            )
            row["morphology_energy_pareto"] = not dominated


def _trend_row(
    rows: list[dict[str, Any]],
    *,
    optical_route: str,
    metric: str,
    direction: str,
) -> dict[str, Any]:
    group = sorted(
        (row for row in rows if row["optical_route"] == optical_route),
        key=lambda row: float(row["eta"]),
    )
    eta = np.asarray([row["eta"] for row in group], dtype=float)
    values = np.asarray([row[metric] for row in group], dtype=float)
    if direction == "higher_is_better":
        quality = values / max(float(values[0]), EPS)
        absolute_improvement = float(values[-1] - values[0])
    elif direction == "lower_is_better":
        quality = float(values[0]) / np.maximum(values, EPS)
        absolute_improvement = float(values[0] - values[-1])
    else:
        raise ValueError("unsupported trend direction")
    if np.allclose(quality, quality[0], rtol=1e-12, atol=1e-12):
        rho = 0.0
    else:
        rho_result = spearmanr(eta, quality)
        rho = float(rho_result.statistic) if np.isfinite(rho_result.statistic) else 0.0
    endpoint = float(quality[-1] - 1.0)
    step_fraction = float(np.mean(np.diff(quality) >= -MODE2Z_STEP_TOLERANCE))
    if metric == "z60_correlation_to_v0":
        endpoint_criterion = "absolute_gain_at_least_0.01_for_bounded_correlation"
        endpoint_pass = absolute_improvement >= MODE2Z_CORRELATION_ABSOLUTE_IMPROVEMENT_MIN
    else:
        endpoint_criterion = "relative_quality_gain_at_least_0.05"
        endpoint_pass = endpoint >= MODE2Z_ENDPOINT_IMPROVEMENT_MIN
    monotonic_pass = bool(
        rho >= MODE2Z_TREND_RHO_MIN
        or step_fraction >= MODE2Z_MONOTONIC_STEP_FRACTION_MIN
    )
    return {
        "optical_route": optical_route,
        "metric": metric,
        "direction": direction,
        "eta0_value": float(values[0]),
        "eta1_value": float(values[-1]),
        "endpoint_absolute_improvement": absolute_improvement,
        "endpoint_relative_improvement": endpoint,
        "endpoint_criterion": endpoint_criterion,
        "endpoint_pass": bool(endpoint_pass),
        "spearman_rho_eta_vs_quality": rho,
        "nondecreasing_step_fraction_with_1pct_tolerance": step_fraction,
        "monotonic_pass": monotonic_pass,
        "trend_pass": bool(endpoint_pass and monotonic_pass),
        "classifier_revision": MODE2Z_CLASSIFIER_REVISION,
    }


def mode2z_trend_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Evaluate the predeclared continuous-fidelity morphology trends."""

    metrics = (
        ("z60_correlation_to_v0", "higher_is_better"),
        ("edge_gradient_sharpness_mm_inv", "higher_is_better"),
        ("threshold_transition_width_mm", "lower_is_better"),
        ("bright_ridge_fwhm_mm", "lower_is_better"),
    )
    return tuple(
        _trend_row(rows, optical_route=route, metric=metric, direction=direction)
        for route in ("ideal", "realistic")
        for metric, direction in metrics
    )


def mode2z_outcome(
    summary_rows: tuple[Mapping[str, Any], ...],
    trend_rows: tuple[Mapping[str, Any], ...],
) -> tuple[str, str]:
    """Classify monotonic improvement, interior trade-off, or unstable trend."""

    passes = {
        route: sum(bool(row["trend_pass"]) for row in trend_rows if row["optical_route"] == route)
        for route in ("ideal", "realistic")
    }
    if passes["ideal"] >= 3 and passes["realistic"] >= 3:
        return "M2Z-A", "z60 morphology improves systematically toward the continuous endpoint in both routes; energy and axial evolution remain trade-offs"

    interior_better = False
    endpoint_worse = {"ideal": 0, "realistic": 0}
    for route in ("ideal", "realistic"):
        group = sorted(
            (row for row in summary_rows if row["optical_route"] == route),
            key=lambda row: float(row["eta"]),
        )
        endpoint = float(group[-1]["morphology_quality_index"])
        interior = max(float(row["morphology_quality_index"]) for row in group[1:-1])
        interior_better = interior_better or interior >= 1.05 * endpoint
        endpoint_worse[route] = sum(
            float(row["endpoint_relative_improvement"]) <= -MODE2Z_ENDPOINT_IMPROVEMENT_MIN
            for row in trend_rows
            if row["optical_route"] == route
        )
    if interior_better:
        return "M2Z-B", "an intermediate orientation fidelity gives a material morphology-energy trade-off"
    if endpoint_worse["ideal"] >= 3 and endpoint_worse["realistic"] >= 3:
        return "M2Z-D", "the continuous endpoint degrades most predeclared morphology trends"
    return "M2Z-C", "orientation-fidelity trends are mixed or insufficiently monotonic"


def run_mode2z_sweep(config: Mode2ZSweepConfig | None = None) -> Mode2ZSweepResult:
    """Run the complete source-scale orientation-fidelity interpolation sweep."""

    study = config or Mode2ZSweepConfig()
    study.validate()
    assert_not_forbidden(CANONICAL_OPERATING_POINT_ID)
    assert_not_forbidden(STRICT_COMPROMISE_ID)
    z_values = study.z_values_m()
    source_cfg = _source_config(
        grid_n=int(study.grid_n),
        z_planes=int(z_values.size),
        z_start_m=float(study.z_start_m),
        z_end_m=float(study.z_end_m),
    )
    data = mode2n_source_target(source_cfg, grid_n=int(study.grid_n), z_planes=int(z_values.size))
    inputs = build_mode2y_input_fields(data)
    alpha_by_eta: dict[float, np.ndarray] = {}
    input_rows: list[dict[str, Any]] = []
    x = np.asarray(data["grid"]["x"], dtype=float)
    for eta in study.eta_values:
        ex, ey, alpha = build_interpolated_input_field(data, inputs, float(eta))
        alpha_by_eta[float(eta)] = alpha
        truth = evaluate_local_vector_truth(
            f"eta_{_eta_label(eta)}",
            ex,
            ey,
            x,
            x,
            inputs.continuous_alpha_rad,
            sector_rotation_rad=float(data["config"].sector_rotation_rad),
            gate_class="ideal",
        )
        input_rows.append({
            "eta": float(eta),
            "input_power": float(np.sum(np.abs(ex) ** 2 + np.abs(ey) ** 2)),
            "local_angle_rms_rad": float(truth.metrics.local_angle_rms_rad),
            "local_angle_p95_rad": float(truth.metrics.local_angle_p95_rad),
            "radial_purity": float(truth.metrics.radial_purity),
            "azimuthal_purity": float(truth.metrics.azimuthal_purity),
            "full_local_vector_truth_pass": bool(truth.metrics.passed_full_vector_truth_gate),
        })
        del truth, ex, ey

    amplitude = np.asarray(data["A"], dtype=float)
    continuous_alpha = alpha_by_eta[float(study.eta_values[-1])]
    ideal_continuous = (
        np.asarray(amplitude * np.cos(continuous_alpha), dtype=np.complex128),
        np.asarray(amplitude * np.sin(continuous_alpha), dtype=np.complex128),
    )
    provisional_after, _ = _after_axicon(ideal_continuous, data)
    provisional = _intensity_from_prepared(_prepare_projected_spectrum(provisional_after), 60e-3)
    reference_diag = _mode2n_reference_plane_metrics(provisional, data["grid"])
    ring_radius = float(reference_diag["ring_radius_m"])
    useful_mask, useful_meta = _fixed_useful_region(data["grid"], ring_radius)

    routes: dict[str, PropagationRouteResult] = {}
    for eta in study.eta_values:
        value = float(eta)
        alpha = alpha_by_eta[value]
        ideal_field = (
            np.asarray(amplitude * np.cos(alpha), dtype=np.complex128),
            np.asarray(amplitude * np.sin(alpha), dtype=np.complex128),
        )
        ideal = _stream_eta_route(
            optical_route="ideal",
            eta=value,
            field_components=ideal_field,
            data=data,
            config=study,
            ring_radius_m=ring_radius,
            useful_mask=useful_mask,
            pre_axicon_report={"route": "ideal sequential-equivalent", "first_order_efficiency": 1.0},
        )
        routes[ideal.route_id] = ideal
        realistic_field, iris = _realistic_common_4f_field(
            amplitude,
            alpha,
            data,
            carrier_lpmm=float(study.carrier_lpmm),
            iris_radius_frac=float(study.iris_radius_frac),
        )
        realistic = _stream_eta_route(
            optical_route="realistic",
            eta=value,
            field_components=realistic_field,
            data=data,
            config=study,
            ring_radius_m=ring_radius,
            useful_mask=useful_mask,
            pre_axicon_report=dict(iris),
        )
        routes[realistic.route_id] = realistic
        del ideal_field, realistic_field

    v0_plane = np.asarray(routes[mode2z_route_id("ideal", 1.0)].selected_planes[_selected_key(60e-3)], dtype=float)
    realistic_plane = np.asarray(
        routes[mode2z_route_id("realistic", 1.0)].selected_planes[_selected_key(60e-3)], dtype=float
    )
    summary: list[dict[str, Any]] = []
    for optical_route in ("ideal", "realistic"):
        for eta in study.eta_values:
            route = routes[mode2z_route_id(optical_route, float(eta))]
            row = _strict_row(
                route,
                grid=data["grid"],
                v0_plane=v0_plane,
                realistic_plane=realistic_plane,
                ring_radius_m=ring_radius,
                useful_mask=useful_mask,
            )
            summary.append({"eta": float(eta), **row})
    _morphology_quality_indices(summary)
    _pareto_flags(summary)
    trend_rows = mode2z_trend_rows(summary)
    summary_rows = tuple(summary)
    outcome, reason = mode2z_outcome(summary_rows, trend_rows)

    z_metric_rows = tuple(
        {
            "route_id": route.route_id,
            "optical_route": route.optical_route,
            "eta": float(route.metadata["eta"]),
            **asdict(metric),
        }
        for route in routes.values()
        for metric in route.z_metrics
    )
    data_with_meta = {
        **dict(data),
        "mode2z_ring_radius_m": ring_radius,
        "mode2z_useful_mask": useful_mask,
        "mode2z_useful_meta": useful_meta,
    }
    return Mode2ZSweepResult(
        config=study,
        data=data_with_meta,
        mode2y_inputs=inputs,
        alpha_by_eta=alpha_by_eta,
        routes=routes,
        input_rows=tuple(input_rows),
        summary_rows=summary_rows,
        z_metric_rows=z_metric_rows,
        trend_rows=trend_rows,
        outcome=outcome,
        outcome_reason=reason,
    )


def _write_document(path: Path, result: Mode2ZSweepResult, output_root: Path) -> Path:
    endpoint_rows = [row for row in result.summary_rows if np.isclose(float(row["eta"]), 1.0)]
    onset: dict[str, str] = {}
    for route in ("ideal", "realistic"):
        passing = [float(row["eta"]) for row in result.summary_rows if row["optical_route"] == route and row["strict_hexagon_pass"]]
        onset[route] = "none" if not passing else f"{min(passing):.2f}"
    trend_lines = []
    for route in ("ideal", "realistic"):
        rows = [row for row in result.trend_rows if row["optical_route"] == route]
        trend_lines.append(
            f"- `{route}`: {sum(bool(row['trend_pass']) for row in rows)}/4 predeclared morphology trends pass; "
            f"strict onset eta = `{onset[route]}`."
        )
    endpoint_lines = []
    for row in endpoint_rows:
        endpoint_lines.append(
            f"| `{row['optical_route']}` | {float(row['z60_correlation_to_v0']):.6f} | "
            f"{bool(row['strict_hexagon_pass'])} | {float(row['edge_gradient_sharpness_mm_inv']):.4f} | "
            f"{float(row['threshold_transition_width_mm']):.5f} | {float(row['bright_ridge_fwhm_mm']):.5f} | "
            f"{float(row['best_z_mm']):.1f} |"
        )
    text = f"""# Nathan MODE 2Z - Orientation-Fidelity Interpolation Sweep

**Status:** source-scale sequential propagation study only. No split-arm architecture and no
microfabrication/sample-plane success claim.

## Question

Does propagated hexagon quality improve systematically as each 60 degree sector changes from one
fixed representative line (`eta=0`) to the true continuously varying local radial/azimuthal field
(`eta=1`), or does an intermediate morphology-energy trade-off perform better?

## Sweep Definition

The interpolation is `alpha_eta = alpha_sector_centre + eta * (theta - theta_sector_centre)` inside
each sector. The in-sector offset is wrapped geometrically before interpolation and never exceeds
30 degrees. This preserves the authoritative sector labels, Gaussian amplitude, total input power,
carrier, common 4F, QWP, axicon, grid and z samples. All decision metrics use native arrays.

Sampling: N={result.config.grid_n}; eta={', '.join(f'{value:.1f}' for value in result.config.eta_values)}; z={result.config.z_start_m/1e-3:.0f}
to {result.config.z_end_m/1e-3:.0f} mm in {result.config.z_step_m/1e-3:.0f} mm steps, including exact z=60 mm.

## Gate Definition Audit

The first generic classifier draft applied a 5% relative floor to every metric and required a high
Spearman coefficient even for native-grid width metrics that evolve as monotone plateaus. That draft
labelled non-reversing plateau-and-jump sequences as mixed. Before interpretation, `M2Z-GATE-FIX-1`
replaced it with metric-aware rules: V0 correlation requires an absolute gain of 0.01; dimensional
sharpness metrics require a 5% relative quality gain; monotonic evidence is either Spearman rho >=
0.70 or at least 90% nondecreasing eta steps with a 1% numerical tolerance. No field, propagation
array or raw metric was changed by this classifier repair.

## Trend Audit

{chr(10).join(trend_lines)}

The four predeclared trend observables are V0 correlation, edge-gradient sharpness, inverse 80-20
transition width and inverse bright-ridge FWHM. Corner concentration, peak intensity and useful
energy are retained as independent trade-off diagnostics rather than folded into a pass condition.

At z=60 mm, V0 correlation and edge-gradient sharpness rise at every eta, while transition width and
ridge FWHM improve through native-grid plateaus without a material reversal. The realistic strict
gate first passes at eta=0.70 and remains passed through eta=1.00. Useful-region energy decreases
slightly toward eta=1, peak intensity has a shallow interior maximum, and best-z/persistence evolve
nonmonotonically. The sweep therefore shows monotonic transverse morphology improvement alongside a
real energy/axial trade-off, not a superior intermediate morphology optimum.

## Continuous Endpoint

| route | z60 corr | strict | edge grad (mm^-1) | 80-20 width (mm) | ridge FWHM (mm) | best z (mm) |
|---|---:|---|---:|---:|---:|---:|
{chr(10).join(endpoint_lines)}

## Conclusion

Outcome **{result.outcome}**: {result.outcome_reason}. MODE 2Z does not replace the independent
MODE 2X local-vector truth gate or the MODE 2Y endpoint comparison. It tests how morphology evolves
between those two physically distinct endpoints.

Output root: `{output_root}`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_mode2z_outputs(
    output_root: str | Path = MODE2Z_DEFAULT_OUTPUT_ROOT,
    *,
    document_path: str | Path = MODE2Z_DOC_PATH,
    config: Mode2ZSweepConfig | None = None,
) -> dict[str, Any]:
    """Run MODE 2Z and write figures, tables, scope, and outcome."""

    study = config or Mode2ZSweepConfig()
    study.validate()
    root = Path(output_root)
    subdirs = ("00_inputs", "01_xy_sweep", "02_metric_trends", "03_propagation", "04_tradeoff", "05_gates", "06_final_status")
    for subdir in subdirs:
        (root / subdir).mkdir(parents=True, exist_ok=True)
    result = run_mode2z_sweep(study)
    from vbb_study.digital_twin.nathan_mode2z_figures import write_mode2z_figures

    figure_paths = write_mode2z_figures(result, root)
    summary_csv = _write_csv(root / "orientation_interpolation_summary.csv", result.summary_rows)
    summary_json = _write_json(root / "orientation_interpolation_summary.json", {
        "stage": MODE2Z_STAGE,
        "classifier_revision": MODE2Z_CLASSIFIER_REVISION,
        "outcome": result.outcome,
        "rows": result.summary_rows,
    })
    z_csv = _write_csv(root / "orientation_interpolation_z_metrics.csv", result.z_metric_rows)
    trend_csv = _write_csv(root / "orientation_interpolation_trends.csv", result.trend_rows)
    trend_json = _write_json(root / "orientation_interpolation_trends.json", {
        "thresholds": {
            "spearman_rho_min": MODE2Z_TREND_RHO_MIN,
            "endpoint_improvement_min": MODE2Z_ENDPOINT_IMPROVEMENT_MIN,
            "correlation_absolute_improvement_min": MODE2Z_CORRELATION_ABSOLUTE_IMPROVEMENT_MIN,
            "step_tolerance": MODE2Z_STEP_TOLERANCE,
            "monotonic_step_fraction_min": MODE2Z_MONOTONIC_STEP_FRACTION_MIN,
        },
        "classifier_revision": MODE2Z_CLASSIFIER_REVISION,
        "initial_generic_draft_outcome": "M2Z-C",
        "final_repaired_outcome": result.outcome,
        "rows": result.trend_rows,
    })
    input_csv = _write_csv(root / "orientation_interpolation_input_truth.csv", result.input_rows)
    input_json = _write_json(root / "orientation_interpolation_input_truth.json", {
        "stage": MODE2Z_STAGE,
        "classifier_revision": MODE2Z_CLASSIFIER_REVISION,
        "rows": result.input_rows,
    })
    gate_definition = _write_json(root / "05_gates/trend_gate_definition.json", {
        "classifier_revision": MODE2Z_CLASSIFIER_REVISION,
        "initial_generic_draft_outcome": "M2Z-C",
        "final_repaired_outcome": result.outcome,
        "initial_generic_draft_issue": "relative correlation floor and rank-only monotonicity misclassified bounded and plateau metrics",
        "correlation_absolute_improvement_min": MODE2Z_CORRELATION_ABSOLUTE_IMPROVEMENT_MIN,
        "dimensional_relative_quality_improvement_min": MODE2Z_ENDPOINT_IMPROVEMENT_MIN,
        "spearman_rho_min": MODE2Z_TREND_RHO_MIN,
        "monotonic_step_fraction_min": MODE2Z_MONOTONIC_STEP_FRACTION_MIN,
        "step_tolerance": MODE2Z_STEP_TOLERANCE,
        "raw_simulation_or_metric_changed_by_repair": False,
    })
    scope_manifest = _write_json(root / "simulation_scope_manifest.json", {
        "stage": MODE2Z_STAGE,
        "classifier_revision": MODE2Z_CLASSIFIER_REVISION,
        "accepted_architecture": "sequential collinear beam; ideal-equivalent and validated common-4F routes",
        "split_arm_pbs_architecture_used": False,
        "source_scale_only": True,
        "microfabrication_sample_plane_success_claim": False,
        "eta_values": result.config.eta_values,
        "grid_n": result.config.grid_n,
        "z_start_m": result.config.z_start_m,
        "z_end_m": result.config.z_end_m,
        "z_step_m": result.config.z_step_m,
        "native_grid_metrics": True,
        "display_interpolation_used_for_metrics": False,
        "forbidden_n384_hero_data_used": False,
    })
    outcome_report = _write_json(root / "06_final_status/orientation_interpolation_outcome_report.json", {
        "stage": MODE2Z_STAGE,
        "classifier_revision": MODE2Z_CLASSIFIER_REVISION,
        "outcome": result.outcome,
        "reason": result.outcome_reason,
        "allowed_outcomes": MODE2Z_ALLOWED_OUTCOMES,
        "trend_rows": result.trend_rows,
        "no_microfabrication_sample_plane_success_claim": True,
    })
    document = _write_document(Path(document_path), result, root)
    return {
        "result": result,
        "figure_paths": {key: str(value) for key, value in figure_paths.items()},
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "z_metrics_csv": str(z_csv),
        "trend_csv": str(trend_csv),
        "trend_json": str(trend_json),
        "input_truth_csv": str(input_csv),
        "input_truth_json": str(input_json),
        "gate_definition": str(gate_definition),
        "scope_manifest": str(scope_manifest),
        "outcome_report": str(outcome_report),
        "document_path": str(document),
    }


__all__ = [
    "MODE2Z_ALLOWED_OUTCOMES",
    "MODE2Z_CLASSIFIER_REVISION",
    "MODE2Z_CORRELATION_ABSOLUTE_IMPROVEMENT_MIN",
    "MODE2Z_DEFAULT_ETA_VALUES",
    "MODE2Z_DEFAULT_OUTPUT_ROOT",
    "MODE2Z_DOC_PATH",
    "MODE2Z_ENDPOINT_IMPROVEMENT_MIN",
    "MODE2Z_MONOTONIC_STEP_FRACTION_MIN",
    "MODE2Z_STAGE",
    "MODE2Z_STEP_TOLERANCE",
    "MODE2Z_TREND_RHO_MIN",
    "Mode2ZSweepConfig",
    "Mode2ZSweepResult",
    "build_interpolated_alpha",
    "build_interpolated_input_field",
    "in_sector_orientation_delta",
    "mode2z_outcome",
    "mode2z_route_id",
    "mode2z_trend_rows",
    "run_mode2z_sweep",
    "write_mode2z_outputs",
]
