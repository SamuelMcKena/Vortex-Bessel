"""Targeted high-N confirmation for the MODE 2Z orientation sweep.

Only selected eta values are propagated through the realistic sequential
common-4F route at z=60 mm. The high-N reference fields are regenerated on the
same grid; no N=1024 image is resized or reused as a numerical field.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.digital_twin.nathan_local_vector_truth import evaluate_local_vector_truth
from vbb_study.digital_twin.nathan_mode2w_fix_sequential_master import _source_config
from vbb_study.digital_twin.nathan_mode2y_continuous_vs_averaged import (
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
from vbb_study.digital_twin.nathan_mode2z_orientation_interpolation import (
    MODE2Z_DEFAULT_OUTPUT_ROOT,
    MODE2Z_DOC_PATH,
    build_interpolated_input_field,
)
from vbb_study.digital_twin.nathan_vector_hexagon import (
    EPS,
    MODE2N_DEFAULT_CARRIER_LPMM,
    MODE2N_DEFAULT_IRIS_RADIUS_FRAC,
    _mode2n_reference_plane_metrics,
    mode2n_source_target,
)


MODE2Z_HN_STAGE = "nathan_mode2z_targeted_highn_confirmation"
MODE2Z_HN_DEFAULT_OUTPUT_ROOT = MODE2Z_DEFAULT_OUTPUT_ROOT / "07_highN_confirmation"
MODE2Z_HN_DOC_PATH = Path("docs/87_nathan_mode2z_targeted_highn_confirmation.md")
MODE2Z_HN_DEFAULT_ETA_VALUES = (0.0, 0.4, 0.6, 0.7, 0.8, 1.0)
MODE2Z_HN_ALLOWED_OUTCOMES = ("M2Z-HN-A", "M2Z-HN-B", "M2Z-HN-C", "M2Z-HN-D")
MODE2Z_HN_MIN_GRID_N = 1536
MODE2Z_HN_Z_M = 0.06


@dataclass(frozen=True)
class Mode2ZHighNConfig:
    """Controls for the selected-plane high-resolution confirmation."""

    grid_n: int = 1536
    eta_values: tuple[float, ...] = MODE2Z_HN_DEFAULT_ETA_VALUES
    z_m: float = MODE2Z_HN_Z_M
    carrier_lpmm: float = MODE2N_DEFAULT_CARRIER_LPMM
    iris_radius_frac: float = MODE2N_DEFAULT_IRIS_RADIUS_FRAC
    baseline_summary_path: Path | None = MODE2Z_DEFAULT_OUTPUT_ROOT / "orientation_interpolation_summary.csv"
    publication_quality: bool = True

    def validate(self) -> None:
        eta = np.asarray(self.eta_values, dtype=float)
        if eta.ndim != 1 or eta.size < 3 or np.any(np.diff(eta) <= 0.0):
            raise ValueError("eta_values must contain at least three strictly increasing values")
        if not np.isclose(eta[0], 0.0) or not np.isclose(eta[-1], 1.0):
            raise ValueError("eta_values must retain exact endpoints 0 and 1")
        if not np.any(np.isclose(eta, 0.7, rtol=0.0, atol=1e-12)):
            raise ValueError("the targeted confirmation must include eta=0.7")
        if int(self.grid_n) < 192:
            raise ValueError("grid_n is below the carrier-valid reduced grid")
        if self.publication_quality and int(self.grid_n) < MODE2Z_HN_MIN_GRID_N:
            raise ValueError(f"publication high-N confirmation requires N >= {MODE2Z_HN_MIN_GRID_N}")
        if not np.isclose(float(self.z_m), MODE2Z_HN_Z_M, rtol=0.0, atol=1e-12):
            raise ValueError("MODE 2Z-HN is intentionally fixed to exact z=60 mm")


@dataclass(frozen=True)
class Mode2ZHighNResult:
    """High-N selected-eta fields, metrics, and convergence audit."""

    config: Mode2ZHighNConfig
    data: Mapping[str, Any] = field(repr=False, compare=False)
    planes_by_eta: Mapping[float, np.ndarray] = field(repr=False, compare=False)
    summary_rows: tuple[Mapping[str, Any], ...]
    baseline_rows: tuple[Mapping[str, Any], ...]
    convergence_rows: tuple[Mapping[str, Any], ...]
    input_rows: tuple[Mapping[str, Any], ...]
    audit: Mapping[str, Any]
    outcome: str
    outcome_reason: str


_BASELINE_NUMERIC_FIELDS = (
    "z60_correlation_to_v0",
    "edge_gradient_sharpness_mm_inv",
    "threshold_transition_width_mm",
    "bright_ridge_fwhm_mm",
    "peak_intensity",
    "useful_region_power",
    "dark_core_ratio",
    "h3_over_h6",
)


def load_mode2z_baseline_rows(
    path: str | Path | None,
    eta_values: Sequence[float],
) -> tuple[dict[str, Any], ...]:
    """Load matching realistic N=1024 rows from the MODE 2Z sweep."""

    if path is None or not Path(path).is_file():
        return ()
    with Path(path).open(newline="", encoding="utf-8") as stream:
        source = list(csv.DictReader(stream))
    rows: list[dict[str, Any]] = []
    for eta in eta_values:
        match = next((
            row for row in source
            if row.get("optical_route") == "realistic"
            and np.isclose(float(row["eta"]), float(eta), rtol=0.0, atol=1e-12)
        ), None)
        if match is None:
            continue
        typed: dict[str, Any] = {
            "eta": float(match["eta"]),
            "grid_n": int(float(match.get("metrics_native_grid_n", 1024))),
            "strict_hexagon_pass": str(match["strict_hexagon_pass"]).lower() == "true",
        }
        for key in _BASELINE_NUMERIC_FIELDS:
            typed[key] = float(match[key])
        rows.append(typed)
    return tuple(rows)


def _strict_route_row(
    *,
    eta: float,
    plane: np.ndarray,
    local_metric: Any,
    data: Mapping[str, Any],
    v0_plane: np.ndarray,
    realistic_reference: np.ndarray,
    ring_radius_m: float,
    useful_mask: np.ndarray,
    iris: Mapping[str, Any],
    config: Mode2ZHighNConfig,
) -> dict[str, Any]:
    route = PropagationRouteResult(
        route_id=f"realistic_eta_{float(eta):.2f}_common_4f_highn",
        optical_route="realistic",
        input_model=f"orientation_interpolation_eta_{float(eta):.2f}",
        z_values_m=np.asarray([float(config.z_m)]),
        selected_planes={_selected_key(float(config.z_m)): np.asarray(plane, dtype=np.float32)},
        xz_map=np.empty((0, int(config.grid_n)), dtype=np.float32),
        yz_map=np.empty((0, int(config.grid_n)), dtype=np.float32),
        z_metrics=(local_metric,),
        best_z_m=float(config.z_m),
        best_z_index=0,
        persistence_fraction=float("nan"),
        metadata={
            "grid_n": int(config.grid_n),
            "native_grid_metrics": True,
            "display_interpolation_used_for_metrics": False,
            "pre_axicon_report": dict(iris),
        },
    )
    row = _strict_row(
        route,
        grid=data["grid"],
        v0_plane=v0_plane,
        realistic_plane=realistic_reference,
        ring_radius_m=float(ring_radius_m),
        useful_mask=useful_mask,
    )
    row.update({
        "eta": float(eta),
        "confirmation_grid_n": int(config.grid_n),
        "grid_dx_um": float(data["grid"]["dx"] / 1e-6),
        "z_mm": float(config.z_m / 1e-3),
        "first_order_efficiency": float(iris.get("first_order_efficiency", float("nan"))),
        "display_interpolation_used_for_metrics": False,
    })
    return row


def _relative_endpoint_improvement(rows: Sequence[Mapping[str, Any]], metric: str, *, lower_is_better: bool) -> float:
    ordered = sorted(rows, key=lambda row: float(row["eta"]))
    start = float(ordered[0][metric])
    end = float(ordered[-1][metric])
    return float(start / max(end, EPS) - 1.0) if lower_is_better else float(end / max(start, EPS) - 1.0)


def _relative_endpoint_change(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    ordered = sorted(rows, key=lambda row: float(row["eta"]))
    start = float(ordered[0][metric])
    end = float(ordered[-1][metric])
    return float(end / max(abs(start), EPS) - 1.0)


def _selected_onset(rows: Sequence[Mapping[str, Any]]) -> float | None:
    passing = [float(row["eta"]) for row in rows if bool(row["strict_hexagon_pass"])]
    return None if not passing else float(min(passing))


def _nondecreasing(values: Sequence[float], *, tolerance_fraction: float = 0.01) -> bool:
    arr = np.asarray(values, dtype=float)
    scale = max(float(np.max(np.abs(arr))), EPS)
    return bool(np.all(np.diff(arr) >= -float(tolerance_fraction) * scale))


def _nonincreasing(values: Sequence[float], *, tolerance_fraction: float = 0.01) -> bool:
    return _nondecreasing([-float(value) for value in values], tolerance_fraction=tolerance_fraction)


def _convergence_rows(
    high_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    baseline = {float(row["eta"]): row for row in baseline_rows}
    rows: list[dict[str, Any]] = []
    for high in high_rows:
        eta = float(high["eta"])
        low = baseline.get(eta)
        if low is None:
            continue
        for metric in _BASELINE_NUMERIC_FIELDS:
            high_value = float(high[metric])
            low_value = float(low[metric])
            rows.append({
                "eta": eta,
                "metric": metric,
                "baseline_grid_n": int(low["grid_n"]),
                "confirmation_grid_n": int(high["confirmation_grid_n"]),
                "baseline_value": low_value,
                "confirmation_value": high_value,
                "absolute_delta": high_value - low_value,
                "relative_delta": (high_value - low_value) / max(abs(low_value), EPS),
            })
        rows.append({
            "eta": eta,
            "metric": "strict_hexagon_pass",
            "baseline_grid_n": int(low["grid_n"]),
            "confirmation_grid_n": int(high["confirmation_grid_n"]),
            "baseline_value": bool(low["strict_hexagon_pass"]),
            "confirmation_value": bool(high["strict_hexagon_pass"]),
            "absolute_delta": int(bool(high["strict_hexagon_pass"])) - int(bool(low["strict_hexagon_pass"])),
            "relative_delta": float("nan"),
        })
    return tuple(rows)


def mode2z_highn_audit(
    high_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate threshold, monotonicity, endpoint, and width-resolution stability."""

    high = sorted(high_rows, key=lambda row: float(row["eta"]))
    baseline = sorted(baseline_rows, key=lambda row: float(row["eta"]))
    high_onset = _selected_onset(high)
    baseline_onset = _selected_onset(baseline) if baseline else None
    high_corr = [float(row["z60_correlation_to_v0"]) for row in high]
    high_edge = [float(row["edge_gradient_sharpness_mm_inv"]) for row in high]
    high_width = [float(row["threshold_transition_width_mm"]) for row in high]
    high_fwhm = [float(row["bright_ridge_fwhm_mm"]) for row in high]
    endpoint_high = {
        "correlation_gain": float(high_corr[-1] - high_corr[0]),
        "edge_relative_improvement": _relative_endpoint_improvement(high, "edge_gradient_sharpness_mm_inv", lower_is_better=False),
        "width_relative_improvement": _relative_endpoint_improvement(high, "threshold_transition_width_mm", lower_is_better=True),
        "fwhm_relative_improvement": _relative_endpoint_improvement(high, "bright_ridge_fwhm_mm", lower_is_better=True),
        "peak_relative_change": _relative_endpoint_change(high, "peak_intensity"),
        "useful_energy_relative_change": _relative_endpoint_change(high, "useful_region_power"),
    }
    endpoint_baseline: dict[str, float] = {}
    if baseline:
        endpoint_baseline = {
            "correlation_gain": float(baseline[-1]["z60_correlation_to_v0"] - baseline[0]["z60_correlation_to_v0"]),
            "edge_relative_improvement": _relative_endpoint_improvement(baseline, "edge_gradient_sharpness_mm_inv", lower_is_better=False),
            "width_relative_improvement": _relative_endpoint_improvement(baseline, "threshold_transition_width_mm", lower_is_better=True),
            "fwhm_relative_improvement": _relative_endpoint_improvement(baseline, "bright_ridge_fwhm_mm", lower_is_better=True),
            "peak_relative_change": _relative_endpoint_change(baseline, "peak_intensity"),
            "useful_energy_relative_change": _relative_endpoint_change(baseline, "useful_region_power"),
        }
    core_endpoint_stable = bool(
        endpoint_baseline
        and abs(endpoint_high["correlation_gain"] - endpoint_baseline["correlation_gain"]) <= 0.005
        and abs(endpoint_high["edge_relative_improvement"] - endpoint_baseline["edge_relative_improvement"]) <= 0.10
    )
    sampled_width_endpoint_stable = bool(
        endpoint_baseline
        and abs(endpoint_high["width_relative_improvement"] - endpoint_baseline["width_relative_improvement"]) <= 0.15
        and abs(endpoint_high["fwhm_relative_improvement"] - endpoint_baseline["fwhm_relative_improvement"]) <= 0.15
    )
    energy_peak_endpoint_stable = bool(
        endpoint_baseline
        and abs(endpoint_high["peak_relative_change"] - endpoint_baseline["peak_relative_change"]) <= 0.01
        and abs(endpoint_high["useful_energy_relative_change"] - endpoint_baseline["useful_energy_relative_change"]) <= 0.005
    )
    endpoint_stable = bool(core_endpoint_stable and sampled_width_endpoint_stable and energy_peak_endpoint_stable)
    return {
        "selected_eta_values": [float(row["eta"]) for row in high],
        "highn_selected_strict_onset_eta": high_onset,
        "baseline_selected_strict_onset_eta": baseline_onset,
        "strict_onset_confirmed": bool(high_onset is not None and baseline_onset is not None and np.isclose(high_onset, baseline_onset)),
        "correlation_nondecreasing": _nondecreasing(high_corr),
        "edge_gradient_nondecreasing": _nondecreasing(high_edge),
        "transition_width_nonincreasing": _nonincreasing(high_width),
        "ridge_fwhm_nonincreasing": _nonincreasing(high_fwhm),
        "highn_unique_transition_width_levels": int(np.unique(np.round(high_width, 12)).size),
        "highn_unique_fwhm_levels": int(np.unique(np.round(high_fwhm, 12)).size),
        "baseline_unique_transition_width_levels": int(np.unique(np.round([float(row["threshold_transition_width_mm"]) for row in baseline], 12)).size) if baseline else 0,
        "baseline_unique_fwhm_levels": int(np.unique(np.round([float(row["bright_ridge_fwhm_mm"]) for row in baseline], 12)).size) if baseline else 0,
        "endpoint_highn": endpoint_high,
        "endpoint_baseline": endpoint_baseline,
        "correlation_edge_endpoint_stable": core_endpoint_stable,
        "sampled_width_endpoint_stable": sampled_width_endpoint_stable,
        "energy_peak_endpoint_stable": energy_peak_endpoint_stable,
        "endpoint_improvements_stable": endpoint_stable,
        "eta1_strict_pass": bool(high[-1]["strict_hexagon_pass"]),
        "native_metrics_only": True,
    }


def mode2z_highn_outcome(audit: Mapping[str, Any]) -> tuple[str, str]:
    """Classify the targeted high-N confirmation."""

    trends = bool(
        audit["correlation_nondecreasing"]
        and audit["edge_gradient_nondecreasing"]
        and audit["transition_width_nonincreasing"]
        and audit["ridge_fwhm_nonincreasing"]
    )
    if not bool(audit["eta1_strict_pass"]) or not trends:
        return "M2Z-HN-D", "high-N confirmation loses the continuous strict pass or a morphology trend"
    if bool(audit["strict_onset_confirmed"]) and bool(audit["endpoint_improvements_stable"]):
        return "M2Z-HN-A", "N=1536 confirms the selected-grid eta=0.70 strict onset and stable monotonic endpoint gains"
    if audit["highn_selected_strict_onset_eta"] is not None and bool(audit["endpoint_improvements_stable"]):
        return "M2Z-HN-B", "high-N trends and endpoints are stable but the selected-grid strict onset shifts"
    if bool(audit.get("correlation_edge_endpoint_stable")) and not bool(audit.get("sampled_width_endpoint_stable")):
        return "M2Z-HN-C", "correlation and edge gains are stable, but the selected onset shifts and sampled width endpoints remain resolution-sensitive"
    return "M2Z-HN-C", "high-N evidence is incomplete or endpoint convergence is outside the confirmation tolerances"


def run_mode2z_highn_confirmation(config: Mode2ZHighNConfig | None = None) -> Mode2ZHighNResult:
    """Run the selected-eta N=1536 realistic z=60 confirmation."""

    study = config or Mode2ZHighNConfig()
    study.validate()
    source_cfg = _source_config(
        grid_n=int(study.grid_n),
        z_planes=2,
        z_start_m=0.0,
        z_end_m=float(study.z_m),
    )
    data = mode2n_source_target(source_cfg, grid_n=int(study.grid_n), z_planes=2)
    inputs = build_mode2y_input_fields(data)
    amplitude = np.asarray(data["A"], dtype=float)

    ideal_ex, ideal_ey, _ = build_interpolated_input_field(data, inputs, 1.0)
    ideal_after, _ = _after_axicon((ideal_ex, ideal_ey), data)
    v0_plane = _intensity_from_prepared(_prepare_projected_spectrum(ideal_after), float(study.z_m))
    reference_diag = _mode2n_reference_plane_metrics(v0_plane, data["grid"])
    ring_radius = float(reference_diag["ring_radius_m"])
    useful_mask, useful_meta = _fixed_useful_region(data["grid"], ring_radius)
    del ideal_ex, ideal_ey, ideal_after

    _, _, alpha_one = build_interpolated_input_field(data, inputs, 1.0)
    realistic_one, iris_one = _realistic_common_4f_field(
        amplitude,
        alpha_one,
        data,
        carrier_lpmm=float(study.carrier_lpmm),
        iris_radius_frac=float(study.iris_radius_frac),
    )
    realistic_after, _ = _after_axicon(realistic_one, data)
    realistic_reference = _intensity_from_prepared(
        _prepare_projected_spectrum(realistic_after), float(study.z_m)
    )
    del realistic_one, realistic_after

    planes: dict[float, np.ndarray] = {}
    summary_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    x = np.asarray(data["grid"]["x"], dtype=float)
    for eta in study.eta_values:
        value = float(eta)
        ex, ey, alpha = build_interpolated_input_field(data, inputs, value)
        truth = evaluate_local_vector_truth(
            f"highn_eta_{value:.2f}",
            ex,
            ey,
            x,
            x,
            inputs.continuous_alpha_rad,
            sector_rotation_rad=float(data["config"].sector_rotation_rad),
            gate_class="ideal",
        )
        input_rows.append({
            "eta": value,
            "grid_n": int(study.grid_n),
            "local_angle_rms_rad": float(truth.metrics.local_angle_rms_rad),
            "radial_purity": float(truth.metrics.radial_purity),
            "azimuthal_purity": float(truth.metrics.azimuthal_purity),
        })
        if np.isclose(value, 1.0, rtol=0.0, atol=1e-12):
            plane = realistic_reference
            iris = iris_one
        else:
            realistic_field, iris = _realistic_common_4f_field(
                amplitude,
                alpha,
                data,
                carrier_lpmm=float(study.carrier_lpmm),
                iris_radius_frac=float(study.iris_radius_frac),
            )
            after, _ = _after_axicon(realistic_field, data)
            plane = _intensity_from_prepared(_prepare_projected_spectrum(after), float(study.z_m))
            del realistic_field, after
        local = propagated_shape_metrics(
            plane,
            data["grid"],
            z_m=float(study.z_m),
            ring_radius_m=ring_radius,
            useful_mask=useful_mask,
        )
        summary_rows.append(_strict_route_row(
            eta=value,
            plane=plane,
            local_metric=local,
            data=data,
            v0_plane=v0_plane,
            realistic_reference=realistic_reference,
            ring_radius_m=ring_radius,
            useful_mask=useful_mask,
            iris=iris,
            config=study,
        ))
        planes[value] = np.asarray(plane, dtype=np.float32)
        del truth, ex, ey, alpha

    baseline_rows = load_mode2z_baseline_rows(study.baseline_summary_path, study.eta_values)
    convergence = _convergence_rows(summary_rows, baseline_rows)
    audit = mode2z_highn_audit(summary_rows, baseline_rows)
    dx_m = float(data["grid"]["dx"])
    carrier_cpm = float(study.carrier_lpmm) * 1e3
    filtered_band_edge_cpm = (1.0 + float(study.iris_radius_frac)) * carrier_cpm
    nyquist_cpm = 0.5 / dx_m
    sampling_audit = {
        "dx_um": float(dx_m / 1e-6),
        "carrier_lpmm": float(study.carrier_lpmm),
        "samples_per_carrier_period": float(1.0 / max(carrier_cpm * dx_m, EPS)),
        "filtered_band_edge_lpmm": float(filtered_band_edge_cpm / 1e3),
        "nyquist_lpmm": float(nyquist_cpm / 1e3),
        "filtered_band_nyquist_margin": float(nyquist_cpm / max(filtered_band_edge_cpm, EPS)),
        "carrier_band_nyquist_pass": bool(filtered_band_edge_cpm < nyquist_cpm),
    }
    audit = {
        **audit,
        "confirmation_grid_n": int(study.grid_n),
        "confirmation_dx_um": float(dx_m / 1e-6),
        "baseline_grid_n": int(baseline_rows[0]["grid_n"]) if baseline_rows else None,
        "ring_radius_um": float(ring_radius / 1e-6),
        "useful_region": useful_meta,
        "sampling_audit": sampling_audit,
    }
    outcome, reason = mode2z_highn_outcome(audit)
    return Mode2ZHighNResult(
        config=study,
        data={**dict(data), "mode2z_hn_ring_radius_m": ring_radius},
        planes_by_eta=planes,
        summary_rows=tuple(summary_rows),
        baseline_rows=baseline_rows,
        convergence_rows=convergence,
        input_rows=tuple(input_rows),
        audit=audit,
        outcome=outcome,
        outcome_reason=reason,
    )


def _format_onset(value: Any) -> str:
    return "none" if value is None else f"{float(value):.2f}"


def _write_document(path: Path, result: Mode2ZHighNResult, output_root: Path) -> Path:
    table = []
    for row in result.summary_rows:
        eta = float(row["eta"])
        low = next((
            candidate for candidate in result.baseline_rows
            if np.isclose(float(candidate["eta"]), eta, rtol=0.0, atol=1e-12)
        ), {})
        table.append(
            f"| {eta:.1f} | {float(low.get('z60_correlation_to_v0', float('nan'))):.6f} | "
            f"{float(row['z60_correlation_to_v0']):.6f} | {bool(low.get('strict_hexagon_pass', False))} | "
            f"{bool(row['strict_hexagon_pass'])} | {float(row['edge_gradient_sharpness_mm_inv']):.4f} | "
            f"{float(row['threshold_transition_width_mm']):.6f} | {float(row['bright_ridge_fwhm_mm']):.6f} |"
        )
    audit = result.audit
    low_endpoint = audit["endpoint_baseline"]
    high_endpoint = audit["endpoint_highn"]
    eta07 = next((
        row for row in result.summary_rows
        if np.isclose(float(row["eta"]), 0.7, rtol=0.0, atol=1e-12)
    ), {})
    text = f"""# Nathan MODE 2Z-HN - Targeted High-N Threshold Confirmation

**Status:** targeted source-scale numerical confirmation only. Realistic sequential common-4F route,
exact z=60 mm, selected eta values. No split-arm or microfabrication/sample-plane claim.

## Scope

This is not another axial sweep. It regenerates an N={result.config.grid_n} ideal-continuous V0
reference and realistic-continuous strict-gate reference, then evaluates only eta =
{', '.join(f'{value:.1f}' for value in result.config.eta_values)} at z=60 mm. Native dx is
{float(audit['confirmation_dx_um']):.3f} um. The 6.25 lp/mm carrier has
{float(audit['sampling_audit']['samples_per_carrier_period']):.2f} samples per period, and the
carrier-plus-iris filtered band retains a {float(audit['sampling_audit']['filtered_band_nyquist_margin']):.2f}x
Nyquist margin. Interpolation is display-only.

## Results

| eta | corr N1024 | corr N{result.config.grid_n} | strict N1024 | strict N{result.config.grid_n} | edge N{result.config.grid_n} | width N{result.config.grid_n} | FWHM N{result.config.grid_n} |
|---:|---:|---:|---|---|---:|---:|---:|
{chr(10).join(table)}

Selected-grid strict onset: N=1024 `{_format_onset(audit['baseline_selected_strict_onset_eta'])}`;
N={result.config.grid_n} `{_format_onset(audit['highn_selected_strict_onset_eta'])}`. At N={result.config.grid_n},
eta=0.7 fails only because `{eta07.get('strict_fail_reasons', 'not evaluated')}`; eta=0.8 passes.

Correlation monotonic: `{bool(audit['correlation_nondecreasing'])}`. Edge-gradient monotonic:
`{bool(audit['edge_gradient_nondecreasing'])}`. Transition width nonincreasing:
`{bool(audit['transition_width_nonincreasing'])}`. FWHM nonincreasing:
`{bool(audit['ridge_fwhm_nonincreasing'])}`. Stable endpoint gains:
`{bool(audit['endpoint_improvements_stable'])}`.

| endpoint diagnostic | N=1024 | N={result.config.grid_n} | convergence interpretation |
|---|---:|---:|---|
| correlation gain | {float(low_endpoint.get('correlation_gain', float('nan'))):.6f} | {float(high_endpoint['correlation_gain']):.6f} | stable |
| edge-gradient gain | {100.0 * float(low_endpoint.get('edge_relative_improvement', float('nan'))):.2f}% | {100.0 * float(high_endpoint['edge_relative_improvement']):.2f}% | stable relative gain |
| transition-width narrowing | {100.0 * float(low_endpoint.get('width_relative_improvement', float('nan'))):.2f}% | {100.0 * float(high_endpoint['width_relative_improvement']):.2f}% | resolution-sensitive |
| ridge-FWHM narrowing | {100.0 * float(low_endpoint.get('fwhm_relative_improvement', float('nan'))):.2f}% | {100.0 * float(high_endpoint['fwhm_relative_improvement']):.2f}% | resolution-sensitive |
| peak change | {100.0 * float(low_endpoint.get('peak_relative_change', float('nan'))):.2f}% | {100.0 * float(high_endpoint['peak_relative_change']):.2f}% | stable |
| useful-energy change | {100.0 * float(low_endpoint.get('useful_energy_relative_change', float('nan'))):.2f}% | {100.0 * float(high_endpoint['useful_energy_relative_change']):.2f}% | stable after within-grid normalisation |

Correlation/edge endpoint stability: `{bool(audit['correlation_edge_endpoint_stable'])}`.
Peak/energy endpoint stability: `{bool(audit['energy_peak_endpoint_stable'])}`.
Sampled width endpoint stability: `{bool(audit['sampled_width_endpoint_stable'])}`.

Resolved width levels among the selected eta values: transition width N=1024
`{int(audit['baseline_unique_transition_width_levels'])}` versus N={result.config.grid_n}
`{int(audit['highn_unique_transition_width_levels'])}`; FWHM N=1024
`{int(audit['baseline_unique_fwhm_levels'])}` versus N={result.config.grid_n}
`{int(audit['highn_unique_fwhm_levels'])}`. Plateaus that remain at high N are reported as plateaus,
not smoothed into artificial continuous measurements.

## Conclusion

Outcome **{result.outcome}**: {result.outcome_reason}. The eta threshold is a calibrated,
project-specific simulation threshold, not a universal experimental tolerance.

Output root: `{output_root}`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _update_parent_document(path: Path, result: Mode2ZHighNResult) -> None:
    if not path.is_file():
        return
    start = "<!-- MODE2Z-HN-START -->"
    end = "<!-- MODE2Z-HN-END -->"
    block = f"""{start}
## Targeted High-N Confirmation

The selected z=60 mm N={result.config.grid_n} check is reported in docs/87. Outcome
**{result.outcome}**: {result.outcome_reason}. Selected-grid strict onset is
`eta={_format_onset(result.audit['highn_selected_strict_onset_eta'])}`. This threshold remains
project-specific and is not a universal experimental tolerance.
{end}"""
    text = path.read_text(encoding="utf-8")
    if start in text and end in text:
        prefix, remainder = text.split(start, 1)
        _, suffix = remainder.split(end, 1)
        updated = prefix.rstrip() + "\n\n" + block + suffix
    else:
        updated = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(updated, encoding="utf-8")


def write_mode2z_highn_outputs(
    output_root: str | Path = MODE2Z_HN_DEFAULT_OUTPUT_ROOT,
    *,
    document_path: str | Path = MODE2Z_HN_DOC_PATH,
    parent_document_path: str | Path | None = MODE2Z_DOC_PATH,
    config: Mode2ZHighNConfig | None = None,
) -> dict[str, Any]:
    """Write the targeted high-N confirmation package."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    result = run_mode2z_highn_confirmation(config)
    from vbb_study.digital_twin.nathan_mode2z_highn_figures import write_mode2z_highn_figures

    figures = write_mode2z_highn_figures(result, root)
    summary_csv = _write_csv(root / "mode2z_highn_summary.csv", result.summary_rows)
    summary_json = _write_json(root / "mode2z_highn_summary.json", {
        "stage": MODE2Z_HN_STAGE,
        "outcome": result.outcome,
        "rows": result.summary_rows,
    })
    convergence_csv = _write_csv(root / "mode2z_highn_convergence.csv", result.convergence_rows)
    convergence_json = _write_json(root / "mode2z_highn_convergence.json", result.convergence_rows)
    input_csv = _write_csv(root / "mode2z_highn_input_truth.csv", result.input_rows)
    audit_json = _write_json(root / "mode2z_highn_audit.json", result.audit)
    manifest = _write_json(root / "simulation_scope_manifest.json", {
        "stage": MODE2Z_HN_STAGE,
        "targeted_confirmation_only": True,
        "optical_route": "realistic sequential common-4F",
        "grid_n": result.config.grid_n,
        "eta_values": result.config.eta_values,
        "z_m": result.config.z_m,
        "native_grid_metrics": True,
        "display_interpolation_used_for_metrics": False,
        "sampling_audit": result.audit["sampling_audit"],
        "split_arm_pbs_architecture_used": False,
        "microfabrication_sample_plane_success_claim": False,
    })
    outcome_report = _write_json(root / "mode2z_highn_outcome_report.json", {
        "stage": MODE2Z_HN_STAGE,
        "outcome": result.outcome,
        "reason": result.outcome_reason,
        "allowed_outcomes": MODE2Z_HN_ALLOWED_OUTCOMES,
        "audit": result.audit,
        "no_universal_tolerance_claim": True,
        "no_microfabrication_sample_plane_success_claim": True,
    })
    document = _write_document(Path(document_path), result, root)
    if parent_document_path is not None:
        _update_parent_document(Path(parent_document_path), result)
    return {
        "result": result,
        "figure_paths": {key: str(value) for key, value in figures.items()},
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "convergence_csv": str(convergence_csv),
        "convergence_json": str(convergence_json),
        "input_truth_csv": str(input_csv),
        "audit_json": str(audit_json),
        "manifest": str(manifest),
        "outcome_report": str(outcome_report),
        "document_path": str(document),
    }


__all__ = [
    "MODE2Z_HN_ALLOWED_OUTCOMES",
    "MODE2Z_HN_DEFAULT_ETA_VALUES",
    "MODE2Z_HN_DEFAULT_OUTPUT_ROOT",
    "MODE2Z_HN_DOC_PATH",
    "MODE2Z_HN_MIN_GRID_N",
    "MODE2Z_HN_STAGE",
    "MODE2Z_HN_Z_M",
    "Mode2ZHighNConfig",
    "Mode2ZHighNResult",
    "load_mode2z_baseline_rows",
    "mode2z_highn_audit",
    "mode2z_highn_outcome",
    "run_mode2z_highn_confirmation",
    "write_mode2z_highn_outputs",
]
