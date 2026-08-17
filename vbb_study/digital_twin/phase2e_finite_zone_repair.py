"""Finite-zone propagation measurements and dual-scale Phase 2E figures."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.signal import find_peaks

from vbb_study.digital_twin.phase2a_contracts import (
    canonical_hardware_manifest,
    hardware_value,
)
from vbb_study.digital_twin.phase2b_visual_cases import _scalar_seed
from vbb_study.digital_twin.phase2e_propagation_repair import (
    CANONICAL_REGION_M,
    _configure_matplotlib,
    _edge_energy_fraction,
    _hash_arrays,
    _normalised,
    _radial_profile,
    _safe_corr,
    _save_figure,
    _sha256_array,
    _write_csv,
    _write_json,
    build_scalar_route_checkpoints,
    refresh_phase2e_artifact_manifest,
)
from vbb_study.digital_twin.phase2e_report_visualisation import phase2e_upstream_hashes
from vbb_study.digital_twin.phase2e_spectral_propagation import (
    DensePropagationMap,
    build_dense_spectral_propagation,
)
from vbb_study.equations.propagation import (
    make_bl_asm_propagator,
    scalable_angular_spectrum_propagate,
)


EPS = np.finfo(float).tiny
VALIDATION_ROOT = Path("outputs/validation/phase2e_propagation_repair")
FIGURE_ROOT = Path("outputs/figures/phase2e_report_visualisation/01b_propagation_maps")
Z_END_M = 180.0e-3
Z_STEP_M = 0.5e-3
Z_VALUES_M = np.arange(0.0, Z_END_M + 0.5 * Z_STEP_M, Z_STEP_M)
DETAIL_HALF_WIDTH_M = 0.5e-3
DETAIL_SAMPLES = 1001
METRIC_Z_MIN_M = 5.0e-3
METRIC_Z_MAX_M = 130.0e-3
METRIC_Z_VALUES_M = np.arange(METRIC_Z_MIN_M, METRIC_Z_MAX_M + 0.5 * Z_STEP_M, Z_STEP_M)
AXIAL_HALF_MAX_FRACTION = 0.5
BUCKET_HALF_MAX_FRACTION = 0.5
RADIUS_STABILITY_FRACTION = 0.10
FEATURE_WEAK_FRACTION = 0.05
MINIMUM_WIDTH_SAMPLES = 6.0


@dataclass(frozen=True)
class ContextPropagation:
    case_id: str
    x_m: np.ndarray = field(repr=False, compare=False)
    z_m: np.ndarray = field(repr=False, compare=False)
    xz: np.ndarray = field(repr=False, compare=False)
    yz: np.ndarray = field(repr=False, compare=False)
    total_power: np.ndarray = field(repr=False, compare=False)
    edge_energy: np.ndarray = field(repr=False, compare=False)
    source_field: np.ndarray = field(repr=False, compare=False)
    grid: Mapping[str, Any] = field(repr=False, compare=False)
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class MeasuredMetrics:
    case_id: str
    z_m: np.ndarray = field(repr=False, compare=False)
    feature_intensity: np.ndarray = field(repr=False, compare=False)
    bucket_power: np.ndarray = field(repr=False, compare=False)
    radius_m: np.ndarray = field(repr=False, compare=False)
    width_m: np.ndarray = field(repr=False, compare=False)
    samples_across_width: np.ndarray = field(repr=False, compare=False)
    valid: np.ndarray = field(repr=False, compare=False)
    invalid_reason: tuple[str, ...]
    definition: Mapping[str, Any]
    zones: Mapping[str, Any]


def _ell(case_id: str) -> int:
    try:
        return {"B0": 0, "V1": 1, "V3": 3}[case_id]
    except KeyError as exc:
        raise ValueError(f"unsupported finite-zone case {case_id!r}") from exc


def _source_for_window(case_id: str, window_m: float, grid_n: int) -> tuple[np.ndarray, Mapping[str, Any], dict[str, Any]]:
    if np.isclose(window_m, 10.0e-3) and grid_n == 512:
        return _scalar_seed(case_id, _ell(case_id), grid_n=512)
    checkpoints = build_scalar_route_checkpoints(
        case_id, _ell(case_id), grid_n=grid_n, window_m=window_m
    )
    source = np.asarray(checkpoints["post_axicon"], dtype=np.complex128).copy()
    grid = checkpoints["grid"]
    metadata = dict(checkpoints["metadata"])
    del checkpoints
    gc.collect()
    return source, grid, metadata


def build_context_propagation(
    case_id: str,
    *,
    window_m: float = 10.0e-3,
    grid_n: int = 512,
    z_values_m: Sequence[float] = Z_VALUES_M,
) -> ContextPropagation:
    """Build memory-bounded full-plane context using the accepted BL-ASM operator."""

    source, grid, metadata = _source_for_window(case_id, window_m, grid_n)
    z = np.asarray(z_values_m, dtype=float)
    x = np.asarray(grid["x"], dtype=float)
    xz = np.empty((z.size, x.size), dtype=np.float32)
    yz = np.empty_like(xz)
    power = np.empty(z.size, dtype=float)
    edge = np.empty(z.size, dtype=float)
    centre = x.size // 2
    propagator = make_bl_asm_propagator(
        source,
        dict(grid),
        float(metadata["wavelength_m"]),
        n_medium=1.0,
        bandlimit=True,
        include_evanescent=True,
    )
    for index, z_m in enumerate(z):
        propagated = source if np.isclose(z_m, 0.0) else propagator(float(z_m))
        intensity = np.abs(propagated) ** 2
        xz[index] = intensity[centre, :]
        yz[index] = intensity[:, centre]
        power[index] = float(np.sum(intensity) * float(grid["dx"]) ** 2)
        edge[index] = _edge_energy_fraction(intensity, grid)
    return ContextPropagation(
        case_id=case_id,
        x_m=x,
        z_m=z,
        xz=xz,
        yz=yz,
        total_power=power,
        edge_energy=edge,
        source_field=source,
        grid=grid,
        provenance={
            **metadata,
            "status": "regenerated_exact_canonical_route",
            "source_plane": "axicon_output_plane",
            "z_origin": "axicon_output_plane",
            "propagation_medium_index": 1.0,
            "propagator": "make_bl_asm_propagator (same transfer function as accepted angular_spectrum_propagate_bl)",
            "pupil_application_count": 1,
            "axicon_application_count": 1,
            "objective_transform_application_count": 0,
            "field_already_focused": False,
            "physical_window_m": float(window_m),
            "native_grid_n": int(grid_n),
            "native_dx_m": float(grid["dx"]),
        },
    )


def build_fixed_detail_map(context: ContextPropagation) -> DensePropagationMap:
    coordinates = np.linspace(-DETAIL_HALF_WIDTH_M, DETAIL_HALF_WIDTH_M, DETAIL_SAMPLES)
    return build_dense_spectral_propagation(
        grid=context.grid,
        wavelength_m=float(context.provenance["wavelength_m"]),
        z_values_m=context.z_m,
        transverse_coordinates_m=coordinates,
        scalar_field=context.source_field,
        source_label=(
            f"{context.case_id} accepted-source fixed-coordinate line evaluation; "
            "presentation and symmetry-validated radial diagnostics only"
        ),
    )


def _linear_crossing(
    radius: np.ndarray,
    profile: np.ndarray,
    first_index: int,
    second_index: int,
    level: float,
) -> float:
    r0, r1 = float(radius[first_index]), float(radius[second_index])
    y0, y1 = float(profile[first_index]), float(profile[second_index])
    if np.isclose(y0, y1):
        return 0.5 * (r0 + r1)
    fraction = float(np.clip((level - y0) / (y1 - y0), 0.0, 1.0))
    return r0 + fraction * (r1 - r0)


def _b0_profile_measurement(
    radius: np.ndarray,
    profile: np.ndarray,
    *,
    dx_m: float,
    output_halfwidth_m: float,
) -> dict[str, Any]:
    result = {
        "valid": False,
        "reason": "unknown",
        "feature_intensity": float("nan"),
        "radius_m": float("nan"),
        "width_m": float("nan"),
        "samples_across_width": float("nan"),
    }
    if profile.size < 8 or not np.all(np.isfinite(profile[:8])):
        result["reason"] = "profile_unavailable"
        return result
    peak = float(profile[0])
    central_region = profile[radius <= min(0.25e-3, 0.8 * output_halfwidth_m)]
    if peak <= EPS or central_region.size < 4:
        result["reason"] = "central_profile_too_weak"
        return result
    if peak < 0.95 * float(np.max(central_region)) or profile[1] >= peak:
        result["reason"] = "central_point_not_intended_local_maximum"
        return result
    below = np.flatnonzero(profile[1:] <= 0.5 * peak)
    if not below.size:
        result["reason"] = "no_outward_half_maximum_crossing"
        return result
    outer_index = int(below[0] + 1)
    hwhm = _linear_crossing(radius, profile, outer_index - 1, outer_index, 0.5 * peak)
    if hwhm >= 0.80 * output_halfwidth_m:
        result["reason"] = "half_maximum_crossing_near_output_boundary"
        return result
    width = 2.0 * hwhm
    samples = width / max(float(dx_m), EPS)
    if samples < MINIMUM_WIDTH_SAMPLES:
        result["reason"] = "insufficient_native_samples_across_hwhm_diameter"
        return result
    result.update(
        valid=True,
        reason="valid",
        feature_intensity=peak,
        radius_m=hwhm,
        width_m=width,
        samples_across_width=samples,
    )
    return result


def _reference_ring_definition(
    radius: np.ndarray,
    profile: np.ndarray,
    dx_m: float,
) -> dict[str, float]:
    search = np.flatnonzero(radius <= 0.25e-3)
    peaks, _ = find_peaks(profile[search])
    eligible = peaks[radius[peaks] >= 2.0 * dx_m]
    if not eligible.size:
        raise RuntimeError("reference ring has no off-axis local maximum")
    peak_index = int(eligible[np.argmax(profile[eligible])])
    half = 0.5 * float(profile[peak_index])
    left = np.flatnonzero(profile[:peak_index] <= half)
    right = np.flatnonzero(profile[peak_index + 1:] <= half)
    if not left.size or not right.size:
        raise RuntimeError("reference ring has no two-sided HWHM crossings")
    left_index = int(left[-1])
    right_index = int(peak_index + 1 + right[0])
    inner = _linear_crossing(radius, profile, left_index, left_index + 1, half)
    outer = _linear_crossing(radius, profile, right_index - 1, right_index, half)
    return {
        "reference_ring_radius_m": float(radius[peak_index]),
        "reference_inner_hwhm_m": inner,
        "reference_outer_hwhm_m": outer,
        "reference_ring_width_m": outer - inner,
    }


def _ring_profile_measurement(
    radius: np.ndarray,
    profile: np.ndarray,
    reference: Mapping[str, float],
    *,
    dx_m: float,
    output_halfwidth_m: float,
) -> dict[str, Any]:
    empty = {
        "valid": False,
        "reason": "unknown",
        "feature_intensity": float("nan"),
        "radius_m": float("nan"),
        "width_m": float("nan"),
        "samples_across_width": float("nan"),
    }
    peaks, _ = find_peaks(profile)
    reference_radius = float(reference["reference_ring_radius_m"])
    tolerance = max(0.65 * reference_radius, 20.0e-6)
    eligible = peaks[np.abs(radius[peaks] - reference_radius) <= tolerance]
    if not eligible.size:
        empty["reason"] = "intended_annular_maximum_not_found"
        return empty
    peak_index = int(eligible[np.argmin(np.abs(radius[eligible] - reference_radius))])
    peak = float(profile[peak_index])
    half = 0.5 * peak
    left = np.flatnonzero(profile[:peak_index] <= half)
    right = np.flatnonzero(profile[peak_index + 1:] <= half)
    if not left.size or not right.size:
        empty["reason"] = "ring_hwhm_crossing_missing"
        return empty
    left_index = int(left[-1])
    right_index = int(peak_index + 1 + right[0])
    inner = _linear_crossing(radius, profile, left_index, left_index + 1, half)
    outer = _linear_crossing(radius, profile, right_index - 1, right_index, half)
    if outer >= 0.80 * output_halfwidth_m:
        empty["reason"] = "ring_crossing_near_output_boundary"
        return empty
    width = outer - inner
    samples = width / max(float(dx_m), EPS)
    if samples < MINIMUM_WIDTH_SAMPLES:
        empty["reason"] = "insufficient_native_samples_across_ring_width"
        return empty
    empty.update(
        valid=True,
        reason="valid",
        feature_intensity=peak,
        radius_m=float(radius[peak_index]),
        width_m=width,
        samples_across_width=samples,
        inner_hwhm_m=inner,
        outer_hwhm_m=outer,
    )
    return empty


def _longest_true_interval(mask: np.ndarray, z_m: np.ndarray) -> tuple[float, float] | None:
    values = np.asarray(mask, dtype=bool)
    if not np.any(values):
        return None
    padded = np.r_[False, values, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1) - 1
    index = int(np.argmax(stops - starts))
    return float(z_m[starts[index]]), float(z_m[stops[index]])


def _sas_plane(
    source: np.ndarray,
    grid: Mapping[str, Any],
    wavelength_m: float,
    z_m: float,
    *,
    pad_factor: int = 4,
) -> tuple[np.ndarray, Mapping[str, Any], Mapping[str, Any]]:
    field, output_grid, metadata = scalable_angular_spectrum_propagate(
        source,
        dict(grid),
        wavelength_m,
        z_m,
        n_medium=1.0,
        pad_factor=pad_factor,
        bandlimit=True,
        skip_final_phase=True,
        allow_invalid=False,
    )
    return np.asarray(np.abs(field) ** 2, dtype=np.float32), output_grid, metadata


def build_measured_metrics(context: ContextPropagation) -> tuple[MeasuredMetrics, list[dict[str, Any]]]:
    """Measure HWHM/ring metrics on a fixed-pad SAS grid derived from the accepted source."""

    wavelength = float(context.provenance["wavelength_m"])
    reference_plane, reference_grid, reference_meta = _sas_plane(
        context.source_field, context.grid, wavelength, 60.0e-3, pad_factor=4
    )
    reference_r, reference_profile = _radial_profile(reference_plane, reference_grid)
    if context.case_id == "B0":
        reference_measurement = _b0_profile_measurement(
            reference_r,
            reference_profile,
            dx_m=float(reference_grid["dx"]),
            output_halfwidth_m=0.5 * int(reference_grid["N"]) * float(reference_grid["dx"]),
        )
        if not reference_measurement["valid"]:
            raise RuntimeError(f"B0 z=60 mm reference HWHM is invalid: {reference_measurement}")
        minima, _ = find_peaks(-reference_profile)
        after_hwhm = minima[reference_r[minima] > float(reference_measurement["radius_m"])]
        if not after_hwhm.size:
            raise RuntimeError("B0 reference profile has no first minimum after HWHM")
        definition: dict[str, Any] = {
            "case_id": "B0",
            "estimator": "central_core_azimuthal_average_HWHM",
            "reference_z_m": 60.0e-3,
            "reference_radius_m": float(reference_measurement["radius_m"]),
            "reference_width_m": float(reference_measurement["width_m"]),
            "fixed_bucket_radius_m": float(reference_r[int(after_hwhm[0])]),
            "minimum_samples_across_width": MINIMUM_WIDTH_SAMPLES,
            "metric_grid": "accepted N=512 source propagated by pad-factor-4 SAS zoom",
            "reference_output_dx_m": float(reference_grid["dx"]),
            "reference_samples_across_width": float(reference_measurement["samples_across_width"]),
        }
    else:
        ring_reference = _reference_ring_definition(
            reference_r, reference_profile, float(reference_grid["dx"])
        )
        definition = {
            "case_id": context.case_id,
            "estimator": "intended_annular_maximum_with_two_sided_HWHM",
            "reference_z_m": 60.0e-3,
            **ring_reference,
            "fixed_annulus_inner_m": ring_reference["reference_inner_hwhm_m"],
            "fixed_annulus_outer_m": ring_reference["reference_outer_hwhm_m"],
            "minimum_samples_across_width": MINIMUM_WIDTH_SAMPLES,
            "metric_grid": "accepted N=512 source propagated by pad-factor-4 SAS zoom",
            "reference_output_dx_m": float(reference_grid["dx"]),
            "reference_samples_across_width": float(
                ring_reference["reference_ring_width_m"] / float(reference_grid["dx"])
            ),
        }
    z = METRIC_Z_VALUES_M.copy()
    intensity = np.full(z.shape, np.nan, dtype=float)
    bucket_power = np.full(z.shape, np.nan, dtype=float)
    radius = np.full(z.shape, np.nan, dtype=float)
    width = np.full(z.shape, np.nan, dtype=float)
    samples = np.full(z.shape, np.nan, dtype=float)
    valid = np.zeros(z.shape, dtype=bool)
    reasons: list[str] = []
    rows: list[dict[str, Any]] = []
    for index, z_m in enumerate(z):
        plane, output_grid, sas_meta = _sas_plane(
            context.source_field, context.grid, wavelength, float(z_m), pad_factor=4
        )
        radial_r, radial_i = _radial_profile(plane, output_grid)
        output_halfwidth = 0.5 * int(output_grid["N"]) * float(output_grid["dx"])
        if context.case_id == "B0":
            measurement = _b0_profile_measurement(
                radial_r, radial_i,
                dx_m=float(output_grid["dx"]),
                output_halfwidth_m=output_halfwidth,
            )
            bucket_mask = np.asarray(output_grid["R"]) <= float(definition["fixed_bucket_radius_m"])
        else:
            measurement = _ring_profile_measurement(
                radial_r, radial_i, definition,
                dx_m=float(output_grid["dx"]),
                output_halfwidth_m=output_halfwidth,
            )
            bucket_mask = (
                (np.asarray(output_grid["R"]) >= float(definition["fixed_annulus_inner_m"]))
                & (np.asarray(output_grid["R"]) <= float(definition["fixed_annulus_outer_m"]))
            )
        reason = str(measurement["reason"])
        if measurement["valid"]:
            intensity[index] = float(measurement["feature_intensity"])
            bucket_power[index] = float(
                np.sum(plane[bucket_mask], dtype=float) * float(output_grid["dx"]) ** 2
            )
            radius[index] = float(measurement["radius_m"])
            width[index] = float(measurement["width_m"])
            samples[index] = float(measurement["samples_across_width"])
            valid[index] = True
        reasons.append(reason)
        rows.append({
            "case_id": context.case_id,
            "z_m": float(z_m),
            "valid": bool(measurement["valid"]),
            "invalid_reason": "" if measurement["valid"] else reason,
            "feature_intensity_raw": intensity[index],
            "fixed_bucket_power_raw": bucket_power[index],
            "feature_radius_m": radius[index],
            "feature_width_m": width[index],
            "samples_across_width": samples[index],
            "metric_dx_m": float(output_grid["dx"]),
            "metric_output_halfwidth_m": output_halfwidth,
            "sas_valid": bool(sas_meta["valid"]),
            "sas_pad_factor": 4,
        })
        del plane, output_grid
    finite_intensity = intensity[np.isfinite(intensity)]
    if not finite_intensity.size:
        raise RuntimeError(f"{context.case_id} has no valid finite-zone feature intensity")
    weak_threshold = FEATURE_WEAK_FRACTION * float(np.max(finite_intensity))
    for index in range(z.size):
        if valid[index] and intensity[index] < weak_threshold:
            valid[index] = False
            intensity[index] = np.nan
            bucket_power[index] = np.nan
            radius[index] = np.nan
            width[index] = np.nan
            samples[index] = np.nan
            reasons[index] = "feature_below_predeclared_5_percent_strength_floor"
            rows[index].update({
                "valid": False,
                "invalid_reason": reasons[index],
                "feature_intensity_raw": float("nan"),
                "fixed_bucket_power_raw": float("nan"),
                "feature_radius_m": float("nan"),
                "feature_width_m": float("nan"),
                "samples_across_width": float("nan"),
            })
    valid_intensity = np.where(valid, intensity, np.nan)
    valid_bucket = np.where(valid, bucket_power, np.nan)
    maximum_intensity = float(np.nanmax(valid_intensity))
    maximum_bucket = float(np.nanmax(valid_bucket))
    reference_radius = float(definition.get(
        "reference_radius_m", definition.get("reference_ring_radius_m")
    ))
    fwhm_mask = valid & (intensity >= AXIAL_HALF_MAX_FRACTION * maximum_intensity)
    strict_mask = (
        fwhm_mask
        & (bucket_power >= BUCKET_HALF_MAX_FRACTION * maximum_bucket)
        & (np.abs(radius - reference_radius) / max(reference_radius, EPS) <= RADIUS_STABILITY_FRACTION)
    )
    fwhm_zone = _longest_true_interval(fwhm_mask, z)
    strict_zone = _longest_true_interval(strict_mask, z)
    if strict_zone is None:
        first_failure = float("nan")
    else:
        later = z[z > strict_zone[1]]
        first_failure = float(later[0]) if later.size else float("nan")
    zones = {
        "configured_nominal_interval_m": list(CANONICAL_REGION_M),
        "configured_nominal_interval_role": "configured_nominal_interval_not_measured_zone",
        "geometric_hard_pupil_estimate_m": 112.5e-3,
        "geometric_gaussian_radius_estimate_m": 125.0e-3,
        "measured_FWHM_axial_zone_m": list(fwhm_zone) if fwhm_zone else None,
        "measured_strict_useful_region_m": list(strict_zone) if strict_zone else None,
        "first_sample_after_strict_region_m": first_failure,
        "criteria": {
            "feature_intensity_fraction": AXIAL_HALF_MAX_FRACTION,
            "fixed_bucket_power_fraction": BUCKET_HALF_MAX_FRACTION,
            "radius_stability_fraction": RADIUS_STABILITY_FRACTION,
            "feature_strength_validity_floor": FEATURE_WEAK_FRACTION,
            "minimum_samples_across_width": MINIMUM_WIDTH_SAMPLES,
        },
        "reference_sas_retained_output_power_fraction": float(reference_meta["retained_power_fraction"]),
    }
    return MeasuredMetrics(
        case_id=context.case_id,
        z_m=z,
        feature_intensity=intensity,
        bucket_power=bucket_power,
        radius_m=radius,
        width_m=width,
        samples_across_width=samples,
        valid=valid,
        invalid_reason=tuple(reasons),
        definition=definition,
        zones=zones,
    ), rows


def _zone_spans(metrics: MeasuredMetrics) -> list[tuple[str, tuple[float, float], str, float]]:
    spans: list[tuple[str, tuple[float, float], str, float]] = []
    spans.append(("configured nominal", CANONICAL_REGION_M, "0.55", 0.12))
    fwhm = metrics.zones["measured_FWHM_axial_zone_m"]
    strict = metrics.zones["measured_strict_useful_region_m"]
    if fwhm:
        spans.append(("measured FWHM", (float(fwhm[0]), float(fwhm[1])), "#56B4E9", 0.10))
    if strict:
        spans.append(("measured strict", (float(strict[0]), float(strict[1])), "#009E73", 0.13))
    return spans


def _mark_z_regions(axis: Any, metrics: MeasuredMetrics, *, vertical: bool) -> None:
    for label, interval, color, alpha in _zone_spans(metrics):
        if vertical:
            axis.axhspan(interval[0] * 1e3, interval[1] * 1e3, color=color, alpha=alpha, label=label)
        else:
            axis.axvspan(interval[0] * 1e3, interval[1] * 1e3, color=color, alpha=alpha, label=label)
    for value, label, style in (
        (112.5e-3, "geometric hard-pupil estimate", "--"),
        (125.0e-3, "geometric Gaussian estimate", ":"),
    ):
        if vertical:
            axis.axhline(value * 1e3, color="#D55E00", linestyle=style, linewidth=0.9, label=label)
        else:
            axis.axvline(value * 1e3, color="#D55E00", linestyle=style, linewidth=0.9, label=label)


def plot_finite_primary(
    context: ContextPropagation,
    detail: DensePropagationMap,
    metrics: MeasuredMetrics,
    stem: Path,
) -> dict[str, Any]:
    arrays = [
        context.xz, context.yz, context.total_power, context.edge_energy,
        detail.xz_intensity, detail.yz_intensity,
        metrics.feature_intensity, metrics.bucket_power, metrics.radius_m,
    ]
    before = _hash_arrays(arrays)
    plt = _configure_matplotlib()
    figure, axes = plt.subplots(2, 4, figsize=(19.0, 9.5), constrained_layout=True)
    full_max = max(float(np.max(context.xz)), float(np.max(context.yz)), EPS)
    detail_max = max(float(np.max(detail.xz_intensity)), float(np.max(detail.yz_intensity)), EPS)
    full_extent = [context.x_m[0] * 1e3, context.x_m[-1] * 1e3, 0.0, context.z_m[-1] * 1e3]
    detail_extent = [detail.x_m[0] * 1e3, detail.x_m[-1] * 1e3, 0.0, detail.z_m[-1] * 1e3]
    for axis, values, extent, maximum, title in (
        (axes[0, 0], context.xz, full_extent, full_max, "(a) full-field x-z context"),
        (axes[0, 1], context.yz, full_extent, full_max, "(b) full-field y-z context"),
        (axes[0, 2], detail.xz_intensity, detail_extent, detail_max, "(c) fixed +/-0.5 mm x-z detail"),
        (axes[0, 3], detail.yz_intensity, detail_extent, detail_max, "(d) fixed +/-0.5 mm y-z detail"),
    ):
        axis.imshow(
            np.asarray(values) / maximum,
            origin="lower", extent=extent, aspect="auto", cmap="inferno",
            vmin=0.0, vmax=1.0, interpolation="none",
        )
        _mark_z_regions(axis, metrics, vertical=True)
        axis.set_title(title)
        axis.set_xlabel("transverse position (mm)")
        axis.set_ylabel("z (mm)")
    z_mm = metrics.z_m * 1e3
    feature_display = metrics.feature_intensity / max(
        float(np.nanmax(metrics.feature_intensity)), EPS
    )
    bucket_display = metrics.bucket_power / max(float(np.nanmax(metrics.bucket_power)), EPS)
    axes[1, 0].plot(z_mm, feature_display, color="#0072B2")
    axes[1, 0].set_title("(e) raw valid on-axis/ring intensity")
    axes[1, 1].plot(z_mm, bucket_display, color="#009E73")
    axes[1, 1].set_title("(f) fixed core/annulus power")
    axes[1, 2].plot(z_mm, metrics.radius_m * 1e6, color="#D55E00")
    axes[1, 2].fill_between(
        z_mm, 0.0, 1.0, where=~metrics.valid,
        color="0.75", alpha=0.25, transform=axes[1, 2].get_xaxis_transform(),
        label="invalid/rejected",
    )
    axes[1, 2].set_title("(g) valid native feature radius")
    axes[1, 2].set_ylabel("radius (um)")
    axes[1, 2].legend(frameon=False, fontsize=8)
    power_norm = context.total_power / max(float(context.total_power[0]), EPS)
    axes[1, 3].plot(context.z_m * 1e3, power_norm, color="#0072B2", label="total plane power")
    edge_axis = axes[1, 3].twinx()
    edge_axis.plot(context.z_m * 1e3, context.edge_energy, color="#CC79A7", label="edge-energy fraction")
    axes[1, 3].set_title("(h) power and numerical boundary")
    axes[1, 3].set_ylabel("power / input power")
    edge_axis.set_ylabel("edge-energy fraction")
    lines = axes[1, 3].get_lines() + edge_axis.get_lines()
    axes[1, 3].legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=8)
    for axis in axes[1, :]:
        _mark_z_regions(axis, metrics, vertical=False)
        axis.set_xlim(0.0, Z_END_M * 1e3)
        axis.set_xlabel("z (mm)")
        axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    figure.legend(unique.values(), unique.keys(), loc="outside lower center", ncol=5, frameon=False)
    figure.suptitle(
        f"{context.case_id} finite propagation | full field, fixed beam detail, and measured-zone criteria",
        fontsize=15,
    )
    paths = _save_figure(figure, stem)
    plt.close(figure)
    after = _hash_arrays(arrays)
    if before != after:
        raise RuntimeError("finite-zone primary plotting mutated an input array")
    return {"paths": paths, "hashes_before": before, "hashes_after": after}


def plot_context_only(
    context: ContextPropagation,
    metrics: MeasuredMetrics,
    stem: Path,
) -> dict[str, Any]:
    arrays = [context.xz, context.yz]
    before = _hash_arrays(arrays)
    plt = _configure_matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 6.5), constrained_layout=True)
    maximum = max(float(np.max(context.xz)), float(np.max(context.yz)), EPS)
    extent = [context.x_m[0] * 1e3, context.x_m[-1] * 1e3, 0.0, context.z_m[-1] * 1e3]
    for axis, values, name in zip(axes, (context.xz, context.yz), ("x-z", "y-z")):
        image = axis.imshow(
            values / maximum, origin="lower", extent=extent, aspect="auto",
            cmap="inferno", vmin=0.0, vmax=1.0, interpolation="none",
        )
        _mark_z_regions(axis, metrics, vertical=True)
        axis.set_title(f"{name} complete 10 mm field")
        axis.set_xlabel("transverse position (mm)")
        axis.set_ylabel("z (mm)")
    figure.colorbar(image, ax=axes, fraction=0.025, pad=0.02).set_label("I / paired global Imax (linear)")
    figure.suptitle("B0 full-field extended-z context | no clipping, log, gamma, or per-z scaling", fontsize=14)
    paths = _save_figure(figure, stem)
    plt.close(figure)
    after = _hash_arrays(arrays)
    if before != after:
        raise RuntimeError("context plotting mutated an input array")
    return {"paths": paths, "hashes_before": before, "hashes_after": after}


def plot_detail_only(
    detail: DensePropagationMap,
    metrics: MeasuredMetrics,
    stem: Path,
) -> dict[str, Any]:
    arrays = [detail.xz_intensity, detail.yz_intensity]
    before = _hash_arrays(arrays)
    plt = _configure_matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 6.5), constrained_layout=True)
    maximum = max(float(np.max(detail.xz_intensity)), float(np.max(detail.yz_intensity)), EPS)
    extent = [detail.x_m[0] * 1e3, detail.x_m[-1] * 1e3, 0.0, detail.z_m[-1] * 1e3]
    for axis, values, name in zip(axes, (detail.xz_intensity, detail.yz_intensity), ("x-z", "y-z")):
        image = axis.imshow(
            values / maximum, origin="lower", extent=extent, aspect="auto",
            cmap="inferno", vmin=0.0, vmax=1.0, interpolation="none",
        )
        _mark_z_regions(axis, metrics, vertical=True)
        axis.set_title(f"{name} fixed +/-0.5 mm detail")
        axis.set_xlabel("transverse position (mm)")
        axis.set_ylabel("z (mm)")
    figure.colorbar(image, ax=axes, fraction=0.025, pad=0.02).set_label("I / paired detail Imax (linear)")
    figure.suptitle("B0 fixed beam-detail propagation | same crop at every z", fontsize=14)
    paths = _save_figure(figure, stem)
    plt.close(figure)
    after = _hash_arrays(arrays)
    if before != after:
        raise RuntimeError("detail plotting mutated an input array")
    return {"paths": paths, "hashes_before": before, "hashes_after": after}


def _snapshot_plane(context: ContextPropagation, z_m: float) -> tuple[np.ndarray, Mapping[str, Any], str]:
    if np.isclose(z_m, 0.0):
        propagator = make_bl_asm_propagator(
            context.source_field, dict(context.grid), float(context.provenance["wavelength_m"])
        )
        field = context.source_field if np.isclose(z_m, 0.0) else propagator(float(z_m))
        return np.asarray(np.abs(field) ** 2, dtype=np.float32), context.grid, "native_full_plane"
    maximum_pad_for_crop = int(np.floor(
        512.0 * float(context.provenance["wavelength_m"]) * z_m
        / (10.0e-3 * 2.0 * DETAIL_HALF_WIDTH_M)
    ))
    pad_factor = int(np.clip(maximum_pad_for_crop, 1, 4))
    plane, grid, _ = _sas_plane(
        context.source_field, context.grid, float(context.provenance["wavelength_m"]),
        float(z_m), pad_factor=pad_factor,
    )
    return plane, grid, f"sas_pad_{pad_factor}"


def _snapshot_z_values(metrics: MeasuredMetrics) -> list[tuple[str, float]]:
    fwhm = metrics.zones["measured_FWHM_axial_zone_m"]
    strict = metrics.zones["measured_strict_useful_region_m"]
    if not fwhm or not strict:
        return [
            ("before formation", 10e-3), ("early candidate", 30e-3),
            ("reference", 60e-3), ("late candidate", 90e-3),
            ("after candidate", 120e-3), ("final", Z_END_M),
        ]
    before = max(0.0, float(fwhm[0]) - 10.0e-3)
    central = 0.5 * (float(strict[0]) + float(strict[1]))
    after = float(metrics.zones["first_sample_after_strict_region_m"])
    return [
        ("before formation", before),
        ("early useful", float(strict[0])),
        ("central useful", central),
        ("late useful", float(strict[1])),
        ("after measured breakdown", after),
        ("final plane", Z_END_M),
    ]


def plot_snapshots(
    context: ContextPropagation,
    metrics: MeasuredMetrics,
    stem: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = []
    rows = []
    for label, z_m in _snapshot_z_values(metrics):
        plane, grid, method = _snapshot_plane(context, z_m)
        payload.append((label, z_m, plane, grid, method))
        rows.append({
            "case_id": context.case_id,
            "snapshot_role": label,
            "z_m": z_m,
            "physical_crop_halfwidth_m": DETAIL_HALF_WIDTH_M,
            "source_method": method,
            "native_dx_m": float(grid["dx"]),
            "metrics_from_snapshot": False,
        })
    arrays = [item[2] for item in payload]
    before = _hash_arrays(arrays)
    plt = _configure_matplotlib()
    figure, axes = plt.subplots(2, 3, figsize=(14.0, 9.0), constrained_layout=True)
    maximum = max(float(np.max(item[2])) for item in payload)
    for axis, (label, z_m, plane, grid, method) in zip(axes.ravel(), payload):
        x = np.asarray(grid["x"]) * 1e3
        axis.set_facecolor("black")
        axis.imshow(
            plane / max(maximum, EPS), origin="lower",
            extent=[x[0], x[-1], x[0], x[-1]], cmap="inferno",
            vmin=0.0, vmax=1.0, interpolation="none",
        )
        axis.set_xlim(-DETAIL_HALF_WIDTH_M * 1e3, DETAIL_HALF_WIDTH_M * 1e3)
        axis.set_ylim(-DETAIL_HALF_WIDTH_M * 1e3, DETAIL_HALF_WIDTH_M * 1e3)
        axis.set_title(f"{label} | z={z_m * 1e3:.1f} mm")
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
    figure.suptitle(
        f"{context.case_id} matched transverse snapshots | fixed +/-0.5 mm crop | shared linear scale",
        fontsize=14,
    )
    paths = _save_figure(figure, stem)
    plt.close(figure)
    after = _hash_arrays(arrays)
    if before != after:
        raise RuntimeError("snapshot plotting mutated an input array")
    return {"paths": paths, "hashes_before": before, "hashes_after": after}, rows


def _line_metrics(detail: DensePropagationMap, bucket_radius_m: float) -> dict[str, np.ndarray]:
    coordinates = np.asarray(detail.x_m, dtype=float)
    centre = int(np.argmin(np.abs(coordinates)))
    radius = coordinates[centre:]
    profile = 0.5 * (
        np.asarray(detail.xz_intensity[:, centre:], dtype=float)
        + np.asarray(detail.yz_intensity[:, centre:], dtype=float)
    )
    axis = profile[:, 0].copy()
    hwhm = np.full(axis.shape, np.nan)
    core_power = np.full(axis.shape, np.nan)
    for index, row in enumerate(profile):
        below = np.flatnonzero(row[1:] <= 0.5 * row[0])
        if below.size and row[0] >= 0.95 * float(np.max(row)):
            outer = int(below[0] + 1)
            hwhm[index] = _linear_crossing(radius, row, outer - 1, outer, 0.5 * row[0])
        mask = radius <= bucket_radius_m
        core_power[index] = float(2.0 * np.pi * np.trapezoid(row[mask] * radius[mask], radius[mask]))
    return {"axis": axis, "hwhm": hwhm, "core_power": core_power}


def run_window_convergence(
    b0_context: ContextPropagation,
    b0_detail: DensePropagationMap,
    bucket_radius_m: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, ContextPropagation]]:
    contexts = {"W10": b0_context}
    details = {"W10": b0_detail}
    for label, window_m, n in (("W15", 15.0e-3, 768), ("W20", 20.0e-3, 1024)):
        contexts[label] = build_context_propagation(
            "B0", window_m=window_m, grid_n=n, z_values_m=Z_VALUES_M
        )
        details[label] = build_fixed_detail_map(contexts[label])
    metrics = {label: _line_metrics(detail, bucket_radius_m) for label, detail in details.items()}
    reference_context = contexts["W20"]
    reference_detail = details["W20"]
    reference_metrics = metrics["W20"]
    rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    for label in ("W10", "W15", "W20"):
        context = contexts[label]
        detail = details[label]
        values = metrics[label]
        common = np.isfinite(values["hwhm"]) & np.isfinite(reference_metrics["hwhm"])
        radius_change = float(np.nanmax(
            np.abs(values["hwhm"][common] - reference_metrics["hwhm"][common])
            / np.maximum(reference_metrics["hwhm"][common], EPS)
        )) if np.any(common) else float("inf")
        trace_l2 = float(
            np.linalg.norm(_normalised(values["axis"]) - _normalised(reference_metrics["axis"]))
            / max(float(np.linalg.norm(_normalised(reference_metrics["axis"]))), EPS)
        )
        map_corr = min(
            _safe_corr(detail.xz_intensity, reference_detail.xz_intensity),
            _safe_corr(detail.yz_intensity, reference_detail.yz_intensity),
        )
        drift = float(np.ptp(context.total_power) / max(float(np.max(context.total_power)), EPS))
        maximum_edge = float(np.max(context.edge_energy))
        gate = bool(
            drift <= 0.05
            and maximum_edge <= 0.01
            and trace_l2 <= 0.01
            and radius_change <= 0.01
            and 1.0 - map_corr <= 1.0e-3
        )
        rows.append({
            "window_id": label,
            "physical_window_m": float(context.provenance["physical_window_m"]),
            "grid_n": int(context.provenance["native_grid_n"]),
            "native_dx_m": float(context.provenance["native_dx_m"]),
            "z_min_m": float(context.z_m[0]),
            "z_max_m": float(context.z_m[-1]),
            "z_step_m": float(context.z_m[1] - context.z_m[0]),
            "propagation_power_drift_fraction": drift,
            "maximum_edge_energy_fraction": maximum_edge,
            "on_axis_trace_normalised_l2_change_to_W20": trace_l2,
            "maximum_hwhm_radius_relative_change_to_W20": radius_change,
            "matched_detail_map_correlation_to_W20": map_corr,
            "fixed_core_power_trace_correlation_to_W20": _safe_corr(
                values["core_power"], reference_metrics["core_power"]
            ),
            "acceptance_pass": gate,
            "radius_method": "symmetry-validated mean of dense physical x/y line HWHM for convergence only",
        })
        for index, z_m in enumerate(context.z_m):
            edge_rows.append({
                "window_id": label,
                "z_m": float(z_m),
                "edge_energy_fraction": float(context.edge_energy[index]),
                "total_power": float(context.total_power[index]),
                "on_axis_intensity": float(values["axis"][index]),
                "fixed_core_power_radial_line_integral": float(values["core_power"][index]),
                "hwhm_radius_m": float(values["hwhm"][index]),
            })
    return rows, edge_rows, contexts


def run_z_sampling_convergence(
    context: ContextPropagation,
    bucket_radius_m: float,
) -> list[dict[str, Any]]:
    maps: dict[str, DensePropagationMap] = {"dz0p5": build_fixed_detail_map(context)}
    for label, step in (("dz1p0", 1.0e-3), ("dz0p25", 0.25e-3)):
        z = np.arange(0.0, Z_END_M + 0.5 * step, step)
        coordinates = np.linspace(-DETAIL_HALF_WIDTH_M, DETAIL_HALF_WIDTH_M, DETAIL_SAMPLES)
        maps[label] = build_dense_spectral_propagation(
            grid=context.grid,
            wavelength_m=float(context.provenance["wavelength_m"]),
            z_values_m=z,
            transverse_coordinates_m=coordinates,
            scalar_field=context.source_field,
            source_label=f"B0 z-step convergence {step:g} m",
        )
    reference = maps["dz0p25"]
    reference_metrics = _line_metrics(reference, bucket_radius_m)
    rows = []
    for label, propagation in maps.items():
        values = _line_metrics(propagation, bucket_radius_m)
        axis_on_reference = np.interp(reference.z_m, propagation.z_m, values["axis"])
        core_on_reference = np.interp(reference.z_m, propagation.z_m, values["core_power"])
        valid = np.isfinite(values["hwhm"])
        radius_on_reference = np.interp(
            reference.z_m,
            propagation.z_m[valid],
            values["hwhm"][valid],
            left=np.nan,
            right=np.nan,
        )
        reference_index = np.searchsorted(reference.z_m, propagation.z_m)
        reference_index = np.clip(reference_index, 0, reference.z_m.size - 1)
        common_z = np.isclose(reference.z_m[reference_index], propagation.z_m, atol=1e-12, rtol=0.0)
        candidate_valid = np.isfinite(values["hwhm"])
        reference_radius_common = reference_metrics["hwhm"][reference_index]
        reference_axis_common = reference_metrics["axis"][reference_index]
        strength_valid = (
            values["axis"] >= FEATURE_WEAK_FRACTION * float(np.max(values["axis"]))
        ) & (
            reference_axis_common >= FEATURE_WEAK_FRACTION * float(np.max(reference_metrics["axis"]))
        )
        zone_valid = (propagation.z_m >= METRIC_Z_MIN_M) & (propagation.z_m <= METRIC_Z_MAX_M)
        common = (
            common_z & candidate_valid & np.isfinite(reference_radius_common)
            & strength_valid & zone_valid
        )
        axis_l2 = float(
            np.linalg.norm(_normalised(axis_on_reference) - _normalised(reference_metrics["axis"]))
            / max(float(np.linalg.norm(_normalised(reference_metrics["axis"]))), EPS)
        )
        radius_change = float(np.nanmax(
            np.abs(values["hwhm"][common] - reference_radius_common[common])
            / np.maximum(reference_radius_common[common], EPS)
        )) if np.any(common) else float("inf")
        rows.append({
            "z_sampling_id": label,
            "z_step_m": float(propagation.z_m[1] - propagation.z_m[0]),
            "z_max_m": float(propagation.z_m[-1]),
            "on_axis_trace_normalised_l2_change_to_dz0p25": axis_l2,
            "fixed_core_power_trace_correlation_to_dz0p25": _safe_corr(
                core_on_reference, reference_metrics["core_power"]
            ),
            "maximum_hwhm_radius_relative_change_to_dz0p25": radius_change,
            "acceptance_pass": bool(axis_l2 <= 0.01 and radius_change <= 0.01),
        })
    return rows


def _figure_record(
    figure_id: str,
    paths: tuple[Path, Path],
    case_id: str,
    role: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "figure_family": "finite_zone_propagation_repair",
        "report_role": role,
        "png_path": paths[0].as_posix(),
        "pdf_path": paths[1].as_posix(),
        "case_ids": case_id,
        "source_artifacts": "accepted Phase 2A/2B axicon-output field",
        "data_basis": "full-plane accepted BL-ASM context plus fixed-coordinate detail and accepted-source SAS metric zoom",
        "normalisation_policy": "paired global linear maps; no per-z normalisation; no clipping in primary",
        "linear_log_mode": "linear",
        "x_unit": "mm",
        "y_unit": "mm",
        "z_unit": "mm",
        "x_limits": "full_and_fixed_detail",
        "y_limits": [0.0, 180.0],
        "comparison_group": "phase2e_finite_zone_repair",
        "matched_axes": True,
        "display_interpolation": "none",
        "metric_bearing": role == "main_text_candidate",
        "metrics_computed_on_native_arrays": True,
        "display_interpolation_used_for_metrics": False,
        "roi_occupancy": {},
        "superseded": False,
        "notes": notes,
    }


def _update_manifest(records: Sequence[Mapping[str, Any]]) -> None:
    root = Path("outputs/figures/phase2e_report_visualisation")
    json_path = root / "00_manifest/phase2e_figure_manifest.json"
    csv_path = root / "00_manifest/phase2e_figure_manifest.csv"
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    replacements = {
        "b0_canonical_propagation_primary": "b0_finite_propagation_primary",
        "v1_canonical_propagation_primary": "v1_finite_propagation_primary",
        "v3_canonical_propagation_primary": "v3_finite_propagation_primary",
    }
    output = []
    for row in rows:
        item = dict(row)
        figure_id = str(item.get("figure_id"))
        if figure_id in replacements:
            item["report_role"] = "superseded_not_report_eligible"
            item["superseded"] = True
            item["superseded_by"] = replacements[figure_id]
            item["supersession_reason"] = (
                "0--100 mm context ended before termination, full width hid morphology, and the former radius estimator was invalid outside formation"
            )
        output.append(item)
    new_ids = {str(row["figure_id"]) for row in records}
    output = [row for row in output if str(row.get("figure_id")) not in new_ids]
    output.extend(dict(row) for row in records)
    _write_json(json_path, output)
    _write_csv(csv_path, output)


def plot_previous_vs_repaired(
    context: ContextPropagation,
    detail: DensePropagationMap,
    metrics: MeasuredMetrics,
    stem: Path,
) -> dict[str, Any]:
    plt = _configure_matplotlib()
    previous_path = FIGURE_ROOT / "b0_canonical_propagation_primary.png"
    previous = plt.imread(previous_path)
    arrays = [previous, context.xz, detail.xz_intensity, metrics.radius_m]
    before = _hash_arrays(arrays)
    figure, axes = plt.subplots(2, 2, figsize=(15.0, 10.0), constrained_layout=True)
    axes[0, 0].imshow(previous)
    axes[0, 0].axis("off")
    axes[0, 0].set_title("superseded: +/-5 mm and 0--100 mm")
    full_max = max(float(np.max(context.xz)), EPS)
    axes[0, 1].imshow(
        context.xz / full_max, origin="lower",
        extent=[context.x_m[0] * 1e3, context.x_m[-1] * 1e3, 0.0, context.z_m[-1] * 1e3],
        aspect="auto", cmap="inferno", vmin=0.0, vmax=1.0, interpolation="none",
    )
    _mark_z_regions(axes[0, 1], metrics, vertical=True)
    axes[0, 1].set_title("repaired full-field context to 180 mm")
    axes[0, 1].set_xlabel("x (mm)")
    axes[0, 1].set_ylabel("z (mm)")
    detail_max = max(float(np.max(detail.xz_intensity)), EPS)
    axes[1, 0].imshow(
        detail.xz_intensity / detail_max, origin="lower",
        extent=[detail.x_m[0] * 1e3, detail.x_m[-1] * 1e3, 0.0, detail.z_m[-1] * 1e3],
        aspect="auto", cmap="inferno", vmin=0.0, vmax=1.0, interpolation="none",
    )
    _mark_z_regions(axes[1, 0], metrics, vertical=True)
    axes[1, 0].set_title("repaired fixed +/-0.5 mm detail")
    axes[1, 0].set_xlabel("x (mm)")
    axes[1, 0].set_ylabel("z (mm)")
    axes[1, 1].plot(metrics.z_m * 1e3, metrics.radius_m * 1e6, color="#D55E00")
    axes[1, 1].fill_between(
        metrics.z_m * 1e3, 0.0, 1.0, where=~metrics.valid,
        transform=axes[1, 1].get_xaxis_transform(), color="0.75", alpha=0.3,
    )
    _mark_z_regions(axes[1, 1], metrics, vertical=False)
    axes[1, 1].set_title("repaired HWHM radius; invalid values remain NaN")
    axes[1, 1].set_xlabel("z (mm)")
    axes[1, 1].set_ylabel("radius (um)")
    axes[1, 1].grid(alpha=0.2)
    figure.suptitle(
        "B0 previous versus repaired propagation | context and morphology now have separate views",
        fontsize=14,
    )
    paths = _save_figure(figure, stem)
    plt.close(figure)
    after = _hash_arrays(arrays)
    if before != after:
        raise RuntimeError("previous-versus-repaired plotting mutated an input array")
    return {"paths": paths, "hashes_before": before, "hashes_after": after}


def _route_semantics() -> dict[str, Any]:
    manifest = canonical_hardware_manifest()
    return {
        "initial_complex_field": "phase2b_visual_cases._scalar_seed B0/V1/V3",
        "initial_field_plane": "axicon_output_plane",
        "propagation_medium_index": 1.0,
        "wavelength_m": float(hardware_value(manifest, "wavelength_m")),
        "z_origin": "axicon_output_plane",
        "z_direction": "downstream_positive",
        "objective_status": "no objective Fourier transform is applied in this scalar propagation route",
        "pupil_plane": "repository-defined post-Fourier-filter numerical plane immediately before axicon",
        "pupil_application_count": 1,
        "axicon_conical_phase_application_count": 1,
        "objective_transform_application_count": 0,
        "field_already_focused": False,
        "physical_window_m": 10.0e-3,
        "native_context_grid_n": 512,
        "native_context_dx_m": 10.0e-3 / 512.0,
        "native_context_z_min_m": 0.0,
        "native_context_z_max_m": Z_END_M,
        "native_context_z_step_m": Z_STEP_M,
        "propagation_transfer_function": "band-limited angular spectrum with Matsushima mask and evanescent decay",
        "context_propagator": "make_bl_asm_propagator; exact cached-spectrum equivalent of accepted angular_spectrum_propagate_bl",
        "detail_renderer": "same-source fixed-coordinate spectral line evaluation; presentation only",
        "metric_renderer": "accepted-source scalable angular spectrum, pad factor 4; azimuthal metrics on returned native zoom arrays",
        "focused_field_propagated_as_axicon_field": False,
        "pupil_applied_twice": False,
        "plotting_or_validation_mutates_stack": False,
    }


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _measure_metric_plane_from_definition(
    source: np.ndarray,
    grid: Mapping[str, Any],
    wavelength_m: float,
    case_id: str,
    definition: Mapping[str, Any],
    z_m: float,
) -> dict[str, Any]:
    plane, output_grid, sas_meta = _sas_plane(
        source, grid, wavelength_m, z_m, pad_factor=4
    )
    radial_r, radial_i = _radial_profile(plane, output_grid)
    output_halfwidth = 0.5 * int(output_grid["N"]) * float(output_grid["dx"])
    if case_id == "B0":
        measurement = _b0_profile_measurement(
            radial_r, radial_i,
            dx_m=float(output_grid["dx"]),
            output_halfwidth_m=output_halfwidth,
        )
        bucket_mask = np.asarray(output_grid["R"]) <= float(definition["fixed_bucket_radius_m"])
    else:
        measurement = _ring_profile_measurement(
            radial_r, radial_i, definition,
            dx_m=float(output_grid["dx"]),
            output_halfwidth_m=output_halfwidth,
        )
        bucket_mask = (
            (np.asarray(output_grid["R"]) >= float(definition["fixed_annulus_inner_m"]))
            & (np.asarray(output_grid["R"]) <= float(definition["fixed_annulus_outer_m"]))
        )
    valid = bool(measurement["valid"])
    return {
        "case_id": case_id,
        "z_m": float(z_m),
        "valid": valid,
        "invalid_reason": "" if valid else str(measurement["reason"]),
        "feature_intensity_raw": float(measurement["feature_intensity"]) if valid else float("nan"),
        "fixed_bucket_power_raw": (
            float(np.sum(plane[bucket_mask], dtype=float) * float(output_grid["dx"]) ** 2)
            if valid else float("nan")
        ),
        "feature_radius_m": float(measurement["radius_m"]) if valid else float("nan"),
        "feature_width_m": float(measurement["width_m"]) if valid else float("nan"),
        "samples_across_width": float(measurement["samples_across_width"]) if valid else float("nan"),
        "metric_dx_m": float(output_grid["dx"]),
        "metric_output_halfwidth_m": output_halfwidth,
        "sas_valid": bool(sas_meta["valid"]),
        "sas_pad_factor": 4,
    }


def _metrics_from_rows(
    case_id: str,
    definition: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[MeasuredMetrics, list[dict[str, Any]], dict[str, Any]]:
    ordered = sorted((dict(row) for row in rows), key=lambda row: float(row["z_m"]))
    z = np.asarray([float(row["z_m"]) for row in ordered], dtype=float)
    valid = np.asarray([
        bool(row["valid"]) if isinstance(row["valid"], bool)
        else str(row["valid"]).strip().lower() == "true"
        for row in ordered
    ], dtype=bool)
    intensity = np.asarray([_float_or_nan(row.get("feature_intensity_raw")) for row in ordered])
    bucket = np.asarray([_float_or_nan(row.get("fixed_bucket_power_raw")) for row in ordered])
    radius = np.asarray([_float_or_nan(row.get("feature_radius_m")) for row in ordered])
    width = np.asarray([_float_or_nan(row.get("feature_width_m")) for row in ordered])
    samples = np.asarray([_float_or_nan(row.get("samples_across_width")) for row in ordered])
    reasons = [str(row.get("invalid_reason", "")) for row in ordered]
    maximum_raw = float(np.nanmax(intensity[valid]))
    weak = valid & (intensity < FEATURE_WEAK_FRACTION * maximum_raw)
    for index in np.flatnonzero(weak):
        valid[index] = False
        intensity[index] = bucket[index] = radius[index] = width[index] = samples[index] = np.nan
        reasons[index] = "feature_below_predeclared_5_percent_strength_floor"
        ordered[index].update({
            "valid": False,
            "invalid_reason": reasons[index],
            "feature_intensity_raw": float("nan"),
            "fixed_bucket_power_raw": float("nan"),
            "feature_radius_m": float("nan"),
            "feature_width_m": float("nan"),
            "samples_across_width": float("nan"),
        })
    maximum_intensity = float(np.nanmax(intensity[valid]))
    maximum_bucket = float(np.nanmax(bucket[valid]))
    reference_radius = float(definition.get(
        "reference_radius_m", definition.get("reference_ring_radius_m")
    ))
    fwhm_mask = valid & (intensity >= AXIAL_HALF_MAX_FRACTION * maximum_intensity)
    strict_mask = (
        fwhm_mask
        & (bucket >= BUCKET_HALF_MAX_FRACTION * maximum_bucket)
        & (np.abs(radius - reference_radius) / max(reference_radius, EPS) <= RADIUS_STABILITY_FRACTION)
    )
    fwhm_zone = _longest_true_interval(fwhm_mask, z)
    strict_zone = _longest_true_interval(strict_mask, z)
    later = z[z > strict_zone[1]] if strict_zone else np.asarray([], dtype=float)
    zones = {
        "configured_nominal_interval_m": list(CANONICAL_REGION_M),
        "configured_nominal_interval_role": "configured_nominal_interval_not_measured_zone",
        "geometric_hard_pupil_estimate_m": 112.5e-3,
        "geometric_gaussian_radius_estimate_m": 125.0e-3,
        "measured_FWHM_axial_zone_m": list(fwhm_zone) if fwhm_zone else None,
        "measured_strict_useful_region_m": list(strict_zone) if strict_zone else None,
        "first_sample_after_strict_region_m": float(later[0]) if later.size else float("nan"),
        "criteria": {
            "feature_intensity_fraction": AXIAL_HALF_MAX_FRACTION,
            "fixed_bucket_power_fraction": BUCKET_HALF_MAX_FRACTION,
            "radius_stability_fraction": RADIUS_STABILITY_FRACTION,
            "feature_strength_validity_floor": FEATURE_WEAK_FRACTION,
            "minimum_samples_across_width": MINIMUM_WIDTH_SAMPLES,
        },
    }
    metrics = MeasuredMetrics(
        case_id=case_id,
        z_m=z,
        feature_intensity=intensity,
        bucket_power=bucket,
        radius_m=radius,
        width_m=width,
        samples_across_width=samples,
        valid=valid,
        invalid_reason=tuple(reasons),
        definition=dict(definition),
        zones=zones,
    )
    valid_radius = radius[valid]
    valid_samples = samples[valid]
    zone_row = {
        "case_id": case_id,
        **zones,
        "valid_radius_min_m": float(np.min(valid_radius)),
        "valid_radius_max_m": float(np.max(valid_radius)),
        "minimum_valid_samples_across_width": float(np.min(valid_samples)),
        "maximum_valid_samples_across_width": float(np.max(valid_samples)),
        "feature_definition": dict(definition),
    }
    return metrics, ordered, zone_row


def finalize_finite_zone_repair() -> dict[str, Any]:
    """Extend pre-formation metrics, refresh plots, and re-evaluate corrected gates."""

    outcome_path = VALIDATION_ROOT / "repair_outcome.json"
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    definitions = {
        str(row["case_id"]): dict(row["feature_definition"])
        for row in outcome["zone_metrics"]
    }
    with (VALIDATION_ROOT / "feature_radius_validity.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        existing_rows = list(csv.DictReader(handle))
    by_case = {
        case_id: [row for row in existing_rows if row["case_id"] == case_id]
        for case_id in ("B0", "V1", "V3")
    }
    contexts: dict[str, ContextPropagation] = {}
    details: dict[str, DensePropagationMap] = {}
    metrics_by_case: dict[str, MeasuredMetrics] = {}
    combined_rows: list[dict[str, Any]] = []
    zone_rows: list[dict[str, Any]] = []
    mutation_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    early_z = np.arange(5.0e-3, 20.0e-3, Z_STEP_M)
    for case_id in ("B0", "V1", "V3"):
        context = build_context_propagation(case_id)
        detail = build_fixed_detail_map(context)
        contexts[case_id] = context
        details[case_id] = detail
        early_rows = [
            _measure_metric_plane_from_definition(
                context.source_field,
                context.grid,
                float(context.provenance["wavelength_m"]),
                case_id,
                definitions[case_id],
                float(z_m),
            )
            for z_m in early_z
        ]
        metrics, case_rows, zone_row = _metrics_from_rows(
            case_id, definitions[case_id], [*early_rows, *by_case[case_id]]
        )
        metrics_by_case[case_id] = metrics
        combined_rows.extend(case_rows)
        zone_rows.append(zone_row)
        primary = plot_finite_primary(
            context, detail, metrics,
            FIGURE_ROOT / f"{case_id.lower()}_finite_propagation_primary",
        )
        snapshots, provenance = plot_snapshots(
            context, metrics, FIGURE_ROOT / f"{case_id.lower()}_transverse_snapshots"
        )
        snapshot_rows.extend(provenance)
        mutation_rows.extend((
            {
                "plot": f"{case_id.lower()}_finite_propagation_primary",
                "hashes_equal": primary["hashes_before"] == primary["hashes_after"],
            },
            {
                "plot": f"{case_id.lower()}_transverse_snapshots",
                "hashes_equal": snapshots["hashes_before"] == snapshots["hashes_after"],
            },
        ))
    b0_context = contexts["B0"]
    b0_detail = details["B0"]
    b0_metrics = metrics_by_case["B0"]
    context_plot = plot_context_only(
        b0_context, b0_metrics, FIGURE_ROOT / "b0_full_field_extended_z"
    )
    detail_plot = plot_detail_only(
        b0_detail, b0_metrics, FIGURE_ROOT / "b0_beam_detail_extended_z"
    )
    previous_plot = plot_previous_vs_repaired(
        b0_context, b0_detail, b0_metrics,
        FIGURE_ROOT / "b0_previous_vs_repaired_propagation",
    )
    for figure_id, payload in (
        ("b0_full_field_extended_z", context_plot),
        ("b0_beam_detail_extended_z", detail_plot),
        ("b0_previous_vs_repaired_propagation", previous_plot),
    ):
        mutation_rows.append({
            "plot": figure_id,
            "hashes_equal": payload["hashes_before"] == payload["hashes_after"],
        })
    z_rows = run_z_sampling_convergence(
        b0_context, float(b0_metrics.definition["fixed_bucket_radius_m"])
    )
    _write_csv(VALIDATION_ROOT / "feature_radius_validity.csv", combined_rows)
    _write_csv(VALIDATION_ROOT / "measured_zone_metrics.csv", zone_rows)
    _write_csv(VALIDATION_ROOT / "z_sampling_convergence.csv", z_rows)
    _write_csv(VALIDATION_ROOT / "finite_zone_snapshot_provenance.csv", snapshot_rows)
    _write_csv(VALIDATION_ROOT / "finite_zone_plot_mutation_audit.csv", mutation_rows)
    window_rows = list(csv.DictReader(
        (VALIDATION_ROOT / "window_convergence.csv").open("r", encoding="utf-8", newline="")
    ))
    window_pass = all(str(row["acceptance_pass"]).lower() == "true" for row in window_rows)
    z_pass = all(bool(row["acceptance_pass"]) for row in z_rows)
    sampling_pass = all(
        float(row["minimum_valid_samples_across_width"]) >= MINIMUM_WIDTH_SAMPLES
        for row in zone_rows
    )
    mutation_pass = all(bool(row["hashes_equal"]) for row in mutation_rows)
    authorised = bool(window_pass and z_pass and sampling_pass and mutation_pass)
    outcome.update({
        "outcome": "PHASE2E-FINITE-A" if authorised else "PHASE2E-FINITE-B",
        "report_figures_authorised": authorised,
        "zone_metrics": zone_rows,
        "window_convergence_pass": window_pass,
        "z_sampling_convergence_pass": z_pass,
        "minimum_width_sampling_pass": sampling_pass,
        "plot_mutation_detected": not mutation_pass,
        "preformation_metric_z_min_m": 5.0e-3,
    })
    _write_json(outcome_path, outcome)
    manifest_path = Path("outputs/figures/phase2e_report_visualisation/00_manifest/phase2e_figure_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest:
        if str(row.get("figure_id")) in {
            "b0_finite_propagation_primary", "v1_finite_propagation_primary",
            "v3_finite_propagation_primary",
        }:
            row["report_role"] = "main_text_candidate" if authorised else "forensic_not_report_authorised"
            row["metric_bearing"] = True
    _write_json(manifest_path, manifest)
    _write_csv(
        Path("outputs/figures/phase2e_report_visualisation/00_manifest/phase2e_figure_manifest.csv"),
        manifest,
    )
    refresh_phase2e_artifact_manifest()
    return outcome


def generate_finite_zone_repair() -> dict[str, Any]:
    VALIDATION_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    upstream_before = phase2e_upstream_hashes()
    semantics = _route_semantics()
    _write_json(VALIDATION_ROOT / "finite_zone_route_semantics.json", semantics)
    records: list[dict[str, Any]] = []
    mutation_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    zone_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    contexts: dict[str, ContextPropagation] = {}
    details: dict[str, DensePropagationMap] = {}
    measured: dict[str, MeasuredMetrics] = {}
    for case_id in ("B0", "V1", "V3"):
        context = build_context_propagation(case_id)
        detail = build_fixed_detail_map(context)
        metrics, rows = build_measured_metrics(context)
        contexts[case_id] = context
        details[case_id] = detail
        measured[case_id] = metrics
        feature_rows.extend(rows)
        primary = plot_finite_primary(
            context, detail, metrics,
            FIGURE_ROOT / f"{case_id.lower()}_finite_propagation_primary",
        )
        mutation_rows.append({
            "plot": f"{case_id.lower()}_finite_propagation_primary",
            "hashes_equal": primary["hashes_before"] == primary["hashes_after"],
        })
        records.append(_figure_record(
            f"{case_id.lower()}_finite_propagation_primary",
            primary["paths"], case_id, "main_text_candidate",
            "Dual-scale 0--180 mm propagation. Configured, measured FWHM, measured strict, and geometric intervals are labelled separately; invalid radii are NaN.",
        ))
        snapshots, case_snapshot_rows = plot_snapshots(
            context, metrics, FIGURE_ROOT / f"{case_id.lower()}_transverse_snapshots"
        )
        snapshot_rows.extend(case_snapshot_rows)
        mutation_rows.append({
            "plot": f"{case_id.lower()}_transverse_snapshots",
            "hashes_equal": snapshots["hashes_before"] == snapshots["hashes_after"],
        })
        records.append(_figure_record(
            f"{case_id.lower()}_transverse_snapshots",
            snapshots["paths"], case_id, "diagnostic_support",
            "Six matched fixed-crop transverse snapshots with one shared linear scale; snapshot arrays do not supply zone metrics.",
        ))
        valid_radius = metrics.radius_m[metrics.valid]
        valid_samples = metrics.samples_across_width[metrics.valid]
        zone_rows.append({
            "case_id": case_id,
            **dict(metrics.zones),
            "valid_radius_min_m": float(np.min(valid_radius)) if valid_radius.size else float("nan"),
            "valid_radius_max_m": float(np.max(valid_radius)) if valid_radius.size else float("nan"),
            "minimum_valid_samples_across_width": float(np.min(valid_samples)) if valid_samples.size else 0.0,
            "maximum_valid_samples_across_width": float(np.max(valid_samples)) if valid_samples.size else 0.0,
            "feature_definition": dict(metrics.definition),
        })

    b0_context = contexts["B0"]
    b0_detail = details["B0"]
    b0_metrics = measured["B0"]
    context_plot = plot_context_only(
        b0_context, b0_metrics, FIGURE_ROOT / "b0_full_field_extended_z"
    )
    detail_plot = plot_detail_only(
        b0_detail, b0_metrics, FIGURE_ROOT / "b0_beam_detail_extended_z"
    )
    previous_plot = plot_previous_vs_repaired(
        b0_context, b0_detail, b0_metrics,
        FIGURE_ROOT / "b0_previous_vs_repaired_propagation",
    )
    for figure_id, payload, notes in (
        ("b0_full_field_extended_z", context_plot, "Complete 10 mm field to 180 mm; contextual use only."),
        ("b0_beam_detail_extended_z", detail_plot, "Fixed +/-0.5 mm crop to 180 mm; same crop at every z."),
        ("b0_previous_vs_repaired_propagation", previous_plot, "Explains why the prior 0--100 mm full-window figure looked infinite and carried invalid radii."),
    ):
        mutation_rows.append({"plot": figure_id, "hashes_equal": payload["hashes_before"] == payload["hashes_after"]})
        records.append(_figure_record(figure_id, payload["paths"], "B0", "diagnostic_support", notes))

    bucket_radius = float(b0_metrics.definition["fixed_bucket_radius_m"])
    window_rows, edge_rows, window_contexts = run_window_convergence(
        b0_context, b0_detail, bucket_radius
    )
    z_rows = run_z_sampling_convergence(b0_context, bucket_radius)
    _write_csv(VALIDATION_ROOT / "window_convergence.csv", window_rows)
    _write_csv(VALIDATION_ROOT / "z_sampling_convergence.csv", z_rows)
    _write_csv(VALIDATION_ROOT / "measured_zone_metrics.csv", zone_rows)
    _write_csv(VALIDATION_ROOT / "feature_radius_validity.csv", feature_rows)
    _write_csv(VALIDATION_ROOT / "edge_energy_vs_z.csv", edge_rows)
    _write_csv(VALIDATION_ROOT / "finite_zone_snapshot_provenance.csv", snapshot_rows)
    _write_csv(VALIDATION_ROOT / "finite_zone_plot_mutation_audit.csv", mutation_rows)
    upstream_after = phase2e_upstream_hashes()
    if upstream_before != upstream_after:
        raise RuntimeError("accepted Phase 2A/2B artifacts changed during finite-zone repair")
    window_pass = all(bool(row["acceptance_pass"]) for row in window_rows)
    z_pass = all(bool(row["acceptance_pass"]) for row in z_rows)
    sampling_pass = all(
        float(row["minimum_valid_samples_across_width"]) >= MINIMUM_WIDTH_SAMPLES
        for row in zone_rows
    )
    measured_regions_exist = all(
        row["measured_FWHM_axial_zone_m"] is not None
        and row["measured_strict_useful_region_m"] is not None
        for row in zone_rows
    )
    mutation_pass = all(bool(row["hashes_equal"]) for row in mutation_rows)
    outcome_code = "PHASE2E-FINITE-A" if (
        window_pass and z_pass and sampling_pass and measured_regions_exist and mutation_pass
    ) else "PHASE2E-FINITE-B"
    if outcome_code != "PHASE2E-FINITE-A":
        for record in records:
            if record["report_role"] == "main_text_candidate":
                record["report_role"] = "forensic_not_report_authorised"
    outcome = {
        "outcome": outcome_code,
        "report_figures_authorised": outcome_code == "PHASE2E-FINITE-A",
        "why_previous_looked_infinite": [
            "z range ended at 100 mm before the propagated on-axis field had clearly decayed",
            "the complete +/-5 mm field compressed a tens-of-micrometres feature into a narrow line",
            "the configured 20--60 mm interval was shown without a separately measured zone",
            "the former radius finder returned values outside its valid formation/sampling domain",
        ],
        "route_semantics": semantics,
        "final_z_range_m": [0.0, Z_END_M],
        "z_step_m": Z_STEP_M,
        "full_field_window_m": 10.0e-3,
        "beam_detail_crop_m": [-DETAIL_HALF_WIDTH_M, DETAIL_HALF_WIDTH_M],
        "zone_metrics": zone_rows,
        "window_convergence_pass": window_pass,
        "z_sampling_convergence_pass": z_pass,
        "minimum_width_sampling_pass": sampling_pass,
        "maximum_edge_energy_fraction": max(float(row["maximum_edge_energy_fraction"]) for row in window_rows),
        "accepted_upstream_artifacts_unchanged": True,
        "plot_mutation_detected": not mutation_pass,
        "configured_interval_is_measured_zone": False,
        "old_disputed_figure_overwritten": False,
        "old_disputed_figure_status": "superseded_not_report_eligible",
        "replacement_figures": [record["png_path"] for record in records],
    }
    _write_json(VALIDATION_ROOT / "repair_outcome.json", outcome)
    _update_manifest(records)
    refresh_phase2e_artifact_manifest()
    del window_contexts
    gc.collect()
    return outcome


__all__ = [
    "ContextPropagation",
    "MeasuredMetrics",
    "build_context_propagation",
    "build_fixed_detail_map",
    "build_measured_metrics",
    "generate_finite_zone_repair",
    "finalize_finite_zone_repair",
    "run_window_convergence",
    "run_z_sampling_convergence",
]
