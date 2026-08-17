"""Forensic Phase 2E propagation reconstruction on the accepted Phase 2A/2B route.

The report plots in this module are consumers of native propagated arrays.  They
do not reconstruct a second optical train, normalise plane by plane, or mutate
the arrays supplied by the propagation API.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import find_peaks

from vbb_study.digital_twin.phase2a_canonical import (
    _axicon_phase,
    _fourier_first_order,
    _normalised_power,
    _panel_from_manifest,
    _pupil_and_aberration,
    _variant_settings,
)
from vbb_study.digital_twin.phase2a_contracts import (
    PHASE2A_CANONICAL_SLM_MODEL,
    canonical_hardware_manifest,
    hardware_value,
)
from vbb_study.digital_twin.phase2b_visual_cases import _scalar_seed
from vbb_study.digital_twin.phase2e_spectral_propagation import (
    DensePropagationMap,
    build_dense_spectral_propagation,
)
from vbb_study.equations.fields import fft2c, make_xy_grid
from vbb_study.equations.propagation import angular_spectrum_propagate_bl
from vbb_study.slm_model import apply_slm, slm_active_aperture


EPS = np.finfo(float).tiny
CANONICAL_REGION_M = (20.0e-3, 60.0e-3)
DEFAULT_PRIMARY_Z_M = np.arange(0.0, 100.0e-3 + 0.5e-3, 1.0e-3)
VALIDATION_ROOT = Path("outputs/validation/phase2e_propagation_repair")
FIGURE_ROOT = Path("outputs/figures/phase2e_report_visualisation/01b_propagation_maps")


@dataclass(frozen=True)
class CanonicalPropagation:
    """Native exact-route propagation arrays and their source-plane contract."""

    case_id: str
    complex_stack: np.ndarray = field(repr=False, compare=False)
    intensity_stack: np.ndarray = field(repr=False, compare=False)
    x_m: np.ndarray = field(repr=False, compare=False)
    y_m: np.ndarray = field(repr=False, compare=False)
    z_m: np.ndarray = field(repr=False, compare=False)
    total_power: np.ndarray = field(repr=False, compare=False)
    source_field: np.ndarray = field(repr=False, compare=False)
    grid: Mapping[str, Any] = field(repr=False, compare=False)
    provenance: Mapping[str, Any]

    @property
    def xz_intensity(self) -> np.ndarray:
        return self.intensity_stack[:, self.intensity_stack.shape[1] // 2, :]

    @property
    def yz_intensity(self) -> np.ndarray:
        return self.intensity_stack[:, :, self.intensity_stack.shape[2] // 2]


def _sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.view(np.uint8))
    return digest.hexdigest()


def _hash_arrays(arrays: Sequence[np.ndarray]) -> dict[str, str]:
    return {str(index): _sha256_array(array) for index, array in enumerate(arrays)}


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2) + "\n", encoding="utf-8")
    return path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    materialised = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialised:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in materialised:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fields})
    return path


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def build_scalar_route_checkpoints(
    case_id: str,
    ell: int,
    *,
    grid_n: int,
    window_m: float = 10.0e-3,
    variant: str = "realistic_fixed_bench_route",
) -> dict[str, Any]:
    """Instrument the Phase 2A scalar constructor without changing its operations."""

    manifest = canonical_hardware_manifest()
    settings = _variant_settings(variant)
    beam_radius = float(hardware_value(manifest, "beam_radius_on_slm_m"))
    grid = make_xy_grid(int(grid_n), float(window_m) / int(grid_n))
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    bx, by = settings["input_decentre_m"]
    raw_input = np.exp(-((X - float(bx)) ** 2 + (Y - float(by)) ** 2) / beam_radius**2)
    panel = _panel_from_manifest(manifest)
    panel_aperture = slm_active_aperture(grid, panel)
    hx, hy = settings["hologram_offset_m"]
    theta = np.arctan2(Y - float(hy), X - float(hx))
    radius_norm = np.hypot(X, Y) / max(2.0 * beam_radius, EPS)
    phase_error = float(settings["slm_phase_error_rms_rad"]) * (2.0 * radius_norm**2 - 1.0)
    phase1 = float(ell) * theta + phase_error
    phase2 = 0.5 * phase_error
    field = np.asarray(raw_input, dtype=np.complex128)
    first_order_fraction = 1.0
    if settings["apply_slms"]:
        slm1 = apply_slm(
            field,
            phase1,
            grid,
            panel,
            quantise_phase=bool(settings["quantise"]),
            apply_fill_factor=bool(settings["physical_hardware"]),
            apply_carrier=False,
            fill_factor_model=PHASE2A_CANONICAL_SLM_MODEL,
        )
        slm2 = apply_slm(
            slm1.total,
            phase2,
            grid,
            panel,
            quantise_phase=bool(settings["quantise"]),
            apply_fill_factor=bool(settings["physical_hardware"]),
            apply_carrier=True,
            fill_factor_model=PHASE2A_CANONICAL_SLM_MODEL,
        )
        post_slm = np.asarray(slm2.total, dtype=np.complex128)
    else:
        post_slm = np.where(panel_aperture, field * np.exp(1j * phase1), 0.0)
    post_filter = post_slm
    if settings["apply_first_order_filter"]:
        post_filter, first_order_fraction = _fourier_first_order(
            post_slm,
            grid,
            float(hardware_value(manifest, "carrier_frequency_cpm")),
            float(hardware_value(manifest, "fourier_iris_radius_cpm")),
            float(settings["iris_offset_fraction"]),
        )
    pupil_radius = float(hardware_value(manifest, "objective_pupil_radius_m"))
    if variant == "analytic_target_control":
        post_pupil = np.asarray(post_filter, dtype=np.complex128)
        pupil_fraction = 1.0
        pupil_application_count = 0
    else:
        post_pupil, pupil_fraction = _pupil_and_aberration(
            post_filter, grid, pupil_radius, settings
        )
        pupil_application_count = 1
    post_axicon = np.asarray(post_pupil, dtype=np.complex128)
    radial_wavevector = 0.0
    if case_id != "G0":
        axicon, radial_wavevector = _axicon_phase(grid, manifest, settings)
        post_axicon = post_pupil * axicon
    pupil_mask = np.asarray(grid["R"], dtype=float) <= pupil_radius
    return {
        "grid": grid,
        "raw_input": np.asarray(raw_input, dtype=np.complex128),
        "post_slm": np.asarray(post_slm, dtype=np.complex128),
        "pre_pupil": np.asarray(post_filter, dtype=np.complex128),
        "post_pupil": np.asarray(post_pupil, dtype=np.complex128),
        "post_axicon": np.asarray(post_axicon, dtype=np.complex128),
        "pupil_mask": pupil_mask,
        "metadata": {
            "case_id": case_id,
            "vortex_charge": int(ell),
            "variant": variant,
            "wavelength_m": float(hardware_value(manifest, "wavelength_m")),
            "window_m": float(window_m),
            "grid_n": int(grid_n),
            "dx_m": float(grid["dx"]),
            "first_order_efficiency": float(first_order_fraction),
            "pupil_retained_power_fraction": float(pupil_fraction),
            "pupil_radius_m": pupil_radius,
            "pupil_application_count": pupil_application_count,
            "axicon_application_count": 0 if case_id == "G0" else 1,
            "radial_wavevector_m_inv": float(radial_wavevector),
            "source_plane": "axicon_output_plane",
            "objective_transform_application_count": 0,
            "field_already_focused": False,
        },
    }


def load_or_build_canonical_propagation(
    case_id: str,
    *,
    grid_n: int = 512,
    z_values_m: Sequence[float] = DEFAULT_PRIMARY_Z_M,
) -> CanonicalPropagation:
    """Regenerate the accepted Phase 2B scalar route with its exact propagator."""

    ell = {"B0": 0, "V1": 1, "V3": 3}.get(case_id)
    if ell is None:
        raise ValueError(f"canonical repair supports B0/V1/V3, got {case_id!r}")
    source, grid, seed_meta = _scalar_seed(case_id, ell, grid_n=int(grid_n))
    z_values = np.asarray(z_values_m, dtype=float)
    shape = (z_values.size, int(grid["N"]), int(grid["N"]))
    complex_stack = np.empty(shape, dtype=np.complex64)
    intensity_stack = np.empty(shape, dtype=np.float32)
    powers = np.empty(z_values.size, dtype=float)
    for index, z_m in enumerate(z_values):
        propagated = np.asarray(source, dtype=np.complex128) if np.isclose(z_m, 0.0) else (
            angular_spectrum_propagate_bl(
                source,
                dict(grid),
                float(seed_meta["wavelength_m"]),
                float(z_m),
                n_medium=1.0,
                bandlimit=True,
                include_evanescent=True,
            )
        )
        intensity = np.abs(propagated) ** 2
        complex_stack[index] = np.asarray(propagated, dtype=np.complex64)
        intensity_stack[index] = np.asarray(intensity, dtype=np.float32)
        powers[index] = float(np.sum(intensity, dtype=float) * float(grid["dx"]) ** 2)
    axis = np.asarray(grid["x"], dtype=float)
    return CanonicalPropagation(
        case_id=case_id,
        complex_stack=complex_stack,
        intensity_stack=intensity_stack,
        x_m=axis.copy(),
        y_m=axis.copy(),
        z_m=z_values.copy(),
        total_power=powers,
        source_field=np.asarray(source, dtype=np.complex128),
        grid=grid,
        provenance={
            **dict(seed_meta),
            "status": "regenerated_exact_canonical_route",
            "source_builder": "phase2b_visual_cases._scalar_seed",
            "propagator": "equations.propagation.angular_spectrum_propagate_bl",
            "source_plane": "axicon_output_plane",
            "source_plane_sequence": "post_SLM -> Fourier_order_filter -> one_objective_pupil -> one_axicon",
            "propagation_medium_index": 1.0,
            "z_origin": "axicon_output_plane",
            "z_direction": "downstream_positive",
            "pupil_application_count": 1,
            "axicon_application_count": 1,
            "objective_transform_application_count": 0,
            "field_already_focused": False,
            "normalisation_mutates_source": False,
        },
    )


def _complex_overlap(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, np.ndarray]:
    inner = np.vdot(candidate, reference)
    phase = float(np.angle(inner)) if abs(inner) > 0.0 else 0.0
    aligned = np.asarray(candidate, dtype=np.complex128) * np.exp(1j * phase)
    denominator = np.sqrt(
        float(np.vdot(reference, reference).real) * float(np.vdot(aligned, aligned).real)
    )
    return float(abs(np.vdot(reference, aligned)) / max(denominator, EPS)), aligned


def _safe_corr(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=float).ravel()
    b = np.asarray(second, dtype=float).ravel()
    if a.size != b.size or a.size < 2:
        return float("nan")
    if np.std(a) <= EPS or np.std(b) <= EPS:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _normalised_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    a = np.asarray(reference, dtype=float)
    b = np.asarray(candidate, dtype=float)
    a = a / max(float(np.sum(a)), EPS)
    b = b / max(float(np.sum(b)), EPS)
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(a)), EPS))


def _interpolate_complex(
    source: np.ndarray,
    source_axis_m: np.ndarray,
    target_axis_m: np.ndarray,
) -> np.ndarray:
    X, Y = np.meshgrid(target_axis_m, target_axis_m, indexing="xy")
    points = np.column_stack((Y.ravel(), X.ravel()))
    real = RegularGridInterpolator(
        (source_axis_m, source_axis_m), np.asarray(source).real,
        method="linear", bounds_error=False, fill_value=0.0,
    )(points)
    imag = RegularGridInterpolator(
        (source_axis_m, source_axis_m), np.asarray(source).imag,
        method="linear", bounds_error=False, fill_value=0.0,
    )(points)
    return (real + 1j * imag).reshape(X.shape)


def _centroid(intensity: np.ndarray, grid: Mapping[str, Any]) -> tuple[float, float]:
    values = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    total = max(float(np.sum(values)), EPS)
    return (
        float(np.sum(values * np.asarray(grid["X"])) / total),
        float(np.sum(values * np.asarray(grid["Y"])) / total),
    )


def _edge_energy_fraction(intensity: np.ndarray, grid: Mapping[str, Any]) -> float:
    values = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    x = np.asarray(grid["x"], dtype=float)
    threshold = 0.90 * float(np.max(np.abs(x)))
    mask = (np.abs(np.asarray(grid["X"])) >= threshold) | (
        np.abs(np.asarray(grid["Y"])) >= threshold
    )
    return float(np.sum(values[mask]) / max(float(np.sum(values)), EPS))


def _radial_profile(
    intensity: np.ndarray,
    grid: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    radius = np.asarray(grid["R"], dtype=float)
    step = float(grid["dx"])
    edges = np.arange(0.0, float(np.max(radius)) + step, step)
    index = np.clip(np.digitize(radius.ravel(), edges) - 1, 0, edges.size - 2)
    values = np.maximum(np.asarray(intensity, dtype=float), 0.0).ravel()
    sums = np.bincount(index, weights=values, minlength=edges.size - 1)
    counts = np.bincount(index, minlength=edges.size - 1)
    profile = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return 0.5 * (edges[:-1] + edges[1:]), profile


def _spectral_support_radius(field: np.ndarray, grid: Mapping[str, Any]) -> float:
    power = np.abs(fft2c(field)) ** 2
    radius = 2.0 * np.pi * np.hypot(np.asarray(grid["FX"]), np.asarray(grid["FY"]))
    order = np.argsort(radius.ravel())
    cumulative = np.cumsum(power.ravel()[order])
    target = 0.999 * max(float(cumulative[-1]), EPS)
    return float(radius.ravel()[order[min(int(np.searchsorted(cumulative, target)), order.size - 1)]])


def compare_current_to_canonical_inputs(
    validation_root: Path = VALIDATION_ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare the N=1024 report source with the accepted N=512 source plane."""

    rows: list[dict[str, Any]] = []
    plot_payload: dict[str, Any] = {}
    manifest = canonical_hardware_manifest()
    pupil_radius = float(hardware_value(manifest, "objective_pupil_radius_m"))
    for case_id, ell in (("B0", 0), ("V1", 1), ("V3", 3)):
        canonical, canonical_grid, _ = _scalar_seed(case_id, ell, grid_n=512)
        current, current_grid, _ = _scalar_seed(case_id, ell, grid_n=1024)
        canonical_axis = np.asarray(canonical_grid["x"], dtype=float)
        current_on_canonical = _interpolate_complex(
            current, np.asarray(current_grid["x"], dtype=float), canonical_axis
        )
        overlap, aligned = _complex_overlap(canonical, current_on_canonical)
        reference_intensity = np.abs(canonical) ** 2
        candidate_intensity = np.abs(aligned) ** 2
        active = reference_intensity >= 1.0e-4 * float(np.max(reference_intensity))
        phase_delta = np.angle(aligned[active] * np.conj(canonical[active]))
        reference_centre = _centroid(reference_intensity, canonical_grid)
        candidate_centre = _centroid(candidate_intensity, canonical_grid)
        reference_r, reference_profile = _radial_profile(reference_intensity, canonical_grid)
        _, candidate_profile = _radial_profile(candidate_intensity, canonical_grid)
        reference_spectrum = np.abs(fft2c(canonical)) ** 2
        candidate_spectrum = np.abs(fft2c(aligned)) ** 2
        reference_mask = np.asarray(canonical_grid["R"]) <= pupil_radius
        current_mask = np.asarray(current_grid["R"]) <= pupil_radius
        mask_y, mask_x = np.meshgrid(canonical_axis, canonical_axis, indexing="ij")
        mask_on_canonical = RegularGridInterpolator(
            (np.asarray(current_grid["x"]), np.asarray(current_grid["x"])),
            current_mask.astype(float), method="nearest", bounds_error=False, fill_value=0.0,
        )(np.column_stack((mask_y.ravel(), mask_x.ravel())))
        mask_on_canonical = mask_on_canonical.reshape(reference_mask.shape) >= 0.5
        rows.append({
            "case_id": case_id,
            "common_plane": "axicon_output_plane",
            "canonical_grid_n": 512,
            "current_grid_n": 1024,
            "complex_overlap_after_global_phase_alignment": overlap,
            "normalised_intensity_correlation": _safe_corr(reference_intensity, candidate_intensity),
            "normalised_l2_intensity_error": _normalised_l2(reference_intensity, candidate_intensity),
            "total_power_ratio_current_to_canonical": float(
                np.sum(candidate_intensity) / max(float(np.sum(reference_intensity)), EPS)
            ),
            "radial_profile_correlation": _safe_corr(reference_profile, candidate_profile),
            "phase_rms_active_support_rad": float(np.sqrt(np.mean(phase_delta**2))),
            "pupil_mask_difference_fraction": float(np.mean(reference_mask != mask_on_canonical)),
            "field_centre_difference_m": float(np.hypot(
                candidate_centre[0] - reference_centre[0],
                candidate_centre[1] - reference_centre[1],
            )),
            "canonical_edge_energy_fraction": _edge_energy_fraction(reference_intensity, canonical_grid),
            "current_edge_energy_fraction": _edge_energy_fraction(candidate_intensity, canonical_grid),
            "spectral_support_relative_l2_error": _normalised_l2(
                reference_spectrum, candidate_spectrum
            ),
            "spectral_support_correlation": _safe_corr(reference_spectrum, candidate_spectrum),
            "canonical_999_spectral_radius_m_inv": _spectral_support_radius(canonical, canonical_grid),
            "current_999_spectral_radius_m_inv": _spectral_support_radius(aligned, canonical_grid),
            "canonical_eligibility_overlap_gate": 0.999,
            "canonical_eligibility_pass": bool(overlap >= 0.999),
        })
        if case_id == "B0":
            plot_payload = {
                "axis_m": canonical_axis,
                "canonical": reference_intensity,
                "current": candidate_intensity,
                "difference": candidate_intensity - reference_intensity,
                "phase_difference": np.angle(aligned * np.conj(canonical)),
                "active": active,
                "radial_r_m": reference_r,
                "radial_reference": reference_profile,
                "radial_current": candidate_profile,
            }
    _write_csv(Path(validation_root) / "canonical_vs_current_input.csv", rows)
    return rows, plot_payload


def _feature_definition(
    case_id: str,
    reference_plane: np.ndarray,
    grid: Mapping[str, Any],
) -> dict[str, Any]:
    radii, profile = _radial_profile(reference_plane, grid)
    dx = float(grid["dx"])
    search = np.flatnonzero(radii <= 0.50e-3)
    minima, _ = find_peaks(-profile[search])
    if case_id == "B0":
        eligible = minima[radii[minima] >= 1.5 * dx]
        index = int(eligible[0]) if eligible.size else int(np.argmin(profile[1:search[-1]]) + 1)
        return {
            "case_id": case_id,
            "reference_z_m": 60.0e-3,
            "observable": "on_axis_intensity",
            "bucket_type": "fixed_circular_core",
            "r_core_bucket_m": float(radii[index]),
            "radius_metric": "first radial-profile minimum after the central maximum",
            "metric_smoothing": "none",
        }
    peaks, _ = find_peaks(profile[search])
    eligible = peaks[radii[peaks] >= 2.0 * dx]
    if not eligible.size:
        eligible = search[radii[search] >= 2.0 * dx]
    peak_index = int(eligible[np.argmax(profile[eligible])])
    lower = minima[minima < peak_index]
    upper = minima[minima > peak_index]
    inner_index = int(lower[-1]) if lower.size else max(0, peak_index - 1)
    outer_index = int(upper[0]) if upper.size else min(profile.size - 1, peak_index + 1)
    return {
        "case_id": case_id,
        "reference_z_m": 60.0e-3,
        "observable": "fixed_reference_ring_intensity",
        "bucket_type": "fixed_annulus",
        "reference_ring_radius_m": float(radii[peak_index]),
        "r_inner_m": float(radii[inner_index]),
        "r_outer_m": float(radii[outer_index]),
        "radius_metric": "native radial-profile maximum within the fixed reference annulus",
        "metric_smoothing": "none",
    }


def propagation_observables(
    propagation: CanonicalPropagation,
    definition: Mapping[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return unsmoothed fixed-observable traces from native intensity planes."""

    z60_index = int(np.argmin(np.abs(propagation.z_m - 60.0e-3)))
    profile_definition = dict(definition or _feature_definition(
        propagation.case_id, propagation.intensity_stack[z60_index], propagation.grid
    ))
    radius = np.asarray(propagation.grid["R"], dtype=float)
    dx2 = float(propagation.grid["dx"]) ** 2
    if propagation.case_id == "B0":
        bucket = radius <= float(profile_definition["r_core_bucket_m"])
    else:
        bucket = (radius >= float(profile_definition["r_inner_m"])) & (
            radius <= float(profile_definition["r_outer_m"])
        )
    count = propagation.z_m.size
    feature_intensity = np.empty(count, dtype=float)
    bucket_power = np.empty(count, dtype=float)
    feature_radius = np.empty(count, dtype=float)
    plane_max = np.empty(count, dtype=float)
    plane_peak_radius = np.empty(count, dtype=float)
    edge_fraction = np.empty(count, dtype=float)
    centre = propagation.intensity_stack.shape[1] // 2
    centre_slice = (slice(centre - 1, centre + 1), slice(centre - 1, centre + 1))
    for index, plane in enumerate(propagation.intensity_stack):
        values = np.asarray(plane, dtype=float)
        radial_r, radial_i = _radial_profile(values, propagation.grid)
        if propagation.case_id == "B0":
            feature_intensity[index] = float(np.mean(values[centre_slice]))
            minima, _ = find_peaks(-radial_i)
            eligible = minima[radial_r[minima] >= 1.5 * float(propagation.grid["dx"])]
            feature_radius[index] = float(
                radial_r[int(eligible[0])] if eligible.size else profile_definition["r_core_bucket_m"]
            )
        else:
            reference_radius = float(profile_definition["reference_ring_radius_m"])
            ring_bin = int(np.argmin(np.abs(radial_r - reference_radius)))
            feature_intensity[index] = float(radial_i[ring_bin])
            annulus_bins = np.flatnonzero(
                (radial_r >= float(profile_definition["r_inner_m"]))
                & (radial_r <= float(profile_definition["r_outer_m"]))
            )
            feature_radius[index] = float(
                radial_r[annulus_bins[np.argmax(radial_i[annulus_bins])]]
                if annulus_bins.size else reference_radius
            )
        bucket_power[index] = float(np.sum(values[bucket], dtype=float) * dx2)
        peak_index = int(np.argmax(values))
        plane_max[index] = float(values.ravel()[peak_index])
        plane_peak_radius[index] = float(radius.ravel()[peak_index])
        edge_fraction[index] = _edge_energy_fraction(values, propagation.grid)
    switches = int(np.count_nonzero(
        np.abs(np.diff(plane_peak_radius)) > 1.5 * float(propagation.grid["dx"])
    ))
    traces = {
        "feature_intensity": feature_intensity,
        "bucket_power": bucket_power,
        "feature_radius_m": feature_radius,
        "plane_max": plane_max,
        "plane_peak_radius_m": plane_peak_radius,
        "edge_energy_fraction": edge_fraction,
    }
    summary = {
        "case_id": propagation.case_id,
        "profile_definition": profile_definition,
        "plane_max_feature_switch_count": switches,
        "power_drift_fraction": float(
            (np.max(propagation.total_power) - np.min(propagation.total_power))
            / max(float(np.max(propagation.total_power)), EPS)
        ),
        "maximum_edge_energy_fraction": float(np.max(edge_fraction)),
        "source_hash": _sha256_array(propagation.source_field),
    }
    return traces, summary


def _normalised(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array / max(float(np.max(array)), EPS)


def _save_figure(fig: Any, stem: Path, *, dpi: int = 400) -> tuple[Path, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, dpi=dpi, bbox_inches="tight", facecolor="white")
    return png, pdf


def _configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def plot_primary_propagation(
    propagation: CanonicalPropagation,
    traces: Mapping[str, np.ndarray],
    profile_definition: Mapping[str, Any],
    stem: Path,
) -> dict[str, Any]:
    """Render the canonical full-field maps without changing their arrays."""

    arrays = [
        propagation.intensity_stack,
        propagation.total_power,
        *[np.asarray(value) for value in traces.values()],
    ]
    before = _hash_arrays(arrays)
    plt = _configure_matplotlib()
    figure = plt.figure(figsize=(16.0, 9.0), constrained_layout=True)
    grid_spec = figure.add_gridspec(2, 3, height_ratios=(1.0, 0.82))
    axes = [
        figure.add_subplot(grid_spec[0, 0]),
        figure.add_subplot(grid_spec[0, 1]),
        figure.add_subplot(grid_spec[0, 2]),
        figure.add_subplot(grid_spec[1, 0]),
        figure.add_subplot(grid_spec[1, 1:]),
    ]
    xz = np.asarray(propagation.xz_intensity, dtype=float)
    yz = np.asarray(propagation.yz_intensity, dtype=float)
    shared_max = max(float(np.max(xz)), float(np.max(yz)), EPS)
    extent_x = [
        propagation.x_m[0] * 1e3, propagation.x_m[-1] * 1e3,
        propagation.z_m[0] * 1e3, propagation.z_m[-1] * 1e3,
    ]
    extent_y = [
        propagation.y_m[0] * 1e3, propagation.y_m[-1] * 1e3,
        propagation.z_m[0] * 1e3, propagation.z_m[-1] * 1e3,
    ]
    image_x = axes[0].imshow(
        xz / shared_max, origin="lower", extent=extent_x, aspect="auto",
        cmap="inferno", vmin=0.0, vmax=1.0, interpolation="none",
    )
    axes[1].imshow(
        yz / shared_max, origin="lower", extent=extent_y, aspect="auto",
        cmap="inferno", vmin=0.0, vmax=1.0, interpolation="none",
    )
    for axis in axes[:2]:
        for marker in CANONICAL_REGION_M:
            axis.axhline(marker * 1e3, color="white", linestyle="--", linewidth=0.9, alpha=0.8)
        axis.set_ylabel("z (mm)")
    axes[0].set_xlabel("x (mm)")
    axes[1].set_xlabel("y (mm)")
    axes[0].set_title("(a) x-z intensity | complete accepted x field")
    axes[1].set_title("(b) y-z intensity | complete accepted y field")
    colorbar = figure.colorbar(image_x, ax=axes[:2], fraction=0.025, pad=0.02)
    colorbar.set_label("I / paired global Imax (linear)")
    z_mm = propagation.z_m * 1e3
    primary_label = "on-axis intensity" if propagation.case_id == "B0" else "fixed-ring intensity"
    axes[2].plot(z_mm, _normalised(traces["feature_intensity"]), color="#0072B2", label=primary_label)
    axes[2].plot(
        z_mm, _normalised(traces["plane_max"]), color="0.55", linewidth=0.8,
        alpha=0.8, label="plane maximum (diagnostic)",
    )
    axes[2].set_title("(c) raw axial observable")
    axes[2].set_xlabel("z (mm)")
    axes[2].set_ylabel("trace / own maximum")
    axes[2].legend(frameon=False)
    axes[3].plot(z_mm, _normalised(traces["bucket_power"]), color="#009E73")
    axes[3].set_title("(d) fixed-bucket integrated power")
    axes[3].set_xlabel("z (mm)")
    axes[3].set_ylabel("bucket power / own maximum")
    accepted_radius = np.where(
        (propagation.z_m >= CANONICAL_REGION_M[0])
        & (propagation.z_m <= CANONICAL_REGION_M[1]),
        np.asarray(traces["feature_radius_m"]) * 1e6,
        np.nan,
    )
    axes[4].plot(z_mm, accepted_radius, color="#D55E00")
    axes[4].set_title("(e) native feature radius in accepted interval (unsmoothed)")
    axes[4].set_xlabel("z (mm)")
    axes[4].set_ylabel("radius (um)")
    for axis in axes[2:]:
        for marker in CANONICAL_REGION_M:
            axis.axvline(marker * 1e3, color="0.55", linestyle="--", linewidth=0.8)
        axis.grid(alpha=0.2)
    title = {
        "B0": "B0 bright-core Bessel",
        "V1": "V1 charge-1 vortex Bessel",
        "V3": "V3 charge-3 vortex Bessel",
    }[propagation.case_id]
    figure.suptitle(
        f"{title} | accepted Phase 2A/2B canonical propagation | native global-linear maps",
        fontsize=15,
    )
    paths = _save_figure(figure, stem)
    plt.close(figure)
    after = _hash_arrays(arrays)
    if before != after:
        raise RuntimeError("primary propagation plotting mutated an input array")
    return {
        "paths": paths,
        "hashes_before": before,
        "hashes_after": after,
        "normalisation": "one paired global linear I/Imax for x-z and y-z; no per-z scaling",
        "profile_definition": dict(profile_definition),
    }


def plot_low_intensity_diagnostic(
    propagation: CanonicalPropagation,
    stem: Path,
    *,
    pupil_limit_m: float,
    gaussian_limit_m: float,
    diagnostic_z_values_m: Sequence[float] | None = None,
) -> dict[str, Any]:
    arrays = [propagation.intensity_stack, propagation.source_field]
    before = _hash_arrays(arrays)
    plt = _configure_matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 6.0), constrained_layout=True)
    if diagnostic_z_values_m is None:
        diagnostic_z = propagation.z_m
        xz = np.asarray(propagation.xz_intensity, dtype=float)
        yz = np.asarray(propagation.yz_intensity, dtype=float)
    else:
        diagnostic_z = np.asarray(diagnostic_z_values_m, dtype=float)
        n = int(propagation.grid["N"])
        centre = n // 2
        xz = np.empty((diagnostic_z.size, n), dtype=np.float32)
        yz = np.empty_like(xz)
        for index, z_m in enumerate(diagnostic_z):
            propagated = propagation.source_field if np.isclose(z_m, 0.0) else angular_spectrum_propagate_bl(
                propagation.source_field,
                dict(propagation.grid),
                float(propagation.provenance["wavelength_m"]),
                float(z_m),
                n_medium=1.0,
                bandlimit=True,
                include_evanescent=True,
            )
            intensity = np.abs(propagated) ** 2
            xz[index] = intensity[centre, :]
            yz[index] = intensity[:, centre]
    shared_max = max(float(np.max(xz)), float(np.max(yz)), EPS)
    clipped_xz = np.minimum(xz / shared_max, 0.01)
    clipped_yz = np.minimum(yz / shared_max, 0.01)
    extents = (
        [propagation.x_m[0] * 1e3, propagation.x_m[-1] * 1e3, diagnostic_z[0] * 1e3, diagnostic_z[-1] * 1e3],
        [propagation.y_m[0] * 1e3, propagation.y_m[-1] * 1e3, diagnostic_z[0] * 1e3, diagnostic_z[-1] * 1e3],
    )
    images = []
    for axis, values, extent, transverse in zip(axes, (clipped_xz, clipped_yz), extents, ("x", "y")):
        images.append(axis.imshow(
            values, origin="lower", extent=extent, aspect="auto", cmap="inferno",
            vmin=0.0, vmax=0.01, interpolation="none",
        ))
        axis.axhspan(CANONICAL_REGION_M[0] * 1e3, CANONICAL_REGION_M[1] * 1e3, color="white", alpha=0.08)
        axis.axhline(pupil_limit_m * 1e3, color="#56B4E9", linestyle="--", linewidth=1.0, label="geometric pupil estimate")
        axis.axhline(gaussian_limit_m * 1e3, color="#009E73", linestyle="-.", linewidth=1.0, label="Gaussian-radius estimate")
        axis.set_xlabel(f"{transverse} (mm)")
        axis.set_ylabel("z (mm)")
        axis.set_title(f"{transverse}-z | clipped linear diagnostic")
    axes[0].legend(loc="upper left", facecolor="white", framealpha=0.92, fontsize=8)
    colorbar = figure.colorbar(images[0], ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("I / global Imax (linear, clipped at 0.01)")
    figure.suptitle(
        "B0 low-intensity diagnostic | linear intensity clipped at 0.01 of the global maximum;\n"
        "weak structures visually amplified; not used for morphology metrics",
        fontsize=14,
    )
    paths = _save_figure(figure, stem)
    plt.close(figure)
    after = _hash_arrays(arrays)
    if before != after:
        raise RuntimeError("low-intensity plotting mutated an input array")
    return {"paths": paths, "hashes_before": before, "hashes_after": after}


def _plot_input_comparison(payload: Mapping[str, Any], stem: Path) -> dict[str, Any]:
    arrays = [np.asarray(value) for key, value in payload.items() if isinstance(value, np.ndarray)]
    before = _hash_arrays(arrays)
    plt = _configure_matplotlib()
    figure, axes = plt.subplots(2, 3, figsize=(14.0, 8.5), constrained_layout=True)
    axis = np.asarray(payload["axis_m"]) * 1e3
    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    reference = np.asarray(payload["canonical"], dtype=float)
    current = np.asarray(payload["current"], dtype=float)
    maximum = max(float(np.max(reference)), float(np.max(current)), EPS)
    axes[0, 0].imshow(reference / maximum, origin="lower", extent=extent, cmap="inferno", vmin=0.0, vmax=1.0, interpolation="none")
    axes[0, 1].imshow(current / maximum, origin="lower", extent=extent, cmap="inferno", vmin=0.0, vmax=1.0, interpolation="none")
    diff = np.asarray(payload["difference"], dtype=float) / maximum
    limit = max(float(np.max(np.abs(diff))), EPS)
    axes[0, 2].imshow(diff, origin="lower", extent=extent, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="none")
    active_phase = np.where(np.asarray(payload["active"]), payload["phase_difference"], np.nan)
    axes[1, 0].imshow(active_phase, origin="lower", extent=extent, cmap="twilight", vmin=-np.pi, vmax=np.pi, interpolation="none")
    axes[1, 1].plot(np.asarray(payload["radial_r_m"]) * 1e3, _normalised(payload["radial_reference"]), label="accepted N=512")
    axes[1, 1].plot(np.asarray(payload["radial_r_m"]) * 1e3, _normalised(payload["radial_current"]), linestyle="--", label="Phase 2E N=1024 -> N=512")
    axes[1, 1].set_xlim(0.0, 0.6)
    axes[1, 1].legend(frameon=False)
    axes[1, 2].axis("off")
    titles = (
        "accepted canonical intensity", "current direct-route input intensity",
        "signed intensity difference", "phase difference on active support",
        "native radial profiles", "",
    )
    for plot_axis, title in zip(axes.ravel(), titles):
        plot_axis.set_title(title)
        if plot_axis.has_data() and title != "native radial profiles":
            plot_axis.set_xlabel("x (mm)")
            plot_axis.set_ylabel("y (mm)")
    axes[1, 1].set_xlabel("radius (mm)")
    axes[1, 1].set_ylabel("normalised intensity")
    figure.suptitle("B0 first-common-plane audit | axicon-output complex fields", fontsize=14)
    paths = _save_figure(figure, stem)
    plt.close(figure)
    after = _hash_arrays(arrays)
    if before != after:
        raise RuntimeError("input-comparison plotting mutated an input array")
    return {"paths": paths, "hashes_before": before, "hashes_after": after}


def _map_on_axes(
    source_map: np.ndarray,
    source_x: np.ndarray,
    source_z: np.ndarray,
    target_x: np.ndarray,
    target_z: np.ndarray,
) -> np.ndarray:
    Z, X = np.meshgrid(target_z, target_x, indexing="ij")
    points = np.column_stack((Z.ravel(), X.ravel()))
    return RegularGridInterpolator(
        (source_z, source_x), np.asarray(source_map, dtype=float),
        method="linear", bounds_error=False, fill_value=0.0,
    )(points).reshape(Z.shape)


def _intensity_on_axis(plane: np.ndarray) -> float:
    centre = plane.shape[0] // 2
    return float(np.mean(np.asarray(plane)[centre - 1:centre + 1, centre - 1:centre + 1]))


def _summary_from_source(
    field: np.ndarray,
    grid: Mapping[str, Any],
    wavelength_m: float,
    z_values_m: np.ndarray,
    profile_definition: Mapping[str, Any],
) -> dict[str, Any]:
    n = int(grid["N"])
    centre = n // 2
    xz = np.empty((z_values_m.size, n), dtype=np.float32)
    yz = np.empty_like(xz)
    axis_trace = np.empty(z_values_m.size, dtype=float)
    bucket_trace = np.empty(z_values_m.size, dtype=float)
    radius_trace = np.empty(z_values_m.size, dtype=float)
    power = np.empty(z_values_m.size, dtype=float)
    edges = np.empty(z_values_m.size, dtype=float)
    radius = np.asarray(grid["R"], dtype=float)
    bucket = radius <= float(profile_definition["r_core_bucket_m"])
    plane60 = None
    dx2 = float(grid["dx"]) ** 2
    for index, z_m in enumerate(z_values_m):
        propagated = field if np.isclose(z_m, 0.0) else angular_spectrum_propagate_bl(
            field, dict(grid), wavelength_m, float(z_m), n_medium=1.0,
            bandlimit=True, include_evanescent=True,
        )
        plane = np.abs(propagated) ** 2
        xz[index] = plane[centre, :]
        yz[index] = plane[:, centre]
        axis_trace[index] = _intensity_on_axis(plane)
        bucket_trace[index] = float(np.sum(plane[bucket]) * dx2)
        radial_r, radial_i = _radial_profile(plane, grid)
        minima, _ = find_peaks(-radial_i)
        eligible = minima[radial_r[minima] >= 1.5 * float(grid["dx"])]
        radius_trace[index] = float(
            radial_r[int(eligible[0])] if eligible.size else profile_definition["r_core_bucket_m"]
        )
        power[index] = float(np.sum(plane) * dx2)
        edges[index] = _edge_energy_fraction(plane, grid)
        if np.isclose(z_m, 60.0e-3):
            plane60 = np.asarray(plane, dtype=np.float32)
    if plane60 is None:
        propagated = angular_spectrum_propagate_bl(field, dict(grid), wavelength_m, 60.0e-3)
        plane60 = np.asarray(np.abs(propagated) ** 2, dtype=np.float32)
    spectral_radius = _spectral_support_radius(field, grid)
    nyquist_radius = np.pi / float(grid["dx"])
    return {
        "x_m": np.asarray(grid["x"], dtype=float),
        "z_m": np.asarray(z_values_m, dtype=float),
        "xz": xz,
        "yz": yz,
        "axis": axis_trace,
        "bucket": bucket_trace,
        "radius": radius_trace,
        "power": power,
        "edge": edges,
        "plane60": plane60,
        "spectral_support_radius_m_inv": spectral_radius,
        "spectral_nyquist_margin": float(nyquist_radius / max(spectral_radius, EPS)),
        "power_drift_fraction": float((np.max(power) - np.min(power)) / max(float(np.max(power)), EPS)),
        "maximum_edge_energy_fraction": float(np.max(edges)),
        "axial_beading_second_difference_rms": float(np.sqrt(np.mean(np.diff(_normalised(axis_trace), n=2) ** 2))),
        "upper_wing_line_fraction": float(
            np.sum(xz[z_values_m >= 50.0e-3][:, np.abs(np.asarray(grid["x"])) >= 0.50e-3])
            / max(float(np.sum(xz[z_values_m >= 50.0e-3])), EPS)
        ),
    }


def _compare_summary(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    reference_grid: Mapping[str, Any],
    candidate_grid: Mapping[str, Any],
) -> dict[str, float]:
    candidate_xz = _map_on_axes(
        candidate["xz"], candidate["x_m"], candidate["z_m"], reference["x_m"], reference["z_m"]
    )
    candidate_yz = _map_on_axes(
        candidate["yz"], candidate["x_m"], candidate["z_m"], reference["x_m"], reference["z_m"]
    )
    candidate_axis = np.interp(reference["z_m"], candidate["z_m"], candidate["axis"])
    candidate_bucket = np.interp(reference["z_m"], candidate["z_m"], candidate["bucket"])
    candidate_radius = np.interp(reference["z_m"], candidate["z_m"], candidate["radius"])
    source_axis = np.asarray(candidate_grid["x"], dtype=float)
    target_axis = np.asarray(reference_grid["x"], dtype=float)
    target_y, target_x = np.meshgrid(target_axis, target_axis, indexing="ij")
    candidate_plane = RegularGridInterpolator(
        (source_axis, source_axis), np.asarray(candidate["plane60"], dtype=float),
        method="linear", bounds_error=False, fill_value=0.0,
    )(np.column_stack((target_y.ravel(), target_x.ravel()))).reshape(reference["plane60"].shape)
    reference_plane = np.asarray(reference["plane60"], dtype=float)
    return {
        "xz_correlation": _safe_corr(reference["xz"], candidate_xz),
        "yz_correlation": _safe_corr(reference["yz"], candidate_yz),
        "on_axis_trace_normalised_l2_change": float(
            np.linalg.norm(_normalised(reference["axis"]) - _normalised(candidate_axis))
            / max(float(np.linalg.norm(_normalised(reference["axis"]))), EPS)
        ),
        "fixed_bucket_trace_normalised_l2_change": float(
            np.linalg.norm(_normalised(reference["bucket"]) - _normalised(candidate_bucket))
            / max(float(np.linalg.norm(_normalised(reference["bucket"]))), EPS)
        ),
        "fixed_bucket_power_change_at_60m_fraction": float(
            abs(candidate_bucket[-1] - reference["bucket"][-1])
            / max(float(reference["bucket"][-1]), EPS)
        ),
        "core_radius_change_at_60m_fraction": float(
            abs(candidate_radius[-1] - reference["radius"][-1])
            / max(float(reference["radius"][-1]), EPS)
        ),
        "common_plane_intensity_correlation": _safe_corr(reference_plane, candidate_plane),
    }


def run_b0_convergence_audit(
    profile_definition: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run bounded grid, window and z-step checks on one instrumented route."""

    important_z = np.arange(20.0e-3, 60.0e-3 + 0.5e-3, 1.0e-3)
    manifest = canonical_hardware_manifest()
    wavelength = float(hardware_value(manifest, "wavelength_m"))
    reference_checkpoints = build_scalar_route_checkpoints("B0", 0, grid_n=512)
    reference = _summary_from_source(
        reference_checkpoints["post_axicon"], reference_checkpoints["grid"],
        wavelength, important_z, profile_definition,
    )
    rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    grid_cache: dict[str, tuple[dict[str, Any], Mapping[str, Any], np.ndarray]] = {}

    def run(label: str, category: str, n: int, window_m: float, step_m: float) -> None:
        checkpoints = build_scalar_route_checkpoints(
            "B0", 0, grid_n=n, window_m=window_m
        )
        z_values = np.arange(20.0e-3, 60.0e-3 + 0.5 * step_m, step_m)
        summary = _summary_from_source(
            checkpoints["post_axicon"], checkpoints["grid"], wavelength,
            z_values, profile_definition,
        )
        comparison = _compare_summary(
            reference, summary, reference_checkpoints["grid"], checkpoints["grid"]
        )
        source_overlap, _ = _complex_overlap(
            reference_checkpoints["post_axicon"],
            _interpolate_complex(
                checkpoints["post_axicon"],
                np.asarray(checkpoints["grid"]["x"], dtype=float),
                np.asarray(reference_checkpoints["grid"]["x"], dtype=float),
            ),
        )
        slm_pitch = float(hardware_value(manifest, "slm_pixel_pitch_m"))
        samples_per_slm_pixel = slm_pitch / float(checkpoints["grid"]["dx"])
        gate = bool(
            summary["power_drift_fraction"] <= 0.05
            and 1.0 - comparison["common_plane_intensity_correlation"] <= 1.0e-3
            and comparison["core_radius_change_at_60m_fraction"] <= 0.01
            and comparison["fixed_bucket_power_change_at_60m_fraction"] <= 0.01
            and comparison["on_axis_trace_normalised_l2_change"] <= 0.01
        )
        rows.append({
            "configuration": label,
            "category": category,
            "grid_n": n,
            "window_m": window_m,
            "dx_m": float(checkpoints["grid"]["dx"]),
            "z_step_m": step_m,
            **comparison,
            "propagation_power_drift_fraction": summary["power_drift_fraction"],
            "maximum_edge_energy_fraction": summary["maximum_edge_energy_fraction"],
            "spectral_nyquist_margin": summary["spectral_nyquist_margin"],
            "samples_per_slm_pixel": samples_per_slm_pixel,
            "resolved_slm_pixel_sampling_pass": bool(samples_per_slm_pixel >= 2.0),
            "source_complex_overlap_to_N512": source_overlap,
            "axial_beading_second_difference_rms": summary["axial_beading_second_difference_rms"],
            "upper_wing_line_fraction": summary["upper_wing_line_fraction"],
            "predeclared_convergence_gate_pass": gate,
        })
        boundary_rows.append({
            "configuration": label,
            "window_m": window_m,
            "grid_n": n,
            "z_step_m": step_m,
            "maximum_edge_energy_fraction": summary["maximum_edge_energy_fraction"],
            "edge_energy_fraction_at_20mm": float(summary["edge"][0]),
            "edge_energy_fraction_at_60mm": float(summary["edge"][-1]),
            "upper_wing_line_fraction": summary["upper_wing_line_fraction"],
            "axial_beading_second_difference_rms": summary["axial_beading_second_difference_rms"],
        })
        if category == "grid_resolution" or category == "reference":
            grid_cache[label] = (
                summary,
                checkpoints["grid"],
                np.asarray(checkpoints["post_axicon"], dtype=np.complex128),
            )
        del checkpoints
        gc.collect()

    run("reference_N512_W10_dz1", "reference", 512, 10.0e-3, 1.0e-3)
    run("grid_N1024_W10_dz1", "grid_resolution", 1024, 10.0e-3, 1.0e-3)
    run("grid_N1536_W10_dz1", "grid_resolution", 1536, 10.0e-3, 1.0e-3)
    run("window_N768_W15_dz1", "physical_window", 768, 15.0e-3, 1.0e-3)
    run("window_N1024_W20_dz1", "physical_window", 1024, 20.0e-3, 1.0e-3)
    run("zstep_N512_W10_dz0p5", "z_step", 512, 10.0e-3, 0.5e-3)
    run("zstep_N512_W10_dz0p25", "z_step", 512, 10.0e-3, 0.25e-3)
    high_reference, high_reference_grid, _ = grid_cache["grid_N1024_W10_dz1"]
    high_candidate, high_candidate_grid, _ = grid_cache["grid_N1536_W10_dz1"]
    high_comparison = _compare_summary(
        high_reference, high_candidate, high_reference_grid, high_candidate_grid
    )
    high_gate = bool(
        high_candidate["power_drift_fraction"] <= 0.05
        and 1.0 - high_comparison["common_plane_intensity_correlation"] <= 1.0e-3
        and high_comparison["core_radius_change_at_60m_fraction"] <= 0.01
        and high_comparison["fixed_bucket_power_change_at_60m_fraction"] <= 0.01
        and high_comparison["on_axis_trace_normalised_l2_change"] <= 0.01
    )
    rows.append({
        "configuration": "grid_N1536_vs_N1024_crosscheck",
        "category": "grid_crosscheck",
        "grid_n": 1536,
        "window_m": 10.0e-3,
        "dx_m": 10.0e-3 / 1536.0,
        "z_step_m": 1.0e-3,
        **high_comparison,
        "propagation_power_drift_fraction": high_candidate["power_drift_fraction"],
        "maximum_edge_energy_fraction": high_candidate["maximum_edge_energy_fraction"],
        "spectral_nyquist_margin": high_candidate["spectral_nyquist_margin"],
        "samples_per_slm_pixel": float(hardware_value(manifest, "slm_pixel_pitch_m")) / (10.0e-3 / 1536.0),
        "resolved_slm_pixel_sampling_pass": False,
        "source_complex_overlap_to_N512": "not_applicable_high_grid_pair",
        "axial_beading_second_difference_rms": high_candidate["axial_beading_second_difference_rms"],
        "upper_wing_line_fraction": high_candidate["upper_wing_line_fraction"],
        "predeclared_convergence_gate_pass": high_gate,
    })
    return rows, boundary_rows


def _pupil_fields() -> tuple[dict[str, np.ndarray], Mapping[str, Any], dict[str, Any]]:
    checkpoints = build_scalar_route_checkpoints("B0", 0, grid_n=512)
    grid = checkpoints["grid"]
    pre_pupil = np.asarray(checkpoints["pre_pupil"], dtype=np.complex128)
    manifest = canonical_hardware_manifest()
    settings = _variant_settings("realistic_fixed_bench_route")
    axicon, _ = _axicon_phase(grid, manifest, settings)
    radius = np.asarray(grid["R"], dtype=float)
    pupil_radius = float(hardware_value(manifest, "objective_pupil_radius_m"))
    hard = radius <= pupil_radius
    taper_start = 0.85 * pupil_radius
    soft = np.ones_like(radius)
    transition = (radius > taper_start) & (radius < pupil_radius)
    soft[radius >= pupil_radius] = 0.0
    soft[transition] = 0.5 * (
        1.0 + np.cos(np.pi * (radius[transition] - taper_start) / (pupil_radius - taper_start))
    )
    canonical_post, _ = _pupil_and_aberration(pre_pupil, grid, pupil_radius, settings)
    fields = {
        "no_pupil": pre_pupil * axicon,
        "one_hard_circular_pupil": np.where(hard, pre_pupil, 0.0) * axicon,
        "one_soft_apodised_pupil": pre_pupil * soft * axicon,
        "canonical_realistic_route": canonical_post * axicon,
    }
    return fields, grid, {
        "pre_pupil_hash": _sha256_array(pre_pupil),
        "pupil_radius_m": pupil_radius,
        "soft_taper_start_fraction": 0.85,
        "classification": "nominal_model_consequence",
        "experimental_classification": "not_experimentally_physical_without_pupil_measurement",
    }


def run_pupil_audit(
    profile_definition: Mapping[str, Any],
    figure_stem: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fields, grid, metadata = _pupil_fields()
    wavelength = float(hardware_value(canonical_hardware_manifest(), "wavelength_m"))
    z_values = np.arange(0.0, 100.0e-3 + 0.5e-3, 1.0e-3)
    summaries = {
        name: _summary_from_source(field, grid, wavelength, z_values, profile_definition)
        for name, field in fields.items()
    }
    canonical = fields["canonical_realistic_route"]
    canonical_plane = summaries["canonical_realistic_route"]["plane60"]
    rows: list[dict[str, Any]] = []
    for name, field in fields.items():
        overlap, _ = _complex_overlap(canonical, field)
        summary = summaries[name]
        rows.append({
            "pupil_model": name,
            "classification": metadata["classification"],
            "same_pre_pupil_field_hash": metadata["pre_pupil_hash"],
            "pupil_application_count": 0 if name == "no_pupil" else 1,
            "retained_power_fraction": float(
                np.sum(np.abs(field) ** 2) / max(float(np.sum(np.abs(fields["no_pupil"]) ** 2)), EPS)
            ),
            "common_plane_complex_overlap_to_canonical": overlap,
            "output_intensity_correlation_at_60mm": _safe_corr(canonical_plane, summary["plane60"]),
            "fixed_core_power_at_60mm": float(summary["bucket"][60]),
            "core_radius_at_60mm": float(summary["radius"][60]),
            "core_radius_relative_std_20_60mm": float(
                np.std(summary["radius"][20:61]) / max(float(np.mean(summary["radius"][20:61])), EPS)
            ),
            "propagation_power_drift_fraction_0_100mm": summary["power_drift_fraction"],
            "maximum_edge_energy_fraction_0_100mm": summary["maximum_edge_energy_fraction"],
        })
    arrays = list(fields.values()) + [summary[key] for summary in summaries.values() for key in ("axis", "bucket", "radius", "plane60")]
    before = _hash_arrays(arrays)
    plt = _configure_matplotlib()
    figure, axes = plt.subplots(2, 4, figsize=(17.0, 8.0), constrained_layout=True)
    maximum = max(float(np.max(summary["plane60"])) for summary in summaries.values())
    axis_mm = np.asarray(grid["x"]) * 1e3
    extent = [axis_mm[0], axis_mm[-1], axis_mm[0], axis_mm[-1]]
    for column, (name, summary) in enumerate(summaries.items()):
        axes[0, column].imshow(
            summary["plane60"] / max(maximum, EPS), origin="lower", extent=extent,
            cmap="inferno", vmin=0.0, vmax=1.0, interpolation="none",
        )
        axes[0, column].set_xlim(-0.30, 0.30)
        axes[0, column].set_ylim(-0.30, 0.30)
        axes[0, column].set_title(name.replace("_", " "))
        axes[0, column].set_xlabel("x (mm)")
        axes[0, column].set_ylabel("y (mm)")
        axes[1, column].plot(z_values * 1e3, _normalised(summary["axis"]), label="on-axis")
        axes[1, column].plot(z_values * 1e3, _normalised(summary["bucket"]), label="fixed-core power")
        axes[1, column].axvspan(20.0, 60.0, color="0.8", alpha=0.25)
        axes[1, column].set_xlabel("z (mm)")
        axes[1, column].set_ylabel("trace / own maximum")
        axes[1, column].grid(alpha=0.2)
    axes[1, 0].legend(frameon=False)
    figure.suptitle(
        "B0 controlled pupil-model audit | same pre-pupil complex field | nominal model consequence",
        fontsize=14,
    )
    paths = _save_figure(figure, figure_stem)
    plt.close(figure)
    after = _hash_arrays(arrays)
    if before != after:
        raise RuntimeError("pupil-audit plotting mutated an input array")
    for summary in summaries.values():
        del summary
    gc.collect()
    return rows, {"paths": paths, "hashes_before": before, "hashes_after": after, **metadata}


def _plot_direct_route_comparison(
    canonical: CanonicalPropagation,
    traces: Mapping[str, np.ndarray],
    stem: Path,
) -> tuple[dict[str, Any], DensePropagationMap]:
    current_field, current_grid, metadata = _scalar_seed("B0", 0, grid_n=1024)
    direct = build_dense_spectral_propagation(
        grid=current_grid,
        wavelength_m=float(metadata["wavelength_m"]),
        z_values_m=canonical.z_m,
        transverse_coordinates_m=canonical.x_m,
        scalar_field=current_field,
        source_label="Phase 2E direct spectral-line route at N=1024",
    )
    arrays = [canonical.intensity_stack, direct.xz_intensity, direct.yz_intensity]
    before = _hash_arrays(arrays)
    plt = _configure_matplotlib()
    figure, axes = plt.subplots(2, 3, figsize=(16.0, 9.0), constrained_layout=True)
    canonical_xz = np.asarray(canonical.xz_intensity, dtype=float)
    canonical_yz = np.asarray(canonical.yz_intensity, dtype=float)
    direct_xz = np.asarray(direct.xz_intensity, dtype=float)
    direct_yz = np.asarray(direct.yz_intensity, dtype=float)
    maximum = max(float(np.max(canonical_xz)), float(np.max(canonical_yz)), EPS)
    extent = [canonical.x_m[0] * 1e3, canonical.x_m[-1] * 1e3, canonical.z_m[0] * 1e3, canonical.z_m[-1] * 1e3]
    panels = (canonical_xz / maximum, direct_xz / maximum, (direct_xz - canonical_xz) / maximum,
              canonical_yz / maximum, direct_yz / maximum, (direct_yz - canonical_yz) / maximum)
    titles = ("canonical x-z", "direct-route x-z", "signed x-z difference",
              "canonical y-z", "direct-route y-z", "signed y-z difference")
    difference_limit = max(float(np.max(np.abs(panels[2]))), float(np.max(np.abs(panels[5]))), EPS)
    for index, (axis, panel, title) in enumerate(zip(axes.ravel(), panels, titles)):
        difference = index in (2, 5)
        axis.imshow(
            panel, origin="lower", extent=extent, aspect="auto",
            cmap="RdBu_r" if difference else "inferno",
            vmin=-difference_limit if difference else 0.0,
            vmax=difference_limit if difference else 1.0,
            interpolation="none",
        )
        axis.set_title(title)
        axis.set_xlabel("transverse position (mm)")
        axis.set_ylabel("z (mm)")
    figure.suptitle("B0 accepted canonical volume versus former direct spectral-line renderer", fontsize=14)
    paths = _save_figure(figure, stem)
    plt.close(figure)
    after = _hash_arrays(arrays)
    if before != after:
        raise RuntimeError("direct-route comparison plotting mutated an input array")
    direct_axis_index = int(np.argmin(np.abs(direct.x_m)))
    direct_axis = 0.5 * (
        direct.xz_intensity[:, direct_axis_index] + direct.yz_intensity[:, direct_axis_index]
    )
    metrics = {
        "paths": paths,
        "hashes_before": before,
        "hashes_after": after,
        "xz_intensity_correlation": _safe_corr(canonical_xz, direct_xz),
        "yz_intensity_correlation": _safe_corr(canonical_yz, direct_yz),
        "on_axis_trace_correlation": _safe_corr(traces["feature_intensity"], direct_axis),
        "source_grid_n": 1024,
        "canonical_grid_n": 512,
        "interpretation": "separate renderer; not used as the canonical report volume",
    }
    return metrics, direct


def run_symmetry_audit() -> dict[str, Any]:
    checkpoints = build_scalar_route_checkpoints(
        "B0", 0, grid_n=512, variant="ideal_optical_route"
    )
    z = np.arange(0.0, 100.0e-3 + 1.0e-3, 2.0e-3)
    definition = {"r_core_bucket_m": 50.0e-6}
    summary = _summary_from_source(
        checkpoints["post_axicon"], checkpoints["grid"],
        float(checkpoints["metadata"]["wavelength_m"]), z, definition,
    )
    correlation = _safe_corr(summary["xz"], summary["yz"])
    result = {
        "route": "ideal_optical_route",
        "xz_yz_correlation": correlation,
        "required_minimum": 0.9999,
        "pass": bool(correlation >= 0.9999),
        "meshgrid_indexing": "xy",
        "centre_pixel_policy": "even grid; matched positive half-pixel centre slices",
        "pupil_offset_m": [0.0, 0.0],
        "pupil_centred": True,
        "transpose_or_axis_swap_detected": bool(correlation < 0.9999),
    }
    del checkpoints, summary
    gc.collect()
    return result


def _route_semantics() -> dict[str, Any]:
    config_z = np.linspace(0.0, 0.2, 601)
    manifest = canonical_hardware_manifest()
    return {
        "disputed_figure": "outputs/figures/phase2e_report_visualisation/01b_propagation_maps/b0_dense_xz_yz_global_linear_dual_range.png",
        "call_chain": [
            "phase2e_report_pipeline.generate_phase2e_outputs",
            "phase2e_report_visualisation.build_phase2e_data",
            "phase2e_report_visualisation._dense_scalar_map",
            "phase2b_visual_cases._scalar_seed",
            "phase2e_spectral_propagation.build_dense_spectral_propagation",
            "phase2e_report_figures.plot_dense_propagation_atlas",
        ],
        "initial_field_constructor": "phase2b_visual_cases._scalar_seed",
        "initial_field_semantic_plane": "axicon_output_plane",
        "plane_flags": {
            "slm_plane": False,
            "post_slm_plane": False,
            "fourier_filtered_plane": False,
            "objective_pupil_plane": False,
            "axicon_output_plane": True,
            "focal_plane": False,
            "sample_plane": False,
        },
        "plane_sequence": "SLM1 -> SLM2+carrier -> Fourier +1-order filter -> objective-pupil mask -> axicon output",
        "propagation_medium_index": 1.0,
        "wavelength_m": float(hardware_value(manifest, "wavelength_m")),
        "grid_n": 1024,
        "physical_window_m": 10.0e-3,
        "pixel_pitch_m": 10.0e-3 / 1024.0,
        "z_origin": "axicon_output_plane",
        "z_direction": "downstream_positive",
        "z_min_m": float(config_z[0]),
        "z_max_m": float(config_z[-1]),
        "z_samples": int(config_z.size),
        "z_step_m": float(config_z[1] - config_z[0]),
        "pupil_radius_m": float(hardware_value(manifest, "objective_pupil_radius_m")),
        "pupil_applied_at": "post-Fourier-filter numerical field immediately before axicon",
        "pupil_application_count": 1,
        "objective_transform_application_count": 0,
        "field_already_focused": False,
        "conical_phase_application_count": 1,
        "normalisation_mutates_complex_field": False,
        "normalisation_mutates_cached_intensity": False,
        "former_propagator": "direct centred inverse-DFT spectral-line synthesis with Matsushima bandlimit",
        "accepted_propagator": "full-plane angular_spectrum_propagate_bl used by Phase 2A/2B",
        "forensic_conclusion": "the source-plane meaning was known and not double-applied, but the report used a separate line-only propagation renderer and did not establish full-field power or boundary equivalence",
    }


def _figure_record(
    figure_id: str,
    paths: tuple[Path, Path],
    case_id: str,
    role: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "figure_family": "canonical_propagation_repair",
        "report_role": role,
        "png_path": paths[0].as_posix(),
        "pdf_path": paths[1].as_posix(),
        "case_ids": case_id,
        "source_artifacts": "Phase 2A/2B exact scalar route",
        "data_basis": "regenerated_exact_canonical_route full-plane complex propagation",
        "normalisation_policy": "paired global linear 0--1; no per-z normalisation",
        "linear_log_mode": "linear",
        "x_unit": "mm",
        "y_unit": "mm",
        "z_unit": "mm",
        "x_limits": [-5.0, 5.0],
        "y_limits": [0.0, 100.0],
        "comparison_group": "phase2e_canonical_propagation_repair",
        "matched_axes": True,
        "display_interpolation": "none",
        "metric_bearing": role == "main_text_candidate",
        "metrics_computed_on_native_arrays": True,
        "display_interpolation_used_for_metrics": False,
        "roi_occupancy": {},
        "superseded": False,
        "notes": notes,
    }


def update_phase2e_manifest(records: Sequence[Mapping[str, Any]]) -> None:
    root = Path("outputs/figures/phase2e_report_visualisation")
    json_path = root / "00_manifest/phase2e_figure_manifest.json"
    csv_path = root / "00_manifest/phase2e_figure_manifest.csv"
    figures = json.loads(json_path.read_text(encoding="utf-8"))
    replacements = {
        "propagation_b0": "b0_canonical_propagation_primary",
        "propagation_v1": "v1_canonical_propagation_primary",
        "propagation_v3": "v3_canonical_propagation_primary",
    }
    amended: list[dict[str, Any]] = []
    for row in figures:
        item = dict(row)
        if item.get("figure_id") in replacements:
            item["report_role"] = "superseded_diagnostic"
            item["superseded"] = True
            item["superseded_by"] = replacements[str(item["figure_id"])]
            item["supersession_reason"] = (
                "separate direct spectral-line renderer and plane-maximum primary trace were not the accepted canonical volume contract"
            )
        amended.append(item)
    new_ids = {str(row["figure_id"]) for row in records}
    amended = [row for row in amended if str(row.get("figure_id")) not in new_ids]
    amended.extend(dict(row) for row in records)
    _write_json(json_path, amended)
    _write_csv(csv_path, amended)


def refresh_phase2e_artifact_manifest() -> None:
    root = Path("outputs/figures/phase2e_report_visualisation")
    path = root / "00_manifest/phase2e_artifact_manifest.json"
    rows = []
    for artifact in sorted(item for item in root.rglob("*") if item.is_file() and item != path):
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        rows.append({
            "relative_path": artifact.relative_to(root).as_posix(),
            "size_bytes": artifact.stat().st_size,
            "sha256": digest,
        })
    _write_json(path, {
        "schema_version": "1.1.0",
        "stage": "phase2e_report_visualisation_and_parameter_sweeps_with_propagation_repair",
        "artifact_count_excluding_manifest": len(rows),
        "artifacts": rows,
    })


def generate_phase2e_propagation_repair() -> dict[str, Any]:
    """Generate only the forensic propagation replacements and audits."""

    VALIDATION_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    semantics = _route_semantics()
    _write_json(VALIDATION_ROOT / "current_route_semantics.json", semantics)
    comparison_rows, comparison_payload = compare_current_to_canonical_inputs()
    comparison_plot = _plot_input_comparison(
        comparison_payload, VALIDATION_ROOT / "canonical_vs_current_input"
    )

    figure_records: list[dict[str, Any]] = []
    mutation_rows: list[dict[str, Any]] = []
    profile_definitions: dict[str, Any] = {}
    axial_rows: list[dict[str, Any]] = []
    case_summaries: dict[str, Any] = {}
    b0_for_diagnostics: CanonicalPropagation | None = None
    b0_traces: dict[str, np.ndarray] | None = None
    for case_id in ("B0", "V1", "V3"):
        propagation = load_or_build_canonical_propagation(case_id)
        traces, summary = propagation_observables(propagation)
        definition = dict(summary["profile_definition"])
        profile_definitions[case_id] = definition
        case_summaries[case_id] = summary
        primary = plot_primary_propagation(
            propagation,
            traces,
            definition,
            FIGURE_ROOT / f"{case_id.lower()}_canonical_propagation_primary",
        )
        mutation_rows.append({
            "plot": f"{case_id.lower()}_canonical_propagation_primary",
            "hashes_equal": primary["hashes_before"] == primary["hashes_after"],
        })
        figure_records.append(_figure_record(
            f"{case_id.lower()}_canonical_propagation_primary",
            primary["paths"], case_id, "main_text_candidate",
            "Exact Phase 2A/2B source and full-plane BL-ASM; accepted 20--60 mm region marked; plane maximum is a thin secondary diagnostic only.",
        ))
        for index, z_m in enumerate(propagation.z_m):
            axial_rows.append({
                "case_id": case_id,
                "z_m": float(z_m),
                "feature_intensity_raw": float(traces["feature_intensity"][index]),
                "fixed_bucket_power_raw": float(traces["bucket_power"][index]),
                "feature_radius_m": float(traces["feature_radius_m"][index]),
                "plane_max_raw_diagnostic_only": float(traces["plane_max"][index]),
                "plane_peak_radius_m_diagnostic_only": float(traces["plane_peak_radius_m"][index]),
                "edge_energy_fraction": float(traces["edge_energy_fraction"][index]),
                "total_power": float(propagation.total_power[index]),
            })
        if case_id == "B0":
            b0_for_diagnostics = propagation
            b0_traces = dict(traces)
        else:
            del propagation, traces
            gc.collect()
    _write_json(VALIDATION_ROOT / "profile_definition.json", profile_definitions)
    _write_csv(VALIDATION_ROOT / "canonical_axial_metrics.csv", axial_rows)

    if b0_for_diagnostics is None or b0_traces is None:
        raise RuntimeError("B0 canonical result was not retained for diagnostics")
    low = plot_low_intensity_diagnostic(
        b0_for_diagnostics,
        FIGURE_ROOT / "b0_low_intensity_diagnostic",
        pupil_limit_m=112.5e-3,
        gaussian_limit_m=125.0e-3,
        diagnostic_z_values_m=np.arange(0.0, 140.0e-3 + 0.5e-3, 1.0e-3),
    )
    mutation_rows.append({"plot": "b0_low_intensity_diagnostic", "hashes_equal": low["hashes_before"] == low["hashes_after"]})
    figure_records.append(_figure_record(
        "b0_low_intensity_diagnostic", low["paths"], "B0", "diagnostic_only",
        "Linear intensity clipped at 0.01 of global maximum; weak structures visually amplified; not used for morphology metrics. Geometric limits are diagnostic estimates, not the accepted region.",
    ))
    direct_metrics, direct = _plot_direct_route_comparison(
        b0_for_diagnostics, b0_traces,
        FIGURE_ROOT / "b0_canonical_vs_direct_route",
    )
    mutation_rows.append({"plot": "b0_canonical_vs_direct_route", "hashes_equal": direct_metrics["hashes_before"] == direct_metrics["hashes_after"]})
    figure_records.append(_figure_record(
        "b0_canonical_vs_direct_route", direct_metrics["paths"], "B0", "diagnostic_only",
        "Same physical source-plane family compared as full-plane accepted propagation versus the former line-only report renderer; direct route is not used for primary figures.",
    ))
    pupil_rows, pupil_plot = run_pupil_audit(
        profile_definitions["B0"], FIGURE_ROOT / "b0_pupil_model_audit"
    )
    mutation_rows.append({"plot": "b0_pupil_model_audit", "hashes_equal": pupil_plot["hashes_before"] == pupil_plot["hashes_after"]})
    figure_records.append(_figure_record(
        "b0_pupil_model_audit", pupil_plot["paths"], "B0", "diagnostic_only",
        "Controlled same-pre-pupil comparison. Hard-pupil structure is classified as a nominal model consequence, not experimentally physical.",
    ))
    _write_csv(VALIDATION_ROOT / "pupil_audit.csv", pupil_rows)

    convergence_rows, boundary_rows = run_b0_convergence_audit(profile_definitions["B0"])
    _write_csv(VALIDATION_ROOT / "convergence_audit.csv", convergence_rows)
    _write_csv(VALIDATION_ROOT / "boundary_audit.csv", boundary_rows)
    symmetry = run_symmetry_audit()
    _write_json(VALIDATION_ROOT / "symmetry_audit.json", symmetry)
    mutation_rows.append({
        "plot": "canonical_vs_current_input",
        "hashes_equal": comparison_plot["hashes_before"] == comparison_plot["hashes_after"],
    })
    _write_csv(VALIDATION_ROOT / "plot_mutation_audit.csv", mutation_rows)

    convergence_pass = all(bool(row["predeclared_convergence_gate_pass"]) for row in convergence_rows)
    grid_rows = [row for row in convergence_rows if row["category"] == "grid_resolution"]
    window_rows = [row for row in convergence_rows if row["category"] == "physical_window"]
    z_rows = [row for row in convergence_rows if row["category"] == "z_step"]
    outcome_code = "PHASE2E-PROP-A" if convergence_pass and symmetry["pass"] else "PHASE2E-PROP-B"
    publication_authorised = bool(convergence_pass and symmetry["pass"])
    if not publication_authorised:
        for record in figure_records:
            if record["figure_id"].endswith("_canonical_propagation_primary"):
                record["report_role"] = "forensic_canonical_baseline_not_publication_authorised"
                record["notes"] += " Publication use is blocked by the predeclared grid-convergence gate."
    outcome = {
        "outcome": outcome_code,
        "canonical_baseline_figures_generated": True,
        "canonical_primary_figures_authorised": publication_authorised,
        "hard_pupil_features_experimentally_authorised": False,
        "previous_source_plane": semantics["initial_field_semantic_plane"],
        "current_direct_source_matches_canonical_gate": all(row["canonical_eligibility_pass"] for row in comparison_rows),
        "minimum_first_common_plane_complex_overlap": min(float(row["complex_overlap_after_global_phase_alignment"]) for row in comparison_rows),
        "pupil_applied_at_repository_contract_plane": True,
        "pupil_application_count": 1,
        "focused_field_mispropagated_as_axicon_field": False,
        "objective_transform_application_count": 0,
        "cached_array_mutation_detected": not all(bool(row["hashes_equal"]) for row in mutation_rows),
        "shared_transverse_peak_diagnosis": "plane maximum can follow whichever core or sidelobe is brightest and is not a fixed beam observable; it is retained only as a grey diagnostic trace",
        "case_summaries": case_summaries,
        "direct_route_comparison": {key: value for key, value in direct_metrics.items() if key not in {"paths", "hashes_before", "hashes_after"}},
        "symmetry_audit": symmetry,
        "convergence_all_predeclared_gates_pass": convergence_pass,
        "grid_resolution_gates_pass": all(bool(row["predeclared_convergence_gate_pass"]) for row in grid_rows),
        "physical_window_gates_pass": all(bool(row["predeclared_convergence_gate_pass"]) for row in window_rows),
        "z_step_gates_pass": all(bool(row["predeclared_convergence_gate_pass"]) for row in z_rows),
        "all_tested_grids_resolve_two_samples_per_slm_pixel": all(
            bool(row["resolved_slm_pixel_sampling_pass"])
            for row in convergence_rows
            if row["category"] in {"reference", "grid_resolution"}
        ),
        "minimum_tested_samples_per_slm_pixel": min(
            float(row["samples_per_slm_pixel"])
            for row in convergence_rows
            if row["category"] in {"reference", "grid_resolution"}
        ),
        "maximum_tested_samples_per_slm_pixel": max(
            float(row["samples_per_slm_pixel"])
            for row in convergence_rows
            if row["category"] in {"reference", "grid_resolution"}
        ),
        "grid_blocker": "accepted N=512 and tested N=1024/N=1536 grids do not meet the SLM model's two-computational-samples-per-8-um-pixel resolved threshold; fixed-core metrics are not grid converged",
        "maximum_edge_energy_fraction": max(float(row["maximum_edge_energy_fraction"]) for row in convergence_rows),
        "structure_classification": {
            "global_linear_primary_structure": "accepted canonical baseline structure; publication interpretation blocked by grid convergence",
            "one_percent_clipped_diagonal_and_vertical_structure": "display_emphasised_weak_structure",
            "hard_pupil_difference": "nominal_hard_pupil_diffraction_model_consequence",
            "experimentally_physical": False,
        },
        "accepted_phase2a_region_m": list(CANONICAL_REGION_M),
        "geometric_estimates_replace_canonical_region": False,
        "old_disputed_outputs_overwritten": False,
        "old_disputed_outputs_marked_superseded": True,
        "plot_mutation_audit": mutation_rows,
        "replacement_figures": [record["png_path"] for record in figure_records],
    }
    _write_json(VALIDATION_ROOT / "propagation_repair_outcome.json", outcome)
    update_phase2e_manifest(figure_records)
    refresh_phase2e_artifact_manifest()
    del b0_for_diagnostics, b0_traces, direct
    gc.collect()
    return outcome


__all__ = [
    "CANONICAL_REGION_M",
    "CanonicalPropagation",
    "build_scalar_route_checkpoints",
    "compare_current_to_canonical_inputs",
    "generate_phase2e_propagation_repair",
    "load_or_build_canonical_propagation",
    "plot_primary_propagation",
    "propagation_observables",
    "run_b0_convergence_audit",
    "run_symmetry_audit",
]
