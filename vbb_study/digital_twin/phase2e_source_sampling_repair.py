"""Source-sampling and optical-route audit for Phase 2E scalar Bessel propagation.

This module does not overwrite Phase 2A/2B/2E outputs.  It provides an
independent, bounded audit of the numerical sampling used to represent the
axicon phase and separates source-scale free-space propagation from
objective-focused/sample-scale propagation.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from vbb_study.digital_twin.phase2a_contracts import (
    canonical_hardware_manifest,
    hardware_value,
)
from vbb_study.equations.fields import compute_kr, make_xy_grid
from vbb_study.equations.propagation import angular_spectrum_propagate_bl


EPS = np.finfo(float).tiny
DEFAULT_OUTPUT_ROOT = Path("outputs/validation/phase2e_source_sampling_repair")
DEFAULT_FIGURE_ROOT = Path("outputs/figures/phase2e_source_sampling_repair")
DEFAULT_WINDOW_M = 10.0e-3
DEFAULT_N_VALUES = (512, 768, 1024, 1280, 1536, 2048)
DEFAULT_COMPARE_Z_M = (20e-3, 40e-3, 60e-3, 80e-3, 100e-3)


@dataclass(frozen=True)
class SamplingDiagnostic:
    grid_n: int
    window_m: float
    dx_m: float
    kr_m_inv: float
    radial_period_m: float
    samples_per_radial_period: float
    adjacent_radial_phase_increment_rad: float
    carrier_period_m: float
    samples_per_carrier_period: float
    sampling_class: str
    quantitative_reference: bool


@dataclass(frozen=True)
class AxisymmetricTrace:
    z_m: np.ndarray
    intensity: np.ndarray
    metadata: Mapping[str, Any]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2) + "\n", encoding="utf-8")
    return path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    materialised = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in materialised:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialised)
    return path


def canonical_sampling_parameters() -> dict[str, float]:
    manifest = canonical_hardware_manifest()
    wavelength = float(hardware_value(manifest, "wavelength_m"))
    n_axicon = float(hardware_value(manifest, "axicon_refractive_index"))
    n_medium = float(hardware_value(manifest, "axicon_external_medium_index"))
    gamma = math.radians(float(hardware_value(manifest, "axicon_base_angle_deg")))
    kr = compute_kr(2.0 * math.pi / wavelength, n_axicon, n_medium, gamma)
    carrier = float(hardware_value(manifest, "carrier_frequency_cpm"))
    return {
        "wavelength_m": wavelength,
        "n_axicon": n_axicon,
        "n_medium": n_medium,
        "axicon_base_angle_rad": gamma,
        "kr_m_inv": kr,
        "carrier_frequency_cpm": carrier,
        "carrier_period_m": 1.0 / carrier,
        "beam_radius_m": float(hardware_value(manifest, "beam_radius_on_slm_m")),
        "nominal_aperture_radius_m": float(hardware_value(manifest, "objective_pupil_radius_m")),
    }


def classify_sampling(samples_per_period: float) -> str:
    if samples_per_period < 4.0:
        return "invalid"
    if samples_per_period < 8.0:
        return "marginal"
    if samples_per_period < 12.0:
        return "acceptable_for_screening"
    return "quantitative_reference"


def sampling_diagnostic(grid_n: int, window_m: float = DEFAULT_WINDOW_M) -> SamplingDiagnostic:
    params = canonical_sampling_parameters()
    dx = float(window_m) / int(grid_n)
    radial_period = 2.0 * math.pi / params["kr_m_inv"]
    samples = radial_period / dx
    phase_step = params["kr_m_inv"] * dx
    carrier_period = params["carrier_period_m"]
    carrier_samples = carrier_period / dx
    label = classify_sampling(samples)
    return SamplingDiagnostic(
        grid_n=int(grid_n),
        window_m=float(window_m),
        dx_m=dx,
        kr_m_inv=params["kr_m_inv"],
        radial_period_m=radial_period,
        samples_per_radial_period=samples,
        adjacent_radial_phase_increment_rad=phase_step,
        carrier_period_m=carrier_period,
        samples_per_carrier_period=carrier_samples,
        sampling_class=label,
        quantitative_reference=(label == "quantitative_reference" and phase_step <= math.pi / 2.0),
    )


def sampling_table(
    n_values: Iterable[int] = DEFAULT_N_VALUES,
    window_m: float = DEFAULT_WINDOW_M,
) -> list[dict[str, Any]]:
    return [sampling_diagnostic(n, window_m).__dict__.copy() for n in n_values]


def optical_route_contract() -> dict[str, Any]:
    return {
        "route_S_source_scale": {
            "source_plane": "post-4F/post-aperture axicon-output plane",
            "sequence": [
                "SLM field",
                "first-order Fourier filter",
                "nominal hard aperture placeholder",
                "axicon phase",
                "free-space propagation in air",
            ],
            "objective_transform": "none",
            "z_origin": "axicon output",
            "permitted_claims": [
                "source-scale Bessel formation",
                "source-scale finite-aperture axial structure",
                "relative morphology",
            ],
            "blocked_claims": [
                "objective-focused sample-plane dimensions",
                "Debye focal components",
                "calibrated material-plane fluence",
            ],
            "aperture_semantics": "nominal_hard_aperture_placeholder_until_calibrated",
        },
        "route_F_objective_sample_scale": {
            "source_plane": "objective entrance pupil",
            "sequence": [
                "objective pupil field",
                "scalar Fourier or vector Debye transform",
                "focal/sample plane",
                "optional vector Fresnel interface",
                "material propagation",
            ],
            "objective_transform": "required",
            "z_origin": "declared focal or sample interface plane",
            "permitted_claims": [
                "focal morphology",
                "longitudinal-field fraction",
                "matched sample-plane interface comparison",
            ],
            "blocked_claims": [
                "source-scale free-space Bessel-zone length unless separately modelled"
            ],
        },
    }


def analytic_b0_source(
    grid_n: int,
    *,
    window_m: float = DEFAULT_WINDOW_M,
    aperture_model: str = "hard",
    aperture_radius_m: float | None = None,
) -> tuple[np.ndarray, Mapping[str, Any], dict[str, Any]]:
    """Construct a controlled Gaussian-axicon source independent of the SLM/4F route."""

    params = canonical_sampling_parameters()
    grid = make_xy_grid(int(grid_n), float(window_m) / int(grid_n))
    radius = np.asarray(grid["R"], dtype=float)
    amplitude = np.exp(-(radius**2) / params["beam_radius_m"] ** 2)
    if aperture_radius_m is None:
        aperture_radius_m = params["nominal_aperture_radius_m"]
    if aperture_model == "none":
        aperture = np.ones_like(radius)
    elif aperture_model == "hard":
        aperture = (radius <= float(aperture_radius_m)).astype(float)
    elif aperture_model == "soft":
        aperture = np.exp(-((radius / float(aperture_radius_m)) ** 8))
    elif aperture_model == "large_hard":
        aperture = (radius <= 1.5 * float(aperture_radius_m)).astype(float)
    else:
        raise ValueError(f"unknown aperture model {aperture_model!r}")
    field = amplitude * aperture * np.exp(-1j * params["kr_m_inv"] * radius)
    return np.asarray(field, dtype=np.complex128), grid, {
        **params,
        "grid_n": int(grid_n),
        "window_m": float(window_m),
        "dx_m": float(grid["dx"]),
        "aperture_model": aperture_model,
        "aperture_radius_m": float(aperture_radius_m),
        "route_id": "route_S_source_scale",
    }


def axisymmetric_on_axis_trace(
    z_values_m: Sequence[float],
    *,
    radial_samples: int = 32768,
    radial_extent_m: float | None = None,
    aperture_model: str = "hard",
    aperture_radius_m: float | None = None,
) -> AxisymmetricTrace:
    """Independent paraxial axisymmetric on-axis Fresnel reference for ideal B0."""

    params = canonical_sampling_parameters()
    if radial_extent_m is None:
        radial_extent_m = 0.5 * DEFAULT_WINDOW_M
    if aperture_radius_m is None:
        aperture_radius_m = params["nominal_aperture_radius_m"]
    r = np.linspace(0.0, float(radial_extent_m), int(radial_samples), dtype=float)
    amplitude = np.exp(-(r**2) / params["beam_radius_m"] ** 2)
    if aperture_model == "none":
        aperture = np.ones_like(r)
    elif aperture_model == "hard":
        aperture = (r <= float(aperture_radius_m)).astype(float)
    elif aperture_model == "soft":
        aperture = np.exp(-((r / float(aperture_radius_m)) ** 8))
    elif aperture_model == "large_hard":
        aperture = (r <= 1.5 * float(aperture_radius_m)).astype(float)
    else:
        raise ValueError(f"unknown aperture model {aperture_model!r}")
    source = amplitude * aperture * np.exp(-1j * params["kr_m_inv"] * r)
    k = 2.0 * math.pi / params["wavelength_m"]
    z_values = np.asarray(z_values_m, dtype=float)
    values = np.full(z_values.shape, np.nan, dtype=float)
    for index, z_m in enumerate(z_values):
        if z_m <= 0.0:
            continue
        phase = np.exp(1j * k * r**2 / (2.0 * z_m))
        field = np.trapezoid(source * phase * r, x=r) / z_m
        values[index] = float(abs(field) ** 2)
    return AxisymmetricTrace(
        z_m=z_values,
        intensity=values,
        metadata={
            "method": "independent_axisymmetric_paraxial_Fresnel_on_axis_integral",
            "radial_samples": int(radial_samples),
            "radial_extent_m": float(radial_extent_m),
            "aperture_model": aperture_model,
            "not_called": "2D BL-ASM",
        },
    )


def selected_plane_convergence(
    n_values: Sequence[int],
    *,
    z_values_m: Sequence[float] = DEFAULT_COMPARE_Z_M,
    window_m: float = DEFAULT_WINDOW_M,
    aperture_model: str = "hard",
) -> list[dict[str, Any]]:
    """Compare central intensity and total power at a bounded set of z planes."""

    rows: list[dict[str, Any]] = []
    for n in n_values:
        field, grid, metadata = analytic_b0_source(
            int(n), window_m=window_m, aperture_model=aperture_model
        )
        centre = int(n) // 2
        for z_m in z_values_m:
            propagated = angular_spectrum_propagate_bl(
                field,
                dict(grid),
                metadata["wavelength_m"],
                float(z_m),
                n_medium=1.0,
                bandlimit=True,
                include_evanescent=True,
            )
            intensity = np.abs(propagated) ** 2
            rows.append({
                "grid_n": int(n),
                "dx_m": float(grid["dx"]),
                "samples_per_radial_period": sampling_diagnostic(int(n), window_m).samples_per_radial_period,
                "aperture_model": aperture_model,
                "z_m": float(z_m),
                "on_axis_intensity_raw": float(intensity[centre, centre]),
                "total_power_raw": float(np.sum(intensity) * float(grid["dx"]) ** 2),
            })
    return rows


def run_quick_audit(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    figure_root: Path = DEFAULT_FIGURE_ROOT,
    n_values: Sequence[int] = (512, 768, 1024, 1536),
) -> dict[str, Any]:
    """Run a bounded audit suitable for a workstation before the expensive full sweep."""

    output_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    sampling = sampling_table(n_values)
    _write_csv(output_root / "source_sampling_summary.csv", sampling)
    _write_json(output_root / "optical_route_contract.json", optical_route_contract())

    selected_rows = selected_plane_convergence(n_values)
    _write_csv(output_root / "selected_plane_convergence.csv", selected_rows)

    z = np.arange(1.0e-3, 180.0e-3 + 0.25e-3, 0.25e-3)
    traces = {
        model: axisymmetric_on_axis_trace(z, aperture_model=model)
        for model in ("none", "hard", "soft", "large_hard")
    }
    trace_rows: list[dict[str, Any]] = []
    for model, result in traces.items():
        scale = max(float(np.nanmax(result.intensity)), EPS)
        for z_m, value in zip(result.z_m, result.intensity):
            trace_rows.append({
                "aperture_model": model,
                "z_m": float(z_m),
                "intensity_raw": float(value),
                "intensity_normalised": float(value / scale),
            })
    _write_csv(output_root / "axisymmetric_aperture_traces.csv", trace_rows)

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    n_axis = [row["grid_n"] for row in sampling]
    spp = [row["samples_per_radial_period"] for row in sampling]
    axes[0].plot(n_axis, spp, marker="o")
    axes[0].axhline(4.0, linestyle="--", linewidth=1.0, label="minimum non-marginal")
    axes[0].axhline(8.0, linestyle=":", linewidth=1.0, label="screening")
    axes[0].axhline(12.0, linestyle="-.", linewidth=1.0, label="quantitative reference")
    axes[0].set_xlabel("source grid N (10 mm window)")
    axes[0].set_ylabel("samples per axicon radial phase period")
    axes[0].set_title("Source-plane axicon sampling")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    for model, result in traces.items():
        axes[1].plot(
            result.z_m * 1e3,
            result.intensity / max(float(np.nanmax(result.intensity)), EPS),
            label=model,
        )
    axes[1].set_xlabel("z (mm)")
    axes[1].set_ylabel("on-axis intensity / own maximum")
    axes[1].set_title("Independent axisymmetric aperture audit")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    figure.savefig(figure_root / "b0_source_sampling_quick_audit.png", dpi=220)
    figure.savefig(figure_root / "b0_source_sampling_quick_audit.pdf")
    plt.close(figure)

    result = {
        "outcome": "PHASE2E-SAMPLING-QUICK-AUDIT",
        "report_figures_authorised": False,
        "n_values": [int(v) for v in n_values],
        "n512_sampling_class": sampling[0]["sampling_class"],
        "n512_samples_per_radial_period": sampling[0]["samples_per_radial_period"],
        "route_contract_written": True,
        "selected_plane_2d_asm_completed": True,
        "axisymmetric_reference_completed": True,
        "next_required_step": "run full transverse and z convergence before authorising report figures",
    }
    _write_json(output_root / "quick_audit_outcome.json", result)
    return result
