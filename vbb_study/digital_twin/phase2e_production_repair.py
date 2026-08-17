"""Phase 2E production repair for source-scale Bessel propagation.

This module promotes the forensic Phase 2E findings into an explicit nominal
source-scale route.  The nominal route retains the Gaussian/SLM/4F field and
axicon, but does not insert the historical 1.8 mm hard ``objective pupil`` in
front of the axicon.  Hard/soft apertures remain available only as labelled
sensitivity cases.

Report-facing propagation is not authorised by this module until the declared
source sampling and selected-plane convergence gates pass.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.digital_twin.phase2a_canonical import _axicon_phase
from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.phase2e_propagation_repair import build_scalar_route_checkpoints
from vbb_study.digital_twin.phase2e_source_sampling_repair import sampling_diagnostic
from vbb_study.digital_twin.phase2e_spectral_propagation import on_axis_spectral_intensity
from vbb_study.equations.propagation import angular_spectrum_propagate_bl


DEFAULT_WINDOW_M = 10.0e-3
DEFAULT_PRODUCTION_N = 3072
DEFAULT_REFERENCE_N_VALUES = (2048, 2560, 3072)
DEFAULT_COMPARE_Z_M = (20e-3, 40e-3, 60e-3, 80e-3, 100e-3)
DEFAULT_OUTPUT_ROOT = Path("outputs/validation/phase2e_production_repair")
EPS = np.finfo(float).tiny


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    materialised = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialised:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialised)


def _ell(case_id: str) -> int:
    try:
        return {"B0": 0, "V1": 1, "V3": 3}[case_id]
    except KeyError as exc:
        raise ValueError(f"unsupported production case {case_id!r}") from exc


def production_sampling_gate(grid_n: int, window_m: float = DEFAULT_WINDOW_M) -> dict[str, Any]:
    diagnostic = sampling_diagnostic(int(grid_n), float(window_m))
    phase_preferred = diagnostic.adjacent_radial_phase_increment_rad <= math.pi / 4.0
    quantitative = diagnostic.samples_per_radial_period >= 12.0
    return {
        **diagnostic.__dict__,
        "minimum_samples_gate": bool(quantitative),
        "preferred_phase_increment_gate": bool(phase_preferred),
        "production_sampling_pass": bool(quantitative and phase_preferred),
    }


def build_nominal_source(
    case_id: str,
    *,
    grid_n: int = DEFAULT_PRODUCTION_N,
    window_m: float = DEFAULT_WINDOW_M,
    aperture_model: str = "none",
) -> tuple[np.ndarray, Mapping[str, Any], dict[str, Any]]:
    """Build the repaired source-scale field at the axicon-output plane.

    ``aperture_model='none'`` is the nominal production route.  ``hard`` and
    ``soft`` are sensitivity cases only and never become nominal implicitly.
    """

    checkpoints = build_scalar_route_checkpoints(
        case_id,
        _ell(case_id),
        grid_n=int(grid_n),
        window_m=float(window_m),
        variant="realistic_fixed_bench_route",
    )
    grid = checkpoints["grid"]
    pre_pupil = np.asarray(checkpoints["pre_pupil"], dtype=np.complex128)
    manifest = canonical_hardware_manifest()
    settings = {
        "axicon_decentre_m": (0.0, 0.0),
        "axicon_tilt_rad": (0.0, 0.0),
    }

    radius = np.asarray(grid["R"], dtype=float)
    nominal_radius = float(hardware_value(manifest, "objective_pupil_radius_m"))
    if aperture_model == "none":
        post_aperture = pre_pupil
        retained = 1.0
        aperture_role = "no_additional_real_space_aperture"
    elif aperture_model == "hard":
        mask = radius <= nominal_radius
        post_aperture = np.where(mask, pre_pupil, 0.0)
        retained = float(np.sum(np.abs(post_aperture) ** 2) / max(np.sum(np.abs(pre_pupil) ** 2), EPS))
        aperture_role = "historical_1p8mm_hard_sensitivity_case"
    elif aperture_model == "soft":
        transmission = np.exp(-((radius / nominal_radius) ** 8))
        post_aperture = pre_pupil * transmission
        retained = float(np.sum(np.abs(post_aperture) ** 2) / max(np.sum(np.abs(pre_pupil) ** 2), EPS))
        aperture_role = "soft_edge_sensitivity_case"
    else:
        raise ValueError("aperture_model must be one of: none, hard, soft")

    axicon, kr = _axicon_phase(grid, manifest, settings)
    source = np.asarray(post_aperture * axicon, dtype=np.complex128)
    gate = production_sampling_gate(int(grid_n), float(window_m))
    metadata = {
        **dict(checkpoints["metadata"]),
        "route_id": "phase2e_source_scale_nominal_no_additional_aperture",
        "source_plane": "axicon_output_plane",
        "objective_transform_application_count": 0,
        "historical_objective_pupil_application_count": 0 if aperture_model == "none" else 1,
        "aperture_model": aperture_model,
        "aperture_role": aperture_role,
        "aperture_retained_power_fraction": retained,
        "radial_wavevector_m_inv": float(kr),
        "production_sampling_pass": gate["production_sampling_pass"],
        "samples_per_radial_period": gate["samples_per_radial_period"],
        "adjacent_radial_phase_increment_rad": gate["adjacent_radial_phase_increment_rad"],
    }
    return source, grid, metadata


def physical_on_axis_trace(
    source: np.ndarray,
    grid: Mapping[str, Any],
    wavelength_m: float,
    z_values_m: Sequence[float],
) -> np.ndarray:
    """Evaluate I(0,0,z) at the physical origin, not at a half-pixel native sample."""

    return on_axis_spectral_intensity(
        grid=grid,
        wavelength_m=float(wavelength_m),
        z_values_m=z_values_m,
        scalar_field=source,
        n_medium=1.0,
        bandlimit=True,
    )


def selected_plane_reference_rows(
    case_id: str,
    *,
    n_values: Sequence[int] = DEFAULT_REFERENCE_N_VALUES,
    z_values_m: Sequence[float] = DEFAULT_COMPARE_Z_M,
    window_m: float = DEFAULT_WINDOW_M,
) -> list[dict[str, Any]]:
    """Run fixed-window selected-plane convergence for the repaired nominal route."""

    rows: list[dict[str, Any]] = []
    for n in n_values:
        source, grid, metadata = build_nominal_source(
            case_id, grid_n=int(n), window_m=window_m, aperture_model="none"
        )
        wavelength = float(metadata["wavelength_m"])
        on_axis = physical_on_axis_trace(source, grid, wavelength, z_values_m)
        for z_m, axis_value in zip(z_values_m, on_axis):
            propagated = angular_spectrum_propagate_bl(
                source,
                dict(grid),
                wavelength,
                float(z_m),
                n_medium=1.0,
                bandlimit=True,
                include_evanescent=True,
            )
            intensity = np.abs(propagated) ** 2
            rows.append({
                "case_id": case_id,
                "grid_n": int(n),
                "window_m": float(window_m),
                "dx_m": float(grid["dx"]),
                "samples_per_radial_period": float(metadata["samples_per_radial_period"]),
                "z_m": float(z_m),
                "physical_on_axis_intensity_raw": float(axis_value),
                "total_power_raw": float(np.sum(intensity) * float(grid["dx"]) ** 2),
                "edge_power_fraction": float(
                    (
                        np.sum(intensity[:8, :]) + np.sum(intensity[-8:, :])
                        + np.sum(intensity[:, :8]) + np.sum(intensity[:, -8:])
                    ) / max(np.sum(intensity), EPS)
                ),
            })
            del propagated, intensity
    return rows


def summarise_reference_convergence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    data = [dict(row) for row in rows]
    reference_n = max(int(row["grid_n"]) for row in data)
    by_key = {(int(row["grid_n"]), float(row["z_m"])): row for row in data}
    z_values = sorted({float(row["z_m"]) for row in data})
    n_values = sorted({int(row["grid_n"]) for row in data})
    comparisons: list[dict[str, Any]] = []
    worst = 0.0
    for n in n_values:
        if n == reference_n:
            continue
        for z in z_values:
            value = float(by_key[(n, z)]["physical_on_axis_intensity_raw"])
            ref = float(by_key[(reference_n, z)]["physical_on_axis_intensity_raw"])
            relative = abs(value - ref) / max(abs(ref), EPS)
            worst = max(worst, relative)
            comparisons.append({"grid_n": n, "reference_n": reference_n, "z_m": z, "relative_on_axis_difference": relative})
    reference_gate = production_sampling_gate(reference_n)
    return {
        "reference_n": reference_n,
        "reference_sampling_pass": bool(reference_gate["production_sampling_pass"]),
        "worst_nonreference_relative_on_axis_difference": float(worst),
        "comparisons": comparisons,
    }


def run_production_repair(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    cases: Sequence[str] = ("B0", "V1", "V3"),
    n_values: Sequence[int] = DEFAULT_REFERENCE_N_VALUES,
    z_values_m: Sequence[float] = DEFAULT_COMPARE_Z_M,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    sampling_rows = [production_sampling_gate(n) for n in n_values]
    _write_csv(output_root / "production_sampling_gate.csv", sampling_rows)

    all_rows: list[dict[str, Any]] = []
    case_summaries: dict[str, Any] = {}
    for case_id in cases:
        rows = selected_plane_reference_rows(case_id, n_values=n_values, z_values_m=z_values_m)
        all_rows.extend(rows)
        case_summaries[case_id] = summarise_reference_convergence(rows)
    _write_csv(output_root / "selected_plane_reference_convergence.csv", all_rows)

    production_gate = production_sampling_gate(max(n_values))
    result = {
        "outcome": "PHASE2E-PRODUCTION-REPAIR-CHECK",
        "nominal_route": "SLM/4F field -> no additional real-space hard pupil -> axicon -> free-space",
        "historical_1p8mm_hard_pupil_nominal": False,
        "hard_and_soft_apertures": "sensitivity_only",
        "physical_on_axis_definition": "spectral evaluation at x=y=0",
        "production_n": int(max(n_values)),
        "production_sampling_pass": bool(production_gate["production_sampling_pass"]),
        "case_summaries": case_summaries,
        "report_figures_authorised": False,
        "next_required_step": "inspect convergence, then regenerate report-facing propagation maps only if quantitative gates are accepted",
    }
    _write_json(output_root / "production_repair_outcome.json", result)
    return result
