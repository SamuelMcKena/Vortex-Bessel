"""Common transverse line-profile evidence for Bessel/vortex system studies.

The purpose of this module is not cosmetic plotting.  It defines a single
physical convention for one-dimensional intensity evidence used by system-error,
aberration and report figures:

* laboratory-frame profiles retain steering/decentre information;
* morphology-frame profiles are sampled through the detected Bessel/vortex axis;
* complex fields are evaluated by the discrete Fourier series at requested
  physical coordinates, rather than interpolating rendered intensity images;
* absolute/common-normalised quantities remain distinct from shape-only
  normalisation.

For a sweep, all common-scale profiles should be divided by the nominal case's
2-D reference-plane peak intensity.  This allows a local line peak above one to
be interpreted correctly as local intensity redistribution rather than as a
claim of increased total optical power.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.digital_twin.vortex_following_propagation import (
    TransverseMorphologyAxis,
    transverse_morphology_axis,
)
from vbb_study.equations.fields import fft2c


EPS = np.finfo(float).tiny


@dataclass(frozen=True)
class TransverseProfileEvidence:
    lab_coordinate_m: np.ndarray = field(repr=False, compare=False)
    relative_coordinate_m: np.ndarray = field(repr=False, compare=False)
    lab_x_intensity: np.ndarray = field(repr=False, compare=False)
    lab_y_intensity: np.ndarray = field(repr=False, compare=False)
    axis_x_intensity: np.ndarray = field(repr=False, compare=False)
    axis_y_intensity: np.ndarray = field(repr=False, compare=False)
    morphology_axis: TransverseMorphologyAxis
    total_2d_power_au_m2: float
    peak_2d_au: float
    metadata: Mapping[str, Any]


def _phase_matrix(
    coordinates_m: np.ndarray,
    *,
    n: int,
    dx_m: float,
    centre_coordinate_m: float,
) -> np.ndarray:
    frequency_index = np.arange(int(n), dtype=float) - int(n) / 2.0
    sample_coordinate = (
        np.asarray(coordinates_m, dtype=float) - float(centre_coordinate_m)
    ) / float(dx_m)
    return np.exp(2j * np.pi * np.outer(frequency_index, sample_coordinate) / float(n))


def spectral_line_fields(
    field: np.ndarray,
    grid: Mapping[str, Any],
    *,
    x_coordinates_m: Sequence[float],
    y_coordinates_m: Sequence[float],
    fixed_x_m: float,
    fixed_y_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate two orthogonal complex line fields by direct Fourier synthesis.

    The x line is sampled at ``y=fixed_y_m`` and the y line at
    ``x=fixed_x_m``.  Requested coordinates may be sub-pixel with respect to the
    native grid; no intensity-image interpolation is used.
    """

    u = np.asarray(field, dtype=np.complex128)
    n = int(grid["N"])
    if u.shape != (n, n):
        raise ValueError("field must match the declared square grid")
    dx = float(grid["dx"])
    native = np.asarray(grid["x"], dtype=float)
    centre = float(native[n // 2])
    x = np.asarray(x_coordinates_m, dtype=float)
    y = np.asarray(y_coordinates_m, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("profile coordinates must be one-dimensional")

    spectrum = fft2c(u)
    phase_x = _phase_matrix(x, n=n, dx_m=dx, centre_coordinate_m=centre)
    phase_y = _phase_matrix(y, n=n, dx_m=dx, centre_coordinate_m=centre)
    phase_fixed_y = _phase_matrix(
        np.asarray([float(fixed_y_m)]),
        n=n,
        dx_m=dx,
        centre_coordinate_m=centre,
    )[:, 0]
    phase_fixed_x = _phase_matrix(
        np.asarray([float(fixed_x_m)]),
        n=n,
        dx_m=dx,
        centre_coordinate_m=centre,
    )[:, 0]

    x_line = phase_fixed_y @ spectrum @ phase_x / float(n * n)
    y_line = spectrum @ phase_fixed_x @ phase_y / float(n * n)
    return (
        np.asarray(x_line, dtype=np.complex128),
        np.asarray(y_line, dtype=np.complex128),
    )


def _energy_centroid(field: np.ndarray, grid: Mapping[str, Any]) -> tuple[float, float]:
    intensity = np.abs(np.asarray(field, dtype=np.complex128)) ** 2
    total = float(np.sum(intensity))
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    return (
        float(np.sum(intensity * X) / max(total, EPS)),
        float(np.sum(intensity * Y) / max(total, EPS)),
    )


def build_transverse_profile_evidence(
    field: np.ndarray,
    grid: Mapping[str, Any],
    *,
    vortex_charge: int,
    lab_coordinate_m: Sequence[float],
    relative_coordinate_m: Sequence[float],
    axis_seed_x_m: float | None = None,
    axis_seed_y_m: float | None = None,
    axis_search_radius_m: float = 1.2e-3,
) -> TransverseProfileEvidence:
    """Build lab-frame and morphology-frame line profiles at one physical plane.

    The laboratory profiles are always sampled through laboratory zero.  The
    morphology profiles are sampled through the detected Bessel/vortex axis and
    use coordinates relative to that same fixed point.

    If no topology seed is supplied, the 2-D energy centroid is used only as a
    *search seed*.  It is never reported as the vortex axis and does not define
    the morphology profile centre.
    """

    u = np.asarray(field, dtype=np.complex128)
    lab = np.asarray(lab_coordinate_m, dtype=float)
    rel = np.asarray(relative_coordinate_m, dtype=float)
    if lab.size < 64 or rel.size < 64:
        raise ValueError("profile evidence requires at least 64 samples per line")

    cx, cy = _energy_centroid(u, grid)
    sx = cx if axis_seed_x_m is None else float(axis_seed_x_m)
    sy = cy if axis_seed_y_m is None else float(axis_seed_y_m)
    axis = transverse_morphology_axis(
        u,
        grid,
        vortex_charge=int(vortex_charge),
        seed_x_m=sx,
        seed_y_m=sy,
        search_radius_m=float(axis_search_radius_m),
    )

    lab_x, lab_y = spectral_line_fields(
        u,
        grid,
        x_coordinates_m=lab,
        y_coordinates_m=lab,
        fixed_x_m=0.0,
        fixed_y_m=0.0,
    )
    axis_x, axis_y = spectral_line_fields(
        u,
        grid,
        x_coordinates_m=axis.x_m + rel,
        y_coordinates_m=axis.y_m + rel,
        fixed_x_m=axis.x_m,
        fixed_y_m=axis.y_m,
    )

    intensity = np.abs(u) ** 2
    dx = float(grid["dx"])
    return TransverseProfileEvidence(
        lab_coordinate_m=lab,
        relative_coordinate_m=rel,
        lab_x_intensity=np.abs(lab_x) ** 2,
        lab_y_intensity=np.abs(lab_y) ** 2,
        axis_x_intensity=np.abs(axis_x) ** 2,
        axis_y_intensity=np.abs(axis_y) ** 2,
        morphology_axis=axis,
        total_2d_power_au_m2=float(np.sum(intensity) * dx * dx),
        peak_2d_au=float(np.max(intensity)),
        metadata={
            "outcome": "TRANSVERSE-PROFILE-EVIDENCE",
            "lab_profile_planes": "I(x,y=0) and I(x=0,y)",
            "morphology_profile_planes": (
                "I(x,y=y_axis) and I(x=x_axis,y) expressed relative to detected axis"
            ),
            "axis_method": axis.method,
            "axis_search_seed_x_m": sx,
            "axis_search_seed_y_m": sy,
            "energy_centroid_x_m": cx,
            "energy_centroid_y_m": cy,
            "profile_sampling": "direct_discrete_Fourier_series_complex_field",
            "intensity_image_interpolation": False,
        },
    )


def profile_metrics(
    evidence: TransverseProfileEvidence,
    *,
    nominal_peak_2d_au: float,
) -> dict[str, float | int | str]:
    """Return common-normalised profile observables for CSV/report tables."""

    normaliser = max(float(nominal_peak_2d_au), EPS)
    lab_x = np.asarray(evidence.lab_x_intensity, dtype=float)
    lab_y = np.asarray(evidence.lab_y_intensity, dtype=float)
    axis_x = np.asarray(evidence.axis_x_intensity, dtype=float)
    axis_y = np.asarray(evidence.axis_y_intensity, dtype=float)
    lab_coord = np.asarray(evidence.lab_coordinate_m, dtype=float)
    rel_coord = np.asarray(evidence.relative_coordinate_m, dtype=float)

    def _integral(values: np.ndarray, coordinate: np.ndarray) -> float:
        return float(np.trapezoid(values, coordinate))

    def _asymmetry(values: np.ndarray, coordinate: np.ndarray) -> float:
        left = _integral(values[coordinate < 0.0], coordinate[coordinate < 0.0])
        right = _integral(values[coordinate > 0.0], coordinate[coordinate > 0.0])
        return float((right - left) / max(right + left, EPS))

    axis = evidence.morphology_axis
    return {
        "profile_axis_x_m": float(axis.x_m),
        "profile_axis_y_m": float(axis.y_m),
        "profile_axis_method": str(axis.method),
        "profile_detected_topological_charge": int(axis.detected_topological_charge),
        "profile_selected_singularity_count": int(axis.selected_singularity_count),
        "peak_2d_over_nominal": float(evidence.peak_2d_au / normaliser),
        "lab_x_line_peak_over_nominal_2d_peak": float(np.max(lab_x) / normaliser),
        "lab_y_line_peak_over_nominal_2d_peak": float(np.max(lab_y) / normaliser),
        "axis_x_line_peak_over_nominal_2d_peak": float(np.max(axis_x) / normaliser),
        "axis_y_line_peak_over_nominal_2d_peak": float(np.max(axis_y) / normaliser),
        "lab_x_line_integral_over_nominal_2d_peak_m": float(_integral(lab_x, lab_coord) / normaliser),
        "lab_y_line_integral_over_nominal_2d_peak_m": float(_integral(lab_y, lab_coord) / normaliser),
        "axis_x_line_integral_over_nominal_2d_peak_m": float(_integral(axis_x, rel_coord) / normaliser),
        "axis_y_line_integral_over_nominal_2d_peak_m": float(_integral(axis_y, rel_coord) / normaliser),
        "axis_x_left_right_asymmetry": _asymmetry(axis_x, rel_coord),
        "axis_y_left_right_asymmetry": _asymmetry(axis_y, rel_coord),
        "total_2d_power_au_m2": float(evidence.total_2d_power_au_m2),
    }


def profile_long_rows(
    evidence: TransverseProfileEvidence,
    *,
    case_id: str,
    family: str,
    sweep_value: float,
    nominal_peak_2d_au: float,
) -> list[dict[str, float | str]]:
    """Return long-form rows suitable for report/textbook plotting elsewhere."""

    normaliser = max(float(nominal_peak_2d_au), EPS)
    rows: list[dict[str, float | str]] = []
    for frame, coordinate, x_values, y_values in (
        (
            "laboratory",
            evidence.lab_coordinate_m,
            evidence.lab_x_intensity,
            evidence.lab_y_intensity,
        ),
        (
            "morphology_axis",
            evidence.relative_coordinate_m,
            evidence.axis_x_intensity,
            evidence.axis_y_intensity,
        ),
    ):
        for axis_name, values in (("x", x_values), ("y", y_values)):
            for coordinate_m, intensity in zip(coordinate, values):
                rows.append(
                    {
                        "case_id": str(case_id),
                        "family": str(family),
                        "sweep_value": float(sweep_value),
                        "coordinate_frame": frame,
                        "profile_axis": axis_name,
                        "coordinate_m": float(coordinate_m),
                        "intensity_au": float(intensity),
                        "intensity_over_nominal_2d_peak": float(intensity / normaliser),
                    }
                )
    return rows


__all__ = [
    "TransverseProfileEvidence",
    "build_transverse_profile_evidence",
    "profile_long_rows",
    "profile_metrics",
    "spectral_line_fields",
]
