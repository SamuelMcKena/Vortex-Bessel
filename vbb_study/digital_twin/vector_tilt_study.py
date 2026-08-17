"""Reusable observables for the Phase-2H vector axicon-tilt study.

This module deliberately keeps study observables separate from the refractive
surface solver.  The solver is responsible for Snell/Fresnel/eikonal physics;
this module turns the resulting fixed-lab vector field into reproducible study
quantities: higher-order cylindrical-vector inputs, exact line profiles,
second-moment beam geometry and analyzer frames.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from vbb_study.calibration.vector_observables import (
    LinearAnalyzerCalibration,
    PetalObservable,
    analyzer_intensity,
    petal_observable,
)
from vbb_study.digital_twin.vortex_profile_evidence import spectral_line_fields
from vbb_study.equations.fields import make_xy_grid
from vbb_study.vector_field import VectorField


EPS = np.finfo(float).tiny


@dataclass(frozen=True)
class BeamMomentMetrics:
    centroid_x_m: float
    centroid_y_m: float
    sigma_major_m: float
    sigma_minor_m: float
    ellipticity: float
    major_axis_angle_rad: float
    peak_intensity: float
    power_au_m2: float


def higher_order_cylindrical_vector_input(
    *,
    ell: int,
    mode: str,
    n: int = 128,
    window_m: float = 3.0e-3,
    waist_m: float = 0.90e-3,
    wavelength_m: float = 1029e-9,
    medium_index: float = 1.0,
) -> VectorField:
    """Return a Gaussian-envelope generalized cylindrical-vector input.

    The local linear-polarization angle is ``ell*phi + delta`` with delta=0
    for radial-like and delta=pi/2 for azimuthal-like.  For ell=1 these reduce
    to the ordinary radial/azimuthal states.  A linear analyzer therefore has
    the ideal zero-tilt harmonic ``2*abs(ell)`` before any physical distortion.

    No axicon/Bessel phase is included here.  The physical Phase-2H refractive
    axicon is responsible for creating the downstream Bessel propagation.
    """

    order = abs(int(ell))
    if order < 1:
        raise ValueError("ell must be non-zero for the cylindrical-vector study")
    key = str(mode).lower().strip()
    if key not in {"radial", "azimuthal"}:
        raise ValueError("mode must be 'radial' or 'azimuthal'")
    if int(n) < 32 or float(window_m) <= 0.0 or float(waist_m) <= 0.0:
        raise ValueError("invalid cylindrical-vector grid or waist")

    grid = make_xy_grid(int(n), float(window_m) / int(n))
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    R = np.hypot(X, Y)
    phi = np.arctan2(Y, X)
    amplitude = np.exp(-(R / float(waist_m)) ** 2)
    delta = 0.0 if key == "radial" else 0.5 * math.pi
    theta = order * phi + delta
    ex = amplitude * np.cos(theta)
    ey = amplitude * np.sin(theta)
    # Do not assign a preferred vector direction at the coordinate singularity.
    centre = R <= 0.5 * float(grid["dx"])
    ex = np.where(centre, 0.0, ex).astype(np.complex128)
    ey = np.where(centre, 0.0, ey).astype(np.complex128)
    return VectorField(
        ex=ex,
        ey=ey,
        ez=np.zeros_like(ex),
        grid=grid,
        wavelength_m=float(wavelength_m),
        medium_index=float(medium_index),
        metadata={
            "vector_input_family": "generalized_cylindrical_vector_gaussian",
            "ell": int(ell),
            "mode": key,
            "expected_linear_analyzer_harmonic": 2 * order,
            "contains_axicon_phase": False,
            "simulation_only_requires_polarization_converter": True,
        },
    )


def beam_moment_metrics(field: VectorField) -> BeamMomentMetrics:
    """Return fixed-lab intensity centroid and covariance ellipse metrics."""

    I = np.maximum(np.asarray(field.intensity, dtype=float), 0.0)
    x = np.asarray(field.grid["x"], dtype=float)
    y = np.asarray(field.grid.get("y", x), dtype=float)
    X, Y = np.meshgrid(x, y, indexing="xy")
    total = float(np.sum(I))
    if total <= EPS:
        raise ValueError("cannot compute beam moments for zero-power field")
    cx = float(np.sum(I * X) / total)
    cy = float(np.sum(I * Y) / total)
    dx = X - cx
    dy = Y - cy
    cxx = float(np.sum(I * dx * dx) / total)
    cyy = float(np.sum(I * dy * dy) / total)
    cxy = float(np.sum(I * dx * dy) / total)
    cov = np.array([[cxx, cxy], [cxy, cyy]], dtype=float)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    order = np.argsort(vals)[::-1]
    major = float(np.sqrt(vals[order[0]]))
    minor = float(np.sqrt(vals[order[1]]))
    major_vec = vecs[:, order[0]]
    angle = float(math.atan2(float(major_vec[1]), float(major_vec[0])))
    grid_dx = float(field.grid["dx"])
    grid_dy = float(field.grid.get("dy", grid_dx))
    return BeamMomentMetrics(
        centroid_x_m=cx,
        centroid_y_m=cy,
        sigma_major_m=major,
        sigma_minor_m=minor,
        ellipticity=float(major / max(minor, EPS)),
        major_axis_angle_rad=angle,
        peak_intensity=float(np.max(I)),
        power_au_m2=float(np.sum(I) * grid_dx * grid_dy),
    )


def vector_line_intensity(
    field: VectorField,
    coordinate_m: Sequence[float],
    *,
    fixed_x_m: float,
    fixed_y_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate exact x/y line intensities from Ex/Ey/Ez by Fourier synthesis."""

    coord = np.asarray(coordinate_m, dtype=float)
    x_total = np.zeros(coord.shape, dtype=float)
    y_total = np.zeros(coord.shape, dtype=float)
    for component in (field.ex, field.ey, field.ez):
        x_line, y_line = spectral_line_fields(
            component,
            field.grid,
            x_coordinates_m=coord,
            y_coordinates_m=coord,
            fixed_x_m=float(fixed_x_m),
            fixed_y_m=float(fixed_y_m),
        )
        x_total += np.abs(x_line) ** 2
        y_total += np.abs(y_line) ** 2
    return x_total, y_total


def ideal_linear_analyzer_frames(
    field: VectorField,
    *,
    angles_deg: Sequence[int] = (0, 45, 90, 135),
) -> dict[int, np.ndarray]:
    """Return ideal linear-analyzer intensity frames for a spatial vector field."""

    return {
        int(angle): analyzer_intensity(
            field.ex,
            field.ey,
            LinearAnalyzerCalibration(float(angle), float(angle)),
        )
        for angle in angles_deg
    }


def centered_coordinate_maps(field: VectorField) -> tuple[np.ndarray, np.ndarray, BeamMomentMetrics]:
    """Return physical coordinate maps centred on the total-intensity centroid."""

    metrics = beam_moment_metrics(field)
    x = np.asarray(field.grid["x"], dtype=float)
    y = np.asarray(field.grid.get("y", x), dtype=float)
    X, Y = np.meshgrid(x, y, indexing="xy")
    return X - metrics.centroid_x_m, Y - metrics.centroid_y_m, metrics


def well_sampled_petal_observable(
    intensity: np.ndarray,
    X_m: np.ndarray,
    Y_m: np.ndarray,
    *,
    pixel_pitch_m: float,
    minimum_radius_pixels: float = 12.0,
    maximum_radius_fraction: float = 0.45,
    radial_bins: int = 320,
) -> PetalObservable:
    """Measure the dominant angular harmonic on a well-sampled Bessel annulus.

    The generic petal observable normally uses the strongest radial ring.  A
    refractive axicon can make that first ring only a few pixels in radius even
    when the overall field satisfies the solver Nyquist gate.  Angular Fourier
    analysis on such a tiny annulus is under-sampled.  For the Phase-2H study we
    therefore select the strongest annulus outside ``minimum_radius_pixels``
    and then call the same calibrated-pixel petal estimator at that radius.

    This is an observable/sampling policy only; it never changes the optical
    field or the refractive solver.
    """

    I = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    X = np.asarray(X_m, dtype=float)
    Y = np.asarray(Y_m, dtype=float)
    if I.shape != X.shape or I.shape != Y.shape:
        raise ValueError("intensity and physical coordinate maps must match")
    q = float(pixel_pitch_m)
    if not np.isfinite(q) or q <= 0.0:
        raise ValueError("pixel_pitch_m must be positive")
    R = np.hypot(X, Y)
    rmax = float(np.max(R))
    lower = float(minimum_radius_pixels) * q
    upper = float(maximum_radius_fraction) * rmax
    if upper <= lower:
        raise ValueError("field support is too small for a well-sampled petal annulus")

    edges = np.linspace(0.0, rmax, int(radial_bins) + 1)
    index = np.clip(np.digitize(R.ravel(), edges) - 1, 0, int(radial_bins) - 1)
    sums = np.bincount(index, weights=I.ravel(), minlength=int(radial_bins))
    counts = np.bincount(index, minlength=int(radial_bins))
    mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    centres = 0.5 * (edges[:-1] + edges[1:])
    valid = (centres >= lower) & (centres <= upper) & (counts >= 8)
    if not np.any(valid):
        raise ValueError("no sufficiently sampled radial annulus is available")
    candidate_indices = np.flatnonzero(valid)
    best = int(candidate_indices[int(np.argmax(mean[valid]))])
    radius = float(centres[best])

    # Increase annulus width slightly if required by the calibrated-pixel count,
    # but keep the radial band local enough that angular harmonics are not mixed
    # across many Bessel rings.
    last_error: Exception | None = None
    for half_width in (0.18, 0.24, 0.30):
        try:
            return petal_observable(
                I,
                X,
                Y,
                ring_radius_m=radius,
                ring_half_width_fraction=half_width,
            )
        except ValueError as exc:
            last_error = exc
    raise ValueError(f"selected petal annulus remains under-sampled: {last_error}")


__all__ = [
    "BeamMomentMetrics",
    "beam_moment_metrics",
    "centered_coordinate_maps",
    "higher_order_cylindrical_vector_input",
    "ideal_linear_analyzer_frames",
    "vector_line_intensity",
    "well_sampled_petal_observable",
]
