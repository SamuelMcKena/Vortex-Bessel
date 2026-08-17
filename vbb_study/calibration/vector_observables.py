"""Experimental observables for the cylindrical/segmented vector-beam studies.

The vector branch is not validated by total intensity alone.  The historical
study uses linear-analyzer images at 0/45/90/135 degrees, polarization/Stokes
maps and the 2|ell|-petal analyzer patterns (two petals for |ell|=1, six for
|ell|=3).  This module promotes those plots to quantitative calibration gates.

Only S0/S1/S2 and the headless linear-polarization angle are called directly
observable with four linear-analyzer frames.  S3/ellipticity require a QWP or
another full-Stokes analyser and remain blocked when that measurement is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from vbb_study.calibration.camera_comparison import (
    CameraCalibration,
    CameraComparison,
    camera_axes,
    compare_simulation_to_camera,
    preprocess_camera_image,
)


EPS = np.finfo(float).tiny


@dataclass(frozen=True)
class LinearAnalyzerCalibration:
    """Measured linear-analyzer state for one nominal angle."""

    nominal_angle_deg: float
    actual_angle_deg: float
    transmission: float = 1.0
    extinction_ratio: float = math.inf

    def validate(self) -> None:
        if not np.isfinite(self.actual_angle_deg):
            raise ValueError("actual analyzer angle must be finite")
        if not np.isfinite(self.transmission) or not 0.0 < self.transmission <= 1.0:
            raise ValueError("analyzer transmission must lie in (0,1]")
        if self.extinction_ratio != math.inf and (
            not np.isfinite(self.extinction_ratio) or self.extinction_ratio < 1.0
        ):
            raise ValueError("extinction_ratio must be >=1 or infinity")


@dataclass(frozen=True)
class PetalObservable:
    harmonic: int
    petal_count: int
    orientation_rad: float
    modulation_fraction: float
    ring_radius_m: float
    ring_sample_count: int


@dataclass(frozen=True)
class VectorAnalyzerComparison:
    frame_comparisons: Mapping[int, CameraComparison]
    measured_stokes: Mapping[str, np.ndarray]
    simulated_stokes: Mapping[str, np.ndarray]
    measured_polarization_angle_rad: np.ndarray
    simulated_polarization_angle_rad: np.ndarray
    polarization_angle_rms_rad: float
    measured_petals: Mapping[int, PetalObservable]
    simulated_petals: Mapping[int, PetalObservable]
    metadata: Mapping[str, object]


def analyzer_intensity(
    ex: np.ndarray,
    ey: np.ndarray,
    calibration: LinearAnalyzerCalibration,
) -> np.ndarray:
    """Return intensity after a calibrated imperfect linear analyzer.

    The leakage term is treated as incoherent transmission along the orthogonal
    analyzer axis.  ``extinction_ratio=inf`` recovers an ideal polarizer.
    """

    calibration.validate()
    Ex = np.asarray(ex, dtype=np.complex128)
    Ey = np.asarray(ey, dtype=np.complex128)
    if Ex.shape != Ey.shape:
        raise ValueError("ex and ey must share one shape")
    theta = math.radians(float(calibration.actual_angle_deg))
    c, s = math.cos(theta), math.sin(theta)
    parallel = c * Ex + s * Ey
    orthogonal = -s * Ex + c * Ey
    leakage = 0.0 if calibration.extinction_ratio == math.inf else 1.0 / float(calibration.extinction_ratio)
    return float(calibration.transmission) * (
        np.abs(parallel) ** 2 + leakage * np.abs(orthogonal) ** 2
    )


def linear_stokes_from_frames(frames: Mapping[int, np.ndarray]) -> dict[str, np.ndarray]:
    """Reconstruct the linear-analyzer observable Stokes subset."""

    required = {0, 45, 90, 135}
    if not required.issubset(frames):
        raise ValueError("0/45/90/135 degree analyzer frames are required")
    I0 = np.asarray(frames[0], dtype=float)
    I45 = np.asarray(frames[45], dtype=float)
    I90 = np.asarray(frames[90], dtype=float)
    I135 = np.asarray(frames[135], dtype=float)
    if len({I0.shape, I45.shape, I90.shape, I135.shape}) != 1:
        raise ValueError("all analyzer frames must have the same shape")
    S0 = I0 + I90
    S1 = I0 - I90
    S2 = I45 - I135
    psi = 0.5 * np.arctan2(S2, S1)
    return {"S0": S0, "S1": S1, "S2": S2, "psi": psi}


def _radial_profile_peak_radius(
    intensity: np.ndarray,
    X_m: np.ndarray,
    Y_m: np.ndarray,
    *,
    radial_bins: int = 256,
) -> float:
    I = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    R = np.hypot(np.asarray(X_m, dtype=float), np.asarray(Y_m, dtype=float))
    rmax = float(np.max(R))
    edges = np.linspace(0.0, rmax, int(radial_bins) + 1)
    index = np.clip(np.digitize(R.ravel(), edges) - 1, 0, int(radial_bins) - 1)
    sums = np.bincount(index, weights=I.ravel(), minlength=int(radial_bins))
    counts = np.bincount(index, minlength=int(radial_bins))
    mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    centres = 0.5 * (edges[:-1] + edges[1:])
    # Ignore the central two radial bins so a bright residual core does not
    # replace the vector/Bessel ring as the petal-analysis radius.
    start = min(2, max(mean.size - 1, 0))
    return float(centres[start + int(np.argmax(mean[start:]))])


def petal_observable(
    intensity: np.ndarray,
    X_m: np.ndarray,
    Y_m: np.ndarray,
    *,
    ring_radius_m: float | None = None,
    ring_half_width_fraction: float = 0.18,
    maximum_harmonic: int = 12,
) -> PetalObservable:
    """Estimate analyzer petal count/orientation directly on calibrated pixels.

    This uses a non-uniform angular Fourier coefficient over annular camera
    pixels, so it remains valid after camera rotation and does not invent
    sub-pixel interpolation.  The dominant non-DC harmonic is the petal count.
    """

    I = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    X = np.asarray(X_m, dtype=float)
    Y = np.asarray(Y_m, dtype=float)
    if I.shape != X.shape or I.shape != Y.shape:
        raise ValueError("intensity and physical coordinate maps must match")
    radius = (
        _radial_profile_peak_radius(I, X, Y)
        if ring_radius_m is None
        else float(ring_radius_m)
    )
    if radius <= 0.0:
        raise ValueError("ring_radius_m must be positive")
    R = np.hypot(X, Y)
    phi = np.arctan2(Y, X)
    half = float(ring_half_width_fraction)
    ring = (R >= (1.0 - half) * radius) & (R <= (1.0 + half) * radius)
    if np.count_nonzero(ring) < 32:
        raise ValueError("too few calibrated pixels in the ring ROI")
    values = I[ring]
    angles = phi[ring]
    centred = values - float(np.mean(values))
    norm = max(float(np.sum(np.abs(centred))), EPS)
    amplitudes = []
    coeffs = []
    for harmonic in range(1, int(maximum_harmonic) + 1):
        coeff = np.sum(centred * np.exp(-1j * harmonic * angles)) / norm
        coeffs.append(coeff)
        amplitudes.append(abs(coeff))
    dominant = int(np.argmax(amplitudes) + 1)
    coeff = coeffs[dominant - 1]
    orientation = float((-np.angle(coeff) / dominant) % (2.0 * np.pi / dominant))
    modulation = float(np.std(values) / max(float(np.mean(values)), EPS))
    return PetalObservable(
        harmonic=dominant,
        petal_count=dominant,
        orientation_rad=orientation,
        modulation_fraction=modulation,
        ring_radius_m=radius,
        ring_sample_count=int(np.count_nonzero(ring)),
    )


def _headless_angle_rms(
    measured: np.ndarray,
    simulated: np.ndarray,
    weight: np.ndarray,
    valid: np.ndarray,
) -> float:
    mask = np.asarray(valid, dtype=bool) & np.isfinite(measured) & np.isfinite(simulated)
    if np.count_nonzero(mask) < 20:
        return float("nan")
    delta = 0.5 * np.angle(np.exp(2j * (np.asarray(measured) - np.asarray(simulated))))
    w = np.maximum(np.asarray(weight, dtype=float), 0.0)
    w = np.where(mask, w, 0.0)
    return float(np.sqrt(np.sum(w * delta * delta) / max(float(np.sum(w)), EPS)))


def compare_vector_analyzer_frames(
    measured_frames: Mapping[int, np.ndarray],
    *,
    camera_calibration: CameraCalibration,
    simulated_ex: np.ndarray,
    simulated_ey: np.ndarray,
    simulated_x_m: np.ndarray,
    simulated_y_m: np.ndarray,
    analyzer_calibrations: Mapping[int, LinearAnalyzerCalibration] | None = None,
    backgrounds: Mapping[int, np.ndarray | float] | None = None,
    expected_ell: int | None = None,
) -> VectorAnalyzerComparison:
    """Compare four measured analyzer frames to one simulated Jones field."""

    required = (0, 45, 90, 135)
    if any(angle not in measured_frames for angle in required):
        raise ValueError("measured_frames must contain 0,45,90,135 degree images")
    calibrations = {
        angle: (
            analyzer_calibrations[angle]
            if analyzer_calibrations is not None and angle in analyzer_calibrations
            else LinearAnalyzerCalibration(angle, angle)
        )
        for angle in required
    }
    simulated_frames = {
        angle: analyzer_intensity(simulated_ex, simulated_ey, calibrations[angle])
        for angle in required
    }
    comparisons: dict[int, CameraComparison] = {}
    measured_processed: dict[int, np.ndarray] = {}
    simulated_camera: dict[int, np.ndarray] = {}
    common_valid: np.ndarray | None = None
    for angle in required:
        background = None if backgrounds is None else backgrounds.get(angle)
        comparison = compare_simulation_to_camera(
            measured_frames[angle],
            camera_calibration,
            simulated_frames[angle],
            simulated_x_m,
            simulated_y_m,
            background=background,
        )
        comparisons[angle] = comparison
        measured_processed[angle] = comparison.measured_intensity
        simulated_camera[angle] = comparison.simulated_on_camera
        common_valid = (
            np.asarray(comparison.valid_mask, dtype=bool)
            if common_valid is None
            else common_valid & np.asarray(comparison.valid_mask, dtype=bool)
        )

    measured_stokes = linear_stokes_from_frames(measured_processed)
    simulated_stokes = linear_stokes_from_frames(simulated_camera)
    valid = np.asarray(common_valid, dtype=bool)
    weight = np.minimum(
        np.asarray(measured_stokes["S0"], dtype=float),
        np.asarray(simulated_stokes["S0"], dtype=float),
    )
    psi_rms = _headless_angle_rms(
        measured_stokes["psi"],
        simulated_stokes["psi"],
        weight,
        valid,
    )

    _, _, Xcam, Ycam = camera_axes(np.shape(measured_frames[0]), camera_calibration)
    measured_petals: dict[int, PetalObservable] = {}
    simulated_petals: dict[int, PetalObservable] = {}
    for angle in required:
        measured_petals[angle] = petal_observable(
            np.where(valid, measured_processed[angle], 0.0), Xcam, Ycam
        )
        simulated_petals[angle] = petal_observable(
            np.where(valid, simulated_camera[angle], 0.0), Xcam, Ycam,
            ring_radius_m=measured_petals[angle].ring_radius_m,
        )

    expected_petals = None if expected_ell is None else 2 * abs(int(expected_ell))
    petal_match_fraction = (
        float(np.mean([m.petal_count == expected_petals for m in measured_petals.values()]))
        if expected_petals is not None
        else float("nan")
    )
    return VectorAnalyzerComparison(
        frame_comparisons=comparisons,
        measured_stokes=measured_stokes,
        simulated_stokes=simulated_stokes,
        measured_polarization_angle_rad=np.asarray(measured_stokes["psi"], dtype=float),
        simulated_polarization_angle_rad=np.asarray(simulated_stokes["psi"], dtype=float),
        polarization_angle_rms_rad=psi_rms,
        measured_petals=measured_petals,
        simulated_petals=simulated_petals,
        metadata={
            "observable_stokes": ["S0", "S1", "S2", "psi"],
            "blocked_without_qwp": ["S3", "ellipticity_chi"],
            "expected_ell": None if expected_ell is None else int(expected_ell),
            "expected_analyzer_petal_count": expected_petals,
            "measured_expected_petal_match_fraction": petal_match_fraction,
            "analyzer_angles_nominal_deg": list(required),
            "comparison_policy": "camera-calibrated analyzer frames; no visual-only spot counting",
        },
    )


__all__ = [
    "LinearAnalyzerCalibration",
    "PetalObservable",
    "VectorAnalyzerComparison",
    "analyzer_intensity",
    "compare_vector_analyzer_frames",
    "linear_stokes_from_frames",
    "petal_observable",
]
