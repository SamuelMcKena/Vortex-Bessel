from __future__ import annotations

import math

import numpy as np

from vbb_study.digital_twin.vortex_refractive_axicon import (
    RefractiveAxiconGeometry,
    trace_refractive_axicon_bundle,
)
from vbb_study.digital_twin.vortex_refractive_axicon_wave import (
    angular_spectrum_second_moments,
    build_refractive_axicon_reference_field,
)
from vbb_study.digital_twin.vortex_rotated_plane import rotation_matrix
from vbb_study.equations.fields import make_xy_grid


WAVELENGTH = 1.029e-6
N_AX = 1.458
N_EXT = 1.0
GAMMA = math.radians(2.0)


def _setup(tilt_y_deg: float):
    n = 161
    window = 3.2e-3
    grid = make_xy_grid(n, window / n)
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    geometry = RefractiveAxiconGeometry(
        base_angle_rad=GAMMA,
        clear_radius_m=1.55e-3,
        centre_thickness_m=2.0e-3,
        refractive_index=N_AX,
        external_index=N_EXT,
    )
    tilt = math.radians(float(tilt_y_deg))
    bundle = trace_refractive_axicon_bundle(
        X,
        Y,
        geometry=geometry,
        tilt_y_rad=tilt,
        apex_exclusion_radius_m=40e-6,
    )
    rotation = rotation_matrix(0.0, tilt)
    incident_local = rotation.T @ np.asarray([0.0, 0.0, 1.0])
    carrier = (
        N_EXT / WAVELENGTH * float(incident_local[0]),
        N_EXT / WAVELENGTH * float(incident_local[1]),
    )
    envelope = np.exp(-(X * X + Y * Y) / (0.9e-3**2)).astype(np.complex128)
    reference = build_refractive_axicon_reference_field(
        envelope,
        grid,
        bundle=bundle,
        wavelength_m=WAVELENGTH,
        incident_spectral_center_cpm=carrier,
        output_n=192,
        output_window_m=3.2e-3,
        use_fresnel_power=False,
    )
    return bundle, reference


def _ray_second_moment_anisotropy(bundle) -> float:
    valid = np.asarray(bundle.valid, dtype=bool)
    outgoing = np.asarray(bundle.outgoing_lab, dtype=float)
    f1 = N_EXT / WAVELENGTH * (outgoing @ bundle.reference_e1_lab)
    f2 = N_EXT / WAVELENGTH * (outgoing @ bundle.reference_e2_lab)
    x = np.asarray(bundle.entrance_x_m)
    y = np.asarray(bundle.entrance_y_m)
    weight = np.exp(-2.0 * (x * x + y * y) / (0.9e-3**2))
    weight = np.where(valid, weight, 0.0)
    total = float(np.sum(weight))
    c1 = float(np.sum(weight * f1) / total)
    c2 = float(np.sum(weight * f2) / total)
    q1 = f1 - c1
    q2 = f2 - c2
    covariance = np.asarray(
        [
            [np.sum(weight * q1 * q1) / total, np.sum(weight * q1 * q2) / total],
            [np.sum(weight * q1 * q2) / total, np.sum(weight * q2 * q2) / total],
        ],
        dtype=float,
    )
    eig = np.linalg.eigvalsh(covariance)
    major = math.sqrt(max(float(eig[-1]), 0.0))
    minor = math.sqrt(max(float(eig[0]), 0.0))
    return (major - minor) / (0.5 * (major + minor))


def test_refractive_reference_field_closes_flux_and_matches_ray_hull() -> None:
    _, reference = _setup(5.0)
    meta = reference.metadata
    assert abs(float(meta["ray_flux_closure_ratio"]) - 1.0) < 1e-10
    assert bool(meta["reference_support_within_output_window"])
    # The physical support is approximately circular/elliptical, so its area
    # should not be forced to fill an arbitrary fraction of a square FFT grid.
    # Instead the pixel mask must reproduce the continuous convex hull of the
    # traced rays to within finite-grid rasterisation error.
    assert float(meta["expected_hull_coverage_fraction"]) > 0.5
    assert float(meta["coverage_relative_error_to_hull"]) < 0.03
    assert np.all(np.isfinite(reference.field))


def test_zero_tilt_refractive_reference_is_spectrally_near_axisymmetric() -> None:
    _, reference = _setup(0.0)
    moments = angular_spectrum_second_moments(reference.field, reference.grid)
    assert moments["spectral_second_moment_anisotropy_fraction"] < 0.03


def test_oblique_refractive_wave_follows_ray_anisotropy_trend() -> None:
    b0, r0 = _setup(0.0)
    b5, r5 = _setup(5.0)
    b10, r10 = _setup(10.0)
    wave = np.asarray(
        [
            angular_spectrum_second_moments(r.field, r.grid)[
                "spectral_second_moment_anisotropy_fraction"
            ]
            for r in (r0, r5, r10)
        ]
    )
    ray = np.asarray([_ray_second_moment_anisotropy(b) for b in (b0, b5, b10)])
    assert wave[2] > wave[1] > wave[0]
    assert ray[2] > ray[1] > ray[0]
    # Diffraction/aperture broadening adds an approximately isotropic component,
    # so exact equality is not expected.  The finite-surface wave anisotropy must
    # nevertheless be a substantial fraction of the geometrical prediction.
    assert wave[2] > 0.35 * ray[2]
