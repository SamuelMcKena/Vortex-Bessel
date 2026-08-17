"""Wave-optics reference built from the explicit refractive-axicon eikonal.

The two-surface ray solver provides an irregular mapping from the physical flat
entrance plane to a plane perpendicular to the mean outgoing ray bundle.  This
module turns that finite-ray-traced eikonal into a regular complex field for
subsequent Fourier/angular-spectrum diffraction.

This is a high-frequency eikonal + scalar diffraction reference.  Surface
refraction and optical path are explicit; the final resampling remains a scalar
wave construction.  It is therefore a materially stronger rigid-tilt model
than an axisymmetric thin phase mask, but absolute lab claims remain blocked
until the real axicon geometry and polarization are supplied and convergence is
demonstrated.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import ConvexHull

from vbb_study.digital_twin.vortex_refractive_axicon import RefractiveAxiconBundle
from vbb_study.equations.fields import make_xy_grid


EPS = np.finfo(float).tiny
TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class RefractiveAxiconReferenceField:
    field: np.ndarray
    grid: Mapping[str, Any]
    coverage_mask: np.ndarray
    metadata: Mapping[str, Any]


def _linear_complex_interpolate(
    points_xy: np.ndarray,
    values: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> np.ndarray:
    """Piecewise-linear interpolation of complex samples on an irregular plane."""

    pts = np.asarray(points_xy, dtype=float)
    val = np.asarray(values, dtype=np.complex128)
    real_interp = LinearNDInterpolator(pts, val.real, fill_value=np.nan)
    imag_interp = LinearNDInterpolator(pts, val.imag, fill_value=np.nan)
    real = real_interp(target_x, target_y)
    imag = imag_interp(target_x, target_y)
    return np.asarray(real + 1j * imag, dtype=np.complex128)


def build_refractive_axicon_reference_field(
    entrance_envelope: np.ndarray,
    entrance_grid: Mapping[str, Any],
    *,
    bundle: RefractiveAxiconBundle,
    wavelength_m: float,
    incident_spectral_center_cpm: tuple[float, float] = (0.0, 0.0),
    output_n: int | None = None,
    output_window_m: float | None = None,
    use_fresnel_power: bool = False,
    minimum_mapping_jacobian: float = 1.0e-6,
) -> RefractiveAxiconReferenceField:
    """Resample the finite-ray-traced eikonal onto the mean-beam reference plane.

    Ray-tube amplitude is obtained from normal-flux conservation,

        |E_ref|^2 (s_out . n_ref) |J| =
        |E_in|^2 (s_in . n_ent) T_Fresnel,

    where ``J`` maps entrance-plane area to reference-plane area.  The optical
    phase is the input carrier/envelope phase plus ``k0 * OPL`` from the physical
    two-surface trace.

    ``use_fresnel_power=True`` requires a polarization-resolved bundle.  With it
    false, interface amplitude loss is intentionally omitted and metadata keeps
    the result blocked for absolute throughput claims.

    ``coverage_fraction`` is intentionally not treated as a fixed target: a
    circular/elliptical traced aperture cannot fill the corners of a square FFT
    window.  Instead, the rasterised coverage is checked against the actual
    convex-hull area of the traced reference-plane rays when that hull lies
    inside the requested output window.
    """

    envelope = np.asarray(entrance_envelope, dtype=np.complex128)
    X = np.asarray(entrance_grid["X"], dtype=float)
    Y = np.asarray(entrance_grid["Y"], dtype=float)
    if envelope.shape != X.shape or X.shape != Y.shape:
        raise ValueError("entrance field/grid shapes do not match")
    if envelope.shape != bundle.valid.shape:
        raise ValueError("entrance field does not match refractive bundle grid")

    dx_in = float(entrance_grid["dx"])
    n_out = int(output_n if output_n is not None else entrance_grid["N"])
    if n_out < 64:
        raise ValueError("output_n must be at least 64")
    if n_out > 512:
        raise ValueError(
            "reference-field Delaunay resampling is intentionally capped at N=512; "
            "use convergence references rather than pretending this is the N=1536 production solver"
        )
    window = float(
        output_window_m
        if output_window_m is not None
        else dx_in * n_out
    )
    output_grid = make_xy_grid(n_out, window / n_out)
    XO = np.asarray(output_grid["X"], dtype=float)
    YO = np.asarray(output_grid["Y"], dtype=float)

    if use_fresnel_power:
        if bundle.fresnel_power_transmission is None:
            raise ValueError(
                "Fresnel power requested but bundle has no declared input polarization"
            )
        fresnel = np.asarray(bundle.fresnel_power_transmission, dtype=float)
        fresnel_status = "polarization_resolved_vector_Fresnel_power"
    else:
        fresnel = np.ones_like(bundle.reference_xi_m, dtype=float)
        fresnel_status = "omitted_for_morphology_reference"

    jac = np.asarray(bundle.mapping_jacobian_abs, dtype=float)
    cos_out = np.asarray(bundle.output_reference_cosine, dtype=float)
    valid = (
        np.asarray(bundle.valid, dtype=bool)
        & np.isfinite(jac)
        & (jac > float(minimum_mapping_jacobian))
        & np.isfinite(cos_out)
        & (cos_out > 0.0)
        & np.isfinite(fresnel)
        & (fresnel >= 0.0)
    )
    if int(np.count_nonzero(valid)) < 0.25 * valid.size:
        raise ValueError("too few valid rays for reference-field construction")

    fsx, fsy = map(float, incident_spectral_center_cpm)
    carrier_phase = TWOPI * (fsx * X + fsy * Y)
    opl = np.asarray(bundle.optical_path_from_entrance_m, dtype=float)
    opl_reference = float(np.median(opl[valid]))
    propagation_phase = (TWOPI / float(wavelength_m)) * (opl - opl_reference)

    geometric_amplitude = np.sqrt(
        float(bundle.input_normal_cosine)
        * fresnel
        / np.maximum(cos_out * jac, EPS)
    )
    sample_field = (
        envelope
        * np.exp(1j * carrier_phase)
        * geometric_amplitude
        * np.exp(1j * propagation_phase)
    )

    points = np.column_stack(
        [
            np.asarray(bundle.reference_xi_m)[valid],
            np.asarray(bundle.reference_eta_m)[valid],
        ]
    )
    values = np.asarray(sample_field)[valid]
    regular = _linear_complex_interpolate(points, values, XO, YO)
    coverage = np.isfinite(regular.real) & np.isfinite(regular.imag)
    regular = np.where(coverage, regular, 0.0j)

    # Geometric coverage contract.  In 2-D scipy ConvexHull.volume is the hull
    # area.  Comparing that continuous area with the rasterised finite mask tests
    # the irregular->regular interpolation without pretending a circular
    # physical aperture should fill a square FFT window.
    hull = ConvexHull(points)
    hull_area = float(hull.volume)
    half_window = 0.5 * float(window)
    support_within_window = bool(
        np.min(points[:, 0]) >= -half_window
        and np.max(points[:, 0]) <= half_window
        and np.min(points[:, 1]) >= -half_window
        and np.max(points[:, 1]) <= half_window
    )
    observed_coverage = float(np.mean(coverage))
    expected_hull_coverage = float(hull_area / (window * window))
    coverage_relative_error = (
        abs(observed_coverage - expected_hull_coverage)
        / max(expected_hull_coverage, EPS)
        if support_within_window
        else float("nan")
    )

    # This closure is evaluated in ray coordinates, before interpolation.  It is
    # the exact algebraic normal-flux contract used to construct the geometric
    # amplitude and therefore catches a broken Jacobian/cosine/Fresnel factor.
    input_density = np.abs(envelope) ** 2
    input_flux = float(
        np.sum(input_density[valid])
        * float(bundle.input_normal_cosine)
        * dx_in
        * dx_in
    )
    transmitted_flux = float(
        np.sum(input_density[valid] * fresnel[valid])
        * float(bundle.input_normal_cosine)
        * dx_in
        * dx_in
    )
    reconstructed_ray_flux = float(
        np.sum(
            np.abs(sample_field[valid]) ** 2
            * cos_out[valid]
            * jac[valid]
        )
        * dx_in
        * dx_in
    )
    closure = reconstructed_ray_flux / max(transmitted_flux, EPS)

    return RefractiveAxiconReferenceField(
        field=np.asarray(regular, dtype=np.complex128),
        grid=output_grid,
        coverage_mask=np.asarray(coverage, dtype=bool),
        metadata={
            "outcome": "REFRACTIVE-AXICON-EIKONAL-WAVE-REFERENCE",
            "reference_plane": "perpendicular_to_mean_outgoing_ray_bundle",
            "interpolation": "scipy_LinearNDInterpolator_complex_components",
            "output_n": int(n_out),
            "output_window_m": float(window),
            "coverage_fraction": observed_coverage,
            "convex_hull_area_m2": hull_area,
            "expected_hull_coverage_fraction": expected_hull_coverage,
            "coverage_relative_error_to_hull": float(coverage_relative_error),
            "reference_support_within_output_window": support_within_window,
            "valid_ray_fraction": float(np.mean(valid)),
            "fresnel_status": fresnel_status,
            "input_flux_au": input_flux,
            "expected_transmitted_flux_au": transmitted_flux,
            "reconstructed_ray_flux_au": reconstructed_ray_flux,
            "ray_flux_closure_ratio": float(closure),
            "opl_reference_m": opl_reference,
            "mean_reference_normal_lab": np.asarray(
                bundle.reference_normal_lab, dtype=float
            ).tolist(),
            "report_figures_authorised": False,
        },
    )


def angular_spectrum_second_moments(
    field: np.ndarray,
    grid: Mapping[str, Any],
) -> dict[str, float]:
    """Centroid/covariance of angular-spectrum intensity in cycles/m."""

    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(np.asarray(field))))
    intensity = np.abs(spectrum) ** 2
    FX = np.asarray(grid["FX"], dtype=float)
    FY = np.asarray(grid["FY"], dtype=float)
    total = float(np.sum(intensity))
    if total <= EPS:
        raise ValueError("zero angular-spectrum power")
    cx = float(np.sum(intensity * FX) / total)
    cy = float(np.sum(intensity * FY) / total)
    dfx = FX - cx
    dfy = FY - cy
    cxx = float(np.sum(intensity * dfx * dfx) / total)
    cyy = float(np.sum(intensity * dfy * dfy) / total)
    cxy = float(np.sum(intensity * dfx * dfy) / total)
    covariance = np.asarray([[cxx, cxy], [cxy, cyy]], dtype=float)
    eig = np.linalg.eigvalsh(covariance)
    major = float(math.sqrt(max(float(eig[-1]), 0.0)))
    minor = float(math.sqrt(max(float(eig[0]), 0.0)))
    anisotropy = (major - minor) / max(0.5 * (major + minor), EPS)
    return {
        "spectral_centroid_fx_cpm": cx,
        "spectral_centroid_fy_cpm": cy,
        "spectral_rms_major_cpm": major,
        "spectral_rms_minor_cpm": minor,
        "spectral_second_moment_anisotropy_fraction": float(anisotropy),
    }


__all__ = [
    "RefractiveAxiconReferenceField",
    "angular_spectrum_second_moments",
    "build_refractive_axicon_reference_field",
]
