"""Literature-backed reference models for vortex-Bessel error studies.

These functions are deliberately independent of the report renderer.  They
provide analytic/reference quantities that the numerical optical train must
reproduce before a physical-error figure can be described as report-ready.

The principal references behind the contracts are:

* Zhao & Li, Applied Optics 37, 2563-2568 (1998): thin axicon under oblique
  illumination is an incident tilted wave multiplied by the axicon
  transmittance followed by diffraction.
* Thaning, Jaroszewicz & Friberg, Applied Optics 42, 9-17 (2003): oblique
  illumination of axicons broadens the focal line and produces astigmatic /
  astroid-caustic behaviour that must be recovered at sufficiently large angle.
* Matsushima, Schimmel & Wyrowski, JOSA A 20, 1755-1762 (2003): diffraction
  between tilted planes is handled by rotation of the angular spectrum.
* Brzobohaty, Cizmar & Zemanek, Optics Express 16, 12688-12700 (2008): rounded
  axicon tips generate a second refracted component and axial modulation with
  period lambda/(1-cos(alpha0)).

No function here changes accepted Phase 2A/2B/2C contracts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from vbb_study.equations.fields import fft2c


EPS = np.finfo(float).tiny


@dataclass(frozen=True)
class GratingOrderGeometry:
    order: int
    wavelength_m: float
    carrier_cpm: float
    input_angle_x_rad: float
    input_angle_y_rad: float
    sx_in: float
    sy_in: float
    sx_order: float
    sy_order: float
    sz_order: float
    propagating: bool


@dataclass(frozen=True)
class RefractiveAxiconGeometry:
    base_angle_rad: float
    refractive_index: float
    external_index: float
    deflection_rad: float
    exact_radial_direction_sine: float
    shallow_radial_direction_sine: float
    shallow_relative_error: float


def grating_order_direction_cosines(
    *,
    wavelength_m: float,
    carrier_cpm: float,
    input_angle_x_rad: float = 0.0,
    input_angle_y_rad: float = 0.0,
    order: int = 1,
) -> GratingOrderGeometry:
    """Direction cosines of a 1-D grating order from tangential momentum.

    For a grating vector G along +x,

        kx,m = kx,in + m 2*pi*G

    or, in direction-cosine form in air,

        sx,m = sin(theta_x) + m*lambda*G.

    ``input_angle_x_rad`` and ``input_angle_y_rad`` are independent laboratory
    pointing components.  The relation is exact for the stated scalar grating
    geometry; it is not an LC electro-optic diffraction-efficiency model.
    """

    wavelength_m = float(wavelength_m)
    carrier_cpm = float(carrier_cpm)
    sx_in = math.sin(float(input_angle_x_rad))
    sy_in = math.sin(float(input_angle_y_rad))
    sx_order = sx_in + int(order) * wavelength_m * carrier_cpm
    sy_order = sy_in
    transverse_sq = sx_order * sx_order + sy_order * sy_order
    propagating = transverse_sq < 1.0
    sz_order = math.sqrt(max(0.0, 1.0 - transverse_sq)) if propagating else float("nan")
    return GratingOrderGeometry(
        order=int(order),
        wavelength_m=wavelength_m,
        carrier_cpm=carrier_cpm,
        input_angle_x_rad=float(input_angle_x_rad),
        input_angle_y_rad=float(input_angle_y_rad),
        sx_in=float(sx_in),
        sy_in=float(sy_in),
        sx_order=float(sx_order),
        sy_order=float(sy_order),
        sz_order=float(sz_order),
        propagating=bool(propagating),
    )


def fourier_lens_position_m(
    geometry: GratingOrderGeometry,
    *,
    focal_length_m: float,
) -> tuple[float, float]:
    """Geometrical Fourier-plane position x=f*tan(theta_x), y=f*tan(theta_y)."""

    if not geometry.propagating:
        return float("nan"), float("nan")
    f = float(focal_length_m)
    return (
        f * geometry.sx_order / max(geometry.sz_order, EPS),
        f * geometry.sy_order / max(geometry.sz_order, EPS),
    )


def first_order_iris_geometry(
    *,
    wavelength_m: float,
    carrier_cpm: float,
    iris_radius_cpm: float,
    focal_length_m: float,
    input_angle_x_rad: float = 0.0,
    input_angle_y_rad: float = 0.0,
    order: int = 1,
) -> dict[str, float | bool]:
    """Expected grating-order displacement relative to the fixed +1-order iris.

    The actual Phase 2A/2E Fourier mask is fixed at ``fx=carrier_cpm, fy=0``.
    Input pointing translates the complete angular spectrum by
    ``sin(theta)/lambda``, so this diagnostic exposes how far the expected order
    centre has moved relative to that fixed mask.
    """

    geom = grating_order_direction_cosines(
        wavelength_m=wavelength_m,
        carrier_cpm=carrier_cpm,
        input_angle_x_rad=input_angle_x_rad,
        input_angle_y_rad=input_angle_y_rad,
        order=order,
    )
    expected_fx = float(carrier_cpm) * int(order) + geom.sx_in / float(wavelength_m)
    expected_fy = geom.sy_in / float(wavelength_m)
    iris_fx = float(carrier_cpm) * int(order)
    iris_fy = 0.0
    offset_cpm = math.hypot(expected_fx - iris_fx, expected_fy - iris_fy)
    offset_ratio = offset_cpm / max(float(iris_radius_cpm), EPS)

    expected_x, expected_y = fourier_lens_position_m(geom, focal_length_m=focal_length_m)
    nominal_geom = grating_order_direction_cosines(
        wavelength_m=wavelength_m,
        carrier_cpm=carrier_cpm,
        order=order,
    )
    iris_x, iris_y = fourier_lens_position_m(nominal_geom, focal_length_m=focal_length_m)
    return {
        "propagating": bool(geom.propagating),
        "expected_fx_cpm": float(expected_fx),
        "expected_fy_cpm": float(expected_fy),
        "fixed_iris_fx_cpm": float(iris_fx),
        "fixed_iris_fy_cpm": float(iris_fy),
        "order_offset_cpm": float(offset_cpm),
        "order_offset_over_iris_radius": float(offset_ratio),
        "expected_fourier_x_m": float(expected_x),
        "expected_fourier_y_m": float(expected_y),
        "fixed_iris_fourier_x_m": float(iris_x),
        "fixed_iris_fourier_y_m": float(iris_y),
        "fourier_shift_x_m": float(expected_x - iris_x),
        "fourier_shift_y_m": float(expected_y - iris_y),
    }


def fourier_order_diagnostics(
    post_slm: np.ndarray,
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    carrier_cpm: float,
    iris_radius_cpm: float,
    focal_length_m: float,
    input_angle_x_rad: float = 0.0,
    input_angle_y_rad: float = 0.0,
    order: int = 1,
) -> dict[str, Any]:
    """Measure the numerical +1 order against its analytic pointing prediction."""

    field = np.asarray(post_slm, dtype=np.complex128)
    spectrum = fft2c(field)
    intensity = np.abs(spectrum) ** 2
    fx = np.asarray(grid["FX"], dtype=float)
    fy = np.asarray(grid["FY"], dtype=float)

    analytic = first_order_iris_geometry(
        wavelength_m=wavelength_m,
        carrier_cpm=carrier_cpm,
        iris_radius_cpm=iris_radius_cpm,
        focal_length_m=focal_length_m,
        input_angle_x_rad=input_angle_x_rad,
        input_angle_y_rad=input_angle_y_rad,
        order=order,
    )
    fixed_mask = (
        (fx - float(analytic["fixed_iris_fx_cpm"])) ** 2
        + (fy - float(analytic["fixed_iris_fy_cpm"])) ** 2
        <= float(iris_radius_cpm) ** 2
    )
    expected_mask = (
        (fx - float(analytic["expected_fx_cpm"])) ** 2
        + (fy - float(analytic["expected_fy_cpm"])) ** 2
        <= float(iris_radius_cpm) ** 2
    )
    total = float(np.sum(intensity))
    fixed_selected = float(np.sum(intensity[fixed_mask])) / max(total, EPS)
    expected_selected = float(np.sum(intensity[expected_mask])) / max(total, EPS)

    # Local centroid around the analytically expected order.  This is a numerical
    # validation of order translation, not a replacement for the analytic law.
    local_weight = np.where(expected_mask, intensity, 0.0)
    local_sum = float(np.sum(local_weight))
    centroid_fx = float(np.sum(local_weight * fx) / max(local_sum, EPS))
    centroid_fy = float(np.sum(local_weight * fy) / max(local_sum, EPS))
    centroid_error_cpm = math.hypot(
        centroid_fx - float(analytic["expected_fx_cpm"]),
        centroid_fy - float(analytic["expected_fy_cpm"]),
    )

    return {
        **analytic,
        "fixed_iris_selected_spectral_fraction": float(fixed_selected),
        "expected_center_selected_spectral_fraction": float(expected_selected),
        "measured_local_centroid_fx_cpm": float(centroid_fx),
        "measured_local_centroid_fy_cpm": float(centroid_fy),
        "centroid_error_cpm": float(centroid_error_cpm),
        "spectrum_intensity": intensity,
        "fixed_iris_mask": fixed_mask,
        "expected_order_mask": expected_mask,
    }


def snell_axicon_geometry(
    *,
    base_angle_rad: float,
    refractive_index: float,
    external_index: float = 1.0,
) -> RefractiveAxiconGeometry:
    """Exact normal-incidence cone deflection for a refractive axicon.

    For a ray entering the flat face normally and leaving a conical face whose
    normal is tilted by gamma from the axis,

        n_axicon sin(gamma) = n_ext sin(gamma + beta),

    hence

        beta = asin((n_axicon/n_ext) sin(gamma)) - gamma.

    The shallow thin-element phase model has transverse direction sine
    approximately ``(n_axicon/n_ext - 1) tan(gamma)``.
    """

    gamma = float(base_angle_rad)
    n_ax = float(refractive_index)
    n_ext = float(external_index)
    argument = n_ax / n_ext * math.sin(gamma)
    if abs(argument) > 1.0:
        raise ValueError("axicon exit surface is beyond the propagating Snell branch")
    beta = math.asin(argument) - gamma
    exact = math.sin(beta)
    shallow = (n_ax / n_ext - 1.0) * math.tan(gamma)
    relative = (shallow - exact) / max(abs(exact), EPS)
    return RefractiveAxiconGeometry(
        base_angle_rad=gamma,
        refractive_index=n_ax,
        external_index=n_ext,
        deflection_rad=float(beta),
        exact_radial_direction_sine=float(exact),
        shallow_radial_direction_sine=float(shallow),
        shallow_relative_error=float(relative),
    )


def exact_refractive_axicon_kr_m_inv(
    *,
    wavelength_m: float,
    base_angle_rad: float,
    refractive_index: float,
    external_index: float = 1.0,
) -> float:
    geometry = snell_axicon_geometry(
        base_angle_rad=base_angle_rad,
        refractive_index=refractive_index,
        external_index=external_index,
    )
    k_external = 2.0 * math.pi * float(external_index) / float(wavelength_m)
    return float(k_external * geometry.exact_radial_direction_sine)


def rounded_tip_modulation_period_m(
    *,
    wavelength_in_medium_m: float,
    cone_angle_rad: float,
) -> float:
    """Brzobohaty et al. round-tip axial interference period.

    A low-spatial-frequency component from the rounded tip interferes with the
    conical quasi-Bessel component.  Their axial wave-vector difference gives

        Lambda_z = lambda / (1 - cos(alpha0)).
    """

    denominator = 1.0 - math.cos(float(cone_angle_rad))
    if denominator <= 0.0:
        return float("inf")
    return float(wavelength_in_medium_m) / denominator


def error_model_fidelity_registry() -> dict[str, dict[str, Any]]:
    """Hard report policy for physical-error models."""

    return {
        "input_beam_angle": {
            "status": "candidate_after_reference_checks",
            "physics": (
                "finite-angle input plane wave before SLM1; fixed grating/4F iris; "
                "thin-axicon diffraction"
            ),
            "required_checks": [
                "analytic grating-order translation",
                "fixed-iris selected-order efficiency",
                "small-angle thin-axicon benchmark",
                "moderate-angle oblique-axicon morphology benchmark",
            ],
            "calibration_blockers": [
                "actual SLM incidence geometry",
                "angle-dependent LCOS phase LUT for large nominal incidence",
            ],
        },
        "axicon_tilt": {
            "status": "blocked_for_report",
            "physics": "requires propagation to/from a rigidly tilted optic plane",
            "required_backend": (
                "rotated-angular-spectrum or equivalent tilted-plane propagation, "
                "plus independent direct-diffraction validation"
            ),
            "supersedes": "rotated_thin_element_opd_small_angle as report evidence",
        },
        "rounded_axicon_tip": {
            "status": "candidate_after_reference_checks",
            "physics": "measured/assumed hyperboloidal sag plus scalar diffraction",
            "required_checks": [
                "sharp-tip limit",
                "axisymmetric Hankel/Fresnel reference for B0",
                "predicted axial beat-period consistency",
                "B0 more sensitive than vortex controls under matched illumination",
            ],
        },
        "high_angle_refractive_axicon": {
            "status": "blocked_for_shallow_phase_quantitative_use",
            "physics": "exact Snell cone deflection or two-surface ray-wave model",
            "required_checks": [
                "exact-vs-shallow kr comparison",
                "explicit physical angle convention",
            ],
        },
    }
