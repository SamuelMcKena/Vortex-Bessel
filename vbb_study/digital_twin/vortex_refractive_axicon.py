"""Explicit two-surface refractive axicon geometry and eikonal reference.

This module is the replacement foundation for rigid-tilt physics.  Unlike the
thin rotated-phase surrogate, it traces the actual flat entrance interface and
conical exit interface of a refractive axicon.  It computes surface
intersections, local normals, vector Snell refraction, optical path length and
(optional) polarization-dependent Fresnel transmission.

The primary output is a ray/eikonal bundle on a plane perpendicular to the mean
outgoing direction.  A later wave-optics layer may resample that eikonal field
and propagate it, but it must pass the phase-gradient/ray-direction consistency
gate implemented here first.

Absolute laboratory predictions require the *physical* axicon geometry:
manufacturer angle convention, clear radius and centre/edge thickness.  These
are deliberately mandatory inputs rather than silently inherited guesses.

References
----------
Z. Bin and L. Zhu, Applied Optics 37, 2563-2568 (1998): diffraction of an
axicon under oblique illumination, with theory checked experimentally.
A. Thaning, Z. Jaroszewicz and A. T. Friberg, Applied Optics 42, 9-17 (2003):
oblique axicons produce broadened focal lines / astroid caustics; direct
diffraction simulations and experiment confirm the asymptotic theory.
J. Dudutis et al., Optics Express 26, 3627-3637 (2018): controlled axicon tilt
was used experimentally to manipulate astigmatism in Bessel-beam glass
processing.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from vbb_study.digital_twin.vortex_axicon_oblique_reference import (
    refract_direction,
)
from vbb_study.digital_twin.vortex_rotated_plane import rotation_matrix


EPS = np.finfo(float).eps
TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class RefractiveAxiconGeometry:
    """Physical plano-conical axicon geometry.

    ``base_angle_rad`` is the angle between the conical surface and the flat
    entrance surface.  The local entrance plane is z=0.  The conical exit
    surface is

        z = centre_thickness_m - r * tan(base_angle_rad).

    Hence the centre is thickest.  ``clear_radius_m`` is the physical radial
    aperture.  Positive edge thickness is enforced.
    """

    base_angle_rad: float
    clear_radius_m: float
    centre_thickness_m: float
    refractive_index: float
    external_index: float = 1.0

    def validate(self) -> None:
        gamma = float(self.base_angle_rad)
        radius = float(self.clear_radius_m)
        thickness = float(self.centre_thickness_m)
        n_ax = float(self.refractive_index)
        n_ext = float(self.external_index)
        if not (0.0 < gamma < 0.5 * math.pi):
            raise ValueError("base_angle_rad must lie in (0, pi/2)")
        if radius <= 0.0 or thickness <= 0.0:
            raise ValueError("clear radius and centre thickness must be positive")
        if n_ax <= 0.0 or n_ext <= 0.0:
            raise ValueError("refractive indices must be positive")
        if self.edge_thickness_m <= 0.0:
            raise ValueError(
                "declared centre thickness is too small for the clear radius/base angle"
            )

    @property
    def edge_thickness_m(self) -> float:
        return float(self.centre_thickness_m) - float(self.clear_radius_m) * math.tan(
            float(self.base_angle_rad)
        )


@dataclass(frozen=True)
class RefractiveAxiconBundle:
    entrance_x_m: np.ndarray
    entrance_y_m: np.ndarray
    valid: np.ndarray
    incident_local: np.ndarray
    internal_local: np.ndarray
    exit_point_local_m: np.ndarray
    exit_normal_local: np.ndarray
    outgoing_local: np.ndarray
    exit_point_lab_m: np.ndarray
    outgoing_lab: np.ndarray
    reference_origin_lab_m: np.ndarray
    reference_normal_lab: np.ndarray
    reference_e1_lab: np.ndarray
    reference_e2_lab: np.ndarray
    reference_xi_m: np.ndarray
    reference_eta_m: np.ndarray
    reference_distance_m: np.ndarray
    internal_distance_m: np.ndarray
    optical_path_from_entrance_m: np.ndarray
    mapping_jacobian_abs: np.ndarray
    input_normal_cosine: float
    output_reference_cosine: np.ndarray
    fresnel_power_transmission: np.ndarray | None
    output_polarization_lab: np.ndarray | None
    metadata: dict[str, Any]


def _normalise(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    if np.any(norm <= EPS):
        raise ValueError("zero-length vector")
    return arr / norm


def _orthonormal_reference_basis(normal_lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = _normalise(np.asarray(normal_lab, dtype=float))
    trial = np.asarray([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(trial, normal))) > 0.9:
        trial = np.asarray([0.0, 1.0, 0.0], dtype=float)
    e1 = trial - np.dot(trial, normal) * normal
    e1 = _normalise(e1)
    e2 = _normalise(np.cross(normal, e1))
    return np.asarray(e1), np.asarray(e2)


def _cone_intersection_distance_m(
    x0: np.ndarray,
    y0: np.ndarray,
    direction_internal_local: np.ndarray,
    *,
    base_angle_rad: float,
    centre_thickness_m: float,
) -> np.ndarray:
    """Positive first intersection of an internal ray with the conical exit."""

    sx, sy, sz = map(float, np.asarray(direction_internal_local, dtype=float))
    m = math.tan(float(base_angle_rad))
    tc = float(centre_thickness_m)
    x = np.asarray(x0, dtype=float)
    y = np.asarray(y0, dtype=float)

    a = sz * sz - m * m * (sx * sx + sy * sy)
    b = -2.0 * (tc * sz + m * m * (x * sx + y * sy))
    c = tc * tc - m * m * (x * x + y * y)

    if abs(a) < 1e-14:
        if abs(b) < 1e-14:
            raise ValueError("degenerate ray/cone intersection geometry")
        root = -c / b
        return np.asarray(root, dtype=float)

    disc = b * b - 4.0 * a * c
    if np.any(disc < -1e-12 * max(tc * tc, 1.0)):
        raise ValueError("ray misses declared conical surface")
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    r1 = (-b - sqrt_disc) / (2.0 * a)
    r2 = (-b + sqrt_disc) / (2.0 * a)
    positive1 = r1 > 0.0
    positive2 = r2 > 0.0
    root = np.where(
        positive1 & positive2,
        np.minimum(r1, r2),
        np.where(positive1, r1, np.where(positive2, r2, np.nan)),
    )
    return np.asarray(root, dtype=float)


def _fresnel_transmit_vector(
    electric1: np.ndarray,
    direction1: np.ndarray,
    direction2: np.ndarray,
    normal12: np.ndarray,
    *,
    n1: float,
    n2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Transmit a real/complex electric vector and return power transmittance.

    The interface normal points from medium 1 to medium 2.  Standard nonmagnetic
    Fresnel electric-field coefficients are used.  The function vectorises over
    leading dimensions.
    """

    d1 = _normalise(np.asarray(direction1, dtype=float))
    d2 = _normalise(np.asarray(direction2, dtype=float))
    normal = _normalise(np.asarray(normal12, dtype=float))
    d1, d2, normal = np.broadcast_arrays(d1, d2, normal)
    e1 = np.broadcast_to(np.asarray(electric1, dtype=np.complex128), d1.shape)
    e1 = e1 - np.sum(e1 * d1, axis=-1, keepdims=True) * d1

    cos_i = np.sum(d1 * normal, axis=-1)
    cos_t = np.sum(d2 * normal, axis=-1)
    if np.any(cos_i <= 0.0) or np.any(cos_t <= 0.0):
        raise ValueError("Fresnel transmission requires forward rays")

    s_vec = np.cross(normal, d1)
    s_norm = np.linalg.norm(s_vec, axis=-1)
    near_normal = s_norm < 1e-12
    safe_s = np.where(near_normal[..., None], np.asarray([0.0, 1.0, 0.0]), s_vec)
    safe_s = _normalise(safe_s)
    p1 = _normalise(np.cross(d1, safe_s))
    p2 = _normalise(np.cross(d2, safe_s))

    ts = (2.0 * float(n1) * cos_i) / (
        float(n1) * cos_i + float(n2) * cos_t
    )
    tp = (2.0 * float(n1) * cos_i) / (
        float(n2) * cos_i + float(n1) * cos_t
    )

    es = np.sum(e1 * safe_s, axis=-1)
    ep = np.sum(e1 * p1, axis=-1)
    transmitted = ts[..., None] * es[..., None] * safe_s + tp[..., None] * ep[..., None] * p2

    if np.any(near_normal):
        t0 = 2.0 * float(n1) / (float(n1) + float(n2))
        projected2 = e1 - np.sum(e1 * d2, axis=-1, keepdims=True) * d2
        transmitted = np.where(near_normal[..., None], t0 * projected2, transmitted)

    ein2 = np.sum(np.abs(e1) ** 2, axis=-1)
    eout2 = np.sum(np.abs(transmitted) ** 2, axis=-1)
    power = (
        float(n2)
        * cos_t
        * eout2
        / np.maximum(float(n1) * cos_i * ein2, np.finfo(float).tiny)
    )
    return np.asarray(transmitted, dtype=np.complex128), np.asarray(power, dtype=float)


def _mapping_jacobian(
    xi: np.ndarray,
    eta: np.ndarray,
    *,
    dx_m: float,
    dy_m: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    dxi_dy, dxi_dx = np.gradient(np.asarray(xi, dtype=float), float(dy_m), float(dx_m))
    deta_dy, deta_dx = np.gradient(np.asarray(eta, dtype=float), float(dy_m), float(dx_m))
    det = dxi_dx * deta_dy - dxi_dy * deta_dx
    return np.abs(det), {
        "dxi_dx": dxi_dx,
        "dxi_dy": dxi_dy,
        "deta_dx": deta_dx,
        "deta_dy": deta_dy,
        "det_signed": det,
    }


def trace_refractive_axicon_bundle(
    entrance_x_m: np.ndarray,
    entrance_y_m: np.ndarray,
    *,
    geometry: RefractiveAxiconGeometry,
    tilt_x_rad: float = 0.0,
    tilt_y_rad: float = 0.0,
    incident_direction_lab: np.ndarray | None = None,
    reference_gap_m: float = 1.0e-3,
    polarization_lab: np.ndarray | None = None,
    apex_exclusion_radius_m: float = 0.0,
) -> RefractiveAxiconBundle:
    """Trace a regular entrance-plane bundle through the physical axicon."""

    geometry.validate()
    X = np.asarray(entrance_x_m, dtype=float)
    Y = np.asarray(entrance_y_m, dtype=float)
    if X.shape != Y.shape or X.ndim != 2:
        raise ValueError("entrance_x_m and entrance_y_m must be same-shape 2-D arrays")
    if X.shape[0] < 5 or X.shape[1] < 5:
        raise ValueError("ray grid is too small")
    dx = float(np.median(np.diff(X[X.shape[0] // 2])))
    dy = float(np.median(np.diff(Y[:, Y.shape[1] // 2])))
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("entrance grid must increase along x/y")

    rotation = rotation_matrix(float(tilt_x_rad), float(tilt_y_rad))
    d_lab = np.asarray(
        [0.0, 0.0, 1.0] if incident_direction_lab is None else incident_direction_lab,
        dtype=float,
    )
    d_lab = _normalise(d_lab)
    d_local = rotation.T @ d_lab
    flat_normal = np.asarray([0.0, 0.0, 1.0], dtype=float)
    d_internal = refract_direction(
        d_local,
        flat_normal,
        n1=float(geometry.external_index),
        n2=float(geometry.refractive_index),
    )

    t_internal = _cone_intersection_distance_m(
        X,
        Y,
        d_internal,
        base_angle_rad=float(geometry.base_angle_rad),
        centre_thickness_m=float(geometry.centre_thickness_m),
    )
    sx, sy, sz = map(float, d_internal)
    x_exit = X + t_internal * sx
    y_exit = Y + t_internal * sy
    z_exit = t_internal * sz
    r_exit = np.hypot(x_exit, y_exit)
    phi_exit = np.arctan2(y_exit, x_exit)
    sg = math.sin(float(geometry.base_angle_rad))
    cg = math.cos(float(geometry.base_angle_rad))
    cone_normal = np.stack(
        [
            sg * np.cos(phi_exit),
            sg * np.sin(phi_exit),
            np.full_like(phi_exit, cg),
        ],
        axis=-1,
    )
    internal_bundle = np.broadcast_to(d_internal, cone_normal.shape)
    outgoing_local = refract_direction(
        internal_bundle,
        cone_normal,
        n1=float(geometry.refractive_index),
        n2=float(geometry.external_index),
    )

    exit_local = np.stack([x_exit, y_exit, z_exit], axis=-1)
    exit_lab = exit_local @ rotation.T
    outgoing_lab = outgoing_local @ rotation.T

    valid = (
        np.isfinite(t_internal)
        & (t_internal > 0.0)
        & (r_exit <= float(geometry.clear_radius_m))
        & (r_exit >= float(apex_exclusion_radius_m))
        & (outgoing_lab[..., 2] > 0.0)
    )

    # Mean optical axis is defined from ray directions, not from an assumed
    # coordinate axis.  This keeps the downstream reference plane perpendicular
    # to the actual mean outgoing bundle and therefore avoids an artificial
    # high-angle carrier in the sampled reference field.
    mean_direction = _normalise(np.mean(outgoing_lab[valid], axis=0))
    e1_lab, e2_lab = _orthonormal_reference_basis(mean_direction)
    mean_exit = np.mean(exit_lab[valid], axis=0)
    axial_deviation = (exit_lab - mean_exit) @ mean_direction
    required_gap = float(np.max(axial_deviation[valid])) + 10.0 * max(dx, dy)
    gap = max(float(reference_gap_m), required_gap)
    p0 = mean_exit + gap * mean_direction

    denominator = outgoing_lab @ mean_direction
    numerator = (p0 - exit_lab) @ mean_direction
    t_reference = numerator / denominator
    if np.any(t_reference[valid] <= 0.0):
        raise ValueError("reference plane does not lie downstream of all valid rays")
    reference_point = exit_lab + t_reference[..., None] * outgoing_lab
    delta_ref = reference_point - p0
    xi = delta_ref @ e1_lab
    eta = delta_ref @ e2_lab

    opl = (
        float(geometry.refractive_index) * t_internal
        + float(geometry.external_index) * t_reference
    )
    jacobian_abs, _ = _mapping_jacobian(xi, eta, dx_m=dx, dy_m=dy)
    cos_in = float(np.dot(d_local, flat_normal))
    cos_out_reference = outgoing_lab @ mean_direction

    fresnel_power: np.ndarray | None
    output_polarization_lab: np.ndarray | None
    if polarization_lab is None:
        fresnel_power = None
        output_polarization_lab = None
        fresnel_status = "polarization_required_for_absolute_amplitude"
    else:
        e_lab = np.asarray(polarization_lab, dtype=np.complex128)
        if e_lab.shape != (3,):
            raise ValueError("polarization_lab must be a 3-vector")
        e_local = rotation.T @ e_lab
        e_internal, t1 = _fresnel_transmit_vector(
            e_local,
            d_local,
            d_internal,
            flat_normal,
            n1=float(geometry.external_index),
            n2=float(geometry.refractive_index),
        )
        e_out_local, t2 = _fresnel_transmit_vector(
            e_internal,
            internal_bundle,
            outgoing_local,
            cone_normal,
            n1=float(geometry.refractive_index),
            n2=float(geometry.external_index),
        )
        e_out_lab = e_out_local @ rotation.T
        norm = np.sqrt(np.sum(np.abs(e_out_lab) ** 2, axis=-1, keepdims=True))
        output_polarization_lab = e_out_lab / np.maximum(norm, np.finfo(float).tiny)
        fresnel_power = np.asarray(t1 * t2, dtype=float)
        fresnel_status = "vector_s_p_fresnel_from_declared_input_polarization"

    return RefractiveAxiconBundle(
        entrance_x_m=X,
        entrance_y_m=Y,
        valid=np.asarray(valid, dtype=bool),
        incident_local=np.asarray(d_local, dtype=float),
        internal_local=np.asarray(d_internal, dtype=float),
        exit_point_local_m=np.asarray(exit_local, dtype=float),
        exit_normal_local=np.asarray(cone_normal, dtype=float),
        outgoing_local=np.asarray(outgoing_local, dtype=float),
        exit_point_lab_m=np.asarray(exit_lab, dtype=float),
        outgoing_lab=np.asarray(outgoing_lab, dtype=float),
        reference_origin_lab_m=np.asarray(p0, dtype=float),
        reference_normal_lab=np.asarray(mean_direction, dtype=float),
        reference_e1_lab=np.asarray(e1_lab, dtype=float),
        reference_e2_lab=np.asarray(e2_lab, dtype=float),
        reference_xi_m=np.asarray(xi, dtype=float),
        reference_eta_m=np.asarray(eta, dtype=float),
        reference_distance_m=np.asarray(t_reference, dtype=float),
        internal_distance_m=np.asarray(t_internal, dtype=float),
        optical_path_from_entrance_m=np.asarray(opl, dtype=float),
        mapping_jacobian_abs=np.asarray(jacobian_abs, dtype=float),
        input_normal_cosine=cos_in,
        output_reference_cosine=np.asarray(cos_out_reference, dtype=float),
        fresnel_power_transmission=fresnel_power,
        output_polarization_lab=output_polarization_lab,
        metadata={
            "outcome": "TWO-SURFACE-REFRACTIVE-AXICON-EIKONAL",
            "base_angle_rad": float(geometry.base_angle_rad),
            "clear_radius_m": float(geometry.clear_radius_m),
            "centre_thickness_m": float(geometry.centre_thickness_m),
            "edge_thickness_m": float(geometry.edge_thickness_m),
            "refractive_index": float(geometry.refractive_index),
            "external_index": float(geometry.external_index),
            "tilt_x_rad": float(tilt_x_rad),
            "tilt_y_rad": float(tilt_y_rad),
            "reference_gap_m": float(gap),
            "valid_fraction": float(np.mean(valid)),
            "fresnel_status": fresnel_status,
            "surface_model": "flat_entrance_plus_sharp_conical_exit",
            "phase_model": "finite_ray_trace_optical_path",
            "report_figures_authorised": False,
        },
    )


def eikonal_direction_consistency(
    bundle: RefractiveAxiconBundle,
    *,
    wavelength_m: float,
    external_index: float,
    trim_pixels: int = 3,
) -> dict[str, float]:
    """Check that the traced OPL gradient reproduces outgoing wavevectors.

    The phase on the reference plane is the incident phase on the tilted
    entrance plane plus ``k0 * OPL``.  Transforming its numerical gradient from
    entrance coordinates to the irregular reference-plane coordinates must give
    the transverse components of ``k_ext * s_out``.  This is an independent
    Fermat/eikonal consistency check on surface intersection, Snell refraction
    and OPL bookkeeping.
    """

    X = np.asarray(bundle.entrance_x_m, dtype=float)
    Y = np.asarray(bundle.entrance_y_m, dtype=float)
    dx = float(np.median(np.diff(X[X.shape[0] // 2])))
    dy = float(np.median(np.diff(Y[:, Y.shape[1] // 2])))
    k0 = TWOPI / float(wavelength_m)
    kext = k0 * float(external_index)
    sx, sy, _ = map(float, bundle.incident_local)
    incident_phase = kext * (sx * X + sy * Y)
    phase = incident_phase + k0 * np.asarray(bundle.optical_path_from_entrance_m)

    dphase_dy, dphase_dx = np.gradient(phase, dy, dx)
    jac_abs, terms = _mapping_jacobian(
        bundle.reference_xi_m,
        bundle.reference_eta_m,
        dx_m=dx,
        dy_m=dy,
    )
    det = terms["det_signed"]
    gx = (
        terms["deta_dy"] * dphase_dx
        - terms["dxi_dy"] * dphase_dy
    ) / np.where(np.abs(det) > 1e-15, det, np.nan)
    gy = (
        -terms["deta_dx"] * dphase_dx
        + terms["dxi_dx"] * dphase_dy
    ) / np.where(np.abs(det) > 1e-15, det, np.nan)

    expected_x = kext * (bundle.outgoing_lab @ bundle.reference_e1_lab)
    expected_y = kext * (bundle.outgoing_lab @ bundle.reference_e2_lab)
    scale = np.maximum(
        np.sqrt(expected_x * expected_x + expected_y * expected_y),
        0.01 * kext,
    )
    error = np.sqrt((gx - expected_x) ** 2 + (gy - expected_y) ** 2) / scale

    valid = np.asarray(bundle.valid, dtype=bool) & np.isfinite(error) & (jac_abs > 1e-8)
    trim = int(trim_pixels)
    if trim > 0:
        interior = np.zeros_like(valid)
        interior[trim:-trim, trim:-trim] = True
        valid &= interior
    values = error[valid]
    if values.size < 32:
        raise ValueError("too few valid rays for eikonal consistency check")
    return {
        "median_relative_direction_error": float(np.median(values)),
        "p95_relative_direction_error": float(np.percentile(values, 95.0)),
        "max_relative_direction_error": float(np.max(values)),
        "valid_samples": float(values.size),
    }


__all__ = [
    "RefractiveAxiconBundle",
    "RefractiveAxiconGeometry",
    "eikonal_direction_consistency",
    "trace_refractive_axicon_bundle",
]
