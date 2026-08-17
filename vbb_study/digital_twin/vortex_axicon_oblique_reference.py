"""Independent ray reference for a rigidly tilted refractive axicon.

This module is deliberately *not* the production wave propagator.  It supplies
an independent two-interface Snell-law reference that a scalar/diffractive
axicon model must qualitatively agree with before oblique-tilt figures can be
interpreted physically.

Geometry
--------
The axicon frame is rigidly rotated with respect to the laboratory frame.  A
laboratory +z ray first refracts through the flat entrance face, then through a
conical exit surface.  Tangential wave-vector conservation is applied at both
interfaces.  The conical surface normal varies with azimuth, so oblique
illumination produces an azimuth-dependent outgoing cone even for a perfect
axisymmetric axicon.

This is the ray/eikonal limit behind the well-known astigmatic broadening of an
obliquely illuminated axicon.  It does not include diffraction, finite
aperture, Fresnel amplitude coefficients, rounded tips, or a measured surface
map.  Those remain separate wave-optics/calibration problems.

References
----------
Z. Bin and L. Zhu, Applied Optics 37, 2563-2568 (1998).
A. Thaning, Z. Jaroszewicz and A. T. Friberg, Applied Optics 42, 9-17 (2003).
J. Dudutis et al., Optics Express 26, 3627-3637 (2018).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from vbb_study.digital_twin.vortex_rotated_plane import rotation_matrix


EPS = np.finfo(float).eps


@dataclass(frozen=True)
class ObliqueAxiconRayReference:
    """Azimuthal ray bundle produced by a tilted refractive axicon."""

    azimuth_rad: np.ndarray
    incident_external_local: np.ndarray
    internal_local: np.ndarray
    outgoing_local: np.ndarray
    outgoing_lab: np.ndarray
    mean_outgoing_lab: np.ndarray
    relative_transverse_direction: np.ndarray
    cone_radius_direction_sine: np.ndarray
    cone_radius_mean: float
    cone_radius_min: float
    cone_radius_max: float
    cone_radius_anisotropy_fraction: float
    second_harmonic_fraction: float
    tilt_x_rad: float
    tilt_y_rad: float
    base_angle_rad: float
    refractive_index: float
    external_index: float


def _normalise(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    if np.any(norm <= EPS):
        raise ValueError("zero-length direction/normal vector")
    return arr / norm


def refract_direction(
    direction: np.ndarray,
    normal_from_medium1_to_medium2: np.ndarray,
    *,
    n1: float,
    n2: float,
) -> np.ndarray:
    """Vector Snell refraction from medium 1 into medium 2.

    ``normal_from_medium1_to_medium2`` points toward the transmitted medium.
    Tangential wave-vector continuity gives

        n1 * s1_t = n2 * s2_t,

    with the positive-normal transmitted branch selected.  The function is
    vectorised over any leading dimensions shared/broadcast by ``direction``
    and ``normal``.
    """

    if float(n1) <= 0.0 or float(n2) <= 0.0:
        raise ValueError("refractive indices must be positive")

    d = _normalise(np.asarray(direction, dtype=float))
    normal = _normalise(np.asarray(normal_from_medium1_to_medium2, dtype=float))
    d, normal = np.broadcast_arrays(d, normal)

    longitudinal = np.sum(d * normal, axis=-1, keepdims=True)
    if np.any(longitudinal <= 0.0):
        raise ValueError("ray is not incident on the forward side of the interface")

    tangent1 = d - longitudinal * normal
    tangent2 = (float(n1) / float(n2)) * tangent1
    tangent2_sq = np.sum(tangent2 * tangent2, axis=-1, keepdims=True)
    radicand = 1.0 - tangent2_sq
    if np.any(radicand < -1e-12):
        raise ValueError("total internal reflection in declared axicon reference geometry")

    transmitted = tangent2 + np.sqrt(np.maximum(radicand, 0.0)) * normal
    return _normalise(transmitted)


def oblique_refractive_axicon_rays(
    *,
    base_angle_rad: float,
    refractive_index: float,
    external_index: float = 1.0,
    tilt_x_rad: float = 0.0,
    tilt_y_rad: float = 0.0,
    azimuth_samples: int = 720,
) -> ObliqueAxiconRayReference:
    """Trace an azimuthal ray ring through a rigidly tilted refractive axicon.

    The flat entrance face is normal to the axicon axis.  The conical exit-face
    outward normal is

        n(phi) = (sin(gamma) cos(phi), sin(gamma) sin(phi), cos(gamma)).

    For zero rigid tilt this produces the usual inward conical deflection.  A
    non-zero rigid tilt makes the outgoing cone radius azimuth dependent; that
    ray anisotropy is an independent precursor of the astigmatic/caustic
    broadening seen in wave diffraction and experiment.
    """

    gamma = float(base_angle_rad)
    n_ax = float(refractive_index)
    n_ext = float(external_index)
    tx = float(tilt_x_rad)
    ty = float(tilt_y_rad)
    count = int(azimuth_samples)

    if not (0.0 < gamma < 0.5 * math.pi):
        raise ValueError("base angle must lie in (0, pi/2)")
    if n_ax <= 0.0 or n_ext <= 0.0:
        raise ValueError("refractive indices must be positive")
    if count < 32:
        raise ValueError("azimuth_samples must be at least 32")

    rotation = rotation_matrix(tx, ty)
    incident_lab = np.asarray([0.0, 0.0, 1.0], dtype=float)
    incident_local = rotation.T @ incident_lab

    flat_normal = np.asarray([0.0, 0.0, 1.0], dtype=float)
    internal_local = refract_direction(
        incident_local,
        flat_normal,
        n1=n_ext,
        n2=n_ax,
    )

    phi = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    sg = math.sin(gamma)
    cg = math.cos(gamma)
    cone_normal = np.column_stack(
        [
            sg * np.cos(phi),
            sg * np.sin(phi),
            np.full_like(phi, cg),
        ]
    )
    internal_bundle = np.broadcast_to(internal_local, cone_normal.shape)
    outgoing_local = refract_direction(
        internal_bundle,
        cone_normal,
        n1=n_ax,
        n2=n_ext,
    )
    outgoing_lab = outgoing_local @ rotation.T

    mean_out = np.mean(outgoing_lab, axis=0)
    transverse = outgoing_lab[:, :2] - mean_out[None, :2]
    cone_radius = np.linalg.norm(transverse, axis=1)
    radius_mean = float(np.mean(cone_radius))
    radius_min = float(np.min(cone_radius))
    radius_max = float(np.max(cone_radius))
    anisotropy = float((radius_max - radius_min) / max(radius_mean, EPS))
    harmonic2 = float(
        abs(np.mean(cone_radius * np.exp(-2j * phi))) / max(radius_mean, EPS)
    )

    return ObliqueAxiconRayReference(
        azimuth_rad=phi,
        incident_external_local=np.asarray(incident_local, dtype=float),
        internal_local=np.asarray(internal_local, dtype=float),
        outgoing_local=np.asarray(outgoing_local, dtype=float),
        outgoing_lab=np.asarray(outgoing_lab, dtype=float),
        mean_outgoing_lab=np.asarray(mean_out, dtype=float),
        relative_transverse_direction=np.asarray(transverse, dtype=float),
        cone_radius_direction_sine=np.asarray(cone_radius, dtype=float),
        cone_radius_mean=radius_mean,
        cone_radius_min=radius_min,
        cone_radius_max=radius_max,
        cone_radius_anisotropy_fraction=anisotropy,
        second_harmonic_fraction=harmonic2,
        tilt_x_rad=tx,
        tilt_y_rad=ty,
        base_angle_rad=gamma,
        refractive_index=n_ax,
        external_index=n_ext,
    )
