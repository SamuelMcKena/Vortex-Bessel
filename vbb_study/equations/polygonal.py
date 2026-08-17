"""Polygonal, hexagonal, and discrete N-fold beam equations.

This module owns the compact scalar formulas for:

* the radial profile of a regular N-gon (polygon in polar coordinates),
* the azimuthal angles for a discrete N-fold plane-wave superposition,
* the transverse field of a finite discrete N-fold beam,
* the axial interference period for N-fold superpositions,
* polygon area and perimeter,
* a simple N-fold rotational-symmetry check metric.

What this module does NOT own:

* the full SLM hologram computation for polygonal beams
  (``vbb_study.vbb_polygonal`` and ``vbb_study.vbb_discrete`` own those),
* propagation of polygonal fields
  (``bessel_twin_core`` owns the propagation engine),
* acceptance-depth metrics for real Bessel-like channels
  (``vbb_study.vbb_hexagon_metrics`` owns those),
* phase-only approximation quality assessment.

Coordinate convention:
  ``flat_radius_m`` (also called the *apothem* or *inradius*) is the
  perpendicular distance from the polygon centre to the midpoint of each
  side.  It equals the radius of the inscribed circle.  For a hexagon:
  ``flat_radius = sqrt(3)/2 * side_length``.
  The *tip radius* (circumradius) is ``R_tip = flat_radius / cos(pi/N)``.

Warning on stability claims:
  A hexagonal pattern that looks clean in the focal plane is NOT
  automatically a z-stable Bessel-like channel.  Report accepted depth
  separately using the symmetry and outline-fidelity metrics.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

TWOPI = 2.0 * math.pi
EPS = 1.0e-30


# ---------------------------------------------------------------------------
# Regular polygon geometry
# ---------------------------------------------------------------------------


def regular_polygon_radial_profile_m(
    phi_rad: Any,
    flat_radius_m: float,
    N: int,
    orientation_rad: float = 0.0,
) -> np.ndarray:
    """Return the polar-coordinate radius of a regular N-gon at each angle.

    For a regular polygon with N sides and apothem (flat radius) ``a``,
    the radial profile is::

        r(phi) = a / cos( (phi - orientation) mod (2*pi/N) - pi/N )

    This is the analytical contour of the polygon in polar coordinates.
    The contour is at ``r(phi)`` for every azimuthal angle ``phi``.

    Parameters
    ----------
    phi_rad:
        Array of azimuthal angles, radians.
    flat_radius_m:
        Apothem (perpendicular distance from centre to midpoint of each
        side), metres.
    N:
        Number of polygon sides (e.g. 6 for a hexagon).
    orientation_rad:
        Rotation offset applied before taking the modulo, radians.
        Zero gives a flat base at the bottom for even N.
    """

    phi = np.asarray(phi_rad, dtype=float)
    N_i = int(N)
    if N_i < 3:
        raise ValueError(f"N must be >= 3 for a polygon, got {N_i}")
    half_sector = math.pi / N_i
    sector = TWOPI / N_i
    # Angle within the current sector, measured from the sector midpoint.
    delta = np.mod(phi - float(orientation_rad), sector) - half_sector
    return float(flat_radius_m) / np.cos(delta)


def regular_polygon_vertices(
    flat_radius_m: float,
    N: int,
    orientation_rad: float = 0.0,
) -> np.ndarray:
    """Return regular-polygon vertices as an ``(N, 2)`` array in metres.

    The input radius is the apothem. Vertices therefore sit at the
    circumradius ``flat_radius / cos(pi/N)``. These vertices define the
    geometric target only; they do not imply that an optical field will keep
    that shape during propagation.
    """

    N_i = int(N)
    if N_i < 3:
        raise ValueError(f"N must be >= 3 for a polygon, got {N_i}")
    radius = polygon_tip_radius_m(flat_radius_m, N_i)
    phi = float(orientation_rad) + np.arange(N_i, dtype=float) * TWOPI / N_i
    return np.column_stack((radius * np.cos(phi), radius * np.sin(phi)))


def polygon_radius_function(
    phi_rad: Any,
    flat_radius_m: float,
    N: int,
    orientation_rad: float = 0.0,
) -> np.ndarray:
    """Alias for the regular-polygon polar target radius.

    Stage 8 notebooks use this name when distinguishing a focal-plane target
    contour from a propagation-stable optical field.
    """

    return regular_polygon_radial_profile_m(phi_rad, flat_radius_m, N, orientation_rad)


def polygonal_target_mask(
    R_m: Any,
    Phi_rad: Any,
    *,
    flat_radius_m: float,
    N: int,
    line_width_m: float | None = None,
    hollow: bool = False,
    orientation_rad: float = 0.0,
) -> np.ndarray:
    """Return a filled or hollow regular-polygon focal-plane target mask.

    A filled mask marks ``R <= r_polygon(phi)``. A hollow mask marks pixels
    within ``line_width_m`` of the polygon outline. This is a target geometry
    mask only; it is not a statement about propagation stability.
    """

    R = np.asarray(R_m, dtype=float)
    Phi = np.asarray(Phi_rad, dtype=float)
    r_poly = polygon_radius_function(Phi, flat_radius_m, N, orientation_rad)
    if hollow:
        width = float(line_width_m if line_width_m is not None else 0.05 * float(flat_radius_m))
        return np.abs(R - r_poly) <= max(width, EPS)
    return R <= r_poly


def polygon_tip_radius_m(flat_radius_m: float, N: int) -> float:
    """Return the circumradius (tip-to-centre) of a regular N-gon in metres.

    ``R_tip = flat_radius / cos(pi/N)``
    """

    return float(flat_radius_m) / math.cos(math.pi / int(N))


def polygon_side_length_m(flat_radius_m: float, N: int) -> float:
    """Return the side length of a regular N-gon in metres.

    ``side = 2 * flat_radius * tan(pi/N)``
    """

    return 2.0 * float(flat_radius_m) * math.tan(math.pi / int(N))


def polygon_area_m2(flat_radius_m: float, N: int) -> float:
    """Return the area of a regular N-gon in square metres.

    ``A = N * flat_radius^2 * tan(pi/N)``
    """

    return int(N) * float(flat_radius_m) ** 2 * math.tan(math.pi / int(N))


def polygon_perimeter_m(flat_radius_m: float, N: int) -> float:
    """Return the perimeter of a regular N-gon in metres.

    ``P = 2 * N * flat_radius * tan(pi/N)``
    """

    return 2.0 * int(N) * float(flat_radius_m) * math.tan(math.pi / int(N))


# ---------------------------------------------------------------------------
# Discrete N-fold plane-wave superposition
# ---------------------------------------------------------------------------


def nfold_plane_wave_angles_rad(N: int, offset_rad: float = 0.0) -> np.ndarray:
    """Return the N equally-spaced azimuthal angles for a discrete N-fold beam.

    The k-th plane wave travels at azimuthal angle
    ``phi_k = 2*pi*k/N + offset`` for k = 0, 1, ..., N-1.

    Parameters
    ----------
    N:
        Number of plane waves in the superposition.
    offset_rad:
        Global azimuthal rotation of the whole pattern, radians.
    """

    k = np.arange(int(N), dtype=float)
    return TWOPI * k / int(N) + float(offset_rad)


def discrete_nfold_field_2d(
    R_m: Any,
    Phi_rad: Any,
    *,
    kr_m_inv: float,
    N: int,
    ell: int = 0,
    offset_rad: float = 0.0,
    amplitude: complex = 1.0,
) -> np.ndarray:
    """Return the transverse field of a discrete N-fold plane-wave superposition.

    Each plane wave travels at cone angle ``theta = arcsin(k_r/k)`` and at
    azimuthal angle ``phi_k``.  At z = 0::

        E(r, phi) = A * sum_{k=0}^{N-1} exp(i k_r r cos(phi - phi_k))
                                        * exp(i ell phi_k)

    The optional ``ell * phi_k`` factor adds a topological phase step to
    each plane wave, producing a discrete analogue of a vortex beam.

    Parameters
    ----------
    R_m, Phi_rad:
        Transverse coordinate grids (metres, radians).
    kr_m_inv:
        Transverse wavevector (cone ring radius in k-space), rad/m.
    N:
        Number of plane waves.
    ell:
        Topological charge applied to each plane wave (default 0).
    offset_rad:
        Global rotation of the plane-wave constellation, radians.
    amplitude:
        Overall complex amplitude scale.
    """

    R = np.asarray(R_m, dtype=float)
    Phi = np.asarray(Phi_rad, dtype=float)
    phi_k = nfold_plane_wave_angles_rad(N, offset_rad=offset_rad)
    field = np.zeros_like(R, dtype=complex)
    for pk in phi_k:
        field += np.exp(1j * float(kr_m_inv) * R * np.cos(Phi - pk)) * np.exp(1j * int(ell) * pk)
    return complex(amplitude) * field


def discrete_nfold_field(
    R_m: Any,
    Phi_rad: Any,
    *,
    kr_m_inv: float,
    N: int,
    ell: int = 0,
    offset_rad: float = 0.0,
    amplitude: complex = 1.0,
) -> np.ndarray:
    """Return a finite N-fold plane-wave superposition.

    This is the same field as :func:`discrete_nfold_field_2d`, exposed under
    the Stage 8 name used by notebooks and tests. It creates a discrete
    lattice/kaleidoscope field rather than a localized hollow polygon.
    """

    return discrete_nfold_field_2d(
        R_m,
        Phi_rad,
        kr_m_inv=kr_m_inv,
        N=N,
        ell=ell,
        offset_rad=offset_rad,
        amplitude=amplitude,
    )


def nfold_angular_spectrum(
    N: int,
    *,
    ell: int = 0,
    offset_rad: float = 0.0,
    amplitudes: Sequence[complex] | None = None,
) -> dict[str, np.ndarray]:
    """Return angles and complex weights for a discrete N-fold spectrum.

    The returned deltas live on a common transverse-k ring. Equal amplitudes
    with optional ``ell`` phase steps are the finite-wave construction used for
    discrete N-fold beams.
    """

    angles = nfold_plane_wave_angles_rad(N, offset_rad=offset_rad)
    if amplitudes is None:
        weights = np.ones(int(N), dtype=complex)
    else:
        weights = np.asarray(amplitudes, dtype=complex)
        if weights.size != int(N):
            raise ValueError("amplitudes must have length N")
    weights = weights * np.exp(1j * int(ell) * angles)
    return {"angles_rad": angles, "weights": weights}


# ---------------------------------------------------------------------------
# Axial interference period
# ---------------------------------------------------------------------------


def nfold_axial_period_m(wavelength_m: float, cone_half_angle_rad: float) -> float:
    """Return the axial period of the intensity pattern in metres.

    For a cone-wave superposition with cone half-angle ``theta``, the axial
    phase velocity gives a longitudinal period::

        Lambda_z = lambda / cos(theta)

    For ``N >= 2`` plane waves on the same cone the on-axis intensity has
    the same axial period.  (N-fold azimuthal structure changes the
    transverse pattern but not the axial period for identical cone angles.)

    Parameters
    ----------
    wavelength_m:
        Free-space wavelength of the light, metres.
    cone_half_angle_rad:
        Half-angle of the cone (angle between each plane wave and the
        optical axis), radians.
    """

    cos_theta = math.cos(float(cone_half_angle_rad))
    return float(wavelength_m) / max(cos_theta, EPS)


# ---------------------------------------------------------------------------
# Symmetry metric
# ---------------------------------------------------------------------------


def rotational_symmetry_order_metric(
    intensity_2d: Any,
    *,
    N: int,
    Phi_rad: Any,
    R_m: Any,
    r_inner_m: float = 0.0,
    r_outer_m: float | None = None,
) -> float:
    """Return a normalised N-fold rotational symmetry score in [0, 1].

    Computes the Nth Fourier coefficient of the angular intensity profile
    averaged over the radial band [r_inner, r_outer], normalised to the
    DC (total intensity) coefficient.

    Score 1.0 means perfect N-fold symmetry; 0.0 means no N-fold component.

    Parameters
    ----------
    intensity_2d:
        2-D intensity array.
    N:
        Expected symmetry order.
    Phi_rad:
        2-D azimuthal angle grid matching intensity_2d.
    R_m:
        2-D radial coordinate grid matching intensity_2d.
    r_inner_m, r_outer_m:
        Radial annulus for the average.  Use the main ring region.
    """

    I = np.asarray(intensity_2d, dtype=float)
    Phi = np.asarray(Phi_rad, dtype=float)
    R = np.asarray(R_m, dtype=float)

    mask = R >= float(r_inner_m)
    if r_outer_m is not None:
        mask &= R <= float(r_outer_m)
    if not np.any(mask):
        return float("nan")

    I_ring = I[mask]
    Phi_ring = Phi[mask]

    dc = float(np.mean(I_ring))
    if dc <= EPS:
        return 0.0

    # N-th Fourier coefficient magnitude.
    cn = float(np.abs(np.mean(I_ring * np.exp(-1j * int(N) * Phi_ring))))
    return float(np.clip(cn / dc, 0.0, 1.0))


def symmetry_order_score(
    intensity_2d: Any,
    *,
    N: int,
    Phi_rad: Any,
    R_m: Any,
    r_inner_m: float = 0.0,
    r_outer_m: float | None = None,
) -> float:
    """Return the N-fold rotational-symmetry score in ``[0, 1]``.

    This is a focal-plane or per-z optical metric. A high score at one plane is
    not, by itself, a propagation-stability claim.
    """

    return rotational_symmetry_order_metric(
        intensity_2d,
        N=N,
        Phi_rad=Phi_rad,
        R_m=R_m,
        r_inner_m=r_inner_m,
        r_outer_m=r_outer_m,
    )


def outline_fidelity_score(predicted_mask: Any, target_mask: Any) -> float:
    """Return the Jaccard overlap score for two outline/target masks.

    A score of 1 means identical masks. A score of 0 means no overlap or two
    empty masks. This is a geometry fidelity score, not a material response.
    """

    pred = np.asarray(predicted_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    union = np.count_nonzero(pred | target)
    if union == 0:
        return 0.0
    return float(np.count_nonzero(pred & target) / union)


def edge_uniformity_score(values: Any, mask: Any | None = None) -> float:
    """Return ``1 / (1 + coefficient_of_variation)`` for edge samples.

    Values near 1 indicate uniform edge intensity. The input may be an array of
    contour samples or a full image plus a boolean mask selecting edge pixels.
    """

    arr = np.asarray(values, dtype=float)
    if mask is not None:
        arr = arr[np.asarray(mask, dtype=bool)]
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    mean = float(np.mean(arr))
    if abs(mean) <= EPS:
        return 0.0
    return float(1.0 / (1.0 + np.std(arr) / (abs(mean) + EPS)))


def core_suppression_score(
    intensity_2d: Any,
    R_m: Any,
    *,
    core_radius_m: float,
    reference_mask: Any | None = None,
) -> float:
    """Return a dark-core score in ``[0, 1]``.

    The score is ``1 - core_peak / reference_peak``. A perfectly dark core
    approaches 1, while core leakage comparable to the reference peak scores
    near 0.
    """

    I = np.asarray(intensity_2d, dtype=float)
    R = np.asarray(R_m, dtype=float)
    core = R <= float(core_radius_m)
    if not np.any(core):
        return float("nan")
    if reference_mask is None:
        reference = np.ones_like(core, dtype=bool)
    else:
        reference = np.asarray(reference_mask, dtype=bool)
    reference_peak = float(np.nanmax(I[reference])) if np.any(reference) else float(np.nanmax(I))
    core_peak = float(np.nanmax(I[core]))
    if reference_peak <= EPS:
        return 0.0
    return float(np.clip(1.0 - core_peak / (reference_peak + EPS), 0.0, 1.0))


def side_lobe_contamination_score(
    intensity_2d: Any,
    signal_mask: Any,
    *,
    evaluation_mask: Any | None = None,
) -> float:
    """Return the side-lobe peak divided by the signal peak.

    Lower is better. This diagnostic is useful for hollow polygonal outlines
    where bright light outside the requested outline is a failure mode.
    """

    I = np.asarray(intensity_2d, dtype=float)
    signal = np.asarray(signal_mask, dtype=bool)
    if evaluation_mask is None:
        evaluation = np.ones_like(signal, dtype=bool)
    else:
        evaluation = np.asarray(evaluation_mask, dtype=bool)
    side = evaluation & (~signal)
    signal_peak = float(np.nanmax(I[signal])) if np.any(signal) else float(np.nanmax(I))
    side_peak = float(np.nanmax(I[side])) if np.any(side) else 0.0
    if signal_peak <= EPS:
        return float("nan")
    return float(max(0.0, side_peak / (signal_peak + EPS)))


def accepted_depth_from_metric_stack(
    z_values_m: Any,
    accepted_mask: Any,
) -> dict[str, float | int | bool]:
    """Return longest contiguous accepted-depth metrics from per-z booleans.

    The accepted mask must come from explicit z-dependent metrics such as
    symmetry retention, outline fidelity, core suppression, and side-lobe
    contamination. This function only measures the accepted span.
    """

    z = np.asarray(z_values_m, dtype=float)
    accepted = np.asarray(accepted_mask, dtype=bool)
    if z.size != accepted.size:
        raise ValueError("z_values_m and accepted_mask must have the same length")
    if z.size == 0:
        return {
            "accepted_depth_um": 0.0,
            "accepted_depth_fraction": 0.0,
            "accepted_plane_count": 0,
            "accepted_any": False,
            "accepted_z_start_um": float("nan"),
            "accepted_z_end_um": float("nan"),
        }
    best_depth = 0.0
    start_um = end_um = float("nan")
    if np.any(accepted):
        padded = np.r_[False, accepted, False]
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        starts = changes[0::2]
        ends = changes[1::2] - 1
        spans = z[ends] - z[starts]
        best_idx = int(np.argmax(spans))
        best_depth = float(spans[best_idx] / 1.0e-6)
        start_um = float(z[starts[best_idx]] / 1.0e-6)
        end_um = float(z[ends[best_idx]] / 1.0e-6)
    total_span = float((np.nanmax(z) - np.nanmin(z)) / 1.0e-6) if z.size > 1 else 0.0
    fraction = float(best_depth / total_span) if total_span > EPS else (1.0 if bool(np.any(accepted)) else 0.0)
    return {
        "accepted_depth_um": best_depth,
        "accepted_depth_fraction": fraction,
        "accepted_plane_count": int(np.count_nonzero(accepted)),
        "accepted_any": bool(np.any(accepted)),
        "accepted_z_start_um": start_um,
        "accepted_z_end_um": end_um,
    }


__all__ = [
    "accepted_depth_from_metric_stack",
    "core_suppression_score",
    "discrete_nfold_field",
    "discrete_nfold_field_2d",
    "edge_uniformity_score",
    "nfold_angular_spectrum",
    "nfold_axial_period_m",
    "nfold_plane_wave_angles_rad",
    "outline_fidelity_score",
    "polygon_area_m2",
    "polygon_perimeter_m",
    "polygon_radius_function",
    "polygon_side_length_m",
    "polygon_tip_radius_m",
    "polygonal_target_mask",
    "regular_polygon_radial_profile_m",
    "regular_polygon_vertices",
    "rotational_symmetry_order_metric",
    "side_lobe_contamination_score",
    "symmetry_order_score",
]
