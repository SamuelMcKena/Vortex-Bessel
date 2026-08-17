from __future__ import annotations

import math

import numpy as np

from vbb_study.digital_twin.vector_refractive_axicon import (
    sample_vector_field_on_tilted_entrance,
)
from vbb_study.digital_twin.vector_refractive_axicon_eikonal import (
    build_tilted_vector_refractive_axicon_field,
    estimate_common_vector_eikonal,
)
from vbb_study.digital_twin.vortex_refractive_axicon import (
    RefractiveAxiconGeometry,
    trace_refractive_axicon_bundle,
)
from vbb_study.digital_twin.vortex_rotated_plane import rotation_matrix
from vbb_study.equations.fields import make_xy_grid
from vbb_study.vector_field import VectorField


WAVELENGTH = 1.029e-6
N_AX = 1.458
N_EXT = 1.0

# Validation envelope for the carrier-tracked rotated-plane -> common-eikonal
# reconstruction on the guarded physical pupil.  These are unit-vector errors,
# approximately angular errors in radians at this scale.  The absolute ceiling is
# 0.1 microradian, with tighter median/p99 tiers.  These are validation tolerances
# only; they do not alter any production eikonal, Snell, Fresnel, remapping, flux,
# transversality or Nyquist gate.
ENTRANCE_EIKONAL_MEDIAN_MAX = 1.0e-8
ENTRANCE_EIKONAL_P99_MAX = 5.0e-8
ENTRANCE_EIKONAL_ABS_MAX = 1.0e-7


def _geometry() -> RefractiveAxiconGeometry:
    return RefractiveAxiconGeometry(
        base_angle_rad=math.radians(2.0),
        clear_radius_m=1.2e-3,
        centre_thickness_m=2.0e-3,
        refractive_index=N_AX,
        external_index=N_EXT,
    )


def _plane_wave(n: int = 128, window_m: float = 3.0e-3) -> VectorField:
    grid = make_xy_grid(n, window_m / n)
    ex = np.full((n, n), 1.0 / np.sqrt(2.0), dtype=np.complex128)
    ey = 1j * ex
    return VectorField(
        ex=ex,
        ey=ey,
        ez=np.zeros_like(ex),
        grid=grid,
        wavelength_m=WAVELENGTH,
        medium_index=N_EXT,
    )


def _plane_wave_entrance_eikonal(source: VectorField, tilt_rad: float):
    """Return the numerical common eikonal and exact local plane-wave direction."""

    envelope, carrier, poynting, _ = sample_vector_field_on_tilted_entrance(
        source,
        tilt_x_rad=0.0,
        tilt_y_rad=float(tilt_rad),
    )
    electric = np.stack([envelope.ex, envelope.ey, envelope.ez], axis=-1)
    rotation = rotation_matrix(0.0, float(tilt_rad))
    estimate = estimate_common_vector_eikonal(
        electric,
        poynting,
        envelope.grid,
        carrier_local_cpm=(float(carrier[0]), float(carrier[1])),
        rotation_local_to_lab=rotation,
        wavelength_m=WAVELENGTH,
        medium_index=N_EXT,
    )
    exact_local = rotation.T @ np.asarray([0.0, 0.0, 1.0])
    return estimate, np.asarray(exact_local, dtype=float)


def test_rotated_plane_common_eikonal_recovers_analytic_plane_wave_direction() -> None:
    """Isolate the field-to-eikonal numerical floor from refractive geometry.

    A laboratory +z plane wave sampled on a rigidly tilted entrance surface has an
    exact constant local wavevector R^T zhat.  Phase 2H represents the large tilted
    carrier analytically and reconstructs only the baseband/common eikonal.  The
    finite FFT square is not itself a physical optic: rotating an infinite plane
    wave on a finite periodic window produces boundary interpolation artefacts.
    Therefore this benchmark is evaluated on the declared physical axicon pupil,
    with a 10% radial guard band.  Downstream rays outside that pupil are never used.

    The absolute numerical envelope is sub-0.1-microradian and many orders below
    the percent-level oblique-axicon anisotropy that motivated the physical
    two-surface model.  Production physics tolerances are unchanged.
    """

    source = _plane_wave()
    geometry = _geometry()
    physical_pupil = np.asarray(source.grid["R"], dtype=float) <= 0.90 * geometry.clear_radius_m
    for tilt_deg in (0.0, 5.0, 10.0):
        estimate, exact_local = _plane_wave_entrance_eikonal(
            source, math.radians(tilt_deg)
        )
        valid = np.asarray(estimate.valid_mask, dtype=bool) & physical_pupil
        error = np.linalg.norm(
            np.asarray(estimate.direction_local, dtype=float) - exact_local,
            axis=-1,
        )[valid]
        assert error.size > 1000
        assert float(np.median(error)) < ENTRANCE_EIKONAL_MEDIAN_MAX
        assert float(np.percentile(error, 99.0)) < ENTRANCE_EIKONAL_P99_MAX
        assert float(np.max(error)) < ENTRANCE_EIKONAL_ABS_MAX


def test_plane_wave_vector_geometry_matches_scalar_reference_with_no_extra_error() -> None:
    """The vector surface solver must not add a second geometry discrepancy.

    The scalar reference receives the exact analytic incident plane-wave direction;
    the vector path receives the numerically reconstructed common eikonal.  We first
    measure that entrance-direction error on the same samples, then require the
    outgoing two-surface Snell result to remain the same order.  This cleanly
    separates rotated-plane/eikonal error from refractive-geometry error.
    """

    geometry = _geometry()
    source = _plane_wave()
    X = np.asarray(source.grid["X"], dtype=float)
    Y = np.asarray(source.grid["Y"], dtype=float)
    incident_lab = np.asarray([0.0, 0.0, 1.0])

    for tilt_deg in (0.0, 5.0, 10.0):
        tilt = math.radians(tilt_deg)
        estimate, exact_local = _plane_wave_entrance_eikonal(source, tilt)
        vector = build_tilted_vector_refractive_axicon_field(
            source,
            geometry=geometry,
            tilt_x_rad=0.0,
            tilt_y_rad=tilt,
            reference_gap_m=0.20e-3,
            output_n=256,
            output_window_m=3.0e-3,
        )
        scalar = trace_refractive_axicon_bundle(
            X,
            Y,
            geometry=geometry,
            tilt_x_rad=0.0,
            tilt_y_rad=tilt,
            incident_direction_lab=incident_lab,
            reference_gap_m=0.20e-3,
            apex_exclusion_radius_m=0.0,
        )
        common = (
            np.asarray(estimate.valid_mask, dtype=bool)
            & np.asarray(vector.geometry_bundle.valid, dtype=bool)
            & np.asarray(scalar.valid, dtype=bool)
        )
        assert np.count_nonzero(common) > 1000

        entrance_error = np.linalg.norm(
            np.asarray(estimate.direction_local, dtype=float) - exact_local,
            axis=-1,
        )[common]
        vector_out = np.asarray(vector.outgoing_direction_lab, dtype=float)[common]
        scalar_out = np.asarray(scalar.outgoing_lab, dtype=float)[common]
        outgoing_error = np.linalg.norm(vector_out - scalar_out, axis=1)

        entrance_median = float(np.median(entrance_error))
        entrance_p99 = float(np.percentile(entrance_error, 99.0))
        entrance_max = float(np.max(entrance_error))
        outgoing_median = float(np.median(outgoing_error))
        outgoing_p99 = float(np.percentile(outgoing_error, 99.0))
        outgoing_max = float(np.max(outgoing_error))

        # Refraction can change angular sensitivity by an order-unity factor, but
        # it must not create a new numerical error scale.  The 2x envelope is a
        # physics-aware conditioning allowance, not a fitted absolute tolerance.
        floor = 2e-11
        assert outgoing_median <= 2.0 * entrance_median + floor
        assert outgoing_p99 <= 2.0 * entrance_p99 + floor
        assert outgoing_max <= 2.0 * entrance_max + floor

        vector_exit = np.asarray(vector.geometry_bundle.exit_point_lab_m, dtype=float)[common]
        scalar_exit = np.asarray(scalar.exit_point_lab_m, dtype=float)[common]
        exit_position_error = np.linalg.norm(vector_exit - scalar_exit, axis=1)
        assert float(np.median(exit_position_error)) < 2e-11
        assert float(np.max(exit_position_error)) < 2e-9

        # Interface geometry is polarization independent; both implementations
        # must also agree on the physical path through the glass within the same
        # entrance-eikonal numerical floor.
        vector_distance = np.asarray(vector.geometry_bundle.internal_distance_m, dtype=float)[common]
        scalar_distance = np.asarray(scalar.internal_distance_m, dtype=float)[common]
        distance_error = np.abs(vector_distance - scalar_distance)
        assert float(np.median(distance_error)) < 2e-11
        assert float(np.max(distance_error)) < 2e-9
