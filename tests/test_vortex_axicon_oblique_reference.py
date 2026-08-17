from __future__ import annotations

import math

import numpy as np

from vbb_study.digital_twin.vortex_axicon_oblique_reference import (
    oblique_refractive_axicon_rays,
    refract_direction,
)
from vbb_study.digital_twin.vortex_error_reference_models import snell_axicon_geometry


def test_vector_snell_preserves_unit_direction_and_normal_incidence() -> None:
    out = refract_direction(
        np.asarray([0.0, 0.0, 1.0]),
        np.asarray([0.0, 0.0, 1.0]),
        n1=1.0,
        n2=1.458,
    )
    assert np.allclose(out, [0.0, 0.0, 1.0], rtol=0.0, atol=1e-14)
    assert np.isclose(np.linalg.norm(out), 1.0, rtol=0.0, atol=1e-14)


def test_zero_tilt_ray_cone_matches_exact_normal_incidence_snell_geometry() -> None:
    gamma = math.radians(2.0)
    ref = oblique_refractive_axicon_rays(
        base_angle_rad=gamma,
        refractive_index=1.458,
        external_index=1.0,
        azimuth_samples=720,
    )
    exact = snell_axicon_geometry(
        base_angle_rad=gamma,
        refractive_index=1.458,
        external_index=1.0,
    )

    assert np.isclose(
        ref.cone_radius_mean,
        exact.exact_radial_direction_sine,
        rtol=0.0,
        atol=2e-14,
    )
    assert ref.cone_radius_anisotropy_fraction < 1e-11
    assert ref.second_harmonic_fraction < 1e-11
    assert np.max(np.abs(np.linalg.norm(ref.outgoing_lab, axis=1) - 1.0)) < 2e-14


def test_oblique_refractive_axicon_has_nonzero_azimuthal_cone_anisotropy() -> None:
    gamma = math.radians(2.0)
    ref5 = oblique_refractive_axicon_rays(
        base_angle_rad=gamma,
        refractive_index=1.458,
        external_index=1.0,
        tilt_y_rad=math.radians(5.0),
        azimuth_samples=720,
    )
    ref10 = oblique_refractive_axicon_rays(
        base_angle_rad=gamma,
        refractive_index=1.458,
        external_index=1.0,
        tilt_y_rad=math.radians(10.0),
        azimuth_samples=720,
    )

    assert ref5.cone_radius_anisotropy_fraction > 1e-3
    assert ref10.cone_radius_anisotropy_fraction > ref5.cone_radius_anisotropy_fraction
    assert ref10.second_harmonic_fraction > ref5.second_harmonic_fraction


def test_x_and_y_tilt_references_are_rotated_versions_of_same_axisymmetric_optic() -> None:
    kwargs = dict(
        base_angle_rad=math.radians(2.0),
        refractive_index=1.458,
        external_index=1.0,
        azimuth_samples=720,
    )
    xref = oblique_refractive_axicon_rays(
        tilt_x_rad=math.radians(7.0),
        **kwargs,
    )
    yref = oblique_refractive_axicon_rays(
        tilt_y_rad=math.radians(7.0),
        **kwargs,
    )

    assert np.isclose(
        xref.cone_radius_anisotropy_fraction,
        yref.cone_radius_anisotropy_fraction,
        rtol=0.0,
        atol=2e-12,
    )
    assert np.isclose(
        xref.second_harmonic_fraction,
        yref.second_harmonic_fraction,
        rtol=0.0,
        atol=2e-12,
    )
