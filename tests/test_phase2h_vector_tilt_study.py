from __future__ import annotations

import math

import numpy as np
import pytest

from vbb_study.digital_twin.vector_refractive_axicon_eikonal import (
    build_tilted_vector_refractive_axicon_field,
)
from vbb_study.digital_twin.vector_tilt_study import (
    beam_moment_metrics,
    centered_coordinate_maps,
    higher_order_cylindrical_vector_input,
    ideal_linear_analyzer_frames,
    vector_line_intensity,
    well_sampled_petal_observable,
)
from vbb_study.digital_twin.vortex_refractive_axicon import RefractiveAxiconGeometry
from vbb_study.vector_field import propagate_vector_asm


GEOMETRY = RefractiveAxiconGeometry(
    base_angle_rad=math.radians(2.0),
    clear_radius_m=1.2e-3,
    centre_thickness_m=2.0e-3,
    refractive_index=1.458,
    external_index=1.0,
)


@pytest.mark.parametrize(("ell", "expected"), [(1, 2), (3, 6)])
@pytest.mark.parametrize("mode", ["radial", "azimuthal"])
def test_generalized_cylindrical_input_has_expected_analyzer_harmonic(
    ell: int, expected: int, mode: str
) -> None:
    field = higher_order_cylindrical_vector_input(
        ell=ell,
        mode=mode,
        n=241,
        window_m=3.0e-3,
        waist_m=0.90e-3,
    )
    Xc, Yc, _ = centered_coordinate_maps(field)
    frames = ideal_linear_analyzer_frames(field)
    for frame in frames.values():
        petals = well_sampled_petal_observable(
            frame,
            Xc,
            Yc,
            pixel_pitch_m=float(field.grid["dx"]),
            minimum_radius_pixels=12.0,
        )
        assert petals.petal_count == expected
        assert petals.modulation_fraction > 0.45


def test_vector_profile_sampling_preserves_constant_total_power_line_symmetry() -> None:
    field = higher_order_cylindrical_vector_input(
        ell=3,
        mode="radial",
        n=128,
        window_m=3.0e-3,
    )
    coord = np.linspace(-1.0e-3, 1.0e-3, 401)
    ix, iy = vector_line_intensity(field, coord, fixed_x_m=0.0, fixed_y_m=0.0)
    np.testing.assert_allclose(ix, ix[::-1], rtol=0.0, atol=2e-10)
    np.testing.assert_allclose(iy, iy[::-1], rtol=0.0, atol=2e-10)


@pytest.mark.parametrize(("ell", "expected"), [(1, 2), (3, 6)])
def test_zero_tilt_physical_axicon_preserves_cylindrical_analyzer_harmonic(ell: int, expected: int) -> None:
    source = higher_order_cylindrical_vector_input(
        ell=ell,
        mode="radial",
        n=128,
        window_m=3.0e-3,
        waist_m=0.75e-3,
    )
    result = build_tilted_vector_refractive_axicon_field(
        source,
        geometry=GEOMETRY,
        tilt_x_rad=0.0,
        tilt_y_rad=0.0,
        reference_gap_m=0.20e-3,
        output_n=384,
        output_window_m=4.5e-3,
    )
    field = propagate_vector_asm(result.field, 30e-3)
    Xc, Yc, metrics = centered_coordinate_maps(field)
    assert abs(metrics.centroid_x_m) < 0.10e-3
    assert abs(metrics.centroid_y_m) < 0.10e-3
    frames = ideal_linear_analyzer_frames(field)
    for frame in frames.values():
        petals = well_sampled_petal_observable(
            frame,
            Xc,
            Yc,
            pixel_pitch_m=float(field.grid["dx"]),
            minimum_radius_pixels=12.0,
        )
        assert petals.petal_count == expected


def test_beam_moments_track_tilt_plane_steering() -> None:
    source = higher_order_cylindrical_vector_input(
        ell=1,
        mode="radial",
        n=128,
        window_m=3.0e-3,
        waist_m=0.75e-3,
    )
    zero = build_tilted_vector_refractive_axicon_field(
        source,
        geometry=GEOMETRY,
        tilt_x_rad=0.0,
        tilt_y_rad=0.0,
        reference_gap_m=0.20e-3,
        output_n=384,
        output_window_m=4.5e-3,
    ).field
    tilted = build_tilted_vector_refractive_axicon_field(
        source,
        geometry=GEOMETRY,
        tilt_x_rad=math.radians(1.0),
        tilt_y_rad=0.0,
        reference_gap_m=0.20e-3,
        output_n=384,
        output_window_m=4.5e-3,
    ).field
    m0 = beam_moment_metrics(propagate_vector_asm(zero, 30e-3))
    mt = beam_moment_metrics(propagate_vector_asm(tilted, 30e-3))
    displacement = math.hypot(mt.centroid_x_m - m0.centroid_x_m, mt.centroid_y_m - m0.centroid_y_m)
    assert displacement > 10e-6
