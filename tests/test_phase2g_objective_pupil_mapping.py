from __future__ import annotations

import numpy as np
import pytest

from vbb_study.digital_twin.objective_pupil_mapping import (
    ObjectivePupilMappingConfig,
    map_post_axicon_to_objective_pupil,
)
from vbb_study.equations.fields import make_xy_grid


def test_identity_objective_pupil_map_preserves_complex_field_in_large_pupil() -> None:
    n = 128
    window = 4e-3
    grid = make_xy_grid(n, window / n)
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    field = np.exp(-(X**2 + Y**2) / (0.7e-3**2)) * np.exp(1j * 900.0 * X)
    result = map_post_axicon_to_objective_pupil(
        field,
        grid,
        wavelength_m=1029e-9,
        config=ObjectivePupilMappingConfig(
            free_space_distance_m=0.0,
            output_window_m=window,
            output_n=n,
            pupil_radius_m=3e-3,
        ),
    )
    overlap = abs(np.vdot(field.ravel(), result.field.ravel())) / (
        np.linalg.norm(field) * np.linalg.norm(result.field)
    )
    assert overlap > 0.999999
    assert result.metadata["mapped_to_propagated_power_ratio"] == pytest.approx(1.0, rel=2e-6)
    assert result.metadata["pupil_capture_fraction_of_mapped"] == pytest.approx(1.0, rel=1e-9)


def test_ideal_two_axis_magnification_uses_power_conserving_jacobian() -> None:
    n = 256
    source_window = 4e-3
    grid = make_xy_grid(n, source_window / n)
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    field = np.exp(-(X**2 / (0.45e-3**2) + Y**2 / (0.32e-3**2))).astype(complex)
    mx, my = 1.5, 0.8
    result = map_post_axicon_to_objective_pupil(
        field,
        grid,
        wavelength_m=1029e-9,
        config=ObjectivePupilMappingConfig(
            free_space_distance_m=0.0,
            output_window_m=source_window * max(mx, my),
            output_n=384,
            pupil_radius_m=4e-3,
            magnification_x=mx,
            magnification_y=my,
        ),
    )
    assert result.metadata["affine_mapping_jacobian_area"] == pytest.approx(abs(mx * my))
    assert result.metadata["mapped_to_propagated_power_ratio"] == pytest.approx(1.0, rel=3e-3)


def test_objective_pupil_capture_is_explicit() -> None:
    n = 128
    window = 4e-3
    grid = make_xy_grid(n, window / n)
    field = np.ones((n, n), dtype=complex)
    result = map_post_axicon_to_objective_pupil(
        field,
        grid,
        wavelength_m=1029e-9,
        config=ObjectivePupilMappingConfig(
            free_space_distance_m=0.0,
            output_window_m=window,
            output_n=n,
            pupil_radius_m=0.8e-3,
        ),
    )
    fraction = float(result.metadata["pupil_capture_fraction_of_mapped"])
    expected_area_fraction = np.pi * (0.8e-3) ** 2 / window**2
    assert fraction == pytest.approx(expected_area_fraction, rel=0.03)
