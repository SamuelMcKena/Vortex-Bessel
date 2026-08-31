from __future__ import annotations

import numpy as np

from vbb_study.digital_twin.physical_error_inference import (
    grid_search_parameter,
    grid_search_two_parameters,
    morphology_rmse,
    plane_normalise_stack,
)


def _stack(a: float, b: float = 0.0) -> np.ndarray:
    z = np.linspace(-1.0, 1.0, 5)[:, None, None]
    x = np.linspace(-1.0, 1.0, 21)[None, :, None]
    y = np.linspace(-1.0, 1.0, 17)[None, None, :]
    return np.exp(-((x - 0.22 * a * z) ** 2 + (y - 0.18 * b * z) ** 2) / 0.18)


def test_plane_normalisation_preserves_shape_and_sets_peaks():
    arr = _stack(0.3, -0.2) * np.linspace(0.5, 2.0, 5)[:, None, None]
    out = plane_normalise_stack(arr)
    assert out.shape == arr.shape
    assert np.allclose(np.max(out.reshape(5, -1), axis=1), 1.0)


def test_morphology_rmse_zero_for_scaled_same_planes():
    a = _stack(0.25, 0.1)
    scale = np.linspace(0.7, 1.9, a.shape[0])[:, None, None]
    assert morphology_rmse(a, a * scale) < 1e-12


def test_one_parameter_grid_search_recovers_grid_truth():
    target = _stack(0.4)
    result = grid_search_parameter(
        parameter="a",
        units="arb",
        candidate_values=[-0.4, 0.0, 0.4, 0.8],
        target_stack=target,
        simulate=lambda value: _stack(value),
    )
    assert result.best_value == 0.4
    assert result.best_cost < 1e-12


def test_two_parameter_grid_search_recovers_joint_truth():
    target = _stack(0.4, -0.3)
    result = grid_search_two_parameters(
        parameter_x="a",
        units_x="arb",
        values_x=[0.0, 0.2, 0.4, 0.6],
        parameter_y="b",
        units_y="arb",
        values_y=[-0.5, -0.3, -0.1, 0.1],
        target_stack=target,
        simulate=lambda a, b: _stack(a, b),
    )
    assert result.best_x == 0.4
    assert result.best_y == -0.3
    assert result.best_cost < 1e-12
    assert result.costs.shape == (4, 4)
