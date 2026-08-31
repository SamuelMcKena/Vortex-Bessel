import numpy as np

from vbb_study.digital_twin.physical_parameter_inference import (
    fit_scalar_parameter,
    normalised_stack_rmse,
)


def _stack(shift: int) -> np.ndarray:
    z = []
    base = np.zeros((21, 21), dtype=float)
    base[10, 10] = 1.0
    for i in range(5):
        plane = np.roll(base, shift + i // 3, axis=1)
        z.append(plane)
    return np.asarray(z)


def test_normalised_stack_rmse_is_zero_for_same_morphology_under_power_scaling():
    a = _stack(2)
    b = 7.5 * a
    assert normalised_stack_rmse(a, b) < 1e-12


def test_scalar_parameter_fit_recovers_known_candidate_and_reports_metrics():
    values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    candidates = [_stack(int(v)) for v in values]
    target = _stack(1)
    fit = fit_scalar_parameter(
        target,
        candidates,
        values,
        parameter_name="synthetic shift",
        parameter_unit="px",
        nominal_value=0.0,
    )
    assert fit.best_value == 1.0
    assert fit.best_cost < 1e-12
    assert fit.nominal_cost > fit.best_cost
    assert fit.improvement_fraction_vs_nominal > 0.99
    assert fit.grid_step == 1.0
    assert fit.second_best_margin > 0.0
