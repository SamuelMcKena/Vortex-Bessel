import json

import numpy as np
import pandas as pd

from iterative_correction_controller import (
    _candidate_mask,
    _interpolate_unit_phasor,
    evaluate_experimental_update,
)


def test_interpolate_unit_phasor_uses_short_complex_path():
    rows = np.asarray([[np.exp(1j*np.deg2rad(170))],
                       [np.exp(1j*np.deg2rad(-170))]])
    value = _interpolate_unit_phasor(0.5, [0.0, 1.0], rows)
    assert np.allclose(np.abs(value), 1.0)
    assert abs(abs(np.rad2deg(np.angle(value[0]))) - 180.0) < 1e-6


def test_candidate_gain_scales_signed_phase_not_wrapped_number():
    correction = np.asarray([[2*np.pi-0.2]])
    accepted = np.asarray([[0.0]])
    candidate = _candidate_mask(correction, accepted, 0.5)
    assert np.isclose(np.angle(np.exp(1j*candidate[0, 0])), -0.1)


def test_experimental_update_requires_all_real_measurement_gates(tmp_path):
    state_path = tmp_path / "closed_loop_state.json"
    candidate = tmp_path / "iteration_000_candidate.npy"
    np.save(candidate, np.zeros((3, 3)))
    state_path.write_text(json.dumps({
        "status": "AWAITING_EXPERIMENTAL_MEASUREMENT", "iteration": 0,
        "candidate_phase_path": candidate.name, "recommended_gain": .05,
    }))
    before = pd.DataFrame({
        "z_mm": [-1.0, 0.0], "measured_vs_ideal_corr": [.70, .72],
        "measured_vs_ideal_rmse": [.20, .19], "measured_ring_cv": [.20, .22],
        "measured_dark_core_ratio": [.02, .02],
    })
    after = pd.DataFrame({
        "z_mm": [-1.0, 0.0], "measured_vs_ideal_corr": [.74, .76],
        "measured_vs_ideal_rmse": [.17, .16], "measured_ring_cv": [.17, .18],
        "measured_dark_core_ratio": [.021, .021],
    })
    before_path, after_path = tmp_path/"before.csv", tmp_path/"after.csv"
    before.to_csv(before_path, index=False); after.to_csv(after_path, index=False)
    result = evaluate_experimental_update(before_path, after_path, state_path)
    assert result["accepted"] is True
    state = json.loads(state_path.read_text())
    assert state["status"] == "EXPERIMENTALLY_ACCEPTED"
    assert (tmp_path/state["accepted_cumulative_phase_path"]).exists()
