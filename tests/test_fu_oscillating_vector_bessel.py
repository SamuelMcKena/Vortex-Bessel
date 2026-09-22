"""Physics checks for the sequential two-SLM polarization-oscillation route."""

from __future__ import annotations

import json

import numpy as np

from vbb_study.digital_twin.fu_oscillating_vector_bessel import paper_masks, run


def test_paper_masks_have_declared_native_periodic_phases() -> None:
    radius = np.array([0.0, 160e-6, 1200e-6])
    theta = np.zeros_like(radius)
    first, second = paper_masks(radius, theta, 3, 160e-6, 1200e-6)
    assert np.allclose(first[:2], 0.0, atol=1e-12)
    assert np.allclose(second[[0, 2]], 0.0, atol=1e-12)


def test_ha2_rotates_six_lobes_and_off_control_stays_fixed(tmp_path) -> None:
    result = run(tmp_path, ell=3, grid_n=512)
    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert result == saved
    assert 45.0 < abs(saved["simulated_rotation_deg_HA2_on"]) < 75.0
    assert abs(saved["simulated_rotation_deg_HA2_off"]) < 1.0
    assert saved["propagation_power_drift_fraction"] < 0.05
    assert len(saved["figures"]) == 3
    assert (tmp_path / "native_phase_design" / "SLM1_HA1_SPP_l3_radians.npy").is_file()
