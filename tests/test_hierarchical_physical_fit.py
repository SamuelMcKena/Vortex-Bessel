from __future__ import annotations

import numpy as np

from vbb_study.digital_twin.hierarchical_physical_fit import (
    apply_registry_family,
    hierarchical_physical_fit,
    overlay_error_config,
    registry_family_groups,
)
from vbb_study.digital_twin.vortex_system_error_sweeps import system_sweep_registry
from vbb_study.digital_twin.vortex_system_route import SystemErrorConfig


def _plain_rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def test_registry_groups_cover_forward_model_planes():
    groups = registry_family_groups()
    assert {"beam", "slm", "4f", "axicon"}.issubset(groups)
    assert "beam_radius_scale" in groups["beam"]
    assert "fourf_iris_offset_x" in groups["4f"]
    assert "axicon_lateral_decentre_x" in groups["axicon"]


def test_overlay_accumulates_errors_from_different_physical_planes():
    reg = system_sweep_registry()
    cfg = SystemErrorConfig()
    cfg = apply_registry_family(cfg, "beam_radius_scale", 0.85, registry=reg)
    cfg = apply_registry_family(cfg, "axicon_lateral_decentre_x", 250e-6, registry=reg)
    assert np.isclose(cfg.beam.radius_x_scale, 0.85)
    assert np.isclose(cfg.beam.radius_y_scale, 0.85)
    assert np.isclose(cfg.axicon.decentre_m[0], 250e-6)

    # Replacing a third plane must not reset the two already accumulated above.
    patch = reg["fourf_iris_offset_x"]["builder"](0.3e-3)
    merged = overlay_error_config(cfg, patch)
    assert np.isclose(merged.beam.radius_x_scale, 0.85)
    assert np.isclose(merged.axicon.decentre_m[0], 250e-6)
    assert np.isclose(merged.fourf.iris_offset_m[0], 0.3e-3)


def test_hierarchical_fit_recovers_two_registry_families_with_mock_forward_model():
    # The forward model is deliberately lightweight here; the poster workflow
    # separately runs the real optical route.  Orthogonal image features ensure
    # that this unit test exercises family ranking + config accumulation only.
    yy, xx = np.mgrid[-1:1:7j, -1:1:7j]
    feature_radius = np.exp(-4.0 * (xx * xx + yy * yy))
    feature_axicon = xx * np.exp(-3.0 * (xx * xx + yy * yy))
    zscale = np.array([0.7, 1.0, 1.3])[:, None, None]

    def simulate(cfg: SystemErrorConfig) -> np.ndarray:
        dr = (cfg.beam.radius_x_scale - 1.0) / 0.15
        da = cfg.axicon.decentre_m[0] / 250e-6
        base = np.ones((3, 7, 7), float)
        return base + 0.18 * dr * zscale * feature_radius + 0.22 * da * zscale * feature_axicon

    truth = SystemErrorConfig()
    truth = apply_registry_family(truth, "beam_radius_scale", 0.85)
    truth = apply_registry_family(truth, "axicon_lateral_decentre_x", 250e-6)
    target = simulate(truth)

    result = hierarchical_physical_fit(
        target_stack=target,
        simulate_config=simulate,
        families=("beam_radius_scale", "axicon_lateral_decentre_x"),
        max_stages=2,
        min_improvement_fraction=1e-6,
        loss_fn=_plain_rmse,
    )
    assert set(result.fitted_families) == {"beam_radius_scale", "axicon_lateral_decentre_x"}
    assert np.isclose(result.final_config.beam.radius_x_scale, 0.85)
    assert np.isclose(result.final_config.axicon.decentre_m[0], 250e-6)
    assert result.final_cost < 1e-12
    assert result.total_improvement_fraction > 0.999
