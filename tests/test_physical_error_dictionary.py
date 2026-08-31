from __future__ import annotations

import numpy as np

from vbb_study.digital_twin.physical_error_dictionary import (
    combine_error_configs,
    correction_handoff_manifest,
    greedy_fit_error_dictionary,
    rank_error_families,
)
from vbb_study.digital_twin.vortex_beam_slm_errors import GaussianBeamError, SLMError
from vbb_study.digital_twin.vortex_explicit_4f import FourFError
from vbb_study.digital_twin.vortex_system_route import AxiconError, SystemErrorConfig


def _toy_stack(config: SystemErrorConfig) -> np.ndarray:
    """Cheap deterministic stack whose channels encode independent error fields."""
    z = np.linspace(-1.0, 1.0, 5)[:, None, None]
    y = np.linspace(-1.0, 1.0, 7)[None, :, None]
    x = np.linspace(-1.0, 1.0, 9)[None, None, :]
    ax = float(config.axicon.decentre_m[0]) / 1e-6
    slm = float(config.slm1.pattern_offset_m[0]) / 1e-6
    iris = float(config.fourf.iris_offset_m[0]) / 1e-6
    beam = float(config.beam.decentre_m[0]) / 1e-6
    base = np.exp(-((x - 0.0015 * ax) ** 2 + (y - 0.0010 * slm) ** 2) / 0.42)
    mod = 1.0 + 0.0012 * iris * z + 0.0008 * beam * x
    return np.maximum(base * mod, 0.0)


def _registry():
    vals = (-400e-6, -200e-6, 0.0, 200e-6, 400e-6)
    return {
        "axicon_x": {
            "values": vals,
            "units": "m",
            "fidelity": "toy",
            "builder": lambda v: SystemErrorConfig(axicon=AxiconError(decentre_m=(float(v), 0.0))),
        },
        "slm1_x": {
            "values": vals,
            "units": "m",
            "fidelity": "toy",
            "builder": lambda v: SystemErrorConfig(slm1=SLMError(pattern_offset_m=(float(v), 0.0))),
        },
        "iris_x": {
            "values": vals,
            "units": "m",
            "fidelity": "toy",
            "builder": lambda v: SystemErrorConfig(fourf=FourFError(iris_offset_m=(float(v), 0.0))),
        },
        "beam_x": {
            "values": vals,
            "units": "m",
            "fidelity": "toy",
            "builder": lambda v: SystemErrorConfig(beam=GaussianBeamError(decentre_m=(float(v), 0.0))),
        },
    }


def test_recursive_config_merge_preserves_existing_error_planes():
    a = SystemErrorConfig(
        beam=GaussianBeamError(decentre_m=(100e-6, 0.0)),
        fourf=FourFError(iris_offset_m=(200e-6, 0.0)),
    )
    b = SystemErrorConfig(
        slm1=SLMError(pattern_offset_m=(-150e-6, 0.0)),
        axicon=AxiconError(decentre_m=(300e-6, 0.0)),
    )
    merged = combine_error_configs(a, b)
    assert merged.beam.decentre_m == a.beam.decentre_m
    assert merged.fourf.iris_offset_m == a.fourf.iris_offset_m
    assert merged.slm1.pattern_offset_m == b.slm1.pattern_offset_m
    assert merged.axicon.decentre_m == b.axicon.decentre_m


def test_family_ranking_selects_correct_physical_model():
    reg = _registry()
    truth = SystemErrorConfig(axicon=AxiconError(decentre_m=(200e-6, 0.0)))
    ranked = rank_error_families(
        families=list(reg),
        target_stack=_toy_stack(truth),
        simulate_config=_toy_stack,
        registry=reg,
    )
    assert ranked[0].family == "axicon_x"
    assert np.isclose(ranked[0].best_value, 200e-6)
    assert ranked[0].best_cost < ranked[1].best_cost


def test_greedy_dictionary_can_accumulate_two_error_planes():
    reg = _registry()
    truth = SystemErrorConfig(
        slm1=SLMError(pattern_offset_m=(-200e-6, 0.0)),
        axicon=AxiconError(decentre_m=(200e-6, 0.0)),
    )
    fit = greedy_fit_error_dictionary(
        families=list(reg),
        target_stack=_toy_stack(truth),
        simulate_config=_toy_stack,
        registry=reg,
        max_stages=2,
        minimum_improvement_fraction=1e-6,
    )
    accepted = {s.accepted_family for s in fit.stages if s.accepted_family}
    assert accepted == {"axicon_x", "slm1_x"}
    assert np.isclose(fit.fitted_config.axicon.decentre_m[0], 200e-6)
    assert np.isclose(fit.fitted_config.slm1.pattern_offset_m[0], -200e-6)
    assert fit.final_cost < fit.initial_cost


def test_handoff_points_to_existing_q20_residual_correction_contract():
    reg = _registry()
    truth = SystemErrorConfig(axicon=AxiconError(decentre_m=(200e-6, 0.0)))
    fit = greedy_fit_error_dictionary(
        families=["axicon_x", "slm1_x"],
        target_stack=_toy_stack(truth),
        simulate_config=_toy_stack,
        registry=reg,
        max_stages=1,
        minimum_improvement_fraction=1e-6,
    )
    handoff = correction_handoff_manifest(fit)
    assert handoff["residual_phase_stage"]["programmed_vortex_removed_from_aberration"] is True
    assert handoff["residual_phase_stage"]["native_slm2_correction"] == "slm2_correction_phase_rad.npy"
    assert "target-minus-model intensity" in handoff["not_permitted"][0]
