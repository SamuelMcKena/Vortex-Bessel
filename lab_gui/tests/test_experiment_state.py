import json

from labcontrol.state import ExperimentState, ExperimentStore, PhysicalAxiconState
from slm_lab_control.config import AppConfig


def test_state_round_trip_preserves_typed_optical_state():
    cfg = AppConfig()
    cfg.slm1.switches.vortex = True
    cfg.slm1.vortex_charge = 20
    state = ExperimentState.from_app_config(cfg)
    state.system.physical_axicon = PhysicalAxiconState.IN
    state.camera.current_z_mm = 38.0

    loaded = ExperimentState.from_dict(json.loads(json.dumps(state.to_dict())))

    assert loaded.active_vortex_contributions() == {"SLM1": 20}
    assert loaded.effective_vortex_charge() == 20
    assert loaded.analysis_family() == "vortex_bessel"
    assert loaded.system.physical_axicon is PhysicalAxiconState.IN
    assert loaded.camera.current_z_mm == 38.0


def test_store_emits_structured_changes_and_defends_against_external_mutation():
    store = ExperimentStore(ExperimentState.from_app_config(AppConfig()))
    received = []
    store.subscribe(lambda event, state: received.append((event, state)))

    event = store.set_vortex("SLM1", True, 20, source="test")

    assert event is not None
    assert "slm1.phase.switches.vortex" in event.changed_paths
    assert "slm1.phase.vortex_charge" in event.changed_paths
    assert received[0][1].analysis_family() == "vortex_bessel"

    detached = store.state
    detached.slm1.phase.vortex_charge = 99
    assert store.state.slm1.phase.vortex_charge == 20


def test_simple_gui_config_and_state_use_the_same_phase_models():
    cfg = AppConfig()
    cfg.transfer_mode = "direct_phase_array"
    cfg.slm1.switches.vortex = True
    cfg.slm1.vortex_charge = -12
    store = ExperimentStore(ExperimentState.from_app_config(cfg))

    restored = store.state.to_app_config()

    assert restored.transfer_mode == "direct_phase_array"
    assert restored.slm1.switches.vortex is True
    assert restored.slm1.vortex_charge == -12
    assert restored.slm1.serial == cfg.slm1.serial
