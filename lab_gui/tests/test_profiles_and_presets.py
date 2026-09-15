import json

from slm_lab_control.config import AppConfig
from slm_lab_control.hardware_profiles import profile_for
from slm_lab_control.presets import load_preset, save_preset


def test_locked_hardware_is_reapplied():
    cfg = AppConfig()
    cfg.slm1.geometry.width_px = 123
    cfg.slm1.geometry.height_px = 456
    cfg.slm1.serial = "wrong"
    cfg.slm1.blaze_period_px = 999
    cfg.slm1.switches.blaze = False
    cfg.apply_locked_hardware()

    p = profile_for("SLM1")
    assert cfg.slm1.serial == p.serial
    assert cfg.slm1.geometry.width_px == p.width_px
    assert cfg.slm1.geometry.height_px == p.height_px
    assert cfg.slm1.blaze_period_px == p.blaze_period_px
    assert cfg.slm1.switches.blaze is p.carrier_enabled


def test_old_preset_cannot_override_locked_hardware(tmp_path):
    path = tmp_path / "old.json"
    data = AppConfig().as_dict()
    data["slm2"]["serial"] = "evil-preset-serial"
    data["slm2"]["geometry"]["width_px"] = 100
    data["slm2"]["blaze_period_px"] = 7
    path.write_text(json.dumps(data), encoding="utf-8")

    cfg = load_preset(path)
    p = profile_for("SLM2")
    assert cfg.slm2.serial == p.serial
    assert cfg.slm2.geometry.width_px == p.width_px
    assert cfg.slm2.blaze_period_px == p.blaze_period_px


def test_preset_round_trip_preserves_experiment_values(tmp_path):
    path = tmp_path / "experiment.json"
    cfg = AppConfig()
    cfg.slm1.switches.vortex = True
    cfg.slm1.vortex_charge = 10
    cfg.slm1.z22_cos_amp_waves = 0.13
    save_preset(path, cfg)
    loaded = load_preset(path)
    assert loaded.slm1.switches.vortex is True
    assert loaded.slm1.vortex_charge == 10
    assert loaded.slm1.z22_cos_amp_waves == 0.13
