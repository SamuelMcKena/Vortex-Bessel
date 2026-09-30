from pathlib import Path

from hexapod_lab.controller_profile import load_hxp_controller_profile


def profile():
    return load_hxp_controller_profile(
        Path(__file__).resolve().parents[1]
        / "assets"
        / "hxp_controller_profile_2026-09-30.json"
    )


def test_controller_backup_profile_matches_live_configuration():
    p = profile()
    assert p.host == "192.168.0.254"
    assert p.port == 5001
    assert p.group == "HEXAPOD"
    assert p.coordinate_system == "Work"
    assert p.work_in_world == (0.0, 0.0, 209.0, 0.0, 0.0, 0.0)
    assert p.base_in_world == (0.0, 0.0, 25.0, 0.0, 0.0, 0.0)
    assert p.tool_in_carriage == (0.0, 0.0, 25.0, 0.0, 0.0, 0.0)
    assert p.geometry["base_diameter_mm"] == 240.0
    assert p.geometry["carriage_diameter_mm"] == 150.0
    assert p.geometry["actuator_length_at_home_preset_mm"] == 167.5410634
    assert len(p.actuators) == 6
    assert all(a.maximum_target_position_mm == 24.25 for a in p.actuators)


def test_attenuator_mapping_uses_confirmed_four_equals_forty():
    p = profile()
    assert p.attenuator_gpio == "GPIO2.DAC1"
    assert p.raw_to_transmission_percent(4.0) == 40.0
    assert p.transmission_percent_to_raw(40.0) == 4.0
    assert p.transmission_percent_to_raw(100.0) == 10.0
