from pathlib import Path

from hexapod_lab.hardware_profiles import load_legacy_hardware_profile


def test_legacy_hardware_profile_recovers_source_derived_defaults():
    evidence = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "legacy_hardware_evidence.json"
    )
    profile = load_legacy_hardware_profile(evidence)
    assert profile.hxp_host == "192.168.0.254"
    assert profile.hxp_port == 5001
    assert profile.hxp_timeout_s == 10.0
    assert profile.hxp_group == "HEXAPOD"
    assert profile.coordinate_system == "Work"
    assert profile.gate_marker_gpio == "GPIO1.DO"
    assert profile.gate_marker_mask == 4
    assert profile.analog_monitor_gpio == "GPIO2.DAC1"
    assert profile.serial_resource == "COM7"


def test_labview_candidate_keeps_polarity_unresolved():
    evidence = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "legacy_hardware_evidence.json"
    )
    profile = load_legacy_hardware_profile(evidence)
    candidate = profile.candidate("labview_v3")
    assert candidate.gpio_name == "GPIO3.DO"
    assert candidate.mask == 1
    assert candidate.polarity_known is False


def test_tcl_candidate_preserves_write_nonwrite_mapping():
    evidence = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "legacy_hardware_evidence.json"
    )
    profile = load_legacy_hardware_profile(evidence)
    candidate = profile.candidate("tcl_legacy")
    assert candidate.gpio_name == "GPIO4.DO"
    assert candidate.mask == 1
    assert candidate.open_value == 1
    assert candidate.closed_value == 0
