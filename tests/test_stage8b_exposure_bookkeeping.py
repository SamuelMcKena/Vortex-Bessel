"""
Stage 8B exposure bookkeeping tests.

Covers: pulse spacing, pulses per spot, dose-per-unit-length,
static total energy, line exposure summary, and warning conditions.
"""

import math
import pytest

from vbb_study.digital_twin.exposure_bookkeeping import (
    pulse_spacing_um,
    pulses_per_spot,
    dose_per_unit_length_proxy,
    static_exposure_total_energy_uJ,
    line_exposure_summary,
)


# ---------------------------------------------------------------------------
# pulse_spacing_um
# ---------------------------------------------------------------------------


def test_pulse_spacing_basic():
    # 1 mm/s at 1 kHz → 1 µm/s * 1000 / 1000 = 1.0 µm
    # 1 mm/s = 1000 µm/s; 1000 µm/s / 1000 Hz = 1.0 µm
    assert math.isclose(pulse_spacing_um(1.0, 1_000.0), 1.0, rel_tol=1e-9)


def test_pulse_spacing_50kHz():
    # 10 mm/s at 50 kHz → 10_000 µm/s / 50_000 Hz = 0.2 µm
    assert math.isclose(pulse_spacing_um(10.0, 50_000.0), 0.2, rel_tol=1e-9)


def test_pulse_spacing_fast_scan():
    # 100 mm/s at 25 kHz → 100_000 / 25_000 = 4.0 µm
    assert math.isclose(pulse_spacing_um(100.0, 25_000.0), 4.0, rel_tol=1e-9)


def test_pulse_spacing_zero_speed_raises():
    with pytest.raises(ValueError):
        pulse_spacing_um(0.0, 50_000.0)


def test_pulse_spacing_zero_rep_rate_raises():
    with pytest.raises(ValueError):
        pulse_spacing_um(1.0, 0.0)


def test_pulse_spacing_negative_speed_raises():
    with pytest.raises(ValueError):
        pulse_spacing_um(-1.0, 50_000.0)


# ---------------------------------------------------------------------------
# pulses_per_spot
# ---------------------------------------------------------------------------


def test_pulses_per_spot_basic():
    # d=3 µm, v=1 mm/s=1000 µm/s, f=1000 Hz → N = 3*1000/1000 = 3.0
    assert math.isclose(pulses_per_spot(3.0, 1.0, 1_000.0), 3.0, rel_tol=1e-9)


def test_pulses_per_spot_high_speed():
    # d=3 µm, v=100 mm/s, f=25 kHz → 3 * 25000 / 100000 = 0.75 (< 1: dotted)
    assert math.isclose(pulses_per_spot(3.0, 100.0, 25_000.0), 0.75, rel_tol=1e-9)


def test_pulses_per_spot_scales_with_rep_rate():
    n1 = pulses_per_spot(5.0, 1.0, 1_000.0)
    n2 = pulses_per_spot(5.0, 1.0, 2_000.0)
    assert math.isclose(n2, 2.0 * n1, rel_tol=1e-9)


def test_pulses_per_spot_scales_inversely_with_speed():
    n1 = pulses_per_spot(5.0, 1.0, 1_000.0)
    n2 = pulses_per_spot(5.0, 2.0, 1_000.0)
    assert math.isclose(n2, 0.5 * n1, rel_tol=1e-9)


def test_pulses_per_spot_zero_speed_raises():
    with pytest.raises(ValueError):
        pulses_per_spot(3.0, 0.0, 50_000.0)


def test_pulses_per_spot_zero_rep_rate_raises():
    with pytest.raises(ValueError):
        pulses_per_spot(3.0, 1.0, 0.0)


def test_pulses_per_spot_zero_diameter_raises():
    with pytest.raises(ValueError):
        pulses_per_spot(0.0, 1.0, 50_000.0)


# ---------------------------------------------------------------------------
# dose_per_unit_length_proxy
# ---------------------------------------------------------------------------


def test_dose_per_unit_length_basic():
    # E=1 µJ, f=1000 Hz, v=1 mm/s=1e-3 m/s → 1e-6 * 1000 / 1e-3 = 1 J/m
    assert math.isclose(dose_per_unit_length_proxy(1.0, 1_000.0, 1.0), 1.0, rel_tol=1e-9)


def test_dose_per_unit_length_scales_with_energy():
    d1 = dose_per_unit_length_proxy(1.0, 1_000.0, 1.0)
    d2 = dose_per_unit_length_proxy(2.0, 1_000.0, 1.0)
    assert math.isclose(d2, 2.0 * d1, rel_tol=1e-9)


def test_dose_per_unit_length_scales_with_rep_rate():
    d1 = dose_per_unit_length_proxy(1.0, 1_000.0, 1.0)
    d2 = dose_per_unit_length_proxy(1.0, 2_000.0, 1.0)
    assert math.isclose(d2, 2.0 * d1, rel_tol=1e-9)


def test_dose_per_unit_length_inversely_proportional_to_speed():
    d1 = dose_per_unit_length_proxy(1.0, 1_000.0, 1.0)
    d2 = dose_per_unit_length_proxy(1.0, 1_000.0, 2.0)
    assert math.isclose(d2, 0.5 * d1, rel_tol=1e-9)


def test_dose_per_unit_length_zero_energy_raises():
    with pytest.raises(ValueError):
        dose_per_unit_length_proxy(0.0, 1_000.0, 1.0)


def test_dose_per_unit_length_zero_speed_raises():
    with pytest.raises(ValueError):
        dose_per_unit_length_proxy(1.0, 1_000.0, 0.0)


def test_dose_per_unit_length_zero_rep_rate_raises():
    with pytest.raises(ValueError):
        dose_per_unit_length_proxy(1.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# static_exposure_total_energy_uJ
# ---------------------------------------------------------------------------


def test_static_exposure_basic():
    assert math.isclose(static_exposure_total_energy_uJ(2.0, 100), 200.0, rel_tol=1e-9)


def test_static_exposure_zero_pulses():
    assert static_exposure_total_energy_uJ(5.0, 0) == 0.0


def test_static_exposure_single_pulse():
    assert math.isclose(static_exposure_total_energy_uJ(3.7, 1), 3.7, rel_tol=1e-9)


def test_static_exposure_negative_energy_raises():
    with pytest.raises(ValueError):
        static_exposure_total_energy_uJ(-1.0, 10)


def test_static_exposure_negative_pulses_raises():
    with pytest.raises(ValueError):
        static_exposure_total_energy_uJ(1.0, -5)


# ---------------------------------------------------------------------------
# line_exposure_summary
# ---------------------------------------------------------------------------


def test_line_summary_pulse_count():
    # 500 µm line, 1 mm/s, 1 kHz → t=0.5 s, N=500
    summary = line_exposure_summary(1.0, 1_000.0, 1.0, 500.0, 3.0)
    assert summary["total_pulses_on_line"] == 500


def test_line_summary_total_energy():
    summary = line_exposure_summary(2.0, 1_000.0, 1.0, 500.0, 3.0)
    # 2 µJ × 500 pulses = 1000 µJ
    assert math.isclose(summary["total_energy_on_line_uJ"], 1000.0, rel_tol=1e-6)


def test_line_summary_pulse_spacing():
    # 1 mm/s, 1 kHz → spacing = 1.0 µm
    summary = line_exposure_summary(1.0, 1_000.0, 1.0, 500.0, 3.0)
    assert math.isclose(summary["pulse_spacing_um"], 1.0, rel_tol=1e-9)


def test_line_summary_pulses_per_spot():
    # d=3 µm, v=1 mm/s, f=1 kHz → N_eff = 3.0
    summary = line_exposure_summary(1.0, 1_000.0, 1.0, 500.0, 3.0)
    assert math.isclose(summary["pulses_per_spot"], 3.0, rel_tol=1e-9)


def test_line_summary_model_status():
    summary = line_exposure_summary(1.0, 1_000.0, 1.0, 500.0, 3.0)
    assert summary["model_status"] == "exposure_bookkeeping"


def test_line_summary_final_export_false():
    summary = line_exposure_summary(1.0, 1_000.0, 1.0, 500.0, 3.0)
    assert summary["final_export_allowed"] is False


def test_line_summary_discontinuous_warning():
    # High scan speed: spacing > diameter → dotted track warning
    summary = line_exposure_summary(1.0, 1_000.0, 100.0, 500.0, 3.0)
    # spacing = 100_000 µm/s / 1000 = 100 µm; d = 3 µm → warning expected
    assert any("gap" in w.lower() or "discontinu" in w.lower() for w in summary["warnings"])


def test_line_summary_fractional_pulses_per_spot_warning():
    # N_eff < 1 → dotted regime warning
    summary = line_exposure_summary(1.0, 1_000.0, 100.0, 500.0, 3.0)
    # N_eff = 3 * 1000 / 100_000 = 0.03 — very dotted
    assert any("dotted" in w.lower() or "discontinu" in w.lower() for w in summary["warnings"])


def test_line_summary_zero_speed_raises():
    with pytest.raises(ValueError):
        line_exposure_summary(1.0, 1_000.0, 0.0, 500.0, 3.0)


def test_line_summary_zero_rep_rate_raises():
    with pytest.raises(ValueError):
        line_exposure_summary(1.0, 0.0, 1.0, 500.0, 3.0)


def test_line_summary_zero_length_raises():
    with pytest.raises(ValueError):
        line_exposure_summary(1.0, 1_000.0, 1.0, 0.0, 3.0)


def test_line_summary_overlap_fraction():
    # spacing = 1 µm, d = 3 µm → overlap = 1 - 1/3 = 0.667
    summary = line_exposure_summary(1.0, 1_000.0, 1.0, 500.0, 3.0)
    assert math.isclose(summary["overlap_fraction"], 1.0 - 1.0 / 3.0, rel_tol=1e-6)
