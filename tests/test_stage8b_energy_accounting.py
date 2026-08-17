"""
Stage 8B energy accounting tests.

Covers: average power, sequential energy ledger, unit conversions,
fraction validation, Fresnel transmission, fluence scaling (area and array modes),
peak intensity estimates.
"""

import math
import numpy as np
import pytest

from vbb_study.digital_twin.energy_accounting import (
    average_power_w,
    validate_fraction,
    compute_energy_ledger,
    fresnel_normal_incidence_transmission,
    energy_fraction_after_components,
    fluence_from_effective_area_j_cm2,
    scale_intensity_to_fluence_j_cm2,
    peak_fluence_j_cm2,
    peak_intensity_w_cm2,
    OpticalComponent,
    default_holographic_chain,
)


# ---------------------------------------------------------------------------
# average_power_w
# ---------------------------------------------------------------------------


def test_average_power_correct_units():
    # 1 µJ at 1 kHz → 1e-6 J * 1e3 Hz = 1e-3 W
    result = average_power_w(1.0, 1_000.0)
    assert math.isclose(result, 1e-3, rel_tol=1e-9)


def test_average_power_50kHz():
    # 10 µJ at 50 kHz → 0.5 W
    assert math.isclose(average_power_w(10.0, 50_000.0), 0.5, rel_tol=1e-9)


def test_average_power_zero_energy():
    assert average_power_w(0.0, 1_000.0) == 0.0


def test_average_power_negative_energy_raises():
    with pytest.raises(ValueError):
        average_power_w(-1.0, 1_000.0)


def test_average_power_zero_rep_rate_raises():
    with pytest.raises(ValueError):
        average_power_w(10.0, 0.0)


# ---------------------------------------------------------------------------
# validate_fraction
# ---------------------------------------------------------------------------


def test_validate_fraction_valid():
    assert validate_fraction(0.5, "test") == 0.5
    assert validate_fraction(0.0, "test") == 0.0
    assert validate_fraction(1.0, "test") == 1.0


def test_validate_fraction_negative_raises():
    with pytest.raises(ValueError, match="must be in"):
        validate_fraction(-0.01, "bad_fraction")


def test_validate_fraction_above_one_raises():
    with pytest.raises(ValueError, match="must be in"):
        validate_fraction(1.001, "too_high")


# ---------------------------------------------------------------------------
# fresnel_normal_incidence_transmission
# ---------------------------------------------------------------------------


def test_fresnel_air_to_znse():
    # n1=1 (air), n2=2.4 (ZnSe)
    T = fresnel_normal_incidence_transmission(1.0, 2.4)
    R = ((1.0 - 2.4) / (1.0 + 2.4)) ** 2
    assert math.isclose(T, 1.0 - R, rel_tol=1e-9)
    assert 0.0 < T < 1.0


def test_fresnel_matched_index():
    # n1 == n2 → T = 1.0 (no reflection)
    T = fresnel_normal_incidence_transmission(1.5, 1.5)
    assert math.isclose(T, 1.0, rel_tol=1e-9)


def test_fresnel_zero_n_raises():
    with pytest.raises(ValueError):
        fresnel_normal_incidence_transmission(0.0, 1.5)


def test_fresnel_negative_n_raises():
    with pytest.raises(ValueError):
        fresnel_normal_incidence_transmission(1.5, -1.0)


def test_fresnel_symmetric():
    # T(n1→n2) == T(n2→n1) (power reflectance is symmetric)
    assert math.isclose(
        fresnel_normal_incidence_transmission(1.0, 1.5),
        fresnel_normal_incidence_transmission(1.5, 1.0),
        rel_tol=1e-9,
    )


# ---------------------------------------------------------------------------
# energy_fraction_after_components
# ---------------------------------------------------------------------------


def test_energy_fraction_two_components():
    comps = [
        OpticalComponent("A", "optics", 0.8),
        OpticalComponent("B", "optics", 0.5),
    ]
    assert math.isclose(energy_fraction_after_components(comps), 0.4, rel_tol=1e-9)


def test_energy_fraction_disabled_component_passthrough():
    comps = [
        OpticalComponent("A", "optics", 0.8),
        OpticalComponent("B", "optics", 0.5, enabled=False),
    ]
    # Disabled component passes energy through; total = 0.8 × 1.0 = 0.8
    assert math.isclose(energy_fraction_after_components(comps), 0.8, rel_tol=1e-9)


def test_energy_fraction_empty_chain():
    assert energy_fraction_after_components([]) == 1.0


# ---------------------------------------------------------------------------
# compute_energy_ledger
# ---------------------------------------------------------------------------


def test_ledger_sequential_multiplication():
    # 100 µJ through 0.75 * 0.73 * 0.90 * 0.85 * 0.95
    chain = default_holographic_chain(
        slm_diffraction_efficiency=0.75,
        selected_first_order_fraction=0.73,
        relay_transmission=0.90,
        objective_transmission=0.85,
        sample_interface_transmission=0.95,
    )
    ledger = compute_energy_ledger(100.0, 50_000.0, chain)
    expected = 100.0 * 0.75 * 0.73 * 0.90 * 0.85 * 0.95
    assert math.isclose(ledger.energy_at_sample_uJ, expected, rel_tol=1e-6)


def test_ledger_cumulative_fraction_correct():
    chain = [
        OpticalComponent("A", "optics", 0.5),
        OpticalComponent("B", "optics", 0.5),
    ]
    ledger = compute_energy_ledger(100.0, 1_000.0, chain)
    assert math.isclose(ledger.total_throughput_fraction, 0.25, rel_tol=1e-9)
    assert math.isclose(ledger.energy_at_sample_uJ, 25.0, rel_tol=1e-9)


def test_ledger_average_power_at_sample():
    chain = [OpticalComponent("A", "optics", 1.0)]
    # 10 µJ at 50 kHz → 0.5 W
    ledger = compute_energy_ledger(10.0, 50_000.0, chain)
    assert math.isclose(ledger.average_power_at_sample_W, 0.5, rel_tol=1e-6)


def test_ledger_power_limit_warning():
    chain = [OpticalComponent("A", "optics", 1.0)]
    # 200 µJ at 50 kHz = 10 W; limit = 5 W → should warn
    ledger = compute_energy_ledger(200.0, 50_000.0, chain, average_power_limit_W=5.0)
    assert any("exceeds limit" in w for w in ledger.ledger_warnings)


def test_ledger_no_power_warning_when_within_limit():
    chain = [OpticalComponent("A", "optics", 0.1)]
    ledger = compute_energy_ledger(10.0, 1_000.0, chain, average_power_limit_W=100.0)
    assert not any("exceeds limit" in w for w in ledger.ledger_warnings)


def test_ledger_invalid_energy_raises():
    with pytest.raises(ValueError):
        compute_energy_ledger(-1.0, 50_000.0, [])


def test_ledger_invalid_rep_rate_raises():
    with pytest.raises(ValueError):
        compute_energy_ledger(10.0, 0.0, [])


def test_ledger_zero_throughput_warning():
    chain = [OpticalComponent("A", "optics", 0.0)]
    ledger = compute_energy_ledger(10.0, 1_000.0, chain)
    assert any("zero" in w.lower() or "no energy" in w.lower() for w in ledger.ledger_warnings)


def test_ledger_model_status():
    ledger = compute_energy_ledger(10.0, 1_000.0, [])
    assert ledger.model_status == "energy_accounting_prediction"


def test_ledger_final_export_false():
    ledger = compute_energy_ledger(10.0, 1_000.0, [])
    assert ledger.final_export_allowed is False


def test_ledger_row_count_matches_components():
    chain = default_holographic_chain()
    ledger = compute_energy_ledger(100.0, 50_000.0, chain)
    assert len(ledger.rows) == len(chain)


def test_invalid_transmission_raises_on_construction():
    with pytest.raises(ValueError):
        OpticalComponent("bad", "optics", transmission_fraction=1.5)


def test_invalid_transmission_negative_raises():
    with pytest.raises(ValueError):
        OpticalComponent("bad", "optics", transmission_fraction=-0.1)


# ---------------------------------------------------------------------------
# fluence_from_effective_area_j_cm2
# ---------------------------------------------------------------------------


def test_fluence_effective_area_basic():
    # 1 µJ in 100 µm² → 1e-6 J / (100 × 1e-8 cm²) = 1e-6 / 1e-6 = 1.0 J/cm²
    result = fluence_from_effective_area_j_cm2(1.0, 100.0)
    assert math.isclose(result, 1.0, rel_tol=1e-9)


def test_fluence_effective_area_zero_energy():
    assert fluence_from_effective_area_j_cm2(0.0, 100.0) == 0.0


def test_fluence_effective_area_zero_area_raises():
    with pytest.raises(ValueError):
        fluence_from_effective_area_j_cm2(1.0, 0.0)


def test_fluence_effective_area_negative_area_raises():
    with pytest.raises(ValueError):
        fluence_from_effective_area_j_cm2(1.0, -1.0)


def test_fluence_effective_area_negative_energy_raises():
    with pytest.raises(ValueError):
        fluence_from_effective_area_j_cm2(-1.0, 100.0)


# ---------------------------------------------------------------------------
# scale_intensity_to_fluence_j_cm2
# ---------------------------------------------------------------------------


def test_scale_intensity_conserves_energy():
    # Flat intensity array: 10×10, dx=dy=1 µm, 2 µJ
    I = np.ones((10, 10))
    F = scale_intensity_to_fluence_j_cm2(I, 1.0, 1.0, 2.0)
    # Total fluence × area should equal pulse energy
    total_energy_J = np.sum(F) * (1.0 * 1.0 * 1e-8)  # F in J/cm², dA in cm²
    E_J = 2.0 * 1e-6
    assert math.isclose(total_energy_J, E_J, rel_tol=1e-6)


def test_scale_intensity_peak_correct():
    # Uniform 1×1 array, 1 µm × 1 µm, 1 µJ
    I = np.ones((1, 1))
    F = scale_intensity_to_fluence_j_cm2(I, 1.0, 1.0, 1.0)
    # 1 µJ / (1 µm²) = 1e-6 J / (1e-8 cm²) = 100 J/cm²
    assert math.isclose(float(F[0, 0]), 100.0, rel_tol=1e-6)


def test_scale_intensity_zero_array_raises():
    with pytest.raises(ValueError, match="zero"):
        scale_intensity_to_fluence_j_cm2(np.zeros((5, 5)), 1.0, 1.0, 1.0)


def test_scale_intensity_negative_values_raise():
    I = np.ones((5, 5))
    I[2, 2] = -0.01
    with pytest.raises(ValueError, match="negative"):
        scale_intensity_to_fluence_j_cm2(I, 1.0, 1.0, 1.0)


def test_scale_intensity_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        scale_intensity_to_fluence_j_cm2(np.array([]), 1.0, 1.0, 1.0)


def test_scale_intensity_zero_dx_raises():
    with pytest.raises(ValueError):
        scale_intensity_to_fluence_j_cm2(np.ones((5, 5)), 0.0, 1.0, 1.0)


def test_scale_intensity_zero_dy_raises():
    with pytest.raises(ValueError):
        scale_intensity_to_fluence_j_cm2(np.ones((5, 5)), 1.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# peak_fluence_j_cm2
# ---------------------------------------------------------------------------


def test_peak_fluence_simple():
    F = np.array([[1.0, 2.0], [3.0, 0.5]])
    assert math.isclose(peak_fluence_j_cm2(F), 3.0, rel_tol=1e-9)


def test_peak_fluence_empty_raises():
    with pytest.raises(ValueError):
        peak_fluence_j_cm2(np.array([]))


# ---------------------------------------------------------------------------
# peak_intensity_w_cm2
# ---------------------------------------------------------------------------


def test_peak_intensity_flat_top():
    # F = 1 J/cm², τ = 1e15 fs = 1 s → I = 1 W/cm²
    result = peak_intensity_w_cm2(1.0, 1e15, "flat_top_equivalent")
    assert math.isclose(result.peak_intensity_w_cm2, 1.0, rel_tol=1e-6)


def test_peak_intensity_100fs():
    # 1 J/cm², 100 fs → 1 / 100e-15 = 1e13 W/cm²
    result = peak_intensity_w_cm2(1.0, 100.0, "flat_top_equivalent")
    assert math.isclose(result.peak_intensity_w_cm2, 1e13, rel_tol=1e-6)


def test_peak_intensity_gaussian_less_than_flat():
    # For same FWHM duration, Gaussian peak = 0.94 × flat-top (factor sqrt(4 ln2/π) < 1).
    # Flat-top is the conservative (higher) estimate, as documented.
    flat = peak_intensity_w_cm2(1.0, 100.0, "flat_top_equivalent")
    gauss = peak_intensity_w_cm2(1.0, 100.0, "gaussian")
    assert gauss.peak_intensity_w_cm2 < flat.peak_intensity_w_cm2
    # Factor should be sqrt(4 ln2/π) ≈ 0.9394
    import math
    expected_ratio = math.sqrt(4.0 * math.log(2.0) / math.pi)
    actual_ratio = gauss.peak_intensity_w_cm2 / flat.peak_intensity_w_cm2
    assert math.isclose(actual_ratio, expected_ratio, rel_tol=1e-6)


def test_peak_intensity_approximation_note_present():
    result = peak_intensity_w_cm2(1.0, 100.0)
    assert "nonlinear" in result.approximation_note.lower()
    assert "plasma" in result.approximation_note.lower() or \
           "thermal" in result.approximation_note.lower()


def test_peak_intensity_model_status():
    result = peak_intensity_w_cm2(1.0, 100.0)
    assert result.model_status == "energy_accounting_prediction"


def test_peak_intensity_negative_fluence_raises():
    with pytest.raises(ValueError):
        peak_intensity_w_cm2(-1.0, 100.0)


def test_peak_intensity_zero_duration_raises():
    with pytest.raises(ValueError):
        peak_intensity_w_cm2(1.0, 0.0)
