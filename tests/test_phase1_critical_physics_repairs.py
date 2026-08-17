from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

import bessel_twin_core as bt
from vbb_study.equations.fields import make_xy_grid
from vbb_study.equations.objective_pupil import (
    fourier_plane_carrier_separation_m,
    fourier_plane_ring_radius_m,
)
from vbb_study.design import J0_FIRST_ZERO, compute_design_from_config, default_config, objective_map_from_config


WAVELENGTH_M = 1029e-9
FOCAL_LENGTH_M = 0.300


def test_fourier_carrier_displacement_includes_wavelength() -> None:
    displacement = fourier_plane_carrier_separation_m(6.25e3, FOCAL_LENGTH_M, WAVELENGTH_M)
    assert displacement == pytest.approx(1.929375e-3, rel=1e-12)


def test_fourier_ring_radius_matches_k0_and_lambda_forms() -> None:
    kr_m_inv = 2.35e5
    k0_m_inv = 2.0 * math.pi / WAVELENGTH_M
    radius = fourier_plane_ring_radius_m(kr_m_inv, FOCAL_LENGTH_M, WAVELENGTH_M)
    assert radius == pytest.approx(FOCAL_LENGTH_M * kr_m_inv / k0_m_inv, rel=1e-14)


@pytest.mark.parametrize(
    ("helper", "frequency"),
    (
        (fourier_plane_carrier_separation_m, 0.0),
        (fourier_plane_ring_radius_m, 0.0),
    ),
)
def test_zero_spatial_frequency_has_zero_fourier_displacement(helper, frequency: float) -> None:
    assert helper(frequency, FOCAL_LENGTH_M, WAVELENGTH_M) == 0.0


@pytest.mark.parametrize(
    ("helper", "frequency"),
    (
        (fourier_plane_carrier_separation_m, 6.25e3),
        (fourier_plane_ring_radius_m, 2.35e5),
    ),
)
def test_doubling_wavelength_doubles_physical_fourier_distance(helper, frequency: float) -> None:
    first = helper(frequency, FOCAL_LENGTH_M, WAVELENGTH_M)
    second = helper(frequency, FOCAL_LENGTH_M, 2.0 * WAVELENGTH_M)
    assert second == pytest.approx(2.0 * first, rel=1e-14)


def _propagation_volume(*, clipping: bool) -> dict:
    grid = make_xy_grid(128, 0.5e-6)
    amplitude = np.exp(-((grid["R"] / 12e-6) ** 2))
    if clipping:
        radial_cpm = 0.40 / float(grid["dx"])
        field = amplitude * np.exp(1j * 2.0 * math.pi * radial_cpm * grid["R"])
        z_values = np.asarray([0.0, 100e-6, 200e-6])
    else:
        field = amplitude
        z_values = np.asarray([0.0, 2e-6, 4e-6])
    return bt.propagate_volume(
        field,
        grid,
        WAVELENGTH_M,
        z_values,
        crop_pixels=96,
        bandlimit=True,
    )


def test_well_sampled_propagation_is_quantitatively_valid() -> None:
    volume = _propagation_volume(clipping=False)
    assert volume["quantitative_metrics_valid"] is True
    assert volume["propagation_power_label"] == "pass"
    assert float(volume["propagation_power_drift_fraction"]) <= 0.05


def test_clipped_propagation_is_not_quantitatively_valid() -> None:
    volume = _propagation_volume(clipping=True)
    assert volume["quantitative_metrics_valid"] is False
    assert float(volume["propagation_power_drift_fraction"]) > 0.05
    assert "exceeds the quantitative limit" in volume["quantitative_metrics_invalid_reason"]


def test_power_validity_flag_mode_returns_invalid_report() -> None:
    report = bt.enforce_propagation_power_validity(_propagation_volume(clipping=True), "flag")
    assert report["quantitative_metrics_valid"] is False
    assert report["validity_action"] == "flag"


def test_power_validity_warn_mode_warns_and_invalidates() -> None:
    with pytest.warns(RuntimeWarning, match="Propagation quantitative validity failed"):
        report = bt.enforce_propagation_power_validity(_propagation_volume(clipping=True), "warn")
    assert report["quantitative_metrics_valid"] is False
    assert report["validity_action"] == "warn"


def test_power_validity_raise_mode_blocks_quantitative_interpretation() -> None:
    with pytest.raises(ValueError, match="Propagation quantitative validity failed"):
        bt.enforce_propagation_power_validity(_propagation_volume(clipping=True), "raise")


def test_expected_first_order_loss_is_not_numerical_power_drift() -> None:
    volume = _propagation_volume(clipping=False)
    volume["first_order_selected_fraction"] = 0.10
    report = bt.enforce_propagation_power_validity(volume, "flag")
    assert report["quantitative_metrics_valid"] is True


def test_absent_power_drift_is_not_silently_reported_as_zero() -> None:
    report = bt.propagation_power_validity_report({})
    assert report["quantitative_metrics_valid"] is False
    assert report["propagation_power_drift_evaluated"] is False
    assert math.isnan(float(report["propagation_power_drift_fraction"]))
    assert report["propagation_power_label"] == "not_evaluated"


def test_target_matched_mapping_changes_with_requested_target_length() -> None:
    base = default_config("fast")
    short = compute_design_from_config(base)
    long_cfg = replace(
        base,
        target=replace(base.target, target_bessel_length_m=2.0 * base.target.target_bessel_length_m),
    )
    long = compute_design_from_config(long_cfg)
    assert short.mapping_mode == "target_matched_inverse_design"
    assert long.objective_map_demag == pytest.approx(2.0 * short.objective_map_demag)


def test_fixed_physical_mapping_does_not_retune_with_target_length() -> None:
    base = replace(default_config("fast"), mapping_mode="fixed_physical_optics")
    short = compute_design_from_config(base)
    long_cfg = replace(
        base,
        target=replace(base.target, target_bessel_length_m=2.0 * base.target.target_bessel_length_m),
    )
    long = compute_design_from_config(long_cfg)
    assert short.mapping_mode == "fixed_physical_optics"
    assert long.objective_map_demag == pytest.approx(short.objective_map_demag, rel=0.0, abs=0.0)
    assert long.predicted_bessel_length_m == pytest.approx(short.predicted_bessel_length_m)


@pytest.mark.parametrize(
    "mapping_mode",
    ("target_matched_inverse_design", "fixed_physical_optics"),
)
def test_mapping_modes_report_mode_source_and_demag(mapping_mode: str) -> None:
    cfg = replace(default_config("fast"), mapping_mode=mapping_mode)
    design = compute_design_from_config(cfg)
    objective_map = objective_map_from_config(cfg, design)
    assert design.mapping_mode == mapping_mode
    assert objective_map.mapping_mode == mapping_mode
    assert design.objective_map_source == objective_map.source
    assert design.objective_map_demag == pytest.approx(objective_map.demag)


def test_fixed_optics_does_not_claim_requested_target_by_construction() -> None:
    base = replace(default_config("fast"), mapping_mode="fixed_physical_optics")
    mismatched = replace(
        base,
        target=replace(base.target, target_bessel_length_m=2.0 * base.target.target_bessel_length_m),
    )
    report = bt.inverse_design_round_trip(mismatched)
    assert report["claim_scope"] == "fixed_bench_prediction"
    assert report["hardware_target_achieved"] is False
    assert report["length_relative_error"] > 0.03


def test_inverse_design_uses_exact_j0_first_zero_reference() -> None:
    cfg = default_config("fast")
    design = compute_design_from_config(cfg)
    kr_sample = 2.0 * J0_FIRST_ZERO / cfg.target.target_core_diameter_m
    expected_w0 = cfg.target.target_bessel_length_m * kr_sample / (
        cfg.laser.k0 * cfg.material.refractive_index
    )
    expected_demag = expected_w0 / cfg.laser.beam_radius_on_slm_m
    assert cfg.mapping_mode == "target_matched_inverse_design"
    assert design.kr_sample_m_inv == pytest.approx(kr_sample, rel=1e-14)
    assert design.w0_sample_m == pytest.approx(expected_w0, rel=1e-14)
    assert design.objective_map_demag == pytest.approx(expected_demag, rel=1e-14)
