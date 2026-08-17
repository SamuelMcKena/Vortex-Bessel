from __future__ import annotations

import math

import numpy as np

from vbb_study.digital_twin.phase2e_source_sampling_repair import (
    analytic_b0_source,
    axisymmetric_on_axis_trace,
    classify_sampling,
    optical_route_contract,
    sampling_diagnostic,
    sampling_table,
)


def test_fixed_window_resolution_sweep_reduces_dx() -> None:
    rows = sampling_table((512, 768, 1024, 1536), window_m=10.0e-3)
    assert {row["window_m"] for row in rows} == {10.0e-3}
    assert [row["dx_m"] for row in rows] == sorted(
        [row["dx_m"] for row in rows], reverse=True
    )
    assert [row["samples_per_radial_period"] for row in rows] == sorted(
        row["samples_per_radial_period"] for row in rows
    )


def test_n512_is_marginal_not_quantitative() -> None:
    result = sampling_diagnostic(512)
    assert 3.0 < result.samples_per_radial_period < 4.0
    assert result.sampling_class == "invalid"
    assert result.quantitative_reference is False
    assert result.adjacent_radial_phase_increment_rad > math.pi / 2.0


def test_quantitative_class_requires_twelve_samples_per_period() -> None:
    assert classify_sampling(3.99) == "invalid"
    assert classify_sampling(4.0) == "marginal"
    assert classify_sampling(8.0) == "acceptable_for_screening"
    assert classify_sampling(12.0) == "quantitative_reference"


def test_route_contract_separates_source_and_focal_scales() -> None:
    contract = optical_route_contract()
    source = contract["route_S_source_scale"]
    focal = contract["route_F_objective_sample_scale"]
    assert source["objective_transform"] == "none"
    assert focal["objective_transform"] == "required"
    assert "objective-focused sample-plane dimensions" in source["blocked_claims"]
    assert source["aperture_semantics"].startswith("nominal_hard_aperture_placeholder")


def test_analytic_source_applies_one_axicon_phase_and_declared_aperture() -> None:
    field, grid, metadata = analytic_b0_source(256, aperture_model="hard")
    assert field.shape == (256, 256)
    assert grid["N"] == 256
    assert metadata["route_id"] == "route_S_source_scale"
    assert metadata["aperture_model"] == "hard"
    assert np.iscomplexobj(field)
    assert np.all(np.isfinite(field))


def test_axisymmetric_reference_is_independent_of_2d_bl_asm() -> None:
    z = np.linspace(1e-3, 20e-3, 16)
    result = axisymmetric_on_axis_trace(z, radial_samples=2048, aperture_model="soft")
    assert result.intensity.shape == z.shape
    assert np.all(np.isfinite(result.intensity))
    assert np.all(result.intensity >= 0.0)
    assert result.metadata["not_called"] == "2D BL-ASM"


def test_aperture_models_change_axisymmetric_trace() -> None:
    z = np.linspace(10e-3, 120e-3, 80)
    hard = axisymmetric_on_axis_trace(z, radial_samples=4096, aperture_model="hard")
    soft = axisymmetric_on_axis_trace(z, radial_samples=4096, aperture_model="soft")
    h = hard.intensity / np.max(hard.intensity)
    s = soft.intensity / np.max(soft.intensity)
    assert np.linalg.norm(h - s) > 1e-3
