from __future__ import annotations

import numpy as np
import pytest

from vbb_study.digital_twin import (
    audit_mode1c_aperture_ring_limit,
    mode1d_achievable_ring_count_table,
    mode1d_design_context,
    mode1d_outcome_report,
    mode1d_required_na_table,
    mode1d_scope_manifest,
    mode1d_source_sweep_row,
    required_na,
    required_pre_kr_for_ring_count,
    required_radius_for_ring_count,
    run_mode1d_source_ring_count_sweep,
)


def test_required_pre_kr_for_ring_count_matches_formula() -> None:
    rings = 12.0
    radius = 2.5e-3

    assert required_pre_kr_for_ring_count(rings, radius) == pytest.approx(2.0 * np.pi * rings / radius)


def test_required_na_increases_with_ring_count() -> None:
    ctx = mode1d_design_context()
    radius = ctx.current_p2_radius_m
    low = required_na(required_pre_kr_for_ring_count(8.0, radius) * ctx.current_mapping_factor, ctx.wavelength_m)
    high = required_na(required_pre_kr_for_ring_count(31.0, radius) * ctx.current_mapping_factor, ctx.wavelength_m)

    assert high > low


def test_required_radius_decreases_with_larger_pre_kr() -> None:
    rings = 31.0

    assert required_radius_for_ring_count(rings, 80_000.0) < required_radius_for_ring_count(rings, 40_000.0)


def test_current_p2_radius_requires_na_greater_than_one_for_v0() -> None:
    row = mode1d_required_na_table()[0]

    assert row["radius_case"] == "current_p2_radius"
    assert row["required_NA_for_V0_ring_count"] > 1.0


def test_slm_safe_radius_still_does_not_reach_v0_with_na_0p45() -> None:
    ctx = mode1d_design_context()
    rows = {row["case"]: row for row in mode1d_achievable_ring_count_table(ctx)}

    assert rows["current_NA_slm_safe_radius"]["ring_count"] < ctx.v0_ring_count
    assert rows["current_NA_slm_safe_radius"]["NA"] == pytest.approx(0.45)


def test_achievable_ring_count_table_matches_mode1c_budget_values() -> None:
    ctx = mode1d_design_context()
    limit = audit_mode1c_aperture_ring_limit()
    rows = {row["case"]: row for row in mode1d_achievable_ring_count_table(ctx)}

    assert rows["current_NA_current_radius"]["ring_count"] == pytest.approx(limit.ring_count_max_current_radius_na_limited)
    assert rows["current_NA_slm_safe_radius"]["ring_count"] == pytest.approx(limit.ring_count_max_slm_radius_na_limited)
    assert ctx.mode1c_current_ring_count == pytest.approx(limit.ring_count_current)


def test_lower_ring_source_sweep_produces_stable_schema() -> None:
    cases = run_mode1d_source_ring_count_sweep((12.0,), grid_n=384, z_planes=9)
    row = mode1d_source_sweep_row(cases[0])
    required = {
        "ring_count_target",
        "ring_count_actual",
        "classification",
        "accepted_hexagon",
        "dark_core_ratio",
        "template_angular_correlation",
        "template_xy_correlation",
        "c60",
        "c120",
        "sector_balance_max_over_min",
    }

    assert required <= set(row)
    assert row["ring_count_actual"] == pytest.approx(12.0)
    assert row["classification"] in {"triangular_lobed_field", "dark_core_structured_field", "visual_hexagonal_field"}


def test_mode2a_2b_remains_blocked_after_mode1d_until_redesigned_mode1_confirmation() -> None:
    ctx = mode1d_design_context()
    cases = run_mode1d_source_ring_count_sweep((12.0,), grid_n=384, z_planes=9)
    report = mode1d_outcome_report(ctx, cases, mode1d_achievable_ring_count_table(ctx))
    manifest = mode1d_scope_manifest(report)

    assert report["suggested_outcome"] == "M1D-A"
    assert report["mode2a_2b_realisation_allowed"] is False
    assert report["mode2a_2b_gate"] == "blocked_pending_redesigned_mode1_confirmation"
    assert manifest["physical_route_approval"] is False
