from __future__ import annotations

from vbb_study.digital_twin import (
    Mode1CConstrainedCandidate,
    audit_mode1c_aperture_ring_limit,
    audit_mode1c_kr_mapping,
    mode1c_candidate_row,
    mode1c_outcome_report,
    run_mode1c_constrained_search,
)


def test_mode1c_kr_mapping_positive_and_finite() -> None:
    mapping = audit_mode1c_kr_mapping()
    assert mapping.k0_m_inv > 0.0
    assert mapping.k_r_pre_m_inv > 0.0
    assert mapping.k_r_surface_m_inv > 0.0
    assert mapping.k_scale_surface_over_pre > 1.0
    assert 0.0 < mapping.current_surface_na_fraction < 1.0


def test_mode1c_aperture_ring_limit_reports_required_radii_and_v0() -> None:
    limit = audit_mode1c_aperture_ring_limit()
    assert limit.p2_radius_current_m > 0.0
    assert limit.p2_radius_max_with_safety_m > limit.p2_radius_current_m
    assert limit.v0_ring_count > 20.0
    assert limit.ring_count_max_slm_radius_na_limited > limit.ring_count_current


def test_current_p2_ring_count_is_around_4_not_old_mixed_0033() -> None:
    limit = audit_mode1c_aperture_ring_limit()
    assert 3.5 < limit.ring_count_current < 4.7
    assert limit.ring_count_current > 100.0 * 0.033


def test_constrained_candidate_rejects_surface_na_excess() -> None:
    candidates = run_mode1c_constrained_search(
        target_ring_counts=(31.0,),
        p2_radius_options_m=(audit_mode1c_aperture_ring_limit().p2_radius_current_m,),
        run_proxy=False,
    )
    cand = candidates[0]
    assert not cand.within_objective_na
    assert cand.run_status == "infeasible_not_run"
    assert "surface NA exceeds objective NA" in cand.failure_reason


def test_constrained_candidate_reports_slm_phase_period_limit() -> None:
    candidates = run_mode1c_constrained_search(
        target_ring_counts=(1000.0,),
        p2_radius_options_m=(1.0e-3,),
        run_proxy=False,
    )
    cand = candidates[0]
    assert not cand.slm_encoded_phase_ok
    assert any("phase period" in reason for reason in cand.failure_reason)


def test_mode1c_candidates_carry_clear_run_status() -> None:
    candidates = run_mode1c_constrained_search(
        target_ring_counts=(4.0,),
        p2_radius_options_m=(audit_mode1c_aperture_ring_limit().p2_radius_current_m,),
        run_proxy=True,
        grid_n_proxy=96,
        z_planes_proxy=5,
    )
    row = mode1c_candidate_row(candidates[0])
    assert row["run_status"] in {"proxy_only_run", "feasible_proxy_not_requested", "infeasible_not_run"}
    assert row["run_status"] == "proxy_only_run"


def test_mode2_blocked_unless_m1c_a() -> None:
    mapping = audit_mode1c_kr_mapping()
    limit = audit_mode1c_aperture_ring_limit()
    blocked = mode1c_outcome_report(mapping=mapping, aperture=limit, candidates=())
    assert blocked["suggested_outcome"] == "M1C-C"
    assert blocked["mode2_realisation_allowed"] is False

    actual_pass = Mode1CConstrainedCandidate(
        target_ring_count=4.0,
        p2_radius_m=limit.p2_radius_current_m,
        k_r_pre_m_inv=mapping.k_r_pre_m_inv,
        k_r_surface_m_inv=mapping.k_r_surface_m_inv,
        surface_na_required=mapping.current_surface_na_fraction,
        within_objective_na=True,
        within_slm_aperture=True,
        phase_period_m=mapping.current_pre_phase_period_m,
        slm_encoded_phase_ok=True,
        run_status="actual_inherited_downstream_run",
        symmetry_class="visual_hexagonal_field",
        template_gate_pass=True,
        failure_reason=(),
    )
    open_gate = mode1c_outcome_report(mapping=mapping, aperture=limit, candidates=(actual_pass,))
    assert open_gate["suggested_outcome"] == "M1C-A"
    assert open_gate["mode2_realisation_allowed"] is True
