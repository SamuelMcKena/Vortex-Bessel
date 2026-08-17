"""Stage 8C.3R.5.2 editable hardware/geometry control-contract tests."""

import json
from dataclasses import fields

import numpy as np

from vbb_study.digital_twin.cslm_route import CSLMRouteConfig, run_cslm_baseline_route
from vbb_study.digital_twin.control_contract import (
    VALID_STATUSES,
    VALID_PROVENANCE,
    NON_MEASURED_PROVENANCE,
    build_cslm_editable_control_registry,
    apply_editable_control_overrides,
    build_default_demo_profile,
    build_measured_bench_template,
    validate_hardware_profile,
    hardware_profile_completeness_report,
    save_hardware_profile,
    load_hardware_profile,
    config_from_profile,
    status_counts,
)


# 1. Every active R5/R5.1 config parameter appears in the editable registry.
def test_every_config_field_has_a_control():
    reg = build_cslm_editable_control_registry()
    mapped = {c.config_field for c in reg if c.config_field}
    config_field_names = {f.name for f in fields(CSLMRouteConfig)}
    missing = config_field_names - mapped
    assert not missing, f"config fields without a control: {missing}"


# 2. Every control reports a valid status and provenance value.
def test_all_controls_valid_status_and_provenance():
    for c in build_cslm_editable_control_registry():
        assert c.status in VALID_STATUSES, c.control_id
        assert c.provenance in VALID_PROVENANCE, c.control_id


# 3. Demo profile round-trips through save/load without changing baseline fields.
def test_demo_profile_round_trip(tmp_path):
    base = CSLMRouteConfig.fast()
    demo = build_default_demo_profile(base)
    p = tmp_path / "demo.json"
    save_hardware_profile(demo, p)
    loaded = load_hardware_profile(p)
    cfg = config_from_profile(loaded, CSLMRouteConfig.fast())
    assert cfg == base


# 4. Measured-bench template contains no invented values.
def test_measured_bench_template_has_no_invented_values():
    tmpl = build_measured_bench_template()
    assert tmpl["profile_status"] == "measured_bench_template_unfilled"
    assert validate_hardware_profile(tmpl) == []
    for cid, entry in tmpl["controls"].items():
        if entry["status"] == "derived_read_only":
            continue
        assert entry["value"] is None, cid
        assert entry["provenance"] == "unknown", cid


# 5. Required 4F controls remain warning-only and cannot set fourier physics True.
def test_4f_controls_warning_only_and_locked():
    reg = {c.control_id: c for c in build_cslm_editable_control_registry()}
    for cid in ("fourier_lens1_focal_length_mm", "slm2_to_lens1_distance_mm",
                "fourier_filter_radius_um", "lens2_to_output_plane_distance_mm"):
        assert reg[cid].status == "warning_only"
        assert reg[cid].affects_active_model is False
    base = CSLMRouteConfig()
    import pytest
    with pytest.raises(ValueError):
        apply_editable_control_overrides(base, {"fourier_filter_physics_available": True})


# 6. Axicon benchmark controls affect only the benchmark branch / not the active route.
def test_axicon_controls_are_benchmark_only():
    reg = {c.control_id: c for c in build_cslm_editable_control_registry()}
    for cid in ("physical_axicon_cone_parameter_rad_per_um",
                "physical_axicon_clear_aperture_radius_um",
                "physical_axicon_to_benchmark_reference_distance_mm"):
        assert reg[cid].status == "benchmark_only"
        assert reg[cid].affects_active_model is False
        assert reg[cid].affects_benchmark_branch is True
    # changing an axicon value does not flip the default active route
    base = CSLMRouteConfig.fast()
    changed = apply_editable_control_overrides(base, {"physical_axicon_cone_parameter_rad_per_um": 1.2})
    assert changed.order_handoff_mode == "none"
    run = run_cslm_baseline_route(changed)
    assert run.order_handoff.mode == "none"
    assert run.axicon_benchmark is None


# 7. SLM2 piston placeholder labelled accurately; no false correction-map claim.
def test_slm2_piston_labelled_accurately():
    reg = {c.control_id: c for c in build_cslm_editable_control_registry()}
    piston = reg["slm2_correction_phase_rad"]
    assert "piston placeholder" in piston.display_name.lower()
    # the description must explicitly disclaim being a correction map
    assert "not an aberration-correction map" in piston.description.lower()
    assert "map" not in piston.display_name.lower()
    spatial = reg["spatial_correction_map_source"]
    assert spatial.status == "future_not_implemented"
    assert spatial.affects_active_model is False


# 8. Advanced numerical controls labelled numerical_advanced.
def test_advanced_numerical_labelled():
    reg = {c.control_id: c for c in build_cslm_editable_control_registry()}
    for cid in ("grid_N", "dx_um", "n_z", "bandlimit"):
        assert reg[cid].status == "numerical_advanced"


# 9. Completeness report lists missing physical-4F, measured-bench, camera inputs.
def test_completeness_report_lists_missing():
    rep = hardware_profile_completeness_report()
    assert rep["active_cslm_diagnostic_branch"] == "complete"
    assert rep["ideal_axicon_benchmark_branch"] == "complete"
    assert rep["physical_4f_route"] == "blocked"
    assert rep["measured_lab_route"] == "blocked"
    assert rep["camera_comparison"] == "blocked"
    assert len(rep["measured_bench_missing"]) > 0
    assert len(rep["camera_comparison_missing"]) > 0
    assert len(rep["physical_4f_blocking_model_gaps"]) > 0
    assert rep["fourier_filter_physics_available"] is False


# 12. No material/camera-physics/4F-physics claims introduced; demo not measured.
def test_no_unsafe_claims_and_demo_not_measured():
    demo = build_default_demo_profile()
    assert demo["profile_status"] == "diagnostic_demo_not_measured_bench"
    # no control is presented as measured
    counts = {p: 0 for p in VALID_PROVENANCE}
    for c in build_cslm_editable_control_registry():
        counts[c.provenance] += 1
    assert counts["measured"] == 0
    blob = json.dumps(demo).lower()
    for token in ("dose", "thermal", "plasma", "nonlinear", "ablation", "camera_physics", "stage8d"):
        assert token not in blob


# Governance: demo profile loads into config without enabling unsafe governance.
def test_governance_locked():
    import pytest
    base = CSLMRouteConfig()
    with pytest.raises(ValueError):
        apply_editable_control_overrides(base, {"diagnostic_only": False})
    with pytest.raises(ValueError):
        apply_editable_control_overrides(base, {"final_export_allowed": True})


# status counts sum to the registry size.
def test_status_counts_sum():
    reg = build_cslm_editable_control_registry()
    assert sum(status_counts(reg).values()) == len(reg)
