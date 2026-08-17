"""Stage 8C.3R.1 free-space reference-plane validation tests.

Covers the thirteen C3R.1 acceptance requirements: zero-control equivalence with
the canonical free-space baseline, honest energy accounting (no hidden
renormalisation), analytical beam-tilt validation, independent vortex/axicon
offsets, beam decentre and tilt applied before propagation, zero-order leakage
core contamination, axis-tracking validity, crop/FOV reliability flags, the
diagnostic-sensitivity-sweep labelling, and the no-material-response boundary.
"""

import numpy as np
from dataclasses import replace

from vbb_study.digital_twin.component_plane_pipeline import (
    ComponentPlaneConfig,
    run_component_plane_pipeline,
)
from vbb_study.digital_twin.component_plane_metrics import (
    compute_axis_tracking,
    run_component_plane_scenario,
)
from vbb_study.digital_twin.component_plane_validation import (
    canonical_free_space_reference,
    zero_control_equivalence,
    compute_energy_audit,
    validate_beam_tilt,
    fov_convergence_check,
)
from vbb_study.digital_twin.component_plane_figures import (
    DEFAULT_ATLAS_SCENARIOS,
    _DIAG_NOTE,
)

CFG = ComponentPlaneConfig.fast()


# 1. Zero-control agrees with the canonical free-space baseline.
def test_zero_control_equivalent_to_canonical_free_space():
    eq = zero_control_equivalence(CFG)
    assert eq["equivalent_within_tolerance"] is True
    assert eq["complex_field_similarity"] >= 1.0 - 1e-9
    assert eq["intensity_similarity"] >= 1.0 - 1e-9
    assert eq["fluence_similarity"] >= 1.0 - 1e-9
    assert eq["raw_field_power_rel_diff"] <= 1e-9
    assert eq["physically_valid_bessel_gauss"] is True
    # canonical reference is genuinely free-space n=1.0
    ref = canonical_free_space_reference(CFG)
    assert ref["n_medium"] == 1.0


# 2. Passive clipping reduces total transmitted energy.
def test_passive_clipping_reduces_transmitted_energy():
    run = run_component_plane_pipeline(
        {"enable_pupil_clipping": True, "pupil_radius_um": 12.0}, config=CFG
    )
    st = run.propagated_stack
    assert st.transmitted_fraction < 0.95
    assert st.sample_pulse_energy_uJ < st.input_pulse_energy_uJ
    au = compute_energy_audit(run, fov_reliable=True)
    assert au["energy_accounting_valid"] is True
    # no component shows power gain
    for row in au["per_plane_ledger"]:
        assert row["transmitted_fraction"] <= 1.0 + 1e-9


# 3. No lost energy is silently restored.
def test_no_energy_restoration():
    base = run_component_plane_pipeline({}, config=CFG)
    run = run_component_plane_pipeline(
        {"enable_pupil_clipping": True, "pupil_radius_um": 10.0}, config=CFG
    )
    au = compute_energy_audit(run, baseline_run=base, fov_reliable=True)
    st = run.propagated_stack
    assert au["renormalisation_factor"] == 1.0
    assert abs(st.sample_pulse_energy_uJ - st.input_pulse_energy_uJ * st.transmitted_fraction) < 1e-6
    assert st.sample_pulse_energy_uJ < 0.6 * st.input_pulse_energy_uJ


# 4. Beam tilt is introduced before propagation (input-plane phase ramp).
def test_beam_tilt_before_propagation():
    base = run_component_plane_pipeline({}, config=CFG)
    tilt = run_component_plane_pipeline(
        {"enable_beam_tilt": True, "beam_tilt_x_mrad": 16.0}, config=CFG
    )
    base_phase_spread = float(np.std(np.angle(base.input_state.field)))
    tilt_phase_spread = float(np.std(np.angle(tilt.input_state.field)))
    assert tilt_phase_spread > base_phase_spread + 0.1
    assert "input_beam_tilt_phase_ramp" in tilt.input_state.applied_components


# 5. Measured tilt slope agrees with the expected free-space slope.
def test_tilt_slope_matches_analytical():
    tv = validate_beam_tilt(24.0, config=CFG)
    assert tv["agrees_within_tolerance"] is True
    assert tv["relative_slope_error"] <= 0.15
    assert tv["grid_pixels_of_displacement"] >= 3.0  # several pixels of displacement
    assert np.sign(tv["measured_slope_x"]) == np.sign(tv["expected_slope_x"])


# 6. Vortex and axicon offsets are independently applied at phase-mask generation.
def test_vortex_axicon_independent():
    base = run_component_plane_pipeline({}, config=CFG).propagated_stack.intensity_zyx
    v = run_component_plane_pipeline(
        {"enable_vortex_centre_offset": True, "vortex_centre_offset_x_um": 8.0}, config=CFG
    ).propagated_stack.intensity_zyx
    a = run_component_plane_pipeline(
        {"enable_axicon_centre_offset": True, "axicon_centre_offset_x_um": 8.0}, config=CFG
    ).propagated_stack.intensity_zyx
    assert not np.allclose(v, base)
    assert not np.allclose(a, base)
    assert not np.allclose(v, a)


# 7. Beam decentre happens before phase-mask interaction.
def test_beam_decentre_before_phase_mask():
    run = run_component_plane_pipeline(
        {"enable_beam_decentre": True, "beam_decentre_x_um": 10.0}, config=CFG
    )
    x = run.input_state.x_um
    amp = np.abs(run.input_state.field)
    X = np.meshgrid(x, x)[0]
    cx = float(np.sum(amp * X) / amp.sum())
    assert cx > 5.0
    assert "input_beam_decentre" in run.input_state.applied_components


# 8. Zero-order leakage before propagation raises core-contamination metrics.
def test_zero_order_leakage_raises_core_contamination():
    base = run_component_plane_pipeline({}, config=CFG)
    leak = run_component_plane_pipeline(
        {"enable_zero_order_leakage": True, "zero_order_leakage_fraction": 0.25}, config=CFG
    )
    base_core = compute_energy_audit(base, fov_reliable=True)["core_energy_fraction"]
    leak_core = compute_energy_audit(leak, fov_reliable=True)["core_energy_fraction"]
    assert leak_core > base_core
    applied = [c for s in leak.propagated_stack.plane_states for c in s.applied_components]
    assert "zero_order_carrier_leakage" in applied


# 9. Actual-axis tracking returns valid centre/trajectory values.
def test_axis_tracking_valid():
    run = run_component_plane_pipeline(
        {"enable_beam_tilt": True, "beam_tilt_x_mrad": 12.0}, config=CFG
    )
    ax = compute_axis_tracking(run.propagated_stack)
    for key in ["intensity_centroid_x_um", "ring_centre_x_um", "peak_x_um",
                "radial_axis_error_um", "ring_fit_quality", "trajectory_fit_quality",
                "axis_intercept_at_z0_x_um", "reference_plane_axis_error_um"]:
        assert np.isfinite(float(ax[key]))
    assert 0.0 <= ax["ring_fit_quality"] <= 1.0
    assert len(ax["centre_trajectory_x_um"]) == run.propagated_stack.intensity_zyx.shape[0]
    assert isinstance(ax["valid_z_fit_range_um"], tuple)


# 10. Crop/FOV reliability flags trigger with insufficient grid/FOV.
def test_fov_reliability_flags_trigger():
    ok = fov_convergence_check({}, config=CFG)
    assert ok["metric_convergence_status"] == "numerically_reliable"
    undersized = replace(CFG, grid_N=56, dx_um=0.5)  # FOV 28 um < beam
    bad = fov_convergence_check({}, config=undersized)
    assert bad["metric_convergence_status"] in ("caution_crop_limited", "invalid_out_of_frame")


# 11. Individual sweeps are labelled diagnostic sensitivity sweeps (not tolerances).
def test_sweeps_labelled_diagnostic_only():
    assert "Diagnostic sensitivity sweep" in _DIAG_NOTE
    assert "Not an experimentally measured laboratory tolerance" in _DIAG_NOTE
    # the individual atlas excludes the combined stress test as primary evidence
    assert "combined_stress" not in DEFAULT_ATLAS_SCENARIOS
    # individual aberration modes are represented separately
    for mode in ("zernike_defocus", "zernike_astigmatism", "zernike_coma", "zernike_spherical"):
        assert mode in DEFAULT_ATLAS_SCENARIOS


# 12. No material-response claim is introduced; outputs are free-space diagnostics.
def test_no_material_response_claim():
    run = run_component_plane_pipeline(
        {"enable_pupil_clipping": True, "pupil_radius_um": 12.0}, config=CFG
    )
    st = run.propagated_stack
    assert run.model_status == "optical_prediction"
    assert run.final_export_allowed is False
    assert float(run.config.n_medium) == 1.0
    eq = zero_control_equivalence(CFG)
    banned = ("absorbed", "dose", "plasma", "ablation", "void", "refractive_index_change",
              "deposited", "damage", "in_sample", "in-material")
    blob = (str(st.metadata) + str(eq) + str(run.reference_plane_state.metadata)).lower()
    assert not any(b in blob for b in banned)


# Reference-plane terminology: free-space n=1.0, not a sample/in-material plane.
def test_reference_plane_terminology():
    run = run_component_plane_pipeline({}, config=CFG)
    assert run.reference_plane_state.plane_name == "free_space_reference_plane"
    assert "n=1.0" in run.reference_plane_state.metadata.get("reference_plane", "")
    au = compute_energy_audit(run, fov_reliable=True)
    assert "reference_plane_pulse_energy_uJ" in au
    assert au["reference_plane_pulse_energy_uJ"] == au["sample_pulse_energy_uJ"]
    # the propagated reference stack carries the reference-energy alias
    assert run.propagated_stack.reference_plane_pulse_energy_uJ == run.propagated_stack.sample_pulse_energy_uJ
