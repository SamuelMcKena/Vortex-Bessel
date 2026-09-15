from slm_lab_control.config import SlmPhaseConfig, SlmGeometry
from slm_lab_control.phase import compose_phase, pupil_mask

import numpy as np


def tiny_config():
    cfg = SlmPhaseConfig()
    cfg.geometry = SlmGeometry(width_px=128, height_px=72, pixel_pitch_um=8.0, wavelength_nm=1030.0)
    cfg.center_x_px = 64
    cfg.center_y_px = 36
    cfg.pupil_diameter_mm = 0.5
    return cfg


def test_blaze_changes_output():
    cfg = tiny_config()
    cfg.switches.blaze = True
    cfg.blaze_period_px = 20
    a = compose_phase(cfg).gray_uint8
    cfg.blaze_period_px = 40
    b = compose_phase(cfg).gray_uint8
    assert a.shape == (72, 128)
    assert (a != b).any()


def test_focus_changes_output():
    cfg = tiny_config()
    cfg.switches.blaze = False
    cfg.switches.focus = True
    cfg.focus_focal_length_mm = 100
    a = compose_phase(cfg).gray_uint8
    cfg.focus_focal_length_mm = 200
    b = compose_phase(cfg).gray_uint8
    assert (a != b).any()


def test_vortex_and_axicon_can_compose():
    cfg = tiny_config()
    cfg.switches.blaze = False
    cfg.switches.axicon = True
    cfg.switches.vortex = True
    cfg.axicon_period_px = 30
    cfg.vortex_charge = 2
    result = compose_phase(cfg)
    assert result.gray_uint8.dtype.name == "uint8"
    assert "axicon" in result.components
    assert "vortex" in result.components


def test_tip_tilt_is_a_dedicated_additive_term_and_does_not_change_carrier():
    cfg = tiny_config()
    cfg.switches.blaze = True
    carrier = compose_phase(cfg).components["blaze"].copy()

    cfg.switches.steering = True
    cfg.steering_x_mrad = 1.25
    cfg.steering_y_mrad = -0.75
    result = compose_phase(cfg)

    assert "steering" in result.components
    assert np.array_equal(result.components["blaze"], carrier)
    assert not np.allclose(result.components["steering"], 0.0)


def test_low_order_zernike_changes_output():
    cfg = tiny_config()
    cfg.switches.blaze = False
    cfg.switches.zernike_z40 = True
    cfg.z22_cos_amp_waves = 0.2
    a = compose_phase(cfg).gray_uint8
    cfg.z22_cos_amp_waves = -0.2
    b = compose_phase(cfg).gray_uint8
    assert (a != b).any()


def test_spherical_interface_is_additive_and_never_gates_outside_pupil():
    cfg = tiny_config()
    cfg.switches.blaze = True
    base = compose_phase(cfg)

    cfg.switches.spherical_interface = True
    cfg.interface_NA = 0.4
    cfg.interface_depth_um = 1000.0
    corrected = compose_phase(cfg)
    mask = pupil_mask(cfg)

    # Outside the spherical computational pupil, the correction contribution is
    # exactly zero: existing blaze/wavefront phase must survive unchanged.
    assert np.allclose(corrected.phase_rad[~mask], base.phase_rad[~mask])
    assert np.any(np.abs(corrected.phase_rad[mask] - base.phase_rad[mask]) > 1e-9)
    assert "spherical_interface" in corrected.components


def test_retrieved_npy_correction_adds_to_existing_blaze_before_final_wrap(tmp_path):
    cfg = tiny_config()
    cfg.switches.blaze = True
    base = compose_phase(cfg)

    raw = np.full((cfg.geometry.height_px, cfg.geometry.width_px), 0.4, dtype=np.float32)
    path = tmp_path / "correction_rad.npy"
    np.save(path, raw)
    cfg.switches.retrieved_correction = True
    cfg.retrieved_correction_path = str(path)
    cfg.retrieved_correction_gain = 0.25
    corrected = compose_phase(cfg)

    assert np.allclose(corrected.components["retrieved_correction"], 0.1)
    assert np.allclose(corrected.phase_rad - base.phase_rad, 0.1)


def test_legacy_circular_pupil_switch_is_safe_and_never_blanks_phase():
    cfg = tiny_config()
    cfg.switches.blaze = True
    baseline = compose_phase(cfg)

    cfg.switches.circular_pupil = True
    result = compose_phase(cfg)

    assert np.array_equal(result.phase_rad, baseline.phase_rad)
    assert np.array_equal(result.gray_uint8, baseline.gray_uint8)
    assert "circular_pupil_gate" not in result.components
    assert any("legacy circular_pupil gate request ignored" in w for w in result.warnings)


def test_changing_reference_pupil_does_not_change_panel_wide_terms():
    cfg = tiny_config()
    cfg.switches.blaze = True
    cfg.switches.vortex = True
    cfg.switches.axicon = True
    cfg.switches.focus = True
    cfg.vortex_charge = 7
    cfg.axicon_period_px = 31.0
    cfg.focus_focal_length_mm = 250.0

    cfg.pupil_diameter_mm = 0.25
    small = compose_phase(cfg)
    cfg.pupil_diameter_mm = 0.80
    large = compose_phase(cfg)

    assert np.array_equal(small.phase_rad, large.phase_rad)
    assert np.array_equal(small.gray_uint8, large.gray_uint8)


def test_pupil_local_zernike_leaves_existing_phase_untouched_outside_reference():
    cfg = tiny_config()
    cfg.switches.blaze = True
    baseline = compose_phase(cfg)

    cfg.switches.zernike_z40 = True
    cfg.z22_cos_amp_waves = 0.2
    corrected = compose_phase(cfg)
    mask = pupil_mask(cfg)

    assert np.allclose(corrected.phase_rad[~mask], baseline.phase_rad[~mask])
    assert np.any(np.abs(corrected.phase_rad[mask] - baseline.phase_rad[mask]) > 1e-9)
