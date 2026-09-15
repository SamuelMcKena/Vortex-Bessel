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


def test_circular_pupil_gate_is_the_only_explicit_blanking_operation():
    cfg = tiny_config()
    cfg.switches.blaze = True
    cfg.switches.circular_pupil = True
    result = compose_phase(cfg)
    assert any("CIRCULAR PUPIL GATE IS ON" in w for w in result.warnings)
