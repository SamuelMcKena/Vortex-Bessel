"""Stage 9A.1 first Fourier-plane carrier calibration session-pack tests."""

import json
from pathlib import Path

import numpy as np
import pytest

from vbb_study.digital_twin.slm_calibration_masks import (
    CarrierSweepConfig,
    command_carrier_phase,
    validate_carrier_sampling,
    build_carrier_mask,
    build_carrier_sweep_masks,
    build_carrier_calibration_study,
    create_fourier_carrier_calibration_session,
    PHYSICAL_FREQUENCY_STATUS,
)

TWOPI = 2.0 * np.pi
CFG = CarrierSweepConfig.demo()


# 1. Carrier masks contain no vortex / axicon / correction-map / hidden aperture.
def test_masks_carrier_only_no_other_terms():
    for m in build_carrier_sweep_masks(CFG):
        meta = m["metadata"]
        assert meta["contains_vortex"] is False
        assert meta["contains_axicon"] is False
        assert meta["contains_correction_map"] is False
        assert meta["contains_aperture_crop"] is False
        # phase-only: full-display coverage, no masked-out (NaN/forced-zero) region
        ph = np.asarray(m["phase_rad"])
        assert ph.shape == (CFG.command_display_height_pixels, CFG.command_display_width_pixels)
        assert np.all(np.isfinite(ph))


# 2. SLM1 masks are flat.
def test_slm1_flat():
    masks = build_carrier_sweep_masks(CFG)
    slm1 = [m for m in masks if m["metadata"]["slm_id"] == "SLM1"]
    assert slm1
    for m in slm1:
        assert m["metadata"]["is_flat"] is True
        assert np.allclose(m["phase_rad"], m["phase_rad"].flat[0])
    with pytest.raises(ValueError):
        build_carrier_mask(CFG, "bad", 8, 0, slm_id="SLM1")  # SLM1 carrier rejected


# 3. Signed x/y carrier cycles produce the expected command-domain ramps.
def test_command_domain_ramps():
    w, h = CFG.command_display_width_pixels, CFG.command_display_height_pixels
    px = np.arange(w); py = np.arange(h)
    for nx in (8, -16):
        phi = command_carrier_phase(w, h, nx, 0)
        exp = np.mod(TWOPI * nx * (px / w), TWOPI)
        assert np.allclose(phi[0], exp)               # x-carrier varies along x
        assert np.allclose(phi[0], phi[1])            # rows identical
    phi_y = command_carrier_phase(w, h, 0, 8)
    assert np.allclose(phi_y[:, 0], np.mod(TWOPI * 8 * (py / h), TWOPI))
    assert np.allclose(phi_y[:, 0], phi_y[:, 1])      # columns identical
    # sign reversal flips the ramp gradient
    assert not np.allclose(command_carrier_phase(w, h, 8, 0), command_carrier_phase(w, h, -8, 0))


# 4. Flat carrier case is a constant phase map.
def test_flat_carrier_constant():
    phi = command_carrier_phase(CFG.command_display_width_pixels, CFG.command_display_height_pixels, 0, 0)
    assert np.allclose(phi, phi.flat[0])


# 5. Sampling guard rejects under-sampled carriers (no aliasing).
def test_sampling_guard():
    tiny = CarrierSweepConfig.demo(command_display_width_pixels=64, command_display_height_pixels=64)
    assert validate_carrier_sampling(tiny, 24, 0)         # 64/24 = 2.7 px/cycle < 8
    assert validate_carrier_sampling(CFG, 8, 0) == []     # 480/8 = 60 px/cycle ok
    with pytest.raises(ValueError):
        build_carrier_mask(tiny, "alias", 24, 0)          # construction blocked


# 6. Metadata records command-domain status; never claims physical frequency calibration.
def test_metadata_command_domain():
    m = build_carrier_mask(CFG, "slm2_carrier_x_+8", 8, 0)
    meta = m["metadata"]
    assert meta["physical_frequency_status"] == PHYSICAL_FREQUENCY_STATUS == "uncalibrated_command_domain"
    assert meta["phase_response_calibration_status"] in ("unknown_or_unverified",)
    assert meta["coordinate_frame"] == "SLM2_phase_map_frame"
    blob = json.dumps(meta).lower()
    assert "cycles_per_mm" not in blob and "cycles_per_m" not in blob
    assert "carrier_cycles_x" in meta and "carrier_cycles_y" in meta


# 7. Session package creates all expected outputs.
def test_session_package_files(tmp_path):
    paths = create_fourier_carrier_calibration_session(
        carrier_config=CFG, output_root=tmp_path / "o", data_root=tmp_path / "d",
        save_atlas_to=tmp_path / "atlas.png")
    for key in ("run_manifest", "acquisition_plan", "capture_manifest_template",
                "hardware_profile_snapshot", "bench_inventory_snapshot",
                "coordinate_contract_snapshot"):
        assert Path(paths[key]).is_file(), key
    exp = paths["experiment_package_dir"]
    for name in ("LAB_README_FIRST_FOURIER_SESSION.md", "bench_setup_sheet.md", "bench_setup_sheet.csv",
                 "camera_capture_checklist.csv", "carrier_sweep_log.csv",
                 "fourier_plane_observation_template.csv", "operator_notes_template.md"):
        assert (exp / name).is_file(), name
    # phase masks exported for SLM1 and SLM2
    assert list(Path(paths["phase_masks_slm1_dir"]).glob("*_metadata.json"))
    assert len(list(Path(paths["phase_masks_slm2_dir"]).glob("*_metadata.json"))) >= 10
    readme = (exp / "LAB_README_FIRST_FOURIER_SESSION.md").read_text(encoding="utf-8")
    assert "command-domain cycles" in readme
    assert "does not validate physical 4F propagation" in readme


# 8. Session run ids do not overwrite previous runs.
def test_session_not_overwritten(tmp_path):
    create_fourier_carrier_calibration_session(
        "fixedrun", carrier_config=CFG, output_root=tmp_path / "o", data_root=tmp_path / "d",
        save_atlas_to=tmp_path / "a1.png")
    with pytest.raises(FileExistsError):
        create_fourier_carrier_calibration_session(
            "fixedrun", carrier_config=CFG, output_root=tmp_path / "o", data_root=tmp_path / "d",
            save_atlas_to=tmp_path / "a2.png")


# 9. Raw-data directories contain no fabricated capture files.
def test_no_fabricated_raw(tmp_path):
    paths = create_fourier_carrier_calibration_session(
        carrier_config=CFG, output_root=tmp_path / "o", data_root=tmp_path / "d",
        save_atlas_to=tmp_path / "a.png")
    raw = Path(paths["data_dir"]) / "raw"
    assert raw.is_dir()
    assert list(raw.iterdir()) == []


# 11. No physical 4F / camera / inverse / AI / material / physical-axicon propagation enabled.
def test_no_unsafe_enabling():
    study = build_carrier_calibration_study(CFG)
    gov = study["governance"]
    assert gov["physical_4f_filter_modelled"] is False
    assert gov["camera_model_enabled"] is False
    assert gov["material_model_enabled"] is False
    assert gov["final_export_allowed"] is False
    assert study["slm2_mode"] == "command_domain_carrier_ramp_only"
    assert "removed" in study["physical_axicon_state"] or "bypass" in study["physical_axicon_state"]
    blob = json.dumps(study).lower()
    for token in ("neural", "inverse", "zernike", "aberration", "dose", "plasma", "thermal",
                  "ablation", "waveguide", "weld", "fused_silica_prediction"):
        assert token not in blob
