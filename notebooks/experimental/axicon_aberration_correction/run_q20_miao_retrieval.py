"""Run the corrected q=20 Miao-style retrieval on the full 18x4 BMG scan.

The runner preserves genuine plane-to-plane beam motion, optimizes k_perp at
each z plane, increases aberration modal order until convergence, reconstructs
the radial phase from k_perp(z), and blocks SLM output until the required bench
calibrations and conjugate-branch check are present.

A geometric input-plane -> SLM2 phase remap is permitted only when the two
planes have been experimentally confirmed to be conjugate.  Otherwise the
complex field must be propagated/back-propagated through the measured relay; a
simple scale/rotation/parity transform is deliberately blocked.
"""
from __future__ import annotations

from pathlib import Path
import csv
import json

import numpy as np
from scipy import ndimage

from modal_vortex_bessel import read_bmg, preprocess, find_dark_core_center, estimate_global_kr
from miao_full_retrieval import (
    fit_plane_adaptive, assemble_full_aperture, interpolate_to_cartesian,
    map_input_phase_to_slm2, correction_manifest,
)


def load_scan_preserve_plane_shift(data_dir, roi_size=768,
                                   expected_planes=18, expected_repeats=4):
    """Average repeat jitter only; never recenter one z plane onto another."""
    folder = Path(data_dir)
    groups = {}
    for p in folder.glob("z*_*.bmg"):
        try:
            zi = int(p.stem.split("_")[0][1:])
        except Exception:
            continue
        groups.setdefault(zi, []).append(p)
    keys = sorted(groups)
    if len(keys) != expected_planes:
        raise ValueError(f"Expected {expected_planes} z planes, found {len(keys)}")
    for zi in keys:
        groups[zi] = sorted(groups[zi])
        if len(groups[zi]) != expected_repeats:
            raise ValueError(f"z{zi}: expected {expected_repeats} repeats, found {len(groups[zi])}")

    frame_centres, all_centres = {}, []
    for zi in keys:
        rows = []
        for p in groups[zi]:
            a = preprocess(read_bmg(p))
            cy, cx, score = find_dark_core_center(a)
            rows.append((cy, cx, score))
            all_centres.append((cy, cx))
        frame_centres[zi] = rows

    global_cy, global_cx = np.median(np.asarray(all_centres, float), axis=0)
    h = roi_size//2
    y0 = max(0, min(2048-roi_size, int(round(global_cy))-h))
    x0 = max(0, min(2048-roi_size, int(round(global_cx))-h))
    estimated_axis_crop_yx = (float(global_cy-y0), float(global_cx-x0))

    images, qc = [], []
    for zi in keys:
        centres = np.asarray([(c[0], c[1]) for c in frame_centres[zi]], float)
        target = np.median(centres, axis=0)
        repeats = []
        for p, (cy, cx, score) in zip(groups[zi], frame_centres[zi]):
            a = preprocess(read_bmg(p))
            # Remove repeat-to-repeat acquisition jitter within this z only.
            # The z-plane target itself is never shifted to another z-plane target.
            a = ndimage.shift(a, (target[0]-cy, target[1]-cx),
                              order=1, mode="constant", cval=0)
            repeats.append(a[y0:y0+roi_size, x0:x0+roi_size])
            qc.append({"z_index": zi, "file": p.name,
                       "core_y_raw_px": float(cy), "core_x_raw_px": float(cx),
                       "core_score": float(score),
                       "crop_origin_y_px": int(y0), "crop_origin_x_px": int(x0)})
        images.append(np.mean(np.stack(repeats), axis=0))
    crop_origin = (int(y0), int(x0))
    return images, np.asarray(keys), estimated_axis_crop_yx, crop_origin, qc


def _axis_in_crop(calibration, estimated_axis_crop_yx, crop_origin_yx):
    raw_axis = calibration.get("camera_optical_axis_yx_px")
    if raw_axis is None:
        return tuple(map(float, estimated_axis_crop_yx)), False
    if len(raw_axis) != 2:
        raise ValueError("camera_optical_axis_yx_px must be [y, x] in raw 2048x2048 camera pixels")
    y0, x0 = crop_origin_yx
    axis = (float(raw_axis[0])-y0, float(raw_axis[1])-x0)
    return axis, True


def run(data_dir, output_dir, *, calibration_json=None,
        z_relative_mm=np.arange(-17.0, 1.0), wavelength_m=1030e-9,
        pixel_pitch_m=5.5e-6, q=20, max_aberration_order=30):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    calibration = {}
    if calibration_json:
        calibration = json.loads(Path(calibration_json).read_text(encoding="utf-8"))

    images, z_keys, estimated_axis, crop_origin, qc = load_scan_preserve_plane_shift(data_dir)
    z_relative_mm = np.asarray(z_relative_mm, float)
    if len(images) != len(z_relative_mm):
        raise ValueError("z_relative_mm must contain one value per plane")
    retrieval_axis, axis_calibrated = _axis_in_crop(calibration, estimated_axis, crop_origin)

    # Global ring estimate is only an optimizer seed now, never the fitted value.
    seed_kp, _ = estimate_global_kr(images, pixel_pitch_m, q, .55)
    retrievals = []
    for i, (image, zmm) in enumerate(zip(images, z_relative_mm)):
        retrievals.append(fit_plane_adaptive(
            image, i, zmm*1e-3, retrieval_axis, pixel_pitch_m, q, seed_kp,
            max_aberration_order=max_aberration_order))

    rows = [{
        "z_index": r.z_index, "z_relative_mm": r.z_relative_m*1e3,
        "retrieval_axis_y_px": r.center_y_px, "retrieval_axis_x_px": r.center_x_px,
        "camera_optical_axis_calibrated": axis_calibrated,
        "k_perp_opt_m_inv": r.k_perp_m_inv,
        "aberration_order_max": r.aberration_order_max,
        "fit_cost": r.fit_cost, "fit_corr": r.fit_corr, "fit_nrmse": r.fit_nrmse,
    } for r in retrievals]
    with (out/"per_plane_retrieval.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (out/"frame_qc_preserved_coordinates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(qc[0])); w.writeheader(); w.writerows(qc)

    z0 = calibration.get("z_at_relative_zero_from_axicon_mm")
    nominal_kp = calibration.get("k_perp_nominal_m_inv")
    conjugate_plane = calibration.get("slm2_is_conjugate_to_input_plane")
    base = {
        "method": "Miao-style per-plane k_perp + adaptive complex Bessel modal retrieval",
        "planes": len(images), "repeats_per_plane": 4,
        "programmed_q": int(q), "programmed_vortex_in_correction": False,
        "plane_to_plane_recentering": False,
        "camera_optical_axis_calibrated": axis_calibrated,
        "retrieval_axis_yx_px_in_crop": list(map(float, retrieval_axis)),
        "global_k_perp_used_only_as_seed_m_inv": float(seed_kp),
        "nominal_k_perp_calibrated": nominal_kp is not None,
        "absolute_z_calibrated": z0 is not None,
        "slm2_conjugacy_confirmed": conjugate_plane is True,
        "hardware_ready": False,
    }
    early_blockers = []
    if z0 is None:
        early_blockers.append("measure z_at_relative_zero_from_axicon_mm before radial phase assembly")
    if nominal_kp is None:
        early_blockers.append("supply the intended/calibrated k_perp_nominal_m_inv before a correction trial")
    if not axis_calibrated:
        early_blockers.append("measure camera_optical_axis_yx_px; the median beam core is diagnostic only")
    if z0 is None:
        base.update({
            "status": "LOCAL_RETRIEVAL_ONLY",
            "hardware_blockers": early_blockers + [
                "resolve the conjugate/180-degree branch with an independent reference",
                "confirm whether SLM2 is conjugate to the reconstructed input plane",
                "calibrate the applicable input-plane/SLM2 transform and 1030-nm LUT",
            ],
        })
        (out/"correction_manifest.json").write_text(json.dumps(base, indent=2), encoding="utf-8")
        return base

    z_abs_m = (float(z0)+z_relative_mm)*1e-3
    if np.any(z_abs_m <= 0):
        raise ValueError("absolute z calibration places at least one plane at z<=0")

    reference = None
    ref_path = calibration.get("input_reference_annular_intensity_npy")
    if ref_path:
        reference = np.load(ref_path)
    full = assemble_full_aperture(retrievals, z_abs_m, wavelength_m,
                                  k_perp_nominal_m_inv=nominal_kp,
                                  reference_intensity_rows=reference)
    cart = interpolate_to_cartesian(full, grid_size=768)
    np.save(out/"retrieved_full_residual_phase_input_plane_rad.npy",
            cart["residual_phase_rad"].astype(np.float32))
    np.save(out/"conjugate_correction_input_plane_rad.npy",
            cart["conjugate_correction_phase_rad"].astype(np.float32))
    np.save(out/"rho_sampled_m.npy", full.rho_m)
    np.save(out/"radial_phase_rad.npy", full.radial_phase_rad)
    np.save(out/"radial_phase_gradient_rad_per_m.npy", full.radial_phase_gradient_rad_per_m)
    np.save(out/"angular_phase_rows_rad.npy", full.angular_phase_rows_rad.astype(np.float32))

    transform_keys = ("slm2_shape", "input_plane_m_per_slm2_pixel",
                      "slm2_center_yx_px", "slm2_rotation_deg",
                      "slm2_parity_x", "slm2_parity_y")
    transform_values_ready = all(calibration.get(k) is not None for k in transform_keys)
    geometric_mapping_ready = bool(conjugate_plane is True and transform_values_ready and axis_calibrated)
    lut_ready = bool(calibration.get("slm2_phase_lut_1030nm_calibrated", False))
    nominal_ready = nominal_kp is not None

    slm_written = False
    if full.branch != "unresolved" and geometric_mapping_ready and nominal_ready:
        slm_map = map_input_phase_to_slm2(
            cart, tuple(calibration["slm2_shape"]),
            float(calibration["input_plane_m_per_slm2_pixel"]),
            tuple(calibration["slm2_center_yx_px"]),
            float(calibration["slm2_rotation_deg"]),
            int(calibration["slm2_parity_x"]), int(calibration["slm2_parity_y"]))
        np.save(out/"slm2_correction_phase_rad.npy", slm_map.astype(np.float32))
        slm_written = True

    mapping_for_manifest = bool(geometric_mapping_ready and nominal_ready)
    manifest = correction_manifest(full, True, mapping_for_manifest, lut_ready, False)
    extra_pretrial = []
    if not nominal_ready:
        extra_pretrial.append("intended/calibrated nominal k_perp is missing")
    if not axis_calibrated:
        extra_pretrial.append("camera optical axis is not independently calibrated")
    if conjugate_plane is not True:
        if conjugate_plane is False:
            extra_pretrial.append(
                "SLM2 is not conjugate to the reconstructed input plane; implement measured relay complex-field back-propagation before mapping to SLM2")
        else:
            extra_pretrial.append("SLM2/input-plane conjugacy has not been established")
    for item in extra_pretrial:
        if item not in manifest["pretrial_blockers"]:
            manifest["pretrial_blockers"].append(item)
        if item not in manifest["hardware_blockers"]:
            manifest["hardware_blockers"].insert(0, item)
    manifest["application_ready_for_low_gain_trial"] = len(manifest["pretrial_blockers"]) == 0
    manifest["hardware_ready"] = len(manifest["hardware_blockers"]) == 0

    manifest.update(base)
    manifest.update({
        "status": "FULL_RETRIEVAL_COMPLETE",
        "branch": full.branch,
        "branch_score_direct": full.branch_score_direct,
        "branch_score_conjugate": full.branch_score_conjugate,
        "absolute_z_at_relative_zero_from_axicon_mm": float(z0),
        "k_perp_nominal_m_inv": None if nominal_kp is None else float(nominal_kp),
        "slm2_is_conjugate_to_input_plane": conjugate_plane,
        "input_plane_to_slm2_geometric_mapping_calibrated": geometric_mapping_ready,
        "slm2_phase_lut_1030nm_calibrated": lut_ready,
        "slm2_phase_map_written": slm_written,
        "slm2_shape_yx": (list(map(int, calibration["slm2_shape"]))
                          if slm_written else None),
        "nonconjugate_relay_backpropagation_implemented": False,
    })
    # base intentionally starts hardware_ready=False; recompute after update.
    manifest["application_ready_for_low_gain_trial"] = len(manifest["pretrial_blockers"]) == 0
    manifest["hardware_ready"] = len(manifest["hardware_blockers"]) == 0
    (out/"correction_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    cal = here/"q20_hardware_calibration.json"
    result = run(here/"z-scan 2 1010", here/"outputs"/"miao_full_q20",
                 calibration_json=cal if cal.exists() else None)
    print(json.dumps(result, indent=2))
