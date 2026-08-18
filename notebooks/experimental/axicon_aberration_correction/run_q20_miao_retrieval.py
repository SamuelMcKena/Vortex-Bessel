"""Run the corrected q=20 Miao-style retrieval on the full 18x4 BMG scan.

This runner preserves genuine plane-to-plane beam motion, optimizes k_perp at
each z plane, increases aberration modal order until convergence, reconstructs
the radial phase from k_perp(z), and blocks SLM output until the required bench
calibrations and conjugate-branch check are present.
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
    fixed_axis = (float(global_cy-y0), float(global_cx-x0))

    images, qc = [], []
    for zi in keys:
        centres = np.asarray([(c[0], c[1]) for c in frame_centres[zi]], float)
        target = np.median(centres, axis=0)
        repeats = []
        for p, (cy, cx, score) in zip(groups[zi], frame_centres[zi]):
            a = preprocess(read_bmg(p))
            # Only repeat-to-repeat acquisition jitter is removed.
            a = ndimage.shift(a, (target[0]-cy, target[1]-cx),
                              order=1, mode="constant", cval=0)
            repeats.append(a[y0:y0+roi_size, x0:x0+roi_size])
            qc.append({"z_index": zi, "file": p.name,
                       "core_y_raw_px": float(cy), "core_x_raw_px": float(cx),
                       "core_score": float(score)})
        images.append(np.mean(np.stack(repeats), axis=0))
    return images, np.asarray(keys), fixed_axis, qc


def run(data_dir, output_dir, *, calibration_json=None,
        z_relative_mm=np.arange(-17.0, 1.0), wavelength_m=1030e-9,
        pixel_pitch_m=5.5e-6, q=20, max_aberration_order=30):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    calibration = {}
    if calibration_json:
        calibration = json.loads(Path(calibration_json).read_text(encoding="utf-8"))

    images, z_keys, fixed_axis, qc = load_scan_preserve_plane_shift(data_dir)
    z_relative_mm = np.asarray(z_relative_mm, float)
    if len(images) != len(z_relative_mm):
        raise ValueError("z_relative_mm must contain one value per plane")

    # Global ring estimate is only an optimizer seed now, never the fitted value.
    seed_kp, _ = estimate_global_kr(images, pixel_pitch_m, q, .55)
    retrievals = []
    for i, (image, zmm) in enumerate(zip(images, z_relative_mm)):
        retrievals.append(fit_plane_adaptive(
            image, i, zmm*1e-3, fixed_axis, pixel_pitch_m, q, seed_kp,
            max_aberration_order=max_aberration_order))

    rows = [{
        "z_index": r.z_index, "z_relative_mm": r.z_relative_m*1e3,
        "fixed_axis_y_px": r.center_y_px, "fixed_axis_x_px": r.center_x_px,
        "k_perp_opt_m_inv": r.k_perp_m_inv,
        "aberration_order_max": r.aberration_order_max,
        "fit_cost": r.fit_cost, "fit_corr": r.fit_corr, "fit_nrmse": r.fit_nrmse,
    } for r in retrievals]
    with (out/"per_plane_retrieval.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (out/"frame_qc_preserved_coordinates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(qc[0])); w.writeheader(); w.writerows(qc)

    z0 = calibration.get("z_at_relative_zero_from_axicon_mm")
    base = {
        "method": "Miao-style per-plane k_perp + adaptive complex Bessel modal retrieval",
        "planes": len(images), "repeats_per_plane": 4,
        "programmed_q": int(q), "programmed_vortex_in_correction": False,
        "plane_to_plane_recentering": False,
        "global_k_perp_used_only_as_seed_m_inv": float(seed_kp),
        "absolute_z_calibrated": z0 is not None,
        "hardware_ready": False,
    }
    if z0 is None:
        base.update({
            "status": "LOCAL_RETRIEVAL_ONLY",
            "hardware_blockers": [
                "measure z_at_relative_zero_from_axicon_mm before radial phase assembly",
                "resolve the conjugate/180-degree branch with an independent reference",
                "calibrate input-plane to SLM2 scale/rotation/parity/centre and 1030-nm LUT",
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
                                  k_perp_nominal_m_inv=calibration.get("k_perp_nominal_m_inv"),
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

    mapping_keys = ("slm2_shape", "input_plane_m_per_slm2_pixel",
                    "slm2_center_yx_px", "slm2_rotation_deg",
                    "slm2_parity_x", "slm2_parity_y")
    mapping_ready = all(calibration.get(k) is not None for k in mapping_keys)
    lut_ready = bool(calibration.get("slm2_phase_lut_1030nm_calibrated", False))
    slm_written = False
    if full.branch != "unresolved" and mapping_ready:
        slm_map = map_input_phase_to_slm2(
            cart, tuple(calibration["slm2_shape"]),
            float(calibration["input_plane_m_per_slm2_pixel"]),
            tuple(calibration["slm2_center_yx_px"]),
            float(calibration["slm2_rotation_deg"]),
            int(calibration["slm2_parity_x"]), int(calibration["slm2_parity_y"]))
        np.save(out/"slm2_correction_phase_rad.npy", slm_map.astype(np.float32))
        slm_written = True

    manifest = correction_manifest(full, True, mapping_ready, lut_ready, False)
    manifest.update(base)
    manifest.update({
        "status": "FULL_RETRIEVAL_COMPLETE",
        "branch": full.branch,
        "branch_score_direct": full.branch_score_direct,
        "branch_score_conjugate": full.branch_score_conjugate,
        "absolute_z_at_relative_zero_from_axicon_mm": float(z0),
        "input_plane_to_slm2_mapping_calibrated": mapping_ready,
        "slm2_phase_lut_1030nm_calibrated": lut_ready,
        "slm2_phase_map_written": slm_written,
    })
    (out/"correction_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    cal = here/"q20_hardware_calibration.json"
    result = run(here/"z-scan 2 1010", here/"outputs"/"miao_full_q20",
                 calibration_json=cal if cal.exists() else None)
    print(json.dumps(result, indent=2))
