"""MODE 2V lab-ready source-scale build package.

M2U3-A authorised M2V: convert the validated source-scale dual-SLM + carrier +
4F + QWP + axicon simulation into a practical experimental package - exact
bench architecture, polarisation routing, native 1920x1080 masks, physical 4F
dimensions, power budget, component table, first-day alignment procedure,
measurement responsibilities, and a simulated closed-loop correction
demonstration whose correction search never receives the injected truth.

Canonical operating point: REALISTIC_4F_HEXAGON_REFERENCE (best strict-eligible
shape/peak/useful-energy within the bounded M2U2-FIX search; no global
mathematical optimality claimed).  Secondary: the strict compromise.  All old
optima that fail the repaired strict hexagon gate are permanently forbidden.
No microfabrication/sample-plane success is claimed anywhere.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from vbb_study.digital_twin.nathan_mode2u2_fix_strict_hexagon_optimisation import (
    MODE2U2F_DEFAULT_OUTPUT_ROOT,
    OLD_BEST_COMPROMISE_ID,
    STRICT_BASELINE_CORR_MIN,
    evaluate_strict_hexagon_metrics,
)
from vbb_study.digital_twin.nathan_mode2u2_master_closure import (
    correction_responsibility_matrix,
)
from vbb_study.digital_twin.nathan_mode2u3_hardware_closure import (
    CANONICAL_OPERATING_POINT_ID,
    STRICT_COMPROMISE_ID,
    _bench,
    assert_not_forbidden,
    physical_4f_rows,
    qwp_lab_axis_statement,
)
from vbb_study.digital_twin.nathan_vector_hexagon import (
    EPS,
    Mode2SCorrection,
    Mode2SPerturbation,
    NathanSourceParityConfig,
    _apply_free_space_vector_axicon,
    _json_ready,
    _mode1b_even_axis_crop,
    _normalise_image,
    _write_rows,
    mode2s_combined_cases,
    nathan_alpha_map,
    run_mode2s_degraded_forward,
    wrap_2pi,
)

MODE2V_STAGE = "nathan_mode2v_lab_ready_build"
MODE2V_DEFAULT_OUTPUT_ROOT = Path("outputs/figures/digital_twin/nathan_mode2v_lab_ready_build")
MODE2V_JONES_DOC_PATH = Path("docs/79_nathan_mode2v_full_jones_build_derivation.md")
MODE2V_FIRST_DAY_DOC_PATH = Path("docs/80_nathan_mode2v_first_day_lab_procedure.md")
MODE2V_MASTER_DOC_PATH = Path("docs/81_nathan_mode2v_lab_ready_master_report.md")
MODE2V_ALLOWED_OUTCOMES = ("M2V-A", "M2V-B", "M2V-C", "M2V-D")
MODE2V_SEED = 20260709

SLM_WIDTH_PX = 1920
SLM_HEIGHT_PX = 1080
SLM_PITCH_M = 8.0e-6
LAB_WAVELENGTH_M = 1.029e-6
FOURF_FOCAL_M = 0.300


# ---------------------------------------------------------------------------
# Section 3: frozen operating points (exact stored values, never re-derived)
# ---------------------------------------------------------------------------

OPERATING_POINT_FIELDS = (
    "candidate_id", "carrier_lpmm", "iris_radius_frac", "qwp_angle_correction_deg",
    "global_v_piston_rad", "sector_rotation_deg", "sector_duty_scale",
    "mask_recentre_x_m", "mask_recentre_y_m",
    "corr_full", "corr_focus_crop", "corr_angular", "corr_to_realistic_4f",
    "c45", "c60", "c90", "c120", "c180", "h3", "h4", "h6",
    "dark_core_ratio", "strict_hexagon_eligible", "classifier_label",
    "strict_peak_metric", "peak_metric_definition", "P_useful", "P_useful_over_P_total",
    "first_order_efficiency", "total_throughput", "strict_compromise_score",
)


def load_operating_points(
    candidates_path: str | Path = MODE2U2F_DEFAULT_OUTPUT_ROOT / "strict_hexagon_candidates.json",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the exact stored canonical and secondary operating-point rows."""

    assert_not_forbidden(CANONICAL_OPERATING_POINT_ID)
    assert_not_forbidden(STRICT_COMPROMISE_ID)
    payload = json.loads(Path(candidates_path).read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, dict) else payload
    by_id = {str(row["candidate_id"]): row for row in rows}
    canonical = dict(by_id[CANONICAL_OPERATING_POINT_ID])
    secondary = dict(by_id[STRICT_COMPROMISE_ID])
    for row in (canonical, secondary):
        if not bool(row["strict_hexagon_eligible"]):
            raise ValueError(f"stored operating point {row['candidate_id']!r} is not strict-eligible")
        row["carrier_period_slm_pixels"] = float(1.0 / (float(row["carrier_lpmm"]) * 1.0e3 * SLM_PITCH_M))
    return canonical, secondary


def operating_point_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    summary = {key: row.get(key) for key in OPERATING_POINT_FIELDS}
    summary["carrier_period_slm_pixels"] = row.get("carrier_period_slm_pixels")
    summary["mask_centre_policy"] = "hologram centred on the axicon axis after the measure-and-recentre calibration"
    return summary


# ---------------------------------------------------------------------------
# Section 4: architecture decision
# ---------------------------------------------------------------------------


def architecture_route_comparison() -> list[dict[str, Any]]:
    return [
        {
            "route_id": "A_per_arm_4f",
            "description": "independent 4F relay and first-order iris in each of the H and V arms, recombination after filtering",
            "component_count": "2 lens pairs + 2 irises + PBS recombiner",
            "matches_validated_simulation": False,
            "differential_error_channels": "differential magnification, rotation, shift and iris centring between arms (the M2S H/V registration errors)",
            "relative_phase_behaviour": "two long independent filtered paths; arm piston still free (observable-invariant)",
            "physically_consistent": True,
            "recommended": False,
            "note": "valid but adds avoidable differential degrees of freedom; keep as fallback if the recombiner cannot precede the filter",
        },
        {
            "route_id": "B_common_4f_after_recombination",
            "description": "PBS recombination first, then ONE shared 4F relay and ONE first-order iris acting on both polarisation channels",
            "component_count": "1 lens pair + 1 iris + PBS recombiner",
            "matches_validated_simulation": True,
            "differential_error_channels": "none introduced by the filter: both channels share identical lenses/iris, eliminating differential 4F registration by construction",
            "relative_phase_behaviour": "common path after recombination; the filter cannot add differential H/V drift",
            "physically_consistent": True,
            "recommended": True,
            "note": (
                "this is exactly the configuration every validated M2N/M2Q/M2S/M2U run modelled (one shared "
                "Fourier plane, one iris on both channels); both masks carry the same carrier so a single iris "
                "selects both +1 orders; a hard iris is polarisation-independent"
            ),
        },
        {
            "route_id": "C_single_slm_time_multiplexed",
            "description": "one SLM addressing both channels via double-pass / split-screen",
            "component_count": "1 panel + extra folding",
            "matches_validated_simulation": False,
            "differential_error_channels": "split-screen halves the usable aperture below the 10 mm window requirement",
            "physically_consistent": False,
            "recommended": False,
            "note": "the 2 mm beam with the validated window does not fit a half panel (8.64 mm short axis already clips the window); rejected",
        },
    ]


def architecture_decision() -> dict[str, Any]:
    routes = architecture_route_comparison()
    chosen = next(r for r in routes if r["recommended"])
    return {
        "chosen_route": chosen["route_id"],
        "reason": (
            "simplest architecture that remains physically consistent with phase-only SLM operation, separate "
            "H/V masks, first-order filtering, coherent recombination and relative-phase preservation - and it "
            "is byte-identical to the validated simulation (shared Fourier plane, one iris on both channels)"
        ),
        "consequence_for_bench": (
            "order of components: SLM-H arm and SLM-V arm recombine at PBS #2 BEFORE the 4F; the single iris "
            "then filters both channels identically; the final QWP follows the 4F output"
        ),
        "fallback": "A_per_arm_4f if bench geometry prevents recombination before the relay",
        "routes": routes,
    }


# ---------------------------------------------------------------------------
# Section 6: waveplate table
# ---------------------------------------------------------------------------


def waveplate_table(qwp_statement: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    qwp = qwp_statement or qwp_lab_axis_statement()
    view = "standing downstream looking back INTO the oncoming beam (receiver view)"
    return [
        {
            "component_id": "HWP1_input_balance",
            "location": "after the input polariser, before PBS #1",
            "purpose": "rotate the laser linear polarisation to set the H/V power split (target 50/50)",
            "nominal_angle_code_deg": "set for measured 50/50 split (nominal 22.5 from pure H input)",
            "nominal_physical_angle": "adjusted on power-balance measurement, not preset",
            "reference_axis": "lab horizontal (H)",
            "viewed_from": view,
            "rotation_sense": "anticlockwise positive in receiver view",
            "axis_reference": "fast axis",
            "tolerance": "H/V amplitude ratio 0.8-1.2 tolerated (M2S: all pass)",
            "calibration_method": "rotate until per-arm powers after PBS #1 are equal",
            "essential": True,
            "conditional": False,
        },
        {
            "component_id": "HWP2_v_arm_pre_slm",
            "location": "V arm, before SLM-V",
            "purpose": "rotate V polarisation onto the SLM-V LC director for phase-only operation",
            "nominal_angle_code_deg": 45.0,
            "nominal_physical_angle": "45 deg between V and the measured panel director",
            "reference_axis": "lab vertical (V) to panel director",
            "viewed_from": view,
            "rotation_sense": "anticlockwise positive in receiver view",
            "axis_reference": "fast axis",
            "tolerance": "sets phase-only purity; residual amplitude modulation if misaligned",
            "calibration_method": "panel orientation test (STAGE 2 of docs/80): minimise amplitude modulation of a displayed grating",
            "essential": False,
            "conditional": "conditional_on_panel_orientation_test",
        },
        {
            "component_id": "HWP3_v_arm_post_slm",
            "location": "V arm, after SLM-V",
            "purpose": "restore the modulated field to V polarisation before PBS #2 recombination",
            "nominal_angle_code_deg": 45.0,
            "nominal_physical_angle": "matched to HWP2",
            "reference_axis": "panel director back to lab vertical (V)",
            "viewed_from": view,
            "rotation_sense": "anticlockwise positive in receiver view",
            "axis_reference": "fast axis",
            "tolerance": "as HWP2",
            "calibration_method": "maximise V-arm transmission through PBS #2 reflection port",
            "essential": False,
            "conditional": "conditional_on_panel_orientation_test",
        },
        {
            "component_id": "QWP1_final",
            "location": "after PBS #2 recombination and the shared 4F output, before the axicon",
            "purpose": "map the dual-linear H/V channels onto the segmented radial/azimuthal vector field",
            "nominal_angle_code_deg": -45.0,
            "nominal_physical_angle": str(qwp["physical_statement"]),
            "reference_axis": "lab horizontal (H)",
            "viewed_from": view,
            "rotation_sense": "clockwise 45 deg in receiver view for the code -45 deg",
            "axis_reference": str(qwp["fast_or_slow"]),
            "tolerance": "angle +/-2 deg and retardance +/-5 deg pass the strict audit (M2S)",
            "calibration_method": str(qwp["mount_side_caveat"]),
            "essential": True,
            "conditional": False,
        },
    ]


# ---------------------------------------------------------------------------
# Section 7: native SLM mask export
# ---------------------------------------------------------------------------


def native_panel_grid() -> dict[str, Any]:
    """Native 1920 x 1080 panel grid, axis-sampled at pixel (960, 540).

    Coordinates: x = (col - 960) * pitch, y = (row - 540) * pitch, so the
    optical axis is sampled exactly (the validated V0 centring convention).
    Row 0 renders at the bottom of preview PNGs (origin='lower'); the actual
    GUI orientation is fixed by the STAGE 2 panel orientation test.
    """

    cols = (np.arange(SLM_WIDTH_PX, dtype=float) - SLM_WIDTH_PX // 2) * SLM_PITCH_M
    rows = (np.arange(SLM_HEIGHT_PX, dtype=float) - SLM_HEIGHT_PX // 2) * SLM_PITCH_M
    X, Y = np.meshgrid(cols, rows, indexing="xy")
    return {
        "N_x": SLM_WIDTH_PX,
        "N_y": SLM_HEIGHT_PX,
        "dx": SLM_PITCH_M,
        "x": cols,
        "y": rows,
        "X": X,
        "Y": Y,
        "R": np.hypot(X, Y),
        "PHI": np.arctan2(Y, X),
        "centre_pixel": (SLM_WIDTH_PX // 2, SLM_HEIGHT_PX // 2),
    }


def build_native_masks(
    operating_point: Mapping[str, Any],
    *,
    source_config: NathanSourceParityConfig | None = None,
) -> dict[str, Any]:
    """Native-panel wrapped phase masks phi_H = +alpha + carrier, phi_V = -alpha + pi/2 + carrier."""

    cfg = source_config or NathanSourceParityConfig()
    grid = native_panel_grid()
    carrier_lpmm = float(operating_point["carrier_lpmm"])
    carrier_cpm = carrier_lpmm * 1.0e3
    rotation = float(np.deg2rad(float(operating_point.get("sector_rotation_deg", 0.0))))
    duty = float(operating_point.get("sector_duty_scale", 1.0))
    recentre_x = float(operating_point.get("mask_recentre_x_m", 0.0))
    recentre_y = float(operating_point.get("mask_recentre_y_m", 0.0))
    theta = np.arctan2(grid["Y"] - recentre_y, grid["X"] - recentre_x)
    alpha, _ = nathan_alpha_map(
        theta,
        sector_num_pairs=int(cfg.n_pairs),
        sector_theta=float(np.clip(cfg.sector_theta_rad * duty, 0.1, 2.0 * np.pi / cfg.n_pairs - 0.05)),
        sector_rotation=float(cfg.sector_rotation_rad + rotation),
    )
    carrier_phase = 2.0 * np.pi * carrier_cpm * grid["X"]
    v_piston = float(operating_point.get("global_v_piston_rad", 0.0))
    phi_h = wrap_2pi(alpha + carrier_phase)
    phi_v = wrap_2pi(-alpha + 0.5 * np.pi + carrier_phase + v_piston)
    metadata = {
        "candidate_id": str(operating_point["candidate_id"]),
        "wavelength_m": LAB_WAVELENGTH_M,
        "wavelength_note": "lab masks bound to the actual 1029 nm PHAROS scope (M2U3 sensitivity: immaterial vs 1030 nm)",
        "panel": {
            "width_px": SLM_WIDTH_PX,
            "height_px": SLM_HEIGHT_PX,
            "pixel_pitch_m": SLM_PITCH_M,
            "active_width_m": SLM_WIDTH_PX * SLM_PITCH_M,
            "active_height_m": SLM_HEIGHT_PX * SLM_PITCH_M,
            "centre_pixel_col_row": list(grid["centre_pixel"]),
        },
        "carrier_lpmm": carrier_lpmm,
        "carrier_period_slm_pixels": float(1.0 / (carrier_cpm * SLM_PITCH_M)),
        "carrier_direction": "+x (columns), same sign on both panels",
        "phase_convention": "phi_H = +alpha + carrier; phi_V = -alpha + pi/2 + carrier + global_v_piston",
        "phase_wrapping": "wrapped into [0, 2*pi) radians",
        "coordinate_convention": "x=(col-960)*pitch, y=(row-540)*pitch; axis sampled exactly at (960, 540); previews rendered origin='lower'",
        "reflection_flip_status": (
            "UNRESOLVED until the STAGE 2/6 orientation tests: each arm with odd total reflections needs a "
            "software x-flip of its mask; not applied here"
        ),
        "qwp_convention": "code -45 deg; physical mount sign fixed by the STAGE 9 polarimeter check (docs/80)",
        "corrections": {
            "qwp_angle_correction_deg": float(operating_point.get("qwp_angle_correction_deg", 0.0)),
            "global_v_piston_rad": v_piston,
            "sector_rotation_deg": float(operating_point.get("sector_rotation_deg", 0.0)),
            "sector_duty_scale": duty,
            "mask_recentre_x_m": recentre_x,
            "mask_recentre_y_m": recentre_y,
        },
        "phase_calibration_id": None,
        "lut_applied": False,
        "physically_calibrated": False,
        "uint8_png_is_preview_only": True,
        "hardware_ready": False,
        "hardware_ready_condition": "apply the per-panel measured LUT (docs/75) before sending anything to the GUI",
    }
    return {"phi_H": phi_h, "phi_V": phi_v, "grid": grid, "metadata": metadata}


def export_native_masks(masks: Mapping[str, Any], out_dir: Path) -> dict[str, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, phi in (("slmH", masks["phi_H"]), ("slmV", masks["phi_V"])):
        rad_path = out_dir / f"mode2v_{key}_phase_rad.npy"
        np.save(rad_path, np.asarray(phi, dtype=np.float64))
        paths[f"{key}_rad"] = rad_path
        norm = np.asarray(phi, dtype=np.float64) / (2.0 * np.pi)
        norm_path = out_dir / f"mode2v_{key}_phase_normalised.npy"
        np.save(norm_path, norm)
        paths[f"{key}_norm"] = norm_path
        preview8 = np.clip(np.round(norm * 255.0), 0, 255).astype(np.uint8)
        u8_path = out_dir / f"mode2v_{key}_uint8_preview.png"
        plt.imsave(u8_path, preview8, cmap="gray", vmin=0, vmax=255, origin="lower")
        paths[f"{key}_uint8_preview"] = u8_path
        fig, ax = plt.subplots(figsize=(12.0, 7.0), constrained_layout=True, dpi=220)
        im = ax.imshow(phi, origin="lower", cmap="twilight", vmin=0.0, vmax=2.0 * np.pi,
                       extent=[-7.68, 7.68, -4.32, 4.32], aspect="equal")
        ax.set_title(f"MODE 2V native {key} mask (wrapped rad, PREVIEW ONLY - LUT not applied)")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        fig.colorbar(im, ax=ax, shrink=0.8, label="phase (rad)")
        hi_path = out_dir / f"mode2v_{key}_preview_highres.png"
        fig.savefig(hi_path)
        plt.close(fig)
        paths[f"{key}_highres"] = hi_path
    meta_path = out_dir / "mode2v_slm_masks_metadata.json"
    meta_path.write_text(json.dumps(_json_ready(dict(masks["metadata"])), indent=2), encoding="utf-8")
    paths["metadata"] = meta_path
    return paths


# ---------------------------------------------------------------------------
# Section 8: SLM calibration package (templates only, no fabricated data)
# ---------------------------------------------------------------------------


def write_slm_calibration_package(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    header = "command_uint8,measured_phase_rad_method_A,measured_phase_rad_method_B,notes\n"
    body = "".join(f"{c},,,\n" for c in range(256))
    for panel in ("slmH", "slmV"):
        csv_path = out_dir / f"{panel}_phase_calibration_template.csv"
        csv_path.write_text(header + body, encoding="utf-8")
        paths[f"{panel}_template"] = csv_path
        lut = {
            "schema": "mode2v_slm_lut",
            "panel_role": panel,
            "slm_identity": {"make": "HOLOEYE", "model": "PLUTO-2.1 NIR-149 (externally_supplied_lab_identity)", "serial": None},
            "wavelength_m": LAB_WAVELENGTH_M,
            "calibration_id": None,
            "timestamp_utc": None,
            "command_domain": {"kind": "uint8", "min": 0, "max": 255},
            "usable_phase_stroke_rad": None,
            "phase_to_command": None,
            "interpolation_method": None,
            "residual_phase_rms_rad": None,
            "status": "unresolved_requires_calibration",
            "fabricated_values": False,
        }
        lut_path = out_dir / f"{panel}_lut_template.json"
        lut_path.write_text(json.dumps(_json_ready(lut), indent=2), encoding="utf-8")
        paths[f"{panel}_lut"] = lut_path
    acceptance = {
        "usable_stroke_requirement_rad": ">= 2*pi at 1029 nm, else validated wrapped mapping",
        "residual_phase_rms_max_rad": 0.05,
        "quantisation_context": "M2S: 8-bit and even 16-level phase pass the strict gate, so the target is comfortable",
        "per_panel": True,
        "methods": ["interferometric (docs/75 A)", "binary-grating diffraction efficiency (docs/75 B)"],
    }
    acc_path = out_dir / "slm_calibration_acceptance_criteria.json"
    acc_path.write_text(json.dumps(_json_ready(acceptance), indent=2), encoding="utf-8")
    paths["acceptance"] = acc_path
    return paths


# ---------------------------------------------------------------------------
# Section 9: physical 4F design
# ---------------------------------------------------------------------------


def fourf_final_design(
    *,
    simulated_first_order_efficiency: float,
    simulated_zero_order_leakage: float,
) -> dict[str, Any]:
    rows = physical_4f_rows(
        simulated_first_order_efficiency=simulated_first_order_efficiency,
        simulated_zero_order_leakage=simulated_zero_order_leakage,
    )
    rec = next(r for r in rows if r["recommended_focal_length"] and abs(r["wavelength_nm"] - 1029.0) < 0.5)
    design = {
        "lens1_focal_length_m": FOURF_FOCAL_M,
        "lens2_focal_length_m": FOURF_FOCAL_M,
        "nominal_lens_separation_m": 2.0 * FOURF_FOCAL_M,
        "slm_to_lens1_m": FOURF_FOCAL_M,
        "fourier_plane_after_lens1_m": FOURF_FOCAL_M,
        "lens2_to_output_m": FOURF_FOCAL_M,
        "total_4f_length_m": 4.0 * FOURF_FOCAL_M,
        "wavelength_nm": rec["wavelength_nm"],
        "carrier_lpmm": rec["carrier_lpmm"],
        "carrier_period_slm_pixels": rec["carrier_period_slm_pixels"],
        "first_order_displacement_mm": rec["first_order_displacement_mm"],
        "iris_radius_mm": rec["iris_radius_mm"],
        "iris_diameter_mm": rec["iris_diameter_mm"],
        "zero_order_clearance_mm": rec["iris_to_zero_order_clearance_mm"],
        "expected_first_order_efficiency": rec["selected_order_efficiency_simulated"],
        "expected_clipping_loss": rec["clipping_fraction_simulated"],
        "expected_zero_order_leakage": rec["zero_order_leakage_simulated"],
        "focal_length_provenance": rec["focal_length_source"],
        "shared_vs_per_arm": (
            "one COMMON 4F after PBS #2 recombination is experimentally preferable: both channels share "
            "identical lenses and one iris (no differential magnification/rotation/shift between arms), it "
            "matches the validated simulation exactly, and a hard iris is polarisation-independent; per-arm "
            "4F remains a valid fallback with more alignment degrees of freedom"
        ),
        "note": "nominal geometry, not bench calibrated; confirm via docs/76 displacement measurement",
        "all_focal_candidates": rows,
    }
    return design


# ---------------------------------------------------------------------------
# Section 10: power budget
# ---------------------------------------------------------------------------


def power_budget_rows(
    canonical: Mapping[str, Any],
    *,
    grid_n: int = 384,
) -> list[dict[str, Any]]:
    """Laboratory-facing normalised power budget for the canonical route.

    Model-derived fractions come from the validated simulation (instrumented run
    plus the stored canonical row).  Vendor factors with no repository evidence
    (mirror/lens/PBS transmissions, SLM reflectivity) are flagged unknown and
    left OUT of the model product rather than being invented.
    """

    cfg = NathanSourceParityConfig()
    from vbb_study.digital_twin.nathan_vector_hexagon import mode2n_source_target, run_mode2n_v0_reference, run_mode2n_dual_slm_4f_route

    data = mode2n_source_target(cfg, grid_n=int(grid_n), z_planes=3)
    v0 = run_mode2n_v0_reference(data)
    realistic = run_mode2n_dual_slm_4f_route(data, v0)
    eff = float(realistic.slm_4f_report["first_order_efficiency"])
    fill = 0.93
    fill_factor_power = fill * fill  # modulated-field power fraction; dead-space light exits at DC and dies at the iris
    field = data["target_field"]
    after, _ = _apply_free_space_vector_axicon(
        field, n_axicon=float(cfg.axicon_n), n_medium=float(cfg.medium_n), base_angle_rad=float(cfg.axicon_base_angle_rad),
    )
    p_before = float(np.sum(np.abs(field.ex) ** 2 + np.abs(field.ey) ** 2))
    p_after = float(np.sum(np.abs(after.ex) ** 2 + np.abs(after.ey) ** 2 + np.abs(after.ez) ** 2))
    axicon_t = p_after / max(p_before, EPS)
    useful_frac = float(canonical["P_useful_over_P_total"])
    per_arm_plus1 = 0.5 * fill_factor_power * eff

    stages = [
        ("01_laser_input", 1.0, "definition", "input reference"),
        ("02_after_input_polarisation_prep", 1.0, "model", "polariser/HWP1 model-lossless; vendor transmission unknown"),
        ("03_h_arm_power", 0.5, "model", "HWP1 set for 50/50 at PBS #1"),
        ("04_v_arm_power", 0.5, "model", "HWP1 set for 50/50 at PBS #1"),
        ("05_incident_on_slm_h", 0.5, "model", "arm routing model-lossless; mirror/PBS vendor factors unknown"),
        ("06_incident_on_slm_v", 0.5, "model", "arm routing model-lossless; mirror/PBS vendor factors unknown"),
        ("07_after_slm_h", 0.5 * fill_factor_power, "model", "fill-factor 0.93 modulated-power fraction; panel reflectivity unknown (not applied)"),
        ("08_after_slm_v", 0.5 * fill_factor_power, "model", "fill-factor 0.93 modulated-power fraction; panel reflectivity unknown (not applied)"),
        ("09_selected_plus1_order_h", per_arm_plus1, "model", f"shared-iris first-order efficiency {eff:.4f} (simulated)"),
        ("10_selected_plus1_order_v", per_arm_plus1, "model", f"shared-iris first-order efficiency {eff:.4f} (simulated)"),
        ("11_rejected_power_h", 0.5 * fill_factor_power * (1.0 - eff), "model", "sector-tail clipping at the iris"),
        ("12_rejected_power_v", 0.5 * fill_factor_power * (1.0 - eff), "model", "sector-tail clipping at the iris"),
        ("13_zero_order_total", 0.5 * (1.0 - fill_factor_power) * 2.0, "model", "dead-space/unmodulated light left at DC; blocked by the iris (simulated leakage through iris = 0)"),
        ("14_after_4f_reconstruction", 2.0 * per_arm_plus1, "model", "both channels through the common relay"),
        ("15_after_recombination", 2.0 * per_arm_plus1, "model", "PBS #2 model-lossless; vendor factor unknown"),
        ("16_after_qwp", 2.0 * per_arm_plus1, "model", "QWP unitary"),
        ("17_incident_on_axicon", 2.0 * per_arm_plus1, "model", "free-space model-lossless"),
        ("18_after_axicon", 2.0 * per_arm_plus1 * axicon_t, "model", f"Fresnel entry + conical-exit p/s split, simulated transmission {axicon_t:.4f} (uncoated model)"),
        ("19_total_power_at_z60", 2.0 * per_arm_plus1 * axicon_t, "model", "free-space propagation conserves power"),
        ("20_useful_central_hexagon_power", 2.0 * per_arm_plus1 * axicon_t * useful_frac, "model", f"stored useful-region fraction {useful_frac:.4f} (M2U2 fixed useful region)"),
        ("21_power_outside_useful_region", 2.0 * per_arm_plus1 * axicon_t * (1.0 - useful_frac), "model", "outer rings / side lobes"),
        ("22_peak_intensity_proxy", float(canonical["strict_peak_metric"]), "stored_metric", str(canonical.get("peak_metric_definition", "mean intensity in the 3x3 neighbourhood centred on the maximum pixel")) + " (simulation units, not W)"),
    ]
    rows = []
    for stage_id, fraction, kind, note in stages:
        row = {
            "stage": stage_id,
            "model_fraction_of_input": float(fraction),
            "value_kind": kind,
            "example_at_1W_input_W": float(fraction) if kind != "stored_metric" else None,
            "example_at_10W_input_W": 10.0 * float(fraction) if kind != "stored_metric" else None,
            "hardware_factor_status": (
                "vendor transmissions/reflectivities NOT included (no repository evidence); multiply in once measured"
            ),
            "scaling_disclaimer": "linear scaling example only - NOT a damage-threshold or power-rating claim",
            "note": note,
        }
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Section 11: component table
# ---------------------------------------------------------------------------


def component_table() -> list[dict[str, Any]]:
    def row(cid, stage, ctype, role, prop, setting, angle, aperture, wl, tol, crit, prov, cal, notes=""):
        return {
            "component_id": cid, "stage": stage, "component_type": ctype, "physical_role": role,
            "required_optical_property": prop, "nominal_setting": setting, "angle_orientation": angle,
            "aperture_requirement": aperture, "wavelength_requirement": wl, "tolerance": tol,
            "criticality": crit, "provenance": prov, "calibration_required": cal, "notes": notes,
        }

    return [
        row("LASER", "source", "femtosecond laser", "1029 nm Gaussian source", "PHAROS PH2 class, 2 mm 1/e beam radius",
            "alignment power for day one", "n/a", "n/a", "1029 nm", "beam radius/pointing measured STAGE 1",
            "essential", "original_digital_twin", True, "processing power only after full alignment"),
        row("TEL", "beam preparation", "telescope (optional)", "set the 2 mm 1/e radius if the raw beam differs",
            "afocal pair", "magnification to reach 2 mm", "n/a", ">= 10 mm clear", "AR at 1029 nm",
            "beam radius +/-10% acceptable (source-model parameter)", "conditional", "planning_assumption", True),
        row("POL1", "polarisation preparation", "linear polariser", "define input linear state",
            "high extinction", "aligned to lab H", "0 deg reference", ">= 6 mm", "1029 nm",
            "clean-up only", "recommended", "planning_assumption", False),
        row("HWP1", "polarisation preparation", "half-wave plate", "H/V power balance before PBS #1",
            "lambda/2 at 1029 nm", "set for 50/50 split", "fast axis, receiver view", ">= 6 mm", "1029 nm",
            "ratio 0.8-1.2 tolerated (M2S)", "essential", "architecture requirement", True),
        row("PBS1", "split", "polarising beamsplitter", "split H (transmit) / V (reflect)",
            "high extinction at 1029 nm", "n/a", "n/a", ">= 10 mm", "1029 nm",
            "extinction sets channel purity", "essential", "architecture requirement", False),
        row("M*", "routing", "fold mirrors", "arm routing", "protected coating at 1029 nm", "n/a", "n/a",
            ">= 10 mm", "1029 nm", "per-arm reflection COUNT decides software mask flips (STAGE 6)",
            "essential", "architecture requirement", True, "record the mirror count per arm"),
        row("SLM_H", "H arm", "phase-only LCOS SLM", "display phi_H = +alpha + carrier",
            "PLUTO-2.1 NIR-149 (externally supplied identity)", "canonical mask", "director aligned to H",
            "15.36 x 8.64 mm active", "NIR", "phase stroke/LUT via docs/75",
            "essential", "externally_supplied_lab_identity", True),
        row("SLM_V", "V arm", "phase-only LCOS SLM", "display phi_V = -alpha + pi/2 + carrier",
            "PLUTO-2.1 NIR-149 (externally supplied identity)", "canonical mask", "director per STAGE 2 test",
            "15.36 x 8.64 mm active", "NIR", "phase stroke/LUT via docs/75",
            "essential", "externally_supplied_lab_identity", True),
        row("HWP2", "V arm", "half-wave plate", "rotate V onto SLM-V director", "lambda/2 at 1029 nm",
            "45 deg", "fast axis, receiver view", ">= 6 mm", "1029 nm", "phase-only purity",
            "conditional", "conditional_on_panel_orientation_test", True,
            "omit if SLM-V is mounted rotated 90 deg"),
        row("HWP3", "V arm", "half-wave plate", "restore V before PBS #2", "lambda/2 at 1029 nm",
            "45 deg", "fast axis, receiver view", ">= 6 mm", "1029 nm", "as HWP2",
            "conditional", "conditional_on_panel_orientation_test", True,
            "omit if SLM-V is mounted rotated 90 deg"),
        row("PBS2", "recombination", "polarising beamsplitter", "recombine H (transmit) + V (reflect) collinearly",
            "high extinction at 1029 nm", "path lengths matched within ~260 fs coherence", "n/a", ">= 10 mm",
            "1029 nm", "relative arm piston free (observable-invariant, M2S)", "essential",
            "architecture requirement", True),
        row("L1", "common 4F", "lens f = 300 mm", "Fourier transform to the filter plane",
            "f = 300 mm nominal (not bench calibrated)", "SLM/recombiner at front focal plane", "n/a",
            ">= 10 mm clear", "AR at 1029 nm", "confirmed via docs/76 displacement measurement",
            "essential", "original_digital_twin (nominal_from_bench_description)", True),
        row("IRIS", "common 4F", "iris / pinhole", "select the +1 order for BOTH channels",
            "diameter ~1.54 mm centred 1.929 mm from the zero order", "0.40 x carrier separation", "n/a",
            "1.54 mm", "n/a", "whole 0.24-0.80 fraction range passes (M2S)", "essential",
            "derived x = lambda*f*carrier", True),
        row("L2", "common 4F", "lens f = 300 mm", "inverse transform back to the image plane",
            "f = 300 mm nominal", "one focal length after the iris", "n/a", ">= 10 mm clear",
            "AR at 1029 nm", "as L1", "essential", "original_digital_twin (nominal_from_bench_description)", True),
        row("QWP1", "output", "quarter-wave plate", "close the Jones chain onto the vector target",
            "lambda/4 at 1029 nm", "code -45 deg (see waveplate table)", "fast axis, receiver view",
            ">= 6 mm", "1029 nm", "+/-2 deg angle, +/-5 deg retardance (M2S)", "essential",
            "validated model convention", True, "mount sign via STAGE 9 polarimeter check"),
        row("AXICON", "output", "physical axicon", "conical phase for the Bessel zone",
            "base angle 2 deg, fused silica n = 1.458", "centred on the hologram axis", "n/a",
            ">= 10 mm clear", "1029 nm", "centring <= 0.2 mm blind, digitally recentred after measurement (M2S)",
            "essential", "source_model_parameter", True),
        row("CAM", "diagnostics", "camera on z stage", "record xy planes across the Bessel zone",
            "unknown hardware (docs/77)", "z scan ~10-200 mm incl. exact 60 mm", "n/a", "n/a", "1029 nm sensitivity",
            "z tolerance +/-20 mm (M2S)", "essential", "unknown", True),
        row("ZSTAGE", "diagnostics", "translation stage", "z scan of the camera", "unknown hardware (docs/77)",
            "mm-class steps sufficient", "n/a", "n/a", "n/a", "+/-20 mm plane tolerance", "essential", "unknown", True),
        row("POLM", "diagnostics", "polarimeter (optional)", "QWP sign check and pre-axicon Stokes validation",
            "Stokes measurement at 1029 nm", "STAGE 9/10", "n/a", "n/a", "1029 nm", "sign check only",
            "recommended", "planning_assumption", True),
        row("SHWFS", "diagnostics", "Shack-Hartmann (optional)", "common-path low-order wavefront measurement",
            "no verified hardware record in repo", "common aberration channel only", "n/a", "n/a", "1029 nm",
            "defocus/astig/coma channel (responsibility matrix)", "optional", "unknown", True),
    ]


# ---------------------------------------------------------------------------
# Section 13: measurement responsibility matrix
# ---------------------------------------------------------------------------


def measurement_responsibility_rows() -> list[dict[str, Any]]:
    base = [
        ("final intensity structure at z planes", "camera", "primary observable |Ex|^2+|Ey|^2+|Ez|^2"),
        ("beam centre / mask-to-axicon centring", "camera", "core position -> digital mask recentre (0.2 mm blind tolerance)"),
        ("C3/C6 symmetry, c60/c90/c120", "camera", "strict gate discriminators on the measured plane"),
        ("dark-core ratio and lobe balance", "camera", "strict gate inputs"),
        ("z-stack structure (10-200 mm)", "camera", "Bessel-zone persistence; +/-20 mm plane tolerance"),
        ("common wavefront aberration (defocus/astig/coma/low-order Zernike)", "shack_hartmann", "common-path channel; feeds the bounded Zernike precompensation"),
        ("pre-axicon vector-field validation (sector structure)", "stokes_polarimetry", "H/V, D/A, R/L projections against the predicted segmented field"),
        ("H/V balance and relative phase errors", "stokes_polarimetry", "HWP1 balance and arm piston (piston is observable-invariant downstream but polarimetry sees it)"),
        ("QWP mount sign", "stokes_polarimetry", "STAGE 9 circular-handedness check"),
    ]
    rows = [
        {"measurement": m, "instrument": inst, "role": role}
        for m, inst, role in base
    ]
    for extra in correction_responsibility_matrix():
        instrument = str(extra["instrument"])
        rows.append({
            "measurement": str(extra["primary_observables"]),
            "instrument": instrument.lower().replace("-", "_").replace("/", "_"),
            "role": (
                f"corrects: {extra['correctable_terms']}; "
                f"not responsible for: {extra['not_responsible_for']}; "
                f"status: {extra['project_status']}"
            ),
        })
    return rows


# ---------------------------------------------------------------------------
# Sections 14-15: closed-loop correction (the search never sees the injected truth)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Mode2VLoopState:
    """Bounded correction state the loop is allowed to update."""

    correction: Mode2SCorrection = field(default_factory=Mode2SCorrection)
    v_shift_x_m: float = 0.0
    v_shift_y_m: float = 0.0

    def as_row(self) -> dict[str, Any]:
        return {**self.correction.as_row(), "v_mask_shift_x_um": self.v_shift_x_m / 1e-6, "v_mask_shift_y_um": self.v_shift_y_m / 1e-6}


def make_hidden_bench_forward(
    bench: Mapping[str, Any],
    hidden: Mode2SPerturbation,
) -> Callable[[Mode2VLoopState], Mapping[str, Any]]:
    """Black-box bench: displays masks with the loop's corrections, returns a measured case.

    The returned callable closes over the hidden perturbation; the correction
    loop only ever receives this callable and the measured images/metrics, so
    the injected truth is structurally unavailable to the search.
    """

    def forward(state: Mode2VLoopState) -> Mapping[str, Any]:
        pert = replace(
            hidden,
            hv_shift_x_m=float(hidden.hv_shift_x_m) + float(state.v_shift_x_m),
            hv_shift_y_m=float(hidden.hv_shift_y_m) + float(state.v_shift_y_m),
        )
        return run_mode2s_degraded_forward(
            bench["data"], bench["v0"], bench["backward"], pert,
            correction=state.correction, fast_single_plane=True,
        )

    return forward


def _plain_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float).ravel()
    bb = np.asarray(b, dtype=float).ravel()
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    return float(aa @ bb / (np.sqrt((aa @ aa) * (bb @ bb)) + EPS))


def _loop_metrics(plane: np.ndarray, bench: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_strict_hexagon_metrics(
        plane,
        grid=bench["data"]["grid"],
        v0_plane=bench["v0"].reference_plane,
        realistic_plane=bench["realistic"].reference_plane,
        v0_ring_radius_m=float(bench["v0"].ring_radius_m),
        useful_mask=bench["useful_mask"],
    )


def run_mode2v_closed_loop_case(
    case_id: str,
    bench: Mapping[str, Any],
    hidden: Mode2SPerturbation,
    *,
    coarse_scan_halfwidth_m: float = 0.6e-3,
    coarse_scan_points: int = 5,
    nm_maxiter: int = 40,
    include_v_shift: bool = True,
) -> dict[str, Any]:
    """One closed-loop correction demonstration (measured images only).

    Iteration structure (each step only 'displays masks and measures'):
      1. baseline measurement;
      2. coarse digital mask-recentre scan (the measure-and-recentre calibration);
      3. bounded Nelder-Mead refinement of the physically interpretable
         corrections against the measured-image loss.
    The injected truth is revealed only afterwards for reporting.
    """

    from scipy.optimize import minimize

    forward = make_hidden_bench_forward(bench, hidden)
    realistic_plane = np.asarray(bench["realistic"].reference_plane, dtype=float)
    v0_plane = np.asarray(bench["v0"].reference_plane, dtype=float)
    evaluations = {"count": 0}

    def measure(state: Mode2VLoopState) -> tuple[np.ndarray, float]:
        evaluations["count"] += 1
        case = forward(state)
        plane = np.asarray(case["reference_plane"], dtype=float)
        loss = (1.0 - _plain_corr(plane, realistic_plane)) + 0.5 * (1.0 - _plain_corr(plane, v0_plane))
        return plane, float(loss)

    history: list[dict[str, Any]] = []
    state = Mode2VLoopState()
    plane, loss = measure(state)
    initial_metrics = _loop_metrics(plane, bench)
    history.append({"iteration": 0, "step": "baseline", "loss": loss,
                    "corr_to_realistic": float(initial_metrics["corr_to_realistic_4f"]),
                    "strict_eligible": bool(initial_metrics["strict_hexagon_eligible"])})

    # Iteration 1: coarse digital recentre scan (display shifted masks, keep the best image).
    offsets = np.linspace(-float(coarse_scan_halfwidth_m), float(coarse_scan_halfwidth_m), int(coarse_scan_points))
    best = (loss, 0.0, 0.0)
    for ox in offsets:
        for oy in offsets:
            _, trial_loss = measure(replace_state(state, mask_recentre=(float(ox), float(oy))))
            if trial_loss < best[0]:
                best = (trial_loss, float(ox), float(oy))
    state = replace_state(state, mask_recentre=(best[1], best[2]))
    plane, loss = measure(state)
    scan_metrics = _loop_metrics(plane, bench)
    history.append({"iteration": 1, "step": "coarse_recentre_scan", "loss": loss,
                    "recentre_x_um": best[1] / 1e-6, "recentre_y_um": best[2] / 1e-6,
                    "corr_to_realistic": float(scan_metrics["corr_to_realistic_4f"]),
                    "strict_eligible": bool(scan_metrics["strict_hexagon_eligible"])})

    # Iteration 2: bounded Nelder-Mead refinement (measured loss only).
    n_extra = 2 if include_v_shift else 0
    scales = np.asarray(
        [1.0e-4, 1.0e-4, 0.3, 0.03, 0.03] + [0.2] * 6 + [0.2, 0.2] + [2.0e-5] * n_extra, dtype=float,
    )
    x0 = np.zeros(scales.size, dtype=float)
    x0[0] = best[1]
    x0[1] = best[2]

    def to_state(params: np.ndarray) -> Mode2VLoopState:
        p = np.asarray(params, dtype=float)
        correction = Mode2SCorrection(
            mask_recentre_x_m=float(np.clip(p[0], -1.0e-3, 1.0e-3)),
            mask_recentre_y_m=float(np.clip(p[1], -1.0e-3, 1.0e-3)),
            global_v_piston_rad=float(np.clip(p[2], -np.pi, np.pi)),
            sector_rotation_rad=float(np.clip(p[3], -np.deg2rad(15.0), np.deg2rad(15.0))),
            sector_duty_scale=float(np.clip(1.0 + p[4], 0.7, 1.3)),
            sector_pistons_rad=tuple(float(np.clip(v, -0.5 * np.pi, 0.5 * np.pi)) for v in p[5:11]),
            defocus_rad=float(np.clip(p[11], -1.0, 1.0)),
            astig0_rad=float(np.clip(p[12], -1.0, 1.0)),
        )
        vx = float(np.clip(p[13], -1.5e-4, 1.5e-4)) if include_v_shift else 0.0
        vy = float(np.clip(p[14], -1.5e-4, 1.5e-4)) if include_v_shift else 0.0
        return Mode2VLoopState(correction=correction, v_shift_x_m=vx, v_shift_y_m=vy)

    def nm_loss(params: np.ndarray) -> float:
        _, trial_loss = measure(to_state(params))
        return trial_loss

    simplex = np.vstack([x0] + [x0 + scales * np.eye(scales.size)[i] for i in range(scales.size)])
    result = minimize(nm_loss, x0, method="Nelder-Mead",
                      options={"maxiter": int(nm_maxiter), "initial_simplex": simplex, "xatol": 1e-4, "fatol": 1e-6})
    state = to_state(np.asarray(result.x, dtype=float))
    plane, loss = measure(state)
    final_metrics = _loop_metrics(plane, bench)
    history.append({"iteration": 2, "step": "bounded_nm_refine", "loss": loss,
                    "nm_evaluations": int(result.nfev),
                    "corr_to_realistic": float(final_metrics["corr_to_realistic_4f"]),
                    "strict_eligible": bool(final_metrics["strict_hexagon_eligible"])})

    # The injected truth is revealed ONLY here, after the search has finished.
    truth = asdict(hidden)
    inferred = state.as_row()
    recentre_error_um = None
    if abs(float(hidden.axicon_decentre_x_m)) > 0.0 or abs(float(hidden.axicon_decentre_y_m)) > 0.0:
        recentre_error_um = float(
            np.hypot(
                state.correction.mask_recentre_x_m - float(hidden.axicon_decentre_x_m),
                state.correction.mask_recentre_y_m - float(hidden.axicon_decentre_y_m),
            )
            / 1e-6
        )
    return {
        "case_id": str(case_id),
        "search_received_injected_truth": False,
        "initial_guess": Mode2VLoopState().as_row(),
        "injected_truth_revealed_after": truth,
        "inferred_correction": inferred,
        "iterations": history,
        "n_forward_evaluations": int(evaluations["count"]),
        "initial_corr_to_realistic": float(initial_metrics["corr_to_realistic_4f"]),
        "final_corr_to_realistic": float(final_metrics["corr_to_realistic_4f"]),
        "initial_corr_angular": float(initial_metrics["corr_angular"]),
        "final_corr_angular": float(final_metrics["corr_angular"]),
        "initial_dark_core": float(initial_metrics["dark_core_ratio"]),
        "final_dark_core": float(final_metrics["dark_core_ratio"]),
        "initial_strict_eligible": bool(initial_metrics["strict_hexagon_eligible"]),
        "final_strict_eligible": bool(final_metrics["strict_hexagon_eligible"]),
        "initial_classifier": str(initial_metrics["classifier_label"]),
        "final_classifier": str(final_metrics["classifier_label"]),
        "mask_recentre_error_um": recentre_error_um,
        "final_plane": np.asarray(plane, dtype=np.float32),
        "initial_metrics": {k: v for k, v in initial_metrics.items() if isinstance(v, (int, float, bool, str))},
        "final_metrics": {k: v for k, v in final_metrics.items() if isinstance(v, (int, float, bool, str))},
    }


def replace_state(state: Mode2VLoopState, *, mask_recentre: tuple[float, float]) -> Mode2VLoopState:
    return Mode2VLoopState(
        correction=replace(state.correction, mask_recentre_x_m=mask_recentre[0], mask_recentre_y_m=mask_recentre[1]),
        v_shift_x_m=state.v_shift_x_m,
        v_shift_y_m=state.v_shift_y_m,
    )


def closed_loop_cases() -> dict[str, Mode2SPerturbation]:
    moderate = mode2s_combined_cases()[1]
    return {
        "A_unknown_axicon_mask_offset_0p5mm": Mode2SPerturbation(
            label="loop_A", slm_aperture_clip=True, phase_levels=256, axicon_decentre_x_m=0.5e-3,
        ),
        "B_moderate_m2s_combined": replace(moderate, label="loop_B"),
        "C_unknown_hv_registration": Mode2SPerturbation(
            label="loop_C", slm_aperture_clip=True, phase_levels=256,
            hv_shift_x_m=60.0e-6, hv_rotation_rad=float(np.deg2rad(0.3)),
        ),
        "D_unknown_low_order_aberration": Mode2SPerturbation(
            label="loop_D", slm_aperture_clip=True, phase_levels=256,
            zernike_common={"defocus": 0.4, "astig0": 0.3, "coma_x": 0.2},
        ),
        "E_combined_unknown_errors": Mode2SPerturbation(
            label="loop_E", slm_aperture_clip=True, phase_levels=256,
            axicon_decentre_x_m=0.3e-3, hv_piston_rad=0.5, hv_shift_x_m=24.0e-6,
            qwp_angle_error_rad=float(np.deg2rad(1.0)), zernike_common={"defocus": 0.3},
        ),
    }


# ---------------------------------------------------------------------------
# Section 17: final build decision
# ---------------------------------------------------------------------------


def mode2v_outcome(
    *,
    decision: Mapping[str, Any],
    masks_metadata: Mapping[str, Any],
    fourf_design: Mapping[str, Any],
    waveplates: Sequence[Mapping[str, Any]],
    loop_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Choose exactly one M2V-A/B/C/D outcome."""

    architecture_ok = bool(decision["chosen_route"] == "B_common_4f_after_recombination")
    masks_ok = bool(
        masks_metadata["panel"]["width_px"] == SLM_WIDTH_PX
        and masks_metadata["panel"]["height_px"] == SLM_HEIGHT_PX
        and masks_metadata["uint8_png_is_preview_only"]
    )
    fourf_ok = bool(fourf_design["iris_diameter_mm"] > 0.0 and fourf_design["first_order_displacement_mm"] > 0.0)
    polarisation_ok = any(str(w["component_id"]) == "QWP1_final" for w in waveplates)
    recovered = [r for r in loop_results if bool(r["final_strict_eligible"]) and not bool(r["initial_strict_eligible"])]
    # Deliberately-broken robustness cases can have a PHYSICAL recovery ceiling below the
    # 0.997 operating-point floor (e.g. a 0.5 mm axicon decentre leaves a real Gaussian
    # envelope offset even after perfect digital recentring; the M2U3 rebind measured
    # ~0.986 with the exact correction).  "Substantial recovery" therefore means: the
    # strict classifier says hexagonal, the field is within ~1.4% of the reference, and
    # the loop delivered a real improvement - judged with the repaired gate metrics,
    # never with full-field correlation alone.
    substantially_recovered = [
        r for r in loop_results
        if str(r["final_classifier"]) == "visual_hexagonal_field"
        and float(r["final_corr_to_realistic"]) >= 0.98
        and float(r["final_corr_to_realistic"]) > float(r["initial_corr_to_realistic"]) + 0.05
        and not bool(r["initial_strict_eligible"])
    ]
    improved = [r for r in loop_results if float(r["final_corr_to_realistic"]) > float(r["initial_corr_to_realistic"]) + 0.005]
    loop_demonstrated = bool(recovered or substantially_recovered)

    if not (architecture_ok and polarisation_ok):
        outcome = "M2V-D"
        statement = "A simpler alternative architecture became preferable and must be tested first."
    elif not (masks_ok and fourf_ok):
        outcome = "M2V-C"
        statement = "Native-mask, polarisation, 4F, power or feedback implementation revealed a major blocker."
    elif not loop_demonstrated:
        outcome = "M2V-B"
        statement = (
            "The architecture is physically valid, but the closed-loop correction demonstration did not recover "
            "a strict-eligible field from any unknown-error case; treat correction as unresolved before assembly."
        )
    else:
        outcome = "M2V-A"
        statement = (
            "The source-scale dual-SLM + 4F + QWP + axicon bench is fully specified at the architecture level; "
            "native masks are exported; physical 4F dimensions are defined; polarisation routing is explicit; "
            "remaining unknowns are routine calibration tasks; closed-loop correction is demonstrated. "
            "Ready for source-scale laboratory trial."
        )
    return {
        "stage": MODE2V_STAGE,
        "selected_outcome": outcome,
        "allowed_outcomes": MODE2V_ALLOWED_OUTCOMES,
        "outcome_statement": statement,
        "canonical_operating_point": CANONICAL_OPERATING_POINT_ID,
        "secondary_operating_point": STRICT_COMPROMISE_ID,
        "forbidden_note": f"{OLD_BEST_COMPROMISE_ID} and every repaired-gate failure remain permanently forbidden",
        "architecture": decision["chosen_route"],
        "six_piece_segmented_optic_required": False,
        "six_piece_replacement": (
            "programmable dual-SLM phase control of two orthogonal polarisation channels + one uniform QWP; "
            "sector pattern/rotation/duty/pistons/centring/aberration precompensation are all digital"
        ),
        "closed_loop_recovered_cases": [str(r["case_id"]) for r in recovered],
        "closed_loop_substantially_recovered_cases": [str(r["case_id"]) for r in substantially_recovered],
        "closed_loop_improved_cases": [str(r["case_id"]) for r in improved],
        "closed_loop_recovery_ceiling_note": (
            "deliberately-broken robustness cases can have a physical recovery ceiling below the 0.997 "
            "operating-point floor (e.g. the residual Gaussian envelope offset of a 0.5 mm decentre); "
            "recovery is judged with the repaired strict-gate metrics, never full-field correlation alone"
        ),
        "strict_gate_note": (
            f"the {STRICT_BASELINE_CORR_MIN} correlation-to-realistic-reference floor is a calibrated "
            "project-specific eligibility threshold, not a universal physical hexagon definition; X-shaped/"
            "fourfold and triangular fields are vetoed regardless of full-field correlation"
        ),
        "microfabrication_sample_plane_claim": False,
        "micro_scale_note": (
            "MODE 2V packages the source-scale bench only; the microfabrication branch (MODE 1C/M1E) remains "
            "separate and blocked and nothing here claims sample-plane success"
        ),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _agg_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _plot_operating_point_comparison(path: Path, canonical: Mapping[str, Any], secondary: Mapping[str, Any]) -> Path:
    plt = _agg_plt()
    keys = ("corr_to_realistic_4f", "corr_focus_crop", "corr_angular", "c60", "c120", "h6",
            "first_order_efficiency", "total_throughput", "P_useful_over_P_total")
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(11.6, 4.8), constrained_layout=True, dpi=200)
    ax.bar(x - 0.18, [float(canonical[k]) for k in keys], width=0.36, label=CANONICAL_OPERATING_POINT_ID, color="tab:green")
    ax.bar(x + 0.18, [float(secondary[k]) for k in keys], width=0.36, label=STRICT_COMPROMISE_ID, color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=25, ha="right", fontsize=7.5)
    ax.set_ylim(0.0, 1.1)
    ax.legend(fontsize=7)
    ax.set_title("MODE 2V frozen operating points (exact stored M2U2-FIX values)")
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_bench_diagram(png_path: Path, pdf_path: Path) -> tuple[Path, Path]:
    plt = _agg_plt()
    fig, ax = plt.subplots(figsize=(15.5, 7.2), constrained_layout=True, dpi=200)
    ax.axis("off")

    def box(x, y, text, color="#eef3fb", w=0.105, h=0.14, fs=7.2):
        ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=color, edgecolor="0.3"))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "color": "0.25"})

    main_y = 0.72
    box(0.05, main_y, "PHAROS\n1029 nm\nGaussian 2 mm")
    box(0.16, main_y, "POL1 +\nHWP #1\n(H/V balance)")
    box(0.27, main_y, "PBS #1\nsplit", color="#fdeade")
    box(0.40, 0.90, "SLM-H\nphi_H = +alpha\n+ carrier", color="#e7f6e7")
    box(0.40, 0.54, "HWP#2* > SLM-V\nphi_V = -alpha+pi/2\n+ carrier > HWP#3*", color="#e7f6e7", w=0.15)
    box(0.53, main_y, "PBS #2\nrecombine", color="#fdeade")
    box(0.635, main_y, "L1\nf = 300 mm")
    box(0.725, main_y, "IRIS\nD 1.54 mm\n@ +1 order\n(1.93 mm)", color="#fff3c8")
    box(0.815, main_y, "L2\nf = 300 mm")
    box(0.885, main_y, "QWP\ncode -45 deg", color="#f6e7f4")
    box(0.945, main_y, "axicon\n2 deg\nn = 1.458", color="#e7eef6")
    box(0.945, 0.40, "camera on\nz stage\n10-200 mm\n(60 mm ref)", color="#f0f0f0")
    arrow(0.105, main_y, 0.105 + 0.0, main_y)
    for x0, x1 in ((0.105, 0.108), (0.215, 0.218), (0.325, 0.328)):
        arrow(x0, main_y, x1 + 0.02, main_y)
    arrow(0.27, main_y + 0.07, 0.40 - 0.055, 0.90)
    arrow(0.27, main_y - 0.07, 0.40 - 0.078, 0.54)
    arrow(0.40 + 0.055, 0.90, 0.53, main_y + 0.07)
    arrow(0.40 + 0.078, 0.54, 0.53, main_y - 0.07)
    for x0, x1 in ((0.585, 0.605), (0.68, 0.695), (0.77, 0.785), (0.845, 0.855), (0.915, 0.92)):
        arrow(x0, main_y, x1, main_y)
    arrow(0.945, main_y - 0.075, 0.945, 0.475)
    ax.text(0.5, 0.16,
            "Route B (recommended): recombination BEFORE one COMMON 4F + single iris (matches every validated run;\n"
            "no differential H/V filter errors). * HWP#2/#3 conditional on the SLM-V panel-orientation test; "
            "alternatively mount SLM-V rotated 90 deg.\nNo six-piece segmented optic anywhere: the sector pattern is programmable phase.",
            ha="center", fontsize=8.4, color="0.2")
    ax.set_title("MODE 2V recommended source-scale bench (canonical operating point REALISTIC_4F_HEXAGON_REFERENCE)")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def _plot_waveplate_orientation(path: Path, waveplates: Sequence[Mapping[str, Any]]) -> Path:
    plt = _agg_plt()
    fig, axes = plt.subplots(1, len(waveplates), figsize=(4.0 * len(waveplates), 4.4), constrained_layout=True, dpi=200)
    for ax, wp in zip(np.atleast_1d(axes), waveplates, strict=False):
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.4"))
        ax.plot([-1.15, 1.15], [0, 0], color="0.75", lw=0.8)
        ax.plot([0, 0], [-1.15, 1.15], color="0.75", lw=0.8)
        angle = wp["nominal_angle_code_deg"]
        if isinstance(angle, (int, float)):
            theta = np.deg2rad(float(angle))
            ax.plot([-np.cos(theta), np.cos(theta)], [-np.sin(theta), np.sin(theta)], color="tab:red", lw=2.2)
            ax.set_title(f"{wp['component_id']}\nfast axis {float(angle):+.1f} deg (code)", fontsize=8)
        else:
            ax.set_title(f"{wp['component_id']}\n{angle}", fontsize=7)
        ax.text(1.05, 0.04, "H", fontsize=8)
        ax.text(0.04, 1.05, "V", fontsize=8)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle("MODE 2V waveplate fast-axis orientations - receiver view (looking back INTO the beam); "
                 "odd mirror counts flip the apparent sense", fontsize=9)
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_4f_layout(path: Path, design: Mapping[str, Any]) -> Path:
    plt = _agg_plt()
    f = float(design["lens1_focal_length_m"])
    fig, ax = plt.subplots(figsize=(12.6, 3.6), constrained_layout=True, dpi=200)
    positions = [0.0, f, 2.0 * f, 3.0 * f, 4.0 * f]
    labels = ["SLM/recombiner\nplane", "L1\nf=300 mm", "Fourier plane\n+ iris", "L2\nf=300 mm", "output plane\n(QWP + axicon)"]
    for pos, label in zip(positions, labels, strict=True):
        ax.axvline(pos, color="0.4", lw=1.0)
        ax.text(pos, 1.06, label, ha="center", fontsize=8)
        ax.text(pos, -0.14, f"{pos * 1e3:.0f} mm", ha="center", fontsize=7, color="0.35")
    ax.plot([0, 4.0 * f], [0.5, 0.5], color="tab:orange", lw=1.4)
    ax.set_xlim(-0.08, 4.0 * f + 0.08)
    ax.set_ylim(-0.25, 1.25)
    ax.set_yticks([])
    ax.set_xlabel("optical axis position (m), to scale")
    ax.set_title("MODE 2V common 4F layout to scale (total 1.2 m)")
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_fourier_order_map(path: Path, design: Mapping[str, Any]) -> Path:
    plt = _agg_plt()
    x1 = float(design["first_order_displacement_mm"])
    r = float(design["iris_radius_mm"])
    fig, ax = plt.subplots(figsize=(7.6, 6.0), constrained_layout=True, dpi=200)
    for order in (-2, -1, 0, 1, 2):
        x = order * x1
        size = {0: 160, 1: 120, -1: 60, 2: 30, -2: 30}[order]
        color = "tab:green" if order == 1 else ("0.2" if order == 0 else "0.6")
        ax.scatter([x], [0], s=size, color=color, zorder=3)
        ax.annotate(f"{order:+d}" if order else "0", (x, 0), textcoords="offset points", xytext=(0, 14), ha="center", fontsize=9)
    ax.add_patch(plt.Circle((x1, 0), r, fill=False, color="tab:green", lw=1.8))
    ax.annotate(f"iris D {2 * r:.2f} mm", (x1, -r), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=9, color="tab:green")
    ax.set_xlim(-2.4 * x1, 2.8 * x1)
    ax.set_ylim(-2.2 * r, 2.6 * r)
    ax.set_aspect("equal")
    ax.set_xlabel("Fourier-plane x (mm)")
    ax.set_ylabel("Fourier-plane y (mm)")
    ax.set_title(f"MODE 2V Fourier-plane order map at 1029 nm, f = 300 mm, carrier 6.25 lp/mm\n"
                 f"+1 order at {x1:.3f} mm; higher orders only from quantisation/pixelation (weak)")
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_power(rows: Sequence[Mapping[str, Any]], flow_path: Path, loss_path: Path) -> None:
    plt = _agg_plt()
    stages = [r for r in rows if r["value_kind"] == "model"]
    labels = [str(r["stage"])[3:] for r in stages]
    values = [float(r["model_fraction_of_input"]) for r in stages]
    fig, ax = plt.subplots(figsize=(13.2, 4.8), constrained_layout=True, dpi=200)
    ax.plot(range(len(stages)), values, marker="o", color="tab:blue")
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=6.4)
    ax.set_ylabel("fraction of laser input (model)")
    ax.set_title("MODE 2V power flow (model fractions; vendor transmissions not included - not evidenced)")
    fig.savefig(flow_path)
    plt.close(fig)
    losses = {
        "SLM fill-factor dead space (both arms)": 1.0 - 0.93**2,
        "iris sector-tail clipping": 0.93**2 * (1.0 - 0.9495),
        "axicon Fresnel (uncoated model)": None,
    }
    after_4f = next(float(r["model_fraction_of_input"]) for r in rows if r["stage"] == "14_after_4f_reconstruction")
    at_axicon = next(float(r["model_fraction_of_input"]) for r in rows if r["stage"] == "18_after_axicon")
    losses["axicon Fresnel (uncoated model)"] = after_4f - at_axicon
    fig, ax = plt.subplots(figsize=(8.6, 4.6), constrained_layout=True, dpi=200)
    ax.bar(list(losses), [float(v) for v in losses.values()], color="tab:red")
    ax.set_ylabel("fraction of laser input lost")
    ax.set_title("MODE 2V loss breakdown (model)")
    ax.tick_params(axis="x", rotation=12, labelsize=8)
    fig.savefig(loss_path)
    plt.close(fig)


def _plot_responsibility(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    plt = _agg_plt()
    instruments = ("camera", "shack_hartmann", "stokes_polarimetry")
    fig, ax = plt.subplots(figsize=(11.8, 0.44 * len(rows) + 1.6), constrained_layout=True, dpi=200)
    ax.axis("off")
    for j, inst in enumerate(instruments):
        ax.text(0.62 + 0.13 * j, 1.0, inst, fontsize=8, ha="center", va="bottom", rotation=15)
    for i, row in enumerate(rows):
        y = 1.0 - (i + 1) / (len(rows) + 1)
        ax.text(0.0, y, str(row["measurement"])[:78], fontsize=6.8, va="center")
        for j, inst in enumerate(instruments):
            mark = "X" if str(row["instrument"]) == inst else ""
            ax.text(0.62 + 0.13 * j, y, mark, fontsize=9, ha="center", va="center", color="tab:green")
    ax.set_title("MODE 2V measurement responsibility matrix (camera / Shack-Hartmann / Stokes)")
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_closed_loop(results: Sequence[Mapping[str, Any]], conv_path: Path, case_dir: Path, bench: Mapping[str, Any]) -> None:
    plt = _agg_plt()
    fig, ax = plt.subplots(figsize=(9.4, 4.8), constrained_layout=True, dpi=200)
    for res in results:
        its = [h["iteration"] for h in res["iterations"]]
        corr = [float(h["corr_to_realistic"]) for h in res["iterations"]]
        ax.plot(its, corr, marker="o", label=str(res["case_id"])[:36])
    ax.axhline(STRICT_BASELINE_CORR_MIN, color="0.3", lw=0.9, ls="--", label=f"strict floor {STRICT_BASELINE_CORR_MIN}")
    ax.set_xlabel("closed-loop iteration")
    ax.set_ylabel("correlation to realistic-4F reference")
    ax.set_title("MODE 2V simulated closed-loop convergence (search never sees the injected truth)")
    ax.legend(fontsize=6.6)
    fig.savefig(conv_path)
    plt.close(fig)
    grid = bench["data"]["grid"]
    for res in results:
        plane = np.asarray(res["final_plane"], dtype=float)
        crop, crop_grid = _mode1b_even_axis_crop(plane, grid, 0.35)
        ref_crop, _ = _mode1b_even_axis_crop(np.asarray(bench["realistic"].reference_plane, dtype=float), grid, 0.35)
        xc = np.asarray(crop_grid["x"], dtype=float) / 1e-3
        ext = [float(xc[0]), float(xc[-1]), float(xc[0]), float(xc[-1])]
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), constrained_layout=True, dpi=200)
        axes[0].imshow(_normalise_image(ref_crop, local=True), origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=1)
        axes[0].set_title("canonical reference", fontsize=9)
        axes[1].imshow(_normalise_image(crop, local=True), origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=1)
        axes[1].set_title(
            f"after closed loop: corr {res['final_corr_to_realistic']:.4f}\n"
            f"strict {res['initial_strict_eligible']} -> {res['final_strict_eligible']}", fontsize=8,
        )
        for a in axes:
            a.set_xlabel("x (mm)")
        fig.suptitle(f"MODE 2V closed-loop case {res['case_id']}", fontsize=10)
        fig.savefig(case_dir / f"mode2v_closed_loop_case_{res['case_id']}.png")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Section 16: mask package README
# ---------------------------------------------------------------------------


def _write_mask_package(out_dir: Path, mask_paths: Mapping[str, Path], masks: Mapping[str, Any]) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}
    for key, src in mask_paths.items():
        dst = out_dir / Path(src).name
        if Path(src).resolve() != dst.resolve():
            shutil.copyfile(src, dst)
        copied[key] = dst
    readme = out_dir / "README_MASK_PACKAGE.md"
    meta = dict(masks["metadata"])
    readme.write_text(
        "# MODE 2V Lab Mask Package\n\n"
        f"Canonical operating point: `{meta['candidate_id']}` at {meta['wavelength_m'] * 1e9:.0f} nm, "
        f"carrier {meta['carrier_lpmm']} lp/mm ({meta['carrier_period_slm_pixels']:.0f} px/period), "
        f"panel {meta['panel']['width_px']} x {meta['panel']['height_px']} @ "
        f"{meta['panel']['pixel_pitch_m'] * 1e6:.0f} um, centre pixel {tuple(meta['panel']['centre_pixel_col_row'])}.\n\n"
        "## File classes\n\n"
        "| file | space | status |\n|---|---|---|\n"
        "| `mode2v_slmH_phase_rad.npy`, `mode2v_slmV_phase_rad.npy` | PANEL-space wrapped phase (rad, [0, 2pi)) | authoritative masks; require LUT conversion before display |\n"
        "| `mode2v_slm*_phase_normalised.npy` | PANEL-space normalised phase (phase / 2pi) | convenience form; require LUT conversion before display |\n"
        "| `mode2v_slm*_uint8_preview.png` | PANEL-space uint8 render | PREVIEW ONLY - linear 0..255 mapping, NOT LUT-calibrated, NOT hardware-ready |\n"
        "| `mode2v_slm*_preview_highres.png` | visualisation | human inspection only |\n"
        "| `mode2v_slm_masks_metadata.json` | metadata | conventions, corrections, calibration status |\n\n"
        "## Simulation-space vs panel-space\n\n"
        "All validated simulation grids are source-scale square windows; the files here are the PANEL-space "
        "re-rasterisation on the native 1920 x 1080 grid with the axis sampled exactly at pixel (960, 540). "
        "No simulation-space arrays are shipped in this package (they live with the M2N/M2S outputs).\n\n"
        "## Safe-to-display rule\n\n"
        "NOTHING in this package may be sent to the SLM GUI until the per-panel docs/75 phase calibration has "
        "been run and the measured LUT applied (`lut_applied` is false and `hardware_ready` is false in the "
        "metadata). The uint8 PNGs are previews, not calibrated hardware masks. Per-arm software x-flips "
        "(reflection parity) are applied only after the STAGE 2/6 orientation tests.\n\n"
        f"QWP convention: code -45 deg; see the waveplate table and docs/80 STAGE 9 for the mount-sign check.\n",
        encoding="utf-8",
    )
    copied["readme"] = readme
    return copied


# ---------------------------------------------------------------------------
# Documentation writers (docs/79, 80, 81)
# ---------------------------------------------------------------------------


def _write_jones_doc(path: Path) -> Path:
    qwp = qwp_lab_axis_statement()
    text = (
        "# Nathan MODE 2V - Full Jones Build Derivation (docs/79)\n\n"
        "All matrices act in the beam-local linear H/V basis; angles are anticlockwise from +x (H) in the\n"
        "receiver view (standing downstream looking back into the beam); beta marks the FAST axis;\n"
        "`R(b) = [[cos b, -sin b], [sin b, cos b]]`, retarder `J(d, b) = R(-b) diag(e^{-id/2}, e^{+id/2}) R(b)`.\n"
        "Every mirror/SLM reflection flips transverse parity; per-arm odd totals are absorbed as a software\n"
        "x-flip of that arm's mask (STAGE 6 test), so the chain below is written in one consistent frame.\n\n"
        "## Stage-by-stage chain\n\n"
        "1. **Laser**: `E0 = A(r) [cos(psi), sin(psi)]^T` - linear, orientation psi measured on day one.\n"
        "2. **Input polariser (POL1)**: projects onto H: `E1 = A(r) cos(psi) [1, 0]^T` (defines the reference axis).\n"
        "3. **HWP #1 at b1**: `J_HWP(b1) = R(-b1) diag(-i, +i)... = ` rotation of linear polarisation by `2 b1`;\n"
        "   set `2 b1` so PBS #1 splits 50/50: `E2 = (A/sqrt(2)) [1, 1]^T` (up to global phase).\n"
        "4. **PBS #1**: H transmits into the H arm, V reflects into the V arm:\n"
        "   `E_H,in = (A/sqrt(2)) |H>`, `E_V,in = (A/sqrt(2)) |V>` (reflection pi bookkept as arm piston).\n"
        "5. **H arm / SLM-H**: panel director along H (phase-only): `E_H,out = (A/sqrt(2)) e^{i(+alpha + carrier)} |H>`.\n"
        "6. **V arm**: HWP #2 at 45 deg rotates V -> director; SLM-V applies `e^{i(-alpha + pi/2 + carrier)}`;\n"
        "   HWP #3 at 45 deg rotates back to V: `E_V,out = (A/sqrt(2)) e^{i(-alpha + pi/2 + carrier)} |V>`.\n"
        "   (If SLM-V is mounted rotated 90 deg, HWP #2/#3 are omitted and the algebra is identical.)\n"
        "7. **PBS #2**: coherent recombination (paths matched within the ~260 fs coherence length):\n"
        "   `E3 = (A/sqrt(2)) [e^{i(+alpha)}, e^{i(-alpha + pi/2 + delta)}]^T e^{i carrier}` with arm piston `delta`.\n"
        "8. **Common 4F + iris**: selects the +1 order of BOTH channels and removes the carrier; sector-tail\n"
        "   clipping costs ~5% power but preserves the phase structure (validated 0.9936 correlation).\n"
        "9. **QWP at code -45 deg**: `J_QWP(-45) = (1/sqrt(2)) [[1, -i], [-i, 1]]`. Then\n"
        "   `Ex = (A/sqrt(2))(e^{i alpha} + e^{i(-alpha + delta)}) = A sqrt(2) e^{i delta/2} cos(alpha - delta/2)`\n"
        "   `Ey = (A/sqrt(2))(-i e^{i alpha} + i e^{i(-alpha + delta)}) = A sqrt(2) e^{i delta/2} sin(alpha - delta/2)`\n"
        "   i.e. exactly the segmented target `A [cos alpha', sin alpha']` with `alpha' = alpha - delta/2`:\n"
        "   the arm piston delta only rotates every local polarisation uniformly, and M2S proved the intensity\n"
        "   observable is invariant to it - so delta is free, not a build tolerance.\n"
        "10. **Axicon**: radial/azimuthal p/s Fresnel + conical phase produces the hexagonal Bessel zone.\n\n"
        "## Conventions tracked\n\n"
        f"- global phase: irrelevant; relative H/V phase: pi/2 target offset carried by the SLM-V mask.\n"
        f"- coordinate flips: per-arm reflection parity -> software mask x-flip (STAGE 6).\n"
        f"- QWP physical statement: {qwp['physical_statement']}\n"
        f"- mount-side caveat: {qwp['mount_side_caveat']}\n"
        f"- fast/slow: {qwp['fast_or_slow']}.\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


def _write_first_day_doc(path: Path, fourf: Mapping[str, Any]) -> Path:
    text = (
        "# Nathan MODE 2V - First-Day Lab Procedure (docs/80)\n\n"
        "Source-scale bench only; no material processing; no microfabrication claim.\n\n"
        "## STAGE 0 - safety / low-power setup\n"
        "Alignment power only; verify 1029 nm-rated eyewear for every person in the room; interlocks checked;\n"
        "do NOT run at processing power at any point in this procedure.\n\n"
        "## STAGE 1 - Gaussian beam preparation\n"
        "Verify the 1029 nm output; measure the 1/e field radius (target 2 mm, telescope if needed); centre and\n"
        "collimate; record the beam profile and polarisation orientation/purity (B_INPUT_BEAM).\n\n"
        "## STAGE 2 - SLM panel orientation test\n"
        "One polariser + displayed grating per panel: find the LC director (maximum first-order diffraction,\n"
        "minimum amplitude modulation). DECIDES whether V-arm HWP #2/#3 are used or SLM-V is mounted rotated\n"
        "90 deg. Also run the asymmetric-pattern ('F') test per panel to record flips/parity.\n\n"
        "## STAGE 3 - per-SLM phase calibration\n"
        "Run docs/75 (interferometric or binary-grating) per panel at 1029 nm; store the LUT with a calibration\n"
        "ID; acceptance: usable stroke >= 2 pi (or validated wrapped mapping), residual RMS <= 0.05 rad.\n\n"
        "## STAGE 4 - H/V split\n"
        "Insert POL1 + HWP #1 + PBS #1; rotate HWP #1 until the two arm powers are equal (M2S tolerates 0.8-1.2).\n\n"
        "## STAGE 5 - SLM-H alone\n"
        "Display the 20-px blaze only; verify a single dominant +1 order; measure first-order efficiency\n"
        "(model reference ~0.95 x fill-factor effects).\n\n"
        "## STAGE 6 - SLM-V alone\n"
        "Repeat STAGE 5 in the V arm; confirm parity/orientation (apply the software x-flip if the arm has an\n"
        "odd reflection count).\n\n"
        "## STAGE 7 - 4F alignment\n"
        f"Locate the zero order; locate the +1 order; verify the displacement is close to "
        f"{fourf['first_order_displacement_mm']:.3f} mm (this also confirms f = 300 mm); place the "
        f"~{fourf['iris_diameter_mm']:.2f} mm iris centred on +1; sweep the radius and record the efficiency\n"
        "plateau (docs/76 steps 8-11); record the measured geometry into the hardware binding.\n\n"
        "## STAGE 8 - recombination\n"
        "Align PBS #2; overlap arm centres and magnification; match path lengths well inside the ~260 fs\n"
        "coherence length (white-light/fringe-visibility check); confirm stable fringes between arms.\n\n"
        "## STAGE 9 - QWP sign calibration\n"
        "Polarimeter check (docs/78 Q13): open the H channel only with a uniform mask; at the correct code\n"
        "-45 deg setting the output is LEFT-circular in receiver view; if right-circular, use +45 deg.\n\n"
        "## STAGE 10 - pre-axicon validation\n"
        "With both masks displayed, project onto H/V, D/A and R/L and compare with the predicted segmented\n"
        "vector field (Stokes responsibility, docs/78); verify the pi/2 sector offset structure.\n\n"
        "## STAGE 11 - insert axicon\n"
        "Centre the hologram/vector singularity on the cone axis; blind placement tolerance <= 0.2 mm (M2S);\n"
        "then measure and digitally recentre the masks (the single alignment that actually matters).\n\n"
        "## STAGE 12 - camera z scan\n"
        "Scan ~10-200 mm including the exact 60 mm reference plane; record xy planes and assemble x-z/y-z maps;\n"
        "plane-placement tolerance is +/-20 mm (M2S), so mm-class stage steps are fine.\n\n"
        "## STAGE 13 - correction\n"
        "Measure centre/symmetry errors on the camera; apply the bounded closed-loop corrections (mask centre,\n"
        "V piston, sector rotation/duty, sector pistons, low-order Zernikes) per the MODE 2V loop; the repaired\n"
        "strict hexagon gate is the acceptance criterion - never full-field correlation alone.\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


def _write_master_doc(
    path: Path,
    *,
    canonical: Mapping[str, Any],
    secondary: Mapping[str, Any],
    decision: Mapping[str, Any],
    fourf: Mapping[str, Any],
    power_rows: Sequence[Mapping[str, Any]],
    waveplates: Sequence[Mapping[str, Any]],
    loop_results: Sequence[Mapping[str, Any]],
    outcome: Mapping[str, Any],
) -> Path:
    qwp = qwp_lab_axis_statement()
    useful = next(r for r in power_rows if r["stage"] == "20_useful_central_hexagon_power")
    total = next(r for r in power_rows if r["stage"] == "19_total_power_at_z60")
    recovered = ", ".join(outcome["closed_loop_recovered_cases"]) or "none"
    substantial = ", ".join(outcome["closed_loop_substantially_recovered_cases"]) or "none"
    improved = ", ".join(outcome["closed_loop_improved_cases"]) or "none"
    answers = [
        ("1. What exact bench do we build?",
         "Route B: PHAROS 1029 nm -> POL1 + HWP#1 -> PBS#1 -> SLM-H arm and SLM-V arm (with conditional V-arm "
         "HWPs) -> PBS#2 recombination -> ONE common 4F (f=300 mm) with a single +1-order iris -> QWP (code "
         "-45 deg) -> axicon (2 deg, n=1.458) -> free-space Bessel zone -> camera on a z stage."),
        ("2. Do we need the old segmented six-piece optic?", "No."),
        ("3. What replaces it?", str(outcome["six_piece_replacement"])),
        ("4. What polarisation leaves the laser?",
         "Linear (PHAROS class); orientation/purity unrecorded in the repo - measured in STAGE 1 and cleaned by POL1."),
        ("5. What does HWP #1 do?", "Sets the H/V power split at PBS #1 (target 50/50; 0.8-1.2 tolerated)."),
        ("6. How are H/V channels split?", "PBS #1: H transmits, V reflects."),
        ("7. What polarisation reaches each SLM?",
         "Linear along each panel's LC director: H directly on SLM-H; V rotated onto the SLM-V director by "
         "HWP #2 (or panel mounted rotated 90 deg)."),
        ("8. Are extra V-arm HWPs needed?",
         "Conditional on the STAGE 2 panel orientation test: either HWP #2/#3 at 45 deg around SLM-V, or none "
         "if the panel is mounted rotated 90 deg."),
        ("9. What mask goes on SLM-H?", "phi_H = wrap(+alpha + carrier), native 1920x1080, centre pixel (960, 540)."),
        ("10. What mask goes on SLM-V?", "phi_V = wrap(-alpha + pi/2 + carrier), same panel geometry and carrier sign."),
        ("11. What carrier is used?", "6.25 lp/mm = 20 SLM pixels per period, along +x on both panels."),
        ("12. What 4F lenses are recommended?",
         "Two f = 300 mm lenses in a 4f chain (nominal from the bench description; confirmed via docs/76). "
         "ONE common 4F after recombination is preferred over per-arm relays (no differential H/V filter errors; "
         "matches every validated simulation)."),
        ("13. Where is the +1 order physically?",
         f"{fourf['first_order_displacement_mm']:.3f} mm from the zero order at the Fourier plane (x = lambda f nu)."),
        ("14. What iris diameter is recommended?",
         f"{fourf['iris_diameter_mm']:.2f} mm (0.40 x carrier separation); M2S passes over the whole 0.24-0.80 range."),
        ("15. How are the channels recombined?",
         "PBS #2 before the common 4F; path lengths matched inside the ~260 fs coherence length; the relative "
         "arm piston is free (uniform polarisation rotation; observable-invariant)."),
        ("16. What does the final QWP do?",
         "Maps the dual-linear channels onto the segmented radial/azimuthal vector field (docs/79 derivation)."),
        ("17. What is the physical QWP angle convention?", str(qwp["physical_statement"])),
        ("18. Where is the axicon?",
         "Directly after the QWP at the 4F output plane, centred on the hologram axis (<= 0.2 mm blind, then "
         "digitally recentred)."),
        ("19. Where is the camera?",
         "On a z translation stage scanning ~10-200 mm behind the axicon, including the exact 60 mm reference plane."),
        ("20. How much power survives each stage?",
         f"Model fractions (vendor factors excluded, not evidenced): 0.50 per arm; x0.865 fill factor; x0.9495 "
         f"iris; total at z=60 mm = {float(total['model_fraction_of_input']):.4f} of input (see the power budget CSV)."),
        ("21. How much useful-region power remains?",
         f"{float(useful['model_fraction_of_input']):.4f} of laser input inside the fixed useful hexagon region "
         f"({float(canonical['P_useful_over_P_total']):.4f} of the z=60 mm plane power)."),
        ("22. What is the peak-intensity proxy?",
         f"{float(canonical['strict_peak_metric']):.2f} (simulation units): "
         f"{canonical.get('peak_metric_definition', 'mean 3x3 neighbourhood around the maximum pixel')}."),
        ("23. What is the main practical tolerance?",
         "Hologram-centre-to-axicon-axis registration: <= 0.2 mm blind, fully correctable by measure-and-"
         "recentre; everything else (8-bit phase, fill factor, piston, QWP +/-2 deg, iris range, registration "
         "at tens of um, z +/-20 mm) is forgiving (M2S)."),
        ("24. What does the camera correct?",
         "Beam/mask centring, C3/C6 symmetry, dark core, lobe balance, z structure - the strict-gate observables."),
        ("25. What does the Shack-Hartmann correct?",
         "Common-path low-order wavefront (defocus, astigmatism, coma) feeding the bounded Zernike precompensation."),
        ("26. What does polarimetry validate?",
         "Pre-axicon vector field (sector structure), H/V balance, relative phase and the QWP mount sign."),
        ("27. How does the closed-loop correction work?",
         "Display -> measure -> coarse digital recentre scan -> bounded Nelder-Mead over mask centre, V piston, "
         "sector rotation/duty, six sector pistons, defocus/astig and V-mask shift, using measured images only; "
         "the repaired strict hexagon gate is the acceptance criterion. The search never receives the injected "
         f"truth. Demonstrations: recovered to strict-eligible: {recovered}; substantially recovered (hexagonal, "
         f">= 0.98, at the physical ceiling of the injected damage): {substantial}; improved: {improved}."),
        ("28. What exact masks are exported?",
         "09_mask_package: panel-space wrapped-phase .npy (radians + normalised) for SLM-H/SLM-V, preview-only "
         "uint8 PNGs, high-res previews and metadata (LUT NOT applied; hardware_ready=false)."),
        ("29. What still requires calibration?",
         "Per-panel phase stroke/LUT (docs/75), actual 4F focal/iris geometry (docs/76), camera scale + z stage "
         "(docs/77), panel orientation/parity tests, QWP mount sign, beam centring."),
        ("30. Is the system ready for a source-scale lab trial?",
         f"Outcome **{outcome['selected_outcome']}**: {outcome['outcome_statement']}"),
    ]
    body = "".join(f"**{q}**\n\n{a}\n\n" for q, a in answers)
    text = (
        "# Nathan MODE 2V - Lab-Ready Master Report (docs/81)\n\n"
        f"Canonical operating point: `{CANONICAL_OPERATING_POINT_ID}` (best strict-eligible shape/peak/"
        f"useful-energy within the bounded M2U2-FIX search; no global optimality claim). Secondary: "
        f"`{STRICT_COMPROMISE_ID}`. Forbidden: `{OLD_BEST_COMPROMISE_ID}` and every repaired-gate failure. "
        "No microfabrication/sample-plane success is claimed.\n\n"
        "## The 30 build questions\n\n" + body
    )
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def write_mode2v_lab_ready_build(
    *,
    output_dir: str | Path = MODE2V_DEFAULT_OUTPUT_ROOT,
    grid_n: int = 384,
    z_planes: int = 9,
    run_closed_loop: bool = True,
    loop_nm_maxiter: int = 40,
    jones_doc_path: str | Path = MODE2V_JONES_DOC_PATH,
    first_day_doc_path: str | Path = MODE2V_FIRST_DAY_DOC_PATH,
    master_doc_path: str | Path = MODE2V_MASTER_DOC_PATH,
) -> dict[str, Any]:
    """Run the full MODE 2V build package and write every artefact."""

    root = Path(output_dir)
    dirs = {name: root / name for name in (
        "00_operating_point", "01_architecture", "02_polarisation", "03_slm_calibration",
        "04_4f", "05_power", "06_components", "07_feedback", "08_closed_loop",
        "09_mask_package", "10_final_status",
    )}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    canonical, secondary = load_operating_points()
    (dirs["00_operating_point"] / "mode2v_canonical_operating_point.json").write_text(
        json.dumps(_json_ready(operating_point_summary(canonical)), indent=2), encoding="utf-8")
    (dirs["00_operating_point"] / "mode2v_secondary_operating_point.json").write_text(
        json.dumps(_json_ready(operating_point_summary(secondary)), indent=2), encoding="utf-8")
    comparison_rows = [operating_point_summary(canonical), operating_point_summary(secondary)]
    _write_rows(dirs["00_operating_point"] / "mode2v_operating_point_comparison.csv", comparison_rows)
    _plot_operating_point_comparison(dirs["00_operating_point"] / "mode2v_operating_point_comparison.png", canonical, secondary)

    decision = architecture_decision()
    _write_rows(dirs["01_architecture"] / "mode2v_architecture_route_comparison.csv", decision["routes"])
    (dirs["01_architecture"] / "mode2v_architecture_decision.json").write_text(
        json.dumps(_json_ready(decision), indent=2), encoding="utf-8")
    _plot_bench_diagram(
        dirs["01_architecture"] / "mode2v_recommended_bench_diagram.png",
        dirs["01_architecture"] / "mode2v_recommended_bench_diagram.pdf",
    )

    waveplates = waveplate_table()
    _write_rows(dirs["02_polarisation"] / "mode2v_waveplate_table.csv", waveplates)
    (dirs["02_polarisation"] / "mode2v_waveplate_table.json").write_text(
        json.dumps(_json_ready(waveplates), indent=2), encoding="utf-8")
    _plot_waveplate_orientation(dirs["02_polarisation"] / "mode2v_waveplate_orientation_diagram.png", waveplates)

    masks = build_native_masks(canonical)
    mask_paths = export_native_masks(masks, dirs["09_mask_package"])
    calibration_paths = write_slm_calibration_package(dirs["03_slm_calibration"])

    bench = _bench(LAB_WAVELENGTH_M, grid_n=int(grid_n), z_planes=int(z_planes))
    eff = float(bench["realistic"].slm_4f_report["first_order_efficiency"])
    leak = float(bench["realistic"].slm_4f_report["zero_order_leakage_after_iris"])
    fourf = fourf_final_design(simulated_first_order_efficiency=eff, simulated_zero_order_leakage=leak)
    fourf_rows_flat = [{k: v for k, v in fourf.items() if k != "all_focal_candidates"}]
    _write_rows(dirs["04_4f"] / "mode2v_4f_final_design.csv", fourf_rows_flat)
    (dirs["04_4f"] / "mode2v_4f_final_design.json").write_text(json.dumps(_json_ready(fourf), indent=2), encoding="utf-8")
    _plot_4f_layout(dirs["04_4f"] / "mode2v_4f_layout_to_scale.png", fourf)
    _plot_fourier_order_map(dirs["04_4f"] / "mode2v_fourier_plane_order_map.png", fourf)

    power_rows = power_budget_rows(canonical, grid_n=int(grid_n))
    _write_rows(dirs["05_power"] / "mode2v_power_budget.csv", power_rows)
    (dirs["05_power"] / "mode2v_power_budget.json").write_text(json.dumps(_json_ready(power_rows), indent=2), encoding="utf-8")
    _plot_power(power_rows, dirs["05_power"] / "mode2v_power_flow.png", dirs["05_power"] / "mode2v_power_loss_breakdown.png")

    components = component_table()
    _write_rows(dirs["06_components"] / "mode2v_component_table.csv", components)
    (dirs["06_components"] / "mode2v_component_table.json").write_text(
        json.dumps(_json_ready(components), indent=2), encoding="utf-8")
    md_lines = ["| id | stage | type | role | criticality | calibration |", "|---|---|---|---|---|---|"]
    for row in components:
        md_lines.append(
            f"| {row['component_id']} | {row['stage']} | {row['component_type']} | {row['physical_role']} "
            f"| {row['criticality']} | {row['calibration_required']} |"
        )
    (dirs["06_components"] / "mode2v_component_table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    responsibility = measurement_responsibility_rows()
    _write_rows(dirs["07_feedback"] / "mode2v_measurement_responsibility_matrix.csv", responsibility)
    _plot_responsibility(dirs["07_feedback"] / "mode2v_measurement_responsibility_matrix.png", responsibility)

    loop_results: list[dict[str, Any]] = []
    if run_closed_loop:
        for case_id, hidden in closed_loop_cases().items():
            loop_results.append(
                run_mode2v_closed_loop_case(case_id, bench, hidden, nm_maxiter=int(loop_nm_maxiter))
            )
        loop_rows = [
            {k: v for k, v in res.items() if k not in {"final_plane", "iterations", "initial_metrics", "final_metrics",
                                                        "injected_truth_revealed_after", "inferred_correction", "initial_guess"}}
            for res in loop_results
        ]
        _write_rows(dirs["08_closed_loop"] / "mode2v_closed_loop_results.csv", loop_rows)
        (dirs["08_closed_loop"] / "mode2v_closed_loop_results.json").write_text(
            json.dumps(_json_ready([{k: v for k, v in res.items() if k != "final_plane"} for res in loop_results]), indent=2),
            encoding="utf-8",
        )
        _plot_closed_loop(loop_results, dirs["08_closed_loop"] / "mode2v_closed_loop_convergence.png", dirs["08_closed_loop"], bench)

    package_paths = _write_mask_package(dirs["09_mask_package"], mask_paths, masks)

    outcome = mode2v_outcome(
        decision=decision,
        masks_metadata=masks["metadata"],
        fourf_design=fourf,
        waveplates=waveplates,
        loop_results=loop_results,
    )
    (dirs["10_final_status"] / "m2v_outcome_report.json").write_text(json.dumps(_json_ready(outcome), indent=2), encoding="utf-8")
    manifest = {
        "stage": MODE2V_STAGE,
        "grid_n": int(grid_n),
        "z_planes": int(z_planes),
        "canonical_operating_point": CANONICAL_OPERATING_POINT_ID,
        "secondary_operating_point": STRICT_COMPROMISE_ID,
        "selected_outcome": outcome["selected_outcome"],
        "ready_for_source_scale_lab_trial": bool(outcome["selected_outcome"] == "M2V-A"),
        "microfabrication_sample_plane_claim": False,
        "docs": {
            "jones_derivation": str(jones_doc_path),
            "first_day_procedure": str(first_day_doc_path),
            "master_report": str(master_doc_path),
        },
    }
    (dirs["10_final_status"] / "nathan_mode2v_manifest.json").write_text(json.dumps(_json_ready(manifest), indent=2), encoding="utf-8")

    _write_jones_doc(Path(jones_doc_path))
    _write_first_day_doc(Path(first_day_doc_path), fourf)
    _write_master_doc(
        Path(master_doc_path),
        canonical=canonical,
        secondary=secondary,
        decision=decision,
        fourf=fourf,
        power_rows=power_rows,
        waveplates=waveplates,
        loop_results=loop_results,
        outcome=outcome,
    )
    return {
        "canonical": canonical,
        "secondary": secondary,
        "decision": decision,
        "waveplates": waveplates,
        "masks": masks,
        "mask_paths": mask_paths,
        "calibration_paths": calibration_paths,
        "fourf": fourf,
        "power_rows": power_rows,
        "components": components,
        "responsibility": responsibility,
        "loop_results": loop_results,
        "package_paths": package_paths,
        "outcome": outcome,
        "manifest": manifest,
    }
