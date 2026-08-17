"""Stage 9A.1 command-domain carrier SLM-mask generation for the first
Fourier-plane calibration session.

The physical SLM pixel pitch and the physical Fourier-plane mapping are NOT yet
calibrated.  Carriers are therefore defined in the **command domain** as signed
cycles across the displayed active SLM width/height, NOT as physical spatial
frequencies (cycles/mm or cycles/m).  The command-domain phase ramp is

    phi_x = 2*pi * N_x * (pixel_x / display_width_pixels)
    phi_y = 2*pi * N_y * (pixel_y / display_height_pixels)
    phi   = wrap(phi_x + phi_y)

SLM1 is flat; SLM2 carries only a wrapped carrier ramp (no vortex, no axicon, no
correction map, no aperture crop).  This stage creates files only; it does not
implement physical 4F propagation, camera physics, or any inverse/correction/AI.

Boundary: ``physical_4f_filter_modelled=False``; ``camera_model_enabled=False``;
``material_model_enabled=False``; ``diagnostic_only=True``;
``final_export_allowed=False``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from vbb_study.equations.holography import wrap_phase_rad, quantize_phase_rad, phase_to_gray

TWOPI = 2.0 * np.pi

STUDY_NAME = "cslm_fourier_carrier_calibration_minimal_v1"
STUDY_STATUS_LABELS = (
    "diagnostic_command_domain_calibration",
    "not_physical_4f_model",
    "not_measured_bench_calibration",
)
CLAIM_BOUNDARY = (
    "command-domain carrier cycles across the displayed SLM area; NOT physical spatial frequency; "
    "no physical 4F model, no camera model, no material model; final_export_allowed=False"
)
PHYSICAL_FREQUENCY_STATUS = "uncalibrated_command_domain"
PHASE_WRAP_CONVENTION = "mod_2pi_0_to_2pi"
DEFAULT_PHASE_RESPONSE_STATUS = "unknown_or_unverified"
DIRECT_CALIBRATION_MODE = "direct_fourier_plane_access"
DIRECT_CAMERA_ACCESS_STATUS = "requires_temporary_fourier_plane_or_conjugate_diagnostic_access"


@dataclass(frozen=True)
class CarrierSweepConfig:
    study_name: str = STUDY_NAME
    command_display_width_pixels: int = 1920
    command_display_height_pixels: int = 1200
    command_display_resolution_source: str = "placeholder_unknown_or_unverified"
    phase_export_format: str = "phase_npy+quantised_npy+gray_png+metadata_json"
    phase_quantisation_levels: int = 256
    phase_response_calibration_status: str = DEFAULT_PHASE_RESPONSE_STATUS
    calibration_mode: str = DIRECT_CALIBRATION_MODE
    camera_plane_label: str = "fourier_plane_or_accessible_equivalent"
    camera_access_status: str = DIRECT_CAMERA_ACCESS_STATUS
    camera_coordinate_status: str = "pixel_only_uncalibrated"
    physical_axicon_state: str = "removed_or_bypassed_not_in_active_path"
    fourier_stop_state: str = "record_actual_state_do_not_assume"
    minimum_pixels_per_carrier_cycle: int = 8
    carrier_cycles: tuple[int, ...] = (-24, -16, -8, 0, 8, 16, 24)
    command_axes: tuple[str, ...] = ("x", "y")
    optional_diagonal_cases: tuple[tuple[int, int], ...] = ((8, 8), (8, -8), (-8, 8), (-8, -8))
    capture_repeats: int = 1
    dark_frame_repeats: int = 5
    flat_reference_repeats: int = 3

    @classmethod
    def demo(cls, **overrides: Any) -> "CarrierSweepConfig":
        """Small command display for figures / notebook / tests (fast, legible)."""
        base = dict(command_display_width_pixels=480, command_display_height_pixels=300,
                    command_display_resolution_source="demo_placeholder_not_measured")
        base.update(overrides)
        return cls(**base)


# ---------------------------------------------------------------------------
# command-domain carrier phase
# ---------------------------------------------------------------------------


def command_carrier_phase(width_px: int, height_px: int, cycles_x: float, cycles_y: float) -> np.ndarray:
    """Return the wrapped command-domain carrier phase map [height, width] (radians)."""
    px = np.arange(int(width_px), dtype=float)
    py = np.arange(int(height_px), dtype=float)
    phi_x = TWOPI * float(cycles_x) * (px / float(width_px))
    phi_y = TWOPI * float(cycles_y) * (py / float(height_px))
    phi = phi_x[None, :] + phi_y[:, None]
    return wrap_phase_rad(phi)


def pixels_per_cycle(width_px: int, height_px: int, cycles_x: float, cycles_y: float) -> dict[str, float]:
    ppx = float("inf") if cycles_x == 0 else float(width_px) / abs(float(cycles_x))
    ppy = float("inf") if cycles_y == 0 else float(height_px) / abs(float(cycles_y))
    return {"x": ppx, "y": ppy, "min": min(ppx, ppy)}


def validate_carrier_sampling(config: CarrierSweepConfig, cycles_x: float, cycles_y: float) -> list[str]:
    """Return sampling issues; a carrier under the min pixels/cycle is rejected (no aliasing)."""
    ppc = pixels_per_cycle(config.command_display_width_pixels,
                           config.command_display_height_pixels, cycles_x, cycles_y)
    issues: list[str] = []
    if ppc["min"] < float(config.minimum_pixels_per_carrier_cycle):
        issues.append(
            f"carrier ({cycles_x},{cycles_y}) gives {ppc['min']:.2f} pixels/cycle < required "
            f"{config.minimum_pixels_per_carrier_cycle}; would alias — reduce carrier cycles or "
            f"increase display resolution.")
    return issues


# ---------------------------------------------------------------------------
# mask construction + export
# ---------------------------------------------------------------------------


def _quant_bits(levels: int) -> int:
    return max(1, int(round(np.log2(int(levels)))))


def build_carrier_mask(
    config: CarrierSweepConfig,
    mask_id: str,
    cycles_x: int,
    cycles_y: int,
    *,
    slm_id: str = "SLM2",
    validate: bool = True,
) -> dict[str, Any]:
    """Build one carrier-only (or flat) mask + SLM-ready quantisation + metadata.

    ``cycles_x == cycles_y == 0`` is a flat reference.  For SLM1 the carrier must
    be zero (SLM1 stays flat).
    """
    if slm_id == "SLM1" and (cycles_x != 0 or cycles_y != 0):
        raise ValueError("SLM1 must remain flat (carrier cycles must be 0).")
    if validate:
        issues = validate_carrier_sampling(config, cycles_x, cycles_y)
        if issues:
            raise ValueError("; ".join(issues))

    w = int(config.command_display_width_pixels)
    h = int(config.command_display_height_pixels)
    phase_rad = command_carrier_phase(w, h, cycles_x, cycles_y)
    bits = _quant_bits(config.phase_quantisation_levels)
    quantised_rad = quantize_phase_rad(phase_rad, bits)
    gray = phase_to_gray(phase_rad, bits=bits)
    checksum = hashlib.sha256(np.ascontiguousarray(gray).tobytes()).hexdigest()

    from vbb_study.digital_twin.calibration_acquisition import utc_now_iso, git_commit_hash
    metadata = {
        "mask_id": mask_id,
        "slm_id": slm_id,
        "command_display_width_pixels": w,
        "command_display_height_pixels": h,
        "carrier_cycles_x": int(cycles_x),
        "carrier_cycles_y": int(cycles_y),
        "pixels_per_cycle": pixels_per_cycle(w, h, cycles_x, cycles_y),
        "phase_wrap_convention": PHASE_WRAP_CONVENTION,
        "quantisation_levels": int(config.phase_quantisation_levels),
        "phase_response_calibration_status": config.phase_response_calibration_status,
        "physical_frequency_status": PHYSICAL_FREQUENCY_STATUS,
        "coordinate_frame": "SLM2_phase_map_frame" if slm_id == "SLM2" else "SLM1_phase_map_frame",
        "export_checksum_sha256": checksum,
        "creation_timestamp_utc": utc_now_iso(),
        "git_commit": git_commit_hash(),
        "is_flat": bool(cycles_x == 0 and cycles_y == 0),
        "contains_vortex": False,
        "contains_axicon": False,
        "contains_correction_map": False,
        "contains_aperture_crop": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "phase_response_note": (
            "the exported grayscale is NOT verified to produce a calibrated 0-2pi response at "
            "1030 nm; phase_response_calibration_status = " + config.phase_response_calibration_status),
    }
    return {"phase_rad": phase_rad, "quantised_rad": quantised_rad, "gray": gray, "metadata": metadata}


def build_carrier_sweep_masks(config: CarrierSweepConfig | None = None) -> list[dict[str, Any]]:
    """Build the full mask set: SLM1 flat, SLM2 flat, x-carriers, y-carriers, diagonals."""
    config = config or CarrierSweepConfig()
    masks: list[dict[str, Any]] = []
    masks.append(build_carrier_mask(config, "slm1_flat", 0, 0, slm_id="SLM1"))
    masks.append(build_carrier_mask(config, "slm2_flat_reference", 0, 0, slm_id="SLM2"))
    if "x" in config.command_axes:
        for n in config.carrier_cycles:
            if n == 0:
                continue
            masks.append(build_carrier_mask(config, f"slm2_carrier_x_{n:+d}", n, 0))
    if "y" in config.command_axes:
        for n in config.carrier_cycles:
            if n == 0:
                continue
            masks.append(build_carrier_mask(config, f"slm2_carrier_y_{n:+d}", 0, n))
    for nx, ny in config.optional_diagonal_cases:
        masks.append(build_carrier_mask(config, f"slm2_carrier_diag_x{nx:+d}_y{ny:+d}", nx, ny))
    return masks


def export_mask(mask: Mapping[str, Any], out_dir: str | Path) -> dict[str, Path]:
    """Export one mask: phase .npy, quantised .npy, grayscale PNG, metadata JSON."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mid = mask["metadata"]["mask_id"]
    paths: dict[str, Path] = {}
    paths["phase_npy"] = out / f"{mid}_phase_rad.npy"
    np.save(paths["phase_npy"], np.asarray(mask["phase_rad"], dtype=np.float32))
    paths["quantised_npy"] = out / f"{mid}_quantised_rad.npy"
    np.save(paths["quantised_npy"], np.asarray(mask["quantised_rad"], dtype=np.float32))
    paths["gray_png"] = out / f"{mid}_gray.png"
    try:
        from PIL import Image
        Image.fromarray(np.asarray(mask["gray"], dtype=np.uint8), mode="L").save(paths["gray_png"])
    except Exception:
        np.save(out / f"{mid}_gray.npy", np.asarray(mask["gray"], dtype=np.uint8))
        paths["gray_png"] = out / f"{mid}_gray.npy"
    paths["metadata_json"] = out / f"{mid}_metadata.json"
    paths["metadata_json"].write_text(json.dumps(mask["metadata"], indent=2), encoding="utf-8")
    return paths


# ---------------------------------------------------------------------------
# study config (work package A)
# ---------------------------------------------------------------------------


def build_carrier_calibration_study(config: CarrierSweepConfig | None = None) -> dict[str, Any]:
    config = config or CarrierSweepConfig()
    return {
        "study_name": config.study_name,
        "study_status": list(STUDY_STATUS_LABELS),
        "claim_boundary": CLAIM_BOUNDARY,
        "slm1_mode": "flat_phase",
        "slm2_mode": "command_domain_carrier_ramp_only",
        "command_display_resolution_source": config.command_display_resolution_source,
        "command_display_width_pixels": config.command_display_width_pixels,
        "command_display_height_pixels": config.command_display_height_pixels,
        "phase_export_format": config.phase_export_format,
        "phase_quantisation_levels": config.phase_quantisation_levels,
        "phase_response_calibration_status": config.phase_response_calibration_status,
        "calibration_mode": config.calibration_mode,
        "camera_plane_label": config.camera_plane_label,
        "camera_access_status": config.camera_access_status,
        "camera_coordinate_status": config.camera_coordinate_status,
        "physical_axicon_state": config.physical_axicon_state,
        "fourier_stop_state": config.fourier_stop_state,
        "carrier_sweep_definition": {
            "carrier_cycles": list(config.carrier_cycles),
            "command_axes": list(config.command_axes),
            "optional_diagonal_cases": [{"x": nx, "y": ny} for nx, ny in config.optional_diagonal_cases],
            "units": "command_cycles_across_displayed_area",
            "physical_frequency_status": PHYSICAL_FREQUENCY_STATUS,
        },
        "minimum_pixels_per_carrier_cycle": config.minimum_pixels_per_carrier_cycle,
        "capture_repeats": config.capture_repeats,
        "dark_frame_repeats": config.dark_frame_repeats,
        "flat_reference_repeats": config.flat_reference_repeats,
        "operator_setup_requirements": [
            "SLM1 displays the flat mask",
            "SLM2 displays only the command-domain carrier mask",
            "physical axicon removed/bypassed/not in active path",
            "temporary diagnostic camera/profiler/IR card/power meter at or conjugate to the Fourier plane",
            "installed downstream final-focus camera is not direct Fourier-plane access",
            "record the Fourier-stop state (do not assume)",
            "do not move camera/SLMs/lenses without logging",
        ],
        "governance": {
            "physical_4f_filter_modelled": False,
            "camera_model_enabled": False,
            "material_model_enabled": False,
            "diagnostic_only": True,
            "final_export_allowed": False,
        },
    }


def load_carrier_calibration_study(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Work package C — first-session acquisition-package generator
# ---------------------------------------------------------------------------


def _carrier_plan_rows(masks, carrier_config: CarrierSweepConfig, run_id: str, data_dir: Path):
    slm1_id = next(m["metadata"]["mask_id"] for m in masks if m["metadata"]["slm_id"] == "SLM1")
    raw = data_dir / "raw"
    rows: list[dict[str, Any]] = []
    for i in range(carrier_config.dark_frame_repeats):
        rows.append({"capture_id": f"dark_{i:02d}", "capture_kind": "dark_frame",
                     "slm1_mask_id": "shuttered", "slm2_mask_id": "shuttered",
                     "carrier_cycles_x": "", "carrier_cycles_y": "", "command_axis": "",
                     "purpose": "background/dark reference", "capture_status": "planned",
                     "raw_target_path": str(raw / f"dark_{i:02d}.png")})
    for i in range(carrier_config.flat_reference_repeats):
        rows.append({"capture_id": f"flat_ref_{i:02d}", "capture_kind": "flat_field",
                     "slm1_mask_id": slm1_id, "slm2_mask_id": "slm2_flat_reference",
                     "carrier_cycles_x": 0, "carrier_cycles_y": 0, "command_axis": "none",
                     "purpose": "zero-carrier reference order position", "capture_status": "planned",
                     "raw_target_path": str(raw / f"flat_ref_{i:02d}.png")})
    for m in masks:
        meta = m["metadata"]
        if meta["slm_id"] != "SLM2" or meta["is_flat"]:
            continue
        nx, ny = meta["carrier_cycles_x"], meta["carrier_cycles_y"]
        axis = "x" if ny == 0 else ("y" if nx == 0 else "diagonal")
        for r in range(carrier_config.capture_repeats):
            cid = f"{meta['mask_id']}_r{r}"
            rows.append({"capture_id": cid, "capture_kind": "fourier_plane_carrier_sweep",
                         "slm1_mask_id": slm1_id, "slm2_mask_id": meta["mask_id"],
                         "carrier_cycles_x": nx, "carrier_cycles_y": ny, "command_axis": axis,
                         "purpose": "observe zero/+1/-1 order position vs command carrier",
                         "capture_status": "planned",
                         "raw_target_path": str(raw / f"{cid}.png")})
    return rows


def create_fourier_carrier_calibration_session(
    run_id: str | None = None,
    *,
    config=None,
    carrier_config: CarrierSweepConfig | None = None,
    output_root: str | Path = "outputs/calibration_runs",
    data_root: str | Path = "data/calibration_runs",
    repo: str | Path | None = None,
    save_atlas_to: str | Path | None = "outputs/figures/digital_twin/stage9a1_command_domain_carrier_mask_atlas.png",
) -> dict[str, Any]:
    """Generate the first Fourier-plane carrier-calibration session pack (files only)."""
    from vbb_study.digital_twin.calibration_acquisition import (
        generate_run_id, utc_now_iso, git_commit_hash, _write_csv, _write_json,
        _CAPTURE_MANIFEST_COLUMNS,
    )
    from vbb_study.digital_twin.cslm_route import CSLMRouteConfig
    from vbb_study.digital_twin.control_contract import build_default_demo_profile
    from vbb_study.digital_twin.bench_inventory import build_bench_inventory_profile
    from vbb_study.digital_twin.coordinate_contract import coordinate_frame_rows, coordinate_transform_rows

    config = config or CSLMRouteConfig()
    carrier_config = carrier_config or CarrierSweepConfig()
    run_id = run_id or generate_run_id("fourcal")
    out_dir = Path(output_root) / run_id
    data_dir = Path(data_root) / run_id
    if out_dir.exists() or data_dir.exists():
        raise FileExistsError(f"calibration run already exists: {run_id}")
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    exp_dir = out_dir / "experiment_package"; exp_dir.mkdir(parents=True, exist_ok=True)
    slm1_dir = out_dir / "phase_masks" / "slm1"; slm1_dir.mkdir(parents=True, exist_ok=True)
    slm2_dir = out_dir / "phase_masks" / "slm2"; slm2_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("raw", "manifests", "derived", "figures"):
        (data_dir / sub).mkdir(parents=True, exist_ok=True)

    masks = build_carrier_sweep_masks(carrier_config)
    mask_export: dict[str, Any] = {}
    for m in masks:
        target = slm1_dir if m["metadata"]["slm_id"] == "SLM1" else slm2_dir
        mask_export[m["metadata"]["mask_id"]] = {
            k: str(v) for k, v in export_mask(m, target).items()}

    plan_rows = _carrier_plan_rows(masks, carrier_config, run_id, data_dir)
    profile = build_default_demo_profile(config)

    paths: dict[str, Any] = {"run_dir": out_dir, "data_dir": data_dir,
                             "experiment_package_dir": exp_dir,
                             "phase_masks_slm1_dir": slm1_dir, "phase_masks_slm2_dir": slm2_dir,
                             "mask_export": mask_export}

    run_manifest = {
        "run_id": run_id, "timestamp_utc": utc_now_iso(), "git_commit": git_commit_hash(repo),
        "study_name": carrier_config.study_name,
        "study_status": list(STUDY_STATUS_LABELS),
        "profile_name": profile["profile_name"], "profile_status": profile["profile_status"],
        "route_mode": config.route_mode, "order_handoff_mode": config.order_handoff_mode,
        "slm1_mode": "flat_phase", "slm2_mode": "command_domain_carrier_ramp_only",
        "command_display_width_pixels": carrier_config.command_display_width_pixels,
        "command_display_height_pixels": carrier_config.command_display_height_pixels,
        "command_display_resolution_source": carrier_config.command_display_resolution_source,
        "physical_axicon_state": carrier_config.physical_axicon_state,
        "fourier_stop_state": carrier_config.fourier_stop_state,
        "carrier_sweep_definition": build_carrier_calibration_study(carrier_config)["carrier_sweep_definition"],
        "physical_frequency_status": PHYSICAL_FREQUENCY_STATUS,
        "governance": {"physical_4f_filter_modelled": False, "camera_model_enabled": False,
                       "material_model_enabled": False, "diagnostic_only": True,
                       "final_export_allowed": False},
        "claim_boundary": CLAIM_BOUNDARY,
        "raw_data_dir": str(data_dir),
        "raw_data_policy": "raw camera files immutable; not committed by default",
    }
    paths["run_manifest"] = out_dir / "run_manifest.json"; _write_json(paths["run_manifest"], run_manifest)

    paths["acquisition_plan"] = out_dir / "acquisition_plan.csv"
    _write_csv(paths["acquisition_plan"],
               ["capture_id", "capture_kind", "slm1_mask_id", "slm2_mask_id", "carrier_cycles_x",
                "carrier_cycles_y", "command_axis", "purpose", "capture_status", "raw_target_path"],
               plan_rows)
    paths["capture_manifest_template"] = out_dir / "capture_manifest_template.csv"
    _write_csv(paths["capture_manifest_template"], _CAPTURE_MANIFEST_COLUMNS,
               [{"capture_id": r["capture_id"], "capture_kind": r["capture_kind"], "run_id": run_id,
                 "capture_status": "planned", "camera_frame_id": "camera_sensor_pixel_frame",
                 "image_units": "pixel", "profile_name": profile["profile_name"],
                 "route_mode": config.route_mode, "order_handoff_mode": config.order_handoff_mode,
                 "slm1_mask_id": r["slm1_mask_id"], "slm2_mask_id": r["slm2_mask_id"],
                 "carrier_frequency_cpm": ""} for r in plan_rows])

    paths["hardware_profile_snapshot"] = out_dir / "hardware_profile_snapshot.json"
    _write_json(paths["hardware_profile_snapshot"], profile)
    paths["bench_inventory_snapshot"] = out_dir / "bench_inventory_snapshot.json"
    _write_json(paths["bench_inventory_snapshot"], build_bench_inventory_profile(config))
    paths["coordinate_contract_snapshot"] = out_dir / "coordinate_contract_snapshot.json"
    _write_json(paths["coordinate_contract_snapshot"],
                {"frames": coordinate_frame_rows(), "transforms": coordinate_transform_rows()})

    _write_first_session_experiment_package(exp_dir, run_id, run_manifest, plan_rows, carrier_config)

    # mask atlas figure (in the run package and in the shared figures dir)
    fig = plot_command_domain_carrier_mask_atlas(masks, carrier_config,
                                                 output_path=out_dir / "figures" / "command_domain_carrier_mask_atlas.png")
    import matplotlib.pyplot as plt
    if save_atlas_to is not None:
        fig.savefig(Path(save_atlas_to), dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["mask_atlas_figure"] = out_dir / "figures" / "command_domain_carrier_mask_atlas.png"
    return paths


def _write_first_session_experiment_package(exp_dir, run_id, run_manifest, plan_rows, carrier_config):
    from vbb_study.digital_twin.calibration_acquisition import _write_csv, _FUSED_SILICA_FIELDS  # noqa: F401
    (exp_dir / "LAB_README_FIRST_FOURIER_SESSION.md").write_text(
        _LAB_README.format(
            run_id=run_id, ts=run_manifest["timestamp_utc"], commit=run_manifest["git_commit"],
            width=carrier_config.command_display_width_pixels,
            height=carrier_config.command_display_height_pixels,
            cycles=list(carrier_config.carrier_cycles),
            darks=carrier_config.dark_frame_repeats,
            flats=carrier_config.flat_reference_repeats,
            claim=run_manifest["claim_boundary"]), encoding="utf-8")
    (exp_dir / "bench_setup_sheet.md").write_text(
        f"# Bench setup — first Fourier carrier session {run_id}\n\n"
        f"- SLM1: flat phase\n- SLM2: command-domain carrier ramp only\n"
        f"- physical axicon: {carrier_config.physical_axicon_state}\n"
        f"- camera: {carrier_config.camera_plane_label}\n"
        f"- Fourier stop: {carrier_config.fourier_stop_state}\n\n"
        f"Claim boundary: {run_manifest['claim_boundary']}\n", encoding="utf-8")
    _write_csv(exp_dir / "bench_setup_sheet.csv",
               ["component", "setting", "value", "unit", "provenance", "notes"],
               [{"component": "SLM1", "setting": "mode", "value": "flat", "provenance": "derived"},
                {"component": "SLM2", "setting": "mode", "value": "command_carrier_only",
                 "provenance": "derived"},
                {"component": "physical_axicon", "setting": "state",
                 "value": carrier_config.physical_axicon_state, "provenance": "derived"}])
    _write_csv(exp_dir / "camera_capture_checklist.csv",
               ["capture_id", "capture_kind", "slm2_mask_id", "acquired", "file_name", "exposure_us",
                "gain", "saturation_ok", "stop_state", "visible_orders", "operator_initials", "notes"],
               [{"capture_id": r["capture_id"], "capture_kind": r["capture_kind"],
                 "slm2_mask_id": r["slm2_mask_id"]} for r in plan_rows])
    _write_csv(exp_dir / "carrier_sweep_log.csv",
               ["capture_id", "command_axis", "carrier_cycles_x", "carrier_cycles_y",
                "zero_order_x_px", "zero_order_y_px", "plus1_x_px", "plus1_y_px",
                "minus1_x_px", "minus1_y_px", "saturation", "clipping", "notes"],
               [{"capture_id": r["capture_id"], "command_axis": r["command_axis"],
                 "carrier_cycles_x": r["carrier_cycles_x"], "carrier_cycles_y": r["carrier_cycles_y"]}
                for r in plan_rows if r["capture_kind"] == "fourier_plane_carrier_sweep"])
    _write_csv(exp_dir / "fourier_plane_observation_template.csv",
               ["capture_id", "zero_order_visible", "plus1_visible", "minus1_visible",
                "order_separation_px", "order_movement_direction", "multiple_unexpected_orders",
                "asymmetry_or_rotation", "stop_state", "notes"], [])
    (exp_dir / "operator_notes_template.md").write_text(
        f"# Operator notes — {run_id}\n\n- date:\n- operator:\n- conditions:\n\n"
        "## Neutral observations (no material predictions, no aberration estimates)\n\n",
        encoding="utf-8")


_LAB_README = """# LAB README — First Fourier-Plane Carrier Calibration Session

Run: {run_id}
Timestamp: {ts}
Git commit: {commit}

Carrier values are **command-domain cycles across the displayed SLM area**
({width} x {height} command pixels). They are NOT physical spatial-frequency
values until SLM geometry and Fourier-plane calibration are recorded.

## Beam-path state
- SLM1: flat phase
- SLM2: command-domain carrier phase only
- physical axicon: remove / bypass / not in active path
- camera: Fourier-plane or accessible equivalent diagnostic plane
- Fourier stop: record the actual state, do not assume

## Session procedure
1. Record profile/run ID and physical bench state.
2. Capture {darks} dark frames using the intended exposure/gain.
3. Display SLM1-flat / SLM2-flat and capture the zero-carrier reference ({flats} repeats).
4. Run the command-x carrier sequence (cycles {cycles}).
5. Run the command-y carrier sequence (same cycles).
6. Run optional diagonal carrier masks if time permits.
7. For each capture, record exposure, gain, camera location, visible orders,
   stop state, and any clipping/saturation.
8. Do not move the camera, SLMs, or lens train without logging it.
9. Keep raw files unchanged and store them under this run ID's raw/ directory.
10. Complete the capture manifest before leaving the lab.

## What to look for
zero-order position; +1 and -1 order positions where visible; order movement
direction under sign reversal; order separation versus command carrier cycles;
saturation; aperture clipping; unexpected multiple orders; asymmetry or rotation.

## Explicit limitations
- This session does not validate physical 4F propagation.
- This session does not estimate aberrations.
- This session does not create a correction map.
- This session does not predict a fused-silica outcome.

Claim boundary: {claim}
"""


# ---------------------------------------------------------------------------
# Work package F — command-domain carrier mask atlas figure
# ---------------------------------------------------------------------------


def plot_command_domain_carrier_mask_atlas(masks, carrier_config: CarrierSweepConfig | None = None,
                                           *, output_path=None, dpi: int = 150, max_side: int = 240):
    """Atlas of the carrier masks (phase colourbars, command-pixel axes)."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    carrier_config = carrier_config or CarrierSweepConfig()
    n = len(masks)
    ncol = 6
    nrow = (n + ncol - 1) // ncol
    fig = plt.figure(figsize=(2.6 * ncol, 2.7 * nrow + 1.0), facecolor="white")
    gs = fig.add_gridspec(nrow, ncol, left=0.04, right=0.97,
                          top=1.0 - 0.7 / (2.7 * nrow + 1.0), bottom=0.04, hspace=0.45, wspace=0.30)
    fig.suptitle("Stage 9A.1 Command-Domain Carrier Mask Atlas  (SLM1 flat; SLM2 carrier only)\n"
                 "carrier = signed cycles across displayed command area; NOT physical spatial "
                 "frequency; phase_response_calibration_status=" + carrier_config.phase_response_calibration_status
                 + "; diagnostic_only; final_export_allowed=False",
                 x=0.04, y=0.995, ha="left", va="top", fontsize=11, fontweight="bold")
    for i, m in enumerate(masks):
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        ph = np.asarray(m["phase_rad"], dtype=float)
        step_y = max(1, ph.shape[0] // max_side); step_x = max(1, ph.shape[1] // max_side)
        disp = ph[::step_y, ::step_x]
        meta = m["metadata"]
        im = ax.imshow(disp, origin="lower", cmap="twilight", vmin=0, vmax=TWOPI,
                       extent=(0, meta["command_display_width_pixels"], 0, meta["command_display_height_pixels"]))
        ax.set_title(f"{meta['mask_id']}\n({meta['slm_id']}) x={meta['carrier_cycles_x']:+d} "
                     f"y={meta['carrier_cycles_y']:+d}", fontsize=7.5, fontweight="bold")
        ax.set_xlabel("command x (px)", fontsize=6.5); ax.set_ylabel("command y (px)", fontsize=6.5)
        ax.tick_params(labelsize=6)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=[0, np.pi, TWOPI])
        cb.ax.set_yticklabels(["0", "pi", "2pi"], fontsize=6)
    if output_path is not None:
        out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight", metadata={
            "Title": "Stage 9A.1 command-domain carrier mask atlas", "final_export_allowed": "False"})
    return fig
