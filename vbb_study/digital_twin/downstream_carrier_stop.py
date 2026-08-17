"""Stage 9A.1B downstream-focus carrier/stop characterisation.

The installed lab camera is treated as a downstream final-focus/output-plane
camera, not as direct Fourier-plane access. This module creates documentation,
schemas, masks, and session packages only. It does not implement physical 4F
propagation, camera imaging, inverse correction, AI, or material response.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.digital_twin.bench_inventory import (
    build_bench_inventory_profile,
    downstream_empirical_carrier_stop_evidence_effect,
)
from vbb_study.digital_twin.calibration_acquisition import (
    _write_csv,
    _write_json,
    generate_run_id,
    git_commit_hash,
    utc_now_iso,
)
from vbb_study.digital_twin.coordinate_contract import coordinate_frame_rows, coordinate_transform_rows
from vbb_study.digital_twin.cslm_route import CSLMRouteConfig
from vbb_study.digital_twin.control_contract import build_default_demo_profile
from vbb_study.digital_twin.measured_image_metrics import compute_measured_image_metrics
from vbb_study.digital_twin.slm_calibration_masks import (
    CarrierSweepConfig,
    build_carrier_sweep_masks,
    export_mask,
    plot_command_domain_carrier_mask_atlas,
)


STAGE = "9A.1B"
STUDY_NAME = "cslm_carrier_stop_characterisation_downstream_v1"
CALIBRATION_MODES = ("direct_fourier_plane_access", "downstream_focus_empirical")
DEFAULT_CALIBRATION_MODE = "downstream_focus_empirical"
DOWNSTREAM_CAMERA_WARNING = (
    "With the installed downstream camera, this experiment characterises the final "
    "response to carrier and stop settings. It does not directly measure Fourier-plane "
    "order positions or by itself calibrate the physical 4F coordinate system."
)
CLAIM_BOUNDARY = (
    "empirical downstream response only; not_direct_fourier_plane_calibration; "
    "not_physical_4f_model_validation; no camera model; no inverse correction; "
    "no AI; no material response; final_export_allowed=False"
)

AXICON_STATE_CHOICES = (
    "axicon_bypassed",
    "axicon_present_fixed",
    "axicon_removed",
    "unknown_recorded",
)

CAPTURE_FAMILIES = (
    "D0_dark_frames",
    "D1_flat_reference",
    "D2_carrier_sweep_fixed_stop",
    "D3_stop_position_sweep_fixed_carrier",
    "D4_stop_radius_or_aperture_sweep_fixed_carrier",
    "D5_repeatability_reference",
)

DOWNSTREAM_CAPTURE_MANIFEST_COLUMNS = (
    "capture_id",
    "capture_family",
    "calibration_mode",
    "camera_plane_label",
    "camera_plane_relationship_to_fourier_plane",
    "physical_axicon_state",
    "downstream_optics_state",
    "fourier_stop_state",
    "fourier_stop_centre_command_or_stage_x",
    "fourier_stop_centre_command_or_stage_y",
    "fourier_stop_radius_command_or_aperture_label",
    "carrier_cycles_x",
    "carrier_cycles_y",
    "SLM1 mask ID",
    "SLM2 mask ID",
    "camera position",
    "exposure",
    "gain",
    "neutral-density/filter state",
    "laser energy setting where available",
    "manual notes",
    "metadata_value_status",
    "raw_target_path",
)

OPERATING_POINT_BOUNDARY_LABELS = (
    "empirical_downstream_operating_point",
    "not_direct_fourier_plane_calibration",
    "not_physical_4f_model_validation",
)


@dataclass(frozen=True)
class DownstreamCarrierStopConfig:
    study_name: str = STUDY_NAME
    calibration_mode: str = DEFAULT_CALIBRATION_MODE
    camera_plane_label: str = "downstream_final_focus"
    camera_plane_relationship_to_fourier_plane: str = (
        "not_at_fourier_plane; installed camera records complete downstream optical response"
    )
    fourier_stop_state: str = "recorded_and_user_editable"
    physical_axicon_state: str = "recorded_and_user_editable"
    default_physical_axicon_recorded_state: str = "unknown_recorded"
    downstream_optics_state: str = "recorded_fixed_existing_route"
    fixed_stop_centre_x: str = "baseline_recorded"
    fixed_stop_centre_y: str = "baseline_recorded"
    fixed_stop_radius_or_aperture_label: str = "baseline_recorded"
    stop_centre_sweep_offsets: tuple[str, ...] = ("negative_small", "baseline", "positive_small")
    stop_radius_or_aperture_sweep_labels: tuple[str, ...] = ("smaller", "baseline", "larger")
    selected_carrier_cycles_x: int = 8
    selected_carrier_cycles_y: int = 0
    carrier_cycles: tuple[int, ...] = (-24, -16, -8, 0, 8, 16, 24)
    command_axes: tuple[str, ...] = ("x", "y")
    command_display_width_pixels: int = 1920
    command_display_height_pixels: int = 1200
    command_display_resolution_source: str = "placeholder_unknown_or_unverified"
    phase_quantisation_levels: int = 256
    phase_response_calibration_status: str = "unknown_or_unverified"
    dark_frame_repeats: int = 5
    flat_reference_repeats: int = 3
    capture_repeats: int = 1
    repeatability_repeats: int = 3
    minimum_pixels_per_carrier_cycle: int = 8

    @classmethod
    def demo(cls, **overrides: Any) -> "DownstreamCarrierStopConfig":
        base = {
            "command_display_width_pixels": 480,
            "command_display_height_pixels": 300,
            "command_display_resolution_source": "demo_placeholder_not_measured",
            "carrier_cycles": (-8, 0, 8),
            "dark_frame_repeats": 2,
            "flat_reference_repeats": 1,
            "repeatability_repeats": 3,
        }
        base.update(overrides)
        return cls(**base)

    def carrier_config(self) -> CarrierSweepConfig:
        return CarrierSweepConfig(
            command_display_width_pixels=self.command_display_width_pixels,
            command_display_height_pixels=self.command_display_height_pixels,
            command_display_resolution_source=self.command_display_resolution_source,
            phase_quantisation_levels=self.phase_quantisation_levels,
            phase_response_calibration_status=self.phase_response_calibration_status,
            camera_plane_label=self.camera_plane_label,
            physical_axicon_state=self.default_physical_axicon_recorded_state,
            fourier_stop_state=self.fourier_stop_state,
            minimum_pixels_per_carrier_cycle=self.minimum_pixels_per_carrier_cycle,
            carrier_cycles=self.carrier_cycles,
            command_axes=self.command_axes,
            optional_diagonal_cases=(),
            capture_repeats=self.capture_repeats,
            dark_frame_repeats=self.dark_frame_repeats,
            flat_reference_repeats=self.flat_reference_repeats,
        )


def build_calibration_access_modes() -> dict[str, dict[str, Any]]:
    """Return the two allowed calibration-access modes."""
    return {
        "direct_fourier_plane_access": {
            "mode_id": "direct_fourier_plane_access",
            "measurement_plane_label": "Fourier_plane_or_conjugate_diagnostic_plane",
            "measurement_plane_relationship_to_fourier_plane": (
                "camera, profiler, IR card, or power meter temporarily at or conjugate to the Fourier plane"
            ),
            "camera_access_status": "not_current_installed_camera_default; temporary diagnostic access required",
            "what_is_observable": [
                "observed zero/+1/-1 order positions in camera pixels",
                "carrier sign convention at Fourier plane",
                "carrier command-to-order displacement in camera pixels",
                "future physical Fourier-plane calibration once camera scale is known",
            ],
            "what_is_not_observable": [
                "material response",
                "physical 4F propagation validation",
                "camera-model parameters",
                "selected-order purity without separate power/order evidence",
            ],
            "physical_4f_readiness_effect": (
                "can contribute to physical_fourier_plane_coordinate_calibrated only after scale and geometry are recorded"
            ),
            "claim_boundary": "direct diagnostic access mode; still no physical 4F model or final export",
            "required_capture_metadata": list(DOWNSTREAM_CAPTURE_MANIFEST_COLUMNS),
        },
        "downstream_focus_empirical": {
            "mode_id": "downstream_focus_empirical",
            "measurement_plane_label": "downstream_final_focus",
            "measurement_plane_relationship_to_fourier_plane": (
                "not at the Fourier plane; records final response of the fixed downstream optical route"
            ),
            "camera_access_status": "current_installed_camera_default",
            "what_is_observable": [
                "final output centroid in camera pixels",
                "final output morphology",
                "relative transmitted intensity",
                "saturation/clipping flags",
                "empirical sensitivity to carrier settings",
                "empirical sensitivity to stop x/y/radius settings",
                "identification of repeatable usable operating points",
            ],
            "what_is_not_observable": [
                "direct Fourier-plane order positions",
                "physical Fourier-plane x/y coordinates",
                "direct stop radius in Fourier-plane mm",
                "direct order-power fractions at the stop",
                "physical 4F readiness being marked READY solely from downstream images",
            ],
            "physical_4f_readiness_effect": (
                "supports practical operating-point selection, repeatability assessment, and later comparison; "
                "does not make physical_4f_readiness_ready"
            ),
            "claim_boundary": CLAIM_BOUNDARY,
            "required_capture_metadata": list(DOWNSTREAM_CAPTURE_MANIFEST_COLUMNS),
        },
    }


def build_downstream_carrier_stop_study(config: DownstreamCarrierStopConfig | None = None) -> dict[str, Any]:
    config = config or DownstreamCarrierStopConfig()
    return {
        "study_name": config.study_name,
        "stage": STAGE,
        "calibration_mode": config.calibration_mode,
        "available_calibration_modes": list(CALIBRATION_MODES),
        "calibration_access_modes": build_calibration_access_modes(),
        "camera_plane_label": config.camera_plane_label,
        "camera_plane_relationship_to_fourier_plane": config.camera_plane_relationship_to_fourier_plane,
        "fourier_stop_state": config.fourier_stop_state,
        "physical_axicon_state": config.physical_axicon_state,
        "allowed_physical_axicon_recorded_states": list(AXICON_STATE_CHOICES),
        "default_physical_axicon_recorded_state": config.default_physical_axicon_recorded_state,
        "downstream_optics_state": config.downstream_optics_state,
        "capture_families": [
            {"family_id": "D0_dark_frames", "purpose": "dark/background frames at intended exposure/gain"},
            {"family_id": "D1_flat_reference", "purpose": "SLM1-flat / SLM2-flat downstream output reference"},
            {"family_id": "D2_carrier_sweep_fixed_stop", "purpose": "carrier sweep with one recorded fixed stop baseline"},
            {"family_id": "D3_stop_position_sweep_fixed_carrier", "purpose": "empirical stop-centre x/y sensitivity at one carrier"},
            {"family_id": "D4_stop_radius_or_aperture_sweep_fixed_carrier", "purpose": "empirical stop-size/aperture sensitivity at one carrier"},
            {"family_id": "D5_repeatability_reference", "purpose": "repeat apparent best configuration at least three times"},
        ],
        "carrier_sweep_definition": {
            "carrier_cycles": list(config.carrier_cycles),
            "command_axes": list(config.command_axes),
            "units": "command_cycles_across_displayed_area",
            "physical_frequency_status": "uncalibrated_command_domain",
        },
        "fixed_stop_baseline": {
            "fourier_stop_centre_command_or_stage_x": config.fixed_stop_centre_x,
            "fourier_stop_centre_command_or_stage_y": config.fixed_stop_centre_y,
            "fourier_stop_radius_command_or_aperture_label": config.fixed_stop_radius_or_aperture_label,
        },
        "selected_carrier_for_stop_sweeps": {
            "carrier_cycles_x": config.selected_carrier_cycles_x,
            "carrier_cycles_y": config.selected_carrier_cycles_y,
        },
        "required_capture_metadata": list(DOWNSTREAM_CAPTURE_MANIFEST_COLUMNS),
        "warning": DOWNSTREAM_CAMERA_WARNING,
        "claim_boundary": CLAIM_BOUNDARY,
        "governance": {
            "physical_4f_filter_modelled": False,
            "fourier_filter_physics_available": False,
            "camera_model_enabled": False,
            "material_model_enabled": False,
            "inverse_correction_enabled": False,
            "ai_enabled": False,
            "diagnostic_only": True,
            "final_export_allowed": False,
        },
    }


def _base_capture_row(
    *,
    capture_id: str,
    family: str,
    config: DownstreamCarrierStopConfig,
    slm1_mask_id: str,
    slm2_mask_id: str,
    carrier_cycles_x: Any,
    carrier_cycles_y: Any,
    raw_target_path: Path,
    stop_x: Any | None = None,
    stop_y: Any | None = None,
    stop_radius: Any | None = None,
) -> dict[str, Any]:
    return {
        "capture_id": capture_id,
        "capture_family": family,
        "calibration_mode": config.calibration_mode,
        "camera_plane_label": config.camera_plane_label,
        "camera_plane_relationship_to_fourier_plane": config.camera_plane_relationship_to_fourier_plane,
        "physical_axicon_state": config.default_physical_axicon_recorded_state,
        "downstream_optics_state": config.downstream_optics_state,
        "fourier_stop_state": config.fourier_stop_state,
        "fourier_stop_centre_command_or_stage_x": stop_x if stop_x is not None else config.fixed_stop_centre_x,
        "fourier_stop_centre_command_or_stage_y": stop_y if stop_y is not None else config.fixed_stop_centre_y,
        "fourier_stop_radius_command_or_aperture_label": (
            stop_radius if stop_radius is not None else config.fixed_stop_radius_or_aperture_label
        ),
        "carrier_cycles_x": carrier_cycles_x,
        "carrier_cycles_y": carrier_cycles_y,
        "SLM1 mask ID": slm1_mask_id,
        "SLM2 mask ID": slm2_mask_id,
        "camera position": "record_at_capture",
        "exposure": "record_at_capture",
        "gain": "record_at_capture",
        "neutral-density/filter state": "record_at_capture",
        "laser energy setting where available": "record_if_available",
        "manual notes": "",
        "metadata_value_status": "unknown_values_allowed_but_status_recorded",
        "raw_target_path": str(raw_target_path),
    }


def downstream_capture_plan_rows(
    config: DownstreamCarrierStopConfig | None = None,
    *,
    data_dir: str | Path = "data/calibration_runs/placeholder",
) -> list[dict[str, Any]]:
    config = config or DownstreamCarrierStopConfig()
    masks = build_carrier_sweep_masks(config.carrier_config())
    slm1_flat = next(m["metadata"]["mask_id"] for m in masks if m["metadata"]["slm_id"] == "SLM1")
    raw = Path(data_dir) / "raw"
    rows: list[dict[str, Any]] = []
    for i in range(config.dark_frame_repeats):
        rows.append(
            _base_capture_row(
                capture_id=f"D0_dark_{i:02d}",
                family="D0_dark_frames",
                config=config,
                slm1_mask_id="shuttered",
                slm2_mask_id="shuttered",
                carrier_cycles_x="",
                carrier_cycles_y="",
                raw_target_path=raw / f"D0_dark_{i:02d}.png",
            )
        )
    for i in range(config.flat_reference_repeats):
        rows.append(
            _base_capture_row(
                capture_id=f"D1_flat_reference_{i:02d}",
                family="D1_flat_reference",
                config=config,
                slm1_mask_id=slm1_flat,
                slm2_mask_id="slm2_flat_reference",
                carrier_cycles_x=0,
                carrier_cycles_y=0,
                raw_target_path=raw / f"D1_flat_reference_{i:02d}.png",
            )
        )
    for mask in masks:
        meta = mask["metadata"]
        if meta["slm_id"] != "SLM2" or meta["is_flat"]:
            continue
        for rep in range(config.capture_repeats):
            rows.append(
                _base_capture_row(
                    capture_id=f"D2_{meta['mask_id']}_r{rep}",
                    family="D2_carrier_sweep_fixed_stop",
                    config=config,
                    slm1_mask_id=slm1_flat,
                    slm2_mask_id=meta["mask_id"],
                    carrier_cycles_x=meta["carrier_cycles_x"],
                    carrier_cycles_y=meta["carrier_cycles_y"],
                    raw_target_path=raw / f"D2_{meta['mask_id']}_r{rep}.png",
                )
            )
    for axis in ("x", "y"):
        for offset in config.stop_centre_sweep_offsets:
            rows.append(
                _base_capture_row(
                    capture_id=f"D3_stop_{axis}_{offset}",
                    family="D3_stop_position_sweep_fixed_carrier",
                    config=config,
                    slm1_mask_id=slm1_flat,
                    slm2_mask_id=f"slm2_carrier_x_{config.selected_carrier_cycles_x:+d}",
                    carrier_cycles_x=config.selected_carrier_cycles_x,
                    carrier_cycles_y=config.selected_carrier_cycles_y,
                    stop_x=offset if axis == "x" else config.fixed_stop_centre_x,
                    stop_y=offset if axis == "y" else config.fixed_stop_centre_y,
                    raw_target_path=raw / f"D3_stop_{axis}_{offset}.png",
                )
            )
    for label in config.stop_radius_or_aperture_sweep_labels:
        rows.append(
            _base_capture_row(
                capture_id=f"D4_stop_aperture_{label}",
                family="D4_stop_radius_or_aperture_sweep_fixed_carrier",
                config=config,
                slm1_mask_id=slm1_flat,
                slm2_mask_id=f"slm2_carrier_x_{config.selected_carrier_cycles_x:+d}",
                carrier_cycles_x=config.selected_carrier_cycles_x,
                carrier_cycles_y=config.selected_carrier_cycles_y,
                stop_radius=label,
                raw_target_path=raw / f"D4_stop_aperture_{label}.png",
            )
        )
    for i in range(config.repeatability_repeats):
        rows.append(
            _base_capture_row(
                capture_id=f"D5_repeat_best_{i:02d}",
                family="D5_repeatability_reference",
                config=config,
                slm1_mask_id=slm1_flat,
                slm2_mask_id=f"slm2_carrier_x_{config.selected_carrier_cycles_x:+d}",
                carrier_cycles_x=config.selected_carrier_cycles_x,
                carrier_cycles_y=config.selected_carrier_cycles_y,
                raw_target_path=raw / f"D5_repeat_best_{i:02d}.png",
            )
        )
    return rows


def validate_downstream_capture_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    issues: list[str] = []
    required = {
        "calibration_mode",
        "camera_plane_label",
        "camera_plane_relationship_to_fourier_plane",
        "physical_axicon_state",
        "downstream_optics_state",
        "fourier_stop_state",
        "carrier_cycles_x",
        "carrier_cycles_y",
        "SLM1 mask ID",
        "SLM2 mask ID",
        "camera position",
        "exposure",
        "gain",
        "neutral-density/filter state",
        "manual notes",
    }
    for row in rows:
        cid = row.get("capture_id", "<missing>")
        for field in required:
            if field not in row:
                issues.append(f"{cid}: missing field {field}")
        if row.get("calibration_mode") == "downstream_focus_empirical":
            blob = json.dumps(row).lower()
            if "cycles/mm" in blob or "cycles_per_mm" in blob:
                issues.append(f"{cid}: downstream carrier values must remain command-domain")
    return issues


def create_downstream_carrier_stop_characterisation_session(
    run_id: str | None = None,
    *,
    config: DownstreamCarrierStopConfig | None = None,
    route_config: CSLMRouteConfig | None = None,
    output_root: str | Path = "outputs/calibration_runs",
    data_root: str | Path = "data/calibration_runs",
    repo: str | Path | None = None,
    save_overview_to: str | Path | None = None,
) -> dict[str, Path]:
    config = config or DownstreamCarrierStopConfig()
    route_config = route_config or CSLMRouteConfig()
    run_id = run_id or generate_run_id("downstream")
    out_dir = Path(output_root) / run_id
    data_dir = Path(data_root) / run_id
    if out_dir.exists() or data_dir.exists():
        raise FileExistsError(f"calibration run already exists: {run_id}")
    for path in (
        out_dir / "figures",
        out_dir / "phase_masks" / "slm1",
        out_dir / "phase_masks" / "slm2",
        out_dir / "experiment_package",
        data_dir / "raw",
        data_dir / "manifests",
        data_dir / "derived",
        data_dir / "figures",
    ):
        path.mkdir(parents=True, exist_ok=True)

    carrier_config = config.carrier_config()
    masks = build_carrier_sweep_masks(carrier_config)
    for mask in masks:
        target = out_dir / "phase_masks" / ("slm1" if mask["metadata"]["slm_id"] == "SLM1" else "slm2")
        export_mask(mask, target)
    rows = downstream_capture_plan_rows(config, data_dir=data_dir)
    profile = build_default_demo_profile(route_config)
    bench_profile = build_bench_inventory_profile(route_config)
    readiness_effect = downstream_empirical_carrier_stop_evidence_effect(available=False)

    manifest = {
        "run_id": run_id,
        "timestamp_utc": utc_now_iso(),
        "git_commit": git_commit_hash(repo),
        "stage": STAGE,
        "study_name": config.study_name,
        "calibration_mode": config.calibration_mode,
        "camera_plane_label": config.camera_plane_label,
        "camera_plane_relationship_to_fourier_plane": config.camera_plane_relationship_to_fourier_plane,
        "physical_axicon_state": config.physical_axicon_state,
        "allowed_physical_axicon_recorded_states": list(AXICON_STATE_CHOICES),
        "default_physical_axicon_recorded_state": config.default_physical_axicon_recorded_state,
        "downstream_optics_state": config.downstream_optics_state,
        "fourier_stop_state": config.fourier_stop_state,
        "capture_families": list(CAPTURE_FAMILIES),
        "carrier_sweep_definition": build_downstream_carrier_stop_study(config)["carrier_sweep_definition"],
        "calibration_access_modes": build_calibration_access_modes(),
        "downstream_empirical_readiness_effect": readiness_effect,
        "claim_boundary": CLAIM_BOUNDARY,
        "warning": DOWNSTREAM_CAMERA_WARNING,
        "profile_name": profile["profile_name"],
        "raw_data_dir": str(data_dir),
        "raw_data_policy": "raw camera files are immutable source evidence; not committed by default",
        "governance": build_downstream_carrier_stop_study(config)["governance"],
    }

    paths = {
        "run_dir": out_dir,
        "data_dir": data_dir,
        "run_manifest": out_dir / "run_manifest.json",
        "acquisition_plan": out_dir / "acquisition_plan.csv",
        "capture_manifest_template": out_dir / "capture_manifest_template.csv",
        "hardware_profile_snapshot": out_dir / "hardware_profile_snapshot.json",
        "bench_inventory_snapshot": out_dir / "bench_inventory_snapshot.json",
        "coordinate_contract_snapshot": out_dir / "coordinate_contract_snapshot.json",
        "experiment_package_dir": out_dir / "experiment_package",
        "phase_masks_slm1_dir": out_dir / "phase_masks" / "slm1",
        "phase_masks_slm2_dir": out_dir / "phase_masks" / "slm2",
        "mask_atlas_figure": out_dir / "figures" / "downstream_carrier_mask_atlas.png",
    }
    _write_json(paths["run_manifest"], manifest)
    _write_csv(paths["acquisition_plan"], DOWNSTREAM_CAPTURE_MANIFEST_COLUMNS, rows)
    _write_csv(paths["capture_manifest_template"], DOWNSTREAM_CAPTURE_MANIFEST_COLUMNS, rows)
    _write_json(paths["hardware_profile_snapshot"], profile)
    _write_json(paths["bench_inventory_snapshot"], bench_profile)
    _write_json(paths["coordinate_contract_snapshot"], {
        "frames": coordinate_frame_rows(),
        "transforms": coordinate_transform_rows(),
    })
    _write_downstream_experiment_package(paths["experiment_package_dir"], run_id, manifest, rows, config)
    fig = plot_command_domain_carrier_mask_atlas(masks, carrier_config, output_path=paths["mask_atlas_figure"])
    import matplotlib.pyplot as plt
    plt.close(fig)
    if save_overview_to is not None:
        plot_downstream_carrier_stop_session_overview(config=config, output_path=save_overview_to)
    return paths


def _write_downstream_experiment_package(
    exp_dir: Path,
    run_id: str,
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    config: DownstreamCarrierStopConfig,
) -> None:
    (exp_dir / "LAB_README_DOWNSTREAM_CARRIER_STOP_SESSION.md").write_text(
        make_lab_readme_downstream(run_id, manifest["timestamp_utc"], manifest["git_commit"], config),
        encoding="utf-8",
    )
    (exp_dir / "bench_setup_sheet.md").write_text(make_bench_setup_sheet(run_id, config), encoding="utf-8")
    _write_csv(
        exp_dir / "bench_setup_sheet.csv",
        ["component", "state_or_setting", "recorded_value", "value_status", "notes"],
        [
            {"component": "SLM1", "state_or_setting": "mask", "recorded_value": "flat_phase", "value_status": "derived", "notes": "SLM1 flat"},
            {"component": "SLM2", "state_or_setting": "mask", "recorded_value": "command-domain carrier-only", "value_status": "derived", "notes": "carrier cycles across command area"},
            {"component": "camera", "state_or_setting": "plane", "recorded_value": config.camera_plane_label, "value_status": "installed", "notes": "not Fourier plane"},
            {"component": "Fourier stop", "state_or_setting": "state", "recorded_value": config.fourier_stop_state, "value_status": "operator_recorded", "notes": "vary deliberately where possible"},
            {"component": "physical axicon", "state_or_setting": "state", "recorded_value": config.physical_axicon_state, "value_status": "operator_recorded", "notes": "bypass/remove only if practical"},
            {"component": "downstream optics", "state_or_setting": "fixed route", "recorded_value": config.downstream_optics_state, "value_status": "operator_recorded", "notes": "all optics after stop are part of response"},
        ],
    )
    _write_csv(
        exp_dir / "camera_capture_checklist.csv",
        ["capture_id", "capture_family", "acquired", "file_name", "exposure", "gain", "saturation_ok", "clipping_ok", "operator_initials", "notes"],
        [{"capture_id": r["capture_id"], "capture_family": r["capture_family"]} for r in rows],
    )
    _write_csv(
        exp_dir / "downstream_carrier_sweep_log.csv",
        [
            "capture_id", "carrier_cycles_x", "carrier_cycles_y", "relative_output_intensity",
            "centroid_x_px", "centroid_y_px", "morphology", "saturation_fraction", "fov_margin_px", "notes",
        ],
        [r for r in rows if r["capture_family"] == "D2_carrier_sweep_fixed_stop"],
    )
    _write_csv(
        exp_dir / "downstream_stop_sweep_log.csv",
        [
            "capture_id", "carrier_cycles_x", "carrier_cycles_y", "stop_x", "stop_y", "stop_radius_or_aperture",
            "relative_output_intensity", "centroid_x_px", "centroid_y_px", "morphology", "notes",
        ],
        [
            {
                "capture_id": r["capture_id"],
                "carrier_cycles_x": r["carrier_cycles_x"],
                "carrier_cycles_y": r["carrier_cycles_y"],
                "stop_x": r["fourier_stop_centre_command_or_stage_x"],
                "stop_y": r["fourier_stop_centre_command_or_stage_y"],
                "stop_radius_or_aperture": r["fourier_stop_radius_command_or_aperture_label"],
            }
            for r in rows
            if r["capture_family"] in {
                "D3_stop_position_sweep_fixed_carrier",
                "D4_stop_radius_or_aperture_sweep_fixed_carrier",
            }
        ],
    )
    _write_csv(
        exp_dir / "downstream_response_observation_template.csv",
        [
            "capture_id", "background_estimate", "saturation_fraction", "total_camera_counts",
            "centroid_x_px", "centroid_y_px", "major_axis_second_moment_px2",
            "minor_axis_second_moment_px2", "spot_or_ring_classification", "field_of_view_margin_px",
            "quality_flags", "empirical_rank", "notes",
        ],
        [],
    )
    (exp_dir / "operator_notes_template.md").write_text(
        f"# Operator notes - downstream carrier/stop session {run_id}\n\n"
        "- date:\n- operator:\n- camera position:\n- downstream optics state:\n\n"
        "## Neutral observations\n\n"
        "Do not infer Fourier-plane order coordinates, stop transmission, Zernikes, phase, or material response.\n",
        encoding="utf-8",
    )


def make_lab_readme_downstream(
    run_id: str = "<run_id>",
    timestamp_utc: str = "<timestamp_utc>",
    git_commit: str | None = "<git_commit>",
    config: DownstreamCarrierStopConfig | None = None,
) -> str:
    config = config or DownstreamCarrierStopConfig()
    return f"""# LAB README - Downstream Carrier/Stop Session

Run: {run_id}
Timestamp: {timestamp_utc}
Git commit: {git_commit}

## Beam state

- SLM1: flat phase
- SLM2: command-domain carrier-only mask
- camera: installed downstream final-focus/output plane
- Fourier stop: state recorded and varied deliberately where possible
- axicon: state recorded; bypass/remove only if physically practical
- all downstream optics: recorded as fixed bench state

## Minimum first session

1. Record run ID, camera position, SLM identifiers, stop state,
   axicon state, and every fixed downstream optic.
2. Capture dark frames at the intended exposure/gain.
3. Display SLM1-flat / SLM2-flat and capture a flat-reference output.
4. Keep the stop at one recorded baseline setting.
   Run the x-carrier sequence and capture one raw image per mask.
5. Keep the same baseline stop setting.
   Run the y-carrier sequence and capture one raw image per mask.
6. Pick one carrier setting that visibly produces a usable output.
   Perform a small stop-centre x/y sweep if adjustable.
7. If the stop aperture/radius is adjustable, perform a small stop-size sweep.
8. Repeat the apparent best configuration at least three times.
9. Do not move camera, lenses, SLMs, or downstream optics without logging it.
10. Keep raw camera files unchanged under the generated run ID.

## Explicit statement

This session measures the downstream optical response of the complete existing route.

It does not directly image the Fourier plane.

It does not establish physical Fourier-plane coordinates, selected-order purity,
or a fully calibrated physical 4F model.

Warning: {DOWNSTREAM_CAMERA_WARNING}
"""


def make_bench_setup_sheet(run_id: str, config: DownstreamCarrierStopConfig) -> str:
    return f"""# Bench setup sheet - downstream carrier/stop session {run_id}

- calibration_mode: {config.calibration_mode}
- camera_plane_label: {config.camera_plane_label}
- relationship_to_fourier_plane: {config.camera_plane_relationship_to_fourier_plane}
- Fourier stop: {config.fourier_stop_state}
- physical axicon state policy: {config.physical_axicon_state}
- allowed recorded axicon states: {', '.join(AXICON_STATE_CHOICES)}
- downstream optics: {config.downstream_optics_state}

Claim boundary: {CLAIM_BOUNDARY}
"""


def compute_downstream_image_metrics(
    image: np.ndarray,
    *,
    bit_depth: int | None = None,
    saturation_level: float | None = None,
) -> dict[str, Any]:
    """Extract only downstream-supported pixel metrics from one image."""
    arr = np.asarray(image, dtype=float)
    base = compute_measured_image_metrics(arr, bit_depth=bit_depth, saturation_level=saturation_level)
    h, w = arr.shape
    background = float(base["background_estimate"])
    work = np.clip(arr - background, 0.0, None)
    total = float(np.sum(work))
    ys, xs = np.mgrid[0:h, 0:w]
    if total > 0:
        cx = float(np.sum(work * xs) / total)
        cy = float(np.sum(work * ys) / total)
        dx = xs - cx
        dy = ys - cy
        cov_xx = float(np.sum(work * dx * dx) / total)
        cov_yy = float(np.sum(work * dy * dy) / total)
        cov_xy = float(np.sum(work * dx * dy) / total)
        trace = cov_xx + cov_yy
        disc = max(0.0, (cov_xx - cov_yy) ** 2 + 4.0 * cov_xy ** 2)
        major = 0.5 * (trace + disc ** 0.5)
        minor = 0.5 * (trace - disc ** 0.5)
    else:
        cx = cy = float("nan")
        major = minor = 0.0
    classification = "annular" if bool(base["is_annular"]) else "spot_or_non_annular"
    out: dict[str, Any] = {
        "background_estimate": background,
        "saturation_fraction": float(base["saturation_fraction"]),
        "total_camera_counts": total,
        "centroid_x_px": cx,
        "centroid_y_px": cy,
        "major_axis_second_moment_px2": major,
        "minor_axis_second_moment_px2": minor,
        "spot_or_ring_classification": classification,
        "field_of_view_margin_px": float(base["field_of_view_margin_px"]),
        "quality_flags": list(base["image_quality_flags"]),
        "claim_boundary_labels": list(OPERATING_POINT_BOUNDARY_LABELS),
    }
    if base["is_annular"]:
        out.update(
            {
                "ring_centre_x_px": base["ring_centre_x_px"],
                "ring_centre_y_px": base["ring_centre_y_px"],
                "ring_radius_px": base["ring_radius_px"],
                "dark_core_fraction": base["dark_core_fraction"],
                "azimuthal_uniformity": base["azimuthal_uniformity"],
            }
        )
    return out


def build_downstream_operating_point_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rank empirical downstream operating points without inferring Fourier-plane physics."""
    summaries: list[dict[str, Any]] = []
    grouped: dict[tuple[Any, Any, Any, Any, Any], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("carrier_cycles_x"),
            row.get("carrier_cycles_y"),
            row.get("fourier_stop_centre_command_or_stage_x"),
            row.get("fourier_stop_centre_command_or_stage_y"),
            row.get("fourier_stop_radius_command_or_aperture_label"),
        )
        grouped.setdefault(key, []).append(row)
    for rank, (key, group) in enumerate(sorted(grouped.items()), start=1):
        intensities = [float(g.get("relative_output_intensity", g.get("total_camera_counts", 0.0)) or 0.0) for g in group]
        centroids = [
            (float(g.get("centroid_x_px", 0.0) or 0.0), float(g.get("centroid_y_px", 0.0) or 0.0))
            for g in group
        ]
        centroid_spread = 0.0
        if len(centroids) > 1:
            arr = np.asarray(centroids, dtype=float)
            centroid_spread = float(np.max(np.linalg.norm(arr - arr.mean(axis=0), axis=1)))
        summaries.append(
            {
                "carrier_setting": {"carrier_cycles_x": key[0], "carrier_cycles_y": key[1]},
                "stop_setting": {"x": key[2], "y": key[3], "radius_or_aperture": key[4]},
                "relative_output_intensity": float(np.mean(intensities)) if intensities else 0.0,
                "quality_flags": sorted({flag for g in group for flag in str(g.get("quality_flags", "")).split(";") if flag}),
                "centroid_stability_px": centroid_spread,
                "morphology_metric": group[0].get("spot_or_ring_classification", "unknown"),
                "repeatability": "repeat_measured" if len(group) > 1 else "single_capture",
                "empirical_ranking": rank,
                "claim_boundary_labels": list(OPERATING_POINT_BOUNDARY_LABELS),
            }
        )
    summaries.sort(key=lambda r: (-r["relative_output_intensity"], r["centroid_stability_px"]))
    for i, row in enumerate(summaries, start=1):
        row["empirical_ranking"] = i
    return summaries


def plot_downstream_carrier_stop_session_overview(
    config: DownstreamCarrierStopConfig | None = None,
    *,
    output_path: str | Path | None = None,
    dpi: int = 160,
):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    config = config or DownstreamCarrierStopConfig.demo()
    study = build_downstream_carrier_stop_study(config)
    fig = plt.figure(figsize=(14.5, 8.0), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    fig.suptitle(
        "Stage 9A.1B Downstream-Focus Carrier and Stop Characterisation\n"
        "installed downstream camera: empirical response only, not Fourier-plane calibration",
        x=0.04,
        y=0.97,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    boxes = [
        (0.05, 0.70, 0.22, 0.16, "SLM1 flat\nSLM2 command carrier\ncycles across command area", "#d7f0ff"),
        (0.33, 0.70, 0.22, 0.16, "Fourier stop\nrecord baseline\nsweep x/y/radius if possible", "#fff1c9"),
        (0.61, 0.70, 0.28, 0.16, "Downstream final-focus camera\ncomplete fixed route response\nnot Fourier-plane image", "#f1e4ff"),
        (0.05, 0.42, 0.25, 0.16, "Direct mode retained\nrequires temporary access at or\nconjugate to Fourier plane", "#e8f5e9"),
        (0.38, 0.42, 0.25, 0.16, "Current default mode\ndownstream_focus_empirical\noperating-point selection", "#e8f5e9"),
        (0.71, 0.42, 0.22, 0.16, "4F readiness\nstill BLOCKED for\nphysical coordinates", "#ffe5e5"),
    ]
    for x, y, w, h, text, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#333333", lw=1.2, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", transform=ax.transAxes, fontsize=10)
    for x0, x1 in ((0.27, 0.33), (0.55, 0.61)):
        ax.annotate("", xy=(x1, 0.78), xytext=(x0, 0.78), xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#333333"))
    ax.text(0.05, 0.26, "Capture families: " + ", ".join(CAPTURE_FAMILIES), transform=ax.transAxes, fontsize=9)
    ax.text(0.05, 0.20, "Warning: " + study["warning"], transform=ax.transAxes, fontsize=9, weight="bold")
    ax.text(0.05, 0.14, "Boundary: " + CLAIM_BOUNDARY, transform=ax.transAxes, fontsize=8.5)
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight", metadata={
            "Title": "Stage 9A.1B downstream carrier stop overview",
            "final_export_allowed": "False",
        })
    return fig


def write_stage9a1b_static_artifacts(root: str | Path = ".") -> dict[str, Path]:
    root = Path(root)
    cfg = DownstreamCarrierStopConfig()
    outputs = {
        "study_config": root / "configs/studies/cslm_carrier_stop_characterisation_downstream_v1.json",
        "doc": root / "docs/45_downstream_focus_carrier_stop_characterisation.md",
        "summary": root / "STAGE9A1B_DOWNSTREAM_CARRIER_STOP_CHARACTERISATION_SUMMARY.md",
        "lab_readme": root / "LAB_README_DOWNSTREAM_CARRIER_STOP_SESSION.md",
        "figure": root / "outputs/figures/digital_twin/stage9a1b_downstream_carrier_stop_session_overview.png",
    }
    _write_json(outputs["study_config"], build_downstream_carrier_stop_study(cfg))
    outputs["doc"].write_text(make_stage9a1b_doc(), encoding="utf-8")
    outputs["summary"].write_text(make_stage9a1b_summary(), encoding="utf-8")
    outputs["lab_readme"].write_text(make_lab_readme_downstream(config=cfg), encoding="utf-8")
    fig = plot_downstream_carrier_stop_session_overview(cfg, output_path=outputs["figure"])
    import matplotlib.pyplot as plt
    plt.close(fig)
    return outputs


def make_stage9a1b_doc() -> str:
    return f"""# Stage 9A.1B Downstream-Focus Carrier and Stop Characterisation

The installed laboratory camera is at a downstream final-focus/output plane, not
at the physical Fourier plane. Stage 9A.1B therefore adds a second calibration
mode, `downstream_focus_empirical`, while preserving the existing
`direct_fourier_plane_access` mode for later temporary access at or conjugate to
the Fourier plane.

## Why Downstream And Fourier-Plane Measurements Differ

Direct Fourier-plane access can observe zero/+1/-1 order positions in camera
pixels and can later support physical Fourier-plane calibration after scale and
geometry are known. The installed downstream camera sees the result of the
complete route after the stop and all downstream optics. It can show how the
final output changes with carrier and stop settings, but it cannot identify
physical order coordinates or stop radius in Fourier-plane units.

## What The Installed Camera Can Establish

- final output centroid in camera pixels;
- final output morphology;
- relative transmitted intensity;
- saturation and clipping flags;
- empirical sensitivity to carrier settings;
- empirical sensitivity to stop x/y/radius settings;
- repeatable usable operating points.

## What Requires Temporary Fourier-Plane Access

- direct Fourier-plane order positions;
- physical Fourier-plane x/y coordinates;
- direct stop radius in Fourier-plane mm;
- order-power fractions at the stop;
- physical 4F readiness marked ready from measured Fourier-plane geometry.

Direct Fourier-plane mapping requires a temporary diagnostic method at or
conjugate to the Fourier plane. The installed downstream camera supports
empirical carrier-and-stop response characterisation only.

## Evidence Storage

The downstream package records `calibration_mode`, camera-plane relationship,
Fourier-stop state, axicon state, downstream optics state, carrier command
cycles, SLM mask IDs, exposure/gain, filters, energy setting if available, and
operator notes. Unknown values may remain unknown, but their status is recorded.

The output summary is labelled with:

```text
empirical_downstream_operating_point
not_direct_fourier_plane_calibration
not_physical_4f_model_validation
```

## Physical-4F Readiness Impact

Downstream empirical evidence supports practical operating-point selection,
repeatability assessment, and later comparison against a physical 4F model. It
does not by itself support `physical_fourier_plane_coordinate_calibrated` or
`physical_4f_readiness_ready`.

Warning: {DOWNSTREAM_CAMERA_WARNING}
"""


def make_stage9a1b_summary() -> str:
    return f"""# Stage 9A.1B Downstream Carrier/Stop Characterisation Summary

Starting checkpoint: Stage 9A.3 verified bibliography and evidence layers
(`21ad69c`).

Stage 9A.1B adds a dual-mode calibration contract:

- `direct_fourier_plane_access`
- `downstream_focus_empirical`

The current laboratory default is `downstream_focus_empirical`, because the
installed camera is at the downstream final-focus/output plane. Existing Stage
9A.1 command-domain carrier masks remain valid and unchanged.

## Created

- `vbb_study/digital_twin/downstream_carrier_stop.py`
- `configs/studies/cslm_carrier_stop_characterisation_downstream_v1.json`
- `docs/45_downstream_focus_carrier_stop_characterisation.md`
- `STAGE9A1B_DOWNSTREAM_CARRIER_STOP_CHARACTERISATION_SUMMARY.md`
- `LAB_README_DOWNSTREAM_CARRIER_STOP_SESSION.md`
- `tests/test_stage9a1b_downstream_carrier_stop_characterisation.py`
- `outputs/figures/digital_twin/stage9a1b_downstream_carrier_stop_session_overview.png`

## Boundary

{CLAIM_BOUNDARY}

## First Bench Action

Run the downstream session: SLM1 flat, SLM2 carrier-only, installed downstream
camera fixed, Fourier stop recorded and varied deliberately where possible,
axicon state recorded, all downstream optics logged as fixed bench state.
"""


if __name__ == "__main__":
    for key, value in write_stage9a1b_static_artifacts(Path.cwd()).items():
        print(f"{key}: {value}")
