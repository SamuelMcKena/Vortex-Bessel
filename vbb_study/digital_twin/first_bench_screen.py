"""Stage 9B.0.2 first bench screen package generator.

This module composes already-declared Stage 9A.1B downstream carrier/stop
logistics and Stage 9B.0.1 nominal candidate-mask outputs into a first bench
screen handoff package. It creates command masks, manifests, operator logs,
and documentation only.

It does not add optical physics, physical 4F calibration/readiness, a camera
model, inverse correction, AI, material response, or raw-image processing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.digital_twin.calibration_acquisition import (
    _write_csv,
    _write_json,
    generate_run_id,
    git_commit_hash,
    utc_now_iso,
)
from vbb_study.digital_twin.candidate_beam_atlas import (
    BENCH_VALIDATION_STATUS,
    HARDWARE_COMMAND_EXPORT_STATUS,
    PHYSICAL_4F_READINESS,
    CandidateSpec,
    build_candidate_atlas_config,
    build_candidate_ranking_validity_rows,
    candidate_manifest,
    evaluate_candidate_stop_sampling_convergence,
    simulate_candidate,
)
from vbb_study.digital_twin.downstream_carrier_stop import (
    CAPTURE_FAMILIES as DOWNSTREAM_CAPTURE_FAMILIES,
    CLAIM_BOUNDARY as DOWNSTREAM_CLAIM_BOUNDARY,
    DOWNSTREAM_CAMERA_WARNING,
    DownstreamCarrierStopConfig,
    build_downstream_carrier_stop_study,
    downstream_capture_plan_rows,
)
from vbb_study.digital_twin.nominal_f300_4f import (
    CARRIER_BOUNDARY_FLAGS,
    CARRIER_REALISM,
    IDEAL_CARRIER_BOUNDARY,
    NominalF300Config,
    load_nominal_f300_profile,
    phase_export_payload,
)
from vbb_study.digital_twin.slm_calibration_masks import (
    build_carrier_sweep_masks,
    export_mask,
)


STAGE = "9B.0.2"
STUDY_NAME = "cslm_first_bench_screen_v1"
STUDY_PATH = Path("configs/studies/cslm_first_bench_screen_v1.json")
FIGURE_OVERVIEW = Path("outputs/figures/digital_twin/stage9b0_2_first_bench_screen_overview.png")
FIGURE_MASK_ATLAS = Path("outputs/figures/digital_twin/stage9b0_2_first_bench_screen_mask_atlas.png")

DEFAULT_FIRST_SCREEN_CANDIDATE_IDS = (
    "gaussian_reference",
    "vortex_ell_1",
    "vortex_ell_2",
)
OPTIONAL_EXTENSION_CANDIDATE_IDS = (
    "vortex_ell_3",
    "vortex_ell_4",
)

PHASES = (
    "A_as_found_bench_record",
    "B_baseline_carrier_stop_check",
    "C_candidate_screen",
    "D_optional_later_extension",
    "E_shutdown_and_notes",
)

UNKNOWN_RECORDED = "unknown_recorded"
RAW_DATA_POLICY = (
    "Raw camera files are immutable source evidence. Save them under the run raw directory, "
    "do not modify them in place, do not hash or process them in Stage 9B.0.2, and do not "
    "commit raw images by default."
)
STOP_MOTION_POLICY = (
    "Do not move SLMs, camera, lenses, axicon, or downstream optics during the screen. "
    "The Fourier stop/pinhole may be deliberately adjusted only during D3/D4 baseline "
    "stop sweeps, and every move must be logged."
)
CLAIM_BOUNDARY = (
    "first bench screen package only; command masks exportable but unvalidated; "
    "not a calibrated physical 4F or camera prediction; no new optical physics; "
    "no pixelated-SLM order physics; no inverse correction; no AI; no material response; "
    "final_export_allowed=False"
)
FIRST_SCREEN_BOUNDARY_TEXT = (
    "First bench screen package. Command masks exportable but unvalidated. "
    "Not a calibrated physical 4F or camera prediction."
)

OPERATOR_SET_REQUIRED_FIELDS = (
    "actual SLM1 identifier",
    "actual SLM2 identifier",
    "SLM display-orientation convention",
    "phase-LUT/wavelength setting used by display software",
    "carrier mask orientation on SLM2",
    "lens identifiers if readable",
    "pinhole mount identifier",
    "pinhole x stage reading",
    "pinhole y stage reading",
    "pinhole diameter if known",
    "axicon state and identifier",
    "all downstream fixed optics",
    "camera identifier",
    "camera position or rail reading",
    "camera exposure",
    "camera gain",
    "neutral-density/filter state",
    "laser energy/power setting where available",
    "date",
    "operator",
    "run notes",
)

FIRST_BENCH_CAPTURE_COLUMNS = (
    "capture_id",
    "capture_phase",
    "capture_family",
    "candidate_id",
    "candidate_family",
    "topological_charge",
    "SLM1 mask ID",
    "SLM2 mask ID",
    "SLM1 phase file",
    "SLM1 grayscale PNG",
    "SLM2 phase file",
    "SLM2 grayscale PNG",
    "carrier cycles x/y",
    "carrier_cycles_x",
    "carrier_cycles_y",
    "carrier realism label",
    "upstream source mode",
    "SLM1-to-SLM2 propagation included",
    "slm1_phase_applied_at_slm1",
    "slm1_to_slm2_propagation_included",
    "slm2_carrier_applied_at_slm2",
    "slm2_contains_vortex",
    "slm2_contains_axicon",
    "stop sampling status",
    "convergence status",
    "candidate-ranking status",
    "hardware_command_export_status",
    "bench_validation_status",
    "physical_4f_readiness",
    "camera plane label",
    "axicon state",
    "downstream optics state",
    "pinhole state",
    "raw file path placeholder",
    "claim boundary",
    "operator_action",
)


@dataclass(frozen=True)
class FirstBenchScreenConfig:
    """Package-level choices for the Stage 9B.0.2 handoff."""

    study_name: str = STUDY_NAME
    first_screen_candidate_ids: tuple[str, ...] = DEFAULT_FIRST_SCREEN_CANDIDATE_IDS
    optional_extension_candidate_ids: tuple[str, ...] = OPTIONAL_EXTENSION_CANDIDATE_IDS
    include_optional_extensions: bool = False
    repeats_per_candidate: int = 3
    candidate_slm2_carrier_cycles_x: int = 8
    candidate_slm2_carrier_cycles_y: int = 0
    camera_plane_label: str = "downstream_final_focus"
    camera_plane_relationship_to_fourier_plane: str = (
        "not at Fourier plane; records complete fixed downstream route response"
    )
    axicon_state: str = UNKNOWN_RECORDED
    downstream_optics_state: str = "operator_record_fixed_existing_route"
    pinhole_state: str = "operator_selected_after_phase_B_baseline"
    baseline_carrier_selection_status: str = (
        "operator must choose usable baseline from observed downstream response; "
        "the included x=+8 carrier file is a starting command mask, not a simulation-selected optimum"
    )
    stop_motion_policy: str = STOP_MOTION_POLICY
    raw_data_policy: str = RAW_DATA_POLICY
    hardware_command_export_status: str = HARDWARE_COMMAND_EXPORT_STATUS
    bench_validation_status: str = BENCH_VALIDATION_STATUS
    physical_4f_readiness: str = PHYSICAL_4F_READINESS
    final_export_allowed: bool = False


def _candidate_spec(candidate_id: str) -> CandidateSpec:
    if candidate_id == "gaussian_reference":
        return CandidateSpec("gaussian_reference", "gaussian_reference", ell=0)
    prefix = "vortex_ell_"
    if candidate_id.startswith(prefix):
        ell = int(candidate_id[len(prefix) :])
        return CandidateSpec(candidate_id, "vortex_charge_sweep", ell=ell)
    raise ValueError(f"Unsupported first-screen candidate_id {candidate_id!r}.")


def _selected_candidate_ids(config: FirstBenchScreenConfig) -> tuple[str, ...]:
    ids = tuple(config.first_screen_candidate_ids)
    if config.include_optional_extensions:
        ids = ids + tuple(config.optional_extension_candidate_ids)
    return ids


def build_first_bench_screen_config(config: FirstBenchScreenConfig | None = None) -> dict[str, Any]:
    """Return the machine-readable first bench screen contract."""

    config = config or FirstBenchScreenConfig()
    return {
        "stage": STAGE,
        "study_name": config.study_name,
        "purpose": "first controlled bench screen package and baseline acquisition handoff",
        "claim_boundary": CLAIM_BOUNDARY,
        "phases": list(PHASES),
        "default_first_screen_candidate_ids": list(config.first_screen_candidate_ids),
        "optional_later_extension_candidate_ids": list(config.optional_extension_candidate_ids),
        "include_optional_extensions_by_default": bool(config.include_optional_extensions),
        "first_screen_candidate_rule": (
            "default screen is gaussian_reference, vortex_ell_1, and vortex_ell_2 only; "
            "vortex_ell_3 and vortex_ell_4 remain optional later extensions"
        ),
        "baseline_candidate": "gaussian_reference",
        "candidate_repeats_minimum": int(config.repeats_per_candidate),
        "operator_set_required_fields": list(OPERATOR_SET_REQUIRED_FIELDS),
        "unknown_values_policy": (
            "unknown values are permitted for the first screen only when explicitly recorded as unknown_recorded"
        ),
        "raw_data_policy": config.raw_data_policy,
        "stop_motion_policy": config.stop_motion_policy,
        "camera_plane_label": config.camera_plane_label,
        "camera_plane_relationship_to_fourier_plane": config.camera_plane_relationship_to_fourier_plane,
        "baseline_carrier_selection_status": config.baseline_carrier_selection_status,
        "carrier_boundary": {
            **dict(CARRIER_BOUNDARY_FLAGS),
            "bench_screen_statement": (
                "The SLM2 carrier is an ideal continuous-ramp command mask in the nominal atlas; "
                "bench captures are empirical downstream images and not measured physical order purity."
            ),
        },
        "candidate_manifest_required_fields": [
            "candidate_id",
            "candidate_family",
            "topological_charge",
            "upstream_source_mode",
            "slm1_to_slm2_propagation_included",
            "slm2_carrier_mode",
            "carrier_realism",
            "stop_sampling_status",
            "convergence_status",
            "hardware_command_export_status",
            "bench_validation_status",
            "physical_4f_readiness",
        ],
        "capture_plan_required_columns": list(FIRST_BENCH_CAPTURE_COLUMNS),
        "required_package_structure": [
            "run_manifest.json",
            "first_bench_screen_manifest.json",
            "capture_plan.csv",
            "hardware_profile_snapshot.json",
            "candidate_atlas_snapshot.json",
            "nominal_4f_profile_snapshot.json",
            "claim_boundary.md",
            "phase_masks/slm1/",
            "phase_masks/slm2/",
            "experiment_package/",
            "figures/",
            "raw/README_RAW_DATA_POLICY.md",
        ],
        "governance": _governance_record(),
    }


def write_first_bench_screen_config(path: str | Path = STUDY_PATH) -> Path:
    """Write the static Stage 9B.0.2 study config."""

    return _write_json(Path(path), build_first_bench_screen_config())


def _governance_record() -> dict[str, bool]:
    return {
        "new_optical_physics_added": False,
        "physical_4f_filter_modelled": False,
        "physical_4f_readiness_ready": False,
        "camera_model_enabled": False,
        "inverse_correction_enabled": False,
        "phase_diversity_enabled": False,
        "zernike_fitting_enabled": False,
        "ai_enabled": False,
        "material_model_enabled": False,
        "pixelated_slm_order_physics_modelled": False,
        "raw_camera_images_processed": False,
        "diagnostic_only": True,
        "final_export_allowed": False,
    }


def _safe_rel(path: str | Path, root: Path) -> str:
    p = Path(path)
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()


def _phase_payload_paths(prefix: str, out_dir: Path) -> dict[str, Path]:
    safe = prefix.replace(" ", "_")
    return {
        "phase_rad": out_dir / f"{safe}_phase_rad.npy",
        "quantised_rad": out_dir / f"{safe}_quantised_rad.npy",
        "gray": out_dir / f"{safe}_gray.png",
        "metadata": out_dir / f"{safe}_metadata.json",
    }


def _write_png(path: Path, array: np.ndarray, *, cmap: str = "viridis") -> Path:
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array)
    if arr.ndim == 2 and np.issubdtype(arr.dtype, np.integer):
        plt.imsave(path, arr, cmap=cmap, vmin=0, vmax=255)
    else:
        plt.imsave(path, arr, cmap=cmap)
    return path


def _write_phase_payload(
    payload: Mapping[str, Any],
    out_dir: Path,
    *,
    contains_vortex: bool,
    contains_axicon: bool = False,
) -> dict[str, Path]:
    mask_id = str(payload["metadata"]["mask_id"])
    paths = _phase_payload_paths(mask_id, out_dir)
    np.save(paths["phase_rad"], np.asarray(payload["phase_rad"], dtype=np.float32))
    np.save(paths["quantised_rad"], np.asarray(payload["quantised_rad"], dtype=np.float32))
    _write_png(paths["gray"], np.asarray(payload["gray"], dtype=np.uint8), cmap="gray")
    metadata = dict(payload["metadata"])
    metadata.update(
        {
            "contains_vortex": bool(contains_vortex),
            "contains_axicon": bool(contains_axicon),
            "hardware_command_export_status": HARDWARE_COMMAND_EXPORT_STATUS,
            "bench_validation_status": BENCH_VALIDATION_STATUS,
            "final_export_allowed": False,
        }
    )
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return paths


def _export_downstream_masks(
    out_dir: Path,
    downstream_config: DownstreamCarrierStopConfig,
) -> dict[str, dict[str, Path]]:
    masks = build_carrier_sweep_masks(downstream_config.carrier_config())
    exports: dict[str, dict[str, Path]] = {}
    for mask in masks:
        target = out_dir / "phase_masks" / ("slm1" if mask["metadata"]["slm_id"] == "SLM1" else "slm2")
        exports[mask["metadata"]["mask_id"]] = export_mask(mask, target)
    return exports


def _mask_file_fields(mask_id: str, exports: Mapping[str, Mapping[str, Path]], root: Path, *, slm: str) -> dict[str, str]:
    if mask_id in {"shuttered", "not_applicable_record_only"}:
        return {
            f"{slm} phase file": mask_id,
            f"{slm} grayscale PNG": mask_id,
        }
    paths = exports.get(mask_id)
    if not paths:
        return {
            f"{slm} phase file": f"missing_mask_export:{mask_id}",
            f"{slm} grayscale PNG": f"missing_mask_export:{mask_id}",
        }
    return {
        f"{slm} phase file": _safe_rel(paths.get("phase_npy", ""), root),
        f"{slm} grayscale PNG": _safe_rel(paths.get("gray_png", ""), root),
    }


def _normalise_downstream_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: FirstBenchScreenConfig,
    mask_exports: Mapping[str, Mapping[str, Path]],
    package_root: Path,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        slm1_id = str(row["SLM1 mask ID"])
        slm2_id = str(row["SLM2 mask ID"])
        raw_placeholder = str(row["raw_target_path"])
        enriched = {
            "capture_id": row["capture_id"],
            "capture_phase": "B_baseline_carrier_stop_check",
            "capture_family": row["capture_family"],
            "candidate_id": "baseline_carrier_stop_characterisation",
            "candidate_family": "baseline_control",
            "topological_charge": "",
            "SLM1 mask ID": slm1_id,
            "SLM2 mask ID": slm2_id,
            "carrier cycles x/y": _cycles_label(row.get("carrier_cycles_x"), row.get("carrier_cycles_y")),
            "carrier_cycles_x": row.get("carrier_cycles_x", ""),
            "carrier_cycles_y": row.get("carrier_cycles_y", ""),
            "carrier realism label": "command_domain_carrier_only_for_empirical_downstream_baseline",
            "upstream source mode": "physical_bench_existing_route",
            "SLM1-to-SLM2 propagation included": "true",
            "slm1_phase_applied_at_slm1": "true" if slm1_id != "shuttered" else "not_applicable_dark",
            "slm1_to_slm2_propagation_included": "true" if slm1_id != "shuttered" else "not_applicable_dark",
            "slm2_carrier_applied_at_slm2": "true" if slm2_id != "shuttered" else "not_applicable_dark",
            "slm2_contains_vortex": "false",
            "slm2_contains_axicon": "false",
            "stop sampling status": "not_simulation_ranked_bench_baseline",
            "convergence status": "not_applicable_bench_capture",
            "candidate-ranking status": "not_ranked_baseline_acquisition",
            "hardware_command_export_status": HARDWARE_COMMAND_EXPORT_STATUS,
            "bench_validation_status": BENCH_VALIDATION_STATUS,
            "physical_4f_readiness": PHYSICAL_4F_READINESS,
            "camera plane label": config.camera_plane_label,
            "axicon state": config.axicon_state,
            "downstream optics state": config.downstream_optics_state,
            "pinhole state": row.get("fourier_stop_state", config.pinhole_state),
            "raw file path placeholder": raw_placeholder,
            "claim boundary": DOWNSTREAM_CLAIM_BOUNDARY,
            "operator_action": _operator_action_for_baseline_family(str(row["capture_family"])),
        }
        enriched.update(_mask_file_fields(slm1_id, mask_exports, package_root, slm="SLM1"))
        enriched.update(_mask_file_fields(slm2_id, mask_exports, package_root, slm="SLM2"))
        out.append(enriched)
    return out


def _operator_action_for_baseline_family(family: str) -> str:
    if family == "D0_dark_frames":
        return "capture dark frames at intended exposure/gain"
    if family == "D1_flat_reference":
        return "display SLM1 flat and SLM2 flat reference"
    if family == "D2_carrier_sweep_fixed_stop":
        return "keep stop fixed and sweep SLM2 carrier masks"
    if family == "D3_stop_position_sweep_fixed_carrier":
        return "move stop centre deliberately and log stage reading"
    if family == "D4_stop_radius_or_aperture_sweep_fixed_carrier":
        return "change stop aperture deliberately and log setting"
    if family == "D5_repeatability_reference":
        return "repeat apparent usable carrier/stop setting"
    return "record capture"


def _cycles_label(x: Any, y: Any) -> str:
    if x in {"", None} and y in {"", None}:
        return ""
    return f"{x}/{y}"


def _candidate_mask_id(candidate_id: str, slm: str) -> str:
    if slm == "SLM1":
        return f"{candidate_id}_SLM1"
    return f"{candidate_id}_SLM2_candidate_carrier_x_+8"


def _export_candidate_masks_and_rows(
    package_dir: Path,
    data_dir: Path,
    config: FirstBenchScreenConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected_ids = _selected_candidate_ids(config)
    slm1_dir = package_dir / "phase_masks" / "slm1"
    slm2_dir = package_dir / "phase_masks" / "slm2"
    rows: list[dict[str, Any]] = []
    manifests: dict[str, Any] = {}
    ranking_rows = {row["candidate_id"]: row for row in build_candidate_ranking_validity_rows()}
    for candidate_id in selected_ids:
        spec = _candidate_spec(candidate_id)
        run = simulate_candidate(spec, NominalF300Config.standard())
        convergence = evaluate_candidate_stop_sampling_convergence(spec)
        manifest = candidate_manifest(spec, run, [], convergence)
        manifest["bench_screen_included_by_default"] = candidate_id in DEFAULT_FIRST_SCREEN_CANDIDATE_IDS
        manifest["bench_screen_phase"] = "C_candidate_screen" if candidate_id in DEFAULT_FIRST_SCREEN_CANDIDATE_IDS else "D_optional_later_extension"
        manifest["bench_screen_note"] = (
            "Use this as a command-mask screen item only. The bench image is empirical and must not be "
            "read as a calibrated physical 4F/camera/material prediction."
        )
        manifests[candidate_id] = manifest
        slm1_payload = phase_export_payload(
            run.slm1_phase_rad,
            mask_id=_candidate_mask_id(candidate_id, "SLM1"),
            slm_id="SLM1",
            config=run.config,
        )
        slm2_payload = phase_export_payload(
            run.slm2_phase_rad,
            mask_id=_candidate_mask_id(candidate_id, "SLM2"),
            slm_id="SLM2",
            config=run.config,
        )
        slm1_paths = _write_phase_payload(
            slm1_payload,
            slm1_dir,
            contains_vortex=bool(spec.ell != 0),
        )
        slm2_paths = _write_phase_payload(
            slm2_payload,
            slm2_dir,
            contains_vortex=False,
            contains_axicon=False,
        )
        rank_row = ranking_rows.get(candidate_id, {})
        phase = "C_candidate_screen" if candidate_id in DEFAULT_FIRST_SCREEN_CANDIDATE_IDS else "D_optional_later_extension"
        family = f"C_{candidate_id}" if phase.startswith("C") else f"D_optional_{candidate_id}"
        for rep in range(int(config.repeats_per_candidate)):
            raw_path = data_dir / "raw" / f"{family}_r{rep:02d}.png"
            rows.append(
                {
                    "capture_id": f"{family}_r{rep:02d}",
                    "capture_phase": phase,
                    "capture_family": family,
                    "candidate_id": candidate_id,
                    "candidate_family": spec.candidate_family,
                    "topological_charge": int(spec.ell),
                    "SLM1 mask ID": slm1_payload["metadata"]["mask_id"],
                    "SLM2 mask ID": slm2_payload["metadata"]["mask_id"],
                    "SLM1 phase file": _safe_rel(slm1_paths["phase_rad"], package_dir),
                    "SLM1 grayscale PNG": _safe_rel(slm1_paths["gray"], package_dir),
                    "SLM2 phase file": _safe_rel(slm2_paths["phase_rad"], package_dir),
                    "SLM2 grayscale PNG": _safe_rel(slm2_paths["gray"], package_dir),
                    "carrier cycles x/y": _cycles_label(
                        config.candidate_slm2_carrier_cycles_x,
                        config.candidate_slm2_carrier_cycles_y,
                    ),
                    "carrier_cycles_x": int(config.candidate_slm2_carrier_cycles_x),
                    "carrier_cycles_y": int(config.candidate_slm2_carrier_cycles_y),
                    "carrier realism label": CARRIER_REALISM,
                    "upstream source mode": "existing_cslm_component_route",
                    "SLM1-to-SLM2 propagation included": "true",
                    "slm1_phase_applied_at_slm1": "true",
                    "slm1_to_slm2_propagation_included": "true",
                    "slm2_carrier_applied_at_slm2": "true",
                    "slm2_contains_vortex": "false",
                    "slm2_contains_axicon": "false",
                    "stop sampling status": str(manifest["stop_sampling_status"]),
                    "convergence status": str(manifest["convergence_status"]),
                    "candidate-ranking status": _ranking_status(rank_row),
                    "hardware_command_export_status": HARDWARE_COMMAND_EXPORT_STATUS,
                    "bench_validation_status": BENCH_VALIDATION_STATUS,
                    "physical_4f_readiness": PHYSICAL_4F_READINESS,
                    "camera plane label": config.camera_plane_label,
                    "axicon state": config.axicon_state,
                    "downstream optics state": config.downstream_optics_state,
                    "pinhole state": config.pinhole_state,
                    "raw file path placeholder": str(raw_path),
                    "claim boundary": CLAIM_BOUNDARY,
                    "operator_action": (
                        "display SLM1 candidate mask and the operator-confirmed SLM2 baseline carrier; "
                        "capture downstream image and record observation"
                    ),
                }
            )
    return manifests, rows


def _ranking_status(row: Mapping[str, Any]) -> str:
    if bool(row.get("ranking_allowed")):
        return f"nominal_atlas_rank_{row.get('robustness_rank')}_for_mask_sequence_only"
    return "not_ranked_or_not_converged"


def build_first_bench_capture_plan_rows(
    *,
    config: FirstBenchScreenConfig | None = None,
    downstream_config: DownstreamCarrierStopConfig | None = None,
    package_dir: str | Path = "outputs/calibration_runs/placeholder",
    data_dir: str | Path = "data/calibration_runs/placeholder",
) -> list[dict[str, Any]]:
    """Build capture-plan rows without writing files."""

    config = config or FirstBenchScreenConfig()
    downstream_config = downstream_config or DownstreamCarrierStopConfig()
    package = Path(package_dir)
    data = Path(data_dir)
    downstream_rows = downstream_capture_plan_rows(downstream_config, data_dir=data)
    fake_exports: dict[str, dict[str, Path]] = {}
    baseline = _normalise_downstream_rows(
        downstream_rows,
        config=config,
        mask_exports=fake_exports,
        package_root=package,
    )
    candidates: list[dict[str, Any]] = []
    for candidate_id in _selected_candidate_ids(config):
        spec = _candidate_spec(candidate_id)
        for rep in range(int(config.repeats_per_candidate)):
            phase = "C_candidate_screen" if candidate_id in DEFAULT_FIRST_SCREEN_CANDIDATE_IDS else "D_optional_later_extension"
            family = f"C_{candidate_id}" if phase.startswith("C") else f"D_optional_{candidate_id}"
            candidates.append(
                {
                    "capture_id": f"{family}_r{rep:02d}",
                    "capture_phase": phase,
                    "capture_family": family,
                    "candidate_id": candidate_id,
                    "candidate_family": spec.candidate_family,
                    "topological_charge": int(spec.ell),
                    "SLM1 mask ID": _candidate_mask_id(candidate_id, "SLM1"),
                    "SLM2 mask ID": _candidate_mask_id(candidate_id, "SLM2"),
                    "SLM1 phase file": f"phase_masks/slm1/{_candidate_mask_id(candidate_id, 'SLM1')}_phase_rad.npy",
                    "SLM1 grayscale PNG": f"phase_masks/slm1/{_candidate_mask_id(candidate_id, 'SLM1')}_gray.png",
                    "SLM2 phase file": f"phase_masks/slm2/{_candidate_mask_id(candidate_id, 'SLM2')}_phase_rad.npy",
                    "SLM2 grayscale PNG": f"phase_masks/slm2/{_candidate_mask_id(candidate_id, 'SLM2')}_gray.png",
                    "carrier cycles x/y": _cycles_label(config.candidate_slm2_carrier_cycles_x, config.candidate_slm2_carrier_cycles_y),
                    "carrier_cycles_x": int(config.candidate_slm2_carrier_cycles_x),
                    "carrier_cycles_y": int(config.candidate_slm2_carrier_cycles_y),
                    "carrier realism label": CARRIER_REALISM,
                    "upstream source mode": "existing_cslm_component_route",
                    "SLM1-to-SLM2 propagation included": "true",
                    "slm1_phase_applied_at_slm1": "true",
                    "slm1_to_slm2_propagation_included": "true",
                    "slm2_carrier_applied_at_slm2": "true",
                    "slm2_contains_vortex": "false",
                    "slm2_contains_axicon": "false",
                    "stop sampling status": "convergence_verified",
                    "convergence status": "passed_for_nominal_scenario",
                    "candidate-ranking status": "nominal_atlas_for_mask_sequence_only",
                    "hardware_command_export_status": HARDWARE_COMMAND_EXPORT_STATUS,
                    "bench_validation_status": BENCH_VALIDATION_STATUS,
                    "physical_4f_readiness": PHYSICAL_4F_READINESS,
                    "camera plane label": config.camera_plane_label,
                    "axicon state": config.axicon_state,
                    "downstream optics state": config.downstream_optics_state,
                    "pinhole state": config.pinhole_state,
                    "raw file path placeholder": str(data / "raw" / f"{family}_r{rep:02d}.png"),
                    "claim boundary": CLAIM_BOUNDARY,
                    "operator_action": "candidate screen capture",
                }
            )
    return baseline + candidates


def validate_first_bench_capture_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return manifest issues for first-bench capture rows."""

    issues: list[str] = []
    for row in rows:
        cid = row.get("capture_id", "<missing>")
        for key in FIRST_BENCH_CAPTURE_COLUMNS:
            if key not in row:
                issues.append(f"{cid}: missing required field {key}")
        blob = json.dumps(row).lower()
        forbidden = (
            "physical_4f_readiness_ready",
            "camera_model_enabled: true",
            "material_model_enabled: true",
            "selected-order purity predicted",
        )
        if any(token in blob for token in forbidden):
            issues.append(f"{cid}: row overstates model validity")
        if str(row.get("capture_phase")) == "C_candidate_screen":
            if row.get("upstream source mode") != "existing_cslm_component_route":
                issues.append(f"{cid}: candidate row must use existing_cslm_component_route")
            if str(row.get("slm2_contains_vortex")).lower() != "false":
                issues.append(f"{cid}: SLM2 must not contain vortex")
            if str(row.get("slm2_contains_axicon")).lower() != "false":
                issues.append(f"{cid}: SLM2 must not contain axicon")
    return issues


def create_first_bench_screen_package(
    run_id: str | None = None,
    *,
    config: FirstBenchScreenConfig | None = None,
    downstream_config: DownstreamCarrierStopConfig | None = None,
    output_root: str | Path = "outputs/calibration_runs",
    data_root: str | Path = "data/calibration_runs",
    repo: str | Path | None = None,
    save_static_figures: bool = False,
) -> dict[str, Path]:
    """Create a first bench screen package under the calibration-run convention."""

    config = config or FirstBenchScreenConfig()
    downstream_config = downstream_config or DownstreamCarrierStopConfig()
    run_id = run_id or generate_run_id("firstbench")
    package_dir = Path(output_root) / run_id
    data_dir = Path(data_root) / run_id
    if package_dir.exists() or data_dir.exists():
        raise FileExistsError(f"calibration run already exists: {run_id}")
    for path in (
        package_dir / "phase_masks" / "slm1",
        package_dir / "phase_masks" / "slm2",
        package_dir / "experiment_package" / "candidate_summaries",
        package_dir / "figures",
        package_dir / "raw",
        data_dir / "raw",
        data_dir / "manifests",
        data_dir / "derived",
        data_dir / "figures",
    ):
        path.mkdir(parents=True, exist_ok=True)

    downstream_exports = _export_downstream_masks(package_dir, downstream_config)
    downstream_rows = downstream_capture_plan_rows(downstream_config, data_dir=data_dir)
    baseline_rows = _normalise_downstream_rows(
        downstream_rows,
        config=config,
        mask_exports=downstream_exports,
        package_root=package_dir,
    )
    candidate_manifests, candidate_rows = _export_candidate_masks_and_rows(package_dir, data_dir, config)
    capture_rows = baseline_rows + candidate_rows
    issues = validate_first_bench_capture_rows(capture_rows)
    if issues:
        raise ValueError("; ".join(issues))

    timestamp = utc_now_iso()
    manifest = _run_manifest(
        run_id=run_id,
        timestamp=timestamp,
        config=config,
        downstream_config=downstream_config,
        package_dir=package_dir,
        data_dir=data_dir,
        repo=repo,
        candidate_manifests=candidate_manifests,
    )
    first_screen_manifest = _first_screen_manifest(
        run_id=run_id,
        timestamp=timestamp,
        config=config,
        downstream_config=downstream_config,
        candidate_manifests=candidate_manifests,
        capture_rows=capture_rows,
    )
    paths = {
        "run_dir": package_dir,
        "data_dir": data_dir,
        "run_manifest": package_dir / "run_manifest.json",
        "first_bench_screen_manifest": package_dir / "first_bench_screen_manifest.json",
        "capture_plan": package_dir / "capture_plan.csv",
        "hardware_profile_snapshot": package_dir / "hardware_profile_snapshot.json",
        "candidate_atlas_snapshot": package_dir / "candidate_atlas_snapshot.json",
        "nominal_4f_profile_snapshot": package_dir / "nominal_4f_profile_snapshot.json",
        "claim_boundary": package_dir / "claim_boundary.md",
        "raw_policy": package_dir / "raw" / "README_RAW_DATA_POLICY.md",
        "experiment_package_dir": package_dir / "experiment_package",
        "overview_figure": package_dir / "figures" / "first_bench_screen_overview.png",
        "mask_atlas_figure": package_dir / "figures" / "first_bench_screen_mask_atlas.png",
    }
    _write_json(paths["run_manifest"], manifest)
    _write_json(paths["first_bench_screen_manifest"], first_screen_manifest)
    _write_csv(paths["capture_plan"], FIRST_BENCH_CAPTURE_COLUMNS, capture_rows)
    _write_json(paths["hardware_profile_snapshot"], build_downstream_carrier_stop_study(downstream_config))
    _write_json(paths["candidate_atlas_snapshot"], build_candidate_atlas_config())
    _write_json(paths["nominal_4f_profile_snapshot"], load_nominal_f300_profile())
    paths["claim_boundary"].write_text(make_claim_boundary_markdown(), encoding="utf-8")
    paths["raw_policy"].write_text(make_raw_data_policy_markdown(run_id, data_dir), encoding="utf-8")
    _write_experiment_package(
        paths["experiment_package_dir"],
        run_id,
        timestamp,
        config,
        candidate_manifests,
        capture_rows,
    )
    plot_first_bench_screen_overview(
        first_screen_manifest,
        output_path=paths["overview_figure"],
    )
    plot_first_bench_screen_mask_atlas(
        candidate_manifests,
        package_dir=package_dir,
        output_path=paths["mask_atlas_figure"],
    )
    if save_static_figures:
        plot_first_bench_screen_overview(first_screen_manifest, output_path=FIGURE_OVERVIEW)
        plot_first_bench_screen_mask_atlas(candidate_manifests, package_dir=package_dir, output_path=FIGURE_MASK_ATLAS)
    return paths


def _run_manifest(
    *,
    run_id: str,
    timestamp: str,
    config: FirstBenchScreenConfig,
    downstream_config: DownstreamCarrierStopConfig,
    package_dir: Path,
    data_dir: Path,
    repo: str | Path | None,
    candidate_manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "git_commit": git_commit_hash(repo),
        "stage": STAGE,
        "study_name": config.study_name,
        "package_dir": str(package_dir),
        "data_dir": str(data_dir),
        "raw_data_dir": str(data_dir / "raw"),
        "raw_data_policy": config.raw_data_policy,
        "claim_boundary": CLAIM_BOUNDARY,
        "first_screen_candidate_ids": list(config.first_screen_candidate_ids),
        "optional_extension_candidate_ids": list(config.optional_extension_candidate_ids),
        "optional_extensions_included": bool(config.include_optional_extensions),
        "baseline_candidate": "gaussian_reference",
        "baseline_carrier_selection_status": config.baseline_carrier_selection_status,
        "candidate_mask_status": HARDWARE_COMMAND_EXPORT_STATUS,
        "bench_validation_status": BENCH_VALIDATION_STATUS,
        "physical_4f_readiness": PHYSICAL_4F_READINESS,
        "camera_plane_label": config.camera_plane_label,
        "camera_plane_relationship_to_fourier_plane": config.camera_plane_relationship_to_fourier_plane,
        "downstream_warning": DOWNSTREAM_CAMERA_WARNING,
        "downstream_capture_families": list(DOWNSTREAM_CAPTURE_FAMILIES),
        "carrier_boundary": dict(CARRIER_BOUNDARY_FLAGS),
        "candidate_manifest_summary": {
            cid: {
                "topological_charge": m["topological_charge"],
                "hardware_command_export_status": m["hardware_command_export_status"],
                "bench_validation_status": m["bench_validation_status"],
                "physical_4f_readiness": m["physical_4f_readiness"],
            }
            for cid, m in candidate_manifests.items()
        },
        "downstream_config": asdict(downstream_config),
        "governance": _governance_record(),
        "final_export_allowed": False,
    }


def _first_screen_manifest(
    *,
    run_id: str,
    timestamp: str,
    config: FirstBenchScreenConfig,
    downstream_config: DownstreamCarrierStopConfig,
    candidate_manifests: Mapping[str, Mapping[str, Any]],
    capture_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    row_counts: dict[str, int] = {}
    for row in capture_rows:
        row_counts[str(row["capture_phase"])] = row_counts.get(str(row["capture_phase"]), 0) + 1
    return {
        "stage": STAGE,
        "study_name": config.study_name,
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "phases": list(PHASES),
        "capture_row_counts_by_phase": row_counts,
        "default_first_screen_candidate_ids": list(config.first_screen_candidate_ids),
        "optional_later_extension_candidate_ids": list(config.optional_extension_candidate_ids),
        "optional_extensions_included": bool(config.include_optional_extensions),
        "candidate_manifests": dict(candidate_manifests),
        "operator_set_required_fields": list(OPERATOR_SET_REQUIRED_FIELDS),
        "unknown_values_policy": "unknown_recorded is allowed but must be explicit",
        "baseline_carrier_selection_status": config.baseline_carrier_selection_status,
        "stop_motion_policy": config.stop_motion_policy,
        "raw_data_policy": config.raw_data_policy,
        "carrier_boundary": {
            **dict(CARRIER_BOUNDARY_FLAGS),
            "ideal_carrier_boundary": IDEAL_CARRIER_BOUNDARY,
        },
        "downstream_baseline": build_downstream_carrier_stop_study(downstream_config),
        "first_screen_boundary": FIRST_SCREEN_BOUNDARY_TEXT,
        "governance": _governance_record(),
        "final_export_allowed": False,
    }


def _write_experiment_package(
    exp_dir: Path,
    run_id: str,
    timestamp: str,
    config: FirstBenchScreenConfig,
    candidate_manifests: Mapping[str, Mapping[str, Any]],
    capture_rows: Sequence[Mapping[str, Any]],
) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "LAB_README_FIRST_BENCH_SCREEN.md").write_text(
        make_lab_readme(run_id, timestamp, config),
        encoding="utf-8",
    )
    (exp_dir / "operator_checklist.md").write_text(make_operator_checklist(run_id), encoding="utf-8")
    _write_csv(
        exp_dir / "as_found_bench_record.csv",
        ["required_field", "recorded_value", "value_status", "notes"],
        [
            {
                "required_field": field,
                "recorded_value": UNKNOWN_RECORDED,
                "value_status": UNKNOWN_RECORDED,
                "notes": "record before Phase B; unknown is allowed only if explicitly marked",
            }
            for field in OPERATOR_SET_REQUIRED_FIELDS
        ],
    )
    _write_csv(
        exp_dir / "carrier_stop_baseline_log.csv",
        [
            "capture_id",
            "capture_family",
            "carrier_cycles_x",
            "carrier_cycles_y",
            "pinhole_x_stage_reading",
            "pinhole_y_stage_reading",
            "pinhole_diameter_or_aperture",
            "visible_output",
            "saturation",
            "clipping",
            "operator_baseline_candidate",
            "notes",
        ],
        [
            {
                "capture_id": row["capture_id"],
                "capture_family": row["capture_family"],
                "carrier_cycles_x": row["carrier_cycles_x"],
                "carrier_cycles_y": row["carrier_cycles_y"],
            }
            for row in capture_rows
            if str(row["capture_phase"]) == "B_baseline_carrier_stop_check"
        ],
    )
    _write_csv(
        exp_dir / "candidate_screen_log.csv",
        [
            "capture_id",
            "candidate_id",
            "topological_charge",
            "SLM1 mask ID",
            "SLM2 mask ID",
            "observed_output_class",
            "relative_brightness_note",
            "centroid_or_position_note",
            "saturation",
            "clipping",
            "repeatability_note",
            "operator_preference",
            "notes",
        ],
        [
            {
                "capture_id": row["capture_id"],
                "candidate_id": row["candidate_id"],
                "topological_charge": row["topological_charge"],
                "SLM1 mask ID": row["SLM1 mask ID"],
                "SLM2 mask ID": row["SLM2 mask ID"],
            }
            for row in capture_rows
            if str(row["capture_phase"]) in {"C_candidate_screen", "D_optional_later_extension"}
        ],
    )
    _write_csv(
        exp_dir / "camera_capture_log.csv",
        [
            "capture_id",
            "raw_file_name",
            "camera_identifier",
            "camera_position_or_rail_reading",
            "exposure",
            "gain",
            "neutral_density_or_filter_state",
            "laser_energy_or_power_setting",
            "raw_file_saved",
            "raw_file_unmodified",
            "notes",
        ],
        [{"capture_id": row["capture_id"]} for row in capture_rows],
    )
    _write_csv(
        exp_dir / "candidate_observation_template.csv",
        [
            "candidate_id",
            "capture_ids",
            "usable_downstream_response",
            "observed_morphology",
            "repeatability",
            "saturation_or_clipping",
            "operator_keep_for_next_session",
            "reason",
            "notes",
        ],
        [
            {
                "candidate_id": cid,
                "capture_ids": ";".join(
                    str(row["capture_id"])
                    for row in capture_rows
                    if row.get("candidate_id") == cid
                ),
            }
            for cid in candidate_manifests
            if cid in DEFAULT_FIRST_SCREEN_CANDIDATE_IDS
        ],
    )
    (exp_dir / "operator_notes_template.md").write_text(
        "# Operator notes - first bench screen\n\n"
        f"- run_id: {run_id}\n"
        "- date:\n- operator:\n- SLM1 identifier:\n- SLM2 identifier:\n"
        "- camera identifier:\n- axicon state:\n- downstream optics state:\n\n"
        "## Observations\n\n"
        "Record only observed downstream behaviour. Do not infer material response, "
        "Zernike correction, camera calibration, or physical 4F readiness.\n",
        encoding="utf-8",
    )
    for cid, manifest in candidate_manifests.items():
        if cid not in DEFAULT_FIRST_SCREEN_CANDIDATE_IDS:
            continue
        (exp_dir / "candidate_summaries" / f"{cid}.md").write_text(
            make_candidate_summary_page(cid, manifest),
            encoding="utf-8",
        )


def make_lab_readme(run_id: str, timestamp: str, config: FirstBenchScreenConfig) -> str:
    return f"""# LAB README - First Bench Screen

Run: {run_id}
Timestamp: {timestamp}
Stage: {STAGE}

## Purpose

This package starts the first controlled bench screen after the downstream
carrier/stop baseline. It provides exact SLM command files and capture tables
for gaussian_reference, vortex_ell_1, and vortex_ell_2 only.

## Phase Order

1. Phase A: record the as-found bench state.
2. Phase B: run the downstream carrier/stop baseline capture families D0-D5.
3. Phase C: screen gaussian_reference, vortex_ell_1, and vortex_ell_2 with at
   least three repeats each.
4. Phase D: optional later extension only; vortex_ell_3 and vortex_ell_4 are
   not part of the default first-screen schedule.
5. Phase E: shutdown notes and raw-data handoff.

## Required Operator Records

Unknown values are allowed only as `unknown_recorded`. Record every field in
`as_found_bench_record.csv`, including actual SLM IDs, display orientation,
phase-LUT/wavelength setting, carrier orientation, pinhole readings, axicon
state, downstream optics, camera settings, filters, and run notes.

## Rules

- {config.stop_motion_policy}
- {config.raw_data_policy}
- Candidate masks are {HARDWARE_COMMAND_EXPORT_STATUS}.
- Bench validation status remains {BENCH_VALIDATION_STATUS}.
- Physical 4F readiness remains {PHYSICAL_4F_READINESS}.

## Boundary

{FIRST_SCREEN_BOUNDARY_TEXT}

{CLAIM_BOUNDARY}
"""


def make_operator_checklist(run_id: str) -> str:
    return f"""# Operator Checklist - {run_id}

## Before Light

- [ ] Open `as_found_bench_record.csv` and fill all known fields.
- [ ] Mark unknown values explicitly as `unknown_recorded`.
- [ ] Confirm SLM1 and SLM2 identifiers.
- [ ] Confirm display orientation and phase-LUT/wavelength setting.
- [ ] Record camera, filter, power/energy, axicon, pinhole, and downstream optics state.

## Phase B Baseline

- [ ] Capture D0 dark frames.
- [ ] Capture D1 flat reference.
- [ ] Capture D2 carrier sweep with the stop fixed.
- [ ] Move the stop only for D3/D4 and log every setting.
- [ ] Capture D5 repeatability references.
- [ ] Choose a usable baseline carrier/stop setting from observed response.

## Phase C Candidate Screen

- [ ] Capture gaussian_reference three times.
- [ ] Capture vortex_ell_1 three times.
- [ ] Capture vortex_ell_2 three times.
- [ ] Do not add vortex_ell_3 or vortex_ell_4 unless explicitly starting the optional extension.

## Raw Data

- [ ] Save raw images under the run raw directory.
- [ ] Do not edit raw images.
- [ ] Do not commit raw images by default.
"""


def make_candidate_summary_page(candidate_id: str, manifest: Mapping[str, Any]) -> str:
    return f"""# Candidate Summary - {candidate_id}

- candidate_family: {manifest['candidate_family']}
- topological_charge: {manifest['topological_charge']}
- upstream_source_mode: {manifest['upstream_source_mode']}
- slm1_phase_applied_at_slm1: {manifest['slm1_phase_applied_at_slm1']}
- slm1_to_slm2_propagation_included: {manifest['slm1_to_slm2_propagation_included']}
- slm2_carrier_mode: {manifest['slm2_carrier_mode']}
- carrier_realism: {manifest['carrier_realism']}
- stop_sampling_status: {manifest['stop_sampling_status']}
- convergence_status: {manifest['convergence_status']}
- hardware_command_export_status: {manifest['hardware_command_export_status']}
- bench_validation_status: {manifest['bench_validation_status']}
- physical_4f_readiness: {manifest['physical_4f_readiness']}

Use this page only as a command-mask and observation checklist. It is not a
camera prediction, physical 4F calibration, correction result, or material
prediction.
"""


def make_raw_data_policy_markdown(run_id: str, data_dir: Path) -> str:
    return f"""# Raw Data Policy

Run: {run_id}

Raw camera files should be saved by the operator under:

```text
{data_dir / "raw"}
```

{RAW_DATA_POLICY}

This folder contains only this policy file when the package is generated. Stage
9B.0.2 does not create fake raw images, process raw camera images, hash raw
images, or commit raw image data.
"""


def make_claim_boundary_markdown() -> str:
    return f"""# Stage 9B.0.2 First Bench Screen Claim Boundary

{FIRST_SCREEN_BOUNDARY_TEXT}

## What This Package Does

- creates exact command-mask files for the planned captures;
- records Phase A through Phase E operator workflow;
- preserves Stage 9A.1B downstream carrier/stop baseline capture families;
- preserves Stage 9B.0.1 candidate-mask ownership and convergence labels;
- gives raw-data placeholders and logs without processing raw images.

## What This Package Does Not Do

- no new optical physics;
- no physical 4F calibration or readiness;
- no camera model;
- no inverse correction, Zernike fitting, phase diversity, optimisation, or AI;
- no pixelated-SLM order physics, zero-order model, or order-efficiency model;
- no material-response, plasma, thermal, damage, or fused-silica prediction.

## Carrier Boundary

carrier_realism = `{CARRIER_REALISM}`

{IDEAL_CARRIER_BOUNDARY}
"""


def plot_first_bench_screen_overview(
    manifest: Mapping[str, Any] | None = None,
    *,
    output_path: str | Path = FIGURE_OVERVIEW,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    manifest = manifest or build_first_bench_screen_config()
    fig = plt.figure(figsize=(14.5, 8.2), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    fig.suptitle(
        "Stage 9B.0.2 First Bench Screen Package\n"
        "Command masks exportable but unvalidated; not a calibrated physical 4F or camera prediction",
        x=0.04,
        y=0.97,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    boxes = [
        (0.05, 0.68, 0.15, 0.15, "Phase A\nas-found bench\nrecord unknowns", "#e7f3ff"),
        (0.24, 0.68, 0.18, 0.15, "Phase B\nD0-D5 carrier/stop\nbaseline", "#fff0c4"),
        (0.46, 0.68, 0.18, 0.15, "Phase C\ngaussian, ell 1, ell 2\nthree repeats", "#e7f8df"),
        (0.68, 0.68, 0.15, 0.15, "Phase D\noptional later\nell 3, ell 4", "#f2e8ff"),
        (0.05, 0.40, 0.25, 0.16, "SLM1\nflat/vortex phase at SLM1\npropagates to SLM2", "#dceeff"),
        (0.37, 0.40, 0.25, 0.16, "SLM2\ncarrier-only command mask\nno vortex or axicon", "#dff4ef"),
        (0.69, 0.40, 0.24, 0.16, "Downstream camera\nempirical response only\nnot Fourier-plane calibration", "#ffe8e8"),
    ]
    for x, y, w, h, text, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#333333", lw=1.1, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, transform=ax.transAxes)
    for x0, x1 in ((0.20, 0.24), (0.42, 0.46), (0.64, 0.68)):
        ax.annotate("", xy=(x1, 0.755), xytext=(x0, 0.755), xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#333333"))
    ax.text(0.05, 0.26, "Required default candidates: gaussian_reference, vortex_ell_1, vortex_ell_2", fontsize=10, transform=ax.transAxes)
    ax.text(0.05, 0.20, "Operator-set values may be unknown only if recorded as unknown_recorded", fontsize=10, transform=ax.transAxes)
    ax.text(0.05, 0.14, "Raw policy: save unmodified camera files under the run raw directory; do not commit raw images by default", fontsize=9.5, transform=ax.transAxes)
    ax.text(0.05, 0.08, "Nominal unvalidated scenario | Ideal continuous-ramp carrier surrogate | Pixelated-SLM order physics not modelled", fontsize=9.2, family="monospace", transform=ax.transAxes)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", metadata={"Title": "Stage 9B.0.2 first bench screen overview", "final_export_allowed": "False"})
    plt.close(fig)
    return out


def plot_first_bench_screen_mask_atlas(
    candidate_manifests: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    package_dir: str | Path | None = None,
    output_path: str | Path = FIGURE_MASK_ATLAS,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    candidate_ids = list((candidate_manifests or {}).keys()) or list(DEFAULT_FIRST_SCREEN_CANDIDATE_IDS)
    candidate_ids = [cid for cid in candidate_ids if cid in DEFAULT_FIRST_SCREEN_CANDIDATE_IDS]
    if not candidate_ids:
        candidate_ids = list(DEFAULT_FIRST_SCREEN_CANDIDATE_IDS)
    cfg = NominalF300Config.exploratory()
    fig, axes = plt.subplots(len(candidate_ids), 2, figsize=(9.5, 3.0 * len(candidate_ids)), facecolor="white")
    axes_arr = np.asarray(axes).reshape(len(candidate_ids), 2)
    for row_axes, cid in zip(axes_arr, candidate_ids):
        spec = _candidate_spec(cid)
        run = simulate_candidate(spec, cfg)
        panels = ((run.slm1_phase_rad, "SLM1 phase"), (run.slm2_phase_rad, "SLM2 carrier only"))
        for ax, (arr, title) in zip(row_axes, panels):
            im = ax.imshow(arr, origin="lower", cmap="twilight", vmin=0.0, vmax=2.0 * np.pi)
            ax.set_title(f"{cid}: {title}", fontsize=9.5, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        "Stage 9B.0.2 First Bench Screen Mask Atlas\n"
        "Command masks exportable but unvalidated; SLM2 contains carrier only",
        fontsize=13.5,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.01,
        "Nominal unvalidated scenario | Ideal continuous-ramp carrier surrogate | Pixelated-SLM order physics not modelled",
        fontsize=8.6,
        family="monospace",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", metadata={"Title": "Stage 9B.0.2 first bench screen mask atlas", "final_export_allowed": "False"})
    plt.close(fig)
    return out


def write_stage9b02_static_artifacts(root: str | Path = ".") -> dict[str, Path]:
    """Write static config, figures, documentation, and summary artifacts."""

    root = Path(root)
    outputs = {
        "study_config": root / STUDY_PATH,
        "doc": root / "docs/49_first_bench_screen_package.md",
        "summary": root / "STAGE9B0_2_FIRST_BENCH_SCREEN_SUMMARY.md",
        "overview_figure": root / FIGURE_OVERVIEW,
        "mask_atlas_figure": root / FIGURE_MASK_ATLAS,
    }
    _write_json(outputs["study_config"], build_first_bench_screen_config())
    outputs["doc"].write_text(make_stage9b02_doc(), encoding="utf-8")
    outputs["summary"].write_text(make_stage9b02_summary(), encoding="utf-8")
    plot_first_bench_screen_overview(output_path=outputs["overview_figure"])
    plot_first_bench_screen_mask_atlas(output_path=outputs["mask_atlas_figure"])
    return outputs


def make_stage9b02_doc() -> str:
    return f"""# Stage 9B.0.2 First Bench Screen Package

Stage 9B.0.2 creates the first controlled bench-screen handoff package. It
does not change the Stage 9B.0/9B.0.1 nominal ranking, does not add optical
physics, and does not process raw camera images.

## Why This Exists

The nominal atlas is useful only after the bench records a practical baseline
carrier/stop state. This stage therefore combines:

- Stage 9A.1B downstream carrier/stop capture families D0-D5;
- Stage 9B.0.1 command-mask candidate outputs;
- operator templates for unknown bench values;
- raw-data placeholders that keep camera files immutable.

## Default Candidate Screen

The first-screen schedule is limited to:

- `gaussian_reference`
- `vortex_ell_1`
- `vortex_ell_2`

`vortex_ell_3` and `vortex_ell_4` are optional later extensions only. They are
not part of the default first-screen capture plan.

## Operator-Set Unknowns

The package requires actual SLM IDs, display orientation, phase-LUT/wavelength
setting, carrier orientation, lens/pinhole/camera/axicon/downstream optics
state, exposure/gain, filters, power/energy where available, date, operator,
and run notes. Unknown values are permitted only when explicitly recorded as
`unknown_recorded`.

## Carrier And Camera Boundary

{IDEAL_CARRIER_BOUNDARY}

The downstream camera capture is empirical. It does not directly calibrate the
Fourier plane, selected-order purity, physical order efficiency, or physical 4F
readiness.

## Raw Data Boundary

{RAW_DATA_POLICY}

## Unsupported

No pixelated-SLM order physics, physical 4F readiness, camera model, inverse
correction, Zernike fitting, phase diversity, AI, material response, plasma,
thermal, damage, or fused-silica prediction is enabled.
"""


def make_stage9b02_summary() -> str:
    return f"""# Stage 9B.0.2 First Bench Screen Package Summary

Starting checkpoint: Stage 9B.0.1 upstream bridge and stop sampling validity
(`92aadef`).

Stage 9B.0.2 adds a first bench screen package generator and static handoff
artifacts. It preserves the existing Stage 9B.0/9B.0.1 nominal rankings and
uses the atlas only as a command-mask sequence before the first bench session.

## Created

- `vbb_study/digital_twin/first_bench_screen.py`
- `configs/studies/cslm_first_bench_screen_v1.json`
- `docs/49_first_bench_screen_package.md`
- `notebooks/digital_twin/02_first_bench_screen_package.ipynb`
- `tests/test_stage9b0_2_first_bench_screen_package.py`
- `outputs/figures/digital_twin/stage9b0_2_first_bench_screen_overview.png`
- `outputs/figures/digital_twin/stage9b0_2_first_bench_screen_mask_atlas.png`

## Default Bench Screen

- Phase A: as-found bench record.
- Phase B: downstream carrier/stop baseline families D0-D5.
- Phase C: gaussian_reference, vortex_ell_1, vortex_ell_2 with at least three repeats.
- Phase D: optional later ell 3/ell 4 extension only.
- Phase E: shutdown notes and raw-data handoff.

## Boundary

{CLAIM_BOUNDARY}
"""


__all__ = [
    "CLAIM_BOUNDARY",
    "DEFAULT_FIRST_SCREEN_CANDIDATE_IDS",
    "FIGURE_MASK_ATLAS",
    "FIGURE_OVERVIEW",
    "FIRST_BENCH_CAPTURE_COLUMNS",
    "FirstBenchScreenConfig",
    "OPERATOR_SET_REQUIRED_FIELDS",
    "OPTIONAL_EXTENSION_CANDIDATE_IDS",
    "PHASES",
    "RAW_DATA_POLICY",
    "STAGE",
    "STUDY_NAME",
    "build_first_bench_capture_plan_rows",
    "build_first_bench_screen_config",
    "create_first_bench_screen_package",
    "make_claim_boundary_markdown",
    "make_raw_data_policy_markdown",
    "plot_first_bench_screen_mask_atlas",
    "plot_first_bench_screen_overview",
    "validate_first_bench_capture_rows",
    "write_first_bench_screen_config",
    "write_stage9b02_static_artifacts",
]


if __name__ == "__main__":
    for key, value in write_stage9b02_static_artifacts(Path.cwd()).items():
        print(f"{key}: {value}")
