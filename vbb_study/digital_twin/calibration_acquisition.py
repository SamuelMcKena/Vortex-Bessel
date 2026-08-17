"""Stage 9A calibration acquisition package, campaign plans, and raw-data ingestion.

This builds the *acquisition and storage* tooling for a real optical calibration
campaign of the CSLM -> 4F -> physical-axicon route.  It does NOT implement any
optical physics: no thin-lens / physical 4F propagation, no +1-order field, no
camera-imaging model, no material response, no inverse correction, no AI.

Boundary (unchanged): n = 1.0 free-space optical-field / fluence diagnostics
only; ``fourier_filter_physics_available = False``; ``camera_model_enabled =
False``; ``material_model_enabled = False``; ``diagnostic_only = True``;
``final_export_allowed = False``.

Raw camera files are treated as immutable source evidence.  Derived preprocessing
is always written separately and recorded; raw data is never overwritten.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from vbb_study.digital_twin.cslm_route import CSLMRouteConfig
from vbb_study.digital_twin.control_contract import build_default_demo_profile
from vbb_study.digital_twin.bench_inventory import (
    build_bench_inventory_profile,
    evaluate_physical_4f_readiness,
)
from vbb_study.digital_twin.coordinate_contract import (
    coordinate_frame_rows,
    coordinate_transform_rows,
)

GOVERNANCE = {
    "fourier_filter_physics_available": False,
    "camera_model_enabled": False,
    "material_model_enabled": False,
    "diagnostic_only": True,
    "final_export_allowed": False,
    "n_medium": 1.0,
}

CLAIM_BOUNDARY = (
    "n=1.0 free-space optical/fluence diagnostic; acquisition & ingestion only; no physical 4F "
    "field, no camera-imaging model, no inverse correction, no AI, no material model; "
    "final_export_allowed=False"
)

CaptureKind = Literal[
    "dark_frame", "flat_field", "input_beam", "fourier_plane_carrier_sweep",
    "fourier_stop_scan", "post_axicon_xy", "post_axicon_z_stack",
    "energy_measurement", "alignment_reference", "manual_observation",
    "downstream_carrier_sweep_fixed_stop", "downstream_stop_position_sweep",
    "downstream_stop_radius_or_aperture_sweep", "downstream_repeatability_reference",
]

DataStatus = Literal[
    "planned", "acquired_unverified", "ingested", "quality_checked",
    "coordinate_calibrated", "analysis_ready", "rejected",
]

VALID_CAPTURE_KINDS: frozenset[str] = frozenset(CaptureKind.__args__)  # type: ignore[attr-defined]
VALID_DATA_STATUSES: frozenset[str] = frozenset(DataStatus.__args__)  # type: ignore[attr-defined]

# Critical capture-manifest fields that must be present (non-empty) for a real capture.
REQUIRED_MANIFEST_FIELDS: tuple[str, ...] = (
    "capture_id", "capture_kind", "run_id", "file_path", "camera_frame_id",
    "image_units", "profile_name", "route_mode", "order_handoff_mode",
)


@dataclass(frozen=True)
class CalibrationCapture:
    capture_id: str
    capture_kind: str
    run_id: str
    file_path: str | None
    raw_file_sha256: str | None
    capture_status: str
    timestamp_utc: str | None
    camera_id: str | None
    camera_frame_id: str
    image_units: str
    z_position_mm: float | None
    exposure_us: float | None
    gain: float | None
    saturation_fraction: float | None
    background_reference_id: str | None
    profile_name: str
    route_mode: str
    order_handoff_mode: str
    slm1_mask_id: str | None
    slm2_mask_id: str | None
    topological_charge: int | None
    carrier_frequency_cpm: float | None
    physical_axicon_enabled: bool | None
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def sha256_of_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit_hash(repo: str | Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo) if repo else None,
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def generate_run_id(prefix: str = "cslmcal") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Work package B — calibration campaign v1 (families 0-6)
# ---------------------------------------------------------------------------


def build_calibration_campaign_v1() -> dict[str, Any]:
    """The canonical initial calibration campaign (families 0-6)."""
    families = [
        {
            "family_id": 0, "name": "dark_background_reference",
            "capture_kind": "dark_frame",
            "purpose": "background subtraction; saturation/quality control; repeatability",
            "identifies": "camera background, noise floor, saturation reference",
            "coordinate_frame": "camera_sensor_pixel_frame",
            "captures": [
                {"capture_id": "F0_dark_lowgain", "capture_kind": "dark_frame",
                 "settings": {"beam": "shuttered", "repeats": 5}},
                {"capture_id": "F0_dark_opgain", "capture_kind": "dark_frame",
                 "settings": {"beam": "shuttered", "repeats": 5}},
            ],
        },
        {
            "family_id": 1, "name": "input_beam_record",
            "capture_kind": "input_beam",
            "purpose": "beam centre/diameter/ellipticity/rotation/aperture context (pixels only)",
            "identifies": "input beam shape in pixels (no calibrated physical radius yet)",
            "coordinate_frame": "camera_sensor_pixel_frame",
            "note": "Do not claim a calibrated physical beam radius until camera/object-plane mapping exists.",
            "captures": [
                {"capture_id": "F1_input_beam", "capture_kind": "input_beam",
                 "settings": {"slm1": "flat", "location": "before_route_if_accessible"}},
            ],
        },
        {
            "family_id": 2, "name": "slm2_carrier_fourier_mapping",
            "capture_kind": "fourier_plane_carrier_sweep",
            "purpose": "carrier sign convention; carrier-frequency->order-position mapping; "
                       "Fourier-stop centring; Fourier-plane coordinate calibration",
            "identifies": "the highest-priority calibration: SLM2 carrier -> Fourier order position",
            "coordinate_frame": "Fourier_plane_physical_position_frame",
            "priority": "highest",
            "captures": [
                {"capture_id": f"F2_carrier_x_{i}", "capture_kind": "fourier_plane_carrier_sweep",
                 "settings": {"slm1": "flat", "carrier_axis": "x", "carrier_cpm": cpm,
                              "record_orders": ["zero", "+1", "-1"]}}
                for i, cpm in enumerate((-30000.0, -15000.0, 0.0, 15000.0, 30000.0))
            ] + [
                {"capture_id": f"F2_carrier_y_{i}", "capture_kind": "fourier_plane_carrier_sweep",
                 "settings": {"slm1": "flat", "carrier_axis": "y", "carrier_cpm": cpm,
                              "record_orders": ["zero", "+1", "-1"]}}
                for i, cpm in enumerate((-30000.0, -15000.0, 0.0, 15000.0, 30000.0))
            ],
        },
        {
            "family_id": 3, "name": "fourier_stop_scan",
            "capture_kind": "fourier_stop_scan",
            "purpose": "practical stop placement; clipping/sensitivity; selected-order operating point",
            "identifies": "measured stop operating point (no stop model in this stage)",
            "coordinate_frame": "Fourier_plane_physical_position_frame",
            "note": "Measured-data procedure only; do not model the stop yet.",
            "captures": [
                {"capture_id": f"F3_stop_{ax}_{i}", "capture_kind": "fourier_stop_scan",
                 "settings": {"selected_carrier": "from_F2", "scan": ax, "step_index": i}}
                for ax in ("centre_x", "centre_y", "radius") for i in range(3)
            ],
        },
        {
            "family_id": 4, "name": "gaussian_through_axicon_baseline",
            "capture_kind": "post_axicon_z_stack",
            "purpose": "Gaussian-through-physical-axicon baseline z-stack",
            "identifies": "axicon Bessel formation for a Gaussian input (pixel z-stack)",
            "coordinate_frame": "camera_sensor_pixel_frame",
            "captures": [
                {"capture_id": f"F4_gauss_z{i:02d}", "capture_kind": "post_axicon_z_stack",
                 "settings": {"slm1": "flat", "order_handoff_mode": "ideal_selected_order_surrogate",
                              "z_index": i}}
                for i in range(11)
            ],
        },
        {
            "family_id": 5, "name": "vortex_through_axicon_baseline_atlas",
            "capture_kind": "post_axicon_z_stack",
            "purpose": "vortex-through-physical-axicon baseline atlas (l=1,2,3)",
            "identifies": "charge-dependent annular Bessel formation (pixel z-stacks)",
            "coordinate_frame": "camera_sensor_pixel_frame",
            "note": "Only the deliberate variable (topological charge) may differ across atlas cases.",
            "captures": [
                {"capture_id": f"F5_l{ell}_z{i:02d}", "capture_kind": "post_axicon_z_stack",
                 "settings": {"slm1": "vortex", "topological_charge": ell,
                              "order_handoff_mode": "ideal_selected_order_surrogate", "z_index": i}}
                for ell in (1, 2, 3) for i in range(11)
            ],
        },
        {
            "family_id": 6, "name": "future_controlled_perturbation_placeholders",
            "capture_kind": "manual_observation",
            "purpose": "planned future controlled perturbations (NOT executed in this stage)",
            "identifies": "future correction-study variables",
            "coordinate_frame": "lab_beam_frame",
            "status": "planned_future_calibration",
            "implementation": "not_implemented_in_current_stage",
            "captures": [
                {"capture_id": f"F6_future_{name}", "capture_kind": "manual_observation",
                 "status": "planned_future_calibration",
                 "implementation": "not_implemented_in_current_stage",
                 "settings": {"perturbation": name}}
                for name in ("tip_tilt", "defocus", "astigmatism_x", "astigmatism_y",
                             "coma_x", "coma_y", "trefoil_x", "trefoil_y",
                             "mask_centre_x_offset", "mask_centre_y_offset",
                             "carrier_frequency_offset", "axicon_x_offset", "axicon_y_offset")
            ],
        },
    ]
    return {
        "campaign_name": "cslm_physical_axicon_calibration_campaign_v1",
        "campaign_status": "planned_calibration_campaign_not_yet_acquired",
        "claim_boundary": CLAIM_BOUNDARY,
        "governance": dict(GOVERNANCE),
        "families": families,
    }


def load_calibration_campaign(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _campaign_plan_rows(campaign: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fam in campaign["families"]:
        for cap in fam["captures"]:
            rows.append({
                "capture_id": cap["capture_id"],
                "family_id": fam["family_id"],
                "family_name": fam["name"],
                "capture_kind": cap.get("capture_kind", fam.get("capture_kind", "")),
                "capture_status": cap.get("status", "planned"),
                "coordinate_frame": fam.get("coordinate_frame", "camera_sensor_pixel_frame"),
                "purpose": fam.get("purpose", ""),
                "settings": json.dumps(cap.get("settings", {})),
            })
    return rows


# ---------------------------------------------------------------------------
# Work package C — acquisition package generator
# ---------------------------------------------------------------------------


def create_calibration_acquisition_package(
    run_id: str | None = None,
    *,
    config: CSLMRouteConfig | None = None,
    campaign: Mapping[str, Any] | None = None,
    output_root: str | Path = "outputs/calibration_runs",
    data_root: str | Path = "data/calibration_runs",
    repo: str | Path | None = None,
) -> dict[str, Any]:
    """Create the canonical acquisition package for one calibration run.

    Does not overwrite an existing run directory.
    """
    config = config or CSLMRouteConfig()
    campaign = campaign or build_calibration_campaign_v1()
    run_id = run_id or generate_run_id()

    out_dir = Path(output_root) / run_id
    data_dir = Path(data_root) / run_id
    if out_dir.exists() or data_dir.exists():
        raise FileExistsError(f"calibration run already exists: {run_id}")
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    exp_dir = out_dir / "experiment_package"
    exp_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("raw", "manifests", "derived", "figures"):
        (data_dir / sub).mkdir(parents=True, exist_ok=True)

    profile = build_default_demo_profile(config)
    readiness = evaluate_physical_4f_readiness(config)

    run_manifest = {
        "run_id": run_id,
        "timestamp_utc": utc_now_iso(),
        "git_commit": git_commit_hash(repo),
        "campaign_name": campaign.get("campaign_name"),
        "campaign_status": campaign.get("campaign_status"),
        "profile_name": profile["profile_name"],
        "profile_status": profile["profile_status"],
        "route_mode": config.route_mode,
        "order_handoff_mode": config.order_handoff_mode,
        "control_values": {k: v["value"] for k, v in profile["controls"].items()},
        "control_units": {k: v["unit"] for k, v in profile["controls"].items()},
        "control_status": {k: v["status"] for k, v in profile["controls"].items()},
        "control_provenance": {k: v["provenance"] for k, v in profile["controls"].items()},
        "governance": dict(GOVERNANCE),
        "physical_4f_status": "not_implemented_blocked",
        "camera_status": "no_camera_model_metadata_only",
        "material_model_status": "disabled",
        "physical_4f_readiness_levels": {
            k: readiness[k]["ready"] for k in
            ("A_active_cslm_diagnostic", "B_ideal_axicon_benchmark",
             "C_initial_scalar_4f_model", "D_measured_bench_camera")
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "raw_data_dir": str(data_dir),
        "raw_data_policy": "raw camera files are immutable source evidence; not committed by default",
    }

    paths: dict[str, Any] = {}
    paths["run_manifest"] = out_dir / "run_manifest.json"
    _write_json(paths["run_manifest"], run_manifest)

    plan_rows = _campaign_plan_rows(campaign)
    paths["acquisition_plan"] = out_dir / "acquisition_plan.csv"
    _write_csv(paths["acquisition_plan"],
               ["capture_id", "family_id", "family_name", "capture_kind", "capture_status",
                "coordinate_frame", "purpose", "settings"], plan_rows)

    paths["capture_manifest_template"] = out_dir / "capture_manifest_template.csv"
    _write_csv(paths["capture_manifest_template"], _CAPTURE_MANIFEST_COLUMNS,
               [{"capture_id": r["capture_id"], "capture_kind": r["capture_kind"],
                 "run_id": run_id, "capture_status": "planned",
                 "camera_frame_id": r["coordinate_frame"], "image_units": "pixel",
                 "profile_name": profile["profile_name"], "route_mode": config.route_mode,
                 "order_handoff_mode": config.order_handoff_mode} for r in plan_rows])

    paths["hardware_profile_snapshot"] = out_dir / "hardware_profile_snapshot.json"
    _write_json(paths["hardware_profile_snapshot"], profile)
    paths["bench_inventory_snapshot"] = out_dir / "bench_inventory_snapshot.json"
    _write_json(paths["bench_inventory_snapshot"], build_bench_inventory_profile(config))
    paths["coordinate_contract_snapshot"] = out_dir / "coordinate_contract_snapshot.json"
    _write_json(paths["coordinate_contract_snapshot"],
                {"frames": coordinate_frame_rows(), "transforms": coordinate_transform_rows()})

    _write_experiment_package(exp_dir, run_id, run_manifest, plan_rows)
    paths["experiment_package_dir"] = exp_dir
    paths["run_dir"] = out_dir
    paths["data_dir"] = data_dir
    return paths


_CAPTURE_MANIFEST_COLUMNS = (
    "capture_id", "capture_kind", "run_id", "file_path", "raw_file_sha256", "capture_status",
    "timestamp_utc", "camera_id", "camera_frame_id", "image_units", "z_position_mm",
    "exposure_us", "gain", "saturation_fraction", "background_reference_id", "profile_name",
    "route_mode", "order_handoff_mode", "slm1_mask_id", "slm2_mask_id", "topological_charge",
    "carrier_frequency_cpm", "physical_axicon_enabled", "notes",
)


def _write_experiment_package(exp_dir: Path, run_id: str,
                              run_manifest: Mapping[str, Any], plan_rows) -> None:
    (exp_dir / "bench_setup_sheet.md").write_text(
        f"# Bench setup sheet — run {run_id}\n\n"
        f"- timestamp: {run_manifest['timestamp_utc']}\n"
        f"- git commit: {run_manifest['git_commit']}\n"
        f"- profile: {run_manifest['profile_name']} ({run_manifest['profile_status']})\n"
        f"- route_mode: {run_manifest['route_mode']}\n"
        f"- order_handoff_mode: {run_manifest['order_handoff_mode']}\n"
        f"- physical_4f_status: {run_manifest['physical_4f_status']}\n"
        f"- camera_status: {run_manifest['camera_status']}\n"
        f"- material_model_status: {run_manifest['material_model_status']}\n\n"
        f"Claim boundary: {run_manifest['claim_boundary']}\n\n"
        "Record bench positions, mounts, and SLM upload IDs here. Raw camera files are immutable.\n",
        encoding="utf-8")
    _write_csv(exp_dir / "bench_setup_sheet.csv",
               ["component", "setting", "value", "unit", "provenance", "notes"],
               [{"component": "claim_boundary", "setting": "stage", "value": "9A",
                 "provenance": "derived", "notes": run_manifest["claim_boundary"]}])
    _write_csv(exp_dir / "camera_capture_checklist.csv",
               ["capture_id", "capture_kind", "acquired", "file_name", "exposure_us", "gain",
                "saturation_ok", "background_reference_id", "operator_initials", "notes"],
               [{"capture_id": r["capture_id"], "capture_kind": r["capture_kind"]} for r in plan_rows])
    _write_csv(exp_dir / "energy_measurement_log.csv",
               ["timestamp_utc", "location", "meter_id", "reading", "unit", "repeats",
                "uncertainty", "notes"], [])
    _write_csv(exp_dir / "physical_axicon_alignment_log.csv",
               ["timestamp_utc", "axicon_id", "centre_x", "centre_y", "axial_position",
                "tip_tilt", "unit", "method", "notes"], [])
    _write_csv(exp_dir / "fused_silica_pilot_observation_template.csv", _FUSED_SILICA_FIELDS, [])
    (exp_dir / "operator_notes_template.md").write_text(
        f"# Operator notes — run {run_id}\n\n- date:\n- operator:\n- conditions:\n\n"
        "## Observations (neutral, no calculated material predictions)\n\n",
        encoding="utf-8")


_FUSED_SILICA_FIELDS = (
    "sample_id", "material_grade", "sample_dimensions", "surface_preparation",
    "pulse_energy_setting", "repetition_rate", "scan_speed_or_stationary_exposure",
    "focus_or_sample_position", "observed_track_continuity", "observed_feature_symmetry",
    "observed_modification_morphology", "surface_effect", "void_or_crack_presence",
    "etch_response_if_measured", "weld_feature_appearance_if_applicable",
    "microscope_file_path", "operator_notes",
)


# ---------------------------------------------------------------------------
# Work package D — raw camera-data ingestion (raw is immutable)
# ---------------------------------------------------------------------------


def validate_capture_manifest(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return validation issues for a capture manifest (empty == valid)."""
    issues: list[str] = []
    seen: set[str] = set()
    for i, row in enumerate(rows):
        cid = str(row.get("capture_id", "")).strip()
        if not cid:
            issues.append(f"row {i}: missing capture_id")
        elif cid in seen:
            issues.append(f"row {i}: duplicate capture_id {cid!r}")
        else:
            seen.add(cid)
        kind = str(row.get("capture_kind", "")).strip()
        if kind and kind not in VALID_CAPTURE_KINDS:
            issues.append(f"{cid}: invalid capture_kind {kind!r}")
        status = str(row.get("capture_status", "")).strip()
        if status and status not in VALID_DATA_STATUSES:
            issues.append(f"{cid}: invalid capture_status {status!r}")
        # Critical fields must be present for a non-planned capture.
        if status not in ("", "planned"):
            for fld in REQUIRED_MANIFEST_FIELDS:
                if not str(row.get(fld, "")).strip():
                    issues.append(f"{cid}: missing required field {fld!r} for status {status!r}")
    return issues


def ingest_calibration_capture(
    run_data_dir: str | Path,
    source_path: str | Path,
    *,
    capture_id: str,
    capture_kind: str,
    run_id: str,
    camera_frame_id: str = "camera_sensor_pixel_frame",
    profile_name: str = "cslm_physical_axicon_demo_profile",
    route_mode: str = "holographic_cslm",
    order_handoff_mode: str = "none",
    image_units: str = "pixel",
    camera_id: str | None = None,
    z_position_mm: float | None = None,
    exposure_us: float | None = None,
    gain: float | None = None,
    background_reference_id: str | None = None,
    slm1_mask_id: str | None = None,
    slm2_mask_id: str | None = None,
    topological_charge: int | None = None,
    carrier_frequency_cpm: float | None = None,
    physical_axicon_enabled: bool | None = None,
    notes: str = "",
    copy_raw: bool = True,
) -> CalibrationCapture:
    """Ingest one raw capture: copy into the immutable raw/ dir and record sha256.

    The source file is never modified.  ``capture_kind`` must be valid.
    """
    if capture_kind not in VALID_CAPTURE_KINDS:
        raise ValueError(f"invalid capture_kind {capture_kind!r}")
    src = Path(source_path)
    if not src.is_file():
        raise FileNotFoundError(f"source capture not found: {src}")
    raw_dir = Path(run_data_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sha = sha256_of_file(src)
    dest = raw_dir / f"{capture_id}{src.suffix.lower()}"
    if copy_raw:
        if dest.exists():
            raise FileExistsError(f"raw capture already ingested: {dest}")
        dest.write_bytes(src.read_bytes())  # byte-for-byte copy; never transform raw
        assert sha256_of_file(dest) == sha, "raw copy checksum mismatch"
        stored = dest
    else:
        stored = src

    return CalibrationCapture(
        capture_id=capture_id, capture_kind=capture_kind, run_id=run_id,
        file_path=str(stored), raw_file_sha256=sha, capture_status="ingested",
        timestamp_utc=utc_now_iso(), camera_id=camera_id, camera_frame_id=camera_frame_id,
        image_units=image_units, z_position_mm=z_position_mm, exposure_us=exposure_us,
        gain=gain, saturation_fraction=None, background_reference_id=background_reference_id,
        profile_name=profile_name, route_mode=route_mode, order_handoff_mode=order_handoff_mode,
        slm1_mask_id=slm1_mask_id, slm2_mask_id=slm2_mask_id, topological_charge=topological_charge,
        carrier_frequency_cpm=carrier_frequency_cpm, physical_axicon_enabled=physical_axicon_enabled,
        notes=notes,
    )


def write_capture_manifest(run_dir: str | Path, captures: Sequence[CalibrationCapture]) -> Path:
    path = Path(run_dir) / "capture_manifest.csv"
    _write_csv(path, _CAPTURE_MANIFEST_COLUMNS, [c.as_dict() for c in captures])
    return path


def ingest_calibration_run(
    run_data_dir: str | Path,
    captures: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
) -> list[CalibrationCapture]:
    """Ingest a list of capture descriptors (each must carry source_path + capture_id/kind)."""
    out: list[CalibrationCapture] = []
    for c in captures:
        c = dict(c)
        src = c.pop("source_path")
        out.append(ingest_calibration_capture(run_data_dir, src, run_id=run_id, **c))
    return out


def save_derived_artifact(
    run_data_dir: str | Path,
    capture_id: str,
    op_name: str,
    array,
    *,
    params: Mapping[str, Any] | None = None,
) -> Path:
    """Save a derived (preprocessed) artefact separately and record it. Never touches raw/."""
    import numpy as np
    derived_dir = Path(run_data_dir) / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    out = derived_dir / f"{capture_id}__{op_name}.npy"
    np.save(out, np.asarray(array))
    manifest = derived_dir / "processing_manifest.csv"
    rows = _read_csv(manifest) if manifest.exists() else []
    rows.append({
        "capture_id": capture_id, "op_name": op_name, "derived_file": str(out),
        "params": json.dumps(dict(params or {})), "timestamp_utc": utc_now_iso(),
        "note": "derived artefact; raw data unchanged",
    })
    _write_csv(manifest, ["capture_id", "op_name", "derived_file", "params", "timestamp_utc", "note"], rows)
    return out


# ---------------------------------------------------------------------------
# diagnostic campaign-overview figure
# ---------------------------------------------------------------------------


def plot_calibration_campaign_overview(
    campaign: Mapping[str, Any] | None = None,
    *,
    config: CSLMRouteConfig | None = None,
    output_path: str | Path | None = None,
    dpi: int = 160,
):
    """Diagnostic-only figure: capture families, planes, data-status pipeline, claim boundary."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    campaign = campaign or build_calibration_campaign_v1()
    readiness = evaluate_physical_4f_readiness(config or CSLMRouteConfig())

    fig = plt.figure(figsize=(15.5, 9.0), facecolor="white")
    gs = fig.add_gridspec(2, 2, left=0.05, right=0.97, top=0.86, bottom=0.06, hspace=0.30, wspace=0.12)
    fig.suptitle("Stage 9A Calibration Campaign Overview\n"
                 "acquisition & ingestion only; n=1.0 free-space; no physical 4F / camera / material "
                 "model; final_export_allowed=False",
                 x=0.04, y=0.975, ha="left", va="top", fontsize=14, fontweight="bold")

    ax = fig.add_subplot(gs[0, :]); ax.set_axis_off()
    ax.text(0.0, 1.0, "Capture families -> required plane -> what it identifies", fontsize=12,
            fontweight="bold", va="top")
    for i, fam in enumerate(campaign["families"]):
        y = 0.86 - i * 0.125
        ncap = len(fam["captures"])
        ax.text(0.0, y, f"F{fam['family_id']} {fam['name']} ({ncap})", fontsize=9.5,
                fontweight="bold", va="top")
        ax.text(0.40, y, fam.get("coordinate_frame", ""), fontsize=8.5, va="top", color="#1565c0")
        ax.text(0.74, y, fam.get("identifies", "")[:46], fontsize=8.2, va="top", color="#444")
        if fam.get("status") == "planned_future_calibration":
            ax.text(0.40, y - 0.05, "[planned_future_calibration / not_implemented]", fontsize=7.5,
                    va="top", color="#b71c1c")

    ax = fig.add_subplot(gs[1, 0]); ax.set_axis_off()
    pipeline = ["planned", "acquired_unverified", "ingested", "quality_checked",
                "coordinate_calibrated", "analysis_ready", "(rejected)"]
    ax.text(0.0, 1.0, "Data-status pipeline", fontsize=12, fontweight="bold", va="top")
    for i, s in enumerate(pipeline):
        ax.text(0.04, 0.85 - i * 0.12, f"{i+1}. {s}", fontsize=9.5, va="top")

    ax = fig.add_subplot(gs[1, 1]); ax.set_axis_off()
    lines = [
        "Physical-4F readiness relation:",
        f"  A active diagnostic : {readiness['A_active_cslm_diagnostic']['ready']}",
        f"  B axicon benchmark  : {readiness['B_ideal_axicon_benchmark']['ready']}",
        f"  C initial scalar 4F : {readiness['C_initial_scalar_4f_model']['ready']} (blocked)",
        f"  D measured bench    : {readiness['D_measured_bench_camera']['ready']} (blocked)",
        "",
        "Family 2 (carrier->Fourier mapping) is the highest-priority",
        "family: it unblocks the Fourier-plane coordinate convention.",
        "Family 4/5 z-stacks feed later effective-aberration study.",
        "",
        "CLAIM: " + CLAIM_BOUNDARY[:60],
        "       " + CLAIM_BOUNDARY[60:120],
    ]
    ax.text(0.0, 1.0, "\n".join(lines), fontsize=8.6, va="top", family="monospace")

    if output_path is not None:
        out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight", metadata={
            "Title": "Stage 9A calibration campaign overview", "final_export_allowed": "False"})
    return fig
