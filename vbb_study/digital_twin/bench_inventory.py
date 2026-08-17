"""Stage 8C.3R.5.3 measured-bench inventory and physical-4F readiness gate.

Turns the R5.2 editable-control registry into a disciplined bench record (with
evidence/provenance/coordinate-frame metadata) and a four-level readiness report
that answers: *do we have enough physically defined, unit-consistent,
provenance-labelled information to begin a component-owned scalar thin-lens 4F
model?*

This implements NO optical transform: no thin-lens, no Fourier propagation, no
+1-order field, no camera physics.  It only records inventory and evaluates
readiness.  Boundary unchanged: n = 1.0 free-space; ``fourier_filter_physics_
available = False``; ``diagnostic_only``; ``final_export_allowed = False``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from vbb_study.digital_twin.cslm_route import CSLMRouteConfig
from vbb_study.digital_twin.control_contract import (
    build_cslm_editable_control_registry,
    EditableControl,
)
from vbb_study.digital_twin.coordinate_contract import (
    build_coordinate_frames,
    build_coordinate_transforms,
    fourier_position_mapping_known,
    slm2_to_lab_mapping_known,
    camera_to_lab_mapping_known,
)

INVENTORY_PROFILE_NAME = "cslm_physical_axicon_bench_inventory"
INVENTORY_PROFILE_STATUS = "diagnostic_demo_inventory_not_measured_bench"
DOWNSTREAM_EMPIRICAL_EVIDENCE_STATE = "downstream_empirical_carrier_stop_response_available"

# Hard blockers for an *initial* component-owned scalar 4F model (control ids).
PHYSICAL_4F_HARD_BLOCKERS: tuple[str, ...] = (
    "wavelength_nm",
    "slm2_pixel_pitch_um",                  # SLM2 transverse coordinate scale
    "slm2_carrier_frequency_cpm",           # carrier freq in physical SLM coordinates
    "slm2_to_lens1_distance_mm",
    "fourier_lens1_focal_length_mm",
    "fourier_lens1_clear_aperture_mm",
    "lens1_to_fourier_plane_distance_mm",
    "fourier_filter_centre_x_um",
    "fourier_filter_centre_y_um",
    "fourier_filter_radius_um",
    "fourier_filter_shape",
    "fourier_plane_to_lens2_distance_mm",
    "fourier_lens2_focal_length_mm",
    "fourier_lens2_clear_aperture_mm",
    "lens2_to_output_plane_distance_mm",
)

# Benchmark-branch active requirements (control ids).
BENCHMARK_REQUIRED: tuple[str, ...] = (
    "wavelength_nm", "grid_N", "slm1_phase_mode", "slm1_topological_charge",
    "slm1_to_slm2_distance_mm", "slm2_correction_phase_rad", "slm2_carrier_frequency_cpm",
    "physical_axicon_cone_parameter_rad_per_um", "physical_axicon_clear_aperture_radius_um",
    "physical_axicon_to_benchmark_reference_distance_mm",
)

# Additional measured-bench / camera-comparison requirements (control ids).
MEASURED_BENCH_EXTRA: tuple[str, ...] = (
    "physical_axicon_centre_x_um", "physical_axicon_centre_y_um",
    "physical_axicon_axial_offset_um", "physical_axicon_mechanical_tip_tilt_mrad",
    "camera_pixel_pitch_um", "camera_magnification", "camera_calibration_status",
    "reference_plane_definition",
)

# Map control groups -> a representative coordinate frame for inventory display.
_GROUP_FRAME = {
    "source_and_input_beam": "lab_beam_frame",
    "slm1": "SLM1_pixel_frame",
    "slm2": "SLM2_pixel_frame",
    "inter_slm_and_4f_geometry": "Fourier_plane_physical_position_frame",
    "physical_axicon_benchmark": "physical_axicon_local_frame",
    "camera_reference_plane": "camera_sensor_pixel_frame",
    "route_and_governance": "lab_beam_frame",
    "advanced_numerical": "lab_beam_frame",
}

_GROUP_COMPONENT_TYPE = {
    "source_and_input_beam": "source",
    "slm1": "spatial_light_modulator",
    "slm2": "spatial_light_modulator",
    "inter_slm_and_4f_geometry": "relay_optic_or_stop",
    "physical_axicon_benchmark": "physical_axicon",
    "camera_reference_plane": "camera",
    "route_and_governance": "governance",
    "advanced_numerical": "numerical",
}


@dataclass(frozen=True)
class BenchInventoryItem:
    component_id: str
    display_name: str
    component_type: str
    route_location: str
    value: Any
    unit: str
    provenance: str
    source_type: str
    source_reference: str | None
    recorded_date: str | None
    uncertainty: Any
    coordinate_frame: str
    status: str
    required_for_initial_4F_model: bool
    required_for_measured_bench_prediction: bool
    notes: str

    def as_row(self) -> dict[str, Any]:
        return {
            "component": self.component_id,
            "value": self.value,
            "unit": self.unit,
            "type": self.component_type,
            "route_location": self.route_location,
            "coordinate_frame": self.coordinate_frame,
            "provenance": self.provenance,
            "source": self.source_type,
            "status": self.status,
            "req_4F": self.required_for_initial_4F_model,
            "req_measured_bench": self.required_for_measured_bench_prediction,
        }

    def as_json(self) -> dict[str, Any]:
        return {
            "value": self.value, "unit": self.unit, "provenance": self.provenance,
            "source_type": self.source_type, "source_reference": self.source_reference,
            "recorded_date": self.recorded_date, "uncertainty": self.uncertainty,
            "coordinate_frame": self.coordinate_frame, "status": self.status,
            "component_type": self.component_type, "route_location": self.route_location,
            "required_for_initial_4F_model": self.required_for_initial_4F_model,
            "required_for_measured_bench_prediction": self.required_for_measured_bench_prediction,
            "notes": self.notes,
        }


def _evidence_for(control: EditableControl, overlay: Mapping[str, Any] | None) -> dict[str, Any]:
    entry = (overlay or {}).get(control.control_id) if overlay else None
    if isinstance(entry, Mapping):
        return {
            "value": entry.get("value", control.value),
            "provenance": entry.get("provenance", control.provenance),
            "source_type": entry.get("source_type",
                                     _default_source_type(entry.get("provenance", control.provenance))),
            "source_reference": entry.get("source_reference"),
            "recorded_date": entry.get("recorded_date"),
            "uncertainty": entry.get("uncertainty"),
            "notes": entry.get("notes", control.description),
        }
    return {
        "value": control.value,
        "provenance": control.provenance,
        "source_type": _default_source_type(control.provenance),
        "source_reference": None,
        "recorded_date": None,
        "uncertainty": None,
        "notes": control.description,
    }


def _default_source_type(provenance: str) -> str:
    return {
        "diagnostic_placeholder": "repository_demo_config_default",
        "estimated": "design_estimate",
        "manufacturer_specification": "manufacturer_datasheet",
        "derived": "model_convention",
        "measured": "bench_measurement",
        "unknown": "none",
    }.get(provenance, "none")


def build_bench_inventory(
    config: CSLMRouteConfig | None = None,
    *,
    inventory_overlay: Mapping[str, Any] | None = None,
) -> tuple[BenchInventoryItem, ...]:
    """Build the bench inventory from the editable-control registry + evidence overlay."""
    registry = build_cslm_editable_control_registry(config)
    bench_groups = {
        "source_and_input_beam", "slm1", "slm2",
        "inter_slm_and_4f_geometry", "physical_axicon_benchmark", "camera_reference_plane",
    }
    # Cross-group controls that are nevertheless genuine bench/route inputs.
    extra_ids = {"wavelength_nm", "grid_N", "dx_um", "n_z"}
    items: list[BenchInventoryItem] = []
    for c in registry:
        if c.group not in bench_groups and c.control_id not in extra_ids:
            continue
        ev = _evidence_for(c, inventory_overlay)
        items.append(BenchInventoryItem(
            component_id=c.control_id,
            display_name=c.display_name,
            component_type=_GROUP_COMPONENT_TYPE.get(c.group, "unknown"),
            route_location=c.physical_location,
            value=ev["value"],
            unit=c.unit,
            provenance=ev["provenance"],
            source_type=ev["source_type"],
            source_reference=ev["source_reference"],
            recorded_date=ev["recorded_date"],
            uncertainty=ev["uncertainty"],
            coordinate_frame=_GROUP_FRAME.get(c.group, "lab_beam_frame"),
            status=c.status,
            required_for_initial_4F_model=c.control_id in PHYSICAL_4F_HARD_BLOCKERS,
            required_for_measured_bench_prediction=(
                c.required_for_measured_bench_mode
                or c.control_id in MEASURED_BENCH_EXTRA
            ),
            notes=ev["notes"],
        ))
    return tuple(items)


def bench_inventory_rows(inventory: Sequence[BenchInventoryItem] | None = None,
                         *, config: CSLMRouteConfig | None = None) -> list[dict[str, Any]]:
    inventory = inventory if inventory is not None else build_bench_inventory(config)
    return [it.as_row() for it in inventory]


def build_bench_inventory_profile(config: CSLMRouteConfig | None = None,
                                  *, inventory_overlay: Mapping[str, Any] | None = None) -> dict[str, Any]:
    inv = build_bench_inventory(config, inventory_overlay=inventory_overlay)
    return {
        "profile_name": INVENTORY_PROFILE_NAME,
        "profile_status": INVENTORY_PROFILE_STATUS,
        "claim_boundary": "diagnostic demo inventory; placeholder values from the repository demo "
                          "config are NOT measured; every unknown remains null; n=1.0 free-space; "
                          "no 4F/camera/material physics; final_export_allowed=False",
        "items": {it.component_id: it.as_json() for it in inv},
    }


def save_bench_inventory(profile: Mapping[str, Any], path: str | Path) -> Path:
    out = Path(path); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2, sort_keys=False), encoding="utf-8")
    return out


def load_bench_inventory(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# physical-4F readiness gate (levels A-D)
# ---------------------------------------------------------------------------


def _bucket(item: BenchInventoryItem) -> str:
    if item.value is None or item.provenance == "unknown":
        return "unknown"
    if item.provenance == "measured":
        return "measured"
    return "placeholder"


def _level(items_by_id: Mapping[str, BenchInventoryItem], required_ids: Sequence[str],
           *, require_measured: bool, extra_blockers: Sequence[str] = ()) -> dict[str, Any]:
    measured, placeholder, unknown, blocked = [], [], [], list(extra_blockers)
    for cid in required_ids:
        it = items_by_id.get(cid)
        if it is None:
            unknown.append(cid); blocked.append(f"{cid}: not in inventory"); continue
        b = _bucket(it)
        if b == "measured":
            measured.append(cid)
        elif b == "unknown":
            unknown.append(cid); blocked.append(f"{cid}: value/provenance unknown")
        else:
            placeholder.append(cid)
            if require_measured:
                blocked.append(f"{cid}: not measured (provenance={it.provenance})")
    ready = not blocked
    next_required = [b.split(":")[0] for b in blocked]
    return {
        "ready": ready,
        "blocked_by": blocked,
        "warning_items": [],
        "measured_items": measured,
        "placeholder_items": placeholder,
        "unknown_items": unknown,
        "next_required_measurements": sorted(set(next_required)),
    }


def evaluate_physical_4f_readiness(
    config: CSLMRouteConfig | None = None,
    *,
    inventory_overlay: Mapping[str, Any] | None = None,
    frames=None,
    transforms=None,
) -> dict[str, Any]:
    """Four-level readiness: A active, B benchmark, C initial scalar 4F, D measured bench."""
    frames = frames if frames is not None else build_coordinate_frames()
    transforms = transforms if transforms is not None else build_coordinate_transforms()
    inv = build_bench_inventory(config, inventory_overlay=inventory_overlay)
    by_id = {it.component_id: it for it in inv}

    # A. active CSLM diagnostic readiness (runs with placeholders).
    level_a = {
        "ready": True, "blocked_by": [], "warning_items": [],
        "measured_items": [], "placeholder_items": [], "unknown_items": [],
        "next_required_measurements": [],
        "note": "active route executes with diagnostic placeholders (n=1.0 free-space).",
    }

    # B. ideal physical-axicon benchmark readiness (executable; report provenance).
    level_b = _level(by_id, BENCHMARK_REQUIRED, require_measured=False)
    level_b["note"] = "benchmark branch is executable with placeholders; provenance reported."

    # C. initial scalar 4F-model readiness (values + coordinate convention).
    coord_blockers: list[str] = []
    if not fourier_position_mapping_known(frames):
        coord_blockers.append("Fourier_plane_physical_position_frame: coordinate convention unknown")
    # SLM2 transverse coordinate scale must be defined (pixel pitch / continuous equivalent).
    slm2_scale = by_id.get("slm2_pixel_pitch_um")
    if slm2_scale is None or slm2_scale.value is None:
        coord_blockers.append("slm2_pixel_pitch_um: SLM2 transverse coordinate scale unknown")
    level_c = _level(by_id, PHYSICAL_4F_HARD_BLOCKERS, require_measured=False,
                     extra_blockers=coord_blockers)
    level_c["note"] = "blocked until every 4F value is defined AND the Fourier-plane coordinate " \
                      "convention + SLM2 transverse scale are explicit."

    # D. measured-bench / camera-comparison readiness (measured + transforms declared).
    d_required = tuple(PHYSICAL_4F_HARD_BLOCKERS) + tuple(MEASURED_BENCH_EXTRA)
    d_coord_blockers: list[str] = []
    if not slm2_to_lab_mapping_known(frames, transforms):
        d_coord_blockers.append("SLM2_pixel_to_lab: coordinate transform not declared/calibrated")
    if not camera_to_lab_mapping_known(frames, transforms):
        d_coord_blockers.append("camera_object_plane_to_lab: coordinate transform not declared/calibrated")
    if not fourier_position_mapping_known(frames):
        d_coord_blockers.append("Fourier_plane_physical_position_frame: coordinate convention unknown")
    level_d = _level(by_id, d_required, require_measured=True, extra_blockers=d_coord_blockers)
    level_d["note"] = "requires measured values + declared SLM2->lab / Fourier->lab / camera->lab " \
                      "transforms + a reference plane + a beam-profile calibration capture (or " \
                      "declared absent)."

    return {
        "A_active_cslm_diagnostic": level_a,
        "B_ideal_axicon_benchmark": level_b,
        "C_initial_scalar_4f_model": level_c,
        "D_measured_bench_camera": level_d,
        DOWNSTREAM_EMPIRICAL_EVIDENCE_STATE: downstream_empirical_carrier_stop_evidence_effect(False),
        "fourier_filter_physics_available": False,
        "diagnostic_only": True,
        "final_export_allowed": False,
        "claim_boundary": "n=1.0 free-space optical/fluence diagnostic; no physical 4F field is "
                          "generated; changing 4F inventory values updates readiness only.",
        "physical_4f_hard_blockers": list(PHYSICAL_4F_HARD_BLOCKERS),
    }


def downstream_empirical_carrier_stop_evidence_effect(available: bool = False) -> dict[str, Any]:
    """Declare what downstream carrier/stop evidence can and cannot unblock."""
    return {
        "evidence_state": DOWNSTREAM_EMPIRICAL_EVIDENCE_STATE,
        "available": bool(available),
        "supports": [
            "practical operating-point selection",
            "repeatability assessment",
            "later comparison against a physical 4F model",
        ],
        "cannot_support": [
            "physical_fourier_plane_coordinate_calibrated",
            "physical_4f_readiness_ready",
            "direct Fourier-plane order positions",
            "direct stop radius in Fourier-plane mm",
            "direct order-power fractions at the stop",
        ],
        "physical_4f_readiness_effect": "does_not_mark_ready",
        "claim_boundary": (
            "downstream final-focus images are empirical response evidence only; direct "
            "Fourier-plane mapping still requires temporary access at or conjugate to the Fourier plane"
        ),
    }


def readiness_summary_rows(readiness: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, label in [
        ("A_active_cslm_diagnostic", "A active CSLM diagnostic"),
        ("B_ideal_axicon_benchmark", "B ideal axicon benchmark"),
        ("C_initial_scalar_4f_model", "C initial scalar 4F model"),
        ("D_measured_bench_camera", "D measured bench / camera"),
    ]:
        lvl = readiness[key]
        rows.append({
            "level": label,
            "ready": lvl["ready"],
            "blocked_by_count": len(lvl["blocked_by"]),
            "measured": len(lvl["measured_items"]),
            "placeholder": len(lvl["placeholder_items"]),
            "unknown": len(lvl["unknown_items"]),
        })
    return rows


def measurement_checklist(readiness: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Prioritised list of measurements required to unblock physical 4F, then measured bench."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for priority, key in [(1, "C_initial_scalar_4f_model"), (2, "D_measured_bench_camera")]:
        for cid in readiness[key]["next_required_measurements"]:
            if cid not in seen:
                seen.add(cid)
                out.append({"priority": priority, "item": cid, "unblocks": key})
    return out


# ---------------------------------------------------------------------------
# readiness figure
# ---------------------------------------------------------------------------


def plot_physical_4f_readiness_gate(
    config: CSLMRouteConfig | None = None,
    *,
    inventory_overlay: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
    dpi: int = 160,
):
    """Diagnostic-only physical-4F readiness gate figure."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    readiness = evaluate_physical_4f_readiness(config, inventory_overlay=inventory_overlay)
    inv = build_bench_inventory(config, inventory_overlay=inventory_overlay)
    buckets = {"measured": 0, "placeholder": 0, "unknown": 0}
    for it in inv:
        buckets[_bucket(it)] += 1

    fig = plt.figure(figsize=(15.0, 8.6), facecolor="white")
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.97, top=0.85, bottom=0.07, hspace=0.42, wspace=0.22)
    fig.suptitle("Stage 8C.3R.5.3 Physical-4F Readiness Gate\n"
                 "n=1.0 free-space; NO physical 4F field generated; no material/camera physics; "
                 "final_export_allowed=False",
                 x=0.04, y=0.975, ha="left", va="top", fontsize=14, fontweight="bold")

    ax = fig.add_subplot(gs[0, 0]); ax.set_axis_off()
    ax.text(0.0, 1.0, "Readiness levels", fontsize=12, fontweight="bold", va="top")
    colour = {True: "#1b5e20", False: "#b71c1c"}
    rows = readiness_summary_rows(readiness)
    for i, r in enumerate(rows):
        ax.text(0.02, 0.82 - i * 0.20, r["level"], fontsize=10.5, va="top")
        ax.text(0.72, 0.82 - i * 0.20, "READY" if r["ready"] else "BLOCKED",
                fontsize=10.5, fontweight="bold", color=colour[r["ready"]], va="top")
        ax.text(0.02, 0.73 - i * 0.20,
                f"   measured {r['measured']} / placeholder {r['placeholder']} / unknown {r['unknown']}"
                f"  (blockers {r['blocked_by_count']})", fontsize=8.2, va="top", color="#555")

    ax = fig.add_subplot(gs[0, 1])
    ax.bar(list(buckets), [buckets[k] for k in buckets], color=["#1b5e20", "#ef6c00", "#b71c1c"])
    for i, k in enumerate(buckets):
        ax.text(i, buckets[k], str(buckets[k]), ha="center", va="bottom", fontsize=9)
    ax.set_title("Bench inventory items by evidence bucket", fontsize=11, fontweight="bold")

    ax = fig.add_subplot(gs[1, 0]); ax.set_axis_off()
    cblock = readiness["C_initial_scalar_4f_model"]["blocked_by"]
    lines = [f"Physical 4F hard blockers ({len(cblock)}):"]
    for b in cblock[:14]:
        lines.append("  - " + b[:66])
    ax.text(0.0, 1.0, "\n".join(lines), fontsize=7.8, va="top", family="monospace")

    ax = fig.add_subplot(gs[1, 1]); ax.set_axis_off()
    chk = measurement_checklist(readiness)
    lines = ["Prioritised measurement checklist:"]
    for c in chk[:14]:
        lines.append(f"  P{c['priority']} {c['item']}")
    lines += ["", "CLAIM: " + readiness["claim_boundary"][:64],
              "       " + readiness["claim_boundary"][64:128]]
    ax.text(0.0, 1.0, "\n".join(lines), fontsize=8.0, va="top", family="monospace")

    if output_path is not None:
        out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight", metadata={
            "Title": "Stage 8C.3R.5.3 physical-4F readiness gate", "final_export_allowed": "False"})
    return fig
