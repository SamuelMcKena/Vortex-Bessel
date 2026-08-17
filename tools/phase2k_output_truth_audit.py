"""Inventory every repository output and apply the Phase 2K truth quarantine.

The purpose of this script is deliberately conservative: no pre-existing
figure, CSV, JSON result, rendered notebook, or presentation asset is promoted
as scientific evidence merely because it exists or because an older regression
test reproduced it.  Every file under ``outputs/`` is inventoried and placed in
an explicit provenance state.  New Phase 2K results are written to a separate
validation directory after analytic/reference tests have passed.

Run from the repository root::

    python tools/phase2k_output_truth_audit.py

The generated files are suitable for CI artifacts and later selective commit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest
from vbb_study.digital_twin.vortex_error_reference_models import snell_axicon_geometry
from vbb_study.equations.scalar_bessel import (
    bessel_gauss_ring_radius_m,
    ring_radius_from_jprime_zero_m,
)


GENERATED_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".pdf", ".svg", ".csv", ".json", ".jsonl",
    ".txt", ".npy", ".npz", ".html", ".gif", ".mp4",
}

CRITICAL_HARDWARE_PARAMETERS = {
    "beam_radius_on_slm_m",
    "slm_model",
    "slm_phase_stroke_rad",
    "slm_phase_lut",
    "fourf_focal_length_m",
    "fourier_iris_radius_m",
    "objective_NA",
    "objective_focal_length_m",
    "relay_effective_focal_length_m",
    "relay_magnification_to_sample",
    "axicon_base_angle_deg",
    "axicon_clear_aperture_radius_m",
    "camera_pixel_scale_m",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _family(path: Path) -> str:
    parts = path.as_posix().split("/")
    if len(parts) >= 3 and parts[0] == "outputs":
        return parts[1] if parts[1] != "figures" else "/".join(parts[:3])
    return parts[0]


def _quarantine_status(path: Path) -> tuple[str, str]:
    p = path.as_posix().lower()
    suffix = path.suffix.lower()
    if "/presentation_phase2i/" in p or "/presentation_phase2j/" in p:
        return (
            "QUARANTINED_PRESENTATION_DERIVATIVE",
            "presentation assets inherit all upstream model assumptions and are not primary evidence",
        )
    if "/publication_study/" in p:
        return (
            "QUARANTINED_LEGACY_PRE_PHASE2K",
            "legacy publication-study output predates the Phase 2K analytic reference corrections",
        )
    if any(token in p for token in ("/phase2b", "/phase2c", "/phase2d", "/phase2e")):
        return (
            "QUARANTINED_REGENERATE_AFTER_CORE_MATH_CHANGE",
            "candidate modern evidence but generated before the corrected finite Bessel-Gauss reference",
        )
    if suffix in GENERATED_SUFFIXES:
        return (
            "QUARANTINED_UNTRACED_OUTPUT",
            "output has not yet passed a producer/dependency/reference trace in Phase 2K",
        )
    return (
        "QUARANTINED_UNCLASSIFIED",
        "nonstandard output type; preserve but do not use quantitatively until traced",
    )


def build_inventory(repo_root: Path) -> list[dict[str, Any]]:
    outputs = repo_root / "outputs"
    if not outputs.exists():
        raise FileNotFoundError(outputs)
    rows: list[dict[str, Any]] = []
    for path in sorted(p for p in outputs.rglob("*") if p.is_file()):
        rel = path.relative_to(repo_root)
        status, reason = _quarantine_status(rel)
        rows.append(
            {
                "path": rel.as_posix(),
                "suffix": path.suffix.lower(),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
                "family": _family(rel),
                "phase2k_status": status,
                "reason": reason,
                "scientific_use_allowed": False,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["path"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def hardware_truth_gate() -> dict[str, Any]:
    manifest = canonical_hardware_manifest()
    params = {row["parameter"]: row for row in manifest["parameters"]}
    unresolved = []
    for name in sorted(CRITICAL_HARDWARE_PARAMETERS):
        row = params.get(name)
        if row is None or row.get("value") is None or row.get("status") == "calibration_required":
            unresolved.append(name)
    return {
        "gate": "PHASE2K_HARDWARE_TRUTH",
        "fixed_bench_quantitative_claim_ready": len(unresolved) == 0,
        "unresolved_critical_parameters": unresolved,
        "policy": (
            "A numerical route may be mathematically correct while remaining physically uncalibrated. "
            "Absolute bench/sample-plane predictions are blocked until every critical geometry/scale "
            "parameter has measured or independently verified provenance."
        ),
        "historical_manifest_fixed_bench_prediction_ready": bool(
            manifest.get("fixed_bench_prediction_ready", False)
        ),
    }


def analytic_reference_snapshot() -> dict[str, Any]:
    # This is not a bench prediction.  It quantifies two mathematical effects
    # that directly invalidate old shorthand assumptions.
    ell = 3
    kr = 1.0e6
    waist = 4.0e-6
    pure_ring = ring_radius_from_jprime_zero_m(ell, kr)
    finite_ring = bessel_gauss_ring_radius_m(ell, kr, waist)

    axicon_rows = []
    for angle_deg in (1.0, 2.0, 5.0, 10.0, 20.0):
        try:
            geometry = snell_axicon_geometry(
                base_angle_rad=np.deg2rad(angle_deg),
                refractive_index=1.458,
                external_index=1.0,
            )
            axicon_rows.append(
                {
                    "base_angle_deg": angle_deg,
                    "exact_deflection_deg": float(np.rad2deg(geometry.deflection_rad)),
                    "thin_phase_relative_kr_error": float(geometry.shallow_relative_error),
                }
            )
        except ValueError as exc:
            axicon_rows.append({"base_angle_deg": angle_deg, "error": str(exc)})

    return {
        "finite_bg_ring_demo": {
            "ell": ell,
            "kr_m_inv": kr,
            "waist_m": waist,
            "pure_bessel_jprime_ring_m": pure_ring,
            "finite_bg_peak_ring_m": finite_ring,
            "relative_shift": (finite_ring - pure_ring) / pure_ring,
            "meaning": "J'_ell gives the infinite-Bessel peak, not generally the finite BG peak",
        },
        "refractive_axicon_reference": axicon_rows,
        "angle_convention": (
            "base_angle_deg is the tilt of the conical-surface normal from the optical axis; "
            "manufacturer apex-angle values must be converted explicitly"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/validation/phase2k_truth_audit"),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    out = args.output_dir
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)

    rows = build_inventory(root)
    write_csv(out / "complete_output_inventory.csv", rows)

    family_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for row in rows:
        family_counts[row["family"]] = family_counts.get(row["family"], 0) + 1
        status_counts[row["phase2k_status"]] = status_counts.get(row["phase2k_status"], 0) + 1

    summary = {
        "phase": "PHASE2K",
        "inventory_file_count": len(rows),
        "all_preexisting_outputs_quarantined": all(not row["scientific_use_allowed"] for row in rows),
        "family_counts": dict(sorted(family_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "hardware_truth_gate": hardware_truth_gate(),
        "analytic_reference_snapshot": analytic_reference_snapshot(),
        "next_gate": (
            "trace each output family to its generating code, validate that producer against analytic or "
            "independent numerical references, regenerate, then compare regenerated and historical artifacts"
        ),
    }
    (out / "phase2k_truth_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
