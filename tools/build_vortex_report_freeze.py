"""Build report-freeze metadata without executing or rewriting optical simulations."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
FREEZE_ROOT = ROOT / "outputs" / "validation" / "report_freeze"
CLAIM_SOURCE = ROOT / "docs" / "reporting" / "VORTEX_CLAIM_TO_EVIDENCE.csv"

CORE_EVIDENCE = (
    ".gitignore",
    "README.md",
    "AGENTS.md",
    "pyproject.toml",
    "requirements-report.txt",
    "docs/00_theory.md",
    "docs/01_conventions.md",
    "docs/04_model_limitations.md",
    "docs/88_phase1_critical_physics_repairs.md",
    "docs/88_phase1_fourier_geometry_impact_audit.md",
    "docs/89_phase1r_regeneration_recovery_and_reconciliation.md",
    "docs/90_phase2a_canonical_lab_realism.md",
    "docs/91_phase2b_visual_diagnostics_and_beam_volume_maps.md",
    "docs/92_phase2c_vectorial_objective_and_interface.md",
    "docs/93_phase2d_experimental_calibration_bridge.md",
    "docs/calibration/01_beam_radius_measurement.md",
    "docs/calibration/02_pulse_energy_and_transmission.md",
    "docs/calibration/03_slm_phase_lut_and_stroke.md",
    "docs/calibration/04_fourier_plane_and_iris.md",
    "docs/calibration/05_objective_relay_and_camera_scale.md",
    "docs/calibration/06_axicon_alignment_and_clear_aperture.md",
    "docs/calibration/07_material_index_and_interface_state.md",
    "docs/calibration/08_complete_calibration_acceptance.md",
    "docs/reporting/REPOSITORY_MAP.md",
    "docs/reporting/VORTEX_REPORT_EVIDENCE_INDEX.md",
    "docs/reporting/VORTEX_CLAIM_TO_EVIDENCE.csv",
    "docs/reporting/REPORT_SCOPE_AND_MATURITY.md",
    "docs/reporting/SOFTWARE_AND_REPRODUCIBILITY.md",
    "docs/reporting/VORTEX_FIGURE_AND_TABLE_PLAN.md",
    "tools/build_vortex_report_freeze.py",
    "tools/prepare_phase1r_lab_notebooks.py",
    "tools/reconcile_phase1r_artifacts.py",
    "tools/run_phase1r_realistic_window_recovery.py",
    "tools/run_phase2a_canonical.py",
    "tools/run_phase2b_visual_diagnostics.py",
    "tools/run_phase2c_objective_interface.py",
    "vbb_study/equations/fields.py",
    "vbb_study/equations/scalar_bessel.py",
    "vbb_study/equations/propagation.py",
    "vbb_study/equations/holography.py",
    "vbb_study/equations/metrics.py",
    "vbb_study/equations/vector_debye.py",
    "vbb_study/equations/vector_fresnel_interface.py",
    "vbb_study/vbb_axicon.py",
    "vbb_study/digital_twin/phase2a_contracts.py",
    "vbb_study/digital_twin/phase2a_canonical.py",
    "vbb_study/digital_twin/phase2b_visual_cases.py",
    "vbb_study/digital_twin/phase2b_figures.py",
    "vbb_study/digital_twin/phase2c_objective_interface.py",
    "vbb_study/digital_twin/phase2b_visual_diagnostics.py",
    "vbb_study/digital_twin/phase2c_figures.py",
    "vbb_study/digital_twin/phase2d_canonical_pipeline.py",
    "vbb_study/digital_twin/phase2d_figures.py",
    "vbb_study/calibration/__init__.py",
    "vbb_study/calibration/io.py",
    "vbb_study/calibration/schema.py",
    "vbb_study/calibration/uncertainty.py",
    "vbb_study/calibration/validation.py",
    "vbb_study/solver_policy.py",
    "tools/run_phase2d_calibration_bridge.py",
    "tools/validate_calibration_bundle.py",
    "tests/test_phase1_critical_physics_repairs.py",
    "tests/test_phase1r_reconciliation.py",
    "tests/test_phase2a_canonical_lab_realism.py",
    "tests/test_phase2b_visual_diagnostics.py",
    "tests/test_phase2c_objective_interface.py",
    "tests/test_phase2d_calibration_bridge.py",
    "tests/test_vortex_report_freeze.py",
    "tests/test_slm2_preserve_vortex.py",
    "tests/test_slm2_preserve_vortex_end_to_end.py",
    "outputs/validation/phase1_critical_repairs/phase1_repair_summary.json",
    "outputs/validation/phase1_critical_repairs/phase1_claim_status.csv",
    "outputs/validation/phase1_reconciliation/phase1r_regeneration_manifest.json",
    "outputs/validation/phase1_reconciliation/phase1r_final_claim_registry.csv",
    "outputs/validation/phase1_reconciliation/phase1r_selected_convergence_results.json",
    "outputs/validation/phase1_reconciliation/phase1r_convergence_runs.csv",
    "outputs/validation/phase2a/canonical_hardware_manifest.json",
    "outputs/validation/phase2a/canonical_case_summary.csv",
    "outputs/validation/phase2a/canonical_power_ledgers.csv",
    "outputs/validation/phase2a/slm_model_comparison.csv",
    "outputs/validation/phase2a/phase2a_claim_registry.csv",
    "outputs/validation/phase2a/phase2a_outcome_report.json",
    "outputs/figures/phase2b_visual_diagnostics/00_manifests/figure_provenance.csv",
    "outputs/figures/phase2b_visual_diagnostics/00_manifests/figure_provenance.json",
    "outputs/figures/phase2b_visual_diagnostics/00_manifests/phase2b_final_manifest.json",
    "outputs/figures/phase2b_visual_diagnostics/08_summary_tables/phase2b_case_summary.csv",
    "outputs/figures/phase2b_visual_diagnostics/08_summary_tables/phase2b_endpoint_reproduction_audit.csv",
    "outputs/figures/phase2b_visual_diagnostics/09_final_reports/phase2b_outcome_report.json",
    "outputs/validation/phase2c/phase2c_objective_benchmark.csv",
    "outputs/validation/phase2c/phase2c_case_summary.csv",
    "outputs/validation/phase2c/phase2c_interface_benchmark.csv",
    "outputs/validation/phase2c/phase2c_quadrature_convergence.csv",
    "outputs/validation/phase2c/phase2c_solver_validation.json",
    "outputs/validation/phase2c/phase2c_claim_registry.csv",
    "outputs/validation/phase2c/phase2c_outcome_report.json",
    "outputs/validation/phase2c/phase2c_figure_manifest.json",
    "outputs/validation/phase2d/calibration_readiness.json",
    "outputs/validation/phase2d/calibration_dependency_graph.csv",
    "outputs/validation/phase2d/canonical_case_summary.csv",
    "outputs/validation/phase2d/claim_maturity_registry.csv",
    "outputs/validation/phase2d/phase2d_claim_registry.csv",
    "outputs/validation/phase2d/phase2d_manifest.json",
    "outputs/validation/phase2d/phase2d_outcome_report.json",
    "outputs/validation/phase2d/solver_claim_policy.csv",
    "outputs/validation/phase2d/uncertainty_validation.csv",
    "calibration/templates/beam_radius_measurement_template.csv",
    "calibration/templates/camera_scale_template.csv",
    "calibration/templates/canonical_lab_calibration_template.json",
    "calibration/templates/energy_transmission_template.csv",
    "calibration/templates/material_interface_template.csv",
    "calibration/templates/objective_relay_template.csv",
    "calibration/templates/slm_phase_lut_template.csv",
)

TEST_COMMANDS = (
    "python -m pytest tests/test_phase1_critical_physics_repairs.py -q",
    "python -m pytest tests/test_phase1r_reconciliation.py -q",
    "python -m pytest tests/test_phase2a_canonical_lab_realism.py -q",
    "python -m pytest tests/test_phase2b_visual_diagnostics.py -q",
    "python -m pytest tests/test_phase2c_objective_interface.py -q",
    "python -m pytest tests/test_slm2_preserve_vortex.py tests/test_slm2_preserve_vortex_end_to_end.py -q",
    "python -m pytest tests/test_vortex_report_freeze.py -q",
    "python -m pytest --collect-only tests -q",
    "python -m compileall -q vbb_study tools tests",
    "git diff --check",
)

SECRET_PATTERNS = {
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    "openai_style_key": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "assigned_secret": re.compile(
        r"(?i)(api[_-]?key|password|auth[_-]?token|access[_-]?token|client[_-]?secret)"
        r"\s*[:=]\s*[\"'][^\"']{8,}[\"']"
    ),
}

TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".json", ".jsonl", ".csv", ".toml", ".yml", ".yaml",
    ".tex", ".bib", ".ipynb", ".ps1", ".sh", ".ini", ".cfg",
}


def _run(*args: str) -> str:
    completed = subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    return completed.stdout.strip()


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> Path:
    materialised = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialised)
    return path


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def _split_paths(value: str) -> list[str]:
    return [part.strip().replace("\\", "/") for part in str(value).split(";") if part.strip()]


def _classify(path: str, size: int) -> str:
    lower = path.lower()
    parts = lower.split("/")
    suffix = Path(path).suffix.lower()
    if any(part.startswith((".tmp", ".pytest")) for part in parts) or "__pycache__" in parts or suffix == ".pyc" or "vbb_study.egg-info" in lower:
        return "temporary/cache"
    if lower.startswith("vbb_study - copy/") or ("/" not in path and suffix == ".ipynb"):
        return "duplicate legacy notebook"
    if lower.startswith("patches/") or suffix in {".patch", ".diff"}:
        return "patch"
    if suffix == ".zip" or size > 90 * 1024 * 1024:
        return "large binary"
    if lower.startswith("archive/") or "superseded" in lower or "old_" in lower:
        return "historical or superseded diagnostics"
    if lower.startswith("calibration/templates/"):
        return "calibration templates"
    if lower.startswith("tests/"):
        return "tests"
    if lower.startswith("tools/") or ("/" not in path and lower.startswith("run_")):
        return "tools"
    if lower.startswith("vbb_study/") or suffix == ".py" or path in {"pyproject.toml", "requirements.txt", "requirements-report.txt"}:
        return "source code"
    if lower.startswith("outputs/validation/") or lower.startswith("baselines"):
        return "governed numerical outputs"
    if lower.startswith("report/evidence_pack/") or lower.startswith("docs/reporting/"):
        return "report evidence"
    if suffix in {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".html"} or lower.startswith("outputs/figures/"):
        return "figures"
    if lower.startswith("outputs/") and suffix in {".csv", ".json", ".jsonl", ".npy", ".npz"}:
        return "governed numerical outputs"
    if lower.startswith("docs/") or lower.startswith("report/") or suffix in {".md", ".tex", ".bib"}:
        return "reports and documentation"
    return "unknown"


def _github_action(category: str, path: str, size: int, selected: set[str]) -> str:
    if path in selected:
        return "selected_report_evidence"
    if category in {"temporary/cache", "duplicate legacy notebook", "large binary"}:
        return "exclude_from_standalone_git"
    if size >= 90 * 1024 * 1024:
        return "exclude_from_standalone_git"
    if category in {"historical or superseded diagnostics", "patch", "unknown"}:
        return "retain_or_review_not_report_evidence"
    return "retain_if_part_of_curated_export"


def _selected_evidence(claims: list[dict[str, str]]) -> set[str]:
    selected = set(CORE_EVIDENCE)
    for row in claims:
        for column in ("source_code_path", "data_path", "figure_path", "test_path"):
            selected.update(_split_paths(row.get(column, "")))
    for case in ("g0", "b0", "v1", "v3"):
        for folder, suffix in (
            ("02_xy_planes", "xy_landmarks_and_sas_focus.png"),
            ("03_xz_yz_slices", "xz_yz_and_power.png"),
            ("04_profiles", "native_profiles.png"),
        ):
            selected.add(f"outputs/figures/phase2b_visual_diagnostics/{folder}/{case}_{suffix}")
        for folder, suffix in (
            ("objective", "scalar_vs_debye.png"),
            ("interface", "scalar_vs_vector_fresnel.png"),
            ("profiles", "objective_profiles.png"),
            ("components", "debye_components.png"),
        ):
            selected.add(f"outputs/figures/phase2c/{folder}/{case}_{suffix}")
    for case in ("b0", "v1", "v3"):
        selected.add(f"outputs/figures/phase2b_visual_diagnostics/05_3d_maps/{case}_3d_intensity_surface.png")
    return selected


def _secret_scan(files: list[Path]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 20 * 1024 * 1024:
            continue
        if any(part.startswith((".tmp", ".pytest")) for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern_id, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append({"path": _relative(path), "pattern_id": pattern_id})
    return findings


def _resolve_reference(source: Path, reference: str, kind: str) -> tuple[str, bool, str]:
    ref = reference.strip().strip("<>").split("#", 1)[0].split("?", 1)[0]
    if not ref or re.match(r"^[a-zA-Z]+://", ref) or ref.startswith(("#", "mailto:")):
        return "", True, "external_or_anchor"
    ref = ref.replace("\\", "/")
    if "*" in ref or "{" in ref or "}" in ref or "<" in ref or ">" in ref:
        return ref, True, "pattern_not_resolved"
    if ref.startswith("Publication_Study/"):
        ref = ref[len("Publication_Study/"):]
    first = ref.split("/", 1)[0]
    if first in {"outputs", "docs", "report", "calibration", "vbb_study", "tests", "tools", "notebooks", "archive", "references", "reference_kernels"}:
        target = ROOT / ref
    else:
        target = source.parent / ref
    candidates = [target]
    if kind == "latex_graphic" and not target.suffix:
        candidates.extend(target.with_suffix(suffix) for suffix in (".pdf", ".png", ".jpg", ".jpeg"))
    if kind == "latex_bibliography" and not target.suffix:
        candidates.append(target.with_suffix(".bib"))
    existing = next((candidate for candidate in candidates if candidate.exists()), None)
    resolved = existing or target
    try:
        resolved_text = _relative(resolved.resolve())
    except ValueError:
        resolved_text = str(resolved.resolve())
    return resolved_text, existing is not None, "resolved" if existing is not None else "missing"


def _reference_audit(files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    md_link = re.compile(r"\]\(([^)]+)\)")
    inline_path = re.compile(
        r"`((?:Publication_Study/)?(?:outputs|docs|report|calibration|vbb_study|tests|tools)/[^`\n]+)`"
    )
    latex_patterns = (
        ("latex_graphic", re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")),
        ("latex_input", re.compile(r"\\(?:input|include)\{([^}]+)\}")),
        ("latex_bibliography", re.compile(r"\\bibliography\{([^}]+)\}")),
    )
    for source in files:
        if source.suffix.lower() not in {".md", ".tex"} or source.stat().st_size > 10 * 1024 * 1024:
            continue
        text = source.read_text(encoding="utf-8", errors="ignore")
        found: list[tuple[str, str]] = [("markdown_link", value) for value in md_link.findall(text)]
        found.extend(("inline_repository_path", value) for value in inline_path.findall(text))
        for kind, pattern in latex_patterns:
            found.extend((kind, value) for value in pattern.findall(text))
        seen: set[tuple[str, str]] = set()
        for kind, reference in found:
            key = (kind, reference)
            if key in seen:
                continue
            seen.add(key)
            resolved, exists, notes = _resolve_reference(source, reference, kind)
            rows.append({
                "source_path": _relative(source),
                "reference_kind": kind,
                "reference": reference,
                "resolved_path": resolved,
                "exists": exists,
                "scope": "report_ready" if _relative(source).startswith("docs/reporting/") or source.name in {"README.md", "AGENTS.md"} else "historical_or_other",
                "notes": notes,
            })
    return rows


def _package_versions() -> dict[str, str | None]:
    names = (
        "numpy", "scipy", "pandas", "matplotlib", "plotly", "nbformat", "pillow", "ipython",
        "pytest", "papermill", "ipykernel", "jupyter", "colorcet", "cmocean",
    )
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_freeze() -> dict[str, Any]:
    claims = _read_csv(CLAIM_SOURCE)
    selected = _selected_evidence(claims)
    missing_selected = sorted(path for path in selected if not (ROOT / path).is_file())
    if missing_selected:
        raise FileNotFoundError("selected report evidence is missing:\n" + "\n".join(missing_selected))

    tracked = set(_run("git", "ls-files", "--", ".").splitlines())
    untracked = set(_run("git", "ls-files", "--others", "--exclude-standard", "--", ".").splitlines())
    ignored = set(_run("git", "ls-files", "--others", "-i", "--exclude-standard", "--", ".").splitlines())

    evidence_rows: list[dict[str, Any]] = []
    for relative in sorted(selected):
        path = ROOT / relative
        size = path.stat().st_size
        evidence_rows.append({
            "path": relative,
            "category": _classify(relative, size),
            "role": "vortex_report_evidence",
            "sha256": _sha256(path),
            "size_bytes": size,
            "git_ignored": relative in ignored,
            "notes": "Referenced in claim matrix or core report-freeze contract.",
        })

    FREEZE_ROOT.mkdir(parents=True, exist_ok=True)
    files_path = _write_csv(
        FREEZE_ROOT / "vortex_report_files.csv",
        evidence_rows,
        ["path", "category", "role", "sha256", "size_bytes", "git_ignored", "notes"],
    )
    claims_path = _write_csv(
        FREEZE_ROOT / "vortex_report_claims.csv",
        claims,
        list(claims[0]),
    )

    disk_files = [path for path in ROOT.rglob("*") if path.is_file()]
    inventory_rows: list[dict[str, Any]] = []
    for path in disk_files:
        relative = _relative(path)
        size = path.stat().st_size
        category = _classify(relative, size)
        if relative in tracked:
            git_status = "tracked"
        elif relative in untracked:
            git_status = "untracked"
        elif relative in ignored:
            git_status = "ignored"
        else:
            git_status = "unclassified_worktree_file"
        inventory_rows.append({
            "path": relative,
            "category": category,
            "size_bytes": size,
            "git_status": git_status,
            "github_action": _github_action(category, relative, size, selected),
        })

    inventory_path = _write_csv(
        FREEZE_ROOT / "repository_inventory.csv",
        inventory_rows,
        ["path", "category", "size_bytes", "git_status", "github_action"],
    )
    audited_files = [
        path for path in disk_files
        if _relative(path) in tracked or _relative(path) in untracked
    ]
    reference_rows = _reference_audit(audited_files)
    reference_path = _write_csv(
        FREEZE_ROOT / "path_reference_audit.csv",
        reference_rows,
        ["source_path", "reference_kind", "reference", "resolved_path", "exists", "scope", "notes"],
    )
    secrets = _secret_scan(audited_files)
    large_files = [
        {"path": row["path"], "size_bytes": row["size_bytes"]}
        for row in inventory_rows if int(row["size_bytes"]) > 90 * 1024 * 1024
    ]
    broken = [row for row in reference_rows if not row["exists"]]
    report_broken = [row for row in broken if row["scope"] == "report_ready"]
    category_counts = dict(sorted(Counter(str(row["category"]) for row in inventory_rows).items()))
    status_counts = dict(sorted(Counter(str(row["git_status"]) for row in inventory_rows).items()))
    blockers_path = ROOT / "outputs" / "validation" / "phase2d" / "phase2d_outcome_report.json"
    blockers = []
    if blockers_path.is_file():
        blockers = json.loads(blockers_path.read_text(encoding="utf-8")).get("important_missing_measurements", [])

    manifest = {
        "schema_version": "1.0.0",
        "freeze_name": "vortex_report_first",
        "hash_algorithm": "SHA-256",
        "repository_path": str(ROOT),
        "parent_git_repository": str(ROOT.parent),
        "current_branch": _run("git", "branch", "--show-current"),
        "current_commit_before_report_preparation": _run("git", "rev-parse", "HEAD"),
        "working_tree_commit_created": False,
        "accepted_numerical_results_regenerated": False,
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "package_versions": _package_versions(),
        "test_commands": list(TEST_COMMANDS),
        "numerical_precision": {
            "phase2a_propagated_intensity_stack": "float32 storage",
            "phase2a_power_and_export_metrics": "Python/NumPy floating point",
            "phase2c_vector_working_arrays": "complex128/float64 unless source states otherwise",
            "metrics_source": "governed CSV/JSON/native arrays; never display pixels",
        },
        "canonical_hardware_manifest_path": "outputs/validation/phase2a/canonical_hardware_manifest.json",
        "solver_policy_provenance": [
            "vbb_study/solver_policy.py",
            "outputs/validation/phase2c/phase2c_claim_registry.csv",
        ],
        "unresolved_calibration_blockers": blockers,
        "selected_evidence_count": len(evidence_rows),
        "selected_evidence_bytes": sum(int(row["size_bytes"]) for row in evidence_rows),
        "selected_evidence": evidence_rows,
        "claims_path": _relative(claims_path),
        "claims_sha256": _sha256(claims_path),
        "files_index_path": _relative(files_path),
        "files_index_sha256": _sha256(files_path),
        "repository_inventory_path": _relative(inventory_path),
        "path_reference_audit_path": _relative(reference_path),
        "audit": {
            "disk_file_count": len(inventory_rows),
            "disk_bytes": sum(int(row["size_bytes"]) for row in inventory_rows),
            "category_counts": category_counts,
            "git_status_counts": status_counts,
            "files_over_90mb": large_files,
            "secret_signature_findings": secrets,
            "broken_reference_count": len(broken),
            "report_ready_broken_reference_count": len(report_broken),
            "report_ready_broken_references": report_broken,
        },
        "standalone_export": {
            "raw_directory_copy_ready": False,
            "curated_git_export_ready": not secrets and not missing_selected and not report_broken,
            "required_exclusions": [
                "temporary/cache trees",
                "root duplicate notebooks",
                "vbb_study - Copy",
                "bulk ZIP archives",
                "files above GitHub size limits",
            ],
            "ignored_selected_figures_require_force_add": any(row["git_ignored"] for row in evidence_rows),
        },
        "scope_boundary": (
            "Numerical and fixed-bench nominal vortex-Bessel evidence only; no experimental validation "
            "or nonlinear material-modification claim."
        ),
    }
    manifest_path = _write_json(FREEZE_ROOT / "vortex_report_manifest.json", manifest)
    return {
        "manifest": str(manifest_path),
        "selected_evidence_count": len(evidence_rows),
        "inventory_file_count": len(inventory_rows),
        "secret_findings": len(secrets),
        "large_files_over_90mb": len(large_files),
        "broken_references": len(broken),
        "report_ready_broken_references": len(report_broken),
        "curated_git_export_ready": manifest["standalone_export"]["curated_git_export_ready"],
    }


def main() -> int:
    result = build_freeze()
    print(json.dumps(result, indent=2))
    return 0 if result["secret_findings"] == 0 and result["report_ready_broken_references"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
