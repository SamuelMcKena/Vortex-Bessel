"""Build the committed canonical figure library from selected historical sources.

This script expects source checkouts beneath ``_figure_sources``:

- ``q20``: current axicon-aberration branch
- ``phase2j``: refined presentation branch (optional)
- ``phase2k``: mathematical/physics-audited branch (optional)

The goal is curation, not archival copying. Older/pre-realignment/superseded q20
renders are deliberately excluded. The historical repository remains the
provenance archive.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "_figure_sources"
OUT = ROOT / "figures"

FIELDS = [
    "id", "path", "topic", "status", "evidence_type", "generator",
    "source_repo", "source_ref", "source_path", "supersedes",
    "presentation_preferred", "notes",
]


def source_branch(root: Path) -> str:
    marker = root / ".CURATED_SOURCE_BRANCH"
    return marker.read_text(encoding="utf-8").strip() if marker.exists() else "unknown"


def find_one(root: Path, basename: str) -> Path | None:
    if not root.exists():
        return None
    matches = [p for p in root.rglob(basename) if ".git" not in p.parts]
    if not matches:
        return None

    def score(path: Path) -> tuple[int, int, str]:
        text = path.as_posix().lower()
        penalty = sum(word in text for word in ("backup", "archive", "old", "debug", "quicklook"))
        reward = sum(word in text for word in ("canonical", "presentation", "phase2j"))
        return (penalty - reward, len(path.parts), text)

    return sorted(matches, key=score)[0]


def copy_one(rows: list[dict[str, str]], source: Path | None, destination: Path, *,
             topic: str, status: str, evidence_type: str, generator: str,
             source_root: Path, supersedes: str = "", presentation_preferred: bool = False,
             notes: str = "") -> None:
    if source is None or not source.is_file():
        print(f"SKIP missing: {source}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    rows.append({
        "id": destination.stem,
        "path": destination.relative_to(ROOT).as_posix(),
        "topic": topic,
        "status": status,
        "evidence_type": evidence_type,
        "generator": generator,
        "source_repo": "SamuelMcKena/phd-structured-beam-study",
        "source_ref": source_branch(source_root),
        "source_path": source.relative_to(source_root).as_posix(),
        "supersedes": supersedes,
        "presentation_preferred": "yes" if presentation_preferred else "no",
        "notes": notes,
    })


def curate_q20(rows: list[dict[str, str]]) -> None:
    qroot = SOURCE_ROOT / "q20"
    qfig = qroot / "notebooks/experimental/axicon_aberration_correction/figures/slm_closed_loop_alignment/modal_q20"
    qdst = OUT / "experimental/q20_aberration"

    groups: dict[str, list[tuple[str, str, bool]]] = {
        "reconstruction": [
            ("annular_aberration_phase.png", "q20_modal_analysis.py", True),
            ("modal_spectrum.png", "q20_modal_analysis.py", False),
            ("polar_measured_fit_corrected.png", "q20_modal_analysis.py", True),
            ("realigned_cartesian_xy_measured_fit_corrected_ideal.png", "q20_modal_analysis.py", True),
            ("realigned_cartesian_xy_page_1.png", "q20_modal_analysis.py", False),
            ("realigned_3d_intensity_isosurfaces.png", "q20_modal_analysis.py", True),
        ],
        "validation": [
            ("realigned_signed_xz_yz_measured_corrected_ideal.png", "q20_modal_analysis.py", True),
            ("realigned_cartesian_similarity_vs_z.png", "q20_modal_analysis.py", False),
            ("realigned_profile_metrics_vs_z.png", "q20_modal_analysis.py", False),
            ("comprehensive_error_validation/all_z_full_signed_xz_yz_maps.png", "comprehensive_error_validation.py", True),
            ("comprehensive_error_validation/all_z_metrics_dashboard.png", "comprehensive_error_validation.py", True),
            ("comprehensive_error_validation/measured_vs_corrected_3d_mesh.png", "comprehensive_error_validation.py", True),
            ("comprehensive_error_validation/measured_beam_axis_trajectory.png", "comprehensive_error_validation.py", False),
            ("comprehensive_error_validation/all_z_images_page_1.png", "comprehensive_error_validation.py", False),
        ],
        "phase_error_recreation": [
            ("phase_error_recreation/phase_error_recreation_agreement_vs_z.png", "phase_error_recreation.py", True),
            ("phase_error_recreation/phase_error_recreation_signed_xz_yz.png", "phase_error_recreation.py", True),
            ("phase_error_recreation/phase_error_recreation_page_1.png", "phase_error_recreation.py", False),
        ],
        "single_mask": [
            ("single_mask_inverse_forward_test/single_mask_metrics_vs_z.png", "single_mask_inverse_forward_test.py", True),
            ("single_mask_inverse_forward_test/single_mask_signed_xz_yz.png", "single_mask_inverse_forward_test.py", True),
            ("single_mask_inverse_forward_test/single_z_minus10_measured_ideal_inverse.png", "single_mask_inverse_forward_test.py", True),
            ("single_z_double_confirmation_minus10.png", "single_z_double_confirmation.py", True),
        ],
        "closed_loop": [
            ("iterative_closed_loop/iteration_000_gain_sweep.png", "iterative_correction_controller.py", True),
            ("iterative_closed_loop/REJECTED_DO_NOT_USE_iteration_000_slm2_preview/NOMINAL_PREVIEW_NOT_FOR_DISPLAY_ITERATION_000_CANDIDATE_overview.png", "iterative_correction_controller.py", False),
        ],
        "slm_preview": [
            ("UNCALIBRATED_DO_NOT_APPLY_q20_modal_correction.png", "q20_modal_analysis.py", False),
            ("slm2_correction_only_preview/NOMINAL_PREVIEW_NOT_FOR_DISPLAY_SLM2_CORRECTION_ONLY_GAIN_0p20_overview.png", "slm2_complete_mask_preview.py", False),
            ("slm2_preview/NOMINAL_PREVIEW_NOT_FOR_DISPLAY_SLM2_overview.png", "slm2_complete_mask_preview.py", False),
        ],
    }

    for group, items in groups.items():
        for rel, generator, preferred in items:
            is_rejected = "REJECTED_DO_NOT_USE" in rel or "CANDIDATE" in rel
            evidence = "measured+model" if group in {"reconstruction", "validation"} else "model_inference"
            if group in {"phase_error_recreation", "single_mask"}:
                evidence = "falsification_test"
            if group == "slm_preview" or is_rejected:
                evidence = "hardware_preview_not_validated"
            status = "canonical"
            if group == "slm_preview":
                status = "secondary"
            if is_rejected:
                status = "rejected_engineering_evidence"
            note = ""
            if evidence == "hardware_preview_not_validated":
                note = "Not hardware-validated; requires calibrated SLM mapping/LUT and a fresh post-correction z-stack."
            if is_rejected:
                note = "Rejected candidate retained only as controller/engineering evidence; do not present as a usable correction mask."
            copy_one(
                rows, qfig / rel, qdst / group / Path(rel).name,
                topic=f"q20_aberration/{group}",
                status=status,
                evidence_type=evidence,
                generator=generator,
                source_root=qroot,
                supersedes="pre_realign_* and early/root q20 renders" if group in {"reconstruction", "validation"} else "",
                presentation_preferred=preferred,
                notes=note,
            )


def curate_presentation(rows: list[dict[str, str]]) -> None:
    root = SOURCE_ROOT / "phase2j"
    names = [
        "fig_01_ideal_beam_profiles.png",
        "fig_02_one_beam_propagation_annotated.png",
        "fig_03_three_beam_taxonomy_contact_sheet.png",
        "fig_04_dual_slm_grating_vs_all_digital.png",
        "fig_05_full_optical_path_split_before_final_section.png",
        "fig_06_lab_perspective_quad_points.png",
        "fig_07_full_system_ideal_vs_error_comparison.png",
        "fig_08_through_sample_design_space.png",
        "fig_09_continuous_write_synopsis.png",
        "fig_10_sensitivity_synopsis.png",
    ]
    for name in names:
        source = find_one(root, name)
        copy_one(
            rows, source, OUT / "presentation" / name,
            topic="presentation", status="canonical",
            evidence_type="model_or_governed_visualisation",
            generator="tools/build_phase2j_presentation_suite.py",
            source_root=root, presentation_preferred=True,
            notes="Presentation visual; quantitative claims remain governed by the underlying model/evidence outputs.",
        )


def curate_publication(rows: list[dict[str, str]]) -> None:
    root = SOURCE_ROOT / "phase2k"
    groups = {
        "ideal_beams": [
            "fig_01_nominal_intensity_cross_sections.png",
            "fig_02_case_intensity_xy.png",
            "fig_03_case_phase_wrapped.png",
            "fig_08_case_longitudinal_xz.png",
            "fig_09_case_radial_profiles.png",
        ],
        "lab_realism": [
            "fig_03_ideal_vs_lab_propagation_xz.png",
            "fig_04_ideal_vs_lab_intensity_xy.png",
        ],
        "system_errors/rounded_tip": ["fig_06_tip_rounding_sensitivity.png"],
        "materials": ["fig_07_bgo_multi_pass_transmission.png", "fig_05_rate_proxies_vs_axicon_angle.png"],
    }
    for group, names in groups.items():
        for name in names:
            copy_one(
                rows, find_one(root, name), OUT / "publication" / group / name,
                topic=f"publication/{group}", status="secondary", evidence_type="model",
                generator="current Publication_Study / Phase 2K workflow", source_root=root,
                notes="Retained when useful; regenerated current or presentation figures take precedence where overlapping.",
            )


def write_docs(rows: list[dict[str, str]]) -> None:
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["path"]))

    (OUT / "README.md").write_text(
        """# Canonical figure library

This directory contains the **current preferred visual outputs** for Vortex-Bessel. It is curated rather than being a dump of every historical render.

## Rule

If multiple historical versions exist, prefer the figure listed here and use `manifest.csv` to trace its generator and source. The old repository remains the provenance archive.

## Layout

- `publication/` — selected reusable outputs from maintained Publication_Study workflows.
- `presentation/` — high-resolution presentation-ready views where available.
- `experimental/q20_aberration/reconstruction/` — current realigned modal reconstruction.
- `experimental/q20_aberration/validation/` — all-z / Cartesian / 3-D diagnostics.
- `experimental/q20_aberration/phase_error_recreation/` — phase-error recreation/falsification.
- `experimental/q20_aberration/single_mask/` — stricter single-input-mask forward tests.
- `experimental/q20_aberration/closed_loop/` — iterative controller diagnostics.
- `experimental/q20_aberration/slm_preview/` — uncalibrated SLM previews; not hardware-validated correction.

## q=20 claim boundary

Measured z-stack images are experimental evidence. Modal phase recovery and proposed correction maps are model inference. Phase-error recreation and single-mask forward propagation are falsification tests. SLM previews are not hardware-ready until phase LUT/stroke, illuminated footprint, parity/rotation and camera-to-SLM mapping are calibrated and a fresh post-correction z-stack verifies the result.

Rejected controller candidates are retained only as engineering evidence and are never canonical usable correction masks. Pre-realignment q20 renders and early `postcorrectionoutput1`-style figures are deliberately not promoted here.

`outputs/` remains the regeneration/transient area; `figures/` is the small committed library for navigation, comparison and presentation reuse.
""",
        encoding="utf-8",
    )


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rows: list[dict[str, str]] = []
    curate_q20(rows)
    curate_presentation(rows)
    curate_publication(rows)
    write_docs(rows)

    images = list(OUT.rglob("*.png")) + list(OUT.rglob("*.jpg")) + list(OUT.rglob("*.jpeg"))
    forbidden = [p for p in images if "pre_realign" in p.name or "postcorrectionoutput1" in p.name]
    if forbidden:
        raise RuntimeError(f"Legacy q20 figures leaked into canonical library: {forbidden}")
    if len(images) < 15:
        raise RuntimeError(f"Only {len(images)} curated images were found; source checkout likely failed")
    print(f"Curated {len(images)} images with {len(rows)} manifest entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
