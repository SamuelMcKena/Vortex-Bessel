"""Guard the clean repository against losing maintained Publication_Study coverage.

The historical workspace remains the provenance archive, but these module names
and current notebook replacements represent scientific workflows that the clean
repository promises to keep reachable.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_VBB_MODULES = (
    "__init__.py",
    "setup_study.py",
    "vbb_axicon.py",
    "vbb_capsule.py",
    "vbb_discrete.py",
    "vbb_hex_outline.py",
    "vbb_hexagon_metrics.py",
    "vbb_hexagon_study.py",
    "vbb_materials.py",
    "vbb_materials_study.py",
    "vbb_metrics.py",
    "vbb_planes.py",
    "vbb_polarized_train.py",
    "vbb_polygonal.py",
    "vbb_regime.py",
    "vbb_sample_study.py",
    "vbb_studies.py",
    "vbb_style.py",
    "vbb_train_viz.py",
    "vbb_validation.py",
    "vbb_vector.py",
    "vbb_viz.py",
)

CURRENT_NOTEBOOK_REPLACEMENTS = (
    "notebooks/00_study_overview_and_conventions.ipynb",
    "notebooks/scalar/02_scalar_ideal_vs_lab_diagnostics.ipynb",
    "notebooks/scalar/03_scalar_robustness_and_self_healing.ipynb",
    "notebooks/scalar/04_scalar_parameter_sweeps.ipynb",
    "notebooks/scalar/05_scalar_validation_suite.ipynb",
    "notebooks/lab_realism/01_holographic_axicon_route.ipynb",
    "notebooks/lab_realism/02_physical_axicon_route.ipynb",
    "notebooks/lab_realism/03_holographic_vs_physical_axicon.ipynb",
    "notebooks/lab_realism/04_objective_pupil_and_first_order_filtering.ipynb",
    "notebooks/lab_realism/05_through_sample_interface.ipynb",
    "notebooks/lab_realism/06_full_source_to_sample_journey.ipynb",
    "notebooks/vector/01_vector_beam_theory_atlas.ipynb",
    "notebooks/vector/02_vector_ideal_vs_lab_case1.ipynb",
    "notebooks/vector/03_vector_hardware_routes.ipynb",
    "notebooks/vector/04_vector_arm_hexagon.ipynb",
    "notebooks/materials/01_material_proxy_fluence_and_thresholds.ipynb",
    "notebooks/materials/02_material_calibration_template.ipynb",
    "notebooks/materials/03_application_design_tables.ipynb",
    "notebooks/advanced/01_capsule_weld_feature_design.ipynb",
    "notebooks/advanced/02_hexagonal_polygonal_beams.ipynb",
    "notebooks/advanced/03_discrete_nfold_beams.ipynb",
)

COMPATIBILITY_ROOT_FILES = (
    "bessel_twin_core.py",
    "publication_diagnostics.py",
    "interface_correction_diagnosis.py",
    "finalize_publication_outputs.py",
    "run_publication_study.py",
    "run_study.py",
)

LATER_CURRENT_WORK = (
    "vbb_study/digital_twin",
    "notebooks/digital_twin",
    "notebooks/experimental/axicon_aberration_correction/Bessel_zscan_alignment_correction.ipynb",
    "tools/run_vortex_system_error_suite.py",
    "tools/audit_vortex_system_error_signatures.py",
    "tools/check_axicon_tip_benchmark.py",
)


def _missing(paths: tuple[str, ...]) -> list[str]:
    return [path for path in paths if not (ROOT / path).exists()]


def main() -> int:
    missing_modules = [
        f"vbb_study/{name}"
        for name in ORIGINAL_VBB_MODULES
        if not (ROOT / "vbb_study" / name).is_file()
    ]
    missing_notebooks = _missing(CURRENT_NOTEBOOK_REPLACEMENTS)
    missing_root = _missing(COMPATIBILITY_ROOT_FILES)
    missing_later = _missing(LATER_CURRENT_WORK)

    groups = {
        "original vbb_study module coverage": missing_modules,
        "current notebook replacement coverage": missing_notebooks,
        "root compatibility/front-door coverage": missing_root,
        "later current work": missing_later,
    }

    failed = False
    for label, missing in groups.items():
        if missing:
            failed = True
            print(f"FAIL {label}")
            for path in missing:
                print(f"  missing: {path}")
        else:
            print(f"PASS {label}")

    print(
        f"checked {len(ORIGINAL_VBB_MODULES)} original module names, "
        f"{len(CURRENT_NOTEBOOK_REPLACEMENTS)} current notebook replacements, "
        f"{len(COMPATIBILITY_ROOT_FILES)} root entry points and "
        f"{len(LATER_CURRENT_WORK)} later-current anchors"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
