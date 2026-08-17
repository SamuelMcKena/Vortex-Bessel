# Publication_Study migration map

The original `Publication_Study` working directory contained multiple generations of notebooks, runner scripts, backup copies and generated outputs.  The new repository keeps the useful scientific workflows but organises them by topic so the current route is obvious.

This table is a navigation map, not a claim that every old cell is byte-for-byte identical to the current implementation.

| Original Publication_Study workflow | Current place to start | Current role |
|---|---|---|
| `00_publication_theory_conventions.ipynb` | `notebooks/00_study_overview_and_conventions.ipynb`, `docs/00_theory.md`, `docs/01_conventions.md` | definitions, conventions and model scope |
| `01_publication_scalar_case_diagnostics.ipynb` | `notebooks/scalar/02_scalar_ideal_vs_lab_diagnostics.ipynb` | Bessel/vortex-Bessel scalar cases and diagnostic views |
| `02_publication_robustness_metrics_visualisations.ipynb` | `notebooks/scalar/03_scalar_robustness_and_self_healing.ipynb` | robustness, obstruction/self-healing and metrics |
| `03_publication_parameter_sweep_atlas.ipynb` | `notebooks/scalar/04_scalar_parameter_sweeps.ipynb` | parameter sweeps and sensitivity |
| `04_publication_validation_benchmarks.ipynb` | `notebooks/scalar/05_scalar_validation_suite.ipynb`, `tests/`, `reference_kernels/` | numerical/physics validation |
| `05_publication_vector_parameter_atlas.ipynb` | `notebooks/vector/01_vector_beam_theory_atlas.ipynb` | vector/Jones-field beam atlas |
| `06_publication_lab_vs_ideal_vector.ipynb` | `notebooks/vector/02_vector_ideal_vs_lab_case1.ipynb`, `notebooks/vector/03_vector_hardware_routes.ipynb` | vector ideal-to-hardware comparison |
| `07_publication_materials_application.ipynb` / `.py` | `notebooks/materials/01_material_proxy_fluence_and_thresholds.ipynb`, `03_application_design_tables.ipynb` | material/application proxy calculations |
| `08_publication_calibration_report_export.ipynb` | `notebooks/materials/02_material_calibration_template.ipynb`, `calibration/`, `docs/calibration/`, `finalize_publication_outputs.py` | calibration bridge and export support |
| `09_publication_capsule_weld_feature_design.ipynb` | `notebooks/advanced/01_capsule_weld_feature_design.ipynb` | capsule/weld-feature design proxy |
| `10_publication_discrete_nfold_beams.ipynb` | `notebooks/advanced/03_discrete_nfold_beams.ipynb` | discrete N-fold structured beams |
| `NB_holographic_axicon.ipynb` | `notebooks/lab_realism/01_holographic_axicon_route.ipynb` | holographic axicon route |
| `NB_physical_axicon.ipynb` | `notebooks/lab_realism/02_physical_axicon_route.ipynb` | physical refractive axicon route |
| `NB_axicon_method_comparison.ipynb` | `notebooks/lab_realism/03_holographic_vs_physical_axicon.ipynb` | route comparison |
| `NB_through_sample.ipynb` | `notebooks/lab_realism/05_through_sample_interface.ipynb` | sample/interface propagation |
| `NB_full_journey.ipynb` | `notebooks/lab_realism/06_full_source_to_sample_journey.ipynb` | source-to-sample optical chain |
| `NB_validation.ipynb` | scalar validation notebook + `tests/` + governed `outputs/validation/` | validation/evidence chain |
| `NB_materials.ipynb` | `notebooks/materials/` | materials/application calculations |
| `NB_hexagon_study.ipynb` and old hex/polygonal checkpoint scripts | `notebooks/advanced/02_hexagonal_polygonal_beams.ipynb`, `notebooks/vector/04_vector_arm_hexagon.ipynb`, `notebooks/digital_twin/` | hexagonal/polygonal/vector structured-beam work |

## Code modules

The original flat `Publication_Study/vbb_study/*.py` code has evolved into the current `vbb_study/` package.  Use the current package rather than the duplicate `vbb_study - Copy` or dated backup directories from the old study folder.

The current package includes the original scalar/vector/material/application functionality plus later calibration and digital-twin code.  Important top-level compatibility modules such as `bessel_twin_core.py`, `publication_diagnostics.py` and `interface_correction_diagnosis.py` are retained where existing notebooks still rely on them.

## What was intentionally not migrated as day-to-day code

The clean repository does not promote dated `backups/`, `__pycache__/`, `vbb_study - Copy/`, `run_* - Copy.py`, or the bulk historical `outputs/` tree.  Those are provenance/history rather than competing current implementations.

The old checkpoint scripts are also not presented as co-equal canonical entry points.  Their useful functionality is represented by the current package/notebook workflows above, and `run_study.py` is the single normal execution entry point.

## Newer work beyond the original Publication_Study

The repository also contains later code that did not exist in the original Publication_Study snapshot, including the current optical digital-twin/calibration architecture and the measured q=20 axicon-aberration workflow under:

`notebooks/experimental/axicon_aberration_correction/`

For the full runnable map use:

```powershell
python run_study.py --list
```