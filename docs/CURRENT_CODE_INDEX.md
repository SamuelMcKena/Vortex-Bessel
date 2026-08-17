# Current code index — what to run for each PhD task

This is the practical **“I want to do X — which code do I use?”** page.

The rule for new work is simple:

1. Use `vbb_study/` as the current model/library code.
2. Use the topic-organised notebooks below for interactive studies.
3. Use `run_study.py` (or the installed `vortex-bessel` command) as the normal front door.
4. Use `tools/` only for a named specialist validation, error-study or figure-regeneration task.
5. Do not go back to dated backups or `*- Copy` files to find a “newer” implementation.

## One-command navigation

```powershell
vortex-bessel --list
vortex-bessel --stage scalar
vortex-bessel --stage lab_realism
vortex-bessel --stage vector
vortex-bessel --stage materials
vortex-bessel --stage advanced
vortex-bessel --stage digital_twin
```

If the package is not installed editable yet, replace `vortex-bessel` with `python run_study.py`.

## Beam generation and ideal propagation

| I want to… | Current place to start |
|---|---|
| Review conventions, beam definitions and canonical cases | `notebooks/00_study_overview_and_conventions.ipynb` |
| Generate/compare ideal Bessel and vortex-Bessel cases | `notebooks/scalar/02_scalar_ideal_vs_lab_diagnostics.ipynb` |
| Inspect B0/V1/V3 transverse and longitudinal behaviour | `notebooks/scalar/02_scalar_ideal_vs_lab_diagnostics.ipynb` |
| Study robustness, obstruction and self-healing | `notebooks/scalar/03_scalar_robustness_and_self_healing.ipynb` |
| Sweep charge, beam/axicon/model parameters | `notebooks/scalar/04_scalar_parameter_sweeps.ipynb` |
| Run scalar validation/reference comparisons | `notebooks/scalar/05_scalar_validation_suite.ipynb`, `tests/`, `reference_kernels/` |
| Work directly with the maintained model code | `vbb_study/` and `bessel_twin_core.py` compatibility layer |

Normal command:

```powershell
vortex-bessel --stage scalar
```

## SLM, 4F filtering, physical axicon and lab-realistic routing

| I want to… | Current place to start |
|---|---|
| Simulate the holographic-axicon route | `notebooks/lab_realism/01_holographic_axicon_route.ipynb` |
| Simulate the physical refractive axicon route | `notebooks/lab_realism/02_physical_axicon_route.ipynb` |
| Compare holographic and physical axicons | `notebooks/lab_realism/03_holographic_vs_physical_axicon.ipynb` |
| Inspect objective pupil / first-order spatial filtering | `notebooks/lab_realism/04_objective_pupil_and_first_order_filtering.ipynb` |
| Propagate through a sample/interface | `notebooks/lab_realism/05_through_sample_interface.ipynb` |
| Follow the full source → SLMs → 4F → axicon → sample journey | `notebooks/lab_realism/06_full_source_to_sample_journey.ipynb` |

Normal command:

```powershell
vortex-bessel --stage lab_realism
```

## Controlled errors: axicon decentre, rounded tip, alignment and propagation errors

This is the current route for the error studies that used to be spread over several phase branches.

| I want to… | Current code |
|---|---|
| Generate the current vortex/system-error suite | `tools/run_vortex_system_error_suite.py` |
| Generate the default-nominal system-error suite | `tools/run_vortex_system_error_suite_default_nominal.py` |
| Generate the later governed system-error evidence set | `tools/run_vortex_system_error_evidence_v5.py` |
| Audit lateral beam/axicon decentre and rounded-tip signatures | `tools/audit_vortex_system_error_signatures.py` |
| Validate system-error propagation against references | `tools/check_system_error_propagation_reference.py` |
| Validate vortex error reference models | `tools/check_vortex_error_reference_models.py` |
| Check explicit 4F parity | `tools/check_vortex_explicit_4f_parity.py` |
| Check rotated-plane fields / carrier centring | `tools/check_rotated_plane_actual_fields.py`, `tools/check_rotated_plane_carrier_centering.py` |
| Independently benchmark a non-ideal/rounded axicon tip | `tools/check_axicon_tip_benchmark.py` |
| Independently benchmark continuous axicon propagation | `tools/check_axicon_continuous_propagation_benchmark.py` |
| Check oblique/tilted axicon propagation | `tools/check_axicon_oblique_benchmark.py`, `tools/check_axicon_oblique_wave_benchmark_v2.py` |
| Run the wider vortex/axicon physics suite | `tools/run_vortex_axicon_physics_suite_v3.py` |
| Run declared aberration cases | `tools/run_vortex_declared_aberration_suite.py` |

For the familiar **lateral axicon displacement** and **rounded-tip** studies, start with:

```powershell
python tools\run_vortex_system_error_suite.py
python tools\audit_vortex_system_error_signatures.py
```

For the independent rounded-tip reference check:

```powershell
python tools\check_axicon_tip_benchmark.py
```

## Vector beams and focusing

| I want to… | Current place to start |
|---|---|
| Review vector/Jones-field beam theory and cases | `notebooks/vector/01_vector_beam_theory_atlas.ipynb` |
| Compare ideal and lab-realistic vector propagation | `notebooks/vector/02_vector_ideal_vs_lab_case1.ipynb` |
| Study vector hardware routes | `notebooks/vector/03_vector_hardware_routes.ipynb` |
| Study the vector/hexagonal arm | `notebooks/vector/04_vector_arm_hexagon.ipynb` |
| Validate the vector refractive reference implementation | `tools/check_phase2h_vector_refractive_reference.py` |
| Re-render the governed Phase 2H vector evidence | `tools/render_phase2h_canonical_evidence.py` and related `render_phase2h_*` tools |

Normal command:

```powershell
vortex-bessel --stage vector
```

## Hexagonal, polygonal and other advanced structured beams

| I want to… | Current place to start |
|---|---|
| Work on capsule/weld feature design | `notebooks/advanced/01_capsule_weld_feature_design.ipynb` |
| Work on hexagonal/polygonal beams | `notebooks/advanced/02_hexagonal_polygonal_beams.ipynb` |
| Work on discrete N-fold structured beams | `notebooks/advanced/03_discrete_nfold_beams.ipynb` |
| Work on the vector hexagonal route | `notebooks/vector/04_vector_arm_hexagon.ipynb` |
| Work on later integrated digital-twin variants | `notebooks/digital_twin/` and `vbb_study/digital_twin/` |

Normal command:

```powershell
vortex-bessel --stage advanced
```

## Materials and application calculations

| I want to… | Current place to start |
|---|---|
| Calculate fluence/threshold/material proxies | `notebooks/materials/01_material_proxy_fluence_and_thresholds.ipynb` |
| Prepare/inspect calibration inputs | `notebooks/materials/02_material_calibration_template.ipynb`, `calibration/`, `docs/calibration/` |
| Produce application/design tables | `notebooks/materials/03_application_design_tables.ipynb` |
| Work directly with material-model code | `vbb_study/vbb_materials.py`, `vbb_study/vbb_materials_study.py` |

Normal command:

```powershell
vortex-bessel --stage materials
```

These remain model/application proxies unless the required laboratory calibration is supplied; the current repository does not silently promote nominal dimensions or fluence to measured values.

## Digital twin / calibrated-bench architecture

Use:

- `notebooks/digital_twin/` for the interactive later-stage digital-twin studies;
- `vbb_study/digital_twin/` for the maintained implementation;
- `configs/hardware/`, `configs/evidence/`, `configs/materials/` and `configs/studies/` for current configuration contracts;
- `calibration/` and `docs/calibration/` for the measurement bridge.

Normal command:

```powershell
vortex-bessel --stage digital_twin
```

## Measured q=20 axicon aberration / inverse correction

The current measured workflow is:

`notebooks/experimental/axicon_aberration_correction/Bessel_zscan_alignment_correction.ipynb`

Supporting modules in the same directory include:

- `modal_vortex_bessel.py` — BeamGage/z-stack loading and constrained vortex-Bessel fitting;
- `q20_modal_analysis.py` — all-plane q=20 modal reconstruction and correction proposal;
- `phase_error_recreation.py` — inverse-error recreation/falsification;
- `single_mask_inverse_forward_test.py` — stricter single-input-mask forward test;
- `comprehensive_error_validation.py` — high-resolution XY/XZ/YZ/profile/3-D diagnostics;
- `measured_beam_path_trajectory.py` — measured beam-centre trajectory;
- `slm2_complete_mask_preview.py` — native-resolution SLM2 preview generation;
- `iterative_correction_controller.py` — held-out-plane/gain-selection controller.

Set the local measurement directory, then run:

```powershell
$env:BESSEL_ZSCAN_DATA_DIR = 'D:\path\to\z-scan'
vortex-bessel --stage experimental
```

The raw acquisitions stay local; they should not be copied into Git just to make the code runnable.

## Presentation figures

For the cleaner/current presentation visual family:

- `tools/build_phase2j_ideal_beam_profile_figure.py`
- `tools/build_phase2j_presentation_suite.py`
- `tools/presentation_phase2j_style.py`

For the wider presentation evidence set:

- `tools/build_phase2i_presentation_figures.py`
- `tools/build_presentation_extended_evidence.py`
- `tools/build_presentation_extended_evidence_v2.py`
- `tools/render_vortex_visual_atlas.py`

These are figure builders, not alternative physics engines. Their scientific validity still depends on the underlying current model/evidence gates.

## Validation, evidence and report reproducibility

| Task | Current code |
|---|---|
| Run the normal regression suite | `python -m pytest tests -q` |
| Compile active source | `python -m compileall -q vbb_study tools tests` |
| Audit standalone path assumptions | `python tools/audit_clean_layout.py` |
| Audit notebooks for stale old-repo runtime paths | `python tools/audit_notebook_runtime_paths.py` |
| Audit propagation | `tools/audit_phase2b_propagation.py` |
| Build report freeze/provenance metadata | `tools/build_vortex_report_freeze.py` |
| Run Phase 2K output truth/semantic audits | `tools/phase2k_output_truth_audit.py`, `tools/phase2k_output_semantic_audit.py` |
| Check experimental-readiness dependencies | `tools/check_phase2i_experimental_readiness.py` |

The machine-readable governed evidence carried into this repository is under `outputs/validation/`.

## Old Publication_Study names

If an old note or message names one of the original numbered notebooks or `NB_*` files, use `docs/PUBLICATION_STUDY_MAP.md` to find its current replacement.

The old repository remains the historical provenance store. **Do not use its backup/copy files as competing implementations for new runs.** If a useful old calculation is genuinely missing here, it should be migrated and tested here rather than resurrected ad hoc from a backup.