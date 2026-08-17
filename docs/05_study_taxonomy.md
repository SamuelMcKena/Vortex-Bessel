# Study Taxonomy

The publication project is a structured-beam simulation atlas. Each study case
should be described with the same small set of labels so scalar, vortex, vector,
hexagonal, discrete, and material-application branches can be compared without
implying the same hardware or validation status.

The source version of these labels lives in
`Publication_Study/vbb_study/study_taxonomy.py`.

## Label Fields

| field | purpose |
|---|---|
| `beam_family` | What kind of structured beam or application branch is being studied. |
| `model_level` | How much physics/hardware fidelity is included. |
| `generation_method` | Which ideal or lab route is being modeled. |
| `hardware_status` | Whether the route is mathematical, lab-proxy, current-hardware, proposed, or unsupported. |
| `material_model_status` | Whether the material layer is absent, proxy-only, pending calibration, or calibrated. |
| `qa_status` | How much checking has been applied to the result. |

## Standard Values

### `beam_family`

- `scalar_bessel`
- `vortex_bessel`
- `vector_bessel`
- `polygonal_hexagonal`
- `discrete_nfold`
- `material_capsule`

### `model_level`

- `analytic`
- `scalar_paraxial`
- `vector_jones`
- `lab_encoded`
- `material_proxy`

### `generation_method`

- `ideal_target`
- `holographic_slm`
- `physical_axicon`
- `segmented_vector_element`
- `discrete_plane_wave`
- `zernike_hex_seed`

### `hardware_status`

- `ideal_math`
- `lab_proxy`
- `case1_existing_hardware`
- `proposed_hardware`
- `unsupported`

### `material_model_status`

- `not_applicable`
- `planning_proxy`
- `calibrated_pending`
- `experiment_calibrated`

### `qa_status`

- `exploratory`
- `smoke_checked`
- `validation_pipeline_checked`
- `publication_candidate`
- `out_of_validity`

## Branch Coverage

Scalar Bessel, vortex Bessel, vector Bessel, polygonal/hexagonal beams,
discrete N-fold beams, and material/capsule application studies are all
first-class atlas branches. They differ in model level and hardware status; the
taxonomy should make those differences explicit instead of treating any branch
as a side experiment.

## Stage 8 Advanced-Beam Labels

Stage 8 uses a stricter publication schema for hexagonal, polygonal, hollow
outline, and discrete N-fold optical cases. These labels are intentionally more
specific than the older broad taxonomy values:

- `beam_family`: `hexagonal_polygonal`, `discrete_nfold`,
  `nfold_vortex_ring`, `hollow_polygon`, `scalar_reference`, or `diagnostic`.
- `model_level`: `ideal_target`, `focal_plane_target`,
  `numerical_propagation`, `lab_realistic`, `hardware_route`,
  `geometry_proxy`, or `diagnostic_only`.
- `generation_method`: `phase_only_slm`, `complex_amplitude_proxy`,
  `amplitude_phase_target`, `discrete_superposition`,
  `holographic_phase_mask`, `future_hardware_required`, or
  `simulation_only`.
- `hardware_status`: `current_lab_realizable`, `future_hardware_required`,
  `simulation_only`, or `diagnostic_only`.
- `propagation_stability_status`: `not_tested`, `focal_plane_only`,
  `propagation_tested_pass`, `propagation_tested_marginal`,
  `propagation_tested_fail`, or `diagnostic_only`.

A hexagonal or polygonal focal-plane pattern is not automatically a
propagation-stable Bessel-like beam. Propagation stability must be measured
with z-dependent metrics such as accepted depth, symmetry retention, outline
fidelity, core suppression, and side-lobe contamination.

Phase-only SLM compatibility is not assumed for complex-amplitude polygonal
targets. If complex amplitude is required, the case is labelled
future_hardware_required or simulation_only unless a tested encoding route is
provided.

## Ideal Target Versus Lab-Realistic Route

The recurring theme is ideal target versus lab-realistic implementation. An
ideal target can be useful and publishable as a reference, but a lab-realistic
claim needs a specific generation method, aperture/filter/SLM or axicon limits,
and an honest hardware status label.
