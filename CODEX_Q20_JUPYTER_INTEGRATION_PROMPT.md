# Codex task: integrate the digital-twin q=20 correction work into the local BeamGage Jupyter workflow

Work in this repository and preserve the existing experimental analysis under:

`notebooks/experimental/axicon_aberration_correction/`

The local working copy is expected to contain the raw BeamGage data folders that are intentionally ignored by Git:

- `z-scan 2 1010/`
- possibly `new z-scan bessel beam 1010/`

The repository `.gitignore` deliberately excludes `*.bmg`; do not assume the absence of raw BMG files from Git means they are absent locally. The current measured dataset contract is 18 z planes x 4 repeats = 72 BMG frames.

## Experimental facts to bind first

Use the tracked measured metadata unless the local BMG files prove otherwise:

- wavelength: 1030 nm
- camera pixel scale: 5.5 um/px
- z step: 1 mm
- q = 20 bench convention (`ell_slm1 = +10`, `ell_slm2 = -10`, effective q=20)
- physical refractive axicon in the lab is nominally described as **20 degrees**

IMPORTANT: do not blindly set the model variable `axicon_base_angle_deg = 20` until you resolve the manufacturer's angle convention. Determine whether the quoted 20 degrees is apex angle, full cone angle, base angle, or the complementary convention. Search local notes/part number if available. If the exact convention is unavailable, treat the measured/fitted transverse wavevector `k_perp` and measured ring/propagation geometry as the primary experimental calibration and report the 20-degree manufacturer nominal separately. Do not force a 20-degree value into the wrong internal geometric parameter.

## Objective

Create a new, clean notebook (do not destroy the existing working notebooks), preferably:

`notebooks/experimental/axicon_aberration_correction/Bessel_zscan_digital_twin_correction.ipynb`

The notebook must take the **real local BMG z-stack** as its input and integrate the new physical-system/digital-twin inverse code developed on branch `poster/figure-curation-v2`.

The final notebook should answer this question:

> Starting from the same measured q=20 propagation stack, how does the correction predicted by the published Miao-style inverse compare with a correction that first estimates physically meaningful system parameters with the full beam/SLM/4F/axicon model and then fits the remaining phase through the complete forward model?

Do not manufacture a post-correction measurement. Until a new camera stack is actually acquired, corrected fields are **model predictions driven by the measured BMG data**.

## New code that must be integrated, not rewritten from scratch

The core new modules are:

- `vbb_study/digital_twin/physical_error_dictionary.py`
- `vbb_study/digital_twin/hierarchical_physical_fit.py`
- `vbb_study/digital_twin/physical_observable_fit.py`
- `vbb_study/digital_twin/parameter_measurement_policy.py`
- `vbb_study/digital_twin/residual_phase_fit.py`
- the updated `vbb_study/digital_twin/vortex_system_route.py`

Reference/validation implementations developed alongside them are in:

- `tools/benchmark_q20_miao_vs_digital_twin.py`
- `tools/benchmark_q20_method_physics_v2.py`
- `tools/benchmark_q20_method_physics_v3.py`
- `tools/benchmark_q20_method_physics_v4.py`
- `tools/benchmark_q20_full_model_phase_refinement_v1.py`
- `tools/benchmark_q20_integrated_correction_v5.py`
- `tools/benchmark_q20_integrated_correction_v6.py`
- `tools/diagnose_miao_full_route_compatibility.py`
- `tools/diagnose_q20_sampling_convergence.py`
- `tools/validate_miao_modal_equation_contract.py`
- `tools/validate_miao_q20_contract.py`

The existing experimental Miao/BMG code under `notebooks/experimental/axicon_aberration_correction/` remains authoritative for loading the real data and for the published-method baseline. In particular inspect and reuse rather than replace:

- `modal_vortex_bessel.py`
- `miao_full_retrieval.py`
- `run_q20_miao_retrieval.py`
- `q20_phase_physics.py`
- `rebuild_q20_presentation_from_bmg.py`
- `q20_experimental_acceptance_metrics.py`

## Required architecture

Implement the notebook in the following stages.

### 1. Load and verify the real BMG stack

- Find the local BMG folder automatically from the known folder names.
- Require exactly 18 z positions and 4 repeats per z position unless a clearly newer complete acquisition is discovered.
- Reuse the existing BMG parser/registration code.
- Register repeated captures within each z position only.
- Do **not** recenter each z plane onto another; preserve real beam walk through z.
- Show the measured XY contact sheet and measured XZ/YZ maps first so we can visually confirm we are using the actual laboratory data.

### 2. Build the experimental target/model calibration

- Use the measured wavelength, pixel scale, q, z-step and available bench metadata.
- Resolve/fit `k_perp` from the real q=20 data using the established Miao route and/or measured ring geometry.
- Treat directly measurable quantities such as input Gaussian radius and SLM LUT as calibration inputs, not arbitrary z-stack fit parameters.
- Inspect the 20-degree axicon specification and convert it to the internal model only if the convention is known. Otherwise use the experimentally inferred `k_perp` to constrain the effective axicon cone response.

### 3. Physical-system estimation with the digital twin

Use the same forward route used by the error studies:

Gaussian -> SLM1 -> SLM2/carrier -> explicit propagated 4F with fixed +1-order iris -> axicon -> free-space propagation.

Do not fit every error parameter at once. Use `parameter_measurement_policy.py` and the observable-specific functions:

- lateral trajectory / first moments for physical alignment quantities such as axicon displacement and suitable iris/beam offsets;
- centered azimuthally averaged radial morphology for iris opening / spatial-filter morphology;
- longitudinal structure for lens despace or tip-shape hypotheses only when the data contains enough information;
- calibration-only quantities (beam radius, SLM phase stroke/LUT, fringing kernel) must remain bound from direct calibration or clearly marked unknown.

Start with a small physically interpretable parameter set and add a parameter only if injection/recovery or uncertainty tests show it is identifiable from this dataset.

For each fitted physical parameter report:

- fitted value and units;
- objective definition in plain mathematical terms;
- confidence/uncertainty or local sensitivity;
- whether the minimum is interior rather than sitting on a search boundary;
- whether a competing parameter produces a nearly indistinguishable solution.

Do not use vague words such as "error family" in scientific plots. Use physical names such as `Axicon lateral displacement`, `4F iris radius`, `4F iris offset`, etc.

### 4. Published Miao-only correction baseline

Run the existing Miao-style intensity-only retrieval on the same measured stack using the established implementation. Preserve the deliberately programmed q=20 vortex term; retrieve only the residual aberration.

Generate the predicted corrected field through the appropriate analytical/forward route. Label it clearly as `Miao-only model prediction` unless a real post-correction BMG acquisition exists.

### 5. Digital-twin-assisted residual correction

After the physical-system parameters are estimated, rebuild the best-fit physical forward model and then fit the **remaining residual phase** through the complete forward model using `residual_phase_fit.py`.

Use held-out z planes:

- fit the residual phase on alternating/selected z planes;
- assess it on unused z planes;
- fail or warn if held-out agreement does not improve or if the result only improves the fitted planes.

The synthetic contract tests on the branch showed this full-model residual fit can outperform the stationary-phase approximation when the complete optical route departs from the assumptions of the analytical model. The local BMG notebook must test whether that remains true for the measured data; do not assume the answer.

### 6. Correction plane

The intended compensating layer is SLM2. Do not simply subtract a phase at an unrelated numerical plane.

- If the local experiment confirms SLM2 is conjugate to the reconstructed reference plane and the scale/centre/rotation/parity are known, map the correction into native SLM2 coordinates.
- If that calibration is not known, use the digital twin to make a **model-predicted SLM2 correction** through the modeled relay, but label it as a numerical prediction rather than hardware-ready.
- Preserve the programmed vortex/carrier and add only the residual correction phase.

### 7. Mandatory visual comparison

Produce a high-resolution four-way scientific comparison based on the **real measured BMG input**, not the old synthetic `Original` benchmark:

1. `Measured BMG`
2. `Miao-only predicted correction`
3. `Digital-twin + Miao predicted correction` (if useful as an intermediate method)
4. `Complete digital-twin residual correction`
5. `Target` may be included as an additional reference column if layout permits.

For each route show:

- a representative transverse XY intensity plane;
- XZ propagation over the measured z range;
- YZ if asymmetry is materially different from XZ;
- an intensity line/radial profile at the same selected z plane.

Use the same physical axes and common intensity normalization for direct visual comparison. Use the black-red-orange-yellow thermal colour map used in the poster work. No interpolation that changes metric values. Export PNG at >=500 dpi and a vector PDF where possible.

The representative z plane must be selected from the target/measured-data criterion before looking at correction performance, not cherry-picked to maximize the new method.

### 8. Quantitative comparison

For every z plane report at minimum:

- Pearson intensity correlation to the same target/reference definition;
- normalized RMSE;
- ring/core metric appropriate to q=20;
- azimuthal ring uniformity where meaningful;
- beam/core trajectory through z.

Show the metrics versus z and summarize the median/mean over the evaluated propagation region.

Define every metric on the figure/notebook. Do not write unexplained `loss` numbers.

### 9. Numerical checks

Before trusting the result:

- verify the Miao modal-equation unit test still passes;
- verify the complete route is sufficiently sampled;
- run a convergence spot-check at 512 and at least one finer grid (768 or 1024) on a reduced subset of z planes;
- preserve fixed physical coordinate planes for longitudinal maps;
- do not allow z-dependent propagation support masks to generate discontinuous XZ maps;
- inspect whether the 20-degree physical axicon requires the vector/high-angle route rather than the shallow scalar approximation. If it does, use the appropriate validated route already in the repository or explicitly state the remaining approximation.

### 10. Outputs

Create an output folder such as:

`notebooks/experimental/axicon_aberration_correction/outputs/digital_twin_correction/`

Save:

- measured BMG stack/contact sheet;
- measured XZ/YZ;
- fitted physical-system parameters JSON/CSV;
- residual phase maps;
- predicted SLM2 correction phase array;
- method-comparison metrics CSV;
- metrics-vs-z figure;
- the high-resolution measured/Miao/digital-twin/target comparison figure;
- a short Markdown summary of what improved, what did not, and what still requires physical calibration.

Also save the numerical arrays required to rerender figures without rereading all BMG files.

## Critical scientific constraints

- Do not substitute the old synthetic q=20 distorted field for the measured BMG data.
- Do not claim an experimental correction unless a genuinely new post-correction BMG stack exists.
- Do not tune hidden synthetic truth values because there is no hidden truth in the measured experiment.
- Do not force all forward-model parameters into one optimizer.
- Do not treat beam-radius or SLM-LUT uncertainty as uniquely inferable from the propagation stack.
- Do not remove the programmed q=20 vortex phase.
- Do not use a model prediction to validate itself; use held-out z planes and later, when available, a new measured post-correction stack.
- Do not overwrite the original BMG data or the existing working notebooks.

## Deliverable from Codex

Actually edit/create the notebook and helper code, run it against the local BMG data, inspect every generated plot, and iterate until the physical fit and correction outputs are numerically and visually coherent. Do not stop after writing code. At the end provide:

1. the new/modified file list;
2. the fitted physical parameters and their interpretation;
3. Miao-only vs digital-twin-assisted quantitative results;
4. links/paths to the final comparison figures;
5. any unresolved calibration quantities, especially the exact interpretation of the nominal 20-degree axicon specification and the SLM2 coordinate/LUT mapping.
