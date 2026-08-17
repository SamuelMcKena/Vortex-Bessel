# Stage 9B.0.1 Upstream CSLM Bridge And Stop-Sampling Validity

Stage 9B.0.1 is a semantic and validation cleanup for the nominal F300 4F
virtual bench. It remains a nominal, unvalidated forward-model scenario.

## Scope Boundary

```text
Nominal unvalidated scenario.
Ideal continuous-ramp carrier surrogate.
Pixelated-SLM order physics not modelled.
Physical 4F readiness remains blocked.
No camera model.
No inverse correction.
No AI.
No material response.
final_export_allowed = False
```

## Corrected Chain

The Stage 9B.0 shortcut applied SLM1 phase directly at the SLM2 input of the
F300 model. Stage 9B.0.1 rejects that shortcut.

Executed candidate chain:

```text
source_field
-> input_conditioning_boundary
-> SLM1_phase_plane
-> SLM1_to_SLM2_segment
-> field_arriving_at_SLM2
-> SLM2_phase_plane
-> SLM2_to_lens1_propagation
-> lens1_thin_phase_and_pupil
-> lens1_to_fourier_plane_propagation
-> fourier_plane_field_pre_stop
-> fourier_stop_pinhole
-> fourier_plane_field_post_stop
-> fourier_plane_to_lens2_propagation
-> lens2_thin_phase_and_pupil
-> lens2_to_nominal_relay_output_propagation
-> nominal_relay_output_plane
```

Candidate runs now use `existing_cslm_component_route` by default. The nominal
F300 model starts from `field_arriving_at_slm2`, not from a synthetic SLM1 phase
shortcut. Passing `slm1_phase_rad` or `input_field` directly to
`run_nominal_f300_4f` raises an error.

## Carrier Boundary

The SLM2 carrier in this nominal model is an ideal continuous phase ramp. It is
a blazed-shift surrogate for positioning the spectrum relative to the nominal
Fourier stop.

It is not:

- pixelated-SLM diffraction-order physics
- fill-factor modelling
- zero-order leakage
- measured order efficiency
- selected-order purity
- calibrated Fourier-plane coordinate mapping

Machine-readable flags:

```text
carrier_realism = ideal_continuous_phase_ramp
ideal_blazed_carrier_shift_surrogate = True
pixelated_slm_diffraction_orders_modelled = False
zero_order_modelled = False
physical_order_efficiency_modelled = False
selected_order_purity_predicted = False
```

## Stop Sampling Profiles

Default nominal geometry:

```text
simulation_plane_width_m = 0.008
pinhole_radius_m = 0.00018
```

Exploratory profile:

```text
grid = 128 x 128
sampling_pitch_m = 6.25e-5
stop_diameter_pixels = 5.76
rounded_stop_diameter_pixels = 6
stop_sampling_status = exploratory_only
ranking_allowed = False
```

Standard ranking profile:

```text
grid = 256 x 256
sampling_pitch_m = 3.125e-5
stop_diameter_pixels = 11.52
rounded_stop_diameter_pixels = 12
stop_sampling_status = ranking_eligible
ranking_allowed = only after convergence passes
```

Convergence thresholds:

```text
energy relative difference <= 0.35
centroid difference <= 250 um
second-moment width relative difference <= 0.35
normalised intensity correlation >= 0.85
```

## Initial Shortlist

Only these candidates are part of the Stage 9B.0.1 initial ranking:

```text
gaussian_reference
vortex_ell_1
vortex_ell_2
vortex_ell_3
vortex_ell_4
```

Exploratory robustness sweeps remain available, but their rows are labelled
`exploratory_only_not_for_final_ranking`.

## Convergence Results

| candidate | status | energy diff | centroid diff (um) | width diff | correlation |
|---|---|---:|---:|---:|---:|
| `gaussian_reference` | passed_for_nominal_scenario | 0.006 | 0.3 | 0.059 | 0.999 |
| `vortex_ell_1` | passed_for_nominal_scenario | 0.052 | 8.4 | 0.112 | 0.990 |
| `vortex_ell_2` | passed_for_nominal_scenario | 0.105 | 32.4 | 0.087 | 0.964 |
| `vortex_ell_3` | passed_for_nominal_scenario | 0.148 | 71.5 | 0.073 | 0.926 |
| `vortex_ell_4` | passed_for_nominal_scenario | 0.222 | 128.5 | 0.046 | 0.859 |

## Ranking Eligible Candidates

All five initial candidates are eligible for nominal first-screen ranking under
the Stage 9B.0.1 sampling rule.

| rank | candidate | nominal score | relative output energy | pinhole fraction |
|---:|---|---:|---:|---:|
| 1 | `gaussian_reference` | 0.992 | 0.985 | 0.988 |
| 2 | `vortex_ell_1` | 0.796 | 0.631 | 0.651 |
| 3 | `vortex_ell_2` | 0.486 | 0.253 | 0.273 |
| 4 | `vortex_ell_3` | 0.387 | 0.074 | 0.084 |
| 5 | `vortex_ell_4` | 0.351 | 0.017 | 0.020 |

This ranking is useful for the first bench-screen order of operations only. It
does not validate the real Fourier stop, real SLM diffraction orders, camera
mapping, or any material outcome.

## Figures

```text
outputs/figures/digital_twin/stage9b0_1_upstream_cslm_to_nominal_4f_chain.png
outputs/figures/digital_twin/stage9b0_1_stop_sampling_convergence.png
outputs/figures/digital_twin/stage9b0_1_candidate_ranking_validity.png
```

## Before First Bench Session

Use Stage 9B.0.1 only to choose a nominal first-screen order and generate
unvalidated SLM command masks. Before treating the 4F response physically,
record actual SLM, lens, pinhole, camera, and axicon identifiers, then run the
downstream carrier-stop characterisation session from Stage 9A.1B.
