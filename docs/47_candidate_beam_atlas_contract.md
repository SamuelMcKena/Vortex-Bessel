# Stage 9B.0/9B.0.1 Candidate Beam Atlas Contract

The candidate atlas uses the nominal F300 4F virtual bench to compare
unvalidated command-mask candidates before bench calibration exists.

It is explicitly:

```text
nominally_simulated
command_masks_exportable_unvalidated
not_bench_validated
final_export_allowed = false
```

Stage 9B.0.1 requires the upstream CSLM bridge. Candidate fields arrive at
SLM2 from the existing component route; SLM1 phase is not applied directly at
the nominal F300 input.

## SLM Role Contract

```text
SLM1: vortex / structured phase conditioning at SLM1
SLM1_to_SLM2_segment: propagated upstream field
SLM2: ideal continuous carrier surrogate and future correction map only
```

SLM2 must not be used to generate an axicon phase in this atlas. The carrier is
not a pixelated-SLM diffraction-order, zero-order leakage, fill-factor, or
selected-order-purity model.

## Candidate Families

The Stage 9B.0.1 initial atlas covers:

- Gaussian reference,
- vortex charge sweep for `ell=1..4`.

The study contract is stored in:

```text
configs/studies/cslm_nominal_4f_candidate_atlas_v1.json
```

## Robustness Sweeps

The atlas reports nominal sensitivity only:

- carrier cycles versus stop-offset transmission,
- pinhole radius versus stop-offset transmission,
- beam radius versus relay quality.

These sweeps are exploratory only unless stop sampling and convergence gates
pass. They are not physical 4F validation and do not infer camera-plane
coordinates.

## Candidate Package

Each exported candidate package is written under:

```text
outputs/nominal_4f_candidate_runs/<run_id>/<candidate_id>/
```

Required files:

```text
run_manifest.json
candidate_manifest.json
nominal_4f_profile_snapshot.json
stop_sampling_convergence_report.json
SLM1 phase_rad.npy
SLM1 quantised_rad.npy
SLM1 gray.png
SLM2 phase_rad.npy
SLM2 quantised_rad.npy
SLM2 gray.png
fourier_plane_pre_stop.png
fourier_stop_transmission.png
fourier_plane_post_stop.png
nominal_relay_output_xy.png
energy_ledger.csv
robustness_summary.csv
claim_boundary.md
```

Every manifest states:

```text
simulation_status = nominal_unvalidated
physical_4f_readiness = blocked
camera_validation = absent
material_prediction = absent
final_export_allowed = false
carrier_realism = ideal_continuous_phase_ramp
pixelated_slm_diffraction_orders_modelled = false
stop_sampling_status = convergence_verified or exploratory_only
```

## Unsupported

Unsupported remains: direct Fourier-plane calibration, measured carrier-to-stop
mapping, camera model, local SLM phase calibration, physical order purity,
validated correction maps, physical axicon handoff, material response, and any
publication-ready final export.
