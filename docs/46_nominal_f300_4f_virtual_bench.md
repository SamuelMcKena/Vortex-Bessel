# Stage 9B.0 Nominal F300 4F Virtual Bench

Stage 9B.0 adds an opt-in scalar forward model for the nominal F300 relay.
Stage 9B.0.1 clarifies that candidate runs enter this model through the
upstream CSLM bridge as `field_arriving_at_slm2`; SLM1 phase is not applied
directly at the F300 input.

```text
field arriving at SLM2 from existing CSLM route
-> SLM2 ideal continuous carrier surrogate
-> 300 mm free-space propagation
-> Lens 1, f = 300 mm, thin scalar phase plus editable clear aperture
-> 300 mm free-space propagation
-> Fourier / pinhole plane
-> 300 mm free-space propagation
-> Lens 2, f = 300 mm, thin scalar phase plus editable clear aperture
-> 300 mm free-space propagation
-> nominal relay-output plane
```

This is a nominal virtual bench only. It is not a measured or calibrated physical
4F model, not a camera model, not an inverse-correction model, not an AI model,
and not a material-response model.

## Boundary Labels

```text
nominal_4f_forward_model
not_bench_calibrated
not_physical_4f_readiness_ready
not_camera_modelled
not_material_modelled
final_export_allowed = false
```

The physical 4F readiness gate remains blocked. Existing CSLM route execution
and the downstream empirical carrier-stop characterisation are not changed.

The SLM2 carrier is an ideal continuous-ramp/blazed-shift surrogate. It is not
a pixelated-SLM diffraction-order, fill-factor, zero-order leakage, physical
order-efficiency, or selected-order-purity model.

## Implemented Components

The nominal model is component-owned and explicit:

| component | status | transform |
|---|---|---|
| `SLM2_phase_plane` | nominal model | SLM2 carrier phase only |
| `SLM2_to_lens1_propagation` | nominal model | free-space BL-ASM segment |
| `lens1_thin_phase_and_pupil` | nominal model | thin scalar lens phase and circular pupil |
| `lens1_to_fourier_plane_propagation` | nominal model | free-space BL-ASM segment |
| `fourier_plane_field_pre_stop` | diagnostic boundary | no transform |
| `fourier_stop_pinhole` | nominal stop parameter | circular amplitude stop at Fourier plane |
| `fourier_plane_field_post_stop` | diagnostic boundary | no transform |
| `fourier_plane_to_lens2_propagation` | nominal model | free-space BL-ASM segment |
| `lens2_thin_phase_and_pupil` | nominal model | thin scalar lens phase and circular pupil |
| `lens2_to_nominal_relay_output_propagation` | nominal model | free-space BL-ASM segment |
| `nominal_relay_output_plane` | diagnostic boundary | no transform |

The stop is applied at the Fourier plane, not as a crop at the relay output.

## Configuration

The nominal profile is stored in:

```text
configs/hardware/cslm_f300_nominal_4f_profile.json
```

All six 300 mm distances/focal lengths are marked as
`nominal_from_bench_description`. Unknown hardware details remain null,
including lens part numbers, lens clear apertures, SLM pixel pitch, SLM phase
calibration, pinhole diameter, physical stop coordinates, camera relation to the
relay-output plane, relay-output-to-axicon distance, and axicon pose.

## Diagnostics

The model reports:

- component sequence and `transform_applied` flags,
- component energy ledger,
- Fourier-plane pre/post-stop fields,
- stop transmission,
- relay-output field,
- sampling and clipping warnings,
- open-stop relay sanity metrics,
- `carrier_coordinate_status = nominal_model_not_bench_calibrated`.

Figures:

```text
outputs/figures/digital_twin/stage9b0_nominal_f300_4f_component_sequence.png
outputs/figures/digital_twin/stage9b0_nominal_f300_4f_stop_robustness.png
```

## Unsupported

Still unsupported: measured physical 4F coordinate calibration, real stop
position/radius, lens clear-aperture/part-number validation, SLM phase response
at 1030 nm, camera imaging, inverse correction, neural estimation, downstream
axicon handoff geometry, and any material prediction.
