# Stage 8C.3R.5.1 CSLM-to-Physical-Axicon Handoff

Stage 8C.3R.5.1 adds an opt-in, physically labelled handoff from the
component-owned CSLM route to the downstream physical axicon benchmark.  It is
not a complete laboratory digital twin and it does not activate 4F filtering.

Claim boundary:

```text
free-space n = 1.0
optical-field / fluence diagnostics only
no material model
diagnostic_only
final_export_allowed=False
```

## Actual CSLM Diagnostic Branch

The active executed CSLM branch remains unfiltered:

```text
source_field
-> input_conditioning_boundary
-> SLM1_phase_plane
-> SLM1_to_SLM2_segment
-> SLM2_phase_plane
-> SLM2_to_pre_4F_diagnostic_segment
-> post_SLM2_pre_4F_diagnostic_plane
```

The output field is `post_slm2_unfiltered_diagnostic_field`.  It contains the
SLM2 correction/carrier phase and is not the final delivered laboratory field.

## Roles

SLM1 owns the vortex/topological-charge phase and phase-only conditioning.

SLM2 owns correction phase, carrier/blaze phase, phase wrapping, and phase
quantisation.  SLM2 does not produce an axicon phase.

The `slm2_correction_phase_rad` term is currently a **uniform piston placeholder**
(a single scalar broadcast to a uniform phase offset).  It is not an
aberration-correction map.  A true future correction requires a spatial phase map
(Zernike coefficient map / imported phase array / calibrated correction mask);
that system is intentionally not implemented in this stage.

The Bessel-forming element is the downstream physical axicon.  In this stage it
is consumed only by the explicit benchmark branch.

## Ideal Selected-Order Surrogate Branch

The benchmark branch is deliberate:

```python
order_handoff_mode = "ideal_selected_order_surrogate"
```

It creates `ideal_selected_order_surrogate_field` from the field arriving at
SLM2 after SLM1 propagation, then applies SLM2 correction phase while omitting
the carrier:

```text
U_slm2_in = propagated SLM1 vortex field at SLM2
U_actual = U_slm2_in * exp(i * quantize(wrap(phi_correction + phi_carrier)))
U_ideal = U_slm2_in * exp(i * quantize(wrap(phi_correction)))
```

This is a carrier-free desired-order benchmark, not a Fourier-plane crop and
not a physical +1 selected-order field.  It reports:

```text
physical_filter_modelled=False
zero_order_rejection_modelled=False
order_efficiency_modelled=False
selected_order_energy_uJ=None
```

## Physical Axicon Benchmark

The ideal surrogate is then passed through the existing scalar physical-axicon
transmission:

```text
ideal_selected_order_handoff_plane
-> physical_axicon_benchmark_plane
-> post_axicon_benchmark_segment
-> ideal_axicon_benchmark_reference_plane
```

The branch is marked:

```text
benchmark_only
not_physically_4F_filtered
not_experimental_prediction
```

The only represented loss is physical axicon clear-aperture clipping.  No 4F
order efficiency, zero-order suppression, or filter loss is invented.

## 4F Boundary

`fourier_filter_physics_available=False`.

The 4F lenses, Fourier plane, +1 order filter, relay optics, and 4F output
remain warning-only declarations.  No field named `filtered_field`,
`plus_one_field`, `4F_output`, or `physical_selected_order` is generated for the
benchmark.

## Required Measurements Before Replacing the Surrogate

A physical 4F-selected route needs:

- measured SLM1/SLM2 pixel pitch, active area, fill factor, and phase response;
- measured SLM1-to-SLM2 and SLM2-to-4F geometry;
- component-owned thin-lens propagation and Fourier-plane coordinate mapping;
- measured +1 filter centre, radius, shape, and alignment;
- order-resolved zero, +1, and residual energy calibration;
- measured physical axicon placement, clear aperture, cone parameter, and
  reference-plane distance.
