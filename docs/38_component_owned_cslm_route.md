# Stage 8C.3R.5 Component-Owned CSLM Route

Stage 8C.3R.5 adds a component-owned concatenated-SLM baseline route for the
programmable vortex-Bessel research path.  It is not a complete laboratory
digital twin.  It is a free-space optical-field and fluence diagnostic scaffold.

Claim boundary:

```text
free-space optical-field and fluence diagnostics
n = 1.0
diagnostic_only
final_export_allowed=False
no material model
```

The physical-axicon route from Stage 8C.3R.4.1 remains frozen as its own narrow
component-owned free-space reference scaffold.

## Conceptual Route Declaration

The declared CSLM architecture is:

```text
source_field
-> input_conditioning_boundary
-> SLM1_phase_plane
-> SLM1_to_SLM2_segment
-> SLM2_phase_plane
-> SLM2_to_pre_4F_diagnostic_segment
-> post_SLM2_pre_4F_diagnostic_plane
-> SLM2_to_fourier_lens_segment
-> Fourier_lens_1
-> Fourier_plane
-> plus_one_order_filter
-> Fourier_lens_2
-> 4F_output_plane
```

Only physically represented transforms are executed.  Unrepresented 4F elements
remain declarations with warning-only or future status.

## Executed Route

The active executed chain is:

```text
source_field
-> input_conditioning_boundary
-> SLM1_phase_plane
-> SLM1_to_SLM2_segment
-> SLM2_phase_plane
-> SLM2_to_pre_4F_diagnostic_segment
-> post_SLM2_pre_4F_diagnostic_plane
```

Active physics:

- source complex field;
- SLM1 vortex / phase-only conditioning;
- free-space propagation from SLM1 to SLM2;
- SLM2 correction/carrier phase handling with no axicon phase term;
- free-space propagation to a post-SLM2 pre-4F diagnostic plane;
- post-SLM2 unfiltered field-state diagnostics.

SLM1 and SLM2 are separate programmable planes.  SLM1 is treated as
`phase_only_conditioning`; in the baseline it owns the vortex/topological-charge
phase and does not claim validated amplitude shaping.  SLM2 is treated as
`phase_correction_and_carrier_preserve_vortex`; it composes carrier/blaze and
optional correction terms only, with wrapping and phase quantisation before
propagation.

Important correction: SLM2 does not produce the axicon phase in this route.
The Bessel-forming element is a downstream physical axicon and is not part of
the default active CSLM diagnostic branch.  Stage 8C.3R.5.1 adds an opt-in ideal
selected-order benchmark branch for that handoff; it is not a physical 4F
prediction.

## Warning-Only / Future Declarations

The following parts of the conceptual CSLM architecture are retained but not
executed as physical transforms:

- `SLM2_to_fourier_lens_segment`;
- `Fourier_lens_1`;
- `Fourier_plane`;
- `plus_one_order_filter`;
- `Fourier_lens_2`;
- `4F_output_plane`.

Objective, pupil, material interface, nonlinear propagation, plasma, thermal,
dose, writing trajectory, and calibrated material response are outside this
stage.

## 4F Feasibility Decision

`fourier_filter_physics_available=False`.

The current code has angular-spectrum free-space propagation and phase-mask
helpers, but it does not yet own a validated component-level 4F model with:

- measured SLM pixel pitch and active area for the actual devices;
- component-owned thin-lens transforms;
- carrier-to-Fourier-plane coordinate mapping in physical units;
- measured Fourier filter centre and radius;
- measured SLM2/lens/filter/lens/output separations;
- order-resolved zero, +1, and residual energy calibration.

Therefore the +1 order filter is a route declaration and configuration contract
only.  No fake filtered output field is generated.

## Energy and Normalisation Policy

Phase-only SLM operations preserve the energy ledger and discrete field power.
Free-space propagation is diagnostic angular-spectrum propagation.  Future
passive apertures or order filters must reduce energy without hidden
renormalisation.  The current inactive 4F filter reports no selected-order
energy because no physical filter field is applied.

## Diagnostic Outputs

The required Stage 8C.3R.5 figures are:

```text
outputs/figures/digital_twin/stage8c3r5_cslm_route_inspection.png
outputs/figures/digital_twin/stage8c3r5_slm_phase_and_field_baselines.png
outputs/figures/digital_twin/stage8c3r5_fourier_order_selection_audit.png
```

The composed SLM phase masks are also saved for inspection at:

```text
outputs/figures/digital_twin/stage8c3r5_phase_masks.npz
```

All outputs are diagnostic-only and `final_export_allowed=False`.

## Hardware Measurements Needed

The route can become calibrated only after supplying measured hardware geometry:

- SLM1 and SLM2 pixel pitch, active area, fill factor, phase response, and
  alignment;
- SLM1-to-SLM2 propagation or relay geometry;
- SLM2 blaze/carrier/correction convention and measured order positions;
- the downstream physical axicon placement and aperture if a Bessel-like
  benchmark is claimed;
- 4F lens focal lengths, clear apertures, and separations;
- Fourier-plane coordinate calibration;
- physical +1 filter centre, radius, and shape;
- order-resolved power measurements;
- reference-plane camera magnification and pixel scale.

Until then this remains a component-owned, free-space diagnostic route.
