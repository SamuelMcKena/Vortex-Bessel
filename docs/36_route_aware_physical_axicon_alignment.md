# Stage 8C.3R.3 Route-Aware Physical-Axicon Alignment

Stage 8C.3R.3 corrects the route-aware architecture for the free-space
physical-axicon diagnostic path. The study remains optical/fluence only,
`n = 1.0`, with `final_export_allowed=False`. It adds no material model, sample
interface, writing trajectory, 3D view, GUI, plasma, nonlinear propagation,
thermal response, or calibrated prediction.

Stage 8C.3R.4 supersedes the narrow C3R.3 route interpretation. The physical
axicon path is now executed as a component-owned physical-axicon route scaffold
with local component pose errors, explicit free-space propagation segments,
boundary-condition rows, and a route-inspection table. See
`docs/37_component_owned_route_architecture.md`.

## Route Concept

The physical-axicon route is represented as an ordered graph:

```text
source_plane
-> after_beam_conditioning
-> propagation: source to before_physical_axicon
-> physical_axicon_plane
-> after_physical_axicon
-> propagation: axicon to after_objective
-> after_objective
-> propagation: after_objective to free_space_reference_plane
-> free_space_reference_plane
```

The segment distances are editable diagnostic demo geometry. The defaults are
not measured laboratory distances and must not be described as the actual bench.

## Location-Aware Perturbations

Perturbations are no longer generic sliders such as `beam_tilt`. Each record
declares:

- `perturbation_type`
- `magnitude`
- `units`
- `injection_location`
- `route`
- `implementation_plane`
- `downstream_elements_affected`
- active / warning-only / future status

Field tilt is represented by:

```text
field_tilt_x_mrad
field_tilt_y_mrad
field_tilt_location
```

Beam decentre is represented by:

```text
beam_decentre_x_um
beam_decentre_y_um
beam_decentre_location
```

The same field tilt produces different downstream behaviour depending on where it
is injected.

## Upstream Tilt Versus Post-Axicon Steering

When field tilt is injected upstream of the physical axicon, the phase ramp is
applied at the selected represented plane, the field propagates through all
remaining pre-axicon segments, and the fixed physical axicon is illuminated by
that incident field. The axicon-plane diagnostics report centroid, angle,
beam-to-axicon offset, beam radii, ellipticity, clear-aperture overlap, energy
arriving at the axicon, energy transmitted through the axicon, and predicted
versus measured walkoff.

When field tilt is injected immediately after the physical axicon, it is labelled
`post_axicon_steering_test`. That is an analytical downstream steering check, not
an upstream alignment fault.

When field tilt is injected after the objective/downstream steering plane, it
does not affect the physical axicon incidence at all; it only affects the
represented downstream free-space segment.

## Mechanical Element Tilt

Mechanical element tilt is not silently treated as generic field tilt.

`physical_axicon_mechanical_tilt_x_mrad` and
`physical_axicon_mechanical_tilt_y_mrad` are currently
`future_not_implemented` unless a documented thin-element/paraxial approximation
is added at the axicon plane. SLM/mirror/lens mechanical tilts are likewise not
made physics-active without an explicit represented plane and model.

## Represented And Unrepresented Planes

Represented for the physical-axicon route:

- source plane
- after beam conditioning
- before physical axicon
- physical axicon plane
- after physical axicon
- post-axicon diagnostic boundary
- free-space reference plane

Warning-only / not represented by the current engine:

- explicit 4F Fourier selection plane
- relay imaging plane
- SLM mechanical tilt approximation
- physical axicon mechanical tilt approximation
- stochastic pointing/stage jitter and focus drift ensembles
- material/interface/dose/nonlinear/thermal/calibrated response

## Figures

The C3R.3 preview outputs are:

```text
outputs/figures/digital_twin/stage8c3r3_route_aware_axicon_pipeline.png
outputs/figures/digital_twin/stage8c3r3_upstream_vs_post_axicon_tilt_comparison.png
outputs/figures/digital_twin/stage8c3r3_axicon_alignment_sensitivity_atlas.png
```

All are diagnostic only. The comparison figure is a field-tilt injection-location
sweep, not a binary source-tilt/post-tilt shortcut.

## Remaining Gaps

Before material modelling or GUI work, the route still needs measured bench
distances, real optical-element geometry, calibrated component apertures,
explicit 4F/relay planes if they are needed, and a documented model for any
mechanical element tilt that should become physics-active.
