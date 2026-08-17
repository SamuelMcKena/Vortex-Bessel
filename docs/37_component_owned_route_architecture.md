# Stage 8C.3R.4 Component-Owned Physical-Axicon Route Scaffold

Stage 8C.3R.4 defines a component-owned physical-axicon route scaffold. The
route is not a complete laboratory digital twin and is no longer treated as
generic field perturbations injected at arbitrary planes. It is now an ordered
component/segment chain:

```text
source_field
-> source_boundary_condition
-> input_aperture
-> source_to_physical_axicon
-> physical_axicon_input_boundary
-> physical_axicon
-> after_physical_axicon_boundary
-> post_axicon_free_space_segment
-> post_axicon_diagnostic_boundary
-> post_axicon_to_reference_segment
-> reference_plane
```

Each represented component or segment declares:

- `component_id`
- `component_type`
- nominal z position
- distance from previous component
- `distance_to_next_element_mm`
- enabled state
- physical pose: decentre, axial offset, tip, roll
- component-specific parameters
- clear aperture
- active / warning-only / future status
- whether the current engine represents it
- whether a physical model is available
- supported misalignment modes
- downstream elements affected

## Active Physics

The physical-axicon route currently executes only free-space scalar optics:

- source complex field
- input aperture at the conditioning plane
- free-space propagation segments
- thin scalar physical axicon phase and clear aperture
- free-space reference-plane diagnostics

Supported local component errors include input-aperture decentre/radius,
physical-axicon lateral decentre, physical-axicon axial offset through adjacent
segment distances, physical-axicon clear aperture, and the scalar axicon cone
phase parameter.

Physical axicon mechanical tilt is not silently converted into field tilt. It
remains `future_not_implemented` unless a documented thin-element/paraxial
approximation is added.

## Boundary Conditions

Field-state controls remain available only as labelled boundary conditions:

```text
field_tilt_x_mrad / field_tilt_y_mrad at field_tilt_location
beam_decentre_x_um / beam_decentre_y_um at beam_decentre_location
```

Every such record declares the boundary plane, physical approximation, which
upstream hardware error it could emulate, and which downstream components consume
it. These controls are not generic component misalignment models.

## Warning-Only / Future

The following remain explicitly warning-only or future-stage in this physical
axicon engine:

- steering mirrors
- SLM1/SLM2
- 4F Fourier filtering
- relay optics
- pupil/objective optics
- mechanical axicon tilt
- material/interface/dose/nonlinear/thermal/calibrated response

## Route Inspection

The pipeline now records a route-inspection row for every executed stage:

- component name/type
- nominal location
- actual pose error
- incoming/outgoing field metrics
- energy before/after
- centroid before/after
- angle before/after
- aperture overlap where relevant
- actual segment distance
- transform applied boolean
- downstream consequences
- model status and warnings

Preview:

```text
outputs/figures/digital_twin/stage8c3_component_route_inspection.png
```

The model remains diagnostic free-space optical/fluence only:

```text
n = 1.0
final_export_allowed = False
no material response
no Stage 8D
```
