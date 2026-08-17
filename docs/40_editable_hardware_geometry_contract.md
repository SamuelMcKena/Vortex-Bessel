# Stage 8C.3R.5.2 — Editable Hardware and Geometry Control Contract

A single metadata layer (`vbb_study/digital_twin/control_contract.py`) over the existing
Stage 8C.3R.5 / R5.1 config dataclasses. `CSLMRouteConfig` remains the **single source of
model values**; the registry only describes and maps editable notebook values onto those
config fields. No new optical physics; the R5.1 active-route and benchmark-route meaning is
unchanged.

Boundary (unchanged): `n = 1.0` free-space optical-field / fluence diagnostics only;
`fourier_filter_physics_available = False`; `diagnostic_only`; `final_export_allowed = False`.
No material / 4F / camera / GUI / 3D physics.

## Parameter-status system

Each `EditableControl` carries one status from this exact vocabulary:

| status | meaning |
|---|---|
| `physics_active` | applied by the active CSLM diagnostic route (`order_handoff_mode="none"`) |
| `benchmark_only` | only used by the opt-in ideal physical-axicon benchmark branch |
| `warning_only` | editable hardware record; **no effect on the current field** (e.g. 4F geometry, SLM pixel pitch, camera) |
| `future_not_implemented` | future capability; stored but never applied (e.g. spatial correction map, beam ellipticity) |
| `numerical_advanced` | numerical grid/propagation control, not laboratory hardware |
| `derived_read_only` | governance/derived value (`diagnostic_only`, `final_export_allowed`, `fourier_filter_physics_available`) |

## Provenance labels

`measured`, `manufacturer_specification`, `estimated`, `diagnostic_placeholder`, `unknown`,
`derived`. A parameter with provenance `unknown` or `diagnostic_placeholder` is never presented
as measured laboratory geometry. In the demo profile there are **zero** `measured` controls.

## Control counts (66 total)

physics_active = 15, benchmark_only = 6, warning_only = 29, future_not_implemented = 8,
numerical_advanced = 5, derived_read_only = 3.

## Which values affect current physics

- **Active field** (`physics_active`): route_mode, wavelength_nm, n_medium, input_pulse_energy_uJ,
  input_beam_radius_um, slm1_phase_mode, slm1_topological_charge, slm1_linear_ramp_cpm,
  slm1_to_slm2_distance_mm, slm2_conjugate_mode, slm2_carrier_frequency_cpm,
  slm2_correction_phase_rad (uniform piston placeholder), slm_phase_quantisation_levels,
  slm2_to_pre_4f_diagnostic_distance_mm, order_handoff_mode.
- **Benchmark branch only** (`benchmark_only`, active when
  `order_handoff_mode="ideal_selected_order_surrogate"`): the six physical-axicon controls.
- **Numerical** (`numerical_advanced`): grid_N, dx_um, n_z, z_max_um, bandlimit.

## Which values are recorded only for future physics

- 4F geometry (`warning_only`): SLM2→lens1, lens focal lengths/apertures, lens→Fourier-plane,
  Fourier-stop centre/radius/shape, Fourier-plane→lens2, lens2→output. Changing these does **not**
  change the active field — component-owned physical 4F propagation is not implemented.
- SLM hardware records (`warning_only`): pixel pitch, active area, resolution, fill factor,
  phase-calibration status (both SLMs).
- Camera / reference-plane records (`warning_only`): model, pixel pitch, resolution, magnification,
  plane location, reference-plane definition, calibration status. No camera imaging physics.
- `future_not_implemented`: beam ellipticity/rotation/centre, input aperture, spatial SLM2
  correction-map source, axicon axial offset and mechanical tip/tilt.

## SLM2 correction term

`slm2_correction_phase_rad` is a **uniform piston placeholder** (a scalar broadcast to a uniform
phase offset). It is not an aberration-correction map. A true correction needs a spatial phase map
(`spatial_correction_map_source`: `none` / `Zernike coefficients` / `imported phase array` /
`calibrated correction mask`), which remains `future_not_implemented` and has no field effect.

## Profiles

- Default demo: `configs/hardware/cslm_physical_axicon_demo_profile.json`
  (`profile_status = diagnostic_demo_not_measured_bench`) — mapped config values with provenance;
  unknown hardware left `null` / `unknown`. Not a real lab profile.
- Blank template: `configs/hardware/cslm_physical_axicon_measured_bench_template.json`
  (`profile_status = measured_bench_template_unfilled`) — every real parameter `value=null`,
  `provenance=unknown`. No invented values.

`config_from_profile` loads the mapped config values back into a `CSLMRouteConfig`;
`apply_editable_control_overrides` applies control-id overrides to config fields (governance flags
are locked; there is no path to set `fourier_filter_physics_available=True`).

## Completeness interpretation

`hardware_profile_completeness_report` separates: active CSLM diagnostic branch (complete with
placeholders), ideal axicon benchmark branch (complete), physical 4F route (**blocked** — model
gap), measured lab route (**blocked** — nothing measured), camera comparison (**blocked**). Each
blocked category lists its missing controls. Figure:
`outputs/figures/digital_twin/stage8c3r5_2_hardware_profile_completeness.png` (diagnostic only).

## Claim boundary

Free-space `n=1.0` optical/fluence diagnostic; no material/4F/camera physics;
`final_export_allowed=False`. This stage adds editable metadata + versioned profiles only; it does
not change any modelled field.
