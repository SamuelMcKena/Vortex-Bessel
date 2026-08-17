# Nathan Vector-Hexagon Digital-Twin Plan

**Status:** corrected implementation plan and audit for the Nathan six-sector vector hexagon branch.  
**Scope:** the main study remains inside the pre-existing Digital Twin laser, axicon, relay, objective, pupil, and micro-scale sample/focal conventions. Nathan Marco's original free-space millimetre-scale simulation is a source-convention reference only; it is not the geometry being optimised.

**Current gate:** visual reproduction ladder in `docs/52_nathan_visual_reproduction_ladder.md`. Route comparisons, HWP/SLM robustness sweeps, focus-model conclusions, and publication-style figures remain paused unless the visual ladder reaches outcome 4.

## Correct Physical Question

Using the existing Digital Twin Study laser parameters, axicon/relay/objective/pupil/sample architecture, and micron-scale focal/sample conventions, can Nathan's six-sector radial/azimuthal vector-field mechanism produce a clean hollow hexagonal Bessel-like field at the existing sample/focal reference?

This question is now gated visually before metric-led route comparisons. A non-zero H6 value, high input-field overlap, or a best-z metric maximum is not sufficient evidence of a hollow hexagon.

## Geometry Preservation Contract

Do not globally alter wavelength, pulse duration, laser beam size, axicon parameters, relay geometry, objective/pupil configuration, sample-plane convention, microfabrication configuration, or locked scalar/vector/lab-realism outputs. The additive configuration is built with:

`NathanMicroHexagonConfig.from_existing_digital_twin_baseline(...)`

`NathanHexagonConfig` remains as a compatibility name for that same inherited-geometry configuration.

## Inherited Digital Twin Parameters

| Parameter | Source module/config | Meaning in this branch | Status |
|---|---|---|---|
| `laser.name` | `TwinConfig.laser` | PHAROS-like source identity | inherited fixed |
| `laser.wavelength_m` | `TwinConfig.laser.wavelength_m` | optical wavelength used by vector fields and focusing | inherited fixed |
| `laser.pulse_duration_s` | `TwinConfig.laser.pulse_duration_s` | pulse-duration provenance for microfabrication context | inherited fixed |
| `laser.beam_radius_on_slm_m` | `TwinConfig.laser.beam_radius_on_slm_m` | Gaussian envelope at the vector-generation plane | inherited fixed |
| `slm.resolution_x/y` | `TwinConfig.slm` | panel active aperture for serial-SLM route | inherited fixed |
| `slm.pixel_pitch_m` | `TwinConfig.slm.pixel_pitch_m` | finite pixel pitch and active-area mapping | inherited fixed |
| `slm.phase_bits` | `TwinConfig.slm.phase_bits` | phase quantisation baseline | inherited, optionally swept only as a labelled diagnostic |
| `slm.fill_factor` | `TwinConfig.slm.fill_factor` | panel-realistic fill factor baseline | inherited, optionally swept only as a labelled diagnostic |
| `slm.blaze_period_px` | `TwinConfig.slm.blaze_period_px` | carrier baseline where the existing route uses carriers | inherited fixed |
| `objective.NA` | `TwinConfig.objective.NA` | micro-scale focusing convention | inherited fixed |
| `objective.f_eff_m` | `TwinConfig.objective.f_eff_m` | ObjectiveMap/focal scaling | inherited fixed |
| `objective.pupil_fill` | `TwinConfig.objective.pupil_fill` | pupil-fill convention | inherited fixed |
| `relay.effective_relay_f_m` | `TwinConfig.relay.effective_relay_f_m` | relay demagnification source | inherited fixed |
| `material.refractive_index` | `TwinConfig.material.refractive_index` | existing design medium used for `kr` design | inherited fixed |
| `target.ell` | `TwinConfig.target.ell` | ordinary scalar Bessel/Bessel-Gauss baseline/control | inherited fixed |
| `target.target_core_diameter_m` | `TwinConfig.target.target_core_diameter_m` | existing transverse micro-scale target | inherited fixed |
| `target.target_bessel_length_m` | `TwinConfig.target.target_bessel_length_m` | existing axial micro-scale target | inherited fixed |
| `target.n_axicon` | `TwinConfig.target.n_axicon` | axicon material/index convention | inherited fixed |
| `design.kr_sample_m_inv` | `compute_design_from_targets(...)` | micro-scale radial wavevector at the sample/focal side | inherited derived |
| `design.gamma_slm_deg` | `compute_design_from_targets(...)` | pre-objective axicon angle if no physical override exists | inherited derived |
| `physical_axicon.*` | `TwinConfig.physical_axicon` | physical-axicon overrides and aperture if present | inherited fixed |
| `grid.*` | `TwinConfig.grid` | baseline numerical window and z sampling | inherited; fast developer runs may downsample explicitly |
| `propagation.method` | `TwinConfig.propagation.method` | existing propagation convention | inherited fixed |

## New Nathan-Specific Parameters

| Parameter | Physical meaning | Status |
|---|---|---|
| `vector.n_pairs = 3` | three radial/azimuthal pairs, six sectors total | fixed for Nathan six-sector study |
| `vector.sector_duty` | relative angular width of radial/azimuthal states in each pair | fixed or swept |
| `vector.sector_rotation_rad` | rotation of the sector boundary pattern | swept |
| `PatternedHWPConfig.case` | continuous, six-wedge, or mosaic patterned-retarder route | route selection |
| `PatternedHWPConfig.tiles_per_sector` | angular mosaic resolution inside each 60 degree sector | swept |
| `PatternedHWPConfig.seam_width_rad` | dead/uncertain retarder seam width | assumed or swept |
| `PatternedHWPConfig.central_defect_radius_m` | optional central patterned-HWP defect | assumed or swept |
| `PatternedHWPConfig.fast_axis_error_rad` | local fast-axis error | assumed or swept |
| `PatternedHWPConfig.retardance_error_rad` | HWP retardance error | assumed or swept |
| `serial_slm.case` | ideal or panel-realistic dual-SLM vector-generator branch | route selection |
| `serial_slm.naive_psi2` | deliberately wrong serial-SLM phase-sign control | control only |
| `serial_slm.wrong_carrier_sign` | deliberately wrong carrier-sign control | control only |
| `grid_n`, `z_planes` | computational sampling for this branch | developer fast or resolution-gated confirmation |

## Plane Map In The Existing Digital Twin

| Plane | Physical/numerical meaning | Coordinates and units | Field representation | Scalar/vector status | Module/function | Plane type |
|---|---|---|---|---|---|---|
| P0 | inherited laser Gaussian input | SLM/pre-objective x,y in metres; plots in micrometres | scalar Gaussian envelope metadata | scalar source envelope | `TwinConfig.laser`, `gaussian_envelope` | source plane |
| P1 | vector-generation input plane | centred handoff grid from `default_nathan_grid` | horizontal Jones input for HWP or 45-degree Jones input for serial SLM | vector/Jones | `run_patterned_hwp_route`, `run_vector_arm` | SLM/vector-generator image plane |
| P2 | common output of patterned-HWP route or serial-SLM chain | same handoff grid | `VectorField(Ex,Ey,Ez)` plus Stokes/circular diagnostics | vector/Jones | `canonical_target_field`, route helpers | vector-generator output handoff |
| P3 | downstream shaping/filter handoff only where the existing architecture supports it | handoff grid plus Fourier coordinates for iris diagnostics | circular components before/after one common iris | vector/Jones | `apply_fourier_iris`, `carrier_collinearity_report` | Fourier/filter handoff |
| P4 | inherited physical axicon/objective input convention | pre-objective/axicon grid in metres | vector field after thin axicon phase and Fresnel s/p split | vector component model | `apply_vector_axicon`, `resolve_vector_axicon_parameters` | axicon/pupil/objective input |
| P5 | inherited focused surface/sample reference plane, z=0 | focused x,y in metres; plots in micrometres | `VectorField` and `SurfaceField` handoff for F0; vectorial pupil-spectrum fields for F2 | F0 uses scalar per-component focus; F2 validates with a Nathan-only vectorial pupil-spectrum reference | `focus_vector_to_surface`, `run_vector_axicon_to_surface`, `build_downstream_focus_validation_gate` | sample/focal reference plane |
| P6+ | micro-scale axial region around/beyond P5 | z in metres from P5 | intensity z-stack plus equal-power focus-gate difference stacks | vector ASM reported as intensity/metrics for F0; F2 reports vectorial pupil-spectrum fields at the same z values | `air_z_values`, `propagate_vector_asm`, `hexagon_metrics_for_stack`, `vectorial_pupil_spectrum_reference` | sample-region axial scan |

## Existing Code Reused

| Need | Existing module/API reused |
|---|---|
| Scalar grid, FFT, Gaussian conventions | `vbb_study.equations.fields.make_xy_grid`, `gaussian_amplitude`, `fft2c`, `ifft2c` |
| Scalar angular-spectrum propagation | `vbb_study.equations.propagation.angular_spectrum_propagate_bl`, `make_bl_asm_propagator` |
| Vector field container and vector ASM | `vbb_study.vector_field.VectorField`, `propagate_vector_asm` |
| Jones/Stokes/circular basis | `vbb_study.equations.vector_jones`, `vbb_study.vector_field.VectorField.stokes`, `.circular_components` |
| Spatial retarder algebra | `vbb_study.vbb_polarized_train.retarder_jones` |
| Serial dual-SLM vector arm | `vbb_study.vector_arm_chain`, `vbb_study.vector_arm_config` |
| SLM pixelation, quantisation, fill factor, carriers | `vbb_study.slm_model`, `vbb_study.vector_fourier` |
| Physical axicon, Fresnel s/p split, ObjectiveMap focusing, SurfaceField handoff | `vbb_study.vector_axicon`, `vbb_study.vbb_studies.SurfaceField` |
| Digital-twin field containers | `vbb_study.digital_twin.field_coupling.OpticalFieldStack` |
| Component-owned CSLM/F300 context | `vbb_study.digital_twin.cslm_route`, `nominal_f300_4f` |
| Hexagon/sixfold metrics | `vbb_study.vbb_hexagon_metrics`, `vbb_study.vector_arm_metrics` |
| Figure/output governance | `docs/22_digital_twin_figure_and_output_spec.md`, `vbb_study.publication.figure_registry` |

## New Code Required

- `vbb_study/digital_twin/nathan_vector_hexagon.py`
  - canonical Nathan target field at the existing vector-generator handoff plane;
  - patterned-HWP route variants;
  - serial dual-SLM route adapter;
  - common inherited axicon/relay/objective/propagation wrapper;
  - downstream F0/F1/F2 focus-validation gate with a scoped vectorial pupil-spectrum reference;
  - route-comparison metrics and robustness sweeps.
- `tests/test_nathan_vector_hexagon_digital_twin.py`
  - target construction, HWP target reproduction, mosaic convergence, serial-SLM target reconstruction, shared axicon execution, focus-gate execution, route-comparison safeguards, and control rejection.
- Digital-twin notebooks under `notebooks/digital_twin/`
  - target field;
  - patterned-HWP route;
  - serial dual-SLM route;
  - inherited micro-scale axicon/objective/sample-reference propagation;
  - robustness/equivalence.

## Modelled Effects

- Nathan six-sector RA target: three radial/azimuthal pairs, six alternating 60 degree sectors.
- Gaussian envelope inherited from `TwinConfig.laser.beam_radius_on_slm_m`.
- Transverse Jones fields, Stokes maps, circular-basis components, and winding diagnostics.
- Direct patterned-HWP route with continuous, wedge, tiled/mosaic, seam/dead-gap, central-defect, orientation-error, retardance-error, and aperture variants.
- Serial dual-SLM route through the existing vector-arm implementation, including ideal and panel-realistic configurations.
- SLM phase quantisation, finite active panel, fill factor, carriers, Fourier-order collinearity, and iris bookkeeping where the serial route enables them.
- Thin physical axicon with Fresnel s/p split, existing ObjectiveMap focusing, and vector ASM z-stack in air.
- Focus validation with F0 current scalar per-component focus bridge, F1 scalar-component surrogate, and F2 Nathan-only vectorial pupil-spectrum reference.
- Raw hollow-hexagon diagnostics: sixfold/H6-like content, wall continuity proxy, core darkness, wall power fraction, sidelobe fraction, useful axial length, and route overlap.

## Deliberately Out Of Scope

- In-medium/sample propagation and material response.
- Camera/sensor model, thresholded image processing, or experimental validation claims.
- Global replacement of the locked ObjectiveMap path by a high-NA Richards-Wolf solver. The Nathan branch instead uses a scoped F2 vectorial pupil-spectrum validation reference before drawing route conclusions.
- Thick axicon walk-off, per-ray Fresnel angles, and polarization-dependent propagation inside the axicon.
- SLM flicker, temporal phase noise, liquid-crystal crosstalk, and measured LUT calibration.
- Bench-calibrated F300 4f geometry; available F300/CSLM modules are used as declared context, not as a validated physical stop model.
- Nathan's 2 mm Gaussian, 176-degree axicon, 60 mm observation plane, and 290 mm propagation range as main-study defaults.
- Pass/fail thresholds invented after observing outputs. Where no pre-existing threshold exists, the report labels metrics exploratory.

## Virtual Optical Geometry And Assumptions

- Input field: inherited PHAROS-like Gaussian from `TwinConfig.laser`; if the current baseline says `lambda0 = 1029 nm` and `w0 = 2 mm`, those values are inherited from the Digital Twin baseline rather than copied from Nathan's source script.
- Canonical target plane: existing vector-generator output handoff plane upstream of the inherited downstream optical train.
- Target angle:
  - radial sectors: `alpha(theta) = theta`;
  - azimuthal sectors: `alpha(theta) = theta + pi/2`;
  - default layout: six 60 degree sectors, alternating azimuthal/radial by the same convention as `VectorArmConfig`.
- Patterned-HWP route:
  - input is horizontal linear polarization;
  - local fast axis is `beta(x,y) = alpha(x,y)/2`;
  - continuous HWP is the primary retarder model;
  - segmented/mosaic variants approximate `beta` by angular tiles and explicitly report convergence to the continuous target.
- Serial SLM route:
  - input is linear at 45 degrees;
  - two panels modulate the horizontal/director component;
  - relay inversion, HWP swap, QWP convention, panel effects, carriers, and iris are delegated to `vector_arm_chain`, `slm_model`, and `vector_fourier`.
- Shared axicon route:
  - all input routes can feed `vector_axicon.run_vector_axicon_to_surface` for F0, the current path under test;
  - F1 is a deliberately weaker scalar-component surrogate for rejection/contrast;
  - F2 uses `apply_vector_axicon` followed by the scoped vectorial pupil-spectrum reference;
  - z propagation is in air around the focused surface plane for F0/F1; F2 is evaluated directly at the same z values;
  - `SurfaceField` remains the handoff boundary for later sample-side work.
  - all downstream axicon/objective/sample-reference conventions are inherited from the current `TwinConfig`.

## Resolution Presets

Fast preset used in automated tests and broad sweeps:

- target/route grid: `N = 96` or `128` depending on diagnostic sensitivity; these are developer checks only;
- axicon propagation grid: route grid unchanged;
- z stack: `7` to `11` air-side planes across the repository design's Bessel-like zone;
- angular samples: `720` to `2048` for metrics, `6144` for winding/comb diagnostics where needed.

Paper-resolution confirmation preset:

- target/route grid: `N = 512` for pre-axicon route comparisons;
- selected axicon confirmation: reuse the repository paper/publication-style `TwinConfig`/ObjectiveMap settings before making scientific claims;
- z stack: repository physical-route zone span, at least `41` planes for confirmation, with optional `181`-plane notebook cell left explicit and off by default if runtime is excessive.

The canonical characterisation lock remains unchanged; any Nathan-specific scalar fingerprints are proposed separately, not appended to the lock.
